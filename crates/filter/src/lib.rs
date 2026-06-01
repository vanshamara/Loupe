//! Metadata store and predicate compiler for filtered search.
//!
//! Each field is indexed with roaring bitmaps, so a predicate turns into a set of
//! allowed vector ids by intersecting and unioning those bitmaps.

use std::collections::{BTreeMap, HashMap};

use loupe_storage::VectorId;
use roaring::RoaringBitmap;

/// A metadata value attached to a vector. Strings cover labels like a tenant id or
/// a document type; integers cover things like a year or a timestamp.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Value {
    Int(i64),
    Str(String),
}

/// A filter over metadata. It compiles to the set of vector ids that match.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Eq(String, Value),
    Range(String, i64, i64),
    In(String, Vec<Value>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
}

enum FieldIndex {
    Int(BTreeMap<i64, RoaringBitmap>),
    Str(HashMap<String, RoaringBitmap>),
}

/// Holds each vector's metadata as per-field bitmaps and turns predicates into the
/// set of ids that match them.
#[derive(Default)]
pub struct MetadataStore {
    total: u32,
    fields: HashMap<String, FieldIndex>,
}

impl MetadataStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// How many ids the store has seen, taken as the largest id plus one.
    pub fn total(&self) -> u32 {
        self.total
    }

    /// Records one vector's metadata. A field's type is fixed by its first value;
    /// a later value of a different type for that field is ignored.
    pub fn insert(&mut self, id: VectorId, fields: &[(&str, Value)]) {
        self.total = self.total.max(id.saturating_add(1));
        for (field, value) in fields {
            match value {
                Value::Int(v) => {
                    let index = self
                        .fields
                        .entry((*field).to_string())
                        .or_insert_with(|| FieldIndex::Int(BTreeMap::new()));
                    if let FieldIndex::Int(map) = index {
                        map.entry(*v).or_default().insert(id);
                    }
                }
                Value::Str(v) => {
                    let index = self
                        .fields
                        .entry((*field).to_string())
                        .or_insert_with(|| FieldIndex::Str(HashMap::new()));
                    if let FieldIndex::Str(map) = index {
                        map.entry(v.clone()).or_default().insert(id);
                    }
                }
            }
        }
    }

    /// Compiles a predicate into the set of vector ids that match it.
    pub fn evaluate(&self, expr: &Expr) -> RoaringBitmap {
        match expr {
            Expr::Eq(field, value) => self.matching(field, value),
            Expr::In(field, values) => {
                let mut result = RoaringBitmap::new();
                for value in values {
                    result |= self.matching(field, value);
                }
                result
            }
            Expr::Range(field, min, max) => self.matching_range(field, *min, *max),
            Expr::And(left, right) => self.evaluate(left) & self.evaluate(right),
            Expr::Or(left, right) => self.evaluate(left) | self.evaluate(right),
        }
    }

    /// The fraction of all vectors that match the predicate, from 0.0 to 1.0.
    pub fn selectivity(&self, expr: &Expr) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        self.evaluate(expr).len() as f32 / self.total as f32
    }

    fn matching(&self, field: &str, value: &Value) -> RoaringBitmap {
        match (self.fields.get(field), value) {
            (Some(FieldIndex::Int(map)), Value::Int(v)) => map.get(v).cloned().unwrap_or_default(),
            (Some(FieldIndex::Str(map)), Value::Str(v)) => map.get(v).cloned().unwrap_or_default(),
            _ => RoaringBitmap::new(),
        }
    }

    fn matching_range(&self, field: &str, min: i64, max: i64) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        if let Some(FieldIndex::Int(map)) = self.fields.get(field) {
            for (_, bitmap) in map.range(min..=max) {
                result |= bitmap;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> MetadataStore {
        let mut store = MetadataStore::new();
        store.insert(
            0,
            &[
                ("tenant", Value::Str("a".into())),
                ("year", Value::Int(2020)),
            ],
        );
        store.insert(
            1,
            &[
                ("tenant", Value::Str("a".into())),
                ("year", Value::Int(2023)),
            ],
        );
        store.insert(
            2,
            &[
                ("tenant", Value::Str("b".into())),
                ("year", Value::Int(2024)),
            ],
        );
        store.insert(
            3,
            &[
                ("tenant", Value::Str("b".into())),
                ("year", Value::Int(2024)),
            ],
        );
        store
    }

    fn ids(bitmap: &RoaringBitmap) -> Vec<u32> {
        bitmap.iter().collect()
    }

    #[test]
    fn total_tracks_largest_id() {
        let store = sample();
        assert_eq!(store.total(), 4);
    }

    #[test]
    fn eq_on_string_field() {
        let store = sample();
        let matches = store.evaluate(&Expr::Eq("tenant".into(), Value::Str("a".into())));
        assert_eq!(ids(&matches), vec![0, 1]);
    }

    #[test]
    fn eq_on_int_field() {
        let store = sample();
        let matches = store.evaluate(&Expr::Eq("year".into(), Value::Int(2024)));
        assert_eq!(ids(&matches), vec![2, 3]);
    }

    #[test]
    fn range_on_int_field_is_inclusive() {
        let store = sample();
        let matches = store.evaluate(&Expr::Range("year".into(), 2023, 2024));
        assert_eq!(ids(&matches), vec![1, 2, 3]);
    }

    #[test]
    fn in_unions_values() {
        let store = sample();
        let matches = store.evaluate(&Expr::In(
            "tenant".into(),
            vec![Value::Str("a".into()), Value::Str("b".into())],
        ));
        assert_eq!(ids(&matches), vec![0, 1, 2, 3]);
    }

    #[test]
    fn and_intersects() {
        let store = sample();
        let expr = Expr::And(
            Box::new(Expr::Eq("tenant".into(), Value::Str("b".into()))),
            Box::new(Expr::Eq("year".into(), Value::Int(2024))),
        );
        assert_eq!(ids(&store.evaluate(&expr)), vec![2, 3]);
    }

    #[test]
    fn or_unions() {
        let store = sample();
        let expr = Expr::Or(
            Box::new(Expr::Eq("tenant".into(), Value::Str("a".into()))),
            Box::new(Expr::Eq("year".into(), Value::Int(2024))),
        );
        assert_eq!(ids(&store.evaluate(&expr)), vec![0, 1, 2, 3]);
    }

    #[test]
    fn unknown_field_matches_nothing() {
        let store = sample();
        let matches = store.evaluate(&Expr::Eq("missing".into(), Value::Int(1)));
        assert!(matches.is_empty());
    }

    #[test]
    fn selectivity_is_matching_fraction() {
        let store = sample();
        let half = store.selectivity(&Expr::Eq("tenant".into(), Value::Str("a".into())));
        assert_eq!(half, 0.5);
    }
}
