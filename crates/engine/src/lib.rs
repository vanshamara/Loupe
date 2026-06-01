//! The collection: an HNSW index plus a metadata store, with a planner that picks
//! a filtered-search strategy per query and an EXPLAIN that shows its reasoning.

use std::collections::HashSet;
use std::io;

use loupe_distance::Metric;
use loupe_filter::MetadataStore;
use loupe_hnsw::HnswIndex;
use loupe_planner::{
    CorrelationClass, PlannerConfig, QueryShape, choose_plan, classify_correlation,
    estimate_correlation,
};
use loupe_storage::VectorId;
use loupe_trace::SearchTrace;
use loupe_wal::{SyncPolicy, WalRecord, WriteAheadLog};
use roaring::RoaringBitmap;

pub use loupe_filter::{Expr, Value};
pub use loupe_hnsw::Neighbor;
pub use loupe_planner::{Plan, PlannedStrategy};
pub use loupe_trace::{ExplainSearch, PlanReport, QuerySummary};

const STRATEGIES: [PlannedStrategy; 4] = [
    PlannedStrategy::Flat,
    PlannedStrategy::PostFilter,
    PlannedStrategy::InTraversal,
    PlannedStrategy::Acorn,
];

/// A set of vectors with metadata, searched through the planner. With a write-ahead
/// log it survives a crash; without one it lives only in memory.
pub struct Collection {
    index: HnswIndex,
    metadata: MetadataStore,
    planner: PlannerConfig,
    correlation_samples: usize,
    seed: u64,
    wal: Option<WriteAheadLog>,
}

impl Collection {
    /// Creates an empty in-memory collection with no durability.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            index: HnswIndex::new(dim, metric, loupe_hnsw::HnswConfig::default()),
            metadata: MetadataStore::new(),
            planner: PlannerConfig::default(),
            correlation_samples: 32,
            seed: 1,
            wal: None,
        }
    }

    /// Opens a durable collection in `dir`, rebuilding it by replaying the
    /// write-ahead log. A fresh directory yields an empty collection. Records torn
    /// off by a crash are skipped, so no partial write corrupts the index.
    pub fn open(dir: &str, dim: usize, metric: Metric) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let wal_path = format!("{dir}/wal.log");
        let records = WriteAheadLog::replay(&wal_path)?;

        let mut collection = Self::new(dim, metric);
        for record in records {
            if let WalRecord::Insert { vector, fields } = record {
                let borrowed: Vec<(&str, Value)> = fields
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.clone()))
                    .collect();
                collection.apply_insert(&vector, &borrowed);
            }
        }
        collection.wal = Some(WriteAheadLog::open(&wal_path, SyncPolicy::PerWrite)?);
        Ok(collection)
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Replaces the planner thresholds (used in tests and tuning).
    pub fn set_planner(&mut self, planner: PlannerConfig) {
        self.planner = planner;
    }

    /// Adds a vector with its metadata and returns its id. With durability on, the
    /// change is written to the log (and flushed) before it is applied.
    pub fn insert(&mut self, vector: &[f32], fields: &[(&str, Value)]) -> VectorId {
        if let Some(wal) = &mut self.wal {
            let record = WalRecord::Insert {
                vector: vector.to_vec(),
                fields: fields
                    .iter()
                    .map(|(key, value)| ((*key).to_string(), value.clone()))
                    .collect(),
            };
            wal.append(&record)
                .expect("failed to append to the write-ahead log");
        }
        self.apply_insert(vector, fields)
    }

    fn apply_insert(&mut self, vector: &[f32], fields: &[(&str, Value)]) -> VectorId {
        let id = self.index.insert(vector);
        self.metadata.insert(id, fields);
        id
    }

    /// Runs a filtered search with the strategy the planner picks.
    pub fn search(&self, query: &[f32], filter: &Expr, k: usize) -> (Vec<Neighbor>, Plan) {
        let allowed = self.metadata.evaluate(filter);
        let plan = self.plan_for(query, &allowed, k);
        let (results, _) = self.run(plan.strategy, query, &allowed, k, plan.ef);
        (results, plan)
    }

    /// Runs the chosen strategy and the rejected ones, returning the chosen
    /// results plus an EXPLAIN: why this plan, and what the others would have cost.
    pub fn explain_search(
        &self,
        query: &[f32],
        filter: &Expr,
        k: usize,
    ) -> (Vec<Neighbor>, ExplainSearch) {
        let allowed = self.metadata.evaluate(filter);
        let matching_count = allowed.len();
        let total = self.metadata.total().max(1);
        let selectivity = matching_count as f32 / total as f32;
        let score = estimate_correlation(
            self.index.accessor(),
            self.index.metric(),
            query,
            &allowed,
            self.correlation_samples,
            self.seed,
        );
        let correlation = classify_correlation(score, self.planner.correlation_band);
        let shape = QueryShape {
            total: total as usize,
            matching_count: matching_count as usize,
            k,
            correlation,
        };
        let plan = choose_plan(&shape, &self.planner);

        let (oracle, flat_trace) = self.run(PlannedStrategy::Flat, query, &allowed, k, plan.ef);
        let oracle_ids: HashSet<VectorId> = oracle.iter().map(|n| n.id).collect();

        let mut reports = Vec::with_capacity(STRATEGIES.len());
        let mut chosen_results = Vec::new();
        let mut chosen_trace = SearchTrace::default();
        for strategy in STRATEGIES {
            let (results, trace) = if strategy == PlannedStrategy::Flat {
                (oracle.clone(), flat_trace.clone())
            } else {
                self.run(strategy, query, &allowed, k, plan.ef)
            };
            let chosen = strategy == plan.strategy;
            if chosen {
                chosen_results = results.clone();
                chosen_trace = trace.clone();
            }
            reports.push(PlanReport {
                strategy: strategy_name(strategy).to_string(),
                chosen,
                distance_computations: trace.distance_computations,
                returned: results.len(),
                recall_proxy: overlap(&results, &oracle_ids),
            });
        }

        let chosen_slot = reports.iter().position(|r| r.chosen).unwrap_or(0);
        let chosen_plan = reports[chosen_slot].clone();
        let counterfactual_plans = reports
            .iter()
            .enumerate()
            .filter(|(slot, _)| *slot != chosen_slot)
            .map(|(_, report)| report.clone())
            .collect();

        let explain = ExplainSearch {
            query_summary: QuerySummary {
                k,
                matching_count,
                selectivity,
                correlation: correlation_name(correlation).to_string(),
            },
            chosen_plan,
            counterfactual_plans,
            trace_metrics: chosen_trace,
            risk_flags: risk_flags(correlation, &reports, k),
            recommendations: recommendations(&reports, chosen_slot),
        };
        (chosen_results, explain)
    }

    fn plan_for(&self, query: &[f32], allowed: &RoaringBitmap, k: usize) -> Plan {
        let total = self.metadata.total().max(1);
        let score = estimate_correlation(
            self.index.accessor(),
            self.index.metric(),
            query,
            allowed,
            self.correlation_samples,
            self.seed,
        );
        let shape = QueryShape {
            total: total as usize,
            matching_count: allowed.len() as usize,
            k,
            correlation: classify_correlation(score, self.planner.correlation_band),
        };
        choose_plan(&shape, &self.planner)
    }

    fn run(
        &self,
        strategy: PlannedStrategy,
        query: &[f32],
        allowed: &RoaringBitmap,
        k: usize,
        ef: usize,
    ) -> (Vec<Neighbor>, SearchTrace) {
        match strategy {
            PlannedStrategy::Flat => self.index.flat_filtered_search(query, k, allowed),
            PlannedStrategy::PostFilter => self.index.post_filter_search(query, k, allowed, ef),
            PlannedStrategy::InTraversal => self
                .index
                .in_traversal_filtered_search(query, k, allowed, ef),
            PlannedStrategy::Acorn => self.index.acorn_filtered_search(query, k, allowed, ef),
        }
    }
}

fn overlap(results: &[Neighbor], oracle_ids: &HashSet<VectorId>) -> f32 {
    if oracle_ids.is_empty() {
        return 1.0;
    }
    let hits = results
        .iter()
        .filter(|n| oracle_ids.contains(&n.id))
        .count();
    hits as f32 / oracle_ids.len() as f32
}

fn risk_flags(correlation: CorrelationClass, reports: &[PlanReport], k: usize) -> Vec<String> {
    let mut flags = Vec::new();
    if correlation == CorrelationClass::AntiCorrelated {
        flags.push("anti_correlated_filter".to_string());
    }
    if let Some(post) = reports.iter().find(|r| r.strategy == "PostFilter")
        && post.returned < k
    {
        flags.push("post_filter_under_returns".to_string());
    }
    flags
}

fn recommendations(reports: &[PlanReport], chosen_slot: usize) -> Vec<String> {
    let mut advice = Vec::new();
    if reports[chosen_slot].recall_proxy < 1.0 {
        advice.push("raise ef or use an exact scan for full recall".to_string());
    }
    advice
}

fn strategy_name(strategy: PlannedStrategy) -> &'static str {
    match strategy {
        PlannedStrategy::Flat => "Flat",
        PlannedStrategy::PostFilter => "PostFilter",
        PlannedStrategy::InTraversal => "InTraversal",
        PlannedStrategy::Acorn => "Acorn",
    }
}

fn correlation_name(correlation: CorrelationClass) -> &'static str {
    match correlation {
        CorrelationClass::Positive => "positive",
        CorrelationClass::Neutral => "neutral",
        CorrelationClass::AntiCorrelated => "anti_correlated",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anti_collection() -> Collection {
        let mut collection = Collection::new(2, Metric::L2);
        collection.set_planner(PlannerConfig {
            flat_threshold: 5,
            broad_selectivity: 0.4,
            ef: 64,
            correlation_band: 0.1,
        });
        for i in 0..200u32 {
            let category = if i < 100 { "low" } else { "high" };
            collection.insert(
                &[i as f32, i as f32],
                &[("cat", Value::Str(category.into()))],
            );
        }
        collection
    }

    #[test]
    fn explain_picks_in_traversal_and_flags_anti_correlation() {
        let collection = anti_collection();
        let query = [185.0, 185.0];
        let filter = Expr::Eq("cat".into(), Value::Str("low".into()));
        let (results, explain) = collection.explain_search(&query, &filter, 10);

        assert_eq!(explain.chosen_plan.strategy, "InTraversal");
        assert!(
            explain
                .risk_flags
                .contains(&"anti_correlated_filter".to_string())
        );
        assert_eq!(explain.counterfactual_plans.len(), 3);
        for neighbor in &results {
            assert!(neighbor.id < 100);
        }
        let post = explain
            .counterfactual_plans
            .iter()
            .find(|p| p.strategy == "PostFilter")
            .unwrap();
        assert!(post.returned < 10);
    }

    #[test]
    fn explain_picks_post_filter_for_broad_aligned_filter() {
        let collection = anti_collection();
        let query = [150.0, 150.0];
        let filter = Expr::Eq("cat".into(), Value::Str("high".into()));
        let (_, explain) = collection.explain_search(&query, &filter, 10);
        assert_eq!(explain.chosen_plan.strategy, "PostFilter");
    }

    #[test]
    fn search_returns_only_matching_ids() {
        let collection = anti_collection();
        let filter = Expr::Eq("cat".into(), Value::Str("low".into()));
        let (results, plan) = collection.search(&[185.0, 185.0], &filter, 5);
        assert_eq!(plan.strategy, PlannedStrategy::InTraversal);
        for neighbor in &results {
            assert!(neighbor.id < 100);
        }
    }

    #[test]
    fn recovers_inserts_and_serves_filtered_queries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        {
            let mut collection = Collection::open(path, 2, Metric::L2).unwrap();
            collection.insert(&[1.0, 1.0], &[("cat", Value::Str("a".into()))]);
            collection.insert(&[2.0, 2.0], &[("cat", Value::Str("b".into()))]);
            collection.insert(&[3.0, 3.0], &[("cat", Value::Str("a".into()))]);
        }

        let recovered = Collection::open(path, 2, Metric::L2).unwrap();
        assert_eq!(recovered.len(), 3);
        let filter = Expr::Eq("cat".into(), Value::Str("a".into()));
        let (results, _) = recovered.search(&[1.0, 1.0], &filter, 5);
        let ids: Vec<VectorId> = results.iter().map(|n| n.id).collect();
        assert_eq!(ids, vec![0, 2]);
    }

    #[test]
    fn recovery_ignores_a_torn_tail() {
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        {
            let mut collection = Collection::open(path, 2, Metric::L2).unwrap();
            collection.insert(&[1.0, 1.0], &[("cat", Value::Str("a".into()))]);
            collection.insert(&[2.0, 2.0], &[("cat", Value::Str("a".into()))]);
        }

        let wal_path = format!("{path}/wal.log");
        let mut raw = std::fs::OpenOptions::new()
            .append(true)
            .open(&wal_path)
            .unwrap();
        raw.write_all(&999u32.to_le_bytes()).unwrap();
        raw.write_all(&0u32.to_le_bytes()).unwrap();
        raw.write_all(&[7u8; 5]).unwrap();
        drop(raw);

        let recovered = Collection::open(path, 2, Metric::L2).unwrap();
        assert_eq!(recovered.len(), 2);
    }
}
