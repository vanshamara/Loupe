//! Ingestion for an embedded arXiv corpus.
//!
//! Each record carries a precomputed embedding plus real metadata (category,
//! year, document type). The embeddings come from a separate step, so this loader
//! works with whatever embedder produced the file. It turns the records into the
//! vectors to index and a metadata store the filters run against.

use std::fs::File;
use std::io::{self, BufRead, BufReader};

use loupe_filter::{MetadataStore, Value};
use serde::Deserialize;

/// One arXiv paper: its embedding and the metadata filters can use.
#[derive(Clone, Debug, Deserialize)]
pub struct ArxivRecord {
    pub embedding: Vec<f32>,
    pub category: String,
    pub year: i64,
    pub doc_type: String,
}

/// Reads records from a JSONL file, one JSON object per line.
pub fn read_jsonl(path: &str) -> io::Result<Vec<ArxivRecord>> {
    read_jsonl_from(BufReader::new(File::open(path)?))
}

pub fn read_jsonl_from(reader: impl BufRead) -> io::Result<Vec<ArxivRecord>> {
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: ArxivRecord = serde_json::from_str(&line)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        records.push(record);
    }
    Ok(records)
}

/// Splits records into the vectors to index and a metadata store built from their
/// category, year, and document type. Ids are assigned in record order.
pub fn build_workload(records: &[ArxivRecord]) -> (Vec<Vec<f32>>, MetadataStore) {
    let mut vectors = Vec::with_capacity(records.len());
    let mut metadata = MetadataStore::new();
    for (index, record) in records.iter().enumerate() {
        let id = index as u32;
        vectors.push(record.embedding.clone());
        metadata.insert(
            id,
            &[
                ("category", Value::Str(record.category.clone())),
                ("year", Value::Int(record.year)),
                ("doc_type", Value::Str(record.doc_type.clone())),
            ],
        );
    }
    (vectors, metadata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use loupe_filter::Expr;
    use std::io::Cursor;

    const SAMPLE: &str = r#"{"embedding":[1.0,2.0],"category":"cs.LG","year":2021,"doc_type":"primary"}
{"embedding":[3.0,4.0],"category":"cs.CL","year":2023,"doc_type":"primary"}
{"embedding":[5.0,6.0],"category":"cs.LG","year":2023,"doc_type":"cross"}"#;

    #[test]
    fn parses_records() {
        let records = read_jsonl_from(Cursor::new(SAMPLE)).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].embedding, vec![1.0, 2.0]);
        assert_eq!(records[1].category, "cs.CL");
    }

    #[test]
    fn builds_vectors_and_metadata() {
        let records = read_jsonl_from(Cursor::new(SAMPLE)).unwrap();
        let (vectors, metadata) = build_workload(&records);
        assert_eq!(vectors.len(), 3);
        assert_eq!(vectors[2], vec![5.0, 6.0]);

        let category = metadata.evaluate(&Expr::Eq("category".into(), Value::Str("cs.LG".into())));
        assert_eq!(category.iter().collect::<Vec<u32>>(), vec![0, 2]);

        let recent = metadata.evaluate(&Expr::Range("year".into(), 2022, 2025));
        assert_eq!(recent.iter().collect::<Vec<u32>>(), vec![1, 2]);
    }

    #[test]
    fn skips_blank_lines() {
        let jsonl =
            "\n{\"embedding\":[1.0],\"category\":\"a\",\"year\":2020,\"doc_type\":\"x\"}\n\n";
        let records = read_jsonl_from(Cursor::new(jsonl)).unwrap();
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn invalid_json_is_an_error() {
        let result = read_jsonl_from(Cursor::new("not json"));
        assert!(result.is_err());
    }
}
