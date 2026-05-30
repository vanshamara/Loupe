//! Stable observability contracts: the per-query metrics a search records and the
//! query-log record that wraps them. Other crates serialize these, so their shape
//! is frozen here and checked by golden tests.

use loupe_storage::VectorId;
use serde::{Deserialize, Serialize};

/// Counts what a single search did. This is the cheap, always-on trace, filled in
/// as the search runs and reported by EXPLAIN and the benchmarks.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchTrace {
    pub distance_computations: u64,
    pub nodes_visited: u64,
}

/// One line of the query log, written per query. The first fields are filled in
/// from the start. `filter`, `plan`, and `recall_smoke` stay empty until the later
/// milestones that add filtering, planning, and recall checks populate them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QueryLogRecord {
    pub query_id: u64,
    pub k: usize,
    pub latency_micros: u64,
    pub returned: Vec<VectorId>,
    pub metrics: SearchTrace,
    pub filter: Option<String>,
    pub plan: Option<String>,
    pub recall_smoke: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_trace_json_is_stable() {
        let metrics = SearchTrace {
            distance_computations: 660,
            nodes_visited: 712,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        assert_eq!(json, r#"{"distance_computations":660,"nodes_visited":712}"#);
    }

    #[test]
    fn query_log_record_json_is_stable() {
        let record = QueryLogRecord {
            query_id: 1,
            k: 10,
            latency_micros: 1234,
            returned: vec![5, 9],
            metrics: SearchTrace {
                distance_computations: 660,
                nodes_visited: 712,
            },
            filter: None,
            plan: None,
            recall_smoke: None,
        };
        let json = serde_json::to_string(&record).unwrap();
        let expected = r#"{"query_id":1,"k":10,"latency_micros":1234,"returned":[5,9],"metrics":{"distance_computations":660,"nodes_visited":712},"filter":null,"plan":null,"recall_smoke":null}"#;
        assert_eq!(json, expected);
    }

    #[test]
    fn query_log_record_round_trips() {
        let record = QueryLogRecord {
            query_id: 7,
            k: 5,
            latency_micros: 42,
            returned: vec![1, 2, 3],
            metrics: SearchTrace::default(),
            filter: Some("doc_type = contract".to_string()),
            plan: Some("HnswPostFilter".to_string()),
            recall_smoke: Some(0.97),
        };
        let json = serde_json::to_string(&record).unwrap();
        let back: QueryLogRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(record, back);
    }
}
