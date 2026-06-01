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
    pub ef_expansions: u32,
    pub tombstoned_visits: u64,
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

/// Version tag on every exported trace graph, so the format can change later
/// without breaking older readers.
pub const TRACE_EXPORT_VERSION: u32 = 1;

/// Whether a node ended up in the returned top-k or was only walked through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    Visited,
    Returned,
}

/// A node in the trace graph: just an id and its status, never the vector itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceNode {
    pub id: VectorId,
    pub status: NodeStatus,
}

/// One graph edge the search walked, from a node to one of its neighbors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceEdge {
    pub from: VectorId,
    pub to: VectorId,
}

/// Hard caps on how big an exported trace graph can get, so it stays cheap and
/// cannot blow up on a large query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceGraphLimits {
    pub max_nodes: usize,
    pub max_edges: usize,
}

impl Default for TraceGraphLimits {
    fn default() -> Self {
        Self {
            max_nodes: 2000,
            max_edges: 16000,
        }
    }
}

/// The local part of the graph that one search touched. This is a debug and
/// EXPLAIN artifact, built only on request, never on a normal query.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceGraphExport {
    pub version: u32,
    pub nodes: Vec<TraceNode>,
    pub edges: Vec<TraceEdge>,
    pub truncated: bool,
}

/// A short description of the filtered query the planner reasoned about.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuerySummary {
    pub k: usize,
    pub matching_count: u64,
    pub selectivity: f32,
    pub correlation: String,
}

/// How one strategy did on a query: its cost, how many results it returned, and
/// how close it came to the exact filtered answer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanReport {
    pub strategy: String,
    pub chosen: bool,
    pub distance_computations: u64,
    pub returned: usize,
    pub recall_proxy: f32,
}

/// The EXPLAIN SEARCH result: the chosen plan, what the rejected plans would have
/// done (the counterfactual), the trace, and any risks or advice. This shape is
/// frozen here so the CLI, gRPC, and the graph explorer all read the same thing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplainSearch {
    pub query_summary: QuerySummary,
    pub chosen_plan: PlanReport,
    pub counterfactual_plans: Vec<PlanReport>,
    pub trace_metrics: SearchTrace,
    pub risk_flags: Vec<String>,
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_trace_json_is_stable() {
        let metrics = SearchTrace {
            distance_computations: 660,
            nodes_visited: 712,
            ef_expansions: 2,
            tombstoned_visits: 5,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        assert_eq!(
            json,
            r#"{"distance_computations":660,"nodes_visited":712,"ef_expansions":2,"tombstoned_visits":5}"#
        );
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
                ef_expansions: 0,
                tombstoned_visits: 0,
            },
            filter: None,
            plan: None,
            recall_smoke: None,
        };
        let json = serde_json::to_string(&record).unwrap();
        let expected = r#"{"query_id":1,"k":10,"latency_micros":1234,"returned":[5,9],"metrics":{"distance_computations":660,"nodes_visited":712,"ef_expansions":0,"tombstoned_visits":0},"filter":null,"plan":null,"recall_smoke":null}"#;
        assert_eq!(json, expected);
    }

    #[test]
    fn trace_graph_export_json_is_stable() {
        let export = TraceGraphExport {
            version: 1,
            nodes: vec![
                TraceNode {
                    id: 5,
                    status: NodeStatus::Returned,
                },
                TraceNode {
                    id: 9,
                    status: NodeStatus::Visited,
                },
            ],
            edges: vec![TraceEdge { from: 5, to: 9 }],
            truncated: false,
        };
        let json = serde_json::to_string(&export).unwrap();
        let expected = r#"{"version":1,"nodes":[{"id":5,"status":"Returned"},{"id":9,"status":"Visited"}],"edges":[{"from":5,"to":9}],"truncated":false}"#;
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

    fn sample_explain() -> ExplainSearch {
        ExplainSearch {
            query_summary: QuerySummary {
                k: 10,
                matching_count: 200,
                selectivity: 0.5,
                correlation: "anti_correlated".to_string(),
            },
            chosen_plan: PlanReport {
                strategy: "InTraversal".to_string(),
                chosen: true,
                distance_computations: 1500,
                returned: 10,
                recall_proxy: 1.0,
            },
            counterfactual_plans: vec![
                PlanReport {
                    strategy: "PostFilter".to_string(),
                    chosen: false,
                    distance_computations: 600,
                    returned: 5,
                    recall_proxy: 0.5,
                },
                PlanReport {
                    strategy: "Flat".to_string(),
                    chosen: false,
                    distance_computations: 200,
                    returned: 10,
                    recall_proxy: 1.0,
                },
            ],
            trace_metrics: SearchTrace {
                distance_computations: 1500,
                nodes_visited: 1800,
                ef_expansions: 0,
                tombstoned_visits: 0,
            },
            risk_flags: vec!["anti_correlated_filter".to_string()],
            recommendations: vec![],
        }
    }

    #[test]
    fn explain_search_json_is_stable() {
        let json = serde_json::to_string(&sample_explain()).unwrap();
        let expected = r#"{"query_summary":{"k":10,"matching_count":200,"selectivity":0.5,"correlation":"anti_correlated"},"chosen_plan":{"strategy":"InTraversal","chosen":true,"distance_computations":1500,"returned":10,"recall_proxy":1.0},"counterfactual_plans":[{"strategy":"PostFilter","chosen":false,"distance_computations":600,"returned":5,"recall_proxy":0.5},{"strategy":"Flat","chosen":false,"distance_computations":200,"returned":10,"recall_proxy":1.0}],"trace_metrics":{"distance_computations":1500,"nodes_visited":1800,"ef_expansions":0,"tombstoned_visits":0},"risk_flags":["anti_correlated_filter"],"recommendations":[]}"#;
        assert_eq!(json, expected);
    }

    #[test]
    fn explain_search_round_trips() {
        let explain = sample_explain();
        let json = serde_json::to_string(&explain).unwrap();
        let back: ExplainSearch = serde_json::from_str(&json).unwrap();
        assert_eq!(explain, back);
    }
}
