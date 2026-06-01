//! Query-log advisor: reads what queries actually did and suggests what to change.
//!
//! It groups logged queries by their filter and flags patterns: filters that
//! return too few results, filters that keep widening the beam, filters the
//! planner had to route away from post-filter, and the most common filter. Every
//! suggestion is advice; the advisor never changes anything.

use std::collections::HashMap;

use loupe_trace::QueryLogRecord;

/// One piece of advice about a filter pattern seen in the logs.
#[derive(Clone, Debug, PartialEq)]
pub struct Recommendation {
    pub category: &'static str,
    pub filter: String,
    pub message: String,
}

const NO_FILTER: &str = "(no filter)";

/// Turns a batch of query logs into recommendations.
pub fn advise(logs: &[QueryLogRecord]) -> Vec<Recommendation> {
    let mut groups: HashMap<String, Vec<&QueryLogRecord>> = HashMap::new();
    for log in logs {
        let key = log.filter.clone().unwrap_or_else(|| NO_FILTER.to_string());
        groups.entry(key).or_default().push(log);
    }

    let mut recommendations = Vec::new();
    for (filter, group) in &groups {
        recommendations.extend(advise_filter(filter, group));
    }

    if let Some((filter, group)) = groups.iter().max_by_key(|(_, group)| group.len())
        && group.len() > 1
    {
        recommendations.push(Recommendation {
            category: "hot_filter",
            filter: filter.clone(),
            message: format!(
                "most common filter ({} queries); a dedicated layout could help (advisory only)",
                group.len()
            ),
        });
    }

    recommendations.sort_by(|a, b| a.filter.cmp(&b.filter).then(a.category.cmp(b.category)));
    recommendations
}

fn advise_filter(filter: &str, group: &[&QueryLogRecord]) -> Vec<Recommendation> {
    let total = group.len();
    let mut out = Vec::new();

    let under_returning = group
        .iter()
        .filter(|log| log.returned.len() < log.k)
        .count();
    if under_returning * 2 > total {
        out.push(Recommendation {
            category: "under_returning",
            filter: filter.to_string(),
            message: format!(
                "returns fewer than k results on {under_returning}/{total} queries; the matching set is small"
            ),
        });
    }

    let widened = group
        .iter()
        .filter(|log| log.metrics.ef_expansions > 0)
        .count();
    if widened * 10 > total * 3 {
        out.push(Recommendation {
            category: "widen_beam",
            filter: filter.to_string(),
            message: format!(
                "widens the beam on {widened}/{total} queries; raise the default ef for this workload"
            ),
        });
    }

    let routed = group
        .iter()
        .filter(|log| matches!(log.plan.as_deref(), Some("InTraversal") | Some("Acorn")))
        .count();
    if routed * 2 > total {
        out.push(Recommendation {
            category: "anti_correlated",
            filter: filter.to_string(),
            message: format!(
                "routed away from post-filter on {routed}/{total} queries; this filter is anti-correlated or strict"
            ),
        });
    }

    let recalls: Vec<f32> = group.iter().filter_map(|log| log.recall_smoke).collect();
    if !recalls.is_empty() {
        let average = recalls.iter().sum::<f32>() / recalls.len() as f32;
        if average < 0.9 {
            out.push(Recommendation {
                category: "low_recall",
                filter: filter.to_string(),
                message: format!("low recall ({average:.2}); use a wider beam or an exact scan"),
            });
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use loupe_trace::SearchTrace;

    fn record(
        filter: &str,
        plan: &str,
        returned: usize,
        k: usize,
        ef_expansions: u32,
    ) -> QueryLogRecord {
        QueryLogRecord {
            query_id: 0,
            k,
            latency_micros: 0,
            returned: (0..returned as u32).collect(),
            metrics: SearchTrace {
                distance_computations: 0,
                nodes_visited: 0,
                ef_expansions,
            },
            filter: Some(filter.to_string()),
            plan: Some(plan.to_string()),
            recall_smoke: None,
        }
    }

    fn has(recs: &[Recommendation], category: &str) -> bool {
        recs.iter().any(|r| r.category == category)
    }

    #[test]
    fn flags_under_returning_filter() {
        let logs = vec![
            record("cat=rare", "Flat", 2, 10, 0),
            record("cat=rare", "Flat", 1, 10, 0),
            record("cat=rare", "Flat", 3, 10, 0),
        ];
        let recs = advise(&logs);
        assert!(has(&recs, "under_returning"));
    }

    #[test]
    fn flags_anti_correlated_routing() {
        let logs = vec![
            record("cat=math", "InTraversal", 10, 10, 0),
            record("cat=math", "InTraversal", 10, 10, 0),
            record("cat=math", "Acorn", 10, 10, 0),
        ];
        let recs = advise(&logs);
        assert!(has(&recs, "anti_correlated"));
    }

    #[test]
    fn flags_beam_widening() {
        let logs = vec![
            record("cat=x", "InTraversal", 10, 10, 2),
            record("cat=x", "InTraversal", 10, 10, 1),
            record("cat=x", "Flat", 10, 10, 0),
        ];
        let recs = advise(&logs);
        assert!(has(&recs, "widen_beam"));
    }

    #[test]
    fn flags_hot_filter() {
        let logs = vec![
            record("cat=a", "Flat", 10, 10, 0),
            record("cat=a", "Flat", 10, 10, 0),
            record("cat=b", "Flat", 10, 10, 0),
        ];
        let recs = advise(&logs);
        assert!(
            recs.iter()
                .any(|r| r.category == "hot_filter" && r.filter == "cat=a")
        );
    }

    #[test]
    fn empty_logs_give_no_advice() {
        assert!(advise(&[]).is_empty());
    }
}
