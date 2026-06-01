//! The query planner: judge how a filter relates to the query, then pick a
//! filtered-search strategy. The rules encode what the benchmark matrix showed:
//! scan exactly when few ids match, search-then-drop only when the filter is broad
//! and not anti-correlated, and use ACORN-style bridging when the filter is strict
//! or points away from the query.

use loupe_distance::Metric;
use loupe_storage::{StorageAccessor, VectorId};
use roaring::RoaringBitmap;

/// How the matching ids relate to the query, as judged online.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrelationClass {
    Positive,
    Neutral,
    AntiCorrelated,
}

/// The strategy the planner picks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannedStrategy {
    Flat,
    PostFilter,
    InTraversal,
    Acorn,
}

/// The planner's decision for one query, with a short reason for EXPLAIN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub strategy: PlannedStrategy,
    pub ef: usize,
    pub max_ef: usize,
    pub adaptive: bool,
    pub reason: &'static str,
}

/// Thresholds the rule ladder uses. Hand-tuned for now; M11 replaces them with a
/// calibrated cost model.
pub struct PlannerConfig {
    pub flat_threshold: usize,
    pub strict_selectivity: f32,
    pub broad_selectivity: f32,
    pub ef: usize,
    pub max_ef: usize,
    pub correlation_band: f32,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            flat_threshold: 2048,
            strict_selectivity: 0.01,
            broad_selectivity: 0.4,
            ef: 64,
            max_ef: 512,
            correlation_band: 0.1,
        }
    }
}

/// What the planner needs to know about a query to choose a plan.
pub struct QueryShape {
    pub total: usize,
    pub matching_count: usize,
    pub k: usize,
    pub correlation: CorrelationClass,
}

/// Picks a filtered-search strategy from the query shape.
pub fn choose_plan(shape: &QueryShape, config: &PlannerConfig) -> Plan {
    let ef = config.ef.max(shape.k);
    let max_ef = config.max_ef.max(ef);

    if shape.matching_count <= config.flat_threshold.max(shape.k) {
        return Plan {
            strategy: PlannedStrategy::Flat,
            ef,
            max_ef,
            adaptive: false,
            reason: "few matches: scan them exactly",
        };
    }

    let selectivity = shape.matching_count as f32 / shape.total.max(1) as f32;

    if shape.correlation == CorrelationClass::AntiCorrelated {
        return Plan {
            strategy: PlannedStrategy::Acorn,
            ef,
            max_ef,
            adaptive: true,
            reason: "anti-correlated filter: bridge through non-matching nodes",
        };
    }

    if selectivity >= config.broad_selectivity {
        return Plan {
            strategy: PlannedStrategy::PostFilter,
            ef,
            max_ef,
            adaptive: true,
            reason: "broad filter: search then drop non-matches",
        };
    }

    if selectivity < config.strict_selectivity {
        return Plan {
            strategy: PlannedStrategy::Acorn,
            ef,
            max_ef,
            adaptive: true,
            reason: "strict filter: bridge through non-matching nodes",
        };
    }

    Plan {
        strategy: PlannedStrategy::InTraversal,
        ef,
        max_ef,
        adaptive: true,
        reason: "selective filter: filter during traversal",
    }
}

/// Turns a raw correlation score into a class using a neutral band around zero.
pub fn classify_correlation(score: f32, band: f32) -> CorrelationClass {
    if score > band {
        CorrelationClass::Positive
    } else if score < -band {
        CorrelationClass::AntiCorrelated
    } else {
        CorrelationClass::Neutral
    }
}

/// Estimates how the matching ids relate to the query by sampling. A positive
/// score means matches sit closer to the query than average; negative means
/// farther. It looks at a small sample, so the cost is a few dozen distances.
pub fn estimate_correlation(
    accessor: &impl StorageAccessor,
    metric: Metric,
    query: &[f32],
    allowed: &RoaringBitmap,
    samples: usize,
    seed: u64,
) -> f32 {
    if accessor.is_empty() || allowed.is_empty() {
        return 0.0;
    }
    let mut rng = Rng::new(seed);
    let allowed_ids: Vec<VectorId> = allowed.iter().collect();

    let matched_take = samples.min(allowed_ids.len());
    let mut matched_sum = 0.0;
    for _ in 0..matched_take {
        let id = allowed_ids[(rng.next_u64() as usize) % allowed_ids.len()];
        matched_sum += metric.distance(query, accessor.vector(id));
    }
    let matched = matched_sum / matched_take as f32;

    let total = accessor.len();
    let global_take = samples.min(total);
    let mut global_sum = 0.0;
    for _ in 0..global_take {
        let id = ((rng.next_u64() as usize) % total) as VectorId;
        global_sum += metric.distance(query, accessor.vector(id));
    }
    let global = global_sum / global_take as f32;

    if global <= 0.0 {
        return 0.0;
    }
    (global - matched) / global
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loupe_storage::InMemoryVectorStore;

    fn shape(matching_count: usize, total: usize, correlation: CorrelationClass) -> QueryShape {
        QueryShape {
            total,
            matching_count,
            k: 10,
            correlation,
        }
    }

    #[test]
    fn flat_for_few_matches() {
        let plan = choose_plan(
            &shape(50, 100_000, CorrelationClass::Neutral),
            &PlannerConfig::default(),
        );
        assert_eq!(plan.strategy, PlannedStrategy::Flat);
        assert!(!plan.adaptive);
    }

    #[test]
    fn post_filter_for_broad_and_not_anti() {
        let plan = choose_plan(
            &shape(60_000, 100_000, CorrelationClass::Neutral),
            &PlannerConfig::default(),
        );
        assert_eq!(plan.strategy, PlannedStrategy::PostFilter);
        assert!(plan.adaptive);
    }

    #[test]
    fn acorn_for_anti_correlated_even_when_broad() {
        let plan = choose_plan(
            &shape(60_000, 100_000, CorrelationClass::AntiCorrelated),
            &PlannerConfig::default(),
        );
        assert_eq!(plan.strategy, PlannedStrategy::Acorn);
        assert!(plan.adaptive);
    }

    #[test]
    fn acorn_for_strict_filter() {
        let plan = choose_plan(
            &shape(3_000, 1_000_000, CorrelationClass::Neutral),
            &PlannerConfig::default(),
        );
        assert_eq!(plan.strategy, PlannedStrategy::Acorn);
        assert!(plan.adaptive);
    }

    #[test]
    fn in_traversal_for_selective_and_not_anti() {
        let plan = choose_plan(
            &shape(10_000, 100_000, CorrelationClass::Neutral),
            &PlannerConfig::default(),
        );
        assert_eq!(plan.strategy, PlannedStrategy::InTraversal);
        assert!(plan.adaptive);
    }

    #[test]
    fn classify_uses_the_band() {
        assert_eq!(classify_correlation(0.5, 0.1), CorrelationClass::Positive);
        assert_eq!(
            classify_correlation(-0.5, 0.1),
            CorrelationClass::AntiCorrelated
        );
        assert_eq!(classify_correlation(0.05, 0.1), CorrelationClass::Neutral);
    }

    fn line_store(count: u32) -> InMemoryVectorStore {
        let mut store = InMemoryVectorStore::new(2);
        for i in 0..count {
            store.add(&[i as f32, i as f32]);
        }
        store
    }

    #[test]
    fn estimate_is_positive_for_nearby_matches() {
        let store = line_store(100);
        let mut allowed = RoaringBitmap::new();
        for id in 48..=52 {
            allowed.insert(id);
        }
        let score = estimate_correlation(&store, Metric::L2, &[50.0, 50.0], &allowed, 16, 7);
        assert!(score > 0.0, "expected positive score, got {score}");
    }

    #[test]
    fn estimate_is_negative_for_far_matches() {
        let store = line_store(100);
        let mut allowed = RoaringBitmap::new();
        for id in 0..5 {
            allowed.insert(id);
        }
        for id in 95..100 {
            allowed.insert(id);
        }
        let score = estimate_correlation(&store, Metric::L2, &[50.0, 50.0], &allowed, 16, 7);
        assert!(score < 0.0, "expected negative score, got {score}");
    }
}
