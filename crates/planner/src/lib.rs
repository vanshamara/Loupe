//! The query planner estimates the cost of each filtered-search strategy and
//! picks the lowest-cost plan for the current query shape.

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

/// Tunable inputs for the cost model.
pub struct PlannerConfig {
    pub flat_threshold: usize,
    pub strict_selectivity: f32,
    pub broad_selectivity: f32,
    pub ef: usize,
    pub max_ef: usize,
    pub correlation_band: f32,
    pub recall_miss_penalty: f32,
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
            recall_miss_penalty: 10_000.0,
        }
    }
}

/// What the planner needs to know about a query to choose a plan.
pub struct QueryShape {
    pub total: usize,
    pub matching_count: usize,
    pub k: usize,
    pub candidate_count: usize,
    pub correlation: CorrelationClass,
    pub historical_miss_rate: f32,
}

impl QueryShape {
    /// Creates a query shape with no historical miss signal.
    pub fn new(
        total: usize,
        matching_count: usize,
        k: usize,
        correlation: CorrelationClass,
    ) -> Self {
        Self {
            total,
            matching_count,
            k,
            candidate_count: k,
            correlation,
            historical_miss_rate: 0.0,
        }
    }
}

/// The cost model's estimate for one candidate strategy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrategyEstimate {
    pub strategy: PlannedStrategy,
    pub expected_work: f32,
    pub recall_miss_risk: f32,
    pub total_cost: f32,
}

/// Estimates strategy costs from selectivity, candidates, correlation, and miss history.
pub fn estimate_strategy_costs(
    shape: &QueryShape,
    config: &PlannerConfig,
) -> Vec<StrategyEstimate> {
    let ef = config.ef.max(shape.k);
    let candidate_count = shape.candidate_count.max(shape.k).max(ef) as f32;
    let selectivity = selectivity(shape);
    let history = shape.historical_miss_rate.clamp(0.0, 1.0);

    let flat_work = flat_work(shape, config);
    let flat_risk = if shape.matching_count >= shape.k {
        0.0
    } else {
        1.0
    };

    let post_work = post_filter_work(shape, selectivity, candidate_count);
    let post_risk = post_filter_risk(selectivity, shape.correlation, history, config);

    let in_traversal_work = in_traversal_work(shape, selectivity, candidate_count);
    let in_traversal_risk = in_traversal_risk(selectivity, shape.correlation, history, config);

    let acorn_work = acorn_work(shape, selectivity, candidate_count, config);
    let acorn_risk = acorn_risk(selectivity, shape.correlation, history, config);

    vec![
        estimate(PlannedStrategy::Flat, flat_work, flat_risk, config),
        estimate(PlannedStrategy::PostFilter, post_work, post_risk, config),
        estimate(
            PlannedStrategy::InTraversal,
            in_traversal_work,
            in_traversal_risk,
            config,
        ),
        estimate(PlannedStrategy::Acorn, acorn_work, acorn_risk, config),
    ]
}

/// Picks the lowest-cost filtered-search strategy for the query shape.
pub fn choose_plan(shape: &QueryShape, config: &PlannerConfig) -> Plan {
    let ef = config.ef.max(shape.k);
    let max_ef = config.max_ef.max(ef);
    let estimates = estimate_strategy_costs(shape, config);
    let winner = estimates
        .iter()
        .min_by(|left, right| left.total_cost.total_cmp(&right.total_cost))
        .expect("planner always estimates every strategy");

    Plan {
        strategy: winner.strategy,
        ef,
        max_ef,
        adaptive: winner.strategy != PlannedStrategy::Flat,
        reason: plan_reason(winner.strategy, shape, config),
    }
}

fn selectivity(shape: &QueryShape) -> f32 {
    shape.matching_count as f32 / shape.total.max(1) as f32
}

fn flat_work(shape: &QueryShape, config: &PlannerConfig) -> f32 {
    let work = shape.matching_count.max(shape.k) as f32;
    if shape.matching_count <= config.flat_threshold.max(shape.k) {
        return work * 0.25;
    }
    if shape.historical_miss_rate >= 0.5
        && shape.matching_count <= config.flat_threshold.max(shape.k) * 4
    {
        return work * 0.25;
    }
    work
}

fn estimate(
    strategy: PlannedStrategy,
    expected_work: f32,
    recall_miss_risk: f32,
    config: &PlannerConfig,
) -> StrategyEstimate {
    StrategyEstimate {
        strategy,
        expected_work,
        recall_miss_risk,
        total_cost: expected_work + recall_miss_risk * config.recall_miss_penalty,
    }
}

fn post_filter_work(shape: &QueryShape, selectivity: f32, candidate_count: f32) -> f32 {
    if shape.correlation == CorrelationClass::AntiCorrelated && selectivity < 0.95 {
        return shape.total.max(1) as f32;
    }

    (candidate_count / selectivity.max(0.001)).min(shape.total.max(1) as f32)
}

fn post_filter_risk(
    selectivity: f32,
    correlation: CorrelationClass,
    history: f32,
    config: &PlannerConfig,
) -> f32 {
    let base = if selectivity >= config.broad_selectivity {
        0.02
    } else if selectivity >= 0.1 {
        0.25
    } else if selectivity >= config.strict_selectivity {
        0.55
    } else {
        0.85
    };
    let correlation_risk = match correlation {
        CorrelationClass::Positive => -0.05,
        CorrelationClass::Neutral => 0.0,
        CorrelationClass::AntiCorrelated => 0.55,
    };
    (base + correlation_risk + history).clamp(0.0, 1.0)
}

fn in_traversal_risk(
    selectivity: f32,
    correlation: CorrelationClass,
    history: f32,
    config: &PlannerConfig,
) -> f32 {
    let base = if selectivity < config.strict_selectivity {
        0.12
    } else if selectivity >= config.broad_selectivity {
        0.06
    } else {
        0.04
    };
    let correlation_risk = match correlation {
        CorrelationClass::Positive => -0.01,
        CorrelationClass::Neutral => 0.0,
        CorrelationClass::AntiCorrelated => 0.22,
    };
    (base + correlation_risk + history * 0.5).clamp(0.0, 1.0)
}

fn acorn_risk(
    selectivity: f32,
    correlation: CorrelationClass,
    history: f32,
    config: &PlannerConfig,
) -> f32 {
    let base = if correlation == CorrelationClass::AntiCorrelated
        || selectivity < config.strict_selectivity
    {
        0.03
    } else if selectivity >= config.broad_selectivity {
        0.08
    } else {
        0.05
    };
    (base + history * 0.25).clamp(0.0, 1.0)
}

fn in_traversal_work(shape: &QueryShape, selectivity: f32, candidate_count: f32) -> f32 {
    if shape.correlation == CorrelationClass::AntiCorrelated {
        return anti_correlated_graph_work(shape, selectivity);
    }

    let multiplier = match shape.correlation {
        CorrelationClass::Positive if selectivity >= 0.1 => 2.0,
        _ if selectivity < 0.01 => 8.0,
        _ if selectivity < 0.2 => 3.0,
        _ => 2.0,
    };
    candidate_count * multiplier
}

fn acorn_work(
    shape: &QueryShape,
    selectivity: f32,
    candidate_count: f32,
    config: &PlannerConfig,
) -> f32 {
    if shape.correlation == CorrelationClass::AntiCorrelated {
        return anti_correlated_graph_work(shape, selectivity);
    }

    if selectivity < config.strict_selectivity {
        candidate_count * 8.0
    } else if selectivity < 0.2 {
        candidate_count * 10.0
    } else {
        candidate_count * 12.0
    }
}

fn anti_correlated_graph_work(shape: &QueryShape, selectivity: f32) -> f32 {
    let total = shape.total.max(1) as f32;
    let graph_fraction = 1.0 - selectivity.clamp(0.0, 1.0) * 0.75;
    total * graph_fraction.max(0.1)
}

fn plan_reason(
    strategy: PlannedStrategy,
    shape: &QueryShape,
    config: &PlannerConfig,
) -> &'static str {
    match strategy {
        PlannedStrategy::Flat if shape.matching_count <= config.flat_threshold.max(shape.k) => {
            "cost model: exact scan has the lowest expected work"
        }
        PlannedStrategy::Flat if shape.correlation == CorrelationClass::AntiCorrelated => {
            "cost model: exact scan is cheaper than graph bridging"
        }
        PlannedStrategy::Flat => "cost model: exact scan avoids recall risk",
        PlannedStrategy::PostFilter => "cost model: broad aligned filter keeps post-filter cheap",
        PlannedStrategy::InTraversal => "cost model: filtered traversal balances recall and work",
        PlannedStrategy::Acorn => "cost model: bridge through non-matching nodes to protect recall",
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
        let mut shape = QueryShape::new(total, matching_count, 10, correlation);
        shape.candidate_count = 64;
        shape
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
    fn flat_for_small_anti_correlated_filter() {
        let plan = choose_plan(
            &shape(1_000, 100_000, CorrelationClass::AntiCorrelated),
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
    fn flat_for_large_anti_correlated_filter_when_graph_walk_is_worse() {
        let plan = choose_plan(
            &shape(100_000, 1_000_000, CorrelationClass::AntiCorrelated),
            &PlannerConfig::default(),
        );
        assert_eq!(plan.strategy, PlannedStrategy::Flat);
        assert!(!plan.adaptive);
    }

    #[test]
    fn post_filter_for_unfiltered_anti_correlated_shape() {
        let plan = choose_plan(
            &shape(1_000_000, 1_000_000, CorrelationClass::AntiCorrelated),
            &PlannerConfig::default(),
        );
        assert_eq!(plan.strategy, PlannedStrategy::PostFilter);
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
    fn estimates_include_every_strategy() {
        let estimates = estimate_strategy_costs(
            &shape(60_000, 100_000, CorrelationClass::Neutral),
            &PlannerConfig::default(),
        );
        assert_eq!(estimates.len(), 4);
        assert!(estimates.iter().any(|estimate| {
            estimate.strategy == PlannedStrategy::PostFilter && estimate.total_cost < 1_000.0
        }));
    }

    #[test]
    fn historical_misses_raise_post_filter_cost() {
        let mut query = shape(60_000, 100_000, CorrelationClass::Neutral);
        query.historical_miss_rate = 0.5;

        let plan = choose_plan(&query, &PlannerConfig::default());

        assert_ne!(plan.strategy, PlannedStrategy::PostFilter);
        assert!(plan.adaptive);
    }

    #[test]
    fn historical_misses_can_pick_exact_scan_above_flat_threshold() {
        let mut query = shape(5_000, 100_000, CorrelationClass::AntiCorrelated);
        query.historical_miss_rate = 0.8;

        let plan = choose_plan(&query, &PlannerConfig::default());

        assert_eq!(plan.strategy, PlannedStrategy::Flat);
        assert!(!plan.adaptive);
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
