//! Runs every filtered strategy across the selectivity x correlation matrix and
//! reports recall and cost per cell, scored against the flat-exact oracle.

use std::collections::HashSet;

use loupe_distance::Metric;
use loupe_filter::{Expr, MetadataStore};
use loupe_hnsw::{FilteredStrategy, HnswIndex, Neighbor, SearchTrace};
use loupe_planner::{
    PlannedStrategy, PlannerConfig, QueryShape, choose_plan, classify_correlation,
    estimate_correlation,
};

use crate::filters::{CORRELATION_MODES, Correlation, SELECTIVITY_LEVELS, generate_filter};

/// The strategies compared per cell. Flat is exact and doubles as the oracle the
/// others are scored against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Flat,
    PostFilter,
    InTraversal,
    Acorn,
    AdaptiveAcorn,
}

const STRATEGIES: [Strategy; 5] = [
    Strategy::Flat,
    Strategy::PostFilter,
    Strategy::InTraversal,
    Strategy::Acorn,
    Strategy::AdaptiveAcorn,
];

/// Settings shared across the whole matrix run.
pub struct MatrixConfig {
    pub k: usize,
    pub ef: usize,
    pub max_ef: usize,
    pub seed: u64,
}

/// One row of the matrix: a strategy's result in a single cell.
#[derive(Clone, Copy, Debug)]
pub struct CellResult {
    pub selectivity: f32,
    pub correlation: Correlation,
    pub strategy: Strategy,
    pub recall: f32,
    pub avg_distance_computations: f64,
    pub avg_returned: f64,
}

/// Runs every strategy in every cell over all queries and returns one result per
/// (cell, strategy). The index is built once and reused.
pub fn run_matrix(
    index: &HnswIndex,
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    metric: Metric,
    config: &MatrixConfig,
) -> Vec<CellResult> {
    let mut results = Vec::new();
    let query_count = queries.len().max(1) as f64;

    for &selectivity in &SELECTIVITY_LEVELS {
        for &correlation in &CORRELATION_MODES {
            let mut recall_sum = [0.0f64; STRATEGIES.len()];
            let mut comps_sum = [0.0f64; STRATEGIES.len()];
            let mut returned_sum = [0.0f64; STRATEGIES.len()];

            for query in queries {
                let allowed =
                    generate_filter(base, query, metric, selectivity, correlation, config.seed);
                let (oracle, oracle_trace) = index.flat_filtered_search(query, config.k, &allowed);
                let oracle_ids: HashSet<u32> = oracle.iter().map(|n| n.id).collect();

                for (slot, &strategy) in STRATEGIES.iter().enumerate() {
                    let (got, trace) = if strategy == Strategy::Flat {
                        (oracle.clone(), oracle_trace.clone())
                    } else {
                        run_strategy(index, strategy, query, &allowed, config)
                    };
                    recall_sum[slot] += f64::from(overlap(&got, &oracle_ids));
                    comps_sum[slot] += trace.distance_computations as f64;
                    returned_sum[slot] += got.len() as f64;
                }
            }

            for (slot, &strategy) in STRATEGIES.iter().enumerate() {
                results.push(CellResult {
                    selectivity,
                    correlation,
                    strategy,
                    recall: (recall_sum[slot] / query_count) as f32,
                    avg_distance_computations: comps_sum[slot] / query_count,
                    avg_returned: returned_sum[slot] / query_count,
                });
            }
        }
    }
    results
}

fn run_strategy(
    index: &HnswIndex,
    strategy: Strategy,
    query: &[f32],
    allowed: &roaring::RoaringBitmap,
    config: &MatrixConfig,
) -> (Vec<Neighbor>, SearchTrace) {
    match strategy {
        Strategy::Flat => index.flat_filtered_search(query, config.k, allowed),
        Strategy::PostFilter => index.post_filter_search(query, config.k, allowed, config.ef),
        Strategy::InTraversal => {
            index.in_traversal_filtered_search(query, config.k, allowed, config.ef)
        }
        Strategy::Acorn => index.acorn_filtered_search(query, config.k, allowed, config.ef),
        Strategy::AdaptiveAcorn => index.adaptive_filtered_search(
            query,
            config.k,
            allowed,
            FilteredStrategy::Acorn,
            config.ef,
            config.max_ef,
        ),
    }
}

fn overlap(got: &[Neighbor], oracle_ids: &HashSet<u32>) -> f32 {
    if oracle_ids.is_empty() {
        return 1.0;
    }
    let hits = got.iter().filter(|n| oracle_ids.contains(&n.id)).count();
    hits as f32 / oracle_ids.len() as f32
}

/// Prints the matrix as a plain table.
pub fn print_matrix(results: &[CellResult]) {
    println!(
        "{:>8}  {:<18}  {:<14}  {:>7}  {:>12}  {:>9}",
        "select", "correlation", "strategy", "recall", "dist/query", "returned"
    );
    for row in results {
        println!(
            "{:>7.1}%  {:<18}  {:<14}  {:>7.3}  {:>12.0}  {:>9.1}",
            row.selectivity * 100.0,
            format!("{:?}", row.correlation),
            format!("{:?}", row.strategy),
            row.recall,
            row.avg_distance_computations,
            row.avg_returned,
        );
    }
}

/// One row of the real-metadata matrix: a strategy's result for a named filter.
#[derive(Clone, Debug)]
pub struct FilterResult {
    pub label: String,
    pub selectivity: f32,
    pub strategy: Strategy,
    pub recall: f32,
    pub avg_distance_computations: f64,
    pub avg_returned: f64,
}

/// Runs every strategy against a set of real metadata filters. Unlike the
/// synthetic matrix, each filter is one predicate evaluated over the whole store,
/// so its allowed set is the same for every query.
pub fn run_metadata_matrix(
    index: &HnswIndex,
    metadata: &MetadataStore,
    filters: &[(String, Expr)],
    queries: &[Vec<f32>],
    config: &MatrixConfig,
) -> Vec<FilterResult> {
    let mut results = Vec::new();
    let query_count = queries.len().max(1) as f64;

    for (label, expr) in filters {
        let allowed = metadata.evaluate(expr);
        let selectivity = metadata.selectivity(expr);
        let mut recall_sum = [0.0f64; STRATEGIES.len()];
        let mut comps_sum = [0.0f64; STRATEGIES.len()];
        let mut returned_sum = [0.0f64; STRATEGIES.len()];

        for query in queries {
            let (oracle, oracle_trace) = index.flat_filtered_search(query, config.k, &allowed);
            let oracle_ids: HashSet<u32> = oracle.iter().map(|n| n.id).collect();
            for (slot, &strategy) in STRATEGIES.iter().enumerate() {
                let (got, trace) = if strategy == Strategy::Flat {
                    (oracle.clone(), oracle_trace.clone())
                } else {
                    run_strategy(index, strategy, query, &allowed, config)
                };
                recall_sum[slot] += f64::from(overlap(&got, &oracle_ids));
                comps_sum[slot] += trace.distance_computations as f64;
                returned_sum[slot] += got.len() as f64;
            }
        }

        for (slot, &strategy) in STRATEGIES.iter().enumerate() {
            results.push(FilterResult {
                label: label.clone(),
                selectivity,
                strategy,
                recall: (recall_sum[slot] / query_count) as f32,
                avg_distance_computations: comps_sum[slot] / query_count,
                avg_returned: returned_sum[slot] / query_count,
            });
        }
    }
    results
}

/// Prints the real-metadata matrix as a plain table.
pub fn print_filter_matrix(results: &[FilterResult]) {
    println!(
        "{:<22}  {:>7}  {:<14}  {:>7}  {:>12}  {:>9}",
        "filter", "select", "strategy", "recall", "dist/query", "returned"
    );
    for row in results {
        println!(
            "{:<22}  {:>6.2}%  {:<14}  {:>7.3}  {:>12.0}  {:>9.1}",
            row.label,
            row.selectivity * 100.0,
            format!("{:?}", row.strategy),
            row.recall,
            row.avg_distance_computations,
            row.avg_returned,
        );
    }
}

/// Mean recall and cost for one strategy across the whole matrix.
#[derive(Clone, Debug)]
pub struct StrategySummary {
    pub name: String,
    pub mean_recall: f32,
    pub mean_distance_computations: f64,
}

/// The planner compared against the fixed strategies over the whole matrix, plus
/// the planner's regret: how often its pick was beaten on both recall and cost.
#[derive(Clone, Debug)]
pub struct PlannerSummary {
    pub strategies: Vec<StrategySummary>,
    pub regret: f32,
}

/// Runs the planner and the four fixed strategies over every cell, then reports
/// each one's mean recall and cost and the planner's regret. Regret counts the
/// queries where some fixed strategy matched or beat the planner's recall at no
/// more cost (and was strictly better on one) — i.e. the planner missed the
/// recall/cost frontier.
pub fn summarize_planner(
    index: &HnswIndex,
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    metric: Metric,
    matrix: &MatrixConfig,
    planner: &PlannerConfig,
    samples: usize,
) -> PlannerSummary {
    let fixed = [
        Strategy::Flat,
        Strategy::PostFilter,
        Strategy::InTraversal,
        Strategy::Acorn,
    ];
    let mut recall_sum = [0.0f64; 5];
    let mut comps_sum = [0.0f64; 5];
    let mut regretted = 0usize;
    let mut total = 0usize;

    for &selectivity in &SELECTIVITY_LEVELS {
        for &correlation in &CORRELATION_MODES {
            for query in queries {
                let allowed =
                    generate_filter(base, query, metric, selectivity, correlation, matrix.seed);
                let (oracle, _) = index.flat_filtered_search(query, matrix.k, &allowed);
                let oracle_ids: HashSet<u32> = oracle.iter().map(|n| n.id).collect();

                let mut fixed_outcomes = Vec::with_capacity(fixed.len());
                for (slot, &strategy) in fixed.iter().enumerate() {
                    let (results, trace) = run_strategy(index, strategy, query, &allowed, matrix);
                    let recall = overlap(&results, &oracle_ids);
                    recall_sum[slot] += f64::from(recall);
                    comps_sum[slot] += trace.distance_computations as f64;
                    fixed_outcomes.push((recall, trace.distance_computations));
                }

                let (planner_comps, planner_recall) = run_planner(
                    index,
                    query,
                    &allowed,
                    matrix,
                    planner,
                    samples,
                    &oracle_ids,
                );
                recall_sum[4] += f64::from(planner_recall);
                comps_sum[4] += planner_comps as f64;

                let dominated = fixed_outcomes.iter().any(|&(recall, comps)| {
                    recall >= planner_recall
                        && comps <= planner_comps
                        && (recall > planner_recall || comps < planner_comps)
                });
                if dominated {
                    regretted += 1;
                }
                total += 1;
            }
        }
    }

    let total_f = total.max(1) as f64;
    let names = ["Flat", "PostFilter", "InTraversal", "Acorn", "Planner"];
    let strategies = names
        .iter()
        .zip(recall_sum.iter())
        .zip(comps_sum.iter())
        .map(|((name, &recall), &comps)| StrategySummary {
            name: (*name).to_string(),
            mean_recall: (recall / total_f) as f32,
            mean_distance_computations: comps / total_f,
        })
        .collect();

    PlannerSummary {
        strategies,
        regret: (regretted as f64 / total_f) as f32,
    }
}

fn run_planner(
    index: &HnswIndex,
    query: &[f32],
    allowed: &roaring::RoaringBitmap,
    matrix: &MatrixConfig,
    planner: &PlannerConfig,
    samples: usize,
    oracle_ids: &HashSet<u32>,
) -> (u64, f32) {
    let score = estimate_correlation(
        index.accessor(),
        index.metric(),
        query,
        allowed,
        samples,
        matrix.seed,
    );
    let shape = QueryShape {
        total: index.len(),
        matching_count: allowed.len() as usize,
        k: matrix.k,
        correlation: classify_correlation(score, planner.correlation_band),
    };
    let plan = choose_plan(&shape, planner);
    let (results, trace) = match plan.strategy {
        PlannedStrategy::Flat => index.flat_filtered_search(query, matrix.k, allowed),
        PlannedStrategy::PostFilter => index.post_filter_search(query, matrix.k, allowed, plan.ef),
        PlannedStrategy::InTraversal => {
            index.in_traversal_filtered_search(query, matrix.k, allowed, plan.ef)
        }
        PlannedStrategy::Acorn => index.acorn_filtered_search(query, matrix.k, allowed, plan.ef),
    };
    (trace.distance_computations, overlap(&results, oracle_ids))
}

/// Prints the planner-vs-fixed summary as a plain table.
pub fn print_planner_summary(summary: &PlannerSummary) {
    println!(
        "{:<14}  {:>11}  {:>16}",
        "strategy", "mean recall", "mean dist/query"
    );
    for row in &summary.strategies {
        println!(
            "{:<14}  {:>11.3}  {:>16.0}",
            row.name, row.mean_recall, row.mean_distance_computations
        );
    }
    println!("planner_regret: {:.3}", summary.regret);
}

#[cfg(test)]
mod tests {
    use super::*;
    use loupe_filter::Value;
    use loupe_hnsw::HnswConfig;

    fn small_setup() -> (HnswIndex, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let base: Vec<Vec<f32>> = (0..200).map(|i| vec![i as f32, i as f32]).collect();
        let mut index = HnswIndex::new(2, Metric::L2, HnswConfig::default());
        for vector in &base {
            index.insert(vector);
        }
        let queries = vec![vec![50.0, 50.0], vec![120.0, 120.0]];
        (index, base, queries)
    }

    fn config() -> MatrixConfig {
        MatrixConfig {
            k: 5,
            ef: 64,
            max_ef: 256,
            seed: 1,
        }
    }

    #[test]
    fn matrix_has_one_row_per_cell_and_strategy() {
        let (index, base, queries) = small_setup();
        let results = run_matrix(&index, &base, &queries, Metric::L2, &config());
        let expected = SELECTIVITY_LEVELS.len() * CORRELATION_MODES.len() * STRATEGIES.len();
        assert_eq!(results.len(), expected);
    }

    #[test]
    fn flat_strategy_always_has_full_recall() {
        let (index, base, queries) = small_setup();
        let results = run_matrix(&index, &base, &queries, Metric::L2, &config());
        for row in results.iter().filter(|r| r.strategy == Strategy::Flat) {
            assert!(row.recall >= 0.999, "flat recall {} too low", row.recall);
        }
    }

    fn metadata_setup() -> (HnswIndex, MetadataStore) {
        let mut index = HnswIndex::new(2, Metric::L2, HnswConfig::default());
        let mut metadata = MetadataStore::new();
        for i in 0..100u32 {
            index.insert(&[i as f32, i as f32]);
            let category = if i % 2 == 0 { "even" } else { "odd" };
            metadata.insert(i, &[("category", Value::Str(category.into()))]);
        }
        (index, metadata)
    }

    #[test]
    fn planner_summary_covers_all_strategies_and_keeps_recall_high() {
        let base: Vec<Vec<f32>> = (0..300).map(|i| vec![i as f32, i as f32]).collect();
        let mut index = HnswIndex::new(2, Metric::L2, HnswConfig::default());
        for vector in &base {
            index.insert(vector);
        }
        let queries = vec![vec![50.0, 50.0], vec![150.0, 150.0], vec![250.0, 250.0]];
        let planner = PlannerConfig {
            flat_threshold: 20,
            ..Default::default()
        };
        let summary =
            summarize_planner(&index, &base, &queries, Metric::L2, &config(), &planner, 16);

        assert_eq!(summary.strategies.len(), 5);
        assert!((0.0..=1.0).contains(&summary.regret));
        let planner_row = summary.strategies.last().unwrap();
        assert_eq!(planner_row.name, "Planner");
        assert!(
            planner_row.mean_recall >= 0.9,
            "planner recall {} too low",
            planner_row.mean_recall
        );
    }

    #[test]
    fn metadata_matrix_rows_and_flat_recall() {
        let (index, metadata) = metadata_setup();
        let queries = vec![vec![40.0, 40.0]];
        let filters = vec![(
            "category=even".to_string(),
            Expr::Eq("category".into(), Value::Str("even".into())),
        )];
        let results = run_metadata_matrix(&index, &metadata, &filters, &queries, &config());
        assert_eq!(results.len(), STRATEGIES.len());
        assert!((results[0].selectivity - 0.5).abs() < 0.01);
        for row in results.iter().filter(|r| r.strategy == Strategy::Flat) {
            assert!(row.recall >= 0.999);
        }
    }
}
