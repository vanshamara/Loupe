//! Runs every filtered strategy across the selectivity x correlation matrix and
//! reports recall and cost per cell, scored against the flat-exact oracle.

use std::collections::{BTreeMap, HashSet};
use std::io;
use std::time::Instant;

use loupe_distance::Metric;
use loupe_filter::{Expr, MetadataStore};
use loupe_hnsw::{FilteredStrategy, HnswIndex, Neighbor, SearchTrace};
use loupe_planner::{
    CorrelationClass, Plan, PlannedStrategy, PlannerConfig, QueryShape, choose_plan,
    classify_correlation, estimate_correlation,
};

use crate::filters::{CORRELATION_MODES, Correlation, SELECTIVITY_LEVELS, generate_filter};

/// The strategies compared per cell. Flat is exact and doubles as the oracle the
/// others are scored against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Strategy {
    Flat,
    PostFilter,
    InTraversal,
    Acorn,
    AdaptiveAcorn,
    Planner,
}

const STRATEGIES: [Strategy; 5] = [
    Strategy::Flat,
    Strategy::PostFilter,
    Strategy::InTraversal,
    Strategy::Acorn,
    Strategy::AdaptiveAcorn,
];

const MATRIX_RUNNERS: [Strategy; 6] = [
    Strategy::Flat,
    Strategy::PostFilter,
    Strategy::InTraversal,
    Strategy::Acorn,
    Strategy::AdaptiveAcorn,
    Strategy::Planner,
];

/// Settings shared across the whole matrix run.
pub struct MatrixConfig {
    pub k: usize,
    pub ef: usize,
    pub max_ef: usize,
    pub seed: u64,
}

/// One row of the matrix: a strategy's result in a single cell.
#[derive(Clone, Debug)]
pub struct CellResult {
    pub selectivity: f32,
    pub correlation: Correlation,
    pub strategy: Strategy,
    pub strategy_chosen: String,
    pub recall: f32,
    pub avg_distance_computations: f64,
    pub avg_latency_micros: f64,
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
    let planner = PlannerConfig::default();
    run_matrix_with_planner(index, base, queries, metric, config, &planner, 32)
}

/// Runs every fixed strategy plus the planner in every matrix cell.
pub fn run_matrix_with_planner(
    index: &HnswIndex,
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    metric: Metric,
    config: &MatrixConfig,
    planner: &PlannerConfig,
    samples: usize,
) -> Vec<CellResult> {
    let mut results = Vec::new();
    let query_count = queries.len().max(1) as f64;

    for &selectivity in &SELECTIVITY_LEVELS {
        for &correlation in &CORRELATION_MODES {
            let mut recall_sum = [0.0f64; MATRIX_RUNNERS.len()];
            let mut comps_sum = [0.0f64; MATRIX_RUNNERS.len()];
            let mut latency_sum = [0.0f64; MATRIX_RUNNERS.len()];
            let mut returned_sum = [0.0f64; MATRIX_RUNNERS.len()];
            let mut chosen_counts: Vec<BTreeMap<String, usize>> =
                vec![BTreeMap::new(); MATRIX_RUNNERS.len()];

            for query in queries {
                let allowed =
                    generate_filter(base, query, metric, selectivity, correlation, config.seed);
                let oracle_start = Instant::now();
                let (oracle, oracle_trace) = index.flat_filtered_search(query, config.k, &allowed);
                let oracle_latency = oracle_start.elapsed().as_micros() as f64;
                let oracle_ids: HashSet<u32> = oracle.iter().map(|n| n.id).collect();

                for (slot, &strategy) in MATRIX_RUNNERS.iter().enumerate() {
                    let outcome = match strategy {
                        Strategy::Flat => StrategyOutcome {
                            results: oracle.clone(),
                            trace: oracle_trace.clone(),
                            latency_micros: oracle_latency,
                            strategy_chosen: strategy_name(strategy).to_string(),
                        },
                        Strategy::Planner => {
                            run_planner_strategy(index, query, &allowed, config, planner, samples)
                        }
                        _ => run_strategy(index, strategy, query, &allowed, config),
                    };
                    recall_sum[slot] += f64::from(overlap(&outcome.results, &oracle_ids));
                    comps_sum[slot] += outcome.trace.distance_computations as f64;
                    latency_sum[slot] += outcome.latency_micros;
                    returned_sum[slot] += outcome.results.len() as f64;
                    *chosen_counts[slot]
                        .entry(outcome.strategy_chosen)
                        .or_default() += 1;
                }
            }

            for (slot, &strategy) in MATRIX_RUNNERS.iter().enumerate() {
                results.push(CellResult {
                    selectivity,
                    correlation,
                    strategy,
                    strategy_chosen: most_common_choice(&chosen_counts[slot]),
                    recall: (recall_sum[slot] / query_count) as f32,
                    avg_distance_computations: comps_sum[slot] / query_count,
                    avg_latency_micros: latency_sum[slot] / query_count,
                    avg_returned: returned_sum[slot] / query_count,
                });
            }
        }
    }
    results
}

struct StrategyOutcome {
    results: Vec<Neighbor>,
    trace: SearchTrace,
    latency_micros: f64,
    strategy_chosen: String,
}

fn run_strategy(
    index: &HnswIndex,
    strategy: Strategy,
    query: &[f32],
    allowed: &roaring::RoaringBitmap,
    config: &MatrixConfig,
) -> StrategyOutcome {
    let start = Instant::now();
    let (results, trace) = match strategy {
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
        Strategy::Planner => unreachable!("planner is not a fixed strategy"),
    };
    StrategyOutcome {
        results,
        trace,
        latency_micros: start.elapsed().as_micros() as f64,
        strategy_chosen: strategy_name(strategy).to_string(),
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
        "{:>8}  {:<18}  {:<14}  {:<20}  {:>7}  {:>12}  {:>10}  {:>9}",
        "select",
        "correlation",
        "runner",
        "chosen",
        "recall",
        "dist/query",
        "latency_us",
        "returned"
    );
    for row in results {
        println!(
            "{:>7.1}%  {:<18}  {:<14}  {:<20}  {:>7.3}  {:>12.0}  {:>10.1}  {:>9.1}",
            row.selectivity * 100.0,
            correlation_name(row.correlation),
            strategy_name(row.strategy),
            row.strategy_chosen,
            row.recall,
            row.avg_distance_computations,
            row.avg_latency_micros,
            row.avg_returned,
        );
    }
}

/// Writes matrix results as CSV for plotting and regression analysis.
pub fn write_matrix_csv(path: &str, results: &[CellResult]) -> io::Result<()> {
    std::fs::write(path, matrix_csv(results))
}

/// Returns matrix results as CSV text.
pub fn matrix_csv(results: &[CellResult]) -> String {
    let mut csv = String::from(
        "engine,selectivity,correlation,runner,strategy_chosen,recall,distance_computations,latency_micros,returned_count\n",
    );
    for row in results {
        csv.push_str(&format!(
            "loupe,{:.6},{},{},{},{:.6},{:.3},{:.3},{:.3}\n",
            row.selectivity,
            correlation_name(row.correlation),
            strategy_name(row.strategy),
            row.strategy_chosen,
            row.recall,
            row.avg_distance_computations,
            row.avg_latency_micros,
            row.avg_returned
        ));
    }
    csv
}

/// Writes planner comparison results as CSV for M11 calibration.
pub fn write_planner_summary_csv(path: &str, summary: &PlannerSummary) -> io::Result<()> {
    std::fs::write(path, planner_summary_csv(summary))
}

/// Returns planner comparison results as CSV text.
pub fn planner_summary_csv(summary: &PlannerSummary) -> String {
    let mut csv =
        String::from("planner,mean_recall,mean_distance_computations,regret,regret_delta\n");
    let regret_delta = summary.rule_ladder_regret - summary.cost_model_regret;

    for planner_name in ["RuleLadder", "CostModel"] {
        if let Some(row) = summary
            .strategies
            .iter()
            .find(|strategy| strategy.name == planner_name)
        {
            let regret = if planner_name == "RuleLadder" {
                summary.rule_ladder_regret
            } else {
                summary.cost_model_regret
            };
            csv.push_str(&format!(
                "{},{:.6},{:.3},{:.6},{:.6}\n",
                row.name, row.mean_recall, row.mean_distance_computations, regret, regret_delta
            ));
        }
    }

    csv
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
                let outcome = if strategy == Strategy::Flat {
                    StrategyOutcome {
                        results: oracle.clone(),
                        trace: oracle_trace.clone(),
                        latency_micros: 0.0,
                        strategy_chosen: strategy_name(strategy).to_string(),
                    }
                } else {
                    run_strategy(index, strategy, query, &allowed, config)
                };
                recall_sum[slot] += f64::from(overlap(&outcome.results, &oracle_ids));
                comps_sum[slot] += outcome.trace.distance_computations as f64;
                returned_sum[slot] += outcome.results.len() as f64;
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
            strategy_name(row.strategy),
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

/// The planner comparison over the whole matrix.
#[derive(Clone, Debug)]
pub struct PlannerSummary {
    pub strategies: Vec<StrategySummary>,
    pub rule_ladder_regret: f32,
    pub cost_model_regret: f32,
}

/// Runs both planner generations and the fixed strategies across every matrix cell.
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
    let mut recall_sum = [0.0f64; 6];
    let mut comps_sum = [0.0f64; 6];
    let mut rule_ladder_regretted = 0usize;
    let mut cost_model_regretted = 0usize;
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
                    let outcome = run_strategy(index, strategy, query, &allowed, matrix);
                    let recall = overlap(&outcome.results, &oracle_ids);
                    recall_sum[slot] += f64::from(recall);
                    comps_sum[slot] += outcome.trace.distance_computations as f64;
                    fixed_outcomes.push((recall, outcome.trace.distance_computations));
                }

                let (rule_ladder_comps, rule_ladder_recall) = run_rule_ladder(
                    index,
                    query,
                    &allowed,
                    matrix,
                    planner,
                    samples,
                    &oracle_ids,
                );
                recall_sum[4] += f64::from(rule_ladder_recall);
                comps_sum[4] += rule_ladder_comps as f64;

                let (cost_model_comps, cost_model_recall) = run_cost_model(
                    index,
                    query,
                    &allowed,
                    matrix,
                    planner,
                    samples,
                    &oracle_ids,
                );
                recall_sum[5] += f64::from(cost_model_recall);
                comps_sum[5] += cost_model_comps as f64;

                if is_dominated(&fixed_outcomes, rule_ladder_recall, rule_ladder_comps) {
                    rule_ladder_regretted += 1;
                }
                if is_dominated(&fixed_outcomes, cost_model_recall, cost_model_comps) {
                    cost_model_regretted += 1;
                }
                total += 1;
            }
        }
    }

    let total_f = total.max(1) as f64;
    let names = [
        "Flat",
        "PostFilter",
        "InTraversal",
        "Acorn",
        "RuleLadder",
        "CostModel",
    ];
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
        rule_ladder_regret: (rule_ladder_regretted as f64 / total_f) as f32,
        cost_model_regret: (cost_model_regretted as f64 / total_f) as f32,
    }
}

fn run_rule_ladder(
    index: &HnswIndex,
    query: &[f32],
    allowed: &roaring::RoaringBitmap,
    matrix: &MatrixConfig,
    planner: &PlannerConfig,
    samples: usize,
    oracle_ids: &HashSet<u32>,
) -> (u64, f32) {
    let outcome = run_rule_ladder_strategy(index, query, allowed, matrix, planner, samples);
    (
        outcome.trace.distance_computations,
        overlap(&outcome.results, oracle_ids),
    )
}

fn run_cost_model(
    index: &HnswIndex,
    query: &[f32],
    allowed: &roaring::RoaringBitmap,
    matrix: &MatrixConfig,
    planner: &PlannerConfig,
    samples: usize,
    oracle_ids: &HashSet<u32>,
) -> (u64, f32) {
    let outcome = run_planner_strategy(index, query, allowed, matrix, planner, samples);
    (
        outcome.trace.distance_computations,
        overlap(&outcome.results, oracle_ids),
    )
}

fn is_dominated(fixed_outcomes: &[(f32, u64)], planner_recall: f32, planner_comps: u64) -> bool {
    fixed_outcomes.iter().any(|&(recall, comps)| {
        recall >= planner_recall
            && comps <= planner_comps
            && (recall > planner_recall || comps < planner_comps)
    })
}

fn run_rule_ladder_strategy(
    index: &HnswIndex,
    query: &[f32],
    allowed: &roaring::RoaringBitmap,
    matrix: &MatrixConfig,
    planner: &PlannerConfig,
    samples: usize,
) -> StrategyOutcome {
    let score = estimate_correlation(
        index.accessor(),
        index.metric(),
        query,
        allowed,
        samples,
        matrix.seed,
    );
    let correlation = classify_correlation(score, planner.correlation_band);
    let mut shape = QueryShape::new(index.len(), allowed.len() as usize, matrix.k, correlation);
    shape.candidate_count = matrix.ef.max(matrix.k);
    let plan = choose_rule_ladder_plan(&shape, planner);
    run_plan(index, query, allowed, matrix, &plan)
}

fn run_planner_strategy(
    index: &HnswIndex,
    query: &[f32],
    allowed: &roaring::RoaringBitmap,
    matrix: &MatrixConfig,
    planner: &PlannerConfig,
    samples: usize,
) -> StrategyOutcome {
    let score = estimate_correlation(
        index.accessor(),
        index.metric(),
        query,
        allowed,
        samples,
        matrix.seed,
    );
    let correlation = classify_correlation(score, planner.correlation_band);
    let mut shape = QueryShape::new(index.len(), allowed.len() as usize, matrix.k, correlation);
    shape.candidate_count = matrix.ef.max(matrix.k);
    let plan = choose_plan(&shape, planner);
    run_plan(index, query, allowed, matrix, &plan)
}

fn run_plan(
    index: &HnswIndex,
    query: &[f32],
    allowed: &roaring::RoaringBitmap,
    matrix: &MatrixConfig,
    plan: &Plan,
) -> StrategyOutcome {
    let start = Instant::now();
    let (results, trace) = if plan.adaptive {
        match plan.strategy {
            PlannedStrategy::Flat => index.flat_filtered_search(query, matrix.k, allowed),
            PlannedStrategy::PostFilter => index.adaptive_filtered_search(
                query,
                matrix.k,
                allowed,
                FilteredStrategy::PostFilter,
                plan.ef,
                plan.max_ef,
            ),
            PlannedStrategy::InTraversal => index.adaptive_filtered_search(
                query,
                matrix.k,
                allowed,
                FilteredStrategy::InTraversal,
                plan.ef,
                plan.max_ef,
            ),
            PlannedStrategy::Acorn => index.adaptive_filtered_search(
                query,
                matrix.k,
                allowed,
                FilteredStrategy::Acorn,
                plan.ef,
                plan.max_ef,
            ),
        }
    } else {
        match plan.strategy {
            PlannedStrategy::Flat => index.flat_filtered_search(query, matrix.k, allowed),
            PlannedStrategy::PostFilter => {
                index.post_filter_search(query, matrix.k, allowed, plan.ef)
            }
            PlannedStrategy::InTraversal => {
                index.in_traversal_filtered_search(query, matrix.k, allowed, plan.ef)
            }
            PlannedStrategy::Acorn => {
                index.acorn_filtered_search(query, matrix.k, allowed, plan.ef)
            }
        }
    };
    StrategyOutcome {
        results,
        trace,
        latency_micros: start.elapsed().as_micros() as f64,
        strategy_chosen: plan_name(plan),
    }
}

fn choose_rule_ladder_plan(shape: &QueryShape, config: &PlannerConfig) -> Plan {
    let ef = config.ef.max(shape.k);
    let max_ef = config.max_ef.max(ef);
    let selectivity = shape.matching_count as f32 / shape.total.max(1) as f32;

    let strategy = if shape.matching_count <= config.flat_threshold.max(shape.k) {
        PlannedStrategy::Flat
    } else if shape.correlation == CorrelationClass::AntiCorrelated {
        PlannedStrategy::Acorn
    } else if selectivity >= config.broad_selectivity {
        PlannedStrategy::PostFilter
    } else if selectivity < config.strict_selectivity {
        PlannedStrategy::Acorn
    } else {
        PlannedStrategy::InTraversal
    };

    Plan {
        strategy,
        ef,
        max_ef,
        adaptive: strategy != PlannedStrategy::Flat,
        reason: "benchmark rule-ladder baseline",
    }
}

fn strategy_name(strategy: Strategy) -> &'static str {
    match strategy {
        Strategy::Flat => "Flat",
        Strategy::PostFilter => "PostFilter",
        Strategy::InTraversal => "InTraversal",
        Strategy::Acorn => "Acorn",
        Strategy::AdaptiveAcorn => "Acorn+AdaptiveEf",
        Strategy::Planner => "Planner",
    }
}

fn plan_name(plan: &Plan) -> String {
    let base = planned_strategy_name(plan.strategy);
    if plan.adaptive {
        return format!("{base}+AdaptiveEf");
    }

    base.to_string()
}

fn planned_strategy_name(strategy: PlannedStrategy) -> &'static str {
    match strategy {
        PlannedStrategy::Flat => "Flat",
        PlannedStrategy::PostFilter => "PostFilter",
        PlannedStrategy::InTraversal => "InTraversal",
        PlannedStrategy::Acorn => "Acorn",
    }
}

fn correlation_name(correlation: Correlation) -> &'static str {
    match correlation {
        Correlation::Random => "random",
        Correlation::Positive => "positive",
        Correlation::AntiCorrelated => "anti_correlated",
        Correlation::ClusteredByTenant => "clustered_by_tenant",
    }
}

fn most_common_choice(counts: &BTreeMap<String, usize>) -> String {
    counts
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(name, _)| name.clone())
        .unwrap_or_else(|| "unknown".to_string())
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
    println!("rule_ladder_regret: {:.3}", summary.rule_ladder_regret);
    println!("cost_model_regret:  {:.3}", summary.cost_model_regret);
    println!(
        "regret_delta:       {:.3}",
        summary.rule_ladder_regret - summary.cost_model_regret
    );
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
        let expected = SELECTIVITY_LEVELS.len() * CORRELATION_MODES.len() * MATRIX_RUNNERS.len();
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

    #[test]
    fn matrix_includes_planner_rows_with_chosen_strategy() {
        let (index, base, queries) = small_setup();
        let results = run_matrix(&index, &base, &queries, Metric::L2, &config());
        let planner_rows: Vec<&CellResult> = results
            .iter()
            .filter(|row| row.strategy == Strategy::Planner)
            .collect();

        assert_eq!(
            planner_rows.len(),
            SELECTIVITY_LEVELS.len() * CORRELATION_MODES.len()
        );
        for row in planner_rows {
            assert!(!row.strategy_chosen.is_empty());
            assert!(row.avg_latency_micros >= 0.0);
        }
    }

    #[test]
    fn matrix_csv_has_plot_ready_columns() {
        let rows = vec![CellResult {
            selectivity: 0.01,
            correlation: Correlation::AntiCorrelated,
            strategy: Strategy::Planner,
            strategy_chosen: "Acorn+AdaptiveEf".to_string(),
            recall: 0.95,
            avg_distance_computations: 123.0,
            avg_latency_micros: 45.0,
            avg_returned: 10.0,
        }];
        let csv = matrix_csv(&rows);

        assert!(csv.starts_with(
            "engine,selectivity,correlation,runner,strategy_chosen,recall,distance_computations,latency_micros,returned_count\n"
        ));
        assert!(csv.contains("loupe,0.010000,anti_correlated,Planner,Acorn+AdaptiveEf"));
    }

    #[test]
    fn planner_summary_csv_has_cost_model_rows() {
        let summary = PlannerSummary {
            strategies: vec![
                StrategySummary {
                    name: "RuleLadder".to_string(),
                    mean_recall: 0.9,
                    mean_distance_computations: 100.0,
                },
                StrategySummary {
                    name: "CostModel".to_string(),
                    mean_recall: 0.95,
                    mean_distance_computations: 80.0,
                },
            ],
            rule_ladder_regret: 0.2,
            cost_model_regret: 0.1,
        };

        let csv = planner_summary_csv(&summary);

        assert!(
            csv.starts_with("planner,mean_recall,mean_distance_computations,regret,regret_delta\n")
        );
        assert!(csv.contains("RuleLadder,0.900000,100.000,0.200000,0.100000"));
        assert!(csv.contains("CostModel,0.950000,80.000,0.100000,0.100000"));
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

        assert_eq!(summary.strategies.len(), 6);
        assert!((0.0..=1.0).contains(&summary.rule_ladder_regret));
        assert!((0.0..=1.0).contains(&summary.cost_model_regret));
        assert!(
            summary
                .strategies
                .iter()
                .any(|row| row.name == "RuleLadder")
        );
        let planner_row = summary.strategies.last().unwrap();
        assert_eq!(planner_row.name, "CostModel");
        assert!(
            planner_row.mean_recall >= 0.9,
            "planner recall {} too low",
            planner_row.mean_recall
        );
    }

    #[test]
    fn rule_ladder_baseline_keeps_m4_strategy_choices() {
        let planner = PlannerConfig {
            flat_threshold: 20,
            ..Default::default()
        };
        let broad = QueryShape::new(100_000, 60_000, 10, CorrelationClass::Neutral);
        let anti = QueryShape::new(100_000, 60_000, 10, CorrelationClass::AntiCorrelated);
        let strict = QueryShape::new(1_000_000, 3_000, 10, CorrelationClass::Neutral);
        let middle = QueryShape::new(100_000, 10_000, 10, CorrelationClass::Neutral);

        assert_eq!(
            choose_rule_ladder_plan(&broad, &planner).strategy,
            PlannedStrategy::PostFilter
        );
        assert_eq!(
            choose_rule_ladder_plan(&anti, &planner).strategy,
            PlannedStrategy::Acorn
        );
        assert_eq!(
            choose_rule_ladder_plan(&strict, &planner).strategy,
            PlannedStrategy::Acorn
        );
        assert_eq!(
            choose_rule_ladder_plan(&middle, &planner).strategy,
            PlannedStrategy::InTraversal
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
