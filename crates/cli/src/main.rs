//! The `loupe` command line: build a durable collection from an embedded corpus,
//! then run `explain` to see the chosen plan and what the rejected plans would
//! have cost.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{self, Write};

use clap::{Parser, Subcommand};
use loupe_advisor::advise;
use loupe_distance::Metric;
use loupe_engine::{Collection, ExplainSearch, PlanReport, QueryLogRecord, Value, parse_filter};
use loupe_planner::PlannerConfig;
use serde::{Deserialize, Serialize};

const RECALL_REGRET_GAP: f32 = 0.05;
const SATISFIED_RECALL: f32 = 0.95;
const MAX_RECALL_RECOVERY_WORK_MULTIPLIER: f64 = 25.0;

#[derive(Parser)]
#[command(name = "loupe", about = "Explainable filtered vector search")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build a durable collection from an embedded JSONL corpus.
    Build {
        #[arg(long)]
        base: String,
        #[arg(long)]
        dir: String,
    },
    /// Explain a filtered search: the chosen plan, why, and the counterfactual.
    Explain {
        #[arg(long)]
        dir: String,
        #[arg(long)]
        query: String,
        #[arg(long, default_value_t = 0)]
        index: usize,
        #[arg(long)]
        filter: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
    },
    /// Replay a workload, log what the planner did, and append it to a log file.
    Replay {
        #[arg(long)]
        dir: String,
        #[arg(long)]
        query: String,
        #[arg(long)]
        filter: String,
        #[arg(long)]
        out: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value_t = false)]
        compare_planners: bool,
        #[arg(long)]
        planner_summary_csv: Option<String>,
        #[arg(long)]
        feedback_log: Option<String>,
    },
    /// Read a query log and print the advisor's recommendations.
    Advise {
        #[arg(long)]
        log: String,
    },
    /// Delete a vector by id from a durable collection.
    Delete {
        #[arg(long)]
        dir: String,
        #[arg(long)]
        id: u32,
    },
    /// Compact a collection, dropping deleted vectors and reclaiming their space.
    Compact {
        #[arg(long)]
        dir: String,
    },
}

#[derive(Deserialize)]
struct Record {
    embedding: Vec<f32>,
    category: String,
    year: i64,
    doc_type: String,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    dim: usize,
    metric: Metric,
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Build { base, dir } => build(&base, &dir),
        Command::Explain {
            dir,
            query,
            index,
            filter,
            k,
        } => explain(&dir, &query, index, &filter, k),
        Command::Replay {
            dir,
            query,
            filter,
            out,
            k,
            compare_planners,
            planner_summary_csv,
            feedback_log,
        } => replay(
            &dir,
            &query,
            &filter,
            &out,
            k,
            compare_planners,
            planner_summary_csv.as_deref(),
            feedback_log.as_deref(),
        ),
        Command::Advise { log } => advise_log(&log),
        Command::Delete { dir, id } => delete(&dir, id),
        Command::Compact { dir } => compact(&dir),
    };
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn build(base: &str, dir: &str) -> io::Result<()> {
    let records = read_records(base)?;
    let dim = records.first().map_or(0, |r| r.embedding.len());
    let metric = Metric::Cosine;

    let mut collection = Collection::open(dir, dim, metric)?;
    for record in &records {
        collection.insert(
            &record.embedding,
            &[
                ("category", Value::Str(record.category.clone())),
                ("year", Value::Int(record.year)),
                ("doc_type", Value::Str(record.doc_type.clone())),
            ],
        );
    }

    let manifest = Manifest { dim, metric };
    std::fs::write(
        manifest_path(dir),
        serde_json::to_string(&manifest).map_err(io::Error::other)?,
    )?;
    println!("built {} records into {dir}", records.len());
    Ok(())
}

fn explain(dir: &str, query: &str, index: usize, filter: &str, k: usize) -> io::Result<()> {
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(manifest_path(dir))?)
        .map_err(io::Error::other)?;
    let collection = Collection::open(dir, manifest.dim, manifest.metric)?;

    let queries = read_records(query)?;
    let query_vector = &queries
        .get(index)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "query index out of range"))?
        .embedding;
    let expr = parse_filter(filter).map_err(io::Error::other)?;

    let (results, report) = collection.explain_search(query_vector, &expr, k);
    println!("filter: {filter}");
    println!();
    print_explain(&report);
    println!();
    let ids: Vec<u32> = results.iter().map(|n| n.id).collect();
    println!("returned ids: {ids:?}");
    Ok(())
}

fn replay(
    dir: &str,
    query: &str,
    filter: &str,
    out: &str,
    k: usize,
    compare_planners: bool,
    planner_summary_csv: Option<&str>,
    feedback_log: Option<&str>,
) -> io::Result<()> {
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(manifest_path(dir))?)
        .map_err(io::Error::other)?;
    let mut collection = Collection::open(dir, manifest.dim, manifest.metric)?;
    let queries = read_records(query)?;
    let expr = parse_filter(filter).map_err(io::Error::other)?;
    if let Some(path) = feedback_log {
        if let Some(rate) = historical_miss_rate_from_log(path, filter)? {
            collection.set_historical_miss_rate(rate);
            println!("historical miss rate: {rate:.3} from {path}");
        } else {
            println!("historical miss rate: no recall_smoke rows for filter {filter:?}");
        }
    }
    let planner_config = PlannerConfig::default();
    let mut planner_summary = if compare_planners || planner_summary_csv.is_some() {
        Some(ReplayPlannerSummary::default())
    } else {
        None
    };

    let mut file = OpenOptions::new().create(true).append(true).open(out)?;
    let mut plan_counts: HashMap<String, usize> = HashMap::new();
    let mut total_latency = 0u64;
    for record in &queries {
        let (_, log) = collection.search_logged(&record.embedding, &expr, filter, k);
        *plan_counts
            .entry(log.plan.clone().unwrap_or_default())
            .or_default() += 1;
        total_latency += log.latency_micros;
        writeln!(
            file,
            "{}",
            serde_json::to_string(&log).map_err(io::Error::other)?
        )?;
        if let Some(summary) = &mut planner_summary {
            let (_, explain) = collection.explain_search(&record.embedding, &expr, k);
            summary.observe(&explain, &planner_config)?;
        }
    }

    let count = queries.len().max(1) as u64;
    println!("replayed {} queries with filter {filter:?}", queries.len());
    println!("avg latency: {} us", total_latency / count);
    println!("plans chosen: {plan_counts:?}");
    if let Some(summary) = &planner_summary {
        print_replay_planner_summary(summary);
        if let Some(path) = planner_summary_csv {
            write_replay_planner_summary_csv(path, summary)?;
            println!("planner summary CSV: {path}");
        }
    }
    println!("log appended to {out}");
    Ok(())
}

fn historical_miss_rate_from_log(path: &str, filter: &str) -> io::Result<Option<f32>> {
    let content = std::fs::read_to_string(path)?;
    let mut miss_sum = 0.0f32;
    let mut count = 0usize;

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let record: QueryLogRecord = serde_json::from_str(line).map_err(io::Error::other)?;
        if record.filter.as_deref() != Some(filter) {
            continue;
        }
        if let Some(recall) = record.recall_smoke {
            miss_sum += 1.0 - recall.clamp(0.0, 1.0);
            count += 1;
        }
    }

    if count == 0 {
        return Ok(None);
    }

    Ok(Some(miss_sum / count as f32))
}

#[derive(Default)]
struct ReplayPlannerSummary {
    rule_ladder: ReplayPlannerStats,
    cost_model: ReplayPlannerStats,
    query_count: usize,
}

impl ReplayPlannerSummary {
    fn observe(&mut self, explain: &ExplainSearch, config: &PlannerConfig) -> io::Result<()> {
        let reports = all_plan_reports(explain);
        let rule_ladder_name = rule_ladder_plan_name(explain, config);
        let rule_ladder_report = find_plan_report(explain, &rule_ladder_name)?;
        let cost_model_report = &explain.chosen_plan;

        self.rule_ladder.observe(
            rule_ladder_report,
            report_has_regret(rule_ladder_report, &reports),
        );
        self.cost_model.observe(
            cost_model_report,
            report_has_regret(cost_model_report, &reports),
        );
        self.query_count += 1;
        Ok(())
    }
}

#[derive(Default)]
struct ReplayPlannerStats {
    recall_sum: f64,
    distance_sum: u64,
    regretted: usize,
    plan_counts: HashMap<String, usize>,
}

impl ReplayPlannerStats {
    fn observe(&mut self, report: &PlanReport, regretted: bool) {
        self.recall_sum += f64::from(report.recall_proxy);
        self.distance_sum += report.distance_computations;
        if regretted {
            self.regretted += 1;
        }
        *self.plan_counts.entry(report.strategy.clone()).or_default() += 1;
    }

    fn mean_recall(&self, count: usize) -> f64 {
        self.recall_sum / count.max(1) as f64
    }

    fn mean_distance_computations(&self, count: usize) -> f64 {
        self.distance_sum as f64 / count.max(1) as f64
    }

    fn regret(&self, count: usize) -> f64 {
        self.regretted as f64 / count.max(1) as f64
    }
}

fn all_plan_reports(explain: &ExplainSearch) -> Vec<&PlanReport> {
    let mut reports = Vec::with_capacity(explain.counterfactual_plans.len() + 1);
    reports.push(&explain.chosen_plan);
    reports.extend(explain.counterfactual_plans.iter());
    reports
}

fn find_plan_report<'a>(explain: &'a ExplainSearch, strategy: &str) -> io::Result<&'a PlanReport> {
    all_plan_reports(explain)
        .into_iter()
        .find(|report| report.strategy == strategy)
        .ok_or_else(|| {
            io::Error::other(format!(
                "missing counterfactual plan for rule-ladder strategy {strategy}"
            ))
        })
}

fn report_has_regret(target: &PlanReport, reports: &[&PlanReport]) -> bool {
    reports
        .iter()
        .any(|candidate| is_pareto_regret(target, candidate) || is_recall_regret(target, candidate))
}

fn is_pareto_regret(target: &PlanReport, candidate: &PlanReport) -> bool {
    candidate.recall_proxy >= target.recall_proxy
        && candidate.distance_computations <= target.distance_computations
        && (candidate.recall_proxy > target.recall_proxy
            || candidate.distance_computations < target.distance_computations)
}

fn is_recall_regret(target: &PlanReport, candidate: &PlanReport) -> bool {
    if target.recall_proxy >= SATISFIED_RECALL {
        return false;
    }
    if candidate.recall_proxy < SATISFIED_RECALL {
        return false;
    }
    if candidate.recall_proxy < target.recall_proxy + RECALL_REGRET_GAP {
        return false;
    }

    let target_work = target.distance_computations.max(1) as f64;
    let candidate_work = candidate.distance_computations as f64;
    candidate_work <= target_work * MAX_RECALL_RECOVERY_WORK_MULTIPLIER
}

fn rule_ladder_plan_name(explain: &ExplainSearch, config: &PlannerConfig) -> String {
    let summary = &explain.query_summary;
    if summary.matching_count as usize <= config.flat_threshold.max(summary.k) {
        return "Flat".to_string();
    }
    if summary.correlation == "anti_correlated" {
        return "Acorn+AdaptiveEf".to_string();
    }
    if summary.selectivity >= config.broad_selectivity {
        return "PostFilter+AdaptiveEf".to_string();
    }
    if summary.selectivity < config.strict_selectivity {
        return "Acorn+AdaptiveEf".to_string();
    }

    "InTraversal+AdaptiveEf".to_string()
}

fn print_replay_planner_summary(summary: &ReplayPlannerSummary) {
    println!("planner comparison:");
    println!(
        "{:<12} {:>11} {:>16} {:>8}  plan mix",
        "planner", "mean recall", "mean dist/query", "regret"
    );
    print_replay_planner_row("RuleLadder", &summary.rule_ladder, summary.query_count);
    print_replay_planner_row("CostModel", &summary.cost_model, summary.query_count);
    println!(
        "regret_delta: {:.3}",
        summary.rule_ladder.regret(summary.query_count)
            - summary.cost_model.regret(summary.query_count)
    );
}

fn print_replay_planner_row(name: &str, stats: &ReplayPlannerStats, count: usize) {
    println!(
        "{:<12} {:>11.3} {:>16.0} {:>8.3}  {:?}",
        name,
        stats.mean_recall(count),
        stats.mean_distance_computations(count),
        stats.regret(count),
        stats.plan_counts
    );
}

fn write_replay_planner_summary_csv(path: &str, summary: &ReplayPlannerSummary) -> io::Result<()> {
    std::fs::write(path, replay_planner_summary_csv(summary))
}

fn replay_planner_summary_csv(summary: &ReplayPlannerSummary) -> String {
    let count = summary.query_count;
    let rule_regret = summary.rule_ladder.regret(count);
    let cost_regret = summary.cost_model.regret(count);
    let regret_delta = rule_regret - cost_regret;
    let mut csv =
        String::from("planner,mean_recall,mean_distance_computations,regret,regret_delta\n");

    csv.push_str(&format!(
        "RuleLadder,{:.6},{:.3},{:.6},{:.6}\n",
        summary.rule_ladder.mean_recall(count),
        summary.rule_ladder.mean_distance_computations(count),
        rule_regret,
        regret_delta
    ));
    csv.push_str(&format!(
        "CostModel,{:.6},{:.3},{:.6},{:.6}\n",
        summary.cost_model.mean_recall(count),
        summary.cost_model.mean_distance_computations(count),
        cost_regret,
        regret_delta
    ));

    csv
}

fn advise_log(log: &str) -> io::Result<()> {
    let content = std::fs::read_to_string(log)?;
    let mut records: Vec<QueryLogRecord> = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        records.push(serde_json::from_str(line).map_err(io::Error::other)?);
    }

    let recommendations = advise(&records);
    if recommendations.is_empty() {
        println!("no recommendations from {} queries", records.len());
        return Ok(());
    }
    println!("recommendations from {} queries:", records.len());
    for recommendation in &recommendations {
        println!(
            "  [{}] {}: {}",
            recommendation.category, recommendation.filter, recommendation.message
        );
    }
    Ok(())
}

fn delete(dir: &str, id: u32) -> io::Result<()> {
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(manifest_path(dir))?)
        .map_err(io::Error::other)?;
    let mut collection = Collection::open(dir, manifest.dim, manifest.metric)?;
    if collection.delete(id) {
        println!(
            "deleted {id}: {} live, {} tombstoned",
            collection.live_count(),
            collection.deleted_count()
        );
    } else {
        println!("nothing to delete for id {id} (unknown or already deleted)");
    }
    Ok(())
}

fn compact(dir: &str) -> io::Result<()> {
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(manifest_path(dir))?)
        .map_err(io::Error::other)?;
    let mut collection = Collection::open(dir, manifest.dim, manifest.metric)?;
    let report = collection.compact_with_report()?;
    println!(
        "compacted: removed {} deleted vectors, {} remain",
        report.removed, report.after.live_count
    );
    println!(
        "before: total={} live={} deleted={}",
        report.before.total_count, report.before.live_count, report.before.deleted_count
    );
    println!(
        "after:  total={} live={} deleted={}",
        report.after.total_count, report.after.live_count, report.after.deleted_count
    );
    Ok(())
}

fn print_explain(report: &ExplainSearch) {
    let summary = &report.query_summary;
    println!("query summary");
    println!("  k:           {}", summary.k);
    println!(
        "  matches:     {} ({:.2}% of the collection)",
        summary.matching_count,
        summary.selectivity * 100.0
    );
    println!("  correlation: {}", summary.correlation);
    println!();
    println!("chosen plan:   {}", report.chosen_plan.strategy);
    println!();
    println!(
        "{:<14} {:>7} {:>12} {:>9} {:>14}",
        "plan", "chosen", "dist/query", "returned", "recall (proxy)"
    );
    print_plan_row(&report.chosen_plan);
    for plan in &report.counterfactual_plans {
        print_plan_row(plan);
    }

    if !report.risk_flags.is_empty() {
        println!();
        println!("risk flags:    {}", report.risk_flags.join(", "));
    }
    if !report.recommendations.is_empty() {
        println!("recommendations:");
        for advice in &report.recommendations {
            println!("  - {advice}");
        }
    }
}

fn print_plan_row(plan: &PlanReport) {
    println!(
        "{:<14} {:>7} {:>12} {:>9} {:>14.3}",
        plan.strategy,
        if plan.chosen { "yes" } else { "" },
        plan.distance_computations,
        plan.returned,
        plan.recall_proxy
    );
}

fn read_records(path: &str) -> io::Result<Vec<Record>> {
    let content = std::fs::read_to_string(path)?;
    let mut records = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        records.push(serde_json::from_str(line).map_err(io::Error::other)?);
    }
    Ok(records)
}

fn manifest_path(dir: &str) -> String {
    format!("{dir}/manifest.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use loupe_engine::QuerySummary;

    fn plan(strategy: &str, distance_computations: u64, recall_proxy: f32) -> PlanReport {
        PlanReport {
            strategy: strategy.to_string(),
            chosen: false,
            distance_computations,
            returned: 10,
            recall_proxy,
        }
    }

    fn explain(
        matching_count: u64,
        selectivity: f32,
        correlation: &str,
        chosen_plan: PlanReport,
        counterfactual_plans: Vec<PlanReport>,
    ) -> ExplainSearch {
        ExplainSearch {
            query_summary: QuerySummary {
                k: 10,
                matching_count,
                selectivity,
                correlation: correlation.to_string(),
            },
            chosen_plan,
            counterfactual_plans,
            trace_metrics: Default::default(),
            risk_flags: Vec::new(),
            recommendations: Vec::new(),
        }
    }

    fn query_log(filter: &str, recall_smoke: Option<f32>) -> QueryLogRecord {
        QueryLogRecord {
            query_id: 1,
            k: 10,
            latency_micros: 10,
            returned: vec![1, 2, 3],
            metrics: Default::default(),
            filter: Some(filter.to_string()),
            plan: Some("PostFilter+AdaptiveEf".to_string()),
            recall_smoke,
        }
    }

    #[test]
    fn rule_ladder_plan_name_matches_m4_choices() {
        let config = PlannerConfig::default();
        let broad = explain(
            60_000,
            0.6,
            "neutral",
            plan("PostFilter+AdaptiveEf", 100, 1.0),
            Vec::new(),
        );
        let anti = explain(
            60_000,
            0.6,
            "anti_correlated",
            plan("Acorn+AdaptiveEf", 100, 1.0),
            Vec::new(),
        );
        let strict = explain(
            3_000,
            0.003,
            "neutral",
            plan("Acorn+AdaptiveEf", 100, 1.0),
            Vec::new(),
        );
        let small = explain(
            100,
            0.001,
            "anti_correlated",
            plan("Flat", 100, 1.0),
            Vec::new(),
        );

        assert_eq!(
            rule_ladder_plan_name(&broad, &config),
            "PostFilter+AdaptiveEf"
        );
        assert_eq!(rule_ladder_plan_name(&anti, &config), "Acorn+AdaptiveEf");
        assert_eq!(rule_ladder_plan_name(&strict, &config), "Acorn+AdaptiveEf");
        assert_eq!(rule_ladder_plan_name(&small, &config), "Flat");
    }

    #[test]
    fn replay_summary_scores_rule_ladder_against_cost_model() {
        let config = PlannerConfig::default();
        let chosen = plan("PostFilter+AdaptiveEf", 100, 0.5);
        let reports = vec![
            plan("Flat", 90, 0.5),
            plan("InTraversal+AdaptiveEf", 500, 1.0),
            plan("Acorn+AdaptiveEf", 300, 1.0),
        ];
        let explain = explain(60_000, 0.6, "anti_correlated", chosen, reports);
        let mut summary = ReplayPlannerSummary::default();

        summary.observe(&explain, &config).unwrap();
        let csv = replay_planner_summary_csv(&summary);

        assert!(csv.contains("RuleLadder,1.000000,300.000,0.000000,-1.000000"));
        assert!(csv.contains("CostModel,0.500000,100.000,1.000000,-1.000000"));
    }

    #[test]
    fn replay_summary_counts_recall_recovery_as_regret() {
        let config = PlannerConfig::default();
        let chosen = plan("Flat", 5_000, 1.0);
        let reports = vec![
            plan("PostFilter+AdaptiveEf", 250, 0.0),
            plan("InTraversal+AdaptiveEf", 300, 0.0),
            plan("Acorn+AdaptiveEf", 260, 0.0),
        ];
        let explain = explain(5_000, 0.5, "anti_correlated", chosen, reports);
        let mut summary = ReplayPlannerSummary::default();

        summary.observe(&explain, &config).unwrap();
        let csv = replay_planner_summary_csv(&summary);

        assert!(csv.contains("RuleLadder,0.000000,260.000,1.000000,1.000000"));
        assert!(csv.contains("CostModel,1.000000,5000.000,0.000000,1.000000"));
    }

    #[test]
    fn historical_miss_rate_reads_matching_recall_smoke() {
        let path =
            std::env::temp_dir().join(format!("loupe-feedback-{}.jsonl", std::process::id()));
        let records = [
            query_log("cat=math", Some(0.8)),
            query_log("cat=math", Some(0.6)),
            query_log("cat=cs", Some(0.1)),
            query_log("cat=math", None),
        ];
        let content = records
            .iter()
            .map(|record| serde_json::to_string(record).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, content).unwrap();

        let rate = historical_miss_rate_from_log(path.to_str().unwrap(), "cat=math")
            .unwrap()
            .unwrap();
        std::fs::remove_file(path).unwrap();

        assert!((rate - 0.3).abs() < 0.001);
    }
}
