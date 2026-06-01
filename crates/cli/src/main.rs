//! The `loupe` command line: build a durable collection from an embedded corpus,
//! then run `explain` to see the chosen plan and what the rejected plans would
//! have cost.

use std::io;

use clap::{Parser, Subcommand};
use loupe_distance::Metric;
use loupe_engine::{Collection, ExplainSearch, PlanReport, Value, parse_filter};
use serde::{Deserialize, Serialize};

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
