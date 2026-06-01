use std::env;
use std::time::Instant;

use loupe_bench::compaction::{print_compaction_probe, run_compaction_probe};
use loupe_bench::matrix::{
    MatrixConfig, print_matrix, print_planner_summary, run_matrix, summarize_planner,
    write_matrix_csv, write_planner_summary_csv,
};
use loupe_bench::plots::write_loupe_plots;
use loupe_bench::{read_fvecs, read_ivecs, recall_at_k};
use loupe_distance::Metric;
use loupe_hnsw::{HnswConfig, HnswIndex, brute_force_search};
use loupe_storage::{InMemoryVectorStore, VectorId};

const K: usize = 10;

/// Loads a SIFT-style dataset, builds the L2 index, runs recall and matrix
/// checks, and optionally runs the extra local diagnostics.
fn main() {
    let args = parse_args();

    let base = read_fvecs(&args.base_path).expect("failed to read base vectors");
    let mut queries = read_fvecs(&args.query_path).expect("failed to read query vectors");
    let mut l2_truth = read_ivecs(&args.groundtruth_path).expect("failed to read ground truth");
    if queries.len() != l2_truth.len() {
        eprintln!("query and ground-truth row counts differ");
        std::process::exit(1);
    }
    if let Some(limit) = args.query_limit {
        truncate_query_rows(&mut queries, &mut l2_truth, limit);
    }
    let dim = base[0].len();

    println!("base vectors: {}", base.len());
    println!("queries:      {}", queries.len());
    println!("dimension:    {dim}");
    println!();

    let build_start = Instant::now();
    let l2_index = build_index(&base, dim, Metric::L2);
    let build_secs = build_start.elapsed().as_secs_f64();

    let query_start = Instant::now();
    let (l2_results, distance_computations) = run_queries(&l2_index, &queries);
    let qps = queries.len() as f64 / query_start.elapsed().as_secs_f64();
    let l2_recall = recall_at_k(&l2_results, &l2_truth, K);

    let avg_distances = distance_computations as f64 / queries.len() as f64;

    println!("build time:        {build_secs:.2}s");
    println!("queries/sec:       {qps:.0}");
    println!("avg distances/query: {avg_distances:.0}");
    println!("recall@{K} (L2):     {l2_recall:.4}");
    if args.skip_cosine {
        println!("recall@{K} (cosine): skipped");
    } else {
        let cosine_truth = brute_force_truth(&base, &queries, dim, Metric::Cosine);
        let cosine_index = build_index(&base, dim, Metric::Cosine);
        let (cosine_results, _) = run_queries(&cosine_index, &queries);
        let cosine_recall = recall_at_k(&cosine_results, &cosine_truth, K);
        println!("recall@{K} (cosine): {cosine_recall:.4}");
    }

    println!();
    println!("=== filtered search matrix (L2) ===");
    let matrix_config = MatrixConfig {
        k: K,
        ef: 64,
        max_ef: 512,
        seed: 1,
    };
    let cells = run_matrix(&l2_index, &base, &queries, Metric::L2, &matrix_config);
    print_matrix(&cells);
    if let Some(path) = &args.matrix_csv_path {
        write_matrix_csv(path, &cells).expect("failed to write matrix CSV");
        println!("matrix CSV: {path}");
    }
    if let Some(dir) = &args.plot_dir {
        let files = write_loupe_plots(dir, &cells).expect("failed to write matrix plots");
        println!("plot: {}", files.fixed_vs_planner.display());
        println!("plot: {}", files.hard_filter_cells.display());
        println!("plot: {}", files.acorn_wins.display());
        println!("plot: {}", files.flat_wins.display());
    }

    println!();
    println!("=== planner vs fixed strategies (over all cells) ===");
    let planner = loupe_planner::PlannerConfig::default();
    let summary = summarize_planner(
        &l2_index,
        &base,
        &queries,
        Metric::L2,
        &matrix_config,
        &planner,
        32,
    );
    print_planner_summary(&summary);
    if let Some(path) = &args.planner_summary_csv_path {
        write_planner_summary_csv(path, &summary).expect("failed to write planner summary CSV");
        println!("planner summary CSV: {path}");
    }

    if args.skip_mmap {
        println!();
        println!("=== memory-mapped index (L2) ===");
        println!("skipped");
    } else {
        println!();
        println!("=== memory-mapped index (L2) ===");
        let temp = std::env::temp_dir();
        let vector_path = temp.join("loupe_sift.vec");
        let graph_path = temp.join("loupe_sift.graph");
        let vector_path = vector_path.to_str().unwrap();
        let graph_path = graph_path.to_str().unwrap();
        l2_index
            .seal(vector_path, graph_path)
            .expect("failed to seal index");
        let vector_bytes = std::fs::metadata(vector_path).map(|m| m.len()).unwrap_or(0);
        let graph_bytes = std::fs::metadata(graph_path).map(|m| m.len()).unwrap_or(0);

        let reopened =
            HnswIndex::open(vector_path, graph_path).expect("failed to open sealed index");
        let disk_start = Instant::now();
        let (disk_results, _) = run_queries(&reopened, &queries);
        let disk_qps = queries.len() as f64 / disk_start.elapsed().as_secs_f64();
        let disk_recall = recall_at_k(&disk_results, &l2_truth, K);

        println!("vector segment:    {:.1} MB", vector_bytes as f64 / 1e6);
        println!("graph segment:     {:.1} MB", graph_bytes as f64 / 1e6);
        println!("disk queries/sec:  {disk_qps:.0}");
        println!("recall@{K} (disk):   {disk_recall:.4}");

        let _ = std::fs::remove_file(vector_path);
        let _ = std::fs::remove_file(graph_path);
    }

    if args.skip_compaction {
        println!();
        println!("=== compaction probe (synthetic L2) ===");
        println!("skipped");
    } else {
        println!();
        println!("=== compaction probe (synthetic L2) ===");
        let compaction = run_compaction_probe(1000, 2);
        print_compaction_probe(&compaction);
    }
}

struct BenchArgs {
    base_path: String,
    query_path: String,
    groundtruth_path: String,
    matrix_csv_path: Option<String>,
    planner_summary_csv_path: Option<String>,
    plot_dir: Option<String>,
    query_limit: Option<usize>,
    skip_cosine: bool,
    skip_mmap: bool,
    skip_compaction: bool,
}

fn parse_args() -> BenchArgs {
    parse_args_from(env::args().skip(1))
}

fn parse_args_from(args: impl IntoIterator<Item = String>) -> BenchArgs {
    let mut positional = Vec::new();
    let mut matrix_csv_path = None;
    let mut planner_summary_csv_path = None;
    let mut plot_dir = None;
    let mut query_limit = None;
    let mut skip_cosine = false;
    let mut skip_mmap = false;
    let mut skip_compaction = false;
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        if arg == "--matrix-csv" {
            let path = args.next().unwrap_or_else(|| {
                eprintln!("--matrix-csv requires a path");
                std::process::exit(1);
            });
            matrix_csv_path = Some(path);
        } else if arg == "--planner-summary-csv" {
            let path = args.next().unwrap_or_else(|| {
                eprintln!("--planner-summary-csv requires a path");
                std::process::exit(1);
            });
            planner_summary_csv_path = Some(path);
        } else if arg == "--plot-dir" {
            let path = args.next().unwrap_or_else(|| {
                eprintln!("--plot-dir requires a path");
                std::process::exit(1);
            });
            plot_dir = Some(path);
        } else if arg == "--query-limit" {
            let limit = args.next().unwrap_or_else(|| {
                eprintln!("--query-limit requires a positive integer");
                std::process::exit(1);
            });
            query_limit = Some(parse_positive_usize("--query-limit", &limit));
        } else if arg == "--skip-cosine" {
            skip_cosine = true;
        } else if arg == "--skip-mmap" {
            skip_mmap = true;
        } else if arg == "--skip-compaction" {
            skip_compaction = true;
        } else {
            positional.push(arg);
        }
    }

    if positional.len() != 3 {
        eprintln!(
            "usage: loupe-bench <base.fvecs> <query.fvecs> <groundtruth.ivecs> [--matrix-csv <path>] [--planner-summary-csv <path>] [--plot-dir <dir>] [--query-limit <n>] [--skip-cosine] [--skip-mmap] [--skip-compaction]"
        );
        std::process::exit(1);
    }

    BenchArgs {
        base_path: positional[0].clone(),
        query_path: positional[1].clone(),
        groundtruth_path: positional[2].clone(),
        matrix_csv_path,
        planner_summary_csv_path,
        plot_dir,
        query_limit,
        skip_cosine,
        skip_mmap,
        skip_compaction,
    }
}

fn parse_positive_usize(name: &str, value: &str) -> usize {
    match value.parse::<usize>() {
        Ok(parsed) if parsed > 0 => parsed,
        _ => {
            eprintln!("{name} requires a positive integer");
            std::process::exit(1);
        }
    }
}

fn truncate_query_rows(
    queries: &mut Vec<Vec<f32>>,
    ground_truth: &mut Vec<Vec<VectorId>>,
    limit: usize,
) {
    let capped = limit.min(queries.len());
    queries.truncate(capped);
    ground_truth.truncate(capped);
}

fn build_index(base: &[Vec<f32>], dim: usize, metric: Metric) -> HnswIndex {
    let mut index = HnswIndex::new(dim, metric, HnswConfig::default());
    for vector in base {
        index.insert(vector);
    }
    index
}

fn run_queries(index: &HnswIndex, queries: &[Vec<f32>]) -> (Vec<Vec<VectorId>>, u64) {
    let mut results = Vec::with_capacity(queries.len());
    let mut distance_computations = 0u64;
    for query in queries {
        let (neighbors, trace) = index.search(query, K);
        distance_computations += trace.distance_computations;
        results.push(neighbors.into_iter().map(|n| n.id).collect());
    }
    (results, distance_computations)
}

fn brute_force_truth(
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    dim: usize,
    metric: Metric,
) -> Vec<Vec<VectorId>> {
    let mut store = InMemoryVectorStore::new(dim);
    for vector in base {
        store.add(vector);
    }
    queries
        .iter()
        .map(|query| {
            brute_force_search(&store, metric, query, K)
                .into_iter()
                .map(|n| n.id)
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scale_control_flags() {
        let args = parse_args_from(
            [
                "base.fvecs",
                "query.fvecs",
                "truth.ivecs",
                "--matrix-csv",
                "matrix.csv",
                "--planner-summary-csv",
                "planner-summary.csv",
                "--plot-dir",
                "plots",
                "--query-limit",
                "25",
                "--skip-cosine",
                "--skip-mmap",
                "--skip-compaction",
            ]
            .into_iter()
            .map(String::from),
        );

        assert_eq!(args.base_path, "base.fvecs");
        assert_eq!(args.query_limit, Some(25));
        assert_eq!(args.matrix_csv_path, Some("matrix.csv".to_string()));
        assert_eq!(
            args.planner_summary_csv_path,
            Some("planner-summary.csv".to_string())
        );
        assert_eq!(args.plot_dir, Some("plots".to_string()));
        assert!(args.skip_cosine);
        assert!(args.skip_mmap);
        assert!(args.skip_compaction);
    }

    #[test]
    fn query_limit_truncates_queries_and_truth_together() {
        let mut queries = vec![vec![1.0], vec![2.0], vec![3.0]];
        let mut ground_truth = vec![vec![1], vec![2], vec![3]];

        truncate_query_rows(&mut queries, &mut ground_truth, 2);

        assert_eq!(queries, vec![vec![1.0], vec![2.0]]);
        assert_eq!(ground_truth, vec![vec![1], vec![2]]);
    }
}
