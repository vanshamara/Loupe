use std::env;
use std::time::Instant;

use loupe_bench::compaction::{print_compaction_probe, run_compaction_probe};
use loupe_bench::matrix::{
    MatrixConfig, print_matrix, print_planner_summary, run_matrix, summarize_planner,
    write_matrix_csv,
};
use loupe_bench::plots::write_loupe_plots;
use loupe_bench::{read_fvecs, read_ivecs, recall_at_k};
use loupe_distance::Metric;
use loupe_hnsw::{HnswConfig, HnswIndex, brute_force_search};
use loupe_storage::{InMemoryVectorStore, VectorId};

const K: usize = 10;

/// Loads a SIFT-style dataset, builds the index under L2 and cosine, runs the
/// queries, and prints recall@10 plus basic timing. Cosine ground truth is
/// computed with the brute-force baseline, since the dataset ships L2 truth only.
fn main() {
    let args = parse_args();

    let base = read_fvecs(&args.base_path).expect("failed to read base vectors");
    let queries = read_fvecs(&args.query_path).expect("failed to read query vectors");
    let l2_truth = read_ivecs(&args.groundtruth_path).expect("failed to read ground truth");
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

    let cosine_truth = brute_force_truth(&base, &queries, dim, Metric::Cosine);
    let cosine_index = build_index(&base, dim, Metric::Cosine);
    let (cosine_results, _) = run_queries(&cosine_index, &queries);
    let cosine_recall = recall_at_k(&cosine_results, &cosine_truth, K);

    let avg_distances = distance_computations as f64 / queries.len() as f64;

    println!("build time:        {build_secs:.2}s");
    println!("queries/sec:       {qps:.0}");
    println!("avg distances/query: {avg_distances:.0}");
    println!("recall@{K} (L2):     {l2_recall:.4}");
    println!("recall@{K} (cosine): {cosine_recall:.4}");

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

    let reopened = HnswIndex::open(vector_path, graph_path).expect("failed to open sealed index");
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

    println!();
    println!("=== compaction probe (synthetic L2) ===");
    let compaction = run_compaction_probe(1000, 2);
    print_compaction_probe(&compaction);
}

struct BenchArgs {
    base_path: String,
    query_path: String,
    groundtruth_path: String,
    matrix_csv_path: Option<String>,
    plot_dir: Option<String>,
}

fn parse_args() -> BenchArgs {
    let mut positional = Vec::new();
    let mut matrix_csv_path = None;
    let mut plot_dir = None;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        if arg == "--matrix-csv" {
            let path = args.next().unwrap_or_else(|| {
                eprintln!("--matrix-csv requires a path");
                std::process::exit(1);
            });
            matrix_csv_path = Some(path);
        } else if arg == "--plot-dir" {
            let path = args.next().unwrap_or_else(|| {
                eprintln!("--plot-dir requires a path");
                std::process::exit(1);
            });
            plot_dir = Some(path);
        } else {
            positional.push(arg);
        }
    }

    if positional.len() != 3 {
        eprintln!(
            "usage: loupe-bench <base.fvecs> <query.fvecs> <groundtruth.ivecs> [--matrix-csv <path>] [--plot-dir <dir>]"
        );
        std::process::exit(1);
    }

    BenchArgs {
        base_path: positional[0].clone(),
        query_path: positional[1].clone(),
        groundtruth_path: positional[2].clone(),
        matrix_csv_path,
        plot_dir,
    }
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
