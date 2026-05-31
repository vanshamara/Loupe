use std::env;
use std::time::Instant;

use loupe_bench::matrix::{
    MatrixConfig, print_matrix, print_planner_summary, run_matrix, summarize_planner,
};
use loupe_bench::{read_fvecs, read_ivecs, recall_at_k};
use loupe_distance::Metric;
use loupe_hnsw::{HnswConfig, HnswIndex, brute_force_search};
use loupe_storage::{InMemoryVectorStore, VectorId};

const K: usize = 10;

/// Loads a SIFT-style dataset, builds the index under L2 and cosine, runs the
/// queries, and prints recall@10 plus basic timing. Cosine ground truth is
/// computed with the brute-force baseline, since the dataset ships L2 truth only.
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: loupe-bench <base.fvecs> <query.fvecs> <groundtruth.ivecs>");
        std::process::exit(1);
    }

    let base = read_fvecs(&args[1]).expect("failed to read base vectors");
    let queries = read_fvecs(&args[2]).expect("failed to read query vectors");
    let l2_truth = read_ivecs(&args[3]).expect("failed to read ground truth");
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
