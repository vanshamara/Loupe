use std::env;
use std::time::Instant;

use loupe_bench::arxiv::{build_workload, read_jsonl};
use loupe_bench::matrix::{MatrixConfig, print_filter_matrix, run_metadata_matrix};
use loupe_distance::Metric;
use loupe_filter::{Expr, Value};
use loupe_hnsw::{HnswConfig, HnswIndex};

const K: usize = 10;

/// Loads the embedded arXiv corpus, builds a cosine index plus a metadata store,
/// and runs the filtered strategies against real metadata filters. Embeddings are
/// normalized, so cosine is the right metric.
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: arxiv <base.jsonl> <query.jsonl>");
        std::process::exit(1);
    }

    let base_records = read_jsonl(&args[1]).expect("failed to read base records");
    let query_records = read_jsonl(&args[2]).expect("failed to read query records");
    let (vectors, metadata) = build_workload(&base_records);
    let queries: Vec<Vec<f32>> = query_records.into_iter().map(|r| r.embedding).collect();
    let dim = vectors[0].len();

    println!("base vectors: {}", vectors.len());
    println!("queries:      {}", queries.len());
    println!("dimension:    {dim}");
    println!();

    let build_start = Instant::now();
    let mut index = HnswIndex::new(dim, Metric::Cosine, HnswConfig::default());
    for vector in &vectors {
        index.insert(vector);
    }
    println!("build time:   {:.2}s", build_start.elapsed().as_secs_f64());
    println!();

    let filters = vec![
        (
            "category=cs.LG".to_string(),
            Expr::Eq("category".into(), Value::Str("cs.LG".into())),
        ),
        (
            "category=math.PR".to_string(),
            Expr::Eq("category".into(), Value::Str("math.PR".into())),
        ),
        (
            "year>=2024".to_string(),
            Expr::Range("year".into(), 2024, 9999),
        ),
        (
            "cs.LG AND year>=2024".to_string(),
            Expr::And(
                Box::new(Expr::Eq("category".into(), Value::Str("cs.LG".into()))),
                Box::new(Expr::Range("year".into(), 2024, 9999)),
            ),
        ),
        (
            "doc_type=cross".to_string(),
            Expr::Eq("doc_type".into(), Value::Str("cross".into())),
        ),
    ];

    let config = MatrixConfig {
        k: K,
        ef: 64,
        max_ef: 512,
        seed: 1,
    };
    let results = run_metadata_matrix(&index, &metadata, &filters, &queries, &config);
    println!("=== arXiv filtered search (cosine) ===");
    print_filter_matrix(&results);
}
