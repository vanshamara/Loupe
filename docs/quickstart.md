# Loupe Quickstart

This page keeps the runnable commands out of the README.

## Requirements

- Rust stable.
- Python 3.11 or newer for benchmarks.
- Docker only for Qdrant, Weaviate, and pgvector benchmark runs.

## Build

```bash
cargo build --all
cargo test --all
```

## CLI Workflow

The canonical workflow creates a small skewed dataset in a temp directory, builds
a collection, explains a hard filtered query, replays the workload, writes an
advisor report, exports metrics, deletes rows, compacts, and checks stats.

```bash
LOUPE_WORKFLOW_DIR=/tmp/loupe-workflow scripts/run-canonical-workflow.sh
```

Useful files from that run:

- `/tmp/loupe-workflow/explain-before.txt`
- `/tmp/loupe-workflow/planner-summary.csv`
- `/tmp/loupe-workflow/retrieval-health.md`
- `/tmp/loupe-workflow/metrics.prom`
- `/tmp/loupe-workflow/stats-after-compact.txt`

Install the CLI if you want to run commands directly:

```bash
cargo install --path crates/cli
```

Then build and explain manually:

```bash
loupe build --base /tmp/loupe-workflow/base.jsonl --dir /tmp/loupe-demo
loupe explain --dir /tmp/loupe-demo --query /tmp/loupe-workflow/query.jsonl \
  --index 0 --filter "category=target"
```

Replay the workload and compare planners:

```bash
loupe replay --dir /tmp/loupe-demo --query /tmp/loupe-workflow/query.jsonl \
  --filter "category=target" --out /tmp/loupe-query-log.jsonl \
  --compare-planners --planner-summary-csv /tmp/loupe-planners.csv
```

Create an advisor report:

```bash
loupe advise --log /tmp/loupe-query-log.jsonl \
  --markdown-out /tmp/loupe-health.md \
  --planner-summary-csv /tmp/loupe-planners.csv
```

Export Prometheus text:

```bash
loupe metrics --log /tmp/loupe-query-log.jsonl \
  --dir /tmp/loupe-demo \
  --planner-summary-csv /tmp/loupe-planners.csv \
  --out /tmp/loupe-metrics.prom
```

Check collection health:

```bash
loupe stats --dir /tmp/loupe-demo
```

## Server

Start the local gRPC server:

```bash
cargo run -p loupe-server --bin loupe-server
```

It listens on `127.0.0.1:50051`.

Run the server smoke test:

```bash
cargo test -p loupe-server --test grpc
```

The test creates a collection, upserts vectors, searches with a filter, and runs
EXPLAIN through the service layer.

## Small Benchmark

Download `siftsmall`:

```bash
bash scripts/download-datasets.sh siftsmall
```

Run Loupe's matrix benchmark:

```bash
cargo run --release -p loupe-bench --bin loupe-bench -- \
  datasets/siftsmall/siftsmall_base.fvecs \
  datasets/siftsmall/siftsmall_query.fvecs \
  datasets/siftsmall/siftsmall_groundtruth.ivecs \
  --matrix-csv /tmp/loupe-matrix.csv \
  --planner-summary-csv /tmp/loupe-planner-summary.csv \
  --plot-dir /tmp/loupe-plots
```

The matrix CSV records selectivity, correlation mode, strategy chosen, recall,
distance work, latency, and returned count.

## Packaged Benchmark Suite

Set up Python dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r scripts/requirements-faiss.txt \
  -r scripts/requirements-competitors.txt
```

Run all engines on `siftsmall`:

```bash
python bench/run_benchmark_suite.py \
  --dataset-name siftsmall \
  --base datasets/siftsmall/siftsmall_base.fvecs \
  --query datasets/siftsmall/siftsmall_query.fvecs \
  --groundtruth datasets/siftsmall/siftsmall_groundtruth.ivecs \
  --run-id siftsmall-smoke \
  --engines all
```

For SIFT1M, use `scripts/download-datasets.sh sift` and point the command at the
`datasets/sift/sift_*` files. Use `--engines loupe,faiss` for a no-Docker run.

Long runs print progress and write partial CSVs under
`/tmp/loupe-benchmark-suite/<run-id>/`.

## Graph Explorer

Start a static server:

```bash
python -m http.server 8080 --directory viz
```

Open `http://localhost:8080`. The page loads `viz/sample-trace.json` by default
and can load another trace JSON from disk.
