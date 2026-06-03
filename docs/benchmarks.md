# Loupe Benchmarks

This page records the current public benchmark proof and its limits.

## Main Claim

Loupe is built for hard filtered vector search. The current SIFT1M results
support this narrower claim:

> Loupe keeps hard-filter recall near perfect while avoiding the under-return
> behavior seen in FAISS post-filtering, and it is much faster than the measured
> Docker baselines for Qdrant, Weaviate, and pgvector on this setup.

This does not mean Loupe is the best vector database overall. FAISS remains the
raw ANN speed baseline. Qdrant, Weaviate, and pgvector are more mature production
systems.

## SIFT1M Results

Dataset: `sift1m`, 1,000,000 base vectors, first 100 query rows.

Hardware:

- Loupe and FAISS: Darwin arm64, 10 CPU cores, 16 GiB memory.
- External engines: Linux x86_64 on EC2 `c7i.8xlarge`, 32 CPU cores, 64 GiB
  memory, Docker.

| Engine | Mean recall | Min recall | Hard-filter recall | Mean latency | Under-return cells |
|---|---:|---:|---:|---:|---:|
| Loupe Planner | 0.990 | 0.966 | 1.000 | 2.06 ms | 1 |
| FAISS | 0.601 | 0.000 | 0.000 | 2.12 ms | 10 |
| Qdrant | 0.931 | 0.285 | 0.984 | 106.37 ms | 0 |
| pgvector | 0.991 | 0.901 | 0.982 | 203.45 ms | 0 |
| Weaviate | 0.959 | 0.294 | 0.982 | 754.90 ms | 0 |

Result bundles:

- Loupe and FAISS:
  `bench/results/sift1m-m5-loupe-faiss-flat-fraction-2026-06-02/summary.md`
- External engines:
  `bench/results/sift1m-external-2026-06-02/summary.md`

## Planner Result

The same Loupe SIFT1M run compares the old rule ladder with the live cost model.

| Planner | Mean recall | Mean dist/query | Regret |
|---|---:|---:|---:|
| RuleLadder | 0.949 | 115,700 | 0.314 |
| CostModel | 0.990 | 44,640 | 0.096 |

The cost model improves recall, reduces distance work, and lowers regret on this
run.

Saved M11 proof bundles:

- `bench/results/m11-replay-win-2026-06-01/summary.md`

## Caveats

The Loupe and FAISS SIFT1M run was collected locally. The Qdrant, Weaviate, and
pgvector run was collected on EC2 with Docker. That means the table is not a
perfect same-process latency comparison.

The result is still useful because the external run used a stronger CPU box, and
the latency gap is large. The honest reading is:

- Loupe wins the hard filtered-search niche in this benchmark.
- FAISS is fast for simple ANN, but under-returns hard filtered cells.
- pgvector gets close on recall, but with much higher latency here.
- Qdrant is the fastest external engine in this run, but has lower mean and min
  recall.
- Weaviate keeps good hard-filter recall but is slowest in this measured setup.

## Reproduce The Loupe And FAISS Run

```bash
bash scripts/download-datasets.sh sift

python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r scripts/requirements-faiss.txt

python bench/run_benchmark_suite.py \
  --dataset-name sift1m \
  --base datasets/sift/sift_base.fvecs \
  --query datasets/sift/sift_query.fvecs \
  --groundtruth datasets/sift/sift_groundtruth.ivecs \
  --run-id sift1m-m5-loupe-faiss \
  --engines loupe,faiss \
  --query-limit 100
```

## Reproduce The External Run

Use a CPU box with enough memory. The recorded run used EC2 `c7i.8xlarge`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r scripts/requirements-competitors.txt

python bench/run_benchmark_suite.py \
  --dataset-name sift1m \
  --base datasets/sift/sift_base.fvecs \
  --query datasets/sift/sift_query.fvecs \
  --groundtruth datasets/sift/sift_groundtruth.ivecs \
  --run-id sift1m-external \
  --engines qdrant,weaviate,pgvector \
  --query-limit 100
```

The benchmark suite starts and stops Docker services for the external engines.
It writes temporary files under `/tmp/loupe-benchmark-suite/<run-id>/` and
packages the final result under `bench/results/<run-id>/`.
