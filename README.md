# Loupe: Filtered Vector Search Engine

[![CI](https://github.com/vanshamara/Loupe/actions/workflows/ci.yml/badge.svg)](https://github.com/vanshamara/Loupe/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/badge/CodeQL-enabled-2da44e?logo=github)](https://github.com/vanshamara/Loupe/security/code-scanning)
[![Rust](https://img.shields.io/badge/Rust-stable-orange?logo=rust)](https://www.rust-lang.org)
[![Rust Edition](https://img.shields.io/badge/Rust%20edition-2024-orange?logo=rust)](Cargo.toml)
[![rustfmt](https://img.shields.io/badge/rustfmt-checked-2da44e?logo=rust)](rust-toolchain.toml)
[![clippy](https://img.shields.io/badge/clippy-clean-2da44e?logo=rust)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Loupe handles the filtered queries where normal ANN search gets unreliable, and
it shows you why it chose the plan it did.

Most vector engines can filter. Loupe focuses on the hard ones: tenant filters,
permissions, document types, date ranges, and other selective metadata. It
chooses a plan per query, explains that choice, and lets you replay workloads to
see whether planner changes helped.

## What Loupe Does

- Builds an HNSW vector index.
- Stores filter metadata as bitmaps.
- Chooses between `Flat`, `PostFilter`, `InTraversal`, and `Acorn`.
- Uses adaptive search width when a graph plan under-returns.
- Shows `EXPLAIN SEARCH` with rejected plans next to the chosen plan.
- Replays saved workloads and compares planner regret.
- Supports WAL recovery, mmap storage, deletes, and compaction.
- Ships CLI, gRPC server, benchmark tools, and a small graph explorer.

## Current Result

The strongest current result is SIFT1M filtered search over the first 100 query
rows. Loupe keeps hard-filter recall high while FAISS under-returns those cells,
and external engines keep recall but are much slower in the measured Docker
baseline.

| Engine | Mean recall | Hard-filter recall | Mean latency |
|---|---:|---:|---:|
| Loupe | 0.990 | 1.000 | 2.06 ms |
| FAISS | 0.601 | 0.000 | 2.12 ms |
| Qdrant | 0.931 | 0.984 | 106.37 ms |
| pgvector | 0.991 | 0.982 | 203.45 ms |
| Weaviate | 0.959 | 0.982 | 754.90 ms |

Read the details, caveats, commands, and result bundle links in
[docs/benchmarks.md](docs/benchmarks.md).

## When To Use It

Use Loupe when filtered recall matters and you need to understand why a query
missed good results.

Use FAISS when raw local ANN speed matters most and filtering is simple.

Use pgvector when vectors belong inside Postgres.

Use Qdrant or Weaviate when you need a mature production vector database today.

Loupe is still early. It is strongest as an explainable filtered-search engine
and benchmarkable planner project, not as a hosted production database.

## Quickstart

Build and test:

```bash
cargo build --all
cargo test --all
```

Run the end-to-end CLI workflow:

```bash
LOUPE_WORKFLOW_DIR=/tmp/loupe-workflow scripts/run-canonical-workflow.sh
```

Start the local gRPC server:

```bash
cargo run -p loupe-server --bin loupe-server
```

More commands are in [docs/quickstart.md](docs/quickstart.md).

## Docs

- [Architecture](docs/architecture.md)
- [Quickstart](docs/quickstart.md)
- [Benchmarks](docs/benchmarks.md)
- [Benchmark result bundles](bench/results/README.md)

## Status

Loupe has the core engine, planner, EXPLAIN path, replay, advisor, metrics, WAL,
mmap storage, deletes, compaction, server API, CLI, benchmark suite, and graph
explorer. The remaining work is final polish: public docs, quality checks, and
small usability fixes.

## License

MIT. See [LICENSE](LICENSE).
