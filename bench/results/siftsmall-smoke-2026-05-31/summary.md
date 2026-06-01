# Benchmark Run: siftsmall-smoke-2026-05-31

Dataset: `siftsmall`
Created: `2026-06-01T03:29:23+00:00`

## Summary

| Engine | Mean recall | Min recall | Hard-filter recall | Mean latency (us) | Under-return cells |
|---|---:|---:|---:|---:|---:|
| Loupe Planner | 0.995 | 0.948 | 1.000 | 817.0 | 2 |
| FAISS | 0.621 | 0.000 | 0.000 | 470.8 | 10 |
| Qdrant | 1.000 | 0.997 | 1.000 | 2756.0 | 0 |
| pgvector | 1.000 | 1.000 | 1.000 | 2618.5 | 0 |
| Weaviate | 1.000 | 0.999 | 1.000 | 13046.7 | 0 |

## Notes

siftsmall smoke run on local Docker services; external engines use returned candidate count as the cost proxy
