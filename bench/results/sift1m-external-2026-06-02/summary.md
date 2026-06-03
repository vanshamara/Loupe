# Benchmark Run: sift1m-external-2026-06-02

Dataset: `sift1m`
Created: `2026-06-03T02:24:18+00:00`

## Summary

| Engine | Mean recall | Min recall | Hard-filter recall | Mean latency (us) | Under-return cells |
|---|---:|---:|---:|---:|---:|
| Qdrant | 0.931 | 0.285 | 0.984 | 106368.1 | 0 |
| pgvector | 0.991 | 0.901 | 0.982 | 203445.4 | 0 |
| Weaviate | 0.959 | 0.294 | 0.982 | 754895.7 | 0 |

## Notes

sift1m benchmark suite run over first 100 queries; external engines use returned candidate count as the cost proxy
