# Benchmark Run: sift1m-m5-loupe-faiss-flat-fraction-2026-06-02

Dataset: `sift1m`
Created: `2026-06-02T22:18:42+00:00`

## Summary

| Engine | Mean recall | Min recall | Hard-filter recall | Mean latency (us) | Under-return cells |
|---|---:|---:|---:|---:|---:|
| Loupe Planner | 0.990 | 0.966 | 1.000 | 2062.0 | 1 |
| FAISS | 0.601 | 0.000 | 0.000 | 2118.4 | 10 |

## Planner Summary

| Planner | Mean recall | Mean dist/query | Regret | Regret delta |
|---|---:|---:|---:|---:|
| RuleLadder | 0.949 | 115700 | 0.314 | 0.218 |
| CostModel | 0.990 | 44640 | 0.096 | 0.218 |

Planner calibration: CostModel beats RuleLadder on this run.

## Notes

sift1m benchmark suite run over first 100 queries; external engines use returned candidate count as the cost proxy
