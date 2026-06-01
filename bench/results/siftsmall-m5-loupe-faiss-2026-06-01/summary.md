# Benchmark Run: siftsmall-m5-loupe-faiss-2026-06-01

Dataset: `siftsmall`
Created: `2026-06-01T20:34:10+00:00`

## Summary

| Engine | Mean recall | Min recall | Hard-filter recall | Mean latency (us) | Under-return cells |
|---|---:|---:|---:|---:|---:|
| Loupe Planner | 0.995 | 0.949 | 1.000 | 812.9 | 2 |
| FAISS | 0.621 | 0.000 | 0.000 | 388.7 | 10 |

## Planner Summary

| Planner | Mean recall | Mean dist/query | Regret | Regret delta |
|---|---:|---:|---:|---:|
| RuleLadder | 0.995 | 860 | 0.140 | 0.000 |
| CostModel | 0.995 | 860 | 0.140 | 0.000 |

Planner calibration: CostModel matches RuleLadder on this run.

## Notes

siftsmall benchmark suite run over first 100 queries; external engines use returned candidate count as the cost proxy
