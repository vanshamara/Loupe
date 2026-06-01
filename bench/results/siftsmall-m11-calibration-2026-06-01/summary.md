# Benchmark Run: siftsmall-m11-calibration-2026-06-01

Dataset: `siftsmall`
Created: `2026-06-01T19:15:40+00:00`

## Summary

| Engine | Mean recall | Min recall | Hard-filter recall | Mean latency (us) | Under-return cells |
|---|---:|---:|---:|---:|---:|
| Loupe Planner | 0.995 | 0.940 | 1.000 | 963.7 | 0 |

## Planner Summary

| Planner | Mean recall | Mean dist/query | Regret | Regret delta |
|---|---:|---:|---:|---:|
| RuleLadder | 0.995 | 936 | 0.140 | 0.000 |
| CostModel | 0.995 | 936 | 0.140 | 0.000 |

Planner calibration: CostModel matches RuleLadder on this run.

## Notes

siftsmall benchmark suite run over first 10 queries
