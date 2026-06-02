# Benchmark Run: siftsmall-m11-costmodel-win-2026-06-02

Dataset: `siftsmall`
Created: `2026-06-02T17:54:11+00:00`

## Summary

| Engine | Mean recall | Min recall | Hard-filter recall | Mean latency (us) | Under-return cells |
|---|---:|---:|---:|---:|---:|
| Loupe Planner | 1.000 | 1.000 | 1.000 | 32.8 | 0 |

## Planner Summary

| Planner | Mean recall | Mean dist/query | Regret | Regret delta |
|---|---:|---:|---:|---:|
| RuleLadder | 0.995 | 936 | 0.140 | 0.080 |
| CostModel | 1.000 | 859 | 0.060 | 0.080 |

Planner calibration: CostModel beats RuleLadder on this run.

## Notes

siftsmall planner calibration after tuning; first 10 queries; CostModel chooses Flat when graph bridging is more expensive
