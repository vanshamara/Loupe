# Benchmark Run: sift1m-m5-loupe-faiss-overnight-2026-06-02

Dataset: `sift1m`
Created: `2026-06-02T11:30:45+00:00`

## Summary

| Engine | Mean recall | Min recall | Hard-filter recall | Mean latency (us) | Under-return cells |
|---|---:|---:|---:|---:|---:|
| Loupe Planner | 0.988 | 0.966 | 0.998 | 871495.5 | 1 |
| FAISS | 0.601 | 0.000 | 0.000 | 1468.5 | 10 |

## Planner Summary

| Planner | Mean recall | Mean dist/query | Regret | Regret delta |
|---|---:|---:|---:|---:|
| RuleLadder | 0.988 | 171201 | 0.314 | 0.000 |
| CostModel | 0.988 | 171201 | 0.314 | 0.000 |

Planner calibration: CostModel matches RuleLadder on this run.

## Notes

sift1m benchmark suite run over first 100 queries; external engines use returned candidate count as the cost proxy
