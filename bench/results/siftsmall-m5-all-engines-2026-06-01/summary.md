# Benchmark Run: siftsmall-m5-all-engines-2026-06-01

Dataset: `siftsmall`
Created: `2026-06-01T20:38:30+00:00`

## Summary

| Engine | Mean recall | Min recall | Hard-filter recall | Mean latency (us) | Under-return cells |
|---|---:|---:|---:|---:|---:|
| Loupe Planner | 0.995 | 0.949 | 1.000 | 816.2 | 2 |
| FAISS | 0.621 | 0.000 | 0.000 | 387.4 | 10 |
| Qdrant | 1.000 | 0.999 | 1.000 | 2706.3 | 0 |
| pgvector | 1.000 | 1.000 | 1.000 | 2745.7 | 0 |
| Weaviate | 1.000 | 0.999 | 1.000 | 12749.1 | 0 |

## Planner Summary

| Planner | Mean recall | Mean dist/query | Regret | Regret delta |
|---|---:|---:|---:|---:|
| RuleLadder | 0.995 | 860 | 0.140 | 0.000 |
| CostModel | 0.995 | 860 | 0.140 | 0.000 |

Planner calibration: CostModel matches RuleLadder on this run.

## Notes

siftsmall benchmark suite run over first 100 queries; external engines use returned candidate count as the cost proxy
