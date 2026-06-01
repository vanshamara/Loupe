# M11 Replay Win: CostModel vs RuleLadder

Dataset: saved skewed replay workload, 10,000 vectors, 20 replay queries.

Filter: `category=target`

The target filter matches 5,000 vectors, which is above the old exact-scan cutoff.
The feedback log reports a high miss rate for this filter. With that history,
`CostModel` chooses `Flat`, while the old `RuleLadder` stays on
`Acorn+AdaptiveEf`.

| Planner | Mean recall | Mean dist/query | Regret | Regret delta |
| --- | ---: | ---: | ---: | ---: |
| RuleLadder | 0.000000 | 260.000 | 1.000000 | 1.000000 |
| CostModel | 1.000000 | 5000.000 | 0.000000 | 1.000000 |

Planner calibration: CostModel beats RuleLadder on this replay workload.

