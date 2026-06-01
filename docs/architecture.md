# Loupe Architecture

Loupe is a vector search engine built around filtered search visibility. The
core idea is simple: every filtered query should say what plan it picked, what
the other plans would have done, and what to change when recall is weak.

## Main Flow

The CLI and gRPC server call into `loupe-engine`. The engine owns the HNSW index,
metadata filters, planner, WAL, and collection stats. A search request turns into
four steps:

1. Build the allowed-id bitmap from metadata filters.
2. Estimate the query shape: selectivity, `k`, vector count, and filter/query
   correlation.
3. Ask the planner for a strategy.
4. Run the strategy and return results plus trace metrics.

`EXPLAIN` follows the same path, but also runs the rejected plans as
counterfactuals. That is what lets Loupe say whether the chosen plan was cheap,
risky, or beaten in hindsight.

## HNSW Index

The HNSW layer is the approximate nearest-neighbor graph. Each vector is a node.
Search starts from an entry point and walks graph neighbors with a beam. A wider
beam does more distance work but usually improves recall.

Loupe keeps the HNSW layer behind a storage accessor. The graph search asks for a
vector by id and does not care whether that vector lives in memory or in a mapped
file. That boundary is why the same filtered-search and planner code works with
both in-memory and mmap storage.

## Filter Bitmaps

Metadata lives in `loupe-filter`. Each indexed field maps values or ranges to
Roaring bitmaps of vector ids. A filter expression compiles to one allowed-id
bitmap.

The search strategies all use this bitmap differently:

- `Flat` scans only ids that pass the filter.
- `PostFilter` runs normal HNSW first, then drops ids that fail the filter.
- `InTraversal` walks the graph while admitting only matching ids to results.
- `Acorn` expands more aggressively around filtered nodes to bridge sparse or
  anti-correlated filters.

This bitmap shape keeps filtering cheap and makes selectivity available to the
planner.

## Strategies

No single filtered-search strategy wins everywhere.

`Flat` is exact and simple. It is best when the matching set is small enough that
scanning it is cheaper than graph search.

`PostFilter` is fast when the filter is broad and aligned with the query. It can
fail badly when the nearest global neighbors do not pass the filter.

`InTraversal` is safer than post-filtering because the filter is part of the walk,
but it can do more graph work.

`Acorn` is for strict or anti-correlated filters. It explores extra bridge nodes
so the search can reach matching results that normal filtered traversal might
miss. Loupe can wrap graph strategies in `AdaptiveEf`, which widens the beam when
the first attempt under-returns.

## Planner And EXPLAIN

The planner scores `Flat`, `PostFilter`, `InTraversal`, and `Acorn` from the query
shape. It uses selectivity, candidate count, `k`, vector count, `ef`, correlation,
and historical recall misses from replay logs.

Normal search only runs the chosen plan. `EXPLAIN` runs the chosen plan plus
counterfactual plans and reports:

- query summary
- chosen plan
- rejected plans
- distance computations
- returned count
- recall proxy
- risk flags
- recommendations

This is the center of the project. Loupe does not just return neighbors. It shows
why it searched that way and whether another strategy would have been better.

## Replay, Advisor, And Metrics

`loupe replay` runs a saved workload against a collection. It can compare the old
rule ladder with the live cost model and write a planner-summary CSV. That CSV
tracks recall, distance work, regret, and regret delta.

`loupe advise` reads query logs and groups them by filter. It flags patterns like
under-returning filters, beam widening, anti-correlated filters, low recall smoke,
and hot filters.

`loupe metrics` exports Prometheus text from logs, collection stats, and planner
summaries. It includes planner-choice counts, per-strategy return and recall-smoke
rates, tombstone rate, deleted-vector count, and planner regret.

Together these tools make the planner measurable instead of magical.

## WAL And Recovery

Durable collections write inserts, deletes, and compaction markers to a write-ahead
log before applying changes. On open, Loupe replays the WAL to rebuild the index
and metadata. Torn records at the end of the log are ignored, so a crash during a
write does not corrupt the collection.

Crash recovery is tested by killing a process mid-write and reopening the
collection.

## mmap Storage

Loupe can seal vectors into an mmap-backed segment. Search still sees the same
storage accessor API, so graph search, filters, EXPLAIN, and the planner do not
need a special disk path.

The current mmap path is a foundation piece. It proves the engine can search
stored vectors without rewriting the upper layers.

## Deletes And Compaction

Deletes are tombstones. A deleted vector stays in the graph until compaction, but
search will not return it. Trace metrics count tombstoned visits so the user can
see when deleted nodes are making search do extra work.

Compaction rewrites the live vectors into a clean collection and records
checkpoint markers in the WAL. `loupe stats` reports live count, deleted count,
dead-node rate, segment count, and whether compaction is recommended.

## Graph Explorer

The optional Graph Explorer in `viz/` loads a bounded trace JSON offline. It
renders the local graph touched by one query, colors returned and filtered nodes,
and lets the user toggle the chosen plan against counterfactual paths.

The viewer does not need raw vectors. It uses ids, statuses, edges, and ordered
plan events. That keeps the export small and safe to share.
