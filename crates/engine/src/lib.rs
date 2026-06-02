//! The collection: an HNSW index plus a metadata store, with a planner that picks
//! a filtered-search strategy per query and an EXPLAIN that shows its reasoning.

use std::collections::HashSet;
use std::io;
use std::time::Instant;

use loupe_distance::Metric;
use loupe_filter::MetadataStore;
use loupe_hnsw::{FilteredStrategy, HnswIndex};
use loupe_planner::{
    CorrelationClass, PlannerConfig, QueryShape, choose_plan, classify_correlation,
    estimate_correlation,
};
use loupe_storage::{StorageAccessor, VectorId};
use loupe_trace::SearchTrace;
use loupe_wal::{SyncPolicy, WalRecord, WriteAheadLog};
use roaring::RoaringBitmap;

pub use loupe_filter::{Expr, Value};
pub use loupe_hnsw::Neighbor;
pub use loupe_planner::{Plan, PlannedStrategy};
pub use loupe_trace::{ExplainSearch, PlanReport, QueryLogRecord, QuerySummary};

/// A vector paired with its metadata fields, used when rebuilding the index
/// during compaction.
type LiveRecord = (Vec<f32>, Vec<(String, Value)>);

const LOGICAL_SEGMENT_ID: u32 = 0;
const DEFAULT_AUTO_COMPACTION_DEAD_FRACTION: f32 = 0.30;
const DEFAULT_AUTO_COMPACTION_MIN_DELETED: usize = 10_000;

const STRATEGIES: [PlannedStrategy; 4] = [
    PlannedStrategy::Flat,
    PlannedStrategy::PostFilter,
    PlannedStrategy::InTraversal,
    PlannedStrategy::Acorn,
];

/// Controls when Loupe rewrites deleted vectors out of the collection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompactionConfig {
    pub dead_fraction_threshold: f32,
    pub min_deleted: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            dead_fraction_threshold: DEFAULT_AUTO_COMPACTION_DEAD_FRACTION,
            min_deleted: DEFAULT_AUTO_COMPACTION_MIN_DELETED,
        }
    }
}

/// Live and deleted counts for one logical segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentStats {
    pub id: u32,
    pub total_count: usize,
    pub live_count: usize,
    pub deleted_count: usize,
    pub dead_fraction: f32,
}

/// Collection health data used by EXPLAIN, compaction, and future stats output.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectionStats {
    pub total_count: usize,
    pub live_count: usize,
    pub deleted_count: usize,
    pub dead_fraction: f32,
    pub compaction_recommended: bool,
    pub segments: Vec<SegmentStats>,
}

/// The before and after view of one compaction run.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactionReport {
    pub removed: usize,
    pub before: CollectionStats,
    pub after: CollectionStats,
}

/// A set of vectors with metadata, searched through the planner. With a write-ahead
/// log it survives a crash; without one it lives only in memory.
pub struct Collection {
    index: HnswIndex,
    metadata: MetadataStore,
    planner: PlannerConfig,
    historical_miss_rate: f32,
    correlation_samples: usize,
    seed: u64,
    wal: Option<WriteAheadLog>,
    dir: Option<String>,
    fields: Vec<Vec<(String, Value)>>,
    compaction: CompactionConfig,
}

impl Collection {
    /// Creates an empty in-memory collection with no durability.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            index: HnswIndex::new(dim, metric, loupe_hnsw::HnswConfig::default()),
            metadata: MetadataStore::new(),
            planner: PlannerConfig::default(),
            historical_miss_rate: 0.0,
            correlation_samples: 32,
            seed: 1,
            wal: None,
            dir: None,
            fields: Vec::new(),
            compaction: CompactionConfig::default(),
        }
    }

    /// Opens a durable collection in `dir`, rebuilding it by replaying the
    /// write-ahead log. A fresh directory yields an empty collection. Records torn
    /// off by a crash are skipped, so no partial write corrupts the index.
    pub fn open(dir: &str, dim: usize, metric: Metric) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let wal_path = format!("{dir}/wal.log");
        let records = WriteAheadLog::replay(&wal_path)?;

        let mut collection = Self::new(dim, metric);
        collection.dir = Some(dir.to_string());
        for record in records {
            match record {
                WalRecord::Insert { vector, fields } => {
                    let borrowed: Vec<(&str, Value)> = fields
                        .iter()
                        .map(|(k, v)| (k.as_str(), v.clone()))
                        .collect();
                    collection.apply_insert(&vector, &borrowed);
                }
                WalRecord::Delete { id } => {
                    collection.index.delete(id);
                }
                WalRecord::CompactBegin | WalRecord::CompactEnd => {}
            }
        }
        collection.wal = Some(WriteAheadLog::open(&wal_path, SyncPolicy::PerWrite)?);
        Ok(collection)
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Replaces the planner thresholds (used in tests and tuning).
    pub fn set_planner(&mut self, planner: PlannerConfig) {
        self.planner = planner;
    }

    /// Sets the planner's workload-level recall miss signal.
    pub fn set_historical_miss_rate(&mut self, rate: f32) {
        self.historical_miss_rate = rate.clamp(0.0, 1.0);
    }

    /// Replaces the automatic compaction trigger.
    pub fn set_compaction_config(&mut self, config: CompactionConfig) {
        self.compaction = config;
    }

    /// Adds a vector with its metadata and returns its id. With durability on, the
    /// change is written to the log (and flushed) before it is applied.
    pub fn insert(&mut self, vector: &[f32], fields: &[(&str, Value)]) -> VectorId {
        if let Some(wal) = &mut self.wal {
            let record = WalRecord::Insert {
                vector: vector.to_vec(),
                fields: fields
                    .iter()
                    .map(|(key, value)| ((*key).to_string(), value.clone()))
                    .collect(),
            };
            wal.append(&record)
                .expect("failed to append to the write-ahead log");
        }
        self.apply_insert(vector, fields)
    }

    fn apply_insert(&mut self, vector: &[f32], fields: &[(&str, Value)]) -> VectorId {
        let id = self.index.insert(vector);
        self.metadata.insert(id, fields);
        let owned: Vec<(String, Value)> = fields
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect();
        self.fields.push(owned);
        id
    }

    /// How many vectors are tombstoned: deleted but not yet compacted away.
    pub fn deleted_count(&self) -> usize {
        self.index.deleted_count()
    }

    /// How many vectors a search can still return.
    pub fn live_count(&self) -> usize {
        self.index.live_count()
    }

    /// Returns collection and segment counts used to decide when compaction helps.
    pub fn stats(&self) -> CollectionStats {
        let total_count = self.index.len();
        let deleted_count = self.index.deleted_count();
        let live_count = self.index.live_count();
        let dead_fraction = dead_fraction(deleted_count, total_count);
        let compaction_recommended = self.should_compact(deleted_count, dead_fraction);
        let segment = SegmentStats {
            id: LOGICAL_SEGMENT_ID,
            total_count,
            live_count,
            deleted_count,
            dead_fraction,
        };

        CollectionStats {
            total_count,
            live_count,
            deleted_count,
            dead_fraction,
            compaction_recommended,
            segments: vec![segment],
        }
    }

    /// Marks a vector deleted. It stays in the graph so search can still walk
    /// through it, but it is never returned. With durability on, the delete is
    /// logged before it is applied. Returns false if the id is unknown or was
    /// already deleted.
    pub fn delete(&mut self, id: VectorId) -> bool {
        if (id as usize) >= self.index.len() || self.index.is_deleted(id) {
            return false;
        }
        if let Some(wal) = &mut self.wal {
            wal.append(&WalRecord::Delete { id })
                .expect("failed to append to the write-ahead log");
        }
        let deleted = self.index.delete(id);
        if deleted {
            self.compact_if_needed()
                .expect("failed to compact deleted vectors");
        }
        deleted
    }

    /// Rewrites the index without the tombstoned vectors, reclaiming their space
    /// and dropping the dead-node overhead from search. With durability on, it
    /// rewrites the log so a later replay rebuilds the compacted collection. Ids
    /// are reassigned in the process. Returns how many vectors were removed.
    pub fn compact(&mut self) -> io::Result<usize> {
        Ok(self.compact_with_report()?.removed)
    }

    /// Rewrites deleted vectors away and returns before and after stats.
    pub fn compact_with_report(&mut self) -> io::Result<CompactionReport> {
        let before = self.stats();
        let removed = self.index.deleted_count();
        if removed == 0 {
            return Ok(CompactionReport {
                removed,
                before: before.clone(),
                after: before,
            });
        }

        let dim = self.index.accessor().dim();
        let metric = self.index.metric();
        let mut live: Vec<LiveRecord> = Vec::with_capacity(self.index.live_count());
        for id in 0..self.index.len() as VectorId {
            if self.index.is_deleted(id) {
                continue;
            }
            let vector = self.index.accessor().vector(id).to_vec();
            live.push((vector, self.fields[id as usize].clone()));
        }

        if let Some(dir) = self.dir.clone() {
            self.rewrite_log(&dir, &live)?;
        }

        let mut index = HnswIndex::new(dim, metric, loupe_hnsw::HnswConfig::default());
        let mut metadata = MetadataStore::new();
        let mut fields = Vec::with_capacity(live.len());
        for (vector, record) in live {
            let borrowed: Vec<(&str, Value)> = record
                .iter()
                .map(|(key, value)| (key.as_str(), value.clone()))
                .collect();
            let id = index.insert(&vector);
            metadata.insert(id, &borrowed);
            fields.push(record);
        }
        self.index = index;
        self.metadata = metadata;
        self.fields = fields;
        let after = self.stats();
        Ok(CompactionReport {
            removed,
            before,
            after,
        })
    }

    /// Runs compaction only when the deleted-vector metric crosses the threshold.
    pub fn compact_if_needed(&mut self) -> io::Result<Option<CompactionReport>> {
        let stats = self.stats();
        if !stats.compaction_recommended {
            return Ok(None);
        }

        self.compact_with_report().map(Some)
    }

    /// Writes the live vectors to a fresh log and swaps it in with an atomic
    /// rename. If a crash interrupts this, the old log is still intact and replays
    /// the pre-compaction state, so recovery is always safe.
    fn rewrite_log(&mut self, dir: &str, live: &[LiveRecord]) -> io::Result<()> {
        let wal_path = format!("{dir}/wal.log");
        let tmp_path = format!("{dir}/wal.log.compacting");
        self.wal = None;
        let _ = std::fs::remove_file(&tmp_path);

        let mut fresh = WriteAheadLog::open(&tmp_path, SyncPolicy::Group)?;
        fresh.append(&WalRecord::CompactBegin)?;
        for (vector, fields) in live {
            fresh.append(&WalRecord::Insert {
                vector: vector.clone(),
                fields: fields.clone(),
            })?;
        }
        fresh.append(&WalRecord::CompactEnd)?;
        fresh.sync()?;
        drop(fresh);

        std::fs::rename(&tmp_path, &wal_path)?;
        sync_directory(dir)?;
        self.wal = Some(WriteAheadLog::open(&wal_path, SyncPolicy::PerWrite)?);
        Ok(())
    }

    fn should_compact(&self, deleted_count: usize, dead_fraction: f32) -> bool {
        deleted_count >= self.compaction.min_deleted
            && dead_fraction >= self.compaction.dead_fraction_threshold
    }

    /// Runs a filtered search with the strategy the planner picks.
    pub fn search(&self, query: &[f32], filter: &Expr, k: usize) -> (Vec<Neighbor>, Plan) {
        let allowed = self.metadata.evaluate(filter);
        let plan = self.plan_for(query, &allowed, k);
        let (results, _) = self.run_plan(&plan, query, &allowed, k);
        (results, plan)
    }

    /// Like `search`, but also returns a query-log record (the cheap always-on
    /// trace) so the workload can be saved and fed to the advisor. `filter_label`
    /// is the human-readable filter that goes in the log.
    pub fn search_logged(
        &self,
        query: &[f32],
        filter: &Expr,
        filter_label: &str,
        k: usize,
    ) -> (Vec<Neighbor>, QueryLogRecord) {
        let allowed = self.metadata.evaluate(filter);
        let plan = self.plan_for(query, &allowed, k);
        let start = Instant::now();
        let (results, trace) = self.run_plan(&plan, query, &allowed, k);
        let latency_micros = start.elapsed().as_micros() as u64;
        let record = QueryLogRecord {
            query_id: hash_query(query),
            k,
            latency_micros,
            returned: results.iter().map(|n| n.id).collect(),
            metrics: trace,
            filter: Some(filter_label.to_string()),
            plan: Some(plan_name(&plan)),
            recall_smoke: None,
        };
        (results, record)
    }

    /// Runs the chosen strategy and the rejected ones, returning the chosen
    /// results plus an EXPLAIN: why this plan, and what the others would have cost.
    pub fn explain_search(
        &self,
        query: &[f32],
        filter: &Expr,
        k: usize,
    ) -> (Vec<Neighbor>, ExplainSearch) {
        let allowed = self.metadata.evaluate(filter);
        let matching_count = allowed.len();
        let total = self.metadata.total().max(1);
        let selectivity = matching_count as f32 / total as f32;
        let score = estimate_correlation(
            self.index.accessor(),
            self.index.metric(),
            query,
            &allowed,
            self.correlation_samples,
            self.seed,
        );
        let correlation = classify_correlation(score, self.planner.correlation_band);
        let mut shape = QueryShape::new(total as usize, matching_count as usize, k, correlation);
        shape.candidate_count = self.planner.ef.max(k);
        shape.historical_miss_rate = self.historical_miss_rate;
        let plan = choose_plan(&shape, &self.planner);

        let flat_plan = report_plan(PlannedStrategy::Flat, &plan);
        let (oracle, flat_trace) = self.run_plan(&flat_plan, query, &allowed, k);
        let oracle_ids: HashSet<VectorId> = oracle.iter().map(|n| n.id).collect();

        let mut reports = Vec::with_capacity(STRATEGIES.len());
        let mut chosen_results = Vec::new();
        let mut chosen_trace = SearchTrace::default();
        for strategy in STRATEGIES {
            let current_plan = report_plan(strategy, &plan);
            let (results, trace) = if current_plan.strategy == PlannedStrategy::Flat {
                (oracle.clone(), flat_trace.clone())
            } else {
                self.run_plan(&current_plan, query, &allowed, k)
            };
            let chosen = current_plan.strategy == plan.strategy;
            if chosen {
                chosen_results = results.clone();
                chosen_trace = trace.clone();
            }
            reports.push(PlanReport {
                strategy: plan_name(&current_plan),
                chosen,
                distance_computations: trace.distance_computations,
                returned: results.len(),
                recall_proxy: overlap(&results, &oracle_ids),
            });
        }

        let chosen_slot = reports.iter().position(|r| r.chosen).unwrap_or(0);
        let chosen_plan = reports[chosen_slot].clone();
        let counterfactual_plans = reports
            .iter()
            .enumerate()
            .filter(|(slot, _)| *slot != chosen_slot)
            .map(|(_, report)| report.clone())
            .collect();

        let risk_flags = risk_flags(correlation, &reports, k);
        let recommendations = recommendations(
            &reports,
            chosen_slot,
            self.index.dead_fraction(),
            &chosen_trace,
        );
        let explain = ExplainSearch {
            query_summary: QuerySummary {
                k,
                matching_count,
                selectivity,
                correlation: correlation_name(correlation).to_string(),
            },
            chosen_plan,
            counterfactual_plans,
            trace_metrics: chosen_trace,
            risk_flags,
            recommendations,
        };
        (chosen_results, explain)
    }

    fn plan_for(&self, query: &[f32], allowed: &RoaringBitmap, k: usize) -> Plan {
        let total = self.metadata.total().max(1);
        let score = estimate_correlation(
            self.index.accessor(),
            self.index.metric(),
            query,
            allowed,
            self.correlation_samples,
            self.seed,
        );
        let correlation = classify_correlation(score, self.planner.correlation_band);
        let mut shape = QueryShape::new(total as usize, allowed.len() as usize, k, correlation);
        shape.candidate_count = self.planner.ef.max(k);
        shape.historical_miss_rate = self.historical_miss_rate;
        choose_plan(&shape, &self.planner)
    }

    fn run_plan(
        &self,
        plan: &Plan,
        query: &[f32],
        allowed: &RoaringBitmap,
        k: usize,
    ) -> (Vec<Neighbor>, SearchTrace) {
        let adaptive_strategy = if plan.adaptive {
            filtered_strategy(plan.strategy)
        } else {
            None
        };

        if let Some(strategy) = adaptive_strategy {
            return self.index.adaptive_filtered_search(
                query,
                k,
                allowed,
                strategy,
                plan.ef,
                plan.max_ef,
            );
        }

        self.run_strategy(plan.strategy, query, allowed, k, plan.ef)
    }

    fn run_strategy(
        &self,
        strategy: PlannedStrategy,
        query: &[f32],
        allowed: &RoaringBitmap,
        k: usize,
        ef: usize,
    ) -> (Vec<Neighbor>, SearchTrace) {
        match strategy {
            PlannedStrategy::Flat => self.index.flat_filtered_search(query, k, allowed),
            PlannedStrategy::PostFilter => self.index.post_filter_search(query, k, allowed, ef),
            PlannedStrategy::InTraversal => self
                .index
                .in_traversal_filtered_search(query, k, allowed, ef),
            PlannedStrategy::Acorn => self.index.acorn_filtered_search(query, k, allowed, ef),
        }
    }
}

/// Parses a simple filter string into an `Expr`. Supports `field=value` (string
/// equals), `field>=N`, and `field<=N` (integer range). Shared by the CLI and the
/// gRPC server.
pub fn parse_filter(text: &str) -> Result<Expr, String> {
    if let Some((field, value)) = text.split_once(">=") {
        let bound: i64 = value
            .trim()
            .parse()
            .map_err(|_| format!("invalid integer in {text:?}"))?;
        return Ok(Expr::Range(field.trim().to_string(), bound, i64::MAX));
    }
    if let Some((field, value)) = text.split_once("<=") {
        let bound: i64 = value
            .trim()
            .parse()
            .map_err(|_| format!("invalid integer in {text:?}"))?;
        return Ok(Expr::Range(field.trim().to_string(), i64::MIN, bound));
    }
    let (field, value) = text
        .split_once('=')
        .ok_or_else(|| format!("filter must be field=value or field>=N, got {text:?}"))?;
    Ok(Expr::Eq(
        field.trim().to_string(),
        Value::Str(value.trim().to_string()),
    ))
}

fn dead_fraction(deleted_count: usize, total_count: usize) -> f32 {
    if total_count == 0 {
        return 0.0;
    }

    deleted_count as f32 / total_count as f32
}

#[cfg(unix)]
fn sync_directory(dir: &str) -> io::Result<()> {
    std::fs::File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_dir: &str) -> io::Result<()> {
    Ok(())
}

fn hash_query(query: &[f32]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for value in query {
        hash ^= u64::from(value.to_bits());
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn overlap(results: &[Neighbor], oracle_ids: &HashSet<VectorId>) -> f32 {
    if oracle_ids.is_empty() {
        return 1.0;
    }
    let hits = results
        .iter()
        .filter(|n| oracle_ids.contains(&n.id))
        .count();
    hits as f32 / oracle_ids.len() as f32
}

fn risk_flags(correlation: CorrelationClass, reports: &[PlanReport], k: usize) -> Vec<String> {
    let mut flags = Vec::new();
    if correlation == CorrelationClass::AntiCorrelated {
        flags.push("anti_correlated_filter".to_string());
    }
    if let Some(post) = reports
        .iter()
        .find(|r| r.strategy.starts_with("PostFilter"))
        && post.returned < k
    {
        flags.push("post_filter_under_returns".to_string());
    }
    flags
}

fn recommendations(
    reports: &[PlanReport],
    chosen_slot: usize,
    dead_fraction: f32,
    chosen: &SearchTrace,
) -> Vec<String> {
    let mut advice = Vec::new();
    if reports[chosen_slot].recall_proxy < 1.0 {
        advice.push("raise ef or use an exact scan for full recall".to_string());
    }
    if dead_fraction >= 0.2 {
        let percent = (dead_fraction * 100.0).round() as u32;
        advice.push(format!(
            "{percent}% of vectors are deleted; run compaction to reclaim space and speed up search"
        ));
    } else if chosen.tombstoned_visits > 0 && chosen.tombstoned_visits * 2 >= chosen.nodes_visited {
        advice.push(
            "search walks through many deleted nodes; compaction would cut that work".to_string(),
        );
    }
    advice
}

fn strategy_name(strategy: PlannedStrategy) -> &'static str {
    match strategy {
        PlannedStrategy::Flat => "Flat",
        PlannedStrategy::PostFilter => "PostFilter",
        PlannedStrategy::InTraversal => "InTraversal",
        PlannedStrategy::Acorn => "Acorn",
    }
}

fn plan_name(plan: &Plan) -> String {
    if plan.adaptive {
        return format!("{}+AdaptiveEf", strategy_name(plan.strategy));
    }

    strategy_name(plan.strategy).to_string()
}

fn report_plan(strategy: PlannedStrategy, chosen: &Plan) -> Plan {
    Plan {
        strategy,
        ef: chosen.ef,
        max_ef: chosen.max_ef,
        adaptive: strategy != PlannedStrategy::Flat,
        reason: if strategy == chosen.strategy {
            chosen.reason
        } else {
            "counterfactual plan"
        },
    }
}

fn filtered_strategy(strategy: PlannedStrategy) -> Option<FilteredStrategy> {
    match strategy {
        PlannedStrategy::Flat => None,
        PlannedStrategy::PostFilter => Some(FilteredStrategy::PostFilter),
        PlannedStrategy::InTraversal => Some(FilteredStrategy::InTraversal),
        PlannedStrategy::Acorn => Some(FilteredStrategy::Acorn),
    }
}

fn correlation_name(correlation: CorrelationClass) -> &'static str {
    match correlation {
        CorrelationClass::Positive => "positive",
        CorrelationClass::Neutral => "neutral",
        CorrelationClass::AntiCorrelated => "anti_correlated",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anti_collection() -> Collection {
        let mut collection = Collection::new(2, Metric::L2);
        collection.set_planner(PlannerConfig {
            flat_threshold: 5,
            strict_selectivity: 0.01,
            broad_selectivity: 0.4,
            ef: 64,
            max_ef: 256,
            correlation_band: 0.1,
            ..PlannerConfig::default()
        });
        for i in 0..200u32 {
            let category = if i < 100 { "low" } else { "high" };
            collection.insert(
                &[i as f32, i as f32],
                &[("cat", Value::Str(category.into()))],
            );
        }
        collection
    }

    #[test]
    fn explain_picks_flat_for_cheap_anti_correlation_and_flags_risk() {
        let collection = anti_collection();
        let query = [185.0, 185.0];
        let filter = Expr::Eq("cat".into(), Value::Str("low".into()));
        let (results, explain) = collection.explain_search(&query, &filter, 10);

        assert_eq!(explain.chosen_plan.strategy, "Flat");
        assert!(
            explain
                .risk_flags
                .contains(&"anti_correlated_filter".to_string())
        );
        assert_eq!(explain.counterfactual_plans.len(), 3);
        for neighbor in &results {
            assert!(neighbor.id < 100);
        }
        let post = explain
            .counterfactual_plans
            .iter()
            .find(|p| p.strategy == "PostFilter+AdaptiveEf")
            .unwrap();
        assert_eq!(post.returned, 10);
    }

    #[test]
    fn explain_picks_flat_for_small_broad_aligned_filter() {
        let collection = anti_collection();
        let query = [150.0, 150.0];
        let filter = Expr::Eq("cat".into(), Value::Str("high".into()));
        let (_, explain) = collection.explain_search(&query, &filter, 10);
        assert_eq!(explain.chosen_plan.strategy, "Flat");
    }

    #[test]
    fn search_returns_only_matching_ids() {
        let collection = anti_collection();
        let filter = Expr::Eq("cat".into(), Value::Str("low".into()));
        let (results, plan) = collection.search(&[185.0, 185.0], &filter, 5);
        assert_eq!(plan.strategy, PlannedStrategy::Flat);
        assert!(!plan.adaptive);
        for neighbor in &results {
            assert!(neighbor.id < 100);
        }
    }

    #[test]
    fn search_uses_adaptive_ef_for_post_filter_under_returns() {
        let mut collection = Collection::new(2, Metric::L2);
        collection.set_planner(PlannerConfig {
            flat_threshold: 5,
            strict_selectivity: 0.01,
            broad_selectivity: 0.4,
            ef: 10,
            max_ef: 128,
            correlation_band: 0.1,
            ..PlannerConfig::default()
        });

        for i in 0..200u32 {
            let category = if i % 2 == 0 { "even" } else { "odd" };
            collection.insert(
                &[i as f32, i as f32],
                &[("cat", Value::Str(category.into()))],
            );
        }

        let filter = Expr::Eq("cat".into(), Value::Str("even".into()));
        let (results, plan) = collection.search(&[80.0, 80.0], &filter, 10);

        assert_eq!(plan.strategy, PlannedStrategy::PostFilter);
        assert!(plan.adaptive);
        assert_eq!(results.len(), 10);
        for neighbor in &results {
            assert_eq!(neighbor.id % 2, 0);
        }
    }

    #[test]
    fn historical_misses_change_planner_choice() {
        let mut collection = Collection::new(2, Metric::L2);
        collection.set_planner(PlannerConfig {
            flat_threshold: 5,
            strict_selectivity: 0.01,
            broad_selectivity: 0.4,
            ef: 10,
            max_ef: 128,
            correlation_band: 0.1,
            ..PlannerConfig::default()
        });

        for i in 0..200u32 {
            let category = if i % 2 == 0 { "even" } else { "odd" };
            collection.insert(
                &[i as f32, i as f32],
                &[("cat", Value::Str(category.into()))],
            );
        }

        let filter = Expr::Eq("cat".into(), Value::Str("even".into()));
        let (_, baseline) = collection.search(&[80.0, 80.0], &filter, 10);
        collection.set_historical_miss_rate(0.8);
        let (_, adjusted) = collection.search(&[80.0, 80.0], &filter, 10);

        assert_eq!(baseline.strategy, PlannedStrategy::PostFilter);
        assert_ne!(adjusted.strategy, PlannedStrategy::PostFilter);
    }

    #[test]
    fn recovers_inserts_and_serves_filtered_queries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        {
            let mut collection = Collection::open(path, 2, Metric::L2).unwrap();
            collection.insert(&[1.0, 1.0], &[("cat", Value::Str("a".into()))]);
            collection.insert(&[2.0, 2.0], &[("cat", Value::Str("b".into()))]);
            collection.insert(&[3.0, 3.0], &[("cat", Value::Str("a".into()))]);
        }

        let recovered = Collection::open(path, 2, Metric::L2).unwrap();
        assert_eq!(recovered.len(), 3);
        let filter = Expr::Eq("cat".into(), Value::Str("a".into()));
        let (results, _) = recovered.search(&[1.0, 1.0], &filter, 5);
        let ids: Vec<VectorId> = results.iter().map(|n| n.id).collect();
        assert_eq!(ids, vec![0, 2]);
    }

    #[test]
    fn delete_excludes_from_search() {
        let mut collection = anti_collection();
        let filter = Expr::Eq("cat".into(), Value::Str("low".into()));
        assert!(collection.delete(50));
        assert!(!collection.delete(50));
        assert_eq!(collection.deleted_count(), 1);
        let (results, _) = collection.search(&[50.0, 50.0], &filter, 5);
        assert!(results.iter().all(|n| n.id != 50));
        assert!(!results.is_empty());
    }

    #[test]
    fn stats_report_live_deleted_and_segment_counts() {
        let mut collection = anti_collection();
        collection.delete(10);
        collection.delete(11);

        let stats = collection.stats();

        assert_eq!(stats.total_count, 200);
        assert_eq!(stats.live_count, 198);
        assert_eq!(stats.deleted_count, 2);
        assert_eq!(stats.segments.len(), 1);
        assert_eq!(stats.segments[0].id, LOGICAL_SEGMENT_ID);
        assert_eq!(stats.segments[0].deleted_count, 2);
        assert!(!stats.compaction_recommended);
    }

    #[test]
    fn delete_triggers_compaction_when_threshold_crosses() {
        let mut collection = Collection::new(2, Metric::L2);
        collection.set_compaction_config(CompactionConfig {
            dead_fraction_threshold: 0.2,
            min_deleted: 4,
        });
        for i in 0..20u32 {
            collection.insert(&[i as f32, i as f32], &[("cat", Value::Str("a".into()))]);
        }

        collection.delete(0);
        collection.delete(1);
        collection.delete(2);
        assert_eq!(collection.deleted_count(), 3);

        collection.delete(3);
        assert_eq!(collection.deleted_count(), 0);
        assert_eq!(collection.len(), 16);
        assert_eq!(collection.live_count(), 16);
    }

    #[test]
    fn compaction_drops_deleted_and_keeps_live() {
        let mut collection = anti_collection();
        for id in 0..40u32 {
            assert!(collection.delete(id));
        }
        assert_eq!(collection.deleted_count(), 40);
        let removed = collection.compact().unwrap();
        assert_eq!(removed, 40);
        assert_eq!(collection.deleted_count(), 0);
        assert_eq!(collection.len(), 160);
        assert_eq!(collection.live_count(), 160);
        let filter = Expr::Eq("cat".into(), Value::Str("low".into()));
        let (results, _) = collection.search(&[95.0, 95.0], &filter, 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn delete_survives_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        {
            let mut collection = Collection::open(path, 2, Metric::L2).unwrap();
            collection.insert(&[1.0, 1.0], &[("cat", Value::Str("a".into()))]);
            collection.insert(&[2.0, 2.0], &[("cat", Value::Str("a".into()))]);
            collection.insert(&[3.0, 3.0], &[("cat", Value::Str("a".into()))]);
            assert!(collection.delete(1));
        }

        let recovered = Collection::open(path, 2, Metric::L2).unwrap();
        assert_eq!(recovered.deleted_count(), 1);
        let filter = Expr::Eq("cat".into(), Value::Str("a".into()));
        let (results, _) = recovered.search(&[2.0, 2.0], &filter, 5);
        assert!(results.iter().all(|n| n.id != 1));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn compaction_survives_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        {
            let mut collection = Collection::open(path, 2, Metric::L2).unwrap();
            for i in 0..10u32 {
                collection.insert(&[i as f32, i as f32], &[("cat", Value::Str("a".into()))]);
            }
            collection.delete(0);
            collection.delete(1);
            collection.delete(2);
            assert_eq!(collection.compact().unwrap(), 3);
        }

        let recovered = Collection::open(path, 2, Metric::L2).unwrap();
        assert_eq!(recovered.len(), 7);
        assert_eq!(recovered.deleted_count(), 0);
        let filter = Expr::Eq("cat".into(), Value::Str("a".into()));
        let (results, _) = recovered.search(&[5.0, 5.0], &filter, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn compaction_writes_checkpoint_markers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        {
            let mut collection = Collection::open(path, 2, Metric::L2).unwrap();
            for i in 0..10u32 {
                collection.insert(&[i as f32, i as f32], &[("cat", Value::Str("a".into()))]);
            }
            collection.delete(0);
            collection.delete(1);
            collection.delete(2);
            assert_eq!(collection.compact().unwrap(), 3);
        }

        let wal_path = format!("{path}/wal.log");
        let records = WriteAheadLog::replay(&wal_path).unwrap();
        assert!(matches!(records.first(), Some(WalRecord::CompactBegin)));
        assert!(matches!(records.last(), Some(WalRecord::CompactEnd)));
        assert_eq!(records.len(), 9);
    }

    #[test]
    fn recovery_ignores_a_torn_tail() {
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        {
            let mut collection = Collection::open(path, 2, Metric::L2).unwrap();
            collection.insert(&[1.0, 1.0], &[("cat", Value::Str("a".into()))]);
            collection.insert(&[2.0, 2.0], &[("cat", Value::Str("a".into()))]);
        }

        let wal_path = format!("{path}/wal.log");
        let mut raw = std::fs::OpenOptions::new()
            .append(true)
            .open(&wal_path)
            .unwrap();
        raw.write_all(&999u32.to_le_bytes()).unwrap();
        raw.write_all(&0u32.to_le_bytes()).unwrap();
        raw.write_all(&[7u8; 5]).unwrap();
        drop(raw);

        let recovered = Collection::open(path, 2, Metric::L2).unwrap();
        assert_eq!(recovered.len(), 2);
    }
}
