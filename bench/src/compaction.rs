//! Before and after probe for tombstone-aware compaction.

use std::time::Instant;

use loupe_distance::Metric;
use loupe_engine::{Collection, Expr, Value};

const K: usize = 10;

/// One before and after compaction measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactionProbe {
    pub before_total_count: usize,
    pub before_live_count: usize,
    pub before_deleted_count: usize,
    pub before_tombstoned_visits: u64,
    pub before_distance_computations: u64,
    pub before_latency_micros: u64,
    pub before_wal_bytes: u64,
    pub removed: usize,
    pub after_total_count: usize,
    pub after_live_count: usize,
    pub after_deleted_count: usize,
    pub after_tombstoned_visits: u64,
    pub after_distance_computations: u64,
    pub after_latency_micros: u64,
    pub after_wal_bytes: u64,
}

/// Builds a small line dataset, deletes every nth id, compacts, and reports both sides.
pub fn run_compaction_probe(vector_count: usize, delete_every: usize) -> CompactionProbe {
    assert!(vector_count > 0, "vector_count must be positive");
    assert!(delete_every > 0, "delete_every must be positive");

    let dir = tempfile::tempdir().expect("failed to create compaction probe directory");
    let path = dir
        .path()
        .to_str()
        .expect("compaction probe path is not utf-8");
    let wal_path = dir.path().join("wal.log");

    let mut collection =
        Collection::open(path, 2, Metric::L2).expect("failed to open compaction probe collection");
    for id in 0..vector_count as u32 {
        let value = id as f32;
        collection.insert(&[value, value], &[("cat", Value::Str("all".into()))]);
    }

    for id in (0..vector_count as u32).step_by(delete_every) {
        collection.delete(id);
    }

    let filter = Expr::Eq("cat".into(), Value::Str("all".into()));
    let query_value = (vector_count / 2) as f32;
    let query = [query_value, query_value];

    let before_stats = collection.stats();
    let before_wal_bytes = file_len(&wal_path);
    let before_start = Instant::now();
    let (_, before_explain) = collection.explain_search(&query, &filter, K);
    let before_latency_micros = before_start.elapsed().as_micros() as u64;

    let compaction = collection
        .compact_with_report()
        .expect("compaction probe failed");

    let after_stats = collection.stats();
    let after_wal_bytes = file_len(&wal_path);
    let after_start = Instant::now();
    let (_, after_explain) = collection.explain_search(&query, &filter, K);
    let after_latency_micros = after_start.elapsed().as_micros() as u64;

    CompactionProbe {
        before_total_count: before_stats.total_count,
        before_live_count: before_stats.live_count,
        before_deleted_count: before_stats.deleted_count,
        before_tombstoned_visits: before_explain.trace_metrics.tombstoned_visits,
        before_distance_computations: before_explain.trace_metrics.distance_computations,
        before_latency_micros,
        before_wal_bytes,
        removed: compaction.removed,
        after_total_count: after_stats.total_count,
        after_live_count: after_stats.live_count,
        after_deleted_count: after_stats.deleted_count,
        after_tombstoned_visits: after_explain.trace_metrics.tombstoned_visits,
        after_distance_computations: after_explain.trace_metrics.distance_computations,
        after_latency_micros,
        after_wal_bytes,
    }
}

/// Prints the compaction probe as a compact before and after table.
pub fn print_compaction_probe(report: &CompactionProbe) {
    println!(
        "{:<8} {:>8} {:>8} {:>8} {:>12} {:>12} {:>11} {:>10}",
        "phase", "total", "live", "deleted", "tombstones", "dist/query", "latency_us", "wal_bytes"
    );
    println!(
        "{:<8} {:>8} {:>8} {:>8} {:>12} {:>12} {:>11} {:>10}",
        "before",
        report.before_total_count,
        report.before_live_count,
        report.before_deleted_count,
        report.before_tombstoned_visits,
        report.before_distance_computations,
        report.before_latency_micros,
        report.before_wal_bytes
    );
    println!(
        "{:<8} {:>8} {:>8} {:>8} {:>12} {:>12} {:>11} {:>10}",
        "after",
        report.after_total_count,
        report.after_live_count,
        report.after_deleted_count,
        report.after_tombstoned_visits,
        report.after_distance_computations,
        report.after_latency_micros,
        report.after_wal_bytes
    );
    println!("removed deleted vectors: {}", report.removed);
}

fn file_len(path: &std::path::Path) -> u64 {
    std::fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_reports_before_and_after_compaction() {
        let report = run_compaction_probe(100, 2);

        assert_eq!(report.before_total_count, 100);
        assert_eq!(report.before_deleted_count, 50);
        assert_eq!(report.removed, 50);
        assert_eq!(report.after_total_count, 50);
        assert_eq!(report.after_live_count, 50);
        assert_eq!(report.after_deleted_count, 0);
        assert!(report.after_tombstoned_visits <= report.before_tombstoned_visits);
        assert!(report.after_wal_bytes < report.before_wal_bytes);
    }
}
