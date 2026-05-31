//! Correlation-aware filter generators for the benchmark matrix.
//!
//! For one query at a target selectivity, the correlation mode decides where the
//! matching ids sit relative to the query: near it, far from it, spread randomly,
//! or grouped into a tenant-like cluster.

use loupe_distance::Metric;
use roaring::RoaringBitmap;

/// Where the matching ids sit relative to the query vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Correlation {
    Random,
    Positive,
    AntiCorrelated,
    ClusteredByTenant,
}

/// The five selectivity levels swept by the matrix, from "everything matches" down
/// to "one in a thousand matches".
pub const SELECTIVITY_LEVELS: [f32; 5] = [1.0, 0.5, 0.1, 0.01, 0.001];

/// The four correlation modes swept by the matrix.
pub const CORRELATION_MODES: [Correlation; 4] = [
    Correlation::Random,
    Correlation::Positive,
    Correlation::AntiCorrelated,
    Correlation::ClusteredByTenant,
];

/// Builds the allowed-id set for one query at the given selectivity and
/// correlation. The same seed always produces the same set.
pub fn generate_filter(
    base: &[Vec<f32>],
    query: &[f32],
    metric: Metric,
    selectivity: f32,
    mode: Correlation,
    seed: u64,
) -> RoaringBitmap {
    let count = base.len();
    let target = (count as f32 * selectivity).round() as usize;
    let target = target.min(count);

    match mode {
        Correlation::Random => random_ids(count, target, seed),
        Correlation::Positive => nearest_ids(base, query, metric, target, false),
        Correlation::AntiCorrelated => nearest_ids(base, query, metric, target, true),
        Correlation::ClusteredByTenant => {
            let center = &base[seed as usize % count];
            nearest_ids(base, center, metric, target, false)
        }
    }
}

fn random_ids(count: usize, target: usize, seed: u64) -> RoaringBitmap {
    let mut ids: Vec<u32> = (0..count as u32).collect();
    let mut rng = Rng::new(seed);
    for i in 0..target {
        let j = i + (rng.next_u64() as usize) % (count - i);
        ids.swap(i, j);
    }
    ids.into_iter().take(target).collect()
}

fn nearest_ids(
    base: &[Vec<f32>],
    point: &[f32],
    metric: Metric,
    target: usize,
    farthest: bool,
) -> RoaringBitmap {
    let mut scored: Vec<(f32, u32)> = base
        .iter()
        .enumerate()
        .map(|(id, vector)| (metric.distance(point, vector), id as u32))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    if farthest {
        scored.reverse();
    }
    scored.into_iter().take(target).map(|(_, id)| id).collect()
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_base(count: usize) -> Vec<Vec<f32>> {
        (0..count).map(|i| vec![i as f32, i as f32]).collect()
    }

    #[test]
    fn selectivity_controls_count() {
        let base = line_base(100);
        let query = [50.0, 50.0];
        let count = |s| generate_filter(&base, &query, Metric::L2, s, Correlation::Random, 1).len();
        assert_eq!(count(1.0), 100);
        assert_eq!(count(0.5), 50);
        assert_eq!(count(0.1), 10);
    }

    #[test]
    fn positive_keeps_nearest_to_query() {
        let base = line_base(100);
        let query = [50.0, 50.0];
        let allowed = generate_filter(&base, &query, Metric::L2, 0.1, Correlation::Positive, 1);
        assert!(allowed.contains(50));
        assert!(!allowed.contains(0));
    }

    #[test]
    fn anti_keeps_far_from_query() {
        let base = line_base(100);
        let query = [50.0, 50.0];
        let allowed = generate_filter(
            &base,
            &query,
            Metric::L2,
            0.1,
            Correlation::AntiCorrelated,
            1,
        );
        assert!(!allowed.contains(50));
        assert!(allowed.contains(0) || allowed.contains(99));
    }

    #[test]
    fn random_is_reproducible() {
        let base = line_base(100);
        let query = [50.0, 50.0];
        let first = generate_filter(&base, &query, Metric::L2, 0.2, Correlation::Random, 42);
        let second = generate_filter(&base, &query, Metric::L2, 0.2, Correlation::Random, 42);
        assert_eq!(first, second);
    }

    #[test]
    fn clustered_includes_its_center() {
        let base = line_base(100);
        let query = [50.0, 50.0];
        let allowed = generate_filter(
            &base,
            &query,
            Metric::L2,
            0.1,
            Correlation::ClusteredByTenant,
            30,
        );
        assert!(allowed.contains(30));
    }
}
