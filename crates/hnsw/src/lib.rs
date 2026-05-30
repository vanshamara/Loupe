//! HNSW vector index and the exact baseline it is checked against.

use loupe_distance::Metric;
use loupe_storage::{StorageAccessor, VectorId};

/// A search result: a vector id and its distance to the query. Smaller distance
/// means closer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    pub id: VectorId,
    pub distance: f32,
}

/// Finds the k nearest vectors to the query by scanning every vector. This is the
/// slow but exact baseline that the approximate index is measured against.
pub fn brute_force_search(
    store: &impl StorageAccessor,
    metric: Metric,
    query: &[f32],
    k: usize,
) -> Vec<Neighbor> {
    let mut results: Vec<Neighbor> = (0..store.len() as VectorId)
        .map(|id| Neighbor {
            id,
            distance: metric.distance(query, store.vector(id)),
        })
        .collect();
    results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    results.truncate(k);
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use loupe_storage::InMemoryVectorStore;

    fn store_with(vectors: &[&[f32]], dim: usize) -> InMemoryVectorStore {
        let mut store = InMemoryVectorStore::new(dim);
        for &v in vectors {
            store.add(v);
        }
        store
    }

    #[test]
    fn returns_closest_first() {
        let store = store_with(&[&[0.0, 0.0], &[10.0, 10.0], &[1.0, 1.0]], 2);
        let results = brute_force_search(&store, Metric::L2, &[0.0, 0.0], 3);
        assert_eq!(results[0].id, 0);
        assert_eq!(results[1].id, 2);
        assert_eq!(results[2].id, 1);
    }

    #[test]
    fn k_one_returns_single_nearest() {
        let store = store_with(&[&[5.0], &[1.0], &[9.0]], 1);
        let results = brute_force_search(&store, Metric::L2, &[2.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn k_larger_than_store_returns_all() {
        let store = store_with(&[&[1.0], &[2.0]], 1);
        let results = brute_force_search(&store, Metric::L2, &[0.0], 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn distances_are_sorted_ascending() {
        let store = store_with(&[&[0.0], &[3.0], &[1.0]], 1);
        let results = brute_force_search(&store, Metric::L2, &[0.0], 3);
        assert!(results[0].distance <= results[1].distance);
        assert!(results[1].distance <= results[2].distance);
    }
}
