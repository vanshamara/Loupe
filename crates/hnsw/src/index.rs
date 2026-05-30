use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use loupe_distance::Metric;
use loupe_storage::{InMemoryVectorStore, StorageAccessor, VectorId};

use crate::{Neighbor, SearchTrace};

/// Knobs that control index quality and speed.
///
/// `m` is how many neighbors a node links to. `m_max0` is the hard cap on a
/// node's neighbor count before it gets pruned. `ef_construction` and
/// `ef_search` are the beam widths used while building and while querying, where
/// a wider beam means better results and more work.
pub struct HnswConfig {
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 64,
        }
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    distance: f32,
    id: VectorId,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.id.cmp(&other.id))
    }
}

/// A single-layer HNSW index built on top of an in-memory vector store. Vectors
/// are read through the storage accessor, so the search code will not change
/// when the store moves to memory-mapped files later.
pub struct HnswIndex {
    store: InMemoryVectorStore,
    metric: Metric,
    config: HnswConfig,
    neighbors: Vec<Vec<VectorId>>,
    entry_point: Option<VectorId>,
}

impl HnswIndex {
    /// Creates an empty index for vectors of the given length and metric.
    pub fn new(dim: usize, metric: Metric, config: HnswConfig) -> Self {
        Self {
            store: InMemoryVectorStore::new(dim),
            metric,
            config,
            neighbors: Vec::new(),
            entry_point: None,
        }
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Adds a vector to the index and returns its id. It finds nearby nodes,
    /// picks a diverse set of neighbors, and links to them both ways.
    pub fn insert(&mut self, vector: &[f32]) -> VectorId {
        let id = self.store.add(vector);
        self.neighbors.push(Vec::new());

        let entry = match self.entry_point {
            None => {
                self.entry_point = Some(id);
                return id;
            }
            Some(e) => e,
        };

        let mut scratch = SearchTrace::default();
        let found = self.search_layer(vector, entry, self.config.ef_construction, &mut scratch);
        let selected = self.select_neighbors(&found, self.config.m);

        self.neighbors[id as usize] = selected.clone();
        for &neighbor in &selected {
            self.neighbors[neighbor as usize].push(id);
            if self.neighbors[neighbor as usize].len() > self.config.m_max0 {
                self.prune(neighbor);
            }
        }
        id
    }

    /// Returns the k nearest vectors to the query along with a trace of the work
    /// the search did.
    pub fn search(&self, query: &[f32], k: usize) -> (Vec<Neighbor>, SearchTrace) {
        let mut trace = SearchTrace::default();
        let entry = match self.entry_point {
            Some(e) => e,
            None => return (Vec::new(), trace),
        };

        let ef = self.config.ef_search.max(k);
        let mut found = self.search_layer(query, entry, ef, &mut trace);
        found.truncate(k);

        let neighbors = found
            .into_iter()
            .map(|c| Neighbor {
                id: c.id,
                distance: c.distance,
            })
            .collect();
        (neighbors, trace)
    }

    fn distance(&self, query: &[f32], id: VectorId, trace: &mut SearchTrace) -> f32 {
        trace.distance_computations += 1;
        self.metric.distance(query, self.store.vector(id))
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry: VectorId,
        ef: usize,
        trace: &mut SearchTrace,
    ) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        let mut to_explore = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        let entry_distance = self.distance(query, entry, trace);
        visited.insert(entry);
        trace.nodes_visited += 1;
        let start = Candidate {
            distance: entry_distance,
            id: entry,
        };
        to_explore.push(Reverse(start));
        results.push(start);

        while let Some(Reverse(current)) = to_explore.pop() {
            if let Some(worst) = results.peek()
                && current.distance > worst.distance
                && results.len() >= ef
            {
                break;
            }

            for &neighbor in &self.neighbors[current.id as usize] {
                if !visited.insert(neighbor) {
                    continue;
                }
                trace.nodes_visited += 1;
                let distance = self.distance(query, neighbor, trace);
                let room_left = results.len() < ef;
                let better_than_worst = results.peek().is_some_and(|w| distance < w.distance);
                if room_left || better_than_worst {
                    let candidate = Candidate {
                        distance,
                        id: neighbor,
                    };
                    to_explore.push(Reverse(candidate));
                    results.push(candidate);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out = results.into_vec();
        out.sort();
        out
    }

    fn select_neighbors(&self, candidates: &[Candidate], m: usize) -> Vec<VectorId> {
        let mut selected: Vec<VectorId> = Vec::new();
        for candidate in candidates {
            if selected.len() >= m {
                break;
            }
            let candidate_vector = self.store.vector(candidate.id);
            let crowds_existing = selected.iter().any(|&chosen| {
                self.metric
                    .distance(candidate_vector, self.store.vector(chosen))
                    < candidate.distance
            });
            if !crowds_existing {
                selected.push(candidate.id);
            }
        }
        selected
    }

    fn prune(&mut self, node: VectorId) {
        let node_vector = self.store.vector(node).to_vec();
        let mut candidates: Vec<Candidate> = self.neighbors[node as usize]
            .iter()
            .map(|&neighbor| Candidate {
                distance: self
                    .metric
                    .distance(&node_vector, self.store.vector(neighbor)),
                id: neighbor,
            })
            .collect();
        candidates.sort();
        self.neighbors[node as usize] = self.select_neighbors(&candidates, self.config.m_max0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brute_force_search;
    use std::collections::HashSet;

    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed)
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn next_f32(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    fn random_vector(rng: &mut Rng, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rng.next_f32()).collect()
    }

    #[test]
    fn finds_exact_nearest_on_small_set() {
        let mut index = HnswIndex::new(2, Metric::L2, HnswConfig::default());
        index.insert(&[0.0, 0.0]);
        index.insert(&[10.0, 0.0]);
        index.insert(&[0.0, 10.0]);
        index.insert(&[5.0, 5.0]);
        let (results, _) = index.search(&[0.1, 0.1], 1);
        assert_eq!(results[0].id, 0);
    }

    #[test]
    fn search_on_empty_index_returns_nothing() {
        let index = HnswIndex::new(3, Metric::L2, HnswConfig::default());
        let (results, _trace) = index.search(&[1.0, 2.0, 3.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn len_tracks_inserts() {
        let mut index = HnswIndex::new(2, Metric::L2, HnswConfig::default());
        assert!(index.is_empty());
        index.insert(&[1.0, 2.0]);
        index.insert(&[3.0, 4.0]);
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn trace_records_work_done() {
        let mut index = HnswIndex::new(2, Metric::L2, HnswConfig::default());
        for i in 0..20 {
            let f = i as f32;
            index.insert(&[f, f]);
        }
        let (_results, trace) = index.search(&[1.0, 1.0], 5);
        assert!(trace.distance_computations > 0);
        assert!(trace.nodes_visited > 0);
    }

    #[test]
    fn recall_matches_brute_force_on_random_data() {
        let dim = 16;
        let count = 1000;
        let k = 10;
        let mut rng = Rng::new(0x1234_5678);

        let config = HnswConfig {
            ef_search: 100,
            ..Default::default()
        };
        let mut index = HnswIndex::new(dim, Metric::L2, config);
        let mut baseline = InMemoryVectorStore::new(dim);
        for _ in 0..count {
            let vector = random_vector(&mut rng, dim);
            index.insert(&vector);
            baseline.add(&vector);
        }

        let mut hits = 0;
        let mut total = 0;
        for _ in 0..50 {
            let query = random_vector(&mut rng, dim);
            let truth: HashSet<VectorId> = brute_force_search(&baseline, Metric::L2, &query, k)
                .iter()
                .map(|n| n.id)
                .collect();
            let (got, _trace) = index.search(&query, k);
            hits += got.iter().filter(|n| truth.contains(&n.id)).count();
            total += k;
        }

        let recall = hits as f32 / total as f32;
        assert!(recall >= 0.9, "recall too low: {recall}");
    }
}
