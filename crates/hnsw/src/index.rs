use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::io;

use loupe_distance::Metric;
use loupe_storage::{
    InMemoryVectorStore, MmapVectorStore, StorageAccessor, VectorId, VectorStore,
    write_vector_segment,
};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use loupe_trace::{
    NodeStatus, TRACE_EXPORT_VERSION, TraceEdge, TraceGraphExport, TraceGraphLimits, TraceNode,
};

use crate::{Neighbor, SearchTrace};

const ACORN_BRIDGE_FANOUT_MULTIPLIER: usize = 4;

/// Collects the local subgraph a search walks through, for the gated diagnostic
/// export. It enforces hard caps so it can never grow without bound, and it holds
/// only node ids and edges, never vectors.
struct GraphCollector {
    limits: TraceGraphLimits,
    nodes: Vec<VectorId>,
    seen: HashSet<VectorId>,
    edges: Vec<(VectorId, VectorId)>,
    truncated: bool,
}

impl GraphCollector {
    fn new(limits: TraceGraphLimits) -> Self {
        Self {
            limits,
            nodes: Vec::new(),
            seen: HashSet::new(),
            edges: Vec::new(),
            truncated: false,
        }
    }

    fn record_node(&mut self, id: VectorId) {
        if !self.seen.insert(id) {
            return;
        }
        if self.nodes.len() < self.limits.max_nodes {
            self.nodes.push(id);
        } else {
            self.truncated = true;
        }
    }

    fn record_edge(&mut self, from: VectorId, to: VectorId) {
        if self.edges.len() < self.limits.max_edges {
            self.edges.push((from, to));
        } else {
            self.truncated = true;
        }
    }

    fn into_export(self, returned: &HashSet<VectorId>) -> TraceGraphExport {
        let nodes = self
            .nodes
            .iter()
            .map(|&id| TraceNode {
                id,
                status: if returned.contains(&id) {
                    NodeStatus::Returned
                } else {
                    NodeStatus::Visited
                },
            })
            .collect();
        let edges = self
            .edges
            .iter()
            .map(|&(from, to)| TraceEdge { from, to })
            .collect();
        TraceGraphExport {
            version: TRACE_EXPORT_VERSION,
            nodes,
            edges,
            truncated: self.truncated,
        }
    }
}

/// Knobs that control index quality and speed.
///
/// `m` is how many neighbors a node links to per layer. `m_max` and `m_max0` are
/// the caps on a node's neighbor count before it gets pruned, on the upper layers
/// and on layer 0. `ef_construction` and `ef_search` are the beam widths used
/// while building and while querying, where a wider beam means better results and
/// more work. `seed` keeps builds reproducible.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct HnswConfig {
    pub m: usize,
    pub m_max: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 64,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

/// Small deterministic generator used only for assigning node levels, so a given
/// seed always builds the same graph.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.max(1))
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

/// Which filtered search strategy the adaptive overlay should run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilteredStrategy {
    PostFilter,
    InTraversal,
    Acorn,
}

/// The parts of the graph that get written to and read from the graph segment.
/// The trailing id list is the tombstone set: nodes still in the graph for
/// connectivity but never returned by a search.
type GraphParts = (
    Metric,
    HnswConfig,
    Vec<Vec<Vec<VectorId>>>,
    Vec<usize>,
    Option<VectorId>,
    usize,
    Vec<VectorId>,
);

/// A multi-layer HNSW index built on top of an in-memory vector store. Each node
/// sits in layer 0 and, with falling probability, in higher layers that act as
/// express lanes for search. Vectors are read through the storage accessor, so
/// the search code will not change when the store moves to memory-mapped files.
pub struct HnswIndex {
    store: VectorStore,
    metric: Metric,
    config: HnswConfig,
    level_multiplier: f32,
    rng: Rng,
    neighbors: Vec<Vec<Vec<VectorId>>>,
    levels: Vec<usize>,
    entry_point: Option<VectorId>,
    max_level: usize,
    deleted: RoaringBitmap,
}

impl HnswIndex {
    /// Creates an empty index for vectors of the given length and metric.
    pub fn new(dim: usize, metric: Metric, config: HnswConfig) -> Self {
        let level_multiplier = 1.0 / (config.m as f32).ln();
        let rng = Rng::new(config.seed);
        Self {
            store: VectorStore::InMemory(InMemoryVectorStore::new(dim)),
            metric,
            config,
            level_multiplier,
            rng,
            neighbors: Vec::new(),
            levels: Vec::new(),
            entry_point: None,
            max_level: 0,
            deleted: RoaringBitmap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// The highest layer any node currently sits in.
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Read access to the stored vectors, for code that needs distances outside
    /// the index (such as the planner's correlation estimate).
    pub fn accessor(&self) -> &impl StorageAccessor {
        &self.store
    }

    /// The metric this index was built with.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Marks a vector as deleted. The node stays in the graph so search can still
    /// walk through it for connectivity, but it is never returned. Returns false
    /// if the id is out of range or was already deleted.
    pub fn delete(&mut self, id: VectorId) -> bool {
        if (id as usize) >= self.len() {
            return false;
        }
        self.deleted.insert(id)
    }

    /// Whether a vector has been tombstoned.
    pub fn is_deleted(&self, id: VectorId) -> bool {
        self.deleted.contains(id)
    }

    /// How many vectors are tombstoned: still in the graph but not returnable.
    pub fn deleted_count(&self) -> usize {
        self.deleted.len() as usize
    }

    /// How many vectors a search can still return.
    pub fn live_count(&self) -> usize {
        self.len() - self.deleted_count()
    }

    /// The share of stored vectors that are tombstoned, used to decide when a
    /// compaction is worth running. Zero on an empty index.
    pub fn dead_fraction(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        self.deleted_count() as f32 / self.len() as f32
    }

    fn is_live(&self, id: VectorId) -> bool {
        !self.deleted.contains(id)
    }

    /// Writes the index to a vector segment and a graph segment. The pair can be
    /// reopened with `open` and searched without loading the vectors into RAM.
    pub fn seal(&self, vector_path: &str, graph_path: &str) -> io::Result<()> {
        write_vector_segment(vector_path, &self.store)?;
        let deleted: Vec<VectorId> = self.deleted.iter().collect();
        let graph: (&Metric, &HnswConfig, _, _, _, _, _) = (
            &self.metric,
            &self.config,
            &self.neighbors,
            &self.levels,
            &self.entry_point,
            &self.max_level,
            &deleted,
        );
        let bytes = bincode::serialize(&graph)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
        std::fs::write(graph_path, bytes)
    }

    /// Opens an index sealed by `seal`. The vectors are memory-mapped and the graph
    /// is loaded into memory. The result is read-only.
    pub fn open(vector_path: &str, graph_path: &str) -> io::Result<Self> {
        let store = VectorStore::Mmap(MmapVectorStore::open(vector_path)?);
        let bytes = std::fs::read(graph_path)?;
        let (metric, config, neighbors, levels, entry_point, max_level, deleted): GraphParts =
            bincode::deserialize(&bytes)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
        let level_multiplier = 1.0 / (config.m as f32).ln();
        let rng = Rng::new(config.seed);
        let deleted: RoaringBitmap = deleted.into_iter().collect();
        Ok(Self {
            store,
            metric,
            config,
            level_multiplier,
            rng,
            neighbors,
            levels,
            entry_point,
            max_level,
            deleted,
        })
    }

    /// Adds a vector to the index and returns its id. It gives the node a random
    /// level, walks down the upper layers to a good entry point, then links it to
    /// nearby nodes on each layer it belongs to.
    pub fn insert(&mut self, vector: &[f32]) -> VectorId {
        let id = match &mut self.store {
            VectorStore::InMemory(store) => store.add(vector),
            VectorStore::Mmap(_) => panic!("cannot insert into a sealed index"),
        };
        let level = self.random_level();
        self.neighbors.push(vec![Vec::new(); level + 1]);
        self.levels.push(level);

        let entry = match self.entry_point {
            None => {
                self.entry_point = Some(id);
                self.max_level = level;
                return id;
            }
            Some(e) => e,
        };

        let mut trace = SearchTrace::default();
        let mut entry_point = entry;
        let mut layer = self.max_level;
        while layer > level {
            entry_point = self.closest_on_layer(vector, entry_point, layer, &mut trace);
            layer -= 1;
        }

        let top_layer = level.min(self.max_level);
        for layer in (0..=top_layer).rev() {
            let found = self.search_layer(
                vector,
                entry_point,
                self.config.ef_construction,
                layer,
                &mut trace,
                &mut None,
            );
            let selected = self.select_neighbors(&found, self.config.m);
            self.neighbors[id as usize][layer] = selected.clone();

            let cap = self.layer_cap(layer);
            for &neighbor in &selected {
                self.neighbors[neighbor as usize][layer].push(id);
                if self.neighbors[neighbor as usize][layer].len() > cap {
                    self.prune(neighbor, layer, cap);
                }
            }
            if let Some(closest) = found.first() {
                entry_point = closest.id;
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(id);
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

        let mut entry_point = entry;
        let mut layer = self.max_level;
        while layer > 0 {
            entry_point = self.closest_on_layer(query, entry_point, layer, &mut trace);
            layer -= 1;
        }

        let ef = self.config.ef_search.max(k);
        let mut found = self.search_layer(query, entry_point, ef, 0, &mut trace, &mut None);
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

    /// Like `search`, but also returns a bounded export of the local graph the
    /// search walked: the visited nodes, which were returned, and the edges
    /// between them. This runs only when asked, never on a normal query.
    pub fn explain_search(
        &self,
        query: &[f32],
        k: usize,
        limits: TraceGraphLimits,
    ) -> (Vec<Neighbor>, SearchTrace, TraceGraphExport) {
        let mut trace = SearchTrace::default();
        let entry = match self.entry_point {
            Some(e) => e,
            None => {
                let empty = TraceGraphExport {
                    version: TRACE_EXPORT_VERSION,
                    nodes: Vec::new(),
                    edges: Vec::new(),
                    truncated: false,
                };
                return (Vec::new(), trace, empty);
            }
        };

        let mut entry_point = entry;
        let mut layer = self.max_level;
        while layer > 0 {
            entry_point = self.closest_on_layer(query, entry_point, layer, &mut trace);
            layer -= 1;
        }

        let ef = self.config.ef_search.max(k);
        let mut collector = Some(GraphCollector::new(limits));
        let mut found = self.search_layer(query, entry_point, ef, 0, &mut trace, &mut collector);
        found.truncate(k);

        let neighbors: Vec<Neighbor> = found
            .into_iter()
            .map(|c| Neighbor {
                id: c.id,
                distance: c.distance,
            })
            .collect();
        let returned: HashSet<VectorId> = neighbors.iter().map(|n| n.id).collect();
        let export = collector
            .expect("collector was set above")
            .into_export(&returned);
        (neighbors, trace, export)
    }

    /// Exact filtered search. Scans only the ids that pass the filter and returns
    /// the true top-k among them. Best when few ids match the filter.
    pub fn flat_filtered_search(
        &self,
        query: &[f32],
        k: usize,
        allowed: &RoaringBitmap,
    ) -> (Vec<Neighbor>, SearchTrace) {
        let mut trace = SearchTrace::default();
        if k == 0 {
            return (Vec::new(), trace);
        }

        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();
        for id in allowed.iter() {
            if !self.is_live(id) {
                trace.tombstoned_visits += 1;
                continue;
            }
            trace.distance_computations += 1;
            trace.nodes_visited += 1;
            let candidate = Candidate {
                id,
                distance: self.metric.distance(query, self.store.vector(id)),
            };
            let room_left = results.len() < k;
            let better_than_worst = results
                .peek()
                .is_some_and(|worst| candidate.distance < worst.distance);
            if room_left || better_than_worst {
                results.push(candidate);
                if results.len() > k {
                    results.pop();
                }
            }
        }
        let mut out = results.into_vec();
        out.sort();
        let neighbors = out
            .into_iter()
            .map(|candidate| Neighbor {
                id: candidate.id,
                distance: candidate.distance,
            })
            .collect();
        (neighbors, trace)
    }

    /// Post-filter search. Runs a normal search with a wider beam, then drops the
    /// ids that do not pass the filter. Best when the filter is broad.
    pub fn post_filter_search(
        &self,
        query: &[f32],
        k: usize,
        allowed: &RoaringBitmap,
        ef: usize,
    ) -> (Vec<Neighbor>, SearchTrace) {
        let mut trace = SearchTrace::default();
        let entry = match self.entry_point {
            Some(e) => e,
            None => return (Vec::new(), trace),
        };
        let mut entry_point = entry;
        let mut layer = self.max_level;
        while layer > 0 {
            entry_point = self.closest_on_layer(query, entry_point, layer, &mut trace);
            layer -= 1;
        }
        let beam = ef.max(k);
        let found = self.search_layer(query, entry_point, beam, 0, &mut trace, &mut None);
        let mut results: Vec<Neighbor> = found
            .into_iter()
            .filter(|candidate| allowed.contains(candidate.id))
            .map(|candidate| Neighbor {
                id: candidate.id,
                distance: candidate.distance,
            })
            .collect();
        results.truncate(k);
        (results, trace)
    }

    /// In-traversal filtered search. Walks the graph normally but only admits
    /// matching ids into the result set. It still moves through non-matching nodes
    /// that look promising, so it bridges short gaps in the matching subset.
    pub fn in_traversal_filtered_search(
        &self,
        query: &[f32],
        k: usize,
        allowed: &RoaringBitmap,
        ef: usize,
    ) -> (Vec<Neighbor>, SearchTrace) {
        self.filtered_search(query, k, allowed, ef, false)
    }

    /// ACORN-style filtered search. Like in-traversal, but it also looks one hop
    /// past non-matching neighbors, using them as bridges. This keeps the matching
    /// subset reachable when a strict filter leaves few direct matching neighbors.
    pub fn acorn_filtered_search(
        &self,
        query: &[f32],
        k: usize,
        allowed: &RoaringBitmap,
        ef: usize,
    ) -> (Vec<Neighbor>, SearchTrace) {
        self.filtered_search(query, k, allowed, ef, true)
    }

    fn filtered_search(
        &self,
        query: &[f32],
        k: usize,
        allowed: &RoaringBitmap,
        ef: usize,
        expand: bool,
    ) -> (Vec<Neighbor>, SearchTrace) {
        let mut trace = SearchTrace::default();
        let entry = match self.entry_point {
            Some(e) => e,
            None => return (Vec::new(), trace),
        };

        let mut entry_point = entry;
        let mut layer = self.max_level;
        while layer > 0 {
            entry_point = self.closest_on_layer(query, entry_point, layer, &mut trace);
            layer -= 1;
        }

        let beam = ef.max(k);
        let mut found =
            self.filtered_search_layer(query, entry_point, beam, allowed, expand, &mut trace);
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

    fn filtered_search_layer(
        &self,
        query: &[f32],
        entry: VectorId,
        ef: usize,
        allowed: &RoaringBitmap,
        expand: bool,
        trace: &mut SearchTrace,
    ) -> Vec<Candidate> {
        let layer = 0;
        let mut visited = HashSet::new();
        let mut to_explore = BinaryHeap::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut to_consider: Vec<VectorId> = Vec::new();

        let entry_distance = self.distance(query, entry, trace);
        visited.insert(entry);
        trace.nodes_visited += 1;
        let start = Candidate {
            distance: entry_distance,
            id: entry,
        };
        to_explore.push(Reverse(start));
        if allowed.contains(entry) && self.is_live(entry) {
            results.push(start);
        } else if !self.is_live(entry) {
            trace.tombstoned_visits += 1;
        }

        while let Some(Reverse(current)) = to_explore.pop() {
            if results.len() >= ef
                && results
                    .peek()
                    .is_some_and(|w| current.distance > w.distance)
            {
                break;
            }

            to_consider.clear();
            to_consider.extend_from_slice(&self.neighbors[current.id as usize][layer]);
            if expand {
                let bridge_limit = ef
                    .saturating_mul(ACORN_BRIDGE_FANOUT_MULTIPLIER)
                    .max(self.config.m_max0);
                for &neighbor in &self.neighbors[current.id as usize][layer] {
                    if to_consider.len() >= bridge_limit {
                        break;
                    }
                    if !allowed.contains(neighbor) {
                        for &bridge in &self.neighbors[neighbor as usize][layer] {
                            if to_consider.len() >= bridge_limit {
                                break;
                            }
                            to_consider.push(bridge);
                        }
                    }
                }
            }

            for &node in &to_consider {
                if !visited.insert(node) {
                    continue;
                }
                trace.nodes_visited += 1;
                let distance = self.distance(query, node, trace);
                if !self.is_live(node) {
                    trace.tombstoned_visits += 1;
                }
                let qualifies =
                    results.len() < ef || results.peek().is_some_and(|w| distance < w.distance);
                if qualifies {
                    let candidate = Candidate { distance, id: node };
                    to_explore.push(Reverse(candidate));
                    if allowed.contains(node) && self.is_live(node) {
                        results.push(candidate);
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out = results.into_vec();
        out.sort();
        out
    }

    /// Runs a filtered strategy and widens the beam if it comes back with fewer
    /// than k matches, retrying until it has enough, runs out of matches, or hits
    /// the beam cap. The returned trace sums the work across every attempt.
    pub fn adaptive_filtered_search(
        &self,
        query: &[f32],
        k: usize,
        allowed: &RoaringBitmap,
        strategy: FilteredStrategy,
        ef: usize,
        max_ef: usize,
    ) -> (Vec<Neighbor>, SearchTrace) {
        let mut current_ef = ef.max(k);
        let cap = max_ef.max(current_ef);
        let mut total = SearchTrace::default();
        loop {
            let (results, attempt) = self.run_filtered(strategy, query, k, allowed, current_ef);
            total.distance_computations += attempt.distance_computations;
            total.nodes_visited += attempt.nodes_visited;
            if results.len() >= k || current_ef >= cap {
                return (results, total);
            }
            current_ef = (current_ef * 2).min(cap);
            total.ef_expansions += 1;
        }
    }

    fn run_filtered(
        &self,
        strategy: FilteredStrategy,
        query: &[f32],
        k: usize,
        allowed: &RoaringBitmap,
        ef: usize,
    ) -> (Vec<Neighbor>, SearchTrace) {
        match strategy {
            FilteredStrategy::PostFilter => self.post_filter_search(query, k, allowed, ef),
            FilteredStrategy::InTraversal => {
                self.in_traversal_filtered_search(query, k, allowed, ef)
            }
            FilteredStrategy::Acorn => self.acorn_filtered_search(query, k, allowed, ef),
        }
    }

    fn random_level(&mut self) -> usize {
        let uniform = self.rng.next_f32().max(f32::MIN_POSITIVE);
        (-uniform.ln() * self.level_multiplier).floor() as usize
    }

    fn layer_cap(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.m_max0
        } else {
            self.config.m_max
        }
    }

    fn distance(&self, query: &[f32], id: VectorId, trace: &mut SearchTrace) -> f32 {
        trace.distance_computations += 1;
        self.metric.distance(query, self.store.vector(id))
    }

    fn closest_on_layer(
        &self,
        query: &[f32],
        entry: VectorId,
        layer: usize,
        trace: &mut SearchTrace,
    ) -> VectorId {
        self.search_layer(query, entry, 1, layer, trace, &mut None)
            .first()
            .map_or(entry, |c| c.id)
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry: VectorId,
        ef: usize,
        layer: usize,
        trace: &mut SearchTrace,
        collector: &mut Option<GraphCollector>,
    ) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        let mut to_explore = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        let entry_distance = self.distance(query, entry, trace);
        visited.insert(entry);
        trace.nodes_visited += 1;
        if let Some(graph) = collector.as_mut() {
            graph.record_node(entry);
        }
        let start = Candidate {
            distance: entry_distance,
            id: entry,
        };
        to_explore.push(Reverse(start));
        if self.is_live(entry) {
            results.push(start);
        } else {
            trace.tombstoned_visits += 1;
        }

        while let Some(Reverse(current)) = to_explore.pop() {
            if let Some(worst) = results.peek()
                && current.distance > worst.distance
                && results.len() >= ef
            {
                break;
            }

            for &neighbor in &self.neighbors[current.id as usize][layer] {
                if !visited.insert(neighbor) {
                    continue;
                }
                trace.nodes_visited += 1;
                if let Some(graph) = collector.as_mut() {
                    graph.record_node(neighbor);
                    graph.record_edge(current.id, neighbor);
                }
                let distance = self.distance(query, neighbor, trace);
                let room_left = results.len() < ef;
                let better_than_worst = results.peek().is_some_and(|w| distance < w.distance);
                if room_left || better_than_worst {
                    let candidate = Candidate {
                        distance,
                        id: neighbor,
                    };
                    to_explore.push(Reverse(candidate));
                    if self.is_live(neighbor) {
                        results.push(candidate);
                        if results.len() > ef {
                            results.pop();
                        }
                    } else {
                        trace.tombstoned_visits += 1;
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

    fn prune(&mut self, node: VectorId, layer: usize, cap: usize) {
        let node_vector = self.store.vector(node).to_vec();
        let mut candidates: Vec<Candidate> = self.neighbors[node as usize][layer]
            .iter()
            .map(|&neighbor| Candidate {
                distance: self
                    .metric
                    .distance(&node_vector, self.store.vector(neighbor)),
                id: neighbor,
            })
            .collect();
        candidates.sort();
        self.neighbors[node as usize][layer] = self.select_neighbors(&candidates, cap);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brute_force_search;

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
    fn builds_more_than_one_layer() {
        let dim = 8;
        let mut rng = Rng::new(42);
        let mut index = HnswIndex::new(dim, Metric::L2, HnswConfig::default());
        for _ in 0..1000 {
            let vector = random_vector(&mut rng, dim);
            index.insert(&vector);
        }
        assert!(index.max_level() >= 1);
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

    #[test]
    fn explain_search_matches_plain_search() {
        let mut index = HnswIndex::new(2, Metric::L2, HnswConfig::default());
        for i in 0..50 {
            let f = i as f32;
            index.insert(&[f, f]);
        }
        let (plain, _) = index.search(&[3.0, 3.0], 5);
        let (explained, _, _) = index.explain_search(&[3.0, 3.0], 5, TraceGraphLimits::default());
        let plain_ids: Vec<VectorId> = plain.iter().map(|n| n.id).collect();
        let explained_ids: Vec<VectorId> = explained.iter().map(|n| n.id).collect();
        assert_eq!(plain_ids, explained_ids);
    }

    #[test]
    fn explain_search_marks_returned_nodes() {
        let mut index = HnswIndex::new(2, Metric::L2, HnswConfig::default());
        for i in 0..50 {
            let f = i as f32;
            index.insert(&[f, f]);
        }
        let (neighbors, _, export) =
            index.explain_search(&[3.0, 3.0], 5, TraceGraphLimits::default());
        assert_eq!(export.version, 1);
        assert!(!export.nodes.is_empty());
        assert!(!export.edges.is_empty());
        let returned: Vec<VectorId> = export
            .nodes
            .iter()
            .filter(|n| n.status == NodeStatus::Returned)
            .map(|n| n.id)
            .collect();
        for neighbor in &neighbors {
            assert!(returned.contains(&neighbor.id));
        }
    }

    #[test]
    fn explain_search_respects_limits() {
        let mut index = HnswIndex::new(2, Metric::L2, HnswConfig::default());
        for i in 0..200 {
            let f = i as f32;
            index.insert(&[f, f]);
        }
        let limits = TraceGraphLimits {
            max_nodes: 5,
            max_edges: 5,
        };
        let (_, _, export) = index.explain_search(&[3.0, 3.0], 10, limits);
        assert!(export.nodes.len() <= 5);
        assert!(export.edges.len() <= 5);
        assert!(export.truncated);
    }

    #[test]
    fn explain_search_on_empty_index_is_empty() {
        let index = HnswIndex::new(2, Metric::L2, HnswConfig::default());
        let (neighbors, _, export) =
            index.explain_search(&[1.0, 1.0], 5, TraceGraphLimits::default());
        assert!(neighbors.is_empty());
        assert!(export.nodes.is_empty());
        assert!(export.edges.is_empty());
    }

    fn line_index(count: u32) -> HnswIndex {
        let mut index = HnswIndex::new(2, Metric::L2, HnswConfig::default());
        for i in 0..count {
            let f = i as f32;
            index.insert(&[f, f]);
        }
        index
    }

    fn even_ids(count: u32) -> RoaringBitmap {
        let mut allowed = RoaringBitmap::new();
        for id in (0..count).step_by(2) {
            allowed.insert(id);
        }
        allowed
    }

    #[test]
    fn flat_filtered_returns_exact_among_allowed() {
        let index = line_index(10);
        let mut allowed = RoaringBitmap::new();
        allowed.insert(2);
        allowed.insert(5);
        allowed.insert(8);
        let (results, _) = index.flat_filtered_search(&[5.1, 5.1], 2, &allowed);
        assert_eq!(results[0].id, 5);
        assert_eq!(results[1].id, 8);
    }

    #[test]
    fn flat_filtered_empty_allowed_is_empty() {
        let index = line_index(5);
        let (results, _) = index.flat_filtered_search(&[0.0, 0.0], 5, &RoaringBitmap::new());
        assert!(results.is_empty());
    }

    #[test]
    fn flat_filtered_zero_k_is_empty() {
        let index = line_index(20);
        let allowed = even_ids(20);
        let (results, trace) = index.flat_filtered_search(&[0.0, 0.0], 0, &allowed);
        assert!(results.is_empty());
        assert_eq!(trace.distance_computations, 0);
    }

    #[test]
    fn post_filter_only_returns_allowed() {
        let index = line_index(100);
        let allowed = even_ids(100);
        let (results, _) = index.post_filter_search(&[40.0, 40.0], 5, &allowed, 100);
        assert!(!results.is_empty());
        for neighbor in &results {
            assert!(allowed.contains(neighbor.id));
        }
    }

    #[test]
    fn post_filter_matches_exact_for_broad_filter() {
        let index = line_index(100);
        let allowed = even_ids(100);
        let query = [41.3, 41.3];
        let (post, _) = index.post_filter_search(&query, 5, &allowed, 100);
        let (flat, _) = index.flat_filtered_search(&query, 5, &allowed);
        let post_ids: Vec<VectorId> = post.iter().map(|n| n.id).collect();
        let flat_ids: Vec<VectorId> = flat.iter().map(|n| n.id).collect();
        assert_eq!(post_ids, flat_ids);
    }

    fn result_ids(results: &[Neighbor]) -> Vec<VectorId> {
        results.iter().map(|n| n.id).collect()
    }

    #[test]
    fn in_traversal_matches_exact_for_broad_filter() {
        let index = line_index(100);
        let allowed = even_ids(100);
        let query = [41.3, 41.3];
        let (got, _) = index.in_traversal_filtered_search(&query, 5, &allowed, 100);
        let (flat, _) = index.flat_filtered_search(&query, 5, &allowed);
        assert_eq!(result_ids(&got), result_ids(&flat));
    }

    #[test]
    fn acorn_matches_exact_for_broad_filter() {
        let index = line_index(100);
        let allowed = even_ids(100);
        let query = [41.3, 41.3];
        let (got, _) = index.acorn_filtered_search(&query, 5, &allowed, 100);
        let (flat, _) = index.flat_filtered_search(&query, 5, &allowed);
        assert_eq!(result_ids(&got), result_ids(&flat));
    }

    #[test]
    fn filtered_strategies_find_sparse_matches() {
        let index = line_index(200);
        let mut allowed = RoaringBitmap::new();
        allowed.insert(10);
        allowed.insert(150);
        allowed.insert(190);
        let query = [150.0, 150.0];
        let (flat, _) = index.flat_filtered_search(&query, 3, &allowed);
        let (in_traversal, _) = index.in_traversal_filtered_search(&query, 3, &allowed, 64);
        let (acorn, _) = index.acorn_filtered_search(&query, 3, &allowed, 64);
        assert_eq!(result_ids(&in_traversal), result_ids(&flat));
        assert_eq!(result_ids(&acorn), result_ids(&flat));
    }

    #[test]
    fn filtered_strategies_return_only_allowed() {
        let index = line_index(100);
        let allowed = even_ids(100);
        let query = [55.0, 55.0];
        let (in_traversal, _) = index.in_traversal_filtered_search(&query, 5, &allowed, 64);
        let (acorn, _) = index.acorn_filtered_search(&query, 5, &allowed, 64);
        for neighbor in in_traversal.iter().chain(acorn.iter()) {
            assert!(allowed.contains(neighbor.id));
        }
    }

    #[test]
    fn adaptive_widens_to_fill_k() {
        let index = line_index(200);
        let allowed = even_ids(200);
        let (results, trace) = index.adaptive_filtered_search(
            &[80.0, 80.0],
            10,
            &allowed,
            FilteredStrategy::PostFilter,
            10,
            512,
        );
        assert_eq!(results.len(), 10);
        assert!(trace.ef_expansions >= 1);
        for neighbor in &results {
            assert!(allowed.contains(neighbor.id));
        }
    }

    #[test]
    fn adaptive_skips_widening_when_initial_beam_suffices() {
        let index = line_index(200);
        let allowed = even_ids(200);
        let (results, trace) = index.adaptive_filtered_search(
            &[80.0, 80.0],
            5,
            &allowed,
            FilteredStrategy::Acorn,
            256,
            512,
        );
        assert_eq!(results.len(), 5);
        assert_eq!(trace.ef_expansions, 0);
    }

    #[test]
    fn adaptive_terminates_when_matches_fewer_than_k() {
        let index = line_index(100);
        let mut allowed = RoaringBitmap::new();
        allowed.insert(10);
        allowed.insert(20);
        let (results, _) = index.adaptive_filtered_search(
            &[15.0, 15.0],
            5,
            &allowed,
            FilteredStrategy::InTraversal,
            8,
            256,
        );
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn sealed_index_searches_like_in_memory() {
        let mut index = HnswIndex::new(2, Metric::L2, HnswConfig::default());
        for i in 0..200u32 {
            index.insert(&[i as f32, i as f32]);
        }
        let query = [73.0, 73.0];
        let (memory, _) = index.search(&query, 10);

        let vectors = tempfile::NamedTempFile::new().unwrap();
        let graph = tempfile::NamedTempFile::new().unwrap();
        let vector_path = vectors.path().to_str().unwrap();
        let graph_path = graph.path().to_str().unwrap();
        index.seal(vector_path, graph_path).unwrap();

        let reopened = HnswIndex::open(vector_path, graph_path).unwrap();
        assert_eq!(reopened.len(), 200);
        assert_eq!(reopened.max_level(), index.max_level());

        let (disk, _) = reopened.search(&query, 10);
        let memory_ids: Vec<VectorId> = memory.iter().map(|n| n.id).collect();
        let disk_ids: Vec<VectorId> = disk.iter().map(|n| n.id).collect();
        assert_eq!(memory_ids, disk_ids);
    }

    #[test]
    fn delete_updates_counts() {
        let mut index = line_index(20);
        assert_eq!(index.live_count(), 20);
        assert_eq!(index.deleted_count(), 0);
        assert!(index.delete(5));
        assert!(!index.delete(5));
        assert!(!index.delete(999));
        assert_eq!(index.deleted_count(), 1);
        assert_eq!(index.live_count(), 19);
        assert!((index.dead_fraction() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn deleted_ids_are_not_returned() {
        let mut index = line_index(50);
        index.delete(3);
        let (results, _) = index.search(&[3.0, 3.0], 5);
        assert!(results.iter().all(|n| n.id != 3));
        assert!(!results.is_empty());
    }

    #[test]
    fn search_counts_tombstoned_visits() {
        let mut index = line_index(60);
        for id in (0..60).step_by(2) {
            index.delete(id);
        }
        let (_, trace) = index.search(&[25.0, 25.0], 5);
        assert!(trace.tombstoned_visits > 0);
    }

    #[test]
    fn filtered_search_excludes_deleted() {
        let mut index = line_index(100);
        let allowed = even_ids(100);
        index.delete(40);
        let query = [40.0, 40.0];
        let (flat, _) = index.flat_filtered_search(&query, 5, &allowed);
        let (in_traversal, _) = index.in_traversal_filtered_search(&query, 5, &allowed, 64);
        let (acorn, _) = index.acorn_filtered_search(&query, 5, &allowed, 64);
        for results in [&flat, &in_traversal, &acorn] {
            assert!(results.iter().all(|n| n.id != 40));
            assert!(!results.is_empty());
        }
    }

    #[test]
    fn sealed_index_preserves_deletes() {
        let mut index = line_index(100);
        index.delete(50);
        index.delete(51);

        let vectors = tempfile::NamedTempFile::new().unwrap();
        let graph = tempfile::NamedTempFile::new().unwrap();
        let vector_path = vectors.path().to_str().unwrap();
        let graph_path = graph.path().to_str().unwrap();
        index.seal(vector_path, graph_path).unwrap();

        let reopened = HnswIndex::open(vector_path, graph_path).unwrap();
        assert_eq!(reopened.deleted_count(), 2);
        assert_eq!(reopened.live_count(), 98);
        let (results, _) = reopened.search(&[50.0, 50.0], 5);
        assert!(results.iter().all(|n| n.id != 50 && n.id != 51));
    }
}
