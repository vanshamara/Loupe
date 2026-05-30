/// Counts what a single search did, so EXPLAIN and the benchmarks can report it
/// later. The search fills this in as it runs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SearchTrace {
    pub distance_computations: u64,
    pub nodes_visited: u64,
}
