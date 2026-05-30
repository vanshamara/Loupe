//! Storage layer that hands out vectors by id.
//!
//! Search reads vectors only through `StorageAccessor`. That way the bytes can
//! live in memory now and in memory-mapped files later, and the search code
//! never has to change.

/// Identifier for a stored vector. `u32` matches the id space used by the filter
/// bitmaps later on.
pub type VectorId = u32;

/// Read-only access to a set of fixed-length vectors, looked up by id.
pub trait StorageAccessor {
    /// How many vectors are stored.
    fn len(&self) -> usize;

    /// True when no vectors are stored.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The length shared by every vector in this store.
    fn dim(&self) -> usize;

    /// The vector for an id. Panics if the id was never added.
    fn vector(&self, id: VectorId) -> &[f32];
}

/// An in-memory vector store. Vectors are kept in one flat buffer, laid out the
/// same way the on-disk format will use, so moving to memory-mapped files later
/// is a small step.
pub struct InMemoryVectorStore {
    dim: usize,
    count: usize,
    data: Vec<f32>,
}

impl InMemoryVectorStore {
    /// Creates an empty store where every vector will have the given length.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            count: 0,
            data: Vec::new(),
        }
    }

    /// Adds a vector and returns its id. The vector length must equal `dim`.
    pub fn add(&mut self, vector: &[f32]) -> VectorId {
        assert_eq!(vector.len(), self.dim, "vector length must equal store dim");
        let id = self.count as VectorId;
        self.data.extend_from_slice(vector);
        self.count += 1;
        id
    }
}

impl StorageAccessor for InMemoryVectorStore {
    fn len(&self) -> usize {
        self.count
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn vector(&self, id: VectorId) -> &[f32] {
        let start = id as usize * self.dim;
        &self.data[start..start + self.dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_store_is_empty() {
        let store = InMemoryVectorStore::new(4);
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        assert_eq!(store.dim(), 4);
    }

    #[test]
    fn add_returns_sequential_ids() {
        let mut store = InMemoryVectorStore::new(2);
        assert_eq!(store.add(&[1.0, 2.0]), 0);
        assert_eq!(store.add(&[3.0, 4.0]), 1);
        assert_eq!(store.add(&[5.0, 6.0]), 2);
        assert_eq!(store.len(), 3);
        assert!(!store.is_empty());
    }

    #[test]
    fn vector_reads_back_what_was_added() {
        let mut store = InMemoryVectorStore::new(3);
        let id = store.add(&[1.0, 2.0, 3.0]);
        assert_eq!(store.vector(id), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn stores_keep_vectors_separate() {
        let mut store = InMemoryVectorStore::new(2);
        let a = store.add(&[1.0, 1.0]);
        let b = store.add(&[4.0, 5.0]);
        assert_eq!(store.vector(a), &[1.0, 1.0]);
        assert_eq!(store.vector(b), &[4.0, 5.0]);
    }

    #[test]
    #[should_panic]
    fn add_rejects_wrong_dimension() {
        let mut store = InMemoryVectorStore::new(3);
        store.add(&[1.0, 2.0]);
    }
}
