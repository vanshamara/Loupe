//! Memory-mapped vector store: the on-disk vector segment and a reader that serves
//! vectors straight from the mapped file, so search needs only bounded RAM.

use std::fs::File;
use std::io::{self, BufWriter, Write};

use memmap2::Mmap;

use crate::{StorageAccessor, VectorId};

const MAGIC: &[u8; 4] = b"LPVS";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 16;

/// Writes a store's vectors to a segment file: a small header (magic, version,
/// dimension, count) followed by the vectors laid out one after another.
pub fn write_vector_segment(path: &str, store: &impl StorageAccessor) -> io::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    file.write_all(MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&(store.dim() as u32).to_le_bytes())?;
    file.write_all(&(store.len() as u32).to_le_bytes())?;
    for id in 0..store.len() as VectorId {
        file.write_all(bytemuck::cast_slice::<f32, u8>(store.vector(id)))?;
    }
    file.flush()
}

/// A vector store backed by a memory-mapped segment file. Reading a vector hands
/// back a slice into the mapping, with no copy and no full load into RAM.
pub struct MmapVectorStore {
    mmap: Mmap,
    dim: usize,
    count: usize,
}

impl MmapVectorStore {
    /// Opens a vector segment written by `write_vector_segment`.
    pub fn open(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: we only read the mapping and keep it for the store's lifetime.
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < HEADER_LEN || &mmap[0..4] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a vector segment",
            ));
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported segment version",
            ));
        }
        let dim = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
        let count = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        if mmap.len() < HEADER_LEN + count * dim * 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "segment is truncated",
            ));
        }
        Ok(Self { mmap, dim, count })
    }

    fn data(&self) -> &[f32] {
        let end = HEADER_LEN + self.count * self.dim * 4;
        bytemuck::cast_slice(&self.mmap[HEADER_LEN..end])
    }
}

impl StorageAccessor for MmapVectorStore {
    fn len(&self) -> usize {
        self.count
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn vector(&self, id: VectorId) -> &[f32] {
        let start = id as usize * self.dim;
        &self.data()[start..start + self.dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InMemoryVectorStore;

    fn sample_store() -> InMemoryVectorStore {
        let mut store = InMemoryVectorStore::new(3);
        store.add(&[1.0, 2.0, 3.0]);
        store.add(&[4.0, 5.0, 6.0]);
        store.add(&[7.0, 8.0, 9.0]);
        store
    }

    #[test]
    fn round_trips_through_disk() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        write_vector_segment(path, &sample_store()).unwrap();

        let mapped = MmapVectorStore::open(path).unwrap();
        assert_eq!(mapped.len(), 3);
        assert_eq!(mapped.dim(), 3);
        assert!(!mapped.is_empty());
        assert_eq!(mapped.vector(0), &[1.0, 2.0, 3.0]);
        assert_eq!(mapped.vector(2), &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn rejects_non_segment_file() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), b"garbage data here").unwrap();
        let result = MmapVectorStore::open(file.path().to_str().unwrap());
        assert!(result.is_err());
    }
}
