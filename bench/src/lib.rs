//! Loading helpers and recall scoring for the benchmark datasets.

pub mod filters;

use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufReader, Read};

use loupe_storage::VectorId;

/// Reads vectors from the `.fvecs` format used by the SIFT datasets. Each record
/// is a 4-byte little-endian dimension followed by that many little-endian `f32`
/// values.
pub fn read_fvecs(path: &str) -> io::Result<Vec<Vec<f32>>> {
    read_fvecs_from(BufReader::new(File::open(path)?))
}

pub fn read_fvecs_from(mut reader: impl Read) -> io::Result<Vec<Vec<f32>>> {
    let mut vectors = Vec::new();
    while let Some(dim) = read_dim(&mut reader)? {
        let mut buf = vec![0u8; dim * 4];
        reader.read_exact(&mut buf)?;
        let vector = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        vectors.push(vector);
    }
    Ok(vectors)
}

/// Reads neighbor-id lists from the `.ivecs` format used for ground truth. The
/// layout matches `.fvecs`, but the values are little-endian `i32`.
pub fn read_ivecs(path: &str) -> io::Result<Vec<Vec<VectorId>>> {
    read_ivecs_from(BufReader::new(File::open(path)?))
}

pub fn read_ivecs_from(mut reader: impl Read) -> io::Result<Vec<Vec<VectorId>>> {
    let mut rows = Vec::new();
    while let Some(dim) = read_dim(&mut reader)? {
        let mut buf = vec![0u8; dim * 4];
        reader.read_exact(&mut buf)?;
        let row = buf
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as VectorId)
            .collect();
        rows.push(row);
    }
    Ok(rows)
}

fn read_dim(reader: &mut impl Read) -> io::Result<Option<usize>> {
    let mut bytes = [0u8; 4];
    let mut filled = 0;
    while filled < 4 {
        let read = reader.read(&mut bytes[filled..])?;
        if read == 0 {
            if filled == 0 {
                return Ok(None);
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated record header",
            ));
        }
        filled += read;
    }
    Ok(Some(i32::from_le_bytes(bytes) as usize))
}

/// The fraction of the true k nearest neighbors that the search also returned,
/// averaged over every query. A value of 1.0 means the search matched the exact
/// answer each time.
pub fn recall_at_k(results: &[Vec<VectorId>], ground_truth: &[Vec<VectorId>], k: usize) -> f32 {
    assert_eq!(
        results.len(),
        ground_truth.len(),
        "result and ground-truth counts differ"
    );
    let mut hits = 0usize;
    let mut total = 0usize;
    for (got, truth) in results.iter().zip(ground_truth) {
        let wanted: HashSet<VectorId> = truth.iter().take(k).copied().collect();
        hits += got.iter().take(k).filter(|id| wanted.contains(id)).count();
        total += wanted.len();
    }
    if total == 0 {
        return 0.0;
    }
    hits as f32 / total as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fvecs_bytes(vectors: &[&[f32]]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for vector in vectors {
            bytes.extend_from_slice(&(vector.len() as i32).to_le_bytes());
            for &value in *vector {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    fn ivecs_bytes(rows: &[&[VectorId]]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for row in rows {
            bytes.extend_from_slice(&(row.len() as i32).to_le_bytes());
            for &value in *row {
                bytes.extend_from_slice(&(value as i32).to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn fvecs_round_trips() {
        let bytes = fvecs_bytes(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let vectors = read_fvecs_from(Cursor::new(bytes)).unwrap();
        assert_eq!(vectors, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    }

    #[test]
    fn ivecs_round_trips() {
        let bytes = ivecs_bytes(&[&[10, 20], &[30, 40]]);
        let rows = read_ivecs_from(Cursor::new(bytes)).unwrap();
        assert_eq!(rows, vec![vec![10, 20], vec![30, 40]]);
    }

    #[test]
    fn empty_input_reads_nothing() {
        let vectors = read_fvecs_from(Cursor::new(Vec::new())).unwrap();
        assert!(vectors.is_empty());
    }

    #[test]
    fn truncated_header_is_an_error() {
        let result = read_fvecs_from(Cursor::new(vec![1u8, 2u8]));
        assert!(result.is_err());
    }

    #[test]
    fn recall_is_one_when_results_match() {
        let results = vec![vec![1, 2, 3]];
        let truth = vec![vec![1, 2, 3]];
        assert_eq!(recall_at_k(&results, &truth, 3), 1.0);
    }

    #[test]
    fn recall_is_half_when_one_of_two_match() {
        let results = vec![vec![1, 9]];
        let truth = vec![vec![1, 2]];
        assert_eq!(recall_at_k(&results, &truth, 2), 0.5);
    }

    #[test]
    fn recall_averages_over_queries() {
        let results = vec![vec![1, 2], vec![5, 6]];
        let truth = vec![vec![1, 2], vec![7, 8]];
        assert_eq!(recall_at_k(&results, &truth, 2), 0.5);
    }
}
