//! Write-ahead log: an append-only record of changes so the index can be rebuilt
//! after a crash. Each record is framed with its length and a CRC, so a write that
//! was torn off by a crash is detected and ignored on replay.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};

use loupe_filter::Value;
use loupe_storage::VectorId;
use serde::{Deserialize, Serialize};

const HEADER_LEN: usize = 8;

/// A single change to the index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum WalRecord {
    Insert {
        vector: Vec<f32>,
        fields: Vec<(String, Value)>,
    },
    Delete {
        id: VectorId,
    },
    CompactBegin,
    CompactEnd,
}

/// When to flush writes to disk. `PerWrite` is durable on every append;
/// `Group` waits for an explicit `sync`, which is faster for batches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncPolicy {
    PerWrite,
    Group,
}

/// An open write-ahead log that records are appended to.
pub struct WriteAheadLog {
    file: File,
    policy: SyncPolicy,
}

impl WriteAheadLog {
    /// Opens (or creates) a log for appending.
    pub fn open(path: &str, policy: SyncPolicy) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { file, policy })
    }

    /// Appends one record. With `PerWrite` it also flushes to disk.
    pub fn append(&mut self, record: &WalRecord) -> io::Result<()> {
        let payload = bincode::serialize(record)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
        let crc = crc32fast::hash(&payload);

        let mut frame = Vec::with_capacity(HEADER_LEN + payload.len());
        frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        frame.extend_from_slice(&crc.to_le_bytes());
        frame.extend_from_slice(&payload);
        self.file.write_all(&frame)?;

        if self.policy == SyncPolicy::PerWrite {
            self.file.sync_data()?;
        }
        Ok(())
    }

    /// Flushes any buffered writes to disk. Used for group commit: append several
    /// records, then sync once.
    pub fn sync(&mut self) -> io::Result<()> {
        self.file.sync_data()
    }

    /// Reads every valid record from a log, stopping at the first record that was
    /// torn off or corrupted by a crash. A missing file replays as empty.
    pub fn replay(path: &str) -> io::Result<Vec<WalRecord>> {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };

        let mut records = Vec::new();
        let mut offset = 0;
        while offset + HEADER_LEN <= bytes.len() {
            let length = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            let crc = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
            let start = offset + HEADER_LEN;
            let end = start + length;
            if end > bytes.len() {
                break;
            }
            let payload = &bytes[start..end];
            if crc32fast::hash(payload) != crc {
                break;
            }
            match bincode::deserialize::<WalRecord>(payload) {
                Ok(record) => records.push(record),
                Err(_) => break,
            }
            offset = end;
        }
        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_of(file: &tempfile::NamedTempFile) -> String {
        file.path().to_str().unwrap().to_string()
    }

    #[test]
    fn replay_returns_appended_records() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let path = path_of(&file);
        {
            let mut wal = WriteAheadLog::open(&path, SyncPolicy::PerWrite).unwrap();
            wal.append(&WalRecord::Insert {
                vector: vec![1.0, 2.0],
                fields: vec![("cat".to_string(), Value::Str("a".to_string()))],
            })
            .unwrap();
            wal.append(&WalRecord::Delete { id: 5 }).unwrap();
        }
        let records = WriteAheadLog::replay(&path).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[1], WalRecord::Delete { id: 5 });
    }

    #[test]
    fn replay_ignores_a_torn_tail() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let path = path_of(&file);
        {
            let mut wal = WriteAheadLog::open(&path, SyncPolicy::Group).unwrap();
            wal.append(&WalRecord::Delete { id: 1 }).unwrap();
            wal.append(&WalRecord::Delete { id: 2 }).unwrap();
            wal.sync().unwrap();
        }
        let mut raw = OpenOptions::new().append(true).open(&path).unwrap();
        raw.write_all(&100u32.to_le_bytes()).unwrap();
        raw.write_all(&0u32.to_le_bytes()).unwrap();
        raw.write_all(&[0u8; 10]).unwrap();
        drop(raw);

        let records = WriteAheadLog::replay(&path).unwrap();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn replay_stops_at_a_bad_crc() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let path = path_of(&file);
        {
            let mut wal = WriteAheadLog::open(&path, SyncPolicy::PerWrite).unwrap();
            wal.append(&WalRecord::Delete { id: 7 }).unwrap();
        }
        let mut raw = OpenOptions::new().append(true).open(&path).unwrap();
        raw.write_all(&4u32.to_le_bytes()).unwrap();
        raw.write_all(&12345u32.to_le_bytes()).unwrap();
        raw.write_all(&[9u8; 4]).unwrap();
        drop(raw);

        let records = WriteAheadLog::replay(&path).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0], WalRecord::Delete { id: 7 });
    }

    #[test]
    fn missing_log_replays_empty() {
        let records = WriteAheadLog::replay("/tmp/loupe-nonexistent-wal-xyz").unwrap();
        assert!(records.is_empty());
    }
}
