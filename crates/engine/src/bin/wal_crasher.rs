use std::env;

use loupe_distance::Metric;
use loupe_engine::{Collection, Value};

/// Opens a durable collection in the given directory and inserts forever, flushing
/// each write to the log. It never exits on its own; the crash test kills it
/// mid-write to check that recovery survives a hard kill.
fn main() {
    let dir = env::args().nth(1).expect("usage: wal_crasher <dir>");
    let mut collection = Collection::open(&dir, 2, Metric::L2).expect("failed to open collection");
    let mut counter: u32 = 0;
    loop {
        let value = counter as f32;
        collection.insert(&[value, value], &[("tenant", Value::Str("t".to_string()))]);
        counter = counter.wrapping_add(1);
    }
}
