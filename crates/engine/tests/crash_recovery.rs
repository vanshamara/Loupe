use std::process::Command;
use std::thread::sleep;
use std::time::{Duration, Instant};

use loupe_distance::Metric;
use loupe_engine::{Collection, Expr, Value};

/// Spawns the crasher, waits until it has actually written a batch of durable
/// records, then hard-kills it (SIGKILL on Unix) while it is still inserting. The
/// collection is reopened and must come back with the committed records intact and
/// searchable, with no corruption from the torn final write.
#[test]
fn survives_a_hard_kill_mid_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_str().unwrap().to_string();
    let wal_path = format!("{path}/wal.log");

    let mut child = Command::new(env!("CARGO_BIN_EXE_wal_crasher"))
        .arg(&path)
        .spawn()
        .expect("failed to spawn crasher");

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        sleep(Duration::from_millis(50));
        let written = std::fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);
        if written > 4000 || Instant::now() > deadline {
            break;
        }
    }

    child.kill().expect("failed to kill crasher");
    let _ = child.wait();

    let recovered = Collection::open(&path, 2, Metric::L2).expect("failed to reopen collection");
    assert!(!recovered.is_empty(), "recovered nothing after a hard kill");

    let filter = Expr::Eq("tenant".into(), Value::Str("t".into()));
    let (results, _) = recovered.search(&[1.0, 1.0], &filter, 5);
    assert!(
        !results.is_empty(),
        "filtered search returned nothing after recovery"
    );
}
