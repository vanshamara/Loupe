#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON:-python3}"
work_dir="${LOUPE_WORKFLOW_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/loupe-workflow.XXXXXX")}"
total_count="${LOUPE_WORKFLOW_TOTAL:-5000}"
target_count="${LOUPE_WORKFLOW_TARGET:-2500}"
query_count="${LOUPE_WORKFLOW_QUERIES:-8}"
delete_count="${LOUPE_WORKFLOW_DELETES:-8}"

base_path="$work_dir/base.jsonl"
query_path="$work_dir/query.jsonl"
feedback_path="$work_dir/feedback.jsonl"
collection_dir="$work_dir/collection"
query_log_path="$work_dir/query-log.jsonl"
planner_summary_path="$work_dir/planner-summary.csv"
health_report_path="$work_dir/retrieval-health.md"
explain_before_path="$work_dir/explain-before.txt"
explain_after_path="$work_dir/explain-after-compact.txt"
stats_before_path="$work_dir/stats-before-delete.txt"
stats_deleted_path="$work_dir/stats-after-delete.txt"
stats_compacted_path="$work_dir/stats-after-compact.txt"
delete_log_path="$work_dir/delete.log"
compact_log_path="$work_dir/compact.log"
loupe_bin="target/debug/loupe"

generate_workload() {
  "$python_bin" - "$base_path" "$query_path" "$feedback_path" "$total_count" "$target_count" "$query_count" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
query_path = Path(sys.argv[2])
feedback_path = Path(sys.argv[3])
total_count = int(sys.argv[4])
target_count = int(sys.argv[5])
query_count = int(sys.argv[6])

base_path.parent.mkdir(parents=True, exist_ok=True)

with base_path.open("w") as handle:
    for idx in range(total_count):
        is_target = idx < target_count
        if is_target:
            embedding = [0.0, 1.0 + idx * 0.00001, 0.0, 0.0]
            category = "target"
            doc_type = "contract"
        else:
            embedding = [1.0 + (idx - target_count) * 0.00001, 0.0, 0.0, 0.0]
            category = "other"
            doc_type = "note"
        record = {
            "embedding": embedding,
            "category": category,
            "year": 2026,
            "doc_type": doc_type,
        }
        handle.write(json.dumps(record) + "\n")

with query_path.open("w") as handle:
    for idx in range(query_count):
        record = {
            "embedding": [1.0, idx * 0.00001, 0.0, 0.0],
            "category": "query",
            "year": 2026,
            "doc_type": "query",
        }
        handle.write(json.dumps(record) + "\n")

feedback_record = {
    "query_id": 1,
    "k": 10,
    "latency_micros": 10,
    "returned": [1, 2, 3],
    "metrics": {
        "distance_computations": 100,
        "nodes_visited": 100,
        "ef_expansions": 0,
        "tombstoned_visits": 0,
    },
    "filter": "category=target",
    "plan": "Acorn+AdaptiveEf",
    "recall_smoke": 0.2,
}
with feedback_path.open("w") as handle:
    for _ in range(8):
        handle.write(json.dumps(feedback_record) + "\n")
PY
}

run_loupe() {
  "$loupe_bin" "$@"
}

delete_vectors() {
  : > "$delete_log_path"
  if [ "$delete_count" -le 0 ]; then
    return
  fi

  for id in $(seq 0 "$((delete_count - 1))"); do
    run_loupe delete --dir "$collection_dir" --id "$id" >> "$delete_log_path"
  done
}

rm -rf "$collection_dir"
rm -f \
  "$base_path" \
  "$query_path" \
  "$feedback_path" \
  "$query_log_path" \
  "$planner_summary_path" \
  "$health_report_path" \
  "$explain_before_path" \
  "$explain_after_path" \
  "$stats_before_path" \
  "$stats_deleted_path" \
  "$stats_compacted_path" \
  "$delete_log_path" \
  "$compact_log_path"

mkdir -p "$work_dir"

echo "workflow dir: $work_dir"
echo "generating skewed filtered workload"
generate_workload

echo "building loupe CLI"
cargo build -p loupe-cli --bin loupe

echo "building durable collection"
run_loupe build --base "$base_path" --dir "$collection_dir"

echo "running EXPLAIN on hard filtered query"
run_loupe explain --dir "$collection_dir" --query "$query_path" --index 0 --filter "category=target" > "$explain_before_path"

echo "replaying workload with planner comparison"
run_loupe replay \
  --dir "$collection_dir" \
  --query "$query_path" \
  --filter "category=target" \
  --out "$query_log_path" \
  --compare-planners \
  --planner-summary-csv "$planner_summary_path" \
  --feedback-log "$feedback_path"

echo "writing retrieval-health report"
run_loupe advise \
  --log "$query_log_path" \
  --markdown-out "$health_report_path" \
  --planner-summary-csv "$planner_summary_path"

echo "recording collection stats"
run_loupe stats --dir "$collection_dir" > "$stats_before_path"

echo "deleting vectors to create tombstones"
delete_vectors
run_loupe stats --dir "$collection_dir" > "$stats_deleted_path"

echo "compacting collection"
run_loupe compact --dir "$collection_dir" > "$compact_log_path"
run_loupe stats --dir "$collection_dir" > "$stats_compacted_path"

echo "re-running EXPLAIN after compaction"
run_loupe explain --dir "$collection_dir" --query "$query_path" --index 0 --filter "category=target" > "$explain_after_path"

echo
echo "canonical workflow complete"
echo "  explain before:    $explain_before_path"
echo "  query log:         $query_log_path"
echo "  planner summary:   $planner_summary_path"
echo "  health report:     $health_report_path"
echo "  stats before:      $stats_before_path"
echo "  stats after delete:$stats_deleted_path"
echo "  compact log:       $compact_log_path"
echo "  stats compacted:   $stats_compacted_path"
echo "  explain compacted: $explain_after_path"
