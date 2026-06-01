"""Packages benchmark CSVs, SVGs, and run metadata into bench/results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import comparison_plots

DEFAULT_OUT_DIR = Path("bench/results")


@dataclass(frozen=True)
class PackagedResults:
    root: Path
    metadata_path: Path
    summary_path: Path
    csv_dir: Path
    planner_summary_dir: Path
    plot_dir: Path


@dataclass(frozen=True)
class PlannerSummaryRow:
    planner: str
    mean_recall: float
    mean_distance_computations: float
    regret: float
    regret_delta: float


def package_results(
    run_id: str,
    dataset: str,
    csv_paths: Sequence[Path],
    plot_dirs: Sequence[Path],
    out_dir: Path,
    notes: str,
    planner_summary_paths: Sequence[Path] = (),
) -> PackagedResults:
    safe_run_id = checked_run_id(run_id)
    root = out_dir / safe_run_id
    csv_dir = root / "csv"
    planner_summary_dir = root / "planner-summaries"
    plot_dir = root / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    planner_summary_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    copied_csvs = copy_csvs(csv_paths, csv_dir)
    copied_planner_summaries = copy_csvs(planner_summary_paths, planner_summary_dir)
    copied_plots = copy_plots(plot_dirs, plot_dir)
    rows = comparison_plots.read_csvs(copied_csvs)
    selected = comparison_plots.comparison_rows(rows)
    metadata = build_metadata(
        run_id=safe_run_id,
        dataset=dataset,
        copied_csvs=copied_csvs,
        copied_planner_summaries=copied_planner_summaries,
        copied_plots=copied_plots,
        notes=notes,
    )
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    summary_path = root / "summary.md"
    summary_path.write_text(summary_markdown(metadata, selected, copied_planner_summaries))

    return PackagedResults(
        root=root,
        metadata_path=metadata_path,
        summary_path=summary_path,
        csv_dir=csv_dir,
        planner_summary_dir=planner_summary_dir,
        plot_dir=plot_dir,
    )


def checked_run_id(run_id: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    if not run_id:
        raise ValueError("run id is required")

    if any(character not in allowed for character in run_id):
        raise ValueError("run id may only contain letters, numbers, dashes, and underscores")

    return run_id


def copy_csvs(paths: Sequence[Path], out_dir: Path) -> List[Path]:
    copied = []

    for path in paths:
        if path.suffix != ".csv":
            raise ValueError(f"expected a CSV path: {path}")

        destination = unique_destination(out_dir, path.name)
        shutil.copy2(path, destination)
        copied.append(destination)

    return copied


def copy_plots(plot_dirs: Sequence[Path], out_dir: Path) -> List[Path]:
    copied = []

    for directory in plot_dirs:
        for path in sorted(directory.glob("*.svg")):
            destination = unique_destination(out_dir, path.name)
            shutil.copy2(path, destination)
            copied.append(destination)

    return copied


def unique_destination(directory: Path, name: str) -> Path:
    destination = directory / name
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    index = 2

    while True:
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def build_metadata(
    run_id: str,
    dataset: str,
    copied_csvs: Sequence[Path],
    copied_planner_summaries: Sequence[Path],
    copied_plots: Sequence[Path],
    notes: str,
) -> Dict[str, object]:
    return {
        "run_id": run_id,
        "dataset": dataset,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "notes": notes,
        "hardware": hardware_metadata(),
        "tool_versions": tool_versions(),
        "artifacts": {
            "csv": [str(path.relative_to(path.parents[1])) for path in copied_csvs],
            "planner_summaries": [
                str(path.relative_to(path.parents[1]))
                for path in copied_planner_summaries
            ],
            "plots": [str(path.relative_to(path.parents[1])) for path in copied_plots],
        },
    }


def hardware_metadata() -> Dict[str, object]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes(),
    }


def memory_bytes() -> Optional[int]:
    if platform.system() == "Darwin":
        output = command_output(["sysctl", "-n", "hw.memsize"])
        if output is not None:
            return int(output)

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                return int(parts[1]) * 1024

    return None


def tool_versions() -> Dict[str, Optional[str]]:
    return {
        "cargo": first_line(["cargo", "--version"]),
        "rustc": first_line(["rustc", "--version"]),
        "docker": first_line(["docker", "--version"]),
        "python": first_line([sys.executable, "--version"]),
    }


def first_line(command: Sequence[str]) -> Optional[str]:
    output = command_output(command)
    if output is None:
        return None

    return output.splitlines()[0]


def command_output(command: Sequence[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    return completed.stdout.strip() or completed.stderr.strip()


def summary_markdown(
    metadata: Dict[str, object],
    rows: Sequence[comparison_plots.BenchmarkRow],
    planner_summary_paths: Sequence[Path] = (),
) -> str:
    summaries = comparison_plots.engine_summaries(rows)
    lines = [
        f"# Benchmark Run: {metadata['run_id']}",
        "",
        f"Dataset: `{metadata['dataset']}`",
        f"Created: `{metadata['created_at_utc']}`",
        "",
        "## Summary",
        "",
        "| Engine | Mean recall | Min recall | Hard-filter recall | Mean latency (us) | Under-return cells |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    grouped = group_rows(rows)
    for label, summary in summaries:
        group = grouped[label]
        min_recall = min(row.recall for row in group)
        hard_rows = [
            row
            for row in group
            if row.correlation == "anti_correlated" and row.selectivity <= 0.01
        ]
        hard_recall = mean([row.recall for row in hard_rows])
        under_return_cells = sum(1 for row in group if row.returned_count < 10.0)
        lines.append(
            f"| {label} | {summary['recall']:.3f} | {min_recall:.3f} | "
            f"{hard_recall:.3f} | {summary['latency_micros']:.1f} | {under_return_cells} |"
        )

    planner_rows = read_planner_summaries(planner_summary_paths)
    if planner_rows:
        lines.extend(
            [
                "",
                "## Planner Summary",
                "",
                "| Planner | Mean recall | Mean dist/query | Regret | Regret delta |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in planner_rows:
            lines.append(
                f"| {row.planner} | {row.mean_recall:.3f} | "
                f"{row.mean_distance_computations:.0f} | {row.regret:.3f} | "
                f"{row.regret_delta:.3f} |"
            )

        verdict = planner_verdict(planner_rows)
        if verdict:
            lines.extend(["", "Planner calibration: " + verdict])

    notes = str(metadata.get("notes", "")).strip()
    if notes:
        lines.extend(["", "## Notes", "", notes])

    lines.append("")
    return "\n".join(lines)


def read_planner_summaries(paths: Sequence[Path]) -> List[PlannerSummaryRow]:
    rows = []

    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for record in reader:
                rows.append(
                    PlannerSummaryRow(
                        planner=record["planner"],
                        mean_recall=float(record["mean_recall"]),
                        mean_distance_computations=float(
                            record["mean_distance_computations"]
                        ),
                        regret=float(record["regret"]),
                        regret_delta=float(record["regret_delta"]),
                    )
                )

    return rows


def planner_verdict(rows: Sequence[PlannerSummaryRow]) -> str:
    by_name = {row.planner: row for row in rows}
    rule = by_name.get("RuleLadder")
    cost = by_name.get("CostModel")
    if rule is None or cost is None:
        return ""

    recall_ok = cost.mean_recall >= rule.mean_recall
    distance_ok = cost.mean_distance_computations <= rule.mean_distance_computations
    regret_ok = cost.regret <= rule.regret
    strictly_better = (
        cost.mean_recall > rule.mean_recall
        or cost.mean_distance_computations < rule.mean_distance_computations
        or cost.regret < rule.regret
    )

    if recall_ok and distance_ok and regret_ok and strictly_better:
        return "CostModel beats RuleLadder on this run."
    if recall_ok and distance_ok and regret_ok:
        return "CostModel matches RuleLadder on this run."

    return "CostModel trails RuleLadder on this run; calibration still needs work."


def group_rows(
    rows: Sequence[comparison_plots.BenchmarkRow],
) -> Dict[str, List[comparison_plots.BenchmarkRow]]:
    grouped: Dict[str, List[comparison_plots.BenchmarkRow]] = {}

    for row in rows:
        grouped.setdefault(row.label, []).append(row)

    return grouped


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package benchmark CSVs, SVG plots, and reproducibility metadata."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--csv", action="append", required=True, type=Path)
    parser.add_argument("--planner-summary-csv", action="append", default=[], type=Path)
    parser.add_argument("--plot-dir", action="append", default=[], type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("--notes", default="")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Copies benchmark artifacts and writes metadata plus a Markdown summary."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    packaged = package_results(
        run_id=args.run_id,
        dataset=args.dataset,
        csv_paths=args.csv,
        plot_dirs=args.plot_dir,
        out_dir=args.out_dir,
        notes=args.notes,
        planner_summary_paths=args.planner_summary_csv,
    )

    print(f"result bundle: {packaged.root}")
    print(f"metadata: {packaged.metadata_path}")
    print(f"summary: {packaged.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
