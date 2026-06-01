"""Combined SVG plots for Loupe and external benchmark CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

WIDTH = 980
HEIGHT = 580
LEFT = 120
RIGHT = 50
TOP = 70
BOTTOM = 90

ENGINE_ORDER = (
    "Loupe Planner",
    "FAISS",
    "Qdrant",
    "pgvector",
    "Weaviate",
)
CORRELATION_ORDER = (
    "random",
    "positive",
    "anti_correlated",
    "clustered_by_tenant",
)
SELECTIVITY_ORDER = (1.0, 0.5, 0.1, 0.01, 0.001)
HARD_SELECTIVITY_ORDER = (0.01, 0.001)
COLORS = {
    "Loupe Planner": "#0f172a",
    "FAISS": "#dc2626",
    "Qdrant": "#2563eb",
    "pgvector": "#16a34a",
    "Weaviate": "#7c3aed",
}


@dataclass(frozen=True)
class BenchmarkRow:
    engine: str
    selectivity: float
    correlation: str
    runner: str
    strategy_chosen: str
    recall: float
    distance_computations: float
    latency_micros: float
    returned_count: float

    @property
    def label(self) -> str:
        if self.engine == "loupe" and self.runner == "Planner":
            return "Loupe Planner"

        if self.engine == "faiss":
            return "FAISS"

        if self.engine == "qdrant":
            return "Qdrant"

        if self.engine == "pgvector":
            return "pgvector"

        if self.engine == "weaviate":
            return "Weaviate"

        return self.engine


@dataclass(frozen=True)
class PlotFiles:
    recall_by_engine: Path
    latency_by_engine: Path
    hard_filter_recall: Path
    faiss_under_return: Path
    planner_vs_external: Path


def read_csvs(paths: Sequence[Path]) -> List[BenchmarkRow]:
    rows = []

    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                row = BenchmarkRow(
                    engine=raw["engine"],
                    selectivity=float(raw["selectivity"]),
                    correlation=raw["correlation"],
                    runner=raw["runner"],
                    strategy_chosen=raw["strategy_chosen"],
                    recall=float(raw["recall"]),
                    distance_computations=float(raw["distance_computations"]),
                    latency_micros=float(raw["latency_micros"]),
                    returned_count=float(raw["returned_count"]),
                )
                rows.append(row)

    return rows


def comparison_rows(rows: Sequence[BenchmarkRow]) -> List[BenchmarkRow]:
    selected = []

    for row in rows:
        if row.engine == "loupe" and row.runner != "Planner":
            continue
        selected.append(row)

    return selected


def write_comparison_plots(out_dir: Path, rows: Sequence[BenchmarkRow]) -> PlotFiles:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = comparison_rows(rows)

    files = PlotFiles(
        recall_by_engine=out_dir / "recall-by-engine.svg",
        latency_by_engine=out_dir / "latency-by-engine.svg",
        hard_filter_recall=out_dir / "hard-filter-recall.svg",
        faiss_under_return=out_dir / "faiss-under-return.svg",
        planner_vs_external=out_dir / "loupe-vs-external.svg",
    )

    files.recall_by_engine.write_text(recall_by_engine_svg(selected))
    files.latency_by_engine.write_text(latency_by_engine_svg(selected))
    files.hard_filter_recall.write_text(hard_filter_recall_svg(selected))
    files.faiss_under_return.write_text(faiss_under_return_svg(selected))
    files.planner_vs_external.write_text(planner_vs_external_svg(selected))

    return files


def recall_by_engine_svg(rows: Sequence[BenchmarkRow]) -> str:
    summaries = engine_summaries(rows)
    values = [(label, summary["recall"]) for label, summary in summaries]

    return bar_chart_svg(
        title="Average Recall by Engine",
        subtitle="Loupe uses the Planner row; external engines use their native filtered-search row",
        values=values,
        y_label="mean recall",
        value_format="{:.3f}",
        max_value=1.0,
    )


def latency_by_engine_svg(rows: Sequence[BenchmarkRow]) -> str:
    summaries = engine_summaries(rows)
    values = [(label, summary["latency_micros"]) for label, summary in summaries]
    max_value = max((value for _, value in values), default=1.0)

    return bar_chart_svg(
        title="Average Latency by Engine",
        subtitle="microseconds per query over the 20 selectivity x correlation cells",
        values=values,
        y_label="mean latency (us)",
        value_format="{:.0f}",
        max_value=max_value,
    )


def hard_filter_recall_svg(rows: Sequence[BenchmarkRow]) -> str:
    body = svg_start(
        "Hard Filter Recall",
        "anti-correlated filters at 1% and 0.1% selectivity",
    )
    labels = ordered_labels(rows)
    cell_width = 145
    cell_height = 64
    start_x = 160
    start_y = 130

    for col, label in enumerate(labels):
        x = start_x + col * cell_width
        body.append(
            f'<text x="{x}" y="104" class="label">{escape(label)}</text>'
        )

    for row_index, selectivity in enumerate(HARD_SELECTIVITY_ORDER):
        y = start_y + row_index * 90
        body.append(
            f'<text x="34" y="{y + 38}" class="label">{selectivity * 100:.1f}%</text>'
        )
        for col, label in enumerate(labels):
            x = start_x + col * cell_width
            match = find_row(rows, label, selectivity, "anti_correlated")
            recall = 0.0 if match is None else match.recall
            draw_heat_cell(body, x, y, 118, cell_height, recall, f"{recall:.2f}")

    body.append(svg_end())
    return "\n".join(body)


def faiss_under_return_svg(rows: Sequence[BenchmarkRow]) -> str:
    body = svg_start(
        "FAISS Under-Return Cells",
        "returned count out of k=10 for FAISS HNSW post-filtering",
    )
    faiss_rows = [row for row in rows if row.label == "FAISS"]
    cell_width = 145
    cell_height = 54
    start_x = 160
    start_y = 130

    for col, selectivity in enumerate(SELECTIVITY_ORDER):
        x = start_x + col * cell_width
        body.append(
            f'<text x="{x}" y="104" class="label">{selectivity * 100:.1f}%</text>'
        )

    for row_index, correlation in enumerate(CORRELATION_ORDER):
        y = start_y + row_index * 76
        body.append(
            f'<text x="22" y="{y + 34}" class="label">{escape(correlation)}</text>'
        )
        for col, selectivity in enumerate(SELECTIVITY_ORDER):
            x = start_x + col * cell_width
            match = find_row(faiss_rows, "FAISS", selectivity, correlation)
            returned = 0.0 if match is None else match.returned_count
            intensity = min(returned / 10.0, 1.0)
            draw_heat_cell(body, x, y, 118, cell_height, intensity, f"{returned:.1f}")

    body.append(svg_end())
    return "\n".join(body)


def planner_vs_external_svg(rows: Sequence[BenchmarkRow]) -> str:
    summaries = engine_summaries(rows)
    max_latency = max((summary["latency_micros"] for _, summary in summaries), default=1.0)
    body = svg_start(
        "Loupe Planner vs External Engines",
        "higher is better on recall; lower is better on latency",
    )
    body.append(axes("mean latency (us)", "mean recall"))

    for label, summary in summaries:
        x = LEFT + (summary["latency_micros"] / max_latency) * plot_width()
        y = TOP + (1.0 - summary["recall"]) * plot_height()
        color = COLORS.get(label, "#334155")
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}" />')
        body.append(
            f'<text x="{x + 12:.1f}" y="{y + 4:.1f}" class="label">'
            f'{escape(label)} ({summary["recall"]:.3f})</text>'
        )

    body.append(svg_end())
    return "\n".join(body)


def engine_summaries(rows: Sequence[BenchmarkRow]) -> List[Tuple[str, Dict[str, float]]]:
    grouped: Dict[str, List[BenchmarkRow]] = {}

    for row in rows:
        grouped.setdefault(row.label, []).append(row)

    summaries = []
    for label in ordered_labels(rows):
        label_rows = grouped.get(label, [])
        if not label_rows:
            continue

        count = len(label_rows)
        summaries.append(
            (
                label,
                {
                    "recall": sum(row.recall for row in label_rows) / count,
                    "latency_micros": sum(row.latency_micros for row in label_rows) / count,
                    "returned_count": sum(row.returned_count for row in label_rows) / count,
                },
            )
        )

    return summaries


def ordered_labels(rows: Sequence[BenchmarkRow]) -> List[str]:
    present = {row.label for row in rows}
    labels = [label for label in ENGINE_ORDER if label in present]
    extra = sorted(present.difference(labels))
    labels.extend(extra)
    return labels


def find_row(
    rows: Sequence[BenchmarkRow],
    label: str,
    selectivity: float,
    correlation: str,
) -> Optional[BenchmarkRow]:
    for row in rows:
        if row.label != label:
            continue
        if row.correlation != correlation:
            continue
        if nearly_equal(row.selectivity, selectivity):
            return row

    return None


def bar_chart_svg(
    title: str,
    subtitle: str,
    values: Sequence[Tuple[str, float]],
    y_label: str,
    value_format: str,
    max_value: float,
) -> str:
    body = svg_start(title, subtitle)
    body.append(axes("", y_label))

    if not values:
        body.append('<text x="120" y="130" class="label">no data</text>')
        body.append(svg_end())
        return "\n".join(body)

    safe_max = max(max_value, 0.000001)
    slot_width = plot_width() / len(values)
    bar_width = min(96.0, slot_width * 0.58)

    for index, (label, value) in enumerate(values):
        bar_height = (value / safe_max) * plot_height()
        x = LEFT + index * slot_width + (slot_width - bar_width) / 2.0
        y = HEIGHT - BOTTOM - bar_height
        color = COLORS.get(label, "#334155")
        text = value_format.format(value)
        body.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" rx="4" fill="{color}" />'
        )
        body.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" '
            f'class="cell-label" text-anchor="middle">{escape(text)}</text>'
        )
        body.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{HEIGHT - 42}" '
            f'class="label" text-anchor="middle">{escape(label)}</text>'
        )

    body.append(svg_end())
    return "\n".join(body)


def draw_heat_cell(
    body: List[str],
    x: float,
    y: float,
    width: float,
    height: float,
    intensity: float,
    label: str,
) -> None:
    fill = heat_color(intensity)
    body.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="6" fill="{fill}" stroke="#cbd5e1" />'
    )
    body.append(
        f'<text x="{x + width / 2:.1f}" y="{y + height / 2 + 4:.1f}" '
        f'class="cell-label" text-anchor="middle">{escape(label)}</text>'
    )


def axes(x_label: str, y_label: str) -> str:
    x1 = LEFT
    y1 = HEIGHT - BOTTOM
    x2 = WIDTH - RIGHT
    y2 = TOP
    x_text = WIDTH / 2 - 70
    y_text = HEIGHT - 20
    rotated_y = HEIGHT / 2 + 90
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y1}" class="axis" />\n'
        f'<line x1="{x1}" y1="{y1}" x2="{x1}" y2="{y2}" class="axis" />\n'
        f'<text x="{x_text:.1f}" y="{y_text:.1f}" class="axis-label">{escape(x_label)}</text>\n'
        f'<text x="20" y="{rotated_y:.1f}" class="axis-label" '
        f'transform="rotate(-90 20,{rotated_y:.1f})">{escape(y_label)}</text>'
    )


def svg_start(title: str, subtitle: str) -> List[str]:
    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}">'
        ),
        (
            "<style>"
            ".title { font: 700 24px system-ui, sans-serif; fill: #0f172a; }"
            ".subtitle { font: 14px system-ui, sans-serif; fill: #475569; }"
            ".label { font: 12px system-ui, sans-serif; fill: #334155; }"
            ".cell-label { font: 12px system-ui, sans-serif; fill: #0f172a; }"
            ".axis { stroke: #475569; stroke-width: 1.5; }"
            ".axis-label { font: 13px system-ui, sans-serif; fill: #334155; }"
            "</style>"
        ),
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text x="{LEFT}" y="34" class="title">{escape(title)}</text>',
        f'<text x="{LEFT}" y="56" class="subtitle">{escape(subtitle)}</text>',
    ]


def svg_end() -> str:
    return "</svg>\n"


def plot_width() -> float:
    return WIDTH - LEFT - RIGHT


def plot_height() -> float:
    return HEIGHT - TOP - BOTTOM


def heat_color(value: float) -> str:
    clamped = max(0.0, min(value, 1.0))
    red = 254 - clamped * 140
    green = 226 - clamped * 29
    blue = 226 - clamped * 132
    return f"rgb({red:.0f},{green:.0f},{blue:.0f})"


def nearly_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=0.000001)


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write combined SVG plots from Loupe and external benchmark CSVs."
    )
    parser.add_argument("csv_paths", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Reads benchmark CSVs, writes combined SVG plots, and prints the file paths."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = read_csvs(args.csv_paths)
    files = write_comparison_plots(args.out_dir, rows)

    for path in [
        files.recall_by_engine,
        files.latency_by_engine,
        files.hard_filter_recall,
        files.faiss_under_return,
        files.planner_vs_external,
    ]:
        print(f"plot: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
