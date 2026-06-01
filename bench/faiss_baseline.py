"""FAISS HNSW adapter for Loupe's filtered-search benchmark matrix."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.benchmark_common import (
    CORRELATION_MODES,
    DEFAULT_CANDIDATES,
    DEFAULT_EF_SEARCH,
    DEFAULT_K,
    DEFAULT_SEED,
    MatrixConfig,
    MatrixRow,
    SELECTIVITY_LEVELS,
    exact_filtered_search,
    filtered_recall,
    generate_filter,
    limit_query_rows,
    read_fvecs,
    read_ivecs,
    write_csv,
)

DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_HNSW_M = 16


def build_faiss_hnsw(
    base: np.ndarray,
    hnsw_m: int,
    ef_construction: int,
):
    faiss = require_faiss()
    dim = base.shape[1]
    index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    index.add(base)
    return index


def require_faiss():
    try:
        import faiss
    except ImportError as error:
        message = (
            "faiss-cpu is required for this benchmark. "
            "Install it with: python3 -m pip install -r scripts/requirements-faiss.txt"
        )
        raise SystemExit(message) from error

    return faiss


def run_matrix(
    index,
    base: np.ndarray,
    queries: np.ndarray,
    config: MatrixConfig,
    progress_every_queries: int = 0,
) -> List[MatrixRow]:
    rows = []
    query_count = max(queries.shape[0], 1)
    total_cells = len(SELECTIVITY_LEVELS) * len(CORRELATION_MODES)
    cell_index = 0
    index.hnsw.efSearch = config.ef_search

    for selectivity in SELECTIVITY_LEVELS:
        for correlation in CORRELATION_MODES:
            cell_index += 1
            cell_start = time.perf_counter()
            print_matrix_cell_start(cell_index, total_cells, selectivity, correlation)

            recall_sum = 0.0
            candidate_sum = 0.0
            latency_sum = 0.0
            returned_sum = 0.0

            for query_index, query in enumerate(queries, start=1):
                allowed_ids = generate_filter(
                    base,
                    query,
                    selectivity,
                    correlation,
                    config.seed,
                )
                oracle = exact_filtered_search(base, query, allowed_ids, config.k)
                results, candidates, latency_micros = search_faiss_post_filter(
                    index,
                    query,
                    allowed_ids,
                    config.k,
                    config.candidates,
                )
                recall_sum += filtered_recall(results, oracle)
                candidate_sum += candidates
                latency_sum += latency_micros
                returned_sum += len(results)
                print_matrix_query_progress(
                    cell_index,
                    total_cells,
                    query_index,
                    queries.shape[0],
                    progress_every_queries,
                    time.perf_counter() - cell_start,
                )

            rows.append(
                MatrixRow(
                    engine="faiss",
                    selectivity=selectivity,
                    correlation=correlation,
                    runner="FaissHnswPostFilter",
                    strategy_chosen="HnswPostFilter",
                    recall=recall_sum / query_count,
                    distance_computations=candidate_sum / query_count,
                    latency_micros=latency_sum / query_count,
                    returned_count=returned_sum / query_count,
                )
            )
            print_matrix_cell_done(
                cell_index,
                total_cells,
                time.perf_counter() - cell_start,
            )

    return rows


def print_matrix_cell_start(
    cell_index: int,
    total_cells: int,
    selectivity: float,
    correlation: str,
) -> None:
    print(
        f"[faiss matrix] cell {cell_index}/{total_cells} start: selectivity={selectivity * 100.0:.1f}% correlation={correlation}",
        flush=True,
    )


def print_matrix_query_progress(
    cell_index: int,
    total_cells: int,
    completed_queries: int,
    total_queries: int,
    progress_every_queries: int,
    elapsed_seconds: float,
) -> None:
    if progress_every_queries <= 0 or total_queries == 0:
        return

    should_print = (
        completed_queries == total_queries
        or completed_queries % progress_every_queries == 0
    )
    if not should_print:
        return

    print(
        f"[faiss matrix] cell {cell_index}/{total_cells}: query {completed_queries}/{total_queries}, elapsed {elapsed_seconds:.1f}s",
        flush=True,
    )


def print_matrix_cell_done(
    cell_index: int,
    total_cells: int,
    elapsed_seconds: float,
) -> None:
    print(
        f"[faiss matrix] cell {cell_index}/{total_cells} done in {elapsed_seconds:.1f}s",
        flush=True,
    )


def search_faiss_post_filter(
    index,
    query: np.ndarray,
    allowed_ids: np.ndarray,
    k: int,
    candidates: int,
) -> Tuple[List[int], int, float]:
    query_matrix = np.ascontiguousarray(query.reshape((1, -1)), dtype=np.float32)
    search_size = max(k, candidates)

    start = time.perf_counter_ns()
    _, labels = index.search(query_matrix, search_size)
    latency_micros = (time.perf_counter_ns() - start) / 1000.0

    valid_labels = [int(label) for label in labels[0] if int(label) >= 0]
    allowed = set(int(item) for item in allowed_ids)
    results = []

    for label in valid_labels:
        if label in allowed:
            results.append(label)
        if len(results) == k:
            break

    return results, len(valid_labels), latency_micros


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FAISS HNSW post-filter over Loupe's benchmark matrix."
    )
    parser.add_argument("base_path", type=Path)
    parser.add_argument("query_path", type=Path)
    parser.add_argument("groundtruth_path", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--k", default=DEFAULT_K, type=int)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES, type=int)
    parser.add_argument("--ef-search", default=DEFAULT_EF_SEARCH, type=int)
    parser.add_argument("--ef-construction", default=DEFAULT_EF_CONSTRUCTION, type=int)
    parser.add_argument("--hnsw-m", default=DEFAULT_HNSW_M, type=int)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--query-limit", type=int)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Builds a FAISS HNSW index, runs filtered post-filter cells, and writes CSV."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    base = read_fvecs(args.base_path)
    queries = read_fvecs(args.query_path)
    ground_truth = read_ivecs(args.groundtruth_path)

    if base.size == 0:
        raise SystemExit("base vectors are empty")

    if queries.shape[0] != ground_truth.shape[0]:
        raise SystemExit("query and ground-truth row counts differ")

    queries, ground_truth = limit_query_rows(queries, ground_truth, args.query_limit)
    index = build_faiss_hnsw(base, args.hnsw_m, args.ef_construction)
    config = MatrixConfig(
        k=args.k,
        candidates=args.candidates,
        ef_search=args.ef_search,
        seed=args.seed,
    )
    rows = run_matrix(index, base, queries, config, progress_every_queries=10)
    write_csv(args.out, rows)
    print(f"faiss CSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
