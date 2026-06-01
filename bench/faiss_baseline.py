"""FAISS HNSW adapter for Loupe's filtered-search benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_K = 10
DEFAULT_CANDIDATES = 512
DEFAULT_EF_SEARCH = 512
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_HNSW_M = 16
DEFAULT_SEED = 1

SELECTIVITY_LEVELS = (1.0, 0.5, 0.1, 0.01, 0.001)
CORRELATION_MODES = (
    "random",
    "positive",
    "anti_correlated",
    "clustered_by_tenant",
)
CSV_FIELDS = (
    "engine",
    "selectivity",
    "correlation",
    "runner",
    "strategy_chosen",
    "recall",
    "distance_computations",
    "latency_micros",
    "returned_count",
)


@dataclass(frozen=True)
class MatrixRow:
    engine: str
    selectivity: float
    correlation: str
    runner: str
    strategy_chosen: str
    recall: float
    distance_computations: float
    latency_micros: float
    returned_count: float

    def as_csv_row(self) -> dict:
        return {
            "engine": self.engine,
            "selectivity": f"{self.selectivity:.6f}",
            "correlation": self.correlation,
            "runner": self.runner,
            "strategy_chosen": self.strategy_chosen,
            "recall": f"{self.recall:.6f}",
            "distance_computations": f"{self.distance_computations:.3f}",
            "latency_micros": f"{self.latency_micros:.3f}",
            "returned_count": f"{self.returned_count:.3f}",
        }


@dataclass(frozen=True)
class MatrixConfig:
    k: int
    candidates: int
    ef_search: int
    seed: int


class XorShift64:
    def __init__(self, seed: int) -> None:
        self.state = seed | 1

    def next_u64(self) -> int:
        mask = (1 << 64) - 1
        value = self.state
        value ^= (value << 13) & mask
        value &= mask
        value ^= value >> 7
        value &= mask
        value ^= (value << 17) & mask
        value &= mask
        self.state = value
        return value


def read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    dim = int(raw[0])
    if dim <= 0:
        raise ValueError(f"{path} has an invalid vector dimension: {dim}")

    record_width = dim + 1
    if raw.size % record_width != 0:
        raise ValueError(f"{path} has a truncated fvecs record")

    records = raw.reshape((-1, record_width))
    dims = records[:, 0]
    if not np.all(dims == dim):
        raise ValueError(f"{path} mixes vector dimensions")

    vectors = records.view(np.float32)[:, 1:]
    return np.ascontiguousarray(vectors, dtype=np.float32)


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.int32)

    dim = int(raw[0])
    if dim <= 0:
        raise ValueError(f"{path} has an invalid row dimension: {dim}")

    record_width = dim + 1
    if raw.size % record_width != 0:
        raise ValueError(f"{path} has a truncated ivecs record")

    records = raw.reshape((-1, record_width))
    dims = records[:, 0]
    if not np.all(dims == dim):
        raise ValueError(f"{path} mixes row dimensions")

    return np.ascontiguousarray(records[:, 1:], dtype=np.int32)


def generate_filter(
    base: np.ndarray,
    query: np.ndarray,
    selectivity: float,
    correlation: str,
    seed: int,
) -> np.ndarray:
    count = base.shape[0]
    target = int(math.floor(count * selectivity + 0.5))
    target = max(0, min(target, count))

    if target == 0 or count == 0:
        return np.empty(0, dtype=np.int64)

    if correlation == "random":
        return random_ids(count, target, seed)

    if correlation == "positive":
        return nearest_ids(base, query, target, farthest=False)

    if correlation == "anti_correlated":
        return nearest_ids(base, query, target, farthest=True)

    if correlation == "clustered_by_tenant":
        center = base[seed % count]
        return nearest_ids(base, center, target, farthest=False)

    raise ValueError(f"unknown correlation mode: {correlation}")


def random_ids(count: int, target: int, seed: int) -> np.ndarray:
    ids = np.arange(count, dtype=np.int64)
    rng = XorShift64(seed)

    for index in range(target):
        remaining = count - index
        swap = index + rng.next_u64() % remaining
        ids[index], ids[swap] = ids[swap], ids[index]

    return np.sort(ids[:target])


def nearest_ids(
    base: np.ndarray,
    point: np.ndarray,
    target: int,
    farthest: bool,
) -> np.ndarray:
    distances = squared_l2(base, point)
    scores = -distances if farthest else distances

    if target >= base.shape[0]:
        selected = np.arange(base.shape[0], dtype=np.int64)
    else:
        selected = np.argpartition(scores, target - 1)[:target]

    return np.sort(selected.astype(np.int64, copy=False))


def squared_l2(base: np.ndarray, query: np.ndarray) -> np.ndarray:
    diff = base - query
    return np.einsum("ij,ij->i", diff, diff)


def exact_filtered_search(
    base: np.ndarray,
    query: np.ndarray,
    allowed_ids: np.ndarray,
    k: int,
) -> List[int]:
    if allowed_ids.size == 0 or k == 0:
        return []

    limit = min(k, allowed_ids.size)
    subset = base[allowed_ids]
    distances = squared_l2(subset, query)

    if limit == allowed_ids.size:
        candidate_positions = np.arange(allowed_ids.size)
    else:
        candidate_positions = np.argpartition(distances, limit - 1)[:limit]

    ordered_positions = candidate_positions[
        np.argsort(distances[candidate_positions], kind="stable")
    ]
    return [int(allowed_ids[position]) for position in ordered_positions]


def filtered_recall(results: Sequence[int], oracle: Sequence[int]) -> float:
    if not oracle:
        return 1.0

    wanted = set(oracle)
    hits = sum(1 for item in results if item in wanted)
    return hits / len(wanted)


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
) -> List[MatrixRow]:
    rows = []
    query_count = max(queries.shape[0], 1)
    index.hnsw.efSearch = config.ef_search

    for selectivity in SELECTIVITY_LEVELS:
        for correlation in CORRELATION_MODES:
            recall_sum = 0.0
            candidate_sum = 0.0
            latency_sum = 0.0
            returned_sum = 0.0

            for query in queries:
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

    return rows


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


def write_csv(path: Path, rows: Sequence[MatrixRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


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

    index = build_faiss_hnsw(base, args.hnsw_m, args.ef_construction)
    config = MatrixConfig(
        k=args.k,
        candidates=args.candidates,
        ef_search=args.ef_search,
        seed=args.seed,
    )
    rows = run_matrix(index, base, queries, config)
    write_csv(args.out, rows)
    print(f"faiss CSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
