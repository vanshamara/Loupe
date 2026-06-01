"""Qdrant adapter for Loupe's filtered-search benchmark matrix."""

from __future__ import annotations

import argparse
import sys
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
from bench.http_common import JsonClient, timed_call

DEFAULT_COLLECTION = "loupe_bench_qdrant"
DEFAULT_URL = "http://localhost:6333"
DEFAULT_BATCH_SIZE = 256
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_HNSW_M = 16
DEFAULT_TIMEOUT_SECS = 30.0


def reset_collection(
    client: JsonClient,
    collection: str,
    dim: int,
    hnsw_m: int,
    ef_construction: int,
) -> None:
    client.request("DELETE", f"/collections/{collection}", expected=(200, 202, 404))
    payload = {
        "vectors": {
            "size": dim,
            "distance": "Euclid",
        },
        "hnsw_config": {
            "m": hnsw_m,
            "ef_construct": ef_construction,
        },
    }
    client.request("PUT", f"/collections/{collection}", payload, expected=(200, 201))


def upload_points(
    client: JsonClient,
    collection: str,
    base: np.ndarray,
    batch_size: int,
) -> None:
    for start in range(0, base.shape[0], batch_size):
        points = []
        batch = base[start : start + batch_size]
        for offset, vector in enumerate(batch):
            vector_id = start + offset
            points.append(
                {
                    "id": vector_id,
                    "vector": vector.astype(float).tolist(),
                    "payload": {
                        "vector_id": vector_id,
                    },
                }
            )

        payload = {
            "points": points,
        }
        client.request(
            "PUT",
            f"/collections/{collection}/points?wait=true",
            payload,
            expected=(200, 202),
        )


def qdrant_filter(allowed_ids: np.ndarray, total_count: int) -> Optional[dict]:
    if allowed_ids.size == total_count:
        return None

    return {
        "must": [
            {
                "has_id": [int(item) for item in allowed_ids],
            }
        ]
    }


def search_qdrant(
    client: JsonClient,
    collection: str,
    query: np.ndarray,
    allowed_ids: np.ndarray,
    total_count: int,
    k: int,
    candidates: int,
    ef_search: int,
) -> Tuple[List[int], int, float]:
    payload = {
        "vector": query.astype(float).tolist(),
        "limit": max(k, candidates),
        "with_payload": True,
        "params": {
            "hnsw_ef": ef_search,
        },
    }
    filter_payload = qdrant_filter(allowed_ids, total_count)
    if filter_payload is not None:
        payload["filter"] = filter_payload

    response, latency_micros = timed_call(
        lambda: client.request(
            "POST",
            f"/collections/{collection}/points/search",
            payload,
            expected=(200,),
        )
    )
    candidates_found = [int(point["id"]) for point in response.get("result", [])]
    return candidates_found[:k], len(candidates_found), latency_micros


def run_matrix(
    client: JsonClient,
    collection: str,
    base: np.ndarray,
    queries: np.ndarray,
    config: MatrixConfig,
) -> List[MatrixRow]:
    rows = []
    query_count = max(queries.shape[0], 1)

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
                results, candidates, latency_micros = search_qdrant(
                    client,
                    collection,
                    query,
                    allowed_ids,
                    base.shape[0],
                    config.k,
                    config.candidates,
                    config.ef_search,
                )
                recall_sum += filtered_recall(results, oracle)
                candidate_sum += candidates
                latency_sum += latency_micros
                returned_sum += len(results)

            rows.append(
                MatrixRow(
                    engine="qdrant",
                    selectivity=selectivity,
                    correlation=correlation,
                    runner="QdrantFilteredHnsw",
                    strategy_chosen="NativeFilteredHnsw",
                    recall=recall_sum / query_count,
                    distance_computations=candidate_sum / query_count,
                    latency_micros=latency_sum / query_count,
                    returned_count=returned_sum / query_count,
                )
            )

    return rows


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qdrant filtered HNSW over Loupe's benchmark matrix."
    )
    parser.add_argument("base_path", type=Path)
    parser.add_argument("query_path", type=Path)
    parser.add_argument("groundtruth_path", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--k", default=DEFAULT_K, type=int)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES, type=int)
    parser.add_argument("--ef-search", default=DEFAULT_EF_SEARCH, type=int)
    parser.add_argument("--ef-construction", default=DEFAULT_EF_CONSTRUCTION, type=int)
    parser.add_argument("--hnsw-m", default=DEFAULT_HNSW_M, type=int)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--timeout-secs", default=DEFAULT_TIMEOUT_SECS, type=float)
    parser.add_argument("--query-limit", type=int)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Loads vectors into Qdrant, runs native filtered search cells, and writes CSV."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    base = read_fvecs(args.base_path)
    queries = read_fvecs(args.query_path)
    ground_truth = read_ivecs(args.groundtruth_path)

    if base.size == 0:
        raise SystemExit("base vectors are empty")

    if queries.shape[0] != ground_truth.shape[0]:
        raise SystemExit("query and ground-truth row counts differ")

    queries, ground_truth = limit_query_rows(queries, ground_truth, args.query_limit)
    client = JsonClient(args.url, args.timeout_secs)
    client.wait_until_ready("/readyz", attempts=60, sleep_secs=1.0)
    reset_collection(
        client,
        args.collection,
        base.shape[1],
        args.hnsw_m,
        args.ef_construction,
    )
    upload_points(client, args.collection, base, args.batch_size)

    config = MatrixConfig(
        k=args.k,
        candidates=args.candidates,
        ef_search=args.ef_search,
        seed=args.seed,
    )
    rows = run_matrix(client, args.collection, base, queries, config)
    write_csv(args.out, rows)
    print(f"qdrant CSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
