"""Weaviate adapter for Loupe's filtered-search benchmark matrix."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
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

DEFAULT_BATCH_SIZE = 100
DEFAULT_CLASS_NAME = "LoupeBenchWeaviate"
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_HNSW_M = 16
DEFAULT_TIMEOUT_SECS = 30.0
DEFAULT_URL = "http://localhost:8080"
UUID_NAMESPACE = uuid.UUID("73334252-6874-4473-9d87-a5fb72916c0b")


def reset_class(
    client: JsonClient,
    class_name: str,
    hnsw_m: int,
    ef_construction: int,
    ef_search: int,
) -> None:
    client.request("DELETE", f"/v1/schema/{class_name}", expected=(200, 202, 404))
    payload = {
        "class": class_name,
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "distance": "l2-squared",
            "ef": ef_search,
            "efConstruction": ef_construction,
            "maxConnections": hnsw_m,
        },
        "properties": [
            {
                "name": "vector_id",
                "dataType": ["int"],
            },
            {
                "name": "filter_ids",
                "dataType": ["int[]"],
            }
        ],
    }
    client.request("POST", "/v1/schema", payload, expected=(200, 201))


def upload_objects(
    client: JsonClient,
    class_name: str,
    base: np.ndarray,
    batch_size: int,
) -> None:
    for start in range(0, base.shape[0], batch_size):
        objects = []
        batch = base[start : start + batch_size]
        for offset, vector in enumerate(batch):
            vector_id = start + offset
            objects.append(
                {
                    "class": class_name,
                    "id": str(uuid.uuid5(UUID_NAMESPACE, str(vector_id))),
                    "properties": {
                        "vector_id": vector_id,
                        "filter_ids": [vector_id],
                    },
                    "vector": vector.astype(float).tolist(),
                }
            )

        payload = {
            "objects": objects,
        }
        response = client.request(
            "POST",
            "/v1/batch/objects",
            payload,
            expected=(200,),
        )
        items = response if isinstance(response, list) else response.get("objects", [])
        errors = [
            item
            for item in items
            if item.get("result", {}).get("errors") is not None
        ]
        if errors:
            raise RuntimeError(f"Weaviate batch insert returned errors: {errors[:3]}")


def weaviate_where(allowed_ids: np.ndarray, total_count: int) -> Optional[str]:
    if allowed_ids.size == total_count:
        return None

    values = ", ".join(str(int(item)) for item in allowed_ids)
    return (
        "{path: [\"filter_ids\"], "
        "operator: ContainsAny, "
        f"valueInt: [{values}]}}"
    )


def weaviate_search_query(
    class_name: str,
    query: np.ndarray,
    allowed_ids: np.ndarray,
    total_count: int,
    limit: int,
) -> str:
    vector = json.dumps(query.astype(float).tolist())
    where = weaviate_where(allowed_ids, total_count)
    where_arg = "" if where is None else f", where: {where}"
    return (
        "{"
        "Get {"
        f"{class_name}("
        f"nearVector: {{vector: {vector}}}, "
        f"limit: {limit}"
        f"{where_arg}"
        ") {"
        "vector_id"
        "}"
        "}"
        "}"
    )


def search_weaviate(
    client: JsonClient,
    class_name: str,
    query: np.ndarray,
    allowed_ids: np.ndarray,
    total_count: int,
    k: int,
    candidates: int,
) -> Tuple[List[int], int, float]:
    limit = max(k, candidates)
    payload = {
        "query": weaviate_search_query(
            class_name,
            query,
            allowed_ids,
            total_count,
            limit,
        )
    }
    response, latency_micros = timed_call(
        lambda: client.request("POST", "/v1/graphql", payload, expected=(200,))
    )

    if response.get("errors"):
        raise RuntimeError(f"Weaviate search returned errors: {response['errors']}")

    rows = response.get("data", {}).get("Get", {}).get(class_name, [])
    candidate_ids = [int(row["vector_id"]) for row in rows]
    return candidate_ids[:k], len(candidate_ids), latency_micros


def run_matrix(
    client: JsonClient,
    class_name: str,
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
                results, candidates, latency_micros = search_weaviate(
                    client,
                    class_name,
                    query,
                    allowed_ids,
                    base.shape[0],
                    config.k,
                    config.candidates,
                )
                recall_sum += filtered_recall(results, oracle)
                candidate_sum += candidates
                latency_sum += latency_micros
                returned_sum += len(results)

            rows.append(
                MatrixRow(
                    engine="weaviate",
                    selectivity=selectivity,
                    correlation=correlation,
                    runner="WeaviateFilteredHnsw",
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
        description="Run Weaviate filtered HNSW over Loupe's benchmark matrix."
    )
    parser.add_argument("base_path", type=Path)
    parser.add_argument("query_path", type=Path)
    parser.add_argument("groundtruth_path", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--class-name", default=DEFAULT_CLASS_NAME)
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
    """Loads vectors into Weaviate, runs native filtered search cells, and writes CSV."""
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
    client.wait_until_ready("/v1/.well-known/ready", attempts=90, sleep_secs=1.0)
    reset_class(
        client,
        args.class_name,
        args.hnsw_m,
        args.ef_construction,
        args.ef_search,
    )
    upload_objects(client, args.class_name, base, args.batch_size)

    config = MatrixConfig(
        k=args.k,
        candidates=args.candidates,
        ef_search=args.ef_search,
        seed=args.seed,
    )
    rows = run_matrix(client, args.class_name, base, queries, config)
    write_csv(args.out, rows)
    print(f"weaviate CSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
