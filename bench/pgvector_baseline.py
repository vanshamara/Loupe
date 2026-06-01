"""pgvector adapter for Loupe's filtered-search benchmark matrix."""

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
    read_fvecs,
    read_ivecs,
    vector_literal,
    write_csv,
)

DEFAULT_BATCH_SIZE = 1000
DEFAULT_DSN = "postgresql://loupe@localhost:5432/loupe_bench"
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_HNSW_M = 16
DEFAULT_TABLE = "loupe_bench_items"


def require_psycopg():
    try:
        import psycopg
    except ImportError as error:
        message = (
            "psycopg is required for this benchmark. "
            "Install it with: python3 -m pip install -r scripts/requirements-competitors.txt"
        )
        raise SystemExit(message) from error

    return psycopg


def checked_identifier(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if not value or value[0].isdigit() or any(char not in allowed for char in value):
        raise ValueError(f"unsafe SQL identifier: {value}")

    return value


def reset_table(
    connection,
    table: str,
    dim: int,
) -> None:
    table = checked_identifier(table)
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
        cursor.execute(
            f"CREATE TABLE {table} ("
            "id integer PRIMARY KEY, "
            f"embedding vector({dim}) NOT NULL"
            ")"
        )
    connection.commit()


def upload_vectors(
    connection,
    table: str,
    base: np.ndarray,
    batch_size: int,
) -> None:
    table = checked_identifier(table)
    with connection.cursor() as cursor:
        for start in range(0, base.shape[0], batch_size):
            records = [
                (start + offset, vector_literal(vector))
                for offset, vector in enumerate(base[start : start + batch_size])
            ]
            cursor.executemany(
                f"INSERT INTO {table} (id, embedding) VALUES (%s, %s::vector)",
                records,
            )
        cursor.execute(f"ANALYZE {table}")
    connection.commit()


def create_hnsw_index(
    connection,
    table: str,
    hnsw_m: int,
    ef_construction: int,
) -> None:
    table = checked_identifier(table)
    with connection.cursor() as cursor:
        cursor.execute(
            f"CREATE INDEX {table}_embedding_hnsw "
            f"ON {table} USING hnsw "
            "(embedding vector_l2_ops) "
            f"WITH (m = {hnsw_m}, ef_construction = {ef_construction})"
        )
        cursor.execute(f"ANALYZE {table}")
    connection.commit()


def set_ef_search(connection, ef_search: int) -> None:
    with connection.cursor() as cursor:
        cursor.execute(f"SET hnsw.ef_search = {int(ef_search)}")


def search_pgvector(
    connection,
    table: str,
    query: np.ndarray,
    allowed_ids: np.ndarray,
    total_count: int,
    k: int,
    candidates: int,
) -> Tuple[List[int], int, float]:
    table = checked_identifier(table)
    limit = max(k, candidates)
    query_literal = vector_literal(query)

    if allowed_ids.size == total_count:
        sql = f"SELECT id FROM {table} ORDER BY embedding <-> %s::vector LIMIT %s"
        params = (query_literal, limit)
    else:
        sql = (
            f"SELECT id FROM {table} "
            "WHERE id = ANY(%s) "
            "ORDER BY embedding <-> %s::vector "
            "LIMIT %s"
        )
        params = ([int(item) for item in allowed_ids], query_literal, limit)

    start = time.perf_counter_ns()
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
    latency_micros = (time.perf_counter_ns() - start) / 1000.0

    candidate_ids = [int(row[0]) for row in rows]
    return candidate_ids[:k], len(candidate_ids), latency_micros


def run_matrix(
    connection,
    table: str,
    base: np.ndarray,
    queries: np.ndarray,
    config: MatrixConfig,
) -> List[MatrixRow]:
    rows = []
    query_count = max(queries.shape[0], 1)
    set_ef_search(connection, config.ef_search)

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
                results, candidates, latency_micros = search_pgvector(
                    connection,
                    table,
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
                    engine="pgvector",
                    selectivity=selectivity,
                    correlation=correlation,
                    runner="PgvectorFilteredHnsw",
                    strategy_chosen="SqlFilteredHnsw",
                    recall=recall_sum / query_count,
                    distance_computations=candidate_sum / query_count,
                    latency_micros=latency_sum / query_count,
                    returned_count=returned_sum / query_count,
                )
            )

    return rows


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pgvector HNSW with SQL filters over Loupe's benchmark matrix."
    )
    parser.add_argument("base_path", type=Path)
    parser.add_argument("query_path", type=Path)
    parser.add_argument("groundtruth_path", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--k", default=DEFAULT_K, type=int)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES, type=int)
    parser.add_argument("--ef-search", default=DEFAULT_EF_SEARCH, type=int)
    parser.add_argument("--ef-construction", default=DEFAULT_EF_CONSTRUCTION, type=int)
    parser.add_argument("--hnsw-m", default=DEFAULT_HNSW_M, type=int)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Loads vectors into pgvector, runs SQL filtered HNSW cells, and writes CSV."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    psycopg = require_psycopg()

    base = read_fvecs(args.base_path)
    queries = read_fvecs(args.query_path)
    ground_truth = read_ivecs(args.groundtruth_path)

    if base.size == 0:
        raise SystemExit("base vectors are empty")

    if queries.shape[0] != ground_truth.shape[0]:
        raise SystemExit("query and ground-truth row counts differ")

    with psycopg.connect(args.dsn) as connection:
        reset_table(
            connection,
            args.table,
            base.shape[1],
        )
        upload_vectors(connection, args.table, base, args.batch_size)
        create_hnsw_index(
            connection,
            args.table,
            args.hnsw_m,
            args.ef_construction,
        )
        config = MatrixConfig(
            k=args.k,
            candidates=args.candidates,
            ef_search=args.ef_search,
            seed=args.seed,
        )
        rows = run_matrix(connection, args.table, base, queries, config)

    write_csv(args.out, rows)
    print(f"pgvector CSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
