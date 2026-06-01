import unittest

import numpy as np

from bench import benchmark_common
from bench import http_common
from bench import pgvector_baseline
from bench import qdrant_baseline
from bench import weaviate_baseline


class CompetitorAdapterTests(unittest.TestCase):
    def test_qdrant_filter_omits_full_allowlist(self):
        allowed = np.array([0, 1, 2], dtype=np.int64)

        self.assertIsNone(qdrant_baseline.qdrant_filter(allowed, 3))

    def test_qdrant_filter_uses_has_id_for_subset(self):
        allowed = np.array([2, 4, 6], dtype=np.int64)

        payload = qdrant_baseline.qdrant_filter(allowed, 10)

        self.assertEqual(payload["must"][0]["has_id"], [2, 4, 6])

    def test_weaviate_query_omits_where_for_full_allowlist(self):
        query = np.array([1.0, 2.0], dtype=np.float32)
        allowed = np.array([0, 1], dtype=np.int64)

        graphql = weaviate_baseline.weaviate_search_query(
            "LoupeBenchWeaviate",
            query,
            allowed,
            2,
            10,
        )

        self.assertIn("nearVector", graphql)
        self.assertNotIn("where:", graphql)

    def test_weaviate_query_uses_int_allowlist_filter(self):
        query = np.array([1.0, 2.0], dtype=np.float32)
        allowed = np.array([1, 3, 5], dtype=np.int64)

        graphql = weaviate_baseline.weaviate_search_query(
            "LoupeBenchWeaviate",
            query,
            allowed,
            10,
            10,
        )

        self.assertIn("ContainsAny", graphql)
        self.assertIn("filter_ids", graphql)
        self.assertIn("valueInt: [1, 3, 5]", graphql)

    def test_pgvector_identifier_rejects_unsafe_table_names(self):
        with self.assertRaises(ValueError):
            pgvector_baseline.checked_identifier("items;drop")

    def test_vector_literal_uses_pgvector_format(self):
        literal = benchmark_common.vector_literal([1.0, 2.5, 3.0])

        self.assertEqual(literal, "[1,2.5,3]")

    def test_ready_check_accepts_plain_text_response(self):
        class PlainTextClient(http_common.JsonClient):
            def __init__(self):
                pass

            def request(self, method, path, payload=None, expected=(200,), decode_json=True):
                self.decode_json = decode_json
                return {}

        client = PlainTextClient()

        client.wait_until_ready("/readyz", attempts=1, sleep_secs=0.0)

        self.assertFalse(client.decode_json)


if __name__ == "__main__":
    unittest.main()
