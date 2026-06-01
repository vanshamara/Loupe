import tempfile
import unittest
from pathlib import Path

import numpy as np

from bench import faiss_baseline


def fvecs_bytes(vectors):
    data = bytearray()
    for vector in vectors:
        data.extend(np.int32(len(vector)).tobytes())
        for value in vector:
            data.extend(np.float32(value).tobytes())
    return bytes(data)


def ivecs_bytes(rows):
    data = bytearray()
    for row in rows:
        data.extend(np.int32(len(row)).tobytes())
        for value in row:
            data.extend(np.int32(value).tobytes())
    return bytes(data)


class FaissBaselineTests(unittest.TestCase):
    def write_temp(self, data):
        handle = tempfile.NamedTemporaryFile(delete=False)
        with handle:
            handle.write(data)
        path = Path(handle.name)
        self.addCleanup(path.unlink)
        return path

    def test_fvecs_loader_reads_sift_records(self):
        path = self.write_temp(fvecs_bytes([[1.0, 2.0], [3.0, 4.0]]))
        vectors = faiss_baseline.read_fvecs(path)

        np.testing.assert_allclose(vectors, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_ivecs_loader_reads_sift_records(self):
        path = self.write_temp(ivecs_bytes([[1, 2], [3, 4]]))
        rows = faiss_baseline.read_ivecs(path)

        np.testing.assert_array_equal(rows, np.array([[1, 2], [3, 4]], dtype=np.int32))

    def test_filters_match_expected_correlation_shapes(self):
        base = np.array([[float(i), float(i)] for i in range(100)], dtype=np.float32)
        query = np.array([50.0, 50.0], dtype=np.float32)

        positive = faiss_baseline.generate_filter(base, query, 0.1, "positive", 1)
        anti = faiss_baseline.generate_filter(base, query, 0.1, "anti_correlated", 1)

        self.assertIn(50, positive)
        self.assertNotIn(0, positive)
        self.assertNotIn(50, anti)
        self.assertTrue(0 in anti or 99 in anti)

    def test_exact_filtered_search_orders_by_distance(self):
        base = np.array(
            [
                [0.0, 0.0],
                [10.0, 10.0],
                [2.0, 2.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        query = np.array([0.0, 0.0], dtype=np.float32)
        allowed = np.array([1, 2, 3], dtype=np.int64)

        result = faiss_baseline.exact_filtered_search(base, query, allowed, 2)

        self.assertEqual(result, [3, 2])

    def test_csv_writer_uses_shared_schema(self):
        row = faiss_baseline.MatrixRow(
            engine="faiss",
            selectivity=0.01,
            correlation="anti_correlated",
            runner="FaissHnswPostFilter",
            strategy_chosen="HnswPostFilter",
            recall=0.8,
            distance_computations=512.0,
            latency_micros=35.0,
            returned_count=4.0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "faiss.csv"
            faiss_baseline.write_csv(output, [row])
            text = output.read_text()

        self.assertTrue(text.startswith(",".join(faiss_baseline.CSV_FIELDS)))
        self.assertIn("faiss,0.010000,anti_correlated", text)


if __name__ == "__main__":
    unittest.main()
