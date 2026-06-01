import csv
import tempfile
import unittest
from pathlib import Path

from bench import comparison_plots


FIELDS = (
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


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class ComparisonPlotTests(unittest.TestCase):
    def test_comparison_rows_keep_only_loupe_planner(self):
        rows = [
            comparison_plots.BenchmarkRow(
                engine="loupe",
                selectivity=0.01,
                correlation="anti_correlated",
                runner="Flat",
                strategy_chosen="Flat",
                recall=1.0,
                distance_computations=10.0,
                latency_micros=10.0,
                returned_count=10.0,
            ),
            comparison_plots.BenchmarkRow(
                engine="loupe",
                selectivity=0.01,
                correlation="anti_correlated",
                runner="Planner",
                strategy_chosen="Acorn+AdaptiveEf",
                recall=0.99,
                distance_computations=20.0,
                latency_micros=20.0,
                returned_count=10.0,
            ),
        ]

        selected = comparison_plots.comparison_rows(rows)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].label, "Loupe Planner")

    def test_writes_all_comparison_plots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "matrix.csv"
            out_dir = root / "plots"
            write_csv(
                csv_path,
                [
                    {
                        "engine": "loupe",
                        "selectivity": "0.010000",
                        "correlation": "anti_correlated",
                        "runner": "Planner",
                        "strategy_chosen": "Acorn+AdaptiveEf",
                        "recall": "0.990000",
                        "distance_computations": "100.000",
                        "latency_micros": "80.000",
                        "returned_count": "10.000",
                    },
                    {
                        "engine": "faiss",
                        "selectivity": "0.010000",
                        "correlation": "anti_correlated",
                        "runner": "FaissHnswPostFilter",
                        "strategy_chosen": "HnswPostFilter",
                        "recall": "0.100000",
                        "distance_computations": "512.000",
                        "latency_micros": "30.000",
                        "returned_count": "1.000",
                    },
                    {
                        "engine": "qdrant",
                        "selectivity": "0.010000",
                        "correlation": "anti_correlated",
                        "runner": "QdrantFilteredHnsw",
                        "strategy_chosen": "NativeFilteredHnsw",
                        "recall": "1.000000",
                        "distance_computations": "100.000",
                        "latency_micros": "90.000",
                        "returned_count": "10.000",
                    },
                ],
            )

            rows = comparison_plots.read_csvs([csv_path])
            files = comparison_plots.write_comparison_plots(out_dir, rows)

            for path in [
                files.recall_by_engine,
                files.latency_by_engine,
                files.hard_filter_recall,
                files.faiss_under_return,
                files.planner_vs_external,
            ]:
                text = path.read_text()
                self.assertTrue(text.startswith("<svg"))

            self.assertIn("Loupe Planner", files.recall_by_engine.read_text())
            self.assertIn("FAISS Under-Return", files.faiss_under_return.read_text())


if __name__ == "__main__":
    unittest.main()
