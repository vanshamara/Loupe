import csv
import tempfile
import unittest
from pathlib import Path

from bench import package_results


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


def write_csv(path, engine, runner, recall):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "engine": engine,
                "selectivity": "0.010000",
                "correlation": "anti_correlated",
                "runner": runner,
                "strategy_chosen": "Planner",
                "recall": f"{recall:.6f}",
                "distance_computations": "100.000",
                "latency_micros": "50.000",
                "returned_count": "10.000",
            }
        )


def write_planner_summary(path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "planner",
                "mean_recall",
                "mean_distance_computations",
                "regret",
                "regret_delta",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "planner": "RuleLadder",
                "mean_recall": "0.900000",
                "mean_distance_computations": "100.000",
                "regret": "0.200000",
                "regret_delta": "0.100000",
            }
        )
        writer.writerow(
            {
                "planner": "CostModel",
                "mean_recall": "0.950000",
                "mean_distance_computations": "80.000",
                "regret": "0.100000",
                "regret_delta": "0.100000",
            }
        )


class PackageResultsTests(unittest.TestCase):
    def test_rejects_unsafe_run_id(self):
        with self.assertRaises(ValueError):
            package_results.checked_run_id("../bad")

    def test_packages_csvs_plots_metadata_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_dir = root / "input-csv"
            plot_dir = root / "input-plots"
            out_dir = root / "results"
            csv_dir.mkdir()
            plot_dir.mkdir()
            loupe_csv = csv_dir / "loupe.csv"
            faiss_csv = csv_dir / "faiss.csv"
            planner_summary = csv_dir / "loupe-planner-summary.csv"
            write_csv(loupe_csv, "loupe", "Planner", 0.99)
            write_csv(faiss_csv, "faiss", "FaissHnswPostFilter", 0.1)
            write_planner_summary(planner_summary)
            (plot_dir / "recall.svg").write_text("<svg></svg>\n")

            packaged = package_results.package_results(
                run_id="siftsmall-smoke",
                dataset="siftsmall",
                csv_paths=[loupe_csv, faiss_csv],
                plot_dirs=[plot_dir],
                out_dir=out_dir,
                notes="test run",
                planner_summary_paths=[planner_summary],
            )

            self.assertTrue((packaged.csv_dir / "loupe.csv").exists())
            self.assertTrue((packaged.csv_dir / "faiss.csv").exists())
            self.assertTrue(
                (packaged.planner_summary_dir / "loupe-planner-summary.csv").exists()
            )
            self.assertTrue((packaged.plot_dir / "recall.svg").exists())
            self.assertIn("siftsmall-smoke", packaged.metadata_path.read_text())
            self.assertIn("planner_summaries", packaged.metadata_path.read_text())
            self.assertIn("FAISS", packaged.summary_path.read_text())
            self.assertIn("Planner Summary", packaged.summary_path.read_text())
            self.assertIn("CostModel", packaged.summary_path.read_text())
            self.assertIn("test run", packaged.summary_path.read_text())


if __name__ == "__main__":
    unittest.main()
