import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from bench import run_benchmark_suite


class RunBenchmarkSuiteTests(unittest.TestCase):
    def test_parse_engines_accepts_all(self):
        engines = run_benchmark_suite.parse_engines("all")

        self.assertEqual(
            list(run_benchmark_suite.ENGINE_ORDER),
            engines,
        )

    def test_parse_engines_dedupes_and_keeps_order(self):
        engines = run_benchmark_suite.parse_engines("faiss,loupe,faiss")

        self.assertEqual(
            [
                "faiss",
                "loupe",
            ],
            engines,
        )

    def test_parse_engines_rejects_unknown_engine(self):
        with self.assertRaises(ValueError):
            run_benchmark_suite.parse_engines("loupe,bad")

    def test_run_suite_builds_commands_and_stops_started_docker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "base.fvecs"
            query = root / "query.fvecs"
            groundtruth = root / "groundtruth.ivecs"
            base.write_bytes(b"base")
            query.write_bytes(b"query")
            groundtruth.write_bytes(b"truth")
            calls = []

            def executor(command, cwd):
                calls.append(
                    (
                        list(command),
                        cwd,
                    )
                )

            config = run_benchmark_suite.SuiteConfig(
                repo_root=root,
                dataset_name="tiny",
                base_path=base,
                query_path=query,
                groundtruth_path=groundtruth,
                run_id="tiny-run",
                engines=[
                    "loupe",
                    "qdrant",
                ],
                work_dir=root / "work",
                results_dir=root / "results",
                python=Path("python3"),
                compose_file=Path("docker/benchmark-compose.yml"),
                notes="test run",
                start_docker=True,
                keep_docker_running=False,
                dry_run=False,
                preflight=False,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                result = run_benchmark_suite.run_suite(config, executor)

            commands = [call[0] for call in calls]
            self.assertEqual(
                [
                    "docker",
                    "compose",
                    "-f",
                    "docker/benchmark-compose.yml",
                    "up",
                    "-d",
                    "qdrant",
                ],
                commands[0],
            )
            self.assertEqual("cargo", commands[1][0])
            self.assertIn("bench/qdrant_baseline.py", commands[2])
            self.assertIn("bench/comparison_plots.py", commands[3])
            self.assertIn("bench/package_results.py", commands[4])
            self.assertEqual(
                [
                    "docker",
                    "compose",
                    "-f",
                    "docker/benchmark-compose.yml",
                    "stop",
                    "qdrant",
                ],
                commands[5],
            )
            self.assertEqual(root / "results" / "tiny-run", result.package_root)
            self.assertEqual(root, calls[0][1])

    def test_dry_run_skips_executor_and_missing_inputs(self):
        calls = []
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir) / "work"
            config = run_benchmark_suite.SuiteConfig(
                repo_root=Path("."),
                dataset_name="tiny",
                base_path=Path("missing-base.fvecs"),
                query_path=Path("missing-query.fvecs"),
                groundtruth_path=Path("missing-groundtruth.ivecs"),
                run_id="tiny-run",
                engines=[
                    "faiss",
                ],
                work_dir=work_dir,
                results_dir=Path("bench/results"),
                python=Path("python3"),
                compose_file=Path("docker/benchmark-compose.yml"),
                notes="test run",
                start_docker=True,
                keep_docker_running=False,
                dry_run=True,
                preflight=True,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                run_benchmark_suite.run_suite(config, lambda command, cwd: calls.append(command))

            self.assertEqual([], calls)
            self.assertFalse(work_dir.exists())


if __name__ == "__main__":
    unittest.main()
