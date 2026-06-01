"""Runs Loupe benchmark adapters and packages a result bundle."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import package_results

ENGINE_ORDER = (
    "loupe",
    "faiss",
    "qdrant",
    "weaviate",
    "pgvector",
)
DEFAULT_ENGINES = (
    "loupe",
    "faiss",
)
DEFAULT_QUERY_LIMIT = 10
DOCKER_SERVICES = {
    "qdrant": "qdrant",
    "weaviate": "weaviate",
    "pgvector": "pgvector",
}
PYTHON_MODULES = {
    "faiss": ("numpy", "faiss"),
    "qdrant": ("numpy",),
    "weaviate": ("numpy",),
    "pgvector": ("numpy", "psycopg"),
}

Executor = Callable[[Sequence[str], Path], None]


@dataclass(frozen=True)
class SuiteConfig:
    repo_root: Path
    dataset_name: str
    base_path: Path
    query_path: Path
    groundtruth_path: Path
    run_id: str
    engines: Sequence[str]
    work_dir: Path
    results_dir: Path
    python: Path
    compose_file: Path
    notes: str
    start_docker: bool
    keep_docker_running: bool
    dry_run: bool
    query_limit: Optional[int]
    loupe_full_checks: bool
    preflight: bool = True


@dataclass(frozen=True)
class SuiteResult:
    csv_paths: Sequence[Path]
    planner_summary_paths: Sequence[Path]
    plot_dirs: Sequence[Path]
    package_root: Path
    summary_path: Path


def run_suite(
    config: SuiteConfig,
    executor: Optional[Executor] = None,
) -> SuiteResult:
    """Runs selected benchmark engines, writes plots, and packages their outputs."""
    if executor is None:
        executor = subprocess_executor

    checked_run_id = package_results.checked_run_id(config.run_id)
    validate_config(config)
    if not config.dry_run:
        config.work_dir.mkdir(parents=True, exist_ok=True)

    csv_paths: List[Path] = []
    planner_summary_paths: List[Path] = []
    plot_dirs: List[Path] = []
    started_services = docker_services(config.engines) if config.start_docker else []

    if config.preflight and not config.dry_run:
        run_preflight(config, executor)

    try:
        if started_services:
            run_command(docker_up_command(config, started_services), config, executor)

        for engine in config.engines:
            csv_path = matrix_csv_path(config, engine)
            csv_paths.append(csv_path)

            if engine == "loupe":
                plot_dir = config.work_dir / "loupe-plots"
                planner_summary_path = planner_summary_csv_path(config, engine)
                planner_summary_paths.append(planner_summary_path)
                plot_dirs.append(plot_dir)
                command = loupe_command(config, csv_path, planner_summary_path, plot_dir)
            else:
                command = adapter_command(config, engine, csv_path)

            run_command(command, config, executor)

        comparison_dir = config.work_dir / "comparison-plots"
        plot_dirs.append(comparison_dir)
        run_command(comparison_command(config, csv_paths, comparison_dir), config, executor)
        run_command(
            package_command(
                config,
                checked_run_id,
                csv_paths,
                planner_summary_paths,
                plot_dirs,
            ),
            config,
            executor,
        )
    finally:
        if started_services and not config.keep_docker_running:
            run_command(docker_stop_command(config, started_services), config, executor)

    package_root = config.results_dir / checked_run_id
    return SuiteResult(
        csv_paths=csv_paths,
        planner_summary_paths=planner_summary_paths,
        plot_dirs=plot_dirs,
        package_root=package_root,
        summary_path=package_root / "summary.md",
    )


def validate_config(config: SuiteConfig) -> None:
    if not config.engines:
        raise ValueError("at least one engine is required")

    for engine in config.engines:
        if engine not in ENGINE_ORDER:
            raise ValueError(f"unknown engine: {engine}")

    if config.dry_run:
        return

    required_paths = (
        config.base_path,
        config.query_path,
        config.groundtruth_path,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"missing benchmark input: {path}")


def run_preflight(config: SuiteConfig, executor: Executor) -> None:
    if "loupe" in config.engines:
        require_program("cargo")

    require_python(config.python)

    modules = python_modules(config.engines)
    if modules:
        command = python_import_command(config.python, modules)
        run_command(command, config, executor)

    if docker_services(config.engines) and config.start_docker:
        require_program("docker")
        if not config.compose_file.exists():
            raise FileNotFoundError(f"missing Docker compose file: {config.compose_file}")


def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"missing required command: {name}")


def require_python(python: Path) -> None:
    if python.is_absolute() or "/" in str(python):
        if not python.exists():
            raise FileNotFoundError(f"missing Python executable: {python}")
        return

    if shutil.which(str(python)) is None:
        raise FileNotFoundError(f"missing Python executable: {python}")


def python_modules(engines: Sequence[str]) -> Sequence[str]:
    modules = []

    for engine in engines:
        for module in PYTHON_MODULES.get(engine, ()):
            if module not in modules:
                modules.append(module)

    return modules


def python_import_command(python: Path, modules: Sequence[str]) -> List[str]:
    quoted = ", ".join(repr(module) for module in modules)
    code = (
        "import importlib.util, sys; "
        f"modules = [{quoted}]; "
        "missing = [name for name in modules if importlib.util.find_spec(name) is None]; "
        "sys.exit('missing Python modules: ' + ', '.join(missing) if missing else 0)"
    )
    return [str(python), "-c", code]


def matrix_csv_path(config: SuiteConfig, engine: str) -> Path:
    return config.work_dir / f"{engine}-matrix.csv"


def planner_summary_csv_path(config: SuiteConfig, engine: str) -> Path:
    return config.work_dir / f"{engine}-planner-summary.csv"


def loupe_command(
    config: SuiteConfig,
    csv_path: Path,
    planner_summary_path: Path,
    plot_dir: Path,
) -> List[str]:
    return [
        "cargo",
        "run",
        "--release",
        "-p",
        "loupe-bench",
        "--bin",
        "loupe-bench",
        "--",
        str(config.base_path),
        str(config.query_path),
        str(config.groundtruth_path),
        "--matrix-csv",
        str(csv_path),
        "--planner-summary-csv",
        str(planner_summary_path),
        "--plot-dir",
        str(plot_dir),
    ] + loupe_scale_flags(config)


def loupe_scale_flags(config: SuiteConfig) -> List[str]:
    flags = query_limit_flags(config)

    if not config.loupe_full_checks:
        flags.extend(
            [
                "--skip-cosine",
                "--skip-mmap",
                "--skip-compaction",
            ]
        )

    return flags


def adapter_command(config: SuiteConfig, engine: str, csv_path: Path) -> List[str]:
    script = {
        "faiss": "bench/faiss_baseline.py",
        "qdrant": "bench/qdrant_baseline.py",
        "weaviate": "bench/weaviate_baseline.py",
        "pgvector": "bench/pgvector_baseline.py",
    }.get(engine)

    if script is None:
        raise ValueError(f"no adapter script for engine: {engine}")

    command = [
        str(config.python),
        script,
        str(config.base_path),
        str(config.query_path),
        str(config.groundtruth_path),
        "--out",
        str(csv_path),
    ]
    command.extend(query_limit_flags(config))
    return command


def query_limit_flags(config: SuiteConfig) -> List[str]:
    if config.query_limit is None:
        return []

    return [
        "--query-limit",
        str(config.query_limit),
    ]


def comparison_command(
    config: SuiteConfig,
    csv_paths: Sequence[Path],
    out_dir: Path,
) -> List[str]:
    command = [
        str(config.python),
        "bench/comparison_plots.py",
    ]

    for path in csv_paths:
        command.append(str(path))

    command.extend(
        [
            "--out-dir",
            str(out_dir),
        ]
    )
    return command


def package_command(
    config: SuiteConfig,
    run_id: str,
    csv_paths: Sequence[Path],
    planner_summary_paths: Sequence[Path],
    plot_dirs: Sequence[Path],
) -> List[str]:
    command = [
        str(config.python),
        "bench/package_results.py",
        "--run-id",
        run_id,
        "--dataset",
        config.dataset_name,
        "--out-dir",
        str(config.results_dir),
        "--notes",
        config.notes,
    ]

    for path in csv_paths:
        command.extend(["--csv", str(path)])

    for path in planner_summary_paths:
        command.extend(["--planner-summary-csv", str(path)])

    for path in plot_dirs:
        command.extend(["--plot-dir", str(path)])

    return command


def docker_services(engines: Sequence[str]) -> List[str]:
    services = []

    for engine in engines:
        service = DOCKER_SERVICES.get(engine)
        if service is not None and service not in services:
            services.append(service)

    return services


def docker_up_command(config: SuiteConfig, services: Sequence[str]) -> List[str]:
    command = [
        "docker",
        "compose",
        "-f",
        str(config.compose_file),
        "up",
        "-d",
    ]
    command.extend(services)
    return command


def docker_stop_command(config: SuiteConfig, services: Sequence[str]) -> List[str]:
    command = [
        "docker",
        "compose",
        "-f",
        str(config.compose_file),
        "stop",
    ]
    command.extend(services)
    return command


def run_command(
    command: Sequence[str],
    config: SuiteConfig,
    executor: Executor,
) -> None:
    print(f"$ {shlex.join(str(part) for part in command)}", flush=True)

    if config.dry_run:
        return

    executor(command, config.repo_root)


def subprocess_executor(command: Sequence[str], cwd: Path) -> None:
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except FileNotFoundError as error:
        raise SystemExit(f"missing command: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        printable = shlex.join(str(part) for part in command)
        raise SystemExit(
            f"command failed with exit code {error.returncode}: {printable}"
        ) from error


def parse_engines(value: str) -> List[str]:
    if value == "all":
        return list(ENGINE_ORDER)

    engines = []
    raw_engines = value.split(",")

    for raw_engine in raw_engines:
        engine = raw_engine.strip()
        if not engine:
            continue
        if engine not in ENGINE_ORDER:
            raise ValueError(f"unknown engine: {engine}")
        if engine not in engines:
            engines.append(engine)

    if not engines:
        raise ValueError("at least one engine is required")

    return engines


def default_work_dir(run_id: str) -> Path:
    return Path(tempfile.gettempdir()) / "loupe-benchmark-suite" / run_id


def default_notes(
    dataset_name: str,
    engines: Sequence[str],
    query_limit: Optional[int],
) -> str:
    query_text = "all queries"
    if query_limit is not None:
        query_text = f"first {query_limit} queries"

    if any(engine in DOCKER_SERVICES or engine == "faiss" for engine in engines):
        return (
            f"{dataset_name} benchmark suite run over {query_text}; external engines "
            "use returned candidate count as the cost proxy"
        )

    return f"{dataset_name} benchmark suite run over {query_text}"


def checked_query_limit(value: int) -> Optional[int]:
    if value < 0:
        raise ValueError("query limit must be zero or positive")

    if value == 0:
        return None

    return value


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Loupe benchmark engines, generate plots, and package results."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--query", required=True, type=Path)
    parser.add_argument("--groundtruth", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--engines",
        default=",".join(DEFAULT_ENGINES),
        help="comma-separated engine list or all",
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--results-dir", default=Path("bench/results"), type=Path)
    parser.add_argument("--python", default=Path(sys.executable), type=Path)
    parser.add_argument(
        "--query-limit",
        default=DEFAULT_QUERY_LIMIT,
        type=int,
        help="number of query rows to run; use 0 for all rows",
    )
    parser.add_argument(
        "--compose-file",
        default=Path("docker/benchmark-compose.yml"),
        type=Path,
    )
    parser.add_argument("--notes")
    parser.add_argument("--no-docker-start", action="store_true")
    parser.add_argument("--keep-docker-running", action="store_true")
    parser.add_argument("--loupe-full-checks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> SuiteConfig:
    engines = parse_engines(args.engines)
    run_id = package_results.checked_run_id(args.run_id)
    query_limit = checked_query_limit(args.query_limit)
    work_dir = args.work_dir

    if work_dir is None:
        work_dir = default_work_dir(run_id)

    notes = args.notes
    if notes is None:
        notes = default_notes(args.dataset_name, engines, query_limit)

    return SuiteConfig(
        repo_root=Path(__file__).resolve().parents[1],
        dataset_name=args.dataset_name,
        base_path=args.base,
        query_path=args.query,
        groundtruth_path=args.groundtruth,
        run_id=run_id,
        engines=engines,
        work_dir=work_dir,
        results_dir=args.results_dir,
        python=args.python,
        compose_file=args.compose_file,
        notes=notes,
        start_docker=not args.no_docker_start,
        keep_docker_running=args.keep_docker_running,
        dry_run=args.dry_run,
        query_limit=query_limit,
        loupe_full_checks=args.loupe_full_checks,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parses CLI options, runs the benchmark suite, and prints the summary path."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    config = config_from_args(args)
    result = run_suite(config)

    if config.dry_run:
        print(f"dry-run result bundle: {result.package_root}")
        print(f"dry-run summary: {result.summary_path}")
    else:
        print(f"result bundle: {result.package_root}")
        print(f"summary: {result.summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
