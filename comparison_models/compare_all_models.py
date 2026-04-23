import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comparison_models.baseline_a.run_experiments import run_from_config as run_baseline_a
from comparison_models.baseline_b.run_experiments import run_from_config as run_baseline_b
from comparison_models.baseline_c.run_experiments import run_from_config as run_baseline_c
from comparison_models.common.experiment_utils import (
    COMPARISON_ROOT,
    MODEL_ORDER,
    build_summary_table,
    discover_generated_datasets,
    export_dataset_result_views,
    is_result_complete,
    load_all_results,
    order_dataset_paths,
    recommended_max_workers,
    write_csv_rows,
)
from comparison_models.proposed_model.run_experiments import run_from_config as run_proposed_model

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass


MODEL_RUNNERS = {
    "baseline_a": run_baseline_a,
    "baseline_b": run_baseline_b,
    "baseline_c": run_baseline_c,
    "proposed_model": run_proposed_model,
}


def _normalize_dataset_filter(value: str) -> tuple[str, str]:
    raw = Path(value).name.lower()
    stem = Path(value).stem.lower()
    return raw, stem


def _matches_dataset_filter(path: Path, dataset_filter: str) -> bool:
    raw_needle, stem_needle = _normalize_dataset_filter(dataset_filter)
    name = path.name.lower()
    stem = path.stem.lower()

    candidates = {raw_needle, stem_needle}
    for needle in candidates:
        if not needle:
            continue
        if needle in {name, stem}:
            return True
        if name.startswith(f"{needle}_") or stem.startswith(f"{needle}_"):
            return True
    return False


def _dataset_path_preference(path: Path) -> tuple[int, int, str]:
    parts_lower = [part.lower() for part in path.parts]
    penalty = 0
    if any(part.startswith("_") for part in path.parts):
        penalty += 2
    if any("smoke" in part for part in parts_lower):
        penalty += 3
    if any("archive" in part for part in parts_lower):
        penalty += 3
    return (penalty, len(path.parts), str(path).lower())


def _dedupe_dataset_variants(paths: list[Path]) -> list[Path]:
    by_stem: dict[str, list[Path]] = {}
    for path in paths:
        by_stem.setdefault(path.stem.lower(), []).append(path)

    chosen = [
        sorted(candidates, key=_dataset_path_preference)[0]
        for _, candidates in sorted(by_stem.items())
    ]
    return order_dataset_paths(chosen)


def _resolve_dataset_filters(
    dataset_filters: list[str] | None,
    dataset_manifest: str | None = None,
) -> list[Path]:
    discovered = discover_generated_datasets()
    manifest_filters: list[str] = []

    if dataset_manifest:
        manifest_path = Path(dataset_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        manifest_filters = [
            str(item["output_path"])
            for item in manifest.get("datasets", [])
            if item.get("output_path")
        ]

    combined_filters = list(dataset_filters or []) + manifest_filters

    if not combined_filters:
        return discovered

    resolved: list[Path] = []
    seen: set[str] = set()

    for dataset_filter in combined_filters:
        candidate = Path(dataset_filter)
        matches: list[Path]

        if candidate.exists():
            matches = [candidate.resolve()]
        else:
            matches = [
                path
                for path in discovered
                if _matches_dataset_filter(path, dataset_filter)
            ]

        if not matches:
            raise FileNotFoundError(
                f"No generated dataset matched filter '{dataset_filter}'."
            )

        for match in matches:
            key = str(match.resolve())
            if key not in seen:
                seen.add(key)
                resolved.append(match.resolve())

    return _dedupe_dataset_variants(resolved)


def _resolve_model_filters(model_filters: list[str] | None) -> list[tuple[str, object]]:
    if not model_filters:
        return [(name, MODEL_RUNNERS[name]) for name in MODEL_ORDER]

    requested = []
    for model_name in model_filters:
        if model_name not in MODEL_RUNNERS:
            raise ValueError(f"Unknown model filter: {model_name}")
        requested.append((model_name, MODEL_RUNNERS[model_name]))
    return requested


def run_single_job(job: dict) -> dict:
    model_name = job["model_name"]
    runner = job["runner"]
    dataset_path = job["dataset_path"]
    seed = job["seed"]
    config = dict(job["config"])
    dataset_name = Path(dataset_path).name

    print(
        f"[START] dataset={dataset_name} model={model_name} seed={seed}",
        flush=True,
    )

    runner(
        {
            "dataset_path": str(dataset_path),
            "seed": seed,
            "pop_size": config["pop_size"],
            "gens": config["gens"],
            "cx_prob": config["cx_prob"],
            "mut_prob": config["mut_prob"],
            "max_boxes_per_route": config["max_boxes_per_route"],
            "verbose": config["verbose"],
            "progress_label": f"{model_name} {dataset_name} seed {seed}",
        }
    )

    return {
        "model": model_name,
        "dataset": dataset_name,
        "seed": seed,
    }


def _format_seconds(seconds: float) -> str:
    minutes, secs = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all comparison models across generated datasets."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset filter or full path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dataset-manifest",
        help="Path to a generation manifest JSON created by VRP/generate_dataset_batch.py",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=MODEL_ORDER,
        help="Optional model filter. Can be passed multiple times.",
    )
    parser.add_argument("--start-seed", type=int, default=2000)
    parser.add_argument("--end-seed", type=int, default=2019)
    parser.add_argument("--pop-size", type=int, default=80)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--cx-prob", type=float, default=0.9)
    parser.add_argument("--mut-prob", type=float, default=0.2)
    parser.add_argument("--max-boxes-per-route", type=int, default=48)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Defaults to 6 for XML50/XML100 runs and 4 when any XML500 dataset is included.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun seeds even if their output files already exist.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset_paths = _resolve_dataset_filters(args.dataset, args.dataset_manifest)
    model_runners = _resolve_model_filters(args.model)

    if not dataset_paths:
        raise FileNotFoundError("No generated datasets found in VRP/generated_datasets")

    max_workers = args.max_workers or recommended_max_workers(dataset_paths)
    cpu_count = os.cpu_count() or 1
    if os.name == "nt" and max_workers > 61:
        raise ValueError(
            "On Windows, ProcessPoolExecutor requires max_workers <= 61."
        )

    print("=" * 80)
    print("DATASETS TO PROCESS")
    print("=" * 80)
    for dataset_path in dataset_paths:
        print(f"- {dataset_path.name}")

    print("")
    print("=" * 80)
    print("MODELS TO PROCESS")
    print("=" * 80)
    for model_name, _ in model_runners:
        print(f"- {model_name}")

    jobs: list[dict] = []
    skipped_existing = 0
    expected_jobs = 0

    print("")
    print("=" * 80)
    print("BUILDING JOB LIST")
    print("=" * 80)

    for dataset_path in dataset_paths:
        dataset_name = dataset_path.name
        for model_name, runner in model_runners:
            pending_for_pair = 0
            for seed in range(args.start_seed, args.end_seed + 1):
                expected_jobs += 1
                if not args.force and is_result_complete(model_name, dataset_path, seed):
                    skipped_existing += 1
                    continue

                jobs.append(
                    {
                        "model_name": model_name,
                        "runner": runner,
                        "dataset_path": str(dataset_path),
                        "seed": seed,
                        "config": {
                            "pop_size": args.pop_size,
                            "gens": args.gens,
                            "cx_prob": args.cx_prob,
                            "mut_prob": args.mut_prob,
                            "max_boxes_per_route": args.max_boxes_per_route,
                            "verbose": args.verbose,
                        },
                    }
                )
                pending_for_pair += 1

            print(
                f"dataset={dataset_name} model={model_name} "
                f"pending={pending_for_pair} skipped_existing={args.end_seed - args.start_seed + 1 - pending_for_pair}"
            )

    total_jobs = len(jobs)

    print("")
    print(f"Expected jobs in selected scope: {expected_jobs}")
    print(f"Skipped completed jobs: {skipped_existing}")
    print(f"Jobs to run now: {total_jobs}")
    print(f"Using max workers: {max_workers}")
    if args.max_workers is not None and max_workers > cpu_count:
        print(
            f"WARNING: requested max_workers={max_workers} exceeds os.cpu_count()={cpu_count}.",
            flush=True,
        )

    if total_jobs == 0:
        print("No pending jobs found. Refreshing summary outputs from existing results.")
    else:
        print("")
        print("=" * 80)
        print("STARTING PARALLEL EXECUTION")
        print("=" * 80)

        completed = 0
        failed = 0
        pipeline_start = time.perf_counter()

        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            future_to_job = {
                executor.submit(run_single_job, job): job
                for job in jobs
            }

            for future in as_completed(future_to_job):
                completed += 1
                elapsed = time.perf_counter() - pipeline_start
                remaining = total_jobs - completed
                eta_seconds = (elapsed / completed) * remaining if completed else 0.0

                try:
                    result = future.result()
                    print(
                        f"[{completed}/{total_jobs}] DONE "
                        f"dataset={result['dataset']} model={result['model']} seed={result['seed']} "
                        f"workers={max_workers} elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta_seconds)}",
                        flush=True,
                    )
                except Exception as exc:
                    failed += 1
                    job = future_to_job[future]
                    print(
                        f"[{completed}/{total_jobs}] FAILED "
                        f"dataset={Path(job['dataset_path']).name} model={job['model_name']} seed={job['seed']} "
                        f"workers={max_workers} elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta_seconds)} "
                        f"error={exc}",
                        flush=True,
                    )

        print("")
        print("=" * 80)
        print("EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Completed jobs: {total_jobs - failed}")
        print(f"Failed jobs: {failed}")

    print("")
    print("=" * 80)
    print("GENERATING FINAL REPORTING OUTPUTS")
    print("=" * 80)

    results_rows = load_all_results()
    export_dataset_result_views(results_rows)

    combined_results_path = COMPARISON_ROOT / "combined_results.csv"
    write_csv_rows(combined_results_path, results_rows)

    summary_rows = build_summary_table(results_rows)
    summary_path = COMPARISON_ROOT / "model_comparison_summary.csv"
    write_csv_rows(summary_path, summary_rows)

    print(f"Combined results saved to: {combined_results_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Dataset-first results saved under: {COMPARISON_ROOT / 'results'}")

    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
