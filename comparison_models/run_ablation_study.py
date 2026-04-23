import argparse
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comparison_models.common.experiment_utils import (  # noqa: E402
    ABLATION_OUTPUTS_ROOT,
    COMPARISON_ROOT,
    build_summary_table,
    discover_generated_datasets,
    export_dataset_result_views,
    is_result_complete,
    load_all_results,
    order_dataset_paths,
    recommended_max_workers,
    write_csv_rows,
)
from comparison_models.common.paper_reporting import (  # noqa: E402
    ABLATION_LABELS,
    ABLATION_MODEL_ORDER,
    ABLATION_VARIANTS,
    REPRESENTATIVE_DATASET_SIZES,
)
from comparison_models.proposed_model.run_experiments import run_from_config  # noqa: E402

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def _normalize_dataset_filter(value: str) -> tuple[str, str]:
    raw = Path(value).name.lower()
    stem = Path(value).stem.lower()
    return raw, stem


def _matches_dataset_filter(path: Path, dataset_filter: str) -> bool:
    raw_needle, stem_needle = _normalize_dataset_filter(dataset_filter)
    name = path.name.lower()
    stem = path.stem.lower()

    for needle in {raw_needle, stem_needle}:
        if not needle:
            continue
        if needle in {name, stem}:
            return True
        if name.startswith(f"{needle}_") or stem.startswith(f"{needle}_"):
            return True
    return False


def _resolve_datasets(dataset_filters: list[str] | None) -> list[Path]:
    discovered = discover_generated_datasets()
    if not dataset_filters:
        return [
            path for path in discovered
            if any(path.stem.startswith(f"XML{size}_") for size in REPRESENTATIVE_DATASET_SIZES)
        ]

    resolved: list[Path] = []
    seen: set[str] = set()
    for dataset_filter in dataset_filters:
        candidate = Path(dataset_filter)
        if candidate.exists():
            matches = [candidate.resolve()]
        else:
            matches = [path.resolve() for path in discovered if _matches_dataset_filter(path, dataset_filter)]
        if not matches:
            raise FileNotFoundError(f"No generated dataset matched filter '{dataset_filter}'.")
        for match in matches:
            key = str(match)
            if key not in seen:
                seen.add(key)
                resolved.append(match)
    return order_dataset_paths(resolved)


def _resolve_variants(variant_filters: list[str] | None) -> list[dict]:
    by_id = {variant["id"]: variant for variant in ABLATION_VARIANTS}
    if not variant_filters:
        return [by_id[variant_id] for variant_id in ABLATION_MODEL_ORDER]
    resolved = []
    for variant_id in variant_filters:
        if variant_id not in by_id:
            raise ValueError(f"Unknown ablation variant '{variant_id}'. Valid values: {', '.join(ABLATION_MODEL_ORDER)}")
        resolved.append(by_id[variant_id])
    return resolved


def _format_seconds(seconds: float) -> str:
    minutes, secs = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _run_job(job: dict) -> dict:
    variant = job["variant"]
    dataset_path = job["dataset_path"]
    seed = job["seed"]
    config = dict(job["config"])
    label = f"{variant['id']} {Path(dataset_path).name} seed {seed}"

    run_from_config(
        {
            "dataset_path": str(dataset_path),
            "seed": seed,
            "pop_size": config["pop_size"],
            "gens": config["gens"],
            "cx_prob": config["cx_prob"],
            "mut_prob": config["mut_prob"],
            "max_boxes_per_route": config["max_boxes_per_route"],
            "verbose": config["verbose"],
            "progress_label": label,
            "model_name_override": variant["id"],
            "outputs_root": str(ABLATION_OUTPUTS_ROOT),
            **variant["flags"],
        }
    )
    return {"variant": variant["id"], "dataset": Path(dataset_path).name, "seed": seed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run proposed-model ablation study variants.")
    parser.add_argument("--dataset", action="append", help="Dataset filter or full path. Can be passed multiple times.")
    parser.add_argument("--variant", action="append", help="Ablation variant id. Can be passed multiple times.")
    parser.add_argument("--start-seed", type=int, default=3000)
    parser.add_argument("--end-seed", type=int, default=3009)
    parser.add_argument("--pop-size", type=int, default=80)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--cx-prob", type=float, default=0.9)
    parser.add_argument("--mut-prob", type=float, default=0.2)
    parser.add_argument("--max-boxes-per-route", type=int, default=70)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset_paths = _resolve_datasets(args.dataset)
    variants = _resolve_variants(args.variant)
    if not dataset_paths:
        raise FileNotFoundError("No generated datasets found for ablation study.")

    max_workers = args.max_workers or recommended_max_workers(dataset_paths)
    cpu_count = os.cpu_count() or 1
    if os.name == "nt" and max_workers > 61:
        raise ValueError("On Windows, ProcessPoolExecutor requires max_workers <= 61.")

    print("=" * 80)
    print("ABLATION DATASETS")
    print("=" * 80)
    for dataset_path in dataset_paths:
        print(f"- {dataset_path.name}")
    print("")
    print("=" * 80)
    print("ABLATION VARIANTS")
    print("=" * 80)
    for variant in variants:
        print(f"- {variant['id']}: {ABLATION_LABELS[variant['id']]}")

    jobs: list[dict] = []
    skipped_existing = 0
    for dataset_path in dataset_paths:
        for variant in variants:
            pending = 0
            for seed in range(args.start_seed, args.end_seed + 1):
                if not args.force and is_result_complete(variant["id"], dataset_path, seed, outputs_root=ABLATION_OUTPUTS_ROOT):
                    skipped_existing += 1
                    continue
                jobs.append(
                    {
                        "variant": variant,
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
                pending += 1
            print(f"dataset={dataset_path.name} variant={variant['id']} pending={pending}")

    print("")
    print(f"Jobs to run now: {len(jobs)}")
    print(f"Skipped completed jobs: {skipped_existing}")
    print(f"Using max workers: {max_workers}")
    if args.max_workers is not None and max_workers > cpu_count:
        print(f"WARNING: requested max_workers={max_workers} exceeds os.cpu_count()={cpu_count}.", flush=True)

    if jobs:
        completed = 0
        failed = 0
        start = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            future_to_job = {executor.submit(_run_job, job): job for job in jobs}
            for future in as_completed(future_to_job):
                completed += 1
                elapsed = time.perf_counter() - start
                remaining = len(jobs) - completed
                eta_seconds = (elapsed / completed) * remaining if completed else 0.0
                try:
                    result = future.result()
                    print(
                        f"[{completed}/{len(jobs)}] DONE variant={result['variant']} dataset={result['dataset']} seed={result['seed']} "
                        f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta_seconds)}",
                        flush=True,
                    )
                except Exception as exc:
                    failed += 1
                    job = future_to_job[future]
                    print(
                        f"[{completed}/{len(jobs)}] FAILED variant={job['variant']['id']} dataset={Path(job['dataset_path']).name} seed={job['seed']} "
                        f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta_seconds)} error={exc}",
                        flush=True,
                    )
        print("")
        print(f"Completed jobs: {len(jobs) - failed}")
        print(f"Failed jobs: {failed}")

    results_rows = load_all_results(ABLATION_OUTPUTS_ROOT)
    export_dataset_result_views(
        results_rows,
        COMPARISON_ROOT / "results" / "ablation" / "datasets",
        model_order=ABLATION_MODEL_ORDER,
    )
    write_csv_rows(COMPARISON_ROOT / "results" / "ablation" / "combined_results.csv", results_rows)
    write_csv_rows(COMPARISON_ROOT / "results" / "ablation" / "summary.csv", build_summary_table(results_rows))
    print(f"Saved ablation outputs to {ABLATION_OUTPUTS_ROOT}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
