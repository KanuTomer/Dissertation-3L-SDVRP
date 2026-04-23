import json
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comparison_models.baseline_a.config import MODEL_NAME
from comparison_models.baseline_a.ga_runner import GARunner
from comparison_models.common.experiment_utils import (
    build_metrics_row,
    discover_generated_datasets,
    get_output_dir,
    normalize_result_for_reporting,
    order_dataset_paths,
    resolve_dataset_path,
    write_history_csv,
    write_report_json,
)
from comparison_models.common.metrics_logger import save_metrics


def run_from_config(config: dict):
    config["verbose"] = bool(config.get("verbose", True))
    dataset_path = resolve_dataset_path(config.get("dataset_path"))
    dataset_name = dataset_path.stem
    seed = int(config.get("seed", 0))
    config["progress_label"] = config.get("progress_label", f"{MODEL_NAME} {dataset_name} seed {seed}")

    out_dir = get_output_dir(MODEL_NAME, dataset_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = GARunner(config)
    result = runner.run(str(dataset_path))
    result["max_boxes_per_route"] = int(config.get("max_boxes_per_route", 48))
    result = normalize_result_for_reporting(MODEL_NAME, dataset_path, result)

    result_path = out_dir / f"result_seed_{seed}.json"
    result_json = json.dumps(result, indent=2)
    result_path.write_text(result_json, encoding="utf-8")
    (out_dir / "result.json").write_text(result_json, encoding="utf-8")

    history = result.get("history", [])
    history_path = out_dir / f"history_seed_{seed}.csv"
    write_history_csv(history, history_path)
    write_history_csv(history, out_dir / "history.csv")

    metrics_row = build_metrics_row(MODEL_NAME, dataset_path, seed, result)
    save_metrics(str(out_dir / "results.csv"), metrics_row)

    report_data = {
        "model": MODEL_NAME,
        "dataset": dataset_name,
        "seed": seed,
        "dataset_path": str(dataset_path),
        "result_file": str(result_path),
        "history_file": str(history_path),
        "summary": metrics_row,
    }
    write_report_json(out_dir / f"report_seed_{seed}.json", report_data)
    write_report_json(out_dir / "report.json", report_data)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline_a experiments across generated datasets.")
    parser.add_argument("--dataset", action="append", help="Optional dataset path. Can be passed multiple times.")
    parser.add_argument("--start-seed", type=int, default=2000)
    parser.add_argument("--end-seed", type=int, default=2019)
    parser.add_argument("--pop-size", type=int, default=80)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--cx-prob", type=float, default=0.9)
    parser.add_argument("--mut-prob", type=float, default=0.2)
    parser.add_argument("--max-boxes-per-route", type=int, default=48)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset_paths = order_dataset_paths([Path(path) for path in args.dataset]) if args.dataset else discover_generated_datasets()
    if not dataset_paths:
        raise FileNotFoundError("No generated datasets found in VRP/generated_datasets")

    for dataset_path in dataset_paths:
        print(f"Running {MODEL_NAME} on dataset {Path(dataset_path).name}")
        for seed in range(args.start_seed, args.end_seed + 1):
            print(f"Running baseline_a seed {seed}", flush=True)
            cfg = {
                "dataset_path": str(dataset_path),
                "seed": seed,
                "pop_size": args.pop_size,
                "gens": args.gens,
                "cx_prob": args.cx_prob,
                "mut_prob": args.mut_prob,
                "max_boxes_per_route": args.max_boxes_per_route,
                "verbose": args.verbose,
            }
            result = run_from_config(cfg)
            print(
                f"Completed baseline_a seed {seed} | "
                f"Score={result['best_score']:.2f} | "
                f"Runtime={result['runtime_seconds']:.2f}s",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
