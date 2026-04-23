from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_ROOT = PROJECT_ROOT / "comparison_models"
OUTPUTS_ROOT = COMPARISON_ROOT / "outputs"
ABLATION_OUTPUTS_ROOT = COMPARISON_ROOT / "outputs_ablation"
RESULTS_ROOT = COMPARISON_ROOT / "results"
FINAL_PLOTS_ROOT = COMPARISON_ROOT / "final_plots"
GRAPHS_ROOT = COMPARISON_ROOT / "graphs"
GENERATED_DATASETS_ROOT = PROJECT_ROOT / "VRP" / "generated_datasets"
MODEL_ORDER = ["baseline_a", "baseline_b", "baseline_c", "proposed_model"]


def _serialize_csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def extract_customer_count(dataset_name: str) -> int | None:
    match = re.match(r"XML(\d+)_", dataset_name)
    return int(match.group(1)) if match else None


def extract_family_group(dataset_name: str) -> str | None:
    match = re.match(r"XML\d+_(\d{4})_", dataset_name)
    return match.group(1) if match else None


def discover_generated_datasets(dataset_dir: Path | None = None) -> list[Path]:
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else GENERATED_DATASETS_ROOT
    if not dataset_dir.exists():
        return []
    return order_dataset_paths(dataset_dir.rglob("*_merged_with_boxes_norm.json"))


def order_dataset_paths(paths) -> list[Path]:
    return sorted(
        [Path(path) for path in paths],
        key=lambda path: (extract_customer_count(path.stem) or float("inf"), path.stem),
    )


def resolve_dataset_path(dataset_path: str | Path) -> Path:
    if dataset_path is None:
        raise ValueError("config must contain 'dataset_path'")
    dataset_path = Path(dataset_path)
    if dataset_path.exists():
        return dataset_path.resolve()

    candidate = PROJECT_ROOT / dataset_path
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Dataset file not found: {dataset_path} (also tried {candidate})"
    )


def get_dataset_name(dataset_path: str | Path) -> str:
    return resolve_dataset_path(dataset_path).stem


def get_output_dir(model_name: str, dataset_path: str | Path, outputs_root: Path | None = None) -> Path:
    outputs_root = outputs_root or OUTPUTS_ROOT
    return outputs_root / model_name / get_dataset_name(dataset_path)


def _load_dataset_box_count(dataset_path: str | Path) -> int:
    dataset_file = resolve_dataset_path(dataset_path)
    with dataset_file.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    customers = data.get("customers", [])
    if not customers:
        return len(data.get("boxes", []))

    real_customers = customers[1:] if len(customers) > 1 else []
    assigned_box_ids = {
        str(box_id)
        for customer in real_customers
        for box_id in customer.get("assigned_boxes", [])
    }
    if assigned_box_ids:
        return len(assigned_box_ids)
    return len(data.get("boxes", []))


def _is_routing_only_model(model_name: str) -> bool:
    return model_name == "baseline_a"


def recommended_max_workers(dataset_paths: list[Path]) -> int:
    customer_counts = [extract_customer_count(Path(path).stem) or 0 for path in dataset_paths]
    return 4 if any(count >= 500 for count in customer_counts) else 6


def _derive_route_metrics(best_info: dict) -> dict:
    routes = list(best_info.get("routes") or [])
    route_count = int(best_info.get("route_count") or len(routes) or 0)
    boxes_total = int(best_info.get("boxes_total") or 0)
    boxes_packed = int(best_info.get("boxes_packed") or 0)
    unpacked_boxes = int(best_info.get("unpacked_boxes") or max(0, boxes_total - boxes_packed))

    route_box_counts = [
        int(route.get("boxes_total") or 0)
        for route in routes
    ]
    route_fill_rates = [
        float(route.get("fill_rate") or 0.0)
        for route in routes
    ]

    if route_fill_rates:
        avg_route_fill = float(best_info.get("avg_route_fill") or statistics.fmean(route_fill_rates))
        min_route_fill = float(best_info.get("min_route_fill") or min(route_fill_rates))
        max_route_fill = float(best_info.get("max_route_fill") or max(route_fill_rates))
    else:
        fallback_fill = float(best_info.get("avg_fill_rate", best_info.get("fill_rate") or 0.0))
        avg_route_fill = float(best_info.get("avg_route_fill") or fallback_fill)
        min_route_fill = float(best_info.get("min_route_fill") or fallback_fill)
        max_route_fill = float(best_info.get("max_route_fill") or fallback_fill)

    avg_route_boxes = float(best_info.get("avg_route_boxes") or statistics.fmean(route_box_counts)) if route_box_counts else float(best_info.get("avg_route_boxes") or 0.0)
    min_route_boxes = int(best_info.get("min_route_boxes") or min(route_box_counts)) if route_box_counts else int(best_info.get("min_route_boxes") or 0)
    max_route_boxes = int(best_info.get("max_route_boxes") or max(route_box_counts)) if route_box_counts else int(best_info.get("max_route_boxes") or 0)
    avg_fill_rate = float(best_info.get("avg_fill_rate") or avg_route_fill)
    min_fill_rate = float(best_info.get("min_fill_rate") or min_route_fill)
    max_fill_rate = float(best_info.get("max_fill_rate") or max_route_fill)
    fill_balance_penalty = float(best_info.get("fill_balance_penalty") or max(0.0, max_fill_rate - min_fill_rate))
    route_fill_std = float(best_info.get("route_fill_std") or (statistics.pstdev(route_fill_rates) if len(route_fill_rates) > 1 else 0.0))
    route_customer_counts = [len(route.get("route") or []) for route in routes]
    avg_customers_per_route = float(best_info.get("avg_customers_per_route") or statistics.fmean(route_customer_counts)) if route_customer_counts else float(best_info.get("avg_customers_per_route") or 0.0)
    min_customers_per_route = int(best_info.get("min_customers_per_route") or min(route_customer_counts)) if route_customer_counts else int(best_info.get("min_customers_per_route") or 0)
    max_customers_per_route = int(best_info.get("max_customers_per_route") or max(route_customer_counts)) if route_customer_counts else int(best_info.get("max_customers_per_route") or 0)
    tiny_route_count = int(best_info.get("tiny_route_count") or 0)
    merged_route_count = int(best_info.get("merged_route_count") or 0)
    overflow_route_count = int(best_info.get("overflow_route_count") or 0)

    return {
        "route_count": route_count,
        "unpacked_boxes": unpacked_boxes,
        "avg_fill_rate": avg_fill_rate,
        "min_fill_rate": min_fill_rate,
        "max_fill_rate": max_fill_rate,
        "avg_route_fill": avg_route_fill,
        "min_route_fill": min_route_fill,
        "max_route_fill": max_route_fill,
        "fill_balance_penalty": fill_balance_penalty,
        "route_fill_std": route_fill_std,
        "avg_route_boxes": avg_route_boxes,
        "min_route_boxes": min_route_boxes,
        "max_route_boxes": max_route_boxes,
        "avg_boxes_per_route": float(best_info.get("avg_boxes_per_route") or avg_route_boxes),
        "min_boxes_per_route": int(best_info.get("min_boxes_per_route") or min_route_boxes),
        "max_boxes_per_route": int(best_info.get("max_boxes_per_route") or max_route_boxes),
        "avg_customers_per_route": avg_customers_per_route,
        "min_customers_per_route": min_customers_per_route,
        "max_customers_per_route": max_customers_per_route,
        "tiny_route_count": tiny_route_count,
        "merged_route_count": merged_route_count,
        "overflow_route_count": overflow_route_count,
    }


def normalize_result_for_reporting(model_name: str, dataset_path: str | Path, result: dict) -> dict:
    normalized = dict(result)
    best_info = dict(normalized.get("best_info") or {})
    normalized["best_info"] = best_info

    total_distance = best_info.get("total_distance")
    dataset_box_count = _load_dataset_box_count(dataset_path)

    if _is_routing_only_model(model_name):
        normalized["best_score"] = total_distance
        best_info["total_distance"] = total_distance
        best_info["packing_time_seconds"] = 0.0
        best_info["feasible_routes"] = 0
        best_info["infeasible_count"] = 0
        best_info["infeasible_routes"] = 0
        best_info["feasibility_rate"] = 0.0
        best_info["boxes_total"] = dataset_box_count
        best_info["boxes_packed"] = 0
        best_info["fill_rate"] = 0.0

    best_info.update(_derive_route_metrics(best_info))

    packing_time_seconds = float(best_info.get("packing_time_seconds") or 0.0)
    feasible_routes = int(best_info.get("feasible_routes") or 0)
    infeasible_routes = int(best_info.get("infeasible_routes", best_info.get("infeasible_count") or 0) or 0)
    boxes_total = int(best_info.get("boxes_total") or 0)
    boxes_packed = int(best_info.get("boxes_packed") or 0)
    fill_rate = float(best_info.get("fill_rate") or 0.0)
    feasibility_rate = float(best_info.get("feasibility_rate") or 0.0)

    if boxes_total < 0:
        raise ValueError(f"{model_name}: boxes_total cannot be negative")
    if boxes_packed > boxes_total:
        raise ValueError(f"{model_name}: boxes_packed cannot exceed boxes_total")
    if not (0.0 <= fill_rate <= 1.0):
        raise ValueError(f"{model_name}: fill_rate must be between 0 and 1")
    if not (0.0 <= feasibility_rate <= 1.0):
        raise ValueError(f"{model_name}: feasibility_rate must be between 0 and 1")
    if boxes_packed == 0 and fill_rate == 1.0:
        raise ValueError(f"{model_name}: fill_rate cannot be 1 when boxes_packed is 0")
    if packing_time_seconds != 0.0 and _is_routing_only_model(model_name):
        raise ValueError(f"{model_name}: packing_time_seconds must be 0 for routing-only models")
    if _is_routing_only_model(model_name) and (feasible_routes + infeasible_routes) != 0:
        raise ValueError(f"{model_name}: routing-only models must not report feasible or infeasible routes")
    if (feasible_routes + infeasible_routes) == 0 and not _is_routing_only_model(model_name):
        raise ValueError(f"{model_name}: packing-aware models must report evaluated route counts")

    return normalized


def build_metrics_row(model_name: str, dataset_path: str | Path, seed: int, result: dict) -> dict:
    dataset_file = resolve_dataset_path(dataset_path)
    dataset_name = dataset_file.stem
    validated_result = normalize_result_for_reporting(model_name, dataset_path, result)
    best_info = validated_result.get("best_info", {})
    num_customers = extract_customer_count(dataset_name)

    return {
        "dataset": dataset_name,
        "dataset_file": dataset_file.name,
        "dataset_path": str(dataset_file),
        "num_customers": num_customers,
        "family_group": extract_family_group(dataset_name),
        "model": model_name,
        "seed": seed,
        "best_score": validated_result.get("best_score"),
        "best_distance": best_info.get("total_distance"),
        "runtime_seconds": validated_result.get("runtime_seconds"),
        "packing_time_seconds": best_info.get("packing_time_seconds"),
        "feasible_routes": best_info.get("feasible_routes"),
        "infeasible_routes": best_info.get("infeasible_routes", best_info.get("infeasible_count")),
        "feasibility_rate": best_info.get("feasibility_rate"),
        "boxes_total": best_info.get("boxes_total"),
        "boxes_packed": best_info.get("boxes_packed"),
        "fill_rate": best_info.get("fill_rate"),
        "avg_fill_rate": best_info.get("avg_fill_rate"),
        "min_fill_rate": best_info.get("min_fill_rate"),
        "max_fill_rate": best_info.get("max_fill_rate"),
        "fill_balance_penalty": best_info.get("fill_balance_penalty"),
        "route_fill_std": best_info.get("route_fill_std"),
        "unpacked_boxes": best_info.get("unpacked_boxes"),
        "route_count": best_info.get("route_count"),
        "avg_route_fill": best_info.get("avg_route_fill"),
        "min_route_fill": best_info.get("min_route_fill"),
        "max_route_fill": best_info.get("max_route_fill"),
        "avg_route_boxes": best_info.get("avg_route_boxes"),
        "min_route_boxes": best_info.get("min_route_boxes"),
        "max_route_boxes": best_info.get("max_route_boxes"),
        "avg_boxes_per_route": best_info.get("avg_boxes_per_route"),
        "min_boxes_per_route": best_info.get("min_boxes_per_route"),
        "max_boxes_per_route": best_info.get("max_boxes_per_route"),
        "avg_customers_per_route": best_info.get("avg_customers_per_route"),
        "min_customers_per_route": best_info.get("min_customers_per_route"),
        "max_customers_per_route": best_info.get("max_customers_per_route"),
        "tiny_route_count": best_info.get("tiny_route_count"),
        "merged_route_count": best_info.get("merged_route_count"),
        "overflow_route_count": best_info.get("overflow_route_count"),
        "route_balance_penalty": best_info.get("route_balance_penalty"),
        "chosen_split_limit": best_info.get("chosen_split_limit"),
        "chosen_split_strategy": best_info.get("chosen_split_strategy"),
        "chosen_distance_limit": best_info.get("chosen_distance_limit"),
        "number_of_split_candidates_tested": best_info.get("number_of_split_candidates_tested"),
        "candidate_route_counts": _serialize_csv_value(best_info.get("candidate_route_counts")),
        "candidate_fill_rates": _serialize_csv_value(best_info.get("candidate_fill_rates")),
        "candidate_distances": _serialize_csv_value(best_info.get("candidate_distances")),
        "max_boxes_per_route": result.get("max_boxes_per_route"),
    }


def write_history_csv(history: list[float], output_path: Path) -> None:
    lines = ["gen,best_score"]
    for index, value in enumerate(history, start=1):
        lines.append(f"{index},{value}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_report_json(output_path: Path, report_data: dict) -> None:
    output_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")


def _coerce_value(value: str):
    if value in ("", "None", None):
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except (TypeError, ValueError):
        return value


def _enrich_row_from_result_json(row: dict, results_dir: Path) -> dict:
    seed = row.get("seed")
    model_name = row.get("model")
    dataset_path = row.get("dataset_path")
    if not isinstance(seed, int) or not model_name or not dataset_path:
        return row

    result_path = results_dir / f"result_seed_{seed}.json"
    if not result_path.exists():
        return row

    try:
        with result_path.open("r", encoding="utf-8-sig") as handle:
            result = json.load(handle)
        enriched = build_metrics_row(model_name, dataset_path, seed, result)
    except Exception:
        return row

    merged = dict(enriched)
    merged.update({key: value for key, value in row.items() if value is not None})
    return merged


def load_all_results(outputs_root: Path | None = None) -> list[dict]:
    outputs_root = outputs_root or OUTPUTS_ROOT
    all_rows: dict[tuple, dict] = {}

    if not outputs_root.exists():
        return []

    for results_csv in outputs_root.glob("*/*/results.csv"):
        with results_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed = {key: _coerce_value(value) for key, value in row.items()}
                if "dataset" not in parsed or parsed["dataset"] is None:
                    parsed["dataset"] = results_csv.parent.name
                if "model" not in parsed or parsed["model"] is None:
                    parsed["model"] = results_csv.parent.parent.name
                if "num_customers" not in parsed or parsed["num_customers"] is None:
                    parsed["num_customers"] = extract_customer_count(parsed["dataset"])
                if "family_group" not in parsed or parsed["family_group"] is None:
                    parsed["family_group"] = extract_family_group(parsed["dataset"])
                parsed = _enrich_row_from_result_json(parsed, results_csv.parent)
                key = (parsed.get("dataset"), parsed.get("model"), parsed.get("seed"))
                all_rows[key] = parsed
    return list(all_rows.values())


def get_completed_seeds(model_name: str, dataset_path: str | Path, outputs_root: Path | None = None) -> set[int]:
    outputs_root = outputs_root or OUTPUTS_ROOT
    results_csv = outputs_root / model_name / get_dataset_name(dataset_path) / "results.csv"
    if not results_csv.exists():
        return set()

    completed: set[int] = set()
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seed = _coerce_value(row.get("seed"))
            if isinstance(seed, int):
                completed.add(seed)
    return completed


def is_result_complete(model_name: str, dataset_path: str | Path, seed: int, outputs_root: Path | None = None) -> bool:
    outputs_root = outputs_root or OUTPUTS_ROOT
    out_dir = outputs_root / model_name / get_dataset_name(dataset_path)
    required_files = [
        out_dir / f"result_seed_{seed}.json",
        out_dir / f"history_seed_{seed}.csv",
        out_dir / f"report_seed_{seed}.json",
        out_dir / "results.csv",
    ]
    if not all(path.exists() for path in required_files):
        return False
    return seed in get_completed_seeds(model_name, dataset_path, outputs_root=outputs_root)


def build_summary_table(results_rows: list[dict]) -> list[dict]:
    def avg(values: list[float]) -> float | None:
        return statistics.fmean(values) if values else None

    def stdev(values: list[float]) -> float | None:
        return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None

    grouped: dict[tuple, list[dict]] = {}
    for row in results_rows:
        key = (row.get("dataset"), row.get("num_customers"), row.get("model"))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for (dataset, num_customers, model), rows in grouped.items():
        scores = [row["best_score"] for row in rows if row.get("best_score") is not None]
        distances = [row["best_distance"] for row in rows if row.get("best_distance") is not None]
        runtimes = [row["runtime_seconds"] for row in rows if row.get("runtime_seconds") is not None]
        feasibility = [row["feasibility_rate"] for row in rows if row.get("feasibility_rate") is not None]
        fill_rates = [row["fill_rate"] for row in rows if row.get("fill_rate") is not None]
        unpacked_boxes = [row["unpacked_boxes"] for row in rows if row.get("unpacked_boxes") is not None]
        route_counts = [row["route_count"] for row in rows if row.get("route_count") is not None]
        avg_route_fills = [row["avg_route_fill"] for row in rows if row.get("avg_route_fill") is not None]
        min_route_fills = [row["min_route_fill"] for row in rows if row.get("min_route_fill") is not None]
        max_route_fills = [row["max_route_fill"] for row in rows if row.get("max_route_fill") is not None]
        route_fill_stds = [row["route_fill_std"] for row in rows if row.get("route_fill_std") is not None]
        tiny_route_counts = [row["tiny_route_count"] for row in rows if row.get("tiny_route_count") is not None]
        merged_route_counts = [row["merged_route_count"] for row in rows if row.get("merged_route_count") is not None]
        overflow_route_counts = [row["overflow_route_count"] for row in rows if row.get("overflow_route_count") is not None]
        route_balance_penalties = [row["route_balance_penalty"] for row in rows if row.get("route_balance_penalty") is not None]
        fill_balance_penalties = [row["fill_balance_penalty"] for row in rows if row.get("fill_balance_penalty") is not None]
        avg_boxes_per_route = [row["avg_boxes_per_route"] for row in rows if row.get("avg_boxes_per_route") is not None]
        avg_customers_per_route = [row["avg_customers_per_route"] for row in rows if row.get("avg_customers_per_route") is not None]

        summary_rows.append(
            {
                "dataset": dataset,
                "num_customers": num_customers,
                "model": model,
                "avg_score": avg(scores),
                "avg_distance": avg(distances),
                "avg_runtime": avg(runtimes),
                "avg_feasibility_rate": avg(feasibility),
                "avg_fill_rate": avg(fill_rates),
                "avg_unpacked_boxes": avg(unpacked_boxes),
                "avg_route_count": avg(route_counts),
                "avg_route_fill": avg(avg_route_fills),
                "min_route_fill": min(min_route_fills) if min_route_fills else None,
                "max_route_fill": max(max_route_fills) if max_route_fills else None,
                "avg_route_fill_std": avg(route_fill_stds),
                "avg_tiny_route_count": avg(tiny_route_counts),
                "avg_merged_route_count": avg(merged_route_counts),
                "avg_overflow_route_count": avg(overflow_route_counts),
                "avg_route_balance_penalty": avg(route_balance_penalties),
                "avg_fill_balance_penalty": avg(fill_balance_penalties),
                "avg_boxes_per_route": avg(avg_boxes_per_route),
                "avg_customers_per_route": avg(avg_customers_per_route),
                "std_dev_score": stdev(scores),
                "std_dev_runtime": stdev(runtimes),
                "best_score": min(scores) if scores else None,
                "worst_score": max(scores) if scores else None,
                "best_runtime": min(runtimes) if runtimes else None,
                "worst_runtime": max(runtimes) if runtimes else None,
            }
        )

    return sorted(summary_rows, key=lambda row: (row.get("num_customers") or 0, row.get("dataset") or "", row.get("model") or ""))


def write_csv_rows(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_dataset_result_views(
    results_rows: list[dict],
    results_root: Path | None = None,
    model_order: list[str] | None = None,
) -> None:
    results_root = results_root or RESULTS_ROOT
    model_order = model_order or MODEL_ORDER
    datasets = sorted({row.get("dataset") for row in results_rows if row.get("dataset")})

    for dataset_name in datasets:
        dataset_dir = results_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_rows = [row for row in results_rows if row.get("dataset") == dataset_name]

        for model_name in model_order:
            model_rows = [row for row in dataset_rows if row.get("model") == model_name]
            write_csv_rows(dataset_dir / f"{model_name}_results.csv", model_rows)
