import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comparison_models.common.experiment_utils import (
    ABLATION_OUTPUTS_ROOT,
    COMPARISON_ROOT,
    MODEL_ORDER,
    RESULTS_ROOT,
    build_summary_table,
    export_dataset_result_views,
    load_all_results,
    write_csv_rows,
)
from comparison_models.common.paper_reporting import (
    ABLATION_LABELS,
    ABLATION_MODEL_ORDER,
    REPRESENTATIVE_DATASET_SIZES,
    aggregate_summary_by_size,
    pct_change,
    ratio,
)


MODEL_LABELS = {
    "baseline_a": "Baseline A",
    "baseline_b": "Baseline B",
    "baseline_c": "Baseline C",
    "proposed_model": "Proposed Model",
    **ABLATION_LABELS,
}


def _round(value, digits):
    return None if value is None else round(value, digits)


def _summary_map(dataset_rows: list[dict]) -> dict[str, dict]:
    return {row["model"]: row for row in dataset_rows}


def _build_table_1(dataset_rows: list[dict]) -> list[dict]:
    rows = []
    for model_name in MODEL_ORDER:
        row = next((item for item in dataset_rows if item["model"] == model_name), None)
        if not row:
            continue
        rows.append(
            {
                "Method": MODEL_LABELS[model_name],
                "Avg Score": _round(row["avg_score"], 2),
                "Avg Distance": _round(row["avg_distance"], 2),
                "Avg Runtime": _round(row["avg_runtime"], 2),
                "Feasibility Rate": _round(row["avg_feasibility_rate"], 3),
                "Std Dev": _round(row["std_dev_score"], 2),
            }
        )
    return rows


def _build_table_2(dataset_rows: list[dict]) -> list[dict]:
    rows = []
    for model_name in MODEL_ORDER:
        row = next((item for item in dataset_rows if item["model"] == model_name), None)
        if not row:
            continue
        rows.append(
            {
                "Method": MODEL_LABELS[model_name],
                "Best Score": _round(row["best_score"], 2),
                "Worst Score": _round(row["worst_score"], 2),
                "Best Runtime": _round(row["best_runtime"], 2),
                "Worst Runtime": _round(row["worst_runtime"], 2),
            }
        )
    return rows


def _pct_change(old_value, new_value, higher_is_better: bool) -> str:
    if old_value in (None, 0) or new_value is None:
        return "n/a"
    raw = ((new_value - old_value) / old_value) * 100
    if not higher_is_better:
        raw = -raw
    return f"{raw:.2f}%"


def _describe_change(label: str, old_value, new_value, higher_is_better: bool, digits: int = 2) -> str:
    if old_value is None or new_value is None:
        return f"{label} unavailable"
    if old_value == 0:
        return f"{label} {round(old_value, digits)} -> {round(new_value, digits)}"

    raw = ((new_value - old_value) / old_value) * 100
    if abs(raw) < 0.005:
        return f"{label} unchanged"
    improved = raw > 0 if higher_is_better else raw < 0
    direction = "improved" if improved else "decreased" if higher_is_better else "increased"
    return f"{label} {direction} by {abs(raw):.2f}%"


def _build_table_3(dataset_rows: list[dict]) -> list[dict]:
    summary = _summary_map(dataset_rows)
    rows = []

    baseline_a = summary.get("baseline_a")
    baseline_b = summary.get("baseline_b")
    baseline_c = summary.get("baseline_c")
    proposed = summary.get("proposed_model")

    if baseline_a and baseline_b:
        rows.append(
            {
                "Feature Added": "Packing Validation",
                "Improvement Observed": (
                    f"{_describe_change('Feasibility rate', baseline_a.get('avg_feasibility_rate'), baseline_b.get('avg_feasibility_rate'), True, 3)}, "
                    f"{_describe_change('avg score', baseline_a.get('avg_score'), baseline_b.get('avg_score'), False)}, "
                    f"{_describe_change('runtime', baseline_a.get('avg_runtime'), baseline_b.get('avg_runtime'), False)}"
                ),
            }
        )

    if baseline_b and baseline_c:
        rows.append(
            {
                "Feature Added": "Strong Penalty",
                "Improvement Observed": (
                    f"{_describe_change('Feasibility rate', baseline_b.get('avg_feasibility_rate'), baseline_c.get('avg_feasibility_rate'), True, 3)}, "
                    f"{_describe_change('score stability', baseline_b.get('std_dev_score'), baseline_c.get('std_dev_score'), False)}, "
                    f"{_describe_change('avg unpacked boxes', baseline_b.get('avg_unpacked_boxes'), baseline_c.get('avg_unpacked_boxes'), False)}"
                ),
            }
        )

    if baseline_c and proposed:
        rows.append(
            {
                "Feature Added": "Adaptive Decoding + Route Repair",
                "Improvement Observed": (
                    f"{_describe_change('Avg distance', baseline_c.get('avg_distance'), proposed.get('avg_distance'), False)}, "
                    f"{_describe_change('Minimum fill rate', baseline_c.get('min_route_fill'), proposed.get('min_route_fill'), True, 3)}, "
                    f"{_describe_change('Route fill std', baseline_c.get('avg_route_fill_std'), proposed.get('avg_route_fill_std'), False, 3)}, "
                    f"{_describe_change('Tiny route count', baseline_c.get('avg_tiny_route_count'), proposed.get('avg_tiny_route_count'), False)}, "
                    f"{_describe_change('Overflow route count', baseline_c.get('avg_overflow_route_count'), proposed.get('avg_overflow_route_count'), False)}, "
                    f"{_describe_change('Merged route count', baseline_c.get('avg_merged_route_count'), proposed.get('avg_merged_route_count'), True)}"
                ),
            }
        )

    return rows


def _build_tradeoff_table(summary_rows: list[dict]) -> list[dict]:
    aggregated = aggregate_summary_by_size(
        summary_rows,
        model_names=["baseline_c", "proposed_model"],
        sizes=REPRESENTATIVE_DATASET_SIZES,
    )
    by_key = {(row["num_customers"], row["model"]): row for row in aggregated}

    rows = []
    for size in REPRESENTATIVE_DATASET_SIZES:
        baseline = by_key.get((size, "baseline_c"))
        proposed = by_key.get((size, "proposed_model"))
        if not baseline or not proposed:
            continue

        distance_gap = pct_change(baseline.get("avg_distance"), proposed.get("avg_distance"), higher_is_better=False)
        if distance_gap is not None:
            distance_gap = -distance_gap
        runtime_ratio = ratio(proposed.get("avg_runtime"), baseline.get("avg_runtime"))

        rows.append(
            {
                "Dataset Size": size,
                "Baseline C Avg Distance": _round(baseline.get("avg_distance"), 2),
                "Proposed Avg Distance": _round(proposed.get("avg_distance"), 2),
                "Distance Gap vs Baseline C (%)": _round(distance_gap, 2),
                "Minimum Fill Improvement (%)": _round(
                    pct_change(baseline.get("min_route_fill"), proposed.get("min_route_fill"), higher_is_better=True),
                    2,
                ),
                "Route Fill Std Improvement (%)": _round(
                    pct_change(baseline.get("avg_route_fill_std"), proposed.get("avg_route_fill_std"), higher_is_better=False),
                    2,
                ),
                "Route Count Reduction (%)": _round(
                    pct_change(baseline.get("avg_route_count"), proposed.get("avg_route_count"), higher_is_better=False),
                    2,
                ),
                "Runtime Ratio (Proposed/Baseline C)": _round(runtime_ratio, 3),
            }
        )
    return rows


def _build_ablation_table(ablation_summary_rows: list[dict]) -> list[dict]:
    aggregated = aggregate_summary_by_size(
        ablation_summary_rows,
        model_names=ABLATION_MODEL_ORDER,
        sizes=REPRESENTATIVE_DATASET_SIZES,
    )
    rows = []
    for row in aggregated:
        rows.append(
            {
                "Dataset Size": row["num_customers"],
                "Variant": MODEL_LABELS.get(row["model"], row["model"]),
                "Datasets Aggregated": row.get("dataset_count"),
                "Avg Distance": _round(row.get("avg_distance"), 2),
                "Min Route Fill": _round(row.get("min_route_fill"), 3),
                "Avg Route Fill Std": _round(row.get("avg_route_fill_std"), 3),
                "Avg Route Count": _round(row.get("avg_route_count"), 2),
                "Avg Overflow Route Count": _round(row.get("avg_overflow_route_count"), 2),
                "Avg Runtime (s)": _round(row.get("avg_runtime"), 2),
            }
        )
    return rows


def main() -> int:
    results_rows = load_all_results()
    export_dataset_result_views(results_rows)
    summary_rows = build_summary_table(results_rows)

    final_rows = []
    all_table_1 = []
    all_table_2 = []

    datasets = sorted({row["dataset"] for row in summary_rows})
    for dataset_name in datasets:
        dataset_rows = [row for row in summary_rows if row["dataset"] == dataset_name]
        dataset_dir = RESULTS_ROOT / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        table_1 = _build_table_1(dataset_rows)
        table_2 = _build_table_2(dataset_rows)
        table_3 = _build_table_3(dataset_rows)

        write_csv_rows(dataset_dir / "table_1_summary.csv", table_1)
        write_csv_rows(dataset_dir / "table_2_extremes.csv", table_2)
        write_csv_rows(dataset_dir / "table_3_feature_improvements.csv", table_3)

        for row in dataset_rows:
            rounded_row = {
                "dataset": row["dataset"],
                "num_customers": row["num_customers"],
                "model": MODEL_LABELS.get(row["model"], row["model"]),

                "avg_score": _round(row["avg_score"], 2),
                "avg_distance": _round(row["avg_distance"], 2),
                "avg_runtime": _round(row["avg_runtime"], 2),

                "avg_feasibility_rate": _round(row["avg_feasibility_rate"], 3),
                "avg_fill_rate": _round(row["avg_fill_rate"], 3),
                "avg_route_fill": _round(row["avg_route_fill"], 3),
                "min_route_fill": _round(row["min_route_fill"], 3),
                "max_route_fill": _round(row["max_route_fill"], 3),

                "avg_route_count": _round(row["avg_route_count"], 2),
                "avg_tiny_route_count": _round(row.get("avg_tiny_route_count"), 2),
                "avg_overflow_route_count": _round(row.get("avg_overflow_route_count"), 2),
                "avg_merged_route_count": _round(row.get("avg_merged_route_count"), 2),

                "avg_route_fill_std": _round(row.get("avg_route_fill_std"), 3),
                "avg_route_balance_penalty": _round(row.get("avg_route_balance_penalty"), 2),
                "avg_fill_balance_penalty": _round(row.get("avg_fill_balance_penalty"), 2),

                "avg_customers_per_route": _round(row.get("avg_customers_per_route"), 2),
                "avg_boxes_per_route": _round(row.get("avg_boxes_per_route"), 2),

                "avg_unpacked_boxes": _round(row["avg_unpacked_boxes"], 2),

                "std_dev_score": _round(row["std_dev_score"], 2),
                "std_dev_runtime": _round(row["std_dev_runtime"], 2),

                "best_score": _round(row["best_score"], 2),
                "worst_score": _round(row["worst_score"], 2),
                "best_runtime": _round(row["best_runtime"], 2),
                "worst_runtime": _round(row["worst_runtime"], 2),
            }
            final_rows.append(rounded_row)

        for row in table_1:
            combined = dict(row)
            combined["Dataset"] = dataset_name
            all_table_1.append(combined)

        for row in table_2:
            combined = dict(row)
            combined["Dataset"] = dataset_name
            all_table_2.append(combined)

    final_rows.sort(key=lambda row: (row.get("num_customers") or 0, row["dataset"], row["model"]))

    final_table_path = COMPARISON_ROOT / "final_comparison_table.csv"
    write_csv_rows(final_table_path, final_rows)
    write_csv_rows(RESULTS_ROOT / "table_1_summary_all_datasets.csv", all_table_1)
    write_csv_rows(RESULTS_ROOT / "table_2_extremes_all_datasets.csv", all_table_2)

    tradeoff_dir = RESULTS_ROOT / "tradeoff"
    tradeoff_dir.mkdir(parents=True, exist_ok=True)
    tradeoff_rows = _build_tradeoff_table(summary_rows)
    write_csv_rows(tradeoff_dir / "baseline_c_vs_proposed_tradeoff_summary.csv", tradeoff_rows)

    ablation_dir = RESULTS_ROOT / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    ablation_rows = load_all_results(ABLATION_OUTPUTS_ROOT)
    if ablation_rows:
        ablation_summary_rows = build_summary_table(ablation_rows)
        write_csv_rows(ablation_dir / "summary.csv", ablation_summary_rows)
        write_csv_rows(ablation_dir / "ablation_summary_table.csv", _build_ablation_table(ablation_summary_rows))

    print(f"Saved final comparison table to {final_table_path}")
    print(f"Saved dataset-specific tables under {RESULTS_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
