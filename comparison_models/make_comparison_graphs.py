import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comparison_models.common.experiment_utils import (
    ABLATION_OUTPUTS_ROOT,
    GRAPHS_ROOT,
    MODEL_ORDER,
    OUTPUTS_ROOT,
    build_summary_table,
    load_all_results,
)
from comparison_models.common.paper_reporting import (
    ABLATION_LABELS,
    ABLATION_MODEL_ORDER,
    REPRESENTATIVE_DATASET_SIZES,
    ROUTE_COMPOSITION_SIZES,
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

MODEL_COLORS = {
    "Baseline A": "#4C78A8",
    "Baseline B": "#F58518",
    "Baseline C": "#54A24B",
    "Proposed Model": "#E45756",
    "Proposed vs Baseline C": "#E45756",
    "Full Proposed": "#E45756",
    "No Adaptive Decoding": "#4C78A8",
    "No Tiny-Route Repair": "#F58518",
    "No Relocation Repair": "#54A24B",
    "No Route-Balance Mutation": "#B279A2",
    "No Final-Best Refinement": "#72B7B2",
    "Distance vs Full": "#E45756",
    "Min Fill vs Full": "#4C78A8",
    "Runtime vs Full": "#B279A2",
}

MODEL_MARKERS = {
    "Baseline A": "o",
    "Baseline B": "s",
    "Baseline C": "^",
    "Proposed Model": "D",
}

PLOT_DPI = 170
AXIS_FONT_SIZE = 13
TITLE_FONT_SIZE = 16
SUBTITLE_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10
MARKER_SIZE = 28
PAPER_GRAPHS_DIRNAME = "paper"


def _filter_rows(rows: list[dict], datasets: list[str] | None, models: list[str] | None) -> list[dict]:
    filtered = rows
    if datasets:
        needles = [item.lower() for item in datasets]
        filtered = [row for row in filtered if any(needle in str(row.get("dataset", "")).lower() for needle in needles)]
    if models:
        filtered = [row for row in filtered if row.get("model") in models]
    return filtered


def _try_import_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError:
        return None
    return np


def _series_color(label: str) -> str | None:
    return MODEL_COLORS.get(label)


def _style_axes(title: str, xlabel: str | None = None, ylabel: str | None = None, subtitle: str | None = None) -> None:
    import matplotlib.pyplot as plt

    plt.title(title, fontsize=TITLE_FONT_SIZE, pad=subtitle and 24 or 12)
    if subtitle:
        plt.gca().text(
            0.5,
            1.02,
            subtitle,
            transform=plt.gca().transAxes,
            ha="center",
            va="bottom",
            fontsize=SUBTITLE_FONT_SIZE,
            color="#555555",
        )
    if xlabel:
        plt.xlabel(xlabel, fontsize=AXIS_FONT_SIZE)
    if ylabel:
        plt.ylabel(ylabel, fontsize=AXIS_FONT_SIZE)
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.18)
    plt.tick_params(axis="both", labelsize=11)


def _smooth_xy(xs: list[float], ys: list[float], points: int = 200) -> tuple[list[float], list[float]]:
    if len(xs) < 3:
        return xs, ys

    np = _try_import_numpy()
    if np is None:
        return xs, ys

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    if len(set(x_arr.tolist())) < 3:
        return xs, ys

    dense_x = np.linspace(float(x_arr.min()), float(x_arr.max()), points)

    try:
        from scipy.interpolate import PchipInterpolator

        dense_y = PchipInterpolator(x_arr, y_arr)(dense_x)
    except ModuleNotFoundError:
        try:
            from scipy.interpolate import make_interp_spline

            spline = make_interp_spline(x_arr, y_arr, k=min(2, len(x_arr) - 1))
            dense_y = spline(dense_x)
        except ModuleNotFoundError:
            dense_y = np.interp(dense_x, x_arr, y_arr)
        except ValueError:
            dense_y = np.interp(dense_x, x_arr, y_arr)
    except ValueError:
        dense_y = np.interp(dense_x, x_arr, y_arr)

    return dense_x.tolist(), dense_y.tolist()


def _aggregate_rows_by_size(rows: list[dict], column: str) -> dict[str, tuple[list[int], list[float], list[float]]]:
    series_map: dict[str, tuple[list[int], list[float], list[float]]] = {}
    for model_name in MODEL_ORDER:
        grouped: dict[int, list[float]] = {}
        for row in rows:
            if row.get("model") != model_name:
                continue
            num_customers = row.get("num_customers")
            value = row.get(column)
            if num_customers is None or value is None:
                continue
            grouped.setdefault(int(num_customers), []).append(float(value))
        if not grouped:
            continue
        xs = sorted(grouped)
        means = [statistics.fmean(grouped[x]) for x in xs]
        stds = [statistics.pstdev(grouped[x]) if len(grouped[x]) > 1 else 0.0 for x in xs]
        series_map[MODEL_LABELS[model_name]] = (xs, means, stds)
    return series_map


def _aggregate_summary_by_size(summary_rows: list[dict], column: str) -> dict[str, tuple[list[int], list[float], list[float]]]:
    series_map: dict[str, tuple[list[int], list[float], list[float]]] = {}
    for model_name in MODEL_ORDER:
        grouped: dict[int, list[float]] = {}
        for row in summary_rows:
            if row.get("model") != model_name:
                continue
            num_customers = row.get("num_customers")
            value = row.get(column)
            if num_customers is None or value is None:
                continue
            grouped.setdefault(int(num_customers), []).append(float(value))
        if not grouped:
            continue
        xs = sorted(grouped)
        means = [statistics.fmean(grouped[x]) for x in xs]
        stds = [statistics.pstdev(grouped[x]) if len(grouped[x]) > 1 else 0.0 for x in xs]
        series_map[MODEL_LABELS[model_name]] = (xs, means, stds)
    return series_map


def _aggregate_improvement_vs_baseline_c_by_size(
    rows: list[dict],
    column: str,
    higher_is_better: bool,
) -> dict[str, tuple[list[int], list[float], list[float]]]:
    grouped: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        num_customers = row.get("num_customers")
        model_name = row.get("model")
        value = row.get(column)
        if num_customers is None or model_name not in {"baseline_c", "proposed_model"} or value is None:
            continue
        grouped.setdefault(int(num_customers), {}).setdefault(model_name, []).append(float(value))

    xs: list[int] = []
    means: list[float] = []
    stds: list[float] = []
    for size in sorted(grouped):
        baseline_values = grouped[size].get("baseline_c") or []
        proposed_values = grouped[size].get("proposed_model") or []
        if not baseline_values or not proposed_values:
            continue
        baseline_mean = statistics.fmean(baseline_values)
        if baseline_mean == 0:
            continue
        improvements = []
        for proposed_value in proposed_values:
            change = ((proposed_value - baseline_mean) / baseline_mean) * 100
            if not higher_is_better:
                change = -change
            improvements.append(change)
        xs.append(size)
        means.append(statistics.fmean(improvements))
        stds.append(statistics.pstdev(improvements) if len(improvements) > 1 else 0.0)

    if not xs:
        return {}
    return {"Proposed vs Baseline C": (xs, means, stds)}


def _aggregate_route_reduction_by_size(summary_rows: list[dict]) -> tuple[list[str], list[float]]:
    grouped: dict[int, dict[str, list[float]]] = {}
    for row in summary_rows:
        num_customers = row.get("num_customers")
        model_name = row.get("model")
        value = row.get("avg_route_count")
        if num_customers is None or model_name not in {"baseline_c", "proposed_model"} or value is None:
            continue
        grouped.setdefault(int(num_customers), {}).setdefault(model_name, []).append(float(value))

    labels: list[str] = []
    reductions: list[float] = []
    for size in sorted(grouped):
        baseline_values = grouped[size].get("baseline_c") or []
        proposed_values = grouped[size].get("proposed_model") or []
        if not baseline_values or not proposed_values:
            continue
        baseline_mean = statistics.fmean(baseline_values)
        proposed_mean = statistics.fmean(proposed_values)
        reduction = 0.0 if baseline_mean == 0 else ((baseline_mean - proposed_mean) / baseline_mean) * 100
        labels.append(str(size))
        reductions.append(reduction)
    return labels, reductions


def _aggregate_series_vs_baselines_by_size(
    summary_rows: list[dict],
    *,
    proposed_column: str,
    baseline_column: str | None = None,
    baseline_models: list[str],
    higher_is_better: bool = False,
) -> tuple[list[str], dict[str, list[float]]]:
    baseline_column = baseline_column or proposed_column
    grouped: dict[int, dict[str, list[float]]] = {}
    for row in summary_rows:
        num_customers = row.get("num_customers")
        model_name = row.get("model")
        if num_customers is None or model_name not in (set(baseline_models) | {"proposed_model"}):
            continue
        target_column = proposed_column if model_name == "proposed_model" else baseline_column
        value = row.get(target_column)
        if value is None:
            continue
        grouped.setdefault(int(num_customers), {}).setdefault(model_name, []).append(float(value))

    categories: list[str] = []
    series_map = {f"Proposed vs {MODEL_LABELS[model]}": [] for model in baseline_models}
    for size in sorted(grouped):
        proposed_values = grouped[size].get("proposed_model") or []
        if not proposed_values:
            continue
        proposed_mean = statistics.fmean(proposed_values)
        categories.append(str(size))
        for baseline_model in baseline_models:
            baseline_values = grouped[size].get(baseline_model) or []
            if not baseline_values:
                series_map[f"Proposed vs {MODEL_LABELS[baseline_model]}"].append(0.0)
                continue
            baseline_mean = statistics.fmean(baseline_values)
            if baseline_mean == 0:
                series_map[f"Proposed vs {MODEL_LABELS[baseline_model]}"].append(0.0)
                continue
            change = ((baseline_mean - proposed_mean) / baseline_mean) * 100
            if higher_is_better:
                change = ((proposed_mean - baseline_mean) / baseline_mean) * 100
            series_map[f"Proposed vs {MODEL_LABELS[baseline_model]}"].append(change)
    return categories, series_map


def _aggregate_pair_by_size(summary_rows: list[dict], column: str, models: tuple[str, str]) -> tuple[list[str], dict[str, list[float]]]:
    grouped: dict[int, dict[str, list[float]]] = {}
    for row in summary_rows:
        num_customers = row.get("num_customers")
        model_name = row.get("model")
        value = row.get(column)
        if num_customers is None or model_name not in models or value is None:
            continue
        grouped.setdefault(int(num_customers), {}).setdefault(model_name, []).append(float(value))

    categories: list[str] = []
    series_map = {MODEL_LABELS[model]: [] for model in models}
    for size in sorted(grouped):
        categories.append(str(size))
        for model in models:
            values = grouped[size].get(model) or []
            series_map[MODEL_LABELS[model]].append(statistics.fmean(values) if values else 0.0)
    return categories, series_map


def _load_result_json(model_name: str, dataset_name: str, seed: int, *, outputs_root: Path = OUTPUTS_ROOT) -> dict | None:
    result_path = outputs_root / model_name / dataset_name / f"result_seed_{seed}.json"
    if not result_path.exists():
        return None
    try:
        return json.loads(result_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return None


def _representative_dataset_by_size(rows: list[dict], target_size: int) -> str | None:
    dataset_names = sorted({
        str(row["dataset"])
        for row in rows
        if row.get("dataset") and row.get("num_customers") == target_size
    })
    return dataset_names[0] if dataset_names else None


def _representative_seed(rows: list[dict], dataset_name: str, models: tuple[str, ...]) -> int | None:
    shared = None
    for model_name in models:
        model_seeds = {
            int(row["seed"])
            for row in rows
            if row.get("dataset") == dataset_name and row.get("model") == model_name and row.get("seed") is not None
        }
        shared = model_seeds if shared is None else (shared & model_seeds)
    return min(shared) if shared else None


def _route_metric_lists(result: dict) -> dict[str, list[float]]:
    routes = list((result.get("best_info") or {}).get("routes") or [])
    return {
        "boxes_per_route": [float(route.get("boxes_total") or 0.0) for route in routes],
        "customers_per_route": [float(len(route.get("route") or [])) for route in routes],
        "fill_rate": [float(route.get("fill_rate") or 0.0) for route in routes],
    }


def _bar_plot(labels, values, title, ylabel, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not labels or not values:
        return
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color="#4C78A8")
    _style_axes(title, ylabel=ylabel)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def _line_plot(
    series_map: dict[str, tuple[list, list]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    *,
    smooth: bool = False,
    show_zero_line: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    if not series_map:
        return
    plt.figure(figsize=(10, 6))
    plotted = False
    for label, (xs, ys) in series_map.items():
        if not xs or not ys:
            continue
        plotted = True
        color = _series_color(label)
        plot_xs, plot_ys = xs, ys
        if smooth:
            plot_xs, plot_ys = _smooth_xy(list(xs), list(ys))
        plt.plot(plot_xs, plot_ys, linewidth=2.3, color=color, label=label)
        plt.scatter(xs, ys, s=MARKER_SIZE, color=color, edgecolors="white", linewidths=0.8, zorder=3)
    if not plotted:
        plt.close()
        return
    if show_zero_line:
        plt.axhline(0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.65)
    _style_axes(title, xlabel=xlabel, ylabel=ylabel)
    plt.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def _line_plot_with_band(
    series_map: dict[str, tuple[list[float], list[float], list[float]]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    *,
    show_zero_line: bool = False,
    subtitle: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if not series_map:
        return

    np = _try_import_numpy()
    plt.figure(figsize=(10.5, 6.2))
    plotted = False
    for label, (xs, ys, stds) in series_map.items():
        if not xs or not ys:
            continue
        plotted = True
        color = _series_color(label)
        smooth_xs, smooth_ys = _smooth_xy(list(xs), list(ys))
        plt.plot(smooth_xs, smooth_ys, linewidth=2.2, color=color, label=label, alpha=0.9)
        plt.scatter(xs, ys, s=MARKER_SIZE + 6, color=color, edgecolors="white", linewidths=0.9, zorder=3)
        if stds and np is not None and len(xs) >= 2:
            lower = [y - s for y, s in zip(ys, stds)]
            upper = [y + s for y, s in zip(ys, stds)]
            band_xs, band_lower = _smooth_xy(list(xs), lower)
            _, band_upper = _smooth_xy(list(xs), upper)
            plt.fill_between(band_xs, band_lower, band_upper, color=color, alpha=0.12)
    if not plotted:
        plt.close()
        return
    if show_zero_line:
        plt.axhline(0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.65)
    _style_axes(title, xlabel=xlabel, ylabel=ylabel, subtitle=subtitle)
    plt.legend(frameon=False, ncol=2 if len(series_map) > 2 else 1, fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def _scatter_plot(series_map: dict[str, tuple[list[float], list[float]]], title: str, xlabel: str, ylabel: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not series_map:
        return
    plt.figure(figsize=(10, 6))
    plotted = False
    for label, (xs, ys) in series_map.items():
        if not xs or not ys:
            continue
        plotted = True
        plt.scatter(xs, ys, label=label, alpha=0.78, color=_series_color(label))
    if not plotted:
        plt.close()
        return
    _style_axes(title, xlabel=xlabel, ylabel=ylabel)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def _horizontal_bar_with_labels(labels: list[str], values: list[float], title: str, xlabel: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not labels or not values:
        return

    colors = ["#3BA55D" if value > 0 else "#AAB2BD" for value in values]
    positions = list(range(len(labels)))

    plt.figure(figsize=(10.5, max(5.5, len(labels) * 0.45)))
    bars = plt.barh(positions, values, color=colors, alpha=0.92)
    plt.yticks(positions, labels)
    plt.axvline(0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.6)
    _style_axes(title, xlabel=xlabel)

    x_span = max([abs(v) for v in values] + [1.0])
    for bar, value in zip(bars, values):
        xpos = value + (0.02 * x_span if value >= 0 else -0.02 * x_span)
        ha = "left" if value >= 0 else "right"
        plt.text(xpos, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", ha=ha, fontsize=10, color="#333333")

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def _grouped_bar(categories: list[str], series_map: dict[str, list[float]], title: str, ylabel: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not categories or not series_map:
        return

    x_positions = list(range(len(categories)))
    width = 0.8 / max(1, len(series_map))

    plt.figure(figsize=(max(10, len(categories) * 0.8), 6))
    for index, (label, values) in enumerate(series_map.items()):
        offsets = [x + (index - (len(series_map) - 1) / 2) * width for x in x_positions]
        plt.bar(offsets, values, width=width, label=label, color=_series_color(label))

    _style_axes(title, ylabel=ylabel)
    plt.xticks(x_positions, categories, rotation=20)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def _grouped_bar_with_labels(
    categories: list[str],
    series_map: dict[str, list[float]],
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    show_zero_line: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    if not categories or not series_map:
        return

    x_positions = list(range(len(categories)))
    width = 0.8 / max(1, len(series_map))

    plt.figure(figsize=(max(10.5, len(categories) * 0.82), 6.2))
    for index, (label, values) in enumerate(series_map.items()):
        offsets = [x + (index - (len(series_map) - 1) / 2) * width for x in x_positions]
        bars = plt.bar(offsets, values, width=width, label=label, color=_series_color(label), alpha=0.92)
        for bar, value in zip(bars, values):
            label_y = value + 0.6 if value >= 0 else value - 0.6
            va = "bottom" if value >= 0 else "top"
            plt.text(bar.get_x() + bar.get_width() / 2, label_y, f"{value:.1f}%", ha="center", va=va, fontsize=8.5, color="#333333")

    if show_zero_line:
        plt.axhline(0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.6)
    _style_axes(title, ylabel=ylabel)
    plt.xticks(x_positions, categories)
    plt.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def _boxplot(series_map: dict[str, list[float]], title: str, ylabel: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [label for label, values in series_map.items() if values]
    values = [values for _, values in series_map.items() if values]
    if not values:
        return

    plt.figure(figsize=(10, 6))
    plt.boxplot(values, tick_labels=labels)
    _style_axes(title, ylabel=ylabel)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def _avg_by_model(rows: list[dict], column: str) -> tuple[list[str], list[float]]:
    labels = []
    values = []
    for model_name in MODEL_ORDER:
        vals = [row[column] for row in rows if row.get("model") == model_name and row.get(column) is not None]
        if not vals:
            continue
        labels.append(MODEL_LABELS[model_name])
        values.append(statistics.fmean(vals))
    return labels, values


def _plot_global_model_bars(rows: list[dict], output_dir: Path) -> None:
    metrics = [
        ("best_distance", "average_best_distance_by_model.png", "Average Best Distance by Model", "Average Best Distance"),
        ("runtime_seconds", "average_runtime_by_model.png", "Average Runtime by Model", "Average Runtime (s)"),
        ("avg_fill_rate", "average_fill_rate_by_model.png", "Average Fill Rate by Model", "Average Fill Rate"),
        ("route_count", "route_count_by_model.png", "Average Route Count by Model", "Average Route Count"),
        ("tiny_route_count", "tiny_route_count_by_model.png", "Tiny Route Count by Model", "Average Tiny Route Count"),
        ("min_fill_rate", "minimum_fill_rate_by_model.png", "Minimum Fill Rate by Model", "Minimum Fill Rate"),
        ("route_fill_std", "route_fill_std_by_model.png", "Route Fill Std Dev by Model", "Route Fill Std Dev"),
        ("route_balance_penalty", "route_balance_penalty_by_model.png", "Route Balance Penalty by Model", "Route Balance Penalty"),
        ("avg_boxes_per_route", "average_boxes_per_route_by_model.png", "Average Boxes per Route by Model", "Average Boxes per Route"),
        ("avg_customers_per_route", "customers_per_route_by_model.png", "Customers per Route by Model", "Average Customers per Route"),
    ]
    for column, filename, title, ylabel in metrics:
        labels, values = _avg_by_model(rows, column)
        _bar_plot(labels, values, title, ylabel, output_dir / filename)


def _plot_feasible_infeasible(rows: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = []
    feasible = []
    infeasible = []
    for model_name in MODEL_ORDER:
        model_rows = [row for row in rows if row.get("model") == model_name]
        feasible_values = [row.get("feasible_routes") for row in model_rows if row.get("feasible_routes") is not None]
        infeasible_values = [row.get("infeasible_routes") for row in model_rows if row.get("infeasible_routes") is not None]
        if not feasible_values and not infeasible_values:
            continue
        labels.append(MODEL_LABELS[model_name])
        feasible.append(statistics.fmean(feasible_values) if feasible_values else 0.0)
        infeasible.append(statistics.fmean(infeasible_values) if infeasible_values else 0.0)

    if not labels:
        return

    plt.figure(figsize=(10, 6))
    plt.bar(labels, feasible, label="Feasible Routes")
    plt.bar(labels, infeasible, bottom=feasible, label="Infeasible Routes")
    plt.title("Feasible vs Infeasible Routes by Model")
    plt.ylabel("Average Route Count")
    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "feasible_vs_infeasible_routes.png")
    plt.close()


def _plot_improvement_over_baselines(summary_rows: list[dict], output_dir: Path) -> None:
    categories, series_map = _aggregate_series_vs_baselines_by_size(
        summary_rows,
        proposed_column="avg_distance",
        baseline_models=["baseline_a", "baseline_b", "baseline_c"],
        higher_is_better=False,
    )
    _grouped_bar_with_labels(
        categories=categories,
        series_map=series_map,
        title="Distance Improvement Percentage Over Baselines by Dataset Size",
        ylabel="Improvement (%)",
        output_path=output_dir / "distance_improvement_percentage_over_baselines.png",
        show_zero_line=True,
    )


def _plot_vs_baseline_c(summary_rows: list[dict], output_dir: Path) -> None:
    labels, route_reduction = _aggregate_route_reduction_by_size(summary_rows)
    _horizontal_bar_with_labels(
        labels=labels,
        values=route_reduction,
        title="Route Count Reduction Relative to Baseline C by Dataset Size",
        xlabel="Reduction (%)",
        output_path=output_dir / "route_count_reduction_vs_baseline_c.png",
    )

    categories, fill_series = _aggregate_series_vs_baselines_by_size(
        summary_rows,
        proposed_column="avg_fill_rate",
        baseline_models=["baseline_c"],
        higher_is_better=True,
    )
    fill_values = fill_series.get("Proposed vs Baseline C", [])
    _horizontal_bar_with_labels(
        labels=categories,
        values=fill_values,
        title="Average Fill Improvement Relative to Baseline C by Dataset Size",
        xlabel="Improvement (%)",
        output_path=output_dir / "average_fill_improvement_vs_baseline_c.png",
    )


def _plot_vs_baseline_c_by_dataset_size(rows: list[dict], output_dir: Path) -> None:
    improvement_specs = [
        (
            "best_distance",
            "distance_improvement_vs_baseline_c_by_dataset_size.png",
            "Distance Improvement vs Baseline C by Dataset Size",
            "Distance Improvement (%)",
            False,
            "Positive values indicate shorter routes than Baseline C",
        ),
        (
            "min_fill_rate",
            "minimum_fill_improvement_vs_baseline_c_by_dataset_size.png",
            "Minimum Fill Improvement vs Baseline C by Dataset Size",
            "Minimum Fill Improvement (%)",
            True,
            "Positive values indicate higher minimum route utilization",
        ),
        (
            "route_fill_std",
            "route_fill_std_improvement_vs_baseline_c_by_dataset_size.png",
            "Route Fill Std Improvement vs Baseline C by Dataset Size",
            "Route Fill Std Improvement (%)",
            False,
            "Positive values indicate lower fill variability than Baseline C",
        ),
    ]
    for column, filename, title, ylabel, higher_is_better, subtitle in improvement_specs:
        series_map = _aggregate_improvement_vs_baseline_c_by_size(rows, column, higher_is_better)
        _line_plot_with_band(
            series_map,
            title,
            "Dataset Size",
            ylabel,
            output_dir / filename,
            show_zero_line=True,
            subtitle=subtitle,
        )


def _plot_overflow_vs_baseline_c(summary_rows: list[dict], output_dir: Path) -> None:
    categories, series_map = _aggregate_pair_by_size(
        summary_rows,
        "avg_overflow_route_count",
        ("baseline_c", "proposed_model"),
    )
    _grouped_bar_with_labels(
        categories=categories,
        series_map={
            "Baseline C": series_map.get("Baseline C", []),
            "Proposed Model": series_map.get("Proposed Model", []),
        },
        title="Average Overflow Route Count: Baseline C vs Proposed by Dataset Size",
        ylabel="Average Overflow Route Count",
        output_path=output_dir / "overflow_route_count_vs_baseline_c.png",
    )


def _plot_runtime_vs_distance(rows: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    aggregated_points = []
    for model_name in ("baseline_c", "proposed_model"):
        grouped: dict[int, list[dict]] = {}
        for row in rows:
            if (
                row.get("model") == model_name
                and row.get("runtime_seconds") is not None
                and row.get("best_distance") is not None
                and row.get("num_customers") is not None
            ):
                grouped.setdefault(int(row["num_customers"]), []).append(row)
        for size, size_rows in grouped.items():
            aggregated_points.append(
                {
                    "model": model_name,
                    "size": size,
                    "runtime_seconds": statistics.fmean(float(row["runtime_seconds"]) for row in size_rows),
                    "best_distance": statistics.fmean(float(row["best_distance"]) for row in size_rows),
                }
            )

    band_specs = [
        ("small_medium", "Small/Medium Datasets (50-500)", lambda n: n is not None and n <= 500),
        ("large", "Large Datasets (550-750)", lambda n: n is not None and n >= 550),
    ]
    marker_map = {"small": "o", "medium": "s", "large": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.3), sharey=True)
    plotted_any = False

    for ax, (_, panel_title, band_filter) in zip(axes, band_specs):
        for model_name in ("baseline_c", "proposed_model"):
            model_rows = [
                row for row in aggregated_points
                if row.get("model") == model_name
                and row.get("runtime_seconds") is not None
                and row.get("best_distance") is not None
                and band_filter(row.get("size"))
            ]
            if not model_rows:
                continue
            plotted_any = True
            color = _series_color(MODEL_LABELS[model_name])
            grouped_by_band = {
                "small": [row for row in model_rows if (row.get("size") or 0) <= 250],
                "medium": [row for row in model_rows if 300 <= (row.get("size") or 0) <= 500],
                "large": [row for row in model_rows if (row.get("size") or 0) >= 550],
            }
            for band_name, band_rows in grouped_by_band.items():
                if not band_rows:
                    continue
                ax.scatter(
                    [float(row["runtime_seconds"]) for row in band_rows],
                    [float(row["best_distance"]) for row in band_rows],
                    color=color,
                    marker=marker_map[band_name],
                    alpha=0.55,
                    s=32,
                    edgecolors="white",
                    linewidths=0.6,
                    label=f"{MODEL_LABELS[model_name]} ({band_name})",
                )

            if len(model_rows) >= 2:
                np = _try_import_numpy()
                if np is not None:
                    xs = np.asarray([float(row["runtime_seconds"]) for row in model_rows], dtype=float)
                    ys = np.asarray([float(row["best_distance"]) for row in model_rows], dtype=float)
                    coeffs = np.polyfit(xs, ys, deg=1)
                    trend_x = np.linspace(xs.min(), xs.max(), 100)
                    trend_y = coeffs[0] * trend_x + coeffs[1]
                    ax.plot(trend_x, trend_y, color=color, linewidth=1.6, alpha=0.45)

        for row in aggregated_points:
            if row.get("size") in {500, 750} and band_filter(row.get("size")) and row.get("model") in {"baseline_c", "proposed_model"}:
                ax.annotate(
                    f"XML{row['size']}",
                    (float(row["runtime_seconds"]), float(row["best_distance"])),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                    alpha=0.75,
                )

        ax.set_title(panel_title, fontsize=13, pad=10)
        ax.set_xlabel("Runtime (s)", fontsize=AXIS_FONT_SIZE)
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.15)
        ax.tick_params(axis="both", labelsize=10)

    axes[0].set_ylabel("Best Distance", fontsize=AXIS_FONT_SIZE)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Runtime vs Distance: Baseline C vs Proposed Model", fontsize=TITLE_FONT_SIZE, y=1.05)
    fig.text(0.5, 0.99, "Marker shape indicates dataset-size band; faint lines show linear runtime-distance trend", ha="center", fontsize=SUBTITLE_FONT_SIZE, color="#555555")
    if not plotted_any:
        plt.close(fig)
        return
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_vs_distance_baseline_c_vs_proposed.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_tradeoff_summary(summary_rows: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    aggregated = aggregate_summary_by_size(
        summary_rows,
        model_names=["baseline_c", "proposed_model"],
        sizes=REPRESENTATIVE_DATASET_SIZES,
    )
    by_key = {(row["num_customers"], row["model"]): row for row in aggregated}

    sizes = [size for size in REPRESENTATIVE_DATASET_SIZES if (size, "baseline_c") in by_key and (size, "proposed_model") in by_key]
    if not sizes:
        return

    min_fill_improvement = []
    fill_std_improvement = []
    route_count_reduction = []
    distance_gap = []
    runtime_overhead = []

    for size in sizes:
        baseline = by_key[(size, "baseline_c")]
        proposed = by_key[(size, "proposed_model")]

        min_fill_improvement.append(pct_change(baseline.get("min_route_fill"), proposed.get("min_route_fill"), higher_is_better=True) or 0.0)
        fill_std_improvement.append(pct_change(baseline.get("avg_route_fill_std"), proposed.get("avg_route_fill_std"), higher_is_better=False) or 0.0)
        route_count_reduction.append(pct_change(baseline.get("avg_route_count"), proposed.get("avg_route_count"), higher_is_better=False) or 0.0)
        distance_gap.append(((proposed.get("avg_distance") or 0.0) - (baseline.get("avg_distance") or 0.0)) / (baseline.get("avg_distance") or 1.0) * 100.0)
        runtime_ratio = ratio(proposed.get("avg_runtime"), baseline.get("avg_runtime"))
        runtime_overhead.append(((runtime_ratio or 1.0) - 1.0) * 100.0)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))
    y_pos = list(range(len(sizes)))
    height = 0.22

    structure_series = [
        ("Min Fill Improvement", min_fill_improvement, "#3BA55D"),
        ("Fill Std Improvement", fill_std_improvement, "#4C78A8"),
        ("Route Count Reduction", route_count_reduction, "#F58518"),
    ]
    for idx, (label, values, color) in enumerate(structure_series):
        offsets = [y + (idx - 1) * height for y in y_pos]
        axes[0].barh(offsets, values, height=height, label=label, color=color, alpha=0.9)
    axes[0].axvline(0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.6)
    axes[0].set_yticks(y_pos, [f"XML{size}" for size in sizes])
    axes[0].set_title("Structural Gains vs Baseline C", fontsize=TITLE_FONT_SIZE, pad=10)
    axes[0].set_xlabel("Improvement (%)", fontsize=AXIS_FONT_SIZE)
    axes[0].grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.15)
    axes[0].legend(frameon=False, fontsize=LEGEND_FONT_SIZE)

    cost_series = [
        ("Distance Gap", distance_gap, "#E45756"),
        ("Runtime Overhead", runtime_overhead, "#B279A2"),
    ]
    for idx, (label, values, color) in enumerate(cost_series):
        offsets = [y + (idx - 0.5) * height for y in y_pos]
        axes[1].barh(offsets, values, height=height, label=label, color=color, alpha=0.9)
        for y, value in zip(offsets, values):
            axes[1].text(value + (0.4 if value >= 0 else -0.4), y, f"{value:.1f}%", va="center", ha="left" if value >= 0 else "right", fontsize=8.5)
    axes[1].axvline(0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.6)
    axes[1].set_yticks(y_pos, [f"XML{size}" for size in sizes])
    axes[1].set_title("Cost of Structural Gains", fontsize=TITLE_FONT_SIZE, pad=10)
    axes[1].set_xlabel("Percentage vs Baseline C", fontsize=AXIS_FONT_SIZE)
    axes[1].grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.15)
    axes[1].legend(frameon=False, fontsize=LEGEND_FONT_SIZE)

    fig.suptitle("Baseline C vs Proposed Model Tradeoff Overview", fontsize=TITLE_FONT_SIZE + 1, y=1.02)
    fig.text(0.5, 0.97, "Positive structural values are better; positive distance/runtime values indicate extra cost", ha="center", fontsize=SUBTITLE_FONT_SIZE, color="#555555")
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_c_vs_proposed_tradeoff_overview.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_ablation_graphs(ablation_rows: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if not ablation_rows:
        return

    summary_rows = build_summary_table(ablation_rows)
    aggregated = aggregate_summary_by_size(
        summary_rows,
        model_names=ABLATION_MODEL_ORDER,
        sizes=REPRESENTATIVE_DATASET_SIZES,
    )
    if not aggregated:
        return

    size_marker = {100: "o", 250: "s", 500: "^", 750: "D"}

    def scatter(metric_x: str, metric_y: str, filename: str, title: str, xlabel: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(10.6, 6.4))
        plotted = False
        for row in aggregated:
            label = MODEL_LABELS.get(row["model"], row["model"])
            x_value = row.get(metric_x)
            y_value = row.get(metric_y)
            if x_value is None or y_value is None:
                continue
            plotted = True
            size = int(row["num_customers"])
            ax.scatter(
                float(x_value),
                float(y_value),
                color=_series_color(label),
                marker=size_marker.get(size, "o"),
                s=56,
                alpha=0.82,
                edgecolors="white",
                linewidths=0.8,
            )
            ax.annotate(f"{label}\nXML{size}", (float(x_value), float(y_value)), textcoords="offset points", xytext=(4, 4), fontsize=8)
        if not plotted:
            plt.close(fig)
            return
        ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=12)
        ax.set_xlabel(xlabel, fontsize=AXIS_FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_FONT_SIZE)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.15)
        ax.tick_params(axis="both", labelsize=10)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

    scatter(
        "avg_runtime",
        "min_route_fill",
        "ablation_runtime_vs_min_fill.png",
        "Ablation Runtime vs Minimum Route Fill",
        "Average Runtime (s)",
        "Minimum Route Fill",
    )
    scatter(
        "avg_distance",
        "min_route_fill",
        "ablation_distance_vs_min_fill.png",
        "Ablation Distance vs Minimum Route Fill",
        "Average Distance",
        "Minimum Route Fill",
    )

    by_key = {(row["num_customers"], row["model"]): row for row in aggregated}
    categories = [MODEL_LABELS[model_name] for model_name in ABLATION_MODEL_ORDER if model_name != "proposed_full"]
    if categories:
        distance_delta = []
        min_fill_delta = []
        runtime_delta = []
        for model_name in ABLATION_MODEL_ORDER:
            if model_name == "proposed_full":
                continue
            dist_changes = []
            fill_changes = []
            runtime_changes = []
            for size in REPRESENTATIVE_DATASET_SIZES:
                full_row = by_key.get((size, "proposed_full"))
                variant_row = by_key.get((size, model_name))
                if not full_row or not variant_row:
                    continue
                dist = pct_change(full_row.get("avg_distance"), variant_row.get("avg_distance"), higher_is_better=False)
                fill = pct_change(full_row.get("min_route_fill"), variant_row.get("min_route_fill"), higher_is_better=True)
                runtime = pct_change(full_row.get("avg_runtime"), variant_row.get("avg_runtime"), higher_is_better=False)
                if dist is not None:
                    dist_changes.append(dist)
                if fill is not None:
                    fill_changes.append(fill)
                if runtime is not None:
                    runtime_changes.append(-runtime)
            distance_delta.append(statistics.fmean(dist_changes) if dist_changes else 0.0)
            min_fill_delta.append(statistics.fmean(fill_changes) if fill_changes else 0.0)
            runtime_delta.append(statistics.fmean(runtime_changes) if runtime_changes else 0.0)

        _grouped_bar_with_labels(
            categories=categories,
            series_map={
                "Distance vs Full": distance_delta,
                "Min Fill vs Full": min_fill_delta,
                "Runtime vs Full": runtime_delta,
            },
            title="Average Metric Change from Full Proposed Model",
            ylabel="Change (%)",
            output_path=output_dir / "ablation_metric_change_vs_full.png",
            show_zero_line=True,
        )


def _plot_route_composition(rows: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    for size in ROUTE_COMPOSITION_SIZES:
        dataset_name = _representative_dataset_by_size(rows, size)
        if not dataset_name:
            continue
        seed = _representative_seed(rows, dataset_name, ("baseline_c", "proposed_model"))
        if seed is None:
            continue

        baseline_result = _load_result_json("baseline_c", dataset_name, seed)
        proposed_result = _load_result_json("proposed_model", dataset_name, seed)
        if not baseline_result or not proposed_result:
            continue

        baseline_metrics = _route_metric_lists(baseline_result)
        proposed_metrics = _route_metric_lists(proposed_result)

        fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.6))
        plot_specs = [
            ("boxes_per_route", "Boxes per Route", axes[0][0]),
            ("customers_per_route", "Customers per Route", axes[0][1]),
            ("fill_rate", "Route Fill Rate", axes[1][0]),
        ]
        for metric_key, title, ax in plot_specs:
            baseline_values = baseline_metrics[metric_key]
            proposed_values = proposed_metrics[metric_key]
            if not baseline_values or not proposed_values:
                continue
            bins = min(8, max(len(baseline_values), len(proposed_values)))
            ax.hist(baseline_values, bins=bins, alpha=0.55, color=_series_color("Baseline C"), label="Baseline C")
            ax.hist(proposed_values, bins=bins, alpha=0.55, color=_series_color("Proposed Model"), label="Proposed Model")
            ax.set_title(title, fontsize=13)
            ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.15)
            ax.tick_params(axis="both", labelsize=10)

        baseline_fill = sorted(baseline_metrics["fill_rate"], reverse=True)
        proposed_fill = sorted(proposed_metrics["fill_rate"], reverse=True)
        axes[1][1].plot(range(1, len(baseline_fill) + 1), baseline_fill, marker="o", color=_series_color("Baseline C"), label="Baseline C")
        axes[1][1].plot(range(1, len(proposed_fill) + 1), proposed_fill, marker="o", color=_series_color("Proposed Model"), label="Proposed Model")
        axes[1][1].set_title("Sorted Route Fill Profile", fontsize=13)
        axes[1][1].set_xlabel("Route Rank", fontsize=AXIS_FONT_SIZE)
        axes[1][1].set_ylabel("Fill Rate", fontsize=AXIS_FONT_SIZE)
        axes[1][1].grid(True, linestyle="--", linewidth=0.55, alpha=0.15)
        axes[1][1].tick_params(axis="both", labelsize=10)

        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=LEGEND_FONT_SIZE)
        fig.suptitle(
            f"Route Composition Comparison: XML{size} ({dataset_name}, seed {seed})",
            fontsize=TITLE_FONT_SIZE,
            y=0.98,
        )
        fig.text(
            0.5,
            0.95,
            "Histograms show route composition; the fill-profile panel shows how evenly route utilization is distributed.",
            ha="center",
            fontsize=SUBTITLE_FONT_SIZE,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(output_dir / f"route_composition_XML{size}.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)


def _plot_paper_tradeoff_distance_vs_fill_std(summary_rows: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    aggregated = aggregate_summary_by_size(
        summary_rows,
        model_names=["baseline_b", "baseline_c", "proposed_model"],
    )
    if not aggregated:
        return

    plt.figure(figsize=(10.8, 6.8))
    plotted = False
    label_offsets = {
        ("proposed_model", 50): (8, 10),
        ("proposed_model", 250): (10, 8),
        ("proposed_model", 500): (10, 8),
        ("proposed_model", 750): (10, 8),
    }
    for model_name in ["baseline_b", "baseline_c", "proposed_model"]:
        label = MODEL_LABELS.get(model_name, model_name)
        model_rows = [
            row for row in aggregated
            if row.get("model") == model_name
            and row.get("avg_distance") is not None
            and row.get("avg_route_fill_std") is not None
        ]
        if not model_rows:
            continue
        model_rows = sorted(model_rows, key=lambda row: int(row["num_customers"]))
        plotted = True
        xs = [float(row["avg_distance"]) for row in model_rows]
        ys = [float(row["avg_route_fill_std"]) for row in model_rows]
        sizes = [int(row["num_customers"]) for row in model_rows]

        plt.scatter(
            xs,
            ys,
            label=label,
            color=_series_color(label),
            marker=MODEL_MARKERS.get(label, "o"),
            s=92 if model_name == "proposed_model" else 56,
            alpha=0.92 if model_name == "proposed_model" else 0.7,
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
        )
        for x_value, y_value, size in zip(xs, ys, sizes):
            if model_name == "proposed_model" and size in {50, 250, 500, 750}:
                dx, dy = label_offsets.get((model_name, size), (8, 8))
                plt.annotate(
                    f"XML{size}",
                    (x_value, y_value),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    fontsize=8.5,
                    color="#333333",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85),
                )

    if not plotted:
        plt.close()
        return

    _style_axes(
        "",
        xlabel="Total Distance",
        ylabel="Route Fill Std (Lower is Better)",
        subtitle=None,
    )
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.10)

    x_values = [float(row["avg_distance"]) for row in aggregated if row.get("avg_distance") is not None]
    y_values = [float(row["avg_route_fill_std"]) for row in aggregated if row.get("avg_route_fill_std") is not None]
    plt.ylim(0.005, 0.028)
    plt.legend(frameon=False, fontsize=LEGEND_FONT_SIZE, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(output_dir / "tradeoff_distance_vs_fill_std.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def _plot_paired_seed_views(rows: list[dict], output_dir: Path) -> None:
    target_prefixes = ("XML100_", "XML250_", "XML500_", "XML750_")
    datasets = sorted({
        row["dataset"]
        for row in rows
        if row.get("dataset") and any(str(row["dataset"]).startswith(prefix) for prefix in target_prefixes)
    })
    for dataset_name in datasets:
        dataset_rows = [row for row in rows if row.get("dataset") == dataset_name and row.get("model") in {"baseline_c", "proposed_model"}]
        paired_dir = output_dir / dataset_name
        paired_dir.mkdir(parents=True, exist_ok=True)

        for column, filename, ylabel in [
            ("best_distance", "paired_best_distance_vs_seed.png", "Best Distance"),
            ("min_fill_rate", "paired_min_fill_vs_seed.png", "Minimum Fill Rate"),
            ("route_fill_std", "paired_route_fill_std_vs_seed.png", "Route Fill Std Dev"),
        ]:
            series_map = {}
            for model_name in ("baseline_c", "proposed_model"):
                model_rows = sorted(
                    [row for row in dataset_rows if row.get("model") == model_name and row.get(column) is not None],
                    key=lambda row: row.get("seed") or 0,
                )
                if not model_rows:
                    continue
                series_map[MODEL_LABELS[model_name]] = (
                    [row["seed"] for row in model_rows],
                    [row[column] for row in model_rows],
                )
            _line_plot(
                series_map,
                f"{ylabel} vs Seed: {dataset_name}",
                "Seed",
                ylabel,
                paired_dir / filename,
            )


def _plot_by_dataset_size(rows: list[dict], output_dir: Path) -> None:
    metrics = [
        ("best_distance", "distance_by_dataset_size.png", "Average Best Distance by Dataset Size", "Average Best Distance"),
        ("runtime_seconds", "runtime_by_dataset_size.png", "Average Runtime by Dataset Size", "Average Runtime (s)"),
        ("min_fill_rate", "minimum_fill_by_dataset_size.png", "Minimum Fill by Dataset Size", "Minimum Fill"),
        ("route_count", "route_count_by_dataset_size.png", "Average Route Count by Dataset Size", "Average Route Count"),
        ("route_fill_std", "route_fill_std_by_dataset_size.png", "Average Route Fill Std Dev by Dataset Size", "Average Route Fill Std Dev"),
    ]

    for column, filename, title, ylabel in metrics:
        series_map = _aggregate_rows_by_size(rows, column)
        _line_plot_with_band(series_map, title, "Number of Customers", ylabel, output_dir / filename)


def _plot_by_max_boxes(rows: list[dict], output_dir: Path) -> None:
    valid_rows = [row for row in rows if row.get("max_boxes_per_route") is not None]
    if not valid_rows:
        return

    metrics = [
        ("best_distance", "distance_by_max_boxes_per_route.png", "Best Distance vs max_boxes_per_route", "Best Distance"),
        ("avg_fill_rate", "fill_rate_by_max_boxes_per_route.png", "Fill Rate vs max_boxes_per_route", "Average Fill Rate"),
        ("route_count", "route_count_by_max_boxes_per_route.png", "Route Count vs max_boxes_per_route", "Average Route Count"),
    ]
    for column, filename, title, ylabel in metrics:
        series_map = {}
        for model_name in MODEL_ORDER:
            model_rows = [row for row in valid_rows if row.get("model") == model_name and row.get(column) is not None]
            grouped = {}
            for row in model_rows:
                grouped.setdefault(int(row["max_boxes_per_route"]), []).append(float(row[column]))
            if not grouped:
                continue
            xs = sorted(grouped)
            ys = [statistics.fmean(grouped[x]) for x in xs]
            series_map[MODEL_LABELS[model_name]] = (xs, ys)
        _line_plot(series_map, title, "max_boxes_per_route", ylabel, output_dir / filename)


def _plot_by_family(summary_rows: list[dict], output_dir: Path) -> None:
    families = sorted({row.get("family_group") for row in summary_rows if row.get("family_group")})
    if not families:
        return

    series_map = {}
    for model_name in MODEL_ORDER:
        values = []
        for family in families:
            family_rows = [row for row in summary_rows if row.get("family_group") == family and row.get("model") == model_name and row.get("avg_distance") is not None]
            values.append(statistics.fmean([row["avg_distance"] for row in family_rows]) if family_rows else 0.0)
        if any(value != 0.0 for value in values):
            series_map[MODEL_LABELS[model_name]] = values

    _grouped_bar(
        categories=families,
        series_map=series_map,
        title="Average Best Distance by Family Group",
        ylabel="Average Best Distance",
        output_path=output_dir / "distance_by_family_group.png",
    )

    fill_series = {}
    for model_name in MODEL_ORDER:
        values = []
        for family in families:
            family_rows = [row for row in summary_rows if row.get("family_group") == family and row.get("model") == model_name and row.get("avg_fill_rate") is not None]
            values.append(statistics.fmean([row["avg_fill_rate"] for row in family_rows]) if family_rows else 0.0)
        if any(value != 0.0 for value in values):
            fill_series[MODEL_LABELS[model_name]] = values
    _grouped_bar(
        categories=families,
        series_map=fill_series,
        title="Average Fill Rate by Family Group",
        ylabel="Average Fill Rate",
        output_path=output_dir / "fill_rate_by_family_group.png",
    )


def _plot_dataset_specific(rows: list[dict], summary_rows: list[dict], output_dir: Path) -> None:
    datasets = sorted({row["dataset"] for row in rows if row.get("dataset")})
    for dataset_name in datasets:
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_rows = [row for row in rows if row.get("dataset") == dataset_name]
        dataset_summary = [row for row in summary_rows if row.get("dataset") == dataset_name]

        for column, filename, title, ylabel in [
            ("best_distance", "average_best_distance_by_model.png", f"Average Best Distance: {dataset_name}", "Average Best Distance"),
            ("runtime_seconds", "average_runtime_by_model.png", f"Average Runtime: {dataset_name}", "Average Runtime (s)"),
            ("avg_fill_rate", "average_fill_rate_by_model.png", f"Average Fill Rate: {dataset_name}", "Average Fill Rate"),
            ("route_count", "route_count_by_model.png", f"Average Route Count: {dataset_name}", "Average Route Count"),
            ("min_fill_rate", "minimum_fill_rate_by_model.png", f"Minimum Fill Rate: {dataset_name}", "Minimum Fill Rate"),
            ("tiny_route_count", "tiny_route_count_by_model.png", f"Tiny Route Count: {dataset_name}", "Average Tiny Route Count"),
            ("route_fill_std", "route_fill_std_by_model.png", f"Route Fill Std Dev: {dataset_name}", "Route Fill Std Dev"),
            ("avg_boxes_per_route", "average_boxes_per_route_by_model.png", f"Average Boxes per Route: {dataset_name}", "Average Boxes per Route"),
            ("avg_customers_per_route", "customers_per_route_by_model.png", f"Customers per Route: {dataset_name}", "Average Customers per Route"),
        ]:
            labels, values = _avg_by_model(dataset_rows, column)
            _bar_plot(labels, values, title, ylabel, dataset_dir / filename)

        series_map = {}
        for model_name in MODEL_ORDER:
            model_rows = sorted(
                [row for row in dataset_rows if row.get("model") == model_name and row.get("best_distance") is not None],
                key=lambda row: row.get("seed") or 0,
            )
            if not model_rows:
                continue
            series_map[MODEL_LABELS[model_name]] = (
                [row["seed"] for row in model_rows],
                [row["best_distance"] for row in model_rows],
            )
        _line_plot(series_map, f"Best Distance vs Seed: {dataset_name}", "Seed", "Best Distance", dataset_dir / "distance_vs_seed.png")

        series_map = {}
        for model_name in MODEL_ORDER:
            histories = []
            model_history_dir = Path(PROJECT_ROOT / "comparison_models" / "outputs" / model_name / dataset_name)
            for history_file in sorted(model_history_dir.glob("history_seed_*.csv")):
                lines = history_file.read_text(encoding="utf-8").splitlines()[1:]
                history = [float(line.split(",")[1]) for line in lines if "," in line]
                if history:
                    histories.append(history)
            if not histories:
                continue
            max_len = max(len(history) for history in histories)
            averaged = []
            for index in range(max_len):
                values = [history[index] for history in histories if index < len(history)]
                averaged.append(statistics.fmean(values))
            series_map[MODEL_LABELS[model_name]] = (list(range(1, len(averaged) + 1)), averaged)
        _line_plot(series_map, f"Convergence Plot: {dataset_name}", "Generation", "Average Best Score", dataset_dir / "convergence_plot.png")

        boxplot_data = {}
        for model_name in MODEL_ORDER:
            values = [row["best_score"] for row in dataset_rows if row.get("model") == model_name and row.get("best_score") is not None]
            if values:
                boxplot_data[MODEL_LABELS[model_name]] = values
        _boxplot(boxplot_data, f"Score Distribution: {dataset_name}", "Best Score", dataset_dir / "score_distribution_boxplot.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate dissertation comparison graphs.")
    parser.add_argument("--dataset", action="append", help="Optional dataset filter; can be passed multiple times.")
    parser.add_argument("--model", action="append", choices=MODEL_ORDER, help="Optional model filter; can be passed multiple times.")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required to generate comparison plots.") from exc

    all_rows = load_all_results()
    filtered_rows = _filter_rows(all_rows, args.dataset, args.model)
    summary_rows = build_summary_table(filtered_rows)
    ablation_rows = load_all_results(ABLATION_OUTPUTS_ROOT)
    ablation_filtered_rows = _filter_rows(ablation_rows, args.dataset, None)

    GRAPHS_ROOT.mkdir(parents=True, exist_ok=True)
    ablation_dir = GRAPHS_ROOT / "ablation"
    tradeoff_dir = GRAPHS_ROOT / "tradeoff"
    route_composition_dir = GRAPHS_ROOT / "route_composition"
    paper_dir = GRAPHS_ROOT / PAPER_GRAPHS_DIRNAME
    ablation_dir.mkdir(parents=True, exist_ok=True)
    tradeoff_dir.mkdir(parents=True, exist_ok=True)
    route_composition_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    _plot_global_model_bars(filtered_rows, GRAPHS_ROOT)
    _plot_feasible_infeasible(filtered_rows, GRAPHS_ROOT)
    _plot_improvement_over_baselines(summary_rows, GRAPHS_ROOT)
    _plot_vs_baseline_c(summary_rows, GRAPHS_ROOT)
    _plot_vs_baseline_c_by_dataset_size(filtered_rows, GRAPHS_ROOT)
    _plot_overflow_vs_baseline_c(summary_rows, GRAPHS_ROOT)
    _plot_runtime_vs_distance(filtered_rows, GRAPHS_ROOT)
    _plot_by_dataset_size(filtered_rows, GRAPHS_ROOT)
    _plot_by_max_boxes(filtered_rows, GRAPHS_ROOT)
    _plot_by_family(summary_rows, GRAPHS_ROOT)
    _plot_dataset_specific(filtered_rows, summary_rows, GRAPHS_ROOT / "datasets")
    _plot_paired_seed_views(filtered_rows, GRAPHS_ROOT / "paired_seed_plots")
    _plot_tradeoff_summary(summary_rows, tradeoff_dir)
    _plot_ablation_graphs(ablation_filtered_rows, ablation_dir)
    _plot_route_composition(filtered_rows, route_composition_dir)
    _plot_paper_tradeoff_distance_vs_fill_std(summary_rows, paper_dir)

    print(f"Saved comparison graphs to {GRAPHS_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
