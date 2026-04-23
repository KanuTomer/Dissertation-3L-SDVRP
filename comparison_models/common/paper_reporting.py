from __future__ import annotations

import statistics


REPRESENTATIVE_DATASET_SIZES = [100, 250, 500, 750]
ROUTE_COMPOSITION_SIZES = [250, 500, 750]

ABLATION_VARIANTS = [
    {
        "id": "proposed_full",
        "label": "Full Proposed",
        "flags": {},
    },
    {
        "id": "proposed_no_adaptive",
        "label": "No Adaptive Decoding",
        "flags": {"enable_adaptive_decoding": False},
    },
    {
        "id": "proposed_no_tiny_repair",
        "label": "No Tiny-Route Repair",
        "flags": {"enable_tiny_route_repair": False},
    },
    {
        "id": "proposed_no_relocation",
        "label": "No Relocation Repair",
        "flags": {"enable_customer_relocation_repair": False},
    },
    {
        "id": "proposed_no_route_balance_mutation",
        "label": "No Route-Balance Mutation",
        "flags": {"enable_route_balance_mutation": False},
    },
    {
        "id": "proposed_no_final_refinement",
        "label": "No Final-Best Refinement",
        "flags": {"enable_final_best_refinement": False},
    },
]

ABLATION_MODEL_ORDER = [variant["id"] for variant in ABLATION_VARIANTS]
ABLATION_LABELS = {variant["id"]: variant["label"] for variant in ABLATION_VARIANTS}


def avg(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def aggregate_summary_by_size(
    summary_rows: list[dict],
    *,
    model_names: list[str] | None = None,
    sizes: list[int] | None = None,
) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    allowed_models = set(model_names or [])
    allowed_sizes = set(sizes or [])

    for row in summary_rows:
        num_customers = row.get("num_customers")
        model_name = row.get("model")
        if num_customers is None or model_name is None:
            continue
        if allowed_models and model_name not in allowed_models:
            continue
        if allowed_sizes and int(num_customers) not in allowed_sizes:
            continue
        grouped.setdefault((int(num_customers), model_name), []).append(row)

    aggregated_rows: list[dict] = []
    for (size, model_name), rows in sorted(grouped.items()):
        aggregated_rows.append(
            {
                "num_customers": size,
                "model": model_name,
                "dataset_count": len(rows),
                "avg_score": avg([row["avg_score"] for row in rows if row.get("avg_score") is not None]),
                "avg_distance": avg([row["avg_distance"] for row in rows if row.get("avg_distance") is not None]),
                "avg_runtime": avg([row["avg_runtime"] for row in rows if row.get("avg_runtime") is not None]),
                "avg_fill_rate": avg([row["avg_fill_rate"] for row in rows if row.get("avg_fill_rate") is not None]),
                "avg_route_count": avg([row["avg_route_count"] for row in rows if row.get("avg_route_count") is not None]),
                "min_route_fill": avg([row["min_route_fill"] for row in rows if row.get("min_route_fill") is not None]),
                "avg_route_fill_std": avg([row["avg_route_fill_std"] for row in rows if row.get("avg_route_fill_std") is not None]),
                "avg_overflow_route_count": avg([row["avg_overflow_route_count"] for row in rows if row.get("avg_overflow_route_count") is not None]),
                "avg_merged_route_count": avg([row["avg_merged_route_count"] for row in rows if row.get("avg_merged_route_count") is not None]),
                "avg_boxes_per_route": avg([row["avg_boxes_per_route"] for row in rows if row.get("avg_boxes_per_route") is not None]),
                "avg_customers_per_route": avg([row["avg_customers_per_route"] for row in rows if row.get("avg_customers_per_route") is not None]),
            }
        )

    return aggregated_rows


def pct_change(baseline: float | None, comparison: float | None, *, higher_is_better: bool) -> float | None:
    if baseline in (None, 0) or comparison is None:
        return None
    raw = ((comparison - baseline) / baseline) * 100.0
    return raw if higher_is_better else -raw


def ratio(comparison: float | None, baseline: float | None) -> float | None:
    if baseline in (None, 0) or comparison is None:
        return None
    return comparison / baseline
