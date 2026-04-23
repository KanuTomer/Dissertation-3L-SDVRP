from __future__ import annotations

import random


DEFAULT_LENGTH_RANGE = (50.0, 800.0)
DEFAULT_WIDTH_RANGE = (50.0, 500.0)
DEFAULT_HEIGHT_RANGE = (20.0, 300.0)


def _bounded_uniform(rng: random.Random, lower: float, upper: float, limit: float) -> float:
    bounded_upper = min(upper, limit)
    bounded_lower = min(lower, bounded_upper)
    return round(rng.uniform(bounded_lower, bounded_upper), 2)


def generate_boxes_for_customers(
    customers: list[dict],
    container: dict,
    min_boxes_per_customer: int = 1,
    max_boxes_per_customer: int = 4,
    seed: int = 42,
) -> tuple[list[dict], dict[int, list[str]]]:
    if min_boxes_per_customer < 1:
        raise ValueError("min_boxes_per_customer must be at least 1")
    if max_boxes_per_customer < min_boxes_per_customer:
        raise ValueError("max_boxes_per_customer must be greater than or equal to min_boxes_per_customer")

    container_length = float(container["L"])
    container_width = float(container["W"])
    container_height = float(container["H"])
    rng = random.Random(seed)

    boxes: list[dict] = []
    assignments: dict[int, list[str]] = {}
    next_box_id = 1

    for customer in customers:
        customer_id = int(customer["customer_id"])
        box_count = rng.randint(min_boxes_per_customer, max_boxes_per_customer)
        customer_box_ids: list[str] = []

        for _ in range(box_count):
            box_id = f"box_{next_box_id}"
            next_box_id += 1

            box = {
                "box_id": box_id,
                "length": _bounded_uniform(rng, *DEFAULT_LENGTH_RANGE, limit=container_length),
                "width": _bounded_uniform(rng, *DEFAULT_WIDTH_RANGE, limit=container_width),
                "height": _bounded_uniform(rng, *DEFAULT_HEIGHT_RANGE, limit=container_height),
            }
            boxes.append(box)
            customer_box_ids.append(box_id)

        assignments[customer_id] = customer_box_ids

    return boxes, assignments
