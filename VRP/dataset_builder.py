from __future__ import annotations

import json
from pathlib import Path

from box_generator import generate_boxes_for_customers
from vrp_parser import parse_vrp_file


DEFAULT_CONTAINER = {"L": 4000.0, "W": 1800.0, "H": 1800.0}


def build_dataset(
    vrp_path: str | Path,
    output_path: str | Path | None = None,
    min_boxes_per_customer: int = 1,
    max_boxes_per_customer: int = 4,
    seed: int = 42,
    container: dict | None = None,
) -> tuple[dict, Path | None]:
    parsed = parse_vrp_file(vrp_path)
    container_data = dict(container or DEFAULT_CONTAINER)
    depot_id = int(parsed["depot_id"])
    real_customers = [
        customer for customer in parsed["customers"]
        if int(customer["customer_id"]) != depot_id
    ]

    boxes, assignments = generate_boxes_for_customers(
        customers=real_customers,
        container=container_data,
        min_boxes_per_customer=min_boxes_per_customer,
        max_boxes_per_customer=max_boxes_per_customer,
        seed=seed,
    )

    customers_with_boxes: list[dict] = []
    for customer in parsed["customers"]:
        customer_id = int(customer["customer_id"])
        customers_with_boxes.append(
            {
                "id": customer_id,
                "x": float(customer["x"]),
                "y": float(customer["y"]),
                "demand": int(customer["demand"]),
                "customer_id": customer_id,
                "is_depot": bool(customer.get("is_depot")),
                "assigned_boxes": [str(box_id) for box_id in assignments.get(customer_id, [])],
            }
        )

    dataset = {
        "instance_name": parsed["instance_name"],
        "inst_name": parsed["inst_name"],
        "name": parsed["name"],
        "depot": [float(parsed["depot"][0]), float(parsed["depot"][1])],
        "depot_id": depot_id,
        "capacity": int(parsed["capacity"]),
        "container": {
            "L": float(container_data["L"]),
            "W": float(container_data["W"]),
            "H": float(container_data["H"]),
        },
        "customers": customers_with_boxes,
        "boxes": [
            {
                "box_id": str(box["box_id"]),
                "length": float(box["length"]),
                "width": float(box["width"]),
                "height": float(box["height"]),
            }
            for box in boxes
        ],
    }

    saved_path: Path | None = None
    if output_path is not None:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        saved_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    return dataset, saved_path
