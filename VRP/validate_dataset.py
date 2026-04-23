from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comparison_models.common.loaders.route_evaluator import evaluate_route, load_merged


def validate_dataset(dataset_path: str | Path) -> tuple[bool, list[str], dict]:
    dataset_path = Path(dataset_path)
    data = json.loads(dataset_path.read_text(encoding="utf-8-sig"))

    errors: list[str] = []
    customers = data.get("customers", [])
    boxes = data.get("boxes", [])
    container = data.get("container", {})

    if not isinstance(customers, list):
        errors.append("customers must be a list")
        customers = []
    if not isinstance(boxes, list):
        errors.append("boxes must be a list")
        boxes = []

    customer_ids = [customer.get("customer_id") for customer in customers]
    if len(customer_ids) != len(set(customer_ids)):
        errors.append("duplicate customer_id values found")

    depot_customer = customers[0] if customers else None
    depot_customer_id = depot_customer.get("customer_id") if depot_customer else None
    real_customers = customers[1:] if len(customers) > 1 else []

    box_ids = [box.get("box_id") for box in boxes]
    if len(box_ids) != len(set(box_ids)):
        errors.append("duplicate box_id values found")

    box_id_set = {str(box_id) for box_id in box_ids}

    empty_assignment_count = 0
    for customer in customers:
        customer_id = customer.get("customer_id")
        assigned_boxes = customer.get("assigned_boxes")
        if not isinstance(assigned_boxes, list):
            errors.append(f"customer {customer_id} has non-list assigned_boxes")
            continue
        if customer_id != depot_customer_id and not assigned_boxes:
            empty_assignment_count += 1
        for box_id in assigned_boxes:
            if not isinstance(box_id, str):
                errors.append(f"customer {customer_id} has non-string box reference: {box_id}")
            if str(box_id) not in box_id_set:
                errors.append(f"customer {customer_id} references missing box_id {box_id}")

    if empty_assignment_count:
        errors.append(f"{empty_assignment_count} customers have no assigned boxes")

    try:
        container_length = float(container["L"])
        container_width = float(container["W"])
        container_height = float(container["H"])
    except Exception:
        errors.append("container must define numeric L, W, and H")
        container_length = container_width = container_height = 0.0

    oversize_boxes = 0
    for box in boxes:
        try:
            length = float(box["length"])
            width = float(box["width"])
            height = float(box["height"])
        except Exception:
            errors.append(f"box {box.get('box_id')} has non-numeric dimensions")
            continue
        if length > container_length or width > container_width or height > container_height:
            oversize_boxes += 1

    if oversize_boxes:
        errors.append(f"{oversize_boxes} boxes exceed container dimensions")

    route_eval_ok = True
    try:
        _, _, loaded_customers, _ = load_merged(dataset_path)
        route = [
            customer["customer_id"]
            for customer in loaded_customers[1: min(6, len(loaded_customers))]
        ]
        evaluate_route(dataset_path, route, use_packing=False)
    except Exception as exc:
        route_eval_ok = False
        errors.append(f"load_merged/evaluate_route compatibility failed: {exc}")

    summary = {
        "dataset": str(dataset_path.resolve()),
        "customer_count": len(customers),
        "real_customer_count": len(real_customers),
        "box_count": len(boxes),
        "depot_box_count": len(depot_customer.get("assigned_boxes", [])) if depot_customer else 0,
        "route_evaluator_compatible": route_eval_ok,
        "valid": not errors,
    }
    return not errors, errors, summary


def iter_dataset_targets(target_path: Path) -> list[Path]:
    target_path = Path(target_path)
    if target_path.is_file():
        if target_path.suffix.lower() == ".json":
            try:
                payload = json.loads(target_path.read_text(encoding="utf-8-sig"))
            except Exception:
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("datasets"), list):
                return [Path(item["output_path"]) for item in payload.get("datasets", []) if item.get("output_path")]
        return [target_path]

    if target_path.is_dir():
        return sorted(target_path.glob("*_merged_with_boxes_norm.json"))

    raise FileNotFoundError(f"Validation target not found: {target_path}")


def validate_many(target_path: str | Path) -> tuple[bool, list[dict]]:
    dataset_paths = iter_dataset_targets(Path(target_path))
    reports = []
    all_valid = True

    for dataset_path in dataset_paths:
        is_valid, errors, summary = validate_dataset(dataset_path)
        reports.append(
            {
                "path": str(Path(dataset_path).resolve()),
                "valid": is_valid,
                "errors": errors,
                "summary": summary,
            }
        )
        all_valid = all_valid and is_valid

    return all_valid, reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated VRP dataset JSON files.")
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to a dataset JSON file, a generation manifest JSON file, or a directory of dataset files",
    )
    args = parser.parse_args()

    targets = iter_dataset_targets(args.dataset_path)
    if len(targets) == 1 and targets[0].is_file() and targets[0] == args.dataset_path:
        is_valid, errors, summary = validate_dataset(args.dataset_path)

        print(f"Validation target: {summary['dataset']}")
        print(f"Customers: {summary['customer_count']}")
        print(f"Boxes: {summary['box_count']}")
        print(f"Route evaluator compatible: {summary['route_evaluator_compatible']}")

        if is_valid:
            print("Validation passed")
            return 0

        print("Validation failed")
        for error in errors:
            print(f"- {error}")
        return 1

    all_valid, reports = validate_many(args.dataset_path)
    print(f"Validated {len(reports)} datasets")
    for report in reports:
        summary = report["summary"]
        status = "PASS" if report["valid"] else "FAIL"
        print(
            f"[{status}] {Path(report['path']).name} "
            f"customers={summary['customer_count']} boxes={summary['box_count']} "
            f"route_evaluator_compatible={summary['route_evaluator_compatible']}"
        )
        if not report["valid"]:
            for error in report["errors"]:
                print(f"- {error}")

    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
