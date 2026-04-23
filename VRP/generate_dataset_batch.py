from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from dataset_builder import DEFAULT_CONTAINER, build_dataset
from validate_dataset import validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML_DIR = PROJECT_ROOT.parent / "XML"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated_datasets"
VRP_NAME_RE = re.compile(r"^XML100_(\d{4})_(\d{2})\.vrp$", re.IGNORECASE)


def natural_xml100_sources(xml_dir: Path) -> list[dict]:
    sources = []
    for path in xml_dir.glob("XML100_*.vrp"):
        match = VRP_NAME_RE.match(path.name)
        if not match:
            continue
        group = match.group(1)
        index = int(match.group(2))
        sources.append(
            {
                "path": path.resolve(),
                "name": path.name,
                "stem": path.stem,
                "group": group,
                "index": index,
            }
        )
    return sorted(sources, key=lambda item: (int(item["group"]), item["index"], item["name"]))


def dataset_seed_for_source(source_stem: str, base_seed: int) -> int:
    return base_seed + sum(ord(char) for char in source_stem)


def get_source_dataset(
    source_path: Path,
    base_seed: int,
    min_boxes_per_customer: int,
    max_boxes_per_customer: int,
    container: dict,
    dataset_cache: dict[str, dict],
) -> dict:
    cache_key = str(source_path.resolve())
    if cache_key not in dataset_cache:
        dataset, _ = build_dataset(
            vrp_path=source_path,
            output_path=None,
            min_boxes_per_customer=min_boxes_per_customer,
            max_boxes_per_customer=max_boxes_per_customer,
            seed=dataset_seed_for_source(source_path.stem, base_seed),
            container=container,
        )
        dataset_cache[cache_key] = dataset
    return dataset_cache[cache_key]


def find_start_position(sources: list[dict], group: str, start: int) -> int:
    for index, source in enumerate(sources):
        if source["group"] == group and source["index"] == start:
            return index
    raise FileNotFoundError(f"Could not find starting source XML100_{group}_{start:02d}.vrp")


def build_combined_dataset(
    target_size: int,
    group: str,
    start: int,
    xml_dir: Path,
    output_dir: Path,
    min_boxes_per_customer: int,
    max_boxes_per_customer: int,
    seed: int,
    container: dict | None = None,
) -> tuple[dict, Path]:
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    xml_dir = xml_dir.resolve()
    output_dir = output_dir.resolve()
    container_data = dict(container or DEFAULT_CONTAINER)

    sources = natural_xml100_sources(xml_dir)
    if not sources:
        raise FileNotFoundError(f"No XML100 source files found in {xml_dir}")

    start_pos = find_start_position(sources, group, start)
    dataset_cache: dict[str, dict] = {}

    selected_sources = []
    combined_customers: list[dict] = []
    combined_boxes: list[dict] = []
    next_customer_id = 1
    next_box_id = 1
    remaining_customers = target_size
    dataset_name = f"XML{target_size}_{group}_{start:02d}"
    depot_row = None
    depot = None
    first_capacity = None

    for source_meta in sources[start_pos:]:
        if remaining_customers <= 0:
            break

        source_dataset = get_source_dataset(
            source_path=source_meta["path"],
            base_seed=seed,
            min_boxes_per_customer=min_boxes_per_customer,
            max_boxes_per_customer=max_boxes_per_customer,
            container=container_data,
            dataset_cache=dataset_cache,
        )

        source_customers = list(source_dataset["customers"])
        source_boxes = {str(box["box_id"]): box for box in source_dataset["boxes"]}
        source_depot_id = int(source_dataset["depot_id"])

        if depot_row is None:
            first_capacity = int(source_dataset.get("capacity", 0) or 0)
            depot = list(source_dataset["depot"])
            original_depot = next(customer for customer in source_customers if int(customer["customer_id"]) == source_depot_id)

            depot_row = {
                "id": next_customer_id,
                "customer_id": next_customer_id,
                "x": float(original_depot["x"]),
                "y": float(original_depot["y"]),
                "demand": int(original_depot["demand"]),
                "is_depot": True,
                "assigned_boxes": [],
            }
            combined_customers.append(depot_row)
            next_customer_id += 1

        source_non_depot = [
            customer
            for customer in source_customers
            if int(customer["customer_id"]) != source_depot_id
        ]
        take_count = min(remaining_customers, len(source_non_depot))
        taken_customers = source_non_depot[:take_count]

        for customer in taken_customers:
            new_assigned_boxes = []
            for old_box_id in customer.get("assigned_boxes", []):
                source_box = source_boxes[str(old_box_id)]
                new_box_id = f"box_{next_box_id}"
                next_box_id += 1
                combined_boxes.append(
                    {
                        "box_id": new_box_id,
                        "length": float(source_box["length"]),
                        "width": float(source_box["width"]),
                        "height": float(source_box["height"]),
                    }
                )
                new_assigned_boxes.append(new_box_id)

            combined_customers.append(
                {
                    "id": next_customer_id,
                    "customer_id": next_customer_id,
                    "x": float(customer["x"]),
                    "y": float(customer["y"]),
                    "demand": int(customer["demand"]),
                    "is_depot": False,
                    "assigned_boxes": new_assigned_boxes,
                }
            )
            next_customer_id += 1

        selected_sources.append(
            {
                "source_file": source_meta["name"],
                "source_group": source_meta["group"],
                "source_index": source_meta["index"],
                "customers_used": take_count,
                "source_dataset_seed": dataset_seed_for_source(source_meta["stem"], seed),
            }
        )
        remaining_customers -= take_count

    if remaining_customers > 0:
        raise ValueError(
            f"Not enough XML100 sources to build XML{target_size} starting from XML100_{group}_{start:02d}. "
            f"Still missing {remaining_customers} customers."
        )

    capacity_scale = max(1, round(target_size / 100))
    combined_dataset = {
        "instance_name": dataset_name,
        "inst_name": dataset_name,
        "name": dataset_name,
        "depot": depot,
        "depot_id": 1,
        "capacity": int((first_capacity or 0) * capacity_scale),
        "container": {
            "L": float(container_data["L"]),
            "W": float(container_data["W"]),
            "H": float(container_data["H"]),
        },
        "customers": combined_customers,
        "boxes": combined_boxes,
        "source_files": [item["source_file"] for item in selected_sources],
        "source_usage": selected_sources,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_merged_with_boxes_norm.json"
    output_path.write_text(json.dumps(combined_dataset, indent=2), encoding="utf-8")
    return combined_dataset, output_path


def available_group_starts(
    sources: list[dict],
    group: str,
    start_at: int = 1,
) -> list[int]:
    return [
        source["index"]
        for source in sources
        if source["group"] == str(group) and source["index"] >= int(start_at)
    ]


def _default_sizes() -> list[int]:
    return list(range(50, 751, 50))


def _resolve_sizes(args) -> list[int]:
    if args.all_sizes or not args.sizes:
        return _default_sizes()
    return sorted({int(size) for size in args.sizes})


def _manifest_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_dataset_matrix(
    *,
    target_sizes: list[int],
    groups: list[str],
    xml_dir: Path,
    output_dir: Path,
    min_boxes_per_customer: int,
    max_boxes_per_customer: int,
    seed: int,
    container: dict | None = None,
    start_at: int = 1,
    variants_per_size: int = 1,
    overwrite: bool = False,
    validate: bool = True,
    manifest_path: Path | None = None,
) -> dict:
    xml_dir = xml_dir.resolve()
    output_dir = output_dir.resolve()
    sources = natural_xml100_sources(xml_dir)
    if not sources:
        raise FileNotFoundError(f"No XML100 source files found in {xml_dir}")

    groups = [str(group) for group in groups]
    target_sizes = sorted({int(size) for size in target_sizes})
    container_data = dict(container or DEFAULT_CONTAINER)

    dataset_entries: list[dict] = []
    skipped_entries: list[dict] = []

    for group in groups:
        starts = available_group_starts(sources, group=group, start_at=start_at)
        if len(starts) < variants_per_size:
            raise ValueError(
                f"Requested {variants_per_size} variants for group {group}, "
                f"but only found {len(starts)} starting positions at or after {start_at}."
            )

        variant_starts = starts[:variants_per_size]
        for variant_index, start in enumerate(variant_starts, start=1):
            for target_size in target_sizes:
                dataset_name = f"XML{target_size}_{group}_{start:02d}"
                output_path = output_dir / f"{dataset_name}_merged_with_boxes_norm.json"
                entry = {
                    "dataset_name": dataset_name,
                    "target_size": target_size,
                    "group": group,
                    "start": start,
                    "variant_index": variant_index,
                    "output_path": str(output_path),
                }

                if output_path.exists() and not overwrite:
                    print(f"Skipping existing dataset: {output_path.name}")
                    skipped_entries.append(entry)
                    continue

                dataset, saved_path = build_combined_dataset(
                    target_size=target_size,
                    group=group,
                    start=start,
                    xml_dir=xml_dir,
                    output_dir=output_dir,
                    min_boxes_per_customer=min_boxes_per_customer,
                    max_boxes_per_customer=max_boxes_per_customer,
                    seed=seed,
                    container=container_data,
                )

                print(f"Generated dataset: {dataset['instance_name']}")
                print(f"Saved to: {saved_path}")
                for item in dataset["source_usage"]:
                    print(f"- {item['source_file']} -> used {item['customers_used']} customers")

                validation_summary = None
                if validate:
                    is_valid, errors, summary = validate_dataset(saved_path)
                    validation_summary = summary
                    if not is_valid:
                        joined_errors = "; ".join(errors)
                        raise ValueError(f"Validation failed for {saved_path.name}: {joined_errors}")
                    print(
                        f"Validated dataset {saved_path.name}: "
                        f"customers={summary['customer_count']} boxes={summary['box_count']}"
                    )

                entry.update(
                    {
                        "output_path": str(saved_path),
                        "source_files": list(dataset.get("source_files", [])),
                        "source_usage": list(dataset.get("source_usage", [])),
                        "validation_summary": validation_summary,
                    }
                )
                dataset_entries.append(entry)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "xml_dir": str(xml_dir),
        "output_dir": str(output_dir),
        "groups": groups,
        "target_sizes": target_sizes,
        "variants_per_size": int(variants_per_size),
        "start_at": int(start_at),
        "seed": int(seed),
        "min_boxes_per_customer": int(min_boxes_per_customer),
        "max_boxes_per_customer": int(max_boxes_per_customer),
        "overwrite": bool(overwrite),
        "validate": bool(validate),
        "datasets": dataset_entries,
        "skipped_existing": skipped_entries,
    }

    if manifest_path is None:
        manifests_dir = output_dir / "_manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifests_dir / f"dataset_batch_{_manifest_timestamp()}.json"
    else:
        manifest_path = manifest_path.resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a custom synthetic dataset by combining consecutive XML100 source files."
    )
    parser.add_argument("--target-size", type=int, help="Target customer count excluding the depot")
    parser.add_argument("--sizes", type=int, nargs="+", help="Explicit target sizes, for example 50 100 150")
    parser.add_argument("--all-sizes", action="store_true", help="Generate sizes 50 through 750 in steps of 50")
    parser.add_argument("--group", help="Starting family group, for example 1111 or 1142")
    parser.add_argument("--groups", nargs="+", help="Generate datasets for multiple family groups")
    parser.add_argument("--start", type=int, help="Starting source index inside the group")
    parser.add_argument("--start-at", type=int, default=1, help="Lowest starting source index for bulk generation")
    parser.add_argument("--variants-per-size", type=int, default=1, help="How many fresh start positions to use per family and size")
    parser.add_argument("--xml-dir", type=Path, default=DEFAULT_XML_DIR, help="Directory containing XML100 source VRP files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated dataset JSON files")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic box generation")
    parser.add_argument("--min-boxes-per-customer", type=int, default=1)
    parser.add_argument("--max-boxes-per-customer", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated datasets if they already exist")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation after generation")
    parser.add_argument("--manifest-path", type=Path, help="Optional path for the generation manifest JSON")
    args = parser.parse_args()

    bulk_mode = bool(args.all_sizes or args.sizes or args.groups or args.variants_per_size > 1 or args.start_at != 1)

    if bulk_mode:
        groups = [str(group) for group in (args.groups or ([args.group] if args.group else []))]
        if not groups:
            raise ValueError("Bulk generation requires --groups or --group.")
        target_sizes = _resolve_sizes(args)
        manifest = build_dataset_matrix(
            target_sizes=target_sizes,
            groups=groups,
            xml_dir=args.xml_dir,
            output_dir=args.output_dir,
            min_boxes_per_customer=args.min_boxes_per_customer,
            max_boxes_per_customer=args.max_boxes_per_customer,
            seed=args.seed,
            start_at=args.start_at,
            variants_per_size=args.variants_per_size,
            overwrite=args.overwrite,
            validate=not args.no_validate,
            manifest_path=args.manifest_path,
        )
        print(
            f"Bulk generation complete: created {len(manifest['datasets'])} datasets, "
            f"skipped {len(manifest['skipped_existing'])} existing datasets"
        )
        return 0

    if args.target_size is None or args.group is None or args.start is None:
        raise ValueError("Single-dataset generation requires --target-size, --group, and --start.")

    dataset, output_path = build_combined_dataset(
        target_size=args.target_size,
        group=str(args.group),
        start=args.start,
        xml_dir=args.xml_dir,
        output_dir=args.output_dir,
        min_boxes_per_customer=args.min_boxes_per_customer,
        max_boxes_per_customer=args.max_boxes_per_customer,
        seed=args.seed,
    )

    print(f"Generated dataset: {dataset['instance_name']}")
    print(f"Saved to: {output_path}")
    print("Source usage:")
    for item in dataset["source_usage"]:
        print(f"- {item['source_file']} -> used {item['customers_used']} customers")

    if args.no_validate:
        return 0

    is_valid, errors, summary = validate_dataset(output_path)
    if not is_valid:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        f"Validation passed: customers={summary['customer_count']} boxes={summary['box_count']} "
        f"route_evaluator_compatible={summary['route_evaluator_compatible']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
