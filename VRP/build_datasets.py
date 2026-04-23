from __future__ import annotations

import argparse
from pathlib import Path

from dataset_builder import build_dataset
from validate_dataset import validate_dataset


def build_all_datasets(
    vrp_dir: Path,
    output_dir: Path,
    min_boxes_per_customer: int,
    max_boxes_per_customer: int,
    seed: int,
) -> list[Path]:
    generated_paths: list[Path] = []

    for vrp_path in sorted(vrp_dir.glob("*.vrp")):
        output_name = f"{vrp_path.stem}_merged_with_boxes_norm.json"
        output_path = output_dir / output_name
        instance_seed = seed + sum(ord(char) for char in vrp_path.stem)

        _, saved_path = build_dataset(
            vrp_path=vrp_path,
            output_path=output_path,
            min_boxes_per_customer=min_boxes_per_customer,
            max_boxes_per_customer=max_boxes_per_customer,
            seed=instance_seed,
        )

        print(f"Built dataset for {vrp_path.name}")
        print(f"Saved dataset to {saved_path}")

        is_valid, errors, _ = validate_dataset(saved_path)
        if not is_valid:
            raise ValueError(f"Validation failed for {saved_path.name}: {'; '.join(errors)}")

        print(f"Validated dataset {saved_path.name}")
        generated_paths.append(saved_path)

    return generated_paths


def main() -> int:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Build merged VRP datasets with generated boxes.")
    parser.add_argument(
        "--vrp-dir",
        type=Path,
        default=script_dir,
        help="Directory containing .vrp files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "generated_datasets",
        help="Directory where generated datasets will be saved",
    )
    parser.add_argument("--min-boxes-per-customer", type=int, default=1)
    parser.add_argument("--max-boxes-per-customer", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generated_paths = build_all_datasets(
        vrp_dir=args.vrp_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        min_boxes_per_customer=args.min_boxes_per_customer,
        max_boxes_per_customer=args.max_boxes_per_customer,
        seed=args.seed,
    )

    print(f"Generated {len(generated_paths)} dataset files")
    for path in generated_paths:
        print(f"Output file: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
