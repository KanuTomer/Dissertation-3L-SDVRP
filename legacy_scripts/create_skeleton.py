#!/usr/bin/env python3
"""
create_skeleton.py

Create a modular package skeleton for the Dataset Generation project.

Usage:
    python create_skeleton.py [--root PATH] [--overwrite]

Default root: current working directory (.)
"""

import argparse
from pathlib import Path
import json
import textwrap

TEMPLATE_FILES = {
    "README.md": """# Dataset Generation

Refactored skeleton for dataset generation used in the dissertation.

Structure:
- dataset_generation/ (package)
- experiments/ (configs)
- experiments_output/ (results)
- main.py (top-level runner)
""",

    "pyproject.toml": """[tool.poetry]
name = "dataset-generation"
version = "0.1.0"
description = "Dataset generation pipeline for Dissertation"

[tool.poetry.dependencies]
python = "^3.10"
""",

    "main.py": '''import argparse
import json
from dataset_generation.runners.run_experiments import run_from_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset generation experiments")
    parser.add_argument("--config", required=True, help="Path to experiment JSON config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    run_from_config(config)
''',

    "experiments/vlr_ga_experiment.json": json.dumps({
        "name": "vlr_ga_experiment",
        "dataset_path": "../datasets/my_route_and_boxes.json",
        "algorithm": "ga",
        "population_size": 60,
        "num_generations": 200,
        "crossover": "order_based",
        "mutation": "vlr",
        "selection": "tournament",
        "repair": "fast",
        "use_isolator": True,
        "output_dir": "experiments_output/vlr_ga_experiment"
    }, indent=2),

    "dataset_generation/__init__.py": '"""Dataset generation package."""\n',

    "dataset_generation/config/__init__.py": '"""Config package for defaults."""\n',
    "dataset_generation/config/defaults.py": '''# default configuration values

DEFAULTS = {
    "population_size": 50,
    "num_generations": 100,
    "output_dir": "experiments_output",
}
''',

    "dataset_generation/loaders/__init__.py": '"""Loaders package."""\n',
    "dataset_generation/loaders/cvrp_loader.py": '''from pathlib import Path
import json

def load_cvrp_and_boxes(path: str) -> dict:
    """Load a merged dataset (CVRP + boxes). Accepts JSON for skeleton."""
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    raise ValueError("Unsupported dataset format in skeleton loader")
''',

    "dataset_generation/algorithms/__init__.py": '"""Algorithms package."""\n',
    "dataset_generation/algorithms/ga/__init__.py": '"""Genetic Algorithm subpackage."""\n',
    "dataset_generation/algorithms/ga/crossover.py": '''def order_crossover(parent_a, parent_b):
    """Order-based crossover skeleton."""
    raise NotImplementedError
''',
    "dataset_generation/algorithms/ga/mutation.py": '''def vlr_mutation(individual):
    """VLR mutation skeleton."""
    raise NotImplementedError
''',
    "dataset_generation/algorithms/ga/selection.py": '''def tournament_selection(population, k=3):
    """Tournament selection skeleton."""
    raise NotImplementedError
''',
    "dataset_generation/algorithms/ga/ga_runner.py": '''from .crossover import order_crossover
from .mutation import vlr_mutation
from .selection import tournament_selection

class GARunner:
    def __init__(self, config: dict):
        self.config = config

    def run(self, data: dict) -> dict:
        # TODO: implement initialization, evaluation, genetic loop
        return {"best_route": [], "metrics": {}}
''',

    "dataset_generation/algorithms/greedy/baseline.py": '''def run_greedy(data: dict) -> dict:
    """Greedy baseline skeleton."""
    return {"route": [], "metrics": {}}
''',

    "dataset_generation/isolators/__init__.py": '"""Isolators package."""\n',
    "dataset_generation/isolators/iterative_isolate.py": '''class IterativeIsolator:
    def __init__(self, config: dict):
        self.config = config

    def isolate(self, result: dict) -> dict:
        # Implement iterative isolation logic here
        return result
''',
    "dataset_generation/isolators/targeted_relocator.py": '''class TargetedRelocator:
    def __init__(self, config: dict):
        self.config = config

    def relocate(self, data: dict) -> dict:
        # implement targeted relocator skeleton
        return data
''',

    "dataset_generation/inspectors/__init__.py": '"""Inspectors package."""\n',
    "dataset_generation/inspectors/inspect_route.py": '''class InspectRoute:
    def __init__(self):
        pass

    def inspect(self, result: dict) -> dict:
        # compute checks, volume validity, ordering, etc.
        return {"valid": True, "issues": []}
''',
    "dataset_generation/inspectors/inspect_boxes.py": '''def inspect_boxes(boxes: dict) -> dict:
    # implement checks for packing validity and box attributes
    return {"ok": True}
''',

    "dataset_generation/parser/parse_boxes.py": '''def parse_boxes(path: str) -> dict:
    # convert box input to internal format
    return {}
''',

    "dataset_generation/runners/__init__.py": '"""Runners package."""\n',
    "dataset_generation/runners/run_ga.py": '''from ..algorithms.ga.ga_runner import GARunner

def run_ga_from_config(config: dict, data: dict):
    runner = GARunner(config)
    return runner.run(data)
''',
    "dataset_generation/runners/run_isolator.py": '''from ..isolators.iterative_isolate import IterativeIsolator

def run_isolator(config: dict, result: dict):
    isolator = IterativeIsolator(config)
    return isolator.isolate(result)
''',
    "dataset_generation/runners/run_experiments.py": '''from ..loaders.cvrp_loader import load_cvrp_and_boxes
from ..algorithms.ga.ga_runner import GARunner
from ..isolators.iterative_isolate import IterativeIsolator
from ..inspectors.inspect_route import InspectRoute
from pathlib import Path
import json

def run_from_config(config: dict):
    data = load_cvrp_and_boxes(config["dataset_path"])
    alg = config.get("algorithm", "ga")
    if alg == "ga":
        runner = GARunner(config)
        result = runner.run(data)
    else:
        raise NotImplementedError("Only GA runner implemented in skeleton")

    if config.get("use_isolator"):
        isolator = IterativeIsolator(config)
        result = isolator.isolate(result)

    inspector = InspectRoute()
    report = inspector.inspect(result)

    out_dir = Path(config.get("output_dir", "experiments_output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    return result
''',

    "dataset_generation/utils/__init__.py": '"""Utils package."""\n',
    "dataset_generation/utils/rotation.py": '''def rotate_box(box, angle_degrees):
    # rotation skeleton
    return box
''',
    "dataset_generation/utils/volume_check.py": '''def check_volume(route) -> dict:
    # volume check skeleton
    return {"ok": True}
''',
    "dataset_generation/utils/merge_cvrp_and_boxes.py": '''def merge_cvrp_and_boxes(cvrp, boxes) -> dict:
    # implementation should merge CVRP route data and box definitions into one dataset
    return {}
''',

    "experiments_output/.gitkeep": "",

    # helpful extras
    "legacy_scripts/.gitkeep": "",
    "dataset_generation/utils/debug.py": '''import logging
logger = logging.getLogger("dataset_generation")
def debug(msg):
    logger.debug(msg)
''',
}

def create_files(root: Path, overwrite: bool = False):
    created = []
    for rel_path, content in TEMPLATE_FILES.items():
        target = root / rel_path
        target_parent = target.parent
        target_parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            print(f"SKIP (exists): {target}")
            continue
        # Write content. If content is already bytes/str
        if isinstance(content, str):
            target.write_text(content)
        else:
            # fallback to JSON/text
            target.write_text(str(content))
        created.append(str(target.relative_to(root)))
        print(f"CREATED: {target}")
    return created

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Root folder to create skeleton in")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"Creating skeleton under: {root}")
    if not root.exists():
        print(f"Root does not exist, creating: {root}")
        root.mkdir(parents=True, exist_ok=True)

    # Safety: ensure the user confirms if root is non-empty and overwrite not set
    non_empty = any(root.iterdir())
    if non_empty and not args.overwrite:
        print("WARNING: Root folder is not empty. Existing files will not be overwritten.")
        print("Run with --overwrite to force writing files (BE CAREFUL).")

    created = create_files(root, overwrite=args.overwrite)
    print("\nDone. Files created/updated:")
    for p in created:
        print(" -", p)
    print("\nNext steps:")
    print(" - Move legacy scripts you want to keep into 'legacy_scripts/'")
    print(" - Implement algorithm details inside dataset_generation/algorithms/")
    print(" - Run your experiment: python main.py --config experiments/vlr_ga_experiment.json")

if __name__ == "__main__":
    main()
