import argparse
import json
from pathlib import Path
from dataset_generation.runners.run_experiments import run_from_config

def load_json_sig(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset generation experiments")
    parser.add_argument("--config", required=False, help="Path to experiment JSON config")
    parser.add_argument("--quick", action="store_true", help="Run an internal quick smoke test")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output in runners")
    args = parser.parse_args()

    if args.quick:
        # built-in quick config
        config = {
            "name": "quick_smoke",
            "dataset_path": "datasets/my_route_and_boxes.json",
            "algorithm": "ga",
            "population_size": 10,
            "num_generations": 2,
            "use_isolator": False,
            "output_dir": "experiments_output/quick_smoke",
            "verbose": args.verbose
        }
    else:
        if not args.config:
            raise SystemExit("Either --config or --quick must be provided")
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise SystemExit(f"Config file not found: {cfg_path.resolve()}")
        config = load_json_sig(cfg_path)

    # pass verbosity into config so GARunner sees it
    if args.verbose:
        config["verbose"] = True

    run_from_config(config)
