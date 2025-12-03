import argparse
import json
from dataset_generation.runners.run_experiments import run_from_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset generation experiments")
    parser.add_argument("--config", required=True, help="Path to experiment JSON config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    run_from_config(config)
