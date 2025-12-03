import json
from pathlib import Path
from dataset_generation.algorithms.ga.ga_runner import GARunner
try:
    from dataset_generation.isolators.iterative_isolate import IterativeIsolator
    _HAS_ISOLATOR = True
except Exception:
    _HAS_ISOLATOR = False

def run_from_config(config: dict):
    dataset_path = config.get("dataset_path")
    if not dataset_path:
        raise ValueError("config must contain 'dataset_path'")

    p = Path(dataset_path)
    if not p.exists():
        p_rel = Path.cwd() / dataset_path
        if p_rel.exists():
            dataset_path = str(p_rel)
        else:
            raise FileNotFoundError(f"Dataset file not found: {dataset_path} (also tried {p_rel})")

    out_dir = Path(config.get("output_dir", "experiments_output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    alg = config.get("algorithm", "ga")
    if alg == "ga":
        runner = GARunner(config)
        result = runner.run(dataset_path)
    else:
        raise NotImplementedError("Only GA implemented in skeleton")

    # Save result
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))

    # Save history.csv (generation index, best_score)
    try:
        history = result.get("history", [])
        csv_lines = ["gen,best_score"]
        for i, val in enumerate(history):
            csv_lines.append(f"{i+1},{val}")
        (out_dir / "history.csv").write_text("\n".join(csv_lines))
    except Exception as e:
        print("Could not write history.csv:", e)

    # Optional isolator / repair pass
    if config.get("use_isolator") and _HAS_ISOLATOR:
        try:
            isolator = IterativeIsolator(config)
            repaired = isolator.isolate(result)
            (out_dir / "result_repaired.json").write_text(json.dumps(repaired, indent=2))
        except Exception as e:
            print("Isolator failed:", e)

    # Inspector (if exists)
    try:
        from dataset_generation.inspectors.inspect_route import InspectRoute
        inspector = InspectRoute()
        report = inspector.inspect(result)
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    except Exception:
        # ignore if inspector not available
        pass

    return result
