import json, subprocess, sys
from pathlib import Path

cfg_path = Path("experiments/vlr_ga_experiment.json")
if not cfg_path.exists():
    print("Config not found:", cfg_path)
    raise SystemExit(1)

# Load original config
orig = json.loads(cfg_path.read_text(encoding="utf-8-sig"))

# Prepare output
out_csv = Path("experiments_output/batch_summary.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", encoding="utf-8") as f:
    f.write("seed,out_dir,best_score,unpacked,infeasible,duration\n")

def run_one(seed, out_dir):
    cfg = dict(orig)   # copy template
    cfg["seed"] = seed
    cfg["output_dir"] = out_dir

    # Make per-run temporary config
    temp = Path(f"experiments/temp_cfg_{seed}.json")
    temp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Execute run
    cmd = [sys.executable, "main.py", "--config", str(temp)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)

    # Remove temp config
    temp.unlink(missing_ok=True)

    # Parse result
    resf = Path(out_dir) / "result.json"
    if resf.exists():
        r = json.loads(resf.read_text(encoding="utf-8-sig"))
        best = r.get("best_score", "NA")
        routes = r.get("best_info", {}).get("routes", [])
        unpacked = sum(
            (rt.get("boxes_total", 0) - rt.get("boxes_packed", 0))
            for rt in routes
        )
        infeasible = sum(1 for rt in routes if not rt.get("feasible", True))
        dur = r.get("duration", "NA")
    else:
        best, unpacked, infeasible, dur = "NA", "NA", "NA", "NA"

    return best, unpacked, infeasible, dur

# ---- Batch run config ----
seed_start = 1000
replicates = 10

# Run loop
for i in range(replicates):
    seed = seed_start + i
    out_dir = f"experiments_output/{orig.get('name','exp')}_seed{seed}"

    best, unpacked, infeasible, dur = run_one(seed, out_dir)

    with out_csv.open("a", encoding="utf-8") as f:
        f.write(f"{seed},{out_dir},{best},{unpacked},{infeasible},{dur}\n")

print("Batch complete. Summary at", out_csv)
