import sys, json, traceback
from pathlib import Path

# Ensure project root is on sys.path so package imports work
ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

res_path = Path("experiments_output/vlr_ga_experiment/result.json")
if not res_path.exists():
    print("result.json not found at", res_path.resolve())
    raise SystemExit(1)

try:
    res = json.loads(res_path.read_text(encoding="utf-8-sig"))
    from dataset_generation.isolators.iterative_isolate import IterativeIsolator
    iso = IterativeIsolator({})
    repaired = iso.isolate(res)
    outp = Path("experiments_output/vlr_ga_experiment/result_repaired.json")
    outp.write_text(json.dumps(repaired, indent=2), encoding="utf-8")
    def summary(r):
        unp = sum((rt.get("boxes_total",0)-rt.get("boxes_packed",0)) for rt in r.get("best_info",{}).get("routes",[]))
        inf = sum(1 for rt in r.get("best_info",{}).get("routes",[]) if not rt.get("feasible",True))
        return unp, inf
    print("Wrote repaired result ->", outp)
    print("Before unpacked,infeasible:", summary(res))
    print("After  unpacked,infeasible:", summary(repaired))
except Exception as e:
    print("Isolator run failed:", e)
    traceback.print_exc()
    raise
