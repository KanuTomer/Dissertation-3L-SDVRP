# tools/normalize_and_sweep.py
import json, csv, sys
from pathlib import Path
import subprocess

# === CONFIG ===
MERGED_DIR = Path(r"C:\Kanu\Dissertation\Benchmark dataset and instance generator for Real-World 3dBPP\Output\merged")
OUT_SUMMARY = Path("experiments_output/all_merged_summary_normalized.csv")
TEMPLATE_CONFIG = Path("experiments/vlr_ga_experiment.json")
RUN_AFTER = False   # set True to run main.py after normalizing each file
POP = None
GENS = None
MUT = None
CX = None
USE_ISO = None

# helpers
def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def attach_boxes_to_customers(data, merged_path):
    customers = data.get("customers", [])
    if any(c.get("assigned_boxes") for c in customers):
        return 0
    boxes_file = data.get("boxes_file")
    attached = 0
    if boxes_file:
        csvp = Path(merged_path).parent / boxes_file
        if csvp.exists():
            with csvp.open("r", encoding="utf-8-sig", newline='') as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            if rows:
                hdrs = list(rows[0].keys())
                box_field = next((h for h in hdrs if 'box' in h.lower() or h.lower() == 'id'), hdrs[0])
                cust_field = next((h for h in hdrs if 'cust' in h.lower() or 'customer' in h.lower() or 'cid' in h.lower()), None)
                if cust_field:
                    cust_map = {str(c.get("customer_id") or c.get("id")): c for c in customers if (c.get("customer_id") or c.get("id")) is not None}
                    for r in rows:
                        bid = r.get(box_field)
                        cid = r.get(cust_field)
                        if bid and cid and str(cid) in cust_map:
                            cust_map[str(cid)].setdefault("assigned_boxes", []).append(bid)
                            attached += 1
                if attached == 0:
                    for i, r in enumerate(rows):
                        bid = r.get(box_field)
                        if not bid: continue
                        c = customers[i % len(customers)]
                        c.setdefault("assigned_boxes", []).append(bid)
                        attached += 1
                if attached > 0:
                    return attached
    boxes = data.get("boxes", [])
    for i, b in enumerate(boxes):
        bid = b.get("box_id") or b.get("id") or str(i)
        c = customers[i % len(customers)]
        c.setdefault("assigned_boxes", []).append(bid)
        attached += 1
    return attached

def normalize_customers(data):
    changed = False
    for i, c in enumerate(data.get("customers", [])):
        if "customer_id" not in c:
            if "id" in c:
                c["customer_id"] = c["id"]
            else:
                c["customer_id"] = i + 1
            changed = True
        if "assigned_boxes" not in c:
            c["assigned_boxes"] = []
            changed = True
    return changed

# main
if not MERGED_DIR.exists():
    print("Merged dir not found:", MERGED_DIR); sys.exit(1)

OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
with OUT_SUMMARY.open("w", encoding="utf-8") as f:
    f.write("file,norm_file,attached,best_score,unpacked,infeasible,duration\n")

merged_files = sorted(MERGED_DIR.glob("*_merged*.json"))
for src in merged_files:
    try:
        print("Processing:", src.name)
        data = read_json(src)
        normalize_customers(data)
        attached = 0
        if not any(c.get("assigned_boxes") for c in data.get("customers", [])):
            attached = attach_boxes_to_customers(data, src)
        norm_path = src.with_name(src.stem + "_norm" + src.suffix)
        write_json(norm_path, data)
        print(f" Wrote: {norm_path.name} (attached={attached})")
    except Exception as e:
        print(" Failed:", src, e)
        continue

    best = "NA"; unpacked="NA"; infeasible="NA"; dur="NA"
    if RUN_AFTER:
        cfg = read_json(TEMPLATE_CONFIG)
        cfg["dataset_path"] = str(norm_path)
        if POP is not None: cfg["population_size"]=POP
        if GENS is not None: cfg["num_generations"]=GENS
        if MUT is not None: cfg["mut_prob"]=MUT
        if CX is not None: cfg["cx_prob"]=CX
        if USE_ISO is not None: cfg["use_isolator"]=USE_ISO
        tmp = Path("experiments/temp_run_norm.json")
        write_json(tmp, cfg)
        subprocess.run([sys.executable, "main.py", "--config", str(tmp)], check=False)
        tmp.unlink(missing_ok=True)
        resf = Path("experiments_output/vlr_ga_experiment/result.json")
        if resf.exists():
            r = read_json(resf)
            best = r.get("best_score","NA")
            routes = r.get("best_info",{}).get("routes",[])
            unpacked = sum((rt.get("boxes_total",0)-rt.get("boxes_packed",0)) for rt in routes)
            infeasible = sum(1 for rt in routes if not rt.get("feasible",True))
            dur = r.get("duration","NA")
    with OUT_SUMMARY.open("a", encoding="utf-8") as f:
        f.write(f'"{src}","{norm_path}",{attached},{best},{unpacked},{infeasible},{dur}\n')

print("Done. Summary:", OUT_SUMMARY)
