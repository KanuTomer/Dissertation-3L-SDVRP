import sys, json, traceback
from pathlib import Path

ROOT = Path.cwd()
LEGACY = ROOT / "legacy_scripts"
for p in (str(ROOT), str(LEGACY)):
    if p not in sys.path:
        sys.path.insert(0, p)

# load result
res_path = Path("experiments_output/vlr_ga_experiment/result.json")
if not res_path.exists():
    print("result.json not found"); raise SystemExit(1)
r = json.loads(res_path.read_text(encoding="utf-8-sig"))
routes = r.get("best_info",{}).get("routes",[])
# find worst route (unpacked highest)
worst = None
best_unpacked = -1
for route in routes:
    unpacked = route.get("boxes_total",0)-route.get("boxes_packed",0)
    if unpacked>best_unpacked:
        best_unpacked = unpacked
        worst = route
if worst is None:
    print("No routes found"); raise SystemExit(0)

print("Inspecting worst route: unpacked=", best_unpacked, "feasible=", worst.get("feasible"))
route_customers = worst.get("route", [])
print("Route customers count:", len(route_customers))

# dataset path from experiments config
cfg = json.loads(Path("experiments/vlr_ga_experiment.json").read_text(encoding="utf-8-sig"))
dpath = cfg.get("dataset_path")
if not dpath:
    print("Dataset path not found in config"); raise SystemExit(1)
dataset = json.loads(Path(dpath).read_text(encoding="utf-8-sig"))

# build box map & collect boxes
box_map = {b.get("box_id"): b for b in dataset.get("boxes", [])}
boxes_for_route = []
for cid in route_customers:
    c = next((x for x in dataset.get("customers",[]) if x.get("customer_id")==cid or x.get("id")==cid), None)
    if c:
        for bid in c.get("assigned_boxes",[]):
            b = box_map.get(bid)
            if b:
                boxes_for_route.append(b)
            else:
                boxes_for_route.append({"box_id":bid,"length":0,"width":0,"height":0})

print("Collected boxes for route:", len(boxes_for_route))

# try packer
try:
    from dataset_generation.utils.packer import place_boxes_in_container
    cont = dataset.get("container", {})
    L = cont.get("L") or cont.get("length") or cont.get("l") or cont.get("Width") or cont.get("W") or 1.0
    W = cont.get("W") or cont.get("width") or cont.get("w") or 1.0
    H = cont.get("H") or cont.get("height") or cont.get("h") or 1.0
    container = {"L": float(L), "W": float(W), "H": float(H)}
    print("Container used:", container)
    placements, packed_vol, placed_count = place_boxes_in_container(container, boxes_for_route)
    print("Packer gave placed_count:", placed_count, "of", len(boxes_for_route))
    placed_ids = set()
    try:
        for p in placements:
            if isinstance(p, dict):
                if "box_id" in p:
                    placed_ids.add(p["box_id"])
                elif "id" in p:
                    placed_ids.add(p["id"])
    except Exception:
        pass
    if placed_ids:
        unplaced = [b.get("box_id") for b in boxes_for_route if b.get("box_id") not in placed_ids]
        print("Unplaced box ids:", unplaced)
    else:
        print("Packer placements returned but no identifiable box ids; placed_count:", placed_count)
except Exception as e:
    print("Packer invocation failed:", e)
    traceback.print_exc()
    boxes_sorted = sorted(boxes_for_route, key=lambda b: (float(b.get("length",0))*float(b.get("width",0))*float(b.get("height",0))), reverse=True)
    print("Top 15 largest boxes (box_id, l,w,h,volume):")
    for b in boxes_sorted[:15]:
        try:
            vol = float(b.get("length",0))*float(b.get("width",0))*float(b.get("height",0))
        except Exception:
            vol = 0
        print(b.get("box_id"), b.get("length"), b.get("width"), b.get("height"), vol)

# write per-customer summary CSV for worst route
box_map = {b.get("box_id"): b for b in dataset.get("boxes", [])}
rows = []
for cid in route_customers:
    c = next((x for x in dataset.get("customers",[]) if x.get("customer_id")==cid or x.get("id")==cid), None)
    if not c:
        continue
    bids = c.get("assigned_boxes", [])
    vol_sum = 0.0
    for bid in bids:
        b = box_map.get(bid)
        if b:
            try:
                vol_sum += float(b.get("length",0))*float(b.get("width",0))*float(b.get("height",0))
            except Exception:
                pass
    rows.append({"customer_id": cid, "assigned_boxes_count": len(bids), "total_box_volume": vol_sum})

outp = Path("experiments_output/vlr_ga_experiment/worst_route_customer_summary.csv")
with outp.open("w", encoding="utf-8") as fh:
    fh.write("customer_id,assigned_boxes_count,total_box_volume\n")
    for r in rows:
        fh.write(f'{r["customer_id"]},{r["assigned_boxes_count"]},{r["total_box_volume"]}\n')
print("Wrote", outp)
print("Top 10 customers by total_box_volume:")
for r in sorted(rows, key=lambda x: x["total_box_volume"], reverse=True)[:10]:
    print(r)
