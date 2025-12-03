# inspect_order_any.py
import json, sys
from route_evaluator import load_merged, evaluate_route
from route_ga_vlr_by_boxes import decode_by_boxcount, build_customer_boxcount_map

# Usage: python inspect_order_any.py <order_json_path> [maxboxes]
if len(sys.argv) < 2:
    print("Usage: python inspect_order_any.py <order_json_path> [maxboxes]")
    sys.exit(1)

ORDER_FILE = sys.argv[1]
MAX_BOXES = int(sys.argv[2]) if len(sys.argv) > 2 else 38  # default to 38 (you ran GA with 38)

# load json
with open(ORDER_FILE, "r") as f:
    jd = json.load(f)

# try to find an order list in multiple possible keys
order = None
for key in ("best_order_isolated", "best_order_fixed", "best_order", "best_order_fixed", "best_order_isolated", "best_order_from_seed"):
    if key in jd:
        order = jd[key]
        break

# some scripts saved under 'best_info' or inside 'best' etc.
if order is None:
    if "best" in jd and isinstance(jd["best"], dict) and "order" in jd["best"]:
        order = jd["best"]["order"]
    elif "best_order" in jd:
        order = jd["best_order"]
    elif "order" in jd:
        order = jd["order"]

if order is None:
    # maybe the GA saved the full result with key 'best_order' as stringified list
    for v in jd.values():
        if isinstance(v, list):
            # heuristic: first list of ints of reasonable length
            if len(v) >= 5 and all(isinstance(x, int) for x in v):
                order = v
                break

if order is None:
    print("Could not find an order list in the JSON. Keys present:", ", ".join(jd.keys()))
    sys.exit(1)

# load merged instance (use default merged path unless provided in json)
merged_path = jd.get("merged_path") or "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
inst_name, container, customers, boxes = load_merged(merged_path)

cust_box_map = build_customer_boxcount_map(customers)
routes = decode_by_boxcount(order, cust_box_map, max_boxes_per_route=MAX_BOXES)

total_unpacked = 0
print("Inspecting order from:", ORDER_FILE)
print("Using merged file:", merged_path)
print("Max boxes per route (decoder):", MAX_BOXES)
print("Number of routes:", len(routes))
print("-" * 72)

for idx, r in enumerate(routes, start=1):
    res = evaluate_route(merged_path, r)
    # support multiple possible return key names
    boxes_total = res.get("boxes_total") or res.get("total_boxes") or res.get("boxes")
    boxes_packed = res.get("boxes_packed") or res.get("packed_boxes") or res.get("packed")
    unpacked = boxes_total - boxes_packed
    fill_rate = res.get("fill_rate") or (res.get("packed_volume",0)/container.get('vol', container['L']*container['W']*container['H']))
    feasible = res.get("feasible")
    total_unpacked += unpacked
    print(f"Route {idx:2d}: len={len(r):2d} boxes={boxes_total:3d} packed={boxes_packed:3d} unpacked={unpacked:2d} fill_rate={fill_rate:.4f} feasible={feasible}")
print("-" * 72)
print("Total unpacked boxes:", total_unpacked)
