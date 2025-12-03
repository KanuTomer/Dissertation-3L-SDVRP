# byboxes_final_targeted_isolate.py
import json
from route_ga_vlr_by_boxes import decode_by_boxcount, build_customer_boxcount_map
from route_evaluator import load_merged, evaluate_route

MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
IN_J = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_focused_swapped.json"
OUT_J = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_final_isolated.json"
MAX_BOXES = 38

# load ordering
jd = json.load(open(IN_J))
order = jd.get('best_order_fixed') or jd.get('best_order') or jd.get('best_order_isolated') or jd.get('best_order_from_isolated') or jd.get('best_order_from_seed') or jd.get('order') or next((v for v in jd.values() if isinstance(v,list) and all(isinstance(x,int) for x in v)), None)
if order is None:
    raise SystemExit("Could not find order list in JSON")

inst, container, customers, boxes = load_merged(MERGED)
cust_box_map = build_customer_boxcount_map(customers)

routes = decode_by_boxcount(order, cust_box_map, max_boxes_per_route=MAX_BOXES)

# compute unpacked per route and collect problem routes
problem_routes = []
for idx, r in enumerate(routes):
    res = evaluate_route(MERGED, r)
    unpacked = res['boxes_total'] - res['boxes_packed']
    if unpacked > 0:
        problem_routes.append((idx, unpacked))

if not problem_routes:
    print("No problem routes found — nothing to isolate.")
    raise SystemExit

print("Problem routes (0-based idx, unpacked_count):", problem_routes)

# make mutable copy
new_routes = [list(r) for r in routes]

# For each problem route, isolate `unpacked` heaviest customers (by cust_box_map)
for idx, to_isolate in problem_routes:
    r = new_routes[idx]
    # sort customers by descending box count
    sorted_custs = sorted(r, key=lambda c: cust_box_map.get(c,0), reverse=True)
    # pick top `to_isolate` customers to isolate
    pick = sorted_custs[:to_isolate]
    for cid in pick:
        new_routes[idx].remove(cid)
        new_routes.append([cid])
        print(f"Isolated customer {cid} from route {idx+1}")

# build the flat order and save
new_order = [c for rr in new_routes for c in rr]
json.dump({'best_order_final_isolated': new_order}, open(OUT_J, 'w'), indent=2)
print("Saved final isolated order to", OUT_J)

# print new report
print("\nInspecting after isolation:")
for i, r in enumerate(new_routes, start=1):
    res = evaluate_route(MERGED, r)
    boxes_total = res['boxes_total']
    boxes_packed = res['boxes_packed']
    unpacked = boxes_total - boxes_packed
    print(f"Route {i:2d}: len={len(r):2d} boxes={boxes_total:3d} packed={boxes_packed:3d} unpacked={unpacked:2d} feasible={res['feasible']}")
