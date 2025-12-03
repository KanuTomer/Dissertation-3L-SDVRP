# byboxes_fast_repair.py
import json
from route_ga_vlr_by_boxes import decode_by_boxcount, build_customer_boxcount_map
from route_evaluator import load_merged, evaluate_route

MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
IN_J = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_full.json"
OUT_J = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_fixed.json"

j = json.load(open(IN_J))
order = j['best_order']
inst, container, customers, boxes = load_merged(MERGED)
cust_box_map = build_customer_boxcount_map(customers)
routes = decode_by_boxcount(order, cust_box_map, max_boxes_per_route=48)

def report(rs):
    for i,r in enumerate(rs,1):
        res = evaluate_route(MERGED, r)
        print(f"Route {i}: len={len(r)} boxes={res['boxes_total']} packed={res['boxes_packed']} feasible={res['feasible']}")

print("Before:")
report(routes)

changed = False
new_routes = [list(r) for r in routes]
for i, r in enumerate(routes):
    res = evaluate_route(MERGED, r)
    if not res['feasible']:
        vols = [(cid, cust_box_map.get(cid,0)) for cid in r]
        vols.sort(key=lambda x: x[1], reverse=True)
        heavy = vols[0][0]
        new_routes[i].remove(heavy)
        new_routes.append([heavy])
        changed = True
        print(f"Moved heavy customer {heavy} out of route {i+1}")

if changed:
    new_order = [cid for rr in new_routes for cid in rr]
    json.dump({'best_order_fixed': new_order}, open(OUT_J,'w'), indent=2)
    print("After:")
    report(new_routes)
    print("Saved fixed ordering to", OUT_J)
else:
    print("No changes made.")
