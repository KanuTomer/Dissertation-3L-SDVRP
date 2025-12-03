# byboxes_final_isolate.py
import json
from route_ga_vlr_by_boxes import decode_by_boxcount, build_customer_boxcount_map
from route_evaluator import load_merged, evaluate_route

MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
IN_J = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_swapped_v2.json"  # uses last swapped order
OUT_J = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_isolated.json"
MAX_BOXES = 40

j = json.load(open(IN_J))
order = j.get('best_order_fixed', j.get('best_order'))

inst, container, customers, boxes = load_merged(MERGED)
cust_box_map = build_customer_boxcount_map(customers)

def split(ordr):
    return decode_by_boxcount(ordr, cust_box_map, max_boxes_per_route=MAX_BOXES)

def report(routes):
    for i,r in enumerate(routes,1):
        res = evaluate_route(MERGED, r)
        print(f"Route {i}: len={len(r)} boxes={res['boxes_total']} packed={res['boxes_packed']} unpacked={res['boxes_total']-res['boxes_packed']} feasible={res['feasible']}")

routes = split(order)
print("Before isolation:")
report(routes)

new_routes = [list(r) for r in routes]
changed = False
for i, r in enumerate(routes):
    res = evaluate_route(MERGED, r)
    if not res['feasible']:
        # pick heaviest customer by boxcount
        vols = [(cid, cust_box_map.get(cid,0)) for cid in r]
        vols.sort(key=lambda x: x[1], reverse=True)
        heaviest = vols[0][0]
        new_routes[i].remove(heaviest)
        new_routes.append([heaviest])
        print(f"Isolated customer {heaviest} from route {i+1}")
        changed = True

new_order = [cid for rr in new_routes for cid in rr]
json.dump({'best_order_isolated': new_order}, open(OUT_J,'w'), indent=2)
print("After isolation:")
report(new_routes)
print("Saved:", OUT_J)
