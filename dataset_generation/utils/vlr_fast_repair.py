# vlr_fast_repair.py
import json
from route_ga_vlr import decode_variable_routes, build_customer_volume_map
from route_evaluator import load_merged, evaluate_route

MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
IN_J = "route_ga_vlr_out/route_ga_vlr_full.json"
OUT_J = "route_ga_vlr_out/route_ga_vlr_fixed_fast.json"

j = json.load(open(IN_J))
order = j['best_order']

inst, container, customers, boxes = load_merged(MERGED)
container_vol = container['L']*container['W']*container['H']
cust_vol_map = build_customer_volume_map(customers, boxes)

# decode using the same vollimit used to produce the file (0.95)
routes = decode_variable_routes(order, cust_vol_map, container_vol, volume_limit_factor=0.95)

def report_routes(routes):
    for idx, r in enumerate(routes,1):
        res = evaluate_route(MERGED, r)
        print(f"Route {idx}: len={len(r)} boxes={res['boxes_total']} packed={res['boxes_packed']} feasible={res['feasible']} fill={res['fill_rate']:.4f}")
report_routes(routes)

# attempt: for every infeasible route, move its heaviest customer into its own new route (end of permutation)
changed = False
new_routes = [list(r) for r in routes]
for i,r in enumerate(routes):
    res = evaluate_route(MERGED, r)
    if not res['feasible']:
        # compute per-customer volume and pick heaviest
        vols = [(cid, cust_vol_map.get(cid,0.0)) for cid in r]
        vols.sort(key=lambda x: x[1], reverse=True)
        heaviest = vols[0][0]
        # remove heaviest from its route and append as its own route at the end
        new_routes[i].remove(heaviest)
        new_routes.append([heaviest])
        changed = True
        print(f"Moved heavy customer {heaviest} out of route {i+1} into its own route.")

if changed:
    new_order = [cid for rr in new_routes for cid in rr]
    json.dump({'best_order_fixed': new_order}, open(OUT_J,'w'), indent=2)
    print("Saved fixed order to", OUT_J)
    print("New route report:")
    report_routes(new_routes)
else:
    print("No changes made (all routes were feasible).")
