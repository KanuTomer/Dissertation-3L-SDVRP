# show_problem_routes.py
import json
from route_ga import decode_into_routes
from route_evaluator import load_merged

j = json.load(open('route_ga_out/route_ga_full.json'))
order = j['best_order']
routes = decode_into_routes(order, 8)

inst, container, customers, boxes = load_merged("../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json")
cust_map = {c['customer_id']: c for c in customers}

for idx, r in enumerate(routes, start=1):
    cust_ids = r
    # print customer ids and (customer id : number of boxes assigned)
    counts = []
    for cid in cust_ids:
        c = cust_map.get(cid)
        counts.append((cid, len(c.get('assigned_boxes', [])) if c else None))
    print(f"Route {idx}: customers = {cust_ids}")
    print(" customer -> boxes_assigned:", counts)
    print()
