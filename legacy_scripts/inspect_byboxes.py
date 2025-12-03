# inspect_byboxes.py
import json
from route_evaluator import evaluate_route, load_merged

JPATH = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_full.json"
MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"

j = json.load(open(JPATH))
order = j['best_order']

# helper to split order into routes using same decoder (maxboxes)
from route_ga_vlr_by_boxes import decode_by_boxcount, build_customer_boxcount_map, load_merged as dummy_load

# load merged to get cust box counts
inst, container, customers, boxes = load_merged(MERGED)
cust_box_map = {c['customer_id']: len(c.get('assigned_boxes', [])) for c in customers}

routes = decode_by_boxcount(order, cust_box_map, max_boxes_per_route=48)

for idx, r in enumerate(routes, start=1):
    res = evaluate_route(MERGED, r)
    print(f"Route {idx}: len={len(r)} customers, boxes={res['boxes_total']}, packed={res['boxes_packed']}, fill_rate={res['fill_rate']:.4f}, feasible={res['feasible']}")
