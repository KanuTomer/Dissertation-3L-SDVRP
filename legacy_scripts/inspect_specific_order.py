# inspect_specific_order.py
import json
from route_evaluator import load_merged, evaluate_route
from route_ga_vlr_by_boxes import decode_by_boxcount, build_customer_boxcount_map

MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
ORDER_FILE = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_isolated.json"  # isolated ordering

j = json.load(open(ORDER_FILE))
order = j.get('best_order_isolated', j.get('best_order_fixed', j.get('best_order')))

inst, container, customers, boxes = load_merged(MERGED)
cust_box_map = build_customer_boxcount_map(customers)

routes = decode_by_boxcount(order, cust_box_map, max_boxes_per_route=40)

for idx, r in enumerate(routes, start=1):
    res = evaluate_route(MERGED, r)
    print(f"Route {idx}: len={len(r)} boxes={res['boxes_total']} packed={res['boxes_packed']} unpacked={res['boxes_total']-res['boxes_packed']} feasible={res['feasible']}")
