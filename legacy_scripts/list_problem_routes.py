# list_problem_routes.py
import json
from route_evaluator import load_merged
from route_ga_vlr_by_boxes import build_customer_boxcount_map, decode_by_boxcount

MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
ORDER_FILE = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_full.json"
MAX_BOXES = 38

jd = json.load(open(ORDER_FILE))
# find order same as inspect_order_any logic
order = jd.get('best_order') or jd.get('best_order_fixed') or jd.get('best_order_isolated') or jd.get('best_order_from_seed') or jd.get('order') or next((v for v in jd.values() if isinstance(v,list) and all(isinstance(x,int) for x in v)), None)
if order is None:
    raise SystemExit("Could not find order list in JSON")

inst, container, customers, boxes = load_merged(MERGED)
cust_box_map = build_customer_boxcount_map(customers)
routes = decode_by_boxcount(order, cust_box_map, max_boxes_per_route=MAX_BOXES)

problem_idxs = [3,4,11]   # 0-based indices for routes 4,5,12
for p in problem_idxs:
    print("Route", p+1, "customers (id, boxcount):")
    for cid in routes[p]:
        print(cid, cust_box_map.get(cid,0))
    print()
