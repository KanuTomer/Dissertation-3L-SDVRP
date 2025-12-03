# inspect_vlr.py
import json
from route_evaluator import evaluate_route
from route_ga_vlr import decode_variable_routes, build_customer_volume_map
from route_evaluator import load_merged

MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
JPATH = "route_ga_vlr_out/route_ga_vlr_full.json"

j = json.load(open(JPATH))
order = j['best_order']

inst, container, customers, boxes = load_merged(MERGED)
container_vol = container['L']*container['W']*container['H']
cust_vol_map = build_customer_volume_map(customers, boxes)
routes = decode_variable_routes(order, cust_vol_map, container_vol, volume_limit_factor=0.95)

for idx, r in enumerate(routes, start=1):
    res = evaluate_route(MERGED, r)
    print(f"Route {idx}: len={len(r)} customers, boxes={res['boxes_total']}, packed={res['boxes_packed']}, fill_rate={res['fill_rate']:.4f}, feasible={res['feasible']}")
