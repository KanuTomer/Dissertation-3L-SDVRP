# inspect_fixed.py
import json
from route_ga import decode_into_routes
from route_evaluator import evaluate_route

j = json.load(open('route_ga_out/route_ga_fixed.json'))
order = j['best_order_fixed']
routes = decode_into_routes(order, 8)
for idx, r in enumerate(routes, start=1):
    res = evaluate_route("../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json", r)
    print(f"Route {idx}: len={len(r)} customers, boxes={res['boxes_total']}, packed={res['boxes_packed']}, fill_rate={res['fill_rate']:.4f}, feasible={res['feasible']}")
