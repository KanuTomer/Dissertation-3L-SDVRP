import json, sys
from route_evaluator import evaluate_route, load_merged

if len(sys.argv) < 2:
    print("Usage: python inspect_order_arg.py <order_json_path>")
    sys.exit(0)

order_path = sys.argv[1]

# Load the ordering JSON (GA output / repaired output)
with open(order_path, "r") as f:
    order_data = json.load(f)

# Get merged instance path if stored, otherwise ask for default
merged_path = order_data.get("merged_path")
if not merged_path:
    merged_path = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"

inst_name, container, customers, boxes = load_merged(merged_path)

routes = order_data["routes"]

print(f"Inspecting: {order_path}")
print(f"Total routes: {len(routes)}")
print("-" * 60)

total_unpacked = 0

for idx, route in enumerate(routes, 1):
    r = evaluate_route(merged_path, route)
    unpacked = r["unpacked_boxes"]
    total_unpacked += unpacked

    print(f"Route {idx}: len={len(route)} boxes={r['total_boxes']} "
          f"packed={r['packed_boxes']} unpacked={unpacked} "
          f"fill_rate={r['packed_volume']/container['vol']:.4f} "
          f"feasible={r['feasible']}")

print("-" * 60)
print("Total unpacked:", total_unpacked)
