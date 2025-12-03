# local_repair_fast.py
import json, time
from route_ga import decode_into_routes
from route_evaluator import evaluate_route

MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
IN_JSON = "route_ga_out/route_ga_full.json"
OUT_JSON = "route_ga_out/route_ga_fast_fixed.json"

def load_order():
    return json.load(open(IN_JSON))['best_order']

def save(order):
    with open(OUT_JSON, "w") as f:
        json.dump({"best_order_fixed": order}, f, indent=2)
    print("Saved:", OUT_JSON)

def unpacked(route):
    r = evaluate_route(MERGED, route)
    return r["boxes_total"] - r["boxes_packed"]

def total_unpacked(order, k):
    return sum(unpacked(r) for r in decode_into_routes(order, k))

def fast_fix(order, k):
    best = order[:]
    best_un = total_unpacked(best, k)
    print("Initial unpacked:", best_un)

    routes = decode_into_routes(best, k)
    infeasible = [i for i,r in enumerate(routes) if unpacked(r) > 0]
    feasible    = [i for i,r in enumerate(routes) if unpacked(r) == 0]

    print("Infeasible routes:", infeasible)

    # Try ONLY one-customer relocations from infeasible → feasible
    for i in infeasible:
        Ri = routes[i]
        for j in feasible:
            Rj = routes[j]
            for a in range(len(Ri)):
                new_routes = [list(r) for r in routes]

                moved = new_routes[i].pop(a)
                new_routes[j].append(moved)

                new_order = [cid for r in new_routes for cid in r]
                un = total_unpacked(new_order, k)

                if un < best_un:
                    print(f"Improvement {best_un} → {un} by moving customer {moved} from route {i+1} to {j+1}")
                    best_un = un
                    best = new_order[:]
                    save(best)
                    return best  # stop early (fast exit)

    print("No fast improvement found.")
    save(best)
    return best

if __name__ == "__main__":
    route_size = 8
    order = load_order()
    fixed = fast_fix(order, route_size)
    print("Final unpacked:", total_unpacked(fixed, route_size))
