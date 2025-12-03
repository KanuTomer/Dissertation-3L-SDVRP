# local_repair.py
import json, os, time
from route_ga import decode_into_routes
from route_evaluator import evaluate_route, load_merged

MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
OUT_JSON = "route_ga_out/route_ga_full.json"
OUT_FIXED = "route_ga_out/route_ga_fixed.json"

def load_best():
    j = json.load(open(OUT_JSON))
    return j['best_order']

def save_fixed(order):
    with open(OUT_FIXED, "w") as f:
        json.dump({'best_order_fixed': order}, f, indent=2)
    print("Saved fixed order to", OUT_FIXED)

def unpacked_count_for_route(route):
    res = evaluate_route(MERGED, route)
    return res['boxes_total'] - res['boxes_packed']

def total_unpacked(order, route_size):
    routes = decode_into_routes(order, route_size)
    return sum(unpacked_count_for_route(r) for r in routes)

def route_feasible(route):
    return evaluate_route(MERGED, route)['feasible']

def try_swaps(order, route_size, max_iters=500):
    routes = decode_into_routes(order, route_size)
    n_routes = len(routes)
    best_order = order[:]
    best_unpacked = total_unpacked(best_order, route_size)
    print("Starting unpacked total:", best_unpacked)
    start = time.time()
    it = 0
    improved = True
    while improved and it < max_iters:
        improved = False
        it += 1
        # recompute routes each iteration
        routes = decode_into_routes(best_order, route_size)
        # identify infeasible and feasible route indices
        infeasible_idxs = [i for i,r in enumerate(routes) if not evaluate_route(MERGED, r)['feasible']]
        feasible_idxs = [i for i,r in enumerate(routes) if evaluate_route(MERGED, r)['feasible']]
        if not infeasible_idxs or not feasible_idxs:
            break
        # try swapping each customer in infeasible routes with each customer in feasible routes
        for i in infeasible_idxs:
            for j in feasible_idxs:
                for a_idx in range(len(routes[i])):
                    for b_idx in range(len(routes[j])):
                        new_routes = [list(r) for r in routes]
                        # swap customer ids
                        new_routes[i][a_idx], new_routes[j][b_idx] = new_routes[j][b_idx], new_routes[i][a_idx]
                        # flatten
                        cand_order = [cid for r in new_routes for cid in r]
                        cand_unpacked = total_unpacked(cand_order, route_size)
                        if cand_unpacked < best_unpacked:
                            print(f"Iter {it}: swap reduced unpacked {best_unpacked} -> {cand_unpacked} (route{i+1}[{a_idx}] <-> route{j+1}[{b_idx}])")
                            best_unpacked = cand_unpacked
                            best_order = cand_order[:]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    dur = time.time()-start
    print("Done. Iterations:", it, "time_s:", round(dur,2), "final_unpacked:", best_unpacked)
    save_fixed(best_order)
    return best_order

if __name__ == "__main__":
    route_size = 8
    order = load_best()
    fixed = try_swaps(order, route_size, max_iters=200)
    # print summary of before/after
    print("Before total_unpacked:", total_unpacked(load_best(), route_size))
    print("After  total_unpacked:", total_unpacked(fixed, route_size))
    print("You can evaluate fixed ordering with evaluate_route per route or re-run GA diagnostics.")
