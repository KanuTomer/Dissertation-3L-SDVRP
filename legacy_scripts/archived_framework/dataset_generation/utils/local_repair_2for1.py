# local_repair_2for1.py
import json, time
from route_ga import decode_into_routes
from route_evaluator import evaluate_route

MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
IN_JSON = "route_ga_out/route_ga_full.json"        # start from GA best
OUT_JSON = "route_ga_out/route_ga_fixed_2for1.json"

def load_order():
    j = json.load(open(IN_JSON))
    return j['best_order']

def save_order(order):
    with open(OUT_JSON, "w") as f:
        json.dump({'best_order_fixed': order}, f, indent=2)
    print("Saved fixed order to", OUT_JSON)

def unpacked_for_route(route):
    r = evaluate_route(MERGED, route)
    return r['boxes_total'] - r['boxes_packed']

def total_unpacked(order, route_size):
    routes = decode_into_routes(order, route_size)
    return sum(unpacked_for_route(r) for r in routes)

def try_2for1(order, route_size, max_outer=2000, max_inner=200):
    best = order[:]
    best_unpacked = total_unpacked(best, route_size)
    print("Start unpacked:", best_unpacked)
    start = time.time()
    it_outer = 0
    improved_any = False

    while it_outer < max_outer:
        it_outer += 1
        # rebuild routes
        routes = decode_into_routes(best, route_size)
        n = len(routes)
        made_progress = False

        # try for each pair of routes (i, j)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                Ri = routes[i]; Rj = routes[j]
                # attempt moving two customers from Ri into Rj while moving one from Rj to Ri
                # iterate small subsets to limit cost
                for a_idx in range(len(Ri)):
                    for b_idx in range(len(Ri)):
                        if a_idx == b_idx: continue
                        for c_idx in range(len(Rj)):
                            # construct candidate routes
                            newRi = Ri[:]
                            newRj = Rj[:]
                            # remove two from Ri (ensure indices distinct and handle order)
                            ids_remove = sorted([a_idx, b_idx], reverse=True)
                            rem1 = newRi.pop(ids_remove[0])
                            rem2 = newRi.pop(ids_remove[1])
                            # pick one from Rj to move to Ri
                            moved_from_j = newRj.pop(c_idx)
                            # apply moves: add moved_from_j into Ri, and append rem1,rem2 into Rj
                            newRi.append(moved_from_j)
                            newRj.extend([rem1, rem2])
                            # flatten new order
                            cand_routes = routes[:]
                            cand_routes[i] = newRi
                            cand_routes[j] = newRj
                            cand_order = [cid for r in cand_routes for cid in r]
                            cand_unpacked = total_unpacked(cand_order, route_size)
                            if cand_unpacked < best_unpacked:
                                print(f"Improved unpacked {best_unpacked} -> {cand_unpacked} (2fromR{i+1}->R{j+1}, 1fromR{j+1}->R{i+1})")
                                best_unpacked = cand_unpacked
                                best = cand_order[:]
                                made_progress = True
                                improved_any = True
                                break
                        if made_progress: break
                    if made_progress: break
                if made_progress: break
            if made_progress: break

        if not made_progress:
            # no progress this outer iteration
            break

    dur = time.time() - start
    print("Done. Iterations:", it_outer, "time_s:", round(dur,2), "final_unpacked:", best_unpacked, "improved_any:", improved_any)
    save_order(best)
    return best

if __name__ == "__main__":
    route_size = 8
    order = load_order()
    _ = try_2for1(order, route_size, max_outer=800, max_inner=100)
    print("Before total_unpacked:", total_unpacked(load_order(), route_size))
    print("After  total_unpacked:", total_unpacked(_, route_size))
