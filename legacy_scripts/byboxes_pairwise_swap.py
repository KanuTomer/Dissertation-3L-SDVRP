# byboxes_pairwise_swap.py
import json, time
from route_ga_vlr_by_boxes import decode_by_boxcount, build_customer_boxcount_map
from route_evaluator import load_merged, evaluate_route

MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
IN_J = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_full.json"
OUT_J = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_swapped.json"

# parameters you can tweak
MAX_BOXES = 42           # must match the GA run
TOP_K_HEAVY = 4          # top-K heavy customers per route to consider
MAX_ITERS = 500          # stop after this many successful swaps or attempts
TIME_LIMIT = 600         # seconds hard time limit

def load_order():
    j = json.load(open(IN_J))
    return j['best_order']

inst, container, customers, boxes = load_merged(MERGED)
cust_box_map = build_customer_boxcount_map(customers)

def split(order):
    return decode_by_boxcount(order, cust_box_map, max_boxes_per_route=MAX_BOXES)

def route_unpacked(route):
    r = evaluate_route(MERGED, route)
    return r['boxes_total'] - r['boxes_packed']

def total_unpacked(order):
    routes = split(order)
    return sum(route_unpacked(r) for r in routes)

def find_heavy_candidates(routes, idx, k=TOP_K_HEAVY):
    # return list of customer ids in route idx sorted by boxcount descending (top k)
    r = routes[idx]
    lst = [(cid, cust_box_map.get(cid,0)) for cid in r]
    lst.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid,_ in lst[:k]]

def perform_swap(order, i, j, a_cid, b_cid):
    routes = split(order)
    new_routes = [list(r) for r in routes]
    # remove items
    if a_cid not in new_routes[i] or b_cid not in new_routes[j]:
        return None
    ai = new_routes[i].index(a_cid)
    bj = new_routes[j].index(b_cid)
    new_routes[i][ai], new_routes[j][bj] = new_routes[j][bj], new_routes[i][ai]
    return [cid for r in new_routes for cid in r]

def report(order):
    routes = split(order)
    for idx,r in enumerate(routes,1):
        res = evaluate_route(MERGED, r)
        print(f"Route {idx}: len={len(r)} boxes={res['boxes_total']} packed={res['boxes_packed']} unpacked={res['boxes_total']-res['boxes_packed']} feasible={res['feasible']}")

def main():
    order = load_order()
    start = time.time()
    best_un = total_unpacked(order)
    print("Starting total_unpacked:", best_un)
    it = 0
    improvements = 0

    while time.time() - start < TIME_LIMIT and improvements < MAX_ITERS:
        routes = split(order)
        unpacked_list = [route_unpacked(r) for r in routes]
        infeasible_idxs = [i for i,u in enumerate(unpacked_list) if u>0]
        if len(infeasible_idxs) == 0:
            break
        # build candidate pairs: try infeasible ↔ infeasible, infeasible ↔ feasible
        candidate_pairs = []
        # prefer trying swaps between infeasible routes first
        for a in infeasible_idxs:
            for b in range(len(routes)):
                if a == b: continue
                candidate_pairs.append((a,b))
        progressed = False
        # try pairs in order
        for (a,b) in candidate_pairs:
            a_cands = find_heavy_candidates(routes, a, TOP_K_HEAVY)
            b_cands = find_heavy_candidates(routes, b, TOP_K_HEAVY)
            for ac in a_cands:
                for bc in b_cands:
                    new_order = perform_swap(order, a, b, ac, bc)
                    if not new_order:
                        continue
                    new_un = total_unpacked(new_order)
                    if new_un < best_un:
                        print(f"Swap improved total_unpacked {best_un} -> {new_un} by swapping {ac}(R{a+1}) <-> {bc}(R{b+1})")
                        order = new_order
                        best_un = new_un
                        improvements += 1
                        progressed = True
                        break
                if progressed:
                    break
            if progressed:
                break
        if not progressed:
            # no single swap helped: break (fast exit)
            print("No improving single swaps found in this pass.")
            break
        it += 1

    print("Finished. improvements:", improvements, "final_unpacked:", best_un, "time_s:", round(time.time()-start,2))
    json.dump({'best_order_fixed': order}, open(OUT_J, 'w'), indent=2)
    print("Saved swapped ordering to", OUT_J)
    print("Final route report:")
    report(order)

if __name__ == "__main__":
    main()
