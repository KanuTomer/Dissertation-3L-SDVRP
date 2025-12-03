# byboxes_pairwise_swap_v2.py
import json, time
from route_ga_vlr_by_boxes import decode_by_boxcount, build_customer_boxcount_map
from route_evaluator import load_merged, evaluate_route

MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
IN_J = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_full.json"   # GA output used as input
OUT_J = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_swapped_v2.json"

# Tunable params
MAX_BOXES = 40         # must match GA run
TOP_K_HEAVY = 6
MAX_SUCCESSFUL_SWAPS = 200
TIME_LIMIT = 600       # seconds

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
    r = routes[idx]
    lst = [(cid, cust_box_map.get(cid,0)) for cid in r]
    lst.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid,_ in lst[:k]]

def perform_swap(order, a, b, ac, bc):
    routes = split(order)
    new_routes = [list(r) for r in routes]
    try:
        ai = new_routes[a].index(ac)
        bj = new_routes[b].index(bc)
    except ValueError:
        return None
    new_routes[a][ai], new_routes[b][bj] = new_routes[b][bj], new_routes[a][ai]
    return [cid for r in new_routes for cid in r]

def load_order():
    d = json.load(open(IN_J))
    return d['best_order']

def report(order):
    routes = split(order)
    for i,r in enumerate(routes,1):
        res = evaluate_route(MERGED, r)
        print(f"Route {i}: len={len(r)} boxes={res['boxes_total']} packed={res['boxes_packed']} unpacked={res['boxes_total']-res['boxes_packed']} feasible={res['feasible']}")

def main():
    order = load_order()
    print("Starting total_unpacked:", total_unpacked(order))
    start = time.time()
    swaps = 0
    improved = True

    while swaps < MAX_SUCCESSFUL_SWAPS and time.time()-start < TIME_LIMIT and improved:
        improved = False
        routes = split(order)
        unpacked = [route_unpacked(r) for r in routes]
        infeasible_idxs = [i for i,u in enumerate(unpacked) if u>0]
        if not infeasible_idxs:
            break
        # create candidate pair order to try: prefer infeasible<->infeasible then infeasible<->feasible
        pairs = []
        for a in infeasible_idxs:
            for b in range(len(routes)):
                if a==b: continue
                pairs.append((a,b))
        # iterate pairs
        for (a,b) in pairs:
            a_cands = find_heavy_candidates(routes, a, TOP_K_HEAVY)
            b_cands = find_heavy_candidates(routes, b, TOP_K_HEAVY)
            found = False
            for ac in a_cands:
                for bc in b_cands:
                    new_order = perform_swap(order, a, b, ac, bc)
                    if not new_order:
                        continue
                    new_un = total_unpacked(new_order)
                    cur_un = total_unpacked(order)
                    if new_un < cur_un:
                        print(f"Swap improved unpacked {cur_un} -> {new_un} by swapping {ac}(R{a+1}) <-> {bc}(R{b+1})")
                        order = new_order
                        swaps += 1
                        improved = True
                        found = True
                        break
                if found: break
            if improved: break
        if not improved:
            print("No improving swaps found in this pass.")
            break

    print("Done. swaps:", swaps, "final_unpacked:", total_unpacked(order), "time_s:", round(time.time()-start,2))
    json.dump({'best_order_fixed': order}, open(OUT_J,'w'), indent=2)
    print("Saved to", OUT_J)
    print("Final report:")
    report(order)

if __name__ == "__main__":
    main()
