# focused_swap.py
import json, time
from route_ga_vlr_by_boxes import decode_by_boxcount, build_customer_boxcount_map
from route_evaluator import load_merged, evaluate_route

MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
IN_J = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_full.json"
OUT_J = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_focused_swapped.json"
MAX_BOXES = 38
TARGET_ROUTES = [3,4,11]   # zero-based indices (routes 4,5,12)

inst, container, customers, boxes = load_merged(MERGED)
cust_box_map = build_customer_boxcount_map(customers)

def split(order):
    return decode_by_boxcount(order, cust_box_map, max_boxes_per_route=MAX_BOXES)

def route_unpacked(route):
    r = evaluate_route(MERGED, route)
    return r['boxes_total'] - r['boxes_packed']

def total_unpacked(order):
    return sum(route_unpacked(r) for r in split(order))

def top_candidates(routes, idx, k=6):
    r = routes[idx]
    lst = [(cid, cust_box_map.get(cid,0)) for cid in r]
    lst.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid,_ in lst[:k]]

order = json.load(open(IN_J)).get('best_order') or json.load(open(IN_J)).get('best_order_fixed') or json.load(open(IN_J)).get('best_order_isolated')
if order is None:
    # fallback: find first large int list
    jd = json.load(open(IN_J))
    order = next((v for v in jd.values() if isinstance(v,list) and len(v)>10 and all(isinstance(x,int) for x in v)), None)

print("Starting total_unpacked:", total_unpacked(order))
routes = split(order)
improved = True
start = time.time()
swaps = 0
while improved and time.time()-start < 600 and swaps < 200:
    improved = False
    routes = split(order)
    cur_un = total_unpacked(order)
    # try each target route vs every other route
    for a in TARGET_ROUTES:
        for b in range(len(routes)):
            if a==b: continue
            a_cands = top_candidates(routes, a, 6)
            b_cands = top_candidates(routes, b, 6)
            done = False
            for ac in a_cands:
                for bc in b_cands:
                    # perform swap
                    new_routes = [list(r) for r in routes]
                    if ac not in new_routes[a] or bc not in new_routes[b]:
                        continue
                    ai = new_routes[a].index(ac); bj = new_routes[b].index(bc)
                    new_routes[a][ai], new_routes[b][bj] = new_routes[b][bj], new_routes[a][ai]
                    new_order = [c for rr in new_routes for c in rr]
                    new_un = total_unpacked(new_order)
                    if new_un < cur_un:
                        print(f"Swap {ac}(R{a+1}) <-> {bc}(R{b+1}) improved {cur_un}->{new_un}")
                        order = new_order
                        swaps += 1
                        improved = True
                        done = True
                        break
                if done: break
            if improved: break
        if improved: break

print("Done. swaps:", swaps, "final_unpacked:", total_unpacked(order))
json.dump({'best_order_fixed': order}, open(OUT_J, 'w'), indent=2)
print("Saved to", OUT_J)
