# byboxes_iterative_isolate.py  (robust version)
import json, time
from route_ga_vlr_by_boxes import decode_by_boxcount, build_customer_boxcount_map
from route_evaluator import load_merged, evaluate_route

MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
IN_J = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_full.json"   # use existing full.json
OUT_J = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_iter_isolated.json"
MAX_BOXES = 38
MAX_ISOLATIONS = 120   # safety cap
SLEEP = 0.05

def load_order(jpath):
    jd = json.load(open(jpath))
    # try keys we used before
    for key in ('best_order_fixed','best_order','best_order_isolated','best_order_from_isolated','best_order_from_seed','order'):
        if key in jd:
            return jd[key]
    # fallback: first big int list
    for v in jd.values():
        if isinstance(v, list) and len(v) > 5 and all(isinstance(x,int) for x in v):
            return v
    raise SystemExit("No order list found in " + jpath)

def save_order(order, path):
    json.dump({'best_order_iter_isolated': order}, open(path,'w'), indent=2)

inst, container, customers, boxes = load_merged(MERGED)
cust_box_map = build_customer_boxcount_map(customers)

order = load_order(IN_J)

def decode(o):
    return decode_by_boxcount(o, cust_box_map, max_boxes_per_route=MAX_BOXES)

def total_unpacked(o):
    routes = decode(o)
    tot = 0
    for r in routes:
        res = evaluate_route(MERGED, r)
        tot += (res['boxes_total'] - res['boxes_packed'])
    return tot

def unpack_per_route_and_routes(o):
    routes = decode(o)
    out = []
    for r in routes:
        res = evaluate_route(MERGED, r)
        out.append((r, res['boxes_total'] - res['boxes_packed']))
    return out, routes

isolations = 0
start = time.time()
print("Starting total_unpacked:", total_unpacked(order))

# iterative loop
while True:
    up_list, routes = unpack_per_route_and_routes(order)
    problem_idxs = [i for i,(r,u) in enumerate(up_list) if u>0]
    if not problem_idxs:
        break
    if isolations >= MAX_ISOLATIONS:
        print("Reached isolation cap:", MAX_ISOLATIONS)
        break

    # We'll attempt to isolate ONE heaviest customer from each problematic route in one pass.
    changed_this_pass = False
    for idx in problem_idxs:
        # recompute routes (safe)
        up_list, routes = unpack_per_route_and_routes(order)
        if idx >= len(routes):
            continue
        route_r, unpacked_count = up_list[idx]
        if unpacked_count <= 0 or not route_r:
            continue
        # choose heaviest customer in that decoded route
        sorted_custs = sorted(route_r, key=lambda c: cust_box_map.get(c,0), reverse=True)
        heaviest = sorted_custs[0]

        # find which decoded route currently contains `heaviest`
        found = False
        for j, rr in enumerate(routes):
            if heaviest in rr:
                # remove from rr and append single-customer route
                rr_copy = [x for x in rr]  # defensive copy
                rr_copy.remove(heaviest)
                routes[j] = rr_copy
                routes.append([heaviest])
                # flatten routes into order
                order = [c for rr in routes for c in rr]
                isolations += 1
                changed_this_pass = True
                found = True
                print(f"Isolated {heaviest} from route {j+1}  (isolations={isolations})")
                break
        if not found:
            # If it's not found (weird), skip safely
            print(f"Warning: heaviest {heaviest} not found in decoded routes - skipping")
        time.sleep(SLEEP)
        if isolations >= MAX_ISOLATIONS:
            break

    if not changed_this_pass:
        # nothing changed this pass -> stop
        break

final_un = total_unpacked(order)
save_order(order, OUT_J)
print("Done. isolations:", isolations, "final_unpacked:", final_un, "time_s:", round(time.time()-start,1))
# final inspect (print summary)
routes = decode(order)
for i,r in enumerate(routes,1):
    res = evaluate_route(MERGED, r)
    boxes_total = res['boxes_total']
    boxes_packed = res['boxes_packed']
    unpacked = boxes_total - boxes_packed
    print(f"Route {i:2d}: len={len(r):2d} boxes={boxes_total:3d} packed={boxes_packed:3d} unpacked={unpacked:2d} feasible={res['feasible']}")
print("Saved to", OUT_J)
