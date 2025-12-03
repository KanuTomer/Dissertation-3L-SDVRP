# byboxes_targeted_relocator.py
import json, time
from route_ga_vlr_by_boxes import decode_by_boxcount, build_customer_boxcount_map, decode_by_boxcount as dummy
from route_evaluator import load_merged, evaluate_route

MERGED = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
IN_J = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_full.json"
OUT_J = "route_ga_vlr_byboxes_out/route_ga_vlr_byboxes_reloc_fixed.json"

def load_current():
    d = json.load(open(IN_J))
    return d['best_order']

inst, container, customers, boxes = load_merged(MERGED)
cust_box_map = build_customer_boxcount_map(customers)
max_boxes = 42  # same as current run; adjust if needed

def split(order):
    return decode_by_boxcount(order, cust_box_map, max_boxes_per_route=max_boxes)

def total_unpacked(order):
    routes = split(order)
    return sum(evaluate_route(MERGED, r)['boxes_total'] - evaluate_route(MERGED, r)['boxes_packed'] for r in routes)

def route_unpacked(route):
    r = evaluate_route(MERGED, route)
    return r['boxes_total'] - r['boxes_packed']

def report(order):
    routes = split(order)
    for i,r in enumerate(routes,1):
        res = evaluate_route(MERGED, r)
        print(f"Route {i}: len={len(r)} boxes={res['boxes_total']} packed={res['boxes_packed']} unpacked={res['boxes_total']-res['boxes_packed']} feasible={res['feasible']}")

def attempt_relocations(order, max_iters=200):
    best = order[:]
    best_un = total_unpacked(best)
    print("Starting total_unpacked:", best_un)
    it = 0
    improved = False
    start = time.time()
    while it < max_iters:
        it += 1
        routes = split(best)
        # recompute unpacked per route and list indices
        unpacked_list = [route_unpacked(r) for r in routes]
        infeasible_idxs = [i for i,u in enumerate(unpacked_list) if u>0]
        if not infeasible_idxs:
            break
        # targets are routes with smallest unpacked (prefer feasible routes or small unpacked)
        target_idxs = sorted(range(len(routes)), key=lambda i: unpacked_list[i])
        changed_this_iter = False
        # try each infeasible route, pick heaviest customer and try move to target routes
        for i in infeasible_idxs:
            Ri = routes[i]
            # compute per-customer box counts (heaviest first)
            cand = sorted([(cid, cust_box_map.get(cid,0)) for cid in Ri], key=lambda x: x[1], reverse=True)
            for (cid, boxes_ct) in cand[:4]:  # only top 4 heaviest, keep it fast
                for j in target_idxs:
                    if j == i: continue
                    Rj = routes[j]
                    # try append to end of target route
                    new_routes = [list(r) for r in routes]
                    # remove cid from Ri
                    new_routes[i].remove(cid)
                    # attempt to add to j (keep order by appending)
                    new_routes[j].append(cid)
                    cand_order = [c for r in new_routes for c in r]
                    cand_un = total_unpacked(cand_order)
                    if cand_un < best_un:
                        print(f"Iter {it}: moved cust {cid} from route {i+1} -> route {j+1} : unpacked {best_un} -> {cand_un}")
                        best_un = cand_un
                        best = cand_order[:]
                        changed_this_iter = True
                        improved = True
                        break
                if changed_this_iter:
                    break
            if changed_this_iter:
                break
        if not changed_this_iter:
            break
    dur = time.time()-start
    print("Done. iters:", it, "time_s:", round(dur,2), "final_unpacked:", best_un, "improved:", improved)
    json.dump({'best_order_fixed': best}, open(OUT_J, 'w'), indent=2)
    print("Saved fixed ordering to", OUT_J)
    return best

if __name__ == "__main__":
    cur = load_current()
    print("Before:")
    report(cur)
    fixed = attempt_relocations(cur, max_iters=400)
    print("\nAfter:")
    report(fixed)
    print("You can inspect the fixed ordering with the usual inspect_byboxes.py (change path if needed).")
