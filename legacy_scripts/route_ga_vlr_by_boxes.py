# route_ga_vlr_by_boxes.py
import argparse, random, time, os, json
from math import sqrt
from route_evaluator import load_merged, evaluate_route

def euclid(a,b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def route_distance(depot, customers_map, route):
    if not route: return 0.0
    dist = 0.0
    first = customers_map[route[0]]
    dist += euclid((depot[0], depot[1]), (first['x'], first['y']))
    for i in range(len(route)-1):
        c1 = customers_map[route[i]]; c2 = customers_map[route[i+1]]
        dist += euclid((c1['x'], c1['y']), (c2['x'], c2['y']))
    last = customers_map[route[-1]]
    dist += euclid((last['x'], last['y']), (depot[0], depot[1]))
    return dist

def build_customer_boxcount_map(customers):
    # customers have 'assigned_boxes' list
    return {c['customer_id']: len(c.get('assigned_boxes', [])) for c in customers}

def decode_by_boxcount(order, cust_boxcount_map, max_boxes_per_route=48):
    routes = []
    cur = []
    cur_boxes = 0
    for cid in order:
        b = cust_boxcount_map.get(cid, 0)
        # if adding this customer would exceed limit and current route not empty -> start new
        if cur and (cur_boxes + b > max_boxes_per_route):
            routes.append(cur)
            cur = [cid]
            cur_boxes = b
        else:
            cur.append(cid)
            cur_boxes += b
    if cur:
        routes.append(cur)
    return routes

def evaluate_permutation(merged_json_path, perm, max_boxes_per_route, penalty_factor):
    inst_name, container, customers, boxes = load_merged(merged_json_path)
    cust_box_map = build_customer_boxcount_map(customers)
    routes = decode_by_boxcount(perm, cust_box_map, max_boxes_per_route=max_boxes_per_route)
    if customers:
        depot = (customers[0]['x'], customers[0]['y'])
    else:
        depot = (0,0)
    customers_map = {c['customer_id']: c for c in customers}
    total_distance = 0.0
    infeasible_count = 0
    details = []
    for r in routes:
        dist = route_distance(depot, customers_map, r)
        pack = evaluate_route(merged_json_path, r)
        feasible = pack['feasible']
        if not feasible:
            infeasible_count += 1
        total_distance += dist
        details.append({'route': r, 'distance': dist, 'feasible': feasible,
                        'boxes_total': pack['boxes_total'], 'boxes_packed': pack['boxes_packed'],
                        'fill_rate': pack['fill_rate']})
    if infeasible_count == 0:
        score = total_distance
    else:
        score = penalty_factor * infeasible_count + 1e-3 * total_distance
    return score, {'total_distance': total_distance, 'infeasible_count': infeasible_count, 'routes': details}

# GA basics (order-based)
def order_crossover(a,b):
    n = len(a)
    i,j = sorted(random.sample(range(n),2))
    child = [-1]*n
    child[i:j+1] = a[i:j+1]
    fill = [x for x in b if x not in child]
    idx = 0
    for k in range(n):
        if child[k] == -1:
            child[k] = fill[idx]; idx+=1
    return child

def swap_mutation(ind, prob):
    if random.random() < prob:
        i,j = random.sample(range(len(ind)),2)
        ind[i],ind[j] = ind[j],ind[i]
    return ind

def tournament_select(pop, scores, k=3):
    best=None; best_s=None
    for _ in range(k):
        i = random.randrange(len(pop))
        s = scores[i]
        if best is None or s < best_s:
            best = pop[i]; best_s = s
    return best

def run_ga_by_boxes(merged, pop_size=80, gens=200, cx_prob=0.9, mut_prob=0.2, max_boxes_per_route=48, penalty_factor=1e8, seed=42):
    random.seed(seed)
    inst_name, container, customers, boxes = load_merged(merged)
    cust_ids = [c['customer_id'] for c in customers]
    n = len(cust_ids)
    pop = [random.sample(cust_ids, n) for _ in range(pop_size-2)]
    pop.append(cust_ids[:]); pop.append(list(reversed(cust_ids)))
    fitness_cache = {}
    def fitness(ind):
        key = tuple(ind)
        if key in fitness_cache:
            return fitness_cache[key]
        val = evaluate_permutation(merged, ind, max_boxes_per_route, penalty_factor)
        fitness_cache[key] = val
        return val
    best=None; best_s=float('inf'); history=[]
    start = time.time()
    for g in range(gens):
        scored = [fitness(ind)[0] for ind in pop]
        for i,s in enumerate(scored):
            if s < best_s:
                best_s = s; best = pop[i][:]
        ranked = sorted(zip(scored, pop), key=lambda x: x[0])
        elite_n = max(4, int(0.05*pop_size))
        new_pop = [ranked[i][1][:] for i in range(elite_n)]
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, scored); p2 = tournament_select(pop, scored)
            if random.random() < cx_prob:
                child = order_crossover(p1, p2)
            else:
                child = p1[:]
            child = swap_mutation(child, mut_prob)
            new_pop.append(child)
        pop = new_pop; history.append(best_s)
    duration = time.time() - start
    best_val, best_info = fitness(best)
    return {'best_order': best, 'best_score': best_val, 'best_info': best_info, 'history': history, 'duration': duration}

def save_result(outdir, res):
    os.makedirs(outdir, exist_ok=True)
    import csv
    with open(os.path.join(outdir,"route_ga_vlr_byboxes_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(['best_score','infeasible_count','total_distance','duration'])
        w.writerow([res['best_score'], res['best_info']['infeasible_count'], res['best_info']['total_distance'], res['duration']])
    with open(os.path.join(outdir,"route_ga_vlr_byboxes_full.json"), "w") as f:
        json.dump({'best_order': res['best_order'], 'best_score': res['best_score'], 'best_info': res['best_info']}, f, indent=2)
    print("Saved results to", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged', required=True)
    parser.add_argument('--pop', type=int, default=80)
    parser.add_argument('--gens', type=int, default=200)
    parser.add_argument('--alpha', type=float, default=1e8)
    parser.add_argument('--maxboxes', type=int, default=48)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', default='route_ga_vlr_byboxes_out')
    args = parser.parse_args()

    print("Running VLR-by-boxcount GA on", args.merged)
    res = run_ga_by_boxes(args.merged, pop_size=args.pop, gens=args.gens, max_boxes_per_route=args.maxboxes, penalty_factor=args.alpha, seed=args.seed)
    print("Done. best_score:", res['best_score'], "infeasible:", res['best_info']['infeasible_count'], "distance:", res['best_info']['total_distance'], "time_s:", round(res['duration'],2))
    save_result(args.outdir, res)
