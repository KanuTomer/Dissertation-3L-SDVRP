# route_ga_vlr.py
import argparse, random, time, os, json
from math import sqrt
from copy import deepcopy
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

def build_customer_volume_map(customers, boxes):
    # boxes: list of box dicts (sequential ids starting 1)
    box_map = {b['box_id']: b for b in boxes}
    cust_vol = {}
    for c in customers:
        total = 0.0
        for bid in c.get('assigned_boxes', []):
            b = box_map.get(bid)
            if b:
                total += b['length'] * b['width'] * b['height']
        cust_vol[c['customer_id']] = total
    return cust_vol

def decode_variable_routes(order, cust_vol_map, container_vol, volume_limit_factor=0.95):
    """
    Greedy sequential decoder: add customers until route volume + next_customer_volume > container_vol*volume_limit_factor,
    then start new route. Returns list of routes (list of customer id lists).
    """
    routes = []
    cur = []
    cur_vol = 0.0
    limit = container_vol * volume_limit_factor
    for cid in order:
        v = cust_vol_map.get(cid, 0.0)
        if cur and (cur_vol + v > limit):
            routes.append(cur)
            cur = [cid]
            cur_vol = v
        else:
            cur.append(cid)
            cur_vol += v
    if cur:
        routes.append(cur)
    return routes

def evaluate_permutation(merged_json_path, perm, volume_limit_factor, penalty_factor):
    inst_name, container, customers, boxes = load_merged(merged_json_path)
    container_vol = container['L'] * container['W'] * container['H']
    cust_vol_map = build_customer_volume_map(customers, boxes)
    # decoder
    routes = decode_variable_routes(perm, cust_vol_map, container_vol, volume_limit_factor)
    # compute distances and packing
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
        details.append({'route': r, 'distance': dist, 'feasible': feasible, 'fill_rate': pack['fill_rate'],
                        'boxes_total': pack['boxes_total'], 'boxes_packed': pack['boxes_packed']})
    # fitness: feasibility-first outside
    if infeasible_count == 0:
        score = total_distance
    else:
        score = penalty_factor * infeasible_count + 1e-3 * total_distance
    return score, {'total_distance': total_distance, 'infeasible_count': infeasible_count, 'routes': details}

# GA operators
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
    best = None; best_s = None
    for _ in range(k):
        i = random.randrange(len(pop))
        s = scores[i]
        if best is None or s < best_s:
            best = pop[i]; best_s = s
    return best

def run_vlr_ga(merged, pop_size=80, gens=200, cx_prob=0.9, mut_prob=0.2, volume_limit_factor=0.95, penalty_factor=1e8, seed=42):
    random.seed(seed)
    inst_name, container, customers, boxes = load_merged(merged)
    cust_ids = [c['customer_id'] for c in customers]
    n = len(cust_ids)
    # initial population: random perms + a seed (natural order)
    pop = [random.sample(cust_ids, n) for _ in range(pop_size-2)]
    pop.append(cust_ids[:])
    pop.append(list(reversed(cust_ids)))
    fitness_cache = {}
    def fitness(ind):
        key = tuple(ind)
        if key in fitness_cache:
            return fitness_cache[key]
        val = evaluate_permutation(merged, ind, volume_limit_factor, penalty_factor)
        fitness_cache[key] = val
        return val
    best = None; best_s = float('inf')
    history = []
    start = time.time()
    for g in range(gens):
        scored = [fitness(ind)[0] for ind in pop]
        # track best
        for i,s in enumerate(scored):
            if s < best_s:
                best_s = s; best = pop[i][:]
        ranked = sorted(zip(scored, pop), key=lambda x: x[0])
        elite_n = max(4, int(0.05*pop_size))
        new_pop = [ranked[i][1][:] for i in range(elite_n)]
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, scored)
            p2 = tournament_select(pop, scored)
            if random.random() < cx_prob:
                child = order_crossover(p1, p2)
            else:
                child = p1[:]
            child = swap_mutation(child, mut_prob)
            new_pop.append(child)
        pop = new_pop
        history.append(best_s)
    duration = time.time() - start
    best_val, best_info = fitness(best)
    return {'best_order': best, 'best_score': best_val, 'best_info': best_info, 'history': history, 'duration': duration}

def save_result(outdir, res):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir,"route_ga_vlr_summary.csv"), "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(['best_score','infeasible_count','total_distance','duration'])
        w.writerow([res['best_score'], res['best_info']['infeasible_count'], res['best_info']['total_distance'], res['duration']])
    with open(os.path.join(outdir,"route_ga_vlr_full.json"), "w") as f:
        json.dump({'best_order': res['best_order'], 'best_score': res['best_score'], 'best_info': res['best_info']}, f, indent=2)
    print("Saved results to", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged', required=True)
    parser.add_argument('--pop', type=int, default=80)
    parser.add_argument('--gens', type=int, default=200)
    parser.add_argument('--alpha', type=float, default=1e8)
    parser.add_argument('--vollimit', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', default='route_ga_vlr_out')
    args = parser.parse_args()

    print("Running VLR-GA on", args.merged)
    res = run_vlr_ga(args.merged, pop_size=args.pop, gens=args.gens, volume_limit_factor=args.vollimit, penalty_factor=args.alpha, seed=args.seed)
    print("Done. best_score:", res['best_score'], "infeasible:", res['best_info']['infeasible_count'], "distance:", res['best_info']['total_distance'], "time_s:", round(res['duration'],2))
    save_result(args.outdir, res)
