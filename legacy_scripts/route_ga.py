# route_ga.py
import argparse, random, time, csv, os
from math import sqrt
from copy import deepcopy
from route_evaluator import load_merged, evaluate_route
import json

def euclid(a,b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def route_distance(depot, customers_map, route):
    # route: list of customer ids
    if not route:
        return 0.0
    dist = 0.0
    # depot -> first
    first = customers_map[route[0]]
    dist += euclid((depot[0], depot[1]), (first['x'], first['y']))
    # between customers
    for i in range(len(route)-1):
        c1 = customers_map[route[i]]
        c2 = customers_map[route[i+1]]
        dist += euclid((c1['x'], c1['y']), (c2['x'], c2['y']))
    # last -> depot
    last = customers_map[route[-1]]
    dist += euclid((last['x'], last['y']), (depot[0], depot[1]))
    return dist

def decode_into_routes(order, route_size):
    # deterministic chunking
    return [order[i:i+route_size] for i in range(0, len(order), route_size)]

def evaluate_chromosome(merged_json_path, order, route_size, penalty_factor):
    """
    Returns: (fitness, details)
    fitness lower is better
    details contains per-route metrics for reporting
    """
    # load structured data
    inst_name, container, customers, boxes = load_merged(merged_json_path)
    depot = container.get('depot', (0,0))  # not stored, but load_merged doesn't return depot; we will use customers[0] as depot fallback
    # Actually load_merged returns customers with x,y; depot not included by function; use first customer as depot? 
    # We'll derive depot as the first customer in the customers list if needed:
    if 'depot' in container:
        depot_coord = container['depot']
    else:
        # fallback: compute centroid as pseudo-depot (or 0,0)
        depot_coord = (0,0)

    # make a quick customers map from route_evaluator load_merged
    customers_map = {c['customer_id']: c for c in customers}

    routes = decode_into_routes(order, route_size)
    total_distance = 0.0
    infeasible_count = 0
    details = []
    # Evaluate each route: distance + packing
    for rt in routes:
        if not rt:
            continue
        # compute route distance using customers_map (if depot unknown use (0,0))
        if depot_coord==(0,0) and customers:
            # attempt best guess: use first customer's coordinates as depot fallback
            depot_coord = (customers[0]['x'], customers[0]['y'])
        dist = route_distance(depot_coord, customers_map, rt)
        # packing evaluation via existing function (uses merged JSON path)
        # evaluate_route expects merged_json_path and list of customer ids
        pack_res = evaluate_route(merged_json_path, rt)
        feasible = pack_res['feasible']
        fill_rate = pack_res['fill_rate']
        if not feasible:
            infeasible_count += 1
        total_distance += dist
        details.append({'route': rt, 'distance': dist, 'feasible': feasible, 'fill_rate': fill_rate,
                        'boxes_total': pack_res['boxes_total'], 'boxes_packed': pack_res['boxes_packed']})
    fitness = total_distance + penalty_factor * infeasible_count
    return fitness, {'total_distance': total_distance, 'infeasible_count': infeasible_count, 'routes': details}

# GA helpers
def tournament_selection(pop, scores, k=3):
    best = None; best_score = None
    for _ in range(k):
        i = random.randrange(len(pop))
        s = scores[i]
        if best is None or s < best_score:
            best = pop[i]; best_score = s
    return best

def order_crossover(a,b):
    n = len(a)
    i,j = sorted(random.sample(range(n),2))
    child = [-1]*n
    child[i:j+1] = a[i:j+1]
    fill = [x for x in b if x not in child]
    idx = 0
    for k in range(n):
        if child[k]==-1:
            child[k] = fill[idx]; idx+=1
    return child

def swap_mutation(ind, prob=0.05):
    if random.random() < prob:
        i,j = random.sample(range(len(ind)),2)
        ind[i],ind[j] = ind[j],ind[i]
    return ind

def run_route_ga(merged_json_path, route_size, pop_size=60, gens=200, cx_prob=0.9, mut_prob=0.2, elitism=4, penalty_factor=1e7, seed=42):
    random.seed(seed)
    # load customers list to get ids
    inst_name, container, customers, boxes = load_merged(merged_json_path)
    cust_ids = [c['customer_id'] for c in customers]
    n = len(cust_ids)
    # initial population: include greedy route (nearest-neighbour / simple order) and random perms
    greedy_order = cust_ids[:]  # baseline: as-is (or we could build NN); keep simple
    pop = [random.sample(cust_ids, n) for _ in range(pop_size-2)]
    pop.append(greedy_order)
    pop.append(greedy_order[::-1])
    best = None; best_score = float('inf')
    history = []
    # precompute cached fitness dictionary
    fitness_cache = {}
    def fitness(ind):
        key = tuple(ind)
        if key in fitness_cache:
            return fitness_cache[key]
        f_raw, info = evaluate_chromosome(merged_json_path, ind, route_size, penalty_factor)
        # feasibility-first scoring:
        if info['infeasible_count'] == 0:
            # fully feasible: score = pure total distance (lower is better)
            score = info['total_distance']
        else:
            # infeasible: heavy penalty multiplied by infeasible routes,
            # plus a tiny component of distance to break ties among infeasible solutions
            score = penalty_factor * info['infeasible_count'] + 1e-3 * info['total_distance']
        fitness_cache[key] = (score, info)
        return fitness_cache[key]

    start = time.time()
    for g in range(gens):
        scores = [fitness(ind)[0] for ind in pop]
        # track best
        for i,s in enumerate(scores):
            if s < best_score:
                best_score = s; best = pop[i][:]
        # sort by score
        ranked = sorted(zip(scores, pop), key=lambda x: x[0])
        new_pop = [ranked[i][1][:] for i in range(min(elitism, len(ranked)))]
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, scores)
            p2 = tournament_selection(pop, scores)
            if random.random() < cx_prob:
                c = order_crossover(p1, p2)
            else:
                c = p1[:]
            c = swap_mutation(c, mut_prob)
            new_pop.append(c)
        pop = new_pop
        history.append(best_score)
    dur = time.time() - start
    # final best info
    best_f, best_info = fitness(best)
    return {'best_order': best, 'best_score': best_f, 'best_info': best_info, 'history': history, 'duration': dur}

def save_result(outdir, res):
    os.makedirs(outdir, exist_ok=True)
    csvpath = os.path.join(outdir, "route_ga_summary.csv")
    with open(csvpath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['best_score','infeasible_count','total_distance','duration'])
        w.writerow([res['best_score'], res['best_info']['infeasible_count'], res['best_info']['total_distance'], res['duration']])
    jsonpath = os.path.join(outdir, "route_ga_full.json")
    with open(jsonpath, 'w') as f:
        json.dump({
            'best_order': res['best_order'],
            'best_score': res['best_score'],
            'best_info': res['best_info'],
            'history': res.get('history',[])
        }, f, indent=2)
    print("Saved summary to", csvpath)
    print("Saved full result to", jsonpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged', required=True, help='merged json path')
    parser.add_argument('--route_size', type=int, default=8)
    parser.add_argument('--pop', type=int, default=60)
    parser.add_argument('--gens', type=int, default=120)
    parser.add_argument('--alpha', type=float, default=1e7, help='penalty per infeasible route')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', default='route_ga_out')
    args = parser.parse_args()

    print("Running combined GA on", args.merged)
    res = run_route_ga(args.merged, args.route_size, pop_size=args.pop, gens=args.gens, penalty_factor=args.alpha, seed=args.seed)
    print("Done. Best score:", res['best_score'], "infeasible routes:", res['best_info']['infeasible_count'],
          "total_distance:", res['best_info']['total_distance'], "time_s:", round(res['duration'],2))
    save_result(args.outdir, res)
