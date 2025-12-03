# ga_blb.py (patched)
import random, time
from copy import deepcopy
from packer import place_boxes_in_container

def fitness_of_perm(perm, container, boxes_map):
    boxes = [boxes_map[b] for b in perm]
    _, packed_vol, placed_count = place_boxes_in_container(container, boxes)
    return packed_vol, placed_count

def run_ga(container, boxes, pop_size=100, gens=200, cx_prob=0.8, mut_prob=0.2, seed=42):
    random.seed(seed)
    # map
    boxes_map = {b['box_id']: b for b in boxes}
    box_ids = list(boxes_map.keys())

    # create greedy order (desc volume)
    greedy_order = sorted(box_ids, key=lambda b: boxes_map[b]['length']*boxes_map[b]['width']*boxes_map[b]['height'], reverse=True)

    # other heuristics: ascending volume, random-key by volume ratio etc.
    rev_greedy = greedy_order[::-1]
    heuristic_pop = [greedy_order, rev_greedy]

    # initialize random permutations
    pop = []
    # seed some heuristics
    pop.extend(heuristic_pop)
    # fill the rest with random permutations
    while len(pop) < pop_size:
        ind = random.sample(box_ids, len(box_ids))
        pop.append(ind)

    # fitness cache
    fitness_cache = {}
    def fitness(ind):
        key = tuple(ind)
        if key in fitness_cache:
            return fitness_cache[key]
        val = fitness_of_perm(ind, container, boxes_map)
        fitness_cache[key] = val
        return val

    best = None
    best_fit = (-1, -1)
    start = time.time()
    for g in range(gens):
        # evaluate all
        scored = [(fitness(ind), ind) for ind in pop]
        # sort by packed_vol then by placed_count
        scored.sort(key=lambda x: (x[0][0], x[0][1]), reverse=True)
        # update best
        if scored[0][0] > best_fit:
            best_fit = scored[0][0]
            best = scored[0][1][:]
        # elitism: keep top N
        elite_n = max(5, int(0.05*pop_size))
        new_pop = [scored[i][1][:] for i in range(elite_n)]
        # tournament selection
        def tourn_select():
            a,b = random.sample(pop, 2)
            return a if fitness(a) > fitness(b) else b

        # generate rest
        while len(new_pop) < pop_size:
            if random.random() < cx_prob:
                parent1 = tourn_select(); parent2 = tourn_select()
                size = len(parent1)
                a,b = sorted(random.sample(range(size),2))
                child = [-1]*size
                child[a:b+1] = parent1[a:b+1]
                fill = [x for x in parent2 if x not in child]
                idx = 0
                for i in range(size):
                    if child[i] == -1:
                        child[i] = fill[idx]; idx+=1
                child1 = child
            else:
                child1 = tourn_select()[:]
            # mutation
            if random.random() < mut_prob:
                i,j = random.sample(range(len(child1)),2)
                child1[i],child1[j] = child1[j],child1[i]
            new_pop.append(child1)
        pop = new_pop
    duration = time.time()-start
    placements, packed_vol, boxes_packed = place_boxes_in_container(container, [boxes_map[b] for b in best])
    return {
        'best_order': best,
        'placements': placements,
        'packed_volume': packed_vol,
        'boxes_packed': boxes_packed,
        'duration': duration
    }
