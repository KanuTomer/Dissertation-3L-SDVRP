#C:\Kanu\Kanu(D)\Dissertation\Dissertation-3L-SDVRP\comparison_models\proposed_model\mutation.py

# dataset_generation/algorithms/ga/mutation.py

import random

def swap_mutation(ind, prob):
    """
    In-place swap mutation with probability prob (returns the mutated individual)
    """
    if random.random() < prob:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]
    return ind

def insertion_mutation(ind, prob):
    if random.random() < prob:
        i, j = random.sample(range(len(ind)), 2)
        val = ind.pop(i)
        ind.insert(j, val)
    return ind


def two_opt_mutation(ind, prob):
    if random.random() < prob:
        i, j = sorted(random.sample(range(len(ind)), 2))
        ind[i:j+1] = reversed(ind[i:j+1])
    return ind


def route_balance_mutation(
    ind,
    prob,
    decoder=None,
    cust_boxcount_map=None,
    max_boxes_per_route=48,
    **_,
):
    if random.random() >= prob or decoder is None or not cust_boxcount_map:
        return ind

    routes = decoder(ind, cust_boxcount_map, max_boxes_per_route=max_boxes_per_route)
    if len(routes) < 2:
        return ind

    indexed_routes = list(enumerate(routes))
    smallest_index, smallest_route = min(
        indexed_routes,
        key=lambda item: (len(item[1]), sum(cust_boxcount_map.get(cid, 0) for cid in item[1])),
    )
    neighbor_choices = [
        item for item in indexed_routes
        if item[0] in {smallest_index - 1, smallest_index + 1}
    ]
    if not smallest_route or not neighbor_choices:
        return ind

    donor_index, donor_route = max(
        neighbor_choices,
        key=lambda item: (len(item[1]), sum(cust_boxcount_map.get(cid, 0) for cid in item[1])),
    )
    if len(donor_route) <= 1:
        return ind

    smallest_boxes = sum(cust_boxcount_map.get(cid, 0) for cid in smallest_route)
    donor_boxes = sum(cust_boxcount_map.get(cid, 0) for cid in donor_route)

    route_size_gap = len(donor_route) - len(smallest_route)
    box_gap = donor_boxes - smallest_boxes

    if route_size_gap < 2 and box_gap < max(6, int(max_boxes_per_route * 0.20)):
        return ind

    candidate_customers = [donor_route[0], donor_route[-1]]

    if len(donor_route) >= 5 and random.random() < 0.25:
        middle_customer = donor_route[len(donor_route) // 2]
        if middle_customer not in candidate_customers:
            candidate_customers.append(middle_customer)

    candidate_customer = random.choice(candidate_customers)

    mutated = [cid for cid in ind if cid != candidate_customer]
    smallest_positions = [idx for idx, cid in enumerate(mutated) if cid in set(smallest_route)]
    if not smallest_positions:
        insert_at = len(mutated)
    elif donor_index < smallest_index:
        insert_at = smallest_positions[0]
    else:
        insert_at = smallest_positions[-1] + 1

    mutated.insert(insert_at, candidate_customer)
    return mutated


def hybrid_mutation(ind, prob, **kwargs):
    enable_route_balance_mutation = bool(kwargs.get("enable_route_balance_mutation", True))
    route_balance_probability = float(kwargs.get("route_balance_probability", 0.08))

    if enable_route_balance_mutation and random.random() < route_balance_probability:
        return route_balance_mutation(ind, prob, **kwargs)

    r = random.random()
    if r < 0.42:
        return swap_mutation(ind, prob)
    elif r < 0.76:
        return insertion_mutation(ind, prob)
    else:
        return two_opt_mutation(ind, prob)
