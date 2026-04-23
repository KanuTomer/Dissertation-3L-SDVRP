# dataset_generation/algorithms/ga/crossover.py
import random

def order_crossover(a, b):
    """
    Order-based crossover (kept from legacy code).
    a, b: parent permutations (lists)
    returns: child permutation
    """
    n = len(a)
    i, j = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[i:j+1] = a[i:j+1]
    fill = [x for x in b if x not in child]
    idx = 0
    for k in range(n):
        if child[k] == -1:
            child[k] = fill[idx]
            idx += 1
    return child
