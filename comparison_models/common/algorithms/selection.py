# dataset_generation/algorithms/ga/selection.py
import random

def tournament_select(pop, scores, k=3):
    """
    Tournament selection: choose k random individuals and return the best (min score).
    pop: list of individuals
    scores: list of numeric fitness values aligned with pop
    """
    best = None
    best_s = None
    for _ in range(k):
        i = random.randrange(len(pop))
        s = scores[i]
        if best is None or s < best_s:
            best = pop[i]
            best_s = s
    return best
