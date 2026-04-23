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
