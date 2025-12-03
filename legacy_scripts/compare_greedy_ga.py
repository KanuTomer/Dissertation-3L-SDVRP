# compare_greedy_ga.py
from run_experiments import load_instance
from greedy_baseline import greedy_pack
from ga_blb import fitness_of_perm, run_ga
import time

MF = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
name, container, boxes = load_instance(MF)
print("Instance:", name, "num boxes:", len(boxes))

# greedy baseline (calls packer on sorted boxes)
res_g = greedy_pack(container, boxes)
print("greedy_pack -> packed_volume:", res_g['packed_volume'], "boxes_packed:", res_g['boxes_packed'])

# construct greedy order (box_ids sorted by descending volume)
box_map = {b['box_id']: b for b in boxes}
greedy_order = sorted(box_map.keys(), key=lambda b: box_map[b]['length']*box_map[b]['width']*box_map[b]['height'], reverse=True)

# evaluate greedy_order with GA fitness function
fit, placed = fitness_of_perm(greedy_order, container, box_map)
print("fitness_of_perm on greedy_order -> packed_volume:", fit, "boxes_packed:", placed)

# evaluate a random order for contrast
import random
rand = greedy_order[:]
random.shuffle(rand)
fit_r, placed_r = fitness_of_perm(rand, container, box_map)
print("fitness_of_perm on random_order -> packed_volume:", fit_r, "boxes_packed:", placed_r)

# run a short GA seeded with heuristics (quick)
t0 = time.time()
res = run_ga(container, boxes, pop_size=30, gens=40, seed=42)
t1 = time.time()
print("run_ga (short) -> packed_volume:", res['packed_volume'], "boxes_packed:", res['boxes_packed'], "time:", round(t1-t0,3))
