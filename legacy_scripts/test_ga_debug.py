# test_ga_debug.py
from run_experiments import load_instance
from ga_blb import run_ga
from greedy_baseline import greedy_pack
import os, json, csv, random
# pick first merged instance
MF = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
name, container, boxes = load_instance(MF)
print("Instance:", name, "boxes:", len(boxes))
# compute greedy order fitness
box_map = {b['box_id']: b for b in boxes}
greedy_order = sorted(box_map.keys(), key=lambda b: box_map[b]['length']*box_map[b]['width']*box_map[b]['height'], reverse=True)
from ga_blb import fitness_of_perm
fit_g, placed_g = fitness_of_perm(greedy_order, container, box_map)
print("Greedy-order fitness -> packed_vol:", fit_g, "boxes_packed:", placed_g)
# random order fitness
rand = list(box_map.keys())[:]
random.shuffle(rand)
fit_r, placed_r = fitness_of_perm(rand, container, box_map)
print("Random-order fitness -> packed_vol:", fit_r, "boxes_packed:", placed_r)
# run GA with small settings to get quick result
res = run_ga(container, boxes, pop_size=30, gens=20, seed=42)
print("GA final -> packed_vol:", res['packed_volume'], "boxes_packed:", res['boxes_packed'], "time:", res['duration'])
