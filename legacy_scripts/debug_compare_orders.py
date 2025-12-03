# debug_compare_orders.py
from run_experiments import load_instance
from greedy_baseline import greedy_pack
from ga_blb import fitness_of_perm
import math

mf = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
name, container, boxes = load_instance(mf)
print("Instance:", name, "num boxes:", len(boxes))

# greedy_pack list
boxes_sorted = sorted(boxes, key=lambda b: b['length']*b['width']*b['height'], reverse=True)
print("\nFirst 10 boxes in boxes_sorted (id, l,w,h, vol):")
for b in boxes_sorted[:10]:
    print(b['box_id'], b['length'], b['width'], b['height'], int(b['length']*b['width']*b['height']))

total_vol_sorted = sum(b['length']*b['width']*b['height'] for b in boxes_sorted)
print("Total volume (all boxes_sorted):", int(total_vol_sorted))

# greedy_order by IDs (what GA uses)
box_map = {b['box_id']: b for b in boxes}
greedy_order_ids = sorted(box_map.keys(), key=lambda bid: box_map[bid]['length']*box_map[bid]['width']*box_map[bid]['height'], reverse=True)
print("\nFirst 10 IDs in greedy_order_ids:", greedy_order_ids[:10])

# construct boxes list from greedy_order_ids
boxes_from_ids = [box_map[bid] for bid in greedy_order_ids]
print("\nFirst 10 boxes_from_ids (id, l,w,h, vol):")
for b in boxes_from_ids[:10]:
    print(b['box_id'], b['length'], b['width'], b['height'], int(b['length']*b['width']*b['height']))

total_vol_from_ids = sum(b['length']*b['width']*b['height'] for b in boxes_from_ids)
print("Total volume (from ids):", int(total_vol_from_ids))

# Compare lists element-wise (first 20)
print("\nComparing first 20 elements (boxes_sorted vs boxes_from_ids) by box_id:")
for i in range(20):
    a = boxes_sorted[i]['box_id']
    b = boxes_from_ids[i]['box_id']
    print(i, "sorted_id:", a, "from_ids_id:", b, "same?", a==b)

# Evaluate fitness_of_perm on greedy_order_ids
fit, placed = fitness_of_perm(greedy_order_ids, container, box_map)
print("\nfitness_of_perm on greedy_order_ids -> packed_vol:", fit, "boxes_packed:", placed)
