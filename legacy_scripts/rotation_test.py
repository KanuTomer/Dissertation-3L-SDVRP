from run_experiments import load_instance
from packer import place_boxes_in_container

MF = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"

name, container, boxes = load_instance(MF)

# Use only the first 20 boxes to quickly see rotations
small_boxes = boxes[:20]

placements, vol, cnt = place_boxes_in_container(container, small_boxes)

print("Placed:", cnt)
print("Total Volume:", vol)

# Check if any box was rotated:
rot_count = 0
for p in placements:
    b = next(bx for bx in small_boxes if bx['box_id'] == p['box_id'])
    if not (
        (p['l'] == b['length'] and p['w'] == b['width'] and p['h'] == b['height'])
    ):
        rot_count += 1

print("Rotated boxes:", rot_count)
