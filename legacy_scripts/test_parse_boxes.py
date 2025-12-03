# test_parse_boxes.py
from glob import glob
from merge_cvrp_and_boxes import read_box_file

files = glob("../Benchmark dataset and instance generator for Real-World 3dBPP/Input/3dBPP_*.txt")[:5]
print("Found", len(files), "box files to test.")

for f in files:
    boxes = read_box_file(f)
    print(f"\n{f} -> {len(boxes)} boxes")
    print("Sample:", boxes[:3])
