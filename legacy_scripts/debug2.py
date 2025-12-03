from run_experiments import load_instance
from packer import place_boxes_in_container

name, container, boxes = load_instance(
    "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged/XML100_1111_01_merged.json"
)

# Test packing only the first 5 boxes
test_boxes = boxes[:5]
placements, vol, cnt = place_boxes_in_container(container, test_boxes)
print("Test boxes:", test_boxes)
print("placements:", placements)
print("packed_volume:", vol, "boxes_packed:", cnt)
