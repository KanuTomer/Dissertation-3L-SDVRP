# greedy_baseline.py
from packer import place_boxes_in_container

def greedy_pack(container, boxes, boxes_per_customer_limit=None):
    # Sort boxes by descending volume
    boxes_sorted = sorted(boxes, key=lambda b: b['length']*b['width']*b['height'], reverse=True)
    placements, packed_vol, placed_count = place_boxes_in_container(container, boxes_sorted, max_boxes=boxes_per_customer_limit)
    return {
        'placements': placements,
        'packed_volume': packed_vol,
        'boxes_packed': placed_count
    }
