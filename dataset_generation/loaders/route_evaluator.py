import json
from pathlib import Path
from typing import Tuple, Dict, List, Any

def load_merged(merged_json_path: str) -> Tuple[str, dict, list, list]:
    p = Path(merged_json_path)
    if not p.exists():
        raise FileNotFoundError(f"Merged JSON not found: {p.resolve()}")
    with p.open("r", encoding="utf-8-sig") as f:
        try:
            d = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse merged JSON {p.resolve()}: {e}") from e

    inst_name = d.get("inst_name") or d.get("name") or p.stem
    container = d.get("container", {})
    customers = d.get("customers", [])
    boxes = d.get("boxes", [])
    return inst_name, container, customers, boxes

# Import the packer we placed in dataset_generation/utils/packer.py
try:
    from dataset_generation.utils.packer import place_boxes_in_container as _place_boxes_in_container
    _HAS_PACKER = True
except Exception:
    _HAS_PACKER = False

def _normalize_container_for_packer(container: dict) -> dict:
    """
    Return a dict with keys 'L','W','H' (numbers) for the packer.
    Tries several likely key names in order.
    """
    # helper to pick numeric value from several keys
    def pick(keys, default=None):
        for k in keys:
            if k in container and container[k] is not None:
                # if container[k] is a dict/str etc, try to coerce to float if possible
                v = container[k]
                try:
                    return float(v)
                except Exception:
                    # leave as-is for now
                    return v
        return default

    L = pick(["L","length","l","Length","len","width","Width"])
    W = pick(["W","width","w","Width","width","depth","Depth","height"])
    H = pick(["H","height","h","Height","depth","Depth"])
    # If any value still missing, set a conservative positive default 1.0
    try:
        L = float(L) if L is not None else 1.0
    except Exception:
        L = 1.0
    try:
        W = float(W) if W is not None else 1.0
    except Exception:
        W = 1.0
    try:
        H = float(H) if H is not None else 1.0
    except Exception:
        H = 1.0

    return {"L": L, "W": W, "H": H}

def evaluate_route(merged_json_path: str, route: List[int]) -> Dict:
    """
    Prepare boxes for the route and call the packer with a normalized container dict.
    """
    try:
        inst_name, container, customers, boxes = load_merged(merged_json_path)
    except Exception:
        return {"feasible": False, "boxes_total": 0, "boxes_packed": 0, "fill_rate": 0.0}

    # Build mapping of box_id -> box dict for lookup
    box_map = {b.get("box_id"): b for b in boxes}

    # Collect boxes for this route by customer assigned_boxes (preserve order)
    boxes_for_route = []
    for cid in route:
        c = next((x for x in customers if x.get("customer_id") == cid), None)
        if c:
            for bid in c.get("assigned_boxes", []):
                b = box_map.get(bid)
                if b:
                    boxes_for_route.append(b)

    # If none, fall back to all boxes (compatibility)
    if not boxes_for_route:
        boxes_for_route = boxes.copy()

    boxes_total = len(boxes_for_route)

    # If packer available, call it using normalized container
    if _HAS_PACKER:
        try:
            packer_container = _normalize_container_for_packer(container)
            placements, packed_vol, placed_count = _place_boxes_in_container(packer_container, boxes_for_route)
            # compute container volume from normalized container
            container_vol = float(packer_container.get("L",1.0)) * float(packer_container.get("W",1.0)) * float(packer_container.get("H",1.0))
            fill_rate = float(packed_vol) / container_vol if container_vol > 0 else 0.0
            feasible = (placed_count == boxes_total) or (boxes_total == 0)
            return {
                "feasible": bool(feasible),
                "boxes_total": int(boxes_total),
                "boxes_packed": int(placed_count),
                "fill_rate": float(fill_rate),
            }
        except Exception as e:
            print(f"[route_evaluator] packer invocation failed: {e}")

    # fallback heuristic
    packed = min(boxes_total, max(0, int(len(route) * 0.5)))
    fill_rate = (packed / boxes_total) if boxes_total > 0 else 0.0
    feasible = True if boxes_total == 0 else (fill_rate >= 0.0)
    return {"feasible": bool(feasible), "boxes_total": boxes_total, "boxes_packed": packed, "fill_rate": fill_rate}
