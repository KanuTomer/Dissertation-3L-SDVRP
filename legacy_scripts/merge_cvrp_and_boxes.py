import os, json, csv, random, re
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

# ---------------------------
# 1. Parse CVRPLIB / VRP files
# ---------------------------
def parse_cvrp_xml(path):
    """
    Parses VRP or XML-style CVRPLIB files and extracts:
    depot, customers, coordinates, demand (if any)
    """
    text = open(path, 'r', encoding='utf-8', errors='ignore').read()

    # Try to parse TSPLIB-like format
    if "NODE_COORD_SECTION" in text:
        customers = []
        coords_started = False

        lines = text.splitlines()
        for ln in lines:
            ln = ln.strip()
            if ln.startswith("NODE_COORD_SECTION"):
                coords_started = True
                continue
            if coords_started:
                if ln.startswith("DEMAND_SECTION") or ln.startswith("DEPOT_SECTION"):
                    break
                parts = ln.split()
                if len(parts) >= 3:
                    cid = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    customers.append({"id": cid, "x": x, "y": y, "demand": None})
        depot = (customers[0]["x"], customers[0]["y"]) if customers else (0, 0)
        return {"name": Path(path).stem, "depot": depot, "customers": customers}

    # Try XML-like structure
    try:
        root = ET.parse(path).getroot()
        customers = []
        depot = None

        for node in root.findall(".//node"):
            cid = node.get("id") or node.findtext("id")
            x = node.findtext("x") or node.findtext("coordX")
            y = node.findtext("y") or node.findtext("coordY")
            demand = node.findtext("demand") or None
            if cid and x and y:
                customers.append({
                    "id": int(cid),
                    "x": float(x),
                    "y": float(y),
                    "demand": float(demand) if demand else None,
                })

        depot = (customers[0]["x"], customers[0]["y"]) if customers else (0, 0)
        return {"name": Path(path).stem, "depot": depot, "customers": customers}

    except:
        raise RuntimeError(f"Unsupported VRP file format: {path}")


# ---------------------------
# 2. Parse 3D Box Files
# ---------------------------
def read_box_file(path):
    """
    Accepts CSV, JSON, or TXT (3dBPP formats)
    Returns list of boxes: {id, l, w, h, weight}
    """
    boxes = []
    path = str(path)

    # CSV
    if path.lower().endswith('.csv'):
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                boxes.append({
                    "id": int(r.get("box_id") or len(boxes)+1),
                    "l": float(r.get("length")),
                    "w": float(r.get("width")),
                    "h": float(r.get("height")),
                    "weight": float(r.get("weight")) if r.get("weight") else None
                })
        return boxes

    # JSON
    if path.lower().endswith('.json'):
        data = json.load(open(path))
        for r in data:
            boxes.append({
                "id": int(r.get("box_id") or len(boxes)+1),
                "l": float(r["l"]),
                "w": float(r["w"]),
                "h": float(r["h"]),
                "weight": float(r.get("weight", 0))
            })
        return boxes

    # TXT (3dBPP_#.txt)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for ln in lines:
        # extract numbers on each line
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", ln)
        if len(nums) >= 3:
            l, w, h = float(nums[-3]), float(nums[-2]), float(nums[-1])
            boxes.append({
                "id": len(boxes)+1,
                "l": l,
                "w": w,
                "h": h,
                "weight": None
            })

    return boxes


# ---------------------------
# 3. Merge VRP customers with box data
# ---------------------------
def merge_instance(cvrp, box_files, outdir, boxes_per_customer=3, seed=42):
    random.seed(seed)
    os.makedirs(outdir, exist_ok=True)

    # Load all boxes
    all_boxes = []
    for bf in box_files:
        all_boxes.extend(read_box_file(bf))

    # Shuffle
    random.shuffle(all_boxes)

    # Assign boxes to customers
    assignments = {}
    idx = 0
    customers = cvrp["customers"]

    for cust in customers:
        assigned = []
        for _ in range(boxes_per_customer):
            assigned.append(all_boxes[idx % len(all_boxes)]["id"])
            idx += 1
        assignments[cust["id"]] = assigned

    # Write customers file
    cust_path = os.path.join(outdir, f"{cvrp['name']}_customers.csv")
    with open(cust_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["customer_id", "x", "y", "demand", "assigned_boxes"])
        for c in customers:
            writer.writerow([c["id"], c["x"], c["y"], c["demand"], ";".join(map(str, assignments[c["id"]]))])

    # Write boxes file
    box_path = os.path.join(outdir, f"{cvrp['name']}_boxes.csv")
    with open(box_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["box_id", "length", "width", "height", "weight"])
        for b in all_boxes:
            writer.writerow([b["id"], b["l"], b["w"], b["h"], b["weight"]])

    # Write merged JSON
    merged_path = os.path.join(outdir, f"{cvrp['name']}_merged.json")
    json.dump({
        "instance_name": cvrp["name"],
        "depot": cvrp["depot"],
        "customers": customers,
        "boxes_file": os.path.basename(box_path),
        # Standard 40-foot trailer interior dimensions (in mm)
        "container": { "L": 4000, "W": 1800, "H": 1800 }  # small van (mm)

    }, open(merged_path, "w"), indent=2)

    print(f"Saved merged instance → {merged_path}")
