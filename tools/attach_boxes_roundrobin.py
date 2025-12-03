# tools/attach_boxes_roundrobin.py
import csv, json, sys
from pathlib import Path
from math import hypot

def load_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8-sig"))

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_boxes_csv(csv_path):
    rows = []
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return r.fieldnames, rows

def attach_round_robin(merged_path):
    merged = load_json(merged_path)
    base = Path(merged_path).parent
    boxes_file = merged.get("boxes_file")
    if not boxes_file:
        raise SystemExit("merged JSON has no 'boxes_file' key; please supply the boxes CSV path")

    csv_path = (base / boxes_file)
    if not csv_path.exists():
        raise SystemExit(f"boxes CSV not found: {csv_path}")

    headers, boxes = load_boxes_csv(csv_path)
    # normalise box dict keys to expected names
    normalized = []
    for b in boxes:
        # attempt to find id field
        if "box_id" in b:
            box_id = b["box_id"]
        elif "id" in b:
            box_id = b["id"]
        else:
            # fallback to first column
            box_id = next(iter(b.values()))
        out = {
            "box_id": box_id,
            # attempt numeric conversion for dims
            "length": float(b.get("length", b.get("L", 0)) or 0),
            "width":  float(b.get("width", b.get("W", 0)) or 0),
            "height": float(b.get("height", b.get("H", 0)) or 0)
        }
        # keep original row for later reference
        out["_orig_row"] = b
        normalized.append(out)

    customers = merged.get("customers", [])
    if not customers:
        raise SystemExit("merged JSON has no customers")

    # if boxes already include a customer id column, attach accordingly
    sample_row = boxes[0] if boxes else {}
    possible_customer_fields = [k for k in sample_row.keys() if "cust" in k.lower() or "customer" in k.lower() or "cid"==k.lower()]
    if possible_customer_fields:
        fld = possible_customer_fields[0]
        # attach based on that field
        for box in normalized:
            owner = box["_orig_row"].get(fld)
            if owner is None or owner == "":
                continue
            # try convert to int index; else match customer_id
            try:
                cid = int(owner)
                # find customer with that id or id field name
                for c in customers:
                    if c.get("customer_id") == cid or c.get("id") == cid:
                        c.setdefault("assigned_boxes", []).append(box["box_id"])
                        break
            except Exception:
                # try match string
                for c in customers:
                    if str(c.get("customer_id","")) == str(owner) or str(c.get("id","")) == str(owner):
                        c.setdefault("assigned_boxes", []).append(box["box_id"])
                        break
        attached = sum(len(c.get("assigned_boxes",[])) for c in customers)
        print(f"Attached {attached} boxes using column {fld}")
    else:
        # round-robin attach
        cust_ids = [c.get("customer_id") or c.get("id") for c in customers]
        if not all(cust_ids):
            # fallback: index-based customers
            cust_ids = list(range(len(customers)))
        i = 0
        for box in normalized:
            cid = cust_ids[i % len(cust_ids)]
            # find customer object and append
            for c in customers:
                if c.get("customer_id") == cid or c.get("id") == cid:
                    c.setdefault("assigned_boxes", []).append(box["box_id"])
                    break
            i += 1
        attached = len(normalized)
        print(f"Round-robin attached {attached} boxes to {len(customers)} customers")

    # write out a new merged file (back up original first)
    p = Path(merged_path)
    backup = p.with_suffix(p.suffix + ".bak")
    if not backup.exists():
        p.replace(backup)  # move original to .bak
        merged_out_name = p  # we'll write new merged with original name
        print(f"Backed up original merged JSON to: {backup}")
    else:
        merged_out_name = p

    # If we moved original to .bak, we need to write new file at original path
    write_json(merged_out_name, merged)
    out_path = str(merged_out_name)
    print("Wrote merged-with-boxes:", out_path)
    return out_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/attach_boxes_roundrobin.py <path/to/merged.json>")
        sys.exit(1)
    attach_round_robin(sys.argv[1])
