#!/usr/bin/env python3
# tools/merge_boxes_into_merged.py
# Usage: python tools/merge_boxes_into_merged.py "<merged_json_path>"
# Produces: <same-dir>/<base>_merged_with_boxes.json

import sys, json, csv, pathlib

PREFERRED_BOX_ID = ("box_id","id","boxid","bid","sku")
PREFERRED_DIMS = {
    "length": ("length","L","l","len"),
    "width": ("width","W","w","wid","width_mm"),
    "height": ("height","H","h","ht","height_mm")
}
PREFERRED_CUST_REF = ("customer_id","cust_id","customer","cid","client_id")

def find_field(names, rowkeys):
    for n in names:
        if n in rowkeys:
            return n
    # case-insensitive fallback
    lowered = {k.lower():k for k in rowkeys}
    for n in names:
        if n.lower() in lowered:
            return lowered[n.lower()]
    return None

def normalize_box_row(row):
    keys = list(row.keys())
    # id
    boxid_field = find_field(PREFERRED_BOX_ID, keys)
    box_id = row.get(boxid_field) if boxid_field else None
    # dims
    dims = {}
    for dim, names in PREFERRED_DIMS.items():
        k = find_field(names, keys)
        v = row.get(k) if k else None
        try:
            dims[dim] = float(v) if v not in (None,"") else 0.0
        except Exception:
            dims[dim] = 0.0
    return {"box_id": box_id, "length": dims["length"], "width": dims["width"], "height": dims["height"], "raw": row}

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/merge_boxes_into_merged.py <merged_json_path>")
        sys.exit(1)
    p = pathlib.Path(sys.argv[1]).resolve()
    if not p.exists():
        print("File not found:", p)
        sys.exit(2)

    j = json.loads(p.read_text(encoding="utf-8-sig"))
    # Normalise customers: id -> customer_id and ensure assigned_boxes key
    customers = j.get("customers", [])
    for c in customers:
        if "customer_id" not in c and "id" in c:
            c["customer_id"] = c["id"]
        if "assigned_boxes" not in c:
            c["assigned_boxes"] = []

    # Determine boxes CSV path
    boxes_file = j.get("boxes_file") or j.get("boxesfile") or j.get("boxesFilename")
    boxes = []
    if boxes_file:
        boxes_path = (p.parent / boxes_file).resolve()
        if not boxes_path.exists():
            print("Warning: boxes_file referenced but not found:", boxes_path)
        else:
            with boxes_path.open("r", encoding="utf-8-sig", newline='') as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
                for row in rows:
                    b = normalize_box_row(row)
                    boxes.append(b)
                # Try to attach boxes to customers if CSV has a customer reference col
                if rows:
                    keys = rows[0].keys()
                    custref = find_field(PREFERRED_CUST_REF, keys)
                    boxid_field = find_field(PREFERRED_BOX_ID, keys)
                    if custref and boxid_field:
                        cmap = {}
                        for row in rows:
                            box_id = row.get(boxid_field)
                            custval = row.get(custref)
                            if box_id is None:
                                continue
                            try:
                                custint = int(custval)
                            except Exception:
                                custint = custval
                            cmap.setdefault(custint, []).append(box_id)
                        for c in customers:
                            cid = c.get("customer_id") or c.get("id")
                            if cid in cmap:
                                c["assigned_boxes"] = cmap[cid]
    else:
        print("No boxes_file key found in merged JSON; writing boxes=[]")

    boxes_out = [{"box_id": b["box_id"], "length": b["length"], "width": b["width"], "height": b["height"]} for b in boxes]

    out = dict(j)
    out["customers"] = customers
    out["boxes"] = boxes_out

    out_path = p.parent / (p.stem + "_with_boxes.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote merged-with-boxes:", out_path)

if __name__ == "__main__":
    main()
