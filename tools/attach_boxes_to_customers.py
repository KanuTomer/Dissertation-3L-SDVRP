#!/usr/bin/env python3
# tools/attach_boxes_to_customers.py
# Usage: python tools/attach_boxes_to_customers.py "<merged_with_boxes.json>"
import sys, json, csv
from pathlib import Path

PREFERRED_BOX_ID = ("box_id","id","boxid","bid","sku")
PREFERRED_CUST_REF = ("customer_id","cust_id","customer","cid","client_id","cust")

def find_field(names, rowkeys):
    for n in names:
        if n in rowkeys:
            return n
    lowered = {k.lower():k for k in rowkeys}
    for n in names:
        if n.lower() in lowered:
            return lowered[n.lower()]
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python attach_boxes_to_customers.py <merged_with_boxes.json>")
        sys.exit(1)
    merged_path = Path(sys.argv[1]).resolve()
    if not merged_path.exists():
        print("File not found:", merged_path); sys.exit(2)
    j = json.loads(merged_path.read_text(encoding="utf-8-sig"))

    boxes_file = j.get("boxes_file") or j.get("boxesfile") or j.get("boxesFilename")
    if not boxes_file:
        print("No 'boxes_file' key found in merged JSON. Trying to use 'boxes' array if present.")
    boxes_csv_path = (merged_path.parent / boxes_file).resolve() if boxes_file else None

    # load CSV rows if available
    csv_rows = []
    if boxes_csv_path and boxes_csv_path.exists():
        with boxes_csv_path.open("r", encoding="utf-8-sig", newline='') as fh:
            reader = csv.DictReader(fh)
            csv_rows = list(reader)
        print(f"Loaded {len(csv_rows)} rows from CSV: {boxes_csv_path}")
    else:
        print("Boxes CSV not found or not referenced. Will attempt to use 'boxes' array in JSON.")
    # create box map from JSON boxes array if present
    boxes_json = j.get("boxes", [])
    box_map = {}
    for b in boxes_json:
        bid = b.get("box_id") or b.get("id") or b.get("boxid")
        if bid is not None:
            box_map[str(bid)] = b

    # prepare customers map (by id or customer_id)
    customers = j.get("customers", [])
    cust_map = {}
    for c in customers:
        cid = c.get("customer_id") or c.get("id") or c.get("cust_id")
        if cid is not None:
            cust_map[str(cid)] = c
        # ensure assigned_boxes exists
        if "assigned_boxes" not in c:
            c["assigned_boxes"] = []

    attached = 0
    # If CSV rows exist, try to attach using a customer ref column
    if csv_rows:
        keys = csv_rows[0].keys()
        boxid_field = find_field(PREFERRED_BOX_ID, keys)
        cust_field = find_field(PREFERRED_CUST_REF, keys)
        if not boxid_field:
            print("Could not identify a box-id column in CSV; headers:", keys)
        else:
            print("Detected box-id field in CSV:", boxid_field)
        if not cust_field:
            print("Could not identify a customer-ref column in CSV; headers:", keys)
        else:
            print("Detected customer-ref field in CSV:", cust_field)

        if boxid_field and cust_field:
            for row in csv_rows:
                bid = row.get(boxid_field)
                custval = row.get(cust_field)
                if bid is None or custval is None or custval=="":
                    continue
                # normalize ids to string keys
                key = str(custval)
                # find customer in cust_map; try numeric and string
                target = None
                if key in cust_map:
                    target = cust_map[key]
                else:
                    # try numeric form
                    try:
                        if str(int(custval)) in cust_map:
                            target = cust_map[str(int(custval))]
                    except Exception:
                        pass
                if target is None:
                    # try match by 'id' numeric vs whatever
                    # skip if no match
                    continue
                # append box id to assigned_boxes if not present
                if bid not in target.get("assigned_boxes", []):
                    target.setdefault("assigned_boxes", []).append(bid)
                    attached += 1

    # If CSV absent or did not attach anything, try to attach by heuristics from boxes_json:
    if attached == 0 and boxes_json:
        # heuristic: if boxes_json items contain a cust/ref field (customer_id, cust_id, owner)
        possible_cust_fields = ("customer_id","cust_id","owner","cust","customer")
        for b in boxes_json:
            bid = b.get("box_id") or b.get("id") or b.get("boxid")
            for f in possible_cust_fields:
                if f in b and b[f] not in (None, ""):
                    cid = str(b[f])
                    if cid in cust_map:
                        if bid not in cust_map[cid].get("assigned_boxes", []):
                            cust_map[cid].setdefault("assigned_boxes", []).append(bid)
                            attached += 1
        if attached>0:
            print("Attached boxes using data from JSON 'boxes' entries using heuristic customer fields.")

    # Final counts
    num_customers_with_boxes = sum(1 for c in customers if c.get("assigned_boxes"))
    print("Attached boxes count (added):", attached)
    print("Customers with assigned_boxes now:", num_customers_with_boxes, "of", len(customers))

    # write out a new file next to input (safe)
    out_path = merged_path.parent / (merged_path.stem + "_attached.json")
    out_path.write_text(json.dumps(j, indent=2), encoding="utf-8")
    print("Wrote attached merged file ->", out_path)
    print("Done.")
if __name__ == '__main__':
    main()
