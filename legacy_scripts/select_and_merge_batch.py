import os, csv
from glob import glob
from merge_cvrp_and_boxes import parse_cvrp_xml, merge_instance

# Buckets: small, medium, large
BUCKETS = [
    (10, 25),   # small
    (40, 60),   # medium
    (80, 120)   # large
]

def count_customers(path):
    try:
        data = parse_cvrp_xml(path)
        return len(data["customers"])
    except:
        return None

def select_instances(vrp_dir, max_per_bucket=3):
    matches = glob(os.path.join(vrp_dir, "*.vrp"))
    selected = {0: [], 1: [], 2: []}

    for file in matches:
        cnt = count_customers(file)
        if cnt is None:
            continue

        for i, (mn, mx) in enumerate(BUCKETS):
            if mn <= cnt <= mx and len(selected[i]) < max_per_bucket:
                selected[i].append((file, cnt))
                break

        if all(len(v) == max_per_bucket for v in selected.values()):
            break

    return selected

if __name__ == "__main__":
    vrp_dir = "../XML"
    box_dir = "../Benchmark dataset and instance generator for Real-World 3dBPP/Input"
    outdir = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged"

    os.makedirs(outdir, exist_ok=True)

    from os.path import isfile
    box_files = [f for f in glob(os.path.join(box_dir, "*")) if isfile(f)]


    selected = select_instances(vrp_dir)

    manifest = []

    for bucket_idx, files in selected.items():
        for (vrp_file, count) in files:
            print(f"Merging: {vrp_file}  | customers = {count}")
            data = parse_cvrp_xml(vrp_file)
            merge_instance(data, box_files, outdir, boxes_per_customer=6, seed=42)

            manifest.append([os.path.basename(vrp_file), count, bucket_idx])

    with open(os.path.join(outdir, "manifest.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["vrp_file", "customers", "bucket"])
        writer.writerows(manifest)

    print("✔ Done. Manifest written.")
