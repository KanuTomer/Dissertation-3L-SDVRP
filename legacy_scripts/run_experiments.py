# run_experiments.py
import glob, json, csv, os, time
import pandas as pd
from greedy_baseline import greedy_pack
from ga_blb import run_ga

OUT_DIR = "../Benchmark dataset and instance generator for Real-World 3dBPP/Output/merged"
RESULTS_DIR = "experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_instance(merged_json_path):
    d = json.load(open(merged_json_path))
    container = d['container']
    boxes_file = os.path.join(os.path.dirname(merged_json_path), d['boxes_file'])
    boxes = []
    # Reassign box IDs sequentially to avoid duplicate IDs from CSVs
    with open(boxes_file) as f:
        rdr = csv.DictReader(f)
        for idx, r in enumerate(rdr, start=1):
            boxes.append({
                'box_id': int(idx),                         # unique, sequential id
                'length': float(r['length']),
                'width':  float(r['width']),
                'height': float(r['height'])
            })
    return d['instance_name'], container, boxes


def run_all():
    rows = []
    merged_files = sorted(glob.glob(os.path.join(OUT_DIR, "*_merged.json")))
    for mf in merged_files:
        name, container, boxes = load_instance(mf)
        cont_vol = container['L']*container['W']*container['H']
        print("Running instance:", name, "boxes:", len(boxes))
        # Baseline
        t0 = time.time()
        res_g = greedy_pack(container, boxes)
        t1 = time.time()
        fr_g = res_g['packed_volume'] / cont_vol
        rows.append({'instance': name, 'method': 'greedy', 'fill_rate': fr_g,
                     'boxes_packed': res_g['boxes_packed'], 'time_s': t1-t0})
        print(" Greedy -> fill_rate:", round(fr_g,4), "boxes:", res_g['boxes_packed'])

        # GA
        t0 = time.time()
        res_ga = run_ga(container, boxes, pop_size=100, gens=200, cx_prob=0.8, mut_prob=0.2, seed=42)
        t1 = time.time()
        fr_ga = res_ga['packed_volume'] / cont_vol
        rows.append({'instance': name, 'method': 'ga', 'fill_rate': fr_ga,
                     'boxes_packed': res_ga['boxes_packed'], 'time_s': res_ga['duration']})
        print(" GA   -> fill_rate:", round(fr_ga,4), "boxes:", res_ga['boxes_packed'], "time:", round(res_ga['duration'],1),"s")

        # Save placements optionally
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "results_summary.csv"), index=False)
    print("All done. Summary saved to", os.path.join(RESULTS_DIR, "results_summary.csv"))

if __name__ == "__main__":
    run_all()
