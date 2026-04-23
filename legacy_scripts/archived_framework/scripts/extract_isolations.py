#!/usr/bin/env python3
# scripts/extract_isolations.py
# Parse log outputs from byboxes_iterative_isolate.py (or printed lines) and create a CSV
# Usage: python scripts/extract_isolations.py --logs-dir results --out results/isolation_summary.csv

import re
import argparse
from pathlib import Path
import pandas as pd

PAT_ISOLATED = re.compile(r'Isolated\s+(\d+)\s+from\s+route\s+(\d+)', re.IGNORECASE)
PAT_START = re.compile(r'Starting total_unpacked:\s*(\d+)', re.IGNORECASE)

def parse_log(path):
    entries = []
    with open(path, encoding='utf8', errors='ignore') as fh:
        content = fh.read().splitlines()
    total_unpacked = None
    for ln in content:
        m = PAT_START.search(ln)
        if m:
            total_unpacked = int(m.group(1))
        m2 = PAT_ISOLATED.search(ln)
        if m2:
            box = int(m2.group(1))
            route = int(m2.group(2))
            entries.append({'log_file': str(path.name), 'isolated_box': box, 'route': route, 'total_unpacked': total_unpacked})
    return entries

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    logs = list(Path(args.logs_dir).glob('isolate_*.log'))
    rows = []
    for l in logs:
        rows.extend(parse_log(l))
    if not rows:
        print("No isolation rows found in", args.logs_dir)
        return
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == '__main__':
    main()
