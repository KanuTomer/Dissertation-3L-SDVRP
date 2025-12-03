#!/usr/bin/env python3
# scripts/summarize_replicates.py
# Aggregates replicate output CSV/JSON files into one summary CSV
# Usage: python scripts/summarize_replicates.py --input-dir ./output --out results/replicates_summary.csv

import argparse
import pandas as pd
import json
from pathlib import Path

def read_any(path: Path):
    if path.suffix.lower() in ('.csv',):
        return pd.read_csv(path)
    if path.suffix.lower() in ('.json', '.ndjson'):
        try:
            return pd.DataFrame(json.load(open(path)))
        except Exception:
            # try line-delimited json
            rows = [json.loads(l) for l in open(path) if l.strip()]
            return pd.DataFrame(rows)
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    indir = Path(args.input_dir)
    all_frames = []
    for ext in ('*.csv','*.json','*.ndjson'):
        for f in indir.rglob(ext):
            try:
                df = read_any(f)
                if df is None or df.empty:
                    continue
                # add file origin to help trace
                df['_source_file'] = str(f.relative_to(Path.cwd()))
                all_frames.append(df)
            except Exception as e:
                print("Skipping", f, ":", e)
    if not all_frames:
        print("No result files found in", indir)
        return
    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    # Basic aggregation: group by experiment name if present
    group_cols = [c for c in ['experiment','exp','seed','replicate'] if c in combined.columns]
    if group_cols:
        summary = combined.groupby(group_cols).agg({
            c: 'mean' for c in combined.select_dtypes('number').columns
        }).reset_index()
    else:
        # fallback: aggregate numeric columns
        summary = combined.select_dtypes('number').agg(['count','mean','std']).T.reset_index()
    summary.to_csv(args.out, index=False)
    print("Wrote summary to", args.out)

if __name__ == '__main__':
    main()
