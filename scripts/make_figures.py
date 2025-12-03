#!/usr/bin/env python3
# scripts/make_figures.py
# Create a couple of PNGs: improvement_vs_seed.png and isolation_counts.png
# Usage: python scripts/make_figures.py --summary results/replicates_summary.csv --out-dir results

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary)
    # Find reasonable numeric columns
    numeric_cols = df.select_dtypes('number').columns.tolist()
    if not numeric_cols:
        print("No numeric columns found in summary CSV.")
        return
    # Try to plot improvement measures if present
    # Heuristic: columns named 'fitness', 'objective', 'unpacked', 'score', 'distance'
    if 'fitness' in df.columns and 'seed' in df.columns:
        plt.figure()
        df.groupby('seed')['fitness'].mean().plot(marker='o')
        plt.title('Mean fitness by seed')
        plt.ylabel('fitness')
        plt.xlabel('seed')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / 'fig_fitness_vs_seed.png')
        plt.close()
        print("Saved fig_fitness_vs_seed.png")
    # If 'unpacked' exists, plot distribution
    if 'unpacked' in df.columns:
        plt.figure()
        df['unpacked'].plot(kind='hist', bins=20)
        plt.title('Distribution of unpacked boxes across replicates')
        plt.xlabel('unpacked')
        plt.tight_layout()
        plt.savefig(outdir / 'fig_unpacked_hist.png')
        plt.close()
        print("Saved fig_unpacked_hist.png")
    # Fallback: plot the first two numeric columns against each other
    if len(numeric_cols) >= 2:
        plt.figure()
        df.plot.scatter(x=numeric_cols[0], y=numeric_cols[1])
        plt.title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
        plt.tight_layout()
        plt.savefig(outdir / f'fig_{numeric_cols[0]}_vs_{numeric_cols[1]}.png')
        plt.close()
        print(f"Saved fig_{numeric_cols[0]}_vs_{numeric_cols[1]}.png")

if __name__ == '__main__':
    main()
