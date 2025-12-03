#!/usr/bin/env python3
"""
make_figures_fixed.py
Robust figure generator for replicate summary CSVs.

Usage:
  python scripts/make_figures_fixed.py --summary "<path/to/replicates_summary.csv>" --out-dir "<out_figures_dir>"

This script will detect numeric columns and create a small set of useful plots.
"""
import argparse, pathlib, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def safe_save(fig, outpath):
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to replicates_summary.csv")
    ap.add_argument("--out-dir", required=True, help="Directory to write PNGs")
    args = ap.parse_args()

    sfile = pathlib.Path(args.summary)
    outdir = pathlib.Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sfile)
    print(f"Read {len(df)} rows, columns: {list(df.columns)}")

    # Normalize column names (lowercase keys for matching)
    cols = {c.lower(): c for c in df.columns}

    # helper to get a column by common names
    def pick(names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    # Candidate columns
    seed_col = pick(["seed", "seed_id", "run_seed"])
    score_col = pick(["best_score", "score", "best", "objective", "fitness"])
    unpacked_col = pick(["unpacked", "unpacked_count", "unpacked_items", "unpacked_qty"])
    infeasible_col = pick(["infeasible", "infeasible_count", "infeasible_routes", "infeasible_flag"])
    duration_col = pick(["duration", "time", "run_time", "elapsed"])

    # Ensure seed exists for plotting order
    if seed_col is None:
        df = df.reset_index().rename(columns={"index":"seed"})
        seed_col = "seed"
    # Convert columns to numeric when possible
    for c in [seed_col, score_col, unpacked_col, infeasible_col, duration_col]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 1) Score vs Seed (line + scatter) if available
    if score_col:
        fig, ax = plt.subplots()
        ax.plot(df[seed_col], df[score_col], marker='o', linestyle='-')
        ax.set_xlabel(seed_col)
        ax.set_ylabel(score_col)
        ax.set_title(f"{score_col} vs {seed_col}")
        out = outdir / "score_vs_seed.png"
        safe_save(fig, out)
        print("Saved:", out)

    # 2) Unpacked distribution (violin-like using boxplot) or histogram
    if unpacked_col:
        fig, ax = plt.subplots()
        # boxplot + scatter to show distribution
        ax.boxplot(df[unpacked_col].dropna().values, vert=True)
        ax.set_ylabel(unpacked_col)
        ax.set_title(f"Distribution of {unpacked_col}")
        out = outdir / "unpacked_distribution.png"
        safe_save(fig, out)
        print("Saved:", out)

    # 3) Infeasible histogram or bar
    if infeasible_col:
        fig, ax = plt.subplots()
        ax.hist(df[infeasible_col].dropna().values, bins='auto')
        ax.set_xlabel(infeasible_col)
        ax.set_title(f"Histogram of {infeasible_col}")
        out = outdir / "infeasible_hist.png"
        safe_save(fig, out)
        print("Saved:", out)

    # 4) Duration vs Seed
    if duration_col:
        fig, ax = plt.subplots()
        ax.plot(df[seed_col], df[duration_col], marker='o', linestyle='-')
        ax.set_xlabel(seed_col)
        ax.set_ylabel(duration_col)
        ax.set_title(f"{duration_col} vs {seed_col}")
        out = outdir / "duration_vs_seed.png"
        safe_save(fig, out)
        print("Saved:", out)

    # 5) If no plots were made, at least dump a scatter matrix of numeric cols
    made = list(outdir.glob("*.png"))
    if not made:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) >= 2:
            pd.plotting.scatter_matrix(df[numeric].dropna(), diagonal='kde', figsize=(8,8))
            plt.suptitle("Scatter matrix of numeric columns")
            out = outdir / "scatter_matrix.png"
            plt.savefig(out, bbox_inches="tight")
            plt.close()
            print("Saved scatter matrix:", out)
        else:
            print("No numeric columns found to plot. Columns:", list(df.columns))

if __name__ == "__main__":
    main()
