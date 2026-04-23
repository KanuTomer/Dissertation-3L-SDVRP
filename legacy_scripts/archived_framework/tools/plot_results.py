# tools/plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

def plot_summary(csv_path):
    df = pd.read_csv(csv_path)
    # remove NA rows if any
    df = df[df.best_score != "NA"]
    df['best_score'] = pd.to_numeric(df['best_score'])
    df['duration'] = pd.to_numeric(df['duration'])
    plt.figure(figsize=(7,4))
    plt.hist(df['best_score'], bins=12)
    plt.title('Distribution of best_score across instances')
    plt.xlabel('best_score')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig('experiments_output/best_score_hist.png', dpi=200)
    plt.close()

    plt.figure(figsize=(7,4))
    plt.scatter(df['duration'], df['best_score'])
    plt.xlabel('duration (s)')
    plt.ylabel('best_score')
    plt.title('best_score vs runtime')
    plt.tight_layout()
    plt.savefig('experiments_output/score_vs_duration.png', dpi=200)
    plt.close()
    print("Wrote experiments_output/best_score_hist.png and score_vs_duration.png")

def plot_convergence(history_csv_path, out_png):
    df = pd.read_csv(history_csv_path, header=None)
    # assume single column of values per generation
    plt.figure(figsize=(7,4))
    plt.plot(df.index, df[0], marker='.')
    plt.xlabel('generation')
    plt.ylabel('score')
    plt.title('Convergence history')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Wrote", out_png)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--summary', default='experiments_output/all_merged_summary_final.csv')
    p.add_argument('--history', default=None, help='path to history.csv (single replicate run) to plot convergence')
    args = p.parse_args()
    if Path(args.summary).exists():
        plot_summary(args.summary)
    if args.history and Path(args.history).exists():
        plot_convergence(args.history, 'experiments_output/convergence.png')
