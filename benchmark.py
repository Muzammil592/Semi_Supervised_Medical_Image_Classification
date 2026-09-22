"""
Day 4 — Benchmark Comparison
================================
Reads the metrics_*.json files produced by each training script and
combines them into a single comparison table (CSV, ready to paste into the
README) and a grouped bar chart.

Usage:
    python src/benchmark.py --outputs_dir outputs
"""
import argparse
import json
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


SETUPS = [
    ("Supervised (15% labeled)", "metrics_baseline.json"),
    ("Pseudo-labeling", "metrics_pseudolabel.json"),
    ("FixMatch", "metrics_fixmatch.json"),
    ("Fully supervised (100%)", "metrics_full_supervised.json"),
]

METRIC_COLS = ["accuracy", "f1", "precision", "recall"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--fig_dir", type=str, default="outputs/figures")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    rows = []
    missing = []

    for label, fname in SETUPS:
        path = os.path.join(args.outputs_dir, fname)
        if not os.path.exists(path):
            missing.append(fname)
            continue
        with open(path) as f:
            m = json.load(f)
        rows.append({"setup": label, **{k: m.get(k) for k in METRIC_COLS}})

    if missing:
        print("WARNING: missing metrics files, skipping those rows:")
        for f in missing:
            print(f"  - {f} (run the corresponding training script first)")

    if not rows:
        raise SystemExit(
            "No metrics files found at all. Run train_baseline.py, "
            "train_pseudolabel.py, train_fixmatch.py, and "
            "train_full_supervised.py first."
        )

    df = pd.DataFrame(rows)
    print("\n=== Comparison Table ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    csv_path = os.path.join(args.outputs_dir, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved table -> {csv_path}")

    # Print a markdown version too, ready to paste into the README
    print("\n=== Markdown (paste into README) ===")
    header = "| Setup | Accuracy | F1 | Precision | Recall |"
    sep = "|---|---|---|---|---|"
    print(header)
    print(sep)
    for _, r in df.iterrows():
        print(f"| {r['setup']} | {r['accuracy']:.3f} | {r['f1']:.3f} | "
              f"{r['precision']:.3f} | {r['recall']:.3f} |")

    # --- Grouped bar chart ---
    x = np.arange(len(df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(METRIC_COLS):
        ax.bar(x + i * width, df[metric], width, label=metric.capitalize())

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["setup"], rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("SSL Technique Comparison on Held-Out Test Set")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    fig_path = os.path.join(args.fig_dir, "benchmark_comparison.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved chart -> {fig_path}")


if __name__ == "__main__":
    main()
