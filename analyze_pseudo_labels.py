"""
Day 2 — Threshold Sweep Diagnostic
=====================================
Reads pseudo_labels_all.csv (from generate_pseudo_labels.py, which scores
EVERY unlabeled image regardless of threshold) and sweeps multiple confidence
thresholds to show the coverage vs. pseudo-label-accuracy tradeoff.

This is exactly the plot recruiters like to see: it shows you understand that
raising the threshold buys you cleaner pseudo-labels at the cost of using
fewer of them, rather than just having picked p=0.95 arbitrarily.

Usage:
    python src/analyze_pseudo_labels.py --manifest_dir outputs/manifests
"""
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", type=str, default="outputs/manifests")
    parser.add_argument("--fig_dir", type=str, default="outputs/figures")
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    )
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    all_path = os.path.join(args.manifest_dir, "pseudo_labels_all.csv")
    df = pd.read_csv(all_path)

    if "true_label" not in df.columns:
        raise ValueError(
            "true_label column missing — this diagnostic needs the hidden "
            "ground truth to measure pseudo-label accuracy. Re-run "
            "generate_pseudo_labels.py from the manifests produced by "
            "prepare_data.py."
        )
    df["correct"] = df["pred_label"] == df["true_label"]

    rows = []
    for t in args.thresholds:
        kept = df[df["confidence"] >= t]
        coverage = len(kept) / len(df)
        acc = kept["correct"].mean() if len(kept) else float("nan")
        rows.append({"threshold": t, "coverage": coverage,
                     "n_kept": len(kept), "pseudo_label_accuracy": acc})
        print(f"threshold={t:.2f} | coverage={coverage:.1%} "
              f"({len(kept)}/{len(df)}) | pseudo-label accuracy={acc:.1%}")

    sweep_df = pd.DataFrame(rows)
    sweep_path = os.path.join(args.manifest_dir, "threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\nSaved sweep table -> {sweep_path}")

    # --- Plot: coverage & accuracy vs threshold ---
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(sweep_df["threshold"], sweep_df["coverage"], "o-", color="tab:blue", label="Coverage (% kept)")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Coverage (fraction of unlabeled pool kept)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(sweep_df["threshold"], sweep_df["pseudo_label_accuracy"], "s-", color="tab:red", label="Pseudo-label accuracy")
    ax2.set_ylabel("Pseudo-label accuracy (vs. true label)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, 1.05)

    plt.title("Confidence Threshold: Coverage vs. Pseudo-Label Accuracy Tradeoff")
    fig.tight_layout()
    fig_path = os.path.join(args.fig_dir, "threshold_sweep.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved plot -> {fig_path}")

    # --- Plot: confidence score histogram ---
    plt.figure(figsize=(7, 5))
    plt.hist(df.loc[df["correct"], "confidence"], bins=30, alpha=0.6, label="Correct predictions")
    plt.hist(df.loc[~df["correct"], "confidence"], bins=30, alpha=0.6, label="Incorrect predictions")
    plt.xlabel("Teacher model confidence")
    plt.ylabel("Count")
    plt.title("Confidence Distribution on Unlabeled Pool")
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(args.fig_dir, "confidence_histogram.png")
    plt.savefig(hist_path, dpi=150)
    print(f"Saved plot -> {hist_path}")


if __name__ == "__main__":
    main()
