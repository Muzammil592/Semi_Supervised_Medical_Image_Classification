"""
Day 5 — Ablation Study Runner
=================================
Sweeps one hyperparameter across several values, re-running the relevant
pipeline stage for each, and aggregates the resulting test metrics into a
single table. This is what turns "we picked threshold=0.95" into an actual
ablation study instead of an unjustified default.

Two supported sweeps:

  pseudolabel_threshold — re-runs generate_pseudo_labels.py + train_pseudolabel.py
                          at each confidence threshold
  fixmatch_lambda_u     — re-runs train_fixmatch.py at each lambda_u value

Each run's metrics are saved under a unique filename (so they don't
overwrite each other or the "main" metrics_*.json used by benchmark.py),
and combined into outputs/ablation_<sweep>.csv plus a markdown table
printed to stdout, ready to paste into the README.

Usage:
    python src/run_ablation.py --sweep pseudolabel_threshold \
        --values 0.8 0.9 0.95 0.99 --manifest_dir outputs/manifests

    python src/run_ablation.py --sweep fixmatch_lambda_u \
        --values 0.25 0.5 1.0 2.0 --manifest_dir outputs/manifests

NOTE: this actually re-trains a model for every value in --values, so a
sweep of N values takes roughly N times as long as a single training run.
Keep --epochs modest for ablations (e.g. half of your main run) unless you
have GPU time to spare — document that tradeoff in your README if you do.
"""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd


def run(cmd: list):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result


def sweep_pseudolabel_threshold(values, manifest_dir, ckpt_path, epochs, outputs_dir):
    rows = []
    for t in values:
        print(f"\n{'='*60}\nAblation: pseudo-label threshold = {t}\n{'='*60}")
        run([
            sys.executable, "src/generate_pseudo_labels.py",
            "--manifest_dir", manifest_dir, "--ckpt_path", ckpt_path,
            "--threshold", str(t),
        ])
        metrics_out = os.path.join(outputs_dir, f"ablation_pseudolabel_t{t}.json")
        run([
            sys.executable, "src/train_pseudolabel.py",
            "--manifest_dir", manifest_dir, "--epochs", str(epochs),
            "--metrics_out", metrics_out,
        ])
        with open(metrics_out) as f:
            m = json.load(f)
        rows.append({"threshold": t, **{k: m.get(k) for k in
                    ["accuracy", "f1", "precision", "recall"]}})
    return pd.DataFrame(rows)


def sweep_fixmatch_lambda_u(values, manifest_dir, epochs, outputs_dir):
    rows = []
    for lam in values:
        print(f"\n{'='*60}\nAblation: FixMatch lambda_u = {lam}\n{'='*60}")
        metrics_out = os.path.join(outputs_dir, f"ablation_fixmatch_lambda{lam}.json")
        run([
            sys.executable, "src/train_fixmatch.py",
            "--manifest_dir", manifest_dir, "--epochs", str(epochs),
            "--lambda_u", str(lam), "--metrics_out", metrics_out,
        ])
        with open(metrics_out) as f:
            m = json.load(f)
        rows.append({"lambda_u": lam, **{k: m.get(k) for k in
                    ["accuracy", "f1", "precision", "recall"]}})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True,
                        choices=["pseudolabel_threshold", "fixmatch_lambda_u"])
    parser.add_argument("--values", type=float, nargs="+", required=True)
    parser.add_argument("--manifest_dir", type=str, default="outputs/manifests")
    parser.add_argument("--ckpt_path", type=str,
                        default="outputs/checkpoints/baseline_best.pt",
                        help="Teacher checkpoint (only used by pseudolabel_threshold sweep)")
    parser.add_argument("--epochs", type=int, default=8,
                        help="Epochs per ablation run — keep modest, this multiplies by len(values)")
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    args = parser.parse_args()

    if args.sweep == "pseudolabel_threshold":
        df = sweep_pseudolabel_threshold(
            args.values, args.manifest_dir, args.ckpt_path, args.epochs, args.outputs_dir
        )
        param_col = "threshold"
    else:
        df = sweep_fixmatch_lambda_u(
            args.values, args.manifest_dir, args.epochs, args.outputs_dir
        )
        param_col = "lambda_u"

    csv_path = os.path.join(args.outputs_dir, f"ablation_{args.sweep}.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n\n=== Ablation results: {args.sweep} ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved -> {csv_path}")

    print("\n=== Markdown (paste into README) ===")
    print(f"| {param_col} | Accuracy | F1 | Precision | Recall |")
    print("|---|---|---|---|---|")
    for _, r in df.iterrows():
        print(f"| {r[param_col]} | {r['accuracy']:.3f} | {r['f1']:.3f} | "
              f"{r['precision']:.3f} | {r['recall']:.3f} |")


if __name__ == "__main__":
    main()
