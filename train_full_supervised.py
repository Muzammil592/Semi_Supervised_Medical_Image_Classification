"""
Day 4 — Fully Supervised Reference Ceiling
==============================================
Trains on ALL available data with REAL labels (the original 15% labeled
set + the 70% "unlabeled" set, using its hidden true_label column as if it
had been labeled all along). This is the number that answers "how much
performance are we leaving on the table by only labeling 15%?" — it's the
ceiling every SSL technique is trying to approach without paying for the
other 85% of labels.

This is a diagnostic/reference run only — in a real deployment you
wouldn't have these labels, which is the entire premise of the project.

Usage:
    python src/train_full_supervised.py \
        --manifest_dir outputs/manifests --epochs 15
"""
import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from dataset import ManifestImageDataset, get_transforms
from train_baseline import build_model, run_epoch
from utils import set_seed, get_device, save_metrics, print_full_report, EarlyStopper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", type=str, default="outputs/manifests")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--metrics_out", type=str,
                        default="outputs/metrics_full_supervised.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_tf, eval_tf = get_transforms(args.img_size)

    # --- Build the "fully labeled" pool: labeled.csv + unlabeled.csv(true_label) ---
    labeled_df = pd.read_csv(os.path.join(args.manifest_dir, "labeled.csv"))
    unlabeled_df = pd.read_csv(os.path.join(args.manifest_dir, "unlabeled.csv"))
    if "true_label" not in unlabeled_df.columns:
        raise ValueError("unlabeled.csv is missing true_label — regenerate manifests from prepare_data.py")
    unlabeled_as_labeled = unlabeled_df.rename(columns={"true_label": "label"})[
        ["filepath", "label", "class_name"]
    ]
    # NOTE: unlabeled.csv's own `label` column (the -1 sentinel) is dropped here,
    # replaced by the real `true_label`, since this script simulates having
    # annotated everything.
    full_df = pd.concat(
        [labeled_df[["filepath", "label", "class_name"]], unlabeled_as_labeled],
        ignore_index=True,
    )
    print(f"Full supervised pool: {len(full_df)} images "
          f"({len(labeled_df)} originally labeled + {len(unlabeled_as_labeled)} newly 'labeled')")

    val_df = full_df.sample(frac=args.val_frac, random_state=args.seed)
    train_df = full_df.drop(val_df.index)

    train_path = os.path.join(args.manifest_dir, "full_supervised_train.csv")
    val_path = os.path.join(args.manifest_dir, "full_supervised_val.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    train_set = ManifestImageDataset(train_path, transform=train_tf)
    val_set = ManifestImageDataset(val_path, transform=eval_tf)
    test_set = ManifestImageDataset(os.path.join(args.manifest_dir, "test.csv"), transform=eval_tf)

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=5, mode="max")

    best_val_f1 = -1.0
    best_ckpt_path = os.path.join(args.ckpt_dir, "full_supervised_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_metrics, _, _ = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} acc={train_metrics['accuracy']:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_metrics['accuracy']:.3f} f1={val_metrics['f1']:.3f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  -> new best model saved (val_f1={best_val_f1:.3f})")

        if stopper.step(val_metrics["f1"]):
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(best_ckpt_path))
    test_loss, test_metrics, y_true, y_pred = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )

    print("\n=== FULLY SUPERVISED (100% labeled, reference ceiling) — TEST RESULTS ===")
    print(test_metrics)
    print_full_report(y_true, y_pred)

    save_metrics(
        {"setup": "fully_supervised_100pct", "n_train": len(train_df), **test_metrics},
        args.metrics_out,
    )


if __name__ == "__main__":
    main()
