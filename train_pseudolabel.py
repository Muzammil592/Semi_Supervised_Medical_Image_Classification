"""
Day 2 — Retrain Student on Labeled + Pseudo-Labeled Data
============================================================
Combines the original 15% labeled set with the high-confidence pseudo-labels
from generate_pseudo_labels.py, then trains a fresh ResNet-18 ("student")
from ImageNet weights on the combined set. Evaluates on the same held-out
test.csv as the Day 1 baseline so the comparison is apples-to-apples.

Usage:
    python src/train_pseudolabel.py \
        --manifest_dir outputs/manifests \
        --epochs 15 --batch_size 32 --lr 1e-4
"""
import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from dataset import ManifestImageDataset, get_transforms
from train_baseline import build_model, run_epoch
from utils import set_seed, get_device, compute_metrics, save_metrics, print_full_report, EarlyStopper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", type=str, default="outputs/manifests")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_frac", type=float, default=0.15,
                        help="Val split taken from the ORIGINAL labeled set only "
                             "(pseudo-labels are noisy, so we don't validate on them)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--metrics_out", type=str,
                        default="outputs/metrics_pseudolabel.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_tf, eval_tf = get_transforms(args.img_size)

    # --- Build combined training manifest ---
    labeled_path = os.path.join(args.manifest_dir, "labeled.csv")
    kept_path = os.path.join(args.manifest_dir, "pseudo_labels_kept.csv")
    if not os.path.exists(kept_path):
        raise FileNotFoundError(
            f"{kept_path} not found. Run generate_pseudo_labels.py first."
        )

    labeled_df = pd.read_csv(labeled_path)
    pseudo_df = pd.read_csv(kept_path)
    print(f"Original labeled: {len(labeled_df)} | Pseudo-labeled (kept): {len(pseudo_df)}")

    # Carve val split out of the ORIGINAL labeled data only, before combining
    val_df = labeled_df.sample(frac=args.val_frac, random_state=args.seed)
    train_labeled_df = labeled_df.drop(val_df.index)

    combined_train_df = pd.concat(
        [train_labeled_df[["filepath", "label", "class_name"]],
         pseudo_df[["filepath", "label", "class_name"]]],
        ignore_index=True,
    )
    combined_path = os.path.join(args.manifest_dir, "combined_train.csv")
    combined_train_df.to_csv(combined_path, index=False)
    val_path = os.path.join(args.manifest_dir, "pseudolabel_val.csv")
    val_df.to_csv(val_path, index=False)

    print(f"Combined training set: {len(combined_train_df)} images "
          f"({len(train_labeled_df)} real + {len(pseudo_df)} pseudo)")
    print(f"Validation set (real labels only): {len(val_df)}")

    train_set = ManifestImageDataset(combined_path, transform=train_tf)
    val_set = ManifestImageDataset(val_path, transform=eval_tf)
    test_set = ManifestImageDataset(
        os.path.join(args.manifest_dir, "test.csv"), transform=eval_tf
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=5, mode="max")

    best_val_f1 = -1.0
    best_ckpt_path = os.path.join(args.ckpt_dir, "pseudolabel_best.pt")

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

    print("\n=== PSEUDO-LABELED (labeled + high-confidence unlabeled) — TEST RESULTS ===")
    print(test_metrics)
    print_full_report(y_true, y_pred)

    save_metrics(
        {"setup": "pseudo_labeling_self_training",
         "n_original_labeled": len(train_labeled_df),
         "n_pseudo_labeled": len(pseudo_df),
         **test_metrics},
        args.metrics_out,
    )


if __name__ == "__main__":
    main()
