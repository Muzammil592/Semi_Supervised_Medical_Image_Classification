"""
Day 3 — FixMatch (Consistency Regularization)
=================================================
For each unlabeled image:
    1. Generate a pseudo-label from the WEAK augmentation (no gradient).
    2. Keep it only if confidence >= threshold (same idea as Day 2, but
       applied fresh every batch instead of once, and combined with a
       consistency requirement rather than baked into the training set).
    3. Force the model to predict that SAME pseudo-label from the STRONG
       augmentation of the same image (this is where the regularization
       signal comes from — the model must be robust to heavy distortion).

Total loss = CE(labeled) + lambda_u * masked_CE(unlabeled, pseudo_label)

Usage:
    python src/train_fixmatch.py \
        --manifest_dir outputs/manifests \
        --epochs 15 --lr 1e-4 \
        --batch_size_labeled 16 --mu 3 \
        --threshold 0.95 --lambda_u 1.0
"""
import argparse
import itertools
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from dataset import ManifestImageDataset
from dataset_fixmatch import UnlabeledFixMatchDataset
from augmentations import get_weak_transform, get_strong_transform, get_eval_transform
from train_baseline import build_model, run_epoch
from utils import set_seed, get_device, compute_metrics, save_metrics, print_full_report, EarlyStopper


def fixmatch_epoch(model, labeled_loader, unlabeled_loader, optimizer,
                   device, threshold: float, lambda_u: float):
    """One epoch of FixMatch training. Epoch length = len(unlabeled_loader);
    the (smaller) labeled loader is cycled to keep supplying batches."""
    model.train()
    labeled_iter = itertools.cycle(labeled_loader)

    total_loss_l, total_loss_u = 0.0, 0.0
    mask_rates, pseudo_correct, pseudo_total = [], 0, 0
    n_batches = 0

    for weak_u, strong_u, true_label_u in tqdm(unlabeled_loader, desc="FixMatch batches", leave=False):
        imgs_l, labels_l = next(labeled_iter)
        imgs_l, labels_l = imgs_l.to(device), labels_l.to(device)
        weak_u, strong_u = weak_u.to(device), strong_u.to(device)

        optimizer.zero_grad()

        # --- Labeled loss (standard supervised CE) ---
        logits_l = model(imgs_l)
        loss_l = F.cross_entropy(logits_l, labels_l)

        # --- Generate pseudo-labels from WEAK view (no gradient) ---
        with torch.no_grad():
            logits_weak = model(weak_u)
            probs_weak = F.softmax(logits_weak, dim=1)
            max_probs, pseudo_labels = probs_weak.max(dim=1)
            mask = (max_probs >= threshold).float()

        # --- Consistency loss: STRONG view must match the pseudo-label ---
        logits_strong = model(strong_u)
        loss_u_per_sample = F.cross_entropy(logits_strong, pseudo_labels, reduction="none")
        loss_u = (loss_u_per_sample * mask).mean()

        loss = loss_l + lambda_u * loss_u
        loss.backward()
        optimizer.step()

        total_loss_l += loss_l.item()
        total_loss_u += loss_u.item()
        mask_rates.append(mask.mean().item())
        n_batches += 1

        # Diagnostics only (never used in the loss): how accurate are the
        # pseudo-labels we actually kept?
        if (true_label_u >= 0).all():
            kept = mask.bool().cpu()
            if kept.any():
                pseudo_correct += (pseudo_labels.cpu()[kept] == true_label_u[kept]).sum().item()
                pseudo_total += kept.sum().item()

    avg_loss_l = total_loss_l / n_batches
    avg_loss_u = total_loss_u / n_batches
    avg_mask_rate = sum(mask_rates) / len(mask_rates)
    pseudo_acc = (pseudo_correct / pseudo_total) if pseudo_total > 0 else float("nan")

    return {
        "loss_labeled": avg_loss_l,
        "loss_unlabeled": avg_loss_u,
        "mask_rate": avg_mask_rate,
        "pseudo_label_accuracy": pseudo_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", type=str, default="outputs/manifests")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size_labeled", type=int, default=16)
    parser.add_argument("--mu", type=int, default=3,
                        help="Unlabeled batch size = mu * batch_size_labeled")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--lambda_u", type=float, default=1.0,
                        help="Weight on the unlabeled consistency loss")
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--metrics_out", type=str,
                        default="outputs/metrics_fixmatch.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    weak_tf = get_weak_transform(args.img_size)
    strong_tf = get_strong_transform(args.img_size)
    eval_tf = get_eval_transform(args.img_size)

    # Val split carved from original labeled data only (same convention as Day 2)
    labeled_path = os.path.join(args.manifest_dir, "labeled.csv")
    labeled_df = pd.read_csv(labeled_path)
    val_df = labeled_df.sample(frac=args.val_frac, random_state=args.seed)
    train_labeled_df = labeled_df.drop(val_df.index)

    train_labeled_path = os.path.join(args.manifest_dir, "fixmatch_train_labeled.csv")
    val_path = os.path.join(args.manifest_dir, "fixmatch_val.csv")
    train_labeled_df.to_csv(train_labeled_path, index=False)
    val_df.to_csv(val_path, index=False)

    labeled_set = ManifestImageDataset(train_labeled_path, transform=weak_tf)
    val_set = ManifestImageDataset(val_path, transform=eval_tf)
    test_set = ManifestImageDataset(os.path.join(args.manifest_dir, "test.csv"), transform=eval_tf)
    unlabeled_set = UnlabeledFixMatchDataset(
        os.path.join(args.manifest_dir, "unlabeled.csv"), weak_tf, strong_tf
    )

    batch_size_unlabeled = args.batch_size_labeled * args.mu
    print(f"Labeled: {len(labeled_set)} (batch={args.batch_size_labeled}) | "
          f"Unlabeled: {len(unlabeled_set)} (batch={batch_size_unlabeled}) | "
          f"Val: {len(val_set)} | Test: {len(test_set)}")

    labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size_labeled,
                                shuffle=True, drop_last=True, num_workers=2)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size_unlabeled,
                                  shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2)

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()  # only used by run_epoch for val/test
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=5, mode="max")

    best_val_f1 = -1.0
    best_ckpt_path = os.path.join(args.ckpt_dir, "fixmatch_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_stats = fixmatch_epoch(
            model, labeled_loader, unlabeled_loader, optimizer,
            device, args.threshold, args.lambda_u,
        )
        val_loss, val_metrics, _, _ = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        print(
            f"Epoch {epoch:02d} | "
            f"loss_l={train_stats['loss_labeled']:.4f} "
            f"loss_u={train_stats['loss_unlabeled']:.4f} "
            f"mask_rate={train_stats['mask_rate']:.1%} "
            f"pseudo_acc={train_stats['pseudo_label_accuracy']:.1%} | "
            f"val_acc={val_metrics['accuracy']:.3f} val_f1={val_metrics['f1']:.3f}"
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

    print("\n=== FIXMATCH (weak/strong consistency) — TEST RESULTS ===")
    print(test_metrics)
    print_full_report(y_true, y_pred)

    save_metrics(
        {"setup": "fixmatch",
         "threshold": args.threshold,
         "lambda_u": args.lambda_u,
         "mu": args.mu,
         **test_metrics},
        args.metrics_out,
    )


if __name__ == "__main__":
    main()
