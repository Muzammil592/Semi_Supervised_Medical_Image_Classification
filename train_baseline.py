"""
Day 1 — Baseline Supervised Model
===================================
Trains a ResNet-18 (ImageNet-pretrained backbone, fine-tuned) using ONLY
labeled.csv (your 15% labeled subset). Evaluates on the held-out test.csv.

This number is your reference point: everything you do in Day 2/3 (pseudo-
labeling, FixMatch) should be shown to beat this baseline in your final
ablation table.

Usage:
    python src/train_baseline.py --manifest_dir outputs/manifests \
        --epochs 15 --batch_size 32 --lr 1e-4
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from dataset import ManifestImageDataset, get_transforms
from utils import set_seed, get_device, compute_metrics, save_metrics, print_full_report, EarlyStopper


def build_model(num_classes: int = 2, freeze_backbone: bool = False):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", type=str, default="outputs/manifests")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_frac", type=float, default=0.15,
                        help="Fraction of the labeled set carved out for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--metrics_out", type=str,
                        default="outputs/metrics_baseline.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_tf, eval_tf = get_transforms(args.img_size)

    # Labeled set -> split into train/val
    full_labeled = ManifestImageDataset(
        os.path.join(args.manifest_dir, "labeled.csv"), transform=train_tf
    )
    n_val = int(len(full_labeled) * args.val_frac)
    n_train = len(full_labeled) - n_val
    train_set, val_set = random_split(
        full_labeled, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    # val set should use eval transforms, not train-time augmentation
    val_set.dataset.transform = eval_tf

    test_set = ManifestImageDataset(
        os.path.join(args.manifest_dir, "test.csv"), transform=eval_tf
    )

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=5, mode="max")

    best_val_f1 = -1.0
    best_ckpt_path = os.path.join(args.ckpt_dir, "baseline_best.pt")

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

    # Final evaluation on held-out test set using the BEST checkpoint
    model.load_state_dict(torch.load(best_ckpt_path))
    test_loss, test_metrics, y_true, y_pred = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )

    print("\n=== BASELINE (labeled subset only) — TEST RESULTS ===")
    print(test_metrics)
    print_full_report(y_true, y_pred)

    save_metrics(
        {"setup": "supervised_baseline_labeled_subset", **test_metrics},
        args.metrics_out,
    )


if __name__ == "__main__":
    main()
