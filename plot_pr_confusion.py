"""
Day 4 — Precision-Recall Curves & Confusion Matrices
=========================================================
Loads each trained checkpoint, runs inference on the shared test set, and
plots:
    1. A single figure overlaying all models' precision-recall curves
       (easy visual comparison of the accuracy/coverage tradeoff each
       model makes at different decision thresholds).
    2. A confusion matrix per model, as a grid of subplots.

Usage:
    python src/plot_pr_confusion.py --manifest_dir outputs/manifests
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix, average_precision_score

sys.path.append(os.path.dirname(__file__))
from dataset import ManifestImageDataset, get_transforms
from train_baseline import build_model
from utils import get_device


MODELS = [
    ("Supervised (15%)", "baseline_best.pt"),
    ("Pseudo-labeling", "pseudolabel_best.pt"),
    ("FixMatch", "fixmatch_best.pt"),
    ("Fully supervised (100%)", "full_supervised_best.pt"),
]


@torch.no_grad()
def get_probs_and_labels(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)[:, 1]  # P(class 1 = PNEUMONIA)
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.tolist())
    return np.array(all_probs), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", type=str, default="outputs/manifests")
    parser.add_argument("--ckpt_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--fig_dir", type=str, default="outputs/figures")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    _, eval_tf = get_transforms(args.img_size)
    test_set = ManifestImageDataset(os.path.join(args.manifest_dir, "test.csv"), transform=eval_tf)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    results = {}
    for label, ckpt_name in MODELS:
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"Skipping '{label}' — checkpoint not found at {ckpt_path}")
            continue
        model = build_model(num_classes=2).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        probs, labels = get_probs_and_labels(model, test_loader, device)
        results[label] = (probs, labels)
        print(f"Scored '{label}' on {len(labels)} test images")

    if not results:
        raise SystemExit("No checkpoints found. Train at least one model first.")

    # --- Precision-Recall curves, overlaid ---
    plt.figure(figsize=(7, 6))
    for label, (probs, labels) in results.items():
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        plt.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — All Setups")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(args.fig_dir, "precision_recall_curves.png")
    plt.savefig(pr_path, dpi=150)
    print(f"Saved -> {pr_path}")

    # --- Confusion matrices, grid of subplots ---
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (label, (probs, labels)) in zip(axes, results.items()):
        preds = (probs >= 0.5).astype(int)
        cm = confusion_matrix(labels, preds)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["NORMAL", "PNEUMONIA"], fontsize=8)
        ax.set_yticklabels(["NORMAL", "PNEUMONIA"], fontsize=8)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.suptitle("Confusion Matrices — All Setups")
    fig.tight_layout()
    cm_path = os.path.join(args.fig_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Saved -> {cm_path}")


if __name__ == "__main__":
    main()
