"""
Day 2 — Pseudo-Labeling: Generate & Filter
=============================================
Loads the Day 1 baseline checkpoint (the "teacher"), runs inference on the
70% unlabeled pool, and keeps only predictions above a confidence threshold.

Produces two files:
    pseudo_labels_all.csv   -> every unlabeled image + predicted label +
                                confidence + whether it was kept. Useful for
                                plotting the confidence distribution.
    pseudo_labels_kept.csv  -> only the kept (high-confidence) rows, in the
                                same filepath/label/class_name format as
                                labeled.csv, ready to concatenate for
                                retraining.

IMPORTANT: `true_label` is carried through ONLY for diagnostics (to report
pseudo-label accuracy in your README / ablation table). It is never used
as a training signal — train_pseudolabel.py drops it before training.

Usage:
    python src/generate_pseudo_labels.py \
        --manifest_dir outputs/manifests \
        --ckpt_path outputs/checkpoints/baseline_best.pt \
        --threshold 0.95
"""
import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from dataset import ManifestImageDataset, get_transforms
from train_baseline import build_model
from utils import get_device


@torch.no_grad()
def score_unlabeled(model, loader, device):
    """Returns (all_probs, all_preds) for every sample in loader, in order."""
    model.eval()
    all_probs, all_preds = [], []
    for imgs, _ in tqdm(loader, desc="Scoring unlabeled pool"):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        all_probs.extend(conf.cpu().tolist())
        all_preds.extend(pred.cpu().tolist())
    return all_probs, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", type=str, default="outputs/manifests")
    parser.add_argument("--ckpt_path", type=str,
                        default="outputs/checkpoints/baseline_best.pt")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Minimum softmax confidence to keep a pseudo-label")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="outputs/manifests")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    _, eval_tf = get_transforms(args.img_size)
    unlabeled_path = os.path.join(args.manifest_dir, "unlabeled.csv")
    unlabeled_set = ManifestImageDataset(unlabeled_path, transform=eval_tf)
    # NOTE: label column in unlabeled.csv is the sentinel -1, so it's never
    # used by score_unlabeled — only the images matter here.
    loader = DataLoader(unlabeled_set, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

    model = build_model(num_classes=2).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    print(f"Loaded teacher checkpoint from {args.ckpt_path}")

    confidences, preds = score_unlabeled(model, loader, device)

    df = pd.read_csv(unlabeled_path)
    class_map = pd.read_csv(os.path.join(args.manifest_dir, "class_map.csv"))
    idx_to_class = dict(zip(class_map["label"], class_map["class_name"]))

    df["pred_label"] = preds
    df["confidence"] = confidences
    df["pred_class_name"] = df["pred_label"].map(idx_to_class)
    df["kept"] = df["confidence"] >= args.threshold

    # Diagnostics only — true_label already exists in unlabeled.csv from Day 1
    if "true_label" in df.columns:
        df["correct"] = df["pred_label"] == df["true_label"]
        overall_acc = df["correct"].mean()
        kept_df = df[df["kept"]]
        kept_acc = kept_df["correct"].mean() if len(kept_df) else float("nan")
        print("\n=== Pseudo-label diagnostics (NOT used for training) ===")
        print(f"Teacher accuracy on ALL unlabeled images:  {overall_acc:.3%}")
        print(f"Confidence threshold:                       {args.threshold}")
        print(f"Kept (>= threshold):                        {len(kept_df)} / {len(df)} ({len(kept_df)/len(df):.1%})")
        print(f"Teacher accuracy on KEPT (pseudo-label) set: {kept_acc:.3%}")
        print("\nKept predictions by class:")
        print(kept_df["pred_class_name"].value_counts())

    # Save full diagnostic file
    all_out_path = os.path.join(args.out_dir, "pseudo_labels_all.csv")
    df.to_csv(all_out_path, index=False)
    print(f"\nSaved full scored pool -> {all_out_path}")

    # Save the kept subset in labeled.csv-compatible format for retraining
    kept_df = df[df["kept"]].copy()
    kept_df = kept_df.rename(columns={"pred_label": "label",
                                       "pred_class_name": "class_name"})
    kept_out_path = os.path.join(args.out_dir, "pseudo_labels_kept.csv")
    kept_df[["filepath", "label", "class_name"]].to_csv(kept_out_path, index=False)
    print(f"Saved kept pseudo-labels -> {kept_out_path}")


if __name__ == "__main__":
    main()
