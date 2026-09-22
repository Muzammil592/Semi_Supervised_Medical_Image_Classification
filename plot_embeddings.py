"""
Day 4 — Feature Embedding Visualization (t-SNE / UMAP)
============================================================
Extracts penultimate-layer features (the 512-dim vector right before the
final classification layer) from the test set, for both the supervised
baseline and an SSL-trained model. Projects both to 2D with t-SNE (or UMAP)
and plots them side by side.

The story this plot should tell: if SSL is doing its job, the SSL model's
2D projection should show tighter, more separated clusters per class than
the baseline's — i.e. the unlabeled data pushed the model toward learning
more class-discriminative features, not just memorizing the 15% labeled set.

Usage:
    python src/plot_embeddings.py \
        --manifest_dir outputs/manifests \
        --model_a_ckpt outputs/checkpoints/baseline_best.pt --model_a_label "Baseline (15%)" \
        --model_b_ckpt outputs/checkpoints/fixmatch_best.pt --model_b_label "FixMatch" \
        --method tsne
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from dataset import ManifestImageDataset, get_transforms
from train_baseline import build_model
from utils import get_device


def get_feature_extractor(model: nn.Module) -> nn.Module:
    """Returns the model with its final FC layer replaced by Identity, so
    forward() returns the penultimate 512-dim feature vector instead of
    class logits."""
    model.fc = nn.Identity()
    return model


@torch.no_grad()
def extract_features(ckpt_path, loader, device):
    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = get_feature_extractor(model).to(device)
    model.eval()

    all_feats, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = model(imgs)
        all_feats.append(feats.cpu().numpy())
        all_labels.extend(labels.tolist())
    return np.concatenate(all_feats, axis=0), np.array(all_labels)


def project_2d(features, method: str, seed: int = 42):
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=seed, init="pca",
                       perplexity=min(30, max(5, len(features) // 10 - 1)))
    elif method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed)
    else:
        raise ValueError(f"Unknown method: {method}")
    return reducer.fit_transform(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", type=str, default="outputs/manifests")
    parser.add_argument("--model_a_ckpt", type=str,
                        default="outputs/checkpoints/baseline_best.pt")
    parser.add_argument("--model_a_label", type=str, default="Baseline (15% labeled)")
    parser.add_argument("--model_b_ckpt", type=str,
                        default="outputs/checkpoints/fixmatch_best.pt")
    parser.add_argument("--model_b_label", type=str, default="FixMatch")
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "umap"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fig_dir", type=str, default="outputs/figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    for ckpt in (args.model_a_ckpt, args.model_b_ckpt):
        if not os.path.exists(ckpt):
            raise SystemExit(f"Checkpoint not found: {ckpt}")

    _, eval_tf = get_transforms(args.img_size)
    test_set = ManifestImageDataset(os.path.join(args.manifest_dir, "test.csv"), transform=eval_tf)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    class_names = {0: "NORMAL", 1: "PNEUMONIA"}
    colors = {0: "tab:blue", 1: "tab:red"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, ckpt, title in [
        (axes[0], args.model_a_ckpt, args.model_a_label),
        (axes[1], args.model_b_ckpt, args.model_b_label),
    ]:
        print(f"Extracting features for '{title}'...")
        feats, labels = extract_features(ckpt, test_loader, device)
        print(f"Projecting to 2D with {args.method}...")
        emb = project_2d(feats, args.method, args.seed)

        for cls_idx, cls_name in class_names.items():
            mask = labels == cls_idx
            ax.scatter(emb[mask, 0], emb[mask, 1], s=12, alpha=0.6,
                      color=colors[cls_idx], label=cls_name)
        ax.set_title(title)
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Feature Embeddings on Test Set ({args.method.upper()})")
    fig.tight_layout()
    out_path = os.path.join(args.fig_dir, f"embeddings_{args.method}.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
