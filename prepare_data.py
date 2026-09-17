"""
Day 1 — Data Preparation
=========================
Builds a single manifest of (image_path, label) from a folder-per-class
dataset, then produces a STRATIFIED 3-way split:

    labeled.csv     -> 15% of the training pool, WITH labels (used for training)
    unlabeled.csv   -> 70% of the training pool, labels are stripped for
                       training but kept in a hidden column so you can later
                       measure pseudo-label accuracy (Day 2/3 diagnostics only —
                       never feed `true_label` into the model).
    test.csv        -> 15% held out, untouched until final evaluation.

Expected input layout (this matches the Kaggle "Chest X-Ray Pneumonia" dataset
once you merge its train/val/test folders back into one pool — we want to
control the split ourselves rather than use the dataset's default one):

    data/raw/
        NORMAL/xxx.jpeg
        PNEUMONIA/xxx.jpeg

Usage:
    python src/prepare_data.py --data_root data/raw --out_dir outputs/manifests \
        --labeled_frac 0.15 --test_frac 0.15 --seed 42
"""
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def build_manifest(data_root: str) -> pd.DataFrame:
    """Walk data_root/<class_name>/*.jpg and build a dataframe."""
    rows = []
    class_names = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])
    if not class_names:
        raise ValueError(
            f"No class subfolders found under {data_root}. "
            "Expected data_root/<class_name>/<image files>."
        )
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print(f"Found classes: {class_to_idx}")

    for cls in class_names:
        cls_dir = os.path.join(data_root, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(IMG_EXTENSIONS):
                rows.append({
                    "filepath": os.path.join(cls_dir, fname),
                    "label": class_to_idx[cls],
                    "class_name": cls,
                })
    df = pd.DataFrame(rows)
    print(f"Total images found: {len(df)}")
    print(df["class_name"].value_counts())
    return df, class_to_idx


def split_data(df: pd.DataFrame, labeled_frac: float, test_frac: float, seed: int):
    # Step 1: carve off the test set first (never touched again)
    trainpool_df, test_df = train_test_split(
        df, test_size=test_frac, stratify=df["label"], random_state=seed
    )

    # Step 2: from the remaining pool, carve off the labeled fraction.
    # labeled_frac is expressed relative to the FULL dataset, so rescale
    # it relative to what's left in trainpool_df.
    remaining_frac = 1 - test_frac
    labeled_frac_within_pool = labeled_frac / remaining_frac

    labeled_df, unlabeled_df = train_test_split(
        trainpool_df,
        train_size=labeled_frac_within_pool,
        stratify=trainpool_df["label"],
        random_state=seed,
    )

    return labeled_df, unlabeled_df, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder containing one subfolder per class")
    parser.add_argument("--out_dir", type=str, default="outputs/manifests")
    parser.add_argument("--labeled_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df, class_to_idx = build_manifest(args.data_root)
    labeled_df, unlabeled_df, test_df = split_data(
        df, args.labeled_frac, args.test_frac, args.seed
    )

    # Save labeled + test with their real labels
    labeled_df.to_csv(os.path.join(args.out_dir, "labeled.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)

    # Save unlabeled with label hidden as `true_label` (diagnostics only)
    unlabeled_out = unlabeled_df.rename(columns={"label": "true_label"}).copy()
    unlabeled_out["label"] = -1  # sentinel: "no label available"
    unlabeled_out.to_csv(os.path.join(args.out_dir, "unlabeled.csv"), index=False)

    # Save class mapping for later scripts
    pd.DataFrame(list(class_to_idx.items()), columns=["class_name", "label"]).to_csv(
        os.path.join(args.out_dir, "class_map.csv"), index=False
    )

    print("\n--- Split summary ---")
    print(f"Labeled:   {len(labeled_df):5d}  ({len(labeled_df)/len(df):.1%})")
    print(f"Unlabeled: {len(unlabeled_df):5d}  ({len(unlabeled_df)/len(df):.1%})")
    print(f"Test:      {len(test_df):5d}  ({len(test_df)/len(df):.1%})")
    print(f"\nSaved manifests to {args.out_dir}/")


if __name__ == "__main__":
    main()
