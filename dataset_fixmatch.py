"""
FixMatch-specific dataset: for each unlabeled image, returns BOTH a weakly
and a strongly augmented view. The two views are generated fresh each time
__getitem__ is called (i.e. fresh random augmentation every epoch, standard
PyTorch DataLoader behavior).
"""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class UnlabeledFixMatchDataset(Dataset):
    """Backed by unlabeled.csv (or pseudo_labels_all.csv). `true_label`,
    if present, is carried through ONLY for diagnostics (measuring how
    accurate the confident pseudo-labels are during training) — it is
    never used as a training signal.
    """

    def __init__(self, csv_path: str, weak_transform, strong_transform):
        self.df = pd.read_csv(csv_path)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.has_true_label = "true_label" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        weak_img = self.weak_transform(img)
        strong_img = self.strong_transform(img)
        true_label = int(row["true_label"]) if self.has_true_label else -1
        return weak_img, strong_img, true_label
