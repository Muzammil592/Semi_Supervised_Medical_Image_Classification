"""
PyTorch Dataset that reads from the manifest CSVs produced by prepare_data.py.
"""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ManifestImageDataset(Dataset):
    """Generic image dataset backed by a CSV with columns: filepath, label."""

    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row["label"])
        return img, label


def get_transforms(img_size: int = 224):
    """Standard train/eval transforms. Swap in RandAugment/CutOut for Day 3."""
    from torchvision import transforms

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_tf, eval_tf
