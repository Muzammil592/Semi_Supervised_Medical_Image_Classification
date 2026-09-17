"""
Shared utilities: reproducibility, metrics, checkpointing.
"""
import os
import random
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)


def set_seed(seed: int = 42):
    """Make results reproducible across numpy / torch / cuda."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(y_true, y_pred, average: str = "binary"):
    """Return a dict of standard classification metrics.

    average='binary' assumes label 1 = positive class (e.g. PNEUMONIA).
    Use average='macro' if you extend this to >2 classes.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=average),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
    }


def save_metrics(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {path}")


def print_full_report(y_true, y_pred, class_names=("NORMAL", "PNEUMONIA")):
    print(classification_report(y_true, y_pred, target_names=list(class_names)))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


class EarlyStopper:
    """Stops training when val metric hasn't improved for `patience` epochs."""

    def __init__(self, patience: int = 5, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        if self.best is None:
            self.best = value
            return False
        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
