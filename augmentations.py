"""
Weak vs. Strong augmentation pipelines for FixMatch.

Weak augmentation: small, safe perturbations (flip + shift) — used to
generate the pseudo-label target.

Strong augmentation: RandAugment (random sequence of strong distortions)
+ Cutout-style random erasing — the model must predict the SAME pseudo-label
from this heavily distorted view. That consistency requirement is what
regularizes the model on unlabeled data.
"""
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_weak_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_strong_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        # RandAugment: randomly applies a chain of strong distortions
        # (rotation, shear, contrast, posterize, etc.) each call.
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Cutout stand-in: randomly erases a rectangular patch post-normalization.
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])


def get_eval_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
