"""
PyTorch Dataset definition for Condition Classifier training and evaluation.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def get_transforms(image_size: int = 224, is_train: bool = False) -> Callable:
    """
    Standard torchvision transformations with ImageNet normalization.
    Augmentations are applied only during training.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


class ConditionDataset(Dataset):
    """
    PyTorch Dataset yielding (tensor, condition_class_id, sample_id, image_path).
    """
    def __init__(self, samples: List[Dict[str, Any]], images_dir: Path, transform: Optional[Callable] = None):
        self.samples = samples
        self.images_dir = Path(images_dir)
        self.transform = transform or get_transforms(is_train=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        s = self.samples[idx]
        image_name = os.path.basename(s["filepath"])
        img_path = self.images_dir / image_name

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found on disk: {img_path}")

        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        label = int(s["condition_label"])
        sample_id = str(s.get("_id", image_name))

        return tensor, label, sample_id, str(img_path)
