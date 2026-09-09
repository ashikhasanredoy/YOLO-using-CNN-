"""
PyTorch Condition Classifier Architecture.
Classifies environmental adverse conditions into 5 categories:
0: clear, 1: rain, 2: fog, 3: night, 4: dawn_dusk.
Supports both transfer learning with torchvision ResNet18 and custom CNN.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Tuple


def get_device() -> torch.device:
    """
    Auto-detects available compute backend:
    CUDA -> Apple Silicon MPS -> CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps (Apple Silicon Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
    return device


class CustomConditionCNN(nn.Module):
    """
    Lightweight standalone convolutional neural network for condition classification.
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


class ResNet18ConditionClassifier(nn.Module):
    """
    Transfer learning condition classifier using torchvision ResNet-18 backbone.
    Replaces the final fully-connected head with a 5-class output layer.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_condition_classifier(model_name: str = "resnet18", num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """Factory method to construct condition classifier."""
    if model_name.lower() == "resnet18":
        return ResNet18ConditionClassifier(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == "custom":
        return CustomConditionCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown condition classifier model name: {model_name}")
