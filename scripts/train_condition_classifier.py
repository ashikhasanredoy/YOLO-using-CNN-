#!/usr/bin/env python3
"""
PyTorch Condition Classifier Training Script.
Trains ResNet-18 (or Custom CNN) to classify adverse environmental conditions:
0: clear, 1: rain, 2: fog, 3: night, 4: dawn_dusk.
Tracks loss/accuracy curves, performs validation, saves best checkpoint,
and generates distribution/training plots.
"""

import os
import sys
import json
import yaml
import random
import argparse
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.condition_cnn import build_condition_classifier, get_device
from src.data.dataset import ConditionDataset, get_transforms
from src.data.metadata import CONDITION_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Condition Classifier CNN")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def plot_class_distribution(train_samples, val_samples, test_samples, save_path: Path):
    """Plots and saves condition class distributions across splits."""
    labels = [CONDITION_NAMES[i] for i in range(5)]
    splits = {"Train": train_samples, "Val": val_samples, "Test": test_samples}
    
    data = {s_name: [0]*5 for s_name in splits}
    for s_name, sample_list in splits.items():
        counts = Counter(s["condition_label"] for s in sample_list)
        for i in range(5):
            data[s_name][i] = counts.get(i, 0)

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(9, 5))
    plt.bar(x - width, data["Train"], width, label="Train", color="#2563eb")
    plt.bar(x, data["Val"], width, label="Val", color="#10b981")
    plt.bar(x + width, data["Test"], width, label="Test", color="#f59e0b")

    plt.xlabel("Condition Class", fontsize=11, fontweight="bold")
    plt.ylabel("Number of Images", fontsize=11, fontweight="bold")
    plt.title("BDD100K Environmental Condition Class Distribution", fontsize=12, fontweight="bold")
    plt.xticks(x, labels, fontsize=10)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved class distribution plot to {save_path}")


def train_condition_classifier(config: dict, epochs: int = None, batch_size: int = None, lr: float = None):
    train_cfg = config.get("training", {})
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    epochs = epochs or train_cfg.get("epochs", 15)
    batch_size = batch_size or train_cfg.get("batch_size", 32)
    lr = lr or train_cfg.get("learning_rate", 0.0003)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    img_size = train_cfg.get("image_size", 224)
    num_workers = train_cfg.get("num_workers", 2)

    dataset_cfg = config.get("dataset", {})
    root_dir = Path(dataset_cfg.get("root", "data/bdd100k"))
    images_dir = root_dir / "images"
    splits_dir = root_dir / "splits"
    models_dir = Path(config.get("paths", {}).get("models", "models"))
    results_dir = Path(config.get("paths", {}).get("results", "results"))

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    def _load_split_json(p):
        with open(p, "r") as f:
            d = json.load(f)
        return d.get("samples", d)

    train_samples = _load_split_json(splits_dir / "train.json")
    val_samples = _load_split_json(splits_dir / "val.json")
    test_samples = _load_split_json(splits_dir / "test.json")

    logger.info(f"Loaded samples: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    # Plot class distribution
    plot_class_distribution(
        train_samples, val_samples, test_samples,
        save_path=results_dir / "condition_class_distribution.png"
    )

    # Create Datasets & DataLoaders
    train_ds = ConditionDataset(train_samples, images_dir, transform=get_transforms(img_size, is_train=True))
    val_ds = ConditionDataset(val_samples, images_dir, transform=get_transforms(img_size, is_train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Build model & device
    device = get_device()
    model_name = config.get("model", {}).get("condition_classifier", "resnet18")
    pretrained = config.get("model", {}).get("pretrained", True)
    model = build_condition_classifier(model_name, num_classes=5, pretrained=pretrained)
    model.to(device)

    # Compute class weights for imbalanced classes (like rare fog)
    train_counts = Counter(s["condition_label"] for s in train_samples)
    total_train = len(train_samples)
    weights = [total_train / (5.0 * max(train_counts.get(i, 1), 1)) for i in range(5)]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_checkpoint_path = models_dir / "condition_classifier_best.pt"

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    logger.info(f"Starting Condition Classifier training for {epochs} epochs on {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train_items = 0

        for images, labels, _, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train_items += images.size(0)

        scheduler.step()
        epoch_train_loss = running_loss / total_train_items
        epoch_train_acc = correct_train / total_train_items

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val_items = 0

        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val_items += images.size(0)

        epoch_val_loss = val_loss / total_val_items
        epoch_val_acc = correct_val / total_val_items

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%"
        )
        sys.stdout.flush()

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "condition_names": CONDITION_NAMES
            }, best_checkpoint_path)
            logger.info(f"  -> Saved new best checkpoint (Val Acc: {best_val_acc*100:.2f}%) to {best_checkpoint_path}")
            sys.stdout.flush()

    # Plot Training & Validation curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", color="#2563eb", linewidth=2)
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", color="#dc2626", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Condition Classifier: Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["train_acc"]], label="Train Acc (%)", color="#2563eb", linewidth=2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["val_acc"]], label="Val Acc (%)", color="#10b981", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Condition Classifier: Accuracy Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()

    curve_path = results_dir / "condition_training_curve.png"
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300)
    plt.close()
    logger.info(f"Saved training curve to {curve_path}")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train_condition_classifier(cfg, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
