#!/usr/bin/env python3
"""
Evaluate Condition Classifier CNN on the Test Split.
Computes Accuracy, Precision, Recall, and F1-score across all 5 conditions,
generates a detailed classification report and Confusion Matrix visualization,
and saves per-image prediction results to results/condition_classifier_metrics.csv.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Condition Classifier CNN")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="models/condition_classifier_best.pt", help="Path to weights")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_condition_classifier(config: dict, checkpoint_path: str = "models/condition_classifier_best.pt"):
    dataset_cfg = config.get("dataset", {})
    root_dir = Path(dataset_cfg.get("root", "data/bdd100k"))
    images_dir = root_dir / "images"
    test_json = root_dir / "splits" / "test.json"
    results_dir = Path(config.get("paths", {}).get("results", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    if not test_json.exists():
        logger.error(f"Test split not found at {test_json}")
        sys.exit(1)

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        logger.error(f"Checkpoint not found at {ckpt_file}")
        sys.exit(1)

    with open(test_json, "r") as f:
        test_data = json.load(f)
    test_samples = test_data.get("samples", test_data)
    logger.info(f"Loaded {len(test_samples)} test samples.")

    # Create Dataset & Loader
    img_size = config.get("training", {}).get("image_size", 224)
    test_ds = ConditionDataset(test_samples, images_dir, transform=get_transforms(img_size, is_train=False))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Load Model
    device = get_device()
    model_name = config.get("model", {}).get("condition_classifier", "resnet18")
    model = build_condition_classifier(model_name, num_classes=5, pretrained=False)
    
    ckpt = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    confidences = []
    filenames = []
    sample_ids = []

    logger.info("Evaluating condition classifier on test set...")
    with torch.no_grad():
        for images, labels, s_ids, img_paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            confidences.extend(confs.cpu().numpy().tolist())
            sample_ids.extend(s_ids)
            filenames.extend([os.path.basename(p) for p in img_paths])

    # Save per-image predictions CSV
    df_metrics = pd.DataFrame({
        "sample_id": sample_ids,
        "filename": filenames,
        "actual_condition_id": y_true,
        "actual_condition": [CONDITION_NAMES[i] for i in y_true],
        "predicted_condition_id": y_pred,
        "predicted_condition": [CONDITION_NAMES[i] for i in y_pred],
        "confidence": confidences,
        "is_correct": [yt == yp for yt, yp in zip(y_true, y_pred)]
    })

    csv_path = results_dir / "condition_classifier_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    logger.info(f"Saved per-image predictions to {csv_path}")

    # Generate Metrics & Classification Report
    target_names = [CONDITION_NAMES[i] for i in range(5)]
    acc = accuracy_score(y_true, y_pred)
    report_str = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
    
    report_file = results_dir / "condition_classifier_report.txt"
    with open(report_file, "w") as f:
        f.write("============================================================\n")
        f.write("CONDITION CLASSIFIER EVALUATION REPORT\n")
        f.write("============================================================\n")
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report_str)
        f.write("\n============================================================\n")

    logger.info(f"Saved classification report to {report_file}")
    print(f"\nOverall Test Accuracy: {acc * 100:.2f}%\n")
    print(report_str)

    # Generate Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted Condition", fontsize=11, fontweight="bold")
    plt.ylabel("Actual Condition", fontsize=11, fontweight="bold")
    plt.title(f"Condition Classifier Confusion Matrix (Acc: {acc*100:.1f}%)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    cm_path = results_dir / "condition_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_path}")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    evaluate_condition_classifier(cfg, checkpoint_path=args.checkpoint)
