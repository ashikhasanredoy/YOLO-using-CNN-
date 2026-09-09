#!/usr/bin/env python3
"""
Run Pipeline A: Baseline / No Enhancement.
Executes standard YOLO detection on test images without any image modification.
Saves detection outputs to results/baseline_detections.json.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.baseline import BaselinePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Pipeline A: Baseline (No Enhancement)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_baseline(config: dict) -> list:
    dataset_cfg = config.get("dataset", {})
    root_dir = Path(dataset_cfg.get("root", "data/bdd100k"))
    test_json = root_dir / "splits" / "test.json"
    results_dir = Path(config.get("paths", {}).get("results", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    if not test_json.exists():
        logger.error(f"Test split not found at {test_json}")
        sys.exit(1)

    with open(test_json, "r") as f:
        data = json.load(f)
    test_samples = data.get("samples", data)

    # Initialize YOLO
    yolo_weights = config.get("model", {}).get("yolo_weights", "yolo11n.pt")
    logger.info(f"Loading YOLO model with weights: {yolo_weights}")
    yolo_model = YOLO(yolo_weights)

    eval_cfg = config.get("evaluation", {})
    conf_thresh = eval_cfg.get("confidence", 0.25)
    iou_thresh = eval_cfg.get("iou", 0.50)

    pipeline = BaselinePipeline(yolo_model, conf_thresh=conf_thresh, iou_thresh=iou_thresh)

    logger.info(f"Executing Pipeline A (Baseline) on {len(test_samples)} test images...")
    results = []
    for s in tqdm(test_samples, desc="Pipeline A [Baseline]"):
        img_path = str(root_dir / "images" / s["filename"])
        res = pipeline.process_image(
            image_path=img_path,
            actual_condition=s["condition_name"],
            sample_id=str(s.get("image_id", ""))
        )
        results.append(res)

    output_json = results_dir / "baseline_detections.json"
    # Exclude enhanced_image numpy arrays if present for json serializability
    serializable = [{k: v for k, v in r.items() if k != "enhanced_image"} for r in results]
    with open(output_json, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved Baseline detection results to {output_json}")

    return results


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    run_baseline(cfg)
