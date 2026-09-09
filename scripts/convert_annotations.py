#!/usr/bin/env python3
"""
BDD100K to YOLO Annotation Converter.
Converts FiftyOne normalized [x_min, y_min, width, height] bounding boxes
to Ultralytics YOLO format [class_id, x_center, y_center, width, height].
Generates data/yolo/bdd100k.yaml and YOLO label txt files.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Standard BDD100k detection classes mapped to indices
BDD_CLASSES = [
    "pedestrian",    # 0 (maps to person)
    "rider",         # 1
    "car",           # 2
    "truck",         # 3
    "bus",           # 4
    "train",         # 5
    "motorcycle",    # 6
    "bicycle",       # 7
    "traffic light", # 8
    "traffic sign"   # 9
]

CLASS_TO_IDX = {c: i for i, c in enumerate(BDD_CLASSES)}

# Optional synonyms for robust matching
LABEL_SYNONYMS = {
    "person": "pedestrian",
    "bike": "bicycle",
    "motor": "motorcycle"
}

# COCO indices for pretrained YOLO11n evaluation
COCO_MAPPING = {
    "pedestrian": 0,    # person
    "rider": 0,         # person
    "bicycle": 1,       # bicycle
    "car": 2,           # car
    "motorcycle": 3,    # motorcycle
    "bus": 5,           # bus
    "train": 6,         # train
    "truck": 7,         # truck
    "traffic light": 9, # traffic light
    "traffic sign": 11  # stop sign / traffic sign
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert BDD100K annotations to YOLO format")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def convert_annotations(config: dict):
    dataset_cfg = config.get("dataset", {})
    root_dir = Path(dataset_cfg.get("root", "data/bdd100k"))
    labels_dir = root_dir / "labels"
    splits_dir = root_dir / "splits"
    yolo_dir = Path(dataset_cfg.get("yolo_dir", "data/yolo"))

    labels_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)

    prepared_manifest = root_dir / "metadata" / "prepared_samples.json"
    if not prepared_manifest.exists():
        logger.error(f"Prepared manifest not found at {prepared_manifest}. Run prepare_dataset.py first.")
        sys.exit(1)

    with open(prepared_manifest, "r") as f:
        data = json.load(f)
    samples = data.get("samples", data)

    logger.info(f"Converting annotations for {len(samples)} samples...")
    converted_count = 0
    box_count = 0

    for s in samples:
        fname = s["filename"]
        base_stem = Path(fname).stem
        label_file = labels_dir / f"{base_stem}.txt"

        dets = s.get("detections", [])
        yolo_lines = []

        for d in dets:
            raw_label = d.get("label", "").lower()
            std_label = LABEL_SYNONYMS.get(raw_label, raw_label)
            if std_label not in CLASS_TO_IDX:
                continue

            class_id = CLASS_TO_IDX[std_label]
            bbox = d.get("bounding_box", [])  # [x_min, y_min, w, h]
            if len(bbox) != 4:
                continue

            x_min, y_min, w, h = bbox
            x_center = x_min + w / 2.0
            y_center = y_min + h / 2.0

            # Clamp normalized coordinates to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            if w <= 0 or h <= 0:
                continue

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        with open(label_file, "w") as f_out:
            f_out.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

        converted_count += 1
        box_count += len(yolo_lines)

    logger.info(f"Generated {converted_count} YOLO label files ({box_count} total bounding boxes) in {labels_dir}")

    # Generate data/yolo/bdd100k.yaml
    yaml_content = {
        "path": str(root_dir.resolve()),
        "train": "splits/train_images.txt",
        "val": "splits/val_images.txt",
        "test": "splits/test_images.txt",
        "names": {i: name for i, name in enumerate(BDD_CLASSES)},
        "coco_mapping": COCO_MAPPING
    }

    yaml_file = yolo_dir / "bdd100k.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    logger.info(f"Generated YOLO dataset configuration file at {yaml_file}")

    # Also create split image lists for YOLO
    for split_name in ["train", "val", "test"]:
        split_json = splits_dir / f"{split_name}.json"
        if split_json.exists():
            with open(split_json, "r") as f:
                s_data = json.load(f)
            split_samples = s_data.get("samples", s_data)
            txt_file = splits_dir / f"{split_name}_images.txt"
            with open(txt_file, "w") as f:
                for s in split_samples:
                    # Write relative image path
                    f.write(f"images/{s['filename']}\n")
            logger.info(f"Wrote {len(split_samples)} image paths to {txt_file}")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    convert_annotations(cfg)
