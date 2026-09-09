#!/usr/bin/env python3
"""
BDD100K Dataset Preparation Script.
Matches downloaded images with ground-truth metadata, assigns deterministic
condition labels (0..4), computes physical derived low-visibility indicators,
and generates stratified Train/Val/Test splits with fixed random seed 42.
"""

import os
import sys
import json
import glob
import yaml
import argparse
import logging
from pathlib import Path
from collections import Counter
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.metadata import map_condition_label, compute_visibility_metrics, is_derived_low_visibility
from src.data.split import create_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare BDD100K dataset and generate splits")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataset(config: dict, seed: int = 42):
    dataset_cfg = config.get("dataset", {})
    root_dir = Path(dataset_cfg.get("root", "data/bdd100k"))
    images_dir = root_dir / "images"
    annotations_file = root_dir / "annotations" / "samples.json"
    splits_dir = root_dir / "splits"
    metadata_dir = root_dir / "metadata"

    splits_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if not annotations_file.exists():
        logger.error(f"Annotations file not found at {annotations_file}")
        sys.exit(1)

    with open(annotations_file, "r") as f:
        raw_data = json.load(f)

    all_samples = raw_data.get("samples", raw_data)
    logger.info(f"Loaded {len(all_samples)} samples from {annotations_file}")

    # Index samples by filename
    sample_lookup = {}
    for s in all_samples:
        rel_path = s.get("filepath", "")
        if rel_path:
            fname = os.path.basename(rel_path)
            sample_lookup[fname] = s

    # Find existing images on disk
    image_files = sorted(list(images_dir.glob("*.jpg")))
    logger.info(f"Found {len(image_files)} local image files in {images_dir}")

    valid_prepared_samples = []
    condition_counts = Counter()
    derived_low_vis_count = 0

    low_vis_cfg = config.get("conditions", {}).get("derived_low_visibility", {})
    lum_thresh = low_vis_cfg.get("luminance_threshold", 45.0)
    contrast_thresh = low_vis_cfg.get("contrast_threshold", 35.0)

    logger.info("Extracting condition metadata and calculating physical visibility metrics...")
    for img_path in tqdm(image_files, desc="Processing images"):
        fname = img_path.name
        sample_meta = sample_lookup.get(fname)
        if not sample_meta:
            continue

        weather = (sample_meta.get("weather") or {}).get("label", "")
        timeofday = (sample_meta.get("timeofday") or {}).get("label", "")
        cond_id, cond_name = map_condition_label(weather, timeofday)

        if cond_id is None:
            # Skip ambiguous/other samples to maintain clean class boundaries
            continue

        # Load image to compute physical metrics
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        h, w, _ = img_bgr.shape
        vis_metrics = compute_visibility_metrics(img_bgr)
        is_low_vis = is_derived_low_visibility(vis_metrics, lum_thresh, contrast_thresh)

        if is_low_vis:
            derived_low_vis_count += 1

        prepared_sample = {
            "image_id": sample_meta.get("_id", fname.split(".")[0]),
            "filepath": f"data/bdd100k/images/{fname}",
            "filename": fname,
            "width": w,
            "height": h,
            "raw_weather": weather,
            "raw_timeofday": timeofday,
            "condition_label": cond_id,
            "condition_name": cond_name,
            "mean_luminance": vis_metrics["mean_luminance"],
            "contrast_rms": vis_metrics["contrast_rms"],
            "dark_channel_mean": vis_metrics["dark_channel_mean"],
            "is_derived_low_visibility": is_low_vis,
            "detections": (sample_meta.get("detections") or {}).get("detections", [])
        }

        valid_prepared_samples.append(prepared_sample)
        condition_counts[cond_name] += 1

    logger.info(f"Total valid condition-labeled samples prepared: {len(valid_prepared_samples)}")
    logger.info("Condition Distribution:")
    for cond_name, count in sorted(condition_counts.items()):
        logger.info(f"  {cond_name:12s}: {count:5d} ({count/len(valid_prepared_samples)*100:5.1f}%)")
    logger.info(f"Derived Low-Visibility Subsets: {derived_low_vis_count} samples ({derived_low_vis_count/len(valid_prepared_samples)*100:5.1f}%)")

    # Save prepared dataset manifest
    prepared_manifest_path = metadata_dir / "prepared_samples.json"
    with open(prepared_manifest_path, "w") as f:
        json.dump({"samples": valid_prepared_samples}, f, indent=2)
    logger.info(f"Saved prepared manifest to {prepared_manifest_path}")

    # Generate Train (70%), Val (15%), Test (15%) splits
    train_s, val_s, test_s = create_splits(
        valid_prepared_samples,
        output_dir=splits_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed
    )

    logger.info(f"Splits generated successfully with seed {seed}:")
    logger.info(f"  Train: {len(train_s)} samples")
    logger.info(f"  Val  : {len(val_s)} samples")
    logger.info(f"  Test : {len(test_s)} samples")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    prepare_dataset(cfg, seed=args.seed)
