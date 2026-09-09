#!/usr/bin/env python3
"""
BDD100K Dataset Schema and Metadata Inspector.
Inspects the actual downloaded Hugging Face BDD100K dataset, printing real
splits, fields, weather tags, time-of-day categories, object detection labels,
and bounding box formats without fabricating any fields.
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from collections import Counter
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect BDD100K dataset schema and statistics")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def inspect_dataset(config: dict):
    dataset_cfg = config.get("dataset", {})
    root_dir = Path(dataset_cfg.get("root", "data/bdd100k"))
    annotations_file = root_dir / "annotations" / "samples.json"
    images_dir = root_dir / "images"
    manifest_file = root_dir / "metadata" / "selected_samples.json"

    if not annotations_file.exists():
        print(f"Error: Annotations file not found at {annotations_file}")
        sys.exit(1)

    with open(annotations_file, "r") as f:
        data = json.load(f)

    all_samples = data.get("samples", data)
    total_samples = len(all_samples)

    print("=" * 65)
    print(" BDD100K REAL DATASET SCHEMA & METADATA INSPECTION")
    print("=" * 65)
    print(f"Dataset root directory: {root_dir.resolve()}")
    print(f"Total samples in full metadata: {total_samples}")

    # Check local downloaded images
    local_images = list(images_dir.glob("*.jpg")) if images_dir.exists() else []
    print(f"Total downloaded images locally: {len(local_images)}")

    # Sample fields inspection
    first_sample = all_samples[0]
    print("\n--- Top-Level Sample Fields ---")
    for k, v in first_sample.items():
        val_type = type(v).__name__
        val_preview = str(v)[:60] + "..." if len(str(v)) > 60 else str(v)
        print(f"  • {k:15s} [{val_type}]: {val_preview}")

    # Inspection of metadata subfields
    print("\n--- Image Metadata Fields ---")
    img_meta = first_sample.get("metadata", {})
    for k, v in img_meta.items():
        print(f"  • {k:15s}: {v}")

    # Inspection of detection annotation fields
    print("\n--- Object Detection Annotation Schema ---")
    detections_obj = first_sample.get("detections", {})
    detections_list = detections_obj.get("detections", [])
    if detections_list:
        first_det = detections_list[0]
        for k, v in first_det.items():
            print(f"  • {k:15s}: {v}")
    else:
        print("  (No detections on sample 0)")

    # Aggregations across all samples
    weather_counts = Counter()
    time_counts = Counter()
    scene_counts = Counter()
    object_label_counts = Counter()
    total_detection_boxes = 0

    for s in all_samples:
        w = (s.get("weather") or {}).get("label", "missing")
        t = (s.get("timeofday") or {}).get("label", "missing")
        sc = (s.get("scene") or {}).get("label", "missing")
        weather_counts[w] += 1
        time_counts[t] += 1
        scene_counts[sc] += 1

        dets = (s.get("detections") or {}).get("detections", [])
        total_detection_boxes += len(dets)
        for d in dets:
            lbl = d.get("label", "unknown")
            object_label_counts[lbl] += 1

    print("\n--- Weather Distribution (Full Metadata) ---")
    for w, count in weather_counts.most_common():
        pct = (count / total_samples) * 100
        print(f"  {w:16s}: {count:6d} ({pct:5.1f}%)")

    print("\n--- Time-of-Day Distribution (Full Metadata) ---")
    for t, count in time_counts.most_common():
        pct = (count / total_samples) * 100
        print(f"  {t:16s}: {count:6d} ({pct:5.1f}%)")

    print("\n--- Scene Distribution (Full Metadata) ---")
    for sc, count in scene_counts.most_common():
        pct = (count / total_samples) * 100
        print(f"  {sc:16s}: {count:6d} ({pct:5.1f}%)")

    print(f"\n--- Object Detection Class Distribution (Total Bounding Boxes: {total_detection_boxes}) ---")
    for lbl, count in object_label_counts.most_common():
        pct = (count / total_detection_boxes) * 100 if total_detection_boxes else 0
        print(f"  {lbl:16s}: {count:6d} ({pct:5.1f}%)")

    # If selected benchmark subset exists, inspect it too
    if manifest_file.exists():
        with open(manifest_file, "r") as f:
            subset_data = json.load(f)
        subset_samples = subset_data.get("samples", [])
        print("\n" + "=" * 65)
        print(f" SELECTED BENCHMARK SUBSET INSPECTION ({len(subset_samples)} samples)")
        print("=" * 65)
        sub_weather = Counter((s.get("weather") or {}).get("label", "missing") for s in subset_samples)
        sub_time = Counter((s.get("timeofday") or {}).get("label", "missing") for s in subset_samples)
        print("Weather Breakdown:")
        for w, count in sub_weather.most_common():
            print(f"  {w:16s}: {count:5d}")
        print("Time-of-Day Breakdown:")
        for t, count in sub_time.most_common():
            print(f"  {t:16s}: {count:5d}")

    print("=" * 65)
    print(" Inspection complete. Zero synthetic metadata used.")
    print("=" * 65)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    inspect_dataset(cfg)
