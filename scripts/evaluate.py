#!/usr/bin/env python3
"""
Master Evaluation and Comparative Analysis Script.
Loads detection predictions from Baseline, Fixed, Adaptive, and Oracle pipelines,
computes standard object detection metrics (mAP@50, mAP@50:95, Precision, Recall, F1),
breaks results down condition-by-condition, calculates relative improvement deltas,
generates comparison charts, and saves side-by-side visual panels.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.comparison import generate_research_results
from src.evaluation.visualization import plot_metric_comparisons, create_side_by_side_comparison
from src.enhancement.fixed import FixedCLAHEEnhancer
from src.enhancement.adaptive import AdaptiveEnhancer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and compare object detection pipelines")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--num-visualizations", type=int, default=15, help="Number of side-by-side sample visualizations")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_ground_truth(test_samples: list) -> dict:
    """
    Constructs ground truth dictionary: {filename: [{'bbox_xyxy': [x1, y1, x2, y2], 'class_name': str, 'class_id': int}]}
    """
    gt_dict = {}
    for s in test_samples:
        fname = s["filename"]
        img_w = s.get("width", 1280)
        img_h = s.get("height", 720)
        dets = s.get("detections", [])

        box_list = []
        for d in dets:
            bbox = d.get("bounding_box", [])  # [x_min, y_min, w, h] normalized
            if len(bbox) != 4:
                continue
            x1 = bbox[0] * img_w
            y1 = bbox[1] * img_h
            x2 = (bbox[0] + bbox[2]) * img_w
            y2 = (bbox[1] + bbox[3]) * img_h
            box_list.append({
                "bbox_xyxy": [x1, y1, x2, y2],
                "class_name": d.get("label", "object"),
                "class_id": None
            })
        gt_dict[fname] = box_list
    return gt_dict


def evaluate_all(config: dict, num_visualizations: int = 15):
    dataset_cfg = config.get("dataset", {})
    root_dir = Path(dataset_cfg.get("root", "data/bdd100k"))
    test_json = root_dir / "splits" / "test.json"
    results_dir = Path(config.get("paths", {}).get("results", "results"))
    vis_dir = Path(config.get("paths", {}).get("visualizations", "results/visualizations"))

    results_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not test_json.exists():
        logger.error(f"Test split not found at {test_json}")
        sys.exit(1)

    with open(test_json, "r") as f:
        test_data = json.load(f)
    test_samples = test_data.get("samples", test_data)
    ground_truth = load_ground_truth(test_samples)

    # Load detection results
    pipe_files = {
        "Baseline": results_dir / "baseline_detections.json",
        "Fixed": results_dir / "fixed_detections.json",
        "Adaptive": results_dir / "adaptive_detections.json",
        "Oracle": results_dir / "oracle_detections.json"
    }

    results_dict = {}
    for pipe_name, fpath in pipe_files.items():
        if fpath.exists():
            with open(fpath, "r") as f:
                results_dict[pipe_name] = json.load(f)
            logger.info(f"Loaded {len(results_dict[pipe_name])} detections for {pipe_name}")
        else:
            logger.warning(f"Detection file not found for {pipe_name} at {fpath}")

    if not results_dict:
        logger.error("No pipeline detection files found to evaluate.")
        sys.exit(1)

    # Generate research tables and summary
    logger.info("Computing metrics, condition-wise breakdown, and ablation tables...")
    df_final, df_condition, df_ablation, summary_text = generate_research_results(
        results_dict=results_dict,
        ground_truth=ground_truth,
        results_dir=results_dir
    )

    # Generate plots
    logger.info("Generating research comparison plots...")
    plot_metric_comparisons(df_final, df_condition, results_dir=results_dir)

    # Generate side-by-side visual comparisons
    if "Baseline" in results_dict and "Fixed" in results_dict and "Adaptive" in results_dict:
        logger.info(f"Generating {num_visualizations} side-by-side comparative visualizations in {vis_dir}...")
        fixed_enhancer = FixedCLAHEEnhancer()
        adaptive_enhancer = AdaptiveEnhancer(config.get("enhancement", {}))

        # Select diverse set across conditions
        selected_vis_samples = test_samples[:num_visualizations]
        base_map = {r["filename"]: r for r in results_dict["Baseline"]}
        fixed_map = {r["filename"]: r for r in results_dict["Fixed"]}
        adapt_map = {r["filename"]: r for r in results_dict["Adaptive"]}

        for idx, s in enumerate(tqdm(selected_vis_samples, desc="Rendering side-by-side panels")):
            fname = s["filename"]
            img_path = str(root_dir / "images" / fname)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            r_base = base_map.get(fname, {"detections": []})
            r_fixed = fixed_map.get(fname, {"detections": []})
            r_adapt = adapt_map.get(fname, {"detections": []})

            # Re-enhance images for visual canvas
            fixed_img = fixed_enhancer.enhance(img_bgr)
            adapt_pred = r_adapt.get("predicted_condition", "clear")
            adapt_img = adaptive_enhancer.enhance(img_bgr, condition=adapt_pred)

            r_fixed["enhanced_image"] = fixed_img
            r_adapt["enhanced_image"] = adapt_img

            save_file = vis_dir / f"comparison_{idx+1:03d}_{Path(fname).stem}.jpg"
            create_side_by_side_comparison(
                img_orig_bgr=img_bgr,
                baseline_res=r_base,
                fixed_res=r_fixed,
                adaptive_res=r_adapt,
                save_path=save_file
            )

    print("\n" + summary_text)
    logger.info("Evaluation and report generation completed successfully.")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    evaluate_all(cfg, num_visualizations=args.num_visualizations)
