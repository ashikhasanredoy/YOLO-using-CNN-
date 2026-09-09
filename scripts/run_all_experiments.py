#!/usr/bin/env python3
"""
Master End-to-End Research Runner.
Executes the complete experimental study sequentially with a single command:
1. Dataset Validation & Schema Inspection
2. Annotation Preparation & Conversion
3. Condition Classifier Training & Evaluation
4. Pipeline A (Baseline: No Enhancement)
5. Pipeline B (Fixed Enhancement: CLAHE)
6. Pipeline C (Adaptive Enhancement: CNN -> Targeted Enhancer -> YOLO)
7. Pipeline D (Oracle: Ground-Truth Condition -> Targeted Enhancer -> YOLO)
8. Comprehensive Metrics Compilation & Hypothesis Testing
9. Comparative Research Visualizations & Plots
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inspect_dataset import inspect_dataset
from scripts.prepare_dataset import prepare_dataset
from scripts.convert_annotations import convert_annotations
from scripts.train_condition_classifier import train_condition_classifier
from scripts.evaluate_condition_classifier import evaluate_condition_classifier
from scripts.run_baseline import run_baseline
from scripts.run_fixed import run_fixed
from scripts.run_adaptive import run_adaptive
from scripts.run_oracle import run_oracle
from scripts.evaluate import evaluate_all

(PROJECT_ROOT / "results").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "results" / "experiment.log", mode="a")
    ]
)
logger = logging.getLogger("AdaptiveYOLOResearch")


def main():
    config_file = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 70)
    logger.info("STARTING COMPLETE ADAPTIVE YOLO RESEARCH EXPERIMENT SUITE")
    logger.info("=" * 70)

    # 1. Dataset Inspection
    logger.info("\n>>> STEP 1: Inspecting Dataset Schema...")
    inspect_dataset(config)

    # 2. Dataset Preparation & Annotation Conversion
    logger.info("\n>>> STEP 2: Preparing Dataset and YOLO Annotations...")
    prepare_dataset(config)
    convert_annotations(config)

    # 3. Condition Classifier Training & Evaluation
    best_ckpt = PROJECT_ROOT / "models" / "condition_classifier_best.pt"
    if not best_ckpt.exists():
        logger.info("\n>>> STEP 3A: Training Condition Classifier CNN...")
        train_condition_classifier(config)
    else:
        logger.info(f"\n>>> STEP 3A: Found existing condition classifier at {best_ckpt}")

    logger.info("\n>>> STEP 3B: Evaluating Condition Classifier...")
    evaluate_condition_classifier(config, checkpoint_path=str(best_ckpt))

    # 4. Pipeline A (Baseline)
    logger.info("\n>>> STEP 4: Running Pipeline A (Baseline)...")
    run_baseline(config)

    # 5. Pipeline B (Fixed Enhancement)
    logger.info("\n>>> STEP 5: Running Pipeline B (Fixed Enhancement)...")
    run_fixed(config)

    # 6. Pipeline C (Adaptive Enhancement)
    logger.info("\n>>> STEP 6: Running Pipeline C (Adaptive Enhancement)...")
    run_adaptive(config, checkpoint_path=str(best_ckpt))

    # 7. Pipeline D (Oracle Condition)
    logger.info("\n>>> STEP 7: Running Pipeline D (Oracle Enhancement)...")
    run_oracle(config)

    # 8. Evaluation, Visualizations & Summary
    logger.info("\n>>> STEP 8: Master Evaluation, Condition Breakdown & Visualization...")
    evaluate_all(config, num_visualizations=15)

    logger.info("=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    logger.info("Results saved in results/:")
    logger.info("  - results/final_comparison.csv")
    logger.info("  - results/condition_wise_comparison.csv")
    logger.info("  - results/ablation_results.csv")
    logger.info("  - results/experiment_summary.txt")
    logger.info("  - results/visualizations/ (Side-by-side comparison images)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
