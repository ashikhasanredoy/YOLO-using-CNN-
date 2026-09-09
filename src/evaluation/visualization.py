"""
Visualization Engine for Object Detection and Enhancement Experiments.
- Draws predicted bounding boxes, labels, and condition tags.
- Generates side-by-side comparisons (Baseline vs Fixed vs Adaptive vs Oracle).
- Generates research metric plots (mAP50, mAP50:95, Precision, Recall, F1, Condition-wise mAP).
"""

from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Vibrant distinct colors for object classes
COLOR_PALETTE = [
    (0, 255, 127),   # spring green
    (255, 140, 0),   # dark orange
    (30, 144, 255),  # dodger blue
    (255, 20, 147),  # deep pink
    (255, 215, 0),   # gold
    (147, 112, 219), # medium purple
    (0, 206, 209),   # dark turquoise
    (220, 20, 60),   # crimson
    (50, 205, 50),   # lime green
    (255, 69, 0)     # orange red
]


def draw_bounding_boxes(
    img_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    title_text: str = "",
    subtitle_text: str = ""
) -> np.ndarray:
    """
    Renders bounding boxes, class labels, confidence scores, and header banners.
    """
    vis = img_bgr.copy()
    h, w, _ = vis.shape

    # Draw detection boxes
    for det in detections:
        box = det["bbox_xyxy"]
        conf = det.get("confidence", 0.0)
        cls_name = det.get("class_name", "object")
        cls_id = det.get("class_id", 0)

        color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label_str = f"{cls_name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), color, -1)
        cv2.putText(vis, label_str, (x1 + 2, max(th + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Top banner for pipeline / condition info
    if title_text or subtitle_text:
        banner_h = 45 if subtitle_text else 30
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        vis = cv2.addWeighted(overlay, 0.75, vis, 0.25, 0)

        if title_text:
            cv2.putText(vis, title_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if subtitle_text:
            cv2.putText(vis, subtitle_text, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1, cv2.LINE_AA)

    return vis


def create_side_by_side_comparison(
    img_orig_bgr: np.ndarray,
    baseline_res: Dict[str, Any],
    fixed_res: Dict[str, Any],
    adaptive_res: Dict[str, Any],
    save_path: Path
):
    """
    Creates a 3-panel horizontal side-by-side visual comparison:
    [Baseline: Original] | [Pipeline B: Fixed CLAHE] | [Pipeline C: Adaptive Enhancement]
    """
    # Panel 1: Original / Baseline
    vis_base = draw_bounding_boxes(
        img_orig_bgr,
        baseline_res.get("detections", []),
        title_text=f"Baseline (Original) - Cond: {baseline_res.get('actual_condition', 'N/A')}",
        subtitle_text=f"Detections: {baseline_res.get('num_detections', 0)}"
    )

    # Panel 2: Fixed Enhancement
    fixed_img = fixed_res.get("enhanced_image", img_orig_bgr)
    vis_fixed = draw_bounding_boxes(
        fixed_img,
        fixed_res.get("detections", []),
        title_text="Fixed Enhancement (CLAHE)",
        subtitle_text=f"Detections: {fixed_res.get('num_detections', 0)}"
    )

    # Panel 3: Adaptive Enhancement
    adapt_img = adaptive_res.get("enhanced_image", img_orig_bgr)
    pred_c = adaptive_res.get("predicted_condition", "N/A")
    conf_c = adaptive_res.get("classifier_confidence", 1.0)
    enh_m = adaptive_res.get("enhancement_method", "none")
    vis_adapt = draw_bounding_boxes(
        adapt_img,
        adaptive_res.get("detections", []),
        title_text=f"Adaptive: Pred={pred_c} ({conf_c*100:.1f}%)",
        subtitle_text=f"Method: {enh_m} | Dets: {adaptive_res.get('num_detections', 0)}"
    )

    # Concatenate horizontally
    h_target = 480
    w_target = int(vis_base.shape[1] * (h_target / vis_base.shape[0]))

    r_base = cv2.resize(vis_base, (w_target, h_target))
    r_fixed = cv2.resize(vis_fixed, (w_target, h_target))
    r_adapt = cv2.resize(vis_adapt, (w_target, h_target))

    comparison_canvas = np.hstack([r_base, r_fixed, r_adapt])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), comparison_canvas)


def plot_metric_comparisons(df_final: pd.DataFrame, df_condition: pd.DataFrame, results_dir: Path):
    """
    Generates all requested research plots:
    - mAP50_comparison.png
    - mAP50_95_comparison.png
    - precision_comparison.png
    - recall_comparison.png
    - f1_comparison.png
    - condition_wise_map.png
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    pipelines = df_final["pipeline"].tolist()
    colors = ["#64748b", "#3b82f6", "#10b981", "#8b5cf6"][:len(pipelines)]

    # Single-metric comparison bar plots
    metrics = [
        ("map50", "mAP@50", "mAP50_comparison.png", (0, 1.0)),
        ("map50_95", "mAP@50:95", "mAP50_95_comparison.png", (0, 1.0)),
        ("precision", "Precision", "precision_comparison.png", (0, 1.0)),
        ("recall", "Recall", "recall_comparison.png", (0, 1.0)),
        ("f1", "F1-Score", "f1_comparison.png", (0, 1.0)),
    ]

    for col, title, fname, ylim in metrics:
        plt.figure(figsize=(7, 5))
        vals = df_final[col].tolist()
        bars = plt.bar(pipelines, vals, color=colors, width=0.55)
        plt.ylabel(title, fontsize=11, fontweight="bold")
        plt.title(f"Overall {title} Comparison Across Pipelines", fontsize=12, fontweight="bold")
        plt.ylim(0, max(max(vals) * 1.25, 0.1))
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.005, f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        plt.savefig(results_dir / fname, dpi=300)
        plt.close()

    # Condition-wise grouped bar plot for mAP@50
    plt.figure(figsize=(11, 6))
    conditions = [c for c in df_condition["condition"].unique() if c != "Overall"]
    all_pipes = df_condition["pipeline"].unique().tolist()
    
    x = np.arange(len(conditions))
    width = 0.8 / len(all_pipes)

    for i, pipe in enumerate(all_pipes):
        sub_df = df_condition[df_condition["pipeline"] == pipe]
        m50_vals = [sub_df[sub_df["condition"] == c]["map50"].values[0] if len(sub_df[sub_df["condition"] == c]) > 0 else 0.0 for c in conditions]
        plt.bar(x + (i - len(all_pipes) / 2.0 + 0.5) * width, m50_vals, width, label=pipe, color=colors[i % len(colors)])

    plt.xlabel("Adverse Environmental Condition", fontsize=11, fontweight="bold")
    plt.ylabel("mAP@50", fontsize=11, fontweight="bold")
    plt.title("Condition-Wise Object Detection Performance (mAP@50)", fontsize=13, fontweight="bold")
    plt.xticks(x, conditions, fontsize=10, fontweight="bold")
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / "condition_wise_map.png", dpi=300)
    plt.close()
