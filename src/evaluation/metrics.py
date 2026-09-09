"""
Object Detection Evaluation Metrics Engine.
Computes mAP@50, mAP@50:95, Precision, Recall, F1-Score,
condition-wise performance breakdowns, and pipeline relative deltas.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def evaluate_detections_for_image(
    pred_boxes: List[Dict[str, Any]],
    gt_boxes: List[Dict[str, Any]],
    iou_thresh: float = 0.50
) -> Tuple[int, int, int]:
    """
    Matches predicted boxes with ground truth boxes at a given IoU threshold.
    Returns (True Positives, False Positives, False Negatives).
    """
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    # Sort predictions by descending confidence
    sorted_preds = sorted(pred_boxes, key=lambda x: x.get("confidence", 0.0), reverse=True)
    gt_matched = [False] * len(gt_boxes)

    tp = 0
    fp = 0

    for pred in sorted_preds:
        p_box = pred["bbox_xyxy"]
        p_cls = pred.get("class_id")

        best_iou = 0.0
        best_gt_idx = -1

        for idx, gt in enumerate(gt_boxes):
            if gt_matched[idx]:
                continue
            # Check class match if available
            g_cls = gt.get("class_id")
            if p_cls is not None and g_cls is not None and p_cls != g_cls:
                continue

            iou = compute_iou(p_box, gt["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

    fn = len(gt_boxes) - sum(gt_matched)
    return tp, fp, fn


def calculate_metrics_summary(pipeline_results: List[Dict[str, Any]], ground_truth: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    """
    Calculates overall mAP@50, mAP@50:95, Precision, Recall, and F1 across all evaluated images.
    """
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    aps = []

    tp_50, fp_50, fn_50 = 0, 0, 0

    for iou_th in iou_thresholds:
        tp_th, fp_th, fn_th = 0, 0, 0
        for res in pipeline_results:
            fname = res["filename"]
            gts = ground_truth.get(fname, [])
            preds = res.get("detections", [])
            tp, fp, fn = evaluate_detections_for_image(preds, gts, iou_thresh=iou_th)
            tp_th += tp
            fp_th += fp
            fn_th += fn

            if np.isclose(iou_th, 0.50):
                tp_50 += tp
                fp_50 += fp
                fn_50 += fn

        prec_th = tp_th / (tp_th + fp_th + 1e-8)
        rec_th = tp_th / (tp_th + fn_th + 1e-8)
        # Standard approx AP = prec * rec
        ap_th = (prec_th * rec_th) if (prec_th + rec_th) > 0 else 0.0
        aps.append(ap_th)

    precision = tp_50 / (tp_50 + fp_50 + 1e-8)
    recall = tp_50 / (tp_50 + fn_50 + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
    map50 = aps[0] if aps else 0.0
    map50_95 = float(np.mean(aps)) if aps else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "map50": float(map50),
        "map50_95": float(map50_95)
    }


def compute_condition_wise_metrics(
    pipeline_results: List[Dict[str, Any]],
    ground_truth: Dict[str, List[Dict[str, Any]]],
    pipeline_name: str
) -> pd.DataFrame:
    """
    Generates condition-wise performance breakdown:
    Clear, Rain, Fog, Night, Dawn/Dusk, and Overall.
    """
    # Group results by condition
    conditions = ["clear", "rain", "fog", "night", "dawn_dusk"]
    cond_buckets = {c: [] for c in conditions}

    for res in pipeline_results:
        cond = res.get("actual_condition", "clear").lower()
        if cond in cond_buckets:
            cond_buckets[cond].append(res)

    rows = []
    # Condition breakdowns
    for cond in conditions:
        bucket = cond_buckets[cond]
        if bucket:
            metrics = calculate_metrics_summary(bucket, ground_truth)
            rows.append({
                "condition": cond.capitalize() if cond != "dawn_dusk" else "Dawn/Dusk",
                "pipeline": pipeline_name,
                "num_samples": len(bucket),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
                "map50": round(metrics["map50"], 4),
                "map50_95": round(metrics["map50_95"], 4)
            })
        else:
            rows.append({
                "condition": cond.capitalize() if cond != "dawn_dusk" else "Dawn/Dusk",
                "pipeline": pipeline_name,
                "num_samples": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "map50": 0.0, "map50_95": 0.0
            })

    # Overall
    overall_metrics = calculate_metrics_summary(pipeline_results, ground_truth)
    rows.append({
        "condition": "Overall",
        "pipeline": pipeline_name,
        "num_samples": len(pipeline_results),
        "precision": round(overall_metrics["precision"], 4),
        "recall": round(overall_metrics["recall"], 4),
        "f1": round(overall_metrics["f1"], 4),
        "map50": round(overall_metrics["map50"], 4),
        "map50_95": round(overall_metrics["map50_95"], 4)
    })

    return pd.DataFrame(rows)


def compute_relative_improvements(baseline_val: float, new_val: float) -> Tuple[float, float]:
    """
    Computes absolute delta and percentage improvement:
    delta = new_val - baseline_val
    percentage = ((new_val - baseline_val) / baseline_val) * 100
    """
    delta = new_val - baseline_val
    if baseline_val > 0:
        pct = (delta / baseline_val) * 100.0
    else:
        pct = 0.0
    return delta, pct
