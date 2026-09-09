"""
Pipeline A: Baseline / No Enhancement.
Passes original BDD100K images directly to YOLO without any image modification.
Serves as the empirical control group.
"""

import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from ultralytics import YOLO


class BaselinePipeline:
    """
    Pipeline A:
    Original Image -> YOLO -> Object Detection -> Evaluation
    """
    def __init__(self, yolo_model: YOLO, conf_thresh: float = 0.25, iou_thresh: float = 0.50):
        self.yolo = yolo_model
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.name = "Pipeline A (Baseline)"

    def process_image(self, image_path: str, actual_condition: str, sample_id: str = "") -> Dict[str, Any]:
        """
        Executes YOLO inference directly on original image.
        """
        t0 = time.perf_counter()
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Direct YOLO inference
        results = self.yolo.predict(
            source=img_bgr,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False
        )[0]
        inference_time = time.perf_counter() - t0

        boxes_list = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.yolo.names.get(cls_id, str(cls_id))
            boxes_list.append({
                "bbox_xyxy": xyxy,
                "confidence": conf,
                "class_id": cls_id,
                "class_name": cls_name
            })

        return {
            "sample_id": sample_id or Path(image_path).stem,
            "filename": os.path.basename(image_path),
            "actual_condition": actual_condition,
            "predicted_condition": actual_condition,  # N/A for baseline
            "classifier_confidence": 1.0,
            "enhancement_method": "none",
            "enhancement_time_ms": 0.0,
            "inference_time_ms": inference_time * 1000.0,
            "num_detections": len(boxes_list),
            "detections": boxes_list
        }
