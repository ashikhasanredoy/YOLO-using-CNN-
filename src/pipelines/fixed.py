"""
Pipeline B: Fixed Enhancement.
Applies the identical CLAHE enhancement to every image regardless of condition,
then passes to YOLO object detection.
"""

import time
import os
from pathlib import Path
from typing import Dict, Any
import cv2
from ultralytics import YOLO
from src.enhancement.fixed import FixedCLAHEEnhancer


class FixedPipeline:
    """
    Pipeline B:
    Original Image -> Fixed CLAHE -> YOLO -> Object Detection -> Evaluation
    """
    def __init__(
        self,
        yolo_model: YOLO,
        enhancer: FixedCLAHEEnhancer = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.50
    ):
        self.yolo = yolo_model
        self.enhancer = enhancer or FixedCLAHEEnhancer()
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.name = "Pipeline B (Fixed CLAHE)"

    def process_image(self, image_path: str, actual_condition: str, sample_id: str = "") -> Dict[str, Any]:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Step 1: Fixed Enhancement
        t_enh_start = time.perf_counter()
        enhanced_bgr = self.enhancer.enhance(img_bgr)
        enhancement_time_ms = (time.perf_counter() - t_enh_start) * 1000.0

        # Step 2: YOLO Detection on enhanced image
        t_inf_start = time.perf_counter()
        results = self.yolo.predict(
            source=enhanced_bgr,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False
        )[0]
        inference_time_ms = (time.perf_counter() - t_inf_start) * 1000.0

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
            "predicted_condition": actual_condition,  # N/A for fixed
            "classifier_confidence": 1.0,
            "enhancement_method": self.enhancer.name,
            "enhancement_time_ms": enhancement_time_ms,
            "inference_time_ms": inference_time_ms,
            "num_detections": len(boxes_list),
            "detections": boxes_list,
            "enhanced_image": enhanced_bgr
        }
