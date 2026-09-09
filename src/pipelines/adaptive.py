"""
Pipeline C: Adaptive Enhancement.
Original Image -> PyTorch CNN Condition Classifier -> Predicted Condition ->
Condition-specific Enhancement -> YOLO -> Object Detection.
"""

import time
import os
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

from src.models.condition_cnn import build_condition_classifier, get_device
from src.data.dataset import get_transforms
from src.data.metadata import CONDITION_NAMES
from src.enhancement.adaptive import AdaptiveEnhancer


class AdaptivePipeline:
    """
    Pipeline C:
    Original Image -> PyTorch CNN -> Condition -> Condition Enhancement -> YOLO -> Metrics
    """
    def __init__(
        self,
        yolo_model: YOLO,
        classifier_weights_path: str,
        classifier_name: str = "resnet18",
        enhancement_config: dict = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.50,
        device: torch.device = None
    ):
        self.yolo = yolo_model
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device or get_device()
        self.name = "Pipeline C (Adaptive Enhancement)"

        # Initialize condition classifier
        self.classifier = build_condition_classifier(classifier_name, num_classes=5, pretrained=False)
        ckpt = torch.load(classifier_weights_path, map_location=self.device)
        self.classifier.load_state_dict(ckpt.get("model_state_dict", ckpt))
        self.classifier.to(self.device)
        self.classifier.eval()

        self.transform = get_transforms(image_size=224, is_train=False)

        # Initialize adaptive router
        self.adaptive_enhancer = AdaptiveEnhancer(enhancement_config)

    def predict_condition(self, img_pil: Image.Image) -> tuple[int, str, float]:
        """Runs CNN inference to predict condition and confidence."""
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.classifier(tensor)
            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            pred_id = int(pred.cpu().item())
            confidence = float(conf.cpu().item())
            pred_name = CONDITION_NAMES.get(pred_id, "clear")
        return pred_id, pred_name, confidence

    def process_image(self, image_path: str, actual_condition: str, sample_id: str = "") -> Dict[str, Any]:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        img_rgb_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # Step 1: CNN Condition Prediction
        t_cls_start = time.perf_counter()
        pred_id, pred_name, confidence = self.predict_condition(img_rgb_pil)
        cls_time_ms = (time.perf_counter() - t_cls_start) * 1000.0

        # Step 2: Route to condition-specific enhancer
        enhancer = self.adaptive_enhancer.get_enhancer(pred_name)
        t_enh_start = time.perf_counter()
        enhanced_bgr = enhancer.enhance(img_bgr)
        enh_time_ms = (time.perf_counter() - t_enh_start) * 1000.0

        # Step 3: YOLO Detection
        t_inf_start = time.perf_counter()
        results = self.yolo.predict(
            source=enhanced_bgr,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False
        )[0]
        inf_time_ms = (time.perf_counter() - t_inf_start) * 1000.0

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
            "predicted_condition": pred_name,
            "classifier_confidence": confidence,
            "classifier_time_ms": cls_time_ms,
            "enhancement_method": enhancer.name,
            "enhancement_time_ms": enh_time_ms,
            "inference_time_ms": inf_time_ms,
            "total_latency_ms": cls_time_ms + enh_time_ms + inf_time_ms,
            "num_detections": len(boxes_list),
            "detections": boxes_list,
            "enhanced_image": enhanced_bgr
        }
