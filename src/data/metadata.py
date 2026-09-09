"""
Metadata extraction, condition mapping, and derived low-visibility classification
for BDD100K research pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2


CONDITION_NAMES = {
    0: "clear",
    1: "rain",
    2: "fog",
    3: "night",
    4: "dawn_dusk"
}

NAME_TO_CONDITION = {v: k for k, v in CONDITION_NAMES.items()}


def map_condition_label(weather_str: str, timeofday_str: str) -> Tuple[Optional[int], str]:
    """
    Deterministically maps BDD100K weather and time-of-day strings to condition class.
    
    Priority mapping:
      1. weather == 'foggy' -> 2 ('fog')
      2. weather == 'rainy' -> 1 ('rain')
      3. timeofday == 'night' -> 3 ('night')
      4. timeofday in ['dawn/dusk', 'dusk', 'dawn'] -> 4 ('dawn_dusk')
      5. timeofday == 'daytime' and weather in ['clear', 'partly cloudy', 'overcast'] -> 0 ('clear')
      
    Returns (class_id, class_name). Returns (None, 'other') if unmapped.
    """
    w = (weather_str or "").strip().lower()
    t = (timeofday_str or "").strip().lower()

    if w == "foggy":
        return 2, "fog"
    if w == "rainy":
        return 1, "rain"
    if t == "night":
        return 3, "night"
    if t in ["dawn/dusk", "dusk", "dawn"]:
        return 4, "dawn_dusk"
    if t == "daytime" and w in ["clear", "partly cloudy", "overcast"]:
        return 0, "clear"

    return None, "other"


def compute_visibility_metrics(image_bgr: np.ndarray) -> Dict[str, float]:
    """
    Computes objective image quality metrics to derive a measurable low-visibility proxy:
    - mean_luminance: Average pixel intensity across grayscale channel (0 to 255)
    - contrast_rms: Standard deviation of pixel intensities (0 to ~128)
    - dark_channel_mean: Mean intensity of the dark channel prior
    """
    if image_bgr is None or image_bgr.size == 0:
        return {"mean_luminance": 0.0, "contrast_rms": 0.0, "dark_channel_mean": 0.0}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Approximate dark channel with 15x15 min filter
    min_channel = np.min(image_bgr, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(min_channel, kernel)
    dark_mean = float(np.mean(dark_channel))

    return {
        "mean_luminance": mean_lum,
        "contrast_rms": contrast,
        "dark_channel_mean": dark_mean
    }


def is_derived_low_visibility(
    metrics: Dict[str, float],
    luminance_threshold: float = 45.0,
    contrast_threshold: float = 35.0
) -> bool:
    """
    Scientifically defines a derived low-visibility sample based on measured physical properties:
    Mean luminance < luminance_threshold OR Contrast RMS < contrast_threshold.
    
    This is kept strictly separate from ground-truth metadata tags.
    """
    return (
        metrics["mean_luminance"] < luminance_threshold or
        metrics["contrast_rms"] < contrast_threshold
    )
