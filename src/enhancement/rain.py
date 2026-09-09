"""
Rain Enhancement Module: Classical Rain Enhancement Baseline.
Edge-preserving bilateral filtering to attenuate streak artifacts,
unsharp masking for boundary restoration, and contrast recovery.
"""

import cv2
import numpy as np
from src.enhancement.base import BaseEnhancer


class ClassicalRainEnhancer(BaseEnhancer):
    """
    Classical Rain Enhancement Baseline:
    Combines edge-preserving bilateral filtering with high-boost unsharp masking
    and luminance contrast expansion.
    """
    def __init__(
        self,
        bilateral_d: int = 7,
        bilateral_sigma_color: float = 50.0,
        bilateral_sigma_space: float = 50.0,
        unsharp_strength: float = 1.2,
        clahe_clip: float = 1.5
    ):
        super().__init__(name="classical_rain_baseline")
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.unsharp_strength = unsharp_strength
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr

        # Step 1: Edge-preserving smoothing to attenuate rain streaks
        smoothed = cv2.bilateralFilter(
            image_bgr,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )

        # Step 2: Unsharp masking to accentuate object boundaries
        # Detail high-pass = image_bgr - smoothed
        blurred = cv2.GaussianBlur(smoothed, (0, 0), sigmaX=2.0)
        high_pass = cv2.addWeighted(smoothed, 1.0, blurred, -1.0, 0)
        sharpened = cv2.addWeighted(smoothed, 1.0, high_pass, self.unsharp_strength, 0)

        # Step 3: Contrast improvement in LAB space
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        enhanced_l = self.clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([enhanced_l, a, b]), cv2.COLOR_LAB2BGR)

        return enhanced
