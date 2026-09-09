"""
Night Enhancement Module: Low-Light Illumination & Contrast Enhancement.
Combines nonlinear Gamma correction, CLAHE, and controlled brightness normalization.
"""

import cv2
import numpy as np
from src.enhancement.base import BaseEnhancer


class LowLightNightEnhancer(BaseEnhancer):
    """
    Night condition low-light enhancement:
    Applies power-law (gamma) expansion to restore shadow details in low-illumination
    urban scenes, followed by CLAHE to avoid blowing out vehicle headlights and street lamps.
    """
    def __init__(
        self,
        gamma: float = 0.60,
        clahe_clip: float = 2.5,
        brightness_boost: float = 1.15
    ):
        super().__init__(name="night_low_light")
        self.gamma = gamma
        self.brightness_boost = brightness_boost
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))

        # Precompute gamma lookup table for speed
        inv_gamma = self.gamma
        self.lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr

        # Step 1: Gamma correction on RGB/BGR
        gamma_corrected = cv2.LUT(image_bgr, self.lut)

        # Step 2: Convert to LAB and apply CLAHE to L channel
        lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        enhanced_l = self.clahe.apply(l)

        # Step 3: Subtle controlled brightness scaling
        if self.brightness_boost != 1.0:
            enhanced_l = np.clip(enhanced_l.astype(np.float32) * self.brightness_boost, 0, 255).astype(np.uint8)

        enhanced_bgr = cv2.cvtColor(cv2.merge([enhanced_l, a, b]), cv2.COLOR_LAB2BGR)
        return enhanced_bgr
