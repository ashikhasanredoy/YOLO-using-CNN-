"""
Dawn/Dusk and Clear Enhancement Modules.
- Dawn/Dusk: Moderate illumination expansion and shadow balancing.
- Clear: Identity pass-through (no unnecessary alteration of clear daylight scenes).
"""

import cv2
import numpy as np
from src.enhancement.base import BaseEnhancer


class DawnDuskEnhancer(BaseEnhancer):
    """
    Dawn / Dusk condition enhancement:
    Balances low-angle solar glare, high dynamic range between horizon and shadowed roadway,
    and moderate underexposure.
    """
    def __init__(
        self,
        gamma: float = 0.85,
        clahe_clip: float = 1.8,
        contrast_alpha: float = 1.1
    ):
        super().__init__(name="dawn_dusk_illumination")
        self.gamma = gamma
        self.contrast_alpha = contrast_alpha
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        self.lut = np.array([((i / 255.0) ** self.gamma) * 255 for i in range(256)]).astype(np.uint8)

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr

        # Gamma adjustment
        gamma_adj = cv2.LUT(image_bgr, self.lut)

        # LAB CLAHE
        lab = cv2.cvtColor(gamma_adj, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        enhanced_l = self.clahe.apply(l)

        # Contrast scaling around midtones
        if self.contrast_alpha != 1.0:
            enhanced_l = np.clip(128 + self.contrast_alpha * (enhanced_l.astype(np.float32) - 128), 0, 255).astype(np.uint8)

        enhanced = cv2.cvtColor(cv2.merge([enhanced_l, a, b]), cv2.COLOR_LAB2BGR)
        return enhanced


class ClearPassThroughEnhancer(BaseEnhancer):
    """
    Clear condition: Returns the original image unmodified.
    Scientific rationale: Clear daylight imagery does not suffer from adverse atmospheric degradation;
    applying unnecessary filters risks injecting artifacts.
    """
    def __init__(self):
        super().__init__(name="clear_pass_through")

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        return image_bgr
