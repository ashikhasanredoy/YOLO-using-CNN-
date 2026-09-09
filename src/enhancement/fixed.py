"""
Fixed Enhancement Pipeline (Pipeline B).
Applies a single, condition-agnostic CLAHE enhancement across all images.
"""

import cv2
import numpy as np
from src.enhancement.base import BaseEnhancer


class FixedCLAHEEnhancer(BaseEnhancer):
    """
    Fixed Enhancement baseline: Applies Contrast Limited Adaptive Histogram
    Equalization (CLAHE) to the luminance channel in LAB color space.
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        super().__init__(name="fixed_clahe")
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size)
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr

        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        enhanced_l = self.clahe.apply(l)
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr
