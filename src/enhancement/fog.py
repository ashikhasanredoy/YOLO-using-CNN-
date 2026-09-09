"""
Fog Enhancement Module: Dark Channel Prior (DCP) Dehazing.
Reproducible implementation of He et al. DCP with guided transmission refinement.
"""

import cv2
import numpy as np
from src.enhancement.base import BaseEnhancer


class FogDehazingEnhancer(BaseEnhancer):
    """
    Reproducible Fog Dehazing via Dark Channel Prior (DCP) and atmospheric recovery.
    """
    def __init__(
        self,
        patch_size: int = 15,
        omega: float = 0.95,
        t0: float = 0.1,
        clahe_clip: float = 1.5
    ):
        super().__init__(name="fog_dehazing_dcp")
        self.patch_size = patch_size
        self.omega = omega
        self.t0 = t0
        self.clahe_clip = clahe_clip
        if clahe_clip > 0:
            self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        else:
            self.clahe = None

    def _dark_channel(self, img_norm: np.ndarray) -> np.ndarray:
        """Min-filter across color channels followed by morphological erosion."""
        min_c = np.min(img_norm, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.patch_size, self.patch_size))
        dark = cv2.erode(min_c, kernel)
        return dark

    def _estimate_atmospheric_light(self, img_norm: np.ndarray, dark: np.ndarray) -> np.ndarray:
        """Estimates global atmospheric light A from top 0.1% brightest pixels in dark channel."""
        h, w = dark.shape
        num_pixels = h * w
        top_k = max(int(num_pixels * 0.001), 1)

        dark_flat = dark.ravel()
        indices = np.argpartition(dark_flat, -top_k)[-top_k:]

        img_flat = img_norm.reshape(-1, 3)
        brightest_pixels = img_flat[indices]
        # Choose pixel with highest intensity sum
        best_idx = np.argmax(np.sum(brightest_pixels, axis=1))
        atmospheric_light = brightest_pixels[best_idx]
        return np.maximum(atmospheric_light, 0.1)

    def _estimate_transmission(self, img_norm: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Estimates coarse transmission map."""
        normalized = img_norm / A
        dark_normalized = self._dark_channel(normalized)
        transmission = 1.0 - self.omega * dark_normalized
        return transmission

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr

        # Normalize image to [0, 1]
        img_norm = image_bgr.astype(np.float32) / 255.0

        # Step 1: Dark channel
        dark = self._dark_channel(img_norm)

        # Step 2: Atmospheric light A
        A = self._estimate_atmospheric_light(img_norm, dark)

        # Step 3: Raw transmission map
        raw_t = self._estimate_transmission(img_norm, A)

        # Step 4: Refine transmission using edge-preserving bilateral filter
        refined_t = cv2.bilateralFilter(raw_t.astype(np.float32), d=9, sigmaColor=0.1, sigmaSpace=15.0)
        refined_t = np.clip(refined_t, self.t0, 1.0)
        refined_t = np.expand_dims(refined_t, axis=2)

        # Step 5: Radiance recovery J = (I - A) / t + A
        recovered = (img_norm - A) / refined_t + A
        recovered = np.clip(recovered * 255.0, 0, 255).astype(np.uint8)

        # Step 6: Subtle CLAHE to restore local scene contrast
        if self.clahe is not None:
            lab = cv2.cvtColor(recovered, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = self.clahe.apply(l)
            recovered = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)

        return recovered
