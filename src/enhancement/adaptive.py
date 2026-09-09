"""
Adaptive Enhancement Router (Pipeline C).
Maps predicted or oracle environmental condition cleanly to the condition-specific enhancer
without nested if/else branching.
"""

from typing import Dict, Any, Union
import numpy as np
from src.enhancement.base import BaseEnhancer
from src.enhancement.dawn_dusk import ClearPassThroughEnhancer, DawnDuskEnhancer
from src.enhancement.rain import ClassicalRainEnhancer
from src.enhancement.fog import FogDehazingEnhancer
from src.enhancement.night import LowLightNightEnhancer


class AdaptiveEnhancer(BaseEnhancer):
    """
    Adaptive Enhancement Router.
    Routes input image to the dedicated condition enhancer based on condition name or ID.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="adaptive_router")
        cfg = config or {}
        fog_cfg = cfg.get("fog", {})
        rain_cfg = cfg.get("rain", {})
        night_cfg = cfg.get("night", {})
        dawn_cfg = cfg.get("dawn_dusk", {})

        # Modular routing dictionary
        self.condition_to_enhancement: Dict[str, BaseEnhancer] = {
            "clear": ClearPassThroughEnhancer(),
            "rain": ClassicalRainEnhancer(
                bilateral_d=rain_cfg.get("bilateral_d", 7),
                bilateral_sigma_color=rain_cfg.get("bilateral_sigma_color", 50.0),
                bilateral_sigma_space=rain_cfg.get("bilateral_sigma_space", 50.0),
                unsharp_strength=rain_cfg.get("unsharp_strength", 1.2),
                clahe_clip=rain_cfg.get("clahe_clip", 1.5),
            ),
            "fog": FogDehazingEnhancer(
                patch_size=fog_cfg.get("patch_size", 15),
                omega=fog_cfg.get("omega", 0.95),
                t0=fog_cfg.get("t0", 0.1),
                clahe_clip=fog_cfg.get("clahe_clip", 1.5),
            ),
            "night": LowLightNightEnhancer(
                gamma=night_cfg.get("gamma", 0.60),
                clahe_clip=night_cfg.get("clahe_clip", 2.5),
                brightness_boost=night_cfg.get("brightness_boost", 1.15),
            ),
            "dawn_dusk": DawnDuskEnhancer(
                gamma=dawn_cfg.get("gamma", 0.85),
                clahe_clip=dawn_cfg.get("clahe_clip", 1.8),
                contrast_alpha=dawn_cfg.get("contrast_alpha", 1.1),
            )
        }

        # Numeric index mapping
        self.idx_to_name = {
            0: "clear",
            1: "rain",
            2: "fog",
            3: "night",
            4: "dawn_dusk"
        }

    def get_enhancer(self, condition: Union[str, int]) -> BaseEnhancer:
        """Resolves condition key to enhancer instance."""
        if isinstance(condition, int):
            condition_name = self.idx_to_name.get(condition, "clear")
        else:
            condition_name = str(condition).lower()
        return self.condition_to_enhancement.get(condition_name, self.condition_to_enhancement["clear"])

    def enhance(self, image_bgr: np.ndarray, condition: Union[str, int] = "clear") -> np.ndarray:
        enhancer = self.get_enhancer(condition)
        return enhancer.enhance(image_bgr)
