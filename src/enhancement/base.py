"""
Base class for image enhancement algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseEnhancer(ABC):
    """Abstract base class for condition-specific and fixed enhancers."""
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Enhances an input BGR image (uint8, shape HxWxC).
        Returns the enhanced BGR image (uint8, shape HxWxC).
        """
        pass

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        return self.enhance(image_bgr)
