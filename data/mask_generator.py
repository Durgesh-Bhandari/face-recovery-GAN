"""data/mask_generator.py — Generate synthetic occlusion masks for training."""
import os
import random
import numpy as np
import cv2
from typing import List


class MaskGenerator:
    """Generates occlusion masks for training data augmentation.

    Supports:
    - Irregular masks (from NVIDIA dataset or random noise)
    - Rectangular block masks (sunglasses, masks, hands)
    - Half-face occlusion (left/right/top/bottom)
    """

    def __init__(
        self,
        mask_dir: str = "./data/masks",
        img_size: int = 256,
    ):
        self.img_size = img_size
        self.irregular_masks = []

        # Load NVIDIA irregular masks if available
        irr_dir = os.path.join(mask_dir, "irregular", "mask")
        if os.path.exists(irr_dir):
            for root, _, files in os.walk(irr_dir):
                for f in files:
                    if f.endswith((".png", ".jpg", ".bmp")):
                        self.irregular_masks.append(os.path.join(root, f))
            print(f"Loaded {len(self.irregular_masks)} irregular mask patterns")

    def random_irregular(self) -> np.ndarray:
        """Generate irregular mask using NVIDIA patterns or noise-based fallback."""
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        if self.irregular_masks and random.random() < 0.7:
            path = random.choice(self.irregular_masks)
            pattern = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if pattern is not None:
                pattern = cv2.resize(pattern, (self.img_size, self.img_size))
                mask = (pattern > 128).astype(np.float32)
                if random.random() < 0.3:
                    mask = 1.0 - mask
                return mask

        # Fallback: random polygon-based irregular mask
        num_vertex = random.randint(4, 12)
        angle = np.linspace(0, 2 * np.pi, num_vertex, endpoint=False)
        radii = np.random.uniform(30, 100, size=num_vertex)
        xs = self.img_size // 2 + (radii * np.cos(angle)).astype(int)
        ys = self.img_size // 2 + (radii * np.sin(angle)).astype(int)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1.0)
        return mask

    def random_rect(self, min_ratio: float = 0.05, max_ratio: float = 0.3) -> np.ndarray:
        """Generate rectangular occlusion (sunglasses/mask shape)."""
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        w = random.randint(int(self.img_size * min_ratio), int(self.img_size * max_ratio))
        h = random.randint(int(self.img_size * min_ratio), int(self.img_size * max_ratio))
        x = random.randint(0, self.img_size - w)
        y = random.randint(0, self.img_size - h)
        mask[y: y + h, x: x + w] = 1.0
        return mask

    def half_face(self) -> np.ndarray:
        """Generate half-face occlusion (left/right/top/bottom)."""
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        side = random.choice(["left", "right", "top", "bottom"])
        ratio = random.uniform(0.3, 0.6)
        if side == "left":
            mask[:, : int(self.img_size * ratio)] = 1.0
        elif side == "right":
            mask[:, int(self.img_size * (1 - ratio)):] = 1.0
        elif side == "top":
            mask[: int(self.img_size * ratio), :] = 1.0
        else:
            mask[int(self.img_size * (1 - ratio)):, :] = 1.0
        return mask

    def __call__(self, mask_types: List[str] = None) -> np.ndarray:
        """Generate a random occlusion mask.

        Args:
            mask_types: List of mask types to choose from.
                        Default: ["irregular", "rect", "half"]

        Returns:
            mask: float32 array of shape (H, W), 1=occluded, 0=visible
        """
        if mask_types is None:
            mask_types = ["irregular", "rect", "half"]
        method = random.choice(mask_types)
        if method == "half":
            mask = self.half_face()
        else:
            mask = getattr(self, f"random_{method}")()
        # Ensure minimum occlusion
        if mask.mean() < 0.02:
            mask[0:10, 0:10] = 1.0
        return mask
