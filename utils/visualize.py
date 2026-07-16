"""utils/visualize.py — Visualization helpers for TensorBoard."""
import numpy as np
import cv2
import torch


def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    """Convert [-1, 1] tensor to [0, 255] uint8 numpy."""
    return ((tensor.cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)


def make_grid(tensors, nrow=4):
    """Make grid of image tensors for TensorBoard.

    Args:
        tensors: list of (B, C, H, W) tensors in [-1, 1]

    Returns:
        grid: (3, H*nrow, W*len(tensors)) numpy array
    """
    B = tensors[0].shape[0]
    rows = []
    for i in range(min(nrow, B)):
        row = np.concatenate([
            tensor_to_np(t[i]) for t in tensors
        ], axis=2)
        rows.append(row)
    return np.concatenate(rows, axis=1)


def draw_mask_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay red mask on image for visualization."""
    overlay = img.copy()
    overlay[:, :, 0] = np.where(mask > 0.5, 255, overlay[:, :, 0])
    return overlay
