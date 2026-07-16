"""utils/metrics.py — Image quality metrics."""
import torch
import numpy as np
import scipy.linalg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """PSNR between two image tensors in [-1, 1]."""
    img1_np = ((img1.cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    img2_np = ((img2.cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    img1_np = img1_np.transpose(1, 2, 0)
    img2_np = img2_np.transpose(1, 2, 0)
    return peak_signal_noise_ratio(img1_np, img2_np, data_range=255)


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """SSIM between two image tensors in [-1, 1]."""
    img1_np = ((img1.cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    img2_np = ((img2.cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    img1_np = img1_np.transpose(1, 2, 0)
    img2_np = img2_np.transpose(1, 2, 0)
    return structural_similarity(img1_np, img2_np, channel_axis=2, data_range=255)


def calculate_fid(real_features, fake_features):
    """FID score between real and fake feature distributions.

    Args:
        real_features: (N, D) numpy array
        fake_features: (N, D) numpy array

    Returns:
        fid: float
    """
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
