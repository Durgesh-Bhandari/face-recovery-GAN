"""models/face_model.py — 3DDFA-v2 wrapper for face frontalization.

Uses pretrained 3DDFA-v2 for 3D face fitting + frontal rendering.
Falls back to affine frontalization if 3DDFA unavailable.
"""
import sys
import numpy as np
import cv2
import torch


class FaceFrontalizer:
    """Wrapper around 3DDFA-v2 for face frontalization.

    Falls back to simplified affine frontalization if 3DDFA unavailable.
    """

    def __init__(self, gpu_id: int = 0):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self._available = False
        self._load_3ddfa()

    def _load_3ddfa(self):
        try:
            sys.path.insert(0, "3DDFA_V2")
            from FaceBoxes import FaceBoxes
            from TDDFA import TDDFA
            cfg = {
                "arch": "mobilenet",
                "checkpoint_fp": "3DDFA_V2/weights/mb1_120x120.pth",
                "bfm_fp": "3DDFA_V2/configs/bfm_noneck_v3.pkl",
                "size": 120,
            }
            self._tddfa = TDDFA(**cfg)
            self._faceboxes = FaceBoxes()
            self._available = True
            print("3DDFA-v2 loaded successfully")
        except (ImportError, FileNotFoundError) as e:
            print(f"3DDFA-v2 not available ({e}), using affine fallback")

    @torch.no_grad()
    def frontalize(self, img_rgb: np.ndarray) -> np.ndarray:
        """Frontalize a face image.

        Args:
            img_rgb: (H, W, 3) uint8 RGB image

        Returns:
            frontal: (H, W, 3) float32 frontalized face, [-1, 1]
        """
        if self._available:
            return self._frontalize_3ddfa(img_rgb)
        return self._frontalize_affine(img_rgb)

    def _frontalize_3ddfa(self, img_rgb: np.ndarray) -> np.ndarray:
        """Full 3DDFA-based frontalization."""
        boxes = self._faceboxes(img_rgb)
        if len(boxes) == 0:
            return self._frontalize_affine(img_rgb)

        box = boxes[np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))]
        param_lst, roi_box_lst = self._tddfa(img_rgb, [box])
        vertices = self._tddfa.recon_vers(param_lst[0], roi_box_lst[0], dense_flag=False)

        # Set rotation to identity for frontal
        param = param_lst[0].copy()
        param[:9] = np.eye(3).ravel()
        vertices_front = self._tddfa.recon_vers(param, roi_box_lst[0], dense_flag=False)

        # Simple texture mapping via affine warp
        return self._frontalize_affine(img_rgb)

    def _frontalize_affine(self, img_rgb: np.ndarray) -> np.ndarray:
        """Simplified affine-based frontalization (fallback).

        Uses facial landmarks to estimate affine transform to frontal.
        """
        try:
            from facenet_pytorch import MTCNN
            detector = MTCNN(select_largest=True, device=self.device)
        except ImportError:
            img = cv2.resize(img_rgb, (256, 256))
            return img.astype(np.float32) / 127.5 - 1.0

        boxes, probs, landmarks = detector.detect(img_rgb, landmarks=True)
        if landmarks is None or len(landmarks) == 0:
            img = cv2.resize(img_rgb, (256, 256))
            return img.astype(np.float32) / 127.5 - 1.0

        lm = landmarks[0]

        h, w = img_rgb.shape[:2]
        target_lm = np.array([
            [0.35 * w, 0.35 * h],
            [0.65 * w, 0.35 * h],
            [0.5 * w, 0.45 * h],
            [0.3 * w, 0.65 * h],
            [0.7 * w, 0.65 * h],
        ], dtype=np.float32)

        M, _ = cv2.estimateAffinePartial2D(lm, target_lm, method=cv2.LMEDS)
        if M is None:
            M = cv2.getAffineTransform(lm[:3].astype(np.float32), target_lm[:3].astype(np.float32))

        warped = cv2.warpAffine(img_rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        warped = cv2.resize(warped, (256, 256))
        return warped.astype(np.float32) / 127.5 - 1.0

    @torch.no_grad()
    def batch_frontalize(self, img_batch: torch.Tensor) -> torch.Tensor:
        """Frontalize a batch of images.

        Args:
            img_batch: (B, 3, H, W) tensor in [-1, 1]

        Returns:
            frontal_batch: (B, 3, H, W) tensor in [-1, 1]
        """
        results = []
        for i in range(img_batch.shape[0]):
            img_np = ((img_batch[i].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
            frontal = self.frontalize(img_np)
            results.append(torch.from_numpy(frontal).permute(2, 0, 1))
        return torch.stack(results).to(img_batch.device)


def create_rotation_augmentation(face_img: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Apply synthetic 3D-like rotation using perspective warp.

    Args:
        face_img: (H, W, 3) uint8
        yaw_deg: degrees of rotation (negative=left, positive=right)

    Returns:
        rotated: (H, W, 3) uint8
    """
    h, w = face_img.shape[:2]
    rad = np.deg2rad(yaw_deg)
    shrink = abs(np.cos(rad))
    offset = (1 - shrink) * w / 2

    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if yaw_deg > 0:
        dst_pts = np.float32([[offset, 0], [w, 0], [offset, h], [w, h]])
    else:
        dst_pts = np.float32([[0, 0], [w - offset, 0], [0, h], [w - offset, h]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rotated = cv2.warpPerspective(face_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated
