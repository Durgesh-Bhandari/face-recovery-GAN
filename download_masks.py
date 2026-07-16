"""download_masks.py — Download/generate training occlusion masks.

Sources:
  - kaggle: NVIDIA irregular masks from Kaggle (recommended)
  - local:  Generate random polygon masks (fallback, no download needed)

Usage:
    python download_masks.py --root ./data/masks --source kaggle
    python download_masks.py --root ./data/masks --source local
"""
import os
import argparse
import sys
import shutil
import subprocess
from pathlib import Path


def ensure(pkg):
    try:
        __import__(pkg)
        return True
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True


def download_kaggle(root):
    ensure("kagglehub")
    import kagglehub
    dataset = "nightbot69/irregular-mask-32k"
    print(f"Downloading from Kaggle ({dataset})...")
    path = kagglehub.dataset_download(dataset)
    print(f"Kaggle cache: {path}")
    # Find mask images
    masks_src = Path(path) / "irregular_mask" / "mask"
    if not masks_src.exists():
        masks_src = Path(path) / "mask"
    if not masks_src.exists():
        # Search recursively
        found = list(Path(path).rglob("*.png")) + list(Path(path).rglob("*.jpg"))
        if not found:
            print(f"No mask images found in {path}")
            return False
        masks_src = found[0].parent
    target = Path(root) / "irregular" / "mask"
    target.mkdir(parents=True, exist_ok=True)
    for f in masks_src.iterdir():
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            shutil.copy2(f, target / f.name)
    count = len(list(target.iterdir()))
    print(f"Copied {count} mask images to {target}")
    return True


def generate_local(root, count=50):
    target = os.path.join(root, "irregular", "mask")
    os.makedirs(target, exist_ok=True)
    import cv2
    import numpy as np
    rng = np.random.RandomState(42)
    for i in range(count):
        mask = np.zeros((512, 512), dtype=np.uint8)
        pts = rng.randint(50, 462, size=(rng.randint(5, 12), 2))
        cv2.fillPoly(mask, [pts], 255)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(target, f"mask_{i:04d}.png"), mask)
    print(f"Generated {count} fallback mask patterns in {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/masks")
    parser.add_argument("--source", default="kaggle", choices=["kaggle", "local"])
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()

    target = os.path.join(args.root, "irregular", "mask")
    if os.path.exists(target) and len(os.listdir(target)) > 0:
        print(f"Masks already exist at {target} ({len(os.listdir(target))} files)")
        return

    if args.source == "kaggle":
        ok = download_kaggle(args.root)
        if not ok:
            print("Kaggle failed, falling back to local generation...")
            generate_local(args.root, args.count)
    else:
        generate_local(args.root, args.count)


if __name__ == "__main__":
    main()
