"""download_masks.py — Download irregular mask patterns.

Tries GitHub zip first, falls back to generating patterns locally.
"""
import os
import argparse
import zipfile
import sys
import requests
from tqdm import tqdm

MASKS_URL = (
    "https://github.com/NVlabs/irregular-mask-dataset/archive/refs/heads/master.zip"
)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/masks", help="Mask download root")
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)

    zip_path = os.path.join(args.root, "masks.zip")
    target_dir = os.path.join(args.root, "irregular")

    if os.path.exists(target_dir):
        print(f"Masks already extracted at {target_dir}")
        return

    if not os.path.exists(zip_path):
        print("Downloading NVIDIA irregular masks...")
        resp = requests.get(MASKS_URL, stream=True, allow_redirects=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    if zipfile.is_zipfile(zip_path):
        print("Extracting masks...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(args.root)
        mask_dir = os.path.join(args.root, "irregular-mask-dataset-master")
        if os.path.exists(mask_dir) and not os.path.exists(target_dir):
            os.rename(mask_dir, target_dir)
        print(f"Masks ready at {target_dir}")
    else:
        eprint(f"Downloaded file is not a valid zip ({os.path.getsize(zip_path)} bytes)")
        os.remove(zip_path)
        print("Generating fallback masks locally...")
        _generate_fallback_masks(target_dir)


def _generate_fallback_masks(target_dir):
    import cv2
    import numpy as np
    mask_out = os.path.join(target_dir, "mask")
    os.makedirs(mask_out, exist_ok=True)
    rng = np.random.RandomState(42)
    for i in range(50):
        mask = np.zeros((512, 512), dtype=np.uint8)
        pts = rng.randint(50, 462, size=(rng.randint(5, 10), 2))
        cv2.fillPoly(mask, [pts], 255)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(mask_out, f"fallback_{i:04d}.png"), mask)
    print(f"Generated 50 fallback mask patterns in {mask_out}")


if __name__ == "__main__":
    main()
