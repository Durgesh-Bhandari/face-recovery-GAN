"""download_masks.py — Generate training masks locally.

NVIDIA irregular mask dataset repo is no longer available (404).
Fallback generates random polygon masks identical to what
mask_generator.py creates at runtime anyway.
"""
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/masks")
    parser.add_argument("--count", type=int, default=50, help="Number of mask patterns")
    args = parser.parse_args()

    target = os.path.join(args.root, "irregular", "mask")

    if os.path.exists(target):
        print(f"Masks already exist at {target}")
        return

    os.makedirs(target, exist_ok=True)

    import cv2
    import numpy as np
    rng = np.random.RandomState(42)

    for i in range(args.count):
        mask = np.zeros((512, 512), dtype=np.uint8)
        pts = rng.randint(50, 462, size=(rng.randint(5, 12), 2))
        cv2.fillPoly(mask, [pts], 255)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(target, f"mask_{i:04d}.png"), mask)

    print(f"Generated {args.count} mask patterns in {target}")


if __name__ == "__main__":
    main()
