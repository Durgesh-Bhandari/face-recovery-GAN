"""download_masks.py — Download irregular mask patterns from NVIDIA."""
import os
import argparse
import requests
import zipfile
from tqdm import tqdm

MASKS_URL = (
    "https://github.com/NVlabs/irregular-mask-dataset/archive/refs/heads/master.zip"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/masks", help="Mask download root")
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)

    zip_path = os.path.join(args.root, "masks.zip")
    print("Downloading NVIDIA irregular masks...")
    resp = requests.get(MASKS_URL, stream=True, allow_redirects=True)
    total = int(resp.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print("Extracting masks...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(args.root)

    # Rename mask directories for easy access
    mask_dir = os.path.join(args.root, "irregular-mask-dataset-master")
    target = os.path.join(args.root, "irregular")
    if os.path.exists(mask_dir) and not os.path.exists(target):
        os.rename(mask_dir, target)

    print(f"Masks ready at {args.root}")


if __name__ == "__main__":
    main()
