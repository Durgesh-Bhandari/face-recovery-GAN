"""download_celebA.py — Download CelebA dataset (aligned+cropped).

Uses gdown for large files (Google Drive virus-scan bypass).
Small text files downloaded via requests.

Usage:
    python download_celebA.py --root ./data/celebA --subset 20000
"""
import os
import argparse
import zipfile
import subprocess
import sys
from tqdm import tqdm

# Google Drive file IDs
CELEBA_FILES = {
    "img_align_celeba.zip": "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
    "list_eval_partition.txt": "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
    "list_landmarks_align_celeba.txt": "0B7EVK8r0v71pd0FJY3Blby1HUTQ",
}


def ensure_gdown():
    """Install gdown if not available."""
    try:
        import gdown
        return True
    except ImportError:
        print("Installing gdown (required for Google Drive downloads)...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gdown"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True


def download_with_gdown(file_id, dest):
    """Download a file from Google Drive using gdown."""
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest, quiet=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/celebA", help="Download root")
    parser.add_argument("--subset", type=int, default=20000,
                        help="Limit images (-1 for all)")
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)
    ensure_gdown()

    for filename, file_id in CELEBA_FILES.items():
        dest = os.path.join(args.root, filename)
        if os.path.exists(dest):
            print(f"{filename} exists, skipping")
            continue
        print(f"Downloading {filename}...")
        download_with_gdown(file_id, dest)

    # Extract zip
    zip_path = os.path.join(args.root, "img_align_celeba.zip")
    extract_dir = os.path.join(args.root, "img_align_celeba")
    if not os.path.exists(extract_dir):
        print("Extracting images...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")

    # Subset if needed
    if args.subset > 0:
        all_imgs = sorted(os.listdir(extract_dir))
        keep = all_imgs[:min(args.subset, len(all_imgs))]
        for fname in all_imgs:
            if fname not in keep:
                os.remove(os.path.join(extract_dir, fname))
        print(f"Kept {len(keep)} images (subset={args.subset})")

    print("CelebA ready.")


if __name__ == "__main__":
    main()
