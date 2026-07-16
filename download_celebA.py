"""download_celebA.py — Download CelebA dataset.

Supports multiple sources (tried in order):
  1. gdrive — Google Drive via gdown (default)
  2. kaggle — Kaggle API (requires kaggle account + API token)

Usage:
    python download_celebA.py --root ./data/celebA --subset 20000
    python download_celebA.py --source kaggle --root ./data/celebA --subset 20000
"""
import os
import argparse
import zipfile
import subprocess
import sys
import shutil
from pathlib import Path


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def ensure_package(pkg):
    try:
        __import__(pkg)
        return True
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True


def download_gdrive(root):
    file_id = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    dest = os.path.join(root, "img_align_celeba.zip")
    url = f"https://drive.google.com/uc?id={file_id}"
    ensure_package("gdown")
    import gdown
    print("Downloading from Google Drive...")
    try:
        gdown.download(url, dest, quiet=False)
    except Exception as e:
        eprint(f"Google Drive failed: {e}")
        eprint("Try: --source kaggle")
        sys.exit(1)
    return dest


def download_kaggle(root):
    ensure_package("kagglehub")
    import kagglehub
    print("Downloading from Kaggle (jessicali9530/celeba-dataset)...")
    path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    print(f"Kaggle cache path: {path}")
    # Find the zip file in the downloaded directory
    zip_files = list(Path(path).rglob("*.zip"))
    if not zip_files:
        eprint(f"No zip found in {path}")
        sys.exit(1)
    src = str(zip_files[0])
    dest = os.path.join(root, "img_align_celeba.zip")
    shutil.copy2(src, dest)
    print(f"Copied {src} -> {dest}")
    return dest


def download_text_files(root):
    """Download small annotation files from Hugging Face as fallback."""
    ensure_package("requests")
    import requests

    files = {
        "list_eval_partition.txt":
            "https://huggingface.co/datasets/ylecun/celeba/resolve/main/list_eval_partition.txt",
        "list_landmarks_align_celeba.txt":
            "https://huggingface.co/datasets/ylecun/celeba/resolve/main/list_landmarks_align_celeba.txt",
    }
    for name, url in files.items():
        dest = os.path.join(root, name)
        if os.path.exists(dest):
            continue
        print(f"Downloading {name}...")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)


def extract_and_subset(root, subset):
    zip_path = os.path.join(root, "img_align_celeba.zip")
    extract_dir = os.path.join(root, "img_align_celeba")
    if not os.path.exists(extract_dir):
        print("Extracting images...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")

    if subset > 0:
        all_imgs = sorted(os.listdir(extract_dir))
        keep = all_imgs[:min(subset, len(all_imgs))]
        for fname in all_imgs:
            if fname not in keep:
                os.remove(os.path.join(extract_dir, fname))
        print(f"Kept {len(keep)} images (subset={subset})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/celebA")
    parser.add_argument("--source", default="gdrive",
                        choices=["gdrive", "kaggle"])
    parser.add_argument("--subset", type=int, default=20000)
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)

    zip_dest = os.path.join(args.root, "img_align_celeba.zip")
    extract_dir = os.path.join(args.root, "img_align_celeba")

    if os.path.exists(extract_dir):
        print(f"Already extracted at {extract_dir}, skipping download")
    elif os.path.exists(zip_dest):
        print(f"Zip exists, extracting...")
        extract_and_subset(args.root, args.subset)
    else:
        if args.source == "gdrive":
            download_gdrive(args.root)
        elif args.source == "kaggle":
            download_kaggle(args.root)
        extract_and_subset(args.root, args.subset)

    # Download text files (small, from HF mirror)
    download_text_files(args.root)

    print("CelebA ready.")


if __name__ == "__main__":
    main()
