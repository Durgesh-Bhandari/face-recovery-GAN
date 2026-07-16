"""download_celebA.py — Download CelebA dataset.

Sources:
  gdrive  — Google Drive via gdown (may hit quota)
  kaggle  — Kaggle (jessicali9530/celeba-dataset), works in Colab

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


def ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def download_gdrive(root):
    file_id = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    dest = os.path.join(root, "img_align_celeba.zip")
    ensure("gdown")
    import gdown
    print("Downloading from Google Drive...")
    try:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", dest, quiet=False)
    except Exception as e:
        eprint(f"Google Drive failed: {e}")
        eprint("Try: --source kaggle")
        sys.exit(1)
    return dest


def copy_images(src_dir, dst_dir, subset):
    os.makedirs(dst_dir, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(exts)])
    if not images:
        eprint(f"No images found in {src_dir}")
        return False
    keep = images[:min(subset, len(images))] if subset > 0 else images
    for fname in keep:
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
    print(f"Copied {len(keep)} images to {dst_dir}")
    return True


def download_kaggle(root, subset):
    ensure("kagglehub")
    import kagglehub

    dataset = "jessicali9530/celeba-dataset"
    print(f"Downloading from Kaggle ({dataset})...")
    path = Path(kagglehub.dataset_download(dataset))
    print(f"Kaggle path: {path}")

    extract_dir = os.path.join(root, "img_align_celeba")

    # Case 1: Already extracted image directory
    for candidate in [path / "img_align_celeba", path / "images", path]:
        if candidate.is_dir() and copy_images(str(candidate), extract_dir, subset):
            return

    # Case 2: Zip file needs extraction
    zips = list(path.rglob("*.zip"))
    if zips:
        src = str(zips[0])
        print(f"Extracting {src}...")
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(root)
        # After extraction, find where images landed
        for candidate in [Path(root) / "img_align_celeba"]:
            if candidate.is_dir():
                if subset > 0:
                    imgs = sorted(candidate.iterdir())
                    for f in imgs[subset:]:
                        f.unlink()
                print(f"Kept {subset or 'all'} images at {candidate}")
                return
        return

    eprint(f"No images or zip found in {path}")
    sys.exit(1)


def download_text_files(root):
    ensure("requests")
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
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/celebA")
    parser.add_argument("--source", default="gdrive", choices=["gdrive", "kaggle"])
    parser.add_argument("--subset", type=int, default=20000)
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)
    extract_dir = os.path.join(args.root, "img_align_celeba")

    if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
        print(f"Images already at {extract_dir}")
    else:
        if args.source == "kaggle":
            download_kaggle(args.root, args.subset)
        else:
            zip_path = os.path.join(args.root, "img_align_celeba.zip")
            if not os.path.exists(zip_path):
                download_gdrive(args.root)
            print("Extracting...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(args.root)
            if args.subset > 0:
                all_imgs = sorted(os.listdir(extract_dir))
                for f in all_imgs[args.subset:]:
                    os.remove(os.path.join(args.root, "img_align_celeba", f))
            print(f"Images ready at {extract_dir}")

    download_text_files(args.root)
    print("CelebA ready.")


if __name__ == "__main__":
    main()
