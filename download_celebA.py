"""download_celebA.py — Download CelebA dataset (aligned+cropped)."""
import os
import argparse
import zipfile
import requests
from tqdm import tqdm

CELEBA_URLS = {
    "img_align_celeba.zip": (
        "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    ),
    "list_eval_partition.txt": (
        "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pY0NSMzRuSXJEVkk"
    ),
    "list_landmarks_align_celeba.txt": (
        "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pd0FJY3Blby1HUTQ"
    ),
}


def download_file(url, dest, chunk_size=8192):
    resp = requests.get(url, stream=True, allow_redirects=True)
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/celebA", help="Download root")
    parser.add_argument("--subset", type=int, default=20000, help="Limit images (-1 for all)")
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)

    # Download
    for filename, url in CELEBA_URLS.items():
        dest = os.path.join(args.root, filename)
        if not os.path.exists(dest):
            print(f"Downloading {filename}...")
            download_file(url, dest)
        else:
            print(f"{filename} exists, skipping")

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
