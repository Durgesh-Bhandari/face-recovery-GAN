"""data/dataloader.py — Dataset and transforms for face recovery training."""
import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from .mask_generator import MaskGenerator


class FaceRecoveryDataset(Dataset):
    """Dataset for self-supervised face recovery training.

    Takes clean CelebA faces and generates:
    - input: face with synthetic occlusion
    - target: original clean face
    - mask: occlusion mask (1=occluded)
    """

    def __init__(
        self,
        root: str,
        mask_dir: str = "./data/masks",
        img_size: int = 256,
        split: str = "train",
        subset: int = -1,
        occlusion_prob: float = 0.9,
        mask_types=None,
    ):
        self.img_size = img_size
        self.split = split
        self.occlusion_prob = occlusion_prob
        self.mask_types = mask_types or ["irregular", "rect", "half"]
        self.mask_gen = MaskGenerator(mask_dir, img_size)

        img_dir = os.path.join(root, "img_align_celeba")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"CelebA images not found at {img_dir}")

        all_imgs = sorted(os.listdir(img_dir))

        # Split using partition file
        partition_file = os.path.join(root, "list_eval_partition.txt")
        if os.path.exists(partition_file):
            with open(partition_file) as f:
                partition = {}
                for line in f:
                    name, part = line.strip().split()
                    partition[name] = int(part)
            split_map = {"train": 0, "val": 1, "test": 2}
            target = split_map.get(split, 0)
            self.image_paths = [
                os.path.join(img_dir, n)
                for n in all_imgs
                if n in partition and partition[n] == target
            ]
        else:
            n = len(all_imgs)
            if split == "train":
                self.image_paths = [os.path.join(img_dir, n) for n in all_imgs[:int(n * 0.85)]]
            elif split == "val":
                self.image_paths = [os.path.join(img_dir, n) for n in all_imgs[int(n * 0.85):int(n * 0.9)]]
            else:
                self.image_paths = [os.path.join(img_dir, n) for n in all_imgs[int(n * 0.9):]]

        if subset > 0 and len(self.image_paths) > subset:
            self.image_paths = self.image_paths[:subset]

        print(f"Split '{split}': {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        target = img.copy()

        if random.random() < self.occlusion_prob:
            mask = self.mask_gen(self.mask_types)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        mask = mask[None, :, :]  # (1, H, W)
        occluded = img * (1.0 - mask)
        inp = np.concatenate([occluded, mask], axis=0)  # (4, H, W)

        return {
            "input": torch.from_numpy(inp).float(),
            "target": torch.from_numpy(target).float(),
            "mask": torch.from_numpy(mask).float(),
            "path": path,
        }


def create_dataloader(config, split="train"):
    dataset = FaceRecoveryDataset(
        root=config["data"]["root"],
        mask_dir=config["data"]["mask_dir"],
        img_size=config["data"]["img_size"],
        split=split,
        subset=config["data"].get("subset", -1),
        occlusion_prob=config["augmentation"]["occlusion_prob"],
        mask_types=config["augmentation"]["mask_types"],
    )

    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=(split == "train"),
    )
