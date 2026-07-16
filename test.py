"""test.py — Inference demo for trained face recovery GAN."""
import os
import argparse
import yaml
import torch
import cv2
import numpy as np
from models.generator import ContextualUNet


def load_model(config, checkpoint_path, device):
    generator = ContextualUNet(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    return generator


def preprocess_image(img_path, img_size=256):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2: (h + s) // 2, (w - s) // 2: (w + s) // 2]
    img = cv2.resize(img, (img_size, img_size))
    img_tensor = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def create_occlusion(img_tensor, mask_type="rect"):
    """Add synthetic occlusion for testing."""
    from data.mask_generator import MaskGenerator
    mg = MaskGenerator()
    mask = mg([mask_type])
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    occluded = img_tensor * (1.0 - mask_tensor)
    inp = torch.cat([occluded, mask_tensor], dim=1)
    return inp, mask_tensor


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default="output.jpg", help="Output image path")
    parser.add_argument("--occlude", type=str, default=None,
                        choices=[None, "irregular", "rect", "half"],
                        help="Add synthetic occlusion before recovery")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(open(args.config))

    generator = load_model(config, args.checkpoint, device)

    img_tensor = preprocess_image(args.input, config["data"]["img_size"]).to(device)

    if args.occlude:
        inp, mask = create_occlusion(img_tensor, args.occlude)
        inp = inp.to(device)
        mask = mask.to(device)
    else:
        inp = torch.cat([img_tensor, torch.zeros_like(img_tensor[:, :1])], dim=1)
        mask = None

    output = generator(inp, mask if mask is not None else inp[:, 3:4])

    out_np = ((output[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    out_np = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    if args.occlude:
        inp_np = ((inp[0, :3].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        inp_np = cv2.cvtColor(inp_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output.replace(".jpg", "_input.jpg"), inp_np)

    cv2.imwrite(args.output, out_np)
    print(f"Output saved: {args.output}")


if __name__ == "__main__":
    main()
