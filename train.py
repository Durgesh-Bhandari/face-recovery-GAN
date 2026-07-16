"""train.py — Main training script for Face Recovery GAN."""
import os
import argparse
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataloader import create_dataloader
from models.generator import ContextualUNet
from models.discriminator import SNPatchGAN
from models.losses import PerceptualLoss, StyleLoss, GANLoss, IDLoss


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def save_checkpoint(generator, discriminator, g_opt, d_opt, epoch, path):
    torch.save({
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "g_optimizer": g_opt.state_dict(),
        "d_optimizer": d_opt.state_dict(),
    }, path)
    print(f"Checkpoint saved: {path}")


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Data
    train_loader = create_dataloader(config, "train")
    val_loader = create_dataloader(config, "val")

    # Models
    generator = ContextualUNet(config).to(device)
    discriminator = SNPatchGAN(config).to(device)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator: {g_params:,} params, Discriminator: {d_params:,} params")

    # Losses
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss().to(device)
    style_loss = StyleLoss().to(device)
    gan_loss = GANLoss()

    id_loss_weight = config["loss"]["id"]
    id_loss = IDLoss(device).to(device) if id_loss_weight > 0 else None

    # Optimizers
    g_opt = torch.optim.Adam(
        generator.parameters(),
        lr=config["training"]["lr_g"],
        betas=tuple(config["training"]["betas"]),
    )
    d_opt = torch.optim.Adam(
        discriminator.parameters(),
        lr=config["training"]["lr_d"],
        betas=tuple(config["training"]["betas"]),
    )

    # Mixed precision
    use_amp = config["training"]["mixed_precision"]
    scaler_g = torch.amp.GradScaler("cuda") if use_amp else None
    scaler_d = torch.amp.GradScaler("cuda") if use_amp else None

    # Logging
    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["paths"]["sample_dir"], exist_ok=True)
    writer = SummaryWriter(config["paths"]["log_dir"])

    epochs = config["training"]["epochs"]
    log_interval = config["training"]["log_interval"]
    sample_interval = config["training"]["sample_interval"]
    checkpoint_interval = config["training"]["checkpoint_interval"]
    grad_clip = config["training"]["gradient_clip"]

    global_step = 0
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            inp = batch["input"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)

            # --- Train Discriminator ---
            with torch.amp.autocast("cuda", enabled=use_amp):
                with torch.no_grad():
                    fake = generator(inp, mask)

                d_real = discriminator(target, target)
                d_fake = discriminator(target, fake.detach())
                d_loss = gan_loss.d_loss(d_real, d_fake)

            d_opt.zero_grad()
            if scaler_d:
                scaler_d.scale(d_loss).backward()
                scaler_d.step(d_opt)
                scaler_d.update()
            else:
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
                d_opt.step()

            # --- Train Generator ---
            with torch.amp.autocast("cuda", enabled=use_amp):
                fake = generator(inp, mask)
                d_fake = discriminator(target, fake)

                loss = l1_loss(fake, target) * config["loss"]["l1"]
                loss += perceptual_loss(fake, target) * config["loss"]["perceptual"]
                loss += style_loss(fake, target) * config["loss"]["style"]
                loss += gan_loss.g_loss(d_fake) * config["loss"]["adversarial"]

                if id_loss is not None and id_loss_weight > 0:
                    loss += id_loss(fake, target) * id_loss_weight

            g_opt.zero_grad()
            if scaler_g:
                scaler_g.scale(loss).backward()
                scaler_g.unscale_(g_opt)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
                scaler_g.step(g_opt)
                scaler_g.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
                g_opt.step()

            epoch_g_loss += loss.item()
            epoch_d_loss += d_loss.item()
            global_step += 1

            if global_step % log_interval == 0:
                writer.add_scalar("Loss/G", loss.item(), global_step)
                writer.add_scalar("Loss/D", d_loss.item(), global_step)
                pbar.set_postfix({"G": f"{loss.item():.4f}", "D": f"{d_loss.item():.4f}"})

            if global_step % sample_interval == 0:
                with torch.no_grad():
                    writer.add_images("Train/Input", inp[:4, :3], global_step)
                    writer.add_images("Train/Mask", mask[:4], global_step)
                    writer.add_images("Train/Output", fake[:4], global_step)
                    writer.add_images("Train/Target", target[:4], global_step)

        epoch_time = time.time() - epoch_start
        avg_g = epoch_g_loss / len(train_loader)
        avg_d = epoch_d_loss / len(train_loader)
        print(f"Epoch {epoch}: G={avg_g:.4f}, D={avg_d:.4f}, Time={epoch_time:.1f}s")
        writer.add_scalar("Loss/Epoch_G", avg_g, epoch)
        writer.add_scalar("Loss/Epoch_D", avg_d, epoch)

        if epoch % checkpoint_interval == 0:
            save_checkpoint(
                generator, discriminator, g_opt, d_opt, epoch,
                os.path.join(config["paths"]["checkpoint_dir"], f"epoch_{epoch}.pt"),
            )

    save_checkpoint(
        generator, discriminator, g_opt, d_opt, epochs,
        os.path.join(config["paths"]["checkpoint_dir"], "final.pt"),
    )
    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    set_seed(args.seed)
    train(config)
