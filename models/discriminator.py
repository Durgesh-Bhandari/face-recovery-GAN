"""models/discriminator.py — Spectral-Norm PatchGAN discriminator."""
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class SNPatchGAN(nn.Module):
    """Spectral-Norm PatchGAN discriminator.

    Input:  real(B,3,H,W), fake(B,3,H,W) → concat to (B,6,H,W)
    Output: (B,1,30,30) patch predictions (real/fake logits)
    """

    def __init__(self, config: dict):
        super().__init__()
        cfg = config["discriminator"]
        ch = cfg["channels"]
        in_ch = cfg["in_channels"]

        layers = []
        prev_ch = in_ch
        for i, c in enumerate(ch):
            layers.append(
                spectral_norm(nn.Conv2d(prev_ch, c, 4, 2, 1, bias=(i == 0)))
            )
            if i > 0:
                layers.append(nn.BatchNorm2d(c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_ch = c

        layers.append(
            spectral_norm(nn.Conv2d(prev_ch, 1, 4, 1, 1))
        )

        self.model = nn.Sequential(*layers)

    def forward(self, real, fake):
        x = torch.cat([real, fake], dim=1)
        return self.model(x)
