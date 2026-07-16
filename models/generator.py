"""models/generator.py — Contextual UNet with self-attention for face inpainting."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    """Partial convolution with mask-aware updating."""

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding, bias=False)
        nn.init.constant_(self.mask_conv.weight, 1.0 / (kernel_size * kernel_size))

    def forward(self, x, mask=None):
        if mask is None:
            return self.conv(x) + self.bias, None
        out = self.conv(x * mask)
        mask_sum = self.mask_conv(mask)
        mask_sum = torch.clamp(mask_sum, min=1e-8)
        out = out / mask_sum + self.bias
        new_mask = (mask_sum > 0).float()
        return out, new_mask.detach()


class SelfAttention(nn.Module):
    """Non-local self-attention block for long-range context."""

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        Q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        K = self.key(x).view(B, -1, H * W)
        V = self.value(x).view(B, -1, H * W)

        attn = torch.softmax(torch.bmm(Q, K), dim=-1)
        out = torch.bmm(V, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x


class UNetDown(nn.Module):
    """Downsampling block for UNet encoder."""

    def __init__(self, in_ch, out_ch, use_partial=False):
        super().__init__()
        self.use_partial = use_partial
        if use_partial:
            self.conv = PartialConv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.norm = nn.BatchNorm2d(out_ch) if not use_partial else None

    def forward(self, x, mask=None):
        if self.use_partial:
            out, mask = self.conv(x, mask)
            return F.leaky_relu(out, 0.2), mask
        return self.conv(x), None


class UNetUp(nn.Module):
    """Upsampling block for UNet decoder."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)


class ContextualUNet(nn.Module):
    """Generator: Contextual UNet with attention and partial convolutions.

    Input:  (B, 4, H, W) — RGB + occlusion mask
    Output: (B, 3, H, W) — completed face
    """

    def __init__(self, config: dict):
        super().__init__()
        cfg = config["generator"]
        ch = cfg["channels"]
        use_pconv = cfg.get("use_partial_conv", True)
        use_attn = cfg.get("use_attention", True)
        dropout = cfg.get("dropout", 0.5)

        # Encoder
        self.down1 = UNetDown(cfg["in_channels"], ch[0], use_partial=use_pconv)
        self.down2 = UNetDown(ch[0], ch[1], use_partial=False)
        self.down3 = UNetDown(ch[1], ch[2], use_partial=False)
        self.down4 = UNetDown(ch[2], ch[3], use_partial=False)
        self.down5 = UNetDown(ch[3], ch[4], use_partial=False)
        self.down6 = UNetDown(ch[4], ch[5], use_partial=False)

        # Bottleneck with dilated convolutions
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch[5], ch[5], 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(ch[5]),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[5], ch[5], 3, 1, 4, dilation=4, bias=False),
            nn.BatchNorm2d(ch[5]),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[5], ch[5], 3, 1, 8, dilation=8, bias=False),
            nn.BatchNorm2d(ch[5]),
            nn.ReLU(inplace=True),
        )

        self.attention = SelfAttention(ch[5]) if use_attn else nn.Identity()

        # Decoder
        self.up1 = UNetUp(ch[5], ch[4], dropout=dropout)
        self.up2 = UNetUp(ch[4] * 2, ch[3], dropout=dropout)
        self.up3 = UNetUp(ch[3] * 2, ch[2], dropout=dropout)
        self.up4 = UNetUp(ch[2] * 2, ch[1], dropout=0.0)
        self.up5 = UNetUp(ch[1] * 2, ch[0], dropout=0.0)
        self.up6 = UNetUp(ch[0] * 2, ch[0], dropout=0.0)

        # Output
        self.final = nn.Sequential(
            nn.Conv2d(ch[0] * 2, cfg["out_channels"], 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x, mask=None):
        d1, m1 = self.down1(x, mask)
        d2, _ = self.down2(d1)
        d3, _ = self.down3(d2)
        d4, _ = self.down4(d3)
        d5, _ = self.down5(d4)
        d6, _ = self.down6(d5)

        bn = self.bottleneck(d6)
        bn = self.attention(bn)

        u1 = self.up1(bn, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        u6 = self.up6(u5, d1)

        out = self.final(u6)

        if mask is not None:
            out = x[:, :3] * (1 - mask) + out * mask

        return out
