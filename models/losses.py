"""models/losses.py — All loss functions for face recovery GAN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGG16Features(nn.Module):
    """VGG-16 feature extractor (pretrained, frozen) for perceptual loss."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = vgg.features

        self.block1 = features[:5]    # relu1_2
        self.block2 = features[5:10]   # relu2_2
        self.block3 = features[10:17]  # relu3_3
        self.block4 = features[17:24]  # relu4_3
        self.block5 = features[24:31]  # relu5_3

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x * 0.5 + 0.5)
        x = (x - mean) / std

        feats = []
        x = self.block1(x); feats.append(x)
        x = self.block2(x); feats.append(x)
        x = self.block3(x); feats.append(x)
        x = self.block4(x); feats.append(x)
        x = self.block5(x); feats.append(x)
        return feats


class PerceptualLoss(nn.Module):
    """Perceptual (content) loss using VGG-16 feature maps."""

    def __init__(self, weights=(1/6, 1/6, 1/6, 1/6, 1/6)):
        super().__init__()
        self.vgg = VGG16Features()
        self.weights = weights

    def forward(self, pred, target):
        pred_feats = self.vgg(pred)
        target_feats = self.vgg(target)
        loss = 0.0
        for w, pf, tf in zip(self.weights, pred_feats, target_feats):
            loss += w * F.l1_loss(pf, tf)
        return loss


class StyleLoss(nn.Module):
    """Style loss using Gram matrix of VGG-16 features."""

    def __init__(self, weights=(1/6, 1/6, 1/6, 1/6, 1/6)):
        super().__init__()
        self.vgg = VGG16Features()
        self.weights = weights

    def gram_matrix(self, x):
        B, C, H, W = x.shape
        features = x.view(B, C, H * W)
        gram = torch.bmm(features, features.permute(0, 2, 1)) / (C * H * W)
        return gram

    def forward(self, pred, target):
        pred_feats = self.vgg(pred)
        target_feats = self.vgg(target)
        loss = 0.0
        for w, pf, tf in zip(self.weights, pred_feats, target_feats):
            loss += w * F.l1_loss(self.gram_matrix(pf), self.gram_matrix(tf))
        return loss


class GANLoss(nn.Module):
    """Hinge adversarial loss (more stable than BCE for GANs)."""

    def d_loss(self, pred_real, pred_fake):
        """Discriminator hinge loss."""
        loss_real = torch.mean(F.relu(1.0 - pred_real))
        loss_fake = torch.mean(F.relu(1.0 + pred_fake))
        return (loss_real + loss_fake) / 2

    def g_loss(self, pred_fake):
        """Generator adversarial loss (non-saturating)."""
        return -torch.mean(pred_fake)


class IDLoss(nn.Module):
    """Identity preservation loss using face embeddings.

    Falls back to cosine similarity on VGG features.
    """

    def __init__(self, device="cuda"):
        super().__init__()
        self.vgg = VGG16Features()
        self.device = device

    def forward(self, pred, target):
        pred_feat = self.vgg(pred)[-1].mean(dim=[2, 3])
        target_feat = self.vgg(target)[-1].mean(dim=[2, 3])
        cos_sim = F.cosine_similarity(pred_feat, target_feat, dim=1)
        return 1.0 - cos_sim.mean()
