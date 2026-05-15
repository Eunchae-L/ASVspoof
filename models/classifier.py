"""z (B, T, D)를 2D로 보고 분류. C1: LCNN-big (MFM), C2: SENet12.

둘 다 freq 축 평균 -> ASP -> FC -> logit.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASP(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim, attn_dim, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(attn_dim),
            nn.Conv1d(attn_dim, in_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        a = torch.softmax(self.attn(x), dim=2)
        mu = (a * x).sum(dim=2)
        sigma = torch.sqrt(((a * (x ** 2)).sum(dim=2) - mu ** 2).clamp(min=1e-8))
        return torch.cat([mu, sigma], dim=1)              # (B, 2C)


class _MFM2D(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return torch.maximum(a, b)


class _LCNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        # output 2*out_ch channels then MFM halves to out_ch
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size=kernel, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch * 2)
        self.mfm = _MFM2D()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mfm(self.bn(self.conv(x)))


class LCNNBig(nn.Module):
    def __init__(self, in_dim: int = 256, asp_attn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        # 4 stages, MFM halves channels: pre-MFM/post-MFM = 64/32, 96/48, 128/64, 64/32
        self.b1 = _LCNNBlock(1, 32)
        self.p1 = nn.MaxPool2d(2)
        self.b2 = _LCNNBlock(32, 48)
        self.p2 = nn.MaxPool2d(2)
        self.b3 = _LCNNBlock(48, 64)
        self.p3 = nn.MaxPool2d(2)
        self.b4 = _LCNNBlock(64, 32)
        self.p4 = nn.MaxPool2d(2)
        # (B, 32, T/16, D/16) -> freq mean -> ASP -> (B, 64)
        self.dropout = nn.Dropout(dropout)
        self.asp = ASP(in_dim=32, attn_dim=asp_attn_dim)
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z.unsqueeze(1)                                # (B, 1, T, D)
        x = self.p1(self.b1(x))
        x = self.p2(self.b2(x))
        x = self.p3(self.b3(x))
        x = self.p4(self.b4(x))
        x = x.mean(dim=3)                                 # (B, 32, T/16)
        x = self.dropout(x)
        v = self.asp(x)                                   # (B, 64)
        return self.fc(v).squeeze(-1)                     # (B,)


class _SE2DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        bottleneck = max(1, out_ch // reduction)
        self.se = nn.Sequential(
            nn.Linear(out_ch, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, out_ch),
            nn.Sigmoid(),
        )
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.short = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.short(x)
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        s = y.mean(dim=(2, 3))
        s = self.se(s).unsqueeze(-1).unsqueeze(-1)
        y = y * s
        return F.relu(y + identity, inplace=True)


class SENet12(nn.Module):
    def __init__(self, in_dim: int = 256, asp_attn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.b1 = _SE2DBlock(32, 32, stride=1)
        self.b2 = _SE2DBlock(32, 64, stride=2)
        self.b3 = _SE2DBlock(64, 128, stride=2)
        self.b4 = _SE2DBlock(128, 256, stride=2)
        # (B, 256, T/8, D/8) -> freq mean -> ASP -> (B, 512)
        self.dropout = nn.Dropout(dropout)
        self.asp = ASP(in_dim=256, attn_dim=asp_attn_dim)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z.unsqueeze(1)                                # (B, 1, T, D)
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = x.mean(dim=3)                                 # (B, 256, T/8)
        x = self.dropout(x)
        v = self.asp(x)                                   # (B, 512)
        return self.fc(v).squeeze(-1)                     # (B,)


def build_classifier(cfg: dict, in_dim: int = 256) -> nn.Module:
    t = cfg.get("type", "lcnn_big").lower()
    attn = cfg.get("asp_attention_dim", 128)
    drop = cfg.get("dropout", 0.1)
    if t in ("lcnn_big", "lcnn-big", "lcnn"):
        return LCNNBig(in_dim=in_dim, asp_attn_dim=attn, dropout=drop)
    if t in ("senet12", "senet-12", "se_net12"):
        return SENet12(in_dim=in_dim, asp_attn_dim=attn, dropout=drop)
    raise ValueError(f"Unknown classifier type: {t}")
