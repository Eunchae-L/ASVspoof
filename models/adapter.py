"""SSL feature -> z adapter. A1: BiLSTM, A2: 1D SE-Res2Net.

(B, T, in_dim) -> (B, T, hidden_dim)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMAdapter(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x, _ = self.lstm(x)
        x = self.out(x)
        return self.norm(x)


class _SE1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        bottleneck = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        s = x.mean(dim=2)
        s = self.fc(s).unsqueeze(2)
        return x * s


class _Res2NetBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale: int = 4, kernel: int = 3):
        super().__init__()
        assert out_ch % scale == 0, "out_ch must be divisible by scale"
        self.scale = scale
        self.conv1 = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)

        sub = out_ch // scale
        self.convs = nn.ModuleList(
            [nn.Conv1d(sub, sub, kernel, padding=kernel // 2, bias=False) for _ in range(scale - 1)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(sub) for _ in range(scale - 1)])
        self.conv3 = nn.Conv1d(out_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.se = _SE1D(out_ch)
        if in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.short = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.short(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        splits = list(torch.chunk(out, self.scale, dim=1))
        ys = [splits[0]]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            inp = splits[i + 1] if i == 0 else splits[i + 1] + ys[-1]
            ys.append(F.relu(bn(conv(inp)), inplace=True))
        out = torch.cat(ys, dim=1)
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        return F.relu(out + identity, inplace=True)


class SERes2NetAdapter(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256,
                 num_blocks: int = 3, scale: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, hidden_dim, 1)
        self.proj_bn = nn.BatchNorm1d(hidden_dim)
        self.blocks = nn.Sequential(*[
            _Res2NetBlock1D(hidden_dim, hidden_dim, scale=scale)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, in_dim)
        x = x.transpose(1, 2)                            # (B, in_dim, T)
        x = F.relu(self.proj_bn(self.proj(x)), inplace=True)
        x = self.blocks(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)                            # (B, T, hidden_dim)
        return self.norm(x)


def build_adapter(cfg: dict, in_dim: int = 1024) -> nn.Module:
    t = cfg.get("type", "bilstm").lower()
    if t == "bilstm":
        return BiLSTMAdapter(
            in_dim=in_dim,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.1),
        )
    if t in ("se_res2net", "res2net", "se-res2net"):
        return SERes2NetAdapter(
            in_dim=in_dim,
            hidden_dim=cfg["hidden_dim"],
            num_blocks=cfg.get("num_blocks", 3),
            scale=cfg.get("scale", 4),
            dropout=cfg.get("dropout", 0.1),
        )
    raise ValueError(f"Unknown adapter type: {t}")
