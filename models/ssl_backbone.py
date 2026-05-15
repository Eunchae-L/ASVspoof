"""Frozen SSL backbone (WavLM / wav2vec2 / HuBERT / XLS-R) + learnable layer weighted sum."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class SSLBackbone(nn.Module):
    def __init__(self, name_or_path: str, layer_weighted_sum: bool = True):
        super().__init__()
        self.name_or_path = name_or_path
        self.model = AutoModel.from_pretrained(name_or_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # hidden_states에는 feature embedding(layer 0)이 포함되어 +1
        n_layers = self.model.config.num_hidden_layers + 1
        self.layer_weighted_sum = layer_weighted_sum
        if layer_weighted_sum:
            self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.feature_dim: int = self.model.config.hidden_size

    def train(self, mode: bool = True):
        # SSL은 항상 eval mode
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, num_samples) at 16 kHz -> features (B, T, D)."""
        with torch.no_grad():
            out = self.model(wav, output_hidden_states=True, return_dict=True)
            hs = torch.stack(out.hidden_states, dim=0)        # (L+1, B, T, D)

        if self.layer_weighted_sum:
            w = torch.softmax(self.layer_weights, dim=0)
            return (hs * w.view(-1, 1, 1, 1)).sum(dim=0)
        return hs[-1]


# 예전 이름
XLSRBackbone = SSLBackbone
