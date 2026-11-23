"""wav -> frozen SSL (layer weighted sum) -> SpecAugment -> adapter -> z -> {workers, classifier}."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from data.augment import SpecAugment
from models.adapter import build_adapter
from models.classifier import build_classifier
from models.ssl_backbone import SSLBackbone
from models.workers import WorkerHead


class SSLMultiTaskModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        ssl_cfg = cfg["model"]["ssl"]
        self.ssl = SSLBackbone(
            ssl_cfg["name_or_path"],
            layer_weighted_sum=ssl_cfg.get("layer_weighted_sum", True),
            normalize_input=ssl_cfg.get("normalize_input", False),
        )

        adapter_cfg = cfg["model"]["adapter"]
        self.adapter = build_adapter(adapter_cfg, in_dim=self.ssl.feature_dim)
        z_dim = adapter_cfg["hidden_dim"]

        worker_cfg = cfg["model"]["workers"]
        self.workers = WorkerHead(worker_cfg, z_dim=z_dim)

        self.classifier = build_classifier(cfg["model"]["classifier"], in_dim=z_dim)

        aug_cfg = cfg.get("augment", {})
        if aug_cfg.get("specaug_p", 0.0) > 0:
            self.specaug = SpecAugment(
                F=aug_cfg.get("specaug_F", 27),
                T=aug_cfg.get("specaug_T", 40),
                n_freq_masks=aug_cfg.get("specaug_n_freq_masks", 2),
                n_time_masks=aug_cfg.get("specaug_n_time_masks", 2),
                p=aug_cfg["specaug_p"],
            )
        else:
            self.specaug = None

    def forward(
        self,
        wav: torch.Tensor,
        apply_specaug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """wav: (B, num_samples) -> (fake_logit, worker_preds, z)."""
        feats = self.ssl(wav)                            # (B, T, D_ssl)
        if apply_specaug and self.specaug is not None and self.training:
            feats = self.specaug(feats)
        z = self.adapter(feats)                          # (B, T, z_dim)
        worker_preds = self.workers(z)                   # dict
        logit = self.classifier(z)                       # (B,)
        return logit, worker_preds, z

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# 예전 이름
XLSRMultiTaskModel = SSLMultiTaskModel
