"""PASE+ 스타일 보조 regression worker (MFCC / CQT / LFCC / log-mel).

target은 clean 파형에서 hop 320 (50 Hz)로 계산해 SSL frame과 맞춘다. CQT는 librosa 없으면 꺼짐.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T

try:
    import librosa
    _HAS_LIBROSA = True
except Exception:  # pragma: no cover - optional dep
    _HAS_LIBROSA = False


class RegressionWorker(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class WorkerHead(nn.Module):
    def __init__(self, cfg: dict, z_dim: int = 256):
        super().__init__()
        self.cfg = cfg
        enabled = cfg.get("enabled", [])
        hidden = cfg.get("mlp_hidden", 128)

        self.workers = nn.ModuleDict()
        if "mfcc" in enabled:
            self.workers["mfcc"] = RegressionWorker(z_dim, hidden, cfg["mfcc_dim"])
        if "cqt" in enabled and _HAS_LIBROSA:
            self.workers["cqt"] = RegressionWorker(z_dim, hidden, cfg["cqt_dim"])
        if "lfcc" in enabled:
            self.workers["lfcc"] = RegressionWorker(z_dim, hidden, cfg["lfcc_dim"])
        if "logmel" in enabled:
            self.workers["logmel"] = RegressionWorker(z_dim, hidden, cfg["logmel_dim"])

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: w(z) for name, w in self.workers.items()}


class TargetComputer:
    # frame 수가 SSL 출력과 ±1 차이날 수 있음 -> worker_loss에서 truncate

    SR = 16000
    HOP = 320
    N_FFT = 400
    WIN = 400

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        enabled = cfg.get("enabled", [])
        self.enabled = [e for e in enabled if (e != "cqt" or _HAS_LIBROSA)]
        self.device = device

        self.mfcc = None
        self.lfcc = None
        self.mel = None

        if "mfcc" in self.enabled:
            self.mfcc = T.MFCC(
                sample_rate=self.SR,
                n_mfcc=cfg["mfcc_dim"],
                melkwargs=dict(
                    n_fft=self.N_FFT, hop_length=self.HOP, win_length=self.WIN,
                    n_mels=max(40, cfg["mfcc_dim"]),
                    f_min=0, f_max=8000, power=2.0, center=True,
                ),
            ).to(device)

        if "lfcc" in self.enabled:
            try:
                self.lfcc = T.LFCC(
                    sample_rate=self.SR,
                    n_lfcc=cfg["lfcc_dim"],
                    speckwargs=dict(
                        n_fft=self.N_FFT, hop_length=self.HOP, win_length=self.WIN,
                        power=2.0, center=True,
                    ),
                ).to(device)
            except AttributeError:
                self.lfcc = None  # very old torchaudio

        if "logmel" in self.enabled:
            self.mel = T.MelSpectrogram(
                sample_rate=self.SR,
                n_fft=self.N_FFT, hop_length=self.HOP, win_length=self.WIN,
                n_mels=cfg["logmel_dim"], f_min=0, f_max=8000, power=2.0,
                center=True,
            ).to(device)

    @torch.no_grad()
    def __call__(self, clean_wav: torch.Tensor) -> Dict[str, torch.Tensor]:
        """clean_wav: (B, num_samples) -> dict of (B, T, target_dim)."""
        out: Dict[str, torch.Tensor] = {}
        if self.mfcc is not None:
            mfcc = self.mfcc(clean_wav)                       # (B, n_mfcc, T)
            out["mfcc"] = mfcc.transpose(1, 2)
        if self.lfcc is not None:
            lfcc = self.lfcc(clean_wav)
            out["lfcc"] = lfcc.transpose(1, 2)
        if self.mel is not None:
            mel = self.mel(clean_wav)
            out["logmel"] = torch.log(mel.clamp(min=1e-6)).transpose(1, 2)

        if "cqt" in self.enabled and _HAS_LIBROSA:
            wav_np = clean_wav.detach().cpu().numpy()
            cqt_list = []
            n_bins = self.cfg["cqt_dim"]
            for b in range(wav_np.shape[0]):
                C = librosa.cqt(
                    wav_np[b],
                    sr=self.SR,
                    hop_length=self.HOP,
                    n_bins=n_bins,
                    bins_per_octave=12,
                )
                logC = np.log(np.abs(C) + 1e-6).astype(np.float32)
                cqt_list.append(torch.from_numpy(logC.T))     # (T, n_bins)
            cqt = torch.nn.utils.rnn.pad_sequence(cqt_list, batch_first=True)
            out["cqt"] = cqt.to(clean_wav.device)
        return out


def worker_loss(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    for name, pred in preds.items():
        if name not in targets:
            continue
        tgt = targets[name].to(pred.dtype)
        T_min = min(pred.shape[1], tgt.shape[1])
        pred_a = pred[:, :T_min]
        tgt_a = tgt[:, :T_min]
        # batch 단위 per-feature 표준화
        mean = tgt_a.mean(dim=(0, 1), keepdim=True)
        std = tgt_a.std(dim=(0, 1), keepdim=True).clamp(min=1e-3)
        tgt_n = (tgt_a - mean) / std
        pred_n = (pred_a - mean) / std
        losses[name] = torch.nn.functional.l1_loss(pred_n, tgt_n)
    return losses
