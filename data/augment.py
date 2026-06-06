"""Waveform/feature augmentation: RawBoost(간략화), codec round-trip, MUSAN, RIR, SpecAugment.

MUSAN/RIR은 경로가 없으면 no-op. codec은 eval 코덱 조건(C01-C11) 대응용.
"""
from __future__ import annotations

import os
import random
import shutil
import subprocess
from glob import glob
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from scipy import signal as sps
import soundfile as sf


def _random_biquad_coeffs(rng: np.random.Generator, order: int = 4):
    nyq = 1.0
    cutoff = rng.uniform(0.05, 0.45)  # normalized to Nyquist
    btype = rng.choice(["low", "high", "band"])
    if btype == "band":
        bw = rng.uniform(0.05, 0.3)
        lo, hi = max(0.01, cutoff - bw / 2), min(0.49, cutoff + bw / 2)
        if hi <= lo:
            lo, hi = 0.1, 0.4
        b, a = sps.butter(order, [lo, hi], btype="band")
    else:
        b, a = sps.butter(order, cutoff, btype=btype)
    return b, a


def linear_convolutive(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    b, a = _random_biquad_coeffs(rng)
    y = sps.lfilter(b, a, x).astype(np.float32)
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / peak
    return y


def non_linear_convolutive(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gain = rng.uniform(0.5, 4.0)
    y = np.tanh(gain * x).astype(np.float32) / np.tanh(gain).astype(np.float32)
    if rng.random() < 0.5:
        y = linear_convolutive(y, rng)
    return y


def impulsive_noise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    density = rng.uniform(1e-4, 5e-3)
    n_imp = max(1, int(n * density))
    idx = rng.integers(0, n, size=n_imp)
    amps = rng.choice([-1.0, 1.0], size=n_imp) * rng.uniform(0.05, 0.3, size=n_imp)
    y = x.copy()
    y[idx] = np.clip(y[idx] + amps.astype(np.float32), -1.0, 1.0)
    return y


def stationary_noise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    snr_db = rng.uniform(10.0, 30.0)
    sig_pow = float(np.mean(x ** 2) + 1e-9)
    noise_pow = sig_pow / (10 ** (snr_db / 10.0))
    noise = rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(noise_pow)
    return (x + noise).astype(np.float32)


def rawboost(x: np.ndarray, alg: int = 5, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """alg 1-4: 단일 연산, 5: 네 개를 랜덤 순서로 각각 p=0.7 적용."""
    rng = rng or np.random.default_rng()
    if alg == 1:
        return linear_convolutive(x, rng)
    if alg == 2:
        return non_linear_convolutive(x, rng)
    if alg == 3:
        return impulsive_noise(x, rng)
    if alg == 4:
        return stationary_noise(x, rng)
    ops = [linear_convolutive, non_linear_convolutive, impulsive_noise, stationary_noise]
    rng.shuffle(ops)
    y = x
    for op in ops:
        if rng.random() < 0.7:
            y = op(y, rng)
    return y.astype(np.float32)


class MUSANMixer:
    def __init__(self, root: Optional[str], snr_min: float = 5.0, snr_max: float = 20.0):
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.files: List[str] = []
        if root and os.path.isdir(root):
            for sub in ("noise", "music", "speech"):
                self.files.extend(glob(os.path.join(root, sub, "**", "*.wav"), recursive=True))

    @property
    def available(self) -> bool:
        return len(self.files) > 0

    def mix(self, x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        if not self.available:
            return x
        path = rng.choice(self.files)
        try:
            noise, nsr = sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            return x
        if noise.ndim > 1:
            noise = noise.mean(axis=1)
        if nsr != sr:
            # noise라 선형보간 리샘플로 충분
            ratio = sr / nsr
            t_old = np.arange(len(noise))
            t_new = np.linspace(0, len(noise) - 1, int(len(noise) * ratio))
            noise = np.interp(t_new, t_old, noise).astype(np.float32)
        if len(noise) < len(x):
            reps = int(np.ceil(len(x) / max(1, len(noise))))
            noise = np.tile(noise, reps)
        start = rng.integers(0, max(1, len(noise) - len(x) + 1))
        noise = noise[start:start + len(x)]
        snr_db = rng.uniform(self.snr_min, self.snr_max)
        sig_pow = float(np.mean(x ** 2) + 1e-9)
        n_pow = float(np.mean(noise ** 2) + 1e-9)
        scale = np.sqrt(sig_pow / (n_pow * (10 ** (snr_db / 10.0))))
        return (x + noise * scale).astype(np.float32)


class RIRConvolver:
    def __init__(self, root: Optional[str]):
        self.files: List[str] = []
        if root and os.path.isdir(root):
            self.files = glob(os.path.join(root, "**", "*.wav"), recursive=True)

    @property
    def available(self) -> bool:
        return len(self.files) > 0

    def convolve(self, x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        if not self.available:
            return x
        path = rng.choice(self.files)
        try:
            rir, _ = sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            return x
        if rir.ndim > 1:
            rir = rir.mean(axis=1)
        rir = rir / (np.max(np.abs(rir)) + 1e-9)
        y = sps.fftconvolve(x, rir, mode="full")[: len(x)]
        peak = float(np.max(np.abs(y)) + 1e-9)
        if peak > 1.0:
            y = y / peak
        return y.astype(np.float32)


# stdin으로 다시 decode 가능한 stream format으로 출력
_FFMPEG_ENCODE_ARGS = {
    "mp3":    ["-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"],
    "aac":    ["-c:a", "aac",        "-b:a", "64k", "-f", "adts"],
    # amr-nb는 8 kHz mono만 지원
    "amr-nb": ["-c:a", "amr_nb",     "-b:a", "12.2k", "-ar", "8000", "-f", "amr"],
    "opus":   ["-c:a", "libopus",    "-b:a", "32k",  "-f", "ogg"],
}


class CodecAugment:
    """mp3/aac/amr-nb/opus (ffmpeg 필요) 또는 narrowband-8k round-trip.

    ffmpeg가 없거나 실패하면 입력을 그대로 반환해서 학습이 멈추지 않게 한다.
    """

    def __init__(self, choices: Optional[List[str]] = None):
        self.choices = list(choices or [])
        self._has_ffmpeg = shutil.which("ffmpeg") is not None
        if not self._has_ffmpeg:
            self.choices = [c for c in self.choices if c == "narrowband-8k"]

    @property
    def available(self) -> bool:
        return len(self.choices) > 0

    def __call__(self, x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        if not self.available:
            return x
        codec = str(rng.choice(self.choices))
        try:
            if codec == "narrowband-8k":
                return self._narrowband(x, sr)
            return self._ffmpeg_roundtrip(x, sr, codec)
        except Exception:
            return x

    @staticmethod
    def _narrowband(x: np.ndarray, sr: int) -> np.ndarray:
        x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        y = torchaudio.functional.resample(x_t, sr, 8000)
        y = torchaudio.functional.resample(y, 8000, sr)
        y = y.squeeze(0).numpy().astype(np.float32)
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        elif len(y) > len(x):
            y = y[: len(x)]
        return y

    @staticmethod
    def _ffmpeg_roundtrip(x: np.ndarray, sr: int, codec: str) -> np.ndarray:
        if codec not in _FFMPEG_ENCODE_ARGS:
            return x
        x32 = x.astype(np.float32, copy=False)
        enc_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
            *_FFMPEG_ENCODE_ARGS[codec], "pipe:1",
        ]
        enc = subprocess.run(enc_cmd, input=x32.tobytes(),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             check=True)
        dec_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", "pipe:0",
            "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1",
        ]
        dec = subprocess.run(dec_cmd, input=enc.stdout,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             check=True)
        y = np.frombuffer(dec.stdout, dtype=np.float32).copy()
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        elif len(y) > len(x):
            y = y[: len(x)]
        return y


class WaveformAugment:
    def __init__(self, cfg: dict, sample_rate: int):
        self.cfg = cfg
        self.sr = sample_rate
        self.musan = MUSANMixer(cfg.get("musan_root"))
        self.rir = RIRConvolver(cfg.get("rir_root"))
        self.codec = CodecAugment(cfg.get("codec_choices") or [])

    def __call__(self, x: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if rng.random() < self.cfg.get("rawboost_p", 0.0):
            x = rawboost(x, alg=self.cfg.get("rawboost_alg", 5), rng=rng)
        # 순서: 기기 왜곡(RawBoost) -> codec -> 환경 잡음/잔향
        if self.codec.available and rng.random() < self.cfg.get("codec_p", 0.0):
            x = self.codec(x, self.sr, rng)
        if self.musan.available and rng.random() < self.cfg.get("musan_p", 0.0):
            x = self.musan.mix(x, self.sr, rng)
        if self.rir.available and rng.random() < self.cfg.get("rir_p", 0.0):
            x = self.rir.convolve(x, self.sr, rng)
        return x.astype(np.float32)


class SpecAugment(nn.Module):
    """Frequency + time masking on (B, T, D) features."""

    def __init__(
        self,
        F: int = 27,
        T: int = 40,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        p: float = 0.5,
    ):
        super().__init__()
        self.F = F
        self.T = T
        self.nF = n_freq_masks
        self.nT = n_time_masks
        self.p = p

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return feats
        out = feats
        B, T, D = out.shape
        for b in range(B):
            if torch.rand(1).item() > self.p:
                continue
            for _ in range(self.nF):
                f = int(torch.randint(0, self.F + 1, (1,)).item())
                if f == 0 or f >= D:
                    continue
                f0 = int(torch.randint(0, D - f, (1,)).item())
                out[b, :, f0:f0 + f] = 0.0
            for _ in range(self.nT):
                t = int(torch.randint(0, self.T + 1, (1,)).item())
                if t == 0 or t >= T:
                    continue
                t0 = int(torch.randint(0, T - t, (1,)).item())
                out[b, t0:t0 + t, :] = 0.0
        return out
