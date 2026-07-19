"""ASVspoof 5 Track 1 dataset.

<root>/flac_T, flac_D, flac_E_eval + <root>/protocols/*.tsv
sample: distorted(증강 파형), clean(worker target용), label(1=spoof), file_id, key
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from data.augment import WaveformAugment
from data.protocols import ProtocolEntry, parse_asvspoof5_protocol


@dataclass
class SplitInfo:
    entries: List[ProtocolEntry]
    audio_dir: str
    name: str            # 'train' / 'dev' / 'eval'


class ASVspoof5Track1Dataset(Dataset):
    def __init__(
        self,
        split: SplitInfo,
        sample_rate: int,
        clip_seconds: float,
        audio_ext: str = ".flac",
        augment: Optional[WaveformAugment] = None,
        is_train: bool = True,
        peak_normalize: bool = True,
    ):
        self.entries = split.entries
        self.audio_dir = split.audio_dir
        self.split_name = split.name
        self.sr = sample_rate
        self.num_samples = int(round(sample_rate * clip_seconds))
        self.audio_ext = audio_ext
        self.augment = augment
        self.is_train = is_train
        self.peak_normalize = peak_normalize

    def __len__(self) -> int:
        return len(self.entries)

    def _load_audio(self, file_id: str) -> np.ndarray:
        path = os.path.join(self.audio_dir, f"{file_id}{self.audio_ext}")
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != self.sr:
            ratio = self.sr / sr
            t_old = np.arange(len(x))
            t_new = np.linspace(0, len(x) - 1, int(round(len(x) * ratio)))
            x = np.interp(t_new, t_old, x).astype(np.float32)
        return x.astype(np.float32)

    def _crop_or_pad(self, x: np.ndarray, rng: random.Random) -> np.ndarray:
        n = self.num_samples
        if x.shape[0] >= n:
            if self.is_train:
                start = rng.randint(0, x.shape[0] - n)
            else:
                start = (x.shape[0] - n) // 2
            return x[start:start + n]
        pad = n - x.shape[0]
        if x.shape[0] >= 2:
            return np.pad(x, (pad // 2, pad - pad // 2), mode="reflect").astype(np.float32)
        return np.pad(x, (0, pad), mode="constant").astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        rng = random.Random()
        x = self._load_audio(entry.file_id)
        if self.peak_normalize:
            peak = float(np.max(np.abs(x)) + 1e-9)
            x = (x / peak * 0.95).astype(np.float32)

        clean = self._crop_or_pad(x, rng)
        if self.augment is not None and self.is_train:
            distorted = self.augment(clean.copy(), rng=np.random.default_rng())
            distorted = np.clip(distorted, -1.0, 1.0).astype(np.float32)
        else:
            distorted = clean

        return {
            "distorted": torch.from_numpy(distorted),
            "clean": torch.from_numpy(clean),
            "label": torch.tensor(entry.label, dtype=torch.long),
            "file_id": entry.file_id,
            "key": entry.key,
        }


def collate_batch(batch: List[dict]) -> dict:
    return {
        "distorted": torch.stack([b["distorted"] for b in batch], dim=0),
        "clean": torch.stack([b["clean"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "file_id": [b["file_id"] for b in batch],
        "key": [b["key"] for b in batch],
    }


def _ensure_file(path: str, hint: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {path}. {hint}"
        )


def build_splits(cfg: dict, splits: Optional[List[str]] = None) -> Dict[str, SplitInfo]:
    if cfg["data"].get("dataset_name", "asvspoof5") != "asvspoof5":
        raise NotImplementedError(
            "v1 supports only dataset_name='asvspoof5'. FoR/LibriSeVoc are v2."
        )
    root = cfg["data"]["root"]

    if splits is None:
        splits = ["train", "dev", "eval"]

    pairs = {
        "train": (cfg["data"]["protocol_train"], cfg["data"]["audio_train"]),
        "dev":   (cfg["data"]["protocol_dev"],   cfg["data"]["audio_dev"]),
        "eval":  (cfg["data"]["protocol_eval"],  cfg["data"]["audio_eval"]),
    }

    out: Dict[str, SplitInfo] = {}
    for name in splits:
        if name not in pairs:
            continue
        protocol_rel, audio_rel = pairs[name]
        protocol_path = os.path.join(root, protocol_rel)
        audio_dir = os.path.join(root, audio_rel)
        if not os.path.isfile(protocol_path):
            if name == "eval":
                # eval은 없어도 됨
                continue
            _ensure_file(
                protocol_path,
                f"Did you extract ASVspoof5_protocols.tar.gz into {root}/protocols/ ?",
            )
        if not os.path.isdir(audio_dir):
            if name == "eval":
                continue
            raise FileNotFoundError(
                f"Audio dir missing: {audio_dir}. Did you extract flac_{name[0].upper()}_*.tar?"
            )
        entries = parse_asvspoof5_protocol(protocol_path)
        out[name] = SplitInfo(entries=entries, audio_dir=audio_dir, name=name)
    return out


def build_dataloaders(cfg: dict) -> Dict[str, DataLoader]:
    splits = build_splits(cfg)
    aug = WaveformAugment(cfg["augment"], cfg["data"]["sample_rate"])

    def _ds(split_name: str, is_train: bool) -> ASVspoof5Track1Dataset:
        return ASVspoof5Track1Dataset(
            splits[split_name],
            sample_rate=cfg["data"]["sample_rate"],
            clip_seconds=cfg["data"]["clip_seconds"],
            audio_ext=cfg["data"].get("audio_ext", ".flac"),
            augment=aug if is_train else None,
            is_train=is_train,
            peak_normalize=cfg["data"].get("peak_normalize", True),
        )

    train_ds = _ds("train", is_train=True)
    dev_ds = _ds("dev", is_train=False)

    if cfg["train"].get("balance_real_fake", True):
        labels = np.array([e.label for e in splits["train"].entries])
        class_count = np.bincount(labels, minlength=2).astype(np.float32)
        class_count = np.where(class_count == 0, 1.0, class_count)
        weights = 1.0 / class_count[labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    common = dict(
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        collate_fn=collate_batch,
    )
    # num_workers=0이면 persistent_workers/prefetch_factor 넘기면 에러
    if cfg["data"]["num_workers"] > 0:
        common["persistent_workers"] = cfg["data"].get("persistent_workers", False)
        common["prefetch_factor"] = cfg["data"].get("prefetch_factor", 2)

    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            train_ds,
            batch_size=cfg["train"]["batch_size"],
            sampler=sampler,
            shuffle=shuffle,
            drop_last=True,
            **common,
        ),
        "dev": DataLoader(
            dev_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            drop_last=False,
            **common,
        ),
    }
    if "eval" in splits:
        loaders["eval"] = DataLoader(
            _ds("eval", is_train=False),
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            drop_last=False,
            **common,
        )
    return loaders
