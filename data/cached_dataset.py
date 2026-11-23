"""scripts/cache_ssl_features.py로 저장한 SSL feature(.npy, fp16)를 읽는 dataset.

{cache_dir}/{split}/{file_id}.npy, {cache_dir}/{split}/_manifest.tsv
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class CachedFeatureDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
        entries: List[Tuple[str, int, str]],   # (file_id, label, key)
        max_T: int = 200,
        split_name: str = "train",
    ):
        self.cache_dir = cache_dir
        self.entries = entries
        self.max_T = max_T
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        file_id, label, key = self.entries[idx]
        path = os.path.join(self.cache_dir, f"{file_id}.npy")
        feat = np.load(path, mmap_mode="r")               # (T, D) fp16
        feat = np.asarray(feat, dtype=np.float32)

        T_actual = feat.shape[0]
        if T_actual >= self.max_T:
            # train/eval 구분 없이 center crop
            start = (T_actual - self.max_T) // 2
            feat = feat[start: start + self.max_T]
        else:
            pad = self.max_T - T_actual
            feat = np.pad(feat, ((0, pad), (0, 0)), mode="constant")

        return {
            "feat": torch.from_numpy(feat),               # (T, D) float32
            "label": torch.tensor(label, dtype=torch.long),
            "file_id": file_id,
            "key": key,
        }


def collate_cached(batch: List[dict]) -> dict:
    return {
        "feat": torch.stack([b["feat"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "file_id": [b["file_id"] for b in batch],
        "key": [b["key"] for b in batch],
    }


def load_manifest(cache_dir: str) -> List[Tuple[str, int, str]]:
    path = os.path.join(cache_dir, "_manifest.tsv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Manifest missing: {path}\n"
            f"Did you run scripts/cache_ssl_features.py for this split?"
        )
    # 캐시를 만든 백본 이름. 없으면 어떤 모델로 뽑은 캐시인지 알 수 없어서 쓰지 않는다.
    marker = os.path.join(cache_dir, "_ssl_backbone.txt")
    if not os.path.isfile(marker):
        raise FileNotFoundError(
            f"{marker} 없음. 백본을 알 수 없는 캐시는 쓰지 않음, "
            f"scripts/cache_ssl_features.py로 다시 생성."
        )
    from utils.compliance import check_ssl
    with open(marker) as fh:
        check_ssl(fh.read().strip())
    out = []
    with open(path) as fh:
        header = fh.readline().strip().split("\t")
        assert header == ["file_id", "label", "key"], f"Unexpected manifest header: {header}"
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            file_id, label, key = parts[0], int(parts[1]), parts[2]
            out.append((file_id, label, key))
    return out


def build_cached_dataloaders(
    cfg: dict,
    cache_dir: str,
    splits: Optional[List[str]] = None,
) -> Dict[str, DataLoader]:
    if splits is None:
        splits = ["train", "dev"]
        if os.path.isdir(os.path.join(cache_dir, "eval")):
            splits.append("eval")

    # SSL frame rate 50 Hz (hop 320 @ 16k)
    frame_rate = cfg["model"]["ssl"].get("frame_rate_hz", 50)
    max_T = int(round(cfg["data"]["clip_seconds"] * frame_rate))

    loaders: Dict[str, DataLoader] = {}
    common = dict(
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        collate_fn=collate_cached,
    )

    for name in splits:
        sub_dir = os.path.join(cache_dir, name)
        if not os.path.isdir(sub_dir):
            raise FileNotFoundError(f"Cache subdir missing: {sub_dir}")
        entries = load_manifest(sub_dir)
        ds = CachedFeatureDataset(
            cache_dir=sub_dir, entries=entries, max_T=max_T, split_name=name,
        )

        if name == "train" and cfg["train"].get("balance_real_fake", True):
            labels = np.array([e[1] for e in entries])
            class_count = np.bincount(labels, minlength=2).astype(np.float32)
            class_count = np.where(class_count == 0, 1.0, class_count)
            weights = 1.0 / class_count[labels]
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(weights).float(),
                num_samples=len(weights),
                replacement=True,
            )
            loaders[name] = DataLoader(
                ds, batch_size=cfg["train"]["batch_size"],
                sampler=sampler, drop_last=True, **common,
            )
        else:
            loaders[name] = DataLoader(
                ds, batch_size=cfg["train"]["batch_size"],
                shuffle=False, drop_last=False, **common,
            )

    return loaders
