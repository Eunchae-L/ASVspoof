"""발화별 SSL feature를 {out}/{file_id}.npy (T, D) fp16으로 저장 (train_cached.py용).

layer weight가 0 초기값이라 hidden state 균등 평균이 저장됨 -> layer weight 학습 불가.
증강도 한 번 고정되므로 epoch마다 RawBoost 다양성은 없음.

    python scripts/cache_ssl_features.py --config configs/colab_a1c1.yaml --split train --out /content/cache/train
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.augment import WaveformAugment
from data.dataset import ASVspoof5Track1Dataset, build_splits
from models.ssl_backbone import SSLBackbone


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True, choices=["train", "dev", "eval"])
    ap.add_argument("--out", required=True, help="Output directory for per-utterance .npy files")
    ap.add_argument("--use", default="clean", choices=["clean", "distorted-fixed"],
                    help="clean = original waveform; distorted-fixed = one-shot RawBoost augmentation")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only first N utterances")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=2)
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from utils.compliance import check_config
    check_config(cfg)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 다른 백본으로 만든 캐시와 섞이지 않게 백본 이름을 같이 저장
    ssl_name = cfg["model"]["ssl"]["name_or_path"]
    marker = out_dir / "_ssl_backbone.txt"
    if marker.exists():
        prev = marker.read_text().strip()
        if prev != ssl_name:
            raise SystemExit(f"[cache] {out_dir} 는 '{prev}' 캐시 (현재 '{ssl_name}'). 다른 --out 경로 사용.")
    elif any(out_dir.glob("*.npy")):
        raise SystemExit(f"[cache] {out_dir} 에 백본 표시 없는 .npy 가 있음. 비우거나 다른 경로 사용.")
    marker.write_text(ssl_name + "\n")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    splits = build_splits(cfg, splits=[args.split])
    if args.split not in splits:
        raise RuntimeError(
            f"split '{args.split}' not available - protocol/audio dir missing."
        )

    aug = WaveformAugment(cfg["augment"], cfg["data"]["sample_rate"]) \
        if args.use == "distorted-fixed" else None
    ds = ASVspoof5Track1Dataset(
        splits[args.split],
        sample_rate=cfg["data"]["sample_rate"],
        clip_seconds=cfg["data"]["clip_seconds"],
        audio_ext=cfg["data"].get("audio_ext", ".flac"),
        augment=aug,
        is_train=(args.use == "distorted-fixed"),
    )
    if args.limit and args.limit > 0:
        ds.entries = ds.entries[: args.limit]

    print(f"[data] {args.split}: {len(ds)} utterances")
    print(f"[cache] using '{args.use}' waveform → SSL features")

    ssl = SSLBackbone(cfg["model"]["ssl"]["name_or_path"], layer_weighted_sum=True).to(device)
    ssl.eval()

    wav_key = "clean" if args.use == "clean" else "distorted"

    def _collate(batch):
        return {
            "wav": torch.stack([b[wav_key] for b in batch], dim=0),
            "file_id": [b["file_id"] for b in batch],
            "label": torch.stack([b["label"] for b in batch], dim=0),
        }

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=_collate,
    )

    n_cached, n_skipped = 0, 0
    use_amp = device.type == "cuda"
    pbar = tqdm(loader, desc=f"cache[{args.split}]")
    with torch.no_grad():
        for batch in pbar:
            file_ids = batch["file_id"]
            missing = [i for i, fid in enumerate(file_ids) if not (out_dir / f"{fid}.npy").exists()]
            if not missing:
                n_skipped += len(file_ids)
                pbar.set_postfix(cached=n_cached, skipped=n_skipped)
                continue

            wav = batch["wav"][missing].to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    feats = ssl(wav)
            else:
                feats = ssl(wav)
            feats_np = feats.detach().cpu().numpy().astype(np.float16)

            for j, i in enumerate(missing):
                fid = file_ids[i]
                np.save(out_dir / f"{fid}.npy", feats_np[j])
                n_cached += 1
            pbar.set_postfix(cached=n_cached, skipped=n_skipped)

    manifest_path = out_dir / "_manifest.tsv"
    with open(manifest_path, "w") as fh:
        fh.write("file_id\tlabel\tkey\n")
        for e in ds.entries:
            fh.write(f"{e.file_id}\t{e.label}\t{e.key}\n")
    print(f"[done] cached {n_cached} new, skipped {n_skipped} existing")
    print(f"[done] manifest → {manifest_path}")


if __name__ == "__main__":
    main()
