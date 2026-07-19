"""캐시된 SSL feature로 adapter + classifier만 학습 (SSL forward, worker 없음).

    python train_cached.py --config configs/colab_a1c1.yaml --cache-dir /content/cache --tag a1c1_cached
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.cached_dataset import build_cached_dataloaders
from models.adapter import build_adapter
from models.classifier import build_classifier
from utils.metrics import compute_eer
from utils.tracking import ExperimentTracker
from utils.run_meta import save_run_meta


class CachedHeadModel(nn.Module):

    def __init__(self, cfg: dict, in_dim: int = 1024):
        super().__init__()
        self.adapter = build_adapter(cfg["model"]["adapter"], in_dim=in_dim)
        z_dim = cfg["model"]["adapter"]["hidden_dim"]
        self.classifier = build_classifier(cfg["model"]["classifier"], in_dim=z_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        z = self.adapter(feats)               # (B, T, z_dim)
        logit = self.classifier(z)            # (B,)
        return logit


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(want: str) -> torch.device:
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cosine_with_warmup(step, total, warmup, peak_lr, min_lr):
    if step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_n = 0
    for batch in tqdm(loader, desc=f"eval[{loader.dataset.split_name}]", leave=False):
        feats = batch["feat"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True).float()
        logit = model(feats)
        loss = bce(logit, label)
        total_loss += float(loss.item())
        total_n += label.numel()
        all_scores.append(torch.sigmoid(logit).detach().cpu().numpy())
        all_labels.append(label.detach().cpu().numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels).astype(np.int64)
    eer, thr = compute_eer(scores, labels)
    return {
        "loss": total_loss / max(1, total_n),
        "eer": eer,
        "threshold": thr,
        "n_pos": int(labels.sum()),
        "n_neg": int((labels == 0).sum()),
    }


def train_one_epoch(
    model, loader, optimizer, scaler, device, cfg,
    epoch: int, total_steps: int, warmup_steps: int, step_state: Dict[str, int],
    tracker: "ExperimentTracker" = None,
):
    import time
    model.train()
    bce = nn.BCEWithLogitsLoss()
    epoch_start = time.time()
    log_every = cfg["train"].get("log_interval", 20)
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    peak_lr = cfg["train"]["lr_peak"]
    min_lr = 1e-6
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    label_smooth = cfg["model"]["classifier"].get("label_smoothing", 0.0)

    sums = {"loss": 0.0, "bce": 0.0, "worker": 0.0}
    n_seen = 0
    pbar = tqdm(loader, desc=f"ep{epoch} train", leave=False)
    for step, batch in enumerate(pbar):
        global_step = step_state["global"]
        lr = cosine_with_warmup(global_step, total_steps, warmup_steps, peak_lr, min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        feats = batch["feat"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True).float()
        if label_smooth > 0:
            label_target = label * (1.0 - label_smooth) + 0.5 * label_smooth
        else:
            label_target = label

        with autocast(enabled=use_amp):
            logit = model(feats)
            loss = bce(logit, label_target)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
            optimizer.step()

        bs = label.numel()
        sums["loss"] += float(loss.item()) * bs
        sums["bce"] += float(loss.item()) * bs   # no worker, bce == total
        n_seen += bs
        step_state["global"] += 1

        if step % log_every == 0:
            pbar.set_postfix(lr=f"{lr:.2e}", loss=f"{sums['loss']/n_seen:.4f}")
            if tracker is not None:
                elapsed = max(1e-6, time.time() - epoch_start)
                gpu_mem_mb = (
                    torch.cuda.max_memory_allocated(device) / 1e6
                    if device.type == "cuda" else 0.0
                )
                tracker.log_step({
                    "train/loss": sums["loss"] / n_seen,
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "train/samples_per_s": n_seen / elapsed,
                    "sys/gpu_mem_mb": gpu_mem_mb,
                    "epoch": epoch,
                }, step=step_state["global"])

    return {
        "loss": sums["loss"] / max(1, n_seen),
        "bce": sums["bce"] / max(1, n_seen),
        "worker": 0.0,
        "epoch_time_s": time.time() - epoch_start,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-dir", required=True,
                    help="Directory with train/ and dev/ subdirs of cached features")
    ap.add_argument("--tag", default="cached")
    ap.add_argument("--adapter", choices=["bilstm", "se_res2net"], default=None)
    ap.add_argument("--classifier", choices=["lcnn_big", "senet12"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--dev-batch-size", type=int, default=None)
    ap.add_argument("--dev-subset", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0,
                    help="if >0, subsample this many train items for a smoke test")
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=None,
                    help="Force-enable W&B logging (overrides cfg['tracking']['backend']).")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false",
                    help="Force-disable W&B logging; local train_log.jsonl still written.")
    ap.add_argument("--wandb-project", default=None,
                    help="W&B project name (overrides cfg['tracking']['project']).")
    ap.add_argument("--wandb-run-name", default=None,
                    help="W&B run name (default: --tag).")
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.adapter is not None:
        cfg["model"]["adapter"]["type"] = args.adapter
    if args.classifier is not None:
        cfg["model"]["classifier"]["type"] = args.classifier
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size

    set_seed(cfg.get("seed", 42))
    device = select_device(cfg.get("device", "cuda"))
    print(f"[device] {device}")
    print(f"[mode] cached SSL features (no waveform, no SSL forward, no workers)")
    print(f"[arch] adapter={cfg['model']['adapter']['type']} classifier={cfg['model']['classifier']['type']}")

    save_dir = Path(cfg["train"]["save_dir"]) / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config_used.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)

    save_run_meta(cfg, save_dir, args)
    tracker = ExperimentTracker.from_config(
        cfg, run_name=args.wandb_run_name or args.tag, save_dir=save_dir,
        enabled=args.wandb, project=args.wandb_project,
    )

    loaders = build_cached_dataloaders(cfg, args.cache_dir, splits=["train", "dev"])

    if args.smoke and args.smoke > 0:
        ds = loaders["train"].dataset
        ds.entries = ds.entries[: args.smoke]
        ds_dev = loaders["dev"].dataset
        ds_dev.entries = ds_dev.entries[: max(50, args.smoke // 5)]
        from torch.utils.data import DataLoader
        from data.cached_dataset import collate_cached
        loaders["train"] = DataLoader(
            ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
            num_workers=cfg["data"]["num_workers"], collate_fn=collate_cached, drop_last=True,
        )
        loaders["dev"] = DataLoader(
            ds_dev, batch_size=cfg["train"]["batch_size"], shuffle=False,
            num_workers=cfg["data"]["num_workers"], collate_fn=collate_cached,
        )

    if args.dev_subset and args.dev_subset > 0:
        ds_dev = loaders["dev"].dataset
        random.Random(cfg.get("seed", 42)).shuffle(ds_dev.entries)
        ds_dev.entries = ds_dev.entries[: args.dev_subset]
        print(f"[dev-subset] using {len(ds_dev.entries)} dev utterances for per-epoch eval")
    if args.dev_batch_size:
        from torch.utils.data import DataLoader
        from data.cached_dataset import collate_cached
        loaders["dev"] = DataLoader(
            loaders["dev"].dataset, batch_size=args.dev_batch_size, shuffle=False,
            num_workers=cfg["data"]["num_workers"], collate_fn=collate_cached,
        )
        print(f"[dev-batch] dev loader batch_size = {args.dev_batch_size}")

    print(f"[data] train={len(loaders['train'].dataset)}  dev={len(loaders['dev'].dataset)}")

    model = CachedHeadModel(cfg, in_dim=cfg["model"]["ssl"].get("feature_dim", 1024)).to(device)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params = {n_train_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr_peak"],
        betas=tuple(cfg["train"]["betas"]),
        weight_decay=cfg["train"]["weight_decay"],
    )
    scaler = GradScaler(enabled=cfg["train"].get("amp", True) and device.type == "cuda")

    epochs = cfg["train"]["epochs"]
    steps_per_epoch = max(1, len(loaders["train"]))
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * cfg["train"]["warmup_ratio"]))
    step_state = {"global": 0}

    best_eer = float("inf")
    best_threshold = 0.5
    bad_epochs = 0
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model, loaders["train"], optimizer, scaler, device, cfg,
            epoch, total_steps, warmup_steps, step_state, tracker=tracker,
        )
        dev_stats = evaluate(model, loaders["dev"], device)
        print(f"[ep{epoch}] train loss={train_stats['loss']:.4f} bce={train_stats['bce']:.4f} | "
              f"dev EER={dev_stats['eer']*100:.2f}% loss={dev_stats['loss']:.4f}")
        history.append({"epoch": epoch, "train": train_stats, "dev": dev_stats})

        tracker.log_epoch({
            "train/loss": train_stats["loss"],
            "train/epoch_time_s": train_stats.get("epoch_time_s", 0.0),
            "dev/loss": dev_stats["loss"],
            "dev/eer": dev_stats["eer"],
        }, epoch=epoch)

        improved = dev_stats["eer"] < best_eer - 1e-5
        if improved:
            best_eer = dev_stats["eer"]
            best_threshold = dev_stats["threshold"]
            bad_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "global_step": step_state["global"],
                "best_eer": best_eer,
                "best_threshold": best_threshold,
            }, save_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["train"]["early_stopping_patience"]:
                print(f"[early-stop] no improvement for {bad_epochs} epochs.")
                break

    torch.save({
        "model": model.state_dict(),
        "config": cfg,
        "epoch": epoch,
        "global_step": step_state["global"],
        "best_eer": best_eer,
        "best_threshold": best_threshold,
    }, save_dir / "last.pt")

    with open(save_dir / "history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    tracker.log_summary({
        "best/eer": best_eer,
        "best/threshold": best_threshold,
    })
    tracker.finish()

    print(f"[done] best dev EER = {best_eer*100:.2f}% threshold = {best_threshold:.4f}")
    print(f"[done] checkpoint: {save_dir/'best.pt'}")


if __name__ == "__main__":
    main()
