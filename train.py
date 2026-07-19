"""ASVspoof 5 Track 1 학습 (SSL + adapter + classifier + multi-task workers).

    python train.py --config configs/baseline.yaml --tag bilstm_lcnn [--adapter ...] [--classifier ...]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import build_dataloaders
from models.model import SSLMultiTaskModel
from models.workers import TargetComputer, worker_loss
from utils.metrics import compute_eer, compute_min_dcf, compute_cllr
from utils.tracking import ExperimentTracker
from utils.run_meta import save_run_meta


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(want: str) -> torch.device:
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if want == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype:
    """BF16 on compute capability >= 8 (A100/H100), else FP16."""
    if device.type == "cuda":
        try:
            cap = torch.cuda.get_device_capability(device)
            if cap[0] >= 8:
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16


def cosine_with_warmup(step: int, total: int, warmup: int, peak_lr: float, min_lr: float) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def print_history_table(history, tag, total_epochs, best_min_dcf, best_threshold,
                        best_eer=None):
    bw = 96
    print()
    print("═" * bw)
    title = f" Training history - tag={tag}  ({len(history)}/{total_epochs} epochs done) "
    print(title.center(bw, "═"))
    print("═" * bw)
    print(f"{'ep':>3} │ {'tr_loss':>8} │ {'bce':>6} │ {'wrk':>6} │ "
          f"{'dev_EER':>8} │ {'dev_mDCF':>9} │ {'dev_Cllr':>8} │ {'dev_loss':>8} │ best")
    print("─" * 3 + "─┼─" + "─" * 8 + "─┼─" + "─" * 6 + "─┼─" + "─" * 6 + "─┼─"
          + "─" * 8 + "─┼─" + "─" * 9 + "─┼─" + "─" * 8 + "─┼─" + "─" * 8 + "─┼─────")
    for h in history:
        ep = h["epoch"]
        tl = h["train"]["loss"]
        bce = h["train"]["bce"]
        w = h["train"]["worker"]
        eer = h["dev"]["eer"] * 100
        mdcf = h["dev"].get("min_dcf", float("nan"))
        cllr = h["dev"].get("cllr", float("nan"))
        dl = h["dev"]["loss"]
        is_best = abs(mdcf - best_min_dcf) < 1e-6
        star = " *" if is_best else ""
        print(f"{ep:>3} │ {tl:>8.4f} │ {bce:>6.3f} │ {w:>6.3f} │ "
              f"{eer:>7.2f}% │ {mdcf:>9.4f} │ {cllr:>8.4f} │ {dl:>8.4f} │{star}")
    print("═" * bw)
    msg = f"  best dev minDCF: {best_min_dcf:.4f}  threshold: {best_threshold:.4f}"
    if best_eer is not None:
        msg += f"  (EER at best: {best_eer*100:.2f}%)"
    print(msg)
    print("═" * bw)
    print()


def evaluate(model, loader, device, cfg=None) -> Dict[str, float]:
    """Per-epoch dev eval (EER, minDCF, Cllr).

    지표는 공식 방향(bona fide = positive)으로 계산하므로 fake logit의 부호를 뒤집는다.
    """
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_n = 0
    use_amp = device.type == "cuda"
    amp_dtype = autocast_dtype(device) if use_amp else torch.float32
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval[{loader.dataset.split_name}]", leave=False):
            wav = batch["distorted"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True).float()
            with autocast(enabled=use_amp, dtype=amp_dtype):
                logit, _, _ = model(wav, apply_specaug=False)
                loss = bce(logit, label)
            total_loss += float(loss.item())
            total_n += label.numel()
            all_logits.append(logit.float().detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels).astype(np.int64)   # 1 = spoof

    eval_cfg = (cfg or {}).get("eval", {}) if cfg else {}
    c_miss = eval_cfg.get("mindcf_c_miss", 1.0)
    c_fa = eval_cfg.get("mindcf_c_fa", 10.0)
    p_spoof = eval_cfg.get("mindcf_p_target", 0.05)   # key 이름은 p_target이지만 spoof prior

    score_bona = -logits
    label_bona = (labels == 0).astype(np.int64)   # 1 = bona fide

    eer, eer_thr = compute_eer(score_bona, label_bona)
    min_dcf, mdcf_thr = compute_min_dcf(score_bona, label_bona, c_miss, c_fa, p_spoof)
    cllr = compute_cllr(score_bona, label_bona)
    return {
        "loss": total_loss / max(1, total_n),
        "eer": eer,
        "threshold": eer_thr,
        "min_dcf": min_dcf,
        "min_dcf_threshold": mdcf_thr,
        "cllr": cllr,
        "n_pos": int(label_bona.sum()),            # bona fide
        "n_neg": int((label_bona == 0).sum()),     # spoof
    }


def _ckpt(model, optimizer, cfg, epoch, global_step,
          best_min_dcf, best_eer, best_threshold, bad_epochs,
          epoch_completed, step_in_epoch=0):
    """epoch_completed=False 이면 epoch 중간 저장."""
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "epoch": epoch,
        "global_step": global_step,
        "best_min_dcf": best_min_dcf,
        "best_eer": best_eer,
        "best_threshold": best_threshold,
        "bad_epochs": bad_epochs,
        "epoch_completed": epoch_completed,
        "step_in_epoch": step_in_epoch,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    cfg,
    target_computer: TargetComputer,
    epoch: int,
    total_steps: int,
    warmup_steps: int,
    step_state: Dict[str, int],
    save_dir=None,
    save_every: int = 0,
    start_step: int = 0,
    ckpt_meta: Dict = None,
    tracker: "ExperimentTracker" = None,
) -> Dict[str, float]:
    import time
    model.train()
    bce = nn.BCEWithLogitsLoss()
    epoch_start = time.time()
    lambda_w = cfg["loss"]["lambda_workers"]
    label_smooth = cfg["model"]["classifier"].get("label_smoothing", 0.0)

    log_every = cfg["train"].get("log_interval", 20)
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    peak_lr = cfg["train"]["lr_peak"]
    min_lr = 1e-6
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    amp_dtype = autocast_dtype(device) if use_amp else torch.float32
    # GradScaler is only needed for FP16; BF16 has wide enough dynamic range.
    use_scaler = use_amp and amp_dtype == torch.float16

    sums = {"loss": 0.0, "bce": 0.0, "worker": 0.0}
    n_seen = 0
    pbar = tqdm(loader, desc=f"ep{epoch} train", leave=False)
    for step, batch in enumerate(pbar):
        if step < start_step:
            continue  # resume: 이미 끝낸 step은 skip
        global_step = step_state["global"]
        lr = cosine_with_warmup(global_step, total_steps, warmup_steps, peak_lr, min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        wav = batch["distorted"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True).float()

        if label_smooth > 0:
            label_target = label * (1.0 - label_smooth) + 0.5 * label_smooth
        else:
            label_target = label

        with autocast(enabled=use_amp, dtype=amp_dtype):
            logit, worker_preds, _ = model(wav, apply_specaug=True)
            loss_bce = bce(logit, label_target)

            loss_worker = torch.tensor(0.0, device=device)
            if lambda_w > 0 and len(worker_preds) > 0:
                with torch.cuda.amp.autocast(enabled=False):
                    targets = target_computer(clean.float())
                    losses_w = worker_loss(
                        {k: v.float() for k, v in worker_preds.items()},
                        targets,
                    )
                if len(losses_w) > 0:
                    loss_worker = sum(losses_w.values()) / len(losses_w)

            loss = loss_bce + lambda_w * loss_worker

        optimizer.zero_grad(set_to_none=True)
        if use_scaler:
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

        bs = label.shape[0]
        sums["loss"] += float(loss.item()) * bs
        sums["bce"] += float(loss_bce.item()) * bs
        sums["worker"] += float(loss_worker.item()) * bs if isinstance(loss_worker, torch.Tensor) else 0.0
        n_seen += bs
        step_state["global"] += 1

        if (step + 1) % log_every == 0:
            pbar.set_postfix({
                "lr": f"{lr:.2e}",
                "loss": f"{sums['loss']/n_seen:.4f}",
                "bce": f"{sums['bce']/n_seen:.4f}",
                "w": f"{sums['worker']/n_seen:.4f}",
            })
            if tracker is not None:
                elapsed = max(1e-6, time.time() - epoch_start)
                gpu_mem_mb = (
                    torch.cuda.max_memory_allocated(device) / 1e6
                    if device.type == "cuda" else 0.0
                )
                tracker.log_step({
                    "train/loss": sums["loss"] / n_seen,
                    "train/bce": sums["bce"] / n_seen,
                    "train/worker": sums["worker"] / n_seen,
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "train/samples_per_s": n_seen / elapsed,
                    "sys/gpu_mem_mb": gpu_mem_mb,
                    "epoch": epoch,
                }, step=step_state["global"])

        # N step마다 last.pt 갱신
        if save_every and save_dir is not None and (step + 1) % save_every == 0:
            m = ckpt_meta or {}
            torch.save(
                _ckpt(model, optimizer, cfg, epoch, step_state["global"],
                      m.get("best_min_dcf", float("inf")), m.get("best_eer", float("inf")),
                      m.get("best_threshold", 0.5), m.get("bad_epochs", 0),
                      epoch_completed=False, step_in_epoch=step + 1),
                save_dir / "last.pt",
            )
    out = {k: (v / max(1, n_seen)) for k, v in sums.items()}
    out["epoch_time_s"] = time.time() - epoch_start
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--adapter", choices=["bilstm", "se_res2net"], default=None)
    ap.add_argument("--classifier", choices=["lcnn_big", "senet12"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--smoke", type=int, default=0,
                    help="if >0, subsample this many train items for a smoke test")
    ap.add_argument("--dev-batch-size", type=int, default=None,
                    help="Override batch size for per-epoch dev eval (no grad → can be 4-8x larger).")
    ap.add_argument("--dev-subset", type=int, default=0,
                    help="If >0, randomly subsample N dev utterances for per-epoch eval. "
                         "Final full-dev eval is done separately by evaluate.py.")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to last.pt to resume from (overrides cfg['train']['resume']).")
    ap.add_argument("--save-every", type=int, default=0,
                    help="If >0, save last.pt every N optimizer steps (intra-epoch checkpoint). "
                         "Lets a long epoch resume mid-way instead of restarting from scratch.")
    ap.add_argument("--save-all-epochs", action="store_true",
                    help="Also save a separate epoch_NN.pt every epoch (keeps every epoch, "
                         "not just best/last). ~1.2GB each - watch disk.")
    # default None -> cfg['tracking'] 따름
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
    print(f"[dataset] {cfg['data'].get('dataset_name')}")
    print(f"[ssl] {cfg['model']['ssl']['name_or_path']}")
    print(f"[arch] adapter={cfg['model']['adapter']['type']} "
          f"classifier={cfg['model']['classifier']['type']} "
          f"lambda_w={cfg['loss']['lambda_workers']}")

    save_dir = Path(cfg["train"]["save_dir"]) / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    save_run_meta(cfg, save_dir, args)
    tracker = ExperimentTracker.from_config(
        cfg, run_name=args.wandb_run_name or args.tag, save_dir=save_dir,
        enabled=args.wandb, project=args.wandb_project,
    )

    loaders = build_dataloaders(cfg)
    if args.smoke and args.smoke > 0:
        ds = loaders["train"].dataset
        ds.entries = ds.entries[: args.smoke]
        loaders["train"] = torch.utils.data.DataLoader(
            ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
            num_workers=cfg["data"]["num_workers"], collate_fn=loaders["train"].collate_fn,
            drop_last=True,
        )
        ds_dev = loaders["dev"].dataset
        ds_dev.entries = ds_dev.entries[: max(50, args.smoke // 5)]
        loaders["dev"] = torch.utils.data.DataLoader(
            ds_dev, batch_size=cfg["train"]["batch_size"], shuffle=False,
            num_workers=cfg["data"]["num_workers"], collate_fn=loaders["dev"].collate_fn,
        )

    # per-epoch dev는 subset + 큰 batch로
    if args.dev_subset and args.dev_subset > 0:
        import random as _random
        ds_dev = loaders["dev"].dataset
        _random.Random(cfg.get("seed", 42)).shuffle(ds_dev.entries)
        ds_dev.entries = ds_dev.entries[: args.dev_subset]
        print(f"[dev-subset] using {len(ds_dev.entries)} dev utterances for per-epoch eval")
    if args.dev_batch_size:
        ds_dev = loaders["dev"].dataset
        loaders["dev"] = torch.utils.data.DataLoader(
            ds_dev, batch_size=args.dev_batch_size, shuffle=False,
            num_workers=cfg["data"]["num_workers"], collate_fn=loaders["dev"].collate_fn,
        )
        print(f"[dev-batch] dev loader batch_size = {args.dev_batch_size}")

    print(f"[data] train={len(loaders['train'].dataset)}  dev={len(loaders['dev'].dataset)}"
          + (f"  eval={len(loaders['eval'].dataset)}" if "eval" in loaders else "  eval=(not extracted)"))

    model = SSLMultiTaskModel(cfg).to(device)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params = {n_train_params/1e6:.2f}M")

    target_computer = TargetComputer(cfg["model"]["workers"], device=device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr_peak"],
        betas=tuple(cfg["train"]["betas"]),
        weight_decay=cfg["train"]["weight_decay"],
    )
    # GradScaler는 FP16일 때만
    _use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    _amp_dtype = autocast_dtype(device) if _use_amp else torch.float32
    scaler = GradScaler(enabled=_use_amp and _amp_dtype == torch.float16)
    print(f"[amp] enabled={_use_amp}  dtype={str(_amp_dtype).split('.')[-1]}  "
          f"grad_scaler={'on' if scaler.is_enabled() else 'off'}")

    epochs = cfg["train"]["epochs"]
    steps_per_epoch = max(1, len(loaders["train"]))
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * cfg["train"]["warmup_ratio"]))
    step_state = {"global": 0}

    resume_path = args.resume or cfg["train"].get("resume")
    start_epoch = 1
    resume_start_step = 0
    best_min_dcf = float("inf")
    best_eer = float("inf")
    best_threshold = 0.5
    bad_epochs = 0
    history: List[Dict] = []

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        step_state["global"] = ckpt.get("global_step", 0)
        # mid-epoch ckpt면 같은 epoch의 step_in_epoch부터 재개. 필드 없는 옛 ckpt는 epoch 완료로 취급.
        if ckpt.get("epoch_completed", True):
            start_epoch = ckpt.get("epoch", 0) + 1
        else:
            start_epoch = ckpt.get("epoch", 1)
            resume_start_step = ckpt.get("step_in_epoch", 0)
            print(f"[resume] mid-epoch ckpt → epoch {start_epoch}, "
                  f"step {resume_start_step}부터 재개")
        best_min_dcf = ckpt.get("best_min_dcf", float("inf"))
        best_eer = ckpt.get("best_eer", float("inf"))
        best_threshold = ckpt.get("best_threshold", 0.5)
        bad_epochs = ckpt.get("bad_epochs", 0)
        print(f"[resume] {resume_path} | done epoch={ckpt.get('epoch')}, "
              f"next epoch={start_epoch}, best minDCF={best_min_dcf:.4f}, "
              f"EER={best_eer*100:.2f}%, bad={bad_epochs}")
        hist_path = save_dir / "history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history = json.load(f)
            print(f"[resume] history.json: {len(history)} entries")
            # minDCF 도입 전 ckpt면 history에서 다시 계산
            if best_min_dcf == float("inf"):
                mdcfs = [h["dev"].get("min_dcf") for h in history if h["dev"].get("min_dcf") is not None]
                if mdcfs:
                    best_min_dcf = min(mdcfs)
                    print(f"[resume] derived best_min_dcf={best_min_dcf:.4f} from history.json")

    for epoch in range(start_epoch, epochs + 1):
        train_stats = train_one_epoch(
            model, loaders["train"], optimizer, scaler, device, cfg,
            target_computer, epoch, total_steps, warmup_steps, step_state,
            save_dir=save_dir, save_every=args.save_every,
            start_step=(resume_start_step if epoch == start_epoch else 0),
            ckpt_meta={
                "best_min_dcf": best_min_dcf, "best_eer": best_eer,
                "best_threshold": best_threshold, "bad_epochs": bad_epochs,
            },
            tracker=tracker,
        )
        dev_stats = evaluate(model, loaders["dev"], device, cfg=cfg)
        history.append({"epoch": epoch, "train": train_stats, "dev": dev_stats})

        tracker.log_epoch({
            "train/loss": train_stats["loss"],
            "train/bce": train_stats["bce"],
            "train/worker": train_stats["worker"],
            "train/epoch_time_s": train_stats.get("epoch_time_s", 0.0),
            "dev/loss": dev_stats["loss"],
            "dev/eer": dev_stats["eer"],
            "dev/min_dcf": dev_stats["min_dcf"],
            "dev/cllr": dev_stats.get("cllr", float("nan")),
        }, epoch=epoch)

        improved = dev_stats["min_dcf"] < best_min_dcf - 1e-5
        if improved:
            best_min_dcf = dev_stats["min_dcf"]
            best_eer = dev_stats["eer"]
            best_threshold = dev_stats["min_dcf_threshold"]
            bad_epochs = 0
            torch.save(
                _ckpt(model, optimizer, cfg, epoch, step_state["global"],
                      best_min_dcf, best_eer, best_threshold, bad_epochs,
                      epoch_completed=True),
                save_dir / "best.pt")
        else:
            bad_epochs += 1

        # resume용
        torch.save(
            _ckpt(model, optimizer, cfg, epoch, step_state["global"],
                  best_min_dcf, best_eer, best_threshold, bad_epochs,
                  epoch_completed=True),
            save_dir / "last.pt")
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if args.save_all_epochs:
            torch.save(
                _ckpt(model, optimizer, cfg, epoch, step_state["global"],
                      best_min_dcf, best_eer, best_threshold, bad_epochs,
                      epoch_completed=True),
                save_dir / f"epoch_{epoch:02d}.pt")

        print_history_table(history, tag=args.tag, total_epochs=epochs,
                            best_min_dcf=best_min_dcf, best_threshold=best_threshold,
                            best_eer=best_eer)

        if bad_epochs >= cfg["train"]["early_stopping_patience"]:
            print(f"[early-stop] no improvement for {bad_epochs} epochs.")
            break

    tracker.log_summary({
        "best/min_dcf": best_min_dcf,
        "best/eer": best_eer,
        "best/threshold": best_threshold,
        "best/epoch": next((h["epoch"] for h in history
                            if abs(h["dev"].get("min_dcf", float("inf")) - best_min_dcf) < 1e-6), -1),
    })
    tracker.finish()

    print(f"[done] best dev minDCF = {best_min_dcf:.4f}  (EER = {best_eer*100:.2f}%)  "
          f"threshold = {best_threshold:.4f}")
    print(f"[done] checkpoint: {save_dir/'best.pt'}")


if __name__ == "__main__":
    main()
