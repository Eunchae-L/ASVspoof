"""WavLM-Base 실험 결과 정리: 학습 곡선, 점수 coverage 검사, dev/eval 그림."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

COMBOS = ("a1c1", "a1c2", "a2c1", "a2c2")
ARCH = {"a1c1": "BiLSTM + LCNN", "a1c2": "BiLSTM + SENet12",
        "a2c1": "SE-Res2Net + LCNN", "a2c2": "SE-Res2Net + SENet12"}


def tag(combo):
    return f"{combo}_wavlmB_nomusan_subset50_epoch20"


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + f".{os.getpid()}.tmp")
    fig.savefig(staged, format="png")
    staged.replace(path)


def run_dir(output, combo, runtime=None):
    if runtime:
        local = Path(runtime) / "checkpoints" / tag(combo)
        if (local / "history.json").exists() or (local / "train_log.jsonl").exists():
            return local
    return Path(output) / "checkpoints" / tag(combo)


def records(path):
    """jsonl 읽기. 쓰는 중인 마지막 줄은 건너뜀."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open() as fh:
        for line in fh:
            if not line.endswith("\n"):
                break
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def summary_rows(output, runtime=None):
    output = Path(output)
    state = read_json(output / "status.json", {})
    results = read_json(output / "results.json", {})
    links = read_json(output / "wandb_sync/links.json", {})
    rows = []
    for combo in COMBOS:
        root = run_dir(output, combo, runtime)
        history = read_json(root / "history.json", [])
        dev = read_json(output / "scores" / f"dev_{tag(combo)}_summary.json", {})
        test = read_json(output / "scores" / f"eval_{tag(combo)}_summary.json", {})
        saved = output / "checkpoints" / tag(combo)
        row = {"combo": combo.upper(), "architecture": ARCH[combo],
               "completed_epochs": len(history),
               "training_active": state.get("status") == f"train_{combo}",
               "full_dev_done": results.get(combo, {}).get("full_dev") == "done",
               "full_eval_done": results.get(combo, {}).get("full_eval") == "done",
               "dev_eer_pct": dev.get("eer", float("nan")) * 100,
               "dev_min_dcf": dev.get("min_dcf"),
               "eval_eer_pct": test.get("eer", float("nan")) * 100,
               "eval_min_dcf": test.get("min_dcf"),
               "eval_cllr": test.get("cllr"),
               "eval_rows": test.get("n_with_labels", 0) + test.get("n_hidden_labels", 0),
               "best_path": str(saved / "best.pt"),
               "best_saved": (saved / "best.pt").is_file(),
               "last_saved": (saved / "last.pt").is_file(),
               "wandb_url": links.get(combo, {}).get("url")}
        # best epoch는 체크포인트에서 직접 읽은 값 사용
        manifest = read_json(output / "reports/checkpoint_manifest.json", {})
        row["best_epoch"] = manifest.get(combo, {}).get("best.pt", {}).get("epoch")
        rows.append(row)
    return rows


def checkpoint_manifest(output, verify=False):
    """best/last.pt를 CPU에서 로드해 백본, MUSAN 설정, epoch, sha256 기록."""
    output = Path(output)
    previous = read_json(output / "reports/checkpoint_manifest.json", {})
    manifest = {}
    for combo in COMBOS:
        manifest[combo] = {}
        for name in ("best.pt", "last.pt"):
            p = output / "checkpoints" / tag(combo) / name
            entry = {"path": str(p), "exists": p.is_file(), "load_verified": False}
            if p.is_file():
                st = p.stat()
                entry.update(bytes=st.st_size,
                             saved_at_utc=datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat())
                old = previous.get(combo, {}).get(name, {})
                if old.get("load_verified") and all(old.get(k) == entry.get(k) for k in ("bytes", "saved_at_utc")):
                    entry = old
                elif verify:
                    try:
                        import torch
                        state = torch.load(p, map_location="cpu", weights_only=False)
                        cfg = state.get("config", {})
                        ssl = cfg.get("model", {}).get("ssl", {}).get("name_or_path")
                        aug = cfg.get("augment", {})
                        if ssl != "microsoft/wavlm-base" or aug.get("musan_p", 0) or aug.get("musan_root"):
                            raise ValueError("Checkpoint is not WavLM Base / MUSAN off")
                        if not state.get("model"):
                            raise ValueError("No model state_dict in checkpoint")
                        entry.update(load_verified=True, backbone=ssl,
                                     epoch=state.get("epoch"), step_in_epoch=state.get("step_in_epoch"),
                                     epoch_completed=state.get("epoch_completed"),
                                     best_min_dcf=state.get("best_min_dcf"))
                        del state
                        with p.open("rb") as f:
                            entry["sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
                        if p.stat().st_mtime_ns != st.st_mtime_ns:
                            raise ValueError("Checkpoint changed during verification; retry after saving finishes")
                    except Exception as exc:
                        entry["load_verified"] = False
                        entry["verification_error"] = str(exc)
            manifest[combo][name] = entry
    write_json(output / "reports/checkpoint_manifest.json", manifest)
    return manifest


def verify_score_coverage(score_path, protocol_path, expected_count):
    """점수 파일 검사: 개수, ID 중복, 유한값, 라벨이 프로토콜과 맞는지."""
    expected = {}
    labels = {"bonafide": 0, "spoof": 1, "hidden": -1}
    with Path(protocol_path).open() as fh:
        for line in fh:
            fields = line.split()
            if len(fields) != 10 or fields[1] in expected:
                raise ValueError("Protocol must contain 10 columns and unique file IDs")
            expected[fields[1]] = labels.get(fields[8].lower(), -1)
    if len(expected) != expected_count:
        raise ValueError(f"Protocol count {len(expected)} != {expected_count}")
    seen = set()
    with Path(score_path).open() as fh:
        for line in fh:
            fields = line.split()
            if len(fields) != 3:
                raise ValueError("Expected file_id, score, label in each score row")
            fid, score, label = fields
            if fid in seen or fid not in expected:
                raise ValueError(f"Duplicate or unexpected score ID: {fid}")
            if not math.isfinite(float(score)) or int(label) != expected[fid]:
                raise ValueError(f"Invalid score or label: {fid}")
            seen.add(fid)
    if seen != expected.keys():
        raise ValueError(f"Incomplete scores: {len(seen)}/{len(expected)}")
    return {"rows": len(seen), "expected_rows": expected_count,
            "unique_ids": True, "finite_scores": True, "labels_match_protocol": True,
            "complete": True}


def plot_training(output, runtime=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .2})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    metrics = [("train", "loss", "Train loss", 1), ("dev", "loss", "Dev loss", 1),
               ("dev", "eer", "Dev EER (%)", 100), ("dev", "min_dcf", "Dev minDCF", 1),
               ("train", "bce", "Train BCE", 1), ("train", "worker", "Worker loss", 1)]
    found = False
    for combo in COMBOS:
        history = read_json(run_dir(output, combo, runtime) / "history.json", [])
        for ax, (section, key, title, scale) in zip(axes.flat, metrics):
            vals = [(h["epoch"], h.get(section, {}).get(key)) for h in history]
            vals = [(ep, val) for ep, val in vals if val is not None and math.isfinite(val)]
            if vals:
                found = True
                ax.plot([p[0] for p in vals], [p[1] * scale for p in vals], marker=".", label=combo.upper())
            ax.set(title=title, xlabel="Epoch")
    for ax in axes.flat:
        if ax.lines:
            ax.legend(fontsize=8)
    if not found:
        axes[0, 0].text(.5, .5, "Training history not available yet", ha="center", transform=axes[0, 0].transAxes)
    fig.suptitle("Epoch history (dev subset 5,000)")
    path = figures / "training_curves.png"
    save_figure(fig, path)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    for combo in COMBOS:
        steps = [r for r in records(run_dir(output, combo, runtime) / "train_log.jsonl") if r.get("kind") == "step"]
        for ax, key, title in zip(axes.flat, ["train/loss", "train/lr", "train/grad_norm", "sys/gpu_mem_mb"],
                                 ["Step train loss", "Learning rate", "Gradient norm", "Peak GPU memory (MB)"]):
            vals = [(r.get("step"), r.get(key)) for r in steps]
            vals = [(x, y) for x, y in vals if x is not None and y is not None and math.isfinite(y)]
            if vals:
                ax.plot([v[0] for v in vals], [v[1] for v in vals], label=combo.upper(), alpha=.8)
            ax.set(title=title, xlabel="Recorded global optimizer step")
            if ax.lines:
                ax.legend(fontsize=8)
    fig.suptitle("Step logs")
    step_path = figures / "step_curves.png"
    save_figure(fig, step_path)
    plt.close(fig)
    return [path, step_path]


def plot_comparison(output):
    import matplotlib.pyplot as plt
    import numpy as np
    rows = summary_rows(output)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(len(rows))
    for ax, metric, title in zip(axes, ["eer_pct", "min_dcf"], ["Full split EER (%)", "Full split minDCF"]):
        for shift, split in [(-.18, "dev"), (.18, "eval")]:
            vals = [r.get(f"{split}_{metric}") if r.get(f"full_{split}_done") else None for r in rows]
            valid = [i for i, v in enumerate(vals) if v is not None and math.isfinite(v)]
            if valid:
                ax.bar(x[valid] + shift, [vals[i] for i in valid], width=.35, label=split)
        ax.set_xticks(x, [r["combo"] for r in rows])
        ax.set_title(title)
        if ax.patches:
            ax.legend()
    fig.suptitle("Full dev / eval (best checkpoint by dev minDCF)")
    path = Path(output) / "figures/results_comparison.png"
    save_figure(fig, path)
    plt.close(fig)
    return path


def plot_detection(output, split="dev"):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from sklearn.metrics import roc_curve
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    available = False
    for combo in COMBOS:
        p = Path(output) / "scores" / f"{split}_{tag(combo)}.tsv"
        summary = read_json(p.with_name(p.stem + "_summary.json"), {})
        if not p.exists() or summary.get("metric_mode") != "official":
            continue
        df = pd.read_csv(p, sep="\t", names=["file_id", "score", "label"])
        df = df[df.label.isin([0, 1])]
        if df.label.nunique() != 2:
            continue
        available = True
        fpr, tpr, _ = roc_curve((df.label == 0).astype(int), df.score)
        fnr = 1 - tpr
        sample = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 2000)).astype(int))
        axes[0].plot(fpr[sample], tpr[sample], label=combo.upper())
        axes[1].plot(norm.ppf(np.clip(fpr[sample], 1e-5, 1-1e-5)),
                     norm.ppf(np.clip(fnr[sample], 1e-5, 1-1e-5)), label=combo.upper())
        if combo == "a1c1":
            for label, name in [(0, "bonafide"), (1, "spoof")]:
                axes[2].hist(df.loc[df.label == label, "score"], bins=60, density=True, alpha=.5, label=name)
    axes[0].set(title="ROC", xlabel="Spoof accepted (FPR)", ylabel="Bonafide accepted (TPR)")
    if available:
        axes[0].plot([0, 1], [0, 1], "--", color="grey", alpha=.5, label="Chance")
    ticks = np.array([.001, .01, .05, .1, .2, .4, .6, .8, .9, .99])
    for setter in (axes[1].set_xticks, axes[1].set_yticks):
        setter(norm.ppf(ticks), [f"{p*100:g}" for p in ticks])
    axes[1].set(title="DET", xlabel="Spoof accepted (%)", ylabel="Bonafide rejected (%)")
    axes[2].set(title="A1C1 score distribution", xlabel="Score (higher = bonafide)", ylabel="Density")
    for ax in axes:
        if ax.lines or ax.patches:
            ax.legend(fontsize=8)
    if not available:
        axes[0].text(.1, .5, f"Labeled {split} scores not available yet", transform=axes[0].transAxes)
    fig.suptitle(f"{split.upper()} detection")
    p = Path(output) / "figures" / f"{split}_detection.png"
    save_figure(fig, p)
    plt.close(fig)
    return p


def render(output, runtime=None, include_scores=False, verify_checkpoints=False):
    output = Path(output)
    (output / "reports").mkdir(parents=True, exist_ok=True)
    if verify_checkpoints:
        checkpoint_manifest(output, verify=True)
    rows = summary_rows(output, runtime)
    with (output / "reports/results_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths = plot_training(output, runtime) + [plot_comparison(output)]
    if include_scores:
        paths += [plot_detection(output, split) for split in ("dev", "eval")]
    # NaN은 json에서 null로
    clean = [{k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in r.items()} for r in rows]
    write_json(output / "reports/results_summary.json", clean)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime")
    parser.add_argument("--include-scores", action="store_true")
    parser.add_argument("--verify-checkpoints", action="store_true")
    args = parser.parse_args()
    for p in render(args.output, args.runtime, args.include_scores, args.verify_checkpoints):
        print(p)
