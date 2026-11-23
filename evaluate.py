"""dev/eval split 평가. 출력 TSV: <file_id>\t<score>\t<label> (1=spoof, 0=bonafide, -1=hidden).

    python evaluate.py --config configs/baseline.yaml --ckpt ckpt/a1c1/best.pt --split dev --metric official [--sliding] [--out scores/dev.tsv]

기본 추론은 공식 baseline(AASIST/RawNet2)과 같이 앞 n 샘플 한 구간만 사용 (짧으면 반복).
--sliding은 비공식 TTA (겹치는 window logit 평균).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import build_splits
from models.model import SSLMultiTaskModel
from utils.metrics import report_metrics, compute_eer


def _load_audio(path: str, sr: int, peak_normalize: bool = True) -> np.ndarray:
    x, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if file_sr != sr:
        ratio = sr / file_sr
        t_old = np.arange(len(x))
        t_new = np.linspace(0, len(x) - 1, int(round(len(x) * ratio)))
        x = np.interp(t_new, t_old, x).astype(np.float32)
    if peak_normalize:
        peak = float(np.max(np.abs(x)) + 1e-9)
        x = (x / peak * 0.95).astype(np.float32)
    return x.astype(np.float32)


def _crop_or_pad_center(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[0] >= n:
        s = (x.shape[0] - n) // 2
        return x[s:s + n]
    pad = n - x.shape[0]
    if x.shape[0] >= 2:
        return np.pad(x, (pad // 2, pad - pad // 2), mode="reflect").astype(np.float32)
    return np.pad(x, (0, pad), mode="constant").astype(np.float32)


def _pad_official(x: np.ndarray, n: int) -> np.ndarray:
    # 공식 baseline의 pad(x, 64600)와 동일: 앞 n 샘플, 짧으면 tile.
    # ref: asvspoof-challenge/asvspoof5 Baseline-AASIST/data_utils.py
    if x.shape[0] >= n:
        return x[:n].astype(np.float32)
    num_repeats = int(n / x.shape[0]) + 1
    return np.tile(x, num_repeats)[:n].astype(np.float32)


def _sliding_windows(x: np.ndarray, n: int, hop: int) -> List[np.ndarray]:
    if x.shape[0] <= n:
        return [_crop_or_pad_center(x, n)]
    windows = []
    s = 0
    while s + n <= x.shape[0]:
        windows.append(x[s:s + n])
        s += hop
    if (x.shape[0] - n) % hop != 0:
        windows.append(x[-n:])
    return windows


def infer_one_logit(model, wav: np.ndarray, n: int, device, sliding: bool, hop: int) -> float:
    if sliding:
        windows = _sliding_windows(wav, n, hop)
    else:
        windows = [_crop_or_pad_center(wav, n)]
    batch = torch.from_numpy(np.stack(windows, axis=0)).to(device)
    with torch.no_grad():
        logit, _, _ = model(batch, apply_specaug=False)
    return float(logit.mean().item())


def run_inference_batched(model, entries, audio_dir, ext, sr, n, device,
                          sliding, hop, batch_size, peak_normalize):
    """여러 발화의 window를 한 batch로 묶어 추론. 발화 score = window logit 평균."""
    n_utt = len(entries)
    sum_lg = np.zeros(n_utt, dtype=np.float64)
    cnt = np.zeros(n_utt, dtype=np.int64)
    loaded = np.ones(n_utt, dtype=bool)

    buf_win: List[np.ndarray] = []
    buf_owner: List[int] = []
    use_amp = device.type == "cuda"

    def _flush():
        if not buf_win:
            return
        batch = torch.from_numpy(np.stack(buf_win, axis=0)).to(device, non_blocking=True)
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    lg, _, _ = model(batch, apply_specaug=False)
            else:
                lg, _, _ = model(batch, apply_specaug=False)
        lg = lg.float().detach().cpu().numpy().reshape(-1)
        for owner, val in zip(buf_owner, lg):
            sum_lg[owner] += float(val)
            cnt[owner] += 1
        buf_win.clear()
        buf_owner.clear()

    for i, entry in enumerate(tqdm(entries, desc=f"infer",
                                   unit="utt", dynamic_ncols=True,
                                   mininterval=1.0, smoothing=0.05)):
        path = os.path.join(audio_dir, f"{entry.file_id}{ext}")
        try:
            wav = _load_audio(path, sr, peak_normalize=peak_normalize)
        except Exception as e:
            print(f"[warn] failed to load {path}: {e}")
            loaded[i] = False
            continue
        windows = _sliding_windows(wav, n, hop) if sliding else [_pad_official(wav, n)]
        for w in windows:
            buf_win.append(w)
            buf_owner.append(i)
            if len(buf_win) >= batch_size:
                _flush()
    _flush()

    logits, labels, file_ids, keys = [], [], [], []
    for i, entry in enumerate(entries):
        if not loaded[i] or cnt[i] == 0:
            continue
        logits.append(sum_lg[i] / cnt[i])
        labels.append(entry.label)
        file_ids.append(entry.file_id)
        keys.append(entry.key)
    return logits, labels, file_ids, keys


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["dev", "eval"], default="dev")
    ap.add_argument("--metric", choices=["internal", "official"], default=None,
                    help="internal: p_fake (larger=fake). official: -fake_logit (larger=bonafide). "
                         "Default: cfg.eval.metric_mode.")
    ap.add_argument("--sliding", action="store_true",
                    help="NON-official test-time augmentation: tile the whole utterance into "
                         "overlapping n-sample windows and average their logits. "
                         "Default (omit) = official ASVspoof5 baseline method: single first-n "
                         "samples segment (repeat-pad if short), one score per utterance.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Utterances' windows are packed into GPU batches of this size. "
                         "Bigger = faster until VRAM-bound (A100: try 64-128). Default 32.")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only evaluate the first N utterances (quick sanity / iteration). "
                         "Full eval is 680K utts and takes hours; use this to smoke-test first.")
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    metric_mode = args.metric or cfg["eval"].get("metric_mode", "internal")
    assert metric_mode in ("internal", "official")

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"[device] {device}")
    print(f"[metric_mode] {metric_mode}")
    print(f"[sliding] {args.sliding}")

    splits = build_splits(cfg, splits=[args.split])
    if args.split not in splits:
        raise SystemExit(
            f"Split '{args.split}' missing - make sure flac_{args.split[0].upper()}_*.tar "
            f"is extracted and protocol file exists under data.root/protocols/."
        )
    split = splits[args.split]
    print(f"[data] split={args.split}  n={len(split.entries)}  audio_dir={split.audio_dir}")

    # 모델 구조는 ckpt에 저장된 학습 config 기준 (yaml이 학습 후 바뀌어도 load 가능하게)
    state = torch.load(args.ckpt, map_location=device)
    ckpt_cfg = state.get("config")
    from utils.compliance import check_config
    check_config(cfg, extra_cfgs=[ckpt_cfg])   # 학습 때 쓴 백본도 같이 확인
    build_cfg = cfg
    if ckpt_cfg is not None and "model" in ckpt_cfg:
        build_cfg = {**cfg, "model": ckpt_cfg["model"], "loss": ckpt_cfg.get("loss", cfg.get("loss", {}))}
        f_clf = cfg.get("model", {}).get("classifier", {}).get("type")
        c_clf = ckpt_cfg["model"].get("classifier", {}).get("type")
        f_ad = cfg.get("model", {}).get("adapter", {}).get("type")
        c_ad = ckpt_cfg["model"].get("adapter", {}).get("type")
        if (f_clf, f_ad) != (c_clf, c_ad):
            print(f"[arch] config 파일({f_ad}/{f_clf}) ≠ 체크포인트({c_ad}/{c_clf}) "
                  f"→ 체크포인트 구조로 모델을 만듭니다.")
        else:
            print(f"[arch] 체크포인트 구조 사용: adapter={c_ad} classifier={c_clf}")
    else:
        print("[arch] 체크포인트에 config 없음(구버전) → --config 파일 구조로 로드 시도")

    model = SSLMultiTaskModel(build_cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[ckpt] loaded {args.ckpt}  best_eer(train_dev)={state.get('best_eer')}")

    sr = cfg["data"]["sample_rate"]
    n = int(round(sr * cfg["data"]["clip_seconds"]))
    hop = int(round(sr * cfg["eval"].get("sliding_window_hop_seconds", 2.0)))
    peak_normalize = cfg["data"].get("peak_normalize", True)   # 학습과 동일해야 함
    print(f"[preproc] peak_normalize = {peak_normalize}")

    entries = split.entries
    if args.limit and args.limit > 0:
        entries = entries[: args.limit]
        print(f"[limit] evaluating first {len(entries)} / {len(split.entries)} utterances only")
    ext = cfg["data"].get("audio_ext", ".flac")
    print(f"[infer] batched (batch_size={args.batch_size}, sliding={args.sliding}) "
          f"over {len(entries):,} utterances…")

    logits, labels, file_ids, keys = run_inference_batched(
        model, entries, split.audio_dir, ext, sr, n, device,
        args.sliding, hop, args.batch_size, peak_normalize,
    )

    logits_np = np.array(logits, dtype=np.float64)
    labels_np = np.array(labels, dtype=np.int64)   # 1 = spoof, 0 = bona fide, -1 = hidden

    # 지표는 metric_mode와 무관하게 항상 공식 방향(bona fide = positive)으로 계산
    score_bona = -logits_np
    label_bona = (labels_np == 0).astype(np.int64)   # 1 = bona fide

    valid = labels_np != -1
    rpt = report_metrics(
        score_bona[valid],
        label_bona[valid],
        c_miss=cfg["eval"].get("mindcf_c_miss", 1.0),
        c_fa=cfg["eval"].get("mindcf_c_fa", 10.0),
        p_spoof=cfg["eval"].get("mindcf_p_target", 0.05),
    )

    # 출력 score 방향만 metric_mode를 따름
    if metric_mode == "internal":
        out_scores = 1.0 / (1.0 + np.exp(-logits_np))  # p_fake, larger = spoof
        score_name = "p_fake"
    else:
        out_scores = -logits_np                        # larger = bona fide
        score_name = "score_bonafide"

    n_eval = int(valid.sum())
    n_hidden = int((~valid).sum())
    print()
    print(f"[result] split={args.split}  metric_mode={metric_mode}  n_with_labels={n_eval}  n_hidden_labels={n_hidden}")
    if n_eval > 0:
        print(f"[result]   EER     = {rpt['eer']*100:.2f}%   threshold@EER = {rpt['eer_threshold']:.4f}")
        print(f"[result]   minDCF  = {rpt['min_dcf']:.4f}   threshold@minDCF = {rpt['min_dcf_threshold']:.4f}")
        print(f"[result]   actDCF  = {rpt['act_dcf']:.4f}   τ_Bayes = {rpt['act_dcf_threshold']:.4f}  (raw scores; calibrate on dev to lower)")
        print(f"[result]   Cllr    = {rpt['cllr']:.4f}")
        print(f"[result]   pos={rpt['n_pos']}  neg={rpt['n_neg']}  cost params={rpt['mindcf_params']}")
    else:
        print("[result] all labels hidden - only score file produced (e.g. eval submission run).")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            for fid, s, lab in zip(file_ids, out_scores.tolist(), labels):
                f.write(f"{fid}\t{s:.6f}\t{lab}\n")
        summary = {
            "split": args.split,
            "metric_mode": metric_mode,
            "score_name": score_name,
            "sliding": bool(args.sliding),
            "sliding_window_hop_seconds": (hop / sr if args.sliding else None),
            "n_with_labels": n_eval,
            "n_hidden_labels": n_hidden,
            "ckpt": args.ckpt,
        }
        if n_eval > 0:
            for k, v in rpt.items():
                if not isinstance(v, dict):
                    summary[k] = v
            summary["mindcf_params"] = rpt["mindcf_params"]
        with open(str(Path(args.out).with_suffix("")) + "_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[out] wrote scores → {args.out}")
        print(f"[out] wrote summary → {Path(args.out).with_suffix('')}_summary.json")


if __name__ == "__main__":
    main()
