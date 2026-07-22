"""EER, minDCF, actDCF, Cllr.

입력은 공식 방향: score 클수록 bona fide, label 1 = bona fide, 0 = spoof.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Returns (eer, threshold)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fnr[idx] + fpr[idx]) / 2.0)
    return eer, float(thr[idx])


def compute_min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
    p_spoof: float = 0.05,
) -> Tuple[float, float]:
    """minDCF (Wang et al., Interspeech 2024, Eq. 1-2). Returns (min_dcf, threshold).

    DCF = beta * P_miss + P_fa, beta = (C_miss/C_fa) * (1 - p_spoof) / p_spoof (~1.90).
    p_spoof는 spoof class prior (target prior 아님).
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    p_miss = 1.0 - tpr          # bona fide rejected
    p_fa = fpr                  # spoof accepted

    beta = (c_miss / c_fa) * ((1.0 - p_spoof) / p_spoof)
    dcf = beta * p_miss + p_fa
    idx = int(np.nanargmin(dcf))
    return float(dcf[idx]), float(thr[idx])


def compute_actdcf(
    scores: np.ndarray,
    labels: np.ndarray,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
    p_spoof: float = 0.05,
) -> Tuple[float, float]:
    """actDCF at tau_Bayes = -ln(beta) (Eq. 2, 4). Returns (act_dcf, tau_bayes).

    score가 calibrated LLR일 때만 의미 있음. raw logit이면 actDCF >> minDCF.
    """
    beta = (c_miss / c_fa) * ((1.0 - p_spoof) / p_spoof)
    tau_bayes = -float(np.log(beta))
    pos = scores[labels == 1]          # bona fide
    neg = scores[labels == 0]          # spoof
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), tau_bayes
    p_miss = float(np.mean(pos < tau_bayes))     # bona fide rejected
    p_fa = float(np.mean(neg >= tau_bayes))      # spoof accepted
    return float(beta * p_miss + p_fa), tau_bayes


def compute_cllr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Cllr. score를 positive class의 LLR로 가정 (minCllr/PAV는 미구현)."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    s_pos = scores[pos_mask].astype(np.float64)
    s_neg = scores[neg_mask].astype(np.float64)

    # 수치 안정적인 log2(1 + exp(x))
    def _softplus_log2(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(x))) / np.log(2.0) + np.maximum(x, 0.0) / np.log(2.0)

    cost_pos = float(_softplus_log2(-s_pos).mean())
    cost_neg = float(_softplus_log2(+s_neg).mean())
    return 0.5 * (cost_pos + cost_neg)


def report_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
    p_spoof: float = 0.05,
) -> dict:
    eer, eer_thr = compute_eer(scores, labels)
    mdcf, mdcf_thr = compute_min_dcf(scores, labels, c_miss, c_fa, p_spoof)
    adcf, adcf_thr = compute_actdcf(scores, labels, c_miss, c_fa, p_spoof)
    cllr = compute_cllr(scores, labels)
    return {
        "eer": eer,
        "eer_threshold": eer_thr,
        "min_dcf": mdcf,
        "min_dcf_threshold": mdcf_thr,
        "act_dcf": adcf,
        "act_dcf_threshold": adcf_thr,
        "cllr": cllr,
        "n_pos": int((labels == 1).sum()),       # bona fide
        "n_neg": int((labels == 0).sum()),       # spoof
        "mindcf_params": {"c_miss": c_miss, "c_fa": c_fa, "p_spoof": p_spoof},
    }
