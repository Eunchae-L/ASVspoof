"""ckpt/ 아래 run들의 best epoch(dev minDCF, 없으면 EER)을 모아 표 + runs_summary.csv.

    python scripts/collect_runs.py [--ckpt-root ckpt] [--out runs_summary.csv] [--sort eer]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def _best_epoch(history: List[Dict]) -> Optional[Dict]:
    if not history:
        return None
    have_mdcf = [h for h in history if h.get("dev", {}).get("min_dcf") is not None]
    key = "min_dcf" if have_mdcf else "eer"
    pool = have_mdcf or history
    return min(pool, key=lambda h: h["dev"].get(key, float("inf")))


def _load_run(run_dir: Path) -> Optional[Dict]:
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        return None
    try:
        with open(hist_path) as f:
            history = json.load(f)
    except Exception:
        return None
    best = _best_epoch(history)
    if best is None:
        return None

    meta = {}
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    git = meta.get("git", {}) or {}
    arch = meta.get("arch", {}) or {}
    dev = best.get("dev", {})
    return {
        "tag": run_dir.name,
        "best_epoch": best.get("epoch"),
        "dev_minDCF": dev.get("min_dcf"),
        "dev_EER_%": (dev.get("eer") * 100) if dev.get("eer") is not None else None,
        "dev_Cllr": dev.get("cllr"),
        "n_epochs": len(history),
        "adapter": arch.get("adapter"),
        "classifier": arch.get("classifier"),
        "ssl": arch.get("ssl"),
        "commit": git.get("short_commit"),
        "dirty": git.get("dirty"),
        "date": meta.get("timestamp"),
    }


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


COLUMNS = ["tag", "best_epoch", "dev_minDCF", "dev_EER_%", "dev_Cllr",
           "n_epochs", "adapter", "classifier", "commit", "dirty", "date"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="ckpt",
                    help="Root dir holding run subdirectories (searched recursively).")
    ap.add_argument("--out", default="runs_summary.csv")
    ap.add_argument("--sort", choices=["minDCF", "eer"], default="minDCF")
    args = ap.parse_args()

    root = Path(args.ckpt_root)
    if not root.exists():
        raise SystemExit(f"[collect] ckpt root not found: {root}")

    runs: List[Dict] = []
    for hist_path in sorted(root.rglob("history.json")):
        row = _load_run(hist_path.parent)
        if row is not None:
            runs.append(row)

    if not runs:
        raise SystemExit(f"[collect] no runs with history.json found under {root}")

    sort_key = "dev_minDCF" if args.sort == "minDCF" else "dev_EER_%"
    runs.sort(key=lambda r: (r.get(sort_key) is None, r.get(sort_key) or float("inf")))

    print(f"\n# Runs leaderboard ({len(runs)} runs, sorted by {args.sort} ↑)\n")
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "|".join(["---"] * len(COLUMNS)) + "|")
    for r in runs:
        print("| " + " | ".join(_fmt(r.get(c)) for c in COLUMNS) + " |")

    out = Path(args.out)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in runs:
            writer.writerow({c: r.get(c) for c in COLUMNS})
    print(f"\n[collect] wrote {out}  ({len(runs)} runs)")


if __name__ == "__main__":
    main()
