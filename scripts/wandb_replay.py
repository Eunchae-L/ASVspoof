"""train_log.jsonl (없으면 history.json)으로 W&B run을 다시 올림. 재학습 불필요.

    python scripts/wandb_replay.py --run-dir ckpt/a1c1 [--project asvspoof5-track1] [--run-name a1c1_replay]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config_used.yaml"
    if cfg_path.exists():
        try:
            import yaml
            with open(cfg_path) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return {"run_meta": json.load(f)}
        except Exception:
            pass
    return {}


def _replay_jsonl(run, jsonl_path: Path) -> int:
    n = 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            kind = rec.pop("kind", None)
            if kind == "summary":
                for k, v in rec.items():
                    run.summary[k] = v
                continue
            step = rec.pop("step", None)
            if step is not None:
                run.log(rec, step=int(step))
            else:
                run.log(rec)
            n += 1
    return n


def _replay_history(run, history_path: Path) -> int:
    with open(history_path) as f:
        history = json.load(f)
    n = 0
    for h in history:
        metrics = {"epoch": h.get("epoch")}
        for k, v in (h.get("train") or {}).items():
            metrics[f"train/{k}"] = v
        for k, v in (h.get("dev") or {}).items():
            metrics[f"dev/{k}"] = v
        run.log(metrics)
        n += 1
    mdcfs = [(h["dev"].get("min_dcf"), h["epoch"]) for h in history
             if h.get("dev", {}).get("min_dcf") is not None]
    if mdcfs:
        best_mdcf, best_ep = min(mdcfs, key=lambda t: t[0])
        run.summary["best/min_dcf"] = best_mdcf
        run.summary["best/epoch"] = best_ep
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="Checkpoint dir containing train_log.jsonl / history.json / run_meta.json")
    ap.add_argument("--project", default="asvspoof5-track1")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--run-name", default=None,
                    help="Default: <run-dir name>_replay")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"[replay] run dir not found: {run_dir}")

    jsonl = run_dir / "train_log.jsonl"
    history = run_dir / "history.json"
    if not jsonl.exists() and not history.exists():
        raise SystemExit(f"[replay] neither train_log.jsonl nor history.json in {run_dir}")

    import wandb
    run_name = args.run_name or f"{run_dir.name}_replay"
    cfg = _load_config(run_dir)
    run = wandb.init(project=args.project, entity=args.entity, name=run_name,
                     config=cfg, resume="never")
    print(f"[replay] W&B run: {run.get_url()}")

    if jsonl.exists():
        n = _replay_jsonl(run, jsonl)
        print(f"[replay] replayed {n} records from {jsonl.name}")
    else:
        n = _replay_history(run, history)
        print(f"[replay] replayed {n} epochs from {history.name} (step curves 없음)")

    run.finish()
    print("[replay] done - 대시보드를 확인하세요.")


if __name__ == "__main__":
    main()
