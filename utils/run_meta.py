"""run_meta.json: git commit/dirty, 실행 명령, 라이브러리/GPU 버전 등. 실패해도 학습은 진행."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def _run_git(args) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _git_info() -> Dict:
    commit = _run_git(["rev-parse", "HEAD"])
    if commit is None:
        return {"commit": None, "dirty": None, "branch": None}
    status = _run_git(["status", "--porcelain"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "commit": commit,
        "short_commit": commit[:8],
        "dirty": bool(status),
        "branch": branch,
    }


def _env_info() -> Dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": None,
        "torchaudio": None,
        "transformers": None,
        "cuda": None,
        "gpu": None,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    try:
        import torchaudio
        info["torchaudio"] = torchaudio.__version__
    except Exception:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info


def capture_run_meta(cfg: dict, args=None, *, timestamp: Optional[str] = None) -> Dict:
    meta = {
        "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "seed": (cfg or {}).get("seed"),
        "tag": getattr(args, "tag", None),
        "git": _git_info(),
        "env": _env_info(),
    }
    # collect_runs.py leaderboard용
    model = (cfg or {}).get("model", {})
    meta["arch"] = {
        "ssl": model.get("ssl", {}).get("name_or_path"),
        "adapter": model.get("adapter", {}).get("type"),
        "classifier": model.get("classifier", {}).get("type"),
        "lambda_workers": (cfg or {}).get("loss", {}).get("lambda_workers"),
    }
    return meta


def save_run_meta(cfg: dict, save_dir, args=None, *, timestamp: Optional[str] = None) -> Dict:
    meta = capture_run_meta(cfg, args, timestamp=timestamp)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    g = meta["git"]
    print(f"[run-meta] commit={g.get('short_commit')} "
          f"dirty={g.get('dirty')} gpu={meta['env'].get('gpu')} "
          f"→ {save_dir / 'run_meta.json'}")
    return meta
