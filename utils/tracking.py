"""W&B logger. wandb가 없거나 로그인 실패해도 학습은 계속되고 train_log.jsonl에는 항상 기록.

우선순위: --wandb/--no-wandb > WANDB_MODE=disabled > cfg["tracking"]["backend"]
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional


class ExperimentTracker:
    def __init__(
        self,
        run_name: str,
        save_dir: Path,
        *,
        backend: str = "wandb",
        project: str = "asvspoof5-track1",
        entity: Optional[str] = None,
        mode: str = "online",
        config: Optional[dict] = None,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.save_dir / "train_log.jsonl"
        self._run = None
        self._wandb = None
        self.url = None

        # append: resume해도 이전 로그 유지
        self._jsonl_fh = open(self.jsonl_path, "a")

        want_wandb = backend == "wandb"
        env_mode = os.environ.get("WANDB_MODE")  # offline / disabled / online
        if env_mode == "disabled":
            want_wandb = False

        if want_wandb:
            try:
                import wandb

                self._wandb = wandb
                self._run = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", project),
                    entity=entity or os.environ.get("WANDB_ENTITY"),
                    name=run_name,
                    dir=str(self.save_dir),
                    config=config or {},
                    mode=env_mode or mode,
                    resume="allow",
                )
                try:
                    self.url = self._run.get_url()
                except Exception:
                    self.url = None
                print(f"[tracking] W&B enabled - project={project} run={run_name}"
                      + (f"  url={self.url}" if self.url else ""))
            except Exception as e:
                self._wandb = None
                self._run = None
                print(f"[tracking] W&B unavailable ({type(e).__name__}: {e}); "
                      f"falling back to local JSONL only → {self.jsonl_path}")
        else:
            print(f"[tracking] W&B disabled; local JSONL only → {self.jsonl_path}")

    @classmethod
    def from_config(cls, cfg: dict, run_name: str, save_dir: Path,
                    *, enabled: Optional[bool] = None,
                    project: Optional[str] = None) -> "ExperimentTracker":
        """enabled: True/False면 강제, None이면 cfg["tracking"]["backend"]."""
        tcfg = (cfg or {}).get("tracking", {}) or {}
        backend = tcfg.get("backend", "wandb")
        if enabled is False:
            backend = "none"
        elif enabled is True:
            backend = "wandb"
        return cls(
            run_name=run_name,
            save_dir=save_dir,
            backend=backend,
            project=project or tcfg.get("project", "asvspoof5-track1"),
            entity=tcfg.get("entity"),
            mode=tcfg.get("mode", "online"),
            config=cfg,
        )

    def _write_jsonl(self, record: Dict) -> None:
        try:
            self._jsonl_fh.write(json.dumps(record) + "\n")
            self._jsonl_fh.flush()
        except Exception:
            pass

    def log_step(self, metrics: Dict[str, float], step: int) -> None:
        rec = {"kind": "step", "step": int(step), **metrics}
        self._write_jsonl(rec)
        if self._run is not None:
            self._run.log(metrics, step=int(step))

    def log_epoch(self, metrics: Dict[str, float], epoch: int) -> None:
        rec = {"kind": "epoch", "epoch": int(epoch), **metrics}
        self._write_jsonl(rec)
        if self._run is not None:
            self._run.log({**metrics, "epoch": int(epoch)})

    def log_summary(self, metrics: Dict[str, float]) -> None:
        rec = {"kind": "summary", **metrics}
        self._write_jsonl(rec)
        if self._run is not None:
            for k, v in metrics.items():
                self._run.summary[k] = v

    def finish(self) -> None:
        try:
            self._jsonl_fh.close()
        except Exception:
            pass
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:
                pass
