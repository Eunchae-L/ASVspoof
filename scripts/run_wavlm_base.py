"""WavLM Base, MUSAN 없이 Track 1 2x2 조합 학습 / dev / eval 실행 (Colab GPU).

데이터 다운로드, protocol 검증, 50% subset, smoke, 학습, 전체 dev/eval 순서.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

MODEL = "microsoft/wavlm-base"
COMBOS = {"a1c1": ("bilstm", "lcnn_big"), "a1c2": ("bilstm", "senet12"),
          "a2c1": ("se_res2net", "lcnn_big"), "a2c2": ("se_res2net", "senet12")}
COLS = ["speaker_id", "file_id", "gender", "codec", "codec_q", "codec_seed",
        "attack_tag", "attack_label", "key", "tmp"]
EXPECTED = {"ASVspoof5.train.tsv": (182357, "cbfa08e2fab1423e828f102bc97a22f0"),
            "ASVspoof5.dev.track_1.tsv": (140950, "e949dee1091a49b04943c548b5cf9860"),
            "ASVspoof5.eval.track_1.tsv": (680774, "50c0591d45a282cb787a5a4614394f67")}


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def copy_protocol_snapshot(source, destination):
    """tmp에 복사한 뒤 교체."""
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def restore_checkpoint(source, destination):
    """Drive 백업에서 체크포인트 복원 (wandb 폴더 제외)."""
    shutil.copytree(source, destination, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("wandb"),
                    ignore_dangling_symlinks=True)


def training_complete(state, epochs, patience):
    """epoch 중간 체크포인트면 미완료로 본다."""
    return bool(state.get("epoch_completed", True)) and (
        int(state.get("epoch", 0)) >= epochs
        or int(state.get("bad_epochs", 0)) >= patience
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--experiment", default="wavlm_base_no_musan_20261106")
    ap.add_argument("--combos", nargs="+", choices=list(COMBOS), default=list(COMBOS))
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--evaluation-only", action="store_true",
                    help="Evaluate dev-selected best models after all requested full dev evaluations finish")
    args = ap.parse_args()
    project = Path(args.project).resolve()
    output = project / "experiments" / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    for name in ("configs", "logs", "checkpoints", "scores", "protocols", "source", "reports"):
        (output / name).mkdir(exist_ok=True)
    state = {"experiment": args.experiment, "backbone": MODEL, "musan": False,
             "status": "starting", "started": time.time()}

    def update(stage, **extra):
        state.update(status=stage, updated=time.time(), **extra)
        save_json(output / "status.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    def run(cmd, name, cwd=None, backup=None):
        log = output / "logs" / f"{name}.log"
        update(name, command=[str(x) for x in cmd], log=str(log))
        with log.open("a", buffering=1) as fh:
            p = subprocess.Popen([str(x) for x in cmd], cwd=cwd, env=os.environ.copy(),
                                 stdout=fh, stderr=subprocess.STDOUT)
            last_backup = 0
            while p.poll() is None:
                if backup and time.time() - last_backup > 120:
                    subprocess.run(["rsync", "-a", "--exclude=epoch_*.pt",
                                    str(backup) + "/", str(output / "checkpoints" / backup.name) + "/"],
                                   check=True, capture_output=True)
                    last_backup = time.time()
                time.sleep(5)
            if backup:
                subprocess.run(["rsync", "-a", "--exclude=epoch_*.pt", str(backup) + "/",
                                str(output / "checkpoints" / backup.name) + "/"], check=True)
            if p.returncode:
                raise RuntimeError(f"{name} exited {p.returncode}; see {log}\n{log.read_text()[-2500:]}")

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU runtime required; CPU training is intentionally disabled")
        gpu = torch.cuda.get_device_properties(0)
        update("environment", gpu=gpu.name, gpu_gb=round(gpu.total_memory / 2**30, 1))
        os.environ.update(HF_HOME=str(project / ".hf_cache"), TOKENIZERS_PARALLELISM="false",
                          PYTHONUNBUFFERED="1", WANDB_MODE="offline", OMP_NUM_THREADS="4")
        # 실행에 쓴 소스 사본 저장
        source = output / "source"
        for folder in ("data", "models", "utils", "configs"):
            for p in (project / folder).glob("*.py" if folder != "configs" else "baseline.yaml"):
                dest = source / folder / p.name
                dest.parent.mkdir(exist_ok=True, parents=True)
                if not dest.exists():
                    shutil.copy2(p, dest)
        for name in ("train.py", "evaluate.py", "requirements.txt"):
            if not (source / name).exists():
                shutil.copy2(project / name, source / name)
        runtime = Path("/content") / args.experiment
        runtime.mkdir(exist_ok=True)
        shutil.copytree(source, runtime, dirs_exist_ok=True)
        sys.path.insert(0, str(runtime))
        run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.46,<5",
             "huggingface_hub>=0.26,<1", "librosa>=0.10", "soundfile", "pyyaml",
             "pandas>=2,<3", "scikit-learn", "matplotlib", "wandb>=0.16"], "dependencies")
        import pandas as pd
        import yaml
        from experiment_dashboard import verify_score_coverage
        from huggingface_hub import HfApi, hf_hub_download
        from utils.compliance import check_config
        data = Path("/content/asvspoof5_data")
        raw = Path("/content/asvspoof5_raw")
        data.mkdir(exist_ok=True)
        raw.mkdir(exist_ok=True)
        proto_dir = data / "protocols"
        proto_dir.mkdir(exist_ok=True)
        repo = "jungjee/asvspoof5"
        files = HfApi().list_repo_files(repo, repo_type="dataset", token=False)

        def download(name):
            update("download", file=name)
            return Path(hf_hub_download(repo, name, repo_type="dataset", local_dir=raw, token=False))

        if not all((proto_dir / n).exists() for n in EXPECTED):
            archive = download(next(n for n in files if n.startswith("ASVspoof5_protocols.tar")))
            with tarfile.open(archive) as tf:
                for member in tf:
                    name = Path(member.name).name
                    if member.isfile() and name in EXPECTED:
                        with tf.extractfile(member) as src, (proto_dir / name).open("wb") as dst:
                            shutil.copyfileobj(src, dst)
            archive.unlink()
        frames = {}
        for name, (count, md5) in EXPECTED.items():
            p = proto_dir / name
            if hashlib.md5(p.read_bytes()).hexdigest() != md5:
                raise RuntimeError(f"Official protocol checksum mismatch: {name}")
            df = pd.read_csv(p, sep=r"\s+", header=None, names=COLS, keep_default_na=False)
            if len(df) != count or df.shape[1] != 10 or df.file_id.duplicated().any():
                raise RuntimeError(f"Invalid protocol: {name}")
            frames[name] = df
            copy_protocol_snapshot(p, output / "protocols" / name)
        train = frames["ASVspoof5.train.tsv"]
        # (key, attack_tag)별 50% 층화 샘플링
        indices = []
        for _, group in train.groupby(["key", "attack_tag"], dropna=False, sort=True):
            indices.extend(group.sample(frac=0.5, random_state=42).index.tolist())
        subset = train.loc[sorted(indices)]
        subset_name = "ASVspoof5.train.subset50.tsv"
        subset.to_csv(proto_dir / subset_name, sep=" ", header=False, index=False)
        copy_protocol_snapshot(proto_dir / subset_name, output / "protocols" / subset_name)
        for a, b in [("train", "dev.track_1"), ("train", "eval.track_1"), ("dev.track_1", "eval.track_1")]:
            da, db = frames[f"ASVspoof5.{a}.tsv"], frames[f"ASVspoof5.{b}.tsv"]
            if set(da.file_id) & set(db.file_id) or set(da.speaker_id) & set(db.speaker_id):
                raise RuntimeError(f"Split overlap: {a} / {b}")
        save_json(output / "data_audit.json", {
            "sampling": "Original key x attack_tag, pandas sample(frac=0.5, random_state=42)",
            "train_subset_rows": len(subset), "train_subset_sha256": hashlib.sha256((proto_dir/subset_name).read_bytes()).hexdigest(),
            "train_attack_counts": dict(Counter(subset.attack_label)),
            "track2_used": False, "split_file_and_speaker_overlap": 0,
        })

        def ensure_audio(prefix, wanted):
            audio = data / prefix
            audio.mkdir(exist_ok=True)
            shards = sorted(n for n in files if n.startswith(prefix + "_") and n.endswith(".tar"))
            # eval tar prefix는 flac_E, 폴더는 flac_E_eval
            wanted = set(wanted)
            for name in shards:
                marker = data / ("." + name + ".done")
                if marker.exists():
                    continue
                archive = download(name)
                update("extract", file=name)
                with tarfile.open(archive) as tf:
                    for member in tf:
                        base = Path(member.name).name
                        if member.isfile() and base.endswith(".flac") and base[:-5] in wanted:
                            dest = audio / base
                            if dest.exists() and dest.stat().st_size == member.size:
                                continue
                            with tf.extractfile(member) as src, dest.open("wb") as dst:
                                shutil.copyfileobj(src, dst)
                marker.touch()
                archive.unlink()
            missing = [fid for fid in wanted if not (audio / (fid + ".flac")).is_file()]
            if missing:
                raise RuntimeError(f"Missing {len(missing)} {prefix} audio files, examples={missing[:3]}")
            return audio

        if args.evaluation_only:
            prior = json.loads((output / "results.json").read_text()) if (output / "results.json").exists() else {}
            for combo in args.combos:
                tag = f"{combo}_wavlmB_nomusan_subset50_epoch20"
                if (prior.get(combo, {}).get("full_dev") != "done"
                        or not (output / "checkpoints" / tag / "best.pt").is_file()):
                    raise RuntimeError(f"Full dev model selection is unfinished: {combo}; eval is not started")
        else:
            ensure_audio("flac_T", subset.file_id)
            ensure_audio("flac_D", frames["ASVspoof5.dev.track_1.tsv"].file_id)
        # Drive에서 RIR 파일 목록을 읽으면 느려서 런타임 디스크에 풀어서 사용
        resource_root = Path("/content/asvspoof5_resources")
        resource_root.mkdir(exist_ok=True)
        rir = resource_root / "RIRS_NOISES" / "simulated_rirs"
        if not args.evaluation_only and not rir.exists():
            rir_zip = Path("/content/rirs_wavlm_base.zip")
            saved_zip = project / "resources/archives/rirs_noises.zip"
            if saved_zip.exists():
                shutil.copy2(saved_zip, rir_zip)
            else:
                run(["wget", "-q", "-O", str(rir_zip),
                     "https://www.openslr.org/resources/28/rirs_noises.zip"], "rir_download")
                saved_zip.parent.mkdir(parents=True, exist_ok=True)
                staged = saved_zip.with_suffix(".zip.tmp")
                shutil.copy2(rir_zip, staged)
                staged.replace(saved_zip)
            run(["unzip", "-q", "-o", str(rir_zip),
                 "RIRS_NOISES/simulated_rirs/*", "-d", str(resource_root)], "rir_extract")
        if not args.evaluation_only and (not rir.exists() or not shutil.which("ffmpeg")):
            raise RuntimeError("Original RIR/codec augmentation resources are unavailable")
        with (runtime / "configs/baseline.yaml").open() as f:
            base_cfg = yaml.safe_load(f)
        # 이전 a1c1 설정이 있으면 그 하이퍼파라미터 사용
        old_cfg = project / "configs/colab_a1c1_v2_subset50.yaml"
        if old_cfg.exists():
            base_cfg = yaml.safe_load(old_cfg.read_text())
            shutil.copy2(old_cfg, output / "configs/original_a1c1.yaml")
        base_cfg["model"]["ssl"].update(name_or_path=MODEL, feature_dim=768, normalize_input=False)
        base_cfg["augment"].update(musan_p=0.0, musan_root=None, rir_p=0.2,
                                  rir_root=str(rir), codec_p=0.3,
                                  codec_choices=["mp3", "aac", "amr-nb", "narrowband-8k"])
        base_cfg["data"].update(root=str(data), protocol_train="protocols/" + subset_name,
                               protocol_dev="protocols/ASVspoof5.dev.track_1.tsv",
                               protocol_eval="protocols/ASVspoof5.eval.track_1.tsv",
                               audio_train="flac_T", audio_dev="flac_D", audio_eval="flac_E_eval",
                               num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
        batch = int(base_cfg["train"]["batch_size"])
        base_cfg["train"].update(epochs=20, batch_size=batch, save_dir=str(runtime / "checkpoints"), resume=None)
        base_cfg.setdefault("tracking", {}).update(mode="offline", project="asvspoof5-track1")
        check_config(base_cfg)
        configs = {}
        for combo in args.combos:
            cfg = json.loads(json.dumps(base_cfg))
            cfg["model"]["adapter"]["type"], cfg["model"]["classifier"]["type"] = COMBOS[combo]
            path = output / "configs" / f"{combo}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            configs[combo] = path
        update("prepared", train_rows=len(subset), batch_size=batch, combos=args.combos)
        if args.prepare_only:
            return
        smoke = runtime / "checkpoints" / "smoke_wavlm_base"
        smoke.mkdir(parents=True, exist_ok=True)
        smoke_backup = output / "checkpoints" / smoke.name / "last.pt"
        smoke_ok = False
        if not args.evaluation_only and smoke_backup.exists():
            saved = torch.load(smoke_backup, map_location="cpu", weights_only=False)
            check_config(base_cfg, extra_cfgs=[saved.get("config")])
            smoke_ok = saved.get("epoch_completed", True) and saved.get("epoch", 0) >= 2
            del saved
        if not args.evaluation_only and not smoke_ok:
            run([sys.executable, "-u", "train.py", "--config", configs[args.combos[0]],
                 "--tag", smoke.name, "--smoke", "200", "--epochs", "2", "--dev-subset", "256",
                 "--dev-batch-size", "64", "--no-wandb"], "smoke", runtime, smoke)
        result_path = output / "results.json"
        results = json.loads(result_path.read_text()) if result_path.exists() else {}
        for combo in ([] if args.evaluation_only else args.combos):
            tag = f"{combo}_wavlmB_nomusan_subset50_epoch20"
            summary = output / "scores" / f"dev_{tag}_summary.json"
            if (results.get(combo, {}).get("full_dev") == "done"
                    and summary.exists()
                    and json.loads(summary.read_text()).get("n_with_labels") == 140950
                    and (output / "scores" / f"dev_{tag}.tsv").exists()
                    and (output / "checkpoints" / tag / "best.pt").exists()):
                update("skip_completed_dev", combo=combo)
                continue
            ckpt = runtime / "checkpoints" / tag
            ckpt.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-u", "train.py", "--config", configs[combo], "--tag", tag,
                   "--epochs", "20", "--dev-subset", "5000", "--dev-batch-size", "64",
                   "--save-all-epochs", "--save-every", "150", "--wandb",
                   "--wandb-project", "asvspoof5-track1", "--wandb-run-name", tag]
            backup_last = output / "checkpoints" / tag / "last.pt"
            if not (ckpt / "last.pt").exists() and backup_last.exists():
                restore_checkpoint(backup_last.parent, ckpt)
            train_done = False
            if (ckpt / "last.pt").exists():
                saved = torch.load(ckpt / "last.pt", map_location="cpu", weights_only=False)
                check_config(base_cfg, extra_cfgs=[saved.get("config")])
                train_done = training_complete(saved, base_cfg["train"]["epochs"],
                                               base_cfg["train"]["early_stopping_patience"])
                del saved
                cmd += ["--resume", ckpt / "last.pt"]
            if not train_done:
                run(cmd, f"train_{combo}", runtime, ckpt)
            run([sys.executable, "-u", "evaluate.py", "--config", configs[combo], "--ckpt", ckpt / "best.pt",
                 "--split", "dev", "--metric", "official", "--batch-size", "64",
                 "--out", output / "scores" / f"dev_{tag}.tsv"], f"dev_{combo}", runtime)
            write_coverage = verify_score_coverage(output / "scores" / f"dev_{tag}.tsv",
                                                   proto_dir / "ASVspoof5.dev.track_1.tsv", 140950)
            save_json(output / "reports" / f"coverage_dev_{combo}.json", write_coverage)
            results[combo] = {"checkpoint": str(output / "checkpoints" / tag / "best.pt"), "full_dev": "done"}
            save_json(output / "results.json", results)
        # eval은 dev로 best를 고른 뒤에만 실행
        eval_ids = frames["ASVspoof5.eval.track_1.tsv"].file_id
        eval_dir = ensure_audio("flac_E", eval_ids)
        target = data / "flac_E_eval"
        if not target.exists():
            target.symlink_to(eval_dir, target_is_directory=True)
        for combo in args.combos:
            tag = f"{combo}_wavlmB_nomusan_subset50_epoch20"
            summary = output / "scores" / f"eval_{tag}_summary.json"
            if (results.get(combo, {}).get("full_eval") == "done"
                    and summary.exists()
                    and json.loads(summary.read_text()).get("n_with_labels", 0)
                    + json.loads(summary.read_text()).get("n_hidden_labels", 0) == 680774
                    and (output / "scores" / f"eval_{tag}.tsv").exists()):
                try:
                    coverage = verify_score_coverage(output / "scores" / f"eval_{tag}.tsv",
                                                     proto_dir / "ASVspoof5.eval.track_1.tsv", 680774)
                except ValueError:
                    print(f"[coverage] {combo}: incomplete eval scores will be regenerated", flush=True)
                else:
                    save_json(output / "reports" / f"coverage_eval_{combo}.json", coverage)
                    update("skip_completed_eval", combo=combo)
                    continue
            run([sys.executable, "-u", "evaluate.py", "--config", configs[combo], "--ckpt",
                 output / "checkpoints" / tag / "best.pt", "--split", "eval", "--metric", "official",
                 "--batch-size", "64", "--out", output / "scores" / f"eval_{tag}.tsv"], f"eval_{combo}", runtime)
            coverage = verify_score_coverage(output / "scores" / f"eval_{tag}.tsv",
                                             proto_dir / "ASVspoof5.eval.track_1.tsv", 680774)
            save_json(output / "reports" / f"coverage_eval_{combo}.json", coverage)
            results[combo]["full_eval"] = "done"
            save_json(output / "results.json", results)
        update("complete", results=results)
    except BaseException as exc:
        update("failed", error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
