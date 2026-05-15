"""HF jungjee/asvspoof5에서 protocols + flac_T/flac_D tar 다운로드 (--include-eval: flac_E 추가, ~80 GB).

    python scripts/download_asvspoof5.py --local-dir /content/raw [--include-eval] [--token $HF_TOKEN]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", required=True,
                    help="Where to put the tars (e.g. /content/drive/MyDrive/ASVspoof5/raw)")
    ap.add_argument("--repo-id", default="jungjee/asvspoof5")
    ap.add_argument("--include-eval", action="store_true",
                    help="Also download flac_E_*.tar (~80 GB).")
    ap.add_argument("--token", default=None,
                    help="Optional HF access token (or set $HF_TOKEN).")
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit("Install with `pip install huggingface_hub>=0.23`") from e

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    patterns = [
        "ASVspoof5_protocols.tar.gz",
        "ASVspoof5_protocols.tar",
        "LICENSE.txt",
        "README.txt",
        "flac_T_*.tar",
        "flac_D_*.tar",
    ]
    if args.include_eval:
        patterns.append("flac_E_*.tar")

    print(f"[download] repo_id={args.repo_id}")
    print(f"[download] local_dir={local_dir}")
    print(f"[download] patterns={patterns}")
    print("[download] this may take 10–60 minutes depending on bandwidth …\n")

    token = args.token or os.environ.get("HF_TOKEN")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=patterns,
        token=token,
    )

    print("\n[download] done. Contents:")
    for p in sorted(local_dir.glob("*")):
        size = p.stat().st_size / 1024 ** 3 if p.is_file() else 0
        if p.is_file():
            print(f"  {p.name:<40} {size:>8.2f} GB")
        else:
            print(f"  {p.name}/")


if __name__ == "__main__":
    main()
