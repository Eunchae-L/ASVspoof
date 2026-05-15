"""ASVspoof 5 protocol parser.

확장자는 .tsv지만 공백 구분, 10컬럼 (ASVSPOOF5_COLUMNS). label = 1 if spoof else 0, hidden은 -1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


ASVSPOOF5_COLUMNS = [
    "speaker_id",
    "file_id",
    "speaker_gender",
    "codec",
    "codec_q",
    "codec_seed",
    "attack_tag",
    "attack_label",
    "key",
    "tmp",
]


@dataclass
class ProtocolEntry:
    file_id: str                 # FLAC_FILE_NAME (no extension)
    label: int                   # 1 = spoof / fake, 0 = bonafide / real
    key: str                     # "bonafide" or "spoof"
    speaker_id: Optional[str] = None
    speaker_gender: Optional[str] = None
    codec: Optional[str] = None
    attack_tag: Optional[str] = None
    attack_label: Optional[str] = None
    extra: dict = field(default_factory=dict)


def parse_asvspoof5_protocol(path: str) -> List[ProtocolEntry]:
    """Parse ASVspoof 5 Track 1 TSV protocol (space-separated, no header)."""
    entries: List[ProtocolEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            row = dict(zip(ASVSPOOF5_COLUMNS, parts))
            key = row.get("key", "").lower()
            if key == "spoof":
                label = 1
            elif key in ("bonafide", "bona-fide", "real"):
                label = 0
            else:
                # eval hidden label은 "-"
                label = -1
            entries.append(
                ProtocolEntry(
                    file_id=row["file_id"],
                    label=label,
                    key=key,
                    speaker_id=row.get("speaker_id"),
                    speaker_gender=row.get("speaker_gender"),
                    codec=row.get("codec"),
                    attack_tag=row.get("attack_tag"),
                    attack_label=row.get("attack_label"),
                )
            )
    return entries


def parse_for_protocol(path: str) -> List[ProtocolEntry]:
    raise NotImplementedError("FoR support is v2; see plan §v2.")


def parse_libriseVoc_protocol(path: str) -> List[ProtocolEntry]:
    raise NotImplementedError("LibriSeVoc support is v2; see plan §v2.")
