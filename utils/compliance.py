"""ASVspoof 5 evaluation plan(Phase 2) open condition 규칙 확인.

§4.2: LibriLight, MLS(영어 포함), MUSAN speech에서 나온 데이터·사전학습 모델은 금지.
ASVspoof 5와 겹치는지 확인할 수 없는 자원도 금지. LibriSpeech 사전학습 모델은 허용.
§4.3: 잡음·RIR·codec 같은 비음성 외부 자원은 허용 (system description에 명시).

train / train_cached / evaluate / cache 시작 시 check_config(cfg) 호출.

SSL 백본 (HF model card 기준)
  허용: wavlm-base, wav2vec2-base/large, wav2vec2-large-960h, hubert-base-ls960 (LibriSpeech 960h)
  금지: wavlm-large, wavlm-base-plus (Libri-Light 60k h)
        wav2vec2-large-lv60*, *-ll60k*, wav2vec2-large-robust* (Libri-Light)
        xls-r, xlsr-53, mms-* (MLS)
        hubert-large-ls960-ft (Libri-Light로 사전학습, LS960은 fine-tune만)
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

ALLOWED_SSL = {
    "microsoft/wavlm-base",
    "facebook/wav2vec2-large",
    "facebook/wav2vec2-base",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-base-960h",
    "facebook/hubert-base-ls960",
}

# (regex, 이유), name_or_path 소문자로 비교
BANNED_SSL_PATTERNS = [
    (r"wavlm-(large|base-plus)", "WavLM Large/Base+ 는 Libri-Light 60k h로 사전학습 (§4.2 금지)"),
    (r"lv60|ll60k|libri-?light", "Libri-Light 사전학습 (§4.2 금지)"),
    (r"xls-?r|xlsr", "XLS-R/XLSR 는 MLS(영어 포함)로 사전학습 (§4.2 금지)"),
    (r"(^|/)mms-", "MMS 는 MLS 포함 사전학습 (§4.2 금지)"),
    (r"robust", "wav2vec2-large-robust 는 Libri-Light 포함 (§4.2 금지)"),
    (r"hubert-large|hubert-xlarge", "HuBERT Large/XL 은 Libri-Light 60k h로 사전학습 (§4.2 금지)"),
]


class ComplianceError(RuntimeError):
    pass


def check_ssl(name_or_path: Optional[str], allow_unlisted: bool = False) -> None:
    if not name_or_path:
        return
    name = str(name_or_path).strip()
    low = name.lower()
    for pat, why in BANNED_SSL_PATTERNS:
        if re.search(pat, low):
            raise ComplianceError(f"[compliance] SSL backbone '{name}' 사용 불가: {why}")
    if name in ALLOWED_SSL:
        return
    if allow_unlisted:
        print(f"[compliance] '{name}' 는 허용 목록에 없음. 사전학습 데이터 직접 확인 "
              f"(§4.2: 중복 여부를 확인할 수 없는 자원도 금지).")
        return
    raise ComplianceError(
        f"[compliance] SSL backbone '{name}' 는 허용 목록에 없음. 사전학습 데이터에 "
        f"LibriLight/MLS/MUSAN-speech 가 없는지 확인 후 ALLOWED_SSL 에 추가."
    )


def check_augment(aug: dict) -> None:
    aug = aug or {}
    if float(aug.get("musan_p", 0.0) or 0.0) > 0 or aug.get("musan_root"):
        raise ComplianceError(
            "[compliance] MUSAN 사용 불가 "
            "(§4.2 MUSAN speech 금지, music 서브셋도 보컬 포함). augment.musan_p=0, musan_root=null."
        )
    rir_root = str(aug.get("rir_root") or "")
    if rir_root and "musan" in rir_root.lower():
        raise ComplianceError("[compliance] rir_root 가 MUSAN 경로를 가리킵니다.")


def check_config(cfg: dict, extra_cfgs: Iterable[Optional[dict]] = (), allow_unlisted: bool = False) -> None:
    """실행 config와 체크포인트에 저장된 config를 같이 확인."""
    for c in [cfg, *extra_cfgs]:
        if not c:
            continue
        check_ssl(((c.get("model") or {}).get("ssl") or {}).get("name_or_path"), allow_unlisted)
        check_augment(c.get("augment") or {})
    print(f"[compliance] OK ssl={cfg['model']['ssl']['name_or_path']}  MUSAN=off  (ASVspoof5 eval plan §4.2/4.3)")
