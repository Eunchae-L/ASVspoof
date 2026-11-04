# ASVspoof 5 Track 1

ASVspoof 5 Track 1(stand-alone speech deepfake detection)용 코드. 동결한 SSL 백본(기본 `microsoft/wavlm-large`)의 hidden state를 learnable layer weight로 합치고, adapter(BiLSTM 또는 SE-Res2Net)로 줄인 표현 z 위에 분류기(LCNN-big 또는 SENet12, ASP pooling)를 둔다. 학습할 때는 z에서 clean 파형의 MFCC / CQT / LFCC / log-mel을 복원하는 보조 regression worker를 같이 붙여 BCE + λ·L1(λ=0.1)로 학습한다. 증강은 RawBoost, codec round-trip(mp3 / aac / amr-nb / narrowband-8k), MUSAN, RIR, SpecAugment를 쓴다. best checkpoint는 dev minDCF 기준으로 고르고, 평가는 EER / minDCF / actDCF / Cllr를 공식 방향(bona fide = positive)으로 계산한다.

원래 ADD 2023 Track 1.2용으로 만든 구조를 ASVspoof 5로 옮긴 것이다.

## 구조

```
configs/baseline.yaml        기본 설정
data/
  protocols.py               ASVspoof 5 protocol 파서
  dataset.py                 waveform dataset / dataloader
  augment.py                 RawBoost, codec, MUSAN, RIR, SpecAugment
  cached_dataset.py          캐시된 SSL feature용 dataset
models/
  ssl_backbone.py            frozen SSL + layer weighted sum
  adapter.py                 BiLSTM / SE-Res2Net adapter
  classifier.py              LCNN-big / SENet12 + ASP
  workers.py                 보조 regression worker와 target 계산
  model.py                   전체 모델
utils/
  metrics.py                 EER, minDCF, actDCF, Cllr
  tracking.py                W&B 로깅 (실패 시 로컬 jsonl)
  run_meta.py                git commit, 환경 정보 기록
scripts/
  download_asvspoof5.py      HF에서 데이터 다운로드
  cache_ssl_features.py      SSL feature 캐시
  collect_runs.py            run별 결과 표 / csv
  wandb_replay.py            로컬 로그로 W&B run 복원
train.py                     학습
train_cached.py              캐시된 feature로 adapter + classifier만 학습
evaluate.py                  dev / eval 평가
notebooks/
  ASVspoof5_colab.ipynb          Colab, full train
  ASVspoof5_colab_subset.ipynb   Colab, 50% 층화 subset + 증강 + eval 분석
  data_analysis.ipynb            샘플레이트 / 길이 / 패딩 분포 확인
```

## 데이터

HF `jungjee/asvspoof5`에서 받는다.

```
pip install -r requirements.txt
python scripts/download_asvspoof5.py --local-dir raw [--include-eval]
```

tar를 `data.root`(기본 `./data/asvspoof5`)에 풀면 `flac_T/`, `flac_D/`, `flac_E_eval/`, `protocols/*.tsv`가 생긴다. 경로가 다르면 `configs/baseline.yaml`의 `data` 항목을 고친다.

## 실행

학습

```
python train.py --config configs/baseline.yaml --tag a1c1 \
    --adapter bilstm --classifier lcnn_big \
    --dev-subset 5000 --dev-batch-size 64 [--epochs N] [--batch_size B]
```

- `--smoke N`: train N개로 빠르게 확인
- `--resume ckpt/baseline/a1c1/last.pt`: 이어서 학습 (`--save-every N`으로 epoch 중간 저장)
- `--save-all-epochs`: epoch별 `epoch_NN.pt` 저장
- `--wandb` / `--no-wandb`, `--wandb-project`, `--wandb-run-name`

결과는 `ckpt/baseline/<tag>/`에 `best.pt`, `last.pt`, `history.json`, `train_log.jsonl`, `run_meta.json`, `config_used.yaml`로 남는다.

캐시된 feature로 학습 (SSL forward 없이 adapter + classifier만, worker 없음)

```
python scripts/cache_ssl_features.py --config configs/baseline.yaml --split train --out cache/train
python scripts/cache_ssl_features.py --config configs/baseline.yaml --split dev   --out cache/dev
python train_cached.py --config configs/baseline.yaml --cache-dir cache --tag a1c1_cached
```

캐시는 layer weight 초기값(균등 평균) 기준이고 증강도 고정되므로 구조 탐색용으로만 썼다.

평가

```
python evaluate.py --config configs/baseline.yaml --ckpt ckpt/baseline/a1c1/best.pt \
    --split dev --metric official --out scores/dev_a1c1.tsv [--sliding] [--batch-size 64] [--limit N]
```

기본은 공식 baseline과 같은 방식(앞 4초 한 구간, 짧으면 반복)이고 `--sliding`은 4초 window / 2초 hop 평균이다. `--metric official`이면 score = −logit(클수록 bona fide). 출력은 `<file_id>\t<score>\t<label>` TSV와 `_summary.json`.

여러 run 비교

```
python scripts/collect_runs.py --ckpt-root ckpt --out runs_summary.csv
```

Colab에서는 `notebooks/ASVspoof5_colab_subset.ipynb`를 위에서부터 실행하면 된다 (`DRIVE_ROOT`만 수정).

## 결과

| 설정 | split | EER | minDCF |
|---|---|---|---|
| a1c1_v2_subset50 (BiLSTM + LCNN-big, train 50% subset, codec/MUSAN/RIR 증강) | eval (pooled) | 4.18% | 0.1143 |

나머지 조합(a1c2, a2c1, a2c2)은 아직 정리하지 못했다.

## 참고

- ASVspoof 5 Evaluation Plan (Delgado et al., 2024), ASVspoof 5 (Wang et al., Interspeech 2024)
- WavLM (Chen et al., 2022), RawBoost (Tak et al., 2022), PASE+ (Ravanelli et al., 2020)
- LCNN (Wu et al., 2018), SENet (Hu et al., 2018), Res2Net (Gao et al., 2019), ASP (Okabe et al., 2018)
