# Phase 2B 연구 진행 계획서: Calibrated Forward Validation

**작성일**: 2026-05-04  
**목표 단계**: Phase 2B  
**직전 기준점**: Phase 2 forward process isolation 완료

## 1. 목적

Phase 2B의 목적은 Phase 2에서 나온 average pooling compression 우호 신호가 실험 설계의 우연인지, 아니면 compression forward process의 실제 장점인지 재검증하는 것이다.

Phase 2에서는 average pooling이 random selection보다 cosine이 높고, Gaussian noise보다 frozen decoder의 delta token NLL이 낮고, latent ablation에도 더 민감했다. 하지만 다음 반론이 남아 있다.

1. Gaussian sigma가 average pooling 난도보다 과하게 강했을 수 있다.
2. 512 samples, 2 epoch smoke run이라 결과가 작고 불안정할 수 있다.
3. frozen T5 decoder bridge가 약해서 open-ended generation 붕괴를 compression 문제로 보기 어렵다.
4. gate가 MSE/cosine/token NLL을 묶어서 판정해 해석이 관대했다.

Phase 2B는 이 네 반론을 직접 다룬다.

## 2. 핵심 변경

| 항목 | Phase 2 | Phase 2B |
|---|---|---|
| sample 수 | 512 | 기본 2048 |
| epoch | 2 | 기본 4 |
| Gaussian sigma | 0.10, 0.20, 0.40 | 0.05, 0.10, 0.15 및 matched sigma |
| token bridge | frozen T5 decoder | frozen T5 decoder + lightweight token head |
| generation control | condition별 sample만 확인 | 원본 `h0` decoder control 추가 |
| gate | 통합 pass/fail | MSE, cosine, decoder NLL, token-head NLL, latent-use, generation 분리 |

## 3. 판정 기준

Phase 2B에서는 strong overall pass보다 evidence count를 더 중요하게 본다.

| Gate | 의미 |
|---|---|
| `P2B-G-MSE` | average pooling이 hidden-state 좌표 복원에서 random/matched Gaussian보다 나은가 |
| `P2B-G-COS` | average pooling이 representation 방향성을 더 잘 보존하는가 |
| `P2B-G-DECODER-NLL` | frozen T5 decoder proxy에서 average pooling이 낮은 delta token NLL을 보이는가 |
| `P2B-G-TOKEN-HEAD-NLL` | lightweight token head에서도 token reconstruction 우위가 재현되는가 |
| `P2B-G-SCHEDULE` | compression stage가 깊어질수록 난도가 단조롭게 증가하는가 |
| `P2B-G-USE` | perturbation, ablation, swap에서 latent 사용 신호가 있는가 |
| `P2B-G-GEN` | open-ended generation 붕괴가 줄어드는가 |

Phase 3로 넘어가기 위한 최소 조건은 다음이다.

> `P2B-G-COS`, `P2B-G-DECODER-NLL`, `P2B-G-TOKEN-HEAD-NLL`, `P2B-G-USE`, `P2B-G-GEN` 중 최소 2개 이상이 재현된다.

MSE 우위는 있으면 좋지만 필수 조건으로 두지 않는다. LACE의 핵심 주장이 단순 좌표 복원이 아니라 language-relevant reverse path이기 때문이다.

## 4. 산출물

| 산출물 | 경로 |
|---|---|
| Kaggle runner | `kaggle/phase2b/run_phase2b.py` |
| Kaggle metadata | `kaggle/phase2b/kernel-metadata.json` |
| push script | `scripts/push_kaggle_phase2b.sh` |
| metrics | `outputs/phase2b/lace_phase2b/metrics.json` |
| summary | `outputs/phase2b/lace_phase2b/summary.md` |
| generation samples | `outputs/phase2b/lace_phase2b/generation_samples.jsonl` |

## 5. 실행 명령

```bash
kaggle kernels push -p kaggle/phase2b --accelerator NvidiaTeslaT4 --timeout 3600
kaggle kernels status dennisparknd/lace-phase-2b-calibrated-validation
kaggle kernels output dennisparknd/lace-phase-2b-calibrated-validation -p outputs/phase2b
```

## 6. 해석 원칙

Phase 2B 결과를 해석할 때는 다음을 분리한다.

1. MSE가 낮은가
2. cosine 방향성이 보존되는가
3. decoder/token-head proxy가 좋아지는가
4. latent를 실제로 쓰는가
5. open-ended generation이 bridge/control 대비 나아지는가

이 중 일부만 좋아도 연구 가설을 부분적으로 지지할 수 있다. 그러나 open-ended generation이 계속 붕괴되면 Phase 3에서는 먼저 decoder bridge 또는 generation objective를 재설계해야 한다.
