# Phase 2C 연구 진행 계획서: Adding Positional Encoding

**작성일**: 2026-05-04  
**목표 단계**: Phase 2C  
**직전 기준점**: Phase 2B calibrated validation 완료

## 1. 목적

Phase 2C의 목적은 Average Pooling compression이 open-ended generation에서 붕괴한 원인 중 하나가 위치/순서 정보 부족인지 검증하는 것이다.

Phase 2B에서는 Average Pooling이 hidden-state MSE나 cosine에서 강한 우위를 보이지 못했지만, frozen decoder delta token NLL과 latent-use 지표에서는 살아 있는 신호를 보였다. 이 조합은 다음 가능성을 남긴다.

> Average Pooling latent에는 content signal이 일부 남아 있지만, reverse expansion이 decoder-readable hidden trajectory를 만들 때 필요한 position scaffold가 부족할 수 있다.

Phase 2C는 이 가능성을 직접 검증한다.

## 2. 핵심 질문

1. fixed sinusoidal positional encoding을 expander에 조건으로 주면 Average Pooling reconstruction이 좋아지는가?
2. 좋아진다면 그 개선은 hidden-state MSE/cosine에서 나타나는가, 아니면 decoder/token proxy에서만 나타나는가?
3. 같은 positional conditioning이 Gaussian noise에도 동일하게 이득을 주는가?
4. position-only control이 비슷한 성능을 내는가, 아니면 실제 latent content가 필요하다는 신호가 유지되는가?
5. wrong-position evaluation에서 성능이 악화되는가, 즉 expander가 positional signal을 실제로 쓰는가?

## 3. 실험 조건

| Condition | Latent source | Positional feature | 목적 |
|---|---|---|---|
| `average_pool` | Average pooled hidden state | learned bias only | Phase 2B baseline |
| `average_pool_positional` | Average pooled hidden state | fixed sinusoidal + learned bias | 위치 scaffold가 Average Pooling을 보완하는지 확인 |
| `position_only` | zero latent with Average Pooling shape | fixed sinusoidal + learned bias | 개선이 content latent 없이 위치 prior만으로 가능한지 확인 |
| `random_select` | random selected hidden states | learned bias only | token-budget corruption baseline |
| `gaussian_noise` | matched Gaussian noisy h0 | learned bias only | continuous corruption baseline |
| `gaussian_noise_positional` | matched Gaussian noisy h0 | fixed sinusoidal + learned bias | positional feature가 모든 condition에 주는 일반 이득 확인 |

## 4. 측정 지표

| 지표 | 측정하는 것 | 좋은 결과의 의미 | 주의할 점 |
|---|---|---|---|
| MSE | hidden-state 좌표 복원 오차 | positional conditioning이 실제 좌표 복원을 돕는다 | MSE 개선만으로 generation 가능성을 주장할 수는 없음 |
| cosine | representation 방향성 보존 | semantic/directional structure가 더 맞는다 | Gaussian이 여전히 강할 수 있음 |
| frozen decoder delta NLL | T5 decoder가 reconstructed hidden을 읽는 정도 | decoder-readable trajectory가 개선된다 | T5 decoder-specific proxy일 수 있음 |
| token-head delta NLL | lightweight classifier가 token 정보를 읽는 정도 | 일반 token reconstruction proxy도 개선된다 | token head 학습량이 작아 noisy할 수 있음 |
| latent ablation/swap | expander가 latent를 실제 사용하는지 | position만이 아니라 content latent도 필요하다 | position-only control과 함께 봐야 함 |
| wrong-position delta | position feature를 틀리게 줬을 때 악화되는지 | positional signal을 실제로 사용한다 | 너무 작으면 positional feature가 무시됐을 수 있음 |
| meaningful generation rate | free generation이 붕괴에서 벗어나는지 | decoder-readable hidden path가 일부 살아난다 | 가장 noisy하므로 보조 지표로 해석 |

## 5. Gate

| Gate | 의미 |
|---|---|
| `P2C-G-RUN` | baseline, positional, position-only, random, Gaussian controls가 같은 split에서 실행됨 |
| `P2C-G-POS-MSE` | `average_pool_positional`이 `average_pool`보다 MSE가 낮음 |
| `P2C-G-POS-COS` | `average_pool_positional`이 `average_pool`보다 cosine이 높음 |
| `P2C-G-POS-DECODER-NLL` | `average_pool_positional`이 frozen decoder delta NLL을 낮춤 |
| `P2C-G-POS-TOKEN-HEAD-NLL` | `average_pool_positional`이 token-head delta NLL을 낮춤 |
| `P2C-G-POS-USE` | positional expander도 latent perturbation/ablation/swap에 반응함 |
| `P2C-G-POS-CONTROL` | `average_pool_positional`이 `position_only`보다 좋아서 위치 prior만의 효과가 아님 |
| `P2C-G-POS-MATTERS` | wrong-position evaluation이 reconstruction을 악화시킴 |
| `P2C-G-GEN` | meaningful generation이 `average_pool` baseline보다 좋아짐 |

Phase 2C의 목적은 strict overall pass가 아니다. 더 중요한 것은 다음이다.

> positional conditioning이 Average Pooling의 약점을 실제로 보완하는지, 그리고 그 보완이 content latent 없이도 가능한 position prior인지 아닌지 분리하는 것.

## 6. 산출물

| 산출물 | 경로 |
|---|---|
| Kaggle runner | `kaggle/phase2c/run_phase2c.py` |
| Kaggle metadata | `kaggle/phase2c/kernel-metadata.json` |
| push script | `scripts/push_kaggle_phase2c.sh` |
| metrics | `outputs/phase2c/lace_phase2c/metrics.json` |
| summary | `outputs/phase2c/lace_phase2c/summary.md` |
| generation samples | `outputs/phase2c/lace_phase2c/generation_samples.jsonl` |
| 결과 보고서 | `docs/experiments/phase-2c-positional-encoding.md` |

## 7. 실행 명령

```bash
kaggle kernels push -p kaggle/phase2c --accelerator NvidiaTeslaT4 --timeout 3600
kaggle kernels status dennisparknd/lace-phase-2c-positional-encoding
kaggle kernels output dennisparknd/lace-phase-2c-positional-encoding -p outputs/phase2c
```

## 8. 해석 원칙

Phase 2C 결과는 다음 순서로 해석한다.

1. positional feature가 Average Pooling 자체를 개선했는지 본다.
2. 개선이 hidden reconstruction인지 token/generation proxy인지 분리한다.
3. position-only control보다 좋아야 content latent가 필요하다고 말할 수 있다.
4. Gaussian positional도 함께 좋아졌다면 positional conditioning은 일반 reverse-expansion scaffold일 수 있다.
5. wrong-position delta가 있어야 positional feature를 실제로 사용했다고 볼 수 있다.
6. open-ended generation은 가장 마지막에 보며, early run에서는 과도하게 주장하지 않는다.
