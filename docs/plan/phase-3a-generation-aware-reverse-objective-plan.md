# Phase 3A 연구 진행 계획서: Generation-aware Reverse Objective

**작성일**: 2026-05-04  
**목표 단계**: Phase 3A  
**직전 기준점**: Phase 2D positional confound bridge 완료  
**상위 목표**: reconstruction-only reverse expansion을 token/generation-aware reverse expansion으로 확장

## 1. 목적

Phase 3A의 목적은 reverse expander 학습에 token-level objective를 직접 포함했을 때, Average Pooling latent가 open-ended generation으로 이어질 수 있는지 확인하는 것이다.

Phase 2B와 2C, 2D에서 확인한 가장 중요한 병목은 다음이다.

> hidden-state reconstruction proxy와 teacher-forced token proxy는 일부 개선되지만, reconstructed `h0_hat` 기반 open-ended generation은 아직 안정적으로 살아나지 않는다.

따라서 Phase 3A는 단순히 MSE를 더 낮추는 실험이 아니다. 이제는 reverse expander가 decoder/token-readable hidden state를 만들도록 훈련 objective를 바꿔야 한다.

## 2. 핵심 질문

| 질문 | 측정 방법 | 좋은 결과 |
|---|---|---|
| token objective가 token proxy를 개선하는가 | `average_pool_rel_pos_tok*` vs `average_pool_rel_pos_recon` | token-head delta NLL 감소 |
| token objective가 frozen decoder proxy도 개선하는가 | decoder delta NLL 비교 | decoder delta NLL 감소 또는 큰 악화 없음 |
| latent use가 유지되는가 | perturbation/ablation/swap | token loss 후에도 latent sensitivity 유지 |
| position-only shortcut으로 빠지지 않는가 | `position_only_*_tok010` controls | Average Pooling이 cosine/shuffled/content 지표에서 우위 |
| generation collapse가 줄어드는가 | meaningful generation rate와 sample audit | 반복/공백 collapse 감소 |
| Gaussian control과 gap이 줄어드는가 | `gaussian_noise_abs_pos_*` controls | Average Pooling token objective가 Gaussian gap 일부 축소 |

## 3. 기본 objective

Phase 3A의 expander training loss는 다음과 같다.

```text
L = L_hidden_mse
  + lambda_cos * L_cos
  + lambda_var * L_var
  + lambda_token * L_token_head
```

여기서 `L_token_head`는 Phase 2B 이후 사용한 lightweight token head를 frozen `h0`에서 먼저 학습한 뒤, expander output `h0_hat`에 적용하는 cross-entropy loss다.

중요한 점은 token head 자체를 같이 학습하지 않는 것이다. token head를 같이 업데이트하면 expander가 좋아진 것인지 classifier가 적응한 것인지 분리하기 어렵다. Phase 3A에서는 token head를 먼저 oracle hidden에 맞춰 학습하고 freeze한 뒤, expander만 업데이트한다.

## 4. 실험 조건

| Arm | Latent source | Positional mode | `lambda_token` | 목적 |
|---|---|---|---:|---|
| `average_pool_rel_pos_recon` | Average Pooling | block-relative | 0.00 | Phase 2D primary baseline 재현 |
| `average_pool_rel_pos_tok005` | Average Pooling | block-relative | 0.05 | 약한 token objective |
| `average_pool_rel_pos_tok010` | Average Pooling | block-relative | 0.10 | 기본 token objective 후보 |
| `average_pool_rel_pos_tok020` | Average Pooling | block-relative | 0.20 | 강한 token objective, collapse 위험 확인 |
| `average_pool_abs_rel_pos_tok010` | Average Pooling | absolute+relative | 0.10 | 더 강한 positional scaffold 후보 |
| `position_only_rel_pos_tok010` | zero latent | block-relative | 0.10 | relative position shortcut control |
| `position_only_abs_rel_pos_tok010` | zero latent | absolute+relative | 0.10 | strongest position-only shortcut control |
| `gaussian_noise_abs_pos_recon` | matched Gaussian noisy h0 | absolute | 0.00 | Phase 2D strongest control baseline |
| `gaussian_noise_abs_pos_tok010` | matched Gaussian noisy h0 | absolute | 0.10 | token objective가 Gaussian path도 강화하는지 확인 |

## 5. Gate

| Gate | 의미 |
|---|---|
| `P3A-G-RUN` | 모든 Phase 3A arm이 같은 split에서 실행됨 |
| `P3A-G-TOKEN-HEAD` | Average Pooling token objective가 reconstruction-only보다 token-head delta NLL을 낮춤 |
| `P3A-G-DECODER` | token objective가 frozen decoder delta NLL을 개선하거나 큰 악화를 만들지 않음 |
| `P3A-G-LATENT-USE` | best Average Pooling token arm이 perturbation/ablation/swap에 계속 반응함 |
| `P3A-G-CONTENT-CONTROL` | best Average Pooling token arm이 position-only token arm보다 content-sensitive함 |
| `P3A-G-GENERATION` | Average Pooling token objective가 meaningful generation을 0보다 높이고 baseline보다 개선함 |
| `P3A-G-GAUSSIAN-GAP` | Average Pooling token objective가 Gaussian control과의 token/generation gap을 줄였는지 확인 |

`P3A-G-GAUSSIAN-GAP`은 strict success gate라기보다 risk gate다. Gaussian이 계속 압도적으로 강하면 compression-forward claim은 여전히 보류해야 한다.

## 6. 실행 산출물

| 산출물 | 경로 |
|---|---|
| Kaggle runner | `kaggle/phase3a/run_phase3a.py` |
| Kaggle metadata | `kaggle/phase3a/kernel-metadata.json` |
| push script | `scripts/push_kaggle_phase3a.sh` |
| metrics | `outputs/phase3a/lace_phase3a/metrics.json` |
| summary | `outputs/phase3a/lace_phase3a/summary.md` |
| generation samples | `outputs/phase3a/lace_phase3a/generation_samples.jsonl` |
| 결과 보고서 | `docs/experiments/phase-3a-generation-aware-reverse-objective.md` |

## 7. 실행 명령

```bash
rtk kaggle kernels push -p kaggle/phase3a --accelerator NvidiaTeslaT4 --timeout 3600
rtk kaggle kernels status dennisparknd/lace-phase-3a-generation-aware-reverse-objective
rtk kaggle kernels output dennisparknd/lace-phase-3a-generation-aware-reverse-objective -p outputs/phase3a
```

## 8. 해석 원칙

Phase 3A에서 token-head NLL이 내려가는 것만으로는 성공이 아니다.

성공으로 보려면 적어도 다음 세 가지를 함께 만족해야 한다.

1. `average_pool_rel_pos_tok*`가 `average_pool_rel_pos_recon`보다 token proxy를 개선한다.
2. 같은 arm이 latent ablation/swap sensitivity를 유지한다.
3. `position_only_*_tok010`이 같은 token objective에서 같이 좋아지는 shortcut을 통제한다.

Open-ended generation이 좋아지면 매우 강한 신호지만, sample 수가 작고 heuristic quality metric이 거칠기 때문에 반드시 qualitative sample audit과 함께 해석한다.

가장 정직한 결론은 다음 형태여야 한다.

> Token objective가 compression latent의 generation-readability를 실제로 높였는가, 아니면 position-only/Gaussian controls도 똑같이 좋아졌는가?
