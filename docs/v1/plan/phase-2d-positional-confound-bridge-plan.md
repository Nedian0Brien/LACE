# Phase 2D 연구 진행 계획서: Positional Confound Bridge

**작성일**: 2026-05-04  
**목표 단계**: Phase 2D  
**직전 기준점**: Phase 2C positional encoding 완료  
**상위 목표**: Phase 3A generation-aware reverse objective로 넘어가기 전 confound 정리

## 1. 목적

Phase 2D의 목적은 Phase 3A로 넘어가기 전에 positional conditioning의 confound를 정리하는 것이다.

Phase 2C에서는 explicit sinusoidal positional encoding이 Average Pooling expander의 MSE, cosine, decoder delta NLL, token-head delta NLL을 모두 크게 개선했다. 그러나 동시에 `position_only`와 `gaussian_noise_positional`도 매우 강했다. 따라서 Phase 2C 결과만으로는 다음을 분리하기 어렵다.

1. Average Pooling latent가 content 정보를 실제로 제공했는가?
2. positional feature가 hidden trajectory의 평균 template를 거의 복원한 것인가?
3. token-head proxy가 content를 읽은 것인가, 위치별 token prior를 읽은 것인가?
4. Absolute position이 중요한가, 아니면 Average Pooling이 잃어버린 block 내부 relative position이 중요한가?

Phase 2D는 이 질문에 답하기 위한 중간 다리다. 2D 자체가 최종 성공 실험은 아니다. 3A에서 token/generation objective를 넣기 전에, 어떤 control을 유지해야 하는지 결정하는 진단 실험이다.

## 2. 핵심 질문

| 질문 | 측정 방법 | 3A와의 연결 |
|---|---|---|
| Absolute position이 주효한가 | `average_pool_abs_pos` vs `average_pool` | global position scaffold만으로 충분한지 확인 |
| Block-relative position이 주효한가 | `average_pool_rel_pos` vs `average_pool` | Average Pooling의 within-block order 손실을 직접 겨냥 |
| Absolute+relative가 가장 안정적인가 | `average_pool_abs_rel_pos` 비교 | 3A의 기본 positional condition 후보 |
| Position-only가 여전히 강한가 | `position_only_*` controls | token loss가 position prior로 shortcut되는지 확인 |
| Token proxy가 content-sensitive한가 | shuffled-label delta NLL | 3A의 token objective가 content를 강제할 수 있는지 확인 |
| Gaussian positional이 계속 더 강한가 | `gaussian_noise_abs_pos` | compression forward의 강한 반론 확인 |

## 3. 실험 조건

| Condition | Latent source | Positional mode | 목적 |
|---|---|---|---|
| `average_pool` | Average pooled hidden state | learned bias only | Phase 2B/2C baseline |
| `average_pool_abs_pos` | Average pooled hidden state | absolute sinusoidal | global position scaffold 효과 |
| `average_pool_rel_pos` | Average pooled hidden state | block-relative sinusoidal | pooling block 내부 위치 복원 효과 |
| `average_pool_abs_rel_pos` | Average pooled hidden state | absolute + block-relative | 3A 후보 condition |
| `position_only_abs_pos` | zero latent | absolute sinusoidal | absolute position prior control |
| `position_only_rel_pos` | zero latent | block-relative sinusoidal | relative position prior control |
| `position_only_abs_rel_pos` | zero latent | absolute + block-relative | strongest position-only control |
| `gaussian_noise` | matched Gaussian noisy h0 | learned bias only | continuous corruption baseline |
| `gaussian_noise_abs_pos` | matched Gaussian noisy h0 | absolute sinusoidal | Phase 2C의 strongest control 재확인 |

## 4. 새 진단 지표

### 4.1 Wrong-position shift sweep

Phase 2C에서는 wrong-position shift를 하나만 봤다. Phase 2D에서는 다음 sweep을 본다.

```text
shift = 1, 2, 4, 8, 16
```

이 지표는 positional feature를 틀리게 줬을 때 reconstruction이 얼마나 악화되는지 측정한다.

좋은 결과는 적어도 작은 shift에서 MSE delta가 양수로 나오는 것이다. 특히 block-relative position은 block size 주기로 반복되므로, shift가 block size의 배수이면 변화가 작거나 0일 수 있다. 그래서 sweep에는 `1`, `2`처럼 작은 shift를 반드시 포함한다.

### 4.2 Shuffled-label delta NLL

Token proxy가 content를 읽는지 확인하기 위해, reconstructed hidden은 그대로 두고 target token labels만 batch 안에서 뒤집어 NLL을 다시 측정한다.

```text
shuffled_delta_token_nll = NLL(h_hat, shuffled_labels) - NLL(h_hat, true_labels)
```

이 값이 크면 reconstructed hidden이 sample-specific token content를 담고 있을 가능성이 크다. 값이 작으면 token proxy가 content보다 position prior나 frequency prior에 의존했을 가능성이 있다.

## 5. Gate

| Gate | 의미 |
|---|---|
| `P2D-G-RUN` | 모든 2D condition이 같은 split에서 실행됨 |
| `P2D-G-ABS-GAIN` | absolute position이 Average Pooling baseline보다 MSE/decoder NLL을 개선 |
| `P2D-G-REL-GAIN` | block-relative position이 Average Pooling baseline보다 MSE/decoder NLL을 개선 |
| `P2D-G-ABSREL-GAIN` | absolute+relative position이 Average Pooling baseline보다 MSE/decoder NLL을 개선 |
| `P2D-G-REL-CONTENT` | relative Average Pooling이 relative position-only보다 cosine/decoder NLL에서 좋음 |
| `P2D-G-ABSREL-CONTENT` | absolute+relative Average Pooling이 strongest position-only보다 content-sensitive함 |
| `P2D-G-SHUFFLED-LABEL` | shuffled label에서 token proxy가 악화됨 |
| `P2D-G-LATENT-USE` | absolute+relative Average Pooling이 perturbation/ablation/swap에 반응 |
| `P2D-G-POS-MATTERS` | wrong-position sweep에서 reconstruction이 악화됨 |
| `P2D-G-GAUSSIAN-STRONGER` | Gaussian absolute positional이 여전히 token proxy에서 더 강한지 확인하는 risk gate |

`P2D-G-GAUSSIAN-STRONGER`는 좋은 gate가 아니라 risk signal이다. 이 gate가 pass되면 compression path의 강한 반론이 유지된다는 뜻이다.

## 6. 산출물

| 산출물 | 경로 |
|---|---|
| Kaggle runner | `kaggle/phase2d/run_phase2d.py` |
| Kaggle metadata | `kaggle/phase2d/kernel-metadata.json` |
| push script | `scripts/push_kaggle_phase2d.sh` |
| metrics | `outputs/phase2d/lace_phase2d/metrics.json` |
| summary | `outputs/phase2d/lace_phase2d/summary.md` |
| generation samples | `outputs/phase2d/lace_phase2d/generation_samples.jsonl` |
| 결과 보고서 | `docs/experiments/phase-2d-positional-confound-bridge.md` |

## 7. 실행 명령

```bash
rtk kaggle kernels push -p kaggle/phase2d --accelerator NvidiaTeslaT4 --timeout 3600
rtk kaggle kernels status dennisparknd/lace-phase-2d-positional-confound-bridge
rtk kaggle kernels output dennisparknd/lace-phase-2d-positional-confound-bridge -p outputs/phase2d
```

## 8. 해석 원칙

Phase 2D는 Phase 3A보다 중요한 실험이 아니다. Phase 2D의 목적은 다음 실험을 더 깨끗하게 만드는 것이다.

따라서 결과 해석은 다음 순서로 한다.

1. Absolute / relative / absolute+relative 중 어떤 positional scaffold가 가장 안정적인지 본다.
2. 같은 positional mode에서 Average Pooling이 position-only보다 좋은지 본다.
3. shuffled-label delta가 충분히 커서 token proxy가 content-sensitive한지 본다.
4. Gaussian positional이 계속 더 강한지 본다.
5. 위 결과를 바탕으로 Phase 3A의 기본 condition과 control set을 결정한다.

Phase 2D의 가장 유용한 결론은 다음 형태여야 한다.

> Phase 3A에서는 어떤 positional condition을 기본으로 쓰고, 어떤 position-only/Gaussian controls를 반드시 포함해야 하는가?
