# Phase 2B 실험 결과: Calibrated Forward Validation

## 1. 실험 목적

Phase 2B의 목적은 Phase 2에서 나온 average pooling compression 우호 신호가 실제로 유지되는지 더 엄격하게 확인하는 것이다.

Phase 2에서는 average pooling이 random selection보다 cosine이 높고, Gaussian noise보다 frozen decoder의 delta token NLL이 낮고, latent ablation에도 더 민감했다. 하지만 그 결과에는 세 가지 중요한 반론이 남아 있었다.

1. Gaussian noise의 sigma가 average pooling 난도보다 강하게 잡혔을 수 있다.
2. 512 samples, 2 epoch smoke run이라 결과가 작은 규모의 우연일 수 있다.
3. frozen T5 decoder bridge가 약해서 open-ended generation 실패를 compression 문제로 해석하기 어렵다.

Phase 2B는 이 반론들을 줄이기 위해 sample 수와 epoch를 늘리고, Gaussian sigma를 calibration으로 맞추고, frozen decoder 외에 lightweight token head를 추가하고, 원본 `h0` decoder control을 함께 측정했다.

## 2. 실행 정보

| 항목 | 값 |
|---|---|
| Kaggle kernel | `dennisparknd/lace-phase-2b-calibrated-validation` |
| kernel version | 1 |
| 실행 상태 | 완료 |
| 모델 | `t5-small` |
| encoder 상태 | frozen |
| 데이터 소스 | `hf:wikitext/wikitext-2-raw-v1:train` |
| 샘플 수 | 2048 |
| 최대 길이 | 128 token |
| hidden shape | `[2048, 128, 512]` |
| active tokens | 210286 |
| stage tokens | 64, 32, 16 |
| generation bridge | enabled |
| token-head bridge | enabled |
| 결과 위치 | `outputs/phase2b/lace_phase2b/` |

생성된 주요 파일:

| 파일 | 설명 |
|---|---|
| `metrics.json` | 전체 정량 결과 |
| `summary.md` | Kaggle runner가 생성한 요약 |
| `train_log.jsonl` | condition/stage별 학습 로그 |
| `generation_samples.jsonl` | qualitative generation sample |
| `checkpoints/*.pt` | condition/stage별 expander checkpoint |

## 3. Gaussian Calibration

Phase 2B에서는 Gaussian sigma를 더 촘촘하게 후보화했다.

```text
0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20
```

Average pooling 초기 난도와 가장 가까운 Gaussian sigma는 다음처럼 선택됐다.

| Stage | average pooling initial loss | matched sigma | sigma initial loss |
|---|---:|---:|---:|
| `z1` | 0.008145 | 0.10 | 0.009999 |
| `z2` | 0.012619 | 0.10 | 0.009999 |
| `z3` | 0.015713 | 0.125 | 0.015627 |

따라서 실제 Gaussian condition은 다음 sigma로 실행됐다.

```text
z1: 0.10
z2: 0.10
z3: 0.125
```

이 점이 Phase 2와 가장 큰 차이다. Phase 2에서는 Gaussian `z2/z3`가 0.20/0.40으로 상당히 강했기 때문에, average pooling이 유리해 보였을 가능성이 있었다. Phase 2B에서는 이 반론을 줄였다.

## 4. Gate 결과

Phase 2B의 strict overall pass는 실패했다.

하지만 Phase 3 candidate 기준은 통과했다.

| Gate | 결과 | 의미 |
|---|---|---|
| `P2B-G-RUN` | pass | required compression/corruption 조건이 같은 split에서 실행됨 |
| `P2B-G-MSE` | fail | average pooling은 MSE에서 random/matched Gaussian을 이기지 못함 |
| `P2B-G-COS` | fail | average pooling은 Gaussian보다 cosine이 낮음 |
| `P2B-G-DECODER-NLL` | pass | average pooling은 frozen decoder delta token NLL에서 가장 좋음 |
| `P2B-G-TOKEN-HEAD-NLL` | fail | lightweight token head에서는 Gaussian이 더 좋음 |
| `P2B-G-SCHEDULE` | fail | average pooling MSE가 완전한 단조 증가를 보이지 않음 |
| `P2B-G-USE` | pass | perturbation/ablation/swap에서 latent 사용 신호가 있음 |
| `P2B-G-GEN` | fail | reconstructed hidden 기반 open-ended generation은 의미 있는 샘플이 없음 |

Phase 3 candidate는 `P2B-G-DECODER-NLL`과 `P2B-G-USE` 두 신호로 통과했다.

```text
evidence_count = 2
min_evidence = 2
phase3_candidate = true
```

이 결과는 "Phase 2B가 성공했다"보다는 다음처럼 읽어야 한다.

> 엄격한 전체 기준으로는 실패했지만, compression forward가 frozen decoder bridge와 latent-use 측면에서 아직 살아 있는 신호를 보였기 때문에 Phase 3의 작은 prototype으로 넘어갈 근거는 있다.

## 5. 핵심 평균 지표

| 지표 | average_pool | random_select | matched gaussian_noise | 해석 |
|---|---:|---:|---:|---|
| 평균 MSE | 0.450913 | 0.443378 | 0.439971 | MSE는 compression 우위가 아님 |
| 평균 cosine | 0.309039 | 0.127890 | 0.357422 | random보다는 좋지만 Gaussian보다 낮음 |
| 평균 decoder delta NLL | 8.152192 | 8.302068 | 8.498025 | frozen decoder proxy에서는 average pooling이 가장 좋음 |
| 평균 token-head delta NLL | 1.441394 | 1.476511 | 1.074497 | token head에서는 Gaussian이 가장 좋음 |
| 평균 relative sensitivity | 0.016086 | - | - | latent perturbation에 반응함 |
| 평균 ablation delta | 0.011956 | - | - | latent 제거 시 성능이 악화됨 |
| 평균 swap delta | 0.034019 | - | - | latent swap에도 반응함 |
| meaningful generation rate | 0.0 | - | - | open-ended generation은 여전히 붕괴 |

## 6. 결과 해석

### 6.1 MSE

MSE는 average pooling에 불리했다.

```text
average_pool: 0.450913
random_select: 0.443378
gaussian_noise: 0.439971
```

이는 "compression이 hidden-state 좌표 복원을 더 잘한다"는 주장은 현재 성립하지 않는다는 뜻이다. 특히 matched Gaussian으로 보정한 뒤에도 Gaussian이 MSE에서 가장 낮았다.

따라서 MSE 기준으로는 LACE의 강한 주장을 뒷받침하지 못한다.

### 6.2 Cosine

Cosine은 mixed 결과다.

```text
average_pool: 0.309039
random_select: 0.127890
gaussian_noise: 0.357422
```

Average pooling은 random selection보다 representation 방향성을 훨씬 잘 보존했다. 이는 Phase 2와 같은 방향이다.

하지만 matched Gaussian보다 낮았다. 이 점은 중요하다. Phase 2에서는 average pooling이 Gaussian보다 token bridge에서 유리했지만, Phase 2B에서는 representation direction 자체는 Gaussian이 더 강했다.

따라서 cosine 결과는 다음처럼 해석해야 한다.

> Average pooling은 random drop류 corruption보다는 구조적인 representation 방향성을 남기지만, matched continuous noise보다 항상 우수하다고 말할 수는 없다.

### 6.3 Frozen Decoder Delta NLL

Frozen decoder delta NLL은 average pooling이 가장 좋았다.

```text
average_pool: 8.152192
random_select: 8.302068
gaussian_noise: 8.498025
```

이 지표는 `h0_hat`를 frozen T5 decoder에 넣었을 때, 원본 `h0` 대비 token NLL이 얼마나 악화되는지 본다.

값이 낮다는 것은 decoder가 `h0_hat`를 상대적으로 더 잘 읽는다는 뜻이다. 여기서 average pooling이 가장 낮았다는 점은 LACE에 우호적이다. MSE와 cosine에서 명확히 이기지 못했는데도, frozen decoder bridge에서는 가장 좋은 값을 냈기 때문이다.

방어 가능한 해석은 다음이다.

> Average pooling은 hidden vector 좌표나 방향성 전체에서는 최강이 아니지만, frozen T5 decoder가 token prediction에 활용하기 쉬운 형태의 정보를 일부 더 잘 보존했을 가능성이 있다.

### 6.4 Lightweight Token Head

Token-head delta NLL은 average pooling이 이기지 못했다.

```text
average_pool: 1.441394
random_select: 1.476511
gaussian_noise: 1.074497
```

이 지표는 frozen decoder와 별개로, 간단한 token classifier가 `h0_hat`에서 원래 token을 예측할 수 있는지 본다.

여기서는 Gaussian이 가장 좋았다. 이는 frozen decoder에서 average pooling이 좋았던 결과가 보편적인 token information 보존이라기보다, T5 decoder bridge와의 특정한 호환성일 수 있음을 보여준다.

따라서 token-head 결과는 LACE 주장에 제동을 건다.

> Average pooling의 decoder NLL 우위가 "일반적인 token reconstruction 우위"로 바로 확장되지는 않는다.

### 6.5 Latent Use

Latent-use gate는 통과했다.

Average pooling의 평균 latent-use 지표는 다음이다.

```text
relative sensitivity: 0.016086
ablation delta: 0.011956
swap delta: 0.034019
```

이 결과는 reverse expander가 average pooling latent를 실제로 사용하고 있다는 신호다.

특히 ablation delta와 swap delta가 양수라는 것은 latent를 제거하거나 다른 sample의 latent로 바꿨을 때 reconstruction이 나빠진다는 뜻이다. 이는 단순히 평균적인 hidden state를 찍어내는 것이 아니라, 입력 latent에 조건화된 복원을 하고 있음을 보여준다.

이 gate는 Phase 2B에서 중요한 긍정 신호다.

### 6.6 Open-ended Generation

Open-ended generation은 여전히 실패했다.

원본 `h0` decoder control에서는 meaningful sample rate가 0.75였다. 즉 frozen T5 decoder 자체는 원본 hidden state에서 꽤 그럴듯한 문장을 생성할 수 있다.

반면 average pooling으로 복원한 `h0_hat`의 meaningful generation rate는 0.0이었다.

이 차이는 중요하다.

Phase 2에서는 open-ended generation 실패를 decoder bridge 문제일 수 있다고 봤다. Phase 2B의 `h0` control은 그 반론을 일부 줄였다. 원본 `h0`에서는 generation이 되는데, reconstructed `h0_hat`에서는 안 된다.

따라서 현재 open-ended generation 실패는 단순히 frozen decoder 사용법 때문만은 아니다.

> Expander가 token prediction proxy에는 일부 유리한 신호를 만들지만, free generation에서 오류가 누적되지 않을 만큼 decoder-readable hidden state를 복원하지는 못하고 있다.

## 7. 최종 판정

| 항목 | 판정 |
|---|---|
| 실행 완결성 | 성공 |
| Gaussian calibration | 성공 |
| scale-up 검증 | 성공 |
| compression MSE 우위 | 실패 |
| compression cosine 우위 | 부분 실패 |
| frozen decoder bridge 우위 | 성공 |
| token-head bridge 우위 | 실패 |
| latent 사용 신호 | 성공 |
| open-ended generation | 실패 |
| strict overall pass | 실패 |
| Phase 3 후보 | 조건부 통과 |

한 줄 결론:

> Phase 2B는 LACE 가설을 강하게 입증하지 못했다. 그러나 matched Gaussian과 더 큰 run에서도 average pooling compression은 frozen decoder bridge와 latent-use에서 살아 있는 신호를 보였으므로, 작은 Phase 3 prototype으로 넘어가 "generation-capable reverse objective"를 직접 시험할 근거는 있다.

## 8. 다음 단계

다음 단계는 두 갈래 중 하나다.

### 선택 A. Phase 3A: Generation-capable objective로 이동

현재 병목은 hidden reconstruction 자체보다 reconstructed hidden이 open-ended generation에 충분히 맞지 않는다는 점이다.

따라서 Phase 3A에서는 reverse expander를 단순 MSE/cosine으로만 학습하지 않고, token objective를 훈련 loss에 일부 포함해야 한다.

예시:

```text
L = L_hidden_rec + lambda_cos * L_cos + lambda_token * L_token_head
```

이 선택은 "LACE가 generation으로 이어지는가?"를 직접 확인한다.

### 선택 B. Phase 2C: Token-head bridge 보강

아직 token-head bridge가 매우 작고 짧게 학습됐다. Phase 2B에서는 token head가 12 train batches만 사용했다.

따라서 token-head를 더 충분히 학습하고, 여러 seed로 반복하면 average pooling의 token-head 결과가 달라질 수 있다.

이 선택은 더 보수적이지만, 연구 속도는 느려진다.

현재 권고는 **Phase 3A**다.

이유는 Phase 2B가 이미 다음 사실을 보여줬기 때문이다.

1. hidden reconstruction 우위만으로는 LACE 주장을 세우기 어렵다.
2. frozen decoder bridge에서는 compression 우호 신호가 남아 있다.
3. open-ended generation은 reconstruction-only objective로는 부족하다.

따라서 다음 질문은 "더 잘 복원할 수 있는가?"가 아니라 다음이어야 한다.

> token/generation objective를 reverse expansion에 직접 넣으면 compression forward의 장점이 실제 생성 경로로 살아나는가?
