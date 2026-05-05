# Phase 1 실험 결과: Latent Compression 및 Reverse Expansion Smoke Run

## 1. 실험 목적

Phase 1의 목적은 LACE의 핵심 루프가 학습 가능한지 확인하는 것이다. Phase 0에서는 `h0`를 `z1/z2/z3`로 줄이고 다시 길이만 맞춰 metric을 보았다. Phase 1에서는 여기서 한 단계 더 나아가, 압축된 latent에서 원래 encoder latent `h0`를 복원하는 expander를 실제로 학습했다.

핵심 질문은 다음과 같다.

> 압축된 latent `z_t`를 보고 더 상세한 latent, 최종적으로 `h0`를 복원하도록 학습할 수 있는가?

이 실험은 LACE가 baseline보다 우수하다는 것을 증명하는 실험이 아니다. 이 실험의 목적은 **학습 가능한 reverse expansion loop가 성립하는지** 확인하는 것이다.

## 2. 실행 정보

| 항목 | 값 |
|---|---|
| Kaggle kernel | `dennisparknd/lace-phase-1-latent-compression` |
| kernel version | 1 |
| 실행 상태 | 완료 |
| 장치 | `cuda` |
| 모델 | `t5-small` |
| encoder 상태 | frozen |
| 샘플 수 | 512 |
| 최대 길이 | 128 token |
| hidden shape | `[512, 128, 512]` |
| hidden dtype | `torch.float16` |
| active tokens | 10720 |
| stage tokens | 64, 32, 16 |
| 데이터 소스 | fallback corpus |
| cache 재로딩 일치 여부 | `true` |

## 3. 실험 구조

Phase 1에서는 두 가지 compression mode를 비교했다.

| mode | 의미 | 역할 |
|---|---|---|
| pooling | average pooling으로 token 수를 줄이는 단순 baseline | 가장 안정적인 하한선 |
| attention | learned query attention으로 token 수를 줄이는 LACE-small 후보 | 학습 가능한 compression path |

전체 흐름은 다음과 같다.

```text
text
→ frozen T5 encoder
→ h0: 128 token
→ compression
→ z1: 64 token
→ z2: 32 token
→ z3: 16 token
→ reverse expander
→ h0_hat
→ MSE / cosine / variance / perturbation 측정
```

## 4. 지표 설명

| 지표 | 쉬운 설명 | 해석 |
|---|---|---|
| train loss | 학습 데이터에서의 복원 손실 | 내려가야 학습 중이라는 뜻 |
| validation loss | 검증 데이터에서의 복원 손실 | 내려가야 일반화 가능성이 있음 |
| MSE | 복원 latent가 원본 latent와 얼마나 다른지 | 낮을수록 좋음 |
| Cosine | 복원 latent와 원본 latent의 방향 유사도 | 높을수록 좋음 |
| Variance ratio | 복원 latent의 분산이 원본 대비 얼마나 남았는지 | 너무 낮으면 collapse 위험 |
| Perturbation sensitivity | latent를 살짝 흔들었을 때 복원 결과가 얼마나 변하는지 | 낮으면 latent를 잘 안 쓸 수 있음 |
| overall gate | Phase 1 기준 통과 여부 | pass면 다음 단계 후보 |

특히 perturbation sensitivity가 중요하다. 모델이 정말 `z_t` latent를 사용한다면 `z_t`를 살짝 바꿨을 때 복원 결과도 어느 정도 바뀌어야 한다. 값이 너무 낮으면 expander가 latent보다 자기 내부 prior에 의존하고 있을 수 있다.

## 5. Pooling baseline 결과

Pooling baseline은 모든 Phase 1 gate를 통과했다.

| 항목 | 값 |
|---|---:|
| train loss | 0.079468 → 0.025681 |
| train loss 감소율 | 약 67.7% |
| validation loss | 0.257082 → 0.023852 |
| validation loss 감소율 | 약 90.7% |
| perturbation sensitivity | 0.012277 |
| overall gate | pass |

단계별 결과:

| Stage | token 수 | MSE | Cosine | Variance ratio | finite |
|---|---:|---:|---:|---:|---|
| `z1` | 64 | 0.005847 | 0.884392 | 0.636817 | true |
| `z2` | 32 | 0.012254 | 0.714241 | 0.452986 | true |
| `z3` | 16 | 0.016237 | 0.593534 | 0.398918 | true |

### 5.1 쉽게 해석하기

Pooling은 단순하지만 안정적이었다. validation loss가 크게 줄었고, `z1 → z2 → z3`로 갈수록 복원이 어려워지는 곡선도 분명하게 나왔다.

```text
MSE:    0.005847 → 0.012254 → 0.016237
Cosine: 0.884392 → 0.714241 → 0.593534
```

이건 좋은 신호다. 압축이 강해질수록 정보가 줄어들고, expander가 그 손실을 어느 정도 복원하려고 학습했다는 뜻이다.

또한 `z3`의 variance ratio가 0.398918로 남아 있다. 완전히 납작한 latent로 collapse된 것은 아니다.

### 5.2 Pooling에서 얻은 결론

Pooling baseline은 Phase 2에서도 보존할 가치가 있다. 지금 단계에서는 LACE의 적이 아니라 기준점이다. learned compression이 이 baseline보다 나아야 LACE 구조가 의미를 갖는다.

## 6. Learned attention compression 결과

Learned attention compression은 loss 감소에는 성공했지만 overall gate는 실패했다.

| 항목 | 값 |
|---|---:|
| train loss | 0.415063 → 0.056589 |
| train loss 감소율 | 약 86.4% |
| validation loss | 1.068796 → 0.054152 |
| validation loss 감소율 | 약 94.9% |
| perturbation sensitivity | 0.001722 |
| overall gate | fail |

단계별 결과:

| Stage | token 수 | MSE | Cosine | Variance ratio | finite |
|---|---:|---:|---:|---:|---|
| `z1` | 64 | 0.015838 | 0.619003 | 0.133218 | true |
| `z2` | 32 | 0.017007 | 0.571143 | 0.112686 | true |
| `z3` | 16 | 0.017687 | 0.544207 | 0.090807 | true |

### 6.1 쉽게 해석하기

Attention compression도 loss는 크게 줄었다. 따라서 "학습이 전혀 안 된다"는 결과는 아니다.

하지만 pooling과 비교하면 복원 품질이 약하다.

| 비교 | Pooling `z3` | Attention `z3` |
|---|---:|---:|
| MSE | 0.016237 | 0.017687 |
| Cosine | 0.593534 | 0.544207 |
| Variance ratio | 0.398918 | 0.090807 |
| Perturbation sensitivity | 0.012277 | 0.001722 |

가장 문제인 값은 perturbation sensitivity다. attention의 sensitivity는 pooling의 약 14% 수준이다. 이는 attention latent를 조금 흔들어도 reconstruction 결과가 크게 변하지 않았다는 뜻이다.

쉽게 말하면 이런 상황일 수 있다.

```text
좋은 상황:
z3가 조금 바뀜 → h0_hat도 의미 있게 바뀜

이번 attention 결과:
z3가 조금 바뀜 → h0_hat이 별로 안 바뀜
```

이 경우 expander가 `z3` 정보를 적극적으로 사용하지 않거나, attention compression이 너무 낮은 분산의 안전한 표현으로 수렴했을 가능성이 있다.

### 6.2 Attention에서 얻은 결론

Learned attention compression은 가능성이 있지만, 현재 구조 그대로는 LACE-small의 핵심 근거로 쓰기 어렵다. 최소한 다음 보완이 필요하다.

- attention latent의 variance를 유지하도록 regularization 강화
- query dropout 추가
- attention entropy regularization 추가
- perturbation sensitivity를 직접 loss 또는 validation metric으로 추적
- pooling baseline과 동일한 조건에서 반복 실행

## 7. Epoch별 학습 흐름

Pooling은 안정적으로 loss가 내려갔다.

| epoch | train loss | validation loss |
|---:|---:|---:|
| 1 | 0.079468 | 0.043254 |
| 2 | 0.036962 | 0.033493 |
| 3 | 0.030454 | 0.028613 |
| 4 | 0.025681 | 0.023852 |

Attention도 loss는 계속 내려갔다.

| epoch | train loss | validation loss |
|---:|---:|---:|
| 1 | 0.415063 | 0.191553 |
| 2 | 0.134715 | 0.094422 |
| 3 | 0.074499 | 0.062108 |
| 4 | 0.056589 | 0.054152 |

두 모드 모두 loss 하향 추세는 분명하다. 따라서 Phase 1에서 가장 먼저 확인하고 싶었던 "reverse expansion 학습 가능성"은 통과했다고 볼 수 있다.

## 8. Gate 판정

| Gate | 의미 | Pooling | Attention | 해석 |
|---|---|---|---|---|
| P1-G1 | validation loss가 충분히 감소하는가 | pass | pass | 둘 다 학습은 된다. |
| P1-G2 | stage가 깊어질수록 복원이 어려워지는가 | pass | pass | 둘 다 `z1 → z2 → z3` 곡선이 있다. |
| P1-G3 | `z3`가 너무 쉬운 shortcut이 아닌가 | pass | pass | 둘 다 `z3`가 `z1`보다 어렵다. |
| P1-G4 | collapse 없이 latent를 실제로 쓰는가 | pass | fail | attention은 perturbation sensitivity가 낮다. |

최종 판정:

| mode | 판정 |
|---|---|
| pooling | Phase 1 smoke pass |
| attention | Phase 1 smoke partial fail |

## 9. 연구적으로 의미 있는 점

이번 실험에서 가장 중요한 점은 다음이다.

> 압축 latent에서 원래 encoder latent를 복원하는 reverse expander는 학습된다.

이것은 LACE 연구의 기본 루프가 완전히 허공은 아니라는 뜻이다. 만약 pooling baseline조차 학습되지 않았다면, 연구 방향을 크게 바꿔야 했을 것이다. 하지만 pooling baseline은 안정적으로 통과했다.

반대로 learned attention compression이 아직 약하다는 점도 중요하다. 이 결과는 LACE 가설을 부정한다기보다, 현재 attention compression 설계가 아직 부족하다는 신호에 가깝다.

현재까지의 정확한 결론은 다음이다.

> LACE의 reverse expansion loop는 학습 가능하다. 그러나 learned compression path가 단순 pooling보다 낫다는 증거는 아직 없다.

## 10. 한계

이번 Phase 1 결과에는 큰 한계가 있다.

1. 실제 데이터셋이 아니라 fallback corpus 512개를 사용했다.
2. text generation renderer는 없다.
3. semantic similarity, BERTScore, NLI 같은 의미 지표는 없다.
4. Gaussian noise baseline과 mask baseline은 아직 비교하지 않았다.
5. learned attention compression은 perturbation sensitivity gate를 통과하지 못했다.
6. seed 1개만 실행했다.

따라서 이 결과는 논문 실험 결과가 아니라 **실험 루프 검증 결과**로 보아야 한다.

## 11. 다음 실험 제안

### 11.1 같은 Phase 1을 실제 데이터로 재실행

가장 먼저 해야 할 일은 fallback corpus를 벗어나는 것이다.

우선순위:

1. WikiText-2 subset
2. XSum subset
3. CNN/DailyMail subset

실제 데이터에서 pooling baseline이 여전히 통과하는지 확인해야 한다.

### 11.2 Attention compression 개선

현재 attention compression은 loss는 줄지만 latent 사용 신호가 약하다. 다음 수정이 필요하다.

- query dropout
- attention entropy regularization
- variance preservation loss 강화
- stage별 attention query 분리
- expander capacity 축소
- perturbation sensitivity를 validation monitor로 사용

### 11.3 Phase 2로 넘길 기준

Phase 2로 넘어가기 전에 attention compression이 최소한 다음 조건을 만족해야 한다.

- validation loss가 감소한다.
- `z1/z2/z3` 곡선이 유지된다.
- `z3` variance ratio가 너무 낮지 않다.
- perturbation sensitivity가 pooling baseline과 같은 order에 들어온다.
- 실제 데이터셋에서도 반복된다.

## 12. 현재 보류해야 할 주장

아직 다음 주장은 하면 안 된다.

- LACE가 Gaussian diffusion보다 낫다.
- LACE가 random masking보다 낫다.
- learned attention compression이 pooling보다 낫다.
- semantic information을 더 잘 보존한다.
- 텍스트 생성 품질이 좋아진다.
- information rate를 정확히 측정했다.

현재 말할 수 있는 것은 다음뿐이다.

> Phase 1 smoke run에서 reverse expander 학습 가능성은 확인됐다. Pooling baseline은 안정적으로 통과했다. Learned attention compression은 학습되지만 latent 사용 신호가 약해 구조 개선이 필요하다.
