# Phase 1 실험 결과: WikiText-2 Subset 재실행

## 1. 실험 목적

이 실험은 Phase 1 smoke run을 실제 데이터셋에 더 가까운 조건으로 다시 실행한 것이다. 이전 Phase 1은 fallback corpus 512개 반복 기반이었다. 그 결과 pooling baseline은 안정적으로 통과했지만, learned attention compression은 latent 사용 신호가 약했다.

이번 실험의 목적은 다음이다.

1. fallback corpus가 아니라 WikiText-2 subset에서 reverse expander가 여전히 학습되는지 확인한다.
2. pooling baseline이 실제 텍스트 분포에서도 stage별 난도 곡선을 유지하는지 확인한다.
3. learned attention compression이 실제 데이터에서 더 나아지는지, 아니면 약점이 더 분명해지는지 확인한다.

## 2. 실행 정보

| 항목 | 값 |
|---|---|
| Kaggle kernel | `dennisparknd/lace-phase-1-latent-compression` |
| kernel version | 2 |
| 실행 상태 | 완료 |
| 장치 | `cuda` |
| 모델 | `t5-small` |
| encoder 상태 | frozen |
| 데이터 소스 | `hf:wikitext/wikitext-2-raw-v1:train` |
| 샘플 수 | 512 |
| 최대 길이 | 128 token |
| hidden shape | `[512, 128, 512]` |
| hidden dtype | `torch.float16` |
| active tokens | 49201 |
| stage tokens | 64, 32, 16 |
| cache 재로딩 일치 여부 | `true` |

## 3. 이전 fallback run과 달라진 점

가장 큰 차이는 active token 수다.

| run | 데이터 | 샘플 수 | active tokens |
|---|---|---:|---:|
| Phase 1 smoke | fallback corpus | 512 | 10720 |
| Phase 1 WikiText-2 | WikiText-2 subset | 512 | 49201 |

WikiText-2는 같은 512개 샘플이어도 실제 token이 훨씬 많다. 즉, 더 긴 문장과 더 다양한 문맥이 들어왔고, latent reconstruction 문제가 더 어려워졌다. 이 때문에 fallback run보다 결과를 더 신뢰할 수 있지만, 동시에 더 엄격한 조건이다.

## 4. Pooling baseline 결과

Pooling baseline은 학습 자체에는 성공했지만, 전체 gate는 실패했다.

| 항목 | 값 |
|---|---:|
| train loss | 0.144267 → 0.065825 |
| train loss 감소율 | 약 54.4% |
| validation loss | 0.305443 → 0.069064 |
| validation loss 감소율 | 약 77.4% |
| perturbation sensitivity | 0.002050 |
| overall gate | fail |

단계별 결과:

| Stage | token 수 | MSE | Cosine | Variance ratio | finite |
|---|---:|---:|---:|---:|---|
| `z1` | 64 | 0.010499 | 0.805115 | 0.509536 | true |
| `z2` | 32 | 0.016689 | 0.659220 | 0.299285 | true |
| `z3` | 16 | 0.020779 | 0.544056 | 0.182608 | true |

## 5. Pooling 결과 해석

Pooling은 여전히 의미 있는 학습 신호를 보인다.

```text
MSE:    0.010499 → 0.016689 → 0.020779
Cosine: 0.805115 → 0.659220 → 0.544056
```

압축이 강해질수록 MSE는 증가하고 cosine은 감소한다. 즉, `z1 → z2 → z3` stage 곡선은 유지됐다. validation loss도 크게 줄었다.

하지만 fallback run과 달리 overall gate는 실패했다. 이유는 perturbation sensitivity가 낮기 때문이다.

| run | pooling perturbation sensitivity |
|---|---:|
| fallback | 0.012277 |
| WikiText-2 | 0.002050 |

이 값은 "latent를 살짝 흔들었을 때 reconstruction이 얼마나 변하는가"를 본다. WikiText-2에서는 pooling latent를 흔들어도 복원 결과가 fallback 때만큼 민감하게 변하지 않았다. 이는 다음 중 하나일 수 있다.

1. WikiText-2 latent 분포에서 perturbation scale `0.05`가 충분히 크지 않다.
2. expander가 세부 latent 변화보다 평균적 복원 패턴에 더 의존한다.
3. sensitivity 기준이 절대값 기반이라 데이터 분포가 바뀌면 너무 엄격해진다.
4. 실제 데이터에서는 현재 expander capacity 또는 loss가 latent 사용을 충분히 강제하지 못한다.

중요한 점은 pooling이 완전히 실패한 것은 아니라는 것이다. 학습도 되고 stage 곡선도 나온다. 다만 **latent 사용 gate를 더 정교하게 설계해야 한다**는 신호가 나왔다.

## 6. Learned attention compression 결과

Attention compression은 이번에도 전체 gate를 통과하지 못했다. fallback run보다 약점이 더 뚜렷하다.

| 항목 | 값 |
|---|---:|
| train loss | 0.475195 → 0.196823 |
| train loss 감소율 | 약 58.6% |
| validation loss | 1.056442 → 0.197848 |
| validation loss 감소율 | 약 81.3% |
| perturbation sensitivity | 0.001456 |
| overall gate | fail |

단계별 결과:

| Stage | token 수 | MSE | Cosine | Variance ratio | finite |
|---|---:|---:|---:|---:|---|
| `z1` | 64 | 0.120532 | 0.236924 | 3.710784 | true |
| `z2` | 32 | 0.122075 | 0.236655 | 3.759628 | true |
| `z3` | 16 | 0.123256 | 0.236275 | 3.797502 | true |

## 7. Attention 결과 해석

Attention compression은 validation loss가 줄었으므로 학습 자체는 진행된다. 그러나 복원 품질은 pooling보다 훨씬 나쁘다.

| 비교 | Pooling `z3` | Attention `z3` |
|---|---:|---:|
| MSE | 0.020779 | 0.123256 |
| Cosine | 0.544056 | 0.236275 |
| Variance ratio | 0.182608 | 3.797502 |
| Perturbation sensitivity | 0.002050 | 0.001456 |

특히 세 가지 문제가 보인다.

### 7.1 MSE가 너무 높다

Attention `z3`의 MSE는 0.123256으로 pooling `z3`의 0.020779보다 훨씬 높다. 현재 attention compression은 `h0` 복원에 필요한 정보를 잘 정리하지 못한다.

### 7.2 Cosine이 낮고 stage 간 차이가 거의 없다

Attention의 cosine은 `z1`, `z2`, `z3` 모두 0.236 근처다.

```text
z1 cosine: 0.236924
z2 cosine: 0.236655
z3 cosine: 0.236275
```

`z1`은 64 token, `z3`는 16 token이므로 둘 사이에는 난도 차이가 나야 한다. 그런데 거의 비슷하다. 이는 attention compression이 stage별로 의미 있는 정보율 차이를 만들지 못하고 있다는 신호다.

### 7.3 Variance ratio가 비정상적으로 높다

Attention의 variance ratio는 3.7~3.8 수준이다. fallback run에서는 attention variance ratio가 낮아서 collapse 의심이 있었는데, WikiText-2에서는 반대로 너무 높다.

이는 attention latent가 원본 `h0`의 분포와 다른 scale로 튀고 있을 가능성을 보여준다. 즉, 단순 collapse라기보다 **불안정한 latent scale 문제**에 가깝다.

## 8. Gate 판정

| Gate | 의미 | Pooling | Attention | 해석 |
|---|---|---|---|---|
| P1-G1 | validation loss가 충분히 감소하는가 | pass | pass | 둘 다 학습 loss는 줄어든다. |
| P1-G2 | stage가 깊어질수록 MSE 증가, cosine 감소가 나타나는가 | pass | pass | attention은 통과했지만 차이가 매우 작다. |
| P1-G3 | `z3`가 `z1`보다 명확히 어려운 병목인가 | pass | fail | attention은 stage 간 난도 차이가 약하다. |
| P1-G4 | collapse 없이 latent를 실제로 쓰는가 | fail | fail | 두 모드 모두 perturbation sensitivity가 낮다. |

최종 판정:

| mode | 판정 |
|---|---|
| pooling | 학습 가능, stage 곡선 유지, sensitivity gate 실패 |
| attention | 학습은 되지만 현재 구조로는 부적합 |

## 9. 중요한 발견

이번 WikiText-2 run에서 가장 중요한 발견은 다음이다.

> fallback corpus에서는 pooling baseline이 완전히 통과했지만, 실제 WikiText-2에서는 sensitivity gate가 실패했다.

이는 실험이 더 현실적인 데이터로 넘어가면서 더 엄격해졌다는 뜻이다. 단순한 반복 fallback corpus에서는 expander가 쉽게 학습했지만, WikiText-2에서는 latent를 적극적으로 사용하도록 강제하는 장치가 더 필요하다.

두 번째 발견은 attention compression의 구조적 문제다.

현재 loss는 stage-wise pair reconstruction을 중심으로 한다. Attention compression에서는 `z2`, `z3`가 `h0`에 직접 anchor되지 않고, expander와 함께 co-adaptation할 수 있다. 이 때문에 attention latent가 원본 latent 분포와 동떨어진 scale로 수렴하거나, stage별 정보율 차이를 제대로 만들지 못할 가능성이 있다.

따라서 다음 실험에서는 attention latent를 더 직접적으로 anchor해야 한다.

## 10. 다음 수정 방향

### 10.1 Pooled-latent anchor loss 추가

Attention `z_t`가 완전히 자유롭게 움직이지 않도록, 같은 token 수의 pooling latent에 가까워지는 보조 loss를 추가한다.

```text
L_anchor = MSE(z_t_attention, stopgrad(pool(h0, N_t)))
```

이 loss는 attention compression이 최소한 안정적인 pooling baseline 주변에서 출발하도록 만든다.

### 10.2 Direct-to-h0 reconstruction loss 추가

현재 stage-wise 구조는 다음을 학습한다.

```text
z3 → z2
z2 → z1
z1 → h0
```

다음 loss를 추가하면 `z2`, `z3`도 `h0` 복원 신호를 직접 받는다.

```text
z2 → h0
z3 → h0
```

이렇게 하면 deeper stage가 `h0`와 너무 멀어지는 문제를 줄일 수 있다.

### 10.3 Latent scale regularization 추가

Attention variance ratio가 3.7 이상으로 높게 나온 것은 scale 불안정 신호다. 다음 regularization이 필요하다.

```text
mean(z_t) ≈ mean(pool(h0, N_t))
var(z_t) ≈ var(pool(h0, N_t))
```

### 10.4 Perturbation sensitivity 재설계

현재 sensitivity는 절대 MSE 변화량이다. 데이터가 fallback에서 WikiText-2로 바뀌면서 값의 scale도 달라졌다. 다음처럼 상대 지표를 함께 기록해야 한다.

```text
relative_sensitivity = perturbation_delta / reconstruction_mse
```

절대 sensitivity는 유지하되, gate 판단에는 relative sensitivity를 함께 쓰는 편이 더 안전하다.

## 11. 현재 결론

이번 결과는 실패라기보다 좋은 진단이다. 더 현실적인 WikiText-2에서 다음 사실을 확인했다.

1. `t5-small` latent cache는 실제 dataset에서도 정상 생성된다.
2. pooling baseline은 여전히 학습되고 stage 곡선도 유지된다.
3. 하지만 sensitivity 기준에서는 pooling도 아직 충분하지 않다.
4. learned attention compression은 현재 설계로는 pooling보다 훨씬 약하다.
5. attention latent에는 anchor, scale control, direct reconstruction signal이 필요하다.

현재 결론은 다음과 같다.

> Phase 1은 실제 데이터에서 더 어려워졌고, learned attention compression의 구조 개선 없이는 Phase 2로 넘어가면 안 된다.

## 12. 다음 단계

바로 Phase 2로 가지 말고, Phase 1.5를 진행하는 것이 좋다.

Phase 1.5의 목표:

- attention compression에 pooled-latent anchor loss 추가
- direct-to-h0 loss 추가
- latent scale regularization 추가
- relative perturbation sensitivity 추가
- WikiText-2 subset으로 재실행

Phase 1.5에서 attention compression이 pooling과 같은 order의 MSE/cosine/sensitivity를 보이면, 그때 Gaussian/mask baseline과 비교하는 Phase 2로 넘어간다.
