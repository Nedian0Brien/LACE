# Phase 0 실험 결과: Latent Cache 및 Pooling Sanity Check

## 1. 실험 목적

Phase 0의 목적은 LACE 연구의 가장 작은 실행 단위를 검증하는 것이다. 이 단계에서는 모델을 학습하지 않는다. 대신 다음 질문을 확인한다.

1. Kaggle GPU에서 `t5-small` encoder를 실행할 수 있는가?
2. 입력 텍스트를 고정 길이 latent `h0`로 안정적으로 변환할 수 있는가?
3. `h0`를 단계적으로 줄인 `z1/z2/z3`를 만들 수 있는가?
4. latent cache를 저장하고 다시 읽어도 값이 보존되는가?
5. token budget을 줄일수록 복원 metric이 자연스럽게 나빠지는가?

이 실험은 LACE 가설을 검증하는 실험이 아니라, 이후 실험이 돌아갈 수 있는지 확인하는 **파이프라인 검증**이다.

## 2. 실행 정보

| 항목 | 값 |
|---|---|
| Kaggle kernel | `dennisparknd/lace-phase-0-latent-cache` |
| kernel version | 2 |
| 실행 상태 | 완료 |
| 장치 | `cuda` |
| 모델 | `t5-small` |
| encoder 상태 | frozen |
| 샘플 수 | 128 |
| 최대 길이 | 128 token |
| hidden dtype | `torch.float16` |
| active tokens | 2661 |
| cache 재로딩 일치 여부 | `true` |
| 데이터 소스 | fallback text corpus |

## 3. 실험 구조

실험 흐름은 다음과 같다.

```text
text
→ frozen T5 encoder
→ h0: [128, 128, 512]
→ average pooling
→ z1: [128, 64, 512]
→ z2: [128, 32, 512]
→ z3: [128, 16, 512]
→ linear interpolation으로 h0 길이에 맞춰 복원
→ MSE / cosine 측정
```

여기서 `h0`는 원본 텍스트를 encoder가 변환한 token-level latent다. `z1`, `z2`, `z3`는 이 latent의 token 수를 줄인 압축 표현이다.

## 4. 결과

| 단계 | token 수 | shape | MSE | Cosine | finite |
|---|---:|---|---:|---:|---|
| `z1` | 64 | `[128, 64, 512]` | 0.008886 | 0.787159 | true |
| `z2` | 32 | `[128, 32, 512]` | 0.013866 | 0.642822 | true |
| `z3` | 16 | `[128, 16, 512]` | 0.017505 | 0.522215 | true |

## 5. 쉬운 해석

이 결과는 예상과 맞다.

`z1`은 `h0`의 token 수를 절반으로 줄인 표현이다. 아직 많은 정보를 보존하므로 MSE가 가장 낮고 cosine이 가장 높다.

`z2`는 token 수를 4분의 1로 줄인 표현이다. `z1`보다 정보가 더 많이 사라져 MSE가 올라가고 cosine이 내려간다.

`z3`는 token 수를 8분의 1로 줄인 표현이다. 가장 강한 압축이므로 복원이 가장 어렵다.

즉, 다음과 같은 자연스러운 곡선이 나왔다.

```text
압축 강도 증가: h0 → z1 → z2 → z3
MSE 증가:      0.008886 → 0.013866 → 0.017505
Cosine 감소:   0.787159 → 0.642822 → 0.522215
```

이 곡선은 "latent token budget을 줄이면 정보가 점진적으로 줄어든다"는 매우 기본적인 사실을 실험적으로 확인한 것이다.

## 6. 성공 판정

| 기준 | 결과 | 판정 |
|---|---|---|
| Kaggle GPU에서 실행되는가 | `cuda` 사용 | pass |
| encoder latent shape가 안정적인가 | `[128, 128, 512]` | pass |
| 단계별 latent shape가 의도대로 줄어드는가 | 64, 32, 16 token | pass |
| cache 재로딩 후 값이 보존되는가 | `cache_allclose = true` | pass |
| 압축이 강해질수록 MSE가 증가하는가 | 단조 증가 | pass |
| 압축이 강해질수록 cosine이 감소하는가 | 단조 감소 | pass |

Phase 0은 통과했다.

## 7. 연구적으로 의미 있는 점

Phase 0에서 가장 중요한 결과는 `cache_allclose = true`다. 이는 `h0` latent를 저장해두고 이후 실험에서 다시 쓸 수 있다는 뜻이다. 개인 연구자 환경에서는 encoder를 매번 다시 돌리는 비용이 부담되므로, latent cache가 안정적으로 동작하는 것은 이후 Phase 1, Phase 2의 기반이 된다.

또한 단계별 MSE/cosine 곡선이 깨지지 않았기 때문에, `z1/z2/z3`라는 compression stage 자체는 실험 대상으로 사용할 수 있다.

## 8. 한계

이 실험에는 중요한 한계가 있다.

1. 학습된 compression이 아니다. 단순 average pooling이다.
2. reverse expander가 없다. interpolation으로 길이만 다시 맞췄다.
3. 실제 WikiText-2/XSum이 아니라 fallback corpus를 사용했다.
4. 의미 보존 metric은 없다.
5. 텍스트 생성 결과는 없다.

따라서 Phase 0으로 주장할 수 있는 것은 제한적이다.

> Kaggle에서 latent cache와 기본 compression metric pipeline이 정상 동작한다.

이 정도까지만 주장할 수 있다.

## 9. 다음 단계로 넘긴 것

Phase 0은 Phase 1에 다음 기반을 넘겼다.

- 고정 `t5-small` encoder 사용
- max length 128
- stage tokens 64, 32, 16
- MSE/cosine 기반 reconstruction metric
- latent cache 저장/재로딩 절차

Phase 1은 이 기반 위에서 실제 reverse expander를 학습한다.
