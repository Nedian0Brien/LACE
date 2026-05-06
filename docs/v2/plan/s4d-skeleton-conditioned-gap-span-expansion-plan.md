# S4d 계획: skeleton-conditioned gap/span expansion

## 목적

S4d는 S4b의 고무적인 신호를 더 엄격하게 검증한다. S4b는 importance 기반 semantic skeleton에서 시작한 multi-step rollout이 random과 position-only보다 좋다는 결과를 보였지만, 아직 “현재 남아 있는 skeleton content가 다음 span 생성에 실제로 쓰였는가”는 충분히 분리하지 못했다.

S4d의 핵심 질문은 다음이다.

> 같은 gap/span 위치 구조에서, 현재 semantic skeleton의 left/right anchor content가 다음 span 생성과 rollout 품질을 position-only, same-position random, wrong-document, no-anchor control보다 개선하는가?

이 실험은 handcrafted content/entity loss를 쓰지 않는다. 학습 objective는 일반적인 span token cross entropy이고, content/entity 지표는 사후 분석용으로만 사용한다.

## 방법

Forward schedule은 기존 importance-ordered masking을 유지한다. Reverse model은 전체 target state를 생성하지 않고, 각 transition에서 새로 unmask되는 contiguous span만 생성한다.

입력 구조는 다음과 같다.

```text
current skeleton tokens + positions
left/right semantic anchor role
new span marker positions
timestep
```

출력 구조는 다음과 같다.

```text
newly unmasked span token ids
```

평가에서는 생성된 span을 current state에 삽입하고, `25% -> 50% -> 75% -> 100%` reverse rollout을 수행한다.

## 조건

| 조건 | 의미 |
|---|---|
| `importance_schedule` | attention-received score가 높은 token을 skeleton으로 남기고, 실제 token content와 left/right anchor를 제공한다. |
| `random_schedule` | 같은 ratio에서 random token skeleton으로 rollout한다. |
| `position_only_schedule` | importance 위치와 gap 구조는 주되 token content를 제거한다. |
| `same_position_random_schedule` | importance와 같은 위치에 같은 문서의 random token content를 넣는다. |
| `wrong_document_same_position_schedule` | importance와 같은 위치에 다른 문서의 skeleton token content를 넣는다. |
| `no_anchor_gap_only_schedule` | 현재 skeleton과 anchor 없이 이번 span marker 위치만 준다. |

## Gate

| Gate | 통과 조건 | 해석 |
|---|---|---|
| `S4D-G-RUN` | 모든 schedule이 학습, span 평가, rollout 평가를 완료한다. | 실험 완결성 |
| `S4D-G-LOSS-FINITE` | 모든 조건의 loss가 유한하다. | 수치 안정성 |
| `S4D-G-IMPORTANCE-BEATS-RANDOM` | importance rollout score가 random보다 높다. | forward schedule 우위 |
| `S4D-G-IMPORTANCE-BEATS-POSITION-ONLY` | importance가 position-only보다 높다. | token content 사용 증거 |
| `S4D-G-IMPORTANCE-BEATS-SAME-POSITION-RANDOM` | importance가 같은 위치 random content보다 높다. | 위치가 아니라 의미 content가 도움을 준다는 증거 |
| `S4D-G-WRONG-DOC-DROPS` | wrong-document content가 importance보다 낮다. | 아무 content나 붙인 효과가 아님 |
| `S4D-G-NO-ANCHOR-DROPS` | no-anchor gap-only가 importance보다 낮다. | left/right anchor가 local expansion에 필요함 |
| `S4D-G-SPAN-CONTENT-NONZERO` | span content recall과 final content recall이 0보다 크다. | S4c의 content/entity 0 붕괴 탈출 |
| `S4D-G-ROLLOUT-NONREGRESSION` | importance drift/repetition이 random보다 크게 나쁘지 않다. | rollout 누적 안정성 |

## 해석 기준

좋은 결과는 `importance_schedule`이 random뿐 아니라 same-position random, wrong-document, no-anchor control까지 이기는 것이다. 이 경우 S4d는 “semantic skeleton이 단순 위치 scaffold가 아니라 gap/span 생성에 쓰이는 정보 구조”라는 해석을 방어할 수 있다.

나쁜 결과는 position-only나 no-anchor가 importance와 비슷하거나 더 좋은 경우다. 이 경우 현재 구조는 여전히 위치 prior와 decoder language prior에 의존하고 있으며, left/right anchor content를 더 강하게 쓰는 구조 개선이 필요하다.

S4d는 open-ended language model 완성도를 입증하는 실험이 아니다. 여기서 확인하려는 것은 reverse process의 한 단위가 “중심 의미 뼈대에서 세부 span으로 확장한다”는 방향으로 작동하는지다.
