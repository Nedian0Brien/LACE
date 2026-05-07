# S4e 계획: shared-condition semantic span expansion

## 목적

S4e는 S4d의 긍정 신호를 더 엄격하고 비용 효율적인 구조로 재검증한다. S4d에서는 각 schedule마다 별도 reverse model을 학습했기 때문에, `importance_schedule`의 우위가 semantic skeleton 자체 때문인지 아니면 조건별 모델 분리의 우연한 학습 차이 때문인지 완전히 분리되지 않았다.

S4e의 핵심 질문은 다음이다.

> 하나의 공유 reverse model 안에서도 semantic skeleton condition이 random, position-only, same-position random, wrong-document, no-anchor control보다 더 좋은 span expansion과 rollout을 만드는가?

이 실험은 [../research-questions.md](../research-questions.md)의 v2 핵심 질문인 “의미 골격 + 위치 보조 구조가 무작위 손상보다 더 나은 역방향 궤적을 만드는가”를 S4 phase에서 검증한다. 또한 [../experiment-roadmap.md](../experiment-roadmap.md)의 S4 구조 개선 단계에 해당하며, S5 open-ended generation scale-up으로 넘어가기 전에 span 단위 정보 전달 병목을 확인한다.

## 방법

Forward schedule과 gap/span 구성은 S4d를 유지한다. 달라지는 점은 reverse model이다.

S4d:

```text
schedule마다 별도 reverse model 학습
```

S4e:

```text
모든 schedule 예제를 하나로 합쳐 하나의 reverse model 학습
각 예제에는 condition embedding + gap descriptor 제공
```

입력 구조는 다음과 같다.

```text
current skeleton tokens + positions
left/right semantic anchor role
new span marker positions
timestep
condition id
gap length
left/right anchor distance
```

출력 구조는 S4d와 동일하다.

```text
newly unmasked span token ids
```

학습 objective는 handcrafted content/entity loss가 아니라 일반적인 span token cross entropy다. Content/entity 지표는 결과 해석과 gate에만 사용한다.

## 조건

| 조건 | 의미 |
|---|---|
| `importance_schedule` | attention-received score가 높은 token을 semantic skeleton으로 남기고 실제 token content와 left/right anchor를 제공한다. |
| `random_schedule` | 같은 ratio에서 random token skeleton으로 rollout한다. |
| `position_only_schedule` | importance 위치와 gap 구조는 주되 token content를 제거한다. |
| `same_position_random_schedule` | importance와 같은 위치에 같은 문서의 random token content를 넣는다. |
| `wrong_document_same_position_schedule` | importance와 같은 위치에 다른 문서의 skeleton token content를 넣는다. |
| `no_anchor_gap_only_schedule` | 현재 skeleton과 anchor 없이 이번 span marker 위치만 준다. |

## Gate

| Gate | 통과 조건 | 해석 |
|---|---|---|
| `S4E-G-RUN` | 모든 schedule의 span 평가와 rollout 평가가 완료된다. | 실험 완결성 |
| `S4E-G-LOSS-FINITE` | 공유 모델의 모든 조건 평가 loss가 유한하다. | 수치 안정성 |
| `S4E-G-SHARED-CONDITION-MODEL` | 하나의 모델이 condition id로 모든 조건을 처리한다. | 조건별 모델 분리 confound 제거 |
| `S4E-G-IMPORTANCE-BEATS-RANDOM` | importance rollout score가 random보다 높다. | forward schedule 우위 유지 |
| `S4E-G-IMPORTANCE-BEATS-POSITION-ONLY` | importance가 position-only보다 높다. | 위치 보조 구조만으로는 부족하다는 증거 |
| `S4E-G-IMPORTANCE-BEATS-SAME-POSITION-RANDOM` | importance가 같은 위치 random content보다 높다. | 같은 위치라도 실제 의미 content가 필요하다는 증거 |
| `S4E-G-WRONG-DOC-DROPS` | wrong-document content가 importance보다 낮다. | 아무 의미 content나 넣은 효과가 아님 |
| `S4E-G-NO-ANCHOR-DROPS` | no-anchor gap-only가 importance보다 낮다. | 좌우 anchor가 span expansion에 필요함 |
| `S4E-G-SPAN-CONTENT-GAIN` | importance span content recall이 S4d baseline보다 오른다. | 새로 생성되는 span 자체의 정보량 개선 |
| `S4E-G-SPAN-ENTITY-GAIN` | importance span entity recall이 S4d baseline보다 오른다. | entity 정보 전달 개선 |
| `S4E-G-FINAL-NONREGRESSION` | final content/entity, drift, repetition이 S4d 대비 크게 나빠지지 않는다. | span 개선이 rollout 품질을 희생하지 않는지 확인 |
| `S4E-G-ARTIFACT-CHECK` | importance artifact rate가 random보다 높아지지 않는다. | content recall 상승이 비정상 생성물 때문인지 방지 |

## 해석 기준

좋은 결과는 `importance_schedule`이 공유 모델 안에서도 random과 모든 strict control을 이기고, 동시에 S4d보다 span content/entity recall을 끌어올리는 것이다. 이 경우 S4e는 “semantic skeleton을 중심으로 한 reverse expansion 구조가 모델 분리 없이도 일반화된다”는 해석을 뒷받침한다.

나쁜 결과는 importance 우위가 사라지거나, span content/entity가 S4d보다 오르지 않는 경우다. 이 경우 S4d의 signal은 여전히 유효할 수 있지만, 현재 shared-condition 구조만으로는 의미 있는 새 span 생성 능력을 충분히 키우지 못한다. 다음 단계는 단순 scale-up보다 decoder 입력 구조, span ordering, anchor 사용 방식, 또는 schedule curriculum을 더 구조적으로 개선하는 쪽이 된다.

S4e는 S5로 가기 위한 scale-up readiness 실험이다. 여기서 확인하려는 것은 문장을 자유롭게 잘 생성하는가가 아니라, 의미 골격에서 세부 span으로 확장되는 reverse process가 같은 모델 안에서도 방어 가능한 우위를 보이는가다.
