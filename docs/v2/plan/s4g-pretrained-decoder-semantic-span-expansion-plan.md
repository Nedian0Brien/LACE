# S4g 계획: pretrained decoder semantic span expansion

## 목적

S4g는 S4e 실패 원인을 분리하기 위한 구조 실험이다. S4e는 shared-condition model에서도 `importance_schedule`이 strict control을 모두 이겼지만, generated span 자체의 content/entity recall은 S4d보다 낮고 artifact rate가 높았다. 사용자는 이 실패가 모델 규모와 언어 생성 능력 부족 때문일 수 있다고 지적했다.

S4g의 핵심 질문은 다음이다.

> S4d/S4e의 semantic skeleton 우위를 유지한 채, pretrained decoder의 언어 prior를 사용하면 generated span artifact가 줄고 content/entity recall이 개선되는가?

이 실험은 [../research-questions.md](../research-questions.md)의 v2 핵심 질문인 “의미 골격 + 위치 보조 구조가 무작위 손상보다 더 나은 역방향 궤적을 만드는가”를 S4 구조 개선 단계에서 검증한다. [../experiment-roadmap.md](../experiment-roadmap.md) 기준으로는 S5 open-ended generation 이전의 pretrained decoder bridge다.

## 방법

S4d/S4e의 forward schedule, gap/span 구성, six-control 비교는 유지한다. 달라지는 점은 reverse decoder다.

S4e:

```text
custom small encoder-decoder
token id span target
```

S4g:

```text
pretrained AutoModelForSeq2SeqLM
text-to-text span target
```

입력은 text prompt로 직렬화한다.

```text
condition id
transition
span positions
span length
left/right anchor position and distance
current skeleton text
```

출력은 newly unmasked span text다. Rollout에서는 생성 span text를 tokenizer로 다시 token id로 바꿔 current state에 삽입한다.

학습 objective는 여전히 일반적인 seq2seq cross entropy다. Content/entity 지표는 결과 해석과 gate에만 사용한다.

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
| `S4G-G-RUN` | 모든 schedule의 span 평가와 rollout 평가가 완료된다. | 실험 완결성 |
| `S4G-G-LOSS-FINITE` | 모든 조건 평가 loss가 유한하다. | 수치 안정성 |
| `S4G-G-PRETRAINED-DECODER` | pretrained seq2seq decoder가 사용된다. | S4e와의 구조 차이 |
| `S4G-G-IMPORTANCE-BEATS-RANDOM` | importance rollout score가 random보다 높다. | forward schedule 우위 유지 |
| `S4G-G-IMPORTANCE-BEATS-POSITION-ONLY` | importance가 position-only보다 높다. | 위치만으로 설명되지 않음 |
| `S4G-G-IMPORTANCE-BEATS-SAME-POSITION-RANDOM` | importance가 같은 위치 random content보다 높다. | 같은 위치라도 실제 의미 content가 필요함 |
| `S4G-G-WRONG-DOC-DROPS` | wrong-document content가 importance보다 낮다. | 아무 content나 넣은 효과가 아님 |
| `S4G-G-NO-ANCHOR-DROPS` | no-anchor gap-only가 importance보다 낮다. | pretrained LM prior만으로 해결되지 않음 |
| `S4G-G-SPAN-CONTENT-GAIN` | importance span content recall이 S4e baseline보다 오른다. | S4e의 generated-span collapse 개선 |
| `S4G-G-SPAN-ENTITY-GAIN` | importance span entity recall이 S4e baseline보다 오른다. | entity 정보 생성 개선 |
| `S4G-G-FINAL-NONREGRESSION` | final content/entity, drift, repetition이 S4e 대비 크게 나빠지지 않는다. | span 개선이 rollout 품질을 희생하지 않는지 확인 |
| `S4G-G-ARTIFACT-CHECK` | importance artifact rate가 S4e보다 내려가고 random보다 높지 않다. | pretrained decoder가 artifact를 줄였는지 확인 |

## 해석 기준

좋은 결과는 `importance_schedule`이 strict control을 계속 이기면서, S4e보다 span content/entity recall이 오르고 artifact rate가 내려가는 것이다. 이 경우 S4e 실패는 상당 부분 작은 custom decoder의 언어 prior 부족 때문이었다는 해석이 가능하다.

나쁜 결과는 pretrained decoder를 사용해도 artifact가 높거나 no-anchor/random control이 함께 좋아지는 경우다. 이 경우 모델 규모만 키우는 것은 충분하지 않으며, span target 구성, content/function token 분리, anchor cross-attention, chunk 단위 target 같은 구조 개선이 우선이다.

S4g는 open-ended generation이 아니다. 위치가 주어진 constrained span expansion이고, S5로 넘어갈지 판단하기 위한 pretrained decoder bridge다.
