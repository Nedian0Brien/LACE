# S4e 결과: shared-condition semantic span expansion

## 1. 이 실험이 측정한 것

S4e는 S4d의 결과를 더 엄격하게 재검증하기 위해 실행했다. S4d는 `importance_schedule`이 random, position-only, same-position random, wrong-document, no-anchor control을 모두 이겼지만, schedule마다 별도 reverse model을 학습했다. 따라서 남는 반론은 다음이었다.

```text
importance가 좋은 것이 semantic skeleton 때문인가?
아니면 조건별로 다른 모델을 학습했기 때문에 생긴 차이인가?
```

S4e는 이 반론을 없애기 위해 여섯 schedule의 학습 예제를 하나로 합쳐 **하나의 공유 reverse model**을 학습했다. 각 예제에는 `condition_id`, `gap_length`, `left_anchor_distance`, `right_anchor_distance`를 추가했다.

입력과 target은 다음이다.

```text
input:  current skeleton tokens + positions
        left/right semantic anchor role
        newly opened span marker positions
        timestep
        condition id
        gap descriptor

target: newly unmasked span token ids
```

실행 정보는 다음이다.

| 항목 | 값 |
|---|---|
| phase | `v2_s4e` |
| Kaggle kernel | `dennisparknd/lace-v2-s4e-shared-condition-span` version 1 |
| model | `t5-small` tokenizer + custom PyTorch shared encoder-decoder |
| data | `wikitext/wikitext-2-raw-v1:train` |
| train samples | 768 |
| eval samples | 192 |
| combined train examples | 277,999 |
| device | `cuda` |
| ratios | `0.25 -> 0.50 -> 0.75 -> 1.00` |
| reverse epochs | 2 |
| runtime | 약 2,241초 |
| output | `outputs/v2_s4e/lace_v2_s4e/` |

비교 조건은 S4d와 동일하다.

| 조건 | 의미 |
|---|---|
| `importance_schedule` | 실제 importance skeleton token content와 left/right anchor를 사용한다. |
| `random_schedule` | 같은 ratio와 token budget에서 random skeleton을 사용한다. |
| `position_only_schedule` | importance 위치와 gap 구조는 주되 token content를 제거한다. |
| `same_position_random_schedule` | importance와 같은 위치에 같은 문서의 random token content를 넣는다. |
| `wrong_document_same_position_schedule` | importance와 같은 위치에 다른 문서의 skeleton token content를 넣는다. |
| `no_anchor_gap_only_schedule` | current skeleton 없이 이번 span marker 위치만 준다. |

## 2. 왜 중요한가

S4e의 중요성은 두 가지다.

첫째, S4d의 모델 분리 confound를 제거한다. 한 모델 안에서 모든 condition을 처리하게 만들면, `importance_schedule`의 우위를 “모델별 우연한 학습 차이”로 설명하기 어려워진다.

둘째, S5 scale-up 전에 가장 큰 병목을 확인한다. S4d는 final rollout에서는 강했지만, 새로 생성되는 span 자체의 content/entity recall은 낮았다. S4e는 공유 조건 구조가 이 병목을 줄일 수 있는지 확인한다.

따라서 S4e의 좋은 결과는 단순히 rollout score가 높다는 것이 아니다. 좋은 결과는 다음을 동시에 만족해야 한다.

```text
1. 공유 모델에서도 importance가 strict controls를 이긴다.
2. S4d보다 generated span의 content/entity recall이 오른다.
3. final rollout 품질은 S4d보다 크게 나빠지지 않는다.
```

## 3. 결과가 의미하는 것

S4e는 `process_ready=true`, `overall_pass=false`, `structure_review_needed=true`, `s5_ready=false`다.

Gate 결과는 다음이다.

| Gate | 통과 | 핵심 수치 |
|---|---:|---|
| `S4E-G-RUN` | true | 여섯 schedule 모두 실행 |
| `S4E-G-LOSS-FINITE` | true | 모든 eval loss 유한 |
| `S4E-G-SHARED-CONDITION-MODEL` | true | 하나의 공유 모델 사용 |
| `S4E-G-IMPORTANCE-BEATS-RANDOM` | true | 0.7569 > 0.6406 |
| `S4E-G-IMPORTANCE-BEATS-POSITION-ONLY` | true | 0.7569 > 0.1576 |
| `S4E-G-IMPORTANCE-BEATS-SAME-POSITION-RANDOM` | true | 0.7569 > 0.4761 |
| `S4E-G-WRONG-DOC-DROPS` | true | 0.7569 > 0.1396 |
| `S4E-G-NO-ANCHOR-DROPS` | true | 0.7569 > 0.1496 |
| `S4E-G-SPAN-CONTENT-GAIN` | false | 0.0029 < S4d 0.0172 |
| `S4E-G-SPAN-ENTITY-GAIN` | false | 0.0013 < S4d 0.0047 |
| `S4E-G-FINAL-NONREGRESSION` | true | final content, drift, repetition은 S4d 대비 비퇴행 |
| `S4E-G-ARTIFACT-CHECK` | false | importance artifact 0.6906 > random 0.5259 |

Teacher-forced span 예측 결과는 다음이다.

| Schedule | Loss | TF Delta Acc | Delta Content | Entity | Artifact |
|---|---:|---:|---:|---:|---:|
| `importance_schedule` | 3.1220 | 0.2876 | 0.0029 | 0.0013 | 0.6906 |
| `random_schedule` | 3.8858 | 0.1652 | 0.0013 | 0.0008 | 0.5259 |
| `position_only_schedule` | 3.4056 | 0.1750 | 0.0016 | 0.0008 | 0.4709 |
| `same_position_random_schedule` | 3.4046 | 0.1682 | 0.0021 | 0.0011 | 0.5328 |
| `wrong_document_same_position_schedule` | 3.4282 | 0.1348 | 0.0018 | 0.0007 | 0.5861 |
| `no_anchor_gap_only_schedule` | 3.4212 | 0.1560 | 0.0005 | 0.0001 | 0.4279 |

Rollout final 결과는 다음이다.

| Schedule | Final F1 | ROUGE-L | Content | Original Content | Entity | Repetition | Drift | Rollout Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `importance_schedule` | 0.4059 | 0.3321 | 0.3464 | 0.3485 | 0.2976 | 0.0375 | 0.6323 | 0.7569 |
| `random_schedule` | 0.3956 | 0.3282 | 0.2404 | 0.2437 | 0.2137 | 0.0428 | 0.7130 | 0.6406 |
| `position_only_schedule` | 0.1738 | 0.1369 | 0.0065 | 0.0065 | 0.0079 | 0.0807 | 0.9240 | 0.1576 |
| `same_position_random_schedule` | 0.3288 | 0.1876 | 0.1819 | 0.1853 | 0.1725 | 0.0653 | 0.7774 | 0.4761 |
| `wrong_document_same_position_schedule` | 0.1489 | 0.1181 | 0.0127 | 0.0127 | 0.0132 | 0.0368 | 0.9220 | 0.1396 |
| `no_anchor_gap_only_schedule` | 0.1837 | 0.1509 | 0.0049 | 0.0049 | 0.0074 | 0.2975 | 0.9230 | 0.1496 |

핵심 결과는 세 가지다.

첫째, 공유 모델에서도 importance 우위는 유지됐다.

```text
importance rollout score: 0.7569
random rollout score:     0.6406
delta:                    +0.1163
```

이는 S4d의 가장 중요한 신호를 강화한다. 조건별 모델을 따로 학습하지 않아도, 실제 semantic skeleton content와 좌우 anchor가 있을 때 final rollout이 더 좋았다.

둘째, strict control도 모두 이겼다.

```text
same-position random: 0.4761
wrong-document:       0.1396
no-anchor:            0.1496
importance:           0.7569
```

이 결과는 “같은 위치에 아무 token content나 넣으면 되는 것”이라는 반론을 다시 약화한다. 특히 wrong-document와 no-anchor는 공유 모델 안에서도 거의 무너졌다.

셋째, 하지만 S4e의 본래 개선 목표였던 generated span content/entity는 실패했다.

```text
S4d span content recall: 0.0172
S4e span content recall: 0.0029

S4d span entity recall:  0.0047
S4e span entity recall:  0.0013
```

S4e는 final rollout을 개선했지만, 새로 생성되는 span 하나하나가 더 의미 있는 content/entity를 담도록 만들지는 못했다. Sample에서도 `Cu`, `Mc`, 쉼표, `and`, `the`, 짧은 고유명 subword 같은 artifact가 많이 보였다.

## 4. 긍정적인 점, 부정적인 점, 남은 모호성

긍정적인 점은 S4d의 핵심 해석이 더 강해졌다는 것이다. 공유 모델 안에서도 importance는 random, position-only, same-position random, wrong-document, no-anchor를 모두 이겼다. 즉 S4d의 우위를 “조건마다 다른 모델을 학습했기 때문”이라고 설명하기는 어려워졌다.

또한 final rollout은 S4d보다 나빠지지 않았다.

```text
S4d final content recall: 0.3340
S4e final content recall: 0.3464

S4d drift proxy: 0.6433
S4e drift proxy: 0.6323

S4d repetition: 0.0469
S4e repetition: 0.0375
```

이는 공유 조건 구조가 final state 보존에는 도움이 될 수 있음을 보여준다. S4e의 rollout score 0.7569도 S4d의 0.7175보다 높다.

부정적인 점은 더 중요하다. S4e는 새 span 자체의 의미 정보를 늘리지 못했다. TF Delta Acc는 importance가 0.2876으로 높지만, content recall은 0.0029에 불과하다. 이는 모델이 의미 content보다 형식 token, punctuation, 빈번한 subword를 맞히는 쪽으로 학습했을 가능성을 보여준다.

Artifact rate도 나쁘다.

```text
importance artifact rate: 0.6906
random artifact rate:     0.5259
```

따라서 S4e의 높은 rollout score를 “좋은 span generation”으로 해석하면 안 된다. 더 방어 가능한 해석은 다음이다.

```text
공유 모델에서도 semantic skeleton은 final trajectory를 좋게 만든다.
하지만 현재 decoder는 새 span을 의미 단위로 생성하기보다
형식 token과 짧은 subword shortcut에 크게 의존한다.
```

남은 모호성은 final rollout metric이 skeleton에 이미 남아 있던 content와 새로 생성된 content를 함께 반영한다는 점이다. Final content recall이 높아진 이유가 새 span 생성 개선 때문인지, 초기 skeleton content가 더 잘 보존된 효과인지 분리해야 한다. S4e의 span-level gate 실패는 후자 가능성을 강하게 시사한다.

계산 비용도 새롭게 확인된 병목이다.

```text
combined train examples: 277,999
runtime: 약 2,241초
```

공유 모델은 해석 confound를 줄이지만, 모든 schedule 예제를 단순 합산하면 실행 비용이 크다. 다음 실험에는 조건별 균형 샘플링이나 span subsampling이 필요하다.

## 5. 다음 실험에서 어떻게 검증할 것인가

S4e의 방어 가능한 결론은 다음이다.

```text
S4e confirms that the S4d rollout advantage is not caused by separate per-condition models.
However, shared-condition training does not solve generated-span semantic content collapse.
```

한국어로는 다음과 같다.

```text
S4e는 semantic skeleton의 rollout 우위가 모델 분리 때문만은 아니라는 점을 확인했다.
하지만 새로 생성되는 span 자체의 의미 정보량을 높이는 데는 실패했다.
```

따라서 다음 단계는 S5 scale-up이 아니다. 지금 바로 모델 크기를 키우면, artifact와 content/entity collapse를 더 큰 비용으로 반복할 가능성이 높다.

다음 구조 개선은 다음 방향이 좋다.

1. 학습 예제를 조건별로 균형 샘플링해 비용을 줄인다.
2. target span을 subword 조각이 아니라 단어 또는 의미 chunk 단위로 묶는다.
3. punctuation/function-token span과 content-token span을 분리해서 평가하고, 필요하면 decoding 경로도 분리한다.
4. left/right anchor를 단순 embedding 합산이 아니라 span query가 직접 cross-attention하는 구조로 바꾼다.
5. final rollout score와 generated-span content score를 분리한 채로 gate를 유지한다.

가장 중요한 다음 질문은 이것이다.

```text
semantic skeleton의 final rollout 우위를 유지하면서,
새로 붙이는 span 자체가 실제 content/entity를 담도록 만드는 구조는 무엇인가?
```
