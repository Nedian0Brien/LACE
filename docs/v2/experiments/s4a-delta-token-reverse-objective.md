# S4a 결과: delta-token reverse objective

## 1. 이 실험이 측정한 것

S4a는 S4의 핵심 모호성을 줄이기 위해 실행한 objective 보정 실험이다.

S4는 다음 reverse transition을 학습했다.

```text
input:  현재 partial state
target: 다음 partial state 전체
```

S4a는 target을 다음 상태 전체가 아니라 새로 unmask될 token/span으로 제한했다.

```text
input:  현재 partial state + 이번 단계에서 채울 위치 marker
target: newly unmasked delta token/span
```

따라서 S4a가 측정한 것은 문장 전체 복원이 아니라, `importance_schedule`의 semantic skeleton이 다음 단계의 세부 token을 예측하는 데 `random_schedule`이나 `position_only_schedule`보다 더 좋은 조건을 주는가다.

실행 정보는 다음이다.

| 항목 | 값 |
|---|---|
| phase | `v2_s4a` |
| Kaggle kernel | `dennisparknd/lace-v2-s4a-delta-token-reverse-objective` |
| model | `t5-small` tokenizer + custom PyTorch encoder-decoder |
| data | `wikitext/wikitext-2-raw-v1:train` |
| train samples | 768 |
| eval samples | 192 |
| device | `cuda` |
| ratios | `0.25 -> 0.50 -> 0.75 -> 1.00` |
| reverse epochs | 2 |
| output | `outputs/v2_s4a/lace_v2_s4a/` |

비교 조건은 다음이다.

| 조건 | 의미 |
|---|---|
| `importance_schedule` | attention-received score가 높은 token을 오래 보존하고, 다음 단계의 delta token만 예측한다. |
| `random_schedule` | 같은 ratio와 같은 token count에서 random order의 delta token만 예측한다. |
| `position_only_schedule` | importance 위치 marker는 주지만 token content는 제거한다. |

## 2. 왜 중요한가

S4에서는 결과가 둘로 갈라졌다.

```text
random:     loss / Token F1 / ROUGE-L / 종합 score 우위
importance: semantic retention / expansion / entity signal 우위
```

이 갈림은 전체 target state를 다시 생성하게 한 objective 때문에 생겼을 수 있다. 전체 state target은 이미 입력에 있는 skeleton token을 다시 유지하는 능력과, 새로 추가될 token을 예측하는 능력을 섞는다.

하지만 연구의 본질은 후자다. 즉, 중심 의미 token이 주어졌을 때 세부 의미 token을 더 잘 붙일 수 있는가가 핵심이다. S4a는 이 질문에 더 직접적으로 접근한다.

## 3. 결과가 의미하는 것

S4a는 `process_ready=true`, `overall_pass=true`, `structure_review_needed=false`, `s5_ready=false`다.

Gate 결과는 다음이다.

| Gate | 통과 | 해석 |
|---|---:|---|
| `S4A-G-RUN` | true | 세 schedule이 모두 실행됐다. |
| `S4A-G-LOSS-FINITE` | true | 모든 schedule의 eval loss가 유한했다. |
| `S4A-G-IMPORTANCE-BEATS-RANDOM` | true | importance score 0.6366이 random score 0.5073보다 높았다. |
| `S4A-G-IMPORTANCE-BEATS-POSITION-ONLY` | true | importance score 0.6366이 position-only score 0.5889보다 높았다. |
| `S4A-G-DELTA-CONTENT` | true | importance delta content recall 0.0136이 random 0.0073보다 높았다. |
| `S4A-G-DELTA-ACCURACY` | true | importance teacher-forced delta accuracy 0.1577이 random 0.1092보다 높았다. |
| `S4A-G-ENTITY-DELTA` | false | entity recall은 random 0.0175가 importance 0.0115보다 높았다. |
| `S4A-G-COPY-LEAKAGE` | true | importance context copy rate 0.0046은 random 0.0076보다 낮았다. |
| `S4A-G-REPETITION` | false | importance repetition 0.1584가 random 0.1062보다 높았다. |
| `S4A-G-BEST-IDENTIFIED` | true | 최고 schedule은 `importance_schedule`이었다. |

Schedule별 결과는 다음이다.

| Schedule | Loss | PPL | TF Delta Acc | Delta F1 | Delta ROUGE-L | Delta Content | Context Copy | Original Content | Entity | Repetition | Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `importance_schedule` | 5.7282 | 307.43 | 0.1577 | 0.1700 | 0.1468 | 0.0136 | 0.0046 | 0.0035 | 0.0115 | 0.1584 | 0.6366 |
| `random_schedule` | 6.7063 | 817.58 | 0.1092 | 0.1258 | 0.1077 | 0.0073 | 0.0076 | 0.0064 | 0.0175 | 0.1062 | 0.5073 |
| `position_only_schedule` | 5.8011 | 330.66 | 0.1483 | 0.1381 | 0.1233 | 0.0156 | 0.0039 | 0.0020 | 0.0079 | 0.1994 | 0.5889 |

핵심 결과는 분명하다.

```text
S4에서는 random이 종합 score에서 이겼다.
S4a에서는 objective를 delta-token 예측으로 바꾸자 importance가 random을 이겼다.
```

특히 teacher-forced delta accuracy가 중요하다.

```text
importance: 0.1577
random:     0.1092
position:   0.1483
```

이 값은 생성 문자열의 표면 품질보다 더 직접적으로 "조건부로 다음 token을 맞힐 수 있는가"를 본다. 여기서 importance가 random보다 확실히 높다는 것은 S4의 실패가 semantic skeleton 가설 자체의 실패라기보다, 전체 target state를 다시 생성하게 한 objective가 연구 질문과 덜 맞았다는 해석을 지지한다.

## 4. 어떤 반론과 혼동 요인을 다뤘는가

첫째, random corruption 반론은 약해졌다. S4에서는 random이 score 0.5607로 importance 0.4839보다 높았다. 하지만 S4a에서는 importance score 0.6366이 random 0.5073보다 높다. 즉 "다음에 붙일 token만 예측한다"는 objective에서는 semantic skeleton 쪽이 더 나은 조건을 준다.

둘째, position-only 반론은 완전히 사라지지는 않았다. Importance는 position-only보다 score와 delta F1/ROUGE-L에서 높았지만, position-only도 score 0.5889, TF Delta Acc 0.1483으로 꽤 강하다. 또한 delta content recall은 position-only 0.0156이 importance 0.0136보다 높다. 따라서 content-bearing skeleton의 효과는 확인됐지만, 위치 marker가 매우 강한 prior라는 caveat는 계속 유지해야 한다.

셋째, entity 복원은 아직 약하다. Random의 entity recall 0.0175가 importance 0.0115보다 높다. 이는 중요도 기반 skeleton이 중심 의미 흐름에는 유리해도, 새로 붙어야 하는 rare entity, 숫자, surface marker를 안정적으로 생성하는 구조는 아직 부족하다는 뜻이다.

넷째, repetition 문제가 남았다. Importance repetition 0.1584는 random 0.1062보다 높다. Sample에서도 `the`, `of`, `and` 같은 짧은 token 반복이 보인다. 따라서 delta objective는 가설을 살렸지만, decoder의 생성 방식은 아직 구조적 보정이 필요하다.

다섯째, S4a는 여전히 constrained experiment다. Teacher-forced delta accuracy와 greedy delta generation을 본 것이지, open-ended generation 성공을 보인 것은 아니다. 따라서 `s5_ready=false`는 유지한다.

## 5. 다음 실험에서 어떻게 검증할 것인가

S4a는 좋은 신호를 줬다.

```text
Semantic skeleton + positional scaffold는 random corruption보다
delta-token reverse objective에서 더 나은 조건을 만든다.
```

하지만 바로 S5로 넘어가지는 않는다. 다음은 S4b 성격의 multi-step delta rollout 또는 구조 보정 실험이 적절하다.

우선순위는 다음이다.

1. `S4b: multi-step delta rollout`
   S4a의 각 step 성능이 실제 `25% -> 50% -> 75% -> 100%` rollout에서 누적될 때 semantic drift와 repetition이 어떻게 변하는지 본다.

2. `S4c: span-infilling reverse decoder`
   지금 decoder는 delta token sequence를 자유 생성한다. 다음에는 위치 marker별 span을 채우는 insertion/infilling 구조로 바꿔 반복을 줄인다.

3. `entity-aware delta head`
   entity recall 실패를 다루기 위해 surface entity, 숫자, rare token에 별도 loss weight 또는 copy/pointer 계열 보조 head를 둔다.

4. `position-only hardening`
   position-only가 강하므로 wrong-document/same-position, shuffled-delta-position, target-length matched random control을 S4b/S4c 가까이에 둔다.

현재 방어 가능한 결론은 다음이다.

```text
S4a supports the core process claim more strongly than S4:
when the reverse objective is restricted to newly unmasked tokens,
the importance-ordered schedule beats random corruption and position-only controls
on the aggregate delta objective.
```

한국어로는 다음과 같이 정리할 수 있다.

```text
S4a는 중요도 기반 semantic skeleton이 다음 세부 token을 예측하는 데
random corruption보다 더 좋은 조건을 준다는 첫 강한 process-level 증거다.
다만 entity 복원과 반복 제어는 아직 약하므로, 다음은 open-ended S5가 아니라
multi-step rollout과 span-infilling 구조 보정으로 가야 한다.
```
