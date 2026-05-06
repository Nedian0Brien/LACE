# S4 결과: importance-ordered reverse diffusion

## 1. 이 실험이 측정한 것

S4는 문장 exact reconstruction probe가 아니라, 중요도 기반 forward masking schedule이 random corruption schedule보다 더 나은 reverse expansion curriculum을 만드는지 확인한 process-level 실험이다.

실험 이름은 다음이다.

```text
S4-importance ordered reverse diffusion
```

Forward process는 원문 `x0`에서 중요도가 낮은 token부터 순차적으로 mask해 semantic skeleton에 도달하는 방식으로 정의했다. Reverse process는 그 반대 방향이다.

```text
forward: x0 -> 75% state -> 50% state -> 25% skeleton
reverse: 25% skeleton -> 50% state -> 75% state -> full state
```

학습 목표는 원문 전체를 한 번에 복원하는 것이 아니라 다음 전이를 학습하는 것이다.

```text
p_theta(x_{t-1} | x_t, t)
```

실행 정보는 다음이다.

| 항목 | 값 |
|---|---|
| phase | `v2_s4` |
| Kaggle kernel | `dennisparknd/lace-v2-s4-importance-ordered-reverse-diffusion` |
| model | `t5-small` tokenizer + custom PyTorch encoder-decoder |
| data | `wikitext/wikitext-2-raw-v1:train` |
| train samples | 768 |
| eval samples | 192 |
| device | `cuda` |
| ratios | `0.25 -> 0.50 -> 0.75 -> 1.00` |
| output | `outputs/v2_s4/lace_v2_s4/` |

비교 조건은 다음이다.

| 조건 | 의미 |
|---|---|
| `importance_schedule` | attention-received score가 높은 token을 오래 보존하고 낮은 token부터 mask하는 schedule |
| `random_schedule` | 같은 ratio와 같은 token count를 random 순서로 보존하는 schedule |
| `position_only_schedule` | importance 위치 scaffold만 주고 token content는 pad로 대체하는 control |

## 2. 왜 중요한가

S3 계열은 attention terminal이 완전히 무의미하지는 않지만, terminal reconstruction probe와 lexical metric만으로 연구 본질을 판단하기 어렵다는 점을 보여줬다.

S4는 질문을 다시 본질로 올린다.

```text
중요도 낮은 token부터 순차적으로 masking하는 forward process의 역과정이
random corruption보다 더 좋은 generation curriculum을 만드는가?
```

따라서 S4의 핵심은 `attention_terminal` 하나의 score가 아니라 trajectory 전체다. 좋은 schedule이라면 중심 의미 token을 보존하면서 다음 단계의 세부 token을 더 잘 붙여야 한다.

## 3. 결과가 의미하는 것

S4는 `process_ready=true`, `overall_pass=false`, `s5_ready=false`다. 모든 schedule과 transition은 실행됐고 loss도 finite였지만, 종합 score에서 `importance_schedule`은 `random_schedule`을 이기지 못했다.

Gate 결과는 다음이다.

| Gate | 통과 | 해석 |
|---|---:|---|
| `S4-G-RUN` | true | 세 schedule이 모두 실행됐다. |
| `S4-G-LOSS-FINITE` | true | 모든 schedule의 loss가 정상 숫자로 계산됐다. |
| `S4-G-IMPORTANCE-BEATS-RANDOM` | false | random score 0.5607이 importance score 0.4839보다 높았다. |
| `S4-G-IMPORTANCE-BEATS-POSITION-ONLY` | true | importance score는 position-only 0.3752보다 충분히 높았다. |
| `S4-G-EXPANSION-RECALL` | true | importance expansion recall 0.0416이 random 0.0203보다 높았다. |
| `S4-G-RETENTION` | true | importance input retention 0.0471이 random 0.0277보다 높았다. |
| `S4-G-REPETITION` | false | importance repetition 0.1287이 random 0.1063보다 tolerance 이상 높았다. |
| `S4-G-BEST-IDENTIFIED` | true | 최고 schedule은 `random_schedule`이었다. |

Schedule별 결과는 다음이다.

| Schedule | Loss | PPL | Token F1 | ROUGE-L | Target Content | Input Retention | Expansion Recall | Original Content | Entity | Repetition | Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `importance_schedule` | 6.3588 | 577.54 | 0.1754 | 0.1332 | 0.0574 | 0.0471 | 0.0416 | 0.0496 | 0.0831 | 0.1287 | 0.4839 |
| `random_schedule` | 6.0172 | 410.43 | 0.2300 | 0.1712 | 0.0300 | 0.0277 | 0.0203 | 0.0271 | 0.0412 | 0.1063 | 0.5607 |
| `position_only_schedule` | 6.5464 | 696.75 | 0.1368 | 0.1052 | 0.0191 | 0.0000 | 0.0191 | 0.0172 | 0.0323 | 0.1468 | 0.3752 |

핵심 비교는 두 갈래로 나뉜다.

첫째, random schedule은 표면 복원 지표에서 더 좋다.

```text
random_schedule:     loss 6.0172, Token F1 0.2300, ROUGE-L 0.1712, score 0.5607
importance_schedule: loss 6.3588, Token F1 0.1754, ROUGE-L 0.1332, score 0.4839
```

둘째, importance schedule은 의미 보존과 확장 지표에서 더 좋다.

```text
target_content_recall:   importance 0.0574 > random 0.0300
input_retention:         importance 0.0471 > random 0.0277
expansion_recall:        importance 0.0416 > random 0.0203
original_content_recall: importance 0.0496 > random 0.0271
entity_recall:           importance 0.0831 > random 0.0412
```

이 결과는 단순한 실패가 아니다. `importance_schedule`은 아직 더 좋은 language model trajectory를 만들지는 못했지만, 중심 의미를 보존하고 다음 단계의 의미 token을 붙이는 방향에서는 random보다 강한 신호를 보였다.

Transition별로 보면 importance schedule도 단계가 진행될수록 좋아졌다.

| Transition | Token F1 | ROUGE-L | Expansion Recall | Score |
|---|---:|---:|---:|---:|
| `0.25->0.50` | 0.1149 | 0.0947 | 0.0179 | 0.3788 |
| `0.50->0.75` | 0.1734 | 0.1337 | 0.0358 | 0.4802 |
| `0.75->1.00` | 0.2379 | 0.1713 | 0.0710 | 0.5927 |

이 패턴은 중요한 긍정 신호다. semantic skeleton에서 바로 full text로 가는 것보다, 단계가 풍부해질수록 transition이 쉬워진다. 즉 reverse expansion curriculum 자체는 작동할 가능성이 있다.

## 4. 어떤 반론과 혼동 요인을 다뤘는가

S4는 "문장 그대로 복원하는 게 중요한가?"라는 문제를 일부 다뤘다. 이번 실험은 원문 전체 복원이 아니라 `x_t -> x_{t-1}` 전이를 보았고, schedule-level reverse curriculum을 비교했다.

다룬 반론은 다음이다.

첫째, position-only 반론은 약해졌다. `importance_schedule` score 0.4839는 `position_only_schedule` 0.3752보다 높고, target content, input retention, expansion recall, entity recall도 모두 더 높다. 따라서 content-bearing skeleton이 위치 scaffold만의 효과는 아니다.

둘째, random corruption 반론은 남아 있다. `random_schedule`은 loss, Token F1, ROUGE-L, 종합 score에서 더 좋았다. 현재 설정만으로는 importance-ordered reverse diffusion이 random corruption보다 더 좋은 language model trajectory라고 주장할 수 없다.

셋째, 의미 보존 지표는 importance 쪽이 강했다. 이는 연구 가설의 핵심 직관, 즉 중심 의미 token부터 확장하면 의미 보존과 semantic faithfulness가 좋아질 수 있다는 방향을 지지한다.

넷째, metric 설계 caveat가 있다. 각 schedule은 서로 다른 target state를 가진다. `random_schedule`의 target은 random order로 보존된 token 집합이고, `importance_schedule`의 target은 importance order로 보존된 token 집합이다. 따라서 Token F1/ROUGE-L을 schedule 간에 단순 비교하면 target 난이도와 token frequency 차이가 섞일 수 있다. 의미 지표와 공통 original-content 지표를 함께 봐야 한다.

다섯째, generation quality는 아직 낮다. sample에는 반복, 빈출 template, 부정확한 entity 생성이 남아 있다. S4는 constrained reverse-transition 실험이지 open-ended generation 성공 증거가 아니다.

## 5. 다음 실험에서 어떻게 검증할 것인가

S4 결과는 방향을 분명히 한다.

```text
importance schedule은 semantic signal을 더 잘 보존하지만,
현재 objective와 model setting에서는 random schedule보다 전체 reverse LM score가 낮다.
```

다음 단계는 S5가 아니라 S4a가 적절하다. S4a는 다음 두 가지를 고쳐야 한다.

1. 공통 target 평가를 추가한다. schedule-specific target만 보면 random과 importance가 서로 다른 target을 맞히는 문제가 생긴다. 각 transition 출력이 같은 original text 또는 같은 held-out semantic target에 얼마나 가까운지 별도로 평가해야 한다.
2. Reverse objective를 "전체 target sequence 생성"에서 "추가되어야 할 token/span 예측"으로 바꾼다. 현재 모델은 입력에 이미 있는 skeleton을 유지하면서 target 전체를 다시 생성해야 하므로, retention과 expansion이 섞인다. 다음에는 newly unmasked token 또는 span만 예측하도록 해야 한다.

S4a 후보 이름은 다음이다.

```text
S4a: delta-token reverse objective
```

핵심 비교는 다음이다.

```text
importance_schedule: predict newly unmasked content tokens
random_schedule:     predict newly unmasked content tokens
position_only:       same positions, no token content
```

성공 기준은 다음과 같이 바꾼다.

- importance가 random보다 delta-token recall이 높아야 한다.
- importance가 random보다 original semantic content recall이 높아야 한다.
- importance가 position-only보다 충분히 높아야 한다.
- repetition은 낮아야 한다.
- final rollout에서 중심 의미가 drift하지 않아야 한다.

현재 방어 가능한 결론은 다음이다.

```text
S4 does not prove that importance-ordered reverse diffusion is a better
language model than random corruption. It does show that the importance
schedule carries stronger semantic retention and expansion signals than
random or position-only schedules.
```

한국어로는 다음과 같이 정리할 수 있다.

```text
S4는 importance schedule이 random보다 종합 reverse LM 성능이 좋다는 증거는 주지 못했다.
하지만 중심 의미 보존과 세부 의미 확장 신호는 random보다 강했다.
다음은 전체 state 복원이 아니라 새로 추가될 delta token/span을 예측하는 objective가 필요하다.
```
