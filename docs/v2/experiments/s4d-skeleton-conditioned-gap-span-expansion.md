# S4d 결과: skeleton-conditioned gap/span expansion

## 1. 이 실험이 측정한 것

S4d는 S4b의 multi-step rollout 신호를 더 엄격하게 검증하기 위해 실행한 구조 실험이다.

S4b는 generated delta를 누적 삽입하는 실제 rollout에서 `importance_schedule`이 random과 position-only보다 좋다는 결과를 줬다. 하지만 아직 다음 반론이 남아 있었다.

```text
좋아진 것이 semantic skeleton content 때문인가?
아니면 같은 위치 구조, gap marker, decoder prior 때문인가?
```

S4d는 이 반론을 직접 겨냥했다. 모델은 전체 target state를 다시 생성하지 않고, 각 transition에서 새로 unmask될 contiguous gap/span만 예측한다.

```text
input:  current skeleton tokens + positions
        left/right semantic anchor role
        newly opened span marker positions
        timestep

target: newly unmasked span token ids
```

실행 정보는 다음이다.

| 항목 | 값 |
|---|---|
| phase | `v2_s4d` |
| Kaggle kernel | `dennisparknd/lace-v2-s4d-skeleton-gap-span-expansion` version 2 |
| model | `t5-small` tokenizer + custom PyTorch encoder-decoder + role embedding |
| data | `wikitext/wikitext-2-raw-v1:train` |
| train samples | 768 |
| eval samples | 192 |
| device | `cuda` |
| ratios | `0.25 -> 0.50 -> 0.75 -> 1.00` |
| reverse epochs | 2 |
| output | `outputs/v2_s4d/lace_v2_s4d/` |

비교 조건은 다음이다.

| 조건 | 의미 |
|---|---|
| `importance_schedule` | 실제 importance skeleton token content와 left/right anchor를 사용한다. |
| `random_schedule` | 같은 ratio와 token budget에서 random skeleton을 사용한다. |
| `position_only_schedule` | importance 위치와 gap 구조는 주되 token content를 제거한다. |
| `same_position_random_schedule` | importance와 같은 위치에 같은 문서의 random token content를 넣는다. |
| `wrong_document_same_position_schedule` | importance와 같은 위치에 다른 문서의 skeleton token content를 넣는다. |
| `no_anchor_gap_only_schedule` | current skeleton 없이 이번 span marker 위치만 준다. |

## 2. 왜 중요한가

S4d는 현재 연구 본질에 가장 가까운 구조 검증이다.

핵심 가설은 문장을 그대로 복원하는 probe가 아니라, forward process에서 중심 의미 token을 남기고 나머지를 순차 masking한 뒤, reverse process에서 중심 의미 token부터 세부 span을 붙여 문장을 확장하는 능력이다.

따라서 중요한 것은 다음이다.

```text
같은 gap 구조에서,
실제 semantic skeleton content가 있을 때만
span 생성과 rollout이 좋아지는가?
```

S4d의 control들은 이 질문을 분해한다. `position_only_schedule`은 위치만의 효과를 본다. `same_position_random_schedule`은 같은 위치에 아무 token content를 넣어도 되는지 본다. `wrong_document_same_position_schedule`은 문맥과 무관한 content가 오히려 방해되는지 본다. `no_anchor_gap_only_schedule`은 span marker만으로 decoder prior가 얼마나 버티는지 본다.

이 조건들을 이겨야 “semantic skeleton이 위치 scaffold 위에 실제 정보를 전달한다”는 해석이 가능하다.

## 3. 결과가 의미하는 것

S4d는 `process_ready=true`, `overall_pass=true`, `structure_review_needed=false`, `s5_ready=false`다.

Gate 결과는 다음이다.

| Gate | 통과 | 핵심 수치 |
|---|---:|---|
| `S4D-G-RUN` | true | 여섯 schedule 모두 실행 |
| `S4D-G-LOSS-FINITE` | true | 모든 eval loss 유한 |
| `S4D-G-IMPORTANCE-BEATS-RANDOM` | true | 0.7175 > 0.6145 |
| `S4D-G-IMPORTANCE-BEATS-POSITION-ONLY` | true | 0.7175 > 0.0133 |
| `S4D-G-IMPORTANCE-BEATS-SAME-POSITION-RANDOM` | true | 0.7175 > 0.4733 |
| `S4D-G-WRONG-DOC-DROPS` | true | 0.7175 > 0.0504 |
| `S4D-G-NO-ANCHOR-DROPS` | true | 0.7175 > 0.0300 |
| `S4D-G-SPAN-CONTENT-NONZERO` | true | span content 0.0172, final content 0.3340 |
| `S4D-G-ROLLOUT-NONREGRESSION` | true | drift는 random보다 낮고 repetition 증가는 tolerance 안 |

Teacher-forced span 예측 결과는 다음이다.

| Schedule | Loss | TF Delta Acc | Delta F1 | Delta Content | Entity | Score |
|---|---:|---:|---:|---:|---:|---:|
| `importance_schedule` | 2.9922 | 0.3091 | 0.1330 | 0.0172 | 0.0047 | 0.9320 |
| `random_schedule` | 3.7453 | 0.1868 | 0.0381 | 0.0071 | 0.0045 | 0.5980 |
| `position_only_schedule` | 3.4122 | 0.1686 | 0.0104 | 0.0000 | 0.0000 | 0.5445 |
| `same_position_random_schedule` | 3.4844 | 0.1611 | 0.0108 | 0.0015 | 0.0016 | 0.5338 |
| `wrong_document_same_position_schedule` | 3.3856 | 0.1620 | 0.0105 | 0.0000 | 0.0000 | 0.5151 |
| `no_anchor_gap_only_schedule` | 4.0014 | 0.1022 | 0.0243 | 0.0000 | 0.0000 | 0.4562 |

Rollout final 결과는 다음이다.

| Schedule | Final F1 | ROUGE-L | Content | Original Content | Entity | Repetition | Drift | Rollout Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `importance_schedule` | 0.3863 | 0.3092 | 0.3340 | 0.3348 | 0.2988 | 0.0469 | 0.6433 | 0.7175 |
| `random_schedule` | 0.3645 | 0.3148 | 0.2448 | 0.2468 | 0.2202 | 0.0296 | 0.7114 | 0.6145 |
| `position_only_schedule` | 0.0778 | 0.0737 | 0.0003 | 0.0003 | 0.0000 | 0.4036 | 0.9388 | 0.0133 |
| `same_position_random_schedule` | 0.3185 | 0.1747 | 0.1952 | 0.1993 | 0.1763 | 0.0646 | 0.7704 | 0.4733 |
| `wrong_document_same_position_schedule` | 0.0814 | 0.0726 | 0.0042 | 0.0042 | 0.0047 | 0.0985 | 0.9356 | 0.0504 |
| `no_anchor_gap_only_schedule` | 0.1247 | 0.1247 | 0.0000 | 0.0000 | 0.0000 | 0.8516 | 0.9313 | 0.0300 |

핵심 결과는 세 가지다.

첫째, importance는 random보다 여전히 높다.

```text
importance rollout score: 0.7175
random rollout score:     0.6145
delta:                    +0.1030
```

S4b와 같은 방향이 유지됐다. S4d는 span 단위 구조와 더 엄격한 control을 붙였는데도 importance 우위가 남았다.

둘째, position-only와 no-anchor는 거의 무너졌다.

```text
position-only final content: 0.0003
no-anchor final content:     0.0000
```

이는 S4c와 반대 방향의 중요한 개선이다. S4c에서는 position-only가 importance보다 높았지만, S4d에서는 role-aware span decoder와 rollout 구조 안에서 위치만으로는 의미 있는 최종 state를 만들지 못했다.

셋째, same-position random과 wrong-document control을 모두 이겼다.

```text
same-position random score: 0.4733
wrong-document score:      0.0504
importance score:          0.7175
```

이 비교가 S4d의 가장 중요한 연구 신호다. 같은 위치에 content를 넣는 것만으로는 충분하지 않고, 문맥에 맞는 실제 skeleton content가 있을 때 성능이 크게 좋아진다. 특히 wrong-document content는 position이 같아도 거의 도움이 되지 않았다.

## 4. 긍정적인 점, 부정적인 점, 남은 모호성

긍정적인 점은 분명하다. S4d는 현재까지 semantic skeleton content 사용 증거를 가장 잘 분리한다. S4b는 random과 position-only를 이겼고, S4d는 여기에 `same_position_random`, `wrong_document_same_position`, `no_anchor_gap_only`까지 붙여 모두 이겼다. 따라서 “좋아진 것이 단순 위치 scaffold 때문”이라는 반론은 꽤 약해졌다.

또 다른 긍정 신호는 entity recall이다.

```text
S4b importance entity recall: 0.2855
S4d importance entity recall: 0.2988
```

S4d는 S4b보다 rollout score는 약간 낮지만, entity recall은 조금 높다. left/right anchor role이 rare entity와 고유명사 주변 복원에 약간 도움이 됐을 가능성이 있다.

부정적인 점도 있다. 생성문 자체는 아직 자연스러운 언어가 아니다. Sample에는 반복, subword artifact, 어색한 phrase가 많다. 따라서 S4d를 “좋은 language generation”의 증거로 말하면 안 된다.

또한 span-level content recall은 아직 낮다.

```text
importance span content recall: 0.0172
importance span entity recall:  0.0047
```

최종 rollout content/entity는 높은 편이지만, 새 span 하나하나를 의미 단위로 정확히 생성하는 능력은 약하다. 현재 지표의 target word count가 평균 0.9475인 것을 보면, 많은 span target이 매우 짧고 subword/형식 token을 포함한다. 즉, final state metric은 skeleton에 이미 남아 있던 content와 rollout 누적 효과를 함께 반영한다.

남은 모호성은 random의 ROUGE-L이다.

```text
importance ROUGE-L: 0.3092
random ROUGE-L:     0.3148
```

S4b와 마찬가지로 random은 표면 순서 겹침에서 여전히 조금 강하다. 하지만 content, original content, entity, drift, overall score는 모두 importance가 낫다. 따라서 현재 방어 가능한 해석은 “표면 overlap이 아니라 의미 보존과 문맥 맞는 확장에서 importance가 낫다”이다.

마지막으로, S4d runner는 계산 비용이 크다. version 1은 AMP 학습 중 `nan`으로 실패했고, version 2는 AMP를 끄고 학습률을 낮춰 완료했다. 여섯 조건을 각각 별도 모델로 학습하기 때문에 실행 시간이 길다. 다음 반복에서는 schedule condition embedding을 넣은 shared model로 비용을 줄이는 것이 좋다.

## 5. 다음 실험에서 어떻게 검증할 것인가

S4d의 방어 가능한 결론은 다음이다.

```text
S4d supports that semantic skeleton content, not merely position or gap markers,
improves constrained reverse span expansion under matched position controls.
```

한국어로는 다음과 같다.

```text
같은 gap/span 위치 구조에서도 실제 의미 골격 token과 좌우 anchor content가 있을 때
reverse expansion이 random, position-only, wrong-document, no-anchor보다 좋아진다.
```

다만 `s5_ready=false`는 유지한다. S4d는 constrained gap/span expansion이지 open-ended generation이 아니다.

다음 방향은 두 갈래다.

1. S4d 구조를 shared-condition model로 줄여 같은 control을 더 싸고 안정적으로 반복한다.
2. S5로 가기 전, constrained setting에서 generated span의 semantic content를 더 높이는 구조를 탐구한다. 후보는 span boundary-aware decoder, confidence-gated refinement, punctuation/content 분리 decoding, 그리고 longer semantic span curriculum이다.

가장 중요한 다음 질문은 이것이다.

```text
S4d에서 확인한 semantic anchor 사용 신호를 유지하면서,
실제 생성 span의 content/entity recall을 어떻게 끌어올릴 것인가?
```
