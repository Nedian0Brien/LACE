# S4g 결과: pretrained decoder semantic span expansion

## 1. 이 실험이 측정한 것

S4g는 S4e 이후 남은 가장 직접적인 반론을 확인하기 위해 실행했다. S4e는 shared-condition model에서도 `importance_schedule`이 strict control을 모두 이겼지만, generated span 자체의 content/entity recall은 거의 무너졌고 artifact rate가 높았다. 이때 남는 반론은 다음이었다.

```text
span generation이 나쁜 이유가 구조 때문인가?
아니면 custom decoder가 너무 작아서 언어 prior가 부족한 것인가?
```

S4g는 S4d/S4e의 six-control gap/span expansion 장치를 유지하되, 작은 custom decoder를 `AutoModelForSeq2SeqLM` 기반 pretrained `t5-small` decoder로 바꿨다. 입력은 구조화된 text prompt로 직렬화했고, target은 이번 단계에서 새로 unmask될 span text만으로 두었다.

```text
input:  condition id/name
        transition ratio
        span positions and span length
        left/right anchor positions and distances
        current skeleton text

target: newly unmasked span text
```

실행 정보는 다음이다.

| 항목 | 값 |
|---|---|
| phase | `v2_s4g` |
| Kaggle kernel | `dennisparknd/lace-v2-s4g-pretrained-decoder-span` version 1 |
| model | `t5-small` via `AutoModelForSeq2SeqLM` |
| data | `wikitext/wikitext-2-raw-v1:train` |
| train samples | 768 |
| eval samples | 192 |
| train examples per schedule cap | 3,000 |
| eval examples per schedule cap | 512 |
| combined train examples | 18,000 |
| rollout eval samples | 48 |
| device | `cuda` |
| ratios | `0.25 -> 0.50 -> 0.75 -> 1.00` |
| reverse epochs | 1 |
| output | `outputs/v2_s4g/lace_v2_s4g/` |

비교 조건은 S4e와 동일하다.

| 조건 | 의미 |
|---|---|
| `importance_schedule` | 실제 importance skeleton token content와 left/right anchor를 사용한다. |
| `random_schedule` | 같은 ratio와 token budget에서 random skeleton을 사용한다. |
| `position_only_schedule` | importance 위치와 gap 구조는 주되 token content를 제거한다. |
| `same_position_random_schedule` | importance와 같은 위치에 같은 문서의 random token content를 넣는다. |
| `wrong_document_same_position_schedule` | importance와 같은 위치에 다른 문서의 skeleton token content를 넣는다. |
| `no_anchor_gap_only_schedule` | current skeleton 없이 이번 span marker 위치만 준다. |

## 2. 왜 중요한가

S4g의 목적은 성능을 최종적으로 높이는 것이 아니라, S4e의 실패 원인을 분리하는 것이다.

좋은 결과는 다음을 동시에 만족해야 했다.

```text
1. pretrained decoder를 써도 importance가 strict controls를 이긴다.
2. S4e보다 span content/entity recall이 오른다.
3. S4e보다 artifact rate가 낮아진다.
4. final rollout 품질은 S4d/S4e보다 크게 나빠지지 않는다.
```

이 네 가지가 동시에 성립하면, S4e 실패는 주로 작은 decoder의 언어 prior 부족으로 볼 수 있다. 반대로 pretrained decoder에서도 content/entity가 0에 머물고 artifact가 높으면, 문제는 모델 크기만이 아니라 target 구성과 reverse 구조에 있다고 보는 편이 더 방어 가능하다.

## 3. 결과가 의미하는 것

S4g는 `process_ready=true`, `overall_pass=false`, `structure_review_needed=true`, `s5_ready=false`다.

Gate 결과는 다음이다.

| Gate | 통과 | 핵심 수치 |
|---|---:|---|
| `S4G-G-RUN` | true | 여섯 schedule 모두 실행 |
| `S4G-G-LOSS-FINITE` | true | 모든 eval loss 유한 |
| `S4G-G-SHARED-CONDITION-MODEL` | true | 하나의 pretrained shared-condition model 사용 |
| `S4G-G-PRETRAINED-DECODER` | true | `t5-small` |
| `S4G-G-IMPORTANCE-BEATS-RANDOM` | true | 0.7314 > 0.6404 |
| `S4G-G-IMPORTANCE-BEATS-POSITION-ONLY` | true | 0.7314 > -0.0237 |
| `S4G-G-IMPORTANCE-BEATS-SAME-POSITION-RANDOM` | true | 0.7314 > 0.4580 |
| `S4G-G-WRONG-DOC-DROPS` | true | 0.7314 > -0.0021 |
| `S4G-G-NO-ANCHOR-DROPS` | true | 0.7314 > -0.0513 |
| `S4G-G-SPAN-CONTENT-GAIN` | false | 0.0000 < S4e 0.0029, S4d 0.0172 |
| `S4G-G-SPAN-ENTITY-GAIN` | false | 0.0000 < S4e 0.0013, S4d 0.0047 |
| `S4G-G-FINAL-NONREGRESSION` | true | final content/drift/repetition은 S4d/S4e 대비 비퇴행 |
| `S4G-G-ARTIFACT-CHECK` | false | artifact 0.9961 > S4e 0.6906 |
| `S4G-G-BEST-IDENTIFIED` | true | best schedule은 `importance_schedule` |

Teacher-forced span 예측 결과는 다음이다.

| Schedule | Loss | TF Delta Acc | Delta F1 | Content Recall | Entity Recall | Artifact |
|---|---:|---:|---:|---:|---:|---:|
| `importance_schedule` | 3.0017 | 0.5271 | 0.0286 | 0.0000 | 0.0000 | 0.9961 |
| `random_schedule` | 3.4987 | 0.4952 | 0.0078 | 0.0000 | 0.0000 | 0.9961 |
| `position_only_schedule` | 3.0836 | 0.5183 | 0.0117 | 0.0000 | 0.0000 | 0.7832 |
| `same_position_random_schedule` | 2.8642 | 0.5222 | 0.0385 | 0.0000 | 0.0000 | 0.9668 |
| `wrong_document_same_position_schedule` | 3.2072 | 0.4955 | 0.0270 | 0.0000 | 0.0000 | 0.9902 |
| `no_anchor_gap_only_schedule` | 3.2250 | 0.5178 | 0.0085 | 0.0000 | 0.0000 | 0.7012 |

Rollout final 결과는 다음이다.

| Schedule | Final F1 | ROUGE-L | Content | Original Content | Entity | Repetition | Drift | Rollout Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `importance_schedule` | 0.3643 | 0.3545 | 0.3432 | 0.3503 | 0.2901 | 0.0015 | 0.6311 | 0.7314 |
| `random_schedule` | 0.3554 | 0.3468 | 0.2582 | 0.2623 | 0.2260 | 0.0016 | 0.6968 | 0.6404 |
| `position_only_schedule` | 0.0215 | 0.0215 | 0.0000 | 0.0000 | 0.0000 | 0.0553 | 0.9478 | -0.0237 |
| `same_position_random_schedule` | 0.2996 | 0.1656 | 0.1924 | 0.1954 | 0.1817 | 0.0068 | 0.7727 | 0.4580 |
| `wrong_document_same_position_schedule` | 0.0373 | 0.0342 | 0.0052 | 0.0052 | 0.0052 | 0.0994 | 0.9407 | -0.0021 |
| `no_anchor_gap_only_schedule` | 0.0273 | 0.0273 | 0.0000 | 0.0000 | 0.0000 | 0.3884 | 0.9469 | -0.0513 |

핵심 결과는 두 층으로 갈라진다.

첫째, process-level signal은 여전히 살아 있다.

```text
importance rollout score:       0.7314
random rollout score:           0.6404
same-position random score:     0.4580
wrong-document score:          -0.0021
no-anchor score:               -0.0513
```

즉 pretrained decoder 조건에서도 실제 semantic skeleton content와 anchor가 있는 경우 final rollout은 random, same-position random, wrong-document, no-anchor보다 높다.

둘째, S4g가 해결하려던 span-level semantic generation은 실패했다.

```text
S4d span content recall: 0.0172
S4e span content recall: 0.0029
S4g span content recall: 0.0000

S4d span entity recall:  0.0047
S4e span entity recall:  0.0013
S4g span entity recall:  0.0000

S4e artifact rate:       0.6906
S4g artifact rate:       0.9961
```

샘플에서도 target이 `brown`, `Australia`, `character`, `10` 같은 content token일 때 prediction은 주로 쉼표, `s`, 빈 조각에 가까운 token으로 나왔다. 따라서 S4g의 높은 final content recall은 새 span이 의미 있게 생성됐다는 증거가 아니라, current skeleton에 이미 남아 있던 content가 final state metric에 반영된 효과로 해석해야 한다.

## 4. 긍정적인 점, 부정적인 점, 남은 모호성

긍정적인 점은 S4d/S4e에서 보인 strict control 대비 importance 우위가 pretrained decoder 조건에서도 유지됐다는 것이다. 특히 `position_only_schedule`, `wrong_document_same_position_schedule`, `no_anchor_gap_only_schedule`이 거의 무너졌다는 점은 여전히 중요하다. 이는 위치 scaffold만으로는 final semantic state를 만들기 어렵고, 문맥에 맞는 semantic skeleton content와 anchor가 필요하다는 해석을 지지한다.

또한 repetition은 크게 낮아졌다.

```text
S4d repetition: 0.0469
S4e repetition: 0.0375
S4g repetition: 0.0015
```

하지만 이 값은 좋은 문장 생성을 의미하지 않는다. 반복이 줄어든 이유는 모델이 풍부한 span을 생성한 것이 아니라, 쉼표나 짧은 token만 생성해 문장을 거의 확장하지 못했기 때문일 수 있다.

부정적인 점은 더 크다. S4g는 pretrained decoder를 써도 generated span content/entity를 전혀 회복하지 못했다. 오히려 artifact rate는 0.9961로 S4e보다 나빠졌다. 이는 "모델이 너무 작아서 실패했다"는 설명만으로는 부족하다는 뜻이다.

방어 가능한 해석은 다음이다.

```text
Pretrained decoder improves neither span content nor artifact behavior under the current text-prompt span objective.
The remaining bottleneck is likely target construction and reverse expansion structure, not only decoder scale.
```

한국어로는 다음과 같다.

```text
현재 span objective에서는 pretrained decoder를 붙여도 새 span이 의미 단위로 생성되지 않는다.
따라서 병목은 단순한 모델 크기보다 span target 구성과 anchor-conditioned reverse 구조에 있다.
```

남은 모호성은 S4g가 `t5-small` 1 epoch, 18,000 balanced examples cap으로 실행됐다는 점이다. 더 오래 학습하거나 더 큰 모델을 쓰면 일부 개선될 가능성은 남아 있다. 그러나 artifact 0.9961과 content/entity 0.0000은 너무 강한 실패 신호이므로, 지금 S5 scale-up으로 넘어가는 것은 연구적으로 방어하기 어렵다.

## 5. 다음 실험에서 어떻게 검증할 것인가

S4g의 결론은 다음이다.

```text
S4g keeps the importance trajectory advantage under a pretrained decoder,
but rejects the hypothesis that the S4e span collapse was mainly a tiny-decoder language-prior problem.
```

다음 단계는 S5 open-ended generation이 아니라, generated span 자체가 의미 정보를 담도록 target과 구조를 다시 설계하는 것이다. 우선순위는 다음이다.

1. subword 조각 단위 span을 그대로 맞히는 target을 줄이고, 단어 또는 의미 chunk 단위 target으로 재구성한다.
2. content span과 function/punctuation span을 분리해 서로 다른 실패를 같은 metric으로 섞지 않는다.
3. left/right anchor를 text prompt에 쓰는 수준을 넘어, span query가 anchor representation을 직접 참고하는 구조를 검토한다.
4. final rollout score와 generated-span content score를 계속 분리한다.
5. strict controls는 S4g와 동일하게 유지한다.

가장 중요한 다음 질문은 이것이다.

```text
semantic skeleton의 final rollout 우위를 유지하면서,
새로 unmask되는 span 자체가 실제 content/entity를 담도록 만드는 target 단위와 decoder 구조는 무엇인가?
```
