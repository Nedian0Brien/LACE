# S4c 결과: span-infilling reverse decoder

## 1. 이 실험이 측정한 것

S4c는 S4a/S4b의 autoregressive delta 생성이 반복과 형식 token 편향을 만들 수 있다는 문제를 줄이기 위해 실행한 구조 보정 실험이다.

기본 아이디어는 다음이다.

```text
input:  현재 partial state + 새로 채울 위치 marker
model:  encoder가 전체 marker/context를 한 번에 본다
target: 각 marker 위치의 원래 token id
```

즉, S4c는 자유로운 delta sequence 생성이 아니라 marker-position infilling이다. 각 새 위치에 marker를 놓고, 그 marker의 encoder hidden state에서 vocab classifier가 token id를 직접 맞힌다.

실행 정보는 다음이다.

| 항목 | 값 |
|---|---|
| phase | `v2_s4c` |
| Kaggle kernel | `dennisparknd/lace-v2-s4c-span-infilling-reverse-decoder` |
| model | `t5-small` tokenizer + custom PyTorch encoder + marker classifier |
| data | `wikitext/wikitext-2-raw-v1:train` |
| train samples | 768 |
| eval samples | 192 |
| device | `cuda` |
| ratios | `0.25 -> 0.50 -> 0.75 -> 1.00` |
| reverse epochs | 2 |
| output | `outputs/v2_s4c/lace_v2_s4c/` |

비교 조건은 다음이다.

| 조건 | 의미 |
|---|---|
| `importance_schedule` | 의미 골격 token content와 위치 marker를 함께 보고 marker 위치 token을 예측한다. |
| `random_schedule` | 같은 ratio와 token budget에서 random current state와 marker를 사용한다. |
| `position_only_schedule` | importance 위치 marker는 유지하지만 visible token content는 `pad_token_id`로 제거한다. |

## 2. 왜 중요한가

S4a와 S4b는 importance schedule에 긍정적인 신호를 줬지만, 생성 sample에는 여전히 반복과 문법 붕괴가 있었다. 이 문제가 autoregressive delta decoder에서 생긴다면, marker별 infilling 구조는 다음 장점을 줄 수 있다.

```text
1. 이미 열릴 위치를 알고 있으므로 불필요한 길이 생성이 줄어든다.
2. 각 위치를 직접 예측하므로 반복적 free generation이 줄어든다.
3. 위치별 빈칸 채우기 형태라 reverse diffusion의 insertion process에 더 가깝다.
```

따라서 S4c의 목적은 "문장을 더 자연스럽게 복원했다"가 아니라, 더 구조화된 reverse decoder가 semantic skeleton content를 실제로 활용하는지를 확인하는 것이다.

## 3. 결과가 의미하는 것

S4c는 `process_ready=true`, `overall_pass=false`, `structure_review_needed=true`, `s5_ready=false`다.

Gate 결과는 다음이다.

| Gate | 통과 | 해석 |
|---|---:|---|
| `S4C-G-RUN` | true | 세 schedule이 모두 실행됐다. |
| `S4C-G-LOSS-FINITE` | true | 모든 schedule의 loss가 유한했다. |
| `S4C-G-IMPORTANCE-BEATS-RANDOM` | true | importance score 0.2403이 random 0.1425보다 높았다. |
| `S4C-G-IMPORTANCE-BEATS-POSITION-ONLY` | false | position-only score 0.3026이 importance보다 높았다. |
| `S4C-G-MASKED-TOKEN-ACCURACY` | true | importance mask accuracy 0.1414가 random 0.1121보다 높았다. |
| `S4C-G-ENTITY-CONTENT-IMPROVEMENT` | false | content recall과 entity recall이 모두 0이었다. |
| `S4C-G-COPY-LEAKAGE` | true | importance의 copy leakage가 random보다 낮았다. |
| `S4C-G-REPETITION-NONWORSE` | true | repetition 자체는 random보다 나쁘지 않았다. |
| `S4C-G-BEST-IDENTIFIED` | true | 최고 score는 `position_only_schedule`이었다. |

Schedule별 결과는 다음이다.

| Schedule | Loss | Mask Acc | Span F1 | Content | Context Leak | Entity | Duplicate | Repetition | Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `importance_schedule` | 5.9434 | 0.1414 | 0.0000 | 0.0000 | 0.7847 | 0.0000 | 0.9347 | 0.0000 | 0.2403 |
| `random_schedule` | 6.8246 | 0.1121 | 0.0000 | 0.0000 | 0.9358 | 0.0000 | 0.9347 | 0.0000 | 0.1425 |
| `position_only_schedule` | 5.9661 | 0.1414 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9347 | 0.0000 | 0.3026 |

가장 중요한 결과는 두 가지다.

첫째, importance는 random보다 loss와 masked-token accuracy에서 낫다.

```text
importance mask accuracy: 0.1414
random mask accuracy:     0.1121
```

이것만 보면 의미 골격이 조금 더 좋은 조건을 주는 것처럼 보인다.

둘째, 하지만 position-only가 같은 mask accuracy를 냈고 score에서는 더 높다.

```text
importance score:    0.2403
position-only score: 0.3026
```

이 결과는 S4c의 핵심 부정 신호다. marker 위치와 transition 단계만으로도 importance와 같은 token accuracy를 얻었기 때문에, 현재 S4c 구조에서는 content-bearing semantic skeleton을 사용했다는 주장을 방어하기 어렵다.

또한 content recall과 entity recall이 모두 0이다. Sample을 보면 예측이 쉼표, 공백 token, 짧은 subword token에 과도하게 몰린다. 따라서 S4c는 반복 문제를 겉으로 줄였지만, 의미 있는 span infilling에는 실패했다.

## 4. 어떤 반론과 혼동 요인을 다뤘는가

첫째, "autoregressive decoder라서 반복이 생긴다"는 반론을 일부 다뤘다. S4c의 repetition rate는 0으로 기록됐지만, 이는 좋은 의미의 반복 감소라기보다 대부분 공백/쉼표/special-like token을 반복적으로 예측한 결과다. `duplicate_prediction_rate`가 0.9347로 매우 높다는 점이 이를 보여준다.

둘째, position-only confound가 강하게 드러났다. Importance와 position-only의 masked-token accuracy가 같기 때문에, 현재 marker classifier는 visible semantic content보다 위치와 target token 분포를 더 많이 이용하는 것으로 보인다.

셋째, score 자체에도 caveat가 있다. `position_only_schedule`은 visible context content가 없으므로 context copy leakage penalty를 거의 받지 않는다. 반대로 importance/random은 실제 context token이 있으므로 copy leakage가 높게 계산된다. 따라서 S4c의 aggregate score는 position-only를 과대평가할 수 있다. 하지만 이 보정을 감안하더라도 content/entity가 0이라는 사실은 구조 실패로 해석해야 한다.

넷째, marker별 독립 예측은 span coherence를 만들지 못했다. 이름, 숫자, 연속 subword, 문법적 phrase는 여러 token이 함께 맞아야 의미가 살아나는데, 현재 head는 marker마다 독립 vocab 분류에 가깝다. 그래서 span-level F1과 ROUGE-L이 모두 0이다.

다섯째, S4c는 implementation path로는 유용하다. 세 schedule, loss, output, sample, gate가 모두 정상 산출됐으므로 `process_ready=true`는 의미가 있다. 다만 연구 결과로는 `overall_pass=false`가 맞다.

## 5. 다음 실험에서 어떻게 검증할 것인가

S4c의 방어 가능한 결론은 다음이다.

```text
S4c shows that naive marker-position infilling is not enough.
It improves over random on masked-token accuracy, but the same gain is explained
by position-only control, and semantic content/entity recovery collapses to zero.
```

한국어로는 다음과 같다.

```text
위치 marker별 token 분류만으로는 semantic skeleton을 쓰는 reverse decoder가 되지 않는다.
현재 구조는 의미보다 위치·형식 token 분포를 먼저 학습한다.
```

따라서 다음 구조 개선 방향은 S4c를 그대로 밀고 가는 것이 아니라, S4b의 rollout 신호를 살리면서 S4c의 실패 원인을 반영해야 한다.

우선순위는 다음이다.

1. Primary metric에서 punctuation, whitespace, special-like token을 분리하고 content token/entity token accuracy를 별도 gate로 둔다.
2. Content word, entity, 숫자, rare token에 loss weight를 부여한다.
3. marker별 독립 분류가 아니라 contiguous span 단위 infilling decoder를 사용한다.
4. `position_only`, `same-position random`, `wrong-document/same-position` control을 S4c 계열에 계속 붙인다.
5. S4b rollout을 기준선으로 유지하고, 새 구조가 final content/entity/repetition/drift에서 S4b를 넘는지 본다.
