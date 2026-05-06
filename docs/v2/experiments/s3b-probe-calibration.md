# S3b 결과: probe calibration

## 1. 이 실험이 측정한 것

S3b는 S3a 이후 남은 해석 문제를 줄이기 위한 probe calibration 실험이다.

실험 이름은 다음이다.

```text
S3b-probe calibration
```

S3a와 가장 크게 다른 점은 reverse model을 조건별로 따로 학습하지 않았다는 것이다. S3b에서는 `attention_terminal` 입력으로 reverse model을 한 번만 학습한 뒤, 평가 시점에 입력만 바꾸었다. 따라서 조건 차이를 "각 조건에 맞게 새로 학습된 모델의 차이"가 아니라 "같은 모델이 평가 입력 변화에 얼마나 민감한가"로 볼 수 있다.

실행 정보는 다음이다.

| 항목 | 값 |
|---|---|
| phase | `v2_s3b` |
| Kaggle kernel | `dennisparknd/lace-v2-s3b-probe-calibration` |
| model | `t5-small` tokenizer + custom PyTorch encoder-decoder |
| data | `wikitext/wikitext-2-raw-v1:train` |
| train samples | 768 |
| eval samples | 192 |
| device | `cuda` |
| keep ratio | 0.25 |
| train condition | `attention_terminal` |
| output | `outputs/v2_s3b/lace_v2_s3b/` |

S3b 조건은 다음이다.

| 조건 | 의미 |
|---|---|
| `attention_terminal` | attention-received score로 고른 content terminal. 학습 입력과 같은 in-distribution 평가 조건 |
| `attention_no_position` | 같은 terminal token을 주되 위치값을 모두 0으로 둔 조건 |
| `attention_shuffled_position` | 같은 terminal token을 주되 선택된 위치값만 섞은 조건 |
| `attention_gold_anchor` | `attention_terminal`에 gold anchor를 추가한 oracle 조건 |
| `random_terminal` | 같은 token 수의 random terminal |
| `same_position_random_terminal` | attention terminal과 같은 위치에 다른 문서 token을 넣은 control |
| `position_only` | token content 없이 attention 위치 scaffold만 제공 |
| `random_terminal_predicted_anchor` | random terminal에 predicted anchor를 추가한 조건 |
| `random_terminal_gold_anchor_oracle` | random terminal에 gold anchor를 추가한 oracle 조건 |

## 2. 왜 중요한가

S3a에서는 `attention_terminal`이 `random_terminal`과 `same_position_random_terminal`보다 높았지만, `position_only`와 차이가 작았다. 또한 `random_terminal_predicted_anchor`가 최고 조건이었다. 이 결과는 content terminal 신호가 있음을 보여줬지만, 위치 scaffold와 anchor 관련 confound를 완전히 제거하지는 못했다.

S3b는 이 문제를 더 엄격하게 본다. 같은 reverse model을 고정했을 때도 `attention_terminal`이 random, same-position random, position-only보다 충분히 높아야 terminal content 사용 주장이 강해진다. 반대로 차이가 작으면 현재 probe는 content-bearing terminal보다 위치/decoder prior에 더 많이 기대는 것으로 봐야 한다.

## 3. 결과가 의미하는 것

S3b는 `diagnostic_ready=true`, `s4_ready=false`다. 모든 조건은 실행됐고 loss도 finite였지만, content terminal 우위를 방어할 만큼 gate가 깨끗하지 않았다.

Gate 결과는 다음이다.

| Gate | 통과 | 해석 |
|---|---:|---|
| `S3B-G-RUN` | true | 9개 S3b 조건이 모두 실행됐다. |
| `S3B-G-LOSS-FINITE` | true | 모든 조건의 loss가 정상 숫자로 계산됐다. |
| `S3B-G-CONTENT-BEATS-SAME-POSITION` | false | `attention_terminal`이 높긴 하지만 margin 0.0130으로 tolerance 0.02에 못 미쳤다. |
| `S3B-G-CONTENT-BEATS-POSITION-ONLY` | false | `attention_terminal`과 `position_only` 차이는 0.0033으로 매우 작다. |
| `S3B-G-CONTENT-BEATS-RANDOM` | false | `attention_terminal`과 `random_terminal` 차이는 0.0044로 작다. |
| `S3B-G-POSITION-ABLATION-DROP` | true | 위치값을 모두 0으로 만들면 score가 0.0939 떨어졌다. |
| `S3B-G-SHUFFLED-POSITION-DROP` | false | 위치값을 섞은 조건이 오히려 최고 score였다. |
| `S3B-G-ANCHOR-SANITY` | true | gold anchor oracle이 predicted anchor보다 약간 높아 S3a의 predicted-anchor anomaly는 완화됐다. |
| `S3B-G-ATTENTION-GOLD-ANCHOR-LIFT` | false | attention terminal에 gold anchor를 추가하면 오히려 크게 나빠졌다. |
| `S3B-G-BEST-IDENTIFIED` | true | 최고 조건은 `attention_shuffled_position`이었다. |

조건별 결과는 다음이다.

| 조건 | Loss | PPL | Token F1 | ROUGE-L | Keyword Recall | Entity Recall | Skeleton Coverage | Repetition | Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `attention_terminal` | 6.1202 | 454.94 | 0.1537 | 0.1399 | 0.0079 | 0.0086 | 0.0039 | 0.6051 | 0.3768 |
| `attention_no_position` | 6.3201 | 555.64 | 0.0948 | 0.0910 | 0.0023 | 0.0030 | 0.0055 | 0.4062 | 0.2828 |
| `attention_shuffled_position` | 6.1231 | 456.27 | 0.1551 | 0.1414 | 0.0086 | 0.0092 | 0.0013 | 0.6053 | 0.3799 |
| `attention_gold_anchor` | 6.2722 | 529.66 | 0.0872 | 0.0790 | 0.0027 | 0.0027 | 0.0031 | 0.3745 | 0.2674 |
| `random_terminal` | 6.1571 | 472.04 | 0.1541 | 0.1363 | 0.0013 | 0.0013 | 0.0003 | 0.5819 | 0.3724 |
| `same_position_random_terminal` | 6.1848 | 485.29 | 0.1505 | 0.1341 | 0.0007 | 0.0007 | 0.0013 | 0.6026 | 0.3638 |
| `position_only` | 6.1716 | 478.95 | 0.1586 | 0.1439 | 0.0007 | 0.0013 | 0.0000 | 0.6883 | 0.3735 |
| `random_terminal_predicted_anchor` | 6.3164 | 553.60 | 0.0723 | 0.0653 | 0.0013 | 0.0024 | 0.0015 | 0.3074 | 0.2443 |
| `random_terminal_gold_anchor_oracle` | 6.2893 | 538.76 | 0.0736 | 0.0657 | 0.0007 | 0.0007 | 0.0004 | 0.3146 | 0.2453 |

핵심 비교는 다음이다.

```text
attention_terminal:                 score 0.3768, F1 0.1537, ROUGE-L 0.1399
attention_no_position:              score 0.2828, F1 0.0948, ROUGE-L 0.0910
attention_shuffled_position:        score 0.3799, F1 0.1551, ROUGE-L 0.1414
position_only:                      score 0.3735, F1 0.1586, ROUGE-L 0.1439
random_terminal:                    score 0.3724, F1 0.1541, ROUGE-L 0.1363
same_position_random_terminal:      score 0.3638, F1 0.1505, ROUGE-L 0.1341
random_terminal_predicted_anchor:   score 0.2443, F1 0.0723, ROUGE-L 0.0653
random_terminal_gold_anchor_oracle: score 0.2453, F1 0.0736, ROUGE-L 0.0657
attention_gold_anchor:              score 0.2674, F1 0.0872, ROUGE-L 0.0790
```

긍정적인 결과는 `attention_no_position`이 명확히 떨어졌다는 점이다. content token을 그대로 두더라도 위치 channel을 모두 0으로 만들면 score가 0.3768에서 0.2828로 하락한다. 따라서 현재 probe는 위치 보조 구조를 완전히 무시하지 않는다. "위치 scaffold는 아무 의미가 없다"는 반론은 약해졌다.

두 번째 긍정적인 결과는 anchor anomaly가 S3a보다 정리됐다는 점이다. S3a에서는 predicted anchor가 gold oracle보다 높았지만, S3b에서는 `random_terminal_gold_anchor_oracle` score 0.2453이 `random_terminal_predicted_anchor` 0.2443보다 아주 조금 높았다. predicted anchor가 의미적으로 우수해서 S3a 최고 조건이 됐다는 해석은 더 약해졌다.

하지만 부정적인 결과가 더 중요하다. `attention_terminal`은 `position_only`, `random_terminal`, `same_position_random_terminal`보다 높거나 비슷하지만 margin이 작다. 특히 `position_only`와의 차이는 0.0033, `random_terminal`과의 차이는 0.0044에 불과하다. 이는 현재 reverse probe가 terminal content를 강하게 사용한다고 보기 어렵다는 뜻이다.

또한 `attention_shuffled_position`이 최고 조건이었다. 이는 "정확한 absolute position 정렬이 중요하다"는 해석을 약하게 만든다. 위치값이 모두 0이면 크게 나빠지지만, 선택된 위치값 사이에서 순서를 섞는 것은 나빠지지 않았다. 따라서 현재 모델은 token별 정밀 위치보다는 "이 입력에는 위치 signal이 있다"는 coarse prior, 또는 선택된 위치값의 분포를 쓰는 것일 수 있다.

생성 품질도 아직 낮다. `attention_terminal` repetition rate는 0.6051이고 sample에서는 "The Egyptians, the, the..." 같은 반복 패턴이 자주 보인다. 이 상태에서 Token F1/ROUGE-L 차이를 semantic generation 품질로 확대 해석하면 안 된다.

## 4. 어떤 반론과 혼동 요인을 다뤘는가

S3b는 조건별 재학습 confound를 줄였다. 같은 reverse model을 사용했기 때문에, S3a보다 "입력 정보가 실제로 복원에 쓰였는가"를 더 직접적으로 볼 수 있다.

다룬 반론은 다음과 같다.

첫째, position-only 반론은 남아 있다. `position_only`는 content token이 없는데도 score 0.3735로 `attention_terminal` 0.3768에 매우 가깝다. 이 결과는 위치 scaffold와 decoder prior가 lexical metric 상당 부분을 설명할 수 있음을 보여준다.

둘째, random terminal 반론도 완전히 사라지지 않았다. `random_terminal` score 0.3724는 `attention_terminal`보다 낮지만 차이가 작다. 같은 모델 평가에서도 content-bearing terminal이 random terminal보다 충분히 우월하다는 증거는 아직 약하다.

셋째, exact-position 반론은 오히려 강화됐다. `attention_no_position`은 낮아졌지만 `attention_shuffled_position`은 최고였다. 즉 absolute position channel 자체는 유용하지만, 정확한 token-position alignment가 쓰인다는 증거는 부족하다.

넷째, anchor 해석은 정리됐다. gold/predicted anchor 조건은 둘 다 크게 나빠졌고, random terminal에서는 gold가 predicted보다 아주 약간 높았다. 따라서 S3a의 predicted-anchor 최고 결과는 anchor predictor가 의미 anchor를 잘 복원했다는 증거가 아니라, 조건별 재학습과 probe/metric artifact일 가능성이 더 크다.

남은 caveat도 크다.

- 1 epoch의 작은 reverse probe라서 정보를 충분히 활용하지 못했을 수 있다.
- 생성 sample은 반복이 심하고, open-ended generation 품질 주장을 할 수 없다.
- `score`는 진단용 합성 지표이며, semantic similarity나 human preference를 대체하지 않는다.
- `entity_recall`은 간단한 대문자/숫자 휴리스틱이라 완전한 entity metric이 아니다.
- `attention_shuffled_position`이 좋은 이유가 실제 robustness인지, 위치 분포 prior인지, 아니면 생성 반복 패턴과 metric의 상호작용인지는 추가 분리가 필요하다.

## 5. 다음 실험에서 어떻게 검증할 것인가

S3b 이후에는 S4 constrained generation으로 바로 넘어가지 않는 것이 방어 가능하다. 현재 결과는 "semantic skeleton + positional scaffold가 random corruption보다 더 나은 reverse trajectory를 만든다"는 v2 claim을 강화하기보다, reverse probe와 metric을 더 보정해야 함을 보여준다.

다음 단계는 S3c 성격의 보정 실험이 적절하다.

1. 반복을 줄이는 decoding/학습 설정을 먼저 도입한다. 현재 repetition rate가 높아서 lexical metric 차이가 의미 복원인지 반복 template의 부산물인지 불명확하다.
2. open-ended generation보다 constrained reconstruction을 유지한다. 예를 들어 skeleton token 주변 span infill, prefix-conditioned expansion, masked span reconstruction이 더 안정적인 중간 과제다.
3. position-only가 강한 이유를 더 좁힌다. terminal 위치값의 분포만 보존하는 `position_distribution_only`, 실제 위치를 bucket으로 낮춘 조건, 같은 위치에 pad가 아닌 neutral token을 넣는 조건이 필요하다.
4. content-use metric을 추가한다. terminal token 중 target의 content/entity와 겹치는 항목이 prediction에 얼마나 살아나는지, 반복 token을 제외하고 측정해야 한다.
5. anchor는 당분간 핵심 경로에서 내린다. S3b 기준으로 anchor 추가는 오히려 distribution shift처럼 작동했다.

현재 방어 가능한 결론은 다음이다.

```text
S3b confirms that the probe uses the existence of positional scaffold,
but it does not yet confirm strong semantic terminal content use.
```

한국어로는 다음과 같이 정리할 수 있다.

```text
S3b는 위치 보조 구조가 완전히 무시되지는 않는다는 점을 확인했지만,
현재 reverse probe가 의미 terminal content를 충분히 사용한다는 증거는 아직 약하다.
```

