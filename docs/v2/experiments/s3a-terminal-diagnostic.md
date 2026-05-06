# S3a 결과: terminal diagnostic

## 1. 이 실험이 측정한 것

S3a는 S3에서 `random_forward_no_anchor`가 가장 높은 score를 얻은 이유를 분해하기 위한 진단 실험이다.

실험 이름은 다음이다.

```text
S3a-terminal diagnostic
```

S3a는 S3의 custom PyTorch encoder-decoder와 `sinusoidal_absolute` 위치 보조 구조를 유지했다. 달라진 점은 terminal 조건을 넓혀 content terminal, same-position random, position-only, predicted anchor, gold-anchor oracle을 같은 짧은 reverse probe에서 비교한 것이다.

실행 정보는 다음이다.

| 항목 | 값 |
|---|---|
| phase | `v2_s3a` |
| Kaggle kernel | `dennisparknd/lace-v2-s3a-terminal-diagnostic` |
| model | `t5-small` tokenizer + custom PyTorch encoder-decoder |
| data | `wikitext/wikitext-2-raw-v1:train` |
| train samples | 768 |
| eval samples | 192 |
| device | `cuda` |
| keep ratio | 0.25 |
| output | `outputs/v2_s3a/lace_v2_s3a/` |

## 2. 왜 중요한가

S3는 semantic terminal state가 random terminal보다 더 좋은 reverse trajectory를 만든다는 증거를 주지 못했다. 하지만 그 실패가 semantic skeleton 가설 자체의 실패인지, random terminal baseline의 위치/길이 prior 때문인지, anchor predictor 병목 때문인지, 또는 lexical metric과 reverse probe가 둔감해서인지 분리되지 않았다.

S3a는 이 원인들을 분리한다.

| 조건 | 의미 |
|---|---|
| `attention_terminal` | attention-received score로 고른 content terminal |
| `idf_terminal` | corpus-level IDF로 고른 terminal |
| `random_terminal` | 같은 token 수의 random terminal |
| `same_position_random_terminal` | attention terminal과 같은 위치에 다른 문서 token을 넣은 control |
| `position_only` | token content 없이 attention 위치 scaffold만 제공 |
| `random_terminal_predicted_anchor` | random terminal에서 예측한 anchor를 붙인 조건 |
| `random_terminal_gold_anchor_oracle` | random terminal에 gold anchor를 직접 붙인 oracle control |

## 3. 결과가 의미하는 것

S3a는 `diagnostic_ready=true`, `s4_ready=false`다. 필요한 진단 조건은 모두 실행됐고 loss도 finite였지만, S4로 넘어갈 만큼 해석이 깨끗하지는 않다.

Gate 결과는 다음이다.

| Gate | 통과 | 해석 |
|---|---:|---|
| `S3A-G-RUN` | true | 7개 진단 조건이 모두 실행됐다. |
| `S3A-G-LOSS-FINITE` | true | 모든 조건의 loss가 정상 숫자로 계산됐다. |
| `S3A-G-CONTENT-BEATS-SAME-POSITION` | true | best content terminal이 same-position random보다 높았다. |
| `S3A-G-CONTENT-BEATS-POSITION-ONLY` | true | best content terminal이 position-only보다 높았다. |
| `S3A-G-ORACLE-LIFT` | false | gold anchor oracle이 predicted anchor보다 높지 않았다. |
| `S3A-G-RANDOM-EXPLAINED` | true | random terminal은 best content terminal보다 낮았다. |
| `S3A-G-BEST-IDENTIFIED` | true | 최고 조건은 `random_terminal_predicted_anchor`였다. |

조건별 결과는 다음이다.

| 조건 | Loss | PPL | Token F1 | ROUGE-L | Keyword Recall | Skeleton Coverage | Nonempty | Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `attention_terminal` | 6.1199 | 454.82 | 0.1540 | 0.1406 | 0.0079 | 0.0039 | 1.0000 | 0.4351 |
| `idf_terminal` | 6.1650 | 475.82 | 0.1437 | 0.1359 | 0.0065 | 0.0000 | 1.0000 | 0.4192 |
| `random_terminal` | 6.1576 | 472.27 | 0.1354 | 0.1262 | 0.0078 | 0.0066 | 1.0000 | 0.4014 |
| `same_position_random_terminal` | 6.1690 | 477.71 | 0.1066 | 0.0969 | 0.0026 | 0.0008 | 1.0000 | 0.3429 |
| `position_only` | 6.1828 | 484.33 | 0.1534 | 0.1349 | 0.0046 | 0.0000 | 1.0000 | 0.4275 |
| `random_terminal_predicted_anchor` | 6.1412 | 464.61 | 0.1613 | 0.1476 | 0.0026 | 0.0012 | 1.0000 | 0.4489 |
| `random_terminal_gold_anchor_oracle` | 6.0995 | 445.65 | 0.1482 | 0.1224 | 0.0085 | 0.0060 | 1.0000 | 0.4115 |

핵심 비교는 다음이다.

```text
attention_terminal:                score 0.4351, F1 0.1540, ROUGE-L 0.1406
random_terminal:                   score 0.4014, F1 0.1354, ROUGE-L 0.1262
same_position_random_terminal:     score 0.3429, F1 0.1066, ROUGE-L 0.0969
position_only:                     score 0.4275, F1 0.1534, ROUGE-L 0.1349
random_terminal_predicted_anchor:  score 0.4489, F1 0.1613, ROUGE-L 0.1476
random_terminal_gold_anchor_oracle: score 0.4115, F1 0.1482, ROUGE-L 0.1224
```

방어 가능한 긍정 신호는 `attention_terminal`이 `random_terminal`과 `same_position_random_terminal`보다 높았다는 점이다. S3에서 보였던 random terminal의 강함은 S3a에서는 재현되지 않았다. 같은 위치에 다른 문서 token을 넣으면 score가 0.3429까지 떨어졌으므로, 위치만 같은 random content는 충분하지 않았다.

하지만 중요한 caveat가 있다. `position_only` score 0.4275는 `attention_terminal` 0.4351과 매우 가깝다. 따라서 content terminal의 우위는 존재하지만 작다. 위치 scaffold와 corpus-level token prior가 여전히 강하게 작동한다.

또 다른 caveat는 `random_terminal_predicted_anchor`가 score 0.4489로 최고였다는 점이다. 그런데 anchor predictor 자체의 품질은 낮았다.

| Terminal | Loss | PPL | Anchor Token F1 | Anchor ROUGE-L | Nonempty |
|---|---:|---:|---:|---:|---:|
| `attention` | 7.3084 | 1492.72 | 0.0381 | 0.0377 | 1.0000 |
| `random` | 7.4775 | 1767.85 | 0.0155 | 0.0155 | 1.0000 |

따라서 predicted anchor가 최고였다는 결과를 "anchor predictor가 의미 anchor를 잘 복원했다"로 해석하면 안 된다. 더 가능성 높은 해석은 predicted anchor 문자열이 reverse model에 유리한 표면 prior 또는 regularization처럼 작동했거나, 현재 lexical metric이 그런 표면 prior에 민감했다는 것이다.

## 4. 어떤 반론과 혼동 요인을 다뤘는가

S3a는 세 가지 반론을 줄였다.

첫째, "random terminal이 semantic terminal만큼 강하다"는 S3 신호는 약해졌다. S3a에서는 `attention_terminal` score 0.4351이 `random_terminal` 0.4014보다 높았다.

둘째, "같은 위치만 맞추면 된다"는 반론도 약해졌다. `same_position_random_terminal`은 score 0.3429로 가장 낮은 축에 속했다.

셋째, "position-only로 충분하다"는 반론은 줄었지만 사라지지는 않았다. `position_only`는 score 0.4275로 `attention_terminal`에 근접했다. 이는 위치 scaffold와 모델 prior가 여전히 강한 confound라는 뜻이다.

S3a가 해결하지 못한 것은 anchor 해석이다. `gold_anchor_oracle`은 predicted anchor보다 낮았다. 이 결과는 oracle anchor가 쓸모없다는 뜻이라기보다, 현재 reverse probe와 1 epoch 학습 설정이 추가 anchor 정보를 안정적으로 활용하지 못한다는 신호로 보는 편이 안전하다.

## 5. 다음 실험에서 어떻게 검증할 것인가

다음 단계는 S4 constrained generation으로 바로 가기보다 S3b 또는 S3a 후속 진단이 더 방어 가능하다.

우선 다음을 분리해야 한다.

1. `position_only`가 왜 `attention_terminal`에 가까운가.
2. predicted anchor가 왜 gold anchor oracle보다 높은가.
3. score가 Token F1/ROUGE-L의 표면 prior에 끌리는가.
4. 같은 reverse model을 조건별로 따로 학습하는 방식이 anchor oracle 비교를 불안정하게 만드는가.

후속 후보는 `S3b-probe calibration`이다. 여기서는 같은 학습 모델에 평가 입력만 바꾸는 ablation, gold anchor 길이/segment ablation, position-only matched control, 반복률과 entity recall 같은 metric을 추가한다.

아직 주장하면 안 되는 것은 다음이다.

- S3a 결과만으로 S4 generation으로 넘어갈 수 있다는 주장
- predicted anchor가 의미 anchor를 잘 예측했다는 주장
- position scaffold confound가 해결됐다는 주장
- attention terminal이 최종 scorer로 확정됐다는 주장
