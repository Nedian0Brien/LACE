# S3 결과: anchor baseline comparison

## 1. 이 실험이 측정한 것

S3는 중요한 token을 forward terminal state로 직접 보존하는 방식이, random forward 뒤에 중요 anchor를 예측해서 보조 조건으로 붙이는 방식보다 더 좋은 reverse trajectory를 만드는지 측정했다.

실험 이름은 다음이다.

```text
S3-anchor baseline comparison
```

S3에서는 gold anchor를 직접 입력하지 않았다. Anchor 조건은 terminal skeleton에서 작은 anchor predictor가 예측한 anchor token을 reverse 입력에 붙였다. 따라서 `random_forward_anchor_prediction`은 oracle anchor baseline이 아니라 predicted anchor baseline이다.

실행 정보는 다음이다.

| 항목 | 값 |
|---|---|
| phase | `v2_s3` |
| Kaggle kernel | `dennisparknd/lace-v2-s3-anchor-baseline-comparison` |
| model | `t5-small` tokenizer + custom PyTorch encoder-decoder |
| data | `wikitext/wikitext-2-raw-v1:train` |
| train samples | 768 |
| eval samples | 192 |
| device | `cuda` |
| keep ratio | 0.25 |
| output | `outputs/v2_s3/lace_v2_s3/` |

## 2. 왜 중요한가

v2의 중심 주장은 중요한 token이 보조 anchor가 아니라 forward process의 terminal state라는 것이다.

S2는 의미 골격 + 위치 보조 구조가 무작위 골격보다 더 좋은 복원 학습 문제를 만들 수 있음을 보였다. 하지만 그 결과만으로는 "중요 token을 terminal state로 남기는 방식"이 "무작위 terminal에서 anchor를 예측해 붙이는 방식"보다 좋은지 알 수 없다.

S3는 이 차이를 직접 비교했다.

| 기호 | 조건 | 의미 |
|---|---|---|
| A | `random_forward_anchor_prediction` | random terminal skeleton + predicted anchor |
| B | `importance_ordered_forward_no_anchor` | importance terminal skeleton만 사용 |
| C | `importance_ordered_forward_anchor_prediction` | importance terminal skeleton + predicted anchor |
| D | `random_forward_no_anchor` | random terminal skeleton만 사용 |

## 3. 결과가 의미하는 것

S3는 `overall_pass=false`, `s4_ready=false`로 실패했다.

Gate 결과는 다음이다.

| Gate | 통과 | 해석 |
|---|---:|---|
| `S3-G-RUN` | true | 네 조건이 모두 실행됐다. |
| `S3-G-LOSS-FINITE` | true | 모든 조건의 loss가 정상 숫자로 계산됐다. |
| `S3-G-IMPORTANCE-BEATS-RANDOM` | false | B가 D보다 좋지 않았다. |
| `S3-G-TERMINAL-NOT-WORSE-THAN-ANCHOR` | true | B는 tolerance 0.02 안에서 A보다 크게 뒤처지지 않았다. |
| `S3-G-BEST-IDENTIFIED` | true | 최고 조건은 `random_forward_no_anchor`였다. |

조건별 결과는 다음이다.

| 조건 | Loss | PPL | Token F1 | ROUGE-L | Keyword Recall | Skeleton Coverage | Nonempty | Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `random_forward_anchor_prediction` | 6.1502 | 468.81 | 0.1559 | 0.1460 | 0.0020 | 0.0026 | 1.0000 | 0.4418 |
| `importance_ordered_forward_no_anchor` | 6.1387 | 463.47 | 0.1515 | 0.1441 | 0.0141 | 0.0084 | 1.0000 | 0.4356 |
| `importance_ordered_forward_anchor_prediction` | 6.1180 | 453.97 | 0.1366 | 0.1252 | 0.0158 | 0.0083 | 1.0000 | 0.4023 |
| `random_forward_no_anchor` | 6.1391 | 463.65 | 0.1612 | 0.1455 | 0.0039 | 0.0083 | 1.0000 | 0.4467 |

핵심 비교는 다음이다.

```text
A random_forward_anchor_prediction:       score 0.4418, F1 0.1559, ROUGE-L 0.1460
B importance_ordered_forward_no_anchor:   score 0.4356, F1 0.1515, ROUGE-L 0.1441
C importance_ordered_forward_anchor_pred: score 0.4023, F1 0.1366, ROUGE-L 0.1252
D random_forward_no_anchor:               score 0.4467, F1 0.1612, ROUGE-L 0.1455
```

따라서 S3의 방어 가능한 결론은 다음이다.

> 현재 S3 설정에서는 importance terminal state가 random terminal state보다 더 좋은 reverse trajectory를 만든다는 증거를 얻지 못했다.

다만 B는 A와 큰 차이로 밀리지는 않았다. `B score 0.4356`은 `A score 0.4418`과 tolerance 0.02 안에 있다. 따라서 "anchor prediction baseline이 importance terminal을 압도했다"는 결과도 아니다.

## 4. 어떤 반론과 혼동 요인을 다뤘는가

S3는 "중요 token은 terminal state가 아니라 그냥 예측해서 붙이면 되는 anchor가 아닌가"라는 반론을 다뤘다.

그 결과 predicted anchor baseline인 A는 B보다 약간 높았지만, 최고 조건은 A가 아니라 D였다. 이는 anchor prediction이 강해서 B를 이긴 것이 아니라, 이 짧은 복원 probe에서는 random terminal 자체가 예상보다 강하게 작동했을 가능성을 보여준다.

Anchor predictor 품질도 낮았다.

| Terminal | Loss | PPL | Anchor Token F1 | Anchor ROUGE-L | Nonempty |
|---|---:|---:|---:|---:|---:|
| `importance` | 7.2944 | 1471.99 | 0.0447 | 0.0443 | 0.9896 |
| `random` | 7.4352 | 1694.61 | 0.0182 | 0.0182 | 0.9635 |

즉 S3는 강한 anchor predictor baseline과 비교한 것이 아니다. A가 B와 비슷하거나 약간 높은 것은 "anchor prediction이 충분히 해결책이다"라기보다, 현재 reconstruction proxy와 모델 구조에서 terminal state 차이가 충분히 드러나지 않았다는 신호로 해석해야 한다.

## 5. 다음 실험에서 어떻게 검증할 것인가

S4 constrained generation으로 바로 넘어가면 안 된다. S3는 `s4_ready=false`다.

다음 단계는 S3 진단 실험이어야 한다. 특히 다음을 분리해야 한다.

1. `random_forward_no_anchor`가 왜 가장 높은 score를 얻었는가.
2. importance terminal의 keyword recall과 skeleton coverage 이점이 왜 Token F1/ROUGE-L 우위로 이어지지 않았는가.
3. predicted anchor가 너무 약해서 anchor baseline 비교가 불충분했는가.
4. 현재 custom reconstruction model과 1 epoch 예산이 terminal quality 차이를 충분히 반영하지 못하는가.

다음 후보는 `S3a-terminal diagnostic`이다. 여기서는 `gold_anchor_oracle`, `same_position_random`, `position_only`, `idf_terminal`, `attention_terminal`, `random_terminal`을 함께 비교해 terminal 정보량과 위치 편향, anchor 예측 실패를 분리한다.

아직 주장하면 안 되는 것은 다음이다.

- semantic skeleton terminal state가 anchor prediction baseline보다 낫다는 주장
- S3 결과로 S4/S5 generation 검증에 바로 넘어가도 된다는 주장
- random corruption baseline이 충분히 약하다는 주장
