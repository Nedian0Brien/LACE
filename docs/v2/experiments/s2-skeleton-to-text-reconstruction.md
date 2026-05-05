# S2 결과: 의미 골격-문장 복원 학습

## 1. 이 실험이 측정한 것

S2는 의미 골격에서 원문으로 되돌아가는 짧은 역방향 복원 학습이 가능한지 측정했다.

이번 실험은 open-ended generation이 아니다. `t5-small`을 조건별로 1 epoch 미세조정하고, held-out 문장에서 teacher-forced loss와 짧은 생성 복원 지표를 비교했다.

실행 정보는 다음이다.

| 항목 | 값 |
|---|---|
| phase | `v2_s2` |
| Kaggle kernel | `dennisparknd/lace-v2-s2-skeleton-to-text-reconstruction` |
| model | `t5-small` |
| data | `wikitext/wikitext-2-raw-v1:train` |
| train samples | 768 |
| eval samples | 192 |
| device | `cuda` |
| keep ratio | 0.25 |
| epochs | 1 |
| output | `outputs/v2_s2/lace_v2_s2/` |

## 2. 왜 중요한가

S0는 의미 골격 자체가 무작위/균일 baseline보다 의미 보존 신호를 갖는지 확인했다. S1은 frozen encoder 검색 평가에서 의미 골격이 원문 식별 단서로 쓰일 수 있음을 확인했다.

하지만 LACE의 핵심 주장은 검색이 아니라 역방향 궤적이다. S2는 같은 모델 구조와 학습 예산에서 의미 골격 + 위치 보조 구조가 무작위 골격이나 위치 전용 입력보다 더 나은 복원 학습 문제를 만드는지 확인한다.

## 3. 결과가 의미하는 것

S2는 `overall_pass=true`, `next_ready=true`로 통과했다.

주요 gate는 모두 통과했다.

| Gate | 통과 | 해석 |
|---|---:|---|
| `S2-G-RUN` | true | 5개 학습 조건과 3개 attention control이 평가됐다. |
| `S2-G-LOSS-FINITE` | true | 모든 주요 조건의 teacher-forced loss가 유한했다. |
| `S2-G-NONEMPTY-GENERATION` | true | 생성물이 거의 항상 비어 있지 않았다. |
| `S2-G-ATTENTION-BEATS-RANDOM` | true | attention 의미 골격이 무작위 골격보다 token F1/ROUGE-L 기준으로 강했다. |
| `S2-G-ATTENTION-BEATS-POSITION` | true | attention 의미 골격이 위치 전용 입력보다 강했다. |
| `S2-G-WRONG-DOC-DROPS` | true | attention 모델에 다른 문서 골격을 넣으면 성능이 크게 떨어졌다. |

학습 조건별 결과는 다음이다.

| 조건 | Loss | PPL | Token F1 | ROUGE-L | Keyword Recall | Skeleton Coverage | Nonempty |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention_scaffold` | 2.7333 | 15.38 | 0.3830 | 0.3117 | 0.3735 | 0.6783 | 1.0000 |
| `idf_scaffold` | 2.4851 | 12.00 | 0.3811 | 0.3053 | 0.4103 | 0.6482 | 1.0000 |
| `position_prior_scaffold` | 2.7386 | 15.47 | 0.3430 | 0.3206 | 0.8504 | 0.9617 | 1.0000 |
| `random_scaffold` | 3.2445 | 25.65 | 0.2286 | 0.1789 | 0.2138 | 0.4146 | 0.9948 |
| `position_only` | 3.8216 | 45.68 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

attention 모델에 대조 입력을 넣은 결과는 다음이다.

| 조건 | Loss | PPL | Token F1 | ROUGE-L | Keyword Recall | Skeleton Coverage | Nonempty |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention_position_only` | 4.1276 | 62.03 | 0.0244 | 0.0240 | 0.0000 | 0.0000 | 1.0000 |
| `attention_same_position_random` | 4.2916 | 73.09 | 0.1203 | 0.0943 | 0.0052 | 0.5982 | 1.0000 |
| `attention_wrong_document` | 4.2939 | 73.25 | 0.1222 | 0.0944 | 0.0072 | 0.5972 | 0.9896 |

핵심 비교는 세 가지다.

첫째, `attention_scaffold`는 `random_scaffold`보다 좋다. Token F1은 0.3830 대 0.2286이고, ROUGE-L은 0.3117 대 0.1789다. 이는 의미 골격 + 위치 보조 구조가 같은 token 수의 무작위 골격보다 더 좋은 복원 입력이라는 S2 수준의 증거다.

둘째, `attention_scaffold`는 위치 전용 입력보다 압도적으로 좋다. `position_only`는 nonempty 생성은 하지만 token F1과 ROUGE-L이 0이다. 위치 표지만으로는 원문 복원 학습이 되지 않았다.

셋째, wrong-document와 same-position random control에서 성능이 크게 하락했다. attention 모델의 correct 입력 token F1은 0.3830인데, wrong-document는 0.1222, same-position random은 0.1203이다. 이는 모델이 단순한 문장 prior만 쓰는 것이 아니라 입력 골격의 내용에 민감하다는 신호다.

## 4. 어떤 반론과 혼동 요인을 줄였는가

S2는 "무작위 token도 같은 예산이면 충분하지 않은가"라는 반론을 줄였다. `random_scaffold`는 생성물이 비어 있지는 않았지만, loss와 token F1/ROUGE-L에서 `attention_scaffold`보다 낮았다.

S2는 "위치 정보만으로 되는 것 아닌가"라는 반론도 줄였다. `position_only`와 `attention_position_only`는 loss가 높고 token-level 복원이 거의 되지 않았다.

다만 위치 보조 구조의 강함은 계속 남아 있다. `position_prior_scaffold`는 token F1에서는 attention보다 낮지만 ROUGE-L이 0.3206으로 높고, keyword recall은 0.8504로 매우 높다. 이는 WikiText 문장 앞부분 위치 편향이 여전히 강하다는 뜻이다.

또한 `idf_scaffold`는 loss 2.4851로 가장 낮고 keyword recall도 attention보다 높다. 따라서 현재 결과는 "attention scorer가 항상 최고"라는 주장이 아니라, "중요도 기반 의미 골격 계열이 무작위 골격보다 복원 학습에 유리하다"는 주장으로 해석해야 한다.

## 5. 다음 실험에서 어떻게 검증할 것인가

S2는 다음 단계로 넘어갈 수 있다. 다만 다음 단계는 open-ended generation으로 바로 크게 뛰기보다, S3 anchor baseline comparison을 먼저 수행하는 것이 더 방어 가능하다.

S3에서 확인할 질문은 다음이다.

> 중요한 token을 terminal state로 남기는 방식이, random forward 뒤 anchor를 보조 조건으로 예측하거나 제공하는 방식보다 나은가?

S3에서는 최소한 다음 비교가 필요하다.

- 무작위 골격 + anchor 보조 조건
- 의미 골격 terminal state + anchor 없음
- 의미 골격 terminal state + anchor 보조 조건
- 무작위 골격 + anchor 없음

S2의 성공 주장은 다음 정도로 제한한다.

> 의미 골격 + 위치 보조 구조는 짧은 skeleton-to-text 복원 학습에서 무작위 골격과 위치 전용 입력보다 더 좋은 역방향 학습 문제를 만들었다.

아직 주장하면 안 되는 것은 다음이다.

- open-ended generation 성공
- attention scorer의 최종 우위
- 위치 보조 구조 없이 의미 골격만으로 충분하다는 주장
- 사람 평가 수준의 문장 품질 개선
