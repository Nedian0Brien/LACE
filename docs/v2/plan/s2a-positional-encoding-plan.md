# S2a 계획: positional encoding 비교

## 1. 이 실험이 측정하는 것

S2a는 S2의 `front/middle/back` 위치 tag를 더 정석적인 positional encoding으로 바꿨을 때, 의미 골격-문장 복원 성능이 어떻게 달라지는지 측정한다.

실험 이름은 다음으로 고정한다.

```text
S2a-positional encoding
```

검증할 연구 질문은 [research-questions.md](../research-questions.md)의 RQ2다.

> 중요도 기반 forward는 무작위 손상보다 좋은 reverse trajectory를 만드는가?

S2a에서는 의미 골격 token 선택은 `attention_received` 상위 token으로 고정하고, 위치 표현만 바꾼다. 따라서 이 실험은 scorer 비교가 아니라 positional scaffold 비교다.

## 2. 왜 중요한가

S2에서 `attention_scaffold`는 `random_scaffold`보다 복원 지표가 좋았다. 하지만 S2의 위치 보조 구조는 `front`, `middle`, `back`이라는 매우 거친 문자열 tag였다.

이 방식은 control을 만들기 쉽지만 일반적인 transformer 위치 부호화라고 보기 어렵다. S3 anchor baseline comparison으로 넘어가기 전에 위치 보조 구조를 더 정교하게 만들 수 있는지 확인해야 한다.

## 3. 비교 조건

S2a는 같은 attention 의미 골격을 사용하고, 위치 처리 방식만 비교한다.

| 조건 | 의미 |
|---|---|
| `no_position` | 의미 골격 token만 사용하고 위치 정보를 넣지 않는다. |
| `coarse_bins` | S2의 `front/middle/back` tag에 해당하는 coarse 위치 embedding을 사용한다. |
| `learned_absolute` | 원래 token index에 대한 learned positional embedding을 더한다. |
| `sinusoidal_absolute` | 원래 token index에 대한 sinusoidal positional encoding을 더한다. |
| `relative_position_bias` | attention score에 상대 위치 bias를 더한다. |
| `rotary_position` | query/key에 rotary position embedding을 적용한다. |

`learned_absolute`, `sinusoidal_absolute`, `relative_position_bias`, `rotary_position`이 이번 실험의 정식 positional encoding 후보군이다. `no_position`과 `coarse_bins`는 해석용 baseline이다.

## 4. Metric과 gate

| Metric/Gate | 측정 | 통과 기준 |
|---|---|---|
| `S2A-G-RUN` | 모든 조건 실행과 산출물 생성 | metrics/summary/samples 생성 |
| `S2A-G-LOSS-FINITE` | reconstruction loss가 정상 숫자인가 | 모든 조건 loss가 finite |
| `S2A-G-STANDARD-BEATS-NONE` | 정식 위치 부호화가 위치 없음보다 나은가 | 네 정식 후보 중 하나가 `no_position`보다 비교 점수 우위 |
| `S2A-G-STANDARD-BEATS-COARSE` | 정식 위치 부호화가 S2 coarse tag보다 나은가 | 네 정식 후보 중 하나가 `coarse_bins`보다 비교 점수 우위 |
| `S2A-G-BEST-IDENTIFIED` | 다음 실험에 넘길 후보가 정해졌는가 | 최고 조건 이름과 metric이 기록됨 |

주요 지표는 teacher-forced loss, token F1, ROUGE-L, keyword recall, skeleton coverage, nonempty rate다. S2a의 비교 점수는 token F1과 ROUGE-L을 우선하되, 초기 생성 지표가 너무 낮을 경우를 대비해 teacher-forced loss도 보조 신호로 반영한다.

## 5. 좋은 결과와 나쁜 결과

좋은 결과는 정식 positional encoding 후보 중 하나가 `coarse_bins`와 `no_position`보다 더 좋은 복원 지표를 보이는 것이다. 이 경우 S3 이후 실험에서는 `front/middle/back` 문자열 tag 대신 더 정교한 위치 보조 구조를 사용할 수 있다.

나쁜 결과는 정식 후보가 `coarse_bins`보다 낫지 않거나, 위치 없음과 차이가 거의 없는 것이다. 이 경우 현재 복원 병목은 위치 표현보다 의미 골격 품질, 학습 예산, decoder 구조, 또는 데이터 편향에 있을 수 있다.

## 6. 산출물

| 산출물 | 위치 |
|---|---|
| Kaggle runner | `kaggle/v2_s2a/run_v2_s2a.py` |
| Kaggle metadata | `kaggle/v2_s2a/kernel-metadata.json` |
| push script | `scripts/push_kaggle_v2_s2a.sh` |
| 결과 문서 | `docs/v2/experiments/s2a-positional-encoding.md` |
| output | `outputs/v2_s2a/` |

## 7. 로컬 검증

Kaggle push 전에는 다음을 확인한다.

```bash
rtk .venv/bin/python -m py_compile kaggle/v2_s2a/run_v2_s2a.py
rtk bash -n scripts/push_kaggle_v2_s2a.sh
git diff --check
```

필요하면 fallback text와 작은 sample로 local smoke run을 수행한다.

## 8. 다음 단계로 넘길 조건

S2a가 통과하면 S3 anchor baseline comparison은 `best_positional_encoding`을 기본 위치 보조 구조로 사용한다.

S2a가 실패하면 S3는 S2의 `front/middle/back`을 그대로 쓰되, 위치 보조 구조가 아직 임시 scaffold라는 caveat를 명시한다.
