# S2 계획: 의미 골격-문장 복원 학습

## 1. 이 실험이 측정하는 것

S2는 의미 골격에서 원문으로 되돌아가는 짧은 역방향 복원 학습이 가능한지 측정한다.

검증할 연구 질문은 [research-questions.md](../research-questions.md)의 RQ2다.

> 중요도 기반 forward는 무작위 손상보다 좋은 reverse trajectory를 만드는가?

이번 단계는 open-ended generation 실험이 아니다. `t5-small`을 조건별로 짧게 미세조정하고, 같은 held-out 문장에서 teacher-forced loss와 짧은 생성 복원 지표를 비교한다.

## 2. 왜 중요한가

S0는 의미 골격 재료가 무작위/균일 baseline보다 나은지 확인했다. S1은 frozen encoder 검색 평가에서 의미 골격이 원문 식별 단서로 쓰일 수 있음을 확인했다.

하지만 LACE의 핵심 주장은 검색이 아니라 역방향 궤적이다. S2는 의미 골격을 입력받은 복원 모델이 무작위 골격이나 위치 전용 입력보다 더 좋은 복원 문제를 학습하는지 처음으로 확인한다.

## 3. 조건과 control

학습 조건은 같은 모델 구조와 학습 예산을 사용한다.

| 조건 | 입력 |
|---|---|
| `attention_scaffold` | attention 기반 의미 골격 + 위치 보조 구조 |
| `idf_scaffold` | IDF 기반 의미 골격 + 위치 보조 구조 |
| `random_scaffold` | 같은 token 수의 무작위 골격 + 위치 보조 구조 |
| `position_prior_scaffold` | 문장 앞부분 token + 위치 보조 구조 |
| `position_only` | token 내용 없이 위치 bin만 제공 |

추가 평가 control은 `attention_scaffold`로 학습한 모델에만 적용한다.

| 평가 control | 목적 |
|---|---|
| `attention_wrong_document` | 다른 문서 골격을 넣었을 때 복원이 무너지는가 |
| `attention_same_position_random` | 같은 위치에 다른 문서 token을 넣으면 복원이 무너지는가 |
| `attention_position_only` | 학습된 모델이 위치만으로도 비슷하게 복원하는가 |

## 4. Metric과 gate

| Metric/Gate | 측정 | 통과 기준 |
|---|---|---|
| `S2-G-RUN` | 조건별 학습과 산출물 생성 | metrics/summary/samples 생성 |
| `S2-G-LOSS-FINITE` | teacher-forced loss가 유한한가 | 모든 주요 조건 loss가 finite |
| `S2-G-NONEMPTY-GENERATION` | 생성물이 비어 있지 않은가 | 주요 조건 nonempty rate 0.8 이상 |
| `S2-G-ATTENTION-BEATS-RANDOM` | attention 의미 골격이 무작위 골격보다 복원에 유리한가 | `attention_scaffold` token F1 또는 ROUGE-L이 `random_scaffold`보다 높음 |
| `S2-G-ATTENTION-BEATS-POSITION` | 의미 token이 위치 전용보다 유리한가 | `attention_scaffold`가 `position_only`보다 높음 |
| `S2-G-WRONG-DOC-DROPS` | wrong-document 입력에서 성능이 하락하는가 | correct attention 평가가 wrong-document 평가보다 높음 |

주요 지표는 teacher-forced loss, token F1, ROUGE-L F1, keyword recall, skeleton coverage, nonempty rate다.

## 5. 좋은 결과와 나쁜 결과

좋은 결과는 `attention_scaffold`가 `random_scaffold`와 `position_only`보다 token F1/ROUGE-L에서 높고, wrong-document/same-position control에서 하락하는 것이다. 이 경우 의미 골격 + 위치 보조 구조가 무작위 손상보다 더 좋은 복원 학습 문제를 만든다는 S2 수준의 증거가 된다.

나쁜 결과는 `random_scaffold`나 `position_only`가 `attention_scaffold`와 비슷하거나 더 좋은 것이다. 이 경우 모델은 의미 골격보다 language prior나 위치 편향에 기대고 있을 수 있다.

`position_prior_scaffold`가 강하면 S2 실패는 아니지만, 핵심 주장을 "의미 골격 단독"으로 말할 수 없다. 이 경우 다음 단계는 의미 골격과 위치 보조 구조의 역할 분리를 더 강화해야 한다.

## 6. 산출물

| 산출물 | 위치 |
|---|---|
| Kaggle runner | `kaggle/v2_s2/run_v2_s2.py` |
| Kaggle metadata | `kaggle/v2_s2/kernel-metadata.json` |
| push script | `scripts/push_kaggle_v2_s2.sh` |
| 결과 문서 | `docs/v2/experiments/s2-skeleton-to-text-reconstruction.md` |
| output | `outputs/v2_s2/` |

## 7. 로컬 검증

Kaggle push 전에는 다음을 확인한다.

```bash
rtk .venv/bin/python -m py_compile kaggle/v2_s2/run_v2_s2.py
rtk bash -n scripts/push_kaggle_v2_s2.sh
git diff --check
```

필요하면 fallback text와 아주 작은 sample로 local smoke run을 수행한다.

## 8. 다음 단계로 넘길 조건

S2가 통과하면 S3 anchor baseline comparison 또는 S4 constrained generation으로 갈 수 있다. 단, S2는 아직 open-ended generation 성공 증거가 아니므로, 다음 단계에서도 teacher-forced proxy와 실제 생성 품질을 분리해야 한다.

S2가 실패하면 S3/S4로 가지 않는다. 먼저 골격 단위, 위치 보조 구조, 학습 예산, 입력 형식, wrong-document control을 재설계한다.
