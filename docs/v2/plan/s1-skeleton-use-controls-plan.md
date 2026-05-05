# S1 계획: 의미 골격 사용 검증

## 1. 실험이 측정하는 것

S1은 S0에서 만든 의미 골격이 단순히 보기 좋은 압축물이 아니라, 원문을 식별하고 복원 방향을 정하는 데 실제로 쓰일 수 있는지 확인한다.

S1의 핵심 질문은 다음이다.

> correct 의미 골격은 random, shuffled, wrong-document, same-position control보다 원문을 더 잘 가리키는가?

이번 단계에서는 큰 생성 모델을 바로 학습하지 않는다. 먼저 frozen `t5-small` encoder를 사용한 검색형 복원 평가를 둔다. 의미 골격을 query로 encode하고, 후보 원문 embedding 중 자기 원문을 얼마나 잘 찾는지 본다.

## 2. 왜 중요한가

S0는 의미 골격 생성 흐름이 작동한다는 것을 확인했다. 하지만 S0만으로는 모델이나 평가기가 골격 정보를 실제로 쓰는지 알 수 없다.

S1은 다음 반론을 직접 다룬다.

- token 수가 같으면 random 골격도 충분한가?
- 같은 위치 패턴이면 다른 문서 token이어도 비슷한가?
- token 순서를 섞어도 성능이 유지되는가?
- 핵심 token을 제거해도 평가가 거의 변하지 않는가?
- 위치 정보만으로 원문을 찾는가?

이 반론이 해소되어야 S2의 skeleton-to-text 복원 학습으로 넘어가는 것이 방어 가능하다.

## 3. 조건과 control

| 조건 | 의미 |
|---|---|
| `idf_correct` | IDF 기반 correct 의미 골격 |
| `attention_correct` | attention-received 기반 correct 의미 골격 |
| `random_same_count` | 같은 token 수의 무작위 골격 |
| `position_prior` | 문장 앞부분 위치 편향 control |
| `position_only` | token 내용 없이 위치 bin만 제공 |
| `shuffled_correct` | correct 골격의 token 순서를 섞음 |
| `wrong_document` | 다른 문서의 correct 골격 |
| `same_position_random` | 현재 문서의 선택 위치에 다른 문서 token을 삽입 |
| `remove_topk` | correct 골격에서 가장 중요한 token 제거 |
| `remove_lowk` | correct 골격에서 덜 중요한 token 제거 |

기본 scorer는 S0 결과를 따라 `idf`와 `attention_received`를 사용한다. 위치 편향이 강했으므로 `position_prior`, `position_only`, `same_position_random`은 필수 control로 둔다.

## 4. Metric과 gate

| Metric/Gate | 측정 | 통과 기준 |
|---|---|---|
| `S1-G-RUN` | 모든 조건 실행과 output 생성 | metrics/summary/samples 생성 |
| `S1-G-CORRECT-BEATS-RANDOM` | correct 골격이 random보다 원문 검색을 잘하는가 | `idf_correct` 또는 `attention_correct`가 `random_same_count`보다 hit@1 우위 |
| `S1-G-WRONG-DOC-DROPS` | 다른 문서 골격에서 성능이 하락하는가 | correct가 `wrong_document`보다 hit@1 우위 |
| `S1-G-POSITION-ONLY-DROPS` | 위치 정보만으로는 부족한가 | correct가 `position_only`보다 hit@1 우위 |
| `S1-G-REMOVAL-DROPS` | correct 골격에서 token을 제거하면 성능이 하락하는가 | `remove_topk`, `remove_lowk`가 모두 correct보다 낮음 |
| `S1-G-TOPK-ORDER` | 중요도 순서가 제거 민감도와 맞는가 | `remove_topk`가 `remove_lowk`보다 낮음. 단, 전체 통과 필수 조건은 아님 |
| `S1-G-SHUFFLE-SENSITIVE` | 순서가 영향을 주는가 | correct가 `shuffled_correct`보다 우위. 단, 전체 통과 필수 조건은 아님 |

주요 metric은 hit@1, hit@5, MRR, 자기 원문 cosine similarity, 가장 가까운 오답과의 margin이다.

## 5. 좋은 결과와 나쁜 결과

좋은 결과는 correct 의미 골격이 random, wrong-document, position-only보다 명확히 높은 검색 성능을 보이는 것이다. 이 경우 의미 골격이 원문 식별 정보와 복원 방향성을 담고 있다고 볼 수 있다.

나쁜 결과는 wrong-document나 same-position random이 correct와 비슷하게 나오는 것이다. 이 경우 평가기가 의미 token을 쓰는 것이 아니라 위치, 문체, 데이터 편향, 평균 embedding prior를 쓰고 있을 수 있다.

순서 섞기에서 성능이 크게 떨어지면 의미 골격이 단순 keyword bag 이상이라는 신호다. 반대로 shuffled가 correct와 비슷하면 S2에서 순서/간격 보조 구조를 더 명시적으로 설계해야 한다.

## 6. 산출물

| 산출물 | 위치 |
|---|---|
| Kaggle runner | `kaggle/v2_s1/run_v2_s1.py` |
| Kaggle metadata | `kaggle/v2_s1/kernel-metadata.json` |
| push script | `scripts/push_kaggle_v2_s1.sh` |
| 결과 문서 | `docs/v2/experiments/s1-skeleton-use-controls.md` |
| output | `outputs/v2_s1/` |

## 7. 로컬 검증

Kaggle push 전에는 다음만 확인한다.

```bash
rtk .venv/bin/python -m py_compile kaggle/v2_s1/run_v2_s1.py
rtk bash -n scripts/push_kaggle_v2_s1.sh
git diff --check
```

필요하면 `--max-samples 16 --no-use-hf-dataset`로 local smoke run을 수행한다.

## 8. 다음 단계로 넘길 조건

S1이 통과하면 S2로 넘어간다. S2에서는 의미 골격과 위치 보조 구조를 입력으로 하는 skeleton-to-text 복원 모델을 학습한다.

S1이 실패하면 S2로 가지 않는다. 먼저 scorer, token grouping, 위치 보조 구조, same-position control을 재설계한다.
