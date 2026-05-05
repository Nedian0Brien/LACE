# S1 결과: 의미 골격 사용 검증

## 1. 이 실험이 측정한 것

S1은 S0에서 만든 의미 골격이 원문을 식별하는 데 실제로 쓰일 수 있는지 확인했다.

이번 실험은 생성 모델을 학습하지 않았다. frozen `t5-small` encoder를 사용해 의미 골격을 query로 인코딩하고, 1024개 후보 원문 중 자기 원문을 얼마나 잘 찾는지 검색형 복원 평가로 측정했다.

고정 조건은 다음이다.

| 항목 | 값 |
|---|---|
| phase | `v2_s1` |
| model | frozen `t5-small` |
| data | `wikitext/wikitext-2-raw-v1:train` |
| samples | 1024 |
| device | `cuda` |
| keep ratio | 0.25 |
| output | `outputs/v2_s1/lace_v2_s1/` |

## 2. 왜 중요한가

S0는 의미 골격을 만들고 보존 품질을 비교한 단계였다. 하지만 S0만으로는 복원 평가기가 의미 골격을 실제 단서로 쓰는지, 아니면 문장 앞부분 위치 편향이나 평균 임베딩 성향에 기대는지 알 수 없다.

S1은 다음 반론을 직접 겨냥했다.

- 같은 token 수의 무작위 골격도 충분한가?
- 다른 문서의 골격도 비슷하게 작동하는가?
- token 내용 없이 위치 정보만으로 원문을 찾는가?
- 같은 위치에 다른 문서 token을 넣어도 성능이 유지되는가?
- 골격 token 순서를 섞어도 성능이 유지되는가?
- 골격 token을 제거하면 성능이 실제로 떨어지는가?

## 3. 결과가 의미하는 것

핵심 결과는 S1 통과다. `attention_correct`가 가장 좋은 정답 골격이었고, Hit@1 0.9111로 `random_same_count` 0.7373보다 높았다.

| 조건 | Hit@1 | Hit@5 | MRR | 자기 원문 유사도 | 여백 |
|---|---:|---:|---:|---:|---:|
| `attention_correct` | 0.9111 | 0.9883 | 0.9467 | 0.7251 | 0.0364 |
| `idf_correct` | 0.7637 | 0.9414 | 0.8432 | 0.6600 | 0.0228 |
| `position_prior` | 0.8154 | 0.9629 | 0.8788 | 0.7814 | 0.0300 |
| `random_same_count` | 0.7373 | 0.9062 | 0.8147 | 0.6946 | 0.0154 |
| `shuffled_correct` | 0.6074 | 0.8799 | 0.7225 | 0.6251 | 0.0100 |
| `remove_topk` | 0.4238 | 0.7539 | 0.5724 | 0.5814 | -0.0055 |
| `remove_lowk` | 0.3750 | 0.6895 | 0.5204 | 0.5731 | -0.0094 |
| `position_only` | 0.0010 | 0.0078 | 0.0097 | 0.4444 | -0.1265 |
| `same_position_random` | 0.0020 | 0.0088 | 0.0114 | 0.5579 | -0.1497 |
| `wrong_document` | 0.0000 | 0.0039 | 0.0086 | 0.5167 | -0.1684 |

Gate 결과는 다음이다.

| Gate | 통과 | 해석 |
|---|---:|---|
| `S1-G-RUN` | true | 10개 조건의 산출물이 생성됐다. |
| `S1-G-CORRECT-BEATS-RANDOM` | true | 정답 골격이 같은 token 수의 무작위 골격보다 강하다. |
| `S1-G-WRONG-DOC-DROPS` | true | 다른 문서 골격은 원문 검색에 거의 도움이 되지 않는다. |
| `S1-G-POSITION-ONLY-DROPS` | true | 위치 정보만으로는 원문을 찾지 못한다. |
| `S1-G-REMOVAL-DROPS` | true | 골격 token을 제거하면 정답 골격 대비 성능이 크게 떨어진다. |
| `S1-G-TOPK-ORDER` | false | 중요 token 제거가 덜 중요한 token 제거보다 더 치명적이라는 순서 주장은 아직 확인되지 않았다. |
| `S1-G-SHUFFLE-SENSITIVE` | true | 순서를 섞으면 성능이 낮아진다. 골격은 단순 keyword bag 이상이다. |
| `overall_pass` | true | S1 검색형 사용 검증은 통과했다. |
| `s2_ready` | true | S2 복원 학습으로 넘어갈 수 있다. |

## 4. 어떤 반론과 혼동 요인을 줄였는가

가장 큰 방어 지점은 `wrong_document`, `position_only`, `same_position_random`이 거의 실패했다는 점이다. 이는 평가기가 단순히 위치 표지나 평균 문장 prior만 보고 원문을 찾는 것은 아니라는 신호다.

다만 위치 편향은 사라지지 않았다. `position_prior`는 Hit@1 0.8154로 매우 강하고, 자기 원문 유사도는 0.7814로 `attention_correct`보다 높았다. 따라서 S2에서는 위치 보조 구조를 버릴 수 없지만, 위치만으로 의미 골격의 기여를 설명해서도 안 된다.

또 하나의 주의점은 제거 실험이다. `remove_topk`와 `remove_lowk`는 둘 다 정답 골격보다 크게 낮았지만, `remove_topk`가 `remove_lowk`보다 더 낮지는 않았다. 따라서 이번 결과로 "attention 중요도 순서가 token별 인과 중요도와 정확히 일치한다"고 주장하면 안 된다.

## 5. 다음 실험에서 어떻게 검증할 것인가

다음 단계는 S2 의미 골격-문장 복원 학습이다. S2는 S1의 결론을 그대로 가져가되, 다음 제약을 유지해야 한다.

- `attention_correct`, `idf_correct`, `random_same_count`, `position_prior`를 주요 비교 조건으로 둔다.
- 위치 보조 구조를 명시적으로 넣되, `position_only`와 `same_position_random`을 계속 control로 유지한다.
- 생성 품질과 teacher-forced proxy를 분리한다.
- `remove_topk`/`remove_lowk`는 중요도 순서 증거가 아니라 골격 token 제거 민감도 control로 해석한다.
- S2의 성공 주장은 "복원 학습에서 의미 골격 + 위치 보조 구조가 무작위 손상보다 더 좋은 역방향 궤적을 만든다"로 제한한다.

S1의 결론은 생성 성공이 아니라, S2를 시작할 만큼 의미 골격 사용 신호가 있다는 것이다.
