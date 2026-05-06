# S4a 계획: delta-token reverse objective

## 1. 무엇을 측정하는가

S4a는 S4의 reverse transition을 유지하되, 예측 대상을 바꾼다.

S4에서는 `x_t -> x_{t-1}` 전이를 학습할 때 다음 상태 전체를 target으로 삼았다.

```text
input:  현재 partial state
target: 다음 partial state 전체
```

S4a에서는 다음 상태 전체를 다시 생성하지 않는다. 현재 상태에서 다음 단계로 갈 때 새로 unmask되는 token/span만 target으로 둔다.

```text
input:  현재 partial state + 이번 단계에서 채울 위치 marker
target: newly unmasked delta token/span
```

따라서 S4a가 측정하는 것은 문장 복원 능력이 아니라, **semantic skeleton이 다음에 붙을 세부 token을 예측하는 데 실제로 정보가 되는가**다.

## 2. 왜 지금 중요한가

S4 결과는 분화됐다. `random_schedule`은 loss, Token F1, ROUGE-L, 종합 score에서 더 좋았지만, `importance_schedule`은 target content recall, input retention, expansion recall, original content recall, entity recall에서 더 좋았다.

이 모호성은 objective에서 생겼을 가능성이 크다. 전체 target state를 다시 생성하게 하면 모델은 두 일을 동시에 해야 한다.

```text
1. 이미 입력에 있는 skeleton token을 유지하거나 다시 생성하기
2. 새로 추가되어야 할 delta token을 예측하기
```

우리가 확인해야 하는 것은 1이 아니라 2다. 연구 가설의 본질은 중심 의미 token에서 세부 의미 token으로 확장하는 능력이기 때문이다.

## 3. 좋은 결과와 나쁜 결과의 의미

좋은 결과는 `importance_schedule`이 `random_schedule`보다 delta target에서 더 높은 점수를 얻는 것이다.

| 비교 | 좋은 결과 | 의미 |
|---|---|---|
| delta token F1 | importance > random | semantic skeleton이 다음 token 예측을 더 쉽게 만든다. |
| delta content recall | importance > random | 중심 의미에서 세부 의미 token을 더 잘 확장한다. |
| entity recall | importance > random | 새로 붙는 고유명사/숫자/핵심 표지가 더 잘 복원된다. |
| importance vs position-only | importance > position-only | 위치 scaffold 이상으로 content-bearing skeleton이 기여한다. |
| context copy rate | importance가 과도하게 높지 않음 | 새 token 예측이 입력 복사로 대체되지 않는다. |

나쁜 결과는 random이 delta objective에서도 이기는 경우다. 이 경우 S4의 실패가 단순히 "전체 state를 target으로 둔 objective 문제"만은 아니며, 현재 semantic skeleton 또는 모델 구조가 다음 token 예측에 충분한 조건을 주지 못한다는 뜻이 된다.

## 4. 어떤 반론과 모호성을 다루는가

S4a는 다음 모호성을 분리한다.

- S4에서 random이 이긴 이유가 전체 state 재생성 objective 때문인가?
- importance schedule의 semantic signal이 실제 delta token 예측으로 이어지는가?
- 위치 marker만 있어도 비슷한 성능이 나오는가?
- 모델이 새 token을 예측하는 대신 입력 context를 복사하는가?
- delta target 자체의 길이와 난이도 차이가 schedule 비교를 왜곡하는가?

특히 `position_only_schedule`은 importance schedule의 delta position을 알려주되 token content를 제거한다. 이 control이 높으면 위치 scaffold만으로 상당 부분 설명된다는 뜻이다.

## 5. 다음 실험으로 넘길 조건

S4a 조건은 다음 세 가지다.

| 조건 | 설명 |
|---|---|
| `importance_schedule` | attention-received score가 높은 token을 오래 보존하고, 다음 단계의 delta token만 예측한다. |
| `random_schedule` | 같은 ratio와 같은 token count에서 random order의 delta token만 예측한다. |
| `position_only_schedule` | importance 위치 marker만 주고 token content 없이 delta token을 예측한다. |

Gate는 다음과 같다.

| Gate | 통과 조건 | 해석 |
|---|---|---|
| `S4A-G-RUN` | 모든 schedule 조건과 transition이 실행됨 | 실험 완결성 |
| `S4A-G-LOSS-FINITE` | 모든 eval loss가 유한함 | 수치 안정성 |
| `S4A-G-IMPORTANCE-BEATS-RANDOM` | importance score가 random보다 tolerance 이상 높음 | delta objective에서 importance trajectory 우위 |
| `S4A-G-IMPORTANCE-BEATS-POSITION-ONLY` | importance score가 position-only보다 tolerance 이상 높음 | content-bearing skeleton 사용 증거 |
| `S4A-G-DELTA-CONTENT` | importance delta content recall이 random보다 높음 | 세부 의미 token 확장 능력 |
| `S4A-G-ENTITY-DELTA` | importance entity recall이 random보다 높음 | 새 의미 표지 복원 능력 |
| `S4A-G-COPY-LEAKAGE` | context copy rate가 random보다 크게 높지 않음 | 입력 복사 artifact 방지 |
| `S4A-G-REPETITION` | repetition이 random보다 크게 나쁘지 않음 | 반복 template artifact 방지 |

S4a가 통과하면 multi-step delta rollout로 넘어갈 수 있다. S4a도 실패하면 S5로 가지 않고 모델 구조 개선을 탐구한다. 후보는 schedule-aware span infilling head, copy/pointer-aware decoder, insertion-style decoder, confidence-gated iterative refinement다.
