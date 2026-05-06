# S4b 계획: multi-step delta rollout

## 1. 무엇을 측정하는가

S4b는 S4a의 one-step delta-token reverse objective가 실제 다단계 reverse rollout에서도 유지되는지 측정한다.

S4a는 각 transition을 독립적으로 평가했다.

```text
input:  현재 partial state + 이번 단계의 위치 marker
target: 새로 채울 delta token ids
```

S4b는 같은 delta model을 학습하되, 평가 시에는 25% state에서 시작해 예측 결과를 누적한다.

```text
25% state -> predict delta -> 50% predicted state
50% predicted state -> predict delta -> 75% predicted state
75% predicted state -> predict delta -> 100% predicted state
```

따라서 S4b가 측정하는 것은 단일 step의 정답 조건부 예측력이 아니라, **semantic skeleton에서 시작한 reverse trajectory가 예측 오류를 누적해도 원문 의미와 표면 token을 더 잘 회복하는가**다.

## 2. 왜 지금 중요한가

S4a는 `importance_schedule`이 delta objective에서 `random_schedule`과 `position_only_schedule`을 이긴다는 좋은 신호를 줬다.

하지만 S4a는 teacher-forced에 가깝다. 각 step의 입력은 정답 partial state다. 실제 reverse process에서는 이전 step에서 예측한 token이 다음 step 입력이 되므로, 작은 오류가 누적되어 semantic drift, 반복, entity 누락으로 커질 수 있다.

S4b는 이 차이를 확인한다. S4a의 gain이 실제 rollout에서도 유지되면 v2의 process claim은 더 강해진다. 반대로 S4a gain이 rollout에서 사라지면 현재 decoder 또는 delta insertion 방식이 아직 trajectory model로는 부족하다는 뜻이다.

## 3. 좋은 결과와 나쁜 결과의 의미

좋은 결과는 `importance_schedule`이 final rollout score에서 `random_schedule`과 `position_only_schedule`보다 높고, 의미/반복 지표에서도 크게 무너지지 않는 것이다.

| 비교 | 좋은 결과 | 의미 |
|---|---|---|
| final token F1 / ROUGE-L | importance > random | 누적 rollout이 표면 token 복원에서도 덜 무너진다. |
| final content recall | importance > random | 중심 의미에서 세부 의미로 확장하는 경로가 유지된다. |
| final entity recall | importance >= random | rare entity와 숫자 표지가 rollout에서 심하게 사라지지 않는다. |
| semantic drift proxy | importance <= random | 의미 drift가 random보다 작다. |
| repetition rate | importance <= random + tolerance | 반복 template artifact가 gain을 만든 것이 아니다. |
| importance vs position-only | importance > position-only | 위치 보조 구조만이 아니라 content-bearing skeleton이 기여한다. |

나쁜 결과는 S4a에서는 importance가 이겼지만 S4b rollout에서는 random 또는 position-only가 이기는 경우다. 이 경우 S4a의 gain은 독립 step objective에는 존재하지만, 현재 autoregressive delta rollout 구조에서는 안정적으로 유지되지 않는다는 뜻이 된다.

## 4. 어떤 반론과 모호성을 다루는가

S4b는 다음 모호성을 다룬다.

- S4a의 one-step gain이 teacher-forced artifact인가?
- importance schedule이 실제 reverse trajectory에서 semantic drift를 줄이는가?
- `position_only_schedule`이 위치 marker만으로 rollout score를 설명할 수 있는가?
- delta generation 반복이 누적되어 final text를 망가뜨리는가?
- 초기 25% skeleton content가 다음 step에 실제 정보로 쓰이는가?

특히 `position_only_schedule`은 target delta 위치와 정답 token은 importance schedule을 따르지만, 다음 step encoder 입력에서는 token content를 계속 제거한다. 따라서 이 조건이 높으면 위치 scaffold와 decoder prior만으로도 rollout이 상당 부분 설명된다는 뜻이다.

## 5. 다음 실험으로 넘길 조건

S4b 조건은 다음 세 가지다.

| 조건 | 설명 |
|---|---|
| `importance_schedule` | attention-received score가 높은 token을 25% skeleton에 남기고, 예측 delta를 누적 삽입한다. |
| `random_schedule` | 같은 ratio와 같은 token 수에서 random schedule의 25% state로 시작해 예측 delta를 누적 삽입한다. |
| `position_only_schedule` | importance 위치는 주지만 encoder 입력의 token content는 제거한 채, importance target delta를 예측한다. |

Gate는 다음과 같다.

| Gate | 통과 조건 | 해석 |
|---|---|---|
| `S4B-G-RUN` | 모든 schedule 조건과 rollout transition이 실행됨 | 실험 완결성 |
| `S4B-G-LOSS-FINITE` | teacher-forced step eval loss가 모두 유한함 | 학습 수치 안정성 |
| `S4B-G-IMPORTANCE-BEATS-RANDOM` | importance rollout score가 random보다 tolerance 이상 높음 | S4a gain이 rollout에서도 유지됨 |
| `S4B-G-IMPORTANCE-BEATS-POSITION-ONLY` | importance rollout score가 position-only보다 tolerance 이상 높음 | content-bearing skeleton 사용 증거 |
| `S4B-G-ROLLOUT-SEMANTIC-CONTENT` | importance final content recall이 random보다 높음 | 의미 내용 복원 우위 |
| `S4B-G-ROLLOUT-DRIFT-CHECK` | importance semantic drift proxy가 random보다 크게 나쁘지 않음 | 의미 drift 방지 |
| `S4B-G-REPETITION-CHECK` | importance repetition이 random보다 크게 나쁘지 않음 | 반복 누적 artifact 방지 |
| `S4B-G-BEST-IDENTIFIED` | 최고 rollout schedule이 식별됨 | 다음 구조 개선의 기준점 확보 |

S4b가 통과하면 S4a의 delta objective gain이 process-level trajectory 신호로 강화된다. 실패하면 S4c의 span-infilling reverse decoder, entity-aware delta head, confidence-gated iterative refinement 같은 구조 보정이 우선이다.
