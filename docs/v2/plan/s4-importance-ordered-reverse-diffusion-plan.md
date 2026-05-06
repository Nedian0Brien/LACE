# S4 계획: importance-ordered reverse diffusion

## 1. 무엇을 측정하는가

S4는 문장을 그대로 복원하는 probe가 아니라, 중요도 기반 forward masking schedule이 random corruption schedule보다 더 나은 reverse generation curriculum을 만드는지 측정한다.

Forward process는 다음처럼 정의한다.

```text
x0: 원문
x1: 중요도 낮은 token 일부 mask
x2: 더 많은 세부 token mask
x3: 중심 의미 token만 남은 semantic skeleton
```

Reverse process는 그 반대 방향이다.

```text
x3 -> x2 -> x1 -> x0-like text
```

따라서 학습 목표는 전체 원문을 한 번에 복원하는 것이 아니라 다음 전이를 학습하는 것이다.

```text
p_theta(x_{t-1} | x_t, t)
```

## 2. 왜 지금 중요한가

S3, S3a, S3b는 terminal content와 위치 scaffold가 완전히 무의미하지는 않다는 점을 보여줬지만, exact reconstruction probe와 lexical metric만으로는 연구의 본질을 평가하지 못했다.

이 연구의 핵심은 "중요한 token이 terminal state로 남는가"가 아니라, 중요도 낮은 token부터 순차적으로 제거하는 forward process의 역과정이 문장의 중심 의미에서 세부 의미로 확장하는 더 좋은 language modeling 경로를 제공하는가다.

S4는 이 질문으로 실험 단위를 올린다. 즉, `attention_terminal` 한 조건의 score를 조금 더 올리는 것이 아니라, 전체 trajectory가 random masking trajectory보다 더 나은지 본다.

## 3. 좋은 결과와 나쁜 결과의 의미

좋은 결과는 `importance_schedule`이 같은 model budget의 `random_schedule`보다 다음 항목에서 높게 나오는 것이다.

| 비교 | 좋은 결과 | 의미 |
|---|---|---|
| transition score | importance > random | 중요도 기반 reverse step이 더 쉬운 확장 문제를 만든다. |
| expansion recall | importance > random | skeleton에서 다음 단계의 세부 token을 더 잘 추가한다. |
| input retention | importance > random | 중심 의미 token을 잃지 않고 확장한다. |
| original content recall | importance > random | 최종 원문 의미와 더 가까운 방향으로 전개된다. |
| repetition rate | importance <= random | 확장이 반복 template에 덜 빠진다. |

나쁜 결과는 importance schedule이 random schedule과 같거나 낮은 경우다. 이 경우 중요도 기반 forward process가 더 좋은 reverse curriculum이라는 주장은 아직 약하다.

`position_only_schedule`이 importance와 비슷하면 위치 scaffold만으로도 trajectory metric이 설명된다는 뜻이다. 이 경우 semantic skeleton claim은 방어하기 어렵다.

## 4. 어떤 반론과 모호성을 다루는가

S4는 다음 반론을 다룬다.

- 문장 exact reconstruction probe의 작은 F1/ROUGE 차이가 연구 본질을 가리는 것 아닌가?
- attention token 몇 개가 아니라, importance-ordered schedule 전체가 random schedule보다 유리한가?
- reverse model이 중심 의미 token을 보존하면서 세부 token을 붙이는가?
- 위치 scaffold만으로도 비슷한 trajectory가 가능한가?
- generation 반복이 metric을 만든 것은 아닌가?

다만 S4는 아직 open-ended generation의 최종 품질 평가가 아니다. S4는 constrained reverse transition, 즉 `x_t -> x_{t-1}` process-level 비교다.

## 5. 다음 실험으로 넘길 조건

S4 조건은 다음 세 가지로 시작한다.

| 조건 | 설명 |
|---|---|
| `importance_schedule` | attention-received score 순서로 낮은 중요도 token부터 mask하는 schedule |
| `random_schedule` | 같은 mask ratio와 같은 token 수를 random 순서로 mask하는 schedule |
| `position_only_schedule` | importance 위치 scaffold만 주고 token content는 pad로 대체하는 control |

기본 ratio는 다음이다.

```text
0.25 -> 0.50 -> 0.75 -> 1.00
```

즉, reverse transition은 다음 세 개다.

```text
25% skeleton -> 50% state
50% state -> 75% state
75% state -> full state
```

Gate는 다음과 같다.

| Gate | 통과 조건 | 해석 |
|---|---|---|
| `S4-G-RUN` | 모든 schedule 조건과 transition이 실행됨 | 실험 완결성 |
| `S4-G-LOSS-FINITE` | 모든 eval loss가 유한함 | 수치 안정성 |
| `S4-G-IMPORTANCE-BEATS-RANDOM` | importance score가 random보다 tolerance 이상 높음 | importance schedule의 trajectory 우위 |
| `S4-G-IMPORTANCE-BEATS-POSITION-ONLY` | importance score가 position-only보다 tolerance 이상 높음 | content-bearing skeleton 사용 증거 |
| `S4-G-EXPANSION-RECALL` | importance expansion recall이 random보다 높음 | 중심 골격에서 세부 token 확장 능력 |
| `S4-G-RETENTION` | importance input retention이 random보다 높음 | 중심 의미 보존 |
| `S4-G-REPETITION` | importance repetition이 random보다 크게 나쁘지 않음 | 반복 template artifact 방지 |

S4가 통과하면 다음은 multi-step rollout 또는 masked-span constrained generation으로 확장한다. 실패하면 scorer 문제가 아니라 schedule/objective 자체를 다시 설계해야 한다.
