# S4b 결과: multi-step delta rollout

## 1. 이 실험이 측정한 것

S4b는 S4a에서 얻은 delta-token objective의 긍정 신호가 실제 역방향 궤적에서도 유지되는지 확인한 실험이다.

S4a는 각 transition을 따로 평가했다.

```text
25% -> 50%
50% -> 75%
75% -> 100%
```

S4b는 여기서 한 걸음 더 나아가, 각 단계에서 모델이 생성한 delta token을 실제 current state에 삽입하고 다음 단계 입력으로 넘겼다.

```text
start: 25% semantic skeleton
step 1: model generates 25% -> 50% delta
step 2: generated 50% state에서 50% -> 75% delta 생성
step 3: generated 75% state에서 75% -> 100% delta 생성
final: 생성된 100% state를 target/original과 비교
```

따라서 S4b가 측정한 것은 단순한 teacher-forced one-step accuracy가 아니라, error accumulation이 있는 실제 constrained rollout에서 `importance_schedule`이 `random_schedule`과 `position_only_schedule`보다 더 좋은 reverse trajectory를 만드는가다.

실행 정보는 다음이다.

| 항목 | 값 |
|---|---|
| phase | `v2_s4b` |
| Kaggle kernel | `dennisparknd/lace-v2-s4b-multi-step-delta-rollout` |
| model | `t5-small` tokenizer + custom PyTorch encoder-decoder |
| data | `wikitext/wikitext-2-raw-v1:train` |
| train samples | 768 |
| eval samples | 192 |
| device | `cuda` |
| ratios | `0.25 -> 0.50 -> 0.75 -> 1.00` |
| reverse epochs | 2 |
| output | `outputs/v2_s4b/lace_v2_s4b/` |

비교 조건은 다음이다.

| 조건 | 의미 |
|---|---|
| `importance_schedule` | attention-received score가 높은 token을 먼저 남긴 semantic skeleton에서 세부 token을 순차 확장한다. |
| `random_schedule` | 같은 ratio와 token budget에서 무작위 순서로 남은 token state를 확장한다. |
| `position_only_schedule` | importance 위치는 주지만 visible token content는 제거한다. |

## 2. 왜 중요한가

S4a는 좋은 신호였지만, 아직 "각 단계만 따로 보면 잘 맞힌다"는 수준이었다. 실제 diffusion language model의 핵심은 한 단계 성능보다 궤적이다. 앞 단계에서 생긴 오류가 다음 단계 입력으로 들어가고, 그 오류가 의미 drift나 반복으로 누적될 수 있기 때문이다.

S4b는 그래서 다음 질문에 답한다.

```text
중심 의미 token에서 출발해 세부 token을 붙여 나가는 방식이
무작위 손상 복원보다 누적 rollout에서도 더 의미 있는 문장 상태를 만드는가?
```

이 질문은 현재 v2 핵심 주장에 더 직접적으로 닿아 있다. S4b에서 importance가 random보다 좋아야 "semantic skeleton + positional scaffold가 더 나은 reverse trajectory를 만든다"는 주장이 S4a보다 강해진다.

## 3. 결과가 의미하는 것

S4b는 `process_ready=true`, `overall_pass=true`, `structure_review_needed=false`, `s5_ready=false`다.

Gate 결과는 다음이다.

| Gate | 통과 | 해석 |
|---|---:|---|
| `S4B-G-RUN` | true | 세 schedule이 모두 실행됐다. |
| `S4B-G-LOSS-FINITE` | true | 모든 schedule의 teacher-forced eval loss가 유한했다. |
| `S4B-G-IMPORTANCE-BEATS-RANDOM` | true | importance rollout score 0.7336이 random 0.6215보다 높았다. |
| `S4B-G-IMPORTANCE-BEATS-POSITION-ONLY` | true | importance rollout score 0.7336이 position-only 0.1858보다 크게 높았다. |
| `S4B-G-ROLLOUT-SEMANTIC-CONTENT` | true | final content recall에서 importance 0.3357이 random 0.2401보다 높았다. |
| `S4B-G-ROLLOUT-DRIFT-CHECK` | true | semantic drift proxy는 importance 0.6438이 random 0.7152보다 낮았다. |
| `S4B-G-REPETITION-CHECK` | true | repetition은 importance 0.0501이 random 0.1103보다 낮았다. |
| `S4B-G-BEST-IDENTIFIED` | true | 최고 schedule은 `importance_schedule`이었다. |

Teacher-forced step 결과는 S4a와 거의 같은 방향이다.

| Schedule | Loss | TF Delta Acc | Delta F1 | Delta ROUGE-L | Delta Content | Entity | Repetition | Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `importance_schedule` | 5.7281 | 0.1581 | 0.1694 | 0.1453 | 0.0144 | 0.0082 | 0.1670 | 0.6351 |
| `random_schedule` | 6.7300 | 0.1098 | 0.1230 | 0.1094 | 0.0050 | 0.0079 | 0.1533 | 0.4992 |
| `position_only_schedule` | 5.7781 | 0.1531 | 0.1567 | 0.1389 | 0.0128 | 0.0017 | 0.2561 | 0.6054 |

하지만 더 중요한 것은 rollout final metric이다.

| Schedule | Final F1 | ROUGE-L | Content | Original Content | Entity | Repetition | Drift Proxy | Rollout Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `importance_schedule` | 0.4090 | 0.3099 | 0.3357 | 0.3403 | 0.2855 | 0.0501 | 0.6438 | 0.7336 |
| `random_schedule` | 0.3832 | 0.3259 | 0.2401 | 0.2433 | 0.2050 | 0.1103 | 0.7152 | 0.6215 |
| `position_only_schedule` | 0.2190 | 0.1765 | 0.0000 | 0.0000 | 0.0000 | 0.2782 | 0.9235 | 0.1858 |

핵심은 다음이다.

```text
S4a의 one-step delta 우위가 S4b의 multi-step rollout에서도 유지됐다.
```

특히 `position_only_schedule`이 teacher-forced에서는 꽤 가까웠지만 rollout에서는 content/entity가 0으로 무너졌다. 이 점이 중요하다. 위치 marker만으로는 다음 token 분포 일부를 맞힐 수 있지만, 의미 있는 최종 상태를 누적 생성하는 데는 content-bearing semantic skeleton이 필요하다는 해석이 가능해졌다.

또한 random은 ROUGE-L 0.3259로 importance 0.3099보다 조금 높다. 그러나 content recall, original content recall, entity recall, repetition, drift proxy는 모두 importance가 낫다. 이는 random이 표면 순서 겹침 일부에서는 강할 수 있지만, 중심 의미를 유지하며 확장하는 trajectory는 importance 쪽이 더 낫다는 해석을 지지한다.

## 4. 어떤 반론과 혼동 요인을 다뤘는가

첫째, "S4a는 teacher-forced 한 단계 proxy일 뿐"이라는 반론을 다뤘다. S4b는 모델 생성 결과를 다음 입력으로 넣는 rollout이므로 error accumulation을 포함한다. 여기서도 importance가 random보다 높았다.

둘째, position-only confound가 약해졌다. S4a에서는 position-only가 score 0.5889로 꽤 강했다. S4b에서도 teacher-forced score는 0.6054로 높지만, 최종 rollout에서는 content/entity가 0이고 drift가 0.9235로 매우 높다. 따라서 위치 보조 구조만으로 semantic rollout 우위를 설명하기는 어렵다.

셋째, random corruption 반론도 약해졌다. Random은 S4에서 종합 score로 이겼고 S4b에서도 ROUGE-L은 높다. 하지만 S4b의 최종 content/entity/repetition/drift가 모두 importance 쪽으로 기울었다. 이 결과는 random이 표면 token overlap에는 유리할 수 있어도, 의미 중심 확장 궤적에는 불리하다는 해석과 맞는다.

넷째, 아직 open-ended generation은 아니다. S4b는 constrained delta insertion이다. 어느 위치가 새로 열릴지 이미 알고 있고, 그 위치에 delta token을 넣는다. 따라서 `s5_ready=false`는 유지해야 한다.

다섯째, 생성 품질은 아직 낮다. Sample을 보면 importance rollout도 `the`, `of`, `was` 같은 일반 token을 많이 생성하고 문법이 거칠다. 다만 random보다 반복률이 낮고 content/entity 보존이 높다는 점에서 방향성은 좋다.

## 5. 다음 실험에서 어떻게 검증할 것인가

S4b는 지금까지 v2 process claim을 가장 강하게 지지하는 결과다.

방어 가능한 결론은 다음이다.

```text
S4b supports the core process claim:
importance-ordered semantic skeletons produce a better constrained reverse rollout
than random corruption and position-only controls on semantic content, entity retention,
repetition, and drift metrics.
```

한국어로는 다음과 같다.

```text
중심 의미 token을 먼저 남긴 뒤 세부 token을 붙이는 reverse trajectory는
무작위 손상 복원보다 누적 rollout에서 의미 보존과 반복 제어가 더 좋다.
```

다만 다음 단계는 곧바로 S5가 아니라 구조 보정이다. 우선순위는 다음이다.

1. `S4c` 결과와 함께 해석해 token-level infilling 구조가 왜 위치/형식 token 지름길로 무너지는지 분해한다.
2. Content-bearing token, entity, 숫자, rare token에 더 높은 loss weight를 주는 `content-aware delta objective`를 만든다.
3. 쉼표, 공백, subword fragment 같은 형식 token을 primary score에서 분리하고, semantic token metric을 gate 중심에 둔다.
4. 위치 marker별 독립 예측이 아니라 contiguous span 단위의 작은 decoder 또는 insertion transformer를 사용한다.
5. 다음 구조가 안정화되면 S5에서 open-ended generation으로 넘어가되, S4b metric을 reference trajectory gate로 유지한다.
