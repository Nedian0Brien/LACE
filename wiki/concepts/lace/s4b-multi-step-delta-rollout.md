---
title: "S4b multi-step delta rollout"
type: "concept"
tags: [LACE, v2, S4b, rollout, reverse-process]
created: 2026-05-06
updated: 2026-05-06
sources: [docs/v2/experiments/s4b-multi-step-delta-rollout.md, outputs/v2_s4b/lace_v2_s4b/summary.md]
---

# S4b multi-step delta rollout

S4b는 [[concepts/lace/s4a-delta-token-reverse-objective|S4a-delta token reverse objective]]의 one-step delta 우위가 실제 역방향 궤적에서도 유지되는지 확인한 실험이다.

핵심 차이는 generated delta를 다음 step 입력으로 다시 넣는다는 점이다.

```text
25% skeleton
-> model-generated 50% state
-> model-generated 75% state
-> model-generated 100% state
```

## 결과

`importance_schedule`은 rollout score 0.7336으로 `random_schedule` 0.6215와 `position_only_schedule` 0.1858을 모두 이겼다.

| 지표 | importance | random | position-only |
|---|---:|---:|---:|
| rollout score | 0.7336 | 0.6215 | 0.1858 |
| final content recall | 0.3357 | 0.2401 | 0.0000 |
| final entity recall | 0.2855 | 0.2050 | 0.0000 |
| repetition rate | 0.0501 | 0.1103 | 0.2782 |
| semantic drift proxy | 0.6438 | 0.7152 | 0.9235 |

## 해석

S4b는 현재까지 [[concepts/lace/forward-reverse-process-본질|Forward-Reverse Process 본질]]에 가장 가까운 긍정 결과다. [[concepts/lace/의미-골격|의미 골격]]에서 출발해 세부 token을 붙이는 궤적이 random corruption보다 content/entity 보존과 반복 제어에서 낫다.

다만 open-ended generation 성공은 아니다. 위치가 주어진 constrained delta insertion이고, `random_schedule`은 ROUGE-L 0.3259로 importance 0.3099보다 높았다. 따라서 표면 순서 겹침에서는 random이 여전히 강할 수 있다.

## 다음 의미

S4b는 다음 구조 보정의 기준선이다. 새 decoder 구조는 최소한 S4b의 final content/entity/repetition/drift를 넘어야 한다. [[concepts/lace/s4c-span-infilling-reverse-decoder|S4c-span infilling reverse decoder]]는 이 기준을 넘지 못한 실패 진단으로 취급한다.
