---
title: "S4a-delta token reverse objective"
type: "concept"
tags: [LACE, v2, S4a, delta-token, reverse-diffusion]
created: 2026-05-06
updated: 2026-05-06
sources: [outputs/v2_s4a/lace_v2_s4a/summary.md, docs/v2/experiments/s4a-delta-token-reverse-objective.md]
---

# S4a-delta token reverse objective

S4a는 [[s4-importance-ordered-reverse-diffusion|S4-importance ordered reverse diffusion]]의 objective를 보정한 실험이다. 다음 상태 전체를 다시 생성하지 않고, 현재 partial state에서 다음 reverse step으로 갈 때 새로 unmask될 token/span만 예측한다.

```text
input:  현재 partial state + 이번 단계에서 채울 위치 marker
target: newly unmasked delta token/span
```

## 핵심 결과

S4a는 `process_ready=true`, `overall_pass=true`, `structure_review_needed=false`, `s5_ready=false`였다.

| schedule | score | TF Delta Acc | Delta F1 | Delta Content | repetition |
|---|---:|---:|---:|---:|---:|
| `importance_schedule` | 0.6366 | 0.1577 | 0.1700 | 0.0136 | 0.1584 |
| `random_schedule` | 0.5073 | 0.1092 | 0.1258 | 0.0073 | 0.1062 |
| `position_only_schedule` | 0.5889 | 0.1483 | 0.1381 | 0.0156 | 0.1994 |

S4에서 `random_schedule`이 종합 score에서 이겼던 것과 달리, S4a에서는 [[의미-골격|semantic skeleton]]을 쓰는 `importance_schedule`이 `random_schedule`과 `position_only_schedule`을 모두 이겼다. 이는 S4의 random 우위가 가설 폐기보다 objective mismatch였다는 해석을 강화한다.

## 남은 모호성

`position_only_schedule`이 여전히 강하다. 특히 TF Delta Acc는 importance 0.1577, position-only 0.1483으로 가깝고, delta content recall은 position-only 0.0156이 importance 0.0136보다 높다. 따라서 [[위치-보조-구조|위치 보조 구조]]의 prior를 계속 강한 control로 유지해야 한다.

Entity recall은 random이 더 높았다.

```text
importance entity recall: 0.0115
random entity recall:     0.0175
```

Repetition도 importance가 나빴다.

```text
importance repetition: 0.1584
random repetition:     0.1062
```

## 다음 방향

S4a는 [[forward-reverse-process-본질|Forward-Reverse Process 본질]]에 더 가까운 긍정 증거지만 open-ended generation 성공은 아니다. 다음은 S5가 아니라 `S4b: multi-step delta rollout` 또는 `S4c: span-infilling reverse decoder`다.

특히 다음 병목을 다뤄야 한다.

- 여러 reverse step을 누적할 때 semantic drift가 커지는지 확인한다.
- 자유 생성 decoder를 위치 marker별 span infilling 구조로 바꿔 repetition을 줄인다.
- entity/숫자/rare token에 별도 loss weight 또는 copy/pointer 보조 head를 둔다.
- `position_only`, same-position random, wrong-document/same-position control을 계속 핵심 비교군으로 유지한다.
