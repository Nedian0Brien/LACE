---
title: "S4e shared-condition semantic span expansion"
type: "concept"
tags: [LACE, v2, S4e, semantic-skeleton, reverse-process, span-expansion, shared-condition]
created: 2026-05-07
updated: 2026-05-07
sources: [docs/v2/experiments/s4e-shared-condition-semantic-span-expansion.md, outputs/v2_s4e/lace_v2_s4e/summary.md]
---

# S4e shared-condition semantic span expansion

S4e는 [[s4d-skeleton-conditioned-gap-span-expansion|S4d]]의 semantic skeleton 우위가 조건별 모델 분리 때문인지 확인한 구조 실험이다. S4d는 strict control을 모두 이겼지만 schedule마다 별도 reverse model을 학습했다. S4e는 여섯 schedule 예제를 하나로 합쳐 하나의 shared-condition model을 학습했다.

입력 구조는 S4d의 gap/span expansion에 condition 정보를 더한 형태다.

```text
input: current skeleton tokens + positions
       left/right semantic anchor role
       newly opened span marker positions
       timestep
       condition id
       gap length and anchor distances

target: newly unmasked span token ids
```

## 핵심 결과

S4e는 `process_ready=true`, `overall_pass=false`, `structure_review_needed=true`, `s5_ready=false`였다.

| Schedule | Rollout score | Final content | Entity | Repetition | Drift |
|---|---:|---:|---:|---:|---:|
| `importance_schedule` | 0.7569 | 0.3464 | 0.2976 | 0.0375 | 0.6323 |
| `random_schedule` | 0.6406 | 0.2404 | 0.2137 | 0.0428 | 0.7130 |
| `position_only_schedule` | 0.1576 | 0.0065 | 0.0079 | 0.0807 | 0.9240 |
| `same_position_random_schedule` | 0.4761 | 0.1819 | 0.1725 | 0.0653 | 0.7774 |
| `wrong_document_same_position_schedule` | 0.1396 | 0.0127 | 0.0132 | 0.0368 | 0.9220 |
| `no_anchor_gap_only_schedule` | 0.1496 | 0.0049 | 0.0074 | 0.2975 | 0.9230 |

공유 모델 안에서도 `importance_schedule`은 random, position-only, same-position random, wrong-document, no-anchor control을 모두 이겼다. 따라서 S4d의 rollout 우위를 단순히 조건별 모델 분리 효과로만 설명하기는 어렵다.

## 실패 지점

S4e의 본래 개선 목표는 generated span 자체의 content/entity recall을 높이는 것이었다. 이 gate는 실패했다.

| Metric | S4d | S4e |
|---|---:|---:|
| span content recall | 0.0172 | 0.0029 |
| span entity recall | 0.0047 | 0.0013 |
| artifact rate | 미정의 | 0.6906 |

Sample에서는 `Cu`, `Mc`, 쉼표, `and`, `the`, 짧은 subword 같은 artifact가 자주 나타났다. 따라서 높은 final rollout score를 좋은 span generation으로 해석하면 안 된다.

## 해석

S4e는 [[forward-reverse-process-본질|Forward-Reverse Process 본질]]에 대해 중요한 분리 실험이다. 공유 모델에서도 [[의미 골격]]과 좌우 anchor가 있으면 final rollout이 좋아진다는 점은 확인했다.

하지만 현재 decoder는 새로 붙이는 span을 의미 단위로 잘 생성하지 못한다. 더 방어 가능한 결론은 다음이다.

```text
semantic skeleton의 final rollout 우위는 모델 분리 때문만은 아니다.
그러나 generated span 자체의 semantic content collapse는 아직 해결되지 않았다.
```

## 다음 질문

S4e 이후 바로 S5 open-ended generation으로 가면 artifact와 content/entity collapse를 더 큰 비용으로 반복할 가능성이 높다. 다음 질문은 다음이다.

```text
semantic skeleton의 final rollout 우위를 유지하면서,
새로 붙이는 span 자체가 실제 content/entity를 담도록 만드는 구조는 무엇인가?
```

후보 방향은 content/function token 분리, 단어 또는 의미 chunk 단위 span target, anchor cross-attention, 조건별 균형 샘플링이다.
