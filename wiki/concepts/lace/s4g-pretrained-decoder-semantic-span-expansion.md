---
title: "S4g pretrained decoder semantic span expansion"
type: "concept"
tags: [LACE, v2, S4g, semantic-skeleton, reverse-process, span-expansion, pretrained-decoder]
created: 2026-05-07
updated: 2026-05-07
sources: [docs/v2/experiments/s4g-pretrained-decoder-semantic-span-expansion.md, outputs/v2_s4g/lace_v2_s4g/summary.md]
---

# S4g pretrained decoder semantic span expansion

S4g는 [[s4e-shared-condition-semantic-span-expansion|S4e]] 이후 제기된 "custom decoder가 너무 작아서 span generation이 무너진 것 아닌가"라는 반론을 확인한 실험이다. S4d/S4e의 six-control gap/span expansion 구조를 유지하면서, 작은 custom decoder 대신 pretrained `t5-small` seq2seq decoder를 사용했다.

입력은 text prompt로 직렬화했다.

```text
condition + transition + span positions
left/right anchor positions and distances
current skeleton text
```

target은 이번 단계에서 새로 unmask될 span text다.

## 핵심 결과

S4g는 `process_ready=true`, `overall_pass=false`, `structure_review_needed=true`, `s5_ready=false`였다.

| Schedule | Rollout score | Final content | Entity | Repetition | Drift |
|---|---:|---:|---:|---:|---:|
| `importance_schedule` | 0.7314 | 0.3432 | 0.2901 | 0.0015 | 0.6311 |
| `random_schedule` | 0.6404 | 0.2582 | 0.2260 | 0.0016 | 0.6968 |
| `position_only_schedule` | -0.0237 | 0.0000 | 0.0000 | 0.0553 | 0.9478 |
| `same_position_random_schedule` | 0.4580 | 0.1924 | 0.1817 | 0.0068 | 0.7727 |
| `wrong_document_same_position_schedule` | -0.0021 | 0.0052 | 0.0052 | 0.0994 | 0.9407 |
| `no_anchor_gap_only_schedule` | -0.0513 | 0.0000 | 0.0000 | 0.3884 | 0.9469 |

Pretrained decoder 조건에서도 `importance_schedule`은 random, position-only, same-position random, wrong-document, no-anchor control을 모두 이겼다. 따라서 [[의미 골격]]과 좌우 anchor가 final reverse trajectory에 정보를 준다는 신호는 유지됐다.

## 실패 지점

S4g의 본래 목표는 generated span 자체의 content/entity recall을 올리고 artifact를 줄이는 것이었다. 이 목표는 실패했다.

| Metric | S4d | S4e | S4g |
|---|---:|---:|---:|
| span content recall | 0.0172 | 0.0029 | 0.0000 |
| span entity recall | 0.0047 | 0.0013 | 0.0000 |
| artifact rate | 미정의 | 0.6906 | 0.9961 |

샘플에서는 target이 `brown`, `Australia`, `character`, `10` 같은 content token이어도 prediction이 쉼표, `s`, 빈 조각에 가까운 token으로 수렴했다. 따라서 S4g의 final content recall은 새 span 생성 성공이 아니라 current skeleton에 남아 있던 content 보존 효과로 해석해야 한다.

## 해석

S4g는 [[forward-reverse-process-본질|Forward-Reverse Process 본질]] 관점에서 중요한 실패다. pretrained decoder를 붙였는데도 span content/entity가 0으로 무너졌으므로, S4e 실패를 작은 decoder 규모 문제만으로 설명하기 어렵다.

방어 가능한 결론은 다음이다.

```text
semantic skeleton의 final rollout 우위는 pretrained decoder 조건에서도 유지된다.
하지만 현재 text-prompt span objective는 새 span을 의미 단위로 생성하게 만들지 못한다.
```

## 다음 질문

S4g 이후 S5 open-ended generation으로 바로 가면 artifact와 content/entity collapse를 더 큰 비용으로 반복할 가능성이 높다. 다음 질문은 다음이다.

```text
semantic skeleton의 final rollout 우위를 유지하면서,
새로 unmask되는 span 자체가 실제 content/entity를 담도록 만드는 target 단위와 decoder 구조는 무엇인가?
```

후보 방향은 content/function token 분리, 단어 또는 의미 chunk 단위 target, anchor-conditioned decoder, 조건별 균형 샘플링이다.
