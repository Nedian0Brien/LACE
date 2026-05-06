---
title: "S4d skeleton-conditioned gap/span expansion"
type: "concept"
tags: [LACE, v2, S4d, semantic-skeleton, reverse-process, span-expansion]
created: 2026-05-06
updated: 2026-05-06
sources: [docs/v2/experiments/s4d-skeleton-conditioned-gap-span-expansion.md, outputs/v2_s4d/lace_v2_s4d/summary.md]
---

# S4d skeleton-conditioned gap/span expansion

S4d는 [[의미 골격]]이 단순 위치 scaffold가 아니라 reverse expansion에 실제 정보를 전달하는지 확인한 구조 실험이다. [[s4b-multi-step-delta-rollout|S4b]]는 multi-step rollout에서 `importance_schedule`이 random과 position-only보다 좋다는 결과를 줬지만, 같은 위치에 아무 token content나 넣어도 되는지까지는 분리하지 못했다.

S4d는 전체 target state를 복원하지 않고, 새로 열릴 contiguous gap/span만 생성했다.

```text
input: current skeleton tokens + positions
       left/right semantic anchor role
       newly opened span marker positions
       timestep

target: newly unmasked span token ids
```

## 핵심 결과

S4d는 `process_ready=true`, `overall_pass=true`, `structure_review_needed=false`, `s5_ready=false`였다.

| Schedule | Rollout score | Final content | Entity | Repetition | Drift |
|---|---:|---:|---:|---:|---:|
| `importance_schedule` | 0.7175 | 0.3340 | 0.2988 | 0.0469 | 0.6433 |
| `random_schedule` | 0.6145 | 0.2448 | 0.2202 | 0.0296 | 0.7114 |
| `position_only_schedule` | 0.0133 | 0.0003 | 0.0000 | 0.4036 | 0.9388 |
| `same_position_random_schedule` | 0.4733 | 0.1952 | 0.1763 | 0.0646 | 0.7704 |
| `wrong_document_same_position_schedule` | 0.0504 | 0.0042 | 0.0047 | 0.0985 | 0.9356 |
| `no_anchor_gap_only_schedule` | 0.0300 | 0.0000 | 0.0000 | 0.8516 | 0.9313 |

가장 중요한 비교는 `same_position_random_schedule`과 `wrong_document_same_position_schedule`이다. 같은 위치에 token content를 넣는 것만으로는 충분하지 않았고, 문맥에 맞는 실제 semantic skeleton content가 있을 때 score와 content/entity recall이 크게 높았다.

## 해석

S4d는 [[forward-reverse-process-본질|Forward-Reverse Process 본질]]에 대해 현재까지 가장 강한 구조적 증거를 제공한다. 같은 gap/span 위치 구조에서도 `importance_schedule`이 random, position-only, same-position random, wrong-document, no-anchor control을 모두 이겼기 때문이다.

다만 이것은 open-ended generation 성공이 아니다. S4d는 constrained gap/span expansion이며, 어떤 위치가 열릴지 이미 알고 있다. 생성문 자체도 아직 자연스럽지 않고, span-level content recall은 0.0172로 낮다.

따라서 방어 가능한 결론은 다음이다.

```text
같은 gap/span 위치 구조에서도 실제 의미 골격 token과 좌우 anchor content가 있을 때
reverse expansion이 random, position-only, wrong-document, no-anchor보다 좋아진다.
```

## 다음 질문

S4d 이후의 질문은 S5 open-ended generation으로 바로 가는 것이 아니라, S4d에서 확인한 semantic anchor 사용 신호를 유지하면서 generated span의 content/entity recall을 어떻게 높일 것인가다.

후보 방향은 shared-condition runner, span boundary-aware decoder, confidence-gated refinement, punctuation/content 분리 decoding, longer semantic span curriculum이다.
