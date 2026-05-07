---
title: "v2 experiment naming rules"
type: "concept"
tags: [LACE, v2, experiment-management, naming, research-process]
created: 2026-05-07
updated: 2026-05-07
sources: [docs/v2/experiment-naming-rules.md, 사용자 대화]
---

# v2 experiment naming rules

v2 실험 이름은 구현 세부가 아니라 연구 질문의 위상을 나타내야 한다. S4a-S4g처럼 세부 실험이 길어지면 연구 상태를 직관적으로 파악하기 어려워지므로, 다음 규칙을 사용한다.

| 단위 | 의미 |
|---|---|
| `S{n}` | 독립된 연구 질문 또는 phase gate |
| `stage` | 같은 `S{n}` 안에서 순차적으로 수행되는 내부 단계 |
| `condition` | 같은 runner 안의 비교 조건 |
| `gate` | phase 통과 여부를 판단하는 metric 묶음 |
| `checkpoint` | 결과를 설명하는 산출물 |

## 핵심 결정

새 연구 질문 또는 독립 gate가 생기면 letter suffix를 더 붙이지 않고 다음 정수 phase로 승격한다.

따라서 `S4h-0`, `S4h-1`, `S4h-2` 같은 이름은 사용하지 않는다. 과거에 `S4h`라고 부르던 구조 후보는 [[s5-semantic-plan-bridge|S5 Semantic Plan Bridge]]로 승격한다.

```text
S5: Semantic Plan Bridge
  stage_1_oracle_plan
  stage_2_plan_prediction
  stage_3_predicted_plan_rollout
```

Open-ended generation은 S5가 통과한 뒤의 S6로 둔다.

## 관련 문서

- `docs/v2/experiment-naming-rules.md`
- [[s5-semantic-plan-bridge|S5 Semantic Plan Bridge]]
