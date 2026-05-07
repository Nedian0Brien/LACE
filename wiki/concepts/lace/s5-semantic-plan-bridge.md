---
title: "S5 Semantic Plan Bridge"
type: "concept"
tags: [LACE, v2, S5, semantic-skeleton, span-expansion, anchor-attention, hierarchical-decoder]
created: 2026-05-07
updated: 2026-05-07
sources: [web/s5-semantic-plan-bridge.html, docs/v2/experiments/s4g-pretrained-decoder-semantic-span-expansion.md, docs/v2/experiments/s5-semantic-plan-bridge.md, docs/v2/experiment-naming-rules.md]
---

# S5 Semantic Plan Bridge

S5는 [[s4g-pretrained-decoder-semantic-span-expansion|S4g]] 이후 제안된 다음 phase다. 과거 대화에서 `S4h`라고 부르던 구조 후보는 구현 코드네임이 아니라 S5의 설계 후보로 승격한다. S4g는 pretrained decoder를 사용해도 span content/entity recall이 0.0000으로 무너졌으므로, 문제를 모델 크기보다 target 단위와 reverse 구조의 문제로 본다.

핵심 아이디어는 다음이다.

```text
subword/punctuation 조각을 바로 맞히지 않는다.
먼저 gap에 들어갈 의미 chunk를 계획하고,
그 다음 표면 문장 span으로 실현한다.
```

## 구조

S5는 다섯 부분으로 구성한다.

| 구성 | 역할 |
|---|---|
| Skeleton Encoder | current skeleton token, 위치, transition ratio를 encoding한다. |
| Gap Query Builder | 연속된 missing span을 하나의 query로 묶고 left/right anchor를 찾는다. |
| Anchor Attention | gap query가 left/right anchor와 global skeleton을 직접 참고한다. |
| Semantic Planner | gap에 들어갈 content word/chunk를 먼저 예측한다. |
| Surface Realizer | content chunk에 조사, 어미, punctuation을 붙여 실제 span text로 실현한다. |

## 직관

예를 들어 다음 skeleton이 있다고 하자.

```text
고양이 / 창가 / 잠들었다
```

S4g식 target은 쉼표, 조사, subword 조각을 바로 맞히게 만들 수 있다. S5는 먼저 다음과 같은 의미 chunk를 만든다.

```text
검은
조용히
```

그 다음 surface realizer가 다음 문장을 만든다.

```text
검은 고양이가 창가에서 조용히 잠들었다.
```

## 실험 gate

S5는 final rollout score만으로 통과시키면 안 된다. S4g에서 final content/entity가 높아 보여도 generated span content/entity는 0.0000이었기 때문이다.

통과 조건은 다음을 분리해서 본다.

- `oracle_plan_schedule` 또는 predicted plan 조건이 no-plan, random-plan, wrong-document, position-only control을 이기는가.
- generated-span-only content/entity recall이 S4g의 0.0000에서 오르는가.
- artifact rate가 S4g의 0.9961에서 내려가는가.
- final rollout score가 S4d/S4e/S4g보다 크게 퇴행하지 않는가.

## S5 version 1 결과

S5 version 1은 [[s4g-pretrained-decoder-semantic-span-expansion|S4g]]의 pretrained text-to-text realizer를 유지하고 `semantic plan` prompt 필드를 추가했다.

결과는 둘로 갈라졌다.

| 조건 | 핵심 결과 |
|---|---|
| `oracle_plan_schedule` | span content recall 0.4144, entity recall 0.1191, artifact 0.1699, rollout score 1.4027 |
| `predicted_plan_schedule` | plan recall 0.0146, span content/entity 0.0000, rollout score 0.7054 |
| `no_plan_schedule` | rollout score 0.7055 |
| `wrong_document_plan_schedule` | rollout score 0.0065 |
| `position_only_plan_schedule` | rollout score -0.0471 |

따라서 S5 version 1은 `stage_1_oracle_plan`은 통과했지만 `stage_2_plan_prediction`과 `stage_3_predicted_plan_rollout`은 실패했다.

핵심 해석은 다음이다.

```text
올바른 의미 chunk를 주면 span 생성은 살아난다.
하지만 현재 구조는 그 의미 chunk를 스스로 찾지 못한다.
```

또한 `shuffled_plan_schedule`이 rollout score 1.3989로 oracle 1.4027과 거의 같았으므로, 이번 plan 효과는 순서 있는 문장 계획이라기보다 content word bag 제공 효과에 가깝다.

## 해석 보정

S5 version 1의 predicted plan 실패는 semantic plan 구조 자체의 폐기 근거로 읽으면 안 된다. 이번 실행은 `t5-small`, train samples 768, condition별 train example cap 3,000, 1 epoch 조건이었고, predicted plan은 learned planner가 아니라 anchor/context 기반 heuristic이었다.

따라서 방어 가능한 해석은 다음이다.

```text
oracle plan은 강하다.
하지만 현재 작은 규모와 heuristic planner로는 그 plan을 스스로 찾지 못한다.
```

다음 S5 iteration은 open-ended generation이 아니라 learned semantic planner scale-up이어야 한다. 이때 plan prediction 자체를 별도 gate로 두고, content-applicable span 기준 plan recall/F1과 predicted-plan rollout 개선을 함께 확인해야 한다.

## 관련 문서

- `web/s5-semantic-plan-bridge.html`
- `docs/v2/experiments/s5-semantic-plan-bridge.md`
- `docs/v2/experiment-naming-rules.md`
- [[s4g-pretrained-decoder-semantic-span-expansion|S4g pretrained decoder semantic span expansion]]
- [[forward-reverse-process-본질|Forward-Reverse Process 본질]]
