---
title: "S5 Semantic Plan Bridge"
type: "concept"
tags: [LACE, v2, S5, semantic-skeleton, span-expansion, anchor-attention, hierarchical-decoder]
created: 2026-05-07
updated: 2026-05-07
sources: [web/s5-semantic-plan-bridge.html, docs/v2/experiments/s4g-pretrained-decoder-semantic-span-expansion.md, docs/v2/experiment-naming-rules.md]
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

- `importance_schedule`이 random, same-position random, wrong-document, no-anchor control을 계속 이기는가.
- generated-span-only content/entity recall이 S4g의 0.0000에서 오르는가.
- artifact rate가 S4g의 0.9961에서 내려가는가.
- final rollout score가 S4d/S4e/S4g보다 크게 퇴행하지 않는가.

## 관련 문서

- `web/s5-semantic-plan-bridge.html`
- `docs/v2/experiment-naming-rules.md`
- [[s4g-pretrained-decoder-semantic-span-expansion|S4g pretrained decoder semantic span expansion]]
- [[forward-reverse-process-본질|Forward-Reverse Process 본질]]
