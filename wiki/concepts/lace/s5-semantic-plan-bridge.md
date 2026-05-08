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

다만 이 말은 heuristic planner를 더 정교하게 만드는 방향으로 가자는 뜻이 아니다. Heuristic planner는 연구 대상이나 방법론적 기여가 아니라, learned planner를 붙이기 전 병목을 드러내는 진단용 비교 장치다. LACE v2의 핵심 주장은 hand-crafted extractor가 아니라 reverse process 안에서 semantic plan을 학습할 수 있는 구조에 있어야 한다.

따라서 방어 가능한 해석은 다음이다.

```text
oracle plan은 강하다.
하지만 현재 구조는 그 plan을 스스로 학습해 찾는 능력을 아직 보이지 못했다.
```

다음 S5 iteration은 open-ended generation이 아니라 learned semantic planner여야 한다. 이때 heuristic은 primary condition에서 제외하고 smoke/control/ablation으로만 유지한다. Plan prediction 자체를 별도 gate로 두고, content-applicable span 기준 plan recall/F1과 predicted-plan rollout 개선을 함께 확인해야 한다.

## Learned planner 구현 확인

다음 구현은 단순히 `predicted_plan_from_context()` 휴리스틱을 더 복잡하게 바꾸는 방식이 아니다. Planner를 별도 seq2seq 모델로 두고, current skeleton, left/right anchor, gap query, transition ratio에서 oracle content/entity plan을 예측하도록 학습한다.

Pipeline은 두 단계다.

```text
current skeleton + anchors + gap query
-> learned semantic planner
-> content/entity plan
-> plan-conditioned span realizer
-> newly unmasked span
```

특히 rollout에서는 이전 step의 생성 결과가 다음 current skeleton이 되므로, eval 시작 전에 learned plan을 한 번만 캐시하면 부족하다. 각 rollout step에서 현재 skeleton을 planner에 넣어 plan을 다시 생성해야 한다.

Primary condition은 `learned_plan_schedule`로 두고, heuristic은 `heuristic_plan_control_schedule`처럼 diagnostic control로만 남긴다. Plan metric은 function-only span의 `none` 정답이 섞인 전체 평균과 별도로, `plan_applicable == 1`인 content-applicable span 기준 recall/F1을 반드시 보고한다.

Realizer는 oracle plan만으로 학습하지 않는다. Oracle plan examples에 plan-dropout examples를 섞어 `semantic plan: none`도 같은 입력 형식에서 경험하게 한다. 이렇게 해야 no-plan baseline이 단순 out-of-distribution 불이익을 받지 않는다. Random/wrong-document plan은 학습에 넣지 않고 eval control로만 둔다.

## Claude 방법론 검토

2026-05-08에 Claude에게 LACE 전체 연구 컨셉과 S5 learned semantic planner 계획에 대한 외부 방법론 검토를 요청했다.

검토의 핵심은 조건부 수용이다. LACE의 기본 직관은 방어 가능하지만, S5 learned planner가 성공하더라도 그것이 diffusion language model의 claim인지, 아니면 planner + conditional realizer pipeline의 claim인지 분리해야 한다고 지적했다.

특히 다음 세 가지를 강한 risk로 보았다.

- `shuffled_plan_schedule`이 oracle plan과 거의 같은 성능을 낸 것은 현재 plan이 ordered sentence plan이 아니라 content word bag으로 작동한다는 신호다.
- S4g에서 pretrained `t5-small`을 붙였는데도 generated span content/entity recall이 0.0000이고 artifact rate가 0.9961로 무너진 원인을 먼저 이해해야 한다.
- Learned planner를 추가하기 전에 direct seq2seq baseline, ordered-vs-shuffled learned plan, planner recall threshold별 downstream 분석, plan-dropout sensitivity가 필요하다.

따라서 다음 runner의 성공 기준은 no-plan/random 대비 개선만으로 두면 약하다. `learned_plan_schedule`은 oracle gap의 일부를 회복하고, direct seq2seq baseline과도 경쟁해야 한다.

## 중간 진단

2026-05-08 중간 진단의 핵심은 다음이다.

```text
LACE v2는 새로운 DLM forward process 후보로서 일부 근거를 얻었다.
하지만 아직 full Diffusion Language Model은 아니다.
```

입증된 것은 constrained reverse rollout에서 semantic skeleton + positional scaffold가 random/position-only보다 의미 보존 궤적을 더 잘 만든다는 점이다. S4b와 S4d가 이 주장을 가장 강하게 지지한다. S5 oracle plan은 올바른 content/entity plan이 있으면 span semantic collapse가 크게 회복된다는 upper-bound를 보였다.

입증되지 않은 것은 모델이 그 plan을 스스로 예측할 수 있는지, ordered semantic plan이 필요한지, learned planner + realizer가 direct seq2seq baseline보다 나은지, 그리고 이것이 표준 diffusion formalism으로 정식화될 수 있는지다.

따라서 S5 learned planner는 단순 후속 개선이 아니라, LACE forward process가 실제 learned reverse expansion으로 이어지는지 확인하는 핵심 gate다.

## 관련 문서

- `web/s5-semantic-plan-bridge.html`
- `docs/v2/experiments/s5-semantic-plan-bridge.md`
- `docs/v2/plan/s5-learned-semantic-planner-plan.md`
- `docs/v2/reviews/midpoint-forward-process-diagnosis.md`
- `docs/v2/reviews/claude-s5-learned-planner-methodology-review.md`
- `docs/v2/experiment-naming-rules.md`
- [[s4g-pretrained-decoder-semantic-span-expansion|S4g pretrained decoder semantic span expansion]]
- [[forward-reverse-process-본질|Forward-Reverse Process 본질]]
