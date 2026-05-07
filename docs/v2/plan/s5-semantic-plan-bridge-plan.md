# S5 계획: Semantic Plan Bridge

## 목적

S5는 S4g 이후의 구조 개선 phase다. S4g는 pretrained `t5-small` decoder를 써도 `importance_schedule`의 final rollout 우위는 유지했지만, 새로 생성되는 span 자체의 content/entity recall은 0.0000에 머물렀고 artifact rate는 0.9961까지 올라갔다.

따라서 S5의 핵심 질문은 다음이다.

> skeleton과 surface span 사이에 semantic plan 중간 표현을 두면, generated span 자체가 더 많은 content/entity를 담고 artifact가 줄어드는가?

이 실험은 [../experiment-roadmap.md](../experiment-roadmap.md)의 `S5: Semantic Plan Bridge`에 해당한다. S5는 open-ended generation이 아니라, S6로 넘어가기 전에 constrained reverse expansion의 의미 밀도를 올릴 수 있는지 확인하는 gate다.

## 방법

S5 첫 runner는 S4g의 pretrained text-to-text realizer를 유지하되, 입력 prompt에 `semantic plan` 필드를 추가한다.

```text
input:  condition id/name
        transition ratio
        span positions and span length
        left/right anchor positions and distances
        current skeleton text
        semantic plan

target: newly unmasked span text
```

semantic plan은 target span의 surface 전체가 아니라 content word/entity 후보를 `;`로 연결한 중간 표현이다.

```text
target span: "the Australian colonies"
oracle plan: "Australian ; colonies"
```

첫 실행은 하나의 `v2_s5` phase 안에서 세 stage를 같이 관리한다.

| Stage | 역할 |
|---|---|
| `stage_1_oracle_plan` | oracle semantic plan을 주면 span content/entity와 artifact가 개선되는지 확인한다. |
| `stage_2_plan_prediction` | skeleton/anchor context에서 만든 heuristic predicted plan이 random plan보다 oracle plan에 가까운지 확인한다. |
| `stage_3_predicted_plan_rollout` | predicted plan을 넣은 rollout이 no-plan보다 나은지 확인한다. |

`stage_2_plan_prediction`은 이번 runner에서 학습된 planner가 아니라 anchor/context content를 사용하는 비모수 heuristic으로 둔다. 이유는 먼저 "plan bridge 자체가 유효한가"를 확인해야 하기 때문이다. oracle plan이 효과가 없으면 learned planner를 붙이는 것은 실험을 위한 실험이 된다.

## 비교 조건

| 조건 | 의미 |
|---|---|
| `oracle_plan_schedule` | importance skeleton과 실제 target span의 oracle semantic plan을 제공한다. |
| `predicted_plan_schedule` | importance skeleton과 anchor/context 기반 heuristic plan을 제공한다. |
| `no_plan_schedule` | importance skeleton은 주지만 semantic plan은 제거한다. |
| `random_plan_schedule` | importance skeleton에 random semantic plan을 붙인다. |
| `same_position_random_plan_schedule` | 같은 위치 구조에 random token content와 random plan을 붙인다. |
| `wrong_document_plan_schedule` | 같은 위치 구조에 다른 문서의 content/plan을 붙인다. |
| `position_only_plan_schedule` | 위치와 span metadata만 남기고 skeleton content와 plan을 제거한다. |
| `shuffled_plan_schedule` | oracle plan의 단어 순서를 섞는다. |

## Metric과 gate

| Gate | 통과 조건 | 해석 |
|---|---|---|
| `S5-G-RUN` | 모든 조건의 teacher-forced span 평가와 rollout 평가가 완료된다. | 실험 완결성 |
| `S5-G-LOSS-FINITE` | 모든 조건의 eval loss가 유한하다. | 수치 안정성 |
| `S5-G-ORACLE-BEATS-NO-PLAN` | oracle plan rollout score가 no-plan보다 높다. | plan bridge의 기본 유효성 |
| `S5-G-ORACLE-BEATS-RANDOM-PLAN` | oracle plan이 random plan보다 높다. | 아무 plan이나 넣은 효과가 아님 |
| `S5-G-WRONG-DOC-PLAN-DROPS` | wrong-document plan이 oracle보다 낮다. | 문맥에 맞는 plan이 필요함 |
| `S5-G-ORACLE-SPAN-CONTENT-GAIN-VS-S4G` | oracle span content recall이 S4g의 0.0000에서 오른다. | generated span collapse 개선 |
| `S5-G-ORACLE-SPAN-ENTITY-GAIN-VS-S4G` | oracle span entity recall이 S4g의 0.0000에서 오른다. | entity 생성 개선 |
| `S5-G-ARTIFACT-LOWER-VS-S4G` | oracle artifact rate가 S4g 0.9961보다 낮다. | plan이 표면 artifact를 줄이는지 확인 |
| `S5-G-PLAN-PREDICTOR-ABOVE-RANDOM` | predicted plan recall이 random plan recall보다 높다. | skeleton/anchor에서 plan 예측 신호가 있는지 확인 |
| `S5-G-PREDICTED-BEATS-NO-PLAN` | predicted plan rollout score가 no-plan보다 높다. | oracle이 아닌 plan도 실제 rollout에 도움이 되는지 확인 |
| `S5-G-ROLLOUT-NONREGRESSION-VS-S4G` | oracle rollout score가 S4g 0.7314에서 크게 퇴행하지 않는다. | span 의미 개선이 전체 rollout을 망치지 않는지 확인 |

## 좋은 결과와 나쁜 결과

좋은 결과는 oracle plan이 no-plan, random plan, wrong-document plan, position-only plan을 이기고, generated span content/entity recall과 artifact rate를 동시에 개선하는 것이다. 이 경우 S4g의 실패는 "pretrained decoder 부족"이 아니라 "semantic plan 없이 곧장 surface span을 생성한 구조"의 문제였다고 볼 수 있다.

더 강한 좋은 결과는 predicted plan도 no-plan을 이기는 것이다. 그러면 S5의 다음 iteration은 heuristic plan을 learned planner로 바꾸는 방향이 된다.

나쁜 결과는 oracle plan을 줘도 content/entity recall이 오르지 않거나 artifact가 줄지 않는 경우다. 이 경우 단순 plan prompt만으로는 부족하며, decoder가 anchor representation을 직접 cross-attend하는 구조 또는 span target 자체의 재구성이 필요하다.

## Confound와 caveat

- Oracle plan은 target content를 일부 누설한다. 따라서 이것은 최종 모델 성능이 아니라 upper-bound 구조 진단이다.
- Predicted plan은 이번 runner에서 heuristic이므로, 실패하더라도 learned planner 가능성을 완전히 배제하지 않는다.
- Plan이 좋아져도 final rollout score만 좋아진다면 S4g와 같은 confound가 반복된다. 반드시 generated-span-only content/entity metric을 분리해서 본다.
- `shuffled_plan_schedule`이 oracle과 비슷하면 plan의 순서 구조가 중요하지 않다는 뜻일 수 있다. 반대로 oracle이 shuffled보다 높으면 plan order가 surface realization에 정보를 준다는 신호다.

## 다음 phase 판단

S5가 통과하면 S6 open-ended generation으로 넘어갈 수 있다. 통과 조건은 최소한 다음이다.

```text
1. oracle plan이 strict controls를 이긴다.
2. generated-span-only content/entity recall이 S4g의 0.0000에서 오른다.
3. artifact rate가 S4g의 0.9961에서 내려간다.
4. predicted plan이 random plan보다 oracle에 가깝다.
5. predicted plan rollout이 no-plan보다 낮지 않다.
```

S5가 실패하면 open-ended scale-up은 보류하고, anchor-conditioned decoder 또는 learned semantic planner를 별도 구조 개선으로 검토한다.

실행 후 결정: heuristic planner는 후속 연구 대상이 아니라 진단용 비교 장치로만 둔다. 다음 S5 primary condition은 heuristic을 정교화하는 방향이 아니라 learned semantic planner로 교체하는 방향이어야 한다.
