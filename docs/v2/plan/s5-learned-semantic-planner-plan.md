# S5 Learned Semantic Planner Implementation Plan

작성 시각: 2026-05-08 09:16 KST

## 1. 무엇을 측정하는가

S5의 다음 primary condition은 heuristic predicted plan이 아니라 learned semantic planner다.

측정 대상은 다음이다.

- 입력: current skeleton, left/right anchor, gap query, transition ratio
- 출력: target span의 content word/entity plan
- downstream: learned plan을 span realizer에 넣었을 때 no-plan/random/wrong-document/heuristic plan보다 rollout이 좋아지는지

핵심 metric은 두 층으로 나눈다.

| 층 | metric | 의미 |
| --- | --- | --- |
| planner 자체 | content-applicable span 기준 plan precision/recall/F1 | 모델이 실제로 target content plan을 예측했는가 |
| realizer downstream | generated span content/entity recall, artifact rate, rollout score | 예측된 plan이 reverse expansion에 실제로 도움이 되는가 |

## 2. 왜 중요한가

S5 version 1은 oracle plan upper-bound가 강하다는 것을 보였다. 하지만 heuristic predicted plan은 random plan보다 낮았고 no-plan rollout과 거의 같았다.

따라서 다음 질문은 "더 좋은 heuristic을 만들 수 있는가"가 아니다. 연구적으로 중요한 질문은 다음이다.

```text
모델이 semantic skeleton과 anchor 구조만 보고,
다음에 채워야 할 content word/entity plan을 학습할 수 있는가?
```

이 질문이 통과해야 LACE v2 claim이 hand-crafted extractor가 아니라 learned reverse process 구조로 방어된다.

## 3. 좋은 결과와 나쁜 결과

좋은 결과는 learned planner가 content-applicable span에서 random plan과 heuristic control보다 높은 plan recall/F1을 내고, 그 learned plan을 넣은 rollout이 no-plan보다 좋아지는 것이다.

더 강한 좋은 결과는 learned plan rollout이 oracle plan upper-bound의 일부를 회복하는 것이다. 이 경우 S5의 해석은 "semantic plan bridge는 유효하고, planner를 학습하면 reverse process에 연결될 수 있다"가 된다.

나쁜 결과는 learned planner가 random/heuristic control을 이기지 못하거나, plan recall은 올랐지만 rollout은 오르지 않는 경우다. 전자는 skeleton/anchor 입력만으로 plan을 예측하기 어렵다는 뜻이고, 후자는 plan 표현 또는 realizer 연결 방식이 아직 맞지 않는다는 뜻이다.

## 4. 다룰 confound와 모호성

첫째, target leakage를 막아야 한다. Planner input에는 target span text, oracle plan, target state text가 들어가면 안 된다. 현재 state와 gap metadata만 허용한다.

둘째, function-only span의 `none` match가 plan metric을 부풀릴 수 있다. 따라서 전체 plan recall과 별도로 `plan_applicable == 1`인 span만 모아 content-applicable plan recall/F1을 보고한다.

셋째, realizer가 plan을 무시하도록 학습되면 learned plan의 효과를 측정할 수 없다. 반대로 oracle plan만 보고 학습하면 no-plan baseline이 지나치게 불리한 out-of-distribution 조건이 될 수 있다. Primary realizer는 oracle plan examples에 plan-dropout examples를 섞어 학습한다. 즉 올바른 plan을 쓰는 법을 배우되, `semantic plan: none`도 같은 입력 형식 안에서 경험하게 한다. Random/wrong-document plan은 학습에 넣지 않고 eval control로만 둔다.

넷째, rollout에서는 이전 step의 생성 결과가 다음 current skeleton이 된다. 따라서 eval set에서 한 번 plan을 미리 만들어 캐시하는 방식만으로는 부족하다. Rollout step마다 현재 skeleton을 다시 planner에 넣어 plan을 생성해야 한다.

다섯째, S5 version 1에서 shuffled plan이 oracle과 거의 같았다. 따라서 다음 실험은 plan order보다 content word bag 예측이 핵심이라는 해석을 유지하고, order-sensitive claim은 아직 하지 않는다.

## 5. 구현 방식

S5 내부 iteration으로 유지하되, 기존 `run_v2_s5.py`를 직접 덮어쓰기보다 별도 runner를 두는 편이 안전하다.

```text
kaggle/v2_s5/run_v2_s5_learned_planner.py
```

구조는 두 모델 파이프라인이다.

```text
current skeleton + anchors + gap query + transition ratio
        |
        v
learned semantic planner
        |
        v
content/entity plan
        |
        v
plan-conditioned span realizer
        |
        v
newly unmasked span
```

구현 단위는 다음이다.

1. `planner_source_text_for_example(item)`을 만든다.
   - 포함: transition, span positions, span length, left/right anchor position, left/right distance, current skeleton
   - 제외: target text, oracle plan, target state text
2. `make_planner_examples(...)`를 만든다.
   - 기존 `make_transition_examples(...)`의 span 분해와 encoder field를 재사용한다.
   - target은 `oracle_plan`이다.
3. `train_planner_model(...)`을 만든다.
   - `AutoModelForSeq2SeqLM`으로 planner를 별도 학습한다.
   - 초기 구현은 `t5-small`, planner epochs 3, train samples 4096 이상을 기본 후보로 둔다.
4. `generate_learned_plan(...)`을 만든다.
   - teacher eval에서는 eval span마다 learned plan을 생성해 `semantic_plan`에 넣는다.
   - rollout eval에서는 각 step의 current skeleton에서 동적으로 plan을 생성한다.
5. `train_realizer_model(...)`을 분리한다.
   - primary realizer는 oracle plan examples와 plan-dropout examples로 학습한다.
   - plan-dropout은 일부 train examples의 `semantic_plan`만 `none`으로 바꾸고 target span은 그대로 두는 방식이다.
   - eval에서는 같은 realizer에 oracle/learned/no/random/wrong/heuristic plan을 주입한다.
6. schedule을 정리한다.
   - primary: `learned_plan_schedule`
   - upper-bound: `oracle_plan_schedule`
   - controls: `no_plan_schedule`, `random_plan_schedule`, `wrong_document_plan_schedule`, `position_only_plan_schedule`
   - diagnostic only: `heuristic_plan_control_schedule`, `shuffled_plan_schedule`
7. gate를 바꾼다.
   - `S5-G-LEARNED-PLAN-ABOVE-RANDOM-APPLICABLE`
   - `S5-G-LEARNED-PLAN-ABOVE-HEURISTIC-APPLICABLE`
   - `S5-G-LEARNED-BEATS-NO-PLAN`
   - `S5-G-LEARNED-SPAN-CONTENT-NONZERO`
   - `S5-G-ORACLE-UPPER-BOUND-STILL-HOLDS`

이 구현에서 가장 중요한 점은 learned planner가 realizer보다 앞에 있는 독립 학습 문제라는 것이다. 즉 S5의 다음 실험은 "span을 더 잘 생성하는 decoder"가 아니라 "reverse expansion 전에 무엇을 채워야 하는지 계획하는 모델"을 검증한다.
