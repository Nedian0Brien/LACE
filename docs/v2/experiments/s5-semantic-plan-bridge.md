# S5 결과: Semantic Plan Bridge

## 1. 이 실험이 측정한 것

S5는 S4g 이후의 핵심 구조 질문을 확인했다.

```text
semantic skeleton과 surface span 사이에 semantic plan 중간 표현을 넣으면,
새로 생성되는 span 자체의 content/entity collapse를 줄일 수 있는가?
```

S4g는 pretrained `t5-small` decoder에서도 final rollout 우위는 유지했지만, generated-span-only content/entity recall은 모두 0.0000이고 artifact rate는 0.9961이었다. S5는 같은 pretrained text-to-text realizer를 유지하되, 입력 prompt에 `semantic plan` 필드를 추가했다.

실행 정보는 다음이다.

| 항목 | 값 |
|---|---|
| phase | `v2_s5` |
| Kaggle kernel | `dennisparknd/lace-v2-s5-semantic-plan-bridge` version 1 |
| model | `t5-small` via `AutoModelForSeq2SeqLM` |
| data | `wikitext/wikitext-2-raw-v1:train` |
| train samples | 768 |
| eval samples | 192 |
| combined train examples | 24,000 |
| eval examples per condition | 512 |
| rollout eval samples | 48 |
| device | `cuda` |
| ratios | `0.25 -> 0.50 -> 0.75 -> 1.00` |
| output | `outputs/v2_s5/lace_v2_s5/` |

S5는 하나의 phase 안에서 다음 세 stage를 같이 관리했다.

| Stage | 목적 | 결과 |
|---|---|---|
| `stage_1_oracle_plan` | target span에서 추출한 oracle content/entity plan을 주면 span 생성이 개선되는지 확인 | 통과 |
| `stage_2_plan_prediction` | anchor/context heuristic plan이 random plan보다 oracle에 가까운지 확인 | 실패 |
| `stage_3_predicted_plan_rollout` | predicted plan rollout이 no-plan보다 좋아지는지 확인 | 실패 |

비교 조건은 다음이다.

| 조건 | 의미 |
|---|---|
| `oracle_plan_schedule` | importance skeleton과 실제 target span의 semantic plan을 제공 |
| `predicted_plan_schedule` | importance skeleton과 anchor/context heuristic plan을 제공 |
| `no_plan_schedule` | importance skeleton은 주지만 semantic plan 제거 |
| `random_plan_schedule` | importance skeleton에 random semantic plan 제공 |
| `same_position_random_plan_schedule` | 같은 위치 구조에 random token content와 random plan 제공 |
| `wrong_document_plan_schedule` | 같은 위치 구조에 다른 문서의 content/plan 제공 |
| `position_only_plan_schedule` | 위치와 span metadata만 제공 |
| `shuffled_plan_schedule` | oracle plan의 단어 순서만 섞음 |

## 2. 왜 중요한가

S5가 확인해야 했던 것은 "T5를 붙이면 문장이 좋아지는가"가 아니다. 이미 S4g에서 pretrained decoder만으로는 span content/entity collapse가 해결되지 않았다.

S5의 핵심은 다음 구분이다.

```text
1. 올바른 semantic plan이 있으면 surface span realizer는 content를 생성할 수 있는가?
2. 그 semantic plan을 skeleton/anchor에서 예측할 수 있는가?
3. 예측된 plan이 multi-step reverse rollout에서도 실제로 도움이 되는가?
```

1번이 실패하면 semantic plan bridge 자체가 별 의미가 없다. 1번은 성공하지만 2번과 3번이 실패하면, 구조 방향은 맞지만 planner가 아직 없다는 뜻이다.

## 3. 결과가 의미하는 것

S5는 `process_ready=true`, `overall_pass=false`, `s6_ready=false`였다.

Gate 결과는 다음이다.

| Gate | 통과 | 핵심 수치 |
|---|---:|---|
| `S5-G-RUN` | true | 8개 조건 모두 실행 |
| `S5-G-LOSS-FINITE` | true | 모든 eval loss 유한 |
| `S5-G-ORACLE-BEATS-NO-PLAN` | true | 1.4027 > 0.7055 |
| `S5-G-ORACLE-BEATS-RANDOM-PLAN` | true | 1.4027 > 0.7022 |
| `S5-G-ORACLE-BEATS-SAME-POSITION-RANDOM-PLAN` | true | 1.4027 > 0.4423 |
| `S5-G-WRONG-DOC-PLAN-DROPS` | true | 1.4027 > 0.0065 |
| `S5-G-ORACLE-BEATS-POSITION-ONLY` | true | 1.4027 > -0.0471 |
| `S5-G-ORACLE-BEATS-SHUFFLED-PLAN` | false | 1.4027 ≈ 1.3989 |
| `S5-G-ORACLE-SPAN-CONTENT-GAIN-VS-S4G` | true | 0.4144 > 0.0000 |
| `S5-G-ORACLE-SPAN-ENTITY-GAIN-VS-S4G` | true | 0.1191 > 0.0000 |
| `S5-G-ARTIFACT-LOWER-VS-S4G` | true | 0.1699 < 0.9961 |
| `S5-G-PLAN-PREDICTOR-ABOVE-RANDOM` | false | 0.0146 < 0.0459 |
| `S5-G-PREDICTED-BEATS-NO-PLAN` | false | 0.7054 ≈ 0.7055 |
| `S5-G-ROLLOUT-NONREGRESSION-VS-S4G` | true | 1.4027 > 0.7314 |

Teacher-forced span 결과는 다음이다.

| Condition | Loss | Plan Recall | TF Delta Acc | Delta F1 | Content Recall | Entity Recall | Artifact |
|---|---:|---:|---:|---:|---:|---:|---:|
| `oracle_plan_schedule` | 1.2737 | 1.0000 | 0.7065 | 0.4167 | 0.4144 | 0.1191 | 0.1699 |
| `shuffled_plan_schedule` | 1.3712 | 1.0000 | 0.6789 | 0.3555 | 0.3418 | 0.1055 | 0.1855 |
| `predicted_plan_schedule` | 2.9808 | 0.0146 | 0.5189 | 0.0165 | 0.0000 | 0.0000 | 0.5293 |
| `no_plan_schedule` | 2.9295 | 0.5977 | 0.5101 | 0.0133 | 0.0000 | 0.0000 | 0.4922 |
| `random_plan_schedule` | 2.8153 | 0.0459 | 0.5181 | 0.0125 | 0.0000 | 0.0000 | 0.5117 |
| `same_position_random_plan_schedule` | 2.9069 | 0.3421 | 0.5026 | 0.0150 | 0.0000 | 0.0000 | 0.5078 |
| `wrong_document_plan_schedule` | 3.2541 | 0.2539 | 0.5228 | 0.0231 | 0.0000 | 0.0000 | 0.6289 |
| `position_only_plan_schedule` | 3.2452 | 0.5879 | 0.4997 | 0.0076 | 0.0000 | 0.0000 | 0.4727 |

Rollout final 결과는 다음이다.

| Condition | Final F1 | ROUGE-L | Content | Original Content | Entity | Drift | Rollout Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| `oracle_plan_schedule` | 0.5928 | 0.5871 | 0.7329 | 0.7433 | 0.5979 | 0.3000 | 1.4027 |
| `shuffled_plan_schedule` | 0.5939 | 0.5716 | 0.7370 | 0.7474 | 0.5979 | 0.2999 | 1.3989 |
| `predicted_plan_schedule` | 0.3532 | 0.3377 | 0.3354 | 0.3425 | 0.2875 | 0.6388 | 0.7054 |
| `no_plan_schedule` | 0.3546 | 0.3376 | 0.3341 | 0.3412 | 0.2875 | 0.6396 | 0.7055 |
| `random_plan_schedule` | 0.3515 | 0.3351 | 0.3354 | 0.3425 | 0.2875 | 0.6392 | 0.7022 |
| `same_position_random_plan_schedule` | 0.2894 | 0.1592 | 0.1898 | 0.1928 | 0.1775 | 0.7760 | 0.4423 |
| `wrong_document_plan_schedule` | 0.0441 | 0.0407 | 0.0052 | 0.0052 | 0.0052 | 0.9397 | 0.0065 |
| `position_only_plan_schedule` | 0.0318 | 0.0318 | 0.0000 | 0.0000 | 0.0000 | 0.9463 | -0.0471 |

가장 중요한 결과는 둘로 갈라진다.

첫째, oracle semantic plan은 매우 강했다.

```text
S4g span content recall:        0.0000
S5 oracle span content recall:  0.4144

S4g span entity recall:         0.0000
S5 oracle span entity recall:   0.1191

S4g artifact rate:              0.9961
S5 oracle artifact rate:        0.1699
```

이는 "semantic plan이 있으면 realizer가 content span을 생성할 수 있다"는 강한 upper-bound 증거다. 즉 S4g의 실패는 단순히 T5가 span을 못 만드는 문제가 아니라, T5에게 어떤 semantic content를 생성해야 하는지 알려주는 중간 구조가 없었던 문제로 해석할 수 있다.

둘째, predicted plan은 실패했다.

```text
predicted plan recall: 0.0146
random plan recall:    0.0459

predicted rollout score: 0.7054
no-plan rollout score:   0.7055
```

즉 anchor/context에서 단순히 주변 content word를 뽑는 heuristic으로는 target span의 semantic plan을 예측하지 못했다. predicted plan은 no-plan과 거의 같은 결과를 냈고, generated span content/entity recall도 0.0000이었다.

## 4. 긍정적인 점, 부정적인 점, 남은 모호성

긍정적인 점은 S5가 구조 방향의 핵심을 확인했다는 것이다. S4g에서 완전히 무너졌던 generated-span-only content/entity recall이 oracle plan 조건에서는 크게 회복됐다. wrong-document, same-position random, position-only가 모두 무너진 것도 중요하다. 이는 plan이 단순한 길이/위치 힌트가 아니라 문맥에 맞는 semantic content여야 함을 보여준다.

특히 wrong-document rollout score가 0.0065, position-only score가 -0.0471이라는 점은 방어 가능하다. 아무 content나 붙이거나 위치만 주면 reverse expansion이 거의 성립하지 않는다.

부정적인 점도 선명하다. S5는 전체 통과가 아니다. oracle plan은 target content를 일부 누설하는 upper-bound 조건이다. 실제 모델이 해야 할 일은 `predicted_plan_schedule`인데, 이 조건은 no-plan과 거의 구분되지 않았다. 따라서 지금 결과만으로는 S6 open-ended generation으로 넘어갈 수 없다.

또 하나의 중요한 부정적 신호는 `shuffled_plan_schedule`이다. shuffled plan rollout score는 1.3989로 oracle 1.4027과 거의 같다. 이는 이번 S5에서 중요한 것은 plan의 순서가 아니라 content word bag일 가능성을 시사한다. 즉 아직 "문장 구조 계획"이라기보다 "content token list 제공"에 가깝다.

방어 가능한 해석은 다음이다.

```text
S5 confirms that an explicit semantic content plan can repair span-level semantic collapse,
but does not yet show that the model can infer that plan from the skeleton.
```

한국어로는 다음과 같다.

```text
올바른 의미 chunk를 주면 span 생성은 살아난다.
하지만 현재 구조는 그 의미 chunk를 스스로 찾지 못한다.
```

남은 모호성은 다음이다.

- oracle plan은 target content를 직접 사용하므로 실제 reverse model 능력으로 과장하면 안 된다.
- predicted plan은 이번 runner에서 학습된 planner가 아니라 heuristic이다. 따라서 learned planner 가능성은 아직 남아 있다.
- plan recall metric은 function-only span에서 `none` plan을 맞히는 경우를 포함하므로, no-plan/position-only의 plan recall 값은 content-plan 예측력으로 읽으면 안 된다.
- T5-small 1 epoch, 24,000 examples 조건이므로 더 큰 모델이나 longer training의 가능성은 남아 있다.
- WikiText tokenization 때문에 `ricula`, `un`, `@@` 같은 subword artifact가 아직 섞인다.

## 5. 다음 실험에서 어떻게 검증할 것인가

S5의 결론은 다음이다.

```text
stage_1_oracle_plan: pass
stage_2_plan_prediction: fail
stage_3_predicted_plan_rollout: fail
s6_ready: false
```

따라서 다음은 S6가 아니라 S5 내부의 planner 개선이다. 이름은 새 코드네임으로 가지치기하지 않고, S5 안의 다음 stage iteration으로 관리한다.

다음 구현 방향은 다음이다.

1. heuristic predicted plan을 learned semantic planner로 교체한다.
2. planner target은 full surface span이 아니라 content word/entity plan으로 둔다.
3. planner 입력에는 current skeleton, left/right anchor, gap position, transition ratio를 넣는다.
4. 평가에서는 plan recall/F1을 content-applicable span 기준으로 따로 집계한다.
5. learned predicted plan을 surface realizer rollout에 넣어 no-plan/random/wrong-document plan과 비교한다.

다음 gate는 명확하다.

```text
S5 안에서 learned planner가 random plan보다 높은 content-plan recall을 내고,
predicted-plan rollout이 no-plan보다 올라가는가?
```

이 gate를 통과하기 전까지 open-ended generation은 보류한다.
