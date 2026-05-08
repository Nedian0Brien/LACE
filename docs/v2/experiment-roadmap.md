# V2 Experiment Roadmap

이 문서는 v2 실험을 위한 단계형 로드맵이다. 기존 v1의 Phase 0-3A와 충돌하지 않도록 v2 실험은 `S` prefix를 사용한다. 여기서 `S`는 semantic skeleton track을 뜻한다.

v1에서 이어갈 실험 규율은 [v1-carryover.md](./v1-carryover.md)를 따른다. 특히 proxy metric과 generation metric을 분리하고, skeleton-use control과 강한 baseline을 초반부터 유지한다.

## S0. Skeleton Pipeline Sanity Check

### 질문

텍스트에서 importance score를 계산하고, 단계별 semantic skeleton을 안정적으로 만들 수 있는가?

### 최소 조건

| 조건 | 설명 |
|---|---|
| random masking | 기존 masked DLM류 기본 corruption baseline |
| uniform length skeleton | importance 없이 길이만 맞춘 skeleton |
| frequency/PMI skeleton | 통계 기반 중요 token 보존 |
| attention-received skeleton | encoder attention 기반 중요 token 보존 |

### 평가

- skeleton token count
- keyword/entity recall
- sentence embedding similarity
- compression level별 semantic similarity curve
- random baseline 대비 preservation gap

### 성공 기준

importance-guided skeleton이 random/uniform skeleton보다 핵심 entity와 semantic similarity를 더 잘 보존해야 한다.

## S1. Skeleton Use Controls

### 질문

Reverse model 또는 reconstruction evaluator가 skeleton을 실제로 사용하는가?

### Control

| Control | 목적 |
|---|---|
| correct skeleton | 정상 조건 |
| shuffled skeleton | 순서 민감도 확인 |
| random skeleton | 무작위 핵심 token 대비 |
| wrong-document skeleton | content mismatch 민감도 확인 |
| remove top-k important tokens | 핵심 token 제거 영향 |
| remove low-k tokens | 부가 token 제거 영향 |

### 성공 기준

correct skeleton이 가장 좋아야 하며, wrong-document와 top-k removal에서 성능이 명확히 하락해야 한다.

## S2. Skeleton-to-Text Reconstruction

### 질문

Semantic skeleton에서 원문 또는 이전 compression level로 복원하는 reverse process를 학습할 수 있는가?

### 학습 태스크

```text
x_t -> x_{t-1}
x_t -> x_0
```

### 비교 조건

- random masking reverse
- uniform skeleton reverse
- PMI/frequency skeleton reverse
- attention skeleton reverse
- oracle keyword skeleton upper bound

### 평가

- token reconstruction accuracy
- BLEU/ROUGE
- BERTScore
- entity/relation preservation
- grammar/perplexity
- skeleton faithfulness

### 성공 기준

importance-guided skeleton reverse가 random masking reverse보다 semantic preservation과 reconstruction에서 좋아야 한다.

## S2a. Positional Encoding Comparison

### 질문

S2의 `front/middle/back` 위치 tag보다 더 정석적인 positional encoding이 의미 골격-문장 복원에 도움이 되는가?

### 비교 조건

- no position
- coarse front/middle/back bins
- learned positional embedding
- sinusoidal positional encoding
- relative position bias
- rotary position embedding

### 성공 기준

정식 positional encoding 후보 중 하나가 `no_position`과 `coarse_bins`보다 token F1 또는 ROUGE-L에서 좋아야 한다. 통과하면 S3의 기본 위치 보조 구조로 넘긴다.

## S3. Anchor Baseline Comparison

### 질문

중요 token을 예측해서 보조 조건으로 주는 방식보다, forward terminal state로 보존하는 방식이 더 나은가?

### 핵심 비교

```text
A. Random forward + anchor prediction
B. Importance-ordered forward + no anchor prediction
C. Importance-ordered forward + anchor prediction
D. Random forward + no anchor prediction
```

### 성공 기준

`B > A`이면 v2 핵심 주장이 강해진다. `B ≈ A`라도 더 단순하고 해석 가능한 trajectory라는 주장이 가능하다. `C`가 가장 좋으면 skeleton forward와 anchor prediction이 상보적이라는 후속 방향이 생긴다.

## S3a. Terminal Diagnostic

### 질문

S3에서 `random_forward_no_anchor`가 가장 좋았던 이유가 무엇인가?

### 진단 대상

- `attention_terminal`
- `idf_terminal`
- `random_terminal`
- `same_position_random_terminal`
- `position_only`
- `gold_anchor_oracle`
- `predicted_anchor`

### 성공 기준

`attention_terminal` 또는 `idf_terminal`이 같은 위치/같은 개수 random terminal보다 좋아야 한다. `gold_anchor_oracle`이 크게 좋다면 anchor predictor 품질 병목이고, oracle도 약하면 현재 reverse model 또는 reconstruction proxy가 terminal 정보량 차이를 잘 반영하지 못하는 것이다.

### 결과

S3a는 `diagnostic_ready=true`, `s4_ready=false`였다. `attention_terminal`은 `random_terminal`과 `same_position_random_terminal`보다 높았지만, `position_only`와 차이가 작고 `random_terminal_predicted_anchor`가 최고 조건이었다. 따라서 content terminal 신호는 일부 회복됐지만, 위치 scaffold와 표면 prior confound가 남아 S4로 바로 넘어가지 않는다.

## S3b. Probe Calibration

### 질문

조건별로 새로 학습한 reverse probe가 아니라, 같은 reverse probe를 고정했을 때도 `attention_terminal` 입력이 position-only, random terminal, same-position random보다 충분히 좋은가?

### 핵심 비교

```text
train: attention_terminal
eval: attention_terminal / attention_no_position / attention_shuffled_position
eval: random_terminal / same_position_random_terminal / position_only
eval: predicted anchor / gold anchor oracle
```

### 성공 기준

`attention_terminal`이 `position_only`, `random_terminal`, `same_position_random_terminal`보다 tolerance 이상 높아야 한다. `attention_no_position`이 낮아지면 위치 channel 자체의 필요성은 확인되지만, `attention_shuffled_position`도 낮아져야 정확한 위치 정렬 사용 주장이 강해진다.

### 결과

S3b는 `diagnostic_ready=true`, `s4_ready=false`였다. `attention_no_position`은 score 0.2828로 크게 떨어졌지만, `attention_terminal` score 0.3768은 `position_only` 0.3735, `random_terminal` 0.3724, `same_position_random_terminal` 0.3638보다 tolerance 0.02 이상 높지 않았다. 최고 조건은 `attention_shuffled_position` score 0.3799였다. 따라서 현재 probe는 위치 channel의 존재에는 민감하지만, 의미 terminal content 사용 증거는 아직 약하다.

## S4. Importance-Ordered Reverse Diffusion

### 질문

중요도 낮은 token부터 순차적으로 mask하는 forward process의 역과정이 random corruption schedule보다 더 좋은 reverse expansion curriculum을 만드는가?

### 평가 순서

1. `25% -> 50%` reverse transition
2. `50% -> 75%` reverse transition
3. `75% -> 100%` reverse transition
4. schedule-level trajectory metric 비교

### 결과

S4는 `process_ready=true`, `overall_pass=false`, `s5_ready=false`였다. `random_schedule`은 loss 6.0172, Token F1 0.2300, ROUGE-L 0.1712, score 0.5607로 종합 score에서 `importance_schedule` score 0.4839보다 높았다. 하지만 `importance_schedule`은 target content recall 0.0574, input retention 0.0471, expansion recall 0.0416, original content recall 0.0496, entity recall 0.0831로 같은 의미 계열 지표에서 random보다 모두 높았다. 따라서 "더 좋은 language model trajectory"는 아직 입증하지 못했지만, 중심 의미 보존과 세부 의미 확장 신호는 importance schedule 쪽이 강했다.

다음은 S5가 아니라 `S4a: delta-token reverse objective`다. 전체 target state를 다시 생성하는 대신 새로 unmask될 token/span만 예측하도록 objective를 바꿔야 한다.

## S4a. Delta-Token Reverse Objective

### 질문

다음 상태 전체를 다시 생성하지 않고 새로 unmask될 delta token/span만 예측하게 하면, importance-ordered schedule이 random corruption보다 더 좋은 reverse objective가 되는가?

### 평가 순서

1. 현재 partial state와 이번 단계에서 채울 위치 marker를 encoder 입력으로 준다.
2. 다음 partial state 전체가 아니라 newly unmasked delta token/span만 decoder target으로 둔다.
3. `importance_schedule`, `random_schedule`, `position_only_schedule`을 같은 ratio와 같은 budget으로 비교한다.
4. teacher-forced delta accuracy와 greedy delta generation metric을 분리해서 본다.

### 결과

S4a는 `process_ready=true`, `overall_pass=true`, `structure_review_needed=false`, `s5_ready=false`였다. `importance_schedule`은 loss 5.7282, TF Delta Acc 0.1577, Delta F1 0.1700, ROUGE-L 0.1468, score 0.6366으로 `random_schedule` loss 6.7063, TF Delta Acc 0.1092, Delta F1 0.1258, ROUGE-L 0.1077, score 0.5073보다 높았다. 또한 `position_only_schedule` score 0.5889보다도 높아 content-bearing skeleton 효과가 위치 marker만으로 완전히 설명되지는 않았다.

다만 `position_only_schedule`도 강했고, delta content recall은 position-only 0.0156이 importance 0.0136보다 높았다. Entity recall은 random 0.0175가 importance 0.0115보다 높았고, repetition도 importance 0.1584가 random 0.1062보다 나빴다. 따라서 S4a는 핵심 process claim을 강화하지만 open-ended generation 성공 증거는 아니다.

다음은 S5가 아니라 `S4b: multi-step delta rollout` 또는 `S4c: span-infilling reverse decoder`다. S4a의 긍정 신호를 실제 rollout으로 연결하고, entity/repetition 병목을 구조적으로 줄여야 한다.

## S4b. Multi-Step Delta Rollout

### 질문

S4a의 one-step delta-token objective 우위가 실제 `25% -> 50% -> 75% -> 100%` rollout에서도 error accumulation을 견디는가?

### 평가 순서

1. 각 transition에서 생성한 delta token을 current state에 삽입한다.
2. 생성된 state를 다음 transition의 입력으로 넘긴다.
3. 최종 100% state를 target/original과 비교한다.
4. final content/entity/repetition/drift를 `importance_schedule`, `random_schedule`, `position_only_schedule`로 비교한다.

### 결과

S4b는 `process_ready=true`, `overall_pass=true`, `structure_review_needed=false`, `s5_ready=false`였다. `importance_schedule`은 rollout score 0.7336으로 `random_schedule` 0.6215와 `position_only_schedule` 0.1858을 모두 넘었다. 최종 content recall은 importance 0.3357, random 0.2401, position-only 0.0000이었고, entity recall도 importance 0.2855가 random 0.2050보다 높았다. Repetition은 importance 0.0501이 random 0.1103보다 낮았고, semantic drift proxy도 importance 0.6438이 random 0.7152보다 낮았다.

다만 `random_schedule`은 ROUGE-L 0.3259로 importance 0.3099보다 높았다. 따라서 random이 표면 순서 겹침 일부에서 강하다는 caveat는 남는다. 하지만 S4b는 현재까지 v2 process claim을 가장 강하게 지지하는 결과다.

## S4c. Span-Infilling Reverse Decoder

### 질문

Autoregressive delta decoder 대신 marker-position infilling 구조를 쓰면 반복을 줄이고 semantic skeleton content를 더 잘 사용할 수 있는가?

### 평가 순서

1. 현재 partial state와 새로 채울 위치 marker를 encoder 입력으로 둔다.
2. 각 marker hidden state에서 vocab classifier가 원래 token id를 맞힌다.
3. `importance_schedule`, `random_schedule`, `position_only_schedule`을 masked-token accuracy, content/entity recall, duplicate/repetition으로 비교한다.

### 결과

S4c는 `process_ready=true`, `overall_pass=false`, `structure_review_needed=true`, `s5_ready=false`였다. `importance_schedule`은 score 0.2403과 masked-token accuracy 0.1414로 `random_schedule` score 0.1425, accuracy 0.1121보다 높았다. 하지만 `position_only_schedule`도 masked-token accuracy 0.1414로 같았고, score는 0.3026으로 가장 높았다. 또한 content recall과 entity recall은 세 조건 모두 0이었다.

따라서 naive marker-position infilling은 semantic skeleton을 쓰는 구조로 보기 어렵다. 현재 구조는 의미 content보다 위치, transition 단계, punctuation/whitespace 같은 형식 token 분포를 먼저 학습한다. 다음 구조 보정은 handcrafted content/entity objective가 아니라, S4b의 rollout 신호를 기준으로 두고 current skeleton과 left/right anchor를 직접 쓰는 contiguous gap/span expansion으로 진행한다.

## S4d. Skeleton-Conditioned Gap/Span Expansion

### 질문

같은 gap/span 위치 구조에서 실제 semantic skeleton content와 left/right anchor가 span 생성과 rollout을 개선하는가?

### 평가 순서

1. 전체 target state가 아니라 새로 열릴 contiguous gap/span만 target으로 둔다.
2. 입력에는 current skeleton token, position, timestep, span marker, left/right anchor role을 포함한다.
3. `importance_schedule`, `random_schedule`, `position_only_schedule`, `same_position_random_schedule`, `wrong_document_same_position_schedule`, `no_anchor_gap_only_schedule`을 비교한다.
4. 생성 span을 current state에 삽입하며 `25% -> 50% -> 75% -> 100%` rollout을 평가한다.

### 결과

S4d는 `process_ready=true`, `overall_pass=true`, `structure_review_needed=false`, `s5_ready=false`였다. `importance_schedule` rollout score는 0.7175로 `random_schedule` 0.6145, `position_only_schedule` 0.0133, `same_position_random_schedule` 0.4733, `wrong_document_same_position_schedule` 0.0504, `no_anchor_gap_only_schedule` 0.0300을 모두 이겼다.

Final content recall은 importance 0.3340, random 0.2448, same-position random 0.1952였고, entity recall은 importance 0.2988, random 0.2202, same-position random 0.1763이었다. 따라서 같은 위치 구조에 아무 content나 넣는 효과가 아니라, 문맥에 맞는 semantic skeleton content와 좌우 anchor가 reverse expansion에 실제 정보를 제공한다는 해석이 가능하다.

다만 생성문은 아직 자연스럽지 않고 span-level content recall은 0.0172로 낮다. 또한 random은 ROUGE-L 0.3148로 importance 0.3092보다 조금 높다. 따라서 S4d는 open-ended generation 성공이 아니라 constrained gap/span expansion에서의 content-use 증거로 해석한다.

## S4e. Shared-Condition Semantic Span Expansion

### 질문

S4d의 semantic skeleton 우위가 조건별 모델 분리 때문이 아니라, 하나의 공유 reverse model 안에서도 유지되는가? 또한 generated span 자체의 content/entity recall을 S4d보다 높일 수 있는가?

### 평가 순서

1. S4d의 여섯 schedule 예제를 하나로 합쳐 shared-condition model을 학습한다.
2. 각 예제에는 `condition_id`, `gap_length`, `left_anchor_distance`, `right_anchor_distance`를 추가한다.
3. S4d와 같은 teacher-forced span metric 및 multi-step rollout metric을 계산한다.
4. S4d 대비 span content/entity gain과 final rollout non-regression을 함께 gate로 둔다.

### 결과

S4e는 `process_ready=true`, `overall_pass=false`, `structure_review_needed=true`, `s5_ready=false`였다. `importance_schedule`은 rollout score 0.7569로 `random_schedule` 0.6406, `position_only_schedule` 0.1576, `same_position_random_schedule` 0.4761, `wrong_document_same_position_schedule` 0.1396, `no_anchor_gap_only_schedule` 0.1496을 모두 이겼다.

Final content recall은 importance 0.3464, random 0.2404였고, final entity recall도 importance 0.2976이 random 0.2137보다 높았다. 이는 S4d의 우위가 조건별 모델 분리 때문만은 아니라는 해석을 강화한다.

하지만 S4e의 본래 개선 목표였던 span-level content/entity는 실패했다. Importance span content recall은 0.0029로 S4d 0.0172보다 낮았고, span entity recall은 0.0013으로 S4d 0.0047보다 낮았다. Artifact rate도 importance 0.6906으로 random 0.5259보다 높았다. 따라서 다음은 open-ended scale-up이 아니라 generated span 자체의 의미 정보량을 높이는 구조 개선이다.

## S4g. Pretrained Decoder Semantic Span Expansion

### 질문

S4e의 generated span content/entity collapse가 작은 custom decoder의 언어 prior 부족 때문인가, 아니면 현재 span target 구성과 reverse expansion 구조 자체의 병목인가?

### 평가 순서

1. S4d/S4e의 six-control gap/span expansion 비교를 유지한다.
2. 작은 custom decoder 대신 pretrained `t5-small` seq2seq decoder를 사용한다.
3. condition, transition, span position, anchor distance, current skeleton text를 prompt로 직렬화한다.
4. target은 newly unmasked span text만 둔다.
5. S4e 대비 span content/entity gain과 artifact 감소를 핵심 gate로 둔다.

### 결과

S4g는 `process_ready=true`, `overall_pass=false`, `structure_review_needed=true`, `s5_ready=false`였다. `importance_schedule`은 rollout score 0.7314로 `random_schedule` 0.6404, `position_only_schedule` -0.0237, `same_position_random_schedule` 0.4580, `wrong_document_same_position_schedule` -0.0021, `no_anchor_gap_only_schedule` -0.0513을 모두 이겼다.

Final content recall은 importance 0.3432, random 0.2582였고, final entity recall은 importance 0.2901, random 0.2260이었다. 따라서 pretrained decoder 조건에서도 semantic skeleton과 anchor가 있는 trajectory의 final rollout 우위는 유지됐다.

하지만 S4g가 해결하려던 span-level semantic generation은 실패했다. Importance span content recall과 span entity recall은 모두 0.0000이고, artifact rate는 0.9961로 S4e 0.6906보다 나빠졌다. 따라서 S4e의 실패를 작은 decoder 규모 문제로만 설명하기 어렵다. 다음은 open-ended scale-up이 아니라 span target 단위 재구성, content/function token 분리, anchor-conditioned decoder 구조 개선이다.

## S5. Semantic Plan Bridge

### 질문

Semantic chunk 또는 plan이라는 중간 표현을 주거나 예측하게 하면, generated span 자체의 content/entity recall을 높이고 artifact를 줄일 수 있는가?

### 명명 규칙

과거 대화에서 `S4h`라고 부르던 구조 후보는 구현 phase명이 아니라 S5의 설계 후보로 승격한다. `S4h-0`, `S4h-1`처럼 가지치기하지 않고, S5 하나의 phase 안에서 stage로 관리한다. 세부 규칙은 [experiment-naming-rules.md](./experiment-naming-rules.md)를 따른다.

### 평가 순서

1. `stage_1_oracle_plan`: 원문에서 자동 추출한 oracle semantic plan을 realizer에 제공했을 때 span content/entity가 오르는지 확인한다.
2. `stage_2_plan_prediction`: skeleton과 anchor만으로 semantic plan을 예측할 수 있는지 확인한다.
3. `stage_3_predicted_plan_rollout`: 예측된 plan을 surface realizer에 넣어 multi-step rollout에서도 도움이 되는지 확인한다.

### 비교 조건

- oracle semantic plan
- random semantic plan
- same-position random plan
- wrong-document plan
- no-plan
- position-only plan
- shuffled plan

### 성공 기준

S5의 첫 gate는 open-ended generation이 아니다. 다음이 확인되어야 한다.

- `oracle_plan_schedule`이 no-plan, random-plan, wrong-document, position-only control을 이긴다.
- generated-span-only content/entity recall이 S4g의 0.0000에서 유의하게 오른다.
- artifact rate가 S4g의 0.9961에서 크게 내려간다.
- final rollout score가 S4d/S4e/S4g보다 크게 퇴행하지 않는다.

### 결과

S5 version 1은 `process_ready=true`, `overall_pass=false`, `s6_ready=false`였다. `stage_1_oracle_plan`은 통과했지만, `stage_2_plan_prediction`과 `stage_3_predicted_plan_rollout`은 실패했다.

Oracle semantic plan 조건은 S4g의 span collapse를 크게 회복했다. `oracle_plan_schedule`의 generated-span content recall은 0.4144, entity recall은 0.1191, artifact rate는 0.1699였다. S4g의 content/entity recall 0.0000, artifact 0.9961과 비교하면 semantic plan bridge 자체는 강한 upper-bound 효과를 보였다. Rollout score도 1.4027로 `no_plan_schedule` 0.7055, `random_plan_schedule` 0.7022, `same_position_random_plan_schedule` 0.4423, `wrong_document_plan_schedule` 0.0065, `position_only_plan_schedule` -0.0471을 모두 이겼다.

하지만 predicted plan 조건은 실패했다. `predicted_plan_schedule`의 plan recall은 0.0146으로 `random_plan_schedule` 0.0459보다 낮았고, generated-span content/entity recall은 0.0000이었다. Rollout score도 0.7054로 `no_plan_schedule` 0.7055와 사실상 같았다. 따라서 S5는 "올바른 semantic plan이 있으면 span 생성이 살아난다"는 것은 확인했지만, "모델이 skeleton/anchor에서 그 plan을 찾을 수 있다"는 것은 아직 확인하지 못했다.

또한 `shuffled_plan_schedule`은 rollout score 1.3989로 oracle 1.4027과 거의 같았다. 현재 plan 효과는 순서 있는 문장 계획이라기보다 content word bag 제공 효과에 가깝다.

## S6. Open-ended Generation

### 질문

Semantic skeleton 기반 reverse process가 open-ended generation에서 random corruption baseline보다 의미 일관성과 반복 제어를 개선하는가?

### 평가

- semantic consistency
- repetition rate
- diversity
- coherence
- skeleton faithfulness
- human preference 또는 LLM-as-judge 보조 평가

## 현재 다음 단계 추천

S0, S1, S2, S2a는 통과했고, S3는 실행됐지만 핵심 gate를 통과하지 못했다. S3a/S3b는 terminal probe confound를 분해했고, S4는 importance-ordered reverse transition을 process-level로 비교했다. S4에서 random은 종합 score와 표면 복원 지표가 더 좋았지만, importance는 의미 보존/확장 지표가 더 좋았다. S4a에서 objective를 newly unmasked delta token/span 예측으로 바꾸자 importance가 random과 position-only를 모두 이겼다. S4b에서는 이 우위가 multi-step rollout에서도 유지됐고, 특히 position-only가 semantic final state에서 무너졌다. S4c의 naive marker-position infilling은 position-only confound와 content/entity collapse로 실패했다. S4d는 left/right anchor role을 쓰는 gap/span expansion으로 바꾸자 same-position random, wrong-document, no-anchor control까지 모두 이겼다. S4e에서는 이 우위가 shared-condition model에서도 유지됐지만, generated span 자체의 content/entity recall은 S4d보다 낮아졌다. S4g에서는 pretrained decoder를 붙여도 final rollout 우위는 유지됐지만 span content/entity recall은 0.0000으로 더 무너졌고 artifact rate는 0.9961까지 올랐다. S5에서는 oracle semantic plan이 span content recall 0.4144와 artifact 0.1699로 S4g collapse를 회복했지만, heuristic predicted plan은 random plan보다 낮고 no-plan rollout과 거의 같았다.

따라서 다음도 아직 open-ended generation이 아니다. S5 내부에서 learned semantic planner를 만들어야 한다. Heuristic planner는 연구 대상이 아니라 smoke/control/ablation으로만 둔다. 다음 gate는 skeleton/anchor/gap query에서 content word/entity plan을 예측하고, 그 learned predicted plan이 no-plan/random/wrong-document plan보다 rollout을 개선하는지 확인하는 것이다.

결과 문서는 다음에 있다.

- [experiments/s0-skeleton-pipeline.md](./experiments/s0-skeleton-pipeline.md)
- [experiments/s1-skeleton-use-controls.md](./experiments/s1-skeleton-use-controls.md)
- [experiments/s2-skeleton-to-text-reconstruction.md](./experiments/s2-skeleton-to-text-reconstruction.md)
- [experiments/s2a-positional-encoding.md](./experiments/s2a-positional-encoding.md)
- [experiments/s3-anchor-baseline-comparison.md](./experiments/s3-anchor-baseline-comparison.md)
- [experiments/s3a-terminal-diagnostic.md](./experiments/s3a-terminal-diagnostic.md)
- [experiments/s3b-probe-calibration.md](./experiments/s3b-probe-calibration.md)
- [experiments/s4-importance-ordered-reverse-diffusion.md](./experiments/s4-importance-ordered-reverse-diffusion.md)
- [experiments/s4a-delta-token-reverse-objective.md](./experiments/s4a-delta-token-reverse-objective.md)
- [experiments/s4b-multi-step-delta-rollout.md](./experiments/s4b-multi-step-delta-rollout.md)
- [experiments/s4c-span-infilling-reverse-decoder.md](./experiments/s4c-span-infilling-reverse-decoder.md)
- [experiments/s4d-skeleton-conditioned-gap-span-expansion.md](./experiments/s4d-skeleton-conditioned-gap-span-expansion.md)
- [experiments/s4e-shared-condition-semantic-span-expansion.md](./experiments/s4e-shared-condition-semantic-span-expansion.md)
- [experiments/s4g-pretrained-decoder-semantic-span-expansion.md](./experiments/s4g-pretrained-decoder-semantic-span-expansion.md)
- [experiments/s5-semantic-plan-bridge.md](./experiments/s5-semantic-plan-bridge.md)

핵심 판단:

1. IDF/attention skeleton은 random/uniform보다 의미 보존 신호를 보였다.
2. `position_prior`가 강해서 lead-position confound는 실제 risk로 확인됐다.
3. S1에서 `attention_correct`는 `random_same_count`, `wrong_document`, `position_only`, `same_position_random`보다 강했다.
4. 하지만 `position_prior`도 강하므로 S2에서는 위치 보조 구조와 위치 전용 control을 함께 유지한다.
5. `remove_topk`가 `remove_lowk`보다 더 치명적이라는 순서 주장은 아직 확인되지 않았다.
6. S2에서 `attention_scaffold`는 `random_scaffold`, `position_only`, wrong-document control보다 강했다.
7. 다만 `idf_scaffold`는 loss가 가장 낮고, `position_prior_scaffold`는 keyword recall이 매우 높아서 scorer와 위치 편향 분리는 계속 필요하다.
8. S2의 `front/middle/back`은 정식 positional encoding이 아니라 coarse tag였으므로 S2a에서 learned/sinusoidal/relative/rotary 방식을 비교한다.
9. S2a에서는 `sinusoidal_absolute`가 가장 좋은 positional scaffold 후보였다. 다만 `coarse_bins` 대비 개선 폭은 작고 생성 품질은 아직 낮다.
10. S3에서는 `importance_ordered_forward_no_anchor`가 `random_forward_anchor_prediction`과는 tolerance 안에서 비슷했지만, `random_forward_no_anchor`보다 낮았다. 따라서 S4로 바로 넘어가지 않는다.
11. S3a에서는 `attention_terminal`이 `random_terminal`과 `same_position_random_terminal`보다 높았지만, `position_only`와 차이가 작고 `random_terminal_predicted_anchor`가 최고였다.
12. S3b에서는 `attention_no_position`이 크게 떨어져 위치 channel의 존재는 중요했지만, `attention_terminal`은 `position_only`, `random_terminal`, `same_position_random_terminal`보다 tolerance 이상 높지 않았고 `attention_shuffled_position`이 최고였다.
13. S4에서는 `random_schedule`이 종합 score와 Token F1/ROUGE-L에서 높았지만, `importance_schedule`은 target content recall, input retention, expansion recall, original content recall, entity recall에서 모두 random보다 높았다.
14. S4a에서는 전체 target state가 아니라 newly unmasked delta token/span만 예측하도록 바꾸자 `importance_schedule`이 score 0.6366, TF Delta Acc 0.1577로 `random_schedule` score 0.5073, TF Delta Acc 0.1092를 이겼다.
15. S4b에서는 generated delta를 다음 step 입력으로 넣는 multi-step rollout에서도 `importance_schedule`이 rollout score 0.7336으로 `random_schedule` 0.6215와 `position_only_schedule` 0.1858을 이겼다.
16. S4c에서는 marker-position infilling이 random보다 mask accuracy는 높였지만, position-only가 같은 accuracy와 더 높은 score를 보여 semantic content 사용 증거로 방어할 수 없었다.
17. S4d에서는 같은 gap/span 위치 구조에서도 `importance_schedule`이 rollout score 0.7175로 `random_schedule` 0.6145, `same_position_random_schedule` 0.4733, `wrong_document_same_position_schedule` 0.0504, `no_anchor_gap_only_schedule` 0.0300을 모두 이겼다.
18. S4e에서는 shared-condition model에서도 `importance_schedule`이 rollout score 0.7569로 모든 strict control을 이겼지만, span content recall은 0.0029로 S4d보다 낮고 artifact rate는 0.6906으로 높았다.
19. S4g에서는 pretrained `t5-small` decoder를 사용해도 `importance_schedule` rollout score가 0.7314로 strict control을 모두 이겼지만, span content/entity recall은 모두 0.0000이고 artifact rate는 0.9961로 높아졌다.
20. S5에서는 `oracle_plan_schedule`이 span content recall 0.4144, entity recall 0.1191, artifact rate 0.1699로 S4g collapse를 회복했지만, `predicted_plan_schedule`은 plan recall 0.0146으로 `random_plan_schedule` 0.0459보다 낮고 no-plan rollout과 거의 같았다.

다음 Kaggle 실험 후보는 새 `S` 코드네임이 아니라 S5 내부의 planner 개선 stage다.

```text
S5: learned semantic planner iteration
```

산출물:

- heuristic planner를 primary condition에서 제외하고 learned semantic planner로 교체
- planner target은 full surface span이 아니라 content word/entity plan으로 구성
- content-applicable span 기준 plan recall/F1을 별도 집계
- predicted-plan rollout이 no-plan/random/wrong-document plan보다 좋아지는지 확인
- S6 open-ended generation으로 넘기기 전 predicted plan bottleneck을 해결

구현 계획은 [plan/s5-learned-semantic-planner-plan.md](./plan/s5-learned-semantic-planner-plan.md)에 둔다. 핵심은 planner model과 plan-conditioned realizer를 분리하고, rollout step마다 현재 skeleton에서 learned plan을 다시 생성하는 것이다.
