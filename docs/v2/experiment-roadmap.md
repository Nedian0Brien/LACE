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

## S5. Open-ended Generation

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

S0, S1, S2, S2a는 통과했고, S3는 실행됐지만 핵심 gate를 통과하지 못했다. S3a/S3b는 terminal probe confound를 분해했고, S4는 importance-ordered reverse transition을 process-level로 비교했다. S4에서 random은 종합 score와 표면 복원 지표가 더 좋았지만, importance는 의미 보존/확장 지표가 더 좋았다. S4a에서 objective를 newly unmasked delta token/span 예측으로 바꾸자 importance가 random과 position-only를 모두 이겼다. 따라서 다음은 open-ended S5가 아니라 S4b multi-step delta rollout 또는 S4c span-infilling reverse decoder다.

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

다음 Kaggle 실험 후보는 다음이다.

```text
S4b: multi-step delta rollout
S4c: span-infilling reverse decoder
```

산출물:

- S4a delta objective를 실제 multi-step rollout으로 연결
- generation 단계별 semantic drift와 repetition 누적 측정
- entity recall 실패와 position-only 강세를 다루는 span-infilling 구조 보정
- importance/random/position-only/wrong-document 동일 budget 비교
