---
title: "LACE 연구 위키 변경 로그"
type: "summary"
tags: [LACE, 연구위키, 로그]
created: 2026-05-06
updated: 2026-05-07
sources: [사용자 대화]
---

# LACE 연구 위키 변경 로그

## [2026-05-06] ingest | S2 복원 평가 지표 설명

- `F1 score`와 `ROUGE-L`의 직관적 의미를 [[concepts/lace/복원-평가지표-token-f1-rouge-l|복원 평가 지표 - Token F1과 ROUGE-L]]에 수집했다.
- S2의 핵심 비교인 `attention_scaffold` 대 `random_scaffold`에서 두 지표가 어떻게 해석되는지 기록했다.
- 관련 개념인 [[concepts/lace/의미-골격|의미 골격]], [[concepts/lace/위치-보조-구조|위치 보조 구조]], [[concepts/lace/s2-의미-골격-문장-복원-학습|S2 의미 골격-문장 복원 학습]] 페이지를 함께 생성했다.
- 초기 위키 색인 `wiki/index.md`를 생성했다.

## [2026-05-06] ingest | attention_scaffold 구조 설명

- `attention_scaffold`의 실제 입력 구조와 생성 절차를 [[concepts/lace/attention-scaffold|attention_scaffold]]에 수집했다.
- attention 기반 의미 골격과 위치 보조 구조가 함께 입력된다는 점을 명시했다.

## [2026-05-06] ingest | S2a positional encoding 비교

- `learned_absolute`, `sinusoidal_absolute`, `relative_position_bias`, `rotary_position` 비교 결과를 [[concepts/lace/s2a-positional-encoding|S2a-positional encoding]]에 수집했다.
- `sinusoidal_absolute`가 가장 좋은 S3 위치 보조 구조 후보로 식별됐지만, coarse bin 대비 개선 폭은 작고 생성 품질 증거는 아니라는 caveat를 기록했다.

## [2026-05-06] structure | v2 연구 진행 현황 페이지 신설

- `web/index.html`에 v2 트랙 단일 진입점 페이지를 만들었다. 기존 위키 개념 ([[concepts/lace/의미-골격|의미 골격]], [[concepts/lace/위치-보조-구조|위치 보조 구조]], [[concepts/lace/attention-scaffold|attention_scaffold]], [[concepts/lace/s2-의미-골격-문장-복원-학습|S2 복원 학습]]) 과 `docs/v2/research-timeline.md` 의 raw 수치를 종합한 synthesis view다.
- 페이지는 hero, status strip, S0–S5 phase 카드, 핵심 개념 3카드, S1·S2 metric bar chart, 최근 timeline 7개, 남은 질문 4개, footer로 구성되며 `design/design-system.html` 토큰을 그대로 재사용한다.
- 새 위키 개념 페이지는 추가하지 않았다. 이 페이지는 기존 개념·실험 결과를 모아 보여주는 진행 상황 dashboard다. 다음 단계 결과가 통과하면 페이지의 phase 카드와 metric bar를 동일 위치에서 갱신한다.

## [2026-05-06] structure | sinusoidal_absolute 독립 문서화

- S2a 문서 안에 있던 `sinusoidal_absolute` 설명을 [[concepts/lace/sinusoidal-absolute|sinusoidal_absolute]] 독립 개념 문서로 분리했다.
- S2a 문서는 실험 결과 중심으로 유지하고, 위치 부호화 방식 자체의 설명은 새 문서로 연결했다.

## [2026-05-06] ingest | S3 anchor baseline comparison 결과

- S3 결과를 [[concepts/lace/s3-anchor-baseline-comparison|S3-anchor baseline comparison]]에 수집했다.
- `importance_ordered_forward_no_anchor`가 `random_forward_no_anchor`보다 낮아 S3 핵심 gate가 실패했고, 다음 단계가 S4가 아니라 S3a 진단 실험이라는 해석을 기록했다.

## [2026-05-06] query synthesis | S3 이후 방법론 부족점

- S3 실패 이후 더 나은 방법론을 만들기 전에 분해해야 할 결핍을 [[concepts/lace/s3-이후-방법론-부족점|S3 이후 방법론 부족점]]에 정리했다.
- 핵심 부족점은 `better reverse trajectory` 운영 정의, random terminal baseline 강도, 약한 predicted anchor baseline, scorer 선택, reverse probe 민감도, 위치 편향/content 사용 분리로 정리했다.

## [2026-05-06] ingest | S3a terminal diagnostic 결과

- S3a 결과를 [[concepts/lace/s3a-terminal-diagnostic|S3a-terminal diagnostic]]에 수집했다.
- `attention_terminal`이 `random_terminal`과 `same_position_random_terminal`보다 높아 content terminal 신호는 회복됐지만, `position_only`와의 차이가 작고 `random_terminal_predicted_anchor`가 최고여서 S4 진입은 보류한다는 해석을 기록했다.

## [2026-05-06] ingest | S3b probe calibration 결과

- S3b 결과를 [[concepts/lace/s3b-probe-calibration|S3b-probe calibration]]에 수집했다.
- 같은 reverse probe에서 평가 입력만 바꿨을 때 `attention_no_position`은 크게 떨어졌지만, `attention_terminal`이 `position_only`, `random_terminal`, `same_position_random_terminal`보다 충분히 높지는 않아 content terminal 사용 증거가 아직 약하다는 해석을 기록했다.
- 다음 방향을 S4가 아니라 반복 감소와 content-use metric 강화를 포함한 S3c 성격의 보정으로 정리했다.

## [2026-05-06] query synthesis | Forward-Reverse Process 본질 재정렬

- 사용자의 문제 제기를 바탕으로 [[concepts/lace/forward-reverse-process-본질|Forward-Reverse Process 본질]]을 생성했다.
- 문장 exact reconstruction이 아니라 중요도 기반 forward masking schedule과 reverse expansion process가 random corruption보다 더 좋은 diffusion language model을 만드는지가 핵심임을 기록했다.
- 다음 실험 초점을 S3c 지엽적 probe 보정보다 process-level schedule/objective 비교로 옮겼다.

## [2026-05-06] ingest | S4 importance-ordered reverse diffusion 결과

- S4 결과를 [[concepts/lace/s4-importance-ordered-reverse-diffusion|S4-importance ordered reverse diffusion]]에 수집했다.
- `random_schedule`이 종합 score와 표면 복원 지표에서 높았지만, `importance_schedule`은 target content recall, input retention, expansion recall, original content recall, entity recall에서 모두 높았다는 분화된 결과를 기록했다.
- 다음 방향을 S5가 아니라 `S4a: delta-token reverse objective`로 정리했다.

## [2026-05-06] ingest | S4a delta-token reverse objective 결과

- S4a 결과를 [[concepts/lace/s4a-delta-token-reverse-objective|S4a-delta token reverse objective]]에 수집했다.
- 전체 target state가 아니라 newly unmasked delta token/span만 예측하게 하자 `importance_schedule`이 score 0.6366, TF Delta Acc 0.1577로 `random_schedule` score 0.5073, TF Delta Acc 0.1092를 이겼다는 결과를 기록했다.
- Entity recall과 repetition gate는 실패했으므로 다음 방향을 S5가 아니라 `S4b: multi-step delta rollout` 또는 `S4c: span-infilling reverse decoder`로 정리했다.

## [2026-05-06] ingest | S4b/S4c 병렬 실험 결과

- S4b 결과를 [[concepts/lace/s4b-multi-step-delta-rollout|S4b multi-step delta rollout]]에 수집했다.
- S4b에서 `importance_schedule`이 rollout score 0.7336으로 `random_schedule` 0.6215와 `position_only_schedule` 0.1858을 이긴 결과를 기록했다.
- S4c 결과를 [[concepts/lace/s4c-span-infilling-reverse-decoder|S4c span-infilling reverse decoder]]에 수집했다.
- S4c는 random보다 masked-token accuracy가 높았지만 position-only가 같은 accuracy와 더 높은 score를 보여 naive marker infilling을 실패 진단으로 정리했다.

## [2026-05-06] ingest | S4d skeleton-conditioned gap/span expansion 결과

- S4d 결과를 [[concepts/lace/s4d-skeleton-conditioned-gap-span-expansion|S4d skeleton-conditioned gap/span expansion]]에 수집했다.
- `importance_schedule` rollout score 0.7175가 `random_schedule` 0.6145, `same_position_random_schedule` 0.4733, `wrong_document_same_position_schedule` 0.0504, `no_anchor_gap_only_schedule` 0.0300을 모두 이긴 결과를 기록했다.
- S4d는 같은 gap/span 위치 구조에서도 실제 semantic skeleton content와 좌우 anchor가 있어야 reverse expansion이 강해진다는 해석을 강화하지만, open-ended generation 증거는 아니라고 caveat를 남겼다.

## [2026-05-07] ingest | S4e shared-condition semantic span expansion 결과

- S4e 결과를 [[concepts/lace/s4e-shared-condition-semantic-span-expansion|S4e shared-condition semantic span expansion]]에 수집했다.
- `importance_schedule` rollout score 0.7569가 `random_schedule` 0.6406, `same_position_random_schedule` 0.4761, `wrong_document_same_position_schedule` 0.1396, `no_anchor_gap_only_schedule` 0.1496을 모두 이긴 결과를 기록했다.
- 다만 span content recall은 0.0029로 S4d보다 낮고 artifact rate는 0.6906으로 높아, 다음 단계가 S5 scale-up이 아니라 generated span content/entity collapse를 줄이는 구조 개선임을 기록했다.

## [2026-05-07] ingest | S4g pretrained decoder semantic span expansion 결과

- S4g 결과를 [[concepts/lace/s4g-pretrained-decoder-semantic-span-expansion|S4g pretrained decoder semantic span expansion]]에 수집했다.
- pretrained `t5-small` decoder 조건에서도 `importance_schedule` rollout score 0.7314가 `random_schedule` 0.6404, `same_position_random_schedule` 0.4580, `wrong_document_same_position_schedule` -0.0021, `no_anchor_gap_only_schedule` -0.0513을 이긴 결과를 기록했다.
- 하지만 span content/entity recall은 모두 0.0000이고 artifact rate는 0.9961이어서, S4e 실패를 작은 decoder 규모 문제만으로 설명하기 어렵고 다음 단계가 S5 scale-up이 아니라 span target/decoder 구조 개선임을 기록했다.

## [2026-05-07] structure | S4g 연구 중간 체크포인트 HTML 생성

- `web/research-checkpoint-s4g.html`을 생성해 S0부터 S4g까지의 연구 상태를 process signal, span semantic collapse, S5 readiness, 다음 구조 개선 gate로 나누어 시각화했다.
- `web/index.html`에 S4g 결과 카드, 최신 타임라인 항목, 체크포인트 링크를 추가했다.
- 체크포인트는 `design/design-system.html`의 중립 색상, 단색 accent, dense metric card 스타일을 따르되, S4g 이후 핵심 판단인 "open-ended generation 보류, S5 구조 개선"이 첫 화면에서 보이도록 구성했다.

## [2026-05-07] structure | S5 Semantic Plan Bridge 설명 HTML 생성

- `web/s5-semantic-plan-bridge.html`을 생성해 [[concepts/lace/s5-semantic-plan-bridge|S5 Semantic Plan Bridge]] 구조를 한국어 문장 예시와 시각 흐름으로 설명했다.
- 설명서는 S4g의 구조적 문제를 subword/punctuation 조각 복원으로 진단하고, S5를 content chunk planning과 surface realization의 계층형 reverse process로 제시한다.
- `web/index.html`과 `web/research-checkpoint-s4g.html`에서 S5 설명서로 이동하는 링크를 추가했다.

## [2026-05-07] structure | v2 실험 코드네임 규칙 확정

- `docs/v2/experiment-naming-rules.md`를 생성해 `S{n}`은 독립 연구 질문 또는 phase gate, `stage`는 같은 phase 안의 내부 단계, `condition`은 같은 runner 안의 비교 조건으로 정의했다.
- 과거에 `S4h`라고 부르던 구조 후보를 구현 phase명으로 쓰지 않고 `S5: Semantic Plan Bridge`로 승격했다.
- open-ended generation은 S5가 통과한 뒤의 `S6`로 정리했다.

## [2026-05-07] ingest | S5 Semantic Plan Bridge 결과

- S5 결과를 [[concepts/lace/s5-semantic-plan-bridge|S5 Semantic Plan Bridge]]에 수집했다.
- `oracle_plan_schedule`은 span content recall 0.4144, entity recall 0.1191, artifact rate 0.1699로 S4g의 span collapse를 회복했고 rollout score 1.4027로 no-plan, random-plan, wrong-document, position-only control을 모두 이겼다.
- 하지만 `predicted_plan_schedule`은 plan recall 0.0146으로 random plan 0.0459보다 낮고 no-plan rollout과 거의 같아, S6가 아니라 S5 내부 learned semantic planner가 다음 병목임을 기록했다.

## [2026-05-07] interpretation | S5 predicted plan 실패 해석 보정

- 사용자의 지적을 반영해 [[concepts/lace/s5-semantic-plan-bridge|S5 Semantic Plan Bridge]]에 scale/data/capacity caveat를 추가했다.
- S5 version 1의 predicted plan 실패는 semantic plan 구조 폐기가 아니라, `t5-small`, 768 train samples, condition별 3,000 example cap, 1 epoch, heuristic planner 조건의 한계로 읽는 편이 더 방어 가능하다고 기록했다.
- 다음 방향은 S6가 아니라 S5 내부 learned semantic planner scale-up으로 정리했다.
