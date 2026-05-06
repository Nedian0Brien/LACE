---
title: "LACE 연구 위키 색인"
type: "summary"
tags: [LACE, 연구위키, 색인]
created: 2026-05-06
updated: 2026-05-06
sources: [사용자 대화, AGENTS.md]
---

# LACE 연구 위키 색인

## 개념

- [[concepts/lace/복원-평가지표-token-f1-rouge-l|복원 평가 지표 - Token F1과 ROUGE-L]] - S2 의미 골격-문장 복원 학습에서 생성문과 정답문의 겹침을 해석하는 핵심 지표.
- [[concepts/lace/의미-골격|의미 골격]] - LACE v2에서 forward process가 도달하는 content-bearing terminal state.
- [[concepts/lace/위치-보조-구조|위치 보조 구조]] - 의미 골격을 문장으로 펼칠 때 위치와 순서 흐름을 보조하는 구조.
- [[concepts/lace/forward-reverse-process-본질|Forward-Reverse Process 본질]] - 중요도 기반 masking schedule과 그 역과정으로 문장을 확장하는 diffusion language model이라는 v2 핵심 문제의 재정의.
- [[concepts/lace/attention-scaffold|attention_scaffold]] - attention 수신 점수로 고른 의미 골격과 위치 보조 구조를 결합한 S2 핵심 입력 조건.
- [[concepts/lace/sinusoidal-absolute|sinusoidal_absolute]] - 원래 token index를 사인/코사인 파형 좌표로 바꿔 의미 골격에 더하는 절대 위치 부호화 방식.
- [[concepts/lace/s2a-positional-encoding|S2a-positional encoding]] - S3 전에 위치 보조 구조 후보를 learned/sinusoidal/relative/rotary 방식으로 비교한 실험.
- [[concepts/lace/s3-anchor-baseline-comparison|S3-anchor baseline comparison]] - semantic terminal state와 predicted anchor baseline을 비교했지만 핵심 gate를 통과하지 못한 실험.
- [[concepts/lace/s3-이후-방법론-부족점|S3 이후 방법론 부족점]] - S3 실패 이후 더 나은 방법론을 만들기 전에 분해해야 할 측정·baseline·scorer·metric 결핍.
- [[concepts/lace/s3a-terminal-diagnostic|S3a-terminal diagnostic]] - S3 실패 후 terminal 정보량, 위치 편향, anchor oracle/predicted anchor 병목을 분리한 진단 실험.
- [[concepts/lace/s3b-probe-calibration|S3b-probe calibration]] - 같은 reverse probe에서 입력만 바꾸어 위치 channel과 terminal content 사용 증거를 분리한 보정 실험.
- [[concepts/lace/s4-importance-ordered-reverse-diffusion|S4-importance ordered reverse diffusion]] - importance schedule이 종합 score에서는 random보다 낮지만 의미 보존/확장 지표에서는 강한 process-level 실험.
- [[concepts/lace/s4a-delta-token-reverse-objective|S4a-delta token reverse objective]] - 전체 state 대신 newly unmasked delta token/span만 예측하자 importance schedule이 random과 position-only를 이긴 실험.

## 연구 실험

- [[concepts/lace/s2-의미-골격-문장-복원-학습|S2 의미 골격-문장 복원 학습]] - `attention_scaffold`와 `random_scaffold` 비교를 통해 짧은 복원 학습에서 의미 골격의 이점을 확인한 실험.
- [[concepts/lace/s2a-positional-encoding|S2a-positional encoding]] - `sinusoidal_absolute`를 S3 기본 위치 보조 구조 후보로 식별한 위치 표현 비교 실험.
- [[concepts/lace/s3-anchor-baseline-comparison|S3-anchor baseline comparison]] - `importance_ordered_forward_no_anchor`가 `random_forward_no_anchor`를 이기지 못해 S3a 진단으로 이어진 실험.
- [[concepts/lace/s3-이후-방법론-부족점|S3 이후 방법론 부족점]] - S4로 넘어가기 전에 S3a에서 분리해야 할 terminal 정보량, 위치 편향, anchor predictor 병목, metric 민감도 문제.
- [[concepts/lace/s3a-terminal-diagnostic|S3a-terminal diagnostic]] - `attention_terminal`이 random/same-position control보다 높았지만 `position_only`와 predicted anchor confound가 남은 실험.
- [[concepts/lace/s3b-probe-calibration|S3b-probe calibration]] - `attention_no_position`은 크게 하락했지만 `attention_terminal`의 content 우위 margin은 충분하지 않아 S4 보류를 재확인한 실험.
- [[concepts/lace/s4-importance-ordered-reverse-diffusion|S4-importance ordered reverse diffusion]] - `25% -> 50% -> 75% -> 100%` reverse transition에서 random은 표면 score, importance는 semantic signal이 강했던 실험.
- [[concepts/lace/s4a-delta-token-reverse-objective|S4a-delta token reverse objective]] - delta-token objective에서 importance가 score 0.6366으로 random 0.5073과 position-only 0.5889를 넘어선 실험.
