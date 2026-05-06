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
- [[concepts/lace/attention-scaffold|attention_scaffold]] - attention 수신 점수로 고른 의미 골격과 위치 보조 구조를 결합한 S2 핵심 입력 조건.
- [[concepts/lace/sinusoidal-absolute|sinusoidal_absolute]] - 원래 token index를 사인/코사인 파형 좌표로 바꿔 의미 골격에 더하는 절대 위치 부호화 방식.
- [[concepts/lace/s2a-positional-encoding|S2a-positional encoding]] - S3 전에 위치 보조 구조 후보를 learned/sinusoidal/relative/rotary 방식으로 비교한 실험.
- [[concepts/lace/s3-anchor-baseline-comparison|S3-anchor baseline comparison]] - semantic terminal state와 predicted anchor baseline을 비교했지만 핵심 gate를 통과하지 못한 실험.

## 연구 실험

- [[concepts/lace/s2-의미-골격-문장-복원-학습|S2 의미 골격-문장 복원 학습]] - `attention_scaffold`와 `random_scaffold` 비교를 통해 짧은 복원 학습에서 의미 골격의 이점을 확인한 실험.
- [[concepts/lace/s2a-positional-encoding|S2a-positional encoding]] - `sinusoidal_absolute`를 S3 기본 위치 보조 구조 후보로 식별한 위치 표현 비교 실험.
- [[concepts/lace/s3-anchor-baseline-comparison|S3-anchor baseline comparison]] - `importance_ordered_forward_no_anchor`가 `random_forward_no_anchor`를 이기지 못해 S3a 진단으로 이어진 실험.
