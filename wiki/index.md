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

## 연구 실험

- [[concepts/lace/s2-의미-골격-문장-복원-학습|S2 의미 골격-문장 복원 학습]] - `attention_scaffold`와 `random_scaffold` 비교를 통해 짧은 복원 학습에서 의미 골격의 이점을 확인한 실험.
