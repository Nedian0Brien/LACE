---
title: "LACE 연구 위키 변경 로그"
type: "summary"
tags: [LACE, 연구위키, 로그]
created: 2026-05-06
updated: 2026-05-06
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

## [2026-05-06] structure | v2 연구 진행 현황 페이지 신설

- `web/index.html`에 v2 트랙 단일 진입점 페이지를 만들었다. 기존 위키 개념 ([[concepts/lace/의미-골격|의미 골격]], [[concepts/lace/위치-보조-구조|위치 보조 구조]], [[concepts/lace/attention-scaffold|attention_scaffold]], [[concepts/lace/s2-의미-골격-문장-복원-학습|S2 복원 학습]]) 과 `docs/v2/research-timeline.md` 의 raw 수치를 종합한 synthesis view다.
- 페이지는 hero, status strip, S0–S5 phase 카드, 핵심 개념 3카드, S1·S2 metric bar chart, 최근 timeline 7개, 남은 질문 4개, footer로 구성되며 `design/design-system.html` 토큰을 그대로 재사용한다.
- 새 위키 개념 페이지는 추가하지 않았다. 이 페이지는 기존 개념·실험 결과를 모아 보여주는 진행 상황 dashboard다. 다음 단계 결과가 통과하면 페이지의 phase 카드와 metric bar를 동일 위치에서 갱신한다.
