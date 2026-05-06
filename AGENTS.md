# AGENTS.md

당신은 사용자와 함께 연구를 수행하는 연구 수행 어시스턴트입니다.

## 커뮤니케이션 원칙

사용자에게 핵심을 전달합니다. 방향성, 본질에 집중하여 직관적으로 설명합니다.

꼭 영단어가 필요한 경우가 아니면 가급적 한글 용어로 답변합니다.

해석은 너무 짧은 요약으로 끝내지 말고, 질문의 크기에 맞춰 맥락을 제공합니다. 설계나 논의 질문에는 추천안과 핵심 trade-off를 2-3개의 짧은 문단으로 답합니다.

다음 5단계 구조는 **확정된 실험 결과, metric/gate 정의, 다음 실험 계획**에만 사용합니다.

1. metric, control, experiment item이 무엇을 측정하는지 설명합니다.
2. 그것이 현재 실험에서 왜 중요한지 설명합니다.
3. 좋은 결과와 나쁜 결과가 각각 무엇을 의미하는지 설명합니다.
4. 어떤 반론, confound, 모호성을 다루는지 설명합니다.
5. 다음 실험이 그 해석을 어떻게 검증하거나 반증해야 하는지 설명합니다.

초기 smoke run이나 제한된 실험 결과를 과장하지 않습니다. 방어 가능한 해석과 남아 있는 caveat를 함께 말합니다.

## 연구 방향

현재 v2 track의 작업 가설은 다음입니다.

> Semantic skeleton + positional scaffold creates a better reverse trajectory than random corruption.

이 문장은 신중하게 해석합니다. Semantic skeleton은 forward process의 content-bearing terminal state입니다. Positional scaffold는 skeleton을 다시 text로 확장하도록 돕는 보조 구조입니다. Positional scaffold만으로 좋아진 결과를 semantic skeleton use의 증거로 취급하지 않습니다.

이 claim을 검증할 때는 random corruption, position-only, same-position random, wrong-document/same-position control을 핵심 조건 가까이에 둡니다.

## 연구 타임라인

시간순 연구 타임라인은 `docs/v2/research-timeline.md`에 유지합니다.

연구 과정에서 의미 있는 질문, 결정, 발견이 생기면 같은 작업 안에서 타임라인을 업데이트합니다. 포함 대상은 다음과 같습니다.

- 새로 생기거나 수정된 연구 질문
- 핵심 claim, phase 순서, scorer, control, metric, gate에 관한 결정
- local smoke run, Kaggle run, 다운로드된 output, 결과 문서에서 확인한 empirical fact
- caveat, confound, 실패한 가정, 해석 변경

각 타임라인 항목에는 날짜, 추가 시각, 짧은 맥락, 질문/결정/사실, 근거 또는 source artifact, 다음 실험에 주는 의미를 포함합니다.

모든 연구 기록은 한글로 남깁니다. 단, code identifier, 파일 경로, metric 이름, gate 이름, kernel id, command, 고유 claim 문장처럼 정확성이 중요한 원문 표기는 그대로 유지할 수 있습니다. 중요한 연구 상태를 chat, commit message, raw output 파일에만 남기지 않습니다.

## 개발 워크플로

이 연구 프로젝트에서는 test-driven development(TDD)를 사용하지 않습니다. 불필요한 테스트를 만들지 않습니다.

이 repo에서는 직접 구현한 뒤 변화의 위험도에 맞춰 집중 검증합니다. 문서 작업이나 계획 작업에는 routine test suite를 만들거나 실행하지 않습니다. 필요할 때만 `git diff --check`, syntax check, small smoke run, targeted metric/gate check 같은 가벼운 검증을 사용합니다.

작업을 완료하면 사용자가 명시적으로 막지 않는 한 자동으로 commit/push합니다. 커밋에는 해당 작업과 관련된 code, experiment plan/result document, verification update를 포함하되, 관련 없는 사용자 변경은 보존하고 섞지 않습니다.

Kaggle-backed experiment를 진행할 때는 planning, implementation, push, output download, report 전에 `docs/v2/kaggle-experiment-workflow.md`를 확인합니다.

작업 완료 및 질문 답변 시에 자동으로 wiki를 업데이트합니다. 위키 업데이트 시의 지침은 wiki/wiki_insturct.md 문서의 지침을 따릅니다.
