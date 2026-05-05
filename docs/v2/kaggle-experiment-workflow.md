# V2 Kaggle Experiment Workflow

이 문서는 v2 semantic skeleton track에서 Kaggle 실험을 진행할 때의 표준 절차다. v1의 Kaggle 운영 원칙은 유지하되, v2에서는 `docs/v2/` 아래에 plan/result 문서를 남기고 phase 이름에는 `v2_s*` 형식을 사용한다.

## 1. 실험 전 확인

새 v2 실험을 시작할 때는 먼저 다음을 확인한다.

1. [연구기획서.md](./연구기획서.md)의 핵심 주장과 맞는가.
2. [research-questions.md](./research-questions.md)의 어떤 RQ를 검증하는가.
3. [experiment-roadmap.md](./experiment-roadmap.md)의 어느 `S` phase에 해당하는가.
4. v1 결과를 직접 증거로 과장하지 않는가.

## 2. 파일 배치

v2 실험은 다음 구조를 따른다.

| 항목 | 위치 예시 |
|---|---|
| Kaggle runner | `kaggle/v2_s0/run_v2_s0.py` |
| Kaggle metadata | `kaggle/v2_s0/kernel-metadata.json` |
| push script | `scripts/push_kaggle_v2_s0.sh` |
| plan document | `docs/v2/plan/s0-skeleton-pipeline-plan.md` |
| result document | `docs/v2/experiments/s0-skeleton-pipeline.md` |
| outputs | `outputs/v2_s0/` |

Runner는 Kaggle script kernel에서 단독 실행될 수 있게 self-contained로 작성한다.

## 3. 계획서 작성 기준

계획서는 실행 전에 작성한다. 최소 항목은 다음이다.

- 검증할 RQ
- skeleton 생성 방식
- 비교할 baseline
- control 조건
- metric과 gate
- 좋은 결과/나쁜 결과의 해석
- 다음 phase로 넘길 조건

특히 v2에서는 다음 confound를 계획서에서 미리 다룬다.

- keyword bag이 semantic skeleton처럼 보이는 문제
- scorer가 frequency만 따라가는 문제
- reverse model이 skeleton을 무시하는 문제
- teacher-forced metric은 좋아지지만 generation은 실패하는 문제

## 4. 로컬 검증

Kaggle push 전에는 문법과 script 형태를 가볍게 확인한다. 문서 작업이나 계획 작업에는 test suite를 만들거나 실행하지 않는다.

```bash
rtk .venv/bin/python -m py_compile kaggle/v2_s0/run_v2_s0.py
rtk bash -n scripts/push_kaggle_v2_s0.sh
```

이 repo에서는 TDD를 사용하지 않는다. 구현 후에는 실험 위험도에 맞춰 syntax check, small smoke run, 또는 Kaggle smoke/full run 중 필요한 것만 수행한다.

## 5. Kaggle 실행

전용 script를 사용한다.

```bash
rtk bash scripts/push_kaggle_v2_s0.sh
rtk kaggle kernels status <owner>/<kernel-id>
rtk kaggle kernels output <owner>/<kernel-id> -p outputs/v2_s0
```

Kaggle kernel id, version, sample count, data source, output path는 결과 문서에 반드시 기록한다.

## 6. 결과 문서화

결과 문서는 다음 순서로 작성한다.

1. 실험 목적
2. 실행 정보
3. skeleton/baseline/control 정의
4. gate 결과
5. 핵심 metric table
6. metric별 해석
7. confound와 caveat
8. 다음 phase 판단

해석할 때는 다음을 분리한다.

- skeleton preservation
- reverse reconstruction
- skeleton-use evidence
- teacher-forced token proxy
- constrained/open-ended generation

## 7. 마무리

작업 마무리 전에는 다음을 확인한다.

```bash
git diff --check
rtk git status --short --branch
```

완료된 작업은 repo 규칙에 따라 커밋하고 푸쉬한다. 단, 작업 전부터 있던 unrelated user change는 포함하지 않는다.
