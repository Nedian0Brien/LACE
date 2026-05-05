# Kaggle 실험 진행 방법

이 문서는 LACE/DLM-ALM 연구에서 GPU가 필요한 실험을 Kaggle kernel로 실행할 때의 표준 절차를 정리한다. 목적은 실험을 빠르게 돌리는 것뿐 아니라, 나중에 결과를 다시 해석할 수 있도록 실행 조건, 산출물, 실패 지점, 해석 caveat를 같은 형태로 남기는 것이다.

## 1. 기본 원칙

Kaggle은 이 repo의 원격 GPU 실행 환경이다. 로컬 환경은 주로 runner 문법 검증, unit test, smoke run, 문서화, 결과 분석에 사용한다.

실험을 시작할 때는 먼저 다음 세 가지를 분리한다.

1. **연구 질문**: 이번 실험이 어떤 가설이나 반론을 검증하는가.
2. **실험 구현**: Kaggle에서 실제로 실행할 self-contained runner, metadata, push script.
3. **판정 기준**: 어떤 metric/gate가 좋아지면 다음 단계로 가고, 어떤 결과가 나오면 가설을 약화하거나 폐기할 것인가.

특히 이 연구에서는 hidden-state reconstruction, token-level proxy, frozen decoder readability, open-ended generation, latent-use evidence가 서로 다른 증거라는 점을 유지해야 한다. 하나의 proxy metric만 좋아졌다고 generation path가 해결됐다고 쓰지 않는다.

## 2. 파일 배치 규칙

새 실험 phase를 만들 때는 기존 phase 구조를 복사하되, phase 이름과 kernel id를 명확히 바꾼다.

| 항목 | 위치 예시 | 역할 |
|---|---|---|
| Kaggle runner | `kaggle/phase3a/run_phase3a.py` | Kaggle script kernel에서 실행되는 self-contained 실험 코드 |
| Kaggle metadata | `kaggle/phase3a/kernel-metadata.json` | Kaggle kernel id, title, code file, GPU/internet 설정 |
| Push script | `scripts/push_kaggle_phase3a.sh` | Kaggle 업로드 명령을 phase별로 고정 |
| Local tests | `tests/test_phase3a_runner.py` | config, condition parser, gate logic 등 위험한 해석 로직 검증 |
| Plan document | `docs/plan/phase-3a-...-plan.md` | 실행 전 연구 질문, arm, gates, 예상 해석 |
| Result document | `docs/experiments/phase-3a-....md` | 실행 후 조건, 숫자, gate, 해석, 다음 단계 |
| Downloaded outputs | `outputs/phase3a/lace_phase3a/` | Kaggle에서 내려받은 metrics, logs, samples, checkpoints |

Runner는 Kaggle에 단독 업로드되어도 실행될 수 있어야 한다. 따라서 핵심 실험 로직은 runner 파일 안에 두고, repo-local module import에 과하게 의존하지 않는다.

## 3. 실행 전 계획서 작성

Kaggle에 올리기 전에 plan document를 먼저 작성한다. 계획서에는 최소한 다음 항목이 있어야 한다.

- 이번 실험이 해결하려는 질문
- 기존 phase에서 넘어온 근거와 남은 반론
- 실험 arm 목록과 각 arm의 역할
- control arm의 의미
- 주요 metric/gate와 pass/fail 기준
- 결과가 좋을 때의 다음 단계
- 결과가 나쁠 때의 해석과 rollback 또는 대체 방향

좋은 계획서는 "무엇을 해보겠다"가 아니라 "어떤 결과가 어떤 해석을 가능하게 하거나 막는지"를 미리 적어둔 문서다. 이렇게 해야 실험 후에 proxy metric이 좋아진 쪽으로 해석을 끌고 가는 일을 줄일 수 있다.

## 4. 로컬 검증

Kaggle push 전에는 로컬에서 빠르게 깨지는 문제를 먼저 제거한다. 이 repo에서는 `rtk` wrapper를 사용한다.

대표 명령:

```bash
rtk .venv/bin/python -m py_compile kaggle/<phase>/run_<phase>.py
rtk bash -n scripts/push_kaggle_<phase>.sh
rtk .venv/bin/python -m unittest discover -s tests -q
```

필요하면 작은 sample count로 smoke run을 추가한다. 다만 로컬 smoke run은 GPU full run의 대체물이 아니다. 로컬에서 확인할 수 있는 것은 대체로 다음에 한정된다.

- import와 syntax가 깨지지 않는가
- config default가 의도한 값인가
- condition name parsing이 올바른가
- gate 판정이 edge case에서 잘못 뒤집히지 않는가
- output writer가 `metrics.json`, `summary.md`, `train_log.jsonl` 등을 생성할 수 있는가

## 5. Kaggle push

각 phase는 전용 push script를 둔다.

```bash
rtk bash scripts/push_kaggle_<phase>.sh
```

직접 실행할 경우의 기본 형태는 다음과 같다.

```bash
rtk kaggle kernels push -p kaggle/<phase> --accelerator NvidiaTeslaT4 --timeout 3600
```

`kernel-metadata.json`의 `id`는 결과 문서에 그대로 기록한다. 같은 kernel id를 재사용해 version을 올릴 수 있지만, 연구 phase가 바뀌면 새 kernel id를 만드는 편이 해석 기록에 안전하다.

Kaggle credential이나 network 문제로 push가 실패하면, 실패한 명령과 에러를 문서 또는 작업 보고에 남긴다. 인증 정보는 repo에 기록하지 않는다.

## 6. 상태 확인과 대기

Push 이후에는 kernel id로 상태를 확인한다.

```bash
rtk kaggle kernels status <owner>/<kernel-id>
```

상태가 `running`이면 일정 간격으로 다시 확인한다. 보고할 때는 단순히 "기다리는 중"이라고만 쓰지 말고, 어떤 kernel id가 실행 중인지와 대략 어느 단계인지 함께 남긴다.

완료 상태가 `complete`가 아니면 다음을 확인한다.

- Kaggle log에 Python exception이 있는가
- dependency download 또는 Hugging Face dataset 접근이 실패했는가
- GPU memory 또는 timeout 문제가 있었는가
- runner가 output directory를 만들기 전에 죽었는가

실패한 run도 연구 기록상 의미가 있으면 result document에 실패 원인과 다음 수정 방향을 남긴다.

## 7. 출력 다운로드

Kernel이 완료되면 outputs 아래 phase별 디렉터리로 결과를 내려받는다.

```bash
rtk mkdir -p outputs/<phase>
rtk kaggle kernels output <owner>/<kernel-id> -p outputs/<phase>
```

다운로드 후 확인할 핵심 파일은 다음과 같다.

| 파일 | 확인 내용 |
|---|---|
| `metrics.json` | 모든 정량 지표와 gates가 기록됐는가 |
| `summary.md` | runner가 생성한 자동 요약이 있는가 |
| `train_log.jsonl` | arm/stage별 학습 곡선이 남아 있는가 |
| `generation_samples.jsonl` | qualitative generation 확인이 가능한가 |
| `checkpoints/*.pt` | 재분석 또는 후속 실험에 필요한 checkpoint가 있는가 |

결과 파일이 크면 최종 커밋 전에 무엇을 git에 포함할지 확인한다. 연구 결과를 재현하거나 해석하는 데 필요한 artifact와, 단순 중간 캐시를 구분한다.

## 8. 결과 해석과 문서화

결과 문서는 `docs/experiments/phase-...md`에 작성한다. 최소 구조는 다음을 따른다.

1. 실험 목적
2. 실행 정보
3. 실험 arm과 control 설명
4. gate 결과
5. 핵심 metric table
6. metric별 해석
7. 남은 caveat와 confound
8. 다음 실험 제안

해석할 때는 다음 순서를 유지한다.

1. metric이 무엇을 측정하는지 설명한다.
2. 현재 LACE/DLM-ALM 실험에서 왜 중요한지 설명한다.
3. 좋은 결과와 나쁜 결과가 각각 무엇을 뜻하는지 설명한다.
4. 어떤 반론이나 confound를 다루는지 설명한다.
5. 다음 실험이 이 해석을 어떻게 검증하거나 반증해야 하는지 쓴다.

이 형식은 특히 다음 혼동을 막기 위한 것이다.

- 낮은 MSE가 곧 좋은 generation을 뜻하지 않는다.
- 높은 cosine이 곧 token reconstruction 성공을 뜻하지 않는다.
- teacher-forced token proxy 개선이 곧 open-ended generation 성공을 뜻하지 않는다.
- latent-use sensitivity가 곧 semantic compression 증거는 아니다.
- Gaussian/noise control보다 좋아야 compression forward process의 고유 장점을 더 강하게 말할 수 있다.

## 9. README와 연구노트 갱신

실험 결과 문서를 작성한 뒤에는 다음 문서도 필요하면 갱신한다.

- `docs/experiments/README.md`: phase 목록, 현재 연구 상태, 다음 우선순위
- `docs/연구노트.md`: 연구 흐름과 주요 판단
- 후속 실험 계획서: 다음 phase를 바로 진행할 경우 `docs/plan/...`

README에는 한 줄 결론을 짧게 적고, 세부 해석은 phase별 결과 문서에 둔다. 연구노트에는 "왜 다음 방향을 선택했는지"가 드러나게 쓴다.

## 10. 마무리 검증과 git 정리

작업 종료 전에는 focused verification을 실행한다. 실험 runner나 gate logic을 건드렸다면 전체 unit test를 다시 돌린다.

```bash
rtk .venv/bin/python -m unittest discover -s tests -q
rtk git status --short --branch
```

완료된 작업은 repo 규칙에 따라 커밋하고 푸쉬한다.

```bash
rtk git add <relevant-files>
rtk git commit -m "docs: document Kaggle experiment workflow"
rtk git push origin main
```

단, 작업 시작 전에 이미 있던 unrelated user change는 함부로 포함하지 않는다. 필요한 경우 `git status --short`와 `git diff --stat`로 범위를 확인하고, 이번 작업에서 만든 문서와 직접 관련된 파일만 stage한다.
