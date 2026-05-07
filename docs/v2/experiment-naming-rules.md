# V2 실험 코드네임 규칙

이 문서는 v2 연구의 실험 이름이 과도하게 가지치기되는 문제를 막기 위한 규칙이다.

핵심 원칙은 다음이다.

```text
코드네임은 실험의 구현 세부가 아니라 연구 질문의 위상을 나타낸다.
```

따라서 이름은 "무엇을 조금 바꿨는가"가 아니라 "어떤 gate를 검증하는가"를 보여줘야 한다.

## 1. 기본 단위

| 단위 | 의미 | 예시 |
|---|---|---|
| `S{n}` | 독립된 연구 질문 또는 phase gate | `S4`, `S5` |
| `stage` | 같은 `S{n}` 안에서 순차적으로 수행되는 내부 단계 | `stage_1_oracle_plan`, `stage_2_plan_prediction` |
| `condition` | 같은 runner 안의 비교 조건 | `importance_schedule`, `wrong_document_plan` |
| `gate` | phase 통과 여부를 판단하는 metric 묶음 | `S5-G-SPAN-CONTENT-GAIN` |
| `checkpoint` | 결과를 설명하는 산출물 | `web/research-checkpoint-s4g.html` |

## 2. `S{n}`을 올리는 기준

새 정수 phase를 만든다.

- 연구 질문이 바뀐다.
- 평가 gate가 바뀐다.
- 성공/실패 해석이 이전 phase와 독립적으로 필요하다.
- 다음 단계로 넘어가기 위한 go/no-go 판단이 생긴다.
- 구조가 너무 달라져 기존 phase의 variant로 부르면 오해가 생긴다.

예를 들어 S4는 "importance-ordered reverse process가 random보다 좋은가"를 확인하는 묶음이었다. S4a-S4g는 모두 이 질문 안에서 objective, rollout, span 구조, pretrained decoder 병목을 분해했다.

하지만 semantic chunk 중간 표현이 유효한지 확인하는 실험은 더 이상 S4의 작은 variant가 아니다. 이 질문은 다음 phase로 승격한다.

```text
S5: Semantic Plan Bridge
```

## 3. letter suffix 사용 제한

`S4a`, `S4b` 같은 letter suffix는 앞으로 제한적으로만 쓴다.

사용 가능:

- 같은 연구 질문 안에서 실험 장치를 조금 바꾸는 경우
- 같은 metric/gate 체계를 유지하는 경우
- 하나의 phase 안에서 최대 3개 정도의 빠른 진단만 필요한 경우

사용 금지:

- `S4h-0`, `S4h-1`처럼 letter 뒤에 다시 하위 번호를 붙이는 방식
- 같은 letter 아래에서 여러 runner, 여러 문서, 여러 gate가 생기는 방식
- 구조적으로 다른 연구 질문을 "이전 phase의 후속 variant"처럼 붙이는 방식

즉 다음은 금지한다.

```text
S4h-0
S4h-1
S4h-2
S4h-oracle
S4h-predicted
```

대신 다음처럼 둔다.

```text
S5: Semantic Plan Bridge
  stage_1_oracle_plan
  stage_2_plan_prediction
  stage_3_predicted_plan_rollout
```

## 4. runner와 문서 이름

runner와 문서는 phase를 기준으로 이름 붙인다.

```text
kaggle/v2_s5/run_v2_s5.py
scripts/push_kaggle_v2_s5.sh
docs/v2/plan/s5-semantic-plan-bridge-plan.md
docs/v2/experiments/s5-semantic-plan-bridge.md
```

내부 stage는 파일명을 늘리지 않고 runner config와 결과 문서 안에서 구분한다.

```json
{
  "phase": "v2_s5",
  "stages": [
    "stage_1_oracle_plan",
    "stage_2_plan_prediction",
    "stage_3_predicted_plan_rollout"
  ]
}
```

## 5. S5 이후 phase 구조

현재 기준의 phase 명칭은 다음처럼 정리한다.

| Phase | 이름 | 역할 |
|---|---|---|
| `S4` | Reverse Process Diagnostics | delta/span/anchor/pretrained decoder 구조에서 process signal과 span collapse를 분해한 phase |
| `S5` | Semantic Plan Bridge | semantic chunk 중간 표현이 실제 reverse expansion에 도움이 되는지 확인하는 phase |
| `S6` | Open-ended Generation | S5가 통과한 뒤 open-ended generation으로 확장하는 phase |

이 규칙에 따라 과거에 "S4h"라고 부르던 구조 설명은 구현 phase명이 아니라 S5의 설계 후보로 취급한다.

## 6. 문서화 규칙

새 phase를 만들 때는 다음을 같이 작성한다.

1. `docs/v2/experiment-roadmap.md`의 phase 항목
2. `docs/v2/plan/s{n}-...-plan.md`
3. 실행 후 `docs/v2/experiments/s{n}-....md`
4. `docs/v2/research-timeline.md` 결정/결과 항목
5. 위키 개념 또는 실험 항목
6. 필요하면 `web/index.html` 또는 checkpoint HTML

단, 같은 phase 안의 stage는 별도 코드네임 문서를 만들지 않는다. 결과 문서 안의 stage 표로 관리한다.

## 7. 이름이 복잡해졌을 때의 판단 기준

다음 중 하나라도 해당하면 새 letter를 붙이지 말고 phase 구조를 다시 정리한다.

- 이름만 보고 실험 목적을 설명하기 어렵다.
- 같은 prefix 아래에 4개 이상의 후속 실험이 생겼다.
- "실험의 실험"처럼 보인다.
- 다음 단계가 `S4h-0`처럼 보인다.
- 사용자가 현재 연구 위치를 직관적으로 파악하기 어렵다.

이 규칙의 목적은 단순한 정리가 아니라 연구 해석을 보호하는 것이다. 이름이 복잡해지면 실패 원인과 다음 gate도 함께 흐려지기 때문이다.
