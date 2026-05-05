# Document Alignment Notes

이 문서는 v1 문서 archive와 v2 문서 foundation을 나눈 기준을 기록한다.

## 결정

기존 연구 문서는 [../v1/](../v1/)로 이동했다. 새 연구 방향은 [./](./) 아래에서 관리한다.

## 이유

v1과 v2는 같은 문제의식에서 출발하지만, 실험 단위가 다르다.

| 항목 | v1 | v2 |
|---|---|---|
| 중심 표현 | frozen encoder latent `h0` | token/semantic skeleton `x_T` |
| forward process | token-budget latent compression | importance-guided semantic compression |
| terminal state | low-token latent `z_t` | 핵심 token skeleton |
| reverse target | `z_t -> h0_hat` | `x_T -> x_{t-1}` 또는 `x_0` |
| 주요 baseline | random select, matched Gaussian | random masking, uniform skeleton, anchor prediction |
| 주요 병목 | decoder-readable hidden state | semantic skeleton validity and use |

## v1을 보존하는 방식

v1 문서는 실패 기록이 아니라 v2의 설계 제약이다.

v1에서 v2로 가져갈 교훈:

1. proxy metric과 generation metric을 분리한다.
2. control 없이 compression superiority를 주장하지 않는다.
3. positional shortcut을 항상 의심한다.
4. Gaussian/noise baseline은 강한 반론으로 유지한다.
5. open-ended generation은 마지막 단계로 둔다.

## v2에서 새로 필요한 문서

v2에는 다음 문서가 우선 필요하다.

| 문서 | 상태 |
|---|---|
| `연구기획서.md` | 작성됨 |
| `research-questions.md` | 작성됨 |
| `experiment-roadmap.md` | 작성됨 |
| `kaggle-experiment-workflow.md` | 작성됨 |
| `plan/s0-skeleton-pipeline-plan.md` | 다음 작업 |
| `experiments/s0-skeleton-pipeline.md` | S0 실행 후 작성 |
