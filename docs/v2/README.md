# V2 Research Foundation

v2는 LACE 연구를 **importance-guided semantic skeleton** 방향으로 재정렬한다.

v2의 핵심 질문은 다음이다.

> Diffusion Language Modeling의 forward process를 random corruption이 아니라 importance-guided semantic compression process로 정의할 수 있는가?

v1이 continuous latent token-budget compression을 먼저 탐색했다면, v2는 중요한 token 또는 semantic unit이 forward process의 terminal compressed state가 될 수 있는지를 직접 검증한다.

## 문서 지도

| 문서 | 역할 |
|---|---|
| [연구기획서.md](./연구기획서.md) | v2의 기본 연구계획서 |
| [research-questions.md](./research-questions.md) | v2에서 검증해야 할 연구 질문과 성공/실패 조건 |
| [experiment-roadmap.md](./experiment-roadmap.md) | v2 실험 phase 구조와 다음 실행 우선순위 |
| [v1-carryover.md](./v1-carryover.md) | v1에서 얻은 성과와 v2로 계승할 실험 규율 |
| [kaggle-experiment-workflow.md](./kaggle-experiment-workflow.md) | v2 Kaggle 실험 문서화/실행 절차 |
| [document-alignment.md](./document-alignment.md) | v1에서 v2로 문서 기준을 옮긴 결정 기록 |
| [research-timeline.md](./research-timeline.md) | v2 연구 질문, 결정, 발견, caveat의 시간순 기록 |

## v2의 중심 주장

v2의 차별점은 다음 문장으로 요약한다.

> Important tokens are not auxiliary anchors; they are the terminal state of the forward diffusion process.

즉, 중요한 token은 reverse denoising을 돕기 위해 별도로 예측되는 보조 신호가 아니라, forward process가 도달해야 하는 compressed state 자체다.

## v1과의 관계

v1 결과는 버리지 않는다. 다만 v1의 결과를 v2의 직접 증거로 과장하지 않는다.

v1에서 확인한 병목은 v2 설계에 다음 제약을 준다.

- hidden reconstruction만 좋아져도 generation이 안 될 수 있다.
- teacher-forced proxy와 open-ended generation은 분리해서 평가해야 한다.
- positional shortcut과 content contribution을 분리해야 한다.
- Gaussian/noise control이 강하면 compression forward의 고유 우위를 주장할 수 없다.

상세한 계승 원칙은 [v1-carryover.md](./v1-carryover.md)에 정리한다.

## 현재 실험 방향

v2의 첫 실험인 S0는 거대한 end-to-end model이 아니라 **의미 골격 bridge**로 시작했고, 결과는 [experiments/s0-skeleton-pipeline.md](./experiments/s0-skeleton-pipeline.md)에 기록했다.

S0에서 확인한 것은 다음이다.

1. IDF/attention 계열 skeleton은 random/uniform보다 의미 보존 신호가 있다.
2. 하지만 `position_prior` baseline이 강하므로 WikiText lead-position confound를 과소평가하면 안 된다.
3. S0는 generation 성공 증거가 아니라 S1/S2 설계를 위한 skeleton pipeline 검증이다.

S1은 [experiments/s1-skeleton-use-controls.md](./experiments/s1-skeleton-use-controls.md)에 기록했다. S1에서 `attention_correct`는 `random_same_count`, `wrong_document`, `position_only`, `same_position_random`보다 강했고, `s2_ready=true`로 통과했다.

S2는 [experiments/s2-skeleton-to-text-reconstruction.md](./experiments/s2-skeleton-to-text-reconstruction.md)에 기록했다. S2에서 `attention_scaffold`는 `random_scaffold`, `position_only`, wrong-document control보다 강했고, `next_ready=true`로 통과했다.

S2a는 [experiments/s2a-positional-encoding.md](./experiments/s2a-positional-encoding.md)에 기록했다. S2a에서 `sinusoidal_absolute`가 가장 좋은 위치 보조 구조 후보로 식별됐지만, `coarse_bins` 대비 개선 폭은 작고 생성 품질은 아직 낮다.

S3는 [experiments/s3-anchor-baseline-comparison.md](./experiments/s3-anchor-baseline-comparison.md)에 기록했다. S3에서 `importance_ordered_forward_no_anchor`는 `random_forward_anchor_prediction`과 tolerance 안에서 비슷했지만, `random_forward_no_anchor`보다 낮아 `overall_pass=false`, `s4_ready=false`였다.

S3a는 [experiments/s3a-terminal-diagnostic.md](./experiments/s3a-terminal-diagnostic.md)에 기록했다. S3a에서 `attention_terminal`은 `random_terminal`과 `same_position_random_terminal`보다 높았지만, `position_only`와 차이가 작고 `random_terminal_predicted_anchor`가 최고 조건이었다.

S3b는 [experiments/s3b-probe-calibration.md](./experiments/s3b-probe-calibration.md)에 기록했다. S3b는 같은 reverse model을 `attention_terminal`로 한 번만 학습한 뒤 평가 입력만 바꿨다. `attention_no_position`은 크게 떨어져 위치 channel의 존재는 중요해 보였지만, `attention_terminal`은 `position_only`, `random_terminal`, `same_position_random_terminal`보다 tolerance 0.02 이상 높지 않았다.

S4는 [experiments/s4-importance-ordered-reverse-diffusion.md](./experiments/s4-importance-ordered-reverse-diffusion.md)에 기록했다. S4는 `25% -> 50% -> 75% -> 100%` reverse transition을 학습해 importance-ordered schedule과 random schedule을 직접 비교했다. 종합 score와 표면 복원 지표에서는 `random_schedule`이 높았지만, `importance_schedule`은 target content recall, input retention, expansion recall, original content recall, entity recall에서 모두 random보다 높았다. 따라서 S5로 가지 않고, 다음은 새로 unmask될 delta token/span을 직접 예측하는 `S4a: delta-token reverse objective`로 둔다.
