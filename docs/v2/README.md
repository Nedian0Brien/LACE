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

다음 실험은 `S2: skeleton-to-text reconstruction`이다. 여기서는 의미 골격과 위치 보조 구조를 입력으로 하는 복원 학습을 시작하되, `position_prior`, `position_only`, `same_position_random` control을 계속 유지한다.
