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
| [kaggle-experiment-workflow.md](./kaggle-experiment-workflow.md) | v2 Kaggle 실험 문서화/실행 절차 |
| [document-alignment.md](./document-alignment.md) | v1에서 v2로 문서 기준을 옮긴 결정 기록 |

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

## 다음 실험 방향

v2의 첫 실험은 거대한 end-to-end model이 아니라 **semantic skeleton bridge**로 시작한다.

목표는 다음을 작게 검증하는 것이다.

1. importance-guided skeleton이 random skeleton보다 원문 의미를 더 잘 보존하는가?
2. reverse model이 correct skeleton을 실제로 사용하는가?
3. shuffled/wrong-document skeleton에서 성능이 하락하는가?
4. skeleton-to-text reconstruction이 token/generation metric으로 이어지는가?
