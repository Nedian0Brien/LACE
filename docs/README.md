# LACE Research Documents

이 디렉터리는 LACE 연구 문서를 버전별로 분리해 관리한다.

## 문서 구조

| 위치 | 역할 |
|---|---|
| [v1/](./v1/) | Phase 0부터 Phase 3A까지 진행한 latent token-budget compression 연구 기록 |
| [v2/](./v2/) | importance-guided semantic skeleton 방향으로 재설계한 연구 기획 및 실험 기반 |

## 현재 기준

현재 새 연구 방향의 기준 문서는 [v2/연구기획서.md](./v2/연구기획서.md)다.

v1 문서는 폐기된 것이 아니라, v2로 이동하기 전 확인한 사전 탐색 결과다. 특히 다음 판단은 v2에서도 유지한다.

- hidden-state reconstruction과 open-ended generation은 분리해서 해석해야 한다.
- token-level proxy 개선만으로 generation 성공을 주장하지 않는다.
- Gaussian/noise baseline과 position-only control은 계속 강한 반론으로 남긴다.
- Kaggle 실험은 실행 조건, output, gate, caveat를 함께 문서화한다.

## 새 실험을 시작할 때

v2 실험을 시작할 때는 다음 순서로 문서를 확인한다.

1. [v2/연구기획서.md](./v2/연구기획서.md)
2. [v2/research-questions.md](./v2/research-questions.md)
3. [v2/experiment-roadmap.md](./v2/experiment-roadmap.md)
4. [v2/kaggle-experiment-workflow.md](./v2/kaggle-experiment-workflow.md)
