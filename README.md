# LACE

**LACE: Latent Adaptive Compression and Expansion for Language Diffusion**

이 저장소는 논문 주제인 *Compression, Not Corruption*을 연구하기 위한 실험 및 문서 스캐폴드다. 핵심 문제의식은 언어 확산 모델의 forward process를 random corruption이 아니라 semantic information-rate schedule로 재정의할 수 있는지 검증하는 것이다.

## 현재 문서 기준

문서는 v1과 v2로 분리한다.

| 위치 | 설명 |
|---|---|
| [docs/v1/](./docs/v1/) | Phase 0-3A까지 진행한 latent token-budget compression 연구 기록 |
| [docs/v2/](./docs/v2/) | importance-guided semantic skeleton 방향으로 재설계한 현재 연구 기반 |

현재 새 연구 방향의 기준은 [docs/v2/연구기획서.md](./docs/v2/연구기획서.md)다.

## v1 요약

v1은 frozen `t5-small` encoder latent `h0`를 단계적으로 압축하고, reverse expander가 이를 복원할 수 있는지 확인했다.

주요 결과:

- Kaggle GPU 기반 latent cache와 실험 pipeline이 동작했다.
- Average Pooling은 learned attention compression보다 안정적인 baseline이었다.
- Average Pooling은 일부 decoder bridge와 latent-use 신호를 보였지만, open-ended generation은 회복하지 못했다.
- token-head objective는 proxy NLL을 낮췄지만 frozen decoder NLL과 generation을 악화시켰다.

따라서 v1은 최종 성공 증거가 아니라, v2로 넘어가기 위한 병목과 confound를 드러낸 사전 탐색으로 보존한다.

## v2 방향

v2는 다음 문장을 중심 주장으로 둔다.

> Important tokens are not auxiliary anchors; they are the terminal state of the forward diffusion process.

즉, 중요한 token은 reverse denoising을 돕는 보조 anchor가 아니라, forward process가 도달해야 하는 semantic skeleton terminal state다.

`S0: semantic skeleton extraction and preservation validation`, `S1: skeleton-use controls`, `S2: skeleton-to-text reconstruction`, `S2a: positional encoding comparison`은 통과했다. `S3: anchor baseline comparison`은 실행됐지만 `importance_ordered_forward_no_anchor`가 `random_forward_no_anchor`를 이기지 못해 핵심 gate를 통과하지 못했다. 다음 실험은 S4가 아니라 `S3a: terminal diagnostic`이다.

## 실험 실행

Kaggle-backed 실험을 진행할 때는 먼저 [docs/v2/kaggle-experiment-workflow.md](./docs/v2/kaggle-experiment-workflow.md)를 확인한다.

가벼운 로컬 검증 기본 명령:

```bash
git diff --check
```

v1 runner와 push scripts는 기존 재현성을 위해 top-level `kaggle/`, `scripts/`에 유지한다. 과거 runner 테스트는 문서 작업 속도를 위해 제거했고, 필요하면 해당 phase runner에 대해 syntax check나 작은 smoke run을 직접 수행한다.
