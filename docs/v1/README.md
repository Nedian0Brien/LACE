# V1 Research Archive

이 폴더는 LACE v1 연구 기록을 보존한다.

v1의 중심 질문은 다음이었다.

> frozen encoder latent `h0`를 token-budget compression으로 줄이고, reverse expander가 이를 다시 `h0` 또는 더 상세한 latent로 복원할 수 있는가?

v1은 semantic skeleton 연구의 최종 형태는 아니지만, v2로 넘어가기 전에 중요한 실패와 병목을 확인했다.

## 주요 문서

| 문서 | 역할 |
|---|---|
| [연구기획서.md](./연구기획서.md) | latent information compression 중심의 기존 연구 기획 |
| [Research Questions.md](./Research%20Questions.md) | v1 노벨티 경계와 autoencoder 오해 방지 기준 |
| [실험설계.md](./실험설계.md) | 개인 연구자용 scale-down latent 실험 설계 |
| [정보압축-forward-process-후보군.md](./정보압축-forward-process-후보군.md) | average pooling, strided selection, learned attention 등 후보군 |
| [연구노트.md](./연구노트.md) | Phase 0-3A 결과와 해석을 모은 연구노트 |
| [experiments/](./experiments/) | 실제 Kaggle 실험 결과 문서 |
| [plan/](./plan/) | Phase별 실험 계획서 |

## v1에서 확인한 것

1. frozen `t5-small` encoder latent cache와 Kaggle 실행 pipeline은 안정적으로 동작한다.
2. token-budget compression stage는 자연스러운 정보 손실 곡선을 만든다.
3. Average Pooling은 learned attention compression보다 안정적인 초기 baseline이었다.
4. Average Pooling은 random selection보다 representation direction과 일부 decoder bridge에서 우호적인 신호를 보였다.
5. positional scaffold는 reverse expansion proxy를 강하게 개선하지만, position-only와 Gaussian positional control도 강했다.
6. lightweight token-head objective는 token-head NLL을 크게 낮췄지만 frozen decoder NLL과 open-ended generation을 악화시켰다.

## v1의 한계

v1은 v2의 핵심 주장인 "중요 token은 auxiliary anchor가 아니라 forward process의 terminal state"를 직접 검증하지 않았다.

따라서 v1 결과는 다음처럼 해석한다.

> v1은 continuous latent compression track의 사전 탐색이다. 이 결과는 LACE가 단순 hidden reconstruction으로는 generation path를 확보하기 어렵다는 점을 보여줬고, v2의 semantic skeleton 방향으로 연구 질문을 더 날카롭게 만드는 근거가 된다.

## 실행 코드 위치

기존 Kaggle runner와 push scripts는 아직 top-level에 둔다.

| 위치 | 내용 |
|---|---|
| `kaggle/phase0` - `kaggle/phase3a` | v1 실험 runner |
| `scripts/push_kaggle_phase*.sh` | v1 Kaggle push scripts |

코드를 물리적으로 옮기지 않은 이유는 기존 runner, script, test의 상대 경로를 깨지 않기 위해서다.

과거 v1 runner 테스트 파일은 문서 작업 속도를 위해 제거했다. v1 runner를 다시 수정할 때만 해당 runner의 syntax check나 작은 smoke run을 수행한다.
