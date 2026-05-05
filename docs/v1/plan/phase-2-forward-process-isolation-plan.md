# Phase 2 연구 진행 계획서: Forward Process Isolation

**작성일**: 2026-05-03  
**복잡도**: Medium-High  
**목표 단계**: Phase 2  
**직전 기준점**: Phase 1 WikiText-2 subset 실행 완료  
**핵심 문서**: `docs/Research Questions.md`, `docs/정보압축-forward-process-후보군.md`

## 1. 개요

Phase 2의 목적은 LACE의 핵심 가설을 autoencoder 복원 실험에서 분리해 검증하는 것이다.

Phase 1은 다음을 확인했다.

1. frozen `t5-small` encoder로 만든 hidden representation `h0`를 Kaggle GPU에서 cache할 수 있다.
2. average pooling 기반 compression latent에서 reverse expander가 `h0`를 어느 정도 복원하도록 학습될 수 있다.
3. WikiText-2 subset에서는 pooling도 latent sensitivity가 약했고, learned attention compression은 현재 구조로는 안정적이지 않았다.

따라서 Phase 2에서는 learned attention compression을 핵심 대표로 삼지 않는다. 대신 더 단순하고 안정적인 fixed compression forward process를 사용해 다음 질문을 검증한다.

> 같은 데이터, 같은 encoder latent, 같은 reverse model capacity에서 정보 압축 forward process는 corruption/noise forward process보다 더 좋은 reverse learning 경로를 제공하는가?

이 단계의 성공은 "완성된 Diffusion Language Model을 만들었다"가 아니다. Phase 2의 성공은 **forward process를 random corruption이 아니라 information compression으로 정의하는 것이 reverse expansion과 generation bridge에 유리하다는 1차 증거**를 얻는 것이다.

## 2. 핵심 연구 질문

### 2.1 Primary Question

```text
Does compression-based forward process outperform corruption-based forward process
under the same reverse expansion setting?
```

한국어로는 다음이다.

> Diffusion Language Modeling의 forward process를 무작위 손상 과정이 아니라 정보 압축 과정으로 설계하면, reverse process 학습이 더 안정적이고 생성으로 이어지기 쉬운가?

### 2.2 Sub Questions

1. Average pooling compression은 random drop/mask보다 낮은 validation reconstruction loss를 보이는가?
2. Strided token selection은 average pooling과 비교해 얼마나 손실이 큰가?
3. Gaussian noise baseline은 같은 학습 예산에서 compression condition보다 reverse trajectory가 불안정한가?
4. Compression forward는 stage가 깊어질수록 예측 가능한 난도 곡선을 만드는가?
5. Recovered hidden state는 frozen decoder 또는 lightweight token head를 통해 token reconstruction/generation proxy로 이어지는가?
6. Phase 2 결과가 downsampling autoencoder가 아니라 DLM forward process 재설계 주장으로 이어질 수 있는가?

## 3. Phase 2에서 주장하지 않을 것

Phase 2에서는 다음을 주장하지 않는다.

- LACE가 완전한 language generation model로 완성됐다는 주장
- learned attention compression이 우수하다는 주장
- human-level generation quality가 개선됐다는 주장
- exact mutual information을 측정했다는 주장
- 모든 DLM baseline보다 우수하다는 주장

Phase 2의 주장은 더 좁다.

> 동일 조건의 reverse expansion 실험에서 compression forward process가 corruption/noise forward process보다 더 좋은 학습 신호와 generation bridge를 제공하는지 본다.

## 4. 비교 조건

Phase 2의 최소 비교 조건은 네 가지다.

| ID | 조건 | 유형 | 입력 shape | 역할 |
|---|---|---|---|---|
| F1 | `average_pool` | fixed compression | `64/32/16 × d` | primary compression condition |
| F2 | `strided_select` | fixed compression | `64/32/16 × d` | 단순 token selection compression |
| F3 | `random_select` | stochastic corruption/drop | `64/32/16 × d` | 같은 token budget을 가진 corruption baseline |
| F4 | `gaussian_noise` | continuous corruption/noise | `128 × d` | 기존 continuous diffusion류 reference |

### 4.1 Average Pooling

```text
h0: 128 × d
z1: mean over every 2 tokens -> 64 × d
z2: mean over every 4 tokens -> 32 × d
z3: mean over every 8 tokens -> 16 × d
```

이 조건은 Phase 2의 핵심 compression condition이다. 단순하지만 정보율을 token budget으로 명확히 통제할 수 있다.

### 4.2 Strided Selection

```text
z1 = h0[::2]
z2 = h0[::4]
z3 = h0[::8]
```

평균을 내지 않고 일부 latent token만 보존한다. average pooling이 "섞는 압축"이라면 strided selection은 "선택하는 압축"이다.

### 4.3 Random Selection

각 sample과 stage에서 target token budget만큼 latent token을 무작위로 선택한다.

```text
z1: random 64 tokens from h0
z2: random 32 tokens from h0
z3: random 16 tokens from h0
```

token budget은 compression condition과 같지만 선택이 무작위라는 점에서 corruption/drop baseline에 가깝다. 반드시 seed를 고정해 재현 가능하게 만든다.

### 4.4 Gaussian Noise

Gaussian noise는 token 수를 줄이지 않고 `h0`에 noise를 더한다.

```text
z_t = h0 + sigma_t * epsilon
epsilon ~ N(0, I)
```

기본 sigma 후보:

```text
t1: 0.10
t2: 0.20
t3: 0.40
```

주의할 점은 Gaussian noise는 compression 조건과 shape가 다르다는 것이다. 따라서 raw loss만으로 단순 우열을 말하면 안 된다. Phase 2에서는 다음 두 방식으로 보정한다.

1. 같은 reverse expander family를 사용하되 variable input length를 허용한다.
2. raw metric과 함께 difficulty-matched metric을 기록한다. 예를 들어 average pooling `z2`와 비슷한 initial reconstruction loss를 만드는 sigma를 별도로 찾는다.

## 5. 통제 조건

공정한 비교를 위해 다음 조건을 고정한다.

| 항목 | Phase 2 기본값 |
|---|---|
| 실행 환경 | Kaggle T4 또는 동급 단일 GPU |
| base model | `t5-small` |
| encoder | frozen |
| decoder | frozen, generation bridge에서만 사용 |
| 데이터 | WikiText-2 subset |
| smoke samples | 512 |
| main samples | 2k 우선, 가능하면 5k |
| max length | 128 token |
| hidden size | 512 |
| compression stages | `64, 32, 16` |
| reverse model | 동일 expander family |
| optimizer | AdamW |
| precision | fp16 가능하면 사용 |
| seed | 42 기본, 여유가 있으면 3 seeds |

## 6. 모델 및 학습 설계

### 6.1 입력 representation

```text
text
-> tokenizer
-> frozen T5 encoder
-> h0 ∈ R^{B × 128 × 512}
```

`h0`는 모든 forward condition의 공통 출발점이다. Phase 2에서는 encoder를 학습하지 않는다.

### 6.2 Forward Process Interface

모든 forward process는 같은 interface를 따른다.

```text
forward_process(h0, attention_mask, stage, seed) -> z_t, z_mask, metadata
```

필수 metadata:

- `condition_name`
- `stage`
- `input_tokens`
- `output_tokens`
- `compression_ratio` 또는 `noise_sigma`
- `seed`

### 6.3 Reverse Expander

Reverse expander는 variable-length latent를 `h0` shape로 복원한다.

```text
z_t ∈ R^{B × N_t × d}
-> length adapter
-> refinement MLP 또는 shallow Transformer
-> h0_hat ∈ R^{B × 128 × d}
```

처음 구현은 Phase 1의 expander를 재사용한다.

권장 기본 구조:

1. positional interpolation으로 `N_t -> 128` 확장
2. MLP refinement
3. LayerNorm
4. residual projection

GPU 여유가 있으면 shallow Transformer block을 추가한다.

### 6.4 Loss

기본 loss:

```text
L = L_rec + lambda_cos * L_cos + lambda_var * L_var + lambda_token * L_token
```

항목별 정의:

| Loss | 정의 | Phase 2 역할 |
|---|---|---|
| `L_rec` | `MSE(h0_hat, h0)` | hidden reconstruction |
| `L_cos` | active token 기준 cosine distance | 방향성 보존 |
| `L_var` | variance collapse penalty | trivial collapse 방지 |
| `L_token` | frozen decoder 또는 token head의 teacher-forced loss | generation bridge |

처음에는 `L_token`을 training loss에 바로 넣지 않고 evaluation-only metric으로 둔다. `L_token`을 loss에 포함하면 "forward process 비교"와 "decoder 적응"이 섞일 수 있기 때문이다.

추천 시작값:

```text
lambda_cos = 0.1
lambda_var = 0.01
lambda_token = 0.0
```

## 7. Generation Bridge 설계

RQ6 때문에 Phase 2에는 language generation으로 이어지는 최소 bridge가 필요하다. 다만 이 단계에서 full DLM을 만들지는 않는다.

### 7.1 Frozen T5 Decoder Evaluation

복원된 hidden state `h0_hat`를 frozen T5 decoder의 encoder output으로 넣고, 원문 token을 teacher-forcing target으로 사용한다.

비교 기준:

```text
NLL(h0_hat -> text) - NLL(h0 -> text)
```

이 값을 `delta_token_nll_vs_h0`로 기록한다. 절대 NLL보다 이 delta가 중요하다. T5가 순수 autoencoding decoder로 학습된 모델이 아니기 때문에, `h0` 원본 기준 대비 얼마나 나빠지는지를 봐야 한다.

### 7.2 Lightweight Token Head Evaluation

Frozen decoder 연결이 불안정하거나 구현 비용이 크면, 대체로 lightweight token head를 사용한다.

```text
h0_hat
-> linear vocab head
-> input token reconstruction
```

이 경우 token head는 모든 condition에서 공유하거나, condition별로 같은 capacity와 같은 step 수로 학습한다. 단, 이 결과는 generation proxy일 뿐 실제 generation 성능으로 과장하지 않는다.

### 7.3 Qualitative Sample

각 condition에서 같은 8개 sample을 골라 다음을 저장한다.

- original text
- condition
- stage
- generated or decoded text
- token loss
- repetition ratio
- 간단한 관찰 메모

이 sample은 논문 주장의 근거가 아니라 오류 양상을 보는 sanity check다.

## 8. 평가 지표

### 8.1 Hidden Reconstruction Metrics

| 지표 | 의미 | 좋은 방향 |
|---|---|---|
| validation MSE | `h0_hat`와 `h0`의 평균 오차 | 낮을수록 좋음 |
| validation cosine | active token 방향 유사도 | 높을수록 좋음 |
| stage monotonicity | `t1 -> t2 -> t3` 난도 증가 | compression에서 일관적이어야 함 |
| variance ratio | 복원 latent 분산 비율 | 너무 낮거나 높으면 위험 |

### 8.2 Latent Use Metrics

| 지표 | 의미 | 좋은 방향 |
|---|---|---|
| relative perturbation sensitivity | latent를 흔들 때 출력이 변하는 정도 | 0에 가까우면 위험 |
| latent ablation delta | latent 일부 제거 시 성능 하락 | 하락이 있어야 latent 사용 |
| latent swap delta | batch 내 latent 교체 시 성능 하락 | 하락이 있어야 sample-specific 사용 |

Phase 1의 perturbation sensitivity는 절대값 기준이라 데이터가 바뀌면 해석이 어려웠다. Phase 2에서는 반드시 상대 지표를 함께 기록한다.

```text
relative_sensitivity =
  MSE(output(z + perturb), output(z)) / max(MSE(output(z), target), eps)
```

### 8.3 Generation Bridge Metrics

| 지표 | 의미 | 좋은 방향 |
|---|---|---|
| token NLL | teacher-forced token loss | 낮을수록 좋음 |
| delta token NLL vs h0 | 원본 `h0` 대비 복원 `h0_hat`의 손실 증가 | 낮을수록 좋음 |
| token accuracy | 단순 token reconstruction 정확도 | 높을수록 좋음 |
| repetition ratio | 생성문 반복 정도 | 낮을수록 좋음 |
| sample validity note | 사람이 읽는 qualitative check | collapse 없어야 함 |

## 9. 성공 Gate

Phase 2는 다음 gate로 판단한다.

| Gate | 이름 | 통과 기준 |
|---|---|---|
| P2-G1 | 실행 완결성 | 네 forward condition이 모두 같은 dataset split에서 실행됨 |
| P2-G2 | Compression 우위 신호 | `average_pool`이 `random_select`보다 validation MSE 또는 cosine에서 우수 |
| P2-G3 | Noise 대비 안정성 | `average_pool`이 difficulty-matched Gaussian noise보다 reverse loss 또는 token NLL에서 우수 |
| P2-G4 | Stage 곡선 | compression stage가 깊어질수록 난도가 예측 가능하게 증가 |
| P2-G5 | Latent 사용 | relative perturbation, ablation, swap 중 최소 2개에서 latent 사용 신호 확인 |
| P2-G6 | Generation bridge | `average_pool` 또는 `strided_select`가 corruption baseline보다 낮은 `delta_token_nll_vs_h0`를 보임 |

### 9.1 강한 성공

다음이 모두 성립하면 강한 성공으로 본다.

- `average_pool`이 `random_select`와 `gaussian_noise`보다 reconstruction metric에서 우수하다.
- `average_pool`이 generation bridge metric에서도 우수하다.
- stage별 난도 곡선이 단조롭다.
- latent perturbation/ablation/swap test에서 expander가 latent를 실제로 사용한다.

### 9.2 약한 성공

다음이면 약한 성공으로 본다.

- reconstruction에서는 compression이 우수하지만 generation bridge에서는 차이가 작다.
- `average_pool`은 우수하지만 `strided_select`는 약하다.
- Gaussian noise와의 비교는 애매하지만 random_select보다 compression이 분명히 낫다.

약한 성공이면 Phase 2B 또는 Phase 3에서 generation bridge를 강화한다.

### 9.3 실패

다음이면 Phase 2 실패로 본다.

- compression condition이 random_select보다 낫지 않다.
- Gaussian noise가 compression보다 모든 metric에서 낫다.
- latent use test가 계속 0에 가깝다.
- hidden reconstruction은 좋지만 token reconstruction/generation bridge가 전혀 이어지지 않는다.

## 10. Sprint 계획

## Sprint 1. Phase 2 runner scaffold

**목표**: Phase 1 runner를 기반으로 Phase 2 실험 구조를 만든다.

**작업**:

- `kaggle/phase2/run_phase2.py` 생성
- `kaggle/phase2/kernel-metadata.json` 생성
- `scripts/push_kaggle_phase2.sh` 생성
- `tests/test_phase2_runner.py` 생성
- Phase 1의 text loading, HF dataset fallback, config parsing 재사용

**검증**:

- `python3 -m py_compile kaggle/phase2/run_phase2.py`
- `python3 -m unittest tests.test_phase2_runner`
- `bash -n scripts/push_kaggle_phase2.sh`

**완료 조건**:

- 로컬에서 torch 없이도 pure function test가 통과한다.
- Kaggle script가 단일 파일로 실행 가능하다.

## Sprint 2. Forward process interface 구현

**목표**: 네 가지 forward condition을 같은 interface로 실행한다.

**작업**:

- `average_pool_forward`
- `strided_select_forward`
- `random_select_forward`
- `gaussian_noise_forward`
- stage token parsing: `64,32,16`
- seed 고정 및 metadata 기록

**검증**:

- 모든 condition이 예상 shape를 반환한다.
- random_select는 같은 seed에서 같은 token index를 반환한다.
- Gaussian noise는 sigma별 variance 증가가 기록된다.

**완료 조건**:

- `metrics.json`에 condition별 input/output token 수와 forward metadata가 저장된다.

## Sprint 3. Reverse expander 공정 비교

**목표**: 같은 reverse expander family로 condition별 `h0_hat`를 학습한다.

**작업**:

- variable-length latent adapter 구현
- Phase 1 expander 재사용 또는 간소화
- condition별 동일 epoch, batch size, optimizer 적용
- train/validation split 고정
- condition별 checkpoint 저장

**검증**:

- 네 condition 모두 train loss가 finite이다.
- validation loss가 저장된다.
- condition별 parameter count가 같거나 비교 가능한 범위임을 기록한다.

**완료 조건**:

- `summary.md`에 condition별 train/validation loss table이 생성된다.

## Sprint 4. Corruption baseline calibration

**목표**: Gaussian noise baseline을 raw 비교와 difficulty-matched 비교로 나눈다.

**작업**:

- sigma 후보: `0.05, 0.10, 0.20, 0.40, 0.80`
- short calibration run으로 initial reconstruction loss 측정
- average_pool `z1/z2/z3`와 비슷한 난도의 sigma 선택
- raw sigma schedule과 matched sigma schedule을 모두 기록

**검증**:

- calibration table이 `metrics.json`에 저장된다.
- matched sigma 선택 기준이 `summary.md`에 명시된다.

**완료 조건**:

- Gaussian noise 결과를 "shape가 다르지만 참고 baseline"으로만 해석하지 않고, 난도 보정 기준까지 포함한다.

## Sprint 5. Latent use test 강화

**목표**: Phase 1에서 약했던 latent sensitivity 해석을 개선한다.

**작업**:

- relative perturbation sensitivity 추가
- latent ablation test 추가
- latent swap test 추가
- condition별 sensitivity metric table 생성

**검증**:

- perturbation, ablation, swap 모두 finite value를 반환한다.
- random_select와 average_pool에서 latent use 차이가 측정된다.

**완료 조건**:

- P2-G5 판정을 할 수 있는 지표가 확보된다.

## Sprint 6. Generation bridge 추가

**목표**: RQ6을 위해 hidden reconstruction과 language generation 사이의 최소 연결을 만든다.

**작업**:

- frozen T5 decoder teacher-forced token NLL 계산
- `NLL(h0_hat) - NLL(h0)` delta 기록
- 실패 시 lightweight token head fallback 구현
- 8개 qualitative sample 저장

**검증**:

- token NLL이 condition별로 기록된다.
- `h0` oracle 대비 delta가 계산된다.
- sample text가 `summary.md` 또는 별도 json에 저장된다.

**완료 조건**:

- Phase 2 결과가 latent reconstruction에서 멈추지 않고 generation proxy로 이어진다.

## Sprint 7. Kaggle 실행 및 결과 정리

**목표**: Phase 2를 Kaggle에서 실행하고 연구 결과 문서를 남긴다.

**작업**:

- Kaggle kernel push
- smoke run 512 samples
- main run 2k samples
- 가능하면 3 seeds 중 최소 2 seeds 실행
- 결과 다운로드
- `docs/experiments/phase-2-forward-process-isolation.md` 작성

**검증**:

- `metrics.json`, `summary.md`, `train_log.jsonl` 생성
- 문서 숫자와 metrics 파일 숫자 일치
- P2-G1~P2-G6 pass/fail 명시

**완료 조건**:

- Phase 2 결과만 보고도 compression forward hypothesis를 다음 단계로 가져갈지 판단할 수 있다.

## 11. 산출물

| 산출물 | 위치 |
|---|---|
| Phase 2 runner | `kaggle/phase2/run_phase2.py` |
| Kaggle metadata | `kaggle/phase2/kernel-metadata.json` |
| push script | `scripts/push_kaggle_phase2.sh` |
| local tests | `tests/test_phase2_runner.py` |
| 계획서 | `docs/plan/phase-2-forward-process-isolation-plan.md` |
| 결과 문서 | `docs/experiments/phase-2-forward-process-isolation.md` |
| metrics | `outputs/phase2/lace_phase2/metrics.json` |
| summary | `outputs/phase2/lace_phase2/summary.md` |
| train log | `outputs/phase2/lace_phase2/train_log.jsonl` |
| samples | `outputs/phase2/lace_phase2/generation_samples.jsonl` |

`outputs/phase2`는 git에 포함하지 않는다.

## 12. 실행 명령 초안

로컬 문법 검사:

```bash
python3 -m py_compile kaggle/phase2/run_phase2.py tests/test_phase2_runner.py
```

로컬 unit test:

```bash
python3 -m unittest tests.test_phase2_runner
```

Kaggle 업로드:

```bash
bash scripts/push_kaggle_phase2.sh
```

직접 push:

```bash
kaggle kernels push -p kaggle/phase2 --accelerator NvidiaTeslaT4 --timeout 3600
```

상태 확인:

```bash
kaggle kernels status dennisparknd/lace-phase-2-forward-process-isolation
```

결과 다운로드:

```bash
mkdir -p outputs/phase2
kaggle kernels output dennisparknd/lace-phase-2-forward-process-isolation -p outputs/phase2
```

## 13. 위험 요소와 대응

| 위험 | 징후 | 대응 |
|---|---|---|
| Gaussian noise와 compression의 shape가 다름 | raw loss 비교가 불공정 | difficulty-matched sigma를 별도 기록 |
| token NLL이 T5 autoencoding과 맞지 않음 | `h0` oracle도 NLL이 높음 | delta NLL vs `h0` 중심으로 해석 |
| generation bridge 구현이 복잡함 | T5 decoder 호출 오류 | lightweight token head fallback 사용 |
| random_select가 seed마다 크게 흔들림 | 결과 variance가 큼 | seed 고정, 가능하면 3 seeds |
| average_pool이 너무 강함 | z3에서도 쉽게 복원 | token 수를 8까지 줄이는 ablation |
| latent use가 계속 약함 | perturbation/swap delta가 작음 | expander capacity 제한, dropout, loss 재조정 |
| attention 개선으로 scope creep | Phase 2가 Phase 1.5로 변질 | attention은 Phase 2 기본 조건에서 제외 |
| Kaggle OOM | CUDA memory error | batch size 축소, samples 축소, fp16 사용 |

## 14. 해석 원칙

Phase 2 결과를 해석할 때 다음 원칙을 지킨다.

1. Reconstruction metric만으로 LACE 성공을 주장하지 않는다.
2. Generation bridge metric이 없으면 RQ6에 답했다고 말하지 않는다.
3. Gaussian noise와 compression은 shape가 다르므로 raw metric만 비교하지 않는다.
4. Average pooling이 이기더라도 "압축 연산이 새롭다"고 주장하지 않는다.
5. 주장은 항상 "forward process를 corruption에서 compression으로 바꾸는 것이 유리한가"로 제한한다.

## 15. Phase 3로 진행하는 기준

다음 조건을 만족하면 Phase 3로 넘어간다.

- `average_pool` 또는 `strided_select`가 corruption baseline보다 reverse reconstruction에서 우수하다.
- generation bridge에서도 compression condition이 최소한 손해를 보지 않거나 일부 이점을 보인다.
- latent use test에서 expander가 input latent를 실제로 사용한다.
- 결과가 WikiText-2 subset에서 재현 가능하다.

Phase 3의 후보는 다음 중 하나다.

1. decoder-conditioned generation을 더 본격화한다.
2. small diffusion decoder를 붙여 generation-capable DLM prototype을 만든다.
3. learned compression을 다시 도입하되 pooling anchor와 scale regularization을 사용한다.

## 16. Rollback 계획

Phase 2 구현이 실패해도 Phase 1 결과와 문서는 유지한다.

- `kaggle/phase2/`만 제거하면 Phase 1에는 영향 없음
- `scripts/push_kaggle_phase2.sh`는 Phase 1 script와 분리
- `outputs/phase2/`는 git ignore 대상
- 실패 결과도 `docs/experiments/phase-2-forward-process-isolation.md`에 남긴다.

## 17. 완료 정의

Phase 2는 다음 조건을 만족하면 완료로 본다.

1. Kaggle에서 Phase 2 kernel이 끝까지 실행된다.
2. 네 forward condition이 같은 데이터 split에서 비교된다.
3. hidden reconstruction metric이 condition별로 기록된다.
4. latent use metric이 condition별로 기록된다.
5. token reconstruction 또는 generation bridge metric이 condition별로 기록된다.
6. `docs/experiments/phase-2-forward-process-isolation.md`에 P2-G1~P2-G6 판정이 기록된다.

이 완료 조건을 만족하면 LACE는 "압축 autoencoder"가 아니라 "DLM forward process를 정보 압축으로 바꾸는 연구"로 다음 단계의 주장을 세울 수 있다.
