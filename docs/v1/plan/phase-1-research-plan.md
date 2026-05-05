# Phase 1 연구 진행 계획서: Latent Compression Proof

**작성일**: 2026-05-03
**복잡도**: Medium
**목표 단계**: Phase 1
**직전 기준점**: Phase 0 Kaggle 실행 완료

## 1. 개요

Phase 1의 목적은 LACE의 핵심 모듈인 compression path와 reverse expander가 실제로 학습 가능한지 확인하는 것이다. Phase 0에서는 고정 `t5-small` encoder로 `h0` latent를 만들고, average pooling으로 `z1/z2/z3`를 만든 뒤, 압축이 강해질수록 MSE가 증가하고 cosine이 감소하는 기본 곡선을 확인했다.

Phase 1에서는 여기서 한 단계 더 나아가 다음 질문을 검증한다.

> 압축된 latent `z_t`로부터 더 상세한 latent, 최종적으로 `h0`를 안정적으로 복원하도록 expander를 학습할 수 있는가?

이 단계의 성공은 "LACE가 baseline보다 좋다"가 아니라, **학습 가능한 latent compression/expansion loop가 성립한다**는 것을 보이는 데 있다.

## 2. Phase 0 출발점

Phase 0 결과는 다음과 같다.

| 항목 | 값 |
|---|---|
| 실행 환경 | Kaggle GPU |
| 모델 | 고정 `t5-small` encoder |
| 샘플 수 | 128 |
| 최대 길이 | 128 token |
| `h0` shape | `[128, 128, 512]` |
| cache 재로딩 | `true` |

단계별 pooling sanity metric:

| 단계 | token 수 | MSE | Cosine |
|---|---:|---:|---:|
| `z1` | 64 | 0.008886 | 0.787159 |
| `z2` | 32 | 0.013866 | 0.642822 |
| `z3` | 16 | 0.017505 | 0.522215 |

이 결과는 token budget 감소가 latent reconstruction 난도를 높인다는 기본 방향성을 보여준다. Phase 1은 이 곡선을 학습 가능한 expander로 개선하고, 병목이 너무 약하거나 너무 강하지 않은지 확인한다.

## 3. 연구 질문

### 3.1 핵심 질문

1. `z1/z2/z3`에서 `h0`를 복원하는 expander의 validation loss가 안정적으로 감소하는가?
2. 압축 단계가 강해질수록 복원 난도가 일관되게 증가하는가?
3. `z3`가 너무 쉽게 `h0`를 복원하지 않으면서도 완전히 붕괴하지 않는가?
4. learned attention compression이 average pooling baseline보다 더 나은 latent 복원 곡선을 만드는가?

### 3.2 Phase 1에서 주장하지 않을 것

- LACE가 Gaussian/mask baseline보다 우수하다는 주장
- 텍스트 생성 품질에 대한 주장
- exact mutual information을 측정했다는 주장
- 의미 보존에서 최종 우위가 있다는 주장

이 주장들은 Phase 2 이후의 비교 실험에서 다룬다.

## 4. 실험 범위

### 4.1 포함 범위

- WikiText-2 또는 XSum subset 기반 latent cache 생성
- 고정 `t5-small` encoder 유지
- `h0 → z1 → z2 → z3` compression path 구성
- `z3 → z2 → z1 → h0` reverse expander 학습
- average pooling baseline 학습
- learned-query attention compression 학습
- stage별 MSE/cosine/variance/perturbation 지표 기록
- Kaggle에서 재현 가능한 Phase 1 kernel 구성

### 4.2 제외 범위

- renderer 또는 text decoder 연결
- BERTScore/NLI/QA consistency 평가
- MINE 기반 mutual information 추정
- LLaDA-small 재학습
- multi-GPU 학습
- 긴 문맥 generation 실험

## 5. 기본 실험 설정

| 항목 | Phase 1 기본값 |
|---|---|
| 실행 환경 | Kaggle T4 또는 동급 단일 GPU |
| encoder | `t5-small` frozen |
| hidden size | 512 |
| max length | 128 |
| stage tokens | `64, 32, 16` |
| 데이터 | WikiText-2 우선, 실패 시 fallback corpus 확대 |
| 샘플 수 | smoke 256, 본 실험 5k~10k |
| batch size | 8~32 |
| epoch | 3~10 |
| optimizer | AdamW |
| precision | fp16 가능하면 사용 |
| seed | 42 우선, 여유가 있으면 3 seeds |

## 6. 모델 설계

### 6.1 입력 latent

```text
text
→ frozen T5 encoder
→ h0 ∈ R^{B × 128 × 512}
```

`h0`는 Phase 1에서도 학습하지 않는다. 먼저 encoder를 고정해야 compression/expansion 모듈의 효과를 분리할 수 있다.

### 6.2 Compression path

Phase 1에서는 두 가지 compression을 비교한다.

| 이름 | 방식 | 목적 |
|---|---|---|
| C0 | average pooling | 하한선 baseline |
| C1 | learned-query attention pooling | LACE-small의 첫 compression path |

기본 stage:

```text
h0: 128 × 512
z1: 64 × 512
z2: 32 × 512
z3: 16 × 512
```

### 6.3 Reverse expander

첫 구현은 복잡한 diffusion block이 아니라, 안정적인 latent reconstruction module로 둔다.

```text
z3 → expand_3 → z2_hat
z2 → expand_2 → z1_hat
z1 → expand_1 → h0_hat
```

구현 후보:

| 버전 | 구조 | 판단 |
|---|---|---|
| E0 | linear interpolation + MLP refinement | 가장 빠른 baseline |
| E1 | interpolation + shallow Transformer block | Phase 1 기본값 |
| E2 | cross-attention query expander | E1이 안정화된 뒤 |

처음에는 E1을 기본값으로 잡되, GPU memory 문제가 있으면 E0로 낮춘다.

## 7. Loss 설계

기본 loss:

```text
L = L_stage + λ_cos L_cos + λ_var L_var
```

세부 항목:

| 항목 | 정의 | 목적 |
|---|---|---|
| `L_stage` | 각 단계 target latent와 예측 latent의 MSE | 복원 가능성 확인 |
| `L_cos` | active token 기준 cosine distance | 방향성 보존 |
| `L_var` | latent variance collapse penalty | trivial collapse 방지 |

처음에는 `λ_cos = 0.1`, `λ_var = 0.01` 수준으로 시작하고, 불안정하면 MSE-only로 되돌린다.

## 8. 평가 지표

### 8.1 필수 지표

| 지표 | 기준 |
|---|---|
| train loss | epoch별 하향 추세 |
| validation loss | train loss와 함께 하향, 큰 divergence 없음 |
| stage MSE | `z1 < z2 < z3` 난도 곡선 유지 |
| stage cosine | `z1 > z2 > z3` 방향성 곡선 유지 |
| latent variance | collapse 없이 `h0` variance 대비 일정 비율 유지 |
| perturbation sensitivity | latent perturbation 시 reconstruction이 유의미하게 변함 |

### 8.2 성공 gate

Phase 1 성공 기준은 다음 네 가지다.

| Gate | 통과 기준 |
|---|---|
| P1-G1. 학습 가능성 | validation reconstruction loss가 초기값 대비 명확히 감소 |
| P1-G2. 단계 곡선 | `z1 → z2 → z3`로 갈수록 복원 난도가 증가 |
| P1-G3. 병목 유효성 | `z3`가 `h0`를 너무 쉽게 복원하지 않음 |
| P1-G4. collapse 방지 | `z3`에서도 variance와 cosine이 완전히 붕괴하지 않음 |

정량 기준 초안:

- validation MSE가 초기 epoch 대비 20% 이상 감소하면 학습 가능성 통과
- `z3` MSE가 `z1` MSE보다 높고, cosine은 낮아야 함
- `z3` cosine이 0에 가까우면 병목이 너무 강한 것으로 판단
- latent variance가 거의 0이면 collapse로 판단

## 9. 산출물

Phase 1 완료 시 다음 파일을 만든다.

| 산출물 | 위치 |
|---|---|
| Kaggle kernel | `kaggle/phase1/run_phase1.py` |
| Kaggle metadata | `kaggle/phase1/kernel-metadata.json` |
| push script | `scripts/push_kaggle_phase1.sh` |
| 결과 문서 | `docs/phase1-result.md` |
| 연구 노트 | `docs/plan/phase-1-research-plan.md` |
| metrics | `outputs/phase1/lace_phase1/metrics.json` |
| train log | `outputs/phase1/lace_phase1/train_log.jsonl` |
| summary | `outputs/phase1/lace_phase1/summary.md` |
| checkpoint | `outputs/phase1/lace_phase1/checkpoints/` |

`outputs/phase1`은 git에 포함하지 않는다. 문서와 실행 코드만 git에 포함한다.

## 10. Sprint 계획

## Sprint 1. 데이터 및 cache 확장

**목표**: Phase 0의 fallback 수준을 벗어나, Phase 1 학습에 쓸 수 있는 latent cache를 만든다.

**작업**:

- `kaggle/phase1/run_phase1.py` 초안 생성
- Phase 0의 text loading/cache logic 재사용
- WikiText-2 또는 XSum dataset source 연결
- dataset source가 없을 때는 fallback corpus를 크게 반복해 smoke run만 수행
- `h0`, attention mask, text id, config를 cache로 저장

**검증**:

- 256 sample smoke run에서 cache 생성
- 5k sample run에서 `h0` shape가 `[N, 128, 512]`
- cache reload 후 tensor shape와 dtype 보존

**Demo 가능한 결과**:

- `metrics.json`에 sample count, hidden shape, active token 수 기록
- `summary.md`에 cache 생성 결과 기록

## Sprint 2. Pooling + expander baseline

**목표**: 학습 가능한 reverse expander가 average pooling latent에서 `h0`를 복원할 수 있는지 확인한다.

**작업**:

- C0 average pooling으로 `z1/z2/z3` 생성
- E0 또는 E1 expander 구현
- `z1 → h0`, `z2 → h0`, `z3 → h0` direct reconstruction 학습
- stage별 MSE/cosine 기록
- train/validation split 추가

**검증**:

- train loss가 감소
- validation loss가 함께 감소
- `z3` 복원이 `z1`보다 어렵게 나옴

**Demo 가능한 결과**:

- pooling baseline의 stage별 reconstruction table
- train curve가 포함된 `summary.md`

## Sprint 3. Stage-wise reverse expansion

**목표**: LACE의 단계형 복원 구조를 실제로 학습한다.

**작업**:

- `z3 → z2`
- `z2 → z1`
- `z1 → h0`
- 각 stage expander를 독립 또는 공유 구조로 비교
- direct-to-h0 복원과 stage-wise 복원을 모두 기록

**검증**:

- stage-wise loss가 감소
- direct-to-h0보다 stage-wise가 안정적인지 확인
- stage별 error accumulation 확인

**Demo 가능한 결과**:

- `z3 → z2 → z1 → h0` 전체 복원 metric
- direct reconstruction과 stage-wise reconstruction 비교 table

## Sprint 4. Learned-query attention compression

**목표**: LACE-small의 첫 번째 learned compression path를 만든다.

**작업**:

- learned query attention pooling 구현
- query 수를 stage token 수와 맞춤: 64, 32, 16
- pooling baseline과 동일 expander capacity로 학습
- C0 vs C1 비교 table 생성

**검증**:

- C1이 최소한 C0와 비슷한 수준으로 학습 가능
- C1이 collapse하지 않음
- C1의 `z3`가 너무 쉬운 shortcut이 되지 않음

**Demo 가능한 결과**:

- C0 average pooling vs C1 attention compression 비교
- stage별 MSE/cosine/variance table

## Sprint 5. Perturbation 및 collapse 검사

**목표**: expander가 latent를 실제로 사용하는지 확인한다.

**작업**:

- `z_t`에 small Gaussian perturbation 추가
- `z_t` 일부 token dropout
- shuffled latent control 추가
- reconstruction 변화량 측정

**검증**:

- perturbation이 reconstruction에 영향을 줌
- shuffled latent에서 성능이 확실히 나빠짐
- variance가 0에 수렴하지 않음

**Demo 가능한 결과**:

- perturbation sensitivity table
- collapse 여부 판단 노트

## Sprint 6. Phase 1 결과 정리

**목표**: 다음 Phase로 갈지 판단 가능한 연구 결과 문서를 만든다.

**작업**:

- `docs/phase1-result.md` 작성
- 성공/실패 gate 판정
- Phase 2로 넘길 baseline 후보 정리
- 실패 시 구조 수정안 정리

**검증**:

- Phase 1 gate 네 가지에 대해 pass/fail 명시
- 재현 명령어와 Kaggle kernel URL 기록
- metrics 파일과 문서 숫자 일치 확인

## 11. 실행 명령 초안

로컬 문법 검사:

```bash
python3 -m py_compile kaggle/phase1/run_phase1.py
```

Kaggle 업로드:

```bash
bash scripts/push_kaggle_phase1.sh
```

명시 명령:

```bash
kaggle kernels push -p kaggle/phase1 --accelerator NvidiaTeslaT4 --timeout 3600
```

상태 확인:

```bash
kaggle kernels status dennisparknd/lace-phase-1-latent-compression
```

결과 다운로드:

```bash
mkdir -p outputs/phase1
kaggle kernels output dennisparknd/lace-phase-1-latent-compression -p outputs/phase1
```

## 12. 판단 기준

### 12.1 Phase 2로 진행

다음 조건을 만족하면 Phase 2로 넘어간다.

- average pooling baseline에서 expander 학습이 안정적으로 됨
- learned attention compression이 collapse하지 않음
- `z1/z2/z3` 난도 곡선이 일관됨
- perturbation 검사에서 latent 사용이 확인됨

### 12.2 Phase 1 반복

다음 조건이면 Phase 1을 한 번 더 반복한다.

- loss는 감소하지만 validation이 불안정함
- attention compression이 pooling보다 항상 나쁨
- `z3`가 너무 쉽게 복원되어 병목이 약함
- `z3`가 완전히 무의미해져 병목이 강함

### 12.3 연구 방향 수정

다음 조건이면 가설 또는 구조를 수정한다.

- pooling baseline조차 학습되지 않음
- expander가 latent perturbation에 둔감함
- 모든 stage가 비슷한 난도로 나와 compression level이 의미 없어짐
- 데이터가 작아 metric이 일관되지 않음

## 13. 위험 요소와 대응

| 위험 | 징후 | 대응 |
|---|---|---|
| Kaggle dataset 미연결 | fallback corpus만 사용 | smoke run으로 제한하고 dataset source 연결 후 본 실험 |
| GPU OOM | batch에서 CUDA memory error | batch size 축소, fp16 사용, max samples 축소 |
| expander shortcut | `z3`가 너무 쉽게 `h0` 복원 | token 수 축소, dropout, capacity 제한 |
| collapse | latent variance가 0에 가까움 | variance penalty, attention temperature 조정 |
| overfit | train loss만 감소 | validation split 확대, dropout 추가 |
| metric 착시 | MSE만 좋아짐 | cosine, perturbation, variance 함께 확인 |
| scope creep | renderer까지 붙이고 싶어짐 | Phase 1에서는 text generation 금지 |

## 14. Rollback 계획

Phase 1 구현이 실패해도 Phase 0 결과와 문서는 보존한다.

- `kaggle/phase1/`만 되돌리면 Phase 0에는 영향 없음
- `scripts/push_kaggle_phase1.sh`는 Phase 0 script와 분리
- `outputs/phase1/`은 git ignore 대상
- 실패 결과도 `docs/phase1-result.md`에 남기되, 성공 주장으로 쓰지 않음

## 15. 완료 정의

Phase 1은 다음 조건을 만족하면 완료로 본다.

1. Kaggle에서 Phase 1 kernel이 끝까지 실행된다.
2. `metrics.json`, `train_log.jsonl`, `summary.md`가 생성된다.
3. pooling baseline과 learned attention compression 중 최소 하나가 안정적으로 학습된다.
4. stage별 난도 곡선이 문서화된다.
5. perturbation/collapse 검사가 포함된다.
6. `docs/phase1-result.md`에 pass/fail 판단이 기록된다.

이 완료 조건을 만족하면 Phase 2에서는 Gaussian/mask baseline과 LACE-small을 같은 조건에서 비교한다.
