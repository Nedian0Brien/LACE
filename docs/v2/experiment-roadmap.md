# V2 Experiment Roadmap

이 문서는 v2 실험을 위한 단계형 로드맵이다. 기존 v1의 Phase 0-3A와 충돌하지 않도록 v2 실험은 `S` prefix를 사용한다. 여기서 `S`는 semantic skeleton track을 뜻한다.

v1에서 이어갈 실험 규율은 [v1-carryover.md](./v1-carryover.md)를 따른다. 특히 proxy metric과 generation metric을 분리하고, skeleton-use control과 강한 baseline을 초반부터 유지한다.

## S0. Skeleton Pipeline Sanity Check

### 질문

텍스트에서 importance score를 계산하고, 단계별 semantic skeleton을 안정적으로 만들 수 있는가?

### 최소 조건

| 조건 | 설명 |
|---|---|
| random masking | 기존 masked DLM류 기본 corruption baseline |
| uniform length skeleton | importance 없이 길이만 맞춘 skeleton |
| frequency/PMI skeleton | 통계 기반 중요 token 보존 |
| attention-received skeleton | encoder attention 기반 중요 token 보존 |

### 평가

- skeleton token count
- keyword/entity recall
- sentence embedding similarity
- compression level별 semantic similarity curve
- random baseline 대비 preservation gap

### 성공 기준

importance-guided skeleton이 random/uniform skeleton보다 핵심 entity와 semantic similarity를 더 잘 보존해야 한다.

## S1. Skeleton Use Controls

### 질문

Reverse model 또는 reconstruction evaluator가 skeleton을 실제로 사용하는가?

### Control

| Control | 목적 |
|---|---|
| correct skeleton | 정상 조건 |
| shuffled skeleton | 순서 민감도 확인 |
| random skeleton | 무작위 핵심 token 대비 |
| wrong-document skeleton | content mismatch 민감도 확인 |
| remove top-k important tokens | 핵심 token 제거 영향 |
| remove low-k tokens | 부가 token 제거 영향 |

### 성공 기준

correct skeleton이 가장 좋아야 하며, wrong-document와 top-k removal에서 성능이 명확히 하락해야 한다.

## S2. Skeleton-to-Text Reconstruction

### 질문

Semantic skeleton에서 원문 또는 이전 compression level로 복원하는 reverse process를 학습할 수 있는가?

### 학습 태스크

```text
x_t -> x_{t-1}
x_t -> x_0
```

### 비교 조건

- random masking reverse
- uniform skeleton reverse
- PMI/frequency skeleton reverse
- attention skeleton reverse
- oracle keyword skeleton upper bound

### 평가

- token reconstruction accuracy
- BLEU/ROUGE
- BERTScore
- entity/relation preservation
- grammar/perplexity
- skeleton faithfulness

### 성공 기준

importance-guided skeleton reverse가 random masking reverse보다 semantic preservation과 reconstruction에서 좋아야 한다.

## S3. Anchor Baseline Comparison

### 질문

중요 token을 예측해서 보조 조건으로 주는 방식보다, forward terminal state로 보존하는 방식이 더 나은가?

### 핵심 비교

```text
A. Random forward + anchor prediction
B. Importance-ordered forward + no anchor prediction
C. Importance-ordered forward + anchor prediction
D. Random forward + no anchor prediction
```

### 성공 기준

`B > A`이면 v2 핵심 주장이 강해진다. `B ≈ A`라도 더 단순하고 해석 가능한 trajectory라는 주장이 가능하다. `C`가 가장 좋으면 skeleton forward와 anchor prediction이 상보적이라는 후속 방향이 생긴다.

## S4. Constrained Generation

### 질문

Skeleton이 constrained generation에서 의미 일관성과 skeleton 충실도를 개선하는가?

### 평가 순서

1. prefix-conditioned expansion
2. masked span reconstruction
3. skeleton-to-sentence generation
4. short constrained generation

Open-ended generation은 이 단계가 통과된 뒤 진행한다.

## S5. Open-ended Generation

### 질문

Semantic skeleton 기반 reverse process가 open-ended generation에서 random corruption baseline보다 의미 일관성과 반복 제어를 개선하는가?

### 평가

- semantic consistency
- repetition rate
- diversity
- coherence
- skeleton faithfulness
- human preference 또는 LLM-as-judge 보조 평가

## 즉시 다음 단계 추천

바로 S2/S3로 가지 말고 S0를 먼저 수행한다.

이유:

1. v2의 핵심 단위인 semantic skeleton이 아직 이 repo에서 실험적으로 생성된 적이 없다.
2. v1은 latent compression track이므로, v2의 token skeleton claim을 직접 지지하지 않는다.
3. importance scorer가 약하면 reverse/generation 실험 실패 원인을 분리하기 어렵다.

따라서 다음 Kaggle 실험 후보는 다음이다.

```text
S0: semantic skeleton extraction and preservation validation
```

산출물:

- `kaggle/v2_s0/run_v2_s0.py`
- `kaggle/v2_s0/kernel-metadata.json`
- `scripts/push_kaggle_v2_s0.sh`
- `docs/v2/plan/s0-skeleton-pipeline-plan.md`
- `docs/v2/experiments/s0-skeleton-pipeline.md`
