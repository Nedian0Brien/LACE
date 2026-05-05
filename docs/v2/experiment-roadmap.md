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

## 현재 다음 단계 추천

S0, S1, S2는 완료됐다.

결과 문서는 다음에 있다.

- [experiments/s0-skeleton-pipeline.md](./experiments/s0-skeleton-pipeline.md)
- [experiments/s1-skeleton-use-controls.md](./experiments/s1-skeleton-use-controls.md)
- [experiments/s2-skeleton-to-text-reconstruction.md](./experiments/s2-skeleton-to-text-reconstruction.md)

핵심 판단:

1. IDF/attention skeleton은 random/uniform보다 의미 보존 신호를 보였다.
2. `position_prior`가 강해서 lead-position confound는 실제 risk로 확인됐다.
3. S1에서 `attention_correct`는 `random_same_count`, `wrong_document`, `position_only`, `same_position_random`보다 강했다.
4. 하지만 `position_prior`도 강하므로 S2에서는 위치 보조 구조와 위치 전용 control을 함께 유지한다.
5. `remove_topk`가 `remove_lowk`보다 더 치명적이라는 순서 주장은 아직 확인되지 않았다.
6. S2에서 `attention_scaffold`는 `random_scaffold`, `position_only`, wrong-document control보다 강했다.
7. 다만 `idf_scaffold`는 loss가 가장 낮고, `position_prior_scaffold`는 keyword recall이 매우 높아서 scorer와 위치 편향 분리는 계속 필요하다.

다음 Kaggle 실험 후보는 다음이다.

```text
S3: anchor baseline comparison
```

산출물:

- `kaggle/v2_s3/run_v2_s3.py`
- `kaggle/v2_s3/kernel-metadata.json`
- `scripts/push_kaggle_v2_s3.sh`
- `docs/v2/plan/s3-anchor-baseline-comparison-plan.md`
- `docs/v2/experiments/s3-anchor-baseline-comparison.md`
