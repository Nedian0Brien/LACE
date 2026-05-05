# V1에서 V2로 이어갈 점

이 문서는 v1 latent compression track에서 얻은 성과를 v2 semantic skeleton track에 어떻게 계승할지 정리한다.

핵심 판단은 다음이다.

> v1에서 v2로 이어갈 것은 Average Pooling 자체가 아니라, compression과 corruption을 공정하게 비교하는 실험 철학, proxy와 generation을 분리하는 해석 규율, 그리고 shortcut/confound를 control로 제거하는 방식이다.

## 1. Compression vs Corruption 비교 구도

v1의 가장 중요한 성과는 compression forward process를 corruption/noise forward process와 직접 비교하는 실험 구도를 만든 것이다.

v1 비교 구도:

```text
average_pool / strided_select
vs
random_select / gaussian_noise
```

v2에서는 이 구도를 semantic skeleton 실험으로 번역한다.

```text
importance-guided skeleton
vs
random masking / uniform skeleton / wrong skeleton / anchor baseline
```

따라서 v2 실험은 "semantic skeleton이 좋아 보인다"에서 멈추면 안 된다. 반드시 같은 token budget, 같은 데이터, 같은 reverse capacity 조건에서 random/uniform/anchor baseline과 비교해야 한다.

## 2. Metric family 분리

v1은 MSE, cosine, token-head NLL, frozen decoder NLL, open-ended generation이 서로 다른 증거라는 점을 보여줬다.

v2에서도 지표를 다음처럼 분리한다.

| v1에서 배운 분리 | v2에서 대응되는 분리 |
|---|---|
| hidden-state MSE | skeleton이 원문 정보를 얼마나 보존하는가 |
| representation cosine | skeleton이 의미 방향을 유지하는가 |
| token-head NLL | teacher-forced token reconstruction proxy |
| frozen decoder NLL | decoder-readable generation path |
| open-ended generation | 실제 생성 품질 |
| perturbation / ablation / swap | skeleton을 실제로 쓰는지 |

v2에서 keyword recall, BERTScore, token reconstruction 중 하나만 좋아졌다고 성공을 주장하지 않는다. 최소한 다음 네 층은 분리해서 보고한다.

1. skeleton preservation
2. skeleton use
3. reconstruction / teacher-forced proxy
4. constrained 또는 open-ended generation

## 3. Latent-use 검증을 Skeleton-use 검증으로 번역

v1의 perturbation, ablation, swap test는 v2에서 skeleton control로 이어간다.

v1 질문:

> Expander가 latent를 실제로 사용하는가?

v2 질문:

> Reverse model이 semantic skeleton을 실제로 사용하는가?

v2에 반드시 포함할 control:

```text
correct skeleton
random skeleton
shuffled skeleton
wrong-document skeleton
remove top-k important tokens
remove low-k tokens
```

좋은 결과는 correct skeleton에서 가장 좋고, wrong-document skeleton과 top-k removal에서 성능이 크게 하락하는 것이다.

나쁜 결과는 skeleton을 바꿔도 output이 거의 변하지 않는 것이다. 이 경우 model은 skeleton을 쓰는 것이 아니라 language prior로 그럴듯한 문장을 만들고 있을 가능성이 높다.

## 4. Positional and Shortcut Confounds

v1 Phase 2C/2D는 positional scaffold가 매우 강한 shortcut이 될 수 있음을 보여줬다. v2에서도 같은 문제가 생길 수 있다.

v2에서 의심해야 할 confound:

- 중요한 token 때문이 아니라 token count가 많아서 좋아진 것은 아닌가?
- 남은 token의 position pattern이 쉬워서 좋아진 것은 아닌가?
- scorer가 semantic importance가 아니라 frequency나 sentence position만 따라간 것은 아닌가?
- skeleton order를 무시해도 비슷한 성능이 나오는 것은 아닌가?

따라서 v2 S0/S1에는 다음 control을 우선 포함한다.

```text
same token count random skeleton
same position pattern random skeleton
same keywords shuffled order
same keywords wrong document
frequency-only skeleton
oracle keyword skeleton
```

## 5. 강한 baseline을 유지하는 태도

v1에서 Gaussian positional은 가장 강한 반론이었다. v2에서 Gaussian 자체가 항상 같은 역할을 하지는 않지만, "가장 강한 쉬운 baseline을 옆에 둬야 한다"는 원칙은 유지한다.

v2의 강한 baseline 후보:

| Baseline | 역할 |
|---|---|
| random masking | 기존 masked DLM류 기본 corruption baseline |
| uniform skeleton | importance 없는 길이 맞춤 baseline |
| frequency/PMI skeleton | 단순 통계 기반 baseline |
| attention-received skeleton | 구현 쉬운 importance heuristic |
| oracle keyword skeleton | skeleton upper bound |
| ADLM-style anchor prediction | 가장 중요한 novelty 비교군 |

특히 v2의 novelty가 anchor-based DLM과의 차별화라면, ADLM-style anchor baseline은 선택이 아니라 필수에 가깝다.

## 6. Token-head 실패에서 얻은 objective 교훈

v1 Phase 3A의 token-head objective는 token-head NLL을 크게 낮췄지만, frozen decoder NLL과 open-ended generation을 악화시켰다.

이 부정 결과는 v2에 중요한 경고를 준다.

> teacher-forced reconstruction이 좋아졌다고 generation-capable reverse process가 확보된 것은 아니다.

따라서 v2 실험은 다음 순서로 진행한다.

```text
S0 skeleton preservation
-> S1 skeleton-use controls
-> S2 skeleton-to-text reconstruction
-> S3 anchor baseline comparison
-> S4 constrained generation
-> S5 open-ended generation
```

Open-ended generation을 너무 빨리 최종 판정으로 쓰면 실패 원인 분리가 어렵다. 반대로 teacher-forced proxy만 보고 성공을 말하면 v1 Phase 3A의 오류를 반복할 수 있다.

## 7. Average Pooling의 위치

Average Pooling은 v2의 핵심 방법론이 아니라 v1의 중요한 기준선으로 남긴다.

v2에서 가능한 사용 방식:

```text
latent-side compression baseline:
average_pool_rel_pos

token-side skeleton baseline:
uniform skeleton / frequency skeleton / attention skeleton
```

Average Pooling 결과를 v2의 직접 증거처럼 쓰지 않는다. v2의 주장은 token-level 또는 semantic-unit-level terminal skeleton에 관한 것이기 때문이다.

## 8. Kaggle 운영 체계

v1에서 만든 Kaggle 운영 체계는 v2에도 그대로 가져간다.

계승할 것:

- self-contained Kaggle runner
- `kernel-metadata.json`
- phase별 push script
- `metrics.json`, `summary.md`, `train_log.jsonl`, sample output
- gate 기반 결과 판정
- 결과 문서화와 연구노트 갱신
- 완료 후 verification, commit, push

v2 naming 예시:

```text
kaggle/v2_s0/run_v2_s0.py
scripts/push_kaggle_v2_s0.sh
docs/v2/plan/s0-skeleton-pipeline-plan.md
docs/v2/experiments/s0-skeleton-pipeline.md
outputs/v2_s0/
```

## 9. V2의 첫 실행 원칙

v2의 첫 목표는 큰 generation model을 바로 만드는 것이 아니다.

첫 목표:

> semantic skeleton이 random/uniform skeleton보다 의미를 더 잘 보존하는지, 그리고 downstream evaluator 또는 reverse model이 그 skeleton을 실제로 사용하는지 검증한다.

따라서 즉시 다음 실험은 S0/S1이다. S0/S1을 통과해야 S2 reconstruction, S3 anchor comparison, S4/S5 generation으로 넘어가는 것이 방어 가능하다.
