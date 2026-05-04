# 정보 압축 Forward Process 후보군

## 1. 목적

본 문서는 LACE 연구에서 사용할 수 있는 **정보 압축 기반 forward process 후보군**을 정리한다.

핵심 연구 질문은 다음이다.

> 언어 diffusion에서 forward process를 random noise/corruption이 아니라 information compression으로 정의하면 의미 보존에 더 유리한가?

이 질문을 검증할 때 반드시 learned attention compression만 사용할 필요는 없다. 중요한 것은 forward process가 원본 latent `h0`를 임의로 망가뜨리는 것이 아니라, **정보율을 줄이는 압축 경로**로 작동하는가이다.

따라서 후보군은 크게 두 층으로 나눈다.

1. **핵심 가설 검증용 후보**: 단순하지만 안정적이고, noise/mask와 공정하게 비교할 수 있는 compression forward process
2. **후속 개선용 후보**: 더 똑똑한 압축을 목표로 하지만 학습 안정성 문제가 있을 수 있는 learned/adaptive compression

## 2. 기본 설정

모든 후보군은 같은 encoder latent에서 출발한다.

```text
text
→ frozen encoder
→ h0 ∈ R^{N × d}
→ compression forward process
→ z_t ∈ R^{N_t × d}
```

1차 실험 기본값:

```text
h0: 128 × d
z1: 64 × d
z2: 32 × d
z3: 16 × d
```

여기서 `t`는 noise level이 아니라 information rate 또는 latent token budget에 대응한다.

```text
t 증가
→ N_t 감소
→ 정보율 감소
→ 복원 난도 증가
```

## 3. 비교 기준

후보군은 다음 기준으로 비교한다.

| 기준 | 질문 |
|---|---|
| 결정성 | 같은 `h0`에서 항상 같은 `z_t`가 나오는가? |
| 학습 필요성 | forward process 자체에 학습 파라미터가 있는가? |
| 정보율 제어 | token budget이나 bottleneck 강도를 명확히 조절할 수 있는가? |
| 안정성 | 작은 데이터와 단일 GPU에서 안정적으로 돌아가는가? |
| 의미 보존 가능성 | 표면 정보보다 의미 정보를 더 오래 남길 가능성이 있는가? |
| baseline 적합성 | Gaussian noise/mask와 공정하게 비교하기 쉬운가? |
| 논문 주장력 | 결과가 좋을 때 핵심 가설을 뒷받침하기 좋은가? |

## 4. 후보군 요약

| 후보 | 유형 | 학습 필요 | 1차 우선순위 | 역할 |
|---|---|---|---|---|
| Average pooling | 고정 압축 | 없음 | 매우 높음 | 핵심 compression condition |
| Strided token selection | 고정 압축 | 없음 | 높음 | 단순 downsampling 대조군 |
| Masked token dropping | 고정/무작위 제거 | 없음 | 높음 | compression과 corruption 사이 경계 비교 |
| PCA / low-rank projection | 고정 또는 사후 학습 | 선택 | 중간 | 차원 방향 압축 |
| Learned linear bottleneck | 학습 압축 | 필요 | 중간 | 가장 단순한 learned compression |
| Learned attention pooling | 학습 압축 | 필요 | 중간 | LACE adaptive 후보 |
| Semantic query pooling | 학습 압축 | 필요 | 낮음 | 의미 중심 compression 후보 |
| VQ bottleneck | 이산 압축 | 필요 | 낮음 | codebook 기반 정보율 제어 |
| Variational bottleneck | 확률적 압축 | 필요 | 낮음 | MI/KL 기반 bottleneck |
| Segment summary latent | 구조적 압축 | 필요/선택 | 낮음 | 문단/구간 단위 의미 압축 |

## 5. 후보 1: Average Pooling

### 정의

가장 단순한 compression forward process다. 인접한 latent token들을 평균내어 token 수를 줄인다.

```text
h0: 128 × d
→ z1: 64 × d
→ z2: 32 × d
→ z3: 16 × d
```

예:

```text
z1[i] = mean(h0[2i], h0[2i + 1])
```

### 장점

- 구현이 매우 쉽다.
- 학습 파라미터가 없어 안정적이다.
- 같은 입력에 항상 같은 압축 결과가 나온다.
- token budget으로 정보율을 명확히 통제할 수 있다.
- Gaussian noise, random mask와 비교하기 쉽다.

### 단점

- 중요한 token과 덜 중요한 token을 구분하지 않는다.
- 의미 구조를 적극적으로 보존한다고 말하기 어렵다.
- reviewer가 "단순 downsampling autoencoder 아닌가?"라고 볼 수 있다.

### 연구상 위치

Average pooling은 단순 baseline이 아니라, **핵심 가설 검증용 primary compression condition**으로 격상할 가치가 있다.

핵심 가설이 다음이라면:

> compression forward process가 random corruption보다 나은가?

average pooling도 충분히 compression forward process다. 만약 average pooling이 Gaussian noise나 random mask보다 좋다면, "corruption보다 compression이 좋은 inductive bias일 수 있다"는 주장을 뒷받침할 수 있다.

### 추천 우선순위

**최우선.**

Phase 2의 기본 compression condition으로 둔다.

## 6. 후보 2: Strided Token Selection

### 정의

평균을 내지 않고 일정 간격으로 latent token을 선택한다.

```text
h0[0], h0[2], h0[4], ...
```

또는 stage별로:

```text
z1 = h0[::2]
z2 = h0[::4]
z3 = h0[::8]
```

### 장점

- 구현이 매우 쉽다.
- average pooling과 달리 원본 token vector를 그대로 보존한다.
- "섞는 압축"과 "선택하는 압축"을 비교할 수 있다.

### 단점

- 선택되지 않은 token 정보는 완전히 사라진다.
- 문장 위치에 민감하다.
- 의미적으로 중요한 token을 놓칠 수 있다.

### 연구상 위치

Average pooling이 "부드러운 압축"이라면, strided selection은 "거친 선택 압축"이다. 둘을 비교하면 compression 방식 자체가 결과에 얼마나 영향을 주는지 볼 수 있다.

### 추천 우선순위

**높음.**

계산 비용이 거의 없으므로 Phase 2에 함께 넣을 수 있다.

## 7. 후보 3: Masked Token Dropping

### 정의

latent token 일부를 제거하거나 0-vector로 바꾼다.

```text
h0
→ 일부 latent token 제거
→ 남은 token 또는 masked latent로 복원
```

두 방식이 있다.

| 방식 | 설명 |
|---|---|
| deterministic drop | 일정 간격 또는 규칙으로 token 제거 |
| random drop | 무작위로 token 제거 |

### 장점

- LLaDA류 masking과 연결하기 쉽다.
- compression과 corruption 사이의 경계 조건으로 사용할 수 있다.
- average pooling과 같은 token budget으로 맞추기 쉽다.

### 단점

- random drop은 compression이라기보다 corruption에 가깝다.
- deterministic drop도 선택되지 않은 token은 완전히 버린다.
- 의미 구조를 보존한다는 보장이 약하다.

### 연구상 위치

이 후보는 pure compression이라기보다 **compression-corruption boundary baseline**이다.

즉, 다음 비교에 유용하다.

```text
average pooling compression
vs
latent token dropping
vs
random mask/drop corruption
```

### 추천 우선순위

**높음.**

Phase 2에서 mask/corruption baseline과 연결하는 다리 역할을 한다.

## 8. 후보 4: PCA / Low-Rank Projection

### 정의

latent token 수를 줄이는 대신, hidden dimension 또는 latent subspace의 rank를 줄인다.

예:

```text
h0 ∈ R^{N × d}
→ z_t ∈ R^{N × r}, r < d
```

또는 다시 원래 dimension으로 project한다.

```text
h0
→ low-rank projection
→ compressed representation
→ reconstruction
```

### 장점

- 정보 압축이라는 해석이 명확하다.
- token 길이가 아니라 vector dimension의 정보율을 조절할 수 있다.
- PCA처럼 학습 없이 고정 basis를 쓸 수도 있다.

### 단점

- 현재 LACE 설계의 `N_t × d` token budget compression과 다르다.
- token 수를 줄이는 diffusion trajectory와 직접 연결이 약하다.
- PCA를 쓰려면 사전 fitting이 필요하다.

### 연구상 위치

Low-rank 계열은 "정보율 제어"를 보여주기 좋지만, 1차 실험의 핵심 후보로 삼기에는 약간 옆길이다. 다만 reviewer가 정보 압축을 더 엄밀히 요구할 때 보조 실험으로 가치가 있다.

### 추천 우선순위

**중간.**

Phase 2 이후 보조 ablation으로 둔다.

## 9. 후보 5: Learned Linear Bottleneck

### 정의

학습 가능한 linear projection 또는 MLP로 latent를 압축한다.

```text
h0
→ linear / MLP bottleneck
→ z_t
```

token 수를 줄이는 방식과 dimension을 줄이는 방식을 모두 사용할 수 있다.

### 장점

- learned compression 중 가장 단순하다.
- attention보다 안정적일 가능성이 높다.
- average pooling과 learned attention 사이의 중간 단계로 좋다.

### 단점

- 단순 projection이므로 의미적으로 중요한 정보를 골라낸다고 보기 어렵다.
- bottleneck이 약하면 identity shortcut이 생길 수 있다.
- bottleneck이 강하면 collapse가 생길 수 있다.

### 연구상 위치

Learned attention compression이 불안정하다면, 바로 attention을 고치기 전에 learned linear bottleneck을 중간 후보로 넣을 수 있다.

### 추천 우선순위

**중간에서 높음.**

Phase 1.5에서 attention 대체 후보로 검토할 만하다.

## 10. 후보 6: Learned Attention Pooling

### 정의

학습 가능한 query들이 전체 `h0`를 attention으로 읽고, 더 적은 수의 latent token을 만든다.

```text
learned queries Q_t
attention(Q_t, h0)
→ z_t ∈ R^{N_t × d}
```

예:

```text
64 queries → z1
32 queries → z2
16 queries → z3
```

### 장점

- 중요한 정보를 선택적으로 압축할 수 있다.
- LACE의 "Adaptive Compression"이라는 이름과 잘 맞는다.
- 잘 되면 average pooling보다 강한 연구 기여가 된다.

### 단점

- 학습이 불안정하다.
- query들이 비슷한 정보를 볼 수 있다.
- latent scale이 흔들릴 수 있다.
- expander와 co-adaptation하면서 실제 정보 병목이 약해질 수 있다.
- 현재 WikiText-2 실험에서는 pooling보다 훨씬 약했다.

### 현재 실험에서 드러난 문제

WikiText-2 subset run에서 attention compression은 다음 문제가 있었다.

| 지표 | Pooling `z3` | Attention `z3` |
|---|---:|---:|
| MSE | 0.020779 | 0.123256 |
| Cosine | 0.544056 | 0.236275 |
| Variance ratio | 0.182608 | 3.797502 |
| Perturbation sensitivity | 0.002050 | 0.001456 |

이는 현재 attention compression이 안정적인 정보 압축 경로를 만들지 못하고 있다는 신호다.

### 개선 방향

- pooled-latent anchor loss
- attention entropy regularization
- query dropout
- latent scale regularization
- direct-to-h0 reconstruction loss

### 추천 우선순위

**당장 primary condition으로 쓰기에는 낮음.**

후속 개선 후보로 둔다. Phase 2 핵심 비교에는 average pooling을 먼저 사용한다.

## 11. 후보 7: Semantic Query Pooling

### 정의

learned attention pooling의 변형이다. query를 단순 learned vector가 아니라 semantic role에 가깝게 분리한다.

예:

```text
query group 1: entity 정보
query group 2: relation 정보
query group 3: event/action 정보
query group 4: discourse 정보
```

실제 구현에서는 role label이 없어도 query group별 regularization을 둘 수 있다.

### 장점

- 의미 보존 중심 compression이라는 연구 목표와 잘 맞는다.
- 나중에 해석 가능성 분석에 좋다.

### 단점

- 설계가 복잡하다.
- supervision 없이 semantic role이 자연히 분리될 가능성이 불확실하다.
- 1차 실험으로는 scope가 크다.

### 추천 우선순위

**낮음.**

초기 논문보다 후속 연구 방향에 가깝다.

## 12. 후보 8: VQ Bottleneck

### 정의

continuous latent를 codebook entry로 양자화한다.

```text
h0
→ encoder/compressor
→ nearest codebook vector
→ z_t
```

### 장점

- 정보율 제어가 명확하다.
- codebook size로 capacity를 조절할 수 있다.
- discrete latent라 해석이 쉬울 수 있다.

### 단점

- codebook collapse 위험이 있다.
- 학습 안정성이 어렵다.
- 현재 연구 질문의 첫 검증에는 과하다.

### 추천 우선순위

**낮음.**

최종 논문 확장 또는 후속 실험으로 둔다.

## 13. 후보 9: Variational Bottleneck

### 정의

latent를 확률분포로 만들고 KL term으로 정보량을 제어한다.

```text
q(z_t | h0) = N(μ_t, σ_t)
z_t ~ q(z_t | h0)
L_info = KL(q(z_t | h0) || p(z_t))
```

### 장점

- information bottleneck 관점과 잘 맞는다.
- mutual information 논의와 연결하기 좋다.
- diffusion의 stochastic process와도 연결 가능하다.

### 단점

- posterior collapse 위험이 있다.
- 튜닝이 어렵다.
- Gaussian noise baseline과 경계가 흐려질 수 있다.

### 추천 우선순위

**낮음에서 중간.**

정보이론적 formalization을 강화할 때 사용한다.

## 14. 후보 10: Segment Summary Latent

### 정의

문장을 token 단위로 압축하는 대신, 구간 또는 segment 단위로 요약 latent를 만든다.

```text
h0 tokens
→ segment 1 summary
→ segment 2 summary
→ ...
→ z_t
```

### 장점

- 긴 문서 generation과 잘 맞는다.
- 문단/담화 구조를 보존하기 좋다.
- semantic compression이라는 직관이 강하다.

### 단점

- segment boundary를 어떻게 정할지 문제가 있다.
- 처음부터 long-context 문제로 커질 수 있다.
- 실험 scope가 커진다.

### 추천 우선순위

**낮음.**

Phase 3 이후 long-context generation에서 다시 검토한다.

## 15. Phase 2에 넣을 추천 후보

현재 연구 상태를 고려하면, Phase 2에는 후보를 너무 많이 넣으면 안 된다. 우선 다음 네 가지면 충분하다.

| 조건 | forward process | 목적 |
|---|---|---|
| B1 | Gaussian noise | 기존 latent diffusion 계열 baseline |
| B2 | random mask/drop | corruption 계열 baseline |
| B3 | average pooling compression | 핵심 compression condition |
| B4 | strided token selection 또는 deterministic drop | 선택/제거형 compression 대조군 |

Learned attention compression은 다음 조건을 만족하기 전까지 primary condition에서 제외한다.

- WikiText-2에서 pooling과 같은 order의 MSE를 보임
- cosine이 pooling에 근접함
- variance ratio가 안정화됨
- perturbation sensitivity가 개선됨

## 16. 현재 연구 전략 수정안

기존에는 learned attention compression을 LACE-small의 기본 후보로 두었지만, 실험 결과를 보면 다음 전략이 더 안전하다.

### 기존 전략

```text
핵심 compression = learned attention compression
average pooling = baseline
```

### 수정 전략

```text
핵심 compression = deterministic token-budget compression
1차 대표 = average pooling
2차 후보 = learned/adaptive compression
```

이 수정은 연구를 약하게 만드는 것이 아니다. 오히려 핵심 가설을 더 명확하게 만든다.

> 먼저 "compression이 corruption보다 나은가?"를 검증하고, 그 다음 "learned compression이 fixed compression보다 나은가?"를 검증한다.

## 17. 최종 권고

현재 단계의 우선순위는 다음이다.

1. **Average pooling을 primary compression forward process로 사용한다.**
2. Gaussian noise, random mask/drop과 같은 조건에서 비교한다.
3. Strided token selection을 보조 compression condition으로 추가한다.
4. Learned attention compression은 Phase 1.5 개선 대상으로 유지한다.
5. Phase 2의 논점은 learned module 성능이 아니라 compression-vs-corruption inductive bias로 둔다.

이렇게 하면 첫 논문 또는 첫 연구 노트의 주장이 더 깔끔해진다.

> 단순한 deterministic compression만으로도 random corruption보다 의미 보존이 낫다면, "Compression, Not Corruption"의 핵심 주장은 이미 강한 실험적 근거를 얻는다.
