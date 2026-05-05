# V2 Research Questions

이 문서는 v2 기준 연구 질문을 정리한다. v2의 목적은 LACE를 단순 latent downsampling 연구가 아니라, **semantic skeleton을 terminal forward state로 사용하는 Diffusion Language Modeling 연구**로 재정렬하는 것이다.

## 1. 핵심 질문

v2의 중심 질문은 다음이다.

> 중요도 기반 semantic compression trajectory가 random corruption이나 anchor prediction보다 더 좋은 generation-capable reverse process를 학습시키는가?

이 질문은 세 가지를 동시에 묻는다.

1. 중요한 token 또는 semantic unit이 compressed terminal state가 될 수 있는가?
2. 그 terminal state에서 원문으로 확장하는 reverse process가 random corruption보다 안정적인가?
3. 이 방식이 token reconstruction을 넘어 constrained/open-ended generation에도 도움이 되는가?

## 2. RQ1. Terminal state는 semantic skeleton인가?

### 측정하는 것

Forward process의 마지막 상태 `x_T`가 단순히 많이 손상된 문장이 아니라, 원문의 핵심 의미를 담은 skeleton인지 본다.

### 왜 중요한가

LACE v2의 핵심은 "중요 token을 마지막까지 남긴다"는 것이다. `x_T`가 의미 skeleton이 아니라 빈약한 keyword bag이라면 reverse process는 language prior에 기대게 된다.

### 좋은 결과

- random masking보다 keyword recall, entity preservation, sentence embedding similarity가 높다.
- skeleton만 보고도 원문의 핵심 사건, 주체, 관계를 대략 복원할 수 있다.

### 나쁜 결과

- random masking과 차이가 없다.
- 핵심 entity는 남지만 relation이나 event structure가 사라진다.
- scorer가 surface frequency에만 반응해 semantic skeleton을 만들지 못한다.

## 3. RQ2. Importance-guided forward는 random corruption보다 좋은 reverse trajectory를 만드는가?

### 측정하는 것

같은 reverse model에서 importance-ordered skeleton이 random skeleton 또는 uniform masking보다 더 복원하기 쉬운지 본다.

### 왜 중요한가

Diffusion에서 forward process의 가치는 reverse learning 문제를 어떻게 만드는지에 있다. skeleton이 좋더라도 reverse model이 안정적으로 확장하지 못하면 generation path로 이어지지 않는다.

### 좋은 결과

- `x_t -> x_{t-1}` 또는 `x_t -> x_0` reconstruction loss가 낮다.
- skeleton-to-text output이 random baseline보다 semantic consistency가 높다.
- compression level이 깊어질수록 난도가 예측 가능하게 증가한다.

### 나쁜 결과

- random masking reverse와 성능 차이가 없다.
- skeleton이 너무 sparse해서 reverse model이 generic text를 생성한다.
- token reconstruction은 좋아도 open-ended generation에서는 붕괴한다.

## 4. RQ3. 중요한 token은 auxiliary anchor가 아니라 terminal state인가?

### 측정하는 것

ADLM-style anchor prediction과 v2 방식의 차이를 직접 비교한다.

비교 축:

```text
A. Random forward + anchor prediction
B. Importance-ordered forward + no anchor prediction
C. Importance-ordered forward + anchor prediction
D. Random forward + no anchor prediction
```

### 왜 중요한가

v2의 novelty는 중요 token을 reverse-time 보조 조건으로 쓰는 것이 아니라, forward trajectory 자체를 정의하는 데 있다.

### 좋은 결과

- `B > A`: terminal skeleton 방식이 anchor prediction보다 강하다.
- 또는 `B ≈ A`: 더 단순하고 해석 가능한 forward trajectory로 비슷한 성능을 얻는다.
- `C`가 가장 좋으면 skeleton forward와 anchor prediction이 상보적이라는 후속 주장이 가능하다.

### 나쁜 결과

- `A >> B`: forward trajectory보다 reverse conditioning이 더 중요하다.
- `D`와 `B` 차이가 작다: importance ordering이 forward process에 별 도움을 주지 않는다.

## 5. RQ4. Reverse model은 skeleton을 실제로 사용하는가?

### 측정하는 것

Correct skeleton, shuffled skeleton, random skeleton, wrong-document skeleton을 비교한다.

### 왜 중요한가

Reverse model이 skeleton을 무시하고 language prior만으로 그럴듯한 문장을 만들면 LACE 주장은 약해진다.

### 좋은 결과

- correct skeleton이 가장 좋다.
- shuffled skeleton에서 order-sensitive metric이 하락한다.
- wrong-document skeleton에서 semantic consistency와 reconstruction이 크게 하락한다.
- top-k important tokens를 제거하면 low-k 제거보다 성능이 더 많이 나빠진다.

### 나쁜 결과

- skeleton을 바꿔도 output 품질이 거의 변하지 않는다.
- wrong-document skeleton에서도 비슷한 문장이 나온다.

## 6. RQ5. v1 latent 결과는 v2에서 어떤 역할인가?

v1 결과는 v2의 직접 증거가 아니라 사전 탐색이다.

v1에서 확인한 핵심 교훈:

- hidden MSE/cosine이 좋아도 generation은 실패할 수 있다.
- token-head proxy는 decoder-compatible objective가 아닐 수 있다.
- positional scaffold는 강력한 shortcut이 될 수 있다.
- Gaussian/noise control은 계속 유지해야 한다.

따라서 v2에서도 평가 지표를 다음처럼 분리한다.

| 지표군 | 의미 |
|---|---|
| skeleton preservation | terminal state가 의미를 보존하는가 |
| reconstruction | skeleton에서 원문 또는 이전 단계가 복원되는가 |
| skeleton use | model이 skeleton을 실제로 쓰는가 |
| token proxy | teacher-forced token metric이 좋아지는가 |
| generation | constrained/open-ended generation이 실제로 개선되는가 |
