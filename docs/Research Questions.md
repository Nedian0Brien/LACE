# LACE Research Questions

이 문서는 LACE 연구의 핵심 질문과 노벨티 경계를 정리한다. 특히 이 연구가 단순한 downsampling autoencoder로 보이지 않도록, 무엇을 주장해야 하고 무엇을 주장하지 말아야 하는지를 명확히 한다.

## 1. 핵심 연구 질문

LACE의 중심 질문은 다음이다.

> Diffusion Language Modeling의 forward process를 무작위 corruption/noise/masking이 아니라 정보 압축 과정으로 정의할 수 있는가?

즉, 이 연구의 주 대상은 "압축 autoencoder를 만들 수 있는가?"가 아니다. 주 대상은 language diffusion에서 시간축 `t`를 정보 손상 정도가 아니라 정보율(information rate)의 단계로 해석할 수 있는지 검증하는 것이다.

기존 diffusion language model 계열은 대체로 forward process에서 원본 표현을 점진적으로 망가뜨린다. 예를 들어 Gaussian noise를 더하거나, token을 mask/drop하거나, discrete state를 무작위화한다. LACE는 이 관점을 바꾼다. 원본 표현을 망가뜨리는 대신, 보존 가능한 정보량을 점진적으로 줄이는 구조적 압축 과정을 forward process로 사용한다.

다만 최종 목표는 latent reconstruction 자체가 아니다. LACE는 결국 language generation이 가능한 Diffusion Language Model을 더 잘 만들기 위한 연구다. 따라서 hidden-state reconstruction은 중간 검증 지표이고, 최종 판정은 compression forward process가 실제 generation-capable DLM의 학습과 샘플링을 개선하는지로 내려야 한다.

따라서 LACE의 요지는 다음 문장으로 압축된다.

> Compression, not Corruption.

## 2. 노벨티의 위치

LACE가 새롭다고 주장하면 안 되는 부분은 분명하다.

- hidden state를 짧은 latent로 downsampling하는 것 자체
- 평균 풀링, strided selection, attention pooling 같은 압축 연산 자체
- bottleneck representation을 만들고 다시 복원하는 autoencoder 구조 자체
- reconstruction loss로 압축 latent의 복원 가능성을 확인하는 실험 자체

이 부분들은 이미 autoencoder, latent diffusion, representation compression, hierarchical encoder-decoder 문헌에서 널리 다뤄진 주제다. 그러므로 LACE가 "새로운 압축 autoencoder"라고 주장하면 노벨티가 약하다.

LACE가 주장해야 하는 노벨티는 다음이다.

- language diffusion의 forward process를 corruption process가 아니라 compression process로 재정의한다.
- diffusion time `t`를 노이즈 강도나 mask 비율이 아니라 정보율 단계로 해석한다.
- reverse process를 단순 decoder가 아니라 낮은 정보율 표현에서 높은 정보율 표현으로 확장하는 diffusion-style trajectory로 본다.
- 동일한 reverse capacity, 동일한 데이터, 동일한 학습 예산에서 corruption 기반 forward process와 compression 기반 forward process를 직접 비교한다.
- 압축 방식의 새로움보다, forward process의 의미를 바꾸는 것이 연구의 핵심이다.

이 차이를 명확히 하지 않으면 LACE는 downsampling autoencoder로 읽힐 가능성이 높다. 반대로 이 차이를 실험적으로 보이면, LACE는 "autoencoder 변형"이 아니라 diffusion language modeling의 forward process 설계에 대한 연구가 된다.

## 3. Downsampling Autoencoder와의 구분 기준

다음 조건에 머물면 LACE는 downsampling autoencoder와 본질적으로 다르다고 말하기 어렵다.

- `h0 -> z -> h0` 복원만 평가한다.
- 압축률별 MSE, cosine similarity만 본다.
- reverse expansion이 diffusion-style trajectory인지 확인하지 않는다.
- corruption/noise/masking 기반 forward process와 직접 비교하지 않는다.
- 최종 언어 생성 또는 denoising/generation objective와 연결하지 않는다.

반대로 다음 조건을 만족하면 LACE의 차별성이 생긴다.

- `t`별 압축 단계가 forward diffusion schedule로 정의된다.
- reverse model이 `z_t -> z_{t-1}` 또는 `z_t -> h0`로 점진적 확장을 학습한다.
- 같은 reverse architecture로 compression forward와 corruption forward를 비교한다.
- compression forward가 더 안정적인 reverse trajectory, 더 좋은 의미 보존, 더 좋은 생성 성능을 보이는지 검증한다.
- 실험 질문이 "잘 복원하는가?"가 아니라 "diffusion forward process로서 더 좋은가?"로 이동한다.

## 4. 세부 연구 질문

### RQ1. 정보 압축은 DLM forward process가 될 수 있는가?

가장 중요한 질문이다. 평균 풀링이나 strided selection 같은 단순 압축도 후보가 될 수 있다. 핵심은 압축 방식의 복잡도가 아니라, 원본 표현을 무작위로 망가뜨리는 대신 정보량을 구조적으로 줄이는 과정이 reverse learning에 더 좋은 조건을 제공하는지다.

검증은 반드시 corruption 기반 baseline과 함께 진행해야 한다.

- Gaussian noise forward
- random mask/drop forward
- average pooling compression forward
- strided token selection forward

이 비교 없이는 LACE의 주장이 autoencoder 복원 실험으로 축소된다.

### RQ2. 압축 forward process는 더 학습 가능한 reverse trajectory를 만드는가?

diffusion model에서 중요한 것은 forward process 자체가 아니라, 그 forward process가 reverse process 학습에 어떤 문제를 만들어내는가이다. LACE는 압축 forward가 무작위 corruption보다 더 구조적인 역문제를 만든다고 가정한다.

관찰해야 할 신호는 다음이다.

- reverse expansion loss가 더 빠르게 감소하는가?
- stage별 복원이 단조롭고 안정적인가?
- 낮은 정보율 latent에서 높은 정보율 표현으로 갈수록 표현 품질이 점진적으로 회복되는가?
- 작은 latent perturbation이 출력에 의미 있는 변화를 만드는가?
- reverse model이 latent를 실제로 사용하고 있는가?

### RQ3. 어떤 정보 압축 schedule이 forward process에 적합한가?

정보 압축에는 여러 방식이 있다. 현재 연구에서 우선순위가 높은 후보는 token-budget compression이다.

- 128 token -> 64 token -> 32 token -> 16 token
- 평균 풀링 기반 압축
- strided selection 기반 압축
- deterministic drop 기반 압축

이 방식들은 단순하지만, 오히려 Phase 2에서는 장점이 있다. 압축 연산 자체의 학습 불안정성을 줄이고, "정보 압축 forward process가 corruption보다 나은가?"라는 핵심 가설을 더 깨끗하게 볼 수 있기 때문이다.

learned attention compression은 중요한 후보지만, 처음부터 핵심 가설의 유일한 대표로 두기에는 위험하다. learned attention이 실패하면 "정보 압축 가설"이 실패한 것인지, "attention compressor 구현"이 실패한 것인지 분리하기 어렵다.

### RQ4. adaptive compression은 필요한가?

평균 풀링이 강한 결과를 보인다면, 핵심 가설은 adaptive compressor 없이도 일부 입증될 수 있다. 이 경우 LACE의 1차 주장은 다음이 된다.

> language diffusion의 forward process는 무작위 corruption보다 구조적 정보 압축으로 정의될 수 있다.

그 다음 단계에서 adaptive compression의 필요성을 묻는다.

- 평균 풀링보다 learned linear bottleneck이 나은가?
- learned attention pooling이 더 의미 있는 latent를 만드는가?
- adaptive compression이 긴 문맥이나 의미 밀도가 높은 텍스트에서 이점을 갖는가?

즉, adaptive compression은 핵심 가설의 출발점이 아니라 확장 질문이어야 한다.

### RQ5. reverse expansion은 latent 정보를 실제로 사용하는가?

압축 forward process를 사용하더라도 reverse model이 latent를 제대로 사용하지 않으면 의미가 없다. 따라서 복원 품질만으로는 부족하다.

필요한 검증은 다음이다.

- latent perturbation sensitivity
- latent swap test
- stage ablation
- compression level별 output degradation
- 동일한 reverse model에서 latent 입력을 제거했을 때의 성능 하락

이 실험들은 LACE가 단순 decoder memorization이 아니라 latent-conditioned reverse expansion을 하고 있는지 확인하기 위한 것이다.

### RQ6. latent reconstruction에서 language generation으로 이어지는가?

이 질문은 보조 질문이 아니라 LACE 연구의 최종 도착점이다. Phase 1의 reconstruction 실험은 사전조건일 뿐이고, 그 자체가 연구의 성공을 의미하지 않는다. 우리는 결국 language generation이 가능한 Diffusion Language Model을 만드는 과정을 개선하려는 것이므로, compression forward process가 실제 생성 모델의 학습과 샘플링에 도움이 되는지까지 확인해야 한다.

따라서 LACE의 실험 흐름은 다음처럼 설계되어야 한다.

1. 압축 latent가 원본 hidden representation의 정보를 충분히 보존하는지 확인한다.
2. compression forward가 corruption forward보다 reverse expansion을 더 안정적으로 학습시키는지 비교한다.
3. reverse expansion 결과가 token prediction 또는 decoder 입력으로 사용될 수 있는지 확인한다.
4. 최종적으로 compression-based DLM이 실제 text generation에서 baseline보다 나은 경로를 제공하는지 평가한다.

중요한 점은 `h0` 복원이 끝이 아니라는 것이다. `z_t -> h0`가 어느 정도 가능하더라도, 그 표현이 language decoder나 token reconstruction objective에 연결되지 않으면 LACE는 autoencoder 실험에 머문다.

따라서 이후에는 다음 목표로 확장해야 한다.

- latent reconstruction quality
- semantic preservation
- token-level reconstruction 또는 decoding quality
- denoising/generation objective
- downstream text generation quality

LACE가 논문으로 설득력을 가지려면, "hidden state 복원이 된다"에서 멈추지 않고 "compression forward가 language generation에 유리한 diffusion path를 만든다"까지 이어져야 한다. 이 연결이 확인되어야만 LACE는 "압축 autoencoder"가 아니라 "generation-capable Diffusion Language Model의 forward process를 개선하는 연구"가 된다.

실제로 확인해야 할 생성 관련 질문은 다음이다.

- reconstructed hidden state를 frozen decoder 또는 lightweight decoder에 넣었을 때 token reconstruction이 가능한가?
- compression forward로 학습한 reverse model이 corruption forward로 학습한 reverse model보다 token-level loss를 낮추는가?
- 생성 샘플에서 의미 보존, 반복, 붕괴, 문법성, 다양성이 개선되는가?
- 낮은 정보율 latent에서 시작한 reverse trajectory가 점진적으로 더 구체적인 문장 표현으로 확장되는가?
- sampling 단계에서 compression schedule이 corruption schedule보다 안정적인가?

## 5. Phase 2에서 필요한 핵심 비교

Phase 2는 forward process isolation experiment가 되어야 한다. 목표는 compressor를 멋지게 만드는 것이 아니라, forward process의 철학을 비교하는 것이다.

### 비교 조건

1. Gaussian noise forward
2. Random mask/drop forward
3. Average pooling compression forward
4. Strided token selection forward

### 통제 조건

- 같은 base encoder hidden state `h0`
- 같은 데이터 subset
- 같은 reverse expander architecture
- 같은 학습 step 수
- 같은 optimizer와 batch size
- 가능한 한 비슷한 정보량 또는 난이도
- 같은 평가 metric

### 주요 metric

- validation reconstruction loss
- cosine similarity
- semantic similarity
- stage-wise monotonic recovery
- latent perturbation sensitivity
- latent ablation sensitivity
- reverse trajectory stability
- token reconstruction loss
- decoder-conditioned generation quality
- generation stability and diversity

## 6. 실패 조건

다음 결과가 나오면 LACE의 강한 주장은 약해진다.

- compression forward가 Gaussian noise나 masking보다 낫지 않다.
- 평균 풀링은 복원되지만 generation objective로 이어지지 않는다.
- reverse model이 latent perturbation에 거의 반응하지 않는다.
- learned compressor만 실패하는 것이 아니라 모든 compression forward가 corruption baseline보다 약하다.
- 성능 차이가 compression 때문이 아니라 reverse model capacity나 정보량 불일치 때문으로 설명된다.
- hidden-state reconstruction은 좋아도 token reconstruction 또는 text generation에서 이점이 없다.

이 경우 LACE는 "좋은 forward process"라기보다 "latent autoencoder reconstruction"에 가까운 결과로 해석될 수 있다.

## 7. 성공 조건

다음 결과가 나오면 LACE의 주장이 강해진다.

- 단순 평균 풀링 compression forward가 random corruption baseline보다 더 안정적으로 reverse expansion을 학습한다.
- compression 단계가 깊어질수록 정보 손실이 예측 가능하고 단조롭게 증가한다.
- reverse expansion이 낮은 정보율 latent에서 높은 정보율 표현을 점진적으로 회복한다.
- latent perturbation과 ablation에 모델이 의미 있게 반응한다.
- hidden-state reconstruction을 넘어 token reconstruction 또는 generation quality에서도 이점이 관찰된다.
- compression-based reverse trajectory가 실제 language generation 샘플링 경로로 사용 가능하다는 신호가 나온다.

가장 중요한 성공 신호는 "압축된 latent를 잘 복원했다"가 아니라 다음이다.

> 동일한 조건에서 정보 압축 forward process가 corruption forward process보다 generation-capable language diffusion reverse learning에 더 좋은 경로를 제공한다.

## 8. 논문에서의 표현 전략

논문에서는 다음처럼 방어적으로 표현하는 것이 좋다.

> We do not claim that downsampling or latent compression itself is new. Our claim is that language diffusion can replace random corruption with a structured information-rate schedule, turning the forward process into progressive compression and the reverse process into progressive expansion.

한국어로는 다음처럼 정리할 수 있다.

> LACE의 노벨티는 압축 연산 자체가 아니라, Diffusion Language Modeling의 forward process를 무작위 손상이 아닌 정보 압축 과정으로 재정의하고, 이를 동일 조건의 corruption baseline과 비교해 검증하는 데 있다.

이 관점을 유지해야 "downsampling autoencoder 아닌가?"라는 비판을 피할 수 있다.
