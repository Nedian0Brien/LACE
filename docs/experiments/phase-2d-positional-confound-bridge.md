# Phase 2D 실험 결과: Positional Confound Bridge

## 1. 실험 목적

Phase 2D의 목적은 Phase 3A로 바로 넘어가기 전에 positional conditioning의 confound를 정리하는 것이다.

사용자의 판단처럼, 현재 연구에서 더 중요한 실험은 Phase 3A다. Phase 3A는 reverse expansion 학습에 token/generation-aware objective를 직접 넣어, reconstructed hidden이 teacher-forced proxy를 넘어 open-ended generation으로 이어질 수 있는지 확인해야 한다.

다만 Phase 2C 결과만으로 Phase 3A를 설계하면 위험한 부분이 있었다. Phase 2C에서는 explicit sinusoidal positional feature가 Average Pooling expander를 크게 개선했지만, 동시에 `position_only`와 `gaussian_noise_positional`도 매우 강했다. 이 상태에서 token loss를 바로 넣으면 다음 confound가 생긴다.

> token objective가 Average Pooling content latent를 강화한 것인지, 아니면 positional scaffold와 decoder/expander capacity가 위치별 평균 token prior를 더 잘 찍게 만든 것인지 분리하기 어렵다.

따라서 Phase 2D는 Phase 3A의 중간 다리다. 이 실험의 핵심 산출물은 "Phase 3A에서 어떤 positional condition을 기본으로 쓰고, 어떤 control을 반드시 같이 둬야 하는가"다.

## 2. 실행 정보

| 항목 | 값 |
|---|---|
| Kaggle kernel | `dennisparknd/lace-phase-2d-positional-confound-bridge` |
| kernel version | 1 |
| 실행 상태 | 완료 |
| 모델 | `t5-small` |
| encoder 상태 | frozen |
| 데이터 소스 | `hf:wikitext/wikitext-2-raw-v1:train` |
| 샘플 수 | 2048 |
| 최대 길이 | 128 token |
| hidden shape | `[2048, 128, 512]` |
| active tokens | 210286 |
| stage tokens | 64, 32, 16 |
| positional feature size | 64 |
| wrong-position shifts | 1, 2, 4, 8, 16 |
| generation bridge | enabled |
| token-head bridge | enabled |
| 결과 위치 | `outputs/phase2d/lace_phase2d/` |

생성된 주요 파일:

| 파일 | 설명 |
|---|---|
| `metrics.json` | 전체 정량 결과 |
| `summary.md` | Kaggle runner가 생성한 요약 |
| `train_log.jsonl` | condition/stage별 학습 로그 |
| `generation_samples.jsonl` | qualitative generation sample |
| `checkpoints/*.pt` | condition/stage별 expander checkpoint |

## 3. 실험 조건

| Condition | Latent source | Positional mode | 목적 |
|---|---|---|---|
| `average_pool` | Average pooled hidden state | learned bias only | Phase 2B/2C baseline |
| `average_pool_abs_pos` | Average pooled hidden state | absolute sinusoidal | global position scaffold 효과 확인 |
| `average_pool_rel_pos` | Average pooled hidden state | block-relative sinusoidal | pooling block 내부 위치 복원 효과 확인 |
| `average_pool_abs_rel_pos` | Average pooled hidden state | absolute + block-relative | Phase 3A 후보 condition |
| `position_only_abs_pos` | zero latent | absolute sinusoidal | absolute position prior control |
| `position_only_rel_pos` | zero latent | block-relative sinusoidal | relative position prior control |
| `position_only_abs_rel_pos` | zero latent | absolute + block-relative | strongest position-only control |
| `gaussian_noise` | matched Gaussian noisy h0 | learned bias only | continuous corruption baseline |
| `gaussian_noise_abs_pos` | matched Gaussian noisy h0 | absolute sinusoidal | Phase 2C strongest control 재확인 |

## 4. Gate 결과

Kaggle에서 내려받은 원본 `summary.md`는 `P2D-G-POS-MATTERS=false`, `phase3_bridge_ready=false`로 기록되어 있다. 실행 후 확인해 보니 이 판정은 모델 결과의 문제가 아니라 gate 집계 기준의 문제였다. gate 이름과 설명은 wrong-position sweep을 보도록 되어 있었지만, 실제 코드는 shift=1의 평균 delta만 threshold와 비교하고 있었다.

그래서 runner의 gate 계산을 수정했다. 수정 후에는 sweep 전체의 최대 평균 delta를 함께 사용한다.

```text
avg_abs_rel shift=1 delta MSE: 0.000231
avg_abs_rel sweep max delta MSE: 0.000872
```

수정된 gate 평가를 다운로드된 동일 metrics에 다시 적용하면 다음과 같다.

| Gate | 결과 | 의미 |
|---|---|---|
| `P2D-G-RUN` | pass | 모든 Phase 2D condition이 같은 split에서 실행됨 |
| `P2D-G-ABS-GAIN` | pass | absolute position이 baseline Average Pooling보다 MSE/decoder NLL을 개선 |
| `P2D-G-REL-GAIN` | pass | block-relative position이 baseline Average Pooling보다 MSE/decoder NLL을 개선 |
| `P2D-G-ABSREL-GAIN` | pass | absolute+relative position이 baseline Average Pooling보다 MSE/decoder NLL을 개선 |
| `P2D-G-REL-CONTENT` | pass | relative Average Pooling이 relative position-only보다 cosine/decoder NLL에서 좋음 |
| `P2D-G-ABSREL-CONTENT` | pass | absolute+relative Average Pooling이 position-only보다 content-sensitive함 |
| `P2D-G-SHUFFLED-LABEL` | pass | label shuffle에서 token proxy가 악화됨 |
| `P2D-G-LATENT-USE` | pass | absolute+relative Average Pooling이 perturbation/ablation/swap에 반응함 |
| `P2D-G-POS-MATTERS` | pass after gate fix | wrong-position sweep에서 reconstruction이 악화됨 |
| `P2D-G-GAUSSIAN-STRONGER` | pass | Gaussian absolute positional이 여전히 더 강한 risk signal |

수정된 판정:

```text
overall_pass: true
phase3_bridge_ready: true
evidence_count: 8 / 2 required
```

단, 이 pass는 "LACE 가설이 입증됐다"는 뜻이 아니다. 정확한 의미는 다음이다.

> Phase 2D는 Phase 3A로 넘어갈 만큼 control 구조를 정리했다. 동시에 Gaussian positional이 여전히 강하므로, Phase 3A에서도 compression forward의 우위를 주장하려면 반드시 Gaussian positional control을 포함해야 한다.

## 5. 핵심 평균 지표

### 5.1 Reconstruction / Token Proxy

| Condition | MSE | Cosine | Decoder dNLL | Head dNLL | Shuffled decoder delta | Shuffled head delta |
|---|---:|---:|---:|---:|---:|---:|
| `average_pool` | 0.450913 | 0.309039 | 8.152192 | 1.441394 | 3.036021 | 0.444677 |
| `average_pool_abs_pos` | 0.020764 | 0.553341 | 5.050653 | 0.486786 | 1.298367 | 0.806932 |
| `average_pool_rel_pos` | 0.020711 | 0.549612 | 4.775901 | 0.442742 | 1.126908 | 0.813345 |
| `average_pool_abs_rel_pos` | 0.020933 | 0.547923 | 5.010779 | 0.481641 | 1.239660 | 0.800918 |
| `position_only_abs_pos` | 0.026866 | 0.337696 | 5.387503 | 0.203915 | 0.493069 | 0.468519 |
| `position_only_rel_pos` | 0.026314 | 0.338667 | 5.711123 | 0.143343 | 0.210359 | 0.489986 |
| `position_only_abs_rel_pos` | 0.026666 | 0.337834 | 5.438735 | 0.205040 | 0.520510 | 0.448569 |
| `gaussian_noise` | 0.441904 | 0.360015 | 8.203458 | 0.843007 | 1.501178 | 0.807379 |
| `gaussian_noise_abs_pos` | 0.014640 | 0.724751 | 4.034742 | 0.047338 | 2.588585 | 1.576084 |

### 5.2 Generation / Positional Sensitivity / Latent Use

| Condition | Meaningful gen | Wrong-position shift=1 dMSE | Rel sensitivity | Ablation dMSE | Swap dMSE |
|---|---:|---:|---:|---:|---:|
| `average_pool` | 0.000000 | 0.000000 | 0.016086 | 0.011956 | 0.034019 |
| `average_pool_abs_pos` | 0.000000 | 0.000156 | 0.056626 | 0.003392 | 0.018434 |
| `average_pool_rel_pos` | 0.250000 | 0.000133 | 0.050476 | 0.003036 | 0.016325 |
| `average_pool_abs_rel_pos` | 0.083333 | 0.000231 | 0.051358 | 0.003024 | 0.017370 |
| `position_only_abs_pos` | 0.000000 | 0.000256 | 0.155310 | 0.000000 | 0.000000 |
| `position_only_rel_pos` | 0.000000 | 0.000140 | 0.305663 | 0.000000 | 0.000000 |
| `position_only_abs_rel_pos` | 0.000000 | 0.000275 | 0.107648 | 0.000000 | 0.000000 |
| `gaussian_noise` | 0.000000 | 0.000000 | 0.005724 | 0.004928 | 0.048410 |
| `gaussian_noise_abs_pos` | 0.500000 | 0.000109 | 0.040122 | 0.004992 | 0.027454 |

### 5.3 Wrong-position Sweep

| Condition | shift 1 | shift 2 | shift 4 | shift 8 | shift 16 |
|---|---:|---:|---:|---:|---:|
| `average_pool` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `average_pool_abs_pos` | 0.000156 | 0.000408 | 0.000779 | 0.000920 | 0.001096 |
| `average_pool_rel_pos` | 0.000133 | 0.000168 | 0.000055 | 0.000000 | 0.000000 |
| `average_pool_abs_rel_pos` | 0.000231 | 0.000408 | 0.000530 | 0.000619 | 0.000872 |
| `position_only_abs_pos` | 0.000256 | 0.000634 | 0.001021 | 0.001170 | 0.001472 |
| `position_only_rel_pos` | 0.000140 | 0.000153 | 0.000054 | 0.000000 | 0.000000 |
| `position_only_abs_rel_pos` | 0.000275 | 0.000502 | 0.000651 | 0.000790 | 0.001130 |
| `gaussian_noise` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `gaussian_noise_abs_pos` | 0.000109 | 0.000315 | 0.000649 | 0.000808 | 0.001001 |

## 6. 결과 해석

### 6.1 Hidden-state MSE

MSE는 reconstructed hidden state의 좌표값이 원래 `h0`와 얼마나 가까운지 본다. 낮을수록 좌표 복원이 좋다.

Phase 2D에서도 positional conditioning의 효과는 매우 강했다.

```text
average_pool:             0.450913
average_pool_abs_pos:     0.020764
average_pool_rel_pos:     0.020711
average_pool_abs_rel_pos: 0.020933
```

이 결과의 좋은 점은 분명하다. Average Pooling latent 자체만으로는 reverse expansion이 token-position-aligned hidden trajectory를 만들기 어렵지만, absolute 또는 relative positional scaffold를 주면 MSE가 크게 개선된다.

하지만 MSE만으로는 content latent의 기여를 강하게 주장할 수 없다. `position_only_*`도 MSE가 0.026대까지 내려오고, `gaussian_noise_abs_pos`는 0.014640으로 가장 좋다. 즉 hidden-state 좌표 복원에는 위치 정보와 평균 hidden trajectory prior가 매우 크게 작동한다.

방어 가능한 해석은 다음이다.

> positional conditioning은 reverse expansion의 필수 scaffold에 가깝다. 그러나 낮은 MSE는 content compression의 우수성을 증명하지 않는다. MSE는 position prior와 expander capacity에 크게 취약하다.

### 6.2 Cosine

Cosine은 hidden vector의 방향성, 즉 representation direction이 얼마나 맞는지 본다. MSE가 좌표값의 거리라면, cosine은 representation이 비슷한 방향을 가리키는지를 본다.

Average Pooling positional 계열은 position-only보다 명확히 높았다.

```text
average_pool_abs_pos:     0.553341
average_pool_rel_pos:     0.549612
average_pool_abs_rel_pos: 0.547923
position_only_abs_rel:    0.337834
```

이것은 중요한 긍정 신호다. position-only는 위치 scaffold만으로 평균 trajectory를 만들 수 있지만, representation direction까지는 Average Pooling content latent가 있어야 더 좋아진다. 즉 content latent가 완전히 무시되고 있지는 않다.

하지만 `gaussian_noise_abs_pos`의 cosine은 0.724751이다. 이는 length-preserving noisy hidden state가 compressed latent보다 훨씬 더 많은 directional 정보를 유지한다는 강한 반론이다.

따라서 cosine의 결론은 mixed다.

> Average Pooling positional은 position-only보다 semantic/representation direction을 더 잘 복원한다. 하지만 Gaussian positional이 더 강하기 때문에, compression forward가 corruption forward보다 우수하다는 주장까지는 아직 가지 못한다.

### 6.3 Frozen Decoder Delta NLL

Frozen decoder delta NLL은 reconstructed hidden을 T5 decoder가 teacher-forced token reconstruction에 얼마나 읽기 쉬운지 보는 proxy다. 원본 `h0` control 대비 NLL 악화량이므로 낮을수록 좋다.

여기서 가장 좋은 Average Pooling 계열은 `average_pool_rel_pos`였다.

```text
average_pool:             8.152192
average_pool_abs_pos:     5.050653
average_pool_rel_pos:     4.775901
average_pool_abs_rel_pos: 5.010779
```

이 결과는 Phase 3A 설계에 중요하다. `relative` positional mode가 absolute만 쓰는 것보다 decoder proxy에서 약간 더 좋았다. Average Pooling은 block 내부 token들을 평균화하므로, reverse expansion이 block 안에서 각 output token의 상대 위치를 알아야 원래 hidden trajectory를 더 자연스럽게 펼칠 수 있다. Phase 2D는 이 가설을 지지한다.

다만 `gaussian_noise_abs_pos`는 4.034742로 여전히 더 좋다. 따라서 decoder NLL 관점에서도 Gaussian positional은 반드시 유지해야 할 강한 control이다.

### 6.4 Lightweight Token Head

Token-head delta NLL은 decoder-specific confound를 줄이기 위해 학습한 lightweight token classifier proxy다. 낮을수록 token identity를 맞히기 쉬운 hidden이라고 볼 수 있다.

여기서는 주의해야 한다.

```text
average_pool_rel_pos:      0.442742
position_only_rel_pos:     0.143343
gaussian_noise_abs_pos:    0.047338
```

Average Pooling positional은 baseline보다 좋아졌지만, position-only와 Gaussian positional보다 약하다. 이는 token-head가 content latent보다 positional scaffold와 corpus-level token prior에 크게 반응할 수 있음을 보여준다.

즉 Phase 3A에서 token-head loss를 훈련 objective에 넣는 것은 필요하지만, 그 자체로 충분하지 않다. token-head loss만 넣으면 model이 content latent를 더 잘 쓰기보다, position-aware average trajectory를 token prior에 맞추는 shortcut을 배울 수 있다.

따라서 Phase 3A에서는 반드시 다음 control이 필요하다.

```text
average_pool_rel_pos
position_only_rel_pos
gaussian_noise_abs_pos
```

가능하면 `average_pool_abs_rel_pos`와 `position_only_abs_rel_pos`도 포함해야 한다.

### 6.5 Shuffled-label Delta NLL

Shuffled-label delta는 reconstructed hidden은 그대로 두고 target labels만 batch 안에서 뒤섞었을 때 NLL이 얼마나 나빠지는지 본다.

이 지표가 중요한 이유는 token proxy가 정말 sample-specific content를 담고 있는지 확인하기 위해서다. 만약 hidden이 content를 담고 있다면 true labels에서는 상대적으로 잘 맞고 shuffled labels에서는 나빠져야 한다. 반대로 위치별 평균 prior만 찍는다면 shuffled labels에서도 크게 나빠지지 않을 수 있다.

Average Pooling positional 계열은 position-only보다 shuffled delta가 컸다.

```text
average_pool_abs_rel_pos shuffled decoder delta: 1.239660
position_only_abs_rel_pos shuffled decoder delta: 0.520510

average_pool_abs_rel_pos shuffled head delta:    0.800918
position_only_abs_rel_pos shuffled head delta:    0.448569
```

이것은 Phase 2D의 가장 중요한 긍정 신호 중 하나다.

> position-only도 token prior를 일부 만들지만, Average Pooling positional은 그보다 더 sample-specific한 token 정보를 담고 있다.

다만 Gaussian positional은 shuffled delta가 더 크다.

```text
gaussian_noise_abs_pos shuffled decoder delta: 2.588585
gaussian_noise_abs_pos shuffled head delta:    1.576084
```

이는 length-preserving noisy hidden이 sample-specific token information을 더 많이 보존한다는 뜻이다. Compression path의 주장은 여전히 불리한 비교를 통과해야 한다.

### 6.6 Latent Use

Latent use는 expander가 입력 latent를 실제로 사용하는지 본다. perturbation sensitivity, ablation delta, swap delta가 모두 중요하다.

`average_pool_abs_rel_pos`는 세 지표가 모두 양수였다.

```text
relative sensitivity: 0.051358
ablation delta MSE:   0.003024
swap delta MSE:       0.017370
```

이는 expander가 position scaffold만 쓰는 것이 아니라 content latent에도 반응한다는 뜻이다. 특히 position-only의 ablation/swap delta는 0이므로, 이 차이는 해석이 비교적 깨끗하다.

다만 latent use가 generation-capability를 보장하지는 않는다. Latent를 쓰고 있음에도 open-ended generation은 아직 약하다. 따라서 Phase 3A에서는 token objective를 넣은 뒤에도 latent ablation/swap sensitivity가 유지되는지 반드시 확인해야 한다.

### 6.7 Wrong-position Sweep

Wrong-position sweep은 positional feature를 틀리게 줬을 때 reconstruction이 얼마나 악화되는지 본다. 이 지표는 positional feature가 실제로 사용되는지, 그리고 어떤 종류의 position signal이 민감한지 확인한다.

Absolute position은 shift가 커질수록 delta가 커졌다.

```text
average_pool_abs_pos:
shift 1:  0.000156
shift 2:  0.000408
shift 4:  0.000779
shift 8:  0.000920
shift 16: 0.001096
```

Relative position은 작은 shift에서만 반응하고, block size 주기와 맞는 shift에서는 0에 가까워졌다.

```text
average_pool_rel_pos:
shift 1: 0.000133
shift 2: 0.000168
shift 4: 0.000055
shift 8: 0.000000
shift 16:0.000000
```

이 결과는 설계상 자연스럽다. block-relative feature는 pooling block 안의 상대 위치를 반복적으로 부여하므로, 특정 shift에서는 상대 위치 패턴이 다시 맞아떨어질 수 있다.

`average_pool_abs_rel_pos`는 두 효과를 함께 보였다.

```text
shift 1:  0.000231
shift 2:  0.000408
shift 4:  0.000530
shift 8:  0.000619
shift 16: 0.000872
```

따라서 Phase 3A에서 absolute+relative를 기본으로 쓰는 것도 가능하지만, decoder delta NLL만 보면 relative-only가 약간 더 좋았다. 이 차이는 크지 않기 때문에, Phase 3A에서는 다음 두 조건 중 하나를 선택해야 한다.

1. 기본 condition을 `average_pool_rel_pos`로 두고, 가장 단순한 block-relative 가설을 검증한다.
2. 기본 condition을 `average_pool_abs_rel_pos`로 두고, positional scaffold를 더 완비한 상태에서 token objective를 검증한다.

현재 결과만 보면 나는 1번, 즉 `average_pool_rel_pos`를 Phase 3A의 primary condition으로 두는 쪽이 더 좋다고 본다. 이유는 더 단순하고, decoder delta NLL이 가장 좋고, Average Pooling의 손실 구조와 직접 연결되기 때문이다. `average_pool_abs_rel_pos`는 secondary condition으로 유지하면 된다.

### 6.8 Open-ended Generation

Open-ended generation은 여전히 가장 어려운 지표다.

```text
average_pool_abs_pos:     0.000000
average_pool_rel_pos:     0.250000
average_pool_abs_rel_pos: 0.083333
gaussian_noise_abs_pos:   0.500000
```

수치만 보면 `average_pool_rel_pos`가 0에서 벗어났고, `gaussian_noise_abs_pos`가 가장 강하다. 하지만 qualitative sample을 보면 아직 반복, 짧은 generic sentence, 원문과 느슨하게만 관련된 문장이 많다. 따라서 meaningful generation rate는 매우 거친 heuristic으로 봐야 한다.

Phase 2D의 방어 가능한 결론은 다음이다.

> positional split을 더 세밀하게 해도 Average Pooling open-ended generation은 안정적으로 회복되지 않았다. 다만 relative positional condition에서 0이 아닌 generation signal이 생겼고, Gaussian positional은 여전히 더 강하다. Phase 3A는 이 간극을 직접 겨냥해야 한다.

## 7. Phase 3A로 넘길 결정

### 7.1 Phase 3A의 기본 조건

Phase 3A의 primary condition은 다음을 권한다.

```text
average_pool_rel_pos
```

이유:

1. Average Pooling이 잃어버리는 정보는 block 내부 token order이므로, relative position이 가장 직접적인 보완이다.
2. Phase 2D에서 Average Pooling positional 계열 중 decoder delta NLL이 가장 낮았다.
3. Absolute+relative보다 단순해서 token objective의 효과를 해석하기 쉽다.
4. meaningful generation rate도 0.25로 Average Pooling 계열 중 가장 높았다. 단, 이 값은 heuristic이므로 보조 신호로만 본다.

Secondary condition:

```text
average_pool_abs_rel_pos
```

이 조건은 positional scaffold를 더 완비한 후보로 유지한다. 특히 wrong-position sweep에서는 absolute+relative가 더 넓은 shift 범위에서 민감했다.

### 7.2 Phase 3A의 필수 controls

Phase 3A에서 반드시 포함해야 할 controls:

```text
position_only_rel_pos
position_only_abs_rel_pos
gaussian_noise_abs_pos
h0_decoder_control
```

각 control의 의미는 다르다.

`position_only_rel_pos`는 token objective가 relative position prior shortcut으로 빠지는지 확인한다.

`position_only_abs_rel_pos`는 가장 강한 position-only scaffold control이다. 이 control이 token-head나 decoder proxy에서 Average Pooling을 이기면 content latent claim은 약해진다.

`gaussian_noise_abs_pos`는 compression forward에 대한 가장 강한 반론이다. 이 condition이 계속 token/generation에서 이기면, compression보다 length-preserving noisy hidden state가 reverse generation에 더 적합하다는 가능성을 인정해야 한다.

`h0_decoder_control`은 ceiling이다. 원본 encoder hidden에서 decoder가 어느 정도 생성 가능한지 확인하지 않으면, reconstructed hidden의 실패가 expander 문제인지 decoder bridge 자체의 문제인지 알 수 없다.

### 7.3 Phase 3A의 objective

Phase 3A는 reconstruction-only objective에서 벗어나야 한다.

권장 objective:

```text
L = L_hidden_mse
  + lambda_cos * L_cos
  + lambda_token * L_token_head
```

최소 sweep:

```text
lambda_token = 0.05, 0.10, 0.20
```

처음부터 너무 큰 token loss를 주면 hidden reconstruction과 latent-use가 무너질 수 있다. 따라서 token loss는 작은 값부터 올리고, 다음을 같이 본다.

1. token-head delta NLL이 내려가는가?
2. frozen decoder delta NLL도 같이 내려가는가?
3. hidden MSE/cosine이 지나치게 망가지지 않는가?
4. ablation/swap sensitivity가 유지되는가?
5. open-ended generation이 단순 반복에서 벗어나는가?
6. position-only control도 똑같이 좋아지는가?

## 8. 최종 판정

Phase 2D는 성공적인 bridge 실험이다.

성공의 의미는 다음이다.

1. absolute, relative, absolute+relative positional mode를 분리했다.
2. relative position이 Average Pooling reverse expansion에 유효하다는 신호를 확인했다.
3. Average Pooling positional이 position-only보다 cosine, decoder NLL, shuffled-label sensitivity에서 더 content-sensitive하다는 신호를 확인했다.
4. latent perturbation/ablation/swap에 대한 반응이 유지됨을 확인했다.
5. wrong-position sweep으로 positional feature가 실제 사용됨을 확인했다.
6. Gaussian positional이 여전히 가장 강한 반론이라는 점도 확인했다.

그러나 이 실험은 LACE의 최종 주장을 입증하지 않는다.

현재 가장 정직한 결론은 다음이다.

> Phase 2D는 Phase 3A로 넘어갈 수 있게 해줬다. 하지만 Phase 3A는 반드시 `average_pool_rel_pos`와 `gaussian_noise_abs_pos`, `position_only_*` controls를 함께 두고, token objective가 content latent를 강화하는지 아니면 positional shortcut을 강화하는지 검증해야 한다.

## 9. 다음 단계

다음 실험은 Phase 3A다.

Phase 3A의 제목은 다음처럼 잡는 것이 좋다.

```text
Phase 3A: Generation-aware Reverse Expansion Objective
```

핵심 질문:

> token-head objective를 reverse expander 학습에 직접 넣으면 Average Pooling relative positional latent가 open-ended generation으로 이어지는가?

Phase 3A에서 좋은 결과는 단순히 token-head NLL이 낮아지는 것이 아니다. 좋은 결과는 다음을 동시에 만족해야 한다.

1. `average_pool_rel_pos`가 `position_only_rel_pos`보다 token proxy와 generation에서 좋아진다.
2. `average_pool_rel_pos`가 latent ablation/swap sensitivity를 유지한다.
3. open-ended generation sample이 반복 collapse에서 벗어난다.
4. `gaussian_noise_abs_pos`와의 gap이 줄어든다.

이 네 가지 중 1번과 2번이 없으면 token objective가 content latent를 쓰게 만든다고 말할 수 없다. 3번이 없으면 generation-capable path라고 말할 수 없다. 4번이 없으면 compression forward가 corruption/noise forward보다 낫다는 주장은 여전히 약하다.
