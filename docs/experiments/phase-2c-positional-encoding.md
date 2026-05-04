# Phase 2C 실험 결과: Adding Positional Encoding

## 1. 실험 목적

Phase 2C의 목적은 Average Pooling compression이 open-ended generation에서 붕괴한 이유 중 하나가 위치/순서 정보 부족인지 검증하는 것이다.

Phase 2B에서는 Average Pooling이 hidden-state MSE와 cosine에서 강하게 이기지 못했지만, frozen decoder delta token NLL과 latent-use 지표에서는 살아 있는 신호를 보였다. 이 조합은 다음 가능성을 남겼다.

> Average Pooling latent에는 content signal이 일부 남아 있지만, reverse expansion이 token-position-aligned hidden trajectory를 만들 위치 좌표계가 부족할 수 있다.

Phase 2C는 기존 expander의 learned position bias는 유지하되, Transformer식 fixed sinusoidal positional feature를 expander 입력에 명시적으로 추가했다. 따라서 이 실험은 "position 정보가 아예 없었는가"가 아니라, "learned bias만으로 부족했던 위치 scaffold를 explicit positional encoding이 보완하는가"를 보는 실험이다.

## 2. 실행 정보

| 항목 | 값 |
|---|---|
| Kaggle kernel | `dennisparknd/lace-phase-2c-positional-encoding` |
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
| wrong-position shift | 17 |
| generation bridge | enabled |
| token-head bridge | enabled |
| 결과 위치 | `outputs/phase2c/lace_phase2c/` |

생성된 주요 파일:

| 파일 | 설명 |
|---|---|
| `metrics.json` | 전체 정량 결과 |
| `summary.md` | Kaggle runner가 생성한 요약 |
| `train_log.jsonl` | condition/stage별 학습 로그 |
| `generation_samples.jsonl` | qualitative generation sample |
| `checkpoints/*.pt` | condition/stage별 expander checkpoint |

## 3. 실험 조건

| Condition | Latent source | Positional feature | 목적 |
|---|---|---|---|
| `average_pool` | Average pooled hidden state | learned bias only | Phase 2B baseline |
| `average_pool_positional` | Average pooled hidden state | fixed sinusoidal + learned bias | Average Pooling이 위치 scaffold로 개선되는지 확인 |
| `position_only` | zero latent with Average Pooling shape | fixed sinusoidal + learned bias | 개선이 content 없이 position prior만으로 가능한지 확인 |
| `random_select` | random selected hidden states | learned bias only | token-budget corruption baseline |
| `gaussian_noise` | matched Gaussian noisy h0 | learned bias only | continuous corruption baseline |
| `gaussian_noise_positional` | matched Gaussian noisy h0 | fixed sinusoidal + learned bias | positional feature가 모든 reverse expansion에 주는 일반 이득 확인 |

## 4. Gaussian Calibration

Phase 2C는 Phase 2B와 같은 matched Gaussian calibration을 사용했다.

| Stage | average pooling initial loss | matched sigma | sigma initial loss |
|---|---:|---:|---:|
| `z1` | 0.008145 | 0.10 | 0.009999 |
| `z2` | 0.012619 | 0.10 | 0.009999 |
| `z3` | 0.015713 | 0.125 | 0.015627 |

실제 Gaussian condition:

```text
z1: 0.10
z2: 0.10
z3: 0.125
```

이 점은 중요하다. Positional Encoding의 효과가 Gaussian 난도 과대 설정 때문인지 줄이기 위해, Phase 2B와 같은 calibrated Gaussian을 유지했다.

## 5. Gate 결과

Phase 2C의 strict overall pass는 실패했다.

또한 Phase 3 candidate도 false로 두었다.

| Gate | 결과 | 의미 |
|---|---|---|
| `P2C-G-RUN` | pass | baseline, positional, position-only, random, Gaussian controls가 같은 split에서 실행됨 |
| `P2C-G-POS-MSE` | pass | `average_pool_positional`이 `average_pool`보다 MSE를 크게 낮춤 |
| `P2C-G-POS-COS` | pass | `average_pool_positional`이 `average_pool`보다 cosine을 높임 |
| `P2C-G-POS-DECODER-NLL` | pass | `average_pool_positional`이 frozen decoder delta NLL을 낮춤 |
| `P2C-G-POS-TOKEN-HEAD-NLL` | pass | `average_pool_positional`이 `average_pool`보다 token-head delta NLL을 낮춤 |
| `P2C-G-POS-USE` | pass | positional average-pool expander도 latent perturbation/ablation/swap에 반응함 |
| `P2C-G-POS-CONTROL` | fail | `position_only`가 token-head delta NLL에서 더 좋아, 모든 개선이 content latent 때문이라고 말할 수 없음 |
| `P2C-G-POS-MATTERS` | pass | wrong-position evaluation이 reconstruction을 악화시켜 positional signal을 실제 사용함 |
| `P2C-G-GEN` | fail | `average_pool_positional`의 meaningful generation rate는 0.0으로 유지됨 |

정량적으로는 evidence count가 6/2로 높다. 하지만 `P2C-G-POS-CONTROL`이 실패했기 때문에 Phase 3 candidate는 false로 두었다.

이 판정은 보수적이다. Positional Encoding이 Average Pooling baseline을 크게 개선한 것은 사실이지만, position-only와 Gaussian positional control이 너무 강해서 "Average Pooling compression의 고유한 장점"이라고 바로 주장하기 어렵기 때문이다.

## 6. 핵심 평균 지표

| 지표 | average_pool | average_pool_positional | position_only | gaussian_noise | gaussian_noise_positional |
|---|---:|---:|---:|---:|---:|
| 평균 MSE | 0.450913 | 0.020764 | 0.026878 | 0.438427 | 0.014739 |
| 평균 cosine | 0.309039 | 0.553341 | 0.336757 | 0.360946 | 0.722765 |
| 평균 decoder delta NLL | 8.152192 | 5.050653 | 5.295180 | 8.717209 | 4.044799 |
| 평균 token-head delta NLL | 1.441394 | 0.486786 | 0.176166 | 0.862452 | 0.049600 |
| 평균 meaningful generation rate | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.333333 |
| 평균 wrong-position delta MSE | 0.000000 | 0.001067 | 0.001470 | 0.000000 | 0.000974 |

## 7. 결과 해석

### 7.1 Hidden-state MSE

MSE에서는 positional conditioning이 매우 크게 개선됐다.

```text
average_pool:             0.450913
average_pool_positional:  0.020764
position_only:            0.026878
gaussian_noise:           0.438427
gaussian_noise_positional:0.014739
```

이 지표는 reconstructed hidden state가 원본 `h0`의 좌표값을 얼마나 잘 복원하는지 본다.

좋은 점은 `average_pool_positional`이 `average_pool`보다 압도적으로 낮은 MSE를 보였다는 것이다. 이는 위치 scaffold가 reverse expansion의 좌표 복원을 크게 안정화한다는 뜻이다.

하지만 caveat가 크다. `position_only`도 0.026878로 매우 낮다. 즉 fixed sinusoidal position과 learned bias만으로도 validation hidden trajectory의 많은 평균 구조를 재현할 수 있다. 또한 `gaussian_noise_positional`은 0.014739로 가장 좋았다.

따라서 MSE 결과의 방어 가능한 해석은 다음이다.

> Explicit positional conditioning은 reverse expansion을 매우 강하게 안정화한다. 그러나 이 개선은 Average Pooling content latent만의 효과가 아니라, position scaffold 자체와 expander capacity가 만드는 평균 hidden trajectory 복원 효과를 크게 포함한다.

### 7.2 Cosine

Cosine도 positional conditioning으로 개선됐다.

```text
average_pool:             0.309039
average_pool_positional:  0.553341
position_only:            0.336757
gaussian_noise:           0.360946
gaussian_noise_positional:0.722765
```

Cosine은 hidden vector의 방향성, 즉 representation direction을 본다. Average Pooling positional은 baseline보다 좋아졌고, position-only보다도 높다. 이 점은 content latent가 방향성 복원에 일부 기여한다는 신호다.

하지만 Gaussian positional이 훨씬 높다. 이는 positional scaffold가 주어졌을 때, compressed latent보다 length-preserving noisy hidden state가 representation direction을 더 잘 회복한다는 뜻이다.

따라서 cosine 결과는 LACE에 대해 mixed다.

> Position-aware expansion은 Average Pooling의 방향성 복원을 개선하지만, continuous noisy h0 + position scaffold가 더 강하다. 이 결과만으로 compression forward process 우위를 주장할 수는 없다.

### 7.3 Frozen Decoder Delta NLL

Frozen decoder delta NLL도 Average Pooling positional에서 크게 개선됐다.

```text
average_pool:             8.152192
average_pool_positional:  5.050653
position_only:            5.295180
gaussian_noise:           8.717209
gaussian_noise_positional:4.044799
```

이 지표는 reconstructed hidden을 frozen T5 decoder에 넣었을 때 원본 `h0` 대비 token NLL이 얼마나 악화되는지 측정한다. 낮을수록 decoder-readable하다.

`average_pool_positional`이 `average_pool`보다 크게 좋아진 것은 중요하다. Phase 2B에서 보였던 decoder-readability 병목이 단순 content 손실만은 아니고, position-aware hidden trajectory 구성 문제였을 가능성을 지지한다.

또한 `average_pool_positional`은 `position_only`보다 약간 좋다. 이 점은 frozen decoder 관점에서는 content latent가 약간이라도 도움이 된다는 신호다.

하지만 다시 Gaussian positional이 가장 좋다. 즉 positional scaffold가 생기면 compression latent보다 length-preserving noisy hidden을 복원하는 쪽이 decoder에 더 잘 맞는다.

### 7.4 Lightweight Token Head

Token-head delta NLL은 가장 조심스럽게 해석해야 한다.

```text
average_pool:             1.441394
average_pool_positional:  0.486786
position_only:            0.176166
gaussian_noise:           0.862452
gaussian_noise_positional:0.049600
```

Average Pooling positional은 baseline보다 좋아졌다. 하지만 position-only가 더 좋다. 이는 token-head proxy가 content latent보다 position-conditioned 평균 hidden trajectory에 크게 반응했을 수 있음을 의미한다.

이 결과는 `P2C-G-POS-CONTROL`을 실패시킨 핵심 이유다.

방어 가능한 해석은 다음이다.

> Token-head improvement는 Average Pooling content가 token information을 더 잘 보존했기 때문이라고 보기 어렵다. 적어도 현재 token-head proxy에서는 position prior와 decoder/expander capacity만으로도 상당한 token predictability가 생긴다.

따라서 token-head 결과는 Average Pooling 주장을 강화하기보다, 오히려 positional conditioning의 confound를 보여준다.

### 7.5 Latent Use

Average Pooling positional의 latent-use 지표는 양수였다.

```text
relative sensitivity: 0.056626
ablation delta:       0.003392
swap delta:           0.018434
```

이 지표들은 expander가 latent를 실제로 사용하는지 본다. perturbation, ablation, swap에 reconstruction이 반응하면 latent를 무시하지 않는다는 신호다.

Position-only의 ablation/swap delta는 0이다. 이는 당연하지만 중요하다. position-only는 content latent가 없기 때문에 ablation/swap으로 sample-specific content가 바뀌지 않는다.

따라서 latent-use에 대해서는 긍정적으로 볼 수 있다.

> Average Pooling positional expander는 position scaffold만 쓰는 것이 아니라 content latent에도 반응한다. 다만 그 content contribution이 token-head나 generation까지 충분히 강하게 이어지지는 않았다.

### 7.6 Wrong-position Diagnostic

Wrong-position delta MSE는 Average Pooling positional에서 양수였다.

```text
average_pool_positional wrong-position delta MSE: 0.001067
average_pool_positional wrong-position delta cosine: -0.022538
```

이 지표는 positional feature를 일정량 shift해서 틀린 위치 정보를 주었을 때 reconstruction이 나빠지는지 본다.

결과는 positional feature가 실제로 사용됐음을 보여준다. wrong-position shift가 MSE를 높이고 cosine을 낮췄기 때문이다.

다만 delta 크기는 절대적으로 아주 크지는 않다. 즉 position 정보가 쓰이긴 하지만, 전체 개선의 대부분이 "position feature가 있을 때 학습이 쉬워진다"는 구조적 효과인지, 각 token position의 정밀한 alignment 효과인지는 추가 ablation이 필요하다.

### 7.7 Open-ended Generation

Open-ended generation은 Average Pooling positional에서도 살아나지 않았다.

```text
h0 control meaningful rate:               0.75
average_pool meaningful rate:             0.00
average_pool_positional meaningful rate:  0.00
position_only meaningful rate:            0.00
gaussian_noise meaningful rate:           0.00
gaussian_noise_positional meaningful rate:0.333333
```

이 결과는 중요하다.

Average Pooling positional은 MSE, cosine, decoder NLL, token-head NLL을 모두 크게 개선했지만 meaningful generation은 0이다. 따라서 teacher-forced proxy와 open-ended generation 사이의 간극이 여전히 남아 있다.

반면 Gaussian positional은 meaningful generation rate가 0.333333까지 올라갔다. 이 샘플들도 완전히 안정적이진 않고 반복/짧은 일반 문장이 섞여 있지만, Average Pooling positional보다 generation behavior가 낫다.

따라서 현재 generation 관점의 해석은 다음이다.

> Positional Encoding은 reverse expansion proxy를 크게 개선하지만, Average Pooling compressed latent만으로 open-ended generation을 회복시키지는 못했다. Generation 회복에는 position scaffold뿐 아니라 더 직접적인 generation-aware objective 또는 더 정보 보존적인 latent가 필요해 보인다.

## 8. 최종 판정

| 항목 | 판정 |
|---|---|
| 실행 완결성 | 성공 |
| positional branch 구현 | 성공 |
| position-only control | 성공 |
| wrong-position diagnostic | 성공 |
| Average Pooling baseline 대비 MSE 개선 | 성공 |
| Average Pooling baseline 대비 cosine 개선 | 성공 |
| Average Pooling baseline 대비 decoder NLL 개선 | 성공 |
| Average Pooling baseline 대비 token-head NLL 개선 | 성공 |
| content latent 필요성 입증 | 부분 성공 |
| Average Pooling positional generation 회복 | 실패 |
| compression forward process 우위 | 미입증 |
| strict overall pass | 실패 |
| Phase 3 candidate | 보류 |

한 줄 결론:

> Positional Encoding은 reverse expansion을 강하게 안정화하지만, 현재 결과는 "Average Pooling compression의 고유한 우위"라기보다 "position-aware expander가 hidden trajectory를 훨씬 쉽게 복원한다"는 증거에 가깝다.

## 9. 다음 단계

Phase 2C 이후 바로 Learned Attention Compression으로 넘어가기보다는, 먼저 positional confound를 더 분리하는 작은 후속 실험이 필요하다.

권고는 **Phase 2D: Positional Objective/Ablation Cleanup**이다.

핵심은 다음 세 가지다.

1. `position_only`가 token-head에서 너무 강한 이유를 분리한다.
2. Average Pooling positional이 content latent를 얼마나 추가로 쓰는지 더 직접적으로 측정한다.
3. Gaussian positional이 generation에서 좋아진 이유가 length-preserving latent 때문인지 확인한다.

구체적인 다음 실험 후보:

| 후보 | 목적 |
|---|---|
| `frozen sinusoidal only` vs `learned bias only` vs `sinusoidal + learned bias` | positional scaffold의 어떤 부분이 효과적인지 분리 |
| `position_only` with shuffled labels/texts | token-head가 content 없이 위치별 token prior를 외우는지 확인 |
| `average_pool_positional + token loss in expander training` | proxy 개선을 generation objective로 연결할 수 있는지 확인 |
| `gaussian_noise_positional` generation audit | meaningful rate 0.333이 실제 개선인지 heuristic artifact인지 확인 |
| `block-relative positional encoding` | absolute position scaffold보다 compression block 내부 상대 위치가 더 중요한지 확인 |

현재 가장 중요한 결론은 이것이다.

> Positional Encoding은 LACE에서 반드시 다뤄야 할 요소다. 하지만 그 자체가 Average Pooling compression을 정당화하지는 않는다. 다음 실험은 position prior와 content latent contribution을 더 날카롭게 분리해야 한다.
