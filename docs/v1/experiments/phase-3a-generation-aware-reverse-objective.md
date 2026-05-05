# Phase 3A 실험 결과: Generation-aware Reverse Objective

## 1. 실험 목적

Phase 3A의 목적은 reverse expander 학습에 token-level objective를 직접 넣었을 때, Average Pooling latent가 generation-readable hidden state로 이어지는지 확인하는 것이다.

Phase 2B, 2C, 2D에서 반복적으로 확인한 병목은 다음이었다.

> hidden reconstruction proxy와 teacher-forced token proxy는 일부 좋아져도, reconstructed `h0_hat` 기반 open-ended generation은 안정적으로 살아나지 않는다.

따라서 Phase 3A에서는 lightweight token head를 단순 평가 지표로만 쓰지 않고, frozen token-head loss를 expander 학습 loss에 직접 추가했다.

```text
L = L_hidden_mse
  + lambda_cos * L_cos
  + lambda_var * L_var
  + lambda_token * L_token_head
```

중요한 설계 선택은 token head를 먼저 원본 `h0`에서 학습한 뒤 freeze했다는 점이다. 그래야 token-head NLL 개선이 classifier adaptation 때문이 아니라 expander output 변화 때문이라고 해석할 수 있다.

## 2. 실행 정보

| 항목 | 값 |
|---|---|
| Kaggle kernel | `dennisparknd/lace-phase-3a-generation-aware-reverse-objective` |
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
| generation bridge | enabled |
| token-head bridge | enabled |
| token-head parameter count | 16,468,324 |
| 결과 위치 | `outputs/phase3a/lace_phase3a/` |
| output size | 약 407MB |

생성된 주요 파일:

| 파일 | 설명 |
|---|---|
| `metrics.json` | 전체 정량 결과 |
| `summary.md` | Kaggle runner가 생성한 요약 |
| `train_log.jsonl` | arm/stage별 학습 로그 |
| `generation_samples.jsonl` | qualitative generation sample |
| `checkpoints/*.pt` | arm/stage별 expander checkpoint |

## 3. 실험 arm

| Arm | Latent source | Positional mode | `lambda_token` | 목적 |
|---|---|---|---:|---|
| `average_pool_rel_pos_recon` | Average Pooling | block-relative | 0.00 | Phase 2D primary baseline 재현 |
| `average_pool_rel_pos_tok005` | Average Pooling | block-relative | 0.05 | 약한 token objective |
| `average_pool_rel_pos_tok010` | Average Pooling | block-relative | 0.10 | 기본 token objective 후보 |
| `average_pool_rel_pos_tok020` | Average Pooling | block-relative | 0.20 | 강한 token objective |
| `average_pool_abs_rel_pos_tok010` | Average Pooling | absolute+relative | 0.10 | 더 강한 positional scaffold 후보 |
| `position_only_rel_pos_tok010` | zero latent | block-relative | 0.10 | relative position shortcut control |
| `position_only_abs_rel_pos_tok010` | zero latent | absolute+relative | 0.10 | strongest position-only shortcut control |
| `gaussian_noise_abs_pos_recon` | matched Gaussian noisy h0 | absolute | 0.00 | Phase 2D strongest control baseline |
| `gaussian_noise_abs_pos_tok010` | matched Gaussian noisy h0 | absolute | 0.10 | token objective가 Gaussian path도 강화하는지 확인 |

## 4. Gate 결과

Phase 3A strict overall pass는 실패했다.

| Gate | 결과 | 의미 |
|---|---|---|
| `P3A-G-RUN` | pass | reconstruction, token-loss sweep, position-only, Gaussian controls가 실행됨 |
| `P3A-G-TOKEN-HEAD` | pass | Average Pooling token objective가 token-head delta NLL을 크게 낮춤 |
| `P3A-G-DECODER` | fail | best token arm이 frozen decoder delta NLL을 크게 악화시킴 |
| `P3A-G-LATENT-USE` | pass | best token arm이 perturbation/ablation/swap 반응을 유지함 |
| `P3A-G-CONTENT-CONTROL` | pass | best token arm이 position-only보다 content-sensitive함 |
| `P3A-G-GENERATION` | fail | Average Pooling token objective가 meaningful generation을 살리지 못함 |
| `P3A-G-GAUSSIAN-GAP` | fail | Gaussian token arm이 token-head에서는 여전히 더 강하고, generation gap도 해결되지 않음 |

자동 판정:

```text
overall_pass: false
phase3a_success: false
best_average_pool_rel_token_arm: average_pool_rel_pos_tok020
evidence_count: 3 / 2 required
```

이 결과는 단순 실패라기보다 중요한 부정 신호다.

> token-head objective는 token-head proxy를 강하게 최적화했지만, frozen decoder readability와 open-ended generation으로 이어지지 않았다.

## 5. 핵심 평균 지표

| Arm | MSE | Cosine | Decoder dNLL | Head dNLL | Shuffled decoder delta | Shuffled head delta | Meaningful gen | Rel sens | Abl dMSE | Swap dMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `average_pool_rel_pos_recon` | 0.020693 | 0.550328 | 4.909255 | 0.471910 | 1.256667 | 0.810003 | 0.000000 | 0.050931 | 0.002917 | 0.016686 |
| `average_pool_rel_pos_tok005` | 0.026442 | 0.408366 | 5.505023 | -1.802002 | 1.347515 | 2.575499 | 0.000000 | 0.041521 | 0.001443 | 0.013128 |
| `average_pool_rel_pos_tok010` | 0.029554 | 0.357759 | 5.731707 | -1.971045 | 1.552877 | 2.843584 | 0.000000 | 0.047900 | 0.000903 | 0.013381 |
| `average_pool_rel_pos_tok020` | 0.035089 | 0.312245 | 6.034574 | -2.051497 | 1.636463 | 2.948761 | 0.000000 | 0.057247 | 0.000272 | 0.015617 |
| `average_pool_abs_rel_pos_tok010` | 0.030632 | 0.355309 | 5.877280 | -1.965484 | 1.482380 | 2.824009 | 0.000000 | 0.048492 | 0.001071 | 0.015442 |
| `position_only_rel_pos_tok010` | 0.030445 | 0.264513 | 7.195593 | -0.574886 | 0.910630 | 0.669421 | 0.000000 | 0.184750 | 0.000000 | 0.000000 |
| `position_only_abs_rel_pos_tok010` | 0.031351 | 0.267868 | 7.087071 | -0.577036 | 0.913244 | 0.653972 | 0.000000 | 0.068221 | 0.000000 | 0.000000 |
| `gaussian_noise_abs_pos_recon` | 0.014822 | 0.720791 | 4.082911 | 0.025488 | 2.446911 | 1.574203 | 0.416667 | 0.039112 | 0.004975 | 0.026930 |
| `gaussian_noise_abs_pos_tok010` | 0.028492 | 0.472129 | 5.272116 | -3.571525 | 1.602952 | 5.501135 | 0.000000 | 0.028053 | 0.001511 | 0.017533 |

## 6. 결과 해석

### 6.1 Token-head NLL

Token-head delta NLL은 가장 크게 개선됐다.

```text
average_pool_rel_pos_recon head dNLL:  0.471910
average_pool_rel_pos_tok005 head dNLL:-1.802002
average_pool_rel_pos_tok010 head dNLL:-1.971045
average_pool_rel_pos_tok020 head dNLL:-2.051497
```

이 지표만 보면 Phase 3A는 매우 성공적으로 보일 수 있다. `lambda_token`이 커질수록 token-head delta가 낮아졌고, best arm은 `average_pool_rel_pos_tok020`이었다.

하지만 이 값이 음수가 됐다는 점이 중요하다. delta는 reconstructed hidden의 token-head NLL에서 원본 `h0`의 oracle token-head NLL을 뺀 값이다. 음수라는 것은 expander output이 원본 `h0`보다도 token head에 더 맞는 representation이 됐다는 뜻이다.

이것은 좋은 신호일 수도 있지만, 이번 결과에서는 오히려 proxy over-optimization으로 보는 것이 더 방어 가능하다. 이유는 frozen decoder NLL과 generation이 같이 좋아지지 않았기 때문이다.

### 6.2 Hidden-state MSE와 Cosine

Token objective는 hidden reconstruction 품질을 악화시켰다.

```text
MSE:
recon:  0.020693
tok005: 0.026442
tok010: 0.029554
tok020: 0.035089

Cosine:
recon:  0.550328
tok005: 0.408366
tok010: 0.357759
tok020: 0.312245
```

이 변화는 거의 단조적이다. `lambda_token`이 커질수록 MSE는 올라가고 cosine은 내려간다.

이것이 의미하는 바는 명확하다.

> token-head loss는 expander output을 원본 T5 encoder hidden manifold에서 벗어나게 만들고 있다.

LACE 관점에서 이것은 위험하다. 우리가 원하는 것은 token classifier에 잘 맞는 임의의 hidden이 아니라, T5 decoder가 읽을 수 있고 generation으로 이어지는 encoder-like hidden이다.

### 6.3 Frozen Decoder Delta NLL

Frozen decoder delta NLL도 악화됐다.

```text
average_pool_rel_pos_recon decoder dNLL: 4.909255
average_pool_rel_pos_tok005 decoder dNLL:5.505023
average_pool_rel_pos_tok010 decoder dNLL:5.731707
average_pool_rel_pos_tok020 decoder dNLL:6.034574
```

이 지표는 reconstructed hidden을 frozen T5 decoder가 teacher-forced token reconstruction에 얼마나 읽을 수 있는지 본다.

Token-head loss가 token-head NLL은 낮췄는데 frozen decoder NLL은 높였다는 점이 Phase 3A의 핵심 실패다. 즉 lightweight token head가 T5 decoder와 같은 hidden geometry를 요구하지 않는다. Expander는 token head의 decision boundary에는 더 맞게 움직였지만, T5 decoder가 기대하는 hidden trajectory에서는 멀어졌다.

따라서 이번 결과의 핵심 해석은 다음이다.

> lightweight token-head objective만으로는 generation-aware objective가 되지 않는다. 오히려 decoder-incompatible hidden을 만들 수 있다.

### 6.4 Position-only Control

Position-only controls도 token objective에서 강해졌지만, Average Pooling token arms가 content-sensitive 지표에서는 더 강했다.

```text
average_pool_rel_pos_tok020 cosine:       0.312245
position_only_rel_pos_tok010 cosine:      0.264513
position_only_abs_rel_pos_tok010 cosine:  0.267868

average_pool_rel_pos_tok020 shuffled head delta: 2.948761
position_only_rel_pos_tok010 shuffled head delta:0.669421
```

이것은 긍정 신호다. Token objective가 완전히 position-only shortcut으로만 빠진 것은 아니다. Average Pooling latent는 여전히 sample-specific content를 제공한다.

하지만 이 긍정 신호는 generation 실패를 뒤집지는 못한다. Content-sensitive하더라도 decoder-readable하지 않으면 LACE의 generation path로는 부족하다.

### 6.5 Latent Use

Best token arm인 `average_pool_rel_pos_tok020`은 latent-use 지표를 유지했다.

```text
relative sensitivity: 0.057247
ablation delta MSE:   0.000272
swap delta MSE:       0.015617
```

ablation delta는 약해졌지만 swap delta와 perturbation sensitivity가 살아 있다. 따라서 token objective가 latent를 완전히 무시하게 만든 것은 아니다.

다만 여기서도 caveat가 있다. Latent use는 "입력이 출력에 영향을 준다"를 보여줄 뿐이다. 그 영향이 T5 decoder가 읽을 수 있는 방향인지, language generation에 유효한 방향인지는 별도의 문제다.

### 6.6 Gaussian Control

Gaussian control은 이번 실험에서 매우 중요한 반론을 더 강하게 만들었다.

```text
gaussian_noise_abs_pos_recon:
MSE:            0.014822
Cosine:         0.720791
Decoder dNLL:   4.082911
Head dNLL:      0.025488
Meaningful gen: 0.416667

gaussian_noise_abs_pos_tok010:
MSE:            0.028492
Cosine:         0.472129
Decoder dNLL:   5.272116
Head dNLL:     -3.571525
Meaningful gen: 0.000000
```

`gaussian_noise_abs_pos_recon`은 reconstruction-only 상태에서 가장 좋은 generation behavior를 보였다. 반면 Gaussian에도 token loss를 넣으면 token-head NLL은 훨씬 좋아졌지만 generation은 0으로 붕괴했다.

이 결과는 중요하다.

> token-head loss가 Average Pooling만 망가뜨린 것이 아니다. Gaussian path에서도 token-head loss는 generation behavior를 망가뜨렸다.

따라서 문제는 Average Pooling compression 자체만의 문제가 아니라, 현재 token-head objective가 frozen decoder generation geometry와 맞지 않는다는 데 있다.

### 6.7 Open-ended Generation

Open-ended generation은 Phase 3A에서 가장 중요한 실패 지점이다.

```text
average_pool_rel_pos_recon: 0.000000
average_pool_rel_pos_tok005:0.000000
average_pool_rel_pos_tok010:0.000000
average_pool_rel_pos_tok020:0.000000
gaussian_noise_abs_pos_recon:0.416667
gaussian_noise_abs_pos_tok010:0.000000
h0 control: 0.750000
```

Average Pooling token objective는 meaningful generation을 전혀 살리지 못했다. Qualitative sample도 대부분 반복, 공백, 짧은 generic fragment였다.

예를 들면 token objective arms에서는 다음 패턴이 반복됐다.

```text
a a a a ...
s s s s ...
the of the of the ...
```

이것은 token-head NLL 개선이 실제 generation behavior 개선이 아니라, token-head classifier가 선호하는 좁은 hidden direction으로 collapse했을 가능성을 보여준다.

## 7. 최종 판정

Phase 3A는 strict success가 아니다.

하지만 이 실험은 매우 유용한 실패다. 실패 이유가 선명하기 때문이다.

성공한 것:

1. frozen token-head loss를 expander training objective에 직접 넣는 runner가 정상 동작했다.
2. `lambda_token` sweep이 의도대로 token-head NLL을 강하게 낮췄다.
3. Average Pooling token arms는 position-only보다 content-sensitive한 신호를 유지했다.
4. latent perturbation/swap sensitivity도 완전히 사라지지 않았다.

실패한 것:

1. MSE와 cosine이 악화됐다.
2. frozen decoder delta NLL이 악화됐다.
3. open-ended generation은 Average Pooling token arms에서 전부 0으로 유지됐다.
4. Gaussian reconstruction-only control은 여전히 generation에서 더 강했다.
5. Gaussian에 token objective를 넣어도 generation이 좋아지지 않고 오히려 0으로 붕괴했다.

따라서 방어 가능한 결론은 다음이다.

> Current lightweight token-head objective is not a sufficient generation-aware objective. It improves a proxy classifier while moving reconstructed hidden states away from the frozen T5 decoder manifold.

한국어로 풀면:

> 현재 token-head loss는 "생성 가능성을 높이는 objective"라기보다 "token-head classifier를 속이기 쉬운 hidden을 만드는 objective"에 가깝다.

## 8. 다음 단계

다음 단계는 Phase 3B로 잡는 것이 좋다.

Phase 3B의 목표는 token-head proxy를 더 세게 넣는 것이 아니라, decoder-compatible objective로 바꾸는 것이다.

후보는 다음 순서로 검토한다.

1. **Decoder NLL distillation objective**
   - frozen T5 decoder의 teacher-forced NLL을 직접 loss에 넣는다.
   - 비용은 크지만, token-head proxy와 decoder geometry 불일치를 줄일 수 있다.

2. **Hidden manifold regularized token objective**
   - `lambda_token`을 낮추고 `lambda_hidden` 또는 cosine을 더 강하게 유지한다.
   - 이번 결과에서는 `lambda_token=0.05`도 generation을 살리지 못했으므로 단독 해결책은 아닐 수 있다.

3. **Two-head agreement objective**
   - token head 하나만 믿지 않고, frozen decoder NLL과 token head가 동시에 좋아지는 방향만 허용한다.

4. **Trainable adapter instead of direct hidden overwrite**
   - expander가 `h0_hat`를 직접 token-head boundary로 밀지 않게, decoder-compatible adapter 또는 residual constraint를 둔다.

현재 가장 추천하는 다음 실험은 다음이다.

```text
Phase 3B: Decoder-compatible Reverse Objective
```

핵심 질문:

> token-head proxy가 아니라 frozen decoder NLL을 훈련 objective에 직접 넣으면, Average Pooling relative positional latent가 generation collapse에서 벗어나는가?

단, Phase 3B는 더 무거울 가능성이 높다. Phase 3A도 Kaggle에서 2D보다 오래 걸렸기 때문에, 3B는 처음부터 full 9-arm sweep으로 가기보다 다음 compact set으로 시작하는 것이 좋다.

```text
average_pool_rel_pos_recon
average_pool_rel_pos_decoder005
average_pool_rel_pos_decoder010
position_only_rel_pos_decoder010
gaussian_noise_abs_pos_recon
gaussian_noise_abs_pos_decoder010
```

이렇게 해야 compute 비용을 줄이면서도 핵심 반론을 유지할 수 있다.
