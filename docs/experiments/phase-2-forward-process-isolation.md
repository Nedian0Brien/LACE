# Phase 2 실험 결과: Forward Process Isolation

## 1. 실험 목적

Phase 2의 목적은 LACE의 핵심 가설을 더 직접적으로 확인하는 것이다.

Phase 1은 압축된 latent에서 `h0`를 복원하는 reverse expander가 학습 가능한지 확인했다. 하지만 그 결과만으로는 LACE가 downsampling autoencoder와 본질적으로 다르다고 말하기 어렵다.

이번 Phase 2에서는 질문을 다음처럼 바꾸었다.

> 같은 데이터, 같은 frozen encoder latent, 같은 reverse expander 조건에서 정보 압축 forward process는 corruption/noise forward process보다 더 좋은 reverse learning 경로를 제공하는가?

즉, 이번 실험은 압축 방식 자체의 새로움을 보려는 것이 아니라, **Diffusion Language Modeling의 forward process를 corruption이 아니라 information compression으로 바꾸는 것이 타당한가**를 보기 위한 첫 번째 isolation 실험이다.

## 2. 실행 정보

| 항목 | 값 |
|---|---|
| Kaggle kernel | `dennisparknd/lace-phase-2-forward-process-isolation` |
| kernel version | 1 |
| 실행 상태 | 완료 |
| 모델 | `t5-small` |
| encoder 상태 | frozen |
| 데이터 소스 | `hf:wikitext/wikitext-2-raw-v1:train` |
| 샘플 수 | 512 |
| 최대 길이 | 128 token |
| hidden shape | `[512, 128, 512]` |
| active tokens | 49201 |
| stage tokens | 64, 32, 16 |
| generation bridge | enabled |
| 결과 위치 | `outputs/phase2/lace_phase2/` |

생성된 주요 파일:

| 파일 | 설명 |
|---|---|
| `metrics.json` | 전체 정량 결과 |
| `summary.md` | Kaggle runner가 생성한 요약 |
| `train_log.jsonl` | condition/stage별 학습 로그 |
| `generation_samples.jsonl` | qualitative generation sample |
| `checkpoints/*.pt` | condition/stage별 expander checkpoint |

## 3. 비교 조건

이번 실험에서는 네 가지 forward process를 비교했다.

| 조건 | 유형 | 역할 |
|---|---|---|
| `average_pool` | fixed compression | 핵심 compression condition |
| `strided_select` | fixed compression | 선택형 token-budget compression |
| `random_select` | stochastic drop/corruption | 같은 token budget을 가진 corruption baseline |
| `gaussian_noise` | continuous noise corruption | 기존 continuous diffusion류 reference |

각 condition은 `z1`, `z2`, `z3` 세 stage로 실행했다.

```text
z1: 64 token 또는 sigma 0.10
z2: 32 token 또는 sigma 0.20
z3: 16 token 또는 sigma 0.40
```

주의할 점은 `gaussian_noise`는 token 수를 줄이지 않고 128 token 길이를 유지한다는 것이다. 따라서 Gaussian 결과는 raw MSE만으로 compression condition과 단순 비교하면 안 된다. 이번 실험에서는 별도 calibration table도 기록했다.

## 4. Gate 결과

자동 gate 기준으로는 P2-G1부터 P2-G6까지 모두 pass였다.

| Gate | 결과 | 의미 |
|---|---|---|
| P2-G1 | pass | 네 forward condition이 모두 같은 split에서 실행됨 |
| P2-G2 | pass | `average_pool`이 `random_select`보다 reconstruction MSE 또는 cosine에서 우수 |
| P2-G3 | pass | `average_pool`이 `gaussian_noise`보다 reconstruction 또는 token NLL에서 우수 |
| P2-G4 | pass | `average_pool` stage 난도가 단조롭게 증가 |
| P2-G5 | pass | latent perturbation/ablation/swap에서 latent 사용 신호 확인 |
| P2-G6 | pass | compression condition이 random baseline보다 낮은 delta token NLL 기록 |

하지만 이 결과는 **강한 성공**으로 해석하면 안 된다. 자동 gate는 통과했지만, 실제 숫자를 보면 더 섬세하게 봐야 한다.

핵심 요약:

| 항목 | `average_pool` | `random_select` | `gaussian_noise` | 해석 |
|---|---:|---:|---:|---|
| 평균 MSE | 0.936960 | 0.922394 | 0.862077 | MSE만 보면 compression 우위가 아님 |
| 평균 cosine | 0.304123 | 0.203049 | 약 0.235703 | 방향성 보존은 `average_pool`이 가장 좋음 |
| 평균 delta token NLL | 6.542922 | 7.162894 | 7.454267 | generation bridge는 `average_pool`이 가장 좋음 |
| 평균 relative sensitivity | 0.008186 | 약 0.010419 | 약 0.006134 | perturbation sensitivity는 작지만 0은 아님 |
| 평균 ablation delta | 0.030264 | 0.006666 | 0.018967 | `average_pool`은 latent 제거에 가장 민감 |

따라서 Phase 2의 더 정확한 판정은 다음이다.

> Phase 2는 "정보 압축 forward가 corruption/noise보다 모든 지표에서 우수하다"를 보인 것은 아니다. 그러나 average pooling compression이 random selection이나 Gaussian noise보다 latent 방향성 보존과 token reconstruction bridge에서 더 좋은 신호를 보였으므로, LACE의 핵심 가설을 다음 단계로 가져갈 근거는 생겼다.

## 5. Condition별 상세 결과

### 5.1 Average Pooling

| Stage | Val loss | MSE | Cosine | Var ratio | Rel sens | Abl delta | Swap delta | Delta token NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `z1` | 0.998180 | 0.929795 | 0.316147 | 2.294140 | 0.006425 | 0.033595 | 0.028762 | 6.745355 |
| `z2` | 1.007426 | 0.937731 | 0.303048 | 1.592847 | 0.007642 | 0.029377 | 0.019169 | 6.844072 |
| `z3` | 1.014038 | 0.943355 | 0.293175 | 1.367364 | 0.010490 | 0.027820 | 0.001743 | 6.039338 |

Average pooling의 가장 좋은 점은 stage 곡선이 자연스럽다는 것이다.

```text
MSE:    0.929795 -> 0.937731 -> 0.943355
Cosine: 0.316147 -> 0.303048 -> 0.293175
```

압축이 강해질수록 MSE는 증가하고 cosine은 감소한다. 이 곡선은 `t`를 information rate로 해석하는 LACE의 기본 관점과 잘 맞는다.

또한 ablation delta가 다른 compression/corruption 조건보다 크다.

```text
average_pool ablation delta 평균: 0.030264
random_select ablation delta 평균: 0.006666
```

이는 average pooling latent를 제거했을 때 복원 성능이 더 크게 흔들린다는 뜻이다. 즉, expander가 average pooling latent를 실제로 어느 정도 사용하고 있다는 신호로 볼 수 있다.

### 5.2 Strided Selection

| Stage | Val loss | MSE | Cosine | Var ratio | Rel sens | Abl delta | Swap delta | Delta token NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `z1` | 0.998784 | 0.928114 | 0.293298 | 3.511339 | 0.006385 | 0.004522 | 0.024377 | 7.633898 |
| `z2` | 1.000823 | 0.923890 | 0.230673 | 5.277825 | 0.008838 | 0.013556 | 0.007886 | 6.732108 |
| `z3` | 0.998526 | 0.916644 | 0.207980 | 6.381182 | 0.010208 | 0.007400 | -0.002063 | 7.013197 |

Strided selection은 MSE만 보면 average pooling보다 낮게 나오는 구간이 있다. 그러나 cosine이 크게 낮고 variance ratio가 높다. 특히 stage가 깊어질수록 variance ratio가 커진다.

```text
Variance ratio: 3.511339 -> 5.277825 -> 6.381182
Cosine:         0.293298 -> 0.230673 -> 0.207980
```

이는 strided selection이 원본 token 일부를 그대로 남기기 때문에 MSE상으로는 유리해 보일 수 있지만, 전체 representation 방향성과 안정성은 average pooling보다 약하다는 뜻으로 해석된다.

### 5.3 Random Selection

| Stage | Val loss | MSE | Cosine | Var ratio | Rel sens | Abl delta | Swap delta | Delta token NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `z1` | 1.004391 | 0.924350 | 0.205329 | 5.889937 | 0.009741 | 0.008971 | -0.011102 | 6.735460 |
| `z2` | 1.001262 | 0.921842 | 0.205796 | 5.328765 | 0.010488 | 0.008119 | -0.017807 | 7.084403 |
| `z3` | 1.001417 | 0.920991 | 0.198021 | 5.300405 | 0.011028 | 0.002909 | -0.019553 | 7.668820 |

Random selection은 MSE가 average pooling보다 낮다. 하지만 cosine은 낮고, stage별 난도 곡선도 약하다.

```text
MSE:    0.924350 -> 0.921842 -> 0.920991
Cosine: 0.205329 -> 0.205796 -> 0.198021
```

특히 `z1 -> z2 -> z3`로 갈수록 MSE가 오히려 낮아진다. 이는 information-rate schedule로서 해석하기 어렵다. LACE에서 중요한 것은 단순히 MSE가 낮은 것이 아니라, forward time `t`가 정보율 감소로 읽힐 수 있어야 한다.

따라서 random selection은 raw MSE에서는 강하지만, **diffusion forward schedule로서의 구조성은 약하다**.

### 5.4 Gaussian Noise

| Stage | Val loss | MSE | Cosine | Var ratio | Rel sens | Abl delta | Swap delta | Delta token NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `z1` | 0.993357 | 0.926180 | 0.328231 | 2.648418 | 0.005356 | -0.002945 | 0.049986 | 7.896796 |
| `z2` | 1.004600 | 0.930609 | 0.260094 | 3.513278 | 0.004114 | -0.001888 | 0.037488 | 7.349305 |
| `z3` | 0.888881 | 0.729444 | 0.118785 | 14.353273 | 0.008931 | 0.061735 | 0.012244 | 7.116700 |

Gaussian noise는 가장 해석이 까다롭다. `z3`에서 MSE가 매우 낮게 나온다.

```text
gaussian_noise z3 MSE: 0.729444
```

하지만 동시에 cosine은 매우 낮고, variance ratio가 비정상적으로 높다.

```text
gaussian_noise z3 cosine: 0.118785
gaussian_noise z3 variance ratio: 14.353273
```

즉, MSE만 보면 좋아 보이지만 representation 방향성과 scale 안정성은 좋지 않다. 또한 token NLL bridge도 average pooling보다 나쁘다.

```text
average_pool 평균 delta token NLL: 6.542922
gaussian_noise 평균 delta token NLL: 7.454267
```

따라서 Gaussian noise는 hidden MSE 일부에서는 강하지만, generation bridge와 representation 안정성에서는 compression보다 불리한 신호를 보였다.

## 6. Gaussian Calibration

초기 forward 난도 calibration은 다음과 같다.

| 항목 | 값 |
|---|---:|
| average_pool `z1` initial loss | 0.007640 |
| average_pool `z2` initial loss | 0.011842 |
| average_pool `z3` initial loss | 0.014766 |
| Gaussian sigma 0.05 initial loss | 0.002500 |
| Gaussian sigma 0.10 initial loss | 0.010001 |
| Gaussian sigma 0.20 initial loss | 0.040013 |
| Gaussian sigma 0.40 initial loss | 0.159995 |
| Gaussian sigma 0.80 initial loss | 0.639710 |

Calibration상으로는 average pooling의 `z1/z2/z3` 난도 모두 sigma 0.10 근처와 가장 가깝다. 그런데 본 실험의 Gaussian stage는 0.10, 0.20, 0.40을 사용했다. 따라서 Gaussian `z2/z3`는 average pooling보다 더 강한 corruption 조건일 수 있다.

이 점 때문에 Gaussian 비교는 다음 실험에서 반드시 보정해야 한다.

> Phase 2B에서는 Gaussian sigma를 `0.05/0.10/0.15` 또는 matched sigma 중심으로 다시 실행하는 것이 좋다.

## 7. Generation Bridge 결과

Phase 2에서 가장 중요한 추가점은 RQ6을 반영해 generation bridge를 넣었다는 것이다. 이번에는 full generation-capable DLM을 만든 것이 아니라, 복원된 `h0_hat`를 frozen T5 decoder에 넣고 teacher-forced token NLL을 측정했다.

핵심 지표는 다음이다.

```text
delta_token_nll_vs_h0 = NLL(h0_hat -> text) - NLL(h0 -> text)
```

평균 결과:

| 조건 | 평균 delta token NLL |
|---|---:|
| `average_pool` | 6.542922 |
| `strided_select` | 약 7.126401 |
| `random_select` | 7.162894 |
| `gaussian_noise` | 7.454267 |

이 결과는 average pooling이 token-level bridge에서 가장 낫다는 신호다. 특히 hidden MSE에서는 random/gaussian보다 약한데도 token NLL delta에서는 가장 좋다. 이는 LACE 가설에 우호적인 결과다.

다만 qualitative generation sample은 아직 좋지 않다. 대부분 빈 문자열에 가까운 공백 또는 쉼표 반복으로 나왔다. 따라서 이번 결과를 "텍스트 생성 성공"으로 해석하면 안 된다.

정확한 해석은 다음이다.

> Teacher-forced token reconstruction proxy에서는 compression forward가 유리한 신호를 보였지만, open-ended generation은 아직 붕괴되어 있다.

## 8. 이번 실험의 의미

이번 Phase 2는 LACE의 핵심 주장에 대해 처음으로 직접적인 비교를 제공했다.

긍정적인 신호:

- `average_pool`은 stage별 난도 곡선이 가장 자연스럽다.
- `average_pool`은 random selection보다 cosine이 높다.
- `average_pool`은 Gaussian noise보다 평균 delta token NLL이 낮다.
- `average_pool`은 latent ablation에 가장 민감해 latent 사용 신호가 있다.
- generation bridge metric이 실험에 처음 포함됐다.

조심해야 할 신호:

- MSE만 보면 `average_pool`이 가장 좋은 조건은 아니다.
- Gaussian noise `z3`는 MSE가 낮지만 variance ratio가 비정상적으로 높다.
- Random selection은 MSE가 낮지만 stage schedule로 해석하기 어렵다.
- Open-ended generation sample은 아직 붕괴되어 있다.
- T5 decoder bridge는 teacher-forced proxy일 뿐, 완성된 DLM generation 평가가 아니다.

따라서 결론은 다음이다.

> Phase 2는 LACE 가설을 강하게 입증하지는 못했지만, compression forward process가 corruption/noise보다 representation 방향성 보존과 token reconstruction bridge에서 유리할 수 있다는 초기 증거를 제공했다.

## 9. 다음 단계

바로 Phase 3로 가기 전에 Phase 2B를 한 번 더 수행하는 것이 좋다.

### 9.1 Phase 2B에서 수정할 것

1. Gaussian sigma를 calibration에 맞춰 재설정한다.
   - 현재 `z1/z2/z3` 모두 sigma 0.10 근처와 가장 비슷하다.
   - 다음 run에서는 `0.05, 0.10, 0.15` 또는 matched sigma를 사용한다.

2. Training epoch를 늘린다.
   - 이번 run은 2 epoch였다.
   - 4~6 epoch로 늘리면 condition 간 차이가 더 명확해질 수 있다.

3. Generation bridge를 개선한다.
   - frozen T5 decoder sample generation은 공백/쉼표 붕괴가 많다.
   - teacher-forced NLL은 유지하되, lightweight token head도 추가해 비교한다.

4. Gate 기준을 더 보수적으로 바꾼다.
   - 현재 P2-G2는 "MSE 또는 cosine"이라 pass가 쉽다.
   - 다음에는 reconstruction, cosine, token NLL을 분리해서 판정해야 한다.

5. 결과 해석에서 MSE와 cosine의 역할을 분리한다.
   - MSE는 magnitude proximity를 본다.
   - cosine은 representation direction을 본다.
   - DLM forward process에는 둘 다 필요하지만, token bridge와 더 잘 맞는 지표를 확인해야 한다.

### 9.2 Phase 3로 넘어가기 위한 기준

Phase 3로 넘어가려면 다음 중 최소 두 가지가 재현되어야 한다.

- `average_pool`이 random/corruption baseline보다 cosine에서 우수하다.
- `average_pool`이 delta token NLL에서 우수하다.
- `average_pool` 또는 `strided_select`가 generation bridge에서 붕괴를 줄인다.
- matched Gaussian 조건에서도 compression이 경쟁력 있다.
- latent ablation/swap test에서 compression latent 사용이 안정적으로 확인된다.

## 10. 최종 판정

현재 Phase 2의 판정은 다음이다.

| 항목 | 판정 |
|---|---|
| 실행 완결성 | 성공 |
| forward process 비교 | 성공 |
| compression의 MSE 우위 | 미입증 |
| compression의 cosine 우위 | 부분 입증 |
| compression의 token bridge 우위 | 부분 입증 |
| open-ended generation | 실패 |
| 다음 단계 진행 가치 | 있음 |

한 줄 결론:

> Average pooling compression은 단순한 downsampling에 불과해 보일 수 있지만, 같은 실험 조건에서 random selection이나 Gaussian noise보다 representation 방향성과 token reconstruction bridge가 더 나은 신호를 보였다. 다만 MSE와 실제 생성 품질은 아직 약하므로, Phase 2B에서 matched Gaussian과 개선된 generation bridge로 재검증해야 한다.

