# S2a 결과: positional encoding 비교

## 1. 이 실험이 측정한 것

S2a는 S2의 `front/middle/back` 위치 tag를 더 정석적인 positional encoding으로 바꿨을 때, 의미 골격-문장 복원 성능이 어떻게 달라지는지 측정했다.

실험 이름은 다음이다.

```text
S2a-positional encoding
```

S2a에서는 attention 기반 의미 골격 token 선택은 고정했다. 비교 대상은 skeleton scorer가 아니라 위치 표현 방식이다.

실행 정보는 다음이다.

| 항목 | 값 |
|---|---|
| phase | `v2_s2a` |
| Kaggle kernel | `dennisparknd/lace-v2-s2a-positional-encoding` |
| model | `t5-small` tokenizer + custom PyTorch encoder-decoder |
| data | `wikitext/wikitext-2-raw-v1:train` |
| train samples | 768 |
| eval samples | 192 |
| device | `cuda` |
| keep ratio | 0.25 |
| output | `outputs/v2_s2a/lace_v2_s2a/` |

## 2. 왜 중요한가

S2에서 `attention_scaffold`는 `random_scaffold`보다 좋았다. 하지만 S2의 위치 보조 구조는 `front`, `middle`, `back`이라는 coarse 문자열 tag였다.

이 방식은 control을 만들기 쉽지만 일반적인 transformer positional encoding은 아니다. S3 anchor baseline comparison으로 넘어가기 전에, 위치 보조 구조를 더 정석적인 방식으로 바꿀 후보를 확인해야 한다.

## 3. 결과가 의미하는 것

S2a는 `overall_pass=true`, `s3_ready=true`로 통과했다.

Gate 결과는 다음이다.

| Gate | 통과 | 해석 |
|---|---:|---|
| `S2A-G-RUN` | true | 6개 조건이 실행됐다. |
| `S2A-G-LOSS-FINITE` | true | 모든 조건의 loss가 정상 숫자로 계산됐다. |
| `S2A-G-STANDARD-BEATS-NONE` | true | 정식 positional encoding 후보 중 하나가 위치 없음보다 좋았다. |
| `S2A-G-STANDARD-BEATS-COARSE` | true | 정식 positional encoding 후보 중 하나가 coarse bin보다 좋았다. |
| `S2A-G-BEST-IDENTIFIED` | true | 최고 후보가 `sinusoidal_absolute`로 식별됐다. |

조건별 결과는 다음이다.

| 조건 | Loss | PPL | Token F1 | ROUGE-L | Keyword Recall | Skeleton Coverage | Nonempty |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sinusoidal_absolute` | 6.0715 | 433.33 | 0.1661 | 0.1509 | 0.0099 | 0.0078 | 1.0000 |
| `relative_position_bias` | 6.0817 | 437.78 | 0.1551 | 0.1346 | 0.0073 | 0.0059 | 1.0000 |
| `coarse_bins` | 6.0901 | 441.47 | 0.1533 | 0.1425 | 0.0047 | 0.0056 | 1.0000 |
| `rotary_position` | 6.1214 | 455.50 | 0.1443 | 0.1333 | 0.0151 | 0.0124 | 1.0000 |
| `learned_absolute` | 6.1152 | 452.68 | 0.1313 | 0.1204 | 0.0056 | 0.0058 | 1.0000 |
| `no_position` | 6.1323 | 460.48 | 0.1218 | 0.1109 | 0.0059 | 0.0069 | 1.0000 |

핵심 비교는 다음이다.

```text
sinusoidal_absolute: Token F1 0.1661, ROUGE-L 0.1509, loss 6.0715
coarse_bins:         Token F1 0.1533, ROUGE-L 0.1425, loss 6.0901
no_position:         Token F1 0.1218, ROUGE-L 0.1109, loss 6.1323
```

따라서 이 lightweight S2a probe 안에서는 `sinusoidal_absolute`가 가장 좋은 위치 표현 후보였다.

## 4. 어떤 반론과 혼동 요인을 줄였는가

S2a는 "S2의 `front/middle/back`이 너무 임시적이지 않은가"라는 문제를 직접 다뤘다. 결과상 `sinusoidal_absolute`는 `coarse_bins`와 `no_position`보다 조금 더 좋았다. 이는 정석적인 위치 부호화가 S2의 coarse tag보다 개선 여지가 있음을 보여준다.

다만 개선 폭은 크지 않다. `sinusoidal_absolute`와 `coarse_bins`의 Token F1 차이는 0.0128이고, ROUGE-L 차이는 0.0084다. 따라서 이 결과를 "위치 표현을 바꾸면 복원이 크게 좋아진다"로 과장하면 안 된다.

또한 생성 샘플에는 반복과 표면적 단어 겹침이 많다. Keyword recall과 skeleton coverage도 전반적으로 낮다. 즉 S2a는 위치 표현 후보를 고르는 probe이지, 문장 생성 품질이 충분하다는 증거가 아니다.

## 5. 다음 실험에서 어떻게 검증할 것인가

S3로 넘어갈 수 있다. 기본 positional scaffold 후보는 `sinusoidal_absolute`로 둔다.

하지만 S3에서는 다음 두 caveat를 유지한다.

- `sinusoidal_absolute`는 S2a의 lightweight custom model에서 가장 좋았을 뿐, T5 fine-tuning 전체 구조에서 최종 검증된 것은 아니다.
- `coarse_bins`와의 차이가 작으므로, S3에서는 가능하면 `sinusoidal_absolute`와 `coarse_bins`를 함께 ablation으로 남긴다.

S2a의 방어 가능한 결론은 다음이다.

> S2의 coarse 위치 tag보다 정석적인 positional encoding을 적용할 수 있으며, 이번 비교에서는 sinusoidal absolute encoding이 가장 좋은 S3 후보로 식별됐다.

아직 주장하면 안 되는 것은 다음이다.

- sinusoidal encoding이 LACE의 최종 위치 표현이라는 주장
- positional encoding만 바꾸면 generation 품질이 해결된다는 주장
- rotary 또는 relative bias가 일반적으로 나쁘다는 주장
