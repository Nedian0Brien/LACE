# S4c 계획: span-infilling reverse decoder

## 1. 무엇을 측정하는가

S4c는 S4a에서 확인된 delta-token objective를 유지하되, reverse decoder의 형태를 바꾼다.

S4a는 현재 state와 이번 단계의 delta position marker를 조건으로 주고, 새로 unmask될 token sequence를 자유 생성했다.

```text
input:  현재 partial state + 이번 단계에서 채울 위치 marker
target: newly unmasked delta token sequence
```

S4c는 같은 reverse transition을 position-level infilling 문제로 바꾼다.

```text
input:  현재 partial state + masked position marker entries
target: 각 marker entry의 원래 token id
```

따라서 S4c가 측정하는 것은 **semantic skeleton과 positional scaffold가 marker별 token 분류에서 random corruption이나 position-only control보다 더 좋은 조건을 주는가**다. 첫 버전에서는 contiguous span도 여러 marker position의 독립 token prediction으로 처리한다. 이는 full span autoregression이 아니라 position-level span infilling이다.

## 2. 왜 지금 중요한가

S4a는 `importance_schedule`이 aggregate delta objective에서 `random_schedule`과 `position_only_schedule`을 이긴 첫 강한 신호를 줬다.

```text
importance score: 0.6366
random score:     0.5073
position-only:    0.5889
```

하지만 남은 caveat도 뚜렷하다.

- entity recall은 random이 더 높았다.
- repetition은 importance가 더 나빴다.
- position-only가 여전히 강했다.

이 중 repetition은 decoder가 delta token sequence를 autoregressive하게 자유 생성하면서 생기는 구조적 artifact일 수 있다. S4c는 free generation loop를 제거하고 marker 위치에서 바로 token을 맞히게 하여, 반복 문제가 semantic skeleton 자체의 문제인지 decoder 구조의 문제인지 분리한다.

## 3. 좋은 결과와 나쁜 결과의 의미

좋은 결과는 `importance_schedule`이 marker-level infilling에서도 `random_schedule`과 `position_only_schedule`보다 높은 score를 얻는 것이다.

| 비교 | 좋은 결과 | 의미 |
|---|---|---|
| masked token accuracy | importance > random | semantic skeleton이 marker별 token 예측에 직접 도움이 된다. |
| span token accuracy/F1 | importance > random | delta span의 표면 token 복원이 좋아진다. |
| content/entity recall | importance > random | 세부 content token과 entity/number 표지가 더 잘 복원된다. |
| importance vs position-only | importance > position-only | 위치 marker만이 아니라 content-bearing skeleton이 기여한다. |
| duplicate/repetition | importance가 random보다 크게 나쁘지 않음 | 자유 생성 decoder의 반복 artifact가 완화된다. |
| context copy leakage | importance가 과도하게 높지 않음 | 새 token 예측이 visible context 복사로 대체되지 않는다. |

나쁜 결과는 position-level infilling에서도 random이나 position-only가 이기는 경우다. 이 경우 S4a의 개선은 decoder objective 보정의 일부 효과였을 수 있지만, 현재 semantic skeleton content가 marker별 세부 token을 안정적으로 복원할 만큼 충분하지 않다는 해석을 유지해야 한다.

## 4. 어떤 반론과 모호성을 다루는가

S4c는 다음 confound를 다룬다.

- S4a의 repetition 문제가 free autoregressive decoder 때문인가?
- marker position만 있어도 대부분의 성능이 설명되는가?
- semantic skeleton content가 실제로 masked position token prediction에 기여하는가?
- entity/rare-token 실패가 decoder 구조 문제인지 schedule 정보 문제인지 분리할 수 있는가?
- context copy가 infilling score를 인위적으로 올리는가?

`position_only_schedule`은 hard control로 유지한다. 이 조건은 importance schedule과 같은 marker 위치를 쓰지만, visible context token content를 `pad_token_id` placeholder로 제거한다. target은 같은 importance position의 token을 따른다. 따라서 이 control이 강하면 positional scaffold 효과가 크다는 뜻이며, semantic skeleton 사용 증거로 과장하지 않는다.

## 5. metric, gate, 다음 판단

S4c의 schedule 조건은 다음 세 가지다.

| 조건 | 설명 |
|---|---|
| `importance_schedule` | attention-received score가 높은 token을 오래 보존하고, 다음 단계 delta position을 marker로 채운다. |
| `random_schedule` | 같은 ratio와 token count에서 random order의 delta position을 marker로 채운다. |
| `position_only_schedule` | importance 위치 marker와 position scaffold만 유지하고 token content는 제거한다. |

주요 metric은 다음이다.

| Metric | 의미 |
|---|---|
| `masked_token_accuracy` | 모든 marker token id에 대한 weighted exact accuracy |
| `span_token_accuracy` | transition/span 단위 marker token accuracy 평균 |
| `span_exact_match` | 한 transition의 marker token을 모두 맞혔는지 |
| `span_token_f1`, `span_rouge_l_f1` | decode된 predicted span과 target span의 lexical overlap |
| `content_recall` | target span의 content word recall |
| `entity_recall` | target span의 entity/number recall |
| `duplicate_prediction_rate` | marker prediction 안에서 같은 token id가 중복되는 비율 |
| `repetition_rate` | decode된 prediction의 bigram repetition |
| `context_copy_leakage` | predicted token id가 visible context token에 포함되는 비율 |

Gate는 `S4C-*` namespace를 사용한다.

| Gate | 통과 조건 | 해석 |
|---|---|---|
| `S4C-G-RUN` | 모든 요청 schedule이 실행됨 | 실험 완결성 |
| `S4C-G-LOSS-FINITE` | 모든 eval loss가 유한함 | 수치 안정성 |
| `S4C-G-IMPORTANCE-BEATS-RANDOM` | importance aggregate score가 random보다 tolerance 이상 높음 | semantic skeleton trajectory 우위 |
| `S4C-G-IMPORTANCE-BEATS-POSITION-ONLY` | importance aggregate score가 position-only보다 tolerance 이상 높음 | content-bearing skeleton 사용 증거 |
| `S4C-G-MASKED-TOKEN-ACCURACY` | importance masked token accuracy가 random보다 높음 | marker-level token 예측 개선 |
| `S4C-G-ENTITY-CONTENT-IMPROVEMENT` | importance의 content+entity recall 합이 random보다 높음 | entity/content handling 개선 |
| `S4C-G-COPY-LEAKAGE` | importance context copy leakage가 random보다 크게 높지 않음 | 복사 artifact 방지 |
| `S4C-G-REPETITION-NONWORSE` | duplicate/repetition이 random보다 크게 나쁘지 않음 | 반복 문제 완화 또는 비악화 |
| `S4C-G-BEST-IDENTIFIED` | 최고 schedule과 score가 식별됨 | 다음 phase 판단 가능 |

S4c가 통과하면 S4a의 positive signal이 decoder 구조를 바꿔도 유지되는지 확인한 것이며, S4b의 multi-step rollout 결과와 함께 S5 전 구조 판단에 사용할 수 있다. S4c가 repetition은 개선하지만 position-only를 이기지 못하면 positional scaffold 의존성이 여전히 크다는 뜻이므로, wrong-document/same-position control이나 entity-aware head를 가까운 후속 실험으로 둔다.
