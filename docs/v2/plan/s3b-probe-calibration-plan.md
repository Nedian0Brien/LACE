# S3b 계획: probe calibration

## 1. 무엇을 측정하는가

S3b는 S3a 이후 남은 두 가지 모호성을 분리한다.

첫째, `attention_terminal`의 이점이 terminal content에서 온 것인지, 조건별로 따로 학습된 probe의 우연한 최적화에서 온 것인지 확인한다. 이를 위해 reverse model은 `attention_terminal` 입력으로 한 번만 학습하고, 평가 시점에 입력만 바꾼다.

둘째, `position_only`와 `random_terminal_predicted_anchor`가 강했던 이유를 분해한다. 이를 위해 같은 모델에 `attention_no_position`, `attention_shuffled_position`, `position_only`, `random_terminal`, `same_position_random_terminal`, `random_terminal_predicted_anchor`, `random_terminal_gold_anchor_oracle`, `attention_gold_anchor`를 넣어 비교한다.

## 2. 왜 지금 중요한가

S3a는 content terminal 신호를 일부 회복했다. `attention_terminal`은 `random_terminal`과 `same_position_random_terminal`보다 높았다. 하지만 `position_only`와 차이가 작고, `random_terminal_predicted_anchor`가 최고 조건이었다.

따라서 S4 constrained generation으로 넘어가기 전에 reverse probe가 실제로 terminal content를 쓰는지 확인해야 한다. S3b가 통과하지 못하면 현재 결과는 "semantic skeleton이 좋은 reverse trajectory를 만든다"보다 "짧은 probe와 lexical metric이 위치 scaffold 또는 표면 prior를 강하게 탄다"는 해석이 더 방어 가능하다.

## 3. 좋은 결과와 나쁜 결과의 의미

좋은 결과는 다음 패턴이다.

| 비교 | 좋은 결과 | 의미 |
|---|---|---|
| `attention_terminal` > `position_only` | margin이 tolerance보다 큼 | 위치 scaffold만으로는 부족하고 terminal content가 쓰임 |
| `attention_terminal` > `random_terminal` | margin이 tolerance보다 큼 | 같은 token budget의 random terminal보다 content-bearing terminal이 유리함 |
| `attention_terminal` > `same_position_random_terminal` | margin이 tolerance보다 큼 | 같은 위치의 다른 문서 token은 충분하지 않음 |
| `attention_terminal` > `attention_no_position` | margin이 tolerance보다 큼 | 위치 보조 구조가 content terminal을 복원 방향으로 정렬함 |
| `random_terminal_gold_anchor_oracle` >= `random_terminal_predicted_anchor` | predicted anchor가 oracle을 이상하게 이기지 않음 | S3a의 predicted-anchor 우위가 줄어듦 |

나쁜 결과는 `attention_terminal`이 `position_only` 또는 `attention_no_position`과 거의 같거나 낮은 경우다. 이 경우 terminal content 사용 증거는 약하고, 위치 prior 또는 decoder prior가 metric을 만든다는 해석이 강해진다. `random_terminal_predicted_anchor`가 계속 gold oracle보다 높으면 anchor 문자열이 의미 anchor라기보다 probe-friendly prior로 작동했을 가능성이 크다.

## 4. 어떤 반론과 모호성을 다루는가

S3b는 S3a의 가장 큰 confound였던 조건별 재학습 문제를 줄인다. S3a에서는 각 조건마다 reverse model을 새로 학습했기 때문에 조건 차이가 입력 정보량 차이인지, 학습 난이도와 초기화의 차이인지 완전히 분리되지 않았다.

S3b는 같은 학습 모델을 고정하고 평가 입력만 바꾼다. 이 설계는 다음 반론을 직접 다룬다.

- `position_only`가 높다면 content terminal이 없어도 probe가 문장 평균 prior를 복원하는 것 아닌가?
- `attention_no_position`이 높다면 positional scaffold가 실제로 필요한가?
- `same_position_random_terminal`이 높다면 위치만 맞춘 다른 문서 token도 충분한가?
- predicted anchor가 gold anchor보다 높다면 anchor predictor가 의미를 잘 맞춘 것이 아니라 metric/probe 편향 아닌가?

## 5. 다음 실험으로 넘길 조건

S3b는 S4 진입 gate가 아니라 S4 전 보정 gate다. `diagnostic_ready=true`는 모든 조건이 실행되고 loss가 finite라는 뜻이다. `s4_ready`는 기본적으로 false로 둔다.

Gate는 다음과 같다.

| Gate | 통과 조건 | 해석 |
|---|---|---|
| `S3B-G-RUN` | 요청 조건이 모두 실행됨 | 실험 완결성 |
| `S3B-G-LOSS-FINITE` | 모든 eval loss가 유한함 | 수치 안정성 |
| `S3B-G-CONTENT-BEATS-SAME-POSITION` | `attention_terminal` score가 `same_position_random_terminal`보다 tolerance 이상 높음 | 같은 위치의 wrong content 반론 처리 |
| `S3B-G-CONTENT-BEATS-POSITION-ONLY` | `attention_terminal` score가 `position_only`보다 tolerance 이상 높음 | 위치-only confound 처리 |
| `S3B-G-CONTENT-BEATS-RANDOM` | `attention_terminal` score가 `random_terminal`보다 tolerance 이상 높음 | random terminal 반론 처리 |
| `S3B-G-POSITION-ABLATION-DROP` | `attention_terminal` score가 `attention_no_position`보다 tolerance 이상 높음 | 위치 보조 구조의 필요성 확인 |
| `S3B-G-SHUFFLED-POSITION-DROP` | `attention_terminal` score가 `attention_shuffled_position`보다 tolerance 이상 높음 | 정확한 위치 정렬 민감도 확인 |
| `S3B-G-ANCHOR-SANITY` | gold anchor oracle이 predicted anchor보다 크게 낮지 않음 | S3a anchor anomaly 재점검 |

측정값은 기존 `eval_loss`, `token_f1`, `rouge_l_f1`, `keyword_recall`, `skeleton_coverage`, `nonempty`에 `entity_recall`과 `repetition_rate`를 추가한다. `score`는 Token F1, ROUGE-L, keyword/entity recall, loss bonus를 합치고 repetition penalty를 작게 뺀 진단용 종합값이다.

