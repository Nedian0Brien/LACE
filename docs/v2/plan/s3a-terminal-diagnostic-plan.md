# S3a 계획: terminal diagnostic

## 1. 검증할 질문

S3a는 다음 질문을 검증한다.

> S3에서 `random_forward_no_anchor`가 가장 높았던 이유는 terminal 정보량 차이가 약해서인가, 위치/길이/빈출 token prior 때문인가, 아니면 anchor predictor와 lexical metric이 병목이었기 때문인가?

S3a는 S4로 넘어가기 위한 성능 실험이 아니라, S3 실패 원인을 분해하는 진단 실험이다.

## 2. 실험 이름

```text
S3a-terminal diagnostic
```

Kaggle phase 이름은 `v2_s3a`로 둔다.

## 3. 핵심 비교 조건

S3a는 S3의 custom PyTorch encoder-decoder, `sinusoidal_absolute` 위치 보조 구조, 1 epoch reverse probe를 유지하되 terminal 조건을 진단형으로 넓힌다.

| 조건 | 의미 |
|---|---|
| `attention_terminal` | attention-received score로 고른 semantic terminal |
| `idf_terminal` | corpus-level IDF로 고른 semantic/statistical terminal |
| `random_terminal` | 같은 keep ratio의 random terminal |
| `same_position_random_terminal` | attention terminal과 같은 위치에 다른 문서 token을 넣은 control |
| `position_only` | token content 없이 attention 위치 scaffold만 제공 |
| `random_terminal_predicted_anchor` | random terminal에서 예측한 anchor를 붙인 S3식 baseline |
| `random_terminal_gold_anchor_oracle` | random terminal에 gold anchor를 직접 붙인 oracle upper bound |

## 4. 평가 지표

S3와 같은 복원 proxy를 유지한다.

- teacher-forced eval loss
- perplexity
- Token F1
- ROUGE-L
- keyword recall
- skeleton coverage
- nonempty generation rate
- 종합 score: `Token F1 + ROUGE-L + 1 / (1 + eval_loss)`

추가로 anchor predictor 자체의 품질을 기록한다.

- anchor Token F1
- anchor ROUGE-L
- predicted anchor nonempty rate

## 5. Gate

S3a gate는 다음 phase 진입 gate라기보다 원인 분해 gate다.

| Gate | 의미 | 통과 기준 |
|---|---|---|
| `S3A-G-RUN` | 모든 진단 조건이 실행됐는가 | 7개 조건 metric 존재 |
| `S3A-G-LOSS-FINITE` | 주요 loss가 정상 숫자인가 | 모든 조건 eval loss finite |
| `S3A-G-CONTENT-BEATS-SAME-POSITION` | content terminal이 같은 위치 random token보다 나은가 | best(`attention_terminal`, `idf_terminal`) score > `same_position_random_terminal` score |
| `S3A-G-CONTENT-BEATS-POSITION-ONLY` | token content가 위치만보다 나은가 | best content score > `position_only` score |
| `S3A-G-ORACLE-LIFT` | gold anchor가 predicted anchor보다 큰 상한을 보이는가 | oracle score > predicted score + tolerance |
| `S3A-G-RANDOM-EXPLAINED` | random 강함이 위치/random control로 설명되는가 | `random_terminal`이 best content보다 높을 때 same-position/position-only 근접 여부를 detail로 기록 |

`overall_pass`는 S3a에서 엄격한 성공 선언으로 쓰지 않는다. 대신 `diagnostic_ready=true`를 "필요한 비교가 실행되고 해석 가능한 수치가 나왔다"는 뜻으로 둔다.

## 6. 좋은 결과와 나쁜 결과

좋은 결과는 `attention_terminal` 또는 `idf_terminal`이 `same_position_random_terminal`, `position_only`, `random_terminal`보다 명확히 높은 것이다. 이 경우 S3 실패는 이전 조건 설계나 anchor 비교 방식이 둔감했던 것으로 해석할 수 있다.

`gold_anchor_oracle`이 크게 좋고 `random_terminal_predicted_anchor`가 약하면 anchor predictor 병목이다. 이 경우 다음 단계는 terminal claim 폐기가 아니라 anchor predictor 품질 개선 또는 oracle anchor upper bound 분리다.

나쁜 결과는 `gold_anchor_oracle`까지 약하거나, `position_only`와 `same_position_random_terminal`이 content terminal과 비슷한 경우다. 이 경우 현재 custom reverse probe와 lexical metric이 terminal 정보량 차이를 읽지 못할 가능성이 커진다.

## 7. 해석 주의점

S3a는 open-ended generation 실험이 아니다. 결과가 좋아도 S4/S5 generation 품질을 직접 증명하지 않는다.

S3a의 주 목적은 다음 중 어느 병목이 큰지 고르는 것이다.

- terminal scorer 문제
- random/position prior 문제
- anchor predictor 문제
- reverse model capacity 문제
- lexical metric 민감도 문제

## 8. 산출물

| 산출물 | 위치 |
|---|---|
| Kaggle runner | `kaggle/v2_s3a/run_v2_s3a.py` |
| Kaggle metadata | `kaggle/v2_s3a/kernel-metadata.json` |
| push script | `scripts/push_kaggle_v2_s3a.sh` |
| output | `outputs/v2_s3a/` |
| result doc | `docs/v2/experiments/s3a-terminal-diagnostic.md` |
