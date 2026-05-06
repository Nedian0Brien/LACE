# S3 계획: anchor baseline comparison

## 1. 검증할 질문

S3는 다음 질문을 검증한다.

> 중요한 token을 forward terminal state로 직접 보존하는 방식이, random forward 뒤에 중요 anchor를 예측해서 보조 조건으로 붙이는 방식보다 더 좋은 reverse trajectory를 만드는가?

이 질문은 v2 핵심 주장인 "semantic skeleton + positional scaffold가 random corruption보다 더 좋은 reverse trajectory를 만든다"를 anchor prediction baseline과 직접 비교한다.

## 2. 실험 이름

```text
S3-anchor baseline comparison
```

Kaggle phase 이름은 `v2_s3`로 둔다.

## 3. 핵심 비교 조건

S3는 같은 복원 모델 구조와 같은 학습 예산에서 다음 네 조건을 비교한다.

| 기호 | 조건 | 의미 |
|---|---|---|
| A | `random_forward_anchor_prediction` | random terminal skeleton을 만들고, 그 terminal에서 중요 anchor를 예측한 뒤 reverse 입력에 붙인다. |
| B | `importance_ordered_forward_no_anchor` | attention 기반 semantic skeleton을 terminal state로 직접 보존하고, 별도 anchor prediction은 쓰지 않는다. |
| C | `importance_ordered_forward_anchor_prediction` | attention 기반 terminal state에 예측 anchor를 추가로 붙인다. |
| D | `random_forward_no_anchor` | random terminal skeleton만 사용한다. |

S2a 결과에 따라 기본 위치 보조 구조는 [[../../wiki/concepts/lace/sinusoidal-absolute|sinusoidal_absolute]]로 둔다. 단, S2a에서 `coarse_bins`와의 차이가 작았으므로 S3 결과 해석에서는 위치 표현 선택이 결론을 과도하게 좌우할 수 있다는 caveat를 유지한다.

## 4. anchor prediction의 운영 정의

`anchor prediction` 조건은 gold anchor를 직접 붙이지 않는다.

절차는 다음이다.

1. frozen `t5-small` encoder attention으로 각 원문의 importance skeleton을 만든다.
2. 같은 token 수로 random terminal skeleton을 만든다.
3. terminal skeleton에서 importance skeleton token sequence를 예측하는 작은 anchor predictor를 학습한다.
4. reverse model 입력에는 gold anchor가 아니라 predictor가 생성한 anchor token을 붙인다.

이렇게 해야 `random + anchor` 조건이 원문에서 뽑은 정답 anchor를 훔쳐 쓰는 oracle baseline이 되지 않는다.

## 5. 평가 지표

S3의 주 지표는 S2/S2a와 같은 복원 proxy를 유지한다.

- teacher-forced eval loss
- perplexity
- Token F1
- ROUGE-L
- keyword recall
- skeleton coverage
- nonempty generation rate

추가로 anchor predictor 자체의 품질을 기록한다.

- anchor Token F1
- anchor ROUGE-L
- predicted anchor nonempty rate

## 6. Gate

| Gate | 의미 | 통과 기준 |
|---|---|---|
| `S3-G-RUN` | 네 조건이 모두 실행됐는가 | A/B/C/D 모두 metric 존재 |
| `S3-G-LOSS-FINITE` | 주요 loss가 정상 숫자인가 | 모든 조건의 eval loss가 finite |
| `S3-G-IMPORTANCE-BEATS-RANDOM` | semantic terminal이 random terminal보다 나은가 | B score > D score |
| `S3-G-TERMINAL-NOT-WORSE-THAN-ANCHOR` | terminal 보존 방식이 anchor prediction baseline보다 뒤처지지 않는가 | B score + tolerance >= A score |
| `S3-G-BEST-IDENTIFIED` | 다음 단계 후보가 식별됐는가 | 최고 조건 존재 |

여기서 score는 `Token F1 + ROUGE-L + 1 / (1 + eval_loss)`로 계산한다. S2a와 같은 이유로, 초기 generation 품질이 약할 때 teacher-forced loss가 완전히 무시되지 않게 하기 위한 보조 점수다.

## 7. 좋은 결과와 나쁜 결과

좋은 결과는 B가 A보다 좋거나 거의 비슷하고, D보다 명확히 좋은 경우다. 이는 semantic skeleton을 terminal state로 직접 보존하는 것이 anchor 예측을 덧붙이는 baseline보다 단순하고 해석 가능한 trajectory라는 주장을 강화한다.

C가 가장 좋으면 semantic skeleton terminal state와 anchor prediction이 상보적이라는 후속 방향이 생긴다. 이 경우 핵심 주장은 "skeleton forward가 anchor prediction을 대체한다"보다 "skeleton forward가 anchor prediction의 더 좋은 기반이 된다"로 조정해야 한다.

나쁜 결과는 A가 B보다 뚜렷하게 좋은 경우다. 이 경우 random corruption + anchor prediction만으로도 충분한 복원 단서가 생긴다는 뜻이므로, LACE의 terminal skeleton 보존 주장은 약해진다.

## 8. 해석 주의점

S3는 여전히 짧은 복원 probe다. 결과가 좋아도 open-ended generation 성공을 뜻하지 않는다.

또한 anchor predictor는 작은 보조 모델이므로, anchor prediction baseline의 최종 상한을 대표하지 않는다. A가 약하더라도 "모든 anchor prediction 방식이 약하다"는 결론은 내리지 않는다.

## 9. 산출물

| 산출물 | 위치 |
|---|---|
| Kaggle runner | `kaggle/v2_s3/run_v2_s3.py` |
| Kaggle metadata | `kaggle/v2_s3/kernel-metadata.json` |
| push script | `scripts/push_kaggle_v2_s3.sh` |
| output | `outputs/v2_s3/` |
| result doc | `docs/v2/experiments/s3-anchor-baseline-comparison.md` |
