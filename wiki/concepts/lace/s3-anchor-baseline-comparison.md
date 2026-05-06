---
title: "S3-anchor baseline comparison"
type: "concept"
tags: [LACE, S3, anchor-baseline, terminal-state, reverse-trajectory]
created: 2026-05-06
updated: 2026-05-06
sources: [docs/v2/experiments/s3-anchor-baseline-comparison.md, outputs/v2_s3/lace_v2_s3/summary.md]
---

# S3-anchor baseline comparison

S3-anchor baseline comparison은 중요한 token을 forward terminal state로 직접 보존하는 방식이, random forward 뒤에 anchor를 예측해서 붙이는 방식보다 더 좋은 reverse trajectory를 만드는지 확인한 실험이다.

핵심 비교는 다음 네 조건이었다.

- `random_forward_anchor_prediction`
- `importance_ordered_forward_no_anchor`
- `importance_ordered_forward_anchor_prediction`
- `random_forward_no_anchor`

## 핵심 결과

S3는 `overall_pass=false`, `s4_ready=false`였다.

```text
random_forward_anchor_prediction:       score 0.4418, Token F1 0.1559, ROUGE-L 0.1460
importance_ordered_forward_no_anchor:   score 0.4356, Token F1 0.1515, ROUGE-L 0.1441
importance_ordered_forward_anchor_pred: score 0.4023, Token F1 0.1366, ROUGE-L 0.1252
random_forward_no_anchor:               score 0.4467, Token F1 0.1612, ROUGE-L 0.1455
```

가장 중요한 실패 신호는 `importance_ordered_forward_no_anchor`가 `random_forward_no_anchor`보다 낮았다는 점이다.

## 해석

이 결과는 [[concepts/lace/의미-골격|의미 골격]] terminal state가 현재 S3 설정에서 random terminal state보다 더 좋은 복원 궤적을 만든다는 증거를 제공하지 못했다.

다만 `importance_ordered_forward_no_anchor`는 `random_forward_anchor_prediction`과 tolerance 안에서 비슷했다. 따라서 anchor prediction baseline이 semantic terminal state를 압도했다고 볼 수도 없다.

현재의 방어 가능한 해석은 다음이다.

> S3는 v2 핵심 주장을 강화하지 못했다. 다음 단계는 S4가 아니라 terminal 정보량과 위치 편향, anchor predictor 병목을 분리하는 S3a 진단 실험이어야 한다.

## Anchor predictor 품질

Anchor predictor 자체도 약했다.

```text
importance terminal -> anchor: Token F1 0.0447, ROUGE-L 0.0443
random terminal -> anchor:     Token F1 0.0182, ROUGE-L 0.0182
```

따라서 S3의 anchor 조건은 강한 anchor prediction baseline이 아니다. Anchor predictor가 약했는데도 random 조건이 높게 나온 점은, 현재 reverse model 또는 reconstruction proxy가 terminal 정보량 차이를 충분히 드러내지 못했을 가능성을 남긴다.

## 다음 연결

다음 후보는 `S3a-terminal diagnostic`이다. 여기서는 다음을 비교해야 한다.

- `attention_terminal`
- `idf_terminal`
- `random_terminal`
- `same_position_random_terminal`
- `position_only`
- `gold_anchor_oracle`
- `predicted_anchor`

관련 지표 해석은 [[concepts/lace/복원-평가지표-token-f1-rouge-l|복원 평가 지표 - Token F1과 ROUGE-L]]에 정리되어 있다.
