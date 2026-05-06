---
title: "S3a-terminal diagnostic"
type: "concept"
tags: [LACE, S3a, terminal-diagnostic, position-control, anchor-oracle]
created: 2026-05-06
updated: 2026-05-06
sources: [docs/v2/experiments/s3a-terminal-diagnostic.md, outputs/v2_s3a/lace_v2_s3a/summary.md]
---

# S3a-terminal diagnostic

S3a-terminal diagnostic은 [[concepts/lace/s3-anchor-baseline-comparison|S3-anchor baseline comparison]] 이후 `random_forward_no_anchor`가 강했던 이유를 분해한 진단 실험이다.

핵심 비교는 [[concepts/lace/의미-골격|의미 골격]] terminal, IDF terminal, random terminal, same-position random terminal, [[concepts/lace/위치-보조-구조|위치 보조 구조]]만 있는 `position_only`, predicted anchor, gold-anchor oracle이었다.

## 핵심 결과

S3a는 `diagnostic_ready=true`, `s4_ready=false`였다.

```text
attention_terminal:                score 0.4351, Token F1 0.1540, ROUGE-L 0.1406
idf_terminal:                      score 0.4192, Token F1 0.1437, ROUGE-L 0.1359
random_terminal:                   score 0.4014, Token F1 0.1354, ROUGE-L 0.1262
same_position_random_terminal:     score 0.3429, Token F1 0.1066, ROUGE-L 0.0969
position_only:                     score 0.4275, Token F1 0.1534, ROUGE-L 0.1349
random_terminal_predicted_anchor:  score 0.4489, Token F1 0.1613, ROUGE-L 0.1476
random_terminal_gold_anchor_oracle: score 0.4115, Token F1 0.1482, ROUGE-L 0.1224
```

## 해석

긍정 신호는 `attention_terminal`이 `random_terminal`과 `same_position_random_terminal`보다 높았다는 점이다. S3에서 보였던 random terminal의 강함은 S3a에서는 약해졌다.

하지만 `position_only`가 `attention_terminal`과 매우 가까웠다. 따라서 content terminal 사용 신호는 있지만, 위치 scaffold와 모델 prior confound가 아직 크다.

또한 `random_terminal_predicted_anchor`가 최고였지만, anchor predictor 품질은 낮았다. random terminal anchor Token F1은 0.0155에 불과했다. 따라서 predicted anchor가 의미 anchor를 잘 예측했다기보다, predicted anchor 문자열이 표면 prior 또는 regularization처럼 작동했을 가능성이 크다.

## 다음 연결

다음 후보는 `S3b-probe calibration`이다. 여기서는 같은 학습 모델에 평가 입력만 바꾸는 ablation, gold anchor 길이/segment ablation, position-only matched control, 반복률과 entity recall 같은 metric을 추가해 현재 reverse probe가 무엇을 실제로 쓰는지 확인해야 한다.
