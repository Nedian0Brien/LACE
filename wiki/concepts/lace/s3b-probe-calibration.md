---
title: "S3b-probe calibration"
type: "concept"
tags: [LACE, v2, S3b, probe-calibration, 의미골격]
created: 2026-05-06
updated: 2026-05-06
sources: [docs/v2/experiments/s3b-probe-calibration.md, outputs/v2_s3b/lace_v2_s3b/summary.md]
---

# S3b-probe calibration

S3b는 [[s3a-terminal-diagnostic|S3a-terminal diagnostic]] 이후 남은 probe confound를 줄이기 위한 실험이다. 핵심은 reverse model을 조건별로 새로 학습하지 않고, `attention_terminal` 입력으로 한 번만 학습한 뒤 평가 입력만 바꾸는 것이다.

## 핵심 결과

`diagnostic_ready=true`, `s4_ready=false`였다.

| 조건 | Score | Token F1 | ROUGE-L | 해석 |
|---|---:|---:|---:|---|
| `attention_terminal` | 0.3768 | 0.1537 | 0.1399 | 학습 입력과 같은 in-distribution content terminal |
| `attention_no_position` | 0.2828 | 0.0948 | 0.0910 | 위치값을 모두 0으로 만들면 크게 하락 |
| `attention_shuffled_position` | 0.3799 | 0.1551 | 0.1414 | 최고 조건. 정확한 위치 정렬 주장은 약함 |
| `position_only` | 0.3735 | 0.1586 | 0.1439 | content 없이도 attention terminal과 거의 같음 |
| `random_terminal` | 0.3724 | 0.1541 | 0.1363 | random terminal 반론이 여전히 남음 |
| `same_position_random_terminal` | 0.3638 | 0.1505 | 0.1341 | 낮지만 tolerance 0.02 이상 차이는 아님 |

## 해석

S3b가 보여준 긍정 신호는 위치 channel의 존재가 중요하다는 점이다. `attention_no_position`은 크게 하락했으므로, 현재 probe가 위치 정보를 완전히 무시하는 것은 아니다.

하지만 더 중요한 결론은 아직 [[의미-골격|의미 골격]] content 사용 증거가 약하다는 점이다. `attention_terminal`은 `position_only`, `random_terminal`, `same_position_random_terminal`보다 충분히 높지 않았다. 특히 `position_only`가 거의 같은 score를 얻은 것은 [[위치-보조-구조|위치 보조 구조]]와 decoder prior가 lexical metric 상당 부분을 설명할 수 있다는 뜻이다.

`attention_shuffled_position`이 최고였다는 점도 caveat다. 위치값을 모두 제거하면 나빠지지만, 선택된 위치값 사이의 정렬을 섞어도 나빠지지 않았다. 따라서 현재 실험에서 확인된 것은 "정확한 token-position alignment"라기보다 "위치 channel 또는 위치값 분포가 도움이 된다"에 가깝다.

## 다음 방향

S3b 이후에는 S4 constrained generation으로 바로 넘어가지 않는다. 다음은 반복을 줄이는 constrained reconstruction 설정, position-only 분해 control, terminal content-use metric 강화를 포함하는 S3c 성격의 보정 실험이 적절하다. Anchor 조건은 S3b에서 오히려 성능을 낮췄으므로 당분간 핵심 경로에서 내린다.

