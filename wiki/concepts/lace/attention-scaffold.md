---
title: "attention_scaffold"
type: "concept"
tags: [LACE, S2, attention-scaffold, 의미골격]
created: 2026-05-06
updated: 2026-05-06
sources: [kaggle/v2_s2/run_v2_s2.py, docs/v2/experiments/s2-skeleton-to-text-reconstruction.md, 사용자 대화]
---

# attention_scaffold

`attention_scaffold`는 S2 의미 골격-문장 복원 학습에서 사용한 핵심 입력 조건이다.

구조는 다음 두 부분으로 이뤄진다.

1. attention 기반 [[concepts/lace/의미-골격|의미 골격]]
2. 선택된 token들의 대략적 [[concepts/lace/위치-보조-구조|위치 보조 구조]]

실제 입력은 다음 형식으로 만들어진다.

```text
restore text | condition: attention_scaffold | positions: front front middle back ... | skeleton: <attention으로 선택된 token들>
```

## 작동 방식

먼저 원문을 frozen `t5-small` encoder에 넣고 모든 layer의 attention을 얻는다. 각 token이 다른 token들로부터 얼마나 많은 attention을 받았는지 평균하여 `attention_received` 점수를 만든다.

그다음 special token과 padding을 제외한 유효 token 중 상위 25%를 고른다. 이 token들이 `skeleton` 부분이 된다.

마지막으로 선택된 token들이 원문에서 앞, 중간, 뒤 중 어디에 있었는지 `front`, `middle`, `back` bin으로 바꾼다. 이 정보가 `positions` 부분이다.

따라서 `attention_scaffold`는 token 내용만 주는 입력이 아니다. 내용 단서와 위치 단서를 함께 주는 복원 입력이다.

## front/middle/back 위치 표현의 성격

`front`, `middle`, `back`으로 위치를 넣는 방식은 일반적인 표준 위치 부호화 방식이라기보다, S2에서 사용한 거친 위치 보조 구조다.

일반적인 transformer 모델에서는 token index에 대한 learned positional embedding, sinusoidal positional encoding, relative position bias, rotary position embedding 같은 방식이 더 흔하다. 이런 방식들은 모델 내부 표현에 위치 정보를 직접 넣는다.

반면 S2의 `front/middle/back`은 모델 내부 위치 부호화가 아니라, 입력 문자열에 사람이 읽을 수 있는 coarse 위치 tag를 붙인 것이다. 목적은 정교한 위치 모델링이 아니라 다음 두 가지였다.

- 의미 골격 token들이 원문에서 대략 어느 구간에 있었는지 알려준다.
- `position_only`, `same_position_random`, `wrong_document` control을 쉽게 만들 수 있게 한다.

따라서 S2 결과를 해석할 때는 `front/middle/back`을 일반적인 positional encoding으로 보면 안 된다. 이것은 실험용 위치 scaffold이며, 다음 단계에서는 더 세밀한 상대 위치, 원래 token index, gap 크기, learned positional scaffold 같은 대안을 비교할 수 있다.

## S2에서의 의미

S2에서 `attention_scaffold`는 `random_scaffold`보다 Token F1과 ROUGE-L이 높았다.

```text
attention_scaffold: Token F1 0.3830, ROUGE-L 0.3117
random_scaffold:    Token F1 0.2286, ROUGE-L 0.1789
```

이는 attention 기반 의미 골격 + 위치 보조 구조가 같은 token 수의 무작위 골격보다 더 좋은 복원 입력이었다는 뜻이다.

단, 이 결과만으로 attention scorer가 최종적으로 가장 좋다고 말하면 안 된다. S2에서는 `idf_scaffold`도 강했고, `position_prior_scaffold`도 일부 지표에서 강했다. 현재 방어 가능한 해석은 중요도 기반 의미 골격 계열이 무작위 골격보다 유리하다는 것이다.

## 관련 개념

- [[concepts/lace/복원-평가지표-token-f1-rouge-l|복원 평가 지표 - Token F1과 ROUGE-L]]
- [[concepts/lace/s2-의미-골격-문장-복원-학습|S2 의미 골격-문장 복원 학습]]
