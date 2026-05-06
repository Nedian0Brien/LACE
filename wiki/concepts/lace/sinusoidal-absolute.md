---
title: "sinusoidal_absolute"
type: "concept"
tags: [LACE, positional-encoding, sinusoidal-absolute, 위치부호화]
created: 2026-05-06
updated: 2026-05-06
sources: [docs/v2/experiments/s2a-positional-encoding.md, wiki/concepts/lace/s2a-positional-encoding.md]
---

# sinusoidal_absolute

`sinusoidal_absolute`는 token의 원래 절대 위치를 사인/코사인 파형 값으로 바꾼 뒤, token embedding에 더하는 위치 부호화 방식이다.

직관적으로는 각 token에 "나는 원문에서 몇 번째 근처에 있던 단서다"라는 좌표표를 붙이는 것이다. [[concepts/lace/위치-보조-구조|위치 보조 구조]]가 `front/middle/back`처럼 세 구간만 알려주는 방식이라면, `sinusoidal_absolute`는 원래 token index를 더 촘촘한 숫자 패턴으로 표현한다.

## 이름의 의미

`sinusoidal`은 위치 번호를 여러 주기의 사인/코사인 값으로 표현한다는 뜻이다. 이 값들은 학습으로 새로 만드는 표가 아니라, 정해진 수식으로 계산되는 고정 좌표다.

`absolute`는 선택된 skeleton token들 사이의 상대 거리만 보는 것이 아니라, 원문 안에서의 절대 token index를 쓴다는 뜻이다. 즉 "선택된 token 목록에서 세 번째"가 아니라 "원래 문장에서 37번째 근처"라는 정보를 준다.

## LACE에서의 역할

LACE 관점에서는 [[concepts/lace/의미-골격|의미 골격]] token이 "무슨 내용인가"를 담당하고, `sinusoidal_absolute`는 "그 내용이 원래 문장 흐름의 어디쯤 있었는가"를 보조한다.

따라서 이 방식은 decoder가 의미 골격을 다시 문장으로 펼칠 때 다음 질문에 답하도록 돕는다.

- 어떤 단서가 앞쪽 배경 설명에 가까운가
- 어떤 단서가 중간 전개에 가까운가
- 어떤 단서가 뒤쪽 결론이나 후속 설명에 가까운가
- 선택된 skeleton token들 사이에 대략 어떤 순서 흐름이 있었는가

## S2a에서의 해석

[[concepts/lace/s2a-positional-encoding|S2a-positional encoding]]에서는 `sinusoidal_absolute`가 가장 좋은 위치 표현 후보로 식별됐다.

```text
sinusoidal_absolute: Token F1 0.1661, ROUGE-L 0.1509, loss 6.0715
coarse_bins:         Token F1 0.1533, ROUGE-L 0.1425, loss 6.0901
no_position:         Token F1 0.1218, ROUGE-L 0.1109, loss 6.1323
```

이는 S2의 `front/middle/back` coarse tag보다 더 정석적인 위치 부호화가 S3 후보로 쓸 만하다는 뜻이다.

## 주의점

`sinusoidal_absolute`가 S2a에서 가장 좋았다고 해서, 이것이 LACE의 최종 위치 표현이라는 뜻은 아니다. S2a는 lightweight probe였고, 생성 샘플에는 여전히 반복과 낮은 skeleton coverage가 남아 있었다.

따라서 현재의 방어 가능한 해석은 다음 정도다.

> S3에서는 `sinusoidal_absolute`를 기본 위치 보조 구조 후보로 사용할 수 있다. 다만 `coarse_bins`와의 차이가 작으므로, 가능하면 둘을 함께 절제 실험으로 남긴다.
