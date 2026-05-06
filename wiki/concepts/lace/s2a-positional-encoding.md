---
title: "S2a-positional encoding"
type: "concept"
tags: [LACE, S2a, positional-encoding, 위치보조구조]
created: 2026-05-06
updated: 2026-05-06
sources: [docs/v2/experiments/s2a-positional-encoding.md, outputs/v2_s2a/lace_v2_s2a/summary.md]
---

# S2a-positional encoding

S2a-positional encoding은 [[concepts/lace/attention-scaffold|attention_scaffold]]의 의미 골격 token 선택은 고정하고, 위치 표현 방식만 비교한 실험이다.

비교 대상은 다음이다.

- `no_position`
- `coarse_bins`
- `learned_absolute`
- `sinusoidal_absolute`
- `relative_position_bias`
- `rotary_position`

## 핵심 결과

가장 좋은 조건은 `sinusoidal_absolute`였다.

```text
sinusoidal_absolute: Token F1 0.1661, ROUGE-L 0.1509, loss 6.0715
coarse_bins:         Token F1 0.1533, ROUGE-L 0.1425, loss 6.0901
no_position:         Token F1 0.1218, ROUGE-L 0.1109, loss 6.1323
```

따라서 S2a는 [[concepts/lace/위치-보조-구조|위치 보조 구조]]를 S2의 `front/middle/back` coarse tag보다 더 정석적인 방식으로 바꿀 수 있음을 보여준다.

## 해석 주의점

개선 폭은 작다. `sinusoidal_absolute`가 가장 좋았지만 `coarse_bins`와 큰 차이를 만든 것은 아니다.

또한 생성 샘플은 아직 반복이 많고 keyword recall과 skeleton coverage가 낮다. S2a는 위치 표현 후보를 고르는 probe이지, open-ended generation 성공 증거가 아니다.

## 다음 단계

S3 anchor baseline comparison에서는 `sinusoidal_absolute`를 기본 positional scaffold 후보로 둔다. 다만 `coarse_bins`와의 차이가 작으므로, 가능하면 S3에서도 둘을 함께 ablation으로 유지한다.

관련 지표 해석은 [[concepts/lace/복원-평가지표-token-f1-rouge-l|복원 평가 지표 - Token F1과 ROUGE-L]]에 정리되어 있다.
