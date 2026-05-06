---
title: "Forward-Reverse Process 본질"
type: "concept"
tags: [LACE, v2, diffusion-language-model, forward-process, reverse-process]
created: 2026-05-06
updated: 2026-05-06
sources: [사용자 대화, docs/v2/experiments/s3b-probe-calibration.md]
---

# Forward-Reverse Process 본질

LACE v2의 본질은 문장을 그대로 복원할 수 있느냐가 아니다. 핵심 질문은 [[의미-골격|문장의 핵심 토큰]]을 남기고 중요성이 떨어지는 토큰을 순차적으로 masking하는 forward process를 정의했을 때, 그 역과정을 학습하는 [[위치-보조-구조|reverse process]]가 random corruption 기반 diffusion보다 더 좋은 language model을 만들 수 있느냐이다.

## 핵심 직관

일반적인 random corruption은 문장의 정보 구조를 고려하지 않고 token을 손상시킨다. 반면 LACE의 forward process는 중요도가 낮은 token부터 제거하고, 마지막에는 중심 의미를 담는 뼈대 token이 남는다.

Reverse process는 이 궤적을 거꾸로 따라간다. 모델은 먼저 문장의 중심 의미를 구성하는 token 또는 semantic skeleton을 다루고, 이후 세부 의미와 표면 token을 단계적으로 붙여 문장을 확장한다.

```text
forward: x_0 -> low-importance masked -> ... -> semantic skeleton
reverse: semantic skeleton -> core expansion -> detail expansion -> x_0-like text
```

## S3 계열의 위치

[[s3-anchor-baseline-comparison|S3]], [[s3a-terminal-diagnostic|S3a]], [[s3b-probe-calibration|S3b]]는 최종 목표가 아니라 측정 장치 점검이었다. 이 실험들은 attention 기반 terminal이 완전히 무의미하지는 않지만, exact reconstruction probe와 lexical metric만으로는 연구 본질을 충분히 평가하지 못한다는 점을 드러냈다.

따라서 이후의 초점은 `attention_terminal` score를 조금 더 올리는 것이 아니라, importance-ordered forward schedule과 reverse expansion objective를 직접 구현하고 random masking schedule과 비교하는 것이다.

## 다음 실험 설계 기준

다음 실험은 다음 질문에 답해야 한다.

- 중요도 기반 masking 순서가 random masking 순서보다 더 나은 reverse curriculum을 만드는가?
- reverse model이 skeleton에서 세부 token으로 의미를 안정적으로 확장하는가?
- 확장 과정에서 중심 의미가 유지되고 semantic drift가 줄어드는가?
- 같은 model budget에서 random corruption diffusion보다 반복이 적고, coherence와 controllability가 좋은가?

평가는 원문 exact reconstruction만으로 하지 않는다. trajectory coherence, semantic drift, skeleton faithfulness, generation quality, repetition/diversity를 함께 봐야 한다.

