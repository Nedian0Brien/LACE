---
title: "S4c span-infilling reverse decoder"
type: "concept"
tags: [LACE, v2, S4c, infilling, decoder]
created: 2026-05-06
updated: 2026-05-06
sources: [docs/v2/experiments/s4c-span-infilling-reverse-decoder.md, outputs/v2_s4c/lace_v2_s4c/summary.md]
---

# S4c span-infilling reverse decoder

S4c는 [[concepts/lace/s4a-delta-token-reverse-objective|S4a]]와 [[concepts/lace/s4b-multi-step-delta-rollout|S4b]]의 autoregressive delta decoder에서 생긴 반복과 표면 token 편향을 줄이기 위해 시도한 marker-position infilling 구조다.

각 새 위치에 marker를 놓고, encoder hidden state에서 vocab classifier가 원래 token id를 직접 예측한다.

## 결과

`importance_schedule`은 random보다 masked-token accuracy가 높았지만, [[concepts/lace/위치-보조-구조|위치 보조 구조]]만 남긴 `position_only_schedule`과 같은 accuracy를 냈다.

| 지표 | importance | random | position-only |
|---|---:|---:|---:|
| score | 0.2403 | 0.1425 | 0.3026 |
| masked-token accuracy | 0.1414 | 0.1121 | 0.1414 |
| content recall | 0.0000 | 0.0000 | 0.0000 |
| entity recall | 0.0000 | 0.0000 | 0.0000 |
| duplicate prediction rate | 0.9347 | 0.9347 | 0.9347 |

## 해석

S4c는 `process_ready=true`였지만 `overall_pass=false`다. 현재 구조는 [[concepts/lace/의미-골격|의미 골격]]의 content를 쓰기보다 위치, transition 단계, punctuation/whitespace token 분포를 먼저 학습했다.

특히 content/entity recall이 모두 0이라는 점 때문에 semantic skeleton use 증거로 방어할 수 없다. Sample에서도 쉼표, 공백 token, 짧은 subword 예측이 많았다.

## 다음 의미

S4c는 그대로 확장할 모델이 아니라 실패 진단이다. 다음 구조는 content/entity weighted objective, punctuation/whitespace 분리 metric, contiguous span infilling decoder, same-position/wrong-document control을 포함해야 한다.
