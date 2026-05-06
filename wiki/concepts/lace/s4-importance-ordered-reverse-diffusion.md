---
title: "S4-importance ordered reverse diffusion"
type: "concept"
tags: [LACE, v2, S4, reverse-diffusion, semantic-skeleton]
created: 2026-05-06
updated: 2026-05-06
sources: [docs/v2/experiments/s4-importance-ordered-reverse-diffusion.md, outputs/v2_s4/lace_v2_s4/summary.md]
---

# S4-importance ordered reverse diffusion

S4는 [[forward-reverse-process-본질|Forward-Reverse Process 본질]]을 직접 실험한 첫 process-level 실험이다. 문장 전체를 한 번에 복원하는 것이 아니라, `25% -> 50% -> 75% -> 100%` reverse transition을 학습해 [[의미-골격|semantic skeleton]]에서 세부 token을 단계적으로 붙이는 curriculum을 비교했다.

## 핵심 결과

`process_ready=true`, `overall_pass=false`, `s5_ready=false`였다.

| Schedule | Score | Token F1 | ROUGE-L | Target Content | Input Retention | Expansion Recall | Entity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `importance_schedule` | 0.4839 | 0.1754 | 0.1332 | 0.0574 | 0.0471 | 0.0416 | 0.0831 |
| `random_schedule` | 0.5607 | 0.2300 | 0.1712 | 0.0300 | 0.0277 | 0.0203 | 0.0412 |
| `position_only_schedule` | 0.3752 | 0.1368 | 0.1052 | 0.0191 | 0.0000 | 0.0191 | 0.0323 |

종합 score와 표면 복원 지표에서는 `random_schedule`이 이겼다. 하지만 의미 보존과 확장 지표에서는 `importance_schedule`이 모두 높았다.

## 해석

S4는 importance-ordered reverse diffusion이 random corruption보다 더 좋은 language model trajectory라는 결론을 주지는 못했다. `random_schedule`은 loss, Token F1, ROUGE-L에서 더 강했다.

하지만 `importance_schedule`은 target content recall, input retention, expansion recall, original content recall, entity recall이 모두 random보다 높았다. 이는 중심 의미 token을 먼저 남기는 schedule이 semantic retention과 semantic expansion에는 더 유리할 수 있음을 보여준다.

따라서 S4의 결론은 실패라기보다 분화된 결과다.

```text
random: surface reconstruction 우위
importance: semantic retention / expansion 우위
```

## 다음 방향

다음은 S5 open-ended generation이 아니라 `S4a: delta-token reverse objective`다. 현재 objective는 target state 전체를 다시 생성하기 때문에 입력 skeleton 유지와 새 token 확장이 섞인다. 다음에는 새로 unmask될 token/span만 예측하도록 바꿔야 한다.
