## 2026-05-06 query | sinusoidal_absolute 설명

- 질문: `sinusoidal_absolute`가 무엇인지 설명.
- 답변 요약: token의 원래 절대 위치를 사인/코사인 파형 값으로 바꿔 token embedding에 더하는 위치 부호화 방식이며, LACE에서는 의미 골격 token에 원문 위치 좌표를 붙여 복원 흐름을 돕는 역할로 해석했다.
- 반영 문서: [[concepts/lace/s2a-positional-encoding]]

## 2026-05-06 query | sinusoidal_absolute 독립 문서화

- 질문: `sinusoidal_absolute` 문서를 별도로 만들기.
- 답변 요약: `sinusoidal_absolute`를 독립 개념 페이지로 분리하고, S2a 실험 문서에서는 해당 페이지로 링크하도록 정리했다.
- 반영 문서: [[concepts/lace/sinusoidal-absolute]], [[concepts/lace/s2a-positional-encoding]]

## 2026-05-06 query | S3 실험 진행

- 질문: S3 실험에 들어가기.
- 답변 요약: `S3-anchor baseline comparison`을 계획, 구현, Kaggle 실행, 결과 다운로드까지 진행했다. 결과는 `overall_pass=false`, `s4_ready=false`였고, 다음 단계는 S3a terminal diagnostic으로 정리했다.
- 반영 문서: [[concepts/lace/s3-anchor-baseline-comparison]]

## 2026-05-06 query | S3 이후 부족한 점

- 질문: 본격적인 연구 방법론을 고민하기 전에, 현재 부족한 것이 무엇인지 정리.
- 답변 요약: S3 실패는 가설 폐기가 아니라 terminal state 정보량, 위치 편향, anchor predictor 병목, lexical metric 민감도, scorer 선택 문제가 아직 분리되지 않았다는 신호로 해석했다.
- 반영 문서: [[concepts/lace/s3-이후-방법론-부족점]]

## 2026-05-06 query | S3a 연구 진행

- 질문: A안으로 S3a terminal diagnostic 진행.
- 답변 요약: `S3a-terminal diagnostic`을 계획, 구현, Kaggle 실행, 결과 다운로드까지 진행했다. `diagnostic_ready=true`, `s4_ready=false`였고, attention terminal은 random/same-position control보다 높았지만 position-only와 predicted-anchor confound가 남았다.
- 반영 문서: [[concepts/lace/s3a-terminal-diagnostic]]

## 2026-05-06 query | S3b 연구 진행

- 질문: S3b probe calibration 진행.
- 답변 요약: `S3b-probe calibration`을 계획, 구현, Kaggle 실행, 결과 다운로드까지 진행했다. 같은 reverse probe에서 `attention_no_position`은 크게 낮아졌지만 `attention_terminal`은 position-only/random/same-position random보다 tolerance 이상 높지 않아 S4 진입은 보류했다.
- 반영 문서: [[concepts/lace/s3b-probe-calibration]]

## 2026-05-06 query | 연구 본질 재정렬

- 질문: 문장 exact reconstruction보다 중요도 기반 forward masking과 reverse expansion으로 더 나은 diffusion language model을 만들 수 있느냐가 본질이라는 문제 제기.
- 답변 요약: S3 계열은 측정 장치 점검으로 한정하고, 이후 초점을 importance-ordered masking schedule과 reverse expansion objective의 process-level 비교로 재정렬했다.
- 반영 문서: [[concepts/lace/forward-reverse-process-본질]]

## 2026-05-06 query | S4 연구 진행

- 질문: S4로 진행.
- 답변 요약: `S4-importance ordered reverse diffusion`을 계획, 구현, Kaggle 실행, 결과 다운로드까지 진행했다. `random_schedule`은 종합 score와 표면 복원에서 이겼지만, `importance_schedule`은 의미 보존/확장 지표에서 더 강했다. 다음은 `S4a: delta-token reverse objective`로 정리했다.
- 반영 문서: [[concepts/lace/s4-importance-ordered-reverse-diffusion]]

## 2026-05-06 query | S4a 연구 진행

- 질문: S4a 실험을 진행하고, 결과가 좋지 않으면 구조 개선 방안을 탐구.
- 답변 요약: `S4a-delta token reverse objective`를 계획, 구현, Kaggle 실행, 결과 다운로드까지 진행했다. 전체 state가 아니라 delta token/span만 예측하도록 바꾸자 `importance_schedule`이 random과 position-only를 모두 이겼다. 다만 entity recall과 repetition은 남은 병목으로 기록했다.
- 반영 문서: [[concepts/lace/s4a-delta-token-reverse-objective]]

## 2026-05-06 query | S4b/S4c 병렬 진행

- 질문: S4b와 S4c를 동시에 진행할 수 있을 것 같으니 서브에이전트 두 개로 병렬 수행.
- 답변 요약: S4b multi-step delta rollout과 S4c span-infilling reverse decoder를 병렬 구현, Kaggle 실행, 결과 다운로드까지 진행했다. S4b는 importance rollout이 random/position-only를 이겼고, S4c는 position-only confound와 content/entity collapse 때문에 실패 진단으로 정리했다.
- 반영 문서: [[concepts/lace/s4b-multi-step-delta-rollout]], [[concepts/lace/s4c-span-infilling-reverse-decoder]]
