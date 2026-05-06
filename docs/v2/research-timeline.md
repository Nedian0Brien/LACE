# V2 연구 타임라인

이 문서는 v2 의미 골격 연구 흐름의 연구 질문, 결정, 경험적 사실, 주의점, 해석 변경을 시간순으로 기록한다.

## 2026-05-06

### 발견: S2a-positional encoding 비교 통과

추가 시각: 2026-05-06 10:09 KST

맥락:

S2의 `front/middle/back` 위치 tag는 실험용 coarse scaffold였으므로, S3로 넘어가기 전에 더 정석적인 positional encoding 후보를 비교했다. 의미 골격 token 선택은 attention 기반으로 고정하고, 위치 표현만 바꿨다.

결과:

S2a는 `overall_pass=true`, `s3_ready=true`로 통과했다. `sinusoidal_absolute`가 loss 6.0715, Token F1 0.1661, ROUGE-L 0.1509로 가장 좋은 후보였다. `coarse_bins`는 loss 6.0901, Token F1 0.1533, ROUGE-L 0.1425였고, `no_position`은 loss 6.1323, Token F1 0.1218, ROUGE-L 0.1109였다.

주의점:

`sinusoidal_absolute`가 가장 좋았지만 `coarse_bins` 대비 개선 폭은 작다. 또한 생성 샘플은 반복과 표면적 단어 겹침이 많고, keyword recall과 skeleton coverage가 낮다. 따라서 S2a는 위치 표현 후보를 고르는 probe이지, generation 품질 성공 증거가 아니다.

근거/출처:

- `outputs/v2_s2a/lace_v2_s2a/summary.md`
- `docs/v2/experiments/s2a-positional-encoding.md`

다음 실험에 주는 의미:

S3의 기본 위치 보조 구조 후보는 `sinusoidal_absolute`로 둔다. 다만 S3에서는 가능하면 `coarse_bins`도 ablation으로 유지해 위치 표현 개선 폭이 실제 anchor 비교에서도 유지되는지 확인한다.

### 결정: S3 전에 S2a-positional encoding을 수행한다

추가 시각: 2026-05-06 09:52 KST

맥락:

S2에서 사용한 `front`, `middle`, `back` 위치 tag는 정식 transformer positional encoding이라기보다 실험용 coarse scaffold였다. S3 anchor baseline comparison으로 넘어가기 전에 위치 보조 구조를 더 정교하게 만들 수 있는지 비교할 필요가 생겼다.

결정:

S3 전에 `S2a-positional encoding` 실험을 수행한다. 의미 골격 token 선택은 attention 기반으로 고정하고, 위치 표현만 `learned_absolute`, `sinusoidal_absolute`, `relative_position_bias`, `rotary_position`으로 비교한다. 해석용 baseline으로 `no_position`과 `coarse_bins`를 포함한다.

근거/출처:

- `docs/v2/experiments/s2-skeleton-to-text-reconstruction.md`
- `wiki/concepts/lace/attention-scaffold.md`
- `docs/v2/plan/s2a-positional-encoding-plan.md`

다음 실험에 주는 의미:

S2a에서 가장 좋은 위치 표현을 S3의 기본 positional scaffold 후보로 넘긴다. 실패하면 S3는 기존 coarse tag를 유지하되, 위치 보조 구조가 임시 구현이라는 caveat를 명시한다.

### 결정: v2 연구 진행 현황 페이지를 web/index.html에 둔다

추가 시각: 2026-05-06 09:48 KST

맥락:

연구 상태가 `docs/v2/research-timeline.md`, `docs/v2/experiment-roadmap.md`, `wiki/concepts/lace/`에 분산돼 있어 한눈에 "지금 어디까지 왔는가"를 파악하기 어려웠다. `AGENTS.md`는 진행 상황을 `web/index.html`에 업데이트하고 `design/design-system.html` 디자인 시스템을 따르도록 요구한다. 기존 `web/` 디렉터리는 비어 있었다.

결정:

`web/index.html`에 v2 트랙 단일 진입점 정적 페이지를 만든다. 페이지는 다음 섹션으로 구성한다 — hero (가설 한글 표기 + 영어 원문 병기), at-a-glance status strip, S0–S5 phase 카드 (done · pending · future 색 매핑), 핵심 개념 3카드 (의미 골격 · 위치 보조 구조 · 역방향 궤적), S1 검색형 사용 검증 metric bar chart, S2 복원 학습 small-multiple bar chart, 최근 timeline 7개, 남은 질문 4개 (위치 편향 · scorer 선택 · 생성 미검증 · S3 우선순위), footer. 모든 수치는 `outputs/v2_s1/lace_v2_s1/summary.md`, `outputs/v2_s2/lace_v2_s2/summary.md`, `research-timeline.md`의 raw 값을 그대로 인용한다. CSS 토큰은 `design/design-system.html`을 그대로 가져오고 다크/라이트 토글은 `localStorage` + `prefers-color-scheme`로 처리한다.

근거/출처:

- `AGENTS.md`
- `design/design-system.html`
- `docs/v2/research-timeline.md`
- `outputs/v2_s1/lace_v2_s1/summary.md`
- `outputs/v2_s2/lace_v2_s2/summary.md`

다음 실험에 주는 의미:

S2a positional encoding과 S3 anchor baseline이 끝나는 시점에 phase 카드와 timeline 항목, metric bar chart를 갱신한다. open-ended generation 주장으로 넘어가지 않는다는 caveat는 hero quote와 open question 섹션에 명시적으로 유지한다. 페이지는 정적 단일 파일이라 새 조건이 추가될 때 metric bar 한 줄을 늘리는 식으로 점진적으로 확장한다.

### 질문: front/middle/back 위치 보조 구조는 일반적인가

추가 시각: 2026-05-06 09:46 KST

맥락:

S2의 `attention_scaffold`는 attention 기반 의미 골격 token과 함께 `front`, `middle`, `back` 위치 tag를 입력 문자열에 넣었다. 이 방식이 일반적인 positional encoding인지, 아니면 실험용 보조 구조인지 구분할 필요가 생겼다.

정리:

`front`, `middle`, `back`은 일반적인 transformer 위치 부호화 방식이라기보다 S2에서 사용한 coarse 위치 보조 구조다. 일반적인 위치 부호화는 learned positional embedding, sinusoidal positional encoding, relative position bias, rotary position embedding처럼 모델 내부 표현에 위치 정보를 넣는 방식이 더 흔하다. S2의 방식은 입력 문자열에 대략적인 위치 구간을 붙이는 실험용 scaffold다.

근거/출처:

- `kaggle/v2_s2/run_v2_s2.py`
- `wiki/concepts/lace/attention-scaffold.md`
- `wiki/concepts/lace/위치-보조-구조.md`

다음 실험에 주는 의미:

S3 이후에는 `front/middle/back`을 최종 위치 표현으로 고정하지 않는다. 더 세밀한 상대 위치, 원래 token index, gap 크기, learned positional scaffold 같은 대안을 비교 후보로 둔다.

### 질문: S2-G-LOSS-FINITE는 무엇을 의미하는가

추가 시각: 2026-05-06 09:06 KST

맥락:

S2 결과 문서의 gate 표에서 `S2-G-LOSS-FINITE=true`가 "모든 주요 조건의 teacher-forced loss가 유한했다"로만 적혀 있어, 이 항목이 성능 우위를 뜻하는지 실행 안정성을 뜻하는지 직관적으로 불분명했다.

정리:

`S2-G-LOSS-FINITE`는 성능 우위 gate가 아니라 수치 안정성 gate다. teacher-forced loss가 `NaN`, `inf`, `-inf`로 깨지지 않고 모든 주요 조건에서 정상적인 유한 숫자로 계산됐다는 뜻이다.

근거/출처:

- `docs/v2/experiments/s2-skeleton-to-text-reconstruction.md`
- `outputs/v2_s2/lace_v2_s2/summary.md`

다음 실험에 주는 의미:

앞으로 gate 문서화에서는 "실험이 정상 실행됐는가"를 확인하는 안정성 gate와 "어떤 조건이 더 좋은가"를 판단하는 성능 비교 gate를 분리해서 설명한다.

### 발견: S2 의미 골격-문장 복원 학습 통과

추가 시각: 2026-05-06 08:48 KST

맥락:

S1은 의미 골격이 검색형 원문 식별 단서로 쓰일 수 있음을 보였다. S2는 같은 `t5-small` 구조와 짧은 학습 예산에서 의미 골격 + 위치 보조 구조가 무작위 골격이나 위치 전용 입력보다 더 나은 복원 학습 문제를 만드는지 확인했다.

결과:

S2는 `overall_pass=true`, `next_ready=true`로 통과했다. `attention_scaffold`는 token F1 0.3830, ROUGE-L 0.3117로 `random_scaffold` token F1 0.2286, ROUGE-L 0.1789보다 높았다. `position_only`는 nonempty 생성은 했지만 token F1과 ROUGE-L이 0이었다. attention 모델에 wrong-document 입력을 넣으면 token F1이 0.1222로 떨어졌다.

주의점:

`idf_scaffold`는 loss 2.4851로 가장 낮았고, `position_prior_scaffold`는 keyword recall 0.8504로 매우 높았다. 따라서 S2 결과는 attention scorer의 최종 우위가 아니라, 중요도 기반 의미 골격 계열이 무작위 골격보다 복원 학습에 유리하다는 제한된 증거로 해석한다.

근거/출처:

- `outputs/v2_s2/lace_v2_s2/summary.md`
- `docs/v2/experiments/s2-skeleton-to-text-reconstruction.md`

다음 실험에 주는 의미:

다음 단계는 S3 anchor baseline comparison이 적절하다. S2는 open-ended generation 성공 증거가 아니므로, 바로 큰 생성 주장으로 넘어가기보다 terminal skeleton 방식과 anchor 보조 조건 방식을 같은 복원 틀에서 비교해야 한다.

### 결정: S2는 짧은 조건별 복원 학습으로 시작한다

추가 시각: 2026-05-06 08:35 KST

맥락:

S1은 frozen encoder 검색 평가에서 `attention_correct` 의미 골격이 무작위, 다른 문서, 위치 전용 control보다 강하다는 것을 확인했다. 하지만 검색형 사용 신호만으로는 LACE의 핵심 주장인 더 나은 역방향 궤적을 입증할 수 없다.

결정:

S2는 `t5-small`을 조건별로 짧게 미세조정하는 의미 골격-문장 복원 학습으로 시작한다. 주요 비교는 `attention_scaffold`, `idf_scaffold`, `random_scaffold`, `position_prior_scaffold`, `position_only`다. `attention_scaffold` 모델에는 wrong-document, same-position random, position-only 평가 control을 추가로 적용한다.

근거/출처:

- `docs/v2/experiments/s1-skeleton-use-controls.md`
- `docs/v2/experiment-roadmap.md`
- `docs/v2/plan/s2-skeleton-to-text-reconstruction-plan.md`

다음 실험에 주는 의미:

S2의 성공은 open-ended generation 성공이 아니라, 의미 골격 + 위치 보조 구조가 무작위 손상보다 더 좋은 복원 학습 문제를 만든다는 제한된 증거로 해석한다. teacher-forced loss, token F1, ROUGE-L, wrong-document 하락을 분리해 기록한다.

## 2026-05-05

### 발견: S1 검색형 의미 골격 사용 검증 통과

추가 시각: 2026-05-05 22:06 KST

맥락:

S0는 의미 골격 생성 흐름이 작동한다는 것을 보였지만, 의미 골격이 실제 복원 단서로 쓰이는지는 확인하지 못했다. S1은 frozen `t5-small` encoder를 사용해 의미 골격을 query로 보고, 1024개 후보 원문 중 자기 원문을 찾는 검색형 복원 평가로 진행했다.

결과:

`attention_correct`가 Hit@1 0.9111로 가장 강했다. 이는 `random_same_count` 0.7373, `wrong_document` 0.0000, `position_only` 0.0010, `same_position_random` 0.0020보다 높다. `shuffled_correct`는 0.6074로 떨어져 순서 정보도 일부 작동했다.

주의점:

`position_prior`는 Hit@1 0.8154로 여전히 강했다. 따라서 S2에서는 위치 보조 구조를 버리지 말고, 위치 전용 control을 계속 유지해야 한다. 또한 `remove_topk` 0.4238이 `remove_lowk` 0.3750보다 낮지 않았으므로, 중요도 순서가 token별 인과 중요도와 정확히 일치한다는 주장은 아직 방어할 수 없다.

근거/출처:

- `outputs/v2_s1/lace_v2_s1/summary.md`
- `docs/v2/experiments/s1-skeleton-use-controls.md`

다음 실험에 주는 의미:

S1은 `overall_pass=true`, `s2_ready=true`로 통과했다. 다음 단계는 S2 의미 골격-문장 복원 학습이다. 핵심 주장은 "의미 골격 + 위치 보조 구조가 무작위 손상보다 더 좋은 역방향 궤적을 만든다"로 유지하되, S2에서는 생성 품질과 teacher-forced proxy를 분리하고 위치 편향 control을 계속 포함한다.

### 결정: S1은 검색형 복원 평가로 시작한다

추가 시각: 2026-05-05 21:55 KST

맥락:

S0는 의미 골격 생성 흐름이 작동한다는 것을 보여줬지만, 의미 골격이 실제 복원 과정에서 쓰이는지는 아직 확인하지 못했다. 바로 큰 생성 모델을 학습하면 실패 원인을 분리하기 어렵다.

결정:

S1은 frozen `t5-small` encoder를 사용한 검색형 복원 평가로 시작한다. 의미 골격을 query로 보고, 후보 원문 중 자기 원문을 얼마나 잘 찾는지 hit@1, hit@5, MRR, cosine margin으로 평가한다.

근거/출처:

- `docs/v2/experiments/s0-skeleton-pipeline.md`
- `docs/v2/experiment-roadmap.md`
- `docs/v2/plan/s1-skeleton-use-controls-plan.md`

다음 실험에 주는 의미:

S1에서는 correct, shuffled, random, wrong-document, position-prior, position-only, same-position random, remove top-k, remove low-k control을 먼저 검증한다. correct 의미 골격이 random/wrong-document/position-only보다 명확히 좋아야 S2 복원 학습으로 넘어간다.

### 결정: 타임라인 항목에는 추가 시각을 함께 기록한다

추가 시각: 2026-05-05 21:17 KST

맥락:

타임라인이 날짜 단위로만 기록되면 같은 날 여러 결정과 발견이 생겼을 때 순서를 복원하기 어렵다. 연구 과정에서는 질문, 판단, 실험 결과, 해석 변경이 짧은 간격으로 이어질 수 있으므로 항목별 추가 시각이 필요하다.

결정:

앞으로 타임라인에 새 항목을 추가할 때는 `추가 시각: YYYY-MM-DD HH:MM KST` 형식으로 기록한다. 이미 과거에 작성된 항목은 정확한 시각을 모르면 임의로 추정하지 않는다.

근거/출처:

- 사용자 지시: "타임라인에 추가할때, 추가된 시각도 같이 기록하도록 하자"
- `AGENTS.md`

다음 실험에 주는 의미:

S1 이후 계획, 실행, 결과 해석, 해석 변경은 모두 날짜뿐 아니라 추가 시각까지 남긴다. 이렇게 하면 같은 날 발생한 실험 설계 변경과 결과 해석의 순서를 더 정확히 추적할 수 있다.

### 결정: 답변과 연구 기록은 한글 용어를 우선한다

추가 시각: 미기록. 2026-05-05 후속 정리에서 시각 기록 규칙이 추가되기 전에 작성된 항목이다.

맥락:

연구 문서와 대화에 영어 연구 용어가 많이 섞이면서, 핵심 판단이 직관적으로 읽히지 않는 문제가 있었다. 사용자는 꼭 영단어가 필요한 경우가 아니라면 가급적 한글 용어로 답변하기를 원했다.

결정:

앞으로 답변과 연구 기록은 한글 용어를 우선한다. 단, 파일 경로, 명령어, 코드 식별자, metric/gate 이름, Kaggle kernel id, 논문/모델 고유명, 이미 고정된 핵심 주장 문장처럼 정확한 재현성이 필요한 표기는 원문을 유지할 수 있다.

근거/출처:

- 사용자 지시: "꼭 영단어가 필요한 경우가 아니면 가급적 한글 용어로 답변하도록 하자."
- `AGENTS.md`

다음 실험에 주는 의미:

S1 이후 계획서와 결과 문서는 "semantic skeleton"만 반복하기보다 "의미 골격", "positional scaffold"는 "위치 보조 구조", "random corruption"은 "무작위 손상"처럼 한글 용어를 먼저 사용한다. 필요한 경우 첫 등장에만 원문을 병기한다.

### 결정: V2 핵심 주장

추가 시각: 미기록. 2026-05-05 후속 정리에서 시각 기록 규칙이 추가되기 전에 작성된 항목이다.

맥락:

S0에서 의미 골격 생성 흐름이 실행 가능하다는 점을 확인했지만, 동시에 강한 문장 앞부분 위치 편향도 확인했다. 이에 따라 v2 연구 방향을 더 좁게 정리했다.

결정:

> 의미 골격 + 위치 보조 구조가 무작위 손상보다 더 나은 역방향 궤적을 만든다.

원문 고정 표현:

> Semantic skeleton + positional scaffold creates a better reverse trajectory than random corruption.

근거/출처:

- `docs/v2/experiments/s0-skeleton-pipeline.md`
- `outputs/v2_s0/lace_v2_s0/summary.md`

해석:

의미 골격은 내용을 담은 forward process의 최종 압축 상태로 취급한다. 위치 보조 구조는 유용하지만, 의미 골격을 다시 text로 확장하기 위한 보조 구조다. 위치 보조 구조만으로 좋아진 결과를 의미 골격 사용의 증거로 취급하면 안 된다.

다음 실험에 주는 의미:

S1에서는 더 큰 복원 또는 생성 주장으로 넘어가기 전에 correct, shuffled, random, wrong-document, position-prior, position-only, same-position random, top-k/low-k removal control로 의미 골격 사용을 검증해야 한다.
