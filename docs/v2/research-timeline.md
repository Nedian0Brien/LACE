# V2 연구 타임라인

이 문서는 v2 의미 골격 연구 흐름의 연구 질문, 결정, 경험적 사실, 주의점, 해석 변경을 시간순으로 기록한다.

## 2026-05-08

### 외부 검토: Claude는 S5 learned planner가 planner+realizer 연구로 이동할 위험을 지적했다

추가 시각: 2026-05-08 09:44 KST

맥락:

사용자는 Claude에게 현재 실험 계획, 만들고자 하는 모델, 전체 연구 컨셉을 상세히 설명하고 타당성 검토를 요청하라고 지시했다.

사실:

Claude review를 CLI로 요청했고, 결과를 `docs/v2/reviews/claude-s5-learned-planner-methodology-review.md`에 저장했다. 핵심 지적은 S5 learned planner가 성공하더라도 diffusion claim이 아니라 planner+conditional realizer claim으로 이동할 수 있다는 것이다. 또한 `shuffled_plan_schedule`이 oracle과 거의 같은 문제, S4g pretrained decoder artifact collapse, rollout score와 content/entity recall의 해리, direct seq2seq baseline 부재를 주요 risk로 지적했다.

근거/출처:

- `docs/v2/reviews/claude-s5-learned-planner-methodology-review.md`
- `docs/v2/experiments/s5-semantic-plan-bridge.md`
- `docs/v2/plan/s5-learned-semantic-planner-plan.md`

다음 실험에 주는 의미:

S5 learned planner는 진행 가능하지만, 다음 runner에는 direct seq2seq baseline, ordered-vs-shuffled learned plan 비교, planner recall threshold별 downstream 분석, plan-dropout sensitivity를 포함해야 한다. 또한 성공 기준은 no-plan/random 대비 개선만이 아니라 oracle gap 일부 회복과 direct baseline 대비 경쟁력까지 포함해야 한다.

### 계획: S5 learned semantic planner는 두 모델 파이프라인으로 구현한다

추가 시각: 2026-05-08 09:16 KST

맥락:

S5의 다음 primary condition을 learned semantic planner로 두기로 했고, 사용자는 이 구조를 실제로 어떻게 구현할지 확인을 요청했다.

결정:

S5 내부 iteration으로 별도 runner를 둔다. Planner model은 current skeleton, left/right anchor, gap query, transition ratio를 입력받아 content word/entity plan을 생성한다. Plan-conditioned realizer는 oracle plan examples와 plan-dropout examples를 함께 사용해 span 실현 방법을 학습하고, eval에서는 oracle/learned/no-plan/random/wrong-document/heuristic plan을 같은 realizer에 주입한다.

근거/출처:

- `kaggle/v2_s5/run_v2_s5.py`
- `docs/v2/plan/s5-learned-semantic-planner-plan.md`
- `wiki/concepts/lace/s5-semantic-plan-bridge.md`

다음 실험에 주는 의미:

기존 `predicted_plan_from_context()`를 더 정교한 휴리스틱으로 바꾸는 것이 아니라, `learned_plan_schedule`을 primary condition으로 추가한다. 특히 rollout에서는 이전 step의 생성 결과가 current skeleton을 바꾸므로, 각 step마다 planner가 learned plan을 다시 생성해야 한다.

## 2026-05-07

### 결정: heuristic planner는 연구 대상이 아니라 진단용 비교 장치로만 둔다

추가 시각: 2026-05-07 15:39 KST

맥락:

S5 version 1은 anchor/context 기반 heuristic predicted plan을 포함했다. 사용자는 heuristic planner가 연구 대상으로 바람직하지 않다고 지적했다.

결정:

이 지적을 수용한다. Heuristic planner는 learned planner를 만들기 전의 임시 진단 장치였을 뿐, LACE v2의 연구 대상이나 방법론 claim으로 삼지 않는다. 앞으로 S5의 핵심 구조는 learned semantic planner로 정의한다. Heuristic은 필요할 때 smoke/control/ablation으로만 사용하고, 성능 개선이나 방법론적 기여를 주장하는 주 조건으로 두지 않는다.

근거/출처:

- 사용자 해석
- `docs/v2/experiments/s5-semantic-plan-bridge.md`
- `wiki/concepts/lace/s5-semantic-plan-bridge.md`

다음 실험에 주는 의미:

다음 S5 iteration의 primary condition은 learned semantic planner여야 한다. Planner는 current skeleton, left/right anchor, gap query, transition ratio를 입력받아 content word/entity plan을 예측하고, no-plan/random-plan/wrong-document-plan은 control로 둔다.

### 해석 보정: S5 predicted plan 실패는 구조 폐기보다 scale/data/capacity 부족으로도 읽어야 함

추가 시각: 2026-05-07 15:37 KST

맥락:

S5 version 1에서 `predicted_plan_schedule`은 plan recall 0.0146으로 실패했다. 사용자는 모델이 본 데이터가 턱없이 적고 모델 규모도 작기 때문에 이 실패가 자연스럽다고 지적했다.

해석:

이 지적은 타당하다. S5의 predicted plan 실패를 곧바로 semantic plan 구조의 실패로 읽으면 과도하다. 이번 runner는 `t5-small`, train samples 768, condition별 train example cap 3,000, 1 epoch 조건이며, predicted plan도 learned planner가 아니라 anchor/context 기반 heuristic이었다. 따라서 S5의 방어 가능한 결론은 "semantic plan bridge가 틀렸다"가 아니라, "oracle plan은 강하지만 현재 규모와 heuristic planner로는 plan을 스스로 예측하지 못한다"이다.

근거/출처:

- 사용자 해석
- `docs/v2/experiments/s5-semantic-plan-bridge.md`
- `outputs/v2_s5/lace_v2_s5/metrics.json`

다음 실험에 주는 의미:

다음 S5 iteration은 open-ended generation이 아니라 learned semantic planner scale-up이어야 한다. 단, 무작정 S6로 가지 않고 plan prediction 자체를 별도 gate로 둔다. 최소 확인 사항은 data scale 증가, planner capacity 증가, content-applicable span 기준 plan recall/F1, 그리고 predicted-plan rollout이 no-plan/random/wrong-document plan보다 오르는지다.

### 발견: S5 Semantic Plan Bridge는 oracle plan에서 span collapse를 회복했지만 predicted plan은 실패

추가 시각: 2026-05-07 15:33 KST

맥락:

S4g는 pretrained `t5-small` decoder를 써도 generated span content/entity recall이 0.0000이고 artifact rate가 0.9961이었다. 따라서 S5에서는 skeleton과 surface span 사이에 semantic plan 중간 표현을 넣어, 올바른 content chunk가 주어질 때 span 생성이 살아나는지 확인했다.

결과:

S5는 Kaggle kernel `dennisparknd/lace-v2-s5-semantic-plan-bridge` version 1로 실행했다. `oracle_plan_schedule`은 span content recall 0.4144, entity recall 0.1191, artifact rate 0.1699로 S4g의 span collapse를 크게 회복했다. Rollout score도 1.4027로 `no_plan_schedule` 0.7055, `random_plan_schedule` 0.7022, `same_position_random_plan_schedule` 0.4423, `wrong_document_plan_schedule` 0.0065, `position_only_plan_schedule` -0.0471을 모두 이겼다.

하지만 `predicted_plan_schedule`은 plan recall 0.0146으로 `random_plan_schedule` 0.0459보다 낮았고, generated span content/entity recall은 0.0000이었다. Rollout score도 0.7054로 no-plan 0.7055와 사실상 같았다. 따라서 S5는 `stage_1_oracle_plan`은 통과했지만 `stage_2_plan_prediction`과 `stage_3_predicted_plan_rollout`은 실패했다.

근거/출처:

- `outputs/v2_s5/lace_v2_s5/metrics.json`
- `outputs/v2_s5/lace_v2_s5/summary.md`
- `docs/v2/experiments/s5-semantic-plan-bridge.md`

다음 실험에 주는 의미:

S6 open-ended generation은 아직 보류한다. 다음은 S5 내부에서 heuristic planner를 learned semantic planner로 교체하고, content-applicable span 기준 plan recall/F1을 별도로 집계한 뒤, predicted plan rollout이 no-plan/random/wrong-document plan보다 좋아지는지 확인해야 한다.

### 결정: S5 Semantic Plan Bridge runner와 계획서 작성

추가 시각: 2026-05-07 15:03 KST

맥락:

사용자는 S5 실험 진행을 승인했다. 직전 코드네임 규칙에 따라 S5는 `S5a/S5b`로 가지치기하지 않고, oracle plan, plan prediction, predicted-plan rollout을 하나의 phase 안의 stage로 관리한다.

결정:

`kaggle/v2_s5/run_v2_s5.py`, `scripts/push_kaggle_v2_s5.sh`, `docs/v2/plan/s5-semantic-plan-bridge-plan.md`를 추가했다. 첫 runner는 S4g의 pretrained text-to-text realizer를 유지하면서 prompt에 `semantic plan` 필드를 추가한다. 조건은 `oracle_plan_schedule`, `predicted_plan_schedule`, `no_plan_schedule`, `random_plan_schedule`, `same_position_random_plan_schedule`, `wrong_document_plan_schedule`, `position_only_plan_schedule`, `shuffled_plan_schedule`로 둔다.

근거/출처:

- `docs/v2/plan/s5-semantic-plan-bridge-plan.md`
- `kaggle/v2_s5/run_v2_s5.py`
- `scripts/push_kaggle_v2_s5.sh`

다음 실험에 주는 의미:

S5는 open-ended generation 실험이 아니라 span-level semantic generation 병목을 확인하는 구조 실험이다. 로컬 smoke는 runner plumbing 확인에만 사용하고, 실제 해석은 Kaggle full run output으로 한다.

### 결정: 실험 코드네임 가지치기 방지 규칙 확정

추가 시각: 2026-05-07 14:46 KST

맥락:

S4a-S4g 이후 다음 구조 후보를 `S4h-0`, `S4h-1`, `S4h-2`처럼 부르면 실험 이름이 연구 질문보다 구현 가지치기를 반영하게 된다. 사용자는 코드네임이 계속 뻗어나가며 복잡해지는 문제를 지적했고, 규칙을 명확히 정의할 필요가 있다고 했다.

결정:

코드네임은 구현 세부가 아니라 연구 질문의 위상을 나타내도록 정리한다. 새 연구 질문 또는 독립 gate가 생기면 letter suffix를 더 붙이지 않고 다음 정수 phase로 승격한다. 이에 따라 과거 대화에서 `S4h`라고 부르던 구조 후보는 구현 phase명이 아니라 `S5: Semantic Plan Bridge`로 승격한다. oracle plan, plan prediction, predicted-plan rollout은 `S5a/S5b`가 아니라 S5 내부의 `stage_1`, `stage_2`, `stage_3`로 관리한다. Open-ended generation은 S5가 통과한 뒤의 `S6`로 둔다.

근거/출처:

- `docs/v2/experiment-naming-rules.md`
- `docs/v2/experiment-roadmap.md`
- `web/s5-semantic-plan-bridge.html`

다음 실험에 주는 의미:

다음 Kaggle runner는 `kaggle/v2_s5/run_v2_s5.py`, 계획서는 `docs/v2/plan/s5-semantic-plan-bridge-plan.md`, 결과 문서는 `docs/v2/experiments/s5-semantic-plan-bridge.md`를 사용한다. 세부 stage는 같은 phase 안에서 관리한다.

### 결정: S5 Semantic Plan Bridge 설명서를 통해 hierarchical span expansion 후보를 구체화

추가 시각: 2026-05-07 14:34 KST

맥락:

S4g 이후 사용자는 구조적 문제 진단에는 동의했지만, 어떤 구조를 만들어야 하는지 직관적으로 감이 오지 않는다고 했다. 따라서 다음 구조 후보를 단순 설명이 아니라 한국어 문장 예시와 시각적 흐름으로 정리할 필요가 생겼다.

결정:

`S5: Semantic Plan Bridge`의 구조 후보를 문서화했다. 핵심은 reverse process가 곧바로 subword/punctuation 조각을 맞히지 않고, 먼저 gap query가 left/right anchor와 global skeleton을 직접 참고해 content word/chunk 단위 semantic plan을 만들고, 이후 surface realizer가 조사, 어미, punctuation을 붙여 문장 span으로 실현하는 것이다.

근거/출처:

- `web/s5-semantic-plan-bridge.html`
- `web/research-checkpoint-s4g.html`
- `docs/v2/experiments/s4g-pretrained-decoder-semantic-span-expansion.md`

다음 실험에 주는 의미:

S5 구현 시 gate는 final rollout score만 보지 않는다. `importance_schedule`이 strict controls를 계속 이기는지, generated-span-only content/entity recall이 S4g의 0.0000에서 오르는지, artifact rate가 S4g의 0.9961에서 내려가는지를 분리해서 봐야 한다.

### 발견: S4g pretrained decoder는 rollout 우위를 유지했지만 span semantic generation을 해결하지 못함

추가 시각: 2026-05-07 14:06 KST

맥락:

S4e는 shared-condition model에서도 `importance_schedule`의 final rollout 우위가 유지됨을 보였지만, generated span content/entity recall은 거의 무너졌고 artifact rate가 높았다. 사용자는 custom decoder 규모가 제대로 된 언어 생성을 하기에는 너무 작을 수 있다고 지적했다. S4g는 이 반론을 확인하기 위해 pretrained `t5-small` seq2seq decoder를 사용했다.

결과:

S4g는 `process_ready=true`, `overall_pass=false`, `structure_review_needed=true`, `s5_ready=false`였다. `importance_schedule` rollout score는 0.7314로 `random_schedule` 0.6404, `position_only_schedule` -0.0237, `same_position_random_schedule` 0.4580, `wrong_document_same_position_schedule` -0.0021, `no_anchor_gap_only_schedule` -0.0513을 모두 이겼다. Final content recall은 importance 0.3432, random 0.2582였고, final entity recall도 importance 0.2901, random 0.2260보다 높았다.

주의점:

S4g의 본래 목표였던 span-level semantic generation은 실패했다. Importance span content recall과 span entity recall은 모두 0.0000이고, artifact rate는 0.9961로 S4e 0.6906보다 높았다. 샘플에서는 target이 `brown`, `Australia`, `character`, `10` 같은 content token일 때도 prediction이 쉼표, `s`, 빈 조각에 가까운 token으로 수렴했다. 따라서 final rollout 우위는 새 span 생성 성공보다 기존 skeleton content 보존 효과가 강하게 반영된 결과로 해석해야 한다.

근거/출처:

- `outputs/v2_s4g/lace_v2_s4g/summary.md`
- `outputs/v2_s4g/lace_v2_s4g/metrics.json`
- `outputs/v2_s4g/lace_v2_s4g/reverse_transition_samples.jsonl`
- `docs/v2/experiments/s4g-pretrained-decoder-semantic-span-expansion.md`

다음 실험에 주는 의미:

S4g는 S4e 실패를 작은 decoder의 언어 prior 부족으로만 설명하기 어렵게 만든다. 다음은 S5 scale-up이 아니라 span target 단위 재구성, content/function token 분리, anchor-conditioned decoder 구조처럼 새로 unmask되는 span 자체가 실제 content/entity를 담도록 만드는 구조 개선이어야 한다.

### 결정: S4g pretrained decoder semantic span expansion 개시

추가 시각: 2026-05-07 13:53 KST

맥락:

S4e는 shared-condition model에서도 `importance_schedule`의 rollout 우위가 유지됨을 보였지만, generated span content/entity recall은 낮고 artifact rate가 높았다. 사용자는 제대로 된 언어 생성을 하기에는 현재 custom decoder 규모가 너무 작다는 점을 지적했다. 따라서 S4e 실패가 구조 문제인지, 작은 decoder의 언어 prior 부족인지 분리해야 한다.

결정:

S4g는 S4d/S4e의 six-control gap/span expansion 장치를 유지하되, 작은 custom decoder를 `AutoModelForSeq2SeqLM` 기반 pretrained decoder로 바꾼다. 입력은 condition, transition, span position, anchor distance, current skeleton text를 직렬화한 text prompt이고, target은 newly unmasked span text다. 학습 objective는 일반 seq2seq cross entropy로 유지한다.

근거/출처:

- `docs/v2/experiments/s4e-shared-condition-semantic-span-expansion.md`
- `docs/v2/plan/s4g-pretrained-decoder-semantic-span-expansion-plan.md`
- `kaggle/v2_s4g/run_v2_s4g.py`

다음 실험에 주는 의미:

S4g가 span content/entity를 높이고 artifact를 낮추면서 strict control 우위를 유지하면, S4e 실패의 상당 부분은 작은 decoder의 언어 prior 부족으로 해석할 수 있다. 반대로 no-anchor/random도 함께 좋아지거나 artifact가 계속 높으면, 단순 scale-up보다 span target 구성과 anchor 사용 구조를 먼저 개선해야 한다.

### 발견: S4e shared-condition model은 rollout 우위를 유지했지만 span content 개선에는 실패

추가 시각: 2026-05-07 13:00 KST

맥락:

S4d는 semantic skeleton content와 좌우 anchor가 gap/span expansion에 정보를 제공한다는 강한 신호를 보였지만, schedule마다 별도 reverse model을 학습했다. S4e는 여섯 schedule을 하나의 shared-condition model로 학습해 모델 분리 confound를 줄이고, S4d보다 generated span content/entity recall이 오르는지 확인했다.

결과:

S4e는 `process_ready=true`, `overall_pass=false`, `structure_review_needed=true`, `s5_ready=false`였다. `importance_schedule` rollout score는 0.7569로 `random_schedule` 0.6406, `position_only_schedule` 0.1576, `same_position_random_schedule` 0.4761, `wrong_document_same_position_schedule` 0.1396, `no_anchor_gap_only_schedule` 0.1496을 모두 이겼다. Final content recall은 importance 0.3464, random 0.2404였고, final entity recall은 importance 0.2976, random 0.2137이었다.

주의점:

S4e의 원래 개선 목표였던 span-level content/entity는 실패했다. Importance span content recall은 0.0029로 S4d 0.0172보다 낮았고, span entity recall도 0.0013으로 S4d 0.0047보다 낮았다. Artifact rate는 importance 0.6906으로 random 0.5259보다 높았다. Combined train examples는 277,999개였고 runtime은 약 2,241초라서 비용 병목도 확인됐다.

근거/출처:

- `outputs/v2_s4e/lace_v2_s4e/summary.md`
- `outputs/v2_s4e/lace_v2_s4e/metrics.json`
- `docs/v2/experiments/s4e-shared-condition-semantic-span-expansion.md`

다음 실험에 주는 의미:

S4e는 S4d의 rollout 우위가 조건별 모델 분리 때문만은 아니라는 점을 강화한다. 그러나 새로 생성되는 span이 의미 content/entity를 담는 능력은 개선하지 못했으므로, S5 scale-up보다 span target 구성, content/function token 분리, anchor cross-attention, 조건별 균형 샘플링 같은 구조 개선을 먼저 진행해야 한다.

### 결정: S4e shared-condition semantic span expansion 개시

추가 시각: 2026-05-07 12:14 KST

맥락:

S4d는 strict control 대비 가장 강한 구조적 신호를 보였지만, 각 schedule마다 별도 reverse model을 학습했다. 따라서 semantic skeleton의 우위가 모델 분리나 조건별 학습 난이도 차이 때문인지 완전히 분리하려면 하나의 공유 모델 안에서 조건을 비교해야 한다. 또한 S4d의 span-level content/entity recall은 아직 낮으므로, S5 open-ended generation scale-up 전에 새로 unmask되는 span 자체의 정보 전달 능력을 더 확인해야 한다.

결정:

S4e는 S4d의 gap/span expansion 구조와 six-control 비교를 유지하되, 모든 schedule 예제를 하나로 합쳐 하나의 shared-condition reverse model을 학습한다. 각 예제에는 `condition_id`, `gap_length`, `left_anchor_distance`, `right_anchor_distance`를 추가한다. 학습 objective는 일반 span token cross entropy로 유지하고, content/entity 지표는 gate와 해석에만 사용한다.

근거/출처:

- `docs/v2/experiments/s4d-skeleton-conditioned-gap-span-expansion.md`
- `docs/v2/plan/s4e-shared-condition-semantic-span-expansion-plan.md`
- `kaggle/v2_s4e/run_v2_s4e.py`

다음 실험에 주는 의미:

S4e가 통과하면 S4d의 의미 골격 사용 신호가 모델 분리 confound 없이도 유지된다는 해석이 강해진다. 통과하지 못하면 단순 scale-up보다 decoder 입력 구조, span 순서, anchor 사용 방식, schedule curriculum을 먼저 개선해야 한다.

## 2026-05-06

### 발견: S4d skeleton-conditioned gap/span expansion 통과

추가 시각: 2026-05-06 22:25 KST

맥락:

S4b는 importance 기반 rollout이 random과 position-only보다 좋다는 결과를 보였지만, 같은 위치 구조에서 아무 token content나 넣어도 되는지, 또는 decoder가 gap marker만으로 버티는지는 더 분리해야 했다. S4d는 `current semantic skeleton + left/right semantic anchor + span marker + timestep`으로 새로 열릴 contiguous gap/span만 생성하고, strict control들을 붙였다.

결과:

S4d는 `process_ready=true`, `overall_pass=true`, `structure_review_needed=false`, `s5_ready=false`였다. `importance_schedule` rollout score는 0.7175로 `random_schedule` 0.6145, `position_only_schedule` 0.0133, `same_position_random_schedule` 0.4733, `wrong_document_same_position_schedule` 0.0504, `no_anchor_gap_only_schedule` 0.0300을 모두 이겼다. Final content recall은 importance 0.3340, random 0.2448, same-position random 0.1952였고, entity recall은 importance 0.2988, random 0.2202, same-position random 0.1763이었다.

주의점:

생성문은 아직 자연스럽지 않고 span-level content recall은 0.0172로 낮다. Random은 ROUGE-L 0.3148로 importance 0.3092보다 조금 높다. Version 1은 AMP 학습 중 `nan`으로 실패했고, version 2는 AMP를 끄고 학습률을 낮춰 완료했으므로 runner 비용도 높다.

근거/출처:

- `outputs/v2_s4d/lace_v2_s4d/summary.md`
- `outputs/v2_s4d/lace_v2_s4d/metrics.json`
- `docs/v2/experiments/s4d-skeleton-conditioned-gap-span-expansion.md`

다음 실험에 주는 의미:

S4d는 같은 gap/span 위치 구조에서도 실제 semantic skeleton content와 좌우 anchor가 있어야 reverse expansion이 강해진다는 현재까지 가장 강한 구조적 증거다. 다만 S5 open-ended generation으로 바로 가지 않고, generated span의 content/entity recall을 끌어올리는 구조와 shared-condition runner로 비용을 줄이는 방향을 검토한다.

### 결정: S4d를 handcrafted objective가 아니라 skeleton-conditioned gap/span expansion으로 개시

추가 시각: 2026-05-06 21:37 KST

맥락:

S4b는 importance 기반 multi-step rollout이 random과 position-only보다 좋은 고무적인 신호를 보였지만, S4c의 naive span-infilling은 content/entity 0 붕괴와 position-only confound를 넘지 못했다. 사용자는 handcrafted content/entity objective가 실험을 위한 실험이 될 수 있다고 지적했고, 구조 자체로 성능과 해석력을 높이는 방향을 선호했다.

결정:

S4d는 content/entity 중심 loss를 쓰지 않는다. 학습 objective는 일반적인 span token cross entropy로 유지하고, 구조는 `current semantic skeleton + left/right semantic anchor + span marker + timestep`을 입력으로 받아 새로 unmask될 contiguous gap/span만 생성한다. 비교군은 `importance_schedule`, `random_schedule`, `position_only_schedule`, `same_position_random_schedule`, `wrong_document_same_position_schedule`, `no_anchor_gap_only_schedule`로 둔다.

근거/출처:

- 사용자 대화
- `docs/v2/plan/s4d-skeleton-conditioned-gap-span-expansion-plan.md`
- `kaggle/v2_s4d/run_v2_s4d.py`

다음 실험에 주는 의미:

S4d의 핵심 확인점은 같은 gap 구조에서 실제 semantic anchor content가 position-only, same-position random, wrong-document, no-anchor control보다 나은 span 생성과 rollout을 만드는지다. 성공하면 S4b의 process-level 신호가 “위치 scaffold만의 효과”가 아니라 skeleton content 사용 증거로 강화된다.

### 발견: S4b multi-step delta rollout 통과

추가 시각: 2026-05-06 17:16 KST

맥락:

S4a는 새로 unmask될 delta token/span만 예측하게 하자 `importance_schedule`이 random과 position-only를 이겼다. 그러나 한 단계 teacher-forced 성능이 실제 역방향 궤적에서도 유지되는지는 별도 검증이 필요했다.

결과:

S4b는 generated delta를 다음 current state에 삽입하며 `25% -> 50% -> 75% -> 100%` multi-step rollout을 평가했다. `importance_schedule`은 rollout score 0.7336으로 `random_schedule` 0.6215와 `position_only_schedule` 0.1858을 모두 이겼다. Final content recall은 importance 0.3357, random 0.2401, position-only 0.0000이었고, entity recall은 importance 0.2855가 random 0.2050보다 높았다. Repetition은 importance 0.0501이 random 0.1103보다 낮았다.

주의점:

`random_schedule`은 ROUGE-L 0.3259로 importance 0.3099보다 높았다. 또한 S4b는 위치가 주어진 constrained delta insertion이므로 open-ended generation 성공 증거는 아니다.

근거/출처:

- `outputs/v2_s4b/lace_v2_s4b/summary.md`
- `outputs/v2_s4b/lace_v2_s4b/metrics.json`
- `docs/v2/experiments/s4b-multi-step-delta-rollout.md`

다음 실험에 주는 의미:

S4b는 현재까지 v2 process claim을 가장 강하게 지지한다. 다만 S5로 바로 가기보다, S4c와 후속 구조 보정을 통해 content/entity token을 더 잘 생성하고 punctuation/whitespace shortcut을 줄여야 한다.

### 발견: S4c naive span-infilling 구조는 position-only confound를 넘지 못함

추가 시각: 2026-05-06 17:16 KST

맥락:

S4a/S4b의 delta decoder에는 반복과 표면 token 편향이 남아 있었다. S4c는 autoregressive delta generation 대신 marker-position infilling 구조로 바꾸면 반복을 줄이고 semantic skeleton content를 더 잘 쓸 수 있는지 확인했다.

결과:

S4c는 `process_ready=true`였지만 `overall_pass=false`였다. `importance_schedule`은 score 0.2403, masked-token accuracy 0.1414로 `random_schedule` score 0.1425, accuracy 0.1121보다 높았다. 그러나 `position_only_schedule`도 masked-token accuracy 0.1414로 같았고 score는 0.3026으로 가장 높았다. Content recall과 entity recall은 세 조건 모두 0이었다.

주의점:

Sample에서는 쉼표, 공백, 짧은 subword 같은 형식 token 예측이 많았고, `duplicate_prediction_rate`가 0.9347로 높았다. 따라서 repetition rate 0.0은 좋은 생성 품질을 뜻하지 않는다.

근거/출처:

- `outputs/v2_s4c/lace_v2_s4c/summary.md`
- `outputs/v2_s4c/lace_v2_s4c/metrics.json`
- `docs/v2/experiments/s4c-span-infilling-reverse-decoder.md`

다음 실험에 주는 의미:

S4c는 그대로 확장할 구조가 아니라 실패 진단이다. 이후 사용자는 handcrafted content/entity objective를 좋은 선택이 아니라고 보았고, 다음 구조는 일반 CE objective를 유지하되 semantic anchor와 contiguous gap/span expansion, same-position/wrong-document control을 포함하는 방향으로 조정됐다.

### 발견: S4a-delta token reverse objective 결과

추가 시각: 2026-05-06 16:34 KST

맥락:

S4에서는 `random_schedule`이 종합 score와 표면 복원 지표에서 이겼지만, `importance_schedule`은 의미 보존/확장 지표에서 더 강했다. S4a는 이 모호성이 전체 target state를 다시 생성하게 한 objective 때문인지 확인하기 위해 실행했다. 입력은 현재 partial state와 이번 단계에서 채울 위치 marker이고, target은 newly unmasked delta token/span이다.

결과:

S4a는 `process_ready=true`, `overall_pass=true`, `structure_review_needed=false`, `s5_ready=false`였다. `importance_schedule`은 score 0.6366, loss 5.7282, TF Delta Acc 0.1577, Delta F1 0.1700, Delta ROUGE-L 0.1468로 `random_schedule` score 0.5073, loss 6.7063, TF Delta Acc 0.1092, Delta F1 0.1258, Delta ROUGE-L 0.1077보다 높았다. `position_only_schedule`도 score 0.5889로 강했지만, importance가 tolerance 이상 앞섰다.

주의점:

이 결과는 S4의 random 우위가 semantic skeleton 가설 폐기보다 objective mismatch였다는 해석을 강화한다. 다만 entity recall은 random 0.0175가 importance 0.0115보다 높았고, repetition도 importance 0.1584가 random 0.1062보다 나빴다. 또한 position-only가 TF Delta Acc 0.1483으로 importance 0.1577에 가까워 위치 scaffold의 강한 prior는 계속 confound로 남는다.

근거/출처:

- `outputs/v2_s4a/lace_v2_s4a/summary.md`
- `outputs/v2_s4a/lace_v2_s4a/metrics.json`
- `docs/v2/experiments/s4a-delta-token-reverse-objective.md`

다음 실험에 주는 의미:

S5 open-ended generation으로 바로 가지 않는다. 다음은 `S4b: multi-step delta rollout` 또는 `S4c: span-infilling reverse decoder`다. S4a의 긍정 신호가 여러 reverse step을 누적해도 유지되는지 확인하고, entity/repetition 병목과 position-only 강세를 구조적으로 줄여야 한다.

### 발견: S4-importance ordered reverse diffusion 결과

추가 시각: 2026-05-06 15:35 KST

맥락:

S4는 문장 exact reconstruction probe가 아니라, 중요도 낮은 token부터 순차적으로 masking하는 forward process의 역과정이 random corruption보다 더 좋은 reverse expansion curriculum을 만드는지 확인하기 위해 실행했다. 조건은 `importance_schedule`, `random_schedule`, `position_only_schedule`이며, reverse transition은 `0.25->0.50`, `0.50->0.75`, `0.75->1.00`이다.

결과:

S4는 `process_ready=true`, `overall_pass=false`, `s5_ready=false`였다. 종합 score는 `random_schedule` 0.5607이 `importance_schedule` 0.4839보다 높았다. Random은 loss 6.0172, Token F1 0.2300, ROUGE-L 0.1712로 표면 복원 지표에서도 높았다. 반면 importance는 target content recall 0.0574, input retention 0.0471, expansion recall 0.0416, original content recall 0.0496, entity recall 0.0831로 의미 보존/확장 지표에서 random보다 모두 높았다.

주의점:

S4는 importance-ordered reverse diffusion이 random보다 더 좋은 language model trajectory라는 증거는 주지 못했다. 하지만 중심 의미를 보존하고 세부 의미 token을 붙이는 방향에서는 importance schedule이 random보다 강한 신호를 보였다. 각 schedule의 target state가 다르므로 Token F1/ROUGE-L의 단순 비교에는 target 난이도와 token frequency 차이가 섞일 수 있다.

근거/출처:

- `outputs/v2_s4/lace_v2_s4/summary.md`
- `outputs/v2_s4/lace_v2_s4/metrics.json`
- `docs/v2/experiments/s4-importance-ordered-reverse-diffusion.md`

다음 실험에 주는 의미:

S5 open-ended generation으로 가지 않는다. 다음은 `S4a: delta-token reverse objective`다. 전체 target state를 다시 생성하는 대신 새로 unmask될 token/span만 예측하고, schedule-specific target 외에 공통 original/semantic target 평가를 추가해야 한다.

### 결정: S3 이후 초점을 문장 복원 probe가 아니라 forward/reverse diffusion process로 재정렬

추가 시각: 2026-05-06 15:09 KST

맥락:

S3b까지는 attention terminal이 random 계열보다 약간 높고, 위치 channel이 완전히 무시되지는 않는다는 점을 확인했다. 하지만 사용자는 "문장을 그대로 복원할 수 있느냐"가 연구의 본질이 아니며, 핵심은 중요성이 낮은 token을 순차적으로 masking하는 forward process와 그 역과정으로 문장을 확장하는 diffusion language model을 만들 수 있느냐라고 정리했다.

결정:

이후 실험 초점은 exact reconstruction probe의 수치 개선이 아니라, importance-ordered masking schedule이 random corruption schedule보다 더 나은 reverse generation process를 제공하는지로 둔다. 모델은 문장의 중심 의미를 담는 뼈대 token을 먼저 다루고, 이후 세부 의미와 표면 token을 단계적으로 붙여 문장을 확장하는 능력을 학습해야 한다.

근거/출처:

- 사용자 대화
- `docs/v2/experiments/s3b-probe-calibration.md`
- `wiki/concepts/lace/forward-reverse-process-본질.md`

다음 실험에 주는 의미:

다음 단계는 S3c의 지엽적 probe 보정보다, `importance-ordered forward schedule`과 `reverse expansion objective`를 명시한 process-level 실험 설계가 되어야 한다. 비교군은 동일 mask ratio와 동일 model budget의 random masking diffusion이며, 평가는 원문 exact reconstruction보다 trajectory coherence, semantic drift, skeleton faithfulness, generation quality, repetition/diversity를 중심으로 설계한다.

### 발견: S3b-probe calibration 결과

추가 시각: 2026-05-06 14:54 KST

맥락:

S3b는 S3a의 조건별 재학습 confound를 줄이기 위해 실행했다. `attention_terminal`로 reverse model을 한 번만 학습한 뒤, 평가 시점에 `attention_no_position`, `attention_shuffled_position`, `position_only`, `random_terminal`, `same_position_random_terminal`, predicted/gold anchor 조건으로 입력만 바꿨다.

결과:

S3b는 `diagnostic_ready=true`, `s4_ready=false`였다. `attention_no_position`은 score 0.2828로 `attention_terminal` 0.3768보다 크게 낮아 위치 channel의 존재가 중요하다는 점은 확인됐다. 하지만 `attention_terminal`은 `position_only` 0.3735, `random_terminal` 0.3724, `same_position_random_terminal` 0.3638보다 tolerance 0.02 이상 높지 않았다. 최고 조건은 `attention_shuffled_position` score 0.3799였다. Anchor 조건은 모두 크게 낮았고, `random_terminal_gold_anchor_oracle` 0.2453이 `random_terminal_predicted_anchor` 0.2443보다 약간 높았다.

주의점:

S3b는 위치 보조 구조가 완전히 무시되지는 않는다는 점을 확인했지만, 현재 reverse probe가 의미 terminal content를 충분히 사용한다는 증거는 약하다. 생성 sample의 repetition도 높아 lexical metric 차이를 semantic generation 품질로 확대 해석하면 안 된다.

근거/출처:

- `outputs/v2_s3b/lace_v2_s3b/summary.md`
- `outputs/v2_s3b/lace_v2_s3b/metrics.json`
- `docs/v2/experiments/s3b-probe-calibration.md`

다음 실험에 주는 의미:

S4 constrained generation으로 바로 넘어가지 않는다. 다음은 반복을 줄이는 constrained reconstruction 설정, position-only 분해 control, terminal content-use metric 강화를 포함하는 S3c 성격의 probe 보정이 적절하다. Anchor 조건은 당분간 핵심 경로에서 내린다.

### 발견: S3a-terminal diagnostic 결과

추가 시각: 2026-05-06 13:38 KST

맥락:

S3a는 S3에서 `random_forward_no_anchor`가 가장 높았던 이유를 분리하기 위해 실행했다. 조건은 `attention_terminal`, `idf_terminal`, `random_terminal`, `same_position_random_terminal`, `position_only`, `random_terminal_predicted_anchor`, `random_terminal_gold_anchor_oracle`이다.

결과:

S3a는 `diagnostic_ready=true`, `s4_ready=false`였다. `attention_terminal`은 score 0.4351, Token F1 0.1540, ROUGE-L 0.1406으로 `random_terminal` score 0.4014와 `same_position_random_terminal` score 0.3429보다 높았다. 하지만 `position_only`도 score 0.4275로 가까웠고, 최고 조건은 `random_terminal_predicted_anchor` score 0.4489였다. `random_terminal_gold_anchor_oracle`은 score 0.4115로 predicted anchor보다 낮았다.

주의점:

S3에서 보였던 random terminal 우위는 S3a에서 약해졌으므로 content terminal 신호는 일부 회복됐다. 하지만 position-only가 너무 가깝고, 낮은 품질의 predicted anchor가 최고 조건이었기 때문에 현재 reverse probe와 lexical metric은 여전히 표면 prior와 위치 scaffold에 민감하다. 이 결과만으로 S4 constrained generation으로 넘어가면 안 된다.

근거/출처:

- `outputs/v2_s3a/lace_v2_s3a/summary.md`
- `outputs/v2_s3a/lace_v2_s3a/metrics.json`
- `docs/v2/experiments/s3a-terminal-diagnostic.md`

다음 실험에 주는 의미:

다음 후보는 `S3b-probe calibration`이다. 같은 학습 모델에 평가 입력만 바꾸는 ablation, gold anchor 길이/segment ablation, position-only matched control, 반복률과 entity recall metric을 추가해 reverse probe가 terminal content를 실제로 쓰는지 확인한다.

### 질문: S3 이후 본격적인 방법론 설계 전에 부족한 것은 무엇인가

추가 시각: 2026-05-06 12:32 KST

맥락:

S3는 `overall_pass=false`, `s4_ready=false`였고, `random_forward_no_anchor`가 가장 높은 score를 얻었다. 이 결과는 곧바로 S4 generation으로 넘어가기 어렵다는 뜻이지만, semantic skeleton 가설 자체를 폐기할 만큼 원인이 분해된 것도 아니다.

정리:

현재 부족한 것은 더 큰 모델 하나가 아니라 식별 방법론이다. 특히 `better reverse trajectory`의 운영 정의, random terminal baseline의 실제 강도, predicted anchor baseline의 약함, scorer 선택, reverse probe의 terminal 정보 민감도, 위치 편향과 content 사용 분리가 아직 충분하지 않다.

근거/출처:

- `docs/v2/experiments/s3-anchor-baseline-comparison.md`
- `outputs/v2_s3/lace_v2_s3/summary.md`
- `wiki/concepts/lace/s3-이후-방법론-부족점.md`

다음 실험에 주는 의미:

다음 단계는 방법론을 바로 크게 바꾸기보다 `S3a-terminal diagnostic`으로 측정 장치를 분해하는 것이다. `gold_anchor_oracle`, `predicted_anchor`, `attention_terminal`, `idf_terminal`, `random_terminal`, `same_position_random_terminal`, `position_only`를 가까운 조건으로 비교해 terminal 정보량, 위치 편향, anchor predictor 병목, metric 민감도를 분리한다.

### 발견: S3-anchor baseline comparison 핵심 gate 실패

추가 시각: 2026-05-06 11:11 KST

맥락:

S3는 중요한 token을 forward terminal state로 직접 보존하는 방식이 random forward 뒤에 anchor를 예측해 붙이는 방식보다 더 좋은 reverse trajectory를 만드는지 확인했다. S2a 결과에 따라 기본 위치 표현은 `sinusoidal_absolute`를 사용했고, anchor 조건에는 gold anchor가 아니라 terminal에서 예측한 anchor를 붙였다.

결과:

S3는 `overall_pass=false`, `s4_ready=false`였다. `importance_ordered_forward_no_anchor`는 score 0.4356, Token F1 0.1515, ROUGE-L 0.1441이었다. `random_forward_anchor_prediction`은 score 0.4418, Token F1 0.1559, ROUGE-L 0.1460이었다. 최고 조건은 `random_forward_no_anchor`로 score 0.4467, Token F1 0.1612, ROUGE-L 0.1455였다.

주의점:

`importance_ordered_forward_no_anchor`는 `random_forward_anchor_prediction`과 tolerance 0.02 안에서 비슷했으므로 anchor prediction baseline에 크게 밀린 것은 아니다. 하지만 `random_forward_no_anchor`보다 낮았기 때문에, 현재 S3 설정에서는 importance terminal state가 random terminal state보다 더 좋은 reverse trajectory라는 핵심 주장을 강화하지 못했다.

근거/출처:

- `outputs/v2_s3/lace_v2_s3/summary.md`
- `outputs/v2_s3/lace_v2_s3/metrics.json`
- `docs/v2/experiments/s3-anchor-baseline-comparison.md`

다음 실험에 주는 의미:

S4 constrained generation으로 바로 넘어가지 않는다. 다음 후보는 `S3a-terminal diagnostic`이다. 여기서는 `random_forward_no_anchor`가 왜 가장 높았는지, `gold_anchor_oracle`과 `same_position_random`, `position_only`, `idf_terminal`, `attention_terminal` 비교로 terminal 정보량과 위치 편향, anchor predictor 병목을 분리한다.

### 결정: S3-anchor baseline comparison에 들어간다

추가 시각: 2026-05-06 11:05 KST

맥락:

S2는 의미 골격 + 위치 보조 구조가 무작위 골격보다 더 좋은 복원 학습 문제를 만든다는 제한된 증거를 제공했다. S2a는 S3의 기본 위치 보조 구조 후보로 `sinusoidal_absolute`를 식별했다. 다음 질문은 semantic skeleton을 forward terminal state로 보존하는 방식이, random forward 뒤에 anchor를 예측해 붙이는 방식보다 더 좋은 reverse trajectory를 만드는가다.

결정:

S3 실험 이름은 `S3-anchor baseline comparison`으로 둔다. 핵심 비교는 `random_forward_anchor_prediction`, `importance_ordered_forward_no_anchor`, `importance_ordered_forward_anchor_prediction`, `random_forward_no_anchor` 네 조건이다.

근거/출처:

- `docs/v2/experiment-roadmap.md`
- `docs/v2/experiments/s2-skeleton-to-text-reconstruction.md`
- `docs/v2/experiments/s2a-positional-encoding.md`
- `docs/v2/plan/s3-anchor-baseline-comparison-plan.md`

다음 실험에 주는 의미:

`importance_ordered_forward_no_anchor`가 `random_forward_anchor_prediction`보다 좋거나 거의 비슷하고, `random_forward_no_anchor`보다 좋으면 v2 핵심 주장인 semantic skeleton terminal state의 장점이 강화된다. `importance_ordered_forward_anchor_prediction`이 가장 좋으면 skeleton terminal state와 anchor prediction이 상보적이라는 후속 방향을 남긴다.

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
