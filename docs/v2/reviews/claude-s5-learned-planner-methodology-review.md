# Claude Review: S5 Learned Semantic Planner 방법론 검토

작성 시각: 2026-05-08 09:44 KST

## 요청 맥락

Claude에게 현재 LACE v2 연구 컨셉, S4/S5 실험 결과, 그리고 다음 S5 learned semantic planner 계획을 설명하고, 연구 방법론 관점에서 타당성 검토를 요청했다.

핵심 요청은 다음이었다.

- LACE 전체 컨셉이 과학적으로 coherent한가?
- S5 learned semantic planner가 타당한 다음 단계인가, 아니면 diffusion claim을 planner+realizer claim으로 약화시키는가?
- 긍정 근거, 반론, confound, 필수 control, 성공 기준, 포기/수정 기준은 무엇인가?
- 다음 runner 구현 전 architecture/training/evaluation에서 무엇을 고쳐야 하는가?

## 핵심 요약

Claude의 총평은 신중한 조건부 수용이다.

```text
LACE의 기본 직관은 방어 가능하지만,
S5 learned planner는 성공하더라도 diffusion language model claim이 아니라
planner + conditional realizer 연구로 이동할 위험이 있다.
```

가장 중요한 지적은 세 가지다.

1. `shuffled_plan_schedule`이 oracle plan과 거의 같은 성능을 낸 것은 심각하다. 현재 plan은 ordered sentence plan이 아니라 content word bag으로 작동할 가능성이 높다.
2. S4g에서 pretrained `t5-small`을 붙였는데도 generated span content/entity recall이 0.0000이고 artifact rate가 0.9961로 무너진 원인을 먼저 이해해야 한다.
3. Learned planner를 붙이면 두 단계 조건부 생성 pipeline이 되므로, 성공하더라도 이것이 semantic diffusion trajectory의 성과인지, skeleton-conditioned seq2seq/planner-realizer 구조의 성과인지 분리해야 한다.

## Claude Review 원문

### 1. 전체 연구 개념의 과학적 일관성

기본 직관은 방어 가능하나, 핵심 주장이 아직 입증되지 않았다.

Diffusion 언어 모델에서 random corruption이 의미론적으로 비효율적이라는 직관 자체는 합리적이다. BERT 계열 masked LM 연구들도 masking 전략이 학습에 영향을 준다는 것을 보였고, importance-ordered unmasking이 reverse trajectory를 더 예측 가능하게 만들 수 있다는 이론적 근거도 있다.

그러나 현재 핵심 주장에 대한 구조적 약점이 있다.

첫째, LACE의 forward process는 표준 diffusion 정의에서 벗어나 있다. 표준 discrete diffusion은 forward process가 마르코프 체인이고 tractable marginal `q(x_t | x_0)`를 보장한다. LACE의 importance-ordered masking은 token별 masking 확률이 해당 문서의 IDF/attention에 의존하므로 `q(x_t | x_0)`가 문서마다 다른 non-stationary process가 된다. 이것이 이론적으로 어떻게 정당화되는지 working paper에 명시되어야 한다.

둘째, semantic skeleton이 positional scaffold보다 낫다는 핵심 주장이 S3에서 실패했고, S4 이후 experiment들은 이 주장을 직접 검증하는 대신 점점 복잡한 conditioning 구조를 추가하는 방향으로 피벗했다. 원래 주장 자체가 아직 깨끗하게 증명되지 않은 채로 pipeline이 복잡해지고 있다는 점을 인식해야 한다.

### 2. S5 Learned Semantic Planner의 방법론적 타당성

이 설계가 LACE를 diffusion 모델 연구에서 planner+realizer 연구로 전환시킬 위험이 있다. 이는 단순한 구현 선택이 아니라 연구의 정체성 문제다.

Learned planner를 추가하면 시스템은 사실상 두 개의 조건부 생성 모델이 된다.

```text
planner:  (skeleton, anchors, gap_query, ratio) -> plan
realizer: (skeleton, anchors, gap_query, ratio, plan) -> span
```

이 구조에서 "diffusion forward process의 semantic ordering이 유리하다"는 원래 주장이 어디에 있는가? Realizer가 plan condition에 의존하는 seq2seq라면, 이것은 사실상 skeleton-conditioned conditional language model이지 diffusion model이 아니다. Rollout 구조가 있다고 해서 diffusion의 이론적 속성을 상속받지는 않는다.

Oracle plan 조건이 강하다는 S5 결과도 조심해서 해석해야 한다. Oracle plan은 target span의 content word를 직접 제공하므로 성능을 올리는 것은 어느 정도 trivial하다. 이것이 semantic planning의 유용성을 증명하는지, 단순히 정답 힌트를 주면 쉬워진다는 것을 보여주는지 분리해야 한다.

S5에서 shuffled oracle plan이 ordered oracle plan과 거의 같은 성능을 보인 것은 매우 중요한 신호다. 이는 realizer가 plan의 순서 정보를 사용하지 못하고 있으며, 현재 plan이 ordered semantic sentence plan이 아니라 content word bag으로 기능하고 있음을 시사한다. 이 문제가 해결되지 않으면 learned planner iteration은 "planner가 bag을 더 잘 예측하는가"를 연구하는 것으로 축소된다.

### 3. 현재 방향의 긍정적 논거

객관적으로 지지할 수 있는 결과도 있다.

S4b의 `importance_schedule > random_schedule > position_only_schedule` 결과는 현재까지 가장 강력한 증거다. Wrong-document와 position-only 제어를 구분한 S4d도 positional artifact 가설을 일정 부분 약화시킨다.

Delta prediction으로의 전환도 방법론적으로 올바른 결정이었다. 전체 target state를 매 step에서 재생성하는 것은 delta prediction보다 훈련 신호가 noisy하므로, 이 변경은 독립적으로 정당화된다.

Wrong-document/same-position, no-anchor/gap-only, position-only 같은 control 설계도 방법론적으로 올바르며 계속 유지되어야 한다.

### 4. 가장 강력한 반론과 혼동 요인

#### S4g pretrained model 실패

T5-small 도입 후 span content recall이 0.0000, artifact rate가 0.9961로 붕괴된 것은 표면적 문제가 아니다. Pretrained language model prior가 오히려 성능을 악화시켰다는 것은 LACE의 conditioning 방식이 pretrained model과 호환되지 않을 가능성을 뜻한다. Learned planner를 pretrained backbone 위에서 구현한다면 같은 문제가 재현될 수 있다.

#### Rollout score의 대리 지표 문제

Rollout score가 content recall, entity recall과 해리되는 현상이 반복된다. S4g에서 rollout 우월성이 유지됐지만 span content recall이 0이라면, rollout score는 텍스트 품질이 아닌 다른 무언가를 측정하고 있을 수 있다. Rollout score가 중심 지표라면 실제 텍스트 품질과 어떻게 연결되는지 독립적인 검증이 필요하다.

#### Skeleton에서 target span content를 예측하는 문제 자체의 난이도

S5에서 heuristic predicted plan recall이 random보다 낮았다는 것은 단순 구현 실패가 아닐 수 있다. Skeleton에서 특정 target span의 content word를 예측하는 것이 본질적으로 어렵다면 learned planner도 같은 한계에 부딪힌다.

#### Dynamic rollout의 train-inference mismatch

Rollout 시 planner는 generated span으로 업데이트된 skeleton을 입력받지만, 훈련 시 planner는 oracle span이 있는 skeleton을 입력받는다. Generated span 품질이 낮을수록 distribution shift가 커진다. Plan-dropout은 realizer robustness 대책이지만 planner의 exposure bias 대책은 아직 명시되어 있지 않다.

#### Two-stage error propagation

Planner error가 realizer로 전파된다. Planner recall이 올라가도 precision이 낮으면 wrong content word가 realizer를 오도한다. Planner metric과 downstream metric은 분리해야 하지만, 두 error가 독립적이라고 보면 안 된다.

### 5. 성공 주장 전 필수 control

필수 control은 다음이다.

1. Strong seq2seq baseline: 같은 훈련 데이터로 fine-tuned된 `skeleton + anchors + gap -> span` 직접 예측 baseline. 이것과 비교하지 않으면 diffusion rollout의 기여를 주장할 수 없다.
2. Planner recall threshold별 downstream 분석: learned plan recall이 높은 subset에서 realizer span quality가 실제로 올라가는지 확인한다.
3. Shuffled learned-plan vs ordered learned-plan: oracle에서 발견된 bag-vs-order 문제가 learned plan에서도 재현되는지 확인한다.
4. Planner와 realizer의 shared encoder vs fully separated 비교: 두 모델이 skeleton을 서로 다르게 인코딩하면 plan과 condition representation 사이의 일관성이 깨질 수 있다.
5. Plan conditioning on/off flip: 같은 realizer에서 plan을 줬을 때와 뺐을 때 성능 차이가 유의미한지 확인한다.
6. Plan-dropout 비율 sensitivity: 0%, 20%, 50% 등을 비교한다.

### 6. 설득력 있는 성공 기준

최소 기준:

- Learned plan recall이 random plan 0.0459보다 충분히 높아야 한다. Claude는 예시로 0.15 이상을 제안했다.
- Span content recall이 현재 0.0000에서 최소 0.10 이상으로 올라야 한다.
- Artifact rate가 현재 0.9961에서 0.50 미만으로 내려가야 한다.
- Learned plan rollout이 no-plan보다 통계적으로 유의하게 높아야 한다.

강한 기준:

- Ordered learned-plan이 shuffled learned-plan보다 span quality에서 유의하게 좋아야 한다.
- `rollout(learned) - rollout(no_plan)`이 `rollout(oracle) - rollout(no_plan)` gap의 절반 이상을 회복해야 한다.
- Strong seq2seq baseline보다 learned-plan rollout이 좋아야 diffusion rollout의 기여를 주장할 수 있다.
- Planner recall이 높은 subset에서 realizer entity recall이 유의하게 향상되어야 한다.

### 7. 방향 포기 또는 수정 기준

포기 기준:

- Learned plan recall이 random plan과 통계적으로 구분되지 않거나 더 낮을 경우.
- Learned plan recall이 향상되더라도 span content recall이 0.05 이하에 머물 경우.
- Shuffled learned-plan과 ordered learned-plan이 같은 성능을 보일 경우.

수정 기준:

- Learned plan condition의 성공이 strong seq2seq baseline과 구분되지 않으면 diffusion framework를 포기하고 skeleton-conditioned seq2seq로 재정의하는 편이 합리적이다.
- Plan-dropout 없이는 rollout이 breakdown하면 pipeline 설계가 dynamic rollout에 적합하지 않다는 신호다.
- 모든 metric에서 S4b rollout score 0.7336을 넘지 못하면 pipeline 복잡도의 가치가 낮다.

### 8. 구현 전 권고사항

#### Architecture

Plan representation을 먼저 결정해야 한다. 현재 plan이 discrete token sequence인지 continuous embedding인지 불분명하다. Discrete token sequence라면 plan의 평균 길이와 최대 길이를 먼저 분석해야 한다.

Planner와 realizer는 skeleton encoder를 공유하는 구조를 우선 고려해야 한다. 완전 분리는 ablation으로 비교하는 편이 낫다.

Plan conditioning은 단순 concatenation보다 cross-attention 방식이 더 적절할 수 있다. S4g에서 pretrained model과의 통합 실패가 conditioning 방식과 관련 있는지 확인해야 한다.

#### Training

Plan-dropout 비율은 임의로 정하지 말고 0%, 20%, 50% 등으로 sensitivity를 확인해야 한다.

Planner를 먼저 수렴시킨 후 realizer를 fine-tune하는 2단계 훈련을 고려해야 한다. 처음부터 joint training하면 planner가 trivial plan을 출력하고 realizer가 이를 무시하는 equilibrium으로 갈 수 있다.

S4g 실패 원인을 먼저 분석해야 한다. Pretrained T5-small 도입 후 artifact rate가 0.9961로 증가한 원인이 positional encoding mismatch인지, conditioning signal 크기 문제인지, 다른 이유인지 확인하지 않으면 learned planner에서도 같은 실패가 반복될 수 있다.

#### Evaluation

Plan precision/recall/F1의 측정 방식을 명시해야 한다. Exact match인지, stemmed match인지, semantic similarity인지에 따라 수치가 크게 달라진다.

Rollout score의 정의를 논문에서 정식화해야 한다. Rollout score와 content/entity recall이 해리된다면, 중심 지표로 쓰기 전에 실제 텍스트 품질과 연결되는지 검증해야 한다.

## 이 리뷰가 다음 실험에 주는 의미

S5 learned planner를 진행하되, 구현 전 gate를 더 엄격하게 바꿔야 한다.

가장 중요한 보정은 다음이다.

1. `learned_plan_schedule`만 추가하지 말고 `direct_seq2seq_baseline`을 함께 둔다.
2. `ordered_learned_plan`과 `shuffled_learned_plan`을 비교해 plan이 bag인지 ordered plan인지 분리한다.
3. Planner quality를 content-applicable span 기준으로 보고하되, planner recall threshold별 downstream quality를 함께 분석한다.
4. S4g의 artifact collapse 원인을 다음 runner에서 별도 diagnostic으로 남긴다.
5. 성공 기준은 learned planner가 no-plan/random만 이기는 수준이 아니라 oracle gap 일부를 회복하고 direct seq2seq baseline과도 경쟁하는 수준으로 둔다.
