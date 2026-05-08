# 중간 진단: LACE v2 Forward Process 연구에서 입증된 것과 입증되지 않은 것

작성 시각: 2026-05-08 10:02 KST

## 연구의 정확한 중심

현재 연구는 "문장을 그대로 잘 복원하는 모델"을 만드는 연구가 아니다. 중심은 다음이다.

```text
Diffusion Language Model의 forward process를
무작위 손상이 아니라 의미 중심 손상으로 재정의할 수 있는가?
```

더 구체적으로는 원문에서 중요성이 낮은 token부터 순차적으로 mask하고, 마지막에는 문장의 중심 의미를 담는 semantic skeleton을 남기는 forward trajectory를 만든다. Reverse process는 그 반대 방향으로 skeleton에서 세부 token/span을 붙이며 문장을 확장한다.

현재 방어 가능한 핵심 claim은 아직 좁다.

```text
Constrained reverse rollout에서 semantic skeleton + positional scaffold는
random corruption이나 position-only보다 더 나은 의미 보존 궤적을 만든다.
```

아직 방어할 수 없는 claim은 다음이다.

```text
LACE는 이미 open-ended Diffusion Language Model이다.
```

## 입증된 것

### 1. Semantic skeleton을 만드는 forward artifact는 만들 수 있다

S0-S2는 attention/IDF 기반 중요 token 보존이 random/uniform보다 의미 정보를 더 잘 담는다는 constrained evidence를 제공했다. 즉 forward process의 terminal state 후보로 쓸 수 있는 semantic skeleton artifact는 만들 수 있다.

다만 이것은 "좋은 generation"의 증거가 아니라, reverse 실험에 들어갈 terminal state와 control을 만들 수 있다는 의미다.

### 2. 전체 target state 재생성보다 delta/span 예측 objective가 연구 질문에 맞다

S4a 이후부터 전체 문장을 매 step 다시 생성하지 않고, 각 단계에서 새로 unmask될 token/span만 예측하도록 바꿨다. 이 전환은 중요했다. LACE의 reverse process는 skeleton에서 세부를 덧붙이는 과정이므로, delta/span objective가 연구 질문과 더 잘 맞는다.

### 3. Constrained multi-step rollout에서 importance trajectory는 random보다 의미 보존이 좋았다

S4b는 현재까지 forward process claim을 가장 직접적으로 지지하는 결과다.

```text
importance_schedule rollout score: 0.7336
random_schedule rollout score:     0.6215
position_only_schedule score:      0.1858
```

특히 position-only는 teacher-forced proxy에서는 어느 정도 버텼지만, multi-step rollout에서는 content/entity가 0으로 무너졌다. 따라서 위치 보조 구조만으로 final semantic state를 만들었다는 반론은 약해졌다.

### 4. 같은 위치 구조에서도 실제 skeleton content가 필요하다는 신호가 있다

S4d는 `same_position_random`, `wrong_document_same_position`, `no_anchor_gap_only`, `position_only` control을 붙였고, importance가 모두 이겼다.

```text
importance rollout score:      0.7175
random rollout score:          0.6145
same-position random score:    0.4733
wrong-document score:          0.0504
no-anchor gap-only score:      0.0300
position-only score:           0.0133
```

이는 "같은 위치에 아무 content나 넣어도 된다"가 아니라 "문맥에 맞는 semantic skeleton content와 anchor가 reverse expansion에 실제 정보를 준다"는 해석을 가능하게 한다.

### 5. Semantic plan upper-bound는 span semantic collapse를 크게 회복한다

S4g에서는 pretrained `t5-small`을 써도 generated span content/entity recall이 0.0000으로 무너졌고 artifact rate는 0.9961이었다. S5의 oracle semantic plan은 이 붕괴를 크게 회복했다.

```text
S5 oracle span content recall: 0.4144
S5 oracle entity recall:       0.1191
S5 oracle artifact rate:       0.1699
S5 oracle rollout score:       1.4027
```

따라서 "올바른 content plan이 있으면 span realizer가 semantic content를 생성할 수 있다"는 upper-bound는 입증됐다.

## 강하게 시사되지만 아직 완전 입증은 아닌 것

### 1. Semantic ordering은 random corruption보다 더 좋은 reverse trajectory일 가능성이 높다

S4b/S4d/S4e/S4g에서 importance는 strict control 대비 final rollout 의미 보존에서 반복적으로 우위를 보였다. 이 패턴은 우연으로 보기 어렵다.

하지만 이것은 constrained rollout 안의 증거다. 아직 표준 DLM training/sampling objective에서 random corruption보다 낫다는 뜻은 아니다.

### 2. Position scaffold는 필요하지만 충분하지 않다

Position signal은 강하다. S0-S3에서 계속 confound로 나타났다. 하지만 S4b/S4d 이후로는 position-only가 final semantic rollout에서 무너졌기 때문에, 위치만으로 설명하기 어렵다는 쪽으로 증거가 이동했다.

현재 결론은 다음이다.

```text
위치 보조 구조는 필요하다.
그러나 위치 보조 구조만으로는 semantic reverse trajectory를 만들 수 없다.
```

### 3. Reverse process에는 content plan 또는 chunk-level 중간 표현이 필요할 가능성이 높다

S4g는 pretrained decoder만으로 해결되지 않았고, S5 oracle plan은 span collapse를 회복했다. 따라서 skeleton에서 바로 surface token을 생성하는 구조보다, content/entity plan 같은 중간 표현이 필요할 가능성이 커졌다.

다만 현재 이것은 oracle upper-bound다. 모델이 plan을 스스로 학습할 수 있는지는 아직 미입증이다.

## 부분 입증에 머문 것

### 1. Semantic skeleton이 random보다 "항상" 좋은 것은 아니다

Random은 여러 실험에서 surface overlap이나 ROUGE-L 같은 표면 지표에서는 여전히 강했다. 따라서 "semantic skeleton이 모든 metric에서 random보다 좋다"는 말은 틀리다.

방어 가능한 표현은 더 좁다.

```text
Random은 표면 token overlap에서 강할 수 있지만,
semantic content/entity 보존과 strict-control rollout에서는 importance가 더 낫다.
```

### 2. Current rollout score는 의미 품질의 완전한 proxy가 아니다

S4g에서 final rollout 우위는 유지됐지만 generated-span content/entity recall은 0.0000이었다. 이는 rollout score가 skeleton에 이미 남아 있던 content 보존 효과를 크게 반영할 수 있음을 뜻한다.

따라서 앞으로는 final rollout score와 generated-span-only content/entity recall을 계속 분리해야 한다.

### 3. Semantic plan은 아직 ordered plan이 아니라 content word bag에 가깝다

S5에서 `shuffled_plan_schedule`은 oracle plan과 거의 같은 rollout score를 냈다.

```text
oracle rollout score:   1.4027
shuffled rollout score: 1.3989
```

따라서 현재 plan은 순서 있는 문장 계획이라기보다 content word bag으로 작동할 가능성이 높다. 이것은 S5 learned planner에서 반드시 다시 검증해야 한다.

## 입증되지 않은 것

### 1. LACE가 아직 full Diffusion Language Model이라는 것은 입증되지 않았다

현재 실험들은 대부분 constrained reverse rollout이다. 어느 위치가 열릴지 알고 있고, target transition도 정해져 있다. 따라서 open-ended sampling, coherent long-form generation, diversity/repetition control은 아직 입증되지 않았다.

### 2. Forward process가 표준 diffusion 관점에서 이론적으로 정식화됐다는 것도 아직 아니다

LACE forward process는 문서별 importance score에 따라 손상 순서가 바뀐다. 이는 일반적인 random corruption diffusion의 stationary/Markov noising process와 다르다.

아직 명확하지 않은 것은 다음이다.

- `q(x_t | x_0)`를 어떻게 정의할 것인가
- timestep별 marginal이 tractable한가
- learned importance scorer를 쓰면 forward process가 어떻게 고정되는가
- 기존 discrete diffusion objective와 어떻게 연결되는가

따라서 현재는 "새로운 forward process 후보"이지, 완성된 diffusion formalism은 아니다.

### 3. 모델이 semantic plan을 스스로 예측할 수 있다는 것은 아직 입증되지 않았다

S5의 predicted plan은 heuristic이었고 실패했다.

```text
predicted plan recall: 0.0146
random plan recall:    0.0459
predicted rollout:     0.7054
no-plan rollout:       0.7055
```

따라서 다음 learned semantic planner는 핵심 병목이다. 이것이 실패하면 S5 oracle result는 "정답 힌트를 주면 좋아진다"는 upper-bound에 머문다.

### 4. Learned planner + realizer가 diffusion claim을 유지한다는 것도 아직 불명확하다

두 단계 구조는 성공하더라도 planner + conditional realizer pipeline으로 해석될 위험이 있다. 따라서 direct seq2seq baseline과 비교해야 한다. 만약 direct baseline과 구분되지 않으면 LACE를 diffusion process 연구로 주장하기 어렵다.

### 5. 일반화는 아직 입증되지 않았다

현재 주요 실험은 WikiText와 작은 모델/제한된 sample cap 위에서 진행됐다. 다른 corpus, 더 긴 문서, 다른 언어, 더 큰 모델, learned importance scorer로 확장해도 같은 결과가 나는지는 모른다.

## 현재 중간 결론

현재까지의 연구는 실패가 아니다. 오히려 핵심 방향 중 일부는 꽤 강하게 살아 있다.

가장 강하게 살아 있는 부분은 다음이다.

```text
무작위로 token을 지우는 것보다,
중심 의미 token을 terminal skeleton으로 보존하고
그 주변을 단계적으로 확장하는 constrained reverse trajectory가
semantic preservation에는 더 유리하다.
```

하지만 아직 넘지 못한 벽도 분명하다.

```text
그 trajectory가 실제 Diffusion Language Model의 forward/reverse process로
학습 가능하고, open-ended generation에서 유리하다는 것은 아직 입증되지 않았다.
```

따라서 다음 연구의 정확한 질문은 다음이어야 한다.

```text
Semantic skeleton에서 다음 span의 content/entity plan을
모델이 스스로 학습해 예측할 수 있는가?

그리고 그 learned plan을 이용한 reverse rollout이
no-plan, random-plan, wrong-document-plan, direct seq2seq baseline보다
실제로 더 나은 semantic expansion을 만드는가?
```

이 질문을 통과해야 S6 open-ended generation 또는 더 큰 scale-up으로 넘어갈 수 있다.
