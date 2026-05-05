# V2 S0 실험 결과: Semantic Skeleton Pipeline Sanity Check

## 1. 실험 목적

S0는 LACE v2의 첫 실행 실험이다. 목적은 reverse model을 학습하기 전에, 원문 `x_0`에서 importance-guided semantic skeleton `x_T`를 안정적으로 만들 수 있는지 확인하는 것이다.

쉽게 말하면 다음 질문을 본다.

> 문장을 줄였을 때 남는 것이 그냥 토큰 조각이나 keyword bag이 아니라, 원문의 의미 뼈대인가?

S0는 generation 실험이 아니다. 이 단계에서 확인할 것은 skeleton 재료의 품질이다. 여기서 skeleton 자체가 random/uniform baseline과 차이가 없다면, 이후 S1/S2에서 reverse model이 실패했을 때 원인이 skeleton인지 model인지 분리할 수 없다.

## 2. 실행 정보

| 항목 | 값 |
|---|---|
| Kaggle kernel | `dennisparknd/lace-v2-s0-skeleton-pipeline` |
| kernel version | 1 |
| 실행 상태 | 완료 |
| 모델 | `t5-small` frozen encoder |
| 데이터 소스 | `hf:wikitext/wikitext-2-raw-v1:train` |
| 샘플 수 | 1024 |
| 최대 길이 | 128 token |
| active tokens | 106649 |
| device | cuda |
| keep ratios | 0.50, 0.25, 0.125 |
| 결과 위치 | `outputs/v2_s0/lace_v2_s0/` |

생성된 주요 파일:

| 파일 | 설명 |
|---|---|
| `metrics.json` | 전체 metric, gate, run info |
| `summary.md` | runner가 생성한 요약 |
| `score_stats.json` | scorer 분포와 position correlation |
| `skeleton_samples.jsonl` | qualitative audit용 skeleton sample |

## 3. 실험 조건

S0는 같은 token budget에서 여러 skeleton 생성 방식을 비교했다.

| 조건 | 의미 |
|---|---|
| `random` | 같은 token count로 무작위 token 보존 |
| `uniform` | 문장 전체에서 균등 간격으로 token 보존 |
| `frequency` | corpus에서 자주 등장하는 token을 우선 보존 |
| `idf` | 드문 token, 즉 informativeness가 높은 token을 우선 보존 |
| `position_prior` | 문장 앞쪽 token을 우선 보존 |
| `attention_received` | frozen T5 encoder attention을 많이 받은 token을 우선 보존 |

여기서 `position_prior`는 원래 핵심 scorer가 아니라 confound control이다. WikiText 첫 문장은 제목/주제/요약 정보가 앞에 몰리는 경향이 있으므로, "semantic skeleton처럼 보이는 결과가 사실은 앞부분만 남긴 효과인가?"를 확인하기 위해 넣었다.

## 4. Gate 결과

| Gate | 결과 | 의미 |
|---|---|---|
| `S0-G-RUN` | pass | 모든 scorer와 keep ratio 조합이 실행됨 |
| `S0-G-COUNT-MATCH` | pass | scorer별 평균 kept token count 차이가 0 |
| `S0-G-ENTITY-GAP` | pass | IDF/attention 계열이 random보다 keyword/entity recall에서 우위 |
| `S0-G-SEMANTIC-GAP` | pass | IDF/attention 계열 중 하나가 random보다 embedding similarity에서 우위 |
| `S0-G-SCORER-NONCOLLAPSE` | pass | attention score가 완전히 위치 또는 단일 score로 collapse하지 않음 |

Runner 판정:

```text
overall_pass: true
s1_ready: true
```

다만 이 pass의 의미는 제한적이다.

> S0는 skeleton pipeline이 S1/S2로 넘어갈 만큼 작동한다는 뜻이지, LACE v2의 generation claim이 입증됐다는 뜻이 아니다.

## 5. 핵심 metric

### 5.1 Keep ratio 0.25

중간 압축률인 25% budget에서 결과가 가장 직관적이다.

| Condition | Kept | Keyword Recall | Entity Recall | Semantic Similarity | Position Mean |
|---|---:|---:|---:|---:|---:|
| `attention_received@0.25` | 26.20 | 0.1898 | 0.2171 | 0.7131 | 0.4504 |
| `idf@0.25` | 26.20 | 0.3144 | 0.2781 | 0.6594 | 0.4913 |
| `position_prior@0.25` | 26.20 | 0.1709 | 0.2895 | 0.7492 | 0.1197 |
| `random@0.25` | 26.20 | 0.0941 | 0.1452 | 0.6716 | 0.5011 |
| `uniform@0.25` | 26.20 | 0.0772 | 0.1387 | 0.6603 | 0.5001 |
| `frequency@0.25` | 26.20 | 0.0002 | 0.0000 | 0.4460 | 0.5112 |

### 5.2 Gate gap

| 항목 | 값 |
|---|---:|
| best keyword gap vs random | 0.2308 |
| best entity gap vs random | 0.2922 |
| best semantic similarity gap vs random | 0.0420 |
| attention mean score std | 0.0040 |
| attention mean position correlation | -0.0772 |

## 6. 결과 해석

### 6.1 Skeleton preservation

Skeleton preservation은 압축된 token subset이 원문의 핵심 단어와 의미를 얼마나 남기는지 본다.

이 관점에서 S0는 긍정적이다. `idf`는 keyword/entity recall에서 random과 uniform을 명확히 이겼다. 예를 들어 25% budget에서 `idf` keyword recall은 0.3144이고 random은 0.0941이다. 이는 단순히 아무 token이나 남기는 것보다, informativeness가 높은 token을 남기는 것이 원문의 식별 가능한 핵심 요소를 더 잘 보존한다는 뜻이다.

`attention_received`도 random보다 semantic similarity가 높았다. 25% budget에서 attention semantic similarity는 0.7131이고 random은 0.6716이다. 이 결과는 frozen encoder attention이 완전히 무의미한 scorer는 아니라는 신호다.

하지만 가장 조심해야 할 점이 있다. `position_prior`가 매우 강했다. 25% budget에서 `position_prior` semantic similarity는 0.7492로 attention보다 높다. WikiText 문단은 첫 문장/앞부분에 주제 정보가 많이 들어 있으므로, "좋은 skeleton"처럼 보이는 결과 일부는 semantic importance가 아니라 lead-position bias일 수 있다.

### 6.2 Scorer quality

Scorer quality는 token importance scorer가 실제 의미 단위를 잡는지, 아니면 쉬운 표면 패턴을 따라가는지 본다.

`frequency`는 거의 실패했다. 25% budget에서 keyword recall은 0.0002, entity recall은 0.0000이다. 자주 등장하는 token은 대부분 기능어와 구두점 계열이라 semantic skeleton으로 부적합하다는 것을 보여준다.

`idf`는 keyword/entity 보존에는 가장 강하다. 다만 qualitative sample을 보면 T5 subword 조각이 그대로 남아 skeleton이 사람이 읽기에는 거칠다. 즉 IDF는 "핵심 식별자 보존"에는 좋지만, relation/event structure를 매끄럽게 보존한다고 말하기는 어렵다.

`attention_received`는 position correlation이 낮았다. 평균 position correlation은 -0.0772로, 단순히 앞쪽 token만 고르는 scorer는 아니다. 그러나 attention skeleton도 punctuation, subword fragment, connector token이 섞인다. 따라서 attention만으로 semantic skeleton scorer를 확정하기에는 이르다.

### 6.3 S1/S2로 넘길 수 있는 것

S0에서 가장 방어 가능한 결론은 다음이다.

> v2 skeleton pipeline은 작동한다. 같은 token budget에서 IDF/attention 계열은 random/uniform보다 의미 보존 신호를 보인다. 그러나 position-prior baseline이 강하므로, 다음 실험에서는 위치 편향을 반드시 control로 둬야 한다.

따라서 `s1_ready=true`는 맞다. 단, 그 의미는 "이제 generation 실험으로 가도 된다"가 아니라 "skeleton-use control을 설계할 artifact와 baseline이 생겼다"이다.

## 7. Caveat

첫 번째 caveat는 metric 자체다. keyword/entity recall은 heuristic이다. 특히 entity proxy는 capitalized token과 숫자를 사용하므로 true NER가 아니다. 이 metric은 빠른 sanity check에는 유용하지만 논문 수준의 entity preservation 증거로는 부족하다.

두 번째 caveat는 tokenizer다. T5 SentencePiece token을 그대로 고르면 skeleton이 사람이 읽는 단어 단위가 아니라 subword 조각으로 남는다. S1/S2에서는 word-level grouping 또는 decoded span cleanup을 검토해야 한다.

세 번째 caveat는 data bias다. WikiText는 문서 첫 문장에 제목/정의/요약 정보가 강하게 모인다. `position_prior`가 좋은 성능을 낸 것은 이 confound가 실제로 크다는 신호다. 이후에는 shuffled paragraph, non-lead sentence, same-position random control을 강화해야 한다.

## 8. 다음 phase 판단

다음은 S1이다.

S1에서 확인해야 할 것은 다음이다.

> Reverse evaluator 또는 model이 correct skeleton을 실제로 사용하는가?

S1에는 최소한 다음 control을 포함해야 한다.

| Control | 이유 |
|---|---|
| correct skeleton | 정상 조건 |
| shuffled skeleton | order와 relation structure를 쓰는지 확인 |
| random skeleton | 같은 token budget corruption baseline |
| wrong-document skeleton | content mismatch 민감도 확인 |
| position-prior skeleton | WikiText lead-position confound 통제 |
| remove top-k important tokens | 핵심 token 제거 영향 확인 |
| remove low-k tokens | 부가 token 제거 대비 |

S0 결과만 놓고 보면, S1의 핵심 비교는 `idf`, `attention_received`, `position_prior`, `random` 네 축으로 시작하는 것이 가장 방어 가능하다.
