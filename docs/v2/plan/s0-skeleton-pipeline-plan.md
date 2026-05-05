# V2 S0 Plan: Semantic Skeleton Pipeline Sanity Check

## 1. 실험이 측정하는 것

S0는 LACE v2의 첫 Kaggle 실험이다. 이 실험은 reverse model을 아직 학습하지 않고, 원문 `x_0`에서 importance-guided semantic skeleton `x_T`를 안정적으로 만들 수 있는지 확인한다.

측정 대상은 다음 네 가지다.

| 항목 | 측정 내용 |
|---|---|
| skeleton construction | token importance score로 압축 단계별 skeleton을 만들 수 있는가 |
| semantic preservation | 같은 token budget에서 random/uniform skeleton보다 의미를 더 보존하는가 |
| scorer quality | scorer가 frequency, 위치, token count shortcut만 따라가지 않는가 |
| phase readiness | S1/S2로 넘어갈 만큼 baseline과 output format이 안정적인가 |

S0는 [../research-questions.md](../research-questions.md)의 RQ1, RQ2를 직접 다룬다. [../experiment-roadmap.md](../experiment-roadmap.md)의 `S0. Skeleton Pipeline Sanity Check`에 해당한다.

## 2. 왜 중요한가

v2의 핵심 주장은 중요한 token이 auxiliary anchor가 아니라 forward process의 terminal state라는 것이다. 따라서 먼저 확인해야 할 것은 model training 성능이 아니라 `x_T` 자체의 정체성이다.

만약 S0에서 importance-guided skeleton이 random/uniform baseline과 차이가 없다면, 이후 S2 reconstruction이나 S4 generation이 실패했을 때 원인을 분리할 수 없다. 반대로 S0가 통과되면 다음 실험은 "semantic skeleton이 존재하는가"가 아니라 "reverse model이 그 skeleton을 실제로 사용하는가"로 좁혀진다.

## 3. 좋은 결과와 나쁜 결과

좋은 결과는 attention/PMI 기반 skeleton이 같은 token count의 random/uniform skeleton보다 keyword/entity recall과 sentence embedding similarity에서 일관되게 높게 나오는 것이다. compression level이 깊어질수록 metric이 하락하더라도, random 대비 preservation gap이 남아 있어야 한다.

나쁜 결과는 다음 중 하나다.

- importance-guided skeleton과 random skeleton의 차이가 작다.
- scorer가 문장 앞부분 token이나 high-frequency token만 남긴다.
- entity는 남지만 relation/event structure가 사라진다.
- attention scorer가 PMI/frequency baseline보다 약하다.

나쁜 결과가 나오면 S2/S3로 바로 가지 않고 scorer와 skeleton definition을 먼저 고친다.

## 4. Confound와 Control

S0에서 반드시 분리해야 할 confound는 "좋은 skeleton처럼 보이지만 실제로는 쉬운 shortcut인 경우"다.

| Confound | Control | 해석 기준 |
|---|---|---|
| token count가 많아서 좋아짐 | same-count random skeleton | 같은 예산에서 importance가 이겨야 함 |
| 위치 패턴이 쉬움 | same-position random skeleton | 남은 위치만 맞춘 skeleton보다 좋아야 함 |
| frequency만 따라감 | frequency/PMI skeleton | attention scorer가 최소한 비슷하거나 상보적이어야 함 |
| keyword bag 문제 | entity/relation sample audit | 핵심 명사만이 아니라 사건/관계가 남아야 함 |
| scorer collapse | score distribution summary | score가 소수 token 또는 위치에 과도하게 몰리지 않아야 함 |

S0는 아직 skeleton-use 실험이 아니므로 wrong-document skeleton, shuffled skeleton, top-k removal은 S1의 본 실험으로 넘긴다. 다만 S0 output schema는 S1에서 이 control들을 바로 만들 수 있게 문장 id, token index, score, kept/masked mask를 저장해야 한다.

## 5. 구현 산출물

| 산출물 | 위치 | 역할 |
|---|---|---|
| Kaggle runner | `kaggle/v2_s0/run_v2_s0.py` | self-contained S0 실행 script |
| Kaggle metadata | `kaggle/v2_s0/kernel-metadata.json` | Kaggle kernel 설정 |
| push script | `scripts/push_kaggle_v2_s0.sh` | S0 kernel push |
| result doc | `docs/v2/experiments/s0-skeleton-pipeline.md` | 실행 후 결과 해석 |
| outputs | `outputs/v2_s0/` | metrics, summary, skeleton samples |

Runner는 Kaggle script kernel에서 단독 실행되어야 한다. 기존 v1 runner처럼 fallback text를 포함하되, 기본 데이터는 `wikitext/wikitext-2-raw-v1` train split을 사용한다.

## 6. Skeleton 생성 방식

초기 S0에서는 learned scorer를 넣지 않는다. learned scorer는 leakage와 collapse risk가 커서 S0의 pipeline sanity check를 흐릴 수 있다.

우선순위는 다음이다.

| Scorer | 설명 | S0에서의 역할 |
|---|---|---|
| uniform length | importance 없이 길이만 맞춤 | 최소 baseline |
| random | 같은 token count로 무작위 보존 | corruption baseline |
| frequency/PMI | corpus 통계 기반 informativeness | 강한 단순 baseline |
| attention-received | frozen encoder attention 수신량 | v2의 첫 semantic scorer 후보 |

Attention scorer는 `t5-small` encoder를 frozen으로 사용한다. token 단위 score를 얻고, compression budget별로 score 상위 token을 보존한다.

## 7. Compression Schedule

초기 token budget은 세 단계로 둔다.

```text
keep ratios: 0.50, 0.25, 0.125
```

각 sample에서 special token과 padding은 metric 계산에서 제외한다. 너무 짧은 문장은 최소 보존 token 수를 만족하지 못하면 제외하거나 별도 count로 기록한다.

## 8. Metric과 Gate

| Metric/Gate | 측정 | 통과 기준 |
|---|---|---|
| `S0-G-RUN` | 모든 scorer와 budget 실행 | metrics/summary/samples 생성 |
| `S0-G-COUNT-MATCH` | scorer별 token budget 일치 | 평균 kept token count gap이 1 token 이하 |
| `S0-G-ENTITY-GAP` | random 대비 entity/keyword recall | attention 또는 PMI가 random보다 높음 |
| `S0-G-SEMANTIC-GAP` | random 대비 embedding similarity | attention 또는 PMI가 random보다 높음 |
| `S0-G-SCORER-NONCOLLAPSE` | score 분포/position 편향 | 단일 위치대/소수 token 집중 없음 |

S0의 `overall_pass`는 `S0-G-RUN`, `S0-G-COUNT-MATCH`, 그리고 preservation gap gate 중 하나 이상이 통과하면 true로 둔다. 단, `overall_pass=true`는 "generation-ready"가 아니라 "S1/S2를 설계할 수 있는 skeleton pipeline이 확보됨"을 뜻한다.

## 9. Output Format

Kaggle output에는 최소 다음 파일을 남긴다.

| 파일 | 내용 |
|---|---|
| `metrics.json` | scorer/budget별 aggregate metric과 gate |
| `summary.md` | runner가 생성한 짧은 결과 요약 |
| `skeleton_samples.jsonl` | 원문, scorer별 skeleton, scores, kept indices |
| `score_stats.json` | score distribution, position bias, token count stats |

`skeleton_samples.jsonl`은 사람이 읽는 qualitative audit가 가능해야 한다. S0에서 가장 중요한 실패 신호는 숫자보다 "skeleton이 그냥 keyword bag인지"일 수 있기 때문이다.

## 10. 로컬 검증

문서 작업에는 test suite를 만들거나 실행하지 않는다. S0 runner를 만든 뒤에는 다음만 확인한다.

```bash
rtk .venv/bin/python -m py_compile kaggle/v2_s0/run_v2_s0.py
rtk bash -n scripts/push_kaggle_v2_s0.sh
git diff --check
```

필요하면 `--max-samples 16 --use-hf-dataset false` 형태의 local smoke run을 한 번 수행한다.

## 11. 다음 실험으로 넘길 조건

S0가 통과되면 바로 큰 generation 실험으로 가지 않는다. 다음은 S1이다.

S1에서 검증할 질문은 다음이다.

> Reverse evaluator 또는 model이 correct skeleton을 실제로 사용하는가?

S1에는 correct, shuffled, random, wrong-document, remove top-k, remove low-k control을 포함한다. S0는 이 control을 만들 수 있는 skeleton artifact와 baseline metric을 제공하는 역할까지만 맡는다.

S0가 실패하면 다음 phase로 넘어가지 않고 scorer 후보를 좁힌다. 특히 attention-received가 frequency/PMI보다 약하면, learned scorer를 도입하기 전에 PMI hybrid 또는 POS/entity-aware rule을 먼저 검토한다.
