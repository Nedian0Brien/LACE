---
title: "복원 평가 지표 - Token F1과 ROUGE-L"
type: "concept"
tags: [LACE, S2, 복원평가, Token-F1, ROUGE-L]
created: 2026-05-06
updated: 2026-05-06
sources: [사용자 대화, docs/v2/experiments/s2-skeleton-to-text-reconstruction.md]
---

# 복원 평가 지표 - Token F1과 ROUGE-L

## 핵심 요약

S2 의미 골격-문장 복원 학습에서 Token F1과 ROUGE-L은 모두 생성된 문장이 정답 문장과 얼마나 겹치는지 보는 지표다.

차이는 다음이다.

- Token F1은 정답 단어와 생성 단어가 얼마나 많이 겹치는지 본다.
- ROUGE-L은 단어가 단순히 겹치는지를 넘어, 정답과 생성문 사이에서 순서가 유지되는 가장 긴 공통 흐름을 본다.

따라서 Token F1은 "내용 조각을 얼마나 회수했는가"에 가깝고, ROUGE-L은 "정답 문장의 흐름을 얼마나 따라갔는가"에 가깝다.

## Token F1

Token F1은 생성문과 정답문 사이의 단어 겹침을 precision과 recall의 균형으로 본다.

직관적으로는 다음 두 질문을 동시에 묻는다.

- 생성문에 들어간 단어 중 정답에도 있는 단어가 얼마나 많은가?
- 정답에 있던 단어 중 생성문이 얼마나 회수했는가?

예를 들어 정답이 다음과 같다고 하자.

```text
the synagogue was rebuilt in the original location
```

생성문이 다음과 같다면:

```text
synagogue rebuilt original location
```

핵심 단어를 꽤 회수했으므로 Token F1은 비교적 높게 나온다. 하지만 문장 순서, 문법, 자연스러움까지 깊게 평가하지는 않는다.

## ROUGE-L

ROUGE-L은 정답문과 생성문 사이의 가장 긴 공통 부분 순서를 본다. 여기서 중요한 점은 단어가 완전히 연속으로 붙어 있어야 하는 것은 아니지만, 순서는 유지되어야 한다는 것이다.

예를 들어 생성문이 다음과 같다면:

```text
original synagogue location rebuilt
```

정답 단어와 많이 겹칠 수는 있지만, 정답 문장의 순서와는 다르다. 이런 경우 Token F1은 어느 정도 높게 나올 수 있지만, ROUGE-L은 낮아질 수 있다.

## S2에서의 해석 규칙

S2에서는 두 지표를 다음처럼 읽는다.

| 관찰 | 해석 |
|---|---|
| Token F1이 높다 | 원문에 있던 단어와 내용 조각을 많이 회수했다. |
| ROUGE-L이 높다 | 원문의 단어 흐름이나 문장 순서를 어느 정도 따라갔다. |
| Token F1은 높은데 ROUGE-L이 낮다 | 키워드는 맞췄지만 문장 구조나 순서가 흔들렸다. |
| 둘 다 높다 | 내용 단서와 순서 흐름이 함께 살아났다. |

## S2 핵심 비교와 연결

S2의 핵심 비교 중 하나는 `attention_scaffold`와 `random_scaffold`다.

```text
attention_scaffold: Token F1 0.3830, ROUGE-L 0.3117
random_scaffold:    Token F1 0.2286, ROUGE-L 0.1789
```

이 결과는 attention 기반 의미 골격이 무작위 골격보다 정답 단어를 더 많이 회수했고, 정답 문장의 흐름도 더 잘 따라갔다는 뜻이다.

다만 이 지표들은 문장 품질 전체를 보장하지 않는다. Token F1과 ROUGE-L은 정답과의 표면적 겹침을 보는 지표이므로, open-ended generation 품질, 사실성, 자연스러움, 사람 선호도는 별도의 평가가 필요하다.

## 관련 개념

- [[concepts/lace/의미-골격|의미 골격]]
- [[concepts/lace/위치-보조-구조|위치 보조 구조]]
- [[concepts/lace/s2-의미-골격-문장-복원-학습|S2 의미 골격-문장 복원 학습]]
