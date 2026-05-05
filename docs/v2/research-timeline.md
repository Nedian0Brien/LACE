# V2 연구 타임라인

이 문서는 v2 의미 골격 연구 흐름의 연구 질문, 결정, 경험적 사실, 주의점, 해석 변경을 시간순으로 기록한다.

## 2026-05-05

### 결정: 답변과 연구 기록은 한글 용어를 우선한다

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
