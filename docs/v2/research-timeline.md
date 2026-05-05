# V2 연구 타임라인

이 문서는 v2 semantic skeleton track의 연구 질문, 결정, empirical fact, caveat, 해석 변경을 시간순으로 기록한다.

## 2026-05-05

### 결정: V2 핵심 Claim

맥락:

S0에서 skeleton pipeline이 실행 가능하다는 점을 확인했지만, 동시에 강한 lead-position confound도 확인했다. 이에 따라 v2 연구 방향을 더 좁게 정리했다.

결정:

> Semantic skeleton + positional scaffold creates a better reverse trajectory than random corruption.

근거/source:

- `docs/v2/experiments/s0-skeleton-pipeline.md`
- `outputs/v2_s0/lace_v2_s0/summary.md`

해석:

Semantic skeleton은 content-bearing terminal state로 취급한다. Positional scaffold는 유용하지만 skeleton을 text로 확장하기 위한 보조 구조다. Positional scaffold만으로 좋아진 결과를 semantic skeleton use의 증거로 취급하면 안 된다.

다음 실험에 주는 의미:

S1에서는 더 큰 reconstruction 또는 generation claim으로 넘어가기 전에 correct, shuffled, random, wrong-document, position-prior, position-only, same-position random, top-k/low-k removal control로 skeleton use를 검증해야 한다.
