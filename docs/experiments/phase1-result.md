# Phase 1 결과

> 이 문서는 Phase 1 version 1 fallback smoke run 결과다. WikiText-2 subset으로 재실행한 최신 결과는 [docs/experiments/phase-1-wikitext-2-subset.md](./experiments/phase-1-wikitext-2-subset.md)에 정리했다.

**실행**: Kaggle kernel `dennisparknd/lace-phase-1-latent-compression`, version 1
**상태**: 완료
**목적**: LACE의 compression path와 reverse expander가 학습 가능한 latent reconstruction loop를 형성하는지 검증

## 실행 정보

- Kaggle kernel: `dennisparknd/lace-phase-1-latent-compression`
- Script: `kaggle/phase1/run_phase1.py`
- 실행 script: `scripts/push_kaggle_phase1.sh`
- 장치: `cuda`
- 모델: 고정 `t5-small` encoder
- 샘플 수: `512`
- 데이터 소스: `fallback`
- max length: `128`
- stage tokens: `64, 32, 16`
- hidden shape: `[512, 128, 512]`
- active tokens: `10720`
- cache 재로딩 일치 여부: `true`

## 검증 질문별 결론

1. `z1/z2/z3`에서 `h0`를 복원하는 expander의 validation loss가 감소하는가?  
   → **감소했다.** pooling과 attention 모두 validation loss가 크게 줄었다.
2. 압축 단계가 강해질수록 복원 난도가 증가하는가?  
   → **관측됐다.** 두 모드 모두 `z1 → z2 → z3`로 갈수록 MSE가 증가하고 cosine이 감소했다.
3. `z3`가 너무 쉬운 shortcut도 아니고 완전 collapse도 아닌가?  
   → **pooling은 통과, attention은 부분 통과**다. attention은 variance는 남아 있지만 perturbation sensitivity가 낮다.
4. learned attention compression이 average pooling baseline과 비교 가능한 수준으로 학습되는가?  
   → **loss는 학습됐지만 아직 baseline보다 약하다.** attention compression은 loss 감소에는 성공했으나 latent 사용 gate는 실패했다.

## Mode별 결과

### Pooling baseline

| 항목 | 값 |
|---|---:|
| train loss | 0.079468 → 0.025681 |
| validation loss | 0.257082 → 0.023852 |
| perturbation sensitivity | 0.012277 |
| overall gate | pass |

| Stage | MSE | Cosine | Variance ratio | Finite |
|---|---:|---:|---:|---|
| `z1` | 0.005847 | 0.884392 | 0.636817 | true |
| `z2` | 0.012254 | 0.714241 | 0.452986 | true |
| `z3` | 0.016237 | 0.593534 | 0.398918 | true |

### Learned attention compression

| 항목 | 값 |
|---|---:|
| train loss | 0.415063 → 0.056589 |
| validation loss | 1.068796 → 0.054152 |
| perturbation sensitivity | 0.001722 |
| overall gate | fail |

| Stage | MSE | Cosine | Variance ratio | Finite |
|---|---:|---:|---:|---|
| `z1` | 0.015838 | 0.619003 | 0.133218 | true |
| `z2` | 0.017007 | 0.571143 | 0.112686 | true |
| `z3` | 0.017687 | 0.544207 | 0.090807 | true |

## 성공 Gate 판정

| Gate | Pooling | Attention | 해석 |
|---|---|---|---|
| P1-G1. 학습 가능성 | pass | pass | 두 모드 모두 validation loss가 크게 감소 |
| P1-G2. 단계 곡선 | pass | pass | 두 모드 모두 MSE 증가, cosine 감소 곡선 형성 |
| P1-G3. 병목 유효성 | pass | pass | `z3`가 `z1`보다 명확히 어려운 병목으로 작동 |
| P1-G4. collapse 방지 | pass | fail | attention은 perturbation sensitivity가 낮아 latent 사용 신호가 약함 |

## 산출물

다운로드된 산출물:

- `outputs/phase1/lace_phase1/metrics.json`
- `outputs/phase1/lace_phase1/train_log.jsonl`
- `outputs/phase1/lace_phase1/summary.md`
- `outputs/phase1/lace_phase1/latent_cache.pt`
- `outputs/phase1/lace_phase1/checkpoints/`
- `outputs/phase1/lace-phase-1-latent-compression.log`

## 해석

Phase 1 smoke run은 **부분 성공**이다. 가장 중요한 확인점인 "reverse expander가 압축 latent에서 `h0` 복원 방향으로 학습되는가"는 확인됐다. Pooling baseline은 모든 gate를 통과했고, stage별 난도 곡선도 Phase 0보다 더 선명하게 정리됐다.

다만 learned attention compression은 아직 LACE-small의 근거로 쓰기에는 약하다. loss는 크게 줄었고 stage 곡선도 생겼지만, perturbation sensitivity가 0.001722로 낮아 `z3` latent 변화가 reconstruction에 충분히 강하게 반영되지 않았다. 즉 attention compression은 학습되긴 하지만, 현재 구조에서는 expander가 latent를 충분히 적극적으로 쓰지 않거나 attention latent가 낮은 분산의 압축 표현으로 수렴했을 가능성이 있다.

또한 이번 run은 실제 WikiText-2/XSum이 아니라 fallback corpus 512개 반복 기반이다. 따라서 이 결과는 **Phase 1 smoke 성공**으로 해석해야 하며, 연구 주장에는 아직 사용할 수 없다.

## 다음 조치

1. Kaggle dataset source로 WikiText-2 또는 XSum subset을 붙여 Phase 1을 다시 실행한다.
2. attention compression에 variance regularization을 더 강하게 걸거나 query dropout을 추가한다.
3. attention mode의 perturbation sensitivity 기준을 다시 측정하되, 기준 자체를 낮추기보다 latent 사용을 강화하는 쪽으로 수정한다.
4. pooling baseline은 Phase 2 baseline으로 보존한다.
