# Phase 0 결과

**실행**: Kaggle kernel `dennisparknd/lace-phase-0-latent-cache`, 버전 2
**상태**: 완료
**목적**: 첫 번째 실행 가능한 LACE latent-cache pipeline 검증

## 설정

| 항목 | 값 |
|---|---|
| 모델 | `t5-small` |
| 장치 | `cuda` |
| 샘플 수 | `128` |
| 최대 길이 | `128` |
| hidden shape | `[128, 128, 512]` |
| hidden dtype | `torch.float16` |
| 유효 token 수 | `2661` |
| cache 재로딩 일치 여부 | `true` |

## 단계별 지표

| 단계 | Token 수 | Shape | MSE | Cosine | 유한값 여부 |
|---|---:|---|---:|---:|---|
| `z1` | 64 | `[128, 64, 512]` | 0.008886 | 0.787159 | true |
| `z2` | 32 | `[128, 32, 512]` | 0.013866 | 0.642822 | true |
| `z3` | 16 | `[128, 16, 512]` | 0.017505 | 0.522215 | true |

## 해석

Phase 0은 통과했다. 고정 encoder 경로, 단계별 pooling, latent cache 저장/재로딩, 기본 지표 계산이 모두 Kaggle GPU에서 정상 동작했다. 단계별 곡선도 예상대로 움직였다. latent token 예산이 작아질수록 reconstruction MSE는 커지고 cosine similarity는 낮아졌다.

이번 실행은 아직 Kaggle dataset source를 붙이지 않았기 때문에 예비 텍스트 corpus를 사용했다. Phase 1에서는 작은 WikiText-2 또는 XSum subset을 연결하고, cache된 `h0` latent를 대상으로 첫 번째 reverse expander를 학습해야 한다.
