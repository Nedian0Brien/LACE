# Phase 0 Kaggle 실행 안내

Phase 0은 실행 가능한 가장 작은 LACE 실험을 검증한다.

```text
texts -> frozen T5 encoder -> h0 latent cache -> z1/z2/z3 pooling sanity metrics
```

## 업로드 및 실행

```bash
bash scripts/push_kaggle_phase0.sh
```

동일한 명령을 직접 쓰면 다음과 같다.

```bash
kaggle kernels push -p kaggle/phase0 --accelerator NvidiaTeslaT4 --timeout 3600
```

## 상태 확인

```bash
kaggle kernels status dennisparknd/lace-phase-0-latent-cache
```

## 결과 다운로드

```bash
mkdir -p outputs/phase0
kaggle kernels output dennisparknd/lace-phase-0-latent-cache -p outputs/phase0
```

예상 산출물:

- `metrics.json`
- `summary.md`
- `latent_cache.pt`

## 성공 기준

- `t5-small` 기준 `hidden_shape`가 `[sample_count, 128, 512]`다.
- `cache_allclose`가 `true`다.
- 단계별 shape가 `[sample_count, 64, 512]`, `[sample_count, 32, 512]`, `[sample_count, 16, 512]`다.
- 목표 token 수가 줄어들수록 MSE는 증가하고 cosine은 대체로 감소한다.
