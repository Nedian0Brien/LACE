# Phase 0 Kaggle Runbook

Phase 0 verifies the smallest runnable LACE experiment:

```text
texts -> frozen T5 encoder -> h0 latent cache -> z1/z2/z3 pooling sanity metrics
```

## Push And Run

```bash
bash scripts/push_kaggle_phase0.sh
```

Equivalent explicit command:

```bash
kaggle kernels push -p kaggle/phase0 --accelerator NvidiaTeslaT4 --timeout 3600
```

## Check Status

```bash
kaggle kernels status dennisparknd/lace-phase-0-latent-cache
```

## Download Outputs

```bash
mkdir -p outputs/phase0
kaggle kernels output dennisparknd/lace-phase-0-latent-cache -p outputs/phase0
```

Expected files:

- `metrics.json`
- `summary.md`
- `latent_cache.pt`

## Success Criteria

- `hidden_shape` is `[sample_count, 128, 512]` for `t5-small`.
- `cache_allclose` is `true`.
- Stage shapes are `[sample_count, 64, 512]`, `[sample_count, 32, 512]`, and `[sample_count, 16, 512]`.
- MSE increases and cosine generally drops as the target token count shrinks.
