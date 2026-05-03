# LACE

**LACE: Latent Adaptive Compression and Expansion for Language Diffusion**

This repository contains the small-scale research scaffold for testing the core
claim behind *Compression, Not Corruption*: language diffusion time should be
treated as an information-rate path rather than a random corruption schedule.

The current runnable target is **Phase 0**, a Kaggle-friendly sanity check:

1. Load a small text batch.
2. Encode it with a frozen `t5-small` encoder.
3. Produce staged latent shapes with average pooling.
4. Save and reload a latent cache.
5. Write `metrics.json` and `summary.md`.

## Phase 0 on Kaggle

```bash
kaggle kernels push -p kaggle/phase0 --accelerator NvidiaTeslaT4 --timeout 3600
kaggle kernels status dennisparknd/lace-phase-0-latent-cache
kaggle kernels output dennisparknd/lace-phase-0-latent-cache -p outputs/phase0
```

The Kaggle script is self-contained so it can run as a Kaggle script kernel
without packaging the full repository.

## Local Smoke Check

The local environment does not need to have GPU dependencies installed just to
check syntax:

```bash
python3 -m py_compile kaggle/phase0/run_phase0.py
```

To run the actual encoder locally, install PyTorch and Transformers first.
