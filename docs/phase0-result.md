# Phase 0 Result

**Run**: Kaggle kernel `dennisparknd/lace-phase-0-latent-cache`, version 2  
**Status**: Complete  
**Purpose**: Verify the first runnable LACE latent-cache pipeline.

## Configuration

| Item | Value |
|---|---|
| Model | `t5-small` |
| Device | `cuda` |
| Samples | `128` |
| Max length | `128` |
| Hidden shape | `[128, 128, 512]` |
| Hidden dtype | `torch.float16` |
| Active tokens | `2661` |
| Cache reload exact match | `true` |

## Stage Metrics

| Stage | Tokens | Shape | MSE | Cosine | Finite |
|---|---:|---|---:|---:|---|
| `z1` | 64 | `[128, 64, 512]` | 0.008886 | 0.787159 | true |
| `z2` | 32 | `[128, 32, 512]` | 0.013866 | 0.642822 | true |
| `z3` | 16 | `[128, 16, 512]` | 0.017505 | 0.522215 | true |

## Interpretation

Phase 0 passed. The frozen encoder path, staged pooling, latent cache write/read,
and basic metrics all work on Kaggle GPU. The stage curve behaves as expected:
smaller latent token budgets produce higher reconstruction MSE and lower cosine
similarity.

The run used the fallback text corpus because no Kaggle dataset source was
attached yet. Phase 1 should attach a small WikiText-2 or XSum subset and train
the first reverse expander against cached `h0` latents.

