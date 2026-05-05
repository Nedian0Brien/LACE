# V2 Research Timeline

This timeline records meaningful research questions, decisions, empirical facts, caveats, and interpretation changes for the v2 semantic skeleton track.

## 2026-05-05

### Decision: Core V2 Claim

Context: After S0 confirmed that the skeleton pipeline is runnable but also exposed a strong lead-position confound, the research direction was narrowed.

Decision:

> Semantic skeleton + positional scaffold creates a better reverse trajectory than random corruption.

Evidence/source:

- `docs/v2/experiments/s0-skeleton-pipeline.md`
- `outputs/v2_s0/lace_v2_s0/summary.md`

Interpretation:

The semantic skeleton should be treated as the content-bearing terminal state. The positional scaffold is useful, but only as auxiliary expansion structure. It cannot by itself prove semantic skeleton use.

Implication:

S1 should test skeleton use with correct, shuffled, random, wrong-document, position-prior, position-only, same-position random, and top-k/low-k removal controls before moving to larger reconstruction or generation claims.
