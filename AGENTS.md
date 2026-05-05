# AGENTS.md

## Communication Preference

When explaining this research project, keep the explanation at a detailed interpretive level rather than a terse summary.

For experiment results, metrics, gates, or next-step planning, prefer the following structure:

1. Explain what the metric, control, or experiment item measures.
2. Explain why it matters in the current LACE/DLM-ALM experiment.
3. Explain what a good or bad result would imply.
4. Explain what objection, confound, or ambiguity the item helps address.
5. Explain how the next experiment should verify or falsify the interpretation.

Be especially careful to separate:

- hidden-state reconstruction quality
- representation direction or semantic structure
- token-level reconstruction proxy
- open-ended generation behavior
- evidence for actual latent use
- claims about compression forward process versus corruption forward process

Avoid overclaiming from early smoke runs. State both the defensible interpretation and the remaining caveats.

## Development Workflow

Do not use test-driven development (TDD) in this research project.

For this repo, prefer direct implementation plus focused verification after the change. Do not run or create routine test suites for documentation-only or planning-only work. Use lightweight checks such as `git diff --check`, syntax checks, small smoke runs, or targeted metric/gate checks only when they materially reduce risk.

At the end of each completed task, automatically commit and push the completed work unless the user explicitly says not to. The commit should include the relevant code, experiment plan/result documents, and verification updates for that task, while preserving unrelated user changes.

When running a Kaggle-backed experiment, consult `docs/v2/kaggle-experiment-workflow.md` before planning, implementing, pushing, downloading outputs, or reporting results.
