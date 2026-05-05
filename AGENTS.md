# AGENTS.md

## Communication Preference

사용자에게 간결하게, 핵심을 전달합니다.
방향성, 본질에 집중하여 직관적으로 설명합니다.

Give interpretive context rather than terse summaries, but match length to the question. Design and discussion questions get a recommendation plus the main trade-off in 2–3 short paragraphs. Do not expand into adjacent decisions the user did not ask about. Catalog-style enumeration of every option is the wrong default.

Use the following 5-step structure **only** for finalized experiment results, metric/gate definitions, and next-step plans. **Do not apply it to option comparisons, scorer candidates, or design alternatives** — those use a short table plus one or two lines per option.

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
