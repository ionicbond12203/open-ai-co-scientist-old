# Hypothesis quality comparison

This independent `uv` project compares persisted hypotheses from this checkout
with persisted hypotheses from `Siong23/ai-co-scientist`. It deliberately does
not add DeepEval to the Gradio application's production dependencies.

The harness uses the same fixed LM Studio judge and the same five
non-retrieval metrics for both systems:

1. Goal alignment
2. Experimental readiness
3. Scientific testability
4. Feasibility
5. Scientific plausibility

Novelty is excluded unless both systems are supplied the same retrieved prior
art. Asking a judge for novelty without shared evidence would measure its
background knowledge, not the two generators.

## Environment

The local `.env` is already configured for:

```text
qwen/qwen3.8-27b @ http://100.117.90.5:1234/v1
```

It uses `lm-studio` as a non-secret placeholder key. `.env`, `.venv`, and
generated reports are ignored by Git. Recreate the environment with:

```powershell
cd eval
uv sync
```

## Evaluate one saved run

The single-run command matches the reference repository's evaluator interface.
From `eval/`, pass the saved run and the goal file that exactly matches its
`research_goal.description`:

```powershell
uv run python scripts/evaluate_run.py `
  ../results/runs/<run-id>.json `
  --goal-file goals/<goal>.txt `
  --llm-metrics `
  --metric-suite hypothesis
```

DeepEval prints its native results in the terminal and writes the full audit
report under `eval/reports/`. The `rag` and `all` suites are also available,
but evidence-dependent metrics are skipped when a run has no substantive
`evidence_sources` passages.

## Fair comparison protocol

Use the same goal and generation settings for both projects. Run each system at
least three times per goal; one stochastic generation is not enough to support
a quality claim. The five goal files here are copied from the reference repo's
benchmark set.

This project saves runs under `results/runs/*.json`. Produce equivalent run
JSON from the reference checkout, then validate a pair without calling LM
Studio:

```powershell
cd eval
uv run python scripts/compare_runs.py `
  --current-run ..\results\runs\<current-run>.json `
  --reference-run ..\.cache\ai-co-scientist-reference\results\runs\<reference-run>.json `
  --validate-only
```

Run the DeepEval comparison by removing `--validate-only`:

```powershell
uv run python scripts/compare_runs.py `
  --current-run ..\results\runs\<current-run>.json `
  --reference-run ..\.cache\ai-co-scientist-reference\results\runs\<reference-run>.json
```

Repeat both flags for repeated paired runs. The report is written under
`eval/reports/`. `delta_current_minus_reference > 0` favors this checkout;
negative values favor the reference implementation. System labels live only in
DeepEval metadata and are not included in the metric prompt.

## Checks

```powershell
uv run pytest
uv run ruff check .
```
