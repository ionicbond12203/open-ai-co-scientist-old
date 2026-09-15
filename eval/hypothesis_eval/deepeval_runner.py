"""Run one case at a time through DeepEval using an LM Studio judge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.test_case import LLMTestCase
from deepeval.test_run.test_run import TestRunResultDisplay

from .artifacts import EvaluationCase
from .metrics import METRIC_NAMES, build_metrics


def evaluate_case(
    case: EvaluationCase,
    *,
    model: Any,
    threshold: float,
    results_folder: Path | None,
    print_results: bool,
) -> dict[str, Any]:
    test_case = LLMTestCase(
        input=case.goal,
        actual_output=case.hypothesis_text,
        metadata={
            "run_id": case.run_id,
            "hypothesis_id": case.hypothesis_id,
            # Kept in metadata for the report; metric prompts do not receive it.
            "system": case.system,
        },
    )
    metrics = build_metrics(model=model, threshold=threshold)
    result = evaluate(
        test_cases=[test_case],
        metrics=metrics,
        identifier=f"{case.system}-{case.run_id}",
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(
            print_results=print_results,
            show_indicator=print_results,
            display_option=TestRunResultDisplay.ALL,
            results_folder=str(results_folder) if results_folder else None,
            inspect_after_run=False,
        ),
        cache_config=CacheConfig(write_cache=False),
    )
    test_results = result.test_results or []
    if len(test_results) != 1:
        raise RuntimeError(f"DeepEval returned {len(test_results)} test results for one case")
    metric_data = test_results[0].metrics_data or []
    if len(metric_data) != len(METRIC_NAMES):
        raise RuntimeError(f"DeepEval returned {len(metric_data)} metrics; expected {len(METRIC_NAMES)}")

    scores: dict[str, float] = {}
    reasons: dict[str, str | None] = {}
    for name, measured in zip(METRIC_NAMES, metric_data):
        if measured.error:
            raise RuntimeError(f"metric {name!r} failed: {measured.error}")
        if isinstance(measured.score, bool) or not isinstance(measured.score, (int, float)):
            raise RuntimeError(f"metric {name!r} returned no numeric score")
        scores[name] = float(measured.score)
        reasons[name] = measured.reason if isinstance(measured.reason, str) else None
    return {"scores": scores, "reasons": reasons}
