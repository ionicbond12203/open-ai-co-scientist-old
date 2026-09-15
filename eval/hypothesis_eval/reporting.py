"""Aggregate case-level metric results into a two-system comparison."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from .artifacts import EvaluationCase
from .metrics import METRIC_NAMES


def _system_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_means = {
        name: round(statistics.fmean(float(result["scores"][name]) for result in results), 4) for name in METRIC_NAMES
    }
    return {
        "run_count": len(results),
        "metric_means": metric_means,
        "overall_mean": round(statistics.fmean(metric_means.values()), 4),
    }


def build_comparison_report(
    cases: Sequence[EvaluationCase],
    evaluations: Sequence[Mapping[str, Any]],
    *,
    judge: Mapping[str, str],
    threshold: float,
) -> dict[str, Any]:
    if len(cases) != len(evaluations):
        raise ValueError("every case must have exactly one evaluation")

    rows: list[dict[str, Any]] = []
    for case, evaluation in zip(cases, evaluations):
        scores = evaluation.get("scores")
        if not isinstance(scores, Mapping) or set(scores) != set(METRIC_NAMES):
            raise ValueError("evaluation does not contain the expected metrics")
        rows.append(
            {
                "system": case.system,
                "run_id": case.run_id,
                "hypothesis_id": case.hypothesis_id,
                "hypothesis_source_step": case.source_step,
                "hypothesis": case.hypothesis_text,
                "scores": {name: float(scores[name]) for name in METRIC_NAMES},
                "reasons": dict(evaluation.get("reasons", {})),
            }
        )

    by_system = {system: [row for row in rows if row["system"] == system] for system in ("current", "reference")}
    summaries = {system: _system_summary(system_rows) for system, system_rows in by_system.items()}
    deltas = {
        name: round(summaries["current"]["metric_means"][name] - summaries["reference"]["metric_means"][name], 4)
        for name in METRIC_NAMES
    }
    deltas["overall_mean"] = round(summaries["current"]["overall_mean"] - summaries["reference"]["overall_mean"], 4)

    paired_outcomes = {"current_wins": 0, "reference_wins": 0, "ties": 0}
    current_rows = by_system["current"]
    reference_rows = by_system["reference"]
    for current, reference in zip(current_rows, reference_rows):
        current_mean = statistics.fmean(current["scores"].values())
        reference_mean = statistics.fmean(reference["scores"].values())
        if abs(current_mean - reference_mean) < 1e-12:
            paired_outcomes["ties"] += 1
        elif current_mean > reference_mean:
            paired_outcomes["current_wins"] += 1
        else:
            paired_outcomes["reference_wins"] += 1

    return {
        "schema_version": 1,
        "method": {
            "selection": "highest Elo hypothesis from the final ranking step",
            "metrics": list(METRIC_NAMES),
            "threshold": threshold,
            "novelty_excluded": "No shared retrieved-prior-art context was supplied.",
        },
        "judge": dict(judge),
        "research_goal": cases[0].goal if cases else None,
        "systems": summaries,
        "delta_current_minus_reference": deltas,
        "paired_outcomes": paired_outcomes,
        "cases": rows,
    }
