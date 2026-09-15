from hypothesis_eval.artifacts import EvaluationCase
from hypothesis_eval.metrics import METRIC_NAMES
from hypothesis_eval.reporting import build_comparison_report


def case(system, run_id):
    return EvaluationCase(
        system=system,
        run_id=run_id,
        goal="Same goal",
        source_step="ranking2",
        hypothesis={"id": run_id, "title": "T", "text": "H"},
    )


def evaluation(score):
    return {"scores": {name: score for name in METRIC_NAMES}, "reasons": {name: "why" for name in METRIC_NAMES}}


def test_report_aggregates_current_minus_reference():
    cases = [case("current", "c1"), case("reference", "r1")]

    report = build_comparison_report(
        cases,
        [evaluation(0.8), evaluation(0.6)],
        judge={"provider": "lm_studio", "model": "judge", "base_url": "http://localhost/v1"},
        threshold=0.7,
    )

    assert report["systems"]["current"]["overall_mean"] == 0.8
    assert report["systems"]["reference"]["overall_mean"] == 0.6
    assert report["delta_current_minus_reference"]["overall_mean"] == 0.2
    assert report["paired_outcomes"] == {"current_wins": 1, "reference_wins": 0, "ties": 0}
    assert "api_key" not in report["judge"]
