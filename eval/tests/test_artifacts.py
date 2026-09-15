import json

import pytest

from hypothesis_eval.artifacts import ArtifactError, load_case, load_comparison_cases


def write_run(path, *, goal="Test goal", ranking_step="ranking2", score=1200, title="Top"):
    path.write_text(
        json.dumps(
            {
                "run_id": path.stem,
                "research_goal": {"description": goal},
                "cycle_details": {
                    "steps": {
                        "generation": {"hypotheses": [{"id": "H1", "title": "First", "text": "Body"}]},
                        ranking_step: {
                            "hypotheses": [
                                {"id": "H1", "title": "Lower", "text": "Body", "elo_score": 900},
                                {"id": "H2", "title": title, "text": "Selected body", "elo_score": score},
                            ]
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_load_case_selects_latest_ranking_winner(tmp_path):
    path = tmp_path / "run.json"
    write_run(path)

    case = load_case(path, "current")

    assert case.source_step == "ranking2"
    assert case.hypothesis_id == "H2"
    assert case.hypothesis_text == "Title: Top\n\nHypothesis: Selected body"


def test_load_case_accepts_reference_ranking_final(tmp_path):
    path = tmp_path / "reference.json"
    write_run(path, ranking_step="ranking_final")

    case = load_case(path, "reference")

    assert case.source_step == "ranking_final"
    assert case.hypothesis_id == "H2"


def test_comparison_rejects_different_goals(tmp_path):
    current = tmp_path / "current.json"
    reference = tmp_path / "reference.json"
    write_run(current, goal="Goal A")
    write_run(reference, goal="Goal B")

    with pytest.raises(ArtifactError, match="different research goals"):
        load_comparison_cases([current], [reference])


def test_comparison_interleaves_blinded_pairs(tmp_path):
    paths = [tmp_path / name for name in ("c1.json", "c2.json", "r1.json", "r2.json")]
    for path in paths:
        write_run(path)

    cases = load_comparison_cases(paths[:2], paths[2:])

    assert [(case.system, case.run_id) for case in cases] == [
        ("current", "c1"),
        ("reference", "r1"),
        ("current", "c2"),
        ("reference", "r2"),
    ]
