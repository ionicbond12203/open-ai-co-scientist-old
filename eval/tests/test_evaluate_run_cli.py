import json

from scripts.evaluate_run import build_parser, parse_run


def test_reference_style_single_run_command_parses_current_artifact_shape(tmp_path):
    goal = "Develop a closed-loop multi-agent AI framework."
    run_path = tmp_path / "run-test.json"
    goal_path = tmp_path / "goal.txt"
    run_path.write_text(
        json.dumps(
            {
                "run_id": "run-test",
                "research_goal": {"description": goal},
                "cycle_details": {
                    "steps": {
                        "ranking2": {
                            "hypotheses": [
                                {
                                    "id": "H1",
                                    "title": "Test hypothesis",
                                    "text": "Changing X will improve Y compared with baseline Z.",
                                    "elo_score": 1234.0,
                                }
                            ]
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    goal_path.write_text(goal, encoding="utf-8")

    args = build_parser().parse_args(
        [
            str(run_path),
            "--goal-file",
            str(goal_path),
            "--llm-metrics",
            "--metric-suite",
            "hypothesis",
        ]
    )
    parsed = parse_run(args.run_json, args.goal_file)

    assert args.llm_metrics is True
    assert args.metric_suite == "hypothesis"
    assert parsed["run_id"] == "run-test"
    assert parsed["hypothesis_source_step"] == "ranking2"
    assert parsed["selected_hypothesis"]["id"] == "H1"
