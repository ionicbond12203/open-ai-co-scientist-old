"""Parse comparable hypotheses from either Co-Scientist run format."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RANKING_STEP = re.compile(r"ranking(?:_?(\d+)|_final)?")


class ArtifactError(ValueError):
    """Raised when run artifacts cannot support a fair comparison."""


@dataclass(frozen=True)
class EvaluationCase:
    """One system's highest-ranked hypothesis for a fixed goal."""

    system: str
    run_id: str
    goal: str
    source_step: str
    hypothesis: Mapping[str, Any]

    @property
    def hypothesis_text(self) -> str:
        title = self.hypothesis.get("title")
        body = self.hypothesis.get("text")
        if not isinstance(body, str) or not body.strip():
            raise ArtifactError(f"{self.run_id}: selected hypothesis has no non-empty text")
        if isinstance(title, str) and title.strip():
            return f"Title: {title.strip()}\n\nHypothesis: {body.strip()}"
        return body.strip()

    @property
    def hypothesis_id(self) -> str | None:
        value = self.hypothesis.get("id", self.hypothesis.get("hypothesis_id"))
        return str(value) if value is not None else None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ArtifactError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{path}: JSON root must be an object")
    return value


def _goal_from_run(run: Mapping[str, Any], path: Path) -> str:
    research_goal = run.get("research_goal")
    if isinstance(research_goal, Mapping):
        goal = research_goal.get("description")
    elif isinstance(research_goal, str):
        goal = research_goal
    else:
        goal = run.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ArtifactError(f"{path}: run has no non-empty research goal")
    return goal.strip()


def _hypotheses_from_step(step: Any, *, path: Path, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(step, Mapping):
        return []
    hypotheses = step.get("hypotheses", [])
    if hypotheses is None:
        return []
    if not isinstance(hypotheses, list) or not all(isinstance(item, Mapping) for item in hypotheses):
        raise ArtifactError(f"{path}: step {name!r} hypotheses must be a list of objects")
    return hypotheses


def _elo(hypothesis: Mapping[str, Any], *, path: Path) -> float:
    value = hypothesis.get("elo_score", hypothesis.get("elo_rating", 0))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"{path}: hypothesis {hypothesis.get('id', '<unknown>')!r} has invalid Elo score")
    return float(value)


def _final_hypotheses(run: Mapping[str, Any], path: Path) -> tuple[str, list[Mapping[str, Any]]]:
    selected = run.get("selected_hypothesis")
    if isinstance(selected, Mapping):
        return "selected_hypothesis", [selected]

    direct = run.get("hypotheses")
    if isinstance(direct, list) and direct and all(isinstance(item, Mapping) for item in direct):
        return "hypotheses", direct

    cycle = run.get("cycle_details")
    steps = cycle.get("steps") if isinstance(cycle, Mapping) else None
    if not isinstance(steps, Mapping):
        raise ArtifactError(f"{path}: expected cycle_details.steps or a top-level hypothesis list")

    ranked: list[tuple[float, int, str]] = []
    for index, name in enumerate(steps):
        if not isinstance(name, str):
            continue
        match = RANKING_STEP.fullmatch(name)
        if match:
            priority = float("inf") if name == "ranking_final" else float(int(match.group(1) or 0))
            ranked.append((priority, index, name))

    for _, _, name in sorted(ranked, reverse=True):
        hypotheses = _hypotheses_from_step(steps[name], path=path, name=name)
        if hypotheses:
            return name, hypotheses

    # Some historical runs do not contain ranking steps. The most recent step
    # with hypotheses is the closest available approximation of final output.
    for name, step in reversed(list(steps.items())):
        hypotheses = _hypotheses_from_step(step, path=path, name=str(name))
        if hypotheses:
            return str(name), hypotheses
    raise ArtifactError(f"{path}: no hypotheses found")


def load_case(path: Path, system: str) -> EvaluationCase:
    run = _load_json(path)
    goal = _goal_from_run(run, path)
    source_step, hypotheses = _final_hypotheses(run, path)
    selected = max(hypotheses, key=lambda item: _elo(item, path=path))
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        run_id = path.stem
    case = EvaluationCase(
        system=system,
        run_id=run_id.strip(),
        goal=goal,
        source_step=source_step,
        hypothesis=selected,
    )
    # Validate before any judge request is made.
    case.hypothesis_text
    return case


def _normalized_goal(goal: str) -> str:
    return " ".join(goal.split()).casefold()


def load_comparison_cases(current_paths: Sequence[Path], reference_paths: Sequence[Path]) -> list[EvaluationCase]:
    """Load paired runs and reject comparisons with different goals/counts."""
    if not current_paths or not reference_paths:
        raise ArtifactError("at least one current and one reference run are required")
    if len(current_paths) != len(reference_paths):
        raise ArtifactError("current and reference run counts must match")

    current = [load_case(path, "current") for path in current_paths]
    reference = [load_case(path, "reference") for path in reference_paths]
    cases: list[EvaluationCase] = []
    for index, (current_case, reference_case) in enumerate(zip(current, reference), start=1):
        if _normalized_goal(current_case.goal) != _normalized_goal(reference_case.goal):
            raise ArtifactError(f"pair {index} has different research goals")
        cases.extend((current_case, reference_case))

    first_goal = _normalized_goal(cases[0].goal)
    if any(_normalized_goal(case.goal) != first_goal for case in cases[1:]):
        raise ArtifactError("all run pairs in one report must use the same research goal")
    return cases
