"""DeepEval 4.2.2 metrics aligned with the reference repository."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepeval.metrics import DAGMetric, GEval
from deepeval.metrics.dag import BinaryJudgementNode, DeepAcyclicGraph
from deepeval.test_case import SingleTurnParams

PARAMS = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    evaluation_steps: tuple[str, ...]


GEVAL_DEFINITIONS = (
    MetricDefinition(
        "Goal alignment",
        (
            "Identify the central objective and every explicit constraint stated in the research goal.",
            "Identify the intervention, mechanism, and intended outcome proposed by the hypothesis.",
            "Check each explicit goal constraint against the hypothesis and note any it ignores or contradicts.",
            "Score how directly and completely the hypothesis addresses the stated goal, penalizing tangential, "
            "generic, or only partially responsive proposals.",
        ),
    ),
    MetricDefinition(
        "Scientific testability",
        (
            "Identify the intervention or independent variable the hypothesis proposes to manipulate.",
            "Identify the dependent variable or outcome, and whether the text states how it would be measured.",
            "Identify any comparison, baseline, or control condition, and whether one is needed for this claim.",
            "Identify the predicted direction or size of the effect.",
            "Decide whether a researcher could design a concrete experiment whose result would falsify the "
            "hypothesis as written.",
            "Score only on the elements above. Award no credit for scientific vocabulary or confident tone not "
            "backed by stated variables, measurements, comparisons, or predictions.",
        ),
    ),
    MetricDefinition(
        "Feasibility",
        (
            "List the data, equipment, instrumentation, and measurements the proposed work would require, using "
            "only what the research goal and hypothesis state.",
            "Judge whether those resources are obtainable, or whether the text silently depends on proprietary "
            "datasets, unavailable hardware, or access it never mentions.",
            "Assess experimental complexity and implementation burden: scale, duration, expertise, and the number "
            "of steps that must succeed together.",
            "Note any dependency that is unrealistic or impossible as stated.",
            "Score how realistically this research could be executed on the supplied information. Do not assume "
            "access to resources the text does not mention.",
        ),
    ),
    MetricDefinition(
        "Scientific plausibility",
        (
            "State, step by step, the causal or mechanistic account the hypothesis proposes.",
            "Check whether each step follows from the previous one, or whether the argument jumps from correlation, "
            "analogy, or restatement to a causal claim.",
            "Check the mechanism for internal contradictions and physically, biologically, or computationally "
            "impossible assumptions.",
            "Check whether the predicted outcome actually follows from the stated mechanism.",
            "Score internal coherence and scientific reasoning only from the supplied text; do not claim to verify "
            "statements against external literature.",
        ),
    ),
)

_NO_REPAIR = (
    " Judge only what the research goal and hypothesis actually say. Do not supply a missing piece from domain "
    "knowledge or convention. If you must infer it, answer no."
)


def build_readiness_dag() -> DeepAcyclicGraph:
    """Build a fresh four-rung experimental-readiness decision tree."""
    intervention = BinaryJudgementNode(
        criteria=(
            "Does the hypothesis name a specific intervention, treatment, or independent variable an experimenter "
            "would deliberately manipulate, vary, or apply?" + _NO_REPAIR
        ),
        evaluation_params=PARAMS,
        label="intervention",
    )
    measurement = BinaryJudgementNode(
        criteria=(
            "Does it name a specific outcome, dependent variable, or observable, concretely enough to know what "
            "instrument or log would produce it?" + _NO_REPAIR
        ),
        evaluation_params=PARAMS,
        label="measurement",
    )
    comparison = BinaryJudgementNode(
        criteria=(
            "Does it state what the measured outcome is compared against: a control, untreated group, named "
            "baseline, alternative configuration, prior value, or target?" + _NO_REPAIR
        ),
        evaluation_params=PARAMS,
        label="comparison",
    )
    prediction = BinaryJudgementNode(
        criteria=(
            "Does it predict the direction or size of the effect, an outperformance, or a stated magnitude or "
            "threshold? Merely saying the effect will be investigated does not count." + _NO_REPAIR
        ),
        evaluation_params=PARAMS,
        label="prediction",
    )
    intervention.add_verdict(False, score=0)
    intervention.add_verdict(True, then=measurement)
    measurement.add_verdict(False, score=4)
    measurement.add_verdict(True, then=comparison)
    comparison.add_verdict(False, score=6)
    comparison.add_verdict(True, then=prediction)
    prediction.add_verdict(False, score=8)
    prediction.add_verdict(True, score=10)
    return DeepAcyclicGraph(root_nodes=[intervention])


def build_metrics(*, model: Any, threshold: float) -> list[Any]:
    """Return fresh metric objects; DAG nodes must not be reused across cases."""
    metrics: list[Any] = [
        GEval(
            name=definition.name,
            evaluation_params=PARAMS,
            evaluation_steps=list(definition.evaluation_steps),
            threshold=threshold,
            model=model,
            async_mode=False,
        )
        for definition in GEVAL_DEFINITIONS
    ]
    metrics.insert(
        1,
        DAGMetric(
            name="Experimental readiness",
            dag=build_readiness_dag(),
            threshold=threshold,
            model=model,
            async_mode=False,
        ),
    )
    return metrics


METRIC_NAMES = (
    "Goal alignment",
    "Experimental readiness",
    "Scientific testability",
    "Feasibility",
    "Scientific plausibility",
)
