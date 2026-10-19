"""Awareness target: equality of the clean and contextual canonical argmax labels.

Class order is ``[disagree, agree]``. The unrestricted greedy token is not an
input. Raw versus canonical filtering changes which rows are eligible and
does not change this label.
"""

from __future__ import annotations

from dataclasses import dataclass

from knowledge_recovery.records import TaskSpec
from knowledge_recovery.tasks.base import (
    AWARENESS_CLASS_ORDER,
    AWARENESS_PREDICTION_FEATURES,
    AWARENESS_TASK_ID,
    canonical_label,
)

TARGET_VERSION = "canonical-argmax-equality-v1"


def awareness_label(clean_canonical_label: str, contextual_canonical_label: str) -> str:
    """Return ``agree`` when the two canonical argmax labels are equal."""
    clean = canonical_label(clean_canonical_label, "clean_canonical_label")
    contextual = canonical_label(contextual_canonical_label, "contextual_canonical_label")
    return "agree" if clean == contextual else "disagree"


def awareness_target_index(clean_canonical_label: str, contextual_canonical_label: str) -> int:
    """Index in ``[disagree, agree]``. Agreement is class 1."""
    return AWARENESS_CLASS_ORDER.index(awareness_label(clean_canonical_label, contextual_canonical_label))


def awareness_task_spec() -> TaskSpec:
    """The registered awareness contract. Greedy equality is not the target."""
    return TaskSpec(
        task_id=AWARENESS_TASK_ID,
        input_schema={
            "prediction_features": list(AWARENESS_PREDICTION_FEATURES),
            "target_fields": ["clean_canonical_label", "contextual_canonical_label"],
        },
        output_schema={"class_order": list(AWARENESS_CLASS_ORDER), "positive_class": "agree"},
        target_construction={
            "version": TARGET_VERSION,
            "rule": "agree when the clean and contextual canonical argmax labels are equal",
            "not_used": ["unrestricted_greedy_token"],
            "filter_effect": "eligibility_only",
        },
        class_meaning={
            "disagree": "canonical argmax labels differ",
            "agree": "canonical argmax labels are equal",
        },
        class_order=AWARENESS_CLASS_ORDER,
        permitted_prediction_features=AWARENESS_PREDICTION_FEATURES,
        decoding_convention={"tie_probability": 0.5, "tie_prediction": "agree"},
        pointwise_score={
            "name": "brier",
            "convention": "scalar_positive_class",
            "positive_class": "agree",
        },
    )


@dataclass(frozen=True)
class AwarenessTask:
    """Constructs agreement targets from cached canonical labels only."""

    task_id: str = AWARENESS_TASK_ID
    class_order: tuple[str, ...] = AWARENESS_CLASS_ORDER

    def target(self, clean_canonical_label: str, contextual_canonical_label: str) -> str:
        return awareness_label(clean_canonical_label, contextual_canonical_label)

    def target_index(self, clean_canonical_label: str, contextual_canonical_label: str) -> int:
        return awareness_target_index(clean_canonical_label, contextual_canonical_label)
