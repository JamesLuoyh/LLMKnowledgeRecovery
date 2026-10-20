"""Knowledge-consistency target: the hard clean canonical argmax.

Class order is ``[A, B]``. The target is not the soft clean distribution and
not an unrestricted token outside A/B. Filtering changes eligibility only.
"""

from __future__ import annotations

from dataclasses import dataclass

from knowledge_recovery.records import TaskSpec
from knowledge_recovery.tasks.base import (
    CONSISTENCY_CLASS_ORDER,
    CONSISTENCY_PREDICTION_FEATURES,
    CONSISTENCY_TASK_ID,
    canonical_label,
)

TARGET_VERSION = "clean-canonical-argmax-v1"


def consistency_target(clean_canonical_label: str) -> str:
    """Return the clean canonical argmax, already restricted to A or B."""
    return canonical_label(clean_canonical_label, "clean_canonical_label")


def consistency_target_index(clean_canonical_label: str) -> int:
    """Index in ``[A, B]``. B is class 1."""
    return CONSISTENCY_CLASS_ORDER.index(consistency_target(clean_canonical_label))


def consistency_task_spec() -> TaskSpec:
    """The registered consistency contract. Ties are resolved before this label."""
    return TaskSpec(
        task_id=CONSISTENCY_TASK_ID,
        input_schema={
            "prediction_features": list(CONSISTENCY_PREDICTION_FEATURES),
            "target_fields": ["clean_canonical_label"],
        },
        output_schema={"class_order": list(CONSISTENCY_CLASS_ORDER), "positive_class": "B"},
        target_construction={
            "version": TARGET_VERSION,
            "rule": "hard clean canonical argmax",
            "not_used": ["soft_clean_distribution", "unrestricted_greedy_token"],
            "filter_effect": "eligibility_only",
        },
        class_meaning={"A": "clean canonical argmax A", "B": "clean canonical argmax B"},
        class_order=CONSISTENCY_CLASS_ORDER,
        permitted_prediction_features=CONSISTENCY_PREDICTION_FEATURES,
        decoding_convention={"tie_probability": 0.5, "tie_prediction": "A"},
        pointwise_score={"name": "brier", "convention": "scalar_positive_class", "positive_class": "B"},
    )


@dataclass(frozen=True)
class ConsistencyTask:
    """Constructs the clean-label target. Prediction code does not receive it as a feature."""

    task_id: str = CONSISTENCY_TASK_ID
    class_order: tuple[str, ...] = CONSISTENCY_CLASS_ORDER

    def target(self, clean_canonical_label: str) -> str:
        return consistency_target(clean_canonical_label)

    def target_index(self, clean_canonical_label: str) -> int:
        return consistency_target_index(clean_canonical_label)
