"""Shared task identities and the prediction-time feature boundary.

Targets may read cached labels. Those labels are not prediction features.
A contextual row cannot receive the clean row's answer, logits, or residual
as an extra input; the clean row's own residual is named ``residual``.
"""

from __future__ import annotations

from collections.abc import Sequence

AWARENESS_TASK_ID = "behavioral_awareness"
CONSISTENCY_TASK_ID = "knowledge_consistency"
AWARENESS_CLASS_ORDER = ("disagree", "agree")
CONSISTENCY_CLASS_ORDER = ("A", "B")
AWARENESS_PREDICTION_FEATURES = ("residual",)
CONSISTENCY_PREDICTION_FEATURES = ("residual", "canonical_ab_probabilities")
PREDICTION_FEATURES = {
    AWARENESS_TASK_ID: AWARENESS_PREDICTION_FEATURES,
    CONSISTENCY_TASK_ID: CONSISTENCY_PREDICTION_FEATURES,
}


def canonical_label(value: object, label: str) -> str:
    """Return an A/B canonical argmax label."""
    if value not in ("A", "B"):
        raise ValueError(f"{label} must be the canonical argmax A or B")
    return str(value)


def check_prediction_features(task_id: str, provided: Sequence[str]) -> None:
    """Reject features that are not declared for this task, before any tensor check."""
    if task_id not in PREDICTION_FEATURES:
        raise ValueError(f"unknown task {task_id}")
    if isinstance(provided, (str, bytes)) or not isinstance(provided, Sequence):
        raise TypeError("prediction features must be a sequence of names")
    allowed = PREDICTION_FEATURES[task_id]
    unknown = []
    for name in provided:
        if type(name) is not str or name == "":
            raise ValueError("each prediction feature must be a non-empty string")
        if name not in allowed:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"{task_id} cannot use {', '.join(unknown)} as prediction-time inputs. "
            "Clean-reference answers, logits, and residuals stay with the target."
        )
