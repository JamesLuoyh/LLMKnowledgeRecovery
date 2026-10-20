"""Awareness training-frequency baseline.

The fitted state is one scalar, the uniform mean of filtered training
agreement labels. Residuals and contextual answer scores are not inputs.
There is no optimizer, checkpoint, or selected layer. An empty training view
is unavailable. A one-class view is a valid probability 0 or 1 and is labeled
degenerate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from knowledge_recovery.provenance import content_digest
from knowledge_recovery.records import MethodSpec
from knowledge_recovery.tasks.base import AWARENESS_TASK_ID

METHOD_ID = "training_frequency"
RULE = "uniform_row_mean"


@dataclass(frozen=True)
class TrainingFrequency:
    """Saved agreement frequency. ``selected_layer`` stays null."""

    training_view_id: str
    probability: float | None
    count: int
    positive_count: int
    available: bool
    unavailable_reason: str | None
    degenerate: bool
    degeneracy: str | None
    task_id: str = AWARENESS_TASK_ID
    method_id: str = METHOD_ID
    selected_layer: int | None = None
    checkpoint_digest: str | None = None

    def __post_init__(self) -> None:
        if self.task_id != AWARENESS_TASK_ID or self.method_id != METHOD_ID:
            raise ValueError("training_frequency is registered only for behavioral_awareness")
        if self.selected_layer is not None or self.checkpoint_digest is not None:
            raise ValueError("training_frequency has no selected layer or checkpoint")
        _view_id(self.training_view_id)
        if type(self.count) is not int or self.count < 0 or type(self.positive_count) is not int:
            raise ValueError("training counts must be non-negative integers")
        if self.positive_count < 0 or self.positive_count > self.count:
            raise ValueError("positive_count must lie between 0 and count")
        if self.available:
            if self.count == 0 or self.probability != self.positive_count / self.count:
                raise ValueError("agreement frequency must be the uniform training mean")
            if self.unavailable_reason is not None:
                raise ValueError("an available frequency has no unavailable reason")
        elif self.probability is not None or self.count != 0:
            raise ValueError("an unavailable frequency has no probability and an empty view")
        if self.degenerate:
            if self.degeneracy not in ("all_agree", "all_disagree"):
                raise ValueError("a degenerate frequency must be labeled all_agree or all_disagree")
        elif self.degeneracy is not None:
            raise ValueError("a non-degenerate frequency has no degeneracy label")

    def predict(self, row_count: int) -> tuple[float, ...]:
        """Repeat the training frequency. Row features are not accepted."""
        if not self.available or self.probability is None:
            raise ValueError(self.unavailable_reason or "training_frequency is unavailable")
        if type(row_count) is not int or row_count < 0:
            raise ValueError("row_count must be a non-negative integer")
        return (self.probability,) * row_count


def fit_training_frequency(
    agreement_targets: Sequence[int],
    *,
    training_view_id: str,
) -> TrainingFrequency:
    """Estimate the agreement frequency from filtered training labels only."""
    targets = _targets(agreement_targets)
    view = _view_id(training_view_id)
    count = len(targets)
    if count == 0:
        return TrainingFrequency(
            training_view_id=view,
            probability=None,
            count=0,
            positive_count=0,
            available=False,
            unavailable_reason="empty training view",
            degenerate=False,
            degeneracy=None,
        )
    positive = sum(targets)
    degenerate = positive in (0, count)
    degeneracy = None
    if positive == count:
        degeneracy = "all_agree"
    elif positive == 0:
        degeneracy = "all_disagree"
    return TrainingFrequency(
        training_view_id=view,
        probability=positive / count,
        count=count,
        positive_count=positive,
        available=True,
        unavailable_reason=None,
        degenerate=degenerate,
        degeneracy=degeneracy,
    )


def training_frequency_spec() -> MethodSpec:
    """Method registration. The architecture, optimizer, and layer are absent."""
    return MethodSpec(
        task_id=AWARENESS_TASK_ID,
        method_id=METHOD_ID,
        adapter_version="training-frequency-v1",
        fit_inputs=("agreement_target",),
        predict_inputs=("row_count",),
        fitting_requirement="filtered_training_frequency",
        architecture=None,
        parameters={
            "weights": RULE,
            "selected_layer": None,
            "optimizer": None,
            "checkpoint_search": None,
        },
        output_validation={"type": "agreement_probability", "class_order": ["disagree", "agree"]},
    )


def frequency_state_digest(state: TrainingFrequency) -> str:
    """Hash the scalar and its training view. Residuals and layers are absent."""
    return content_digest({
        "kind": "training_frequency_state",
        "schema": 1,
        "training_view_id": state.training_view_id,
        "rule": RULE,
        "count": state.count,
        "positive_count": state.positive_count,
        "probability": state.probability,
        "available": state.available,
        "degenerate": state.degenerate,
        "degeneracy": state.degeneracy,
    })


def _targets(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("agreement targets must be a sequence of 0/1 labels")
    parsed = []
    for value in values:
        if type(value) is not int or value not in (0, 1):
            raise ValueError("each agreement target must be 0 or 1")
        parsed.append(value)
    return tuple(parsed)


def _view_id(value: object) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("training_view_id must be a full SHA-256 hex digest")
    return value
