"""Knowledge-consistency contextual-answer baseline.

Each evaluation row emits that row's cached canonical A/B distribution as the
prediction of the clean canonical label. The clean label is not an input.
There is no fitting stage, training frequency, optimizer, checkpoint, or
selected layer. Clean rows therefore reproduce the clean model's own
canonical probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from knowledge_recovery.provenance import content_digest
from knowledge_recovery.records import MethodSpec
from knowledge_recovery.tasks.base import CONSISTENCY_TASK_ID

METHOD_ID = "contextual_answer"
IMPLEMENTATION_VERSION = "contextual-canonical-ab-v1"
ANSWER_SCORE = "canonical_ab"


@dataclass(frozen=True)
class ContextualAnswer:
    """Fixed method. ``selected_layer`` stays null."""

    task_spec_digest: str
    task_id: str = CONSISTENCY_TASK_ID
    method_id: str = METHOD_ID
    implementation_version: str = IMPLEMENTATION_VERSION
    answer_score: str = ANSWER_SCORE
    selected_layer: int | None = None
    checkpoint_digest: str | None = None

    def __post_init__(self) -> None:
        if self.task_id != CONSISTENCY_TASK_ID or self.method_id != METHOD_ID:
            raise ValueError("contextual_answer is registered only for knowledge_consistency")
        if self.selected_layer is not None or self.checkpoint_digest is not None:
            raise ValueError("contextual_answer has no selected layer or checkpoint")
        if self.implementation_version != IMPLEMENTATION_VERSION or self.answer_score != ANSWER_SCORE:
            raise ValueError("contextual_answer score convention is canonical A/B")
        _digest(self.task_spec_digest)

    def predict(
        self,
        canonical_ab_probabilities: Sequence[tuple[float, float]],
    ) -> tuple[tuple[float, float], ...]:
        """Return the cached canonical distributions unchanged."""
        if isinstance(canonical_ab_probabilities, (str, bytes)) or not isinstance(
            canonical_ab_probabilities, Sequence
        ):
            raise TypeError("canonical_ab_probabilities must be a sequence of A/B pairs")
        return tuple(_pair(item) for item in canonical_ab_probabilities)


def contextual_answer_method(*, task_spec_digest: str) -> ContextualAnswer:
    """Build the method from its task contract. There is no training view."""
    return ContextualAnswer(task_spec_digest=task_spec_digest)


def fit_contextual_answer(*_args: object, **_kwargs: object) -> None:
    """This baseline is not fitted."""
    raise TypeError("contextual_answer has no fitting stage, optimizer, or selected layer")


def contextual_answer_spec() -> MethodSpec:
    """Method registration with no architecture, optimizer, or layer."""
    return MethodSpec(
        task_id=CONSISTENCY_TASK_ID,
        method_id=METHOD_ID,
        adapter_version=IMPLEMENTATION_VERSION,
        fit_inputs=(),
        predict_inputs=("canonical_ab_probabilities",),
        fitting_requirement="none",
        architecture=None,
        parameters={"selected_layer": None, "optimizer": None, "training_frequency": None},
        output_validation={"type": "ab_distribution", "class_order": ["A", "B"]},
    )


def contextual_state_digest(method: ContextualAnswer) -> str:
    """Hash the task contract and score convention. There are no fitted weights."""
    return content_digest({
        "kind": "contextual_answer_state",
        "schema": 1,
        "task_spec_digest": method.task_spec_digest,
        "answer_score": method.answer_score,
        "implementation_version": method.implementation_version,
    })


def _pair(value: object) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError("each cached distribution must be an A/B pair")
    parsed = []
    for item in value:
        if type(item) is bool or type(item) not in (int, float):
            raise TypeError("cached probabilities must be real numbers")
        number = float(item)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError("cached probabilities must be between 0 and 1")
        parsed.append(number)
    return parsed[0], parsed[1]


def _digest(value: object) -> None:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("task_spec_digest must be a full SHA-256 hex digest")
