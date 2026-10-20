"""Task-specific baselines."""

from knowledge_recovery.methods.awareness import (
    TrainingFrequency,
    fit_training_frequency,
    frequency_state_digest,
    training_frequency_spec,
)
from knowledge_recovery.methods.consistency import (
    ContextualAnswer,
    contextual_answer_method,
    contextual_answer_spec,
    contextual_state_digest,
    fit_contextual_answer,
)
from knowledge_recovery.methods.registry import MethodTaskMismatch, registered_pairs, require_task

__all__ = [
    "ContextualAnswer",
    "MethodTaskMismatch",
    "TrainingFrequency",
    "contextual_answer_method",
    "contextual_answer_spec",
    "contextual_state_digest",
    "fit_contextual_answer",
    "fit_training_frequency",
    "frequency_state_digest",
    "registered_pairs",
    "require_task",
    "training_frequency_spec",
]
