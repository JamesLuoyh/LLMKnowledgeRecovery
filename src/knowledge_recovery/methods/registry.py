"""Task-specific method registration.

Methods are stored by ``(task_id, method_id)``. A shared output width does
not make an awareness method usable for knowledge consistency.
"""

from __future__ import annotations

from typing import Any

from knowledge_recovery.errors import KnowledgeRecoveryError
from knowledge_recovery.methods.awareness import TrainingFrequency
from knowledge_recovery.methods.consistency import ContextualAnswer

REGISTRY: dict[tuple[str, str], type] = {
    ("behavioral_awareness", "training_frequency"): TrainingFrequency,
    ("knowledge_consistency", "contextual_answer"): ContextualAnswer,
}


class MethodTaskMismatch(KnowledgeRecoveryError):
    """A method was offered to a task it is not registered for."""

    def __init__(self, *, method_id: str, method_task_id: str, requested_task_id: str) -> None:
        self.method_id = method_id
        self.method_task_id = method_task_id
        self.requested_task_id = requested_task_id
        super().__init__(
            f"MethodTaskMismatch: {method_id} from {method_task_id} "
            f"cannot be used for {requested_task_id}.\n"
            "Rejecting this pair does not depend on tensor shapes."
        )


def registered_pairs() -> tuple[tuple[str, str], ...]:
    """Return the registered task and method pairs in lexicographic order."""
    return tuple(sorted(REGISTRY))


def require_task(method: Any, task_id: str) -> Any:
    """Return ``method`` only when that object is the registered implementation."""
    method_id = str(getattr(method, "method_id", ""))
    method_task = str(getattr(method, "task_id", ""))
    expected = REGISTRY.get((task_id, method_id))
    if expected is None or type(method) is not expected or method_task != task_id:
        raise MethodTaskMismatch(
            method_id=method_id or "unknown",
            method_task_id=method_task or "unknown",
            requested_task_id=task_id,
        )
    return method
