"""Recovery-task contracts."""

from knowledge_recovery.tasks.agreement import (
    TARGET_VERSION as AWARENESS_TARGET_VERSION,
    AwarenessTask,
    awareness_label,
    awareness_target_index,
    awareness_task_spec,
)
from knowledge_recovery.tasks.base import (
    AWARENESS_CLASS_ORDER,
    AWARENESS_TASK_ID,
    CONSISTENCY_CLASS_ORDER,
    CONSISTENCY_TASK_ID,
    check_prediction_features,
)
from knowledge_recovery.tasks.clean_answer import (
    TARGET_VERSION as CONSISTENCY_TARGET_VERSION,
    ConsistencyTask,
    consistency_target,
    consistency_target_index,
    consistency_task_spec,
)

__all__ = [
    "AWARENESS_CLASS_ORDER",
    "AWARENESS_TARGET_VERSION",
    "AWARENESS_TASK_ID",
    "CONSISTENCY_CLASS_ORDER",
    "CONSISTENCY_TARGET_VERSION",
    "CONSISTENCY_TASK_ID",
    "AwarenessTask",
    "ConsistencyTask",
    "awareness_label",
    "awareness_target_index",
    "awareness_task_spec",
    "check_prediction_features",
    "consistency_target",
    "consistency_target_index",
    "consistency_task_spec",
]
