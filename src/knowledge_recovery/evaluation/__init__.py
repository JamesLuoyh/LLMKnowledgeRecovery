"""Point metrics, paired behavioral counts, and saved-prediction evaluation."""

from knowledge_recovery.evaluation.evaluate import (
    EvaluationRow,
    evaluate_saved,
    metrics_from_predictions,
    recompute_exported_slice,
)
from knowledge_recovery.evaluation.behavior import (
    BehaviorRow,
    BehaviorTable,
    behavior_document,
    behavior_tables,
)
from knowledge_recovery.evaluation.metrics import (
    BRIER_CONVENTION,
    DEFAULT_ECE_BINS,
    BinaryMetrics,
    CalibrationBin,
    awareness_metrics,
    consistency_metrics,
    metrics_document,
    roc_auc,
)

__all__ = [
    "BRIER_CONVENTION",
    "DEFAULT_ECE_BINS",
    "BinaryMetrics",
    "CalibrationBin",
    "EvaluationRow",
    "BehaviorRow",
    "BehaviorTable",
    "awareness_metrics",
    "behavior_document",
    "behavior_tables",
    "consistency_metrics",
    "evaluate_saved",
    "metrics_document",
    "metrics_from_predictions",
    "recompute_exported_slice",
    "roc_auc",
]
