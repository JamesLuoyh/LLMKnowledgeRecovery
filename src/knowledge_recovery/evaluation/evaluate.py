"""Score a saved checkpoint and the two baselines.

Predictions are the evaluation record. Cross-entropy, Brier, K, and ECE are
recomputed from those predictions. This module does not fit a probe and does
not construct a language model. Validation losses stay in the selection
manifest's layer profile, beside the selected-layer test metrics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy

from knowledge_recovery.evaluation.metrics import (
    BRIER_CONVENTION,
    DEFAULT_ECE_BINS,
    awareness_metrics,
    consistency_metrics,
    metrics_document,
)
from knowledge_recovery.methods.awareness import TrainingFrequency, frequency_state_digest
from knowledge_recovery.methods.consistency import ContextualAnswer, contextual_state_digest
from knowledge_recovery.methods.registry import require_task
from knowledge_recovery.probes.linear import LINEAR_METHOD_ID, LinearProbe, Standardizer, apply_standardizer
from knowledge_recovery.provenance import content_digest, evaluation_id
from knowledge_recovery.tasks.base import (
    AWARENESS_CLASS_ORDER,
    AWARENESS_TASK_ID,
    CONSISTENCY_CLASS_ORDER,
    CONSISTENCY_TASK_ID,
)
from knowledge_recovery.training.manifest import SelectionManifest, load_selection_manifest

PREDICTION_VERSION = "saved-predictions-v1"
METRICS_VERSION = "metrics-v1"
_DOMAINS = ("id", "ood")
_AWARENESS_BASELINE = "training_frequency"
_CONSISTENCY_BASELINE = "contextual_answer"


@dataclass(frozen=True)
class EvaluationRow:
    """One cached evaluation row. Residuals are keyed by layer index."""

    row_id: str
    base_group_id: str
    order_id: str
    variant_id: str
    domain: str
    position: str
    awareness_target: int
    consistency_target: int
    residuals: Mapping[int, tuple[float, ...]]
    canonical_ab: tuple[float, float]
    instance_id: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.domain not in _DOMAINS:
            raise ValueError("domain must be id or ood")
        if self.position not in ("correct_at_A", "correct_at_B"):
            raise ValueError("position must be correct_at_A or correct_at_B")
        for label in ("awareness_target", "consistency_target"):
            if getattr(self, label) not in (0, 1):
                raise ValueError(f"{label} must be 0 or 1")
        if self.weight != 1.0:
            raise ValueError("the initial evaluation uses uniform row weight 1")
        if len(self.canonical_ab) != 2:
            raise ValueError("canonical_ab must be an A/B pair")


def evaluate_saved(
    rows: Sequence[EvaluationRow],
    *,
    awareness_manifest: Path | str,
    consistency_manifest: Path | str,
    frequency: TrainingFrequency,
    contextual: ContextualAnswer,
    evaluation_view_id: str,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> dict[str, Any]:
    """Score ID and OOD rows from manifests that were written before this call."""
    awareness = load_selection_manifest(awareness_manifest)
    consistency = load_selection_manifest(consistency_manifest)
    if awareness.task_id != AWARENESS_TASK_ID or consistency.task_id != CONSISTENCY_TASK_ID:
        raise ValueError("each manifest must belong to its own task")
    parsed = _rows(rows)
    awareness_probe = require_task(_probe(awareness), AWARENESS_TASK_ID)
    consistency_probe = require_task(_probe(consistency), CONSISTENCY_TASK_ID)
    require_task(frequency, AWARENESS_TASK_ID)
    require_task(contextual, CONSISTENCY_TASK_ID)
    predictions = [
        *_linear_predictions(parsed, awareness_probe),
        *_frequency_predictions(parsed, frequency),
        *_linear_predictions(parsed, consistency_probe),
        *_contextual_predictions(parsed, contextual),
    ]
    return {
        "ece_bins": ece_bins,
        "confidence_intervals": None,
        "evaluation_view_id": evaluation_view_id,
        "selection_digests": {
            AWARENESS_TASK_ID: awareness.digest,
            CONSISTENCY_TASK_ID: consistency.digest,
        },
        "evaluation_ids": {
            AWARENESS_TASK_ID: {
                LINEAR_METHOD_ID: _linear_evaluation_id(awareness_probe, awareness, evaluation_view_id, ece_bins),
                _AWARENESS_BASELINE: _baseline_evaluation_id(
                    frequency_state_digest(frequency), evaluation_view_id, ece_bins, positive_class="agree", tie="agree",
                ),
            },
            CONSISTENCY_TASK_ID: {
                LINEAR_METHOD_ID: _linear_evaluation_id(consistency_probe, consistency, evaluation_view_id, ece_bins),
                _CONSISTENCY_BASELINE: _baseline_evaluation_id(
                    contextual_state_digest(contextual), evaluation_view_id, ece_bins, positive_class="B", tie="A",
                ),
            },
        },
        "validation_layer_profile": {
            AWARENESS_TASK_ID: _profile(awareness),
            CONSISTENCY_TASK_ID: _profile(consistency),
        },
        "test_metrics": {
            AWARENESS_TASK_ID: {
                LINEAR_METHOD_ID: _method_metrics(predictions, AWARENESS_TASK_ID, LINEAR_METHOD_ID, ece_bins, awareness.selected_layer),
                _AWARENESS_BASELINE: _method_metrics(predictions, AWARENESS_TASK_ID, _AWARENESS_BASELINE, ece_bins, None),
            },
            CONSISTENCY_TASK_ID: {
                LINEAR_METHOD_ID: _method_metrics(predictions, CONSISTENCY_TASK_ID, LINEAR_METHOD_ID, ece_bins, consistency.selected_layer),
                _CONSISTENCY_BASELINE: _method_metrics(predictions, CONSISTENCY_TASK_ID, _CONSISTENCY_BASELINE, ece_bins, None),
            },
        },
        "predictions": predictions,
    }


def metrics_from_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    task_id: str,
    ece_bins: int = DEFAULT_ECE_BINS,
):
    """Recompute point metrics from saved probabilities. No model and no fitting."""
    probabilities = [float(item["positive_probability"]) for item in predictions]
    targets = [int(item["target"]) for item in predictions]
    if task_id == AWARENESS_TASK_ID:
        return awareness_metrics(probabilities, targets, ece_bins=ece_bins)
    if task_id == CONSISTENCY_TASK_ID:
        return consistency_metrics(probabilities, targets, ece_bins=ece_bins)
    raise ValueError(f"unknown task {task_id}")


def recompute_exported_slice(
    report: Mapping[str, Any],
    *,
    task_id: str,
    method_id: str,
    domain: str,
):
    """Recompute one pooled slice from the saved prediction list."""
    chosen = [
        item for item in report["predictions"]
        if item["task_id"] == task_id and item["method_id"] == method_id and item["domain"] == domain
    ]
    return metrics_from_predictions(chosen, task_id=task_id, ece_bins=int(report["ece_bins"]))


def _probe(manifest: SelectionManifest) -> LinearProbe:
    chosen = next(item for item in manifest.layers if item.layer == manifest.selected_layer)
    return LinearProbe(
        task_id=manifest.task_id,
        layer=chosen.layer,
        epoch=chosen.epoch,
        weight=chosen.weight,
        bias=chosen.bias,
        mean=chosen.mean,
        scale=chosen.scale,
        preprocessing=chosen.preprocessing,
        validation_cross_entropy=chosen.validation_cross_entropy,
        selected_layer=chosen.layer,
    )


def _linear_predictions(rows: Sequence[EvaluationRow], probe: LinearProbe) -> list[dict[str, Any]]:
    if not rows:
        return []
    if any(probe.layer not in row.residuals for row in rows):
        raise ValueError(f"every row must contain residual features for selected layer {probe.layer}")
    design = numpy.asarray([row.residuals[probe.layer] for row in rows], dtype=numpy.float64)
    probabilities = probe.predict(design)
    transformed = apply_standardizer(design, Standardizer(probe.preprocessing, probe.mean, probe.scale, ()))
    logits = numpy.asarray(transformed, dtype=numpy.float64) @ numpy.asarray(probe.weight) + probe.bias
    task_id = probe.task_id
    return [
        _prediction(
            row,
            task_id=task_id,
            method_id=LINEAR_METHOD_ID,
            target=row.awareness_target if task_id == AWARENESS_TASK_ID else row.consistency_target,
            probability=probabilities[index],
            logit=float(logits[index]),
            selected_layer=probe.selected_layer,
            class_order=AWARENESS_CLASS_ORDER if task_id == AWARENESS_TASK_ID else CONSISTENCY_CLASS_ORDER,
            method_state_digest=_linear_state_digest(probe),
        )
        for index, row in enumerate(rows)
    ]


def _frequency_predictions(rows: Sequence[EvaluationRow], frequency: TrainingFrequency) -> list[dict[str, Any]]:
    probabilities = frequency.predict(len(rows))
    return [
        _prediction(
            row,
            task_id=AWARENESS_TASK_ID,
            method_id=_AWARENESS_BASELINE,
            target=row.awareness_target,
            probability=probabilities[index],
            logit=_finite_logit(probabilities[index]),
            selected_layer=None,
            class_order=AWARENESS_CLASS_ORDER,
            method_state_digest=frequency_state_digest(frequency),
        )
        for index, row in enumerate(rows)
    ]


def _contextual_predictions(rows: Sequence[EvaluationRow], contextual: ContextualAnswer) -> list[dict[str, Any]]:
    distributions = contextual.predict([row.canonical_ab for row in rows])
    return [
        _prediction(
            row,
            task_id=CONSISTENCY_TASK_ID,
            method_id=_CONSISTENCY_BASELINE,
            target=row.consistency_target,
            probability=distributions[index][1],
            logit=_finite_logit(distributions[index][1]),
            selected_layer=None,
            class_order=CONSISTENCY_CLASS_ORDER,
            method_state_digest=contextual_state_digest(contextual),
        )
        for index, row in enumerate(rows)
    ]


def _prediction(
    row: EvaluationRow,
    *,
    task_id: str,
    method_id: str,
    target: int,
    probability: float,
    logit: float | None,
    selected_layer: int | None,
    class_order: tuple[str, ...],
    method_state_digest: str,
) -> dict[str, Any]:
    return {
        "row_id": row.row_id,
        "base_group_id": row.base_group_id,
        "order_id": row.order_id,
        "variant_id": row.variant_id,
        "instance_id": row.instance_id,
        "domain": row.domain,
        "position": row.position,
        "task_id": task_id,
        "method_id": method_id,
        "target": target,
        "positive_probability": probability,
        "logit": logit,
        "weight": row.weight,
        "selected_layer": selected_layer,
        "class_order": list(class_order),
        "method_state_digest": method_state_digest,
        "eligible": True,
    }


def _method_metrics(
    predictions: Sequence[Mapping[str, Any]],
    task_id: str,
    method_id: str,
    ece_bins: int,
    selected_layer: int | None,
) -> dict[str, Any]:
    chosen = [item for item in predictions if item["task_id"] == task_id and item["method_id"] == method_id]
    return {
        "selected_layer": selected_layer,
        "domains": {domain: _domain_metrics(chosen, task_id, domain, ece_bins) for domain in _DOMAINS},
    }


def _domain_metrics(predictions: Sequence[Mapping[str, Any]], task_id: str, domain: str, ece_bins: int) -> dict[str, Any]:
    subset = [item for item in predictions if item["domain"] == domain]
    return {
        "pooled": metrics_document(metrics_from_predictions(subset, task_id=task_id, ece_bins=ece_bins)),
        "by_variant": {
            variant: metrics_document(metrics_from_predictions(
                [item for item in subset if item["variant_id"] == variant],
                task_id=task_id,
                ece_bins=ece_bins,
            ))
            for variant in _ordered(subset, "variant_id")
        },
        "by_position": {
            position: metrics_document(metrics_from_predictions(
                [item for item in subset if item["position"] == position],
                task_id=task_id,
                ece_bins=ece_bins,
            ))
            for position in _ordered(subset, "position")
        },
    }


def _ordered(predictions: Sequence[Mapping[str, Any]], field: str) -> tuple[str, ...]:
    seen: list[str] = []
    for item in predictions:
        if item[field] not in seen:
            seen.append(item[field])
    return tuple(seen)


def _profile(manifest: SelectionManifest) -> list[dict[str, Any]]:
    return [
        {
            "layer": item.layer,
            "epoch": item.epoch,
            "validation_cross_entropy": item.validation_cross_entropy,
        }
        for item in manifest.layers
    ]


def _linear_state_digest(probe: LinearProbe) -> str:
    return content_digest({
        "kind": "linear_probe_state",
        "schema": 1,
        "task_id": probe.task_id,
        "layer": probe.layer,
        "epoch": probe.epoch,
        "weight": list(probe.weight),
        "bias": probe.bias,
        "mean": list(probe.mean),
        "scale": list(probe.scale),
        "preprocessing": probe.preprocessing,
    })


def _linear_evaluation_id(probe: LinearProbe, manifest: SelectionManifest, view_id: str, ece_bins: int) -> str:
    transform = content_digest({
        "preprocessing": probe.preprocessing,
        "mean": list(probe.mean),
        "scale": list(probe.scale),
    })
    return _evaluation_digest(
        method_state_digest=_linear_state_digest(probe),
        evaluation_view_id=view_id,
        ece_bins=ece_bins,
        positive_class="agree" if probe.task_id == AWARENESS_TASK_ID else "B",
        tie="agree" if probe.task_id == AWARENESS_TASK_ID else "A",
        transform_digest=transform,
        selection_manifest_digest=manifest.digest,
    )


def _baseline_evaluation_id(
    method_state_digest: str,
    view_id: str,
    ece_bins: int,
    *,
    positive_class: str,
    tie: str,
) -> str:
    return _evaluation_digest(
        method_state_digest=method_state_digest,
        evaluation_view_id=view_id,
        ece_bins=ece_bins,
        positive_class=positive_class,
        tie=tie,
        transform_digest=None,
        selection_manifest_digest=None,
    )


def _evaluation_digest(
    *,
    method_state_digest: str,
    evaluation_view_id: str,
    ece_bins: int,
    positive_class: str,
    tie: str,
    transform_digest: str | None,
    selection_manifest_digest: str | None,
) -> str:
    return evaluation_id(
        method_state_digest=method_state_digest,
        evaluation_view_id=evaluation_view_id,
        prediction_convention={"positive_class": positive_class, "tie_prediction": tie},
        metric_definitions={
            "cross_entropy": "logits_or_probabilities",
            "brier": BRIER_CONVENTION,
            "k": "mean_true_class_probability",
            "ece": "top_label_confidence",
        },
        aggregation_rules={"weighting": "uniform_row", "pooled_ece": "bin_pooled_rows"},
        calibration_bins={"count": ece_bins, "interval": "[b/n,(b+1)/n)", "final_edge": "closed"},
        prediction_version=PREDICTION_VERSION,
        metrics_version=METRICS_VERSION,
        transform_digest=transform_digest,
        selection_manifest_digest=selection_manifest_digest,
    ).digest


def _finite_logit(probability: float) -> float | None:
    if probability <= 0.0 or probability >= 1.0:
        return None
    return float(math.log(probability / (1.0 - probability)))


def _rows(rows: Sequence[EvaluationRow]) -> tuple[EvaluationRow, ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of EvaluationRow records")
    if any(type(row) is not EvaluationRow for row in rows):
        raise TypeError("rows must be EvaluationRow records")
    return tuple(rows)
