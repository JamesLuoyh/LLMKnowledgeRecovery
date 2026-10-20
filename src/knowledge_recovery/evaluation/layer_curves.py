"""Score every saved linear-probe layer on the ID and OOD test rows.

Official evaluation keeps the selected layer. These curves keep that
checkpoint and also score the best validation checkpoint of every other
layer. Training frequency and the contextual-answer baseline do not depend
on the layer. The awareness variant baseline is the agreement rate of each
variant on the rows being scored, which is one minus that variant's
misreporting rate on those same rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy

from knowledge_recovery.evaluation.evaluate import EvaluationRow
from knowledge_recovery.evaluation.metrics import (
    DEFAULT_ECE_BINS,
    awareness_metrics,
    consistency_metrics,
    roc_auc,
)
from knowledge_recovery.methods.awareness import TrainingFrequency
from knowledge_recovery.methods.consistency import ContextualAnswer, contextual_answer_method
from knowledge_recovery.methods.registry import require_task
from knowledge_recovery.probes.linear import LinearProbe
from knowledge_recovery.tasks.base import AWARENESS_TASK_ID, CONSISTENCY_TASK_ID
from knowledge_recovery.training.manifest import SelectionManifest, load_selection_manifest

TRAINING_FREQUENCY_BASELINE = "training_frequency"
VARIANT_AGREEMENT_BASELINE = "variant_agreement_rate"
CONTEXTUAL_ANSWER_BASELINE = "contextual_answer"
DOMAINS = ("id", "ood")


@dataclass(frozen=True)
class CurveMetrics:
    """ECE, Brier, accuracy, and AUC for one slice. ``None`` is an undefined value."""

    count: int
    ece: float | None
    brier: float | None
    accuracy: float | None
    auc: float | None


@dataclass(frozen=True)
class LayerScores:
    """One layer's chosen checkpoint, pooled and split by variant."""

    layer: int
    epoch: int
    pooled: CurveMetrics
    by_variant: Mapping[str, CurveMetrics]


@dataclass(frozen=True)
class DomainCurves:
    """Probe series and layer-independent baselines for one domain."""

    domain: str
    layers: tuple[LayerScores, ...]
    baselines: Mapping[str, CurveMetrics]
    variant_baselines: Mapping[str, Mapping[str, CurveMetrics]]


@dataclass(frozen=True)
class TaskCurves:
    """ID and OOD curves for one task. ``selected_layer`` is the manifest choice."""

    task_id: str
    selected_layer: int
    domains: Mapping[str, DomainCurves]


def variant_agreement_probabilities(rows: Sequence[EvaluationRow]) -> tuple[float, ...]:
    """Constant P(agree) for each row: one minus that variant's misreporting rate.

    Misreporting is disagreement with the clean canonical label. The rate is
    the fraction of these rows, within the same variant, whose awareness
    target is disagree. A variant that is not in ``rows`` is not consulted.
    """
    parsed = _rows(rows)
    totals: dict[str, int] = {}
    positives: dict[str, int] = {}
    for row in parsed:
        totals[row.variant_id] = totals.get(row.variant_id, 0) + 1
        positives[row.variant_id] = positives.get(row.variant_id, 0) + row.awareness_target
    rates = {variant: positives[variant] / totals[variant] for variant in totals}
    return tuple(rates[row.variant_id] for row in parsed)


def score_layers(
    rows: Sequence[EvaluationRow],
    manifest: SelectionManifest,
    *,
    ece_bins: int = DEFAULT_ECE_BINS,
    frequency: TrainingFrequency | None = None,
    contextual: ContextualAnswer | None = None,
) -> TaskCurves:
    """Score each manifest layer. The selected layer is not the only one returned."""
    parsed = _rows(rows)
    _check_baseline(manifest, frequency, contextual)
    probes = _probes(manifest)
    domains = {
        domain: _score_domain(
            domain,
            tuple(row for row in parsed if row.domain == domain),
            probes,
            task_id=manifest.task_id,
            ece_bins=ece_bins,
            frequency=frequency,
            contextual=contextual,
        )
        for domain in DOMAINS
    }
    return TaskCurves(task_id=manifest.task_id, selected_layer=manifest.selected_layer, domains=domains)


def load_task_curves(*, run: str, config: str | None, task_id: str) -> TaskCurves:
    """Load cached test rows and the saved per-layer checkpoints for one task."""
    from knowledge_recovery.cli import StageRequest
    from knowledge_recovery.models.assets import activate_offline
    from knowledge_recovery.pipeline import (
        _config,
        _evaluation_rows,
        _load_current_views,
        _load_frequency,
        _resolve_run,
    )
    from knowledge_recovery.provenance import identify
    from knowledge_recovery.tasks import consistency_task_spec

    if task_id not in (AWARENESS_TASK_ID, CONSISTENCY_TASK_ID):
        raise ValueError(f"unknown task {task_id}")
    request = StageRequest(stage="evaluate", config=config, run=run)
    run_directory, storage, _attempt = _resolve_run(request)
    activate_offline(storage / "assets" / "huggingface")
    views = _load_current_views(storage)
    rows, _evaluation_view_id = _evaluation_rows(storage, views)
    ece_bins = _config(request).probe_run.ece_bins if config else DEFAULT_ECE_BINS
    manifest = load_selection_manifest(run_directory / task_id / "linear" / "selection_manifest.json")
    if manifest.task_id != task_id:
        raise ValueError(f"selection manifest task is {manifest.task_id}, not {task_id}")
    if task_id == AWARENESS_TASK_ID:
        return score_layers(rows, manifest, ece_bins=ece_bins, frequency=_load_frequency(run_directory))
    contextual = contextual_answer_method(task_spec_digest=identify(consistency_task_spec()).digest)
    return score_layers(rows, manifest, ece_bins=ece_bins, contextual=contextual)


def _score_domain(
    domain: str,
    rows: tuple[EvaluationRow, ...],
    probes: Sequence[LinearProbe],
    *,
    task_id: str,
    ece_bins: int,
    frequency: TrainingFrequency | None,
    contextual: ContextualAnswer | None,
) -> DomainCurves:
    targets = _targets(rows, task_id)
    variant_order, variant_indexes = _variants(rows)
    every = range(len(rows))
    layers = []
    for probe in probes:
        probabilities = _predict(rows, probe)
        layers.append(LayerScores(
            layer=probe.layer,
            epoch=probe.epoch,
            pooled=_at(task_id, probabilities, targets, every, ece_bins),
            by_variant={
                variant: _at(task_id, probabilities, targets, variant_indexes[variant], ece_bins)
                for variant in variant_order
            },
        ))
    baseline_probabilities = _baselines(rows, task_id=task_id, frequency=frequency, contextual=contextual)
    baselines = {
        name: _at(task_id, probabilities, targets, every, ece_bins)
        for name, probabilities in baseline_probabilities.items()
    }
    variant_baselines = {
        variant: {
            name: _at(task_id, probabilities, targets, variant_indexes[variant], ece_bins)
            for name, probabilities in baseline_probabilities.items()
        }
        for variant in variant_order
    }
    return DomainCurves(
        domain=domain,
        layers=tuple(sorted(layers, key=lambda item: item.layer)),
        baselines=baselines,
        variant_baselines=variant_baselines,
    )


def _baselines(
    rows: Sequence[EvaluationRow],
    *,
    task_id: str,
    frequency: TrainingFrequency | None,
    contextual: ContextualAnswer | None,
) -> dict[str, tuple[float, ...]]:
    if task_id == AWARENESS_TASK_ID:
        if frequency is None:
            raise ValueError("behavioral awareness curves require the training-frequency baseline")
        return {
            TRAINING_FREQUENCY_BASELINE: frequency.predict(len(rows)),
            VARIANT_AGREEMENT_BASELINE: variant_agreement_probabilities(rows),
        }
    if contextual is None:
        raise ValueError("knowledge consistency curves require the contextual-answer baseline")
    distributions = contextual.predict([row.canonical_ab for row in rows])
    return {CONTEXTUAL_ANSWER_BASELINE: tuple(pair[1] for pair in distributions)}


def _at(
    task_id: str,
    probabilities: Sequence[float],
    targets: Sequence[int],
    indexes: Sequence[int],
    ece_bins: int,
) -> CurveMetrics:
    chosen_probabilities = [float(probabilities[index]) for index in indexes]
    chosen_targets = [int(targets[index]) for index in indexes]
    if task_id == AWARENESS_TASK_ID:
        binary = awareness_metrics(chosen_probabilities, chosen_targets, ece_bins=ece_bins)
    else:
        binary = consistency_metrics(chosen_probabilities, chosen_targets, ece_bins=ece_bins)
    return CurveMetrics(
        count=binary.count,
        ece=binary.ece,
        brier=binary.brier,
        accuracy=binary.accuracy,
        auc=roc_auc(chosen_probabilities, chosen_targets),
    )


def _predict(rows: Sequence[EvaluationRow], probe: LinearProbe) -> tuple[float, ...]:
    if not rows:
        return ()
    if any(probe.layer not in row.residuals for row in rows):
        raise ValueError(f"every row must contain residual features for layer {probe.layer}")
    design = numpy.asarray([row.residuals[probe.layer] for row in rows], dtype=numpy.float64)
    return probe.predict(design)


def _probes(manifest: SelectionManifest) -> tuple[LinearProbe, ...]:
    return tuple(
        LinearProbe(
            task_id=manifest.task_id,
            layer=item.layer,
            epoch=item.epoch,
            weight=item.weight,
            bias=item.bias,
            mean=item.mean,
            scale=item.scale,
            preprocessing=item.preprocessing,
            validation_cross_entropy=item.validation_cross_entropy,
            selected_layer=item.layer,
        )
        for item in manifest.layers
    )


def _check_baseline(
    manifest: SelectionManifest,
    frequency: TrainingFrequency | None,
    contextual: ContextualAnswer | None,
) -> None:
    if manifest.task_id == AWARENESS_TASK_ID:
        if frequency is None:
            raise ValueError("behavioral awareness curves require the training-frequency baseline")
        require_task(frequency, AWARENESS_TASK_ID)
        return
    if manifest.task_id == CONSISTENCY_TASK_ID:
        if contextual is None:
            raise ValueError("knowledge consistency curves require the contextual-answer baseline")
        require_task(contextual, CONSISTENCY_TASK_ID)
        return
    raise ValueError(f"unknown task {manifest.task_id}")


def _targets(rows: Sequence[EvaluationRow], task_id: str) -> tuple[int, ...]:
    if task_id == AWARENESS_TASK_ID:
        return tuple(row.awareness_target for row in rows)
    return tuple(row.consistency_target for row in rows)


def _variants(rows: Sequence[EvaluationRow]) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    order: list[str] = []
    indexes: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        bucket = indexes.get(row.variant_id)
        if bucket is None:
            order.append(row.variant_id)
            bucket = []
            indexes[row.variant_id] = bucket
        bucket.append(index)
    return tuple(order), {variant: tuple(indexes[variant]) for variant in order}


def _rows(rows: Sequence[EvaluationRow]) -> tuple[EvaluationRow, ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of EvaluationRow records")
    if any(type(row) is not EvaluationRow for row in rows):
        raise TypeError("rows must be EvaluationRow records")
    return tuple(rows)
