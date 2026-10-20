import inspect
import json
import math

import pytest

from knowledge_recovery.config import LINEAR_V1
from knowledge_recovery.evaluation import evaluate as evaluate_module
from knowledge_recovery.evaluation.evaluate import (
    EvaluationRow,
    evaluate_saved,
    metrics_from_predictions,
    recompute_exported_slice,
)
from knowledge_recovery.methods import contextual_answer_method, fit_training_frequency
from knowledge_recovery.probes.linear import Standardizer
from knowledge_recovery.provenance import content_digest, identify
from knowledge_recovery.reports.export import export_report, iter_metric_rows
from knowledge_recovery.tasks import consistency_task_spec
from knowledge_recovery.tasks.base import AWARENESS_TASK_ID, CONSISTENCY_TASK_ID
from knowledge_recovery.training.checkpoints import EpochCheckpoint
from knowledge_recovery.training.fit import LayerFit
from knowledge_recovery.training.manifest import write_selection_manifest
from knowledge_recovery.training.select import select_probe

VIEW = "ab" * 32
LOGIT = math.log(9.0)


def test_saved_predictions_reproduce_metrics_without_fitting_or_model_loading(tmp_path):
    assert "fit_linear_probe" not in inspect.getsource(evaluate_module)
    assert "model_factory" not in inspect.signature(evaluate_saved).parameters
    assert "model" not in inspect.signature(metrics_from_predictions).parameters
    with pytest.raises(FileNotFoundError, match="before test evaluation"):
        evaluate_saved(
            _rows(),
            awareness_manifest=tmp_path / "missing",
            consistency_manifest=tmp_path / "missing",
            frequency=_frequency(),
            contextual=_contextual(),
            evaluation_view_id=_view(),
        )
    awareness = _manifest(tmp_path / "awareness", AWARENESS_TASK_ID)
    consistency = _manifest(tmp_path / "consistency", CONSISTENCY_TASK_ID)
    report = evaluate_saved(
        _rows(),
        awareness_manifest=awareness,
        consistency_manifest=consistency,
        frequency=_frequency(),
        contextual=_contextual(),
        evaluation_view_id=_view(),
    )
    linear = report["test_metrics"][AWARENESS_TASK_ID]["linear"]
    pooled = linear["domains"]["id"]["pooled"]
    variants = linear["domains"]["id"]["by_variant"]
    variant_ece = [variants[name]["ece"] for name in ("v1", "v2")]
    assert linear["selected_layer"] == 1
    assert report["test_metrics"][CONSISTENCY_TASK_ID]["linear"]["selected_layer"] == 1
    assert report["test_metrics"][AWARENESS_TASK_ID]["training_frequency"]["selected_layer"] is None
    assert report["test_metrics"][CONSISTENCY_TASK_ID]["contextual_answer"]["selected_layer"] is None
    probabilities = [
        item["positive_probability"]
        for item in report["predictions"]
        if item["method_id"] == "linear" and item["task_id"] == AWARENESS_TASK_ID and item["domain"] == "id"
    ]
    assert probabilities == pytest.approx([0.9, 0.9])
    assert pooled["brier"] == pytest.approx(0.41)
    assert pooled["mean_true_class_probability"] == pytest.approx(0.5)
    assert pooled["ece"] == pytest.approx(0.4)
    assert sum(variant_ece) / 2 == pytest.approx(0.5)
    assert pooled["ece"] != pytest.approx(sum(variant_ece) / 2)
    assert set(variants) == {"v1", "v2"}
    profile = report["validation_layer_profile"][AWARENESS_TASK_ID]
    assert [item["layer"] for item in profile] == [1, 2]
    assert "validation_cross_entropy" not in json.dumps(report["test_metrics"])
    recomputed = recompute_exported_slice(
        report, task_id=AWARENESS_TASK_ID, method_id="linear", domain="id",
    )
    assert recomputed.brier == pooled["brier"]
    assert recomputed.cross_entropy == pooled["cross_entropy"]
    assert recomputed.mean_true_class_probability == pooled["mean_true_class_probability"]
    assert recomputed.ece == pooled["ece"]
    frequency_ood = [
        item["positive_probability"]
        for item in report["predictions"]
        if item["method_id"] == "training_frequency" and item["domain"] == "ood"
    ]
    assert frequency_ood == [0.5]
    contextual_probability = next(
        item["positive_probability"]
        for item in report["predictions"]
        if item["method_id"] == "contextual_answer" and item["row_id"] == "ood"
    )
    assert contextual_probability == 0.7
    changed_bins = evaluate_saved(
        _rows(),
        awareness_manifest=awareness,
        consistency_manifest=consistency,
        frequency=_frequency(),
        contextual=_contextual(),
        evaluation_view_id=_view(),
        ece_bins=20,
    )
    assert changed_bins["evaluation_ids"][AWARENESS_TASK_ID]["linear"] != report["evaluation_ids"][AWARENESS_TASK_ID]["linear"]
    assert changed_bins["selection_digests"] == report["selection_digests"]
    assert changed_bins["test_metrics"][AWARENESS_TASK_ID]["linear"]["selected_layer"] == 1
    paths = export_report(report, tmp_path / "reports")
    exported = json.loads(paths["json"].read_text())
    assert "Infinity" not in paths["json"].read_text()
    assert paths["latex"].read_text().startswith("% generated by knowledge_recovery.reports.export")
    assert "v1" in paths["latex"].read_text() and "pooled" in paths["latex"].read_text()
    assert "behavioral\\_awareness" in paths["latex"].read_text()
    saved = next(
        item["brier"]
        for item in iter_metric_rows(exported)
        if item["task"] == AWARENESS_TASK_ID and item["method"] == "linear" and item["domain"] == "id" and item["population"] == "pooled"
    )
    assert saved == pooled["brier"]
    assert exported["confidence_intervals"] is None


def _rows() -> tuple[EvaluationRow, ...]:
    return (
        _row("id-v1", "id", "v1", "correct_at_A", 1, 1, (0.2, 0.8)),
        _row("id-v2", "id", "v2", "correct_at_B", 0, 0, (0.8, 0.2)),
        _row("ood", "ood", "NT-C", "correct_at_A", 1, 0, (0.3, 0.7)),
    )


def _row(row_id, domain, variant, position, awareness, consistency, canonical) -> EvaluationRow:
    return EvaluationRow(
        row_id=row_id,
        base_group_id=row_id,
        order_id=position,
        variant_id=variant,
        domain=domain,
        position=position,
        awareness_target=awareness,
        consistency_target=consistency,
        residuals={1: (0.0,), 2: (100.0,)},
        canonical_ab=canonical,
    )


def _manifest(directory, task_id: str):
    directory.mkdir()
    manifest = select_probe((
        _fit(task_id, layer=1, loss=0.1, weight=(0.0,), bias=LOGIT),
        _fit(task_id, layer=2, loss=0.5, weight=(1.0,), bias=0.0),
    ))
    assert manifest.selected_layer == 1
    write_selection_manifest(directory, manifest)
    return directory


def _fit(task_id: str, *, layer: int, loss: float, weight: tuple[float, ...], bias: float) -> LayerFit:
    return LayerFit(
        layer=layer,
        task_id=task_id,
        method_id="linear",
        init_seed=1,
        shuffle_seed=2,
        standardizer=Standardizer("none", (0.0,), (1.0,), ()),
        checkpoints=(EpochCheckpoint(
            layer=layer,
            epoch=1,
            validation_cross_entropy=loss,
            weight=weight,
            bias=bias,
            completed=True,
        ),),
        optimizer_steps=1,
        stop_reason="max_epochs",
        positive_count=1,
        count=2,
        profile=LINEAR_V1,
    )


def _frequency():
    return fit_training_frequency([1, 0], training_view_id=VIEW)


def _contextual():
    return contextual_answer_method(task_spec_digest=identify(consistency_task_spec()).digest)


def _view() -> str:
    return content_digest({"role": "evaluation"})
