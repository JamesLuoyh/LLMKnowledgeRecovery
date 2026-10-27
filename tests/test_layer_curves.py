import math

import pytest

from knowledge_recovery.config import LINEAR_V1
from knowledge_recovery.evaluation.evaluate import EvaluationRow
from knowledge_recovery.evaluation.layer_curves import (
    CONTEXTUAL_ANSWER_BASELINE,
    TRAINING_FREQUENCY_BASELINE,
    VARIANT_AGREEMENT_BASELINE,
    score_layers,
    variant_agreement_probabilities,
)
from knowledge_recovery.evaluation.metrics import awareness_metrics, consistency_metrics, roc_auc
from knowledge_recovery.methods import contextual_answer_method, fit_training_frequency
from knowledge_recovery.probes.linear import Standardizer
from knowledge_recovery.provenance import identify
from knowledge_recovery.tasks import consistency_task_spec
from knowledge_recovery.tasks.base import AWARENESS_TASK_ID, CONSISTENCY_TASK_ID
from knowledge_recovery.training.checkpoints import EpochCheckpoint
from knowledge_recovery.training.fit import LayerFit
from knowledge_recovery.training.select import select_probe

VIEW = "ab" * 32


def test_every_layer_is_scored_and_variant_rates_follow_the_plotted_rows():
    rows = _rows()
    curves = score_layers(rows, _manifest(AWARENESS_TASK_ID), frequency=_frequency())
    layers = curves.domains["id"].layers
    assert curves.selected_layer == 2
    assert [item.layer for item in layers] == [1, 2]
    assert layers[0].pooled.accuracy == pytest.approx(2 / 3)
    assert layers[1].pooled.accuracy == pytest.approx(1 / 3)

    id_rows = [row for row in rows if row.domain == "id"]
    probabilities = variant_agreement_probabilities(id_rows)
    targets = [row.awareness_target for row in id_rows]
    expected = awareness_metrics(list(probabilities), targets)
    baseline = curves.domains["id"].baselines[VARIANT_AGREEMENT_BASELINE]
    assert probabilities == (0.5, 0.5, 1.0)
    assert baseline.brier == expected.brier
    assert baseline.accuracy == expected.accuracy
    assert baseline.ece == expected.ece
    assert baseline.auc == roc_auc(probabilities, targets)
    clean = curves.domains["id"].variant_baselines["clean"][VARIANT_AGREEMENT_BASELINE]
    assert clean.ece == 0.0
    assert clean.auc == 0.5

    ood_rows = [row for row in rows if row.domain == "ood"]
    assert variant_agreement_probabilities(ood_rows) == (0.0, 0.0, 1.0)
    misreporting = curves.domains["ood"].variant_baselines["NT-C"][VARIANT_AGREEMENT_BASELINE]
    assert misreporting.accuracy == 1.0
    assert misreporting.ece == 0.0
    assert misreporting.auc is None

    frequency = curves.domains["id"].baselines[TRAINING_FREQUENCY_BASELINE]
    frequency_metrics = awareness_metrics([0.5, 0.5, 0.5], targets)
    assert frequency.brier == frequency_metrics.brier
    assert frequency.accuracy == frequency_metrics.accuracy
    with pytest.raises(ValueError, match="training-frequency"):
        score_layers(rows, _manifest(AWARENESS_TASK_ID))


def test_consistency_uses_the_contextual_answer_baseline_only():
    rows = _rows()
    curves = score_layers(rows, _manifest(CONSISTENCY_TASK_ID), contextual=_contextual())
    ood_rows = [row for row in rows if row.domain == "ood"]
    probabilities = [row.canonical_ab[1] for row in ood_rows]
    targets = [row.consistency_target for row in ood_rows]
    expected = consistency_metrics(probabilities, targets)
    baseline = curves.domains["ood"].baselines[CONTEXTUAL_ANSWER_BASELINE]
    assert set(curves.domains["ood"].baselines) == {CONTEXTUAL_ANSWER_BASELINE}
    assert VARIANT_AGREEMENT_BASELINE not in curves.domains["id"].baselines
    assert baseline.brier == expected.brier
    assert baseline.auc == roc_auc(probabilities, targets)
    assert [item.layer for item in curves.domains["ood"].layers] == [1, 2]
    with pytest.raises(ValueError, match="contextual-answer"):
        score_layers(rows, _manifest(CONSISTENCY_TASK_ID))


def test_pooled_figure_has_two_panels_and_variant_figure_has_grey_baselines():
    pytest.importorskip("matplotlib")
    from knowledge_recovery.evaluation import layer_plots

    curves = score_layers(_rows(), _manifest(AWARENESS_TASK_ID), frequency=_frequency())
    pooled = layer_plots.pooled_figure(curves, "accuracy")
    variant = layer_plots.variant_figure(curves, "brier", "id", "clean")
    try:
        assert len(pooled.axes) == 2
        assert pooled.axes[0].get_title() == "ID"
        assert pooled.axes[1].get_title() == "OOD"
        assert pooled.axes[0].get_xlabel() == "layer"
        dotted = [line for line in variant.axes[0].get_lines() if line.get_linestyle() == ":"]
        assert len(dotted) == 2
        assert all(_grey(line.get_color()) for line in dotted)
        probe = [line for line in variant.axes[0].get_lines() if line.get_label() == "linear probe"]
        assert len(probe) == 1
        assert probe[0].get_linestyle() == "-"
    finally:
        layer_plots._pyplot().close(pooled)
        layer_plots._pyplot().close(variant)


def test_without_ood_rows_there_are_no_ood_curves_and_one_pooled_panel():
    pytest.importorskip("matplotlib")
    from knowledge_recovery.evaluation import layer_plots

    rows = tuple(row for row in _rows() if row.domain == "id")
    curves = score_layers(rows, _manifest(AWARENESS_TASK_ID), frequency=_frequency())
    assert set(curves.domains) == {"id"}
    pooled = layer_plots.pooled_figure(curves, "accuracy")
    try:
        assert [axis.get_title() for axis in pooled.axes] == ["ID"]
    finally:
        layer_plots._pyplot().close(pooled)


def _grey(color: object) -> bool:
    if isinstance(color, str):
        return color in {"0.40", "0.4", "0.65", "0.55"}
    red, green, blue = color[:3]
    return red == pytest.approx(green) and green == pytest.approx(blue)


def _rows() -> tuple[EvaluationRow, ...]:
    return (
        _row("clean-a", "id", "clean", 1, 1, (0.2, 0.8)),
        _row("clean-b", "id", "clean", 0, 0, (0.7, 0.3)),
        _row("tc", "id", "T-C", 1, 1, (0.1, 0.9)),
        _row("ntc-a", "ood", "NT-C", 0, 0, (0.6, 0.4)),
        _row("ntc-b", "ood", "NT-C", 0, 1, (0.4, 0.6)),
        _row("ntic", "ood", "NT-IC", 1, 1, (0.25, 0.75)),
    )


def _row(row_id, domain, variant, awareness, consistency, canonical) -> EvaluationRow:
    return EvaluationRow(
        row_id=row_id,
        base_group_id=row_id,
        order_id="correct_at_A",
        variant_id=variant,
        domain=domain,
        position="correct_at_A",
        awareness_target=awareness,
        consistency_target=consistency,
        residuals={1: (0.0,), 2: (0.0,)},
        canonical_ab=canonical,
    )


def _manifest(task_id: str):
    return select_probe((
        _fit(task_id, layer=1, loss=0.4, bias=math.log(9.0)),
        _fit(task_id, layer=2, loss=0.05, bias=-math.log(4.0)),
    ))


def _fit(task_id: str, *, layer: int, loss: float, bias: float) -> LayerFit:
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
            weight=(0.0,),
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
