"""Choose a completed checkpoint and then the shallower tied layer.

Only finished epochs compete. An exact validation-loss tie keeps the
earliest checkpoint, and an exact layer tie keeps the shallower layer.
The manifest is the selection result. Test labels are not an argument, and
the initial protocol does not refit on train plus validation.
"""

from __future__ import annotations

from collections.abc import Sequence

from knowledge_recovery.training.checkpoints import EpochCheckpoint
from knowledge_recovery.training.fit import LayerFit
from knowledge_recovery.training.manifest import (
    LayerChoice,
    SelectionManifest,
    manifest_record,
    selection_digest,
)

def select_checkpoint(checkpoints: Sequence[EpochCheckpoint]) -> EpochCheckpoint:
    """Return the earliest completed checkpoint among exact loss ties."""
    if isinstance(checkpoints, (str, bytes)) or not isinstance(checkpoints, Sequence) or not checkpoints:
        raise ValueError("at least one completed-epoch checkpoint is required")
    if any(not item.completed for item in checkpoints):
        raise ValueError("only completed-epoch checkpoints compete")
    layers = {item.layer for item in checkpoints}
    if len(layers) != 1:
        raise ValueError("a checkpoint tie break applies to one layer")
    return min(checkpoints, key=lambda item: (item.validation_cross_entropy, item.epoch))


def select_probe(layer_fits: Sequence[LayerFit]) -> SelectionManifest:
    """Select each layer's checkpoint, then the shallower layer on an exact tie."""
    if isinstance(layer_fits, (str, bytes)) or not isinstance(layer_fits, Sequence) or not layer_fits:
        raise ValueError("at least one fitted layer is required")
    task_ids = {item.task_id for item in layer_fits}
    method_ids = {item.method_id for item in layer_fits}
    if len(task_ids) != 1 or len(method_ids) != 1:
        raise ValueError("a selection manifest covers one task and one method")
    if len({item.layer for item in layer_fits}) != len(layer_fits):
        raise ValueError("each layer is selected once")
    choices = tuple(sorted((_choice(item) for item in layer_fits), key=lambda item: item.layer))
    winner = min(choices, key=lambda item: (item.validation_cross_entropy, item.layer))
    record = {
        "kind": "selection_manifest",
        "schema": 1,
        "task_id": layer_fits[0].task_id,
        "method_id": layer_fits[0].method_id,
        "metric": "validation_cross_entropy",
        "tie_break": "earliest_checkpoint_then_shallower_layer",
        "refit_on_train_validation": False,
        "selected_layer": winner.layer,
        "selected_epoch": winner.epoch,
        "validation_cross_entropy": winner.validation_cross_entropy,
        "layers": [
            {
                "layer": item.layer,
                "epoch": item.epoch,
                "validation_cross_entropy": item.validation_cross_entropy,
                "weight": list(item.weight),
                "bias": item.bias,
                "mean": list(item.mean),
                "scale": list(item.scale),
                "preprocessing": item.preprocessing,
            }
            for item in choices
        ],
    }
    manifest = SelectionManifest(
        task_id=record["task_id"],
        method_id=record["method_id"],
        selected_layer=winner.layer,
        selected_epoch=winner.epoch,
        validation_cross_entropy=winner.validation_cross_entropy,
        layers=choices,
        digest=selection_digest(record),
    )
    if manifest_record(manifest) != record:
        raise ValueError("selection record drifted from the chosen checkpoints")
    return manifest


def _choice(fitted: LayerFit) -> LayerChoice:
    checkpoint = select_checkpoint(fitted.checkpoints)
    return LayerChoice(
        layer=fitted.layer,
        epoch=checkpoint.epoch,
        validation_cross_entropy=checkpoint.validation_cross_entropy,
        weight=checkpoint.weight,
        bias=checkpoint.bias,
        mean=fitted.standardizer.mean,
        scale=fitted.standardizer.scale,
        preprocessing=fitted.standardizer.mode,
    )
