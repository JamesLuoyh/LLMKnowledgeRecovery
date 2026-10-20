import inspect

import pytest

from knowledge_recovery.config import LINEAR_V1
from knowledge_recovery.probes.linear import Standardizer
from knowledge_recovery.tasks.base import AWARENESS_TASK_ID
from knowledge_recovery.training.checkpoints import EpochCheckpoint
from knowledge_recovery.training.fit import LayerFit
from knowledge_recovery.training.manifest import load_selection_manifest, write_selection_manifest
from knowledge_recovery.training.select import select_checkpoint, select_probe


def test_exact_ties_keep_the_earliest_checkpoint_and_the_shallower_layer(tmp_path):
    assert "test_labels" not in inspect.signature(select_probe).parameters
    assert "fit_linear_probe" not in inspect.getsource(select_probe)
    early = _checkpoint(layer=2, epoch=2, loss=0.2)
    later = _checkpoint(layer=2, epoch=5, loss=0.2)
    better = _checkpoint(layer=2, epoch=4, loss=0.1)
    assert select_checkpoint((later, early)).epoch == 2
    assert select_checkpoint((early, better)).epoch == 4
    with pytest.raises(ValueError, match="completed-epoch"):
        select_checkpoint((early, _checkpoint(layer=2, epoch=1, loss=0.0, completed=False)))

    tied = select_probe((
        _fit(1, (_checkpoint(layer=1, epoch=3, loss=0.2),)),
        _fit(2, (early, later)),
    ))
    assert tied.selected_layer == 1
    assert tied.selected_epoch == 3
    assert tied.refit_on_train_validation is False
    assert tied.metric == "validation_cross_entropy"
    deeper = select_probe((
        _fit(1, (_checkpoint(layer=1, epoch=1, loss=0.4),)),
        _fit(2, (better,)),
    ))
    assert deeper.selected_layer == 2 and deeper.selected_epoch == 4
    path = write_selection_manifest(tmp_path, tied)
    loaded = load_selection_manifest(path)
    assert loaded == tied
    assert "validation_cross_entropy" in loaded.layers[0].__dict__
    assert "test_loss" not in loaded.layers[0].__dict__


def _checkpoint(*, layer: int, epoch: int, loss: float, completed: bool = True) -> EpochCheckpoint:
    return EpochCheckpoint(
        layer=layer,
        epoch=epoch,
        validation_cross_entropy=loss,
        weight=(0.25,),
        bias=0.0,
        completed=completed,
    )


def _fit(layer: int, checkpoints: tuple[EpochCheckpoint, ...]) -> LayerFit:
    return LayerFit(
        layer=layer,
        task_id=AWARENESS_TASK_ID,
        method_id="linear",
        init_seed=1,
        shuffle_seed=2,
        standardizer=Standardizer("none", (0.0,), (1.0,), ()),
        checkpoints=checkpoints,
        optimizer_steps=1,
        stop_reason="max_epochs",
        positive_count=1,
        count=2,
        profile=LINEAR_V1,
    )
