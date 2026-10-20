import hashlib
import math

import numpy
import pytest

from knowledge_recovery.config import LINEAR_V1, LinearV1
from knowledge_recovery.methods import require_task
from knowledge_recovery.methods.registry import MethodTaskMismatch
from knowledge_recovery.probes.linear import (
    apply_standardizer,
    fit_standardizer,
    initialize_logistic,
    initialization_seed,
)
from knowledge_recovery.provenance import canonical_json
from knowledge_recovery.tasks.base import AWARENESS_TASK_ID, CONSISTENCY_TASK_ID
from knowledge_recovery.training import (
    CpuTensorGenerator,
    derived_seed,
    mean_binary_cross_entropy,
    pcg64_generator,
)
from knowledge_recovery.training.fit import AdamState, LayerFit, adamw_step, fit_linear_probe
from knowledge_recovery.training.losses import binary_cross_entropy_gradients, finite_loss

TASK = AWARENESS_TASK_ID


def test_cpu_generator_matches_the_pytorch_mt19937_uniform_stream():
    generator = CpuTensorGenerator(5489)
    assert generator.random_uint32() == 3499211612
    bound = 1.0 / math.sqrt(4)
    draws = CpuTensorGenerator(7).uniform(5, -bound, bound)
    assert draws.dtype == numpy.float32
    assert numpy.all(draws >= -bound)
    assert numpy.all(draws < bound)


def test_derived_seeds_follow_the_sha256_stream_and_do_not_depend_on_fit_order():
    digest = hashlib.sha256(canonical_json({
        "layer": 2,
        "method_id": "linear",
        "seed": 42,
        "stream": "init",
        "task_id": TASK,
    }).encode()).digest()
    expected = int.from_bytes(digest[:8], "big") % 2**63
    assert derived_seed(seed=42, task_id=TASK, method_id="linear", layer=2, stream="init") == expected
    assert initialization_seed(training_seed=42, task_id=TASK, method_id="linear", layer=2) == expected
    assert derived_seed(seed=42, task_id=TASK, method_id="linear", layer=2, stream="shuffle") != expected
    left = _fit(layers=(1, 2), seed=42, epochs=1)
    right = _fit(layers=(2, 1), seed=42, epochs=1)
    assert left[0].checkpoints[0].weight == right[0].checkpoints[0].weight
    assert left[1].checkpoints[0].weight == right[1].checkpoints[0].weight
    assert left[0].init_seed != left[1].init_seed


def test_standardizer_uses_training_rows_population_deviation_and_float32_features():
    train = numpy.array([[1.0, 0.0], [2.0, 0.0], [3.0, 4.0]], dtype=numpy.float64)
    standardizer = fit_standardizer(train)
    assert standardizer.mean == pytest.approx((2.0, 4.0 / 3.0))
    assert standardizer.scale[0] ** 2 == pytest.approx(2.0 / 3.0)
    assert standardizer.scale[0] ** 2 != pytest.approx(1.0)
    zeroed = fit_standardizer(numpy.array([[0.0, 5.0], [2.0, 5.0]]))
    assert zeroed.zero_variance == (1,)
    assert zeroed.scale[1] == 1.0
    transformed = apply_standardizer(numpy.array([[0.1, 5.0]]), zeroed)
    assert transformed.dtype == numpy.float32
    with pytest.raises(ValueError, match="feature normalization"):
        fit_standardizer(train, domains=("id", "ood", "id"))


def test_adamw_decays_weights_only_and_matches_the_hand_step():
    state = AdamState.zeros(2)
    weight, bias = adamw_step(
        numpy.array([0.0, 0.0]),
        0.0,
        numpy.array([-0.5, 0.0]),
        -0.5,
        state,
        LINEAR_V1,
    )
    expected = 0.0005 / (0.5 + 1e-8)
    assert weight[0] == pytest.approx(expected)
    assert weight[1] == pytest.approx(0.0)
    assert bias == pytest.approx(expected)
    decayed, untouched = adamw_step(
        numpy.array([1.0]),
        1.0,
        numpy.array([0.0]),
        0.0,
        AdamState.zeros(1),
        LINEAR_V1,
    )
    assert decayed[0] == pytest.approx(1.0 - 1e-7)
    assert untouched == pytest.approx(1.0)
    assert finite_loss(0.0, 1) == pytest.approx(math.log(2.0))
    assert math.isfinite(finite_loss(-1000.0, 1))


def test_one_epoch_matches_mean_bce_and_keeps_the_short_batch():
    profile = _profile(max_epochs=1, patience=5, batch_size=256)
    train = numpy.zeros((1, 1, 2))
    train[0, 0, 0] = 1.0
    validation = train.copy()
    fitted = fit_linear_probe(
        train, [1], validation, [1],
        layers=(1,), task_id=TASK, training_seed=3,
        preprocessing="none", profile=profile,
    )[0]
    weight = numpy.asarray(initialize_logistic(
        seed=fitted.init_seed, width=2,
    )[0])
    bias = initialize_logistic(seed=fitted.init_seed, width=2)[1]
    features = numpy.array([[1.0, 0.0]])
    _, grad_w, grad_b = binary_cross_entropy_gradients(features, weight, bias, numpy.array([1.0]))
    expected_w, expected_b = adamw_step(weight, bias, grad_w, grad_b, AdamState.zeros(2), profile)
    assert fitted.checkpoints[0].weight == pytest.approx(tuple(expected_w))
    assert fitted.checkpoints[0].bias == pytest.approx(expected_b)
    assert fitted.optimizer_steps == 1
    wide = fit_linear_probe(
        numpy.zeros((300, 1, 1)), [0] * 300, numpy.zeros((2, 1, 1)), [0, 1],
        layers=(1,), task_id=TASK, training_seed=1,
        preprocessing="none", profile=profile,
    )[0]
    assert wide.optimizer_steps == 2
    recomputed = mean_binary_cross_entropy(
        apply_standardizer(numpy.zeros((2, 1)), wide.standardizer).astype(numpy.float64)
        @ numpy.asarray(wide.checkpoints[0].weight) + wide.checkpoints[0].bias,
        numpy.array([0.0, 1.0]),
    )
    assert wide.checkpoints[0].validation_cross_entropy == pytest.approx(recomputed)


def test_linear_v1_stops_after_twenty_nonimproving_epochs_and_rejects_ood():
    train = numpy.array([[[10.0, 0.0]], [[10.0, 0.0]]])
    validation = numpy.array([[[10.0, 0.0]]])
    fitted = fit_linear_probe(
        train, [1, 1], validation, [0],
        layers=(1,), task_id=TASK, training_seed=5,
        preprocessing="none",
    )[0]
    assert fitted.profile is LINEAR_V1
    assert fitted.profile.learning_rate == 1e-3
    assert fitted.profile.weight_decay == 1e-4
    assert fitted.profile.batch_size == 256
    assert fitted.profile.max_epochs == 200
    assert fitted.profile.early_stopping_patience == 20
    assert fitted.profile.betas == (0.9, 0.999)
    assert fitted.profile.epsilon == 1e-8
    assert fitted.stop_reason == "early_stopping"
    assert len(fitted.checkpoints) == 21
    assert all(item.completed for item in fitted.checkpoints)
    losses = [item.validation_cross_entropy for item in fitted.checkpoints]
    assert losses[0] == min(losses)
    assert fitted.positive_count == 2 and fitted.count == 2
    shuffle = pcg64_generator(fitted.shuffle_seed)
    assert len(shuffle.permutation(fitted.count)) == fitted.count
    with pytest.raises(ValueError, match="OOD rows never enter fitting"):
        fit_linear_probe(
            train, [1, 1], validation, [0],
            layers=(1,), task_id=TASK, training_seed=5,
            train_domains=("id", "ood"),
        )
    with pytest.raises(ValueError, match="the loss"):
        mean_binary_cross_entropy(numpy.array([0.0, 0.2]), numpy.array([1.0, 0.0]), domains=("ood", "id"))


def test_an_awareness_linear_probe_is_rejected_for_consistency_when_shapes_match():
    fitted = _fit(layers=(1,), seed=1, epochs=1)[0]
    probe = _probe(fitted, task_id=TASK)
    other = _probe(fitted, task_id=CONSISTENCY_TASK_ID)
    assert probe.predict(numpy.array([[0.0, 1.0]]))[0] == other.predict(numpy.array([[0.0, 1.0]]))[0]
    require_task(probe, TASK)
    with pytest.raises(MethodTaskMismatch, match="tensor shapes"):
        require_task(probe, CONSISTENCY_TASK_ID)


def _profile(*, max_epochs: int, patience: int, batch_size: int) -> LinearV1:
    return LinearV1(
        optimizer="AdamW",
        learning_rate=1e-3,
        weight_decay=1e-4,
        weight_decay_on="weights",
        betas=(0.9, 0.999),
        epsilon=1e-8,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        scheduler=None,
        gradient_clipping=None,
        mixed_precision=False,
    )


def _fit(*, layers: tuple[int, ...], seed: int, epochs: int) -> tuple[LayerFit, ...]:
    width = 2
    train = numpy.arange(8 * len(layers) * width, dtype=numpy.float64).reshape(8, len(layers), width)
    validation = train[:3]
    return fit_linear_probe(
        train,
        [index % 2 for index in range(8)],
        validation,
        [0, 1, 0],
        layers=layers,
        task_id=TASK,
        training_seed=seed,
        profile=_profile(max_epochs=epochs, patience=5, batch_size=4),
    )


def _probe(fitted: LayerFit, *, task_id: str):
    from knowledge_recovery.probes.linear import LinearProbe

    checkpoint = fitted.checkpoints[0]
    return LinearProbe(
        task_id=task_id,
        layer=fitted.layer,
        epoch=checkpoint.epoch,
        weight=checkpoint.weight,
        bias=checkpoint.bias,
        mean=fitted.standardizer.mean,
        scale=fitted.standardizer.scale,
        preprocessing=fitted.standardizer.mode,
        validation_cross_entropy=checkpoint.validation_cross_entropy,
        selected_layer=fitted.layer,
    )
