"""Fit one independent logistic model per layer.

The resolved profile is ``linear_v1`` unless the caller supplies another
complete profile. AdamW uses learning rate ``1e-3``, weight decay ``1e-4``
on weights only, betas ``(0.9, 0.999)``, and epsilon ``1e-8``. Each epoch
shuffles training rows once, keeps the final short batch, and steps on mean
BCE. Validation cross-entropy is the mean over every validation row after
the epoch finishes. Training stops after 20 completed epochs without a
strict decrease, or at 200 epochs. OOD rows never enter this loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy

from knowledge_recovery.config import LINEAR_V1, LinearV1
from knowledge_recovery.probes.linear import (
    LINEAR_METHOD_ID,
    Standardizer,
    apply_standardizer,
    fit_standardizer,
    initialize_logistic,
    initialization_seed,
    shuffle_seed,
)
from knowledge_recovery.tasks.base import AWARENESS_TASK_ID, CONSISTENCY_TASK_ID
from knowledge_recovery.training.checkpoints import EpochCheckpoint
from knowledge_recovery.training.losses import (
    binary_cross_entropy_gradients,
    mean_binary_cross_entropy,
    reject_ood,
)
from knowledge_recovery.training.seeds import pcg64_generator

TRAINING_IMPLEMENTATION_VERSION = "linear-fit-v1"
_TASKS = (AWARENESS_TASK_ID, CONSISTENCY_TASK_ID)


@dataclass
class AdamState:
    """Moment vectors for one linear layer. The step counter starts at zero."""

    exp_avg_weight: numpy.ndarray
    exp_avg_sq_weight: numpy.ndarray
    exp_avg_bias: float
    exp_avg_sq_bias: float
    step: int

    @classmethod
    def zeros(cls, width: int) -> AdamState:
        return cls(numpy.zeros(width, dtype=numpy.float32), numpy.zeros(width, dtype=numpy.float32), 0.0, 0.0, 0)


@dataclass(frozen=True)
class LayerFit:
    """Completed-epoch history for one layer. Later epochs do not rewrite earlier ones."""

    layer: int
    task_id: str
    method_id: str
    init_seed: int
    shuffle_seed: int
    standardizer: Standardizer
    checkpoints: tuple[EpochCheckpoint, ...]
    optimizer_steps: int
    stop_reason: str
    positive_count: int
    count: int
    profile: LinearV1


def adamw_step(
    weight: numpy.ndarray,
    bias: float,
    grad_weight: numpy.ndarray,
    grad_bias: float,
    state: AdamState,
    profile: LinearV1,
) -> tuple[numpy.ndarray, float]:
    """One decoupled AdamW update. Bias is excluded from weight decay."""
    _check_profile(profile)
    beta1, beta2 = profile.betas
    state.step += 1
    step = state.step
    grad_w = numpy.asarray(grad_weight, dtype=numpy.float32)
    state.exp_avg_weight = beta1 * state.exp_avg_weight + (1.0 - beta1) * grad_w
    state.exp_avg_sq_weight = beta2 * state.exp_avg_sq_weight + (1.0 - beta2) * grad_w * grad_w
    state.exp_avg_bias = beta1 * state.exp_avg_bias + (1.0 - beta1) * grad_bias
    state.exp_avg_sq_bias = beta2 * state.exp_avg_sq_bias + (1.0 - beta2) * grad_bias * grad_bias
    correction1 = 1.0 - beta1**step
    correction2 = 1.0 - beta2**step
    step_size = profile.learning_rate / correction1
    decayed = numpy.asarray(weight, dtype=numpy.float32) * (1.0 - profile.learning_rate * profile.weight_decay)
    denom_weight = numpy.sqrt(state.exp_avg_sq_weight / correction2) + profile.epsilon
    denom_bias = math_sqrt(state.exp_avg_sq_bias / correction2) + profile.epsilon
    updated = decayed - step_size * state.exp_avg_weight / denom_weight
    updated_bias = bias - step_size * state.exp_avg_bias / denom_bias
    return numpy.asarray(updated, dtype=numpy.float32), float(numpy.float32(updated_bias))


def fit_linear_probe(
    train_residuals: numpy.ndarray,
    train_targets: Sequence[int],
    validation_residuals: numpy.ndarray,
    validation_targets: Sequence[int],
    *,
    layers: Sequence[int],
    task_id: str,
    training_seed: int,
    method_id: str = LINEAR_METHOD_ID,
    preprocessing: str = "standardize",
    profile: LinearV1 | None = None,
    train_domains: Sequence[str] | None = None,
    validation_domains: Sequence[str] | None = None,
    processes: int = 1,
) -> tuple[LayerFit, ...]:
    """Fit each requested layer from its own seed. Input order does not change a layer.

    With ``processes`` above 1, layers are fit in that many spawned processes
    whose BLAS runs on one thread. Mini-batches are small, so one thread per
    layer is much faster than one multithreaded fit at a time.
    """
    resolved = LINEAR_V1 if profile is None else profile
    _check_profile(resolved)
    if task_id not in _TASKS or method_id != LINEAR_METHOD_ID:
        raise ValueError("linear fitting is registered for the two initial tasks")
    if type(training_seed) is not int:
        raise ValueError("training seed must be an integer")
    train = _cube(train_residuals, "training residuals")
    validation = _cube(validation_residuals, "validation residuals")
    train_y = _targets(train_targets, train.shape[0])
    valid_y = _targets(validation_targets, validation.shape[0])
    reject_ood(train_domains, int(train.shape[0]), stage="fitting")
    reject_ood(validation_domains, int(validation.shape[0]), stage="fitting")
    chosen = _layers(layers, int(train.shape[1]))
    if validation.shape[1:] != train.shape[1:]:
        raise ValueError("validation residuals must have the training width and layer count")
    if type(processes) is not int or processes < 1:
        raise ValueError("processes must be a positive integer")
    options = {
        "task_id": task_id,
        "method_id": method_id,
        "training_seed": training_seed,
        "preprocessing": preprocessing,
        "profile": resolved,
        "train_domains": train_domains,
        "validation_domains": validation_domains,
    }
    # Each job holds one layer; it becomes float64 only inside its own fit.
    jobs = [
        (train[:, layer - 1, :], train_y, validation[:, layer - 1, :], valid_y, layer, options)
        for layer in chosen
    ]
    if processes == 1 or len(jobs) == 1:
        fits = [_fit_layer_job(job) for job in jobs]
    else:
        fits = _fit_in_processes(jobs, min(processes, len(jobs)))
    return tuple(sorted(fits, key=lambda item: item.layer))


# BLAS thread settings a fitting process starts with. They are read when numpy loads.
_SINGLE_THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _fit_in_processes(jobs: list[tuple], processes: int) -> list[LayerFit]:
    import multiprocessing
    import os

    saved = {name: os.environ.get(name) for name in _SINGLE_THREAD_ENVIRONMENT}
    os.environ.update(dict.fromkeys(_SINGLE_THREAD_ENVIRONMENT, "1"))
    try:
        # Spawned children copy the environment at start, before they import numpy.
        pool = multiprocessing.get_context("spawn").Pool(processes)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    with pool:
        return pool.map(_fit_layer_job, jobs, chunksize=1)


def _fit_layer_job(job: tuple) -> LayerFit:
    train, train_y, validation, valid_y, layer, options = job
    return _fit_layer(
        numpy.asarray(train, dtype=numpy.float64),
        train_y,
        numpy.asarray(validation, dtype=numpy.float64),
        valid_y,
        layer=layer,
        **options,
    )


def _fit_layer(
    train: numpy.ndarray,
    train_targets: numpy.ndarray,
    validation: numpy.ndarray,
    validation_targets: numpy.ndarray,
    *,
    layer: int,
    task_id: str,
    method_id: str,
    training_seed: int,
    preprocessing: str,
    profile: LinearV1,
    train_domains: Sequence[str] | None,
    validation_domains: Sequence[str] | None,
) -> LayerFit:
    standardizer = fit_standardizer(train, mode=preprocessing, domains=train_domains)
    design = apply_standardizer(train, standardizer)
    held_out = apply_standardizer(validation, standardizer)
    init = initialization_seed(training_seed=training_seed, task_id=task_id, method_id=method_id, layer=layer)
    shuffle = shuffle_seed(training_seed=training_seed, task_id=task_id, method_id=method_id, layer=layer)
    weight_tuple, bias = initialize_logistic(seed=init, width=design.shape[1])
    weight = numpy.asarray(weight_tuple, dtype=numpy.float32)
    state = AdamState.zeros(design.shape[1])
    generator = pcg64_generator(shuffle)
    checkpoints: list[EpochCheckpoint] = []
    best = math_inf()
    stall = 0
    steps = 0
    stop_reason = "max_epochs"
    count = int(design.shape[0])
    for epoch in range(1, profile.max_epochs + 1):
        order = generator.permutation(count)
        for start in range(0, count, profile.batch_size):
            batch = order[start:start + profile.batch_size]
            _, grad_weight, grad_bias = binary_cross_entropy_gradients(
                design[batch], weight, bias, train_targets[batch],
            )
            weight, bias = adamw_step(weight, bias, grad_weight, grad_bias, state, profile)
            steps += 1
        validation_loss = mean_binary_cross_entropy(
            held_out @ weight + bias,
            validation_targets,
            domains=validation_domains,
        )
        checkpoints.append(EpochCheckpoint(
            layer=layer,
            epoch=epoch,
            validation_cross_entropy=validation_loss,
            weight=tuple(float(item) for item in weight),
            bias=float(bias),
            completed=True,
        ))
        if validation_loss < best:
            best = validation_loss
            stall = 0
        else:
            stall += 1
        if stall >= profile.early_stopping_patience:
            stop_reason = "early_stopping"
            break
    return LayerFit(
        layer=layer,
        task_id=task_id,
        method_id=method_id,
        init_seed=init,
        shuffle_seed=shuffle,
        standardizer=standardizer,
        checkpoints=tuple(checkpoints),
        optimizer_steps=steps,
        stop_reason=stop_reason,
        positive_count=int(train_targets.sum()),
        count=count,
        profile=profile,
    )


def _check_profile(profile: LinearV1) -> None:
    if type(profile) is not LinearV1:
        raise TypeError("the optimization profile must be a resolved LinearV1 record")
    if profile.optimizer != "AdamW" or profile.weight_decay_on != "weights":
        raise ValueError("linear_v1 uses AdamW with weight decay on weights only")
    if profile.mixed_precision or profile.scheduler is not None or profile.gradient_clipping is not None:
        raise ValueError("linear fitting has no scheduler, gradient clipping, or mixed precision")
    if profile.batch_size < 1 or profile.max_epochs < 1 or profile.early_stopping_patience < 1:
        raise ValueError("batch size, epoch cap, and patience must be positive")
    if profile.betas != (0.9, 0.999) or profile.epsilon != 1e-8:
        raise ValueError("AdamW betas are (0.9, 0.999) and epsilon is 1e-8")


def _cube(values: numpy.ndarray, label: str) -> numpy.ndarray:
    """Check the shape. Float arrays keep their dtype; each layer is widened to float64 when fit."""
    array = numpy.asarray(values)
    if array.dtype.kind != "f":
        array = array.astype(numpy.float64)
    if array.ndim != 3 or array.shape[0] == 0 or array.shape[1] == 0 or array.shape[2] == 0:
        raise ValueError(f"{label} must have shape [rows, layers, width]")
    return array


def _targets(values: Sequence[int], count: int) -> numpy.ndarray:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != count:
        raise ValueError("targets must cover every residual row")
    parsed = []
    for value in values:
        if type(value) is not int or value not in (0, 1):
            raise ValueError("each target must be 0 or 1")
        parsed.append(value)
    return numpy.asarray(parsed, dtype=numpy.float32)


def _layers(layers: Sequence[int], available: int) -> tuple[int, ...]:
    if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence) or not layers:
        raise ValueError("at least one layer is required")
    chosen = tuple(layers)
    if any(type(layer) is not int or layer < 1 or layer > available for layer in chosen):
        raise ValueError("layer indices start at 1 and must exist in the residual tensor")
    if len(set(chosen)) != len(chosen):
        raise ValueError("each layer is fitted once")
    return chosen


def math_sqrt(value: float) -> float:
    return float(numpy.sqrt(value))


def math_inf() -> float:
    return float("inf")
