"""One logistic model per layer.

Standardization uses the training rows in the order they are given, in
float64, with population standard deviation (denominator ``n``). A feature
whose training standard deviation is zero is rescaled by 1. Weights and the
scalar bias are drawn independently from ``[-1/sqrt(d), 1/sqrt(d))`` on the
CPU mt19937 stream. OOD rows are rejected before any statistic is computed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy

from knowledge_recovery.tasks.base import AWARENESS_TASK_ID, CONSISTENCY_TASK_ID
from knowledge_recovery.training.losses import reject_ood, sigmoid
from knowledge_recovery.training.seeds import CpuTensorGenerator, derived_seed

LINEAR_METHOD_ID = "linear"
POSITIVE_CLASS = {
    AWARENESS_TASK_ID: "agree",
    CONSISTENCY_TASK_ID: "B",
}


@dataclass(frozen=True)
class Standardizer:
    """Training-view transform recorded in the checkpoint."""

    mode: str
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    zero_variance: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.mode not in ("standardize", "none"):
            raise ValueError("preprocessing must be standardize or none")
        if len(self.mean) != len(self.scale) or not self.mean:
            raise ValueError("mean and scale must cover every feature")
        if any(item == 0 for item in self.scale):
            raise ValueError("a zero standard deviation is replaced by 1")


@dataclass(frozen=True)
class LinearProbe:
    """Saved logistic weights for one task and one layer."""

    task_id: str
    layer: int
    epoch: int
    weight: tuple[float, ...]
    bias: float
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    preprocessing: str
    validation_cross_entropy: float
    method_id: str = LINEAR_METHOD_ID
    selected_layer: int | None = None

    def __post_init__(self) -> None:
        if self.task_id not in POSITIVE_CLASS or self.method_id != LINEAR_METHOD_ID:
            raise ValueError("linear probes are registered per task with method_id linear")
        if len(self.weight) != len(self.mean) or len(self.weight) != len(self.scale):
            raise ValueError("weight, mean, and scale must share the feature width")
        if self.selected_layer is not None and self.selected_layer != self.layer:
            raise ValueError("selected_layer is the layer these weights were fitted on")

    def predict(self, residuals: numpy.ndarray) -> tuple[float, ...]:
        """Return the positive-class probability. Awareness uses agreement; consistency uses B."""
        transformed = apply_standardizer(residuals, Standardizer(
            self.preprocessing, self.mean, self.scale, (),
        ))
        logits = numpy.asarray(transformed, dtype=numpy.float32) @ numpy.asarray(self.weight, dtype=numpy.float32) + numpy.float32(self.bias)
        return tuple(float(item) for item in sigmoid(logits))


def fit_standardizer(
    features: numpy.ndarray,
    *,
    mode: str = "standardize",
    domains: Sequence[str] | None = None,
) -> Standardizer:
    """Estimate a per-feature transform from training rows only."""
    design = numpy.asarray(features, dtype=numpy.float64)
    if design.ndim != 2 or design.shape[0] == 0 or design.shape[1] == 0:
        raise ValueError("training features must have shape [rows, width]")
    reject_ood(domains, int(design.shape[0]), stage="feature normalization")
    width = int(design.shape[1])
    if mode == "none":
        return Standardizer("none", tuple(0.0 for _ in range(width)), tuple(1.0 for _ in range(width)), ())
    if mode != "standardize":
        raise ValueError("preprocessing must be standardize or none")
    count = design.shape[0]
    mean = design.sum(axis=0) / count
    variance = ((design - mean) ** 2).sum(axis=0) / count
    scale = numpy.sqrt(variance)
    zero_variance = tuple(int(index) for index in numpy.flatnonzero(scale == 0.0))
    scale = scale.copy()
    scale[scale == 0.0] = 1.0
    return Standardizer(
        "standardize",
        tuple(float(item) for item in mean),
        tuple(float(item) for item in scale),
        zero_variance,
    )


def apply_standardizer(features: numpy.ndarray, standardizer: Standardizer) -> numpy.ndarray:
    """Apply a stored transform and return float32 features."""
    design = numpy.asarray(features, dtype=numpy.float32).astype(numpy.float64)
    if design.ndim != 2 or design.shape[1] != len(standardizer.mean):
        raise ValueError("features must have the recorded width")
    mean = numpy.asarray(standardizer.mean, dtype=numpy.float64)
    scale = numpy.asarray(standardizer.scale, dtype=numpy.float64)
    return ((design - mean) / scale).astype(numpy.float32)


def initialize_logistic(*, seed: int, width: int) -> tuple[tuple[float, ...], float]:
    """Draw weight, then bias, from the CPU uniform stream."""
    if type(width) is not int or width < 1:
        raise ValueError("width must be a positive integer")
    bound = 1.0 / math.sqrt(width)
    draws = CpuTensorGenerator(seed).uniform(width + 1, -bound, bound)
    weight = tuple(float(item) for item in draws[:width])
    return weight, float(draws[width])


def initialization_seed(*, training_seed: int, task_id: str, method_id: str, layer: int) -> int:
    """Seed passed to the CPU tensor generator."""
    return derived_seed(
        seed=training_seed, task_id=task_id, method_id=method_id, layer=layer, stream="init",
    )


def shuffle_seed(*, training_seed: int, task_id: str, method_id: str, layer: int) -> int:
    """Seed passed to NumPy PCG64."""
    return derived_seed(
        seed=training_seed, task_id=task_id, method_id=method_id, layer=layer, stream="shuffle",
    )
