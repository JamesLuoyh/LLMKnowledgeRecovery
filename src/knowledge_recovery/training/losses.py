"""Logits-based binary cross-entropy.

The mean is the unregularized training and validation loss. Weight decay
stays in the optimizer. OOD rows are rejected before any reduction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy


def reject_ood(domains: Sequence[str] | None, count: int, *, stage: str) -> None:
    """Refuse a row whose domain is OOD before it can enter ``stage``."""
    if domains is None:
        return
    if isinstance(domains, (str, bytes)) or not isinstance(domains, Sequence) or len(domains) != count:
        raise ValueError("domain labels must cover every row")
    if any(item == "ood" for item in domains):
        raise ValueError(f"OOD rows never enter {stage}")


def mean_binary_cross_entropy(
    logits: numpy.ndarray,
    targets: numpy.ndarray,
    *,
    domains: Sequence[str] | None = None,
) -> float:
    """Mean stable BCE. A logit of large magnitude stays finite."""
    scores = numpy.asarray(logits, dtype=numpy.float64)
    labels = numpy.asarray(targets, dtype=numpy.float32)
    if scores.shape != labels.shape or scores.ndim != 1 or scores.shape[0] == 0:
        raise ValueError("logits and targets must be non-empty vectors of equal length")
    reject_ood(domains, int(scores.shape[0]), stage="the loss")
    losses = numpy.maximum(scores, 0.0) - scores * labels + numpy.log1p(numpy.exp(-numpy.abs(scores)))
    return float(losses.mean())


def binary_cross_entropy_gradients(
    features: numpy.ndarray,
    weight: numpy.ndarray,
    bias: float,
    targets: numpy.ndarray,
) -> tuple[float, numpy.ndarray, float]:
    """Mean-batch BCE and its gradients for one linear logit."""
    design = numpy.asarray(features, dtype=numpy.float32)
    coefficient = numpy.asarray(weight, dtype=numpy.float32)
    labels = numpy.asarray(targets, dtype=numpy.float32)
    logits = design @ coefficient + numpy.float32(bias)
    loss = mean_binary_cross_entropy(logits, labels)
    probabilities = _sigmoid(logits)
    residual = (probabilities - labels) / labels.shape[0]
    return loss, design.T @ residual, float(residual.sum())


def _sigmoid(logits: numpy.ndarray) -> numpy.ndarray:
    positive = logits >= 0
    output = numpy.empty_like(logits)
    output[positive] = 1.0 / (1.0 + numpy.exp(-logits[positive]))
    exponential = numpy.exp(logits[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def sigmoid(logits: numpy.ndarray) -> numpy.ndarray:
    """Numerically stable logistic function."""
    return _sigmoid(numpy.asarray(logits, dtype=numpy.float64))


def finite_loss(logit: float, target: int) -> float:
    """One-row BCE used by hand fixtures."""
    return mean_binary_cross_entropy(numpy.array([logit]), numpy.array([target]))
