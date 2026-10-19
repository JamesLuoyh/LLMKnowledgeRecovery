"""Cross-entropy, accuracy, K, scalar Brier, and top-label ECE.

Brier is the mean squared error of one positive-class probability, not the
sum of both classes. ECE bins the probability of the predicted label on
``[0, 1]``; it does not bin the probability of the true label. An exact
probability of zero on an observed label is an infinite cross-entropy with
an explicit flag, so exports do not need nonstandard JSON numbers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

BRIER_CONVENTION = "scalar_positive_class"
DEFAULT_ECE_BINS = 15


@dataclass(frozen=True)
class CalibrationBin:
    """One equal-width confidence bin. An empty bin has no accuracy."""

    lower: float
    upper: float
    count: int
    accuracy: float | None
    mean_confidence: float | None


@dataclass(frozen=True)
class BinaryMetrics:
    """Uniform-row metrics. ``None`` means a zero denominator or an infinite value."""

    positive_class: str
    tie_prediction: str
    count: int
    cross_entropy: float | None
    cross_entropy_infinite: bool
    accuracy: float | None
    mean_true_class_probability: float | None
    brier: float | None
    ece: float | None
    bins: tuple[CalibrationBin, ...]
    unavailable_reason: str | None


def awareness_metrics(
    agreement_probabilities: Sequence[float],
    targets: Sequence[int],
    *,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> BinaryMetrics:
    """Score P(agree). A probability of exactly 0.5 predicts agreement.

    Targets use class order ``[disagree, agree]``: 1 is agreement.
    """
    return _binary_metrics(
        agreement_probabilities,
        targets,
        positive_class="agree",
        tie_prediction="agree",
        tie_predicts_positive=True,
        ece_bins=ece_bins,
    )


def consistency_metrics(
    b_probabilities: Sequence[float],
    targets: Sequence[int],
    *,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> BinaryMetrics:
    """Score P(B). A probability of exactly 0.5 predicts A.

    Targets use class order ``[A, B]``: 1 is B.
    """
    return _binary_metrics(
        b_probabilities,
        targets,
        positive_class="B",
        tie_prediction="A",
        tie_predicts_positive=False,
        ece_bins=ece_bins,
    )


def metrics_document(metrics: BinaryMetrics) -> dict[str, object]:
    """JSON-safe metrics. Infinite cross-entropy is a flag and a null value."""
    return {
        "positive_class": metrics.positive_class,
        "tie_prediction": metrics.tie_prediction,
        "count": metrics.count,
        "cross_entropy": metrics.cross_entropy,
        "cross_entropy_infinite": metrics.cross_entropy_infinite,
        "accuracy": metrics.accuracy,
        "mean_true_class_probability": metrics.mean_true_class_probability,
        "brier": metrics.brier,
        "brier_convention": BRIER_CONVENTION,
        "ece": metrics.ece,
        "ece_bins": len(metrics.bins),
        "unavailable_reason": metrics.unavailable_reason,
        "bins": [
            {
                "lower": item.lower,
                "upper": item.upper,
                "count": item.count,
                "accuracy": item.accuracy,
                "mean_confidence": item.mean_confidence,
            }
            for item in metrics.bins
        ],
    }


def _binary_metrics(
    probabilities: Sequence[float],
    targets: Sequence[int],
    *,
    positive_class: str,
    tie_prediction: str,
    tie_predicts_positive: bool,
    ece_bins: int,
) -> BinaryMetrics:
    positive = _probabilities(probabilities)
    observed = _targets(targets)
    if len(positive) != len(observed):
        raise ValueError("probabilities and targets must have the same length")
    bins = _bin_count(ece_bins)
    edges = _edges(bins)
    if not positive:
        return BinaryMetrics(
            positive_class=positive_class,
            tie_prediction=tie_prediction,
            count=0,
            cross_entropy=None,
            cross_entropy_infinite=False,
            accuracy=None,
            mean_true_class_probability=None,
            brier=None,
            ece=None,
            bins=_empty_bins(edges),
            unavailable_reason=(
                "Zero rows: cross-entropy, accuracy, K, Brier, and ECE are unavailable. "
                "The denominator is 0."
            ),
        )
    correct = 0
    true_class_total = 0.0
    brier_total = 0.0
    cross_entropy_total = 0.0
    infinite = False
    bin_counts = [0] * bins
    bin_correct = [0] * bins
    bin_confidence = [0.0] * bins
    for probability, target in zip(positive, observed, strict=True):
        predicted = _predict(probability, tie_predicts_positive=tie_predicts_positive)
        # Top-label confidence is at least one half: it is the probability of
        # the predicted class, not the probability of the true class.
        confidence = probability if predicted == 1 else 1.0 - probability
        true_probability = probability if target == 1 else 1.0 - probability
        correct += predicted == target
        true_class_total += true_probability
        brier_total += (probability - target) ** 2
        if true_probability == 0.0:
            infinite = True
        else:
            cross_entropy_total += -math.log(true_probability)
        index = _bin_index(confidence, bins)
        bin_counts[index] += 1
        bin_correct[index] += predicted == target
        bin_confidence[index] += confidence
    count = len(positive)
    calibrated = _populated_bins(edges, bin_counts, bin_correct, bin_confidence)
    ece = sum(
        (item.count / count) * abs(item.accuracy - item.mean_confidence)
        for item in calibrated
        if item.count
    )
    return BinaryMetrics(
        positive_class=positive_class,
        tie_prediction=tie_prediction,
        count=count,
        cross_entropy=None if infinite else cross_entropy_total / count,
        cross_entropy_infinite=infinite,
        accuracy=correct / count,
        mean_true_class_probability=true_class_total / count,
        brier=brier_total / count,
        ece=ece,
        bins=calibrated,
        unavailable_reason=None,
    )


def _probabilities(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("probabilities must be a sequence of probabilities")
    parsed: list[float] = []
    for value in values:
        if type(value) is bool or type(value) not in (int, float):
            raise TypeError("probabilities must be a sequence of probabilities")
        if not math.isfinite(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("each probability must be finite and between 0 and 1")
        parsed.append(float(value))
    return tuple(parsed)


def _targets(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("targets must be a sequence of 0/1 labels")
    parsed: list[int] = []
    for value in values:
        if type(value) is not int or value not in (0, 1):
            raise ValueError("each target must be 0 or 1")
        parsed.append(value)
    return tuple(parsed)


def _bin_count(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("ece_bins must be a positive integer")
    return value


def _predict(probability: float, *, tie_predicts_positive: bool) -> int:
    if probability > 0.5:
        return 1
    if probability < 0.5:
        return 0
    return 1 if tie_predicts_positive else 0


def _edges(bins: int) -> tuple[float, ...]:
    return tuple(index / bins for index in range(bins + 1))


def _bin_index(confidence: float, bins: int) -> int:
    """Place ``confidence`` in ``[b/n, (b+1)/n)``, with the last edge closed."""
    for index in range(bins - 1, -1, -1):
        if confidence >= index / bins:
            return index
    return 0


def _empty_bins(edges: Sequence[float]) -> tuple[CalibrationBin, ...]:
    return tuple(
        CalibrationBin(edges[index], edges[index + 1], 0, None, None)
        for index in range(len(edges) - 1)
    )


def _populated_bins(
    edges: Sequence[float],
    counts: Sequence[int],
    correct: Sequence[int],
    confidence: Sequence[float],
) -> tuple[CalibrationBin, ...]:
    bins: list[CalibrationBin] = []
    for index, count in enumerate(counts):
        if count == 0:
            bins.append(CalibrationBin(edges[index], edges[index + 1], 0, None, None))
            continue
        bins.append(CalibrationBin(
            edges[index],
            edges[index + 1],
            count,
            correct[index] / count,
            confidence[index] / count,
        ))
    return tuple(bins)
