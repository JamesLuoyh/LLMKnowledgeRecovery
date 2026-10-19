"""Raw, conditional, and greedy scores for one binary A/B continuation.

Log-softmax and the two-token log-sum-exp run in float32. The result keeps
only the A and B probabilities: the full-vocabulary probability vector is not
materialized. Token ids are continuation ids supplied by the audit. This
module does not retokenize a prompt with an answer appended to it.
"""

from __future__ import annotations

import array
import math
from collections.abc import Sequence
from dataclasses import dataclass

from knowledge_recovery.records import AnswerScores

SCORING_VERSION = "binary-ab-float32-v1"


@dataclass(frozen=True)
class CanonicalTokens:
    """Continuation token ids for labels A and B, in that label order."""

    A: int
    B: int

    def __post_init__(self) -> None:
        for label in ("A", "B"):
            value = getattr(self, label)
            if type(value) is not int or value < 0:
                raise ValueError(f"token id for {label} must be a non-negative integer")
        if self.A == self.B:
            raise ValueError("canonical token ids for A and B must be distinct")


def score_binary_answers(logits: Sequence[float], tokens: CanonicalTokens) -> AnswerScores:
    """Score one next-token distribution under the audited A/B token ids.

    Raw probabilities are the full-vocabulary softmax values at those ids.
    Canonical probabilities are the softmax of the two canonical logits, so
    they are conditional on the continuation being A or B. Their total raw
    mass is ``exp(logsumexp(A, B) - logsumexp(vocabulary))``. An exact tie
    among canonical logits selects A. An exact tie over the vocabulary
    selects the smallest token id.
    """
    if not isinstance(tokens, CanonicalTokens):
        raise TypeError("tokens must be CanonicalTokens")
    values = _float32_logits(logits)
    _check_bounds(tokens, len(values))
    full_log_z = _logsumexp(values)
    canonical_logits = (values[tokens.A], values[tokens.B])
    canonical_log_z = _logsumexp(canonical_logits)
    raw = tuple(_probability(logit - full_log_z) for logit in canonical_logits)
    canonical = tuple(_probability(logit - canonical_log_z) for logit in canonical_logits)
    mass = _probability(canonical_log_z - full_log_z)
    greedy = _smallest_argmax(values)
    return AnswerScores(
        raw_ab_probabilities=raw,
        canonical_ab_probabilities=canonical,
        canonical_label_mass=mass,
        unrestricted_greedy_token_id=greedy,
        canonical_greedy_label="B" if canonical_logits[1] > canonical_logits[0] else "A",
        invalid_label=greedy not in (tokens.A, tokens.B),
        scoring_version=SCORING_VERSION,
    )


def _float32_logits(logits: Sequence[float]) -> tuple[float, ...]:
    if isinstance(logits, (str, bytes)) or not isinstance(logits, Sequence):
        raise TypeError("logits must be a sequence of real numbers")
    if len(logits) == 0:
        raise ValueError("logits must contain one value per vocabulary token")
    return tuple(_float32_logit(item) for item in logits)


def _float32_logit(value: object) -> float:
    if type(value) is bool or type(value) not in (int, float):
        raise TypeError("logits must be a sequence of real numbers")
    if not math.isfinite(value):
        raise ValueError("nonfinite numbers are rejected")
    converted = array.array("f", [float(value)])[0]
    if not math.isfinite(converted):
        raise ValueError("nonfinite numbers are rejected")
    return converted


def _check_bounds(tokens: CanonicalTokens, vocabulary_size: int) -> None:
    for label in ("A", "B"):
        token_id = getattr(tokens, label)
        if token_id >= vocabulary_size:
            raise ValueError(
                f"token id for {label} is {token_id}, outside the vocabulary of size {vocabulary_size}"
            )


def _logsumexp(values: Sequence[float]) -> float:
    """Float32 log-sum-exp. ``values`` are already float32 scalars."""
    peak = max(values)
    total = 0.0
    for value in values:
        total = _round_float32(total + math.exp(value - peak))
    return _round_float32(peak + math.log(total))


def _probability(log_probability: float) -> float:
    probability = _round_float32(math.exp(_round_float32(log_probability)))
    if not math.isfinite(probability):
        raise ValueError("nonfinite numbers are rejected")
    return probability


def _round_float32(value: float) -> float:
    return array.array("f", [value])[0]


def _smallest_argmax(values: Sequence[float]) -> int:
    chosen = 0
    best = values[0]
    for index, value in enumerate(values):
        if value > best:
            chosen = index
            best = value
    return chosen
