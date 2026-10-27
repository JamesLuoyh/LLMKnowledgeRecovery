"""Raw, conditional, and greedy scores for one binary A/B continuation.

Extraction reduces each row's next-token logits to four float32 values
(``AB_SUMMARY_FIELDS``): the A and B logits, the vocabulary log-normalizer,
and the greedy token id. ``score_binary_summary`` scores one such row. The
full-vocabulary probability vector is never stored. Token ids are
continuation ids supplied by the audit. This module does not retokenize a
prompt with an answer appended to it.

Raw probabilities are the full-vocabulary softmax values at the A and B ids.
Canonical probabilities are the softmax of the two canonical logits, so they
are conditional on the continuation being A or B. Their total raw mass is
``exp(logsumexp(A, B) - log_z)``. An exact tie among canonical logits selects
A. An exact tie over the vocabulary selects the smallest token id.
"""

from __future__ import annotations

import array
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy

from knowledge_recovery.records import AnswerScores

SCORING_VERSION = "binary-ab-float32-v1"
# Stored per row, in this order.
AB_SUMMARY_FIELDS = ("logit_A", "logit_B", "log_z", "greedy_token_id")
# Token ids are stored as float32, which is exact below 2**24.
_MAX_SUMMARY_VOCABULARY = 2 ** 24


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


def summarize_binary_logits(logits: numpy.ndarray, tokens: CanonicalTokens) -> numpy.ndarray:
    """Reduce ``[rows, vocab]`` logits to ``[rows, 4]`` in ``AB_SUMMARY_FIELDS`` order.

    The greedy token is the smallest id among exact ties.
    """
    if not isinstance(tokens, CanonicalTokens):
        raise TypeError("tokens must be CanonicalTokens")
    values = numpy.asarray(logits, dtype=numpy.float32)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("logits are [rows, vocab]")
    if values.shape[1] > _MAX_SUMMARY_VOCABULARY:
        raise ValueError("the vocabulary is too large to store token ids as float32")
    if not numpy.isfinite(values).all():
        raise ValueError("nonfinite numbers are rejected")
    _check_bounds(tokens, values.shape[1])
    peak = values.max(axis=1)
    log_z = peak + numpy.log(numpy.exp(values - peak[:, None]).sum(axis=1, dtype=numpy.float32))
    greedy = values.argmax(axis=1)
    return numpy.stack(
        [values[:, tokens.A], values[:, tokens.B], log_z, greedy.astype(numpy.float32)],
        axis=1,
    ).astype(numpy.float32)


def score_binary_summary(summary: Sequence[float], tokens: CanonicalTokens) -> AnswerScores:
    """Score one stored summary row under the audited A/B token ids."""
    if not isinstance(tokens, CanonicalTokens):
        raise TypeError("tokens must be CanonicalTokens")
    values = _float32_logits(summary)
    if len(values) != len(AB_SUMMARY_FIELDS):
        raise ValueError(f"a summary row holds {', '.join(AB_SUMMARY_FIELDS)}")
    logit_a, logit_b, full_log_z, greedy_value = values
    if greedy_value < 0 or greedy_value != int(greedy_value):
        raise ValueError("the stored greedy token id is not a token id")
    greedy = int(greedy_value)
    canonical_logits = (logit_a, logit_b)
    canonical_log_z = _logsumexp(canonical_logits)
    return AnswerScores(
        raw_ab_probabilities=tuple(_probability(logit - full_log_z) for logit in canonical_logits),
        canonical_ab_probabilities=tuple(_probability(logit - canonical_log_z) for logit in canonical_logits),
        canonical_label_mass=_probability(canonical_log_z - full_log_z),
        unrestricted_greedy_token_id=greedy,
        canonical_greedy_label="B" if logit_b > logit_a else "A",
        invalid_label=greedy not in (tokens.A, tokens.B),
        scoring_version=SCORING_VERSION,
    )


def _float32_logits(logits: Sequence[float]) -> tuple[float, ...]:
    if isinstance(logits, (str, bytes)) or not isinstance(logits, Sequence):
        raise TypeError("logits must be a sequence of real numbers")
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
