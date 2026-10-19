"""Frozen base-group splits.

Group ids are sorted lexicographically, then shuffled with
``numpy.random.Generator(numpy.random.PCG64(split_seed)).permutation(n)``.
Train and validation receive floor counts of the decimal fractions. The
remainder goes to test, so a binary rounding shortfall cannot drop groups
or move the leftover into another split. The probe training seed is not an
input. The grouping version names the installed NumPy release because that
permutation is part of the saved assignment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import Decimal

import numpy

from knowledge_recovery.records import SplitManifest

GROUPING_ALGORITHM = "lexicographic-pcg64-floor-remainder-v1"
SPLIT_NAMES = ("train", "val", "test")


def grouping_version() -> str:
    """Algorithm id plus the NumPy version that implements PCG64 permutation."""
    return f"{GROUPING_ALGORITHM}/numpy-{numpy.__version__}"


def freeze_grouped_split(
    group_ids: Sequence[str],
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    split_seed: int,
) -> SplitManifest:
    """Assign every base group once. Input order does not change the mapping."""
    groups = _group_ids(group_ids)
    fractions = (
        _fraction(train_fraction, "train_fraction"),
        _fraction(val_fraction, "val_fraction"),
        _fraction(test_fraction, "test_fraction"),
    )
    if abs(sum(fractions, Decimal(0)) - 1) > Decimal("1e-12"):
        raise ValueError("split fractions must sum to 1")
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a non-negative integer")
    ordered = tuple(sorted(groups))
    counts = _allocations(len(ordered), fractions)
    generator = numpy.random.Generator(numpy.random.PCG64(split_seed))
    permutation = generator.permutation(len(ordered))
    shuffled = [ordered[int(index)] for index in permutation]
    names = (
        ["train"] * counts[0]
        + ["val"] * counts[1]
        + ["test"] * counts[2]
    )
    return SplitManifest(
        group_to_split={group_id: split for group_id, split in zip(shuffled, names, strict=True)},
        source_universe=ordered,
        grouping_version=grouping_version(),
        split_seed=split_seed,
        fractions={name: float(value) for name, value in zip(SPLIT_NAMES, (
            train_fraction, val_fraction, test_fraction,
        ), strict=True)},
        allocations={name: count for name, count in zip(SPLIT_NAMES, counts, strict=True)},
    )


def _group_ids(group_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(group_ids, (str, bytes)) or not isinstance(group_ids, Sequence):
        raise TypeError("group_ids must be a sequence of base-group ids")
    parsed: list[str] = []
    for group_id in group_ids:
        if type(group_id) is not str or group_id == "":
            raise ValueError("each base-group id must be a non-empty string")
        parsed.append(group_id)
    if len(set(parsed)) != len(parsed):
        raise ValueError("duplicate base-group id")
    return tuple(parsed)


def _fraction(value: object, label: str) -> Decimal:
    if type(value) is bool or type(value) not in (int, float):
        raise TypeError(f"{label} must be a fraction")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite numbers are rejected")
    if not 0 < number < 1:
        raise ValueError(f"{label} must be strictly between 0 and 1")
    return Decimal(str(number))


def _allocations(count: int, fractions: tuple[Decimal, Decimal, Decimal]) -> tuple[int, int, int]:
    """Floor the decimal train and validation products and give the remainder to test."""
    train = math.floor(Decimal(count) * fractions[0])
    validation = math.floor(Decimal(count) * fractions[1])
    test = count - train - validation
    if test < 0:
        raise ValueError("split fractions assign more groups than the source universe")
    return train, validation, test
