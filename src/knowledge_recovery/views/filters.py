"""Fixed four-step filter for clean orders.

A row passes only in this order: the unrestricted greedy token is an option,
that option's normalized A/B probability is greater than the threshold, and
that option is the correct label. A base question is eligible when both of
its clean orders pass. The report counts samples before the filters and
after each step, for both orders together and for each correct-answer
position. There is no raw-versus-canonical choice.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge_recovery.provenance import content_digest, identify
from knowledge_recovery.records import FilterReport

SPLITS = ("train", "val", "test")
ORDERS = ("correct_at_A", "correct_at_B")
COUNT_SETS = ("both_orders", "correct_at_A", "correct_at_B")
COUNT_STEPS = (
    "before",
    "after_greedy_is_option",
    "after_option_probability",
    "after_greedy_correct",
    "after_both_orders_correct",
)


@dataclass(frozen=True)
class CleanOrder:
    """One clean prompt row. Probabilities are in label order ``(A, B)``."""

    base_group_id: str
    split: str
    order_id: str
    raw_ab_probabilities: tuple[float, float]
    canonical_ab_probabilities: tuple[float, float]
    canonical_label_mass: float
    invalid_label: bool

    def __post_init__(self) -> None:
        if type(self.base_group_id) is not str or self.base_group_id == "":
            raise ValueError("base_group_id must be a non-empty string")
        if self.split not in SPLITS:
            raise ValueError("split must be train, val, or test")
        if self.order_id not in ORDERS:
            raise ValueError("order_id must be correct_at_A or correct_at_B")
        object.__setattr__(
            self, "raw_ab_probabilities", _probability_pair(self.raw_ab_probabilities, "raw")
        )
        object.__setattr__(
            self,
            "canonical_ab_probabilities",
            _probability_pair(self.canonical_ab_probabilities, "canonical"),
        )
        if type(self.canonical_label_mass) is bool or type(self.canonical_label_mass) not in (int, float):
            raise TypeError("canonical_label_mass must be a probability")
        mass = float(self.canonical_label_mass)
        if not math.isfinite(mass) or not 0.0 <= mass <= 1.0:
            raise ValueError("canonical_label_mass must be between 0 and 1")
        object.__setattr__(self, "canonical_label_mass", mass)
        if type(self.invalid_label) is not bool:
            raise ValueError("invalid_label must be a boolean")


@dataclass(frozen=True)
class _Pair:
    base_group_id: str
    split: str
    correct_at_a: CleanOrder
    correct_at_b: CleanOrder
    passed: bool


@dataclass(frozen=True)
class FilterOutcome:
    """Eligible base questions and the digest of the count report."""

    eligible_base_group_ids: tuple[str, ...]
    report_digest: str
    min_correct_prob: float


def apply_filter_gate(
    orders: Sequence[CleanOrder],
    *,
    output_directory: str | Path,
    min_correct_prob: float = 0.8,
    variant_id: str = "clean",
) -> FilterOutcome:
    """Write the four-step counts and return questions that pass both orders.

    ``min_correct_prob`` is a strict lower bound on the greedy option's
    normalized probability. A value equal to the threshold does not pass.
    """
    threshold = _threshold(min_correct_prob)
    if type(variant_id) is not str or variant_id == "":
        raise ValueError("variant_id must be a non-empty string")
    pairs = _pairs(orders, threshold)
    directory = Path(output_directory)
    report = _report(pairs, threshold=threshold, variant_id=variant_id, directory=directory)
    digest = _write(directory, report, _render(report))
    eligible = tuple(pair.base_group_id for pair in pairs if pair.passed)
    return FilterOutcome(
        eligible_base_group_ids=eligible,
        report_digest=digest,
        min_correct_prob=threshold,
    )


def _pairs(orders: Sequence[CleanOrder], threshold: float) -> tuple[_Pair, ...]:
    if isinstance(orders, (str, bytes)) or not isinstance(orders, Sequence):
        raise TypeError("orders must be a sequence of CleanOrder values")
    grouped: dict[str, dict[str, CleanOrder]] = {}
    for order in orders:
        if type(order) is not CleanOrder:
            raise TypeError("orders must be a sequence of CleanOrder values")
        found = grouped.setdefault(order.base_group_id, {})
        if order.order_id in found:
            raise ValueError(f"duplicate clean order for {order.base_group_id} {order.order_id}")
        found[order.order_id] = order
    pairs: list[_Pair] = []
    for base_group_id in sorted(grouped):
        found = grouped[base_group_id]
        if set(found) != set(ORDERS):
            missing = [name for name in ORDERS if name not in found]
            raise ValueError(
                f"base group {base_group_id} is missing clean orders: {', '.join(missing)}"
            )
        first, second = found["correct_at_A"], found["correct_at_B"]
        if first.split != second.split:
            raise ValueError(f"base group {base_group_id} has clean orders in different splits")
        passed = _last_step(first, threshold) >= 3 and _last_step(second, threshold) >= 3
        pairs.append(_Pair(base_group_id, first.split, first, second, passed))
    return tuple(pairs)


def _last_step(order: CleanOrder, threshold: float) -> int:
    """Return 0–3, the last of the first three steps this order passes."""
    if order.invalid_label:
        return 0
    if max(order.canonical_ab_probabilities) <= threshold:
        return 1
    expected = "A" if order.order_id == "correct_at_A" else "B"
    if _greedy_option(order) != expected:
        return 2
    return 3


def _greedy_option(order: CleanOrder) -> str:
    """Normalized A/B argmax. An exact tie selects A."""
    probability_a, probability_b = order.canonical_ab_probabilities
    return "B" if probability_b > probability_a else "A"


def _report(
    pairs: Sequence[_Pair],
    *,
    threshold: float,
    variant_id: str,
    directory: Path,
) -> FilterReport:
    return FilterReport(
        paired_eligibility={
            "variant": variant_id,
            "threshold": threshold,
            "comparison": "greater_than",
            "require_both_orders": True,
            "eligible_base_group_ids": [pair.base_group_id for pair in pairs if pair.passed],
        },
        order_failures=_failures(pairs, threshold),
        population_counts={
            "overall": _block(pairs, threshold),
            "by_split": {split: _block(_in_split(pairs, split), threshold) for split in SPLITS},
        },
        split_denominators={split: _units(_in_split(pairs, split)) for split in SPLITS},
        canonical_only_diagnostics={
            "policy": "fixed_four_step",
            "steps": list(COUNT_STEPS[1:]),
            "option_probability": "normalized_ab",
        },
        paths={
            "directory": str(directory),
            "json": "filter_report.json",
            "text": "filter_report.txt",
        },
    )


def _block(pairs: Sequence[_Pair], threshold: float) -> dict[str, Any]:
    return {
        "both_orders": _series(pairs, threshold, "both_orders"),
        "correct_at_A": _series(pairs, threshold, "correct_at_A"),
        "correct_at_B": _series(pairs, threshold, "correct_at_B"),
    }


def _series(pairs: Sequence[_Pair], threshold: float, count_set: str) -> dict[str, int]:
    chosen = []
    for pair in pairs:
        if count_set == "correct_at_A":
            chosen.append(pair.correct_at_a)
        elif count_set == "correct_at_B":
            chosen.append(pair.correct_at_b)
        else:
            chosen.extend((pair.correct_at_a, pair.correct_at_b))
    return {
        "before": len(chosen),
        "after_greedy_is_option": sum(_last_step(order, threshold) >= 1 for order in chosen),
        "after_option_probability": sum(_last_step(order, threshold) >= 2 for order in chosen),
        "after_greedy_correct": sum(_last_step(order, threshold) >= 3 for order in chosen),
        "after_both_orders_correct": sum(
            _last_step(order, threshold) >= 3 and _pair_passed(pairs, order.base_group_id)
            for order in chosen
        ),
    }


def _pair_passed(pairs: Sequence[_Pair], base_group_id: str) -> bool:
    return next(pair.passed for pair in pairs if pair.base_group_id == base_group_id)


def _failures(pairs: Sequence[_Pair], threshold: float) -> dict[str, Any]:
    """First failed step, counted on the same three sample sets."""
    body = {}
    for count_set in COUNT_SETS:
        orders = []
        for pair in pairs:
            if count_set == "correct_at_A":
                orders.append(pair.correct_at_a)
            elif count_set == "correct_at_B":
                orders.append(pair.correct_at_b)
            else:
                orders.extend((pair.correct_at_a, pair.correct_at_b))
        failed = {"1": 0, "2": 0, "3": 0, "4": 0}
        for order in orders:
            reached = _last_step(order, threshold)
            if reached < 3:
                failed[str(reached + 1)] += 1
            elif not _pair_passed(pairs, order.base_group_id):
                failed["4"] += 1
        body[count_set] = failed
    return body


def _units(pairs: Sequence[_Pair]) -> dict[str, int]:
    count = len(pairs)
    return {
        "base_questions": count,
        "intervention_instances": count,
        "paired_order_rows": count,
        "prompt_rows": count * 2,
    }


def _in_split(pairs: Sequence[_Pair], split: str) -> tuple[_Pair, ...]:
    return tuple(pair for pair in pairs if pair.split == split)


def _render(report: FilterReport) -> str:
    fields = report.payload()
    eligibility = fields["paired_eligibility"]
    lines = [
        "Fixed clean filter",
        f"Variant: {eligibility['variant']}",
        f"Threshold: greater than {eligibility['threshold']} on the normalized option probability",
        "1. The unrestricted greedy token is an option.",
        "2. That option's normalized probability is greater than the threshold.",
        "3. That option is the correct label.",
        "4. Both clean orders passed steps 1-3.",
        "",
        "Overall",
    ]
    _append_block(lines, fields["population_counts"]["overall"])
    for split in SPLITS:
        lines.extend(["", f"Split {split}"])
        _append_block(lines, fields["population_counts"]["by_split"][split])
    eligible = eligibility["eligible_base_group_ids"]
    lines.extend(["", f"Eligible base questions: {', '.join(eligible) if eligible else 'none'}"])
    return "\n".join(lines) + "\n"


def _append_block(lines: list[str], block: dict[str, Any]) -> None:
    for count_set in COUNT_SETS:
        lines.append(count_set)
        counts = block[count_set]
        for step in COUNT_STEPS:
            lines.append(f"  {step}: {counts[step]}")


def _write(directory: Path, report: FilterReport, text: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    artifact = identify(report)
    sidecar = json.dumps(artifact.sidecar(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_text(directory / "filter_report.json", sidecar)
    _atomic_text(directory / "filter_report.txt", text)
    if content_digest(json.loads(artifact.canonical_json)) != artifact.digest:
        raise RuntimeError("filter report digest does not match its canonical JSON")
    return artifact.digest


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _threshold(value: object) -> float:
    if type(value) is bool or type(value) not in (int, float):
        raise TypeError("min_correct_prob must be a probability")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("min_correct_prob must be between 0 and 1")
    return number


def _probability_pair(value: object, label: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{label} probabilities must be an A/B pair")
    parsed = []
    for item in value:
        if type(item) is bool or type(item) not in (int, float):
            raise TypeError(f"{label} probabilities must be real numbers")
        number = float(item)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{label} probabilities must be between 0 and 1")
        parsed.append(number)
    return parsed[0], parsed[1]
