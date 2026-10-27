"""Order-paired behavioral counts.

Each table is named ``{scoring}__{population}__{split}``. ``raw_token``
compares unrestricted greedy labels and separates invalid-label transitions
from A/B switches. ``canonical_label`` compares canonical argmax labels.
Changed-answer counts use the base questions whose clean answers are correct
in both orders, and each context instance is compared with that clean pair
on its own. Clean instances therefore have zero deviation. A zero denominator
is an unavailable rate with an explanatory count, not an accuracy of zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

SCORINGS = ("raw_token", "canonical_label")
POPULATIONS = ("all_valid", "raw_retained", "canonical_retained")
SPLITS = ("train", "val", "test")
ORDERS = ("correct_at_A", "correct_at_B")


@dataclass(frozen=True)
class BehaviorRow:
    """One prompt row. A raw label of ``None`` is an invalid greedy token."""

    base_group_id: str
    split: str
    variant_id: str
    instance_id: str
    order_id: str
    canonical_label: str
    raw_token_label: str | None
    raw_token_id: int

    def __post_init__(self) -> None:
        for label in ("base_group_id", "variant_id", "instance_id"):
            if type(getattr(self, label)) is not str or getattr(self, label) == "":
                raise ValueError(f"{label} must be a non-empty string")
        if self.split not in SPLITS:
            raise ValueError("split must be train, val, or test")
        if self.order_id not in ORDERS:
            raise ValueError("order_id must be correct_at_A or correct_at_B")
        if self.canonical_label not in ("A", "B"):
            raise ValueError("canonical_label must be A or B")
        if self.raw_token_label not in ("A", "B", None):
            raise ValueError("raw_token_label must be A, B, or None")
        if type(self.raw_token_id) is not int or self.raw_token_id < 0:
            raise ValueError("raw_token_id must be a non-negative integer")


@dataclass(frozen=True)
class CountRate:
    """A count divided by a named denominator. ``value`` is absent at zero."""

    count: int
    denominator: int
    value: float | None
    unavailable_reason: str | None


@dataclass(frozen=True)
class OrderDeviation:
    changed: CountRate
    valid_switches: CountRate
    invalid_transitions: CountRate


@dataclass(frozen=True)
class InstanceDeviation:
    variant_id: str
    instance_id: str
    correct_at_A: OrderDeviation
    correct_at_B: OrderDeviation
    changed_both_orders: CountRate
    changed_either_order: CountRate


@dataclass(frozen=True)
class BehaviorTable:
    name: str
    scoring: str
    population: str
    split: str
    base_questions: int
    clean_correct_at_A: CountRate
    clean_correct_at_B: CountRate
    correct_both_orders: CountRate
    instances: tuple[InstanceDeviation, ...]


def behavior_tables(
    rows: Sequence[BehaviorRow],
    *,
    raw_retained: Sequence[str],
    canonical_retained: Sequence[str],
    clean_variant: str = "clean",
) -> tuple[BehaviorTable, ...]:
    """Build raw-token and canonical-label tables for every split and population."""
    if type(clean_variant) is not str or clean_variant == "":
        raise ValueError("clean_variant must be a non-empty string")
    grouped = _group(rows, clean_variant)
    populations = {
        "all_valid": tuple(grouped),
        "raw_retained": _population(raw_retained, grouped, "raw_retained"),
        "canonical_retained": _population(canonical_retained, grouped, "canonical_retained"),
    }
    tables = []
    for scoring in SCORINGS:
        for population in POPULATIONS:
            chosen = {group_id: grouped[group_id] for group_id in populations[population]}
            for split in SPLITS:
                tables.append(_table(scoring, population, split, chosen, clean_variant))
    return tuple(tables)


def behavior_document(tables: Sequence[BehaviorTable]) -> dict[str, Any]:
    """JSON-safe tables. Unavailable rates stay null rather than becoming zero."""
    return {"tables": [_table_document(table) for table in tables]}


def _group(rows: Sequence[BehaviorRow], clean_variant: str) -> dict[str, dict[tuple[str, str], dict[str, BehaviorRow]]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of BehaviorRow values")
    grouped: dict[str, dict[tuple[str, str], dict[str, BehaviorRow]]] = {}
    splits: dict[str, str] = {}
    for row in rows:
        if type(row) is not BehaviorRow:
            raise TypeError("rows must be a sequence of BehaviorRow values")
        previous = splits.setdefault(row.base_group_id, row.split)
        if previous != row.split:
            raise ValueError(f"base group {row.base_group_id} appears in more than one split")
        instances = grouped.setdefault(row.base_group_id, {})
        orders = instances.setdefault((row.variant_id, row.instance_id), {})
        if row.order_id in orders:
            raise ValueError(
                f"duplicate row for {row.base_group_id} {row.variant_id} "
                f"{row.instance_id} {row.order_id}"
            )
        orders[row.order_id] = row
    if not grouped:
        return {}
    for group_id, instances in grouped.items():
        clean = [key for key in instances if key[0] == clean_variant]
        if len(clean) != 1:
            raise ValueError(f"base group {group_id} must have one {clean_variant} instance")
        for key, orders in instances.items():
            if set(orders) != set(ORDERS):
                raise ValueError(f"base group {group_id} instance {key[1]} is missing an order")
    return grouped


def _population(
    group_ids: Sequence[str],
    grouped: dict[str, dict[tuple[str, str], dict[str, BehaviorRow]]],
    label: str,
) -> tuple[str, ...]:
    if isinstance(group_ids, (str, bytes)) or not isinstance(group_ids, Sequence):
        raise TypeError(f"{label} must be a sequence of base-group ids")
    parsed = []
    for group_id in group_ids:
        if type(group_id) is not str or group_id == "":
            raise ValueError(f"{label} ids must be non-empty strings")
        if group_id not in grouped:
            raise ValueError(f"{label} contains an unknown base group {group_id}")
        parsed.append(group_id)
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"duplicate base group in {label}")
    return tuple(sorted(parsed))


def _table(
    scoring: str,
    population: str,
    split: str,
    grouped: dict[str, dict[tuple[str, str], dict[str, BehaviorRow]]],
    clean_variant: str,
) -> BehaviorTable:
    group_ids = [group_id for group_id in sorted(grouped) if _split_of(grouped[group_id]) == split]
    questions = [grouped[group_id] for group_id in group_ids]
    denominator = len(questions)
    correct_a = {
        group_id for group_id in group_ids
        if _correct(_clean_row(grouped[group_id], clean_variant, "correct_at_A"), scoring)
    }
    correct_b = {
        group_id for group_id in group_ids
        if _correct(_clean_row(grouped[group_id], clean_variant, "correct_at_B"), scoring)
    }
    # Match by group id: comparing question dicts is slow and scans the list per item.
    both = [grouped[group_id] for group_id in group_ids if group_id in correct_a and group_id in correct_b]
    both_count = len(both)
    if both_count > min(len(correct_a), len(correct_b)) or len(correct_a) > denominator:
        raise ValueError("paired counts violate set-intersection bounds")
    instances = _instance_keys(questions)
    table = BehaviorTable(
        name=f"{scoring}__{population}__{split}",
        scoring=scoring,
        population=population,
        split=split,
        base_questions=denominator,
        clean_correct_at_A=_rate(len(correct_a), denominator, "0 base questions in this split and population"),
        clean_correct_at_B=_rate(len(correct_b), denominator, "0 base questions in this split and population"),
        correct_both_orders=_rate(both_count, denominator, "0 base questions in this split and population"),
        instances=tuple(
            _instance(key, both, both_count, scoring, clean_variant) for key in instances
        ),
    )
    _check_bounds(table)
    return table


def _instance(
    key: tuple[str, str],
    both: Sequence[dict[tuple[str, str], dict[str, BehaviorRow]]],
    denominator: int,
    scoring: str,
    clean_variant: str,
) -> InstanceDeviation:
    covered = [question for question in both if key in question]
    denominator = len(covered)
    changed_a: list[str] = []
    changed_b: list[str] = []
    valid_a = invalid_a = valid_b = invalid_b = 0
    for question in covered:
        group_id = _clean_row(question, clean_variant, "correct_at_A").base_group_id
        kind_a = _change(question, key, "correct_at_A", scoring, clean_variant)
        kind_b = _change(question, key, "correct_at_B", scoring, clean_variant)
        if kind_a != "none":
            changed_a.append(group_id)
            valid_a += kind_a == "valid_switch"
            invalid_a += kind_a == "invalid_transition"
        if kind_b != "none":
            changed_b.append(group_id)
            valid_b += kind_b == "valid_switch"
            invalid_b += kind_b == "invalid_transition"
    both_ids = set(changed_a) & set(changed_b)
    either_ids = set(changed_a) | set(changed_b)
    reason = "0 base questions correct in both clean orders"
    return InstanceDeviation(
        variant_id=key[0],
        instance_id=key[1],
        correct_at_A=_deviation(len(changed_a), valid_a, invalid_a, denominator, reason),
        correct_at_B=_deviation(len(changed_b), valid_b, invalid_b, denominator, reason),
        changed_both_orders=_rate(len(both_ids), denominator, reason),
        changed_either_order=_rate(len(either_ids), denominator, reason),
    )


def _deviation(changed: int, valid: int, invalid: int, denominator: int, reason: str) -> OrderDeviation:
    if valid + invalid != changed:
        raise ValueError("paired counts violate set-intersection bounds")
    return OrderDeviation(
        changed=_rate(changed, denominator, reason),
        valid_switches=_rate(valid, denominator, reason),
        invalid_transitions=_rate(invalid, denominator, reason),
    )


def _change(
    question: dict[tuple[str, str], dict[str, BehaviorRow]],
    key: tuple[str, str],
    order_id: str,
    scoring: str,
    clean_variant: str,
) -> str:
    clean = _clean_row(question, clean_variant, order_id)
    other = question[key][order_id]
    if scoring == "canonical_label":
        return "none" if clean.canonical_label == other.canonical_label else "valid_switch"
    if clean.raw_token_id == other.raw_token_id:
        return "none"
    if clean.raw_token_label in ("A", "B") and other.raw_token_label in ("A", "B"):
        return "valid_switch"
    return "invalid_transition"


def _correct(row: BehaviorRow, scoring: str) -> bool:
    expected = "A" if row.order_id == "correct_at_A" else "B"
    if scoring == "canonical_label":
        return row.canonical_label == expected
    return row.raw_token_label == expected


def _clean_row(
    question: dict[tuple[str, str], dict[str, BehaviorRow]],
    clean_variant: str,
    order_id: str,
) -> BehaviorRow:
    matches = [key for key in question if key[0] == clean_variant]
    return question[matches[0]][order_id]


def _split_of(question: dict[tuple[str, str], dict[str, BehaviorRow]]) -> str:
    row = next(iter(next(iter(question.values())).values()))
    return row.split


def _instance_keys(
    questions: Sequence[dict[tuple[str, str], dict[str, BehaviorRow]]],
) -> tuple[tuple[str, str], ...]:
    if not questions:
        return ()
    return tuple(sorted({key for question in questions for key in question}))


def _check_bounds(table: BehaviorTable) -> None:
    both = table.correct_both_orders.count
    if both > min(table.clean_correct_at_A.count, table.clean_correct_at_B.count):
        raise ValueError("paired counts violate set-intersection bounds")
    if table.clean_correct_at_A.count > table.base_questions or table.clean_correct_at_B.count > table.base_questions:
        raise ValueError("paired counts violate set-intersection bounds")
    for instance in table.instances:
        changed_a = instance.correct_at_A.changed.count
        changed_b = instance.correct_at_B.changed.count
        changed_both = instance.changed_both_orders.count
        changed_either = instance.changed_either_order.count
        if changed_a > both or changed_b > both or changed_both > min(changed_a, changed_b):
            raise ValueError("paired counts violate set-intersection bounds")
        if changed_either != changed_a + changed_b - changed_both:
            raise ValueError("paired counts violate set-intersection bounds")
        if changed_either > both:
            raise ValueError("paired counts violate set-intersection bounds")


def _rate(count: int, denominator: int, reason: str) -> CountRate:
    if denominator == 0:
        if count != 0:
            raise ValueError("nonzero count with a zero denominator")
        return CountRate(0, 0, None, f"zero denominator: {reason}")
    return CountRate(count, denominator, count / denominator, None)


def _table_document(table: BehaviorTable) -> dict[str, Any]:
    return {
        "name": table.name,
        "scoring": table.scoring,
        "population": table.population,
        "split": table.split,
        "base_questions": table.base_questions,
        "clean_correct_at_A": _rate_document(table.clean_correct_at_A),
        "clean_correct_at_B": _rate_document(table.clean_correct_at_B),
        "correct_both_orders": _rate_document(table.correct_both_orders),
        "instances": [
            {
                "variant_id": item.variant_id,
                "instance_id": item.instance_id,
                "correct_at_A": _deviation_document(item.correct_at_A),
                "correct_at_B": _deviation_document(item.correct_at_B),
                "changed_both_orders": _rate_document(item.changed_both_orders),
                "changed_either_order": _rate_document(item.changed_either_order),
            }
            for item in table.instances
        ],
    }


def _deviation_document(deviation: OrderDeviation) -> dict[str, Any]:
    return {
        "changed": _rate_document(deviation.changed),
        "valid_switches": _rate_document(deviation.valid_switches),
        "invalid_transitions": _rate_document(deviation.invalid_transitions),
    }


def _rate_document(rate: CountRate) -> dict[str, Any]:
    return {
        "count": rate.count,
        "denominator": rate.denominator,
        "value": rate.value,
        "unavailable_reason": rate.unavailable_reason,
    }
