"""Fixed four-step filter for clean orders.

A row passes only in this order: the unrestricted greedy token is an option,
that option's normalized A/B probability is greater than the threshold, and
that option is the correct label. A base question is eligible when both of
its clean orders pass. The report counts samples before the filters and
after each step, for both orders together and for each correct-answer
position. There is no raw-versus-canonical choice.

The optional variant gate applies steps 1 and 2 to each intervened variant,
starting from the questions that passed the clean filter, and then requires
both orders to pass. It has no correct-label step.

The misreport report counts, for each intervened variant, the rows whose
normalized A/B argmax differs from the clean row of the same question and
order. It counts before any filter and after every clean and variant step.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
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
VARIANT_COUNT_STEPS = (
    "before",
    "after_greedy_is_option",
    "after_option_probability",
    "after_both_orders_confident",
)
MISREPORT_STEPS = (
    "before",
    "after_clean_greedy_is_option",
    "after_clean_option_probability",
    "after_clean_greedy_correct",
    "after_clean_both_orders_correct",
)
MISREPORT_VARIANT_STEPS = (
    "after_variant_greedy_is_option",
    "after_variant_option_probability",
    "after_variant_both_orders_confident",
)
MISREPORT_FINAL_STEP = "after_confident_on_every_variant"


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


@dataclass(frozen=True)
class VariantFilterOutcome:
    """Questions that passed each variant, and the final eligible set.

    Without ``require_confident_on_every_variant`` the final set is the clean-filtered pool unchanged.
    """

    eligible_base_group_ids: tuple[str, ...]
    passed_by_variant: Mapping[str, tuple[str, ...]]
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
    report = _report(pairs, threshold=threshold, variant_id=variant_id)
    digest = _write(directory, report, _render(report))
    eligible = tuple(pair.base_group_id for pair in pairs if pair.passed)
    return FilterOutcome(
        eligible_base_group_ids=eligible,
        report_digest=digest,
        min_correct_prob=threshold,
    )


def apply_variant_gate(
    orders_by_variant: Mapping[str, Sequence[CleanOrder]],
    pool_ids: Sequence[str],
    *,
    output_directory: str | Path,
    min_correct_prob: float = 0.8,
    require_confident_on_every_variant: bool = False,
) -> VariantFilterOutcome:
    """Write three-step counts per variant over the clean-filtered pool.

    Orders outside ``pool_ids`` are ignored. Every pool question must have
    both orders in every variant. With ``require_confident_on_every_variant`` the final set keeps
    only questions that passed on every variant.
    """
    threshold = _threshold(min_correct_prob)
    if not isinstance(orders_by_variant, Mapping) or not orders_by_variant:
        raise ValueError("orders_by_variant must be a non-empty mapping")
    if type(require_confident_on_every_variant) is not bool:
        raise TypeError("require_confident_on_every_variant must be a boolean")
    pool = tuple(sorted(set(pool_ids)))
    allowed = set(pool)
    pairs_by_variant: dict[str, tuple[_Pair, ...]] = {}
    for variant_id, orders in orders_by_variant.items():
        if type(variant_id) is not str or variant_id == "":
            raise ValueError("variant ids must be non-empty strings")
        if isinstance(orders, (str, bytes)) or not isinstance(orders, Sequence):
            raise TypeError("orders must be a sequence of CleanOrder values")
        chosen = [order for order in orders if order.base_group_id in allowed]
        pairs = _pairs(chosen, threshold, require_correct=False, label=variant_id)
        missing = allowed - {pair.base_group_id for pair in pairs}
        if missing:
            raise ValueError(
                f"variant {variant_id} is missing pool questions: {', '.join(sorted(missing))}"
            )
        pairs_by_variant[variant_id] = pairs
    passed = {
        variant_id: tuple(pair.base_group_id for pair in pairs if pair.passed)
        for variant_id, pairs in pairs_by_variant.items()
    }
    if require_confident_on_every_variant:
        survivors = set(pool)
        for ids in passed.values():
            survivors &= set(ids)
        eligible = tuple(sorted(survivors))
    else:
        eligible = pool
    splits = {pair.base_group_id: pair.split for pairs in pairs_by_variant.values() for pair in pairs}
    directory = Path(output_directory)
    report = _variant_report(
        pairs_by_variant,
        pool=pool,
        eligible=eligible,
        splits=splits,
        threshold=threshold,
        require_confident_on_every_variant=require_confident_on_every_variant,
    )
    digest = _write(directory, report, _render_variants(report), stem="variant_filter_report")
    return VariantFilterOutcome(
        eligible_base_group_ids=eligible,
        passed_by_variant=passed,
        report_digest=digest,
        min_correct_prob=threshold,
    )


def write_misreport_report(
    clean_orders: Sequence[CleanOrder],
    orders_by_variant: Mapping[str, Sequence[CleanOrder]],
    *,
    output_directory: str | Path,
    min_correct_prob: float = 0.8,
    variant_steps: bool = False,
    require_confident_on_every_variant: bool = False,
) -> dict[str, Any]:
    """Write misreport counts per variant before and after each filter step.

    A misreport is a row whose normalized A/B argmax differs from the clean
    row's argmax for the same question and order. Each step keeps the rows
    that pass it and every earlier step. Clean steps use the clean row, and
    variant steps use the variant row. ``require_confident_on_every_variant`` adds the cross-variant
    step and needs ``variant_steps``.
    """
    threshold = _threshold(min_correct_prob)
    if require_confident_on_every_variant and not variant_steps:
        raise ValueError("require_confident_on_every_variant needs variant_steps")
    clean = {pair.base_group_id: pair for pair in _pairs(clean_orders, threshold)}
    variants: dict[str, dict[str, _Pair]] = {}
    for variant_id, orders in orders_by_variant.items():
        if type(variant_id) is not str or variant_id == "":
            raise ValueError("variant ids must be non-empty strings")
        pairs = {
            pair.base_group_id: pair
            for pair in _pairs(orders, threshold, require_correct=False, label=variant_id)
        }
        if set(pairs) != set(clean):
            raise ValueError(f"variant {variant_id} does not cover the same questions as clean")
        variants[variant_id] = pairs
    steps = list(MISREPORT_STEPS)
    if variant_steps:
        steps.extend(MISREPORT_VARIANT_STEPS)
    if require_confident_on_every_variant:
        steps.append(MISREPORT_FINAL_STEP)
    all_passed = {
        group for group in clean
        if clean[group].passed and all(pairs[group].passed for pairs in variants.values())
    }
    counts: dict[str, Any] = {}
    for variant_id, pairs in variants.items():
        rows = []
        for group, pair in clean.items():
            for order_id in ORDERS:
                rows.append((
                    pair.split,
                    order_id,
                    _misreport_reached(order_id, pair, pairs[group], threshold, group in all_passed),
                    _greedy_option(_order_of(pairs[group], order_id))
                    != _greedy_option(_order_of(pair, order_id)),
                ))
        counts[variant_id] = {
            "overall": _misreport_block(rows, steps),
            "by_split": {
                split: _misreport_block([row for row in rows if row[0] == split], steps)
                for split in SPLITS
            },
        }
    report = {
        "definition": "variant normalized A/B argmax differs from the clean argmax for the same question and order",
        "threshold": threshold,
        "variants": list(variants),
        "steps": steps,
        "counts": counts,
    }
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        directory / "misreport_report.json",
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )
    _atomic_text(directory / "misreport_report.txt", _render_misreports(report))
    return report


def _order_of(pair: _Pair, order_id: str) -> CleanOrder:
    return pair.correct_at_a if order_id == "correct_at_A" else pair.correct_at_b


def _misreport_reached(
    order_id: str, clean: _Pair, variant: _Pair, threshold: float, all_passed: bool
) -> int:
    """Index of the last step passed, counting ``before`` as step 0."""
    clean_step = _last_step(_order_of(clean, order_id), threshold)
    if clean_step < 3:
        return clean_step
    if not clean.passed:
        return 3
    variant_step = _last_step(_order_of(variant, order_id), threshold)
    if variant_step < 2:
        return 4 + variant_step
    if not variant.passed:
        return 6
    return 8 if all_passed else 7


def _misreport_block(rows: Sequence[tuple[str, str, int, bool]], steps: Sequence[str]) -> dict[str, Any]:
    block = {}
    for count_set in COUNT_SETS:
        chosen = [row for row in rows if count_set == "both_orders" or row[1] == count_set]
        block[count_set] = {
            step: {
                "rows": sum(row[2] >= index for row in chosen),
                "misreports": sum(row[2] >= index and row[3] for row in chosen),
            }
            for index, step in enumerate(steps)
        }
    return block


def _render_misreports(report: Mapping[str, Any]) -> str:
    lines = [
        "Misreports per intervened variant",
        "A misreport is a row whose normalized A/B argmax differs from the clean row",
        "of the same question and order. Each step keeps rows that pass it and every",
        f"earlier step. Threshold: greater than {report['threshold']}.",
    ]
    for variant_id in report["variants"]:
        scopes = [("Overall", report["counts"][variant_id]["overall"])] + [
            (f"Split {split}", report["counts"][variant_id]["by_split"][split]) for split in SPLITS
        ]
        for scope, block in scopes:
            lines.extend(["", f"Variant: {variant_id} | {scope}"])
            for count_set in COUNT_SETS:
                lines.append(count_set)
                for step in report["steps"]:
                    cell = block[count_set][step]
                    rate = f"{cell['misreports'] / cell['rows']:.3f}" if cell["rows"] else "n/a"
                    lines.append(f"  {step}: {cell['misreports']}/{cell['rows']} ({rate})")
    return "\n".join(lines) + "\n"


def _pairs(
    orders: Sequence[CleanOrder],
    threshold: float,
    *,
    require_correct: bool = True,
    label: str = "clean",
) -> tuple[_Pair, ...]:
    if isinstance(orders, (str, bytes)) or not isinstance(orders, Sequence):
        raise TypeError("orders must be a sequence of CleanOrder values")
    grouped: dict[str, dict[str, CleanOrder]] = {}
    for order in orders:
        if type(order) is not CleanOrder:
            raise TypeError("orders must be a sequence of CleanOrder values")
        found = grouped.setdefault(order.base_group_id, {})
        if order.order_id in found:
            raise ValueError(f"duplicate {label} order for {order.base_group_id} {order.order_id}")
        found[order.order_id] = order
    needed = 3 if require_correct else 2
    pairs: list[_Pair] = []
    for base_group_id in sorted(grouped):
        found = grouped[base_group_id]
        if set(found) != set(ORDERS):
            missing = [name for name in ORDERS if name not in found]
            raise ValueError(
                f"base group {base_group_id} is missing {label} orders: {', '.join(missing)}"
            )
        first, second = found["correct_at_A"], found["correct_at_B"]
        if first.split != second.split:
            raise ValueError(f"base group {base_group_id} has {label} orders in different splits")
        passed = _last_step(first, threshold) >= needed and _last_step(second, threshold) >= needed
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


def _count_set(pairs: Sequence[_Pair], count_set: str) -> list[CleanOrder]:
    chosen: list[CleanOrder] = []
    for pair in pairs:
        if count_set == "correct_at_A":
            chosen.append(pair.correct_at_a)
        elif count_set == "correct_at_B":
            chosen.append(pair.correct_at_b)
        else:
            chosen.extend((pair.correct_at_a, pair.correct_at_b))
    return chosen


def _variant_report(
    pairs_by_variant: Mapping[str, Sequence[_Pair]],
    *,
    pool: Sequence[str],
    eligible: Sequence[str],
    splits: Mapping[str, str],
    threshold: float,
    require_confident_on_every_variant: bool,
) -> FilterReport:
    population: dict[str, Any] = {
        variant_id: {
            "overall": _variant_block(pairs, threshold),
            "by_split": {
                split: _variant_block(_in_split(pairs, split), threshold) for split in SPLITS
            },
        }
        for variant_id, pairs in pairs_by_variant.items()
    }
    if require_confident_on_every_variant:
        population["final"] = {
            "overall": _final_units(eligible),
            "by_split": {
                split: _final_units([group for group in eligible if splits.get(group) == split])
                for split in SPLITS
            },
        }
    return FilterReport(
        paired_eligibility={
            "variants": list(pairs_by_variant),
            "threshold": threshold,
            "comparison": "greater_than",
            "require_both_orders": True,
            "require_correct_label": False,
            "require_confident_on_every_variant": require_confident_on_every_variant,
            "pool_size": len(pool),
            "eligible_base_group_ids": list(eligible),
        },
        order_failures={
            variant_id: _variant_failures(pairs, threshold)
            for variant_id, pairs in pairs_by_variant.items()
        },
        population_counts=population,
        split_denominators={
            split: _final_units([group for group in pool if splits.get(group) == split])
            for split in SPLITS
        },
        canonical_only_diagnostics={
            "policy": "variant_three_step",
            "pool": "clean_filter_eligible",
            "steps": list(VARIANT_COUNT_STEPS[1:]),
            "option_probability": "normalized_ab",
        },
    )


def _variant_block(pairs: Sequence[_Pair], threshold: float) -> dict[str, Any]:
    block = {}
    for count_set in COUNT_SETS:
        chosen = _count_set(pairs, count_set)
        passed = {pair.base_group_id for pair in pairs if pair.passed}
        block[count_set] = {
            "before": len(chosen),
            "after_greedy_is_option": sum(_last_step(order, threshold) >= 1 for order in chosen),
            "after_option_probability": sum(_last_step(order, threshold) >= 2 for order in chosen),
            "after_both_orders_confident": sum(order.base_group_id in passed for order in chosen),
        }
    return block


def _variant_failures(pairs: Sequence[_Pair], threshold: float) -> dict[str, Any]:
    """First failed step per order. Step 3 counts orders whose pair partner failed."""
    passed = {pair.base_group_id for pair in pairs if pair.passed}
    body = {}
    for count_set in COUNT_SETS:
        failed = {"1": 0, "2": 0, "3": 0}
        for order in _count_set(pairs, count_set):
            reached = _last_step(order, threshold)
            if reached < 2:
                failed[str(reached + 1)] += 1
            elif order.base_group_id not in passed:
                failed["3"] += 1
        body[count_set] = failed
    return body


def _final_units(group_ids: Sequence[str]) -> dict[str, int]:
    count = len(group_ids)
    return {"base_questions": count, "prompt_rows_per_variant": count * 2}


def _render_variants(report: FilterReport) -> str:
    fields = report.payload()
    eligibility = fields["paired_eligibility"]
    pool = eligibility["pool_size"]
    lines = [
        "Intervened-variant filter",
        f"Pool: base questions that passed the clean filter ({pool} questions, {pool * 2} rows per variant)",
        f"Threshold: greater than {eligibility['threshold']} on the normalized option probability",
        "1. The unrestricted greedy token is an option.",
        "2. That option's normalized probability is greater than the threshold.",
        "3. Both orders passed steps 1-2.",
        "The option does not need to be the correct label.",
    ]
    counts = fields["population_counts"]
    for variant_id in eligibility["variants"]:
        lines.extend(["", f"Variant: {variant_id}", "Overall"])
        _append_block(lines, counts[variant_id]["overall"], VARIANT_COUNT_STEPS)
        for split in SPLITS:
            lines.extend(["", f"Split {split}"])
            _append_block(lines, counts[variant_id]["by_split"][split], VARIANT_COUNT_STEPS)
    if eligibility["require_confident_on_every_variant"]:
        final = counts["final"]
        lines.extend([
            "",
            "Final (confident on every variant)",
            f"Kept: questions that passed step 3 on {', '.join(eligibility['variants'])}",
            "Overall",
        ])
        lines.extend(f"  {key}: {value}" for key, value in final["overall"].items())
        for split in SPLITS:
            lines.append(f"Split {split}")
            lines.extend(f"  {key}: {value}" for key, value in final["by_split"][split].items())
    else:
        lines.extend(["", "Final filter (confident on every variant): not applied"])
    return "\n".join(lines) + "\n"


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


def _append_block(
    lines: list[str], block: dict[str, Any], steps: Sequence[str] = COUNT_STEPS
) -> None:
    for count_set in COUNT_SETS:
        lines.append(count_set)
        counts = block[count_set]
        for step in steps:
            lines.append(f"  {step}: {counts[step]}")


def _write(directory: Path, report: FilterReport, text: str, *, stem: str = "filter_report") -> str:
    directory.mkdir(parents=True, exist_ok=True)
    artifact = identify(report)
    # Paths sit beside the record, not in it, so moving storage keeps the digest.
    paths = {"directory": str(directory), "json": f"{stem}.json", "text": f"{stem}.txt"}
    sidecar = json.dumps({**artifact.sidecar(), "paths": paths}, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_text(directory / f"{stem}.json", sidecar)
    _atomic_text(directory / f"{stem}.txt", text)
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
