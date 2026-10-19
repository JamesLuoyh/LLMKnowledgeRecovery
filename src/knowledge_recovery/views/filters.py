"""Paired clean-answer filter and the probability-basis gate.

A base question is eligible under a basis only when both clean orders assign
at least ``min_correct_prob`` to the correct label. Raw and canonical masks
are both written. Conditional A/B probability is at least the raw probability
of the same label, so a raw-only order means the stored scores are not a
conditional distribution. That failure is reported and is not used to choose
a basis.

The diagnostic record contains no selected basis. A later run inherits a
recorded decision only when the decision scope matches and the requested
basis does not contradict it. Split seeds, fractions, the threshold, probe
methods, and added evaluation contexts are outside that scope.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge_recovery.errors import PROBABILITY_BASES, FilterPolicyDecisionRequired, KnowledgeRecoveryError
from knowledge_recovery.provenance import content_digest, identify
from knowledge_recovery.records import DecisionRecord, FilterReport

SPLITS = ("train", "val", "test")
ORDERS = ("correct_at_A", "correct_at_B")
SCOPE_FIELDS = (
    "dataset_adapter",
    "source_snapshot",
    "model_content_signature",
    "prompt_contract_digest",
    "canonical_answers",
    "require_both_orders",
)


class ProbabilityConsistencyError(KnowledgeRecoveryError):
    """Raw eligibility is not contained in canonical eligibility."""

    def __init__(self, *, report_id: str, raw_only_orders: Sequence[Mapping[str, str]]) -> None:
        self.report_id = report_id
        self.raw_only_orders = tuple(raw_only_orders)
        shown = ", ".join(
            f"{item['base_group_id']}/{item['order_id']}" for item in self.raw_only_orders
        )
        super().__init__(
            "Probability consistency failed: raw-only retention is not zero.\n"
            f"Raw-only orders: {shown}\n"
            f"Report: {report_id}\n"
            "A conditional A/B probability is at least the raw probability of "
            "that label. The probability basis was not inferred."
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


@dataclass(frozen=True)
class AppliedDecision:
    """The recorded basis, kept beside the current report and the original decision."""

    probability_basis: str
    original_decision_id: str
    original_report_digest: str
    current_report_digest: str
    min_correct_prob: float
    scope: Mapping[str, Any]


def record_decision(
    *,
    probability_basis: str,
    filter_report_digest: str,
    scope: Mapping[str, Any],
    timestamp: str,
    user_instruction: str,
) -> DecisionRecord:
    """Record an explicit basis. Unknown scope fields, including seeds, are dropped."""
    if probability_basis not in PROBABILITY_BASES:
        raise ValueError("probability_basis must be raw or canonical")
    return DecisionRecord(
        probability_basis=probability_basis,
        filter_report_digest=filter_report_digest,
        scope=_scope(scope),
        timestamp=timestamp,
        user_instruction=user_instruction,
    )


def apply_filter_gate(
    orders: Sequence[CleanOrder],
    *,
    output_directory: str | Path,
    scope: Mapping[str, Any],
    min_correct_prob: float = 0.8,
    variant_id: str = "clean",
    decision: DecisionRecord | None = None,
    requested_basis: str | None = None,
    reset_decision: bool = False,
) -> AppliedDecision:
    """Write both masks and the readable report, then apply or reopen the gate.

    ``requested_basis`` without a matching decision record does not select a
    basis, including when one basis retains more questions than the other.
    """
    threshold = _threshold(min_correct_prob)
    if type(variant_id) is not str or variant_id == "":
        raise ValueError("variant_id must be a non-empty string")
    if requested_basis is not None and requested_basis not in PROBABILITY_BASES:
        raise ValueError("requested_basis must be raw or canonical")
    if decision is not None and type(decision) is not DecisionRecord:
        raise TypeError("decision must be a DecisionRecord")
    current_scope = _scope(scope)
    pairs = _pairs(orders)
    directory = Path(output_directory)
    report = _report(pairs, threshold=threshold, variant_id=variant_id, directory=directory)
    outcome = _outcome(
        decision,
        requested_basis=requested_basis,
        reset_decision=reset_decision,
        current_scope=current_scope,
        raw_only=_raw_only(pairs, threshold),
    )
    digest = _write(directory, report, _render(report, outcome.note))
    if outcome.kind == "inconsistent":
        raise ProbabilityConsistencyError(report_id=digest, raw_only_orders=outcome.raw_only)
    if outcome.kind == "unset":
        raise FilterPolicyDecisionRequired(report_id=digest)
    if outcome.kind == "contradiction":
        raise FilterPolicyDecisionRequired(
            report_id=digest,
            requested_basis=outcome.requested_basis,
            recorded_basis=outcome.recorded_basis,
        )
    applied = AppliedDecision(
        probability_basis=outcome.basis,
        original_decision_id=identify(outcome.decision).digest,
        original_report_digest=outcome.decision.filter_report_digest,
        current_report_digest=digest,
        min_correct_prob=threshold,
        scope=current_scope,
    )
    _atomic_text(
        directory / "applied_decision.json",
        json.dumps(_applied_document(applied), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return applied


@dataclass(frozen=True)
class _Outcome:
    kind: str
    note: str
    raw_only: tuple[Mapping[str, str], ...] = ()
    basis: str | None = None
    decision: DecisionRecord | None = None
    requested_basis: str | None = None
    recorded_basis: str | None = None


def _outcome(
    decision: DecisionRecord | None,
    *,
    requested_basis: str | None,
    reset_decision: bool,
    current_scope: Mapping[str, Any],
    raw_only: Sequence[Mapping[str, str]],
) -> _Outcome:
    if raw_only:
        return _Outcome(
            "inconsistent",
            "Raw-only retention is not zero. No probability basis was selected.",
            tuple(raw_only),
        )
    if decision is None or reset_decision:
        note = "The recorded decision was reset." if reset_decision else (
            "No decision record applies."
        )
        if requested_basis is not None:
            note += f" Requested basis {requested_basis} was not applied."
        note += " Probe training stops at this gate."
        return _Outcome("unset", note)
    if _scope_or_none(decision.scope) != current_scope:
        field = _first_difference(decision.scope, current_scope)
        return _Outcome(
            "unset",
            f"Recorded decision does not apply because {field} differs. "
            "The gate is open and no basis was selected.",
        )
    if requested_basis is not None and requested_basis != decision.probability_basis:
        return _Outcome(
            "contradiction",
            f"Requested basis {requested_basis} contradicts recorded basis "
            f"{decision.probability_basis}.",
            basis=decision.probability_basis,
            decision=decision,
            requested_basis=requested_basis,
            recorded_basis=decision.probability_basis,
        )
    return _Outcome(
        "inherit",
        f"Inherited recorded basis {decision.probability_basis}.",
        basis=decision.probability_basis,
        decision=decision,
    )


def _pairs(orders: Sequence[CleanOrder]) -> tuple[_Pair, ...]:
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
        pairs.append(_Pair(base_group_id, first.split, first, second))
    return tuple(pairs)


def _report(pairs: Sequence[_Pair], *, threshold: float, variant_id: str, directory: Path) -> FilterReport:
    eligibility = {
        "variant": variant_id,
        "threshold": threshold,
        "require_both_orders": True,
        "raw": _mask(pairs, "raw", threshold),
        "canonical": _mask(pairs, "canonical", threshold),
    }
    return FilterReport(
        paired_eligibility=eligibility,
        order_failures=_failures(pairs, threshold),
        population_counts=_populations(pairs, threshold),
        split_denominators={split: _units(_in_split(pairs, split)) for split in SPLITS},
        canonical_only_diagnostics=_diagnostics(pairs, threshold),
        paths={
            "directory": str(directory),
            "json": "filter_report.json",
            "text": "filter_report.txt",
        },
    )


def _mask(pairs: Sequence[_Pair], basis: str, threshold: float) -> list[dict[str, Any]]:
    return [
        {
            "base_group_id": pair.base_group_id,
            "split": pair.split,
            "eligible": _pair_passes(pair, basis, threshold),
        }
        for pair in pairs
    ]


def _pair_passes(pair: _Pair, basis: str, threshold: float) -> bool:
    return all(_order_passes(order, basis, threshold) for order in (pair.correct_at_a, pair.correct_at_b))


def _order_passes(order: CleanOrder, basis: str, threshold: float) -> bool:
    probabilities = order.raw_ab_probabilities if basis == "raw" else order.canonical_ab_probabilities
    index = 0 if order.order_id == "correct_at_A" else 1
    return probabilities[index] >= threshold


def _failures(pairs: Sequence[_Pair], threshold: float) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for basis in PROBABILITY_BASES:
        body[basis] = {}
        for split in SPLITS:
            chosen = _in_split(pairs, split)
            failed = {
                "correct_at_A": [
                    pair for pair in chosen if not _order_passes(pair.correct_at_a, basis, threshold)
                ],
                "correct_at_B": [
                    pair for pair in chosen if not _order_passes(pair.correct_at_b, basis, threshold)
                ],
            }
            both = [pair for pair in failed["correct_at_A"] if pair in failed["correct_at_B"]]
            either = {pair.base_group_id for pair in failed["correct_at_A"] + failed["correct_at_B"]}
            denominator = len(chosen)
            body[basis][split] = {
                "correct_at_A": _rate(len(failed["correct_at_A"]), denominator, "base questions"),
                "correct_at_B": _rate(len(failed["correct_at_B"]), denominator, "base questions"),
                "either": _rate(len(either), denominator, "base questions"),
                "both": _rate(len(both), denominator, "base questions"),
            }
    return body


def _populations(pairs: Sequence[_Pair], threshold: float) -> dict[str, Any]:
    return {
        "units": _units(pairs),
        "retention": _retention(pairs, threshold),
        "by_split": {
            split: {
                "units": _units(_in_split(pairs, split)),
                "retention": _retention(_in_split(pairs, split), threshold),
            }
            for split in SPLITS
        },
    }


def _retention(pairs: Sequence[_Pair], threshold: float) -> dict[str, Any]:
    raw = {pair.base_group_id for pair in pairs if _pair_passes(pair, "raw", threshold)}
    canonical = {pair.base_group_id for pair in pairs if _pair_passes(pair, "canonical", threshold)}
    denominator = len(pairs)
    return {
        "raw": {
            "retained": _rate(len(raw), denominator, "base questions"),
            "removed": _rate(denominator - len(raw), denominator, "base questions"),
        },
        "canonical": {
            "retained": _rate(len(canonical), denominator, "base questions"),
            "removed": _rate(denominator - len(canonical), denominator, "base questions"),
        },
        "overlap": {
            "both": _rate(len(raw & canonical), denominator, "base questions"),
            "raw_only": _rate(len(raw - canonical), denominator, "base questions"),
            "canonical_only": _rate(len(canonical - raw), denominator, "base questions"),
            "neither": _rate(denominator - len(raw | canonical), denominator, "base questions"),
        },
    }


def _diagnostics(pairs: Sequence[_Pair], threshold: float) -> dict[str, Any]:
    orders = [order for pair in pairs for order in (pair.correct_at_a, pair.correct_at_b)]
    masses = [order.canonical_label_mass for order in orders]
    invalid = sum(order.invalid_label for order in orders)
    canonical_only = [
        pair.base_group_id
        for pair in pairs
        if _pair_passes(pair, "canonical", threshold) and not _pair_passes(pair, "raw", threshold)
    ]
    mass: dict[str, Any] = {"count": len(masses), "mean": None, "minimum": None, "maximum": None}
    if masses:
        mass["mean"] = sum(masses) / len(masses)
        mass["minimum"] = min(masses)
        mass["maximum"] = max(masses)
    return {
        "invalid_labels": _rate(invalid, len(orders), "prompt rows"),
        "canonical_mass": mass,
        "canonical_only_base_groups": canonical_only,
        "raw_only_orders": _raw_only(pairs, threshold),
    }


def _raw_only(pairs: Sequence[_Pair], threshold: float) -> list[dict[str, str]]:
    found = []
    for pair in pairs:
        for order in (pair.correct_at_a, pair.correct_at_b):
            if _order_passes(order, "raw", threshold) and not _order_passes(order, "canonical", threshold):
                found.append({
                    "base_group_id": pair.base_group_id,
                    "split": pair.split,
                    "order_id": order.order_id,
                })
    return found


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


def _rate(count: int, denominator: int, unit: str) -> dict[str, Any]:
    if denominator == 0:
        return {
            "count": 0,
            "denominator": 0,
            "percentage": None,
            "unavailable_reason": f"zero denominator: 0 {unit}",
        }
    return {
        "count": count,
        "denominator": denominator,
        "percentage": count / denominator,
        "unavailable_reason": None,
    }


def _render(report: FilterReport, note: str) -> str:
    fields = report.payload()
    lines = [
        "Paired clean filter",
        f"Variant: {fields['paired_eligibility']['variant']}",
        f"Threshold: {fields['paired_eligibility']['threshold']}",
        "Both clean orders are required.",
        "The probability basis is never inferred from the retention rate.",
        note,
        "",
        "Units",
    ]
    units = fields["population_counts"]["units"]
    for name in ("base_questions", "intervention_instances", "paired_order_rows", "prompt_rows"):
        lines.append(f"{name}: {units[name]}")
    lines.extend(["", "Retention by base question"])
    _append_retention(lines, fields["population_counts"]["retention"])
    for split in SPLITS:
        lines.extend(["", f"Split {split}"])
        section = fields["population_counts"]["by_split"][split]
        lines.append(f"base questions: {section['units']['base_questions']}")
        _append_retention(lines, section["retention"])
        for basis in PROBABILITY_BASES:
            lines.append(f"{basis} order failures")
            failures = fields["order_failures"][basis][split]
            for name in ("correct_at_A", "correct_at_B", "either", "both"):
                lines.append(f"  {name}: {_format_rate(failures[name])}")
    diagnostics = fields["canonical_only_diagnostics"]
    lines.extend([
        "",
        f"Invalid canonical labels: {_format_rate(diagnostics['invalid_labels'])}",
        f"Canonical-only base questions: {', '.join(diagnostics['canonical_only_base_groups']) or 'none'}",
        f"Raw-only orders: {len(diagnostics['raw_only_orders'])}",
    ])
    return "\n".join(lines) + "\n"


def _append_retention(lines: list[str], retention: Mapping[str, Any]) -> None:
    for basis in PROBABILITY_BASES:
        retained = retention[basis]["retained"]
        removed = retention[basis]["removed"]
        lines.append(f"{basis} retained: {_format_rate(retained)}; removed: {_format_rate(removed)}")
    overlap = retention["overlap"]
    for name in ("both", "raw_only", "canonical_only", "neither"):
        lines.append(f"{name}: {_format_rate(overlap[name])}")


def _format_rate(rate: Mapping[str, Any]) -> str:
    if rate["unavailable_reason"] is not None:
        return f"unavailable ({rate['unavailable_reason']})"
    return f"{rate['count']}/{rate['denominator']}"


def _write(directory: Path, report: FilterReport, text: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    applied = directory / "applied_decision.json"
    if applied.exists():
        applied.unlink()
    artifact = identify(report)
    sidecar = json.dumps(artifact.sidecar(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_text(directory / "filter_report.json", sidecar)
    _atomic_text(directory / "filter_report.txt", text)
    if content_digest(json.loads(artifact.canonical_json)) != artifact.digest:
        raise RuntimeError("filter report digest does not match its canonical JSON")
    return artifact.digest


def _applied_document(applied: AppliedDecision) -> dict[str, Any]:
    return {
        "probability_basis": applied.probability_basis,
        "original_decision_id": applied.original_decision_id,
        "original_report_digest": applied.original_report_digest,
        "current_report_digest": applied.current_report_digest,
        "min_correct_prob": applied.min_correct_prob,
        "scope": {
            "dataset_adapter": applied.scope["dataset_adapter"],
            "source_snapshot": applied.scope["source_snapshot"],
            "model_content_signature": applied.scope["model_content_signature"],
            "prompt_contract_digest": applied.scope["prompt_contract_digest"],
            "canonical_answers": ["A", "B"],
            "require_both_orders": True,
        },
    }


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


def _scope(value: Mapping[str, Any]) -> dict[str, Any]:
    parsed = _scope_or_none(value)
    if parsed is None:
        raise ValueError(
            "decision scope requires the dataset adapter, source snapshot, "
            "model content signature, prompt contract, canonical answers A/B, "
            "and both clean orders"
        )
    return parsed


def _scope_or_none(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    required = (
        "dataset_adapter",
        "source_snapshot",
        "model_content_signature",
        "prompt_contract_digest",
    )
    parsed: dict[str, Any] = {}
    for field in required:
        item = value.get(field)
        if type(item) is not str or item == "":
            return None
        parsed[field] = item
    answers = value.get("canonical_answers")
    if isinstance(answers, str) or not isinstance(answers, Sequence) or tuple(answers) != ("A", "B"):
        return None
    if value.get("require_both_orders") is not True:
        return None
    parsed["canonical_answers"] = ("A", "B")
    parsed["require_both_orders"] = True
    return parsed


def _first_difference(recorded: object, current: Mapping[str, Any]) -> str:
    parsed = _scope_or_none(recorded)
    if parsed is None:
        return "scope"
    for field in SCOPE_FIELDS:
        if parsed[field] != current[field]:
            return field
    return "scope"
