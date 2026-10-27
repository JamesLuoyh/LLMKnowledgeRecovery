"""Recover two-option items from four-option MMLU rows.

Each row reads ``question``, ``choices``, and ``answer`` (an index into
``choices``). The correct answer is ``choices[answer]``. One incorrect answer
is kept: drawn with a seeded digest of the item, or the model's most or least
preferred incorrect choice from ``rank-distractors``. Either way it is fixed
per item before any variant or order is built, so the
existing ``{correct}``/``{incorrect}`` presets and the A/B option orders apply
unchanged. The base-group ID hashes the exact question and the sorted choice
texts: MMLU reuses generic question stems with different choices, and the same
item listed with its choices in another order must land in one split.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from knowledge_recovery.datasets.base import NormalizationReport, RowNote
from knowledge_recovery.datasets.distractor_ranking import SELECTIONS, select_preferred_incorrect
from knowledge_recovery.provenance import content_digest
from knowledge_recovery.records import BaseItem

NORMALIZATION_VERSION = "mmlu_v1"


def base_group_id(question: str, choices: Iterable[str]) -> str:
    """Hash the exact question and the choice texts, ignoring choice order."""
    return content_digest({
        "kind": "base_group",
        "schema": 1,
        "normalization_version": NORMALIZATION_VERSION,
        "question": question,
        "choices": sorted(choices),
    })


def pick_distractor(group_id: str, correct: str, choices: Iterable[str], *, seed: int) -> str | None:
    """Pick one incorrect choice text. Order of ``choices`` does not change the result."""
    candidates = sorted({choice for choice in choices if choice != correct})
    if not candidates:
        return None
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def recover_mmlu(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_identity: str,
    conflict_directory: Path,
    distractor_seed: int | None,
    distractor_selection: str = "seeded_random",
    preferences: Mapping[str, Mapping[str, Any]] | None = None,
) -> NormalizationReport:
    """Collapse repeated items. An item with conflicting answers is excluded, not fatal.

    ``seeded_random`` draws the distractor from ``distractor_seed``.
    ``most_preferred`` and ``least_preferred`` read ``preferences``, the
    ``rank-distractors`` output keyed by base-group ID.
    """
    del conflict_directory
    if type(source_identity) is not str or source_identity == "":
        raise ValueError("source_identity must be a non-empty string")
    if distractor_selection not in SELECTIONS:
        raise ValueError(f"distractor_selection must be one of {SELECTIONS}")
    if distractor_selection == "seeded_random":
        if type(distractor_seed) is not int or distractor_seed < 0:
            raise ValueError("distractor_seed must be a non-negative integer")
    elif preferences is None:
        raise ValueError(f"{distractor_selection} needs the rank-distractors preferences")
    if isinstance(rows, (str, bytes, Mapping)) or not isinstance(rows, Iterable):
        raise TypeError("rows must be an iterable of records")
    invalid: list[RowNote] = []
    exclusions: list[RowNote] = []
    # group id -> (index, row id, question, choices, correct text, subject) per row
    grouped: dict[str, list[tuple[int, str, str, tuple[str, ...], str, str]]] = {}
    order: list[str] = []
    seen_ids: set[str] = set()
    raw = 0
    for index, row in enumerate(rows):
        raw += 1
        outcome = _read_row(row, index, seen_ids)
        if outcome[0] == "invalid":
            invalid.append(outcome[1])
            continue
        _kind, source_id, question, choices, correct, subject = outcome
        group_id = base_group_id(question, choices)
        if group_id not in grouped:
            grouped[group_id] = []
            order.append(group_id)
        grouped[group_id].append((index, source_id, question, choices, correct, subject))
    items: list[BaseItem] = []
    duplicate_rows = 0
    for group_id in order:
        members = grouped[group_id]
        _index, _source_id, question, choices, correct, subject = members[0]
        if len({member[4] for member in members}) > 1:
            exclusions.extend(
                RowNote(member[0], member[1], "conflicting_answers") for member in members
            )
            continue
        metadata: dict[str, Any] = {"choices": list(choices), "subject": subject}
        if distractor_selection == "seeded_random":
            incorrect = pick_distractor(group_id, correct, choices, seed=distractor_seed)
            metadata["distractor_seed"] = distractor_seed
        elif pick_distractor(group_id, correct, choices, seed=0) is None:
            incorrect = None
        else:
            ranked = preferences.get(group_id)
            if ranked is None:
                invalid.extend(RowNote(member[0], member[1], "missing_distractor_preference") for member in members)
                continue
            scores = ranked["mean_probability"]
            incorrect = select_preferred_incorrect(correct, scores, selection=distractor_selection)
            metadata["distractor_selection"] = distractor_selection
            metadata["choice_preference"] = dict(scores)
        if incorrect is None:
            invalid.extend(RowNote(member[0], member[1], "no_distinct_distractor") for member in members)
            continue
        duplicate_rows += len(members) - 1
        items.append(BaseItem(
            source_identity=source_identity,
            source_row_ids=tuple(member[1] for member in members),
            question=question,
            correct_answer=correct,
            incorrect_answer=incorrect,
            source_metadata=metadata,
            normalization_version=NORMALIZATION_VERSION,
            duplicate_group_id=group_id,
        ))
    return NormalizationReport(
        raw_source_rows=raw,
        recovered_base_items=len(items),
        duplicate_rows=duplicate_rows,
        invalid_rows=tuple(invalid),
        exclusions=tuple(exclusions),
        items=tuple(items),
    )


class MmluAdapter:
    """MMLU source adapter. It has no split and no model handle."""

    adapter_name = "mmlu"

    def __init__(
        self,
        *,
        distractor_seed: int | None,
        distractor_selection: str = "seeded_random",
        preferences: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.distractor_seed = distractor_seed
        self.distractor_selection = distractor_selection
        self.preferences = preferences

    def recover(self, rows, *, source_identity: str, conflict_directory: Path) -> NormalizationReport:
        return recover_mmlu(
            rows,
            source_identity=source_identity,
            conflict_directory=conflict_directory,
            distractor_seed=self.distractor_seed,
            distractor_selection=self.distractor_selection,
            preferences=self.preferences,
        )


def _read_row(row: Any, index: int, seen_ids: set[str]):
    if type(row) is not dict:
        return ("invalid", RowNote(index, None, "row_not_mapping"))
    source_id = row.get("id")
    if type(source_id) is not str or source_id == "":
        return ("invalid", RowNote(index, None, "missing_source_id"))
    if source_id in seen_ids:
        return ("invalid", RowNote(index, source_id, "duplicate_source_row_id"))
    seen_ids.add(source_id)
    question = row.get("question")
    if type(question) is not str:
        return ("invalid", RowNote(index, source_id, "non_string_question"))
    if question.strip() == "":
        return ("invalid", RowNote(index, source_id, "empty_question"))
    choices = row.get("choices")
    if type(choices) is not list or len(choices) < 2:
        return ("invalid", RowNote(index, source_id, "choices_not_list"))
    if any(type(choice) is not str for choice in choices):
        return ("invalid", RowNote(index, source_id, "non_string_choice"))
    if any(choice.strip() == "" for choice in choices):
        return ("invalid", RowNote(index, source_id, "empty_choice"))
    answer = row.get("answer")
    if type(answer) is not int or not 0 <= answer < len(choices):
        return ("invalid", RowNote(index, source_id, "answer_out_of_range"))
    subject = row.get("subject", "")
    if type(subject) is not str:
        return ("invalid", RowNote(index, source_id, "non_string_subject"))
    return ("keep", source_id, question, tuple(choices), choices[answer], subject)
