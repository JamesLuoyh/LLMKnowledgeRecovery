"""Recover clean sycophancy items from ``prompt_template == '{question}'`` rows.

Only ``base.question``, ``base.correct_answer``, and ``base.incorrect_answer``
are read. The base-group ID is a versioned SHA-256 of the exact question text.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from knowledge_recovery.datasets.base import NormalizationReport, RowNote
from knowledge_recovery.errors import KnowledgeRecoveryError
from knowledge_recovery.provenance import content_digest
from knowledge_recovery.records import BaseItem

NORMALIZATION_VERSION = "sycophancy_v1"
EXACT_TEMPLATE = "{question}"
_ANSWER_FIELDS = ("question", "correct_answer", "incorrect_answer")


class ConflictingBaseAnswers(KnowledgeRecoveryError):
    """One exact question has more than one correct/incorrect pair."""

    def __init__(self, conflicts: tuple[Mapping[str, Any], ...], path: Path, *, written: bool) -> None:
        self.conflicts = conflicts
        self.path = path
        self.written = written
        shown = ", ".join(repr(item["question"]) for item in conflicts)
        location = f"Conflicting records: {path}" if written else (
            f"Conflict directory does not exist, so no file was written: {path}"
        )
        super().__init__(
            "Conflicting answers for one exact question stop preparation.\n"
            f"Questions: {shown}\n"
            f"{location}\n"
            "No subset was selected or quarantined."
        )


def base_group_id(question: str) -> str:
    """Hash the exact question, including whitespace and Unicode."""
    return content_digest({
        "kind": "base_group",
        "schema": 1,
        "normalization_version": NORMALIZATION_VERSION,
        "question": question,
    })


def recover_sycophancy(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_identity: str,
    conflict_directory: Path,
) -> NormalizationReport:
    """Collapse identical triples and stop if one question has conflicting answers."""
    if type(source_identity) is not str or source_identity == "":
        raise ValueError("source_identity must be a non-empty string")
    if isinstance(rows, (str, bytes, Mapping)) or not isinstance(rows, Iterable):
        raise TypeError("rows must be an iterable of records")
    invalid: list[RowNote] = []
    exclusions: list[RowNote] = []
    grouped: dict[tuple[str, str, str], list[str]] = {}
    order: list[tuple[str, str, str]] = []
    seen_ids: set[str] = set()
    raw = 0
    for index, row in enumerate(rows):
        raw += 1
        outcome = _read_row(row, index, seen_ids)
        kind = outcome[0]
        if kind == "invalid":
            invalid.append(outcome[1])
        elif kind == "exclude":
            exclusions.append(outcome[1])
        else:
            triple = outcome[1]
            if triple not in grouped:
                grouped[triple] = []
                order.append(triple)
            grouped[triple].append(outcome[2])
    conflicts = _conflicts(grouped, order)
    if conflicts:
        path = _write_conflicts(Path(conflict_directory), conflicts)
        raise ConflictingBaseAnswers(conflicts, path, written=path.is_file())
    items: list[BaseItem] = []
    duplicate_rows = 0
    for question, correct, incorrect in order:
        source_row_ids = tuple(grouped[(question, correct, incorrect)])
        duplicate_rows += len(source_row_ids) - 1
        items.append(BaseItem(
            source_identity=source_identity,
            source_row_ids=source_row_ids,
            question=question,
            correct_answer=correct,
            incorrect_answer=incorrect,
            source_metadata={"prompt_template": EXACT_TEMPLATE},
            normalization_version=NORMALIZATION_VERSION,
            duplicate_group_id=base_group_id(question),
        ))
    return NormalizationReport(
        raw_source_rows=raw,
        recovered_base_items=len(items),
        duplicate_rows=duplicate_rows,
        invalid_rows=tuple(invalid),
        exclusions=tuple(exclusions),
        items=tuple(items),
    )


class SycophancyAdapter:
    """Sycophancy source adapter. It has no split and no model handle."""

    adapter_name = "sycophancy"

    def recover(self, rows, *, source_identity: str, conflict_directory: Path) -> NormalizationReport:
        return recover_sycophancy(
            rows,
            source_identity=source_identity,
            conflict_directory=conflict_directory,
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
    if "prompt_template" not in row:
        return ("invalid", RowNote(index, source_id, "missing_prompt_template"))
    template = row["prompt_template"]
    if template != EXACT_TEMPLATE:
        shown = template if type(template) is str else None
        return ("exclude", RowNote(index, source_id, "ignored_template", shown))
    base = row.get("base")
    if type(base) is not dict:
        return ("invalid", RowNote(index, source_id, "base_not_mapping"))
    values: dict[str, str] = {}
    for field in _ANSWER_FIELDS:
        if field not in base:
            return ("invalid", RowNote(index, source_id, f"missing_base_{field}"))
        value = base[field]
        if type(value) is not str:
            return ("invalid", RowNote(index, source_id, f"non_string_base_{field}"))
        if value == "":
            return ("invalid", RowNote(index, source_id, f"empty_{field}"))
        values[field] = value
    if values["correct_answer"] == values["incorrect_answer"]:
        return ("invalid", RowNote(index, source_id, "equal_answers"))
    triple = (values["question"], values["correct_answer"], values["incorrect_answer"])
    return ("keep", triple, source_id)


def _conflicts(grouped, order) -> tuple[dict[str, Any], ...]:
    by_question: dict[str, list[tuple[str, str, str]]] = {}
    for triple in order:
        by_question.setdefault(triple[0], []).append(triple)
    conflicts = []
    for question, triples in by_question.items():
        if len(triples) == 1:
            continue
        conflicts.append({
            "question": question,
            "groups": [
                {
                    "correct_answer": correct,
                    "incorrect_answer": incorrect,
                    "source_row_ids": list(grouped[(question, correct, incorrect)]),
                }
                for _question, correct, incorrect in triples
            ],
        })
    return tuple(conflicts)


def _write_conflicts(directory: Path, conflicts: tuple[Mapping[str, Any], ...]) -> Path:
    path = directory / "conflicting_answers.json"
    if not directory.is_dir():
        return path
    path.write_text(json.dumps(conflicts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
