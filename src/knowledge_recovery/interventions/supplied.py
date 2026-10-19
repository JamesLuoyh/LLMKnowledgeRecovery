"""Join frozen per-question text on ``base_item_id``. Never by row position."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from knowledge_recovery.errors import KnowledgeRecoveryError
from knowledge_recovery.interventions.base import make_instance
from knowledge_recovery.records import BaseItem, InterventionInstance, InterventionSpec


class MissingSuppliedContext(KnowledgeRecoveryError):
    """A base item has no supplied row. It is not turned into a clean prompt."""

    def __init__(self, *, base_item_id: str, variant_id: str, source: str) -> None:
        self.base_item_id = base_item_id
        self.variant_id = variant_id
        self.source = source
        super().__init__(
            "Missing supplied context.\n"
            f"base_item_id: {base_item_id}\n"
            f"variant_id: {variant_id}\n"
            f"source: {source}\n"
            "The missing row was not replaced with a clean prompt."
        )


class AmbiguousSuppliedContext(KnowledgeRecoveryError):
    """More than one row has the same base item and context id."""

    def __init__(self, *, base_item_id: str, context_id: str, count: int) -> None:
        self.base_item_id = base_item_id
        self.context_id = context_id
        self.count = count
        super().__init__(
            "Ambiguous supplied context.\n"
            f"base_item_id: {base_item_id}\n"
            f"context_id: {context_id}\n"
            f"matching rows: {count}\n"
            "The rows were not attached by position and were not replaced with a clean prompt."
        )


class UnsupportedSuppliedFormat(KnowledgeRecoveryError):
    """Parquet is a declared format, but this loader reads JSONL only."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(
            f"Unsupported supplied-text format: {path}\n"
            "Missing component: a Parquet supplied-text loader. Use JSONL."
        )


class SuppliedTextProvider:
    """Load a local JSONL table and join it on the base-item id."""

    name = "supplied_text"
    version = "supplied_text_v1"

    def materialize(
        self,
        base_item: BaseItem,
        spec: InterventionSpec,
        preparation_assets: Mapping[str, Any],
    ) -> tuple[InterventionInstance, ...]:
        del preparation_assets
        if spec.provider_name != self.name:
            raise ValueError(f"supplied_text provider cannot materialize {spec.provider_name}")
        parameters = spec.provider_parameters
        source = Path(str(parameters["source"]))
        rows = load_jsonl(source)
        join_on = str(parameters["join_on"])
        text_field = str(parameters["text_field"])
        instance_field = str(parameters["instance_field"])
        on_missing = str(parameters.get("on_missing") or "error")
        matched = [row for row in rows if row.get(join_on) == base_item.duplicate_group_id]
        if not matched:
            if on_missing == "exclude_pair":
                return ()
            raise MissingSuppliedContext(
                base_item_id=base_item.duplicate_group_id,
                variant_id=spec.variant_id,
                source=str(source),
            )
        return tuple(
            _instance(base_item, spec, row, text_field, instance_field)
            for row in _unique_contexts(matched, base_item.duplicate_group_id, instance_field)
        )


def load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """Read JSONL records in file order. A Parquet path is rejected."""
    if path.suffix.lower() == ".parquet":
        raise UnsupportedSuppliedFormat(path)
    if not path.is_file():
        raise MissingSuppliedContext(base_item_id="<file>", variant_id="<file>", source=str(path))
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip() == "":
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {line_number} is not JSON") from exc
        if type(parsed) is not dict:
            raise ValueError(f"{path} line {line_number} must be an object")
        rows.append(parsed)
    return tuple(rows)


def _unique_contexts(rows, base_item_id: str, instance_field: str) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        context_id = row.get(instance_field)
        if type(context_id) is not str or context_id == "":
            raise ValueError(f"{instance_field} must be a non-empty string")
        grouped.setdefault(context_id, []).append(row)
    chosen = []
    for context_id, group in grouped.items():
        if len(group) != 1:
            raise AmbiguousSuppliedContext(
                base_item_id=base_item_id, context_id=context_id, count=len(group)
            )
        chosen.append(group[0])
    return tuple(chosen)


def _instance(base_item, spec, row, text_field: str, instance_field: str) -> InterventionInstance:
    text = row.get(text_field)
    if type(text) is not str or text == "":
        raise ValueError(f"{text_field} must be a non-empty string")
    source_id = row.get("source_id")
    source_digest = row.get("source_digest")
    if type(source_id) is not str or source_id == "":
        raise ValueError("source_id must be a non-empty string")
    if (
        type(source_digest) is not str
        or len(source_digest) != 64
        or any(char not in "0123456789abcdef" for char in source_digest)
    ):
        raise ValueError("source_digest must be a full SHA-256 hex digest")
    context_id = row[instance_field]
    return make_instance(
        base_item=base_item,
        spec=spec,
        context_blocks=({"block_id": context_id, "text": text},),
        context_id=context_id,
        source_digest=source_digest,
        data_dependencies=("supplied_text", source_id),
    )
