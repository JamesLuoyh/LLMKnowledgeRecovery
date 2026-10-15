"""Dataset adapters return base items. They do not split or query a model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from knowledge_recovery.records import BaseItem


@dataclass(frozen=True)
class RowNote:
    """One source row that did not become its own base item."""

    index: int
    source_row_id: str | None
    reason: str
    template: str | None = None


@dataclass(frozen=True)
class NormalizationReport:
    """Counts kept separate from model-correctness filtering."""

    raw_source_rows: int
    recovered_base_items: int
    duplicate_rows: int
    invalid_rows: tuple[RowNote, ...]
    exclusions: tuple[RowNote, ...]
    items: tuple[BaseItem, ...]

    def template_counts(self) -> dict[str, int]:
        """Count ignored source templates by their exact text."""
        counts: dict[str, int] = {}
        for note in self.exclusions:
            key = note.template if note.template is not None else "<non-string>"
            counts[key] = counts.get(key, 0) + 1
        return counts


@runtime_checkable
class DatasetAdapter(Protocol):
    """Recover base items and provenance from source rows."""

    adapter_name: str

    def recover(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        source_identity: str,
        conflict_directory: Path,
    ) -> NormalizationReport:
        """Return normalized items. Do not assign splits or call a model."""
