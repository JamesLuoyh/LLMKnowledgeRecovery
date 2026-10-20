"""Read-only analysis views over cached rows.

Training and validation contain the configured training variants on their
own splits. OOD variants appear only in the OOD evaluation view. Changing
that assignment does not change the training view id.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from knowledge_recovery.provenance import Artifact, view_id
from knowledge_recovery.tasks.agreement import TARGET_VERSION as AWARENESS_TARGET_VERSION
from knowledge_recovery.tasks.agreement import awareness_target_index
from knowledge_recovery.tasks.base import AWARENESS_CLASS_ORDER, AWARENESS_TASK_ID, CONSISTENCY_CLASS_ORDER, CONSISTENCY_TASK_ID
from knowledge_recovery.tasks.clean_answer import TARGET_VERSION as CONSISTENCY_TARGET_VERSION
from knowledge_recovery.tasks.clean_answer import consistency_target_index

TASKS = (
    (AWARENESS_TASK_ID, AWARENESS_CLASS_ORDER, AWARENESS_TARGET_VERSION),
    (CONSISTENCY_TASK_ID, CONSISTENCY_CLASS_ORDER, CONSISTENCY_TARGET_VERSION),
)


@dataclass(frozen=True)
class ScoredRow:
    """One cached prompt with the canonical labels the tasks read."""

    row_key: str
    base_group_id: str
    split: str
    variant_id: str
    order_id: str
    clean_canonical_label: str
    contextual_canonical_label: str
    instance_id: str = ""
    contextual_canonical_probabilities: tuple[float, float] = (0.5, 0.5)

    def __post_init__(self) -> None:
        if self.split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        if self.clean_canonical_label not in ("A", "B") or self.contextual_canonical_label not in ("A", "B"):
            raise ValueError("canonical labels must be A or B")
        if self.instance_id == "":
            object.__setattr__(self, "instance_id", self.variant_id)
        probabilities = tuple(float(value) for value in self.contextual_canonical_probabilities)
        if len(probabilities) != 2 or any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("contextual canonical probabilities must be an A/B pair")
        object.__setattr__(self, "contextual_canonical_probabilities", probabilities)


def build_task_views(
    rows: Sequence[ScoredRow],
    *,
    eligible: Sequence[str],
    train_variants: Sequence[str],
    eval_variants: Sequence[str],
    split_digest: str,
    filter_digest: str,
) -> dict[str, Artifact]:
    """Build train, validation, ID evaluation, and OOD evaluation views for both tasks."""
    eligible_ids = tuple(eligible)
    views: dict[str, Artifact] = {}
    for task_id, class_order, target_version in TASKS:
        specs = (
            ("train", "train", "id", train_variants),
            ("validation", "val", "id", train_variants),
            ("evaluation_id", "test", "id", train_variants),
            ("evaluation_ood", "test", "ood", eval_variants),
        )
        for name, split, domain, variants in specs:
            chosen = _select(rows, eligible_ids, split, variants)
            role = "evaluation" if name.startswith("evaluation") else name
            views[f"{task_id}/{name}"] = view_id(
                role=role,
                row_references=tuple(_reference(row, task_id) for row in chosen),
                group_ids=tuple(sorted({row.base_group_id for row in chosen})),
                target_construction_version=target_version,
                class_order=class_order,
                eligibility={
                    "domain": domain,
                    "variants": list(variants),
                    "eligible_base_group_ids": list(eligible_ids),
                    "split": split,
                },
                weights={"scheme": "uniform", "value": 1},
                split_provenance=split_digest,
                filter_provenance=filter_digest,
            )
    return views


def _select(
    rows: Sequence[ScoredRow],
    eligible: Sequence[str],
    split: str,
    variants: Sequence[str],
) -> tuple[ScoredRow, ...]:
    allowed = set(variants)
    kept = [
        row
        for row in rows
        if row.split == split and row.variant_id in allowed and row.base_group_id in set(eligible)
    ]
    return tuple(sorted(kept, key=lambda row: (row.base_group_id, row.variant_id, row.order_id, row.row_key)))


def _reference(row: ScoredRow, task_id: str) -> dict[str, object]:
    if task_id == AWARENESS_TASK_ID:
        target = awareness_target_index(row.clean_canonical_label, row.contextual_canonical_label)
    else:
        target = consistency_target_index(row.clean_canonical_label)
    return {
        "row_key": row.row_key,
        "base_group_id": row.base_group_id,
        "variant_id": row.variant_id,
        "instance_id": row.instance_id,
        "order_id": row.order_id,
        "target": target,
        "canonical_ab": list(row.contextual_canonical_probabilities),
    }
