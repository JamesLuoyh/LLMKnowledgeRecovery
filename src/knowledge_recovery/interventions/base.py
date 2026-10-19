"""Intervention providers turn one specification into exact context instances."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from knowledge_recovery.provenance import content_digest
from knowledge_recovery.records import BaseItem, InterventionInstance, InterventionSpec

PAIRING_SAME_CONTENT = "same_content_across_option_orders"


@runtime_checkable
class InterventionProvider(Protocol):
    """Materialize context without probe predictions or split assignments."""

    name: str
    version: str

    def materialize(
        self,
        base_item: BaseItem,
        spec: InterventionSpec,
        preparation_assets: Mapping[str, Any],
    ) -> tuple[InterventionInstance, ...]:
        """Return exact instances. Do not read split or probe seeds."""


def instance_id(
    *,
    base_item_id: str,
    variant_id: str,
    context_id: str | None,
    content: str,
) -> str:
    """Stable instance identity. It does not depend on row order or a split seed."""
    return content_digest({
        "kind": "intervention_instance_id",
        "schema": 1,
        "base_item_id": base_item_id,
        "variant_id": variant_id,
        "context_id": context_id,
        "content_digest": content,
    })


def blocks_digest(blocks: tuple[Any, ...]) -> str:
    return content_digest({"kind": "context_blocks", "schema": 1, "blocks": blocks})


def make_instance(
    *,
    base_item: BaseItem,
    spec: InterventionSpec,
    context_blocks: tuple[Any, ...],
    context_id: str | None,
    source_digest: str,
    data_dependencies: tuple[str, ...],
) -> InterventionInstance:
    """Build one instance and record where its text will be inserted."""
    content = blocks_digest(context_blocks)
    anchors: tuple[Any, ...] = ()
    if context_blocks:
        anchors = ({"slot": spec.placement},) if type(spec.placement) is str else (spec.placement,)
    return InterventionInstance(
        base_item_id=base_item.duplicate_group_id,
        variant_id=spec.variant_id,
        instance_id=instance_id(
            base_item_id=base_item.duplicate_group_id,
            variant_id=spec.variant_id,
            context_id=context_id,
            content=content,
        ),
        context_blocks=context_blocks,
        insertion_anchors=anchors,
        source_digest=source_digest,
        content_digest=content,
        preparation_seed=None,
        data_dependencies=data_dependencies,
    )
