"""Clean reference: the same base item with no inserted context."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from knowledge_recovery.interventions.base import make_instance
from knowledge_recovery.provenance import content_digest
from knowledge_recovery.records import BaseItem, InterventionInstance, InterventionSpec


class IdentityProvider:
    """Produce the clean instance. Both option orders reuse it unchanged."""

    name = "identity"
    version = "identity_v1"

    def materialize(
        self,
        base_item: BaseItem,
        spec: InterventionSpec,
        preparation_assets: Mapping[str, Any],
    ) -> tuple[InterventionInstance, ...]:
        del preparation_assets
        if spec.provider_name != self.name:
            raise ValueError(f"identity provider cannot materialize {spec.provider_name}")
        source = content_digest({"kind": "identity_source", "schema": 1, "version": self.version})
        return (make_instance(
            base_item=base_item,
            spec=spec,
            context_blocks=(),
            context_id=None,
            source_digest=source,
            data_dependencies=(),
        ),)
