"""Resolve configured variants into a stable instance manifest."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from knowledge_recovery.config import ExtractionConfig
from knowledge_recovery.interventions.base import PAIRING_SAME_CONTENT
from knowledge_recovery.interventions.registry import ProviderRegistry, builtin_registry
from knowledge_recovery.provenance import content_digest
from knowledge_recovery.records import BaseItem, InterventionInstance, InterventionSpec, freeze

_FAMILIES = {
    "identity": "reference",
    "template": "sycophancy",
    "supplied_text": "supplied_text",
}


@dataclass(frozen=True)
class MaterializationManifest:
    """Exact instances and their digest. Seeds are not part of this record."""

    instances: tuple[InterventionInstance, ...]
    exclusions: tuple[Any, ...]
    digest: str


def specs_from_config(
    extraction: ExtractionConfig,
    registry: ProviderRegistry | None = None,
) -> tuple[InterventionSpec, ...]:
    """Turn configured definitions into versioned specifications."""
    chosen = registry or builtin_registry()
    return tuple(_spec(definition, extraction, chosen) for definition in extraction.interventions)


def materialize_interventions(
    items: Sequence[BaseItem],
    specs: Sequence[InterventionSpec],
    registry: ProviderRegistry | None = None,
    preparation_assets: Mapping[str, Any] | None = None,
) -> MaterializationManifest:
    """Materialize each spec for each item. Split and probe seeds are not read."""
    chosen = registry or builtin_registry()
    assets = preparation_assets if preparation_assets is not None else {}
    instances: list[InterventionInstance] = []
    exclusions: list[Any] = []
    for item in items:
        for spec in specs:
            produced = chosen.get(spec.provider_name).materialize(item, spec, assets)
            if produced:
                instances.extend(produced)
                continue
            if spec.provider_parameters.get("on_missing") != "exclude_pair":
                raise ValueError(
                    f"{spec.variant_id} produced no instances. "
                    "A missing context is not replaced with a clean prompt."
                )
            exclusions.append(freeze({
                "base_item_id": item.duplicate_group_id,
                "variant_id": spec.variant_id,
                "reason": "missing_context",
            }))
    body = [instance.payload() for instance in instances]
    return MaterializationManifest(
        instances=tuple(instances),
        exclusions=tuple(exclusions),
        digest=content_digest({
            "kind": "materialization_manifest",
            "schema": 1,
            "instances": body,
            "exclusions": exclusions,
        }),
    )


def _spec(definition, extraction: ExtractionConfig, registry: ProviderRegistry) -> InterventionSpec:
    provider = registry.get(definition.provider)
    parameters = dict(definition.parameters)
    if definition.provider == "template":
        if definition.preset is not None:
            parameters["preset"] = definition.preset
        if definition.template is not None:
            parameters["template"] = definition.template
        parameters["placeholder_source"] = "answer_text"
        required = ("correct_answer", "incorrect_answer")
    elif definition.provider == "supplied_text":
        parameters.update({
            "source": definition.source,
            "join_on": definition.join_on,
            "text_field": definition.text_field,
            "instance_field": definition.instance_field,
            "on_missing": definition.on_missing or "error",
        })
        required = ()
    else:
        required = ()
    if definition.tags:
        tags = definition.tags
    elif definition.provider == "template":
        tags = ("sycophancy", definition.preset or "template")
    else:
        tags = (_FAMILIES[definition.provider],)
    placement = extraction.default_placement if definition.placement is None else definition.placement
    return InterventionSpec(
        variant_id=definition.id,
        family=definition.family or _FAMILIES[definition.provider],
        tags=tags,
        provider_name=provider.name,
        provider_version=provider.version,
        provider_parameters=parameters,
        required_item_fields=required,
        placement=placement,
        pairing_policy=definition.pairing or PAIRING_SAME_CONTENT,
    )
