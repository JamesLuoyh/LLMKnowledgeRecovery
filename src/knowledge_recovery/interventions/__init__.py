"""Intervention providers and materialization."""

from knowledge_recovery.interventions.base import InterventionProvider
from knowledge_recovery.interventions.identity import IdentityProvider
from knowledge_recovery.interventions.materialize import (
    MaterializationManifest,
    materialize_interventions,
    specs_from_config,
)
from knowledge_recovery.interventions.registry import ProviderRegistry, builtin_registry
from knowledge_recovery.interventions.supplied import (
    AmbiguousSuppliedContext,
    MissingSuppliedContext,
    SuppliedTextProvider,
)
from knowledge_recovery.interventions.templates import PRESETS, TemplateProvider, bind_answer_text

__all__ = [
    "AmbiguousSuppliedContext",
    "IdentityProvider",
    "InterventionProvider",
    "MaterializationManifest",
    "MissingSuppliedContext",
    "PRESETS",
    "ProviderRegistry",
    "SuppliedTextProvider",
    "TemplateProvider",
    "bind_answer_text",
    "builtin_registry",
    "materialize_interventions",
    "specs_from_config",
]
