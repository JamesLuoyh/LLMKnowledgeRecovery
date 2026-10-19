"""Explicit provider names. An unknown name is a missing component."""

from __future__ import annotations

from knowledge_recovery.config import ConfigurationError
from knowledge_recovery.interventions.base import InterventionProvider


class ProviderRegistry:
    """Stable provider names registered by the caller or the builtin set."""

    def __init__(self) -> None:
        self._providers: dict[str, InterventionProvider] = {}

    def register(self, provider: InterventionProvider) -> None:
        if provider.name in self._providers:
            raise ConfigurationError(f"intervention provider {provider.name!r} is already registered")
        self._providers[provider.name] = provider

    def get(self, name: str) -> InterventionProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise ConfigurationError(
                f"intervention provider {name!r} is not registered. "
                "Missing component: that intervention provider."
            ) from exc


def builtin_registry() -> ProviderRegistry:
    """Identity, configurable templates, and supplied per-question text."""
    from knowledge_recovery.interventions.identity import IdentityProvider
    from knowledge_recovery.interventions.supplied import SuppliedTextProvider
    from knowledge_recovery.interventions.templates import TemplateProvider

    registry = ProviderRegistry()
    registry.register(IdentityProvider())
    registry.register(TemplateProvider())
    registry.register(SuppliedTextProvider())
    return registry
