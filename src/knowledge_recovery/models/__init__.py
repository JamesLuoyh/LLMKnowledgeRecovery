"""Local model assets and the Qwen hook map."""

from knowledge_recovery.models.assets import LocalAssets, activate_offline, resolve_local_assets
from knowledge_recovery.models.base import freeze_model

__all__ = [
    "LocalAssets",
    "activate_offline",
    "freeze_model",
    "resolve_local_assets",
]
