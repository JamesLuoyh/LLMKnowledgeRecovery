"""Last-token selection and the extraction runner."""

from knowledge_recovery.extraction.positions import last_nonpadding_index, select_last_nonpadding
from knowledge_recovery.extraction.runner import (
    BatchExceededMemory,
    ExtractionResult,
    offline_network_guard,
    run_extraction,
)


def build_row_key(*args, **kwargs):
    """Import the identity builder lazily to avoid a model-adapter cycle."""
    from knowledge_recovery.extraction.identity import build_row_key as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "BatchExceededMemory",
    "ExtractionResult",
    "build_row_key",
    "last_nonpadding_index",
    "offline_network_guard",
    "run_extraction",
    "select_last_nonpadding",
]
