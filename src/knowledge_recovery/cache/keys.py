"""Extraction row keys.

The key hashes the model, prompt, and extraction signatures only.
Reporting and training settings are accepted so a caller can show that they
were considered, and they are left out of the digest. A repeated setting
therefore addresses the same stored row.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from knowledge_recovery.provenance import extraction_row_key
from knowledge_recovery.records import EXTRACTION_EXCLUDED_KEYS

# Recorded on analysis, run, or evaluation artifacts. None of these is an
# input to ``cache_row_key``. The one list lives beside the extraction records.
REPORTING_SETTINGS = EXTRACTION_EXCLUDED_KEYS


def cache_row_key(
    *,
    model_signature: str,
    prompt_signature: str,
    extraction_signature: str,
    reporting: Mapping[str, Any] | None = None,
) -> str:
    """Return the extraction-row digest. ``reporting`` does not enter it."""
    _ignored_reporting(reporting)
    return extraction_row_key(
        model_signature=model_signature,
        prompt_signature=prompt_signature,
        extraction_signature=extraction_signature,
    ).digest


def _ignored_reporting(reporting: Mapping[str, Any] | None) -> None:
    if reporting is None:
        return
    if not isinstance(reporting, Mapping):
        raise TypeError("reporting settings must be a mapping")
    unknown = [key for key in reporting if key not in REPORTING_SETTINGS]
    if unknown:
        raise ValueError(
            "reporting settings must be analysis or training fields, not extraction inputs: "
            + ", ".join(sorted(unknown))
        )
