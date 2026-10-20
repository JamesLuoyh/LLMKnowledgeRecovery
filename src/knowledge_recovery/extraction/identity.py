"""Extraction row keys shared by the runner and the compatibility tests.

Batch size, split seed, loss, ECE, and OOD assignment are reporting fields.
They are accepted and then left out of the digest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from knowledge_recovery.cache.keys import cache_row_key
from knowledge_recovery.cache.schemas import TENSOR_SCHEMA_VERSION
from knowledge_recovery.errors import SUPPORTED_POSITION_SELECTOR
from knowledge_recovery.models.qwen import (
    EXTRACTION_ADAPTER_VERSION,
    HOOK_DEFINITIONS,
)
from knowledge_recovery.provenance import extraction_signature, prompt_signature

SAVED_REPRESENTATIONS = (
    "post_block_residual",
    "final_normalized_state",
    "full_vocabulary_logits",
)


def build_row_key(
    *,
    model_signature: str,
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
    prompt_contract_digest: str,
    position_selector: str = SUPPORTED_POSITION_SELECTOR,
    selected_position: int,
    execution_precision: str,
    quantization: str = "none",
    attention_backend: str = "eager",
    numeric_runtime_signature: Mapping[str, Any] | None = None,
    reporting: Mapping[str, Any] | None = None,
) -> str:
    """Hash one logical forward pass. ``reporting`` does not enter the key."""
    ids = tuple(input_ids)
    mask = tuple(attention_mask)
    if len(ids) != len(mask):
        raise ValueError("input ids and the attention mask must have equal length")
    if position_selector != SUPPORTED_POSITION_SELECTOR:
        raise ValueError(f"the initial selector is {SUPPORTED_POSITION_SELECTOR}")
    if type(selected_position) is not int or not 0 <= selected_position < len(ids):
        raise ValueError("selected_position must be an index of the logical sequence")
    prompt = prompt_signature(
        input_ids=ids,
        logical_attention_mask=mask,
        position_ids=tuple(range(len(ids))),
        other_forward_inputs={},
        prompt_contract_digest=prompt_contract_digest,
    )
    extraction = extraction_signature(
        extraction_adapter_version=EXTRACTION_ADAPTER_VERSION,
        layer_hook_definitions=HOOK_DEFINITIONS,
        position_selector=position_selector,
        resolved_logical_positions=(selected_position,),
        saved_representation_kinds=SAVED_REPRESENTATIONS,
        logits_scope="full_vocabulary",
        execution_precision=execution_precision,
        quantization=quantization,
        attention_backend=attention_backend,
        deterministic_settings={
            "dropout": False,
            "augmentation": False,
            "seeded_inference": False,
        },
        numeric_runtime_signature=(
            dict(numeric_runtime_signature)
            if numeric_runtime_signature is not None
            else {
                "backend": "unspecified",
                "attention_backend": attention_backend,
                "execution_precision": execution_precision,
                "residual_storage_dtype": "bf16",
                "logits_storage_dtype": "float32",
            }
        ),
        residual_storage_dtype="bf16",
        logits_storage_dtype="float32",
        tensor_schema_version=TENSOR_SCHEMA_VERSION,
    )
    return cache_row_key(
        model_signature=model_signature,
        prompt_signature=prompt.digest,
        extraction_signature=extraction.digest,
        reporting=reporting,
    )
