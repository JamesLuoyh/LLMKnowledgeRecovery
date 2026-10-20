"""Versioned canonical JSON and separate artifact identities.

``H(record) = SHA-256(UTF-8(canonical JSON))``. Dictionary keys are sorted,
array order is kept, and Python's built-in ``hash`` is not used.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from knowledge_recovery.records import (
    AnalysisView,
    EvaluationIdentity,
    ExtractionRow,
    ExtractionSpec,
    FrozenModel,
    ModelInput,
    ProbeTraining,
    PromptContract,
    Record,
    freeze,
)

CANONICAL_JSON_VERSION = 1


def canonical_json(value: Any) -> str:
    """Serialize ``value`` with canonical-JSON version 1.

    The encoding is compact, keys are sorted by Unicode code point, and
    strings keep their exact content. Integers, floats (including ``-0.0``),
    and booleans stay distinct. Nonfinite numbers are rejected.
    """
    return json.dumps(
        _plain(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_digest(value: Any) -> str:
    """Return the lowercase SHA-256 hex digest of the canonical JSON of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    frozen = value if _already_frozen(value) else freeze(value)
    return _plain_frozen(frozen)


def _already_frozen(value: Any) -> bool:
    if value is None or type(value) in (str, bool, int, float):
        return True
    if type(value) is tuple:
        return all(_already_frozen(item) for item in value)
    if type(value) is MappingProxyType:
        return all(type(key) is str and _already_frozen(item) for key, item in value.items())
    return False


def _plain_frozen(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int, float):
        return value
    if type(value) is tuple:
        return [_plain_frozen(item) for item in value]
    return {key: _plain_frozen(item) for key, item in value.items()}


@dataclass(frozen=True)
class Artifact:
    """Human-readable record stored beside the digest that identifies it."""

    record: Mapping[str, Any]
    canonical_json: str
    digest: str
    canonical_json_version: int = CANONICAL_JSON_VERSION

    def sidecar(self) -> dict[str, Any]:
        """Return the parsed record, its digest, and the JSON version."""
        return {
            "canonical_json_version": self.canonical_json_version,
            "digest": self.digest,
            "record": json.loads(self.canonical_json),
        }


Artifact.__hash__ = None


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    return tuple(value)


def identify(record: Record) -> Artifact:
    """Hash one record and keep that record beside the digest."""
    payload = record.payload()
    text = canonical_json(payload)
    return Artifact(
        record=freeze(payload),
        canonical_json=text,
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def model_signature(
    *,
    weight_content_digest: str,
    model_config_digest: str,
    tokenizer_content_digest: str,
    model_adapter_version: str,
) -> Artifact:
    """Identity of frozen weights, model config, tokenizer, and adapter."""
    return identify(FrozenModel(
        weight_content_digest=weight_content_digest,
        model_config_digest=model_config_digest,
        tokenizer_content_digest=tokenizer_content_digest,
        model_adapter_version=model_adapter_version,
    ))


def prompt_contract(
    *,
    chat: bool,
    system_message: str | None,
    instruction: str,
    verbalizers: tuple[str, ...] | list[str],
    prompt_boundary: str,
    renderer_version: str,
    token_audit_content: Mapping[str, Any],
) -> Artifact:
    """Identity of the chat contract. Timestamps and paths are not arguments."""
    return identify(PromptContract(
        chat=chat,
        system_message=system_message,
        instruction=instruction,
        verbalizers=_sequence(verbalizers, "verbalizers"),
        prompt_boundary=prompt_boundary,
        renderer_version=renderer_version,
        token_audit_content=token_audit_content,
    ))


def prompt_signature(
    *,
    input_ids: tuple[int, ...] | list[int],
    logical_attention_mask: tuple[int, ...] | list[int],
    position_ids: tuple[int, ...] | list[int],
    other_forward_inputs: Mapping[str, Any],
    prompt_contract_digest: str,
) -> Artifact:
    """Identity of the logical model input and its prompt contract."""
    return identify(ModelInput(
        input_ids=_sequence(input_ids, "input_ids"),
        logical_attention_mask=_sequence(logical_attention_mask, "logical_attention_mask"),
        position_ids=_sequence(position_ids, "position_ids"),
        other_forward_inputs=other_forward_inputs,
        prompt_contract_digest=prompt_contract_digest,
    ))


def extraction_signature(
    *,
    extraction_adapter_version: str,
    layer_hook_definitions: tuple[Any, ...] | list[Any],
    position_selector: str,
    resolved_logical_positions: tuple[Any, ...] | list[Any],
    saved_representation_kinds: tuple[str, ...] | list[str],
    logits_scope: str,
    execution_precision: str,
    quantization: str,
    attention_backend: str,
    deterministic_settings: Mapping[str, Any],
    numeric_runtime_signature: Mapping[str, Any],
    residual_storage_dtype: str,
    logits_storage_dtype: str,
    tensor_schema_version: int,
) -> Artifact:
    """Identity of hooks, positions, precision, and stored tensor schema."""
    return identify(ExtractionSpec(
        extraction_adapter_version=extraction_adapter_version,
        layer_hook_definitions=_sequence(layer_hook_definitions, "layer_hook_definitions"),
        position_selector=position_selector,
        resolved_logical_positions=_sequence(resolved_logical_positions, "resolved_logical_positions"),
        saved_representation_kinds=_sequence(saved_representation_kinds, "saved_representation_kinds"),
        logits_scope=logits_scope,
        execution_precision=execution_precision,
        quantization=quantization,
        attention_backend=attention_backend,
        deterministic_settings=deterministic_settings,
        numeric_runtime_signature=numeric_runtime_signature,
        residual_storage_dtype=residual_storage_dtype,
        logits_storage_dtype=logits_storage_dtype,
        tensor_schema_version=tensor_schema_version,
    ))


def extraction_row_key(
    *,
    model_signature: str,
    prompt_signature: str,
    extraction_signature: str,
) -> Artifact:
    """Identity of one cached forward pass.

    The only inputs are the three signatures. Run, evaluation, split, batch,
    host, and ECE settings are not arguments.
    """
    return identify(ExtractionRow(
        model_signature=model_signature,
        prompt_signature=prompt_signature,
        extraction_signature=extraction_signature,
    ))


def view_id(
    *,
    role: str,
    row_references: tuple[Any, ...] | list[Any],
    group_ids: tuple[Any, ...] | list[Any],
    target_construction_version: str,
    class_order: tuple[str, ...] | list[str],
    eligibility: Any,
    weights: Any,
    split_provenance: Any,
    filter_provenance: Any,
) -> Artifact:
    """Identity of one ordered selection of cached rows."""
    return identify(AnalysisView(
        role=role,
        row_references=_sequence(row_references, "row_references"),
        group_ids=_sequence(group_ids, "group_ids"),
        target_construction_version=target_construction_version,
        class_order=_sequence(class_order, "class_order"),
        eligibility=eligibility,
        weights=weights,
        split_provenance=split_provenance,
        filter_provenance=filter_provenance,
    ))


def run_id(
    *,
    train_view_id: str,
    validation_view_id: str,
    task_spec: Any,
    architecture: Any,
    feature_spec: Any,
    method_id: str,
    task_adapter_version: str,
    method_parameters: Any,
    preprocessing_spec: Any,
    loss_spec: Any,
    optimizer_spec: Any,
    training_seed: int,
    stopping_rules: Any,
    selection_rules: Any,
    training_implementation_version: str,
    training_numeric_signature: Any,
) -> Artifact:
    """Identity of a probe-training specification, not of its checkpoints."""
    return identify(ProbeTraining(
        train_view_id=train_view_id,
        validation_view_id=validation_view_id,
        task_spec=task_spec,
        architecture=architecture,
        feature_spec=feature_spec,
        method_id=method_id,
        task_adapter_version=task_adapter_version,
        method_parameters=method_parameters,
        preprocessing_spec=preprocessing_spec,
        loss_spec=loss_spec,
        optimizer_spec=optimizer_spec,
        training_seed=training_seed,
        stopping_rules=stopping_rules,
        selection_rules=selection_rules,
        training_implementation_version=training_implementation_version,
        training_numeric_signature=training_numeric_signature,
    ))


def evaluation_id(
    *,
    method_state_digest: str,
    evaluation_view_id: str,
    prediction_convention: Any,
    metric_definitions: Any,
    aggregation_rules: Any,
    calibration_bins: Any,
    prediction_version: str,
    metrics_version: str,
    transform_digest: str | None = None,
    selection_manifest_digest: str | None = None,
    uncertainty: Any | None = None,
) -> Artifact:
    """Identity of scoring a saved method. Absent optional components are omitted."""
    return identify(EvaluationIdentity(
        method_state_digest=method_state_digest,
        evaluation_view_id=evaluation_view_id,
        prediction_convention=prediction_convention,
        metric_definitions=metric_definitions,
        aggregation_rules=aggregation_rules,
        calibration_bins=calibration_bins,
        prediction_version=prediction_version,
        metrics_version=metrics_version,
        transform_digest=transform_digest,
        selection_manifest_digest=selection_manifest_digest,
        uncertainty=uncertainty,
    ))
