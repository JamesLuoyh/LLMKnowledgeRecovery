"""Immutable records. Artifact identity is a SHA-256 digest, not ``hash()``."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, ClassVar

from knowledge_recovery.errors import PROBABILITY_BASES

SCHEMA_VERSION = 1
VIEW_ROLES = ("train", "validation", "evaluation")
# These names are execution or analysis settings. They are recorded on their
# own artifacts and are rejected if they appear inside an extraction identity.
EXTRACTION_EXCLUDED_KEYS = frozenset({
    "run_id",
    "eval_id",
    "evaluation_id",
    "split_seed",
    "split_ratio",
    "batch_size",
    "hostname",
    "ece_bins",
    "ece",
    "probe_seed",
    "training_seed",
    "probability_basis",
    "timestamp",
    "processing_order",
})
RUN_EXCLUDED_KEYS = frozenset({
    "eval_id",
    "evaluation_id",
    "evaluation_view_id",
    "calibration_bins",
    "ece_bins",
    "ece",
    "hostname",
    "batch_size",
    "ood_variants",
    "test_variants",
    "timestamp",
    "processing_order",
})
EVALUATION_EXCLUDED_KEYS = frozenset({
    "run_id",
    "hostname",
    "batch_size",
    "training_seed",
    "split_seed",
    "output_references",
})
PROMPT_CONTRACT_EXCLUDED_KEYS = frozenset({
    "timestamp",
    "audited_at",
    "path",
    "filesystem_path",
    "hostname",
})


def freeze(value: Any) -> Any:
    """Return a JSON-compatible value that callers cannot mutate in place."""
    if value is None or type(value) is str:
        return value
    if type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite numbers are rejected")
        return value
    if type(value) in (list, tuple):
        return tuple(freeze(item) for item in value)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("dictionary keys must be strings")
            frozen[key] = freeze(item)
        return MappingProxyType(frozen)
    raise TypeError(f"{type(value).__name__} values cannot be recorded")


def reject_excluded_keys(value: Any, banned: frozenset[str], *, identity: str) -> None:
    """Walk nested objects and reject keys that belong to a different identity."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in banned:
                raise ValueError(f"{identity} must not contain {key}")
            reject_excluded_keys(item, banned, identity=identity)
    elif type(value) is tuple:
        for item in value:
            reject_excluded_keys(item, banned, identity=identity)


class Record:
    """Frozen content addressed by ``provenance.identify``, never by ``hash()``."""

    kind: ClassVar[str]
    schema: ClassVar[int] = SCHEMA_VERSION
    omit_if_none: ClassVar[tuple[str, ...]] = ()
    __hash__ = None

    def __post_init__(self) -> None:
        for item in fields(self):
            object.__setattr__(self, item.name, freeze(getattr(self, item.name)))
        self.validate()

    def validate(self) -> None:
        """Override to reject content that would name the wrong artifact."""

    def payload(self) -> dict[str, Any]:
        """Kind, schema, and fields. Inapplicable optional fields are omitted."""
        body: dict[str, Any] = {"kind": self.kind, "schema": self.schema}
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None and item.name in self.omit_if_none:
                continue
            body[item.name] = value
        return body


def record(kind: str):
    """Declare a frozen schema-1 record of one artifact kind."""

    def decorate(cls):
        cls.kind = kind
        cls.schema = SCHEMA_VERSION
        wrapped = dataclass(frozen=True)(cls)
        wrapped.__hash__ = None
        return wrapped

    return decorate


def _text(value: object, label: str) -> None:
    if type(value) is not str or value == "":
        raise ValueError(f"{label} must be a non-empty string")


def _digest(value: object, label: str) -> None:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a full SHA-256 hex digest")


def _int_tuple(value: object, label: str) -> None:
    if type(value) is not tuple or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{label} must be a sequence of non-negative integers")


def _texts(value: object, label: str) -> None:
    if type(value) is not tuple:
        raise ValueError(f"{label} must be a sequence of strings")
    for item in value:
        _text(item, label)


def _mapping(value: object, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")


@record("frozen_model")
class FrozenModel(Record):
    """Content identity of local weights, config, and tokenizer.

    Directory names are not fields: moving an unchanged asset keeps this digest.
    """

    weight_content_digest: str
    model_config_digest: str
    tokenizer_content_digest: str
    model_adapter_version: str

    def validate(self) -> None:
        _digest(self.weight_content_digest, "weight_content_digest")
        _digest(self.model_config_digest, "model_config_digest")
        _digest(self.tokenizer_content_digest, "tokenizer_content_digest")
        _text(self.model_adapter_version, "model_adapter_version")
        reject_excluded_keys(self.payload(), EXTRACTION_EXCLUDED_KEYS, identity="model signature")


@record("prompt_contract")
class PromptContract(Record):
    """Template and audit content that the prompt signature binds.

    Audit timestamps and filesystem locations are not fields of this contract.
    """

    chat: bool
    system_message: str | None
    instruction: str
    verbalizers: tuple[str, ...]
    prompt_boundary: str
    renderer_version: str
    token_audit_content: Mapping[str, Any]

    def validate(self) -> None:
        _text(self.instruction, "instruction")
        _text(self.prompt_boundary, "prompt_boundary")
        _text(self.renderer_version, "renderer_version")
        _texts(self.verbalizers, "verbalizer")
        if not self.verbalizers:
            raise ValueError("at least one verbalizer is required")
        _mapping(self.token_audit_content, "token_audit_content")
        if type(self.chat) is not bool:
            raise ValueError("chat must be a boolean")
        if self.chat:
            _text(self.system_message, "system_message")
        elif self.system_message is not None:
            raise ValueError("no-chat mode has no system message")
        reject_excluded_keys(
            self.payload(), PROMPT_CONTRACT_EXCLUDED_KEYS, identity="prompt contract"
        )


@record("model_input")
class ModelInput(Record):
    """Unpadded logical forward inputs. Batch size is not an input."""

    input_ids: tuple[int, ...]
    logical_attention_mask: tuple[int, ...]
    position_ids: tuple[int, ...]
    other_forward_inputs: Mapping[str, Any]
    prompt_contract_digest: str

    def validate(self) -> None:
        _int_tuple(self.input_ids, "input_ids")
        _int_tuple(self.logical_attention_mask, "logical_attention_mask")
        _int_tuple(self.position_ids, "position_ids")
        if not (
            len(self.input_ids)
            == len(self.logical_attention_mask)
            == len(self.position_ids)
        ):
            raise ValueError(
                "input_ids, logical_attention_mask, and position_ids must have equal length"
            )
        if any(item not in (0, 1) for item in self.logical_attention_mask):
            raise ValueError("logical_attention_mask values must be 0 or 1")
        _mapping(self.other_forward_inputs, "other_forward_inputs")
        _digest(self.prompt_contract_digest, "prompt_contract_digest")
        reject_excluded_keys(self.payload(), EXTRACTION_EXCLUDED_KEYS, identity="prompt signature")


@record("extraction_spec")
class ExtractionSpec(Record):
    """How a forward pass is hooked, positioned, and stored."""

    extraction_adapter_version: str
    layer_hook_definitions: tuple[Any, ...]
    position_selector: str
    resolved_logical_positions: tuple[Any, ...]
    saved_representation_kinds: tuple[Any, ...]
    logits_scope: str
    execution_precision: str
    quantization: str
    attention_backend: str
    deterministic_settings: Mapping[str, Any]
    numeric_runtime_signature: Mapping[str, Any]
    residual_storage_dtype: str
    logits_storage_dtype: str
    tensor_schema_version: int

    def validate(self) -> None:
        for label in (
            "extraction_adapter_version",
            "position_selector",
            "logits_scope",
            "execution_precision",
            "quantization",
            "attention_backend",
            "residual_storage_dtype",
            "logits_storage_dtype",
        ):
            _text(getattr(self, label), label)
        if type(self.layer_hook_definitions) is not tuple:
            raise ValueError("layer_hook_definitions must be a sequence")
        if type(self.resolved_logical_positions) is not tuple:
            raise ValueError("resolved_logical_positions must be a sequence")
        _texts(self.saved_representation_kinds, "saved representation")
        if not self.saved_representation_kinds:
            raise ValueError("at least one saved representation is required")
        _mapping(self.deterministic_settings, "deterministic_settings")
        _mapping(self.numeric_runtime_signature, "numeric_runtime_signature")
        if type(self.tensor_schema_version) is not int or self.tensor_schema_version < 1:
            raise ValueError("tensor_schema_version must be a positive integer")
        reject_excluded_keys(
            self.payload(), EXTRACTION_EXCLUDED_KEYS, identity="extraction signature"
        )


@record("extraction_row")
class ExtractionRow(Record):
    """Cache key of one forward pass: model, prompt, and extraction signatures only."""

    model_signature: str
    prompt_signature: str
    extraction_signature: str

    def validate(self) -> None:
        _digest(self.model_signature, "model_signature")
        _digest(self.prompt_signature, "prompt_signature")
        _digest(self.extraction_signature, "extraction_signature")


@record("analysis_view")
class AnalysisView(Record):
    """One training, validation, or evaluation selection of cached rows."""

    role: str
    row_references: tuple[Any, ...]
    group_ids: tuple[Any, ...]
    target_construction_version: str
    class_order: tuple[Any, ...]
    eligibility: Any
    weights: Any
    split_provenance: Any
    filter_provenance: Any

    def validate(self) -> None:
        if self.role not in VIEW_ROLES:
            raise ValueError("role must be train, validation, or evaluation")
        for label in ("row_references", "group_ids", "class_order"):
            if type(getattr(self, label)) is not tuple:
                raise ValueError(f"{label} must be a sequence")
        _texts(self.class_order, "class")
        if not self.class_order:
            raise ValueError("class_order must be non-empty")
        _text(self.target_construction_version, "target_construction_version")
        if self.eligibility is None or self.weights is None:
            raise ValueError("eligibility and weights are required")
        if self.split_provenance is None or self.filter_provenance is None:
            raise ValueError("split and filter provenance are required")


@record("probe_training")
class ProbeTraining(Record):
    """Inputs that can change a fitted or selected probe.

    Checkpoint digests, selected layers, and evaluation-only settings are not
    fields. Baseline methods that have no optimizer use a later constructor.
    """

    train_view_id: str
    validation_view_id: str
    task_spec: Any
    architecture: Any
    feature_spec: Any
    method_id: str
    task_adapter_version: str
    method_parameters: Any
    preprocessing_spec: Any
    loss_spec: Any
    optimizer_spec: Any
    training_seed: int
    stopping_rules: Any
    selection_rules: Any
    training_implementation_version: str
    training_numeric_signature: Any

    def validate(self) -> None:
        _digest(self.train_view_id, "train_view_id")
        _digest(self.validation_view_id, "validation_view_id")
        for label in ("method_id", "task_adapter_version", "training_implementation_version"):
            _text(getattr(self, label), label)
        if type(self.training_seed) is not int:
            raise ValueError("training_seed must be an integer")
        for label in (
            "task_spec",
            "architecture",
            "feature_spec",
            "method_parameters",
            "preprocessing_spec",
            "loss_spec",
            "optimizer_spec",
            "stopping_rules",
            "selection_rules",
            "training_numeric_signature",
        ):
            if getattr(self, label) is None:
                raise ValueError(f"{label} is required")
        reject_excluded_keys(self.payload(), RUN_EXCLUDED_KEYS, identity="probe run")


@record("evaluation")
class EvaluationIdentity(Record):
    """Scoring identity of a saved method state. Output paths are not fields."""

    omit_if_none = (
        "transform_digest",
        "selection_manifest_digest",
        "uncertainty",
    )
    method_state_digest: str
    evaluation_view_id: str
    prediction_convention: Any
    metric_definitions: Any
    aggregation_rules: Any
    calibration_bins: Any
    prediction_version: str
    metrics_version: str
    transform_digest: str | None = None
    selection_manifest_digest: str | None = None
    uncertainty: Any | None = None

    def validate(self) -> None:
        _digest(self.method_state_digest, "method_state_digest")
        _digest(self.evaluation_view_id, "evaluation_view_id")
        _text(self.prediction_version, "prediction_version")
        _text(self.metrics_version, "metrics_version")
        if self.prediction_convention is None or self.metric_definitions is None:
            raise ValueError("prediction convention and metric definitions are required")
        if self.aggregation_rules is None or self.calibration_bins is None:
            raise ValueError("aggregation rules and calibration bins are required")
        for label in ("transform_digest", "selection_manifest_digest"):
            value = getattr(self, label)
            if value is not None:
                _digest(value, label)
        reject_excluded_keys(self.payload(), EVALUATION_EXCLUDED_KEYS, identity="evaluation")


@record("base_item")
class BaseItem(Record):
    """One clean question after normalization, with every source row retained."""

    source_identity: str
    source_row_ids: tuple[str, ...]
    question: str
    correct_answer: str
    incorrect_answer: str
    source_metadata: Mapping[str, Any]
    normalization_version: str
    duplicate_group_id: str

    def validate(self) -> None:
        for label in (
            "source_identity",
            "question",
            "correct_answer",
            "incorrect_answer",
            "normalization_version",
            "duplicate_group_id",
        ):
            _text(getattr(self, label), label)
        _texts(self.source_row_ids, "source row ID")
        if not self.source_row_ids:
            raise ValueError("at least one source row ID is required")
        if len(set(self.source_row_ids)) != len(self.source_row_ids):
            raise ValueError("duplicate source row ID")
        if self.correct_answer == self.incorrect_answer:
            raise ValueError("correct and incorrect answer texts must differ")
        _mapping(self.source_metadata, "source_metadata")


@record("intervention_spec")
class InterventionSpec(Record):
    """Versioned description of how context is produced and where it is placed."""

    variant_id: str
    family: str
    tags: tuple[Any, ...]
    provider_name: str
    provider_version: str
    provider_parameters: Mapping[str, Any]
    required_item_fields: tuple[Any, ...]
    placement: Any
    pairing_policy: str

    def validate(self) -> None:
        for label in ("variant_id", "family", "provider_name", "provider_version", "pairing_policy"):
            _text(getattr(self, label), label)
        _texts(self.tags, "tag")
        _texts(self.required_item_fields, "required item field")
        _mapping(self.provider_parameters, "provider_parameters")
        if self.placement is None:
            raise ValueError("placement is required")


@record("intervention_instance")
class InterventionInstance(Record):
    """Exact context text materialized for one base item."""

    base_item_id: str
    variant_id: str
    instance_id: str
    context_blocks: tuple[Any, ...]
    insertion_anchors: tuple[Any, ...]
    source_digest: str
    content_digest: str
    preparation_seed: int | None
    data_dependencies: tuple[Any, ...]

    def validate(self) -> None:
        for label in ("base_item_id", "variant_id", "instance_id"):
            _text(getattr(self, label), label)
        _digest(self.source_digest, "source_digest")
        _digest(self.content_digest, "content_digest")
        for label in ("context_blocks", "insertion_anchors", "data_dependencies"):
            if type(getattr(self, label)) is not tuple:
                raise ValueError(f"{label} must be a sequence")
        if self.preparation_seed is not None and type(self.preparation_seed) is not int:
            raise ValueError("preparation_seed must be an integer or null")


@record("prompt_record")
class PromptRecord(Record):
    """Rendered prompt metadata. Token arrays keep the order they were stored in."""

    base_item_id: str
    order_id: str
    variant_id: str
    intervention_instance_id: str
    correct_label: str
    ordered_answer_texts: tuple[Any, ...]
    resolved_blocks: Any
    placement: Any
    rendered_text: str
    input_token_ids: tuple[Any, ...]
    attention: Any
    token_audit_id: str
    prompt_hash: str
    selected_input_position: int

    def validate(self) -> None:
        for label in (
            "base_item_id",
            "order_id",
            "variant_id",
            "intervention_instance_id",
            "correct_label",
            "rendered_text",
        ):
            _text(getattr(self, label), label)
        _texts(self.ordered_answer_texts, "answer text")
        _int_tuple(self.input_token_ids, "input_token_ids")
        _digest(self.token_audit_id, "token_audit_id")
        _digest(self.prompt_hash, "prompt_hash")
        if type(self.selected_input_position) is not int or self.selected_input_position < 0:
            raise ValueError("selected_input_position must be a non-negative integer")
        if self.resolved_blocks is None or self.placement is None or self.attention is None:
            raise ValueError("resolved blocks, placement, and attention are required")


@record("extraction_record")
class ExtractionRecord(Record):
    """Pointers to shard tensors. The vectors themselves stay in the cache."""

    prompt_id: str
    extraction_key: str
    residual_reference: Any
    logits_reference: Any
    final_normalized_state: Any
    numeric_metadata: Mapping[str, Any]
    shard_references: tuple[Any, ...]

    def validate(self) -> None:
        _digest(self.prompt_id, "prompt_id")
        _digest(self.extraction_key, "extraction_key")
        _mapping(self.numeric_metadata, "numeric_metadata")
        if type(self.shard_references) is not tuple or self.residual_reference is None:
            raise ValueError("residual and shard references are required")
        if self.logits_reference is None:
            raise ValueError("logits reference is required")


@record("answer_scores")
class AnswerScores(Record):
    """Raw and canonical A/B scores plus the unrestricted greedy token."""

    raw_ab_probabilities: tuple[Any, ...]
    canonical_ab_probabilities: tuple[Any, ...]
    canonical_label_mass: float
    unrestricted_greedy_token_id: int
    canonical_greedy_label: str
    invalid_label: bool
    scoring_version: str

    def validate(self) -> None:
        for label in ("raw_ab_probabilities", "canonical_ab_probabilities"):
            value = getattr(self, label)
            if type(value) is not tuple or len(value) != 2 or any(type(item) is not float for item in value):
                raise ValueError(f"{label} must be two probabilities in label order")
            if any(not math.isfinite(item) for item in value):
                raise ValueError("nonfinite numbers are rejected")
        if type(self.canonical_label_mass) is not float or not math.isfinite(self.canonical_label_mass):
            raise ValueError("canonical_label_mass must be a finite float")
        if type(self.unrestricted_greedy_token_id) is not int or self.unrestricted_greedy_token_id < 0:
            raise ValueError("unrestricted_greedy_token_id must be a non-negative integer")
        _text(self.canonical_greedy_label, "canonical_greedy_label")
        _text(self.scoring_version, "scoring_version")
        if type(self.invalid_label) is not bool:
            raise ValueError("invalid_label must be a boolean")


@record("split_manifest")
class SplitManifest(Record):
    """Saved base-group assignment. Its digest is computed beside this record.

    ``split_seed`` is not the probe training seed.
    """

    group_to_split: Mapping[str, str]
    source_universe: tuple[Any, ...]
    grouping_version: str
    split_seed: int
    fractions: Mapping[str, Any]
    allocations: Mapping[str, Any]

    def validate(self) -> None:
        _mapping(self.group_to_split, "group_to_split")
        _texts(self.source_universe, "source group")
        _text(self.grouping_version, "grouping_version")
        if type(self.split_seed) is not int:
            raise ValueError("split_seed must be an integer")
        _mapping(self.fractions, "fractions")
        _mapping(self.allocations, "allocations")


@record("filter_report")
class FilterReport(Record):
    """Both paired eligibility results. The report digest is stored beside it."""

    paired_eligibility: Mapping[str, Any]
    order_failures: Mapping[str, Any]
    population_counts: Mapping[str, Any]
    split_denominators: Mapping[str, Any]
    canonical_only_diagnostics: Mapping[str, Any]
    paths: Mapping[str, Any]

    def validate(self) -> None:
        for label in (
            "paired_eligibility",
            "order_failures",
            "population_counts",
            "split_denominators",
            "canonical_only_diagnostics",
            "paths",
        ):
            _mapping(getattr(self, label), label)


@record("decision_record")
class DecisionRecord(Record):
    """Explicit probability-basis choice. It is never inferred from a retention rate."""

    probability_basis: str
    filter_report_digest: str
    scope: Any
    timestamp: str
    user_instruction: str

    def validate(self) -> None:
        if self.probability_basis not in PROBABILITY_BASES:
            raise ValueError("probability_basis must be raw or canonical")
        _digest(self.filter_report_digest, "filter_report_digest")
        if self.scope is None:
            raise ValueError("scope is required")
        _text(self.timestamp, "timestamp")
        _text(self.user_instruction, "user_instruction")


@record("task_spec")
class TaskSpec(Record):
    """Target construction and the meaning of each class."""

    task_id: str
    input_schema: Any
    output_schema: Any
    target_construction: Any
    class_meaning: Any
    class_order: tuple[Any, ...]
    permitted_prediction_features: tuple[Any, ...]
    decoding_convention: Any
    pointwise_score: Any

    def validate(self) -> None:
        _text(self.task_id, "task_id")
        _texts(self.class_order, "class")
        _texts(self.permitted_prediction_features, "prediction feature")
        for label in (
            "input_schema",
            "output_schema",
            "target_construction",
            "class_meaning",
            "decoding_convention",
            "pointwise_score",
        ):
            if getattr(self, label) is None:
                raise ValueError(f"{label} is required")


@record("method_spec")
class MethodSpec(Record):
    """Task-specific method interface. A shared architecture does not share weights."""

    task_id: str
    method_id: str
    adapter_version: str
    fit_inputs: tuple[Any, ...]
    predict_inputs: tuple[Any, ...]
    fitting_requirement: str
    architecture: Any
    parameters: Mapping[str, Any]
    output_validation: Any

    def validate(self) -> None:
        for label in ("task_id", "method_id", "adapter_version", "fitting_requirement"):
            _text(getattr(self, label), label)
        _texts(self.fit_inputs, "fit input")
        _texts(self.predict_inputs, "predict input")
        _mapping(self.parameters, "parameters")
        if self.output_validation is None:
            raise ValueError("output_validation is required")


@record("probe_spec")
class ProbeSpec(Record):
    """Architecture and preprocessing requested from cached representations."""

    layers: tuple[Any, ...]
    positions: tuple[Any, ...]
    preprocessing: Any
    output_interface: Any
    architecture: Any
    compatible_tasks: tuple[Any, ...]
    compatible_losses: tuple[Any, ...]

    def validate(self) -> None:
        for label in ("layers", "positions", "compatible_tasks", "compatible_losses"):
            if type(getattr(self, label)) is not tuple:
                raise ValueError(f"{label} must be a sequence")
        for label in ("preprocessing", "output_interface", "architecture"):
            if getattr(self, label) is None:
                raise ValueError(f"{label} is required")


@record("run_manifest")
class RunManifest(Record):
    """Training specification plus output digests. The run id hashes only the specification."""

    train_view_id: str
    validation_view_id: str
    extraction_references: tuple[Any, ...]
    task: Any
    architecture: Any
    features: Any
    preprocessing: Any
    training_seed: int
    loss: Any
    optimizer: Any
    stopping_rules: Any
    selection_rules: Any
    environment_versions: Mapping[str, Any]
    checkpoint_digest: str | None
    selection_digest: str | None

    def validate(self) -> None:
        _digest(self.train_view_id, "train_view_id")
        _digest(self.validation_view_id, "validation_view_id")
        if type(self.training_seed) is not int:
            raise ValueError("training_seed must be an integer")
        if type(self.extraction_references) is not tuple:
            raise ValueError("extraction_references must be a sequence")
        _mapping(self.environment_versions, "environment_versions")
        reject_excluded_keys(
            self.environment_versions,
            frozenset({"hostname", "batch_size", "timestamp", "processing_order"}),
            identity="run manifest",
        )
        for label in ("checkpoint_digest", "selection_digest"):
            value = getattr(self, label)
            if value is not None:
                _digest(value, label)
        for label in (
            "task",
            "architecture",
            "features",
            "preprocessing",
            "loss",
            "optimizer",
            "stopping_rules",
            "selection_rules",
        ):
            if getattr(self, label) is None:
                raise ValueError(f"{label} is required")


@record("evaluation_manifest")
class EvaluationManifest(Record):
    """Evaluation identity plus the paths of its outputs."""

    task_id: str
    method_id: str
    method_state_digest: str
    evaluation_view_id: str
    metric_definitions: Any
    aggregation_rules: Any
    calibration_bins: Any
    prediction_convention: Any
    prediction_version: str
    metrics_version: str
    output_references: Any
    checkpoint_digest: str | None = None
    selection_digest: str | None = None
    transform_digest: str | None = None
    uncertainty: Any | None = None

    def validate(self) -> None:
        _text(self.task_id, "task_id")
        _text(self.method_id, "method_id")
        _digest(self.method_state_digest, "method_state_digest")
        _digest(self.evaluation_view_id, "evaluation_view_id")
        _text(self.prediction_version, "prediction_version")
        _text(self.metrics_version, "metrics_version")
        if self.output_references is None or self.calibration_bins is None:
            raise ValueError("calibration bins and output references are required")
        for label in ("checkpoint_digest", "selection_digest", "transform_digest"):
            value = getattr(self, label)
            if value is not None:
                _digest(value, label)


@record("execution_provenance")
class ExecutionProvenance(Record):
    """Host and batch facts stored beside a run, never inside an extraction key."""

    timestamp: str
    hostname: str
    batch_size: int
    processing_order: tuple[Any, ...]

    def validate(self) -> None:
        _text(self.timestamp, "timestamp")
        _text(self.hostname, "hostname")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(self.processing_order) is not tuple:
            raise ValueError("processing_order must be a sequence")
