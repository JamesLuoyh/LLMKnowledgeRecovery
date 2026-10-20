"""Typed extraction, analysis-view, and probe-run configuration.

Unknown keys and missing schema versions are errors. Null is a real value
for the probability basis, the decision record, and the token audit; none of
those nulls is replaced with an inferred setting.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from knowledge_recovery.errors import (
    SUPPORTED_POSITION_SELECTOR,
    AnswerInstructionMismatch,
    KnowledgeRecoveryError,
    MissingLocalAsset,
    UnsupportedPositionSelector,
)
from knowledge_recovery.records import freeze

SCHEMA_VERSION = 1
NAMED_PLACEMENTS = (
    "before_question",
    "after_question_before_options",
    "after_options_before_instruction",
)
INITIAL_SYSTEM_MESSAGE = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
INITIAL_INSTRUCTION = "Answer with exactly one letter. Do not include any other text."
_TOP_LEVEL = ("extraction", "analysis_view", "probe_run")
_PRECISIONS = ("bf16", "fp16", "fp32")
_PROVIDERS = ("identity", "template", "supplied_text")
_PAIRING = "same_content_across_option_orders"
_DATASET_PATHS = {
    "clean_question_path": "base.question",
    "correct_answer_path": "base.correct_answer",
    "incorrect_answer_path": "base.incorrect_answer",
}
_DEFINITION_OPTIONAL = (
    "preset",
    "template",
    "source",
    "join_on",
    "text_field",
    "instance_field",
    "placement",
    "pairing",
    "on_missing",
    "family",
    "tags",
    "parameters",
)


class ConfigurationError(KnowledgeRecoveryError):
    """The configuration text violates the extraction, view, or probe-run contract."""


@dataclass(frozen=True)
class LinearV1:
    """Resolved ``linear_v1`` profile. These are engineering defaults, not tuned values."""

    optimizer: str
    learning_rate: float
    weight_decay: float
    weight_decay_on: str
    betas: tuple[float, float]
    epsilon: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    scheduler: None
    gradient_clipping: None
    mixed_precision: bool


LINEAR_V1 = LinearV1(
    optimizer="AdamW",
    learning_rate=1e-3,
    weight_decay=1e-4,
    weight_decay_on="weights",
    betas=(0.9, 0.999),
    epsilon=1e-8,
    batch_size=256,
    max_epochs=200,
    early_stopping_patience=20,
    scheduler=None,
    gradient_clipping=None,
    mixed_precision=False,
)


@dataclass(frozen=True)
class InterventionDefinition:
    """One configured variant. Absent optional fields stay null."""

    id: str
    provider: str
    preset: str | None
    template: str | None
    source: str | None
    join_on: str | None
    text_field: str | None
    instance_field: str | None
    placement: Any
    pairing: str | None
    on_missing: str | None
    family: str | None
    tags: tuple[str, ...]
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ExtractionConfig:
    schema_version: int
    storage_root: str
    offline: bool
    allow_download: bool
    model_path: str
    dataset_path: str
    model_name: str
    model_revision: str
    tokenizer_revision: str
    precision: str
    quantization: str
    dataset_adapter: str
    dataset_name: str
    dataset_split: str
    dataset_revision: str
    regenerate_variants: bool
    clean_question_path: str
    correct_answer_path: str
    incorrect_answer_path: str
    source_filter_field: str
    source_filter_value: str
    chat: bool
    system_message: str | None
    instruction: str
    intervention_registry: str
    default_placement: str
    preset_placeholder_source: str
    interventions: tuple[InterventionDefinition, ...]
    answer_mode: str
    labels: tuple[str, ...]
    verbalizers: tuple[str, ...]
    token_audit: str | None
    selector: str
    layers: str
    representation: str
    save_final_normalized_state: bool
    residual_storage_dtype: str
    logits_storage_dtype: str
    logits_scope: str


@dataclass(frozen=True)
class AnalysisViewConfig:
    schema_version: int
    train_fraction: float
    val_fraction: float
    test_fraction: float
    split_seed: int
    filter_variant: str
    filter_splits: tuple[str, ...]
    min_correct_prob: float
    paired_orders: bool
    train_variants: tuple[str, ...]
    eval_variants: tuple[str, ...]
    agreement_basis: str
    recovery_target: str


@dataclass(frozen=True)
class ProbeRunConfig:
    schema_version: int
    architecture: str
    loss: str
    positive_class_awareness: str
    positive_class_consistency: str
    training_seed: int
    selection_metric: str
    default_profile: str
    profile: LinearV1
    behavioral_awareness: tuple[str, ...]
    knowledge_consistency: tuple[str, ...]
    brier_convention: str
    ece_bins: int
    summaries: tuple[str, ...]
    confidence_intervals: bool
    save_per_example_predictions: bool


@dataclass(frozen=True)
class ExperimentConfig:
    """The three configuration objects, loaded from one file and not yet executed."""

    path: str
    extraction: ExtractionConfig
    analysis_view: AnalysisViewConfig
    probe_run: ProbeRunConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Read ``path`` and return the three typed objects. This does not touch assets."""
    source = Path(path)
    if not source.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    document = _object(loaded, "configuration", _TOP_LEVEL)
    return ExperimentConfig(
        path=str(source),
        extraction=_extraction(document["extraction"]),
        analysis_view=_analysis_view(document["analysis_view"]),
        probe_run=_probe_run(document["probe_run"]),
    )


def validate_before_expensive_work(
    config: ExperimentConfig,
    *,
    repository_root: Path | None = None,
) -> None:
    """Check the storage root and local assets without creating them.

    A missing root is an error. This function does not create directories and does
    not substitute the repository or the home directory.
    """
    root = Path(config.extraction.storage_root)
    if not root.is_dir():
        raise ConfigurationError(
            f"Storage root does not exist: {root}\n"
            "The storage root is checked before any expensive work. "
            "It was not created, and heavy files will not be written into "
            "the repository or home directory."
        )
    resolved = root.resolve()
    repository = (repository_root or Path(__file__).resolve().parents[2]).resolve()
    if resolved == repository or repository in resolved.parents:
        raise ConfigurationError(
            f"Storage root {resolved} is inside the repository {repository}. "
            "Heavy files will not be written there."
        )
    if resolved == Path.home().resolve():
        raise ConfigurationError(
            f"Storage root {resolved} is the home directory. "
            "Heavy files will not be written there."
        )
    model_path = config.extraction.model_path
    dataset_path = config.extraction.dataset_path
    if not Path(model_path).exists():
        raise MissingLocalAsset(kind="model", expected_location=model_path)
    if not Path(dataset_path).exists():
        raise MissingLocalAsset(kind="dataset", expected_location=dataset_path)


def _extraction(raw: Any) -> ExtractionConfig:
    body = _object(
        raw,
        "extraction",
        (
            "schema_version",
            "storage",
            "assets",
            "model",
            "dataset",
            "prompt",
            "interventions",
            "answers",
            "internals",
        ),
    )
    storage = _object(_take(body, "storage", "extraction.storage"), "extraction.storage", ("root",))
    assets = _object(
        body["assets"],
        "extraction.assets",
        ("offline", "allow_download", "model_path", "dataset_path"),
    )
    model = _object(
        body["model"],
        "extraction.model",
        ("name", "revision", "tokenizer_revision", "precision", "quantization"),
    )
    dataset = _object(
        body["dataset"],
        "extraction.dataset",
        (
            "adapter",
            "name",
            "split",
            "revision",
            "regenerate_variants",
            "clean_question_path",
            "correct_answer_path",
            "incorrect_answer_path",
            "source_row_filter",
        ),
    )
    prompt = _object(
        body["prompt"],
        "extraction.prompt",
        ("chat", "system_message", "instruction"),
    )
    interventions = _object(
        body["interventions"],
        "extraction.interventions",
        ("registry", "default_placement", "preset_placeholder_source", "definitions"),
    )
    answers = _object(
        body["answers"],
        "extraction.answers",
        ("mode", "labels", "verbalizers", "token_audit"),
    )
    internals = _object(
        body["internals"],
        "extraction.internals",
        (
            "selector",
            "layers",
            "representation",
            "save_final_normalized_state",
            "residual_storage_dtype",
            "logits_storage_dtype",
            "logits_scope",
        ),
    )
    offline = _bool(assets, "offline", "extraction.assets")
    allow_download = _bool(assets, "allow_download", "extraction.assets")
    if allow_download or not offline:
        raise ConfigurationError("offline mode cannot download assets")
    precision = _text(model["precision"], "extraction.model.precision")
    if precision not in _PRECISIONS:
        raise ConfigurationError(
            f"extraction.model.precision {precision!r} is not an explicit precision. "
            "There is no silent fallback."
        )
    if model["quantization"] != "none":
        raise ConfigurationError(
            f"extraction.model.quantization {model['quantization']!r} is not implemented. "
            "Missing component: that quantization mode. It was not changed to none."
        )
    adapter = _text(dataset["adapter"], "extraction.dataset.adapter")
    if adapter != "sycophancy":
        raise ConfigurationError(
            f"extraction.dataset.adapter {adapter!r} is not implemented. "
            "Missing component: a dataset adapter for this source. "
            "Clean-question fields will not be guessed."
        )
    for key, expected in _DATASET_PATHS.items():
        if dataset[key] != expected:
            raise ConfigurationError(
                f"extraction.dataset.{key} must be {expected!r}. "
                "No fallback field search is permitted."
            )
    source_filter = _object(
        dataset["source_row_filter"],
        "extraction.dataset.source_row_filter",
        ("field", "exact_value"),
    )
    if source_filter["field"] != "prompt_template" or source_filter["exact_value"] != "{question}":
        raise ConfigurationError(
            "extraction.dataset.source_row_filter must keep prompt_template "
            "exactly equal to '{question}'."
        )
    if dataset["regenerate_variants"] is not True:
        raise ConfigurationError(
            "extraction.dataset.regenerate_variants must be true. "
            "Source template text is not reused as the clean question."
        )
    chat = _bool(prompt, "chat", "extraction.prompt")
    system_message = prompt["system_message"]
    if system_message is not None:
        system_message = _text(system_message, "extraction.prompt.system_message")
    if chat and system_message is None:
        raise ConfigurationError(
            "extraction.prompt.system_message must be explicit when chat is true. "
            "A template default is not substituted."
        )
    if not chat and system_message is not None:
        raise ConfigurationError(
            "extraction.prompt.system_message must be null when chat is false. "
            "No-chat input has no system text."
        )
    instruction = _text(prompt["instruction"], "extraction.prompt.instruction")
    labels = _strings(answers, "labels", "extraction.answers", expected=("A", "B"))
    verbalizers = _strings(answers, "verbalizers", "extraction.answers", expected=("A", "B"))
    if answers["mode"] != "mcqa":
        kind = "direct_answer" if answers["mode"] == "direct" else "incompatible_label_set"
        raise AnswerInstructionMismatch(
            request_kind=kind,
            instruction=instruction,
            labels=None if kind == "direct_answer" else labels,
        )
    selector = _text(internals["selector"], "extraction.internals.selector")
    if selector != SUPPORTED_POSITION_SELECTOR:
        raise UnsupportedPositionSelector(selector=selector)
    _expect(internals, "layers", "all_transformer_blocks", "extraction.internals")
    _expect(internals, "representation", "post_block_pre_final_norm", "extraction.internals")
    _expect(internals, "residual_storage_dtype", "bf16", "extraction.internals")
    _expect(internals, "logits_storage_dtype", "float32", "extraction.internals")
    _expect(internals, "logits_scope", "full_vocabulary", "extraction.internals")
    if internals["save_final_normalized_state"] is not True:
        raise ConfigurationError(
            "extraction.internals.save_final_normalized_state must be true. "
            "The final normalized state is stored separately from the final block residual."
        )
    placeholder_source = _text(
        interventions["preset_placeholder_source"],
        "extraction.interventions.preset_placeholder_source",
    )
    if placeholder_source != "answer_text":
        raise ConfigurationError(
            "extraction.interventions.preset_placeholder_source must be answer_text. "
            "{correct} and {incorrect} bind to answer text, not to option letters."
        )
    registry = _text(interventions["registry"], "extraction.interventions.registry")
    if registry != "builtin_and_configured":
        raise ConfigurationError(
            f"extraction.interventions.registry {registry!r} is not implemented. "
            "Missing component: that intervention registry."
        )
    placement = _named_placement(
        interventions["default_placement"], "extraction.interventions.default_placement"
    )
    return ExtractionConfig(
        schema_version=_schema(body, "extraction"),
        storage_root=_text(storage["root"], "extraction.storage.root"),
        offline=offline,
        allow_download=allow_download,
        model_path=_text(assets["model_path"], "extraction.assets.model_path"),
        dataset_path=_text(assets["dataset_path"], "extraction.assets.dataset_path"),
        model_name=_text(model["name"], "extraction.model.name"),
        model_revision=_text(model["revision"], "extraction.model.revision"),
        tokenizer_revision=_text(model["tokenizer_revision"], "extraction.model.tokenizer_revision"),
        precision=precision,
        quantization="none",
        dataset_adapter=adapter,
        dataset_name=_text(dataset["name"], "extraction.dataset.name"),
        dataset_split=_text(dataset["split"], "extraction.dataset.split"),
        dataset_revision=_text(dataset["revision"], "extraction.dataset.revision"),
        regenerate_variants=True,
        clean_question_path="base.question",
        correct_answer_path="base.correct_answer",
        incorrect_answer_path="base.incorrect_answer",
        source_filter_field="prompt_template",
        source_filter_value="{question}",
        chat=chat,
        system_message=system_message,
        instruction=instruction,
        intervention_registry=registry,
        default_placement=placement,
        preset_placeholder_source=placeholder_source,
        interventions=_definitions(interventions["definitions"]),
        answer_mode="mcqa",
        labels=labels,
        verbalizers=verbalizers,
        token_audit=_audit(answers["token_audit"]),
        selector=selector,
        layers="all_transformer_blocks",
        representation="post_block_pre_final_norm",
        save_final_normalized_state=True,
        residual_storage_dtype="bf16",
        logits_storage_dtype="float32",
        logits_scope="full_vocabulary",
    )


def _analysis_view(raw: Any) -> AnalysisViewConfig:
    body = _object(raw, "analysis_view", ("schema_version", "split", "filter_correct", "analysis"))
    split = _object(
        body["split"],
        "analysis_view.split",
        ("train_fraction", "val_fraction", "test_fraction", "seed"),
    )
    fractions = tuple(_fraction(split, key, "analysis_view.split") for key in (
        "train_fraction",
        "val_fraction",
        "test_fraction",
    ))
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-12):
        raise ConfigurationError("analysis_view.split fractions must sum to 1")
    chosen = _object(
        body["filter_correct"],
        "analysis_view.filter_correct",
        ("variant", "splits", "min_correct_prob", "paired_orders"),
    )
    if chosen["paired_orders"] is not True:
        raise ConfigurationError("analysis_view.filter_correct.paired_orders must be true")
    analysis = _object(
        body["analysis"],
        "analysis_view.analysis",
        ("train_variants", "eval_variants", "agreement_basis", "recovery_target"),
    )
    if analysis["agreement_basis"] != "canonical":
        raise ConfigurationError(
            "analysis_view.analysis.agreement_basis must be canonical. "
            "That is the primary awareness target."
        )
    if analysis["recovery_target"] != "clean_canonical_argmax":
        raise ConfigurationError(
            "analysis_view.analysis.recovery_target must be clean_canonical_argmax"
        )
    return AnalysisViewConfig(
        schema_version=_schema(body, "analysis_view"),
        train_fraction=fractions[0],
        val_fraction=fractions[1],
        test_fraction=fractions[2],
        split_seed=_seed(split, "seed", "analysis_view.split"),
        filter_variant=_text(chosen["variant"], "analysis_view.filter_correct.variant"),
        filter_splits=_strings(
            chosen, "splits", "analysis_view.filter_correct", expected=("train", "val", "test")
        ),
        min_correct_prob=_probability(chosen["min_correct_prob"], "analysis_view.filter_correct.min_correct_prob"),
        paired_orders=True,
        train_variants=_strings(analysis, "train_variants", "analysis_view.analysis"),
        eval_variants=_strings(analysis, "eval_variants", "analysis_view.analysis"),
        agreement_basis="canonical",
        recovery_target="clean_canonical_argmax",
    )


def _probe_run(raw: Any) -> ProbeRunConfig:
    body = _object(raw, "probe_run", ("schema_version", "probe", "methods", "evaluation"))
    probe = _object(
        body["probe"],
        "probe_run.probe",
        (
            "architecture",
            "loss",
            "positive_class_awareness",
            "positive_class_consistency",
            "training_seed",
            "selection_metric",
            "default_profile",
        ),
    )
    if probe["architecture"] != "layerwise_linear":
        raise ConfigurationError(
            f"probe_run.probe.architecture {probe['architecture']!r} is not implemented. "
            "Missing component: that probe architecture."
        )
    if probe["loss"] != "binary_cross_entropy":
        raise ConfigurationError(
            f"probe_run.probe.loss {probe['loss']!r} is not implemented. "
            "Missing component: that loss."
        )
    _expect(probe, "positive_class_awareness", "agreement", "probe_run.probe")
    _expect(probe, "positive_class_consistency", "B", "probe_run.probe")
    _expect(probe, "selection_metric", "validation_cross_entropy", "probe_run.probe")
    if probe["default_profile"] != "linear_v1":
        raise ConfigurationError(
            f"probe_run.probe.default_profile {probe['default_profile']!r} is not implemented. "
            "Missing component: that probe profile."
        )
    methods = _object(
        body["methods"],
        "probe_run.methods",
        ("behavioral_awareness", "knowledge_consistency"),
    )
    evaluation = _object(
        body["evaluation"],
        "probe_run.evaluation",
        ("brier_convention", "ece_bins", "summaries", "confidence_intervals", "save_per_example_predictions"),
    )
    if evaluation["brier_convention"] != "scalar_positive_class":
        raise ConfigurationError(
            "probe_run.evaluation.brier_convention must be scalar_positive_class"
        )
    if evaluation["confidence_intervals"] is not False:
        raise ConfigurationError(
            "probe_run.evaluation.confidence_intervals must be false. "
            "Confidence intervals are not part of the initial configuration."
        )
    if evaluation["save_per_example_predictions"] is not True:
        raise ConfigurationError(
            "probe_run.evaluation.save_per_example_predictions must be true"
        )
    return ProbeRunConfig(
        schema_version=_schema(body, "probe_run"),
        architecture="layerwise_linear",
        loss="binary_cross_entropy",
        positive_class_awareness="agreement",
        positive_class_consistency="B",
        training_seed=_seed(probe, "training_seed", "probe_run.probe"),
        selection_metric="validation_cross_entropy",
        default_profile="linear_v1",
        profile=LINEAR_V1,
        behavioral_awareness=_strings(
            methods,
            "behavioral_awareness",
            "probe_run.methods",
            expected=("linear_agreement", "training_frequency"),
        ),
        knowledge_consistency=_strings(
            methods,
            "knowledge_consistency",
            "probe_run.methods",
            expected=("linear_clean_answer", "contextual_answer"),
        ),
        brier_convention="scalar_positive_class",
        ece_bins=_positive_int(evaluation, "ece_bins", "probe_run.evaluation"),
        summaries=_strings(
            evaluation,
            "summaries",
            "probe_run.evaluation",
            expected=("per_variant", "per_correct_position", "pooled_by_domain"),
        ),
        confidence_intervals=False,
        save_per_example_predictions=True,
    )


def _definitions(raw: Any) -> tuple[InterventionDefinition, ...]:
    if type(raw) is not list or not raw:
        raise ConfigurationError("extraction.interventions.definitions must be a non-empty list")
    parsed = tuple(_definition(item, index) for index, item in enumerate(raw))
    ids = [item.id for item in parsed]
    if len(set(ids)) != len(ids):
        raise ConfigurationError("extraction.interventions.definitions has a duplicate id")
    return parsed


def _definition(raw: Any, index: int) -> InterventionDefinition:
    label = f"extraction.interventions.definitions[{index}]"
    body = _object(raw, label, ("id", "provider"), _DEFINITION_OPTIONAL)
    provider = _text(body["provider"], f"{label}.provider")
    if provider not in _PROVIDERS:
        raise ConfigurationError(
            f"{label}.provider {provider!r} is not implemented. "
            "Missing component: that intervention provider."
        )
    preset = _optional_text(body, "preset", label)
    template = _optional_text(body, "template", label)
    source = _optional_text(body, "source", label)
    join_on = _optional_text(body, "join_on", label)
    text_field = _optional_text(body, "text_field", label)
    instance_field = _optional_text(body, "instance_field", label)
    if provider == "identity" and any((preset, template, source, join_on, text_field, instance_field)):
        raise ConfigurationError(f"{label} identity provider does not take template or source fields")
    if provider == "template":
        if preset is None and template is None:
            raise ConfigurationError(f"{label} template provider needs a preset name or a template string")
        if preset is not None and template is not None:
            raise ConfigurationError(f"{label} template provider takes a preset or a template string, not both")
        if any((source, join_on, text_field, instance_field)):
            raise ConfigurationError(f"{label} template provider does not take supplied-text fields")
    if provider == "supplied_text":
        if preset is not None or template is not None:
            raise ConfigurationError(f"{label} supplied_text provider does not take a template")
        if None in (source, join_on, text_field, instance_field):
            raise ConfigurationError(
                f"{label} supplied_text provider requires source, join_on, text_field, and instance_field"
            )
        if join_on != "base_item_id":
            raise ConfigurationError(f"{label}.join_on must be base_item_id")
    pairing = _optional_text(body, "pairing", label)
    if pairing is not None and pairing != _PAIRING:
        raise ConfigurationError(
            f"{label}.pairing must be {_PAIRING}. Order-specific pairing is not implemented."
        )
    on_missing = _optional_text(body, "on_missing", label)
    if on_missing is not None and on_missing not in ("error", "exclude_pair"):
        raise ConfigurationError(f"{label}.on_missing must be error or exclude_pair")
    family = _optional_text(body, "family", label)
    tags = _strings(body, "tags", label) if "tags" in body else ()
    try:
        parameters = freeze(body["parameters"]) if "parameters" in body else freeze({})
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label}.parameters must be a JSON object") from exc
    if not isinstance(parameters, Mapping):
        raise ConfigurationError(f"{label}.parameters must be a mapping")
    resolved_placement = _placement_value(body["placement"], label) if "placement" in body else None
    return InterventionDefinition(
        id=_text(body["id"], f"{label}.id"),
        provider=provider,
        preset=preset,
        template=template,
        source=source,
        join_on=join_on,
        text_field=text_field,
        instance_field=instance_field,
        placement=resolved_placement,
        pairing=pairing,
        on_missing=on_missing,
        family=family,
        tags=tags,
        parameters=parameters,
    )


def _placement_value(value: Any, label: str) -> Any:
    if type(value) is str:
        return _named_placement(value, label)
    anchor = _object(value, f"{label}.placement", ("mode", "expected_text", "relation"))
    if anchor["mode"] != "character_anchor":
        raise ConfigurationError(f"{label}.placement.mode must be character_anchor")
    if anchor["relation"] not in ("before", "after"):
        raise ConfigurationError(f"{label}.placement.relation must be before or after")
    _text(anchor["expected_text"], f"{label}.placement.expected_text")
    return freeze(anchor)


def _named_placement(value: Any, label: str) -> str:
    if value not in NAMED_PLACEMENTS:
        raise ConfigurationError(
            f"{label} placement must be one of: {', '.join(NAMED_PLACEMENTS)}"
        )
    return str(value)


def _audit(value: Any) -> str | None:
    if value is None:
        return None
    text = _text(value, "extraction.answers.token_audit")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ConfigurationError("extraction.answers.token_audit must be null or a SHA-256 hex digest")
    return text


def _schema(raw: Mapping[str, Any], label: str) -> int:
    value = raw["schema_version"]
    if type(value) is not int or value != SCHEMA_VERSION:
        raise ConfigurationError(f"{label}.schema_version must be {SCHEMA_VERSION}")
    return value


def _object(
    value: Any,
    label: str,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> dict[str, Any]:
    if type(value) is not dict:
        raise ConfigurationError(f"{label} must be a mapping")
    unknown = sorted(set(value) - set(required) - set(optional))
    if unknown:
        raise ConfigurationError(f"{label} has unknown keys: {', '.join(unknown)}")
    missing = [key for key in required if key not in value]
    if missing:
        raise ConfigurationError(f"{label} is missing required keys: {', '.join(missing)}")
    return value


def _take(raw: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in raw:
        raise ConfigurationError(f"{label} is missing required keys: {key}")
    return raw[key]


def _text(value: Any, label: str) -> str:
    if type(value) is not str or value == "":
        raise ConfigurationError(f"{label} must be a non-empty string")
    return value


def _optional_text(raw: Mapping[str, Any], key: str, label: str) -> str | None:
    if key not in raw or raw[key] is None:
        return None
    return _text(raw[key], f"{label}.{key}")


def _bool(raw: Mapping[str, Any], key: str, label: str) -> bool:
    value = raw[key]
    if type(value) is not bool:
        raise ConfigurationError(f"{label}.{key} must be an explicit boolean")
    return value


def _strings(
    raw: Mapping[str, Any],
    key: str,
    label: str,
    *,
    expected: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    value = raw[key]
    if type(value) is not list or not value or any(type(item) is not str or item == "" for item in value):
        raise ConfigurationError(f"{label}.{key} must be a non-empty list of strings")
    if len(set(value)) != len(value):
        raise ConfigurationError(f"{label}.{key} contains a duplicate")
    chosen = tuple(value)
    if expected is not None and chosen != expected:
        raise ConfigurationError(f"{label}.{key} must be {list(expected)}")
    return chosen


def _fraction(raw: Mapping[str, Any], key: str, label: str) -> float:
    value = raw[key]
    if type(value) is bool or type(value) not in (int, float):
        raise ConfigurationError(f"{label}.{key} must be a fraction")
    number = float(value)
    if not math.isfinite(number) or not 0 < number < 1:
        raise ConfigurationError(f"{label}.{key} must be strictly between 0 and 1")
    return number


def _probability(value: Any, label: str) -> float:
    if type(value) is bool or type(value) not in (int, float):
        raise ConfigurationError(f"{label} must be a probability")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ConfigurationError(f"{label} must be between 0 and 1")
    return number


def _seed(raw: Mapping[str, Any], key: str, label: str) -> int:
    value = raw[key]
    if type(value) is not int or value < 0:
        raise ConfigurationError(f"{label}.{key} must be a non-negative integer")
    return value


def _positive_int(raw: Mapping[str, Any], key: str, label: str) -> int:
    value = raw[key]
    if type(value) is not int or value < 1:
        raise ConfigurationError(f"{label}.{key} must be a positive integer")
    return value


def _expect(raw: Mapping[str, Any], key: str, expected: str, label: str) -> None:
    if raw[key] != expected:
        raise ConfigurationError(f"{label}.{key} must be {expected!r}")
