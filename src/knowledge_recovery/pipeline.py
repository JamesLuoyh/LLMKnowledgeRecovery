"""Stage graph. The command-line module only dispatches into this table.

Filter counts are written before any probe is fit. A method that cannot fit
is recorded and the other tasks still run. Extraction-cache misses on a
probe-only stage are reported and do not construct a model.
"""

from __future__ import annotations

import dataclasses
import json
import os
import string
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy

from knowledge_recovery.answers.canonical import (
    CanonicalTokens,
    score_binary_summary,
    summarize_binary_logits,
)
from knowledge_recovery.cache.store import ExtractionStore
from knowledge_recovery.cli import StageRequest
from knowledge_recovery.config import ExperimentConfig, load_config, validate_before_expensive_work
from knowledge_recovery.datasets.distractor_ranking import (
    PREFERRED_SELECTIONS,
    RANKING_VERSION,
    distinct_choices,
    rank_items,
)
from knowledge_recovery.datasets.mmlu import recover_mmlu
from knowledge_recovery.datasets.sycophancy import recover_sycophancy
from knowledge_recovery.errors import (
    SUPPORTED_POSITION_SELECTOR,
    CacheIncomplete,
    CacheIncompatible,
    KnowledgeRecoveryError,
    MissingLocalAsset,
)
from knowledge_recovery.evaluation.behavior import BehaviorRow, behavior_document, behavior_tables
from knowledge_recovery.evaluation.evaluate import EvaluationRow, evaluate_saved
from knowledge_recovery.evaluation.metrics import DEFAULT_ECE_BINS
from knowledge_recovery.extraction.identity import build_row_key
from knowledge_recovery.extraction.positions import last_nonpadding_index, select_last_nonpadding
from knowledge_recovery.extraction.runner import run_extraction, unique_missing_rows
from knowledge_recovery.interventions.materialize import materialize_interventions, specs_from_config
from knowledge_recovery.methods.awareness import RULE as TRAINING_FREQUENCY_RULE
from knowledge_recovery.methods.awareness import TrainingFrequency, fit_training_frequency
from knowledge_recovery.methods.consistency import ANSWER_SCORE as CONTEXTUAL_ANSWER_SCORE
from knowledge_recovery.methods.consistency import IMPLEMENTATION_VERSION as CONTEXTUAL_IMPLEMENTATION_VERSION
from knowledge_recovery.methods.consistency import contextual_answer_method
from knowledge_recovery.models.assets import activate_offline, resolve_local_assets
from knowledge_recovery.models.qwen import (
    MODEL_ADAPTER_VERSION,
    QwenContinuationTokenizer,
    forward_padded,
    load_frozen_model,
    numeric_runtime_signature,
)
from knowledge_recovery.prompts.render import RENDERER_VERSION, render_orders
from knowledge_recovery.prompts.token_audit import (
    MappedContinuationTokenizer,
    PromptExample,
    audit_canonical_tokens,
    extraction_manifest_audit,
    read_token_audit,
    write_token_audit,
)
from knowledge_recovery.provenance import canonical_json, content_digest, identify, model_signature, prompt_contract, run_id
from knowledge_recovery.records import non_default_fields
from knowledge_recovery.reports.export import export_report
from knowledge_recovery.tasks import awareness_task_spec, consistency_task_spec
from knowledge_recovery.tasks.base import AWARENESS_TASK_ID, CONSISTENCY_TASK_ID
from knowledge_recovery.training.fit import TRAINING_IMPLEMENTATION_VERSION, fit_linear_probe
from knowledge_recovery.training.manifest import write_selection_manifest
from knowledge_recovery.training.select import select_probe
from knowledge_recovery.views.build import ScoredRow, build_task_views, view_summary
from knowledge_recovery.views.filters import (
    CleanOrder,
    apply_filter_gate,
    apply_variant_gate,
    write_misreport_report,
)
from knowledge_recovery.views.splits import freeze_grouped_split

STAGE_ORDER = (
    "validate",
    "rank-distractors",
    "prepare",
    "audit-tokens",
    "extract",
    "diagnose",
    "build-view",
    "train",
    "evaluate",
    "report",
)
# First attempted extraction batch. An out-of-memory error halves it; it is not in the row key.
EXTRACTION_BATCH_SIZE = 32
# Row fields an extract worker's forward pass reads. The rendered text stays in the parent.
EXTRACT_WORKER_FIELDS = (
    "row_key", "input_ids", "attention_mask", "selected_position", "question", "order_id", "variant_id",
)
METHOD_BINDINGS = (
    ("behavioral_awareness", "linear_agreement", "linear"),
    ("behavioral_awareness", "training_frequency", "training_frequency"),
    ("knowledge_consistency", "linear_clean_answer", "linear"),
    ("knowledge_consistency", "contextual_answer", "contextual_answer"),
)


class MethodUnavailable(KnowledgeRecoveryError):
    """A trainable method has no usable view. Other methods still run."""

    def __init__(
        self,
        *,
        task_id: str,
        method_id: str,
        reason: str,
        label_counts: Mapping[str, int] | None = None,
    ) -> None:
        self.task_id = task_id
        self.method_id = method_id
        self.reason = reason
        self.label_counts = {} if label_counts is None else dict(label_counts)
        lines = [
            f"MethodUnavailable: {task_id}/{method_id} is unavailable.",
            f"Reason: {reason}",
        ]
        if self.label_counts:
            shown = ", ".join(f"{key}={value}" for key, value in sorted(self.label_counts.items()))
            lines.append(f"Label counts: {shown}")
        super().__init__("\n".join(lines))


class PipelineIncomplete(KnowledgeRecoveryError):
    """Every requested method was recorded, and at least one did not finish."""

    def __init__(self, summary: Mapping[str, Any]) -> None:
        self.summary = summary
        lines = [
            "PipelineIncomplete: requested work did not all finish.",
            f"Overall: {summary['overall']}",
        ]
        for task in summary["tasks"]:
            lines.append(f"{task['task_id']}: {task['status']} {task['detail']}".rstrip())
        super().__init__("\n".join(lines))


def execute_tasks(jobs: Sequence[tuple[str, Callable[[], str]]]) -> dict[str, Any]:
    """Run each job. ``MethodUnavailable`` is recorded; other errors propagate."""
    results = []
    for name, job in jobs:
        try:
            detail = job()
        except MethodUnavailable as exc:
            results.append({"task_id": name, "status": "unavailable", "detail": str(exc)})
        else:
            results.append({"task_id": name, "status": "complete", "detail": detail})
    overall = "complete" if results and all(item["status"] == "complete" for item in results) else "incomplete"
    return {"tasks": results, "overall": overall}


def validate_stage(request: StageRequest) -> None:
    config = _config(request)
    validate_before_expensive_work(config)
    activate_offline(_storage(config) / "assets" / "huggingface")


def prepare_stage(request: StageRequest) -> None:
    """Normalize, freeze splits, then materialize and render. No model is loaded."""
    config = _config(request)
    storage = _storage(config)
    activate_offline(storage / "assets" / "huggingface")
    rows = _read_rows(Path(config.extraction.dataset_path))
    (storage / "normalized").mkdir(parents=True, exist_ok=True)
    source_identity = _file_digest(Path(config.extraction.dataset_path))
    if config.extraction.dataset_adapter == "mmlu":
        selection = config.extraction.distractor_selection
        normalized = recover_mmlu(
            rows,
            source_identity=source_identity,
            conflict_directory=storage / "normalized",
            distractor_seed=config.extraction.distractor_seed,
            distractor_selection=selection,
            preferences=(
                _distractor_preferences(config, source_identity)
                if selection in PREFERRED_SELECTIONS
                else None
            ),
        )
    else:
        normalized = recover_sycophancy(
            rows,
            source_identity=source_identity,
            conflict_directory=storage / "normalized",
        )
    # The split universe is the base-group ids. Intervention coverage is not an input.
    split = freeze_grouped_split(
        [item.duplicate_group_id for item in normalized.items],
        train_fraction=config.analysis_view.train_fraction,
        val_fraction=config.analysis_view.val_fraction,
        test_fraction=config.analysis_view.test_fraction,
        split_seed=config.analysis_view.split_seed,
    )
    specs = specs_from_config(config.extraction)
    materialized = materialize_interventions(normalized.items, specs)
    by_id = {item.duplicate_group_id: item for item in normalized.items}
    rendered = []
    for instance in materialized.instances:
        item = by_id[instance.base_item_id]
        for prompt in render_orders(
            item,
            instance,
            instruction=config.extraction.instruction,
            chat=config.extraction.chat,
            system_message=config.extraction.system_message,
        ):
            rendered.append({
                "base_group_id": prompt.base_item_id,
                "order_id": prompt.order_id,
                "variant_id": prompt.variant_id,
                "instance_id": prompt.intervention_instance_id,
                "correct_label": prompt.correct_label,
                "ordered_answer_texts": list(prompt.ordered_answer_texts),
                "user_text": prompt.user_text,
                "question": item.question,
                "blocks": _plain(prompt.blocks),
                "split": split.group_to_split[prompt.base_item_id],
            })
    _write_json(storage / "normalized" / "items.json", {
        "raw_source_rows": normalized.raw_source_rows,
        "recovered_base_items": normalized.recovered_base_items,
        "duplicate_rows": normalized.duplicate_rows,
        "invalid_rows": [_plain(item.payload()) for item in normalized.invalid_rows],
        "exclusions": [_plain(item.payload()) for item in normalized.exclusions],
        "template_counts": normalized.template_counts(),
        "items": [_plain(item.payload()) for item in normalized.items],
    })
    _write_json(storage / "splits" / "split.json", identify(split).sidecar())
    _write_json(storage / "interventions" / "manifest.json", {
        "digest": materialized.digest,
        "instances": [_plain(instance.payload()) for instance in materialized.instances],
    })
    _write_json(storage / "prompts" / "rendered.json", rendered)


def rank_distractors_stage(request: StageRequest) -> None:
    """Score every MMLU choice with the model for most/least preferred distractor selection.

    Each question is asked with all of its choices in every cyclic order. Only
    the option-letter logits at the last input token are kept. The output is
    read by ``prepare`` and is not part of the extraction cache. With more
    than one visible GPU, each GPU ranks every n-th question in its own process.
    """
    config = _config(request)
    if config.extraction.dataset_adapter != "mmlu":
        raise KnowledgeRecoveryError("rank-distractors applies only to the mmlu adapter")
    validate_before_expensive_work(config)
    storage = _storage(config)
    activate_offline(storage / "assets" / "huggingface")
    source_identity = _file_digest(Path(config.extraction.dataset_path))
    (storage / "normalized").mkdir(parents=True, exist_ok=True)
    candidates = _rank_candidates(config, source_identity)
    tokenizer = _tokenizer(Path(config.extraction.model_path))
    encode = _prompt_encoder(config, tokenizer)
    width = max(len(distinct_choices(item)) for item in candidates)
    probe = encode(candidates[0].question)
    letter_ids: dict[str, int] = {}
    for letter in string.ascii_uppercase[:width]:
        ids = tokenizer.continuation_ids(probe, letter)
        if len(ids) != 1:
            raise KnowledgeRecoveryError(f"option letter {letter!r} is not a single token: {list(ids)}")
        letter_ids[letter] = int(ids[0])
    ordered_ids = list(letter_ids.values())
    devices = _visible_devices(config)
    from tqdm import tqdm

    total = sum(len(distinct_choices(item)) for item in candidates)
    with tqdm(total=total, desc="rank-distractors", unit="prompt", dynamic_ncols=True) as progress:
        if len(devices) > 1:
            jobs = [
                (device, (request.config, shard, len(devices), ordered_ids))
                for shard, device in enumerate(devices)
            ]
            ranked = {}
            for part in _run_on_devices("rank-distractors", _rank_worker, jobs, progress.update):
                ranked.update(part)
        else:
            ranked = _rank_shard(config, candidates, encode, ordered_ids, progress.update)
    _write_json(storage / DISTRACTOR_PREFERENCES, {
        **_ranking_identity(config, source_identity),
        "letter_token_ids": letter_ids,
        "orders": "cyclic_rotations",
        "items": ranked,
    })


def audit_tokens_stage(request: StageRequest) -> None:
    """Validate the continuation contract and record it. This is the recheck command."""
    config = _config(request)
    _audit(config, recheck=True)


def extract_stage(request: StageRequest) -> None:
    config = _config(request)
    storage = _storage(config)
    collection_path = storage / "extractions" / "collection.json"
    if not Path(config.extraction.model_path).exists() and collection_path.is_file():
        _reuse_collection(storage, collection_path, config)
        return
    rendered = _read_rendered(storage)
    audit = _audit(config, recheck=False, rendered=rendered)
    assets = resolve_local_assets(config.extraction.model_path)
    signature = model_signature(
        weight_content_digest=assets.weight_content_digest,
        model_config_digest=assets.model_config_digest,
        tokenizer_content_digest=assets.tokenizer_content_digest,
        model_adapter_version=MODEL_ADAPTER_VERSION,
    ).digest
    contract = _prompt_contract(config, audit).digest
    model_config = json.loads(Path(config.extraction.model_path, "config.json").read_text(encoding="utf-8"))
    runtime = numeric_runtime_signature(model_config, precision=config.extraction.precision)
    prepared = _tokenize(config, audit, rendered, {
        "model_signature": signature,
        "prompt_contract_digest": contract,
        "position_selector": SUPPORTED_POSITION_SELECTOR,
        "execution_precision": config.extraction.precision,
        "attention_backend": str(runtime["attention_backend"]),
        "numeric_runtime_signature": runtime,
        "reporting": _reporting(config),
    })
    del rendered
    store = ExtractionStore(storage)
    statuses = store.row_statuses([row["row_key"] for row in prepared])
    missing = [row["row_key"] for row in prepared if statuses[row["row_key"]] != "indexed"]

    canonical_ids = (int(audit.token_ids["A"]), int(audit.token_ids["B"]))
    devices = _visible_devices(config) if missing else []
    if len(devices) > 1:
        # Every GPU stores a disjoint share of the missing rows; the lookup below reads them all.
        pending = [
            {field: row[field] for field in EXTRACT_WORKER_FIELDS}
            for row in unique_missing_rows(prepared, missing)
        ]
        from tqdm import tqdm

        jobs = [
            (device, (request.config, pending[shard::len(devices)], canonical_ids))
            for shard, device in enumerate(devices)
        ]
        with tqdm(total=len(pending), desc="extract", unit="row", dynamic_ncols=True) as progress:
            sizes = _run_on_devices("extract", _extract_worker, jobs, progress.update)
        del pending
        probed = run_extraction(
            prepared,
            store=store,
            model_factory=None,
            forward_batch=None,
            initial_batch_size=EXTRACTION_BATCH_SIZE,
            precision=config.extraction.precision,
            probe_only=True,
        )
        result = dataclasses.replace(
            probed,
            constructed_model=True,
            batch_sizes=tuple(size for part in sizes for size in part),
        )
    elif missing:
        result = run_extraction(
            prepared,
            store=store,
            model_factory=_model_factory(config),
            forward_batch=_extract_forward(model_config, canonical_ids),
            initial_batch_size=EXTRACTION_BATCH_SIZE,
            precision=config.extraction.precision,
        )
    else:
        result = run_extraction(
            prepared,
            store=store,
            model_factory=_model_factory(config),
            forward_batch=_extract_forward(model_config, canonical_ids),
            initial_batch_size=EXTRACTION_BATCH_SIZE,
            precision=config.extraction.precision,
            probe_only=True,
        )
    store.write_collection_manifest(collection_path, {
        "model_signature": signature,
        "prompt_contract_digest": contract,
        "request_signature": _extraction_request_signature(config),
        "canonical_token_ids": {label: int(audit.token_ids[label]) for label in ("A", "B")},
        "audit_id": audit.audit_id,
        "subset_label": model_config.get("subset_label"),
        "numeric": {
            "execution_precision": result.precision,
            "residual_storage_dtype": "bf16",
            "logits_storage_dtype": "float32",
            "model_mode": "eval",
            "gradients_enabled": False,
            "dropout": False,
            "extraction_augmentation": False,
            "probe_seed_enters_inference": False,
            "attention_backend": runtime["attention_backend"],
            "runtime_signature": runtime,
            "execution_batch_sizes": list(result.batch_sizes),
            "constructed_model": result.constructed_model,
        },
        "resolved_audit": extraction_manifest_audit(config.extraction.token_audit, audit),
        "rows": [
            {
                "row_key": row["row_key"],
                "base_group_id": row["base_group_id"],
                "split": row["split"],
                "variant_id": row["variant_id"],
                "instance_id": row["instance_id"],
                "order_id": row["order_id"],
                "question": row["question"],
                "correct_label": row["correct_label"],
            }
            for row in prepared
        ],
    })


def _model_factory(config: ExperimentConfig) -> Callable[[], object]:
    def factory() -> object:
        return load_frozen_model(config.extraction.model_path, precision=config.extraction.precision)

    return factory


def _extract_forward(model_config: Mapping[str, Any], canonical_ids: tuple[int, int]) -> Callable[..., Any]:
    """Forward a batch and keep the A/B logit summary in place of the full logits."""
    canonical = CanonicalTokens(A=canonical_ids[0], B=canonical_ids[1])

    def forward(model: object, rows: Sequence[Mapping[str, Any]], *, batch_size: int, precision: str):
        if model_config.get("logit_rule") == "integration_subset_v1":
            from knowledge_recovery.integration import subset_forward

            produced = subset_forward(model, rows, batch_size=batch_size, precision=precision)
        else:
            produced = forward_padded(model, rows, batch_size=batch_size, precision=precision)
        summaries = summarize_binary_logits(numpy.stack([item["logits"] for item in produced]), canonical)
        return tuple({**item, "logits": summary} for item, summary in zip(produced, summaries, strict=True))

    return forward


def _extract_worker(
    config_path: str,
    rows: list[dict[str, Any]],
    canonical_ids: tuple[int, int],
    *,
    progress: Callable[[int], None],
) -> tuple[int, ...]:
    """Store one GPU's share of the missing rows. Returns the batch sizes used."""
    config = load_config(config_path)
    model_config = json.loads(Path(config.extraction.model_path, "config.json").read_text(encoding="utf-8"))
    result = run_extraction(
        rows,
        store=ExtractionStore(_storage(config)),
        model_factory=_model_factory(config),
        forward_batch=_extract_forward(model_config, canonical_ids),
        initial_batch_size=EXTRACTION_BATCH_SIZE,
        precision=config.extraction.precision,
        progress=progress,
    )
    return result.batch_sizes


def diagnose_stage(request: StageRequest) -> None:
    """Score cached logits and write the fixed four-step counts. No basis is chosen."""
    config = _config(request)
    storage = _storage(config)
    activate_offline(storage / "assets" / "huggingface")
    collection = json.loads((storage / "extractions" / "collection.json").read_text(encoding="utf-8"))
    # The collection's split copy can be stale after a re-split, so splits/split.json decides.
    _, group_splits = _load_split(storage)
    rows = [
        {**row, "split": _split_of(group_splits, row["base_group_id"])}
        for row in collection["rows"]
    ]
    store = ExtractionStore(storage)
    keys = [row["row_key"] for row in rows]
    references = {
        item.extraction_row_key: item
        for item in store.lookup(keys, model_factory=_refuse_model, extraction_command="kr extract --config cache")
    }
    tokens = CanonicalTokens(
        A=int(collection["canonical_token_ids"]["A"]),
        B=int(collection["canonical_token_ids"]["B"]),
    )
    scored = []
    for row in rows:
        logits = store.read_logits_row(references[row["row_key"]])
        scores = score_binary_summary([float(value) for value in logits.tolist()], tokens)
        raw_label = None
        if scores.unrestricted_greedy_token_id == tokens.A:
            raw_label = "A"
        elif scores.unrestricted_greedy_token_id == tokens.B:
            raw_label = "B"
        scored.append({**row, "scores": scores, "raw_label": raw_label})
    clean_orders = [
        CleanOrder(
            base_group_id=row["base_group_id"],
            split=row["split"],
            order_id=row["order_id"],
            raw_ab_probabilities=row["scores"].raw_ab_probabilities,
            canonical_ab_probabilities=row["scores"].canonical_ab_probabilities,
            canonical_label_mass=row["scores"].canonical_label_mass,
            invalid_label=row["scores"].invalid_label,
        )
        for row in scored
        if row["variant_id"] == config.analysis_view.filter_variant
    ]
    outcome = apply_filter_gate(
        clean_orders,
        output_directory=storage / "analyses" / "filter",
        min_correct_prob=config.analysis_view.min_correct_prob,
        variant_id=config.analysis_view.filter_variant,
    )
    eligible = outcome.eligible_base_group_ids
    variant_digest = None
    variant_directory = storage / "analyses" / "filter"
    intervened = [
        variant_id
        for variant_id in dict.fromkeys(row["variant_id"] for row in scored)
        if variant_id != config.analysis_view.filter_variant
    ]
    orders_by_variant = {
        variant_id: [
            CleanOrder(
                base_group_id=row["base_group_id"],
                split=row["split"],
                order_id=row["order_id"],
                raw_ab_probabilities=row["scores"].raw_ab_probabilities,
                canonical_ab_probabilities=row["scores"].canonical_ab_probabilities,
                canonical_label_mass=row["scores"].canonical_label_mass,
                invalid_label=row["scores"].invalid_label,
            )
            for row in scored
            if row["variant_id"] == variant_id
        ]
        for variant_id in intervened
    }
    write_misreport_report(
        clean_orders,
        orders_by_variant,
        output_directory=variant_directory,
        min_correct_prob=config.analysis_view.min_correct_prob,
        variant_steps=config.analysis_view.variant_stats,
        require_confident_on_every_variant=config.analysis_view.require_confident_on_every_variant,
    )
    if config.analysis_view.variant_stats:
        variant_outcome = apply_variant_gate(
            orders_by_variant,
            eligible,
            output_directory=variant_directory,
            min_correct_prob=config.analysis_view.min_correct_prob,
            require_confident_on_every_variant=config.analysis_view.require_confident_on_every_variant,
        )
        variant_digest = variant_outcome.report_digest
        eligible = variant_outcome.eligible_base_group_ids
    else:
        for suffix in ("json", "txt"):
            (variant_directory / f"variant_filter_report.{suffix}").unlink(missing_ok=True)
    clean_labels = {
        (row["base_group_id"], row["order_id"]): row["scores"].canonical_greedy_label
        for row in scored
        if row["variant_id"] == "clean"
    }
    behavior_instances = _behavior_instance_labels(scored)
    behavior_rows = [
        BehaviorRow(
            base_group_id=row["base_group_id"],
            split=row["split"],
            variant_id=row["variant_id"],
            instance_id=behavior_instances[(row["base_group_id"], row["variant_id"], row["instance_id"])],
            order_id=row["order_id"],
            canonical_label=row["scores"].canonical_greedy_label,
            raw_token_label=row["raw_label"],
            raw_token_id=row["scores"].unrestricted_greedy_token_id,
        )
        for row in scored
    ]
    tables = behavior_tables(
        behavior_rows,
        raw_retained=eligible,
        canonical_retained=eligible,
        clean_variant="clean",
    )
    _write_json(storage / "analyses" / "behavior.json", {
        "population_rule": "fixed_four_step_eligible",
        **behavior_document(tables),
    })
    _write_json(storage / "analyses" / "scored.json", {
        "filter_digest": outcome.report_digest,
        "variant_filter_digest": variant_digest,
        "require_confident_on_every_variant": config.analysis_view.require_confident_on_every_variant,
        "eligible_base_group_ids": list(eligible),
        "rows": [
            {
                "row_key": row["row_key"],
                "base_group_id": row["base_group_id"],
                "split": row["split"],
                "variant_id": row["variant_id"],
                "instance_id": row["instance_id"],
                "order_id": row["order_id"],
                "clean_canonical_label": clean_labels[(row["base_group_id"], row["order_id"])],
                "contextual_canonical_label": row["scores"].canonical_greedy_label,
                "contextual_canonical_probabilities": list(row["scores"].canonical_ab_probabilities),
            }
            for row in scored
        ],
    })
    if collection.get("subset_label"):
        _write_json(storage / "analyses" / "subset.json", {"subset_label": collection["subset_label"]})


def build_view_stage(request: StageRequest) -> None:
    config = _config(request)
    storage = _storage(config)
    scored = json.loads((storage / "analyses" / "scored.json").read_text(encoding="utf-8"))
    split, group_splits = _load_split(storage)
    rows = [
        ScoredRow(**{**row, "split": _split_of(group_splits, row["base_group_id"])})
        for row in scored["rows"]
    ]
    views = build_task_views(
        rows,
        eligible=scored["eligible_base_group_ids"],
        train_variants=config.analysis_view.train_variants,
        eval_variants=config.analysis_view.eval_variants,
        split_digest=split["digest"],
        filter_digest=scored["filter_digest"],
    )
    index = {}
    for name, artifact in views.items():
        record = artifact.sidecar()
        path = storage / "views" / f"{artifact.digest}.json"
        _write_json(path, {"name": name, "view_id": artifact.digest, "sidecar": record})
        index[name] = {"view_id": artifact.digest, "path": path.name}
    _write_json(storage / "views" / "index.json", index)
    print(view_summary(views, skipped_ood=not config.analysis_view.eval_variants))


def train_stage(request: StageRequest) -> None:
    """Fit methods into a content-addressed run bundle."""
    config = _config(request)
    storage = _storage(config)
    report = storage / "analyses" / "filter" / "filter_report.json"
    if not report.is_file():
        raise KnowledgeRecoveryError(
            "Filter counts are written before training.\n"
            f"Missing report: {report}\n"
            "Run kr diagnose --config <path> first. There is no raw-versus-canonical choice to make."
        )
    activate_offline(storage / "assets" / "huggingface")
    index = json.loads((storage / "views" / "index.json").read_text(encoding="utf-8"))
    loaded = {name: json.loads((storage / "views" / entry["path"]).read_text(encoding="utf-8")) for name, entry in index.items()}
    method_runs = _method_run_ids(config, loaded)
    bundle_id = content_digest({"kind": "probe_run_bundle", "schema": 1, "method_run_ids": method_runs})
    bundle = storage / "runs" / bundle_id
    attempt_id = content_digest({"kind": "probe_training_attempt", "schema": 1, "nonce": uuid.uuid4().hex})
    attempt = bundle / attempt_id
    _write_json(attempt / "views.json", {
        name: value for name, value in loaded.items()
        if name.endswith("/train") or name.endswith("/validation")
    })
    jobs = [
        (
            f"{task_id}/{config_name}",
            lambda task_id=task_id, method_id=method_id, config_name=config_name: _train_method(
                config, storage, loaded, task_id, method_id, config_name, attempt
            ),
        )
        for task_id, config_name, method_id in METHOD_BINDINGS
    ]
    summary = execute_tasks(jobs)
    manifest = {
        "kind": "probe_training_attempt",
        "schema": 1,
        "bundle_id": bundle_id,
        "attempt_id": attempt_id,
        "method_run_ids": method_runs,
        "summary": summary,
    }
    _write_json(attempt / "manifest.json", manifest)
    _write_json(bundle / "manifest.json", {
        "kind": "probe_run_bundle",
        "schema": 1,
        "bundle_id": bundle_id,
        "method_run_ids": method_runs,
        "current_attempt_id": attempt_id,
        "path": f"{attempt_id}/manifest.json",
    })
    _write_json(storage / "runs" / "summary.json", {
        **summary, "bundle_id": bundle_id, "attempt_id": attempt_id,
    })
    _write_json(storage / "runs" / "current.json", {
        "bundle_id": bundle_id, "attempt_id": attempt_id,
        "path": f"{bundle_id}/{attempt_id}/manifest.json",
    })
    if summary["overall"] != "complete":
        raise PipelineIncomplete(summary)


def evaluate_stage(request: StageRequest) -> None:
    """Evaluate saved method state and cached tensors without model construction."""
    run_directory, storage, manifest = _resolve_run(request)
    activate_offline(storage / "assets" / "huggingface")
    views = _load_current_views(storage)
    rows, evaluation_view_id = _evaluation_rows(storage, views)
    frequency = _load_frequency(run_directory)
    contextual = contextual_answer_method(task_spec_digest=identify(consistency_task_spec()).digest)
    report = evaluate_saved(
        rows,
        awareness_manifest=run_directory / AWARENESS_TASK_ID / "linear" / "selection_manifest.json",
        consistency_manifest=run_directory / CONSISTENCY_TASK_ID / "linear" / "selection_manifest.json",
        frequency=frequency,
        contextual=contextual,
        evaluation_view_id=evaluation_view_id,
        ece_bins=_config(request).probe_run.ece_bins if request.config else DEFAULT_ECE_BINS,
    )
    evaluation_bundle_id = content_digest({
        "kind": "evaluation_bundle",
        "schema": 1,
        "run_bundle_id": manifest["bundle_id"],
        "evaluation_ids": report["evaluation_ids"],
    })
    report = {
        **report,
        "evaluation_bundle_id": evaluation_bundle_id,
        "run_bundle_id": manifest["bundle_id"],
        "run_attempt_id": manifest["attempt_id"],
    }
    _write_json(storage / "evaluations" / evaluation_bundle_id / "report.json", report)
    pointer = {
        "evaluation_bundle_id": evaluation_bundle_id,
        "run_bundle_id": manifest["bundle_id"],
        "path": f"{evaluation_bundle_id}/report.json",
    }
    _write_json(run_directory / "evaluations" / "current.json", pointer)
    _write_json(storage / "evaluations" / "current.json", pointer)


def report_stage(request: StageRequest) -> None:
    run_directory, storage, manifest = _resolve_run(request)
    pointer_path = run_directory / "evaluations" / "current.json"
    if not pointer_path.is_file():
        summary = {
            "tasks": [{"task_id": "report", "status": "unavailable", "detail": "saved evaluation report is missing"}],
            "overall": "incomplete",
        }
        _write_json(storage / "reports" / manifest["bundle_id"] / "summary.json", summary)
        raise PipelineIncomplete(summary)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("run_bundle_id") != manifest["bundle_id"]:
        raise KnowledgeRecoveryError("evaluation pointer does not belong to the requested run bundle")
    path = storage / "evaluations" / pointer["path"]
    if not path.is_file():
        raise KnowledgeRecoveryError(f"evaluation report does not exist: {path}")
    export_report(
        json.loads(path.read_text(encoding="utf-8")),
        storage / "reports" / pointer["evaluation_bundle_id"],
    )


def _train_method(config, storage, views, task_id: str, method_id: str, config_name: str, bundle: Path) -> str:
    del config_name
    if method_id == "contextual_answer":
        return "selected_layer null; no fit"
    train = views[f"{task_id}/train"]
    train_rows = train["sidecar"]["record"]["row_references"]
    if method_id == "training_frequency":
        state = fit_training_frequency([int(row["target"]) for row in train_rows], training_view_id=train["view_id"])
        if not state.available:
            raise MethodUnavailable(task_id=task_id, method_id=method_id, reason=state.unavailable_reason or "empty training view")
        _write_json(bundle / task_id / "training_frequency.json", {
            "training_view_id": state.training_view_id, "probability": state.probability,
            "count": state.count, "positive_count": state.positive_count,
            "available": state.available, "unavailable_reason": state.unavailable_reason,
            "degenerate": state.degenerate, "degeneracy": state.degeneracy, "selected_layer": None,
        })
        return f"frequency {state.probability}" + (f" degenerate {state.degeneracy}" if state.degenerate else "")
    validation = views[f"{task_id}/validation"]
    valid_rows = validation["sidecar"]["record"]["row_references"]
    if not train_rows:
        raise MethodUnavailable(task_id=task_id, method_id=method_id, reason="empty training view")
    if not valid_rows:
        raise MethodUnavailable(task_id=task_id, method_id=method_id, reason="empty validation view")
    targets = [int(row["target"]) for row in train_rows]
    counts = Counter(targets)
    if len(counts) < 2:
        raise MethodUnavailable(task_id=task_id, method_id=method_id, reason="one-class training target",
                                label_counts={str(label): int(counts.get(label, 0)) for label in (0, 1)})
    store = ExtractionStore(storage)
    train_residuals = _residuals(store, train_rows)
    valid_residuals = _residuals(store, valid_rows)
    fits = fit_linear_probe(
        train_residuals, targets, valid_residuals, [int(row["target"]) for row in valid_rows],
        layers=tuple(range(1, train_residuals.shape[1] + 1)), task_id=task_id,
        training_seed=config.probe_run.training_seed, train_domains=("id",) * len(targets),
        validation_domains=("id",) * len(valid_rows),
        processes=LINEAR_FIT_PROCESSES,
    )
    selection = select_probe(fits)
    write_selection_manifest(bundle / task_id / method_id, selection)
    return f"selected layer {selection.selected_layer}"


def _method_run_ids(config, views: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    profile = {
        "optimizer": config.probe_run.profile.optimizer, "learning_rate": config.probe_run.profile.learning_rate,
        "weight_decay": config.probe_run.profile.weight_decay, "weight_decay_on": config.probe_run.profile.weight_decay_on,
        "betas": list(config.probe_run.profile.betas), "epsilon": config.probe_run.profile.epsilon,
        "minibatch_rows": config.probe_run.profile.batch_size, "max_epochs": config.probe_run.profile.max_epochs,
        "early_stopping_patience": config.probe_run.profile.early_stopping_patience,
    }
    identifiers: dict[str, str] = {}
    for task_id, config_name, method_id in METHOD_BINDINGS:
        name = f"{task_id}/{config_name}"
        train_view = views[f"{task_id}/train"]["view_id"]
        if method_id == "linear":
            validation_view = views[f"{task_id}/validation"]["view_id"]
            task = awareness_task_spec() if task_id == AWARENESS_TASK_ID else consistency_task_spec()
            identifiers[name] = run_id(
                train_view_id=train_view, validation_view_id=validation_view, task_spec=_plain(task.payload()),
                architecture={"name": "layerwise_linear"}, feature_spec={"layers": "all", "position": "last_nonpadding_input_token"},
                method_id=config_name, task_adapter_version=identify(task).digest, method_parameters={},
                preprocessing_spec={"mode": "standardize"}, loss_spec={"name": config.probe_run.loss},
                optimizer_spec=profile, training_seed=config.probe_run.training_seed,
                stopping_rules={"max_epochs": profile["max_epochs"], "patience": profile["early_stopping_patience"]},
                selection_rules={"metric": config.probe_run.selection_metric, "tie": "earliest_then_shallower"},
                training_implementation_version=TRAINING_IMPLEMENTATION_VERSION,
                training_numeric_signature={"feature_dtype": "float32", "statistics_dtype": "float64"},
            ).digest
        elif method_id == "training_frequency":
            identifiers[name] = content_digest({"kind": "method_run", "schema": 1, "method": method_id,
                                                "training_view_id": train_view, "rule": TRAINING_FREQUENCY_RULE})
        else:
            identifiers[name] = content_digest({"kind": "method_run", "schema": 1, "method": method_id,
                                                "task_spec_digest": identify(consistency_task_spec()).digest,
                                                "answer_score": CONTEXTUAL_ANSWER_SCORE,
                                                "implementation_version": CONTEXTUAL_IMPLEMENTATION_VERSION})
    return identifiers


def _load_current_views(storage: Path) -> dict[str, Any]:
    index_path = storage / "views" / "index.json"
    if not index_path.is_file():
        raise KnowledgeRecoveryError(f"view index does not exist: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return {
        name: json.loads((storage / "views" / entry["path"]).read_text(encoding="utf-8"))
        for name, entry in index.items()
    }


def _evaluation_rows(storage: Path, views: Mapping[str, Mapping[str, Any]]) -> tuple[tuple[EvaluationRow, ...], str]:
    store = ExtractionStore(storage)
    result: list[EvaluationRow] = []
    view_ids: dict[str, str] = {}
    for suffix, domain in (("evaluation_id", "id"), ("evaluation_ood", "ood")):
        if domain == "ood" and f"{AWARENESS_TASK_ID}/{suffix}" not in views:
            continue
        awareness = views[f"{AWARENESS_TASK_ID}/{suffix}"]
        consistency = views[f"{CONSISTENCY_TASK_ID}/{suffix}"]
        view_ids[f"{AWARENESS_TASK_ID}/{suffix}"] = awareness["view_id"]
        view_ids[f"{CONSISTENCY_TASK_ID}/{suffix}"] = consistency["view_id"]
        consistency_targets = {row["row_key"]: int(row["target"]) for row in consistency["sidecar"]["record"]["row_references"]}
        awareness_rows = awareness["sidecar"]["record"]["row_references"]
        references = store.lookup([row["row_key"] for row in awareness_rows], model_factory=_refuse_model,
                                  extraction_command="kr extract --config <path>") if awareness_rows else ()
        for row, reference in zip(awareness_rows, references, strict=True):
            residual = store.read_residual_row(reference)
            # Row views of one float32 array per row. Python floats would need ~400 GB for MMLU.
            residuals = dict(enumerate(residual, start=1))
            row_id = content_digest({"kind": "evaluation_row", "schema": 1, "row_key": row["row_key"],
                                     "base_group_id": row["base_group_id"], "variant_id": row["variant_id"],
                                     "instance_id": row["instance_id"], "order_id": row["order_id"]})
            result.append(EvaluationRow(
                row_id=row_id, base_group_id=row["base_group_id"], order_id=row["order_id"],
                variant_id=row["variant_id"], domain=domain, position=row["order_id"],
                awareness_target=int(row["target"]), consistency_target=consistency_targets[row["row_key"]],
                residuals=residuals, canonical_ab=tuple(float(value) for value in row["canonical_ab"]),
                instance_id=row["instance_id"],
            ))
    return tuple(result), content_digest({"kind": "combined_evaluation_view", "schema": 1, "views": view_ids})


def _load_frequency(run_directory: Path) -> TrainingFrequency:
    path = run_directory / AWARENESS_TASK_ID / "training_frequency.json"
    if not path.is_file():
        raise MethodUnavailable(task_id=AWARENESS_TASK_ID, method_id="training_frequency", reason="frequency state was not saved")
    state = json.loads(path.read_text(encoding="utf-8"))
    return TrainingFrequency(
        training_view_id=state["training_view_id"], probability=state["probability"], count=int(state["count"]),
        positive_count=int(state["positive_count"]), available=bool(state["available"]),
        unavailable_reason=state["unavailable_reason"], degenerate=bool(state["degenerate"]), degeneracy=state["degeneracy"],
    )


def _resolve_run(request: StageRequest) -> tuple[Path, Path, dict[str, Any]]:
    token = request.run
    if not token:
        raise KnowledgeRecoveryError("evaluate/report requires --run naming a run bundle or directory")
    candidate = Path(token)
    if candidate.is_dir():
        run_directory = candidate.resolve()
        if run_directory.parent.name == "runs":
            storage = run_directory.parent.parent
            run_directory = _current_attempt(run_directory)
        elif run_directory.parent.parent.name == "runs":
            storage = run_directory.parent.parent.parent
        else:
            raise KnowledgeRecoveryError("--run directory must identify a run bundle or one of its attempts")
    elif request.config:
        storage = _storage(_config(request))
        if token == "current":
            pointer = storage / "runs" / "current.json"
            if not pointer.is_file():
                raise KnowledgeRecoveryError(f"run pointer does not exist: {pointer}")
            run_directory = (storage / "runs" / json.loads(pointer.read_text(encoding="utf-8"))["path"]).parent
        else:
            run_directory = storage / "runs" / token
            if run_directory.is_dir() and run_directory.parent.name == "runs":
                run_directory = _current_attempt(run_directory)
    else:
        raise KnowledgeRecoveryError("--run must be an existing run bundle/attempt when --config is omitted; otherwise pass --config")
    path = run_directory / "manifest.json"
    if not path.is_file():
        raise KnowledgeRecoveryError(f"run manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("attempt_id") != run_directory.name or manifest.get("bundle_id") != run_directory.parent.name:
        raise KnowledgeRecoveryError("run attempt path does not match its manifest")
    return run_directory, storage, manifest


def _current_attempt(bundle: Path) -> Path:
    path = bundle / "manifest.json"
    if not path.is_file():
        raise KnowledgeRecoveryError(f"run bundle manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("bundle_id") != bundle.name or not manifest.get("current_attempt_id"):
        raise KnowledgeRecoveryError("run bundle path does not match its manifest")
    return bundle / str(manifest["current_attempt_id"])


def _residuals(store: ExtractionStore, rows: Sequence[Mapping[str, Any]]) -> numpy.ndarray:
    keys = [row["row_key"] for row in rows]
    try:
        references = store.lookup(keys, model_factory=_refuse_model, extraction_command="kr extract --config cache")
    except CacheIncomplete:
        raise
    # Callers reject empty views. Fill one float32 array instead of stacking a list of rows, which needs twice the memory.
    first = store.read_residual_row(references[0])
    stacked = numpy.empty((len(references), *first.shape), dtype=numpy.float32)
    stacked[0] = first
    for position, reference in enumerate(references[1:], start=1):
        stacked[position] = store.read_residual_row(reference)
    return stacked


def _behavior_instance_labels(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], str]:
    """Give repeated contexts stable within-question slots without collapsing pairs."""
    grouped: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        grouped.setdefault((row["base_group_id"], row["variant_id"]), set()).add(row["instance_id"])
    labels: dict[tuple[str, str, str], str] = {}
    for (base_group_id, variant_id), instance_ids in grouped.items():
        ordered = sorted(instance_ids)
        for index, instance_id in enumerate(ordered, start=1):
            label = variant_id if len(ordered) == 1 else f"{variant_id}#{index}"
            labels[(base_group_id, variant_id, instance_id)] = label
    return labels


def _audit(config: ExperimentConfig, *, recheck: bool, rendered: Sequence[Mapping[str, Any]] | None = None):
    storage = _storage(config)
    activate_offline(storage / "assets" / "huggingface")
    if rendered is None:
        rendered = _read_rendered(storage)
    tokenizer = _tokenizer(Path(config.extraction.model_path))
    assets = resolve_local_assets(config.extraction.model_path)
    examples = []
    seen: set[str] = set()
    for row in rendered:
        if row["user_text"] in seen:
            continue
        seen.add(row["user_text"])
        examples.append(PromptExample(
            user_text=row["user_text"],
            chat=config.extraction.chat,
            system_message=config.extraction.system_message,
        ))
        if len(examples) == 2:
            break
    established = _established_audit(storage)
    audit = audit_canonical_tokens(
        tokenizer,
        examples,
        canonical_strings=config.extraction.verbalizers,
        model_content_digest=assets.weight_content_digest,
        tokenizer_content_digest=assets.tokenizer_content_digest,
        chat=config.extraction.chat,
        system_message=config.extraction.system_message,
        instruction=config.extraction.instruction,
        answer_mode=config.extraction.answer_mode,
        renderer_version=RENDERER_VERSION,
        prompt_boundary=tokenizer.prompt_boundary(config.extraction.chat),
        established=established,
        configured_audit_id=config.extraction.token_audit,
        recheck=recheck,
        config_path=_config_token(config.path),
    )
    write_token_audit(storage / "audits", audit)
    return audit


def _read_rendered(storage: Path) -> list[dict[str, Any]]:
    return json.loads((storage / "prompts" / "rendered.json").read_text(encoding="utf-8"))


# Below this many prompts, tokenizing in the parent is faster than starting a pool.
PARALLEL_TOKENIZE_MIN_ROWS = 20_000
TOKENIZE_CHUNK_ROWS = 2_000
TOKENIZE_MAX_PROCESSES = 32


def _tokenize(
    config: ExperimentConfig,
    audit,
    rendered: Sequence[Mapping[str, Any]],
    key_settings: Mapping[str, Any],
    *,
    min_parallel_rows: int = PARALLEL_TOKENIZE_MIN_ROWS,
) -> list[dict[str, Any]]:
    """Token ids and the extraction row key per rendered prompt, kept in memory.

    ``key_settings`` holds the ``build_row_key`` arguments shared by every row.
    Large inputs are tokenized and keyed in a CPU process pool; the result is
    the same row for row.
    """
    texts = [row["user_text"] for row in rendered]
    setup = (config.extraction.model_path, config.extraction.chat, config.extraction.system_message, dict(key_settings))
    chunks = [texts[start:start + TOKENIZE_CHUNK_ROWS] for start in range(0, len(texts), TOKENIZE_CHUNK_ROWS)]
    processes = min(len(os.sched_getaffinity(0)), TOKENIZE_MAX_PROCESSES, len(chunks))
    from contextlib import nullcontext

    from tqdm import tqdm

    if len(texts) < min_parallel_rows or processes < 2:
        _start_row_encoder(*setup)
        pool = nullcontext(None)
    else:
        import multiprocessing

        pool = multiprocessing.get_context("spawn").Pool(processes, initializer=_start_row_encoder, initargs=setup)
    encoded: list[tuple[list[int], str]] = []
    with pool as workers, tqdm(total=len(texts), desc="tokenize", unit="row", dynamic_ncols=True) as progress:
        parts = map(_encode_rows, chunks) if workers is None else workers.imap(_encode_rows, chunks)
        for part in parts:
            encoded.extend(part)
            progress.update(len(part))
    tokenized = []
    for row, (input_ids, row_key) in zip(rendered, encoded, strict=True):
        mask = [1] * len(input_ids)
        tokenized.append({
            **row,
            "input_ids": input_ids,
            "attention_mask": mask,
            "selected_position": last_nonpadding_index(mask),
            "token_audit_id": audit.audit_id,
            "row_key": row_key,
        })
    return tokenized


# Per-process tokenizer and shared key arguments for ``_encode_rows``.
_ROW_ENCODER: tuple[Any, bool, str | None, dict[str, Any]] | None = None


def _start_row_encoder(model_path: str, chat: bool, system_message: str | None, key_settings: dict[str, Any]) -> None:
    global _ROW_ENCODER
    _ROW_ENCODER = (_tokenizer(Path(model_path)), chat, system_message, key_settings)


def _encode_rows(texts: Sequence[str]) -> list[tuple[list[int], str]]:
    """Token ids and row key for each user text, in order."""
    assert _ROW_ENCODER is not None, "_start_row_encoder runs first"
    tokenizer, chat, system_message, key_settings = _ROW_ENCODER
    encoded = []
    for text in texts:
        input_ids = list(tokenizer.encode_prompt(PromptExample(user_text=text, chat=chat, system_message=system_message)))
        mask = [1] * len(input_ids)
        row_key = build_row_key(
            input_ids=input_ids,
            attention_mask=mask,
            selected_position=last_nonpadding_index(mask),
            **key_settings,
        )
        encoded.append((input_ids, row_key))
    return encoded


def _reuse_collection(storage: Path, collection_path: Path, config: ExperimentConfig) -> None:
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    expected = _extraction_request_signature(config)
    if collection.get("request_signature") != expected:
        raise CacheIncompatible(
            mismatches=("identity",),
            detail="cached collection does not match the requested model, prompt, dataset, or interventions",
        )
    store = ExtractionStore(storage)
    keys = [row["row_key"] for row in collection["rows"]]
    run_extraction(
        [{"row_key": key} for key in keys],
        store=store,
        model_factory=_refuse_model,
        forward_batch=None,
        initial_batch_size=1,
        precision=config.extraction.precision,
        probe_only=True,
    )


def _prompt_contract(config: ExperimentConfig, audit):
    return prompt_contract(
        chat=config.extraction.chat,
        system_message=config.extraction.system_message,
        instruction=config.extraction.instruction,
        verbalizers=config.extraction.verbalizers,
        prompt_boundary=audit.record["prompt_boundary"],
        renderer_version=RENDERER_VERSION,
        token_audit_content=_plain(audit.record),
    )


def _extraction_request_signature(config: ExperimentConfig) -> str:
    extraction = config.extraction
    storage = Path(extraction.storage_root)
    prepared_identity: dict[str, Any] = {}
    normalized_path = storage / "normalized" / "items.json"
    if normalized_path.is_file():
        normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
        prepared_identity["normalized_items"] = content_digest({
            "kind": "normalized_item_set",
            "schema": 1,
            "items": normalized.get("items", []),
        })
    intervention_path = storage / "interventions" / "manifest.json"
    if intervention_path.is_file():
        intervention = json.loads(intervention_path.read_text(encoding="utf-8"))
        prepared_identity["intervention_manifest"] = intervention.get("digest")
    return content_digest({
        "kind": "extraction_request",
        "schema": 1,
        "model": {
            "name": extraction.model_name,
            "revision": extraction.model_revision,
            "tokenizer_revision": extraction.tokenizer_revision,
            "precision": extraction.precision,
            "quantization": extraction.quantization,
        },
        "dataset": {
            "adapter": extraction.dataset_adapter,
            "name": extraction.dataset_name,
            "split": extraction.dataset_split,
            "revision": extraction.dataset_revision,
            **non_default_fields(extraction, ("distractor_seed", "distractor_selection")),
        },
        "prompt": {
            "chat": extraction.chat,
            "system_message": extraction.system_message,
            "instruction": extraction.instruction,
            "labels": list(extraction.labels),
            "verbalizers": list(extraction.verbalizers),
            "selector": extraction.selector,
        },
        "interventions": [_plain(non_default_fields(item)) for item in extraction.interventions],
        "prepared_identity": prepared_identity,
    })


def _reporting(config: ExperimentConfig) -> dict[str, Any]:
    """Analysis fields that must not change an extraction row key."""
    return {
        "split_seed": config.analysis_view.split_seed,
        "split_ratio": {
            "train": config.analysis_view.train_fraction,
            "val": config.analysis_view.val_fraction,
            "test": config.analysis_view.test_fraction,
        },
        "training_seed": config.probe_run.training_seed,
        "filter_threshold": config.analysis_view.min_correct_prob,
        "variant_stats": config.analysis_view.variant_stats,
        "require_confident_on_every_variant": config.analysis_view.require_confident_on_every_variant,
        "loss": config.probe_run.loss,
        "ece_bins": config.probe_run.ece_bins,
        "ood_variants": list(config.analysis_view.eval_variants),
    }


def _tokenizer(model_path: Path):
    mapped = model_path / "tokenizer_map.json"
    if mapped.is_file():
        return MappedContinuationTokenizer.read(mapped)
    if not model_path.exists():
        raise MissingLocalAsset(kind="tokenizer", expected_location=str(mapped))
    return QwenContinuationTokenizer(model_path)


def _established_audit(storage: Path):
    pointer = storage / "audits" / "current.json"
    if not pointer.is_file():
        return None
    body = json.loads(pointer.read_text(encoding="utf-8"))
    return read_token_audit(storage / "audits" / body["path"])


def _load_split(storage: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Read the frozen split sidecar and its base-group assignment."""
    path = storage / "splits" / "split.json"
    if not path.is_file():
        raise KnowledgeRecoveryError(f"split manifest does not exist: {path}\nRun kr prepare --config <path> first.")
    split = json.loads(path.read_text(encoding="utf-8"))
    return split, dict(split["record"]["group_to_split"])


def _split_of(group_splits: Mapping[str, str], base_group_id: str) -> str:
    try:
        return group_splits[base_group_id]
    except KeyError:
        raise KnowledgeRecoveryError(
            f"base group {base_group_id} is not in splits/split.json.\n"
            "The split and the cached rows come from different prepared items. Rerun kr prepare."
        ) from None


def _config(request: StageRequest) -> ExperimentConfig:
    if not request.config:
        raise KnowledgeRecoveryError("this stage requires --config pointing at the three typed objects")
    return load_config(request.config)


DISTRACTOR_PREFERENCES = Path("normalized") / "distractor_preferences.json"
RANK_BATCH_SIZE = 16
# Layers fit at once in single-threaded processes. Each holds about 40 GB for the
# 377k-row MMLU training view (its float32 slice plus float64 and standardized copies).
LINEAR_FIT_PROCESSES = 8


def _ranking_identity(config: ExperimentConfig, source_identity: str) -> dict[str, Any]:
    """Inputs that must match between rank-distractors and prepare."""
    extraction = config.extraction
    return {
        "kind": "mmlu_distractor_preferences",
        "ranking_version": RANKING_VERSION,
        "source_identity": source_identity,
        "model_path": extraction.model_path,
        "model_revision": extraction.model_revision,
        "precision": extraction.precision,
        "chat": extraction.chat,
        "system_message": extraction.system_message,
        "instruction": extraction.instruction,
    }


def _distractor_preferences(config: ExperimentConfig, source_identity: str) -> Mapping[str, Any]:
    path = _storage(config) / DISTRACTOR_PREFERENCES
    command = f"kr rank-distractors --config {_config_token(config.path)}"
    if not path.is_file():
        raise KnowledgeRecoveryError(
            f"{config.extraction.distractor_selection} needs model choice preferences: {path}\n"
            f"Run: {command}"
        )
    saved = json.loads(path.read_text(encoding="utf-8"))
    expected = _ranking_identity(config, source_identity)
    changed = sorted(key for key, value in expected.items() if saved.get(key) != value)
    if changed:
        raise KnowledgeRecoveryError(
            f"Distractor preferences in {path} were made with different {', '.join(changed)}.\n"
            f"Run: {command}"
        )
    return saved["items"]


def _rank_candidates(config: ExperimentConfig, source_identity: str) -> tuple[Any, ...]:
    # The seed only fills a placeholder distractor; the ranking reads the choices.
    return recover_mmlu(
        _read_rows(Path(config.extraction.dataset_path)),
        source_identity=source_identity,
        conflict_directory=_storage(config) / "normalized",
        distractor_seed=0,
    ).items


def _prompt_encoder(config: ExperimentConfig, tokenizer: Any) -> Callable[[str], tuple[int, ...]]:
    def encode(user_text: str) -> tuple[int, ...]:
        return tokenizer.encode_prompt(PromptExample(
            user_text=user_text,
            chat=config.extraction.chat,
            system_message=config.extraction.system_message,
        ))

    return encode


def _rank_shard(
    config: ExperimentConfig,
    candidates: Sequence[Any],
    encode: Callable[[str], tuple[int, ...]],
    ordered_ids: Sequence[int],
    progress: Callable[[int], None],
) -> dict[str, dict[str, Any]]:
    model = load_frozen_model(config.extraction.model_path, precision=config.extraction.precision)
    return rank_items(
        candidates,
        instruction=config.extraction.instruction,
        encode=encode,
        letter_logits=lambda batch, count: _letter_logits(model, batch, ordered_ids[:count]),
        batch_size=RANK_BATCH_SIZE,
        progress=progress,
    )


def _visible_devices(config: ExperimentConfig) -> list[str]:
    """GPU ids visible to this process, as ``CUDA_VISIBLE_DEVICES`` names them.

    The integration mock runs on the CPU in one process.
    """
    model_config = json.loads(Path(config.extraction.model_path, "config.json").read_text(encoding="utf-8"))
    if model_config.get("model_type") == "integration_mock":
        return []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [device.strip() for device in visible.split(",") if device.strip()]
    import torch

    return [str(index) for index in range(torch.cuda.device_count())]


def _run_on_devices(
    stage: str,
    target: Callable[..., Any],
    jobs: Sequence[tuple[str, tuple[Any, ...]]],
    progress: Callable[[int], None],
) -> list[Any]:
    """Spawn one worker per ``(device, args)`` job; return results in job order.

    Each worker calls ``target(*args, progress=...)`` with only its GPU visible.
    The first failure stops every worker.
    """
    import multiprocessing
    import queue

    context = multiprocessing.get_context("spawn")
    messages = context.Queue()
    workers = [
        context.Process(target=_device_worker, args=(target, device, shard, args, messages), daemon=True)
        for shard, (device, args) in enumerate(jobs)
    ]
    for worker in workers:
        worker.start()
    results: dict[int, Any] = {}
    try:
        while len(results) < len(workers):
            try:
                kind, shard, payload = messages.get(timeout=60)
            except queue.Empty:
                dead = [worker.exitcode for worker in workers if worker.exitcode not in (None, 0)]
                if dead:
                    raise KnowledgeRecoveryError(f"a {stage} worker exited with code {dead[0]}")
                continue
            if kind == "progress":
                progress(payload)
            elif kind == "done":
                results[shard] = payload
            else:
                raise KnowledgeRecoveryError(f"a {stage} worker failed on GPU {jobs[shard][0]}:\n{payload}")
    finally:
        for worker in workers:
            if worker.is_alive() and len(results) < len(workers):
                worker.terminate()
            worker.join()
    return [results[shard] for shard in range(len(workers))]


def _device_worker(target: Callable[..., Any], device: str, shard: int, args: tuple[Any, ...], messages: Any) -> None:
    # Set before the first CUDA call so "cuda" names this GPU only.
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    try:
        result = target(*args, progress=lambda count: messages.put(("progress", shard, count)))
        messages.put(("done", shard, result))
    except BaseException:
        import traceback

        messages.put(("failed", shard, traceback.format_exc()))


def _rank_worker(
    config_path: str,
    shard: int,
    shards: int,
    ordered_ids: list[int],
    *,
    progress: Callable[[int], None],
) -> dict[str, dict[str, Any]]:
    config = load_config(config_path)
    source_identity = _file_digest(Path(config.extraction.dataset_path))
    candidates = _rank_candidates(config, source_identity)[shard::shards]
    encode = _prompt_encoder(config, _tokenizer(Path(config.extraction.model_path)))
    return _rank_shard(config, candidates, encode, ordered_ids, progress)


def _letter_logits(model: Any, batch: Sequence[Sequence[int]], token_ids: Sequence[int]) -> numpy.ndarray:
    """Option-letter logits at the last real token, without storing hidden states."""
    width = max(len(ids) for ids in batch)
    padded = [list(ids) + [0] * (width - len(ids)) for ids in batch]
    mask = [[1] * len(ids) + [0] * (width - len(ids)) for ids in batch]
    inner = getattr(model, "inner", None)
    if inner is None:
        full = numpy.asarray(model.forward(padded, mask))
        return select_last_nonpadding(full, numpy.asarray(mask))[:, list(token_ids)]
    import torch

    ids = torch.tensor(padded, dtype=torch.long, device=model.device)
    attention = torch.tensor(mask, dtype=torch.long, device=model.device)
    with torch.no_grad():
        hidden = inner.model(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state
        last = attention.sum(dim=1) - 1
        picked = hidden[torch.arange(len(batch), device=hidden.device), last]
        logits = inner.lm_head(picked)[:, list(token_ids)]
    return logits.float().cpu().numpy()


def _storage(config: ExperimentConfig) -> Path:
    return Path(config.extraction.storage_root)


def _config_token(path: str) -> str:
    if path and " " not in path and not path.startswith("-"):
        return path
    return "config.yaml"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    source = path / "rows.json" if path.is_dir() else path
    if not source.is_file():
        raise MissingLocalAsset(kind="dataset", expected_location=str(source))
    payload = json.loads(source.read_text(encoding="utf-8"))
    if type(payload) is not list:
        raise KnowledgeRecoveryError("dataset snapshot must be a JSON list of source rows")
    return payload


def _file_digest(path: Path) -> str:
    source = path / "rows.json" if path.is_dir() else path
    return content_digest({"kind": "source_file", "schema": 1, "sha256": _sha256(source)})


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _refuse_model() -> None:
    raise KnowledgeRecoveryError("a probe-only lookup must not construct a model")


STAGE_HANDLERS = {
    "validate": validate_stage,
    "rank-distractors": rank_distractors_stage,
    "prepare": prepare_stage,
    "audit-tokens": audit_tokens_stage,
    "extract": extract_stage,
    "diagnose": diagnose_stage,
    "build-view": build_view_stage,
    "train": train_stage,
    "evaluate": evaluate_stage,
    "report": report_stage,
}
