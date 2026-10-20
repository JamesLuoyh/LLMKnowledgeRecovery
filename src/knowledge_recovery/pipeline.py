"""Stage graph. The command-line module only dispatches into this table.

Filter counts are written before any probe is fit. A method that cannot fit
is recorded and the other tasks still run. Extraction-cache misses on a
probe-only stage are reported and do not construct a model.
"""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy

from knowledge_recovery.answers.canonical import CanonicalTokens, score_binary_answers
from knowledge_recovery.cache.store import ExtractionStore
from knowledge_recovery.cli import StageRequest
from knowledge_recovery.config import ExperimentConfig, load_config, validate_before_expensive_work
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
from knowledge_recovery.extraction.positions import last_nonpadding_index
from knowledge_recovery.extraction.runner import run_extraction
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
from knowledge_recovery.records import PromptRecord
from knowledge_recovery.reports.export import export_report
from knowledge_recovery.tasks import awareness_task_spec, consistency_task_spec
from knowledge_recovery.tasks.base import AWARENESS_TASK_ID, CONSISTENCY_TASK_ID
from knowledge_recovery.training.fit import TRAINING_IMPLEMENTATION_VERSION, fit_linear_probe
from knowledge_recovery.training.manifest import write_selection_manifest
from knowledge_recovery.training.select import select_probe
from knowledge_recovery.views.build import ScoredRow, build_task_views
from knowledge_recovery.views.filters import CleanOrder, apply_filter_gate, apply_variant_gate
from knowledge_recovery.views.splits import freeze_grouped_split

STAGE_ORDER = (
    "validate",
    "prepare",
    "audit-tokens",
    "extract",
    "diagnose",
    "build-view",
    "train",
    "evaluate",
    "report",
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
    normalized = recover_sycophancy(
        rows,
        source_identity=_file_digest(Path(config.extraction.dataset_path)),
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
    audit = _audit(config, recheck=False)
    tokenized = _tokenize(config, audit)
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
    prepared = []
    for row in tokenized:
        row_key = build_row_key(
            model_signature=signature,
            input_ids=row["input_ids"],
            attention_mask=row["attention_mask"],
            prompt_contract_digest=contract,
            position_selector=SUPPORTED_POSITION_SELECTOR,
            selected_position=row["selected_position"],
            execution_precision=config.extraction.precision,
            attention_backend=str(runtime["attention_backend"]),
            numeric_runtime_signature=runtime,
            reporting=_reporting(config),
        )
        prepared.append({**row, "row_key": row_key})
    store = ExtractionStore(storage)
    statuses = store.row_statuses([row["row_key"] for row in prepared])
    missing = [row["row_key"] for row in prepared if statuses[row["row_key"]] != "indexed"]

    def factory() -> object:
        return load_frozen_model(config.extraction.model_path, precision=config.extraction.precision)

    def forward(model: object, rows: Sequence[Mapping[str, Any]], *, batch_size: int, precision: str):
        if model_config.get("logit_rule") == "integration_subset_v1":
            from knowledge_recovery.integration import subset_forward

            return subset_forward(model, rows, batch_size=batch_size, precision=precision)
        return forward_padded(model, rows, batch_size=batch_size, precision=precision)

    if missing:
        result = run_extraction(
            prepared,
            store=store,
            model_factory=factory,
            forward_batch=forward,
            initial_batch_size=4,
            precision=config.extraction.precision,
            execution_backend=str(model_config.get("model_type") or "local"),
        )
    else:
        result = run_extraction(
            prepared,
            store=store,
            model_factory=factory,
            forward_batch=forward,
            initial_batch_size=4,
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
        scores = score_binary_answers([float(value) for value in logits.tolist()], tokens)
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
    if config.analysis_view.variant_stats:
        intervened = [
            variant_id
            for variant_id in dict.fromkeys(row["variant_id"] for row in scored)
            if variant_id != config.analysis_view.filter_variant
        ]
        variant_outcome = apply_variant_gate(
            {
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
            },
            eligible,
            output_directory=variant_directory,
            min_correct_prob=config.analysis_view.min_correct_prob,
            require_all=config.analysis_view.require_all_variants,
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
        "probability_basis": None,
        **behavior_document(tables),
    })
    _write_json(storage / "analyses" / "scored.json", {
        "filter_digest": outcome.report_digest,
        "variant_filter_digest": variant_digest,
        "require_all_variants": config.analysis_view.require_all_variants,
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
        _write_json(storage / "analyses" / "subset.json", {
            "subset_label": collection["subset_label"],
            "stops_for_probability_basis": False,
        })


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
            residuals = {index + 1: tuple(float(value) for value in residual[index]) for index in range(residual.shape[0])}
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
    stacked = [store.read_residual_row(reference) for reference in references]
    return numpy.stack(stacked).astype(numpy.float32)


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


def _audit(config: ExperimentConfig, *, recheck: bool):
    storage = _storage(config)
    activate_offline(storage / "assets" / "huggingface")
    rendered = json.loads((storage / "prompts" / "rendered.json").read_text(encoding="utf-8"))
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
    _tokenize(config, audit)
    return audit


def _tokenize(config: ExperimentConfig, audit) -> list[dict[str, Any]]:
    storage = _storage(config)
    rendered = json.loads((storage / "prompts" / "rendered.json").read_text(encoding="utf-8"))
    tokenizer = _tokenizer(Path(config.extraction.model_path))
    tokenized = []
    for row in rendered:
        example = PromptExample(
            user_text=row["user_text"],
            chat=config.extraction.chat,
            system_message=config.extraction.system_message,
        )
        input_ids = list(tokenizer.encode_prompt(example))
        mask = [1] * len(input_ids)
        position = last_nonpadding_index(mask)
        blocks = dict(row["blocks"])
        blocks["model_input_text"] = tokenizer.prompt_string(example)
        record = PromptRecord(
            base_item_id=row["base_group_id"],
            order_id=row["order_id"],
            variant_id=row["variant_id"],
            intervention_instance_id=row["instance_id"],
            correct_label=row["correct_label"],
            ordered_answer_texts=tuple(row["ordered_answer_texts"]),
            resolved_blocks=blocks,
            placement=blocks["placement"],
            rendered_text=row["user_text"],
            input_token_ids=tuple(input_ids),
            attention={"mask": mask, "padding": "right"},
            token_audit_id=audit.audit_id,
            prompt_hash=content_digest({"kind": "rendered_user_text", "schema": 1, "text": row["user_text"]}),
            selected_input_position=position,
        )
        tokenized.append({
            **row,
            "prompt": _plain(record.payload()),
            "input_ids": input_ids,
            "attention_mask": mask,
            "selected_position": position,
            "token_audit_id": audit.audit_id,
        })
    _write_json(storage / "prompts" / "tokenized.json", tokenized)
    return tokenized


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
        },
        "prompt": {
            "chat": extraction.chat,
            "system_message": extraction.system_message,
            "instruction": extraction.instruction,
            "labels": list(extraction.labels),
            "verbalizers": list(extraction.verbalizers),
            "selector": extraction.selector,
        },
        "interventions": [_plain(vars(item)) for item in extraction.interventions],
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
        "require_all_variants": config.analysis_view.require_all_variants,
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
    "prepare": prepare_stage,
    "audit-tokens": audit_tokens_stage,
    "extract": extract_stage,
    "diagnose": diagnose_stage,
    "build-view": build_view_stage,
    "train": train_stage,
    "evaluate": evaluate_stage,
    "report": report_stage,
}
