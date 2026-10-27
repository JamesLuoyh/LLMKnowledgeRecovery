import argparse
import json

import numpy
import pytest

from knowledge_recovery.cache.store import ExtractionStore
from knowledge_recovery.cli import StageRequest, build_parser, main
from knowledge_recovery.errors import CacheIncomplete, KnowledgeRecoveryError
from knowledge_recovery.extraction.identity import build_row_key
from knowledge_recovery.integration import _write_config
from knowledge_recovery.pipeline import (
    STAGE_HANDLERS,
    STAGE_ORDER,
    PipelineIncomplete,
    train_stage,
)
from knowledge_recovery.provenance import content_digest, run_id
from knowledge_recovery.views.build import ScoredRow, build_task_views, view_summary

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
COMMAND = "kr extract --config cache"


def _base_key():
    return dict(
        model_signature=DIGEST_A,
        input_ids=(1, 2, 3, 4),
        attention_mask=(1, 1, 1, 1),
        prompt_contract_digest=DIGEST_B,
        selected_position=3,
        execution_precision="bf16",
        attention_backend="sdpa",
        logits_scope="full_vocabulary",
        numeric_runtime_signature={"backend": "integration_mock", "execution_precision": "bf16"},
    )


def test_stage_order_writes_the_filter_before_training():
    assert STAGE_ORDER.index("diagnose") < STAGE_ORDER.index("train")
    assert tuple(STAGE_HANDLERS) == STAGE_ORDER


def test_the_parser_has_no_force_or_skip_flag():
    parser = build_parser()
    subparsers = [action for action in parser._actions if isinstance(action, argparse._SubParsersAction)]
    for subparser in subparsers[0].choices.values():
        text = subparser.format_help()
        assert "--force" not in text
        assert "--skip" not in text


def test_a_known_stage_is_dispatched(capsys):
    code = main(["validate", "--config", "configs/does-not-exist.yaml"])
    assert code != 0
    assert "is not registered" not in capsys.readouterr().err


def test_analysis_changes_keep_extraction_keys_and_prompt_model_position_do_not(tmp_path):
    base = _base_key()
    original = build_row_key(**base)
    for reporting in (
        {"split_seed": 1},
        {"split_seed": 99},
        {"training_seed": 7},
        {"split_ratio": {"train": 0.5, "val": 0.25, "test": 0.25}},
        {"filter_threshold": 0.9},
        {"min_correct_prob": 0.8},
        {"loss": "binary_cross_entropy"},
        {"loss": "another_loss"},
        {"ece_bins": 15},
        {"ece_bins": 20},
        {"ood_variants": ["NT-C", "NT-IC"]},
        {"ood_variants": ["NT-C"]},
        {"batch_size": 8},
    ):
        assert build_row_key(**base, reporting=reporting) == original
    store = ExtractionStore(tmp_path)
    store.write_rows([{
        "row_key": original,
        "residuals": numpy.ones((1, 2), dtype=numpy.float32),
        "logits": numpy.ones(3, dtype=numpy.float32),
        "final_normalized": numpy.zeros(2, dtype=numpy.float32),
    }])

    def boom():
        raise AssertionError("model construction is forbidden")

    for changed in (
        {**base, "input_ids": (1, 2, 3, 5)},
        {**base, "model_signature": "c" * 64},
        {**base, "selected_position": 1},
    ):
        missing = build_row_key(**changed)
        assert missing != original
        with pytest.raises(CacheIncomplete):
            store.lookup([missing], model_factory=boom, extraction_command=COMMAND)
    found = store.lookup([original], model_factory=boom, extraction_command=COMMAND)
    assert found[0].extraction_row_key == original


def test_ece_bins_and_ood_variants_are_not_probe_run_inputs():
    shared = dict(
        train_view_id=DIGEST_A,
        validation_view_id=DIGEST_B,
        task_spec={"task": "behavioral_awareness"},
        architecture={"name": "linear"},
        feature_spec={"layers": "all"},
        method_id="linear",
        task_adapter_version="v1",
        method_parameters={},
        preprocessing_spec={"mode": "standardize"},
        loss_spec={"name": "binary_cross_entropy"},
        optimizer_spec={"name": "AdamW"},
        training_seed=42,
        stopping_rules={"patience": 20},
        selection_rules={"metric": "validation_cross_entropy"},
        training_implementation_version="linear-fit-v1",
        training_numeric_signature={"dtype": "float64"},
    )
    original = run_id(**shared).digest
    assert "ece_bins" not in run_id.__code__.co_varnames
    assert "ood_variants" not in run_id.__code__.co_varnames
    changed = dict(shared)
    changed["training_seed"] = 43
    assert run_id(**changed).digest != original
    changed["training_seed"] = 42
    changed["loss_spec"] = {"name": "another_loss"}
    assert run_id(**changed).digest != original


def test_ood_variants_do_not_change_the_training_view():
    rows = [
        ScoredRow("a" * 64, "g", "train", "clean", "correct_at_A", "A", "A"),
        ScoredRow("b" * 64, "g", "test", "NT-C", "correct_at_A", "A", "B"),
        ScoredRow("c" * 64, "g", "test", "NT-IC", "correct_at_A", "A", "A"),
    ]
    first = build_task_views(
        rows,
        eligible=("g",),
        train_variants=("clean",),
        eval_variants=("NT-C",),
        split_digest=DIGEST_A,
        filter_digest=DIGEST_B,
    )
    second = build_task_views(
        rows,
        eligible=("g",),
        train_variants=("clean",),
        eval_variants=("NT-IC",),
        split_digest=DIGEST_A,
        filter_digest=DIGEST_B,
    )
    assert first["behavioral_awareness/train"].digest == second["behavioral_awareness/train"].digest
    assert first["behavioral_awareness/evaluation_ood"].digest != second["behavioral_awareness/evaluation_ood"].digest
    train_keys = [row["row_key"] for row in first["behavioral_awareness/train"].record["row_references"]]
    assert train_keys == ["a" * 64]


def test_empty_ood_variants_leave_out_the_ood_view_and_the_summary_says_so():
    rows = [
        ScoredRow("a" * 64, "g", "train", "clean", "correct_at_A", "A", "A"),
        ScoredRow("b" * 64, "g", "test", "clean", "correct_at_A", "A", "B"),
    ]
    views = build_task_views(
        rows,
        eligible=("g",),
        train_variants=("clean",),
        eval_variants=(),
        split_digest=DIGEST_A,
        filter_digest=DIGEST_B,
    )
    assert sorted(views) == [
        f"{task}/{name}"
        for task in ("behavioral_awareness", "knowledge_consistency")
        for name in ("evaluation_id", "train", "validation")
    ]
    summary = view_summary(views, skipped_ood=True)
    assert "Built 6 views." in summary
    assert "eval_variants is empty" in summary
    assert "behavioral_awareness/evaluation_id" in summary and "clean: 1" in summary
    same = build_task_views(
        rows,
        eligible=("g",),
        train_variants=("clean",),
        eval_variants=("clean",),
        split_digest=DIGEST_A,
        filter_digest=DIGEST_B,
    )
    assert "OOD evaluation has the same rows as ID evaluation" in view_summary(same)


def test_training_requires_the_filter_report(tmp_path):
    config = tmp_path / "config.yaml"
    _write_config(config, storage=tmp_path / "storage", model=tmp_path / "model", dataset=tmp_path / "rows.json")
    with pytest.raises(KnowledgeRecoveryError, match="Filter counts are written before training"):
        train_stage(StageRequest(stage="train", config=str(config)))


def test_method_failures_leave_the_other_tasks_in_an_incomplete_summary(tmp_path):
    storage = tmp_path / "storage"
    (storage / "analyses" / "filter").mkdir(parents=True)
    (storage / "analyses" / "filter" / "filter_report.json").write_text("{}\n", encoding="utf-8")
    view_id = content_digest({"kind": "view", "name": "hand"})
    _write_views(storage, view_id)
    config = tmp_path / "config.yaml"
    _write_config(config, storage=storage, model=tmp_path / "model", dataset=tmp_path / "rows.json")
    with pytest.raises(PipelineIncomplete) as raised:
        train_stage(StageRequest(stage="train", config=str(config)))
    summary = json.loads((storage / "runs" / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall"] == "incomplete"
    names = [item["task_id"] for item in summary["tasks"]]
    assert names == [
        "behavioral_awareness/linear_agreement",
        "behavioral_awareness/training_frequency",
        "knowledge_consistency/linear_clean_answer",
        "knowledge_consistency/contextual_answer",
    ]
    by_name = {item["task_id"]: item["status"] for item in summary["tasks"]}
    assert by_name["behavioral_awareness/linear_agreement"] == "unavailable"
    assert by_name["behavioral_awareness/training_frequency"] == "complete"
    assert by_name["knowledge_consistency/linear_clean_answer"] == "unavailable"
    assert by_name["knowledge_consistency/contextual_answer"] == "complete"
    assert "one-class" in str(raised.value) or "empty validation" in str(raised.value)
    first = json.loads((storage / "runs" / "current.json").read_text(encoding="utf-8"))
    first_manifest = storage / "runs" / first["bundle_id"] / first["attempt_id"] / "manifest.json"
    assert first_manifest.is_file()

    with pytest.raises(PipelineIncomplete):
        train_stage(StageRequest(stage="train", config=str(config)))
    second = json.loads((storage / "runs" / "current.json").read_text(encoding="utf-8"))
    assert second["bundle_id"] == first["bundle_id"]
    assert second["attempt_id"] != first["attempt_id"]
    assert first_manifest.is_file()
    assert (storage / "runs" / second["bundle_id"] / second["attempt_id"] / "manifest.json").is_file()


def _write_views(storage, view_id: str) -> None:
    views = storage / "views"
    views.mkdir(parents=True)
    files = {
        "behavioral_awareness/train": [{"row_key": "d" * 64, "target": 1}, {"row_key": "e" * 64, "target": 1}],
        "behavioral_awareness/validation": [],
        "knowledge_consistency/train": [{"row_key": "d" * 64, "target": 0}],
        "knowledge_consistency/validation": [],
    }
    index = {}
    for name, rows in files.items():
        filename = name.replace("/", "__") + ".json"
        payload = {"view_id": view_id, "sidecar": {"record": {"row_references": rows}}}
        (views / filename).write_text(json.dumps(payload), encoding="utf-8")
        index[name] = {"view_id": view_id, "path": filename}
    (views / "index.json").write_text(json.dumps(index), encoding="utf-8")
