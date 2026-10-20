import json

from knowledge_recovery.integration import FAIL_QUESTION, PASS_QUESTION, SUBSET_LABEL, run_labeled_subset
from knowledge_recovery.models.qwen import load_qwen


def test_labeled_subset_reaches_the_filter_without_a_basis_choice(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("Qwen weights were loaded")

    monkeypatch.setattr("knowledge_recovery.models.qwen.load_qwen", boom)
    result = run_labeled_subset(tmp_path)
    assert result["model_path"]
    assert load_qwen is not None
    assert result["stops_for_probability_basis"] is False
    assert not (result["storage"] / "decisions").exists()
    assert result["eligible_base_group_ids"] == [result["pass_group"]]
    assert result["fail_group"] not in result["eligible_base_group_ids"]
    overall = result["report"]["record"]["population_counts"]["overall"]
    assert overall["both_orders"] == {
        "before": 4,
        "after_greedy_is_option": 3,
        "after_option_probability": 3,
        "after_greedy_correct": 3,
        "after_both_orders_correct": 2,
    }
    assert overall["correct_at_A"]["after_both_orders_correct"] == 1
    assert overall["correct_at_B"]["after_greedy_is_option"] == 1
    text = (result["storage"] / "analyses" / "filter" / "filter_report.txt").read_text(encoding="utf-8")
    assert "normalized" in text
    assert "raw-versus-canonical" not in text
    subset = json.loads((result["storage"] / "analyses" / "subset.json").read_text(encoding="utf-8"))
    assert subset == {"stops_for_probability_basis": False, "subset_label": SUBSET_LABEL}
    behavior = json.loads((result["storage"] / "analyses" / "behavior.json").read_text(encoding="utf-8"))
    assert behavior["probability_basis"] is None
    clean = [
        instance
        for table in behavior["tables"]
        if table["scoring"] == "canonical_label" and table["population"] == "all_valid"
        for instance in table["instances"]
        if instance["variant_id"] == "clean"
    ]
    assert clean
    assert all(instance["correct_at_A"]["changed"]["count"] == 0 for instance in clean)
    assert all(instance["correct_at_B"]["changed"]["count"] == 0 for instance in clean)
    collection = json.loads((result["storage"] / "extractions" / "collection.json").read_text(encoding="utf-8"))
    assert collection["subset_label"] == SUBSET_LABEL
    assert collection["numeric"]["probe_seed_enters_inference"] is False
    assert collection["numeric"]["residual_storage_dtype"] == "bf16"
    assert collection["numeric"]["logits_storage_dtype"] == "float32"
    assert collection["resolved_audit"]["token_audit"]
    assert collection["resolved_audit"]["configured_token_audit"] is None
    assert PASS_QUESTION != FAIL_QUESTION


def test_a_second_extract_does_not_need_the_model_files(tmp_path, monkeypatch):
    result = run_labeled_subset(tmp_path)
    calls = []

    def boom(*_args, **_kwargs):
        calls.append("model")
        raise AssertionError("model files were required for a cache hit")

    monkeypatch.setattr("knowledge_recovery.pipeline.load_frozen_model", boom)
    model = tmp_path / "model"
    for child in model.iterdir():
        child.unlink()
    model.rmdir()
    from knowledge_recovery.cli import StageRequest
    from knowledge_recovery.pipeline import extract_stage

    extract_stage(StageRequest(stage="extract", config=str(tmp_path / "config.yaml")))
    assert calls == []
