import json

from knowledge_recovery.integration import FAIL_QUESTION, PASS_QUESTION, SUBSET_LABEL, run_labeled_subset


def test_labeled_subset_reaches_the_filter_without_loading_qwen_weights(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("Qwen weights were loaded")

    monkeypatch.setattr("knowledge_recovery.models.qwen._load_qwen_weights", boom)
    result = run_labeled_subset(tmp_path)
    assert result["model_path"]
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
    subset = json.loads((result["storage"] / "analyses" / "subset.json").read_text(encoding="utf-8"))
    assert subset == {"subset_label": SUBSET_LABEL}
    behavior = json.loads((result["storage"] / "analyses" / "behavior.json").read_text(encoding="utf-8"))
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
    run_labeled_subset(tmp_path)
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


def test_a_new_split_without_reextraction_reaches_diagnose_and_views(tmp_path, monkeypatch):
    import yaml

    from knowledge_recovery.cli import StageRequest
    from knowledge_recovery.pipeline import STAGE_HANDLERS
    from knowledge_recovery.views.splits import freeze_grouped_split

    result = run_labeled_subset(tmp_path)
    storage = result["storage"]
    groups = [result["pass_group"], result["fail_group"]]
    old = json.loads((storage / "splits" / "split.json").read_text(encoding="utf-8"))["record"]["group_to_split"]
    seed = next(
        candidate for candidate in range(43, 200)
        if freeze_grouped_split(
            groups, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2, split_seed=candidate,
        ).group_to_split != old
    )
    config_path = tmp_path / "config.yaml"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["analysis_view"]["split"]["seed"] = seed
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    # Without model files, extract reuses the collection and keeps its old split copy.
    def boom(*_args, **_kwargs):
        raise AssertionError("a re-split loaded the model")

    monkeypatch.setattr("knowledge_recovery.pipeline.load_frozen_model", boom)
    model = tmp_path / "model"
    for child in model.iterdir():
        child.unlink()
    model.rmdir()
    for stage in ("prepare", "extract", "diagnose", "build-view"):
        STAGE_HANDLERS[stage](StageRequest(stage=stage, config=str(config_path)))

    new = json.loads((storage / "splits" / "split.json").read_text(encoding="utf-8"))["record"]["group_to_split"]
    assert new != old
    collection = json.loads((storage / "extractions" / "collection.json").read_text(encoding="utf-8"))
    assert {row["base_group_id"]: row["split"] for row in collection["rows"]} == old
    scored = json.loads((storage / "analyses" / "scored.json").read_text(encoding="utf-8"))
    assert {row["base_group_id"]: row["split"] for row in scored["rows"]} == new
    index = json.loads((storage / "views" / "index.json").read_text(encoding="utf-8"))
    pass_group = result["pass_group"]
    for name, split in (("train", "train"), ("validation", "val"), ("evaluation_id", "test")):
        view = json.loads((storage / "views" / index[f"behavioral_awareness/{name}"]["path"]).read_text(encoding="utf-8"))
        assert (pass_group in view["sidecar"]["record"]["group_ids"]) == (new[pass_group] == split)


def test_diagnose_writes_variant_counts_and_can_apply_the_final_filter(tmp_path):
    import yaml

    from knowledge_recovery.cli import StageRequest
    from knowledge_recovery.pipeline import diagnose_stage

    result = run_labeled_subset(tmp_path)
    storage = result["storage"]
    directory = storage / "analyses" / "filter"
    record = json.loads((directory / "variant_filter_report.json").read_text(encoding="utf-8"))["record"]
    assert record["paired_eligibility"]["variants"] == ["T-C", "NT-C", "T-IC", "NT-IC"]
    assert record["paired_eligibility"]["pool_size"] == 1
    for variant_id in ("T-C", "NT-C", "T-IC", "NT-IC"):
        assert record["population_counts"][variant_id]["overall"]["both_orders"] == {
            "before": 2,
            "after_greedy_is_option": 2,
            "after_option_probability": 2,
            "after_both_orders_confident": 2,
        }
    misreports = json.loads((directory / "misreport_report.json").read_text(encoding="utf-8"))
    assert misreports["variants"] == ["T-C", "NT-C", "T-IC", "NT-IC"]
    assert "after_variant_both_orders_confident" in misreports["steps"]
    scored = json.loads((storage / "analyses" / "scored.json").read_text(encoding="utf-8"))
    assert scored["variant_filter_digest"]
    assert scored["eligible_base_group_ids"] == [result["pass_group"]]

    config_path = tmp_path / "config.yaml"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["analysis_view"]["filter_correct"]["require_confident_on_every_variant"] = True
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    diagnose_stage(StageRequest(stage="diagnose", config=str(config_path)))
    final = json.loads((directory / "variant_filter_report.json").read_text(encoding="utf-8"))["record"]
    assert final["population_counts"]["final"]["overall"]["base_questions"] == 1
    assert "Final (confident on every variant)" in (directory / "variant_filter_report.txt").read_text(encoding="utf-8")
    misreports = json.loads((directory / "misreport_report.json").read_text(encoding="utf-8"))
    assert misreports["steps"][-1] == "after_confident_on_every_variant"

    document["analysis_view"]["filter_correct"]["variant_stats"] = False
    document["analysis_view"]["filter_correct"]["require_confident_on_every_variant"] = False
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    diagnose_stage(StageRequest(stage="diagnose", config=str(config_path)))
    assert not (directory / "variant_filter_report.json").exists()
    scored = json.loads((storage / "analyses" / "scored.json").read_text(encoding="utf-8"))
    assert scored["variant_filter_digest"] is None


def test_extraction_stores_a_four_value_logit_summary(tmp_path):
    from knowledge_recovery.answers import AB_SUMMARY_FIELDS
    from knowledge_recovery.cache.store import ExtractionStore

    result = run_labeled_subset(tmp_path)
    collection = json.loads((result["storage"] / "extractions" / "collection.json").read_text(encoding="utf-8"))
    store = ExtractionStore(result["storage"])
    reference = store.lookup([collection["rows"][0]["row_key"]], model_factory=None, extraction_command="kr extract")[0]
    assert store.read_logits_row(reference).shape == (len(AB_SUMMARY_FIELDS),)
