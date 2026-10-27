import ast
import hashlib
import inspect
from pathlib import Path

import pytest

from knowledge_recovery.provenance import (
    CANONICAL_JSON_VERSION,
    canonical_json,
    content_digest,
    evaluation_id,
    extraction_row_key,
    extraction_signature,
    model_signature,
    prompt_contract,
    prompt_signature,
    run_id,
    view_id,
)
from knowledge_recovery.records import (
    EVALUATION_EXCLUDED_KEYS,
    EXTRACTION_EXCLUDED_KEYS,
    RUN_EXCLUDED_KEYS,
)

ROOT = Path(__file__).resolve().parents[1]
A = hashlib.sha256(b"a").hexdigest()
B = hashlib.sha256(b"b").hexdigest()
C = hashlib.sha256(b"c").hexdigest()
KNOWN = "efbd0040190fb0871831e606c581f8a66db79d8e2bb836745a70051306956070"


def _model():
    return model_signature(
        weight_content_digest=A,
        model_config_digest=B,
        tokenizer_content_digest=C,
        model_adapter_version="qwen-adapter-1",
    )


def _contract(**overrides):
    payload = {
        "chat": True,
        "system_message": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "instruction": "Answer with exactly one letter. Do not include any other text.",
        "verbalizers": ["A", "B"],
        "prompt_boundary": "continuation",
        "renderer_version": "render-1",
        "token_audit_content": {"canonical_strings": ["A", "B"]},
    }
    payload.update(overrides)
    return prompt_contract(**payload)


def _prompt(**overrides):
    payload = {
        "input_ids": [11, 22, 33],
        "logical_attention_mask": [1, 1, 1],
        "position_ids": [0, 1, 2],
        "other_forward_inputs": {},
        "prompt_contract_digest": _contract().digest,
    }
    payload.update(overrides)
    return prompt_signature(**payload)


def _extraction(**overrides):
    payload = {
        "extraction_adapter_version": "extract-1",
        "layer_hook_definitions": [
            {"module": "model.layers.0", "layer": 1},
            {"module": "model.layers.1", "layer": 2},
        ],
        "position_selector": "last_nonpadding_input_token",
        "resolved_logical_positions": [3],
        "saved_representation_kinds": ["post_block_pre_final_norm", "final_normalized"],
        "logits_scope": "ab_summary",
        "execution_precision": "bf16",
        "quantization": "none",
        "attention_backend": "sdpa",
        "deterministic_settings": {"dropout": False},
        "numeric_runtime_signature": {"torch": "2.4.0"},
        "residual_storage_dtype": "bf16",
        "logits_storage_dtype": "float32",
        "tensor_schema_version": 1,
    }
    payload.update(overrides)
    return extraction_signature(**payload)


def _row(**overrides):
    payload = {
        "model_signature": _model().digest,
        "prompt_signature": _prompt().digest,
        "extraction_signature": _extraction().digest,
    }
    payload.update(overrides)
    return extraction_row_key(**payload)


def _view(**overrides):
    payload = {
        "role": "train",
        "row_references": ["row-1", "row-2"],
        "group_ids": ["g1", "g2"],
        "target_construction_version": "targets-1",
        "class_order": ["A", "B"],
        "eligibility": {"clean_both": True},
        "weights": [1.0, 1.0],
        "split_provenance": A,
        "filter_provenance": B,
    }
    payload.update(overrides)
    return view_id(**payload)


def _run(**overrides):
    payload = {
        "train_view_id": A,
        "validation_view_id": B,
        "task_spec": {"task": "agreement"},
        "architecture": {"name": "linear"},
        "feature_spec": {"layers": "selected"},
        "method_id": "linear_agreement",
        "task_adapter_version": "adapter-1",
        "method_parameters": {},
        "preprocessing_spec": {"normalize": "standardize"},
        "loss_spec": {"name": "binary_cross_entropy"},
        "optimizer_spec": {"name": "adamw"},
        "training_seed": 42,
        "stopping_rules": {"patience": 20},
        "selection_rules": {"metric": "validation_cross_entropy"},
        "training_implementation_version": "train-1",
        "training_numeric_signature": {"numpy": "2.0.0"},
    }
    payload.update(overrides)
    return run_id(**payload)


def _evaluation(**overrides):
    payload = {
        "method_state_digest": A,
        "evaluation_view_id": B,
        "prediction_convention": {"scores": "probabilities"},
        "metric_definitions": {"brier": "scalar_positive_class"},
        "aggregation_rules": {"summaries": ["per_variant", "pooled_by_domain"]},
        "calibration_bins": 15,
        "prediction_version": "predict-1",
        "metrics_version": "metrics-1",
    }
    payload.update(overrides)
    return evaluation_id(**payload)


def test_canonical_json_sorts_keys_and_keeps_array_order():
    assert canonical_json({"b": 1, "a": {"d": 1, "c": [2, 1]}}) == '{"a":{"c":[2,1],"d":1},"b":1}'
    assert content_digest({"b": 1, "a": 2}) == content_digest({"a": 2, "b": 1})
    assert content_digest([1, 2]) != content_digest([2, 1])


def test_canonical_json_preserves_string_content_and_unicode():
    assert canonical_json({"text": " café "}) == '{"text":" café "}'
    assert canonical_json({"text": "a\nb"}) == '{"text":"a\\nb"}'
    encoded = canonical_json({"text": "café"}).encode("utf-8")
    assert "café".encode() in encoded
    assert b"\\u00e9" not in encoded


def test_numeric_spellings_stay_distinct_and_nonfinite_numbers_fail():
    assert canonical_json(1) == "1"
    assert canonical_json(1.0) == "1.0"
    assert canonical_json(True) == "true"
    assert canonical_json(-0.0) == "-0.0"
    assert len({content_digest(1), content_digest(1.0), content_digest(True), content_digest(-0.0)}) == 4
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="nonfinite"):
            canonical_json({"nested": [value]})


def test_digest_is_sha256_of_the_utf8_canonical_json():
    assert canonical_json({"b": [2, 3], "a": 1}) == '{"a":1,"b":[2,3]}'
    assert content_digest({"b": [2, 3], "a": 1}) == KNOWN
    text = canonical_json({"z": "prompt "})
    assert content_digest({"z": "prompt "}) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert content_digest({"z": "prompt "}) == content_digest({"z": "prompt "})


def test_rejected_json_values_name_the_problem():
    with pytest.raises(TypeError, match="set"):
        canonical_json({"x": {1, 2}})
    with pytest.raises(TypeError, match="bytes"):
        canonical_json(b"abc")
    with pytest.raises(ValueError, match="dictionary keys"):
        canonical_json({1: "a"})


def test_sources_do_not_call_builtin_hash():
    for name in ("records.py", "provenance.py"):
        tree = ast.parse((ROOT / "src" / "knowledge_recovery" / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "hash"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "hash"


def test_sidecar_rehashes_to_the_same_digest():
    artifact = _model()
    side = artifact.sidecar()
    assert side["canonical_json_version"] == CANONICAL_JSON_VERSION == 1
    assert side["digest"] == artifact.digest
    assert content_digest(side["record"]) == artifact.digest
    with pytest.raises(TypeError):
        hash(artifact)


def test_extraction_row_key_contains_only_the_three_signatures():
    row = _row()
    assert row.record["kind"] == "extraction_row"
    assert row.record["schema"] == 1
    assert set(row.record) == {
        "kind",
        "schema",
        "model_signature",
        "prompt_signature",
        "extraction_signature",
    }
    assert list(inspect.signature(extraction_row_key).parameters) == [
        "model_signature",
        "prompt_signature",
        "extraction_signature",
    ]
    with pytest.raises(TypeError):
        extraction_row_key(
            model_signature=A,
            prompt_signature=B,
            extraction_signature=C,
            run_id=A,
            eval_id=B,
            split_seed=1,
            batch_size=4,
            hostname="gpu-1",
            ece_bins=15,
        )


def test_analysis_settings_do_not_change_an_extraction_row_key():
    original = _row()
    again = _row()
    assert original.digest == again.digest
    for needle in ("run_id", "eval_id", "split_seed", "batch_size", "hostname", "ece_bins", "input_ids"):
        assert needle not in original.canonical_json
    changed_prompt = _row(prompt_signature=_prompt(input_ids=[11, 33, 22]).digest)
    changed_model = _row(model_signature=_model().digest[:-1] + ("0" if _model().digest[-1] != "0" else "1"))
    assert changed_prompt.digest != original.digest
    assert changed_model.digest != original.digest


@pytest.mark.parametrize("key", sorted(EXTRACTION_EXCLUDED_KEYS))
def test_extraction_signature_rejects_excluded_keys(key):
    with pytest.raises(ValueError, match=key):
        _extraction(numeric_runtime_signature={key: "recorded-elsewhere"})


def test_prompt_array_order_changes_the_signature_and_key_order_does_not():
    forward = _prompt()
    reversed_ids = _prompt(input_ids=[33, 22, 11])
    assert forward.digest != reversed_ids.digest
    left = _prompt(other_forward_inputs={"b": 1, "a": [1, 2]})
    right = _prompt(other_forward_inputs={"a": [1, 2], "b": 1})
    assert left.digest == right.digest
    assert left.canonical_json != left.canonical_json.replace("[1,2]", "[2,1]")
    hooks = _extraction()
    swapped = _extraction(layer_hook_definitions=[
        {"module": "model.layers.1", "layer": 2},
        {"module": "model.layers.0", "layer": 1},
    ])
    assert hooks.digest != swapped.digest


def test_prompt_contract_keeps_text_and_rejects_audit_locations():
    exact = _contract(instruction="Answer with exactly one letter. Do not include any other text. ")
    assert exact.digest != _contract().digest
    assert "other text. " in exact.canonical_json
    assert _contract(verbalizers=["A", "B"]).digest != _contract(verbalizers=["B", "A"]).digest
    silent = _contract(chat=False, system_message=None)
    assert '"chat":false' in silent.canonical_json
    assert '"system_message":null' in silent.canonical_json
    with pytest.raises(ValueError, match="no-chat"):
        _contract(chat=False, system_message="You are Qwen.")
    with pytest.raises(ValueError, match="system_message"):
        _contract(chat=True, system_message=None)
    for key in ("timestamp", "audited_at", "path", "filesystem_path", "hostname"):
        with pytest.raises(ValueError, match=key):
            _contract(token_audit_content={key: "hidden"})
    assert "path" not in inspect.signature(prompt_contract).parameters
    assert "timestamp" not in inspect.signature(model_signature).parameters


def test_chat_change_reaches_the_row_only_through_the_prompt_signature():
    on = _contract()
    off = _contract(chat=False, system_message=None)
    shared_extraction = _extraction().digest
    shared_model = _model().digest
    row_on = extraction_row_key(
        model_signature=shared_model,
        prompt_signature=_prompt(prompt_contract_digest=on.digest).digest,
        extraction_signature=shared_extraction,
    )
    row_off = extraction_row_key(
        model_signature=shared_model,
        prompt_signature=_prompt(prompt_contract_digest=off.digest).digest,
        extraction_signature=shared_extraction,
    )
    assert row_on.digest != row_off.digest
    assert "system_message" not in row_on.canonical_json
    assert row_on.record["prompt_signature"] == _prompt(prompt_contract_digest=on.digest).digest


def test_view_order_is_part_of_the_view_identity():
    assert _view().digest != _view(row_references=["row-2", "row-1"]).digest
    assert _view().digest != _view(class_order=["B", "A"]).digest
    assert _view(role="validation").record["kind"] == "analysis_view"
    with pytest.raises(ValueError, match="role"):
        _view(role="val")


def test_ece_and_evaluation_views_do_not_enter_the_probe_run():
    training = _run()
    assert training.record["kind"] == "probe_training"
    assert "calibration_bins" not in training.canonical_json
    assert "checkpoint_digest" not in training.canonical_json
    assert "evaluation_view_id" not in training.canonical_json
    assert training.record["training_seed"] == 42
    bins_15 = _evaluation()
    bins_20 = _evaluation(calibration_bins=20)
    assert bins_15.digest != bins_20.digest
    assert '"calibration_bins":15' in bins_15.canonical_json
    assert "uncertainty" not in bins_15.canonical_json
    uncertain = _evaluation(uncertainty={"method": "bootstrap", "seed": 7}, transform_digest=C)
    assert uncertain.digest != bins_15.digest
    assert "uncertainty" in uncertain.canonical_json
    assert '"transform_digest"' in uncertain.canonical_json
    with pytest.raises(TypeError):
        run_id(**{**inspect.signature(run_id).bind(
            train_view_id=A,
            validation_view_id=B,
            task_spec={},
            architecture={},
            feature_spec={},
            method_id="linear_agreement",
            task_adapter_version="adapter-1",
            method_parameters={},
            preprocessing_spec={},
            loss_spec={},
            optimizer_spec={},
            training_seed=42,
            stopping_rules={},
            selection_rules={},
            training_implementation_version="train-1",
            training_numeric_signature={},
        ).arguments, "calibration_bins": 15})


@pytest.mark.parametrize("key", sorted(RUN_EXCLUDED_KEYS))
def test_probe_run_rejects_evaluation_and_execution_keys(key):
    with pytest.raises(ValueError, match=key):
        _run(method_parameters={key: "not-a-training-input"})


@pytest.mark.parametrize("key", sorted(EVALUATION_EXCLUDED_KEYS))
def test_evaluation_rejects_training_parent_and_output_paths(key):
    with pytest.raises(ValueError, match=key):
        _evaluation(metric_definitions={key: "not-part-of-scoring"})


def test_identity_fields_match_the_specified_names():
    expected = {
        "frozen_model": {
            "weight_content_digest",
            "model_config_digest",
            "tokenizer_content_digest",
            "model_adapter_version",
        },
        "model_input": {
            "input_ids",
            "logical_attention_mask",
            "position_ids",
            "other_forward_inputs",
            "prompt_contract_digest",
        },
        "extraction_spec": {
            "extraction_adapter_version",
            "layer_hook_definitions",
            "position_selector",
            "resolved_logical_positions",
            "saved_representation_kinds",
            "logits_scope",
            "execution_precision",
            "quantization",
            "attention_backend",
            "deterministic_settings",
            "numeric_runtime_signature",
            "residual_storage_dtype",
            "logits_storage_dtype",
            "tensor_schema_version",
        },
        "analysis_view": {
            "role",
            "row_references",
            "group_ids",
            "target_construction_version",
            "class_order",
            "eligibility",
            "weights",
            "split_provenance",
            "filter_provenance",
        },
        "probe_training": {
            "train_view_id",
            "validation_view_id",
            "task_spec",
            "architecture",
            "feature_spec",
            "method_id",
            "task_adapter_version",
            "method_parameters",
            "preprocessing_spec",
            "loss_spec",
            "optimizer_spec",
            "training_seed",
            "stopping_rules",
            "selection_rules",
            "training_implementation_version",
            "training_numeric_signature",
        },
        "evaluation": {
            "method_state_digest",
            "evaluation_view_id",
            "prediction_convention",
            "metric_definitions",
            "aggregation_rules",
            "calibration_bins",
            "prediction_version",
            "metrics_version",
        },
    }
    artifacts = {
        "frozen_model": _model(),
        "model_input": _prompt(),
        "extraction_spec": _extraction(),
        "analysis_view": _view(),
        "probe_training": _run(),
        "evaluation": _evaluation(),
    }
    for kind, fields in expected.items():
        assert artifacts[kind].record["kind"] == kind
        assert set(artifacts[kind].record) == fields | {"kind", "schema"}
