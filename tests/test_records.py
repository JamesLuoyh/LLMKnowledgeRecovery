import hashlib
from dataclasses import FrozenInstanceError, fields

import pytest

from knowledge_recovery.provenance import identify
from knowledge_recovery.records import (
    SCHEMA_VERSION,
    AnswerScores,
    BaseItem,
    DecisionRecord,
    EvaluationManifest,
    ExecutionProvenance,
    ExtractionRecord,
    ExtractionRow,
    FilterReport,
    InterventionInstance,
    InterventionSpec,
    MethodSpec,
    ProbeSpec,
    ProbeTraining,
    PromptRecord,
    Record,
    RunManifest,
    SplitManifest,
    TaskSpec,
)

DIGEST = hashlib.sha256(b"record").hexdigest()
OTHER = hashlib.sha256(b"other").hexdigest()


def _classes():
    import knowledge_recovery.records as module

    return tuple(
        value
        for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, Record) and value is not Record
    )


def _item(**overrides):
    payload = {
        "source_identity": "cfierro/sycophancy_eval_answer",
        "source_row_ids": ["row-b", "row-a"],
        "question": " café ",
        "correct_answer": "yes",
        "incorrect_answer": "no",
        "source_metadata": {"split": "test"},
        "normalization_version": "sycophancy-1",
        "duplicate_group_id": "group-1",
    }
    payload.update(overrides)
    return BaseItem(**payload)


def test_every_record_has_a_distinct_kind_and_schema():
    classes = _classes()
    kinds = [cls.kind for cls in classes]
    assert len(kinds) == len(set(kinds))
    assert all(cls.schema == SCHEMA_VERSION == 1 for cls in classes)
    assert "extraction_row" in kinds
    assert "execution_provenance" in kinds


def test_records_are_frozen_unhashable_and_preserve_text():
    item = _item()
    with pytest.raises(FrozenInstanceError):
        item.question = "changed"
    with pytest.raises(TypeError):
        item.source_metadata["split"] = "train"
    with pytest.raises(TypeError):
        hash(item)
    assert item.question == " café "
    assert item.source_row_ids == ("row-b", "row-a")
    body = identify(item)
    assert " café " in body.canonical_json
    assert body.record["kind"] == "base_item"
    assert identify(item).digest == body.digest
    assert identify(_item(source_row_ids=["row-a", "row-b"])).digest != body.digest


def test_base_item_rejects_empty_or_equal_answers():
    with pytest.raises(ValueError, match="must differ"):
        _item(correct_answer="yes", incorrect_answer="yes")
    with pytest.raises(ValueError, match="question"):
        _item(question="")
    with pytest.raises(ValueError, match="source row ID"):
        _item(source_row_ids=[])
    spaced = _item(correct_answer=" yes", incorrect_answer="yes")
    assert spaced.correct_answer == " yes"


def test_decision_basis_is_explicit_and_the_filter_report_does_not_choose_one():
    decision = DecisionRecord(
        probability_basis="canonical",
        filter_report_digest=DIGEST,
        scope={"dataset": "sycophancy"},
        timestamp="2026-01-01T00:00:00Z",
        user_instruction="use canonical",
    )
    assert decision.probability_basis == "canonical"
    with pytest.raises(ValueError, match="probability_basis"):
        DecisionRecord(
            probability_basis="inferred",
            filter_report_digest=DIGEST,
            scope={},
            timestamp="2026-01-01T00:00:00Z",
            user_instruction="pick the larger retention",
        )
    assert "probability_basis" not in {item.name for item in fields(FilterReport)}
    assert "probability_basis" in {item.name for item in fields(DecisionRecord)}


def test_execution_facts_have_a_record_outside_the_extraction_row():
    execution = ExecutionProvenance(
        timestamp="2026-01-01T00:00:00Z",
        hostname="gpu-1",
        batch_size=4,
        processing_order=["row-b", "row-a"],
    )
    row = ExtractionRow(
        model_signature=DIGEST, prompt_signature=OTHER, extraction_signature=DIGEST
    )
    assert {item.name for item in fields(ExtractionRow)} == {
        "model_signature",
        "prompt_signature",
        "extraction_signature",
    }
    assert "hostname" in identify(execution).canonical_json
    assert "gpu-1" not in identify(row).canonical_json
    with pytest.raises(ValueError, match="hostname"):
        RunManifest(
            train_view_id=DIGEST,
            validation_view_id=OTHER,
            extraction_references=["shard-1"],
            task={"id": "agreement"},
            architecture={"name": "linear"},
            features={"layers": "all"},
            preprocessing={"normalize": "standardize"},
            training_seed=42,
            loss={"name": "binary_cross_entropy"},
            optimizer={"name": "adamw"},
            stopping_rules={"patience": 20},
            selection_rules={"metric": "validation_cross_entropy"},
            environment_versions={"hostname": "gpu-1"},
            checkpoint_digest=None,
            selection_digest=None,
        )


def test_run_manifest_outputs_are_not_part_of_the_stored_training_fields():
    training_fields = {item.name for item in fields(RunManifest)}
    assert "checkpoint_digest" in training_fields
    assert "ece_bins" not in training_fields
    assert "test_variants" not in training_fields
    blank = RunManifest(
        train_view_id=DIGEST,
        validation_view_id=OTHER,
        extraction_references=["shard-1"],
        task="agreement",
        architecture="linear",
        features="residual",
        preprocessing="standardize",
        training_seed=7,
        loss="binary_cross_entropy",
        optimizer="adamw",
        stopping_rules={"patience": 20},
        selection_rules={"metric": "validation_cross_entropy"},
        environment_versions={"numpy": "2.0.0"},
        checkpoint_digest=None,
        selection_digest=None,
    )
    finished = RunManifest(
        train_view_id=DIGEST,
        validation_view_id=OTHER,
        extraction_references=["shard-1"],
        task="agreement",
        architecture="linear",
        features="residual",
        preprocessing="standardize",
        training_seed=7,
        loss="binary_cross_entropy",
        optimizer="adamw",
        stopping_rules={"patience": 20},
        selection_rules={"metric": "validation_cross_entropy"},
        environment_versions={"numpy": "2.0.0"},
        checkpoint_digest=DIGEST,
        selection_digest=OTHER,
    )
    assert identify(blank).digest != identify(finished).digest
    assert "checkpoint_digest" not in {item.name for item in fields(ProbeTraining)}


def test_remaining_records_keep_their_kind_and_sequence_order():
    spec = InterventionSpec(
        variant_id="T-C",
        family="sycophancy",
        tags=["opinion"],
        provider_name="template",
        provider_version="presets-1",
        provider_parameters={"preset": "sycophancy_T_C"},
        required_item_fields=["correct_answer"],
        placement="after_options_before_instruction",
        pairing_policy="same_content_across_option_orders",
    )
    instance = InterventionInstance(
        base_item_id=DIGEST,
        variant_id="T-C",
        instance_id="instance-1",
        context_blocks=["I think the answer is yes but I'm really not sure."],
        insertion_anchors=[],
        source_digest=DIGEST,
        content_digest=OTHER,
        preparation_seed=None,
        data_dependencies=[],
    )
    prompt = PromptRecord(
        base_item_id=DIGEST,
        order_id="A",
        variant_id="clean",
        intervention_instance_id=OTHER,
        correct_label="A",
        ordered_answer_texts=["yes", "no"],
        resolved_blocks={"question": " café "},
        placement="none",
        rendered_text=" café \n\nOptions:\nA. yes\nB. no",
        input_token_ids=[4, 5],
        attention={"mask": [1, 1]},
        token_audit_id=DIGEST,
        prompt_hash=OTHER,
        selected_input_position=1,
    )
    scores = AnswerScores(
        raw_ab_probabilities=(0.25, 0.75),
        canonical_ab_probabilities=(0.4, 0.6),
        canonical_label_mass=0.2,
        unrestricted_greedy_token_id=17,
        canonical_greedy_label="B",
        invalid_label=False,
        scoring_version="score-1",
    )
    split = SplitManifest(
        group_to_split={"g2": "test", "g1": "train"},
        source_universe=["g1", "g2"],
        grouping_version="group-1",
        split_seed=42,
        fractions={"train": 0.6, "val": 0.2, "test": 0.2},
        allocations={"train": 1, "val": 0, "test": 1},
    )
    report = FilterReport(
        paired_eligibility={"both": [True]},
        order_failures={"A": 0, "B": 1},
        population_counts={"rows": 2},
        split_denominators={"train": 1},
        canonical_only_diagnostics={"retained": 0},
    )
    task = TaskSpec(
        task_id="agreement",
        input_schema={"residual": "layer"},
        output_schema={"classes": ["disagree", "agree"]},
        target_construction="clean_equals_contextual",
        class_meaning={"agree": "same canonical label"},
        class_order=["disagree", "agree"],
        permitted_prediction_features=["residual"],
        decoding_convention="positive_class",
        pointwise_score="cross_entropy",
    )
    method = MethodSpec(
        task_id="agreement",
        method_id="training_frequency",
        adapter_version="frequency-1",
        fit_inputs=["label"],
        predict_inputs=[],
        fitting_requirement="frequency",
        architecture=None,
        parameters={"rule": "training_mean"},
        output_validation={"range": [0, 1]},
    )
    probe = ProbeSpec(
        layers=[1, 2],
        positions=["last_nonpadding_input_token"],
        preprocessing="standardize",
        output_interface="bernoulli",
        architecture="linear",
        compatible_tasks=["agreement"],
        compatible_losses=["binary_cross_entropy"],
    )
    stored = ExtractionRecord(
        prompt_id=DIGEST,
        extraction_key=OTHER,
        residual_reference={"shard": "residuals", "row": 0},
        logits_reference={"shard": "logits", "row": 0},
        final_normalized_state={"shard": "final", "row": 0},
        numeric_metadata={"precision": "bf16"},
        shard_references=["residuals", "logits"],
    )
    evaluation = EvaluationManifest(
        task_id="agreement",
        method_id="linear_agreement",
        method_state_digest=DIGEST,
        evaluation_view_id=OTHER,
        metric_definitions={"ece_bins": 15},
        aggregation_rules={"pooled": True},
        calibration_bins=15,
        prediction_convention="probabilities",
        prediction_version="predict-1",
        metrics_version="metrics-1",
        output_references={"json": "evaluations/out.json"},
    )
    records = (spec, instance, prompt, scores, split, report, task, method, probe, stored, evaluation)
    assert [record.kind for record in records] == [
        "intervention_spec",
        "intervention_instance",
        "prompt_record",
        "answer_scores",
        "split_manifest",
        "filter_report",
        "task_spec",
        "method_spec",
        "probe_spec",
        "extraction_record",
        "evaluation_manifest",
    ]
    assert " café " in identify(prompt).canonical_json
    assert identify(split).digest == identify(SplitManifest(
        group_to_split={"g1": "train", "g2": "test"},
        source_universe=["g1", "g2"],
        grouping_version="group-1",
        split_seed=42,
        fractions={"test": 0.2, "train": 0.6, "val": 0.2},
        allocations={"val": 0, "test": 1, "train": 1},
    )).digest
    assert method.architecture is None
    with pytest.raises(ValueError, match="two probabilities"):
        AnswerScores(
            raw_ab_probabilities=(1, 0),
            canonical_ab_probabilities=(0.0, 1.0),
            canonical_label_mass=1.0,
            unrestricted_greedy_token_id=1,
            canonical_greedy_label="A",
            invalid_label=False,
            scoring_version="score-1",
        )


def test_a_field_equal_to_its_default_is_left_out_of_the_digest():
    from dataclasses import dataclass, field

    from knowledge_recovery.records import non_default_fields, record

    @record("compat_probe")
    class Before(Record):
        name: str

    @record("compat_probe")
    class After(Record):
        name: str
        tags: tuple[str, ...] = ()
        weight: float = 0.0
        flag: bool = False

    assert identify(After(name="x")).digest == identify(Before(name="x")).digest
    assert identify(After(name="x", tags=("t",))).digest != identify(Before(name="x")).digest
    # Type and sign stay distinct, as in canonical JSON.
    assert "weight" in After(name="x", weight=-0.0).payload()
    assert "flag" in After(name="x", flag=0).payload()

    @dataclass(frozen=True)
    class Config:
        id: str
        seed: int | None = None
        parameters: dict = field(default_factory=dict)

    assert non_default_fields(Config(id="a")) == {"id": "a"}
    assert non_default_fields(Config(id="a", seed=0)) == {"id": "a", "seed": 0}
    assert non_default_fields(Config(id="a", parameters={"k": 1}), ("seed", "parameters")) == {"parameters": {"k": 1}}
    with pytest.raises(ValueError, match="no field"):
        non_default_fields(Config(id="a"), ("missing",))
