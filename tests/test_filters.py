import json

import pytest

from knowledge_recovery.errors import FilterPolicyDecisionRequired
from knowledge_recovery.provenance import content_digest
from knowledge_recovery.views import (
    SCOPE_FIELDS,
    CleanOrder,
    ProbabilityConsistencyError,
    apply_filter_gate,
    record_decision,
)

SCOPE = {
    "dataset_adapter": "sycophancy",
    "source_snapshot": "snapshot-1",
    "model_content_signature": "model-1",
    "prompt_contract_digest": "prompt-1",
    "canonical_answers": ["A", "B"],
    "require_both_orders": True,
}


def _order(group, split, order_id, raw, canonical, mass=0.25, invalid=False):
    return CleanOrder(
        base_group_id=group,
        split=split,
        order_id=order_id,
        raw_ab_probabilities=raw,
        canonical_ab_probabilities=canonical,
        canonical_label_mass=mass,
        invalid_label=invalid,
    )


def _fixture():
    """High conditional probability with low raw probability on g0.

    Correct-label probability is index 0 at correct-at-A and index 1 at
    correct-at-B. g0 passes 0.8 only after conditioning on A/B. g1 passes on
    both bases, and its A row is an invalid greedy label with a high canonical
    probability, so invalid labels stay in the diagnostic rather than the mask.
    """
    return [
        _order("g0", "train", "correct_at_A", (0.2, 0.8), (0.8, 0.2)),
        _order("g0", "train", "correct_at_B", (0.8, 0.2), (0.2, 0.8)),
        _order("g1", "train", "correct_at_A", (0.9, 0.1), (0.9, 0.1), invalid=True),
        _order("g1", "train", "correct_at_B", (0.1, 0.85), (0.15, 0.85)),
        _order("g2", "val", "correct_at_A", (0.2, 0.8), (0.8, 0.2)),
        _order("g2", "val", "correct_at_B", (0.9, 0.1), (0.6, 0.4)),
        _order("g3", "test", "correct_at_A", (0.0, 1.0), (0.0, 1.0), mass=0.5),
        _order("g3", "test", "correct_at_B", (1.0, 0.0), (1.0, 0.0), mass=0.5),
    ]


def _record(directory):
    payload = json.loads((directory / "filter_report.json").read_text())
    assert payload["digest"] == content_digest(payload["record"])
    return payload["record"]


def test_high_conditional_and_low_raw_probability_keeps_raw_only_at_zero(tmp_path):
    with pytest.raises(FilterPolicyDecisionRequired) as caught:
        apply_filter_gate(_fixture(), output_directory=tmp_path, scope=SCOPE, requested_basis="canonical")
    assert caught.value.exit_code != 0
    assert caught.value.decision_state == "unset"
    record = _record(tmp_path)
    text = (tmp_path / "filter_report.txt").read_text()
    assert "The probability basis is never inferred from the retention rate." in text
    assert "Requested basis canonical was not applied." in text
    assert "probability_basis" not in record
    assert not (tmp_path / "applied_decision.json").exists()
    retention = record["population_counts"]["retention"]
    assert retention["raw"]["retained"] == {"count": 1, "denominator": 4, "percentage": 0.25, "unavailable_reason": None}
    assert retention["canonical"]["retained"]["count"] == 2
    assert retention["overlap"] == {
        "both": {"count": 1, "denominator": 4, "percentage": 0.25, "unavailable_reason": None},
        "raw_only": {"count": 0, "denominator": 4, "percentage": 0.0, "unavailable_reason": None},
        "canonical_only": {"count": 1, "denominator": 4, "percentage": 0.25, "unavailable_reason": None},
        "neither": {"count": 2, "denominator": 4, "percentage": 0.5, "unavailable_reason": None},
    }
    assert record["canonical_only_diagnostics"]["canonical_only_base_groups"] == ["g0"]
    assert record["canonical_only_diagnostics"]["raw_only_orders"] == []
    masks = {item["base_group_id"]: item["eligible"] for item in record["paired_eligibility"]["raw"]}
    assert masks == {"g0": False, "g1": True, "g2": False, "g3": False}
    canonical = {item["base_group_id"]: item["eligible"] for item in record["paired_eligibility"]["canonical"]}
    assert canonical["g0"] is True and canonical["g1"] is True
    assert record["population_counts"]["units"] == {
        "base_questions": 4,
        "intervention_instances": 4,
        "paired_order_rows": 4,
        "prompt_rows": 8,
    }
    invalid = record["canonical_only_diagnostics"]["invalid_labels"]
    assert invalid["count"] == 1 and invalid["denominator"] == 8
    mass = record["canonical_only_diagnostics"]["canonical_mass"]
    assert mass == {"count": 8, "mean": 0.3125, "minimum": 0.25, "maximum": 0.5}
    train = record["order_failures"]["raw"]["train"]
    assert train["correct_at_A"]["count"] == 1
    assert train["either"]["count"] == train["correct_at_A"]["count"] + train["correct_at_B"]["count"] - train["both"]["count"]
    val = record["order_failures"]["canonical"]["val"]
    assert (val["correct_at_A"]["count"], val["correct_at_B"]["count"], val["either"]["count"], val["both"]["count"]) == (0, 1, 1, 0)
    assert "Raw-only orders: 0" in text
    assert "raw_only: 0/4" in text


def test_input_order_does_not_change_the_saved_report(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    orders = _fixture()
    with pytest.raises(FilterPolicyDecisionRequired):
        apply_filter_gate(orders, output_directory=first, scope=SCOPE)
    with pytest.raises(FilterPolicyDecisionRequired):
        apply_filter_gate(list(reversed(orders)), output_directory=second, scope=SCOPE)
    forward = _record(first)
    backward = _record(second)
    assert forward["paired_eligibility"] == backward["paired_eligibility"]
    assert [item["base_group_id"] for item in forward["paired_eligibility"]["raw"]] == ["g0", "g1", "g2", "g3"]


def test_raw_only_retention_fails_before_a_basis_can_be_chosen(tmp_path):
    orders = [
        _order("bad", "train", "correct_at_A", (0.95, 0.05), (0.4, 0.6)),
        _order("bad", "train", "correct_at_B", (0.05, 0.95), (0.7, 0.3)),
    ]
    with pytest.raises(ProbabilityConsistencyError) as caught:
        apply_filter_gate(
            orders,
            output_directory=tmp_path,
            scope=SCOPE,
            decision=None,
            requested_basis="raw",
        )
    assert caught.value.exit_code != 0
    assert caught.value.report_id == json.loads((tmp_path / "filter_report.json").read_text())["digest"]
    assert {item["order_id"] for item in caught.value.raw_only_orders} == {"correct_at_A", "correct_at_B"}
    assert "was not inferred" in str(caught.value)
    assert (tmp_path / "filter_report.txt").is_file()
    assert not (tmp_path / "applied_decision.json").exists()
    record = _record(tmp_path)
    assert record["population_counts"]["retention"]["overlap"]["raw_only"]["count"] == 1


def test_a_compatible_decision_is_inherited_across_threshold_and_split_seed(tmp_path):
    first = tmp_path / "first"
    with pytest.raises(FilterPolicyDecisionRequired) as caught:
        apply_filter_gate(_fixture(), output_directory=first, scope={**SCOPE, "split_seed": 1})
    decision = record_decision(
        probability_basis="raw",
        filter_report_digest=caught.value.report_id,
        scope={**SCOPE, "split_seed": 1, "training_seed": 7},
        timestamp="2026-01-01T00:00:00Z",
        user_instruction="use raw",
    )
    assert "split_seed" not in decision.scope
    assert decision.probability_basis == "raw"
    applied = apply_filter_gate(
        _fixture(),
        output_directory=tmp_path / "second",
        scope={**SCOPE, "split_seed": 99},
        min_correct_prob=0.5,
        decision=decision,
    )
    assert applied.probability_basis == "raw"
    assert applied.original_report_digest == caught.value.report_id
    assert applied.current_report_digest != caught.value.report_id
    assert applied.min_correct_prob == 0.5
    saved = json.loads((tmp_path / "second" / "applied_decision.json").read_text())
    assert saved["probability_basis"] == "raw"
    assert saved["original_decision_id"] == applied.original_decision_id
    assert saved["current_report_digest"] == applied.current_report_digest


def test_scope_change_or_contradiction_reopens_the_gate_after_the_report(tmp_path):
    with pytest.raises(FilterPolicyDecisionRequired) as caught:
        apply_filter_gate(_fixture(), output_directory=tmp_path / "first", scope=SCOPE)
    decision = record_decision(
        probability_basis="canonical",
        filter_report_digest=caught.value.report_id,
        scope=SCOPE,
        timestamp="2026-01-01T00:00:00Z",
        user_instruction="use canonical",
    )
    contradicted = tmp_path / "contradicted"
    with pytest.raises(FilterPolicyDecisionRequired) as conflict:
        apply_filter_gate(
            _fixture(),
            output_directory=contradicted,
            scope=SCOPE,
            decision=decision,
            requested_basis="raw",
        )
    assert conflict.value.decision_state == "contradiction"
    assert conflict.value.exit_code != 0
    assert (contradicted / "filter_report.json").is_file()
    assert not (contradicted / "applied_decision.json").exists()
    reopened = tmp_path / "reopened"
    with pytest.raises(FilterPolicyDecisionRequired) as scoped:
        apply_filter_gate(
            _fixture(),
            output_directory=reopened,
            scope={**SCOPE, "model_content_signature": "model-2"},
            decision=decision,
            requested_basis="canonical",
        )
    assert scoped.value.decision_state == "unset"
    assert "model_content_signature differs" in (reopened / "filter_report.txt").read_text()
    assert set(SCOPE_FIELDS) == {
        "dataset_adapter",
        "source_snapshot",
        "model_content_signature",
        "prompt_contract_digest",
        "canonical_answers",
        "require_both_orders",
    }


def test_reset_and_incomplete_pairs_are_explicit(tmp_path):
    with pytest.raises(FilterPolicyDecisionRequired) as caught:
        apply_filter_gate(_fixture(), output_directory=tmp_path / "first", scope=SCOPE)
    decision = record_decision(
        probability_basis="raw",
        filter_report_digest=caught.value.report_id,
        scope=SCOPE,
        timestamp="2026-01-01T00:00:00Z",
        user_instruction="use raw",
    )
    reset = tmp_path / "reset"
    with pytest.raises(FilterPolicyDecisionRequired) as opened:
        apply_filter_gate(
            _fixture(),
            output_directory=reset,
            scope=SCOPE,
            decision=decision,
            reset_decision=True,
        )
    assert opened.value.decision_state == "unset"
    assert "reset" in (reset / "filter_report.txt").read_text()
    with pytest.raises(ValueError, match="missing clean orders"):
        apply_filter_gate(_fixture()[:1], output_directory=tmp_path / "partial", scope=SCOPE)
    empty = tmp_path / "empty"
    with pytest.raises(FilterPolicyDecisionRequired):
        apply_filter_gate([], output_directory=empty, scope=SCOPE)
    record = _record(empty)
    assert record["population_counts"]["retention"]["raw"]["retained"]["percentage"] is None
    assert "zero denominator" in record["population_counts"]["retention"]["raw"]["retained"]["unavailable_reason"]
