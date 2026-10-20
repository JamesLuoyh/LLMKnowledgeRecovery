import json

import pytest

from knowledge_recovery.provenance import content_digest
from knowledge_recovery.views import CleanOrder, apply_filter_gate


def _order(group, split, order_id, canonical, *, invalid=False, raw=(0.1, 0.1)):
    return CleanOrder(
        base_group_id=group,
        split=split,
        order_id=order_id,
        raw_ab_probabilities=raw,
        canonical_ab_probabilities=canonical,
        canonical_label_mass=0.2,
        invalid_label=invalid,
    )


def _fixture():
    """One passing pair, one wrong greedy option, one invalid token, and one exact 0.8 tie-to-threshold."""
    return [
        _order("q_pass", "train", "correct_at_A", (0.9, 0.1)),
        _order("q_pass", "train", "correct_at_B", (0.05, 0.95)),
        _order("q_wrong", "train", "correct_at_A", (0.1, 0.9)),
        _order("q_wrong", "train", "correct_at_B", (0.1, 0.9)),
        _order("q_invalid", "val", "correct_at_A", (0.95, 0.05), invalid=True),
        _order("q_invalid", "val", "correct_at_B", (0.05, 0.95), invalid=True),
        _order("q_boundary", "test", "correct_at_A", (0.8, 0.2)),
        _order("q_boundary", "test", "correct_at_B", (0.2, 0.8)),
    ]


def _record(directory):
    payload = json.loads((directory / "filter_report.json").read_text())
    assert payload["digest"] == content_digest(payload["record"])
    return payload["record"]


def test_counts_fall_at_each_step_and_only_a_passing_pair_is_eligible(tmp_path):
    outcome = apply_filter_gate(_fixture(), output_directory=tmp_path)
    assert outcome.eligible_base_group_ids == ("q_pass",)
    assert outcome.min_correct_prob == 0.8
    record = _record(tmp_path)
    overall = record["population_counts"]["overall"]
    assert overall["both_orders"] == {
        "before": 8,
        "after_greedy_is_option": 6,
        "after_option_probability": 4,
        "after_greedy_correct": 3,
        "after_both_orders_correct": 2,
    }
    assert overall["correct_at_A"] == {
        "before": 4,
        "after_greedy_is_option": 3,
        "after_option_probability": 2,
        "after_greedy_correct": 1,
        "after_both_orders_correct": 1,
    }
    assert overall["correct_at_B"] == {
        "before": 4,
        "after_greedy_is_option": 3,
        "after_option_probability": 2,
        "after_greedy_correct": 2,
        "after_both_orders_correct": 1,
    }
    assert overall["correct_at_A"]["after_both_orders_correct"] == overall["correct_at_B"]["after_both_orders_correct"]
    assert overall["both_orders"]["after_both_orders_correct"] == 2 * overall["correct_at_A"]["after_both_orders_correct"]
    train = record["population_counts"]["by_split"]["train"]
    assert train["both_orders"]["before"] == 4
    assert train["both_orders"]["after_both_orders_correct"] == 2
    assert record["population_counts"]["by_split"]["val"]["both_orders"]["after_greedy_is_option"] == 0
    assert record["population_counts"]["by_split"]["test"]["both_orders"]["after_greedy_is_option"] == 2
    assert record["population_counts"]["by_split"]["test"]["both_orders"]["after_option_probability"] == 0
    text = (tmp_path / "filter_report.txt").read_text()
    assert "greater than 0.8" in text
    assert "Eligible base questions: q_pass" in text
    assert "probability basis" not in text
    assert record["paired_eligibility"]["comparison"] == "greater_than"
    assert record["canonical_only_diagnostics"]["option_probability"] == "normalized_ab"


def test_input_order_does_not_change_the_counts(tmp_path):
    forward = apply_filter_gate(_fixture(), output_directory=tmp_path / "forward")
    backward = apply_filter_gate(list(reversed(_fixture())), output_directory=tmp_path / "backward")
    assert forward.eligible_base_group_ids == backward.eligible_base_group_ids
    assert _record(tmp_path / "forward")["population_counts"] == _record(tmp_path / "backward")["population_counts"]


def test_an_exact_threshold_and_an_invalid_greedy_token_stop_early():
    boundary = _order("edge", "train", "correct_at_A", (0.8, 0.2))
    invalid = _order("edge", "val", "correct_at_A", (0.99, 0.01), invalid=True)
    from knowledge_recovery.views.filters import _last_step

    assert _last_step(boundary, 0.8) == 1
    assert _last_step(invalid, 0.8) == 0


def test_incomplete_pairs_and_an_empty_input_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="missing clean orders"):
        apply_filter_gate(_fixture()[:1], output_directory=tmp_path / "partial")
    outcome = apply_filter_gate([], output_directory=tmp_path / "empty")
    assert outcome.eligible_base_group_ids == ()
    empty = _record(tmp_path / "empty")["population_counts"]["overall"]["both_orders"]
    assert empty == {
        "before": 0,
        "after_greedy_is_option": 0,
        "after_option_probability": 0,
        "after_greedy_correct": 0,
        "after_both_orders_correct": 0,
    }
