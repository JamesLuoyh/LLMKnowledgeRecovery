import pytest

from knowledge_recovery.evaluation import BehaviorRow, behavior_tables

# Clean answers that are correct in both orders: q1 and q2.
# q3 is correct only at A, so a flip there stays outside the changed-answer denominator.
# q2 / T-C / i2 keeps the canonical label and changes the raw token to an invalid id.


def _row(group, split, variant, instance, order, canonical, raw_label, raw_id):
    return BehaviorRow(
        base_group_id=group,
        split=split,
        variant_id=variant,
        instance_id=instance,
        order_id=order,
        canonical_label=canonical,
        raw_token_label=raw_label,
        raw_token_id=raw_id,
    )


def _pair(group, split, variant, instance, answers):
    order_a, order_b = answers
    return [
        _row(group, split, variant, instance, "correct_at_A", *order_a),
        _row(group, split, variant, instance, "correct_at_B", *order_b),
    ]


def _rows():
    clean = {
        "q1": (("A", "A", 1), ("B", "B", 2)),
        "q2": (("A", "A", 1), ("B", "B", 2)),
        "q3": (("A", "A", 1), ("A", "A", 4)),
        "q4": (("B", "B", 3), ("A", "A", 4)),
    }
    first = {
        "q1": (("B", "B", 3), ("B", "B", 2)),
        "q2": (("A", "A", 1), ("A", "A", 4)),
        "q3": (("B", "B", 3), ("B", "B", 2)),
        "q4": (("B", "B", 3), ("A", "A", 4)),
    }
    second = {
        "q1": (("B", "B", 3), ("A", "A", 4)),
        "q2": (("A", None, 9), ("B", "B", 2)),
        "q3": (("B", "B", 3), ("B", "B", 2)),
        "q4": (("B", "B", 3), ("A", "A", 4)),
    }
    rows = []
    for group, split in (("q1", "train"), ("q2", "train"), ("q3", "train"), ("q4", "val")):
        rows.extend(_pair(group, split, "clean", "clean", clean[group]))
        rows.extend(_pair(group, split, "T-C", "i1", first[group]))
        rows.extend(_pair(group, split, "T-C", "i2", second[group]))
    return rows


def _tables():
    return {
        table.name: table
        for table in behavior_tables(
            _rows(),
            raw_retained=["q1", "q2"],
            canonical_retained=["q1", "q2", "q3"],
        )
    }


def _instance(table, variant, instance):
    found = [item for item in table.instances if item.variant_id == variant and item.instance_id == instance]
    assert len(found) == 1
    return found[0]


def test_population_denominators_and_named_raw_and_canonical_tables():
    tables = _tables()
    assert len(tables) == 18
    assert set(tables) == {
        f"{scoring}__{population}__{split}"
        for scoring in ("raw_token", "canonical_label")
        for population in ("all_valid", "raw_retained", "canonical_retained")
        for split in ("train", "val", "test")
    }
    canonical = tables["canonical_label__all_valid__train"]
    raw = tables["raw_token__all_valid__train"]
    assert canonical.base_questions == 3
    assert (canonical.clean_correct_at_A.count, canonical.clean_correct_at_A.denominator) == (3, 3)
    assert (canonical.clean_correct_at_B.count, canonical.correct_both_orders.count) == (2, 2)
    assert canonical.correct_both_orders.value == 2 / 3
    assert tables["raw_token__raw_retained__train"].base_questions == 2
    assert tables["raw_token__raw_retained__train"].correct_both_orders.value == 1
    assert tables["canonical_label__canonical_retained__train"].clean_correct_at_A.denominator == 3
    assert raw.clean_correct_at_A.count == canonical.clean_correct_at_A.count
    first = _instance(canonical, "T-C", "i1")
    assert (first.correct_at_A.changed.count, first.correct_at_B.changed.count) == (1, 1)
    assert first.changed_both_orders.count == 0
    assert first.changed_either_order.count == 2
    assert first.changed_either_order.count == (
        first.correct_at_A.changed.count + first.correct_at_B.changed.count - first.changed_both_orders.count
    )
    second = _instance(canonical, "T-C", "i2")
    assert (second.correct_at_A.changed.count, second.correct_at_B.changed.count, second.changed_both_orders.count) == (1, 1, 1)
    assert second.correct_at_A.invalid_transitions.count == 0
    raw_second = _instance(raw, "T-C", "i2")
    assert raw_second.correct_at_A.changed.count == 2
    assert raw_second.correct_at_A.valid_switches.count == 1
    assert raw_second.correct_at_A.invalid_transitions.count == 1
    assert raw_second.changed_both_orders.count == 1
    assert raw_second.changed_both_orders.count <= min(
        raw_second.correct_at_A.changed.count, raw_second.correct_at_B.changed.count
    )


def test_clean_deviation_is_zero_and_a_zero_denominator_is_unavailable():
    tables = _tables()
    clean = _instance(tables["raw_token__all_valid__train"], "clean", "clean")
    assert clean.correct_at_A.changed.count == 0
    assert clean.correct_at_A.changed.value == 0.0
    assert clean.changed_both_orders.value == 0.0
    canonical_clean = _instance(tables["canonical_label__all_valid__train"], "clean", "clean")
    assert canonical_clean.correct_at_B.changed.value == 0.0
    val = tables["canonical_label__all_valid__val"]
    assert val.clean_correct_at_A.value == 0.0
    assert val.correct_both_orders.value == 0.0
    withheld = _instance(val, "T-C", "i1")
    assert withheld.changed_both_orders.count == 0
    assert withheld.changed_both_orders.value is None
    assert "zero denominator" in withheld.changed_both_orders.unavailable_reason
    empty = tables["raw_token__raw_retained__val"]
    assert empty.base_questions == 0
    assert empty.clean_correct_at_A.value is None
    assert empty.clean_correct_at_A.count == 0
    assert "0 base questions" in empty.clean_correct_at_A.unavailable_reason
    assert tables["canonical_label__all_valid__test"].base_questions == 0


def test_distinct_invalid_tokens_are_invalid_transitions_and_inputs_are_checked():
    rows = [
        *_pair("q", "train", "clean", "clean", (("A", "A", 1), ("B", "B", 2))),
        *_pair("q", "train", "T-C", "i1", (("A", None, 9), ("B", None, 8))),
    ]
    table = behavior_tables(rows, raw_retained=["q"], canonical_retained=["q"])
    raw = next(item for item in table if item.name == "raw_token__all_valid__train")
    changed = _instance(raw, "T-C", "i1")
    assert changed.correct_at_A.invalid_transitions.count == 1
    assert changed.correct_at_B.invalid_transitions.count == 1
    assert changed.changed_both_orders.count == 1
    assert changed.correct_at_A.valid_switches.count == 0
    with pytest.raises(ValueError, match="unknown base group"):
        behavior_tables(rows, raw_retained=["missing"], canonical_retained=["q"])
    with pytest.raises(ValueError, match="missing an order"):
        behavior_tables(rows[:1], raw_retained=[], canonical_retained=[])
