import inspect

import pytest

from knowledge_recovery.tasks import (
    AWARENESS_CLASS_ORDER,
    CONSISTENCY_CLASS_ORDER,
    AwarenessTask,
    ConsistencyTask,
    awareness_label,
    awareness_task_spec,
    check_prediction_features,
    consistency_target,
    consistency_task_spec,
)


def test_greedy_token_change_keeps_agreement_and_the_clean_target():
    clean_canonical = "A"
    contextual_canonical = "A"
    clean_greedy = 17
    contextual_greedy = 44
    assert clean_greedy != contextual_greedy
    assert "unrestricted_greedy_token" not in inspect.signature(awareness_label).parameters
    assert "probability_basis" not in inspect.signature(awareness_label).parameters
    assert "probability_basis" not in inspect.signature(consistency_target).parameters
    for _basis in ("raw", "canonical"):
        assert awareness_label(clean_canonical, contextual_canonical) == "agree"
        assert AwarenessTask().target_index(clean_canonical, contextual_canonical) == 1
        assert consistency_target(clean_canonical) == "A"
        assert ConsistencyTask().target_index(clean_canonical) == 0
    with pytest.raises(TypeError):
        AwarenessTask().target("A", "A", unrestricted_greedy_token=contextual_greedy)
    with pytest.raises(TypeError):
        consistency_target("A", probability_basis="raw")
    assert awareness_label("A", "B") == "disagree"
    assert consistency_target("B") == "B"
    assert type(consistency_target("A")) is str


def test_task_contracts_use_canonical_class_orders_and_block_clean_reference_features():
    awareness = awareness_task_spec()
    consistency = consistency_task_spec()
    assert awareness.class_order == AWARENESS_CLASS_ORDER == ("disagree", "agree")
    assert consistency.class_order == CONSISTENCY_CLASS_ORDER == ("A", "B")
    assert awareness.target_construction["not_used"] == ("unrestricted_greedy_token",)
    assert awareness.target_construction["filter_effect"] == "eligibility_only"
    assert consistency.target_construction["rule"] == "hard clean canonical argmax"
    assert "soft_clean_distribution" in consistency.target_construction["not_used"]
    assert awareness.permitted_prediction_features == ("residual",)
    assert consistency.permitted_prediction_features == ("residual", "canonical_ab_probabilities")
    assert awareness.decoding_convention["tie_prediction"] == "agree"
    assert consistency.decoding_convention["tie_prediction"] == "A"
    check_prediction_features("behavioral_awareness", [])
    check_prediction_features("behavioral_awareness", ["residual"])
    check_prediction_features("knowledge_consistency", ["residual", "canonical_ab_probabilities"])
    with pytest.raises(ValueError, match="clean_residual"):
        check_prediction_features("behavioral_awareness", ["residual", "clean_residual"])
    with pytest.raises(ValueError, match="clean_logits"):
        check_prediction_features("knowledge_consistency", ["clean_logits"])
    with pytest.raises(ValueError, match="clean_canonical_label"):
        check_prediction_features("knowledge_consistency", ["clean_canonical_label"])
