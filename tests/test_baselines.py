import inspect
import json

import pytest

from knowledge_recovery.methods import (
    contextual_answer_method,
    contextual_answer_spec,
    fit_contextual_answer,
    fit_training_frequency,
    frequency_state_digest,
    registered_pairs,
    require_task,
    training_frequency_spec,
)
from knowledge_recovery.methods.registry import MethodTaskMismatch
from knowledge_recovery.provenance import identify
from knowledge_recovery.tasks import consistency_task_spec

VIEW = "ab" * 32


def test_frequency_is_the_uniform_training_mean_and_ignores_residuals():
    state = fit_training_frequency([1, 0, 1], training_view_id=VIEW)
    assert state.probability == 2 / 3
    assert state.positive_count == 2 and state.count == 3
    assert state.degenerate is False and state.degeneracy is None
    assert state.selected_layer is None and state.checkpoint_digest is None
    assert state.predict(2) == (2 / 3, 2 / 3)
    assert "residual" not in inspect.signature(fit_training_frequency).parameters
    assert list(inspect.signature(state.predict).parameters) == ["row_count"]
    with pytest.raises(TypeError):
        fit_training_frequency([1, 0], training_view_id=VIEW, residuals=[[0.1, 0.2]])
    with pytest.raises(ValueError, match="row_count"):
        state.predict([[0.1, 0.2], [0.3, 0.4]])
    other_view = fit_training_frequency([1, 0, 1], training_view_id="cd" * 32)
    assert frequency_state_digest(state) == frequency_state_digest(
        fit_training_frequency([1, 0, 1], training_view_id=VIEW)
    )
    assert frequency_state_digest(state) != frequency_state_digest(other_view)
    spec = training_frequency_spec()
    assert spec.parameters["optimizer"] is None
    assert spec.parameters["selected_layer"] is None
    assert spec.architecture is None
    assert "residual" not in spec.fit_inputs


def test_one_class_views_are_labeled_degenerate_and_an_empty_view_is_unavailable():
    agreed = fit_training_frequency([1, 1, 1], training_view_id=VIEW)
    refused = fit_training_frequency([0, 0], training_view_id=VIEW)
    assert agreed.available is True and agreed.degenerate is True and agreed.degeneracy == "all_agree"
    assert agreed.probability == 1.0
    assert refused.probability == 0.0 and refused.degeneracy == "all_disagree"
    assert agreed.predict(1) == (1.0,)
    empty = fit_training_frequency([], training_view_id=VIEW)
    assert empty.available is False and empty.probability is None
    assert empty.unavailable_reason == "empty training view"
    assert empty.degenerate is False
    with pytest.raises(ValueError, match="empty training view"):
        empty.predict(3)
    document = {
        "selected_layer": agreed.selected_layer,
        "checkpoint_digest": agreed.checkpoint_digest,
        "degeneracy": agreed.degeneracy,
    }
    assert '"selected_layer": null' in json.dumps(document)


def test_contextual_answer_returns_the_cached_distribution_for_that_row():
    method = contextual_answer_method(task_spec_digest=identify(consistency_task_spec()).digest)
    clean = (0.25, 0.75)
    contextual = (0.9, 0.1)
    assert method.predict([clean, contextual]) == (clean, contextual)
    assert method.selected_layer is None and method.checkpoint_digest is None
    assert "training_view_id" not in inspect.signature(contextual_answer_method).parameters
    with pytest.raises(TypeError, match="no fitting stage"):
        fit_contextual_answer([clean])
    spec = contextual_answer_spec()
    assert spec.fitting_requirement == "none"
    assert spec.parameters["selected_layer"] is None
    assert spec.architecture is None
    assert spec.predict_inputs == ("canonical_ab_probabilities",)
    saved = json.dumps({"selected_layer": method.selected_layer})
    assert '"selected_layer": null' in saved


def test_an_awareness_method_cannot_be_used_for_consistency_when_shapes_match():
    frequency = fit_training_frequency([1, 0], training_view_id=VIEW)
    contextual = contextual_answer_method(task_spec_digest=identify(consistency_task_spec()).digest)
    awareness_output = frequency.predict(2)
    consistency_output = contextual.predict([(0.4, 0.6), (0.2, 0.8)])
    assert len(awareness_output) == len(consistency_output)
    with pytest.raises(MethodTaskMismatch, match="tensor shapes"):
        require_task(frequency, "knowledge_consistency")
    with pytest.raises(MethodTaskMismatch, match="tensor shapes"):
        require_task(contextual, "behavioral_awareness")

    class SameShape:
        task_id = "behavioral_awareness"
        method_id = "training_frequency"

        def predict(self, rows):
            return tuple((0.25, 0.75) for _ in rows)

    decoy = SameShape()
    assert len(decoy.predict([None, None])[0]) == len(consistency_output[0])
    with pytest.raises(MethodTaskMismatch, match="tensor shapes"):
        require_task(decoy, "knowledge_consistency")
    assert registered_pairs() == (
        ("behavioral_awareness", "training_frequency"),
        ("knowledge_consistency", "contextual_answer"),
    )
    with pytest.raises(MethodTaskMismatch, match="linear_agreement"):
        require_task(type("Probe", (), {"task_id": "behavioral_awareness", "method_id": "linear_agreement"})(), "behavioral_awareness")
