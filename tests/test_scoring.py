import array
import math

import pytest

from knowledge_recovery.answers import SCORING_VERSION, CanonicalTokens, score_binary_answers
from knowledge_recovery.records import AnswerScores


def _score(logits, token_a=1, token_b=2):
    return score_binary_answers(logits, CanonicalTokens(A=token_a, B=token_b))


def test_high_conditional_probability_can_have_low_raw_probability():
    scores = _score([20.0, 0.0, 0.0])
    assert scores.raw_ab_probabilities[0] < 1e-6
    assert scores.raw_ab_probabilities[1] < 1e-6
    assert scores.canonical_label_mass < 1e-6
    assert scores.canonical_ab_probabilities[0] == pytest.approx(0.5, abs=1e-6)
    assert scores.canonical_ab_probabilities[1] == pytest.approx(0.5, abs=1e-6)
    assert scores.canonical_greedy_label == "A"
    assert scores.unrestricted_greedy_token_id == 0
    assert scores.invalid_label is True
    assert scores.scoring_version == SCORING_VERSION


def test_canonical_tie_selects_a_and_vocabulary_tie_selects_the_smallest_id():
    # B's token id is smaller than A's. Equal logits still name label A,
    # while the vocabulary argmax names token 0.
    scores = score_binary_answers([0.0, 0.0, 0.0], CanonicalTokens(A=2, B=0))
    assert scores.canonical_greedy_label == "A"
    assert scores.unrestricted_greedy_token_id == 0
    assert scores.invalid_label is False
    assert scores.canonical_ab_probabilities[0] == scores.canonical_ab_probabilities[1]


def test_float32_ties_use_the_rounded_logits():
    rounded = array.array("f", [0.1])[0]
    scores = score_binary_answers([rounded + 1e-12, rounded, 0.0], CanonicalTokens(A=0, B=1))
    assert scores.unrestricted_greedy_token_id == 0
    assert scores.canonical_greedy_label == "A"


def test_strict_maxima_select_the_winning_label_and_mark_it_valid():
    chosen_a = _score([0.0, 2.0, 1.0, 0.0])
    assert chosen_a.canonical_greedy_label == "A"
    assert chosen_a.unrestricted_greedy_token_id == 1
    assert chosen_a.invalid_label is False
    assert chosen_a.canonical_ab_probabilities[0] > chosen_a.canonical_ab_probabilities[1]
    assert chosen_a.raw_ab_probabilities[0] > chosen_a.raw_ab_probabilities[1]

    chosen_b = _score([0.0, 1.0, 4.0, 0.0])
    assert chosen_b.canonical_greedy_label == "B"
    assert chosen_b.unrestricted_greedy_token_id == 2
    assert chosen_b.invalid_label is False


def test_raw_mass_matches_the_canonical_conditional():
    scores = _score([0.0, 1.0, -1.0, 0.5])
    raw_a, raw_b = scores.raw_ab_probabilities
    canonical_a, canonical_b = scores.canonical_ab_probabilities
    assert canonical_a + canonical_b == pytest.approx(1.0, abs=1e-6)
    assert raw_a + raw_b == pytest.approx(scores.canonical_label_mass, abs=1e-5)
    assert raw_a == pytest.approx(canonical_a * scores.canonical_label_mass, abs=1e-5)
    assert raw_b == pytest.approx(canonical_b * scores.canonical_label_mass, abs=1e-5)
    assert scores.canonical_label_mass < 1.0
    assert isinstance(scores, AnswerScores)
    assert all(type(item) is float for item in scores.raw_ab_probabilities)
    assert type(scores.canonical_label_mass) is float


def test_equal_inputs_produce_equal_scores_without_a_vocabulary_probability_vector():
    logits = [0.0, 0.25, -0.25, 1.0]
    assert _score(logits) == _score(list(logits))
    assert len(_score(logits).raw_ab_probabilities) == 2
    assert not hasattr(_score(logits), "vocabulary_probabilities")


def test_token_ids_and_logits_are_rejected_when_they_break_the_contract():
    with pytest.raises(ValueError, match="distinct"):
        CanonicalTokens(A=1, B=1)
    with pytest.raises(ValueError, match="non-negative"):
        CanonicalTokens(A=True, B=2)
    with pytest.raises(ValueError, match="vocabulary of size"):
        _score([0.0, 1.0], token_a=0, token_b=2)
    with pytest.raises(ValueError, match="nonfinite"):
        _score([0.0, math.inf, 0.0])
    with pytest.raises(ValueError, match="nonfinite"):
        _score([0.0, 1e40, 0.0])
    with pytest.raises(ValueError, match="one value per vocabulary token"):
        _score([])
    with pytest.raises(TypeError, match="real numbers"):
        _score("0 1 2")
    with pytest.raises(TypeError, match="CanonicalTokens"):
        score_binary_answers([0.0, 1.0, 0.0], (1, 2))
