import json
import math

import pytest

from knowledge_recovery.evaluation import (
    BRIER_CONVENTION,
    awareness_metrics,
    consistency_metrics,
    metrics_document,
    roc_auc,
)


def test_hand_fixture_matches_ce_brier_k_and_ece():
    # q, r: (0.75, 1), (0.25, 0), (0.5, 1). Awareness treats 0.5 as agreement.
    # Brier terms: 0.0625, 0.0625, 0.25. Mean 0.125.
    # K terms: 0.75, 0.75, 0.5. Mean 2/3.
    # Confidences 0.75, 0.75, 0.5, all correct.
    # ECE = (2/3)*|1-0.75| + (1/3)*|1-0.5| = 1/3.
    report = awareness_metrics([0.75, 0.25, 0.5], [1, 0, 1])
    assert report.count == 3
    assert report.accuracy == 1.0
    assert report.brier == 0.125
    assert report.mean_true_class_probability == pytest.approx(2 / 3)
    assert report.cross_entropy == pytest.approx((-2 * math.log(0.75) - math.log(0.5)) / 3)
    assert report.cross_entropy_infinite is False
    assert report.ece == pytest.approx(1 / 3)
    assert report.unavailable_reason is None
    occupied = [
        (item.lower, item.count, item.accuracy, item.mean_confidence)
        for item in report.bins
        if item.count
    ]
    assert occupied == [(7 / 15, 1, 1.0, 0.5), (11 / 15, 2, 1.0, 0.75)]
    assert len(report.bins) == 15
    assert report.bins[0].lower == 0.0
    assert report.bins[-1].upper == 1.0
    assert report.bins[0].count == 0
    assert report.bins[0].accuracy is None
    assert report.bins[0].mean_confidence is None


def test_ece_uses_top_label_confidence_and_brier_is_not_summed_over_classes():
    # P(agree)=0.875 predicts agreement, which is wrong. Top-label confidence
    # is 0.875 and the true-class probability is 0.125, so those two inputs
    # fall in different bins.
    report = awareness_metrics([0.875], [0])
    assert report.accuracy == 0.0
    assert report.bins[13].count == 1
    assert report.bins[13].mean_confidence == 0.875
    assert report.bins[13].accuracy == 0.0
    assert report.bins[1].count == 0
    assert report.ece == 0.875
    assert report.mean_true_class_probability == 0.125
    assert report.brier == 0.765625

    scalar = consistency_metrics([0.25], [1])
    assert scalar.brier == 0.5625
    assert scalar.positive_class == "B"


def test_ties_follow_the_task_and_exact_zero_probability_is_infinite_cross_entropy():
    awareness = awareness_metrics([0.5], [0])
    consistency = consistency_metrics([0.5], [0])
    assert awareness.tie_prediction == "agree"
    assert consistency.tie_prediction == "A"
    assert awareness.accuracy == 0.0
    assert consistency.accuracy == 1.0
    assert awareness.brier == consistency.brier == 0.25
    assert awareness.ece == pytest.approx(0.5)
    assert consistency.ece == pytest.approx(0.5)

    impossible = consistency_metrics([0.0], [1])
    assert impossible.cross_entropy is None
    assert impossible.cross_entropy_infinite is True
    assert impossible.brier == 1.0
    assert impossible.accuracy == 0.0
    assert impossible.mean_true_class_probability == 0.0
    assert impossible.bins[14].count == 1
    assert impossible.bins[14].mean_confidence == 1.0
    document = metrics_document(impossible)
    encoded = json.dumps(document, allow_nan=False)
    assert "Infinity" not in encoded and "NaN" not in encoded
    assert document["cross_entropy"] is None
    assert document["cross_entropy_infinite"] is True
    assert document["brier_convention"] == BRIER_CONVENTION

    certain = awareness_metrics([1.0], [1])
    assert certain.cross_entropy == 0.0
    assert certain.cross_entropy_infinite is False
    assert certain.bins[14].count == 1
    assert certain.bins[14].lower == 14 / 15


def test_small_probabilities_are_not_clipped_and_zero_rows_stay_unavailable():
    tiny = 1e-300
    report = awareness_metrics([tiny], [1])
    assert report.cross_entropy == -math.log(tiny)
    assert report.cross_entropy_infinite is False

    empty = consistency_metrics([], [])
    assert empty.count == 0
    assert empty.accuracy is None
    assert empty.brier is None
    assert empty.ece is None
    assert empty.cross_entropy is None
    assert empty.mean_true_class_probability is None
    assert empty.cross_entropy_infinite is False
    assert empty.unavailable_reason is not None
    assert "denominator is 0" in empty.unavailable_reason
    assert len(empty.bins) == 15
    assert all(item.count == 0 and item.accuracy is None for item in empty.bins)
    json.dumps(metrics_document(empty), allow_nan=False)


def test_bin_edges_are_half_open_until_the_closed_right_endpoint():
    boundary = 8 / 15
    on_edge = awareness_metrics([boundary], [1])
    assert on_edge.bins[8].count == 1
    assert on_edge.bins[7].count == 0
    below = awareness_metrics([math.nextafter(boundary, 0.0)], [1])
    assert below.bins[7].count == 1
    assert below.bins[8].count == 0
    one_bin = awareness_metrics([0.2, 0.9], [0, 1], ece_bins=1)
    assert len(one_bin.bins) == 1
    assert (one_bin.bins[0].lower, one_bin.bins[0].upper, one_bin.bins[0].count) == (0.0, 1.0, 2)


def test_low_confidence_bins_stay_empty_because_top_label_confidence_is_at_least_half():
    report = awareness_metrics([0.0, 0.125, 0.5, 1.0], [0, 0, 1, 1])
    assert all(item.count == 0 for item in report.bins[:7])
    assert report.bins[7].count == 1
    assert report.bins[14].count == 2


def test_roc_auc_uses_average_ranks_and_is_undefined_for_one_class():
    # Positives 0.35 and 0.8 against negatives 0.1 and 0.4: 1 + 2 wins of 4 pairs.
    assert roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    assert roc_auc([0.2, 0.2], [0, 1]) == 0.5
    assert roc_auc([0.3, 0.3, 0.3], [0, 1, 0]) == 0.5
    assert roc_auc([0.1, 0.9], [1, 1]) is None
    assert roc_auc([], []) is None
    with pytest.raises(ValueError, match="same length"):
        roc_auc([0.5], [])
    with pytest.raises(ValueError, match="between 0 and 1"):
        roc_auc([1.5], [1])
    with pytest.raises(ValueError, match="0 or 1"):
        roc_auc([0.5], [2])


def test_row_order_does_not_change_aggregates_and_bad_rows_are_rejected():
    forward = awareness_metrics([0.75, 0.25, 0.5], [1, 0, 1])
    backward = awareness_metrics([0.5, 0.25, 0.75], [1, 0, 1])
    assert forward.brier == backward.brier
    assert forward.ece == backward.ece
    assert forward.cross_entropy == backward.cross_entropy
    with pytest.raises(ValueError, match="same length"):
        awareness_metrics([0.5], [])
    with pytest.raises(ValueError, match="between 0 and 1"):
        awareness_metrics([1.5], [1])
    with pytest.raises(ValueError, match="nonfinite|between 0 and 1"):
        awareness_metrics([math.nan], [1])
    with pytest.raises(ValueError, match="0 or 1"):
        consistency_metrics([0.5], [2])
    with pytest.raises(ValueError, match="0 or 1"):
        consistency_metrics([0.5], [True])
    with pytest.raises(TypeError, match="probabilities"):
        awareness_metrics("0.5", [1])
    with pytest.raises(ValueError, match="ece_bins"):
        awareness_metrics([0.5], [1], ece_bins=0)
