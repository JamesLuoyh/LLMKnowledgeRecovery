from knowledge_recovery.views import CleanOrder, apply_filter_gate


def order(group, split, order_id, raw, canonical, *, invalid=False):
    return CleanOrder(
        base_group_id=group,
        split=split,
        order_id=order_id,
        raw_ab_probabilities=raw,
        canonical_ab_probabilities=canonical,
        canonical_label_mass=raw[0] + raw[1],
        invalid_label=invalid,
    )


# Probabilities are (A, B). The correct label is A for correct_at_A and B for correct_at_B.
orders = [
    order("g0", "train", "correct_at_A", (0.2, 0.05), (0.9, 0.1)),
    order("g0", "train", "correct_at_B", (0.05, 0.2), (0.1, 0.9)),
]
outcome = apply_filter_gate(orders, output_directory="filter-out")
print(outcome.eligible_base_group_ids)
print(open("filter-out/filter_report.txt", encoding="utf-8").read())
