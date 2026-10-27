"""Rank MMLU incorrect choices by the model's own preference.

Each question is asked once per cyclic rotation of its distinct choices, so
every choice sits at every letter once. The letter probabilities are
normalized over the shown letters and averaged per choice text. Averaging
over rotations removes the position preference that one fixed order would
carry into the ranking. No model or tokenizer is imported here; the stage
passes in callables.
"""

from __future__ import annotations

import string
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy

from knowledge_recovery.records import BaseItem

RANKING_VERSION = "mmlu_distractor_rank_v1"
SELECTIONS = ("seeded_random", "most_preferred", "least_preferred")
PREFERRED_SELECTIONS = ("most_preferred", "least_preferred")


def distinct_choices(item: BaseItem) -> tuple[str, ...]:
    """The item's choice texts in source order, each kept once."""
    return tuple(dict.fromkeys(item.source_metadata["choices"]))


def rotations(choices: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Every cyclic order of ``choices``; each text appears once at each letter."""
    return tuple(tuple(choices[shift:]) + tuple(choices[:shift]) for shift in range(len(choices)))


def render_multiple_choice(question: str, ordered: Sequence[str], *, instruction: str) -> str:
    """Same layout as the binary prompt, with one lettered line per choice."""
    lines = "\n".join(f"{letter}. {text}" for letter, text in zip(string.ascii_uppercase, ordered))
    return f"{question}\n\nOptions:\n{lines}\n\n{instruction}"


def select_preferred_incorrect(correct: str, scores: Mapping[str, float], *, selection: str) -> str | None:
    """Highest- or lowest-scoring incorrect choice. Ties go to the first text in sorted order."""
    if selection not in PREFERRED_SELECTIONS:
        raise ValueError(f"selection must be one of {PREFERRED_SELECTIONS}")
    candidates = sorted(text for text in scores if text != correct)
    if not candidates:
        return None
    sign = -1.0 if selection == "most_preferred" else 1.0
    return min(candidates, key=lambda text: (sign * scores[text], text))


def rank_items(
    items: Sequence[BaseItem],
    *,
    instruction: str,
    encode: Callable[[str], Sequence[int]],
    letter_logits: Callable[[Sequence[Sequence[int]], int], numpy.ndarray],
    batch_size: int,
    progress: Callable[[int], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return ``{base_group_id: {"choices", "correct", "mean_probability"}}``.

    ``encode`` turns user text into model input ids. ``letter_logits(batch,
    n)`` returns one row per prompt holding the logits of letters A..n.
    Prompts are grouped by choice count and sorted by length before batching.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    prompts: list[tuple[int, str, tuple[str, ...], tuple[int, ...]]] = []
    for position, item in enumerate(items):
        for ordered in rotations(distinct_choices(item)):
            text = render_multiple_choice(item.question, ordered, instruction=instruction)
            prompts.append((position, item.duplicate_group_id, ordered, tuple(encode(text))))
    totals: list[dict[str, float]] = [dict.fromkeys(distinct_choices(item), 0.0) for item in items]
    order = sorted(range(len(prompts)), key=lambda index: (len(prompts[index][2]), len(prompts[index][3])))
    start = 0
    while start < len(order):
        width = len(prompts[order[start]][2])
        batch = [order[start]]
        while (
            len(batch) < batch_size
            and start + len(batch) < len(order)
            and len(prompts[order[start + len(batch)]][2]) == width
        ):
            batch.append(order[start + len(batch)])
        logits = numpy.asarray(letter_logits([prompts[index][3] for index in batch], width), dtype=numpy.float64)
        if logits.shape != (len(batch), width):
            raise RuntimeError("letter_logits must return one row of letter logits per prompt")
        shifted = numpy.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = shifted / shifted.sum(axis=1, keepdims=True)
        for row, index in enumerate(batch):
            position, _group, ordered, _ids = prompts[index]
            for column, text in enumerate(ordered):
                totals[position][text] += float(probabilities[row, column])
        start += len(batch)
        if progress is not None:
            progress(len(batch))
    ranked: dict[str, dict[str, Any]] = {}
    for item, total in zip(items, totals, strict=True):
        count = len(total)
        ranked[item.duplicate_group_id] = {
            "choices": list(total),
            "correct": item.correct_answer,
            "mean_probability": {text: value / count for text, value in total.items()},
        }
    return ranked
