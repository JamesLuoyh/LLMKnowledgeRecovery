"""Named slots and one exact character anchor. Missing or repeated anchors fail."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from knowledge_recovery.config import NAMED_PLACEMENTS
from knowledge_recovery.errors import KnowledgeRecoveryError
from knowledge_recovery.provenance import content_digest


class PlacementError(KnowledgeRecoveryError):
    """An insertion slot or character anchor cannot be resolved exactly."""

    def __init__(self, *, reason: str, detail: str) -> None:
        self.reason = reason
        super().__init__(
            f"PlacementError: {reason}.\n{detail}\n"
            "The insertion was not approximated."
        )


def resolve_placement(question: str, context: str | None, placement: Any) -> tuple[str, Mapping[str, Any]]:
    """Return the question text to render and the recorded placement.

    A named slot leaves the question unchanged. A character anchor inserts
    ``context`` at the single matching span and records that span.
    """
    if context is None:
        return question, {"mode": "omitted", "context": None}
    slot = _named_slot(placement)
    if slot is not None:
        if slot not in NAMED_PLACEMENTS:
            raise PlacementError(
                reason="unknown_slot",
                detail=(
                    f"Slot: {slot!r}\n"
                    f"Named slots: {', '.join(NAMED_PLACEMENTS)}."
                ),
            )
        return question, {"mode": "named_slot", "slot": slot}
    anchor = _anchor(placement)
    start, end, occurrences = _span(question, anchor["expected_text"])
    relation = anchor["relation"]
    if relation == "before":
        rendered = question[:start] + context + question[start:]
    elif relation == "after":
        rendered = question[:end] + context + question[end:]
    else:
        raise PlacementError(
            reason="unknown_relation",
            detail=f"Relation: {relation!r}. Use before or after.",
        )
    return rendered, {
        "mode": "character_anchor",
        "expected_text": anchor["expected_text"],
        "expected_text_digest": content_digest({
            "kind": "anchor_text",
            "schema": 1,
            "text": anchor["expected_text"],
        }),
        "start": start,
        "end": end,
        "occurrences": occurrences,
        "relation": relation,
    }


def _named_slot(placement: Any) -> str | None:
    if type(placement) is str:
        return placement
    if isinstance(placement, Mapping) and placement.get("mode") != "character_anchor" and "slot" in placement:
        return str(placement["slot"])
    return None


def _anchor(placement: Any) -> Mapping[str, Any]:
    if not isinstance(placement, Mapping) or placement.get("mode") != "character_anchor":
        raise PlacementError(
            reason="unknown_slot",
            detail=f"Placement: {placement!r}\nNamed slots: {', '.join(NAMED_PLACEMENTS)}.",
        )
    expected = placement.get("expected_text")
    relation = placement.get("relation")
    if type(expected) is not str or type(relation) is not str:
        raise PlacementError(
            reason="missing",
            detail="A character anchor needs expected_text and relation.",
        )
    return placement


def _span(question: str, expected: str) -> tuple[int, int, int]:
    if expected == "":
        raise PlacementError(reason="missing", detail="Anchor text is empty.")
    starts = []
    cursor = 0
    while True:
        found = question.find(expected, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + 1
    if not starts:
        raise PlacementError(
            reason="missing",
            detail=f"Anchor {expected!r} does not occur in the question.",
        )
    if len(starts) != 1:
        raise PlacementError(
            reason="ambiguous",
            detail=f"Anchor {expected!r} occurs {len(starts)} times in the question.",
        )
    start = starts[0]
    return start, start + len(expected), 1
