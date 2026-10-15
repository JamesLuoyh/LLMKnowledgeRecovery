"""Render exact user text, then represent chat wrapping as data.

Token IDs are not produced here. The audit pull request owns tokenization.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from knowledge_recovery.prompts.placement import resolve_placement
from knowledge_recovery.records import BaseItem, InterventionInstance, freeze

ORDERS = (("correct_at_A", "A"), ("correct_at_B", "B"))


@dataclass(frozen=True)
class RenderedPrompt:
    """One order of one intervention instance, without token IDs."""

    base_item_id: str
    order_id: str
    variant_id: str
    intervention_instance_id: str
    correct_label: str
    ordered_answer_texts: tuple[str, str]
    user_text: str
    chat: bool
    messages: tuple[Mapping[str, str], ...]
    add_generation_prompt: bool
    special_tokens: str
    blocks: Mapping[str, Any]


def render_user_text(
    *,
    question: str,
    option_a: str,
    option_b: str,
    context: str | None,
    instruction: str,
    placement: Any,
) -> str:
    """Join question, options, optional context, and the answer instruction.

    Parts are separated by LF blank lines. Clean context is omitted together
    with its separator. Nothing is stripped, quoted, or given a trailing newline.
    """
    rendered_question, resolved = resolve_placement(question, context, placement)
    options = f"Options:\nA. {option_a}\nB. {option_b}"
    slot = resolved.get("slot") if resolved["mode"] == "named_slot" else resolved["mode"]
    if resolved["mode"] == "character_anchor":
        parts = [rendered_question, options, instruction]
    elif slot == "before_question":
        parts = [context, rendered_question, options, instruction]
    elif slot == "after_question_before_options":
        parts = [rendered_question, context, options, instruction]
    else:
        parts = [rendered_question, options, context, instruction]
    return "\n\n".join(part for part in parts if part is not None)


def render_orders(
    item: BaseItem,
    instance: InterventionInstance,
    *,
    instruction: str,
    chat: bool,
    system_message: str | None,
    placement: Any | None = None,
) -> tuple[RenderedPrompt, RenderedPrompt]:
    """Render both option orders from one instance. The context text is shared."""
    context = _context(instance)
    chosen = instance.insertion_anchors[0] if placement is None and instance.insertion_anchors else placement
    if chosen is None:
        chosen = "after_options_before_instruction"
    if isinstance(chosen, Mapping) and chosen.get("mode") != "character_anchor" and "slot" in chosen:
        chosen = chosen["slot"]
    prompts = []
    for order_id, label in ORDERS:
        option_a, option_b = _options(item, label)
        user_text = render_user_text(
            question=item.question,
            option_a=option_a,
            option_b=option_b,
            context=context,
            instruction=instruction,
            placement=chosen,
        )
        messages, add_generation_prompt, special_tokens = _chat(
            user_text, chat=chat, system_message=system_message
        )
        _rendered_question, resolved = resolve_placement(item.question, context, chosen)
        prompts.append(RenderedPrompt(
            base_item_id=item.duplicate_group_id,
            order_id=order_id,
            variant_id=instance.variant_id,
            intervention_instance_id=instance.instance_id,
            correct_label=label,
            ordered_answer_texts=(option_a, option_b),
            user_text=user_text,
            chat=chat,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            special_tokens=special_tokens,
            blocks=freeze({
                "question": item.question,
                "options": f"Options:\nA. {option_a}\nB. {option_b}",
                "context": context,
                "instruction": instruction,
                "placement": resolved,
            }),
        ))
    return tuple(prompts)


def _context(instance: InterventionInstance) -> str | None:
    if not instance.context_blocks:
        return None
    return "\n\n".join(block["text"] for block in instance.context_blocks)


def _options(item: BaseItem, correct_label: str) -> tuple[str, str]:
    if correct_label == "A":
        return item.correct_answer, item.incorrect_answer
    if correct_label == "B":
        return item.incorrect_answer, item.correct_answer
    raise ValueError("correct label must be A or B")


def _chat(user_text: str, *, chat: bool, system_message: str | None):
    if chat:
        if type(system_message) is not str or system_message == "":
            raise ValueError("chat mode requires the literal system message")
        messages = (
            freeze({"role": "system", "content": system_message}),
            freeze({"role": "user", "content": user_text}),
        )
        return messages, True, "native_template"
    if system_message is not None:
        raise ValueError("no-chat mode has no system text or role delimiters")
    return (freeze({"role": "user", "content": user_text}),), False, "no_bos_eos"
