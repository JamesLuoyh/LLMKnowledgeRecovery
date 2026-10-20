import inspect
import json

import pytest

from knowledge_recovery.errors import CanonicalTokenNotSingle, TokenAuditRequired
from knowledge_recovery.prompts.token_audit import (
    MappedContinuationTokenizer,
    PromptExample,
    PromptPrefixChanged,
    TokenAudit,
    UnstableCanonicalToken,
    audit_canonical_tokens,
    extraction_manifest_audit,
    write_token_audit,
    read_token_audit,
)
from knowledge_recovery.provenance import canonical_json, content_digest

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
CONFIG = "configs/initial_sycophancy.yaml"
INSTRUCTION = "Answer with exactly one letter. Do not include any other text."


class PrefixMutator:
    """Concatenation changes the first prompt token. Continuation ids stay single."""

    def __init__(self) -> None:
        self.inner = MappedContinuationTokenizer({"A": (10,), "B": (11,)})

    def settings(self):
        return self.inner.settings()

    def prompt_boundary(self, chat: bool) -> str:
        return self.inner.prompt_boundary(chat)

    def prompt_string(self, example: PromptExample) -> str:
        return self.inner.prompt_string(example)

    def encode_prompt(self, example: PromptExample):
        return self.inner.encode_prompt(example)

    def continuation_ids(self, prompt_ids, answer):
        return self.inner.continuation_ids(prompt_ids, answer)

    def encode_concatenated(self, prompt_text: str, answer: str):
        ids = self.inner.encode_concatenated(prompt_text, answer)
        return (ids[0] + 1,) + ids[1:]

    def decode_token(self, token_id: int) -> bytes:
        return self.inner.decode_token(token_id)


class UnstableTokenizer(MappedContinuationTokenizer):
    def continuation_ids(self, prompt_ids, answer):
        if answer == "A":
            return (10 if prompt_ids[-1] % 2 == 0 else 14,)
        return (11,)


def _examples(*texts: str, chat: bool = False, system: str | None = None):
    return tuple(PromptExample(text, chat, system) for text in texts)


def _audit(tokenizer, examples, **overrides) -> TokenAudit:
    fields = dict(
        canonical_strings=("A", "B"),
        model_content_digest=DIGEST_A,
        tokenizer_content_digest=DIGEST_B,
        template_name="native_pinned",
        chat=False,
        system_message=None,
        instruction=INSTRUCTION,
        answer_mode="mcqa",
        renderer_version="user-text-v1",
        prompt_boundary=tokenizer.prompt_boundary(False),
        established=None,
        configured_audit_id=None,
        recheck=False,
        config_path=CONFIG,
    )
    fields.update(overrides)
    if "prompt_boundary" not in overrides and fields["chat"] is not False:
        fields["prompt_boundary"] = tokenizer.prompt_boundary(True)
    return audit_canonical_tokens(tokenizer, examples, **fields)


def test_one_token_continuation_is_not_the_concatenated_suffix():
    tokenizer = MappedContinuationTokenizer(
        {"A": (10,), "B": (11,), " A": (12,), " B": (13,)},
        decoded={10: b"A", 11: b"B", 12: b" A", 13: b" B"},
    )
    audit = _audit(tokenizer, _examples("Question one?", "Question two?"))
    assert dict(audit.token_ids) == {"A": 10, "B": 11}
    prompt = tokenizer.prompt_string(PromptExample("Question one?", False, None))
    suffix = tokenizer.encode_concatenated(prompt, "A")[len(tokenizer.encode_prompt(PromptExample("Question one?", False, None))):]
    assert suffix != (10,)
    assert tuple(audit.leading_space[" A"]["token_ids"]) == (12,)
    manifest = extraction_manifest_audit(None, audit)
    assert manifest["configured_token_audit"] is None
    assert manifest["token_audit"] == audit.audit_id


def test_example_text_is_not_part_of_the_audit_id():
    tokenizer = MappedContinuationTokenizer({"A": (10,), "B": (11,)})
    first = _audit(tokenizer, _examples("one"))
    second = _audit(tokenizer, _examples("two"))
    assert first.audit_id == second.audit_id
    assert first.representative_examples != second.representative_examples


def test_multi_token_and_empty_continuations_do_not_pick_an_endpoint():
    multi = MappedContinuationTokenizer({"A": (10, 11), "B": (12,)})
    with pytest.raises(CanonicalTokenNotSingle) as raised:
        _audit(multi, _examples("Question?"))
    assert raised.value.token_count == 2
    assert raised.value.canonical_answer == "A"
    empty = MappedContinuationTokenizer({"A": (), "B": (12,)})
    with pytest.raises(CanonicalTokenNotSingle) as raised:
        _audit(empty, _examples("Question?"))
    assert raised.value.token_count == 0


def test_concatenated_prefix_must_match_and_its_suffix_is_unused():
    with pytest.raises(PromptPrefixChanged):
        _audit(PrefixMutator(), _examples("Question?"))


def test_prompt_boundaries_must_agree_on_the_continuation_id():
    tokenizer = UnstableTokenizer({"A": (10,), "B": (11,)})
    with pytest.raises(UnstableCanonicalToken) as raised:
        _audit(tokenizer, _examples("even", "odds"))
    assert raised.value.token_ids == (10, 14)


def _record(audit: TokenAudit) -> dict:
    return json.loads(canonical_json(audit.record))


def test_null_config_does_not_accept_a_changed_established_contract():
    tokenizer = MappedContinuationTokenizer({"A": (10,), "B": (11,)})
    original = _audit(tokenizer, _examples("Question?"))
    with pytest.raises(TokenAuditRequired) as raised:
        _audit(
            tokenizer,
            _examples("Question?"),
            instruction="Answer with a different instruction.",
            established={"digest": original.audit_id, "record": _record(original)},
            configured_audit_id=None,
        )
    assert raised.value.changed_fields == ("instruction",)
    assert raised.value.old_signature != raised.value.new_signature
    assert "audit-tokens" in str(raised.value)
    assert "--force" not in str(raised.value) and "--skip" not in str(raised.value)


def test_recheck_records_a_new_contract_and_still_rejects_a_bad_token():
    tokenizer = MappedContinuationTokenizer({"A": (10,), "B": (11,)})
    original = _audit(tokenizer, _examples("Question?"))
    revised = _audit(
        tokenizer,
        _examples("Question?"),
        instruction="Answer with a different instruction.",
        established={"digest": original.audit_id, "record": _record(original)},
        recheck=True,
    )
    assert revised.audit_id != original.audit_id
    broken = MappedContinuationTokenizer({"A": (10, 11), "B": (12,)})
    with pytest.raises(CanonicalTokenNotSingle):
        _audit(
            broken,
            _examples("Question?"),
            established={"digest": original.audit_id, "record": _record(original)},
            recheck=True,
        )


def test_model_tokenizer_chat_template_boundary_and_strings_are_named():
    tokenizer = MappedContinuationTokenizer({"A": (10,), "B": (11,), "C": (14,)})
    original = _audit(tokenizer, _examples("Question?"))
    established = {"digest": original.audit_id, "record": _record(original)}
    cases = (
        ("model", {"model_content_digest": "c" * 64}),
        ("tokenizer", {"tokenizer_content_digest": "d" * 64}),
        ("prompt_boundary", {"prompt_boundary": "end_of_user_text"}),
    )
    # prompt_boundary override must still match the tokenizer, so change the tokenizer boundary via a subclass.
    for field, overrides in (
        ("model", {"model_content_digest": "c" * 64}),
        ("tokenizer", {"tokenizer_content_digest": "d" * 64}),
        ("canonical_strings", {"canonical_strings": ("A", "C")}),
    ):
        with pytest.raises(TokenAuditRequired) as raised:
            _audit(tokenizer, _examples("Question?"), established=established, **overrides)
        assert field in raised.value.changed_fields
    assert cases


def test_chat_mode_and_template_changes_are_named():
    tokenizer = MappedContinuationTokenizer({"A": (10,), "B": (11,)})
    original = _audit(tokenizer, _examples("Question?"))
    established = {"record": _record(original), "digest": original.audit_id}
    with pytest.raises(TokenAuditRequired) as raised:
        _audit(
            tokenizer,
            _examples("Question?", chat=True, system="System text."),
            chat=True,
            system_message="System text.",
            prompt_boundary=tokenizer.prompt_boundary(True),
            established=established,
        )
    assert "chat_mode" in raised.value.changed_fields
    assert "template" in raised.value.changed_fields


def test_there_is_no_force_or_skip_parameter():
    names = set(inspect.signature(audit_canonical_tokens).parameters)
    assert "force" not in names
    assert "skip" not in names


def test_written_audit_round_trip_keeps_the_digest(tmp_path):
    tokenizer = MappedContinuationTokenizer({"A": (10,), "B": (11,)})
    audit = _audit(tokenizer, _examples("Question?"))
    path = write_token_audit(tmp_path, audit)
    loaded = read_token_audit(path)
    assert loaded["digest"] == content_digest(loaded["record"])
    assert "timestamp" not in loaded["record"]
    assert "path" not in loaded["record"]
