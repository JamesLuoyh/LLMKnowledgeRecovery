"""Canonical-token audit under the continuation contract.

The scoring token is the continuation of the already encoded prompt.
Concatenated prompt-plus-answer text is checked only to prove that the
prompt tokens are still an exact prefix; its suffix is not the score.
A leading-space encoding is recorded and is never substituted for a bare
A or B that is not itself one token. No force or skip argument bypasses
these checks.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from knowledge_recovery.errors import (
    TOKEN_AUDIT_FIELDS,
    CanonicalTokenNotSingle,
    KnowledgeRecoveryError,
    TokenAuditRequired,
    token_recheck_command,
)
from knowledge_recovery.provenance import canonical_json, content_digest
from knowledge_recovery.records import freeze

ANSWER_BOUNDARY = "continuation_of_prompt_token_ids"
PREFIX_CHECK = "exact_prompt_prefix"


class PromptPrefixChanged(KnowledgeRecoveryError):
    """Concatenated tokenization did not keep the prompt tokens as a prefix."""

    def __init__(self, *, answer: str, prompt_length: int) -> None:
        self.answer = answer
        self.prompt_length = prompt_length
        super().__init__(
            "PromptPrefixChanged: concatenated prompt-plus-answer tokenization "
            f"changed the prompt prefix before {answer!r}.\n"
            f"Prompt length: {prompt_length}\n"
            "The concatenated suffix was not used as the continuation token."
        )


class UnstableCanonicalToken(KnowledgeRecoveryError):
    """One canonical string continued as different token ids on two prompts."""

    def __init__(self, *, answer: str, token_ids: Sequence[int]) -> None:
        self.answer = answer
        self.token_ids = tuple(token_ids)
        shown = ", ".join(str(item) for item in self.token_ids)
        super().__init__(
            f"UnstableCanonicalToken: {answer!r} is not one stable continuation token.\n"
            f"Token ids: {shown}\n"
            "Representative prompt boundaries disagreed. Neither id was chosen."
        )


@dataclass(frozen=True)
class PromptExample:
    """One rendered user text used as a prompt-boundary example."""

    user_text: str
    chat: bool
    system_message: str | None

    def __post_init__(self) -> None:
        if type(self.user_text) is not str or self.user_text == "":
            raise ValueError("prompt example text is required")
        if type(self.chat) is not bool:
            raise ValueError("chat must be a boolean")
        if self.chat:
            if type(self.system_message) is not str or self.system_message == "":
                raise ValueError("chat mode requires the system message")
        elif self.system_message is not None:
            raise ValueError("no-chat mode has no system message")


class ContinuationTokenizer(Protocol):
    """Tokenize a fixed prompt and continue an answer without retokenizing it."""

    def settings(self) -> Mapping[str, Any]:
        """Tokenizer settings that affect token identity. Paths are not included."""

    def prompt_boundary(self, chat: bool) -> str:
        """Name of the prompt-boundary convention this tokenizer implements."""

    def prompt_string(self, example: PromptExample) -> str:
        """Exact string whose tokens are the conditioning prefix."""

    def encode_prompt(self, example: PromptExample) -> tuple[int, ...]:
        """Token ids of ``prompt_string``. Chat wrapping belongs to the tokenizer."""

    def continuation_ids(self, prompt_ids: Sequence[int], answer: str) -> tuple[int, ...]:
        """Ids of ``answer`` continuing ``prompt_ids``.

        This must not be the suffix of a concatenation that changed ``prompt_ids``.
        """

    def encode_concatenated(self, prompt_text: str, answer: str) -> tuple[int, ...]:
        """Ids of the concatenated text, used only for the prefix check."""

    def decode_token(self, token_id: int) -> bytes:
        """Decoded bytes of one token id."""


@dataclass(frozen=True)
class TokenAudit:
    """Inspectable audit. ``audit_id`` hashes the contract, not example text."""

    audit_id: str
    record: Mapping[str, Any]
    representative_examples: tuple[str, ...]
    leading_space: Mapping[str, Any]

    @property
    def token_ids(self) -> Mapping[str, int]:
        boundary = self.record["answer_boundary"]
        return boundary["token_ids"]


def audit_canonical_tokens(
    tokenizer: ContinuationTokenizer,
    examples: Sequence[PromptExample],
    *,
    canonical_strings: Sequence[str] = ("A", "B"),
    model_content_digest: str,
    tokenizer_content_digest: str,
    chat: bool,
    system_message: str | None,
    instruction: str,
    answer_mode: str,
    renderer_version: str,
    prompt_boundary: str,
    established: Mapping[str, Any] | None = None,
    configured_audit_id: str | None = None,
    recheck: bool = False,
    config_path: str,
) -> TokenAudit:
    """Validate one-token continuations and compare them with an established audit.

    ``recheck`` records a new valid contract. It does not skip the one-token
    or prefix checks, and there is no force or skip parameter.
    A null configured id does not accept a changed established contract.
    """
    strings = _strings(canonical_strings)
    _chat_fields(chat, system_message)
    if prompt_boundary != tokenizer.prompt_boundary(chat):
        raise ValueError(
            f"prompt boundary {prompt_boundary!r} does not match the tokenizer boundary "
            f"{tokenizer.prompt_boundary(chat)!r}"
        )
    chosen, examples_text = _continuations(tokenizer, examples, strings, chat, system_message)
    if len(set(chosen.values())) != len(chosen):
        raise ValueError(
            "canonical strings must be distinct continuation tokens. "
            "No string was replaced with another string's token."
        )
    record = {
        "kind": "token_audit",
        "schema": 1,
        "model_content_digest": _digest(model_content_digest, "model_content_digest"),
        "tokenizer_content_digest": _digest(tokenizer_content_digest, "tokenizer_content_digest"),
        "tokenizer_settings": dict(tokenizer.settings()),
        "chat": chat,
        "template": {
            "system_message": system_message,
            "renderer_version": renderer_version,
        },
        "instruction": instruction,
        "prompt_boundary": prompt_boundary,
        "answer_boundary": {
            "convention": ANSWER_BOUNDARY,
            "answer_mode": answer_mode,
            "prefix_check": PREFIX_CHECK,
            "token_ids": {text: chosen[text] for text in strings},
            "decoded_bytes_hex": {
                text: tokenizer.decode_token(chosen[text]).hex() for text in strings
            },
        },
        "canonical_strings": list(strings),
    }
    plain = json.loads(canonical_json(record))
    audit = TokenAudit(
        audit_id=content_digest(plain),
        record=freeze(plain),
        representative_examples=examples_text,
        leading_space=freeze(_leading_space(tokenizer, examples, strings)),
    )
    _enforce_contract(
        audit,
        established=established,
        configured_audit_id=configured_audit_id,
        recheck=recheck,
        config_path=config_path,
    )
    return audit


def extraction_manifest_audit(
    configured_audit_id: str | None,
    audit: TokenAudit,
) -> dict[str, Any]:
    """Fill a null configuration field with the validated audit id."""
    if audit.audit_id is None or len(audit.audit_id) != 64:
        raise ValueError("a resolved extraction manifest requires the audit id")
    return {
        "token_audit": audit.audit_id,
        "configured_token_audit": configured_audit_id,
    }


def write_token_audit(directory: Path | str, audit: TokenAudit) -> Path:
    """Write the hashed contract beside the example text it was checked on."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    record = json.loads(canonical_json(audit.record))
    if content_digest(record) != audit.audit_id:
        raise ValueError("token audit digest does not match its record")
    payload = {
        "canonical_json_version": 1,
        "digest": audit.audit_id,
        "record": record,
        "representative_examples": list(audit.representative_examples),
        "leading_space": json.loads(canonical_json(audit.leading_space)),
    }
    path = target / f"{audit.audit_id}.json"
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    current = target / "current.json"
    _atomic_text(current, json.dumps({"digest": audit.audit_id, "path": path.name}, sort_keys=True) + "\n")
    return path


def read_token_audit(path: Path | str) -> dict[str, Any]:
    """Load an audit file and reject a digest that does not match the record."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if content_digest(payload["record"]) != payload["digest"]:
        raise ValueError("token audit digest does not match its record")
    return payload


class MappedContinuationTokenizer:
    """Byte-level prompt encoding with an explicit one-token continuation map.

    Prompt bytes and continuation ids are different maps. The continuation of
    ``A`` is whatever the map says, not the trailing byte of ``prompt + "A"``.
    This tokenizer does not read Qwen weights.
    """

    def __init__(
        self,
        continuations: Mapping[str, Sequence[int]],
        *,
        decoded: Mapping[int, bytes] | None = None,
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        self.continuations = {text: tuple(ids) for text, ids in continuations.items()}
        self.decoded = {} if decoded is None else dict(decoded)
        self._settings = {"add_special_tokens": False, "kind": "mapped_continuation"}
        if settings:
            self._settings.update(dict(settings))

    @classmethod
    def read(cls, path: Path | str) -> MappedContinuationTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("kind") != "mapped_continuation_tokenizer" or payload.get("schema") != 1:
            raise ValueError("tokenizer_map.json is not a mapped continuation tokenizer")
        continuations = {text: tuple(ids) for text, ids in payload["continuations"].items()}
        decoded = {
            int(token_id): bytes.fromhex(hex_text)
            for token_id, hex_text in payload.get("decoded", {}).items()
        }
        return cls(continuations, decoded=decoded, settings=payload.get("settings") or {})

    def settings(self) -> Mapping[str, Any]:
        return dict(self._settings)

    def prompt_boundary(self, chat: bool) -> str:
        return "mapped_chat_template_v1" if chat else "end_of_user_text"

    def prompt_string(self, example: PromptExample) -> str:
        if example.chat:
            return (
                "<|im_start|>system\n"
                f"{example.system_message}<|im_end|>\n"
                "<|im_start|>user\n"
                f"{example.user_text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        return example.user_text

    def encode_prompt(self, example: PromptExample) -> tuple[int, ...]:
        return _byte_ids(self.prompt_string(example))

    def continuation_ids(self, prompt_ids: Sequence[int], answer: str) -> tuple[int, ...]:
        if not tuple(prompt_ids):
            raise ValueError("a continuation requires the prompt token sequence")
        if answer in self.continuations:
            return self.continuations[answer]
        return _byte_ids(answer)

    def encode_concatenated(self, prompt_text: str, answer: str) -> tuple[int, ...]:
        return _byte_ids(prompt_text + answer)

    def decode_token(self, token_id: int) -> bytes:
        if token_id in self.decoded:
            return self.decoded[token_id]
        for text, ids in self.continuations.items():
            if ids == (token_id,):
                return text.encode("utf-8")
        if token_id >= 256:
            return bytes((token_id - 256,))
        raise ValueError(f"token {token_id} has no decoded bytes")


def _continuations(
    tokenizer: ContinuationTokenizer,
    examples: Sequence[PromptExample],
    strings: tuple[str, ...],
    chat: bool,
    system_message: str | None,
) -> tuple[dict[str, int], tuple[str, ...]]:
    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence) or not examples:
        raise ValueError("at least one representative prompt is required")
    chosen: dict[str, int] | None = None
    texts: list[str] = []
    for example in examples:
        if type(example) is not PromptExample:
            raise TypeError("prompt examples must be PromptExample values")
        if example.chat != chat or example.system_message != system_message:
            raise ValueError("every representative prompt uses the audited chat contract")
        prompt_text = tokenizer.prompt_string(example)
        prompt_ids = tuple(tokenizer.encode_prompt(example))
        if not prompt_ids:
            raise ValueError("the prompt token sequence is empty")
        texts.append(prompt_text)
        found: dict[str, int] = {}
        for answer in strings:
            _require_exact_prefix(tokenizer, prompt_text, prompt_ids, answer)
            found[answer] = _one_token(tokenizer.continuation_ids(prompt_ids, answer), answer)
        if chosen is None:
            chosen = found
            continue
        for answer in strings:
            if found[answer] != chosen[answer]:
                raise UnstableCanonicalToken(answer=answer, token_ids=(chosen[answer], found[answer]))
    if chosen is None:
        raise ValueError("at least one representative prompt is required")
    return chosen, tuple(texts)


def _one_token(token_ids: Sequence[int], answer: str) -> int:
    tokens = tuple(token_ids)
    if len(tokens) != 1:
        raise CanonicalTokenNotSingle(canonical_answer=answer, token_count=len(tokens))
    token_id = tokens[0]
    if type(token_id) is not int or token_id < 0:
        raise ValueError("a continuation token id must be a non-negative integer")
    return token_id


def _require_exact_prefix(
    tokenizer: ContinuationTokenizer,
    prompt_text: str,
    prompt_ids: Sequence[int],
    answer: str,
) -> None:
    concatenated = tuple(tokenizer.encode_concatenated(prompt_text, answer))
    prefix = tuple(prompt_ids)
    if concatenated[: len(prefix)] != prefix:
        raise PromptPrefixChanged(answer=answer, prompt_length=len(prefix))


def _leading_space(
    tokenizer: ContinuationTokenizer,
    examples: Sequence[PromptExample],
    strings: tuple[str, ...],
) -> dict[str, Any]:
    """Record a leading-space encoding without adopting it as the contract."""
    example = examples[0]
    prompt_ids = tuple(tokenizer.encode_prompt(example))
    recorded: dict[str, Any] = {}
    for answer in strings:
        if answer.startswith(" "):
            continue
        spaced = " " + answer
        token_ids = tuple(tokenizer.continuation_ids(prompt_ids, spaced))
        recorded[spaced] = {"token_count": len(token_ids), "token_ids": list(token_ids)}
    return recorded


def _enforce_contract(
    audit: TokenAudit,
    *,
    established: Mapping[str, Any] | None,
    configured_audit_id: str | None,
    recheck: bool,
    config_path: str,
) -> None:
    previous = _established_record(established)
    current = json.loads(canonical_json(audit.record))
    records_match = previous is not None and json.loads(canonical_json(previous)) == current
    configured_names_this_audit = configured_audit_id in (None, audit.audit_id)
    if previous is None and configured_names_this_audit:
        return
    if records_match and configured_names_this_audit:
        return
    if recheck:
        return
    if previous is None:
        changed: tuple[str, ...] = TOKEN_AUDIT_FIELDS
        old_signature = configured_audit_id or "no-established-audit"
    else:
        changed = _changed_fields(previous, audit.record) or TOKEN_AUDIT_FIELDS
        old_signature = content_digest(json.loads(canonical_json(previous)))
        if old_signature == audit.audit_id and configured_audit_id not in (None, audit.audit_id):
            old_signature = configured_audit_id
    raise TokenAuditRequired(
        changed_fields=changed,
        old_signature=old_signature,
        new_signature=audit.audit_id,
        recheck_command=token_recheck_command(config_path),
    )


def _established_record(established: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if established is None:
        return None
    if "record" in established and "digest" in established:
        return established["record"]
    return established


def _changed_fields(old: Mapping[str, Any], new: Mapping[str, Any]) -> tuple[str, ...]:
    previous = _components(old)
    current = _components(new)
    return tuple(name for name in TOKEN_AUDIT_FIELDS if previous[name] != current[name])


def _components(record: Mapping[str, Any]) -> dict[str, Any]:
    plain = json.loads(canonical_json(record))
    boundary = plain["answer_boundary"]
    return {
        "model": plain["model_content_digest"],
        "tokenizer": {
            "content": plain["tokenizer_content_digest"],
            "settings": plain["tokenizer_settings"],
        },
        "chat_mode": plain["chat"],
        "template": plain["template"],
        "prompt_boundary": plain["prompt_boundary"],
        "instruction": plain["instruction"],
        "answer_boundary": boundary,
        "canonical_strings": plain["canonical_strings"],
    }


def _strings(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError("at least one canonical string is required")
    parsed = []
    for value in values:
        if type(value) is not str or value == "" or "\n" in value:
            raise ValueError("canonical strings must be non-empty single lines")
        parsed.append(value)
    if len(set(parsed)) != len(parsed):
        raise ValueError("canonical strings must be unique")
    return tuple(parsed)


def _chat_fields(chat: bool, system_message: str | None) -> None:
    if type(chat) is not bool:
        raise ValueError("chat must be a boolean")
    if chat and (type(system_message) is not str or system_message == ""):
        raise ValueError("chat mode requires the system message")
    if not chat and system_message is not None:
        raise ValueError("no-chat mode has no system message")


def _digest(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a full SHA-256 hex digest")
    return value


def _byte_ids(text: str) -> tuple[int, ...]:
    encoded = text.encode("utf-8")
    if not encoded:
        raise ValueError("text to encode is empty")
    return tuple(256 + byte for byte in encoded)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
