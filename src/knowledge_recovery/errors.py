"""Actionable domain exceptions.

Each message names the field, path, or command a later stage must show.
"""

from __future__ import annotations

from collections.abc import Sequence

TOKEN_AUDIT_FIELDS = (
    "model",
    "tokenizer",
    "chat_mode",
    "template",
    "prompt_boundary",
    "instruction",
    "answer_boundary",
    "canonical_strings",
)
PROBABILITY_BASES = ("raw", "canonical")
ASSET_KINDS = ("model", "tokenizer", "dataset")
CACHE_MISMATCHES = ("schema", "identity", "tensor_shape", "checksum")
ANSWER_REQUEST_KINDS = ("direct_answer", "incompatible_label_set")
SUPPORTED_POSITION_SELECTOR = "last_nonpadding_input_token"
_NO_INFERENCE = (
    "Never silently fall back to model inference during a probe-only command."
)


class KnowledgeRecoveryError(Exception):
    """Domain failure whose message is safe to show and whose exit is nonzero."""

    exit_code = 1


class TokenAuditRequired(KnowledgeRecoveryError):
    """A validated answer-token contract no longer matches the requested one."""

    def __init__(
        self,
        *,
        changed_fields: Sequence[str],
        old_signature: str,
        new_signature: str,
        recheck_command: str,
    ) -> None:
        self.changed_fields = _members(changed_fields, TOKEN_AUDIT_FIELDS, "changed field")
        self.old_signature = _line(old_signature, "old_signature")
        self.new_signature = _line(new_signature, "new_signature")
        if self.old_signature == self.new_signature:
            raise ValueError("old and new token-audit signatures must differ")
        self.recheck_command = _command(recheck_command, "kr audit-tokens")
        fields = ", ".join(self.changed_fields)
        super().__init__(
            "TokenAuditRequired: model, tokenizer, chat mode, template, "
            "prompt boundary, instruction, answer boundary, or canonical "
            "strings differ from the validated signature.\n"
            f"Changed fields: {fields}\n"
            f"Old signature: {self.old_signature}\n"
            f"New signature: {self.new_signature}\n"
            "Bare versus leading-space answer tokens must be rechecked.\n"
            "Recheck whether the intended answer is represented by bare "
            "A/B, a leading-space alternative, or another tokenization.\n"
            f"Recheck command: {self.recheck_command}"
        )


class CanonicalTokenNotSingle(KnowledgeRecoveryError):
    """A canonical answer is not one continuation token."""

    def __init__(self, *, canonical_answer: str, token_count: int) -> None:
        self.canonical_answer = _line(canonical_answer, "canonical_answer")
        self.token_count = _nonnegative(token_count, "token_count")
        if self.token_count == 1:
            raise ValueError("one verified token is not CanonicalTokenNotSingle")
        super().__init__(
            "CanonicalTokenNotSingle: canonical answer "
            f"{self.canonical_answer!r} does not correspond to one verified "
            "token under the continuation contract.\n"
            f"Observed token count: {self.token_count}\n"
            "Do not approximate by taking its first or last token."
        )


class AnswerInstructionMismatch(KnowledgeRecoveryError):
    """The one-letter instruction was paired with an unsupported answer request."""

    def __init__(
        self,
        *,
        request_kind: str,
        instruction: str,
        labels: Sequence[str] | None = None,
    ) -> None:
        self.request_kind = _one(request_kind, ANSWER_REQUEST_KINDS, "request_kind")
        self.instruction = _line(instruction, "instruction")
        if self.request_kind == "direct_answer":
            if labels is not None:
                raise ValueError("direct_answer does not take a label set")
            self.labels: tuple[str, ...] = ()
            requested = "a direct-answer task was requested with the one-letter instruction"
        else:
            self.labels = _lines(labels, "label")
            requested = "an incompatible label set was requested with the one-letter instruction"
        lines = [
            f"AnswerInstructionMismatch: {requested}.",
            f"Instruction: {self.instruction}",
        ]
        if self.labels:
            lines.append("Labels: " + ", ".join(self.labels))
        lines.append("Require an appropriate instruction and a new answer/token audit.")
        super().__init__("\n".join(lines))


class UnsupportedPositionSelector(KnowledgeRecoveryError):
    """Teacher-forced extraction was requested before that feature exists."""

    def __init__(self, *, selector: str) -> None:
        self.selector = _line(selector, "selector")
        if self.selector == SUPPORTED_POSITION_SELECTOR:
            raise ValueError(f"{SUPPORTED_POSITION_SELECTOR} is the supported selector")
        super().__init__(
            "UnsupportedPositionSelector: teacher-forced extraction was "
            "requested before that feature exists.\n"
            f"Requested selector: {self.selector}\n"
            "Missing component: teacher-forced position selector."
        )


class FilterPolicyDecisionRequired(KnowledgeRecoveryError):
    """Reports are complete, but the explicit raw/canonical choice is not usable."""

    def __init__(
        self,
        *,
        report_id: str,
        requested_basis: str | None = None,
        recorded_basis: str | None = None,
    ) -> None:
        self.report_id = _token(report_id, "report_id")
        if requested_basis is None and recorded_basis is None:
            self.decision_state = "unset"
            self.requested_basis = None
            self.recorded_basis = None
            message = (
                "FilterPolicyDecisionRequired:\n"
                "Both filtering reports have been saved.\n"
                "Choose probability_basis=raw or probability_basis=canonical\n"
                "and record the decision against this report before probe training.\n"
                f"Report: {self.report_id}\n"
                "Record command: kr record-filter-decision "
                f"--report {self.report_id} --basis <chosen-basis>"
            )
        elif requested_basis is None or recorded_basis is None:
            raise ValueError("a contradiction requires both the requested and recorded basis")
        else:
            self.requested_basis = _one(requested_basis, PROBABILITY_BASES, "basis")
            self.recorded_basis = _one(recorded_basis, PROBABILITY_BASES, "basis")
            if self.requested_basis == self.recorded_basis:
                raise ValueError("requested basis matches the recorded decision")
            self.decision_state = "contradiction"
            message = (
                "FilterPolicyDecisionRequired:\n"
                "Both filtering reports are complete, but the requested "
                f"probability basis {self.requested_basis!r} contradicts the "
                f"recorded decision {self.recorded_basis!r}.\n"
                f"Report: {self.report_id}\n"
                "Requesting a contradictory basis reopens the gate after "
                "updated diagnostics."
            )
        super().__init__(message)


class MissingLocalAsset(KnowledgeRecoveryError):
    """A pinned model, tokenizer, or dataset artifact is absent offline."""

    def __init__(self, *, kind: str, expected_location: str) -> None:
        self.kind = _one(kind, ASSET_KINDS, "asset kind")
        self.expected_location = _line(expected_location, "expected_location")
        super().__init__(
            "MissingLocalAsset: a required pinned "
            f"{self.kind} artifact is unavailable offline.\n"
            f"Expected location: {self.expected_location}\n"
            "The artifact was not downloaded."
        )


class CacheIncompatible(KnowledgeRecoveryError):
    """Saved artifacts do not match the requested schema, identity, or bytes."""

    def __init__(self, *, mismatches: Sequence[str], detail: str) -> None:
        self.mismatches = _members(mismatches, CACHE_MISMATCHES, "cache mismatch")
        self.detail = _line(detail, "detail")
        super().__init__(
            "CacheIncompatible: artifact schemas, identities, tensor shapes, "
            "or checksums do not match.\n"
            f"Mismatches: {', '.join(self.mismatches)}\n"
            f"{self.detail}\n{_NO_INFERENCE}"
        )


class CacheIncomplete(KnowledgeRecoveryError):
    """Required extraction rows or layers are absent from a completed lookup."""

    def __init__(
        self,
        *,
        extraction_command: str,
        missing_keys: Sequence[str] = (),
        missing_layers: Sequence[int] = (),
    ) -> None:
        self.extraction_command = _command(extraction_command, "kr extract")
        self.missing_keys = _lines(missing_keys, "missing key", allow_empty=True)
        self.missing_layers = _layers(missing_layers)
        if not self.missing_keys and not self.missing_layers:
            raise ValueError("CacheIncomplete requires missing keys or layers")
        lines = ["CacheIncomplete: required rows or layers are missing."]
        if self.missing_keys:
            lines.append("Missing keys: " + ", ".join(self.missing_keys))
        if self.missing_layers:
            lines.append("Missing layers: " + ", ".join(map(str, self.missing_layers)))
        lines.extend((
            f"Required extraction command: {self.extraction_command}",
            _NO_INFERENCE,
        ))
        super().__init__("\n".join(lines))


def token_recheck_command(config_path: str) -> str:
    """Command shown when an established token contract changes."""
    return _stage_command("audit-tokens", config_path)


def extraction_command(config_path: str) -> str:
    """Command shown when a probe-only lookup finds missing rows."""
    return _stage_command("extract", config_path)


def _stage_command(stage: str, config_path: str) -> str:
    path = _token(config_path, "config_path")
    return _command(f"kr {stage} --config {path}", f"kr {stage}")


def _line(value: object, label: str) -> str:
    if not isinstance(value, str) or value.strip() == "" or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be a non-empty single line")
    return value


def _token(value: object, label: str) -> str:
    text = _line(value, label)
    if any(char.isspace() for char in text) or text.startswith("-"):
        raise ValueError(f"{label} must be a single token")
    return text


def _one(value: object, allowed: Sequence[str], label: str) -> str:
    if value not in allowed:
        raise ValueError(f"{label} must be one of: {', '.join(allowed)}")
    return str(value)


def _members(values: Sequence[str], allowed: Sequence[str], label: str) -> tuple[str, ...]:
    return tuple(_one(item, allowed, label) for item in _nonempty_sequence(values, label))


def _lines(values: Sequence[str] | None, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if values is None:
        raise ValueError(f"at least one {label} is required")
    items = _nonempty_sequence(values, label, allow_empty=allow_empty)
    return tuple(_line(item, label) for item in items)


def _nonempty_sequence(values: Sequence[object], label: str, *, allow_empty: bool = False) -> Sequence[object]:
    if isinstance(values, str) or (not values and not allow_empty):
        raise ValueError(f"at least one {label} is required")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label}")
    return values


def _nonnegative(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _layers(values: Sequence[int]) -> tuple[int, ...]:
    chosen = tuple(_nonnegative(layer, "layer") for layer in _nonempty_sequence(values, "layer", allow_empty=True))
    if any(layer < 1 for layer in chosen):
        raise ValueError("layer indices start at 1")
    return chosen


def _command(value: object, prefix: str) -> str:
    command = _line(value, "command")
    if not command.startswith(prefix + " ") and command != prefix:
        raise ValueError(f"command must start with {prefix!r}")
    if "--force" in command.split() or "--skip" in command.split():
        raise ValueError("a force or skip flag cannot bypass this check")
    return command
