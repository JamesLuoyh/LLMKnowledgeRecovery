import pytest

from knowledge_recovery.errors import (
    ANSWER_REQUEST_KINDS,
    ASSET_KINDS,
    CACHE_MISMATCHES,
    PROBABILITY_BASES,
    TOKEN_AUDIT_FIELDS,
    AnswerInstructionMismatch,
    CacheIncomplete,
    CacheIncompatible,
    CanonicalTokenNotSingle,
    FilterPolicyDecisionRequired,
    KnowledgeRecoveryError,
    MissingLocalAsset,
    TokenAuditRequired,
    UnsupportedPositionSelector,
    extraction_command,
    token_recheck_command,
)

NAMED = (
    TokenAuditRequired,
    CanonicalTokenNotSingle,
    AnswerInstructionMismatch,
    UnsupportedPositionSelector,
    FilterPolicyDecisionRequired,
    MissingLocalAsset,
    CacheIncompatible,
    CacheIncomplete,
)
RECHECK = token_recheck_command("configs/initial_sycophancy.yaml")
EXTRACT = extraction_command("configs/initial_sycophancy.yaml")


def test_named_exceptions_exit_nonzero():
    assert [cls.__name__ for cls in NAMED] == [
        "TokenAuditRequired",
        "CanonicalTokenNotSingle",
        "AnswerInstructionMismatch",
        "UnsupportedPositionSelector",
        "FilterPolicyDecisionRequired",
        "MissingLocalAsset",
        "CacheIncompatible",
        "CacheIncomplete",
    ]
    assert all(issubclass(cls, KnowledgeRecoveryError) and cls.exit_code != 0 for cls in NAMED)


@pytest.mark.parametrize("field", TOKEN_AUDIT_FIELDS)
def test_token_audit_names_the_changed_field(field):
    err = TokenAuditRequired(
        changed_fields=[field],
        old_signature="sig-old",
        new_signature="sig-new",
        recheck_command=RECHECK,
    )
    message = str(err)
    assert err.changed_fields == (field,)
    assert f"Changed fields: {field}\nOld signature: sig-old\nNew signature: sig-new" in message
    assert "Bare versus leading-space answer tokens must be rechecked." in message
    assert "bare A/B, a leading-space alternative, or another tokenization." in message
    assert message.endswith(f"Recheck command: {RECHECK}")


def test_token_audit_rejects_an_unchanged_or_bypassed_contract():
    with pytest.raises(ValueError, match="must differ"):
        TokenAuditRequired(
            changed_fields=["model"],
            old_signature="same",
            new_signature="same",
            recheck_command=RECHECK,
        )
    with pytest.raises(ValueError, match="force or skip"):
        TokenAuditRequired(
            changed_fields=["chat_mode"],
            old_signature="old",
            new_signature="new",
            recheck_command="kr audit-tokens --force",
        )
    with pytest.raises(ValueError, match="changed field"):
        TokenAuditRequired(
            changed_fields=["batch_size"],
            old_signature="old",
            new_signature="new",
            recheck_command=RECHECK,
        )


def test_scoring_and_selector_errors_name_the_refused_fallback():
    canonical = CanonicalTokenNotSingle(canonical_answer="A", token_count=2)
    assert "continuation contract" in str(canonical)
    assert "Do not approximate by taking its first or last token." in str(canonical)
    with pytest.raises(ValueError, match="one verified token"):
        CanonicalTokenNotSingle(canonical_answer="A", token_count=1)
    selector = UnsupportedPositionSelector(selector="teacher_forced")
    assert "requested before that feature exists" in str(selector)
    assert "Requested selector: teacher_forced" in str(selector)
    assert "Missing component: teacher-forced position selector." in str(selector)
    with pytest.raises(ValueError, match="supported selector"):
        UnsupportedPositionSelector(selector="last_nonpadding_input_token")


@pytest.mark.parametrize("kind", ANSWER_REQUEST_KINDS)
def test_answer_instruction_requires_a_new_audit(kind):
    err = AnswerInstructionMismatch(
        request_kind=kind,
        instruction="Answer with exactly one letter.",
        labels=None if kind == "direct_answer" else ("Yes", "No"),
    )
    assert "one-letter instruction" in str(err)
    assert "Require an appropriate instruction and a new answer/token audit." in str(err)


def test_filter_gate_messages_for_unset_and_contradictory_bases():
    unset = FilterPolicyDecisionRequired(report_id="report-7")
    assert unset.decision_state == "unset" and unset.exit_code != 0
    assert str(unset).splitlines()[:4] == [
        "FilterPolicyDecisionRequired:",
        "Both filtering reports have been saved.",
        "Choose probability_basis=raw or probability_basis=canonical",
        "and record the decision against this report before probe training.",
    ]
    assert "kr record-filter-decision --report report-7 --basis <chosen-basis>" in str(unset)
    conflict = FilterPolicyDecisionRequired(
        report_id="report-7", requested_basis="raw", recorded_basis="canonical"
    )
    assert "requested probability basis 'raw'" in str(conflict)
    assert "recorded decision 'canonical'" in str(conflict)
    assert "reopens the gate after updated diagnostics" in str(conflict)
    with pytest.raises(ValueError, match="matches"):
        FilterPolicyDecisionRequired(
            report_id="report-7", requested_basis="raw", recorded_basis="raw"
        )
    with pytest.raises(ValueError, match="basis"):
        FilterPolicyDecisionRequired(
            report_id="report-7", requested_basis="inferred", recorded_basis="canonical"
        )
    assert PROBABILITY_BASES == ("raw", "canonical")


@pytest.mark.parametrize("kind", ASSET_KINDS)
def test_missing_asset_shows_its_location(kind):
    message = str(MissingLocalAsset(kind=kind, expected_location=f"/assets/{kind}"))
    assert f"pinned {kind} artifact is unavailable offline" in message
    assert f"Expected location: /assets/{kind}" in message
    assert "The artifact was not downloaded." in message


@pytest.mark.parametrize("mismatch", CACHE_MISMATCHES)
def test_cache_errors_name_the_mismatch_or_the_missing_rows(mismatch):
    incompatible = str(CacheIncompatible(mismatches=[mismatch], detail=f"{mismatch} differed"))
    assert "schemas, identities, tensor shapes, or checksums do not match" in incompatible
    assert f"Mismatches: {mismatch}" in incompatible
    assert "probe-only command" in incompatible
    incomplete = CacheIncomplete(missing_keys=["row-a"], missing_layers=[1, 28], extraction_command=EXTRACT)
    text = str(incomplete)
    assert incomplete.missing_keys == ("row-a",) and incomplete.missing_layers == (1, 28)
    assert "Missing keys: row-a" in text and "Missing layers: 1, 28" in text
    assert text.splitlines()[-2] == f"Required extraction command: {EXTRACT}"
    with pytest.raises(ValueError, match="layer indices start at 1"):
        CacheIncomplete(missing_layers=[0], extraction_command=EXTRACT)
