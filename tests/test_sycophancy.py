import ast
import json
from pathlib import Path

import pytest

from knowledge_recovery.datasets.base import DatasetAdapter
from knowledge_recovery.datasets.sycophancy import (
    NORMALIZATION_VERSION,
    ConflictingBaseAnswers,
    SycophancyAdapter,
    base_group_id,
    recover_sycophancy,
)
from knowledge_recovery.provenance import content_digest

SOURCE = "cfierro/sycophancy_eval_answer@pinned"
MODULE = Path(__file__).resolve().parents[1] / "src" / "knowledge_recovery" / "datasets" / "sycophancy.py"


def _row(row_id, template="{question}", **base):
    return {"id": row_id, "prompt_template": template, "base": base}


def test_base_group_id_hashes_the_exact_question():
    question = " café "
    digest = base_group_id(question)
    assert digest == content_digest({
        "kind": "base_group",
        "schema": 1,
        "normalization_version": NORMALIZATION_VERSION,
        "question": question,
    })
    assert digest != base_group_id("café")
    assert " " in question and base_group_id(question) != base_group_id(question.strip())


def test_fixture_counts_templates_invalid_rows_and_collapsed_ids(tmp_path):
    rows = [
        _row("r1", question="What?", correct_answer="yes", incorrect_answer="no"),
        _row("r2", "other {question} please", question="SHOULD_NOT_APPEAR", correct_answer="a", incorrect_answer="a"),
        _row("r3", question="What?", correct_answer="yes", incorrect_answer="no"),
        _row("r4", question="", correct_answer="yes", incorrect_answer="no"),
        _row("r5", question="Equal?", correct_answer="same", incorrect_answer="same"),
        {
            "id": "r6",
            "prompt_template": "{question}",
            "question": "TOP LEVEL ONLY",
            "base": {"correct_answer": "yes", "incorrect_answer": "no"},
        },
        _row("r7", "Explain: {question}", question="Also ignored", correct_answer="a", incorrect_answer="b"),
        _row("r8", question="Spaced?", correct_answer=" yes", incorrect_answer="yes"),
    ]
    report = SycophancyAdapter().recover(rows, source_identity=SOURCE, conflict_directory=tmp_path)
    assert isinstance(SycophancyAdapter(), DatasetAdapter)
    assert report.raw_source_rows == 8
    assert report.template_counts() == {"other {question} please": 1, "Explain: {question}": 1}
    reasons = {note.source_row_id: note.reason for note in report.invalid_rows}
    assert reasons == {
        "r4": "empty_question",
        "r5": "equal_answers",
        "r6": "missing_base_question",
    }
    assert report.recovered_base_items == 2
    assert report.duplicate_rows == 1
    assert report.raw_source_rows == (
        sum(len(item.source_row_ids) for item in report.items) + len(report.invalid_rows) + len(report.exclusions)
    )
    collapsed = report.items[0]
    assert collapsed.source_row_ids == ("r1", "r3")
    assert collapsed.question == "What?"
    assert collapsed.duplicate_group_id == base_group_id("What?")
    spaced = report.items[1]
    assert spaced.correct_answer == " yes"
    assert "SHOULD_NOT_APPEAR" not in {item.question for item in report.items}
    assert "TOP LEVEL ONLY" not in {item.question for item in report.items}
    assert list(tmp_path.iterdir()) == []


def test_conflicting_answers_are_written_and_stop_preparation(tmp_path):
    rows = [
        _row("ok", question="Other?", correct_answer="one", incorrect_answer="two"),
        _row("c1", question=" café ", correct_answer="yes", incorrect_answer="no"),
        _row("c2", question=" café ", correct_answer="yes", incorrect_answer="maybe"),
    ]
    with pytest.raises(ConflictingBaseAnswers) as caught:
        recover_sycophancy(rows, source_identity=SOURCE, conflict_directory=tmp_path)
    error = caught.value
    assert error.exit_code != 0
    assert "stop preparation" in str(error)
    assert "No subset was selected or quarantined." in str(error)
    payload = json.loads((tmp_path / "conflicting_answers.json").read_text(encoding="utf-8"))
    assert payload[0]["question"] == " café "
    assert payload[0]["groups"][0]["source_row_ids"] == ["c1"]
    assert payload[0]["groups"][1]["incorrect_answer"] == "maybe"
    assert "Other?" not in json.dumps(payload)


def test_adapter_does_not_split_or_load_a_model():
    source = MODULE.read_text(encoding="utf-8")
    imported = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    for name in imported:
        assert not name.startswith(("numpy", "torch", "transformers", "random", "knowledge_recovery.models"))
    assert "hash(" not in source
