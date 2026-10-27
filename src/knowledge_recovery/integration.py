"""Labeled local subset through extraction and the fixed filter.

The subset is not the full sycophancy corpus. A Qwen snapshot is loaded only
when its directory is the model path and its config says so. This module's
own snapshot is ``model_type: integration_mock`` and is labeled as such.
The filter does not stop to choose a raw-versus-canonical probability basis.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy
import yaml

from knowledge_recovery.cli import StageRequest
from knowledge_recovery.datasets.sycophancy import base_group_id
from knowledge_recovery.errors import MissingLocalAsset
from knowledge_recovery.models.qwen import forward_padded
from knowledge_recovery.pipeline import STAGE_ORDER, STAGE_HANDLERS

SUBSET_LABEL = "integration-subset"
PASS_QUESTION = "integration-subset pass"
FAIL_QUESTION = "integration-subset fail-b"
TOKEN_A = 10
TOKEN_B = 11
VOCAB = 32
FILTER_STAGES = ("validate", "prepare", "audit-tokens", "extract", "diagnose")


def subset_forward(
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    precision: str,
) -> tuple[dict[str, numpy.ndarray], ...]:
    """Drive the named hooks, then place the labeled logits on the last real token."""
    model.pending_logits = [
        labeled_logits(row["question"], row["order_id"], row.get("variant_id", "clean"))
        for row in rows
    ]
    try:
        return forward_padded(model, rows, batch_size=batch_size, precision=precision)
    finally:
        model.pending_logits = None


def labeled_logits(question: str, order_id: str, variant_id: str = "clean") -> numpy.ndarray:
    """Fixed full-vocabulary logits for the two labeled questions.

    The failing order puts its mass on a token outside A and B. The others
    put it on the correct option, above the 0.8 normalized threshold.
    """
    logits = numpy.full(VOCAB, -10.0, dtype=numpy.float32)
    if question == FAIL_QUESTION and order_id == "correct_at_B":
        logits[0] = 5.0
        return logits
    preferred = TOKEN_A if order_id == "correct_at_A" else TOKEN_B
    other = TOKEN_B if preferred == TOKEN_A else TOKEN_A
    if variant_id in ("T-IC", "NT-IC"):
        preferred, other = other, preferred
    logits[preferred] = 5.0
    logits[other] = 0.0
    return logits


def run_labeled_subset(directory: Path | str, *, model_path: str | None = None) -> dict[str, Any]:
    """Run the subset through the filter gate.

    ``model_path`` must be supplied by the caller. When it is omitted, this
    function builds the labeled mock snapshot inside ``directory`` and passes
    that path explicitly. It does not search a default cache.
    """
    root = Path(directory)
    if model_path is None:
        model_path = str(_write_mock_snapshot(root / "model"))
    if model_path == "":
        raise MissingLocalAsset(kind="model", expected_location="(no path supplied)")
    storage = root / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    data = root / "rows.json"
    data.write_text(json.dumps(_source_rows(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config_path = root / "config.yaml"
    _write_config(config_path, storage=storage, model=Path(model_path), dataset=data)
    for stage in FILTER_STAGES:
        STAGE_HANDLERS[stage](StageRequest(stage=stage, config=str(config_path)))
    report = json.loads((storage / "analyses" / "filter" / "filter_report.json").read_text(encoding="utf-8"))
    return {
        "storage": storage,
        "model_path": model_path,
        "eligible_base_group_ids": report["record"]["paired_eligibility"]["eligible_base_group_ids"],
        "pass_group": base_group_id(PASS_QUESTION),
        "fail_group": base_group_id(FAIL_QUESTION),
        "report": report,
    }


def _write_mock_snapshot(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({
        "model_type": "integration_mock",
        "subset_label": SUBSET_LABEL,
        "hidden_size": 4,
        "num_hidden_layers": 2,
        "vocab_size": VOCAB,
        "logit_rule": "integration_subset_v1",
        "torch_dtype": "bfloat16",
    }, indent=2) + "\n", encoding="utf-8")
    (path / "weights.bin").write_bytes(b"integration-subset-weights-v1")
    (path / "tokenizer_map.json").write_text(json.dumps({
        "kind": "mapped_continuation_tokenizer",
        "schema": 1,
        "settings": {"add_special_tokens": False, "kind": "mapped_continuation"},
        "continuations": {"A": [TOKEN_A], "B": [TOKEN_B], " A": [12], " B": [13]},
        "decoded": {"10": "41", "11": "42", "12": "2041", "13": "2042"},
    }, indent=2) + "\n", encoding="utf-8")
    return path


def _source_rows() -> list[dict[str, Any]]:
    return [
        _row("subset-pass", PASS_QUESTION, "yes", "no"),
        _row("subset-fail", FAIL_QUESTION, "left", "right"),
    ]


def _row(row_id: str, question: str, correct: str, incorrect: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "prompt_template": "{question}",
        "base": {
            "question": question,
            "correct_answer": correct,
            "incorrect_answer": incorrect,
        },
    }


def _write_config(path: Path, *, storage: Path, model: Path, dataset: Path) -> None:
    shipped = Path(__file__).resolve().parents[2] / "configs" / "initial_sycophancy.yaml"
    document = yaml.safe_load(shipped.read_text(encoding="utf-8"))
    document["extraction"]["storage"]["root"] = str(storage)
    document["extraction"]["assets"]["model_path"] = str(model)
    document["extraction"]["assets"]["dataset_path"] = str(dataset)
    path.write_text(yaml.safe_dump(document), encoding="utf-8")


# Imported so a reader can see the gate stops before training.
assert FILTER_STAGES[-1] == "diagnose"
assert STAGE_ORDER.index("diagnose") < STAGE_ORDER.index("train")
