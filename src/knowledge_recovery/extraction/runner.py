"""Batched extraction into the sealed row cache.

A complete cache hit returns references and does not call ``model_factory``.
An out-of-memory failure may lower the batch size. It does not change
precision, weights, or the loaded model. The probe seed is not an argument.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy

from knowledge_recovery.cache.store import ExtractionStore, RowReference
from knowledge_recovery.errors import CacheIncomplete, CacheIncompatible, KnowledgeRecoveryError
from knowledge_recovery.models.base import freeze_model

RowFactory = Callable[[], object]
ForwardBatch = Callable[..., Sequence[Mapping[str, Any]]]


class BatchExceededMemory(Exception):
    """The forward pass must be retried with a smaller batch and the same precision."""


@dataclass(frozen=True)
class ExtractionResult:
    """References that were already stored or just sealed, plus the batches used."""

    references: tuple[RowReference, ...]
    constructed_model: bool
    batch_sizes: tuple[int, ...]
    precision: str


def run_extraction(
    rows: Sequence[Mapping[str, Any]],
    *,
    store: ExtractionStore,
    model_factory: RowFactory | None,
    forward_batch: ForwardBatch | None,
    initial_batch_size: int,
    precision: str,
    probe_only: bool = False,
    probe_only_hook: Callable[[], None] | None = None,
    execution_backend: str = "local",
) -> ExtractionResult:
    """Look up every row. Compute only keys that are absent from the index.

    ``probe_only`` reports missing keys and does not construct a model or
    open a network connection. Incomplete shards are not treated as absent.
    """
    if type(initial_batch_size) is not int or initial_batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    if precision not in ("bf16", "fp16", "fp32"):
        raise ValueError("precision must stay one of bf16, fp16, or fp32")
    keys = tuple(_row_key(row) for row in rows)
    if probe_only:
        with offline_network_guard():
            if probe_only_hook is not None:
                probe_only_hook()
            _require_cached(store, keys)
        return ExtractionResult(
            references=store.lookup(keys, model_factory=_refuse_model, extraction_command="kr extract --config cache"),
            constructed_model=False,
            batch_sizes=(),
            precision=precision,
        )
    missing = _missing_keys(store, keys)
    if not missing:
        return ExtractionResult(
            references=store.lookup(keys, model_factory=_refuse_model, extraction_command="kr extract --config cache"),
            constructed_model=False,
            batch_sizes=(),
            precision=precision,
        )
    if model_factory is None or forward_batch is None:
        raise CacheIncomplete(
            extraction_command="kr extract --config cache",
            missing_keys=missing,
        )
    model = model_factory()
    freeze_model(model)
    pending = [row for row in rows if row["row_key"] in set(missing)]
    batch_sizes = _compute_missing(
        pending,
        store=store,
        model=model,
        forward_batch=forward_batch,
        batch_size=initial_batch_size,
        precision=precision,
        execution_backend=execution_backend,
    )
    return ExtractionResult(
        references=store.lookup(keys, model_factory=_refuse_model, extraction_command="kr extract --config cache"),
        constructed_model=True,
        batch_sizes=batch_sizes,
        precision=precision,
    )


@contextmanager
def offline_network_guard() -> Iterator[None]:
    """Refuse new connections for the duration of a probe-only lookup."""
    original_create = socket.create_connection
    original_connect = socket.socket.connect

    def blocked(*_args: Any, **_kwargs: Any) -> None:
        raise KnowledgeRecoveryError(
            "Probe-only command denied network access.\n"
            "Cached rows are read from the extraction store."
        )

    socket.create_connection = blocked
    socket.socket.connect = blocked
    try:
        yield
    finally:
        socket.create_connection = original_create
        socket.socket.connect = original_connect


def _compute_missing(
    pending: list[Mapping[str, Any]],
    *,
    store: ExtractionStore,
    model: object,
    forward_batch: ForwardBatch,
    batch_size: int,
    precision: str,
    execution_backend: str,
) -> tuple[int, ...]:
    sizes: list[int] = []
    size = batch_size
    while pending:
        chunk = pending[:size]
        try:
            produced = forward_batch(model, chunk, batch_size=size, precision=precision)
        except BatchExceededMemory:
            if size == 1:
                raise
            size = max(1, size // 2)
            continue
        if len(tuple(produced)) != len(chunk):
            raise RuntimeError("the forward pass did not return one row per prompt")
        store_rows = []
        for row, item in zip(chunk, produced, strict=True):
            store_rows.append(_store_row(str(row["row_key"]), item, precision, size, execution_backend))
        store.write_rows(store_rows)
        sizes.append(size)
        pending = pending[len(chunk):]
    return tuple(sizes)


def _store_row(
    key: str,
    produced: Mapping[str, Any],
    precision: str,
    batch_size: int,
    execution_backend: str,
) -> dict[str, Any]:
    residuals = numpy.asarray(produced["residuals"], dtype=numpy.float32)
    logits = numpy.asarray(produced["logits"], dtype=numpy.float32)
    final = numpy.asarray(produced["final_normalized"], dtype=numpy.float32)
    if residuals.ndim != 2 or logits.ndim != 1 or final.ndim != 1:
        raise ValueError("residuals are [layers, hidden], logits are [vocab], and the final norm is [hidden]")
    if residuals.shape[0] < 1 or final.shape != (residuals.shape[1],):
        raise ValueError("the final normalized state has the residual width and is not a layer")
    if numpy.array_equal(final, residuals[-1]):
        raise ValueError(
            "the final normalized state matched the final block residual and was not stored in its place"
        )
    if produced.get("execution_precision", precision) != precision:
        raise RuntimeError("an out-of-memory retry changed execution precision")
    del batch_size, execution_backend
    return {
        "row_key": key,
        "residuals": residuals,
        "logits": logits,
        "final_normalized": final,
    }


def _missing_keys(store: ExtractionStore, keys: Sequence[str]) -> tuple[str, ...]:
    missing = []
    saw_unindexed = False
    for key in keys:
        status = store.row_status(key)
        if status == "indexed":
            continue
        if status == "incomplete":
            raise CacheIncompatible(
                mismatches=("checksum",),
                detail=f"shard for {key} is incomplete and was not resumed or overwritten",
            )
        if status == "incompatible":
            raise CacheIncompatible(
                mismatches=("checksum",),
                detail=f"shard for {key} failed its checksum and was not reused",
            )
        if status == "complete_unindexed":
            saw_unindexed = True
            continue
        if status != "absent":
            raise CacheIncompatible(mismatches=("schema",), detail=f"row {key} has status {status}")
        missing.append(key)
    if saw_unindexed:
        store.commit_index()
        return _missing_keys(store, keys)
    return tuple(missing)


def _require_cached(store: ExtractionStore, keys: Sequence[str]) -> None:
    missing = []
    for key in keys:
        status = store.row_status(key)
        if status == "indexed":
            continue
        if status == "incomplete":
            raise CacheIncompatible(
                mismatches=("checksum",),
                detail=f"shard for {key} is incomplete and a probe-only command cannot finish it",
            )
        missing.append(key)
    if missing:
        raise CacheIncomplete(extraction_command="kr extract --config cache", missing_keys=missing)


def _row_key(row: Mapping[str, Any]) -> str:
    key = row.get("row_key")
    if type(key) is not str or len(key) != 64:
        raise ValueError("each extraction row needs a SHA-256 row key")
    return key


def _refuse_model() -> None:
    raise KnowledgeRecoveryError("a completed cache lookup must not construct a model")
