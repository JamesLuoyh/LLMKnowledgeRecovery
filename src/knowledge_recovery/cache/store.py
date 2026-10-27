"""Atomic extraction shards and the global row-key index.

A shard is staged, checksummed, and sealed with a completion marker before
the index learns its keys. Lookup of a sealed, indexed row returns a
reference and does not construct a model. A probe-only lookup of anything
else reports the missing key and does not infer the row.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy

from knowledge_recovery.cache.schemas import (
    LOGITS_DTYPE,
    RESIDUAL_DTYPE,
    TENSOR_SCHEMA_VERSION,
    bf16_to_f32,
    checksum_bytes,
    f32_to_bf16,
    file_checksum,
    shard_manifest,
    validate_manifest,
)
from knowledge_recovery.errors import CacheIncomplete, CacheIncompatible

INDEX_NAME = "index.json"
COMPLETE_NAME = "COMPLETE"


@dataclass(frozen=True)
class RowReference:
    """Location of one sealed row. The arrays stay on disk."""

    extraction_row_key: str
    shard_id: str
    row_offset: int
    residual_path: str
    logits_path: str
    layers: tuple[int, ...]
    layout: str


@dataclass(frozen=True)
class StagedShard:
    """A shard directory whose completion marker has not been written."""

    shard_id: str
    row_keys: tuple[str, ...]


class ExtractionStore:
    """Immutable shards under ``<root>/extractions`` plus one row-key index."""

    def __init__(self, storage_root: Path | str) -> None:
        self.root = Path(storage_root)
        if not self.root.is_absolute():
            raise ValueError("storage_root must be an absolute path outside the repository")
        if not self.root.exists():
            raise ValueError("storage_root must exist before any shard is written")
        self.extraction_root = self.root / "extractions"
        self.shard_root = self.extraction_root / "shards"
        self.staging_root = self.root / "tmp" / "shards"
        self.index_path = self.extraction_root / INDEX_NAME
        self.lock_path = self.extraction_root / ".store.lock"
        self.shard_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)

    def stage_rows(self, rows: Sequence[Mapping[str, Any]]) -> StagedShard:
        """Write tensor bytes and a manifest. The shard stays incomplete."""
        parsed = _rows(rows)
        keys = tuple(item["row_key"] for item in parsed)
        if any(status != "absent" for status in self.row_statuses(keys).values()):
            raise ValueError("a row key that is already stored is not staged again")
        return self._stage_parsed(parsed)

    def batched_writer(self, *, flush_every: int) -> BatchedIndexWriter:
        """Seal shards per call and publish their keys every ``flush_every`` shards."""
        return BatchedIndexWriter(self, flush_every=flush_every)

    def _stage_parsed(self, parsed: Sequence[Mapping[str, Any]]) -> StagedShard:
        keys = tuple(item["row_key"] for item in parsed)
        shard_id = checksum_bytes("\n".join(keys).encode())
        directory = self.staging_root / shard_id
        if directory.exists() or (self.shard_root / shard_id).exists():
            raise ValueError("a shard for these row keys is already present")
        directory.mkdir()
        spec = _write_tensors(directory, parsed)
        manifest = shard_manifest(row_keys=keys, checksums=_checksums(directory, spec["files"]), **spec["manifest"])
        if spec["final_norm_shape"] is not None:
            manifest["final_norm_shape"] = spec["final_norm_shape"]
        payload = _dumps(manifest).encode()
        (directory / "manifest.json").write_bytes(payload)
        _verify_files(directory, manifest)
        final = self.shard_root / shard_id
        os.rename(directory, final)
        return StagedShard(shard_id=shard_id, row_keys=keys)

    def complete_shard(self, staged: StagedShard) -> None:
        """Verify the staged bytes and write the completion marker last."""
        directory = self.shard_root / staged.shard_id
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise CacheIncomplete(
                extraction_command="kr extract --config staged",
                missing_keys=staged.row_keys,
            )
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
        _verify_files(directory, manifest)
        _atomic_bytes(directory / COMPLETE_NAME, checksum_bytes(payload).encode())

    def commit_index(self) -> None:
        """Publish every sealed shard under the global-index lock."""
        with self._exclusive_lock():
            self._commit_index_unlocked()

    def _commit_index_unlocked(self, shard_ids: Sequence[str] | None = None) -> None:
        """Index sealed shards. Shards already named by the index are not re-read."""
        index = _read_index(self.index_path)
        if shard_ids is None:
            indexed = {entry["shard_id"] for entry in index["rows"].values()}
            shard_ids = sorted(path.name for path in self.shard_root.iterdir() if path.name not in indexed)
        for shard_id in shard_ids:
            directory = self.shard_root / shard_id
            if not (directory / COMPLETE_NAME).is_file():
                continue
            manifest = _sealed_manifest(directory)
            for offset, key in enumerate(manifest["row_keys"]):
                index["rows"][key] = {"shard_id": shard_id, "row_offset": offset}
        _atomic_bytes(self.index_path, _dumps(index).encode())

    def write_rows(self, rows: Sequence[Mapping[str, Any]]) -> tuple[RowReference, ...]:
        """Stage, seal, and index rows that are not already published."""
        with self._exclusive_lock():
            statuses = self.row_statuses([str(row["row_key"]) for row in rows])
            pending = [row for row in rows if statuses[str(row["row_key"])] == "absent"]
            if pending:
                staged = self.stage_rows(pending)
                self.complete_shard(staged)
                self._commit_index_unlocked([staged.shard_id])
        return self.lookup(
            [str(row["row_key"]) for row in rows],
            model_factory=_forbidden_model,
            extraction_command="kr extract --config cache-write",
        )

    def write_collection_manifest(self, path: Path, payload: Mapping[str, Any]) -> None:
        """Atomically publish the collection pointer under the store lock."""
        expected = self.extraction_root / "collection.json"
        if path != expected:
            raise ValueError(f"collection manifest must be written to {expected}")
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        with self._exclusive_lock():
            _atomic_bytes(path, encoded)

    @contextmanager
    def _exclusive_lock(self):
        """Serialize row reservation, shard publication, and index updates."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def row_status(self, row_key: str) -> str:
        """Distinguish a missing key from an unfinished or unpublished shard."""
        return self.row_statuses([row_key])[row_key]

    def row_statuses(self, row_keys: Sequence[str]) -> dict[str, str]:
        """``row_status`` for many keys, verifying each referenced shard once."""
        keys = [_key(item) for item in row_keys]
        index = _read_index(self.index_path)
        statuses: dict[str, str] = {}
        verified: dict[str, bool] = {}
        unindexed = []
        for key in keys:
            if key not in index["rows"]:
                unindexed.append(key)
                continue
            shard_id = index["rows"][key]["shard_id"]
            if shard_id not in verified:
                try:
                    _sealed_manifest(self.shard_root / shard_id)
                    verified[shard_id] = True
                except CacheIncompatible:
                    verified[shard_id] = False
            statuses[key] = "indexed" if verified[shard_id] else "incompatible"
        if unindexed:
            indexed_shards = {entry["shard_id"] for entry in index["rows"].values()}
            found: dict[str, str] = {}
            for directory in self.shard_root.iterdir():
                manifest_path = directory / "manifest.json"
                if directory.name in indexed_shards or not manifest_path.is_file():
                    continue
                state = "complete_unindexed" if (directory / COMPLETE_NAME).is_file() else "incomplete"
                for key in json.loads(manifest_path.read_bytes()).get("row_keys", []):
                    found.setdefault(key, state)
            for key in unindexed:
                statuses[key] = found.get(key, "absent")
        return statuses

    def lookup(
        self,
        row_keys: Sequence[str],
        *,
        model_factory: Callable[[], object],
        extraction_command: str,
        required_layers: Sequence[int] | None = None,
    ) -> tuple[RowReference, ...]:
        """Return sealed references. ``model_factory`` is never called."""
        del model_factory
        keys = tuple(_key(item) for item in row_keys)
        index = _read_index(self.index_path)
        missing = [key for key in keys if key not in index["rows"]]
        if missing:
            raise CacheIncomplete(extraction_command=extraction_command, missing_keys=missing)
        references = []
        manifests: dict[str, dict[str, Any]] = {}
        for key in keys:
            entry = index["rows"][key]
            directory = self.shard_root / entry["shard_id"]
            if entry["shard_id"] not in manifests:
                manifests[entry["shard_id"]] = _sealed_manifest(directory)
            manifest = manifests[entry["shard_id"]]
            offset = entry["row_offset"]
            if manifest["row_keys"][offset] != key:
                raise CacheIncompatible(mismatches=("identity",), detail=f"index offset for {key} names another row")
            layers = tuple(manifest["layers"])
            if required_layers is not None:
                absent_layers = [layer for layer in required_layers if layer not in layers]
                if absent_layers:
                    raise CacheIncomplete(extraction_command=extraction_command, missing_layers=absent_layers)
            residual_name = "residual.bf16" if manifest["layout"] == "equal_width" else f"layer_{layers[0]}.bf16"
            references.append(RowReference(
                extraction_row_key=key,
                shard_id=entry["shard_id"],
                row_offset=offset,
                residual_path=str(directory / residual_name),
                logits_path=str(directory / "logits.f32"),
                layers=layers,
                layout=manifest["layout"],
            ))
        return tuple(references)

    def read_residual_row(self, reference: RowReference) -> numpy.ndarray:
        """Memory-map stored residuals for one row. Logits are not opened."""
        directory = self.shard_root / reference.shard_id
        manifest = _sealed_manifest(directory)
        if manifest["layout"] == "equal_width":
            shape = tuple(manifest["residual_shape"])
            mapped = numpy.memmap(directory / "residual.bf16", dtype="<u2", mode="r", shape=shape)
            return bf16_to_f32(numpy.array(mapped[reference.row_offset]))
        layers = []
        widths = []
        for layer, width in zip(manifest["layers"], manifest["layer_widths"], strict=True):
            mapped = numpy.memmap(
                directory / f"layer_{layer}.bf16",
                dtype="<u2",
                mode="r",
                shape=(len(manifest["row_keys"]), width),
            )
            layers.append(bf16_to_f32(numpy.array(mapped[reference.row_offset])))
            widths.append(width)
        return numpy.stack(layers, axis=0) if len(set(widths)) == 1 else numpy.array(layers, dtype=object)

    def read_logits_row(self, reference: RowReference) -> numpy.ndarray:
        """Memory-map one stored full-vocabulary logit vector."""
        directory = self.shard_root / reference.shard_id
        manifest = _sealed_manifest(directory)
        shape = tuple(manifest["logits_shape"])
        mapped = numpy.memmap(directory / "logits.f32", dtype="<f4", mode="r", shape=shape)
        return numpy.array(mapped[reference.row_offset], dtype=numpy.float32)

    def read_final_norm_row(self, reference: RowReference) -> numpy.ndarray:
        """Memory-map the separately stored final norm for one row."""
        directory = self.shard_root / reference.shard_id
        manifest = _sealed_manifest(directory)
        shape = tuple(manifest["final_norm_shape"])
        mapped = numpy.memmap(directory / "final_norm.bf16", dtype="<u2", mode="r", shape=shape)
        return bf16_to_f32(numpy.array(mapped[reference.row_offset]))


class BatchedIndexWriter:
    """Write many shards without rereading or rewriting the index for each one.

    ``write_rows`` on the store parses the whole index and scans every shard
    directory per call, which grows with the store. This writer reads the index
    once, seals each shard with its completion marker immediately, and merges
    the new keys into a fresh read of the on-disk index every ``flush_every``
    shards and on exit. A crash between flushes leaves sealed, unindexed
    shards, which the next lookup reports as ``complete_unindexed`` and
    ``commit_index`` publishes. Keys sealed by another process after this
    writer started are not seen; run one writer per store.
    """

    def __init__(self, store: ExtractionStore, *, flush_every: int) -> None:
        if type(flush_every) is not int or flush_every < 1:
            raise ValueError("flush_every must be a positive integer")
        self.store = store
        self.flush_every = flush_every
        self.known = set(_read_index(store.index_path)["rows"])
        self.unpublished: dict[str, dict[str, Any]] = {}
        self.shards_since_flush = 0

    def __enter__(self) -> BatchedIndexWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.flush()

    def write_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Seal rows whose keys are not stored yet. References come from ``lookup`` after a flush."""
        parsed = [item for item in _rows(rows) if item["row_key"] not in self.known]
        if not parsed:
            return
        with self.store._exclusive_lock():
            staged = self.store._stage_parsed(parsed)
            self.store.complete_shard(staged)
        for offset, key in enumerate(staged.row_keys):
            self.unpublished[key] = {"shard_id": staged.shard_id, "row_offset": offset}
            self.known.add(key)
        self.shards_since_flush += 1
        if self.shards_since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Merge unpublished keys into the on-disk index under the store lock."""
        if not self.unpublished:
            return
        with self.store._exclusive_lock():
            index = _read_index(self.store.index_path)
            index["rows"].update(self.unpublished)
            _atomic_bytes(self.store.index_path, _dumps(index).encode())
        self.unpublished.clear()
        self.shards_since_flush = 0


def _rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("at least one extraction row is required")
    parsed = []
    seen: set[str] = set()
    for row in rows:
        key = _key(row.get("row_key"))
        if key in seen:
            raise ValueError("duplicate row key in one shard")
        seen.add(key)
        residuals = row.get("residuals")
        logits = numpy.asarray(row.get("logits"), dtype=numpy.float32)
        if logits.ndim != 1 or logits.shape[0] < 1:
            raise ValueError("logits must be a full-vocabulary vector")
        if row.get("final_normalized") is None:
            raise ValueError("every row stores its final normalized state")
        final = numpy.asarray(row["final_normalized"], dtype=numpy.float32)
        if final.ndim != 1:
            raise ValueError("final normalized state is one vector per row")
        parsed.append({
            "row_key": key,
            "residuals": residuals,
            "logits": logits,
            "final_normalized": final,
        })
    return parsed


def _write_tensors(directory: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    logits = numpy.stack([row["logits"] for row in rows]).astype("<f4", copy=False)
    logits_path = directory / "logits.f32"
    numpy.ascontiguousarray(logits, dtype="<f4").tofile(logits_path)
    first = rows[0]["residuals"]
    files: dict[str, dict[str, Any]] = {
        "logits.f32": {"dtype": LOGITS_DTYPE, "shape": list(logits.shape)},
    }
    if isinstance(first, Mapping):
        layers = tuple(sorted(int(layer) for layer in first))
        widths = []
        for layer in layers:
            stacked = numpy.stack([numpy.asarray(row["residuals"][layer], dtype=numpy.float32) for row in rows])
            name = f"layer_{layer}.bf16"
            numpy.ascontiguousarray(f32_to_bf16(stacked), dtype="<u2").tofile(directory / name)
            files[name] = {"dtype": RESIDUAL_DTYPE, "shape": list(stacked.shape)}
            widths.append(int(stacked.shape[1]))
        manifest = {
            "layout": "named_layers",
            "layers": layers,
            "layer_widths": widths,
            "residual_shape": None,
            "logits_shape": list(logits.shape),
            "files": files,
        }
    else:
        stacked = numpy.stack([numpy.asarray(row["residuals"], dtype=numpy.float32) for row in rows])
        if stacked.ndim != 3:
            raise ValueError("equal-width residuals have shape [n, layers, width]")
        name = "residual.bf16"
        numpy.ascontiguousarray(f32_to_bf16(stacked), dtype="<u2").tofile(directory / name)
        files[name] = {"dtype": RESIDUAL_DTYPE, "shape": list(stacked.shape)}
        manifest = {
            "layout": "equal_width",
            "layers": tuple(range(1, stacked.shape[1] + 1)),
            "layer_widths": (int(stacked.shape[2]),) * stacked.shape[1],
            "residual_shape": list(stacked.shape),
            "logits_shape": list(logits.shape),
            "files": files,
        }
    final_shape = _write_final_norm(directory, rows, files)
    return {"files": files, "manifest": manifest, "final_norm_shape": final_shape}


def _write_final_norm(
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
    files: dict[str, dict[str, Any]],
) -> list[int]:
    """Store the final norm beside the block residuals. It is not a layer column."""
    stacked = numpy.stack([numpy.asarray(row["final_normalized"], dtype=numpy.float32) for row in rows])
    name = "final_norm.bf16"
    numpy.ascontiguousarray(f32_to_bf16(stacked), dtype="<u2").tofile(directory / name)
    files[name] = {"dtype": RESIDUAL_DTYPE, "shape": list(stacked.shape)}
    return list(stacked.shape)


def _checksums(directory: Path, files: Mapping[str, Any]) -> dict[str, str]:
    return {name: file_checksum(directory / name) for name in files}


def _verify_files(directory: Path, manifest: Mapping[str, Any]) -> None:
    validate_manifest(manifest)
    for name, digest in manifest["checksums"].items():
        path = directory / name
        if not path.is_file() or file_checksum(path) != digest:
            raise CacheIncompatible(mismatches=("checksum",), detail=f"checksum mismatch for {name}")
        spec = manifest["files"][name]
        itemsize = 2 if name.endswith(".bf16") else 4
        expected = int(numpy.prod(spec["shape"])) * itemsize
        if path.stat().st_size != expected:
            raise CacheIncompatible(mismatches=("tensor_shape",), detail=f"byte length of {name} does not match its shape")


def _sealed_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    payload = manifest_path.read_bytes()
    marker = (directory / COMPLETE_NAME).read_text()
    if marker != checksum_bytes(payload):
        raise CacheIncompatible(mismatches=("checksum",), detail="completion marker does not match the manifest")
    manifest = json.loads(payload)
    try:
        validate_manifest(manifest)
    except ValueError as exc:
        raise CacheIncompatible(mismatches=("schema",), detail=str(exc)) from exc
    _verify_files(directory, manifest)
    return manifest


def _read_index(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"kind": "extraction_index", "schema": TENSOR_SCHEMA_VERSION, "rows": {}}
    index = json.loads(path.read_bytes())
    if index.get("kind") != "extraction_index" or index.get("schema") != TENSOR_SCHEMA_VERSION:
        raise CacheIncompatible(mismatches=("schema",), detail="row-key index schema does not match")
    return index


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _key(value: object) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("extraction row key must be a full SHA-256 hex digest")
    return value


def _forbidden_model() -> None:
    raise RuntimeError("a completed cache write must not construct a model")
