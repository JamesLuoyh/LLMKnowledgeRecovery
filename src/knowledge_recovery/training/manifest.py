"""Selection manifest persisted before any test evaluation.

The record stores validation losses and the chosen checkpoint. Test labels,
test metrics, and a train-plus-validation refit are not fields.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge_recovery.provenance import content_digest

SELECTION_METRIC = "validation_cross_entropy"
TIE_BREAK = "earliest_checkpoint_then_shallower_layer"
MANIFEST_NAME = "selection_manifest.json"


@dataclass(frozen=True)
class LayerChoice:
    """Best completed checkpoint for one layer, including its training transform."""

    layer: int
    epoch: int
    validation_cross_entropy: float
    weight: tuple[float, ...]
    bias: float
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    preprocessing: str


@dataclass(frozen=True)
class SelectionManifest:
    """Frozen layer choice. ``refit_on_train_validation`` stays false."""

    task_id: str
    method_id: str
    selected_layer: int
    selected_epoch: int
    validation_cross_entropy: float
    layers: tuple[LayerChoice, ...]
    digest: str
    metric: str = SELECTION_METRIC
    tie_break: str = TIE_BREAK
    refit_on_train_validation: bool = False

    def __post_init__(self) -> None:
        if self.metric != SELECTION_METRIC or self.tie_break != TIE_BREAK:
            raise ValueError("selection uses validation cross-entropy with the fixed tie break")
        if self.refit_on_train_validation:
            raise ValueError("the initial protocol does not refit on train plus validation")
        chosen = [item for item in self.layers if item.layer == self.selected_layer]
        if len(chosen) != 1 or chosen[0].epoch != self.selected_epoch:
            raise ValueError("the selected layer must be one recorded layer choice")
        if chosen[0].validation_cross_entropy != self.validation_cross_entropy:
            raise ValueError("the manifest loss is the selected checkpoint's validation loss")


def manifest_record(manifest: SelectionManifest) -> dict[str, Any]:
    """JSON record. The digest is computed from this object with the digest omitted."""
    return {
        "kind": "selection_manifest",
        "schema": 1,
        "task_id": manifest.task_id,
        "method_id": manifest.method_id,
        "metric": manifest.metric,
        "tie_break": manifest.tie_break,
        "refit_on_train_validation": False,
        "selected_layer": manifest.selected_layer,
        "selected_epoch": manifest.selected_epoch,
        "validation_cross_entropy": manifest.validation_cross_entropy,
        "layers": [
            {
                "layer": item.layer,
                "epoch": item.epoch,
                "validation_cross_entropy": item.validation_cross_entropy,
                "weight": list(item.weight),
                "bias": item.bias,
                "mean": list(item.mean),
                "scale": list(item.scale),
                "preprocessing": item.preprocessing,
            }
            for item in manifest.layers
        ],
    }


def selection_digest(record: Mapping[str, Any]) -> str:
    """Digest of a manifest record. Test fields are not part of the record."""
    return content_digest(dict(record))


def write_selection_manifest(directory: Path | str, manifest: SelectionManifest) -> Path:
    """Atomically write the manifest. Callers do this before test evaluation."""
    target = Path(directory)
    if not target.is_absolute():
        raise ValueError("selection manifests are written under an absolute storage directory")
    target.mkdir(parents=True, exist_ok=True)
    payload = {"digest": manifest.digest, "record": manifest_record(manifest)}
    if payload["digest"] != selection_digest(payload["record"]):
        raise ValueError("selection digest does not match the manifest record")
    path = target / MANIFEST_NAME
    temporary = path.with_suffix(".json.partial")
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with temporary.open("wb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def load_selection_manifest(path: Path | str) -> SelectionManifest:
    """Load a manifest that was written before test evaluation."""
    source = Path(path)
    if source.is_dir():
        source = source / MANIFEST_NAME
    if not source.is_file():
        raise FileNotFoundError("selection manifest must be written before test evaluation")
    payload = json.loads(source.read_bytes())
    record = payload["record"]
    if selection_digest(record) != payload["digest"]:
        raise ValueError("selection manifest digest does not match its record")
    layers = tuple(_layer(item) for item in record["layers"])
    return SelectionManifest(
        task_id=record["task_id"],
        method_id=record["method_id"],
        selected_layer=record["selected_layer"],
        selected_epoch=record["selected_epoch"],
        validation_cross_entropy=float(record["validation_cross_entropy"]),
        layers=layers,
        digest=payload["digest"],
        metric=record["metric"],
        tie_break=record["tie_break"],
        refit_on_train_validation=bool(record["refit_on_train_validation"]),
    )


def layer_choices(manifest: SelectionManifest) -> Sequence[LayerChoice]:
    """Validation profile in layer order."""
    return manifest.layers


def _layer(item: Mapping[str, Any]) -> LayerChoice:
    return LayerChoice(
        layer=int(item["layer"]),
        epoch=int(item["epoch"]),
        validation_cross_entropy=float(item["validation_cross_entropy"]),
        weight=tuple(float(value) for value in item["weight"]),
        bias=float(item["bias"]),
        mean=tuple(float(value) for value in item["mean"]),
        scale=tuple(float(value) for value in item["scale"]),
        preprocessing=str(item["preprocessing"]),
    )
