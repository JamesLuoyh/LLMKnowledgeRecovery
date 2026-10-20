"""Tensor-shard schema.

Equal-width residuals are stored as one ``[n, L, d]`` bfloat16 array.
Architectures whose layers differ in width use one named array per layer.
Full-vocabulary logits are float32. Checksums cover the stored bytes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy

TENSOR_SCHEMA_VERSION = 1
RESIDUAL_DTYPE = "bfloat16"
LOGITS_DTYPE = "float32"
INDEX_KIND = "extraction_index"
SHARD_KIND = "extraction_shard"


def f32_to_bf16(values: numpy.ndarray) -> numpy.ndarray:
    """Round float32 values to bfloat16 and return the raw ``uint16`` bits."""
    bits = numpy.ascontiguousarray(values, dtype=numpy.float32).view(numpy.uint32)
    truncated = bits >> 16
    rounding = ((truncated & 1) + numpy.uint32(0x7FFF)).astype(numpy.uint32)
    return ((bits + rounding) >> 16).astype(numpy.uint16)


def bf16_to_f32(bits: numpy.ndarray) -> numpy.ndarray:
    """Expand stored bfloat16 bits back to float32."""
    shifted = numpy.ascontiguousarray(bits, dtype=numpy.uint16).astype(numpy.uint32) << 16
    return shifted.view(numpy.float32)


def file_checksum(path: Any) -> str:
    """SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_bytes(payload: bytes) -> str:
    """SHA-256 hex digest of an in-memory byte string."""
    return hashlib.sha256(payload).hexdigest()


def shard_manifest(
    *,
    row_keys: Sequence[str],
    layout: str,
    layers: Sequence[int],
    layer_widths: Sequence[int],
    residual_shape: Sequence[int] | None,
    logits_shape: Sequence[int],
    files: Mapping[str, Mapping[str, Any]],
    checksums: Mapping[str, str],
) -> dict[str, Any]:
    """Build the JSON manifest that is sealed before the shard is published."""
    if layout not in ("equal_width", "named_layers"):
        raise ValueError("layout must be equal_width or named_layers")
    if tuple(layers) != tuple(sorted(layers)) or any(layer < 1 for layer in layers):
        raise ValueError("layer indices start at 1 and are stored in order")
    return {
        "kind": SHARD_KIND,
        "tensor_schema_version": TENSOR_SCHEMA_VERSION,
        "residual_dtype": RESIDUAL_DTYPE,
        "logits_dtype": LOGITS_DTYPE,
        "layout": layout,
        "row_keys": list(row_keys),
        "layers": list(layers),
        "layer_widths": list(layer_widths),
        "residual_shape": None if residual_shape is None else list(residual_shape),
        "logits_shape": list(logits_shape),
        "files": {name: dict(spec) for name, spec in files.items()},
        "checksums": dict(checksums),
        "endianness": "little",
    }


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject a manifest whose schema, dtypes, or layer index are unusable."""
    if manifest.get("kind") != SHARD_KIND:
        raise ValueError("shard manifest kind is extraction_shard")
    if manifest.get("tensor_schema_version") != TENSOR_SCHEMA_VERSION:
        raise ValueError("tensor schema version does not match")
    if manifest.get("residual_dtype") != RESIDUAL_DTYPE or manifest.get("logits_dtype") != LOGITS_DTYPE:
        raise ValueError("residual storage is bfloat16 and logits storage is float32")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or any(type(layer) is not int or layer < 1 for layer in layers):
        raise ValueError("layer indices start at 1")
    if manifest.get("endianness") != "little":
        raise ValueError("shards are little-endian")
