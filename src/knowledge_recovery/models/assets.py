"""Offline local asset resolution.

Digests name file bytes. Moving a directory does not change them, and a
missing path is not downloaded. Cache-only callers skip this resolution.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from knowledge_recovery.errors import MissingLocalAsset
from knowledge_recovery.provenance import content_digest

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


@dataclass(frozen=True)
class LocalAssets:
    """Content digests of a pinned local snapshot. Weights are not loaded."""

    model_path: str
    weight_content_digest: str
    model_config_digest: str
    tokenizer_content_digest: str


def activate_offline(huggingface_home: Path | str | None = None) -> None:
    """Set offline flags before a model or dataset library is initialized."""
    for key, value in OFFLINE_ENV.items():
        os.environ[key] = value
    if huggingface_home is not None:
        home = Path(huggingface_home)
        home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(home / "hub")


def resolve_local_assets(model_path: Path | str) -> LocalAssets:
    """Hash a local model directory. The directory is not created or downloaded."""
    activate_offline()
    root = Path(model_path)
    if not root.exists():
        raise MissingLocalAsset(kind="model", expected_location=str(root))
    if not root.is_dir():
        raise MissingLocalAsset(kind="model", expected_location=str(root))
    config = root / "config.json"
    if not config.is_file():
        raise MissingLocalAsset(kind="model", expected_location=str(config))
    tokenizer_files = _tokenizer_files(root)
    if not tokenizer_files:
        raise MissingLocalAsset(kind="tokenizer", expected_location=str(root / "tokenizer_map.json"))
    weight_files = [
        path
        for path in _files(root)
        if path.name != "config.json" and path not in tokenizer_files
    ]
    if not weight_files:
        raise MissingLocalAsset(kind="model", expected_location=str(root / "weights.bin"))
    return LocalAssets(
        model_path=str(root),
        weight_content_digest=_tree_digest(weight_files, root),
        model_config_digest=_tree_digest((config,), root),
        tokenizer_content_digest=_tree_digest(tokenizer_files, root),
    )


def resolve_execution_precision(
    requested: str,
    *,
    bf16_supported: bool,
    fp16_supported: bool,
) -> str:
    """Return ``requested`` or fail. Precision is not replaced with another dtype."""
    if requested == "bf16" and not bf16_supported:
        raise ValueError(
            "extraction.model.precision 'bf16' is not supported on this GPU. "
            "There is no silent fallback."
        )
    if requested == "fp16" and not fp16_supported:
        raise ValueError(
            "extraction.model.precision 'fp16' is not supported on this GPU. "
            "There is no silent fallback."
        )
    if requested not in ("bf16", "fp16", "fp32"):
        raise ValueError(f"precision {requested!r} is not an explicit precision")
    return requested


def _tokenizer_files(root: Path) -> tuple[Path, ...]:
    names = ("tokenizer_map.json", "tokenizer.json", "tokenizer_config.json")
    return tuple(path for name in names if (path := root / name).is_file())


def _files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted((path for path in root.rglob("*") if path.is_file()), key=lambda item: item.as_posix()))


def _tree_digest(paths: Sequence[Path], root: Path) -> str:
    files = [
        {"path": path.relative_to(root).as_posix(), "sha256": _file_sha256(path)}
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix())
    ]
    return content_digest({"kind": "asset_files", "schema": 1, "files": files})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
