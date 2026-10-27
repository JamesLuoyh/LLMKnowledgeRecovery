import json

import numpy
import pytest

from knowledge_recovery.cache import (
    RESIDUAL_DTYPE,
    ExtractionStore,
    bf16_to_f32,
    cache_row_key,
    f32_to_bf16,
)
from knowledge_recovery.cache.schemas import checksum_bytes
from knowledge_recovery.errors import CacheIncomplete, CacheIncompatible
from knowledge_recovery.provenance import content_digest

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
COMMAND = "kr extract --config configs/initial_sycophancy.yaml"


def _row(key: str, *, scale: float = 1.0, named: bool = False) -> dict:
    logits = numpy.array([scale, -scale, 0.25], dtype=numpy.float32)
    if named:
        residuals = {1: numpy.array([scale, 0.0], dtype=numpy.float32), 2: numpy.array([scale], dtype=numpy.float32)}
    else:
        residuals = numpy.array([[scale, 0.0], [-scale, 0.5]], dtype=numpy.float32)
    final = numpy.array([scale + 4, 1.0], dtype=numpy.float32)
    return {"row_key": key, "residuals": residuals, "logits": logits, "final_normalized": final}


def _boom():
    raise AssertionError("model construction is forbidden")


def test_reporting_settings_do_not_allocate_a_new_extraction_row():
    first = cache_row_key(
        model_signature=DIGEST_A,
        prompt_signature=DIGEST_B,
        extraction_signature=DIGEST_C,
        reporting={"ece_bins": 15, "run_id": "ab" * 32, "training_seed": 1, "ood_variants": ["NT-C"]},
    )
    second = cache_row_key(
        model_signature=DIGEST_A,
        prompt_signature=DIGEST_B,
        extraction_signature=DIGEST_C,
        reporting={"ece_bins": 20, "run_id": "cd" * 32, "training_seed": 2, "split_seed": 9},
    )
    assert first == second
    changed_prompt = cache_row_key(
        model_signature=DIGEST_A,
        prompt_signature="d" * 64,
        extraction_signature=DIGEST_C,
        reporting={"ece_bins": 15},
    )
    assert changed_prompt != first
    assert len(first) == 64


def test_staged_shards_stay_incomplete_until_the_completion_marker_and_index(tmp_path):
    store = ExtractionStore(tmp_path)
    key = _published_key("row-one")
    assert store.row_status(key) == "absent"
    staged = store.stage_rows([_row(key)])
    assert store.row_status(key) == "incomplete"
    with pytest.raises(CacheIncomplete, match="Never silently fall back") as missing:
        store.lookup([key], model_factory=_boom, extraction_command=COMMAND)
    assert key in missing.value.missing_keys
    store.complete_shard(staged)
    assert store.row_status(key) == "complete_unindexed"
    with pytest.raises(CacheIncomplete, match=key):
        store.lookup([key], model_factory=_boom, extraction_command=COMMAND)
    store.commit_index()
    assert store.row_status(key) == "indexed"
    references = store.lookup([key], model_factory=_boom, extraction_command=COMMAND)
    assert references[0].extraction_row_key == key
    assert references[0].row_offset == 0
    residual = store.read_residual_row(references[0])
    assert residual.shape == (2, 2)
    assert residual.dtype == numpy.float32
    assert numpy.allclose(residual[0, 0], 1.0)


def test_a_second_write_reuses_the_indexed_row_and_keeps_older_keys(tmp_path):
    store = ExtractionStore(tmp_path)
    first = _published_key("first")
    second = _published_key("second")
    store.write_rows([_row(first)])
    store.write_rows([_row(first), _row(second, scale=2.0)])
    references = store.lookup([first, second], model_factory=_boom, extraction_command=COMMAND)
    assert [item.extraction_row_key for item in references] == [first, second]
    shard_ids = {item.shard_id for item in references}
    assert len(shard_ids) == 2
    assert store.row_status(first) == "indexed"
    reread = store.write_rows([_row(first)])
    assert reread[0].shard_id == references[0].shard_id
    assert len(list((tmp_path / "extractions" / "shards").iterdir())) == 2


def test_checksum_schema_and_identity_mismatches_are_incompatible(tmp_path):
    store = ExtractionStore(tmp_path)
    key = _published_key("sealed")
    other = _published_key("other")
    references = store.write_rows([_row(key), _row(other, scale=3.0)])
    logits = (tmp_path / "extractions" / "shards" / references[0].shard_id / "logits.f32")
    payload = bytearray(logits.read_bytes())
    payload[0] ^= 0xFF
    logits.write_bytes(payload)
    with pytest.raises(CacheIncompatible, match="checksum") as incompatible:
        store.lookup([key], model_factory=_boom, extraction_command=COMMAND)
    assert incompatible.value.mismatches == ("checksum",)
    assert "Never silently fall back" in str(incompatible.value)

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    store = ExtractionStore(fresh)
    key = _published_key("schema")
    store.write_rows([_row(key, named=True)])
    directory = next((fresh / "extractions" / "shards").iterdir())
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["layout"] == "named_layers"
    assert manifest["layer_widths"] == [2, 1]
    assert manifest["residual_dtype"] == RESIDUAL_DTYPE
    manifest["tensor_schema_version"] = 2
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_path.write_bytes(body)
    (directory / "COMPLETE").write_text(checksum_bytes(body))
    with pytest.raises(CacheIncompatible, match="schema"):
        store.lookup([key], model_factory=_boom, extraction_command=COMMAND)

    manifest["tensor_schema_version"] = 1
    manifest["row_keys"] = [other]
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_path.write_bytes(body)
    (directory / "COMPLETE").write_text(checksum_bytes(body))
    with pytest.raises(CacheIncompatible, match="identity"):
        store.lookup([key], model_factory=_boom, extraction_command=COMMAND)


def test_bfloat16_storage_round_trips_the_truncated_value():
    original = numpy.array([1.5, -2.0, 0.0], dtype=numpy.float32)
    restored = bf16_to_f32(f32_to_bf16(original))
    assert restored.dtype == numpy.float32
    assert numpy.array_equal(restored, original)


def test_probe_only_lookup_reports_each_missing_key_without_inference(tmp_path):
    store = ExtractionStore(tmp_path)
    present = _published_key("present")
    absent = _published_key("absent")
    store.write_rows([_row(present)])
    called = {"model": False}

    def factory():
        called["model"] = True
        return object()

    with pytest.raises(CacheIncomplete) as missing:
        store.lookup([present, absent], model_factory=factory, extraction_command=COMMAND)
    assert missing.value.missing_keys == (absent,)
    assert called["model"] is False
    with pytest.raises(CacheIncomplete, match="Missing layers: 9"):
        store.lookup([present], model_factory=factory, extraction_command=COMMAND, required_layers=(1, 9))


def test_a_write_verifies_only_its_own_shard(tmp_path, monkeypatch):
    from knowledge_recovery.cache import store as store_module

    store = ExtractionStore(tmp_path)
    for label in range(5):
        store.write_rows([_row(_published_key(f"earlier-{label}"))])
    verified = []
    original = store_module._sealed_manifest
    monkeypatch.setattr(store_module, "_sealed_manifest", lambda path: verified.append(path.name) or original(path))
    references = store.write_rows([_row(_published_key("newest"))])
    assert set(verified) == {references[0].shard_id}


def _published_key(label: str) -> str:
    return content_digest({"kind": "test_row", "label": label})


def test_batched_writer_publishes_keys_in_bulk_and_skips_stored_rows(tmp_path):
    store = ExtractionStore(tmp_path)
    earlier = _published_key("earlier")
    store.write_rows([_row(earlier)])
    keys = [_published_key(f"batched-{index}") for index in range(5)]
    index_path = tmp_path / "extractions" / "index.json"
    with store.batched_writer(flush_every=2) as writer:
        writer.write_rows([_row(earlier), _row(keys[0])])
        assert store.row_status(keys[0]) == "complete_unindexed"
        writer.write_rows([_row(keys[1])])
        assert set(json.loads(index_path.read_bytes())["rows"]) == {earlier, keys[0], keys[1]}
        writer.write_rows([_row(keys[2]), _row(keys[3])])
        writer.write_rows([_row(keys[3])])
        assert store.row_status(keys[2]) == "complete_unindexed"
    references = store.lookup([earlier, *keys[:4]], model_factory=_boom, extraction_command=COMMAND)
    assert [item.row_offset for item in references] == [0, 0, 0, 0, 1]
    assert len(list((tmp_path / "extractions" / "shards").iterdir())) == 4
    assert store.row_status(keys[4]) == "absent"


def test_batched_writer_flush_merges_with_keys_indexed_after_it_started(tmp_path):
    store = ExtractionStore(tmp_path)
    mine = _published_key("mine")
    other = _published_key("other")
    with store.batched_writer(flush_every=100) as writer:
        writer.write_rows([_row(mine)])
        store.write_rows([_row(other)])
    assert store.row_status(mine) == store.row_status(other) == "indexed"


def test_a_crash_before_flush_leaves_sealed_shards_that_commit_index_publishes(tmp_path):
    store = ExtractionStore(tmp_path)
    key = _published_key("crash")
    writer = store.batched_writer(flush_every=100)
    writer.write_rows([_row(key)])
    assert store.row_status(key) == "complete_unindexed"
    store.commit_index()
    assert store.row_status(key) == "indexed"


def test_a_row_without_its_final_norm_is_not_stored(tmp_path):
    store = ExtractionStore(tmp_path)
    row = _row(content_digest({"kind": "row", "label": "no-final"}))
    del row["final_normalized"]
    with pytest.raises(ValueError, match="final normalized state"):
        store.write_rows([row])
