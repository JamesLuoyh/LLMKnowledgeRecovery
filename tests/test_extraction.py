import inspect
import shutil
import socket

import numpy
import pytest

from knowledge_recovery.cache.store import ExtractionStore
from knowledge_recovery.errors import CacheIncompatible, CacheIncomplete, KnowledgeRecoveryError, MissingLocalAsset
from knowledge_recovery.extraction.positions import last_nonpadding_index
from knowledge_recovery.extraction.runner import BatchExceededMemory, run_extraction
from knowledge_recovery.models.assets import resolve_execution_precision, resolve_local_assets
from knowledge_recovery.models.base import freeze_model
from knowledge_recovery.models.qwen import (
    EMBEDDING_MODULE,
    FINAL_NORM_MODULE,
    MockQwen,
    capture_selected,
    layer_index,
    layer_module_name,
    refuse_hidden_state_tuple,
)
from knowledge_recovery.provenance import content_digest


def _key(label: str) -> str:
    return content_digest({"kind": "row", "label": label})


def _produced(scale: float = 1.0) -> dict:
    return {
        "residuals": numpy.full((2, 3), scale, dtype=numpy.float32),
        "final_normalized": numpy.full(3, scale + 4, dtype=numpy.float32),
        "logits": numpy.arange(5, dtype=numpy.float32) * scale,
    }


def test_layer_indices_start_at_one_and_exclude_embedding_and_final_norm():
    assert layer_module_name(0) == "model.layers.0"
    assert layer_index("model.layers.0") == 1
    assert layer_index("model.layers.2") == 3
    with pytest.raises(ValueError, match="not layer 1"):
        layer_index(EMBEDDING_MODULE)
    with pytest.raises(ValueError, match="not a residual"):
        layer_index(FINAL_NORM_MODULE)
    with pytest.raises(ValueError, match="hidden-state tuple"):
        refuse_hidden_state_tuple((0, 1, 2))


def test_hooks_select_the_last_real_token_and_match_forward_logits():
    model = MockQwen(n_layers=2, hidden=4, vocab=6, precision="bf16")
    freeze_model(model)
    ids = [[4, 5, 0], [7, 8, 9]]
    mask = [[1, 1, 0], [1, 1, 1]]
    captured = capture_selected(model, ids, mask)
    assert captured.layer_indices == (1, 2)
    direct = model.forward(ids, mask)
    assert last_nonpadding_index(mask[0]) == 1
    assert numpy.allclose(captured.logits[0], direct[0, 1])
    assert not numpy.allclose(captured.logits[0], direct[0, 2])
    assert captured.embedding.shape == (2, 4)
    assert not numpy.allclose(captured.residuals[0, 0], captured.embedding[0])
    assert not numpy.array_equal(captured.final_normalized[0], captured.residuals[0, -1])
    for row in range(2):
        length = 2 if row == 0 else 3
        single = capture_selected(model, [ids[row][:length]], [mask[row][:length]])
        assert numpy.allclose(captured.residuals[row], single.residuals[0])
        assert numpy.allclose(captured.logits[row], single.logits[0])
        assert numpy.allclose(captured.final_normalized[row], single.final_normalized[0])


def test_a_block_tuple_contributes_its_first_element_only():
    class Site:
        def __init__(self) -> None:
            self.hooks = []

        def register_forward_hook(self, hook):
            self.hooks.append(hook)
            return lambda: self.hooks.remove(hook)

        def emit(self, output):
            for hook in list(self.hooks):
                hook(self, None, output)

    class Tiny:
        def __init__(self) -> None:
            self.model = type("Inner", (), {})()
            self.model.embed_tokens = Site()
            self.model.layers = [Site()]
            self.model.norm = Site()
            self.lm_head = Site()

        def forward(self, _ids, _mask):
            embed = numpy.zeros((1, 2, 2), dtype=numpy.float32)
            embed[0, :, 0] = [3, 9]
            self.model.embed_tokens.emit(embed)
            hidden = embed + 1
            self.model.layers[0].emit((hidden, "not-a-layer-count"))
            norm = hidden + 7
            self.model.norm.emit(norm)
            logits = numpy.zeros((1, 2, 3), dtype=numpy.float32)
            logits[0, :, 0] = [4, 8]
            self.lm_head.emit(logits)
            return logits

    captured = capture_selected(Tiny(), [[1, 1]], [[1, 1]])
    assert captured.layer_indices == (1,)
    # Mask [1, 1] selects the last real token. The tuple's first element is
    # hidden = embed + 1, so that position is 10; the extra string is ignored.
    assert captured.residuals[0, 0, 0] == 10
    assert captured.final_normalized[0, 0] == 17
    assert captured.logits[0, 0] == 8


def test_storage_dtypes_and_separate_final_norm(tmp_path):
    store = ExtractionStore(tmp_path)
    key = _key("stored")
    references = store.write_rows([{"row_key": key, **_produced()}])
    residual_file = tmp_path / "extractions" / "shards" / references[0].shard_id / "residual.bf16"
    logits_file = tmp_path / "extractions" / "shards" / references[0].shard_id / "logits.f32"
    assert numpy.fromfile(residual_file, dtype="<u2").size > 0
    assert numpy.fromfile(logits_file, dtype="<f4").dtype == numpy.float32
    final = store.read_final_norm_row(references[0])
    residual = store.read_residual_row(references[0])
    assert final is not None
    assert not numpy.array_equal(final, residual[-1])
    assert store.read_logits_row(references[0]).dtype == numpy.float32


def test_out_of_memory_retries_lower_only_the_batch_size(tmp_path):
    store = ExtractionStore(tmp_path)
    keys = [_key(f"row-{index}") for index in range(3)]
    calls = []
    factories = []

    def factory():
        factories.append("model")
        model = MockQwen(n_layers=1, hidden=1, vocab=2, precision="bf16")
        return model

    def forward(_model, rows, *, batch_size, precision):
        calls.append((batch_size, precision))
        if batch_size > 1:
            raise BatchExceededMemory()
        return [_produced() for _ in rows]

    result = run_extraction(
        [{"row_key": key} for key in keys],
        store=store,
        model_factory=factory,
        forward_batch=forward,
        initial_batch_size=4,
        precision="bf16",
    )
    assert factories == ["model"]
    assert calls == [(4, "bf16"), (2, "bf16"), (1, "bf16"), (1, "bf16"), (1, "bf16")]
    assert result.precision == "bf16"
    assert result.constructed_model is True
    assert result.batch_sizes == (1, 1, 1)


def test_cache_hit_and_resume_do_not_construct_a_model(tmp_path):
    store = ExtractionStore(tmp_path)
    key = _key("cached")

    def boom():
        raise AssertionError("model construction is forbidden")

    store.write_rows([{"row_key": key, **_produced()}])
    hit = run_extraction(
        [{"row_key": key}],
        store=store,
        model_factory=boom,
        forward_batch=boom,
        initial_batch_size=2,
        precision="bf16",
    )
    assert hit.constructed_model is False
    assert hit.references[0].extraction_row_key == key

    other = _key("unindexed")
    staged = store.stage_rows([{"row_key": other, **_produced(2)}])
    store.complete_shard(staged)
    resumed = run_extraction(
        [{"row_key": other}],
        store=store,
        model_factory=boom,
        forward_batch=boom,
        initial_batch_size=2,
        precision="bf16",
    )
    assert resumed.constructed_model is False
    assert store.row_status(other) == "indexed"


def test_an_incomplete_shard_is_not_treated_as_absent(tmp_path):
    store = ExtractionStore(tmp_path)
    key = _key("partial")
    store.stage_rows([{"row_key": key, **_produced()}])

    def boom():
        raise AssertionError("model construction is forbidden")

    with pytest.raises(CacheIncompatible, match="incomplete"):
        run_extraction(
            [{"row_key": key}],
            store=store,
            model_factory=boom,
            forward_batch=boom,
            initial_batch_size=1,
            precision="bf16",
        )


def test_probe_only_denies_model_construction_and_network(tmp_path):
    store = ExtractionStore(tmp_path)
    seen = {}

    def factory():
        seen["model"] = True

    def hook():
        try:
            socket.create_connection(("192.0.2.1", 9), timeout=0.01)
        except KnowledgeRecoveryError as exc:
            seen["network"] = "network" in str(exc).lower()

    with pytest.raises(CacheIncomplete):
        run_extraction(
            [{"row_key": _key("missing")}],
            store=store,
            model_factory=factory,
            forward_batch=None,
            initial_batch_size=1,
            precision="bf16",
            probe_only=True,
            probe_only_hook=hook,
        )
    assert seen == {"network": True}
    assert "probe_seed" not in inspect.signature(run_extraction).parameters
    assert "training_seed" not in inspect.signature(run_extraction).parameters


def test_identical_bytes_keep_their_digest_when_the_directory_moves(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    (source / "weights.bin").write_bytes(b"weights")
    (source / "tokenizer_map.json").write_text("{}\n", encoding="utf-8")
    moved = tmp_path / "elsewhere"
    shutil.copytree(source, moved)
    assert resolve_local_assets(source).weight_content_digest == resolve_local_assets(moved).weight_content_digest
    missing = tmp_path / "absent"
    with pytest.raises(MissingLocalAsset, match="not downloaded") as raised:
        resolve_local_assets(missing)
    assert raised.value.expected_location == str(missing)


def test_precision_has_no_silent_fallback():
    with pytest.raises(ValueError, match="no silent fallback"):
        resolve_execution_precision("bf16", bf16_supported=False, fp16_supported=True)
    assert resolve_execution_precision("bf16", bf16_supported=True, fp16_supported=False) == "bf16"
    assert resolve_execution_precision("fp32", bf16_supported=False, fp16_supported=False) == "fp32"


def test_missing_rows_are_computed_shortest_first(tmp_path):
    store = ExtractionStore(tmp_path)
    lengths = [5, 2, 9, 2, 4]
    rows = [{"row_key": _key(f"row-{index}"), "input_ids": [1] * length} for index, length in enumerate(lengths)]
    seen = []

    def forward(_model, chunk, *, batch_size, precision):
        seen.append([len(row["input_ids"]) for row in chunk])
        return [_produced() for _ in chunk]

    result = run_extraction(
        rows,
        store=store,
        model_factory=lambda: MockQwen(n_layers=1, hidden=1, vocab=1, precision="bf16"),
        forward_batch=forward,
        initial_batch_size=2,
        precision="bf16",
    )
    assert seen == [[2, 2], [4, 5], [9]]
    # References still follow the caller's row order.
    assert [item.extraction_row_key for item in result.references] == [row["row_key"] for row in rows]


def test_a_device_out_of_memory_error_asks_for_a_smaller_batch(monkeypatch):
    from knowledge_recovery.models import qwen

    class OutOfMemoryError(RuntimeError):
        pass

    def exhausted(*_args, **_kwargs):
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(qwen, "capture_selected", exhausted)
    model = MockQwen(n_layers=1, hidden=2, vocab=3, precision="bf16")
    row = {"input_ids": [1, 2], "attention_mask": [1, 1]}
    with pytest.raises(BatchExceededMemory):
        qwen.forward_padded(model, [row], batch_size=1, precision="bf16")

    def broken(*_args, **_kwargs):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(qwen, "capture_selected", broken)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        qwen.forward_padded(model, [row], batch_size=1, precision="bf16")


def test_a_selected_forward_computes_logits_at_the_selected_token_only():
    model = MockQwen(n_layers=2, hidden=4, vocab=6, precision="bf16")
    freeze_model(model)
    ids = [[4, 5, 0], [7, 8, 9]]
    mask = [[1, 1, 0], [1, 1, 1]]
    full = capture_selected(model, ids, mask)
    seen = {}

    def forward_selected(input_ids, attention_mask, positions):
        seen["positions"] = list(positions)
        logits = model.forward(input_ids, attention_mask)
        model.lm_head.emit(logits[numpy.arange(2), positions])

    model.forward_selected = forward_selected
    selected = capture_selected(model, ids, mask)
    assert seen["positions"] == [1, 2]
    # The last lm_head call is the [batch, vocab] one and is taken as already selected.
    assert numpy.array_equal(selected.logits, full.logits)
    assert numpy.array_equal(selected.residuals, full.residuals)


def test_the_logits_scope_enters_the_row_key():
    from knowledge_recovery.extraction.identity import build_row_key

    common = {
        "model_signature": _key("model"),
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "prompt_contract_digest": _key("contract"),
        "selected_position": 2,
        "execution_precision": "bf16",
    }
    full = build_row_key(**common)
    assert build_row_key(**common, logits_scope="full_vocabulary") == full
    assert build_row_key(**common, logits_scope="ab_summary") != full
    with pytest.raises(ValueError, match="logits_scope must be one of"):
        build_row_key(**common, logits_scope="top_k")
