"""Qwen2 hook map.

Layer indices start at one and follow module names, not the length of a
generic hidden-state tuple:

- ``model.embed_tokens`` is the embedding. It is not layer one.
- ``model.layers.{i}`` is transformer block ``i``. Its output is the
  post-block residual, and the stored layer index is ``i + 1``.
- ``model.norm`` is the final RMSNorm. It is stored separately and is not
  a substitute for the last block's residual.
- ``lm_head`` produces the full-vocabulary logits.

A decoder block that returns a tuple contributes its first element. The
tuple length is not a layer count.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy

from knowledge_recovery.extraction.positions import select_last_nonpadding
from knowledge_recovery.models.assets import activate_offline, resolve_execution_precision
from knowledge_recovery.prompts.token_audit import PromptExample

MODEL_ADAPTER_VERSION = "qwen-named-hooks-v1"
EXTRACTION_ADAPTER_VERSION = "qwen-named-hooks-v1"
EMBEDDING_MODULE = "model.embed_tokens"
FINAL_NORM_MODULE = "model.norm"
LM_HEAD_MODULE = "lm_head"
HOOK_DEFINITIONS = (
    {"module": EMBEDDING_MODULE, "role": "embedding", "stored_as_layer": False},
    {"module": "model.layers.{i}", "role": "post_block_pre_final_norm", "layer_index": "i+1"},
    {"module": FINAL_NORM_MODULE, "role": "final_normalized_state", "stored_as_layer": False},
    {"module": LM_HEAD_MODULE, "role": "full_vocabulary_logits", "stored_as_layer": False},
)


def layer_module_name(zero_based_index: int) -> str:
    """Module name of transformer block ``zero_based_index``."""
    if type(zero_based_index) is not int or zero_based_index < 0:
        raise ValueError("a transformer block index is a non-negative integer")
    return f"model.layers.{zero_based_index}"


def layer_index(module_name: str) -> int:
    """Map ``model.layers.{i}`` to layer ``i + 1``. Embedding and norm are not layers."""
    prefix = "model.layers."
    if not module_name.startswith(prefix) or module_name == prefix:
        raise ValueError(
            f"{module_name} is not a transformer block. "
            "model.embed_tokens is not layer 1 and model.norm is not a residual layer."
        )
    suffix = module_name[len(prefix):]
    if not suffix.isdigit():
        raise ValueError(f"{module_name} is not a transformer block")
    return int(suffix) + 1


def refuse_hidden_state_tuple(hidden_states: Sequence[Any]) -> None:
    """Refuse to invent layer meanings from how many tensors a forward returned."""
    raise ValueError(
        "Qwen residuals are read from named modules model.layers.{i}, not from a hidden-state tuple. "
        f"A tuple of length {len(hidden_states)} was refused. "
        "Embedding output is not layer 1, and model.norm is not the final block residual."
    )


class _HookSite:
    """A module stand-in that records forward hooks."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._hooks: list = []

    def register_forward_hook(self, hook):
        self._hooks.append(hook)

        def remove() -> None:
            self._hooks.remove(hook)

        return remove

    def emit(self, output: Any) -> Any:
        for hook in list(self._hooks):
            hook(self, None, output)
        return output


class _Parameter:
    def __init__(self) -> None:
        self.requires_grad = True

    def requires_grad_(self, value: bool) -> _Parameter:
        self.requires_grad = value
        return self


class _Inner:
    def __init__(self, n_layers: int) -> None:
        self.embed_tokens = _HookSite(EMBEDDING_MODULE)
        self.layers = [_HookSite(layer_module_name(index)) for index in range(n_layers)]
        self.norm = _HookSite(FINAL_NORM_MODULE)


class MockQwen:
    """Hook-compatible stand-in used when no Qwen snapshot is loaded.

    ``pending_logits`` replaces the logits at the last real token so a labeled
    fixture can drive scoring. Residuals still come from the named blocks.
    """

    def __init__(self, *, n_layers: int, hidden: int, vocab: int, precision: str) -> None:
        if n_layers < 1 or hidden < 1 or vocab < 1:
            raise ValueError("the mock Qwen needs at least one layer, hidden unit, and vocab row")
        self.n_layers = n_layers
        self.hidden = hidden
        self.vocab = vocab
        self.execution_precision = precision
        self.training = True
        self.model = _Inner(n_layers)
        self.lm_head = _HookSite(LM_HEAD_MODULE)
        self._parameters = [_Parameter()]
        self.pending_logits: list[numpy.ndarray] | None = None

    def eval(self) -> MockQwen:
        self.training = False
        return self

    def parameters(self):
        return iter(self._parameters)

    def forward(self, input_ids: Any, attention_mask: Any) -> numpy.ndarray:
        if self.training:
            raise RuntimeError("dropout and training mode are disabled for extraction")
        ids = numpy.asarray(input_ids, dtype=numpy.int64)
        mask = numpy.asarray(attention_mask, dtype=numpy.int64)
        if ids.ndim != 2 or mask.shape != ids.shape:
            raise ValueError("input ids and the attention mask are [batch, seq]")
        batch, seq = ids.shape
        position = numpy.arange(self.hidden, dtype=numpy.float32) + 1
        embed = numpy.zeros((batch, seq, self.hidden), dtype=numpy.float32)
        for row in range(batch):
            for column in range(seq):
                embed[row, column] = (int(ids[row, column]) + 1) * position
                if int(mask[row, column]) == 0:
                    embed[row, column] = -1
        self.model.embed_tokens.emit(embed)
        hidden = embed
        for index, layer in enumerate(self.model.layers):
            residual = hidden.copy()
            real = mask == 1
            residual[real] = hidden[real] + (index + 1)
            hidden = layer.emit(residual)
        normalized = numpy.where(mask[..., None] == 1, hidden + 1000, hidden)
        self.model.norm.emit(normalized)
        logits = numpy.full((batch, seq, self.vocab), -10.0, dtype=numpy.float32)
        for row in range(batch):
            for column in range(seq):
                logits[row, column, int(ids[row, column]) % self.vocab] = 4
            if self.pending_logits is not None:
                last = int(numpy.flatnonzero(mask[row] == 1)[-1])
                logits[row, last] = numpy.asarray(self.pending_logits[row], dtype=numpy.float32)
        self.lm_head.emit(logits)
        return logits


@dataclass(frozen=True)
class CapturedForward:
    """Selected-token states. Layer 0 of ``residuals`` is layer index 1."""

    residuals: numpy.ndarray
    final_normalized: numpy.ndarray
    logits: numpy.ndarray
    embedding: numpy.ndarray
    layer_indices: tuple[int, ...]

    def store_rows(self) -> tuple[dict[str, numpy.ndarray], ...]:
        rows = []
        for index in range(self.residuals.shape[0]):
            rows.append({
                "residuals": numpy.array(self.residuals[index], dtype=numpy.float32),
                "final_normalized": numpy.array(self.final_normalized[index], dtype=numpy.float32),
                "logits": numpy.array(self.logits[index], dtype=numpy.float32),
            })
        return tuple(rows)


class _Capture:
    def __init__(self) -> None:
        self.embedding = None
        self.residuals: dict[int, numpy.ndarray] = {}
        self.final_normalized = None
        self.logits = None

    def note_embedding(self, output: Any) -> None:
        self.embedding = _hidden(output)

    def note_residual(self, layer: int, output: Any) -> None:
        self.residuals[layer] = _hidden(output)

    def note_norm(self, output: Any) -> None:
        self.final_normalized = _hidden(output)

    def note_logits(self, output: Any) -> None:
        self.logits = _hidden(output)


def capture_selected(module: Any, input_ids: Any, attention_mask: Any) -> CapturedForward:
    """Run one forward and keep the last real token from each named hook.

    Hooks are removed before this function returns, so a second call does not
    double-count. ``refuse_hidden_state_tuple`` is not used to choose layers.
    """
    capture = _Capture()
    removers = []
    removers.append(module.model.embed_tokens.register_forward_hook(
        lambda _module, _inputs, output: capture.note_embedding(output)
    ))
    for index, layer in enumerate(module.model.layers):
        layer_id = index + 1
        removers.append(layer.register_forward_hook(
            lambda _module, _inputs, output, layer_id=layer_id: capture.note_residual(layer_id, output)
        ))
    removers.append(module.model.norm.register_forward_hook(
        lambda _module, _inputs, output: capture.note_norm(output)
    ))
    removers.append(module.lm_head.register_forward_hook(
        lambda _module, _inputs, output: capture.note_logits(output)
    ))
    try:
        module.forward(input_ids, attention_mask)
    finally:
        for handle in removers:
            _drop_hook(handle)
    if capture.embedding is None or capture.final_normalized is None or capture.logits is None:
        raise RuntimeError("named Qwen hooks did not see embedding, final norm, and logits")
    indices = tuple(sorted(capture.residuals))
    if indices != tuple(range(1, len(module.model.layers) + 1)):
        raise RuntimeError(f"layer indices start at 1; hooks recorded {indices}")
    mask = numpy.asarray(attention_mask)
    residuals = numpy.stack(
        [select_last_nonpadding(capture.residuals[index], mask) for index in indices],
        axis=1,
    )
    return CapturedForward(
        residuals=residuals,
        final_normalized=select_last_nonpadding(capture.final_normalized, mask),
        logits=select_last_nonpadding(capture.logits, mask),
        embedding=numpy.asarray(capture.embedding, dtype=numpy.float32),
        layer_indices=indices,
    )


def forward_padded(model: Any, rows: Sequence[Mapping[str, Any]], *, batch_size: int, precision: str) -> tuple[dict[str, numpy.ndarray], ...]:
    """Pad a batch on the right and select the last real token.

    ``batch_size`` is the attempted execution batch. It is not a scientific input.
    """
    del batch_size
    recorded = getattr(model, "execution_precision", None)
    if recorded != precision:
        raise RuntimeError(
            f"execution precision {precision!r} does not match the loaded precision {recorded!r}. "
            "Batch-size retries must not change it."
        )
    width = max(len(row["input_ids"]) for row in rows)
    ids = []
    mask = []
    for row in rows:
        logical = list(row["input_ids"])
        pad = width - len(logical)
        ids.append(logical + [0] * pad)
        mask.append(list(row["attention_mask"]) + [0] * pad)
    return capture_selected(model, ids, mask).store_rows()


def load_frozen_model(path: Path | str, *, precision: str) -> Any:
    """Load the local snapshot named by ``config.json``. Nothing is downloaded.

    ``model_type`` ``integration_mock`` builds the hook stand-in. A Qwen
    snapshot is loaded with ``local_files_only`` and is not replaced with the
    stand-in.
    """
    activate_offline()
    root = Path(path)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    kind = config.get("model_type")
    if kind == "integration_mock":
        model = MockQwen(
            n_layers=int(config["num_hidden_layers"]),
            hidden=int(config["hidden_size"]),
            vocab=int(config["vocab_size"]),
            precision=precision,
        )
        model.eval()
        return model
    if kind in ("qwen2", "qwen2_5"):
        return _load_qwen_weights(root, precision=precision, config=config)
    raise RuntimeError(
        f"Missing component: a model adapter for model_type {kind!r}. "
        "The integration mock was not substituted."
    )


def load_qwen(path: Path | str, *, precision: str) -> Any:
    """Load local Qwen weights. Callers pass an explicit directory."""
    root = Path(path)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    return _load_qwen_weights(root, precision=precision, config=config)


class QwenAdapter:
    """Torch module exposed under the documented hook names."""

    def __init__(self, inner: Any, *, precision: str, device: str) -> None:
        self.inner = inner
        self.model = inner.model
        self.lm_head = inner.lm_head
        self.execution_precision = precision
        self.device = device
        self.training = False

    def eval(self) -> QwenAdapter:
        self.inner.eval()
        self.training = False
        return self

    def parameters(self):
        return self.inner.parameters()

    def forward(self, input_ids: Any, attention_mask: Any):
        import torch

        ids = torch.tensor(numpy.asarray(input_ids), dtype=torch.long, device=self.device)
        mask = torch.tensor(numpy.asarray(attention_mask), dtype=torch.long, device=self.device)
        with torch.no_grad():
            result = self.inner(input_ids=ids, attention_mask=mask, use_cache=False)
        return result.logits


class QwenContinuationTokenizer:
    """Hugging Face tokenizer used only after a local snapshot is supplied."""

    def __init__(self, path: Path | str) -> None:
        activate_offline()
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing component: the transformers tokenizer library. "
                "The snapshot was not downloaded."
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(Path(path), local_files_only=True)

    def settings(self) -> dict[str, Any]:
        return {
            "kind": "huggingface",
            "add_special_tokens": False,
            "padding_side": getattr(self.tokenizer, "padding_side", "right"),
        }

    def prompt_boundary(self, chat: bool) -> str:
        return "huggingface_apply_chat_template" if chat else "end_of_user_text"

    def prompt_string(self, example: PromptExample) -> str:
        if not example.chat:
            return example.user_text
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": example.system_message},
                {"role": "user", "content": example.user_text},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )

    def encode_prompt(self, example: PromptExample) -> tuple[int, ...]:
        if example.chat:
            encoding = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": example.system_message},
                    {"role": "user", "content": example.user_text},
                ],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
            )
            return tuple(int(item) for item in encoding["input_ids"])
        return tuple(int(item) for item in self.tokenizer.encode(example.user_text, add_special_tokens=False))

    def continuation_ids(self, prompt_ids: Sequence[int], answer: str) -> tuple[int, ...]:
        if not tuple(prompt_ids):
            raise ValueError("a continuation requires the prompt token sequence")
        return tuple(int(item) for item in self.tokenizer.encode(answer, add_special_tokens=False))

    def encode_concatenated(self, prompt_text: str, answer: str) -> tuple[int, ...]:
        return tuple(int(item) for item in self.tokenizer.encode(prompt_text + answer, add_special_tokens=False))

    def decode_token(self, token_id: int) -> bytes:
        return self.tokenizer.decode([token_id]).encode("utf-8")


def numeric_runtime_signature(model_config: Mapping[str, Any], *, precision: str) -> dict[str, Any]:
    """Resolve the backend facts that make cached numerical output compatible."""
    if model_config.get("model_type") == "integration_mock":
        return {
            "backend": "integration_mock",
            "numpy_version": numpy.__version__,
            "device_type": "cpu",
            "device_capability": None,
            "attention_backend": "mock_eager",
            "execution_precision": precision,
        }
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "Missing component: torch and transformers are required to resolve the Qwen runtime."
        ) from exc
    cuda = bool(torch.cuda.is_available())
    capability = list(torch.cuda.get_device_capability()) if cuda else None
    device_name = torch.cuda.get_device_name(0) if cuda else "cpu"
    return {
        "backend": "torch",
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "numpy_version": numpy.__version__,
        "device_type": "cuda" if cuda else "cpu",
        "device_name": device_name,
        "device_capability": capability,
        "attention_backend": "eager",
        "execution_precision": precision,
    }


def _load_qwen_weights(path: Path, *, precision: str, config: dict[str, Any]) -> QwenAdapter:
    activate_offline()
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Missing component: torch and transformers. The snapshot was not downloaded."
        ) from exc
    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    fp16 = bool(torch.cuda.is_available())
    resolve_execution_precision(precision, bf16_supported=bf16, fp16_supported=fp16)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    del config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inner = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
    )
    inner.to(device)
    inner.eval()
    for parameter in inner.parameters():
        parameter.requires_grad_(False)
    return QwenAdapter(inner, precision=precision, device=str(device))


def _drop_hook(handle: Any) -> None:
    remove = getattr(handle, "remove", None)
    if remove is not None:
        remove()
        return
    handle()


def _hidden(output: Any) -> numpy.ndarray:
    """First element of a block tuple, otherwise the array itself."""
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, numpy.ndarray):
        return numpy.asarray(output, dtype=numpy.float32)
    detach = getattr(output, "detach", None)
    if detach is None:
        raise TypeError("a Qwen hook did not return an array")
    tensor = detach()
    if hasattr(tensor, "float"):
        tensor = tensor.float()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    return numpy.asarray(tensor.numpy(), dtype=numpy.float32)
