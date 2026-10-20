"""Initialization and shuffle streams.

The seed for one layer is the first eight bytes of SHA-256(canonical JSON),
read as an unsigned big-endian integer and reduced modulo ``2**63``.
Initialization draws from the PyTorch CPU mt19937 uniform transform.
Shuffling draws from NumPy's PCG64 generator. A layer's stream does not
depend on which other layer was fitted first.
"""

from __future__ import annotations

import hashlib

import numpy

from knowledge_recovery.provenance import canonical_json

_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UMASK = 0x80000000
_LMASK = 0x7FFFFFFF
_SEED_MODULUS = 2**63


def derived_seed(*, seed: int, task_id: str, method_id: str, layer: int, stream: str) -> int:
    """Return the unsigned 63-bit seed for ``init`` or ``shuffle``."""
    if type(seed) is not int:
        raise ValueError("training seed must be an integer")
    if type(layer) is not int or layer < 1:
        raise ValueError("layer indices start at 1")
    if stream not in ("init", "shuffle"):
        raise ValueError("stream must be init or shuffle")
    if type(task_id) is not str or task_id == "" or type(method_id) is not str or method_id == "":
        raise ValueError("task_id and method_id must be non-empty strings")
    digest = hashlib.sha256(canonical_json({
        "seed": seed,
        "task_id": task_id,
        "method_id": method_id,
        "layer": layer,
        "stream": stream,
    }).encode()).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


def pcg64_generator(seed: int) -> numpy.random.Generator:
    """NumPy generator used to shuffle training rows."""
    return numpy.random.Generator(numpy.random.PCG64(seed))


class CpuTensorGenerator:
    """CPU mt19937 uniform generator matching PyTorch's float32 ``uniform_``."""

    def __init__(self, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("CPU generator seed must be a non-negative integer")
        self._state = _init_state(seed)
        self._left = 1
        self._next = 0

    def random_uint32(self) -> int:
        """Return the next tempered 32-bit integer."""
        self._left -= 1
        if self._left == 0:
            _twist(self._state)
            self._left = _N
            self._next = 0
        value = _temper(self._state[self._next])
        self._next += 1
        return value

    def uniform(self, count: int, low: float, high: float) -> numpy.ndarray:
        """Draw ``count`` float32 values from ``[low, high)``."""
        if type(count) is not int or count < 1:
            raise ValueError("uniform count must be a positive integer")
        draws = numpy.empty(count, dtype=numpy.float32)
        for index in range(count):
            draws[index] = _uniform_float(self.random_uint32(), low, high)
        return draws


def _init_state(seed: int) -> list[int]:
    state = [0] * _N
    state[0] = seed & 0xFFFFFFFF
    for index in range(1, _N):
        previous = state[index - 1]
        state[index] = (1812433253 * (previous ^ (previous >> 30)) + index) & 0xFFFFFFFF
    return state


def _twist(state: list[int]) -> None:
    for index in range(_N - _M):
        state[index] = (state[index + _M] ^ _mix(state[index], state[index + 1])) & 0xFFFFFFFF
    for index in range(_N - _M, _N - 1):
        state[index] = (state[index + (_M - _N)] ^ _mix(state[index], state[index + 1])) & 0xFFFFFFFF
    state[_N - 1] = (state[_M - 1] ^ _mix(state[_N - 1], state[0])) & 0xFFFFFFFF


def _mix(left: int, right: int) -> int:
    mixed = (left & _UMASK) | (right & _LMASK)
    return (mixed >> 1) ^ (_MATRIX_A if right & 1 else 0)


def _temper(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 11
    value ^= (value << 7) & 0x9D2C5680
    value ^= (value << 15) & 0xEFC60000
    value ^= value >> 18
    return value & 0xFFFFFFFF


def _uniform_float(value: int, low: float, high: float) -> numpy.float32:
    unit = (value & ((1 << 24) - 1)) * (1.0 / (1 << 24))
    return numpy.float32(unit * (high - low) + low)
