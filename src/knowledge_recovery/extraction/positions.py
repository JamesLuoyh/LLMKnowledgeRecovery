"""Last non-padding input position.

Right padding makes the final tensor column a pad. The selected index is the
last mask value that is 1, never ``seq_len - 1`` by itself.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy


def last_nonpadding_index(attention_mask: Sequence[int]) -> int:
    """Return the last real token. An all-pad mask is an error."""
    if isinstance(attention_mask, (str, bytes)) or not isinstance(attention_mask, Sequence):
        raise TypeError("attention_mask must be a sequence of 0 and 1")
    last = None
    for index, bit in enumerate(attention_mask):
        if type(bit) is not int or bit not in (0, 1):
            raise ValueError("attention_mask values must be 0 or 1")
        if bit == 1:
            last = index
    if last is None:
        raise ValueError("attention mask has no real token")
    return last


def select_last_nonpadding(states: numpy.ndarray, attention_mask: numpy.ndarray) -> numpy.ndarray:
    """Gather ``states[:, last_real, ...]`` for each row.

    ``states`` is ``[batch, seq, ...]`` and ``attention_mask`` is ``[batch, seq]``.
    """
    array = numpy.asarray(states)
    mask = numpy.asarray(attention_mask)
    if array.ndim < 2 or mask.ndim != 2:
        raise ValueError("states must be [batch, seq, ...] and the mask [batch, seq]")
    if array.shape[0] != mask.shape[0] or array.shape[1] != mask.shape[1]:
        raise ValueError("attention mask does not match the sequence")
    chosen = [array[row, last_nonpadding_index(mask[row].tolist())] for row in range(array.shape[0])]
    return numpy.stack(chosen)
