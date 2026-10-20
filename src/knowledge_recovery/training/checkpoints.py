"""Completed-epoch checkpoint records.

Selection reads these records. An unfinished epoch is not a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpochCheckpoint:
    """Weights after one finished epoch, with that epoch's validation loss."""

    layer: int
    epoch: int
    validation_cross_entropy: float
    weight: tuple[float, ...]
    bias: float
    completed: bool

    def __post_init__(self) -> None:
        if type(self.layer) is not int or self.layer < 1:
            raise ValueError("layer indices start at 1")
        if type(self.epoch) is not int or self.epoch < 1:
            raise ValueError("only completed epochs are numbered from 1")
        if type(self.completed) is not bool:
            raise ValueError("completed must be a boolean")
        if not self.weight or any(type(item) is not float for item in self.weight):
            raise ValueError("weight must be a non-empty sequence of floats")
        if type(self.bias) is not float:
            raise ValueError("bias must be a float")
        if type(self.validation_cross_entropy) is not float:
            raise ValueError("validation cross-entropy must be a float")
