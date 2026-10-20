"""Frozen model adapter helpers.

Qwen layer indices are assigned from module names in ``models.qwen``.
This module does not load weights.
"""

from __future__ import annotations


def freeze_model(model: object) -> None:
    """Put ``model`` in evaluation mode and disable gradients.

    Extraction calls this once, after a cache miss has already decided that
    a forward pass is required. Dropout and training mode stay off.
    """
    eval_mode = getattr(model, "eval", None)
    if eval_mode is not None:
        eval_mode()
    parameters = getattr(model, "parameters", None)
    if parameters is not None:
        for parameter in parameters():
            setter = getattr(parameter, "requires_grad_", None)
            if setter is not None:
                setter(False)
            elif getattr(parameter, "requires_grad", False):
                raise RuntimeError("extraction requires gradients to be disabled")
    if getattr(model, "training", False):
        raise RuntimeError("extraction requires evaluation mode")
