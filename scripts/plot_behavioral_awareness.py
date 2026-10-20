"""Plot per-layer behavioral-awareness metrics and log them to wandb."""

from __future__ import annotations

from collections.abc import Sequence

from knowledge_recovery.evaluation.layer_plots import main_for_task
from knowledge_recovery.tasks.base import AWARENESS_TASK_ID


def main(argv: Sequence[str] | None = None) -> int:
    return main_for_task(
        AWARENESS_TASK_ID,
        default_output="plots/behavioral_awareness",
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
