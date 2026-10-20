"""Plot per-layer knowledge-consistency metrics and log them to wandb."""

from __future__ import annotations

from collections.abc import Sequence

from knowledge_recovery.evaluation.layer_plots import main_for_task
from knowledge_recovery.tasks.base import CONSISTENCY_TASK_ID


def main(argv: Sequence[str] | None = None) -> int:
    return main_for_task(
        CONSISTENCY_TASK_ID,
        default_output="plots/knowledge_consistency",
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
