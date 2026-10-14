"""Dispatch named stages.

This module parses a stage and its identifiers, calls a registered handler,
and prints domain errors. It does not normalize data, score answers, fit
probes, or load models.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from knowledge_recovery.errors import KnowledgeRecoveryError

CONFIG_STAGES = (
    "validate",
    "prepare",
    "audit-tokens",
    "extract",
    "diagnose",
)
VIEW_STAGES = ("build-view", "train")
RUN_STAGES = ("evaluate", "report")
DECISION_STAGE = "record-filter-decision"

StageHandler = Callable[["StageRequest"], None]
# Later pull requests register stage implementations here.
STAGE_HANDLERS: dict[str, StageHandler] = {}


@dataclass(frozen=True)
class StageRequest:
    """Identifiers copied from the command line, with no resolved artifacts."""

    stage: str
    config: str | None = None
    report: str | None = None
    basis: str | None = None
    run: str | None = None


def main(
    argv: Sequence[str] | None = None,
    *,
    handlers: Mapping[str, StageHandler] | None = None,
) -> int:
    """Parse ``argv`` and dispatch. Return a process exit code."""
    try:
        request = _request(build_parser().parse_args(list(argv) if argv is not None else None))
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    handler = (STAGE_HANDLERS if handlers is None else handlers).get(request.stage)
    if handler is None:
        print(
            f"kr: stage {request.stage!r} is not registered.\n"
            "This command only dispatches stages and does not implement them.",
            file=sys.stderr,
        )
        return 1
    try:
        handler(request)
    except KnowledgeRecoveryError as exc:
        print(exc, file=sys.stderr)
        return exc.exit_code
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kr")
    stages = parser.add_subparsers(dest="stage", required=True)

    def add_config(stage: str) -> None:
        stages.add_parser(stage).add_argument("--config", required=True)

    for stage in CONFIG_STAGES:
        add_config(stage)
    decision = stages.add_parser(DECISION_STAGE)
    decision.add_argument("--report", required=True)
    decision.add_argument("--basis", required=True)
    for stage in VIEW_STAGES:
        add_config(stage)
    for stage in RUN_STAGES:
        stages.add_parser(stage).add_argument("--run", required=True)
    return parser


def _request(namespace: argparse.Namespace) -> StageRequest:
    return StageRequest(
        stage=namespace.stage,
        config=getattr(namespace, "config", None),
        report=getattr(namespace, "report", None),
        basis=getattr(namespace, "basis", None),
        run=getattr(namespace, "run", None),
    )
