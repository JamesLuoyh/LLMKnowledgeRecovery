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

StageHandler = Callable[["StageRequest"], None]
# Later pull requests register stage implementations here.
STAGE_HANDLERS: dict[str, StageHandler] = {}


@dataclass(frozen=True)
class StageRequest:
    """Identifiers copied from the command line, with no resolved artifacts."""

    stage: str
    config: str | None = None
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
    handler = _handlers(handlers).get(request.stage)
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
    for stage in VIEW_STAGES:
        add_config(stage)
    for stage in RUN_STAGES:
        run = stages.add_parser(stage)
        run.add_argument("--run", required=True)
        run.add_argument("--config")
    return parser


def _handlers(handlers: Mapping[str, StageHandler] | None) -> Mapping[str, StageHandler]:
    """Use an injected table, else the empty registry, else the stage graph.

    The graph is imported by name so this module's source does not import
    dataset, model, or fitting code. ``STAGE_HANDLERS`` stays empty.
    """
    if handlers is not None:
        return handlers
    if STAGE_HANDLERS:
        return STAGE_HANDLERS
    module = __import__("knowledge_recovery.pipeline", fromlist=["STAGE_HANDLERS"])
    return module.STAGE_HANDLERS


def _request(namespace: argparse.Namespace) -> StageRequest:
    return StageRequest(
        stage=namespace.stage,
        config=getattr(namespace, "config", None),
        run=getattr(namespace, "run", None),
    )
