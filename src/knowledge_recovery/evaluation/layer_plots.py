"""Per-layer probe figures and the wandb log of those figures.

Each metric gets one pooled figure, with ID and OOD (when present) side by side, and one
figure per variant. Baselines are grey dotted horizontal lines. A solid
vertical marker is the layer chosen on validation cross-entropy.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from knowledge_recovery.errors import KnowledgeRecoveryError
from knowledge_recovery.evaluation.layer_curves import (
    CONTEXTUAL_ANSWER_BASELINE,
    TRAINING_FREQUENCY_BASELINE,
    VARIANT_AGREEMENT_BASELINE,
    DomainCurves,
    TaskCurves,
    load_task_curves,
)

METRIC_NAMES = ("ece", "brier", "accuracy", "auc")
_METRIC_LABELS = {"ece": "ECE", "brier": "Brier", "accuracy": "Accuracy", "auc": "AUC"}
_BASELINE_STYLE = {
    TRAINING_FREQUENCY_BASELINE: ("training frequency", "0.65"),
    VARIANT_AGREEMENT_BASELINE: ("variant agreement rate", "0.40"),
    CONTEXTUAL_ANSWER_BASELINE: ("contextual answer", "0.55"),
}
_pyplot_module = None


def pooled_figure(curves: TaskCurves, metric: str):
    """One panel per domain present, ID then OOD, with that domain's rows pooled."""
    chosen = _metric(metric)
    pyplot = _pyplot()
    domains = [domain for domain in ("id", "ood") if domain in curves.domains]
    figure, axes = pyplot.subplots(1, len(domains), figsize=(5 * len(domains), 4.2), sharey=True, squeeze=False)
    for axis, domain in zip(axes[0], domains, strict=True):
        _draw(axis, curves.domains[domain], chosen, variant=None, selected_layer=curves.selected_layer)
        axis.set_title(domain.upper())
    figure.suptitle(f"{curves.task_id} {_METRIC_LABELS[chosen]}")
    figure.tight_layout()
    return figure


def variant_figure(curves: TaskCurves, metric: str, domain: str, variant: str):
    """One panel for a single variant. That variant belongs to one domain."""
    chosen = _metric(metric)
    domain_curves = curves.domains[domain]
    if not domain_curves.layers or variant not in domain_curves.layers[0].by_variant:
        raise ValueError(f"variant {variant} is not in the {domain} curves")
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(6, 4.2))
    _draw(axis, domain_curves, chosen, variant=variant, selected_layer=curves.selected_layer)
    axis.set_title(f"{curves.task_id} {domain} {variant}")
    figure.tight_layout()
    return figure


def write_figures(curves: TaskCurves, output: Path | str) -> dict[str, Path]:
    """Write one PNG per pooled metric and one PNG per variant metric."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    pyplot = _pyplot()
    written: dict[str, Path] = {}
    for metric in METRIC_NAMES:
        pooled_key = f"{metric}/pooled"
        pooled_path = directory / f"{metric}__pooled.png"
        figure = pooled_figure(curves, metric)
        figure.savefig(pooled_path, dpi=150)
        pyplot.close(figure)
        written[pooled_key] = pooled_path
        for domain, variant in _variants(curves):
            key = f"{metric}/{domain}/{variant}"
            path = directory / f"{metric}__{domain}__{variant}.png"
            figure = variant_figure(curves, metric, domain, variant)
            figure.savefig(path, dpi=150)
            pyplot.close(figure)
            written[key] = path
    return written


def log_wandb(
    paths: Mapping[str, Path],
    *,
    project: str,
    entity: str | None,
    config: Mapping[str, object],
) -> None:
    """Log saved figures to one wandb run. This opens a network connection."""
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb logging requires the plots extra: uv sync --extra plots") from exc
    with wandb.init(project=project, entity=entity, job_type="layer_curves", config=dict(config)) as run:
        run.log({key: wandb.Image(str(path)) for key, path in paths.items()})


def main_for_task(task_id: str, *, default_output: str, argv: Sequence[str] | None = None) -> int:
    """Score one task, write its figures, and log them. Return a process exit code."""
    parser = argparse.ArgumentParser(
        description=f"Plot per-layer {task_id} probe metrics and log them to wandb.",
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--config")
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--output", default=default_output)
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    try:
        curves = load_task_curves(run=args.run, config=args.config, task_id=task_id)
        paths = write_figures(curves, Path(args.output))
        log_wandb(
            paths,
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "task_id": curves.task_id,
                "selected_layer": curves.selected_layer,
                "run": args.run,
                "output": args.output,
            },
        )
    except KnowledgeRecoveryError as exc:
        print(exc, file=sys.stderr)
        return exc.exit_code
    return 0


def _draw(
    axis,
    domain: DomainCurves,
    metric: str,
    *,
    variant: str | None,
    selected_layer: int,
) -> None:
    if variant is None:
        values = [getattr(point.pooled, metric) for point in domain.layers]
        baselines = domain.baselines
    else:
        values = [getattr(point.by_variant[variant], metric) for point in domain.layers]
        baselines = domain.variant_baselines[variant]
    layers = [point.layer for point in domain.layers]
    axis.plot(layers, _series(values), color="C0", linestyle="-", marker="o", label="linear probe")
    for name, metrics in baselines.items():
        value = getattr(metrics, metric)
        if value is None:
            continue
        label, color = _BASELINE_STYLE[name]
        axis.axhline(value, color=color, linestyle=":", linewidth=1.4, label=label)
    if layers:
        axis.axvline(selected_layer, color="0.35", linestyle="-", linewidth=0.8, label="selected layer")
        axis.set_xticks(layers)
        axis.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("layer")
    axis.set_ylabel(_METRIC_LABELS[metric])
    axis.legend(fontsize=8)


def _variants(curves: TaskCurves) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    for domain in ("id", "ood"):
        if domain not in curves.domains:
            continue
        layers = curves.domains[domain].layers
        if not layers:
            continue
        found.extend((domain, variant) for variant in layers[0].by_variant)
    return tuple(found)


def _series(values: Sequence[float | None]) -> list[float]:
    return [float("nan") if value is None else float(value) for value in values]


def _metric(metric: str) -> str:
    if metric not in _METRIC_LABELS:
        raise ValueError("metric must be ece, brier, accuracy, or auc")
    return metric


def _pyplot():
    global _pyplot_module
    if _pyplot_module is None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("plotting requires the plots extra: uv sync --extra plots") from exc
        _pyplot_module = plt
    return _pyplot_module
