import ast
from pathlib import Path

import pytest

from knowledge_recovery.cli import STAGE_HANDLERS, StageRequest, main
from knowledge_recovery.errors import MissingLocalAsset

ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "src" / "knowledge_recovery" / "cli.py"
FORBIDDEN_IMPORT_PREFIXES = (
    "numpy",
    "torch",
    "transformers",
    "datasets",
    "knowledge_recovery.datasets",
    "knowledge_recovery.records",
    "knowledge_recovery.provenance",
    "knowledge_recovery.prompts",
    "knowledge_recovery.models",
    "knowledge_recovery.probes",
    "knowledge_recovery.training",
    "knowledge_recovery.evaluation",
)


def test_package_script_points_at_the_dispatcher():
    text = (ROOT / "pyproject.toml").read_text()
    assert 'name = "knowledge-recovery"' in text
    assert 'kr = "knowledge_recovery.cli:main"' in text


def test_cli_imports_only_dispatch_dependencies():
    tree = ast.parse(CLI_PATH.read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert imported
    for name in imported:
        assert not name.startswith(FORBIDDEN_IMPORT_PREFIXES)
    assert "knowledge_recovery.errors" in imported


def test_no_stage_handler_is_registered():
    assert STAGE_HANDLERS == {}


def test_registered_handler_receives_unopened_config(tmp_path):
    config = tmp_path / "missing.yaml"
    seen: list[StageRequest] = []
    code = main(
        ["validate", "--config", str(config)],
        handlers={"validate": seen.append},
    )
    assert code == 0
    assert seen == [StageRequest(stage="validate", config=str(config))]
    assert not config.exists()


def test_run_identifier_passes_through():
    seen: list[StageRequest] = []
    assert main(
        ["evaluate", "--run", "run-7"],
        handlers={"evaluate": seen.append},
    ) == 0
    assert seen == [StageRequest(stage="evaluate", run="run-7")]
    assert main(["record-filter-decision", "--report", "report-7", "--basis", "raw"]) != 0


def test_domain_error_prints_its_message_and_exits_nonzero(capsys):
    def handler(_request: StageRequest) -> None:
        raise MissingLocalAsset(kind="dataset", expected_location="/assets/dataset")

    code = main(["diagnose", "--config", "configs/x.yaml"], handlers={"diagnose": handler})
    assert code == MissingLocalAsset.exit_code
    assert code != 0
    assert "Expected location: /assets/dataset" in capsys.readouterr().err


def test_unregistered_and_unknown_stages_exit_nonzero(capsys):
    assert main(["train", "--config", "configs/x.yaml"], handlers={}) != 0
    assert "is not registered" in capsys.readouterr().err
    assert main(["not-a-stage"]) != 0


def test_unexpected_handler_failure_propagates():
    def handler(_request: StageRequest) -> None:
        raise RuntimeError("bug")

    with pytest.raises(RuntimeError, match="bug"):
        main(["report", "--run", "run-7"], handlers={"report": handler})
