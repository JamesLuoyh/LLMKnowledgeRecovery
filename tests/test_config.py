import copy
from pathlib import Path

import pytest
import yaml

from knowledge_recovery.config import (
    INITIAL_INSTRUCTION,
    INITIAL_SYSTEM_MESSAGE,
    LINEAR_V1,
    ConfigurationError,
    load_config,
    validate_before_expensive_work,
)
from knowledge_recovery.errors import AnswerInstructionMismatch, MissingLocalAsset, UnsupportedPositionSelector

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "tests" / "fixtures" / "initial_sycophancy.yaml"


def _document():
    return yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))


def _load(document, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_config(path)


def _shipped():
    return load_config(SHIPPED)


def test_shipped_file_is_three_explicit_contracts():
    config = _shipped()
    extraction = config.extraction
    view = config.analysis_view
    probe = config.probe_run
    assert (extraction.schema_version, view.schema_version, probe.schema_version) == (1, 1, 1)
    assert extraction.chat is True
    assert extraction.system_message == INITIAL_SYSTEM_MESSAGE
    assert extraction.instruction == INITIAL_INSTRUCTION
    assert not extraction.system_message.endswith("\n")
    assert extraction.precision == "bf16"
    assert extraction.allow_download is False and extraction.offline is True
    assert extraction.token_audit is None
    assert extraction.clean_question_path == "base.question"
    assert extraction.correct_answer_path == "base.correct_answer"
    assert extraction.incorrect_answer_path == "base.incorrect_answer"
    assert extraction.source_filter_value == "{question}"
    assert [item.id for item in extraction.interventions] == ["clean", "T-C", "NT-C", "T-IC", "NT-IC"]
    assert [item.preset for item in extraction.interventions] == [
        None,
        "sycophancy_T_C",
        "sycophancy_NT_C",
        "sycophancy_T_IC",
        "sycophancy_NT_IC",
    ]
    assert view.min_correct_prob == 0.8
    assert not hasattr(view, "probability_basis")
    assert not hasattr(view, "decision_record")
    assert view.train_variants == ("clean", "T-C", "T-IC")
    assert view.eval_variants == ("NT-C", "NT-IC")
    assert view.split_seed == 42 and probe.training_seed == 42
    assert not hasattr(extraction, "split_seed")
    assert not hasattr(extraction, "training_seed")
    assert not hasattr(extraction, "probability_basis")
    assert not hasattr(extraction, "ece_bins")
    assert probe.profile == LINEAR_V1
    assert LINEAR_V1.learning_rate == 1e-3
    assert LINEAR_V1.weight_decay == 1e-4
    assert LINEAR_V1.weight_decay_on == "weights"
    assert LINEAR_V1.betas == (0.9, 0.999)
    assert LINEAR_V1.epsilon == 1e-8
    assert (LINEAR_V1.batch_size, LINEAR_V1.max_epochs, LINEAR_V1.early_stopping_patience) == (256, 200, 20)
    assert probe.ece_bins == 15
    assert probe.confidence_intervals is False


def test_unknown_keys_and_schema_versions_are_rejected(tmp_path):
    extra = _document()
    extra["extraction"]["batch_size"] = 8
    with pytest.raises(ConfigurationError, match="unknown keys: batch_size"):
        _load(extra, tmp_path)
    nested = _document()
    nested["analysis_view"]["filter_correct"]["retention_rate"] = 0.5
    with pytest.raises(ConfigurationError, match="unknown keys: retention_rate"):
        _load(nested, tmp_path)
    missing = _document()
    del missing["probe_run"]["schema_version"]
    with pytest.raises(ConfigurationError, match="schema_version"):
        _load(missing, tmp_path)
    wrong = _document()
    wrong["analysis_view"]["schema_version"] = 2
    with pytest.raises(ConfigurationError, match="schema_version must be 1"):
        _load(wrong, tmp_path)


def test_a_probability_basis_is_not_a_configuration_field(tmp_path):
    inferred = _document()
    inferred["analysis_view"]["filter_correct"]["probability_basis"] = "canonical"
    with pytest.raises(ConfigurationError, match="unknown keys: probability_basis"):
        _load(inferred, tmp_path)
    recorded = _document()
    recorded["analysis_view"]["filter_correct"]["decision_record"] = "choice.json"
    with pytest.raises(ConfigurationError, match="unknown keys: decision_record"):
        _load(recorded, tmp_path)


def test_precision_and_system_message_are_explicit(tmp_path):
    omitted = _document()
    del omitted["extraction"]["model"]["precision"]
    with pytest.raises(ConfigurationError, match="precision"):
        _load(omitted, tmp_path)
    rejected = _document()
    rejected["extraction"]["model"]["precision"] = "int8"
    with pytest.raises(ConfigurationError, match="no silent fallback"):
        _load(rejected, tmp_path)
    explicit = _document()
    explicit["extraction"]["model"]["precision"] = "fp32"
    assert _load(explicit, tmp_path).extraction.precision == "fp32"
    custom = _document()
    custom["extraction"]["prompt"]["system_message"] = "Custom assistant."
    assert _load(custom, tmp_path).extraction.system_message == "Custom assistant."
    missing_message = _document()
    missing_message["extraction"]["prompt"]["system_message"] = None
    with pytest.raises(ConfigurationError, match="must be explicit"):
        _load(missing_message, tmp_path)
    no_chat = _document()
    no_chat["extraction"]["prompt"]["chat"] = False
    no_chat["extraction"]["prompt"]["system_message"] = None
    assert _load(no_chat, tmp_path).extraction.system_message is None
    leaked = _document()
    leaked["extraction"]["prompt"]["chat"] = False
    with pytest.raises(ConfigurationError, match="must be null"):
        _load(leaked, tmp_path)


def test_offline_mode_cannot_download(tmp_path):
    download = _document()
    download["extraction"]["assets"]["allow_download"] = True
    with pytest.raises(ConfigurationError, match="cannot download"):
        _load(download, tmp_path)
    online = _document()
    online["extraction"]["assets"]["offline"] = False
    with pytest.raises(ConfigurationError, match="cannot download"):
        _load(online, tmp_path)


def test_answer_and_position_contracts_fail_closed(tmp_path):
    letters = _document()
    letters["extraction"]["interventions"]["preset_placeholder_source"] = "option_letter"
    with pytest.raises(ConfigurationError, match="option letters"):
        _load(letters, tmp_path)
    selector = _document()
    selector["extraction"]["internals"]["selector"] = "teacher_forced"
    with pytest.raises(UnsupportedPositionSelector):
        _load(selector, tmp_path)
    direct = _document()
    direct["extraction"]["answers"]["mode"] = "direct"
    with pytest.raises(AnswerInstructionMismatch):
        _load(direct, tmp_path)
    fallback = _document()
    fallback["extraction"]["dataset"]["clean_question_path"] = "question"
    with pytest.raises(ConfigurationError, match="No fallback field search"):
        _load(fallback, tmp_path)


def test_split_seed_and_training_seed_stay_separate(tmp_path):
    document = _document()
    document["analysis_view"]["split"]["seed"] = 7
    document["probe_run"]["probe"]["training_seed"] = 9
    loaded = _load(document, tmp_path)
    assert loaded.analysis_view.split_seed == 7
    assert loaded.probe_run.training_seed == 9


def test_missing_storage_root_is_not_created(tmp_path):
    missing = tmp_path / "absent-root"
    document = _document()
    document["extraction"]["storage"]["root"] = str(missing)
    document["extraction"]["assets"]["model_path"] = str(tmp_path / "model")
    document["extraction"]["assets"]["dataset_path"] = str(tmp_path / "data")
    config = _load(document, tmp_path)
    with pytest.raises(ConfigurationError, match="was not created"):
        validate_before_expensive_work(config, repository_root=ROOT)
    assert not missing.exists()
    assert not Path("REQUIRED_LOCAL_PATH").exists()


def test_storage_root_cannot_be_the_repository_or_home(tmp_path):
    document = _document()
    document["extraction"]["storage"]["root"] = str(ROOT)
    config = _load(document, tmp_path)
    with pytest.raises(ConfigurationError, match="inside the repository"):
        validate_before_expensive_work(config, repository_root=ROOT)
    document = _document()
    document["extraction"]["storage"]["root"] = str(Path.home())
    home_config = _load(document, tmp_path)
    with pytest.raises(ConfigurationError, match="home directory"):
        validate_before_expensive_work(home_config, repository_root=tmp_path)


def test_assets_are_checked_before_work(tmp_path):
    root = tmp_path / "storage"
    root.mkdir()
    model = root / "model"
    dataset = root / "dataset"
    model.mkdir()
    dataset.mkdir()
    document = _document()
    document["extraction"]["storage"]["root"] = str(root)
    document["extraction"]["assets"]["model_path"] = str(model)
    document["extraction"]["assets"]["dataset_path"] = str(dataset)
    ready = _load(document, tmp_path)
    validate_before_expensive_work(ready, repository_root=ROOT)
    absent_model = copy.deepcopy(document)
    absent_model["extraction"]["assets"]["model_path"] = str(root / "missing-model")
    with pytest.raises(MissingLocalAsset, match="model"):
        validate_before_expensive_work(_load(absent_model, tmp_path), repository_root=ROOT)
    assert not (root / "missing-model").exists()


def test_variant_filter_flags_default_off_and_the_final_filter_needs_stats(tmp_path):
    shipped = _shipped().analysis_view
    assert shipped.variant_stats is True
    assert shipped.require_confident_on_every_variant is False
    omitted = _document()
    del omitted["analysis_view"]["filter_correct"]["variant_stats"]
    del omitted["analysis_view"]["filter_correct"]["require_confident_on_every_variant"]
    view = _load(omitted, tmp_path).analysis_view
    assert (view.variant_stats, view.require_confident_on_every_variant) == (False, False)
    dependent = _document()
    dependent["analysis_view"]["filter_correct"]["variant_stats"] = False
    dependent["analysis_view"]["filter_correct"]["require_confident_on_every_variant"] = True
    with pytest.raises(ConfigurationError, match="require_confident_on_every_variant needs variant_stats"):
        _load(dependent, tmp_path)
    loose = _document()
    loose["analysis_view"]["filter_correct"]["variant_stats"] = "yes"
    with pytest.raises(ConfigurationError, match="variant_stats must be an explicit boolean"):
        _load(loose, tmp_path)
    renamed = _document()
    renamed["analysis_view"]["filter_correct"]["require_all_variants"] = True
    with pytest.raises(ConfigurationError, match="renamed to require_confident_on_every_variant"):
        _load(renamed, tmp_path)


def test_empty_eval_variants_mean_no_ood_view(tmp_path):
    document = _document()
    document["analysis_view"]["analysis"]["eval_variants"] = []
    assert _load(document, tmp_path).analysis_view.eval_variants == ()
    document["analysis_view"]["analysis"]["train_variants"] = []
    with pytest.raises(ConfigurationError, match="train_variants must be a non-empty list"):
        _load(document, tmp_path)


def test_the_working_config_still_loads():
    load_config(ROOT / "configs" / "initial_sycophancy.yaml")
