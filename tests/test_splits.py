import inspect
import math
from decimal import Decimal

import numpy
import pytest

from knowledge_recovery.provenance import identify
from knowledge_recovery.views import GROUPING_ALGORITHM, freeze_grouped_split, grouping_version


def _freeze(group_ids, **overrides):
    payload = {
        "train_fraction": 0.6,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "split_seed": 42,
    }
    payload.update(overrides)
    return freeze_grouped_split(group_ids, **payload)


def test_assignment_is_the_pcg64_permutation_of_lexicographically_sorted_ids():
    group_ids = ["b", "a", "é", " a", "B", "id-" + ("x" * 80)]
    manifest = _freeze(group_ids)
    ordered = tuple(sorted(group_ids))
    permutation = numpy.random.Generator(numpy.random.PCG64(42)).permutation(len(ordered))
    shuffled = [ordered[int(index)] for index in permutation]
    assert manifest.source_universe == ordered
    assert ordered[0] == " a"
    assert "B" in manifest.source_universe and "b" in manifest.source_universe
    assert manifest.group_to_split[shuffled[0]] == "train"
    assert manifest.group_to_split[shuffled[1]] == "train"
    assert manifest.group_to_split[shuffled[2]] == "train"
    assert manifest.group_to_split[shuffled[3]] == "val"
    assert manifest.group_to_split[shuffled[4]] == "test"
    assert manifest.group_to_split[shuffled[5]] == "test"
    assert manifest.allocations == {"train": 3, "val": 1, "test": 2}
    assert set(manifest.group_to_split) == set(group_ids)
    assert manifest.grouping_version == grouping_version()
    assert manifest.grouping_version == f"{GROUPING_ALGORITHM}/numpy-{numpy.__version__}"
    assert manifest.split_seed == 42
    assert "training_seed" not in inspect.signature(freeze_grouped_split).parameters


def test_remainder_goes_to_test_and_decimal_floor_keeps_the_boundary_group():
    small = _freeze(["g1", "g0"])
    assert small.allocations == {"train": 1, "val": 0, "test": 1}
    single = _freeze(["only"])
    assert single.allocations == {"train": 0, "val": 0, "test": 1}
    assert single.group_to_split == {"only": "test"}
    # floor(90 * 0.7) is 63. The binary product is 62.99999999999999.
    boundary = _freeze(
        [f"g{index:03d}" for index in range(90)],
        train_fraction=0.7,
        val_fraction=0.2,
        test_fraction=0.1,
    )
    assert math.floor(90 * 0.7) == 62
    assert math.floor(Decimal(90) * Decimal("0.7")) == 63
    assert boundary.allocations == {"train": 63, "val": 18, "test": 9}
    assert sum(boundary.allocations.values()) == 90
    assert set(boundary.group_to_split.values()) == {"train", "val", "test"}


def test_the_saved_mapping_depends_on_the_sorted_universe_and_the_split_seed():
    groups = [f"group-{index:02d}" for index in range(20)]
    forward = _freeze(groups)
    backward = _freeze(list(reversed(groups)))
    assert identify(forward).digest == identify(backward).digest
    assert forward.source_universe == tuple(sorted(groups))
    other_seed = _freeze(groups, split_seed=0)
    assert other_seed.group_to_split != forward.group_to_split
    assert other_seed.split_seed == 0
    assert identify(forward).digest == identify(_freeze(groups)).digest
    with pytest.raises(TypeError):
        freeze_grouped_split(groups, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2, split_seed=1, training_seed=9)


def test_empty_universe_is_frozen_and_invalid_inputs_are_rejected():
    empty = _freeze([])
    assert empty.source_universe == ()
    assert empty.group_to_split == {}
    assert empty.allocations == {"train": 0, "val": 0, "test": 0}
    with pytest.raises(ValueError, match="duplicate"):
        _freeze(["a", "a"])
    with pytest.raises(ValueError, match="non-empty"):
        _freeze(["a", ""])
    with pytest.raises(TypeError, match="sequence"):
        _freeze("ab")
    with pytest.raises(ValueError, match="sum to 1"):
        _freeze(["a", "b"], test_fraction=0.3)
    with pytest.raises(ValueError, match="strictly between"):
        _freeze(["a"], train_fraction=1)
    with pytest.raises(ValueError, match="non-negative"):
        _freeze(["a"], split_seed=-1)
    with pytest.raises(ValueError, match="non-negative"):
        _freeze(["a"], split_seed=True)
    with pytest.raises(ValueError, match="nonfinite"):
        _freeze(["a"], train_fraction=math.nan)
