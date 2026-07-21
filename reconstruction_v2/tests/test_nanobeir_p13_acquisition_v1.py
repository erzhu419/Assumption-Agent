from pathlib import Path

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_completecase_acquisition_v1 as mature,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p13_acquisition_v1 as acquisition,
)


def test_allocation_uses_frozen_36_item_contract() -> None:
    blocks = acquisition.allocate_blocks(
        bytes(range(32)),
        acquisition.FAMILIES[0],
        tuple(f"q{i}" for i in range(50)),
    )
    assert {name: len(values) for name, values in blocks.items()} == {
        "C_confirm": 10,
        "A_form": 8,
        "F_search": 5,
        "A_hold": 5,
        "M_search": 5,
        "RESERVE": 3,
    }


def test_wrapper_context_restores_mature_acquisition() -> None:
    original_schema = mature.SCHEMA
    original_availability = mature.availability
    with acquisition._patched_mature_acquisition():
        assert mature.SCHEMA == acquisition.SCHEMA
        assert mature.availability is acquisition.availability
    assert mature.SCHEMA == original_schema
    assert mature.availability is original_availability


def test_formal_refuses_consumed_root_before_private_access(
    tmp_path: Path,
) -> None:
    base = tmp_path / "reconstruction_v2"
    (base / acquisition.RUN_ROOT_RELATIVE).mkdir(parents=True)
    with pytest.raises(acquisition.OneShotRefusal, match="root already exists"):
        acquisition.run_formal(tmp_path)
