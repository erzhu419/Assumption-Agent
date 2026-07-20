from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p11_c_confirm_runtime_v1 as p11_runtime,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_c_confirm_runtime_v1 as runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "reconstruction_v2"


@dataclass(frozen=True)
class Item:
    query: str


def _row(ordinal: int, valid: bool) -> dict:
    return {
        "completion_sha256": f"{ordinal + 1:064x}",
        "completion_token_count": 20 + ordinal,
        "expansions": ["a", "b", "c", "d"] if valid else [],
        "generation_valid": valid,
        "ordinal": ordinal,
    }


def test_totalization_preserves_valid_and_totalizes_invalid() -> None:
    output = {"items": [_row(0, True), _row(1, False)], "schema": "qwen"}
    totalized, audit = runtime.totalize_qwen_output(
        output, [Item("original"), Item("Einstein Relativity theory")]
    )
    assert totalized["items"][0] == output["items"][0]
    assert totalized["items"][1]["generation_valid"] is True
    assert totalized["items"][1]["expansions"] == [
        "Einstein Relativity theory named entities terminology",
        "Einstein Relativity theory relationship comparison",
        "Einstein Relativity theory causal mechanism explanation",
        "Einstein Relativity theory conditions exclusions context",
    ]
    assert audit["source_valid_generation_count"] == 1
    assert audit["totalized_generation_count"] == 1
    assert audit["items"][1]["completion_sha256"] == f"{2:064x}"


def test_totalization_fallback_is_length_bounded_and_distinct() -> None:
    output = {"items": [_row(0, False)], "schema": "qwen"}
    totalized, _audit = runtime.totalize_qwen_output(output, [Item("x" * 1200)])
    expansions = totalized["items"][0]["expansions"]
    assert len(expansions) == len(set(expansions)) == 4
    assert max(map(len, expansions)) <= 1000
    assert all(value.startswith("x" * 900) for value in expansions)


def test_public_slot_rename_is_recursive_and_rehash_ready() -> None:
    value = {
        "P11": 1,
        "nested": [{"P11_rows": [1], "status": "same_source_P11_stops"}],
        "self_sha256": "discard",
    }
    assert runtime._rename_public_slot(value) == {
        "P12": 1,
        "nested": [{"P12_rows": [1], "status": "same_source_P12_stops"}],
    }


def test_compatibility_context_restores_controller() -> None:
    original_schema = p11_runtime.SCHEMA
    original_acquisition = p11_runtime.acquisition
    with runtime._patched_controller():
        assert p11_runtime.SCHEMA == runtime.SCHEMA
        assert p11_runtime.acquisition is runtime.acquisition
    assert p11_runtime.SCHEMA == original_schema
    assert p11_runtime.acquisition is original_acquisition


def test_private_view_loads_without_opening_label_pack() -> None:
    acquisition_result = runtime._load_acquisition(BASE)
    with runtime._patched_controller():
        items = p11_runtime.load_views(BASE, acquisition_result)
    assert len(items) == 36
    assert {item.family for item in items} == set(runtime.acquisition.FAMILIES)


def test_formal_refuses_consumed_root_before_freeze_access(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = project / "reconstruction_v2" / runtime.RUN_ROOT_RELATIVE
    root.mkdir(parents=True)
    with pytest.raises(runtime.OneShotRefusal):
        runtime.run_formal(project)
