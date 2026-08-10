from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

universe = importlib.import_module("hegel_machine.phase3_q1_universe_v1")


def test_target_blind_universe_counts_roots_and_order() -> None:
    first, second = universe.all_production_universes_v1()
    assert (first.input_signature_id, second.input_signature_id) == (1, 2)
    assert (len(first.rows), len(second.rows)) == (480, 85)
    assert first.universe_root == universe.ODD_UNIVERSE_ROOT
    assert second.universe_root == universe.SINK_UNIVERSE_ROOT
    assert tuple(row[3] for row in first.rows) == tuple(range(480))
    assert tuple(row[3] for row in second.rows) == tuple(range(85))
    assert all(row[4] == 1 for row in first.rows)
    assert all(row[4] == 2 for row in second.rows)


def test_universe_generator_exposes_no_truth_or_split_surface() -> None:
    exported = set(universe.__all__)
    forbidden = ("truth", "target", "split", "role", "match", "seed")
    assert all(not any(token in name.lower() for token in forbidden) for name in exported)
    source = Path(universe.__file__).read_text(encoding="utf-8")
    assert "phase3_m25_rows_v1" not in source
    assert "phase3_dsl_v1" not in source


def test_typed_rows_reject_signature_and_content_drift() -> None:
    first = universe.production_universe_v1(1)
    with pytest.raises(universe.Q1UniverseError) as error:
        universe.ProductionUniverseV1(2, first.rows)
    assert error.value.code in {
        universe.FAIL_Q1_UNIVERSE_ROW_COUNT,
        universe.REJECT_Q1_UNIVERSE_SIGNATURE,
    }
    with pytest.raises(universe.Q1UniverseError) as error:
        universe.production_universe_v1(3)
    assert error.value.code == universe.REJECT_Q1_UNIVERSE_SIGNATURE
    with pytest.raises(universe.Q1UniverseError) as error:
        universe.production_universe_v1(True)
    assert error.value.code == universe.REJECT_Q1_UNIVERSE_SIGNATURE
    with pytest.raises(universe.Q1UniverseError) as error:
        universe.ProductionUniverseV1(True, first.rows)
    assert error.value.code == universe.REJECT_Q1_UNIVERSE_SIGNATURE


def test_universe_rejects_mutable_or_subclassed_row_containers() -> None:
    odd = universe.production_universe_v1(1)
    mutable = list(odd.rows)
    with pytest.raises(universe.Q1UniverseError) as error:
        universe.ProductionUniverseV1(1, mutable)
    assert error.value.code == universe.FAIL_Q1_UNIVERSE_ROW_ORDER

    class TupleAlias(tuple):
        pass

    with pytest.raises(universe.Q1UniverseError) as error:
        universe.ProductionUniverseV1(1, TupleAlias(odd.rows))
    assert error.value.code == universe.FAIL_Q1_UNIVERSE_ROW_ORDER

    drifted_row = TupleAlias(odd.rows[0])
    with pytest.raises(universe.Q1UniverseError) as error:
        universe.ProductionUniverseV1(1, (drifted_row,) + odd.rows[1:])
    assert error.value.code == universe.FAIL_Q1_UNIVERSE_ROW_ORDER

    with pytest.raises(universe.Q1UniverseError) as error:
        universe._universe_row_v1(0, True, odd.rows[0][5])
    assert error.value.code == universe.REJECT_Q1_UNIVERSE_SIGNATURE
    with pytest.raises(universe.Q1UniverseError) as error:
        universe._universe_row_v1(0, 1, list(odd.rows[0][5]))
    assert error.value.code == universe.FAIL_Q1_UNIVERSE_ROW_ORDER


def test_universe_rejects_python_equality_aliases_in_row_prefix() -> None:
    odd = universe.production_universe_v1(1)

    class BytesAlias(bytes):
        pass

    mutations = (
        (True,) + odd.rows[0][1:],
        (odd.rows[0][0], True) + odd.rows[0][2:],
        odd.rows[0][:2]
        + (BytesAlias(odd.rows[0][2]),)
        + odd.rows[0][3:],
    )
    for mutation in mutations:
        with pytest.raises(universe.Q1UniverseError) as error:
            universe.ProductionUniverseV1(1, (mutation,) + odd.rows[1:])
        assert error.value.code == universe.FAIL_Q1_UNIVERSE_ROW_ORDER


def test_universe_rows_have_stable_first_and_last_inputs() -> None:
    odd = universe.production_universe_v1(1)
    sink = universe.production_universe_v1(2)
    assert odd.rows[0][5][3:] == (5, (0, 0, 0, 0, 0))
    assert odd.rows[-1][5][3:] == (8, (1, 1, 1, 1, 1, 1, 1, 1))
    assert sink.rows[0][5][3:] == (0, 0, 0, 0)
    assert sink.rows[-1][5][3:] == (4, 4, 4, 4)
    assert odd.universe_root.hex() == (
        "b7e6eed1174ceee1944bd540cbb0ace4c5c7fc0dffea79dd956a51f7a2410a05"
    )
    assert sink.universe_root.hex() == (
        "1a46f9967ad48df6ec1d9be609de701413ce86171fb0f92495a982bef0f40ff5"
    )


def test_universe_roots_and_counts_match_preregistered_config() -> None:
    config = json.loads(
        (ROOT / "config" / "phase3_q1_capacity_preflight_v1.json").read_text(
            encoding="utf-8"
        )
    )
    actual = universe.all_production_universes_v1()
    rows = config["universes"]
    assert len(rows) == len(actual) == 2
    for expected, generated in zip(rows, actual, strict=True):
        assert expected["input_signature_id"] == generated.input_signature_id
        assert expected["row_count"] == len(generated.rows)
        assert expected["historical_payload_universe_root"] == (
            f"sha256:{generated.universe_root.hex()}"
        )
        assert expected["truth_root_in_preflight"] is None
