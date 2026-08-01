from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from hegel_machine.phase3_m25_rows_v1 import (
    FAIL_CANONICAL_INPUT_HASH_MISMATCH,
    FAIL_INPUT_SIGNATURE_MISMATCH,
    FAIL_ROW_ORDERING,
    FAIL_TARGET_OUTPUT_TYPE,
    FAIL_UNIVERSE_INDEX_DUPLICATE,
    FAIL_UNIVERSE_INDEX_GAP,
    M25TypedRowError,
    REJECT_MACHINE_ID_LENGTH,
    REJECT_MACHINE_ID_NON_ASCII,
    REJECT_MACHINE_ID_SYNTAX,
    REJECT_ODD_BIT_COUNT,
    REJECT_ODD_BIT_TYPE,
    REJECT_ODD_SET_SIZE,
    REJECT_SINK_BALANCE,
    REJECT_SINK_VALUE,
    bounded_universe_row_v1,
    canonical_input_hash_v1,
    complete_typed_rows_report_v1,
    decode_odd_input_v1,
    decode_sink_input_v1,
    generate_odd_role_rows_v1,
    generate_sink_role_rows_v1,
    id_digest_preimage_v1,
    id_digest_v1,
    odd_input_v1,
    sink_input_v1,
    target_truth_row_v1,
    validate_typed_role_rows,
)
from hegel_machine.strict_cbor_v1 import canonical_cbor_encode


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "golden_vectors" / "phase3_m25_typed_rows_v1.json"
RUST_ROOT = ROOT / "rust" / "formal_bridge_m25"
RUST_BINARY = RUST_ROOT / "target" / "debug" / "hegel-formal-bridge-m25"


def _fixture() -> dict[str, object]:
    value = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _error_code(callable_: object, *args: object, **kwargs: object) -> str:
    assert callable(callable_)
    with pytest.raises(M25TypedRowError) as caught:
        callable_(*args, **kwargs)  # type: ignore[operator]
    return caught.value.code


def test_fixture_is_v112_synthetic_and_has_no_authority_effect() -> None:
    fixture = _fixture()
    assert fixture["schema_version"] == "hegel-phase3-m25-typed-rows-golden/1"
    assert fixture["artifact_kind"] == "SYNTHETIC_NON_AUTHORITATIVE"
    assert fixture["machine_freeze_id"] == "hegel-freeze-p2b-p3-v1.1.2"
    assert fixture["authority_boundary"] == {
        "gate_effect": "NONE",
        "m3_gates_before": 14,
        "m3_gates_after": 14,
        "child_state": "NOT_RUN",
        "contains_real_secret_material": False,
        "authoritative_root_generation": False,
        "formal_roots_generated": False,
        "seed_genesis_performed": False,
        "signature_claim": False,
    }


def test_python_independently_reproduces_every_checked_in_typed_vector() -> None:
    fixture = _fixture()
    report = complete_typed_rows_report_v1()
    assert report["machine_freeze_id"] == fixture["machine_freeze_id"]
    assert report["id_digest"] == fixture["id_digest"]
    assert report["roles"] == fixture["roles"]


def test_id_digest_exact_preimage_digest_and_failure_precedence() -> None:
    vector = _fixture()["id_digest"]
    assert isinstance(vector, dict)
    machine_id = vector["machine_id"]
    assert isinstance(machine_id, str)
    assert id_digest_preimage_v1(machine_id).hex() == vector["preimage_hex"]
    assert id_digest_v1(machine_id).hex() == vector["digest_hex"]
    assert _error_code(id_digest_v1, "é") == REJECT_MACHINE_ID_NON_ASCII
    assert _error_code(id_digest_v1, "a" * 257) == REJECT_MACHINE_ID_LENGTH
    for invalid in ("", " leading", "trailing ", "bad?character"):
        assert _error_code(id_digest_v1, invalid) == REJECT_MACHINE_ID_SYNTAX


def test_typed_inputs_round_trip_without_bool_coercion() -> None:
    odd = odd_input_v1(5, (0, 0, 0, 0, 1))
    sink = sink_input_v1(0, 1, 0, 1)
    assert decode_odd_input_v1(canonical_cbor_encode(odd)) == odd
    assert decode_sink_input_v1(canonical_cbor_encode(sink)) == sink
    assert canonical_input_hash_v1(odd).hex() == (
        "41de3d87149e3d5a9491c856d674c500cdb5dec260cce2c4f4e9c9e7114ee9ea"
    )
    assert canonical_input_hash_v1(sink).hex() == (
        "7e00b8d30c05362e63c9f5b4a8217bc8ce90356fcdc9009507aece95d87539f2"
    )


@pytest.mark.parametrize(
    ("set_size", "bits", "code"),
    [
        (4, (0, 0, 0, 0), REJECT_ODD_SET_SIZE),
        (9, (0,) * 9, REJECT_ODD_SET_SIZE),
        (5, (0, 0, 0, 0), REJECT_ODD_BIT_COUNT),
        (5, (0, 0, 0, 0, True), REJECT_ODD_BIT_TYPE),
        (5, (0, 0, 0, 0, 2), REJECT_ODD_BIT_TYPE),
    ],
)
def test_odd_input_exact_negative_codes(
    set_size: int, bits: tuple[object, ...], code: str
) -> None:
    assert _error_code(odd_input_v1, set_size, bits) == code


@pytest.mark.parametrize(
    ("values", "code"),
    [
        ((True, 0, 0, 0), REJECT_SINK_VALUE),
        ((0, 0, 0, 5), REJECT_SINK_VALUE),
        ((0, 1, 0, 0), REJECT_SINK_BALANCE),
    ],
)
def test_sink_input_exact_negative_codes(
    values: tuple[object, object, object, object], code: str
) -> None:
    assert _error_code(sink_input_v1, *values) == code


def test_generators_freeze_cardinality_order_strata_and_truth_profile() -> None:
    odd = generate_odd_role_rows_v1()
    sink = generate_sink_role_rows_v1()
    assert len(odd.universe_rows) == len(odd.truth_rows) == 480
    assert len(sink.universe_rows) == len(sink.truth_rows) == 85

    odd_inputs = [row[5] for row in odd.universe_rows]
    odd_strata = Counter(
        (item[3], row[5]) for item, row in zip(odd_inputs, odd.truth_rows)
    )
    assert odd_strata == {
        (5, 0): 16,
        (5, 1): 16,
        (6, 0): 32,
        (6, 1): 32,
        (7, 0): 64,
        (7, 1): 64,
        (8, 0): 128,
        (8, 1): 128,
    }
    assert [tuple(item[3:]) for item in sink.universe_rows[:2]] == [
        (0, 2, sink_input_v1(0, 0, 0, 0)),
        (1, 2, sink_input_v1(0, 1, 0, 1)),
    ]
    assert Counter(row[5][6] for row in sink.universe_rows) == {
        0: 15,
        1: 18,
        2: 19,
        3: 18,
        4: 15,
    }
    assert all(row[5] == 1 and type(row[5]) is int for row in sink.truth_rows)


def test_row_builders_reject_signature_output_and_hash_type_confusion() -> None:
    odd = odd_input_v1(5, (0, 0, 0, 0, 0))
    assert (
        _error_code(bounded_universe_row_v1, 0, 2, odd)
        == FAIL_INPUT_SIGNATURE_MISMATCH
    )
    for invalid in (True, 2):
        assert (
            _error_code(target_truth_row_v1, 0, b"\x00" * 32, invalid)
            == FAIL_TARGET_OUTPUT_TYPE
        )
    assert (
        _error_code(target_truth_row_v1, 0, b"short", 0)
        == FAIL_CANONICAL_INPUT_HASH_MISMATCH
    )


def _mutated_rows() -> tuple[list[object], list[object]]:
    rows = generate_odd_role_rows_v1()
    return list(rows.universe_rows[:3]), list(rows.truth_rows[:3])


def test_role_validation_rejects_duplicate_gap_and_wrong_order_exactly() -> None:
    universe, truth = _mutated_rows()
    universe[1] = tuple(universe[1][:3]) + (0,) + tuple(universe[1][4:])
    assert (
        _error_code(
            validate_typed_role_rows,
            universe,
            truth,
            expected_input_signature_id=1,
        )
        == FAIL_UNIVERSE_INDEX_DUPLICATE
    )

    universe, truth = _mutated_rows()
    universe[1] = tuple(universe[1][:3]) + (2,) + tuple(universe[1][4:])
    universe[2] = tuple(universe[2][:3]) + (3,) + tuple(universe[2][4:])
    assert (
        _error_code(
            validate_typed_role_rows,
            universe,
            truth,
            expected_input_signature_id=1,
        )
        == FAIL_UNIVERSE_INDEX_GAP
    )

    universe, truth = _mutated_rows()
    universe[1], universe[2] = universe[2], universe[1]
    assert (
        _error_code(
            validate_typed_role_rows,
            universe,
            truth,
            expected_input_signature_id=1,
        )
        == FAIL_ROW_ORDERING
    )


def test_role_validation_rejects_truth_hash_mismatch() -> None:
    universe, truth = _mutated_rows()
    truth[0] = tuple(truth[0][:4]) + (b"\xff" * 32,) + tuple(truth[0][5:])
    assert (
        _error_code(
            validate_typed_role_rows,
            universe,
            truth,
            expected_input_signature_id=1,
        )
        == FAIL_CANONICAL_INPUT_HASH_MISMATCH
    )


@pytest.fixture(scope="session")
def typed_rust_binary() -> Path:
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo is required for independent Rust typed-row replay")
    completed = subprocess.run(
        [
            cargo,
            "build",
            "--quiet",
            "--locked",
            "--target-dir",
            str(RUST_ROOT / "target"),
            "--manifest-path",
            str(RUST_ROOT / "Cargo.toml"),
        ],
        cwd=RUST_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        pytest.fail(f"Rust typed-row build failed: {completed.stderr}")
    return RUST_BINARY


def _rust_request(binary: Path, request: dict[str, object]) -> dict[str, object]:
    completed = subprocess.run(
        [str(binary)],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    response = json.loads(completed.stdout)
    assert isinstance(response, dict)
    if completed.returncode != 0:
        raise AssertionError(response)
    return response


def test_rust_cli_independently_generates_both_full_role_reports(
    typed_rust_binary: Path,
) -> None:
    fixture = _fixture()
    expected_roles = fixture["roles"]
    assert isinstance(expected_roles, list)
    for expected in expected_roles:
        assert isinstance(expected, dict)
        response = _rust_request(
            typed_rust_binary,
            {"op": "typed_rows", "role_id": expected["input_signature_id"]},
        )
        assert response.pop("ok") is True
        assert response.pop("op") == "typed_rows"
        assert response == expected


def test_rust_cli_independently_reproduces_id_digest(
    typed_rust_binary: Path,
) -> None:
    expected = _fixture()["id_digest"]
    assert isinstance(expected, dict)
    response = _rust_request(
        typed_rust_binary,
        {"op": "id_digest", "machine_id": expected["machine_id"]},
    )
    assert response == {"ok": True, "op": "id_digest", **expected}


def test_fixture_mutation_cannot_hide_a_python_root_mismatch() -> None:
    fixture = deepcopy(_fixture())
    roles = fixture["roles"]
    assert isinstance(roles, list) and isinstance(roles[0], dict)
    roles[0]["universe_root_hex"] = "00" * 32
    assert complete_typed_rows_report_v1()["roles"] != roles
