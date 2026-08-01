from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from hegel_machine.phase3_m25_replay_v1 import (
    ARTIFACT_KIND,
    CHECKED_IN_REPORT_PATH,
    DEFAULT_RUST_BINARY,
    GOLDEN_VECTOR_PATH,
    PYTHON_SOURCE_FILES,
    RUST_CRATE_ROOT,
    RUST_SOURCE_FILES,
    SUPPORTED_OPERATIONS,
    dual_synthetic_replay_report,
    load_golden_vectors,
    python_synthetic_replay,
    validate_dual_synthetic_replay_report,
    validate_historical_dual_synthetic_replay_report,
)
from hegel_machine.cli import main
from hegel_machine.hashing import stable_hash


@pytest.fixture(scope="session")
def m25_rust_binary() -> Path:
    """Build from the currently bound Rust source; never trust a stale binary."""

    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo is required to bind the Rust replay binary to current source")
    completed = subprocess.run(
        [
            cargo,
            "build",
            "--quiet",
            "--locked",
            "--target-dir",
            str(RUST_CRATE_ROOT / "target"),
            "--manifest-path",
            str(RUST_CRATE_ROOT / "Cargo.toml"),
        ],
        cwd=RUST_CRATE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        pytest.fail(
            "failed to build Rust M2.5 replay: "
            f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
        )
    assert DEFAULT_RUST_BINARY.is_file()
    return DEFAULT_RUST_BINARY


def test_golden_fixture_is_explicitly_synthetic_and_covers_every_primitive() -> None:
    fixture = load_golden_vectors()
    assert fixture["artifact_kind"] == ARTIFACT_KIND
    authority = fixture["authority_boundary"]
    assert isinstance(authority, dict)
    assert authority == {
        "gate_effect": "NONE",
        "m3_gates_before": 14,
        "m3_gates_after": 14,
        "child_state": "NOT_RUN",
        "contains_real_secret_material": False,
        "authoritative_root_generation": False,
        "seed_genesis_performed": False,
        "custodian_signature_claim": False,
    }
    vectors = fixture["vectors"]
    assert isinstance(vectors, list)
    assert {vector["op"] for vector in vectors} == SUPPORTED_OPERATIONS
    assert len(vectors) == 20
    assert len(SUPPORTED_OPERATIONS) == 8
    assert sum(vector["op"] == "reject_decode" for vector in vectors) == 13

    raw_fixture = GOLDEN_VECTOR_PATH.read_text(encoding="utf-8")
    assert "synthetic_public_test_input_hex" in raw_fixture
    assert '"master_seed_hex"' not in raw_fixture
    assert "SYNTHETIC_NON_AUTHORITATIVE" in raw_fixture


def test_python_replay_matches_every_checked_in_expected_value() -> None:
    report = python_synthetic_replay()
    assert report["implementation"] == "python"
    assert report["vector_count"] == 20
    assert report["expected_match_count"] == 20
    assert report["all_expected_outputs_match"] is True
    assert all(result["expected"] == result["actual"] for result in report["results"])

    source_hashes = report["source_hashes"]
    assert isinstance(source_hashes, dict)
    assert set(source_hashes) == {
        path.relative_to(GOLDEN_VECTOR_PATH.parents[1]).as_posix()
        for path in PYTHON_SOURCE_FILES
    }
    assert all(value.startswith("sha256:") for value in source_hashes.values())
    assert report["source_set_sha256"].startswith("sha256:")


def test_checked_in_dual_report_is_pinned_historical_not_current() -> None:
    checked_in = json.loads(CHECKED_IN_REPORT_PATH.read_text(encoding="utf-8"))
    validate_historical_dual_synthetic_replay_report(checked_in)
    with pytest.raises(AssertionError, match="source hashes are stale"):
        validate_dual_synthetic_replay_report(checked_in)


def test_dual_replay_matches_expected_and_each_other_without_advancing_authority(
    m25_rust_binary: Path,
) -> None:
    report = dual_synthetic_replay_report(m25_rust_binary)
    validate_dual_synthetic_replay_report(report)
    assert report["artifact_kind"] == "SYNTHETIC_NON_AUTHORITATIVE"
    assert report["status"] == "SYNTHETIC_FOUNDATION_DUAL_REPLAY_PASS"
    assert report["vector_count"] == 20
    assert report["both_endpoints_match_expected"] is True
    assert report["cross_language_actual_equal"] is True
    assert report["cross_language_mismatches"] == []

    python_results = {
        result["name"]: result["actual"] for result in report["python"]["results"]
    }
    rust_results = {
        result["name"]: result["actual"] for result in report["rust"]["results"]
    }
    assert python_results == rust_results
    rejection_results = [
        result for result in report["python"]["results"] if result["op"] == "reject_decode"
    ]
    assert len(rejection_results) == 13
    assert all(result["actual"]["accepted"] is False for result in rejection_results)
    assert report["rust"]["all_expected_outputs_match"] is True
    assert report["rust"]["binary_sha256"].startswith("sha256:")
    assert report["rust"]["binary_source_binding_claim"] is False
    assert (
        report["rust"]["binary_provenance"]
        == "CALLER_SUPPLIED_UNATTESTED_SYNTHETIC_REPLAY"
    )
    assert set(report["rust"]["source_hashes"]) == {
        path.relative_to(GOLDEN_VECTOR_PATH.parents[1]).as_posix()
        for path in RUST_SOURCE_FILES
    }

    assert report["m3_gates_satisfied"] == 14
    assert report["m3_gates_total"] == 24
    assert report["m3_gate_delta"] == 0
    assert report["child_state"] == "NOT_RUN"
    assert report["m3_entry_allowed"] is False
    assert report["split_seed_first_instantiated"] is False
    assert report["custodian_signature_claim"] is False
    assert report["formal_input_roots"] is None
    assert report["formal_output_roots"] is None
    assert report["m3_execution_manifest_root"] is None
    assert report["formal_roots_generated"] is False

    checked_in = json.loads(CHECKED_IN_REPORT_PATH.read_text(encoding="utf-8"))
    validate_historical_dual_synthetic_replay_report(checked_in)
    assert checked_in["python"]["results"] == report["python"]["results"]
    assert checked_in["rust"]["results"] == report["rust"]["results"]


def test_authority_boundary_tamper_is_rejected_before_replay(tmp_path: Path) -> None:
    fixture = json.loads(GOLDEN_VECTOR_PATH.read_text(encoding="utf-8"))
    fixture["authority_boundary"]["m3_gates_after"] = 15
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(fixture), encoding="utf-8")
    with pytest.raises(ValueError, match="authority boundary"):
        load_golden_vectors(tampered)


def test_dual_report_rejects_authority_or_self_id_tampering() -> None:
    checked_in = json.loads(CHECKED_IN_REPORT_PATH.read_text(encoding="utf-8"))
    report = deepcopy(checked_in)
    report["m3_gate_delta"] = 1
    with pytest.raises(AssertionError, match="m3_gate_delta"):
        validate_historical_dual_synthetic_replay_report(report)

    report = deepcopy(checked_in)
    report["diagnostic_report_id"] = "phase3_m25_synthetic_dual_replay_" + "00" * 32
    with pytest.raises(AssertionError, match="report ID mismatch"):
        validate_historical_dual_synthetic_replay_report(report)


@pytest.mark.parametrize(
    ("field", "forged_value", "message"),
    [
        ("vector_count", 0, "vector_count"),
        ("python", {}, "python endpoint report field-set"),
        ("schema_version", "forged", "schema_version"),
    ],
)
def test_dual_report_rejects_self_consistent_nested_forgery(
    field: str,
    forged_value: object,
    message: str,
) -> None:
    report = json.loads(CHECKED_IN_REPORT_PATH.read_text(encoding="utf-8"))
    report[field] = forged_value
    report.pop("diagnostic_report_id")
    report["diagnostic_report_id"] = stable_hash(
        report,
        prefix="phase3_m25_synthetic_dual_replay_",
    )
    with pytest.raises(AssertionError, match=message):
        validate_historical_dual_synthetic_replay_report(report)


def test_dual_report_rejects_bool_integer_type_confusion_with_valid_self_id() -> None:
    report = json.loads(CHECKED_IN_REPORT_PATH.read_text(encoding="utf-8"))
    for endpoint_name in ("python", "rust"):
        rejection = next(
            result
            for result in report[endpoint_name]["results"]
            if result["op"] == "reject_decode"
        )
        rejection["expected"]["accepted"] = 0
        rejection["actual"]["accepted"] = 0
    report.pop("diagnostic_report_id")
    report["diagnostic_report_id"] = stable_hash(
        report,
        prefix="phase3_m25_synthetic_dual_replay_",
    )
    with pytest.raises(AssertionError, match="stale or forged"):
        validate_historical_dual_synthetic_replay_report(report)


def test_dual_replay_cli_emits_non_authoritative_artifact(
    m25_rust_binary: Path,
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "m25-dual.json"
    assert (
        main(
            [
                "phase3-m25-synthetic-replay",
                "--rust-binary",
                str(m25_rust_binary),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert printed == written
    assert written["status"] == "SYNTHETIC_FOUNDATION_DUAL_REPLAY_PASS"
    assert written["m3_gate_delta"] == 0
    assert written["formal_roots_generated"] is False
