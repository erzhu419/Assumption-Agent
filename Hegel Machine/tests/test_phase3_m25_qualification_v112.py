from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from hegel_machine import phase3_m25_qualification_v112 as qualification_module
from hegel_machine.cli import main
from hegel_machine.hashing import stable_hash
from hegel_machine.phase3_m25_rows_v1 import complete_typed_rows_report_v1
from hegel_machine.phase3_m25_qualification_v112 import (
    DEFAULT_RUST_BINARY,
    EXPECTED_ROLE_ROOTS,
    M25QualificationError,
    dual_typed_rows_qualification_report,
    validate_checked_typed_rows_qualification_report,
    validate_dual_typed_rows_qualification_report,
)


ROOT = Path(__file__).resolve().parents[1]
RUST_ROOT = ROOT / "rust" / "formal_bridge_m25"
CHECKED_IN = ROOT / "artifacts" / "phase3_m25_wire_completion_qualification_v112.json"


@pytest.fixture(scope="session")
def m25_v112_rust_binary() -> Path:
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo is required for independent Rust v1.1.2 replay")
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
        pytest.fail(f"Rust v1.1.2 qualification build failed: {completed.stderr}")
    assert DEFAULT_RUST_BINARY.is_file()
    return DEFAULT_RUST_BINARY


def test_v112_qualification_matches_both_full_roles_and_preserves_authority(
    m25_v112_rust_binary: Path,
) -> None:
    report = dual_typed_rows_qualification_report(m25_v112_rust_binary)
    validate_dual_typed_rows_qualification_report(
        report,
        m25_v112_rust_binary,
    )
    assert report["status"] == "DUAL_TYPED_ROWS_AND_ROOTS_CANDIDATE_PASS"
    assert report["python_report"] == report["rust_report"]
    assert report["candidate_role_roots"] == EXPECTED_ROLE_ROOTS
    assert report["qualified_row_counts"] == {"odd": 480, "sink": 85}
    assert report["source_snapshot_stable_during_replay"] is True
    assert report["rust_execution"]["binary_sha256"].startswith("sha256:")
    assert report["rust_execution"]["binary_source_binding_claim"] is False
    assert report["rust_execution"]["listed_rust_sources_are_build_attestation"] is False
    assert report["formal_input_roots"] is None
    assert report["formal_roots_generated"] is False
    assert report["m3_execution_manifest_root"] is None
    assert report["authority_boundary"] == {
        "candidate_roots_are_formal_roots": False,
        "authoritative_root_generation": False,
        "seed_genesis_performed": False,
        "signature_claim": False,
        "m3_gate_delta": 0,
        "m3_gates_before": 14,
        "m3_gates_after": 14,
        "child_state": "NOT_RUN",
        "m3_start_authorized": False,
    }


def test_checked_in_v112_qualification_is_current(
) -> None:
    report = json.loads(CHECKED_IN.read_text(encoding="utf-8"))
    validate_checked_typed_rows_qualification_report(report)


def test_v112_qualification_rejects_root_or_authority_escalation(
    m25_v112_rust_binary: Path,
) -> None:
    report = dual_typed_rows_qualification_report(m25_v112_rust_binary)
    mutated = deepcopy(report)
    mutated["authority_boundary"]["candidate_roots_are_formal_roots"] = True
    with pytest.raises(M25QualificationError):
        validate_dual_typed_rows_qualification_report(
            mutated,
            m25_v112_rust_binary,
        )

    mutated = deepcopy(report)
    mutated["candidate_role_roots"]["odd"]["universe_root_hex"] = "00" * 32
    assert EXPECTED_ROLE_ROOTS["odd"]["universe_root_hex"] != "00" * 32
    with pytest.raises(M25QualificationError):
        validate_dual_typed_rows_qualification_report(
            mutated,
            m25_v112_rust_binary,
        )

    mutated = deepcopy(report)
    mutated["rust_execution"]["binary_source_binding_claim"] = True
    with pytest.raises(M25QualificationError):
        validate_dual_typed_rows_qualification_report(
            mutated,
            m25_v112_rust_binary,
        )


def test_v112_qualification_cli_emits_non_authoritative_artifact(
    tmp_path: Path,
    m25_v112_rust_binary: Path,
) -> None:
    output = tmp_path / "m25-v112-qualification.json"
    assert main(
        [
            "phase3-m25-v112-qualify",
            "--rust-binary",
            str(m25_v112_rust_binary),
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_kind"] == "DETERMINISTIC_CANDIDATE_NON_AUTHORITATIVE"
    assert payload["authority_boundary"]["m3_gate_delta"] == 0


def test_v112_qualification_rejects_json_numeric_type_confusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forged = deepcopy(complete_typed_rows_report_v1())
    forged["roles"][0]["row_count"] = 480.0
    monkeypatch.setattr(qualification_module, "_rust_report", lambda _binary: forged)
    with pytest.raises(M25QualificationError, match="Rust typed rows"):
        dual_typed_rows_qualification_report(Path("/bin/true"))


def test_v112_qualification_rejects_wrong_rust_operation_echo() -> None:
    response = {
        "ok": True,
        "op": "wrong_op",
        "machine_id": "hegel-old-dsl-v1.1.0",
        "preimage_hex": "00",
        "digest_hex": "00",
    }
    with pytest.raises(M25QualificationError, match="does not echo"):
        qualification_module._strip_rust_response_envelope(
            response,
            expected_operation="id_digest",
            exact_payload_fields=frozenset(
                {"machine_id", "preimage_hex", "digest_hex"}
            ),
        )


def test_v112_qualification_binds_top_level_child_dsl_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualification_module, "CHILD_DSL_ID", "hegel-old-dsl-v9.9.9")
    with pytest.raises(M25QualificationError, match="child DSL ID"):
        qualification_module._rust_report(Path("/bin/true"))


def test_v112_qualification_rejects_fixture_authority_escalation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = json.loads(
        qualification_module.GOLDEN_VECTOR_PATH.read_text(encoding="utf-8")
    )
    fixture["authority_boundary"]["formal_roots_generated"] = True
    tampered = tmp_path / "typed-rows.json"
    tampered.write_text(json.dumps(fixture), encoding="utf-8")
    monkeypatch.setattr(qualification_module, "GOLDEN_VECTOR_PATH", tampered)
    with pytest.raises(M25QualificationError, match="authority_boundary"):
        dual_typed_rows_qualification_report(Path("/bin/true"))


@pytest.mark.parametrize("mutation", ["count_float", "gate_float"])
def test_checked_qualification_rejects_json_numeric_type_confusion(
    mutation: str,
) -> None:
    report = json.loads(CHECKED_IN.read_text(encoding="utf-8"))
    if mutation == "count_float":
        report["qualified_row_counts"]["odd"] = 480.0
    else:
        report["authority_boundary"]["m3_gates_before"] = 14.0
    report.pop("diagnostic_report_id")
    report["diagnostic_report_id"] = stable_hash(
        report,
        prefix="phase3_m25_wire_completion_qualification_",
    )
    with pytest.raises(M25QualificationError):
        validate_checked_typed_rows_qualification_report(report)
