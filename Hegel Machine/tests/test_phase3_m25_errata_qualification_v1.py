from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from hegel_machine import phase3_m25_errata_qualification_v1 as qualification
from hegel_machine.cli import main
from hegel_machine.hashing import stable_hash
from hegel_machine.phase3_m25_errata_qualification_v1 import (
    EXPECTED_VECTOR_COUNTS,
    M25ErrataQualificationError,
    dual_errata_qualification_report,
    validate_checked_errata_qualification_report,
    validate_dual_errata_qualification_report,
)
from hegel_machine.phase3_m25_errata_vectors_v1 import (
    generate_errata_vector_report_v1,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKED_IN = ROOT / "artifacts" / "phase3_m25_errata_qualification_v1.json"


@pytest.fixture(scope="session")
def m25_errata_report() -> dict[str, object]:
    try:
        qualification._approved_rust_toolchain()
    except M25ErrataQualificationError as exc:
        pytest.skip(f"approved local Rust toolchain is required: {exc}")
    return dual_errata_qualification_report()


def test_python_errata_report_matches_checked_golden_exactly() -> None:
    fixture = json.loads(qualification.GOLDEN_VECTOR_PATH.read_text(encoding="utf-8"))
    python_response = generate_errata_vector_report_v1()
    python_core = {
        field: python_response[field]
        for field in qualification.REPORT_CORE_FIELDS
    }
    assert fixture["report"] == python_core
    report = fixture["report"]
    assert {field: len(report[field]) for field in EXPECTED_VECTOR_COUNTS} == (
        EXPECTED_VECTOR_COUNTS
    )


def test_dual_errata_qualification_matches_golden_and_only_authorizes_external_step(
    m25_errata_report: dict[str, object],
) -> None:
    report = m25_errata_report
    validate_checked_errata_qualification_report(report)
    assert report["status"] == "DUAL_EXACT_WIRE_ERRATA_GOLDEN_PASS"
    assert report["python_report"] == report["rust_report"]
    assert report["cross_language_exact_match"] is True
    assert report["golden_exact_match"] is True
    assert report["dual_golden_start_guard"]["passed_check_count"] == 10
    assert report["dual_golden_start_guard"]["external_genesis_start_allowed"] is True
    assert report["external_genesis_start_authorization"] == {
        "authorization_is_side_effect_free": True,
        "external_genesis_start_allowed": True,
        "m3_gates_satisfied": 14,
        "m3_gates_total": 24,
        "child_state": "NOT_RUN",
        "gate24_qualified": False,
        "m3_entry_allowed": False,
        "m3_run_started": False,
        "phase3_m3_start_authorized": False,
        "checked_artifact_replay_alone_sufficient": False,
        "fresh_dual_replay_required": True,
    }
    assert report["formal_input_roots"] is None
    assert report["formal_roots_generated"] is False
    assert report["m3_execution_manifest_root"] is None
    assert report["authority_boundary"]["m3_gate_delta"] == 0
    assert report["authority_boundary"]["child_state"] == "NOT_RUN"
    assert report["rust_execution"]["binary_source_binding_claim"] is True
    assert report["rust_execution"]["fresh_target_directory"] is True
    assert report["rust_execution"]["working_tree_built"] is False
    assert report["rust_execution"]["inherited_environment_allowed"] is False
    assert report["rust_execution"]["binary_hash_and_exec_same_open_inode"] is True
    assert report["rust_execution"]["cargo_lock_registry_archives_verified"] is True
    assert report["rust_execution"]["cargo_lock_registry_archive_count"] > 0
    assert (
        report["rust_execution"]["ancestor_cargo_config_absence_verified"]
        is True
    )
    assert report["python_execution"]["working_tree_executed"] is False
    assert report["python_execution"]["minimal_module_closure"] is True
    assert report["python_execution"]["package_init_executed"] is False
    assert report["repository_secret_absence_receipt"]["pass"] is True


def test_checked_in_errata_qualification_is_current() -> None:
    if not CHECKED_IN.is_file():
        pytest.skip("qualification artifact is emitted only after Commit A")
    report = json.loads(CHECKED_IN.read_text(encoding="utf-8"))
    validate_checked_errata_qualification_report(report)


def test_errata_qualification_rejects_authority_escalation(
    m25_errata_report: dict[str, object],
) -> None:
    report = m25_errata_report
    mutated = deepcopy(report)
    mutated["authority_boundary"]["formal_roots_generated"] = True
    with pytest.raises(M25ErrataQualificationError):
        qualification._validate_report_envelope(mutated)

    mutated = deepcopy(report)
    mutated["external_genesis_start_authorization"]["m3_entry_allowed"] = True
    with pytest.raises(M25ErrataQualificationError):
        qualification._validate_report_envelope(mutated)

    mutated = deepcopy(report)
    mutated["rust_execution"]["binary_source_binding_claim"] = False
    with pytest.raises(M25ErrataQualificationError):
        qualification._validate_report_envelope(mutated)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("qualified_vector_counts", "objects"), 21.0),
        (("authority_boundary", "m3_gates_before"), 14.0),
        (("authority_boundary", "external_genesis_start_authorized"), 1),
    ],
)
def test_errata_qualification_rejects_json_numeric_type_confusion(
    m25_errata_report: dict[str, object],
    path: tuple[str, str],
    value: object,
) -> None:
    report = m25_errata_report
    mutated = deepcopy(report)
    parent = mutated[path[0]]
    assert isinstance(parent, dict)
    parent[path[1]] = value
    mutated.pop("diagnostic_report_id")
    mutated["diagnostic_report_id"] = stable_hash(
        mutated, prefix="phase3_m25_errata_qualification_"
    )
    with pytest.raises(M25ErrataQualificationError):
        qualification._validate_report_envelope(mutated)


def test_qualification_rejects_dirty_or_uncommitted_commit_a_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uncommitted = tmp_path / "not-in-commit-a.json"
    uncommitted.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(qualification, "SOURCE_PATHS", (str(uncommitted),))
    with pytest.raises(M25ErrataQualificationError):
        qualification._assert_sources_match_commit(
            qualification.repository_head_commit()
        )


def test_qualification_output_cannot_overwrite_a_bound_input() -> None:
    with pytest.raises(M25ErrataQualificationError):
        qualification.validate_errata_qualification_output_path(
            qualification.GOLDEN_VECTOR_PATH
        )


def test_public_qualification_api_rejects_caller_supplied_toolchain() -> None:
    with pytest.raises(TypeError):
        dual_errata_qualification_report(cargo_executable="/tmp/fake-cargo")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        validate_dual_errata_qualification_report(  # type: ignore[call-arg]
            {},
            cargo_executable="/tmp/fake-cargo",
        )


def test_isolated_registry_copies_only_lock_verified_archives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"synthetic crate archive"
    checksum = hashlib.sha256(payload).hexdigest()
    source_home = tmp_path / "source-cargo"
    cache = source_home / "registry" / "cache" / "synthetic-index"
    index = source_home / "registry" / "index" / "synthetic-index"
    cache.mkdir(parents=True)
    index.mkdir(parents=True)
    (cache / "demo-1.2.3.crate").write_bytes(payload)
    lock = tmp_path / "Cargo.lock"
    lock.write_text(
        "version = 3\n\n"
        "[[package]]\n"
        'name = "demo"\n'
        'version = "1.2.3"\n'
        'source = "registry+https://example.invalid/index"\n'
        f'checksum = "{checksum}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("CARGO_HOME", str(source_home))
    destination = tmp_path / "isolated-cargo"
    manifest, count = qualification._copy_offline_cargo_registry(
        destination,
        cargo_lock=lock,
    )
    assert count == 1
    assert manifest == qualification._cargo_registry_input_manifest(
        destination / "registry"
    )
    assert not (destination / "registry" / "src").exists()
    assert (
        destination
        / "registry"
        / "cache"
        / "synthetic-index"
        / "demo-1.2.3.crate"
    ).read_bytes() == payload


def test_isolated_registry_rejects_archive_checksum_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_home = tmp_path / "source-cargo"
    cache = source_home / "registry" / "cache" / "synthetic-index"
    index = source_home / "registry" / "index" / "synthetic-index"
    cache.mkdir(parents=True)
    index.mkdir(parents=True)
    (cache / "demo-1.2.3.crate").write_bytes(b"tampered")
    lock = tmp_path / "Cargo.lock"
    lock.write_text(
        "version = 3\n\n"
        "[[package]]\n"
        'name = "demo"\n'
        'version = "1.2.3"\n'
        'source = "registry+https://example.invalid/index"\n'
        f'checksum = "{hashlib.sha256(b"expected").hexdigest()}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("CARGO_HOME", str(source_home))
    with pytest.raises(M25ErrataQualificationError):
        qualification._copy_offline_cargo_registry(
            tmp_path / "isolated-cargo",
            cargo_lock=lock,
        )


def test_qualification_rejects_visible_ancestor_cargo_config(tmp_path: Path) -> None:
    cwd = tmp_path / "snapshot" / "project"
    cargo_home = tmp_path / "cargo-home"
    cwd.mkdir(parents=True)
    cargo_home.mkdir()
    qualification._assert_no_cargo_config(cwd, cargo_home)
    config = tmp_path / "snapshot" / ".cargo" / "config.toml"
    config.parent.mkdir()
    config.write_text("[net]\noffline = false\n", encoding="utf-8")
    with pytest.raises(M25ErrataQualificationError):
        qualification._assert_no_cargo_config(cwd, cargo_home)


def test_errata_qualification_cli_emits_non_authoritative_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "errata-qualification.json"
    assert main(
        [
            "phase3-m25-errata-qualify",
            "--output",
            str(output),
        ]
    ) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["artifact_kind"] == "DETERMINISTIC_CANDIDATE_NON_AUTHORITATIVE"
    assert report["dual_golden_start_guard"]["external_genesis_start_allowed"] is True
    assert report["authority_boundary"]["external_genesis_started"] is False
    assert report["authority_boundary"]["m3_run_started"] is False


def test_fresh_dual_validator_replays_commit_a_sources(
    m25_errata_report: dict[str, object],
) -> None:
    validate_dual_errata_qualification_report(m25_errata_report)
