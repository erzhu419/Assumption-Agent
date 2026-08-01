from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

import pytest

from hegel_machine.cli import main
from hegel_machine.phase3_m25_external_v1 import (
    EXACT_ERRATA_BLOCKERS,
    ExternalGenesisPreflightError,
    FAIL_ACTOR_KEY_ID_COLLISION,
    FAIL_M25_EXACT_ERRATA_REQUIRED,
    FAIL_PROCESS_NONZERO_EXIT,
    FAIL_PUBLICATION_COMMIT_CONTAINS_IMPLEMENTATION_CHANGE,
    FAIL_PUBLIC_BUNDLE_SECRET_FIELD,
    FAIL_SECRET_FILE_PERMISSIONS,
    FAIL_SECRET_FILE_SIZE,
    FAIL_SECRET_PIPE_RUNTIME,
    FAIL_SECRET_STATE_INSIDE_REPOSITORY,
    FAIL_SECRET_STATE_PATH_INVALID,
    FAIL_SECRET_STATE_PERMISSIONS,
    FAIL_SPLIT_SEED_ALREADY_INSTANTIATED,
    FAIL_SPLIT_SEED_PENDING_EXTERNAL_RECOVERY_REQUIRED,
    MarkerSnapshot,
    assert_external_genesis_start_allowed,
    assert_marker_does_not_require_external_recovery,
    assert_public_payload_contains_no_secret_fields,
    assert_seed_instantiation_marker_absent,
    external_genesis_preflight_report,
    validate_calculator_process_result,
    validate_commit_b_changed_paths,
    validate_distinct_actor_key_ids,
    validate_external_genesis_preflight_report,
    validate_marker_snapshot,
    validate_secret_fd_number,
    validate_secret_fd_payload,
    validate_secret_file,
    validate_secret_state_directory,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKED_IN = ROOT / "artifacts" / "phase3_m25_external_preflight_v1.json"


def _assert_code(error: pytest.ExceptionInfo[ExternalGenesisPreflightError], code: str) -> None:
    assert error.value.code == code


@pytest.fixture
def posix_tmp_path() -> Path:
    """Use the WSL filesystem because DrvFS does not preserve 0700/0600 modes."""

    with tempfile.TemporaryDirectory(prefix="hegel-m25-external-", dir="/tmp") as value:
        yield Path(value)


def test_preflight_keeps_exact_12_blockers_and_14_of_24_not_run() -> None:
    report = external_genesis_preflight_report()
    validate_external_genesis_preflight_report(report)

    assert len(EXACT_ERRATA_BLOCKERS) == 12
    assert [blocker.blocker_id.split("_", 1)[0] for blocker in EXACT_ERRATA_BLOCKERS] == [
        f"E{index}" for index in range(1, 13)
    ]
    assert report["exact_errata_blocker_count"] == 12
    assert report["m3_gates_satisfied"] == 14
    assert report["m3_gates_total"] == 24
    assert report["child_state"] == "NOT_RUN"
    assert report["m3_entry_allowed"] is False
    assert report["m3_run_started"] is False
    assert report["external_genesis_start_allowed"] is False
    assert report["diagnostic_report_id"].startswith("phase3_m25_external_preflight_")

    effects = report["authority_side_effects"]
    assert isinstance(effects, dict)
    assert effects and all(value is False for value in effects.values())


def test_checked_in_external_preflight_is_current() -> None:
    report = json.loads(CHECKED_IN.read_text(encoding="utf-8"))
    validate_external_genesis_preflight_report(report)


def test_external_preflight_cli_publishes_stop_without_starting_genesis(
    tmp_path: Path,
) -> None:
    output = tmp_path / "external-preflight.json"
    assert main(["phase3-m25-external-preflight", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "EXACT_ERRATA_REQUIRED_EXTERNAL_GENESIS_BLOCKED"
    assert report["external_genesis_start_allowed"] is False
    assert all(value is False for value in report["authority_side_effects"].values())


def test_errata_registry_explicitly_covers_slot_topology_and_path_alias_details() -> None:
    by_id = {blocker.blocker_id: blocker for blocker in EXACT_ERRATA_BLOCKERS}
    assert "15 output slots" in by_id[
        "E1_M3_RUN_GENESIS_SLOT_CARDINALITY"
    ].title
    assert any(
        "step 25" in item and "step 30" in item
        for item in by_id["E2_BRIDGE_TOPOLOGY_ORDER"].evidence
    )

    e9 = by_id["E9_ROOT_PREIMAGES_INSTANCE_IDS_AND_PATH_ALIAS"]
    evidence = " ".join(e9.evidence)
    assert "TargetSpecFormal.claim_level_id" in evidence
    assert "MismatchRecord.mismatch_kind_id" in evidence
    assert "assignment_ordering_rule_id" in evidence
    assert "legal-transition" in evidence
    assert "static_role_metadata" in evidence
    assert "space forbidden by IdDigestV1" in evidence
    assert {
        "m3_run_state_exact_prefix",
        "input_signature_static_role_metadata_schema",
        "target_claim_level_enum_registry",
        "mismatch_kind_enum_registry",
        "split_contract_numeric_rule_registry",
        "traversal_and_bucket_field_id_registries",
    }.issubset(e9.required_machine_fields)


def test_preflight_report_rejects_self_consistent_authority_escalation() -> None:
    report = deepcopy(external_genesis_preflight_report())
    report["m3_gates_satisfied"] = 15
    with pytest.raises(ExternalGenesisPreflightError) as error:
        validate_external_genesis_preflight_report(report)
    _assert_code(error, FAIL_M25_EXACT_ERRATA_REQUIRED)


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("m3_gates_satisfied", 14.0),
        ("external_genesis_start_allowed", 0),
    ],
)
def test_preflight_rejects_json_numeric_type_confusion(
    field: str,
    forged: object,
) -> None:
    report = deepcopy(external_genesis_preflight_report())
    report[field] = forged
    with pytest.raises(ExternalGenesisPreflightError) as error:
        validate_external_genesis_preflight_report(report)
    _assert_code(error, FAIL_M25_EXACT_ERRATA_REQUIRED)


def test_external_start_guard_fails_before_csprng_or_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden_urandom(_length: int) -> bytes:
        calls.append("csprng")
        raise AssertionError("CSPRNG must not be reached")

    def forbidden_open(*_args, **_kwargs):
        calls.append("marker")
        raise AssertionError("marker creation must not be reached")

    monkeypatch.setattr(os, "urandom", forbidden_urandom)
    monkeypatch.setattr(os, "open", forbidden_open)

    with pytest.raises(ExternalGenesisPreflightError) as error:
        assert_external_genesis_start_allowed()
    _assert_code(error, FAIL_M25_EXACT_ERRATA_REQUIRED)
    assert calls == []


def test_secret_state_directory_must_be_external_nonsymlink_0700(
    posix_tmp_path: Path,
) -> None:
    tmp_path = posix_tmp_path
    repository = tmp_path / "repo"
    repository.mkdir(mode=0o700)
    external = tmp_path / "custody"
    external.mkdir(mode=0o700)
    external.chmod(0o700)

    assert validate_secret_state_directory(
        external,
        repository_root=repository,
    ) == external.resolve()

    inside = repository / "secret"
    inside.mkdir(mode=0o700)
    inside.chmod(0o700)
    with pytest.raises(ExternalGenesisPreflightError) as error:
        validate_secret_state_directory(inside, repository_root=repository)
    _assert_code(error, FAIL_SECRET_STATE_INSIDE_REPOSITORY)

    external.chmod(0o755)
    with pytest.raises(ExternalGenesisPreflightError) as error:
        validate_secret_state_directory(external, repository_root=repository)
    _assert_code(error, FAIL_SECRET_STATE_PERMISSIONS)

    external.chmod(0o700)
    alias = tmp_path / "custody-link"
    alias.symlink_to(external, target_is_directory=True)
    with pytest.raises(ExternalGenesisPreflightError) as error:
        validate_secret_state_directory(alias, repository_root=repository)
    _assert_code(error, FAIL_SECRET_STATE_PATH_INVALID)


def test_secret_file_must_be_inside_state_0600_and_exact_size(
    posix_tmp_path: Path,
) -> None:
    tmp_path = posix_tmp_path
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    state.chmod(0o700)
    seed = state / "seed.bin"
    seed.write_bytes(bytes(range(32)))
    seed.chmod(0o600)

    assert validate_secret_file(
        seed,
        secret_state_directory=state,
        expected_size=32,
    ) == seed.resolve()

    seed.chmod(0o644)
    with pytest.raises(ExternalGenesisPreflightError) as error:
        validate_secret_file(seed, secret_state_directory=state, expected_size=32)
    _assert_code(error, FAIL_SECRET_FILE_PERMISSIONS)

    seed.chmod(0o600)
    with pytest.raises(ExternalGenesisPreflightError) as error:
        validate_secret_file(seed, secret_state_directory=state, expected_size=31)
    _assert_code(error, FAIL_SECRET_FILE_SIZE)


@pytest.mark.parametrize("fd", [-1, 0, 2, 4, True, "3"])
def test_secret_transport_requires_exact_fd3(fd: object) -> None:
    with pytest.raises(ExternalGenesisPreflightError) as error:
        validate_secret_fd_number(fd)
    _assert_code(error, FAIL_SECRET_PIPE_RUNTIME)


@pytest.mark.parametrize("payload", [b"", bytes(31), bytes(33), bytearray(32), None])
def test_secret_transport_requires_exact_32_bytes_then_eof(payload: object) -> None:
    with pytest.raises(ExternalGenesisPreflightError) as error:
        validate_secret_fd_payload(payload)
    _assert_code(error, FAIL_SECRET_PIPE_RUNTIME)


def test_marker_second_invocation_and_pending_crash_fail_without_redraw() -> None:
    pending = MarkerSnapshot(
        state="PENDING",
        split_version_digest=bytes(range(32)),
        seed_commitment_manifest_root=None,
        custodian_key_id=bytes(range(16)),
        created_at_unix_seconds=1_704_067_200,
    )
    assert validate_marker_snapshot(pending) is pending

    with pytest.raises(ExternalGenesisPreflightError) as error:
        assert_seed_instantiation_marker_absent(marker_exists=True)
    _assert_code(error, FAIL_SPLIT_SEED_ALREADY_INSTANTIATED)

    with pytest.raises(ExternalGenesisPreflightError) as error:
        assert_marker_does_not_require_external_recovery(pending)
    _assert_code(error, FAIL_SPLIT_SEED_PENDING_EXTERNAL_RECOVERY_REQUIRED)

    complete = MarkerSnapshot(
        state="COMPLETE",
        split_version_digest=bytes(range(32)),
        seed_commitment_manifest_root=bytes(reversed(range(32))),
        custodian_key_id=bytes(range(16)),
        created_at_unix_seconds=1_704_067_201,
    )
    assert validate_marker_snapshot(complete) is complete
    assert_marker_does_not_require_external_recovery(complete)


def test_marker_state_root_xor_is_fail_closed() -> None:
    malformed = MarkerSnapshot(
        state="PENDING",
        split_version_digest=bytes(32),
        seed_commitment_manifest_root=bytes(32),
        custodian_key_id=bytes(16),
        created_at_unix_seconds=0,
    )
    with pytest.raises(ExternalGenesisPreflightError) as error:
        validate_marker_snapshot(malformed)
    _assert_code(error, FAIL_SECRET_STATE_PATH_INVALID)


def test_actor_key_ids_must_be_pairwise_distinct() -> None:
    values = validate_distinct_actor_key_ids(
        custodian_key_id=b"c" * 16,
        python_attester_key_id=b"p" * 16,
        rust_attester_key_id=b"r" * 16,
        auditor_key_id=b"a" * 16,
    )
    assert len(set(values)) == 4

    with pytest.raises(ExternalGenesisPreflightError) as error:
        validate_distinct_actor_key_ids(
            custodian_key_id=b"c" * 16,
            python_attester_key_id=b"c" * 16,
            rust_attester_key_id=b"r" * 16,
            auditor_key_id=b"a" * 16,
        )
    _assert_code(error, FAIL_ACTOR_KEY_ID_COLLISION)


def test_public_payload_allows_commitment_but_rejects_nested_secret_fields() -> None:
    public = {
        "seed_commitment_digest": "sha256:" + "00" * 32,
        "split_roots": ["sha256:" + "11" * 32],
    }
    assert_public_payload_contains_no_secret_fields(public)
    validate_calculator_process_result(exit_code=0, public_payload=public)

    with pytest.raises(ExternalGenesisPreflightError) as error:
        assert_public_payload_contains_no_secret_fields(
            {"nested": [{"master_seed_hex": "00" * 32}]}
        )
    _assert_code(error, FAIL_PUBLIC_BUNDLE_SECRET_FIELD)

    with pytest.raises(ExternalGenesisPreflightError) as error:
        validate_calculator_process_result(exit_code=9, public_payload=public)
    _assert_code(error, FAIL_PROCESS_NONZERO_EXIT)


def test_commit_b_diff_allows_public_artifacts_and_rejects_executable_changes() -> None:
    assert validate_commit_b_changed_paths(
        (
            "Hegel Machine/artifacts/phase3_m25_external/public-index.json",
            "Hegel Machine/docs/phase3_m25_external_status.md",
        ),
        allowed_public_prefixes=(
            "Hegel Machine/artifacts/phase3_m25_external",
            "Hegel Machine/docs/phase3_m25_external_status.md",
        ),
        executable_prefixes=(
            "Hegel Machine/src",
            "Hegel Machine/rust",
            "Hegel Machine/tests",
        ),
    )

    with pytest.raises(ExternalGenesisPreflightError) as error:
        validate_commit_b_changed_paths(
            ("Hegel Machine/src/hegel_machine/phase3_m25_wire_v1.py",),
            allowed_public_prefixes=("Hegel Machine/artifacts",),
            executable_prefixes=("Hegel Machine/src",),
        )
    _assert_code(error, FAIL_PUBLICATION_COMMIT_CONTAINS_IMPLEMENTATION_CHANGE)
