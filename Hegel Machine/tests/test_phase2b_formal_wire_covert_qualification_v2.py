from __future__ import annotations

import ast
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import hashlib
import io
import inspect
import json
import math
import os
import random
import signal
import struct
import subprocess
import sys
import tarfile
import time

import pytest

from hegel_machine import phase2b_formal_covert_auditor_v1 as auditor
from hegel_machine import phase2b_formal_trusted_wire_v3 as wire
from hegel_machine import phase2b_formal_wire_covert_qualification_v2 as qualification


ROOT = Path(__file__).parents[1]


@lru_cache(maxsize=2)
def _fixture(mode: str) -> qualification._FixtureExecutionV2:
    return qualification._build_fixture_execution_v2(mode, row_count=8)


def _attempt() -> qualification.Phase2BFormalWireCovertAttemptManifestV2:
    digest = "0" * 64
    return qualification.Phase2BFormalWireCovertAttemptManifestV2(
        "campaign-1",
        "attempt-1",
        qualification.PROFILE_ID,
        "0" * 40,
        "sha256:" + digest,
        1,
        qualification.DEPENDENCY_LOCK_SHA256,
        digest,
        digest,
        digest,
        digest,
        digest,
        "sha256:" + digest,
        1,
        None,
        "sha256:" + digest,
        digest,
        1,
        qualification.LIVE_MODE_ORDER,
        qualification.LIVE_HARD_TOTAL_TIMEOUT_SECONDS,
        auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
        64,
    )


def _write_pinned_source_archive(
    archive: Path,
    *,
    source_commit_sha: str,
    overrides: dict[str, bytes] | None = None,
) -> None:
    overrides = {} if overrides is None else overrides
    expected_paths = sorted(qualification._expected_live_file_sha256_v2(_attempt()))
    with tarfile.open(
        archive,
        "w",
        format=tarfile.PAX_FORMAT,
        pax_headers={"comment": source_commit_sha},
    ) as bundle:
        for relative in expected_paths:
            payload = overrides.get(relative, (ROOT / relative).read_bytes())
            member = tarfile.TarInfo(relative)
            member.mode = 0o600
            member.mtime = 0
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))


def _executable_row(path: str) -> dict[str, object]:
    return {
        "execution_path_security": {
            "caller_effective_uid": 0,
            "checked_components_in_order": [
                {
                    "mode_octal": "0755",
                    "owner_uid": 0,
                    "realpath": path,
                    "type": "regular",
                }
            ],
            "root_owned_no_group_or_other_write": True,
            "same_uid_path_replacement_excluded": True,
        },
        "realpath": path,
        "sha256": "1" * 64,
        "version": "test-version",
    }


def _systemd_show_bytes(
    attempt: qualification.Phase2BFormalWireCovertAttemptManifestV2,
    source_tree: Path,
    *,
    marker: str = "",
    invocation: str = "",
    load: str = "not-found",
    active: str = "inactive",
    sub: str = "dead",
    main_pid: int = 0,
    cgroup: str = "",
    result: str = "success",
) -> bytes:
    values = {
        "ActiveState": active,
        "ControlGroup": cgroup,
        "Description": marker,
        "ExecMainCode": "1",
        "ExecMainStatus": "0",
        "FinalKillSignal": "9",
        "InvocationID": invocation,
        "KillMode": "control-group",
        "KillSignal": "15",
        "LoadState": load,
        "MainPID": str(main_pid),
        "MemoryMax": str(attempt.requested_cgroup_memory_max_bytes),
        "RemainAfterExit": "yes",
        "Result": result,
        "RuntimeMaxUSec": str(
            qualification.LIVE_HARD_TOTAL_TIMEOUT_SECONDS * 1_000_000
        ) + "us",
        "SendSIGKILL": "yes",
        "SubState": sub,
        "TasksMax": str(attempt.requested_cgroup_pids_max),
        "TimeoutStopUSec": "30000000us",
        "Type": "exec",
        "WorkingDirectory": str(source_tree),
    }
    return "".join(f"{key}={values[key]}\n" for key in qualification._SYSTEMD_SHOW_PROPERTIES_V2).encode()


def _launch_intent() -> tuple[
    qualification.Phase2BFormalWireCovertAttemptManifestV2,
    Path,
    dict[str, object],
]:
    attempt = _attempt()
    source_tree = Path("/tmp/hegel-qv2-attempt/evidence/source-tree")
    python_row = _executable_row("/usr/bin/python3")
    systemd_run_row = _executable_row("/usr/bin/systemd-run")
    systemctl_row = _executable_row("/usr/bin/systemctl")
    nonce = "2" * 32
    entry = "development-calibration"
    unit = "hegel-qv2-" + hashlib.sha256(
        (
            attempt.attempt_manifest_root + "\0" + entry + "\0" + nonce
        ).encode("ascii")
    ).hexdigest()[:32] + ".service"
    child_argv = qualification._source_execution_bootstrap_argv_v2(
        python_realpath=python_row["realpath"],
        project_root=source_tree,
        qualification_source_sha256=attempt.qualification_source_sha256,
        source_arguments=[
            "--entry",
            entry,
            "--profile",
            str(source_tree.parent / "inputs/profile.json"),
            "--attempt",
            str(source_tree.parent / "inputs/attempt.json"),
        ],
    )
    child_sha = hashlib.sha256(
        qualification.canonicalize_rfc8785_v1(child_argv)
    ).hexdigest()
    marker = "hegel-qv2-owned-" + hashlib.sha256(
        b"HEGEL/PHASE2B/QV2/UNIT_OWNERSHIP/V2\0"
        + attempt.attempt_manifest_root.encode("ascii")
        + bytes.fromhex(nonce)
        + bytes.fromhex(child_sha)
    ).hexdigest()
    systemd_argv = [
        systemd_run_row["realpath"],
        "--user", "--pipe", "--wait", "--quiet",
        f"--unit={unit.removesuffix('.service')}",
        f"--property=Description={marker}",
        "--property=Type=exec",
        "--property=RemainAfterExit=yes",
        f"--property=MemoryMax={attempt.requested_cgroup_memory_max_bytes}",
        f"--property=TasksMax={attempt.requested_cgroup_pids_max}",
        f"--property=RuntimeMaxSec={qualification.LIVE_HARD_TOTAL_TIMEOUT_SECONDS}s",
        "--property=KillMode=control-group",
        "--property=KillSignal=SIGTERM",
        "--property=SendSIGKILL=yes",
        "--property=FinalKillSignal=SIGKILL",
        "--property=TimeoutStopSec=30s",
        f"--working-directory={source_tree}",
        "--setenv=LANG=C.UTF-8", "--setenv=LC_ALL=C.UTF-8", "--",
        *child_argv,
    ]
    prelaunch = qualification._command_evidence_v2(
        argv=qualification._systemd_show_argv_v2(systemctl_row["realpath"], unit),
        purpose="prelaunch-absent-check",
        returncode=0,
        stdout=_systemd_show_bytes(attempt, source_tree),
        stderr=b"",
    )
    return attempt, source_tree, {
        "schema_version": qualification.OUTER_LAUNCH_INTENT_SCHEMA_V2,
        "attempt_manifest_root": attempt.attempt_manifest_root,
        "entry_kind": entry,
        "profile_id": attempt.profile_id,
        "source_tree_root": str(source_tree),
        "runtime_environment_manifest_root": attempt.runtime_environment_manifest_root,
        "runtime_environment_manifest_size_bytes": attempt.runtime_environment_manifest_size_bytes,
        "executables": {
            "python": python_row,
            "systemd_run": systemd_run_row,
            "systemctl": systemctl_row,
        },
        "ownership_nonce": nonce,
        "unit_name": unit,
        "ownership_marker": marker,
        "child_argv": child_argv,
        "child_argv_sha256": child_sha,
        "systemd_argv": systemd_argv,
        "prelaunch_command": prelaunch,
        "prior_calibration_attempt_pair": None,
    }


def test_checked_profile_is_exact_canonical_candidate_with_live_disabled() -> None:
    payload = qualification.phase2b_formal_wire_covert_qualification_profile_bytes_v2()
    assert payload == (ROOT / "config/phase2b_formal_wire_covert_qualification_v2.json").read_bytes()
    profile = qualification.load_phase2b_formal_wire_covert_qualification_profile_v2(payload)
    assert profile.profile_id == qualification.PROFILE_ID
    shape = profile.profile["fixture_contract"]["rich_positive_shape"]
    assert shape["calibration_campaign_status"] == "PENDING_FULL_1024_DEVELOPMENT_MEASUREMENT"
    assert shape["live_qualification_enabled"] is False
    assert shape["old_identity_only_counts_valid"] is False
    assert {
        name for name, value in shape.items() if value is None
    } == {
        "calibration_artifact_cas_root",
        "calibration_artifact_sha256",
        "calibration_artifact_size_bytes",
        "calibration_attempt_id",
        "calibration_attempt_manifest_cas_root",
        "calibration_attempt_manifest_root",
        "calibration_attempt_manifest_sha256",
        "calibration_attempt_manifest_size_bytes",
        "calibration_basis_profile_cas_root",
        "calibration_basis_profile_id",
        "calibration_basis_profile_sha256",
        "calibration_basis_profile_size_bytes",
        "calibration_campaign_id",
        "calibration_mechanics_artifact_cas_root",
        "calibration_mechanics_artifact_sha256",
        "calibration_mechanics_artifact_size_bytes",
        "calibration_outer_supervisor_result_cas_root",
        "calibration_outer_supervisor_result_sha256",
        "calibration_outer_supervisor_result_size_bytes",
        "calibration_rfc8785_receipt_cas_root",
        "calibration_rfc8785_receipt_sha256",
        "calibration_rfc8785_receipt_size_bytes",
        "combined_fixture_root",
        "known_leak_frozen_static_projection",
        "known_null_frozen_static_projection",
    }
    readiness = profile.profile["readiness_boundary"]
    assert "required_runtime_evidence_fields_in_order" not in readiness
    live = readiness["live_execution_contract"]
    assert list(live["attempt_manifest_fields"]) == list(
        qualification._attempt_manifest_projection_v2(_attempt())
    )
    assert live["attempt_manifest_fields_are_exact_top_level_projection"] is True
    assert live["hard_total_timeout_seconds"] == 28_800
    assert live["work_deadline_reserve_seconds"] == 3_000
    assert live["terminal_publication_reserve_seconds"] == 2_400
    assert live["cleanup_control_reserve_seconds"] == 600
    assert live["outer_mechanics_verifier_issues_r03_r04_r05"] is False
    assert live["task4_issues_r03_r04_r05"] is False
    assert live["systemd_user_manager_same_uid_is_cooperative_outer_tcb"] is True
    assert live["fresh_dedicated_user_service_manager_session_required"] is True
    assert live["outer_process_group_ownership_tcb"] == (
        "cooperative-pinned-direct-child-no-orphan-descendants"
    )
    assert live["adversarial_orphan_descendant_cleanup_claimed"] is False
    boundary = profile.profile["readiness_boundary"][
        "task4_projection_only_boundary"
    ]
    assert boundary == qualification._TASK4_PROJECTION_ONLY_BOUNDARY_V2
    encoded = payload.decode("utf-8")
    for stale in ("16863", "24367", "7991", "84315"):
        assert stale not in encoded


def test_dependency_lock_is_canonical_exact_and_closes_the_source_loader() -> None:
    payload = (
        ROOT / "config/phase2b_formal_wire_covert_qualification_v2.lock.json"
    ).read_bytes()
    assert payload.endswith(b"\n")
    assert b"\r" not in payload
    decoded = json.loads(payload)
    assert payload == qualification.canonicalize_rfc8785_v1(decoded) + b"\n"
    assert hashlib.sha256(payload).hexdigest() == qualification.DEPENDENCY_LOCK_SHA256
    runtime = qualification._verify_dependency_lock_bytes_v2(payload)
    assert runtime["python_implementation"] == "CPython"
    assert runtime["python_version"] == "3.10.12"
    assert runtime["python_abi"] == "cp310"
    assert runtime["lock_platform"] == "linux/amd64"
    assert set(decoded) == {"dependency_lock", "schema_version"}
    lock = decoded["dependency_lock"]
    assert lock["standard_library_only"] is True
    assert lock["standard_library_only_scope"] == "PYTHON_THIRD_PARTY_PACKAGES_ONLY"
    assert lock["host_runtime_manifest_scope"] == (
        "COOPERATIVE_FINGERPRINT_NOT_FULL_LIBPYTHON_STDLIB_NATIVE_CLOSURE"
    )
    assert lock["step5_full_runtime_image_and_import_native_closure_required"] is True
    assert lock["third_party_dependencies"] == []
    bootstrap = lock["execution_bootstrap"]
    assert bootstrap == {
        "broad_package_init_bypassed": True,
        "bypassed_package_init_sha256_non_tcb": (
            qualification.BYPASSED_PACKAGE_INIT_SHA256
        ),
        "direct_file_open_once_bootstrap_required": True,
        "python_argv_flags": ["-I", "-S", "-B"],
        "source_only_loader_required": True,
    }
    local = lock["project_local_dependencies"]
    assert local["runtime_import_closure"] == (
        qualification._BOOTSTRAP_FIXED_SOURCE_SHA256_V2
    )
    assert len(local["runtime_import_closure"]) == 7
    assert local["wire"] == {
        "config_sha256": qualification.WIRE_CONFIG_SHA256,
        "documentation_sha256": qualification.WIRE_DOCUMENTATION_SHA256,
        "source_sha256": qualification.WIRE_SOURCE_SHA256,
        "test_sha256": qualification.WIRE_TEST_SHA256,
    }
    assert local["auditor"] == {
        "config_sha256": qualification.AUDITOR_CONFIG_SHA256,
        "documentation_sha256": qualification.AUDITOR_DOCUMENTATION_SHA256,
        "policy_binding_id": auditor.POLICY_BINDING_ID,
        "semantics_id": auditor.SEMANTICS_ID,
        "source_sha256": qualification.AUDITOR_SOURCE_SHA256,
        "test_sha256": qualification.AUDITOR_TEST_SHA256,
    }
    profile_lock = qualification._EXPECTED_PROFILE["dependency_lock"]
    assert profile_lock["sha256"] == qualification.DEPENDENCY_LOCK_SHA256
    assert profile_lock["runtime_import_closure_sha256"] == (
        local["runtime_import_closure"]
    )


def test_direct_cli_requires_initial_isolation_and_reexecutes_open_once() -> None:
    source = (
        ROOT
        / "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py"
    ).resolve(strict=True)
    base_arguments = [
        "--entry",
        "rfc-selfcheck-version",
        "--project-root",
        str(ROOT.resolve(strict=True)),
    ]
    unisolated = subprocess.run(
        [sys.executable, str(source), *base_arguments],
        check=False,
        capture_output=True,
        timeout=20,
    )
    assert unisolated.returncode != 0
    assert unisolated.stdout == b""
    assert b"requires initial Python -I -S -B flags" in unisolated.stderr

    qualified = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(source), *base_arguments],
        check=False,
        capture_output=True,
        timeout=20,
    )
    assert qualified.returncode == 0
    assert qualified.stdout == b"hegel-rfc8785-stdlib-selfcheck-v1\n"
    assert qualified.stderr == b""

    tampered_argv = qualification._source_execution_bootstrap_argv_v2(
        python_realpath=os.path.realpath(sys.executable),
        project_root=ROOT.resolve(strict=True),
        qualification_source_sha256="0" * 64,
        source_arguments=base_arguments,
    )
    tampered = subprocess.run(
        tampered_argv,
        check=False,
        capture_output=True,
        timeout=20,
    )
    assert tampered.returncode != 0
    assert tampered.stdout == b""
    assert b"source bootstrap byte mismatch" in tampered.stderr

    current_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    dependency_tampered_argv = qualification._source_execution_bootstrap_argv_v2(
        python_realpath=os.path.realpath(sys.executable),
        project_root=ROOT.resolve(strict=True),
        qualification_source_sha256=current_sha256,
        source_arguments=base_arguments,
    )
    dependency_manifest = json.loads(dependency_tampered_argv[6])
    dependency_manifest["sha256"]["src/hegel_machine/hashing.py"] = "0" * 64
    dependency_tampered_argv[6] = json.dumps(
        dependency_manifest, separators=(",", ":"), sort_keys=True
    )
    dependency_tampered = subprocess.run(
        dependency_tampered_argv,
        check=False,
        capture_output=True,
        timeout=20,
    )
    assert dependency_tampered.returncode != 0
    assert dependency_tampered.stdout == b""
    assert b"source bootstrap byte mismatch" in dependency_tampered.stderr


def test_runtime_manifest_cli_emits_one_canonical_artifact_and_rejects_extras(
    tmp_path: Path,
) -> None:
    source = (
        ROOT
        / "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py"
    ).resolve(strict=True)
    argv = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(source),
        "--entry",
        "build-runtime-environment-manifest",
        "--project-root",
        str(ROOT.resolve(strict=True)),
    ]
    completed = subprocess.run(
        argv, check=False, capture_output=True, timeout=30
    )
    assert completed.returncode == 0
    assert completed.stderr == b""
    runtime = qualification._decode_canonical_transport_object_v2(
        completed.stdout,
        "runtime manifest CLI KAT",
        maximum_bytes=1_048_576,
    )
    assert runtime["schema_version"] == (
        qualification.RUNTIME_ENVIRONMENT_MANIFEST_SCHEMA_V2
    )
    assert runtime["host_runtime_fingerprint_only_not_full_runtime_image_closure"] is True

    rejected = subprocess.run(
        [*argv, "--calibration-attempt-directory", str(tmp_path.resolve())],
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert rejected.returncode != 0
    assert rejected.stdout == b""
    assert b"entry argument set is not exact" in rejected.stderr

    abbreviated = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(source),
            "--entry",
            "build-runtime-environment-manifest",
            "--project-r",
            str(ROOT.resolve(strict=True)),
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert abbreviated.returncode != 0
    assert abbreviated.stdout == b""
    assert b"unrecognized arguments" in abbreviated.stderr

    repeated = subprocess.run(
        [*argv, "--project-root", str(ROOT.resolve(strict=True))],
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert repeated.returncode != 0
    assert repeated.stdout == b""
    assert b"CLI option is repeated" in repeated.stderr

    relative = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(source),
            "--entry",
            "build-runtime-environment-manifest",
            "--project-root",
            ".",
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert relative.returncode != 0
    assert relative.stdout == b""
    assert b"must name an absolute path" in relative.stderr


def test_development_attempt_cli_parses_one_closed_argument_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    project_root = ROOT.resolve(strict=True)
    expected_artifact = qualification.canonicalize_rfc8785_v1(
        {"attempt": "closed"}
    ) + b"\n"
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        qualification,
        "_verify_qualified_cli_source_bootstrap_v2",
        lambda root: "1" * 64,
    )

    def build(**kwargs: object) -> bytes:
        observed.update(kwargs)
        return expected_artifact

    monkeypatch.setattr(
        qualification, "_build_development_attempt_manifest_bytes_v2", build
    )
    argv = (
        "--entry",
        "build-development-attempt-manifest",
        "--project-root",
        str(project_root),
        "--campaign-id",
        "campaign-cli",
        "--attempt-id",
        "attempt-cli",
        "--source-commit-sha",
        "a" * 40,
        "--source-archive",
        str((tmp_path / "source.tar").resolve()),
        "--runtime-environment-manifest",
        str((tmp_path / "runtime.json").resolve()),
        "--rfc8785-prerequisite-receipt",
        str((tmp_path / "rfc" / "receipt.json").resolve()),
        "--cgroup-memory-max-bytes",
        str(auditor.MAXIMUM_ESTIMATED_RSS_BYTES),
        "--cgroup-pids-max",
        "64",
    )
    assert qualification._main_v2(argv) == 0
    stdout, stderr = capfd.readouterr()
    assert stdout.encode() == expected_artifact
    assert stderr == ""
    assert observed == {
        "attempt_id": "attempt-cli",
        "campaign_id": "campaign-cli",
        "project_root": project_root,
        "requested_cgroup_memory_max_bytes": auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
        "requested_cgroup_pids_max": 64,
        "rfc8785_prerequisite_receipt": (tmp_path / "rfc" / "receipt.json").resolve(),
        "runtime_environment_manifest": (tmp_path / "runtime.json").resolve(),
        "source_archive": (tmp_path / "source.tar").resolve(),
        "source_commit_sha": "a" * 40,
    }


@pytest.mark.parametrize("tamper", ("source", "config"))
def test_attempt_bearing_cli_rejects_tcb_splice_before_outer_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    bootstrap_sha256 = "1" * 64
    attempt = _attempt()
    if tamper == "config":
        attempt = replace(
            attempt, qualification_source_sha256=bootstrap_sha256
        )
    profile_path = (tmp_path / "profile.json").resolve()
    profile_path.write_bytes(
        qualification.phase2b_formal_wire_covert_qualification_profile_bytes_v2()
    )
    attempt_path = (tmp_path / "attempt.json").resolve()
    attempt_path.write_bytes(
        qualification.phase2b_formal_wire_covert_attempt_manifest_bytes_v2(attempt)
    )
    output_root = (tmp_path / "output").resolve()
    output_root.mkdir()
    dummy = (tmp_path / "unused").resolve()
    monkeypatch.setattr(
        qualification,
        "_verify_qualified_cli_source_bootstrap_v2",
        lambda root: bootstrap_sha256,
    )
    monkeypatch.setattr(
        qualification,
        "run_phase2b_formal_wire_covert_outer_attempt_v2",
        lambda **kwargs: pytest.fail("TCB splice reached outer dispatch"),
    )
    expected = "executed source bytes" if tamper == "source" else "profile bytes"
    with pytest.raises(ValueError, match=expected):
        qualification._main_v2(
            (
                "--entry",
                "outer-development-calibration",
                "--profile",
                str(profile_path),
                "--attempt",
                str(attempt_path),
                "--rfc8785-prerequisite-receipt",
                str(dummy),
                "--source-archive",
                str(dummy),
                "--runtime-environment-manifest",
                str(dummy),
                "--output-root",
                str(output_root),
            )
        )
    assert list(output_root.iterdir()) == []


def test_development_attempt_builder_binds_five_files_archive_runtime_and_rfc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = ROOT.resolve(strict=True)
    source = project_root / qualification._BOOTSTRAP_MAIN_RELATIVE_PATH_V2
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    source_commit_sha = "a" * 40
    archive = (tmp_path / "source.tar").resolve()
    _write_pinned_source_archive(
        archive, source_commit_sha=source_commit_sha
    )
    runtime_bytes = qualification.canonicalize_rfc8785_v1(
        {"schema_version": "test-runtime-manifest"}
    ) + b"\n"
    runtime_path = (tmp_path / "runtime.json").resolve()
    runtime_path.write_bytes(runtime_bytes)
    raw_bundle = (tmp_path / "rfc-raw").resolve()
    raw_bundle.mkdir()
    rfc_bytes = qualification.canonicalize_rfc8785_v1(
        {"schema_version": "test-rfc-receipt"}
    ) + b"\n"
    receipt_path = raw_bundle / "receipt.json"
    receipt_path.write_bytes(rfc_bytes)
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        qualification,
        "_verify_qualified_cli_source_bootstrap_v2",
        lambda root: source_sha256 if root == project_root else pytest.fail("root"),
    )
    monkeypatch.setattr(
        qualification,
        "phase2b_formal_wire_covert_runtime_environment_manifest_bytes_v2",
        lambda: runtime_bytes,
    )

    def verify_receipt(
        payload: bytes,
        attempt: qualification.Phase2BFormalWireCovertAttemptManifestV2,
        **kwargs: object,
    ) -> dict[str, object]:
        observed["receipt_payload"] = payload
        observed["attempt"] = attempt
        observed["receipt_kwargs"] = kwargs
        return {"closed": "receipt"}

    def verify_raw(
        receipt: dict[str, object],
        bundle_directory: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        observed["raw_receipt"] = receipt
        observed["raw_directory"] = bundle_directory
        observed["raw_kwargs"] = kwargs
        return {"closed": "raw"}

    monkeypatch.setattr(
        qualification, "_verify_rfc8785_prerequisite_receipt_v2", verify_receipt
    )
    monkeypatch.setattr(
        qualification, "_verify_rfc8785_raw_execution_bundle_v2", verify_raw
    )
    artifact = qualification._build_development_attempt_manifest_bytes_v2(
        campaign_id="campaign-cli",
        attempt_id="attempt-cli",
        source_commit_sha=source_commit_sha,
        source_archive=archive,
        runtime_environment_manifest=runtime_path,
        rfc8785_prerequisite_receipt=receipt_path,
        requested_cgroup_memory_max_bytes=auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
        requested_cgroup_pids_max=64,
        project_root=project_root,
    )
    attempt = qualification.load_phase2b_formal_wire_covert_attempt_manifest_v2(
        artifact
    )
    assert attempt.source_commit_sha == source_commit_sha
    assert attempt.source_archive_root == "sha256:" + hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()
    assert attempt.runtime_environment_manifest_root == "sha256:" + hashlib.sha256(
        runtime_bytes
    ).hexdigest()
    assert attempt.rfc8785_prerequisite_receipt_sha256 == hashlib.sha256(
        rfc_bytes
    ).hexdigest()
    for role, relative in qualification._QUALIFICATION_TRANSACTION_RELATIVE_PATHS_V2.items():
        assert getattr(
            attempt,
            f"qualification_{'documentation' if role == 'documentation' else role}_sha256",
        ) == hashlib.sha256((project_root / relative).read_bytes()).hexdigest()
    assert observed["receipt_payload"] == rfc_bytes
    assert observed["raw_receipt"] == {"closed": "receipt"}
    assert observed["raw_directory"] == raw_bundle
    assert observed["raw_kwargs"]["fresh_execute"] is False
    assert observed["raw_kwargs"]["replay_project_root"] == project_root


@pytest.mark.parametrize("tamper", ("commit", "source"))
def test_development_attempt_builder_rejects_archive_tamper_before_live_runners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    project_root = ROOT.resolve(strict=True)
    source = project_root / qualification._BOOTSTRAP_MAIN_RELATIVE_PATH_V2
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    requested_commit = "a" * 40
    archive = (tmp_path / f"source-{tamper}.tar").resolve()
    _write_pinned_source_archive(
        archive,
        source_commit_sha=("b" * 40 if tamper == "commit" else requested_commit),
        overrides=(
            {qualification._BOOTSTRAP_MAIN_RELATIVE_PATH_V2: b"tampered source\n"}
            if tamper == "source"
            else None
        ),
    )
    runtime_bytes = qualification.canonicalize_rfc8785_v1(
        {"schema_version": "test-runtime-manifest"}
    ) + b"\n"
    runtime_path = (tmp_path / "runtime.json").resolve()
    runtime_path.write_bytes(runtime_bytes)
    raw_bundle = (tmp_path / "rfc-raw").resolve()
    raw_bundle.mkdir()
    receipt_path = raw_bundle / "receipt.json"
    receipt_path.write_bytes(b"{}\n")
    monkeypatch.setattr(
        qualification,
        "_verify_qualified_cli_source_bootstrap_v2",
        lambda root: source_sha256,
    )
    monkeypatch.setattr(
        qualification,
        "phase2b_formal_wire_covert_runtime_environment_manifest_bytes_v2",
        lambda: pytest.fail("tampered archive reached runtime probe"),
    )
    monkeypatch.setattr(
        qualification,
        "_verify_rfc8785_prerequisite_receipt_v2",
        lambda *args, **kwargs: pytest.fail("tampered archive reached RFC probe"),
    )
    with pytest.raises(ValueError, match=("PAX commit" if tamper == "commit" else "pinned-file hash")):
        qualification._build_development_attempt_manifest_bytes_v2(
            campaign_id="campaign-cli",
            attempt_id="attempt-cli",
            source_commit_sha=requested_commit,
            source_archive=archive,
            runtime_environment_manifest=runtime_path,
            rfc8785_prerequisite_receipt=receipt_path,
            requested_cgroup_memory_max_bytes=auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
            requested_cgroup_pids_max=64,
            project_root=project_root,
        )


def test_candidate_entrypoint_rejects_atomically_before_any_full_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = qualification.load_phase2b_formal_wire_covert_qualification_profile_v2(
        qualification.phase2b_formal_wire_covert_qualification_profile_bytes_v2()
    )
    monkeypatch.setattr(
        qualification,
        "_execute_isolated_audit_v2",
        lambda mode: pytest.fail(f"unexpected full audit: {mode}"),
    )
    result = qualification.qualify_phase2b_formal_wire_covert_candidate_v2(profile)
    assert type(result) is qualification.Phase2BFormalWireCovertQualificationRejectionV2
    assert result.completed_audit_runs == 0
    assert result.qualification_receipt_id is None
    assert result.reason_code == qualification.FAIL_PROFILE


def test_exact_label_factorial_and_transform_quota_are_locked() -> None:
    labels = qualification._synthetic_label_inputs_v2()
    cells = Counter(
        (
            row.case_type,
            row.family,
            row.scale,
            row.binding,
            row.answerability.value,
            row.joint_decision_class,
        )
        for row in labels
    )
    assert len(labels) == 1024
    assert len(cells) == 64
    assert set(cells.values()) == {16}
    operations = Counter(
        qualification._synthetic_source_case_v2(index, binding_leak=False)
        .graph.transforms[0]
        .operation
        for index in range(1024)
    )
    assert operations == Counter(
        {operation: 128 for operation in qualification.EXACT_TRANSFORM_OPERATIONS}
    )
    for cell in cells:
        row_indices = [
            row.row_index
            for row in labels
            if (
                row.case_type,
                row.family,
                row.scale,
                row.binding,
                row.answerability.value,
                row.joint_decision_class,
            )
            == cell
        ]
        assert Counter((index // 64) % 8 for index in row_indices) == Counter(
            {index: 2 for index in range(8)}
        )


def test_known_leak_changes_only_eligibility_observation_ids_for_all_rows() -> None:
    for index in range(1024):
        null = qualification._synthetic_source_case_v2(index, binding_leak=False)
        leak = qualification._synthetic_source_case_v2(index, binding_leak=True)
        restored = wire.FormalSourceCaseV3(
            replace(
                leak.graph,
                eligibility_mask=replace(
                    leak.graph.eligibility_mask,
                    observation_ids=null.graph.eligibility_mask.observation_ids,
                ),
            )
        )
        assert restored == null
        assert (
            leak.graph.eligibility_mask.role_candidate_ids
            == null.graph.eligibility_mask.role_candidate_ids
        )
        assert (
            leak.graph.eligibility_mask.scale_ids
            == null.graph.eligibility_mask.scale_ids
        )
        binding = qualification._authority_label_input_v2(index).binding
        assert len(leak.graph.eligibility_mask.observation_ids) == (
            1 if binding == "binding-a" else 2
        )
        assert len(null.graph.eligibility_mask.observation_ids) >= 2


def test_known_leak_mask_role_or_scale_splice_is_not_hidden_by_delta_replay() -> None:
    null = qualification._synthetic_source_case_v2(0, binding_leak=False)
    leak = qualification._synthetic_source_case_v2(0, binding_leak=True)
    extra_role = "00000000-0000-4000-8000-000000000001"
    role_registry = tuple(sorted((*null.graph.role_candidate_ids, extra_role)))
    null_with_extra_role = wire.FormalSourceCaseV3(
        replace(null.graph, role_candidate_ids=role_registry)
    )
    leak_with_extra_role = wire.FormalSourceCaseV3(
        replace(
            leak.graph,
            role_candidate_ids=role_registry,
            eligibility_mask=replace(
                leak.graph.eligibility_mask,
                role_candidate_ids=(extra_role,),
            ),
        )
    )
    role_restored = wire.FormalSourceCaseV3(
        replace(
            leak_with_extra_role.graph,
            eligibility_mask=replace(
                leak_with_extra_role.graph.eligibility_mask,
                observation_ids=(
                    null_with_extra_role.graph.eligibility_mask.observation_ids
                ),
            ),
        )
    )
    assert role_restored != null_with_extra_role

    scale_tampered = wire.FormalSourceCaseV3(
        replace(
            leak.graph,
            eligibility_mask=replace(
                leak.graph.eligibility_mask,
                scale_ids=leak.graph.eligibility_mask.scale_ids[:-1],
            ),
        )
    )
    scale_restored = wire.FormalSourceCaseV3(
        replace(
            scale_tampered.graph,
            eligibility_mask=replace(
                scale_tampered.graph.eligibility_mask,
                observation_ids=null.graph.eligibility_mask.observation_ids,
            ),
        )
    )
    assert scale_restored != null


@pytest.mark.parametrize("mode", qualification.LIVE_MODE_ORDER)
def test_lightweight_projection_equals_trusted_decode_for_base_and_transform(
    mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = qualification._build_fixture_execution_v2(mode, row_count=8)
    decoded = tuple(
        execution.adapter.decode_envelope(envelope)
        for envelope in execution.adapter.envelopes
    )
    original_run_batch = qualification._SyntheticRichRecognizerV2.run_batch
    verified_batch_sizes: list[int] = []

    def verified_run_batch(
        self: qualification._SyntheticRichRecognizerV2,
        envelopes: tuple[bytes, ...],
    ) -> tuple[auditor.FormalRecognizerDecisionV1, ...]:
        verified_batch_sizes.append(len(envelopes))
        for envelope in envelopes:
            trusted = self._adapter.decode_envelope(envelope)
            qualification._verify_lightweight_trusted_projection_equivalence_v2(
                envelope, trusted
            )
        return original_run_batch(self, envelopes)

    monkeypatch.setattr(
        qualification._SyntheticRichRecognizerV2,
        "run_batch",
        verified_run_batch,
    )
    for item in decoded:
        auditor._validate_decoded(item, execution.adapter)
    pins, _ = auditor._capture_adapter_execution_pins(
        execution.adapter, execution.recognizer
    )
    input_root = auditor._input_root(execution.adapter.envelopes)
    snapshot_root = auditor._invariance_input_snapshot_root(
        execution.adapter.envelopes,
        decoded,
        execution.adapter,
        execution.recognizer,
        input_root,
        pins,
    )
    result = auditor._run_invariance_core(
        execution.adapter.envelopes,
        decoded,
        execution.adapter,
        execution.recognizer,
        input_root,
        snapshot_root,
        pins,
    )
    assert result.invariant_run_count == 81
    assert verified_batch_sizes == [8] * 81
    assert execution.recognizer.row_counts == [8] * 81


def test_lightweight_trusted_projection_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _fixture("KNOWN_NULL")
    envelope = execution.adapter.envelopes[0]
    decoded = execution.adapter.decode_envelope(envelope)
    trusted = qualification._trusted_synthetic_public_projection_v2(decoded)
    monkeypatch.setattr(
        qualification,
        "_trusted_synthetic_public_projection_v2",
        lambda value: (trusted[0] + "-drift", trusted[1], trusted[2]),
    )
    with pytest.raises(ValueError, match="lightweight/trusted"):
        qualification._verify_lightweight_trusted_projection_equivalence_v2(
            envelope, decoded
        )


def _valid_target_family_rows() -> list[dict[str, object]]:
    exceedances = [0, 4, 4, 100, 10_000]
    raw = [Fraction(value + 1, 10_001) for value in exceedances]
    order = sorted(range(5), key=lambda index: (raw[index], index))
    ranks = [0] * 5
    adjusted = [Fraction(0, 1)] * 5
    running = Fraction(0, 1)
    for zero_rank, index in enumerate(order):
        ranks[index] = zero_rank + 1
        running = max(running, min(Fraction(1, 1), (5 - zero_rank) * raw[index]))
        adjusted[index] = running
    return [
        {
            "target": target,
            "exceedance_count": exceedances[index],
            "observed_target_max_omnibus_hex": float(index + 1).hex(),
            "raw_p": f"{raw[index].numerator}/{raw[index].denominator}",
            "holm_adjusted_p": (
                f"{adjusted[index].numerator}/{adjusted[index].denominator}"
            ),
            "holm_critical_value": (
                f"1/{100 * (5 - ranks[index] + 1)}"
            ),
            "holm_rank": ranks[index],
            "holm_rejected": adjusted[index] < Fraction(1, 100),
        }
        for index, target in enumerate(auditor.TARGETS)
    ]


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("exceedance_count", 1),
        ("raw_p", "2/10001"),
        ("holm_rank", 5),
        ("holm_adjusted_p", "1/1"),
        ("holm_critical_value", "1/100"),
        ("holm_rejected", False),
        ("observed_target_max_omnibus_hex", (-1.0).hex()),
    ),
)
def test_target_family_b_plus_one_holm_tamper_matrix_fails_closed(
    field: str, replacement: object
) -> None:
    rows = _valid_target_family_rows()
    qualification._validate_target_family_holm_projection_v2(
        rows, context="unit target family"
    )
    tampered = [dict(row) for row in rows]
    tampered[0][field] = replacement
    with pytest.raises(ValueError):
        qualification._validate_target_family_holm_projection_v2(
            tampered, context="unit target family"
        )


def test_complete_resource_projection_requires_exact_static_union_dynamic_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = set(qualification._RESOURCE_PLAN_FIELDS_V2)
    complete = {name: 0 for name in expected}

    class Replay:
        def __init__(self, **values: object) -> None:
            assert set(values) == expected

        def __post_init__(self) -> None:
            return None

    monkeypatch.setattr(auditor, "ResourcePlanV1", Replay)
    assert qualification._validate_complete_resource_plan_projection_v2(
        complete, context="unit resource"
    ) is complete
    for value in (
        {key: item for key, item in complete.items() if key != "admissible"},
        {**complete, "unexpected": 0},
    ):
        with pytest.raises(ValueError, match="fields are not exact"):
            qualification._validate_complete_resource_plan_projection_v2(
                value, context="unit resource"
            )


def test_known_leak_feature_max_t_plus_one_binding_is_replayed() -> None:
    rows = _valid_target_family_rows()
    binding = next(row for row in rows if row["target"] == "binding")
    binding["observed_target_max_omnibus_hex"] = (30.0).hex()
    alias = (
        "uuid_leaf/observation_id/"
        + qualification._json_path_text_v2(
            qualification._REGISTERED_BINDING_LEAK_OWNER_PATH_V2
        )
        + "/presence"
    )
    witness = {
        "balanced_accuracy_advantage_hex": (0.0).hex(),
        "bias_corrected_nmi_hex": (0.5).hex(),
        "failure_witness": True,
        "feature_aliases": [alias],
        "feature_id": "unit-feature",
        "feature_localization_rejected": True,
        "feature_max_t_exceedance_count": 5,
        "feature_max_t_p": "6/10001",
        "feature_owner": "eligibility_mask",
        "feature_owner_path": qualification._json_path_text_v2(
            qualification._REGISTERED_BINDING_LEAK_OWNER_PATH_V2
        ),
        "material_effect": True,
        "observed_omnibus_hex": (25.0).hex(),
        "passed": False,
        "raw_nmi_hex": (0.5).hex(),
        "registered_presence_alias": alias,
        "target_family_holm_rejected": True,
        "target_holm_critical_value": binding["holm_critical_value"],
    }
    qualification._validate_known_leak_witness_semantics_v2(
        witness,
        audit_row={"failure_witness_count": 1},
        target_rows=rows,
        context="unit witness",
    )
    tampered = dict(witness)
    tampered["feature_max_t_exceedance_count"] = 6
    with pytest.raises(ValueError, match="numeric witness"):
        qualification._validate_known_leak_witness_semantics_v2(
            tampered,
            audit_row={"failure_witness_count": 1},
            target_rows=rows,
            context="unit witness",
        )


def test_small_eight_operation_build_uses_final_fresh_api_and_exact_ten_namespaces() -> None:
    execution = _fixture("KNOWN_NULL")
    assert type(execution.build_success) is wire.FormalTrustedWireBuildSuccessV3
    assert len(execution.global_transform_catalog.entries) == 8
    assert execution.label_factorial_proof_root.startswith(
        "phase2b_formal_wire_covert_label_factorial_v2_"
    )
    assert execution.operation_counts == tuple(
        (operation, 1) for operation in qualification.EXACT_TRANSFORM_OPERATIONS
    )
    assert execution.adapter.supports_atomic_variant_suite is True
    assert execution.adapter.fresh_build_verification_root.startswith(
        "phase2b_formal_wire_v3_fresh_build_verification_"
    )
    assert execution.adapter.private_alignment.wire_issued_alignment_id == (
        execution.build_success.private_audit_alignment.alignment_id
    )
    expected = set(wire.NAMESPACE_REGISTRY_V3)
    for envelope in execution.adapter.envelopes:
        decoded = execution.adapter.decode_envelope(envelope)
        assert {leaf.namespace for leaf in decoded.namespace_leaves} == expected


@pytest.mark.parametrize("mode", qualification.LIVE_MODE_ORDER)
def test_rich32_fixture_catalog_retains_one_uuid_bearing_use_per_row(
    mode: str,
) -> None:
    cases = tuple(
        qualification._synthetic_source_case_v2(
            index,
            binding_leak=mode == "KNOWN_BINDING_LEAK",
            operation_index=index % 8,
        )
        for index in range(32)
    )
    catalog, _ = wire.issue_formal_global_transform_catalog_v3(
        source_cases=cases
    )
    operation_counts = tuple(
        (
            operation,
            sum(
                case.graph.transforms[0].operation == operation
                for case in cases
            ),
        )
        for operation in qualification.EXACT_TRANSFORM_OPERATIONS
    )
    qualification._validate_fixture_transform_catalog_contract_v2(
        catalog=catalog,
        operation_counts=operation_counts,
        row_count=32,
    )
    assert len(catalog.entries) == 32
    assert len(
        {
            entry.exact_source_transform_use_sha256
            for entry in catalog.entries
        }
    ) == 32
    assert Counter(
        entry.operation for entry in catalog.entries
    ) == Counter(
        {operation: 4 for operation in qualification.EXACT_TRANSFORM_OPERATIONS}
    )
    assert operation_counts == tuple(
        (operation, 4) for operation in qualification.EXACT_TRANSFORM_OPERATIONS
    )
    with pytest.raises(ValueError, match="source use per row"):
        qualification._validate_fixture_transform_catalog_contract_v2(
            catalog=catalog,
            operation_counts=tuple(
                (operation, 3 if index == 0 else 4 if index < 7 else 5)
                for index, operation in enumerate(
                    qualification.EXACT_TRANSFORM_OPERATIONS
                )
            ),
            row_count=32,
        )


def test_small_known_leak_pair_delta_and_bound_authority() -> None:
    execution = _fixture("KNOWN_BINDING_LEAK")
    assert execution.source_pair_delta_root.startswith(
        "phase2b_formal_wire_covert_source_pair_delta_v2_"
    )
    assert tuple(
        len(case.graph.eligibility_mask.observation_ids)
        for case in execution.source_cases
    ) == tuple(
        1 if row.binding == "binding-a" else 2 for row in execution.labels
    )
    binding = execution.adapter.bind_private_authority(
        execution.authority, execution.adapter.envelopes
    )
    assert binding.alignment_issuance_verified is True
    assert binding.label_row_binding_verified is True


def test_standard_error_and_1025_are_atomic_rejections() -> None:
    with pytest.raises(ValueError, match="0..1023"):
        qualification._synthetic_source_case_v2(1024, binding_leak=False)
    case = qualification._synthetic_source_case_v2(
        0, binding_leak=False, standard_error=True
    )
    catalog, catalog_binding = wire.issue_formal_global_transform_catalog_v3(
        source_cases=(case,)
    )
    values = qualification._registered_synthetic_kat_vectors_v2("KNOWN_NULL")
    keys = wire.FormalWireKeySourcesV3(*values[:3])
    receipt = wire.issue_key_source_transcript_receipt_v3(
        key_sources=keys,
        latent_seed_commitments=(wire.seed_commitment_v3(values[3]),),
        answer_seed_commitments=(wire.seed_commitment_v3(values[4]),),
    )
    result = wire.build_formal_trusted_wire_batch_v3(
        source_cases=(case,),
        global_transform_catalog=catalog,
        transform_catalog_binding_receipt=catalog_binding,
        run_id=values[5],
        key_sources=keys,
        key_source_receipt=receipt,
    )
    assert type(result) is wire.FormalTrustedWireBatchRejectionV3
    assert result.reason == "STANDARD_ERROR_UNSUPPORTED"
    oversized = wire.build_formal_trusted_wire_batch_v3(
        source_cases=(case,) * 1025,
        global_transform_catalog=catalog,
        transform_catalog_binding_receipt=catalog_binding,
        run_id=values[5],
        key_sources=keys,
        key_source_receipt=receipt,
    )
    assert type(oversized) is wire.FormalTrustedWireBatchRejectionV3
    assert oversized.reason == "source_case_count_outside_1_through_1024"
    assert oversized.source_case_count == 1025
    assert oversized.envelopes == ()
    assert oversized.envelope_ids == ()
    assert oversized.batch_id is None


def test_stale_splice_and_duplicate_inputs_fail_before_audit() -> None:
    execution = _fixture("KNOWN_NULL")
    evidence = {
        "run_id": execution.run_id,
        "key_sources": execution.key_sources,
        "source_cases": tuple(reversed(execution.source_cases)),
        "global_transform_catalog": execution.global_transform_catalog,
        "transform_catalog_binding_receipt": (
            execution.transform_catalog_binding_receipt
        ),
    }
    with pytest.raises(ValueError, match="source commitments"):
        auditor.FormalWireV3AdapterV1(execution.build_success, **evidence)
    with pytest.raises(ValueError, match="contiguous"):
        auditor.issue_formal_wire_v3_audit_authority_v1(
            execution.build_success, tuple(reversed(execution.labels))
        )
    duplicate = (execution.source_cases[0], execution.source_cases[0])
    catalog, binding = wire.issue_formal_global_transform_catalog_v3(
        source_cases=duplicate
    )
    key_receipt = wire.issue_key_source_transcript_receipt_v3(
        key_sources=execution.key_sources,
        latent_seed_commitments=(wire.seed_commitment_v3(b"L" * 32),),
        answer_seed_commitments=(wire.seed_commitment_v3(b"A" * 32),),
    )
    rejected = wire.build_formal_trusted_wire_batch_v3(
        source_cases=duplicate,
        global_transform_catalog=catalog,
        transform_catalog_binding_receipt=binding,
        run_id=execution.run_id,
        key_sources=execution.key_sources,
        key_source_receipt=key_receipt,
    )
    assert type(rejected) is wire.FormalTrustedWireBatchRejectionV3
    assert rejected.reason in {
        "duplicate_private_source_case",
        "formal_wire_preflight_or_atomic_build_failed",
    }
    assert rejected.source_case_count == 2


def test_registered_synthetic_vectors_are_exact_reproducible_and_nonclaiming() -> None:
    first = qualification._registered_synthetic_kat_vectors_v2("KNOWN_NULL")
    second = qualification._registered_synthetic_kat_vectors_v2("KNOWN_NULL")
    leak = qualification._registered_synthetic_kat_vectors_v2(
        "KNOWN_BINDING_LEAK"
    )
    assert first == second
    assert first != leak
    assert len(first) == len(set(first)) == 6
    assert all(type(value) is bytes and len(value) == 32 for value in first)
    profile = qualification._EXPECTED_PROFILE
    assert profile["fixture_contract"]["input_class"] == "SYNTHETIC_DEV_ONLY"


def test_live_attempt_is_host_environment_only_and_calibration_blocks_inner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = qualification.load_phase2b_formal_wire_covert_qualification_profile_v2(
        qualification.phase2b_formal_wire_covert_qualification_profile_bytes_v2()
    )
    attempt = _attempt()
    monkeypatch.setattr(
        qualification,
        "prepare_live_inner_process_v2",
        lambda: pytest.fail("calibration gate must precede runtime inspection"),
    )
    with pytest.raises(
        qualification.Phase2BFormalWireCovertQualificationErrorV2,
        match="shape calibration",
    ):
        qualification.build_live_inner_candidate_artifact_v2(profile, attempt)
    assert attempt.runtime_image_digest is None


def test_development_calibration_entry_is_reachable_but_never_live_claiming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = qualification.load_phase2b_formal_wire_covert_qualification_profile_v2(
        qualification.phase2b_formal_wire_covert_qualification_profile_bytes_v2()
    )
    reached: list[bool] = []

    def stop_before_full() -> dict[str, object]:
        reached.append(True)
        raise RuntimeError("STOP_BEFORE_FULL_1024")

    monkeypatch.setattr(
        qualification, "prepare_live_inner_process_v2", stop_before_full
    )
    with pytest.raises(RuntimeError, match="STOP_BEFORE_FULL_1024"):
        qualification.build_development_calibration_artifact_v2(
            profile, _attempt()
        )
    assert reached == [True]
    assert (
        qualification._EXPECTED_PROFILE["fixture_contract"][
            "rich_positive_shape"
        ]["calibration_artifact_cannot_be_promoted"]
        is True
    )


def test_post_receipt_validation_failure_releases_only_locally_owned_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _fixture("KNOWN_NULL")
    owned = object()
    released: list[object] = []
    monkeypatch.setattr(
        qualification,
        "_build_fixture_execution_v2",
        lambda mode: execution,
    )
    monkeypatch.setattr(
        auditor,
        "run_formal_covert_audit_v1",
        lambda *args: owned,
    )
    monkeypatch.setattr(
        qualification,
        "_validate_isolated_audit_receipt_v2",
        lambda *args: (_ for _ in ()).throw(RuntimeError("post-receipt")),
    )
    monkeypatch.setattr(
        auditor,
        "FormalCovertAuditReceiptV1",
        type(owned),
    )
    monkeypatch.setattr(
        auditor,
        "release_formal_covert_audit_receipt_v1",
        lambda receipt: released.append(receipt) or "released",
    )
    with pytest.raises(RuntimeError, match="post-receipt"):
        qualification._execute_isolated_audit_v2("KNOWN_NULL")
    assert released == [owned]


def test_candidate_success_projects_and_releases_both_receipts_over_registry_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live: set[int] = set()

    class FakeReceipt:
        def __init__(self, receipt_id: str) -> None:
            self.receipt_id = receipt_id
            self.feature_manifest = SimpleNamespace(
                manifest_root="feature-root",
                deduplicated_feature_count=4,
                raw_semantic_feature_count=4,
            )
            self.resource_plan = SimpleNamespace(
                unique_statistical_partitions=4,
                hypotheses=20,
            )
            live.add(id(self))

    def validate(receipt: FakeReceipt) -> str:
        assert id(receipt) in live
        return receipt.receipt_id

    def release(receipt: FakeReceipt) -> str:
        assert id(receipt) in live
        live.remove(id(receipt))
        return receipt.receipt_id

    monkeypatch.setattr(auditor, "FormalCovertAuditReceiptV1", FakeReceipt)
    monkeypatch.setattr(auditor, "validate_formal_covert_audit_receipt_v1", validate)
    monkeypatch.setattr(auditor, "release_formal_covert_audit_receipt_v1", release)
    alias = (
        "uuid_leaf/observation_id/"
        + qualification._json_path_text_v2(
            qualification._REGISTERED_BINDING_LEAK_OWNER_PATH_V2
        )
        + "/presence"
    )

    def diagnostics(mode: str) -> qualification.AuditRunDiagnosticsV2:
        return qualification.AuditRunDiagnosticsV2(
            mode,
            1,
            1,
            (1,),
            2,
            1,
            1,
            qualification.EXACT_RECOGNIZER_INVOCATIONS,
            qualification.EXACT_AUTHORITY_ROWS,
            qualification.EXACT_AUTHORITY_ROWS,
        )

    for index in range(auditor.MAXIMUM_IN_PROCESS_ISSUED_RECEIPTS + 1):
        assert len(live) + 2 <= auditor.MAXIMUM_IN_PROCESS_ISSUED_RECEIPTS
        positive_receipt = FakeReceipt(f"positive-{index}")
        negative_receipt = FakeReceipt(f"negative-{index}")
        positive = SimpleNamespace(
            mode="KNOWN_NULL",
            label_factorial_proof_root="phase2b_formal_wire_covert_label_factorial_v2_x",
            audit_receipt=positive_receipt,
            batch_id="batch",
            transform_catalog_root="catalog",
            diagnostics=diagnostics("KNOWN_NULL"),
            operation_counts=tuple(
                (operation, qualification.EXACT_ROWS_PER_TRANSFORM_OPERATION)
                for operation in qualification.EXACT_TRANSFORM_OPERATIONS
            ),
            namespace_uuid_leaf_paths=1,
            channel_leaf_paths=1,
            exact_owned_channel_only_leaf_paths=1,
            full_raw_byte_categorical_hypotheses=1,
        )
        negative = SimpleNamespace(
            mode="KNOWN_BINDING_LEAK",
            label_factorial_proof_root=positive.label_factorial_proof_root,
            audit_receipt=negative_receipt,
            diagnostics=diagnostics("KNOWN_BINDING_LEAK"),
            disposition="FAIL",
            leak_feature_id="feature",
            leak_feature_aliases=(alias,),
            leak_observed_omnibus_hex=(2.0).hex(),
            leak_feature_max_t_p="1/10001",
            leak_target_holm_critical_value="1/500",
            source_pair_delta_root="pair-root",
        )
        candidate = qualification.Phase2BFormalWireCovertCandidateReceiptV2._issue(
            qualification._RECEIPT_ISSUE_TOKEN,
            synthetic_fixture_root="fixture-root",
            positive=positive,
            negative=negative,
        )
        assert candidate.positive_audit_receipt_id == f"positive-{index}"
        assert not hasattr(candidate, "positive_audit_receipt")
        projection = candidate._receipt_projection()
        assert projection["projection_only_boundary"] == (
            qualification._TASK4_PROJECTION_ONLY_BOUNDARY_V2
        )
        assert projection["claims"] == {
            "actual_corpus_audit": False,
            "actual_execution_claimed": False,
            "dependency_lock_live_verified": False,
            "r03_formal_trusted_wire_implementation_qualified": False,
            "r04_formal_covert_auditor_implementation_qualified": False,
            "r05_synthetic_dev_wire_covert_qualification_verified": False,
            "synthetic_fixture_archive_verified": False,
        }
        assert live == set()
        if index == 0:
            object.__setattr__(candidate, "actual_execution_claimed", True)
            with pytest.raises(ValueError, match="runtime/readiness boundary"):
                candidate._validate()


def test_cgroup_v2_parser_requires_finite_memory_and_pids(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    target = cgroup / "unit.service"
    (proc / "self").mkdir(parents=True)
    target.mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/unit.service\n", encoding="utf-8")
    rows = {
        "memory.max": str(auditor.MAXIMUM_ESTIMATED_RSS_BYTES),
        "memory.current": "1024",
        "memory.peak": "2048",
        "memory.events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n",
        "pids.max": "64",
        "pids.current": "1",
        "pids.events": "max 0\n",
    }
    for name, payload in rows.items():
        (target / name).write_text(payload, encoding="ascii")
    evidence = qualification.inspect_current_live_cgroup_v2(
        proc_root=proc, cgroup_root=cgroup
    )
    assert evidence["memory_max"] == auditor.MAXIMUM_ESTIMATED_RSS_BYTES
    assert evidence["pids_max"] == 64
    assert evidence["cpu_max"] is None
    (target / "memory.max").write_text("max\n", encoding="ascii")
    with pytest.raises(ValueError, match="finite"):
        qualification.inspect_current_live_cgroup_v2(
            proc_root=proc, cgroup_root=cgroup
        )


def test_production_has_zero_assert_and_no_pickle_pipe_or_process_pool() -> None:
    path = ROOT / "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
    for forbidden in (
        "multiprocessing.Pipe",
        "ProcessPoolExecutor",
        "pickle.dumps",
        "pickle.loads",
    ):
        assert forbidden not in source


def test_launch_intent_binds_attempt_unit_marker_and_prelaunch_absence() -> None:
    attempt, source_tree, intent = _launch_intent()
    assert qualification._validate_launch_intent_projection_v2(
        intent,
        attempt=attempt,
        entry_kind="development-calibration",
        source_tree=source_tree,
    ) == intent
    tampered = dict(intent)
    tampered["unit_name"] = "hegel-qv2-" + "f" * 32 + ".service"
    with pytest.raises(ValueError, match="ownership preimage"):
        qualification._validate_launch_intent_projection_v2(
            tampered,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
        )


def test_public_mechanics_rejects_asserted_or_forged_pair_replay_authority() -> None:
    profile = object.__new__(
        qualification.Phase2BFormalWireCovertQualificationProfileV2
    )
    common = {
        "inner_artifact_bytes": b"{}\n",
        "outer_supervisor_result_bytes": b"{}\n",
        "rfc8785_prerequisite_receipt_bytes": b"{}\n",
    }
    signature = inspect.signature(
        qualification.verify_phase2b_formal_wire_covert_outer_mechanics_v2
    )
    assert "_calibration_pair_replay_token" not in signature.parameters
    assert "_calibration_pair_projection" not in signature.parameters
    with pytest.raises(TypeError, match="unexpected keyword"):
        qualification.verify_phase2b_formal_wire_covert_outer_mechanics_v2(
            profile,
            _attempt(),
            calibration_pair_pure_replay_verified=True,
            **common,
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        qualification.verify_phase2b_formal_wire_covert_outer_mechanics_v2(
            profile,
            _attempt(),
            _calibration_pair_replay_token=object(),
            **common,
        )


def test_calibration_pair_projection_and_private_token_are_content_bound() -> None:
    attempt = _attempt()
    attempt_name = "attempt-" + hashlib.sha256(
        attempt.attempt_manifest_root.encode("ascii")
    ).hexdigest()[:48]
    projection = {
        "schema_version": (
            "hegel-machine-phase2b-formal-wire-covert-calibration-attempt-pair/1"
        ),
        "attempt_directory": str(Path("/tmp/hegel-qv2-pair") / attempt_name),
        "attempt_manifest_root": attempt.attempt_manifest_root,
        "terminal_sha256": "3" * 64,
        "terminal_size_bytes": 1024,
        "retained_tree_before_terminal": {
            "regular_file_count": 6,
            "root": "phase2b_formal_wire_covert_failure_tree_v3_" + "4" * 64,
            "row_count": 10,
            "top_level_names": ["evidence", "reservation.json"],
            "total_regular_bytes": 4096,
        },
        "pair_retention_required": True,
        "pair_must_be_jointly_signed_by_step5": True,
    }
    assert qualification._validate_calibration_pair_projection_shape_v2(
        projection, required=True
    ) == projection
    tampered = dict(projection)
    tampered["attempt_directory"] = str(
        Path("/tmp/hegel-qv2-pair") / ("attempt-" + "f" * 48)
    )
    with pytest.raises(ValueError, match="retained-tree projection|not closed"):
        qualification._validate_calibration_pair_projection_shape_v2(
            tampered, required=True
        )

    payloads = tuple(f"pair-{index}".encode("ascii") for index in range(6))
    with pytest.raises(TypeError, match="issuer is private"):
        qualification._issue_verified_calibration_pair_replay_token_v2(
            object(), payloads, projection, "outer-mechanics"
        )
    token = qualification._issue_verified_calibration_pair_replay_token_v2(
        qualification._CALIBRATION_PAIR_REPLAY_TOKEN_ISSUER_V2,
        payloads,
        projection,
        "outer-mechanics",
    )
    qualification._consume_verified_calibration_pair_replay_token_v2(
        token, payloads, projection, "outer-mechanics"
    )
    with pytest.raises(TypeError, match="fresh retained-directory verified"):
        qualification._consume_verified_calibration_pair_replay_token_v2(
            token, payloads, projection, "outer-mechanics"
        )
    foreign_projection = dict(projection)
    foreign_projection["terminal_sha256"] = "5" * 64
    foreign_token = qualification._issue_verified_calibration_pair_replay_token_v2(
        qualification._CALIBRATION_PAIR_REPLAY_TOKEN_ISSUER_V2,
        payloads,
        projection,
        "outer-mechanics",
    )
    with pytest.raises(TypeError, match="fresh retained-directory verified"):
        qualification._consume_verified_calibration_pair_replay_token_v2(
            foreign_token, payloads, foreign_projection, "outer-mechanics"
        )
    with pytest.raises(TypeError, match="fresh retained-directory verified"):
        qualification._consume_verified_calibration_pair_replay_token_v2(
            foreign_token, payloads, projection, "outer-mechanics"
        )


def test_pair_loader_derives_six_bytes_from_retained_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = _attempt()
    attempt_name = "attempt-" + hashlib.sha256(
        attempt.attempt_manifest_root.encode("ascii")
    ).hexdigest()[:48]
    pair_directory = tmp_path / attempt_name
    inputs = pair_directory / "evidence/inputs"
    cas = pair_directory / "evidence/cas"
    inputs.mkdir(parents=True)
    cas.mkdir()
    profile_bytes = b"paired-profile\n"
    attempt_bytes = b"paired-attempt\n"
    inner_bytes = b"paired-inner\n"
    outer_bytes = b"paired-outer\n"
    mechanics_bytes = b"paired-mechanics\n"
    rfc_bytes = b"paired-rfc\n"
    (inputs / "profile.json").write_bytes(profile_bytes)
    (inputs / "attempt.json").write_bytes(attempt_bytes)
    (pair_directory / "evidence/inner.json").write_bytes(inner_bytes)
    (pair_directory / "evidence/outer-supervisor.json").write_bytes(outer_bytes)
    (pair_directory / "evidence/mechanics.json").write_bytes(mechanics_bytes)
    rfc_digest = hashlib.sha256(rfc_bytes).hexdigest()
    (cas / rfc_digest).write_bytes(rfc_bytes)
    attempt_fd = qualification._open_real_directory_fd_v2(pair_directory)
    try:
        tree = qualification._retained_attempt_tree_before_terminal_v2(attempt_fd)
    finally:
        os.close(attempt_fd)
    terminal_bytes = qualification._canonical_inner_artifact_v2(
        {
            "schema_version": qualification.OUTER_ATTEMPT_SUCCESS_SCHEMA_V2,
            "entry_kind": "development-calibration",
            "disposition": "MECHANICS_VERIFIED_NOT_AUTHORITY",
            "attempt_manifest_root": attempt.attempt_manifest_root,
            "retained_tree_before_terminal": tree,
            "artifacts": {
                "rfc8785_prerequisite_receipt": {
                    "content_sha256": rfc_digest,
                    "object_size_bytes": len(rfc_bytes),
                }
            },
        }
    )
    (pair_directory / "terminal.json").write_bytes(terminal_bytes)

    def verified_directory(directory: Path, **kwargs: object) -> bytes:
        assert directory == pair_directory
        assert kwargs["_pair_replay_depth"] == 1
        assert type(kwargs["retained_attempt_fd"]) is int
        assert type(kwargs["retained_output_root_fd"]) is int
        return terminal_bytes

    monkeypatch.setattr(
        qualification,
        "verify_phase2b_formal_wire_covert_outer_attempt_directory_v2",
        verified_directory,
    )
    payloads, projection, token = (
        qualification._load_verified_development_calibration_pair_v2(
            pair_directory,
            deadline_ns=None,
            pair_replay_depth=1,
            token_purpose="outer-mechanics",
        )
    )
    assert payloads == (
        profile_bytes,
        attempt_bytes,
        inner_bytes,
        outer_bytes,
        rfc_bytes,
        mechanics_bytes,
    )
    assert projection["retained_tree_before_terminal"] == tree
    assert projection["terminal_sha256"] == hashlib.sha256(terminal_bytes).hexdigest()
    qualification._consume_verified_calibration_pair_replay_token_v2(
        token, payloads, projection, "outer-mechanics"
    )


def test_cross_revision_pair_accepts_only_pending_basis_for_frozen_current() -> None:
    basis_profile = qualification._deep_thaw(qualification._EXPECTED_PROFILE)
    basis_profile["status"] = "P0_PENDING_CALIBRATION_BASIS_FOR_CROSS_REVISION_KAT"
    basis_id = qualification.PROFILE_ID_PREFIX + hashlib.sha256(
        qualification.PROFILE_DOMAIN
        + qualification.canonicalize_rfc8785_v1(basis_profile)
    ).hexdigest()
    basis_bytes = qualification.canonicalize_rfc8785_v1(
        {
            "qualification_profile": basis_profile,
            "qualification_profile_id": basis_id,
        }
    ) + b"\n"
    assert basis_id != qualification.PROFILE_ID
    frozen_shape = qualification._deep_thaw(
        qualification._EXPECTED_PROFILE["fixture_contract"]["rich_positive_shape"]
    )
    frozen_shape["calibration_campaign_status"] = (
        "FROZEN_FULL_1024_DEVELOPMENT_MEASUREMENT"
    )
    frozen_shape["live_qualification_enabled"] = True
    frozen_shape["calibration_basis_profile_id"] = basis_id
    frozen_shape["calibration_basis_profile_sha256"] = hashlib.sha256(
        basis_bytes
    ).hexdigest()
    frozen_shape["calibration_basis_profile_size_bytes"] = len(basis_bytes)
    assert qualification._load_retained_development_basis_profile_id_v2(
        basis_bytes,
        frozen_shape=frozen_shape,
        expected_entry_kind="development-calibration",
    ) == basis_id

    with pytest.raises(ValueError, match="only a development attempt"):
        qualification._load_retained_development_basis_profile_id_v2(
            basis_bytes,
            frozen_shape=frozen_shape,
            expected_entry_kind="live-inner",
        )

    current_bytes = (
        qualification.phase2b_formal_wire_covert_qualification_profile_bytes_v2()
    )
    current_shape = dict(frozen_shape)
    current_shape["calibration_basis_profile_id"] = qualification.PROFILE_ID
    current_shape["calibration_basis_profile_sha256"] = hashlib.sha256(
        current_bytes
    ).hexdigest()
    current_shape["calibration_basis_profile_size_bytes"] = len(current_bytes)
    with pytest.raises(ValueError, match="identity/bytes are not frozen"):
        qualification._load_retained_development_basis_profile_id_v2(
            current_bytes,
            frozen_shape=current_shape,
            expected_entry_kind="development-calibration",
        )

    spliced_shape = dict(frozen_shape)
    spliced_shape["calibration_basis_profile_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="identity/bytes are not frozen"):
        qualification._load_retained_development_basis_profile_id_v2(
            basis_bytes,
            frozen_shape=spliced_shape,
            expected_entry_kind="development-calibration",
        )

    old_live_profile = qualification._deep_thaw(basis_profile)
    old_live_shape = old_live_profile["fixture_contract"]["rich_positive_shape"]
    old_live_shape["calibration_campaign_status"] = (
        "FROZEN_FULL_1024_DEVELOPMENT_MEASUREMENT"
    )
    old_live_shape["live_qualification_enabled"] = True
    old_live_id = qualification.PROFILE_ID_PREFIX + hashlib.sha256(
        qualification.PROFILE_DOMAIN
        + qualification.canonicalize_rfc8785_v1(old_live_profile)
    ).hexdigest()
    old_live_bytes = qualification.canonicalize_rfc8785_v1(
        {
            "qualification_profile": old_live_profile,
            "qualification_profile_id": old_live_id,
        }
    ) + b"\n"
    old_live_frozen_shape = dict(frozen_shape)
    old_live_frozen_shape["calibration_basis_profile_id"] = old_live_id
    old_live_frozen_shape["calibration_basis_profile_sha256"] = hashlib.sha256(
        old_live_bytes
    ).hexdigest()
    old_live_frozen_shape["calibration_basis_profile_size_bytes"] = len(
        old_live_bytes
    )
    with pytest.raises(qualification.Phase2BFormalWireCovertQualificationErrorV2):
        qualification._load_retained_development_basis_profile_id_v2(
            old_live_bytes,
            frozen_shape=old_live_frozen_shape,
            expected_entry_kind="development-calibration",
        )


def test_verified_cleanup_requires_ordered_absence_and_rejects_kill_without_term() -> None:
    attempt, source_tree, intent = _launch_intent()
    show_argv = qualification._systemd_show_argv_v2(
        intent["executables"]["systemctl"]["realpath"], intent["unit_name"]
    )
    absent = _systemd_show_bytes(attempt, source_tree)
    commands = [
        intent["prelaunch_command"],
        qualification._command_evidence_v2(
            argv=show_argv,
            purpose="failure-owned-show",
            returncode=0,
            stdout=absent,
            stderr=b"",
        ),
        qualification._command_evidence_v2(
            argv=show_argv,
            purpose="failure-absent-0",
            returncode=0,
            stdout=absent,
            stderr=b"",
        ),
    ]
    cleanup = {
        "cleanup_start_command_index": 1,
        "cleanup_verified": True,
        "commands": commands,
        "drained_stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "drained_stderr_size_bytes": 0,
        "drained_stdout_sha256": hashlib.sha256(b"").hexdigest(),
        "drained_stdout_size_bytes": 0,
        "kill_sent": False,
        "launcher_reaped": True,
        "launcher_returncode": 0,
        "owned_invocation_id": None,
        "ownership_marker": intent["ownership_marker"],
        "post_cleanup_unit_absent": True,
        "prelaunch_unit_absent": True,
        "term_sent": False,
        "transcript_root": qualification.stable_hash(
            commands,
            prefix="phase2b_formal_wire_covert_failure_cleanup_v2_",
        ),
        "unit_name": intent["unit_name"],
    }
    qualification._verify_failure_cleanup_projection_v2(
        cleanup,
        cleanup_verified=True,
        attempt=attempt,
        entry_kind="development-calibration",
        source_tree=source_tree,
        launch_intent=intent,
    )
    kill = qualification._command_evidence_v2(
        argv=[
            intent["executables"]["systemctl"]["realpath"],
            "--user", "kill", "--kill-whom=all", "--signal=SIGKILL",
            intent["unit_name"],
        ],
        purpose="failure-owned-sigkill",
        returncode=0,
        stdout=b"",
        stderr=b"",
    )
    bad_commands = commands[:2] + [kill] + commands[2:]
    bad = dict(cleanup)
    bad.update(
        commands=bad_commands,
        kill_sent=True,
        transcript_root=qualification.stable_hash(
            bad_commands,
            prefix="phase2b_formal_wire_covert_failure_cleanup_v2_",
        ),
    )
    with pytest.raises(ValueError, match="SIGKILL lacked TERM"):
        qualification._verify_failure_cleanup_projection_v2(
            bad,
            cleanup_verified=True,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
        )


def test_unproven_cleanup_is_nonpromotable_but_preserves_failed_control_row() -> None:
    attempt, source_tree, intent = _launch_intent()
    show_argv = qualification._systemd_show_argv_v2(
        intent["executables"]["systemctl"]["realpath"], intent["unit_name"]
    )
    commands = [
        intent["prelaunch_command"],
        qualification._command_evidence_v2(
            argv=show_argv,
            purpose="failure-owned-show",
            returncode=1,
            stdout=b"",
            stderr=b"manager unavailable",
        ),
    ]
    cleanup = {
        "cleanup_error_type": "RuntimeError",
        "cleanup_start_command_index": 1,
        "cleanup_verified": False,
        "commands": commands,
        "external_intervention_required": True,
        "owned_invocation_id": None,
        "ownership_marker": intent["ownership_marker"],
        "prelaunch_unit_absent": True,
        "transcript_root": qualification.stable_hash(
            commands,
            prefix="phase2b_formal_wire_covert_failure_cleanup_v2_",
        ),
        "unit_name": intent["unit_name"],
    }
    qualification._verify_failure_cleanup_projection_v2(
        cleanup,
        cleanup_verified=False,
        attempt=attempt,
        entry_kind="development-calibration",
        source_tree=source_tree,
        launch_intent=intent,
    )


def test_cleanup_main_boundary_accepts_exact_51_and_rejects_52() -> None:
    attempt, source_tree, intent = _launch_intent()
    show_argv = qualification._systemd_show_argv_v2(
        intent["executables"]["systemctl"]["realpath"], intent["unit_name"]
    )
    commands = [intent["prelaunch_command"]] + [
        qualification._command_evidence_v2(
            argv=show_argv,
            purpose=f"entry-show-{index}",
            returncode=1,
            stdout=b"",
            stderr=b"failure boundary",
        )
        for index in range(50)
    ]

    def projection(rows: list[dict[str, object]], boundary: int) -> dict[str, object]:
        return {
            "cleanup_error_type": "RuntimeError",
            "cleanup_start_command_index": boundary,
            "cleanup_verified": False,
            "commands": rows,
            "external_intervention_required": True,
            "owned_invocation_id": None,
            "ownership_marker": intent["ownership_marker"],
            "prelaunch_unit_absent": True,
            "transcript_root": qualification.stable_hash(
                rows,
                prefix="phase2b_formal_wire_covert_failure_cleanup_v2_",
            ),
            "unit_name": intent["unit_name"],
        }

    qualification._verify_failure_cleanup_projection_v2(
        projection(commands, 51),
        cleanup_verified=False,
        attempt=attempt,
        entry_kind="development-calibration",
        source_tree=source_tree,
        launch_intent=intent,
    )
    too_long = commands + [
        qualification._command_evidence_v2(
            argv=show_argv,
            purpose="entry-show-50",
            returncode=1,
            stdout=b"",
            stderr=b"failure boundary",
        )
    ]
    with pytest.raises(ValueError, match="command boundary"):
        qualification._verify_failure_cleanup_projection_v2(
            projection(too_long, 52),
            cleanup_verified=False,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
        )


def test_cleanup_rejects_cross_boundary_splice_and_suffix_over_40() -> None:
    attempt, source_tree, intent = _launch_intent()
    show_argv = qualification._systemd_show_argv_v2(
        intent["executables"]["systemctl"]["realpath"], intent["unit_name"]
    )
    cleanup_show = qualification._command_evidence_v2(
        argv=show_argv,
        purpose="failure-owned-show",
        returncode=1,
        stdout=b"",
        stderr=b"unproven",
    )

    def unproven(rows: list[dict[str, object]], boundary: int) -> dict[str, object]:
        return {
            "cleanup_error_type": "RuntimeError",
            "cleanup_start_command_index": boundary,
            "cleanup_verified": False,
            "commands": rows,
            "external_intervention_required": True,
            "owned_invocation_id": None,
            "ownership_marker": intent["ownership_marker"],
            "prelaunch_unit_absent": True,
            "transcript_root": qualification.stable_hash(
                rows,
                prefix="phase2b_formal_wire_covert_failure_cleanup_v2_",
            ),
            "unit_name": intent["unit_name"],
        }

    crossed = [intent["prelaunch_command"], cleanup_show]
    with pytest.raises(ValueError, match="crosses its phase boundary"):
        qualification._verify_failure_cleanup_projection_v2(
            unproven(crossed, 2),
            cleanup_verified=False,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
        )
    oversized_suffix = [intent["prelaunch_command"]] + [cleanup_show] * 41
    with pytest.raises(ValueError, match="suffix exceeds"):
        qualification._verify_failure_cleanup_projection_v2(
            unproven(oversized_suffix, 1),
            cleanup_verified=False,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
        )


def test_verified_cleanup_rejects_rows_after_final_absence() -> None:
    attempt, source_tree, intent = _launch_intent()
    show_argv = qualification._systemd_show_argv_v2(
        intent["executables"]["systemctl"]["realpath"], intent["unit_name"]
    )
    absent = _systemd_show_bytes(attempt, source_tree)
    commands = [
        intent["prelaunch_command"],
        qualification._command_evidence_v2(
            argv=show_argv, purpose="failure-owned-show", returncode=0,
            stdout=absent, stderr=b"",
        ),
        qualification._command_evidence_v2(
            argv=show_argv, purpose="failure-absent-0", returncode=0,
            stdout=absent, stderr=b"",
        ),
        qualification._command_evidence_v2(
            argv=show_argv, purpose="failure-absent-1", returncode=0,
            stdout=absent, stderr=b"",
        ),
    ]
    cleanup = {
        "cleanup_start_command_index": 1,
        "cleanup_verified": True,
        "commands": commands,
        "drained_stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "drained_stderr_size_bytes": 0,
        "drained_stdout_sha256": hashlib.sha256(b"").hexdigest(),
        "drained_stdout_size_bytes": 0,
        "kill_sent": False,
        "launcher_reaped": True,
        "launcher_returncode": 0,
        "owned_invocation_id": None,
        "ownership_marker": intent["ownership_marker"],
        "post_cleanup_unit_absent": True,
        "prelaunch_unit_absent": True,
        "term_sent": False,
        "transcript_root": qualification.stable_hash(
            commands,
            prefix="phase2b_formal_wire_covert_failure_cleanup_v2_",
        ),
        "unit_name": intent["unit_name"],
    }
    with pytest.raises(ValueError, match="continued after final absence"):
        qualification._verify_failure_cleanup_projection_v2(
            cleanup,
            cleanup_verified=True,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
        )


def test_systemctl_fast_success_transcript_is_exact_five_commands() -> None:
    attempt, source_tree, intent = _launch_intent()
    systemctl = intent["executables"]["systemctl"]["realpath"]
    show_argv = qualification._systemd_show_argv_v2(systemctl, intent["unit_name"])
    invocation = "a" * 32
    entry = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="running",
        main_pid=123,
        cgroup="/user.slice/hegel-qv2.service",
    )
    terminal = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="exited",
    )
    absent = _systemd_show_bytes(attempt, source_tree)
    commands = [
        intent["prelaunch_command"],
        qualification._command_evidence_v2(
            argv=show_argv, purpose="entry-show-0", returncode=0,
            stdout=entry, stderr=b"",
        ),
        qualification._command_evidence_v2(
            argv=show_argv, purpose="terminal-show-0", returncode=0,
            stdout=terminal, stderr=b"",
        ),
        qualification._command_evidence_v2(
            argv=[systemctl, "--user", "stop", intent["unit_name"]],
            purpose="stop-after-terminal-snapshot", returncode=0,
            stdout=b"", stderr=b"",
        ),
        qualification._command_evidence_v2(
            argv=show_argv, purpose="post-stop-absent-check-0", returncode=0,
            stdout=absent, stderr=b"",
        ),
    ]
    observed_entry, observed_terminal, observed_absent = (
        qualification._verify_systemctl_transcript_v2(
            commands,
            systemctl_realpath=systemctl,
            unit_name=intent["unit_name"],
            attempt=attempt,
            source_tree=source_tree,
            ownership_marker=intent["ownership_marker"],
        )
    )
    assert observed_entry["InvocationID"] == invocation
    assert observed_terminal["InvocationID"] == invocation
    assert qualification._systemd_show_is_exact_absent_v2(observed_absent)


def test_nonterminal_atomic_commit_is_recoverable_but_terminal_commit_is_not() -> None:
    cas_uncertain = qualification._AtomicPublicationUncertainV2(
        "a" * 64, "CAS committed before replay failure"
    )
    terminal_uncertain = qualification._AtomicPublicationUncertainV2(
        "terminal.json", "terminal committed before replay failure"
    )
    assert not qualification._atomic_uncertainty_blocks_another_terminal_v2(
        cas_uncertain
    )
    assert qualification._atomic_uncertainty_blocks_another_terminal_v2(
        terminal_uncertain
    )


def test_atomic_postlink_failure_preserves_target_and_reports_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path.resolve()
    directory_fd = qualification._open_real_directory_fd_v2(directory)
    real_fsync = qualification.os.fsync

    def fail_parent_fsync(descriptor: int) -> None:
        if descriptor == directory_fd:
            raise OSError("injected parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(qualification.os, "fsync", fail_parent_fsync)
    try:
        with pytest.raises(
            qualification._AtomicPublicationUncertainV2
        ) as captured:
            qualification._atomic_write_new_file_at_fd_v2(
                directory_fd, "object", b"committed-bytes"
            )
        assert captured.value.target_name == "object"
        assert (directory / "object").read_bytes() == b"committed-bytes"
        assert not any(name.startswith(".object.tmp-") for name in os.listdir(directory))
    finally:
        os.close(directory_fd)


def test_atomic_postlink_temp_cleanup_error_cannot_mask_committed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path.resolve()
    directory_fd = qualification._open_real_directory_fd_v2(directory)
    real_unlink = qualification.os.unlink

    def fail_temporary_unlink(
        path: object, *args: object, **kwargs: object
    ) -> None:
        if str(path).startswith(".object.tmp-"):
            raise OSError("injected temp unlink failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(qualification.os, "unlink", fail_temporary_unlink)
    try:
        with pytest.raises(
            qualification._AtomicPublicationUncertainV2
        ) as captured:
            qualification._atomic_write_new_file_at_fd_v2(
                directory_fd, "object", b"committed-with-temp"
            )
        assert captured.value.target_name == "object"
        assert captured.value.publication_state == "TARGET_LINK_COMMITTED"
        assert captured.value.temporary_name is not None
        assert captured.value.parent_identity is not None
        assert captured.value.target_identity is not None
        assert (directory / "object").read_bytes() == b"committed-with-temp"
        assert any(name.startswith(".object.tmp-") for name in os.listdir(directory))
    finally:
        os.close(directory_fd)


def test_atomic_after_link_before_python_marker_is_reconciled_as_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path.resolve()
    directory_fd = qualification._open_real_directory_fd_v2(directory)
    real_link = qualification.os.link

    def link_then_interrupt(*args: object, **kwargs: object) -> None:
        real_link(*args, **kwargs)
        raise KeyboardInterrupt("injected after kernel link")

    monkeypatch.setattr(qualification.os, "link", link_then_interrupt)
    try:
        with pytest.raises(
            qualification._AtomicPublicationUncertainV2
        ) as captured:
            qualification._atomic_write_new_file_at_fd_v2(
                directory_fd, "object", b"linked-before-marker"
            )
        assert captured.value.publication_state == "TARGET_LINK_COMMITTED"
        assert (directory / "object").read_bytes() == b"linked-before-marker"
        assert not any(name.startswith(".object.tmp-") for name in os.listdir(directory))
    finally:
        os.close(directory_fd)


def test_prelink_persistent_staging_residue_is_exactly_bound_and_tamper_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path.resolve()
    directory_fd = qualification._open_real_directory_fd_v2(directory)
    real_write_all = qualification._write_all_fd_v2
    real_unlink = qualification.os.unlink
    real_remove = qualification.os.remove

    def partial_then_fail(
        descriptor: int, payload: bytes, *, deadline_ns: int | None = None
    ) -> None:
        real_write_all(descriptor, payload[:7], deadline_ns=deadline_ns)
        raise OSError("injected prelink write failure")

    def persistent_cleanup_failure(
        path: object, *args: object, **kwargs: object
    ) -> None:
        if str(path).startswith(".object.tmp-"):
            raise OSError("injected persistent staging cleanup failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(qualification, "_write_all_fd_v2", partial_then_fail)
    monkeypatch.setattr(qualification.os, "unlink", persistent_cleanup_failure)
    monkeypatch.setattr(qualification.os, "remove", persistent_cleanup_failure)
    try:
        with pytest.raises(
            qualification._AtomicPublicationUncertainV2
        ) as captured:
            qualification._atomic_write_new_file_at_fd_v2(
                directory_fd, "object", b"staging-content"
            )
        assert captured.value.publication_state == "STAGED_TEMP_ONLY"
        residues = qualification._reconcile_nonterminal_atomic_uncertainty_v2(
            captured.value,
            attempt_fd=directory_fd,
            deadline_ns=time.monotonic_ns() + 10_000_000_000,
        )
        assert len(residues) == 1
        assert residues[0]["state"] == "STAGED_TEMP_ONLY"
        projection = qualification._retained_attempt_tree_before_terminal_v2(
            directory_fd,
            managed_atomic_residues=residues,
        )
        assert projection["regular_file_count"] == 1
        with pytest.raises(ValueError, match="unknown temp"):
            qualification._retained_attempt_tree_before_terminal_v2(directory_fd)

        temporary_name = str(residues[0]["temporary_name"])
        payload = (directory / temporary_name).read_bytes()
        retained_old_inode_fd = os.open(
            temporary_name, os.O_RDONLY | os.O_CLOEXEC, dir_fd=directory_fd
        )
        try:
            real_unlink(temporary_name, dir_fd=directory_fd)
            (directory / temporary_name).write_bytes(payload)
            with pytest.raises(ValueError, match="inode/linkage"):
                qualification._retained_attempt_tree_before_terminal_v2(
                    directory_fd,
                    managed_atomic_residues=residues,
                )
        finally:
            os.close(retained_old_inode_fd)
        real_unlink(temporary_name, dir_fd=directory_fd)
    finally:
        # Use the saved functions because the shared os module is monkeypatched.
        monkeypatch.setattr(qualification.os, "remove", real_remove)
        os.close(directory_fd)


def test_terminal_postlink_enumeration_error_remains_terminal_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path.resolve()
    directory_fd = qualification._open_real_directory_fd_v2(directory)
    real_names = qualification._directory_names_at_fd_v2
    calls = 0

    def fail_after_link(descriptor: int) -> tuple[str, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_names(descriptor)
        raise OSError("injected postlink enumeration failure")

    monkeypatch.setattr(
        qualification, "_directory_names_at_fd_v2", fail_after_link
    )
    try:
        with pytest.raises(
            qualification._AtomicPublicationUncertainV2
        ) as captured:
            qualification._publish_terminal_once_at_fd_v2(
                directory_fd,
                b"terminal-bytes",
                deadline_ns=qualification.time.monotonic_ns() + 10_000_000_000,
            )
        assert captured.value.target_name == "terminal.json"
        assert (directory / "terminal.json").read_bytes() == b"terminal-bytes"
    finally:
        os.close(directory_fd)


@pytest.mark.parametrize(
    ("uncertainty_point", "retain_residue"),
    (
        ("postlink", False),
        ("prelink", False),
        ("postlink", True),
        ("prelink", True),
    ),
)
def test_nonterminal_atomic_uncertainty_finalizes_claims_false_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uncertainty_point: str,
    retain_residue: bool,
) -> None:
    profile_bytes = b"synthetic-profile-bytes\n"
    attempt = replace(
        _attempt(),
        qualification_config_sha256=hashlib.sha256(profile_bytes).hexdigest(),
        qualification_lock_sha256=qualification.DEPENDENCY_LOCK_SHA256,
    )
    attempt_bytes = b"synthetic-attempt-bytes\n"
    output_root = (tmp_path / "attempts").resolve()
    output_root.mkdir()
    real_attempt_verifier = (
        qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2
    )
    real_unlink = qualification.os.unlink
    real_remove = qualification.os.remove
    real_write_all = qualification._write_all_fd_v2
    temporary_unlink_failures = 0
    prelink_injected = False
    prospective_returned_at: list[int] = []

    def read_input(
        path: Path, *, maximum_bytes: int, name: str, deadline_ns: int | None = None
    ) -> bytes:
        del path, maximum_bytes, deadline_ns
        return profile_bytes if name == "runner profile" else attempt_bytes

    def fail_profile_temporary_unlink(
        path: object, *args: object, **kwargs: object
    ) -> None:
        nonlocal temporary_unlink_failures
        if str(path).startswith(".profile.json.tmp-"):
            temporary_unlink_failures += 1
            raise OSError("injected persistent profile temp unlink failure")
        real_unlink(path, *args, **kwargs)

    def optionally_fail_profile_write(
        descriptor: int,
        payload: bytes,
        *,
        deadline_ns: int | None = None,
    ) -> None:
        nonlocal prelink_injected
        descriptor_name = Path(
            os.path.realpath(f"/proc/self/fd/{descriptor}")
        ).name
        if (
            uncertainty_point == "prelink"
            and descriptor_name.startswith(".profile.json.tmp-")
            and not prelink_injected
        ):
            prelink_injected = True
            real_write_all(descriptor, payload, deadline_ns=deadline_ns)
            raise OSError("injected profile failure before link")
        real_write_all(descriptor, payload, deadline_ns=deadline_ns)

    def fail_profile_temporary_remove(
        path: object, *args: object, **kwargs: object
    ) -> None:
        if str(path).startswith(".profile.json.tmp-"):
            raise OSError("injected persistent profile temp remove failure")
        real_remove(path, *args, **kwargs)

    monkeypatch.setattr(qualification, "_read_bounded_regular_file_v2", read_input)
    monkeypatch.setattr(
        qualification,
        "load_phase2b_formal_wire_covert_qualification_profile_v2",
        lambda payload: object(),
    )
    monkeypatch.setattr(
        qualification,
        "load_phase2b_formal_wire_covert_attempt_manifest_v2",
        lambda payload: attempt,
    )
    monkeypatch.setattr(qualification.os, "unlink", fail_profile_temporary_unlink)
    if retain_residue:
        monkeypatch.setattr(
            qualification.os, "remove", fail_profile_temporary_remove
        )
    monkeypatch.setattr(
        qualification, "_write_all_fd_v2", optionally_fail_profile_write
    )

    def tracking_attempt_verifier(*args: object, **kwargs: object) -> bytes:
        result = real_attempt_verifier(*args, **kwargs)
        if kwargs.get("prospective_terminal") is True:
            prospective_returned_at.append(qualification.time.monotonic_ns())
        return result

    monkeypatch.setattr(
        qualification,
        "verify_phase2b_formal_wire_covert_outer_attempt_directory_v2",
        tracking_attempt_verifier,
    )
    dummy = tmp_path.resolve() / "unused"
    terminal_bytes = qualification.run_phase2b_formal_wire_covert_outer_attempt_v2(
        entry_kind="development-calibration",
        profile_path=dummy,
        attempt_manifest_path=dummy,
        rfc8785_prerequisite_receipt_path=dummy,
        source_archive_path=dummy,
        runtime_environment_manifest_path=dummy,
        output_root=output_root,
    )
    terminal = qualification._decode_canonical_transport_object_v2(
        terminal_bytes, "test failure terminal", maximum_bytes=8 * 1_024 * 1_024
    )
    attempt_directory = output_root / (
        "attempt-"
        + hashlib.sha256(attempt.attempt_manifest_root.encode("ascii")).hexdigest()[:48]
    )
    assert terminal["disposition"] == "ATOMIC_EMPTY_REJECTION"
    if retain_residue:
        assert len(terminal["managed_atomic_residues"]) == 1
        residue = terminal["managed_atomic_residues"][0]
        assert residue["state"] == (
            "TARGET_AND_MANAGED_TEMP_HARDLINK"
            if uncertainty_point == "postlink"
            else "STAGED_TEMP_ONLY"
        )
        assert (
            attempt_directory
            / "evidence/inputs"
            / residue["temporary_name"]
        ).is_file()
    else:
        assert terminal["managed_atomic_residues"] == []
    assert terminal["claims"][
        "r05_synthetic_dev_wire_covert_qualification_verified"
    ] is False
    assert len(prospective_returned_at) == 1
    assert (
        terminal["timing"]["finalization_ready_monotonic_nanoseconds"]
        >= prospective_returned_at[0]
    )
    assert (attempt_directory / "terminal.json").read_bytes() == terminal_bytes
    assert sorted(path.name for path in attempt_directory.glob("terminal.json")) == [
        "terminal.json"
    ]
    assert temporary_unlink_failures >= (
        2 if uncertainty_point == "postlink" else 1
    )
    residual_paths = [
        path.name.startswith(".profile.json.tmp-")
        for path in attempt_directory.rglob("*")
    ]
    assert sum(residual_paths) == (1 if retain_residue else 0)
    assert (attempt_directory / "evidence/inputs/profile.json").exists() is (
        uncertainty_point == "postlink"
    )
    assert (
        qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
            attempt_directory,
            expected_terminal_bytes=terminal_bytes,
        )
        == terminal_bytes
    )


def test_attempt_mkdir_without_recovered_fd_is_reopened_and_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_bytes = b"synthetic-profile-bytes\n"
    attempt = replace(
        _attempt(),
        qualification_config_sha256=hashlib.sha256(profile_bytes).hexdigest(),
        qualification_lock_sha256=qualification.DEPENDENCY_LOCK_SHA256,
    )
    output_root = (tmp_path / "attempts").resolve()
    output_root.mkdir()
    attempt_name = "attempt-" + hashlib.sha256(
        attempt.attempt_manifest_root.encode("ascii")
    ).hexdigest()[:48]

    monkeypatch.setattr(
        qualification,
        "_read_bounded_regular_file_v2",
        lambda path, **kwargs: (
            profile_bytes
            if kwargs.get("name") == "runner profile"
            else b"synthetic-attempt-bytes\n"
        ),
    )
    monkeypatch.setattr(
        qualification,
        "load_phase2b_formal_wire_covert_qualification_profile_v2",
        lambda payload: object(),
    )
    monkeypatch.setattr(
        qualification,
        "load_phase2b_formal_wire_covert_attempt_manifest_v2",
        lambda payload: attempt,
    )

    def committed_without_descriptor(
        parent_fd: int,
        name: str,
        *,
        expected_path: Path | None = None,
        deadline_ns: int | None = None,
    ) -> int:
        del expected_path, deadline_ns
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        cause = OSError("injected post-mkdir open failure")
        raise qualification._DirectoryCreationCommittedV2(name, None, cause)

    monkeypatch.setattr(
        qualification,
        "_mkdir_new_directory_at_fd_v2",
        committed_without_descriptor,
    )
    dummy = tmp_path.resolve() / "unused"
    with pytest.raises(RuntimeError, match="empty directory rolled back"):
        qualification.run_phase2b_formal_wire_covert_outer_attempt_v2(
            entry_kind="development-calibration",
            profile_path=dummy,
            attempt_manifest_path=dummy,
            rfc8785_prerequisite_receipt_path=dummy,
            source_archive_path=dummy,
            runtime_environment_manifest_path=dummy,
            output_root=output_root,
        )
    assert not (output_root / attempt_name).exists()


def test_terminal_postcommit_parent_replacement_is_atomic_uncertainty(
    tmp_path: Path,
) -> None:
    parent = tmp_path.resolve()
    child = parent / ("attempt-" + "a" * 48)
    moved = parent / "moved-attempt"
    child.mkdir()
    parent_fd = qualification._open_real_directory_fd_v2(parent)
    child_fd = qualification._open_real_directory_fd_v2(child)
    child.rename(moved)
    child.mkdir()
    try:
        with pytest.raises(
            qualification._AtomicPublicationUncertainV2
        ) as captured:
            qualification._verify_parent_identity_after_terminal_commit_v2(
                parent_fd,
                child.name,
                child_fd,
                name="injected replacement",
                deadline_ns=qualification.time.monotonic_ns() + 10_000_000_000,
            )
        assert captured.value.target_name == "terminal.json"
    finally:
        os.close(child_fd)
        os.close(parent_fd)


def test_cas_partial_descriptor_acquisition_closes_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (tmp_path / "source.bin").resolve()
    source.write_bytes(b"source")
    cas = (tmp_path / "cas").resolve()
    cas.mkdir()
    cas_fd = qualification._open_real_directory_fd_v2(cas)
    real_open = qualification.os.open
    real_close = qualification.os.close
    source_descriptors: list[int] = []
    closed: list[int] = []

    def tracking_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        descriptor = real_open(path, flags, *args, **kwargs)
        if str(path) == str(source):
            source_descriptors.append(descriptor)
        return descriptor

    def tracking_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(qualification.os, "open", tracking_open)
    monkeypatch.setattr(qualification.os, "close", tracking_close)
    monkeypatch.setattr(
        qualification.os,
        "dup",
        lambda descriptor: (_ for _ in ()).throw(OSError("injected dup failure")),
    )
    try:
        with pytest.raises(OSError, match="injected dup failure"):
            qualification._cas_put_file_v2(
                cas,
                source,
                expected_sha256=hashlib.sha256(b"source").hexdigest(),
                expected_size=len(b"source"),
                store_namespace="test",
                name="partial acquisition",
                retained_cas_fd=cas_fd,
            )
        assert len(source_descriptors) == 1
        assert source_descriptors[0] in closed
    finally:
        # Use the saved real close because the module-level os object is patched.
        real_close(cas_fd)


def test_cas_after_link_before_python_marker_is_reconciled_as_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (tmp_path / "source.bin").resolve()
    payload = b"CAS linked before Python marker"
    source.write_bytes(payload)
    cas = (tmp_path / "cas").resolve()
    cas.mkdir()
    cas_fd = qualification._open_real_directory_fd_v2(cas)
    real_link = qualification.os.link

    def link_then_interrupt(*args: object, **kwargs: object) -> None:
        real_link(*args, **kwargs)
        raise KeyboardInterrupt("injected CAS post-link interruption")

    monkeypatch.setattr(qualification.os, "link", link_then_interrupt)
    try:
        with pytest.raises(
            qualification._AtomicPublicationUncertainV2
        ) as captured:
            qualification._cas_put_file_v2(
                cas,
                source,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_size=len(payload),
                store_namespace="test",
                name="CAS post-link marker",
                retained_cas_fd=cas_fd,
            )
        assert captured.value.publication_state == "TARGET_LINK_COMMITTED"
        assert qualification._reconcile_nonterminal_atomic_uncertainty_v2(
            captured.value,
            attempt_fd=cas_fd,
            deadline_ns=time.monotonic_ns() + 10_000_000_000,
        ) == []
        assert (cas / hashlib.sha256(payload).hexdigest()).read_bytes() == payload
    finally:
        os.close(cas_fd)


def test_cas_prelink_persistent_staging_residue_is_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (tmp_path / "source.bin").resolve()
    payload = b"CAS staging residue"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    cas = (tmp_path / "cas").resolve()
    cas.mkdir()
    cas_fd = qualification._open_real_directory_fd_v2(cas)
    real_write_all = qualification._write_all_fd_v2
    real_unlink = qualification.os.unlink
    real_remove = qualification.os.remove
    injected = False

    def partial_destination_write(
        descriptor: int, block: bytes, *, deadline_ns: int | None = None
    ) -> None:
        nonlocal injected
        name = Path(os.path.realpath(f"/proc/self/fd/{descriptor}")).name
        if name.startswith(f".{digest}.tmp-") and not injected:
            injected = True
            real_write_all(descriptor, block[:5], deadline_ns=deadline_ns)
            raise OSError("injected CAS prelink write failure")
        real_write_all(descriptor, block, deadline_ns=deadline_ns)

    def persistent_cleanup_failure(
        path: object, *args: object, **kwargs: object
    ) -> None:
        if str(path).startswith(f".{digest}.tmp-"):
            raise OSError("injected CAS staging cleanup failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(qualification, "_write_all_fd_v2", partial_destination_write)
    monkeypatch.setattr(qualification.os, "unlink", persistent_cleanup_failure)
    monkeypatch.setattr(qualification.os, "remove", persistent_cleanup_failure)
    try:
        with pytest.raises(
            qualification._AtomicPublicationUncertainV2
        ) as captured:
            qualification._cas_put_file_v2(
                cas,
                source,
                expected_sha256=digest,
                expected_size=len(payload),
                store_namespace="test",
                name="CAS prelink staging",
                retained_cas_fd=cas_fd,
            )
        assert captured.value.publication_state == "STAGED_TEMP_ONLY"
        residues = qualification._reconcile_nonterminal_atomic_uncertainty_v2(
            captured.value,
            attempt_fd=cas_fd,
            deadline_ns=time.monotonic_ns() + 10_000_000_000,
        )
        assert len(residues) == 1
        assert residues[0]["state"] == "STAGED_TEMP_ONLY"
        real_unlink(str(residues[0]["temporary_name"]), dir_fd=cas_fd)
    finally:
        monkeypatch.setattr(qualification.os, "remove", real_remove)
        os.close(cas_fd)


@pytest.mark.parametrize("failure_at", (2, 3, 4))
def test_incremental_owned_directory_acquisition_closes_each_prior_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_at: int,
) -> None:
    root = tmp_path.resolve()
    for name in ("one", "two", "three", "four"):
        (root / name).mkdir()
    root_fd = qualification._open_real_directory_fd_v2(root)
    real_open = qualification._open_directory_at_fd_v2
    calls = 0
    opened: list[int] = []

    def fail_selected_open(
        parent_fd: int, name: str, *, expected_path: Path | None = None
    ) -> int:
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError(f"injected directory open {failure_at}")
        descriptor = real_open(parent_fd, name, expected_path=expected_path)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(
        qualification, "_open_directory_at_fd_v2", fail_selected_open
    )
    owned: list[int] = []
    try:
        with pytest.raises(OSError, match=f"open {failure_at}"):
            try:
                for name in ("one", "two", "three", "four"):
                    qualification._open_and_retain_directory_at_fd_v2(
                        owned, root_fd, name
                    )
            finally:
                qualification._close_owned_descriptors_v2(
                    reversed(owned), name="injected incremental acquisition"
                )
        assert len(opened) == failure_at - 1
        for descriptor in opened:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        os.close(root_fd)


def test_relative_regular_target_closes_when_parent_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    (root / "target.bin").write_bytes(b"target")
    root_fd = qualification._open_real_directory_fd_v2(root)
    real_dup = qualification.os.dup
    real_close = qualification.os.close
    parent_descriptors: list[int] = []
    target_descriptors: list[int] = []
    closed: list[int] = []

    def tracking_dup(descriptor: int) -> int:
        duplicated = real_dup(descriptor)
        parent_descriptors.append(duplicated)
        return duplicated

    real_open = qualification.os.open

    def tracking_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        descriptor = real_open(path, flags, *args, **kwargs)
        if str(path) == "target.bin":
            target_descriptors.append(descriptor)
        return descriptor

    failed_parent = False

    def fail_parent_close_once(descriptor: int) -> None:
        nonlocal failed_parent
        if descriptor in parent_descriptors and not failed_parent:
            failed_parent = True
            raise OSError("injected parent close failure")
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(qualification.os, "dup", tracking_dup)
    monkeypatch.setattr(qualification.os, "open", tracking_open)
    monkeypatch.setattr(qualification.os, "close", fail_parent_close_once)
    try:
        with pytest.raises(RuntimeError, match="could not close"):
            qualification._open_relative_regular_at_fd_v2(
                root_fd, "target.bin"
            )
        assert len(target_descriptors) == 1
        assert target_descriptors[0] in closed
        with pytest.raises(OSError):
            os.fstat(target_descriptors[0])
    finally:
        # The injected failed close left only the traversal duplicate open.
        for descriptor in parent_descriptors:
            try:
                real_close(descriptor)
            except OSError:
                pass
        real_close(root_fd)


def test_source_archive_parent_acquisition_failure_closes_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = (tmp_path / "archive.tar").resolve()
    archive.write_bytes(b"not-read-after-parent-failure")
    destination = tmp_path / "fresh-tree"
    real_open = qualification.os.open
    real_close = qualification.os.close
    archive_descriptors: list[int] = []
    closed: list[int] = []

    def tracking_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        descriptor = real_open(path, flags, *args, **kwargs)
        if str(path) == str(archive):
            archive_descriptors.append(descriptor)
        return descriptor

    def tracking_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(qualification.os, "open", tracking_open)
    monkeypatch.setattr(qualification.os, "close", tracking_close)
    monkeypatch.setattr(
        qualification,
        "_open_real_directory_fd_v2",
        lambda path: (_ for _ in ()).throw(OSError("injected parent failure")),
    )
    with pytest.raises(OSError, match="injected parent failure"):
        qualification._safe_extract_source_archive_v2(
            archive,
            destination,
            expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            expected_size=archive.stat().st_size,
        )
    assert len(archive_descriptors) == 1
    assert archive_descriptors[0] in closed


def test_archive_empty_directory_path_must_equal_retained_full_tree(
    tmp_path: Path,
) -> None:
    archive = (tmp_path / "source.tar").resolve()
    payload = b"same-regular-file\n"
    with tarfile.open(archive, "w") as bundle:
        directory = tarfile.TarInfo("empty-a/")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o700
        bundle.addfile(directory)
        regular = tarfile.TarInfo("payload.txt")
        regular.size = len(payload)
        regular.mode = 0o600
        bundle.addfile(regular, io.BytesIO(payload))
    archive_bytes = archive.read_bytes()
    (
        archive_regular_root,
        archive_regular_count,
        archive_regular_bytes,
        _,
        _,
        archive_rows,
        archive_root,
    ) = qualification._inspect_source_archive_regular_manifest_v2(
        archive,
        expected_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        expected_size=len(archive_bytes),
    )
    assert {row["path"] for row in archive_rows} == {"empty-a", "payload.txt"}
    retained = (tmp_path / "retained").resolve()
    (retained / "empty-b").mkdir(parents=True)
    (retained / "payload.txt").write_bytes(payload)
    retained_rows, retained_root = qualification._source_tree_manifest_v2(retained)
    assert qualification._regular_file_manifest_root_v2(retained_rows) == (
        archive_regular_root
    )
    assert archive_regular_count == 1
    assert archive_regular_bytes == len(payload)
    with pytest.raises(ValueError, match="full-tree paths"):
        qualification._verify_source_archive_full_tree_equivalence_v2(
            archive_rows,
            archive_root,
            retained_rows,
            retained_root,
        )


def test_close_owned_descriptors_continues_after_first_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def injected_close(descriptor: int) -> None:
        calls.append(descriptor)
        if descriptor == 101:
            raise OSError("first close failed")

    monkeypatch.setattr(qualification.os, "close", injected_close)
    with pytest.raises(RuntimeError, match="could not close 1"):
        qualification._close_owned_descriptors_v2(
            (101, 102), name="injected close set"
        )
    assert calls == [101, 102]


def test_finalization_reserve_covers_two_bounded_tree_scans_and_cleanup() -> None:
    assert qualification._OUTER_TERMINAL_PUBLICATION_RESERVE_SECONDS_V2 == 2_400
    assert qualification._OUTER_FAILURE_CLEANUP_RESERVE_SECONDS_V2 == 3_000
    streamed_seconds = (
        qualification._FINALIZATION_TREE_SCAN_PASSES_V2
        * qualification._MAXIMUM_RETAINED_ATTEMPT_TREE_BYTES_V2
        // qualification._FINALIZATION_MINIMUM_STREAM_BYTES_PER_SECOND_V2
    )
    metadata_seconds = (
        qualification._FINALIZATION_TREE_SCAN_PASSES_V2
        * qualification._MAXIMUM_RETAINED_ATTEMPT_TREE_ROWS_V2
        // qualification._FINALIZATION_MINIMUM_METADATA_ROWS_PER_SECOND_V2
    )
    assert (
        streamed_seconds
        + metadata_seconds
        + qualification._FINALIZATION_FSYNC_AND_PUBLICATION_MARGIN_SECONDS_V2
        == qualification._OUTER_TERMINAL_PUBLICATION_RESERVE_SECONDS_V2
    )
    assert (
        qualification._OUTER_FAILURE_CLEANUP_RESERVE_SECONDS_V2
        - qualification._OUTER_TERMINAL_PUBLICATION_RESERVE_SECONDS_V2
        == qualification._OUTER_CLEANUP_CONTROL_RESERVE_SECONDS_V2
    )


def test_bounded_command_timeout_kills_stubborn_group_reaps_and_closes_pipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = qualification.subprocess.Popen
    observed: list[subprocess.Popen[bytes]] = []

    def recording_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)
        observed.append(process)
        return process

    monkeypatch.setattr(qualification.subprocess, "Popen", recording_popen)
    started = time.monotonic_ns()
    with pytest.raises(TimeoutError, match="absolute deadline"):
        qualification._run_bounded_control_command_v2(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                (
                    "import signal,time;"
                    "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                    "print('ready',flush=True);time.sleep(60)"
                ),
            ],
            purpose="test-stubborn-timeout",
            deadline_ns=started + 300_000_000,
            cleanup_deadline_ns=started + 6_000_000_000,
        )
    assert len(observed) == 1
    process = observed[0]
    assert process.poll() is not None
    assert process.stdout is not None and process.stdout.closed
    assert process.stderr is not None and process.stderr.closed
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)


def test_bounded_command_expired_deadline_rejects_before_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        qualification.subprocess,
        "Popen",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    with pytest.raises(TimeoutError, match="before child launch"):
        qualification._run_bounded_control_command_v2(
            [sys.executable, "--version"],
            purpose="test-expired-command",
            deadline_ns=time.monotonic_ns() - 1,
        )
    assert calls == []


def test_cleanup_helper_reaps_launcher_then_finalizer_never_resignals_old_pgid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, source_tree, intent = _launch_intent()
    launcher = subprocess.Popen(
        [sys.executable, "-I", "-S", "-B", "-c", "pass"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    commands = [intent["prelaunch_command"]]
    absent = _systemd_show_bytes(attempt, source_tree)

    def exact_absent_show(
        argv: list[str], *, purpose: str, **kwargs: object
    ) -> tuple[dict[str, object], bytes, bytes]:
        del kwargs
        return (
            qualification._command_evidence_v2(
                argv=argv,
                purpose=purpose,
                returncode=0,
                stdout=absent,
                stderr=b"",
            ),
            absent,
            b"",
        )

    monkeypatch.setattr(
        qualification, "_run_bounded_control_command_v2", exact_absent_show
    )
    cleanup = qualification._cleanup_transient_attempt_v2(
        launcher=launcher,
        systemctl_realpath=intent["executables"]["systemctl"]["realpath"],
        unit_name=intent["unit_name"],
        ownership_marker=intent["ownership_marker"],
        owned_invocation_id=None,
        prelaunch_absent_verified=True,
        attempt=attempt,
        source_tree=source_tree,
        deadline_ns=time.monotonic_ns() + 10_000_000_000,
        commands=commands,
        cleanup_start_command_index=1,
    )
    assert cleanup["launcher_reaped"] is True
    assert launcher.poll() == 0
    signals: list[int] = []
    monkeypatch.setattr(
        qualification,
        "_terminate_process_group_v2",
        lambda process, *, deadline_ns: signals.append(process.pid),
    )
    qualification._finalize_outer_launcher_v2(
        launcher, deadline_ns=time.monotonic_ns() + 5_000_000_000
    )
    assert signals == []
    assert launcher.stdout is not None and launcher.stdout.closed
    assert launcher.stderr is not None and launcher.stderr.closed


def test_finalizer_inner_poll_reap_never_signals_reused_numeric_pgid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        def __init__(self, descriptor: int) -> None:
            self._descriptor = descriptor
            self.closed = False

        def fileno(self) -> int:
            return self._descriptor

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        pid = 424242

        def __init__(self) -> None:
            self.stdout = FakeStream(101)
            self.stderr = FakeStream(102)
            self._poll_results = iter((None, 0))
            self.poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            return next(self._poll_results)

    process = FakeProcess()
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        qualification.os,
        "killpg",
        lambda pgid, sig: killpg_calls.append((pgid, sig)),
    )
    qualification._finalize_outer_launcher_v2(
        process,  # type: ignore[arg-type]
        deadline_ns=time.monotonic_ns() + 5_000_000_000,
    )
    assert process.poll_count == 2
    assert killpg_calls == []
    assert process.stdout.closed
    assert process.stderr.closed


def test_finalizer_term_grace_reap_never_probes_or_signals_old_pgid_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStream:
        def __init__(self, descriptor: int) -> None:
            self._descriptor = descriptor
            self.closed = False

        def fileno(self) -> int:
            return self._descriptor

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        pid = 424243

        def __init__(self) -> None:
            self.stdout = FakeStream(103)
            self.stderr = FakeStream(104)
            self._poll_results = iter((None, None, 0))
            self.poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            return next(self._poll_results)

    process = FakeProcess()
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        qualification.os,
        "killpg",
        lambda pgid, sig: killpg_calls.append((pgid, sig)),
    )
    qualification._finalize_outer_launcher_v2(
        process,  # type: ignore[arg-type]
        deadline_ns=time.monotonic_ns() + 5_000_000_000,
    )
    assert process.poll_count == 3
    assert killpg_calls == [
        (process.pid, 0),
        (process.pid, int(signal.SIGTERM)),
    ]
    assert process.stdout.closed
    assert process.stderr.closed


def test_cleanup_early_failure_finalizer_kills_stubborn_launcher_and_closes_pipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, source_tree, intent = _launch_intent()
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            (
                "import signal,time;"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "print('ready',flush=True);time.sleep(60)"
            ),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert launcher.stdout is not None
    assert launcher.stdout.readline() == b"ready\n"
    commands = [intent["prelaunch_command"]]
    monkeypatch.setattr(
        qualification,
        "_run_bounded_control_command_v2",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected first cleanup show failure")
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="first cleanup show"):
            qualification._cleanup_transient_attempt_v2(
                launcher=launcher,
                systemctl_realpath=intent["executables"]["systemctl"]["realpath"],
                unit_name=intent["unit_name"],
                ownership_marker=intent["ownership_marker"],
                owned_invocation_id=None,
                prelaunch_absent_verified=True,
                attempt=attempt,
                source_tree=source_tree,
                deadline_ns=time.monotonic_ns() + 10_000_000_000,
                commands=commands,
                cleanup_start_command_index=1,
            )
    finally:
        qualification._finalize_outer_launcher_v2(
            launcher, deadline_ns=time.monotonic_ns() + 6_000_000_000
        )
    assert launcher.poll() == -int(signal.SIGKILL)
    assert launcher.stdout.closed
    assert launcher.stderr is not None and launcher.stderr.closed
    with pytest.raises(ProcessLookupError):
        os.killpg(launcher.pid, 0)


def test_rfc_raw_expired_deadline_rejects_before_any_fresh_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner_calls: list[object] = []
    monkeypatch.setattr(
        qualification,
        "_run_bounded_control_command_v2",
        lambda *args, **kwargs: runner_calls.append((args, kwargs)),
    )
    with pytest.raises(TimeoutError, match="RFC raw entry"):
        qualification._verify_rfc8785_raw_execution_bundle_v2(
            {},
            tmp_path.resolve(),
            expected_qualification_source_sha256="0" * 64,
            fresh_execute=True,
            deadline_ns=qualification.time.monotonic_ns() - 1,
        )
    assert runner_calls == []


def test_rfc_raw_post_open_shape_failure_closes_all_owned_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    bundle = root / "bundle"
    bundle.mkdir()
    qualification_sha = "0" * 64
    raw = {
        "bundle_manifest": [],
        "commands": [],
        "qualification_source_sha256": qualification_sha,
        "raw_artifact_total_bytes": 0,
        "transcript_root": "unused-before-command-order-rejection",
        "working_tree_root": str(root),
    }
    receipt = {
        "executables": {
            "node": {"realpath": "/usr/bin/node"},
            "python": {"realpath": "/usr/bin/python3"},
        },
        "raw_execution": raw,
    }
    evidence_bytes = qualification.canonicalize_rfc8785_v1(receipt)
    evidence_sha = hashlib.sha256(evidence_bytes).hexdigest()
    (bundle / evidence_sha).write_bytes(evidence_bytes)
    expected = {
        "docs/Hegel_Machine_Phase2B_RFC8785_Implementation_v1.md": (
            qualification.RFC8785_DOCUMENTATION_SHA256
        ),
        "golden_vectors/phase2b_rfc8785_v1.json": (
            qualification.RFC8785_GOLDEN_VECTORS_SHA256
        ),
        "src/hegel_machine/rfc8785_v1.py": qualification.RFC8785_SOURCE_SHA256,
        "tests/test_rfc8785_v1.py": qualification.RFC8785_TEST_SHA256,
        "tools/phase2b_rfc8785_oracle_v1.mjs": (
            qualification.RFC8785_NODE_ORACLE_SHA256
        ),
        **qualification._BOOTSTRAP_FIXED_SOURCE_SHA256_V2,
        qualification._BOOTSTRAP_MAIN_RELATIVE_PATH_V2: qualification_sha,
    }

    def fake_hash(
        root_fd: int,
        relative: str,
        *,
        maximum_bytes: int,
        deadline_ns: int | None = None,
    ) -> tuple[str, object]:
        del root_fd, maximum_bytes, deadline_ns
        return expected[relative], object()

    monkeypatch.setattr(
        qualification, "_hash_relative_regular_at_fd_v2", fake_hash
    )
    monkeypatch.setattr(
        qualification,
        "_read_relative_regular_at_fd_v2",
        lambda *args, **kwargs: (b"sealed", object()),
    )
    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(ValueError, match="command order drifted"):
        qualification._verify_rfc8785_raw_execution_bundle_v2(
            receipt,
            bundle,
            expected_qualification_source_sha256=qualification_sha,
            replay_project_root=root,
            fresh_execute=False,
        )
    after = len(os.listdir("/proc/self/fd"))
    assert after == before


def test_rfc_sealed_subprocess_rebases_oracle_and_vectors_to_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle = (tmp_path / "oracle.mjs").resolve()
    vectors = (tmp_path / "vectors.json").resolve()
    calls: list[tuple[list[str], dict[str, object]]] = []
    sentinel = object()

    def fake_run(argv: list[str], **kwargs: object) -> object:
        calls.append((argv, kwargs))
        return sentinel

    monkeypatch.setattr(qualification.subprocess, "run", fake_run)
    proxy = qualification._RFC8785SealedSubprocessV2(
        node_fd=11,
        oracle_display=oracle,
        oracle_bytes=b"export const sealed = true;\n",
        vectors_display=vectors,
        vectors_fd=12,
    )
    assert proxy.run(
        ["node", str(oracle), "--check", str(vectors)],
        check=True,
        capture_output=True,
        text=True,
    ) is sentinel
    argv, kwargs = calls[-1]
    assert argv[:4] == [
        "/proc/self/fd/11",
        "--input-type=module",
        "--eval",
        "export const sealed = true;\n",
    ]
    assert argv[-1] == "/proc/self/fd/12"
    assert kwargs["pass_fds"] == (11, 12)
    rng = random.Random(qualification._RFC8785_RANDOM_DIFFERENTIAL_SEED_V2)

    def next_batch() -> list[str]:
        values: list[str] = []
        while len(values) < 2_000:
            bits = rng.getrandbits(64)
            value = struct.unpack(">d", bits.to_bytes(8, "big"))[0]
            if math.isfinite(value):
                values.append(f"{bits:016x}")
        return values

    with pytest.raises(ValueError, match="batch drifted"):
        proxy.run(
            ["node", str(oracle), "--number-bits", "0000000000000000"],
            check=True,
            capture_output=True,
            text=True,
        )
    for _ in range(50):
        assert proxy.run(
            ["node", str(oracle), "--number-bits", *next_batch()],
            check=True,
            capture_output=True,
            text=True,
        ) is sentinel
        assert calls[-1][1]["pass_fds"] == (11,)
    proxy.assert_complete()
    with pytest.raises(ValueError, match="batch drifted"):
        proxy.run(
            ["node", str(oracle), "--number-bits", *("0" * 16 for _ in range(2_000))],
            check=True,
            capture_output=True,
            text=True,
        )
    with pytest.raises(ValueError, match="allowlist"):
        proxy.run(
            ["node", str(tmp_path / "replacement.mjs"), "--check", str(vectors)],
            check=True,
            capture_output=True,
            text=True,
        )

    malformed = qualification._RFC8785SealedSubprocessV2(
        node_fd=11,
        oracle_display=oracle,
        oracle_bytes=b"export const sealed = true;\n",
        vectors_display=vectors,
        vectors_fd=12,
    )
    malformed.run(
        ["node", str(oracle), "--check", str(vectors)],
        check=True,
        capture_output=True,
        text=True,
    )
    invalid = ["0" * 16 for _ in range(2_000)]
    invalid[17] = "A" * 16
    with pytest.raises(ValueError, match="batch drifted"):
        malformed.run(
            ["node", str(oracle), "--number-bits", *invalid],
            check=True,
            capture_output=True,
            text=True,
        )
    with pytest.raises(ValueError, match="did not close"):
        malformed.assert_complete()


@pytest.mark.slow
def test_real_rfc_builder_source_only_18_node_100k_and_fresh_replay(
    tmp_path: Path,
) -> None:
    project_root = ROOT.resolve(strict=True)
    bundle = (tmp_path / "real-rfc8785-bundle").resolve()
    qualification_source = (
        project_root
        / "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py"
    )
    qualification_sha256 = hashlib.sha256(
        qualification_source.read_bytes()
    ).hexdigest()
    receipt_bytes = qualification.build_rfc8785_prerequisite_bundle_v2(
        project_root=project_root,
        bundle_directory=bundle,
    )
    receipt = qualification._decode_canonical_transport_object_v2(
        receipt_bytes,
        "real RFC prerequisite receipt",
        maximum_bytes=1_048_576,
    )
    assert receipt["source_only_kat_18"] == {
        "collected": 18,
        "exit_code": 0,
        "passed": 18,
        "runner": "STDLIB_SOURCE_ONLY_PINNED_TEST_HARNESS_V1",
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "stdout_sha256": hashlib.sha256(b".................. [100%]\n").hexdigest(),
    }
    assert receipt["differential"]["finite_binary64_cases"] == 100_000
    assert receipt["differential"]["mismatch_count"] == 0
    projection = qualification._verify_rfc8785_raw_execution_bundle_v2(
        receipt,
        bundle,
        expected_qualification_source_sha256=qualification_sha256,
        replay_project_root=project_root,
        fresh_execute=True,
        deadline_ns=time.monotonic_ns() + 1_800_000_000_000,
    )
    assert projection["command_count"] == 56
    assert projection["raw_artifact_total_bytes"] > 0


@pytest.mark.slow
@pytest.mark.parametrize("mode", qualification.LIVE_MODE_ORDER)
def test_rich32_invariance_core_only_executes_one_atomic_80_schedule_suite(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = qualification._build_fixture_execution_v2(
        mode, row_count=32
    )
    decoded = tuple(
        execution.adapter.decode_envelope(row) for row in execution.adapter.envelopes
    )
    for item in decoded:
        auditor._validate_decoded(item, execution.adapter)
    recognizer = execution.recognizer
    pins, _ = auditor._capture_adapter_execution_pins(execution.adapter, recognizer)
    input_root = auditor._input_root(execution.adapter.envelopes)
    snapshot_root = auditor._invariance_input_snapshot_root(
        execution.adapter.envelopes,
        decoded,
        execution.adapter,
        recognizer,
        input_root,
        pins,
    )
    original_decode = auditor.FormalWireV3AdapterV1.decode_envelope
    original_suite = auditor.FormalWireV3AdapterV1.make_variant_suite_envelopes
    original_recognizer = qualification._SyntheticRichRecognizerV2.run_batch
    decode_count = 0
    decode_seconds = 0.0
    suite_seconds = 0.0
    recognizer_seconds = 0.0
    recognizer_batch_count = 0
    transformed_equivalence_count = 0
    trusted_projection_by_sha256 = {
        hashlib.sha256(envelope).digest(): (
            qualification._trusted_synthetic_public_projection_v2(item)
        )
        for envelope, item in zip(
            execution.adapter.envelopes, decoded, strict=True
        )
    }

    def decode_spy(
        self: auditor.FormalWireV3AdapterV1, envelope: bytes
    ) -> auditor.DecodedFormalEnvelopeV1:
        nonlocal decode_count, decode_seconds
        started = time.monotonic()
        try:
            item = original_decode(self, envelope)
            trusted_projection_by_sha256[hashlib.sha256(envelope).digest()] = (
                qualification._trusted_synthetic_public_projection_v2(item)
            )
            return item
        finally:
            decode_count += 1
            decode_seconds += time.monotonic() - started

    def suite_spy(
        self: auditor.FormalWireV3AdapterV1,
        **kwargs: object,
    ) -> object:
        nonlocal suite_seconds
        started = time.monotonic()
        try:
            return original_suite(self, **kwargs)
        finally:
            suite_seconds += time.monotonic() - started

    def recognizer_spy(
        self: qualification._SyntheticRichRecognizerV2,
        envelopes: tuple[bytes, ...],
    ) -> tuple[auditor.FormalRecognizerDecisionV1, ...]:
        nonlocal recognizer_batch_count, recognizer_seconds
        nonlocal transformed_equivalence_count
        started = time.monotonic()
        try:
            decisions = original_recognizer(self, envelopes)
            for envelope, decision in zip(envelopes, decisions, strict=True):
                trusted = trusted_projection_by_sha256.get(
                    hashlib.sha256(envelope).digest()
                )
                observed = (
                    decision.family,
                    decision.binding,
                    decision.scale_set,
                )
                if trusted is None or observed != trusted:
                    raise AssertionError(
                        "rich32 lightweight/trusted decision projection drifted"
                    )
                if recognizer_batch_count > 0:
                    transformed_equivalence_count += 1
            recognizer_batch_count += 1
            return decisions
        finally:
            recognizer_seconds += time.monotonic() - started

    monkeypatch.setattr(
        auditor.FormalWireV3AdapterV1, "decode_envelope", decode_spy
    )
    monkeypatch.setattr(
        auditor.FormalWireV3AdapterV1, "make_variant_suite_envelopes", suite_spy
    )
    monkeypatch.setattr(
        qualification._SyntheticRichRecognizerV2, "run_batch", recognizer_spy
    )
    started = time.monotonic()
    result = auditor._run_invariance_core(
        execution.adapter.envelopes,
        decoded,
        execution.adapter,
        recognizer,
        input_root,
        snapshot_root,
        pins,
    )
    total_seconds = time.monotonic() - started
    timing = {
        "decode_seconds": decode_seconds,
        "mode": mode,
        "recognizer_seconds": recognizer_seconds,
        "suite_seconds_inclusive": suite_seconds,
        "total_seconds": total_seconds,
    }
    assert result.exact_invariant_comparisons == 32 * 80 * 5
    assert result.consumer_decode_count == 32 * 80
    assert decode_count == 32 * 80
    assert recognizer_batch_count == 81
    assert transformed_equivalence_count == 32 * 80
    assert result.variant_suite_unique_envelope_object_count <= 32 * 49
    assert recognizer.calls == 81
    assert recognizer.row_counts == [32] * 81
    assert total_seconds <= 225.0, timing
