from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import gc
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
from hegel_machine import phase2b_exact_transform_core_v1 as exact_core
from hegel_machine import phase2b_formal_trusted_wire_v3 as wire
from hegel_machine import phase2b_formal_wire_covert_qualification_v2 as qualification


ROOT = Path(__file__).parents[1]


def _test_auditor_cgroup_snapshot(
    *,
    current: int = 4_096,
    peak: int = 8_192,
) -> auditor.UnifiedCgroupMemorySnapshotV1:
    membership = "/test.slice/qv2.scope"
    device = 11
    inode = 22
    return auditor.UnifiedCgroupMemorySnapshotV1(
        unified_membership=membership,
        memory_scope_root=qualification.stable_hash(
            {
                "leaf_device": device,
                "leaf_inode": inode,
                "unified_membership": membership,
            },
            prefix="phase2b_formal_covert_unified_cgroup_memory_scope_v1_",
        ),
        leaf_device=device,
        leaf_inode=inode,
        memory_current_bytes=current,
        memory_peak_bytes=peak,
        memory_max_bytes=24 * 1_024 * 1_024 * 1_024,
    )


@lru_cache(maxsize=2)
def _fixture(mode: str) -> qualification._FixtureExecutionV2:
    return qualification._build_fixture_execution_v2(mode, row_count=8)


def _decode_batch_with_production_guards(
    adapter: auditor.FormalWireV3AdapterV1,
    envelopes: tuple[bytes, ...],
) -> tuple[auditor.DecodedFormalEnvelopeV1, ...]:
    auditor._preflight_decode_memory(len(envelopes))
    decoded: list[auditor.DecodedFormalEnvelopeV1] = []
    for envelope in envelopes:
        item = adapter.decode_envelope(envelope)
        if type(item) is not auditor.DecodedFormalEnvelopeV1:
            raise TypeError("resource-only decode returned a non-exact DTO")
        auditor._validate_decoded(item, adapter)
        if (
            auditor.DEEP_SIZE_SAFETY_MULTIPLIER
            * auditor._deep_object_graph_bytes(item)
            > auditor.MAXIMUM_DECODE_GRAPH_BYTES_PER_ROW
        ):
            raise ValueError("resource-only decoded row exceeds the deep graph bound")
        decoded.append(item)
    return tuple(decoded)


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
        qualification.OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2,
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
    overrides: dict[str, str] | None = None,
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
    if overrides is not None:
        if type(overrides) is not dict or any(
            key not in values or type(value) is not str
            for key, value in overrides.items()
        ):
            raise ValueError("test systemd show overrides are malformed")
        values.update(overrides)
    return "".join(f"{key}={values[key]}\n" for key in qualification._SYSTEMD_SHOW_PROPERTIES_V2).encode()


def _launch_intent(
    attempt: qualification.Phase2BFormalWireCovertAttemptManifestV2 | None = None,
    source_tree: Path | None = None,
) -> tuple[
    qualification.Phase2BFormalWireCovertAttemptManifestV2,
    Path,
    dict[str, object],
]:
    attempt = _attempt() if attempt is None else attempt
    source_tree = (
        Path("/tmp/hegel-qv2-attempt/evidence/source-tree")
        if source_tree is None
        else source_tree
    )
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


def _unproven_cleanup_projection(
    intent: dict[str, object],
    commands: list[dict[str, object]],
    *,
    boundary: int,
    owned_invocation_id: str | None,
) -> dict[str, object]:
    return {
        "cleanup_error_type": "RuntimeError",
        "cleanup_start_command_index": boundary,
        "cleanup_verified": False,
        "commands": commands,
        "external_intervention_required": True,
        "owned_invocation_id": owned_invocation_id,
        "ownership_marker": intent["ownership_marker"],
        "prelaunch_unit_absent": True,
        "transcript_root": qualification.stable_hash(
            commands,
            prefix="phase2b_formal_wire_covert_failure_cleanup_v2_",
        ),
        "unit_name": intent["unit_name"],
    }


def _materialize_launched_failure_v4(
    root: Path,
    *,
    cleanup_verified: bool = True,
    capture_persistence_verified: bool = True,
    capture_truncated: bool = False,
    capture_complete: bool | None = None,
) -> tuple[Path, bytes, dict[str, object], dict[str, object]]:
    if capture_complete is None:
        capture_complete = cleanup_verified
    if type(capture_complete) is not bool:
        raise TypeError("test capture completion must be exact bool")
    profile_bytes = b"failure-v4-profile\n"
    runtime_bytes = b"failure-v4-runtime\n"
    attempt = replace(
        _attempt(),
        qualification_config_sha256=hashlib.sha256(profile_bytes).hexdigest(),
        runtime_environment_manifest_root=(
            "sha256:" + hashlib.sha256(runtime_bytes).hexdigest()
        ),
        runtime_environment_manifest_size_bytes=len(runtime_bytes),
    )
    attempt_name = "attempt-" + hashlib.sha256(
        attempt.attempt_manifest_root.encode("ascii")
    ).hexdigest()[:48]
    attempt_directory = root / attempt_name
    evidence = attempt_directory / "evidence"
    inputs = evidence / "inputs"
    cas = evidence / "cas"
    source_tree = evidence / "source-tree"
    rfc_raw_bundle = evidence / "rfc8785-raw-bundle"
    inputs.mkdir(parents=True)
    cas.mkdir()
    source_tree.mkdir()
    rfc_raw_bundle.mkdir()
    _, _, intent = _launch_intent(attempt, source_tree)
    attempt_bytes = qualification.phase2b_formal_wire_covert_attempt_manifest_bytes_v2(
        attempt
    )
    launch_intent_bytes = qualification._canonical_inner_artifact_v2(intent)
    (inputs / "profile.json").write_bytes(profile_bytes)
    (inputs / "attempt.json").write_bytes(attempt_bytes)
    (inputs / "runtime-environment.json").write_bytes(runtime_bytes)
    (evidence / "launch-intent.json").write_bytes(launch_intent_bytes)
    reservation = qualification._canonical_inner_artifact_v2(
        {
            "schema_version": (
                "hegel-machine-phase2b-formal-wire-covert-attempt-reservation/2"
            ),
            "campaign_id": attempt.campaign_id,
            "attempt_id": attempt.attempt_id,
            "profile_id": attempt.profile_id,
            "attempt_manifest_root": attempt.attempt_manifest_root,
            "entry_kind": "development-calibration",
            "reserved_exclusively": True,
            "terminal_publication_count": 0,
        }
    )
    (attempt_directory / "reservation.json").write_bytes(reservation)

    stdout_prefix = b"stdout-prefix\x00"
    stderr_prefix = b"stderr-prefix\xff"
    stdout_tail = b"stdout-tail\n"
    stderr_tail = (
        b"T" * (256 * 1_024 - len(stderr_prefix))
        if capture_truncated
        else b"stderr-tail\x80"
    )
    if capture_truncated and capture_complete:
        raise ValueError("test fixture cannot claim complete truncated capture")
    evidence_fd = qualification._open_real_directory_fd_v2(evidence)
    cas_fd = qualification._open_real_directory_fd_v2(cas)
    attempt_fd = qualification._open_real_directory_fd_v2(attempt_directory)
    deadline = time.monotonic_ns() + 30_000_000_000
    try:
        prefix = qualification._persist_failure_child_pipe_prefix_v2(
            evidence_fd,
            stdout=stdout_prefix,
            stderr=stderr_prefix,
            deadline_ns=deadline,
        )
        if capture_persistence_verified:
            capture = qualification._persist_failure_child_pipe_capture_v2(
                evidence_fd,
                cas_fd,
                stdout=stdout_prefix + stdout_tail,
                stderr=stderr_prefix + stderr_tail,
                prefix_projection=prefix,
                capture_complete=capture_complete,
                capture_truncated=capture_truncated,
                store_namespace=attempt.attempt_manifest_root,
                deadline_ns=deadline,
            )
        else:
            capture = (
                qualification._failure_child_pipe_persistence_unverified_projection_v2(
                    evidence_fd,
                    persistence_error_type="OSError",
                    capture_truncated=capture_truncated,
                    deadline_ns=deadline,
                )
            )
        absent = _systemd_show_bytes(attempt, source_tree)
        show_argv = qualification._systemd_show_argv_v2(
            intent["executables"]["systemctl"]["realpath"], intent["unit_name"]
        )
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
            "cleanup_verified": cleanup_verified,
            "commands": commands,
            "owned_invocation_id": None,
            "ownership_marker": intent["ownership_marker"],
            "prelaunch_unit_absent": True,
            "transcript_root": qualification.stable_hash(
                commands,
                prefix="phase2b_formal_wire_covert_failure_cleanup_v2_",
            ),
            "unit_name": intent["unit_name"],
        }
        if cleanup_verified:
            cleanup.update(
                drained_stderr_sha256=hashlib.sha256(stderr_tail).hexdigest(),
                drained_stderr_size_bytes=len(stderr_tail),
                drained_stdout_sha256=hashlib.sha256(stdout_tail).hexdigest(),
                drained_stdout_size_bytes=len(stdout_tail),
                kill_sent=False,
                launcher_reaped=True,
                launcher_returncode=1,
                post_cleanup_unit_absent=True,
                term_sent=False,
            )
        else:
            cleanup.update(
                cleanup_error_type="RuntimeError",
                external_intervention_required=True,
            )
        failure_projection = (
            qualification._derive_failure_v4_retained_projection_at_fd_v2(
                attempt_fd,
                managed_atomic_residues=[],
                deadline_ns=deadline,
            )
        )
        failure_cas_projection = failure_projection["failure_cas_projection"]
        tree = failure_projection["retained_tree_before_terminal"]
        started = time.monotonic_ns()
        terminal_bytes = qualification._closed_outer_failure_artifact_v2(
            attempt,
            entry_kind="development-calibration",
            stage="INNER_EXECUTION",
            primary_failure_exception_type="ValueError",
            reason_code=qualification.FAIL_ISOLATED_EXECUTION,
            cleanup=cleanup,
            cleanup_failure_exception_type=(
                None if cleanup_verified else "RuntimeError"
            ),
            child_pipe_capture=capture,
            failure_cas_projection=failure_cas_projection,
            cleanup_verified=cleanup_verified,
            external_intervention_required=not cleanup_verified,
            launcher_started=True,
            managed_atomic_residues=[],
            retained_tree_before_terminal=tree,
            started_ns=started,
            deadline_ns=(
                started
                + qualification.LIVE_HARD_TOTAL_TIMEOUT_SECONDS * 1_000_000_000
            ),
        )
    finally:
        os.close(attempt_fd)
        os.close(cas_fd)
        os.close(evidence_fd)
    return attempt_directory, terminal_bytes, capture, cleanup


def test_failure_v4_child_pipe_capture_round_trips_prospective_and_published(
    tmp_path: Path,
) -> None:
    attempt_directory, terminal_bytes, capture, cleanup = (
        _materialize_launched_failure_v4(tmp_path)
    )
    assert qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
        attempt_directory,
        expected_terminal_bytes=terminal_bytes,
        prospective_terminal=True,
    ) == terminal_bytes
    (attempt_directory / "terminal.json").write_bytes(terminal_bytes)
    assert qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
        attempt_directory,
        expected_terminal_bytes=terminal_bytes,
    ) == terminal_bytes
    terminal = qualification._decode_canonical_transport_object_v2(
        terminal_bytes,
        "failure /4 test terminal",
        maximum_bytes=8 * 1_024 * 1_024,
    )
    assert terminal["schema_version"] == qualification.OUTER_ATTEMPT_FAILURE_SCHEMA_V2
    assert terminal["primary_failure_stage"] == "INNER_EXECUTION"
    assert terminal["primary_failure_exception_type"] == "ValueError"
    assert terminal["cleanup_failure_exception_type"] is None
    assert terminal["cleanup"] == cleanup
    assert terminal["child_pipe_capture"] == capture
    assert capture["capture_complete"] is True
    assert capture["capture_boundary"] == "COMPLETE_EOF_AND_LAUNCHER_REAPED"


@pytest.mark.parametrize(
    ("cleanup_verified", "capture_persistence_verified", "disposition"),
    (
        (
            False,
            True,
            "ATOMIC_FAILURE_CLEANUP_UNPROVEN_EXTERNAL_INTERVENTION_REQUIRED",
        ),
        (
            True,
            False,
            "ATOMIC_FAILURE_PIPE_CAPTURE_PERSISTENCE_UNVERIFIED",
        ),
        (
            False,
            False,
            "ATOMIC_FAILURE_CLEANUP_UNPROVEN_EXTERNAL_INTERVENTION_REQUIRED",
        ),
    ),
)
def test_failure_v4_negative_capture_boundaries_are_closed_and_nonpromotable(
    tmp_path: Path,
    cleanup_verified: bool,
    capture_persistence_verified: bool,
    disposition: str,
) -> None:
    attempt_directory, terminal_bytes, capture, _ = (
        _materialize_launched_failure_v4(
            tmp_path,
            cleanup_verified=cleanup_verified,
            capture_persistence_verified=capture_persistence_verified,
        )
    )
    assert qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
        attempt_directory,
        expected_terminal_bytes=terminal_bytes,
        prospective_terminal=True,
    ) == terminal_bytes
    (attempt_directory / "terminal.json").write_bytes(terminal_bytes)
    assert qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
        attempt_directory,
        expected_terminal_bytes=terminal_bytes,
    ) == terminal_bytes
    terminal = qualification._decode_canonical_transport_object_v2(
        terminal_bytes,
        "negative failure /4 test terminal",
        maximum_bytes=8 * 1_024 * 1_024,
    )
    assert terminal["disposition"] == disposition
    assert terminal["primary_failure_stage"] == "INNER_EXECUTION"
    assert terminal["primary_failure_exception_type"] == "ValueError"
    assert terminal["cleanup_failure_exception_type"] == (
        None if cleanup_verified else "RuntimeError"
    )
    assert terminal["claims"][
        "r05_synthetic_dev_wire_covert_qualification_verified"
    ] is False
    assert capture["capture_complete"] is False
    assert capture["capture_persistence_verified"] is capture_persistence_verified


def test_failure_v4_unproven_cleanup_cannot_claim_complete_post_drain_capture(
    tmp_path: Path,
) -> None:
    # EOF/reap can precede a later final-absence failure, but the unproven
    # cleanup schema has no positive reap projection for pure replay.  The
    # producer must therefore preserve the bytes without claiming complete.
    with pytest.raises(ValueError, match="cannot claim complete child pipe"):
        _materialize_launched_failure_v4(
            tmp_path,
            cleanup_verified=False,
            capture_persistence_verified=True,
            capture_complete=True,
        )


def test_failure_v4_truncated_at_cap_capture_is_retained_and_replayed(
    tmp_path: Path,
) -> None:
    attempt_directory, terminal_bytes, capture, _ = (
        _materialize_launched_failure_v4(
            tmp_path,
            cleanup_verified=False,
            capture_persistence_verified=True,
            capture_truncated=True,
        )
    )
    assert capture["capture_complete"] is False
    assert capture["capture_truncated_at_cap"] is True
    assert capture["capture_boundary"] == "INCOMPLETE_TRUNCATED_AT_STREAM_CAP"
    assert capture["stderr"]["final"]["size_bytes"] == 256 * 1_024
    assert qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
        attempt_directory,
        expected_terminal_bytes=terminal_bytes,
        prospective_terminal=True,
    ) == terminal_bytes
    (attempt_directory / "terminal.json").write_bytes(terminal_bytes)
    assert qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
        attempt_directory,
        expected_terminal_bytes=terminal_bytes,
    ) == terminal_bytes


@pytest.mark.parametrize("missing_directory", ("cas", "rfc8785-raw-bundle"))
def test_failure_v4_unverified_capture_requires_prelaunch_baseline_directories(
    tmp_path: Path,
    missing_directory: str,
) -> None:
    attempt_directory, terminal_bytes, _, _ = _materialize_launched_failure_v4(
        tmp_path,
        cleanup_verified=True,
        capture_persistence_verified=False,
    )
    terminal = qualification._decode_canonical_transport_object_v2(
        terminal_bytes,
        "missing launched baseline terminal",
        maximum_bytes=8 * 1_024 * 1_024,
    )
    (attempt_directory / "evidence" / missing_directory).rmdir()
    attempt_fd = qualification._open_real_directory_fd_v2(attempt_directory)
    try:
        terminal["retained_tree_before_terminal"] = (
            qualification._retained_attempt_tree_before_terminal_v2(attempt_fd)
        )
    finally:
        os.close(attempt_fd)
    spliced = qualification._canonical_inner_artifact_v2(terminal)
    expected_message = (
        "failure CAS retained snapshot drifted"
        if missing_directory == "cas"
        else "launched failure input set drifted"
    )
    with pytest.raises(ValueError, match=expected_message):
        qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
            attempt_directory,
            expected_terminal_bytes=spliced,
            prospective_terminal=True,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "raw-byte",
        "cas-byte",
        "missing-raw",
        "missing-cas",
        "path-splice",
        "hash-splice",
        "size-bool",
        "stream-swap",
    ),
)
def test_failure_v4_child_pipe_capture_rejects_retained_and_projection_tamper(
    tmp_path: Path,
    mutation: str,
) -> None:
    attempt_directory, terminal_bytes, capture, _ = (
        _materialize_launched_failure_v4(tmp_path)
    )
    terminal = qualification._decode_canonical_transport_object_v2(
        terminal_bytes,
        "failure /4 tamper terminal",
        maximum_bytes=8 * 1_024 * 1_024,
    )
    projection = json.loads(json.dumps(capture))
    evidence = attempt_directory / "evidence"
    bundle_digest = projection["bundle_cas"]["content_sha256"]
    if mutation == "raw-byte":
        target = evidence / "failure-child-stdout.bin"
        target.write_bytes(target.read_bytes() + b"tamper")
    elif mutation == "cas-byte":
        (evidence / "cas" / bundle_digest).write_bytes(b"tampered bundle")
    elif mutation == "missing-raw":
        (evidence / "failure-child-stderr.bin").unlink()
    elif mutation == "missing-cas":
        (evidence / "cas" / bundle_digest).unlink()
    elif mutation == "path-splice":
        projection["stdout"]["final"]["path"] = (
            "evidence/failure-child-stderr.bin"
        )
    elif mutation == "hash-splice":
        projection["stdout"]["final"]["sha256"] = projection["stderr"][
            "final"
        ]["sha256"]
    elif mutation == "size-bool":
        projection["stdout"]["final"]["size_bytes"] = True
    elif mutation == "stream-swap":
        projection["stdout"], projection["stderr"] = (
            projection["stderr"],
            projection["stdout"],
        )
    else:
        raise AssertionError("unhandled failure pipe tamper")
    if mutation in {"path-splice", "hash-splice", "size-bool", "stream-swap"}:
        core = {key: value for key, value in projection.items() if key != "capture_root"}
        projection["capture_root"] = qualification.stable_hash(
            core,
            prefix="phase2b_formal_wire_covert_failure_pipe_capture_v1_",
        )
    evidence_fd = qualification._open_real_directory_fd_v2(evidence)
    try:
        with pytest.raises((ValueError, FileNotFoundError, OSError)):
            qualification._verify_failure_child_pipe_capture_projection_v2(
                evidence_fd,
                projection,
                attempt_manifest_root=terminal["attempt_manifest_root"],
                deadline_ns=time.monotonic_ns() + 10_000_000_000,
            )
    finally:
        os.close(evidence_fd)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-partial",
        "unknown-partial",
        "partial-hash",
        "false-complete",
        "partial-order",
        "orphan-final",
        "non-prefix-final",
    ),
)
def test_failure_v4_persistence_unverified_capture_rejects_partial_splices(
    tmp_path: Path,
    mutation: str,
) -> None:
    attempt_directory, terminal_bytes, capture, _ = (
        _materialize_launched_failure_v4(
            tmp_path,
            cleanup_verified=True,
            capture_persistence_verified=False,
        )
    )
    terminal = qualification._decode_canonical_transport_object_v2(
        terminal_bytes,
        "unverified failure capture tamper terminal",
        maximum_bytes=8 * 1_024 * 1_024,
    )
    projection = json.loads(json.dumps(capture))
    evidence = attempt_directory / "evidence"
    if mutation == "missing-partial":
        (evidence / "failure-child-stdout-prefix.bin").unlink()
    elif mutation == "unknown-partial":
        (evidence / "failure-child-unknown.bin").write_bytes(b"unknown")
    elif mutation == "partial-hash":
        projection["retained_partial_artifacts"][0]["sha256"] = "f" * 64
    elif mutation == "false-complete":
        projection["capture_complete"] = True
    elif mutation == "partial-order":
        projection["retained_partial_artifacts"].reverse()
    elif mutation == "orphan-final":
        stdout_prefix = evidence / "failure-child-stdout-prefix.bin"
        stdout_final = evidence / "failure-child-stdout.bin"
        stdout_final.write_bytes(stdout_prefix.read_bytes() + b"tail")
        stdout_prefix.unlink()
        projection["retained_partial_artifacts"] = [
            qualification._failure_pipe_file_reference_v2(
                "evidence/failure-child-stdout.bin", stdout_final.read_bytes()
            ),
            projection["retained_partial_artifacts"][1],
        ]
    elif mutation == "non-prefix-final":
        stdout_final = evidence / "failure-child-stdout.bin"
        stdout_final.write_bytes(b"not-the-retained-stdout-prefix")
        projection["retained_partial_artifacts"] = [
            projection["retained_partial_artifacts"][0],
            qualification._failure_pipe_file_reference_v2(
                "evidence/failure-child-stdout.bin", stdout_final.read_bytes()
            ),
            projection["retained_partial_artifacts"][1],
        ]
    else:
        raise AssertionError("unhandled unverified capture tamper")
    if mutation in {
        "partial-hash",
        "false-complete",
        "partial-order",
        "orphan-final",
        "non-prefix-final",
    }:
        core = {key: value for key, value in projection.items() if key != "capture_root"}
        projection["capture_root"] = qualification.stable_hash(
            core,
            prefix="phase2b_formal_wire_covert_failure_pipe_capture_v1_",
        )
    evidence_fd = qualification._open_real_directory_fd_v2(evidence)
    try:
        with pytest.raises((ValueError, FileNotFoundError, OSError)):
            qualification._verify_failure_child_pipe_capture_projection_v2(
                evidence_fd,
                projection,
                attempt_manifest_root=terminal["attempt_manifest_root"],
                deadline_ns=time.monotonic_ns() + 10_000_000_000,
            )
    finally:
        os.close(evidence_fd)


@pytest.mark.parametrize(
    "retained_names",
    (
        (),
        ("failure-child-stdout-prefix.bin",),
        (
            "failure-child-stdout-prefix.bin",
            "failure-child-stderr-prefix.bin",
        ),
        (
            "failure-child-stdout-prefix.bin",
            "failure-child-stderr-prefix.bin",
            "failure-child-stdout.bin",
        ),
        (
            "failure-child-stdout-prefix.bin",
            "failure-child-stderr-prefix.bin",
            "failure-child-stdout.bin",
            "failure-child-stderr.bin",
        ),
    ),
)
def test_failure_pipe_persistence_unverified_accepts_exact_reachable_prefixes(
    tmp_path: Path, retained_names: tuple[str, ...]
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    payloads = {
        "failure-child-stdout-prefix.bin": b"stdout-prefix\x00",
        "failure-child-stderr-prefix.bin": b"stderr-prefix\xff",
        "failure-child-stdout.bin": b"stdout-prefix\x00stdout-tail",
        "failure-child-stderr.bin": b"stderr-prefix\xffstderr-tail",
    }
    for name in retained_names:
        (evidence / name).write_bytes(payloads[name])
    evidence_fd = qualification._open_real_directory_fd_v2(evidence)
    try:
        deadline = time.monotonic_ns() + 10_000_000_000
        projection = (
            qualification._failure_child_pipe_persistence_unverified_projection_v2(
                evidence_fd,
                persistence_error_type="OSError",
                capture_truncated=False,
                deadline_ns=deadline,
            )
        )
        expected_scan_order = (
            "failure-child-stdout-prefix.bin",
            "failure-child-stdout.bin",
            "failure-child-stderr-prefix.bin",
            "failure-child-stderr.bin",
        )
        assert [
            Path(row["path"]).name
            for row in projection["retained_partial_artifacts"]
        ] == [name for name in expected_scan_order if name in retained_names]
        qualification._verify_failure_child_pipe_capture_projection_v2(
            evidence_fd,
            projection,
            attempt_manifest_root=_attempt().attempt_manifest_root,
            deadline_ns=deadline,
        )
    finally:
        os.close(evidence_fd)


@pytest.mark.parametrize(
    "mutation",
    (
        "valid-unattributed-extra",
        "unprojected-extra",
        "missing-projected-object",
        "object-name-content-mismatch",
        "projection-order",
        "projection-size-bool",
    ),
)
def test_failure_v4_cas_projection_closes_exact_self_addressed_object_set(
    tmp_path: Path, mutation: str
) -> None:
    attempt_directory, terminal_bytes, _, _ = _materialize_launched_failure_v4(
        tmp_path
    )
    terminal = qualification._decode_canonical_transport_object_v2(
        terminal_bytes,
        "failure CAS exact-set terminal",
        maximum_bytes=8 * 1_024 * 1_024,
    )
    cas = attempt_directory / "evidence" / "cas"
    extra_payload = b"unattributed-partial-cas-object\x00"
    extra_digest = hashlib.sha256(extra_payload).hexdigest()
    (cas / extra_digest).write_bytes(extra_payload)

    def refresh_projection_and_tree() -> bytes:
        attempt_fd = qualification._open_real_directory_fd_v2(attempt_directory)
        try:
            projection = (
                qualification._derive_failure_v4_retained_projection_at_fd_v2(
                    attempt_fd,
                    managed_atomic_residues=terminal["managed_atomic_residues"],
                    deadline_ns=time.monotonic_ns() + 10_000_000_000,
                )
            )
            terminal["failure_cas_projection"] = projection[
                "failure_cas_projection"
            ]
            terminal["retained_tree_before_terminal"] = projection[
                "retained_tree_before_terminal"
            ]
        finally:
            os.close(attempt_fd)
        return qualification._canonical_inner_artifact_v2(terminal)

    if mutation == "valid-unattributed-extra":
        spliced = refresh_projection_and_tree()
        assert terminal["failure_cas_projection"][
            "partial_cas_roles_attributed"
        ] is False
        assert qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
            attempt_directory,
            expected_terminal_bytes=spliced,
            prospective_terminal=True,
        ) == spliced
        return
    if mutation == "unprojected-extra":
        spliced = terminal_bytes
    elif mutation == "missing-projected-object":
        spliced = refresh_projection_and_tree()
        (cas / extra_digest).unlink()
    elif mutation == "object-name-content-mismatch":
        (cas / extra_digest).write_bytes(b"different bytes")
        spliced = terminal_bytes
    else:
        refresh_projection_and_tree()
        rows = terminal["failure_cas_projection"][
            "objects_in_lexicographic_digest_order"
        ]
        if mutation == "projection-order":
            rows.reverse()
            terminal["failure_cas_projection"]["objects_root"] = (
                qualification.stable_hash(
                    rows,
                    prefix="phase2b_formal_wire_covert_failure_cas_v1_",
                )
            )
        elif mutation == "projection-size-bool":
            rows[0]["object_size_bytes"] = True
        else:
            raise AssertionError("unhandled failure CAS mutation")
        spliced = qualification._canonical_inner_artifact_v2(terminal)
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
            attempt_directory,
            expected_terminal_bytes=spliced,
            prospective_terminal=True,
        )


def test_failure_cas_projection_distinguishes_absent_empty_and_managed_temp(
    tmp_path: Path,
) -> None:
    attempt_directory = tmp_path / "attempt"
    attempt_directory.mkdir()
    attempt_fd = qualification._open_real_directory_fd_v2(attempt_directory)
    try:
        deadline = time.monotonic_ns() + 10_000_000_000
        absent = qualification._failure_cas_projection_at_attempt_fd_v2(
            attempt_fd, managed_atomic_residues=[], deadline_ns=deadline
        )
        assert absent["cas_directory_present"] is False
        assert absent["objects_in_lexicographic_digest_order"] == []
        cas = attempt_directory / "evidence" / "cas"
        cas.mkdir(parents=True)
        empty = qualification._failure_cas_projection_at_attempt_fd_v2(
            attempt_fd, managed_atomic_residues=[], deadline_ns=deadline
        )
        assert empty["cas_directory_present"] is True
        assert empty["objects_in_lexicographic_digest_order"] == []

        payload = b"managed-hardlink-cas-object"
        digest = hashlib.sha256(payload).hexdigest()
        target = cas / digest
        target.write_bytes(payload)
        temporary_name = f".{digest}.tmp-123-456"
        temporary = cas / temporary_name
        os.link(target, temporary)
        opened = target.stat(follow_symlinks=False)
        residue = {
            "device": str(opened.st_dev),
            "inode": str(opened.st_ino),
            "link_count": opened.st_nlink,
            "relative_parent": "evidence/cas",
            "same_inode": True,
            "size_bytes": len(payload),
            "state": "TARGET_AND_MANAGED_TEMP_HARDLINK",
            "target_name": digest,
            "target_sha256": digest,
            "temporary_name": temporary_name,
        }
        projected = qualification._failure_cas_projection_at_attempt_fd_v2(
            attempt_fd,
            managed_atomic_residues=[residue],
            deadline_ns=deadline,
        )
        assert projected["objects_in_lexicographic_digest_order"] == [
            {"content_sha256": digest, "object_size_bytes": len(payload)}
        ]
        assert qualification._verify_managed_atomic_residues_at_fd_v2(
            attempt_fd, [residue], deadline_ns=deadline
        ) == {"evidence/cas/" + temporary_name}
        qualification._retained_attempt_tree_before_terminal_v2(
            attempt_fd,
            managed_atomic_residues=[residue],
            deadline_ns=deadline,
        )
        with pytest.raises(ValueError, match="unknown non-object name"):
            qualification._failure_cas_projection_at_attempt_fd_v2(
                attempt_fd,
                managed_atomic_residues=[],
                deadline_ns=deadline,
            )
    finally:
        os.close(attempt_fd)


def test_managed_atomic_residue_target_language_matches_exact_publishers() -> None:
    digest = hashlib.sha256(b"intended complete payload").hexdigest()

    def row(
        relative_parent: str,
        target_name: str,
        *,
        state: str,
        target_sha256: str = digest,
    ) -> dict[str, object]:
        hardlink = state == "TARGET_AND_MANAGED_TEMP_HARDLINK"
        return {
            "device": "0",
            "inode": "1",
            "link_count": 2 if hardlink else 1,
            "relative_parent": relative_parent,
            "same_inode": hardlink,
            # A staged failure before the first byte is written is reachable.
            "size_bytes": 1 if hardlink else 0,
            "state": state,
            "target_name": target_name,
            "target_sha256": target_sha256,
            "temporary_name": f".{target_name}.tmp-1-2",
        }

    legal_targets = [
        ("", "reservation.json", digest),
        *[
            ("evidence", name, digest)
            for name in (
                "failure-child-stderr-prefix.bin",
                "failure-child-stderr.bin",
                "failure-child-stdout-prefix.bin",
                "failure-child-stdout.bin",
                "inner.json",
                "launch-intent.json",
                "mechanics.json",
                "outer-supervisor.json",
            )
        ],
        *[
            ("evidence/inputs", name, digest)
            for name in (
                "attempt.json",
                "profile.json",
                "runtime-environment.json",
            )
        ],
        ("evidence/cas", digest, digest),
        ("evidence/rfc8785-raw-bundle", "receipt.json", digest),
        ("evidence/rfc8785-raw-bundle", digest, digest),
        (
            "evidence/rfc8785-raw-bundle",
            "command-000-python-version.argv.json",
            digest,
        ),
        (
            "evidence/rfc8785-raw-bundle",
            "command-001-node-version.stdout.bin",
            digest,
        ),
        (
            "evidence/rfc8785-raw-bundle",
            "command-002-v8-version.stderr.bin",
            digest,
        ),
    ]
    for parent, target, target_sha256 in legal_targets:
        for state in (
            "STAGED_TEMP_ONLY",
            "TARGET_AND_MANAGED_TEMP_HARDLINK",
        ):
            item = row(
                parent,
                target,
                state=state,
                target_sha256=target_sha256,
            )
            assert qualification._managed_atomic_residue_paths_v2([item]) == {
                (
                    item["temporary_name"]
                    if not parent
                    else parent + "/" + str(item["temporary_name"])
                ): item
            }

    for parent, target in (
        ("", "terminal.json"),
        ("evidence", "arbitrary-known-parent.json"),
        ("evidence/source-tree", "qualification.py"),
        ("evidence/unknown", "attempt.json"),
        ("unknown-parent", "reservation.json"),
        (
            "evidence/rfc8785-raw-bundle",
            "command-000-Python-version.stdout.bin",
        ),
    ):
        with pytest.raises(ValueError, match="not publisher-reachable"):
            qualification._managed_atomic_residue_paths_v2(
                [row(parent, target, state="STAGED_TEMP_ONLY")]
            )

    mismatched_digest = hashlib.sha256(b"different intended payload").hexdigest()
    for state in (
        "STAGED_TEMP_ONLY",
        "TARGET_AND_MANAGED_TEMP_HARDLINK",
    ):
        with pytest.raises(ValueError, match="not publisher-reachable"):
            qualification._managed_atomic_residue_paths_v2(
                [
                    row(
                        "evidence/rfc8785-raw-bundle",
                        digest,
                        state=state,
                        target_sha256=mismatched_digest,
                    )
                ]
            )


def test_failure_v4_combined_projection_walks_once_and_hashes_each_inode_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_directory = tmp_path / "attempt"
    cas = attempt_directory / "evidence" / "cas"
    cas.mkdir(parents=True)
    (attempt_directory / "ordinary.bin").write_bytes(b"ordinary")

    payload = b"content-addressed-target"
    digest = hashlib.sha256(payload).hexdigest()
    target = cas / digest
    target.write_bytes(payload)
    hardlink_name = f".{digest}.tmp-101-202"
    hardlink = cas / hardlink_name
    os.link(target, hardlink)
    hardlink_stat = target.stat(follow_symlinks=False)

    intended_payload = b"complete-intended-CAS-payload"
    intended_digest = hashlib.sha256(intended_payload).hexdigest()
    staged_name = f".{intended_digest}.tmp-303-404"
    staged = cas / staged_name
    staged.write_bytes(intended_payload[:7])
    staged_stat = staged.stat(follow_symlinks=False)
    residues = [
        {
            "device": str(hardlink_stat.st_dev),
            "inode": str(hardlink_stat.st_ino),
            "link_count": hardlink_stat.st_nlink,
            "relative_parent": "evidence/cas",
            "same_inode": True,
            "size_bytes": hardlink_stat.st_size,
            "state": "TARGET_AND_MANAGED_TEMP_HARDLINK",
            "target_name": digest,
            "target_sha256": digest,
            "temporary_name": hardlink_name,
        },
        {
            "device": str(staged_stat.st_dev),
            "inode": str(staged_stat.st_ino),
            "link_count": staged_stat.st_nlink,
            "relative_parent": "evidence/cas",
            "same_inode": False,
            "size_bytes": staged_stat.st_size,
            "state": "STAGED_TEMP_ONLY",
            "target_name": intended_digest,
            "target_sha256": intended_digest,
            "temporary_name": staged_name,
        },
    ]
    real_fwalk = qualification.os.fwalk
    real_hash = qualification._hash_regular_at_fd_v2
    walks = 0
    hashed_inodes: list[tuple[int, int]] = []

    def counted_fwalk(*args: object, **kwargs: object):
        nonlocal walks
        walks += 1
        return real_fwalk(*args, **kwargs)

    def counted_hash(*args: object, **kwargs: object):
        digest_value, opened = real_hash(*args, **kwargs)
        hashed_inodes.append((opened.st_dev, opened.st_ino))
        return digest_value, opened

    monkeypatch.setattr(qualification.os, "fwalk", counted_fwalk)
    monkeypatch.setattr(qualification, "_hash_regular_at_fd_v2", counted_hash)
    attempt_fd = qualification._open_real_directory_fd_v2(attempt_directory)
    try:
        projection = qualification._derive_failure_v4_retained_projection_at_fd_v2(
            attempt_fd,
            managed_atomic_residues=residues,
            deadline_ns=time.monotonic_ns() + 10_000_000_000,
        )
        with pytest.raises(ValueError, match="not publisher-reachable"):
            qualification._derive_failure_v4_retained_projection_at_fd_v2(
                attempt_fd,
                managed_atomic_residues=[
                    {
                        **residues[1],
                        "target_name": "not-a-sha",
                        "temporary_name": ".not-a-sha.tmp-303-404",
                    }
                ],
                deadline_ns=time.monotonic_ns() + 10_000_000_000,
            )
        with pytest.raises(TimeoutError, match="future absolute deadline"):
            qualification._derive_failure_v4_retained_projection_at_fd_v2(
                attempt_fd,
                managed_atomic_residues=residues,
                deadline_ns=time.monotonic_ns() - 1,
            )
        with pytest.raises(ValueError, match="not publisher-reachable"):
            qualification._derive_failure_v4_retained_projection_at_fd_v2(
                attempt_fd,
                managed_atomic_residues=[
                    {
                        **residues[1],
                        "target_name": "f" * 64,
                        "temporary_name": f".{('f' * 64)}.tmp-303-404",
                    }
                ],
                deadline_ns=time.monotonic_ns() + 10_000_000_000,
            )
    finally:
        os.close(attempt_fd)
    assert walks == 1
    assert len(hashed_inodes) == 3
    assert len(set(hashed_inodes)) == 3
    assert projection["failure_cas_projection"][
        "objects_in_lexicographic_digest_order"
    ] == [{"content_sha256": digest, "object_size_bytes": len(payload)}]
    assert projection["retained_tree_before_terminal"]["regular_file_count"] == 4
    assert hashlib.sha256(staged.read_bytes()).hexdigest() != intended_digest


def test_failure_v4_materialization_and_prospective_replay_use_exactly_two_combined_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_combined = qualification._derive_failure_v4_retained_projection_at_fd_v2
    calls = 0

    def counted_combined(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return real_combined(*args, **kwargs)

    monkeypatch.setattr(
        qualification,
        "_derive_failure_v4_retained_projection_at_fd_v2",
        counted_combined,
    )
    monkeypatch.setattr(
        qualification,
        "_failure_cas_projection_at_attempt_fd_v2",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy CAS scan entered /failure/4")
        ),
    )
    monkeypatch.setattr(
        qualification,
        "_retained_attempt_tree_before_terminal_v2",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy tree scan entered /failure/4")
        ),
    )
    attempt_directory, terminal_bytes, _, _ = _materialize_launched_failure_v4(
        tmp_path
    )
    assert calls == 1
    assert qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
        attempt_directory,
        expected_terminal_bytes=terminal_bytes,
        prospective_terminal=True,
    ) == terminal_bytes
    assert calls == qualification._FINALIZATION_TREE_SCAN_PASSES_V2 == 2


def test_failure_v4_combined_projection_checks_deadline_for_each_directory_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_directory = tmp_path / "attempt"
    (attempt_directory / "first-empty-directory").mkdir(parents=True)

    def root_only_fwalk(
        *args: object, dir_fd: int, **kwargs: object
    ):
        yield ".", ["first-empty-directory"], [], dir_fd

    def forbidden_stat(*args: object, **kwargs: object):
        raise AssertionError("expired directory entry reached os.stat")

    ticks = iter((0, 0, 101))
    monkeypatch.setattr(qualification.os, "fwalk", root_only_fwalk)
    monkeypatch.setattr(qualification.os, "stat", forbidden_stat)
    monkeypatch.setattr(
        qualification.time,
        "monotonic_ns",
        lambda: next(ticks),
    )
    attempt_fd = qualification._open_real_directory_fd_v2(attempt_directory)
    try:
        with pytest.raises(TimeoutError, match="absolute deadline"):
            qualification._derive_failure_v4_retained_projection_at_fd_v2(
                attempt_fd,
                managed_atomic_residues=[],
                deadline_ns=100,
            )
    finally:
        os.close(attempt_fd)


def test_atomic_reconcile_known_parent_is_fd_relative_without_walk_or_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_directory = tmp_path / "attempt"
    cas = attempt_directory / "evidence" / "cas"
    cas.mkdir(parents=True)
    intended = hashlib.sha256(b"complete payload").hexdigest()
    temporary_name = f".{intended}.tmp-505-606"
    temporary = cas / temporary_name
    temporary.write_bytes(b"partial")
    opened = temporary.stat(follow_symlinks=False)
    attempt_fd = qualification._open_real_directory_fd_v2(attempt_directory)
    cas_fd = qualification._open_real_directory_fd_v2(cas)
    exc = qualification._AtomicPublicationUncertainV2(
        intended,
        "test staged CAS residue",
        parent_identity=(opened.st_dev, cas.stat().st_ino),
        target_identity=(opened.st_dev, opened.st_ino),
        target_sha256=intended,
        temporary_name=temporary_name,
        publication_state="STAGED_TEMP_ONLY",
    )
    real_remove = qualification.os.remove

    def persistent_remove(path: object, *args: object, **kwargs: object) -> None:
        if str(path) == temporary_name:
            raise OSError("persistent test residue")
        real_remove(path, *args, **kwargs)

    monkeypatch.setattr(qualification.os, "remove", persistent_remove)
    monkeypatch.setattr(
        qualification,
        "_hash_regular_at_fd_v2",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reconcile hashed staged bytes")
        ),
    )
    try:
        rows = qualification._reconcile_nonterminal_atomic_uncertainty_v2(
            exc,
            attempt_fd=attempt_fd,
            deadline_ns=time.monotonic_ns() + 10_000_000_000,
            known_parent_directories={"evidence/cas": cas_fd},
        )
    finally:
        os.close(cas_fd)
        os.close(attempt_fd)
    assert len(rows) == 1
    assert rows[0]["target_name"] == intended
    assert rows[0]["target_sha256"] == intended
    assert rows[0]["size_bytes"] == len(b"partial")


def test_failure_pipe_persistence_fallback_uses_remaining_terminal_reserve(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    evidence_fd = qualification._open_real_directory_fd_v2(evidence)
    try:
        terminal_deadline = time.monotonic_ns() + 10_000_000_000
        qualification._persist_failure_child_pipe_prefix_v2(
            evidence_fd,
            stdout=b"retained-prefix",
            stderr=b"\xff",
            deadline_ns=terminal_deadline,
        )
        expired_cleanup_deadline = time.monotonic_ns() - 1
        with pytest.raises(TimeoutError):
            qualification._failure_child_pipe_persistence_unverified_projection_v2(
                evidence_fd,
                persistence_error_type="TimeoutError",
                capture_truncated=False,
                deadline_ns=expired_cleanup_deadline,
            )
        projection = (
            qualification._failure_child_pipe_persistence_unverified_projection_v2(
                evidence_fd,
                persistence_error_type="TimeoutError",
                capture_truncated=False,
                deadline_ns=terminal_deadline,
            )
        )
        assert projection["capture_persistence_verified"] is False
        assert projection["capture_complete"] is False
        assert len(projection["retained_partial_artifacts"]) == 2
        qualification._verify_failure_child_pipe_capture_projection_v2(
            evidence_fd,
            projection,
            attempt_manifest_root=_attempt().attempt_manifest_root,
            deadline_ns=terminal_deadline,
        )
    finally:
        os.close(evidence_fd)


def test_failure_v4_verified_cleanup_rejects_incomplete_capture_splice(
    tmp_path: Path,
) -> None:
    attempt_directory, terminal_bytes, _, _ = _materialize_launched_failure_v4(
        tmp_path
    )
    terminal = qualification._decode_canonical_transport_object_v2(
        terminal_bytes,
        "failure /4 incomplete splice terminal",
        maximum_bytes=8 * 1_024 * 1_024,
    )
    capture = terminal["child_pipe_capture"]
    capture["capture_complete"] = False
    capture["capture_boundary"] = "INCOMPLETE_PREFIX_OR_PARTIAL_DRAIN_ONLY"
    capture_core = {
        key: value for key, value in capture.items() if key != "capture_root"
    }
    capture["capture_root"] = qualification.stable_hash(
        capture_core,
        prefix="phase2b_formal_wire_covert_failure_pipe_capture_v1_",
    )
    spliced = qualification._canonical_inner_artifact_v2(terminal)
    with pytest.raises(ValueError, match="complete child pipe capture"):
        qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
            attempt_directory,
            expected_terminal_bytes=spliced,
            prospective_terminal=True,
        )


def test_failure_v3_replay_remains_closed_and_v3_v4_field_splices_are_rejected(
    tmp_path: Path,
) -> None:
    attempt_directory, terminal_bytes, capture, _ = (
        _materialize_launched_failure_v4(tmp_path)
    )
    terminal_v4 = qualification._decode_canonical_transport_object_v2(
        terminal_bytes,
        "failure schema splice terminal",
        maximum_bytes=8 * 1_024 * 1_024,
    )
    missing_capture_v4 = dict(terminal_v4)
    missing_capture_v4.pop("child_pipe_capture")
    with pytest.raises(ValueError, match="not closed"):
        qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
            attempt_directory,
            expected_terminal_bytes=qualification._canonical_inner_artifact_v2(
                missing_capture_v4
            ),
            prospective_terminal=True,
        )

    evidence = attempt_directory / "evidence"
    for name in (
        "failure-child-stdout-prefix.bin",
        "failure-child-stderr-prefix.bin",
        "failure-child-stdout.bin",
        "failure-child-stderr.bin",
    ):
        (evidence / name).unlink()
    (evidence / "cas" / capture["bundle_cas"]["content_sha256"]).unlink()
    attempt_fd = qualification._open_real_directory_fd_v2(attempt_directory)
    try:
        legacy_tree = qualification._retained_attempt_tree_before_terminal_v2(
            attempt_fd
        )
    finally:
        os.close(attempt_fd)
    legacy = dict(terminal_v4)
    legacy["schema_version"] = qualification._OUTER_ATTEMPT_FAILURE_SCHEMA_V3
    legacy["failure_stage"] = legacy.pop("primary_failure_stage")
    legacy.pop("primary_failure_exception_type")
    legacy.pop("cleanup_failure_exception_type")
    legacy.pop("child_pipe_capture")
    legacy.pop("failure_cas_projection")
    legacy["retained_tree_before_terminal"] = legacy_tree
    legacy_bytes = qualification._canonical_inner_artifact_v2(legacy)
    with pytest.raises(ValueError, match="replay-only"):
        qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
            attempt_directory,
            expected_terminal_bytes=legacy_bytes,
            prospective_terminal=True,
        )
    with pytest.raises(ValueError, match="cannot be timing-finalized"):
        qualification._retime_verified_outer_terminal_v2(
            legacy_bytes,
            prospective_replay_completed_ns=time.monotonic_ns(),
            prospective_deadline_ns=time.monotonic_ns() + 10_000_000_000,
        )

    legacy_with_v4_field = dict(legacy)
    legacy_with_v4_field["child_pipe_capture"] = capture
    legacy_with_v4_bytes = qualification._canonical_inner_artifact_v2(
        legacy_with_v4_field
    )
    (attempt_directory / "terminal.json").write_bytes(legacy_with_v4_bytes)
    with pytest.raises(ValueError, match="not closed"):
        qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
            attempt_directory,
            expected_terminal_bytes=legacy_with_v4_bytes,
        )
    (attempt_directory / "terminal.json").write_bytes(legacy_bytes)
    assert qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
        attempt_directory,
        expected_terminal_bytes=legacy_bytes,
    ) == legacy_bytes


def test_child_pipe_capture_appends_chunked_non_utf8_tail_exactly_once() -> None:
    stdout_prefix = b"stdout-prefix\x00\xff"
    stderr_prefix = b"stderr-prefix\xfe\x80"
    stdout_tail_parts = (b"stdout-tail-a\x81", b"stdout-tail-b\n")
    stderr_tail_parts = (b"stderr-tail-a\x82", b"stderr-tail-b\r\n")
    child_code = "\n".join(
        (
            "import os, time",
            f"os.write(1, bytes.fromhex('{stdout_prefix.hex()}'))",
            f"os.write(2, bytes.fromhex('{stderr_prefix.hex()}'))",
            "time.sleep(0.05)",
            f"os.write(1, bytes.fromhex('{stdout_tail_parts[0].hex()}'))",
            f"os.write(2, bytes.fromhex('{stderr_tail_parts[0].hex()}'))",
            "time.sleep(0.05)",
            f"os.write(1, bytes.fromhex('{stdout_tail_parts[1].hex()}'))",
            f"os.write(2, bytes.fromhex('{stderr_tail_parts[1].hex()}'))",
        )
    )
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", "-B", "-c", child_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    def read_exact(descriptor: int, size: int) -> bytes:
        retained = bytearray()
        while len(retained) < size:
            block = os.read(descriptor, size - len(retained))
            if not block:
                raise AssertionError("child pipe reached EOF before its frozen prefix")
            retained.extend(block)
        return bytes(retained)

    assert process.stdout is not None and process.stderr is not None
    observed_stdout_prefix = read_exact(process.stdout.fileno(), len(stdout_prefix))
    observed_stderr_prefix = read_exact(process.stderr.fileno(), len(stderr_prefix))
    stdout_buffer = bytearray(observed_stdout_prefix)
    stderr_buffer = bytearray(observed_stderr_prefix)
    capture_state = {
        "capture_complete": False,
        "capture_truncated": False,
    }
    stdout_tail, stderr_tail, returncode = qualification._capture_process_pipes_v2(
        process,
        deadline_ns=time.monotonic_ns() + 5_000_000_000,
        stdout_cap=len(stdout_prefix) + sum(map(len, stdout_tail_parts)),
        stderr_cap=len(stderr_prefix) + sum(map(len, stderr_tail_parts)),
        wait_for_process_exit=True,
        stdout_buffer=stdout_buffer,
        stderr_buffer=stderr_buffer,
        capture_state=capture_state,
    )
    qualification._close_process_pipes_v2(
        process, name="chunked non-UTF8 capture test"
    )
    assert stdout_tail == b"".join(stdout_tail_parts)
    assert stderr_tail == b"".join(stderr_tail_parts)
    assert bytes(stdout_buffer) == stdout_prefix + stdout_tail
    assert bytes(stderr_buffer) == stderr_prefix + stderr_tail
    assert returncode == 0
    assert capture_state == {
        "capture_complete": True,
        "capture_truncated": False,
    }
    assert process.poll() == 0
    assert process.stdout.closed and process.stderr.closed


def test_child_pipe_capture_cap_failure_retains_bounded_prefix_and_marks_truncated() -> None:
    prefix = b"at-cap"
    extra = b"\xff"
    child_code = "\n".join(
        (
            "import os, time",
            f"os.write(1, bytes.fromhex('{prefix.hex()}'))",
            "time.sleep(0.05)",
            f"os.write(1, bytes.fromhex('{extra.hex()}'))",
            "time.sleep(5)",
        )
    )
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", "-B", "-c", child_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    observed_prefix = os.read(process.stdout.fileno(), len(prefix))
    assert observed_prefix == prefix
    stdout_buffer = bytearray(observed_prefix)
    stderr_buffer = bytearray()
    capture_state = {
        "capture_complete": False,
        "capture_truncated": False,
    }
    try:
        with pytest.raises(ValueError, match="stdout exceeded its cap"):
            qualification._capture_process_pipes_v2(
                process,
                deadline_ns=time.monotonic_ns() + 5_000_000_000,
                stdout_cap=len(prefix),
                stderr_cap=0,
                wait_for_process_exit=True,
                stdout_buffer=stdout_buffer,
                stderr_buffer=stderr_buffer,
                capture_state=capture_state,
            )
        assert bytes(stdout_buffer) == prefix
        assert bytes(stderr_buffer) == b""
        assert capture_state == {
            "capture_complete": False,
            "capture_truncated": True,
        }
    finally:
        qualification._finalize_outer_launcher_v2(
            process, deadline_ns=time.monotonic_ns() + 5_000_000_000
        )
    assert process.poll() is not None
    assert process.stdout.closed
    assert process.stderr is not None and process.stderr.closed


def test_checked_profile_is_exact_canonical_candidate_with_live_disabled() -> None:
    payload = qualification.phase2b_formal_wire_covert_qualification_profile_bytes_v2()
    assert payload == (ROOT / "config/phase2b_formal_wire_covert_qualification_v2.json").read_bytes()
    profile = qualification.load_phase2b_formal_wire_covert_qualification_profile_v2(payload)
    assert profile.profile_id == qualification.PROFILE_ID
    assert qualification.PROFILE_SCHEMA_VERSION == (
        "hegel-machine-phase2b-formal-wire-covert-qualification/6"
    )
    assert qualification.AUDITOR_SCHEMA_VERSION_V2 == (
        "hegel-machine-phase2b-formal-covert-auditor/5"
    )
    assert auditor.SCHEMA_VERSION == qualification.AUDITOR_SCHEMA_VERSION_V2
    assert qualification.AUDITOR_PROFILE_SCHEMA_VERSION_V2 == (
        "hegel-machine-phase2b-formal-covert-auditor-profile/5"
    )
    assert auditor.SEMANTICS_VERSION == (
        "hegel-machine-phase2b-formal-covert-statistics/11-"
        "counter-ordering-erratum"
    )
    assert auditor.SEMANTICS_ID == qualification.AUDITOR_SEMANTICS_ID_V2
    assert auditor.POLICY_BINDING_ID == qualification.AUDITOR_POLICY_BINDING_ID_V2
    assert profile.profile["auditor_binding"]["schema_version"] == (
        qualification.AUDITOR_SCHEMA_VERSION_V2
    )
    assert profile.profile["auditor_binding"]["profile_schema_version"] == (
        qualification.AUDITOR_PROFILE_SCHEMA_VERSION_V2
    )
    assert profile.profile["auditor_binding"][
        "receipt_issuance_runtime_evidence_required"
    ] is True
    assert profile.profile["auditor_binding"][
        "receipt_issuance_runtime_evidence_fields"
    ] == qualification._RECEIPT_ISSUANCE_RUNTIME_FIELDS_V2
    assert profile.profile["auditor_binding"][
        "maximum_static_phase_plan_rss_bytes"
    ] == auditor.MAXIMUM_STATIC_PHASE_PLAN_RSS_BYTES == 25_769_803_776
    assert profile.profile["auditor_binding"][
        "execution_scope_physical_memory_peak_cap_bytes"
    ] == auditor.EXECUTION_SCOPE_PHYSICAL_MEMORY_PEAK_CAP_BYTES == 12_884_901_888
    assert (
        auditor.MAXIMUM_STATIC_PHASE_PLAN_RSS_BYTES
        != auditor.EXECUTION_SCOPE_PHYSICAL_MEMORY_PEAK_CAP_BYTES
    )
    resource_fields = fields(auditor.ResourcePlanV1)
    assert len(qualification.FROZEN_STATIC_RESOURCE_PLAN_FIELDS_V2) == 45
    assert len(qualification.DYNAMIC_RESOURCE_PLAN_FIELDS_NOT_EXACT_COMPARED_V2) == 11
    assert len(resource_fields) == 56
    assert sum(item.init for item in resource_fields) == 55
    assert tuple(item.name for item in resource_fields if not item.init) == (
        "resource_plan_root",
    )
    assert len(fields(auditor.StatisticalRuntimeEvidenceV1)) == 27
    assert len(fields(auditor.ReceiptIssuanceRuntimeEvidenceV1)) == 18
    receipt_fields = tuple(item.name for item in fields(auditor.FormalCovertAuditReceiptV1))
    assert receipt_fields[17:22] == (
        "invariant_runtime_evidence",
        "statistics_runtime_evidence",
        "receipt_issuance_runtime_evidence",
        "results",
        "target_family_results",
    )
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
    assert live["requested_memory_max_equals_auditor_physical_cap"] is False
    assert live["post_exit_numeric_memory_peak_sample_claimed"] is False
    assert qualification._deep_thaw(
        live["post_output_physical_gate_contract"]
    ) == qualification._POST_OUTPUT_PHYSICAL_GATE_CONTRACT_V2
    assert live["outer_supervisor_result_schema"] == (
        "hegel-machine-phase2b-formal-wire-covert-outer-supervisor-result/3"
    )
    assert auditor.STATISTICS_WORKER_TIMEOUT_SECONDS == 3_600
    phase_deadline_seconds = 2 * (
        auditor.PARTITION_PREPARATION_TIMEOUT_SECONDS
        + auditor.INVARIANCE_WORKER_TIMEOUT_SECONDS
        + auditor.STATISTICS_WORKER_TIMEOUT_SECONDS
    )
    assert phase_deadline_seconds == 22_800
    normal_work_window_seconds = (
        live["hard_total_timeout_seconds"]
        - live["terminal_publication_reserve_seconds"]
        - live["cleanup_control_reserve_seconds"]
    )
    assert normal_work_window_seconds == 25_800
    assert phase_deadline_seconds < normal_work_window_seconds
    qualification_preparation_headroom_seconds = (
        normal_work_window_seconds - phase_deadline_seconds
    )
    assert qualification_preparation_headroom_seconds == 3_000
    assert (
        phase_deadline_seconds
        + qualification_preparation_headroom_seconds
        + live["terminal_publication_reserve_seconds"]
        + live["cleanup_control_reserve_seconds"]
        == live["hard_total_timeout_seconds"]
        == 28_800
    )
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
        "policy_binding_id": qualification.AUDITOR_POLICY_BINDING_ID_V2,
        "profile_schema_version": qualification.AUDITOR_PROFILE_SCHEMA_VERSION_V2,
        "schema_version": qualification.AUDITOR_SCHEMA_VERSION_V2,
        "semantics_id": qualification.AUDITOR_SEMANTICS_ID_V2,
        "source_sha256": qualification.AUDITOR_SOURCE_SHA256,
        "test_sha256": qualification.AUDITOR_TEST_SHA256,
    }
    profile_lock = qualification._EXPECTED_PROFILE["dependency_lock"]
    assert decoded["schema_version"] == (
        "hegel-machine-phase2b-formal-wire-covert-qualification-dependency-lock/3"
    )
    assert profile_lock["schema_version"] == decoded["schema_version"]
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
        str(qualification.OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2),
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
        "requested_cgroup_memory_max_bytes": (
            qualification.OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2
        ),
        "requested_cgroup_pids_max": 64,
        "rfc8785_prerequisite_receipt": (tmp_path / "rfc" / "receipt.json").resolve(),
        "runtime_environment_manifest": (tmp_path / "runtime.json").resolve(),
        "source_archive": (tmp_path / "source.tar").resolve(),
        "source_commit_sha": "a" * 40,
    }


def test_attempt_manifest_v3_requires_exact_operational_twenty_four_gib_memory_max() -> None:
    attempt = _attempt()
    exact_twelve_gib = 12 * 1_024**3
    exact_twenty_four_gib = 24 * 1_024**3
    assert (
        auditor.EXECUTION_SCOPE_PHYSICAL_MEMORY_PEAK_CAP_BYTES
        == exact_twelve_gib
    )
    assert auditor.MAXIMUM_ESTIMATED_RSS_BYTES == exact_twelve_gib
    assert auditor.MAXIMUM_STATIC_PHASE_PLAN_RSS_BYTES == exact_twenty_four_gib
    assert qualification.OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2 == exact_twenty_four_gib
    assert attempt.requested_cgroup_memory_max_bytes == exact_twenty_four_gib
    assert attempt.requested_cgroup_memory_max_bytes != (
        auditor.EXECUTION_SCOPE_PHYSICAL_MEMORY_PEAK_CAP_BYTES
    )
    assert attempt.attempt_manifest_root.startswith(
        "phase2b_formal_wire_covert_attempt_manifest_v3_"
    )
    for rejected in (
        exact_twelve_gib,
        exact_twenty_four_gib - 1,
        exact_twenty_four_gib + 1,
        True,
    ):
        with pytest.raises((TypeError, ValueError), match="cgroup limits"):
            replace(
                attempt,
                requested_cgroup_memory_max_bytes=rejected,
            )


@pytest.mark.parametrize(
    "rejected",
    (
        12 * 1_024**3,
        24 * 1_024**3 - 1,
        24 * 1_024**3 + 1,
        True,
    ),
)
def test_development_attempt_builder_rejects_nonexact_operational_memory_max_before_io(
    rejected: object,
) -> None:
    with pytest.raises(ValueError, match="cgroup limits"):
        qualification._build_development_attempt_manifest_bytes_v2(
            campaign_id="campaign-rejected-memory",
            attempt_id="attempt-rejected-memory",
            source_commit_sha="0" * 40,
            source_archive=Path("/never-read/source.tar"),
            runtime_environment_manifest=Path("/never-read/runtime.json"),
            rfc8785_prerequisite_receipt=Path("/never-read/rfc.json"),
            requested_cgroup_memory_max_bytes=rejected,  # type: ignore[arg-type]
            requested_cgroup_pids_max=64,
            project_root=Path("/never-read/project"),
        )


def test_retained_pre_decoupling_and_r3_profile_identities_are_rejected() -> None:
    retained_pre_decoupling_profile_id = (
        "phase2b_formal_wire_covert_qualification_profile_v2_"
        "3a055d4dea2c4d00c7ccbce465746ed5f70cc51977460ee027bd5807ab6e5d46"
    )
    pre_decoupling_profile = qualification._deep_thaw(
        qualification._EXPECTED_PROFILE
    )
    auditor_binding = pre_decoupling_profile["auditor_binding"]
    auditor_binding["maximum_static_phase_plan_rss_bytes"] = (
        auditor.EXECUTION_SCOPE_PHYSICAL_MEMORY_PEAK_CAP_BYTES
    )
    auditor_binding["semantics_id"] = (
        "phase2b_formal_covert_semantics_"
        "d50564375ce978d2ee585662adc6d9ac5c2335aa662214f508897f1076918321"
    )
    auditor_binding["policy_binding_id"] = (
        "phase2b_formal_covert_policy_binding_"
        "122bb24b8d8983a294a66e64c59fd74ac4017eaead1b7888141ff1679ac4272b"
    )
    dependency_lock = pre_decoupling_profile["dependency_lock"]
    dependency_lock["sha256"] = (
        "8c375133fb17d484d129464975ed4deb737d8f1e200d850f9ac75f4046bbfa07"
    )
    dependency_lock["runtime_import_closure_sha256"][
        "src/hegel_machine/phase2b_formal_covert_auditor_v1.py"
    ] = "f4ea0bb54fd7b4f5707a6eea22ac8b728279e4f2e5559899626e4bf477d87312"
    pinned = pre_decoupling_profile["wire_binding"]["pinned_artifact_sha256"]
    pinned.update(
        {
            "auditor_config": (
                "e98052fd9e545ef17c577ab294388793a44502c91b22a1d784b41b86ecf54e83"
            ),
            "auditor_documentation": (
                "07000f2f505731704392883062ac3d7ae1cd19411026e327f688ce441a7c6936"
            ),
            "auditor_source": (
                "f4ea0bb54fd7b4f5707a6eea22ac8b728279e4f2e5559899626e4bf477d87312"
            ),
            "auditor_test": (
                "0a383c06b4d3a7d048134e74f1465e85f2bc35add262bc065463e6b984d01b4f"
            ),
        }
    )
    pre_decoupling_encoded = qualification.canonicalize_rfc8785_v1(
        pre_decoupling_profile
    )
    assert retained_pre_decoupling_profile_id == (
        qualification.PROFILE_ID_PREFIX
        + hashlib.sha256(
            qualification.PROFILE_DOMAIN + pre_decoupling_encoded
        ).hexdigest()
    )
    assert qualification.PROFILE_ID != retained_pre_decoupling_profile_id
    pre_decoupling_transport = qualification.canonicalize_rfc8785_v1(
        {
            "qualification_profile": pre_decoupling_profile,
            "qualification_profile_id": retained_pre_decoupling_profile_id,
        }
    ) + b"\n"
    with pytest.raises(
        qualification.Phase2BFormalWireCovertQualificationErrorV2,
        match="qualification profile content drift",
    ):
        qualification.load_phase2b_formal_wire_covert_qualification_profile_v2(
            pre_decoupling_transport
        )

    retained_r3_profile_id = (
        "phase2b_formal_wire_covert_qualification_profile_v2_"
        "36c8913b900f2feb87e6ca5f6b3855818220498c646560216b55da78a4160ccf"
    )
    old_profile = json.loads(json.dumps(pre_decoupling_profile))
    live = old_profile["readiness_boundary"]["live_execution_contract"]
    live["requested_memory_max_equals_auditor_physical_cap"] = True
    live["post_output_physical_gate_contract"]["systemd_memory_max_bytes"] = (
        auditor.EXECUTION_SCOPE_PHYSICAL_MEMORY_PEAK_CAP_BYTES
    )
    old_encoded = qualification.canonicalize_rfc8785_v1(old_profile)
    assert retained_r3_profile_id == (
        qualification.PROFILE_ID_PREFIX
        + hashlib.sha256(qualification.PROFILE_DOMAIN + old_encoded).hexdigest()
    )
    assert qualification.PROFILE_ID != retained_r3_profile_id
    old_transport = qualification.canonicalize_rfc8785_v1(
        {
            "qualification_profile": old_profile,
            "qualification_profile_id": retained_r3_profile_id,
        }
    ) + b"\n"
    with pytest.raises(
        qualification.Phase2BFormalWireCovertQualificationErrorV2,
        match="qualification profile content drift",
    ):
        qualification.load_phase2b_formal_wire_covert_qualification_profile_v2(
            old_transport
        )


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
        requested_cgroup_memory_max_bytes=(
            qualification.OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2
        ),
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
            requested_cgroup_memory_max_bytes=(
                qualification.OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2
            ),
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
    complete["resource_plan_root"] = (
        "phase2b_formal_covert_statistics_resource_plan_" + "0" * 64
    )

    class Replay:
        def __init__(self, **values: object) -> None:
            assert set(values) == expected - {"resource_plan_root"}
            self.values = values

        def __post_init__(self) -> None:
            return None

    original_closed = qualification._closed_primitive_v2

    def closed(value: object) -> object:
        if type(value) is Replay:
            return {**value.values, "resource_plan_root": complete["resource_plan_root"]}
        return original_closed(value)

    monkeypatch.setattr(auditor, "ResourcePlanV1", Replay)
    monkeypatch.setattr(qualification, "_closed_primitive_v2", closed)
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
    with pytest.raises(ValueError, match="derived root"):
        qualification._validate_complete_resource_plan_projection_v2(
            {
                **complete,
                "resource_plan_root": (
                    "phase2b_formal_covert_statistics_resource_plan_" + "1" * 64
                ),
            },
            context="unit resource",
        )


@pytest.mark.parametrize(
    ("phase", "prefix"),
    (
        ("INVARIANCE", "phase2b_formal_covert_invariant_resource_plan_"),
        ("STATISTICS", "phase2b_formal_covert_statistics_resource_plan_"),
    ),
)
def test_phase_resource_preflight_projection_is_exact_and_closed(
    phase: str,
    prefix: str,
) -> None:
    resource_plan_root = prefix + "0" * 64
    parent_current = (
        auditor.EXECUTION_SCOPE_PHYSICAL_MEMORY_PEAK_CAP_BYTES // 2 + 1
    )
    projected = parent_current * 2
    preflight = auditor._PhaseResourcePreflightEvidenceV1(
        phase=phase,
        resource_plan_root=resource_plan_root,
        parent_current_rss_bytes_at_preflight=parent_current,
        static_parent_bytes_bound=90,
        fork_parent_rss_multiplier=2,
        additional_bytes_bound=0,
        projected_aggregate_rss_bytes_at_preflight=projected,
        maximum_estimated_rss_bytes=(
            auditor.MAXIMUM_STATIC_PHASE_PLAN_RSS_BYTES
        ),
        host_available_memory_safety_margin_bytes=(
            auditor.HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES
        ),
    )
    projection = qualification._closed_primitive_v2(preflight)
    assert qualification._replay_phase_resource_preflight_projection_v2(
        projection,
        phase=phase,
        resource_plan_root=resource_plan_root,
        context="unit phase preflight",
    ) == preflight
    assert projected > auditor.EXECUTION_SCOPE_PHYSICAL_MEMORY_PEAK_CAP_BYTES
    assert projected <= auditor.MAXIMUM_STATIC_PHASE_PLAN_RSS_BYTES
    with pytest.raises(ValueError, match="phase resource-preflight semantics drifted"):
        qualification._replay_phase_resource_preflight_projection_v2(
            {
                **projection,
                "maximum_estimated_rss_bytes": (
                    auditor.EXECUTION_SCOPE_PHYSICAL_MEMORY_PEAK_CAP_BYTES
                ),
            },
            phase=phase,
            resource_plan_root=resource_plan_root,
            context="unit static-physical cap splice",
        )
    for malformed in (
        {key: value for key, value in projection.items() if key != "phase"},
        {**projection, "unexpected": 0},
    ):
        with pytest.raises(ValueError, match="fields are not exact"):
            qualification._replay_phase_resource_preflight_projection_v2(
                malformed,
                phase=phase,
                resource_plan_root=resource_plan_root,
                context="unit phase preflight",
            )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("phase", "STATISTICS"),
        (
            "resource_plan_root",
            "phase2b_formal_covert_invariant_resource_plan_" + "1" * 64,
        ),
        ("parent_current_rss_bytes_at_preflight", 0),
        ("static_parent_bytes_bound", 0),
        ("fork_parent_rss_multiplier", 3),
        ("additional_bytes_bound", 11),
        ("projected_aggregate_rss_bytes_at_preflight", 211),
        (
            "maximum_estimated_rss_bytes",
            auditor.MAXIMUM_STATIC_PHASE_PLAN_RSS_BYTES + 1,
        ),
        (
            "host_available_memory_safety_margin_bytes",
            auditor.HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES + 1,
        ),
    ),
)
def test_invariant_phase_resource_preflight_tamper_matrix_fails_closed(
    field_name: str,
    replacement: object,
) -> None:
    resource_plan_root = (
        "phase2b_formal_covert_invariant_resource_plan_" + "0" * 64
    )
    projection = {
        "phase": "INVARIANCE",
        "resource_plan_root": resource_plan_root,
        "parent_current_rss_bytes_at_preflight": 100,
        "static_parent_bytes_bound": 90,
        "fork_parent_rss_multiplier": 2,
        "additional_bytes_bound": 10,
        "projected_aggregate_rss_bytes_at_preflight": 210,
        "maximum_estimated_rss_bytes": (
            auditor.MAXIMUM_STATIC_PHASE_PLAN_RSS_BYTES
        ),
        "host_available_memory_safety_margin_bytes": (
            auditor.HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES
        ),
    }
    projection[field_name] = replacement
    with pytest.raises(ValueError):
        qualification._replay_phase_resource_preflight_projection_v2(
            projection,
            phase="INVARIANCE",
            resource_plan_root=resource_plan_root,
            context="unit invariant phase preflight",
        )


def test_complete_two_phase_runtime_reconstruction_crossbinds_composite_plan() -> None:
    rows = 2
    columns = (auditor._FeatureColumn(("unit",), ("a", "a")),)
    deadline_ns = time.monotonic_ns() + 5_000_000_000
    partition_projection = auditor._partition_allocation_projection(
        rows,
        columns,
        deadline_ns,
    )
    partitions = auditor._materialize_partitions_bounded(
        columns,
        partition_projection,
        deadline_ns,
    )
    envelope_and_decode_bytes_bound = rows * auditor.ENVELOPE_BYTES * 2
    invariant_plan = auditor._invariance_resource_plan(
        rows,
        envelope_and_decode_bytes_bound=envelope_and_decode_bytes_bound,
        adapter_retained_graph_bytes_bound=1,
        variant_suite_resident_bytes_per_row_bound=1,
        variant_suite_resident_fixed_bytes_bound=1,
    )
    plan = auditor._resource_plan(
        rows,
        partitions,
        (2,) * len(auditor.TARGETS),
        preextraction_feature_matrix_bytes_bound=rows * 72,
        feature_matrix_bytes_bound=rows * 72,
        envelope_and_decode_bytes_bound=envelope_and_decode_bytes_bound,
        adapter_retained_graph_bytes_bound=1,
        partition_projection=partition_projection,
        partition_preparation_deadline_ns=deadline_ns,
        invariance_plan=invariant_plan,
    )
    invariant_static = (
        plan.envelope_and_decode_bytes_bound
        + plan.adapter_retained_graph_bytes_bound
        + 64 * 1_048_576
    )
    invariant_additional = (
        plan.invariance_variant_suite_resident_bytes_bound
        + plan.invariance_recognizer_live_output_bytes_bound
        + plan.invariance_worker_artifact_bytes_bound
    )
    invariant_preflight = auditor._PhaseResourcePreflightEvidenceV1(
        phase="INVARIANCE",
        resource_plan_root=plan.invariance_resource_plan_root,
        parent_current_rss_bytes_at_preflight=(
            plan.invariance_parent_current_rss_bytes_at_plan
        ),
        static_parent_bytes_bound=invariant_static,
        fork_parent_rss_multiplier=2,
        additional_bytes_bound=invariant_additional,
        projected_aggregate_rss_bytes_at_preflight=(
            max(
                plan.invariance_parent_current_rss_bytes_at_plan,
                invariant_static,
            )
            * 2
            + invariant_additional
        ),
        maximum_estimated_rss_bytes=plan.maximum_invariance_estimated_rss_bytes,
        host_available_memory_safety_margin_bytes=(
            plan.host_available_memory_safety_margin_bytes
        ),
    )
    invariant_available = (
        invariant_preflight.projected_aggregate_rss_bytes_at_preflight
        + invariant_preflight.host_available_memory_safety_margin_bytes
    )
    parent_rss = invariant_preflight.parent_current_rss_bytes_at_preflight
    cgroup_snapshot = auditor.UnifiedCgroupMemorySnapshotV1(
        unified_membership="/test.slice/qv2.scope",
        memory_scope_root=qualification.stable_hash(
            {
                "leaf_device": 11,
                "leaf_inode": 22,
                "unified_membership": "/test.slice/qv2.scope",
            },
            prefix="phase2b_formal_covert_unified_cgroup_memory_scope_v1_",
        ),
        leaf_device=11,
        leaf_inode=22,
        memory_current_bytes=4_096,
        memory_peak_bytes=8_192,
        memory_max_bytes=24 * 1_024 * 1_024 * 1_024,
    )
    invariant_runtime = auditor.InvariantRuntimeEvidenceV1(
        invariance_resource_plan_root=plan.invariance_resource_plan_root,
        resource_preflight=invariant_preflight,
        system_available_memory_bytes_at_preflight=invariant_available,
        cgroup_remaining_memory_bytes_at_preflight=None,
        host_available_memory_bytes_at_preflight=invariant_available,
        parent_current_rss_bytes_at_start=parent_rss,
        parent_peak_current_rss_bytes_observed=parent_rss,
        parent_current_rss_bytes_at_completion=parent_rss,
        parent_historical_peak_rss_bytes=parent_rss,
        worker_peak_rss_bytes=0,
        worker_result_artifact_bytes=1,
        worker_result_artifact_sha256="sha256:" + "0" * 64,
        diagnostic_parent_plus_worker_historical_peak_sum_bytes=parent_rss,
        unified_cgroup_membership=cgroup_snapshot.unified_membership,
        unified_cgroup_memory_scope_root=cgroup_snapshot.memory_scope_root,
        unified_cgroup_leaf_device=cgroup_snapshot.leaf_device,
        unified_cgroup_leaf_inode=cgroup_snapshot.leaf_inode,
        physical_memory_current_bytes_at_completion=(
            cgroup_snapshot.memory_current_bytes
        ),
        physical_memory_peak_bytes_at_completion=cgroup_snapshot.memory_peak_bytes,
        physical_memory_max_bytes=cgroup_snapshot.memory_max_bytes,
        physical_memory_cap_bytes=(
            auditor.EXECUTION_SCOPE_PHYSICAL_MEMORY_PEAK_CAP_BYTES
        ),
        invariance_wall_time_nanoseconds=1,
        worker_timeout_seconds=auditor.INVARIANCE_WORKER_TIMEOUT_SECONDS,
        invariant_run_count=81,
        exact_invariant_comparisons=rows * 80 * 5,
        fork_input_snapshot_root=(
            "phase2b_formal_covert_invariance_input_snapshot_" + "0" * 64
        ),
        adapter_execution_pin_root=(
            "phase2b_formal_covert_adapter_execution_pins_" + "0" * 64
        ),
        variant_suite_root="phase2b_formal_variant_suite_" + "0" * 64,
        consumer_decode_count=rows * 80,
        variant_suite_unique_envelope_object_count=0,
        variant_suite_resident_object_graph_bytes_observed=0,
        recognizer_output_root=(
            "phase2b_formal_covert_recognizer_output_" + "0" * 64
        ),
        execution_containment=auditor.INVARIANCE_EXECUTION_CONTAINMENT,
        parent_side_adapter_tcb=auditor.PARENT_SIDE_ADAPTER_TCB,
        parent_side_adapter_sandboxed=False,
        process_group_empty_at_exit=True,
        child_address_space_limit_bytes=1,
        within_frozen_physical_peak_cap=True,
    )
    statistics_static = (
        plan.permutation_label_mask_bytes_bound
        + plan.partition_matrix_bytes_bound
        + plan.adapter_retained_graph_bytes_bound
        + 64 * 1_048_576
    )
    statistics_additional = (
        plan.metric_cache_bytes_bound * plan.maximum_parallel_workers
        + plan.permutation_metric_tuple_bytes_bound
        + plan.target_max_vector_bytes_bound
        + plan.worker_result_artifact_bytes_bound
        + plan.parent_result_working_bytes_bound
    )
    statistics_multiplier = 1 + plan.maximum_parallel_workers
    statistics_preflight = auditor._PhaseResourcePreflightEvidenceV1(
        phase="STATISTICS",
        resource_plan_root=plan.resource_plan_root,
        parent_current_rss_bytes_at_preflight=plan.parent_current_rss_bytes_at_plan,
        static_parent_bytes_bound=statistics_static,
        fork_parent_rss_multiplier=statistics_multiplier,
        additional_bytes_bound=statistics_additional,
        projected_aggregate_rss_bytes_at_preflight=(
            max(plan.parent_current_rss_bytes_at_plan, statistics_static)
            * statistics_multiplier
            + statistics_additional
        ),
        maximum_estimated_rss_bytes=plan.maximum_estimated_rss_bytes,
        host_available_memory_safety_margin_bytes=(
            plan.host_available_memory_safety_margin_bytes
        ),
    )
    statistics_available = (
        statistics_preflight.projected_aggregate_rss_bytes_at_preflight
        + statistics_preflight.host_available_memory_safety_margin_bytes
    )
    statistics_runtime = auditor.StatisticalRuntimeEvidenceV1(
        resource_plan_root=plan.resource_plan_root,
        resource_preflight=statistics_preflight,
        system_available_memory_bytes_at_preflight=statistics_available,
        cgroup_remaining_memory_bytes_at_preflight=None,
        host_available_memory_bytes_at_preflight=statistics_available,
        parent_historical_peak_rss_bytes_before_receipt_finalization=1,
        worker_peak_rss_bytes=(1,),
        worker_result_artifact_bytes=(1,),
        worker_result_artifact_sha256=("sha256:" + "0" * 64,),
        parent_result_object_graph_bytes_observed=0,
        diagnostic_parent_plus_worker_historical_peak_sum_bytes=2,
        unified_cgroup_membership=cgroup_snapshot.unified_membership,
        unified_cgroup_memory_scope_root=cgroup_snapshot.memory_scope_root,
        unified_cgroup_leaf_device=cgroup_snapshot.leaf_device,
        unified_cgroup_leaf_inode=cgroup_snapshot.leaf_inode,
        physical_memory_current_bytes_before_receipt_finalization=(
            cgroup_snapshot.memory_current_bytes
        ),
        physical_memory_peak_bytes_before_receipt_finalization=(
            cgroup_snapshot.memory_peak_bytes
        ),
        physical_memory_max_bytes=cgroup_snapshot.memory_max_bytes,
        physical_memory_cap_bytes=(
            auditor.EXECUTION_SCOPE_PHYSICAL_MEMORY_PEAK_CAP_BYTES
        ),
        statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds=1,
        worker_timeout_seconds=auditor.STATISTICS_WORKER_TIMEOUT_SECONDS,
        parent_postprocess_deadline_schema=auditor.PARENT_POSTPROCESS_DEADLINE_SCHEMA,
        parent_postprocess_deadline_preemptive=False,
        external_supervisor_required=True,
        receipt_finalization_complete_pending_validation_count=1,
        receipt_finalization_physical_peak_within_cap=True,
        within_frozen_physical_peak_cap=True,
    )
    issuance_runtime = auditor.ReceiptIssuanceRuntimeEvidenceV1(
        absolute_deadline_kind="STATISTICS",
        absolute_deadline_monotonic_nanoseconds=1,
        absolute_deadline_timeout_seconds=auditor.STATISTICS_WORKER_TIMEOUT_SECONDS,
        diagnostic_worker_historical_peak_rss_bytes_sum=1,
        diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization=1,
        diagnostic_parent_plus_worker_historical_peak_sum_bytes=2,
        unified_cgroup_membership=cgroup_snapshot.unified_membership,
        unified_cgroup_memory_scope_root=cgroup_snapshot.memory_scope_root,
        unified_cgroup_leaf_device=cgroup_snapshot.leaf_device,
        unified_cgroup_leaf_inode=cgroup_snapshot.leaf_inode,
        physical_memory_current_bytes_before_receipt_finalization=(
            cgroup_snapshot.memory_current_bytes
        ),
        physical_memory_peak_bytes_before_receipt_finalization=(
            cgroup_snapshot.memory_peak_bytes
        ),
        physical_memory_max_bytes=cgroup_snapshot.memory_max_bytes,
        physical_memory_cap_bytes=(
            auditor.EXECUTION_SCOPE_PHYSICAL_MEMORY_PEAK_CAP_BYTES
        ),
        receipt_finalization_complete_pending_validation_count=1,
        receipt_finalization_locked_deadline_check_count=1,
        receipt_finalization_locked_deadline_within_absolute_limit=True,
        receipt_finalization_locked_physical_peak_within_cap=True,
    )
    plan_projection = qualification._closed_primitive_v2(plan)
    invariant_projection = qualification._closed_primitive_v2(invariant_runtime)
    statistics_projection = qualification._closed_primitive_v2(statistics_runtime)
    issuance_projection = qualification._closed_primitive_v2(issuance_runtime)
    assert invariant_runtime.resource_preflight.maximum_estimated_rss_bytes == (
        auditor.MAXIMUM_STATIC_PHASE_PLAN_RSS_BYTES
    )
    assert statistics_runtime.resource_preflight.maximum_estimated_rss_bytes == (
        auditor.MAXIMUM_STATIC_PHASE_PLAN_RSS_BYTES
    )
    assert statistics_runtime.physical_memory_cap_bytes == (
        auditor.EXECUTION_SCOPE_PHYSICAL_MEMORY_PEAK_CAP_BYTES
    )
    for runtime in (invariant_runtime, statistics_runtime, issuance_runtime):
        with pytest.raises(ValueError):
            replace(
                runtime,
                physical_memory_cap_bytes=(
                    auditor.MAXIMUM_STATIC_PHASE_PLAN_RSS_BYTES
                ),
            )
    with pytest.raises(ValueError, match="frozen scalar drift"):
        replace(
            statistics_runtime.resource_preflight,
            maximum_estimated_rss_bytes=(
                auditor.EXECUTION_SCOPE_PHYSICAL_MEMORY_PEAK_CAP_BYTES
            ),
        )
    assert qualification._validate_complete_invariant_runtime_projection_v2(
        invariant_projection,
        plan_projection,
        context="unit complete runtime",
    ) is invariant_projection
    assert qualification._validate_complete_statistics_runtime_projection_v2(
        statistics_projection,
        plan_projection,
        context="unit complete runtime",
    ) is statistics_projection
    assert qualification._validate_complete_receipt_issuance_runtime_projection_v2(
        issuance_projection,
        statistics_projection,
        context="unit complete runtime",
    ) is issuance_projection
    for projection, validator, plan_or_statistics in (
        (
            invariant_projection,
            qualification._validate_complete_invariant_runtime_projection_v2,
            plan_projection,
        ),
        (
            statistics_projection,
            qualification._validate_complete_statistics_runtime_projection_v2,
            plan_projection,
        ),
        (
            issuance_projection,
            qualification._validate_complete_receipt_issuance_runtime_projection_v2,
            statistics_projection,
        ),
    ):
        static_cap_splice = json.loads(json.dumps(projection))
        static_cap_splice["physical_memory_cap_bytes"] = (
            auditor.MAXIMUM_STATIC_PHASE_PLAN_RSS_BYTES
        )
        with pytest.raises(ValueError):
            validator(
                static_cap_splice,
                plan_or_statistics,
                context="unit static-cap physical replay splice",
            )
    for field_name, replacement in (
        ("diagnostic_worker_historical_peak_rss_bytes_sum", 2),
        (
            "diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization",
            2,
        ),
        ("unified_cgroup_membership", "/test.slice/drift.scope"),
        ("physical_memory_peak_bytes_before_receipt_finalization", 8_193),
        ("receipt_finalization_complete_pending_validation_count", 0),
        ("receipt_finalization_locked_deadline_check_count", 0),
        ("receipt_finalization_locked_deadline_within_absolute_limit", False),
        ("receipt_finalization_locked_physical_peak_within_cap", False),
    ):
        tampered = dict(issuance_projection)
        tampered[field_name] = replacement
        with pytest.raises(ValueError):
            qualification._validate_complete_receipt_issuance_runtime_projection_v2(
                tampered,
                statistics_projection,
                context="unit complete runtime",
            )
    insufficient_issuance = auditor.ReceiptIssuanceRuntimeEvidenceV1(
        absolute_deadline_kind="PARTITION_PREPARATION",
        absolute_deadline_monotonic_nanoseconds=1,
        absolute_deadline_timeout_seconds=(
            auditor.PARTITION_PREPARATION_TIMEOUT_SECONDS
        ),
        diagnostic_worker_historical_peak_rss_bytes_sum=0,
        diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization=1,
        diagnostic_parent_plus_worker_historical_peak_sum_bytes=1,
        unified_cgroup_membership=cgroup_snapshot.unified_membership,
        unified_cgroup_memory_scope_root=cgroup_snapshot.memory_scope_root,
        unified_cgroup_leaf_device=cgroup_snapshot.leaf_device,
        unified_cgroup_leaf_inode=cgroup_snapshot.leaf_inode,
        physical_memory_current_bytes_before_receipt_finalization=(
            cgroup_snapshot.memory_current_bytes
        ),
        physical_memory_peak_bytes_before_receipt_finalization=(
            cgroup_snapshot.memory_peak_bytes
        ),
        physical_memory_max_bytes=cgroup_snapshot.memory_max_bytes,
        physical_memory_cap_bytes=(
            auditor.EXECUTION_SCOPE_PHYSICAL_MEMORY_PEAK_CAP_BYTES
        ),
        receipt_finalization_complete_pending_validation_count=1,
        receipt_finalization_locked_deadline_check_count=1,
        receipt_finalization_locked_deadline_within_absolute_limit=True,
        receipt_finalization_locked_physical_peak_within_cap=True,
    )
    insufficient_projection = qualification._closed_primitive_v2(
        insufficient_issuance
    )
    assert qualification._validate_complete_receipt_issuance_runtime_projection_v2(
        insufficient_projection,
        None,
        context="unit insufficient runtime",
    ) is insufficient_projection
    for projection, validator, field_name, prefix in (
        (
            invariant_projection,
            qualification._validate_complete_invariant_runtime_projection_v2,
            "invariance_resource_plan_root",
            "phase2b_formal_covert_invariant_resource_plan_",
        ),
        (
            statistics_projection,
            qualification._validate_complete_statistics_runtime_projection_v2,
            "resource_plan_root",
            "phase2b_formal_covert_statistics_resource_plan_",
        ),
    ):
        tampered = json.loads(json.dumps(projection))
        tampered[field_name] = prefix + "f" * 64
        with pytest.raises(ValueError):
            validator(
                tampered,
                plan_projection,
                context="unit complete runtime",
            )
        tampered = json.loads(json.dumps(projection))
        tampered["resource_preflight"]["resource_plan_root"] = prefix + "f" * 64
        with pytest.raises(ValueError):
            validator(
                tampered,
                plan_projection,
                context="unit complete runtime",
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


def test_capacity_bounded_v4_coordinate_and_sampling_shapes_complete_exact_core() -> None:
    coordinate = qualification._core_fixture_profile_v2(
        exact_core.CoreTransformOperationV1.COORDINATE_AFFINE
    )
    coordinate_certificate = coordinate.contract.certificate
    assert type(coordinate_certificate) is exact_core.CoreCoordinateAffineCertificateV1
    assert len(coordinate.source_components) == 1
    assert len(coordinate.contract.output_metadata) == 1
    assert coordinate_certificate.dimension == 1
    assert coordinate_certificate.source_frame_binding == qualification._FRAME_A
    assert coordinate_certificate.target_frame_binding == qualification._FRAME_B
    assert coordinate.source_components[0].metadata.spatial_support == (
        exact_core.CoreSpatialSupportV1(
            qualification._FRAME_A,
            (Fraction(0),),
            (Fraction(1),),
        )
    )
    assert coordinate.contract.output_metadata[0].spatial_support == (
        exact_core.CoreSpatialSupportV1(
            qualification._FRAME_B,
            (Fraction(1),),
            (Fraction(3),),
        )
    )
    forward = coordinate.contract.kernel_rows[0]
    inverse = coordinate_certificate.inverse_rows[0]
    assert (
        tuple(term.coefficient for term in forward.terms),
        forward.offset,
        tuple(term.coefficient for term in inverse.terms),
        inverse.offset,
    ) == (
        (Fraction(2),),
        Fraction(1),
        (Fraction(1, 2),),
        Fraction(-1, 2),
    )
    coordinate_result = exact_core.run_exact_transform_core_v1(coordinate)
    assert coordinate_result.disposition is exact_core.CoreTransformDispositionV1.COMPLETE
    assert coordinate_result.output_components[0].numeric_value == Fraction(2)

    sampling = qualification._core_fixture_profile_v2(
        exact_core.CoreTransformOperationV1.SAMPLING_RESOLUTION
    )
    sampling_certificate = sampling.contract.certificate
    assert type(sampling_certificate) is exact_core.CoreSamplingResolutionCertificateV1
    assert len(sampling.source_components) == 2
    assert len(sampling.contract.output_metadata) == 1
    assert len(sampling_certificate.selected_inputs) == 1
    assert len(sampling_certificate.discarded_inputs) == 1
    assert set(sampling_certificate.selected_inputs).isdisjoint(
        sampling_certificate.discarded_inputs
    )
    assert set(sampling_certificate.selected_inputs).union(
        sampling_certificate.discarded_inputs
    ) == {item.ref for item in sampling.source_components}
    assert sampling_certificate.grid_points == (
        exact_core.CoreSamplingGridPointV1(
            sampling_certificate.selected_inputs[0],
            (Fraction(0),),
        ),
    )
    sampling_result = exact_core.run_exact_transform_core_v1(sampling)
    assert sampling_result.disposition is exact_core.CoreTransformDispositionV1.COMPLETE
    assert sampling_result.output_components[0].numeric_value == Fraction(1)


@pytest.mark.parametrize("mode", qualification.LIVE_MODE_ORDER)
def test_capacity_bounded_v4_n8_projection_is_exact_and_below_auditor_cap(
    mode: str,
) -> None:
    execution = _fixture(mode)
    decoded = tuple(
        execution.adapter.decode_envelope(envelope)
        for envelope in execution.adapter.envelopes
    )
    projection = auditor._feature_cardinality_projection(decoded)
    assert qualification._validate_fixture_feature_capacity_projection_v2(
        mode, projection
    ) is projection
    expected = qualification._FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"][mode]
    assert projection.projected_concrete_feature_count < auditor.MAXIMUM_FEATURES
    assert projection.projected_concrete_feature_count == expected["projections"][0][
        "projected_concrete_feature_count"
    ]
    assert projection.projection_root == expected["projections"][0][
        "projection_root"
    ]


@pytest.mark.parametrize("mode", qualification.LIVE_MODE_ORDER)
def test_capacity_bounded_v4_owner_union_is_execution_derived_and_mutation_closed(
    mode: str,
) -> None:
    execution = _fixture(mode)
    decoded = tuple(
        execution.adapter.decode_envelope(envelope)
        for envelope in execution.adapter.envelopes
    )
    independently_decoded_owner_paths = {
        owner.path for row in decoded for owner in row.channel_only_leaf_owners
    }
    expected_contract = qualification._FIXTURE_FEATURE_CAPACITY_CONTRACT_V2[
        "modes"
    ][mode]
    expected = expected_contract["channel_only_owned_union_path_count"]
    independently_derived_owner_projection = (
        qualification._channel_only_owner_union_projection_v2(
            owner for row in decoded for owner in row.channel_only_leaf_owners
        )
    )
    assert execution.exact_owned_channel_only_leaf_paths == (
        len(independently_decoded_owner_paths)
    )
    assert execution.exact_owned_channel_only_leaf_paths == expected
    assert independently_derived_owner_projection == (
        expected,
        expected_contract["channel_only_owned_union_root"],
    )
    assert execution.channel_only_owner_union_root == (
        expected_contract["channel_only_owned_union_root"]
    )
    assert qualification._validate_fixture_channel_only_owner_union_v2(
        mode,
        execution.row_count,
        execution.exact_owned_channel_only_leaf_paths,
        execution.channel_only_owner_union_root,
    ) == independently_derived_owner_projection
    independent_feature_projection = auditor._feature_cardinality_projection(decoded)
    assert execution.feature_capacity_projection == independent_feature_projection
    receipt_projection = (
        qualification._fixture_capacity_receipt_projection_from_execution_v2(
            execution
        )
    )
    assert receipt_projection == {
        "audited_rows": execution.row_count,
        "channel_leaf_count": independent_feature_projection.channel_union_path_count,
        "concrete_feature_count": (
            independent_feature_projection.projected_concrete_feature_count
        ),
        "preextraction_feature_matrix_bytes_bound": (
            independent_feature_projection.projected_feature_matrix_bytes_bound
        ),
        "raw_semantic_feature_count": (
            independent_feature_projection.projected_raw_semantic_feature_count
        ),
        "uuid_leaf_count": independent_feature_projection.uuid_union_path_count,
    }
    with pytest.raises(ValueError, match="owner union count drift"):
        replace(
            execution,
            exact_owned_channel_only_leaf_paths=(
                execution.exact_owned_channel_only_leaf_paths + 1
            ),
        )
    with pytest.raises(ValueError, match="owner union root drift"):
        replace(
            execution,
            channel_only_owner_union_root=(
                "phase2b_formal_wire_covert_channel_only_owner_union_v2_"
                + "0" * 64
            ),
        )


def test_capacity_bounded_v4_owner_union_rejects_conflicting_owner_mutation() -> None:
    path = ("public_case", "graph", "conflict")
    with pytest.raises(ValueError, match="conflicting owner"):
        qualification._channel_only_owner_union_projection_v2(
            (
                auditor.ChannelOnlyLeafOwnerV1(path, "uuid_leaf"),
                auditor.ChannelOnlyLeafOwnerV1(path, "missingness"),
            )
        )


@pytest.mark.parametrize("mode", qualification.LIVE_MODE_ORDER)
def test_closed_feature_capacity_projection_rejects_every_axis_mutation(
    mode: str,
) -> None:
    execution = _fixture(mode)
    closed = qualification._closed_primitive_v2(
        execution.feature_capacity_projection
    )
    assert qualification._validate_closed_feature_capacity_projection_v2(
        mode,
        execution.row_count,
        closed,
    ) is closed
    for field in auditor._FeatureCardinalityProjectionV1.__dataclass_fields__.values():
        tampered = dict(closed)
        if field.name == "projection_root":
            tampered[field.name] = (
                "phase2b_formal_covert_feature_projection_" + "0" * 64
            )
        else:
            assert type(tampered[field.name]) is int
            tampered[field.name] += 1
        with pytest.raises((TypeError, ValueError)):
            qualification._validate_closed_feature_capacity_projection_v2(
                mode,
                execution.row_count,
                tampered,
            )


def test_receipt_projections_do_not_read_nonexistent_owner_count_manifest_field() -> None:
    assert "channel_only_owned_leaf_count" not in {
        field.name for field in auditor.FeatureManifestV1.__dataclass_fields__.values()
    }
    qualification_tree = ast.parse(inspect.getsource(qualification))
    assert not [
        node
        for node in ast.walk(qualification_tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "channel_only_owned_leaf_count"
    ]


def test_fixture_build_preflights_before_first_decode_and_mirrors_decode_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_decode = auditor.FormalWireV3AdapterV1.decode_envelope
    original_validate = auditor._validate_decoded
    original_deep = auditor._deep_object_graph_bytes

    def preflight(rows: int) -> object:
        assert rows == 8
        events.append("preflight")
        return object()

    def decode(
        self: auditor.FormalWireV3AdapterV1,
        envelope: bytes,
    ) -> auditor.DecodedFormalEnvelopeV1:
        events.append("decode")
        return original_decode(self, envelope)

    def validate(
        item: auditor.DecodedFormalEnvelopeV1,
        adapter: auditor.FormalWireV3AdapterV1,
    ) -> None:
        events.append("validate")
        original_validate(item, adapter)

    def deep(value: object) -> int:
        if type(value) is auditor.DecodedFormalEnvelopeV1:
            events.append("deep")
        return original_deep(value)

    monkeypatch.setattr(auditor, "_preflight_decode_memory", preflight)
    monkeypatch.setattr(auditor.FormalWireV3AdapterV1, "decode_envelope", decode)
    monkeypatch.setattr(auditor, "_validate_decoded", validate)
    monkeypatch.setattr(auditor, "_deep_object_graph_bytes", deep)
    execution = qualification._build_fixture_execution_v2(
        "KNOWN_NULL", row_count=8
    )
    assert execution.row_count == 8
    assert events == ["preflight", *[item for _ in range(8) for item in (
        "decode", "validate", "deep"
    )]]


def test_fixture_build_rejects_nonexact_decoded_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auditor, "_preflight_decode_memory", lambda rows: object())
    monkeypatch.setattr(
        auditor.FormalWireV3AdapterV1,
        "decode_envelope",
        lambda self, envelope: object(),
    )
    with pytest.raises(TypeError, match="non-exact decoded DTO"):
        qualification._build_fixture_execution_v2("KNOWN_NULL", row_count=8)


def test_fixture_build_rejects_semantically_invalid_decoded_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_decode = auditor.FormalWireV3AdapterV1.decode_envelope

    def corrupt(
        self: auditor.FormalWireV3AdapterV1,
        envelope: bytes,
    ) -> auditor.DecodedFormalEnvelopeV1:
        item = original_decode(self, envelope)
        object.__setattr__(item, "payload_bytes", b"{}")
        return item

    monkeypatch.setattr(auditor, "_preflight_decode_memory", lambda rows: object())
    monkeypatch.setattr(auditor.FormalWireV3AdapterV1, "decode_envelope", corrupt)
    with pytest.raises(
        auditor.FormalCovertAuditRejectionV1,
        match="FAIL_WIRE_CANONICAL_REPLAY",
    ):
        qualification._build_fixture_execution_v2("KNOWN_NULL", row_count=8)


def test_fixture_build_rejects_decoded_row_over_deep_graph_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_deep = auditor._deep_object_graph_bytes

    def oversized(value: object) -> int:
        if type(value) is auditor.DecodedFormalEnvelopeV1:
            return (
                auditor.MAXIMUM_DECODE_GRAPH_BYTES_PER_ROW
                // auditor.DEEP_SIZE_SAFETY_MULTIPLIER
                + 1
            )
        return original_deep(value)

    monkeypatch.setattr(auditor, "_preflight_decode_memory", lambda rows: object())
    monkeypatch.setattr(auditor, "_deep_object_graph_bytes", oversized)
    with pytest.raises(ValueError, match="conservative graph bound"):
        qualification._build_fixture_execution_v2("KNOWN_NULL", row_count=8)


@pytest.mark.parametrize("mode", qualification.LIVE_MODE_ORDER)
def test_pure_static_replay_rejects_synchronized_capacity_and_owner_splices(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _fixture(mode)
    contract = qualification._FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"][mode]
    full_row = next(row for row in contract["projections"] if row["rows"] == 1_024)
    full_projection = replace(
        execution.feature_capacity_projection,
        rows=qualification.EXACT_AUTHORITY_ROWS,
        descriptor_source_utf8_bytes=(
            execution.feature_capacity_projection.descriptor_source_utf8_bytes * 128
        ),
        present_generic_channel_feature_utf8_bytes=(
            execution.feature_capacity_projection.present_generic_channel_feature_utf8_bytes
            * 128
        ),
        projected_feature_matrix_bytes_bound=full_row[
            "projected_feature_matrix_bytes_bound"
        ],
    )
    assert full_projection.projection_root == full_row["projection_root"]
    resource = {name: 0 for name in qualification._RESOURCE_PLAN_FIELDS_V2}
    resource["preextraction_feature_matrix_bytes_bound"] = (
        full_projection.projected_feature_matrix_bytes_bound
    )
    audit = {
        "audited_rows": qualification.EXACT_AUTHORITY_ROWS,
        "channel_leaf_paths": full_projection.channel_union_path_count,
        "channel_only_owned_leaf_paths": contract[
            "channel_only_owned_union_path_count"
        ],
        "channel_only_owned_leaf_union_root": contract[
            "channel_only_owned_union_root"
        ],
        "concrete_features": full_projection.projected_concrete_feature_count,
        "eligibility_quota": {
            "receipt_id": (
                "phase2b_formal_eligibility_mask_quota_" + "1" * 64
            )
        },
        "feature_capacity_projection": qualification._closed_primitive_v2(
            full_projection
        ),
        "feature_capacity_projection_root": full_projection.projection_root,
        "feature_manifest_root": "manifest-root",
        "full_raw_byte_categorical_hypotheses": 1,
        "namespace_uuid_leaf_paths": full_projection.uuid_union_path_count,
        "raw_semantic_features": (
            full_projection.projected_raw_semantic_feature_count
        ),
        "resource_plan": resource,
    }
    run = {
        "audit": audit,
        "batch_id": "unused",
        "label_factorial_proof_root": "label-root",
        "mode": mode,
        "namespace_row_proof_root": "namespace-root",
        "operation_counts": {},
        "run_fixture_root": "fixture-root",
        "source_pair_delta_root": None,
        "wire": {
            "batch_id": "batch-root",
            "build_success_id": "build-root",
            "transform_catalog_root": "catalog-root",
        },
    }
    monkeypatch.setattr(
        qualification,
        "_validate_complete_resource_plan_projection_v2",
        lambda value, *, context: value,
    )
    frozen = qualification._static_shape_from_run_projection_v2(run)
    assert frozen["feature_capacity_projection_root"] == full_projection.projection_root

    six_splices = (
        ("audited_rows", None),
        ("channel_leaf_paths", None),
        ("concrete_features", None),
        ("preextraction_feature_matrix_bytes_bound", "resource_plan"),
        ("raw_semantic_features", None),
        ("namespace_uuid_leaf_paths", None),
    )
    for field_name, container in six_splices:
        spliced = json.loads(json.dumps(run))
        target = spliced["audit"] if container is None else spliced["audit"][container]
        target[field_name] += 1
        forged_measurement = json.loads(json.dumps(frozen))
        static_field = {
            "audited_rows": "rows",
            "channel_leaf_paths": "channel_leaf_count",
            "concrete_features": "concrete_features",
            "preextraction_feature_matrix_bytes_bound": (
                "preextraction_feature_matrix_bytes_bound"
            ),
            "raw_semantic_features": "raw_semantic_features",
            "namespace_uuid_leaf_paths": "uuid_leaf_count",
        }[field_name]
        if container is None:
            forged_measurement[static_field] += 1
        else:
            forged_measurement["resource_plan_static"][static_field] += 1
            forged_measurement["resource_plan_static_root"] = (
                qualification.stable_hash(
                    forged_measurement["resource_plan_static"],
                    prefix=(
                        "phase2b_formal_wire_covert_static_resource_plan_v2_"
                    ),
                )
            )
        forged_measurement_root = qualification.stable_hash(
            forged_measurement,
            prefix="phase2b_formal_wire_covert_static_shape_v3_",
        )
        assert forged_measurement_root
        with pytest.raises((TypeError, ValueError)):
            qualification._static_shape_from_run_projection_v2(spliced)

    for field_name, replacement in (
        (
            "channel_only_owned_leaf_paths",
            contract["channel_only_owned_union_path_count"] + 1,
        ),
        (
            "channel_only_owned_leaf_union_root",
            "phase2b_formal_wire_covert_channel_only_owner_union_v2_" + "0" * 64,
        ),
    ):
        spliced = json.loads(json.dumps(run))
        spliced["audit"][field_name] = replacement
        forged_measurement = json.loads(json.dumps(frozen))
        forged_measurement[
            "channel_only_owned_leaf_count"
            if field_name == "channel_only_owned_leaf_paths"
            else "channel_only_owned_leaf_union_root"
        ] = replacement
        assert qualification.stable_hash(
            forged_measurement,
            prefix="phase2b_formal_wire_covert_static_shape_v3_",
        )
        with pytest.raises((TypeError, ValueError)):
            qualification._static_shape_from_run_projection_v2(spliced)

    other_mode = next(item for item in qualification.LIVE_MODE_ORDER if item != mode)
    other_contract = qualification._FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"][
        other_mode
    ]
    swapped = json.loads(json.dumps(run))
    swapped["audit"]["channel_only_owned_leaf_paths"] = other_contract[
        "channel_only_owned_union_path_count"
    ]
    swapped["audit"]["channel_only_owned_leaf_union_root"] = other_contract[
        "channel_only_owned_union_root"
    ]
    with pytest.raises(ValueError):
        qualification._static_shape_from_run_projection_v2(swapped)


def test_post_audit_fake_receipt_uses_execution_owner_count_and_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invariant_runtime_type = auditor.InvariantRuntimeEvidenceV1
    statistics_runtime_type = auditor.StatisticalRuntimeEvidenceV1
    issuance_runtime_type = auditor.ReceiptIssuanceRuntimeEvidenceV1
    invariant_runtime_field_names = tuple(
        item.name for item in fields(invariant_runtime_type)
    )
    statistics_runtime_field_names = tuple(
        item.name for item in fields(statistics_runtime_type)
    )
    issuance_runtime_field_names = tuple(
        item.name for item in fields(issuance_runtime_type)
    )

    class FakeInvariant(SimpleNamespace):
        def __post_init__(self) -> None:
            return None

    class FakeStatistics(SimpleNamespace):
        def __post_init__(self) -> None:
            return None

    class FakeIssuance(SimpleNamespace):
        def __post_init__(self) -> None:
            return None

    class FakeReceipt(SimpleNamespace):
        def __post_init__(self) -> None:
            return None

    mode = "KNOWN_NULL"
    contract = qualification._FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"][mode]
    full_projection = next(
        row for row in contract["projections"] if row["rows"] == 1_024
    )
    capacity = {
        "audited_rows": 1_024,
        "channel_leaf_count": contract["union_shape"][
            "channel_union_path_count"
        ],
        "concrete_feature_count": full_projection[
            "projected_concrete_feature_count"
        ],
        "preextraction_feature_matrix_bytes_bound": full_projection[
            "projected_feature_matrix_bytes_bound"
        ],
        "raw_semantic_feature_count": full_projection[
            "projected_raw_semantic_feature_count"
        ],
        "uuid_leaf_count": contract["union_shape"]["uuid_union_path_count"],
    }
    manifest = SimpleNamespace(
        alias_coverage=(
            SimpleNamespace(semantic_aliases=("owner/raw_byte_categorical",)),
        ),
        channel_leaf_count=capacity["channel_leaf_count"],
        deduplicated_feature_count=capacity["concrete_feature_count"],
        manifest_root="manifest-root",
        raw_semantic_feature_count=capacity["raw_semantic_feature_count"],
        uuid_leaf_count=capacity["uuid_leaf_count"],
    )
    assert not hasattr(manifest, "channel_only_owned_leaf_count")
    resource = SimpleNamespace(
        admissible=True,
        invariance_resource_plan_root=(
            "phase2b_formal_covert_invariant_resource_plan_" + "0" * 64
        ),
        label_permutations=qualification.EXACT_LABEL_PERMUTATIONS,
        preextraction_feature_matrix_bytes_bound=capacity[
            "preextraction_feature_matrix_bytes_bound"
        ],
        resource_plan_root=(
            "phase2b_formal_covert_statistics_resource_plan_" + "1" * 64
        ),
    )
    cgroup_membership = "/test.slice/qv2.scope"
    cgroup_device = 11
    cgroup_inode = 22
    cgroup_scope_root = qualification.stable_hash(
        {
            "leaf_device": cgroup_device,
            "leaf_inode": cgroup_inode,
            "unified_membership": cgroup_membership,
        },
        prefix="phase2b_formal_covert_unified_cgroup_memory_scope_v1_",
    )
    cgroup_current = 4_096
    cgroup_peak = 8_192
    cgroup_max = 24 * 1_024 * 1_024 * 1_024
    invariant = FakeInvariant(
        consumer_decode_count=qualification.EXACT_AUTHORITY_ROWS * 80,
        invariance_resource_plan_root=resource.invariance_resource_plan_root,
        invariance_wall_time_nanoseconds=1,
        resource_preflight=SimpleNamespace(),
        variant_suite_root="variant-suite-root",
        worker_peak_rss_bytes=1_024,
        worker_result_artifact_sha256="invariant-worker-sha",
        unified_cgroup_membership=cgroup_membership,
        unified_cgroup_memory_scope_root=cgroup_scope_root,
        unified_cgroup_leaf_device=cgroup_device,
        unified_cgroup_leaf_inode=cgroup_inode,
        physical_memory_current_bytes_at_completion=cgroup_current,
        physical_memory_peak_bytes_at_completion=cgroup_peak,
        physical_memory_max_bytes=cgroup_max,
        physical_memory_cap_bytes=auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
        within_frozen_physical_peak_cap=True,
    )
    statistics = FakeStatistics(
        diagnostic_parent_plus_worker_historical_peak_sum_bytes=2_048,
        external_supervisor_required=True,
        host_available_memory_bytes_at_preflight=1,
        parent_historical_peak_rss_bytes_before_receipt_finalization=1_024,
        unified_cgroup_membership=cgroup_membership,
        unified_cgroup_memory_scope_root=cgroup_scope_root,
        unified_cgroup_leaf_device=cgroup_device,
        unified_cgroup_leaf_inode=cgroup_inode,
        physical_memory_current_bytes_before_receipt_finalization=cgroup_current,
        physical_memory_peak_bytes_before_receipt_finalization=cgroup_peak,
        physical_memory_max_bytes=cgroup_max,
        physical_memory_cap_bytes=auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
        receipt_finalization_complete_pending_validation_count=1,
        receipt_finalization_physical_peak_within_cap=True,
        resource_plan_root=resource.resource_plan_root,
        resource_preflight=SimpleNamespace(),
        statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds=1,
        worker_peak_rss_bytes=(1_024,),
        worker_result_artifact_sha256=("statistics-worker-sha",),
        within_frozen_physical_peak_cap=True,
    )
    issuance = FakeIssuance(
        absolute_deadline_kind="STATISTICS",
        absolute_deadline_monotonic_nanoseconds=1,
        absolute_deadline_timeout_seconds=auditor.STATISTICS_WORKER_TIMEOUT_SECONDS,
        diagnostic_worker_historical_peak_rss_bytes_sum=1_024,
        diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization=1_024,
        diagnostic_parent_plus_worker_historical_peak_sum_bytes=2_048,
        unified_cgroup_membership=cgroup_membership,
        unified_cgroup_memory_scope_root=cgroup_scope_root,
        unified_cgroup_leaf_device=cgroup_device,
        unified_cgroup_leaf_inode=cgroup_inode,
        physical_memory_current_bytes_before_receipt_finalization=cgroup_current,
        physical_memory_peak_bytes_before_receipt_finalization=cgroup_peak,
        physical_memory_max_bytes=cgroup_max,
        physical_memory_cap_bytes=auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
        receipt_finalization_complete_pending_validation_count=1,
        receipt_finalization_locked_deadline_check_count=1,
        receipt_finalization_locked_deadline_within_absolute_limit=True,
        receipt_finalization_locked_physical_peak_within_cap=True,
    )
    probabilities = SimpleNamespace(numerator=1, denominator=1)
    target_family_results = tuple(
        SimpleNamespace(
            exceedance_count=0,
            holm_adjusted_p=probabilities,
            holm_critical_value=probabilities,
            holm_rank=index + 1,
            holm_rejected=False,
            observed_target_max_omnibus=0.0,
            raw_p=probabilities,
            target=target,
        )
        for index, target in enumerate(auditor.TARGETS)
    )
    fake_receipt_id = "phase2b_formal_covert_audit_transcript_" + "0" * 64
    fake_quota_receipt_id = (
        "phase2b_formal_eligibility_mask_quota_" + "1" * 64
    )
    receipt = FakeReceipt(
        actual_corpus_audit=False,
        adapter_execution_pin_root="adapter-pin-root",
        audited_row_count=qualification.EXACT_AUTHORITY_ROWS,
        authority_independence_verified=False,
        authority_origin_authenticated=False,
        disposition=auditor.FormalCovertDisposition.PASS,
        eligibility_mask_quota=SimpleNamespace(
            exact_quota_matched=True,
            receipt_id=fake_quota_receipt_id,
            targets=(),
        ),
        exact_invariant_comparisons=qualification.EXACT_AUTHORITY_ROWS * 80 * 5,
        feature_manifest=manifest,
        holm_family_size=len(auditor.TARGETS),
        invariant_execution_root="invariant-root",
        invariant_run_count=qualification.EXACT_RECOGNIZER_INVOCATIONS,
        invariant_runtime_evidence=invariant,
        pairwise_holm_decision_eligible=False,
        pairwise_holm_family_size=0,
        permutation_schedule_root="permutation-root",
        permutation_seed_commitment="permutation-seed-commitment",
        permutation_seed_reveal_hex="00",
        positive_decision_claim=auditor.POSITIVE_DECISION_CLAIM,
        receipt_id=fake_receipt_id,
        resource_plan=resource,
        results=(),
        receipt_issuance_runtime_evidence=issuance,
        statistics_runtime_evidence=statistics,
        target_family_results=target_family_results,
        wire_alignment_issuance_verified=True,
    )
    base = _fixture(mode)
    full_capacity_projection = replace(
        base.feature_capacity_projection,
        rows=qualification.EXACT_AUTHORITY_ROWS,
        descriptor_source_utf8_bytes=(
            base.feature_capacity_projection.descriptor_source_utf8_bytes * 128
        ),
        present_generic_channel_feature_utf8_bytes=(
            base.feature_capacity_projection.present_generic_channel_feature_utf8_bytes
            * 128
        ),
        projected_feature_matrix_bytes_bound=full_projection[
            "projected_feature_matrix_bytes_bound"
        ],
    )
    assert full_capacity_projection.projection_root == full_projection[
        "projection_root"
    ]
    build_success = SimpleNamespace(
        key_source_receipt=SimpleNamespace(receipt_id="key-source-root"),
        private_audit_alignment=SimpleNamespace(
            alignment_id="alignment-root",
            source_row_index_by_public_row=tuple(
                range(qualification.EXACT_AUTHORITY_ROWS)
            ),
        ),
        public_batch=SimpleNamespace(batch_id="batch-root"),
        success_id="build-root",
        variant_context=SimpleNamespace(context_id="variant-root"),
    )
    execution = SimpleNamespace(
        adapter=base.adapter,
        build_success=build_success,
        channel_only_owner_union_root=contract[
            "channel_only_owned_union_root"
        ],
        exact_owned_channel_only_leaf_paths=contract[
            "channel_only_owned_union_path_count"
        ],
        feature_capacity_projection=full_capacity_projection,
        label_factorial_proof_root=base.label_factorial_proof_root,
        namespace_row_proof_root=base.namespace_row_proof_root,
        operation_counts=tuple(
            (operation, qualification.EXACT_ROWS_PER_TRANSFORM_OPERATION)
            for operation in qualification.EXACT_TRANSFORM_OPERATIONS
        ),
        row_count=qualification.EXACT_AUTHORITY_ROWS,
        run_fixture_root=base.run_fixture_root,
        source_pair_delta_root=None,
        transform_catalog_binding_receipt=SimpleNamespace(
            transform_catalog_root=(
                base.transform_catalog_binding_receipt.transform_catalog_root
            )
        ),
    )
    monkeypatch.setattr(auditor, "FormalCovertAuditReceiptV1", FakeReceipt)
    monkeypatch.setattr(auditor, "InvariantRuntimeEvidenceV1", FakeInvariant)
    monkeypatch.setattr(auditor, "StatisticalRuntimeEvidenceV1", FakeStatistics)
    monkeypatch.setattr(
        auditor, "ReceiptIssuanceRuntimeEvidenceV1", FakeIssuance
    )
    public_validations: list[object] = []

    def validate_once(value: object) -> str:
        public_validations.append(value)
        return value.receipt_id

    monkeypatch.setattr(
        auditor,
        "validate_formal_covert_audit_receipt_v1",
        validate_once,
    )
    monkeypatch.setattr(
        auditor,
        "_validate_invariant_phase_preflight_against_resource_plan",
        lambda plan, preflight: None,
    )
    monkeypatch.setattr(
        auditor,
        "_validate_statistics_phase_preflight_against_resource_plan",
        lambda plan, preflight: None,
    )
    monkeypatch.setattr(
        qualification,
        "_fixture_capacity_receipt_projection_from_execution_v2",
        lambda value: dict(capacity),
    )
    evidence = qualification._validate_isolated_audit_receipt_v2(
        mode,
        time.perf_counter_ns(),
        execution,
        receipt,
    )
    monkeypatch.setattr(
        qualification,
        "_static_resource_plan_projection_v2",
        lambda value: {"fake_static_resource": True},
    )
    static_projection = qualification._frozen_static_shape_projection_v2(evidence)
    original_closed_primitive = qualification._closed_primitive_v2
    resource_projection = {
        item.name: 0 for item in fields(auditor.ResourcePlanV1)
    }
    resource_projection.update(
        {
            "admissible": True,
            "invariance_resource_plan_root": (
                resource.invariance_resource_plan_root
            ),
            "preextraction_feature_matrix_bytes_bound": capacity[
                "preextraction_feature_matrix_bytes_bound"
            ],
            "resource_plan_root": resource.resource_plan_root,
        }
    )
    invariant_projection = {
        name: 0 for name in invariant_runtime_field_names
    }
    invariant_projection.update(
        {
            "invariance_resource_plan_root": (
                resource.invariance_resource_plan_root
            ),
            "resource_preflight": {},
        }
    )
    statistics_projection = {
        name: 0 for name in statistics_runtime_field_names
    }
    statistics_projection.update(
        {
            "resource_plan_root": resource.resource_plan_root,
            "resource_preflight": {},
        }
    )
    issuance_projection = {
        name: getattr(issuance, name) for name in issuance_runtime_field_names
    }

    def fake_closed_primitive(value: object) -> object:
        if value is resource:
            return dict(resource_projection)
        if value is invariant:
            return dict(invariant_projection)
        if value is statistics:
            return dict(statistics_projection)
        if value is issuance:
            return dict(issuance_projection)
        return original_closed_primitive(value)

    monkeypatch.setattr(
        qualification,
        "_closed_primitive_v2",
        fake_closed_primitive,
    )
    monkeypatch.setattr(
        qualification,
        "_validate_complete_resource_plan_projection_v2",
        lambda value, *, context: value,
    )
    monkeypatch.setattr(
        qualification,
        "_validate_complete_invariant_runtime_projection_v2",
        lambda value, resource_plan, *, context: value,
    )
    monkeypatch.setattr(
        qualification,
        "_validate_complete_statistics_runtime_projection_v2",
        lambda value, resource_plan, *, context: value,
    )
    monkeypatch.setattr(
        qualification,
        "_validate_complete_receipt_issuance_runtime_projection_v2",
        lambda value, statistics_runtime, *, context: value,
    )
    monkeypatch.setattr(
        qualification,
        "_validate_executed_runtime_bundle_v2",
        lambda **kwargs: None,
    )
    primitive_projection = qualification._audit_primitive_projection_v2(evidence)
    monkeypatch.setattr(auditor, "InvariantRuntimeEvidenceV1", invariant_runtime_type)
    monkeypatch.setattr(
        auditor,
        "StatisticalRuntimeEvidenceV1",
        statistics_runtime_type,
    )
    monkeypatch.setattr(
        auditor,
        "ReceiptIssuanceRuntimeEvidenceV1",
        issuance_runtime_type,
    )
    assert static_projection["channel_only_owned_leaf_count"] == (
        contract["channel_only_owned_union_path_count"]
    )
    assert static_projection["channel_only_owned_leaf_union_root"] == (
        contract["channel_only_owned_union_root"]
    )
    assert static_projection["feature_capacity_projection_root"] == (
        full_projection["projection_root"]
    )
    assert static_projection["feature_capacity_projection"][
        "projection_root"
    ] == static_projection["feature_capacity_projection_root"]
    assert primitive_projection["audit"]["channel_only_owned_leaf_paths"] == (
        static_projection["channel_only_owned_leaf_count"]
    )
    assert primitive_projection["audit"][
        "channel_only_owned_leaf_union_root"
    ] == static_projection["channel_only_owned_leaf_union_root"]
    assert primitive_projection["audit"][
        "feature_capacity_projection"
    ] == static_projection["feature_capacity_projection"]
    assert primitive_projection["audit"][
        "feature_capacity_projection_root"
    ] == static_projection["feature_capacity_projection_root"]
    replayed_static = qualification._static_shape_from_run_projection_v2(
        primitive_projection
    )
    measurement = {
        "diagnostics": primitive_projection["diagnostics"],
        "dynamic_fields_are_recorded_but_not_exact_compared": list(
            qualification.DYNAMIC_RESOURCE_PLAN_FIELDS_NOT_EXACT_COMPARED_V2
        ),
        "frozen_static_projection": replayed_static,
        "frozen_static_projection_root": qualification.stable_hash(
            replayed_static,
            prefix="phase2b_formal_wire_covert_static_shape_v3_",
        ),
        "mode": mode,
        "observed_complete_receipt_issuance_runtime": issuance_projection,
        "observed_complete_receipt_issuance_runtime_root": qualification.stable_hash(
            issuance_projection,
            prefix="phase2b_formal_wire_covert_receipt_issuance_runtime_v4_",
        ),
        "observed_complete_resource_plan": resource_projection,
        "observed_complete_resource_plan_root": qualification.stable_hash(
            resource_projection,
            prefix="phase2b_formal_wire_covert_observed_resource_plan_v3_",
        ),
        "observed_receipt_id": receipt.receipt_id,
    }
    assert qualification._validate_run_and_measurement_binding_v2(
        primitive_projection,
        measurement,
    ) == replayed_static
    assert public_validations == [receipt]
    tampered_run = json.loads(json.dumps(primitive_projection))
    tampered_run["audit"]["receipt_issuance"]["receipt_id"] = "other-receipt"
    with pytest.raises(ValueError, match="resource/issuance bindings"):
        qualification._validate_run_and_measurement_binding_v2(
            tampered_run,
            measurement,
        )
    tampered_measurement = json.loads(json.dumps(measurement))
    tampered_measurement["observed_receipt_id"] = "other-receipt"
    with pytest.raises(ValueError, match="cross-binding"):
        qualification._validate_run_and_measurement_binding_v2(
            primitive_projection,
            tampered_measurement,
        )
    tampered_measurement = json.loads(json.dumps(measurement))
    tampered_measurement[
        "observed_complete_receipt_issuance_runtime_root"
    ] = "phase2b_formal_wire_covert_receipt_issuance_runtime_v4_" + "f" * 64
    with pytest.raises(ValueError, match="cross-binding"):
        qualification._validate_run_and_measurement_binding_v2(
            primitive_projection,
            tampered_measurement,
        )
    tamper_paths = (
        ("invariance", "invariance_resource_plan_root"),
        ("statistics", "resource_plan_root"),
    )
    for group_name, field_name in tamper_paths:
        for complete in (False, True):
            tampered_run = json.loads(json.dumps(primitive_projection))
            target = tampered_run["audit"][group_name]
            if complete:
                target = target["complete_runtime_projection"]
            prefix = (
                "phase2b_formal_covert_invariant_resource_plan_"
                if group_name == "invariance"
                else "phase2b_formal_covert_statistics_resource_plan_"
            )
            target[field_name] = prefix + "f" * 64
            with pytest.raises(ValueError, match="resource/issuance bindings"):
                qualification._validate_run_and_measurement_binding_v2(
                    tampered_run,
                    measurement,
                )


@pytest.mark.parametrize("mode", qualification.LIVE_MODE_ORDER)
def test_full1024_receipt_capacity_projection_is_exactly_frozen(mode: str) -> None:
    mode_contract = qualification._FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"][
        mode
    ]
    expected_projection = next(
        row for row in mode_contract["projections"] if row["rows"] == 1_024
    )
    valid = {
        "audited_rows": 1_024,
        "channel_leaf_count": mode_contract["union_shape"][
            "channel_union_path_count"
        ],
        "concrete_feature_count": expected_projection[
            "projected_concrete_feature_count"
        ],
        "preextraction_feature_matrix_bytes_bound": expected_projection[
            "projected_feature_matrix_bytes_bound"
        ],
        "raw_semantic_feature_count": expected_projection[
            "projected_raw_semantic_feature_count"
        ],
        "uuid_leaf_count": mode_contract["union_shape"][
            "uuid_union_path_count"
        ],
    }
    assert qualification._validate_fixture_capacity_receipt_projection_v2(
        mode, valid
    ) is valid
    for name in tuple(valid):
        tampered = dict(valid)
        tampered[name] += 1
        with pytest.raises(ValueError, match="receipt capacity projection drift"):
            qualification._validate_fixture_capacity_receipt_projection_v2(
                mode, tampered
            )
    with pytest.raises(TypeError, match="projection is not exact"):
        qualification._validate_fixture_capacity_receipt_projection_v2(
            mode, {**valid, "unexpected": 0}
        )


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


def _valid_candidate_runtime_bundle(
    rows: int = qualification.EXACT_AUTHORITY_ROWS,
) -> tuple[
    auditor.ResourcePlanV1,
    auditor.StatisticalRuntimeEvidenceV1,
    auditor.ReceiptIssuanceRuntimeEvidenceV1,
]:
    columns = (auditor._FeatureColumn(("unit",), tuple("a" for _ in range(rows))),)
    deadline_ns = time.monotonic_ns() + 5_000_000_000
    partition_projection = auditor._partition_allocation_projection(
        rows,
        columns,
        deadline_ns,
    )
    partitions = auditor._materialize_partitions_bounded(
        columns,
        partition_projection,
        deadline_ns,
    )
    envelope_and_decode_bytes_bound = rows * auditor.ENVELOPE_BYTES * 2
    invariant_plan = auditor._invariance_resource_plan(
        rows,
        envelope_and_decode_bytes_bound=envelope_and_decode_bytes_bound,
        adapter_retained_graph_bytes_bound=1,
        variant_suite_resident_bytes_per_row_bound=1,
        variant_suite_resident_fixed_bytes_bound=1,
    )
    plan = auditor._resource_plan(
        rows,
        partitions,
        (2,) * len(auditor.TARGETS),
        preextraction_feature_matrix_bytes_bound=rows * 72,
        feature_matrix_bytes_bound=rows * 72,
        envelope_and_decode_bytes_bound=envelope_and_decode_bytes_bound,
        adapter_retained_graph_bytes_bound=1,
        partition_projection=partition_projection,
        partition_preparation_deadline_ns=deadline_ns,
        invariance_plan=invariant_plan,
    )
    statistics_static = (
        plan.permutation_label_mask_bytes_bound
        + plan.partition_matrix_bytes_bound
        + plan.adapter_retained_graph_bytes_bound
        + 64 * 1_048_576
    )
    statistics_additional = (
        plan.metric_cache_bytes_bound * plan.maximum_parallel_workers
        + plan.permutation_metric_tuple_bytes_bound
        + plan.target_max_vector_bytes_bound
        + plan.worker_result_artifact_bytes_bound
        + plan.parent_result_working_bytes_bound
    )
    statistics_multiplier = 1 + plan.maximum_parallel_workers
    preflight = auditor._PhaseResourcePreflightEvidenceV1(
        phase="STATISTICS",
        resource_plan_root=plan.resource_plan_root,
        parent_current_rss_bytes_at_preflight=plan.parent_current_rss_bytes_at_plan,
        static_parent_bytes_bound=statistics_static,
        fork_parent_rss_multiplier=statistics_multiplier,
        additional_bytes_bound=statistics_additional,
        projected_aggregate_rss_bytes_at_preflight=(
            max(plan.parent_current_rss_bytes_at_plan, statistics_static)
            * statistics_multiplier
            + statistics_additional
        ),
        maximum_estimated_rss_bytes=plan.maximum_estimated_rss_bytes,
        host_available_memory_safety_margin_bytes=(
            plan.host_available_memory_safety_margin_bytes
        ),
    )
    available = (
        preflight.projected_aggregate_rss_bytes_at_preflight
        + preflight.host_available_memory_safety_margin_bytes
    )
    worker_peaks = (0,) * plan.maximum_parallel_workers
    cgroup_snapshot = _test_auditor_cgroup_snapshot()
    statistics = auditor.StatisticalRuntimeEvidenceV1(
        resource_plan_root=plan.resource_plan_root,
        resource_preflight=preflight,
        system_available_memory_bytes_at_preflight=available,
        cgroup_remaining_memory_bytes_at_preflight=None,
        host_available_memory_bytes_at_preflight=available,
        parent_historical_peak_rss_bytes_before_receipt_finalization=0,
        worker_peak_rss_bytes=worker_peaks,
        worker_result_artifact_bytes=(1,) * 5,
        worker_result_artifact_sha256=("sha256:" + "0" * 64,) * 5,
        parent_result_object_graph_bytes_observed=0,
        diagnostic_parent_plus_worker_historical_peak_sum_bytes=0,
        unified_cgroup_membership=cgroup_snapshot.unified_membership,
        unified_cgroup_memory_scope_root=cgroup_snapshot.memory_scope_root,
        unified_cgroup_leaf_device=cgroup_snapshot.leaf_device,
        unified_cgroup_leaf_inode=cgroup_snapshot.leaf_inode,
        physical_memory_current_bytes_before_receipt_finalization=(
            cgroup_snapshot.memory_current_bytes
        ),
        physical_memory_peak_bytes_before_receipt_finalization=(
            cgroup_snapshot.memory_peak_bytes
        ),
        physical_memory_max_bytes=cgroup_snapshot.memory_max_bytes,
        physical_memory_cap_bytes=auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
        statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds=1,
        worker_timeout_seconds=auditor.STATISTICS_WORKER_TIMEOUT_SECONDS,
        parent_postprocess_deadline_schema=auditor.PARENT_POSTPROCESS_DEADLINE_SCHEMA,
        parent_postprocess_deadline_preemptive=False,
        external_supervisor_required=True,
        receipt_finalization_complete_pending_validation_count=1,
        receipt_finalization_physical_peak_within_cap=True,
        within_frozen_physical_peak_cap=True,
    )
    issuance = auditor.ReceiptIssuanceRuntimeEvidenceV1(
        absolute_deadline_kind="STATISTICS",
        absolute_deadline_monotonic_nanoseconds=1,
        absolute_deadline_timeout_seconds=auditor.STATISTICS_WORKER_TIMEOUT_SECONDS,
        diagnostic_worker_historical_peak_rss_bytes_sum=0,
        diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization=0,
        diagnostic_parent_plus_worker_historical_peak_sum_bytes=0,
        unified_cgroup_membership=cgroup_snapshot.unified_membership,
        unified_cgroup_memory_scope_root=cgroup_snapshot.memory_scope_root,
        unified_cgroup_leaf_device=cgroup_snapshot.leaf_device,
        unified_cgroup_leaf_inode=cgroup_snapshot.leaf_inode,
        physical_memory_current_bytes_before_receipt_finalization=(
            cgroup_snapshot.memory_current_bytes
        ),
        physical_memory_peak_bytes_before_receipt_finalization=(
            cgroup_snapshot.memory_peak_bytes
        ),
        physical_memory_max_bytes=cgroup_snapshot.memory_max_bytes,
        physical_memory_cap_bytes=auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
        receipt_finalization_complete_pending_validation_count=1,
        receipt_finalization_locked_deadline_check_count=1,
        receipt_finalization_locked_deadline_within_absolute_limit=True,
        receipt_finalization_locked_physical_peak_within_cap=True,
    )
    return plan, statistics, issuance


def _inadmissible_candidate_runtime_bundle(
    plan: auditor.ResourcePlanV1,
    statistics: auditor.StatisticalRuntimeEvidenceV1,
) -> tuple[auditor.ResourcePlanV1, auditor.StatisticalRuntimeEvidenceV1]:
    invariant_parent = auditor.MAXIMUM_STATIC_PHASE_PLAN_RSS_BYTES
    invariant_peak = (
        max(
            invariant_parent,
            plan.envelope_and_decode_bytes_bound
            + plan.adapter_retained_graph_bytes_bound
            + 64 * 1_048_576,
        )
        * 2
        + plan.invariance_variant_suite_resident_bytes_bound
        + plan.invariance_recognizer_live_output_bytes_bound
        + plan.invariance_worker_artifact_bytes_bound
    )
    invariant_plan = auditor._InvariantResourcePlanV1(
        rows=plan.rows,
        envelope_and_decode_bytes_bound=plan.envelope_and_decode_bytes_bound,
        adapter_retained_graph_bytes_bound=(
            plan.adapter_retained_graph_bytes_bound
        ),
        parent_current_rss_bytes_at_plan=invariant_parent,
        invariance_variant_suite_output_bytes_bound=(
            plan.invariance_variant_suite_output_bytes_bound
        ),
        variant_suite_resident_bytes_per_row_bound=(
            plan.variant_suite_resident_bytes_per_row_bound
        ),
        variant_suite_resident_fixed_bytes_bound=(
            plan.variant_suite_resident_fixed_bytes_bound
        ),
        invariance_variant_suite_resident_bytes_bound=(
            plan.invariance_variant_suite_resident_bytes_bound
        ),
        invariance_recognizer_live_output_bytes_bound=(
            plan.invariance_recognizer_live_output_bytes_bound
        ),
        invariance_worker_artifact_bytes_bound=(
            plan.invariance_worker_artifact_bytes_bound
        ),
        estimated_invariance_peak_rss_bytes=invariant_peak,
        maximum_invariance_estimated_rss_bytes=(
            plan.maximum_invariance_estimated_rss_bytes
        ),
        invariance_rss_formula=plan.invariance_rss_formula,
        host_available_memory_safety_margin_bytes=(
            plan.host_available_memory_safety_margin_bytes
        ),
        admissible=False,
    )
    inadmissible_plan = replace(
        plan,
        invariance_parent_current_rss_bytes_at_plan=invariant_parent,
        invariance_resource_plan_root=invariant_plan.plan_root,
        estimated_invariance_peak_rss_bytes=invariant_peak,
        admissible=False,
    )
    inadmissible_preflight = replace(
        statistics.resource_preflight,
        resource_plan_root=inadmissible_plan.resource_plan_root,
    )
    inadmissible_statistics = replace(
        statistics,
        resource_plan_root=inadmissible_plan.resource_plan_root,
        resource_preflight=inadmissible_preflight,
    )
    inadmissible_plan.__post_init__()
    inadmissible_statistics.__post_init__()
    return inadmissible_plan, inadmissible_statistics


def _valid_executed_runtime_bundle_projection() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    plan, statistics_runtime, issuance_runtime = _valid_candidate_runtime_bundle()
    invariant_static = (
        plan.envelope_and_decode_bytes_bound
        + plan.adapter_retained_graph_bytes_bound
        + 64 * 1_048_576
    )
    invariant_additional = (
        plan.invariance_variant_suite_resident_bytes_bound
        + plan.invariance_recognizer_live_output_bytes_bound
        + plan.invariance_worker_artifact_bytes_bound
    )
    invariant_preflight = auditor._PhaseResourcePreflightEvidenceV1(
        phase="INVARIANCE",
        resource_plan_root=plan.invariance_resource_plan_root,
        parent_current_rss_bytes_at_preflight=(
            plan.invariance_parent_current_rss_bytes_at_plan
        ),
        static_parent_bytes_bound=invariant_static,
        fork_parent_rss_multiplier=2,
        additional_bytes_bound=invariant_additional,
        projected_aggregate_rss_bytes_at_preflight=(
            max(
                plan.invariance_parent_current_rss_bytes_at_plan,
                invariant_static,
            )
            * 2
            + invariant_additional
        ),
        maximum_estimated_rss_bytes=plan.maximum_invariance_estimated_rss_bytes,
        host_available_memory_safety_margin_bytes=(
            plan.host_available_memory_safety_margin_bytes
        ),
    )
    invariant_available = (
        invariant_preflight.projected_aggregate_rss_bytes_at_preflight
        + invariant_preflight.host_available_memory_safety_margin_bytes
    )
    parent_rss = invariant_preflight.parent_current_rss_bytes_at_preflight
    adapter_pin = "phase2b_formal_covert_adapter_execution_pins_" + "0" * 64
    cgroup_snapshot = _test_auditor_cgroup_snapshot()
    invariant_runtime = auditor.InvariantRuntimeEvidenceV1(
        invariance_resource_plan_root=plan.invariance_resource_plan_root,
        resource_preflight=invariant_preflight,
        system_available_memory_bytes_at_preflight=invariant_available,
        cgroup_remaining_memory_bytes_at_preflight=None,
        host_available_memory_bytes_at_preflight=invariant_available,
        parent_current_rss_bytes_at_start=parent_rss,
        parent_peak_current_rss_bytes_observed=parent_rss,
        parent_current_rss_bytes_at_completion=parent_rss,
        parent_historical_peak_rss_bytes=parent_rss,
        worker_peak_rss_bytes=0,
        worker_result_artifact_bytes=1,
        worker_result_artifact_sha256="sha256:" + "0" * 64,
        diagnostic_parent_plus_worker_historical_peak_sum_bytes=parent_rss,
        unified_cgroup_membership=cgroup_snapshot.unified_membership,
        unified_cgroup_memory_scope_root=cgroup_snapshot.memory_scope_root,
        unified_cgroup_leaf_device=cgroup_snapshot.leaf_device,
        unified_cgroup_leaf_inode=cgroup_snapshot.leaf_inode,
        physical_memory_current_bytes_at_completion=(
            cgroup_snapshot.memory_current_bytes
        ),
        physical_memory_peak_bytes_at_completion=cgroup_snapshot.memory_peak_bytes,
        physical_memory_max_bytes=cgroup_snapshot.memory_max_bytes,
        physical_memory_cap_bytes=auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
        invariance_wall_time_nanoseconds=1,
        worker_timeout_seconds=auditor.INVARIANCE_WORKER_TIMEOUT_SECONDS,
        invariant_run_count=qualification.EXACT_RECOGNIZER_INVOCATIONS,
        exact_invariant_comparisons=(
            qualification.EXACT_AUTHORITY_ROWS * 80 * 5
        ),
        fork_input_snapshot_root=(
            "phase2b_formal_covert_invariance_input_snapshot_" + "0" * 64
        ),
        adapter_execution_pin_root=adapter_pin,
        variant_suite_root="phase2b_formal_variant_suite_" + "0" * 64,
        consumer_decode_count=qualification.EXACT_AUTHORITY_ROWS * 80,
        variant_suite_unique_envelope_object_count=0,
        variant_suite_resident_object_graph_bytes_observed=0,
        recognizer_output_root=(
            "phase2b_formal_covert_recognizer_output_" + "0" * 64
        ),
        execution_containment=auditor.INVARIANCE_EXECUTION_CONTAINMENT,
        parent_side_adapter_tcb=auditor.PARENT_SIDE_ADAPTER_TCB,
        parent_side_adapter_sandboxed=False,
        process_group_empty_at_exit=True,
        child_address_space_limit_bytes=1,
        within_frozen_physical_peak_cap=True,
    )
    plan_projection = qualification._closed_primitive_v2(plan)
    invariant_projection = qualification._closed_primitive_v2(invariant_runtime)
    statistics_projection = qualification._closed_primitive_v2(statistics_runtime)
    issuance_projection = qualification._closed_primitive_v2(issuance_runtime)
    receipt_id = "phase2b_formal_covert_audit_transcript_" + "0" * 64
    quota_receipt_id = "phase2b_formal_eligibility_mask_quota_" + "1" * 64
    seed_reveal = "0" * 64
    diagnostics = qualification._diagnostic_projection(
        qualification.AuditRunDiagnosticsV2(
            mode="KNOWN_NULL",
            wall_time_ns=1,
            parent_historical_peak_rss_before_receipt_finalization_kib=0,
            statistics_worker_historical_peak_rss_kib=tuple(
                0 for _ in statistics_runtime.worker_peak_rss_bytes
            ),
            parent_plus_worker_historical_peak_rss_kib=0,
            unified_cgroup_membership=cgroup_snapshot.unified_membership,
            unified_cgroup_memory_scope_root=cgroup_snapshot.memory_scope_root,
            unified_cgroup_leaf_device=cgroup_snapshot.leaf_device,
            unified_cgroup_leaf_inode=cgroup_snapshot.leaf_inode,
            physical_memory_current_before_receipt_finalization_bytes=(
                cgroup_snapshot.memory_current_bytes
            ),
            physical_memory_peak_before_receipt_finalization_bytes=(
                cgroup_snapshot.memory_peak_bytes
            ),
            physical_memory_max_bytes=cgroup_snapshot.memory_max_bytes,
            physical_memory_cap_bytes=auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
            statistics_wall_time_ns=(
                statistics_runtime.statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds
            ),
            host_available_memory_bytes_at_preflight=(
                statistics_runtime.host_available_memory_bytes_at_preflight
            ),
            recognizer_adapter_invocations=(
                qualification.EXACT_RECOGNIZER_INVOCATIONS
            ),
            minimum_rows_per_invocation=qualification.EXACT_AUTHORITY_ROWS,
            maximum_rows_per_invocation=qualification.EXACT_AUTHORITY_ROWS,
        )
    )
    run = {
        "mode": "KNOWN_NULL",
        "disposition": "PASS",
        "diagnostics": diagnostics,
        "wire": {"adapter_execution_pin_root": adapter_pin},
        "audit": {
            "receipt_id": receipt_id,
            "audited_rows": qualification.EXACT_AUTHORITY_ROWS,
            "feature_manifest_root": (
                "phase2b_formal_covert_feature_manifest_" + "0" * 64
            ),
            "concrete_features": plan.concrete_features,
            "resource_plan": plan_projection,
            "resource_plan_root": qualification.stable_hash(
                plan_projection,
                prefix="phase2b_formal_wire_covert_resource_plan_v3_",
            ),
            "invariance": {
                "invariance_resource_plan_root": plan.invariance_resource_plan_root,
                "consumer_decode_count": invariant_runtime.consumer_decode_count,
                "exact_comparisons": invariant_runtime.exact_invariant_comparisons,
                "execution_root": (
                    "phase2b_formal_covert_invariant_execution_" + "0" * 64
                ),
                "run_count": invariant_runtime.invariant_run_count,
                "variant_suite_root": invariant_runtime.variant_suite_root,
                "worker_artifact_sha256": (
                    invariant_runtime.worker_result_artifact_sha256
                ),
                "worker_peak_rss_bytes": invariant_runtime.worker_peak_rss_bytes,
                "wall_time_nanoseconds": (
                    invariant_runtime.invariance_wall_time_nanoseconds
                ),
                "unified_cgroup_membership": (
                    invariant_runtime.unified_cgroup_membership
                ),
                "unified_cgroup_memory_scope_root": (
                    invariant_runtime.unified_cgroup_memory_scope_root
                ),
                "unified_cgroup_leaf_device": (
                    invariant_runtime.unified_cgroup_leaf_device
                ),
                "unified_cgroup_leaf_inode": (
                    invariant_runtime.unified_cgroup_leaf_inode
                ),
                "physical_memory_current_bytes_at_completion": (
                    invariant_runtime.physical_memory_current_bytes_at_completion
                ),
                "physical_memory_peak_bytes_at_completion": (
                    invariant_runtime.physical_memory_peak_bytes_at_completion
                ),
                "physical_memory_max_bytes": invariant_runtime.physical_memory_max_bytes,
                "physical_memory_cap_bytes": (
                    invariant_runtime.physical_memory_cap_bytes
                ),
                "within_frozen_physical_peak_cap": (
                    invariant_runtime.within_frozen_physical_peak_cap
                ),
                "complete_runtime_projection": invariant_projection,
                "complete_runtime_projection_root": qualification.stable_hash(
                    invariant_projection,
                    prefix="phase2b_formal_wire_covert_invariant_runtime_v4_",
                ),
            },
            "statistics": {
                "resource_plan_root": statistics_runtime.resource_plan_root,
                "diagnostic_parent_plus_worker_historical_peak_sum_bytes": (
                    statistics_runtime.diagnostic_parent_plus_worker_historical_peak_sum_bytes
                ),
                "unified_cgroup_membership": (
                    statistics_runtime.unified_cgroup_membership
                ),
                "unified_cgroup_memory_scope_root": (
                    statistics_runtime.unified_cgroup_memory_scope_root
                ),
                "unified_cgroup_leaf_device": (
                    statistics_runtime.unified_cgroup_leaf_device
                ),
                "unified_cgroup_leaf_inode": (
                    statistics_runtime.unified_cgroup_leaf_inode
                ),
                "physical_memory_current_bytes_before_receipt_finalization": (
                    statistics_runtime.physical_memory_current_bytes_before_receipt_finalization
                ),
                "physical_memory_peak_bytes_before_receipt_finalization": (
                    statistics_runtime.physical_memory_peak_bytes_before_receipt_finalization
                ),
                "physical_memory_max_bytes": statistics_runtime.physical_memory_max_bytes,
                "physical_memory_cap_bytes": (
                    statistics_runtime.physical_memory_cap_bytes
                ),
                "external_supervisor_required": (
                    statistics_runtime.external_supervisor_required
                ),
                "pairwise_holm_decision_eligible": False,
                "pairwise_holm_family_size": 5 * plan.concrete_features,
                "permutation_schedule_root": (
                    "phase2b_formal_covert_permutation_schedule_" + "0" * 64
                ),
                "permutation_seed_commitment": auditor._permutation_seed_commitment(
                    bytes.fromhex(seed_reveal)
                ),
                "permutation_seed_reveal_hex": seed_reveal,
                "worker_artifact_sha256": list(
                    statistics_runtime.worker_result_artifact_sha256
                ),
                "worker_peak_rss_bytes": list(
                    statistics_runtime.worker_peak_rss_bytes
                ),
                "statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds": (
                    statistics_runtime.statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds
                ),
                "within_frozen_physical_peak_cap": (
                    statistics_runtime.within_frozen_physical_peak_cap
                ),
                "complete_runtime_projection": statistics_projection,
                "complete_runtime_projection_root": qualification.stable_hash(
                    statistics_projection,
                    prefix="phase2b_formal_wire_covert_statistics_runtime_v4_",
                ),
            },
            "receipt_issuance": {
                "receipt_id": receipt_id,
                "absolute_deadline_kind": issuance_runtime.absolute_deadline_kind,
                "diagnostic_worker_historical_peak_rss_bytes_sum": (
                    issuance_runtime.diagnostic_worker_historical_peak_rss_bytes_sum
                ),
                "diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization": (
                    issuance_runtime.diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization
                ),
                "diagnostic_parent_plus_worker_historical_peak_sum_bytes": (
                    issuance_runtime.diagnostic_parent_plus_worker_historical_peak_sum_bytes
                ),
                "unified_cgroup_membership": (
                    issuance_runtime.unified_cgroup_membership
                ),
                "unified_cgroup_memory_scope_root": (
                    issuance_runtime.unified_cgroup_memory_scope_root
                ),
                "unified_cgroup_leaf_device": (
                    issuance_runtime.unified_cgroup_leaf_device
                ),
                "unified_cgroup_leaf_inode": (
                    issuance_runtime.unified_cgroup_leaf_inode
                ),
                "physical_memory_current_bytes_before_receipt_finalization": (
                    issuance_runtime.physical_memory_current_bytes_before_receipt_finalization
                ),
                "physical_memory_peak_bytes_before_receipt_finalization": (
                    issuance_runtime.physical_memory_peak_bytes_before_receipt_finalization
                ),
                "physical_memory_max_bytes": issuance_runtime.physical_memory_max_bytes,
                "physical_memory_cap_bytes": issuance_runtime.physical_memory_cap_bytes,
                "receipt_finalization_complete_pending_validation_count": (
                    issuance_runtime.receipt_finalization_complete_pending_validation_count
                ),
                "receipt_finalization_locked_deadline_check_count": (
                    issuance_runtime.receipt_finalization_locked_deadline_check_count
                ),
                "receipt_finalization_locked_deadline_within_absolute_limit": (
                    issuance_runtime.receipt_finalization_locked_deadline_within_absolute_limit
                ),
                "receipt_finalization_locked_physical_peak_within_cap": (
                    issuance_runtime.receipt_finalization_locked_physical_peak_within_cap
                ),
                "complete_runtime_projection": issuance_projection,
                "complete_runtime_projection_root": qualification.stable_hash(
                    issuance_projection,
                    prefix=(
                        "phase2b_formal_wire_covert_receipt_issuance_runtime_v4_"
                    ),
                ),
            },
            "eligibility_quota": {"receipt_id": quota_receipt_id},
            "eligibility_quota_root": qualification.stable_hash(
                {"receipt_id": quota_receipt_id},
                prefix="phase2b_formal_wire_covert_eligibility_quota_v2_",
            ),
            "target_family_max_t_holm": _valid_target_family_rows(),
        },
    }
    measurement = {
        "observed_receipt_id": receipt_id,
        "observed_complete_resource_plan": plan_projection,
        "observed_complete_receipt_issuance_runtime": issuance_projection,
        "diagnostics": diagnostics,
    }
    return (
        plan_projection,
        invariant_projection,
        statistics_projection,
        issuance_projection,
        run,
        measurement,
    )


@pytest.mark.parametrize(
    "case",
    (
        "visible_rows",
        "visible_features",
        "adapter_pin",
        "resident_overbound",
        "compact_invariant_consumer_count",
        "compact_invariant_run_count",
        "compact_invariant_exact_comparisons",
        "compact_invariant_variant_root",
        "compact_invariant_worker_sha",
        "compact_invariant_worker_peak",
        "compact_invariant_wall",
        "compact_invariant_cap",
        "worker_count",
        "artifact_count",
        "artifact_sum",
        "parent_graph",
        "compact_worker_sha",
        "compact_worker_peak",
        "compact_statistics_wall",
        "compact_statistics_cap",
        "compact_statistics_conservative",
        "compact_statistics_external",
        "complete_statistics_scope",
        "complete_issuance_scope",
        "synchronized_invariant_scope",
        "compact_issuance_bool_alias",
        "diagnostics_scope",
        "receipt_physical_peak",
        "diagnostics_statistics_wall",
        "schedule_root",
        "execution_root",
        "seed_commitment",
        "pairwise_family",
        "target_family",
        "synchronized_fake_receipt_id",
        "synchronized_quota_own_id_and_root",
        "quota_root",
    ),
)
def test_executed_runtime_bundle_dto_valid_splice_matrix_fails_closed(
    case: str,
) -> None:
    _, _, _, _, original_run, original_measurement = (
        _valid_executed_runtime_bundle_projection()
    )
    run = json.loads(json.dumps(original_run))
    measurement = json.loads(json.dumps(original_measurement))
    audit_row = run["audit"]
    resource = audit_row["resource_plan"]
    invariant = audit_row["invariance"]
    statistics = audit_row["statistics"]
    issuance = audit_row["receipt_issuance"]
    invariant_complete = invariant["complete_runtime_projection"]
    statistics_complete = statistics["complete_runtime_projection"]
    issuance_complete = issuance["complete_runtime_projection"]
    qualification._validate_executed_runtime_bundle_v2(
        resource_plan=resource,
        statistics_runtime=statistics_complete,
        receipt_issuance_runtime=issuance_complete,
        diagnostics=run["diagnostics"],
        run=run,
        measurement=measurement,
        expected_mode="KNOWN_NULL",
        visible_audited_rows=resource["rows"],
        visible_concrete_features=resource["concrete_features"],
        context="unit valid bundle",
    )
    assert (
        audit_row["eligibility_quota"]["receipt_id"]
        != audit_row["receipt_id"]
    )

    def refresh_invariant_root() -> None:
        invariant["complete_runtime_projection_root"] = qualification.stable_hash(
            invariant_complete,
            prefix="phase2b_formal_wire_covert_invariant_runtime_v4_",
        )

    def refresh_statistics_root() -> None:
        statistics["complete_runtime_projection_root"] = qualification.stable_hash(
            statistics_complete,
            prefix="phase2b_formal_wire_covert_statistics_runtime_v4_",
        )

    def refresh_issuance_root() -> None:
        issuance["complete_runtime_projection_root"] = qualification.stable_hash(
            issuance_complete,
            prefix="phase2b_formal_wire_covert_receipt_issuance_runtime_v4_",
        )

    if case == "visible_rows":
        audit_row["audited_rows"] += 1
    elif case == "visible_features":
        audit_row["concrete_features"] += 1
    elif case == "adapter_pin":
        run["wire"]["adapter_execution_pin_root"] = (
            "phase2b_formal_covert_adapter_execution_pins_" + "1" * 64
        )
    elif case == "resident_overbound":
        invariant_complete[
            "variant_suite_resident_object_graph_bytes_observed"
        ] = resource["invariance_variant_suite_resident_bytes_bound"] + 1
        refresh_invariant_root()
    elif case == "compact_invariant_consumer_count":
        invariant["consumer_decode_count"] += 80
    elif case == "compact_invariant_run_count":
        invariant["run_count"] += 1
    elif case == "compact_invariant_exact_comparisons":
        invariant["exact_comparisons"] += 400
    elif case == "compact_invariant_variant_root":
        invariant["variant_suite_root"] = (
            "phase2b_formal_variant_suite_" + "1" * 64
        )
    elif case == "compact_invariant_worker_sha":
        invariant["worker_artifact_sha256"] = "sha256:" + "1" * 64
    elif case == "compact_invariant_worker_peak":
        invariant["worker_peak_rss_bytes"] += 1
    elif case == "compact_invariant_wall":
        invariant["wall_time_nanoseconds"] += 1
    elif case == "compact_invariant_cap":
        invariant["within_frozen_physical_peak_cap"] = False
    elif case == "worker_count":
        statistics_complete["worker_peak_rss_bytes"].append(0)
        statistics["worker_peak_rss_bytes"].append(0)
        run["diagnostics"]["statistics_worker_historical_peak_rss_kib"].append(0)
        measurement["diagnostics"] = json.loads(
            json.dumps(run["diagnostics"])
        )
        refresh_statistics_root()
    elif case == "artifact_count":
        statistics_complete["worker_result_artifact_bytes"] = (
            statistics_complete["worker_result_artifact_bytes"][:-1]
        )
        statistics_complete["worker_result_artifact_sha256"] = (
            statistics_complete["worker_result_artifact_sha256"][:-1]
        )
        statistics["worker_artifact_sha256"] = list(
            statistics_complete["worker_result_artifact_sha256"]
        )
        refresh_statistics_root()
    elif case == "artifact_sum":
        statistics_complete["worker_result_artifact_bytes"] = [
            resource["worker_result_artifact_bytes_bound"] + 1,
            1,
            1,
            1,
            1,
        ]
        refresh_statistics_root()
    elif case == "parent_graph":
        statistics_complete["parent_result_object_graph_bytes_observed"] = (
            resource["parent_result_working_bytes_bound"] + 1
        )
        refresh_statistics_root()
    elif case == "compact_worker_sha":
        statistics["worker_artifact_sha256"][0] = "sha256:" + "1" * 64
    elif case == "compact_worker_peak":
        statistics["worker_peak_rss_bytes"][0] += 1
    elif case == "compact_statistics_wall":
        statistics[
            "statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds"
        ] += 1
    elif case == "compact_statistics_cap":
        statistics["within_frozen_physical_peak_cap"] = False
    elif case == "compact_statistics_conservative":
        statistics["diagnostic_parent_plus_worker_historical_peak_sum_bytes"] += 1
    elif case == "compact_statistics_external":
        statistics["external_supervisor_required"] = False
    elif case == "complete_statistics_scope":
        statistics_complete["unified_cgroup_leaf_inode"] += 1
        refresh_statistics_root()
    elif case == "complete_issuance_scope":
        issuance_complete["unified_cgroup_leaf_device"] += 1
        refresh_issuance_root()
    elif case == "synchronized_invariant_scope":
        new_device = invariant_complete["unified_cgroup_leaf_device"] + 1
        new_scope = qualification.stable_hash(
            {
                "leaf_device": new_device,
                "leaf_inode": invariant_complete["unified_cgroup_leaf_inode"],
                "unified_membership": invariant_complete[
                    "unified_cgroup_membership"
                ],
            },
            prefix="phase2b_formal_covert_unified_cgroup_memory_scope_v1_",
        )
        invariant_complete["unified_cgroup_leaf_device"] = new_device
        invariant_complete["unified_cgroup_memory_scope_root"] = new_scope
        invariant["unified_cgroup_leaf_device"] = new_device
        invariant["unified_cgroup_memory_scope_root"] = new_scope
        refresh_invariant_root()
    elif case == "compact_issuance_bool_alias":
        issuance["receipt_finalization_complete_pending_validation_count"] = True
    elif case == "diagnostics_scope":
        run["diagnostics"]["unified_cgroup_memory_scope_root"] = (
            "phase2b_formal_covert_unified_cgroup_memory_scope_v1_" + "1" * 64
        )
        measurement["diagnostics"] = dict(run["diagnostics"])
    elif case == "receipt_physical_peak":
        issuance_complete[
            "physical_memory_peak_bytes_before_receipt_finalization"
        ] += 1
        refresh_issuance_root()
    elif case == "diagnostics_statistics_wall":
        run["diagnostics"]["statistics_wall_time_ns"] += 1
        measurement["diagnostics"] = dict(run["diagnostics"])
    elif case == "schedule_root":
        statistics["permutation_schedule_root"] = "schedule"
    elif case == "execution_root":
        invariant["execution_root"] = "execution"
    elif case == "seed_commitment":
        statistics["permutation_seed_commitment"] = "sha256:" + "1" * 64
    elif case == "pairwise_family":
        statistics["pairwise_holm_family_size"] += 5
    elif case == "target_family":
        audit_row["target_family_max_t_holm"] = audit_row[
            "target_family_max_t_holm"
        ][:-1]
    elif case == "synchronized_fake_receipt_id":
        fake_id = "phase2b_formal_covert_audit_transcript_fake"
        audit_row["receipt_id"] = fake_id
        issuance["receipt_id"] = fake_id
        measurement["observed_receipt_id"] = fake_id
    elif case == "synchronized_quota_own_id_and_root":
        audit_row["eligibility_quota"]["receipt_id"] = audit_row["receipt_id"]
        audit_row["eligibility_quota_root"] = qualification.stable_hash(
            audit_row["eligibility_quota"],
            prefix="phase2b_formal_wire_covert_eligibility_quota_v2_",
        )
    elif case == "quota_root":
        audit_row["eligibility_quota_root"] = (
            "phase2b_formal_wire_covert_eligibility_quota_v2_" + "f" * 64
        )
    else:  # pragma: no cover - parameter list is closed above
        raise AssertionError(case)

    with pytest.raises(ValueError):
        qualification._validate_executed_runtime_bundle_v2(
            resource_plan=resource,
            statistics_runtime=statistics_complete,
            receipt_issuance_runtime=issuance_complete,
            diagnostics=run["diagnostics"],
            run=run,
            measurement=measurement,
            expected_mode="KNOWN_NULL",
            visible_audited_rows=resource["rows"],
            visible_concrete_features=resource["concrete_features"],
            context=f"unit bundle {case}",
        )


def test_candidate_success_projects_and_releases_both_receipts_over_registry_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live: set[int] = set()
    validations: list[int] = []
    plan, statistics_runtime, issuance_runtime = _valid_candidate_runtime_bundle()
    frozen_capacity_row = (
        qualification._fixture_feature_capacity_projection_row_v2
    )

    def unit_capacity_row(mode: str, rows: int) -> dict[str, object]:
        row = dict(frozen_capacity_row(mode, rows))
        if rows == qualification.EXACT_AUTHORITY_ROWS:
            row["projected_concrete_feature_count"] = plan.concrete_features
        return row

    monkeypatch.setattr(
        qualification,
        "_fixture_feature_capacity_projection_row_v2",
        unit_capacity_row,
    )

    class FakeReceipt:
        def __init__(self, receipt_id: str) -> None:
            self.receipt_id = receipt_id
            self.audited_row_count = plan.rows
            self.feature_manifest = SimpleNamespace(
                manifest_root="feature-root",
                deduplicated_feature_count=plan.concrete_features,
                raw_semantic_feature_count=plan.concrete_features,
            )
            self.resource_plan = plan
            self.statistics_runtime_evidence = statistics_runtime
            self.receipt_issuance_runtime_evidence = issuance_runtime
            live.add(id(self))

    def validate(receipt: FakeReceipt) -> str:
        assert id(receipt) in live
        validations.append(id(receipt))
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
            mode=mode,
            wall_time_ns=1,
            parent_historical_peak_rss_before_receipt_finalization_kib=0,
            statistics_worker_historical_peak_rss_kib=tuple(
                0 for _ in statistics_runtime.worker_peak_rss_bytes
            ),
            parent_plus_worker_historical_peak_rss_kib=0,
            unified_cgroup_membership=(
                statistics_runtime.unified_cgroup_membership
            ),
            unified_cgroup_memory_scope_root=(
                statistics_runtime.unified_cgroup_memory_scope_root
            ),
            unified_cgroup_leaf_device=statistics_runtime.unified_cgroup_leaf_device,
            unified_cgroup_leaf_inode=statistics_runtime.unified_cgroup_leaf_inode,
            physical_memory_current_before_receipt_finalization_bytes=(
                statistics_runtime.physical_memory_current_bytes_before_receipt_finalization
            ),
            physical_memory_peak_before_receipt_finalization_bytes=(
                statistics_runtime.physical_memory_peak_bytes_before_receipt_finalization
            ),
            physical_memory_max_bytes=statistics_runtime.physical_memory_max_bytes,
            physical_memory_cap_bytes=statistics_runtime.physical_memory_cap_bytes,
            statistics_wall_time_ns=1,
            host_available_memory_bytes_at_preflight=(
                statistics_runtime.host_available_memory_bytes_at_preflight
            ),
            recognizer_adapter_invocations=(
                qualification.EXACT_RECOGNIZER_INVOCATIONS
            ),
            minimum_rows_per_invocation=qualification.EXACT_AUTHORITY_ROWS,
            maximum_rows_per_invocation=qualification.EXACT_AUTHORITY_ROWS,
        )

    for index in range(auditor.MAXIMUM_IN_PROCESS_ISSUED_RECEIPTS + 1):
        assert len(live) + 2 <= auditor.MAXIMUM_IN_PROCESS_ISSUED_RECEIPTS
        positive_id = (
            "phase2b_formal_covert_audit_transcript_"
            + f"{2 * index:064x}"
        )
        negative_id = (
            "phase2b_formal_covert_audit_transcript_"
            + f"{2 * index + 1:064x}"
        )
        positive_receipt = FakeReceipt(positive_id)
        negative_receipt = FakeReceipt(negative_id)
        positive = SimpleNamespace(
            mode="KNOWN_NULL",
            label_factorial_proof_root="phase2b_formal_wire_covert_label_factorial_v2_x",
            audit_receipt=positive_receipt,
            validated_receipt_id=positive_receipt.receipt_id,
            batch_id="batch",
            transform_catalog_root="catalog",
            diagnostics=diagnostics("KNOWN_NULL"),
            operation_counts=tuple(
                (operation, qualification.EXACT_ROWS_PER_TRANSFORM_OPERATION)
                for operation in qualification.EXACT_TRANSFORM_OPERATIONS
            ),
            namespace_uuid_leaf_paths=1,
            channel_leaf_paths=1,
            exact_owned_channel_only_leaf_paths=(
                qualification._FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"][
                    "KNOWN_NULL"
                ]["channel_only_owned_union_path_count"]
            ),
            channel_only_owner_union_root=(
                qualification._FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"][
                    "KNOWN_NULL"
                ]["channel_only_owned_union_root"]
            ),
            full_raw_byte_categorical_hypotheses=1,
        )
        negative = SimpleNamespace(
            mode="KNOWN_BINDING_LEAK",
            label_factorial_proof_root=positive.label_factorial_proof_root,
            audit_receipt=negative_receipt,
            validated_receipt_id=negative_receipt.receipt_id,
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
        assert candidate.positive_audit_receipt_id == positive_id
        assert candidate.qualification_receipt_id.startswith(
            "phase2b_formal_wire_covert_qualification_receipt_v6_"
        )
        assert not hasattr(candidate, "positive_audit_receipt")
        projection = candidate._receipt_projection()
        assert projection["projection_only_boundary"] == (
            qualification._TASK4_PROJECTION_ONLY_BOUNDARY_V2
        )
        assert projection["positive"]["resource_plan_projection"] == (
            candidate.positive_resource_plan_projection
        )
        assert projection["positive"]["resource_plan_projection_root"] == (
            candidate.positive_resource_plan_projection_root
        )
        assert projection["negative"]["resource_plan_projection"] == (
            candidate.negative_resource_plan_projection
        )
        assert projection["negative"]["resource_plan_projection_root"] == (
            candidate.negative_resource_plan_projection_root
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
        assert validations == []
        if index == 0:
            def refresh_candidate_id() -> None:
                object.__setattr__(
                    candidate,
                    "qualification_receipt_id",
                    qualification.QUALIFICATION_RECEIPT_PREFIX
                    + hashlib.sha256(
                        qualification.QUALIFICATION_RECEIPT_DOMAIN
                        + qualification.canonicalize_rfc8785_v1(
                            candidate._receipt_projection()
                        )
                    ).hexdigest(),
                )

            alternate_plan = replace(
                plan,
                effective_cpu_capacity_at_preflight=(
                    plan.effective_cpu_capacity_at_preflight + 1
                ),
            )
            alternate_plan.__post_init__()
            original_plan = candidate.negative_resource_plan_projection
            original_plan_root = candidate.negative_resource_plan_projection_root
            alternate_plan_projection = qualification._closed_primitive_v2(
                alternate_plan
            )
            object.__setattr__(
                candidate,
                "negative_resource_plan_projection",
                alternate_plan_projection,
            )
            object.__setattr__(
                candidate,
                "negative_resource_plan_projection_root",
                qualification.stable_hash(
                    alternate_plan_projection,
                    prefix="phase2b_formal_wire_covert_resource_plan_v3_",
                ),
            )
            refresh_candidate_id()
            with pytest.raises(ValueError):
                candidate._validate()
            object.__setattr__(
                candidate, "negative_resource_plan_projection", original_plan
            )
            object.__setattr__(
                candidate,
                "negative_resource_plan_projection_root",
                original_plan_root,
            )
            refresh_candidate_id()

            original_positive_diagnostics = candidate.positive_diagnostics
            object.__setattr__(
                candidate,
                "positive_diagnostics",
                replace(
                    original_positive_diagnostics,
                    mode="KNOWN_BINDING_LEAK",
                ),
            )
            refresh_candidate_id()
            with pytest.raises(ValueError):
                candidate._validate()
            object.__setattr__(
                candidate,
                "positive_diagnostics",
                original_positive_diagnostics,
            )
            refresh_candidate_id()

            inadmissible_plan, inadmissible_statistics = (
                _inadmissible_candidate_runtime_bundle(
                    plan,
                    statistics_runtime,
                )
            )
            inadmissible_plan_projection = qualification._closed_primitive_v2(
                inadmissible_plan
            )
            inadmissible_statistics_projection = (
                qualification._closed_primitive_v2(inadmissible_statistics)
            )
            qualification._validate_complete_resource_plan_projection_v2(
                inadmissible_plan_projection,
                context="candidate inadmissible KAT",
            )
            qualification._validate_complete_statistics_runtime_projection_v2(
                inadmissible_statistics_projection,
                inadmissible_plan_projection,
                context="candidate inadmissible KAT",
            )
            assert inadmissible_plan_projection["admissible"] is False
            for branch in ("positive", "negative"):
                plan_field = f"{branch}_resource_plan_projection"
                plan_root_field = f"{branch}_resource_plan_projection_root"
                statistics_field = f"{branch}_statistics_runtime_projection"
                statistics_root_field = (
                    f"{branch}_statistics_runtime_projection_root"
                )
                original_branch_plan = getattr(candidate, plan_field)
                original_branch_plan_root = getattr(candidate, plan_root_field)
                original_branch_statistics = getattr(candidate, statistics_field)
                original_branch_statistics_root = getattr(
                    candidate,
                    statistics_root_field,
                )
                object.__setattr__(
                    candidate,
                    plan_field,
                    inadmissible_plan_projection,
                )
                object.__setattr__(
                    candidate,
                    plan_root_field,
                    qualification.stable_hash(
                        inadmissible_plan_projection,
                        prefix="phase2b_formal_wire_covert_resource_plan_v3_",
                    ),
                )
                object.__setattr__(
                    candidate,
                    statistics_field,
                    inadmissible_statistics_projection,
                )
                object.__setattr__(
                    candidate,
                    statistics_root_field,
                    qualification.stable_hash(
                        inadmissible_statistics_projection,
                        prefix=(
                            "phase2b_formal_wire_covert_statistics_runtime_v4_"
                        ),
                    ),
                )
                refresh_candidate_id()
                with pytest.raises(ValueError):
                    candidate._validate()
                object.__setattr__(candidate, plan_field, original_branch_plan)
                object.__setattr__(
                    candidate,
                    plan_root_field,
                    original_branch_plan_root,
                )
                object.__setattr__(
                    candidate,
                    statistics_field,
                    original_branch_statistics,
                )
                object.__setattr__(
                    candidate,
                    statistics_root_field,
                    original_branch_statistics_root,
                )
                refresh_candidate_id()
            candidate._validate()

            changed_worker_peaks = (
                1_024,
                *tuple(
                    0
                    for _ in range(
                        len(statistics_runtime.worker_peak_rss_bytes) - 1
                    )
                ),
            )
            changed_statistics = replace(
                statistics_runtime,
                worker_peak_rss_bytes=changed_worker_peaks,
                diagnostic_parent_plus_worker_historical_peak_sum_bytes=1_024,
            )
            changed_statistics_projection = qualification._closed_primitive_v2(
                changed_statistics
            )
            original_statistics = candidate.positive_statistics_runtime_projection
            original_statistics_root = (
                candidate.positive_statistics_runtime_projection_root
            )
            original_diagnostics = candidate.positive_diagnostics
            object.__setattr__(
                candidate,
                "positive_statistics_runtime_projection",
                changed_statistics_projection,
            )
            object.__setattr__(
                candidate,
                "positive_statistics_runtime_projection_root",
                qualification.stable_hash(
                    changed_statistics_projection,
                    prefix="phase2b_formal_wire_covert_statistics_runtime_v4_",
                ),
            )
            object.__setattr__(
                candidate,
                "positive_diagnostics",
                qualification.AuditRunDiagnosticsV2(
                    mode="KNOWN_NULL",
                    wall_time_ns=1,
                    parent_historical_peak_rss_before_receipt_finalization_kib=0,
                    statistics_worker_historical_peak_rss_kib=tuple(
                        value // 1_024 for value in changed_worker_peaks
                    ),
                    parent_plus_worker_historical_peak_rss_kib=1,
                    unified_cgroup_membership=(
                        statistics_runtime.unified_cgroup_membership
                    ),
                    unified_cgroup_memory_scope_root=(
                        statistics_runtime.unified_cgroup_memory_scope_root
                    ),
                    unified_cgroup_leaf_device=(
                        statistics_runtime.unified_cgroup_leaf_device
                    ),
                    unified_cgroup_leaf_inode=(
                        statistics_runtime.unified_cgroup_leaf_inode
                    ),
                    physical_memory_current_before_receipt_finalization_bytes=(
                        statistics_runtime.physical_memory_current_bytes_before_receipt_finalization
                    ),
                    physical_memory_peak_before_receipt_finalization_bytes=(
                        statistics_runtime.physical_memory_peak_bytes_before_receipt_finalization
                    ),
                    physical_memory_max_bytes=(
                        statistics_runtime.physical_memory_max_bytes
                    ),
                    physical_memory_cap_bytes=(
                        statistics_runtime.physical_memory_cap_bytes
                    ),
                    statistics_wall_time_ns=1,
                    host_available_memory_bytes_at_preflight=(
                        statistics_runtime.host_available_memory_bytes_at_preflight
                    ),
                    recognizer_adapter_invocations=(
                        qualification.EXACT_RECOGNIZER_INVOCATIONS
                    ),
                    minimum_rows_per_invocation=(
                        qualification.EXACT_AUTHORITY_ROWS
                    ),
                    maximum_rows_per_invocation=(
                        qualification.EXACT_AUTHORITY_ROWS
                    ),
                ),
            )
            refresh_candidate_id()
            with pytest.raises(ValueError):
                candidate._validate()
            object.__setattr__(
                candidate,
                "positive_statistics_runtime_projection",
                original_statistics,
            )
            object.__setattr__(
                candidate,
                "positive_statistics_runtime_projection_root",
                original_statistics_root,
            )
            object.__setattr__(
                candidate, "positive_diagnostics", original_diagnostics
            )

            changed_issuance = replace(
                issuance_runtime,
                diagnostic_worker_historical_peak_rss_bytes_sum=1_024,
                diagnostic_parent_plus_worker_historical_peak_sum_bytes=1_024,
            )
            changed_issuance_projection = qualification._closed_primitive_v2(
                changed_issuance
            )
            original_issuance = (
                candidate.positive_receipt_issuance_runtime_projection
            )
            original_issuance_root = (
                candidate.positive_receipt_issuance_runtime_projection_root
            )
            object.__setattr__(
                candidate,
                "positive_receipt_issuance_runtime_projection",
                changed_issuance_projection,
            )
            object.__setattr__(
                candidate,
                "positive_receipt_issuance_runtime_projection_root",
                qualification.stable_hash(
                    changed_issuance_projection,
                    prefix=(
                        "phase2b_formal_wire_covert_receipt_issuance_runtime_v4_"
                    ),
                ),
            )
            refresh_candidate_id()
            with pytest.raises(ValueError):
                candidate._validate()
            object.__setattr__(
                candidate,
                "positive_receipt_issuance_runtime_projection",
                original_issuance,
            )
            object.__setattr__(
                candidate,
                "positive_receipt_issuance_runtime_projection_root",
                original_issuance_root,
            )
            refresh_candidate_id()
            candidate._validate()

            positive_statistics = candidate.positive_statistics_runtime_projection
            positive_statistics_root = (
                candidate.positive_statistics_runtime_projection_root
            )
            original_physical_peak = positive_statistics[
                "physical_memory_peak_bytes_before_receipt_finalization"
            ]
            positive_statistics[
                "physical_memory_peak_bytes_before_receipt_finalization"
            ] = original_physical_peak + 1
            object.__setattr__(
                candidate,
                "positive_statistics_runtime_projection_root",
                qualification.stable_hash(
                    positive_statistics,
                    prefix="phase2b_formal_wire_covert_statistics_runtime_v4_",
                ),
            )
            refresh_candidate_id()
            with pytest.raises(ValueError):
                candidate._validate()
            positive_statistics[
                "physical_memory_peak_bytes_before_receipt_finalization"
            ] = original_physical_peak
            object.__setattr__(
                candidate,
                "positive_statistics_runtime_projection_root",
                positive_statistics_root,
            )
            refresh_candidate_id()
            candidate._validate()

            negative_statistics = candidate.negative_statistics_runtime_projection
            negative_issuance = (
                candidate.negative_receipt_issuance_runtime_projection
            )
            negative_statistics_root = (
                candidate.negative_statistics_runtime_projection_root
            )
            negative_issuance_root = (
                candidate.negative_receipt_issuance_runtime_projection_root
            )
            negative_diagnostics = candidate.negative_diagnostics
            original_negative_peak = negative_statistics[
                "physical_memory_peak_bytes_before_receipt_finalization"
            ]
            lower_negative_peak = original_negative_peak - 1
            assert lower_negative_peak >= negative_statistics[
                "physical_memory_current_bytes_before_receipt_finalization"
            ]
            negative_statistics[
                "physical_memory_peak_bytes_before_receipt_finalization"
            ] = lower_negative_peak
            negative_issuance[
                "physical_memory_peak_bytes_before_receipt_finalization"
            ] = lower_negative_peak
            object.__setattr__(
                candidate,
                "negative_statistics_runtime_projection_root",
                qualification.stable_hash(
                    negative_statistics,
                    prefix="phase2b_formal_wire_covert_statistics_runtime_v4_",
                ),
            )
            object.__setattr__(
                candidate,
                "negative_receipt_issuance_runtime_projection_root",
                qualification.stable_hash(
                    negative_issuance,
                    prefix=(
                        "phase2b_formal_wire_covert_receipt_issuance_runtime_v4_"
                    ),
                ),
            )
            object.__setattr__(
                candidate,
                "negative_diagnostics",
                replace(
                    negative_diagnostics,
                    physical_memory_peak_before_receipt_finalization_bytes=(
                        lower_negative_peak
                    ),
                ),
            )
            refresh_candidate_id()
            with pytest.raises(ValueError, match="changed physical memory scope"):
                candidate._validate()
            negative_statistics[
                "physical_memory_peak_bytes_before_receipt_finalization"
            ] = original_negative_peak
            negative_issuance[
                "physical_memory_peak_bytes_before_receipt_finalization"
            ] = original_negative_peak
            object.__setattr__(
                candidate,
                "negative_statistics_runtime_projection_root",
                negative_statistics_root,
            )
            object.__setattr__(
                candidate,
                "negative_receipt_issuance_runtime_projection_root",
                negative_issuance_root,
            )
            object.__setattr__(
                candidate,
                "negative_diagnostics",
                negative_diagnostics,
            )
            refresh_candidate_id()
            candidate._validate()

            original_device = negative_statistics["unified_cgroup_leaf_device"]
            original_scope = negative_statistics[
                "unified_cgroup_memory_scope_root"
            ]
            alternate_device = original_device + 1
            alternate_scope = qualification.stable_hash(
                {
                    "leaf_device": alternate_device,
                    "leaf_inode": negative_statistics[
                        "unified_cgroup_leaf_inode"
                    ],
                    "unified_membership": negative_statistics[
                        "unified_cgroup_membership"
                    ],
                },
                prefix="phase2b_formal_covert_unified_cgroup_memory_scope_v1_",
            )
            for projection in (negative_statistics, negative_issuance):
                projection["unified_cgroup_leaf_device"] = alternate_device
                projection["unified_cgroup_memory_scope_root"] = alternate_scope
            object.__setattr__(
                candidate,
                "negative_statistics_runtime_projection_root",
                qualification.stable_hash(
                    negative_statistics,
                    prefix="phase2b_formal_wire_covert_statistics_runtime_v4_",
                ),
            )
            object.__setattr__(
                candidate,
                "negative_receipt_issuance_runtime_projection_root",
                qualification.stable_hash(
                    negative_issuance,
                    prefix=(
                        "phase2b_formal_wire_covert_receipt_issuance_runtime_v4_"
                    ),
                ),
            )
            object.__setattr__(
                candidate,
                "negative_diagnostics",
                replace(
                    negative_diagnostics,
                    unified_cgroup_leaf_device=alternate_device,
                    unified_cgroup_memory_scope_root=alternate_scope,
                ),
            )
            refresh_candidate_id()
            with pytest.raises(ValueError, match="changed physical memory scope"):
                candidate._validate()
            for projection in (negative_statistics, negative_issuance):
                projection["unified_cgroup_leaf_device"] = original_device
                projection["unified_cgroup_memory_scope_root"] = original_scope
            object.__setattr__(
                candidate,
                "negative_statistics_runtime_projection_root",
                negative_statistics_root,
            )
            object.__setattr__(
                candidate,
                "negative_receipt_issuance_runtime_projection_root",
                negative_issuance_root,
            )
            object.__setattr__(
                candidate,
                "negative_diagnostics",
                negative_diagnostics,
            )
            refresh_candidate_id()
            candidate._validate()

            original_negative_id = candidate.negative_audit_receipt_id
            object.__setattr__(
                candidate,
                "negative_audit_receipt_id",
                candidate.positive_audit_receipt_id,
            )
            refresh_candidate_id()
            with pytest.raises(ValueError, match="branch-distinct"):
                candidate._validate()
            object.__setattr__(
                candidate, "negative_audit_receipt_id", original_negative_id
            )
            original_positive_id = candidate.positive_audit_receipt_id
            object.__setattr__(
                candidate,
                "positive_audit_receipt_id",
                "phase2b_formal_covert_audit_transcript_fake",
            )
            refresh_candidate_id()
            with pytest.raises(ValueError, match="exact Auditor receipt ID"):
                candidate._validate()
            object.__setattr__(
                candidate, "positive_audit_receipt_id", original_positive_id
            )
            refresh_candidate_id()
            candidate._validate()

            original_worker_sum = candidate.positive_receipt_issuance_runtime_projection[
                "diagnostic_worker_historical_peak_rss_bytes_sum"
            ]
            candidate.positive_receipt_issuance_runtime_projection[
                "diagnostic_worker_historical_peak_rss_bytes_sum"
            ] = original_worker_sum + 1
            with pytest.raises(ValueError):
                candidate._validate()
            candidate.positive_receipt_issuance_runtime_projection[
                "diagnostic_worker_historical_peak_rss_bytes_sum"
            ] = original_worker_sum
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
    requested_memory_max = 24 * 1_024 * 1_024 * 1_024
    rows = {
        "cgroup.procs": str(os.getpid()),
        "cgroup.stat": "nr_descendants 0\nnr_dying_descendants 0\n",
        "memory.max": str(requested_memory_max),
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
    opened = target.stat(follow_symlinks=False)
    assert evidence["memory_max"] == requested_memory_max
    assert evidence["memory_peak"] == 2_048
    assert evidence["unified_cgroup_leaf_device"] == opened.st_dev
    assert evidence["unified_cgroup_leaf_inode"] == opened.st_ino
    assert evidence["unified_cgroup_memory_scope_root"] == qualification.stable_hash(
        {
            "leaf_device": opened.st_dev,
            "leaf_inode": opened.st_ino,
            "unified_membership": "/unit.service",
        },
        prefix="phase2b_formal_covert_unified_cgroup_memory_scope_v1_",
    )
    assert evidence["pids_max"] == 64
    assert evidence["cgroup_procs"] == [os.getpid()]
    assert evidence["cgroup_stat"]["nr_descendants"] == 0
    assert evidence["cpu_max"] is None
    for malformed in ("01024", " 1024\n", "1024\n\n"):
        (target / "memory.current").write_text(malformed, encoding="ascii")
        with pytest.raises(ValueError, match="canonical"):
            qualification.inspect_current_live_cgroup_v2(
                proc_root=proc, cgroup_root=cgroup
            )
    (target / "memory.current").write_text("1" * 4_097, encoding="ascii")
    with pytest.raises(ValueError, match="outside bounds"):
        qualification.inspect_current_live_cgroup_v2(
            proc_root=proc, cgroup_root=cgroup
        )
    (target / "memory.current").write_text("1024", encoding="ascii")
    (target / "memory.events").write_text("", encoding="ascii")
    with pytest.raises(ValueError, match="outside bounds"):
        qualification.inspect_current_live_cgroup_v2(
            proc_root=proc, cgroup_root=cgroup
        )
    (target / "memory.events").write_text("oom 0\n", encoding="ascii")
    with pytest.raises(ValueError, match="structural snapshot drifted"):
        qualification.inspect_current_live_cgroup_v2(
            proc_root=proc, cgroup_root=cgroup
        )
    (target / "memory.events").write_text("oom  0\noom_kill 0\n", encoding="ascii")
    with pytest.raises(ValueError, match="malformed"):
        qualification.inspect_current_live_cgroup_v2(
            proc_root=proc, cgroup_root=cgroup
        )
    (target / "memory.events").write_text(rows["memory.events"], encoding="ascii")
    (target / "memory.peak").write_text(
        str(auditor.MAXIMUM_ESTIMATED_RSS_BYTES + 1), encoding="ascii"
    )
    over_cap = qualification.inspect_current_live_cgroup_v2(
        proc_root=proc, cgroup_root=cgroup
    )
    assert over_cap["memory_peak"] == auditor.MAXIMUM_ESTIMATED_RSS_BYTES + 1
    (target / "memory.max").write_text(
        str(auditor.MAXIMUM_ESTIMATED_RSS_BYTES), encoding="ascii"
    )
    (target / "memory.current").write_text(
        str(auditor.MAXIMUM_ESTIMATED_RSS_BYTES + 1), encoding="ascii"
    )
    (target / "memory.peak").write_text(
        str(auditor.MAXIMUM_ESTIMATED_RSS_BYTES + 2), encoding="ascii"
    )
    (target / "memory.events").write_text(
        "low 0\nhigh 0\nmax 1\noom 0\noom_kill 0\n", encoding="ascii"
    )
    transient_overage = qualification.inspect_current_live_cgroup_v2(
        proc_root=proc, cgroup_root=cgroup
    )
    assert transient_overage["memory_current"] == (
        auditor.MAXIMUM_ESTIMATED_RSS_BYTES + 1
    )
    assert transient_overage["memory_peak"] == (
        auditor.MAXIMUM_ESTIMATED_RSS_BYTES + 2
    )
    assert transient_overage["memory_events"]["max"] == 1
    (target / "memory.events").write_text(rows["memory.events"], encoding="ascii")
    (target / "memory.max").write_text(str(requested_memory_max), encoding="ascii")
    (target / "memory.peak").write_text("2048", encoding="ascii")
    (target / "memory.current").write_text("2049", encoding="ascii")
    with pytest.raises(ValueError, match="structural snapshot drifted"):
        qualification.inspect_current_live_cgroup_v2(
            proc_root=proc, cgroup_root=cgroup
        )
    (target / "memory.current").write_text("1024", encoding="ascii")
    (target / "memory.max").write_text("max\n", encoding="ascii")
    with pytest.raises(ValueError, match="finite"):
        qualification.inspect_current_live_cgroup_v2(
            proc_root=proc, cgroup_root=cgroup
        )


def _post_output_gate_fixture() -> tuple[
    qualification.Phase2BFormalWireCovertAttemptManifestV2,
    dict[str, object],
    dict[str, object],
]:
    attempt = _attempt()
    membership = "/test.slice/post-output.scope"
    device = 11
    inode = 22
    snapshot = {
        "cgroup_path": membership,
        "cgroup_procs": [os.getpid()],
        "cgroup_stat": {"nr_descendants": 0, "nr_dying_descendants": 0},
        "cpu_affinity": [0],
        "cpu_capacity": 1,
        "cpu_max": None,
        "memory_current": 1_024,
        "memory_events": {
            "high": 0,
            "low": 0,
            "max": 0,
            "oom": 0,
            "oom_kill": 0,
        },
        "memory_max": attempt.requested_cgroup_memory_max_bytes,
        "memory_peak": 2_048,
        "pids_current": 1,
        "pids_events": {"max": 0},
        "pids_max": attempt.requested_cgroup_pids_max,
        "unified_cgroup_leaf_device": device,
        "unified_cgroup_leaf_inode": inode,
        "unified_cgroup_memory_scope_root": qualification.stable_hash(
            {
                "leaf_device": device,
                "leaf_inode": inode,
                "unified_membership": membership,
            },
            prefix="phase2b_formal_covert_unified_cgroup_memory_scope_v1_",
        ),
    }
    runtime_exit = {
        "cgroup": json.loads(json.dumps(snapshot)),
    }
    runtime_exit["cgroup"]["memory_peak"] = 1_024
    runtime_exit["cgroup"]["memory_current"] = 512
    return attempt, runtime_exit, snapshot


def _expected_cgroup_physical_resource_failure(
    *,
    prefix: str,
    stage: str,
    snapshot: dict[str, object],
    cap: int,
    reserve: int,
) -> str:
    memory_events = snapshot["memory_events"]
    pids_events = snapshot["pids_events"]
    assert type(memory_events) is dict
    assert type(pids_events) is dict
    peak = snapshot["memory_peak"]
    assert type(peak) is int
    return (
        f"{prefix}; stage={stage};"
        f"scope={snapshot['unified_cgroup_memory_scope_root']};"
        f"current={snapshot['memory_current']};peak={peak};"
        f"memory.max={snapshot['memory_max']};cap={cap};reserve={reserve};"
        f"overage={max(0, peak - (cap - reserve))};"
        f"memory.events.max={memory_events['max']};"
        f"memory.events.oom={memory_events['oom']};"
        f"memory.events.oom_kill={memory_events['oom_kill']};"
        f"pids.max={snapshot['pids_max']};"
        f"pids.events.max={pids_events['max']}"
    )


@pytest.mark.parametrize("event_only", (False, True), ids=("overcap-events", "events"))
def test_live_entry_physical_gate_has_exact_transient_overage_or_event_detail(
    monkeypatch: pytest.MonkeyPatch,
    event_only: bool,
) -> None:
    attempt, _runtime_exit, snapshot = _post_output_gate_fixture()
    cap = auditor.MAXIMUM_ESTIMATED_RSS_BYTES
    snapshot["memory_events"]["max"] = 1
    if not event_only:
        snapshot["memory_current"] = cap
        snapshot["memory_peak"] = cap + 1
    interpreter_runtime = {"fixture": "source-only-runtime"}
    monkeypatch.setattr(
        qualification,
        "_verify_source_only_runtime_bootstrap_v2",
        lambda: interpreter_runtime,
    )
    with pytest.raises(
        qualification.Phase2BFormalWireCovertQualificationErrorV2
    ) as captured:
        qualification._validate_live_runtime_entry_v2(
            {"cgroup": snapshot, "interpreter_runtime": interpreter_runtime},
            attempt,
        )
    expected = _expected_cgroup_physical_resource_failure(
        prefix="live inner entry physical resource gate rejected",
        stage="RUNTIME_ENTRY_SAMPLE",
        snapshot=snapshot,
        cap=cap,
        reserve=0,
    )
    assert captured.value.code == qualification.FAIL_RESOURCE
    assert captured.value.detail == expected
    assert "\n" not in expected and "\r" not in expected
    assert len(expected.encode("ascii")) <= 1_024


@pytest.mark.parametrize("event_only", (False, True), ids=("overcap-events", "events"))
def test_live_exit_physical_gate_has_exact_detail_before_event_drift(
    monkeypatch: pytest.MonkeyPatch,
    event_only: bool,
) -> None:
    attempt, runtime_exit, snapshot = _post_output_gate_fixture()
    entry_cgroup = runtime_exit["cgroup"]
    assert type(entry_cgroup) is dict
    candidate = json.loads(json.dumps(entry_cgroup))
    candidate["memory_events"]["max"] = 1
    cap = auditor.MAXIMUM_ESTIMATED_RSS_BYTES
    if not event_only:
        candidate["memory_current"] = cap
        candidate["memory_peak"] = cap + 1
    real_read_text = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path == Path("/proc/self/status"):
            return "Threads:\t1\nSigBlk:\t0000000000000000\n"
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(signal, "getsignal", lambda _number: signal.SIG_DFL)
    monkeypatch.setattr(
        qualification, "inspect_current_live_cgroup_v2", lambda: candidate
    )
    with pytest.raises(
        qualification.Phase2BFormalWireCovertQualificationErrorV2
    ) as captured:
        qualification._capture_live_runtime_exit_v2(runtime_exit, attempt, [])
    expected = _expected_cgroup_physical_resource_failure(
        prefix="live inner exit physical resource gate rejected",
        stage="RUNTIME_EXIT_SAMPLE",
        snapshot=candidate,
        cap=cap,
        reserve=0,
    )
    assert captured.value.code == qualification.FAIL_RESOURCE
    assert captured.value.detail == expected
    assert "\n" not in expected and "\r" not in expected
    assert len(expected.encode("ascii")) <= 1_024

    scope_splice = json.loads(json.dumps(candidate))
    scope_splice["unified_cgroup_leaf_inode"] += 1
    monkeypatch.setattr(
        qualification, "inspect_current_live_cgroup_v2", lambda: scope_splice
    )
    with pytest.raises(ValueError, match="live inner exit cgroup scope drifted"):
        qualification._capture_live_runtime_exit_v2(runtime_exit, attempt, [])


@pytest.mark.parametrize("event_only", (False, True), ids=("overcap-events", "events"))
def test_post_output_full_gate_has_exact_detail_after_scope_identity(
    event_only: bool,
) -> None:
    attempt, runtime_exit, snapshot = _post_output_gate_fixture()
    cap = auditor.MAXIMUM_ESTIMATED_RSS_BYTES
    reserve = qualification.POST_OUTPUT_DIRECT_EXIT_TAIL_RESERVE_BYTES_V2
    snapshot["memory_events"]["max"] = 1
    if not event_only:
        snapshot["memory_current"] = cap + 1
        snapshot["memory_peak"] = cap + 2
    with pytest.raises(
        qualification.Phase2BFormalWireCovertQualificationErrorV2
    ) as captured:
        qualification._validate_post_output_full_cgroup_snapshot_v2(
            runtime_exit, attempt, snapshot
        )
    expected = _expected_cgroup_physical_resource_failure(
        prefix="post-output full physical resource gate rejected",
        stage="POST_OUTPUT_FULL_SAMPLE",
        snapshot=snapshot,
        cap=cap,
        reserve=reserve,
    )
    assert captured.value.code == qualification.FAIL_RESOURCE
    assert captured.value.detail == expected
    assert "\n" not in expected and "\r" not in expected
    assert len(expected.encode("ascii")) <= 1_024

    scope_splice = json.loads(json.dumps(snapshot))
    scope_splice["unified_cgroup_leaf_inode"] += 1
    with pytest.raises(
        qualification.Phase2BFormalWireCovertQualificationErrorV2
    ) as scope_failure:
        qualification._validate_post_output_full_cgroup_snapshot_v2(
            runtime_exit, attempt, scope_splice
        )
    assert scope_failure.value.code == qualification.FAIL_RESOURCE
    assert scope_failure.value.detail == (
        "post-output cgroup scope drifted after stdout flush"
    )


def test_post_output_gate_separates_twenty_four_gib_hard_max_from_twelve_gib_cap() -> None:
    attempt, runtime_exit, snapshot = _post_output_gate_fixture()
    cap = auditor.MAXIMUM_ESTIMATED_RSS_BYTES
    ceiling = cap - qualification.POST_OUTPUT_DIRECT_EXIT_TAIL_RESERVE_BYTES_V2
    assert attempt.requested_cgroup_memory_max_bytes == 24 * 1_024**3
    assert snapshot["memory_max"] == attempt.requested_cgroup_memory_max_bytes
    assert snapshot["memory_max"] == 2 * cap
    contract = qualification._POST_OUTPUT_PHYSICAL_GATE_CONTRACT_V2
    assert contract["success_peak_threshold_bytes"] == ceiling
    assert (
        contract["systemd_memory_max_bytes"]
        == attempt.requested_cgroup_memory_max_bytes
    )
    assert (
        qualification._validate_post_output_full_cgroup_snapshot_v2(
            runtime_exit, attempt, snapshot
        )
        == snapshot["memory_peak"]
    )
    old_limit_snapshot = json.loads(json.dumps(snapshot))
    old_limit_snapshot["memory_max"] = auditor.MAXIMUM_ESTIMATED_RSS_BYTES
    with pytest.raises(
        qualification.Phase2BFormalWireCovertQualificationErrorV2
    ) as captured:
        qualification._validate_post_output_full_cgroup_snapshot_v2(
            runtime_exit, attempt, old_limit_snapshot
        )
    expected = _expected_cgroup_physical_resource_failure(
        prefix="post-output full physical resource gate rejected",
        stage="POST_OUTPUT_FULL_SAMPLE",
        snapshot=old_limit_snapshot,
        cap=cap,
        reserve=qualification.POST_OUTPUT_DIRECT_EXIT_TAIL_RESERVE_BYTES_V2,
    )
    assert captured.value.code == qualification.FAIL_RESOURCE
    assert captured.value.detail == expected
    assert "memory.max=12884901888;cap=12884901888;reserve=67108864" in expected
    assert "\n" not in expected and "\r" not in expected
    assert len(expected.encode("ascii")) <= 1_024


def test_post_output_success_order_is_write_flush_full_final_direct_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, runtime_exit, full_snapshot = _post_output_gate_fixture()
    events: list[str] = []

    class Buffer:
        def write(self, payload: bytes) -> int:
            events.append("write")
            return len(payload)

        def flush(self) -> None:
            events.append("flush")

    class DirectExit(BaseException):
        pass

    artifact = qualification._canonical_inner_artifact_v2(
        {
            "post_output_physical_gate_contract": dict(
                qualification._POST_OUTPUT_PHYSICAL_GATE_CONTRACT_V2
            ),
            "runtime_exit_evidence": runtime_exit,
        }
    )
    monkeypatch.setattr(qualification, "_inner_stdout_buffer_v2", Buffer)
    monkeypatch.setattr(
        qualification,
        "_prepare_post_output_quiescent_runtime_v2",
        lambda: events.append("quiesce"),
    )
    monkeypatch.setattr(
        qualification,
        "inspect_current_live_cgroup_v2",
        lambda: events.append("full-snapshot") or full_snapshot,
    )
    monkeypatch.setattr(
        qualification,
        "_fresh_current_cgroup_memory_peak_v2",
        lambda snapshot: events.append("final-reread") or 3_072,
    )

    def direct_exit(code: int) -> None:
        events.append(f"direct-exit-{code}")
        raise DirectExit(code)

    monkeypatch.setattr(qualification, "_DIRECT_PROCESS_EXIT_V2", direct_exit)
    with pytest.raises(DirectExit) as raised:
        qualification._write_flush_gate_and_direct_exit_v2(artifact, attempt)
    assert raised.value.args == (0,)
    assert events == [
        "write",
        "flush",
        "quiesce",
        "full-snapshot",
        "final-reread",
        "direct-exit-0",
    ]


def test_valid_canonical_stdout_then_final_gate_failure_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, runtime_exit, full_snapshot = _post_output_gate_fixture()
    written_payloads: list[bytes] = []
    failure_lines: list[bytes] = []
    events: list[str] = []

    class Buffer:
        def write(self, payload: bytes) -> int:
            events.append("write-valid-bytes")
            written_payloads.append(payload)
            return len(payload)

        def flush(self) -> None:
            events.append("flush-valid-bytes")

    class DirectExit(BaseException):
        pass

    artifact = qualification._canonical_inner_artifact_v2(
        {
            "post_output_physical_gate_contract": dict(
                qualification._POST_OUTPUT_PHYSICAL_GATE_CONTRACT_V2
            ),
            "runtime_exit_evidence": runtime_exit,
        }
    )
    monkeypatch.setattr(qualification, "_inner_stdout_buffer_v2", Buffer)
    monkeypatch.setattr(
        qualification, "_prepare_post_output_quiescent_runtime_v2", lambda: None
    )
    monkeypatch.setattr(
        qualification,
        "inspect_current_live_cgroup_v2",
        lambda: events.append("full-snapshot") or full_snapshot,
    )
    monkeypatch.setattr(
        qualification,
        "_fresh_current_cgroup_memory_peak_v2",
        lambda snapshot: (
            events.append("final-reread")
            or
            auditor.MAXIMUM_ESTIMATED_RSS_BYTES
            - qualification.POST_OUTPUT_DIRECT_EXIT_TAIL_RESERVE_BYTES_V2
            + 1
        ),
    )
    monkeypatch.setattr(
        qualification,
        "_write_post_output_failure_line_v2",
        lambda payload: failure_lines.append(payload),
    )

    def direct_exit(code: int) -> None:
        events.append(f"direct-exit-{code}")
        raise DirectExit(code)

    monkeypatch.setattr(qualification, "_DIRECT_PROCESS_EXIT_V2", direct_exit)
    with pytest.raises(DirectExit) as raised:
        qualification._write_flush_gate_and_direct_exit_v2(artifact, attempt)
    assert raised.value.args == (86,)
    assert written_payloads == [artifact]
    assert qualification._decode_canonical_transport_object_v2(
        written_payloads[0], "failed-tail valid stdout", maximum_bytes=1_048_576
    )["runtime_exit_evidence"] == runtime_exit
    assert events == [
        "write-valid-bytes",
        "flush-valid-bytes",
        "full-snapshot",
        "final-reread",
        "direct-exit-86",
    ]
    assert len(failure_lines) == 1
    assert failure_lines[0].startswith(
        (qualification.FAIL_RESOURCE + ": ").encode("ascii")
    )
    assert failure_lines[0].count(b"\n") == 1


@pytest.mark.parametrize(
    "mutation",
    (
        "scope",
        "events",
        "full-cap",
        "full-reserve",
        "final-decrease",
        "final-reserve",
        "final-cap",
    ),
)
def test_post_output_scope_events_cap_and_tail_reserve_fail_closed(
    mutation: str,
) -> None:
    attempt, runtime_exit, full_snapshot = _post_output_gate_fixture()
    cap = auditor.MAXIMUM_ESTIMATED_RSS_BYTES
    ceiling = cap - qualification.POST_OUTPUT_DIRECT_EXIT_TAIL_RESERVE_BYTES_V2
    if mutation == "scope":
        full_snapshot["unified_cgroup_leaf_inode"] += 1
    elif mutation == "events":
        full_snapshot["memory_events"]["max"] = 1
    elif mutation == "full-cap":
        full_snapshot["memory_peak"] = cap + 1
    elif mutation == "full-reserve":
        full_snapshot["memory_peak"] = ceiling + 1
    if mutation.startswith("full") or mutation in {"scope", "events"}:
        with pytest.raises(
            qualification.Phase2BFormalWireCovertQualificationErrorV2
        ):
            qualification._validate_post_output_full_cgroup_snapshot_v2(
                runtime_exit, attempt, full_snapshot
            )
        return
    full_peak = qualification._validate_post_output_full_cgroup_snapshot_v2(
        runtime_exit, attempt, full_snapshot
    )
    final_peak = (
        full_peak - 1
        if mutation == "final-decrease"
        else ceiling + 1
        if mutation == "final-reserve"
        else cap + 1
    )
    with pytest.raises(qualification.Phase2BFormalWireCovertQualificationErrorV2):
        qualification._validate_final_post_output_memory_peak_v2(
            full_snapshot, full_peak, final_peak
        )


def test_final_memory_peak_uses_fresh_nofollow_openat_on_same_leaf(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    leaf = cgroup / "unit.service"
    (proc / "self").mkdir(parents=True)
    leaf.mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/unit.service\n", encoding="ascii")
    (leaf / "memory.peak").write_text("3072\n", encoding="ascii")
    opened = leaf.stat(follow_symlinks=False)
    expected = {
        "cgroup_path": "/unit.service",
        "unified_cgroup_leaf_device": opened.st_dev,
        "unified_cgroup_leaf_inode": opened.st_ino,
    }
    assert qualification._fresh_current_cgroup_memory_peak_v2(
        expected, proc_root=proc, cgroup_root=cgroup
    ) == 3_072
    (leaf / "memory.peak").unlink()
    (leaf / "target").write_text("3072\n", encoding="ascii")
    (leaf / "memory.peak").symlink_to("target")
    with pytest.raises(qualification.Phase2BFormalWireCovertQualificationErrorV2):
        qualification._fresh_current_cgroup_memory_peak_v2(
            expected, proc_root=proc, cgroup_root=cgroup
        )


def test_final_memory_peak_rejects_symlink_leaf_directory(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    real_leaf = cgroup / "real-unit.service"
    linked_leaf = cgroup / "unit.service"
    (proc / "self").mkdir(parents=True)
    real_leaf.mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/unit.service\n", encoding="ascii")
    (real_leaf / "memory.peak").write_text("3072\n", encoding="ascii")
    opened = real_leaf.stat(follow_symlinks=False)
    linked_leaf.symlink_to(real_leaf, target_is_directory=True)
    expected = {
        "cgroup_path": "/unit.service",
        "unified_cgroup_leaf_device": opened.st_dev,
        "unified_cgroup_leaf_inode": opened.st_ino,
    }
    with pytest.raises(
        qualification.Phase2BFormalWireCovertQualificationErrorV2,
        match="final peak memory.peak is unreadable",
    ):
        qualification._fresh_current_cgroup_memory_peak_v2(
            expected, proc_root=proc, cgroup_root=cgroup
        )


@pytest.mark.parametrize("mutation", ("membership", "device", "inode"))
def test_final_memory_peak_rejects_each_same_leaf_binding_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    leaf = cgroup / "unit.service"
    (proc / "self").mkdir(parents=True)
    leaf.mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/unit.service\n", encoding="ascii")
    (leaf / "memory.peak").write_text("3072\n", encoding="ascii")
    opened = leaf.stat(follow_symlinks=False)
    expected = {
        "cgroup_path": "/unit.service",
        "unified_cgroup_leaf_device": opened.st_dev,
        "unified_cgroup_leaf_inode": opened.st_ino,
    }
    if mutation == "membership":
        (proc / "self/cgroup").write_text("0::/other.service\n", encoding="ascii")
    elif mutation == "device":
        expected["unified_cgroup_leaf_device"] += 1
    else:
        expected["unified_cgroup_leaf_inode"] += 1
    with pytest.raises(
        qualification.Phase2BFormalWireCovertQualificationErrorV2,
        match=(
            "final peak unified membership drifted"
            if mutation == "membership"
            else "final peak cgroup leaf identity drifted"
        ),
    ) as caught:
        qualification._fresh_current_cgroup_memory_peak_v2(
            expected, proc_root=proc, cgroup_root=cgroup
        )
    assert caught.value.code == qualification.FAIL_RESOURCE


def test_final_memory_peak_replays_membership_after_peak_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    leaf = cgroup / "unit.service"
    (proc / "self").mkdir(parents=True)
    leaf.mkdir(parents=True)
    membership_path = proc / "self/cgroup"
    membership_path.write_text("0::/unit.service\n", encoding="ascii")
    (leaf / "memory.peak").write_text("3072\n", encoding="ascii")
    opened = leaf.stat(follow_symlinks=False)
    expected = {
        "cgroup_path": "/unit.service",
        "unified_cgroup_leaf_device": opened.st_dev,
        "unified_cgroup_leaf_inode": opened.st_ino,
    }
    real_reader = qualification._read_bounded_cgroup_ascii_v2
    membership_reads = 0

    def migrate_after_peak(
        path: Path,
        name: str,
        *,
        maximum_bytes: int = 64 * 1_024,
    ) -> str:
        nonlocal membership_reads
        if path == membership_path:
            membership_reads += 1
            if membership_reads == 2:
                return "0::/other.service\n"
        return real_reader(path, name, maximum_bytes=maximum_bytes)

    monkeypatch.setattr(
        qualification, "_read_bounded_cgroup_ascii_v2", migrate_after_peak
    )
    with pytest.raises(
        qualification.Phase2BFormalWireCovertQualificationErrorV2,
        match="final peak unified membership drifted during read",
    ) as caught:
        qualification._fresh_current_cgroup_memory_peak_v2(
            expected, proc_root=proc, cgroup_root=cgroup
        )
    assert caught.value.code == qualification.FAIL_RESOURCE
    assert membership_reads == 2


@pytest.mark.parametrize(
    "active_state",
    (
        "trace",
        "profile",
        "python-handler",
        "selected-ignored",
        "unselected-handler",
        "pre-existing-mask",
        "threads",
        "post-block-sigblk",
    ),
)
def test_post_output_quiescence_behavior_rejects_active_python_state_without_leak(
    monkeypatch: pytest.MonkeyPatch,
    active_state: str,
) -> None:
    selected = set(qualification.LIVE_SELECTED_DEFAULT_SIGNALS)
    real_pthread_sigmask = signal.pthread_sigmask
    original_mask = real_pthread_sigmask(signal.SIG_BLOCK, set())
    original_gc_enabled = gc.isenabled()
    real_read_text = Path.read_text
    monkeypatch.setattr(sys, "gettrace", lambda: None)
    monkeypatch.setattr(sys, "getprofile", lambda: None)
    monkeypatch.setattr(signal, "valid_signals", lambda: set(selected))
    monkeypatch.setattr(signal, "getsignal", lambda _number: signal.SIG_DFL)
    if active_state == "trace":
        monkeypatch.setattr(sys, "gettrace", lambda: object())
    elif active_state == "profile":
        monkeypatch.setattr(sys, "getprofile", lambda: object())
    elif active_state == "python-handler":
        handled = next(iter(selected))
        monkeypatch.setattr(
            signal,
            "getsignal",
            lambda number: (lambda *_args: None)
            if number == handled
            else signal.SIG_DFL,
        )
    elif active_state == "selected-ignored":
        ignored = next(iter(selected))
        monkeypatch.setattr(
            signal,
            "getsignal",
            lambda number: signal.SIG_IGN
            if number == ignored
            else signal.SIG_DFL,
        )
    elif active_state == "unselected-handler":
        unselected = signal.SIGUSR1
        assert unselected not in selected
        monkeypatch.setattr(
            signal, "valid_signals", lambda: set(selected) | {unselected}
        )
        monkeypatch.setattr(
            signal,
            "getsignal",
            lambda number: (lambda *_args: None)
            if number == unselected
            else signal.SIG_DFL,
        )
    elif active_state == "pre-existing-mask":
        monkeypatch.setattr(
            signal,
            "pthread_sigmask",
            lambda how, signals: {signal.SIGUSR1},
        )
    else:

        def isolated_quiescence_status(
            path: Path, *args: object, **kwargs: object
        ) -> str:
            text = real_read_text(path, *args, **kwargs)
            if path != Path("/proc/self/status"):
                return text
            rows = []
            for line in text.splitlines():
                if line.startswith("Threads:"):
                    line = (
                        "Threads:\t2"
                        if active_state == "threads"
                        else "Threads:\t1"
                    )
                elif (
                    line.startswith("SigBlk:")
                    and active_state == "post-block-sigblk"
                ):
                    line = "SigBlk:\t0000000000000000"
                rows.append(line)
            return "\n".join(rows) + "\n"

        monkeypatch.setattr(Path, "read_text", isolated_quiescence_status)
    try:
        real_pthread_sigmask(signal.SIG_SETMASK, set())
        with pytest.raises(
            qualification.Phase2BFormalWireCovertQualificationErrorV2,
            match=(
                "active trace/profile"
                if active_state in {"trace", "profile"}
                else "Python signal handlers"
                if active_state
                in {"python-handler", "selected-ignored", "unselected-handler"}
                else "pre-existing signal mask"
                if active_state == "pre-existing-mask"
                else "signal/thread quiescence drifted"
            ),
        ) as caught:
            qualification._prepare_post_output_quiescent_runtime_v2()
        assert caught.value.code == qualification.FAIL_RESOURCE
    finally:
        real_pthread_sigmask(signal.SIG_SETMASK, original_mask)
        gc.enable() if original_gc_enabled else gc.disable()
    assert gc.isenabled() is original_gc_enabled
    assert real_pthread_sigmask(signal.SIG_BLOCK, set()) == original_mask


def test_post_output_quiescence_success_disables_gc_and_blocks_then_fixture_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = set(qualification.LIVE_SELECTED_DEFAULT_SIGNALS)
    real_pthread_sigmask = signal.pthread_sigmask
    original_mask = real_pthread_sigmask(signal.SIG_BLOCK, set())
    original_gc_enabled = gc.isenabled()
    observed_mask: set[signal.Signals] | None = None
    observed_gc_enabled: bool | None = None
    real_read_text = Path.read_text

    def isolated_single_task_status(
        path: Path, *args: object, **kwargs: object
    ) -> str:
        text = real_read_text(path, *args, **kwargs)
        if path == Path("/proc/self/status"):
            rows = [
                "Threads:\t1" if line.startswith("Threads:") else line
                for line in text.splitlines()
            ]
            return "\n".join(rows) + "\n"
        return text

    monkeypatch.setattr(sys, "gettrace", lambda: None)
    monkeypatch.setattr(sys, "getprofile", lambda: None)
    monkeypatch.setattr(signal, "valid_signals", lambda: set(selected))
    monkeypatch.setattr(signal, "getsignal", lambda _number: signal.SIG_DFL)
    # Pytest/plugins own helper threads outside the fresh inner-process TCB.
    # Isolate only that inventory; the kernel-backed signal mask remains real.
    monkeypatch.setattr(Path, "read_text", isolated_single_task_status)
    try:
        real_pthread_sigmask(signal.SIG_SETMASK, set())
        gc.enable()
        qualification._prepare_post_output_quiescent_runtime_v2()
        observed_mask = real_pthread_sigmask(signal.SIG_BLOCK, set())
        observed_gc_enabled = gc.isenabled()
    finally:
        real_pthread_sigmask(signal.SIG_SETMASK, original_mask)
        gc.enable() if original_gc_enabled else gc.disable()
    assert observed_mask == selected
    assert observed_gc_enabled is False
    assert gc.isenabled() is original_gc_enabled
    assert real_pthread_sigmask(signal.SIG_BLOCK, set()) == original_mask


def test_inner_cli_success_cannot_return_or_run_interpreter_teardown() -> None:
    writer_tree = ast.parse(
        inspect.getsource(qualification._write_flush_gate_and_direct_exit_v2)
    )
    direct_zero_calls = [
        node
        for node in ast.walk(writer_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "direct_exit"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 0
    ]
    assert len(direct_zero_calls) == 1
    main_tree = ast.parse(inspect.getsource(qualification._main_v2))
    last = main_tree.body[0].body[-1]
    assert isinstance(last, ast.Expr)
    assert isinstance(last.value, ast.Call)
    assert isinstance(last.value.func, ast.Name)
    assert last.value.func.id == "_write_flush_gate_and_direct_exit_v2"
    writer_source = inspect.getsource(
        qualification._write_flush_gate_and_direct_exit_v2
    )
    assert (
        writer_source.index("stdout_buffer.write")
        < writer_source.index("stdout_buffer.flush")
        < writer_source.index("_prepare_post_output_quiescent_runtime_v2")
        < writer_source.index("inspect_current_live_cgroup_v2")
        < writer_source.index("_fresh_current_cgroup_memory_peak_v2")
        < writer_source.index("direct_exit(0)")
    )
    quiescent_source = inspect.getsource(
        qualification._prepare_post_output_quiescent_runtime_v2
    )
    for required in (
        "sys.gettrace()",
        "sys.getprofile()",
        "callable(value)",
        "gc.disable()",
        "signal.pthread_sigmask",
        'fields.get("Threads") != "1"',
    ):
        assert required in quiescent_source


def test_outer_v3_post_output_attestation_is_closed_and_tamper_evident() -> None:
    attempt = _attempt()
    inner = {
        "post_output_physical_gate_contract": dict(
            qualification._POST_OUTPUT_PHYSICAL_GATE_CONTRACT_V2
        )
    }
    inner_bytes = qualification._canonical_inner_artifact_v2(inner)
    child_argv_sha256 = "a" * 64
    invocation_id = "b" * 32
    attestation = qualification._expected_post_output_gate_attestation_v2(
        attempt,
        inner,
        inner_bytes,
        expected_entry="development-calibration",
        child_argv_sha256=child_argv_sha256,
        owned_invocation_id=invocation_id,
    )
    assert qualification.OUTER_SUPERVISOR_RESULT_SCHEMA_V2.endswith("/3")
    assert attestation["systemd_memory_max_bytes"] == (
        qualification.OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2
    )
    assert attestation["pinned_qualification_source_sha256"] == (
        attempt.qualification_source_sha256
    )
    assert attestation["child_exec_main_code"] == 1
    assert attestation["child_exec_main_status"] == 0
    assert attestation["child_systemd_exit_kind"] == "CLD_EXITED"
    assert attestation["exact_exec_start_argv_sha256"] == child_argv_sha256
    assert attestation["systemd_owned_invocation_id"] == invocation_id
    assert attestation["success_exit_status_override_can_satisfy_attestation"] is False
    assert attestation["canonical_stdout_matches_inner_artifact"] is True
    assert attestation["empty_stderr"] is True
    assert attestation["post_exit_numeric_memory_peak_sample_claimed"] is False
    qualification._validate_post_output_gate_attestation_v2(
        attestation,
        attempt,
        inner,
        inner_bytes,
        context="KAT",
        expected_entry="development-calibration",
        child_argv_sha256=child_argv_sha256,
        owned_invocation_id=invocation_id,
    )
    tampered = json.loads(json.dumps(attestation))
    tampered["post_exit_numeric_memory_peak_sample_claimed"] = True
    with pytest.raises(ValueError, match="attestation drifted"):
        qualification._validate_post_output_gate_attestation_v2(
            tampered,
            attempt,
            inner,
            inner_bytes,
            context="KAT",
            expected_entry="development-calibration",
            child_argv_sha256=child_argv_sha256,
            owned_invocation_id=invocation_id,
        )
    tampered = json.loads(json.dumps(attestation))
    tampered["systemd_memory_max_bytes"] = auditor.MAXIMUM_ESTIMATED_RSS_BYTES
    with pytest.raises(ValueError, match="attestation drifted"):
        qualification._validate_post_output_gate_attestation_v2(
            tampered,
            attempt,
            inner,
            inner_bytes,
            context="KAT",
            expected_entry="development-calibration",
            child_argv_sha256=child_argv_sha256,
            owned_invocation_id=invocation_id,
        )
    live_source = inspect.getsource(
        qualification._validate_outer_supervisor_result_v2
    )
    pure_source = inspect.getsource(
        qualification._validate_outer_supervisor_semantics_pure_v2
    )
    assert "_validate_post_output_gate_attestation_v2" in live_source
    assert "_validate_post_output_gate_attestation_v2" in pure_source


def test_nonzero_tail_gate_exit_cannot_be_reclassified_as_systemd_success() -> None:
    attempt, source_tree, intent = _launch_intent()
    forged = qualification._parse_systemd_show_v2(
        _systemd_show_bytes(
            attempt,
            source_tree,
            marker=intent["ownership_marker"],
            invocation="c" * 32,
            load="loaded",
            active="active",
            sub="exited",
            result="success",
            overrides={"ExecMainCode": "1", "ExecMainStatus": "86"},
        ),
        set(qualification._SYSTEMD_SHOW_PROPERTIES_V2),
    )
    with pytest.raises(ValueError, match="active/exited success projection"):
        qualification._systemd_terminal_success_v2(
            forged, name="forged SuccessExitStatus"
        )


def _full1024_resource_plan_for_inner_v6(
    mode: str,
) -> tuple[auditor.ResourcePlanV1, auditor._FeatureCardinalityProjectionV1]:
    """Build the real frozen 1,024-row scalar plan without materializing a matrix."""

    rows = qualification.EXACT_AUTHORITY_ROWS
    frozen_projection = qualification._fixture_feature_capacity_projection_row_v2(
        mode, rows
    )
    feature_count = frozen_projection["projected_concrete_feature_count"]
    assert type(feature_count) is int
    mode_contract = qualification._FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"][
        mode
    ]
    union_shape = mode_contract["union_shape"]
    capacity_projection = auditor._FeatureCardinalityProjectionV1(
        rows=rows,
        uuid_union_path_count=union_shape["uuid_union_path_count"],
        channel_union_path_count=union_shape["channel_union_path_count"],
        generic_channel_union_path_count=union_shape[
            "generic_channel_union_path_count"
        ],
        provenance_union_path_count=union_shape["provenance_union_path_count"],
        descriptor_union_path_count=union_shape["descriptor_union_path_count"],
        descriptor_segment_position_count=union_shape[
            "descriptor_segment_position_count"
        ],
        eligibility_absent_feature_count=union_shape[
            "eligibility_absent_feature_count"
        ],
        descriptor_source_utf8_bytes=303 * 128,
        present_generic_channel_feature_utf8_bytes=34_760 * 128,
        semantic_path_repetition_utf8_bytes=(
            3_610_791 if mode == "KNOWN_NULL" else 3_593_170
        ),
        projected_concrete_feature_count=feature_count,
        projected_raw_semantic_feature_count=frozen_projection[
            "projected_raw_semantic_feature_count"
        ],
        projected_feature_matrix_bytes_bound=frozen_projection[
            "projected_feature_matrix_bytes_bound"
        ],
    )
    capacity_projection.__post_init__()
    assert capacity_projection.projection_root == frozen_projection["projection_root"]

    cardinalities = (2,) * feature_count
    maximum_mask_bytes = sys.getsizeof((1 << rows) - 1)
    maximum_count_bytes = sys.getsizeof(rows)
    partition_shell_bytes = sys.getsizeof(auditor._Partition((), ()))
    projected_partition_bytes = sys.getsizeof((None,) * feature_count) + sum(
        partition_shell_bytes
        + 2 * sys.getsizeof((None,) * 2)
        + 2 * (maximum_mask_bytes + maximum_count_bytes)
        for _ in range(feature_count)
    )
    partition_projection = auditor._PartitionAllocationProjectionV1(
        rows,
        cardinalities,
        projected_partition_bytes * auditor.DEEP_SIZE_SAFETY_MULTIPLIER,
    )
    shared_partition = auditor._Partition(
        (1, ((1 << rows) - 1) ^ 1),
        (1, rows - 1),
    )
    partitions = (shared_partition,) * feature_count
    envelope_and_decode_bytes_bound = rows * auditor.ENVELOPE_BYTES * 2
    invariant_plan = auditor._invariance_resource_plan(
        rows,
        envelope_and_decode_bytes_bound=envelope_and_decode_bytes_bound,
        adapter_retained_graph_bytes_bound=1,
        variant_suite_resident_bytes_per_row_bound=1,
        variant_suite_resident_fixed_bytes_bound=1,
    )
    feature_matrix_bytes_bound = frozen_projection[
        "projected_feature_matrix_bytes_bound"
    ]
    assert type(feature_matrix_bytes_bound) is int
    plan = auditor._resource_plan(
        rows,
        partitions,
        (2,) * len(auditor.TARGETS),
        preextraction_feature_matrix_bytes_bound=feature_matrix_bytes_bound,
        feature_matrix_bytes_bound=feature_matrix_bytes_bound,
        envelope_and_decode_bytes_bound=envelope_and_decode_bytes_bound,
        adapter_retained_graph_bytes_bound=1,
        partition_projection=partition_projection,
        partition_preparation_deadline_ns=time.monotonic_ns() + 10_000_000_000,
        invariance_plan=invariant_plan,
    )
    assert plan.admissible is True
    assert plan.concrete_features == feature_count
    return plan, capacity_projection


def _valid_full1024_inner_run_v6(
    mode: str,
    *,
    receipt_digit: str,
    invariant_peak: int,
    receipt_peak: int,
) -> tuple[dict[str, object], dict[str, object]]:
    plan, capacity_projection = _full1024_resource_plan_for_inner_v6(mode)
    rows = qualification.EXACT_AUTHORITY_ROWS
    membership = "/test.slice/qv2.scope"
    device = 11
    inode = 22
    scope_root = qualification.stable_hash(
        {
            "leaf_device": device,
            "leaf_inode": inode,
            "unified_membership": membership,
        },
        prefix="phase2b_formal_covert_unified_cgroup_memory_scope_v1_",
    )
    memory_max = qualification.OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2
    memory_current = 512

    invariant_static = (
        plan.envelope_and_decode_bytes_bound
        + plan.adapter_retained_graph_bytes_bound
        + 64 * 1_048_576
    )
    invariant_additional = (
        plan.invariance_variant_suite_resident_bytes_bound
        + plan.invariance_recognizer_live_output_bytes_bound
        + plan.invariance_worker_artifact_bytes_bound
    )
    invariant_preflight = auditor._PhaseResourcePreflightEvidenceV1(
        phase="INVARIANCE",
        resource_plan_root=plan.invariance_resource_plan_root,
        parent_current_rss_bytes_at_preflight=(
            plan.invariance_parent_current_rss_bytes_at_plan
        ),
        static_parent_bytes_bound=invariant_static,
        fork_parent_rss_multiplier=2,
        additional_bytes_bound=invariant_additional,
        projected_aggregate_rss_bytes_at_preflight=(
            max(
                plan.invariance_parent_current_rss_bytes_at_plan,
                invariant_static,
            )
            * 2
            + invariant_additional
        ),
        maximum_estimated_rss_bytes=plan.maximum_invariance_estimated_rss_bytes,
        host_available_memory_safety_margin_bytes=(
            plan.host_available_memory_safety_margin_bytes
        ),
    )
    invariant_available = (
        invariant_preflight.projected_aggregate_rss_bytes_at_preflight
        + invariant_preflight.host_available_memory_safety_margin_bytes
    )
    adapter_pin = (
        "phase2b_formal_covert_adapter_execution_pins_" + receipt_digit * 64
    )
    invariant_runtime = auditor.InvariantRuntimeEvidenceV1(
        invariance_resource_plan_root=plan.invariance_resource_plan_root,
        resource_preflight=invariant_preflight,
        system_available_memory_bytes_at_preflight=invariant_available,
        cgroup_remaining_memory_bytes_at_preflight=None,
        host_available_memory_bytes_at_preflight=invariant_available,
        parent_current_rss_bytes_at_start=(
            invariant_preflight.parent_current_rss_bytes_at_preflight
        ),
        parent_peak_current_rss_bytes_observed=(
            invariant_preflight.parent_current_rss_bytes_at_preflight
        ),
        parent_current_rss_bytes_at_completion=(
            invariant_preflight.parent_current_rss_bytes_at_preflight
        ),
        parent_historical_peak_rss_bytes=(
            invariant_preflight.parent_current_rss_bytes_at_preflight
        ),
        worker_peak_rss_bytes=0,
        worker_result_artifact_bytes=1,
        worker_result_artifact_sha256="sha256:" + receipt_digit * 64,
        diagnostic_parent_plus_worker_historical_peak_sum_bytes=(
            invariant_preflight.parent_current_rss_bytes_at_preflight
        ),
        unified_cgroup_membership=membership,
        unified_cgroup_memory_scope_root=scope_root,
        unified_cgroup_leaf_device=device,
        unified_cgroup_leaf_inode=inode,
        physical_memory_current_bytes_at_completion=memory_current,
        physical_memory_peak_bytes_at_completion=invariant_peak,
        physical_memory_max_bytes=memory_max,
        physical_memory_cap_bytes=auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
        invariance_wall_time_nanoseconds=1,
        worker_timeout_seconds=auditor.INVARIANCE_WORKER_TIMEOUT_SECONDS,
        invariant_run_count=qualification.EXACT_RECOGNIZER_INVOCATIONS,
        exact_invariant_comparisons=rows * 80 * 5,
        fork_input_snapshot_root=(
            "phase2b_formal_covert_invariance_input_snapshot_"
            + receipt_digit * 64
        ),
        adapter_execution_pin_root=adapter_pin,
        variant_suite_root="phase2b_formal_variant_suite_" + receipt_digit * 64,
        consumer_decode_count=rows * 80,
        variant_suite_unique_envelope_object_count=0,
        variant_suite_resident_object_graph_bytes_observed=0,
        recognizer_output_root=(
            "phase2b_formal_covert_recognizer_output_" + receipt_digit * 64
        ),
        execution_containment=auditor.INVARIANCE_EXECUTION_CONTAINMENT,
        parent_side_adapter_tcb=auditor.PARENT_SIDE_ADAPTER_TCB,
        parent_side_adapter_sandboxed=False,
        process_group_empty_at_exit=True,
        child_address_space_limit_bytes=1,
        within_frozen_physical_peak_cap=True,
    )

    statistics_static = (
        plan.permutation_label_mask_bytes_bound
        + plan.partition_matrix_bytes_bound
        + plan.adapter_retained_graph_bytes_bound
        + 64 * 1_048_576
    )
    statistics_additional = (
        plan.metric_cache_bytes_bound * plan.maximum_parallel_workers
        + plan.permutation_metric_tuple_bytes_bound
        + plan.target_max_vector_bytes_bound
        + plan.worker_result_artifact_bytes_bound
        + plan.parent_result_working_bytes_bound
    )
    statistics_multiplier = 1 + plan.maximum_parallel_workers
    statistics_preflight = auditor._PhaseResourcePreflightEvidenceV1(
        phase="STATISTICS",
        resource_plan_root=plan.resource_plan_root,
        parent_current_rss_bytes_at_preflight=plan.parent_current_rss_bytes_at_plan,
        static_parent_bytes_bound=statistics_static,
        fork_parent_rss_multiplier=statistics_multiplier,
        additional_bytes_bound=statistics_additional,
        projected_aggregate_rss_bytes_at_preflight=(
            max(plan.parent_current_rss_bytes_at_plan, statistics_static)
            * statistics_multiplier
            + statistics_additional
        ),
        maximum_estimated_rss_bytes=plan.maximum_estimated_rss_bytes,
        host_available_memory_safety_margin_bytes=(
            plan.host_available_memory_safety_margin_bytes
        ),
    )
    statistics_available = (
        statistics_preflight.projected_aggregate_rss_bytes_at_preflight
        + statistics_preflight.host_available_memory_safety_margin_bytes
    )
    worker_peaks = (0,) * plan.maximum_parallel_workers
    statistics_runtime = auditor.StatisticalRuntimeEvidenceV1(
        resource_plan_root=plan.resource_plan_root,
        resource_preflight=statistics_preflight,
        system_available_memory_bytes_at_preflight=statistics_available,
        cgroup_remaining_memory_bytes_at_preflight=None,
        host_available_memory_bytes_at_preflight=statistics_available,
        parent_historical_peak_rss_bytes_before_receipt_finalization=0,
        worker_peak_rss_bytes=worker_peaks,
        worker_result_artifact_bytes=(1,) * len(auditor.TARGETS),
        worker_result_artifact_sha256=(
            "sha256:" + receipt_digit * 64,
        )
        * len(auditor.TARGETS),
        parent_result_object_graph_bytes_observed=0,
        diagnostic_parent_plus_worker_historical_peak_sum_bytes=0,
        unified_cgroup_membership=membership,
        unified_cgroup_memory_scope_root=scope_root,
        unified_cgroup_leaf_device=device,
        unified_cgroup_leaf_inode=inode,
        physical_memory_current_bytes_before_receipt_finalization=memory_current,
        physical_memory_peak_bytes_before_receipt_finalization=receipt_peak,
        physical_memory_max_bytes=memory_max,
        physical_memory_cap_bytes=auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
        statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds=1,
        worker_timeout_seconds=auditor.STATISTICS_WORKER_TIMEOUT_SECONDS,
        parent_postprocess_deadline_schema=auditor.PARENT_POSTPROCESS_DEADLINE_SCHEMA,
        parent_postprocess_deadline_preemptive=False,
        external_supervisor_required=True,
        receipt_finalization_complete_pending_validation_count=1,
        receipt_finalization_physical_peak_within_cap=True,
        within_frozen_physical_peak_cap=True,
    )
    issuance_runtime = auditor.ReceiptIssuanceRuntimeEvidenceV1(
        absolute_deadline_kind="STATISTICS",
        absolute_deadline_monotonic_nanoseconds=1,
        absolute_deadline_timeout_seconds=auditor.STATISTICS_WORKER_TIMEOUT_SECONDS,
        diagnostic_worker_historical_peak_rss_bytes_sum=0,
        diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization=0,
        diagnostic_parent_plus_worker_historical_peak_sum_bytes=0,
        unified_cgroup_membership=membership,
        unified_cgroup_memory_scope_root=scope_root,
        unified_cgroup_leaf_device=device,
        unified_cgroup_leaf_inode=inode,
        physical_memory_current_bytes_before_receipt_finalization=memory_current,
        physical_memory_peak_bytes_before_receipt_finalization=receipt_peak,
        physical_memory_max_bytes=memory_max,
        physical_memory_cap_bytes=auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
        receipt_finalization_complete_pending_validation_count=1,
        receipt_finalization_locked_deadline_check_count=1,
        receipt_finalization_locked_deadline_within_absolute_limit=True,
        receipt_finalization_locked_physical_peak_within_cap=True,
    )

    plan_projection = qualification._closed_primitive_v2(plan)
    invariant_projection = qualification._closed_primitive_v2(invariant_runtime)
    statistics_projection = qualification._closed_primitive_v2(statistics_runtime)
    issuance_projection = qualification._closed_primitive_v2(issuance_runtime)
    capacity = qualification._closed_primitive_v2(capacity_projection)
    assert type(plan_projection) is dict
    assert type(invariant_projection) is dict
    assert type(statistics_projection) is dict
    assert type(issuance_projection) is dict
    assert type(capacity) is dict

    receipt_id = (
        "phase2b_formal_covert_audit_transcript_" + receipt_digit * 64
    )
    quota_receipt_id = (
        "phase2b_formal_eligibility_mask_quota_" + receipt_digit * 64
    )
    diagnostics = qualification._diagnostic_projection(
        qualification.AuditRunDiagnosticsV2(
            mode=mode,
            wall_time_ns=1,
            parent_historical_peak_rss_before_receipt_finalization_kib=0,
            statistics_worker_historical_peak_rss_kib=tuple(
                0 for _ in worker_peaks
            ),
            parent_plus_worker_historical_peak_rss_kib=0,
            unified_cgroup_membership=membership,
            unified_cgroup_memory_scope_root=scope_root,
            unified_cgroup_leaf_device=device,
            unified_cgroup_leaf_inode=inode,
            physical_memory_current_before_receipt_finalization_bytes=(
                memory_current
            ),
            physical_memory_peak_before_receipt_finalization_bytes=receipt_peak,
            physical_memory_max_bytes=memory_max,
            physical_memory_cap_bytes=auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
            statistics_wall_time_ns=1,
            host_available_memory_bytes_at_preflight=statistics_available,
            recognizer_adapter_invocations=(
                qualification.EXACT_RECOGNIZER_INVOCATIONS
            ),
            minimum_rows_per_invocation=rows,
            maximum_rows_per_invocation=rows,
        )
    )
    target_rows = _valid_target_family_rows()
    quota = {
        "exact_quota_matched": True,
        "receipt_id": quota_receipt_id,
        "targets": [
            {
                "checked_integer_cells": 1,
                "exact_quota_matched": True,
                "maximum_absolute_cross_product_difference": 0,
                "target": target,
            }
            for target in ("family", "scale", "answerable_vs_abstain")
        ],
    }
    mode_contract = qualification._FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"][
        mode
    ]
    union_shape = mode_contract["union_shape"]
    seed_reveal = receipt_digit * 64
    run_fixture_root = "unit-full1024-run-" + mode.lower()
    audit_row = {
        "actual_corpus_audit": False,
        "audited_rows": rows,
        "authority_independence_verified": False,
        "authority_origin_authenticated": False,
        "channel_leaf_paths": union_shape["channel_union_path_count"],
        "channel_only_owned_leaf_paths": mode_contract[
            "channel_only_owned_union_path_count"
        ],
        "channel_only_owned_leaf_union_root": mode_contract[
            "channel_only_owned_union_root"
        ],
        "concrete_features": plan.concrete_features,
        "eligibility_quota": quota,
        "eligibility_quota_root": qualification.stable_hash(
            quota,
            prefix="phase2b_formal_wire_covert_eligibility_quota_v2_",
        ),
        "failure_witness_count": 0 if mode == "KNOWN_NULL" else 1,
        "feature_capacity_projection": capacity,
        "feature_capacity_projection_root": capacity_projection.projection_root,
        "feature_manifest_root": (
            "phase2b_formal_covert_feature_manifest_" + receipt_digit * 64
        ),
        "full_raw_byte_categorical_hypotheses": plan.hypotheses,
        "invariance": {
            "complete_runtime_projection": invariant_projection,
            "complete_runtime_projection_root": qualification.stable_hash(
                invariant_projection,
                prefix="phase2b_formal_wire_covert_invariant_runtime_v4_",
            ),
            "consumer_decode_count": invariant_runtime.consumer_decode_count,
            "exact_comparisons": invariant_runtime.exact_invariant_comparisons,
            "execution_root": (
                "phase2b_formal_covert_invariant_execution_"
                + receipt_digit * 64
            ),
            "invariance_resource_plan_root": plan.invariance_resource_plan_root,
            "physical_memory_cap_bytes": auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
            "physical_memory_current_bytes_at_completion": memory_current,
            "physical_memory_max_bytes": memory_max,
            "physical_memory_peak_bytes_at_completion": invariant_peak,
            "run_count": invariant_runtime.invariant_run_count,
            "unified_cgroup_leaf_device": device,
            "unified_cgroup_leaf_inode": inode,
            "unified_cgroup_membership": membership,
            "unified_cgroup_memory_scope_root": scope_root,
            "variant_suite_root": invariant_runtime.variant_suite_root,
            "wall_time_nanoseconds": invariant_runtime.invariance_wall_time_nanoseconds,
            "worker_artifact_sha256": (
                invariant_runtime.worker_result_artifact_sha256
            ),
            "worker_peak_rss_bytes": invariant_runtime.worker_peak_rss_bytes,
            "within_frozen_physical_peak_cap": True,
        },
        "namespace_uuid_leaf_paths": union_shape["uuid_union_path_count"],
        "positive_decision_claim": auditor.POSITIVE_DECISION_CLAIM,
        "raw_semantic_features": capacity_projection.projected_raw_semantic_feature_count,
        "receipt_id": receipt_id,
        "receipt_issuance": {
            "absolute_deadline_kind": issuance_runtime.absolute_deadline_kind,
            "complete_runtime_projection": issuance_projection,
            "complete_runtime_projection_root": qualification.stable_hash(
                issuance_projection,
                prefix="phase2b_formal_wire_covert_receipt_issuance_runtime_v4_",
            ),
            "diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization": 0,
            "diagnostic_parent_plus_worker_historical_peak_sum_bytes": 0,
            "diagnostic_worker_historical_peak_rss_bytes_sum": 0,
            "physical_memory_cap_bytes": auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
            "physical_memory_current_bytes_before_receipt_finalization": memory_current,
            "physical_memory_max_bytes": memory_max,
            "physical_memory_peak_bytes_before_receipt_finalization": receipt_peak,
            "receipt_finalization_complete_pending_validation_count": 1,
            "receipt_finalization_locked_deadline_check_count": 1,
            "receipt_finalization_locked_deadline_within_absolute_limit": True,
            "receipt_finalization_locked_physical_peak_within_cap": True,
            "receipt_id": receipt_id,
            "unified_cgroup_leaf_device": device,
            "unified_cgroup_leaf_inode": inode,
            "unified_cgroup_membership": membership,
            "unified_cgroup_memory_scope_root": scope_root,
        },
        "resource_plan": plan_projection,
        "resource_plan_root": qualification.stable_hash(
            plan_projection,
            prefix="phase2b_formal_wire_covert_resource_plan_v3_",
        ),
        "statistics": {
            "complete_runtime_projection": statistics_projection,
            "complete_runtime_projection_root": qualification.stable_hash(
                statistics_projection,
                prefix="phase2b_formal_wire_covert_statistics_runtime_v4_",
            ),
            "diagnostic_parent_plus_worker_historical_peak_sum_bytes": 0,
            "external_supervisor_required": True,
            "pairwise_holm_decision_eligible": False,
            "pairwise_holm_family_size": 5 * plan.concrete_features,
            "permutation_schedule_root": (
                "phase2b_formal_covert_permutation_schedule_"
                + receipt_digit * 64
            ),
            "permutation_seed_commitment": auditor._permutation_seed_commitment(
                bytes.fromhex(seed_reveal)
            ),
            "permutation_seed_reveal_hex": seed_reveal,
            "physical_memory_cap_bytes": auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
            "physical_memory_current_bytes_before_receipt_finalization": memory_current,
            "physical_memory_max_bytes": memory_max,
            "physical_memory_peak_bytes_before_receipt_finalization": receipt_peak,
            "resource_plan_root": plan.resource_plan_root,
            "statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds": 1,
            "unified_cgroup_leaf_device": device,
            "unified_cgroup_leaf_inode": inode,
            "unified_cgroup_membership": membership,
            "unified_cgroup_memory_scope_root": scope_root,
            "worker_artifact_sha256": list(
                statistics_runtime.worker_result_artifact_sha256
            ),
            "worker_peak_rss_bytes": list(statistics_runtime.worker_peak_rss_bytes),
            "within_frozen_physical_peak_cap": True,
        },
        "target_family_max_t_holm": target_rows,
    }
    wire_row = {
        "adapter_execution_pin_root": adapter_pin,
        "batch_id": "unit-batch-" + mode.lower(),
        "build_success_id": "unit-build-" + mode.lower(),
        "fresh_build_verification_root": "unit-fresh-build-" + mode.lower(),
        "key_source_receipt_id": "unit-key-source-" + mode.lower(),
        "private_alignment_id": "unit-alignment-" + mode.lower(),
        "transform_catalog_root": "unit-transform-catalog",
        "variant_context_id": "unit-variant-" + mode.lower(),
    }
    if mode == "KNOWN_NULL":
        known_leak_witness = None
    else:
        binding = next(row for row in target_rows if row["target"] == "binding")
        alias = (
            "uuid_leaf/observation_id/"
            + qualification._json_path_text_v2(
                qualification._REGISTERED_BINDING_LEAK_OWNER_PATH_V2
            )
            + "/presence"
        )
        known_leak_witness = {
            "balanced_accuracy_advantage_hex": (0.0).hex(),
            "bias_corrected_nmi_hex": (0.04).hex(),
            "failure_witness": True,
            "feature_aliases": [alias],
            "feature_id": "unit-binding-leak-feature",
            "feature_localization_rejected": True,
            "feature_max_t_exceedance_count": 5,
            "feature_max_t_p": "6/10001",
            "feature_owner": "eligibility_mask",
            "feature_owner_path": qualification._json_path_text_v2(
                qualification._REGISTERED_BINDING_LEAK_OWNER_PATH_V2
            ),
            "material_effect": True,
            "observed_omnibus_hex": (2.0).hex(),
            "passed": False,
            "raw_nmi_hex": (0.04).hex(),
            "registered_presence_alias": alias,
            "target_family_holm_rejected": True,
            "target_holm_critical_value": binding["holm_critical_value"],
        }
    run = {
        "audit": audit_row,
        "diagnostics": diagnostics,
        "disposition": "PASS" if mode == "KNOWN_NULL" else "FAIL",
        "known_leak_witness": known_leak_witness,
        "label_factorial_proof_root": "unit-label-factorial-proof",
        "mode": mode,
        "namespace_row_proof_root": "unit-namespace-row-proof",
        "operation_counts": {
            operation: qualification.EXACT_ROWS_PER_TRANSFORM_OPERATION
            for operation in qualification.EXACT_TRANSFORM_OPERATIONS
        },
        "run_fixture_root": run_fixture_root,
        "source_pair_delta_root": (
            None if mode == "KNOWN_NULL" else "unit-source-pair-delta"
        ),
        "wire": wire_row,
    }
    frozen = qualification._static_shape_from_run_projection_v2(run)
    measurement = {
        "diagnostics": diagnostics,
        "dynamic_fields_are_recorded_but_not_exact_compared": list(
            qualification.DYNAMIC_RESOURCE_PLAN_FIELDS_NOT_EXACT_COMPARED_V2
        ),
        "frozen_static_projection": frozen,
        "frozen_static_projection_root": qualification.stable_hash(
            frozen,
            prefix="phase2b_formal_wire_covert_static_shape_v3_",
        ),
        "mode": mode,
        "observed_complete_receipt_issuance_runtime": issuance_projection,
        "observed_complete_receipt_issuance_runtime_root": qualification.stable_hash(
            issuance_projection,
            prefix="phase2b_formal_wire_covert_receipt_issuance_runtime_v4_",
        ),
        "observed_complete_resource_plan": plan_projection,
        "observed_complete_resource_plan_root": qualification.stable_hash(
            plan_projection,
            prefix="phase2b_formal_wire_covert_observed_resource_plan_v3_",
        ),
        "observed_receipt_id": receipt_id,
    }
    return run, measurement


def _valid_development_inner_v6(
    attempt: qualification.Phase2BFormalWireCovertAttemptManifestV2,
    source_tree: Path,
) -> dict[str, object]:
    positive_run, positive_measurement = _valid_full1024_inner_run_v6(
        "KNOWN_NULL",
        receipt_digit="2",
        invariant_peak=2_048,
        receipt_peak=4_096,
    )
    negative_run, negative_measurement = _valid_full1024_inner_run_v6(
        "KNOWN_BINDING_LEAK",
        receipt_digit="3",
        invariant_peak=4_096,
        receipt_peak=8_192,
    )
    membership = "/test.slice/qv2.scope"
    device = 11
    inode = 22
    scope_root = qualification.stable_hash(
        {
            "leaf_device": device,
            "leaf_inode": inode,
            "unified_membership": membership,
        },
        prefix="phase2b_formal_covert_unified_cgroup_memory_scope_v1_",
    )

    def cgroup_row(*, current: int, peak: int) -> dict[str, object]:
        return {
            "cgroup_path": membership,
            "cgroup_procs": [os.getpid()],
            "cgroup_stat": {"nr_descendants": 0, "nr_dying_descendants": 0},
            "cpu_affinity": [0],
            "cpu_capacity": 1,
            "cpu_max": None,
            "memory_current": current,
            "memory_events": {
                "high": 0,
                "low": 0,
                "max": 0,
                "oom": 0,
                "oom_kill": 0,
            },
            "memory_max": attempt.requested_cgroup_memory_max_bytes,
            "memory_peak": peak,
            "pids_current": 1,
            "pids_events": {"max": 0},
            "pids_max": attempt.requested_cgroup_pids_max,
            "unified_cgroup_leaf_device": device,
            "unified_cgroup_leaf_inode": inode,
            "unified_cgroup_memory_scope_root": scope_root,
        }

    selected_signals = [
        number.name for number in qualification.LIVE_SELECTED_DEFAULT_SIGNALS
    ]
    runtime_entry = {
        "cgroup": cgroup_row(current=512, peak=1_024),
        "interpreter_runtime": {
            "bytecode_candidates_found": [],
            "dont_write_bytecode": 1,
            "executed_main_source_sha256": attempt.qualification_source_sha256,
            "ignore_environment": 1,
            "isolated": 1,
            "no_site": 1,
            "package_directory": str(source_tree / "src/hegel_machine"),
            "preopened_source_snapshot_root": (
                "phase2b_formal_wire_covert_preopened_sources_v2_" + "4" * 64
            ),
            "source_execution_bootstrap_active": True,
            "source_execution_bootstrap_manifest_root": (
                "phase2b_formal_wire_covert_source_bootstrap_manifest_v2_"
                + "5" * 64
            ),
            "source_only_importer_active": True,
        },
        "selected_default_signals": selected_signals,
        "sigblk_hex": "0000000000000000",
        "threads": 1,
    }
    runtime_exit = {
        "audit_receipts_validated_and_released_in_process": True,
        "cgroup": cgroup_row(current=512, peak=8_192),
        "cgroup_events_unchanged_from_entry": True,
        "selected_default_signals": selected_signals,
        "sigblk_hex": "0000000000000000",
        "threads": 1,
    }
    runs = [positive_run, negative_run]
    return {
        "artifact_cannot_be_finalized_or_promoted": True,
        "attempt_manifest_root": attempt.attempt_manifest_root,
        "calibration_attempt_id": attempt.attempt_id,
        "calibration_campaign_id": attempt.campaign_id,
        "claim_level": "DEVELOPMENT_MEASUREMENT_ONLY_NOT_R03_R04_R05",
        "claims": {
            "actual_corpus_audit": False,
            "actual_unsealed_960_generation": False,
            "campaign_execution_authorized": False,
            "fixture_specific_only": True,
            "joint_decision_classes": 2,
            "key_independence_verified": False,
            "origin_authenticated": False,
            "r03_formal_trusted_wire_implementation_qualified": False,
            "r04_formal_covert_auditor_implementation_qualified": False,
            "r05_synthetic_dev_wire_covert_qualification_verified": False,
            "wire_wider_graphs_and_100_class_profile_qualified": False,
        },
        "content_sha256_is_not_a_signature": True,
        "external_supervisor_evidence_required": True,
        "frozen_static_measurement_candidates_in_order": [
            positive_measurement,
            negative_measurement,
        ],
        "independent_review_and_profile_update_required": True,
        "mode_order": list(qualification.LIVE_MODE_ORDER),
        "outer_hard_deadline_self_verified": False,
        "post_output_physical_gate_contract": dict(
            qualification._POST_OUTPUT_PHYSICAL_GATE_CONTRACT_V2
        ),
        "profile_id": attempt.profile_id,
        "projection_only_boundary": dict(
            qualification._TASK4_PROJECTION_ONLY_BOUNDARY_V2
        ),
        "runs_in_order": runs,
        "runtime_entry_evidence": runtime_entry,
        "runtime_exit_evidence": runtime_exit,
        "same_run_shape_acceptance_forbidden": True,
        "schema_version": (
            "hegel-machine-phase2b-formal-wire-covert-development-calibration/6"
        ),
        "synthetic_fixture_root": (
            qualification._combined_fixture_root_from_run_roots_v2(
                positive_run["run_fixture_root"],
                negative_run["run_fixture_root"],
            )
        ),
    }


def _valid_outer_main_pid_replay_v3(tmp_path: Path) -> SimpleNamespace:
    source_archive_bytes = b"outer-main-pid-source-archive"
    source_digest = hashlib.sha256(source_archive_bytes).hexdigest()
    source_sha = hashlib.sha256(
        (ROOT / qualification._BOOTSTRAP_MAIN_RELATIVE_PATH_V2).read_bytes()
    ).hexdigest()
    python_row = _executable_row("/usr/bin/python3")
    systemd_run_row = _executable_row("/usr/bin/systemd-run")
    systemctl_row = _executable_row("/usr/bin/systemctl")
    runtime = {
        "cgroup2": {
            "controllers": ["memory", "pids"],
            "membership": "0::/test.slice/qv2.scope",
            "mount": {
                "filesystem_type": "cgroup2",
                "mount_options": "rw,nosuid,nodev,noexec,relatime",
                "mount_point": "/sys/fs/cgroup",
                "mount_root": "/",
            },
        },
        "cpu_affinity": [0],
        "dependency_lock_runtime": {
            "lock_platform": "linux/amd64",
            "platform_machine": "x86_64",
            "platform_system": "Linux",
            "python_abi": "cp310",
            "python_cache_tag": "cpython-310",
            "python_implementation": "CPython",
            "python_soabi": "cpython-310-x86_64-linux-gnu",
            "python_version": "3.10.12",
        },
        "dependency_lock_sha256": qualification.DEPENDENCY_LOCK_SHA256,
        "environment_scope": (
            "ONLY_LANG_AND_LC_ALL_OVERRIDES_ARE_QUALIFIED;"
            "OTHER_USER_MANAGER_ENVIRONMENT_IS_NOT_QUALIFIED"
        ),
        "executables": {
            "python": python_row,
            "systemd_run": systemd_run_row,
            "systemctl": systemctl_row,
        },
        "fresh_dedicated_user_service_manager_session_required": True,
        "host": {
            "architecture": "x86_64",
            "kernel_release": "test-release",
            "kernel_system": "Linux",
            "kernel_version": "test-version",
            "libc_name": "glibc",
            "libc_version": "2.31",
        },
        "host_runtime_fingerprint_only_not_full_runtime_image_closure": True,
        "interpreter_contract": {
            "argv_flags": ["-I", "-S", "-B"],
            "forbid_pyc_pyo": True,
            "ignore_python_environment": True,
            "no_site": True,
            "source_only_loader": True,
        },
        "runtime_image_digest": None,
        "schema_version": qualification.RUNTIME_ENVIRONMENT_MANIFEST_SCHEMA_V2,
        "selected_locale_overrides": {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        "systemd_user_manager_same_uid_is_cooperative_outer_tcb": True,
    }
    runtime_bytes = qualification._canonical_inner_artifact_v2(runtime)
    runtime_digest = hashlib.sha256(runtime_bytes).hexdigest()
    rfc_receipt = {"executables": {"python": python_row}}
    rfc_bytes = qualification._canonical_inner_artifact_v2(rfc_receipt)
    rfc_digest = hashlib.sha256(rfc_bytes).hexdigest()
    attempt = replace(
        _attempt(),
        source_archive_root="sha256:" + source_digest,
        source_archive_size_bytes=len(source_archive_bytes),
        dependency_lock_sha256=qualification.DEPENDENCY_LOCK_SHA256,
        qualification_source_sha256=source_sha,
        qualification_lock_sha256=qualification.DEPENDENCY_LOCK_SHA256,
        runtime_environment_manifest_root="sha256:" + runtime_digest,
        runtime_environment_manifest_size_bytes=len(runtime_bytes),
        rfc8785_prerequisite_receipt_root="sha256:" + rfc_digest,
        rfc8785_prerequisite_receipt_sha256=rfc_digest,
        rfc8785_prerequisite_receipt_size_bytes=len(rfc_bytes),
    )
    source_tree = (tmp_path / "attempt/evidence/source-tree").resolve()
    outer_tree = (tmp_path / "outer-runner-tree").resolve()
    source_tree.mkdir(parents=True)
    outer_tree.mkdir(parents=True)
    cas_directory = source_tree.parent / "cas"
    raw_directory = source_tree.parent / "rfc8785-raw-bundle"
    cas_directory.mkdir()
    raw_directory.mkdir()
    (cas_directory / source_digest).write_bytes(source_archive_bytes)
    (cas_directory / runtime_digest).write_bytes(runtime_bytes)

    bootstrap_manifest = qualification._source_execution_bootstrap_manifest_v2(
        source_sha
    )
    expected_files = dict(bootstrap_manifest["sha256"])
    tree_rows: list[dict[str, object]] = []
    child_snapshots: dict[str, dict[str, object]] = {}
    outer_snapshots: dict[str, dict[str, object]] = {}
    for relative, digest in sorted(expected_files.items()):
        payload = (ROOT / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
        path = source_tree / relative
        outer_path = outer_tree / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        outer_path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        outer_path.write_bytes(payload)
        opened = path.stat(follow_symlinks=False)
        outer_opened = outer_path.stat(follow_symlinks=False)
        tree_rows.append(
            {
                "path": relative,
                "sha256": digest,
                "size_bytes": len(payload),
                "type": "regular",
            }
        )
        child_snapshots[relative] = {
            "descriptor_realpath": str(path),
            "device": opened.st_dev,
            "inode": opened.st_ino,
            "mtime_nanoseconds": opened.st_mtime_ns,
            "sha256": digest,
            "size_bytes": len(payload),
        }
        outer_snapshots[relative] = {
            "descriptor_realpath": str(outer_path),
            "device": outer_opened.st_dev,
            "inode": outer_opened.st_ino,
            "mtime_nanoseconds": outer_opened.st_mtime_ns,
            "sha256": digest,
            "size_bytes": len(payload),
        }
    tree_rows.sort(key=lambda row: (row["path"], row["type"]))
    tree_root = qualification.stable_hash(
        tree_rows, prefix="phase2b_formal_wire_covert_source_tree_v2_"
    )
    regular_root = qualification._regular_file_manifest_root_v2(tree_rows)
    outer_preopened_root = qualification.stable_hash(
        outer_snapshots, prefix="phase2b_formal_wire_covert_preopened_sources_v2_"
    )
    child_preopened_root = qualification.stable_hash(
        child_snapshots, prefix="phase2b_formal_wire_covert_preopened_sources_v2_"
    )
    bootstrap_root = qualification.stable_hash(
        bootstrap_manifest,
        prefix="phase2b_formal_wire_covert_source_bootstrap_manifest_v2_",
    )
    inner = _valid_development_inner_v6(attempt, source_tree)
    interpreter = inner["runtime_entry_evidence"]["interpreter_runtime"]
    interpreter["preopened_source_snapshot_root"] = child_preopened_root
    interpreter["source_execution_bootstrap_manifest_root"] = bootstrap_root
    qualification._validate_development_calibration_artifact_v2(inner, attempt)
    inner_bytes = qualification._canonical_inner_artifact_v2(inner)
    main_snapshot = outer_snapshots[qualification._BOOTSTRAP_MAIN_RELATIVE_PATH_V2]
    outer_tcb = {
        "broad_package_init_executed": False,
        "bypassed_package_init_sha256": qualification.BYPASSED_PACKAGE_INIT_SHA256,
        "executed_main_source_bootstrap": {
            **main_snapshot,
            "bootstrap_kind": "EXTERNAL_PYTHON_C_OPEN_ONCE_V2",
            "reexecuted_from_stable_descriptor": True,
        },
        "interpreter_flags": {
            "dont_write_bytecode": 1,
            "ignore_environment": 1,
            "isolated": 1,
            "no_site": 1,
        },
        "live_files_sha256": expected_files,
        "preopened_source_snapshots": outer_snapshots,
        "source_bootstrap_manifest": bootstrap_manifest,
        "source_execution_bootstrap_active": True,
        "source_only_importer_active": True,
        "source_realpath": str(
            outer_tree / qualification._BOOTSTRAP_MAIN_RELATIVE_PATH_V2
        ),
        "source_sha256": source_sha,
        "source_tree_root": str(outer_tree),
    }
    _, _, intent = _launch_intent(attempt, source_tree)
    intent["executables"] = {
        "python": python_row,
        "systemd_run": systemd_run_row,
        "systemctl": systemctl_row,
    }
    systemctl = systemctl_row["realpath"]
    show_argv = qualification._systemd_show_argv_v2(
        systemctl, intent["unit_name"]
    )
    invocation = "b" * 32
    main_pid = inner["runtime_entry_evidence"]["cgroup"]["cgroup_procs"][0]
    entry = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="running",
        main_pid=main_pid,
        cgroup="/test.slice/qv2.scope",
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
            argv=show_argv,
            purpose="entry-show-0",
            returncode=0,
            stdout=entry,
            stderr=b"",
        ),
        qualification._command_evidence_v2(
            argv=show_argv,
            purpose="terminal-show-0",
            returncode=0,
            stdout=terminal,
            stderr=b"",
        ),
        qualification._command_evidence_v2(
            argv=show_argv,
            purpose="terminal-show-1",
            returncode=0,
            stdout=terminal,
            stderr=b"",
        ),
        qualification._command_evidence_v2(
            argv=[systemctl, "--user", "stop", intent["unit_name"]],
            purpose="stop-after-terminal-snapshot",
            returncode=0,
            stdout=b"",
            stderr=b"",
        ),
        qualification._command_evidence_v2(
            argv=show_argv,
            purpose="post-stop-absent-check-0",
            returncode=0,
            stdout=absent,
            stderr=b"",
        ),
    ]
    effective_properties = {
        "FinalKillSignal": int(signal.SIGKILL),
        "KillMode": "control-group",
        "KillSignal": int(signal.SIGTERM),
        "MemoryMax": attempt.requested_cgroup_memory_max_bytes,
        "RemainAfterExit": True,
        "RuntimeMaxUSec": qualification.LIVE_HARD_TOTAL_TIMEOUT_SECONDS
        * 1_000_000,
        "SendSIGKILL": True,
        "TasksMax": attempt.requested_cgroup_pids_max,
        "TimeoutStopUSec": 30_000_000,
        "Type": "exec",
    }
    extraction = {
        "archive_format": "TAR_REGULAR_AND_DIRECTORY_ENTRIES_ONLY",
        "archive_sha256": source_digest,
        "archive_size_bytes": len(source_archive_bytes),
        "directory_count": 0,
        "forbidden_bytecode_count": 0,
        "member_count": len(tree_rows),
        "regular_file_count": len(tree_rows),
        "regular_file_manifest_root": regular_root,
        "source_tree_manifest_row_count": len(tree_rows),
        "source_tree_manifest_root": tree_root,
        "symlink_hardlink_special_count": 0,
        "total_regular_bytes": sum(row["size_bytes"] for row in tree_rows),
    }
    raw_projection: dict[str, object] = {}
    inner_sha = hashlib.sha256(inner_bytes).hexdigest()
    body = {
        "attempt_manifest_root": attempt.attempt_manifest_root,
        "child_source_bootstrap_preopened_snapshots": child_snapshots,
        "child_source_bootstrap_preopened_snapshots_root": child_preopened_root,
        "content_sha256_is_not_a_signature": True,
        "inner_artifact_cas": qualification._cas_projection_v2(
            inner_sha, len(inner_bytes), attempt.attempt_manifest_root
        ),
        "launcher": {
            "argv": intent["child_argv"],
            "environment": {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
            "python": python_row,
            "systemd_argv": intent["systemd_argv"],
            "systemd_run": systemd_run_row,
            "systemctl": systemctl_row,
            "systemctl_commands": commands,
        },
        "live_files_sha256": expected_files,
        "mode": "development-calibration",
        "outer_runner_tcb_entry": outer_tcb,
        "outer_runner_tcb_exit": outer_tcb,
        "post_output_physical_gate_attestation": (
            qualification._expected_post_output_gate_attestation_v2(
                attempt,
                inner,
                inner_bytes,
                expected_entry="development-calibration",
                child_argv_sha256=intent["child_argv_sha256"],
                owned_invocation_id=invocation,
            )
        ),
        "profile_id": attempt.profile_id,
        "result": {
            "control_group_empty": True,
            "control_group_property_at_terminal_snapshot": "",
            "cgroup_path_absent_at_terminal_snapshot": True,
            "exec_main_code": 1,
            "exec_main_status": 0,
            "kill_sent": False,
            "launcher_exit_code": 0,
            "launcher_reaped": True,
            "main_pid_at_terminal_snapshot": 0,
            "result": "success",
            "term_sent": False,
            "timed_out": False,
            "unit_absent_after_collection": True,
            "unit_active_state_at_terminal_snapshot": "active",
            "unit_collected_after_terminal_snapshot": True,
            "unit_sub_state_at_terminal_snapshot": "exited",
        },
        "rfc8785_receipt_cas": qualification._cas_projection_v2(
            rfc_digest, len(rfc_bytes), attempt.attempt_manifest_root
        ),
        "rfc8785_raw_bundle_directory": str(raw_directory),
        "rfc8785_raw_bundle_projection": raw_projection,
        "runtime_environment_manifest_cas": qualification._cas_projection_v2(
            runtime_digest, len(runtime_bytes), attempt.attempt_manifest_root
        ),
        "runtime_environment_manifest_object_path": str(
            cas_directory / runtime_digest
        ),
        "runtime_environment_manifest_root": (
            attempt.runtime_environment_manifest_root
        ),
        "runtime_image_digest": None,
        "schema_version": qualification.OUTER_SUPERVISOR_RESULT_SCHEMA_V2,
        "source_archive_cas": qualification._cas_projection_v2(
            source_digest, len(source_archive_bytes), attempt.attempt_manifest_root
        ),
        "source_archive_extraction": extraction,
        "source_archive_object_path": str(cas_directory / source_digest),
        "source_archive_root": attempt.source_archive_root,
        "source_commit_sha": attempt.source_commit_sha,
        "source_tree_root": str(source_tree),
        "stderr": {
            "sha256": hashlib.sha256(b"").hexdigest(),
            "size_bytes": 0,
        },
        "stdout": {"sha256": inner_sha, "size_bytes": len(inner_bytes)},
        "timing": {
            "elapsed_to_supervisor_ready_nanoseconds": 200,
            "service_ended_monotonic_nanoseconds": 200,
            "started_monotonic_nanoseconds": 100,
            "supervisor_ready_monotonic_nanoseconds": 300,
            "watchdog_deadline_monotonic_nanoseconds": 100
            + qualification.LIVE_HARD_TOTAL_TIMEOUT_SECONDS * 1_000_000_000,
        },
        "unit": {
            "cgroup_path": "/test.slice/qv2.scope",
            "child_argv_sha256": intent["child_argv_sha256"],
            "effective_properties": effective_properties,
            "invocation_id": invocation,
            "owned_invocation_id": invocation,
            "ownership_marker": intent["ownership_marker"],
            "ownership_nonce": intent["ownership_nonce"],
            "prelaunch_absent_verified": True,
            "selected_environment_root": attempt.runtime_environment_manifest_root,
            "unit_name": intent["unit_name"],
            "working_directory": str(source_tree),
        },
    }
    return SimpleNamespace(
        attempt=attempt,
        body=body,
        expected_files=expected_files,
        inner=inner,
        inner_bytes=inner_bytes,
        main_pid=main_pid,
        child_preopened_root=child_preopened_root,
        outer_preopened_root=outer_preopened_root,
        raw_projection=raw_projection,
        rfc_bytes=rfc_bytes,
        rfc_receipt=rfc_receipt,
        runtime_bytes=runtime_bytes,
        source_tree=source_tree,
        tree_root=tree_root,
        tree_rows=tree_rows,
    )


def test_outer_live_and_pure_main_pid_crossbinding_rejects_splices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = _valid_outer_main_pid_replay_v3(tmp_path)
    assert (
        replay.inner["runtime_entry_evidence"]["interpreter_runtime"][
            "preopened_source_snapshot_root"
        ]
        != replay.outer_preopened_root
    )
    real_sha256_file = qualification._sha256_file_v2

    def replay_file_sha256(path: Path, **kwargs: object) -> str:
        del kwargs
        try:
            relative = str(path.relative_to(replay.source_tree))
        except ValueError:
            relative = ""
        if relative in replay.expected_files:
            return replay.expected_files[relative]
        if path.parent == replay.source_tree.parent / "cas":
            return path.name
        return real_sha256_file(path)

    monkeypatch.setattr(
        qualification,
        "_expected_live_file_sha256_v2",
        lambda _attempt: replay.expected_files,
    )
    monkeypatch.setattr(
        qualification,
        "_verify_live_package_tree_is_source_only_v2",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(qualification, "_sha256_file_v2", replay_file_sha256)
    monkeypatch.setattr(
        qualification,
        "_inspect_source_archive_regular_manifest_v2",
        lambda *_args, **_kwargs: (
            qualification._regular_file_manifest_root_v2(replay.tree_rows),
            len(replay.tree_rows),
            sum(row["size_bytes"] for row in replay.tree_rows),
            len(replay.tree_rows),
            0,
            replay.tree_rows,
            replay.tree_root,
        ),
    )
    monkeypatch.setattr(
        qualification,
        "_source_tree_manifest_v2",
        lambda *_args, **_kwargs: (replay.tree_rows, replay.tree_root),
    )
    monkeypatch.setattr(
        qualification,
        "_verify_live_executable_v2",
        lambda row, name, **_kwargs: (
            qualification._verify_executable_projection_shape_v2(row, name)
        ),
    )
    monkeypatch.setattr(
        qualification,
        "_verify_runtime_environment_manifest_v2",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        qualification,
        "_verify_rfc8785_raw_execution_bundle_v2",
        lambda *_args, **_kwargs: replay.raw_projection,
    )

    def validate_live(
        body: dict[str, object],
        inner: dict[str, object],
        inner_bytes: bytes,
    ) -> None:
        qualification._validate_outer_supervisor_result_v2(
            body,
            attempt=replay.attempt,
            inner=inner,
            inner_bytes=inner_bytes,
            rfc_receipt=replay.rfc_receipt,
            rfc_receipt_bytes=replay.rfc_bytes,
            expected_entry="development-calibration",
        )

    def validate_pure(
        body: dict[str, object],
        inner: dict[str, object],
        inner_bytes: bytes,
    ) -> None:
        qualification._validate_outer_supervisor_semantics_pure_v2(
            body,
            attempt=replay.attempt,
            inner=inner,
            inner_bytes=inner_bytes,
            rfc_receipt=replay.rfc_receipt,
            rfc_receipt_bytes=replay.rfc_bytes,
            runtime_manifest_bytes=replay.runtime_bytes,
            source_tree_rows=replay.tree_rows,
            source_archive_tree_rows=replay.tree_rows,
            raw_bundle_projection=replay.raw_projection,
            expected_entry="development-calibration",
        )

    validate_live(replay.body, replay.inner, replay.inner_bytes)
    validate_pure(replay.body, replay.inner, replay.inner_bytes)

    observed_old_limit = json.loads(json.dumps(replay.body))
    attestation = observed_old_limit["post_output_physical_gate_attestation"]
    assert attestation["systemd_memory_max_bytes"] == (
        qualification.OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2
    )
    observed_old_limit["unit"]["effective_properties"]["MemoryMax"] = (
        auditor.MAXIMUM_ESTIMATED_RSS_BYTES
    )
    changed_show_rows = 0
    for index, command in enumerate(
        observed_old_limit["launcher"]["systemctl_commands"]
    ):
        if not (
            command["purpose"].startswith("entry-show-")
            or command["purpose"].startswith("terminal-show-")
        ):
            continue
        argv, purpose, returncode, stdout, stderr = (
            qualification._verify_command_evidence_v2(
                command, name=f"old-limit systemctl[{index}]"
            )
        )
        exact_operational_row = (
            "MemoryMax="
            + str(qualification.OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2)
            + "\n"
        ).encode("ascii")
        exact_old_limit_row = (
            "MemoryMax=" + str(auditor.MAXIMUM_ESTIMATED_RSS_BYTES) + "\n"
        ).encode("ascii")
        assert stdout.count(exact_operational_row) == 1
        changed_stdout = stdout.replace(exact_operational_row, exact_old_limit_row)
        command.clear()
        command.update(
            qualification._command_evidence_v2(
                argv=argv,
                purpose=purpose,
                returncode=returncode,
                stdout=changed_stdout,
                stderr=stderr,
            )
        )
        changed_show_rows += 1
    assert changed_show_rows == 3
    assert observed_old_limit["launcher"]["systemd_argv"] == (
        replay.body["launcher"]["systemd_argv"]
    )
    assert observed_old_limit["post_output_physical_gate_attestation"] == (
        replay.body["post_output_physical_gate_attestation"]
    )
    for validator, detail in (
        (validate_live, "outer unit effective properties drifted"),
        (validate_pure, "pure outer effective service properties drifted"),
    ):
        with pytest.raises(ValueError, match=detail):
            validator(observed_old_limit, replay.inner, replay.inner_bytes)

    inner_splice = json.loads(json.dumps(replay.inner))
    for name in ("runtime_entry_evidence", "runtime_exit_evidence"):
        inner_splice[name]["cgroup"]["cgroup_procs"] = [replay.main_pid + 1]
    qualification._validate_development_calibration_artifact_v2(
        inner_splice, replay.attempt
    )
    inner_splice_bytes = qualification._canonical_inner_artifact_v2(inner_splice)
    inner_splice_sha = hashlib.sha256(inner_splice_bytes).hexdigest()
    inner_bound_outer = json.loads(json.dumps(replay.body))
    inner_bound_outer["inner_artifact_cas"] = qualification._cas_projection_v2(
        inner_splice_sha,
        len(inner_splice_bytes),
        replay.attempt.attempt_manifest_root,
    )
    inner_bound_outer["stdout"] = {
        "sha256": inner_splice_sha,
        "size_bytes": len(inner_splice_bytes),
    }
    inner_bound_outer["post_output_physical_gate_attestation"] = (
        qualification._expected_post_output_gate_attestation_v2(
            replay.attempt,
            inner_splice,
            inner_splice_bytes,
            expected_entry="development-calibration",
            child_argv_sha256=inner_bound_outer["unit"]["child_argv_sha256"],
            owned_invocation_id=inner_bound_outer["unit"]["owned_invocation_id"],
        )
    )
    for validator in (validate_live, validate_pure):
        with pytest.raises(ValueError, match="membership|transcript"):
            validator(inner_bound_outer, inner_splice, inner_splice_bytes)

    transcript_splice = json.loads(json.dumps(replay.body))
    entry_row = transcript_splice["launcher"]["systemctl_commands"][1]
    spliced_entry = _systemd_show_bytes(
        replay.attempt,
        replay.source_tree,
        marker=transcript_splice["unit"]["ownership_marker"],
        invocation=transcript_splice["unit"]["invocation_id"],
        load="loaded",
        active="active",
        sub="running",
        main_pid=replay.main_pid + 1,
        cgroup=transcript_splice["unit"]["cgroup_path"],
    )
    transcript_splice["launcher"]["systemctl_commands"][1] = (
        qualification._command_evidence_v2(
            argv=entry_row["argv"],
            purpose="entry-show-0",
            returncode=0,
            stdout=spliced_entry,
            stderr=b"",
        )
    )
    for validator in (validate_live, validate_pure):
        with pytest.raises(ValueError, match="membership|transcript"):
            validator(transcript_splice, replay.inner, replay.inner_bytes)

    child_root_splice = json.loads(json.dumps(replay.inner))
    child_root_splice["runtime_entry_evidence"]["interpreter_runtime"][
        "preopened_source_snapshot_root"
    ] = "phase2b_formal_wire_covert_preopened_sources_v2_" + "d" * 64
    qualification._validate_development_calibration_artifact_v2(
        child_root_splice, replay.attempt
    )
    child_root_splice_bytes = qualification._canonical_inner_artifact_v2(
        child_root_splice
    )
    child_root_splice_sha = hashlib.sha256(child_root_splice_bytes).hexdigest()
    child_root_bound_outer = json.loads(json.dumps(replay.body))
    child_root_bound_outer["inner_artifact_cas"] = (
        qualification._cas_projection_v2(
            child_root_splice_sha,
            len(child_root_splice_bytes),
            replay.attempt.attempt_manifest_root,
        )
    )
    child_root_bound_outer["stdout"] = {
        "sha256": child_root_splice_sha,
        "size_bytes": len(child_root_splice_bytes),
    }
    child_root_bound_outer["post_output_physical_gate_attestation"] = (
        qualification._expected_post_output_gate_attestation_v2(
            replay.attempt,
            child_root_splice,
            child_root_splice_bytes,
            expected_entry="development-calibration",
            child_argv_sha256=child_root_bound_outer["unit"][
                "child_argv_sha256"
            ],
            owned_invocation_id=child_root_bound_outer["unit"][
                "owned_invocation_id"
            ],
        )
    )
    for validator in (validate_live, validate_pure):
        with pytest.raises(ValueError, match="preopened|identity|membership"):
            validator(
                child_root_bound_outer,
                child_root_splice,
                child_root_splice_bytes,
            )

    child_map_splice = json.loads(json.dumps(replay.body))
    nonmain_relative = next(
        relative
        for relative in sorted(
            child_map_splice["child_source_bootstrap_preopened_snapshots"]
        )
        if relative != qualification._BOOTSTRAP_MAIN_RELATIVE_PATH_V2
    )
    child_map_splice["child_source_bootstrap_preopened_snapshots"][
        nonmain_relative
    ]["inode"] += 1
    child_map_splice["child_source_bootstrap_preopened_snapshots"][
        nonmain_relative
    ]["mtime_nanoseconds"] += 1
    child_map_splice["child_source_bootstrap_preopened_snapshots_root"] = (
        qualification.stable_hash(
            child_map_splice["child_source_bootstrap_preopened_snapshots"],
            prefix="phase2b_formal_wire_covert_preopened_sources_v2_",
        )
    )
    for validator in (validate_live, validate_pure):
        with pytest.raises(ValueError, match="preopened"):
            validator(child_map_splice, replay.inner, replay.inner_bytes)

    child_main = (
        replay.source_tree / qualification._BOOTSTRAP_MAIN_RELATIVE_PATH_V2
    )
    original_payload = child_main.read_bytes()
    original_stat = child_main.stat(follow_symlinks=False)

    os.utime(
        child_main,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000_000),
    )
    assert child_main.stat(follow_symlinks=False).st_mtime_ns != original_stat.st_mtime_ns
    for validator in (validate_live, validate_pure):
        with pytest.raises(ValueError):
            validator(replay.body, replay.inner, replay.inner_bytes)
    os.utime(
        child_main,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    retained_original = child_main.with_name(child_main.name + ".retained-original")
    child_main.rename(retained_original)
    child_main.write_bytes(original_payload)
    os.utime(
        child_main,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert child_main.stat(follow_symlinks=False).st_ino != original_stat.st_ino
    for validator in (validate_live, validate_pure):
        with pytest.raises(ValueError):
            validator(replay.body, replay.inner, replay.inner_bytes)
    child_main.unlink()
    retained_original.rename(child_main)

    child_main.write_bytes(original_payload + b"\nsplice")
    for validator in (validate_live, validate_pure):
        with pytest.raises(ValueError):
            validator(replay.body, replay.inner, replay.inner_bytes)
    child_main.write_bytes(original_payload)
    os.utime(
        child_main,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    live_source = inspect.getsource(
        qualification._validate_outer_supervisor_result_v2
    )
    pure_source = inspect.getsource(
        qualification._validate_outer_supervisor_semantics_pure_v2
    )
    assert (
        'inner_entry["cgroup"].get("cgroup_procs") != [entry_main_pid]'
        in live_source
    )
    assert (
        'inner_entry["cgroup"].get("cgroup_procs") != [entry_main_pid]'
        in pure_source
    )
    producer_source = inspect.getsource(
        qualification._run_phase2b_formal_wire_covert_outer_attempt_impl_v2
    )
    assert producer_source.count(
        "_replay_extracted_child_preopened_sources_v2("
    ) == 2
    assert "post_child_snapshots_root" in producer_source
    assert "child preopened source snapshots drifted during execution" in producer_source


@pytest.mark.parametrize(
    "emit_failure_stderr",
    (False, True),
    ids=(
        "terminal-status86-empty-stderr",
        "bounded-fail-resource-stderr-capture",
    ),
)
def test_outer_runner_canonical_stdout_then_tail_exit86_publishes_only_failure_v4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    emit_failure_stderr: bool,
) -> None:
    profile_bytes = b"synthetic-tail-profile\n"
    attempt_bytes = b"synthetic-tail-attempt\n"
    source_archive_bytes = b"synthetic-source-archive"
    runtime_bytes = b"synthetic-runtime-manifest"
    rfc_bytes = b"synthetic-rfc-receipt"
    pinned_source = b"# synthetic pinned qualification source\n"
    pinned_source_sha = hashlib.sha256(pinned_source).hexdigest()
    attempt = replace(
        _attempt(),
        source_archive_root=(
            "sha256:" + hashlib.sha256(source_archive_bytes).hexdigest()
        ),
        source_archive_size_bytes=len(source_archive_bytes),
        qualification_source_sha256=pinned_source_sha,
        qualification_config_sha256=hashlib.sha256(profile_bytes).hexdigest(),
        qualification_lock_sha256=qualification.DEPENDENCY_LOCK_SHA256,
        runtime_environment_manifest_root=(
            "sha256:" + hashlib.sha256(runtime_bytes).hexdigest()
        ),
        runtime_environment_manifest_size_bytes=len(runtime_bytes),
        rfc8785_prerequisite_receipt_root=(
            "sha256:" + hashlib.sha256(rfc_bytes).hexdigest()
        ),
        rfc8785_prerequisite_receipt_sha256=(
            hashlib.sha256(rfc_bytes).hexdigest()
        ),
        rfc8785_prerequisite_receipt_size_bytes=len(rfc_bytes),
    )
    output_root = (tmp_path / "attempts").resolve()
    output_root.mkdir()
    profile_path = (tmp_path / "profile.json").resolve()
    attempt_path = (tmp_path / "attempt.json").resolve()
    source_archive_path = (tmp_path / "source.tar").resolve()
    runtime_path = (tmp_path / "runtime.json").resolve()
    rfc_directory = (tmp_path / "rfc").resolve()
    rfc_directory.mkdir()
    rfc_path = rfc_directory / "receipt.json"
    profile_path.write_bytes(profile_bytes)
    attempt_path.write_bytes(attempt_bytes)
    source_archive_path.write_bytes(source_archive_bytes)
    runtime_path.write_bytes(runtime_bytes)
    rfc_path.write_bytes(rfc_bytes)
    attempt_directory = output_root / (
        "attempt-"
        + hashlib.sha256(attempt.attempt_manifest_root.encode("ascii")).hexdigest()[
            :48
        ]
    )
    source_tree = attempt_directory / "evidence/source-tree"
    dependency_runtime: dict[str, object] = {}
    python_row = _executable_row(str(Path(sys.executable).resolve()))
    launcher_context: dict[str, object] = {}
    invocation_id = "b" * 32
    synthetic_child_snapshots: dict[str, object] = {}
    synthetic_child_root = qualification.stable_hash(
        synthetic_child_snapshots,
        prefix="phase2b_formal_wire_covert_preopened_sources_v2_",
    )
    expected_bootstrap_manifest_root = qualification.stable_hash(
        qualification._source_execution_bootstrap_manifest_v2(pinned_source_sha),
        prefix="phase2b_formal_wire_covert_source_bootstrap_manifest_v2_",
    )
    valid_inner = _valid_development_inner_v6(attempt, source_tree)
    interpreter_runtime = valid_inner["runtime_entry_evidence"][
        "interpreter_runtime"
    ]
    interpreter_runtime["preopened_source_snapshot_root"] = synthetic_child_root
    interpreter_runtime["source_execution_bootstrap_manifest_root"] = (
        expected_bootstrap_manifest_root
    )
    for run, measurement in zip(
        valid_inner["runs_in_order"],
        valid_inner["frozen_static_measurement_candidates_in_order"],
        strict=True,
    ):
        qualification._validate_run_and_measurement_binding_v2(run, measurement)
    qualification._validate_development_calibration_artifact_v2(
        valid_inner, attempt
    )
    validated_interpreter_runtime = (
        qualification._validate_interpreter_runtime_projection_v2(
            interpreter_runtime, "tail-exit86 counterfactual-success inner"
        )
    )
    assert (
        validated_interpreter_runtime["preopened_source_snapshot_root"]
        == synthetic_child_root
    )
    assert (
        validated_interpreter_runtime["source_execution_bootstrap_manifest_root"]
        == expected_bootstrap_manifest_root
    )
    canonical_stdout = qualification._canonical_inner_artifact_v2(valid_inner)
    canonical_stdout_path = (tmp_path / "valid-inner-v6.json").resolve()
    canonical_stdout_path.write_bytes(canonical_stdout)
    exact_stderr = (
        qualification._bounded_post_output_failure_line_v2(
            qualification.Phase2BFormalWireCovertQualificationErrorV2(
                qualification.FAIL_RESOURCE,
                "injected final fresh peak above cooperative tail threshold",
            )
        )
        if emit_failure_stderr
        else b""
    )
    real_popen = subprocess.Popen
    real_terminal_success = qualification._systemd_terminal_success_v2
    terminal_success_observations: list[dict[str, object]] = []

    def observed_terminal_success(
        show: dict[str, str], *, name: str
    ) -> bool:
        observation: dict[str, object] = {
            "name": name,
            "Result": show["Result"],
            "ExecMainCode": show["ExecMainCode"],
            "ExecMainStatus": show["ExecMainStatus"],
        }
        terminal_success_observations.append(observation)
        try:
            result = real_terminal_success(show, name=name)
        except ValueError:
            observation["raised"] = "ValueError"
            raise
        observation["returned"] = result
        return result

    def fake_extract(
        _archive: Path,
        destination: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        main = (
            destination
            / "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py"
        )
        config = (
            destination
            / "config/phase2b_formal_wire_covert_qualification_v2.json"
        )
        lock = (
            destination
            / "config/phase2b_formal_wire_covert_qualification_v2.lock.json"
        )
        main.parent.mkdir(parents=True)
        config.parent.mkdir(parents=True)
        main.write_bytes(pinned_source)
        config.write_bytes(profile_bytes)
        lock.write_bytes(b"synthetic-lock\n")
        return {}

    def fake_systemd_popen(
        argv: list[str],
        **kwargs: object,
    ) -> subprocess.Popen[bytes]:
        assert (
            f"--property=MemoryMax={qualification.OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2}"
            in argv
        )
        launcher_context["unit_name"] = (
            next(item for item in argv if item.startswith("--unit="))
            .removeprefix("--unit=")
            + ".service"
        )
        launcher_context["ownership_marker"] = next(
            item for item in argv if item.startswith("--property=Description=")
        ).removeprefix("--property=Description=")
        stderr_write = (
            f"os.write(2,bytes.fromhex('{exact_stderr.hex()}'))\n"
            if emit_failure_stderr
            else ""
        )
        script = (
            "import os,sys\n"
            "with open(sys.argv[1],'rb') as stream:\n"
            " payload=stream.read()\n"
            "os.write(1,payload)\n"
            + stderr_write
            + "os._exit(86)\n"
        )
        process = real_popen(
            [sys.executable, "-c", script, str(canonical_stdout_path)],
            **kwargs,
        )
        launcher_context["pid"] = process.pid
        return process

    def control_command(
        argv: list[str],
        *,
        purpose: str,
        **_kwargs: object,
    ) -> tuple[dict[str, object], bytes, bytes]:
        if purpose == "prelaunch-absent-check":
            stdout = _systemd_show_bytes(attempt, source_tree)
        elif purpose == "entry-show-0":
            stdout = _systemd_show_bytes(
                attempt,
                source_tree,
                marker=str(launcher_context["ownership_marker"]),
                invocation=invocation_id,
                load="loaded",
                active="active",
                sub="running",
                main_pid=valid_inner["runtime_entry_evidence"]["cgroup"][
                    "cgroup_procs"
                ][0],
                cgroup="/test.slice/qv2.scope",
            )
        elif purpose == "terminal-show-0":
            stdout = _systemd_show_bytes(
                attempt,
                source_tree,
                marker=str(launcher_context["ownership_marker"]),
                invocation=invocation_id,
                load="loaded",
                active="active",
                sub="exited",
                main_pid=0,
                cgroup="",
                result="exit-code",
                overrides={"ExecMainStatus": "86"},
            )
        elif purpose in {"failure-owned-show", "failure-absent-0"}:
            stdout = _systemd_show_bytes(attempt, source_tree)
        else:
            raise AssertionError(f"unexpected systemd control purpose: {purpose}")
        stderr = b""
        evidence = qualification._command_evidence_v2(
            argv=argv,
            purpose=purpose,
            returncode=0,
            stdout=stdout,
            stderr=stderr,
        )
        return evidence, stdout, stderr

    monkeypatch.setattr(
        qualification,
        "load_phase2b_formal_wire_covert_qualification_profile_v2",
        lambda _payload: object(),
    )
    monkeypatch.setattr(
        qualification,
        "load_phase2b_formal_wire_covert_attempt_manifest_v2",
        lambda _payload: attempt,
    )
    monkeypatch.setattr(
        qualification, "_outer_runner_tcb_snapshot_v2", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(qualification, "_safe_extract_source_archive_v2", fake_extract)
    monkeypatch.setattr(
        qualification,
        "_replay_extracted_child_preopened_sources_v2",
        lambda *_args, **_kwargs: (
            synthetic_child_snapshots,
            synthetic_child_root,
        ),
    )
    monkeypatch.setattr(
        qualification,
        "_verify_live_package_tree_is_source_only_v2",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        qualification,
        "_expected_live_file_sha256_v2",
        lambda _attempt: {
            "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py": (
                pinned_source_sha
            )
        },
    )
    monkeypatch.setattr(
        qualification,
        "_verify_dependency_lock_bytes_v2",
        lambda _payload: dependency_runtime,
    )
    monkeypatch.setattr(
        qualification,
        "_verify_rfc8785_prerequisite_receipt_v2",
        lambda *_args, **_kwargs: {"executables": {"python": python_row}},
    )
    monkeypatch.setattr(
        qualification,
        "_copy_rfc8785_raw_execution_bundle_v2",
        lambda **_kwargs: (b"", {}),
    )
    monkeypatch.setattr(
        qualification,
        "_verify_live_executable_v2",
        lambda value, *_args, **_kwargs: value,
    )
    monkeypatch.setattr(
        qualification.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(
        qualification,
        "_executable_projection_for_runner_v2",
        lambda path, **_kwargs: _executable_row(str(path)),
    )
    monkeypatch.setattr(
        qualification,
        "_verify_runtime_environment_manifest_v2",
        lambda *_args, **_kwargs: {
            "dependency_lock_runtime": dependency_runtime
        },
    )
    monkeypatch.setattr(qualification.subprocess, "Popen", fake_systemd_popen)
    monkeypatch.setattr(
        qualification, "_run_bounded_control_command_v2", control_command
    )
    monkeypatch.setattr(
        qualification,
        "_systemd_terminal_success_v2",
        observed_terminal_success,
    )

    terminal_bytes = qualification.run_phase2b_formal_wire_covert_outer_attempt_v2(
        entry_kind="development-calibration",
        profile_path=profile_path,
        attempt_manifest_path=attempt_path,
        rfc8785_prerequisite_receipt_path=rfc_path,
        source_archive_path=source_archive_path,
        runtime_environment_manifest_path=runtime_path,
        output_root=output_root,
    )
    terminal = qualification._decode_canonical_transport_object_v2(
        terminal_bytes,
        "tail-exit86 outer terminal",
        maximum_bytes=8 * 1_024 * 1_024,
    )
    assert terminal["schema_version"] == qualification.OUTER_ATTEMPT_FAILURE_SCHEMA_V2
    assert terminal["disposition"] == "ATOMIC_EMPTY_REJECTION"
    assert terminal["primary_failure_stage"] == "INNER_EXECUTION"
    assert terminal["primary_failure_exception_type"] == "ValueError"
    if not emit_failure_stderr:
        assert [
            observation
            for observation in terminal_success_observations
            if observation["name"] == "terminal loop"
        ] == [
            {
                "name": "terminal loop",
                "Result": "exit-code",
                "ExecMainCode": "1",
                "ExecMainStatus": "86",
                "raised": "ValueError",
            }
        ]
    assert terminal["published_accepted_audit_runs"] == 0
    assert terminal["run_roots"] == []
    assert terminal["qualification_receipt_id"] is None
    assert terminal["claims"] == {key: False for key in terminal["claims"]}
    assert terminal["cleanup_verified"] is True
    assert terminal["cleanup"]["launcher_returncode"] == 86
    assert terminal["cleanup"]["term_sent"] is False
    assert terminal["cleanup"]["kill_sent"] is False
    failure_cas = terminal["failure_cas_projection"]
    child_capture = terminal["child_pipe_capture"]
    expected_bundle = qualification._failure_child_pipe_bundle_bytes_v2(
        canonical_stdout, exact_stderr
    )
    bundle_digest = hashlib.sha256(expected_bundle).hexdigest()
    assert child_capture["bundle_cas"] == qualification._cas_projection_v2(
        bundle_digest,
        len(expected_bundle),
        attempt.attempt_manifest_root,
    )
    assert failure_cas["object_count"] == 1
    assert failure_cas["objects_in_lexicographic_digest_order"] == [
        {
            "content_sha256": bundle_digest,
            "object_size_bytes": len(expected_bundle),
        }
    ]
    assert failure_cas["total_regular_bytes"] == len(expected_bundle)
    assert failure_cas["partial_cas_roles_attributed"] is False
    assert not (attempt_directory / "evidence/inner.json").exists()
    assert not (attempt_directory / "evidence/outer-supervisor.json").exists()
    assert not (attempt_directory / "evidence/mechanics.json").exists()
    cas_objects = list((attempt_directory / "evidence/cas").iterdir())
    assert [path.name for path in cas_objects] == [bundle_digest]
    assert cas_objects[0].read_bytes() == expected_bundle
    evidence_fd = qualification._open_real_directory_fd_v2(
        attempt_directory / "evidence"
    )
    try:
        captured_stdout, captured_stderr, capture_complete = (
            qualification._verify_failure_child_pipe_capture_projection_v2(
                evidence_fd,
                terminal["child_pipe_capture"],
                attempt_manifest_root=attempt.attempt_manifest_root,
                deadline_ns=time.monotonic_ns() + 10_000_000_000,
            )
        )
    finally:
        os.close(evidence_fd)
    assert capture_complete is True
    assert captured_stdout == canonical_stdout
    assert captured_stderr == exact_stderr
    captured_inner = qualification._decode_canonical_transport_object_v2(
        captured_stdout,
        "captured canonical tail-exit86 stdout",
        maximum_bytes=8 * 1_024 * 1_024,
    )
    qualification._validate_development_calibration_artifact_v2(
        captured_inner, attempt
    )
    assert captured_inner["schema_version"].endswith(
        "development-calibration/6"
    )
    assert captured_inner["post_output_physical_gate_contract"] == dict(
        qualification._POST_OUTPUT_PHYSICAL_GATE_CONTRACT_V2
    )
    assert (
        qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
            attempt_directory,
            expected_terminal_bytes=terminal_bytes,
        )
        == terminal_bytes
    )


def test_post_output_failure_line_is_one_bounded_exact_resource_record() -> None:
    line = qualification._bounded_post_output_failure_line_v2(
        ValueError("first\nsecond\r" + "x" * 2_000)
    )
    assert 1 <= len(line) <= 1_024
    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1
    assert line.startswith((qualification.FAIL_RESOURCE + ": ").encode("ascii"))


@pytest.mark.parametrize(
    "case",
    (
        "second_receipt_scope",
        "exit_scope_root",
        "exit_peak_below_receipt",
        "exit_peak_above_auditor_cap",
        "exit_peak_below_current",
        "second_receipt_memory_max",
        "second_receipt_peak_decreases",
        "first_invariant_peak_below_entry",
        "second_invariant_peak_below_first_receipt",
    ),
)
def test_inner_exit_cgroup_crossbinds_both_receipt_scopes_and_physical_peaks(
    case: str,
) -> None:
    _, _, _, _, positive_run, _ = _valid_executed_runtime_bundle_projection()
    positive = json.loads(json.dumps(positive_run))
    negative = json.loads(json.dumps(positive_run))
    negative["mode"] = "KNOWN_BINDING_LEAK"
    runs = [positive, negative]
    issuance = positive["audit"]["receipt_issuance"][
        "complete_runtime_projection"
    ]
    exit_cgroup = {
        "cgroup_path": issuance["unified_cgroup_membership"],
        "unified_cgroup_memory_scope_root": (
            issuance["unified_cgroup_memory_scope_root"]
        ),
        "unified_cgroup_leaf_device": issuance["unified_cgroup_leaf_device"],
        "unified_cgroup_leaf_inode": issuance["unified_cgroup_leaf_inode"],
        "memory_current": issuance[
            "physical_memory_current_bytes_before_receipt_finalization"
        ],
        "memory_peak": issuance[
            "physical_memory_peak_bytes_before_receipt_finalization"
        ],
        "memory_max": issuance["physical_memory_max_bytes"],
    }
    entry_cgroup = json.loads(json.dumps(exit_cgroup))
    qualification._validate_receipt_physical_scope_against_exit_cgroup_v2(
        runs, entry_cgroup, exit_cgroup, context="unit inner exit"
    )
    if case == "second_receipt_scope":
        negative["audit"]["receipt_issuance"]["complete_runtime_projection"][
            "unified_cgroup_leaf_inode"
        ] += 1
    elif case == "exit_scope_root":
        exit_cgroup["unified_cgroup_memory_scope_root"] = (
            "phase2b_formal_covert_unified_cgroup_memory_scope_v1_" + "f" * 64
        )
    elif case == "exit_peak_below_receipt":
        exit_cgroup["memory_peak"] -= 1
    elif case == "exit_peak_above_auditor_cap":
        exit_cgroup["memory_peak"] = auditor.MAXIMUM_ESTIMATED_RSS_BYTES + 1
    elif case == "exit_peak_below_current":
        exit_cgroup["memory_current"] = exit_cgroup["memory_peak"] + 1
    elif case == "second_receipt_memory_max":
        negative["audit"]["receipt_issuance"]["complete_runtime_projection"][
            "physical_memory_max_bytes"
        ] += 1
    elif case == "second_receipt_peak_decreases":
        negative["audit"]["receipt_issuance"]["complete_runtime_projection"][
            "physical_memory_peak_bytes_before_receipt_finalization"
        ] -= 1
    elif case == "first_invariant_peak_below_entry":
        positive["audit"]["invariance"]["complete_runtime_projection"][
            "physical_memory_peak_bytes_at_completion"
        ] -= 1
    elif case == "second_invariant_peak_below_first_receipt":
        negative["audit"]["invariance"]["complete_runtime_projection"][
            "physical_memory_peak_bytes_at_completion"
        ] -= 1
    else:  # pragma: no cover - parameter list is closed above
        raise AssertionError(case)
    with pytest.raises(ValueError):
        qualification._validate_receipt_physical_scope_against_exit_cgroup_v2(
            runs, entry_cgroup, exit_cgroup, context="unit inner exit"
        )


@pytest.mark.parametrize(
    ("event_field", "replacement"),
    (("memory_events", {"x": 0}), ("pids_events", {"y": 0})),
)
def test_inner_runtime_replay_requires_canonical_event_counter_witnesses(
    monkeypatch: pytest.MonkeyPatch,
    event_field: str,
    replacement: dict[str, int],
) -> None:
    attempt = _attempt()
    membership = "/test.slice/qv2-events.scope"
    leaf_device = 11
    leaf_inode = 22
    cgroup = {
        "cgroup_path": membership,
        "cgroup_procs": [12_345],
        "cgroup_stat": {"nr_descendants": 0, "nr_dying_descendants": 0},
        "cpu_affinity": [0],
        "cpu_capacity": 1,
        "cpu_max": None,
        "memory_current": 1_024,
        "memory_events": {"low": 0, "high": 0, "max": 0, "oom": 0, "oom_kill": 0},
        "memory_max": attempt.requested_cgroup_memory_max_bytes,
        "memory_peak": 2_048,
        "pids_current": 1,
        "pids_events": {"max": 0},
        "pids_max": attempt.requested_cgroup_pids_max,
        "unified_cgroup_leaf_device": leaf_device,
        "unified_cgroup_leaf_inode": leaf_inode,
        "unified_cgroup_memory_scope_root": qualification.stable_hash(
            {
                "leaf_device": leaf_device,
                "leaf_inode": leaf_inode,
                "unified_membership": membership,
            },
            prefix="phase2b_formal_covert_unified_cgroup_memory_scope_v1_",
        ),
    }
    selected_signals = [
        number.name for number in qualification.LIVE_SELECTED_DEFAULT_SIGNALS
    ]
    entry = {
        "cgroup": json.loads(json.dumps(cgroup)),
        "interpreter_runtime": {},
        "selected_default_signals": selected_signals,
        "sigblk_hex": "0000000000000000",
        "threads": 1,
    }
    exit_row = {
        "audit_receipts_validated_and_released_in_process": True,
        "cgroup": json.loads(json.dumps(cgroup)),
        "cgroup_events_unchanged_from_entry": True,
        "selected_default_signals": selected_signals,
        "sigblk_hex": "0000000000000000",
        "threads": 1,
    }
    monkeypatch.setattr(
        qualification,
        "_validate_interpreter_runtime_projection_v2",
        lambda value, context: None,
    )
    qualification._validate_runtime_entry_exit_contract_v2(
        entry, exit_row, attempt, context="unit retained inner"
    )
    assert entry["cgroup"]["memory_max"] == (
        qualification.OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2
    )
    assert exit_row["cgroup"]["memory_max"] == (
        qualification.OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2
    )
    for runtime_row in (entry, exit_row):
        runtime_cgroup = runtime_row["cgroup"]
        runtime_cgroup["memory_max"] = auditor.MAXIMUM_ESTIMATED_RSS_BYTES
        with pytest.raises(ValueError, match="cgroup integer evidence drifted"):
            qualification._validate_runtime_entry_exit_contract_v2(
                entry, exit_row, attempt, context="unit retained inner"
            )
        runtime_cgroup["memory_max"] = (
            qualification.OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2
        )
    entry["cgroup"]["memory_max"] = auditor.MAXIMUM_ESTIMATED_RSS_BYTES
    exit_row["cgroup"]["memory_max"] = auditor.MAXIMUM_ESTIMATED_RSS_BYTES
    with pytest.raises(ValueError, match="cgroup integer evidence drifted"):
        qualification._validate_runtime_entry_exit_contract_v2(
            entry, exit_row, attempt, context="unit retained inner"
        )
    entry["cgroup"]["memory_max"] = (
        qualification.OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2
    )
    exit_row["cgroup"]["memory_max"] = (
        qualification.OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2
    )
    entry["cgroup"][event_field] = replacement
    exit_row["cgroup"][event_field] = replacement
    with pytest.raises(ValueError, match="missing required counters"):
        qualification._validate_runtime_entry_exit_contract_v2(
            entry, exit_row, attempt, context="unit retained inner"
        )


def test_production_has_zero_assert_and_no_pickle_pipe_or_process_pool() -> None:
    path = ROOT / "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
    public_receipt_validations = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "validate_formal_covert_audit_receipt_v1"
    ]
    assert len(public_receipt_validations) == 1
    shared_bundle_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_executed_runtime_bundle_v2"
    ]
    assert len(shared_bundle_calls) == 5
    quota_id_validation_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_require_eligibility_quota_receipt_id_v2"
    ]
    assert len(quota_id_validation_calls) == 3
    assert 'quota.get("receipt_id") != receipt_id' not in source
    assert 'resource_plan.get("admissible") is not True' in source
    for method in (
        qualification.Phase2BFormalWireCovertCandidateReceiptV2._issue,
        qualification.Phase2BFormalWireCovertCandidateReceiptV2._validate,
    ):
        method_source = inspect.getsource(method)
        assert (
            method_source.index("_validate_complete_resource_plan_projection_v2")
            < method_source.index(
                "_validate_complete_statistics_runtime_projection_v2"
            )
            < method_source.index(
                "_validate_complete_receipt_issuance_runtime_projection_v2"
            )
            < method_source.index("_validate_executed_runtime_bundle_v2")
        )
    assert "receipt.__post_init__()" not in source
    assert qualification.QUALIFICATION_RECEIPT_DOMAIN == (
        b"HEGEL/PHASE2B/FORMAL_WIRE_COVERT_QUALIFICATION/RECEIPT/V6\x00"
    )
    assert qualification.QUALIFICATION_RECEIPT_PREFIX == (
        "phase2b_formal_wire_covert_qualification_receipt_v6_"
    )
    assert (
        "HEGEL/PHASE2B/FORMAL_WIRE_COVERT_QUALIFICATION/RECEIPT/V5"
        not in source
    )
    assert "phase2b_formal_wire_covert_qualification_receipt_v4_" not in source
    current_schema_tokens = (
        "hegel-machine-phase2b-formal-wire-covert-qualification/6",
        "hegel-machine-phase2b-formal-wire-covert-qualification-dependency-lock/3",
        "hegel-machine-phase2b-formal-wire-covert-development-calibration/6",
        "hegel-machine-phase2b-formal-wire-covert-live-inner-candidate/6",
        "hegel-machine-phase2b-formal-wire-covert-static-shape/3",
        "phase2b_formal_wire_covert_qualification_receipt_v6_",
        "phase2b_formal_wire_covert_resource_plan_v3_",
        "phase2b_formal_wire_covert_observed_resource_plan_v3_",
        "phase2b_formal_wire_covert_invariant_runtime_v4_",
        "phase2b_formal_wire_covert_statistics_runtime_v4_",
        "phase2b_formal_wire_covert_receipt_issuance_runtime_v4_",
    )
    for schema in current_schema_tokens:
        assert schema in source
    for stale in (
        "hegel-machine-phase2b-formal-wire-covert-qualification/5",
        "hegel-machine-phase2b-formal-wire-covert-development-calibration/5",
        "hegel-machine-phase2b-formal-wire-covert-live-inner-candidate/5",
        "hegel-machine-phase2b-formal-wire-covert-qualification/4",
        "hegel-machine-phase2b-formal-wire-covert-development-calibration/4",
        "hegel-machine-phase2b-formal-wire-covert-live-inner-candidate/4",
        "phase2b_formal_wire_covert_qualification_receipt_v5_",
        "phase2b_formal_wire_covert_invariant_runtime_v3_",
        "phase2b_formal_wire_covert_statistics_runtime_v3_",
        "phase2b_formal_wire_covert_receipt_issuance_runtime_v3_",
    ):
        assert stale not in source
    for current_inner_schema in (
        "hegel-machine-phase2b-formal-wire-covert-development-calibration/6",
        "hegel-machine-phase2b-formal-wire-covert-live-inner-candidate/6",
    ):
        assert current_inner_schema.replace("/6", "/5") not in source
    assert source.count(
        "hegel-machine-phase2b-formal-wire-covert-development-calibration/6"
    ) == 5
    assert source.count(
        "hegel-machine-phase2b-formal-wire-covert-live-inner-candidate/6"
    ) == 4
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
            "--user", "kill", "--kill-who=all", "--signal=SIGKILL",
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
    with pytest.raises(ValueError, match="absence confirmation|SIGKILL lacked TERM"):
        qualification._verify_failure_cleanup_projection_v2(
            bad,
            cleanup_verified=True,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
        )


def test_failure_cleanup_kill_option_is_schema_exact_and_systemd249_compatible() -> None:
    help_result = subprocess.run(
        ["systemctl", "kill", "--help"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    legacy_option = "--kill-" + "whom=all"
    assert help_result.returncode == 0
    assert b"--kill-who=WHO" in help_result.stdout
    assert legacy_option.encode("ascii") not in help_result.stdout

    attempt, source_tree, intent = _launch_intent()
    systemctl = intent["executables"]["systemctl"]["realpath"]
    show_argv = qualification._systemd_show_argv_v2(
        systemctl, intent["unit_name"]
    )
    invocation = "a" * 32
    live = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="running",
        main_pid=31337,
        cgroup="/user.slice/hegel-qv2-schema-option.service",
    )
    absent = _systemd_show_bytes(attempt, source_tree)

    def projection(option: str) -> dict[str, object]:
        commands = [
            intent["prelaunch_command"],
            qualification._command_evidence_v2(
                argv=show_argv,
                purpose="failure-owned-show",
                returncode=0,
                stdout=live,
                stderr=b"",
            ),
            qualification._command_evidence_v2(
                argv=[
                    systemctl,
                    "--user",
                    "kill",
                    option,
                    "--signal=SIGTERM",
                    intent["unit_name"],
                ],
                purpose="failure-owned-sigterm",
                returncode=0,
                stdout=b"",
                stderr=b"",
            ),
            qualification._command_evidence_v2(
                argv=show_argv,
                purpose="failure-term-grace-0",
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
        return {
            "cleanup_start_command_index": 1,
            "cleanup_verified": True,
            "commands": commands,
            "drained_stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "drained_stderr_size_bytes": 0,
            "drained_stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "drained_stdout_size_bytes": 0,
            "kill_sent": False,
            "launcher_reaped": True,
            "launcher_returncode": 1,
            "owned_invocation_id": invocation,
            "ownership_marker": intent["ownership_marker"],
            "post_cleanup_unit_absent": True,
            "prelaunch_unit_absent": True,
            "term_sent": True,
            "transcript_root": qualification.stable_hash(
                commands,
                prefix="phase2b_formal_wire_covert_failure_cleanup_v2_",
            ),
            "unit_name": intent["unit_name"],
        }

    current = projection("--kill-who=all")
    legacy = projection(legacy_option)
    qualification._verify_failure_cleanup_projection_v2(
        current,
        cleanup_verified=True,
        attempt=attempt,
        entry_kind="development-calibration",
        source_tree=source_tree,
        launch_intent=intent,
        failure_schema_version=qualification.OUTER_ATTEMPT_FAILURE_SCHEMA_V2,
    )
    qualification._verify_failure_cleanup_projection_v2(
        legacy,
        cleanup_verified=True,
        attempt=attempt,
        entry_kind="development-calibration",
        source_tree=source_tree,
        launch_intent=intent,
        failure_schema_version=qualification._OUTER_ATTEMPT_FAILURE_SCHEMA_V3,
    )
    with pytest.raises(ValueError):
        qualification._verify_failure_cleanup_projection_v2(
            legacy,
            cleanup_verified=True,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
            failure_schema_version=qualification.OUTER_ATTEMPT_FAILURE_SCHEMA_V2,
        )
    with pytest.raises(ValueError):
        qualification._verify_failure_cleanup_projection_v2(
            current,
            cleanup_verified=True,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
            failure_schema_version=qualification._OUTER_ATTEMPT_FAILURE_SCHEMA_V3,
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
    invocation = "a" * 32
    live = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="running",
        main_pid=31337,
        cgroup="/user.slice/hegel-qv2-boundary.service",
    )
    starting = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="activating",
        sub="start",
        main_pid=31337,
        cgroup="/user.slice/hegel-qv2-boundary.service",
    )
    exited = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="exited",
    )
    absent = _systemd_show_bytes(attempt, source_tree)
    commands = [intent["prelaunch_command"]]
    commands.extend(
        qualification._command_evidence_v2(
            argv=show_argv,
            purpose=f"entry-show-{index}",
            returncode=0,
            stdout=starting if index < 7 else live,
            stderr=b"",
        )
        for index in range(8)
    )
    commands.extend(
        qualification._command_evidence_v2(
            argv=show_argv,
            purpose=f"terminal-show-{index}",
            returncode=0,
            stdout=live if index < 31 else exited,
            stderr=b"",
        )
        for index in range(33)
    )
    commands.append(
        qualification._command_evidence_v2(
            argv=[
                intent["executables"]["systemctl"]["realpath"],
                "--user",
                "stop",
                intent["unit_name"],
            ],
            purpose="stop-after-terminal-snapshot",
            returncode=0,
            stdout=b"",
            stderr=b"",
        )
    )
    commands.extend(
        qualification._command_evidence_v2(
            argv=show_argv,
            purpose=f"post-stop-absent-check-{index}",
            returncode=0,
            stdout=exited if index < 7 else absent,
            stderr=b"",
        )
        for index in range(8)
    )
    assert len(commands) == 51

    def projection(rows: list[dict[str, object]], boundary: int) -> dict[str, object]:
        return {
            "cleanup_error_type": "RuntimeError",
            "cleanup_start_command_index": boundary,
            "cleanup_verified": False,
            "commands": rows,
            "external_intervention_required": True,
            "owned_invocation_id": invocation,
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
            purpose="post-stop-absent-check-8",
            returncode=0,
            stdout=absent,
            stderr=b"",
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


def test_unproven_cleanup_v4_accepts_bounded_entry_retries_and_final_failures() -> None:
    attempt, source_tree, intent = _launch_intent()
    show_argv = qualification._systemd_show_argv_v2(
        intent["executables"]["systemctl"]["realpath"], intent["unit_name"]
    )
    invocation = "a" * 32
    live = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="running",
        main_pid=31337,
        cgroup="/user.slice/hegel-qv2-entry-retry.service",
    )
    absent = _systemd_show_bytes(attempt, source_tree)

    def row(
        purpose: str,
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> dict[str, object]:
        return qualification._command_evidence_v2(
            argv=show_argv,
            purpose=purpose,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def projection(
        rows: list[dict[str, object]], boundary: int, owned: str | None
    ) -> dict[str, object]:
        return {
            "cleanup_error_type": "RuntimeError",
            "cleanup_start_command_index": boundary,
            "cleanup_verified": False,
            "commands": rows,
            "external_intervention_required": True,
            "owned_invocation_id": owned,
            "ownership_marker": intent["ownership_marker"],
            "prelaunch_unit_absent": True,
            "transcript_root": qualification.stable_hash(
                rows,
                prefix="phase2b_formal_wire_covert_failure_cleanup_v2_",
            ),
            "unit_name": intent["unit_name"],
        }

    retried = [
        intent["prelaunch_command"],
        row("entry-show-0", returncode=1, stderr=b"not ready"),
        row("entry-show-1", returncode=1, stderr=b"not ready"),
        row("entry-show-2", stdout=live),
        row("terminal-show-0", returncode=1, stderr=b"primary failure"),
        row("failure-owned-show", stdout=absent),
        row("failure-absent-0", stdout=absent),
    ]
    qualification._verify_failure_cleanup_projection_v2(
        projection(retried, 5, invocation),
        cleanup_verified=False,
        attempt=attempt,
        entry_kind="development-calibration",
        source_tree=source_tree,
        launch_intent=intent,
    )

    eight_failed_entries = [intent["prelaunch_command"]] + [
        row(f"entry-show-{index}", returncode=1, stderr=b"not ready")
        for index in range(8)
    ] + [
        row("failure-owned-show", stdout=absent),
        row("failure-absent-0", stdout=absent),
    ]
    qualification._verify_failure_cleanup_projection_v2(
        projection(eight_failed_entries, 9, None),
        cleanup_verified=False,
        attempt=attempt,
        entry_kind="development-calibration",
        source_tree=source_tree,
        launch_intent=intent,
    )

    fifty_failed_entries = [intent["prelaunch_command"]] + [
        row(f"entry-show-{index}", returncode=1, stderr=b"not ready")
        for index in range(50)
    ]
    with pytest.raises(ValueError, match="entry probe count exceeds eight"):
        qualification._verify_failure_cleanup_projection_v2(
            projection(fifty_failed_entries, 51, None),
            cleanup_verified=False,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
        )


def test_unproven_cleanup_v4_rejects_ownership_and_fsm_splices() -> None:
    attempt, source_tree, intent = _launch_intent()
    systemctl = intent["executables"]["systemctl"]["realpath"]
    show_argv = qualification._systemd_show_argv_v2(
        systemctl, intent["unit_name"]
    )
    invocation_a = "a" * 32
    invocation_b = "b" * 32

    def show(
        *,
        marker: str = intent["ownership_marker"],
        invocation: str = invocation_a,
        active: str = "active",
        sub: str = "running",
        main_pid: int = 31337,
        cgroup: str = "/user.slice/hegel-qv2-fsm.service",
    ) -> bytes:
        return _systemd_show_bytes(
            attempt,
            source_tree,
            marker=marker,
            invocation=invocation,
            load="loaded",
            active=active,
            sub=sub,
            main_pid=main_pid,
            cgroup=cgroup,
        )

    absent = _systemd_show_bytes(attempt, source_tree)

    def show_row(
        purpose: str,
        stdout: bytes,
        *,
        returncode: int = 0,
        stderr: bytes = b"",
    ) -> dict[str, object]:
        return qualification._command_evidence_v2(
            argv=show_argv,
            purpose=purpose,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    term_row = qualification._command_evidence_v2(
        argv=[
            systemctl,
            "--user",
            "kill",
            "--kill-who=all",
            "--signal=SIGTERM",
            intent["unit_name"],
        ],
        purpose="failure-owned-sigterm",
        returncode=0,
        stdout=b"",
        stderr=b"",
    )

    def projection(
        rows: list[dict[str, object]], boundary: int, owned: str | None
    ) -> dict[str, object]:
        return {
            "cleanup_error_type": "RuntimeError",
            "cleanup_start_command_index": boundary,
            "cleanup_verified": False,
            "commands": rows,
            "external_intervention_required": True,
            "owned_invocation_id": owned,
            "ownership_marker": intent["ownership_marker"],
            "prelaunch_unit_absent": True,
            "transcript_root": qualification.stable_hash(
                rows,
                prefix="phase2b_formal_wire_covert_failure_cleanup_v2_",
            ),
            "unit_name": intent["unit_name"],
        }

    wrong_marker = [
        intent["prelaunch_command"],
        show_row("entry-show-0", show(marker="wrong-marker")),
        show_row("entry-show-1", show()),
        show_row("failure-owned-show", absent),
    ]
    invocation_splice = [
        intent["prelaunch_command"],
        show_row("entry-show-0", show()),
        show_row("failure-owned-show", show(invocation=invocation_b)),
        show_row("failure-absent-0", absent),
    ]
    failed_then_continued = [
        intent["prelaunch_command"],
        show_row(
            "failure-owned-show", b"", returncode=1, stderr=b"manager failed"
        ),
        show_row("failure-absent-0", absent),
    ]
    nonlive_then_term = [
        intent["prelaunch_command"],
        show_row(
            "failure-owned-show",
            show(active="active", sub="exited", main_pid=0, cgroup=""),
        ),
        term_row,
    ]
    live_then_absence = [
        intent["prelaunch_command"],
        show_row("failure-owned-show", show()),
        show_row("failure-absent-0", absent),
    ]
    term_live_then_absence = [
        intent["prelaunch_command"],
        show_row("failure-owned-show", show()),
        term_row,
        show_row("failure-term-grace-0", show()),
        show_row("failure-absent-0", absent),
    ]
    nonlive_then_absence = [
        intent["prelaunch_command"],
        show_row(
            "failure-owned-show",
            show(active="active", sub="exited", main_pid=0, cgroup=""),
        ),
        show_row("failure-absent-0", absent),
    ]
    projected_invocation_splice = [
        intent["prelaunch_command"],
        show_row("entry-show-0", show()),
        show_row("failure-owned-show", absent),
    ]
    continued_after_absent = [
        intent["prelaunch_command"],
        show_row("failure-owned-show", absent),
        show_row("failure-absent-0", absent),
        show_row("failure-absent-1", absent),
    ]
    successful_entry_stderr = [
        intent["prelaunch_command"],
        show_row("entry-show-0", show(), stderr=b"unexpected"),
        show_row("entry-show-1", show()),
        show_row("failure-owned-show", absent),
    ]
    ready_then_more_entry = [
        intent["prelaunch_command"],
        show_row("entry-show-0", show()),
        show_row("entry-show-1", show()),
        show_row("failure-owned-show", absent),
    ]
    cases = (
        ("wrong-marker", wrong_marker, 3, None, "unowned unit"),
        ("invocation-splice", invocation_splice, 2, invocation_a, "InvocationID changed"),
        ("failed-then-continued", failed_then_continued, 1, None, "did not close"),
        ("nonlive-then-term", nonlive_then_term, 1, invocation_a, "required stop"),
        ("live-then-absence", live_then_absence, 1, invocation_a, "required SIGTERM"),
        (
            "term-live-then-absence",
            term_live_then_absence,
            1,
            invocation_a,
            "required SIGKILL",
        ),
        (
            "nonlive-then-absence",
            nonlive_then_absence,
            1,
            invocation_a,
            "required stop",
        ),
        (
            "projected-invocation-splice",
            projected_invocation_splice,
            2,
            invocation_b,
            "main-prefix InvocationID is not projected exactly",
        ),
        ("continued-after-absent", continued_after_absent, 1, None, "continued after final absence"),
        ("successful-entry-stderr", successful_entry_stderr, 3, None, "did not close"),
        ("ready-then-more-entry", ready_then_more_entry, 3, invocation_a, "after ready"),
    )
    for name, rows, boundary, owned, message in cases:
        try:
            qualification._verify_failure_cleanup_projection_v2(
                projection(rows, boundary, owned),
                cleanup_verified=False,
                attempt=attempt,
                entry_kind="development-calibration",
                source_tree=source_tree,
                launch_intent=intent,
            )
        except ValueError as exc:
            assert message in str(exc), (name, str(exc))
        else:
            pytest.fail(f"unproven cleanup accepted {name} splice")


def test_unproven_cleanup_signal_only_internal_failure_boundary_is_exact() -> None:
    attempt, source_tree, intent = _launch_intent()
    systemctl = intent["executables"]["systemctl"]["realpath"]
    show_argv = qualification._systemd_show_argv_v2(
        systemctl, intent["unit_name"]
    )
    invocation = "a" * 32
    live = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="running",
        main_pid=31337,
        cgroup="/user.slice/hegel-qv2-signal-boundary.service",
    )
    absent = _systemd_show_bytes(attempt, source_tree)

    def show_row(purpose: str, stdout: bytes) -> dict[str, object]:
        return qualification._command_evidence_v2(
            argv=show_argv,
            purpose=purpose,
            returncode=0,
            stdout=stdout,
            stderr=b"",
        )

    def signal_row(signal_name: str) -> dict[str, object]:
        return qualification._command_evidence_v2(
            argv=[
                systemctl,
                "--user",
                "kill",
                "--kill-who=all",
                f"--signal={signal_name}",
                intent["unit_name"],
            ],
            purpose=(
                "failure-owned-sigterm"
                if signal_name == "SIGTERM"
                else "failure-owned-sigkill"
            ),
            returncode=0,
            stdout=b"",
            stderr=b"",
        )

    owned = show_row("failure-owned-show", live)
    term = signal_row("SIGTERM")
    kill = signal_row("SIGKILL")
    term_grace_live = show_row("failure-term-grace-0", live)
    final_absent = show_row("failure-absent-0", absent)

    def projection(
        rows: list[dict[str, object]], *, verified: bool
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "cleanup_start_command_index": 1,
            "cleanup_verified": verified,
            "commands": rows,
            "owned_invocation_id": invocation,
            "ownership_marker": intent["ownership_marker"],
            "prelaunch_unit_absent": True,
            "transcript_root": qualification.stable_hash(
                rows,
                prefix="phase2b_formal_wire_covert_failure_cleanup_v2_",
            ),
            "unit_name": intent["unit_name"],
        }
        if verified:
            result.update(
                drained_stderr_sha256=hashlib.sha256(b"").hexdigest(),
                drained_stderr_size_bytes=0,
                drained_stdout_sha256=hashlib.sha256(b"").hexdigest(),
                drained_stdout_size_bytes=0,
                kill_sent=False,
                launcher_reaped=True,
                launcher_returncode=1,
                post_cleanup_unit_absent=True,
                term_sent=True,
            )
        else:
            result.update(
                cleanup_error_type="TimeoutError",
                external_intervention_required=True,
            )
        return result

    term_only = [intent["prelaunch_command"], owned, term]
    qualification._verify_failure_cleanup_projection_v2(
        projection(term_only, verified=False),
        cleanup_verified=False,
        attempt=attempt,
        entry_kind="development-calibration",
        source_tree=source_tree,
        launch_intent=intent,
    )

    term_then_kill_without_grace = [
        intent["prelaunch_command"], owned, term, kill
    ]
    with pytest.raises(ValueError, match="TERM grace sequence"):
        qualification._verify_failure_cleanup_projection_v2(
            projection(term_then_kill_without_grace, verified=False),
            cleanup_verified=False,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
        )

    kill_only_after_term_grace = [
        intent["prelaunch_command"], owned, term, term_grace_live, kill
    ]
    qualification._verify_failure_cleanup_projection_v2(
        projection(kill_only_after_term_grace, verified=False),
        cleanup_verified=False,
        attempt=attempt,
        entry_kind="development-calibration",
        source_tree=source_tree,
        launch_intent=intent,
    )

    kill_then_later_row_without_grace = [
        *kill_only_after_term_grace,
        final_absent,
    ]
    with pytest.raises(ValueError, match="KILL grace sequence"):
        qualification._verify_failure_cleanup_projection_v2(
            projection(kill_then_later_row_without_grace, verified=False),
            cleanup_verified=False,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
        )

    with pytest.raises(ValueError, match="TERM grace sequence"):
        qualification._verify_failure_cleanup_projection_v2(
            projection(term_only, verified=True),
            cleanup_verified=True,
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


def test_systemctl_fast_success_transcript_is_exact_six_commands() -> None:
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
            argv=show_argv, purpose="terminal-show-1", returncode=0,
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

    retry_commands = [
        commands[0],
        qualification._command_evidence_v2(
            argv=show_argv,
            purpose="entry-show-0",
            returncode=1,
            stdout=b"",
            stderr=b"manager not ready",
        ),
        qualification._command_evidence_v2(
            argv=show_argv,
            purpose="entry-show-1",
            returncode=0,
            stdout=entry,
            stderr=b"",
        ),
        *commands[2:],
    ]
    assert qualification._verify_systemctl_transcript_v2(
        retry_commands,
        systemctl_realpath=systemctl,
        unit_name=intent["unit_name"],
        attempt=attempt,
        source_tree=source_tree,
        ownership_marker=intent["ownership_marker"],
    )[0]["InvocationID"] == invocation

    continued_after_ready = [
        *commands[:2],
        qualification._command_evidence_v2(
            argv=show_argv,
            purpose="entry-show-1",
            returncode=0,
            stdout=entry,
            stderr=b"",
        ),
        *commands[2:],
    ]
    with pytest.raises(ValueError, match="after ready"):
        qualification._verify_systemctl_transcript_v2(
            continued_after_ready,
            systemctl_realpath=systemctl,
            unit_name=intent["unit_name"],
            attempt=attempt,
            source_tree=source_tree,
            ownership_marker=intent["ownership_marker"],
        )

    successful_entry_stderr = list(commands)
    successful_entry_stderr[1] = qualification._command_evidence_v2(
        argv=show_argv,
        purpose="entry-show-0",
        returncode=0,
        stdout=entry,
        stderr=b"unexpected",
    )
    with pytest.raises(ValueError, match="entry probe wrote stderr"):
        qualification._verify_systemctl_transcript_v2(
            successful_entry_stderr,
            systemctl_realpath=systemctl,
            unit_name=intent["unit_name"],
            attempt=attempt,
            source_tree=source_tree,
            ownership_marker=intent["ownership_marker"],
        )

    stop_stdout = list(commands)
    stop_stdout[-2] = qualification._command_evidence_v2(
        argv=[systemctl, "--user", "stop", intent["unit_name"]],
        purpose="stop-after-terminal-snapshot",
        returncode=0,
        stdout=b"unexpected",
        stderr=b"",
    )
    with pytest.raises(ValueError, match="stop evidence drifted"):
        qualification._verify_systemctl_transcript_v2(
            stop_stdout,
            systemctl_realpath=systemctl,
            unit_name=intent["unit_name"],
            attempt=attempt,
            source_tree=source_tree,
            ownership_marker=intent["ownership_marker"],
        )

    absent_then_owned = [
        commands[0],
        qualification._command_evidence_v2(
            argv=show_argv,
            purpose="entry-show-0",
            returncode=0,
            stdout=absent,
            stderr=b"",
        ),
        qualification._command_evidence_v2(
            argv=show_argv,
            purpose="entry-show-1",
            returncode=0,
            stdout=entry,
            stderr=b"",
        ),
        *commands[2:],
    ]
    assert qualification._verify_systemctl_transcript_v2(
        absent_then_owned,
        systemctl_realpath=systemctl,
        unit_name=intent["unit_name"],
        attempt=attempt,
        source_tree=source_tree,
        ownership_marker=intent["ownership_marker"],
    )[0]["InvocationID"] == invocation

    for main_pid, cgroup in (
        (123, ""),
        (0, "/user.slice/hegel-qv2.service"),
    ):
        isolated_entry_drift = list(commands)
        isolated_entry_drift[1] = qualification._command_evidence_v2(
            argv=show_argv,
            purpose="entry-show-0",
            returncode=0,
            stdout=_systemd_show_bytes(
                attempt,
                source_tree,
                marker=intent["ownership_marker"],
                invocation=invocation,
                load="loaded",
                active="active",
                sub="running",
                main_pid=main_pid,
                cgroup=cgroup,
            ),
            stderr=b"",
        )
        with pytest.raises(ValueError, match="lacks PID/cgroup"):
            qualification._verify_systemctl_transcript_v2(
                isolated_entry_drift,
                systemctl_realpath=systemctl,
                unit_name=intent["unit_name"],
                attempt=attempt,
                source_tree=source_tree,
                ownership_marker=intent["ownership_marker"],
            )

    for overrides in (
        {"MainPID": "00"},
        {"ExecMainCode": "01"},
        {"ExecMainCode": "2"},
        {"MemoryMax": "1"},
    ):
        drifted = list(commands)
        drifted[-3] = qualification._command_evidence_v2(
            argv=show_argv,
            purpose="terminal-show-1",
            returncode=0,
            stdout=_systemd_show_bytes(
                attempt,
                source_tree,
                marker=intent["ownership_marker"],
                invocation=invocation,
                load="loaded",
                active="active",
                sub="exited",
                overrides=overrides,
            ),
            stderr=b"",
        )
        with pytest.raises(ValueError):
            qualification._verify_systemctl_transcript_v2(
                drifted,
                systemctl_realpath=systemctl,
                unit_name=intent["unit_name"],
                attempt=attempt,
                source_tree=source_tree,
                ownership_marker=intent["ownership_marker"],
            )


def test_failure_v4_entry_semantic_failures_are_exact_final_boundaries() -> None:
    attempt, source_tree, intent = _launch_intent()
    show_argv = qualification._systemd_show_argv_v2(
        intent["executables"]["systemctl"]["realpath"], intent["unit_name"]
    )
    invocation = "a" * 32
    absent = _systemd_show_bytes(attempt, source_tree)
    nonready = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="activating",
        sub="start",
        main_pid=123,
        cgroup="/user.slice/hegel-qv2-entry-boundary.service",
    )
    ready = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="running",
        main_pid=123,
        cgroup="/user.slice/hegel-qv2-entry-boundary.service",
    )

    def row(index: int, stdout: bytes) -> dict[str, object]:
        return qualification._command_evidence_v2(
            argv=show_argv,
            purpose=f"entry-show-{index}",
            returncode=0,
            stdout=stdout,
            stderr=b"",
        )

    failed_terminal = qualification._command_evidence_v2(
        argv=show_argv,
        purpose="terminal-show-0",
        returncode=1,
        stdout=b"",
        stderr=b"manager probe failed",
    )
    cases = (
        (
            [row(0, nonready), row(1, absent)],
            invocation,
            row(2, ready),
            invocation,
        ),
        (
            [
                row(
                    0,
                    _systemd_show_bytes(
                        attempt,
                        source_tree,
                        marker=intent["ownership_marker"],
                        invocation=invocation,
                        load="loaded",
                        active="active",
                        sub="running",
                        main_pid=123,
                        cgroup="/user.slice/hegel-qv2-entry-boundary.service",
                        overrides={"MainPID": "00123"},
                    ),
                )
            ],
            invocation,
            failed_terminal,
            invocation,
        ),
        (
            [
                row(
                    0,
                    _systemd_show_bytes(
                        attempt,
                        source_tree,
                        marker=intent["ownership_marker"],
                        invocation=invocation,
                        load="loaded",
                        active="active",
                        sub="running",
                        main_pid=123,
                        cgroup="",
                    ),
                )
            ],
            invocation,
            failed_terminal,
            invocation,
        ),
        (
            [
                row(
                    0,
                    _systemd_show_bytes(
                        attempt,
                        source_tree,
                        marker=intent["ownership_marker"],
                        invocation=invocation,
                        load="loaded",
                        active="active",
                        sub="running",
                        main_pid=0,
                        cgroup="/user.slice/hegel-qv2-entry-boundary.service",
                    ),
                )
            ],
            invocation,
            failed_terminal,
            invocation,
        ),
        (
            [
                row(
                    0,
                    _systemd_show_bytes(
                        attempt,
                        source_tree,
                        marker=intent["ownership_marker"],
                        invocation=invocation,
                        load="loaded",
                        active="active",
                        sub="running",
                        main_pid=123,
                        cgroup="/../unsafe",
                    ),
                )
            ],
            invocation,
            failed_terminal,
            invocation,
        ),
        (
            [
                row(
                    0,
                    _systemd_show_bytes(
                        attempt,
                        source_tree,
                        marker=intent["ownership_marker"],
                        invocation=invocation,
                        load="loaded",
                        active="activating",
                        sub="start",
                        main_pid=123,
                        cgroup="/user.slice/hegel-qv2-entry-boundary.service",
                        overrides={"MemoryMax": "1"},
                    ),
                )
            ],
            None,
            row(1, ready),
            invocation,
        ),
    )
    for (
        prefix_rows,
        projected_invocation,
        continuation,
        continued_projection,
    ) in cases:
        commands = [intent["prelaunch_command"], *prefix_rows]
        qualification._verify_failure_cleanup_projection_v2(
            _unproven_cleanup_projection(
                intent,
                commands,
                boundary=len(commands),
                owned_invocation_id=projected_invocation,
            ),
            cleanup_verified=False,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
        )
        continued = [*commands, continuation]
        with pytest.raises(ValueError):
            qualification._verify_failure_cleanup_projection_v2(
                _unproven_cleanup_projection(
                    intent,
                    continued,
                    boundary=len(continued),
                    owned_invocation_id=continued_projection,
                ),
                cleanup_verified=False,
                attempt=attempt,
                entry_kind="development-calibration",
                source_tree=source_tree,
                launch_intent=intent,
            )

    # Exact absence before the first owned observation is a normal startup
    # retry.  Once the same entry sequence reaches an owned ready state, a
    # later real command failure remains a valid /failure/4 final boundary and
    # the InvocationID must stay locked.
    absent_owned_then_failure = [
        intent["prelaunch_command"],
        row(0, absent),
        row(1, ready),
        failed_terminal,
    ]
    qualification._verify_failure_cleanup_projection_v2(
        _unproven_cleanup_projection(
            intent,
            absent_owned_then_failure,
            boundary=len(absent_owned_then_failure),
            owned_invocation_id=invocation,
        ),
        cleanup_verified=False,
        attempt=attempt,
        entry_kind="development-calibration",
        source_tree=source_tree,
        launch_intent=intent,
    )


def test_failure_v4_prestop_semantic_drift_is_exact_final_only() -> None:
    attempt, source_tree, intent = _launch_intent()
    systemctl = intent["executables"]["systemctl"]["realpath"]
    show_argv = qualification._systemd_show_argv_v2(
        systemctl, intent["unit_name"]
    )
    invocation = "b" * 32
    entry = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="running",
        main_pid=321,
        cgroup="/user.slice/hegel-qv2-prestop.service",
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

    def show_row(purpose: str, stdout: bytes) -> dict[str, object]:
        return qualification._command_evidence_v2(
            argv=show_argv,
            purpose=purpose,
            returncode=0,
            stdout=stdout,
            stderr=b"",
        )

    base = [
        intent["prelaunch_command"],
        show_row("entry-show-0", entry),
        show_row("terminal-show-0", terminal),
    ]
    stop = qualification._command_evidence_v2(
        argv=[systemctl, "--user", "stop", intent["unit_name"]],
        purpose="stop-after-terminal-snapshot",
        returncode=0,
        stdout=b"",
        stderr=b"",
    )
    for overrides in (
        {"ExecMainCode": "01"},
        {"ExecMainCode": "2"},
        {"ExecMainStatus": "00"},
        {"MainPID": "00"},
        {"MemoryMax": "1"},
    ):
        drift = _systemd_show_bytes(
            attempt,
            source_tree,
            marker=intent["ownership_marker"],
            invocation=invocation,
            load="loaded",
            active="active",
            sub="exited",
            overrides=overrides,
        )
        commands = [*base, show_row("terminal-show-1", drift)]
        qualification._verify_failure_cleanup_projection_v2(
            _unproven_cleanup_projection(
                intent,
                commands,
                boundary=len(commands),
                owned_invocation_id=invocation,
            ),
            cleanup_verified=False,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
        )
        continued = [*commands, stop]
        with pytest.raises(ValueError):
            qualification._verify_failure_cleanup_projection_v2(
                _unproven_cleanup_projection(
                    intent,
                    continued,
                    boundary=len(continued),
                    owned_invocation_id=invocation,
                ),
                cleanup_verified=False,
                attempt=attempt,
                entry_kind="development-calibration",
                source_tree=source_tree,
                launch_intent=intent,
            )


def test_failure_v4_terminal_loop_semantic_drift_is_exact_final_only() -> None:
    attempt, source_tree, intent = _launch_intent()
    systemctl = intent["executables"]["systemctl"]["realpath"]
    show_argv = qualification._systemd_show_argv_v2(
        systemctl, intent["unit_name"]
    )
    invocation = "d" * 32
    entry = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="running",
        main_pid=456,
        cgroup="/user.slice/hegel-qv2-terminal-loop.service",
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

    def show_row(purpose: str, stdout: bytes) -> dict[str, object]:
        return qualification._command_evidence_v2(
            argv=show_argv,
            purpose=purpose,
            returncode=0,
            stdout=stdout,
            stderr=b"",
        )

    base = [intent["prelaunch_command"], show_row("entry-show-0", entry)]
    for overrides in (
        {"ExecMainCode": "01"},
        {"ExecMainCode": "2"},
        {"ExecMainStatus": "00"},
        {"MainPID": "00"},
        {"MemoryMax": "1"},
    ):
        drift = _systemd_show_bytes(
            attempt,
            source_tree,
            marker=intent["ownership_marker"],
            invocation=invocation,
            load="loaded",
            active="active",
            sub="exited",
            overrides=overrides,
        )
        commands = [*base, show_row("terminal-show-0", drift)]
        qualification._verify_failure_cleanup_projection_v2(
            _unproven_cleanup_projection(
                intent,
                commands,
                boundary=len(commands),
                owned_invocation_id=invocation,
            ),
            cleanup_verified=False,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
        )
        with pytest.raises(ValueError):
            qualification._verify_failure_cleanup_projection_v2(
                _unproven_cleanup_projection(
                    intent,
                    [*commands, show_row("terminal-show-1", terminal)],
                    boundary=len(commands) + 1,
                    owned_invocation_id=invocation,
                ),
                cleanup_verified=False,
                attempt=attempt,
                entry_kind="development-calibration",
                source_tree=source_tree,
                launch_intent=intent,
            )


def test_failure_v4_cleanup_show_semantics_and_absence_bound_are_exact() -> None:
    attempt, source_tree, intent = _launch_intent()
    systemctl = intent["executables"]["systemctl"]["realpath"]
    show_argv = qualification._systemd_show_argv_v2(
        systemctl, intent["unit_name"]
    )
    invocation = "c" * 32
    absent = _systemd_show_bytes(attempt, source_tree)
    live = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="running",
        main_pid=789,
        cgroup="/user.slice/hegel-qv2-cleanup-phase.service",
    )
    nonlive = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="exited",
    )
    drift = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="exited",
        overrides={"MemoryMax": "1"},
    )

    def show_row(purpose: str, stdout: bytes) -> dict[str, object]:
        return qualification._command_evidence_v2(
            argv=show_argv,
            purpose=purpose,
            returncode=0,
            stdout=stdout,
            stderr=b"",
        )

    stop = qualification._command_evidence_v2(
        argv=[systemctl, "--user", "stop", intent["unit_name"]],
        purpose="failure-owned-stop",
        returncode=0,
        stdout=b"",
        stderr=b"",
    )

    def signal_row(purpose: str, signal_name: str) -> dict[str, object]:
        return qualification._command_evidence_v2(
            argv=[
                systemctl,
                "--user",
                "kill",
                "--kill-who=all",
                f"--signal={signal_name}",
                intent["unit_name"],
            ],
            purpose=purpose,
            returncode=0,
            stdout=b"",
            stderr=b"",
        )

    reappeared = [
        intent["prelaunch_command"],
        show_row("failure-owned-show", absent),
        show_row("failure-absent-0", nonlive),
    ]
    qualification._verify_failure_cleanup_projection_v2(
        _unproven_cleanup_projection(
            intent, reappeared, boundary=1, owned_invocation_id=None
        ),
        cleanup_verified=False,
        attempt=attempt,
        entry_kind="development-calibration",
        source_tree=source_tree,
        launch_intent=intent,
    )
    with pytest.raises(ValueError, match="reappeared"):
        qualification._verify_failure_cleanup_projection_v2(
            _unproven_cleanup_projection(
                intent,
                [*reappeared, show_row("failure-absent-1", absent)],
                boundary=1,
                owned_invocation_id=None,
            ),
            cleanup_verified=False,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
        )

    drifted = [
        intent["prelaunch_command"],
        show_row("failure-owned-show", nonlive),
        stop,
        show_row("failure-absent-0", drift),
    ]
    qualification._verify_failure_cleanup_projection_v2(
        _unproven_cleanup_projection(
            intent, drifted, boundary=1, owned_invocation_id=invocation
        ),
        cleanup_verified=False,
        attempt=attempt,
        entry_kind="development-calibration",
        source_tree=source_tree,
        launch_intent=intent,
    )
    with pytest.raises(ValueError):
        qualification._verify_failure_cleanup_projection_v2(
            _unproven_cleanup_projection(
                intent,
                [*drifted, show_row("failure-absent-1", absent)],
                boundary=1,
                owned_invocation_id=invocation,
            ),
            cleanup_verified=False,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
        )

    sigterm = signal_row("failure-owned-sigterm", "SIGTERM")
    sigkill = signal_row("failure-owned-sigkill", "SIGKILL")
    # Each continuation is otherwise producer-reachable if the one targeted
    # scalar/path check is removed: a live initial/TERM-grace row requires the
    # corresponding signal, while a non-live KILL-grace row requires stop.
    for bad_live_show in (
        _systemd_show_bytes(
            attempt,
            source_tree,
            marker=intent["ownership_marker"],
            invocation=invocation,
            load="loaded",
            active="active",
            sub="running",
            main_pid=789,
            cgroup="/user.slice/hegel-qv2-cleanup-phase.service",
            overrides={"MainPID": "00789"},
        ),
        _systemd_show_bytes(
            attempt,
            source_tree,
            marker=intent["ownership_marker"],
            invocation=invocation,
            load="loaded",
            active="active",
            sub="running",
            main_pid=789,
            cgroup="/../unsafe",
        ),
    ):
        phase_cases = (
            (
                [
                    intent["prelaunch_command"],
                    show_row("failure-owned-show", bad_live_show),
                ],
                sigterm,
            ),
            (
                [
                    intent["prelaunch_command"],
                    show_row("failure-owned-show", live),
                    sigterm,
                    show_row("failure-term-grace-0", bad_live_show),
                ],
                sigkill,
            ),
        )
        for commands, continuation in phase_cases:
            qualification._verify_failure_cleanup_projection_v2(
                _unproven_cleanup_projection(
                    intent,
                    commands,
                    boundary=1,
                    owned_invocation_id=invocation,
                ),
                cleanup_verified=False,
                attempt=attempt,
                entry_kind="development-calibration",
                source_tree=source_tree,
                launch_intent=intent,
            )
            with pytest.raises(ValueError):
                qualification._verify_failure_cleanup_projection_v2(
                    _unproven_cleanup_projection(
                        intent,
                        [*commands, continuation],
                        boundary=1,
                        owned_invocation_id=invocation,
                    ),
                    cleanup_verified=False,
                    attempt=attempt,
                    entry_kind="development-calibration",
                    source_tree=source_tree,
                    launch_intent=intent,
                )

    bad_nonlive_kill_grace = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="exited",
        main_pid=0,
        cgroup="",
        overrides={"MainPID": "00"},
    )
    kill_grace_bad_commands = [
        intent["prelaunch_command"],
        show_row("failure-owned-show", live),
        sigterm,
        show_row("failure-term-grace-0", live),
        sigkill,
        show_row("failure-kill-grace-0", bad_nonlive_kill_grace),
    ]
    qualification._verify_failure_cleanup_projection_v2(
        _unproven_cleanup_projection(
            intent,
            kill_grace_bad_commands,
            boundary=1,
            owned_invocation_id=invocation,
        ),
        cleanup_verified=False,
        attempt=attempt,
        entry_kind="development-calibration",
        source_tree=source_tree,
        launch_intent=intent,
    )
    with pytest.raises(ValueError):
        qualification._verify_failure_cleanup_projection_v2(
            _unproven_cleanup_projection(
                intent,
                [*kill_grace_bad_commands, stop],
                boundary=1,
                owned_invocation_id=invocation,
            ),
            cleanup_verified=False,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
        )

    eight_present = [
        intent["prelaunch_command"],
        show_row("failure-owned-show", nonlive),
        stop,
        *[
            show_row(f"failure-absent-{index}", nonlive)
            for index in range(8)
        ],
    ]
    qualification._verify_failure_cleanup_projection_v2(
        _unproven_cleanup_projection(
            intent, eight_present, boundary=1, owned_invocation_id=invocation
        ),
        cleanup_verified=False,
        attempt=attempt,
        entry_kind="development-calibration",
        source_tree=source_tree,
        launch_intent=intent,
    )
    with pytest.raises(ValueError, match="exceeds eight"):
        qualification._verify_failure_cleanup_projection_v2(
            _unproven_cleanup_projection(
                intent,
                [*eight_present, show_row("failure-absent-8", nonlive)],
                boundary=1,
                owned_invocation_id=invocation,
            ),
            cleanup_verified=False,
            attempt=attempt,
            entry_kind="development-calibration",
            source_tree=source_tree,
            launch_intent=intent,
        )


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
        if str(path).startswith(".reservation.json.tmp-"):
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
                directory_fd, "reservation.json", b"staging-content"
            )
        assert captured.value.publication_state == "STAGED_TEMP_ONLY"
        residues = qualification._reconcile_nonterminal_atomic_uncertainty_v2(
            captured.value,
            attempt_fd=directory_fd,
            deadline_ns=time.monotonic_ns() + 10_000_000_000,
            known_parent_directories={"": directory_fd},
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
    real_combined = qualification._derive_failure_v4_retained_projection_at_fd_v2
    temporary_unlink_failures = 0
    prelink_injected = False
    prospective_returned_at: list[int] = []
    combined_scans = 0

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

    def tracking_combined_scan(*args: object, **kwargs: object):
        nonlocal combined_scans
        combined_scans += 1
        return real_combined(*args, **kwargs)

    monkeypatch.setattr(
        qualification,
        "verify_phase2b_formal_wire_covert_outer_attempt_directory_v2",
        tracking_attempt_verifier,
    )
    monkeypatch.setattr(
        qualification,
        "_derive_failure_v4_retained_projection_at_fd_v2",
        tracking_combined_scan,
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
    assert combined_scans == qualification._FINALIZATION_TREE_SCAN_PASSES_V2 == 2
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
    assert combined_scans == 3


@pytest.mark.parametrize(
    ("residue_parent", "uncertainty_point"),
    (
        ("rfc8785-raw-bundle", "prelink"),
        ("rfc8785-raw-bundle", "postlink"),
        ("cas", "prelink"),
        ("cas", "postlink"),
    ),
)
def test_rfc_or_cas_atomic_uncertainty_uses_retained_parent_and_finalizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    residue_parent: str,
    uncertainty_point: str,
) -> None:
    profile_bytes = b"synthetic-profile-bytes\n"
    attempt_bytes = b"synthetic-attempt-bytes\n"
    attempt = replace(
        _attempt(),
        qualification_config_sha256=hashlib.sha256(profile_bytes).hexdigest(),
        qualification_lock_sha256=qualification.DEPENDENCY_LOCK_SHA256,
    )
    output_root = (tmp_path / "attempts").resolve()
    output_root.mkdir()
    source_archive = (tmp_path / "source.tar").resolve()
    runtime_manifest = (tmp_path / "runtime.json").resolve()
    source_archive.write_bytes(b"a")
    runtime_manifest.write_bytes(b"r")
    source_rfc_directory = (tmp_path / "source-rfc").resolve()
    source_rfc_directory.mkdir()
    source_rfc_receipt = source_rfc_directory / "receipt.json"
    source_rfc_receipt.write_bytes(b"rfc-receipt")
    real_write_all = qualification._write_all_fd_v2
    real_unlink = qualification.os.unlink
    real_remove = qualification.os.remove
    real_attempt_verifier = (
        qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2
    )
    real_combined = qualification._derive_failure_v4_retained_projection_at_fd_v2
    prelink_injected = False
    combined_scans = 0
    atomic_payload = b"retained RFC raw object"
    atomic_target_name = (
        "command-000-python-version.stdout.bin"
        if residue_parent == "rfc8785-raw-bundle"
        else hashlib.sha256(atomic_payload).hexdigest()
    )

    def read_input(
        path: Path, *, maximum_bytes: int, name: str, deadline_ns: int | None = None
    ) -> bytes:
        del path, maximum_bytes, deadline_ns
        if name in {"runner profile", "extracted qualification profile/config"}:
            return profile_bytes
        if name == "runner attempt manifest":
            return attempt_bytes
        if name == "extracted qualification dependency lock":
            return b"synthetic-lock"
        if name == "runner RFC-8785 prerequisite receipt":
            return b"rfc-receipt"
        if name == "runner runtime environment manifest":
            return b"runtime"
        raise AssertionError(f"unexpected preflight read: {name}")

    def fake_extract(
        archive: Path,
        destination: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        del archive, kwargs
        main = (
            destination
            / "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py"
        )
        main.parent.mkdir(parents=True)
        main.write_bytes(b"synthetic qualification source")
        return {}

    def persist_raw_temporary(
        path: object, *args: object, **kwargs: object
    ) -> None:
        if str(path).startswith(f".{atomic_target_name}.tmp-"):
            raise OSError("injected persistent RFC raw temp cleanup failure")
        real_unlink(path, *args, **kwargs)

    def persist_raw_remove(
        path: object, *args: object, **kwargs: object
    ) -> None:
        if str(path).startswith(f".{atomic_target_name}.tmp-"):
            raise OSError("injected persistent RFC raw temp reconcile failure")
        real_remove(path, *args, **kwargs)

    def optionally_fail_raw_write(
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
            and descriptor_name.startswith(f".{atomic_target_name}.tmp-")
            and not prelink_injected
        ):
            prelink_injected = True
            real_write_all(descriptor, payload, deadline_ns=deadline_ns)
            raise OSError("injected RFC raw failure before link")
        real_write_all(descriptor, payload, deadline_ns=deadline_ns)

    def fail_copy_from_preopened_raw_parent(**kwargs: object):
        destination = kwargs["destination_directory"]
        destination_fd = kwargs["destination_directory_fd"]
        assert isinstance(destination, Path)
        assert type(destination_fd) is int
        assert (
            os.fstat(destination_fd).st_dev,
            os.fstat(destination_fd).st_ino,
        ) == (
            destination.stat(follow_symlinks=False).st_dev,
            destination.stat(follow_symlinks=False).st_ino,
        )
        atomic_parent_fd = destination_fd
        owned_atomic_parent_fd: int | None = None
        if residue_parent == "cas":
            owned_atomic_parent_fd = qualification._open_real_directory_fd_v2(
                destination.parent / "cas"
            )
            atomic_parent_fd = owned_atomic_parent_fd
        try:
            qualification._atomic_write_new_file_at_fd_v2(
                atomic_parent_fd,
                atomic_target_name,
                atomic_payload,
                deadline_ns=kwargs["deadline_ns"],
            )
        finally:
            if owned_atomic_parent_fd is not None:
                os.close(owned_atomic_parent_fd)
        raise AssertionError("RFC raw uncertainty injection did not fail")

    def tracking_combined(*args: object, **kwargs: object):
        nonlocal combined_scans
        combined_scans += 1
        return real_combined(*args, **kwargs)

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
    monkeypatch.setattr(
        qualification, "_outer_runner_tcb_snapshot_v2", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(qualification, "_sha256_file_v2", lambda *args, **kwargs: "0" * 64)
    monkeypatch.setattr(qualification, "_safe_extract_source_archive_v2", fake_extract)
    synthetic_child_snapshots: dict[str, object] = {}
    synthetic_child_root = qualification.stable_hash(
        synthetic_child_snapshots,
        prefix="phase2b_formal_wire_covert_preopened_sources_v2_",
    )
    monkeypatch.setattr(
        qualification,
        "_replay_extracted_child_preopened_sources_v2",
        lambda *args, **kwargs: (
            synthetic_child_snapshots,
            synthetic_child_root,
        ),
    )
    monkeypatch.setattr(
        qualification, "_verify_live_package_tree_is_source_only_v2", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        qualification,
        "_expected_live_file_sha256_v2",
        lambda value: {
            "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py": "0" * 64
        },
    )
    monkeypatch.setattr(
        qualification, "_verify_dependency_lock_bytes_v2", lambda payload: {}
    )
    monkeypatch.setattr(
        qualification,
        "_verify_rfc8785_prerequisite_receipt_v2",
        lambda *args, **kwargs: {"raw_execution": {}},
    )
    monkeypatch.setattr(
        qualification,
        "_copy_rfc8785_raw_execution_bundle_v2",
        fail_copy_from_preopened_raw_parent,
    )
    monkeypatch.setattr(qualification.os, "unlink", persist_raw_temporary)
    monkeypatch.setattr(qualification.os, "remove", persist_raw_remove)
    monkeypatch.setattr(qualification, "_write_all_fd_v2", optionally_fail_raw_write)
    monkeypatch.setattr(
        qualification,
        "_derive_failure_v4_retained_projection_at_fd_v2",
        tracking_combined,
    )
    monkeypatch.setattr(
        qualification,
        "verify_phase2b_formal_wire_covert_outer_attempt_directory_v2",
        lambda *args, **kwargs: real_attempt_verifier(*args, **kwargs),
    )

    terminal_bytes = qualification.run_phase2b_formal_wire_covert_outer_attempt_v2(
        entry_kind="development-calibration",
        profile_path=(tmp_path / "profile.json").resolve(),
        attempt_manifest_path=(tmp_path / "attempt.json").resolve(),
        rfc8785_prerequisite_receipt_path=source_rfc_receipt,
        source_archive_path=source_archive,
        runtime_environment_manifest_path=runtime_manifest,
        output_root=output_root,
    )
    terminal = qualification._decode_canonical_transport_object_v2(
        terminal_bytes, "RFC raw failure terminal", maximum_bytes=8 * 1_024 * 1_024
    )
    attempt_directory = output_root / (
        "attempt-"
        + hashlib.sha256(attempt.attempt_manifest_root.encode("ascii")).hexdigest()[:48]
    )
    assert terminal["disposition"] == "ATOMIC_EMPTY_REJECTION"
    assert terminal["primary_failure_stage"] == "PREFLIGHT"
    assert terminal["claims"] == {
        key: False for key in terminal["claims"]
    }
    assert len(terminal["managed_atomic_residues"]) == 1
    residue = terminal["managed_atomic_residues"][0]
    assert residue["relative_parent"] == "evidence/" + residue_parent
    assert residue["target_name"] == atomic_target_name
    assert residue["state"] == (
        "STAGED_TEMP_ONLY"
        if uncertainty_point == "prelink"
        else "TARGET_AND_MANAGED_TEMP_HARDLINK"
    )
    assert combined_scans == qualification._FINALIZATION_TREE_SCAN_PASSES_V2 == 2
    assert (attempt_directory / "terminal.json").read_bytes() == terminal_bytes
    assert (
        qualification.verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
            attempt_directory,
            expected_terminal_bytes=terminal_bytes,
        )
        == terminal_bytes
    )
    assert combined_scans == 3


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
            known_parent_directories={"": cas_fd},
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
            known_parent_directories={"": cas_fd},
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


def test_cleanup_closed_pipes_cannot_upgrade_false_capture_complete(
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
    launcher.wait(timeout=5)
    qualification._close_process_pipes_v2(
        launcher, name="closed-pipe capture-proof regression"
    )
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
    capture_state = {
        "capture_complete": False,
        "capture_truncated": False,
    }
    with pytest.raises(RuntimeError, match="lack prior EOF/reap capture proof"):
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
            commands=[intent["prelaunch_command"]],
            cleanup_start_command_index=1,
            stdout_capture=bytearray(),
            stderr_capture=bytearray(),
            capture_state=capture_state,
        )
    assert capture_state == {
        "capture_complete": False,
        "capture_truncated": False,
    }


def test_cleanup_helper_uses_systemd249_kill_option_for_term_and_stubborn_kill(
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
                "import os,signal,time;"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "os.write(1,b'ready\\n');time.sleep(60)"
            ),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert launcher.stdout is not None
    prefix = os.read(launcher.stdout.fileno(), len(b"ready\n"))
    assert prefix == b"ready\n"
    stdout_capture = bytearray(prefix)
    stderr_capture = bytearray()
    capture_state = {
        "capture_complete": False,
        "capture_truncated": False,
    }
    commands = [intent["prelaunch_command"]]
    invocation = "b" * 32
    live = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="running",
        main_pid=launcher.pid,
        cgroup="/user.slice/hegel-qv2-stubborn.service",
    )
    nonlive = _systemd_show_bytes(
        attempt,
        source_tree,
        marker=intent["ownership_marker"],
        invocation=invocation,
        load="loaded",
        active="active",
        sub="exited",
        main_pid=0,
        cgroup="",
    )
    absent = _systemd_show_bytes(attempt, source_tree)
    observed_kill_argv: list[list[str]] = []

    def manager_command(
        argv: list[str], *, purpose: str, **kwargs: object
    ) -> tuple[dict[str, object], bytes, bytes]:
        del kwargs
        stdout = b""
        if purpose == "failure-owned-show" or purpose.startswith(
            "failure-term-grace-"
        ):
            stdout = live
        elif purpose.startswith("failure-kill-grace-"):
            stdout = nonlive
        elif purpose.startswith("failure-absent-"):
            stdout = absent
        elif purpose == "failure-owned-sigterm":
            observed_kill_argv.append(argv)
            os.killpg(launcher.pid, signal.SIGTERM)
        elif purpose == "failure-owned-sigkill":
            observed_kill_argv.append(argv)
            os.killpg(launcher.pid, signal.SIGKILL)
        elif purpose != "failure-owned-stop":
            raise AssertionError(f"unexpected cleanup purpose: {purpose}")
        return (
            qualification._command_evidence_v2(
                argv=argv,
                purpose=purpose,
                returncode=0,
                stdout=stdout,
                stderr=b"",
            ),
            stdout,
            b"",
        )

    monkeypatch.setattr(
        qualification, "_run_bounded_control_command_v2", manager_command
    )
    monkeypatch.setattr(qualification.time, "sleep", lambda _: None)
    cleanup = qualification._cleanup_transient_attempt_v2(
        launcher=launcher,
        systemctl_realpath=intent["executables"]["systemctl"]["realpath"],
        unit_name=intent["unit_name"],
        ownership_marker=intent["ownership_marker"],
        owned_invocation_id=invocation,
        prelaunch_absent_verified=True,
        attempt=attempt,
        source_tree=source_tree,
        deadline_ns=time.monotonic_ns() + 30_000_000_000,
        commands=commands,
        cleanup_start_command_index=1,
        stdout_capture=stdout_capture,
        stderr_capture=stderr_capture,
        capture_state=capture_state,
    )
    assert observed_kill_argv == [
        [
            intent["executables"]["systemctl"]["realpath"],
            "--user",
            "kill",
            "--kill-who=all",
            "--signal=SIGTERM",
            intent["unit_name"],
        ],
        [
            intent["executables"]["systemctl"]["realpath"],
            "--user",
            "kill",
            "--kill-who=all",
            "--signal=SIGKILL",
            intent["unit_name"],
        ],
    ]
    assert cleanup["term_sent"] is True
    assert cleanup["kill_sent"] is True
    assert cleanup["launcher_reaped"] is True
    assert cleanup["post_cleanup_unit_absent"] is True
    assert capture_state["capture_complete"] is True
    assert bytes(stdout_capture) == prefix
    assert bytes(stderr_capture) == b""
    assert launcher.poll() == -signal.SIGKILL
    assert launcher.stdout.closed
    assert launcher.stderr is not None and launcher.stderr.closed
    qualification._verify_failure_cleanup_projection_v2(
        cleanup,
        cleanup_verified=True,
        attempt=attempt,
        entry_kind="development-calibration",
        source_tree=source_tree,
        launch_intent=intent,
    )


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
    feature_projection = auditor._feature_cardinality_projection(decoded)
    assert qualification._validate_fixture_feature_capacity_projection_v2(
        mode, feature_projection
    ) is feature_projection
    expected_feature_projection = (
        qualification._FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"][mode][
            "projections"
        ][1]
    )
    assert feature_projection.projected_concrete_feature_count == (
        expected_feature_projection["projected_concrete_feature_count"]
    )
    assert feature_projection.projected_raw_semantic_feature_count == (
        expected_feature_projection["projected_raw_semantic_feature_count"]
    )
    assert feature_projection.projection_root == expected_feature_projection[
        "projection_root"
    ]
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


def test_resource_only_decode_helper_fresh_preflight_precedes_each_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDecoded:
        def __init__(self, envelope: bytes) -> None:
            self.envelope = envelope

    events: list[str] = []

    def preflight(rows: int) -> object:
        events.append(f"preflight:{rows}")
        return object()

    def decode(envelope: bytes) -> FakeDecoded:
        events.append("decode")
        return FakeDecoded(envelope)

    def validate(item: FakeDecoded, adapter: object) -> None:
        events.append("validate")

    def deep(item: FakeDecoded) -> int:
        events.append("deep")
        return 1

    adapter = SimpleNamespace(decode_envelope=decode)
    monkeypatch.setattr(auditor, "DecodedFormalEnvelopeV1", FakeDecoded)
    monkeypatch.setattr(auditor, "_preflight_decode_memory", preflight)
    monkeypatch.setattr(auditor, "_validate_decoded", validate)
    monkeypatch.setattr(auditor, "_deep_object_graph_bytes", deep)
    first = _decode_batch_with_production_guards(adapter, (b"a", b"b"))
    second = _decode_batch_with_production_guards(adapter, (b"c",))
    assert tuple(item.envelope for item in first) == (b"a", b"b")
    assert tuple(item.envelope for item in second) == (b"c",)
    assert events == [
        "preflight:2",
        "decode",
        "validate",
        "deep",
        "decode",
        "validate",
        "deep",
        "preflight:1",
        "decode",
        "validate",
        "deep",
    ]


@pytest.mark.slow
@pytest.mark.parametrize("mode", qualification.LIVE_MODE_ORDER)
def test_full1024_resource_only_projection_is_exact_and_admissible(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed_receipts_before = dict(auditor._IN_PROCESS_EXECUTED_RECEIPTS)
    receipt_reservations_before = set(auditor._IN_PROCESS_RECEIPT_RESERVATIONS)
    receipt_lifecycle_calls: list[str] = []

    def forbid_receipt_lifecycle(*args: object, **kwargs: object) -> None:
        receipt_lifecycle_calls.append("called")
        pytest.fail("resource-only projection must not construct or touch a receipt")

    for name in (
        "_construct_formal_covert_audit_receipt_v1",
        "_register_preconstructed_formal_covert_audit_receipt_v1",
        "_validate_pending_formal_covert_audit_receipt_v1",
        "_commit_pending_formal_covert_audit_receipt_v1",
        "validate_formal_covert_audit_receipt_v1",
        "release_formal_covert_audit_receipt_v1",
    ):
        monkeypatch.setattr(auditor, name, forbid_receipt_lifecycle)

    execution = decoded = aligned_authority = eligibility_signatures = None
    sufficiency = feature_projection = manifest = columns = None
    partition_projection = partitions = adapter_pins = invariant_plan = None
    invariant_host_memory = plan = statistics_host_memory = None
    try:
        execution = qualification._build_fixture_execution_v2(mode)
        adapter_pins, _ = auditor._capture_adapter_execution_pins(
            execution.adapter,
            execution.recognizer,
        )
        decoded = _decode_batch_with_production_guards(
            execution.adapter,
            execution.adapter.envelopes,
        )
        aligned_authority = auditor._align_private_authority(
            execution.adapter.envelopes,
            execution.authority,
            execution.adapter.private_alignment,
        )
        eligibility_signatures = tuple(
            auditor._eligibility_structural_signature(item) for item in decoded
        )
        sufficiency = tuple(
            auditor._sufficiency(aligned_authority, target, eligibility_signatures)
            for target in auditor.TARGETS
        )
        assert all(item.sufficient for item in sufficiency)
        label_class_counts = tuple(item.label_class_count for item in sufficiency)
        envelope_and_decode_bytes_bound = (
            auditor._measured_envelope_decode_bytes_bound(
                len(execution.adapter.envelopes),
                execution.adapter.envelopes,
                decoded,
            )
        )
        feature_projection = auditor._feature_cardinality_projection(decoded)
        assert qualification._validate_fixture_feature_capacity_projection_v2(
            mode, feature_projection
        ) is feature_projection
        auditor._preflight_feature_extraction_memory(
            feature_projection,
            envelope_and_decode_bytes_bound,
        )
        invariant_plan = auditor._invariance_resource_plan(
            len(execution.adapter.envelopes),
            envelope_and_decode_bytes_bound=envelope_and_decode_bytes_bound,
            adapter_retained_graph_bytes_bound=(
                adapter_pins.retained_graph_bytes_bound
            ),
            variant_suite_resident_bytes_per_row_bound=(
                adapter_pins.variant_suite_resident_bytes_per_row_bound
            ),
            variant_suite_resident_fixed_bytes_bound=(
                adapter_pins.variant_suite_resident_fixed_bytes_bound
            ),
        )
        assert invariant_plan.admissible is True
        invariant_host_memory = auditor._preflight_invariance_host_memory(
            invariant_plan
        )
        auditor._preflight_feature_extraction_memory(
            feature_projection,
            envelope_and_decode_bytes_bound,
        )
        manifest, columns = auditor._extract_feature_columns(
            execution.adapter.envelopes,
            decoded,
            feature_projection,
        )
        assert manifest.deduplicated_feature_count == (
            feature_projection.projected_concrete_feature_count
        )
        assert manifest.raw_semantic_feature_count == (
            feature_projection.projected_raw_semantic_feature_count
        )
        feature_matrix_bytes_bound = auditor._measured_feature_matrix_bytes_bound(
            len(execution.adapter.envelopes),
            columns,
        )
        assert feature_matrix_bytes_bound <= (
            feature_projection.projected_feature_matrix_bytes_bound
        )
        partition_deadline_ns = (
            time.monotonic_ns()
            + auditor.PARTITION_PREPARATION_TIMEOUT_SECONDS * 1_000_000_000
        )
        partition_projection = auditor._partition_allocation_projection(
            len(execution.adapter.envelopes),
            columns,
            partition_deadline_ns,
        )
        auditor._preflight_partition_allocation_memory(
            feature_matrix_bytes_bound=feature_matrix_bytes_bound,
            envelope_and_decode_bytes_bound=envelope_and_decode_bytes_bound,
            partition_projection=partition_projection,
            concrete_features=manifest.deduplicated_feature_count,
        )
        partitions = auditor._materialize_partitions_bounded(
            columns,
            partition_projection,
            partition_deadline_ns,
        )
        del decoded, columns
        decoded = None
        columns = None
        plan = auditor._resource_plan(
            len(execution.adapter.envelopes),
            partitions,
            label_class_counts,
            preextraction_feature_matrix_bytes_bound=(
                feature_projection.projected_feature_matrix_bytes_bound
            ),
            feature_matrix_bytes_bound=feature_matrix_bytes_bound,
            envelope_and_decode_bytes_bound=envelope_and_decode_bytes_bound,
            adapter_retained_graph_bytes_bound=(
                adapter_pins.retained_graph_bytes_bound
            ),
            partition_projection=partition_projection,
            partition_preparation_deadline_ns=partition_deadline_ns,
            invariance_plan=invariant_plan,
        )
        assert plan.admissible is True
        auditor._validate_invariant_phase_preflight_against_resource_plan(
            plan,
            invariant_host_memory.resource_preflight,
        )
        statistics_host_memory = auditor._preflight_host_memory(plan)
        auditor._validate_statistics_phase_preflight_against_resource_plan(
            plan,
            statistics_host_memory.resource_preflight,
        )
        assert plan.concrete_features == (
            feature_projection.projected_concrete_feature_count
        )
        assert plan.preextraction_feature_matrix_bytes_bound == (
            feature_projection.projected_feature_matrix_bytes_bound
        )
        assert plan.estimated_peak_rss_bytes <= (
            auditor.MAXIMUM_STATIC_PHASE_PLAN_RSS_BYTES
        )
        assert plan.maximum_parallel_workers == 5
        assert (
            plan.estimated_invariance_peak_rss_bytes
            <= auditor.MAXIMUM_STATIC_PHASE_PLAN_RSS_BYTES
        )
        if mode == "KNOWN_BINDING_LEAK":
            assert plan.estimated_intersection_work_units == 12_205_620_440
            assert plan.unique_statistical_partitions == 17_562
        assert (
            plan.estimated_intersection_work_units
            <= auditor.MAXIMUM_STATISTICAL_INTERSECTION_WORK_UNITS
        )
        assert (
            plan.estimated_total_statistical_work_units
            <= auditor.MAXIMUM_TOTAL_STATISTICAL_WORK_UNITS
        )
        assert plan.partition_allocation_bytes_bound == (
            partition_projection.projected_partition_matrix_bytes_bound
        )
        assert plan.partition_matrix_bytes_bound <= (
            plan.partition_allocation_bytes_bound
        )
        assert plan.invariance_parent_current_rss_bytes_at_plan == (
            invariant_plan.parent_current_rss_bytes_at_plan
        )
        assert plan.invariance_resource_plan_root == invariant_plan.plan_root
        assert invariant_host_memory.resource_preflight.resource_plan_root == (
            plan.invariance_resource_plan_root
        )
        assert statistics_host_memory.resource_preflight.resource_plan_root == (
            plan.resource_plan_root
        )
        final_projection = (
            statistics_host_memory.resource_preflight
            .projected_aggregate_rss_bytes_at_preflight
        )
        assert final_projection == (
            max(
                statistics_host_memory.resource_preflight
                .parent_current_rss_bytes_at_preflight,
                statistics_host_memory.resource_preflight
                .static_parent_bytes_bound,
            )
            * statistics_host_memory.resource_preflight.fork_parent_rss_multiplier
            + statistics_host_memory.resource_preflight.additional_bytes_bound
        )
        assert final_projection <= auditor.MAXIMUM_STATIC_PHASE_PLAN_RSS_BYTES
        assert statistics_host_memory.effective_available_bytes >= (
            final_projection
            + auditor.HOST_AVAILABLE_MEMORY_SAFETY_MARGIN_BYTES
        )
    finally:
        del execution
        del decoded
        del aligned_authority
        del eligibility_signatures
        del sufficiency
        del feature_projection
        del manifest
        del columns
        del partition_projection
        del partitions
        del adapter_pins
        del invariant_plan
        del invariant_host_memory
        del plan
        del statistics_host_memory
        gc.collect()
        assert dict(auditor._IN_PROCESS_EXECUTED_RECEIPTS) == (
            executed_receipts_before
        )
        assert set(auditor._IN_PROCESS_RECEIPT_RESERVATIONS) == (
            receipt_reservations_before
        )
        assert receipt_lifecycle_calls == []
