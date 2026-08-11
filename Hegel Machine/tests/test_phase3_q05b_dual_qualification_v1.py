from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import errno
import fcntl
from hashlib import sha256
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

HOST = importlib.import_module("hegel_machine.phase3_q05b_host_replay_v1")
ADMISSION = importlib.import_module("hegel_machine.phase3_q05b_actual_admission_v1")
ARTIFACT = importlib.import_module("hegel_machine.phase3_q05b_actual_artifact_v1")
WIRE = importlib.import_module("hegel_machine.phase3_q1_qualification_wire_v1")
CAPACITY = importlib.import_module("hegel_machine.phase3_q1_capacity_preflight_v1")
SNAPSHOT = importlib.import_module("hegel_machine.phase3_q1_partition_snapshot_v1")
PROJECTION = importlib.import_module("hegel_machine.phase3_q1_archive_projection_v1")
COVERAGE = importlib.import_module("hegel_machine.phase3_q1_semantic_coverage_v1")
STRICT_CBOR = importlib.import_module("hegel_machine.strict_cbor_v1")

TOOL_PATH = ROOT / "tools/phase3_q05b_dual_qualification_v1.py"
SPEC = importlib.util.spec_from_file_location(
    "phase3_q05b_dual_qualification_v1",
    TOOL_PATH,
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def _docker_absence_sample(container_name: str) -> dict[str, object]:
    stdout = b""
    stderr = f"Error: No such object: {container_name}\n".encode("ascii")
    return {
        "schema_version": (
            "hegel-phase3a-q05b-docker-authoritative-absence/1"
        ),
        "container_identity": container_name,
        "inspect_exit_code": 1,
        "inspect_stdout_hex": stdout.hex(),
        "inspect_stdout_sha256": sha256(stdout).hexdigest(),
        "inspect_stderr_hex": stderr.hex(),
        "inspect_stderr_sha256": sha256(stderr).hexdigest(),
    }


def _docker_execution_authority(
    *,
    source_commit: str = "ab" * 20,
    nonce: bytes = b"N" * 32,
) -> dict[str, object]:
    slot_rows = ADMISSION.docker_execution_slot_rows_v1(source_commit, nonce)
    initial_rows = []
    for row in slot_rows:
        absence = _docker_absence_sample(row["container_name"])
        initial_rows.append(
            ADMISSION.build_docker_initial_name_absence_row_v1(
                source_commit,
                nonce,
                row["slot_id"],
                absence,
                deepcopy(absence),
            )
        )
    return ADMISSION.build_docker_execution_authority_v1(
        source_commit,
        nonce,
        initial_rows,
    )


def _docker_slot_row(
    authority: dict[str, object],
    slot: str,
) -> dict[str, object]:
    return next(
        deepcopy(row)
        for row in authority["ordered_slot_rows"]
        if row["slot"] == slot
    )


def _docker29_inline_seccomp_option(payload: bytes) -> str:
    """Render the inspect-only Docker 29 seccomp representation."""

    value = TOOL._strict_json_value_v1(payload, "test seccomp policy")
    assert type(value) is dict
    return "seccomp=" + json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _owned_inspect_payload(
    authority: dict[str, object],
    slot: str,
    command: list[str] | tuple[str, ...],
    container_id: str,
    *,
    running: bool = True,
) -> bytes:
    row = _docker_slot_row(authority, slot)
    image = (
        TOOL.RUST_IMAGE
        if slot in {"RUST_TEST", "RUST_RELEASE", "RUST_ENDPOINT"}
        else TOOL.PYTHON_IMAGE
    )
    image_index = list(command).index(image)
    policy_relative = (
        TOOL.BUILD_SECCOMP_RELATIVE_PATH
        if slot in {"RUST_TEST", "RUST_RELEASE"}
        else TOOL.RUNTIME_SECCOMP_RELATIVE_PATH
    )
    security_options = [
        "no-new-privileges",
        _docker29_inline_seccomp_option((ROOT / policy_relative).read_bytes()),
    ]
    value = [
        {
            "Id": container_id,
            "Name": f"/{row['container_name']}",
            "State": {
                "Running": running,
                "OOMKilled": False,
                "ExitCode": 0,
            },
            "Config": {
                "Image": image,
                "Cmd": list(command)[image_index + 1 :],
                "Labels": dict(row["expected_container_labels"]),
            },
            "HostConfig": {
                "AutoRemove": False,
                "SecurityOpt": security_options,
            },
        }
    ]
    return json.dumps(value, separators=(",", ":")).encode("ascii")


def _offline_rust_test_fixture_v1(
    tmp_path: Path,
    *,
    slot: str = "RUST_TEST",
) -> tuple[
    dict[str, object],
    dict[str, object],
    Path,
    list[str],
]:
    """Materialize one production-shaped offline-build command fixture."""

    authority = _docker_execution_authority()
    slot_row = _docker_slot_row(authority, slot)
    cid_parent = tmp_path / "cid"
    cid_parent.mkdir(mode=0o700)
    cidfile = cid_parent / "build.cid"
    seccomp = tmp_path / "sealed" / TOOL.BUILD_SECCOMP_RELATIVE_PATH
    seccomp.parent.mkdir(parents=True)
    seccomp.write_bytes((ROOT / TOOL.BUILD_SECCOMP_RELATIVE_PATH).read_bytes())
    seccomp.chmod(0o444)
    commands = TOOL.rust_build_commands_v1(
        tmp_path / "snapshot",
        tmp_path / "cargo-home",
        tmp_path / "target-output",
        "0" * 64,
        cidfile,
        build_seccomp=seccomp,
        docker_slot_row=slot_row,
    )
    command = commands[0 if slot == "RUST_TEST" else 1]
    return authority, slot_row, cidfile, command


def _held_python_start_fixture_v1(
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    Path,
    Path,
    list[str],
]:
    authority = _docker_execution_authority()
    slot_row = _docker_slot_row(authority, "PYTHON_ENDPOINT")
    cid_parent = tmp_path / "cid"
    cid_parent.mkdir(mode=0o700)
    cidfile = cid_parent / "python.cid"
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    seccomp = tmp_path / "sealed" / TOOL.RUNTIME_SECCOMP_RELATIVE_PATH
    seccomp.parent.mkdir(parents=True)
    seccomp.write_bytes((ROOT / TOOL.RUNTIME_SECCOMP_RELATIVE_PATH).read_bytes())
    seccomp.chmod(0o444)
    command = TOOL.python_endpoint_command_v1(
        tmp_path / "snapshot",
        tmp_path / "output",
        control,
        seccomp,
        docker_slot_row=slot_row,
        cidfile=cidfile,
    )
    return authority, slot_row, cidfile, control, command


def _bind_synthetic_actor_docker_ownership_v1(
    actor: object,
    authority: dict[str, object],
    slot: str,
    command: list[str] | tuple[str, ...],
) -> None:
    row = _docker_slot_row(authority, slot)
    principal = TOOL._docker_execution_principal_v1(command, authority, slot)
    actor.docker_execution_authority_manifest_sha256 = authority[
        "manifest_sha256"
    ]
    actor.docker_execution_slot_row = row
    actor.ownership_label_root = principal["ownership_label_root"]
    actor.precreate_absence_evidence = (
        ADMISSION.build_docker_precreate_absence_v1(
            authority,
            row["slot_id"],
            _docker_absence_sample(row["container_name"]),
            _docker_absence_sample(row["container_name"]),
        )
    )
    actor.docker_execution_principal = principal


def _strict_json(payload: bytes) -> dict[str, object]:
    def pairs(rows):
        result = {}
        for key, value in rows:
            assert key not in result
            result[key] = value
        return result

    value = json.loads(
        payload,
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(AssertionError(token)),
    )
    assert type(value) is dict
    return value


def _dummy_negative_corpus_binding():
    rows = tuple(
        (
            f"vector-{index:02d}".encode("ascii"),
            13 if index < 5 else 18,
            b"EXPECTED_FAILURE",
            b"EXPECTED_FAILURE",
            bytes([index + 1]) * 32,
        )
        for index in range(10)
    )
    category_roots = tuple(
        (
            category,
            STRICT_CBOR.content_hash(
                "HEGEL/Q05B/QUALIFICATION/NEGATIVE_VECTOR_CATEGORY/V1",
                (category, tuple(row for row in rows if row[1] == category)),
            ),
        )
        for category in (13, 18)
    )
    authority = (
        b"q1_state",
        b"NOT_RUN",
        b"q1_gate_count",
        0,
        b"q1_gate_mask",
        0,
        b"q1_output_slots",
        (None,) * 8,
        b"certificate_active",
        False,
    )
    value = (
        1,
        b"hegel-q05b-negative-vector-corpus/1",
        rows,
        category_roots,
        authority,
    )
    return (
        STRICT_CBOR.canonical_cbor_encode(value),
        STRICT_CBOR.content_hash(
            "HEGEL/Q05B/QUALIFICATION/NEGATIVE_VECTOR_CORPUS/V1",
            value,
        ),
        category_roots,
    )


def test_host_partition_replay_object_is_exact_machine_shape() -> None:
    replay = HOST.PartitionStrictReplayV1(
        1,
        b"a" * 32,
        b"b" * 32,
        (b"c" * 32,) * 4,
        (b"d" * 32,) * 4,
        (b"e" * 32,) * 4,
        (b"f" * 32,) * 4,
        (b"g" * 32,) * 4,
    )
    value = replay.canonical_object()
    assert len(value) == 10
    assert value[:3] == (
        1,
        HOST.PARTITION_STRICT_REPLAY_SCHEMA_ID,
        1,
    )
    assert not any(item == 0x3707 for item in value)


def test_host_semantic_witness_is_not_receipt_and_has_no_future_isolation() -> None:
    scratch_partitions = (
        SimpleNamespace(scratch_ledger_roots=(b"s" * 32,) * 4),
        SimpleNamespace(scratch_ledger_roots=(b"t" * 32,) * 4),
    )
    replay = HOST.DualHostReplayV1(
        SimpleNamespace(
            actor_id="PYTHON_ENDPOINT",
            payloads=(b"leaf", b"odd", b"sink", b"sidecar", b"golden"),
            partition_replays=scratch_partitions,
        ),
        SimpleNamespace(
            actor_id="RUST_ENDPOINT",
            partition_replays=scratch_partitions,
        ),
        b"neutral-cbor",
        b"a" * 32,
        b"b" * 32,
        b"c" * 32,
        b"d" * 32,
        b"e" * 32,
        tuple((predicate_id, bytes([predicate_id]) * 32) for predicate_id in (
            6, 7, 8, 12, 14, 15, 17
        )),
        b"k" * 32,
        (11, 13, 16, 18, 19),
        b"f" * 32,
    )
    negative_cbor, negative_root, category_roots = _dummy_negative_corpus_binding()
    payload = HOST.host_semantic_witness_bytes_v1(
        replay,
        negative_cbor,
        negative_root,
        category_roots,
    )
    decoded = HOST.decode_host_semantic_witness_v1(
        payload,
        replay,
        negative_cbor,
        negative_root,
        category_roots,
    )
    assert decoded["status"] == HOST.HOST_SEMANTIC_WITNESS_STATUS
    assert decoded["q1_authority"] == {
        "certificate_active": False,
        "formal_output_roots": [None] * 8,
        "gate_count": 0,
        "gate_mask": 0,
        "q2_state": "NOT_RUN",
        "state": "NOT_RUN",
    }
    assert decoded["pending_predicate_ids"] == [11, 19]
    assert [row[0] for row in decoded["predicate_evidence_rows"]] == [
        6, 7, 8, 12, 13, 14, 15, 16, 17, 18
    ]
    assert not any("isolation" in key or "resource" in key for key in decoded)
    assert "receipt" not in decoded
    tampered = _strict_json(payload)
    tampered["predicate_evidence_rows"][0][0] = True
    body = dict(tampered)
    body.pop("witness_root")
    tampered["witness_root"] = sha256(
        HOST.HOST_SEMANTIC_WITNESS_ROOT_DOMAIN
        + (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "ascii"
        )
    ).hexdigest()
    with pytest.raises(HOST.Q05BHostReplayError) as bool_alias:
        HOST.decode_host_semantic_witness_v1(
            (json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "ascii"
            )
        )
    assert bool_alias.value.code == HOST.FAIL_HOST_WIRE


def test_host_semantic_staging_and_control_stdout_are_separate(
    tmp_path: Path,
) -> None:
    scratch_partitions = (
        SimpleNamespace(scratch_ledger_roots=(b"s" * 32,) * 4),
        SimpleNamespace(scratch_ledger_roots=(b"t" * 32,) * 4),
    )
    replay = HOST.DualHostReplayV1(
        SimpleNamespace(
            actor_id="PYTHON_ENDPOINT",
            payloads=(b"leaf", b"odd", b"sink", b"sidecar", b"golden"),
            partition_replays=scratch_partitions,
        ),
        SimpleNamespace(
            actor_id="RUST_ENDPOINT",
            partition_replays=scratch_partitions,
        ),
        b"neutral-cbor",
        b"a" * 32,
        b"b" * 32,
        b"c" * 32,
        b"d" * 32,
        b"e" * 32,
        tuple((predicate_id, bytes([predicate_id]) * 32) for predicate_id in (
            6, 7, 8, 12, 14, 15, 17
        )),
        b"k" * 32,
        (11, 13, 16, 18, 19),
        b"f" * 32,
    )
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    negative_cbor, negative_root, category_roots = _dummy_negative_corpus_binding()
    witness, evidence = TOOL.write_host_semantic_staging_v1(
        replay,
        staging,
        HOST,
        negative_cbor,
        negative_root,
        category_roots,
    )
    assert evidence["file_count"] == 6
    assert (staging / TOOL.HOST_SEMANTIC_WITNESS_RELATIVE_PATH).read_bytes() == witness
    assert {
        path.relative_to(staging / TOOL.HOST_STAGED_SIDECAR_ROOT).as_posix()
        for path in (staging / TOOL.HOST_STAGED_SIDECAR_ROOT).rglob("*.cbor")
    } == set(TOOL.HOST_STAGED_SIDECAR_PATHS)
    control = TOOL.host_control_stdout_bytes_v1(
        replay,
        witness,
        b"c" * 32,
        b"d" * 32,
        HOST,
        (
            ("hegel_machine", None, None),
            (
                "hegel_machine.phase3_q05b_host_replay_v1",
                "src/hegel_machine/phase3_q05b_host_replay_v1.py",
                "11" * 32,
            ),
        ),
    )
    TOOL._validate_held_actor_stdout_v1(control, "TRUSTED_HOST_REPLAY")
    decoded = _strict_json(control)
    assert decoded["status"] == TOOL.HOST_CONTROL_STDOUT_STATUS
    assert decoded["qualification_receipt"] is None
    assert decoded["final_isolation_root"] is None
    assert "predicate_evidence_rows" not in decoded
    assert control != witness
    staging.chmod(0o755)


def test_internal_host_package_bootstrap_works_under_isolated_python() -> None:
    program = (
        "import pathlib,runpy;"
        f"m=runpy.run_path({str(TOOL_PATH)!r},run_name='q05b_isolated');"
        f"h=m['_load_host_replay_module_v1'](pathlib.Path({str(ROOT)!r}));"
        "s=__import__('sys');p=s.modules['hegel_machine'];"
        "a=s.modules['hegel_machine.phase3_q05b_actual_admission_v1'];"
        "bad=[n for n in s.modules if n.startswith('hegel_machine.') and "
        "any(x in n for x in ('__init__','phase3_dsl','target','truth','split','role'))];"
        "assert not bad;print(h.__name__,a.__name__,p.__file__,p.__spec__)"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", program],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.stdout == (
        "hegel_machine.phase3_q05b_host_replay_v1 "
        "hegel_machine.phase3_q05b_actual_admission_v1 None None\n"
    )


def _write_exact_sidecar_tree(root: Path, *, payload: bytes = b"x") -> None:
    for relative in WIRE.ORDERED_OUTPUT_RELATIVE_PATHS:
        path = root / relative.decode("ascii")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(WIRE.OUTPUT_FILE_MODE)


def test_sidecar_reader_rejects_symlink_mode_and_special_file(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    _write_exact_sidecar_tree(tree)
    first = tree / WIRE.ORDERED_OUTPUT_RELATIVE_PATHS[0].decode("ascii")
    assert HOST.read_frozen_file_v1(tree, WIRE.ORDERED_OUTPUT_RELATIVE_PATHS[0]) == b"x"

    first.chmod(0o644)
    with pytest.raises(HOST.Q05BHostReplayError) as mode:
        HOST.read_frozen_file_v1(tree, WIRE.ORDERED_OUTPUT_RELATIVE_PATHS[0])
    assert mode.value.code == HOST.FAIL_HOST_TREE
    first.chmod(WIRE.OUTPUT_FILE_MODE)

    first.unlink()
    first.symlink_to(tree / WIRE.ORDERED_OUTPUT_RELATIVE_PATHS[1].decode("ascii"))
    with pytest.raises(HOST.Q05BHostReplayError):
        HOST.read_exact_sidecar_tree_v1(tree)
    first.unlink()
    first.write_bytes(b"x")
    first.chmod(WIRE.OUTPUT_FILE_MODE)

    fifo = tree / "extra-fifo"
    os.mkfifo(fifo)
    with pytest.raises(HOST.Q05BHostReplayError) as special:
        HOST.read_exact_sidecar_tree_v1(tree)
    assert special.value.code == HOST.FAIL_HOST_TREE

    fifo.unlink()
    short_alias = Path("/tmp") / f"q05b-{os.getpid()}-{id(tree)}"
    short_alias.symlink_to(tree, target_is_directory=True)
    unix_socket = socket.socket(socket.AF_UNIX)
    try:
        unix_socket.bind(str(short_alias / "extra-socket"))
        with pytest.raises(HOST.Q05BHostReplayError):
            HOST.read_exact_sidecar_tree_v1(tree)
    finally:
        unix_socket.close()
        short_alias.unlink(missing_ok=True)


def test_neutral_agreement_is_five_file_byte_exact() -> None:
    payloads = (b"a", b"b", b"c", b"d", b"e")
    assert HOST.require_neutral_byte_agreement_v1(payloads, payloads) == b"e"
    changed = (*payloads[:2], b"x", *payloads[3:])
    with pytest.raises(HOST.Q05BHostReplayError) as error:
        HOST.require_neutral_byte_agreement_v1(payloads, changed)
    assert error.value.code == HOST.FAIL_HOST_DISAGREEMENT


def _valid_actor_stdout(actor_id: str, implementation_id: str) -> bytes:
    value = {
        "action_id": "bounded-node3-golden-v1",
        "actor_id": actor_id,
        "file_count": 5,
        "implementation_id": implementation_id,
        "neutral_manifest_length": 4134,
        "neutral_manifest_raw_sha256": HOST.EXPECTED_RAW_SHA256[3].hex(),
        "neutral_manifest_relative_path": WIRE.NODE3_GOLDEN_MANIFEST_RELATIVE_PATH.decode(),
        "neutral_manifest_root": HOST.EXPECTED_ROOTS["golden"].hex(),
        "q1_formal_roots": None,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "q1_state": "NOT_RUN",
        "runtime_identity_sha256": "22" * 32,
        "schema_version": "hegel-q05b-actor-envelope/1",
        "sidecar_manifest_length": 552,
        "sidecar_manifest_raw_sha256": HOST.EXPECTED_RAW_SHA256[2].hex(),
        "sidecar_manifest_relative_path": WIRE.SIDECAR_MANIFEST_RELATIVE_PATH.decode(),
        "sidecar_manifest_root": HOST.EXPECTED_ROOTS["sidecar"].hex(),
        "source_identity_sha256": "11" * 32,
        "status": "BOUNDED_NODE3_CANDIDATE_EMITTED_NOT_QUALIFIED",
    }
    return TOOL._canonical_json_bytes(value)


def _mock_held_python_actor(
    tmp_path: Path,
    case: str,
):
    """Create a real held child process around fake proc/cgroup observations."""

    root = tmp_path / case
    root.mkdir(mode=0o700)
    control = root / "control"
    control.mkdir(mode=0o700)
    cid_parent = root / "cid"
    cid_parent.mkdir(mode=0o700)
    cidfile = cid_parent / "python.cid"
    container_id = "ab" * 32
    cidfile.write_bytes(container_id.encode("ascii"))
    cidfile.chmod(0o600)
    cid_status = cid_parent.lstat()
    cid_parent_identity = (
        cid_status.st_dev,
        cid_status.st_ino,
        stat.S_IMODE(cid_status.st_mode),
        cid_status.st_nlink,
    )
    observed_id, cid_evidence = TOOL._seal_cidfile_v1(
        cidfile,
        cid_parent_identity,
    )
    assert observed_id == container_id
    actor_stdout = _valid_actor_stdout(
        "PYTHON_ENDPOINT",
        "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_PYTHON_V1",
    )
    for name, payload in (
        ("actor.stdout", actor_stdout),
        ("exit-code", TOOL.HELD_SUCCESS_EXIT_BYTES),
        ("done", TOOL.HELD_DONE_BYTES),
    ):
        path = control / name
        path.write_bytes(payload)
        path.chmod(0o600)

    snapshot = root / "python-source"
    output = root / "python-output"
    host_source = root / "host-source"
    for directory in (snapshot, output, host_source):
        directory.mkdir(mode=0o700)
    seccomp_path = host_source / TOOL.RUNTIME_SECCOMP_RELATIVE_PATH
    seccomp_path.parent.mkdir(parents=True)
    seccomp_payload = b'{"defaultAction":"SCMP_ACT_ERRNO"}\n'
    seccomp_path.write_bytes(seccomp_payload)
    seccomp_path.chmod(0o444)
    seccomp_evidence = TOOL.sealed_policy_file_evidence_v1(
        seccomp_path,
        TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
    )
    docker_authority = _docker_execution_authority()
    docker_slot_row = _docker_slot_row(
        docker_authority,
        "PYTHON_ENDPOINT",
    )
    command = TOOL.python_endpoint_command_v1(
        snapshot,
        output,
        control,
        seccomp_path,
        docker_slot_row=docker_slot_row,
        cidfile=cidfile,
    )
    mount_registry = TOOL.sealed_actor_mount_registry_v1(1, command)

    pid = 1234
    proc_root = root / "proc"
    pid_root = proc_root / str(pid)
    pid_root.mkdir(parents=True)
    cgroup_root = root / "cgroup"
    cgroup_path = f"/docker/{container_id}"
    cgroup_directory = cgroup_root / "docker" / container_id
    cgroup_directory.mkdir(parents=True)
    proc_cgroup = f"0::{cgroup_path}\n".encode("ascii")
    proc_limits = (
        b"Limit                     Soft Limit           Hard Limit           Units     \n"
        b"Max open files            256                  256                  files     \n"
    )
    (pid_root / "cgroup").write_bytes(proc_cgroup)
    (pid_root / "limits").write_bytes(proc_limits)
    cgroup_payloads = {
        "memory.current": b"100\n",
        "memory.peak": b"999\n",
        "memory.events": (
            b"low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n"
        ),
        "pids.current": b"2\n",
        "pids.peak": b"4\n",
    }
    for name, payload in cgroup_payloads.items():
        (cgroup_directory / name).write_bytes(payload)

    live_document = [
        {
            "Id": container_id,
            "Name": f"/{docker_slot_row['container_name']}",
            "State": {"Running": True, "OOMKilled": False, "Pid": pid},
            "Config": {
                "Image": TOOL.PYTHON_IMAGE,
                "User": f"{os.getuid()}:{os.getgid()}",
                "Entrypoint": None,
                "Cmd": list(mount_registry.container_argv),
                "WorkingDir": mount_registry.working_directory,
                "Labels": dict(docker_slot_row["expected_container_labels"]),
                "Env": [
                    f"{key}={value}"
                    for key, value in mount_registry.environment_rows
                ],
            },
            "HostConfig": {
                "AutoRemove": False,
                "NetworkMode": "none",
                "ReadonlyRootfs": True,
                "CapDrop": ["ALL"],
                "SecurityOpt": [
                    "no-new-privileges",
                    _docker29_inline_seccomp_option(seccomp_payload),
                ],
                "IpcMode": "none",
                "PidMode": "",
                "CgroupnsMode": "private",
                "UsernsMode": "",
                "Privileged": False,
                "Devices": [],
                "DeviceRequests": None,
                "CpusetCpus": "0-11",
                "Memory": 14 * 1024 * 1024 * 1024,
                "MemorySwap": 14 * 1024 * 1024 * 1024,
                "PidsLimit": 128,
                "Ulimits": [{"Name": "nofile", "Hard": 256, "Soft": 256}],
                "Tmpfs": {"/tmp": "rw,noexec,nosuid,nodev,size=2g,mode=1777"},
            },
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": source,
                    "Destination": destination,
                    "RW": writable,
                }
                for destination, source, writable in mount_registry.mount_rows
            ],
        }
    ]
    post_document = [
        {
            "Id": container_id,
            "Name": f"/{docker_slot_row['container_name']}",
            "State": {"Running": False, "OOMKilled": False, "ExitCode": 0},
            "Config": {
                "Image": TOOL.PYTHON_IMAGE,
                "Cmd": list(mount_registry.container_argv),
                "Labels": dict(docker_slot_row["expected_container_labels"]),
            },
            "HostConfig": {
                "AutoRemove": False,
                "SecurityOpt": [
                    "no-new-privileges",
                    _docker29_inline_seccomp_option(seccomp_payload),
                ],
            },
        }
    ]
    live_payload = json.dumps(live_document, separators=(",", ":")).encode("ascii")
    post_payload = json.dumps(post_document, separators=(",", ":")).encode("ascii")

    child_script = (
        "import os,sys,time;"
        "os.write(1,bytes.fromhex(sys.argv[1]));"
        "p=sys.argv[2];"
        "exec('while not os.path.exists(p):\\n time.sleep(0.01)')"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child_script, actor_stdout.hex(), str(control / "release")],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None and process.stderr is not None
    stdout_drain = TOOL.BoundedPipeDrainV1(1024 * 1024, bytearray(), 0, False, sha256(), [])
    stderr_drain = TOOL.BoundedPipeDrainV1(16 * 1024 * 1024, bytearray(), 0, False, sha256(), [])
    stdout_thread = threading.Thread(
        target=TOOL._drain_pipe_v1,
        args=(process.stdout, stdout_drain),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=TOOL._drain_pipe_v1,
        args=(process.stderr, stderr_drain),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    start_ns = time.monotonic_ns()
    initial_sample = TOOL.collect_bound_live_resource_transcript_v1(
        1,
        container_id,
        docker_slot_row["container_name"],
        mount_registry,
        live_payload,
        lambda: live_payload,
        seccomp_evidence=seccomp_evidence,
        proc_root=proc_root,
        cgroup_root=cgroup_root,
    )
    finish_ns = time.monotonic_ns()
    initial_sample["sample_ordinal"] = 1
    initial_sample["sample_monotonic_ns"] = start_ns
    initial_sample["sample_duration_ns"] = finish_ns - start_ns
    actor = TOOL.HeldActorProcessV1(
        role_id=1,
        actor_id="PYTHON_ENDPOINT",
        container_name=docker_slot_row["container_name"],
        command=tuple(command),
        cidfile=cidfile,
        control_root=control,
        mount_registry=mount_registry,
        process=process,
        stdout_drain=stdout_drain,
        stderr_drain=stderr_drain,
        stdout_thread=stdout_thread,
        stderr_thread=stderr_thread,
        sample_thread=None,
        sample_rows=[initial_sample],
        sample_errors=[],
        sample_complete=threading.Event(),
        child_done_observed=threading.Event(),
        sample_stop=threading.Event(),
        sample_lock=threading.Lock(),
        container_id=container_id,
        cid_parent_identity=cid_parent_identity,
        cidfile_evidence=cid_evidence,
        cleanup_errors=[],
        seccomp_evidence=seccomp_evidence,
        docker_execution_authority_manifest_sha256=docker_authority[
            "manifest_sha256"
        ],
        docker_execution_slot_row=docker_slot_row,
        ownership_label_root=TOOL._docker_execution_principal_v1(
            command,
            docker_authority,
            "PYTHON_ENDPOINT",
        )["ownership_label_root"],
        precreate_absence_evidence=ADMISSION.build_docker_precreate_absence_v1(
            docker_authority,
            docker_slot_row["slot_id"],
            _docker_absence_sample(docker_slot_row["container_name"]),
            _docker_absence_sample(docker_slot_row["container_name"]),
        ),
        docker_execution_principal=TOOL._docker_execution_principal_v1(
            command,
            docker_authority,
            "PYTHON_ENDPOINT",
        ),
    )

    def mock_sampler() -> None:
        actor.sample_stop.wait()
        actor.sample_complete.set()

    sample_thread = threading.Thread(target=mock_sampler, daemon=True)
    actor.sample_thread = sample_thread
    sample_thread.start()
    actor.child_done_observed.set()

    def inspect_reader(requested_id: str) -> bytes:
        assert requested_id == container_id
        return live_payload if process.poll() is None else post_payload

    docker_calls: list[list[str]] = []

    def command_runner(command, **_kwargs):
        docker_calls.append(list(command))
        if list(command)[2] == "inspect":
            target = list(command)[-1]
            return SimpleNamespace(
                returncode=1,
                stdout=b"",
                stderr=f"Error: No such object: {target}\n".encode("ascii"),
            )
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    return (
        actor,
        inspect_reader,
        proc_root,
        cgroup_root,
        command_runner,
        docker_calls,
        control,
        actor_stdout,
    )


def test_sealed_stdout_manifest_is_separate_exact_and_actor_bound() -> None:
    python_stdout = _valid_actor_stdout(
        "PYTHON_ENDPOINT",
        "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_PYTHON_V1",
    )
    rust_stdout = _valid_actor_stdout(
        "RUST_ENDPOINT",
        "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_RUST_V1",
    )
    manifest = HOST.sealed_actor_stdout_manifest_bytes_v1(
        python_stdout,
        rust_stdout,
    )
    python, rust = HOST.validate_sealed_actor_stdout_set_v1(
        python_stdout,
        rust_stdout,
        manifest,
    )
    assert (python["actor_id"], rust["actor_id"]) == (
        "PYTHON_ENDPOINT",
        "RUST_ENDPOINT",
    )
    assert b'"rows":[[1,"PYTHON_ENDPOINT"' in manifest
    with pytest.raises(HOST.Q05BHostReplayError) as changed:
        HOST.validate_sealed_actor_stdout_set_v1(
            python_stdout,
            rust_stdout,
            manifest.replace(b"PYTHON_ENDPOINT", b"PYTHON_ENDPOINX"),
        )
    assert changed.value.code == HOST.FAIL_HOST_STDOUT


def test_isolation_config_is_exact_zero_null_and_actual_not_executed() -> None:
    config = TOOL.load_isolation_config_v1(ROOT)
    assert config["resource_roles"] == [list(row) for row in TOOL.ROLE_ROWS]
    assert config["docker"]["pids_limit"] == 128
    assert config["docker"]["nofile_ulimit"] == "256:256"
    assert config["docker"]["memory"] == config["docker"]["memory_swap"] == "14g"
    assert [row[:3] for row in config["source_allowlist_policy"]["actor_rows"]] == [
        [1, "PYTHON_ENDPOINT", 40],
        [2, "RUST_ENDPOINT", 27],
        [3, "TRUSTED_HOST_REPLAY", 48],
    ]
    inspect_policy = config["runtime_command_inspect_policy"]
    assert inspect_policy["environment_rows"] == [
        [
            1,
            "PYTHON_ENDPOINT",
            [list(row) for row in sorted(TOOL.PYTHON_RUNTIME_ENVIRONMENT.items())],
        ],
        [
            2,
            "RUST_ENDPOINT",
            [list(row) for row in sorted(TOOL.RUST_RUNTIME_ENVIRONMENT.items())],
        ],
        [
            3,
            "TRUSTED_HOST_REPLAY",
            [list(row) for row in sorted(TOOL.PYTHON_RUNTIME_ENVIRONMENT.items())],
        ],
    ]
    assert config["held_actor_protocol"]["wrapper_script_exact"] == (
        TOOL.HELD_ACTOR_WRAPPER_SCRIPT
    )
    assert config["engineering_status"] == TOOL.COMMIT_A_ACTUAL_ENGINEERING_STATUS
    assert config["actual_preconditions"] == TOOL.COMMIT_A_ACTUAL_PRECONDITIONS_V1
    assert config["actual_preconditions"]["implementation_blocked_predicate_ids"] == []
    assert config["actual_preconditions"]["pending_actual_evidence_predicate_ids"] == list(
        range(1, 21)
    )
    assert "current_actual_admitted" not in config["actual_preconditions"]
    authority = config["dry_run_authority"]
    assert authority["qualification_predicate_count"] == 0
    assert authority["qualification_predicate_mask"] == 0
    assert authority["q1_state"] == "NOT_RUN"
    assert authority["q1_gate_count"] == authority["q1_gate_mask"] == 0
    assert authority["q1_formal_output_roots"] == [None] * 8
    assert authority["q1_receipt"] is None
    assert authority["m3_formal_roots"] is None
    assert authority["artifact_written"] is False


@pytest.mark.parametrize("replacement", [True, 128.0])
def test_config_numeric_bool_and_float_aliases_fail_closed(
    tmp_path: Path,
    replacement: object,
) -> None:
    (tmp_path / "config").mkdir()
    for relative in (
        TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
        TOOL.BUILD_SECCOMP_RELATIVE_PATH,
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    value = json.loads((ROOT / TOOL.CONFIG_RELATIVE_PATH).read_bytes())
    value["docker"]["pids_limit"] = replacement
    (tmp_path / TOOL.CONFIG_RELATIVE_PATH).write_text(
        json.dumps(value),
        encoding="ascii",
    )
    with pytest.raises(TOOL.Q05BDualSupervisorError) as error:
        TOOL.load_isolation_config_v1(tmp_path)
    assert error.value.code == TOOL.FAIL_CONFIG


def _assert_hardened(command: list[str], cpuset: str) -> None:
    assert command[:3] == [
        TOOL.DOCKER_EXECUTABLE,
        f"--host={TOOL.DOCKER_HOST}",
        "run",
    ]
    required = {
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--ipc=none",
        "--pids-limit=128",
        "--ulimit=nofile=256:256",
        "--memory=14g",
        "--memory-swap=14g",
        f"--cpuset-cpus={cpuset}",
    }
    assert required <= set(command)
    assert "--rm" not in command
    seccomp = [item for item in command if item.startswith("--security-opt=seccomp=")]
    assert len(seccomp) == 1
    assert seccomp[0].split("=", 2)[2].startswith("/sealed/")
    assert not any("docker.sock" in item and "type=bind" in item for item in command)


def test_dry_run_commands_freeze_parallel_cpu_and_no_source_rust_runtime() -> None:
    sealed = Path("/sealed")
    authority = _docker_execution_authority()
    python = TOOL.python_endpoint_command_v1(
        sealed / "python-source",
        sealed / "python-output",
        sealed / "python-control",
        sealed / "host-source" / TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
        docker_slot_row=_docker_slot_row(authority, "PYTHON_ENDPOINT"),
    )
    rust = TOOL.rust_runtime_command_v1(
        sealed / "prebuilt-runtime",
        sealed / "rust-output",
        sealed / "rust-control",
        sealed / "host-source" / TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
        docker_slot_row=_docker_slot_row(authority, "RUST_ENDPOINT"),
    )
    host = TOOL.trusted_host_command_v1(
        sealed / "host-source",
        sealed / "python-output",
        sealed / "rust-output",
        sealed / "stdout" / "python.stdout",
        sealed / "stdout" / "rust.stdout",
        sealed / "stdout" / "manifest.json",
        sealed / "host-control",
        sealed / "host-staging",
        sealed / "host-source" / TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
        docker_slot_row=_docker_slot_row(authority, "TRUSTED_HOST_REPLAY"),
    )
    _assert_hardened(python, "0-11")
    _assert_hardened(rust, "12-23")
    _assert_hardened(host, "0-11")
    assert not any("rust-source" in item and "type=bind" in item for item in rust)
    assert "cargo" not in rust[rust.index(TOOL.RUST_IMAGE) + 1 :]
    assert any("python-output" in item and item.endswith(",readonly") for item in host)
    assert any("rust-output" in item and item.endswith(",readonly") for item in host)
    assert sum("/inputs/stdout/" in item and item.endswith(",readonly") for item in host) == 3
    assert any(item.startswith("--cidfile=/sealed/") for item in python)
    assert any(item.startswith("--cidfile=/sealed/") for item in rust)
    assert any(item.startswith("--cidfile=/sealed/") for item in host)
    for command in (python, rust, host):
        assert TOOL.HELD_ACTOR_WRAPPER_SCRIPT in command
        assert "/bin/sh" in command
        assert any("dst=/control" in item and not item.endswith(",readonly") for item in command)
        assert any(
            item
            == "--security-opt=seccomp=/sealed/host-source/"
            + TOOL.RUNTIME_SECCOMP_RELATIVE_PATH
            for item in command
        )
    wrapper = TOOL.HELD_ACTOR_WRAPPER_SCRIPT
    assert wrapper.index("cat /control/actor.stdout") < wrapper.index(
        "ACTOR_COMPLETE_HELD"
    )
    assert wrapper.index("ACTOR_COMPLETE_HELD") < wrapper.index(
        "while test ! -f /control/release"
    )
    assert "cat /control/actor.stdout" not in wrapper[
        wrapper.index("while test ! -f /control/release") :
    ]
    build_test = TOOL.rust_build_commands_v1(
        sealed / "rust-source",
        sealed / "cargo-home",
        sealed / "target-output",
        "ab" * 32,
        build_seccomp=(
            sealed / "host-source" / TOOL.BUILD_SECCOMP_RELATIVE_PATH
        ),
        docker_slot_row=_docker_slot_row(authority, "RUST_TEST"),
    )[0]
    build_release = TOOL.rust_build_commands_v1(
        sealed / "rust-source",
        sealed / "cargo-home",
        sealed / "target-output",
        "ab" * 32,
        build_seccomp=(
            sealed / "host-source" / TOOL.BUILD_SECCOMP_RELATIVE_PATH
        ),
        docker_slot_row=_docker_slot_row(authority, "RUST_RELEASE"),
    )[1]
    for command in (build_test, build_release):
        _assert_hardened(command, "12-23")
        assert "--network=none" in command
        assert "CARGO_NET_OFFLINE=true" in command
        assert "HEGEL_Q05B_RUST_SOURCE_IDENTITY_SHA256=" + "ab" * 32 in command
        assert any("cargo-home" in item and item.endswith(",readonly") for item in command)
        assert (
            "--security-opt=seccomp=/sealed/host-source/"
            + TOOL.BUILD_SECCOMP_RELATIVE_PATH
        ) in command


@pytest.mark.parametrize(
    "relative_path",
    (
        TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
        TOOL.BUILD_SECCOMP_RELATIVE_PATH,
    ),
)
def test_docker29_inspect_seccomp_semantics_are_inline_strict_and_type_exact(
    tmp_path: Path,
    relative_path: str,
) -> None:
    policy_path = tmp_path / relative_path
    policy_path.parent.mkdir(parents=True)
    payload = (ROOT / relative_path).read_bytes()
    policy_path.write_bytes(payload)
    policy_path.chmod(0o444)
    evidence = TOOL.sealed_policy_file_evidence_v1(
        policy_path,
        relative_path,
    )
    command_options = (
        "no-new-privileges",
        f"seccomp={policy_path.as_posix()}",
    )
    inline = _docker29_inline_seccomp_option(payload)
    assert TOOL.validate_docker_inspect_seccomp_semantics_v1(
        ["no-new-privileges", inline],
        command_options,
        evidence,
        relative_path,
    ) == evidence

    value = TOOL._strict_json_value_v1(payload, "test policy")
    assert type(value) is dict
    semantic_drift = deepcopy(value)
    semantic_drift["defaultAction"] = "SCMP_ACT_ERRNO"
    type_alias = deepcopy(value)
    assert type(type_alias["defaultErrnoRet"]) is int
    type_alias["defaultErrnoRet"] = float(type_alias["defaultErrnoRet"])
    rejected_observations = (
        ["no-new-privileges", f"seccomp={policy_path.as_posix()}"],
        ["no-new-privileges", inline, "apparmor=unconfined"],
        ["no-new-privileges:true", inline],
        [inline, "no-new-privileges"],
        [
            "no-new-privileges",
            "seccomp="
            + json.dumps(semantic_drift, separators=(",", ":")),
        ],
        [
            "no-new-privileges",
            'seccomp={"defaultAction":"x","defaultAction":"x"}',
        ],
        ["no-new-privileges", 'seccomp={"value":NaN}'],
        [
            "no-new-privileges",
            "seccomp="
            + json.dumps(type_alias, separators=(",", ":")),
        ],
    )
    for observed in rejected_observations:
        with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
            TOOL.validate_docker_inspect_seccomp_semantics_v1(
                observed,
                command_options,
                evidence,
                relative_path,
            )
        assert failure.value.code == TOOL.FAIL_POLICY

    with pytest.raises(TOOL.Q05BDualSupervisorError) as inline_command:
        TOOL.validate_docker_inspect_seccomp_semantics_v1(
            ["no-new-privileges", inline],
            ("no-new-privileges", inline),
            evidence,
            relative_path,
        )
    assert inline_command.value.code == TOOL.FAIL_POLICY


def test_docker29_inspect_seccomp_rejects_coordinated_finite_float(
    tmp_path: Path,
) -> None:
    relative_path = TOOL.RUNTIME_SECCOMP_RELATIVE_PATH
    value = json.loads((ROOT / relative_path).read_bytes())
    assert type(value) is dict
    assert type(value["defaultErrnoRet"]) is int
    value["defaultErrnoRet"] = float(value["defaultErrnoRet"])
    payload = json.dumps(value, separators=(",", ":")).encode("ascii")
    assert b'"defaultErrnoRet":1.0' in payload

    policy_path = tmp_path / relative_path
    policy_path.parent.mkdir(parents=True)
    policy_path.write_bytes(payload)
    policy_path.chmod(0o444)
    evidence = TOOL.sealed_policy_file_evidence_v1(
        policy_path,
        relative_path,
    )
    command_options = (
        "no-new-privileges",
        f"seccomp={policy_path.as_posix()}",
    )
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.validate_docker_inspect_seccomp_semantics_v1(
            ["no-new-privileges", "seccomp=" + payload.decode("ascii")],
            command_options,
            evidence,
            relative_path,
        )
    assert failure.value.code == TOOL.FAIL_POLICY
    assert "finite JSON float" in failure.value.detail


def test_live_resource_transcript_replays_inspect_cgroup_and_proc_limits(
    tmp_path: Path,
) -> None:
    container_id = "ab" * 32
    seccomp_path = tmp_path / "sealed" / TOOL.RUNTIME_SECCOMP_RELATIVE_PATH
    seccomp_path.parent.mkdir(parents=True)
    seccomp_payload = (ROOT / TOOL.RUNTIME_SECCOMP_RELATIVE_PATH).read_bytes()
    seccomp_path.write_bytes(seccomp_payload)
    seccomp_path.chmod(0o444)
    seccomp_evidence = TOOL.sealed_policy_file_evidence_v1(
        seccomp_path,
        TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
    )
    docker_authority = _docker_execution_authority()
    docker_slot_row = _docker_slot_row(
        docker_authority,
        "PYTHON_ENDPOINT",
    )
    mount_registry = TOOL.sealed_actor_mount_registry_v1(
        1,
        TOOL.python_endpoint_command_v1(
            Path("/sealed/python-source"),
            Path("/sealed/python-output"),
            Path("/sealed/python-control"),
            seccomp_path,
            docker_slot_row=docker_slot_row,
        ),
    )
    inspect_payload = json.dumps(
        [
            {
                "Id": container_id,
                "Name": f"/{docker_slot_row['container_name']}",
                "State": {"Running": True, "OOMKilled": False, "Pid": 1234},
                "Config": {
                    "Image": TOOL.PYTHON_IMAGE,
                    "User": f"{os.getuid()}:{os.getgid()}",
                    "Entrypoint": None,
                    "Cmd": list(mount_registry.container_argv),
                    "WorkingDir": mount_registry.working_directory,
                    "Labels": dict(docker_slot_row["expected_container_labels"]),
                    "Env": [
                        f"{key}={value}"
                        for key, value in mount_registry.environment_rows
                    ],
                },
                "HostConfig": {
                    "AutoRemove": False,
                    "NetworkMode": "none",
                    "ReadonlyRootfs": True,
                    "CapDrop": ["ALL"],
                    "SecurityOpt": [
                        "no-new-privileges",
                        _docker29_inline_seccomp_option(seccomp_payload),
                    ],
                    "IpcMode": "none",
                    "PidMode": "",
                    "CgroupnsMode": "private",
                    "UsernsMode": "",
                    "Privileged": False,
                    "Devices": [],
                    "DeviceRequests": None,
                    "CpusetCpus": "0-11",
                    "Memory": 14 * 1024 * 1024 * 1024,
                    "MemorySwap": 14 * 1024 * 1024 * 1024,
                    "PidsLimit": 128,
                    "Ulimits": [{"Name": "nofile", "Hard": 256, "Soft": 256}],
                    "Tmpfs": {
                        "/tmp": "rw,noexec,nosuid,nodev,size=2g,mode=1777"
                    },
                },
                "Mounts": [
                    {
                        "Type": "bind",
                        "Source": "/sealed/python-source",
                        "Destination": "/snapshot",
                        "RW": False,
                    },
                    {
                        "Type": "bind",
                        "Source": "/sealed/python-output",
                        "Destination": "/output",
                        "RW": True,
                    },
                    {
                        "Type": "bind",
                        "Source": "/sealed/python-control",
                        "Destination": "/control",
                        "RW": True,
                    },
                ],
            }
        ],
        separators=(",", ":"),
    ).encode("ascii")
    cgroup = {
        "memory.current": b"100\n",
        "memory.peak": b"200\n",
        "memory.events": (
            b"low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n"
        ),
        "pids.current": b"2\n",
        "pids.peak": b"4\n",
    }
    proc_limits = (
        b"Limit                     Soft Limit           Hard Limit           Units     \n"
        b"Max open files            256                  256                  files     \n"
    )
    cgroup_path = f"/docker/{container_id}"
    proc_cgroup = f"0::{cgroup_path}\n".encode("ascii")
    mount_sources = {
        "/snapshot": "/sealed/python-source",
        "/output": "/sealed/python-output",
        "/control": "/sealed/python-control",
    }
    assert mount_registry.expected_sources == mount_sources
    with pytest.raises(TOOL.Q05BDualSupervisorError) as self_chosen_sources:
        TOOL.live_resource_transcript_v1(
            1,
            container_id,
            docker_slot_row["container_name"],
            mount_sources,  # type: ignore[arg-type]
            inspect_payload,
            cgroup,
            proc_cgroup,
            cgroup_path,
            (31, 41),
            proc_limits,
            seccomp_evidence=seccomp_evidence,
        )
    assert self_chosen_sources.value.code == TOOL.FAIL_POLICY
    transcript = TOOL.live_resource_transcript_v1(
        1,
        container_id,
        docker_slot_row["container_name"],
        mount_registry,
        inspect_payload,
        cgroup,
        proc_cgroup,
        cgroup_path,
        (31, 41),
        proc_limits,
        seccomp_evidence=seccomp_evidence,
    )
    assert transcript["captured_while_running"] is True
    assert transcript["memory_peak_bytes"] == 200
    assert transcript["pids_peak"] == 4
    no_unit_padding = TOOL.live_resource_transcript_v1(
        1,
        container_id,
        docker_slot_row["container_name"],
        mount_registry,
        inspect_payload,
        cgroup,
        proc_cgroup,
        cgroup_path,
        (31, 41),
        b"Max open files 256 256 files\n",
        seccomp_evidence=seccomp_evidence,
    )
    assert no_unit_padding["nofile_soft"] == 256
    assert no_unit_padding["nofile_hard"] == 256
    malformed_nofile_rows = (
        (b"Max open files\t256 256 files     \n", "format differs"),
        (b"Max open files 256 256 files extra\n", "format differs"),
        (b"Max open files 257 256 files     \n", "values differ"),
    )
    for malformed_payload, expected_detail in malformed_nofile_rows:
        with pytest.raises(TOOL.Q05BDualSupervisorError) as malformed_nofile:
            TOOL.live_resource_transcript_v1(
                1,
                container_id,
                docker_slot_row["container_name"],
                mount_registry,
                inspect_payload,
                cgroup,
                proc_cgroup,
                cgroup_path,
                (31, 41),
                malformed_payload,
                seccomp_evidence=seccomp_evidence,
            )
        assert malformed_nofile.value.code == TOOL.FAIL_POLICY
        assert expected_detail in str(malformed_nofile.value)
    wrong_mount_document = json.loads(inspect_payload)
    wrong_mount_document[0]["Mounts"][0]["Source"] = "/sealed/wrong-source"
    wrong_mount_payload = json.dumps(
        wrong_mount_document,
        separators=(",", ":"),
    ).encode("ascii")
    with pytest.raises(TOOL.Q05BDualSupervisorError) as wrong_mount:
        TOOL.live_resource_transcript_v1(
            1,
            container_id,
            docker_slot_row["container_name"],
            mount_registry,
            wrong_mount_payload,
            cgroup,
            proc_cgroup,
            cgroup_path,
            (31, 41),
            proc_limits,
            seccomp_evidence=seccomp_evidence,
        )
    assert wrong_mount.value.code == TOOL.FAIL_POLICY
    hardening_mutations = (
        lambda value: value[0]["HostConfig"].__setitem__("Privileged", True),
        lambda value: value[0]["HostConfig"].__setitem__("PidMode", "host"),
        lambda value: value[0]["HostConfig"].__setitem__("CgroupnsMode", "host"),
        lambda value: value[0]["HostConfig"].__setitem__(
            "DeviceRequests", [{"Driver": "nvidia"}]
        ),
        lambda value: value[0]["HostConfig"].__setitem__(
            "Ulimits", [{"Name": "nofile", "Hard": 257, "Soft": 256}]
        ),
        lambda value: value[0]["Config"].__setitem__("Cmd", ["/bin/sh"]),
        lambda value: value[0]["Config"]["Env"].append("LEAKED_SECRET=1"),
        lambda value: value[0]["HostConfig"].__setitem__(
            "SecurityOpt",
            ["no-new-privileges:almost", mount_registry.security_options[1]],
        ),
    )
    for mutate in hardening_mutations:
        changed_document = json.loads(inspect_payload)
        mutate(changed_document)
        changed_payload = json.dumps(
            changed_document,
            separators=(",", ":"),
        ).encode("ascii")
        with pytest.raises(TOOL.Q05BDualSupervisorError) as changed_policy:
            TOOL.live_resource_transcript_v1(
                1,
                container_id,
                docker_slot_row["container_name"],
                mount_registry,
                changed_payload,
                cgroup,
                proc_cgroup,
                cgroup_path,
                (31, 41),
                proc_limits,
                seccomp_evidence=seccomp_evidence,
            )
        assert changed_policy.value.code == TOOL.FAIL_POLICY
    proc_root = tmp_path / "proc"
    pid_root = proc_root / "1234"
    pid_root.mkdir(parents=True)
    (pid_root / "cgroup").write_bytes(proc_cgroup)
    (pid_root / "limits").write_bytes(proc_limits)
    cgroup_root = tmp_path / "cgroupfs"
    bound_cgroup = cgroup_root / "docker" / container_id
    bound_cgroup.mkdir(parents=True)
    final_cgroup = dict(cgroup)
    final_cgroup["memory.peak"] = b"999\n"
    for name, payload in final_cgroup.items():
        (bound_cgroup / name).write_bytes(payload)
    collected = TOOL.collect_bound_live_resource_transcript_v1(
        1,
        container_id,
        docker_slot_row["container_name"],
        mount_registry,
        inspect_payload,
        lambda: inspect_payload,
        seccomp_evidence=seccomp_evidence,
        proc_root=proc_root,
        cgroup_root=cgroup_root,
    )
    assert collected["anchored_collection"] is True
    assert collected["memory_peak_bytes"] == 999
    with pytest.raises(TOOL.Q05BDualSupervisorError) as precaptured_after:
        TOOL.collect_bound_live_resource_transcript_v1(
            1,
            container_id,
            docker_slot_row["container_name"],
            mount_registry,
            inspect_payload,
            inspect_payload,  # type: ignore[arg-type]
            seccomp_evidence=seccomp_evidence,
            proc_root=proc_root,
            cgroup_root=cgroup_root,
        )
    assert precaptured_after.value.code == TOOL.FAIL_POLICY
    (pid_root / "cgroup").write_text(
        f"0::/docker/{'cd' * 32}\n",
        encoding="ascii",
    )
    with pytest.raises(TOOL.Q05BDualSupervisorError) as forged_cgroup:
        TOOL.collect_bound_live_resource_transcript_v1(
            1,
            container_id,
            docker_slot_row["container_name"],
            mount_registry,
            inspect_payload,
            lambda: inspect_payload,
            seccomp_evidence=seccomp_evidence,
            proc_root=proc_root,
            cgroup_root=cgroup_root,
        )
    assert forged_cgroup.value.code == TOOL.FAIL_POLICY
    (pid_root / "cgroup").write_bytes(proc_cgroup)
    post_exit = json.dumps(
        [
            {
                "Id": container_id,
                "State": {"Running": False, "OOMKilled": False, "ExitCode": 0},
                "HostConfig": {
                    "AutoRemove": False,
                    "SecurityOpt": [
                        "no-new-privileges",
                        _docker29_inline_seccomp_option(seccomp_payload),
                    ],
                },
            }
        ],
        separators=(",", ":"),
    ).encode("ascii")
    control = tmp_path / "resource-control"
    control.mkdir(mode=0o700)
    actor_stdout = _valid_actor_stdout(
        "PYTHON_ENDPOINT",
        "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_PYTHON_V1",
    )
    for name, payload in (
        ("actor.stdout", actor_stdout),
        ("exit-code", TOOL.HELD_SUCCESS_EXIT_BYTES),
        ("done", TOOL.HELD_DONE_BYTES),
    ):
        path = control / name
        path.write_bytes(payload)
        path.chmod(0o600)
    completion = TOOL.seal_held_actor_completion_v1(control, "PYTHON_ENDPOINT")
    held_sample = TOOL.held_final_resource_sample_v1(
        control,
        completion,
        1,
        container_id,
        docker_slot_row["container_name"],
        mount_registry,
        inspect_payload,
        lambda: inspect_payload,
        seccomp_evidence=seccomp_evidence,
        proc_root=proc_root,
        cgroup_root=cgroup_root,
    )
    continuous_sample = dict(collected)
    continuous_sample["sample_ordinal"] = 1
    continuous_sample["sample_monotonic_ns"] = (
        held_sample["sample_monotonic_ns"] - 600_000_000
    )
    # Collection itself may exceed 250 ms; the contract bounds only the
    # unsampled gap from the preceding collection finish to the next start.
    continuous_sample["sample_duration_ns"] = 500_000_000
    held_sample["sample_ordinal"] = 2
    final = TOOL.final_resource_transcript_v1(
        (continuous_sample, held_sample),
        post_exit,
        command_security_options=mount_registry.security_options,
        seccomp_evidence=seccomp_evidence,
    )
    assert final["continuous_sampling_through_child_completion"] is True
    assert final["fresh_held_final_before_release"] is True
    assert final["post_release_wrapper_only_exits"] is True
    assert final["peak_scope"] == "CHILD_PLUS_WRAPPER_THROUGH_HELD_FINAL_SAMPLE"
    assert final["explicit_remove_admitted_after_this_transcript"] is True
    assert final["final_memory_peak_bytes"] == 999
    assert final["maximum_inter_sample_gap_ns"] == 100_000_000
    assert len(final["live_sample_objects"]) == 2
    excessive_gap = dict(continuous_sample)
    excessive_gap["sample_monotonic_ns"] -= 200_000_001
    with pytest.raises(TOOL.Q05BDualSupervisorError) as sampling_gap:
        TOOL.final_resource_transcript_v1(
            (excessive_gap, held_sample),
            post_exit,
            command_security_options=mount_registry.security_options,
            seccomp_evidence=seccomp_evidence,
        )
    assert sampling_gap.value.code == TOOL.FAIL_POLICY
    assert TOOL.docker_explicit_remove_command_v1(container_id)[2:] == [
        "rm",
        container_id,
    ]
    release = TOOL.release_held_actor_v1(control, completion, held_sample)
    TOOL.validate_held_actor_exit_v1(control, release, actor_stdout, 0)
    control.chmod(0o755)
    changed = dict(cgroup)
    changed["memory.events"] = changed["memory.events"].replace(
        b"oom_kill 0", b"oom_kill 1"
    )
    with pytest.raises(TOOL.Q05BDualSupervisorError) as oom:
        TOOL.live_resource_transcript_v1(
            1,
            container_id,
            docker_slot_row["container_name"],
            mount_registry,
            inspect_payload,
            changed,
            proc_cgroup,
            cgroup_path,
            (31, 41),
            proc_limits,
            seccomp_evidence=seccomp_evidence,
        )
    assert oom.value.code == TOOL.FAIL_POLICY


def test_held_actor_done_seal_final_sample_release_and_stdout_forward(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    actor_stdout = _valid_actor_stdout(
        "PYTHON_ENDPOINT",
        "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_PYTHON_V1",
    )
    for name, payload in (
        ("actor.stdout", actor_stdout),
        ("exit-code", TOOL.HELD_SUCCESS_EXIT_BYTES),
        ("done", TOOL.HELD_DONE_BYTES),
    ):
        path = control / name
        path.write_bytes(payload)
        path.chmod(0o600)
    completion = TOOL.seal_held_actor_completion_v1(
        control,
        "PYTHON_ENDPOINT",
    )
    assert not (control / "release").exists()
    with pytest.raises(TypeError):
        TOOL.held_final_resource_sample_v1(  # type: ignore[call-arg]
            {"captured_while_running": True},
            completion,
        )
    with pytest.raises(TOOL.Q05BDualSupervisorError) as shallow_release:
        TOOL.release_held_actor_v1(
            control,
            completion,
            {
                "captured_while_running": True,
                "actor_child_complete_held": True,
                "completion_manifest_sha256": completion["manifest_sha256"],
            },
        )
    assert shallow_release.value.code == TOOL.FAIL_POLICY
    assert not (control / "release").exists()
    control.chmod(0o755)


def test_bounded_pipe_drain_discards_overflow_without_unbounded_memory() -> None:
    payload = b"x" * 100_000
    state = TOOL.BoundedPipeDrainV1(
        1024,
        bytearray(),
        0,
        False,
        sha256(),
        [],
    )
    TOOL._drain_pipe_v1(io.BytesIO(payload), state)
    assert state.total == len(payload)
    assert state.overflow is True
    assert len(state.payload) == 1025
    assert state.digest.hexdigest() == sha256(payload).hexdigest()
    assert state.errors == []


def test_cidfile_is_anchored_nlink1_and_replacement_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_id = "ab" * 32

    def new_parent(name: str) -> tuple[Path, tuple[int, int, int, int]]:
        parent = tmp_path / name
        parent.mkdir(mode=0o700)
        status = parent.lstat()
        return parent, (
            status.st_dev,
            status.st_ino,
            stat.S_IMODE(status.st_mode),
            status.st_nlink,
        )

    parent, identity = new_parent("valid")
    cidfile = parent / "actor.cid"
    cidfile.write_bytes(container_id.encode("ascii"))
    cidfile.chmod(0o600)
    observed, evidence = TOOL._seal_cidfile_v1(cidfile, identity)
    assert observed == container_id
    assert stat.S_IMODE(cidfile.stat().st_mode) == 0o444
    assert evidence["file_nlink"] == 1
    assert evidence["payload_sha256"] == sha256(cidfile.read_bytes()).hexdigest()

    hard_parent, hard_identity = new_parent("hardlink")
    hard = hard_parent / "actor.cid"
    hard.write_bytes(container_id.encode("ascii"))
    hard.chmod(0o600)
    os.link(hard, hard_parent / "alias.cid")
    with pytest.raises(TOOL.Q05BDualSupervisorError) as hardlink:
        TOOL._seal_cidfile_v1(hard, hard_identity)
    assert hardlink.value.code == TOOL.FAIL_POLICY

    symlink_parent, symlink_identity = new_parent("symlink")
    target = symlink_parent / "target.cid"
    target.write_bytes(container_id.encode("ascii"))
    target.chmod(0o600)
    symlink = symlink_parent / "actor.cid"
    symlink.symlink_to(target.name)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as symlinked:
        TOOL._seal_cidfile_v1(symlink, symlink_identity)
    assert symlinked.value.code == TOOL.FAIL_POLICY

    newline_parent, newline_identity = new_parent("legacy-newline")
    newline = newline_parent / "actor.cid"
    newline.write_bytes((container_id + "\n").encode("ascii"))
    newline.chmod(0o600)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as legacy_newline:
        TOOL._seal_cidfile_v1(newline, newline_identity)
    assert legacy_newline.value.code == TOOL.FAIL_POLICY

    replaced_parent, replaced_identity = new_parent("replaced")
    replaced = replaced_parent / "actor.cid"
    replaced.write_bytes(container_id.encode("ascii"))
    replaced.chmod(0o600)
    real_fchmod = TOOL.os.fchmod
    replaced_once = False

    def replace_after_chmod(descriptor: int, mode: int) -> None:
        nonlocal replaced_once
        real_fchmod(descriptor, mode)
        if not replaced_once:
            replaced_once = True
            replacement = replaced_parent / "replacement.cid"
            replacement.write_bytes(("cd" * 32).encode("ascii"))
            replacement.chmod(0o444)
            os.replace(replacement, replaced)

    monkeypatch.setattr(TOOL.os, "fchmod", replace_after_chmod)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as replacement:
        TOOL._seal_cidfile_v1(replaced, replaced_identity)
    assert replacement.value.code == TOOL.FAIL_POLICY


def test_cidfile_readiness_rejects_replacement_and_symlink(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "cid-readiness"
    parent.mkdir(mode=0o700)
    parent_status = parent.lstat()
    parent_identity = (
        parent_status.st_dev,
        parent_status.st_ino,
        stat.S_IMODE(parent_status.st_mode),
        parent_status.st_nlink,
    )
    cidfile = parent / "actor.cid"
    cidfile.write_bytes(b"")
    cidfile.chmod(0o600)
    ready, first_identity = TOOL._observe_unsealed_cidfile_v1(
        cidfile,
        parent_identity,
    )
    assert ready is False
    assert first_identity == (cidfile.stat().st_dev, cidfile.stat().st_ino)

    cidfile.write_bytes(b"ab" * 16)
    cidfile.chmod(0o600)
    ready, same_identity = TOOL._observe_unsealed_cidfile_v1(
        cidfile,
        parent_identity,
        first_identity,
    )
    assert ready is False
    assert same_identity == first_identity

    replacement = parent / "replacement.cid"
    replacement.write_bytes(("cd" * 32).encode("ascii"))
    replacement.chmod(0o600)
    os.replace(replacement, cidfile)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as replaced:
        TOOL._observe_unsealed_cidfile_v1(
            cidfile,
            parent_identity,
            first_identity,
        )
    assert replaced.value.code == TOOL.FAIL_POLICY

    cidfile.unlink()
    target = parent / "target.cid"
    target.write_bytes(("ef" * 32).encode("ascii"))
    target.chmod(0o600)
    cidfile.symlink_to(target.name)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as symlinked:
        TOOL._observe_unsealed_cidfile_v1(cidfile, parent_identity)
    assert symlinked.value.code == TOOL.FAIL_POLICY


@pytest.mark.parametrize("slot", ("RUST_TEST", "RUST_RELEASE"))
def test_offline_build_waits_for_delayed_complete_cidfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    slot: str,
) -> None:
    authority, slot_row, cidfile, command = _offline_rust_test_fixture_v1(
        tmp_path,
        slot=slot,
    )
    container_id = "ab" * 32

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode: int | None = None

        def poll(self):
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout=None):
            del timeout
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    process = FakeProcess()

    def popen(*_args, **_kwargs):
        cidfile.write_bytes(b"")
        cidfile.chmod(0o600)
        return process

    monkeypatch.setattr(TOOL.subprocess, "Popen", popen)
    sleep_count = 0

    def advance_cidfile(_seconds: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        assert stat.S_IMODE(cidfile.stat().st_mode) == 0o600
        if sleep_count == 1:
            cidfile.write_bytes(container_id[:32].encode("ascii"))
        elif sleep_count == 2:
            cidfile.write_bytes(container_id.encode("ascii"))
        cidfile.chmod(0o600)

    monkeypatch.setattr(TOOL.time, "sleep", advance_cidfile)

    def inspect_reader(observed_id: str) -> bytes:
        assert observed_id == container_id
        running = process.returncode is None
        return _owned_inspect_payload(
            authority,
            slot,
            command,
            container_id,
            running=running,
        )

    docker_calls: list[list[str]] = []

    def runner(row, **_kwargs):
        command_row = list(row)
        docker_calls.append(command_row)
        if command_row[2] == "inspect":
            return SimpleNamespace(
                returncode=1,
                stdout=b"",
                stderr=(
                    f"Error: No such object: {command_row[-1]}\n"
                ).encode("ascii"),
            )
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    evidence = TOOL.run_offline_rust_build_container_v1(
        command,
        cidfile,
        docker_execution_authority=authority,
        docker_slot=slot,
        timeout_seconds=2.0,
        inspect_reader=inspect_reader,
        command_runner=runner,
    )
    assert sleep_count == 2
    assert evidence["exit_code"] == 0
    assert evidence["cidfile_evidence"]["file_size"] == 64
    assert evidence["cidfile_evidence"]["payload_hex"] == container_id.encode(
        "ascii"
    ).hex()
    assert stat.S_IMODE(cidfile.stat().st_mode) == 0o444
    assert [row[2] for row in docker_calls] == [
        "inspect",
        "inspect",
        "rm",
        "inspect",
    ]
    assert docker_calls[0][-1] == slot_row["container_name"]
    assert docker_calls[1][-1] == slot_row["container_name"]
    assert docker_calls[2][-1] == container_id
    assert docker_calls[3][-1] == container_id


def test_offline_build_empty_cidfile_times_out_with_read_only_name_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, slot_row, cidfile, command = _offline_rust_test_fixture_v1(
        tmp_path
    )

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode: int | None = None
            self.killed = False

        def poll(self):
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def wait(self, timeout=None):
            del timeout
            if self.returncode is None:
                self.returncode = -9
            return self.returncode

    process = FakeProcess()

    def popen(*_args, **_kwargs):
        cidfile.write_bytes(b"")
        cidfile.chmod(0o600)
        return process

    monkeypatch.setattr(TOOL.subprocess, "Popen", popen)
    docker_calls: list[list[str]] = []

    def runner(row, **_kwargs):
        command_row = list(row)
        docker_calls.append(command_row)
        if command_row[2] == "inspect":
            return SimpleNamespace(
                returncode=1,
                stdout=b"",
                stderr=(
                    f"Error: No such object: {command_row[-1]}\n"
                ).encode("ascii"),
            )
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    with pytest.raises(TOOL.Q05BDualSupervisorError) as timeout:
        TOOL.run_offline_rust_build_container_v1(
            command,
            cidfile,
            docker_execution_authority=authority,
            docker_slot="RUST_TEST",
            timeout_seconds=0.01,
            command_runner=runner,
        )
    assert timeout.value.code == TOOL.FAIL_POLICY
    assert "offline Rust cidfile timed out" in timeout.value.detail
    assert "owned-container-id-unresolved:potential-late-create" in (
        timeout.value.detail
    )
    assert process.killed is True
    assert all(row[2] == "inspect" for row in docker_calls)
    assert len(docker_calls) == 22
    assert {row[-1] for row in docker_calls} == {
        slot_row["container_name"]
    }
    assert cidfile.read_bytes() == b""
    assert stat.S_IMODE(cidfile.stat().st_mode) == 0o600


def test_offline_build_rejected_cid_replacement_is_never_cleanup_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, slot_row, cidfile, command = _offline_rust_test_fixture_v1(
        tmp_path
    )
    cid_parent = cidfile.parent
    replacement_id = "cd" * 32

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode: int | None = None

        def poll(self):
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout=None):
            del timeout
            if self.returncode is None:
                self.returncode = -9
            return self.returncode

    process = FakeProcess()

    def popen(*_args, **_kwargs):
        cidfile.write_bytes(b"")
        cidfile.chmod(0o600)
        return process

    monkeypatch.setattr(TOOL.subprocess, "Popen", popen)
    replaced = False

    def replace_once(_seconds: float) -> None:
        nonlocal replaced
        if replaced:
            return
        replacement = cid_parent / "replacement.cid"
        replacement.write_bytes(replacement_id.encode("ascii"))
        replacement.chmod(0o600)
        os.replace(replacement, cidfile)
        replaced = True

    monkeypatch.setattr(TOOL.time, "sleep", replace_once)
    docker_calls: list[list[str]] = []

    def runner(row, **_kwargs):
        command_row = list(row)
        docker_calls.append(command_row)
        if command_row[2] == "inspect":
            return SimpleNamespace(
                returncode=1,
                stdout=b"",
                stderr=(
                    f"Error: No such object: {command_row[-1]}\n"
                ).encode("ascii"),
            )
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        TOOL.run_offline_rust_build_container_v1(
            command,
            cidfile,
            docker_execution_authority=authority,
            docker_slot="RUST_TEST",
            timeout_seconds=2.0,
            command_runner=runner,
        )
    assert rejected.value.code == TOOL.FAIL_POLICY
    assert "replaced while waiting" in rejected.value.detail
    assert "owned-container-id-unresolved:potential-late-create" in (
        rejected.value.detail
    )
    assert replaced is True
    assert all(row[-1] != replacement_id for row in docker_calls)
    assert {row[-1] for row in docker_calls} == {
        slot_row["container_name"]
    }
    assert all(row[2] == "inspect" for row in docker_calls)
    assert cidfile.read_bytes() == replacement_id.encode("ascii")
    assert stat.S_IMODE(cidfile.stat().st_mode) == 0o600


def test_held_cleanup_never_reacquires_replacement_cid(
    tmp_path: Path,
) -> None:
    (
        actor,
        _inspect_reader,
        _proc_root,
        _cgroup_root,
        command_runner,
        docker_calls,
        control,
        _actor_stdout,
    ) = _mock_held_python_actor(tmp_path, "cleanup-replaced-cid")
    replacement_id = "cd" * 32
    actor.container_id = None
    actor.cidfile_evidence = None
    actor.cidfile.unlink()
    actor.cidfile.write_bytes(replacement_id.encode("ascii"))
    actor.cidfile.chmod(0o600)

    cleanup_errors = TOOL._abort_held_actor_cleanup_v1(actor, command_runner)
    assert cleanup_errors == (
        "docker-owned-container-id-unresolved:potential-late-create",
    )
    assert all(row[-1] != replacement_id for row in docker_calls)
    assert {row[-1] for row in docker_calls} == {actor.container_name}
    assert actor.container_id is None
    assert stat.S_IMODE(actor.cidfile.stat().st_mode) == 0o600
    control.chmod(0o755)


def test_close_held_actor_walks_final_sample_stdout_exit_and_remove(
    tmp_path: Path,
) -> None:
    (
        actor,
        inspect_reader,
        proc_root,
        cgroup_root,
        command_runner,
        docker_calls,
        control,
        actor_stdout,
    ) = _mock_held_python_actor(tmp_path, "success")
    evidence = TOOL.close_held_actor_process_v1(
        actor,
        child_timeout_seconds=2.0,
        inspect_reader=inspect_reader,
        proc_root=proc_root,
        cgroup_root=cgroup_root,
        command_runner=command_runner,
    )
    assert evidence["stdout_hex"] == actor_stdout.hex()
    assert evidence["stderr_length"] == 0
    assert evidence["continuous_sample_count"] == 1
    assert evidence["cidfile_evidence"] == actor.cidfile_evidence
    assert evidence["final_resource_transcript"]["sample_count"] == 2
    assert evidence["final_resource_transcript"][
        "continuous_sampling_through_child_completion"
    ] is True
    assert docker_calls == [
        TOOL.docker_explicit_remove_command_v1("ab" * 32),
        [
            TOOL.DOCKER_EXECUTABLE,
            f"--host={TOOL.DOCKER_HOST}",
            "inspect",
            "ab" * 32,
        ],
    ]
    assert evidence["docker_absence_evidence"]["inspect_exit_code"] == 1
    assert actor.process.poll() == 0
    assert (control / "release").read_bytes() == TOOL.HELD_RELEASE_BYTES
    control.chmod(0o755)


def test_close_held_actor_sampler_failure_force_removes_owned_id_only(
    tmp_path: Path,
) -> None:
    (
        actor,
        inspect_reader,
        proc_root,
        cgroup_root,
        command_runner,
        docker_calls,
        control,
        _actor_stdout,
    ) = _mock_held_python_actor(tmp_path, "sampler-failure")
    actor.sample_errors.append("forced sampler failure")
    removed = False

    def owned_runner(command, **_kwargs):
        nonlocal removed
        row = list(command)
        docker_calls.append(row)
        assert row[-1] == actor.container_id
        if row[2] == "inspect":
            if removed:
                return SimpleNamespace(
                    returncode=1,
                    stdout=b"",
                    stderr=(
                        f"Error: No such object: {actor.container_id}\n"
                    ).encode("ascii"),
                )
            return SimpleNamespace(
                returncode=0,
                stdout=inspect_reader(actor.container_id),
                stderr=b"",
            )
        assert row[2:4] == ["rm", "-f"]
        removed = True
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.close_held_actor_process_v1(
            actor,
            child_timeout_seconds=2.0,
            inspect_reader=inspect_reader,
            proc_root=proc_root,
            cgroup_root=cgroup_root,
            command_runner=owned_runner,
        )
    assert failure.value.code == TOOL.FAIL_POLICY
    assert removed is True
    assert not any(call[2] == "kill" for call in docker_calls)
    assert [call[2:4] for call in docker_calls] == [
        ["inspect", actor.container_id],
        ["rm", "-f"],
        ["inspect", actor.container_id],
    ]
    assert all(call[-1] == actor.container_id for call in docker_calls)
    assert docker_calls[-1][2] == "inspect"
    assert actor.cleanup_errors == []
    assert actor.process.poll() is not None
    assert (control / "release").read_bytes() == b"ABORT_NO_QUALIFICATION\n"
    control.chmod(0o755)


def test_wait_for_held_child_done_rechecks_sampler_error_after_complete_race() -> None:
    sample_errors: list[str] = []

    class CompleteAfterPublishingError:
        def is_set(self) -> bool:
            sample_errors.append("RuntimeError: exact sampler root cause")
            return True

    actor = SimpleNamespace(
        sample_errors=sample_errors,
        process=SimpleNamespace(poll=lambda: None),
        child_done_observed=SimpleNamespace(wait=lambda _timeout: False),
        sample_complete=CompleteAfterPublishingError(),
    )
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL._wait_for_held_child_done_v1(actor, 1.0)
    assert failure.value.code == TOOL.FAIL_POLICY
    assert failure.value.detail == (
        "held actor sampler failed: RuntimeError: exact sampler root cause"
    )


def test_wait_for_held_child_done_complete_without_error_remains_generic() -> None:
    actor = SimpleNamespace(
        sample_errors=[],
        process=SimpleNamespace(poll=lambda: None),
        child_done_observed=SimpleNamespace(wait=lambda _timeout: False),
        sample_complete=SimpleNamespace(is_set=lambda: True),
    )
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL._wait_for_held_child_done_v1(actor, 1.0)
    assert failure.value.code == TOOL.FAIL_POLICY
    assert failure.value.detail == "held actor sampler stopped before child done"


def test_wait_for_held_child_done_rechecks_sampler_error_at_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_errors: list[str] = []
    monotonic_values = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(TOOL.time, "monotonic", lambda: next(monotonic_values))

    class IncompleteAfterPublishingError:
        def is_set(self) -> bool:
            sample_errors.append("OSError: exact deadline sampler root cause")
            return False

    actor = SimpleNamespace(
        sample_errors=sample_errors,
        process=SimpleNamespace(poll=lambda: None),
        child_done_observed=SimpleNamespace(wait=lambda _timeout: False),
        sample_complete=IncompleteAfterPublishingError(),
    )
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL._wait_for_held_child_done_v1(actor, 1.0)
    assert failure.value.code == TOOL.FAIL_POLICY
    assert failure.value.detail == (
        "held actor sampler failed: OSError: exact deadline sampler root cause"
    )


def test_failed_owned_id_remove_is_reported_without_name_fallback(
    tmp_path: Path,
) -> None:
    (
        actor,
        inspect_reader,
        proc_root,
        cgroup_root,
        _command_runner,
        _docker_calls,
        control,
        _actor_stdout,
    ) = _mock_held_python_actor(tmp_path, "cleanup-residual")
    actor.sample_errors.append("forced sampler failure")
    calls: list[list[str]] = []

    def failing_runner(command, **_kwargs):
        row = list(command)
        calls.append(row)
        assert row[-1] == actor.container_id
        if row[2] == "inspect":
            return SimpleNamespace(
                returncode=0,
                stdout=inspect_reader(actor.container_id),
                stderr=b"",
            )
        assert row[2:4] == ["rm", "-f"]
        return SimpleNamespace(returncode=2, stdout=b"", stderr=b"forced\n")

    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.close_held_actor_process_v1(
            actor,
            child_timeout_seconds=2.0,
            inspect_reader=inspect_reader,
            proc_root=proc_root,
            cgroup_root=cgroup_root,
            command_runner=failing_runner,
        )
    assert failure.value.code == TOOL.FAIL_POLICY
    assert "cleanup closure failed" in failure.value.detail
    assert "Docker owned-ID cleanup removal failed" in failure.value.detail
    assert not any(call[2] == "kill" for call in calls)
    assert [call[2:4] for call in calls] == [
        ["inspect", actor.container_id],
        ["rm", "-f"],
    ]
    assert all(call[-1] == actor.container_id for call in calls)
    assert any("docker-owned-id-cleanup" in row for row in actor.cleanup_errors)
    assert actor.process.poll() is not None
    control.chmod(0o755)


def test_cleanup_actor_set_preserves_first_error_and_closes_second_owned_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _mock_held_python_actor(tmp_path, "cleanup-set-first")
    second = _mock_held_python_actor(tmp_path, "cleanup-set-second")
    first_actor, first_runner = first[0], first[4]
    second_actor, second_inspect = second[0], second[1]
    container_id = second_actor.container_id
    assert container_id is not None
    calls: list[list[str]] = []
    removed = False

    def second_runner(command, **_kwargs):
        nonlocal removed
        row = list(command)
        calls.append(row)
        assert row[-1] == container_id
        if row[2] == "inspect":
            if removed:
                return SimpleNamespace(
                    returncode=1,
                    stdout=b"",
                    stderr=f"Error: No such object: {container_id}\n".encode(
                        "ascii"
                    ),
                )
            return SimpleNamespace(
                returncode=0,
                stdout=second_inspect(container_id),
                stderr=b"",
            )
        assert row[2:4] == ["rm", "-f"]
        removed = True
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    real_abort = TOOL._abort_held_actor_cleanup_v1

    def first_fails(actor, runner):
        if actor is first_actor:
            return ("injected-first-cleanup-error",)
        return real_abort(actor, runner)

    with monkeypatch.context() as injected:
        injected.setattr(TOOL, "_abort_held_actor_cleanup_v1", first_fails)
        errors = TOOL._cleanup_actor_set_v1(
            (first_actor, second_actor),
            second_runner,
        )
    assert any("injected-first-cleanup-error" in row for row in errors)
    assert removed is True
    assert second_actor.process.poll() is not None
    assert not any(row[2] == "kill" for row in calls)
    assert all(row[-1] == container_id for row in calls if row[2] == "rm")
    assert real_abort(first_actor, first_runner) == ()
    first[6].chmod(0o755)
    second[6].chmod(0o755)


def test_abort_latch_interrupt_still_closes_and_revalidates_owned_id(
    tmp_path: Path,
) -> None:
    fixture = _mock_held_python_actor(tmp_path, "abort-latch-interrupt")
    actor, inspect_reader, _proc, _cgroup, _runner, _calls, control, _stdout = fixture
    assert actor.sample_thread is not None
    actor.sample_stop.set()
    actor.sample_thread.join(timeout=1.0)
    actor.sample_thread = None

    class InterruptingStop:
        @staticmethod
        def set() -> None:
            raise KeyboardInterrupt("injected immediately after cleanup latch")

    actor.sample_stop = InterruptingStop()
    container_id = actor.container_id
    assert container_id is not None
    calls: list[list[str]] = []
    removed = False

    def runner(command, **_kwargs):
        nonlocal removed
        row = list(command)
        calls.append(row)
        assert row[-1] == container_id
        if row[2] == "inspect":
            if removed:
                return SimpleNamespace(
                    returncode=1,
                    stdout=b"",
                    stderr=f"Error: No such object: {container_id}\n".encode(
                        "ascii"
                    ),
                )
            return SimpleNamespace(
                returncode=0,
                stdout=inspect_reader(container_id),
                stderr=b"",
            )
        assert row[2:4] == ["rm", "-f"]
        removed = True
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    errors = TOOL._abort_held_actor_cleanup_v1(actor, runner)
    assert removed is True
    assert actor.failure_cleanup_attempted is True
    assert actor.failure_cleanup_complete is True
    assert any("resource-sampler-stop:KeyboardInterrupt" in row for row in errors)
    call_count = len(calls)
    remove_count = sum(row[2] == "rm" for row in calls)
    actor.sample_stop = threading.Event()
    replay = TOOL._abort_held_actor_cleanup_v1(
        actor,
        runner,
    )
    assert replay == errors
    assert len(calls) == call_count + 1
    assert calls[-1][2:] == ["inspect", container_id]
    assert sum(row[2] == "rm" for row in calls) == remove_count
    control.chmod(0o755)


def test_abort_in_progress_fault_retries_before_owned_id_cleanup(
    tmp_path: Path,
) -> None:
    fixture = _mock_held_python_actor(tmp_path, "abort-in-progress-retry")
    actor, inspect_reader, _proc, _cgroup, _runner, _calls, control, _stdout = fixture
    assert actor.sample_thread is not None
    actor.sample_stop.set()
    actor.sample_thread.join(timeout=1.0)
    actor.sample_thread = None

    class InterruptingStop:
        @staticmethod
        def set() -> None:
            raise KeyboardInterrupt("injected after cleanup attempt store")

    class FailOnceErrors(list[str]):
        fail = True

        def append(self, value: str) -> None:
            if self.fail:
                self.fail = False
                raise MemoryError("injected before owned-ID resolver")
            super().append(value)

    actor.sample_stop = InterruptingStop()
    actor.cleanup_errors = FailOnceErrors()
    container_id = actor.container_id
    assert container_id is not None
    calls: list[list[str]] = []
    removed = False

    def runner(command, **_kwargs):
        nonlocal removed
        row = list(command)
        calls.append(row)
        assert row[-1] == container_id
        if row[2] == "inspect":
            if removed:
                return SimpleNamespace(
                    returncode=1,
                    stdout=b"",
                    stderr=f"Error: No such object: {container_id}\n".encode(
                        "ascii"
                    ),
                )
            return SimpleNamespace(
                returncode=0,
                stdout=inspect_reader(container_id),
                stderr=b"",
            )
        assert row[2:4] == ["rm", "-f"]
        removed = True
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    with pytest.raises(MemoryError, match="before owned-ID resolver"):
        TOOL._abort_held_actor_cleanup_v1(actor, runner)
    assert actor.failure_cleanup_attempted is True
    assert actor.failure_cleanup_complete is False
    assert calls == []

    errors = TOOL._abort_held_actor_cleanup_v1(actor, runner)
    assert removed is True
    assert actor.failure_cleanup_complete is True
    assert any("resource-sampler-stop:KeyboardInterrupt" in row for row in errors)
    call_count = len(calls)
    remove_count = sum(row[2] == "rm" for row in calls)
    actor.sample_stop = threading.Event()
    assert TOOL._abort_held_actor_cleanup_v1(actor, runner) == errors
    assert len(calls) == call_count + 1
    assert calls[-1][2:] == ["inspect", container_id]
    assert sum(row[2] == "rm" for row in calls) == remove_count
    control.chmod(0o755)


def test_outer_actor_sweep_retains_faulted_slot_and_closes_next_owned_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _mock_held_python_actor(tmp_path, "outer-sweep-first")
    second = _mock_held_python_actor(tmp_path, "outer-sweep-second")
    first_actor, first_runner = first[0], first[4]
    second_actor, second_runner = second[0], second[4]
    backend = SimpleNamespace(
        active_mount_binding_slots=[None, None, None],
        active_actor_slots=[first_actor, second_actor, None],
        endpoint_actors=None,
        host_actor=None,
        command_runner=second_runner,
    )
    real_abort = TOOL._abort_held_actor_cleanup_v1

    def first_interrupts(actor, runner):
        if actor is first_actor:
            raise KeyboardInterrupt("injected first-slot cleanup fault")
        return real_abort(actor, runner)

    with monkeypatch.context() as injected:
        injected.setattr(TOOL, "_abort_held_actor_cleanup_v1", first_interrupts)
        errors = TOOL._cleanup_failed_actual_backend_v1(backend)
    assert any("first-slot cleanup fault" in row for row in errors)
    assert backend.active_actor_slots[0] is first_actor
    assert backend.active_actor_slots[1] is None
    assert second_actor.failure_cleanup_complete is True
    assert second_actor.process.poll() is not None
    assert real_abort(first_actor, first_runner) == ()
    first[6].chmod(0o755)
    second[6].chmod(0o755)


@pytest.mark.parametrize("join_fault", (False, True))
def test_offline_build_drain_start_or_join_fault_removes_owned_id_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    join_fault: bool,
) -> None:
    authority, slot_row, cidfile, command = _offline_rust_test_fixture_v1(
        tmp_path
    )

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode: int | None = None
            self.killed = False

        def poll(self):
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def wait(self, timeout=None):
            del timeout
            if self.returncode is None:
                self.returncode = -9
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(TOOL.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(TOOL.time, "sleep", lambda _seconds: None)
    created = 0

    class FakeThread:
        def __init__(self, *_args, **_kwargs) -> None:
            nonlocal created
            created += 1
            self.ordinal = created
            self.ident: int | None = None

        def start(self) -> None:
            if self.ordinal == 2:
                raise RuntimeError("second drain start failed")
            self.ident = 1

        def join(self, timeout=None) -> None:
            del timeout
            if join_fault and self.ordinal == 1:
                raise RuntimeError("first drain join failed")

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(TOOL.threading, "Thread", FakeThread)
    docker_calls: list[list[str]] = []
    container_id = "45" * 32
    owned = _owned_inspect_payload(
        authority,
        "RUST_TEST",
        command,
        container_id,
    )
    name_inspects = 0
    removed = False

    def runner(row, **_kwargs):
        nonlocal name_inspects, removed
        command_row = list(row)
        docker_calls.append(command_row)
        target = command_row[-1]
        if command_row[2] == "inspect" and target == slot_row["container_name"]:
            name_inspects += 1
            if name_inspects <= 2:
                return SimpleNamespace(
                    returncode=1,
                    stdout=b"",
                    stderr=(
                        f"Error: No such object: {target}\n"
                    ).encode("ascii"),
                )
            return SimpleNamespace(returncode=0, stdout=owned, stderr=b"")
        if command_row[2] == "inspect" and target == container_id:
            if not removed:
                return SimpleNamespace(returncode=0, stdout=owned, stderr=b"")
            return SimpleNamespace(
                returncode=1,
                stdout=b"",
                stderr=(
                    f"Error: No such object: {target}\n"
                ).encode("ascii"),
            )
        assert command_row[2:4] == ["rm", "-f"]
        assert target == container_id
        removed = True
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    expected = TOOL.Q05BDualSupervisorError if join_fault else RuntimeError
    with pytest.raises(expected) as failure:
        TOOL.run_offline_rust_build_container_v1(
            command,
            cidfile,
            docker_execution_authority=authority,
            docker_slot="RUST_TEST",
            command_runner=runner,
        )
    if join_fault:
        assert "stdout-pipe-join" in failure.value.detail
    else:
        assert "second drain start failed" in str(failure.value)
    assert process.killed is True
    assert removed is True
    assert not any(row[2] == "kill" for row in docker_calls)
    assert all(
        row[-1] == container_id
        for row in docker_calls
        if row[2] == "rm"
    )


def test_cleanup_discovers_delayed_owned_container_but_removes_only_cid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _docker_execution_authority()
    slot = "RUST_TEST"
    row = _docker_slot_row(authority, slot)
    command = TOOL.rust_build_commands_v1(
        Path("/sealed/rust"),
        Path("/sealed/cargo"),
        Path("/sealed/target"),
        build_seccomp=Path("/sealed/seccomp.json"),
        docker_slot_row=row,
    )[0]
    principal = TOOL._docker_execution_principal_v1(command, authority, slot)
    container_id = "cd" * 32
    owned = _owned_inspect_payload(authority, slot, command, container_id)
    calls: list[list[str]] = []
    name_inspects = 0
    removed = False
    monkeypatch.setattr(TOOL.time, "sleep", lambda _seconds: None)

    def runner(argv, **_kwargs):
        nonlocal name_inspects, removed
        command_row = list(argv)
        calls.append(command_row)
        target = command_row[-1]
        if command_row[2] == "inspect" and target == row["container_name"]:
            name_inspects += 1
            if name_inspects < 3:
                return SimpleNamespace(
                    returncode=1,
                    stdout=b"",
                    stderr=f"Error: No such object: {target}\n".encode("ascii"),
                )
            return SimpleNamespace(returncode=0, stdout=owned, stderr=b"")
        if command_row[2] == "inspect" and target == container_id:
            if removed:
                return SimpleNamespace(
                    returncode=1,
                    stdout=b"",
                    stderr=f"Error: No such object: {target}\n".encode("ascii"),
                )
            return SimpleNamespace(returncode=0, stdout=owned, stderr=b"")
        assert command_row[2:4] == ["rm", "-f"]
        assert target == container_id
        removed = True
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    errors: list[str] = []
    TOOL._docker_remove_and_quiet_absence_v1(None, principal, runner, errors)
    assert errors == []
    assert removed is True
    assert all(
        call[-1] == container_id
        for call in calls
        if call[2] in {"kill", "rm"}
    )
    assert not any(call[2] == "kill" for call in calls)


def test_known_cid_cleanup_does_not_follow_same_name_aba_replacement() -> None:
    authority = _docker_execution_authority()
    slot = "RUST_RELEASE"
    row = _docker_slot_row(authority, slot)
    command = TOOL.rust_build_commands_v1(
        Path("/sealed/rust"),
        Path("/sealed/cargo"),
        Path("/sealed/target"),
        build_seccomp=Path("/sealed/seccomp.json"),
        docker_slot_row=row,
    )[1]
    principal = TOOL._docker_execution_principal_v1(command, authority, slot)
    container_id = "ef" * 32
    owned = _owned_inspect_payload(authority, slot, command, container_id)
    calls: list[list[str]] = []
    removed = False

    def runner(argv, **_kwargs):
        nonlocal removed
        command_row = list(argv)
        calls.append(command_row)
        assert command_row[-1] != row["container_name"]
        if command_row[2] == "inspect":
            if removed:
                return SimpleNamespace(
                    returncode=1,
                    stdout=b"",
                    stderr=(
                        f"Error: No such object: {container_id}\n"
                    ).encode("ascii"),
                )
            return SimpleNamespace(returncode=0, stdout=owned, stderr=b"")
        assert command_row[2:4] == ["rm", "-f"]
        assert command_row[-1] == container_id
        removed = True
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    errors: list[str] = []
    TOOL._docker_remove_and_quiet_absence_v1(
        container_id,
        principal,
        runner,
        errors,
    )
    assert errors == []
    assert removed is True
    assert all(call[-1] == container_id for call in calls)


def test_cleanup_rejects_bare_name_as_destructive_target_without_runner_call() -> None:
    authority = _docker_execution_authority()
    slot = "PYTHON_ENDPOINT"
    row = _docker_slot_row(authority, slot)
    command = TOOL.python_endpoint_command_v1(
        Path("/sealed/snapshot"),
        Path("/sealed/output"),
        Path("/sealed/control"),
        Path("/sealed/seccomp.json"),
        docker_slot_row=row,
    )
    principal = TOOL._docker_execution_principal_v1(command, authority, slot)
    calls: list[list[str]] = []
    errors: list[str] = []
    TOOL._docker_remove_and_quiet_absence_v1(
        row["container_name"],
        principal,
        lambda argv, **_kwargs: calls.append(list(argv)),
        errors,
    )
    assert calls == []
    assert errors == ["docker cleanup principal/quiet-window registry differs"]


def test_unknown_cid_foreign_same_name_is_never_mutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _docker_execution_authority()
    slot = "PYTHON_ENDPOINT"
    row = _docker_slot_row(authority, slot)
    command = TOOL.python_endpoint_command_v1(
        Path("/sealed/snapshot"),
        Path("/sealed/output"),
        Path("/sealed/control"),
        Path("/sealed/seccomp.json"),
        docker_slot_row=row,
    )
    principal = TOOL._docker_execution_principal_v1(command, authority, slot)
    foreign_value = json.loads(
        _owned_inspect_payload(authority, slot, command, "12" * 32)
    )
    foreign_value[0]["Config"]["Labels"][TOOL.DOCKER_EXECUTION_SLOT_LABEL] = (
        "TRUSTED_HOST_REPLAY"
    )
    foreign = json.dumps(foreign_value, separators=(",", ":")).encode("ascii")
    calls: list[list[str]] = []
    monkeypatch.setattr(TOOL.time, "sleep", lambda _seconds: None)

    def runner(argv, **_kwargs):
        command_row = list(argv)
        calls.append(command_row)
        assert command_row[2] == "inspect"
        assert command_row[-1] == row["container_name"]
        return SimpleNamespace(returncode=0, stdout=foreign, stderr=b"")

    errors: list[str] = []
    TOOL._docker_remove_and_quiet_absence_v1(None, principal, runner, errors)
    assert errors and "ownership principal differs" in errors[0]
    assert all(call[2] == "inspect" for call in calls)


def test_distinct_admission_nonces_have_disjoint_five_name_registries() -> None:
    first = _docker_execution_authority(nonce=b"A" * 32)
    second = _docker_execution_authority(nonce=b"B" * 32)
    first_names = {
        row["container_name"] for row in first["ordered_slot_rows"]
    }
    second_names = {
        row["container_name"] for row in second["ordered_slot_rows"]
    }
    assert len(first_names) == len(second_names) == 5
    assert first_names.isdisjoint(second_names)
    assert first["execution_namespace"] != second["execution_namespace"]
    assert first["attempt_nonce_sha256"] == sha256(b"A" * 32).hexdigest()
    assert all(len(name) < 128 for name in first_names | second_names)


def test_docker_commands_bind_exact_three_labels_and_distinct_build_names() -> None:
    authority = _docker_execution_authority()
    test_row = _docker_slot_row(authority, "RUST_TEST")
    release_row = _docker_slot_row(authority, "RUST_RELEASE")
    test_command = TOOL.rust_build_commands_v1(
        Path("/sealed/rust"),
        Path("/sealed/cargo"),
        Path("/sealed/target"),
        build_seccomp=Path("/sealed/seccomp.json"),
        docker_slot_row=test_row,
    )[0]
    release_command = TOOL.rust_build_commands_v1(
        Path("/sealed/rust"),
        Path("/sealed/cargo"),
        Path("/sealed/target"),
        build_seccomp=Path("/sealed/seccomp.json"),
        docker_slot_row=release_row,
    )[1]
    assert test_row["container_name"] != release_row["container_name"]
    for command, row in ((test_command, test_row), (release_command, release_row)):
        name_index = command.index("--name")
        assert command[name_index + 1] == row["container_name"]
        assert command[name_index + 2 : name_index + 5] == [
            f"--label={key}={value}" for key, value in row["labels"]
        ]
        assert [item for item in command if item.startswith("--label=")] == [
            f"--label={key}={value}" for key, value in row["labels"]
        ]


def test_precreate_foreign_name_fails_before_popen_and_never_mutates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _docker_execution_authority()
    slot = "RUST_TEST"
    row = _docker_slot_row(authority, slot)
    cid_parent = tmp_path / "cid"
    cid_parent.mkdir(mode=0o700)
    cidfile = cid_parent / "actor.cid"
    seccomp = tmp_path / TOOL.BUILD_SECCOMP_RELATIVE_PATH
    seccomp.parent.mkdir(parents=True)
    seccomp.write_bytes((ROOT / TOOL.BUILD_SECCOMP_RELATIVE_PATH).read_bytes())
    seccomp.chmod(0o444)
    command = TOOL.rust_build_commands_v1(
        Path("/sealed/rust"),
        Path("/sealed/cargo"),
        Path("/sealed/target"),
        "0" * 64,
        cidfile,
        build_seccomp=seccomp,
        docker_slot_row=row,
    )[0]
    foreign = json.loads(
        _owned_inspect_payload(authority, slot, command, "34" * 32)
    )
    foreign[0]["Config"]["Labels"][TOOL.DOCKER_EXECUTION_NAMESPACE_LABEL] = (
        "00" * 32
    )
    foreign_payload = json.dumps(
        foreign,
        separators=(",", ":"),
    ).encode("ascii")
    calls: list[list[str]] = []
    popen_calls: list[list[str]] = []
    monkeypatch.setattr(
        TOOL.subprocess,
        "Popen",
        lambda argv, **_kwargs: popen_calls.append(list(argv)),
    )

    def runner(argv, **_kwargs):
        command_row = list(argv)
        calls.append(command_row)
        assert command_row[2] == "inspect"
        assert command_row[-1] == row["container_name"]
        return SimpleNamespace(returncode=0, stdout=foreign_payload, stderr=b"")

    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.run_offline_rust_build_container_v1(
            command,
            cidfile,
            docker_execution_authority=authority,
            docker_slot=slot,
            command_runner=runner,
        )
    assert failure.value.code == TOOL.FAIL_POLICY
    assert "authoritative not-found" in failure.value.detail
    assert popen_calls == []
    assert all(call[2] == "inspect" for call in calls)


def test_backend_samples_admission_nonce_once_before_stage1(tmp_path: Path) -> None:
    project = tmp_path / "project"
    work = tmp_path / "work"
    cargo_cache = tmp_path / "cargo-cache"
    project.mkdir(mode=0o700)
    work.mkdir(mode=0o700)
    cargo_cache.mkdir(mode=0o700)
    calls: list[int] = []

    def nonce_source(size: int) -> bytes:
        calls.append(size)
        if len(calls) != 1:
            raise AssertionError("nonce source sampled more than once")
        return b"Q" * size

    backend = TOOL.ConcreteQ05BActualBackendV1(
        project,
        "ab" * 20,
        project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH,
        cargo_cache,
        work,
        admission_nonce_source=nonce_source,
    )
    assert calls == [32]
    assert backend.admission_nonce == b"Q" * 32
    assert not hasattr(backend, "admission_nonce_source")
    assert len(backend.docker_execution_slot_rows) == 5


def test_stage1_image_base_label_cross_rejects_tampered_rust_label() -> None:
    authority = _docker_execution_authority()

    def evidence(image: str, labels: dict[str, str] | None) -> dict[str, object]:
        raw = json.dumps(
            [{"Config": {"Labels": labels}}],
            separators=(",", ":"),
        ).encode("ascii")
        return {
            "requested_reference": image,
            "raw_inspect_hex": raw.hex(),
            "raw_inspect_sha256": sha256(raw).hexdigest(),
        }

    images = {
        "python": evidence(TOOL.PYTHON_IMAGE, None),
        "rust": evidence(
            TOOL.RUST_IMAGE,
            {
                "org.opencontainers.image.source": (
                    "https://github.com/rust-lang/docker-rust"
                )
            },
        ),
    }
    replay = TOOL.cross_docker_authority_to_pinned_image_labels_v1(
        authority,
        images,
    )
    assert replay["rust_base_labels"] == {
        "org.opencontainers.image.source": (
            "https://github.com/rust-lang/docker-rust"
        )
    }
    tampered = deepcopy(images)
    raw = json.dumps(
        [{"Config": {"Labels": {"org.opencontainers.image.source": "foreign"}}}],
        separators=(",", ":"),
    ).encode("ascii")
    tampered["rust"]["raw_inspect_hex"] = raw.hex()
    tampered["rust"]["raw_inspect_sha256"] = sha256(raw).hexdigest()
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.cross_docker_authority_to_pinned_image_labels_v1(
            authority,
            tampered,
        )
    assert failure.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    assert "base-label cross differs" in failure.value.detail


@pytest.mark.parametrize("kind", ("offline", "held"))
@pytest.mark.parametrize("fault_point", ("worker_construction", "guard_restore"))
def test_post_popen_handoff_fault_closes_owned_id_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    fault_point: str,
) -> None:
    authority = _docker_execution_authority()
    slot = "RUST_TEST" if kind == "offline" else "PYTHON_ENDPOINT"
    slot_row = _docker_slot_row(authority, slot)
    cid_parent = tmp_path / "cid"
    cid_parent.mkdir(mode=0o700)
    cidfile = cid_parent / "actor.cid"
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    relative_seccomp = (
        TOOL.BUILD_SECCOMP_RELATIVE_PATH
        if kind == "offline"
        else TOOL.RUNTIME_SECCOMP_RELATIVE_PATH
    )
    seccomp = tmp_path / relative_seccomp
    seccomp.parent.mkdir(parents=True, exist_ok=True)
    seccomp.write_bytes((ROOT / relative_seccomp).read_bytes())
    seccomp.chmod(0o444)
    if kind == "offline":
        command = TOOL.rust_build_commands_v1(
            Path("/sealed/rust"),
            Path("/sealed/cargo"),
            Path("/sealed/target"),
            "0" * 64,
            cidfile,
            build_seccomp=seccomp,
            docker_slot_row=slot_row,
        )[0]
    else:
        command = TOOL.python_endpoint_command_v1(
            Path("/sealed/snapshot"),
            Path("/sealed/output"),
            control,
            seccomp,
            docker_slot_row=slot_row,
            cidfile=cidfile,
        )
    container_id = "56" * 32
    owned = _owned_inspect_payload(authority, slot, command, container_id)

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode: int | None = None

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout=None):
            del timeout
            if self.returncode is None:
                self.returncode = -15
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(TOOL.subprocess, "Popen", lambda *_args, **_kwargs: process)
    if fault_point == "worker_construction":
        monkeypatch.setattr(
            TOOL,
            "_unstarted_pipe_workers_v1",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                MemoryError("injected post-Popen worker construction fault")
            ),
        )
    else:
        real_guard = TOOL._docker_ownership_signal_guard_v1
        guard_exits = 0

        @TOOL.contextmanager
        def faulting_guard():
            nonlocal guard_exits
            with real_guard():
                yield
            guard_exits += 1
            if guard_exits == 1:
                raise KeyboardInterrupt(
                    "injected at post-Popen ownership-guard restore"
                )

        monkeypatch.setattr(
            TOOL,
            "_docker_ownership_signal_guard_v1",
            faulting_guard,
        )
    monkeypatch.setattr(TOOL.time, "sleep", lambda _seconds: None)
    calls: list[list[str]] = []
    name_inspects = 0
    removed = False

    def runner(argv, **_kwargs):
        nonlocal name_inspects, removed
        command_row = list(argv)
        calls.append(command_row)
        target = command_row[-1]
        if command_row[2] == "inspect" and target == slot_row["container_name"]:
            name_inspects += 1
            if name_inspects <= 2:
                return SimpleNamespace(
                    returncode=1,
                    stdout=b"",
                    stderr=f"Error: No such object: {target}\n".encode("ascii"),
                )
            return SimpleNamespace(returncode=0, stdout=owned, stderr=b"")
        if command_row[2] == "inspect" and target == container_id:
            if removed:
                return SimpleNamespace(
                    returncode=1,
                    stdout=b"",
                    stderr=f"Error: No such object: {target}\n".encode("ascii"),
                )
            return SimpleNamespace(returncode=0, stdout=owned, stderr=b"")
        assert command_row[2:4] == ["rm", "-f"]
        assert target == container_id
        removed = True
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    expected = (
        TOOL.Q05BDualSupervisorError
        if fault_point == "guard_restore" and kind == "held"
        else (
            MemoryError
            if fault_point == "worker_construction"
            else KeyboardInterrupt
        )
    )
    expected_match = (
        "worker construction fault"
        if fault_point == "worker_construction"
        else "ownership-guard restore"
    )
    with pytest.raises(expected, match=expected_match):
        if kind == "offline":
            TOOL.run_offline_rust_build_container_v1(
                command,
                cidfile,
                docker_execution_authority=authority,
                docker_slot=slot,
                command_runner=runner,
            )
        else:
            TOOL.start_held_actor_process_v1(
                1,
                "PYTHON_ENDPOINT",
                slot_row["container_name"],
                command,
                cidfile,
                control,
                docker_execution_authority=authority,
                docker_slot=slot,
                command_runner=runner,
            )
    assert removed is True
    assert all(
        call[-1] == container_id
        for call in calls
        if call[2] in {"kill", "rm"}
    )
    assert not any(call[2] == "kill" for call in calls)


@pytest.mark.parametrize("join_fault", (False, True))
def test_held_sampler_start_or_pipe_join_fault_still_closes_owned_id_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    join_fault: bool,
) -> None:
    authority, slot_row, cidfile, control, command = (
        _held_python_start_fixture_v1(tmp_path)
    )
    container_id = "67" * 32
    owned = _owned_inspect_payload(
        authority,
        "PYTHON_ENDPOINT",
        command,
        container_id,
    )

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode: int | None = None

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout=None):
            del timeout
            if self.returncode is None:
                self.returncode = -15
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(TOOL.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(TOOL.time, "sleep", lambda _seconds: None)
    created = 0

    class FakeThread:
        def __init__(self, *_args, **_kwargs) -> None:
            nonlocal created
            created += 1
            self.ordinal = created
            self.ident: int | None = None

        def start(self) -> None:
            if self.ordinal == 3:
                raise RuntimeError("resource sampler start failed")
            self.ident = self.ordinal

        def join(self, timeout=None) -> None:
            del timeout
            if join_fault and self.ordinal == 1:
                raise RuntimeError("stdout join failed")

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(TOOL.threading, "Thread", FakeThread)
    calls: list[list[str]] = []
    name_inspects = 0
    removed = False

    def runner(argv, **_kwargs):
        nonlocal name_inspects, removed
        row = list(argv)
        calls.append(row)
        target = row[-1]
        if row[2] == "inspect" and target == slot_row["container_name"]:
            name_inspects += 1
            if name_inspects <= 2:
                return SimpleNamespace(
                    returncode=1,
                    stdout=b"",
                    stderr=f"Error: No such object: {target}\n".encode("ascii"),
                )
            return SimpleNamespace(returncode=0, stdout=owned, stderr=b"")
        if row[2] == "inspect" and target == container_id:
            if not removed:
                return SimpleNamespace(returncode=0, stdout=owned, stderr=b"")
            return SimpleNamespace(
                returncode=1,
                stdout=b"",
                stderr=f"Error: No such object: {target}\n".encode("ascii"),
            )
        assert row[2:4] == ["rm", "-f"]
        assert target == container_id
        removed = True
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.start_held_actor_process_v1(
            1,
            "PYTHON_ENDPOINT",
            slot_row["container_name"],
            command,
            cidfile,
            control,
            docker_execution_authority=authority,
            docker_slot="PYTHON_ENDPOINT",
            command_runner=runner,
        )
    assert failure.value.code == TOOL.FAIL_POLICY
    assert "resource sampler start failed" in failure.value.detail
    if join_fault:
        assert "stdout-pipe-join" in failure.value.detail
    assert removed is True
    assert not any(row[2] == "kill" for row in calls)
    assert all(row[-1] == container_id for row in calls if row[2] == "rm")


@pytest.mark.parametrize("kind", ("offline", "held"))
@pytest.mark.parametrize("ownership", ("owned", "foreign"))
def test_popen_side_effect_then_raise_uses_read_only_name_and_owned_cid_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    ownership: str,
) -> None:
    if kind == "offline":
        authority, slot_row, cidfile, command = _offline_rust_test_fixture_v1(
            tmp_path
        )
        slot = "RUST_TEST"
        control = None
    else:
        authority, slot_row, cidfile, control, command = (
            _held_python_start_fixture_v1(tmp_path)
        )
        slot = "PYTHON_ENDPOINT"
    container_id = "89" * 32
    owned = _owned_inspect_payload(
        authority,
        slot,
        command,
        container_id,
    )
    foreign_value = json.loads(owned)
    foreign_value[0]["Config"]["Labels"][
        TOOL.DOCKER_EXECUTION_NAMESPACE_LABEL
    ] = "00" * 32
    foreign = json.dumps(foreign_value, separators=(",", ":")).encode("ascii")
    popen_raised = False

    def popen(*_args, **_kwargs):
        nonlocal popen_raised
        popen_raised = True
        raise KeyboardInterrupt("post-fork injected failure")

    monkeypatch.setattr(TOOL.subprocess, "Popen", popen)
    monkeypatch.setattr(TOOL.time, "sleep", lambda _seconds: None)
    calls: list[list[str]] = []
    name_inspects = 0
    removed = False

    def runner(argv, **_kwargs):
        nonlocal name_inspects, removed
        row = list(argv)
        calls.append(row)
        target = row[-1]
        assert row[2] in {"inspect", "rm"}
        if row[2] == "inspect" and target == slot_row["container_name"]:
            name_inspects += 1
            if not popen_raised:
                return SimpleNamespace(
                    returncode=1,
                    stdout=b"",
                    stderr=f"Error: No such object: {target}\n".encode("ascii"),
                )
            return SimpleNamespace(
                returncode=0,
                stdout=owned if ownership == "owned" else foreign,
                stderr=b"",
            )
        if row[2] == "inspect" and target == container_id:
            if not removed:
                return SimpleNamespace(returncode=0, stdout=owned, stderr=b"")
            return SimpleNamespace(
                returncode=1,
                stdout=b"",
                stderr=f"Error: No such object: {target}\n".encode("ascii"),
            )
        assert ownership == "owned"
        assert row[2:4] == ["rm", "-f"]
        assert target == container_id
        removed = True
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    expected = KeyboardInterrupt if ownership == "owned" else TOOL.Q05BDualSupervisorError
    with pytest.raises(expected) as failure:
        if kind == "offline":
            TOOL.run_offline_rust_build_container_v1(
                command,
                cidfile,
                docker_execution_authority=authority,
                docker_slot=slot,
                command_runner=runner,
            )
        else:
            assert control is not None
            TOOL.start_held_actor_process_v1(
                1,
                "PYTHON_ENDPOINT",
                slot_row["container_name"],
                command,
                cidfile,
                control,
                docker_execution_authority=authority,
                docker_slot=slot,
                command_runner=runner,
            )
    assert name_inspects == 3
    if ownership == "owned":
        assert removed is True
        assert "post-fork injected failure" in str(failure.value)
        assert all(row[-1] == container_id for row in calls if row[2] == "rm")
    else:
        assert removed is False
        assert not any(row[2] == "rm" for row in calls)
        assert failure.value.code == TOOL.FAIL_POLICY
        assert "cleanup closure failed" in failure.value.detail


def _synthetic_docker_completion_ownership(
    authority: dict[str, object],
    slot: str,
    command: list[str],
    container_id: str = "78" * 32,
) -> dict[str, object]:
    slot_row = _docker_slot_row(authority, slot)
    principal = TOOL._docker_execution_principal_v1(command, authority, slot)
    live_payload = _owned_inspect_payload(
        authority,
        slot,
        command,
        container_id,
        running=True,
    )
    post_payload = _owned_inspect_payload(
        authority,
        slot,
        command,
        container_id,
        running=False,
    )
    cidfile = {
        "container_id": container_id,
        "payload_hex": container_id.encode("ascii").hex(),
    }
    return {
        "actor_id": slot,
        "container_id": container_id,
        "docker_execution_authority_manifest_sha256": authority[
            "manifest_sha256"
        ],
        "docker_execution_slot_row": slot_row,
        "ownership_label_root": principal["ownership_label_root"],
        "precreate_absence_evidence": ADMISSION.build_docker_precreate_absence_v1(
            authority,
            slot_row["slot_id"],
            _docker_absence_sample(slot_row["container_name"]),
            _docker_absence_sample(slot_row["container_name"]),
        ),
        "command_sha256": principal["command_sha256"],
        "cidfile_evidence": cidfile,
        "live_ownership_inspect_evidence": (
            TOOL._validate_owned_docker_inspect_payload_v1(
                live_payload,
                principal,
            )
        ),
        "post_exit_inspect_hex": post_payload.hex(),
        "post_exit_inspect_sha256": sha256(post_payload).hexdigest(),
        "post_ownership_inspect_evidence": (
            TOOL._validate_owned_docker_inspect_payload_v1(
                post_payload,
                principal,
            )
        ),
        "explicit_remove_exit_code": 0,
        "explicit_remove_command": TOOL.docker_explicit_remove_command_v1(
            container_id
        ),
        "cleanup_target_kind": "OWNERSHIP_VALIDATED_CONTAINER_ID",
        "container_name_was_never_a_destructive_target": True,
        "docker_absence_evidence": _docker_absence_sample(container_id),
    }


@pytest.mark.parametrize(
    "tamper",
    ("precreate", "ownership_label", "post_labels", "remove_target"),
)
def test_forged_closer_completion_docker_ownership_is_rejected(
    tamper: str,
) -> None:
    authority = _docker_execution_authority()
    slot = "PYTHON_ENDPOINT"
    slot_row = _docker_slot_row(authority, slot)
    command = TOOL.python_endpoint_command_v1(
        Path("/sealed/snapshot"),
        Path("/sealed/output"),
        Path("/sealed/control"),
        Path("/sealed/seccomp.json"),
        docker_slot_row=slot_row,
    )
    completion = _synthetic_docker_completion_ownership(
        authority,
        slot,
        command,
    )
    TOOL.strict_replay_docker_completion_ownership_v1(
        completion,
        authority,
        slot,
        command,
    )
    forged = deepcopy(completion)
    if tamper == "precreate":
        forged["precreate_absence_evidence"]["slot"] = "RUST_ENDPOINT"
    elif tamper == "ownership_label":
        forged["ownership_label_root"] = "00" * 32
    elif tamper == "post_labels":
        payload = json.loads(
            bytes.fromhex(
                forged["post_ownership_inspect_evidence"]["inspect_hex"]
            )
        )
        payload[0]["Config"]["Labels"][TOOL.DOCKER_EXECUTION_SLOT_LABEL] = (
            "RUST_ENDPOINT"
        )
        forged["post_ownership_inspect_evidence"]["inspect_hex"] = json.dumps(
            payload,
            separators=(",", ":"),
        ).encode("ascii").hex()
    else:
        forged["explicit_remove_command"][-1] = slot_row["container_name"]
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.strict_replay_docker_completion_ownership_v1(
            forged,
            authority,
            slot,
            command,
        )
    assert failure.value.code in {TOOL.FAIL_POLICY, TOOL.FAIL_ACTUAL_ADMISSION}


@pytest.mark.parametrize("tamper", ("principal", "precreate"))
def test_injected_actor_starter_ownership_is_replayed_before_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    authority = _docker_execution_authority()
    slot = "PYTHON_ENDPOINT"
    slot_row = _docker_slot_row(authority, slot)
    command = TOOL.python_endpoint_command_v1(
        Path("/sealed/snapshot"),
        Path("/sealed/output"),
        Path("/sealed/control"),
        Path("/sealed/seccomp.json"),
        docker_slot_row=slot_row,
        cidfile=Path("/sealed/cid/python.cid"),
    )
    principal = TOOL._docker_execution_principal_v1(command, authority, slot)
    precreate = ADMISSION.build_docker_precreate_absence_v1(
        authority,
        slot_row["slot_id"],
        _docker_absence_sample(slot_row["container_name"]),
        _docker_absence_sample(slot_row["container_name"]),
    )
    if tamper == "principal":
        principal = deepcopy(principal)
        principal["ownership_label_root"] = "00" * 32
    else:
        precreate = deepcopy(precreate)
        precreate["slot"] = "RUST_ENDPOINT"
    actor = TOOL.HeldActorProcessV1(
        role_id=1,
        actor_id=slot,
        container_name=slot_row["container_name"],
        command=tuple(command),
        cidfile=Path("/sealed/cid/python.cid"),
        control_root=Path("/sealed/control"),
        mount_registry=SimpleNamespace(),
        process=SimpleNamespace(),
        stdout_drain=TOOL.BoundedPipeDrainV1(
            1, bytearray(), 0, False, sha256(), []
        ),
        stderr_drain=TOOL.BoundedPipeDrainV1(
            1, bytearray(), 0, False, sha256(), []
        ),
        stdout_thread=threading.Thread(),
        stderr_thread=threading.Thread(),
        sample_thread=None,
        sample_rows=[],
        sample_errors=[],
        sample_complete=threading.Event(),
        child_done_observed=threading.Event(),
        sample_stop=threading.Event(),
        sample_lock=threading.Lock(),
        container_id=None,
        cid_parent_identity=(1, 2, 0o700, 2),
        cidfile_evidence=None,
        cleanup_errors=[],
        docker_execution_authority_manifest_sha256=authority[
            "manifest_sha256"
        ],
        docker_execution_slot_row=slot_row,
        ownership_label_root=principal["ownership_label_root"],
        precreate_absence_evidence=precreate,
        docker_execution_principal=principal,
    )
    backend = object.__new__(TOOL.ConcreteQ05BActualBackendV1)
    backend.docker_execution_authority = authority
    backend.actor_starter = lambda *_args, **_kwargs: actor
    backend.inspect_reader = lambda _container_id: b""
    backend.command_runner = lambda *_args, **_kwargs: None
    backend.actor_mount_bindings = {}
    backend.actor_mount_launch_replays = {}
    binding = SimpleNamespace(
        role_id=1,
        actor_id=slot,
        exact_command=tuple(command),
    )
    monkeypatch.setattr(
        TOOL,
        "close_held_actor_mount_binding_v1",
        lambda _binding: None,
    )

    def raise_original(_actors, _runner, original, _context):
        raise original

    monkeypatch.setattr(TOOL, "_raise_after_actor_cleanup_v1", raise_original)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        backend._launch_prepared_actor_mount_binding_v1(
            binding,
            slot,
            Path("/sealed/cid/python.cid"),
            Path("/sealed/control"),
        )
    assert failure.value.code == TOOL.FAIL_ACTUAL_ADMISSION


@pytest.mark.parametrize("discovery", ("owned", "foreign", "preset_flags"))
def test_injected_starter_side_effect_then_raise_uses_cid_only_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    discovery: str,
) -> None:
    authority = _docker_execution_authority()
    slot = "PYTHON_ENDPOINT"
    slot_row = _docker_slot_row(authority, slot)
    cidfile = Path("/sealed/cid/python.cid")
    control = Path("/sealed/control")
    command = TOOL.python_endpoint_command_v1(
        Path("/sealed/snapshot"),
        Path("/sealed/output"),
        control,
        Path("/sealed/seccomp.json"),
        docker_slot_row=slot_row,
        cidfile=cidfile,
    )
    registry = TOOL.sealed_actor_mount_registry_v1(1, command)
    binding = TOOL.HeldActorMountBindingV1(
        1,
        slot,
        tuple(command),
        registry,
        {"mount_binding_root": "11" * 32},
        (),
        SimpleNamespace(),
    )
    container_id = "91" * 32
    owned_payload = _owned_inspect_payload(
        authority,
        slot,
        command,
        container_id,
    )
    foreign_value = json.loads(owned_payload)
    foreign_value[0]["Config"]["Labels"][
        TOOL.DOCKER_EXECUTION_SLOT_LABEL
    ] = "RUST_ENDPOINT"
    foreign_payload = json.dumps(
        foreign_value,
        separators=(",", ":"),
    ).encode("ascii")
    calls: list[list[str]] = []
    removed = False

    def runner(docker_command, **_kwargs):
        nonlocal removed
        row = list(docker_command)
        calls.append(row)
        target = row[-1]
        if row[2] == "inspect":
            if target == slot_row["container_name"]:
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        owned_payload if discovery == "owned" else foreign_payload
                    ),
                    stderr=b"",
                )
            assert target == container_id
            if removed:
                return SimpleNamespace(
                    returncode=1,
                    stdout=b"",
                    stderr=f"Error: No such object: {container_id}\n".encode(
                        "ascii"
                    ),
                )
            return SimpleNamespace(
                returncode=0,
                stdout=owned_payload,
                stderr=b"",
            )
        assert row[2:4] == ["rm", "-f"]
        assert target == container_id
        removed = True
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    backend = object.__new__(TOOL.ConcreteQ05BActualBackendV1)
    backend.docker_execution_authority = authority
    backend.active_actor_slots = [None, None, None]

    if discovery == "preset_flags":
        class FinishedDockerCLI:
            @staticmethod
            def poll() -> int:
                return 0

            @staticmethod
            def wait(*, timeout: float) -> int:
                assert timeout == 2.0
                return 0

        actor = TOOL.HeldActorProcessV1(
            role_id=1,
            actor_id=slot,
            container_name=slot_row["container_name"],
            command=tuple(command),
            cidfile=cidfile,
            control_root=control,
            mount_registry=registry,
            process=FinishedDockerCLI(),
            stdout_drain=TOOL.BoundedPipeDrainV1(
                1, bytearray(), 0, False, sha256(), []
            ),
            stderr_drain=TOOL.BoundedPipeDrainV1(
                1, bytearray(), 0, False, sha256(), []
            ),
            stdout_thread=threading.Thread(),
            stderr_thread=threading.Thread(),
            sample_thread=None,
            sample_rows=[],
            sample_errors=[],
            sample_complete=threading.Event(),
            child_done_observed=threading.Event(),
            sample_stop=threading.Event(),
            sample_lock=threading.Lock(),
            container_id=container_id,
            cid_parent_identity=(1, 2, 0o700, 2),
            cidfile_evidence=None,
            cleanup_errors=[],
        )
        _bind_synthetic_actor_docker_ownership_v1(
            actor,
            authority,
            slot,
            command,
        )
        actor.failure_cleanup_attempted = True
        actor.failure_cleanup_complete = True

        def preset_starter(*_args, **kwargs):
            kwargs["ownership_sink"](actor)
            return actor

        backend.actor_starter = preset_starter
    else:
        backend.actor_starter = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("starter raised after daemon side effect")
        )
    backend.inspect_reader = lambda _container_id: b""
    backend.command_runner = runner
    backend.actor_mount_bindings = {}
    backend.actor_mount_launch_replays = {}
    monkeypatch.setattr(
        TOOL,
        "close_held_actor_mount_binding_v1",
        lambda _binding: None,
    )

    if discovery == "owned":
        with pytest.raises(
            RuntimeError,
            match="starter raised after daemon side effect",
        ):
            backend._launch_prepared_actor_mount_binding_v1(
                binding,
                slot,
                cidfile,
                control,
            )
        assert removed is True
    elif discovery == "foreign":
        with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
            backend._launch_prepared_actor_mount_binding_v1(
                binding,
                slot,
                cidfile,
                control,
            )
        assert rejected.value.code == TOOL.FAIL_POLICY
        assert "unbound starter cleanup closure failed" in rejected.value.detail
        assert removed is False
    else:
        with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
            backend._launch_prepared_actor_mount_binding_v1(
                binding,
                slot,
                cidfile,
                control,
            )
        assert rejected.value.code == TOOL.FAIL_ACTUAL_ADMISSION
        assert "different mount principal" in rejected.value.detail
        assert removed is True
    assert not any(
        row[2] == "rm" and row[-1] == slot_row["container_name"]
        for row in calls
    )
    assert all(
        row[-1] == container_id
        for row in calls
        if row[2] == "rm"
    )


def test_docker_absence_rejects_daemon_transport_failure() -> None:
    target = "hegel-q05b-daemon-down"

    def daemon_down(_row, **_kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout=b"",
            stderr=b"Cannot connect to the Docker daemon\n",
        )

    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL._docker_absence_evidence_v1(target, daemon_down)
    assert failure.value.code == TOOL.FAIL_POLICY
    assert "authoritative not-found" in failure.value.detail


def test_docker29_authoritative_absence_shape_is_target_bound() -> None:
    target = "hegel-q05b-docker29-absence"

    def docker29(_row, **_kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout=b"[]\n",
            stderr=f"error: no such object: {target}\n".encode("ascii"),
        )

    evidence = TOOL._docker_absence_evidence_v1(target, docker29)
    assert bytes.fromhex(evidence["inspect_stdout_hex"]) == b"[]\n"
    assert bytes.fromhex(evidence["inspect_stderr_hex"]) == (
        f"error: no such object: {target}\n".encode("ascii")
    )
    with pytest.raises(TOOL.Q05BDualSupervisorError):
        TOOL._docker_absence_evidence_v1(target + "-wrong", docker29)


def test_local_docker_absence_cli_matches_frozen_authoritative_contract() -> None:
    target = f"hegel-q05b-absent-{os.getpid()}-{time.monotonic_ns()}"
    command = [
        TOOL.DOCKER_EXECUTABLE,
        f"--host={TOOL.DOCKER_HOST}",
        "inspect",
        target,
    ]
    try:
        probe = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        pytest.skip("local pinned Docker CLI is unavailable")
    if b"Cannot connect to the Docker daemon" in probe.stderr:
        pytest.skip("local Docker daemon is unavailable")
    evidence = TOOL._docker_absence_evidence_v1(target, subprocess.run)
    assert evidence["inspect_exit_code"] == 1
    assert bytes.fromhex(evidence["inspect_stdout_hex"]) == probe.stdout
    assert bytes.fromhex(evidence["inspect_stderr_hex"]) == probe.stderr


def test_local_docker29_cidfile_is_exact_64_lowerhex_without_lf(
    tmp_path: Path,
) -> None:
    image_probe = subprocess.run(
        [
            TOOL.DOCKER_EXECUTABLE,
            f"--host={TOOL.DOCKER_HOST}",
            "image",
            "inspect",
            TOOL.RUST_IMAGE,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if image_probe.returncode != 0:
        pytest.skip("local pinned Rust image is unavailable")
    cid_parent = tmp_path / "cid-contract"
    cid_parent.mkdir(mode=0o700)
    cidfile = cid_parent / "docker.cid"
    parent_status = cid_parent.lstat()
    parent_identity = (
        parent_status.st_dev,
        parent_status.st_ino,
        stat.S_IMODE(parent_status.st_mode),
        parent_status.st_nlink,
    )
    name = f"hegel-q05b-cid-contract-{os.getpid()}-{time.monotonic_ns()}"
    command = [
        TOOL.DOCKER_EXECUTABLE,
        f"--host={TOOL.DOCKER_HOST}",
        "run",
        "--name",
        name,
        f"--cidfile={cidfile}",
        "--pull=never",
        "--network=none",
        TOOL.RUST_IMAGE,
        "sleep",
        "2",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        umask=0o077,
    )
    container_id: str | None = None
    observed_identity: tuple[int, int] | None = None
    try:
        deadline = time.monotonic() + 5.0
        ready = False
        while time.monotonic() < deadline:
            ready, observed_identity = TOOL._observe_unsealed_cidfile_v1(
                cidfile,
                parent_identity,
                observed_identity,
            )
            if ready:
                break
            if process.poll() is not None:
                pytest.fail("pinned Docker exited before a complete CID")
            time.sleep(0.01)
        assert ready is True
        assert observed_identity is not None
        payload = cidfile.read_bytes()
        unsealed = cidfile.lstat()
        assert len(payload) == 64
        assert re.fullmatch(rb"[0-9a-f]{64}", payload) is not None
        assert b"\n" not in payload
        assert stat.S_IMODE(unsealed.st_mode) == 0o600
        assert unsealed.st_nlink == 1
        assert unsealed.st_size == 64
        container_id, evidence = TOOL._seal_cidfile_v1(
            cidfile,
            parent_identity,
            observed_identity,
        )
        assert evidence["file_size"] == 64
        assert bytes.fromhex(evidence["payload_hex"]) == payload
        stdout, stderr = process.communicate(timeout=10.0)
        assert process.returncode == 0, (stdout, stderr)
        removed = subprocess.run(
            TOOL.docker_explicit_remove_command_v1(container_id),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert removed.returncode == 0, (removed.stdout, removed.stderr)
        TOOL._docker_absence_evidence_v1(container_id, subprocess.run)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)
        cleanup_mutations: list[list[str]] = []

        def inspect_target(target: str):
            return subprocess.run(
                [
                    TOOL.DOCKER_EXECUTABLE,
                    f"--host={TOOL.DOCKER_HOST}",
                    "inspect",
                    target,
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        def exact_contract_id(payload: bytes) -> str | None:
            try:
                value = TOOL._strict_json_value_v1(
                    payload,
                    "local Docker CID contract cleanup",
                )
            except TOOL.Q05BDualSupervisorError:
                return None
            if type(value) is not list or len(value) != 1:
                return None
            document = value[0]
            config = document.get("Config") if type(document) is dict else None
            host = document.get("HostConfig") if type(document) is dict else None
            discovered = document.get("Id") if type(document) is dict else None
            expected_labels = {
                "org.opencontainers.image.source": (
                    "https://github.com/rust-lang/docker-rust"
                )
            }
            if (
                type(discovered) is not str
                or re.fullmatch(r"[0-9a-f]{64}", discovered) is None
                or document.get("Name") != f"/{name}"
                or type(config) is not dict
                or config.get("Image") != TOOL.RUST_IMAGE
                or config.get("Cmd") != ["sleep", "2"]
                or config.get("Labels") != expected_labels
                or type(host) is not dict
                or host.get("AutoRemove") is not False
            ):
                return None
            return discovered

        cleanup_id = (
            container_id
            if type(container_id) is str
            and re.fullmatch(r"[0-9a-f]{64}", container_id) is not None
            else None
        )
        try:
            if cleanup_id is None:
                # A name is only a read-only discovery target.  Foreign,
                # malformed, or daemon-unknown results cause zero mutation.
                discovery = inspect_target(name)
                if (
                    discovery.returncode == 0
                    and discovery.stderr == b""
                ):
                    cleanup_id = exact_contract_id(discovery.stdout)
                elif discovery.returncode == 1:
                    try:
                        TOOL._docker_absence_evidence_v1(name, subprocess.run)
                    except TOOL.Q05BDualSupervisorError:
                        pass
            if cleanup_id is not None:
                owned = inspect_target(cleanup_id)
                if (
                    owned.returncode == 0
                    and owned.stderr == b""
                    and exact_contract_id(owned.stdout) == cleanup_id
                ):
                    remove_command = [
                        TOOL.DOCKER_EXECUTABLE,
                        f"--host={TOOL.DOCKER_HOST}",
                        "rm",
                        "-f",
                        cleanup_id,
                    ]
                    cleanup_mutations.append(remove_command)
                    subprocess.run(
                        remove_command,
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    TOOL._docker_absence_evidence_v1(
                        cleanup_id,
                        subprocess.run,
                    )
                elif owned.returncode == 1:
                    try:
                        TOOL._docker_absence_evidence_v1(
                            cleanup_id,
                            subprocess.run,
                        )
                    except TOOL.Q05BDualSupervisorError:
                        pass
        except OSError:
            # Transport/CLI uncertainty is a zero-mutation cleanup result.
            pass
        assert all(
            row[2:4] == ["rm", "-f"]
            and re.fullmatch(r"[0-9a-f]{64}", row[-1]) is not None
            and row[-1] != name
            for row in cleanup_mutations
        )


def test_dry_run_is_read_only_zero_null_and_canonical_stdout(tmp_path: Path) -> None:
    before = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
    }
    plan = TOOL.dry_run_plan_v1(ROOT)
    after = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert plan["status"] == TOOL.STATUS_DRY_RUN
    assert plan["qualification_predicate_count"] == 0
    assert plan["qualification_predicate_mask"] == 0
    assert plan["qualification_candidate_receipt"] is None
    assert plan["qualification_final_receipt"] is None
    assert plan["q1_state"] == "NOT_RUN"
    assert plan["q1_formal_output_roots"] == [None] * 8
    assert plan["receipt_created"] is False
    assert plan["artifact_path"] is None
    assert plan["artifact_written"] is False
    assert plan["actual_blockers"] == []
    assert plan["actual_implementation_blockers"] == []
    assert plan["pending_actual_evidence_predicate_ids"] == list(range(1, 21))
    assert plan["actual_entrypoint_implemented"] is True
    assert plan["actual_execution_status"] == "NOT_EXECUTED_AT_COMMIT_A"
    assert plan["actual_admitted"] is False

    completed = subprocess.run(
        [sys.executable, str(TOOL_PATH), "--dry-run"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.stderr == b""
    assert completed.stdout.count(b"\n") == 1
    decoded = _strict_json(completed.stdout)
    assert TOOL._canonical_json_bytes(decoded) == completed.stdout
    assert decoded["artifact_written"] is False
    assert not any(tmp_path.iterdir())


def test_run_actual_owns_private_work_root_and_returns_orchestration_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    cargo = tmp_path / "cargo-cache"
    cargo.mkdir()
    artifact = project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH
    source_commit = "ab" * 20
    observed: dict[str, object] = {}
    hostile_environment_temp = tmp_path / "host-environment-temp"
    hostile_environment_temp.mkdir(mode=0o777)
    hostile_environment_temp.chmod(0o777)
    monkeypatch.setenv("TMPDIR", str(hostile_environment_temp))

    class FakeBackend:
        def __init__(self, *args):
            observed["constructor_args"] = args
            work_root = args[4]
            status = work_root.lstat()
            assert stat.S_IMODE(status.st_mode) == 0o700
            assert status.st_nlink == 2
            assert status.st_uid == os.geteuid()
            assert tuple(work_root.iterdir()) == ()
            assert work_root.parent == TOOL.ACTUAL_TEMPORARY_PARENT
            observed["work_root"] = work_root
            self.endpoint_actors = None
            self.command_runner = lambda *_args, **_kwargs: None
            self.admission_work_root_descriptor = os.open("/dev/null", os.O_RDONLY)
            self.admission_issued_marker_descriptor = os.open("/dev/null", os.O_RDONLY)
            self.admission_spending_marker_descriptor = os.open("/dev/null", os.O_RDONLY)
            self.admission_consumed_marker_descriptor = os.open("/dev/null", os.O_RDONLY)
            observed["backend"] = self
            observed["descriptors"] = (
                self.admission_work_root_descriptor,
                self.admission_issued_marker_descriptor,
                self.admission_spending_marker_descriptor,
                self.admission_consumed_marker_descriptor,
            )

    monkeypatch.setattr(
        TOOL,
        "verify_actual_source_commit_v1",
        lambda root, requested: observed.setdefault("verified", (root, requested))[1],
    )
    monkeypatch.setattr(TOOL, "ConcreteQ05BActualBackendV1", FakeBackend)
    expected = {"qualification_count": 20, "q1_state": "NOT_RUN"}

    def fake_orchestrate(
        commit,
        target,
        backend,
        *,
        pre_handoff_cleanup,
    ):
        observed["orchestrate"] = (commit, target, backend)
        assert TOOL._detach_and_close_actual_admission_descriptors_v1(backend) == ()
        pre_handoff_cleanup()
        return dict(expected)

    monkeypatch.setattr(TOOL, "orchestrate_actual_with_backend_v1", fake_orchestrate)
    real_close = os.close
    close_counts: dict[int, int] = {}
    admission_closed: set[int] = set()

    def exact_close(descriptor):
        if (
            descriptor in observed.get("descriptors", ())
            and len(admission_closed) < len(observed["descriptors"])
        ):
            assert descriptor not in admission_closed
            admission_closed.add(descriptor)
            close_counts[descriptor] = close_counts.get(descriptor, 0) + 1
        real_close(descriptor)

    monkeypatch.setattr(TOOL.os, "close", exact_close)
    assert TOOL.run_actual_v1(project, source_commit, artifact, cargo) == expected
    assert observed["verified"] == (project, source_commit)
    assert observed["constructor_args"][:4] == (
        project,
        source_commit,
        artifact,
        cargo,
    )
    assert observed["orchestrate"][:2] == (source_commit, artifact)
    assert not Path(observed["work_root"]).exists()
    backend = observed["backend"]
    assert all(
        getattr(backend, field) is None
        for field in TOOL._ACTUAL_ADMISSION_DESCRIPTOR_FIELDS
    )
    for descriptor in observed["descriptors"]:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert set(close_counts) == set(observed["descriptors"])
    assert all(count == 1 for count in close_counts.values())


def test_run_actual_rejects_untrusted_temporary_parent_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    cargo = tmp_path / "cargo"
    cargo.mkdir()
    artifact = project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH
    monkeypatch.setattr(
        TOOL, "verify_actual_source_commit_v1", lambda _root, requested: requested
    )
    real_lstat = Path.lstat

    def untrusted_parent(self):
        value = real_lstat(self)
        if self == TOOL.ACTUAL_TEMPORARY_PARENT:
            return SimpleNamespace(
                st_mode=(stat.S_IFDIR | 0o777),
                st_nlink=value.st_nlink,
                st_uid=value.st_uid,
            )
        return value

    monkeypatch.setattr(Path, "lstat", untrusted_parent)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.run_actual_v1(project, "aa" * 20, artifact, cargo)
    assert failure.value.code == TOOL.FAIL_POLICY


def _dummy_held_actor_for_outer_cleanup_v1(
    tmp_path: Path,
    role_id: int,
) -> TOOL.HeldActorProcessV1:
    actor_id = TOOL.ROLE_ROWS[role_id - 1][1]
    return TOOL.HeldActorProcessV1(
        role_id=role_id,
        actor_id=actor_id,
        container_name=f"hegel-q05b-fixture-{role_id}",
        command=("fixture",),
        cidfile=tmp_path / f"actor-{role_id}.cid",
        control_root=tmp_path / f"control-{role_id}",
        mount_registry=SimpleNamespace(),
        process=SimpleNamespace(),
        stdout_drain=TOOL.BoundedPipeDrainV1(
            1, bytearray(), 0, False, sha256(), []
        ),
        stderr_drain=TOOL.BoundedPipeDrainV1(
            1, bytearray(), 0, False, sha256(), []
        ),
        stdout_thread=threading.Thread(),
        stderr_thread=threading.Thread(),
        sample_thread=None,
        sample_rows=[],
        sample_errors=[],
        sample_complete=threading.Event(),
        child_done_observed=threading.Event(),
        sample_stop=threading.Event(),
        sample_lock=threading.Lock(),
        container_id=None,
        cid_parent_identity=(1, role_id, 0o700, 2),
        cidfile_evidence=None,
        cleanup_errors=[],
    )


def test_run_actual_failure_cleans_actors_and_detaches_all_held_fds_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    cargo = tmp_path / "cargo-cache"
    cargo.mkdir()
    artifact = project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH
    source_commit = "cd" * 20
    observed: dict[str, object] = {"cleaned_actors": []}

    class FakeBackend:
        def __init__(self, *_args):
            first = _dummy_held_actor_for_outer_cleanup_v1(tmp_path, 1)
            second = _dummy_held_actor_for_outer_cleanup_v1(tmp_path, 2)
            self.active_mount_binding_slots = [None, None, None]
            self.active_actor_slots = [first, second, None]
            self.endpoint_actors = (first, second)
            self.host_actor = None
            self.command_runner = lambda *_a, **_k: None
            for field in TOOL._ACTUAL_ADMISSION_DESCRIPTOR_FIELDS:
                setattr(self, field, os.open("/dev/null", os.O_RDONLY))
            observed["backend"] = self
            observed["descriptors"] = tuple(
                getattr(self, field)
                for field in TOOL._ACTUAL_ADMISSION_DESCRIPTOR_FIELDS
            )
            observed["actors"] = self.endpoint_actors

    monkeypatch.setattr(
        TOOL, "verify_actual_source_commit_v1", lambda _root, requested: requested
    )
    monkeypatch.setattr(TOOL, "ConcreteQ05BActualBackendV1", FakeBackend)

    def fake_actor_cleanup(actor, runner):
        assert actor in observed["backend"].endpoint_actors
        assert callable(runner)
        observed["cleaned_actors"].append(actor)
        return ()

    monkeypatch.setattr(
        TOOL,
        "_abort_held_actor_cleanup_v1",
        fake_actor_cleanup,
    )
    monkeypatch.setattr(
        TOOL,
        "orchestrate_actual_with_backend_v1",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected-stage-failure")
        ),
    )
    real_close = os.close
    close_counts: dict[int, int] = {}
    admission_closed: set[int] = set()

    def exact_close(descriptor):
        if (
            descriptor in observed["descriptors"]
            and len(admission_closed) < len(observed["descriptors"])
        ):
            backend = observed["backend"]
            assert all(
                getattr(backend, field) is None
                for field in TOOL._ACTUAL_ADMISSION_DESCRIPTOR_FIELDS
            )
            assert descriptor not in admission_closed
            admission_closed.add(descriptor)
            close_counts[descriptor] = close_counts.get(descriptor, 0) + 1
        real_close(descriptor)

    monkeypatch.setattr(TOOL.os, "close", exact_close)
    with pytest.raises(RuntimeError, match="injected-stage-failure"):
        TOOL.run_actual_v1(project, source_commit, artifact, cargo)
    backend = observed["backend"]
    assert observed["cleaned_actors"] == list(observed["actors"])
    assert len(observed["cleaned_actors"]) == 2
    assert backend.active_actor_slots == [None, None, None]
    assert backend.endpoint_actors is None
    assert backend.host_actor is None
    assert all(count == 1 for count in close_counts.values())
    assert set(close_counts) == set(observed["descriptors"])


def test_run_actual_detach_failure_defers_then_cleans_private_root_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    cargo = tmp_path / "cargo-cache"
    cargo.mkdir()
    artifact = project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH
    source_commit = "cf" * 20
    observed: dict[str, object] = {
        "cleanup_calls": 0,
        "destination_was_present_before_cleanup": False,
        "write_calls": 0,
    }

    class FakeBackend:
        def __init__(self, *_args):
            self.work_root = _args[4]
            self.active_mount_binding_slots = [None, None, None]
            self.active_actor_slots = [None, None, None]
            self.endpoint_actors = None
            self.host_actor = None
            self.command_runner = lambda *_a, **_k: None
            for field in TOOL._ACTUAL_ADMISSION_DESCRIPTOR_FIELDS:
                setattr(self, field, None)
            observed["backend"] = self
            observed["work_root"] = self.work_root

    monkeypatch.setattr(
        TOOL, "verify_actual_source_commit_v1", lambda _root, requested: requested
    )
    monkeypatch.setattr(TOOL, "ConcreteQ05BActualBackendV1", FakeBackend)
    real_temporary_directory = tempfile.TemporaryDirectory
    managers: list[object] = []

    class CountingTemporaryDirectory:
        def __init__(self, *args, **kwargs):
            self.inner = real_temporary_directory(*args, **kwargs)
            self.name = self.inner.name
            managers.append(self)

        def cleanup(self):
            observed["cleanup_calls"] += 1
            destination = observed.get("destination")
            observed["destination_was_present_before_cleanup"] = (
                isinstance(destination, Path) and destination.exists()
            )
            self.inner.cleanup()

    monkeypatch.setattr(
        TOOL.tempfile,
        "TemporaryDirectory",
        CountingTemporaryDirectory,
    )
    real_write = TOOL.os.write

    def partial_then_fail(descriptor: int, payload: bytes) -> int:
        try:
            descriptor_path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            descriptor_path = Path("/unavailable")
        if descriptor_path == observed.get("destination"):
            observed["write_calls"] += 1
            if observed["write_calls"] == 1:
                return real_write(descriptor, payload[: len(payload) // 2])
            raise OSError(errno.EIO, "injected detach partial-write failure")
        return real_write(descriptor, payload)

    monkeypatch.setattr(TOOL.os, "write", partial_then_fail)

    def fail_during_detach(
        _commit,
        _target,
        backend,
        *,
        pre_handoff_cleanup,
    ):
        assert callable(pre_handoff_cleanup)
        target_output = backend.work_root / "target-output"
        release = target_output / "release"
        destination_parent = target_output / "runtime-binary"
        release.mkdir(parents=True)
        destination_parent.mkdir(mode=0o700)
        destination_parent.chmod(0o700)
        source = release / "hegel-q1-archive-projection-oracle"
        source.write_bytes(b"ELF-Q05B-OUTER-CLEANUP-TEST\n" * 32)
        source.chmod(0o755)
        destination = destination_parent / source.name
        observed["destination"] = destination
        TOOL.detach_cargo_release_binary_v1(source, destination)
        raise AssertionError("detach failure did not stop orchestration")

    monkeypatch.setattr(
        TOOL,
        "orchestrate_actual_with_backend_v1",
        fail_during_detach,
    )
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.run_actual_v1(project, source_commit, artifact, cargo)
    assert failure.value.code == TOOL.FAIL_SOURCE
    assert "deferred outer-owned-root cleanup required" in failure.value.detail
    assert observed["write_calls"] == 2
    assert observed["destination_was_present_before_cleanup"] is True
    assert observed["cleanup_calls"] == 1
    assert len(managers) == 1
    assert not Path(observed["work_root"]).exists()
    assert not os.path.lexists(artifact)


@pytest.mark.parametrize(
    ("failure_kind", "write_residual", "expected_artifact_written"),
    (
        ("q_true", True, True),
        ("runtime", True, True),
        ("q_false", False, False),
    ),
)
def test_run_actual_cleanup_composite_preserves_truthful_artifact_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    write_residual: bool,
    expected_artifact_written: bool,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    cargo = tmp_path / "cargo-cache"
    cargo.mkdir()
    artifact = project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH
    source_commit = "ce" * 20

    class FakeBackend:
        def __init__(self, *_args):
            first = _dummy_held_actor_for_outer_cleanup_v1(tmp_path, 1)
            second = _dummy_held_actor_for_outer_cleanup_v1(tmp_path, 2)
            self.active_mount_binding_slots = [None, None, None]
            self.active_actor_slots = [first, second, None]
            self.endpoint_actors = (first, second)
            self.host_actor = None
            self.command_runner = lambda *_a, **_k: None
            for field in TOOL._ACTUAL_ADMISSION_DESCRIPTOR_FIELDS:
                setattr(self, field, None)

    monkeypatch.setattr(
        TOOL, "verify_actual_source_commit_v1", lambda _root, requested: requested
    )
    monkeypatch.setattr(TOOL, "ConcreteQ05BActualBackendV1", FakeBackend)
    monkeypatch.setattr(
        TOOL,
        "_abort_held_actor_cleanup_v1",
        lambda _actor, _runner: ("injected-actor-cleanup-error",),
    )

    def fail_orchestration(
        _commit,
        target,
        _backend,
        *,
        pre_handoff_cleanup,
    ):
        assert callable(pre_handoff_cleanup)
        if write_residual:
            target.write_bytes(b"injected-unowned-residual")
        if failure_kind == "q_true":
            raise TOOL.Q05BDualSupervisorError(
                TOOL.FAIL_ARTIFACT,
                "injected owned residual",
                artifact_written=True,
            )
        if failure_kind == "q_false":
            raise TOOL.Q05BDualSupervisorError(
                TOOL.FAIL_POLICY,
                "injected no-artifact failure",
            )
        raise RuntimeError("injected non-supervisor residual")

    monkeypatch.setattr(
        TOOL,
        "orchestrate_actual_with_backend_v1",
        fail_orchestration,
    )
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.run_actual_v1(project, source_commit, artifact, cargo)
    assert failure.value.code == TOOL.FAIL_POLICY
    assert failure.value.artifact_written is expected_artifact_written
    assert (
        TOOL._error_object(failure.value)["artifact_written"]
        is expected_artifact_written
    )
    assert "injected-actor-cleanup-error" in failure.value.detail
    assert artifact.exists() is write_residual


def test_run_actual_rejects_bool_existing_target_and_cargo_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    cargo = tmp_path / "cargo-cache"
    cargo.mkdir()
    cargo_link = tmp_path / "cargo-cache-link"
    cargo_link.symlink_to(cargo, target_is_directory=True)
    artifact = project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH
    called = False

    def unexpected_verify(*_args):
        nonlocal called
        called = True

    monkeypatch.setattr(TOOL, "verify_actual_source_commit_v1", unexpected_verify)
    with pytest.raises(TOOL.Q05BDualSupervisorError):
        TOOL.run_actual_v1(project, True, artifact, cargo)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as linked:
        TOOL.run_actual_v1(project, "ef" * 20, artifact, cargo_link)
    assert linked.value.code == TOOL.FAIL_SOURCE
    artifact.write_bytes(b"preexisting")
    with pytest.raises(TOOL.Q05BDualSupervisorError) as existing:
        TOOL.run_actual_v1(project, "ef" * 20, artifact, cargo)
    assert existing.value.code == TOOL.FAIL_ARTIFACT
    assert artifact.read_bytes() == b"preexisting"
    assert called is False


def test_cli_run_passes_unresolved_cargo_path_assigns_value_and_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    cargo = tmp_path / "cargo"
    cargo.mkdir()
    cargo_link = tmp_path / "cargo-link"
    cargo_link.symlink_to(cargo, target_is_directory=True)
    artifact = project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH
    observed: dict[str, object] = {}

    def fake_run(root, commit, target, cargo_source):
        observed["args"] = (root, commit, target, cargo_source)
        return {"status": "MOCK_ACTUAL_RETURN", "q1_state": "NOT_RUN"}

    monkeypatch.setattr(TOOL, "run_actual_v1", fake_run)
    exit_code = TOOL.main(
        [
            "--run",
            "--project-root",
            str(project),
            "--source-commit",
            "12" * 20,
            "--artifact",
            str(artifact),
            "--cargo-cache-source",
            str(cargo_link),
        ]
    )
    assert exit_code == 0
    assert observed["args"] == (
        project.resolve(),
        "12" * 20,
        artifact,
        cargo_link,
    )
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    stdout = captured.out
    assert stdout.count(b"\n") == 1
    assert TOOL._canonical_json_bytes(_strict_json(stdout)) == stdout

    assert TOOL.main(
        [
            "--run",
            "--project-root",
            str(project),
            "--source-commit",
            "12" * 20,
            "--artifact",
            str(artifact),
        ]
    ) == 1
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    failure = _strict_json(captured.out)
    assert failure["status"] == TOOL.FAIL_SOURCE
    assert TOOL.main(
        [
            "--dry-run",
            "--project-root",
            str(project),
            "--cargo-cache-source",
            str(cargo),
        ]
    ) == 1
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    failure = _strict_json(captured.out)
    assert failure["status"] == TOOL.FAIL_POLICY
    assert TOOL.main(["--run", "--dry-run"]) == 1
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    assert _strict_json(captured.out)["status"] == TOOL.FAIL_POLICY
    assert TOOL.main(["--run", "--caller-work-root", "/tmp/x"]) == 1
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    assert _strict_json(captured.out)["status"] == TOOL.FAIL_POLICY


def test_cli_run_rejects_raw_project_symlink_without_resolving(
    tmp_path: Path,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    project_link = tmp_path / "project-link"
    project_link.symlink_to(project, target_is_directory=True)
    cargo = tmp_path / "cargo"
    cargo.mkdir()
    artifact = project_link / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH
    assert TOOL.main(
        [
            "--run",
            "--project-root",
            str(project_link),
            "--source-commit",
            "34" * 20,
            "--artifact",
            str(artifact),
            "--cargo-cache-source",
            str(cargo),
        ]
    ) == 1
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    failure = _strict_json(captured.out)
    assert failure["status"] == TOOL.FAIL_SOURCE
    assert failure["artifact_written"] is False


def _synthetic_commit_a_actual_config_bytes() -> bytes:
    value = TOOL._strict_json_value_v1(
        (ROOT / TOOL.CONFIG_RELATIVE_PATH).read_bytes(),
        "current dual isolation config",
    )
    assert type(value) is dict
    assert value["engineering_status"] == TOOL.COMMIT_A_ACTUAL_ENGINEERING_STATUS
    assert value["actual_preconditions"] == TOOL.COMMIT_A_ACTUAL_PRECONDITIONS_V1
    return TOOL._canonical_json_bytes(value)


def _synthetic_git_source_transcript(
    project_root: Path,
    source_commit: str,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": TOOL.ACTUAL_GIT_SOURCE_TRANSCRIPT_SCHEMA_VERSION,
        "project_root": project_root.resolve().as_posix(),
        "requested_source_commit": source_commit,
        "command_rows": [],
    }
    commands = (
        (
            1,
            "VERIFY_HEAD",
            [
                "git",
                "-C",
                body["project_root"],
                "rev-parse",
                "--verify",
                "HEAD",
            ],
            (source_commit + "\n").encode("ascii"),
        ),
        (
            2,
            "VERIFY_CLEAN_STATUS_Z",
            [
                "git",
                "-C",
                body["project_root"],
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "-z",
            ],
            b"",
        ),
    )
    body["command_rows"] = [
        {
            "ordinal": ordinal,
            "purpose": purpose,
            "argv": argv,
            "returncode": 0,
            "stdout_hex": stdout.hex(),
            "stderr_hex": "",
            "stdout_sha256": sha256(stdout).hexdigest(),
            "stderr_sha256": sha256(b"").hexdigest(),
        }
        for ordinal, purpose, argv, stdout in commands
    ]
    transcript = dict(body)
    transcript["transcript_root"] = sha256(
        TOOL.ACTUAL_GIT_SOURCE_TRANSCRIPT_ROOT_DOMAIN
        + TOOL._canonical_json_bytes(body)
    ).hexdigest()
    ADMISSION.validate_git_source_transcript_v1(transcript, source_commit)
    return transcript


def _synthetic_pinned_image_evidence_v1(reference: str, seed: int):
    labels = (
        {
            "org.opencontainers.image.source": (
                "https://github.com/rust-lang/docker-rust"
            )
        }
        if reference == TOOL.RUST_IMAGE
        else None
    )
    inspect = [
        {
            "Id": "sha256:" + f"{seed:02x}" * 32,
            "RepoDigests": [reference],
            "Os": "linux",
            "Architecture": "amd64",
            "Config": {"Env": ["PATH=/usr/bin"], "Labels": labels},
        }
    ]
    raw = TOOL._canonical_json_bytes(inspect)
    body = {
        "schema_version": "hegel-phase3a-q05b-pinned-local-image-evidence/1",
        "requested_reference": reference,
        "image_id": inspect[0]["Id"],
        "repo_digests": [reference],
        "os": "linux",
        "architecture": "amd64",
        "raw_inspect_hex": raw.hex(),
        "raw_inspect_sha256": sha256(raw).hexdigest(),
    }
    return {**body, "evidence_sha256": sha256(TOOL._canonical_json_bytes(body)).hexdigest()}


def _synthetic_sealed_tree_v1(
    root: Path,
    path_payload_rows: list[tuple[str, bytes]],
    seed: int,
):
    file_rows = [
        [
            relative,
            seed,
            seed * 100 + index,
            1,
            1000,
            1000,
            0o444,
            len(payload),
            seed * 1000 + index,
            seed * 1000 + index,
            sha256(payload).hexdigest(),
        ]
        for index, (relative, payload) in enumerate(path_payload_rows, start=1)
    ]
    directories = sorted(
        {
            "/".join(relative.split("/")[:depth])
            for relative, _ in path_payload_rows
            for depth in range(1, len(relative.split("/")))
        }
    )
    directory_rows = [
        [relative, seed, seed * 1000 + index, 2, 1000, 1000, 0o555, seed, seed]
        for index, relative in enumerate(directories, start=1)
    ]
    body = {
        "schema_version": "hegel-phase3a-q05b-sealed-tree-identity/1",
        "root_path": root.resolve().as_posix(),
        "root_device": seed,
        "root_inode": seed * 10,
        "root_nlink": 2,
        "root_mode": 0o555,
        "directory_rows": directory_rows,
        "file_rows": file_rows,
    }
    return {**body, "manifest_sha256": sha256(TOOL._canonical_json_bytes(body)).hexdigest()}


def _synthetic_sealed_snapshot_v1(tree: dict[str, object]):
    body = {
        "schema_version": "hegel-phase3a-q05b-sealed-snapshot-identity/1",
        "root_device": tree["root_device"],
        "root_inode": tree["root_inode"],
        "root_mode": tree["root_mode"],
        "file_rows": tree["file_rows"],
    }
    return {**body, "manifest_sha256": sha256(TOOL._canonical_json_bytes(body)).hexdigest()}


def _synthetic_seccomp_evidence_v1(
    absolute: Path,
    relative: str,
    payload: bytes,
    seed: int,
):
    body = {
        "schema_version": "hegel-phase3a-q05b-sealed-policy-file/1",
        "absolute_path": absolute.resolve().as_posix(),
        "snapshot_relative_path": relative,
        "file_device": seed,
        "file_inode": seed * 10,
        "file_nlink": 1,
        "file_uid": 1000,
        "file_gid": 1000,
        "file_mode": 0o444,
        "file_size": len(payload),
        "file_mtime_ns": seed,
        "file_ctime_ns": seed,
        "payload_sha256": sha256(payload).hexdigest(),
    }
    return {**body, "manifest_sha256": sha256(TOOL._canonical_json_bytes(body)).hexdigest()}


def _synthetic_binary_evidence_v1(path: Path, payload: bytes):
    body = {
        "schema_version": "hegel-phase3a-q05b-sealed-prebuilt-rust-binary/1",
        "binary_path": path.resolve().as_posix(),
        "device": 91,
        "inode": 92,
        "nlink": 1,
        "uid": 1000,
        "gid": 1000,
        "mode": 0o555,
        "size": len(payload),
        "mtime_ns": 93,
        "ctime_ns": 94,
        "sha256": sha256(payload).hexdigest(),
        "payload_hex": payload.hex(),
    }
    return {**body, "manifest_sha256": sha256(TOOL._canonical_json_bytes(body)).hexdigest()}


def _synthetic_actual_admission_fixture(
    tmp_path: Path,
    source_commit: str = "42" * 20,
):
    commit_a_config = _synthetic_commit_a_actual_config_bytes()
    artifact_parent = tmp_path / "artifacts"
    artifact_parent.mkdir()
    artifact = (artifact_parent / "qualification.json").resolve()
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    work_path = lambda relative: (work / relative).as_posix()
    layout = {
        "python_snapshot": work_path("snapshots/python"),
        "rust_snapshot": work_path("snapshots/rust"),
        "host_snapshot": work_path("snapshots/host"),
        "cargo_home": work_path("cargo-home"),
        "target_output": work_path("target-output"),
        "cargo_release_binary": work_path(
            "target-output/release/hegel-q1-archive-projection-oracle"
        ),
        "runtime_binary_parent": work_path("target-output/runtime-binary"),
        "python_output": work_path("python-output"),
        "python_control": work_path("python-control"),
        "python_cid_parent": work_path("python-cid"),
        "python_cidfile": work_path("python-cid/python.cid"),
        "rust_output": work_path("rust-output"),
        "rust_control": work_path("rust-control"),
        "rust_cid_parent": work_path("rust-cid"),
        "rust_cidfile": work_path("rust-cid/rust.cid"),
        "host_output": work_path("host-output-unused"),
        "host_control": work_path("host-control"),
        "host_cid_parent": work_path("host-cid"),
        "host_cidfile": work_path("host-cid/host.cid"),
        "host_staging": work_path("host-staging"),
        "build_cid_parent": work_path("build-cid"),
        "build_test_cidfile": work_path("build-cid/test.cid"),
        "build_release_cidfile": work_path("build-cid/release.cid"),
        "stdout_root": work_path("stdout"),
        "binary": work_path(
            "target-output/runtime-binary/hegel-q1-archive-projection-oracle"
        ),
    }
    work_identity = TOOL.actual_work_root_identity_v1(work, layout)
    cargo_cache_source = tmp_path / "external-cargo-cache"
    cargo_cache_source.mkdir(mode=0o700)
    cargo_cache_status = cargo_cache_source.lstat()
    config = TOOL._strict_json_value_v1(commit_a_config, "Commit-A config")
    runtime_seccomp_payload = (
        ROOT / TOOL.RUNTIME_SECCOMP_RELATIVE_PATH
    ).read_bytes()
    build_seccomp_payload = (
        ROOT / TOOL.BUILD_SECCOMP_RELATIVE_PATH
    ).read_bytes()
    cargo_lock_payload = (
        "version = 3\n\n"
        "[[package]]\n"
        'name = "pkg"\n'
        'version = "1.0.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{"64" * 32}"\n'
    ).encode("ascii")
    actor_material = {
        "PYTHON_ENDPOINT": [("src/python.py", b"python\n")],
        "RUST_ENDPOINT": [
            ("rust/q1_archive_projection_oracle/Cargo.lock", cargo_lock_payload),
            ("src/rust.rs", b"rust\n"),
        ],
        "TRUSTED_HOST_REPLAY": [
            (TOOL.RUNTIME_SECCOMP_RELATIVE_PATH, runtime_seccomp_payload),
            (TOOL.BUILD_SECCOMP_RELATIVE_PATH, build_seccomp_payload),
            (TOOL.CONFIG_RELATIVE_PATH, commit_a_config),
        ],
    }

    def source_evidence(
        actor_id: str,
        material: list[tuple[str, bytes]],
        _seed: int,
    ):
        rows = []
        preimages = []
        framed = sha256()
        for index, (relative, payload) in enumerate(material, start=1):
            row = [
                relative,
                0o100644,
                TOOL.sha1(
                    b"blob "
                    + str(len(payload)).encode("ascii")
                    + b"\x00"
                    + payload
                ).hexdigest(),
                len(payload),
                sha256(payload).hexdigest(),
            ]
            rows.append(row)
            preimages.append(row + [payload.hex()])
            encoded = relative.encode()
            framed.update(len(encoded).to_bytes(4, "big"))
            framed.update(encoded)
            framed.update(len(payload).to_bytes(8, "big"))
            framed.update(payload)
        return {
            "schema_version": "hegel-phase3a-q05b-actor-source-evidence/1",
            "actor_id": actor_id,
            "commit": source_commit,
            "project_git_prefix": "Hegel Machine/",
            "path_registry_sha256": sha256(
                TOOL._canonical_json_bytes([row[0] for row in rows])
            ).hexdigest(),
            "source_identity_sha256": framed.hexdigest(),
            "rows": rows,
            "blob_preimage_rows": preimages,
        }

    source_rows = {
        actor_id: source_evidence(actor_id, actor_material[actor_id], 40 + index)
        for index, actor_id in enumerate(actor_material, start=1)
    }
    source_object_rows = {
        row[0]: (row[1], row[2])
        for evidence in source_rows.values()
        for row in evidence["rows"]
    }
    tree_payloads: dict[str, bytes] = {}

    def build_tree(rows: dict[str, tuple[int, str]]) -> str:
        entries: dict[str, dict[str, tuple[int, str]] | tuple[int, str]] = {}
        for relative, identity in rows.items():
            head, separator, tail = relative.partition("/")
            if separator:
                child = entries.setdefault(head, {})
                assert type(child) is dict
                child[tail] = identity
            else:
                entries[head] = identity
        payload = bytearray()
        for name in sorted(entries):
            value = entries[name]
            if type(value) is dict:
                mode = "40000"
                object_id = build_tree(value)
            else:
                mode = f"{value[0]:o}"
                object_id = value[1]
            payload.extend(mode.encode("ascii"))
            payload.extend(b" ")
            payload.extend(name.encode("utf-8"))
            payload.extend(b"\x00")
            payload.extend(bytes.fromhex(object_id))
        raw = bytes(payload)
        object_id = TOOL.sha1(
            b"tree " + str(len(raw)).encode("ascii") + b"\x00" + raw
        ).hexdigest()
        tree_payloads[object_id] = raw
        return object_id

    project_tree_id = build_tree(source_object_rows)
    root_tree_payload = (
        b"40000 Hegel Machine\x00" + bytes.fromhex(project_tree_id)
    )
    root_tree_id = TOOL.sha1(
        b"tree "
        + str(len(root_tree_payload)).encode("ascii")
        + b"\x00"
        + root_tree_payload
    ).hexdigest()
    tree_payloads[root_tree_id] = root_tree_payload
    commit_payload = (
        f"tree {root_tree_id}\n\nsynthetic actual-admission fixture\n".encode(
            "ascii"
        )
    )
    closure_body = {
        "schema_version": "hegel-phase3a-q05b-git-source-object-closure/1",
        "commit": source_commit,
        "commit_payload_hex": commit_payload.hex(),
        "commit_payload_sha256": sha256(commit_payload).hexdigest(),
        "root_tree_object_id": root_tree_id,
        "project_tree_prefix": "Hegel Machine",
        "project_tree_object_id": project_tree_id,
        "allowlist_union": sorted(source_object_rows),
        "tree_object_rows": [
            [object_id, tree_payloads[object_id].hex()]
            for object_id in sorted(tree_payloads)
        ],
    }
    source_object_closure = {
        **closure_body,
        "closure_sha256": sha256(
            TOOL._canonical_json_bytes(closure_body)
        ).hexdigest(),
    }
    snapshot_layout_keys = {
        "PYTHON_ENDPOINT": "python_snapshot",
        "RUST_ENDPOINT": "rust_snapshot",
        "TRUSTED_HOST_REPLAY": "host_snapshot",
    }
    snapshot_rows = {
        actor_id: _synthetic_sealed_tree_v1(
            Path(layout[snapshot_layout_keys[actor_id]]),
            actor_material[actor_id],
            50 + index,
        )
        for index, actor_id in enumerate(actor_material, start=1)
    }
    image_rows = {
        "python": _synthetic_pinned_image_evidence_v1(TOOL.PYTHON_IMAGE, 61),
        "rust": _synthetic_pinned_image_evidence_v1(TOOL.RUST_IMAGE, 62),
    }
    cargo_tree = _synthetic_sealed_tree_v1(
        Path(layout["cargo_home"]), [("registry/pkg", b"cargo\n")], 63
    )
    cargo_snapshot = _synthetic_sealed_snapshot_v1(cargo_tree)
    cargo_material = {
        "schema_version": "hegel-phase3a-q05b-sealed-cargo-home/1",
        "locked_registry_package_count": 1,
        "locked_packages": [["pkg", "1.0.0", "64" * 32]],
        "file_count": 1,
        "file_rows": [["registry/pkg", 0o100644, 6, sha256(b"cargo\n").hexdigest()]],
        "file_preimage_rows": [["registry/pkg", 0o100644, b"cargo\n".hex()]],
        "manifest_sha256": sha256(
            TOOL._canonical_json_bytes(
                [["registry/pkg", 0o100644, 6, sha256(b"cargo\n").hexdigest()]]
            )
        ).hexdigest(),
        "sealed_snapshot_identity": cargo_snapshot,
        "root_mode": "0555",
        "file_modes": "0444_OR_0555",
        "cargo_home_mount": "READ_ONLY_PREUNPACKED",
        "root_path": cargo_tree["root_path"],
        "root_nlink": cargo_tree["root_nlink"],
        "sealed_tree_identity": cargo_tree,
    }
    runtime_seccomp = _synthetic_seccomp_evidence_v1(
        Path(layout["host_snapshot"]) / TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
        TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
        runtime_seccomp_payload,
        65,
    )
    build_seccomp = _synthetic_seccomp_evidence_v1(
        Path(layout["host_snapshot"]) / TOOL.BUILD_SECCOMP_RELATIVE_PATH,
        TOOL.BUILD_SECCOMP_RELATIVE_PATH,
        build_seccomp_payload,
        66,
    )
    binary = _synthetic_binary_evidence_v1(
        Path(layout["binary"]), b"ELF-synthetic\n"
    )
    runtime_seccomp_path = Path(runtime_seccomp["absolute_path"])
    build_seccomp_path = Path(build_seccomp["absolute_path"])
    docker_authority = _docker_execution_authority(
        source_commit=source_commit,
        nonce=b"N" * 32,
    )
    test_commands = TOOL.rust_build_commands_v1(
        Path(layout["rust_snapshot"]),
        Path(layout["cargo_home"]),
        Path(layout["target_output"]),
        source_rows["RUST_ENDPOINT"]["source_identity_sha256"],
        Path(layout["build_test_cidfile"]),
        build_seccomp=build_seccomp_path,
        docker_slot_row=_docker_slot_row(docker_authority, "RUST_TEST"),
    )
    release_commands = TOOL.rust_build_commands_v1(
        Path(layout["rust_snapshot"]),
        Path(layout["cargo_home"]),
        Path(layout["target_output"]),
        source_rows["RUST_ENDPOINT"]["source_identity_sha256"],
        Path(layout["build_release_cidfile"]),
        build_seccomp=build_seccomp_path,
        docker_slot_row=_docker_slot_row(docker_authority, "RUST_RELEASE"),
    )
    planned_commands = {
        "python": TOOL.python_endpoint_command_v1(
            Path(layout["python_snapshot"]),
            Path(layout["python_output"]),
            Path(layout["python_control"]),
            runtime_seccomp_path,
            docker_slot_row=_docker_slot_row(
                docker_authority,
                "PYTHON_ENDPOINT",
            ),
            cidfile=Path(layout["python_cidfile"]),
        ),
        "rust": TOOL.rust_runtime_command_v1(
            Path(layout["binary"]),
            Path(layout["rust_output"]),
            Path(layout["rust_control"]),
            runtime_seccomp_path,
            docker_slot_row=_docker_slot_row(
                docker_authority,
                "RUST_ENDPOINT",
            ),
            cidfile=Path(layout["rust_cidfile"]),
        ),
        "host_template": TOOL.trusted_host_command_v1(
            Path(layout["host_snapshot"]),
            Path(layout["python_output"]),
            Path(layout["rust_output"]),
            Path(layout["stdout_root"]) / "python.stdout",
            Path(layout["stdout_root"]) / "rust.stdout",
            Path(layout["stdout_root"]) / "manifest.json",
            Path(layout["host_control"]),
            Path(layout["host_staging"]),
            runtime_seccomp_path,
            docker_slot_row=_docker_slot_row(
                docker_authority,
                "TRUSTED_HOST_REPLAY",
            ),
            cidfile=Path(layout["host_cidfile"]),
        ),
        "rust_test": test_commands[0],
        "rust_release": release_commands[1],
    }
    source_parent_identity = {
        "device": 91,
        "inode": 90,
        "nlink": 2,
        "uid": 1000,
        "gid": 1000,
        "mode": 0o755,
    }
    detached_parent_identity = {
        "device": binary["device"],
        "inode": 93,
        "nlink": 2,
        "uid": binary["uid"],
        "gid": binary["gid"],
        "mode": 0o700,
    }
    source_file_identity = {
        "device": 91,
        "inode": 94,
        "nlink": 2,
        "uid": 1000,
        "gid": 1000,
        "mode": 0o755,
        "size": binary["size"],
        "mtime_ns": 92,
        "ctime_ns": 92,
    }
    detached_file_identity = {
        "device": binary["device"],
        "inode": binary["inode"],
        "nlink": binary["nlink"],
        "uid": binary["uid"],
        "gid": binary["gid"],
        "mode": 0o755,
        "size": binary["size"],
        "mtime_ns": binary["mtime_ns"],
        "ctime_ns": binary["ctime_ns"] - 1,
    }
    detach_body = {
        "schema_version": (
            "hegel-phase3a-q05b-detached-cargo-release-binary/1"
        ),
        "source_path": layout["cargo_release_binary"],
        "detached_path": layout["binary"],
        "source_parent_before": source_parent_identity,
        "source_parent_after": source_parent_identity,
        "source_fd_before": source_file_identity,
        "source_fd_after": source_file_identity,
        "source_path_before": source_file_identity,
        "source_path_after": source_file_identity,
        "source_sha256_before": binary["sha256"],
        "source_sha256_after": binary["sha256"],
        "detached_parent_before": detached_parent_identity,
        "detached_parent_after": detached_parent_identity,
        "detached_fd": detached_file_identity,
        "detached_path_identity": detached_file_identity,
        "detached_sha256": binary["sha256"],
        "source_and_detached_bytes_equal": True,
    }
    binary_detach = {
        **detach_body,
        "manifest_sha256": sha256(
            TOOL._canonical_json_bytes(detach_body)
        ).hexdigest(),
    }
    stage1_evidence = {
        "config_hex": commit_a_config.hex(),
        "config_sha256": sha256(commit_a_config).hexdigest(),
        "fixed_artifact_path": artifact.as_posix(),
        "layout": layout,
        "cargo_cache_source": cargo_cache_source.resolve(strict=True).as_posix(),
        "cargo_cache_root_identity": [
            cargo_cache_status.st_dev,
            cargo_cache_status.st_ino,
            cargo_cache_status.st_nlink,
            stat.S_IMODE(cargo_cache_status.st_mode),
        ],
        "source_evidence": source_rows,
        "source_object_closure": source_object_closure,
        "image_evidence": image_rows,
        "planned_commands": planned_commands,
        "q1_authority": config["dry_run_authority"],
        "docker_execution_authority": docker_authority,
    }
    stage2_evidence = {
        "snapshot_evidence": snapshot_rows,
        "cargo_lock_hex": cargo_lock_payload.hex(),
        "cargo_lock_sha256": sha256(cargo_lock_payload).hexdigest(),
        "cargo_evidence": cargo_material,
        "seccomp_evidence": {"runtime": runtime_seccomp, "build": build_seccomp},
    }
    stage3_evidence = {
        "rust_test": {
            "command": planned_commands["rust_test"],
            "command_sha256": sha256(
                TOOL._canonical_json_bytes(planned_commands["rust_test"])
            ).hexdigest(),
            "seccomp_evidence": build_seccomp,
        },
        "rust_release_build": {
            "command": planned_commands["rust_release"],
            "command_sha256": sha256(
                TOOL._canonical_json_bytes(planned_commands["rust_release"])
            ).hexdigest(),
            "seccomp_evidence": build_seccomp,
        },
        "binary_detach": binary_detach,
        "binary": binary,
        "rust_snapshot_post_build": snapshot_rows["RUST_ENDPOINT"],
        "cargo_snapshot_post_build": cargo_snapshot,
        "cargo_tree_post_build": cargo_tree,
    }
    assert set(stage1_evidence) == {
        "config_hex",
        "config_sha256",
        "fixed_artifact_path",
        "layout",
        "cargo_cache_source",
        "cargo_cache_root_identity",
        "source_evidence",
        "source_object_closure",
        "image_evidence",
        "planned_commands",
        "q1_authority",
        "docker_execution_authority",
    }
    assert set(stage2_evidence) == {
        "snapshot_evidence",
        "cargo_lock_hex",
        "cargo_lock_sha256",
        "cargo_evidence",
        "seccomp_evidence",
    }
    assert set(stage3_evidence) == {
        "rust_test",
        "rust_release_build",
        "binary_detach",
        "binary",
        "rust_snapshot_post_build",
        "cargo_snapshot_post_build",
        "cargo_tree_post_build",
    }
    stages = tuple(
        TOOL.actual_stage_evidence_v1(
            stage_id,
            TOOL.ACTUAL_ORCHESTRATION_STAGE_REGISTRY[stage_id - 1][1],
            source_commit,
            evidence,
            qualification_count=0,
            qualification_mask=0,
            candidate_receipt_hex=None,
            final_receipt_hex=None,
        )
        for stage_id, evidence in enumerate(
            (stage1_evidence, stage2_evidence, stage3_evidence),
            start=1,
        )
    )
    fresh = TOOL.GitSourceStatusV1(source_commit, True, 0)
    git_transcript = _synthetic_git_source_transcript(tmp_path, source_commit)
    absence = TOOL.actual_artifact_absence_evidence_v1(artifact)
    fresh_image_rows = []
    for label, reference in (("python", TOOL.PYTHON_IMAGE), ("rust", TOOL.RUST_IMAGE)):
        evidence = image_rows[label]
        fresh_image_rows.append(
            {
                "label": label,
                "reference": reference,
                "evidence": evidence,
                "evidence_root": ADMISSION.fresh_runtime_evidence_object_root_v1(
                    "PINNED_IMAGE", label, evidence
                ),
            }
        )
    fresh_actor_rows = []
    for actor_id in (
        "PYTHON_ENDPOINT",
        "RUST_ENDPOINT",
        "TRUSTED_HOST_REPLAY",
    ):
        identity = TOOL._fresh_actor_source_identity_v1(
            source_rows[actor_id], snapshot_rows[actor_id]
        )
        fresh_actor_rows.append(
            {
                "actor_id": actor_id,
                "source_identity": identity,
                "source_identity_root": (
                    ADMISSION.fresh_runtime_evidence_object_root_v1(
                        "ACTOR_SOURCE", actor_id, identity
                    )
                ),
                "snapshot_evidence": snapshot_rows[actor_id],
                "snapshot_evidence_root": (
                    ADMISSION.fresh_runtime_evidence_object_root_v1(
                        "ACTOR_SNAPSHOT", actor_id, snapshot_rows[actor_id]
                    )
                ),
            }
        )
    seccomp_rows = []
    for label, relative, evidence in (
        ("runtime", TOOL.RUNTIME_SECCOMP_RELATIVE_PATH, runtime_seccomp),
        ("build", TOOL.BUILD_SECCOMP_RELATIVE_PATH, build_seccomp),
    ):
        seccomp_rows.append(
            {
                "label": label,
                "relative_path": relative,
                "evidence": evidence,
                "evidence_root": ADMISSION.fresh_runtime_evidence_object_root_v1(
                    "SECCOMP_POLICY", label, evidence
                ),
            }
        )
    cargo_identity = TOOL._fresh_cargo_material_identity_v1(cargo_material)
    binary_identity = TOOL._fresh_binary_identity_v1(binary)
    fresh_runtime = TOOL.build_fresh_runtime_evidence_set_v1(
        source_commit,
        fresh_image_rows,
        fresh_actor_rows,
        cargo_identity,
        cargo_snapshot,
        cargo_tree,
        seccomp_rows,
        binary_identity,
    )
    bundle = TOOL.build_actual_precondition_bundle_v1(
        source_commit,
        commit_a_config,
        artifact,
        work_identity,
        stages,
        fresh,
        git_transcript,
        absence,
        fresh_runtime,
    )
    decision = TOOL.build_actual_admission_decision_v1(
        source_commit,
        commit_a_config,
        artifact,
        b"N" * 32,
        bundle,
    )
    boundary = TOOL.build_stage3_to4_admission_boundary_v1(
        source_commit,
        commit_a_config,
        artifact,
        bundle,
        decision,
    )
    return {
        "source_commit": source_commit,
        "commit_a_config": commit_a_config,
        "artifact": artifact,
        "work_identity": work_identity,
        "stages": stages,
        "fresh": fresh,
        "git_transcript": git_transcript,
        "absence": absence,
        "fresh_runtime": fresh_runtime,
        "bundle": bundle,
        "decision": decision,
        "boundary": boundary,
    }


def test_actual_admission_bundle_decision_boundary_roundtrip_stays_zero(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_actual_admission_fixture(tmp_path)
    pure_bundle = ADMISSION.build_actual_precondition_bundle_v1(
        fixture["source_commit"],
        fixture["commit_a_config"],
        fixture["artifact"].as_posix(),
        fixture["work_identity"],
        fixture["bundle"]["prior_stage_root_rows"],
        [row["preimage"] for row in fixture["bundle"]["ordered_precondition_rows"]],
    )
    assert pure_bundle == fixture["bundle"]
    bundle_payload = TOOL._canonical_json_bytes(fixture["bundle"])
    assert TOOL.decode_actual_precondition_bundle_v1(
        bundle_payload,
        fixture["source_commit"],
        fixture["commit_a_config"],
        fixture["artifact"],
        fixture["work_identity"],
        fixture["stages"],
        fixture["fresh"],
        fixture["git_transcript"],
        fixture["absence"],
        fixture["fresh_runtime"],
    ) == fixture["bundle"]
    decision_payload = TOOL.canonical_actual_admission_decision_bytes_v1(
        fixture["decision"],
        fixture["commit_a_config"],
        fixture["source_commit"],
        fixture["artifact"],
        fixture["bundle"],
    )
    assert TOOL.decode_actual_admission_decision_v1(
        decision_payload,
        fixture["commit_a_config"],
        fixture["source_commit"],
        fixture["artifact"],
        fixture["bundle"],
    ) == fixture["decision"]
    assert TOOL.decode_stage3_to4_admission_boundary_v1(
        TOOL._canonical_json_bytes(fixture["boundary"]),
        fixture["source_commit"],
        fixture["commit_a_config"],
        fixture["artifact"],
        fixture["work_identity"],
        fixture["stages"],
        fixture["fresh"],
        fixture["git_transcript"],
        fixture["absence"],
        fixture["fresh_runtime"],
    ) == fixture["boundary"]
    decision = fixture["decision"]
    assert decision["decision"] == "ADMITTED_FOR_ONE_ATTEMPT"
    assert decision["attempt_nonce_hex"] == (b"N" * 32).hex()
    assert decision["artifact_path"] == fixture["artifact"].as_posix()
    assert fixture["bundle"]["precondition_count"] == len(
        TOOL.ACTUAL_RUNTIME_PRECONDITION_REGISTRY
    )
    assert fixture["bundle"]["precondition_mask"] == (
        1 << len(TOOL.ACTUAL_RUNTIME_PRECONDITION_REGISTRY)
    ) - 1
    assert all(
        row["passed"] is True
        for row in fixture["bundle"]["ordered_precondition_rows"]
    )
    assert decision["qualification_authority_at_admission"] == {
        "candidate_receipt": None,
        "final_receipt": None,
        "predicate_count": 0,
        "predicate_mask": 0,
        "predicate_total": 20,
    }
    assert decision["closed_q1_authority"]["state"] == "NOT_RUN"
    assert decision["closed_q1_authority"]["formal_output_roots"] == [None] * 8
    current = TOOL.load_isolation_config_v1(ROOT)
    assert current["engineering_status"] == TOOL.COMMIT_A_ACTUAL_ENGINEERING_STATUS
    assert current["actual_preconditions"] == TOOL.COMMIT_A_ACTUAL_PRECONDITIONS_V1


@pytest.mark.parametrize(
    ("seccomp_index", "replacement_sha256"),
    ((0, "a1" * 32), (1, "b2" * 32)),
)
def test_actual_admission_predicate8_rejects_coordinated_seccomp_rehash(
    tmp_path: Path,
    seccomp_index: int,
    replacement_sha256: str,
) -> None:
    fixture = _synthetic_actual_admission_fixture(tmp_path)
    fresh = deepcopy(fixture["fresh_runtime"])
    seccomp_row = fresh["seccomp_rows"][seccomp_index]
    evidence = seccomp_row["evidence"]
    evidence["payload_sha256"] = replacement_sha256
    evidence_body = dict(evidence)
    evidence_body.pop("manifest_sha256")
    evidence["manifest_sha256"] = sha256(
        ADMISSION.canonical_json_bytes_v1(evidence_body)
    ).hexdigest()
    seccomp_row["evidence_root"] = (
        ADMISSION.fresh_runtime_evidence_object_root_v1(
            "SECCOMP_POLICY",
            seccomp_row["label"],
            evidence,
        )
    )

    host_actor = fresh["actor_rows"][2]
    host_snapshot = host_actor["snapshot_evidence"]
    relative_path = seccomp_row["relative_path"]
    snapshot_file_row = next(
        row for row in host_snapshot["file_rows"] if row[0] == relative_path
    )
    snapshot_file_row[10] = replacement_sha256
    snapshot_body = dict(host_snapshot)
    snapshot_body.pop("manifest_sha256")
    host_snapshot["manifest_sha256"] = sha256(
        ADMISSION.canonical_json_bytes_v1(snapshot_body)
    ).hexdigest()
    host_actor["snapshot_evidence_root"] = (
        ADMISSION.fresh_runtime_evidence_object_root_v1(
            "ACTOR_SNAPSHOT",
            host_actor["actor_id"],
            host_snapshot,
        )
    )
    source_identity = host_actor["source_identity"]
    source_identity["snapshot_file_registry_sha256"] = sha256(
        ADMISSION.canonical_json_bytes_v1(
            [
                [row[0], row[6], row[7], row[10]]
                for row in host_snapshot["file_rows"]
            ]
        )
    ).hexdigest()
    host_actor["source_identity_root"] = (
        ADMISSION.fresh_runtime_evidence_object_root_v1(
            "ACTOR_SOURCE",
            host_actor["actor_id"],
            source_identity,
        )
    )

    cargo = fresh["cargo"]
    binary = fresh["binary"]
    rebuilt = ADMISSION.build_fresh_runtime_evidence_set_v1(
        fixture["source_commit"],
        fresh["image_rows"],
        fresh["actor_rows"],
        cargo["material_identity"],
        cargo["snapshot_evidence"],
        cargo["tree_evidence"],
        fresh["seccomp_rows"],
        binary["identity"],
    )
    preimages = deepcopy(
        [
            row["preimage"]
            for row in fixture["bundle"]["ordered_precondition_rows"]
        ]
    )
    preimages[4]["image_rows"] = rebuilt["image_rows"]
    preimages[5]["actor_rows"] = rebuilt["actor_rows"]
    preimages[7]["seccomp_rows"] = rebuilt["seccomp_rows"]
    for index in range(4, 8):
        preimages[index]["fresh_runtime_evidence_root"] = rebuilt[
            "fresh_runtime_evidence_root"
        ]

    with pytest.raises(
        ADMISSION.Q05BActualAdmissionError,
        match="payload differs from Commit-A config",
    ):
        ADMISSION.build_actual_precondition_bundle_v1(
            fixture["source_commit"],
            fixture["commit_a_config"],
            fixture["artifact"].as_posix(),
            fixture["work_identity"],
            fixture["bundle"]["prior_stage_root_rows"],
            preimages,
        )


def test_actual_admission_fresh_leaf_set_tamper_and_type_alias_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_actual_admission_fixture(tmp_path)
    fresh = json.loads(json.dumps(fixture["fresh_runtime"]))
    tampered_leaf = json.loads(json.dumps(fresh))
    tampered_leaf["image_rows"][0]["evidence"]["architecture"] = "arm64"
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.validate_fresh_runtime_evidence_set_v1(
            tampered_leaf, fixture["source_commit"]
        )
    tampered_set_root = json.loads(json.dumps(fresh))
    tampered_set_root["fresh_runtime_evidence_root"] = "00" * 32
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.validate_fresh_runtime_evidence_set_v1(
            tampered_set_root, fixture["source_commit"]
        )
    bool_alias = json.loads(json.dumps(fresh))
    bool_alias["actor_rows"][0]["snapshot_evidence"]["root_nlink"] = True
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.validate_fresh_runtime_evidence_set_v1(
            bool_alias, fixture["source_commit"]
        )
    preimages = json.loads(
        json.dumps(
            [row["preimage"] for row in fixture["bundle"]["ordered_precondition_rows"]]
        )
    )
    preimages[4]["fresh_runtime_evidence_root"] = "01" * 32
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.build_actual_precondition_bundle_v1(
            fixture["source_commit"],
            fixture["commit_a_config"],
            fixture["artifact"].as_posix(),
            fixture["work_identity"],
            fixture["bundle"]["prior_stage_root_rows"],
            preimages,
        )


def test_fresh_runtime_collector_rejects_missing_seccomp_source_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _synthetic_actual_admission_fixture(tmp_path)
    stage1 = fixture["stages"][0]["evidence"]
    stage2 = fixture["stages"][1]["evidence"]
    stage3 = fixture["stages"][2]["evidence"]
    source = json.loads(json.dumps(stage1["source_evidence"]))
    host = source["TRUSTED_HOST_REPLAY"]
    host["blob_preimage_rows"] = [
        row
        for row in host["blob_preimage_rows"]
        if row[0] != TOOL.BUILD_SECCOMP_RELATIVE_PATH
    ]
    host["rows"] = [
        row for row in host["rows"] if row[0] != TOOL.BUILD_SECCOMP_RELATIVE_PATH
    ]
    snapshots = stage2["snapshot_evidence"]
    images = stage1["image_evidence"]
    cargo = stage2["cargo_evidence"]
    seccomp = stage2["seccomp_evidence"]
    binary = stage3["binary"]
    monkeypatch.setattr(
        TOOL,
        "local_pinned_image_evidence_v1",
        lambda image, runner: images["python" if image == TOOL.PYTHON_IMAGE else "rust"],
    )
    monkeypatch.setattr(
        TOOL,
        "actor_source_evidence_v1",
        lambda _root, _commit, actor_id: source[actor_id],
    )
    monkeypatch.setattr(
        TOOL,
        "sealed_snapshot_path_evidence_v1",
        lambda _root, allowlist: snapshots[
            next(
                actor_id
                for actor_id, rows in TOOL.ACTOR_SOURCE_ALLOWLISTS.items()
                if rows is allowlist
            )
        ],
    )
    monkeypatch.setattr(
        TOOL,
        "sealed_snapshot_identity_v1",
        lambda _root, _paths: cargo["sealed_snapshot_identity"],
    )
    monkeypatch.setattr(
        TOOL,
        "sealed_tree_identity_v1",
        lambda _root, _paths, **_kwargs: cargo["sealed_tree_identity"],
    )
    paths = {
        "python_snapshot": tmp_path / "python",
        "rust_snapshot": tmp_path / "rust",
        "host_snapshot": tmp_path / "host",
        "cargo_home": tmp_path / "cargo",
        "binary": tmp_path / "binary",
    }
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.collect_fresh_runtime_evidence_set_v1(
            tmp_path.resolve(),
            fixture["source_commit"],
            paths,
            source,
            snapshots,
            images,
            cargo,
            seccomp,
            binary,
            command_runner=lambda *_args, **_kwargs: None,
        )
    assert failure.value.code == TOOL.FAIL_ACTUAL_ADMISSION


def test_actual_admission_bundle_and_boundary_frozen_byte_caps(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_actual_admission_fixture(tmp_path)
    base_preimages = [
        row["preimage"] for row in fixture["bundle"]["ordered_precondition_rows"]
    ]

    def build_with_padding(length: int):
        fresh = json.loads(json.dumps(fixture["fresh_runtime"]))
        evidence = fresh["image_rows"][0]["evidence"]
        raw = json.loads(bytes.fromhex(evidence["raw_inspect_hex"]))
        raw[0]["Padding"] = "x" * length
        raw_payload = ADMISSION.canonical_json_bytes_v1(raw)
        evidence["raw_inspect_hex"] = raw_payload.hex()
        evidence["raw_inspect_sha256"] = sha256(raw_payload).hexdigest()
        evidence_body = dict(evidence)
        evidence_body.pop("evidence_sha256")
        evidence["evidence_sha256"] = sha256(
            ADMISSION.canonical_json_bytes_v1(evidence_body)
        ).hexdigest()
        fresh["image_rows"][0]["evidence_root"] = (
            ADMISSION.fresh_runtime_evidence_object_root_v1(
                "PINNED_IMAGE", "python", evidence
            )
        )
        cargo = fresh["cargo"]
        binary = fresh["binary"]
        rebuilt = ADMISSION.build_fresh_runtime_evidence_set_v1(
            fixture["source_commit"],
            fresh["image_rows"],
            fresh["actor_rows"],
            cargo["material_identity"],
            cargo["snapshot_evidence"],
            cargo["tree_evidence"],
            fresh["seccomp_rows"],
            binary["identity"],
        )
        preimages = json.loads(json.dumps(base_preimages))
        preimages[4]["image_rows"] = rebuilt["image_rows"]
        preimages[5]["actor_rows"] = rebuilt["actor_rows"]
        preimages[6]["cargo_material_identity"] = rebuilt["cargo"][
            "material_identity"
        ]
        preimages[6]["cargo_material_identity_root"] = rebuilt["cargo"][
            "material_identity_root"
        ]
        preimages[6]["cargo_snapshot_evidence"] = rebuilt["cargo"][
            "snapshot_evidence"
        ]
        preimages[6]["cargo_snapshot_evidence_root"] = rebuilt["cargo"][
            "snapshot_evidence_root"
        ]
        preimages[6]["cargo_tree_evidence"] = rebuilt["cargo"]["tree_evidence"]
        preimages[6]["cargo_tree_evidence_root"] = rebuilt["cargo"][
            "tree_evidence_root"
        ]
        preimages[7]["seccomp_rows"] = rebuilt["seccomp_rows"]
        preimages[7]["binary_identity"] = rebuilt["binary"]["identity"]
        preimages[7]["binary_identity_root"] = rebuilt["binary"][
            "identity_root"
        ]
        for index in range(4, 8):
            preimages[index]["fresh_runtime_evidence_root"] = rebuilt[
                "fresh_runtime_evidence_root"
            ]
        return ADMISSION.build_actual_precondition_bundle_v1(
            fixture["source_commit"],
            fixture["commit_a_config"],
            fixture["artifact"].as_posix(),
            fixture["work_identity"],
            fixture["bundle"]["prior_stage_root_rows"],
            preimages,
        )

    low = 0
    high = 3 * 1024 * 1024
    while low < high:
        middle = (low + high + 1) // 2
        try:
            build_with_padding(middle)
        except ADMISSION.Q05BActualAdmissionError:
            high = middle - 1
        else:
            low = middle
    near = build_with_padding(low)
    near_payload = ADMISSION.canonical_json_bytes_v1(near)
    assert len(near_payload) <= ADMISSION.ACTUAL_PRECONDITION_BUNDLE_MAX_BYTES
    assert (
        ADMISSION.ACTUAL_PRECONDITION_BUNDLE_MAX_BYTES - len(near_payload)
    ) < 8
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        build_with_padding(low + 1)
    decision = ADMISSION.build_actual_admission_decision_v1(
        fixture["source_commit"],
        fixture["commit_a_config"],
        fixture["artifact"].as_posix(),
        b"Z" * 32,
        near,
    )
    boundary = ADMISSION.build_stage3_to4_admission_boundary_v1(
        fixture["source_commit"],
        fixture["commit_a_config"],
        fixture["artifact"].as_posix(),
        near,
        decision,
    )
    assert len(ADMISSION.canonical_json_bytes_v1(boundary)) < (
        ADMISSION.ACTUAL_ADMISSION_BOUNDARY_MAX_BYTES
    )
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.decode_actual_precondition_bundle_v1(
            b"x" * (ADMISSION.ACTUAL_PRECONDITION_BUNDLE_MAX_BYTES + 1),
            fixture["source_commit"],
            fixture["commit_a_config"],
            fixture["artifact"].as_posix(),
            fixture["work_identity"],
            fixture["bundle"]["prior_stage_root_rows"],
            base_preimages,
        )
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.decode_stage3_to4_admission_boundary_v1(
            b"x" * (ADMISSION.ACTUAL_ADMISSION_BOUNDARY_MAX_BYTES + 1),
            fixture["source_commit"],
            fixture["commit_a_config"],
            fixture["artifact"].as_posix(),
            fixture["bundle"],
            fixture["decision"],
        )


def _synthetic_mount_stdout_tree_v1(root: Path) -> dict[str, object]:
    file_rows = []
    for ordinal, relative in enumerate(
        ("manifest.json", "python.stdout", "rust.stdout"), start=1
    ):
        file_rows.append(
            [
                relative,
                103,
                3_700 + ordinal,
                1,
                1000,
                1000,
                0o444,
                64 + ordinal,
                1,
                1,
                sha256(f"stdout-{relative}".encode()).hexdigest(),
            ]
        )
    body = {
        "schema_version": "hegel-phase3a-q05b-sealed-tree-identity/1",
        "root_path": root.as_posix(),
        "root_device": 103,
        "root_inode": 3_700,
        "root_nlink": 2,
        "root_mode": 0o555,
        "directory_rows": [],
        "file_rows": file_rows,
    }
    return {
        **body,
        "manifest_sha256": sha256(
            TOOL._canonical_json_bytes(body)
        ).hexdigest(),
    }


def _synthetic_actor_mount_binding_v1(
    root: Path,
    role_id: int,
    fresh_runtime: dict[str, object] | None = None,
) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    docker_authority = _docker_execution_authority()
    docker_slot = {
        1: "PYTHON_ENDPOINT",
        2: "RUST_ENDPOINT",
        3: "TRUSTED_HOST_REPLAY",
    }[role_id]
    docker_slot_row = _docker_slot_row(docker_authority, docker_slot)
    actors = (
        {row["actor_id"]: row for row in fresh_runtime["actor_rows"]}
        if fresh_runtime is not None
        else {}
    )
    runtime_seccomp = (
        next(
            row["evidence"]
            for row in fresh_runtime["seccomp_rows"]
            if row["label"] == "runtime"
        )
        if fresh_runtime is not None
        else None
    )

    def sealed_tree(path: Path, ordinal: int) -> dict[str, object]:
        body = {
            "schema_version": "hegel-phase3a-q05b-sealed-tree-identity/1",
            "root_path": path.as_posix(),
            "root_device": 100 + role_id,
            "root_inode": 10_000 + role_id * 100 + ordinal,
            "root_nlink": 2,
            "root_mode": 0o555,
            "directory_rows": [],
            "file_rows": [
                [
                    "payload.bin",
                    100 + role_id,
                    20_000 + role_id * 100 + ordinal,
                    1,
                    1000,
                    1000,
                    0o444,
                    1,
                    1,
                    1,
                    sha256(f"tree-{role_id}-{ordinal}".encode()).hexdigest(),
                ]
            ],
        }
        return {
            **body,
            "manifest_sha256": sha256(
                TOOL._canonical_json_bytes(body)
            ).hexdigest(),
        }

    def synthetic_seccomp(path: Path) -> dict[str, object]:
        body = {
            "schema_version": "hegel-phase3a-q05b-sealed-policy-file/1",
            "absolute_path": path.as_posix(),
            "snapshot_relative_path": TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
            "file_device": 200 + role_id,
            "file_inode": 30_000 + role_id,
            "file_nlink": 1,
            "file_uid": 1000,
            "file_gid": 1000,
            "file_mode": 0o444,
            "file_size": 64,
            "file_mtime_ns": 1,
            "file_ctime_ns": 1,
            "payload_sha256": sha256(b"synthetic-seccomp").hexdigest(),
        }
        return {
            **body,
            "manifest_sha256": sha256(
                TOOL._canonical_json_bytes(body)
            ).hexdigest(),
        }

    def synthetic_binary(path: Path) -> dict[str, object]:
        return {
            "schema_version": (
                "hegel-phase3a-q05b-fresh-prebuilt-rust-binary-identity/1"
            ),
            "binary_path": path.as_posix(),
            "device": 300 + role_id,
            "inode": 40_000 + role_id,
            "nlink": 1,
            "uid": 1000,
            "gid": 1000,
            "mode": 0o555,
            "size": 128,
            "mtime_ns": 1,
            "ctime_ns": 1,
            "sha256": sha256(b"synthetic-binary").hexdigest(),
            "sealed_binary_manifest_sha256": "81" * 32,
            "stage_3_binary_evidence_sha256": "82" * 32,
        }

    seccomp_evidence = runtime_seccomp or synthetic_seccomp(
        root / "runtime-seccomp.json"
    )
    seccomp = Path(seccomp_evidence["absolute_path"])
    sources: dict[str, Path] = {}
    if role_id == 1:
        snapshot = (
            Path(actors["PYTHON_ENDPOINT"]["snapshot_evidence"]["root_path"])
            if fresh_runtime is not None
            else root / "python-snapshot"
        )
        sources = {
            "/control": root / "python-control",
            "/output": root / "python-output",
            "/snapshot": snapshot,
        }
        command = TOOL.python_endpoint_command_v1(
            sources["/snapshot"],
            sources["/output"],
            sources["/control"],
            seccomp,
            docker_slot_row=docker_slot_row,
            cidfile=root / "python.cid",
        )
    elif role_id == 2:
        binary = (
            fresh_runtime["binary"]["identity"]
            if fresh_runtime is not None
            else synthetic_binary(root / "oracle")
        )
        sources = {
            "/control": root / "rust-control",
            "/output": root / "rust-output",
            "/runtime/hegel-q1-archive-projection-oracle": Path(
                binary["binary_path"]
            ),
        }
        command = TOOL.rust_runtime_command_v1(
            sources["/runtime/hegel-q1-archive-projection-oracle"],
            sources["/output"],
            sources["/control"],
            seccomp,
            docker_slot_row=docker_slot_row,
            cidfile=root / "rust.cid",
        )
    elif role_id == 3:
        snapshot = (
            Path(
                actors["TRUSTED_HOST_REPLAY"]["snapshot_evidence"][
                    "root_path"
                ]
            )
            if fresh_runtime is not None
            else root / "host-snapshot"
        )
        sources = {
            "/control": root / "host-control",
            "/inputs/python": root / "python-output",
            "/inputs/rust": root / "rust-output",
            "/inputs/stdout/python.stdout": root / "python.stdout",
            "/inputs/stdout/rust.stdout": root / "rust.stdout",
            "/inputs/stdout/manifest.json": root / "manifest.json",
            "/snapshot": snapshot,
            "/staging": root / "host-staging",
        }
        command = TOOL.trusted_host_command_v1(
            sources["/snapshot"],
            sources["/inputs/python"],
            sources["/inputs/rust"],
            sources["/inputs/stdout/python.stdout"],
            sources["/inputs/stdout/rust.stdout"],
            sources["/inputs/stdout/manifest.json"],
            sources["/control"],
            sources["/staging"],
            seccomp,
            docker_slot_row=docker_slot_row,
            cidfile=root / "host.cid",
        )
    else:
        raise AssertionError("unexpected synthetic mount role")
    registry = TOOL.sealed_actor_mount_registry_v1(role_id, command)
    stdout_tree = (
        _synthetic_mount_stdout_tree_v1(root) if role_id == 3 else None
    )
    expected = ADMISSION.ACTUAL_ACTOR_MOUNT_ROLE_REGISTRY[role_id - 1][2]
    source_rows = []
    for ordinal, (destination, writable, source_type, mode) in enumerate(
        expected, start=1
    ):
        authority_kind, authority_label = next(
            (kind, label)
            for row_role, row_destination, kind, label in (
                ADMISSION.ACTUAL_ACTOR_MOUNT_AUTHORITY_REGISTRY
            )
            if row_role == role_id and row_destination == destination
        )
        source = sources[destination]
        if authority_kind == "PRELAUNCH_WRITABLE_DIRECTORY":
            evidence = ADMISSION.build_prelaunch_writable_directory_evidence_v1(
                role_id,
                destination,
                source.as_posix(),
                100 + role_id,
                role_id * 1000 + ordinal,
                2,
                1000,
                1000,
                mode,
            )
            identity = (
                100 + role_id,
                role_id * 1000 + ordinal,
                2,
                1000,
                1000,
                mode,
                None,
            )
        elif authority_kind == "FRESH_ACTOR_SNAPSHOT":
            actor_id = (
                "PYTHON_ENDPOINT" if role_id == 1 else "TRUSTED_HOST_REPLAY"
            )
            evidence = (
                actors[actor_id]["snapshot_evidence"]
                if fresh_runtime is not None
                else sealed_tree(source, ordinal)
            )
            identity = (
                evidence["root_device"],
                evidence["root_inode"],
                evidence["root_nlink"],
                1000,
                1000,
                evidence["root_mode"],
                None,
            )
        elif authority_kind == "FRESH_PREBUILT_RUST_BINARY":
            evidence = (
                fresh_runtime["binary"]["identity"]
                if fresh_runtime is not None
                else synthetic_binary(source)
            )
            identity = (
                evidence["device"],
                evidence["inode"],
                evidence["nlink"],
                evidence["uid"],
                evidence["gid"],
                evidence["mode"],
                evidence["size"],
            )
        elif authority_kind == "SEALED_ENDPOINT_TREE":
            evidence = sealed_tree(source, ordinal)
            identity = (
                evidence["root_device"],
                evidence["root_inode"],
                evidence["root_nlink"],
                1000,
                1000,
                evidence["root_mode"],
                None,
            )
        elif authority_kind == "SEALED_STDOUT_FILE":
            relative = destination.rsplit("/", 1)[-1]
            assert stdout_tree is not None
            file_row = next(
                row for row in stdout_tree["file_rows"] if row[0] == relative
            )
            evidence = {
                "schema_version": (
                    "hegel-phase3a-q05b-sealed-stdout-mount-file/1"
                ),
                "tree_manifest_sha256": stdout_tree["manifest_sha256"],
                "relative_path": relative,
                "file_row": file_row,
            }
            identity = (
                file_row[1],
                file_row[2],
                file_row[3],
                file_row[4],
                file_row[5],
                file_row[6],
                file_row[7],
            )
        else:
            raise AssertionError("unexpected synthetic mount authority")
        source_rows.append(
            ADMISSION.build_actor_mount_source_row_v1(
                role_id,
                destination,
                registry.expected_sources[destination],
                writable,
                source_type,
                identity[0],
                identity[1],
                identity[2],
                identity[3],
                identity[4],
                identity[5],
                identity[6],
                authority_kind,
                authority_label,
                evidence,
            )
        )
    seccomp_kind, seccomp_label = next(
        (kind, label)
        for row_role, row_destination, kind, label in (
            ADMISSION.ACTUAL_ACTOR_MOUNT_AUTHORITY_REGISTRY
        )
        if row_role == role_id and row_destination == "@seccomp"
    )
    seccomp_row = ADMISSION.build_actor_mount_source_row_v1(
        role_id,
        "@seccomp",
        seccomp.as_posix(),
        False,
        "REGULAR_FILE",
        seccomp_evidence["file_device"],
        seccomp_evidence["file_inode"],
        seccomp_evidence["file_nlink"],
        seccomp_evidence["file_uid"],
        seccomp_evidence["file_gid"],
        seccomp_evidence["file_mode"],
        seccomp_evidence["file_size"],
        seccomp_kind,
        seccomp_label,
        seccomp_evidence,
    )
    return ADMISSION.build_actor_mount_binding_v1(
        command,
        TOOL.actor_mount_registry_object_v1(registry),
        source_rows,
        seccomp_row,
    )


def _synthetic_mount_launch_replay_v1(
    binding: dict[str, object],
) -> dict[str, object]:
    def replay(row: dict[str, object]) -> dict[str, object]:
        if row["source_type"] == "DIRECTORY":
            payload_sha256 = None
        elif row["authority_kind"] == "FRESH_PREBUILT_RUST_BINARY":
            payload_sha256 = row["authority_evidence"]["sha256"]
        else:
            payload_sha256 = row["authority_evidence"]["payload_sha256"]
        return {
            "destination": row["destination"],
            "source": row["source"],
            "source_device": row["source_device"],
            "source_inode": row["source_inode"],
            "source_nlink": row["source_nlink"],
            "source_uid": row["source_uid"],
            "source_gid": row["source_gid"],
            "source_mode": row["source_mode"],
            "source_type": row["source_type"],
            "payload_sha256": payload_sha256,
            "path_matches_held_descriptor": True,
            "held_descriptor_read_only": True,
        }

    return ADMISSION.build_actor_mount_launch_replay_v1(
        binding,
        [replay(row) for row in binding["source_rows"]],
        replay(binding["seccomp_row"]),
    )


def test_fresh_runtime_checkpoint_order_roots_and_byte_equality(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_actual_admission_fixture(tmp_path)
    boundary_payload = ADMISSION.canonical_json_bytes_v1(fixture["boundary"])
    issued = ADMISSION.build_actual_admission_issued_marker_evidence_v1(
        fixture["decision"]["attempt_id"],
        fixture["boundary"]["boundary_root"],
        boundary_payload,
        file_device=10,
        file_inode=11,
        file_nlink=1,
        file_mode=0o444,
        work_root_device=fixture["work_identity"]["device"],
        work_root_inode=fixture["work_identity"]["inode"],
        work_root_mode=0o700,
    )
    issue_record = ADMISSION.build_actual_admission_issue_record_v1(
        fixture["boundary"], issued
    )
    spending = ADMISSION.build_actual_admission_spending_intent_v1(issue_record)
    consumed = ADMISSION.build_actual_admission_consumed_marker_evidence_v1(
        issue_record,
        spending,
        spending_file_device=10,
        spending_file_inode=12,
        spending_file_nlink=1,
        spending_file_mode=0o444,
        file_device=10,
        file_inode=11,
        file_nlink=2,
        file_mode=0o444,
        work_root_device=fixture["work_identity"]["device"],
        work_root_inode=fixture["work_identity"]["inode"],
        work_root_mode=0o700,
    )
    bindings = {
        role_id: _synthetic_actor_mount_binding_v1(
            tmp_path / f"mount-role-{role_id}",
            role_id,
            fixture["fresh_runtime"],
        )
        for role_id in (1, 2, 3)
    }
    role3_sources = {
        row["destination"]: row for row in bindings[3]["source_rows"]
    }
    stage5_stdout_tree = _synthetic_mount_stdout_tree_v1(
        tmp_path / "mount-role-3"
    )
    stage5_five_sidecars = {
        "canonical_rows": [],
        "python_output_tree": role3_sources["/inputs/python"][
            "authority_evidence"
        ],
        "rust_output_tree": role3_sources["/inputs/rust"][
            "authority_evidence"
        ],
    }
    stage5_endpoint_stdout_set = {
        "python_stdout_hex": "",
        "rust_stdout_hex": "",
        "manifest_hex": "",
        "sealed_stdout_tree": stage5_stdout_tree,
    }
    strict_endpoint_replay_roots = ["62" * 32, "63" * 32]
    work_replay = {
        "schema_version": "hegel-phase3a-q05b-admission-work-root-replay/1",
        "absolute_path": fixture["work_identity"]["absolute_path"],
        "device": fixture["work_identity"]["device"],
        "inode": fixture["work_identity"]["inode"],
        "nlink": fixture["work_identity"]["nlink"],
        "mode": 0o700,
        "path_matches_anchored_descriptor": True,
    }
    launches = [
        _synthetic_mount_launch_replay_v1(bindings[role_id])
        for role_id in (1, 2)
    ]
    checkpoint_1 = ADMISSION.build_fresh_runtime_checkpoint_v1(
        fixture["source_commit"],
        fixture["artifact"].as_posix(),
        1,
        fixture["decision"]["attempt_id"],
        fixture["boundary"]["boundary_root"],
        issue_record["issue_record_root"],
        consumed["consumed_marker_root"],
        fixture["fresh_runtime"],
        fixture["fresh_runtime"],
        fixture["absence"],
        [bindings[1], bindings[2]],
        None,
        None,
    )
    actor_completion_rows = [
        {
            "actor_id": binding["actor_id"],
            "command_sha256": binding["command_mount_registry"][
                "command_sha256"
            ],
            "mount_registry_sha256": binding["command_mount_registry"][
                "registry_sha256"
            ],
            "seccomp_evidence": binding["seccomp_row"]["authority_evidence"],
            "control_root": f"{59 + binding['role_id']:02x}" * 32,
        }
        for binding in (bindings[1], bindings[2])
    ]
    stage5_live = ADMISSION.build_actual_admission_live_marker_replay_v1(
        "STAGE_05_BEFORE_EVIDENCE",
        issue_record,
        consumed,
        work_root_device=fixture["work_identity"]["device"],
        work_root_inode=fixture["work_identity"]["inode"],
        work_root_nlink=fixture["work_identity"]["nlink"],
        work_root_mode=0o700,
        issued_file_device=10,
        issued_file_inode=11,
        issued_file_nlink=2,
        consumed_file_device=10,
        consumed_file_inode=11,
        consumed_file_nlink=2,
        spending_file_device=10,
        spending_file_inode=12,
        spending_file_nlink=1,
    )
    injected = {
        "actual_admission_attempt_id": fixture["decision"]["attempt_id"],
        "actual_admission_boundary_root": fixture["boundary"]["boundary_root"],
        "actual_admission_issue_record_root": issue_record["issue_record_root"],
        "actual_admission_consumed_marker_evidence": consumed,
        "actual_admission_work_root_replay": work_replay,
        "actual_admission_consume_git_source_transcript": fixture[
            "git_transcript"
        ],
        "actual_admission_consume_artifact_absence": fixture["absence"],
        "actual_admission_fresh_checkpoint_root_rows": [[
            1,
            ADMISSION.ACTUAL_FRESH_RUNTIME_CHECKPOINT_REGISTRY[0][1],
            checkpoint_1["checkpoint_root"],
        ]],
        "actual_actor_mount_binding_root_rows": [[
            binding["role_id"], binding["actor_id"], binding["mount_binding_root"]
        ] for binding in (bindings[1], bindings[2])],
        "actual_actor_mount_launch_root_rows": [[
            launch["role_id"], launch["actor_id"], launch["launch_replay_root"]
        ] for launch in launches],
        "actual_admission_live_marker_replay": stage5_live,
    }
    stage5_evidence = ADMISSION.build_actual_stage_5_evidence_v1(
        fixture["source_commit"],
        actor_completion_rows,
        stage5_five_sidecars,
        stage5_endpoint_stdout_set,
        strict_endpoint_replay_roots,
        injected,
    )
    strong_stage5_context = {
        "issue_record": issue_record,
        "consumed_marker_evidence": consumed,
        "checkpoint_1": checkpoint_1,
        "mount_launch_replay_rows": launches,
    }
    strong_checkpoint_context = {
        "stage_5_issue_record": issue_record,
        "stage_5_consumed_marker_evidence": consumed,
        "stage_5_checkpoint_1": checkpoint_1,
        "stage_5_mount_launch_replay_rows": launches,
    }
    assert ADMISSION.validate_actual_stage_5_evidence_surface_v1(
        stage5_evidence, fixture["source_commit"]
    ) == stage5_evidence
    assert ADMISSION.validate_actual_stage_5_evidence_v1(
        stage5_evidence,
        fixture["source_commit"],
        **strong_stage5_context,
    ) == stage5_evidence
    with pytest.raises(TypeError):
        ADMISSION.validate_actual_stage_5_evidence_v1(
            stage5_evidence, fixture["source_commit"]
        )
    for live_file_fields in (
        ("issued_file_device", "consumed_file_device"),
        ("issued_file_inode", "consumed_file_inode"),
        ("spending_file_device",),
        ("spending_file_inode",),
    ):
        forged_stage5 = json.loads(json.dumps(stage5_evidence))
        forged_live = forged_stage5["evidence"][
            "actual_admission_live_marker_replay"
        ]
        for field in live_file_fields:
            forged_live[field] += 1
        forged_live_body = dict(forged_live)
        forged_live_body.pop("live_marker_replay_root")
        forged_live["live_marker_replay_root"] = sha256(
            ADMISSION.ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_ROOT_DOMAIN
            + ADMISSION.canonical_json_bytes_v1(forged_live_body)
        ).hexdigest()
        forged_stage5_body = dict(forged_stage5)
        forged_stage5_body.pop("stage_evidence_root")
        forged_stage5["stage_evidence_root"] = sha256(
            ADMISSION.ACTUAL_STAGE_EVIDENCE_ROOT_DOMAIN
            + (5).to_bytes(2, "big")
            + ADMISSION.canonical_json_bytes_v1(forged_stage5_body)
        ).hexdigest()
        assert ADMISSION.validate_actual_stage_5_evidence_surface_v1(
            forged_stage5, fixture["source_commit"]
        ) == forged_stage5
        with pytest.raises(ADMISSION.Q05BActualAdmissionError):
            ADMISSION.validate_actual_stage_5_evidence_v1(
                forged_stage5,
                fixture["source_commit"],
                **strong_stage5_context,
            )
    alternate_controls = json.loads(
        json.dumps(stage5_evidence["evidence"]["actor_completion_rows"])
    )
    alternate_controls[0]["control_root"] = "64" * 32
    coordinated_stage5_tampers = (
        ADMISSION.build_actual_stage_5_evidence_v1(
            fixture["source_commit"],
            alternate_controls,
            stage5_five_sidecars,
            stage5_endpoint_stdout_set,
            stage5_evidence["evidence"]["strict_endpoint_replay_roots"],
            injected,
        ),
        ADMISSION.build_actual_stage_5_evidence_v1(
            fixture["source_commit"],
            stage5_evidence["evidence"]["actor_completion_rows"],
            stage5_five_sidecars,
            stage5_endpoint_stdout_set,
            ["65" * 32, "63" * 32],
            injected,
        ),
    )
    dynamic = ADMISSION.build_dynamic_mount_authority_set_v1(
        fixture["source_commit"],
        stage5_evidence,
        role3_sources["/inputs/python"]["authority_evidence"],
        role3_sources["/inputs/rust"]["authority_evidence"],
        stage5_stdout_tree,
        **strong_stage5_context,
    )
    assert ADMISSION.decode_dynamic_mount_authority_set_v1(
        ADMISSION.canonical_json_bytes_v1(dynamic),
        fixture["source_commit"],
        stage5_evidence,
        **strong_stage5_context,
    ) == dynamic
    checkpoints = [checkpoint_1]
    for checkpoint_id, checkpoint_name in (
        ADMISSION.ACTUAL_FRESH_RUNTIME_CHECKPOINT_REGISTRY[1:]
    ):
        checkpoint_bindings = {
            1: [bindings[1], bindings[2]],
            2: [bindings[3]],
            3: [bindings[1], bindings[2], bindings[3]],
        }[checkpoint_id]
        value = ADMISSION.build_fresh_runtime_checkpoint_v1(
            fixture["source_commit"],
            fixture["artifact"].as_posix(),
            checkpoint_id,
            fixture["decision"]["attempt_id"],
            fixture["boundary"]["boundary_root"],
            issue_record["issue_record_root"],
            consumed["consumed_marker_root"],
            fixture["fresh_runtime"],
            fixture["fresh_runtime"],
            fixture["absence"],
            checkpoint_bindings,
            dynamic,
            stage5_evidence,
            **strong_checkpoint_context,
        )
        assert value["checkpoint_name"] == checkpoint_name
        assert value["canonical_sets_byte_equal"] is True
        assert value["observed_fresh_runtime_evidence"] == fixture["fresh_runtime"]
        assert ADMISSION.decode_fresh_runtime_checkpoint_v1(
            ADMISSION.canonical_json_bytes_v1(value),
            fixture["source_commit"],
            fixture["artifact"].as_posix(),
            checkpoint_id,
            fixture["decision"]["attempt_id"],
            fixture["boundary"]["boundary_root"],
            issue_record["issue_record_root"],
            consumed["consumed_marker_root"],
            fixture["fresh_runtime"],
            fixture["absence"],
            checkpoint_bindings,
            dynamic,
            stage5_evidence,
            **strong_checkpoint_context,
        ) == value
        checkpoints.append(value)
    assert [row["checkpoint_id"] for row in checkpoints] == [1, 2, 3]
    five_sidecars = stage5_evidence["evidence"]["five_sidecars"]
    endpoint_stdout_set = stage5_evidence["evidence"]["endpoint_stdout_set"]
    live = ADMISSION.build_actual_admission_live_marker_replay_v1(
        "PRE_ARTIFACT_ASSEMBLY",
        issue_record,
        consumed,
        work_root_device=fixture["work_identity"]["device"],
        work_root_inode=fixture["work_identity"]["inode"],
        work_root_nlink=fixture["work_identity"]["nlink"],
        work_root_mode=0o700,
        issued_file_device=10,
        issued_file_inode=11,
        issued_file_nlink=2,
        consumed_file_device=10,
        consumed_file_inode=11,
        consumed_file_nlink=2,
        spending_file_device=10,
        spending_file_inode=12,
        spending_file_nlink=1,
    )
    section = ARTIFACT.build_actual_admission_artifact_evidence_v1(
        source_commit=fixture["source_commit"],
        artifact_path=fixture["artifact"].as_posix(),
        commit_a_config_bytes=fixture["commit_a_config"],
        commit_a_config_git_blob_oid="99" * 20,
        prior_stage_evidence_rows=list(fixture["stages"]),
        issue_record=issue_record,
        consumed_marker_evidence=consumed,
        consume_work_root_replay=work_replay,
        consume_git_source_transcript=fixture["git_transcript"],
        consume_artifact_absence_evidence=fixture["absence"],
        fresh_runtime_checkpoint_rows=checkpoints,
        pre_artifact_live_marker_replay=live,
        anti_replay_scope=ADMISSION.ACTUAL_ADMISSION_RUN_LOCAL_ANTI_REPLAY_SCOPE,
        stage_5_evidence=stage5_evidence,
        stage_5_actor_completion_rows=actor_completion_rows,
        stage_5_strict_endpoint_replay_roots=strict_endpoint_replay_roots,
        stage_5_live_marker_replay=stage5_live,
        stage_5_mount_launch_replay_rows=launches,
        five_sidecars=five_sidecars,
        endpoint_stdout_set=endpoint_stdout_set,
    )
    registry = section["root_registry"]
    assert {
        "consume_after_spend_before_endpoints_checkpoint_root",
        "stage6_before_host_launch_checkpoint_root",
        "stage7_before_predicate19_checkpoint_root",
        "stage_5_evidence_root",
        "stage5_mount_binding_root_rows",
        "dynamic_authority_root",
    } <= set(registry)
    assert {
        "consume_fresh_runtime_checkpoint_root",
        "stage5_fresh_runtime_checkpoint_root",
        "stage7_fresh_runtime_checkpoint_root",
    }.isdisjoint(registry)
    assert registry["stage_5_evidence_root"] == stage5_evidence[
        "stage_evidence_root"
    ]
    assert registry["stage5_mount_binding_root_rows"] == [
        [row["role_id"], row["actor_id"], row["mount_binding_root"]]
        for row in checkpoint_1["mount_binding_rows"]
    ]
    assert registry["dynamic_authority_root"] == dynamic[
        "dynamic_authority_root"
    ]
    reconstructed_stage5, _, _ = ARTIFACT._reconstruct_actual_stage_5_evidence_v1(
        source_commit=fixture["source_commit"],
        actor_completion_rows=actor_completion_rows,
        five_sidecars=five_sidecars,
        endpoint_stdout_set=endpoint_stdout_set,
        strict_endpoint_replay_roots=strict_endpoint_replay_roots,
        issue_record=issue_record,
        consumed_marker=consumed,
        consume_work_root_replay=work_replay,
        consume_git_source_transcript=fixture["git_transcript"],
        consume_artifact_absence_evidence=fixture["absence"],
        fresh_runtime_checkpoint_rows=checkpoints,
        stage_5_live_marker_replay=stage5_live,
        stage_5_mount_launch_replay_rows=launches,
    )
    assert ADMISSION.canonical_json_bytes_v1(reconstructed_stage5) == (
        ADMISSION.canonical_json_bytes_v1(stage5_evidence)
    )
    concrete = object.__new__(TOOL.ConcreteQ05BActualBackendV1)
    concrete.source_commit = fixture["source_commit"]
    concrete.completed_stage = 4
    concrete.stage_evidence_rows = {}
    concrete.admission_consumed = True
    concrete.admission_boundary = fixture["boundary"]
    concrete.admission_issue_record = issue_record
    concrete.admission_consumed_marker_evidence = consumed
    concrete.admission_work_root_replay = work_replay
    concrete.admission_consume_git_source_transcript = fixture["git_transcript"]
    concrete.admission_consume_artifact_absence = fixture["absence"]
    concrete.admission_fresh_runtime_checkpoints = {1: checkpoint_1}
    concrete.admission_work_root_descriptor = object()
    concrete.admission_issued_marker_descriptor = object()
    concrete.admission_spending_marker_descriptor = object()
    concrete.admission_consumed_marker_descriptor = object()
    concrete.actor_mount_bindings = {1: bindings[1], 2: bindings[2]}
    concrete.actor_mount_launch_replays = {1: launches[0], 2: launches[1]}

    def concrete_stage5_live(checkpoint: str) -> dict[str, object]:
        assert checkpoint == "STAGE_05_BEFORE_EVIDENCE"
        return stage5_live

    concrete.strict_replay_actual_admission_live_authority_v1 = concrete_stage5_live
    concrete_stage5 = concrete._stage(
        5,
        {
            key: stage5_evidence["evidence"][key]
            for key in ADMISSION.ACTUAL_STAGE_5_BASE_EVIDENCE_KEYS
        },
    )
    assert ADMISSION.canonical_json_bytes_v1(concrete_stage5) == (
        ADMISSION.canonical_json_bytes_v1(reconstructed_stage5)
    )
    concrete.endpoint_tree_identities = (
        role3_sources["/inputs/python"]["authority_evidence"],
        role3_sources["/inputs/rust"]["authority_evidence"],
    )
    concrete.stdout_tree_identity = stage5_stdout_tree
    assert concrete._stage_5_dynamic_mount_authority_set_v1() == dynamic
    legacy_minimal = json.loads(json.dumps(stage5_evidence))
    legacy_minimal["evidence"] = {
        key: legacy_minimal["evidence"][key]
        for key in ADMISSION.ACTUAL_STAGE_5_BASE_EVIDENCE_KEYS
    }
    legacy_body = dict(legacy_minimal)
    legacy_body.pop("stage_evidence_root")
    legacy_minimal["stage_evidence_root"] = sha256(
        ADMISSION.ACTUAL_STAGE_EVIDENCE_ROOT_DOMAIN
        + (5).to_bytes(2, "big")
        + ADMISSION.canonical_json_bytes_v1(legacy_body)
    ).hexdigest()
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.validate_actual_stage_5_evidence_v1(
            legacy_minimal,
            fixture["source_commit"],
            **strong_stage5_context,
        )
    extra_stage5 = json.loads(json.dumps(stage5_evidence))
    extra_stage5["evidence"]["unexpected_stage5_field"] = True
    extra_body = dict(extra_stage5)
    extra_body.pop("stage_evidence_root")
    extra_stage5["stage_evidence_root"] = sha256(
        ADMISSION.ACTUAL_STAGE_EVIDENCE_ROOT_DOMAIN
        + (5).to_bytes(2, "big")
        + ADMISSION.canonical_json_bytes_v1(extra_body)
    ).hexdigest()
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.validate_actual_stage_5_evidence_v1(
            extra_stage5,
            fixture["source_commit"],
            **strong_stage5_context,
        )
    forged_launch_section = json.loads(json.dumps(section))
    forged_launch = forged_launch_section["stage_5_mount_launch_replay_rows"][0]
    forged_launch["source_replay_rows"][0]["source_device"] += 1
    forged_launch_body = dict(forged_launch)
    forged_launch_body.pop("launch_replay_root")
    forged_launch["launch_replay_root"] = sha256(
        ADMISSION.ACTUAL_ACTOR_MOUNT_LAUNCH_REPLAY_ROOT_DOMAIN
        + (1).to_bytes(1, "big")
        + ADMISSION.canonical_json_bytes_v1(forged_launch_body)
    ).hexdigest()
    forged_launch_section["root_registry"][
        "stage5_mount_launch_replay_root_rows"
    ][0][2] = forged_launch["launch_replay_root"]
    forged_section_body = dict(forged_launch_section)
    forged_section_body.pop("actual_admission_evidence_root")
    forged_launch_section["actual_admission_evidence_root"] = sha256(
        ARTIFACT.ACTUAL_ADMISSION_ARTIFACT_EVIDENCE_ROOT_DOMAIN
        + ADMISSION.canonical_json_bytes_v1(forged_section_body)
    ).hexdigest()
    with pytest.raises(ARTIFACT.Q05BActualArtifactError):
        ARTIFACT._replay_actual_admission_artifact_evidence_v1(
            forged_launch_section,
            source_commit=fixture["source_commit"],
            commit_a_config_bytes=fixture["commit_a_config"],
            commit_a_config_git_blob_oid="99" * 20,
            stage_5_actor_completion_rows=actor_completion_rows,
            stage_5_strict_endpoint_replay_roots=strict_endpoint_replay_roots,
            five_sidecars=five_sidecars,
            endpoint_stdout_set=endpoint_stdout_set,
        )
    for launch_tamper in ("payload", "bool_alias"):
        forged = json.loads(json.dumps(launches[1]))
        if launch_tamper == "payload":
            regular = next(
                row
                for row in forged["source_replay_rows"]
                if row["payload_sha256"] is not None
            )
            regular["payload_sha256"] = "00" * 32
        else:
            forged["all_paths_match_prelaunch_held_descriptors"] = 1
        forged_body = dict(forged)
        forged_body.pop("launch_replay_root")
        forged["launch_replay_root"] = sha256(
            ADMISSION.ACTUAL_ACTOR_MOUNT_LAUNCH_REPLAY_ROOT_DOMAIN
            + (2).to_bytes(1, "big")
            + ADMISSION.canonical_json_bytes_v1(forged_body)
        ).hexdigest()
        with pytest.raises(ADMISSION.Q05BActualAdmissionError):
            ADMISSION.validate_actor_mount_launch_replay_v1(
                forged, bindings[2]
            )
    coordinated_stage5_live = json.loads(json.dumps(stage5_live))
    coordinated_stage5_live["work_root_nlink"] += 1
    stage5_live_body = dict(coordinated_stage5_live)
    stage5_live_body.pop("live_marker_replay_root")
    coordinated_stage5_live["live_marker_replay_root"] = sha256(
        ADMISSION.ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_ROOT_DOMAIN
        + ADMISSION.canonical_json_bytes_v1(stage5_live_body)
    ).hexdigest()
    coordinated_injected = json.loads(json.dumps(injected))
    coordinated_injected["actual_admission_work_root_replay"]["nlink"] += 1
    coordinated_injected["actual_admission_live_marker_replay"] = (
        coordinated_stage5_live
    )
    coordinated_full_stage5 = ADMISSION.build_actual_stage_5_evidence_v1(
        fixture["source_commit"],
        actor_completion_rows,
        stage5_five_sidecars,
        stage5_endpoint_stdout_set,
        strict_endpoint_replay_roots,
        coordinated_injected,
    )
    coordinated_dynamic = ADMISSION.build_dynamic_mount_authority_set_v1(
        fixture["source_commit"],
        coordinated_full_stage5,
        dynamic["python_output_tree"],
        dynamic["rust_output_tree"],
        dynamic["stdout_tree"],
        **strong_stage5_context,
    )
    coordinated_checkpoints = [checkpoint_1]
    for checkpoint_id in (2, 3):
        checkpoint_bindings = {
            2: [bindings[3]],
            3: [bindings[1], bindings[2], bindings[3]],
        }[checkpoint_id]
        coordinated_checkpoints.append(
            ADMISSION.build_fresh_runtime_checkpoint_v1(
                fixture["source_commit"],
                fixture["artifact"].as_posix(),
                checkpoint_id,
                fixture["decision"]["attempt_id"],
                fixture["boundary"]["boundary_root"],
                issue_record["issue_record_root"],
                consumed["consumed_marker_root"],
                fixture["fresh_runtime"],
                fixture["fresh_runtime"],
                fixture["absence"],
                checkpoint_bindings,
                coordinated_dynamic,
                coordinated_full_stage5,
                **strong_checkpoint_context,
            )
        )
    coordinated_live_section = json.loads(json.dumps(section))
    coordinated_live_section["consume_work_root_replay"]["nlink"] += 1
    coordinated_live_section["stage_5_live_marker_replay"] = (
        coordinated_stage5_live
    )
    coordinated_live_section["fresh_runtime_checkpoint_rows"][1:] = (
        coordinated_checkpoints[1:]
    )
    coordinated_live_section["root_registry"][
        "stage5_live_marker_replay_root"
    ] = coordinated_stage5_live["live_marker_replay_root"]
    coordinated_live_section["root_registry"][
        "stage6_before_host_launch_checkpoint_root"
    ] = coordinated_checkpoints[1]["checkpoint_root"]
    coordinated_live_section["root_registry"][
        "stage7_before_predicate19_checkpoint_root"
    ] = coordinated_checkpoints[2]["checkpoint_root"]
    coordinated_live_section["root_registry"]["stage_5_evidence_root"] = (
        coordinated_full_stage5["stage_evidence_root"]
    )
    coordinated_live_section["root_registry"]["dynamic_authority_root"] = (
        coordinated_dynamic["dynamic_authority_root"]
    )
    coordinated_preartifact = coordinated_live_section[
        "pre_artifact_live_marker_replay"
    ]
    coordinated_preartifact["work_root_nlink"] += 1
    coordinated_preartifact_body = dict(coordinated_preartifact)
    coordinated_preartifact_body.pop("live_marker_replay_root")
    coordinated_preartifact["live_marker_replay_root"] = sha256(
        ADMISSION.ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_ROOT_DOMAIN
        + ADMISSION.canonical_json_bytes_v1(coordinated_preartifact_body)
    ).hexdigest()
    coordinated_live_section["root_registry"][
        "pre_artifact_live_marker_replay_root"
    ] = coordinated_preartifact["live_marker_replay_root"]
    coordinated_live_body = dict(coordinated_live_section)
    coordinated_live_body.pop("actual_admission_evidence_root")
    coordinated_live_section["actual_admission_evidence_root"] = sha256(
        ARTIFACT.ACTUAL_ADMISSION_ARTIFACT_EVIDENCE_ROOT_DOMAIN
        + ADMISSION.canonical_json_bytes_v1(coordinated_live_body)
    ).hexdigest()
    with pytest.raises(ARTIFACT.Q05BActualArtifactError):
        ARTIFACT._replay_actual_admission_artifact_evidence_v1(
            coordinated_live_section,
            source_commit=fixture["source_commit"],
            commit_a_config_bytes=fixture["commit_a_config"],
            commit_a_config_git_blob_oid="99" * 20,
            stage_5_actor_completion_rows=actor_completion_rows,
            stage_5_strict_endpoint_replay_roots=strict_endpoint_replay_roots,
            five_sidecars=five_sidecars,
            endpoint_stdout_set=endpoint_stdout_set,
        )
    admission_root = section["actual_admission_evidence_root"]
    body = json.loads(json.dumps(section))
    body.pop("actual_admission_evidence_root")
    assert admission_root not in json.dumps(body, sort_keys=True)
    assert [row["stage_id"] for row in section["prior_stage_evidence_rows"]] == [1, 2, 3]
    assert not any(
        forbidden in json.dumps(section, sort_keys=True)
        for forbidden in (
            "artifact_set_root", "final_delivery", "postpublication",
            "stage_08", "stage_09", "stage_10",
        )
    )
    replayed_section = ARTIFACT._replay_actual_admission_artifact_evidence_v1(
        section,
        source_commit=fixture["source_commit"],
        commit_a_config_bytes=fixture["commit_a_config"],
        commit_a_config_git_blob_oid="99" * 20,
        stage_5_actor_completion_rows=actor_completion_rows,
        stage_5_strict_endpoint_replay_roots=strict_endpoint_replay_roots,
        five_sidecars=five_sidecars,
        endpoint_stdout_set=endpoint_stdout_set,
    )
    assert replayed_section == section
    for completion_field, replacement in (
        ("actor_id", "RUST_ENDPOINT"),
        ("command_sha256", "a0" * 32),
        ("mount_registry_sha256", "a1" * 32),
        ("seccomp_evidence", {"forged": True}),
    ):
        forged_completions = json.loads(json.dumps(actor_completion_rows))
        forged_completions[0][completion_field] = replacement
        with pytest.raises(ARTIFACT.Q05BActualArtifactError) as failure:
            ARTIFACT._replay_actual_admission_artifact_evidence_v1(
                section,
                source_commit=fixture["source_commit"],
                commit_a_config_bytes=fixture["commit_a_config"],
                commit_a_config_git_blob_oid="99" * 20,
                stage_5_actor_completion_rows=forged_completions,
                stage_5_strict_endpoint_replay_roots=(
                    strict_endpoint_replay_roots
                ),
                five_sidecars=five_sidecars,
                endpoint_stdout_set=endpoint_stdout_set,
            )
        assert failure.value.code == "REJECT_Q05B_ARTIFACT_ADMISSION"
    for alternate_stage5 in coordinated_stage5_tampers:
        alternate_dynamic = ADMISSION.build_dynamic_mount_authority_set_v1(
            fixture["source_commit"],
            alternate_stage5,
            dynamic["python_output_tree"],
            dynamic["rust_output_tree"],
            dynamic["stdout_tree"],
            **strong_stage5_context,
        )
        coordinated_section = json.loads(json.dumps(section))
        for checkpoint_id, registry_key in (
            (2, "stage6_before_host_launch_checkpoint_root"),
            (3, "stage7_before_predicate19_checkpoint_root"),
        ):
            checkpoint_bindings = {
                2: [bindings[3]],
                3: [bindings[1], bindings[2], bindings[3]],
            }[checkpoint_id]
            coordinated_checkpoint = ADMISSION.build_fresh_runtime_checkpoint_v1(
                fixture["source_commit"],
                fixture["artifact"].as_posix(),
                checkpoint_id,
                fixture["decision"]["attempt_id"],
                fixture["boundary"]["boundary_root"],
                issue_record["issue_record_root"],
                consumed["consumed_marker_root"],
                fixture["fresh_runtime"],
                fixture["fresh_runtime"],
                fixture["absence"],
                checkpoint_bindings,
                alternate_dynamic,
                alternate_stage5,
                **strong_checkpoint_context,
            )
            coordinated_section["fresh_runtime_checkpoint_rows"][
                checkpoint_id - 1
            ] = coordinated_checkpoint
            coordinated_section["root_registry"][registry_key] = (
                coordinated_checkpoint["checkpoint_root"]
            )
        coordinated_section["root_registry"]["stage_5_evidence_root"] = (
            alternate_stage5["stage_evidence_root"]
        )
        coordinated_section["root_registry"]["dynamic_authority_root"] = (
            alternate_dynamic["dynamic_authority_root"]
        )
        coordinated_body = json.loads(json.dumps(coordinated_section))
        coordinated_body.pop("actual_admission_evidence_root")
        coordinated_section["actual_admission_evidence_root"] = sha256(
            ARTIFACT.ACTUAL_ADMISSION_ARTIFACT_EVIDENCE_ROOT_DOMAIN
            + ADMISSION.canonical_json_bytes_v1(coordinated_body)
        ).hexdigest()
        assert ARTIFACT._replay_actual_admission_artifact_evidence_v1(
            coordinated_section,
            source_commit=fixture["source_commit"],
            commit_a_config_bytes=fixture["commit_a_config"],
            commit_a_config_git_blob_oid="99" * 20,
            stage_5_actor_completion_rows=alternate_stage5["evidence"][
                "actor_completion_rows"
            ],
            stage_5_strict_endpoint_replay_roots=alternate_stage5[
                "evidence"
            ]["strict_endpoint_replay_roots"],
            five_sidecars=five_sidecars,
            endpoint_stdout_set=endpoint_stdout_set,
        ) == coordinated_section
        with pytest.raises(ARTIFACT.Q05BActualArtifactError):
            ARTIFACT._replay_actual_admission_artifact_evidence_v1(
                coordinated_section,
                source_commit=fixture["source_commit"],
                commit_a_config_bytes=fixture["commit_a_config"],
                commit_a_config_git_blob_oid="99" * 20,
                stage_5_actor_completion_rows=actor_completion_rows,
                stage_5_strict_endpoint_replay_roots=(
                    strict_endpoint_replay_roots
                ),
                five_sidecars=five_sidecars,
                endpoint_stdout_set=endpoint_stdout_set,
            )
    bool_alias_section = json.loads(json.dumps(section))
    bool_alias_section["fresh_runtime_checkpoint_rows"][0]["checkpoint_id"] = True
    with pytest.raises(ARTIFACT.Q05BActualArtifactError):
        ARTIFACT._replay_actual_admission_artifact_evidence_v1(
            bool_alias_section,
            source_commit=fixture["source_commit"],
            commit_a_config_bytes=fixture["commit_a_config"],
            commit_a_config_git_blob_oid="99" * 20,
            stage_5_actor_completion_rows=actor_completion_rows,
            stage_5_strict_endpoint_replay_roots=strict_endpoint_replay_roots,
            five_sidecars=five_sidecars,
            endpoint_stdout_set=endpoint_stdout_set,
        )
    root_alias_section = json.loads(json.dumps(section))
    root_alias_section["root_registry"]["boundary_root"] = "00" * 32
    with pytest.raises(ARTIFACT.Q05BActualArtifactError):
        ARTIFACT._replay_actual_admission_artifact_evidence_v1(
            root_alias_section,
            source_commit=fixture["source_commit"],
            commit_a_config_bytes=fixture["commit_a_config"],
            commit_a_config_git_blob_oid="99" * 20,
            stage_5_actor_completion_rows=actor_completion_rows,
            stage_5_strict_endpoint_replay_roots=strict_endpoint_replay_roots,
            five_sidecars=five_sidecars,
            endpoint_stdout_set=endpoint_stdout_set,
        )
    alias = json.loads(json.dumps(checkpoints[0]))
    alias["canonical_sets_byte_equal"] = 1
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.decode_fresh_runtime_checkpoint_v1(
            ADMISSION.canonical_json_bytes_v1(alias),
            fixture["source_commit"],
            fixture["artifact"].as_posix(),
            1,
            fixture["decision"]["attempt_id"],
            fixture["boundary"]["boundary_root"],
            issue_record["issue_record_root"],
            consumed["consumed_marker_root"],
            fixture["fresh_runtime"],
            fixture["absence"],
            [bindings[1], bindings[2]],
            None,
            None,
        )
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.build_fresh_runtime_checkpoint_v1(
            fixture["source_commit"],
            fixture["artifact"].as_posix(),
            True,
            fixture["decision"]["attempt_id"],
            fixture["boundary"]["boundary_root"],
            issue_record["issue_record_root"],
            consumed["consumed_marker_root"],
            fixture["fresh_runtime"],
            fixture["fresh_runtime"],
            fixture["absence"],
            [bindings[1], bindings[2]],
            None,
            None,
        )
    bool_alias = json.loads(json.dumps(bindings[1]))
    bool_alias["source_rows"][0]["writable"] = 1
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.validate_actor_mount_binding_v1(bool_alias)
    root_alias = json.loads(json.dumps(bindings[1]))
    root_alias["mount_binding_root"] = "00" * 32
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.validate_actor_mount_binding_v1(root_alias)
    first = bindings[1]["source_rows"][0]

    def rebuilt_first_source(
        *,
        role_id=1,
        destination="/control",
        authority_kind=first["authority_kind"],
        authority_label=first["authority_label"],
        authority_evidence=first["authority_evidence"],
    ):
        forged = ADMISSION.build_actor_mount_source_row_v1(
            role_id,
            destination,
            first["source"],
            first["writable"],
            first["source_type"],
            first["source_device"],
            first["source_inode"],
            first["source_nlink"],
            first["source_uid"],
            first["source_gid"],
            first["source_mode"],
            first["source_size"],
            authority_kind,
            authority_label,
            authority_evidence,
        )
        return ADMISSION.build_actor_mount_binding_v1(
            bindings[1]["exact_command"],
            bindings[1]["command_mount_registry"],
            [forged, *bindings[1]["source_rows"][1:]],
            bindings[1]["seccomp_row"],
        )

    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        rebuilt_first_source(
            authority_kind="FORGED_AUTHORITY",
            authority_label="forged/authority",
            authority_evidence={"opaque": True},
        )
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        rebuilt_first_source(role_id=2)
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        rebuilt_first_source(destination="/output")
    wrong_schema = dict(first["authority_evidence"])
    wrong_schema["schema_version"] = "forged/1"
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        rebuilt_first_source(authority_evidence=wrong_schema)
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        rebuilt_first_source(authority_label="PYTHON_ENDPOINT/forged")
    wrong_owner = dict(first["authority_evidence"])
    wrong_owner["uid"] += 1
    wrong_owner_body = dict(wrong_owner)
    wrong_owner_body.pop("directory_identity_root")
    wrong_owner["directory_identity_root"] = sha256(
        ADMISSION.ACTUAL_PRELAUNCH_WRITABLE_DIRECTORY_ROOT_DOMAIN
        + (1).to_bytes(1, "big")
        + ADMISSION.canonical_json_bytes_v1(wrong_owner_body)
    ).hexdigest()
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        rebuilt_first_source(authority_evidence=wrong_owner)

    def row_with_evidence(row, evidence):
        return ADMISSION.build_actor_mount_source_row_v1(
            row["role_id"],
            row["destination"],
            row["source"],
            row["writable"],
            row["source_type"],
            row["source_device"],
            row["source_inode"],
            row["source_nlink"],
            row["source_uid"],
            row["source_gid"],
            row["source_mode"],
            row["source_size"],
            row["authority_kind"],
            row["authority_label"],
            evidence,
        )

    def checkpoint_one(role1, role2):
        return ADMISSION.build_fresh_runtime_checkpoint_v1(
            fixture["source_commit"],
            fixture["artifact"].as_posix(),
            1,
            fixture["decision"]["attempt_id"],
            fixture["boundary"]["boundary_root"],
            "81" * 32,
            "82" * 32,
            fixture["fresh_runtime"],
            fixture["fresh_runtime"],
            fixture["absence"],
            [role1, role2],
        )

    snapshot_row = bindings[1]["source_rows"][2]
    changed_snapshot = json.loads(
        json.dumps(snapshot_row["authority_evidence"])
    )
    changed_snapshot["file_rows"][0][10] = "91" * 32
    snapshot_body = dict(changed_snapshot)
    snapshot_body.pop("manifest_sha256")
    changed_snapshot["manifest_sha256"] = sha256(
        TOOL._canonical_json_bytes(snapshot_body)
    ).hexdigest()
    changed_role1_sources = list(bindings[1]["source_rows"])
    changed_role1_sources[2] = row_with_evidence(
        snapshot_row, changed_snapshot
    )
    changed_role1 = ADMISSION.build_actor_mount_binding_v1(
        bindings[1]["exact_command"],
        bindings[1]["command_mount_registry"],
        changed_role1_sources,
        bindings[1]["seccomp_row"],
    )
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        checkpoint_one(changed_role1, bindings[2])

    changed_seccomp = json.loads(
        json.dumps(bindings[1]["seccomp_row"]["authority_evidence"])
    )
    changed_seccomp["payload_sha256"] = "92" * 32
    seccomp_body = dict(changed_seccomp)
    seccomp_body.pop("manifest_sha256")
    changed_seccomp["manifest_sha256"] = sha256(
        TOOL._canonical_json_bytes(seccomp_body)
    ).hexdigest()
    changed_role1_seccomp = ADMISSION.build_actor_mount_binding_v1(
        bindings[1]["exact_command"],
        bindings[1]["command_mount_registry"],
        bindings[1]["source_rows"],
        row_with_evidence(bindings[1]["seccomp_row"], changed_seccomp),
    )
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        checkpoint_one(changed_role1_seccomp, bindings[2])

    binary_row = bindings[2]["source_rows"][2]
    changed_binary = dict(binary_row["authority_evidence"])
    changed_binary["sha256"] = "93" * 32
    changed_role2_sources = list(bindings[2]["source_rows"])
    changed_role2_sources[2] = row_with_evidence(
        binary_row, changed_binary
    )
    changed_role2 = ADMISSION.build_actor_mount_binding_v1(
        bindings[2]["exact_command"],
        bindings[2]["command_mount_registry"],
        changed_role2_sources,
        bindings[2]["seccomp_row"],
    )
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        checkpoint_one(bindings[1], changed_role2)

    changed_dynamic = json.loads(json.dumps(dynamic))
    changed_dynamic["python_output_tree"]["file_rows"][0][10] = "85" * 32
    changed_python_body = dict(changed_dynamic["python_output_tree"])
    changed_python_body.pop("manifest_sha256")
    changed_dynamic["python_output_tree"]["manifest_sha256"] = sha256(
        ADMISSION.canonical_json_bytes_v1(changed_python_body)
    ).hexdigest()
    changed_dynamic_body = dict(changed_dynamic)
    changed_dynamic_body.pop("dynamic_authority_root")
    changed_dynamic["dynamic_authority_root"] = sha256(
        ADMISSION.ACTUAL_DYNAMIC_MOUNT_AUTHORITY_SET_ROOT_DOMAIN
        + ADMISSION.canonical_json_bytes_v1(changed_dynamic_body)
    ).hexdigest()
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.build_fresh_runtime_checkpoint_v1(
            fixture["source_commit"],
            fixture["artifact"].as_posix(),
            2,
            fixture["decision"]["attempt_id"],
            fixture["boundary"]["boundary_root"],
            "81" * 32,
            "82" * 32,
            fixture["fresh_runtime"],
            fixture["fresh_runtime"],
            fixture["absence"],
            [bindings[3]],
            changed_dynamic,
            stage5_evidence,
            **strong_checkpoint_context,
        )
    forged_root_dynamic = json.loads(json.dumps(dynamic))
    forged_root_dynamic["stage_5_evidence_root"] = "84" * 32
    forged_root_body = dict(forged_root_dynamic)
    forged_root_body.pop("dynamic_authority_root")
    forged_root_dynamic["dynamic_authority_root"] = sha256(
        ADMISSION.ACTUAL_DYNAMIC_MOUNT_AUTHORITY_SET_ROOT_DOMAIN
        + ADMISSION.canonical_json_bytes_v1(forged_root_body)
    ).hexdigest()
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.build_fresh_runtime_checkpoint_v1(
            fixture["source_commit"],
            fixture["artifact"].as_posix(),
            2,
            fixture["decision"]["attempt_id"],
            fixture["boundary"]["boundary_root"],
            "81" * 32,
            "82" * 32,
            fixture["fresh_runtime"],
            fixture["fresh_runtime"],
            fixture["absence"],
            [bindings[3]],
            forged_root_dynamic,
            stage5_evidence,
            **strong_checkpoint_context,
        )
    stage5_bool_alias = json.loads(json.dumps(stage5_evidence))
    stage5_bool_alias["q1_authority"]["certificate_active"] = 0
    alias_body = dict(stage5_bool_alias)
    alias_body.pop("stage_evidence_root")
    stage5_bool_alias["stage_evidence_root"] = sha256(
        ADMISSION.ACTUAL_STAGE_EVIDENCE_ROOT_DOMAIN
        + (5).to_bytes(2, "big")
        + ADMISSION.canonical_json_bytes_v1(alias_body)
    ).hexdigest()
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.build_dynamic_mount_authority_set_v1(
            fixture["source_commit"],
            stage5_bool_alias,
            dynamic["python_output_tree"],
            dynamic["rust_output_tree"],
            dynamic["stdout_tree"],
            **strong_stage5_context,
        )
    for alternate_stage5 in coordinated_stage5_tampers:
        alternate_dynamic = ADMISSION.build_dynamic_mount_authority_set_v1(
            fixture["source_commit"],
            alternate_stage5,
            dynamic["python_output_tree"],
            dynamic["rust_output_tree"],
            dynamic["stdout_tree"],
            **strong_stage5_context,
        )
        alternate_checkpoint = ADMISSION.build_fresh_runtime_checkpoint_v1(
            fixture["source_commit"],
            fixture["artifact"].as_posix(),
            3,
            fixture["decision"]["attempt_id"],
            fixture["boundary"]["boundary_root"],
            "81" * 32,
            "82" * 32,
            fixture["fresh_runtime"],
            fixture["fresh_runtime"],
            fixture["absence"],
            [bindings[1], bindings[2], bindings[3]],
            alternate_dynamic,
            alternate_stage5,
            **strong_checkpoint_context,
        )
        with pytest.raises(ADMISSION.Q05BActualAdmissionError):
            ADMISSION.decode_fresh_runtime_checkpoint_v1(
                ADMISSION.canonical_json_bytes_v1(alternate_checkpoint),
                fixture["source_commit"],
                fixture["artifact"].as_posix(),
                3,
                fixture["decision"]["attempt_id"],
                fixture["boundary"]["boundary_root"],
                "81" * 32,
                "82" * 32,
                fixture["fresh_runtime"],
                fixture["absence"],
                [bindings[1], bindings[2], bindings[3]],
                dynamic,
                stage5_evidence,
                **strong_checkpoint_context,
            )


def test_pure_admission_rejects_forged_shallow_bundle_and_accepts_pretty_config(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_actual_admission_fixture(tmp_path)
    bundle = fixture["bundle"]
    forged = {
        "source_commit": fixture["source_commit"],
        "artifact_path": fixture["artifact"].as_posix(),
        "commit_a_config_length": len(fixture["commit_a_config"]),
        "commit_a_config_sha256": sha256(fixture["commit_a_config"]).hexdigest(),
        "bundle_root": "77" * 32,
        "prior_stage_root_rows": bundle["prior_stage_root_rows"],
        "work_root_identity": fixture["work_identity"],
    }
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.build_actual_admission_decision_v1(
            fixture["source_commit"],
            fixture["commit_a_config"],
            fixture["artifact"].as_posix(),
            b"F" * 32,
            forged,
        )

    pretty = (
        json.dumps(
            json.loads(fixture["commit_a_config"]),
            ensure_ascii=True,
            indent=2,
        )
        + "\n"
    ).encode("ascii")
    preimages = json.loads(
        json.dumps(
            [row["preimage"] for row in bundle["ordered_precondition_rows"]]
        )
    )
    preimages[1]["commit_a_config_hex"] = pretty.hex()
    preimages[1]["runtime_loaded_config_hex"] = pretty.hex()
    preimages[1]["config_length"] = len(pretty)
    preimages[1]["config_sha256"] = sha256(pretty).hexdigest()
    pretty_bundle = ADMISSION.build_actual_precondition_bundle_v1(
        fixture["source_commit"],
        pretty,
        fixture["artifact"].as_posix(),
        fixture["work_identity"],
        bundle["prior_stage_root_rows"],
        preimages,
    )
    assert pretty_bundle["commit_a_config_sha256"] == sha256(pretty).hexdigest()
    assert pretty_bundle["precondition_count"] == 12


def test_commit_a_config_full_policy_rejects_top_delete_and_nested_tamper() -> None:
    config = json.loads(_synthetic_commit_a_actual_config_bytes())
    mutations = []
    extra = json.loads(json.dumps(config))
    extra["unexpected"] = True
    mutations.append(extra)
    deleted = json.loads(json.dumps(config))
    del deleted["docker"]
    mutations.append(deleted)
    images = json.loads(json.dumps(config))
    images["images"] = {}
    mutations.append(images)
    docker_bool_alias = json.loads(json.dumps(config))
    docker_bool_alias["docker"]["pids_limit"] = True
    mutations.append(docker_bool_alias)
    for value in mutations:
        with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
            TOOL._validate_commit_a_actual_config_bytes_v1(
                TOOL._canonical_json_bytes(value)
            )
        assert failure.value.code == TOOL.FAIL_ACTUAL_ADMISSION


def test_pure_admission_static_policy_and_preimage9_are_self_contained(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_actual_admission_fixture(tmp_path)
    config = json.loads(fixture["commit_a_config"])
    config["docker"]["network"] = "bridge"
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.build_actual_precondition_bundle_v1(
            fixture["source_commit"],
            ADMISSION.canonical_json_bytes_v1(config),
            fixture["artifact"].as_posix(),
            fixture["work_identity"],
            fixture["bundle"]["prior_stage_root_rows"],
            [
                row["preimage"]
                for row in fixture["bundle"]["ordered_precondition_rows"]
            ],
        )
    preimages = json.loads(
        json.dumps(
            [
                row["preimage"]
                for row in fixture["bundle"]["ordered_precondition_rows"]
            ]
        )
    )
    preimages[8]["command_mount_resource_policy_sha256"] = "00" * 32
    with pytest.raises(ADMISSION.Q05BActualAdmissionError):
        ADMISSION.build_actual_precondition_bundle_v1(
            fixture["source_commit"],
            fixture["commit_a_config"],
            fixture["artifact"].as_posix(),
            fixture["work_identity"],
            fixture["bundle"]["prior_stage_root_rows"],
            preimages,
        )
    assert (
        ADMISSION.command_mount_resource_policy_root_v1(
            fixture["commit_a_config"]
        )
        == ADMISSION.EXPECTED_COMMAND_MOUNT_RESOURCE_POLICY_ROOT
    )


def test_actual_admission_tamper_and_single_attempt_latch_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_actual_admission_fixture(tmp_path)
    bundle_mutations = []
    bool_pass = json.loads(json.dumps(fixture["bundle"]))
    bool_pass["ordered_precondition_rows"][0]["passed"] = 1
    bundle_mutations.append(bool_pass)
    bool_id = json.loads(json.dumps(fixture["bundle"]))
    bool_id["ordered_precondition_rows"][0]["predicate_id"] = True
    bundle_mutations.append(bool_id)
    arbitrary_root = json.loads(json.dumps(fixture["bundle"]))
    arbitrary_root["ordered_precondition_rows"][0]["evidence_root"] = "00" * 32
    bundle_mutations.append(arbitrary_root)
    for value in bundle_mutations:
        with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
            TOOL.decode_actual_precondition_bundle_v1(
                TOOL._canonical_json_bytes(value),
                fixture["source_commit"],
                fixture["commit_a_config"],
                fixture["artifact"],
                fixture["work_identity"],
                fixture["stages"],
                fixture["fresh"],
                fixture["git_transcript"],
                fixture["absence"],
                fixture["fresh_runtime"],
            )
        assert failure.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    bad_decision = json.loads(json.dumps(fixture["decision"]))
    bad_decision["precondition_count"] = True
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.decode_actual_admission_decision_v1(
            TOOL._canonical_json_bytes(bad_decision),
            fixture["commit_a_config"],
            fixture["source_commit"],
            fixture["artifact"],
            fixture["bundle"],
        )
    assert failure.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    second = TOOL.build_actual_admission_decision_v1(
        fixture["source_commit"],
        fixture["commit_a_config"],
        fixture["artifact"],
        b"M" * 32,
        fixture["bundle"],
    )
    assert second["attempt_id"] != fixture["decision"]["attempt_id"]
    latch = TOOL.ActualAdmissionAttemptLatchV1(fixture["boundary"])
    assert latch.consume_once(fixture["boundary"]) == fixture["decision"]["attempt_id"]
    with pytest.raises(TOOL.Q05BDualSupervisorError) as second_consume:
        latch.consume_once(fixture["boundary"])
    assert second_consume.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    concurrent_latch = TOOL.ActualAdmissionAttemptLatchV1(fixture["boundary"])
    barrier = threading.Barrier(3)
    outcomes = []

    def consume_concurrently():
        barrier.wait()
        try:
            outcomes.append(("ok", concurrent_latch.consume_once(fixture["boundary"])))
        except TOOL.Q05BDualSupervisorError as error:
            outcomes.append(("error", error.code))

    workers = [threading.Thread(target=consume_concurrently) for _ in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert sorted(kind for kind, _ in outcomes) == ["error", "ok"]
    assert [value for kind, value in outcomes if kind == "error"] == [
        TOOL.FAIL_ACTUAL_ADMISSION
    ]
    current_config = TOOL._validate_commit_a_actual_config_bytes_v1(
        (ROOT / TOOL.CONFIG_RELATIVE_PATH).read_bytes()
    )
    assert current_config["engineering_status"] == (
        TOOL.COMMIT_A_ACTUAL_ENGINEERING_STATUS
    )
    assert current_config["actual_preconditions"] == (
        TOOL.COMMIT_A_ACTUAL_PRECONDITIONS_V1
    )


def test_actual_admission_marker_issue_consume_preserves_inode_and_spends(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_actual_admission_fixture(tmp_path)
    issue_record, root_descriptor, issued_descriptor = (
        TOOL.issue_actual_admission_marker_v1(
        Path(fixture["work_identity"]["absolute_path"]),
        fixture["work_identity"],
        fixture["boundary"],
        )
    )
    spending_descriptor = None
    consumed_descriptor = None
    try:
        assert (
            TOOL.fcntl.fcntl(issued_descriptor, TOOL.fcntl.F_GETFL)
            & os.O_ACCMODE
        ) == os.O_RDONLY
        replayed_record, replayed_boundary = (
            TOOL.validate_actual_admission_issue_record_v1(issue_record)
        )
        assert replayed_record == issue_record
        assert replayed_boundary == fixture["boundary"]
        pure_record, pure_boundary = (
            ADMISSION.validate_actual_admission_issue_record_v1(issue_record)
        )
        assert ADMISSION.canonical_json_bytes_v1(pure_record) == (
            TOOL._canonical_json_bytes(replayed_record)
        )
        assert ADMISSION.canonical_json_bytes_v1(pure_boundary) == (
            TOOL._canonical_json_bytes(replayed_boundary)
        )
        marker = issue_record["issued_marker_evidence"]
        pure_issued = ADMISSION.build_actual_admission_issued_marker_evidence_v1(
            fixture["boundary"]["attempt_id"],
            fixture["boundary"]["boundary_root"],
            ADMISSION.canonical_json_bytes_v1(fixture["boundary"]),
            file_device=marker["file_device"],
            file_inode=marker["file_inode"],
            file_nlink=marker["file_nlink"],
            file_mode=marker["file_mode"],
            work_root_device=marker["work_root_device"],
            work_root_inode=marker["work_root_inode"],
            work_root_mode=marker["work_root_mode"],
        )
        assert ADMISSION.canonical_json_bytes_v1(pure_issued) == (
            TOOL._canonical_json_bytes(marker)
        )
        assert TOOL.validate_actual_admission_issued_marker_evidence_v1(
            marker,
            ADMISSION.canonical_json_bytes_v1(fixture["boundary"]),
        ) == pure_issued
        scope_alias = json.loads(json.dumps(issue_record))
        scope_alias["anti_replay_scope"]["cli_accepts_boundary_or_nonce"] = 0
        scope_alias_body = dict(scope_alias)
        scope_alias_body.pop("issue_record_root")
        scope_alias["issue_record_root"] = sha256(
            TOOL.ACTUAL_ADMISSION_ISSUE_RECORD_ROOT_DOMAIN
            + TOOL._canonical_json_bytes(scope_alias_body)
        ).hexdigest()
        with pytest.raises(TOOL.Q05BDualSupervisorError) as scope_failure:
            TOOL.validate_actual_admission_issue_record_v1(scope_alias)
        assert scope_failure.value.code == TOOL.FAIL_ACTUAL_ADMISSION
        issued = Path(fixture["work_identity"]["absolute_path"]) / marker[
            "issued_relative_path"
        ]
        consumed = Path(fixture["work_identity"]["absolute_path"]) / marker[
            "consumed_relative_path"
        ]
        issued_status = issued.lstat()
        assert stat.S_IMODE(issued_status.st_mode) == 0o444
        assert issued_status.st_nlink == 1
        (
            consumed_evidence,
            spending_descriptor,
            consumed_descriptor,
        ) = TOOL.consume_actual_admission_marker_v1(
            root_descriptor, issued_descriptor, issue_record
        )
        assert (
            TOOL.fcntl.fcntl(spending_descriptor, TOOL.fcntl.F_GETFL)
            & os.O_ACCMODE
        ) == os.O_RDONLY
        assert (
            TOOL.fcntl.fcntl(consumed_descriptor, TOOL.fcntl.F_GETFL)
            & os.O_ACCMODE
        ) == os.O_RDONLY
        consumed_status = consumed.lstat()
        issued_final_status = issued.lstat()
        assert consumed_status.st_ino == issued_status.st_ino
        assert consumed_status.st_dev == issued_status.st_dev
        assert issued_final_status.st_ino == issued_status.st_ino
        assert consumed_status.st_nlink == 2
        assert issued_final_status.st_nlink == 2
        assert consumed_evidence["file_inode"] == issued_status.st_ino
        assert consumed_evidence["spent_before_preflight"] is True
        assert TOOL.validate_actual_admission_consumed_marker_evidence_v1(
            consumed_evidence,
            issue_record,
        ) == consumed_evidence
        pure_consumed = (
            ADMISSION.validate_actual_admission_consumed_marker_evidence_v1(
                consumed_evidence, issue_record
            )
        )
        assert ADMISSION.canonical_json_bytes_v1(pure_consumed) == (
            TOOL._canonical_json_bytes(consumed_evidence)
        )
        pure_spending = ADMISSION.build_actual_admission_spending_intent_v1(
            issue_record
        )
        assert ADMISSION.canonical_json_bytes_v1(pure_spending).hex() == (
            consumed_evidence["spending_intent_hex"]
        )
        assert TOOL.validate_actual_admission_spending_intent_v1(
            pure_spending, issue_record
        ) == pure_spending
        checkpoint = "TEST_TOOL_PURE_CANONICAL_EQUALITY"
        tool_live = TOOL.replay_live_actual_admission_markers_v1(
            root_descriptor,
            issued_descriptor,
            spending_descriptor,
            consumed_descriptor,
            Path(fixture["work_identity"]["absolute_path"]),
            fixture["work_identity"],
            issue_record,
            consumed_evidence,
            checkpoint,
        )
        pure_live = ADMISSION.build_actual_admission_live_marker_replay_v1(
            checkpoint,
            issue_record,
            consumed_evidence,
            work_root_device=tool_live["work_root_device"],
            work_root_inode=tool_live["work_root_inode"],
            work_root_nlink=tool_live["work_root_nlink"],
            work_root_mode=tool_live["work_root_mode"],
            issued_file_device=tool_live["issued_file_device"],
            issued_file_inode=tool_live["issued_file_inode"],
            issued_file_nlink=tool_live["issued_file_nlink"],
            consumed_file_device=tool_live["consumed_file_device"],
            consumed_file_inode=tool_live["consumed_file_inode"],
            consumed_file_nlink=tool_live["consumed_file_nlink"],
            spending_file_device=tool_live["spending_file_device"],
            spending_file_inode=tool_live["spending_file_inode"],
            spending_file_nlink=tool_live["spending_file_nlink"],
        )
        assert ADMISSION.canonical_json_bytes_v1(pure_live) == (
            TOOL._canonical_json_bytes(tool_live)
        )
        bool_alias = dict(consumed_evidence)
        bool_alias["spent_before_preflight"] = 1
        bool_alias_body = dict(bool_alias)
        bool_alias_body.pop("consumed_marker_root")
        bool_alias["consumed_marker_root"] = sha256(
            TOOL.ACTUAL_ADMISSION_CONSUMED_MARKER_ROOT_DOMAIN
            + TOOL._canonical_json_bytes(bool_alias_body)
        ).hexdigest()
        with pytest.raises(TOOL.Q05BDualSupervisorError) as bool_failure:
            TOOL.validate_actual_admission_consumed_marker_evidence_v1(
                bool_alias,
                issue_record,
            )
        assert bool_failure.value.code == TOOL.FAIL_ACTUAL_ADMISSION
        pure_bool_alias = dict(consumed_evidence)
        pure_bool_alias["file_nlink"] = True
        with pytest.raises(ADMISSION.Q05BActualAdmissionError):
            ADMISSION.validate_actual_admission_consumed_marker_evidence_v1(
                pure_bool_alias, issue_record
            )
        pure_root_tamper = dict(issue_record)
        pure_root_tamper["issue_record_root"] = "00" * 32
        with pytest.raises(ADMISSION.Q05BActualAdmissionError):
            ADMISSION.validate_actual_admission_issue_record_v1(
                pure_root_tamper
            )
        with pytest.raises(TOOL.Q05BDualSupervisorError) as reused:
            TOOL.consume_actual_admission_marker_v1(
                root_descriptor,
                issued_descriptor,
                issue_record,
            )
        assert reused.value.code == TOOL.FAIL_ACTUAL_ADMISSION
        consumed.unlink()
        os.fsync(root_descriptor)
        with pytest.raises(TOOL.Q05BDualSupervisorError) as reissued:
            TOOL.issue_actual_admission_marker_v1(
                Path(fixture["work_identity"]["absolute_path"]),
                fixture["work_identity"],
                fixture["boundary"],
            )
        assert reissued.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    finally:
        if issued_descriptor is not None:
            os.close(issued_descriptor)
        if spending_descriptor is not None:
            os.close(spending_descriptor)
        if consumed_descriptor is not None:
            os.close(consumed_descriptor)
        os.close(root_descriptor)


def test_actual_admission_issue_rejects_live_work_root_nlink_drift(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_actual_admission_fixture(tmp_path)
    work = Path(fixture["work_identity"]["absolute_path"])
    (work / "late-subdirectory").mkdir(mode=0o700)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.issue_actual_admission_marker_v1(
            work,
            fixture["work_identity"],
            fixture["boundary"],
        )
    assert failure.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    assert not any(path.name.endswith(".issued") for path in work.iterdir())


def _replace_admission_marker_at_v1(
    root_descriptor: int,
    relative_name: str,
    payload: bytes,
) -> None:
    os.unlink(relative_name, dir_fd=root_descriptor)
    descriptor = os.open(
        relative_name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=root_descriptor,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(root_descriptor)


def test_actual_admission_consume_rejects_issued_path_replacement(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_actual_admission_fixture(tmp_path)
    work = Path(fixture["work_identity"]["absolute_path"])
    issue_record, root_descriptor, issued_descriptor = (
        TOOL.issue_actual_admission_marker_v1(
            work,
            fixture["work_identity"],
            fixture["boundary"],
        )
    )
    try:
        marker = issue_record["issued_marker_evidence"]
        _replace_admission_marker_at_v1(
            root_descriptor,
            marker["issued_relative_path"],
            bytes.fromhex(issue_record["pure_boundary_hex"]),
        )
        with pytest.raises(TOOL.Q05BDualSupervisorError) as replaced:
            TOOL.consume_actual_admission_marker_v1(
                root_descriptor,
                issued_descriptor,
                issue_record,
            )
        assert replaced.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    finally:
        os.close(issued_descriptor)
        os.close(root_descriptor)


def test_actual_admission_spending_tombstone_survives_link_eio_and_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _synthetic_actual_admission_fixture(tmp_path)
    work = Path(fixture["work_identity"]["absolute_path"])
    issue_record, root_descriptor, issued_descriptor = (
        TOOL.issue_actual_admission_marker_v1(
        work,
        fixture["work_identity"],
        fixture["boundary"],
        )
    )
    real_link = TOOL.os.link
    monkeypatch.setattr(
        TOOL.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError(5, "injected link EIO")
        ),
    )
    with pytest.raises(TOOL.Q05BDualSupervisorError) as first:
        TOOL.consume_actual_admission_marker_v1(
            root_descriptor,
            issued_descriptor,
            issue_record,
        )
    assert first.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    marker = issue_record["issued_marker_evidence"]
    spending = work / marker["spending_relative_path"]
    assert spending.is_file()
    assert stat.S_IMODE(spending.lstat().st_mode) == 0o444
    monkeypatch.setattr(TOOL.os, "link", real_link)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as same_process_retry:
        TOOL.consume_actual_admission_marker_v1(
            root_descriptor,
            issued_descriptor,
            issue_record,
        )
    assert same_process_retry.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    os.close(root_descriptor)
    reopened = TOOL._open_anchored_admission_work_root_v1(
        work,
        fixture["work_identity"],
    )
    try:
        with pytest.raises(TOOL.Q05BDualSupervisorError) as reopened_retry:
            TOOL.consume_actual_admission_marker_v1(
                reopened,
                issued_descriptor,
                issue_record,
            )
        assert reopened_retry.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    finally:
        os.close(reopened)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as reissue:
        TOOL.issue_actual_admission_marker_v1(
            work,
            fixture["work_identity"],
            fixture["boundary"],
        )
    assert reissue.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    os.close(issued_descriptor)


def test_actual_admission_live_replay_rejects_consumed_path_removal(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_actual_admission_fixture(tmp_path)
    work = Path(fixture["work_identity"]["absolute_path"])
    issue_record, root_descriptor, issued_descriptor = (
        TOOL.issue_actual_admission_marker_v1(
        work,
        fixture["work_identity"],
        fixture["boundary"],
        )
    )
    consumed, spending_descriptor, consumed_descriptor = (
        TOOL.consume_actual_admission_marker_v1(
            root_descriptor,
            issued_descriptor,
            issue_record,
        )
    )
    try:
        replay = TOOL.replay_live_actual_admission_markers_v1(
            root_descriptor,
            issued_descriptor,
            spending_descriptor,
            consumed_descriptor,
            work,
            fixture["work_identity"],
            issue_record,
            consumed,
            "TEST_LIVE_BEFORE_REMOVAL",
        )
        assert replay["issued_consumed_same_inode"] is True
        consumed_path = work / issue_record["issued_marker_evidence"][
            "consumed_relative_path"
        ]
        consumed_path.unlink()
        os.fsync(root_descriptor)
        with pytest.raises(TOOL.Q05BDualSupervisorError) as removed:
            TOOL.replay_live_actual_admission_markers_v1(
                root_descriptor,
                issued_descriptor,
                spending_descriptor,
                consumed_descriptor,
                work,
                fixture["work_identity"],
                issue_record,
                consumed,
                "TEST_LIVE_AFTER_REMOVAL",
            )
        assert removed.value.code == TOOL.FAIL_ACTUAL_ADMISSION
        with pytest.raises(TOOL.Q05BDualSupervisorError) as reissue:
            TOOL.issue_actual_admission_marker_v1(
                work,
                fixture["work_identity"],
                fixture["boundary"],
            )
        assert reissue.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    finally:
        os.close(issued_descriptor)
        os.close(spending_descriptor)
        os.close(consumed_descriptor)
        os.close(root_descriptor)


def test_actual_admission_live_replay_rejects_spending_path_replacement(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_actual_admission_fixture(tmp_path)
    work = Path(fixture["work_identity"]["absolute_path"])
    issue_record, root_descriptor, issued_descriptor = (
        TOOL.issue_actual_admission_marker_v1(
            work,
            fixture["work_identity"],
            fixture["boundary"],
        )
    )
    consumed, spending_descriptor, consumed_descriptor = (
        TOOL.consume_actual_admission_marker_v1(
            root_descriptor,
            issued_descriptor,
            issue_record,
        )
    )
    try:
        marker = issue_record["issued_marker_evidence"]
        _replace_admission_marker_at_v1(
            root_descriptor,
            marker["spending_relative_path"],
            bytes.fromhex(consumed["spending_intent_hex"]),
        )
        with pytest.raises(TOOL.Q05BDualSupervisorError) as replaced:
            TOOL.replay_live_actual_admission_markers_v1(
                root_descriptor,
                issued_descriptor,
                spending_descriptor,
                consumed_descriptor,
                work,
                fixture["work_identity"],
                issue_record,
                consumed,
                "TEST_LIVE_AFTER_SPENDING_REPLACEMENT",
            )
        assert replaced.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    finally:
        os.close(issued_descriptor)
        os.close(spending_descriptor)
        os.close(consumed_descriptor)
        os.close(root_descriptor)


def _synthetic_issue_record_v1(boundary: dict[str, object]) -> dict[str, object]:
    payload = TOOL._canonical_json_bytes(boundary)
    issued_name, spending_name, consumed_name, failed_name = (
        TOOL._admission_marker_names_v1(boundary["attempt_id"])
    )
    marker_body: dict[str, object] = {
        "schema_version": TOOL.ACTUAL_ADMISSION_ISSUED_MARKER_SCHEMA_VERSION,
        "attempt_id": boundary["attempt_id"],
        "boundary_root": boundary["boundary_root"],
        "issued_relative_path": issued_name,
        "spending_relative_path": spending_name,
        "consumed_relative_path": consumed_name,
        "failed_relative_path": failed_name,
        "payload_length": len(payload),
        "payload_sha256": sha256(payload).hexdigest(),
        "file_device": 1,
        "file_inode": 2,
        "file_nlink": 1,
        "file_mode": 0o444,
        "work_root_device": 1,
        "work_root_inode": 3,
        "work_root_mode": 0o700,
        "issue_method": "DIRFD_O_NOFOLLOW_O_CREAT_O_EXCL_FSYNC_CHMOD0444_FSYNC",
    }
    marker = dict(marker_body)
    marker["issued_marker_root"] = sha256(
        TOOL.ACTUAL_ADMISSION_ISSUED_MARKER_ROOT_DOMAIN
        + TOOL._canonical_json_bytes(marker_body)
    ).hexdigest()
    record_body: dict[str, object] = {
        "schema_version": TOOL.ACTUAL_ADMISSION_ISSUE_RECORD_SCHEMA_VERSION,
        "attempt_id": boundary["attempt_id"],
        "boundary_root": boundary["boundary_root"],
        "pure_boundary_hex": payload.hex(),
        "anti_replay_scope": dict(TOOL.ACTUAL_ADMISSION_RUN_LOCAL_ANTI_REPLAY_SCOPE),
        "issued_marker_evidence": marker,
    }
    record = dict(record_body)
    record["issue_record_root"] = sha256(
        TOOL.ACTUAL_ADMISSION_ISSUE_RECORD_ROOT_DOMAIN
        + TOOL._canonical_json_bytes(record_body)
    ).hexdigest()
    TOOL.validate_actual_admission_issue_record_v1(record)
    return record


class _MockActualBackend:
    def __init__(
        self,
        fail_stage: int | None = None,
        fail_live_checkpoint: str | None = None,
    ) -> None:
        self.fail_stage = fail_stage
        self.fail_live_checkpoint = fail_live_checkpoint
        self.observed: list[int] = []
        self.receipts: tuple[str, str] | None = None
        self.boundary = None
        self.issue_record = None
        self.admission_latch = None
        self.live_checkpoints: list[str] = []
        self.active_mount_binding_slots = [None, None, None]
        self.active_actor_slots = [None, None, None]
        self.endpoint_actors = None
        self.host_actor = None
        self.command_runner = lambda *_args, **_kwargs: None

    def issue_stage3_to4_admission_boundary_v1(self, context):
        assert self.observed == [1, 2, 3]
        prior = [
            [index, context[f"stage_{index:02d}"]["stage_evidence_root"]]
            for index in range(1, 4)
        ]
        work = {
            "schema_version": "hegel-phase3a-q05b-admission-work-root-identity/1",
            "absolute_path": "/mock/work",
            "device": 1,
            "inode": 2,
            "nlink": 2,
            "mode": 0o700,
            "layout_sha256": "61" * 32,
        }
        body = {
            "schema_version": TOOL.ACTUAL_ADMISSION_BOUNDARY_SCHEMA_VERSION,
            "source_commit": context["source_commit"],
            "artifact_path": self.artifact_path.as_posix(),
            "attempt_id": "62" * 32,
            "prior_stage_root_rows": prior,
            "work_root_identity": work,
            "precondition_bundle_hex": TOOL._canonical_json_bytes({"mock": "bundle"}).hex(),
            "precondition_bundle_root": "63" * 32,
            "decision_hex": TOOL._canonical_json_bytes({"mock": "decision"}).hex(),
            "decision_root": "64" * 32,
            "qualification_authority_at_boundary": dict(
                TOOL.ACTUAL_ADMISSION_QUALIFICATION_AUTHORITY
            ),
            "closed_q1_authority": {
                **TOOL.ACTUAL_ADMISSION_CLOSED_Q1_AUTHORITY,
                "formal_output_roots": [None] * 8,
            },
        }
        self.boundary = dict(body)
        self.boundary["boundary_root"] = sha256(
            TOOL.ACTUAL_ADMISSION_BOUNDARY_ROOT_DOMAIN
            + TOOL._canonical_json_bytes(body)
        ).hexdigest()
        self.issue_record = _synthetic_issue_record_v1(self.boundary)
        self.admission_latch = TOOL.ActualAdmissionAttemptLatchV1(self.boundary)
        return self.issue_record

    def strict_replay_stage3_to4_admission_boundary_v1(self, issue_record, _context):
        assert TOOL._canonical_json_bytes(issue_record) == TOOL._canonical_json_bytes(
            self.issue_record
        )
        return issue_record

    def strict_replay_actual_admission_live_authority_v1(self, checkpoint):
        self.live_checkpoints.append(checkpoint)
        if checkpoint == self.fail_live_checkpoint:
            raise TOOL.Q05BDualSupervisorError(
                TOOL.FAIL_ACTUAL_ADMISSION,
                "mock live admission authority changed",
            )
        body = {
            "schema_version": "hegel-phase3a-q05b-live-admission-marker-replay/1",
            "checkpoint": checkpoint,
            "attempt_id": self.boundary["attempt_id"],
            "boundary_root": self.boundary["boundary_root"],
            "issue_record_root": self.issue_record["issue_record_root"],
            "consumed_marker_root": "65" * 32,
            "work_root_device": 1,
            "work_root_inode": 2,
            "work_root_nlink": 2,
            "work_root_mode": 0o700,
            "issued_file_device": 1,
            "issued_file_inode": 3,
            "issued_file_nlink": 2,
            "consumed_file_device": 1,
            "consumed_file_inode": 3,
            "consumed_file_nlink": 2,
            "spending_file_device": 1,
            "spending_file_inode": 4,
            "spending_file_nlink": 1,
            "boundary_payload_sha256": sha256(
                TOOL._canonical_json_bytes(self.boundary)
            ).hexdigest(),
            "issued_consumed_same_inode": True,
            "work_root_path_matches_held_descriptor": True,
            "issued_path_matches_held_descriptor": True,
            "spending_path_matches_held_descriptor": True,
            "consumed_path_matches_held_descriptor": True,
        }
        value = dict(body)
        value["live_marker_replay_root"] = sha256(
            TOOL.ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_ROOT_DOMAIN
            + TOOL._canonical_json_bytes(body)
        ).hexdigest()
        return TOOL.validate_actual_admission_live_marker_replay_surface_v1(
            value,
            checkpoint,
        )

    def __getattr__(self, name: str):
        match = re.fullmatch(r"stage_([0-9]{2})_v1", name)
        if match is None:
            raise AttributeError(name)
        stage_id = int(match.group(1))

        def run(context):
            assert context["q1_state"] == "NOT_RUN"
            assert context["q1_formal_output_roots"] == [None] * 8
            if stage_id != len(self.observed) + 1:
                raise AssertionError("stage order differs")
            self.observed.append(stage_id)
            if stage_id == 4:
                assert context["stage3_to4_admission_issue_record"] == self.issue_record
                assert self.admission_latch is not None
                self.admission_latch.consume_once(self.boundary)
            if self.fail_stage == stage_id:
                raise TOOL.Q05BDualSupervisorError(
                    TOOL.FAIL_POLICY,
                    f"mock failure at stage {stage_id}",
                )
            if self.receipts is None and stage_id >= 8:
                full_leaf = WIRE.full_v16_leaf_manifest_v1()
                semantic_root, projection_root = (
                    WIRE.q1_semantic_and_projection_roots_v1(full_leaf)
                )
                predicate_rows = tuple(
                    (predicate_id, predicate_name, True, bytes([predicate_id]) * 32)
                    for predicate_id, predicate_name in WIRE.QUALIFICATION_PREDICATE_REGISTRY[:19]
                )
                source_raw = bytes.fromhex(context["source_commit"])
                pre_receipt_root = WIRE.pre_receipt_evidence_root_v1(
                    source_raw,
                    predicate_rows,
                )
                candidate_object = WIRE.Q05BQualificationCandidateReceiptV1(
                    source_commit=source_raw,
                    q1_semantic_binding_root=semantic_root,
                    q1_projection_profile_root=projection_root,
                    q0_receipt_root=(
                        WIRE.Q0_SATURATION_RECEIPT_ROOT_FROM_Q1_PREREGISTRATION
                    ),
                    full_leaf_manifest_root=full_leaf.manifest_root,
                    implementation_roots=(b"p" * 32, b"r" * 32, b"h" * 32),
                    neutral_manifest_roots=(b"n" * 32,) * 3,
                    bounded_state_roots=(b"o" * 32, b"s" * 32),
                    bundle_evidence_root=b"b" * 32,
                    isolation_evidence_root=b"i" * 32,
                    resource_evidence_root=b"e" * 32,
                    pre_receipt_evidence_root=pre_receipt_root,
                    predicate_rows_1_through_19=predicate_rows,
                )
                final_object = WIRE.Q05BQualificationReceiptV1(candidate_object)
                self.receipts = (
                    candidate_object.canonical_bytes.hex(),
                    final_object.canonical_bytes.hex(),
                )
            if stage_id <= 7:
                count, mask, candidate, final = 0, 0, None, None
            elif stage_id == 8:
                assert self.receipts is not None
                count, mask, candidate, final = 19, 0x7FFFF, self.receipts[0], None
            else:
                assert self.receipts is not None
                count, mask, candidate, final = (
                    20,
                    0xFFFFF,
                    self.receipts[0],
                    self.receipts[1],
                )
            evidence = {"mock_only": True, "prior_stage_count": stage_id - 1}
            if stage_id >= 4:
                evidence["actual_admission_attempt_id"] = self.boundary["attempt_id"]
                evidence["actual_admission_boundary_root"] = self.boundary[
                    "boundary_root"
                ]
            if stage_id == 10:
                artifact_value = context["strict_replayed_artifact"]
                artifact_payload = TOOL._canonical_json_bytes(artifact_value)
                candidate_object = WIRE.decode_qualification_candidate_receipt_v1(
                    bytes.fromhex(candidate)
                )
                final_object = WIRE.decode_qualification_receipt_v1(
                    bytes.fromhex(final)
                )
                evidence = {
                    "artifact_length": len(artifact_payload),
                    "artifact_set_root": sha256(
                        b"HEGEL/Q05B/MOCK/ARTIFACT_SET/V1\x00"
                        + artifact_payload
                    ).hexdigest(),
                    "artifact_sha256": sha256(
                        artifact_payload
                    ).hexdigest(),
                    "candidate_receipt_hex": candidate,
                    "candidate_receipt_root": candidate_object.receipt_root.hex(),
                    "final_receipt_hex": final,
                    "final_receipt_root": final_object.receipt_root.hex(),
                    "strict_replay_equal": True,
                }
            if stage_id >= 4:
                evidence["actual_admission_live_marker_replay"] = (
                    self.strict_replay_actual_admission_live_authority_v1(
                        f"STAGE_{stage_id:02d}_BEFORE_EVIDENCE"
                    )
                )
            return TOOL.actual_stage_evidence_v1(
                stage_id,
                TOOL.ACTUAL_ORCHESTRATION_STAGE_REGISTRY[stage_id - 1][1],
                context["source_commit"],
                evidence,
                qualification_count=count,
                qualification_mask=mask,
                candidate_receipt_hex=candidate,
                final_receipt_hex=final,
            )

        return run

    def canonical_artifact_value_v1(self, context):
        assert len([key for key in context if key.startswith("stage_")]) == 9
        assert self.receipts is not None
        return {
            "candidate_receipt_hex": self.receipts[0],
            "final_receipt_hex": self.receipts[1],
            "mock_only_not_qualification": True,
            "q1_authority": {
                "certificate_active": False,
                "formal_output_roots": [None] * 8,
                "gate_count": 0,
                "gate_mask": 0,
                "state": "NOT_RUN",
            },
            "schema_version": "hegel-q05b-orchestration-mock-test-only/1",
        }

    def strict_replay_artifact_value_v1(self, value, context):
        assert context["stage_09"]["qualification_count"] == 20
        assert value["mock_only_not_qualification"] is True
        return value


def _synthetic_stage8_stage9_registries_v1(
    source_commit: str = "90" * 20,
) -> tuple[dict[str, object], dict[str, object]]:
    full_leaf = WIRE.full_v16_leaf_manifest_v1()
    semantic_root, projection_root = WIRE.q1_semantic_and_projection_roots_v1(
        full_leaf
    )
    predicate_rows = tuple(
        (predicate_id, predicate_name, True, bytes([predicate_id]) * 32)
        for predicate_id, predicate_name in WIRE.QUALIFICATION_PREDICATE_REGISTRY[:19]
    )
    source_raw = bytes.fromhex(source_commit)
    candidate_object = WIRE.Q05BQualificationCandidateReceiptV1(
        source_commit=source_raw,
        q1_semantic_binding_root=semantic_root,
        q1_projection_profile_root=projection_root,
        q0_receipt_root=WIRE.Q0_SATURATION_RECEIPT_ROOT_FROM_Q1_PREREGISTRATION,
        full_leaf_manifest_root=full_leaf.manifest_root,
        implementation_roots=(b"p" * 32, b"r" * 32, b"h" * 32),
        neutral_manifest_roots=(b"n" * 32,) * 3,
        bounded_state_roots=(b"o" * 32, b"s" * 32),
        bundle_evidence_root=b"b" * 32,
        isolation_evidence_root=b"i" * 32,
        resource_evidence_root=b"e" * 32,
        pre_receipt_evidence_root=WIRE.pre_receipt_evidence_root_v1(
            source_raw,
            predicate_rows,
        ),
        predicate_rows_1_through_19=predicate_rows,
    )
    final_object = WIRE.Q05BQualificationReceiptV1(candidate_object)
    ordered_rows = [
        [row[0], row[1].decode("ascii"), row[2], row[3].hex()]
        for row in predicate_rows
    ]
    candidate = {
        "actual_admission_evidence_root": "a1" * 32,
        "bundle_evidence_root": candidate_object.bundle_evidence_root.hex(),
        "candidate_receipt_cbor_hex": candidate_object.canonical_bytes.hex(),
        "candidate_receipt_root": candidate_object.receipt_root.hex(),
        "closed_q1_authority": ARTIFACT.CLOSED_Q1_AUTHORITY,
        "host_execution_binding_root": "a2" * 32,
        "isolation_evidence_root": candidate_object.isolation_evidence_root.hex(),
        "ordered_predicate_rows": ordered_rows,
        "qualification_count": 19,
        "qualification_mask": 0x7FFFF,
        "resource_evidence_root": candidate_object.resource_evidence_root.hex(),
    }
    derived = {
        **candidate,
        "artifact_set_root": "a3" * 32,
        "final_receipt_cbor_hex": final_object.canonical_bytes.hex(),
        "final_receipt_root": final_object.receipt_root.hex(),
        "qualification_count": 20,
        "qualification_mask": 0xFFFFF,
    }
    assert set(candidate) == set(ARTIFACT.STAGE8_CANDIDATE_REGISTRY_KEYS)
    assert set(derived) == set(ARTIFACT.STAGE9_DERIVED_REGISTRY_KEYS)
    ARTIFACT.validate_stage8_candidate_registry_v1(
        candidate,
        candidate["actual_admission_evidence_root"],
    )
    ARTIFACT.validate_stage9_derived_registry_v1(
        derived,
        candidate,
        candidate["actual_admission_evidence_root"],
    )
    return candidate, derived


def _synthetic_final_delivery_live_surfaces_v1() -> dict[str, dict[str, object]]:
    backend = _MockActualBackend()
    backend.boundary = {
        "attempt_id": "b1" * 32,
        "boundary_root": "b2" * 32,
    }
    backend.issue_record = {"issue_record_root": "b3" * 32}
    return {
        checkpoint: backend.strict_replay_actual_admission_live_authority_v1(
            checkpoint
        )
        for checkpoint in (
            "PRE_ARTIFACT_ASSEMBLY",
            "STAGE_10_BEFORE_EVIDENCE",
            "PREPUBLICATION_AFTER_STAGE10",
            "POSTPUBLICATION_AFTER_ANCHORED_ARTIFACT_REPLAY",
        )
    }


def _different_live_authority_v1(
    value: dict[str, object],
) -> dict[str, object]:
    body = deepcopy(value)
    body.pop("live_marker_replay_root")
    body["attempt_id"] = "bf" * 32
    body["live_marker_replay_root"] = sha256(
        TOOL.ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_ROOT_DOMAIN
        + TOOL._canonical_json_bytes(body)
    ).hexdigest()
    return TOOL.validate_actual_admission_live_marker_replay_surface_v1(
        body,
        body["checkpoint"],
    )


class _StrictSyntheticArtifactBackend(_MockActualBackend):
    """Second-scale kernel fixture with real receipt and artifact adapters.

    The expensive eleven-section semantic replay is supplied by a test-local,
    byte-exact builder; artifact schema, candidate/final receipt decoding,
    Stage8--10 adapters, publication, and final delivery remain production code.
    """

    def __init__(self, source_commit: str) -> None:
        super().__init__()
        self.candidate, self.derived = _synthetic_stage8_stage9_registries_v1(
            source_commit
        )
        self.artifact_value: dict[str, object] | None = None

    def _start_explicit_stage(self, stage_id: int) -> None:
        assert stage_id == len(self.observed) + 1
        self.observed.append(stage_id)

    def stage_08_v1(self, context):
        self._start_explicit_stage(8)
        evidence = {
            "actual_admission_evidence_root": self.candidate[
                "actual_admission_evidence_root"
            ],
            "artifact_section_names": list(ARTIFACT.SECTION_NAMES),
            "bundle_evidence_root": self.candidate["bundle_evidence_root"],
            "candidate_receipt_root": self.candidate[
                "candidate_receipt_root"
            ],
            "closed_q1_authority": self.candidate["closed_q1_authority"],
            "host_execution_binding_root": self.candidate[
                "host_execution_binding_root"
            ],
            "isolation_evidence_root": self.candidate[
                "isolation_evidence_root"
            ],
            "ordered_predicate_rows": self.candidate[
                "ordered_predicate_rows"
            ],
            "resource_evidence_root": self.candidate[
                "resource_evidence_root"
            ],
            "strict_evidence_replay_equal": True,
            "actual_admission_live_marker_replay": (
                self.strict_replay_actual_admission_live_authority_v1(
                    "STAGE_08_BEFORE_EVIDENCE"
                )
            ),
        }
        return TOOL.actual_stage_evidence_v1(
            8,
            TOOL.ACTUAL_ORCHESTRATION_STAGE_REGISTRY[7][1],
            context["source_commit"],
            evidence,
            qualification_count=19,
            qualification_mask=0x7FFFF,
            candidate_receipt_hex=self.candidate[
                "candidate_receipt_cbor_hex"
            ],
            final_receipt_hex=None,
        )

    def stage_09_v1(self, context):
        self._start_explicit_stage(9)
        pre_artifact_live = (
            self.strict_replay_actual_admission_live_authority_v1(
                "PRE_ARTIFACT_ASSEMBLY"
            )
        )
        sections: dict[str, object] = {
            name: {
                "schema_version": (
                    "hegel-phase3a-q05b-strict-synthetic-section-test-only/1"
                ),
                "section_name": name,
            }
            for name in ARTIFACT.SECTION_NAMES
        }
        sections["actual_admission"] = {
            "schema_version": (
                "hegel-phase3a-q05b-strict-synthetic-admission-test-only/1"
            ),
            "actual_admission_evidence_root": self.candidate[
                "actual_admission_evidence_root"
            ],
            "pre_artifact_live_marker_replay": pre_artifact_live,
            "root_registry": {
                "pre_artifact_live_marker_replay_root": pre_artifact_live[
                    "live_marker_replay_root"
                ]
            },
        }
        artifact = {
            "derived": dict(self.derived),
            "schema_version": ARTIFACT.ARTIFACT_SCHEMA_VERSION,
            "sections": sections,
            "status": "Q05B_QUALIFICATION_20_OF_20_Q1_NOT_RUN",
        }
        body = json.loads(ARTIFACT._canonical_json(artifact))
        body["derived"].pop("artifact_set_root")
        artifact["derived"]["artifact_set_root"] = ARTIFACT._json_root(
            ARTIFACT.ARTIFACT_SET_ROOT_DOMAIN,
            body,
        ).hex()
        self.derived = dict(artifact["derived"])
        self.artifact_value = artifact
        evidence = {
            "actual_admission_evidence_root": self.derived[
                "actual_admission_evidence_root"
            ],
            "artifact_set_root": self.derived["artifact_set_root"],
            "bundle_evidence_root": self.derived["bundle_evidence_root"],
            "candidate_receipt_root": self.derived["candidate_receipt_root"],
            "closed_q1_authority": self.derived["closed_q1_authority"],
            "final_receipt_root": self.derived["final_receipt_root"],
            "host_execution_binding_root": self.derived[
                "host_execution_binding_root"
            ],
            "isolation_evidence_root": self.derived[
                "isolation_evidence_root"
            ],
            "ordered_predicate_rows": self.derived[
                "ordered_predicate_rows"
            ],
            "predicate20_added_after_candidate_replay": True,
            "qualification_count": 20,
            "qualification_mask": 0xFFFFF,
            "resource_evidence_root": self.derived[
                "resource_evidence_root"
            ],
            "strict_derived_cross_equal": True,
            "actual_admission_live_marker_replay": (
                self.strict_replay_actual_admission_live_authority_v1(
                    "STAGE_09_BEFORE_EVIDENCE"
                )
            ),
        }
        return TOOL.actual_stage_evidence_v1(
            9,
            TOOL.ACTUAL_ORCHESTRATION_STAGE_REGISTRY[8][1],
            context["source_commit"],
            evidence,
            qualification_count=20,
            qualification_mask=0xFFFFF,
            candidate_receipt_hex=self.derived[
                "candidate_receipt_cbor_hex"
            ],
            final_receipt_hex=self.derived["final_receipt_cbor_hex"],
        )

    def canonical_artifact_value_v1(self, context):
        assert context["stage_09"]["qualification_count"] == 20
        assert self.artifact_value is not None
        return deepcopy(self.artifact_value)

    def strict_replay_artifact_value_v1(self, value, context):
        assert self.artifact_value is not None
        assert value == self.artifact_value
        return ARTIFACT.decode_and_replay_actual_artifact_v1(
            ARTIFACT._canonical_json(value)
        )

    def stage_10_v1(self, context):
        self._start_explicit_stage(10)
        assert context["strict_replayed_artifact"] == self.artifact_value
        payload = ARTIFACT.canonical_actual_artifact_bytes_v1(
            self.artifact_value
        )
        evidence = {
            "actual_admission_evidence_root": self.derived[
                "actual_admission_evidence_root"
            ],
            "artifact_length": len(payload),
            "artifact_set_root": self.derived["artifact_set_root"],
            "artifact_sha256": sha256(payload).hexdigest(),
            "bundle_evidence_root": self.derived["bundle_evidence_root"],
            "candidate_receipt_hex": self.derived[
                "candidate_receipt_cbor_hex"
            ],
            "candidate_receipt_root": self.derived[
                "candidate_receipt_root"
            ],
            "closed_q1_authority": self.derived["closed_q1_authority"],
            "final_receipt_hex": self.derived["final_receipt_cbor_hex"],
            "final_receipt_root": self.derived["final_receipt_root"],
            "host_execution_binding_root": self.derived[
                "host_execution_binding_root"
            ],
            "isolation_evidence_root": self.derived[
                "isolation_evidence_root"
            ],
            "ordered_predicate_rows": self.derived[
                "ordered_predicate_rows"
            ],
            "qualification_count": 20,
            "qualification_mask": 0xFFFFF,
            "resource_evidence_root": self.derived[
                "resource_evidence_root"
            ],
            "strict_replay_equal": True,
            "actual_admission_live_marker_replay": (
                self.strict_replay_actual_admission_live_authority_v1(
                    "STAGE_10_BEFORE_EVIDENCE"
                )
            ),
        }
        return TOOL.actual_stage_evidence_v1(
            10,
            TOOL.ACTUAL_ORCHESTRATION_STAGE_REGISTRY[9][1],
            context["source_commit"],
            evidence,
            qualification_count=20,
            qualification_mask=0xFFFFF,
            candidate_receipt_hex=self.derived[
                "candidate_receipt_cbor_hex"
            ],
            final_receipt_hex=self.derived["final_receipt_cbor_hex"],
        )


def test_actual_orchestration_rejects_backend_self_attested_nonqualification_artifact(
    tmp_path: Path,
) -> None:
    artifact = (tmp_path / "artifact.json").resolve()
    backend = _MockActualBackend()
    backend.artifact_path = artifact
    publisher_calls: list[object] = []

    def capture_publisher(
        path: Path,
        value: object,
    ) -> TOOL.AnchoredPublishedArtifactV1:
        publisher_calls.append((path, value))
        raise AssertionError("invalid artifact reached publisher")

    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        TOOL.orchestrate_actual_with_backend_v1(
            "12" * 20,
            artifact,
            backend,
            publisher=capture_publisher,
        )
    assert rejected.value.code == TOOL.FAIL_ARTIFACT
    assert backend.observed == list(range(1, 10))
    assert publisher_calls == []
    assert not artifact.exists()


def test_strict_synthetic_eleven_section_artifact_runs_stage8_to_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "13" * 20
    artifact_path = (tmp_path / "strict-synthetic-artifact.json").resolve()
    backend = _StrictSyntheticArtifactBackend(source_commit)
    backend.artifact_path = artifact_path

    def strict_minimal_section_replay(
        sections: object,
        *,
        candidate_only: bool,
    ) -> dict[str, object]:
        assert backend.artifact_value is not None
        assert type(sections) is dict
        assert set(sections) == set(ARTIFACT.SECTION_NAMES)
        assert sections == backend.artifact_value["sections"]
        if candidate_only:
            return deepcopy(backend.candidate)
        return deepcopy(backend.artifact_value)

    monkeypatch.setattr(
        ARTIFACT,
        "_replay_actual_evidence_v1",
        strict_minimal_section_replay,
    )
    handles: list[TOOL.AnchoredPublishedArtifactV1] = []

    def capture_publisher(
        path: Path,
        value: object,
    ) -> TOOL.AnchoredPublishedArtifactV1:
        handle = TOOL.atomic_publish_canonical_artifact_v1(path, value)
        handles.append(handle)
        return handle

    result = TOOL.orchestrate_actual_with_backend_v1(
        source_commit,
        artifact_path,
        backend,
        publisher=capture_publisher,
    )
    published = ARTIFACT.decode_and_replay_actual_artifact_v1(
        artifact_path.read_bytes()
    )
    assert backend.observed == list(range(1, 11))
    assert set(published["sections"]) == set(ARTIFACT.SECTION_NAMES)
    assert published["derived"]["candidate_receipt_root"] == result[
        "candidate_receipt_root"
    ]
    assert published["derived"]["final_receipt_root"] == result[
        "final_receipt_root"
    ]
    assert result["qualification_count"] == 20
    assert result["qualification_mask"] == 0xFFFFF
    assert result["q1_state"] == "NOT_RUN"
    assert result["q1_formal_output_roots"] == [None] * 8
    identity = result["actual_final_delivery_identity"]
    assert len(identity["ordered_stage_root_rows"]) == 10
    assert identity["actual_admission_evidence_root"] == backend.candidate[
        "actual_admission_evidence_root"
    ]
    assert result["actual_final_delivery_root"] == sha256(
        TOOL.ACTUAL_FINAL_DELIVERY_ROOT_DOMAIN
        + TOOL._canonical_json_bytes(identity)
    ).hexdigest()
    assert len(handles) == 1 and handles[0].closed is True


def test_postpublication_admission_fd_close_eio_rolls_back_owned_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "16" * 20
    artifact_path = (tmp_path / "descriptor-close-eio.json").resolve()
    backend = _StrictSyntheticArtifactBackend(source_commit)
    backend.artifact_path = artifact_path
    descriptors = tuple(os.open("/dev/null", os.O_RDONLY) for _ in range(4))
    for field, descriptor in zip(
        TOOL._ACTUAL_ADMISSION_DESCRIPTOR_FIELDS,
        descriptors,
        strict=True,
    ):
        setattr(backend, field, descriptor)

    def strict_minimal_section_replay(
        sections: object,
        *,
        candidate_only: bool,
    ) -> dict[str, object]:
        assert backend.artifact_value is not None
        assert sections == backend.artifact_value["sections"]
        return deepcopy(
            backend.candidate if candidate_only else backend.artifact_value
        )

    monkeypatch.setattr(
        ARTIFACT,
        "_replay_actual_evidence_v1",
        strict_minimal_section_replay,
    )
    handles: list[TOOL.AnchoredPublishedArtifactV1] = []

    def capture_publisher(path, value):
        handle = TOOL.atomic_publish_canonical_artifact_v1(path, value)
        handles.append(handle)
        return handle

    real_close = os.close
    closed_admission: set[int] = set()
    injected = False

    def close_with_one_eio(descriptor):
        nonlocal injected
        if descriptor in descriptors and descriptor not in closed_admission:
            assert all(
                getattr(backend, field) is None
                for field in TOOL._ACTUAL_ADMISSION_DESCRIPTOR_FIELDS
            )
            closed_admission.add(descriptor)
            real_close(descriptor)
            if descriptor == descriptors[1] and not injected:
                injected = True
                raise OSError(errno.EIO, "injected admission close failure")
            return
        real_close(descriptor)

    monkeypatch.setattr(TOOL.os, "close", close_with_one_eio)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.orchestrate_actual_with_backend_v1(
            source_commit,
            artifact_path,
            backend,
            publisher=capture_publisher,
        )
    assert failure.value.code == TOOL.FAIL_POLICY
    assert injected is True
    assert closed_admission == set(descriptors)
    assert all(
        getattr(backend, field) is None
        for field in TOOL._ACTUAL_ADMISSION_DESCRIPTOR_FIELDS
    )
    assert not artifact_path.exists()
    assert len(handles) == 1 and handles[0].closed is True


def test_temporary_root_cleanup_eio_occurs_under_handle_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "17" * 20
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    cargo = tmp_path / "cargo"
    cargo.mkdir()
    artifact_path = project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH
    backend = _StrictSyntheticArtifactBackend(source_commit)
    backend.artifact_path = artifact_path
    descriptors = tuple(os.open("/dev/null", os.O_RDONLY) for _ in range(4))
    for field, descriptor in zip(
        TOOL._ACTUAL_ADMISSION_DESCRIPTOR_FIELDS,
        descriptors,
        strict=True,
    ):
        setattr(backend, field, descriptor)

    def strict_minimal_section_replay(
        sections: object,
        *,
        candidate_only: bool,
    ) -> dict[str, object]:
        assert backend.artifact_value is not None
        assert sections == backend.artifact_value["sections"]
        return deepcopy(
            backend.candidate if candidate_only else backend.artifact_value
        )

    monkeypatch.setattr(
        ARTIFACT,
        "_replay_actual_evidence_v1",
        strict_minimal_section_replay,
    )
    monkeypatch.setattr(
        TOOL, "verify_actual_source_commit_v1", lambda _root, requested: requested
    )
    work_roots: list[Path] = []

    def backend_factory(*args):
        work_roots.append(args[4])
        return backend

    monkeypatch.setattr(TOOL, "ConcreteQ05BActualBackendV1", backend_factory)
    real_temporary_directory = tempfile.TemporaryDirectory
    cleanup_calls = 0

    class CleanupEIOManager:
        def __init__(self, *args, **kwargs):
            self.inner = real_temporary_directory(*args, **kwargs)
            self.name = self.inner.name

        def cleanup(self):
            nonlocal cleanup_calls
            cleanup_calls += 1
            self.inner.cleanup()
            raise OSError(errno.EIO, "injected temporary cleanup failure")

    monkeypatch.setattr(TOOL.tempfile, "TemporaryDirectory", CleanupEIOManager)
    with pytest.raises(OSError, match="temporary cleanup failure"):
        TOOL.run_actual_v1(project, source_commit, artifact_path, cargo)
    assert cleanup_calls == 1
    assert len(work_roots) == 1 and not work_roots[0].exists()
    assert not artifact_path.exists()
    assert all(
        getattr(backend, field) is None
        for field in TOOL._ACTUAL_ADMISSION_DESCRIPTOR_FIELDS
    )
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_publisher_handle_close_error_after_final_handoff_is_not_false_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "18" * 20
    artifact_path = (tmp_path / "handle-close-after-handoff.json").resolve()
    backend = _StrictSyntheticArtifactBackend(source_commit)
    backend.artifact_path = artifact_path

    def strict_minimal_section_replay(
        sections: object,
        *,
        candidate_only: bool,
    ) -> dict[str, object]:
        assert backend.artifact_value is not None
        assert sections == backend.artifact_value["sections"]
        return deepcopy(
            backend.candidate if candidate_only else backend.artifact_value
        )

    monkeypatch.setattr(
        ARTIFACT,
        "_replay_actual_evidence_v1",
        strict_minimal_section_replay,
    )
    real_close = TOOL.close_anchored_published_artifact_v1
    close_attempts = 0

    def close_then_report_error(handle):
        nonlocal close_attempts
        close_attempts += 1
        real_close(handle)
        raise TOOL.Q05BDualSupervisorError(
            TOOL.FAIL_ARTIFACT,
            "injected post-handoff handle close error",
        )

    monkeypatch.setattr(
        TOOL,
        "close_anchored_published_artifact_v1",
        close_then_report_error,
    )
    result = TOOL.orchestrate_actual_with_backend_v1(
        source_commit,
        artifact_path,
        backend,
    )
    assert close_attempts == 1
    assert result["qualification_count"] == 20
    assert result["publisher_handle_close_status"] == (
        "ERROR_AFTER_FINAL_HANDOFF"
    )
    assert artifact_path.is_file()
    assert ARTIFACT.decode_and_replay_actual_artifact_v1(
        artifact_path.read_bytes()
    )["derived"]["final_receipt_root"] == result["final_receipt_root"]


def test_publisher_without_ownership_handle_reports_unowned_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "18" * 20
    artifact_path = (tmp_path / "publisher-no-handle-residual.json").resolve()
    backend = _StrictSyntheticArtifactBackend(source_commit)
    backend.artifact_path = artifact_path

    def strict_minimal_section_replay(
        sections: object,
        *,
        candidate_only: bool,
    ) -> dict[str, object]:
        assert backend.artifact_value is not None
        assert sections == backend.artifact_value["sections"]
        return deepcopy(
            backend.candidate if candidate_only else backend.artifact_value
        )

    monkeypatch.setattr(
        ARTIFACT,
        "_replay_actual_evidence_v1",
        strict_minimal_section_replay,
    )

    def publish_without_handle(path: Path, value: object) -> None:
        path.write_bytes(TOOL._canonical_json_bytes(value))
        path.chmod(0o444)
        return None

    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.orchestrate_actual_with_backend_v1(
            source_commit,
            artifact_path,
            backend,
            publisher=publish_without_handle,
        )
    assert failure.value.code == TOOL.FAIL_ARTIFACT
    assert failure.value.artifact_written is True
    assert TOOL._error_object(failure.value)["artifact_written"] is True
    assert artifact_path.is_file()


def test_owned_rollback_eio_reports_truthful_artifact_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "19" * 20
    artifact_path = (tmp_path / "rollback-eio-residual.json").resolve()
    backend = _StrictSyntheticArtifactBackend(source_commit)
    backend.artifact_path = artifact_path

    def strict_minimal_section_replay(
        sections: object,
        *,
        candidate_only: bool,
    ) -> dict[str, object]:
        assert backend.artifact_value is not None
        assert sections == backend.artifact_value["sections"]
        return deepcopy(
            backend.candidate if candidate_only else backend.artifact_value
        )

    monkeypatch.setattr(
        ARTIFACT,
        "_replay_actual_evidence_v1",
        strict_minimal_section_replay,
    )
    monkeypatch.setattr(
        TOOL,
        "rollback_anchored_published_artifact_v1",
        lambda *_args: (_ for _ in ()).throw(
            OSError(errno.EIO, "injected rollback failure")
        ),
    )

    def fail_before_handoff():
        raise TOOL.Q05BDualSupervisorError(
            TOOL.FAIL_POLICY,
            "injected pre-handoff failure",
        )

    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.orchestrate_actual_with_backend_v1(
            source_commit,
            artifact_path,
            backend,
            pre_handoff_cleanup=fail_before_handoff,
        )
    assert failure.value.code == TOOL.FAIL_ARTIFACT
    assert failure.value.artifact_written is True
    assert "same-inode rollback failed" in failure.value.detail
    assert TOOL._error_object(failure.value)["artifact_written"] is True
    assert artifact_path.is_file()


def test_rollback_and_handle_close_failures_preserve_composite_and_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "1a" * 20
    artifact_path = (tmp_path / "rollback-and-close-eio.json").resolve()
    backend = _StrictSyntheticArtifactBackend(source_commit)
    backend.artifact_path = artifact_path

    def strict_minimal_section_replay(
        sections: object,
        *,
        candidate_only: bool,
    ) -> dict[str, object]:
        assert backend.artifact_value is not None
        assert sections == backend.artifact_value["sections"]
        return deepcopy(
            backend.candidate if candidate_only else backend.artifact_value
        )

    monkeypatch.setattr(
        ARTIFACT,
        "_replay_actual_evidence_v1",
        strict_minimal_section_replay,
    )
    monkeypatch.setattr(
        TOOL,
        "rollback_anchored_published_artifact_v1",
        lambda *_args: (_ for _ in ()).throw(
            OSError(errno.EIO, "injected rollback failure")
        ),
    )
    real_close = TOOL.close_anchored_published_artifact_v1
    close_attempts = 0

    def close_then_fail(handle):
        nonlocal close_attempts
        close_attempts += 1
        real_close(handle)
        raise OSError(errno.EIO, "injected handle close failure")

    monkeypatch.setattr(
        TOOL,
        "close_anchored_published_artifact_v1",
        close_then_fail,
    )

    def fail_before_handoff():
        raise TOOL.Q05BDualSupervisorError(
            TOOL.FAIL_POLICY,
            "injected pre-handoff failure",
        )

    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.orchestrate_actual_with_backend_v1(
            source_commit,
            artifact_path,
            backend,
            pre_handoff_cleanup=fail_before_handoff,
        )
    assert close_attempts == 1
    assert failure.value.code == TOOL.FAIL_ARTIFACT
    assert failure.value.artifact_written is True
    assert "same-inode rollback failed" in failure.value.detail
    assert "handle_close=OSError" in failure.value.detail
    assert artifact_path.is_file()


def test_normal_owned_failure_rolls_back_and_reports_artifact_not_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "1b" * 20
    artifact_path = (tmp_path / "normal-owned-failure.json").resolve()
    backend = _StrictSyntheticArtifactBackend(source_commit)
    backend.artifact_path = artifact_path

    def strict_minimal_section_replay(
        sections: object,
        *,
        candidate_only: bool,
    ) -> dict[str, object]:
        assert backend.artifact_value is not None
        assert sections == backend.artifact_value["sections"]
        return deepcopy(
            backend.candidate if candidate_only else backend.artifact_value
        )

    monkeypatch.setattr(
        ARTIFACT,
        "_replay_actual_evidence_v1",
        strict_minimal_section_replay,
    )

    def fail_before_handoff():
        raise TOOL.Q05BDualSupervisorError(
            TOOL.FAIL_POLICY,
            "injected normal pre-handoff failure",
        )

    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.orchestrate_actual_with_backend_v1(
            source_commit,
            artifact_path,
            backend,
            pre_handoff_cleanup=fail_before_handoff,
        )
    assert failure.value.code == TOOL.FAIL_POLICY
    assert failure.value.artifact_written is False
    assert TOOL._error_object(failure.value)["artifact_written"] is False
    assert not artifact_path.exists()
    with pytest.raises(TypeError):
        TOOL.Q05BDualSupervisorError(
            TOOL.FAIL_ARTIFACT,
            "bool alias",
            artifact_written=1,
        )


def test_actual_orchestration_failure_before_final_has_no_artifact_or_receipt(
    tmp_path: Path,
) -> None:
    for fail_stage in (1, 4, 7, 8, 9):
        target = (tmp_path / f"failed-{fail_stage}.json").resolve()
        backend = _MockActualBackend(fail_stage)
        backend.artifact_path = target
        with pytest.raises(TOOL.Q05BDualSupervisorError):
            TOOL.orchestrate_actual_with_backend_v1(
                "34" * 20,
                target,
                backend,
            )
        assert not target.exists()
        assert backend.observed == list(range(1, fail_stage + 1))


def test_orchestration_freezes_stage_rows_against_backend_nested_mutation(
    tmp_path: Path,
) -> None:
    class MutatingBackend(_MockActualBackend):
        returned_stage1: dict[str, object] | None = None
        original_stage1_root: str | None = None

        def stage_01_v1(self, context):
            result = super().__getattr__("stage_01_v1")(context)
            self.returned_stage1 = result
            self.original_stage1_root = result["stage_evidence_root"]
            return result

        def stage_02_v1(self, context):
            assert self.returned_stage1 is not None
            self.returned_stage1["stage_evidence_root"] = "fe" * 32
            context["stage_01"]["stage_evidence_root"] = "fd" * 32
            return super().__getattr__("stage_02_v1")(context)

    artifact = (tmp_path / "mutation-rejected-before-publish.json").resolve()
    backend = MutatingBackend()
    backend.artifact_path = artifact
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.orchestrate_actual_with_backend_v1(
            "14" * 20,
            artifact,
            backend,
        )
    assert failure.value.code == TOOL.FAIL_ARTIFACT
    assert backend.boundary["prior_stage_root_rows"][0][1] == (
        backend.original_stage1_root
    )
    assert backend.boundary["prior_stage_root_rows"][0][1] not in {
        "fd" * 32,
        "fe" * 32,
    }
    assert not artifact.exists()


@pytest.mark.parametrize(
    ("field", "alias"),
    (
        ("certificate_active", 0),
        ("gate_count", False),
        ("gate_mask", False),
    ),
)
def test_actual_stage_q1_authority_rejects_bool_int_alias(
    field: str,
    alias: object,
) -> None:
    source_commit = "15" * 20
    stage = TOOL.actual_stage_evidence_v1(
        1,
        TOOL.ACTUAL_ORCHESTRATION_STAGE_REGISTRY[0][1],
        source_commit,
        {"synthetic": "q1-alias"},
        qualification_count=0,
        qualification_mask=0,
        candidate_receipt_hex=None,
        final_receipt_hex=None,
    )
    forged = deepcopy(stage)
    forged["q1_authority"][field] = alias
    forged_body = deepcopy(forged)
    forged_body.pop("stage_evidence_root")
    forged["stage_evidence_root"] = sha256(
        b"HEGEL/Q05B/ACTUAL/STAGE_EVIDENCE/V1\x00"
        + (1).to_bytes(2, "big")
        + TOOL._canonical_json_bytes(forged_body)
    ).hexdigest()
    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        TOOL.validate_actual_stage_evidence_v1(
            forged,
            1,
            TOOL.ACTUAL_ORCHESTRATION_STAGE_REGISTRY[0][1],
            source_commit,
        )
    assert rejected.value.code == TOOL.FAIL_POLICY


def test_stage8_stage9_receipt_registry_tamper_matrix_is_fail_closed() -> None:
    candidate, derived = _synthetic_stage8_stage9_registries_v1()
    admission_root = candidate["actual_admission_evidence_root"]

    candidate_tampers: list[dict[str, object]] = []
    missing = deepcopy(candidate)
    missing.pop("host_execution_binding_root")
    candidate_tampers.append(missing)
    for field, value in (
        ("actual_admission_evidence_root", "ff" * 32),
        ("bundle_evidence_root", ("AB" * 32)),
        ("qualification_count", True),
        ("candidate_receipt_cbor_hex", ""),
        ("candidate_receipt_cbor_hex", "0"),
        ("candidate_receipt_cbor_hex", "00"),
    ):
        tampered = deepcopy(candidate)
        tampered[field] = value
        candidate_tampers.append(tampered)
    reordered = deepcopy(candidate)
    reordered["ordered_predicate_rows"][0:2] = reversed(
        reordered["ordered_predicate_rows"][0:2]
    )
    candidate_tampers.append(reordered)
    bool_alias = deepcopy(candidate)
    bool_alias["ordered_predicate_rows"][0][0] = True
    candidate_tampers.append(bool_alias)

    for tampered in candidate_tampers:
        with pytest.raises(ARTIFACT.Q05BActualArtifactError):
            ARTIFACT.validate_stage8_candidate_registry_v1(
                tampered,
                admission_root,
            )

    derived_tampers: list[dict[str, object]] = []
    missing_derived = deepcopy(derived)
    missing_derived.pop("resource_evidence_root")
    derived_tampers.append(missing_derived)
    for field, value in (
        ("artifact_set_root", "A3" * 32),
        ("host_execution_binding_root", "f1" * 32),
        ("qualification_mask", True),
        ("final_receipt_cbor_hex", ""),
        ("final_receipt_cbor_hex", "00"),
    ):
        tampered = deepcopy(derived)
        tampered[field] = value
        derived_tampers.append(tampered)
    reordered_derived = deepcopy(derived)
    reordered_derived["ordered_predicate_rows"] = list(
        reversed(reordered_derived["ordered_predicate_rows"])
    )
    derived_tampers.append(reordered_derived)
    derived_bool_alias = deepcopy(derived)
    derived_bool_alias["ordered_predicate_rows"][0][0] = True
    derived_tampers.append(derived_bool_alias)

    for tampered in derived_tampers:
        with pytest.raises(ARTIFACT.Q05BActualArtifactError):
            ARTIFACT.validate_stage9_derived_registry_v1(
                tampered,
                candidate,
                admission_root,
            )


def test_stage10_adapter_crosses_stage9_receipts_roots_and_artifact_bytes() -> None:
    candidate, derived = _synthetic_stage8_stage9_registries_v1("91" * 20)
    artifact_payload = TOOL._canonical_json_bytes({"strict": "artifact"})
    stage9_evidence = {
        "actual_admission_evidence_root": derived[
            "actual_admission_evidence_root"
        ],
        "artifact_set_root": derived["artifact_set_root"],
        "bundle_evidence_root": derived["bundle_evidence_root"],
        "candidate_receipt_root": derived["candidate_receipt_root"],
        "closed_q1_authority": derived["closed_q1_authority"],
        "final_receipt_root": derived["final_receipt_root"],
        "host_execution_binding_root": derived[
            "host_execution_binding_root"
        ],
        "isolation_evidence_root": derived["isolation_evidence_root"],
        "ordered_predicate_rows": derived["ordered_predicate_rows"],
        "predicate20_added_after_candidate_replay": True,
        "qualification_count": 20,
        "qualification_mask": 0xFFFFF,
        "resource_evidence_root": derived["resource_evidence_root"],
        "strict_derived_cross_equal": True,
    }
    stage9 = {
        "candidate_receipt_hex": derived["candidate_receipt_cbor_hex"],
        "evidence": stage9_evidence,
        "final_receipt_hex": derived["final_receipt_cbor_hex"],
        "qualification_count": 20,
        "qualification_mask": 0xFFFFF,
    }
    stage10_evidence = {
        "actual_admission_evidence_root": derived[
            "actual_admission_evidence_root"
        ],
        "artifact_length": len(artifact_payload),
        "artifact_set_root": derived["artifact_set_root"],
        "artifact_sha256": sha256(artifact_payload).hexdigest(),
        "bundle_evidence_root": derived["bundle_evidence_root"],
        "candidate_receipt_hex": derived["candidate_receipt_cbor_hex"],
        "candidate_receipt_root": derived["candidate_receipt_root"],
        "closed_q1_authority": derived["closed_q1_authority"],
        "final_receipt_hex": derived["final_receipt_cbor_hex"],
        "final_receipt_root": derived["final_receipt_root"],
        "host_execution_binding_root": derived[
            "host_execution_binding_root"
        ],
        "isolation_evidence_root": derived["isolation_evidence_root"],
        "ordered_predicate_rows": derived["ordered_predicate_rows"],
        "qualification_count": 20,
        "qualification_mask": 0xFFFFF,
        "resource_evidence_root": derived["resource_evidence_root"],
        "strict_replay_equal": True,
    }
    stage10 = {
        "candidate_receipt_hex": derived["candidate_receipt_cbor_hex"],
        "evidence": stage10_evidence,
        "final_receipt_hex": derived["final_receipt_cbor_hex"],
        "qualification_count": 20,
        "qualification_mask": 0xFFFFF,
    }
    assert TOOL._validate_actual_stage10_adapter_v1(
        stage10,
        stage9,
        derived,
        artifact_payload,
    ) is stage10

    for field, value in (
        ("artifact_sha256", "00" * 32),
        ("actual_admission_evidence_root", "01" * 32),
        ("bundle_evidence_root", "02" * 32),
        ("host_execution_binding_root", "03" * 32),
        ("isolation_evidence_root", "04" * 32),
        ("resource_evidence_root", "05" * 32),
        ("qualification_count", True),
    ):
        tampered = deepcopy(stage10)
        tampered["evidence"][field] = value
        with pytest.raises(TOOL.Q05BDualSupervisorError) as error:
            TOOL._validate_actual_stage10_adapter_v1(
                tampered,
                stage9,
                derived,
                artifact_payload,
            )
        assert error.value.code == TOOL.FAIL_ARTIFACT

    malformed = deepcopy(stage10)
    malformed["final_receipt_hex"] = "00"
    with pytest.raises(TOOL.Q05BDualSupervisorError):
        TOOL._validate_actual_stage10_adapter_v1(
            malformed,
            stage9,
            derived,
            artifact_payload,
        )


def test_final_delivery_identity_binds_ten_stages_admission_and_four_live_roots(
    tmp_path: Path,
) -> None:
    candidate, derived = _synthetic_stage8_stage9_registries_v1("92" * 20)
    live = _synthetic_final_delivery_live_surfaces_v1()
    path = (tmp_path / "final-delivery.json").resolve()
    artifact_value = {"synthetic": "strict-final-delivery"}
    payload = TOOL._canonical_json_bytes(artifact_value)
    handle = TOOL.atomic_publish_canonical_artifact_v1(path, artifact_value)
    try:
        anchored = TOOL.read_anchored_published_artifact_v1(
            handle,
            path,
            payload,
        )
        stage_rows = [
            [stage_id, stage_name, f"{stage_id:02x}" * 32]
            for stage_id, stage_name in TOOL.ACTUAL_ORCHESTRATION_STAGE_REGISTRY
        ]
        admission_section = {
            "actual_admission_evidence_root": candidate[
                "actual_admission_evidence_root"
            ],
            "pre_artifact_live_marker_replay": live[
                "PRE_ARTIFACT_ASSEMBLY"
            ],
            "root_registry": {
                "pre_artifact_live_marker_replay_root": live[
                    "PRE_ARTIFACT_ASSEMBLY"
                ]["live_marker_replay_root"]
            },
        }
        identity, delivery_root = TOOL.build_actual_final_delivery_identity_v1(
            source_commit="92" * 20,
            artifact_path=path,
            ordered_stage_root_rows=stage_rows,
            actual_admission_section=admission_section,
            stage_10_live_marker_replay=live["STAGE_10_BEFORE_EVIDENCE"],
            prepublication_live_marker_replay=live[
                "PREPUBLICATION_AFTER_STAGE10"
            ],
            postpublication_live_marker_replay=live[
                "POSTPUBLICATION_AFTER_ANCHORED_ARTIFACT_REPLAY"
            ],
            published_handle=handle,
            final_delivery_bytes=anchored,
            first_anchored_replay_bytes=anchored,
            artifact_set_root=derived["artifact_set_root"],
            candidate_receipt_root=derived["candidate_receipt_root"],
            final_receipt_root=derived["final_receipt_root"],
        )
        assert identity["ordered_stage_root_rows"] == stage_rows
        assert identity["actual_admission_evidence_root"] == candidate[
            "actual_admission_evidence_root"
        ]
        assert {
            identity["pre_artifact_live_marker_replay_root"],
            identity["stage_10_live_marker_replay_root"],
            identity["prepublication_live_marker_replay_root"],
            identity["postpublication_live_marker_replay_root"],
        } == {
            row["live_marker_replay_root"] for row in live.values()
        }
        assert delivery_root == sha256(
            TOOL.ACTUAL_FINAL_DELIVERY_ROOT_DOMAIN
            + TOOL._canonical_json_bytes(identity)
        ).hexdigest()

        for tampered_rows in (
            stage_rows[:-1],
            [stage_rows[1], stage_rows[0], *stage_rows[2:]],
            [*stage_rows[:-1], [10, stage_rows[-1][1], "AB" * 32]],
        ):
            with pytest.raises(TOOL.Q05BDualSupervisorError) as error:
                TOOL.build_actual_final_delivery_identity_v1(
                    source_commit="92" * 20,
                    artifact_path=path,
                    ordered_stage_root_rows=tampered_rows,
                    actual_admission_section=admission_section,
                    stage_10_live_marker_replay=live[
                        "STAGE_10_BEFORE_EVIDENCE"
                    ],
                    prepublication_live_marker_replay=live[
                        "PREPUBLICATION_AFTER_STAGE10"
                    ],
                    postpublication_live_marker_replay=live[
                        "POSTPUBLICATION_AFTER_ANCHORED_ARTIFACT_REPLAY"
                    ],
                    published_handle=handle,
                    final_delivery_bytes=anchored,
                    first_anchored_replay_bytes=anchored,
                    artifact_set_root=derived["artifact_set_root"],
                    candidate_receipt_root=derived["candidate_receipt_root"],
                    final_receipt_root=derived["final_receipt_root"],
                )
            assert error.value.code == TOOL.FAIL_ARTIFACT
    finally:
        TOOL.close_anchored_published_artifact_v1(handle)


@pytest.mark.parametrize(
    "changed_checkpoint",
    [
        "PRE_ARTIFACT_ASSEMBLY",
        "STAGE_10_BEFORE_EVIDENCE",
        "PREPUBLICATION_AFTER_STAGE10",
        "POSTPUBLICATION_AFTER_ANCHORED_ARTIFACT_REPLAY",
    ],
)
def test_final_delivery_live_authority_mismatch_rolls_back_owned_artifact(
    tmp_path: Path,
    changed_checkpoint: str,
) -> None:
    candidate, derived = _synthetic_stage8_stage9_registries_v1("93" * 20)
    live = _synthetic_final_delivery_live_surfaces_v1()
    live[changed_checkpoint] = _different_live_authority_v1(
        live[changed_checkpoint]
    )
    path = (tmp_path / f"live-mismatch-{changed_checkpoint}.json").resolve()
    artifact_value = {"synthetic": changed_checkpoint}
    payload = TOOL._canonical_json_bytes(artifact_value)
    handle = TOOL.atomic_publish_canonical_artifact_v1(path, artifact_value)
    anchored = TOOL.read_anchored_published_artifact_v1(handle, path, payload)
    stage_rows = [
        [stage_id, stage_name, f"{stage_id:02x}" * 32]
        for stage_id, stage_name in TOOL.ACTUAL_ORCHESTRATION_STAGE_REGISTRY
    ]
    admission_section = {
        "actual_admission_evidence_root": candidate[
            "actual_admission_evidence_root"
        ],
        "pre_artifact_live_marker_replay": live["PRE_ARTIFACT_ASSEMBLY"],
        "root_registry": {
            "pre_artifact_live_marker_replay_root": live[
                "PRE_ARTIFACT_ASSEMBLY"
            ]["live_marker_replay_root"]
        },
    }
    try:
        with pytest.raises(TOOL.Q05BDualSupervisorError) as error:
            TOOL.build_actual_final_delivery_identity_v1(
                source_commit="93" * 20,
                artifact_path=path,
                ordered_stage_root_rows=stage_rows,
                actual_admission_section=admission_section,
                stage_10_live_marker_replay=live[
                    "STAGE_10_BEFORE_EVIDENCE"
                ],
                prepublication_live_marker_replay=live[
                    "PREPUBLICATION_AFTER_STAGE10"
                ],
                postpublication_live_marker_replay=live[
                    "POSTPUBLICATION_AFTER_ANCHORED_ARTIFACT_REPLAY"
                ],
                published_handle=handle,
                final_delivery_bytes=anchored,
                first_anchored_replay_bytes=anchored,
                artifact_set_root=derived["artifact_set_root"],
                candidate_receipt_root=derived["candidate_receipt_root"],
                final_receipt_root=derived["final_receipt_root"],
            )
        assert error.value.code == TOOL.FAIL_ACTUAL_ADMISSION
        TOOL.rollback_anchored_published_artifact_v1(handle, path, payload)
        assert not path.exists()
    finally:
        TOOL.close_anchored_published_artifact_v1(handle)


def test_invalid_artifact_is_rejected_before_postpublication_live_or_publish(
    tmp_path: Path,
) -> None:
    artifact = (tmp_path / "postpublication-live-failure.json").resolve()
    backend = _MockActualBackend(
        fail_live_checkpoint="POSTPUBLICATION_AFTER_ANCHORED_ARTIFACT_REPLAY"
    )
    backend.artifact_path = artifact
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.orchestrate_actual_with_backend_v1(
            "35" * 20,
            artifact,
            backend,
        )
    assert failure.value.code == TOOL.FAIL_ARTIFACT
    assert backend.observed == list(range(1, 10))
    assert (
        "POSTPUBLICATION_AFTER_ANCHORED_ARTIFACT_REPLAY"
        not in backend.live_checkpoints
    )
    assert not artifact.exists()


def test_atomic_publisher_internal_postlink_fault_rolls_back_owned_inode(
    tmp_path: Path,
) -> None:
    artifact = (tmp_path / "publisher-internal-postlink-fault.json").resolve()

    def fail_after_link(checkpoint: str) -> None:
        assert checkpoint == "AFTER_FINAL_LINK_VALIDATED_BEFORE_TEMP_UNLINK"
        raise RuntimeError("injected internal postlink failure")

    with pytest.raises(RuntimeError, match="internal postlink failure"):
        TOOL.atomic_publish_canonical_artifact_v1(
            artifact,
            {"publisher": "internal-fault"},
            fault_hook=fail_after_link,
        )
    assert not artifact.exists()
    assert not tuple(tmp_path.iterdir())


def test_owned_handle_rejects_same_bytes_replacement(
    tmp_path: Path,
) -> None:
    artifact = (tmp_path / "owned-handle-replaced.json").resolve()
    value = {"owned": "same-bytes-replacement"}
    payload = TOOL._canonical_json_bytes(value)
    handle = TOOL.atomic_publish_canonical_artifact_v1(artifact, value)
    try:
        artifact.unlink()
        artifact.write_bytes(payload)
        artifact.chmod(0o444)
        replacement_inode = artifact.stat().st_ino
        assert replacement_inode != handle.file_inode
        with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
            TOOL.read_anchored_published_artifact_v1(
                handle,
                artifact,
                payload,
            )
        assert failure.value.code == TOOL.FAIL_ARTIFACT
        assert artifact.exists()
        assert artifact.stat().st_ino == replacement_inode
    finally:
        TOOL.close_anchored_published_artifact_v1(handle)


@pytest.mark.parametrize("same_bytes", [True, False])
def test_final_delivery_owned_replay_rejects_replacement(
    tmp_path: Path,
    same_bytes: bool,
) -> None:
    artifact = (
        tmp_path / f"final-delivery-callback-replacement-{same_bytes}.json"
    ).resolve()
    value = {"owned": "final-replay-replacement"}
    payload = TOOL._canonical_json_bytes(value)
    handle = TOOL.atomic_publish_canonical_artifact_v1(artifact, value)
    try:
        first = TOOL.read_anchored_published_artifact_v1(
            handle,
            artifact,
            payload,
        )
        replacement = (
            first
            if same_bytes
            else TOOL._canonical_json_bytes({"external_replacement": True})
        )
        artifact.unlink()
        artifact.write_bytes(replacement)
        artifact.chmod(0o444)
        replacement_inode = artifact.stat().st_ino
        assert replacement_inode != handle.file_inode
        with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
            TOOL.read_anchored_published_artifact_v1(
                handle,
                artifact,
                payload,
            )
        assert failure.value.code == TOOL.FAIL_ARTIFACT
        assert artifact.read_bytes() == replacement
        assert artifact.stat().st_ino == replacement_inode
    finally:
        TOOL.close_anchored_published_artifact_v1(handle)


def test_invalid_artifact_does_not_invoke_unowned_publisher(
    tmp_path: Path,
) -> None:
    artifact = (tmp_path / "unowned-publisher-residual.json").resolve()
    backend = _MockActualBackend()
    backend.artifact_path = artifact

    calls: list[object] = []

    def publish_unowned_then_raise(path: Path, value: object) -> None:
        calls.append(value)
        path.write_bytes(TOOL._canonical_json_bytes(value))
        path.chmod(0o444)
        raise RuntimeError("injected unowned publisher failure")

    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.orchestrate_actual_with_backend_v1(
            "37" * 20,
            artifact,
            backend,
            publisher=publish_unowned_then_raise,
        )
    assert failure.value.code == TOOL.FAIL_ARTIFACT
    assert backend.observed == list(range(1, 10))
    assert calls == []
    assert not artifact.exists()


def test_actual_stage8_rejects_hex_shaped_nonreceipt() -> None:
    source_commit = "56" * 20
    forged = TOOL.actual_stage_evidence_v1(
        8,
        TOOL.ACTUAL_ORCHESTRATION_STAGE_REGISTRY[7][1],
        source_commit,
        {"forged": True},
        qualification_count=19,
        qualification_mask=0x7FFFF,
        candidate_receipt_hex="aa",
        final_receipt_hex=None,
    )
    with pytest.raises(TOOL.Q05BDualSupervisorError) as error:
        TOOL.validate_actual_stage_evidence_v1(
            forged,
            8,
            TOOL.ACTUAL_ORCHESTRATION_STAGE_REGISTRY[7][1],
            source_commit,
        )
    assert error.value.code == TOOL.FAIL_POLICY


def test_atomic_artifact_primitive_is_noreplace_mode444_and_canonical(tmp_path: Path) -> None:
    artifact = tmp_path / "evidence.json"
    value = {"z": 1, "a": [False, None]}
    handle = TOOL.atomic_publish_canonical_artifact_v1(artifact, value)
    payload = TOOL._canonical_json_bytes(value)
    assert TOOL.read_anchored_published_artifact_v1(
        handle,
        artifact,
        payload,
    ) == payload
    assert TOOL.read_published_canonical_artifact_v1(artifact, payload) == payload
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o444
    TOOL.close_anchored_published_artifact_v1(handle)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as second:
        TOOL.atomic_publish_canonical_artifact_v1(artifact, value)
    assert second.value.code == TOOL.FAIL_ARTIFACT


def test_published_artifact_replay_rejects_hardlink_and_symlink(
    tmp_path: Path,
) -> None:
    payload = TOOL._canonical_json_bytes({"a": 1})
    artifact = tmp_path / "artifact.json"
    handle = TOOL.atomic_publish_canonical_artifact_v1(artifact, {"a": 1})
    TOOL.close_anchored_published_artifact_v1(handle)
    alias = tmp_path / "alias.json"
    os.link(artifact, alias)
    with pytest.raises(TOOL.Q05BDualSupervisorError):
        TOOL.read_published_canonical_artifact_v1(artifact, payload)
    alias.unlink()
    artifact.unlink()
    target = tmp_path / "target.json"
    target.write_bytes(payload)
    target.chmod(0o444)
    artifact.symlink_to(target)
    with pytest.raises(TOOL.Q05BDualSupervisorError):
        TOOL.read_published_canonical_artifact_v1(artifact, payload)


def test_atomic_artifact_validation_failure_removes_published_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "evidence.json"
    original_stat = TOOL.os.stat

    def invalid_final_stat(path, *args, **kwargs):
        value = original_stat(path, *args, **kwargs)
        if path == artifact.name and kwargs.get("dir_fd") is not None:
            class InvalidFinal:
                st_mode = stat.S_IFDIR | 0o444
                st_size = value.st_size
                st_dev = value.st_dev
                st_ino = value.st_ino
                st_nlink = value.st_nlink

            return InvalidFinal()
        return value

    monkeypatch.setattr(TOOL.os, "stat", invalid_final_stat)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.atomic_publish_canonical_artifact_v1(artifact, {"a": 1})
    assert failure.value.code == TOOL.FAIL_ARTIFACT
    assert not artifact.exists()
    assert not tuple(tmp_path.iterdir())


def test_atomic_artifact_rejects_close_to_link_temporary_name_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "evidence.json"
    original_link = TOOL.os.link
    replacement = TOOL._canonical_json_bytes({"b": 2})
    assert len(replacement) == len(TOOL._canonical_json_bytes({"a": 1}))

    def swap_then_link(source, destination, **kwargs):
        directory = kwargs["src_dir_fd"]
        os.unlink(source, dir_fd=directory)
        replacement_fd = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory,
        )
        try:
            assert os.write(replacement_fd, replacement) == len(replacement)
            os.fsync(replacement_fd)
            os.fchmod(replacement_fd, 0o444)
            os.fsync(replacement_fd)
        finally:
            os.close(replacement_fd)
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(TOOL.os, "link", swap_then_link)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
        TOOL.atomic_publish_canonical_artifact_v1(artifact, {"a": 1})
    assert failure.value.code == TOOL.FAIL_ARTIFACT
    assert not artifact.exists()
    assert not tuple(tmp_path.iterdir())


@pytest.fixture(scope="module")
def odd_partition_evidence():
    limits = CAPACITY.PreflightLimitsV1(maximum_ast_node_count=3)
    snapshot = SNAPSHOT.build_q1_partition_snapshot_v1(1, limits=limits)
    records = PROJECTION.records_from_partition_snapshot_v1(snapshot)
    coverage = COVERAGE.build_q1_semantic_coverage_v1(snapshot)
    evidence = WIRE.node3_partition_evidence_v1(snapshot, records, coverage)
    replay = HOST.strict_replay_partition_streams_v1(evidence)
    assert replay.input_signature_id == 1
    assert len(replay.canonical_object()) == 10
    return evidence


def _swap_first_two_frames(blob: bytes) -> bytes:
    frames = []
    offset = 0
    while offset < len(blob):
        length = int.from_bytes(blob[offset : offset + 4], "big")
        end = offset + 4 + length
        frames.append(blob[offset:end])
        offset = end
    assert len(frames) >= 2
    frames[0], frames[1] = frames[1], frames[0]
    return b"".join(frames)


def test_host_rejects_record_set_duplicate_accepted_by_candidate_decoder(
    odd_partition_evidence,
) -> None:
    record_set = list(odd_partition_evidence.record_set_object)
    programs = list(record_set[4])
    programs[1] = programs[0]
    record_set[4] = tuple(programs)
    tampered = replace(
        odd_partition_evidence,
        record_set_object=tuple(record_set),
    )
    with pytest.raises(HOST.Q05BHostReplayError) as error:
        HOST.strict_replay_partition_streams_v1(tampered)
    assert error.value.code == HOST.FAIL_HOST_STREAM


def test_host_rejects_external_sort_run_manifest_hash_flip(
    odd_partition_evidence,
) -> None:
    streams = list(odd_partition_evidence.stream_rows)
    stream = list(streams[0])
    trace = list(stream[3])
    manifests = list(trace[4])
    manifest = list(manifests[0])
    digest = bytearray(manifest[8])
    digest[0] ^= 1
    manifest[8] = bytes(digest)
    manifests[0] = tuple(manifest)
    trace[4] = tuple(manifests)
    stream[3] = tuple(trace)
    streams[0] = tuple(stream)
    tampered = replace(odd_partition_evidence, stream_rows=tuple(streams))
    with pytest.raises(HOST.Q05BHostReplayError) as error:
        HOST.strict_replay_partition_streams_v1(tampered)
    assert error.value.code == HOST.FAIL_HOST_STREAM


def test_host_rejects_framed_record_swap_without_updated_commitments(
    odd_partition_evidence,
) -> None:
    streams = list(odd_partition_evidence.stream_rows)
    stream = list(streams[0])
    blobs = list(stream[2])
    blobs[0] = _swap_first_two_frames(blobs[0])
    stream[2] = tuple(blobs)
    streams[0] = tuple(stream)
    tampered = replace(odd_partition_evidence, stream_rows=tuple(streams))
    with pytest.raises(HOST.Q05BHostReplayError) as error:
        HOST.strict_replay_partition_streams_v1(tampered)
    assert error.value.code == HOST.FAIL_HOST_STREAM


def test_git_blob_snapshot_uses_commit_bytes_not_dirty_worktree(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    project = repository / "Hegel Machine"
    project.mkdir()
    (project / "a.txt").write_text("committed\n", encoding="ascii")
    subprocess.run(
        ["git", "-C", str(repository), "add", "Hegel Machine/a.txt"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Q05B Test",
            "-c",
            "user.email=q05b@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    assert TOOL.git_project_prefix_v1(project) == "Hegel Machine/"
    rows = TOOL.git_blob_manifest_v1(project, commit, ("a.txt",))
    assert len(TOOL.git_blob_manifest_sha256_v1(rows)) == 64
    raw_digest = sha256()
    path_bytes = b"a.txt"
    blob = b"committed\n"
    raw_digest.update(len(path_bytes).to_bytes(4, "big"))
    raw_digest.update(path_bytes)
    raw_digest.update(len(blob).to_bytes(8, "big"))
    raw_digest.update(blob)
    assert TOOL.git_source_identity_digest_v1(project, commit, rows) == raw_digest.hexdigest()
    assert TOOL.git_source_identity_digest_v1(
        project, commit, rows
    ) != TOOL.git_blob_manifest_sha256_v1(rows)
    closure = TOOL.git_source_object_closure_evidence_v1(
        project,
        commit,
        ("a.txt",),
    )
    assert closure["project_tree_prefix"] == "Hegel Machine"
    assert closure["commit"] == commit
    assert len(closure["tree_object_rows"]) == 2
    commit_payload = bytes.fromhex(closure["commit_payload_hex"])
    assert sha256(commit_payload).hexdigest() == closure["commit_payload_sha256"]
    assert all(
        len(object_id) == 40 and bytes.fromhex(payload_hex)
        for object_id, payload_hex in closure["tree_object_rows"]
    )
    (project / "a.txt").write_text("dirty\n", encoding="ascii")
    destination = tmp_path / "snapshot"
    TOOL.materialize_git_blob_snapshot_v1(project, commit, rows, destination)
    assert (destination / "a.txt").read_text(encoding="ascii") == "committed\n"
    assert stat.S_IMODE((destination / "a.txt").stat().st_mode) == 0o444


def test_snapshot_tree_replay_rejects_special_entries(tmp_path: Path) -> None:
    root = tmp_path / "snapshot-tree"
    root.mkdir()
    os.mkfifo(root / "unexpected-fifo")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root, flags)
    try:
        with pytest.raises(TOOL.Q05BDualSupervisorError) as special:
            TOOL._snapshot_file_paths_v1(descriptor)
        assert special.value.code == TOOL.FAIL_SOURCE
    finally:
        os.close(descriptor)


def test_sealed_tree_identity_binds_root_directories_and_files(tmp_path: Path) -> None:
    root = tmp_path / "sealed-output"
    (root / "neutral").mkdir(parents=True)
    (root / "preimages").mkdir()
    payloads = {
        "neutral/a.cbor": b"a",
        "preimages/b.cbor": b"b",
    }
    for relative, payload in payloads.items():
        path = root / relative
        path.write_bytes(payload)
        path.chmod(0o444)
    TOOL.seal_directory_tree_read_only_v1(root)
    identity = TOOL.sealed_tree_identity_v1(root, tuple(sorted(payloads)))
    assert identity["schema_version"] == TOOL.SEALED_TREE_IDENTITY_SCHEMA_VERSION
    assert identity["root_path"] == root.as_posix()
    assert identity["root_nlink"] >= 2
    assert [row[0] for row in identity["directory_rows"]] == [
        "neutral",
        "preimages",
    ]
    assert [row[0] for row in identity["file_rows"]] == sorted(payloads)
    assert all(len(row) == 9 for row in identity["directory_rows"])
    assert all(len(row) == 11 for row in identity["file_rows"])


def test_unified_sealed_tree_identity_supports_exact_executable_snapshot_mode(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sealed-snapshot"
    root.mkdir(mode=0o700)
    executable = root / "tool.py"
    executable.write_bytes(b"#!/usr/bin/env python3\n")
    executable.chmod(0o555)
    root.chmod(0o555)
    identity = TOOL.sealed_tree_identity_v1(
        root,
        ("tool.py",),
        expected_file_modes={"tool.py": 0o555},
    )
    assert identity["file_rows"][0][6] == 0o555
    with pytest.raises(TOOL.Q05BDualSupervisorError):
        TOOL.sealed_tree_identity_v1(root, ("tool.py",))


def test_sealed_tree_identity_anchors_payload_across_root_rename_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sealed-root"
    evil = tmp_path / "evil-root"
    backup = tmp_path / "anchored-original"
    root.mkdir()
    evil.mkdir()
    (root / "value.bin").write_bytes(b"original")
    (evil / "value.bin").write_bytes(b"evil----")
    (root / "value.bin").chmod(0o444)
    (evil / "value.bin").chmod(0o444)
    root.chmod(0o555)
    evil.chmod(0o555)
    original_reader = TOOL._read_snapshot_file_v1
    swapped = False
    observed_payload: bytes | None = None

    def swap_while_reading(
        descriptor: int,
        relative: str,
    ) -> tuple[bytes, os.stat_result]:
        nonlocal observed_payload, swapped
        if not swapped:
            root.rename(backup)
            evil.rename(root)
            swapped = True
        try:
            value = original_reader(descriptor, relative)
            observed_payload = value[0]
            return value
        finally:
            if swapped:
                root.rename(evil)
                backup.rename(root)

    monkeypatch.setattr(TOOL, "_read_snapshot_file_v1", swap_while_reading)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        TOOL.sealed_tree_identity_v1(root, ("value.bin",))
    assert rejected.value.code == TOOL.FAIL_POLICY
    assert swapped is True
    assert observed_payload == b"original"


def test_snapshot_file_replay_rejects_external_hardlink(tmp_path: Path) -> None:
    root = tmp_path / "snapshot-hardlink"
    root.mkdir()
    (root / "file").write_bytes(b"payload")
    os.link(root / "file", tmp_path / "external-alias")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root, flags)
    try:
        with pytest.raises(TOOL.Q05BDualSupervisorError) as hardlink:
            TOOL._read_snapshot_file_v1(descriptor, "file")
        assert hardlink.value.code == TOOL.FAIL_SOURCE
    finally:
        os.close(descriptor)


def _cargo_detach_paths_v1(
    tmp_path: Path,
    *,
    hardlink_source: bool = False,
) -> tuple[Path, Path, Path | None]:
    source_parent = tmp_path / "target-output" / "release"
    destination_parent = tmp_path / "target-output" / "runtime-binary"
    source_parent.mkdir(parents=True)
    destination_parent.mkdir(mode=0o700)
    destination_parent.chmod(0o700)
    source = source_parent / "hegel-q1-archive-projection-oracle"
    source.write_bytes(b"ELF-Q05B-DETACH-TEST\n" * 8)
    source.chmod(0o755)
    alias: Path | None = None
    if hardlink_source:
        deps = source_parent / "deps"
        deps.mkdir()
        alias = deps / "hegel-q1-archive-projection-oracle-deadbeef"
        os.link(source, alias)
    destination = (
        destination_parent / "hegel-q1-archive-projection-oracle"
    )
    return source, destination, alias


def test_cargo_release_hardlink_is_detached_then_sealed_and_bound(
    tmp_path: Path,
) -> None:
    source, destination, alias = _cargo_detach_paths_v1(
        tmp_path, hardlink_source=True
    )
    assert alias is not None
    assert source.stat().st_ino == alias.stat().st_ino
    assert source.stat().st_nlink == 2

    detached = TOOL.detach_cargo_release_binary_v1(source, destination)
    assert detached["source_fd_before"]["nlink"] == 2
    assert detached["source_fd_before"] == detached["source_fd_after"]
    assert detached["source_fd_after"] == detached["source_path_after"]
    assert detached["source_sha256_before"] == detached["source_sha256_after"]
    assert detached["source_sha256_after"] == detached["detached_sha256"]
    assert detached["detached_fd"]["nlink"] == 1
    assert detached["detached_fd"]["mode"] == 0o755
    assert (source.stat().st_dev, source.stat().st_ino) != (
        destination.stat().st_dev,
        destination.stat().st_ino,
    )

    sealed = TOOL.seal_prebuilt_binary_v1(destination)
    assert sealed["nlink"] == 1
    assert sealed["uid"] == detached["detached_fd"]["uid"]
    assert sealed["gid"] == detached["detached_fd"]["gid"]
    assert sealed["mode"] == 0o555
    assert TOOL.validate_detached_binary_binding_v1(
        detached, sealed, source, destination
    ) == detached


def test_cargo_binary_detach_source_path_swap_fails_and_defers_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, destination, _alias = _cargo_detach_paths_v1(tmp_path)
    backup = source.with_name("held-original")
    replacement = source.with_name("replacement")
    replacement.write_bytes(source.read_bytes())
    replacement.chmod(0o755)
    real_write = TOOL.os.write
    swapped = False

    def swap_source_path(descriptor: int, payload: bytes) -> int:
        nonlocal swapped
        if not swapped:
            source.rename(backup)
            replacement.rename(source)
            swapped = True
        return real_write(descriptor, payload)

    monkeypatch.setattr(TOOL.os, "write", swap_source_path)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        TOOL.detach_cargo_release_binary_v1(source, destination)
    assert rejected.value.code == TOOL.FAIL_SOURCE
    assert "deferred outer-owned-root cleanup required" in rejected.value.detail
    assert swapped is True
    assert destination.exists()
    assert tuple(destination.parent.iterdir()) == (destination,)
    source.unlink()
    backup.rename(source)


def test_cargo_binary_detach_source_hardlink_tamper_fails_and_defers_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, destination, alias = _cargo_detach_paths_v1(
        tmp_path, hardlink_source=True
    )
    assert alias is not None
    real_write = TOOL.os.write
    tampered = False

    def tamper_source_inode(descriptor: int, payload: bytes) -> int:
        nonlocal tampered
        written = real_write(descriptor, payload)
        if not tampered:
            source_descriptor = os.open(alias, os.O_WRONLY)
            try:
                os.pwrite(source_descriptor, b"X", 0)
                os.fsync(source_descriptor)
            finally:
                os.close(source_descriptor)
            tampered = True
        return written

    monkeypatch.setattr(TOOL.os, "write", tamper_source_inode)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        TOOL.detach_cargo_release_binary_v1(source, destination)
    assert rejected.value.code == TOOL.FAIL_SOURCE
    assert "deferred outer-owned-root cleanup required" in rejected.value.detail
    assert tampered is True
    assert destination.exists()
    assert tuple(destination.parent.iterdir()) == (destination,)


def test_cargo_binary_detach_partial_write_failure_defers_created_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, destination, _alias = _cargo_detach_paths_v1(tmp_path)
    real_write = TOOL.os.write
    calls = 0

    def fail_after_partial_write(descriptor: int, payload: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, payload[: len(payload) // 2])
        raise OSError(errno.EIO, "synthetic partial-write failure")

    monkeypatch.setattr(TOOL.os, "write", fail_after_partial_write)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        TOOL.detach_cargo_release_binary_v1(source, destination)
    assert rejected.value.code == TOOL.FAIL_SOURCE
    assert "deferred outer-owned-root cleanup required" in rejected.value.detail
    assert calls == 2
    assert destination.exists()
    assert destination.stat().st_size == source.stat().st_size // 2
    assert tuple(destination.parent.iterdir()) == (destination,)


def test_cargo_binary_detach_first_created_fstat_failure_defers_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, destination, _alias = _cargo_detach_paths_v1(tmp_path)
    real_fstat = TOOL.os.fstat
    injected = False

    def fail_first_created_fstat(descriptor: int):
        nonlocal injected
        try:
            descriptor_path = Path(
                os.readlink(f"/proc/self/fd/{descriptor}")
            )
        except OSError:
            descriptor_path = Path("/unavailable")
        if not injected and descriptor_path == destination:
            injected = True
            raise OSError(errno.EIO, "synthetic first created fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(TOOL.os, "fstat", fail_first_created_fstat)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        TOOL.detach_cargo_release_binary_v1(source, destination)
    assert rejected.value.code == TOOL.FAIL_SOURCE
    assert "deferred outer-owned-root cleanup required" in rejected.value.detail
    assert injected is True
    assert destination.exists()
    assert destination.stat().st_size == 0
    assert tuple(destination.parent.iterdir()) == (destination,)


def test_cargo_binary_detach_first_fstat_swap_preserves_replacement_and_defers_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, destination, _alias = _cargo_detach_paths_v1(tmp_path)
    held_created = tmp_path / "held-created-before-fstat"
    replacement = tmp_path / "replacement-before-fstat"
    replacement.write_bytes(b"replacement-before-fstat")
    replacement.chmod(0o755)
    real_fstat = TOOL.os.fstat
    destination_descriptor: int | None = None
    swapped = False

    def fail_created_fstat_and_swap(descriptor: int):
        nonlocal destination_descriptor, swapped
        try:
            descriptor_path = Path(
                os.readlink(f"/proc/self/fd/{descriptor}")
            )
        except OSError:
            descriptor_path = Path("/unavailable")
        if destination_descriptor is None and descriptor_path == destination:
            destination_descriptor = descriptor
        if descriptor == destination_descriptor:
            if not swapped:
                destination.rename(held_created)
                replacement.rename(destination)
                swapped = True
            raise OSError(errno.EIO, "synthetic permanent created fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(TOOL.os, "fstat", fail_created_fstat_and_swap)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        TOOL.detach_cargo_release_binary_v1(source, destination)
    assert rejected.value.code == TOOL.FAIL_SOURCE
    assert "deferred outer-owned-root cleanup required" in rejected.value.detail
    assert swapped is True
    assert destination.read_bytes() == b"replacement-before-fstat"
    assert held_created.exists()
    destination.unlink()
    held_created.unlink()


def test_cargo_binary_detach_destination_nlink_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, destination, _alias = _cargo_detach_paths_v1(tmp_path)
    external_alias = tmp_path / "detached-external-alias"
    real_fchmod = TOOL.os.fchmod
    linked = False

    def add_detached_hardlink(descriptor: int, mode: int) -> None:
        nonlocal linked
        real_fchmod(descriptor, mode)
        if not linked:
            os.link(destination, external_alias)
            linked = True

    monkeypatch.setattr(TOOL.os, "fchmod", add_detached_hardlink)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        TOOL.detach_cargo_release_binary_v1(source, destination)
    assert rejected.value.code == TOOL.FAIL_SOURCE
    assert "deferred outer-owned-root cleanup required" in rejected.value.detail
    assert linked is True
    assert destination.exists()
    assert external_alias.exists()
    assert destination.stat().st_ino == external_alias.stat().st_ino
    assert external_alias.stat().st_nlink == 2
    destination.unlink()
    external_alias.unlink()


def test_cargo_binary_detach_destination_double_swap_never_unlinks_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, destination, _alias = _cargo_detach_paths_v1(tmp_path)
    held_created = tmp_path / "held-created-inode"
    replacement = tmp_path / "replacement-inode"
    replacement.write_bytes(b"replacement")
    replacement.chmod(0o755)
    real_fchmod = TOOL.os.fchmod
    real_unlink = TOOL.os.unlink
    swapped = False
    unlink_calls = 0

    def swap_destination_path(descriptor: int, mode: int) -> None:
        nonlocal swapped
        real_fchmod(descriptor, mode)
        if not swapped:
            destination.rename(held_created)
            replacement.rename(destination)
            swapped = True

    def reject_destination_unlink(*args, **kwargs):
        nonlocal unlink_calls
        unlink_calls += 1
        raise AssertionError(
            f"detach helper attempted path deletion: {args!r} {kwargs!r}"
        )

    monkeypatch.setattr(TOOL.os, "fchmod", swap_destination_path)
    monkeypatch.setattr(TOOL.os, "unlink", reject_destination_unlink)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        TOOL.detach_cargo_release_binary_v1(source, destination)
    assert rejected.value.code == TOOL.FAIL_SOURCE
    assert "deferred outer-owned-root cleanup required" in rejected.value.detail
    assert swapped is True
    assert unlink_calls == 0
    assert destination.read_bytes() == b"replacement"
    assert held_created.exists()
    real_unlink(destination)
    real_unlink(held_created)


def test_detached_binary_binding_rejects_coordinated_manifest_tamper(
    tmp_path: Path,
) -> None:
    source, destination, _alias = _cargo_detach_paths_v1(tmp_path)
    detached = TOOL.detach_cargo_release_binary_v1(source, destination)
    sealed = TOOL.seal_prebuilt_binary_v1(destination)
    tampered = deepcopy(detached)
    tampered["detached_fd"]["uid"] += 1
    tampered["detached_path_identity"]["uid"] += 1
    body = dict(tampered)
    body.pop("manifest_sha256")
    tampered["manifest_sha256"] = sha256(
        TOOL._canonical_json_bytes(body)
    ).hexdigest()
    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        TOOL.validate_detached_binary_binding_v1(
            tampered, sealed, source, destination
        )
    assert rejected.value.code == TOOL.FAIL_SOURCE


def test_cargo_archive_and_index_negative_vectors_fail_closed(tmp_path: Path) -> None:
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as bundle:
        member = tarfile.TarInfo("evil-1.0.0/../../escape")
        member.size = 1
        bundle.addfile(member, io.BytesIO(b"x"))
    archive_payload = archive_buffer.getvalue()
    checksum = sha256(archive_payload).hexdigest()
    external = tmp_path / "external-cache"
    archive_path = external / "registry/cache/index.test/evil-1.0.0.crate"
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(archive_payload)
    lock_payload = (
        "version = 3\n\n"
        "[[package]]\n"
        'name = "evil"\n'
        'version = "1.0.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{checksum}"\n'
    ).encode("ascii")
    with pytest.raises(TOOL.Q05BDualSupervisorError) as archive:
        TOOL.sealed_cargo_home_material_v1(lock_payload, external)
    assert archive.value.code == TOOL.FAIL_SOURCE

    wrong_checksum = "00" * 32
    index_payload = (
        b"\x03\x02\x00\x00\x00etag: test\x001.0.0\x00"
        + json.dumps(
            {
                "name": "evil",
                "vers": "1.0.0",
                "cksum": wrong_checksum,
            },
            separators=(",", ":"),
        ).encode("ascii")
        + b"\x00"
    )
    with pytest.raises(TOOL.Q05BDualSupervisorError):
        TOOL.validate_registry_index_entry_v1(
            index_payload,
            "evil",
            "1.0.0",
            "11" * 32,
        )


def test_real_sealed_cargo_home_supports_pinned_offline_rust_tests(
    tmp_path: Path,
) -> None:
    external_cache = Path(
        "/home/erzhu419/.local/state/hegel-machine/rust-cargo-cache"
    )
    if not external_cache.is_dir():
        pytest.skip("workspace offline Cargo cache is absent")
    image = subprocess.run(
        [
            TOOL.DOCKER_EXECUTABLE,
            f"--host={TOOL.DOCKER_HOST}",
            "image",
            "inspect",
            TOOL.RUST_IMAGE,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if image.returncode != 0:
        pytest.skip("pinned Rust image is not locally present")
    cargo_home = tmp_path / "cargo-home"
    target_output = tmp_path / "target"
    cidfile = tmp_path / "rust-build.cid"
    target_output.mkdir()
    lock_payload = (ROOT / "rust/q1_archive_projection_oracle/Cargo.lock").read_bytes()
    evidence = TOOL.materialize_sealed_cargo_home_v1(
        lock_payload,
        external_cache,
        cargo_home,
    )
    assert evidence["locked_registry_package_count"] == 21
    assert evidence["cargo_home_mount"] == "READ_ONLY_PREUNPACKED"
    assert tuple(cargo_home.glob("registry/src/*/libc-*"))
    assert stat.S_IMODE(cargo_home.stat().st_mode) == 0o555
    source_digest = sha256()
    for relative in TOOL.RUST_SOURCE_ALLOWLIST:
        payload = (ROOT / relative).read_bytes()
        path_bytes = relative.encode("utf-8")
        source_digest.update(len(path_bytes).to_bytes(4, "big"))
        source_digest.update(path_bytes)
        source_digest.update(len(payload).to_bytes(8, "big"))
        source_digest.update(payload)
    test_command, _ = TOOL.rust_build_commands_v1(
        ROOT,
        cargo_home,
        target_output,
        source_digest.hexdigest(),
        cidfile,
        build_seccomp=(ROOT / TOOL.BUILD_SECCOMP_RELATIVE_PATH),
        docker_slot_row=_docker_slot_row(
            _docker_execution_authority(),
            "RUST_TEST",
        ),
    )
    try:
        completed = subprocess.run(
            test_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15 * 60,
        )
        assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    finally:
        if cidfile.exists():
            container_id = cidfile.read_text(encoding="ascii").strip()
            inspected = subprocess.run(
                [
                    TOOL.DOCKER_EXECUTABLE,
                    f"--host={TOOL.DOCKER_HOST}",
                    "inspect",
                    container_id,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert inspected.returncode == 0
            state = json.loads(inspected.stdout)[0]["State"]
            assert state["Running"] is False
            removed = subprocess.run(
                TOOL.docker_explicit_remove_command_v1(container_id),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert removed.returncode == 0
        for directory in sorted(
            (path for path in cargo_home.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o755)
        cargo_home.chmod(0o755)
        cidfile.unlink(missing_ok=True)


def _arm_mock_concrete_admission(backend, fresh_runtime_evidence):
    boundary = {
        "attempt_id": "71" * 32,
        "boundary_root": "72" * 32,
    }
    work_identity = TOOL.actual_work_root_identity_v1(
        backend.work_root,
        {"mock_stage4_layout": 1},
    )
    issue_record, root_descriptor, issued_descriptor = (
        TOOL.issue_actual_admission_marker_v1(
            backend.work_root,
            work_identity,
            boundary,
        )
    )
    backend.admission_boundary = boundary
    backend.admission_issue_record = issue_record
    backend.admission_work_root_descriptor = root_descriptor
    backend.admission_issued_marker_descriptor = issued_descriptor
    backend.admission_work_root_identity = work_identity
    backend.admission_artifact_absence = TOOL.actual_artifact_absence_evidence_v1(
        backend.artifact_path
    )
    backend.admission_fresh_runtime_evidence = fresh_runtime_evidence
    backend.admission_git_transcript_collector = (
        lambda project_root, source_commit: _synthetic_git_source_transcript(
            project_root,
            source_commit,
        )
    )
    backend.admission_consumed = False
    return {"stage3_to4_admission_issue_record": issue_record}


def _materialize_real_mount_admission_fresh_v1(
    backend: object,
    fixture_root: Path,
) -> dict[str, object]:
    """Create tiny real sealed mount objects with live dev/inode identities."""

    backend.docker_execution_authority = _docker_execution_authority(
        source_commit=backend.source_commit,
        nonce=backend.admission_nonce,
    )

    actor_files = {
        "PYTHON_ENDPOINT": ("python.marker",),
        "RUST_ENDPOINT": ("rust.marker",),
        "TRUSTED_HOST_REPLAY": (
            TOOL.BUILD_SECCOMP_RELATIVE_PATH,
            TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
        ),
    }
    snapshot_keys = {
        "PYTHON_ENDPOINT": "python_snapshot",
        "RUST_ENDPOINT": "rust_snapshot",
        "TRUSTED_HOST_REPLAY": "host_snapshot",
    }
    snapshot_rows: dict[str, dict[str, object]] = {}
    for actor_id, relatives in actor_files.items():
        root = backend.paths[snapshot_keys[actor_id]]
        root.mkdir(mode=0o700)
        for relative in relatives:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative in {
                TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
                TOOL.BUILD_SECCOMP_RELATIVE_PATH,
            }:
                path.write_bytes((ROOT / relative).read_bytes())
            else:
                path.write_bytes(
                    (actor_id + ":" + relative + "\n").encode("ascii")
                )
            path.chmod(0o444)
        TOOL.seal_directory_tree_read_only_v1(root)
        snapshot_rows[actor_id] = TOOL.sealed_tree_identity_v1(
            root, tuple(sorted(relatives))
        )

    runtime_seccomp_path = (
        backend.paths["host_snapshot"] / TOOL.RUNTIME_SECCOMP_RELATIVE_PATH
    )
    build_seccomp_path = (
        backend.paths["host_snapshot"] / TOOL.BUILD_SECCOMP_RELATIVE_PATH
    )
    runtime_seccomp = TOOL.sealed_policy_file_evidence_v1(
        runtime_seccomp_path,
        TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
    )
    build_seccomp = TOOL.sealed_policy_file_evidence_v1(
        build_seccomp_path,
        TOOL.BUILD_SECCOMP_RELATIVE_PATH,
    )
    backend.seccomp_evidence = {
        "runtime": runtime_seccomp,
        "build": build_seccomp,
    }
    backend.paths["cargo_release_binary"].parent.mkdir(
        parents=True, exist_ok=True
    )
    backend.paths["cargo_release_binary"].write_bytes(
        b"ELF-Q05B-MOUNT-TEST\n"
    )
    backend.paths["cargo_release_binary"].chmod(0o755)
    release_deps = backend.paths["cargo_release_binary"].parent / "deps"
    release_deps.mkdir()
    os.link(
        backend.paths["cargo_release_binary"],
        release_deps / "hegel-q1-archive-projection-oracle-deadbeef",
    )
    runtime_binary_parent = TOOL._private_empty_directory_v1(
        backend.paths["target_output"], "runtime-binary"
    )
    assert runtime_binary_parent == backend.paths["runtime_binary_parent"]
    backend.binary_detach_evidence = TOOL.detach_cargo_release_binary_v1(
        backend.paths["cargo_release_binary"], backend.paths["binary"]
    )
    binary_evidence = TOOL.seal_prebuilt_binary_v1(backend.paths["binary"])
    TOOL.validate_detached_binary_binding_v1(
        backend.binary_detach_evidence,
        binary_evidence,
        backend.paths["cargo_release_binary"],
        backend.paths["binary"],
    )
    backend.binary_evidence = binary_evidence

    base_root = fixture_root / "base-admission"
    base_root.mkdir(parents=True)
    base = _synthetic_actual_admission_fixture(
        base_root, backend.source_commit
    )["fresh_runtime"]
    actor_rows = []
    for index, actor_id in enumerate(
        ("PYTHON_ENDPOINT", "RUST_ENDPOINT", "TRUSTED_HOST_REPLAY"),
        start=1,
    ):
        snapshot = snapshot_rows[actor_id]
        registry_rows = [
            [row[0], row[6], row[7], row[10]]
            for row in snapshot["file_rows"]
        ]
        source_identity = {
            "schema_version": (
                "hegel-phase3a-q05b-fresh-actor-source-identity/1"
            ),
            "actor_id": actor_id,
            "source_commit": backend.source_commit,
            "project_git_prefix": "Hegel Machine/",
            "path_registry_sha256": f"{20 + index:02x}" * 32,
            "source_identity_sha256": f"{30 + index:02x}" * 32,
            "blob_count": len(snapshot["file_rows"]),
            "snapshot_file_registry_sha256": sha256(
                ADMISSION.canonical_json_bytes_v1(registry_rows)
            ).hexdigest(),
            "stage_1_source_evidence_sha256": f"{40 + index:02x}" * 32,
        }
        actor_rows.append(
            {
                "actor_id": actor_id,
                "source_identity": source_identity,
                "source_identity_root": (
                    ADMISSION.fresh_runtime_evidence_object_root_v1(
                        "ACTOR_SOURCE", actor_id, source_identity
                    )
                ),
                "snapshot_evidence": snapshot,
                "snapshot_evidence_root": (
                    ADMISSION.fresh_runtime_evidence_object_root_v1(
                        "ACTOR_SNAPSHOT", actor_id, snapshot
                    )
                ),
            }
        )
    seccomp_rows = []
    for label, relative, evidence in (
        (
            "runtime",
            TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
            runtime_seccomp,
        ),
        ("build", TOOL.BUILD_SECCOMP_RELATIVE_PATH, build_seccomp),
    ):
        seccomp_rows.append(
            {
                "label": label,
                "relative_path": relative,
                "evidence": evidence,
                "evidence_root": (
                    ADMISSION.fresh_runtime_evidence_object_root_v1(
                        "SECCOMP_POLICY", label, evidence
                    )
                ),
            }
        )
    binary_identity = TOOL._fresh_binary_identity_v1(binary_evidence)
    return ADMISSION.build_fresh_runtime_evidence_set_v1(
        backend.source_commit,
        base["image_rows"],
        actor_rows,
        base["cargo"]["material_identity"],
        base["cargo"]["snapshot_evidence"],
        base["cargo"]["tree_evidence"],
        seccomp_rows,
        binary_identity,
    )


def test_prepare_mount_source_return_fault_closes_registered_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    backend = TOOL.ConcreteQ05BActualBackendV1(
        project,
        "ab" * 20,
        project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH,
        tmp_path,
        work,
    )
    backend._create_layout_v1()
    fresh = _materialize_real_mount_admission_fresh_v1(
        backend,
        tmp_path / "store-fault-mount-authority",
    )
    backend.admission_fresh_runtime_evidence = fresh
    seccomp_path = Path(backend.seccomp_evidence["runtime"]["absolute_path"])
    command = TOOL.python_endpoint_command_v1(
        backend.paths["python_snapshot"],
        backend.paths["python_output"],
        backend.paths["python_control"],
        seccomp_path,
        docker_slot_row=_docker_slot_row(
            backend.docker_execution_authority,
            "PYTHON_ENDPOINT",
        ),
        cidfile=backend.paths["python_cidfile"],
    )

    real_open = TOOL._open_held_actor_mount_source_v1
    registered: list[TOOL.HeldActorMountSourceV1] = []

    def open_then_fail(*args, **kwargs):
        value = real_open(*args, **kwargs)
        registered.append(value)
        raise MemoryError("injected after anchored source handoff")

    monkeypatch.setattr(
        TOOL,
        "_open_held_actor_mount_source_v1",
        open_then_fail,
    )
    with pytest.raises(MemoryError, match="anchored source handoff"):
        backend._prepare_actor_mount_binding_v1(
            1,
            "PYTHON_ENDPOINT",
            command,
            ownership_slot_index=0,
        )
    assert len(registered) == 1
    assert registered[0].descriptor == -1
    assert registered[0].close_state == "CLOSED"
    assert backend.active_mount_binding_slots == [None, None, None]


def test_actor_mount_binding_rejects_self_chosen_source_and_closes_fds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    backend = TOOL.ConcreteQ05BActualBackendV1(
        project,
        "ab" * 20,
        project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH,
        tmp_path,
        work,
    )
    backend._create_layout_v1()
    fresh = _materialize_real_mount_admission_fresh_v1(
        backend, tmp_path / "mount-authority"
    )
    backend.admission_fresh_runtime_evidence = fresh
    seccomp_path = Path(backend.seccomp_evidence["runtime"]["absolute_path"])
    correct = TOOL.python_endpoint_command_v1(
        backend.paths["python_snapshot"],
        backend.paths["python_output"],
        backend.paths["python_control"],
        seccomp_path,
        docker_slot_row=_docker_slot_row(
            backend.docker_execution_authority,
            "PYTHON_ENDPOINT",
        ),
        cidfile=backend.paths["python_cidfile"],
    )
    real_fstat = os.fstat
    first_fstat = True

    def preopen_owner_drift(descriptor):
        nonlocal first_fstat
        observed = real_fstat(descriptor)
        if not first_fstat:
            return observed
        first_fstat = False
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_nlink=observed.st_nlink,
            st_uid=observed.st_uid + 1,
            st_gid=observed.st_gid,
            st_size=observed.st_size,
            st_mtime_ns=observed.st_mtime_ns,
            st_ctime_ns=observed.st_ctime_ns,
        )

    with monkeypatch.context() as preopen_drift:
        preopen_drift.setattr(TOOL.os, "fstat", preopen_owner_drift)
        with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected_preopen:
            backend._prepare_actor_mount_binding_v1(
                1, "PYTHON_ENDPOINT", correct
            )
    assert rejected_preopen.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    assert "held mount source admission differs" in rejected_preopen.value.detail
    held = backend._prepare_actor_mount_binding_v1(
        1, "PYTHON_ENDPOINT", correct
    )
    descriptors = [source.descriptor for source in (*held.sources, held.seccomp)]
    assert all(
        (fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE)
        == os.O_RDONLY
        for descriptor in descriptors
    )
    writable = next(
        source for source in held.sources if source.destination == "/control"
    )
    observed = os.fstat(writable.descriptor)
    forged_owner = SimpleNamespace(
        st_mode=observed.st_mode,
        st_dev=observed.st_dev,
        st_ino=observed.st_ino,
        st_nlink=observed.st_nlink,
        st_uid=observed.st_uid + 1,
        st_gid=observed.st_gid,
        st_size=observed.st_size,
        st_mtime_ns=observed.st_mtime_ns,
        st_ctime_ns=observed.st_ctime_ns,
    )
    with monkeypatch.context() as owner_drift:
        owner_drift.setattr(
            TOOL.os,
            "fstat",
            lambda descriptor: (
                forged_owner
                if descriptor == writable.descriptor
                else real_fstat(descriptor)
            ),
        )
        with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected_owner:
            TOOL._held_mount_source_replay_v1(
                writable,
                after_start=True,
            )
    assert rejected_owner.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    assert "identity changed" in rejected_owner.value.detail
    TOOL.close_held_actor_mount_binding_v1(held)
    assert held.closed is True
    assert all(source.descriptor == -1 for source in (*held.sources, held.seccomp))
    for descriptor in descriptors:
        with pytest.raises(OSError):
            fcntl.fcntl(descriptor, fcntl.F_GETFL)

    wrong_snapshot = tmp_path / "wrong-snapshot"
    wrong_snapshot.mkdir(mode=0o700)
    marker = wrong_snapshot / "python.marker"
    marker.write_bytes(b"PYTHON_ENDPOINT:python.marker\n")
    marker.chmod(0o444)
    TOOL.seal_directory_tree_read_only_v1(wrong_snapshot)
    wrong = TOOL.python_endpoint_command_v1(
        wrong_snapshot,
        backend.paths["python_output"],
        backend.paths["python_control"],
        seccomp_path,
        docker_slot_row=_docker_slot_row(
            backend.docker_execution_authority,
            "PYTHON_ENDPOINT",
        ),
        cidfile=backend.paths["python_cidfile"],
    )
    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        backend._prepare_actor_mount_binding_v1(
            1, "PYTHON_ENDPOINT", wrong
        )
    assert rejected.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    assert "Source differs from sealed path registry" in rejected.value.detail


def test_actor_mount_binding_rejects_prelaunch_snapshot_path_swap(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    backend = TOOL.ConcreteQ05BActualBackendV1(
        project,
        "ab" * 20,
        project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH,
        tmp_path,
        work,
    )
    backend._create_layout_v1()
    fresh = _materialize_real_mount_admission_fresh_v1(
        backend, tmp_path / "prelaunch-swap"
    )
    backend.admission_fresh_runtime_evidence = fresh
    snapshot = backend.paths["python_snapshot"]
    original = snapshot.with_name("python-original")
    os.rename(snapshot, original)
    snapshot.mkdir(mode=0o700)
    marker = snapshot / "python.marker"
    marker.write_bytes(b"PYTHON_ENDPOINT:python.marker\n")
    marker.chmod(0o444)
    TOOL.seal_directory_tree_read_only_v1(snapshot)
    command = TOOL.python_endpoint_command_v1(
        snapshot,
        backend.paths["python_output"],
        backend.paths["python_control"],
        Path(backend.seccomp_evidence["runtime"]["absolute_path"]),
        docker_slot_row=_docker_slot_row(
            backend.docker_execution_authority,
            "PYTHON_ENDPOINT",
        ),
        cidfile=backend.paths["python_cidfile"],
    )
    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        backend._prepare_actor_mount_binding_v1(
            1, "PYTHON_ENDPOINT", command
        )
    assert rejected.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    assert "held mount source admission differs" in rejected.value.detail


def test_partial_mount_preparation_close_error_still_closes_every_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    backend = TOOL.ConcreteQ05BActualBackendV1(
        project,
        "ab" * 20,
        project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH,
        tmp_path,
        work,
    )
    backend._create_layout_v1()
    fresh = _materialize_real_mount_admission_fresh_v1(
        backend, tmp_path / "partial-mount-close"
    )
    backend.admission_fresh_runtime_evidence = fresh
    seccomp_path = Path(
        backend.seccomp_evidence["runtime"]["absolute_path"]
    )
    command = TOOL.python_endpoint_command_v1(
        backend.paths["python_snapshot"],
        backend.paths["python_output"],
        backend.paths["python_control"],
        seccomp_path,
        docker_slot_row=_docker_slot_row(
            backend.docker_execution_authority,
            "PYTHON_ENDPOINT",
        ),
        cidfile=backend.paths["python_cidfile"],
    )
    opened: list[TOOL.HeldActorMountSourceV1] = []
    opened_descriptors: list[int] = []
    real_open_source = TOOL._open_held_actor_mount_source_v1

    def record_open(*args, **kwargs):
        source = real_open_source(*args, **kwargs)
        opened.append(source)
        opened_descriptors.append(source.descriptor)
        return source

    def reject_binding(*_args, **_kwargs):
        raise TOOL.Q05BDualSupervisorError(
            TOOL.FAIL_ACTUAL_ADMISSION,
            "injected binding assembly failure",
        )

    real_close = os.close
    close_counts: dict[int, int] = {}

    def flaky_close(descriptor):
        close_counts[descriptor] = close_counts.get(descriptor, 0) + 1
        if opened_descriptors and descriptor == opened_descriptors[0]:
            raise OSError(errno.EIO, "injected first close failure")
        return real_close(descriptor)

    monkeypatch.setattr(TOOL, "_open_held_actor_mount_source_v1", record_open)
    monkeypatch.setattr(
        TOOL._ACTUAL_ADMISSION,
        "build_actor_mount_binding_v1",
        reject_binding,
    )
    monkeypatch.setattr(TOOL.os, "close", flaky_close)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        backend._prepare_actor_mount_binding_v1(
            1, "PYTHON_ENDPOINT", command
        )
    assert rejected.value.code == TOOL.FAIL_POLICY
    assert "partial actor mount preparation close failed" in rejected.value.detail
    assert len(opened) == 4
    descriptor_rows = list(opened_descriptors)
    assert len(descriptor_rows) == 4
    assert set(close_counts.values()) == {1}
    assert opened[0].close_state == "UNCERTAIN_CLOSE"
    assert all(source.descriptor == -1 for source in opened)
    assert all(source.close_state == "CLOSED" for source in opened[1:])
    failed_descriptor = descriptor_rows[0]
    for descriptor in descriptor_rows:
        if descriptor == failed_descriptor:
            continue
        with pytest.raises(OSError):
            fcntl.fcntl(descriptor, fcntl.F_GETFL)
    real_close(failed_descriptor)
    replacement_path = tmp_path / "partial-close-replacement"
    replacement_path.write_bytes(b"replacement")
    replacement = os.open(replacement_path, os.O_RDONLY)
    calls_before = dict(close_counts)
    errors = TOOL._close_held_mount_sources_best_effort_v1(opened)
    assert errors and "uncertain close already recorded" in errors[0]
    assert close_counts == calls_before
    assert fcntl.fcntl(replacement, fcntl.F_GETFL) >= 0
    real_close(replacement)


@pytest.mark.parametrize("failure_position", ("first", "middle"))
def test_mount_close_error_cleans_started_actor_and_all_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_position: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    backend = TOOL.ConcreteQ05BActualBackendV1(
        project,
        "ab" * 20,
        project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH,
        tmp_path,
        work,
    )
    backend._create_layout_v1()
    fresh = _materialize_real_mount_admission_fresh_v1(
        backend, tmp_path / f"launch-close-{failure_position}"
    )
    backend.admission_fresh_runtime_evidence = fresh
    seccomp_path = Path(
        backend.seccomp_evidence["runtime"]["absolute_path"]
    )
    command = TOOL.python_endpoint_command_v1(
        backend.paths["python_snapshot"],
        backend.paths["python_output"],
        backend.paths["python_control"],
        seccomp_path,
        docker_slot_row=_docker_slot_row(
            backend.docker_execution_authority,
            "PYTHON_ENDPOINT",
        ),
        cidfile=backend.paths["python_cidfile"],
    )
    binding = backend._prepare_actor_mount_binding_v1(
        1, "PYTHON_ENDPOINT", command
    )
    descriptors = [
        source.descriptor for source in (*binding.sources, binding.seccomp)
    ]
    fail_descriptor = descriptors[
        0 if failure_position == "first" else len(descriptors) // 2
    ]
    stop = threading.Event()
    sampler = threading.Thread(target=stop.wait, daemon=True)
    sampler.start()
    registry = TOOL.sealed_actor_mount_registry_v1(1, command)
    docker_slot_row = _docker_slot_row(
        backend.docker_execution_authority,
        "PYTHON_ENDPOINT",
    )
    docker_principal = TOOL._docker_execution_principal_v1(
        command,
        backend.docker_execution_authority,
        "PYTHON_ENDPOINT",
    )
    precreate = ADMISSION.build_docker_precreate_absence_v1(
        backend.docker_execution_authority,
        docker_slot_row["slot_id"],
        _docker_absence_sample(docker_slot_row["container_name"]),
        _docker_absence_sample(docker_slot_row["container_name"]),
    )
    cid_parent = backend.paths["python_cidfile"].parent.lstat()
    actor = TOOL.HeldActorProcessV1(
        1,
        "PYTHON_ENDPOINT",
        docker_slot_row["container_name"],
        tuple(command),
        backend.paths["python_cidfile"],
        backend.paths["python_control"],
        registry,
        SimpleNamespace(),
        TOOL.BoundedPipeDrainV1(1, bytearray(), 0, False, sha256(), []),
        TOOL.BoundedPipeDrainV1(1, bytearray(), 0, False, sha256(), []),
        threading.Thread(),
        threading.Thread(),
        sampler,
        [],
        [],
        threading.Event(),
        threading.Event(),
        threading.Event(),
        threading.Lock(),
        None,
        (
            cid_parent.st_dev,
            cid_parent.st_ino,
            stat.S_IMODE(cid_parent.st_mode),
            cid_parent.st_nlink,
        ),
        None,
        [],
        TOOL.sealed_policy_file_evidence_v1(
            seccomp_path,
            TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
        ),
        backend.docker_execution_authority["manifest_sha256"],
        docker_slot_row,
        docker_principal["ownership_label_root"],
        precreate,
        docker_principal,
    )
    backend.actor_starter = lambda *_args, **_kwargs: actor
    cleaned: list[TOOL.HeldActorProcessV1] = []

    def cleanup(observed, _runner):
        cleaned.append(observed)
        stop.set()
        return ()

    monkeypatch.setattr(TOOL, "_abort_held_actor_cleanup_v1", cleanup)
    real_close = os.close
    close_counts: dict[int, int] = {}

    def flaky_close(descriptor):
        close_counts[descriptor] = close_counts.get(descriptor, 0) + 1
        if descriptor == fail_descriptor:
            raise OSError(errno.EIO, "injected mount close failure")
        return real_close(descriptor)

    with monkeypatch.context() as close_fault:
        close_fault.setattr(TOOL.os, "close", flaky_close)
        with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
            backend._launch_prepared_actor_mount_binding_v1(
                binding,
                "PYTHON_ENDPOINT",
                backend.paths["python_cidfile"],
                backend.paths["python_control"],
            )
    assert rejected.value.code == TOOL.FAIL_POLICY
    assert "mount authority close failed" in rejected.value.detail
    assert set(close_counts) == set(descriptors)
    assert set(close_counts.values()) == {1}
    assert cleaned == [actor]
    assert binding.closed is False
    assert all(
        source.descriptor == -1
        for source in (*binding.sources, binding.seccomp)
    )
    failed_source = next(
        source
        for source in (*binding.sources, binding.seccomp)
        if source.close_state == "UNCERTAIN_CLOSE"
    )
    assert failed_source.destination in {
        source.destination for source in (*binding.sources, binding.seccomp)
    }
    assert sum(
        source.close_state == "CLOSED"
        for source in (*binding.sources, binding.seccomp)
    ) == len(descriptors) - 1
    for descriptor in descriptors:
        if descriptor == fail_descriptor:
            assert fcntl.fcntl(descriptor, fcntl.F_GETFL) >= 0
        else:
            with pytest.raises(OSError):
                fcntl.fcntl(descriptor, fcntl.F_GETFL)
    calls_before = dict(close_counts)
    with pytest.raises(TOOL.Q05BDualSupervisorError) as no_retry:
        TOOL.close_held_actor_mount_binding_v1(binding)
    assert no_retry.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    assert close_counts == calls_before
    real_close(fail_descriptor)


def test_mount_close_success_does_not_probe_reused_numeric_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    replacement_path = tmp_path / "replacement"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    replacement_path.write_bytes(b"replacement")
    first_fd = os.open(first_path, os.O_RDONLY)
    second_fd = os.open(second_path, os.O_RDONLY)

    def source(path: Path, descriptor: int, destination: str):
        return TOOL.HeldActorMountSourceV1(
            destination,
            path,
            descriptor,
            {},
            None,
            (0,) * 9,
        )

    sources = (
        source(first_path, first_fd, "/first"),
        source(second_path, second_fd, "/second"),
    )
    real_close = os.close
    replacement_fd: list[int] = []
    close_counts: dict[int, int] = {}

    def close_then_reuse(descriptor):
        close_counts[descriptor] = close_counts.get(descriptor, 0) + 1
        real_close(descriptor)
        if descriptor == first_fd:
            replacement_fd.append(os.open(replacement_path, os.O_RDONLY))

    monkeypatch.setattr(TOOL.os, "close", close_then_reuse)
    assert TOOL._close_held_mount_sources_best_effort_v1(sources) == ()
    assert replacement_fd == [first_fd]
    assert close_counts == {first_fd: 1, second_fd: 1}
    assert fcntl.fcntl(replacement_fd[0], fcntl.F_GETFL) >= 0
    assert TOOL._close_held_mount_sources_best_effort_v1(sources) == ()
    assert close_counts == {first_fd: 1, second_fd: 1}
    assert fcntl.fcntl(replacement_fd[0], fcntl.F_GETFL) >= 0
    real_close(replacement_fd[0])


@pytest.mark.parametrize("swap_target", ("snapshot", "seccomp"))
def test_actor_mount_binding_rejects_starter_swap_restore_and_spends_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_target: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    backend = TOOL.ConcreteQ05BActualBackendV1(
        project,
        "ab" * 20,
        project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH,
        tmp_path,
        work,
    )
    backend._create_layout_v1()
    fresh = _materialize_real_mount_admission_fresh_v1(
        backend, tmp_path / f"starter-{swap_target}-swap"
    )
    seccomp = backend.seccomp_evidence["runtime"]
    seccomp_path = Path(seccomp["absolute_path"])
    backend.cargo_evidence = {"sealed": True}
    backend.planned_commands = {
        "python": TOOL.python_endpoint_command_v1(
            backend.paths["python_snapshot"],
            backend.paths["python_output"],
        backend.paths["python_control"],
        seccomp_path,
        docker_slot_row=_docker_slot_row(
            backend.docker_execution_authority,
            "PYTHON_ENDPOINT",
        ),
        cidfile=backend.paths["python_cidfile"],
        ),
        "rust": TOOL.rust_runtime_command_v1(
            backend.paths["binary"],
            backend.paths["rust_output"],
            backend.paths["rust_control"],
            seccomp_path,
            docker_slot_row=_docker_slot_row(
                backend.docker_execution_authority,
                "RUST_ENDPOINT",
            ),
            cidfile=backend.paths["rust_cidfile"],
        ),
    }
    backend.completed_stage = 3
    context = _arm_mock_concrete_admission(backend, fresh)
    monkeypatch.setattr(
        TOOL,
        "collect_fresh_runtime_evidence_set_v1",
        lambda *_args, **_kwargs: fresh,
    )
    prepared_bindings: list[TOOL.HeldActorMountBindingV1] = []
    prepared_descriptors: list[int] = []
    prepare = backend._prepare_actor_mount_binding_v1

    def record_prepared_binding(*args, **kwargs):
        binding = prepare(*args, **kwargs)
        prepared_bindings.append(binding)
        prepared_descriptors.extend(
            source.descriptor for source in (*binding.sources, binding.seccomp)
        )
        return binding

    backend._prepare_actor_mount_binding_v1 = record_prepared_binding
    stopped: list[threading.Event] = []

    def starter(
        role_id,
        actor_id,
        container_name,
        command,
        cidfile,
        control_root,
        **_kwargs,
    ):
        assert role_id == 1
        target = (
            backend.paths["python_snapshot"]
            if swap_target == "snapshot"
            else seccomp_path
        )
        backup = target.with_name(target.name + ".swapped")
        parent_mode = stat.S_IMODE(target.parent.lstat().st_mode)
        if swap_target == "seccomp":
            target.parent.chmod(0o755)
        try:
            os.rename(target, backup)
            os.rename(backup, target)
        finally:
            if swap_target == "seccomp":
                target.parent.chmod(parent_mode)
        registry = TOOL.sealed_actor_mount_registry_v1(role_id, command)
        stop = threading.Event()
        sampler = threading.Thread(target=stop.wait, daemon=True)
        sampler.start()
        stopped.append(stop)
        cid = cidfile.parent.lstat()
        return TOOL.HeldActorProcessV1(
            role_id,
            actor_id,
            container_name,
            tuple(command),
            cidfile,
            control_root,
            registry,
            SimpleNamespace(),
            TOOL.BoundedPipeDrainV1(1, bytearray(), 0, False, sha256(), []),
            TOOL.BoundedPipeDrainV1(1, bytearray(), 0, False, sha256(), []),
            threading.Thread(),
            threading.Thread(),
            sampler,
            [],
            [],
            threading.Event(),
            threading.Event(),
            threading.Event(),
            threading.Lock(),
            None,
            (cid.st_dev, cid.st_ino, stat.S_IMODE(cid.st_mode), cid.st_nlink),
            None,
            [],
            TOOL.sealed_policy_file_evidence_v1(
                seccomp_path,
                TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
            ),
        )

    cleaned: list[TOOL.HeldActorProcessV1] = []

    def cleanup(actor, _runner):
        cleaned.append(actor)
        stopped.pop(0).set()
        return ()

    backend.actor_starter = starter
    monkeypatch.setattr(TOOL, "_abort_held_actor_cleanup_v1", cleanup)
    descriptor_count_before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        backend.stage_04_v1(context)
    assert rejected.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    assert backend.admission_consumed is True
    assert 1 in backend.admission_fresh_runtime_checkpoints
    assert backend.completed_stage == 3
    assert len(cleaned) == 1
    assert len(prepared_bindings) == 2
    assert all(binding.closed is True for binding in prepared_bindings)
    assert all(
        source.descriptor == -1
        for binding in prepared_bindings
        for source in (*binding.sources, binding.seccomp)
    )
    for descriptor in prepared_descriptors:
        with pytest.raises(OSError):
            fcntl.fcntl(descriptor, fcntl.F_GETFL)
    # The persistent admission marker chain remains held by design; the
    # launch-scoped mount descriptors above are the resources that must close.
    assert len(os.listdir("/proc/self/fd")) >= descriptor_count_before
    with pytest.raises(TOOL.Q05BDualSupervisorError) as reused:
        backend.stage_04_v1(context)
    assert reused.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    assert "consumed" in reused.value.detail


@pytest.mark.parametrize("forgery", ("command", "mount_registry"))
def test_actor_mount_binding_rejects_forged_starter_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forgery: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    backend = TOOL.ConcreteQ05BActualBackendV1(
        project,
        "ab" * 20,
        project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH,
        tmp_path,
        work,
    )
    backend._create_layout_v1()
    fresh = _materialize_real_mount_admission_fresh_v1(
        backend, tmp_path / f"forged-{forgery}"
    )
    seccomp_path = Path(
        backend.seccomp_evidence["runtime"]["absolute_path"]
    )
    python_command = TOOL.python_endpoint_command_v1(
        backend.paths["python_snapshot"],
        backend.paths["python_output"],
        backend.paths["python_control"],
        seccomp_path,
        docker_slot_row=_docker_slot_row(
            backend.docker_execution_authority,
            "PYTHON_ENDPOINT",
        ),
        cidfile=backend.paths["python_cidfile"],
    )
    rust_command = TOOL.rust_runtime_command_v1(
        backend.paths["binary"],
        backend.paths["rust_output"],
            backend.paths["rust_control"],
            seccomp_path,
            docker_slot_row=_docker_slot_row(
                backend.docker_execution_authority,
                "RUST_ENDPOINT",
            ),
            cidfile=backend.paths["rust_cidfile"],
    )
    backend.cargo_evidence = {"sealed": True}
    backend.planned_commands = {
        "python": python_command,
        "rust": rust_command,
    }
    backend.completed_stage = 3
    context = _arm_mock_concrete_admission(backend, fresh)
    monkeypatch.setattr(
        TOOL,
        "collect_fresh_runtime_evidence_set_v1",
        lambda *_args, **_kwargs: fresh,
    )

    wrong_snapshot = tmp_path / "wrong-snapshot"
    wrong_snapshot.mkdir(mode=0o700)
    marker = wrong_snapshot / "python.marker"
    marker.write_bytes(b"PYTHON_ENDPOINT:python.marker\n")
    marker.chmod(0o444)
    TOOL.seal_directory_tree_read_only_v1(wrong_snapshot)
    wrong_registry = TOOL.sealed_actor_mount_registry_v1(
        1,
        TOOL.python_endpoint_command_v1(
            wrong_snapshot,
            backend.paths["python_output"],
            backend.paths["python_control"],
            seccomp_path,
            docker_slot_row=_docker_slot_row(
                backend.docker_execution_authority,
                "PYTHON_ENDPOINT",
            ),
            cidfile=backend.paths["python_cidfile"],
        ),
    )
    stopped: list[threading.Event] = []

    def starter(
        role_id,
        actor_id,
        container_name,
        command,
        cidfile,
        control_root,
        **_kwargs,
    ):
        assert role_id == 1
        registry = TOOL.sealed_actor_mount_registry_v1(role_id, command)
        stop = threading.Event()
        sampler = threading.Thread(target=stop.wait, daemon=True)
        sampler.start()
        stopped.append(stop)
        cid_parent = cidfile.parent.lstat()
        actor = TOOL.HeldActorProcessV1(
            role_id,
            actor_id,
            container_name,
            tuple(command),
            cidfile,
            control_root,
            registry,
            SimpleNamespace(),
            TOOL.BoundedPipeDrainV1(1, bytearray(), 0, False, sha256(), []),
            TOOL.BoundedPipeDrainV1(1, bytearray(), 0, False, sha256(), []),
            threading.Thread(),
            threading.Thread(),
            sampler,
            [],
            [],
            threading.Event(),
            threading.Event(),
            threading.Event(),
            threading.Lock(),
            None,
            (
                cid_parent.st_dev,
                cid_parent.st_ino,
                stat.S_IMODE(cid_parent.st_mode),
                cid_parent.st_nlink,
            ),
            None,
            [],
            TOOL.sealed_policy_file_evidence_v1(
                seccomp_path,
                TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
            ),
        )
        _bind_synthetic_actor_docker_ownership_v1(
            actor,
            backend.docker_execution_authority,
            "PYTHON_ENDPOINT",
            command,
        )
        _kwargs["ownership_sink"](actor)
        if forgery == "command":
            actor.command = (*actor.command, "--forged-after-start")
        else:
            actor.mount_registry = wrong_registry
        return actor

    cleaned: list[TOOL.HeldActorProcessV1] = []

    def cleanup(actor, _runner):
        cleaned.append(actor)
        stopped.pop(0).set()
        return ()

    backend.actor_starter = starter
    monkeypatch.setattr(TOOL, "_abort_held_actor_cleanup_v1", cleanup)
    prepared_bindings: list[TOOL.HeldActorMountBindingV1] = []
    prepare = backend._prepare_actor_mount_binding_v1

    def record_prepared(*args, **kwargs):
        binding = prepare(*args, **kwargs)
        prepared_bindings.append(binding)
        return binding

    backend._prepare_actor_mount_binding_v1 = record_prepared
    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        backend.stage_04_v1(context)
    assert rejected.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    assert "started actor mount authority differs" in rejected.value.detail
    assert backend.admission_consumed is True
    assert backend.completed_stage == 3
    assert len(cleaned) == 1
    assert len(prepared_bindings) == 2
    assert all(binding.closed for binding in prepared_bindings)
    assert all(
        source.descriptor == -1
        for binding in prepared_bindings
        for source in (*binding.sources, binding.seccomp)
    )


def test_actor_completion_rejects_wrong_live_mount_source(
    tmp_path: Path,
) -> None:
    binding = _synthetic_actor_mount_binding_v1(tmp_path / "binding", 1)
    registry = binding["command_mount_registry"]
    rows = [
        {
            "Destination": destination,
            "Source": source,
            "RW": writable,
            "Type": "bind",
        }
        for destination, source, writable in registry["mount_rows"]
    ]
    rows[0]["Source"] = (tmp_path / "self-chosen-source").as_posix()
    inspect_payload = TOOL._canonical_json_bytes([{"Mounts": rows}])
    completion = {
        "actor_id": binding["actor_id"],
        "command_sha256": registry["command_sha256"],
        "mount_registry_sha256": registry["registry_sha256"],
        "final_resource_transcript": {
            "live_sample_objects": [
                {
                    "mount_registry_sha256": registry["registry_sha256"],
                    "mount_command_sha256": registry["command_sha256"],
                    "inspect_payload_hex": inspect_payload.hex(),
                    "inspect_after_payload_hex": inspect_payload.hex(),
                }
            ],
            "post_exit_inspect_hex": inspect_payload.hex(),
        },
    }
    launch = {
        "role_id": binding["role_id"],
        "actor_id": binding["actor_id"],
        "mount_binding_root": binding["mount_binding_root"],
        "launch_replay_root": "77" * 32,
    }
    with pytest.raises(TOOL.Q05BDualSupervisorError) as rejected:
        TOOL.strict_replay_actor_completion_mount_sources_v1(
            completion, binding, launch
        )
    assert rejected.value.code == TOOL.FAIL_ACTUAL_ADMISSION
    assert "Mount.Source registry differs" in rejected.value.detail


def test_consume_runtime_drift_spends_attempt_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected consume-time replay cannot make its nonce reusable."""

    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    backend = TOOL.ConcreteQ05BActualBackendV1(
        project,
        "ab" * 20,
        project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH,
        tmp_path,
        work,
    )
    backend._create_layout_v1()
    issue_fresh = _materialize_real_mount_admission_fresh_v1(
        backend,
        tmp_path / "drift-real-mount-admission",
    )
    observed_root = tmp_path / "observed-evidence"
    observed_root.mkdir()
    observed_fixture = _synthetic_actual_admission_fixture(
        observed_root, backend.source_commit
    )
    assert TOOL._canonical_json_bytes(
        issue_fresh
    ) != TOOL._canonical_json_bytes(observed_fixture["fresh_runtime"])
    seccomp_path = Path(backend.seccomp_evidence["runtime"]["absolute_path"])
    backend.cargo_evidence = {"sealed": True}
    backend.planned_commands = {
        "python": TOOL.python_endpoint_command_v1(
            backend.paths["python_snapshot"],
            backend.paths["python_output"],
            backend.paths["python_control"],
            seccomp_path,
            docker_slot_row=_docker_slot_row(
                backend.docker_execution_authority,
                "PYTHON_ENDPOINT",
            ),
            cidfile=backend.paths["python_cidfile"],
        ),
        "rust": TOOL.rust_runtime_command_v1(
            backend.paths["binary"],
            backend.paths["rust_output"],
            backend.paths["rust_control"],
            seccomp_path,
            docker_slot_row=_docker_slot_row(
                backend.docker_execution_authority,
                "RUST_ENDPOINT",
            ),
            cidfile=backend.paths["rust_cidfile"],
        ),
    }
    backend.completed_stage = 3
    context = _arm_mock_concrete_admission(
        backend, issue_fresh
    )
    monkeypatch.setattr(
        TOOL,
        "collect_fresh_runtime_evidence_set_v1",
        lambda *_args, **_kwargs: observed_fixture["fresh_runtime"],
    )
    try:
        with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
            backend.stage_04_v1(context)
        assert failure.value.code == TOOL.FAIL_ACTUAL_ADMISSION
        assert "checkpoint set bytes differ" in failure.value.detail
        assert backend.admission_consumed is True
        assert backend.admission_spending_marker_descriptor is not None
        assert backend.admission_consumed_marker_descriptor is not None
        assert backend.admission_consume_artifact_absence == (
            backend.admission_artifact_absence
        )
        assert backend.admission_fresh_runtime_checkpoints == {}
        with pytest.raises(TOOL.Q05BDualSupervisorError) as reused:
            backend.stage_04_v1(context)
        assert reused.value.code == TOOL.FAIL_ACTUAL_ADMISSION
        assert "consumed" in reused.value.detail
    finally:
        for attribute in (
            "admission_consumed_marker_descriptor",
            "admission_spending_marker_descriptor",
            "admission_issued_marker_descriptor",
            "admission_work_root_descriptor",
        ):
            descriptor = getattr(backend, attribute)
            if type(descriptor) is int:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                setattr(backend, attribute, None)


def test_concrete_backend_stage4_through_stage7_mock_causal_wiring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the concrete causal path without Docker or golden generation."""

    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    artifact = project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH

    class CleanableProcess:
        returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    def actor_starter(
        role_id,
        actor_id,
        container_name,
        command,
        cidfile,
        control_root,
        **_kwargs,
    ):
        registry = TOOL.sealed_actor_mount_registry_v1(role_id, command)
        sample_stop = threading.Event()
        sampler = threading.Thread(target=sample_stop.wait, daemon=True)
        sampler.start()
        drain = TOOL.BoundedPipeDrainV1(1024, bytearray(), 0, False, sha256(), [])
        cid_parent = cidfile.parent.lstat()
        seccomp_path = Path(
            registry.security_options[1].removeprefix("seccomp=")
        )
        actor = TOOL.HeldActorProcessV1(
            role_id=role_id,
            actor_id=actor_id,
            container_name=container_name,
            command=tuple(command),
            cidfile=cidfile,
            control_root=control_root,
            mount_registry=registry,
            process=CleanableProcess(),
            stdout_drain=drain,
            stderr_drain=TOOL.BoundedPipeDrainV1(
                1024, bytearray(), 0, False, sha256(), []
            ),
            stdout_thread=threading.Thread(),
            stderr_thread=threading.Thread(),
            sample_thread=sampler,
            sample_rows=[],
            sample_errors=[],
            sample_complete=threading.Event(),
            child_done_observed=threading.Event(),
            sample_stop=sample_stop,
            sample_lock=threading.Lock(),
            container_id=None,
            cid_parent_identity=(
                cid_parent.st_dev,
                cid_parent.st_ino,
                stat.S_IMODE(cid_parent.st_mode),
                cid_parent.st_nlink,
            ),
            cidfile_evidence=None,
            cleanup_errors=[],
            seccomp_evidence=TOOL.sealed_policy_file_evidence_v1(
                seccomp_path,
                TOOL.RUNTIME_SECCOMP_RELATIVE_PATH,
            ),
        )
        _bind_synthetic_actor_docker_ownership_v1(
            actor,
            _kwargs["docker_execution_authority"],
            _kwargs["docker_slot"],
            command,
        )
        _kwargs["ownership_sink"](actor)
        return actor

    python_stdout = _valid_actor_stdout(
        "PYTHON_ENDPOINT",
        "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_PYTHON_V1",
    )
    rust_stdout = _valid_actor_stdout(
        "RUST_ENDPOINT",
        "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_RUST_V1",
    )
    host_stdout = TOOL._canonical_json_bytes(
        {
            "loaded_module_root": "44" * 32,
            "loaded_module_rows": [["hegel_machine", None, None]],
        }
    )
    real_tree_identity = TOOL.sealed_tree_identity_v1

    def seal_test_tree(
        root: Path,
        rows: tuple[tuple[str, bytes], ...],
    ) -> dict[str, object]:
        for relative, payload in rows:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            path.chmod(0o444)
        TOOL.seal_directory_tree_read_only_v1(root)
        return real_tree_identity(root, tuple(sorted(relative for relative, _ in rows)))

    def completion(actor: TOOL.HeldActorProcessV1, stdout: bytes) -> dict[str, object]:
        mount_rows = [
            {
                "Destination": destination,
                "RW": writable,
                "Source": source,
                "Type": "bind",
            }
            for destination, source, writable in actor.mount_registry.mount_rows
        ]
        inspect_payload = TOOL._canonical_json_bytes(
            [
                {
                    "Mounts": mount_rows,
                }
            ]
        )
        docker = _synthetic_docker_completion_ownership(
            backend.docker_execution_authority,
            actor.actor_id,
            list(actor.command),
        )
        post_document = json.loads(bytes.fromhex(docker["post_exit_inspect_hex"]))
        post_document[0]["Mounts"] = mount_rows
        post_payload = json.dumps(
            post_document,
            separators=(",", ":"),
        ).encode("ascii")
        principal = TOOL._docker_execution_principal_v1(
            actor.command,
            backend.docker_execution_authority,
            actor.actor_id,
        )
        docker["post_exit_inspect_hex"] = post_payload.hex()
        docker["post_exit_inspect_sha256"] = sha256(post_payload).hexdigest()
        docker["post_ownership_inspect_evidence"] = (
            TOOL._validate_owned_docker_inspect_payload_v1(
                post_payload,
                principal,
            )
        )
        docker.update({
            "command_sha256": actor.mount_registry.command_sha256,
            "mount_registry_sha256": actor.mount_registry.registry_sha256,
            "seccomp_evidence": actor.seccomp_evidence,
            "stdout_hex": stdout.hex(),
            "final_resource_transcript": {
                "actor_id": actor.actor_id,
                "peak_scope": "CHILD_PLUS_WRAPPER_THROUGH_HELD_FINAL_SAMPLE",
                "live_sample_objects": [
                    {
                        "inspect_payload_hex": inspect_payload.hex(),
                        "inspect_after_payload_hex": inspect_payload.hex(),
                        "mount_registry_sha256": (
                            actor.mount_registry.registry_sha256
                        ),
                        "mount_command_sha256": actor.mount_registry.command_sha256,
                    }
                ],
                "post_exit_inspect_hex": post_payload.hex(),
            },
        })
        return docker

    def actor_group_closer(actors, **_kwargs):
        rows = []
        for actor in actors:
            actor.sample_stop.set()
            if actor.sample_thread is not None:
                actor.sample_thread.join(timeout=1.0)
                assert not actor.sample_thread.is_alive()
            payload = {
                "PYTHON_ENDPOINT": python_stdout,
                "RUST_ENDPOINT": rust_stdout,
                "TRUSTED_HOST_REPLAY": host_stdout,
            }[actor.actor_id]
            if actor.actor_id == "TRUSTED_HOST_REPLAY":
                seal_test_tree(
                    backend.paths["host_staging"],
                    tuple(
                        (f"sidecars/{relative}", sidecar)
                        for relative, sidecar in zip(
                            TOOL.HOST_STAGED_SIDECAR_PATHS,
                            payloads,
                            strict=True,
                        )
                    )
                    + ((TOOL.HOST_SEMANTIC_WITNESS_RELATIVE_PATH, witness),),
                )
            rows.append(completion(actor, payload))
        return tuple(rows)

    backend = TOOL.ConcreteQ05BActualBackendV1(
        project,
        "ab" * 20,
        artifact,
        tmp_path,
        work,
        actor_starter=actor_starter,
        actor_group_closer=actor_group_closer,
    )
    backend._create_layout_v1()
    mount_fresh = _materialize_real_mount_admission_fresh_v1(
        backend,
        tmp_path / "real-mount-admission",
    )
    seccomp_path = Path(backend.seccomp_evidence["runtime"]["absolute_path"])
    backend.source_evidence = {
        actor_id: {
            "source_identity_sha256": ("33" if actor_id == "TRUSTED_HOST_REPLAY" else "11") * 32,
        }
        for actor_id in (
            "PYTHON_ENDPOINT",
            "RUST_ENDPOINT",
            "TRUSTED_HOST_REPLAY",
        )
    }
    backend.snapshot_evidence = {
        actor_id: {"actor_id": actor_id}
        for actor_id in backend.source_evidence
    }
    backend.source_object_closure = {"closure": True}
    backend.cargo_evidence = {
        "file_rows": [["cargo-file", 0o100644]],
        "sealed_snapshot_identity": {"cargo": True},
        "sealed_tree_identity": {"tree": "cargo"},
    }
    backend.image_evidence = {
        "python": {"image": "python"},
        "rust": {"image": "rust"},
    }
    backend.planned_commands = {
        "python": TOOL.python_endpoint_command_v1(
            backend.paths["python_snapshot"],
            backend.paths["python_output"],
            backend.paths["python_control"],
            seccomp_path,
            docker_slot_row=_docker_slot_row(
                backend.docker_execution_authority,
                "PYTHON_ENDPOINT",
            ),
            cidfile=backend.paths["python_cidfile"],
        ),
        "rust": TOOL.rust_runtime_command_v1(
            backend.paths["binary"],
            backend.paths["rust_output"],
            backend.paths["rust_control"],
            seccomp_path,
            docker_slot_row=_docker_slot_row(
                backend.docker_execution_authority,
                "RUST_ENDPOINT",
            ),
            cidfile=backend.paths["rust_cidfile"],
        ),
    }
    backend.completed_stage = 3

    payloads = (b"leaf", b"odd", b"sink", b"sidecar", b"golden")
    output_identities: dict[Path, dict[str, object]] = {}
    replay_partition = SimpleNamespace(scratch_ledger_roots=(b"s" * 32,) * 4)
    def replay_actor(host_replay_root: bytes) -> SimpleNamespace:
        return SimpleNamespace(
            payloads=payloads,
            leaf_manifest=SimpleNamespace(manifest_root=b"a" * 32),
            partitions=(
                SimpleNamespace(evidence_root=b"b" * 32),
                SimpleNamespace(evidence_root=b"c" * 32),
            ),
            sidecar_manifest=SimpleNamespace(manifest_root=b"d" * 32),
            golden_manifest=SimpleNamespace(manifest_root=b"e" * 32),
            shadow_assembler=SimpleNamespace(root=b"f" * 32),
            host_replay_root=host_replay_root,
            partition_replays=(replay_partition, replay_partition),
        )

    python_replay_actor = replay_actor(b"g" * 32)
    rust_replay_actor = replay_actor(b"r" * 32)
    dual = SimpleNamespace(
        python=python_replay_actor,
        rust=rust_replay_actor,
        dual_replay_root=b"h" * 32,
        predicate11_semantic_component_root=b"i" * 32,
        predicate_evidence_rows=((6, b"j" * 32), (7, b"k" * 32)),
        shadow_assembler=SimpleNamespace(root=b"f" * 32),
    )
    witness = b'{"witness":"mock"}\n'

    class FakeHostModule:
        @staticmethod
        def replay_actor_sidecars_v1(actor_id, _stdout, _root):
            return {
                "PYTHON_ENDPOINT": python_replay_actor,
                "RUST_ENDPOINT": rust_replay_actor,
            }[actor_id]

        @staticmethod
        def decode_host_semantic_witness_v1(_payload, *_args):
            return {
                "witness_root": "55" * 32,
                "pending_predicate_ids": [11, 19],
                "host_scratch_evidence_root": "66" * 32,
                "host_scratch_partition_roots": [
                    ["73" * 32] * 4,
                    ["73" * 32] * 4,
                ],
            }

        @staticmethod
        def dual_actor_host_replay_v1(*_args):
            return dual

        @staticmethod
        def read_exact_sidecar_tree_v1(_root):
            return payloads

    fake_host = FakeHostModule()
    monkeypatch.setattr(TOOL, "_load_host_replay_module_v1", lambda _root: fake_host)

    def fake_seal_actor_sidecars(root, _module):
        identity = seal_test_tree(
            root,
            tuple(zip(TOOL.HOST_STAGED_SIDECAR_PATHS, payloads, strict=True)),
        )
        output_identities[root] = identity
        return payloads, identity

    monkeypatch.setattr(
        TOOL,
        "seal_actor_sidecar_tree_v1",
        fake_seal_actor_sidecars,
    )

    stdout_identity: dict[str, object] = {}

    def fake_seal_stdout(root, _py, _rust, _module, **_kwargs):
        manifest = b'{"stdout":"manifest"}\n'
        identity = seal_test_tree(
            root,
            (
                ("manifest.json", manifest),
                ("python.stdout", python_stdout),
                ("rust.stdout", rust_stdout),
            ),
        )
        stdout_identity.clear()
        stdout_identity.update(identity)
        return (
            root / "python.stdout",
            root / "rust.stdout",
            root / "manifest.json",
            manifest,
            identity,
        )

    monkeypatch.setattr(
        TOOL,
        "seal_endpoint_stdout_set_v1",
        fake_seal_stdout,
    )

    def fake_tree_identity(root, *_args, **_kwargs):
        if root == backend.paths["cargo_home"]:
            return backend.cargo_evidence["sealed_tree_identity"]
        return real_tree_identity(root, *_args, **_kwargs)

    monkeypatch.setattr(TOOL, "sealed_tree_identity_v1", fake_tree_identity)
    monkeypatch.setattr(
        TOOL,
        "_read_sealed_regular_file_v1",
        lambda *_args, **_kwargs: witness,
    )
    monkeypatch.setattr(TOOL, "verify_actual_source_commit_v1", lambda *_args: None)
    monkeypatch.setattr(
        TOOL,
        "actor_source_evidence_v1",
        lambda _root, _commit, actor_id: backend.source_evidence[actor_id],
    )
    monkeypatch.setattr(
        TOOL,
        "sealed_snapshot_path_evidence_v1",
        lambda root, _paths: backend.snapshot_evidence[
            {
                backend.paths["python_snapshot"]: "PYTHON_ENDPOINT",
                backend.paths["rust_snapshot"]: "RUST_ENDPOINT",
                backend.paths["host_snapshot"]: "TRUSTED_HOST_REPLAY",
            }[root]
        ],
    )
    monkeypatch.setattr(
        TOOL,
        "git_source_object_closure_evidence_v1",
        lambda *_args: backend.source_object_closure,
    )
    monkeypatch.setattr(
        TOOL,
        "sealed_policy_file_evidence_v1",
        lambda path, relative, **_kwargs: backend.seccomp_evidence[
            "runtime" if relative == TOOL.RUNTIME_SECCOMP_RELATIVE_PATH else "build"
        ],
    )
    monkeypatch.setattr(
        TOOL,
        "replay_sealed_prebuilt_binary_v1",
        lambda *_args: backend.binary_evidence,
    )
    monkeypatch.setattr(
        TOOL,
        "sealed_snapshot_identity_v1",
        lambda *_args: backend.cargo_evidence["sealed_snapshot_identity"],
    )
    monkeypatch.setattr(
        TOOL,
        "local_pinned_image_evidence_v1",
        lambda image, **_kwargs: backend.image_evidence[
            "python" if image == TOOL.PYTHON_IMAGE else "rust"
        ],
    )

    negative = SimpleNamespace(
        canonical_object=lambda: (1, b"negative", (), (), ()),
        corpus_root=b"n" * 32,
        category_roots=((13, b"o" * 32), (18, b"p" * 32)),
    )
    real_import = TOOL.importlib.import_module

    def fake_import(name):
        if name == "hegel_machine.phase3_q05b_negative_vectors_v1":
            return SimpleNamespace(run_q05b_negative_vector_corpus_v1=lambda: negative)
        return real_import(name)

    monkeypatch.setattr(TOOL.importlib, "import_module", fake_import)

    monkeypatch.setattr(
        TOOL,
        "collect_fresh_runtime_evidence_set_v1",
        lambda *_args, **_kwargs: mount_fresh,
    )
    stage4 = backend.stage_04_v1(
        _arm_mock_concrete_admission(backend, mount_fresh)
    )
    stage5 = backend.stage_05_v1({})
    stage6 = backend.stage_06_v1({})
    stage7 = backend.stage_07_v1({})
    assert [stage4["stage_id"], stage5["stage_id"], stage6["stage_id"], stage7["stage_id"]] == [4, 5, 6, 7]
    for expected_checkpoint_count, row in zip(
        (1, 1, 2, 3),
        (stage4, stage5, stage6, stage7),
        strict=True,
    ):
        assert row["qualification_count"] == 0
        assert row["qualification_mask"] == 0
        assert row["candidate_receipt_hex"] is None
        assert row["final_receipt_hex"] is None
        assert row["q1_authority"]["state"] == "NOT_RUN"
        assert row["q1_authority"]["formal_output_roots"] == [None] * 8
        live = row["evidence"]["actual_admission_live_marker_replay"]
        assert live["checkpoint"] == f"STAGE_{row['stage_id']:02d}_BEFORE_EVIDENCE"
        assert live["issued_consumed_same_inode"] is True
        assert live["work_root_path_matches_held_descriptor"] is True
        assert live["issued_path_matches_held_descriptor"] is True
        assert live["spending_path_matches_held_descriptor"] is True
        assert live["consumed_path_matches_held_descriptor"] is True
        checkpoint_rows = row["evidence"][
            "actual_admission_fresh_checkpoint_root_rows"
        ]
        assert len(checkpoint_rows) == expected_checkpoint_count
        assert [checkpoint[0] for checkpoint in checkpoint_rows] == list(
            range(1, expected_checkpoint_count + 1)
        )
        assert checkpoint_rows == [
            [
                checkpoint_id,
                TOOL.ACTUAL_FRESH_RUNTIME_CHECKPOINT_REGISTRY[
                    checkpoint_id - 1
                ][1],
                backend.admission_fresh_runtime_checkpoints[checkpoint_id][
                    "checkpoint_root"
                ],
            ]
            for checkpoint_id in range(1, expected_checkpoint_count + 1)
        ]
        assert row["evidence"][
            "actual_admission_consume_artifact_absence"
        ] == backend.admission_consume_artifact_absence
        expected_mount_roles = [1, 2] if row["stage_id"] <= 5 else [1, 2, 3]
        assert row["evidence"]["actual_actor_mount_binding_root_rows"] == [
            [
                role_id,
                TOOL.ROLE_ROWS[role_id - 1][1],
                backend.actor_mount_bindings[role_id]["mount_binding_root"],
            ]
            for role_id in expected_mount_roles
        ]
        assert row["evidence"]["actual_actor_mount_launch_root_rows"] == [
            [
                role_id,
                TOOL.ROLE_ROWS[role_id - 1][1],
                backend.actor_mount_launch_replays[role_id][
                    "launch_replay_root"
                ],
            ]
            for role_id in expected_mount_roles
        ]
    assert [
        row["role_id"]
        for row in backend.admission_fresh_runtime_checkpoints[1][
            "mount_binding_rows"
        ]
    ] == [1, 2]
    assert [
        row["role_id"]
        for row in backend.admission_fresh_runtime_checkpoints[2][
            "mount_binding_rows"
        ]
    ] == [3]
    assert [
        row["role_id"]
        for row in backend.admission_fresh_runtime_checkpoints[3][
            "mount_binding_rows"
        ]
    ] == [1, 2, 3]
    dynamic_two = backend.admission_fresh_runtime_checkpoints[2][
        "dynamic_authority_set"
    ]
    dynamic_three = backend.admission_fresh_runtime_checkpoints[3][
        "dynamic_authority_set"
    ]
    assert dynamic_two["stage_5_evidence_root"] == stage5["stage_evidence_root"]
    assert TOOL._canonical_json_bytes(dynamic_three) == TOOL._canonical_json_bytes(
        dynamic_two
    )
    assert backend.dynamic_mount_authority_set == dynamic_two
    assert len(stage7["evidence"]["three_actor_live_mount_replays"]) == 3
    assert all(
        replay["all_live_and_post_exit_sources_exact"] is True
        for replay in stage7["evidence"]["three_actor_live_mount_replays"]
    )
    assert stage6["evidence"]["pending_predicate_ids"] == [11, 19]
    assert stage7["evidence"]["qualification_receipt"] is None


@pytest.mark.parametrize(
    "closer_fault",
    ("raise", "malformed", "tampered_flags_raise"),
)
def test_stage6_closer_fault_retains_then_cleans_registered_host_actor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    closer_fault: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    backend = TOOL.ConcreteQ05BActualBackendV1(
        project,
        "ab" * 20,
        project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH,
        tmp_path,
        work,
    )
    backend._create_layout_v1()
    backend.docker_execution_authority = _docker_execution_authority(
        source_commit=backend.source_commit,
        nonce=backend.admission_nonce,
    )
    runtime_seccomp = {
        "absolute_path": (work / "sealed-runtime-seccomp.json").as_posix(),
    }
    backend.seccomp_evidence = {"runtime": runtime_seccomp}
    backend.source_evidence = {
        "TRUSTED_HOST_REPLAY": {
            "source_identity_sha256": "33" * 32,
        }
    }
    backend.endpoint_stdout = (
        _valid_actor_stdout(
            "PYTHON_ENDPOINT",
            "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_PYTHON_V1",
        ),
        _valid_actor_stdout(
            "RUST_ENDPOINT",
            "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_RUST_V1",
        ),
    )
    backend.stdout_paths = (
        backend.paths["stdout_root"] / "python.stdout",
        backend.paths["stdout_root"] / "rust.stdout",
        backend.paths["stdout_root"] / "manifest.json",
    )
    backend.stdout_manifest = b"sealed stdout manifest"
    backend.completed_stage = 5
    binding = SimpleNamespace(binding={"mount_binding_root": "41" * 32})
    monkeypatch.setattr(
        backend,
        "_stage_5_dynamic_mount_authority_set_v1",
        lambda: {"dynamic_authority_root": "42" * 32},
    )
    monkeypatch.setattr(
        backend,
        "_collect_fresh_runtime_checkpoint_v1",
        lambda *_args, **_kwargs: {"checkpoint_root": "43" * 32},
    )
    monkeypatch.setattr(
        backend,
        "_prepare_actor_mount_binding_v1",
        lambda *_args, **_kwargs: binding,
    )
    launched: list[TOOL.HeldActorProcessV1] = []
    container_id = "92" * 32
    docker_calls: list[list[str]] = []
    removed = False

    def docker_runner(command, **_kwargs):
        nonlocal removed
        row = list(command)
        docker_calls.append(row)
        assert row[-1] == container_id
        if row[2] == "inspect":
            if removed:
                return SimpleNamespace(
                    returncode=1,
                    stdout=b"",
                    stderr=f"Error: No such object: {container_id}\n".encode(
                        "ascii"
                    ),
                )
            assert backend.host_command is not None
            return SimpleNamespace(
                returncode=0,
                stdout=_owned_inspect_payload(
                    backend.docker_execution_authority,
                    "TRUSTED_HOST_REPLAY",
                    backend.host_command,
                    container_id,
                ),
                stderr=b"",
            )
        assert row[2:4] == ["rm", "-f"]
        removed = True
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    backend.command_runner = docker_runner

    class FinishedDockerCLI:
        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait(*, timeout: float) -> int:
            assert timeout == 2.0
            return 0

    def launch(_binding, slot, cidfile, control_root):
        assert slot == "TRUSTED_HOST_REPLAY"
        assert backend.host_command is not None
        slot_row = _docker_slot_row(
            backend.docker_execution_authority,
            slot,
        )
        cid_parent = cidfile.parent.lstat()
        actor = TOOL.HeldActorProcessV1(
            role_id=3,
            actor_id=slot,
            container_name=slot_row["container_name"],
            command=tuple(backend.host_command),
            cidfile=cidfile,
            control_root=control_root,
            mount_registry=TOOL.sealed_actor_mount_registry_v1(
                3,
                backend.host_command,
            ),
            process=FinishedDockerCLI(),
            stdout_drain=TOOL.BoundedPipeDrainV1(
                1, bytearray(), 0, False, sha256(), []
            ),
            stderr_drain=TOOL.BoundedPipeDrainV1(
                1, bytearray(), 0, False, sha256(), []
            ),
            stdout_thread=threading.Thread(),
            stderr_thread=threading.Thread(),
            sample_thread=None,
            sample_rows=[],
            sample_errors=[],
            sample_complete=threading.Event(),
            child_done_observed=threading.Event(),
            sample_stop=threading.Event(),
            sample_lock=threading.Lock(),
            container_id=container_id,
            cid_parent_identity=(
                cid_parent.st_dev,
                cid_parent.st_ino,
                stat.S_IMODE(cid_parent.st_mode),
                cid_parent.st_nlink,
            ),
            cidfile_evidence=None,
            cleanup_errors=[],
            seccomp_evidence=runtime_seccomp,
        )
        _bind_synthetic_actor_docker_ownership_v1(
            actor,
            backend.docker_execution_authority,
            slot,
            backend.host_command,
        )
        backend.active_actor_slots[2] = actor
        launched.append(actor)
        return actor

    monkeypatch.setattr(
        backend,
        "_launch_prepared_actor_mount_binding_v1",
        launch,
    )

    def closer(_actors, **_kwargs):
        if closer_fault == "tampered_flags_raise":
            launched[0].failure_cleanup_attempted = True
            launched[0].failure_cleanup_complete = True
            raise RuntimeError("injected closer flag forgery")
        if closer_fault == "raise":
            raise RuntimeError("injected trusted-host closer fault")
        return ({"actor_id": "FOREIGN_ACTOR"},)

    backend.actor_group_closer = closer
    with pytest.raises(BaseException) as rejected:
        backend.stage_06_v1({})
    if closer_fault in {"raise", "tampered_flags_raise"}:
        assert isinstance(rejected.value, RuntimeError)
        expected = (
            "closer flag forgery"
            if closer_fault == "tampered_flags_raise"
            else "trusted-host closer fault"
        )
        assert expected in str(rejected.value)
    else:
        assert isinstance(rejected.value, TOOL.Q05BDualSupervisorError)
        assert "trusted-host completion registry differs" in str(rejected.value)
    assert len(launched) == 1
    assert removed is True
    assert all(
        row[-1] == container_id
        for row in docker_calls
        if row[2] == "rm"
    )
    assert not any(
        row[2] == "rm"
        and row[-1]
        == _docker_slot_row(
            backend.docker_execution_authority,
            "TRUSTED_HOST_REPLAY",
        )["container_name"]
        for row in docker_calls
    )
    assert backend.active_actor_slots[2] is None
    assert backend.host_actor is None
    assert backend.completed_stage == 5


def test_concrete_backend_partial_endpoint_start_cleans_first_actor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    backend = TOOL.ConcreteQ05BActualBackendV1(
        project,
        "ab" * 20,
        project / TOOL.ACTUAL_ARTIFACT_RELATIVE_PATH,
        tmp_path,
        work,
    )
    backend._create_layout_v1()
    mount_fresh = _materialize_real_mount_admission_fresh_v1(
        backend,
        tmp_path / "partial-real-mount-admission",
    )
    seccomp = backend.seccomp_evidence["runtime"]
    seccomp_path = Path(seccomp["absolute_path"])
    backend.cargo_evidence = {"sealed": True}
    backend.planned_commands = {
        "python": TOOL.python_endpoint_command_v1(
            backend.paths["python_snapshot"],
            backend.paths["python_output"],
            backend.paths["python_control"],
            seccomp_path,
            docker_slot_row=_docker_slot_row(
                backend.docker_execution_authority,
                "PYTHON_ENDPOINT",
            ),
            cidfile=backend.paths["python_cidfile"],
        ),
        "rust": TOOL.rust_runtime_command_v1(
            backend.paths["binary"],
            backend.paths["rust_output"],
            backend.paths["rust_control"],
            seccomp_path,
            docker_slot_row=_docker_slot_row(
                backend.docker_execution_authority,
                "RUST_ENDPOINT",
            ),
            cidfile=backend.paths["rust_cidfile"],
        ),
    }
    backend.completed_stage = 3
    stop = threading.Event()
    sampler = threading.Thread(target=stop.wait, daemon=True)
    sampler.start()
    first_actor: TOOL.HeldActorProcessV1 | None = None

    def starter(role_id, actor_id, container_name, command, cidfile, control, **_kwargs):
        nonlocal first_actor
        if role_id == 2:
            raise RuntimeError("second actor start failed")
        registry = TOOL.sealed_actor_mount_registry_v1(role_id, command)
        cid_status = cidfile.parent.lstat()
        first_actor = TOOL.HeldActorProcessV1(
            role_id,
            actor_id,
            container_name,
            tuple(command),
            cidfile,
            control,
            registry,
            SimpleNamespace(),
            TOOL.BoundedPipeDrainV1(1, bytearray(), 0, False, sha256(), []),
            TOOL.BoundedPipeDrainV1(1, bytearray(), 0, False, sha256(), []),
            threading.Thread(),
            threading.Thread(),
            sampler,
            [],
            [],
            threading.Event(),
            threading.Event(),
            threading.Event(),
            threading.Lock(),
            None,
            (
                cid_status.st_dev,
                cid_status.st_ino,
                stat.S_IMODE(cid_status.st_mode),
                cid_status.st_nlink,
            ),
            None,
            [],
            seccomp,
        )
        _bind_synthetic_actor_docker_ownership_v1(
            first_actor,
            backend.docker_execution_authority,
            "PYTHON_ENDPOINT",
            command,
        )
        _kwargs["ownership_sink"](first_actor)
        return first_actor

    cleanup: list[TOOL.HeldActorProcessV1] = []
    backend.actor_starter = starter
    monkeypatch.setattr(TOOL, "verify_actual_source_commit_v1", lambda *_args: None)
    monkeypatch.setattr(
        TOOL,
        "_abort_held_actor_cleanup_v1",
        lambda actor, _runner: (
            cleanup.append(actor),
            ("docker-rm:nonzero:2", "docker-residual:still inspectable"),
        )[1],
    )
    try:
        with pytest.raises(TOOL.Q05BDualSupervisorError) as failure:
            monkeypatch.setattr(
                TOOL,
                "collect_fresh_runtime_evidence_set_v1",
                lambda *_args, **_kwargs: mount_fresh,
            )
            backend.stage_04_v1(
                _arm_mock_concrete_admission(backend, mount_fresh)
        )
        assert failure.value.code == TOOL.FAIL_POLICY
        assert "parallel endpoint startup cleanup failed" in failure.value.detail
        assert "docker-residual" in failure.value.detail
        assert first_actor is not None
        assert cleanup == [first_actor]
        assert backend.active_actor_slots[0] is first_actor
        assert backend.completed_stage == 3
        assert backend.endpoint_actors is None
        with pytest.raises(TOOL.Q05BDualSupervisorError) as reused:
            backend.stage_04_v1(
                {"stage3_to4_admission_issue_record": backend.admission_issue_record}
            )
        assert reused.value.code == TOOL.FAIL_ACTUAL_ADMISSION
        assert "consumed" in reused.value.detail
        assert cleanup == [first_actor]
    finally:
        stop.set()
        sampler.join(timeout=1)


def test_group_close_is_synchronous_and_cleans_only_remaining_actors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actors: list[TOOL.HeldActorProcessV1] = []
    for role_id, actor_id in ((1, "PYTHON_ENDPOINT"), (2, "RUST_ENDPOINT")):
        cid_parent = tmp_path / f"cid-{role_id}"
        control = tmp_path / f"control-{role_id}"
        cid_parent.mkdir(mode=0o700)
        control.mkdir(mode=0o700)
        cid_status = cid_parent.lstat()
        actors.append(
            TOOL.HeldActorProcessV1(
                role_id=role_id,
                actor_id=actor_id,
                container_name=f"hegel-q05b-group-{role_id}",
                command=("unused",),
                cidfile=cid_parent / "actor.cid",
                control_root=control,
                mount_registry=SimpleNamespace(),
                process=SimpleNamespace(),
                stdout_drain=TOOL.BoundedPipeDrainV1(
                    1, bytearray(), 0, False, sha256(), []
                ),
                stderr_drain=TOOL.BoundedPipeDrainV1(
                    1, bytearray(), 0, False, sha256(), []
                ),
                stdout_thread=threading.Thread(),
                stderr_thread=threading.Thread(),
                sample_thread=None,
                sample_rows=[],
                sample_errors=[],
                sample_complete=threading.Event(),
                child_done_observed=threading.Event(),
                sample_stop=threading.Event(),
                sample_lock=threading.Lock(),
                container_id=None,
                cid_parent_identity=(
                    cid_status.st_dev,
                    cid_status.st_ino,
                    stat.S_IMODE(cid_status.st_mode),
                    cid_status.st_nlink,
                ),
                cidfile_evidence=None,
                cleanup_errors=[],
            )
        )
    close_calls: list[str] = []

    def close_actor(actor, **_kwargs):
        close_calls.append(actor.actor_id)
        if actor.actor_id == "PYTHON_ENDPOINT":
            raise RuntimeError("first synchronous close failed")
        return {"actor_id": actor.actor_id}

    monkeypatch.setattr(TOOL, "close_held_actor_process_v1", close_actor)
    cleaned: list[str] = []
    monkeypatch.setattr(
        TOOL,
        "_cleanup_actor_set_v1",
        lambda values, _runner: (
            cleaned.extend(actor.actor_id for actor in values),
            (),
        )[1],
    )
    with pytest.raises(RuntimeError, match="first synchronous close failed"):
        TOOL.close_held_actor_group_v1(
            tuple(actors),
            child_timeout_seconds=1.0,
        )
    assert close_calls == ["PYTHON_ENDPOINT"]
    assert cleaned == ["RUST_ENDPOINT"]
    assert not any(
        thread.name.startswith("q05b-close-")
        for thread in threading.enumerate()
    )
