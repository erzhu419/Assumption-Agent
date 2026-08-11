from __future__ import annotations

from copy import deepcopy
from hashlib import sha1, sha256
import io
import json
import os
from pathlib import Path
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hegel_machine import phase3_q05b_actual_artifact_v1 as A
from hegel_machine import phase3_q05b_actual_admission_v1 as AD
from hegel_machine import phase3_q05b_host_replay_v1 as H
from hegel_machine import phase3_q1_archive_projection_v1 as P
from hegel_machine import phase3_q1_capacity_preflight_v1 as C
from hegel_machine import phase3_q1_partition_snapshot_v1 as S
from hegel_machine import phase3_q1_qualification_wire_v1 as W
from hegel_machine import phase3_q1_semantic_coverage_v1 as V
from hegel_machine.strict_cbor_v1 import canonical_cbor_encode


_PRODUCTION_COMMIT_A_STATIC_POLICY_ROOT = AD.EXPECTED_COMMIT_A_STATIC_POLICY_ROOT
_PRODUCTION_COMMAND_MOUNT_RESOURCE_POLICY_ROOT = (
    AD.EXPECTED_COMMAND_MOUNT_RESOURCE_POLICY_ROOT
)


def _j(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


_ATTEMPT_NONCE = b"A" * 32
_RUNTIME_SECCOMP_PAYLOAD = b'{"synthetic":"runtime-seccomp"}\n'
_BUILD_SECCOMP_PAYLOAD = b'{"synthetic":"build-seccomp"}\n'


def _docker29_security_options(seccomp_payload: bytes) -> list[str]:
    inline = json.dumps(
        json.loads(seccomp_payload),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    return [
        "no-new-privileges",
        "seccomp=" + inline,
    ]


def _docker_slot_map(commit: str) -> dict[str, dict[str, object]]:
    return {
        row["slot"]: row
        for row in AD.docker_execution_slot_rows_v1(commit, _ATTEMPT_NONCE)
    }


def _command_principal(command: list[str]) -> tuple[str, dict[str, str]]:
    name_index = command.index("--name")
    name = command[name_index + 1]
    labels = {}
    for token in command[name_index + 2 : name_index + 5]:
        key, value = token.removeprefix("--label=").split("=", 1)
        labels[key] = value
    return name, labels


def _tree(payloads: dict[str, bytes], root: str, directories: tuple[str, ...], modes: dict[str, int] | None = None) -> dict[str, object]:
    directory_rows = [[path, 1, 100 + i, 2, 1000, 1000, 0o555, 10, 11] for i, path in enumerate(directories)]
    file_rows = [[path, 1, 1000 + i, 1, 1000, 1000, (modes or {}).get(path, 0o444), len(payload), 10, 11, sha256(payload).hexdigest()] for i, (path, payload) in enumerate(payloads.items())]
    body = {"schema_version": "hegel-phase3a-q05b-sealed-tree-identity/1", "root_path": root, "root_device": 1, "root_inode": 2, "root_nlink": 2 + len(directories), "root_mode": 0o555, "directory_rows": directory_rows, "file_rows": file_rows}
    body["manifest_sha256"] = sha256(_j(body)).hexdigest()
    return body


def _production_layout(root: str = "/sealed") -> dict[str, str]:
    def path(relative: str) -> str:
        return root.rstrip("/") + "/" + relative

    return {
        "python_snapshot": path("snapshots/python"),
        "rust_snapshot": path("snapshots/rust"),
        "host_snapshot": path("snapshots/host"),
        "cargo_home": path("cargo-home"),
        "target_output": path("target-output"),
        "cargo_release_binary": path(
            "target-output/release/hegel-q1-archive-projection-oracle"
        ),
        "runtime_binary_parent": path("target-output/runtime-binary"),
        "python_output": path("python-output"),
        "python_control": path("python-control"),
        "python_cid_parent": path("python-cid"),
        "python_cidfile": path("python-cid/python.cid"),
        "rust_output": path("rust-output"),
        "rust_control": path("rust-control"),
        "rust_cid_parent": path("rust-cid"),
        "rust_cidfile": path("rust-cid/rust.cid"),
        "host_output": path("host-output-unused"),
        "host_control": path("host-control"),
        "host_cid_parent": path("host-cid"),
        "host_cidfile": path("host-cid/host.cid"),
        "host_staging": path("host-staging"),
        "build_cid_parent": path("build-cid"),
        "build_test_cidfile": path("build-cid/test.cid"),
        "build_release_cidfile": path("build-cid/release.cid"),
        "stdout_root": path("stdout"),
        "binary": path(
            "target-output/runtime-binary/hegel-q1-archive-projection-oracle"
        ),
    }


def _git_closure(payloads: dict[str, tuple[int, bytes]], prefix: str = "Hegel Machine"):
    root: dict[str, object] = {}
    blob_rows = []
    for path, (mode, payload) in sorted(payloads.items()):
        oid = sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        blob_rows.append([path, mode, oid, payload.hex()])
        node = root
        for part in (prefix + "/" + path).split("/")[:-1]:
            node = node.setdefault(part, {})
        node[(prefix + "/" + path).split("/")[-1]] = (mode, oid)
    trees: dict[str, bytes] = {}

    def emit(node: dict[str, object]) -> str:
        rows = []
        for name, item in sorted(node.items()):
            if type(item) is dict:
                oid = emit(item); mode = 0o40000
            else:
                mode, oid = item
            rows.append(f"{mode:o} {name}".encode() + b"\0" + bytes.fromhex(oid))
        payload = b"".join(rows)
        oid = sha1(b"tree " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        trees[oid] = payload
        return oid

    root_oid = emit(root)
    commit_payload = f"tree {root_oid}\nauthor Q <q@x> 0 +0000\ncommitter Q <q@x> 0 +0000\n\nsynthetic\n".encode()
    commit = sha1(b"commit " + str(len(commit_payload)).encode() + b"\0" + commit_payload).hexdigest()
    return commit, root_oid, blob_rows, [[oid, trees[oid].hex()] for oid in sorted(trees)], commit_payload.hex()


def _source(actor: str, commit: str, paths: list[str], table: dict[str, tuple[int, str, bytes]], root: str) -> dict[str, object]:
    rows = [[path, table[path][0], table[path][1], len(table[path][2]), sha256(table[path][2]).hexdigest()] for path in paths]
    digest = sha256()
    for path in paths:
        raw = path.encode(); payload = table[path][2]
        digest.update(len(raw).to_bytes(4, "big")); digest.update(raw); digest.update(len(payload).to_bytes(8, "big")); digest.update(payload)
    dirs = tuple(sorted({"/".join(path.split("/")[:n]) for path in paths for n in range(1, len(path.split("/"))) }))
    snap_payloads = {path: table[path][2] for path in paths}
    modes = {path: 0o555 if table[path][0] == 0o100755 else 0o444 for path in paths}
    return {"actor_id": actor, "command": [], "control_evidence": {}, "runtime_identity_sha256": sha256((actor + "-runtime").encode()).hexdigest(), "snapshot_identity": _tree(snap_payloads, root, dirs, modes), "source_evidence": {"actor_id": actor, "allowlist_count": len(paths), "blob_rows": rows, "commit": commit, "git_blob_manifest_sha256": sha256(_j(rows)).hexdigest(), "path_registry_sha256": sha256(_j(paths)).hexdigest(), "source_identity_sha256": digest.hexdigest()}}


def _command(config: dict[str, object], role: int, mounts: list[tuple[str, str, bool]], source: str, runtime: str, slot_row: dict[str, object], layout: dict[str, str] | None = None) -> list[str]:
    docker = config["docker"]; image = config["images"][{1: "python_endpoint", 2: "rust_runtime", 3: "trusted_host"}[role]]
    name = slot_row["container_name"]
    cpuset = {1: "0-11", 2: "12-23", 3: "0-11"}[role]
    cidfile = (
        layout[{1: "python_cidfile", 2: "rust_cidfile", 3: "host_cidfile"}[role]]
        if layout is not None
        else f"/sealed/{name}.cid"
    )
    runtime_seccomp = (
        layout["host_snapshot"].rstrip("/")
        + "/"
        + config["seccomp"]["runtime_profile"]
        if layout is not None
        else "/sealed/runtime-seccomp.json"
    )
    label_tokens = [f"--label={key}={value}" for key, value in slot_row["labels"]]
    prefix = [docker["executable"], f"--host={docker['host']}", "run", "--name", name, *label_tokens, f"--cidfile={cidfile}", f"--pull={docker['pull_policy']}", f"--network={docker['network']}", "--read-only", f"--cap-drop={docker['cap_drop']}", "--security-opt=no-new-privileges", f"--security-opt=seccomp={runtime_seccomp}", f"--ipc={docker['ipc']}", "--cgroupns=private", f"--pids-limit={docker['pids_limit']}", f"--ulimit=nofile={docker['nofile_ulimit']}", f"--memory={docker['memory']}", f"--memory-swap={docker['memory_swap']}", f"--cpuset-cpus={cpuset}", f"--tmpfs={docker['runtime_tmpfs']}", "--user=1000:1000", "-e", "HOME=/tmp", "-e", "LANG=C.UTF-8", "-e", "LC_ALL=C.UTF-8", "-e", "TZ=UTC"]
    for src, dst, readonly in mounts:
        prefix += ["--mount", f"type=bind,src={src},dst={dst}" + (",readonly" if readonly else "")]
    if role in (1, 3): prefix += ["-w", "/snapshot"]
    payload = list(config["actor_commands"][{1: "python", 2: "rust", 3: "trusted_host"}[role]])
    if role == 3:
        payload[-3] = source; payload[-1] = runtime
    wrapper = "synthetic-held-wrapper"
    return prefix + [image, "/bin/sh", "-ceu", wrapper, "hegel-q05b-held-actor", *payload]


def _actor_stdout(actor: dict[str, object], implementation: str, payloads, sidecar, golden) -> bytes:
    value = {"action_id": "bounded-node3-golden-v1", "actor_id": actor["actor_id"], "file_count": 5, "implementation_id": implementation, "neutral_manifest_length": len(payloads[4]), "neutral_manifest_raw_sha256": sha256(payloads[4]).hexdigest(), "neutral_manifest_relative_path": W.NODE3_GOLDEN_MANIFEST_RELATIVE_PATH.decode(), "neutral_manifest_root": golden.manifest_root.hex(), "q1_formal_roots": None, "q1_gate_count": 0, "q1_gate_mask": 0, "q1_output_slots": [None] * 8, "q1_state": "NOT_RUN", "runtime_identity_sha256": actor["runtime_identity_sha256"], "schema_version": "hegel-q05b-actor-envelope/1", "sidecar_manifest_length": len(payloads[3]), "sidecar_manifest_raw_sha256": sha256(payloads[3]).hexdigest(), "sidecar_manifest_relative_path": W.SIDECAR_MANIFEST_RELATIVE_PATH.decode(), "sidecar_manifest_root": sidecar.manifest_root.hex(), "source_identity_sha256": actor["source_evidence"]["source_identity_sha256"], "status": "BOUNDED_NODE3_CANDIDATE_EMITTED_NOT_QUALIFIED"}
    return _j(value)


def _mounts(command: list[str]) -> dict[str, tuple[str, bool]]:
    rows = {}
    for index, item in enumerate(command):
        if item == "--mount":
            match = __import__("re").fullmatch(r"type=bind,src=([^,]+),dst=([^,]+)(,readonly)?", command[index + 1])
            rows[match.group(2)] = (match.group(1), match.group(3) is None)
    return rows


def _inspect(command: list[str], role: int, config: dict[str, object], cid: str, running: bool) -> bytes:
    image = config["images"][{1: "python_endpoint", 2: "rust_runtime", 3: "trusted_host"}[role]]
    env = [f"{key}={value}" for key, value in config["runtime_command_inspect_policy"]["environment_rows"][role - 1][2]]
    sources = _mounts(command)
    security = _docker29_security_options(_RUNTIME_SECCOMP_PAYLOAD)
    name, labels = _command_principal(command)
    if role == 2:
        labels.update(dict(AD.DOCKER_RUST_BASE_LABEL_ROWS))
    value = [{"Id": cid, "Image": "sha256:" + sha256(image.encode()).hexdigest(), "Name": "/" + name, "State": {"Running": running, "OOMKilled": False, "Pid": 123 if running else 0, "ExitCode": 0}, "Config": {"Image": image, "User": "1000:1000", "Entrypoint": None, "Cmd": command[command.index(image) + 1:], "WorkingDir": "/snapshot" if role in (1, 3) else "", "Env": env, "Labels": labels}, "HostConfig": {"AutoRemove": False, "NetworkMode": "none", "ReadonlyRootfs": True, "CapDrop": ["ALL"], "SecurityOpt": security, "IpcMode": "none", "PidMode": "", "CgroupnsMode": "private", "UsernsMode": "", "Privileged": False, "Devices": [], "DeviceRequests": None, "CpusetCpus": {1: "0-11", 2: "12-23", 3: "0-11"}[role], "Memory": 14 * 1024**3, "MemorySwap": 14 * 1024**3, "PidsLimit": 128, "Tmpfs": {"/tmp": "rw,noexec,nosuid,nodev,size=2g,mode=1777"}, "Ulimits": [{"Name": "nofile", "Hard": 256, "Soft": 256}]}, "Mounts": [{"Type": "bind", "Destination": dst, "Source": src, "RW": writable} for dst, (src, writable) in sources.items()]}]
    return _j(value)


def _sample(command: list[str], role: int, config: dict[str, object], cid: str, ordinal: int, command_sha: str, mount_sha: str, completion: str | None = None) -> dict[str, object]:
    before = _inspect(command, role, config, cid, True); after = _inspect(command, role, config, cid, True)
    path = f"/docker/{cid}"; proc = f"0::{path}\n".encode(); limits = b"Max open files            256                  256                  files     \n"
    cgroups = {"memory.current": b"100\n", "memory.events": b"low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n", "memory.peak": f"{100 + ordinal}\n".encode(), "pids.current": b"1\n", "pids.peak": b"2\n"}
    digest = sha256(); raw_path = path.encode(); digest.update(len(raw_path).to_bytes(4, "big")); digest.update(raw_path); digest.update((1).to_bytes(8, "big")); digest.update((2).to_bytes(8, "big"))
    for name in sorted(cgroups):
        raw = name.encode(); payload = cgroups[name]; digest.update(len(raw).to_bytes(4, "big")); digest.update(raw); digest.update(len(payload).to_bytes(8, "big")); digest.update(payload)
    value = {"schema_version": "hegel-phase3a-q05b-live-container-resource-transcript/1", "container_id": cid, "role_id": role, "captured_while_running": True, "cpuset_cpus": {1: "0-11", 2: "12-23", 3: "0-11"}[role], "memory_limit_bytes": 14 * 1024**3, "memory_swap_limit_bytes": 14 * 1024**3, "pids_limit": 128, "nofile_soft": 256, "nofile_hard": 256, "oom_killed": False, "memory_current_bytes": 100, "memory_peak_bytes": 100 + ordinal, "pids_current": 1, "pids_peak": 2, "memory_events": [["high", 0], ["low", 0], ["max", 0], ["oom", 0], ["oom_group_kill", 0], ["oom_kill", 0]], "cgroup_path": path, "cgroup_directory_device": 1, "cgroup_directory_inode": 2, "inspect_sha256": sha256(before).hexdigest(), "inspect_payload_hex": before.hex(), "inspect_after_sha256": sha256(after).hexdigest(), "inspect_after_payload_hex": after.hex(), "proc_cgroup_sha256": sha256(proc).hexdigest(), "proc_cgroup_payload_hex": proc.hex(), "cgroup_sha256": digest.hexdigest(), "cgroup_payload_rows": [[name, cgroups[name].hex()] for name in sorted(cgroups)], "proc_limits_sha256": sha256(limits).hexdigest(), "proc_limits_payload_hex": limits.hex(), "mount_registry_sha256": mount_sha, "mount_command_sha256": command_sha, "proc_pid_directory_device": 1, "proc_pid_directory_inode": 2, "anchored_collection": True, "sample_ordinal": ordinal, "sample_monotonic_ns": 1000 + (ordinal - 1) * 100, "sample_duration_ns": 10}
    if completion is not None:
        value.update({"actor_child_complete_held": True, "completion_manifest_sha256": completion, "fresh_after_done_collection": True})
    return value


def _direct_proc_nofile_sample(payload: bytes) -> dict[str, object]:
    cid = "a" * 64
    inspect = _j([{"Id": cid, "State": {"Running": True, "Pid": 123}}])
    cgroup_path = f"/docker/{cid}"
    proc_cgroup = f"0::{cgroup_path}\n".encode()
    cgroups = {
        "memory.current": b"100\n",
        "memory.events": (
            b"low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n"
            b"oom_group_kill 0\n"
        ),
        "memory.peak": b"101\n",
        "pids.current": b"1\n",
        "pids.peak": b"2\n",
    }
    cgroup_digest = sha256()
    path_bytes = cgroup_path.encode()
    cgroup_digest.update(len(path_bytes).to_bytes(4, "big"))
    cgroup_digest.update(path_bytes)
    cgroup_digest.update((1).to_bytes(8, "big"))
    cgroup_digest.update((2).to_bytes(8, "big"))
    for name in sorted(cgroups):
        name_bytes = name.encode()
        cgroup_payload = cgroups[name]
        cgroup_digest.update(len(name_bytes).to_bytes(4, "big"))
        cgroup_digest.update(name_bytes)
        cgroup_digest.update(len(cgroup_payload).to_bytes(8, "big"))
        cgroup_digest.update(cgroup_payload)
    return {
        "schema_version": (
            "hegel-phase3a-q05b-live-container-resource-transcript/1"
        ),
        "container_id": cid,
        "role_id": 1,
        "captured_while_running": True,
        "cpuset_cpus": "0-11",
        "memory_limit_bytes": 14 * 1024**3,
        "memory_swap_limit_bytes": 14 * 1024**3,
        "pids_limit": 128,
        "nofile_soft": 256,
        "nofile_hard": 256,
        "oom_killed": False,
        "memory_current_bytes": 100,
        "memory_peak_bytes": 101,
        "pids_current": 1,
        "pids_peak": 2,
        "memory_events": [
            ["high", 0],
            ["low", 0],
            ["max", 0],
            ["oom", 0],
            ["oom_group_kill", 0],
            ["oom_kill", 0],
        ],
        "cgroup_path": cgroup_path,
        "cgroup_directory_device": 1,
        "cgroup_directory_inode": 2,
        "inspect_sha256": sha256(inspect).hexdigest(),
        "inspect_payload_hex": inspect.hex(),
        "inspect_after_sha256": sha256(inspect).hexdigest(),
        "inspect_after_payload_hex": inspect.hex(),
        "proc_cgroup_sha256": sha256(proc_cgroup).hexdigest(),
        "proc_cgroup_payload_hex": proc_cgroup.hex(),
        "cgroup_sha256": cgroup_digest.hexdigest(),
        "cgroup_payload_rows": [
            [name, cgroups[name].hex()] for name in sorted(cgroups)
        ],
        "proc_limits_sha256": sha256(payload).hexdigest(),
        "proc_limits_payload_hex": payload.hex(),
        "mount_registry_sha256": "0" * 64,
        "mount_command_sha256": "0" * 64,
        "proc_pid_directory_device": 1,
        "proc_pid_directory_inode": 2,
        "anchored_collection": True,
        "sample_ordinal": 1,
        "sample_monotonic_ns": 1000,
        "sample_duration_ns": 10,
    }


def _resource(command: list[str], role: int, config: dict[str, object], cid: str, completion: str, actor: str) -> dict[str, object]:
    command_sha, mount_sha = A._command_mount_registry_v1(command, role, actor, config)
    samples = [_sample(command, role, config, cid, 1, command_sha, mount_sha), _sample(command, role, config, cid, 2, command_sha, mount_sha, completion)]
    rows = [[i, s["inspect_sha256"], s["proc_cgroup_sha256"], s["cgroup_sha256"], s["proc_limits_sha256"], s["memory_peak_bytes"], s["pids_peak"], s["memory_events"], s.get("actor_child_complete_held", False), s.get("completion_manifest_sha256"), s["sample_monotonic_ns"], s["sample_duration_ns"]] for i, s in enumerate(samples, 1)]
    post = _inspect(command, role, config, cid, False)
    body = {"schema_version": "hegel-phase3a-q05b-final-container-resource-transcript/1", "container_id": cid, "role_id": role, "sampling_interval_milliseconds": 250, "continuous_sampling_through_child_completion": True, "fresh_held_final_before_release": True, "post_release_wrapper_only_exits": True, "post_exit_zero_and_no_oom": True, "peak_scope": "CHILD_PLUS_WRAPPER_THROUGH_HELD_FINAL_SAMPLE", "actor_exit_code": 0, "oom_killed": False, "sample_count": 2, "sample_rows": rows, "maximum_inter_sample_gap_ns": 90, "live_sample_objects": samples, "final_memory_peak_bytes": 102, "final_pids_peak": 2, "post_exit_inspect_sha256": sha256(post).hexdigest(), "post_exit_inspect_hex": post.hex(), "explicit_remove_admitted_after_this_transcript": True}
    body["transcript_sha256"] = sha256(_j(body)).hexdigest(); return body


def _control_identity(actor: str, stdout: bytes, released: bool) -> dict[str, object]:
    payloads = {"actor.stdout": stdout, "done": b"ACTOR_COMPLETE_HELD\n", "exit-code": b"0\n", "release": b"HOST_FINAL_SAMPLE_SEALED\n"}
    names = ["actor.stdout", "done", "exit-code"] + (["release"] if released else [])
    rows = [[name, 1, 20 + i, 1, 1000, 1000, 0o444, len(payloads[name]), 1, 2, sha256(payloads[name]).hexdigest()] for i, name in enumerate(names)]
    body = {"schema_version": "hegel-phase3a-q05b-held-control-evidence/1", "actor_id": actor, "root_device": 1, "root_inode": 2, "root_mode": 0o555 if released else 0o700, "file_rows": rows, "actor_stdout_hex": stdout.hex()}; body["manifest_sha256"] = sha256(_j(body)).hexdigest(); return body


def _cid(cid: str, path: str) -> dict[str, object]:
    payload = cid.encode(); body = {"schema_version": "hegel-phase3a-q05b-sealed-docker-cidfile/1", "cidfile_path": path, "relative_name": Path(path).name, "parent_device": 1, "parent_inode": 2, "parent_mode": 0o700, "parent_nlink": 2, "file_device": 1, "file_inode": 3, "file_mode": 0o444, "file_nlink": 1, "file_uid": 1000, "file_gid": 1000, "file_size": len(payload), "payload_hex": payload.hex(), "payload_sha256": sha256(payload).hexdigest(), "container_id": cid}; body["manifest_sha256"] = sha256(_j(body)).hexdigest(); return body


def _seccomp(relative: str, absolute: str, payload: bytes) -> dict[str, object]:
    body = {"schema_version": "hegel-phase3a-q05b-sealed-policy-file/1", "absolute_path": absolute, "snapshot_relative_path": relative, "file_device": 1, "file_inode": 9, "file_nlink": 1, "file_uid": 1000, "file_gid": 1000, "file_mode": 0o444, "file_size": len(payload), "file_mtime_ns": 1, "file_ctime_ns": 2, "payload_sha256": sha256(payload).hexdigest()}; body["manifest_sha256"] = sha256(_j(body)).hexdigest(); return body


def _seccomp_from_snapshot(snapshot: dict[str, object], relative: str, payload: bytes) -> dict[str, object]:
    row = next(row for row in snapshot["file_rows"] if row[0] == relative)
    body = {
        "schema_version": "hegel-phase3a-q05b-sealed-policy-file/1",
        "absolute_path": snapshot["root_path"].rstrip("/") + "/" + relative,
        "snapshot_relative_path": relative,
        "file_device": row[1],
        "file_inode": row[2],
        "file_nlink": row[3],
        "file_uid": row[4],
        "file_gid": row[5],
        "file_mode": row[6],
        "file_size": row[7],
        "file_mtime_ns": row[8],
        "file_ctime_ns": row[9],
        "payload_sha256": sha256(payload).hexdigest(),
    }
    body["manifest_sha256"] = sha256(_j(body)).hexdigest()
    return body


def _absence(cid: str) -> dict[str, object]:
    stdout = b"[]\n"; stderr = f"error: no such object: {cid}\n".encode()
    return {"schema_version": AD.DOCKER_AUTHORITATIVE_ABSENCE_SCHEMA_VERSION, "container_identity": cid, "inspect_exit_code": 1, "inspect_stdout_hex": stdout.hex(), "inspect_stdout_sha256": sha256(stdout).hexdigest(), "inspect_stderr_hex": stderr.hex(), "inspect_stderr_sha256": sha256(stderr).hexdigest()}


def _docker_authority(
    commit: str, nonce: bytes = _ATTEMPT_NONCE
) -> dict[str, object]:
    slots = AD.docker_execution_slot_rows_v1(commit, nonce)
    initial = [
        AD.build_docker_initial_name_absence_row_v1(
            commit,
            nonce,
            row["slot_id"],
            _absence(row["container_name"]),
            _absence(row["container_name"]),
        )
        for row in slots
    ]
    return AD.build_docker_execution_authority_v1(
        commit, nonce, initial
    )


def _owned_inspect(
    payload: bytes,
    command: list[str],
    cid: str,
    authority: dict[str, object],
    slot_row: dict[str, object],
) -> dict[str, object]:
    labels = slot_row["labels"]
    image = next(
        item
        for item in command
        if item in {
            "python@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3",
            "rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89",
        }
    )
    label_root = sha256(
        b"HEGEL/Q05B/DOCKER/OWNERSHIP_LABELS/V1\x00" + _j(labels)
    ).hexdigest()
    body = {
        "schema_version": "hegel-phase3a-q05b-docker-owned-inspect/1",
        "docker_execution_authority_manifest_sha256": authority["manifest_sha256"],
        "slot_id": slot_row["slot_id"],
        "slot": slot_row["slot"],
        "container_id": cid,
        "container_name": slot_row["container_name"],
        "ownership_label_root": label_root,
        "image": image,
        "command_sha256": sha256(_j(command)).hexdigest(),
        "inspect_hex": payload.hex(),
        "inspect_sha256": sha256(payload).hexdigest(),
    }
    return {
        **body,
        "ownership_inspect_root": sha256(
            b"HEGEL/Q05B/DOCKER/OWNED_INSPECT/V1\x00" + _j(body)
        ).hexdigest(),
    }


def _docker_success_ownership(
    *,
    authority: dict[str, object],
    slot_row: dict[str, object],
    command: list[str],
    cid: str,
    live: bytes,
    post: bytes,
) -> dict[str, object]:
    label_root = sha256(
        b"HEGEL/Q05B/DOCKER/OWNERSHIP_LABELS/V1\x00" + _j(slot_row["labels"])
    ).hexdigest()
    precreate = AD.build_docker_precreate_absence_v1(
        authority,
        slot_row["slot_id"],
        _absence(slot_row["container_name"]),
        _absence(slot_row["container_name"]),
    )
    return {
        "docker_execution_authority_manifest_sha256": authority["manifest_sha256"],
        "docker_execution_slot_row": slot_row,
        "ownership_label_root": label_root,
        "precreate_absence_evidence": precreate,
        "live_ownership_inspect_evidence": _owned_inspect(
            live, command, cid, authority, slot_row
        ),
        "post_ownership_inspect_evidence": _owned_inspect(
            post, command, cid, authority, slot_row
        ),
        "explicit_remove_command": [
            "/usr/bin/docker",
            "--host=unix:///var/run/docker.sock",
            "rm",
            cid,
        ],
        "cleanup_target_kind": "OWNERSHIP_VALIDATED_CONTAINER_ID",
        "container_name_was_never_a_destructive_target": True,
    }


def _image_evidence(reference: str, environment: list[list[str]]) -> dict[str, object]:
    image_id = "sha256:" + sha256(reference.encode()).hexdigest()
    labels = (
        dict(AD.DOCKER_RUST_BASE_LABEL_ROWS)
        if reference.startswith("rust@")
        else None
    )
    payload = _j([{"Id": image_id, "RepoDigests": [reference], "Os": "linux", "Architecture": "amd64", "Config": {"Env": [f"{key}={value}" for key, value in environment], "Labels": labels}}])
    body = {"schema_version": "hegel-phase3a-q05b-pinned-local-image-evidence/1", "requested_reference": reference, "image_id": image_id, "repo_digests": [reference], "os": "linux", "architecture": "amd64", "raw_inspect_hex": payload.hex(), "raw_inspect_sha256": sha256(payload).hexdigest()}
    body["evidence_sha256"] = sha256(_j(body)).hexdigest()
    return body


def _negative_object_for_test():
    if os.environ.get("Q05B_FAST_LAYERED_TEST") != "1":
        return A._production_negative_corpus_v1()
    negative = A._negative
    rows = []
    for index in range(10):
        failure = negative.NO_FAILURE if index == 0 else f"FAST_FAIL_{index:02d}".encode()
        rows.append(negative.Q05BNegativeVectorRowV1(f"FAST_{index:02d}".encode(), 13 if index < 5 else 18, failure, failure, bytes([index + 1]) * 32))
    corpus = negative.Q05BNegativeVectorCorpusV1(tuple(rows))
    A._production_negative_corpus_v1 = lambda: corpus
    return corpus


def _control(actor: dict[str, object], stdout: bytes, resource: dict[str, object], config: dict[str, object], seccomp_payload: bytes, authority: dict[str, object], slot_row: dict[str, object], seccomp_evidence: dict[str, object] | None = None) -> dict[str, object]:
    completion = _control_identity(actor["actor_id"], stdout, False); cid = resource["container_id"]; command_sha, mount_sha = A._command_mount_registry_v1(actor["command"], resource["role_id"], actor["actor_id"], config)
    if seccomp_evidence is None:
        seccomp_evidence = _seccomp(config["seccomp"]["runtime_profile"], "/sealed/runtime-seccomp.json", seccomp_payload)
    live = bytes.fromhex(resource["live_sample_objects"][-1]["inspect_payload_hex"])
    post = bytes.fromhex(resource["post_exit_inspect_hex"])
    ownership = _docker_success_ownership(
        authority=authority,
        slot_row=slot_row,
        command=actor["command"],
        cid=cid,
        live=live,
        post=post,
    )
    return {"schema_version": "hegel-phase3a-q05b-held-actor-complete-evidence/1", "actor_id": actor["actor_id"], "container_id": cid, **ownership, "command_sha256": command_sha, "mount_registry_sha256": mount_sha, "cidfile_evidence": _cid(cid, next(item.split("=", 1)[1] for item in actor["command"] if item.startswith("--cidfile="))), "seccomp_evidence": seccomp_evidence, "control_root_path": _mounts(actor["command"])["/control"][0], "control_root_nlink": 2, "completion_evidence": completion, "continuous_sample_count": 1, "held_final_resource": resource["live_sample_objects"][-1], "release_evidence": _control_identity(actor["actor_id"], stdout, True), "post_exit_inspect_hex": resource["post_exit_inspect_hex"], "post_exit_inspect_sha256": resource["post_exit_inspect_sha256"], "final_resource_transcript": resource, "stdout_hex": stdout.hex(), "stdout_sha256": sha256(stdout).hexdigest(), "stdout_length": len(stdout), "stderr_sha256": sha256(b"").hexdigest(), "stderr_length": 0, "explicit_remove_exit_code": 0, "docker_absence_evidence": _absence(cid)}


def _build_command(config: dict[str, object], mounts: list[tuple[str, str, bool]], source: str, cid_path: str, suffix: list[str], slot_row: dict[str, object], seccomp_path: str = "/sealed/build-seccomp.json") -> list[str]:
    docker = config["docker"]
    label_tokens = [f"--label={key}={value}" for key, value in slot_row["labels"]]
    command = [docker["executable"], f"--host={docker['host']}", "run", "--name", slot_row["container_name"], *label_tokens, f"--cidfile={cid_path}", f"--pull={docker['pull_policy']}", f"--network={docker['network']}", "--read-only", f"--cap-drop={docker['cap_drop']}", "--security-opt=no-new-privileges", f"--security-opt=seccomp={seccomp_path}", f"--ipc={docker['ipc']}", "--cgroupns=private", f"--pids-limit={docker['pids_limit']}", f"--ulimit=nofile={docker['nofile_ulimit']}", f"--memory={docker['memory']}", f"--memory-swap={docker['memory_swap']}", "--cpuset-cpus=12-23", f"--tmpfs={docker['build_tmpfs']}", "--user=1000:1000", "-e", "HOME=/tmp", "-e", "LANG=C.UTF-8", "-e", "LC_ALL=C.UTF-8", "-e", "TZ=UTC"]
    for src, dst, readonly in mounts:
        command += ["--mount", f"type=bind,src={src},dst={dst}" + (",readonly" if readonly else "")]
    command += ["-e", "CARGO_HOME=/cargo-home", "-e", "CARGO_NET_OFFLINE=true", "-e", "CARGO_TARGET_DIR=/target-output", "-e", f"HEGEL_Q05B_RUST_SOURCE_IDENTITY_SHA256={source}", "-w", "/snapshot/rust/q1_archive_projection_oracle", config["images"]["rust_build"], *suffix]
    return command


def _build_inspect(command: list[str], config: dict[str, object], cid: str, source: str, running: bool) -> bytes:
    image = config["images"]["rust_build"]
    environment = dict(config["runtime_command_inspect_policy"]["environment_rows"][1][2])
    environment.update({"CARGO_HOME": "/cargo-home", "CARGO_NET_OFFLINE": "true", "CARGO_TARGET_DIR": "/target-output", "HEGEL_Q05B_RUST_SOURCE_IDENTITY_SHA256": source})
    env = [f"{key}={value}" for key, value in environment.items()]
    mounts = _mounts(command)
    security = _docker29_security_options(_BUILD_SECCOMP_PAYLOAD)
    name, labels = _command_principal(command)
    labels.update(dict(AD.DOCKER_RUST_BASE_LABEL_ROWS))
    value = [{"Id": cid, "Image": "sha256:" + sha256(image.encode()).hexdigest(), "Name": "/" + name, "State": {"Running": running, "OOMKilled": False, "Pid": 456 if running else 0, "ExitCode": 0}, "Config": {"Image": image, "User": "1000:1000", "Entrypoint": None, "Cmd": command[command.index(image) + 1:], "WorkingDir": "/snapshot/rust/q1_archive_projection_oracle", "Env": env, "Labels": labels}, "HostConfig": {"AutoRemove": False, "NetworkMode": "none", "ReadonlyRootfs": True, "CapDrop": ["ALL"], "SecurityOpt": security, "IpcMode": "none", "PidMode": "", "CgroupnsMode": "private", "UsernsMode": "", "Privileged": False, "Devices": [], "DeviceRequests": None, "CpusetCpus": "12-23", "Memory": 14 * 1024**3, "MemorySwap": 14 * 1024**3, "PidsLimit": 128, "Tmpfs": {"/tmp": config["docker"]["build_tmpfs"].removeprefix("/tmp:")}, "Ulimits": [{"Name": "nofile", "Hard": 256, "Soft": 256}]}, "Mounts": [{"Type": "bind", "Destination": dst, "Source": src, "RW": writable} for dst, (src, writable) in mounts.items()]}]
    return _j(value)


def _build_run(config: dict[str, object], mounts: list[tuple[str, str, bool]], source: str, suffix: list[str], cid: str, cid_path: str, seccomp_payload: bytes, slot_row: dict[str, object], authority: dict[str, object], seccomp_path: str = "/sealed/build-seccomp.json", seccomp_evidence: dict[str, object] | None = None) -> dict[str, object]:
    command = _build_command(config, mounts, source, cid_path, suffix, slot_row, seccomp_path)
    live = _build_inspect(command, config, cid, source, True)
    post = _build_inspect(command, config, cid, source, False)
    stdout = b"cargo synthetic success\n"
    if seccomp_evidence is None:
        seccomp_evidence = _seccomp(config["seccomp"]["build_profile"], seccomp_path, seccomp_payload)
    ownership = _docker_success_ownership(
        authority=authority,
        slot_row=slot_row,
        command=command,
        cid=cid,
        live=live,
        post=post,
    )
    body = {"schema_version": "hegel-phase3a-q05b-offline-rust-container-run/1", "command": command, "command_sha256": sha256(_j(command)).hexdigest(), **ownership, "cidfile_evidence": _cid(cid, cid_path), "seccomp_evidence": seccomp_evidence, "live_inspect_hex": live.hex(), "live_inspect_sha256": sha256(live).hexdigest(), "post_inspect_hex": post.hex(), "post_inspect_sha256": sha256(post).hexdigest(), "stdout_hex": stdout.hex(), "stdout_sha256": sha256(stdout).hexdigest(), "stdout_length": len(stdout), "stderr_hex": "", "stderr_sha256": sha256(b"").hexdigest(), "stderr_length": 0, "exit_code": 0, "docker_absence_evidence": _absence(cid)}
    body["evidence_sha256"] = sha256(_j(body)).hexdigest()
    return body


def _cargo_material(cargo_root: str = "/sealed/cargo-home") -> dict[str, object]:
    crate_payload = b"pub fn synthetic() -> bool { true }\n"
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("crate-1.0.0/src/lib.rs")
        info.size = len(crate_payload); info.mode = 0o644; info.mtime = 0
        archive.addfile(info, io.BytesIO(crate_payload))
    crate_archive = archive_buffer.getvalue(); checksum = sha256(crate_archive).hexdigest()
    lock = f'version = 3\n\n[[package]]\nname = "crate"\nversion = "1.0.0"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\nchecksum = "{checksum}"\n'.encode()
    index_doc = json.dumps({"name": "crate", "vers": "1.0.0", "cksum": checksum}, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    files = {
        ".package-cache": b"",
        ".package-cache-mutate": b"",
        "registry/cache/test/crate-1.0.0.crate": crate_archive,
        "registry/index/test/.cache/cr/at/crate": b"\x03\x00\x00\x00\x00etag: synthetic\x001.0.0\x00" + index_doc + b"\x00",
        "registry/index/test/config.json": _j({"dl": "https://static.crates.io/crates", "api": "https://crates.io"}),
        "registry/src/test/crate-1.0.0/.cargo-ok": b'{"v":1}',
        "registry/src/test/crate-1.0.0/src/lib.rs": crate_payload,
    }
    rows = [[path, 0o100644, payload.hex()] for path, payload in sorted(files.items())]
    manifest_rows = [[path, mode, len(bytes.fromhex(payload_hex)), sha256(bytes.fromhex(payload_hex)).hexdigest()] for path, mode, payload_hex in rows]
    cargo_tree = _tree(dict(sorted(files.items())), cargo_root, ("registry", "registry/cache", "registry/cache/test", "registry/index", "registry/index/test", "registry/index/test/.cache", "registry/index/test/.cache/cr", "registry/index/test/.cache/cr/at", "registry/src", "registry/src/test", "registry/src/test/crate-1.0.0", "registry/src/test/crate-1.0.0/src"))
    return {
        "lock": lock,
        "checksum": checksum,
        "rows": rows,
        "manifest_rows": manifest_rows,
        "cargo_tree": cargo_tree,
    }


def _cargo(config: dict[str, object], rust_actor: dict[str, object], build_seccomp: bytes, *, source_commit: str = "ab" * 20, authority: dict[str, object] | None = None, material: dict[str, object] | None = None, layout: dict[str, str] | None = None, build_seccomp_evidence: dict[str, object] | None = None) -> dict[str, object]:
    if material is None:
        material = _cargo_material()
    lock = material["lock"]
    checksum = material["checksum"]
    rows = material["rows"]
    manifest_rows = material["manifest_rows"]
    cargo_tree = material["cargo_tree"]
    cargo_root = cargo_tree["root_path"]
    target_output = layout["target_output"] if layout is not None else "/sealed/target-output"
    source_path = layout["cargo_release_binary"] if layout is not None else "/sealed/target-output/release/hegel-q1-archive-projection-oracle"
    binary_path = layout["binary"] if layout is not None else "/sealed/target-output/runtime-binary/hegel-q1-archive-projection-oracle"
    test_cidfile = layout["build_test_cidfile"] if layout is not None else "/sealed/rust-test.cid"
    release_cidfile = layout["build_release_cidfile"] if layout is not None else "/sealed/rust-build.cid"
    build_seccomp_path = (
        layout["host_snapshot"].rstrip("/") + "/" + config["seccomp"]["build_profile"]
        if layout is not None
        else "/sealed/build-seccomp.json"
    )
    if authority is None:
        authority = _docker_authority(source_commit)
    slot_map = {row["slot"]: row for row in authority["ordered_slot_rows"]}
    source = rust_actor["source_evidence"]["source_identity_sha256"]
    mounts = [(rust_actor["snapshot_identity"]["root_path"], "/snapshot", True), (cargo_root, "/cargo-home", True), (target_output, "/target-output", False)]
    test_suffix = list(config["rust_build_policy"]["commands"][0]); build_suffix = list(config["rust_build_policy"]["commands"][1])
    binary = b"synthetic-rust-binary-v1"
    binary_sha = sha256(binary).hexdigest()
    source_parent = {
        "device": 1, "inode": 70, "nlink": 2, "uid": 1000,
        "gid": 1000, "mode": 0o755,
    }
    detached_parent = {
        "device": 1, "inode": 75, "nlink": 2, "uid": 1000,
        "gid": 1000, "mode": 0o700,
    }
    source_identity = {
        "device": 1, "inode": 76, "nlink": 2, "uid": 1000,
        "gid": 1000, "mode": 0o755, "size": len(binary),
        "mtime_ns": 1, "ctime_ns": 1,
    }
    detached_identity = {
        "device": 1, "inode": 77, "nlink": 1, "uid": 1000,
        "gid": 1000, "mode": 0o755, "size": len(binary),
        "mtime_ns": 1, "ctime_ns": 1,
    }
    detach = {
        "schema_version": "hegel-phase3a-q05b-detached-cargo-release-binary/1",
        "source_path": source_path,
        "detached_path": binary_path,
        "source_parent_before": source_parent,
        "source_parent_after": dict(source_parent),
        "source_fd_before": source_identity,
        "source_fd_after": dict(source_identity),
        "source_path_before": dict(source_identity),
        "source_path_after": dict(source_identity),
        "source_sha256_before": binary_sha,
        "source_sha256_after": binary_sha,
        "detached_parent_before": detached_parent,
        "detached_parent_after": dict(detached_parent),
        "detached_fd": detached_identity,
        "detached_path_identity": dict(detached_identity),
        "detached_sha256": binary_sha,
        "source_and_detached_bytes_equal": True,
    }
    detach["manifest_sha256"] = sha256(_j(detach)).hexdigest()
    return {
        "schema_version": "hegel-phase3a-q05b-cargo-build-binary-evidence/1",
        "lock_hex": lock.hex(),
        "locked_packages": [["crate", "1.0.0", checksum]],
        "sealed_cargo_files": rows,
        "sealed_cargo_manifest_sha256": sha256(_j(manifest_rows)).hexdigest(),
        "sealed_cargo_tree": cargo_tree,
        "cargo_snapshot_post_build": cargo_tree,
        "rust_snapshot_post_build": rust_actor["snapshot_identity"],
        "rust_test": _build_run(config, mounts, source, test_suffix, "b" * 64, test_cidfile, build_seccomp, slot_map["RUST_TEST"], authority, build_seccomp_path, build_seccomp_evidence),
        "rust_release_build": _build_run(config, mounts, source, build_suffix, "c" * 64, release_cidfile, build_seccomp, slot_map["RUST_RELEASE"], authority, build_seccomp_path, build_seccomp_evidence),
        "binary_detach_evidence": detach,
        "binary_hex": binary.hex(), "binary_sha256": binary_sha, "binary_path": binary_path,
        "binary_runtime_identity_sha256": sha256(b"HEGEL/Q05B/RUST_RUNTIME_IDENTITY/V1\x00" + len(binary).to_bytes(8, "big") + binary).hexdigest(),
        "binary_file_identity": {"path": binary_path, "device": 1, "inode": 77, "nlink": 1, "uid": 1000, "gid": 1000, "mode": 0o555, "size": len(binary), "mtime_ns": 1, "ctime_ns": 2, "sha256": binary_sha},
        "target_output_root_path": target_output,
        "rust_image_inspect_hex": _j([{"RepoDigests": [config["images"]["rust_build"]]}]).hex(),
        "rust_image_inspect_sha256": sha256(_j([{"RepoDigests": [config["images"]["rust_build"]]}])).hexdigest(),
    }


def _stage_row(stage_id: int, commit: str, evidence: dict[str, object]) -> dict[str, object]:
    body = {
        "candidate_receipt_hex": None,
        "evidence": evidence,
        "final_receipt_hex": None,
        "q1_authority": {
            "certificate_active": False,
            "formal_output_roots": [None] * 8,
            "gate_count": 0,
            "gate_mask": 0,
            "state": "NOT_RUN",
        },
        "qualification_count": 0,
        "qualification_mask": 0,
        "schema_version": AD.ACTUAL_STAGE_SCHEMA_VERSION,
        "source_commit": commit,
        "stage_id": stage_id,
        "stage_name": A._ACTUAL_STAGE_1_TO_3_NAMES[stage_id - 1]
        if stage_id <= 3 else AD.ACTUAL_STAGE_5_NAME,
        "status": "STAGE_COMPLETE_IN_MEMORY_NOT_PUBLISHED",
    }
    return {
        **body,
        "stage_evidence_root": sha256(
            AD.ACTUAL_STAGE_EVIDENCE_ROOT_DOMAIN
            + stage_id.to_bytes(2, "big")
            + _j(body)
        ).hexdigest(),
    }


def _minimal_join_admission_v1(cargo: dict[str, object]) -> dict[str, object]:
    cargo_tree = cargo["sealed_cargo_tree"]
    cargo_snapshot_body = {
        "schema_version": "hegel-phase3a-q05b-sealed-snapshot-identity/1",
        "root_device": cargo_tree["root_device"],
        "root_inode": cargo_tree["root_inode"],
        "root_mode": cargo_tree["root_mode"],
        "file_rows": cargo_tree["file_rows"],
    }
    cargo_snapshot = {
        **cargo_snapshot_body,
        "manifest_sha256": sha256(_j(cargo_snapshot_body)).hexdigest(),
    }
    binary_file = cargo["binary_file_identity"]
    sealed_body = {
        "schema_version": "hegel-phase3a-q05b-sealed-prebuilt-rust-binary/1",
        "binary_path": binary_file["path"],
        "device": binary_file["device"],
        "inode": binary_file["inode"],
        "nlink": binary_file["nlink"],
        "uid": binary_file["uid"],
        "gid": binary_file["gid"],
        "mode": binary_file["mode"],
        "size": binary_file["size"],
        "mtime_ns": binary_file["mtime_ns"],
        "ctime_ns": binary_file["ctime_ns"],
        "sha256": binary_file["sha256"],
        "payload_hex": cargo["binary_hex"],
    }
    sealed = {
        **sealed_body,
        "manifest_sha256": sha256(_j(sealed_body)).hexdigest(),
    }
    stage_3_evidence = {
        "rust_test": cargo["rust_test"],
        "rust_release_build": cargo["rust_release_build"],
        "binary_detach": cargo["binary_detach_evidence"],
        "binary": sealed,
        "rust_snapshot_post_build": cargo["rust_snapshot_post_build"],
        "cargo_snapshot_post_build": cargo_snapshot,
        "cargo_tree_post_build": cargo_tree,
    }
    stage_3 = _stage_row(3, "ab" * 20, stage_3_evidence)
    fresh_binary = {
        "schema_version": "hegel-phase3a-q05b-fresh-prebuilt-rust-binary-identity/1",
        "binary_path": sealed["binary_path"],
        "device": sealed["device"],
        "inode": sealed["inode"],
        "nlink": sealed["nlink"],
        "uid": sealed["uid"],
        "gid": sealed["gid"],
        "mode": sealed["mode"],
        "size": sealed["size"],
        "mtime_ns": sealed["mtime_ns"],
        "ctime_ns": sealed["ctime_ns"],
        "sha256": sealed["sha256"],
        "sealed_binary_manifest_sha256": sealed["manifest_sha256"],
        "stage_3_binary_evidence_sha256": sha256(_j(sealed)).hexdigest(),
    }
    offline = {
        "schema_version": "hegel-phase3a-q05b-fresh-offline-build-identity/1",
        "stage_3_root": stage_3["stage_evidence_root"],
        "rust_test_transcript_sha256": sha256(
            _j(stage_3_evidence["rust_test"])
        ).hexdigest(),
        "rust_release_build_transcript_sha256": sha256(
            _j(stage_3_evidence["rust_release_build"])
        ).hexdigest(),
        "rust_snapshot_manifest_sha256": stage_3_evidence[
            "rust_snapshot_post_build"
        ]["manifest_sha256"],
        "cargo_snapshot_manifest_sha256": cargo_snapshot["manifest_sha256"],
        "cargo_tree_manifest_sha256": cargo_tree["manifest_sha256"],
        "binary_manifest_sha256": sealed["manifest_sha256"],
        "stage_3_evidence_sha256": sha256(_j(stage_3_evidence)).hexdigest(),
    }
    ordered = [{"preimage": {}} for _ in range(12)]
    ordered[6] = {
        "preimage": {
            "cargo_lock_sha256": sha256(
                bytes.fromhex(cargo["lock_hex"])
            ).hexdigest(),
            "cargo_snapshot_evidence": cargo_snapshot,
            "cargo_tree_evidence": cargo_tree,
            "offline_build_identity": offline,
        }
    }
    ordered[7] = {"preimage": {"binary_identity": fresh_binary}}
    bundle = {"ordered_precondition_rows": ordered}
    boundary = {"precondition_bundle_hex": _j(bundle).hex()}
    observed = {"binary": {"identity": fresh_binary}}
    return {
        "prior_stage_evidence_rows": [{}, {}, stage_3],
        "issue_record": {"pure_boundary_hex": _j(boundary).hex()},
        "fresh_runtime_checkpoint_rows": [
            {"observed_fresh_runtime_evidence": deepcopy(observed)}
            for _ in range(3)
        ],
    }


def _direct_cargo_join_inputs_v1():
    config = json.loads(
        (ROOT / "config/phase3_q05b_dual_isolation_v1.json").read_text()
    )
    build_seccomp = _BUILD_SECCOMP_PAYLOAD
    actor = {
        "source_evidence": {"source_identity_sha256": "11" * 32},
        "snapshot_identity": _tree(
            {"rust-source": b"synthetic"},
            "/sealed/rust-snapshot",
            (),
        ),
    }
    pinned = _image_evidence(
        config["images"]["rust_build"],
        config["runtime_command_inspect_policy"]["environment_rows"][1][2],
    )
    cargo = _cargo(config, actor, build_seccomp)
    cargo["rust_image_inspect_hex"] = pinned["raw_inspect_hex"]
    cargo["rust_image_inspect_sha256"] = pinned["raw_inspect_sha256"]
    return config, build_seccomp, pinned, cargo


def _direct_stage12_join_inputs_v1() -> dict[str, object]:
    config = json.loads(
        (ROOT / "config/phase3_q05b_dual_isolation_v1.json").read_text()
    )
    layout = _production_layout()
    runtime_payload = _RUNTIME_SECCOMP_PAYLOAD
    build_payload = _BUILD_SECCOMP_PAYLOAD
    config["seccomp"]["runtime_profile_sha256"] = sha256(
        runtime_payload
    ).hexdigest()
    config["seccomp"]["build_profile_sha256"] = sha256(
        build_payload
    ).hexdigest()
    actor_paths = {
        "PYTHON_ENDPOINT": [
            "config/phase3_q05b_dual_isolation_v1.json",
            "tools/synthetic_python.py",
        ],
        "RUST_ENDPOINT": [
            "rust/q1_archive_projection_oracle/Cargo.lock",
            "rust/q1_archive_projection_oracle/src/main.rs",
        ],
        "TRUSTED_HOST_REPLAY": [
            config["seccomp"]["build_profile"],
            config["seccomp"]["runtime_profile"],
            "src/hegel_machine/phase3_q05b_host_replay_v1.py",
        ],
    }
    for paths in actor_paths.values():
        paths.sort()
    material = _cargo_material(layout["cargo_home"])
    source_payloads = {
        "config/phase3_q05b_dual_isolation_v1.json": (0o100644, _j(config)),
        "tools/synthetic_python.py": (0o100644, b"VALUE = 1\n"),
        "rust/q1_archive_projection_oracle/Cargo.lock": (
            0o100644,
            material["lock"],
        ),
        "rust/q1_archive_projection_oracle/src/main.rs": (
            0o100644,
            b"fn main() {}\n",
        ),
        config["seccomp"]["build_profile"]: (0o100644, build_payload),
        config["seccomp"]["runtime_profile"]: (0o100644, runtime_payload),
        "src/hegel_machine/phase3_q05b_host_replay_v1.py": (
            0o100644,
            b"HOST = True\n",
        ),
    }
    commit, tree_oid, blob_rows, tree_rows, commit_hex = _git_closure(
        source_payloads
    )
    payload_table = {
        path: (mode, oid, bytes.fromhex(payload_hex))
        for path, mode, oid, payload_hex in blob_rows
    }
    snapshot_roots = {
        "PYTHON_ENDPOINT": layout["python_snapshot"],
        "RUST_ENDPOINT": layout["rust_snapshot"],
        "TRUSTED_HOST_REPLAY": layout["host_snapshot"],
    }
    actors = [
        _source(
            actor_id,
            commit,
            actor_paths[actor_id],
            payload_table,
            snapshot_roots[actor_id],
        )
        for actor_id in actor_paths
    ]
    actor_map = {actor["actor_id"]: actor for actor in actors}
    authority = _docker_authority(commit)
    slots = {row["slot"]: row for row in authority["ordered_slot_rows"]}
    runtime_evidence = _seccomp_from_snapshot(
        actor_map["TRUSTED_HOST_REPLAY"]["snapshot_identity"],
        config["seccomp"]["runtime_profile"],
        runtime_payload,
    )
    build_evidence = _seccomp_from_snapshot(
        actor_map["TRUSTED_HOST_REPLAY"]["snapshot_identity"],
        config["seccomp"]["build_profile"],
        build_payload,
    )
    cargo = _cargo(
        config,
        actor_map["RUST_ENDPOINT"],
        build_payload,
        source_commit=commit,
        authority=authority,
        material=material,
        layout=layout,
        build_seccomp_evidence=build_evidence,
    )
    actor_map["PYTHON_ENDPOINT"]["command"] = _command(
        config,
        1,
        [
            (layout["python_snapshot"], "/snapshot", True),
            (layout["python_output"], "/output", False),
            (layout["python_control"], "/control", False),
        ],
        "",
        actor_map["PYTHON_ENDPOINT"]["runtime_identity_sha256"],
        slots["PYTHON_ENDPOINT"],
        layout,
    )
    actor_map["RUST_ENDPOINT"]["command"] = _command(
        config,
        2,
        [
            (cargo["binary_path"], "/runtime/hegel-q1-archive-projection-oracle", True),
            (layout["rust_output"], "/output", False),
            (layout["rust_control"], "/control", False),
        ],
        "",
        cargo["binary_runtime_identity_sha256"],
        slots["RUST_ENDPOINT"],
        layout,
    )
    actor_map["TRUSTED_HOST_REPLAY"]["command"] = _command(
        config,
        3,
        [
            (layout["host_snapshot"], "/snapshot", True),
            (layout["python_output"], "/inputs/python", True),
            (layout["rust_output"], "/inputs/rust", True),
            (layout["stdout_root"] + "/python.stdout", "/inputs/stdout/python.stdout", True),
            (layout["stdout_root"] + "/rust.stdout", "/inputs/stdout/rust.stdout", True),
            (layout["stdout_root"] + "/manifest.json", "/inputs/stdout/manifest.json", True),
            (layout["host_control"], "/control", False),
            (layout["host_staging"], "/staging", False),
        ],
        actor_map["TRUSTED_HOST_REPLAY"]["source_evidence"][
            "source_identity_sha256"
        ],
        actor_map["TRUSTED_HOST_REPLAY"]["runtime_identity_sha256"],
        slots["TRUSTED_HOST_REPLAY"],
        layout,
    )
    for actor in actors:
        actor["control_evidence"] = {
            "seccomp_evidence": runtime_evidence,
        }
    pinned_images = {
        "python": _image_evidence(
            config["images"]["python_endpoint"],
            config["runtime_command_inspect_policy"]["environment_rows"][0][2],
        ),
        "rust": _image_evidence(
            config["images"]["rust_build"],
            config["runtime_command_inspect_policy"]["environment_rows"][1][2],
        ),
    }
    source = {
        "source_commit": commit,
        "project_tree_prefix": "Hegel Machine",
        "git_blob_payload_table": blob_rows,
        "git_commit_object_hex": commit_hex,
        "git_tree_object_rows": tree_rows,
        "external_commit_replay": {"tree_oid": tree_oid},
        "actor_source_path_rows": [
            [actor_id, actor_paths[actor_id]] for actor_id in actor_paths
        ],
    }
    full_source: dict[str, object] = {}
    for actor in actors:
        actor_id = actor["actor_id"]
        paths = actor_paths[actor_id]
        rows = [
            [
                path,
                payload_table[path][0],
                payload_table[path][1],
                len(payload_table[path][2]),
                sha256(payload_table[path][2]).hexdigest(),
            ]
            for path in paths
        ]
        full_source[actor_id] = {
            "schema_version": "hegel-phase3a-q05b-actor-source-evidence/1",
            "actor_id": actor_id,
            "commit": commit,
            "project_git_prefix": "Hegel Machine/",
            "path_registry_sha256": actor["source_evidence"][
                "path_registry_sha256"
            ],
            "source_identity_sha256": actor["source_evidence"][
                "source_identity_sha256"
            ],
            "rows": rows,
            "blob_preimage_rows": [
                [*row, payload_table[row[0]][2].hex()] for row in rows
            ],
        }
    tree_payloads = {row[0]: bytes.fromhex(row[1]) for row in tree_rows}
    project_tree_oid = A._parse_git_tree_v1(tree_payloads[tree_oid])[
        "Hegel Machine"
    ][1]
    commit_payload = bytes.fromhex(commit_hex)
    closure_body = {
        "schema_version": "hegel-phase3a-q05b-git-source-object-closure/1",
        "commit": commit,
        "commit_payload_hex": commit_hex,
        "commit_payload_sha256": sha256(commit_payload).hexdigest(),
        "root_tree_object_id": tree_oid,
        "project_tree_prefix": "Hegel Machine",
        "project_tree_object_id": project_tree_oid,
        "allowlist_union": list(payload_table),
        "tree_object_rows": tree_rows,
    }
    closure = {
        **closure_body,
        "closure_sha256": sha256(_j(closure_body)).hexdigest(),
    }
    host_template = list(actor_map["TRUSTED_HOST_REPLAY"]["command"])
    for flag in (
        "--host-source-identity-root-hex",
        "--host-runtime-identity-root-hex",
    ):
        host_template[host_template.index(flag) + 1] = "0" * 64
    planned = {
        "python": actor_map["PYTHON_ENDPOINT"]["command"],
        "rust": actor_map["RUST_ENDPOINT"]["command"],
        "host_template": host_template,
        "rust_test": cargo["rust_test"]["command"],
        "rust_release": cargo["rust_release_build"]["command"],
    }
    artifact_path = (
        "/synthetic/project/" + config["artifact_layout"]["relative_path"]
    )
    stage_1_evidence = {
        "config_hex": payload_table[
            "config/phase3_q05b_dual_isolation_v1.json"
        ][2].hex(),
        "config_sha256": sha256(
            payload_table["config/phase3_q05b_dual_isolation_v1.json"][2]
        ).hexdigest(),
        "fixed_artifact_path": artifact_path,
        "layout": layout,
        "cargo_cache_source": "/synthetic/external-cargo-cache",
        "cargo_cache_root_identity": [1, 90, 2, 0o700],
        "source_evidence": full_source,
        "source_object_closure": closure,
        "image_evidence": pinned_images,
        "planned_commands": planned,
        "docker_execution_authority": authority,
        "q1_authority": config["dry_run_authority"],
    }
    cargo_tree = cargo["sealed_cargo_tree"]
    cargo_snapshot_body = {
        "schema_version": "hegel-phase3a-q05b-sealed-snapshot-identity/1",
        "root_device": cargo_tree["root_device"],
        "root_inode": cargo_tree["root_inode"],
        "root_mode": cargo_tree["root_mode"],
        "file_rows": cargo_tree["file_rows"],
    }
    cargo_snapshot = {
        **cargo_snapshot_body,
        "manifest_sha256": sha256(_j(cargo_snapshot_body)).hexdigest(),
    }
    cargo_file_rows = [
        [
            path,
            mode,
            len(bytes.fromhex(payload_hex)),
            sha256(bytes.fromhex(payload_hex)).hexdigest(),
        ]
        for path, mode, payload_hex in cargo["sealed_cargo_files"]
    ]
    cargo_evidence = {
        "schema_version": "hegel-phase3a-q05b-sealed-cargo-home/1",
        "locked_registry_package_count": len(cargo["locked_packages"]),
        "locked_packages": cargo["locked_packages"],
        "file_count": len(cargo_file_rows),
        "file_rows": cargo_file_rows,
        "file_preimage_rows": cargo["sealed_cargo_files"],
        "manifest_sha256": cargo["sealed_cargo_manifest_sha256"],
        "sealed_snapshot_identity": cargo_snapshot,
        "root_mode": "0555",
        "file_modes": "0444_OR_0555",
        "cargo_home_mount": "READ_ONLY_PREUNPACKED",
        "root_path": cargo_tree["root_path"],
        "root_nlink": cargo_tree["root_nlink"],
        "sealed_tree_identity": cargo_tree,
    }
    snapshots = {
        actor["actor_id"]: actor["snapshot_identity"] for actor in actors
    }
    stage_2_evidence = {
        "snapshot_evidence": snapshots,
        "cargo_lock_hex": cargo["lock_hex"],
        "cargo_lock_sha256": sha256(bytes.fromhex(cargo["lock_hex"])).hexdigest(),
        "cargo_evidence": cargo_evidence,
        "seccomp_evidence": {
            "runtime": runtime_evidence,
            "build": build_evidence,
        },
    }
    image_rows = [
        {
            "label": label,
            "reference": pinned_images[label]["requested_reference"],
            "evidence": pinned_images[label],
            "evidence_root": AD.fresh_runtime_evidence_object_root_v1(
                "PINNED_IMAGE", label, pinned_images[label]
            ),
        }
        for label in ("python", "rust")
    ]
    fresh_actor_rows = []
    for actor_id in actor_paths:
        source_evidence = full_source[actor_id]
        snapshot = snapshots[actor_id]
        source_identity = {
            "schema_version": "hegel-phase3a-q05b-fresh-actor-source-identity/1",
            "actor_id": actor_id,
            "source_commit": commit,
            "project_git_prefix": "Hegel Machine/",
            "path_registry_sha256": source_evidence["path_registry_sha256"],
            "source_identity_sha256": source_evidence["source_identity_sha256"],
            "blob_count": len(source_evidence["rows"]),
            "snapshot_file_registry_sha256": sha256(
                _j([[row[0], row[6], row[7], row[10]] for row in snapshot["file_rows"]])
            ).hexdigest(),
            "stage_1_source_evidence_sha256": sha256(
                _j(source_evidence)
            ).hexdigest(),
        }
        fresh_actor_rows.append(
            {
                "actor_id": actor_id,
                "source_identity": source_identity,
                "source_identity_root": AD.fresh_runtime_evidence_object_root_v1(
                    "ACTOR_SOURCE", actor_id, source_identity
                ),
                "snapshot_evidence": snapshot,
                "snapshot_evidence_root": AD.fresh_runtime_evidence_object_root_v1(
                    "ACTOR_SNAPSHOT", actor_id, snapshot
                ),
            }
        )
    cargo_material_identity = {
        "schema_version": "hegel-phase3a-q05b-fresh-cargo-material-identity/1",
        "root_path": cargo_evidence["root_path"],
        "root_nlink": cargo_evidence["root_nlink"],
        "file_count": cargo_evidence["file_count"],
        "locked_registry_package_count": cargo_evidence[
            "locked_registry_package_count"
        ],
        "locked_packages_sha256": sha256(
            _j(cargo_evidence["locked_packages"])
        ).hexdigest(),
        "file_registry_sha256": sha256(_j(cargo_evidence["file_rows"])).hexdigest(),
        "material_manifest_sha256": cargo_evidence["manifest_sha256"],
        "sealed_snapshot_manifest_sha256": cargo_snapshot["manifest_sha256"],
        "sealed_tree_manifest_sha256": cargo_tree["manifest_sha256"],
        "stage_2_cargo_evidence_sha256": sha256(_j(cargo_evidence)).hexdigest(),
    }
    seccomp_rows = [
        {
            "label": label,
            "relative_path": relative,
            "evidence": evidence,
            "evidence_root": AD.fresh_runtime_evidence_object_root_v1(
                "SECCOMP_POLICY", label, evidence
            ),
        }
        for label, relative, evidence in (
            ("runtime", config["seccomp"]["runtime_profile"], runtime_evidence),
            ("build", config["seccomp"]["build_profile"], build_evidence),
        )
    ]
    stages = [
        _stage_row(1, commit, stage_1_evidence),
        _stage_row(2, commit, stage_2_evidence),
        {},
    ]
    ordered = [{"preimage": {}} for _ in range(12)]
    ordered[0]["preimage"] = {
        "git_source_transcript": {"project_root": "/synthetic/project"}
    }
    ordered[4]["preimage"] = {"image_rows": image_rows}
    ordered[5]["preimage"] = {"actor_rows": fresh_actor_rows}
    ordered[6]["preimage"] = {
        "cargo_lock_sha256": stage_2_evidence["cargo_lock_sha256"],
        "cargo_material_identity": cargo_material_identity,
        "cargo_snapshot_evidence": cargo_snapshot,
        "cargo_tree_evidence": cargo_tree,
    }
    ordered[7]["preimage"] = {"seccomp_rows": seccomp_rows}
    ordered[8]["preimage"] = {
        "planned_command_registry_sha256": sha256(_j(planned)).hexdigest()
    }
    work_identity = {
        "absolute_path": "/sealed",
        "layout_sha256": sha256(_j(layout)).hexdigest(),
    }
    bundle = {
        "ordered_precondition_rows": ordered,
        "work_root_identity": work_identity,
    }
    boundary = {"precondition_bundle_hex": _j(bundle).hex()}
    admission = {
        "prior_stage_evidence_rows": stages,
        "issue_record": {"pure_boundary_hex": _j(boundary).hex()},
        "artifact_path": artifact_path,
    }
    return {
        "source": source,
        "payload_table": payload_table,
        "config": config,
        "pinned_images": pinned_images,
        "actors": actors,
        "cargo": cargo,
        "admission": admission,
        "bundle": bundle,
        "commit": commit,
    }


def _reencode_direct_stage12_admission_v1(inputs: dict[str, object]) -> None:
    admission = inputs["admission"]
    boundary = {"precondition_bundle_hex": _j(inputs["bundle"]).hex()}
    admission["issue_record"]["pure_boundary_hex"] = _j(boundary).hex()


def _call_direct_stage12_join_v1(inputs: dict[str, object]) -> None:
    A._cross_prior_stage12_top_v1(
        inputs["source"],
        inputs["payload_table"],
        inputs["config"],
        inputs["pinned_images"],
        inputs["actors"],
        inputs["cargo"],
        inputs["admission"],
    )


def _direct_docker_ownership_inputs_v1() -> dict[str, object]:
    config = json.loads(
        (ROOT / "config/phase3_q05b_dual_isolation_v1.json").read_text()
    )
    commit = "ab" * 20
    authority = _docker_authority(commit)
    slots = {row["slot"]: row for row in authority["ordered_slot_rows"]}
    layout = _production_layout()
    actor_rows = []
    actor_specs = (
        (1, "PYTHON_ENDPOINT", "PYTHON_ENDPOINT", "a" * 64),
        (2, "RUST_ENDPOINT", "RUST_ENDPOINT", "d" * 64),
        (3, "TRUSTED_HOST_REPLAY", "TRUSTED_HOST_REPLAY", "e" * 64),
    )
    for role, actor_id, slot_name, cid in actor_specs:
        command = _command(
            config,
            role,
            [],
            "11" * 32,
            "22" * 32,
            slots[slot_name],
            layout,
        )
        live = _inspect(command, role, config, cid, True)
        post = _inspect(command, role, config, cid, False)
        sample = {
            "inspect_payload_hex": live.hex(),
            "inspect_after_payload_hex": live.hex(),
        }
        ownership = _docker_success_ownership(
            authority=authority,
            slot_row=slots[slot_name],
            command=command,
            cid=cid,
            live=live,
            post=post,
        )
        control = {
            **ownership,
            "container_id": cid,
            "cidfile_evidence": {"container_id": cid},
            "held_final_resource": sample,
            "final_resource_transcript": {"live_sample_objects": [sample]},
            "post_exit_inspect_hex": post.hex(),
            "docker_absence_evidence": _absence(cid),
        }
        actor_rows.append(
            {
                "actor_id": actor_id,
                "command": command,
                "control_evidence": control,
            }
        )

    cargo = {}
    for key, slot_name, cid, suffix in (
        ("rust_test", "RUST_TEST", "b" * 64, ["cargo", "test"]),
        ("rust_release_build", "RUST_RELEASE", "c" * 64, ["cargo", "build"]),
    ):
        command = _build_command(
            config,
            [],
            "11" * 32,
            f"/sealed/{key}.cid",
            suffix,
            slots[slot_name],
        )
        live = _build_inspect(command, config, cid, "11" * 32, True)
        post = _build_inspect(command, config, cid, "11" * 32, False)
        cargo[key] = {
            **_docker_success_ownership(
                authority=authority,
                slot_row=slots[slot_name],
                command=command,
                cid=cid,
                live=live,
                post=post,
            ),
            "command": command,
            "cidfile_evidence": {"container_id": cid},
            "live_inspect_hex": live.hex(),
            "post_inspect_hex": post.hex(),
            "docker_absence_evidence": _absence(cid),
        }
    planned = {
        "rust_test": cargo["rust_test"]["command"],
        "rust_release": cargo["rust_release_build"]["command"],
        "python": actor_rows[0]["command"],
        "rust": actor_rows[1]["command"],
        "host_template": actor_rows[2]["command"],
    }
    decision = {
        "schema_version": AD.ACTUAL_ADMISSION_SCHEMA_VERSION,
        "decision": AD.ACTUAL_ADMISSION_DECISION_ID,
        "source_commit": commit,
        "attempt_nonce_hex": _ATTEMPT_NONCE.hex(),
        "attempt_id": "12" * 32,
        "decision_root": "34" * 32,
    }
    pinned = {
        "python": _image_evidence(
            config["images"]["python_endpoint"],
            config["runtime_command_inspect_policy"]["environment_rows"][0][2],
        ),
        "rust": _image_evidence(
            config["images"]["rust_build"],
            config["runtime_command_inspect_policy"]["environment_rows"][1][2],
        ),
    }
    return {
        "config": config,
        "pinned": pinned,
        "actors": actor_rows,
        "cargo": cargo,
        "authority": authority,
        "decision": decision,
        "planned": planned,
        "slots": slots,
    }


def _call_direct_docker_ownership_v1(inputs: dict[str, object]) -> None:
    A._cross_docker_execution_ownership_surfaces_v1(
        inputs["config"],
        inputs["pinned"],
        inputs["actors"],
        inputs["cargo"],
        inputs["authority"],
        inputs["decision"],
        inputs["planned"],
    )


def _validate_direct_cargo_v1(
    cargo: dict[str, object],
    config: dict[str, object],
    build_seccomp: bytes,
    pinned: dict[str, object],
) -> None:
    A._validate_cargo_v1(
        cargo,
        config,
        "11" * 32,
        "/sealed/rust-snapshot",
        build_seccomp,
        pinned["image_id"],
    )


def _rehash_detach_v1(cargo: dict[str, object]) -> None:
    detach = cargo["binary_detach_evidence"]
    body = dict(detach)
    body.pop("manifest_sha256")
    detach["manifest_sha256"] = sha256(_j(body)).hexdigest()


def _git_transcript(commit: str) -> dict[str, object]:
    project = "/synthetic/project"
    rows = []
    for ordinal, purpose, argv, stdout in (
        (1, "VERIFY_HEAD", ["git", "-C", project, "rev-parse", "--verify", "HEAD"], (commit + "\n").encode()),
        (2, "VERIFY_CLEAN_STATUS_Z", ["git", "-C", project, "status", "--porcelain=v1", "--untracked-files=all", "-z"], b""),
    ):
        rows.append({
            "ordinal": ordinal,
            "purpose": purpose,
            "argv": argv,
            "returncode": 0,
            "stdout_hex": stdout.hex(),
            "stderr_hex": "",
            "stdout_sha256": sha256(stdout).hexdigest(),
            "stderr_sha256": sha256(b"").hexdigest(),
        })
    body = {
        "schema_version": AD.ACTUAL_GIT_SOURCE_TRANSCRIPT_SCHEMA_VERSION,
        "project_root": project,
        "requested_source_commit": commit,
        "command_rows": rows,
    }
    return {
        **body,
        "transcript_root": sha256(
            AD.ACTUAL_GIT_SOURCE_TRANSCRIPT_ROOT_DOMAIN + _j(body)
        ).hexdigest(),
    }


def _mount_registry(
    command: list[str], role_id: int, config: dict[str, object]
) -> dict[str, object]:
    mounts = _mounts(command)
    image = AD.EXPECTED_PYTHON_IMAGE_REFERENCE if role_id in (1, 3) else AD.EXPECTED_RUST_IMAGE_REFERENCE
    image_index = command.index(image)
    environment_rows = config["runtime_command_inspect_policy"][
        "environment_rows"
    ][role_id - 1][2]
    body = {
        "schema_version": "hegel-phase3a-q05b-sealed-command-mount-registry/1",
        "role_id": role_id,
        "command_sha256": sha256(_j(command)).hexdigest(),
        "mount_rows": [[destination, mounts[destination][0], mounts[destination][1]] for destination in sorted(mounts)],
        "container_argv": command[image_index + 1 :],
        "security_options": [item.removeprefix("--security-opt=") for item in command if item.startswith("--security-opt=")],
        "environment_rows": environment_rows,
        "working_directory": config["runtime_command_inspect_policy"][
            "working_directory_rows"
        ][role_id - 1][2],
    }
    return {**body, "registry_sha256": sha256(_j(body)).hexdigest()}


def _mount_binding(
    role_id: int,
    command: list[str],
    fresh: dict[str, object],
    five_sidecars: dict[str, object],
    endpoint: dict[str, object],
    config: dict[str, object],
) -> dict[str, object]:
    registry = _mount_registry(command, role_id, config)
    mounts = _mounts(command)
    fresh_actors = {row["actor_id"]: row for row in fresh["actor_rows"]}
    runtime_seccomp = fresh["seccomp_rows"][0]["evidence"]
    actor_id = ("PYTHON_ENDPOINT", "RUST_ENDPOINT", "TRUSTED_HOST_REPLAY")[role_id - 1]
    source_rows = []
    for destination, writable, source_type, source_mode in AD.ACTUAL_ACTOR_MOUNT_ROLE_REGISTRY[role_id - 1][2]:
        source = mounts[destination][0]
        kind, label = next(
            (kind, label)
            for row_role, row_destination, kind, label in AD.ACTUAL_ACTOR_MOUNT_AUTHORITY_REGISTRY
            if row_role == role_id and row_destination == destination
        )
        if kind == "PRELAUNCH_WRITABLE_DIRECTORY":
            identity = (1, 5000 + len(source_rows), 2, 1000, 1000, 0o700, None)
            authority = AD.build_prelaunch_writable_directory_evidence_v1(
                role_id, destination, source, *identity[:6]
            )
        elif kind == "FRESH_ACTOR_SNAPSHOT":
            authority = fresh_actors[actor_id]["snapshot_evidence"]
            identity = (authority["root_device"], authority["root_inode"], authority["root_nlink"], 1000, 1000, authority["root_mode"], None)
        elif kind == "FRESH_PREBUILT_RUST_BINARY":
            authority = fresh["binary"]["identity"]
            identity = (authority["device"], authority["inode"], authority["nlink"], authority["uid"], authority["gid"], authority["mode"], authority["size"])
        elif kind == "SEALED_ENDPOINT_TREE":
            authority = five_sidecars["python_output_tree" if destination == "/inputs/python" else "rust_output_tree"]
            identity = (authority["root_device"], authority["root_inode"], authority["root_nlink"], 1000, 1000, authority["root_mode"], None)
        elif kind == "SEALED_STDOUT_FILE":
            relative = destination.rsplit("/", 1)[-1]
            stdout_tree = endpoint["sealed_stdout_tree"]
            file_row = next(row for row in stdout_tree["file_rows"] if row[0] == relative)
            authority = {
                "schema_version": "hegel-phase3a-q05b-sealed-stdout-mount-file/1",
                "tree_manifest_sha256": stdout_tree["manifest_sha256"],
                "relative_path": relative,
                "file_row": file_row,
            }
            identity = tuple(file_row[1:8])
        else:
            raise AssertionError(kind)
        source_rows.append(
            AD.build_actor_mount_source_row_v1(
                role_id, destination, source, writable, source_type,
                identity[0], identity[1], identity[2], identity[3], identity[4],
                identity[5], identity[6], kind, label, authority,
            )
        )
    seccomp_kind, seccomp_label = next(
        (kind, label)
        for row_role, destination, kind, label in AD.ACTUAL_ACTOR_MOUNT_AUTHORITY_REGISTRY
        if row_role == role_id and destination == "@seccomp"
    )
    seccomp_row = AD.build_actor_mount_source_row_v1(
        role_id, "@seccomp", runtime_seccomp["absolute_path"], False,
        "REGULAR_FILE", runtime_seccomp["file_device"],
        runtime_seccomp["file_inode"], runtime_seccomp["file_nlink"],
        runtime_seccomp["file_uid"], runtime_seccomp["file_gid"],
        runtime_seccomp["file_mode"], runtime_seccomp["file_size"],
        seccomp_kind, seccomp_label, runtime_seccomp,
    )
    return AD.build_actor_mount_binding_v1(command, registry, source_rows, seccomp_row)


def _mount_launch_replay(binding: dict[str, object]) -> dict[str, object]:
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

    return AD.build_actor_mount_launch_replay_v1(
        binding,
        [replay(row) for row in binding["source_rows"]],
        replay(binding["seccomp_row"]),
    )


def _actual_admission_section(
    *,
    config: dict[str, object],
    config_oid: str,
    commit: str,
    source_wire_profile: dict[str, object],
    layout: dict[str, str],
    actors: list[dict[str, object]],
    pinned_images: list[list[object]],
    cargo: dict[str, object],
    runtime_seccomp: bytes,
    build_seccomp: bytes,
    five_sidecars: dict[str, object],
    endpoint: dict[str, object],
    strict_endpoint_replay_roots: list[str],
) -> dict[str, object]:
    config_bytes = _j(config)
    project_root = "/synthetic/project"
    artifact_path = (
        project_root.rstrip("/")
        + "/"
        + config["artifact_layout"]["relative_path"]
    )
    payload_table = {
        row[0]: (row[1], row[2], bytes.fromhex(row[3]))
        for row in source_wire_profile["git_blob_payload_table"]
    }
    path_rows = dict(source_wire_profile["actor_source_path_rows"])
    full_source_evidence: dict[str, object] = {}
    for actor in actors:
        actor_id = actor["actor_id"]
        paths = path_rows[actor_id]
        rows = [
            [
                path,
                payload_table[path][0],
                payload_table[path][1],
                len(payload_table[path][2]),
                sha256(payload_table[path][2]).hexdigest(),
            ]
            for path in paths
        ]
        condensed = actor["source_evidence"]
        full_source_evidence[actor_id] = {
            "schema_version": "hegel-phase3a-q05b-actor-source-evidence/1",
            "actor_id": actor_id,
            "commit": commit,
            "project_git_prefix": "Hegel Machine/",
            "path_registry_sha256": condensed["path_registry_sha256"],
            "source_identity_sha256": condensed["source_identity_sha256"],
            "rows": rows,
            "blob_preimage_rows": [
                [*row, payload_table[row[0]][2].hex()] for row in rows
            ],
        }
    commit_payload = bytes.fromhex(source_wire_profile["git_commit_object_hex"])
    tree_payloads = {
        row[0]: bytes.fromhex(row[1])
        for row in source_wire_profile["git_tree_object_rows"]
    }
    project_tree_object_id = source_wire_profile["external_commit_replay"][
        "tree_oid"
    ]
    for component in source_wire_profile["project_tree_prefix"].split("/"):
        mode, project_tree_object_id = A._parse_git_tree_v1(
            tree_payloads[project_tree_object_id]
        )[component]
        assert mode == 0o40000
    closure_body = {
        "schema_version": "hegel-phase3a-q05b-git-source-object-closure/1",
        "commit": commit,
        "commit_payload_hex": commit_payload.hex(),
        "commit_payload_sha256": sha256(commit_payload).hexdigest(),
        "root_tree_object_id": source_wire_profile["external_commit_replay"][
            "tree_oid"
        ],
        "project_tree_prefix": source_wire_profile["project_tree_prefix"],
        "project_tree_object_id": project_tree_object_id,
        "allowlist_union": list(payload_table),
        "tree_object_rows": source_wire_profile["git_tree_object_rows"],
    }
    source_closure = {
        **closure_body,
        "closure_sha256": sha256(_j(closure_body)).hexdigest(),
    }
    host_template = list(actors[2]["command"])
    for flag in (
        "--host-source-identity-root-hex",
        "--host-runtime-identity-root-hex",
    ):
        host_template[host_template.index(flag) + 1] = "0" * 64
    planned_commands = {
        "python": actors[0]["command"],
        "rust": actors[1]["command"],
        "host_template": host_template,
        "rust_test": cargo["rust_test"]["command"],
        "rust_release": cargo["rust_release_build"]["command"],
    }
    docker_authority = _docker_authority(commit)
    work = {
        "schema_version": "hegel-phase3a-q05b-admission-work-root-identity/1",
        "absolute_path": "/sealed",
        "device": 1,
        "inode": 2,
        "nlink": 2,
        "mode": 0o700,
        "layout_sha256": sha256(_j(layout)).hexdigest(),
    }
    absence = {
        "schema_version": "hegel-phase3a-q05b-admission-artifact-absence/1",
        "artifact_path": artifact_path,
        "parent_path": project_root + "/artifacts",
        "parent_device": 1,
        "parent_inode": 3,
        "parent_nlink": 2,
        "parent_mode": 0o700,
        "target_absent": True,
        "nofollow_dirfd_checked": True,
    }
    image_rows = []
    for label, evidence in pinned_images:
        image_rows.append({
            "label": label,
            "reference": evidence["requested_reference"],
            "evidence": evidence,
            "evidence_root": AD.fresh_runtime_evidence_object_root_v1("PINNED_IMAGE", label, evidence),
        })
    actor_rows = []
    for actor in actors:
        actor_id = actor["actor_id"]
        snapshot = actor["snapshot_identity"]
        snapshot_file_registry_sha256 = sha256(
            _j(
                [
                    [row[0], row[6], row[7], row[10]]
                    for row in snapshot["file_rows"]
                ]
            )
        ).hexdigest()
        fresh_source = {
            "schema_version": "hegel-phase3a-q05b-fresh-actor-source-identity/1",
            "actor_id": actor_id,
            "source_commit": commit,
            "project_git_prefix": "Hegel Machine/",
            "path_registry_sha256": actor["source_evidence"]["path_registry_sha256"],
            "source_identity_sha256": actor["source_evidence"]["source_identity_sha256"],
            "blob_count": actor["source_evidence"]["allowlist_count"],
            "snapshot_file_registry_sha256": snapshot_file_registry_sha256,
            "stage_1_source_evidence_sha256": sha256(
                _j(full_source_evidence[actor_id])
            ).hexdigest(),
        }
        actor_rows.append({
            "actor_id": actor_id,
            "source_identity": fresh_source,
            "source_identity_root": AD.fresh_runtime_evidence_object_root_v1("ACTOR_SOURCE", actor_id, fresh_source),
            "snapshot_evidence": snapshot,
            "snapshot_evidence_root": AD.fresh_runtime_evidence_object_root_v1("ACTOR_SNAPSHOT", actor_id, snapshot),
        })
    cargo_tree = cargo["sealed_cargo_tree"]
    cargo_snapshot_body = {
        "schema_version": "hegel-phase3a-q05b-sealed-snapshot-identity/1",
        "root_device": cargo_tree["root_device"],
        "root_inode": cargo_tree["root_inode"],
        "root_mode": cargo_tree["root_mode"],
        "file_rows": cargo_tree["file_rows"],
    }
    cargo_snapshot = {**cargo_snapshot_body, "manifest_sha256": sha256(_j(cargo_snapshot_body)).hexdigest()}
    cargo_file_rows = [
        [
            path,
            mode,
            len(bytes.fromhex(payload_hex)),
            sha256(bytes.fromhex(payload_hex)).hexdigest(),
        ]
        for path, mode, payload_hex in cargo["sealed_cargo_files"]
    ]
    cargo_evidence = {
        "schema_version": "hegel-phase3a-q05b-sealed-cargo-home/1",
        "locked_registry_package_count": len(cargo["locked_packages"]),
        "locked_packages": cargo["locked_packages"],
        "file_count": len(cargo_file_rows),
        "file_rows": cargo_file_rows,
        "file_preimage_rows": cargo["sealed_cargo_files"],
        "manifest_sha256": cargo["sealed_cargo_manifest_sha256"],
        "sealed_snapshot_identity": cargo_snapshot,
        "root_mode": "0555",
        "file_modes": "0444_OR_0555",
        "cargo_home_mount": "READ_ONLY_PREUNPACKED",
        "root_path": cargo_tree["root_path"],
        "root_nlink": cargo_tree["root_nlink"],
        "sealed_tree_identity": cargo_tree,
    }
    cargo_material = {
        "schema_version": "hegel-phase3a-q05b-fresh-cargo-material-identity/1",
        "root_path": cargo_tree["root_path"],
        "root_nlink": cargo_tree["root_nlink"],
        "file_count": len(cargo_tree["file_rows"]),
        "locked_registry_package_count": len(cargo["locked_packages"]),
        "locked_packages_sha256": sha256(_j(cargo["locked_packages"])).hexdigest(),
        "file_registry_sha256": sha256(_j(cargo_file_rows)).hexdigest(),
        "material_manifest_sha256": cargo_evidence["manifest_sha256"],
        "sealed_snapshot_manifest_sha256": cargo_snapshot["manifest_sha256"],
        "sealed_tree_manifest_sha256": cargo_tree["manifest_sha256"],
        "stage_2_cargo_evidence_sha256": sha256(_j(cargo_evidence)).hexdigest(),
    }
    host_snapshot = actors[2]["snapshot_identity"]
    runtime_evidence = _seccomp_from_snapshot(
        host_snapshot, config["seccomp"]["runtime_profile"], runtime_seccomp
    )
    build_evidence = _seccomp_from_snapshot(
        host_snapshot, config["seccomp"]["build_profile"], build_seccomp
    )
    seccomp_rows = []
    for label, relative, evidence in (
        ("runtime", config["seccomp"]["runtime_profile"], runtime_evidence),
        ("build", config["seccomp"]["build_profile"], build_evidence),
    ):
        seccomp_rows.append({
            "label": label,
            "relative_path": relative,
            "evidence": evidence,
            "evidence_root": AD.fresh_runtime_evidence_object_root_v1("SECCOMP_POLICY", label, evidence),
        })
    stage_1_evidence = {
        "config_hex": config_bytes.hex(),
        "config_sha256": sha256(config_bytes).hexdigest(),
        "fixed_artifact_path": artifact_path,
        "layout": layout,
        "cargo_cache_source": "/synthetic/external-cargo-cache",
        "cargo_cache_root_identity": [1, 90, 2, 0o700],
        "source_evidence": full_source_evidence,
        "source_object_closure": source_closure,
        "image_evidence": dict(pinned_images),
        "planned_commands": planned_commands,
        "docker_execution_authority": docker_authority,
        "q1_authority": deepcopy(config["dry_run_authority"]),
    }
    stage_2_evidence = {
        "snapshot_evidence": {
            actor["actor_id"]: actor["snapshot_identity"] for actor in actors
        },
        "cargo_lock_hex": cargo["lock_hex"],
        "cargo_lock_sha256": sha256(bytes.fromhex(cargo["lock_hex"])).hexdigest(),
        "cargo_evidence": cargo_evidence,
        "seccomp_evidence": {
            "runtime": runtime_evidence,
            "build": build_evidence,
        },
    }
    stage_1 = _stage_row(1, commit, stage_1_evidence)
    stage_2 = _stage_row(2, commit, stage_2_evidence)
    binary_file = cargo["binary_file_identity"]
    sealed_binary_body = {
        "schema_version": "hegel-phase3a-q05b-sealed-prebuilt-rust-binary/1",
        "binary_path": binary_file["path"],
        "device": binary_file["device"],
        "inode": binary_file["inode"],
        "nlink": binary_file["nlink"],
        "uid": binary_file["uid"],
        "gid": binary_file["gid"],
        "mode": binary_file["mode"],
        "size": binary_file["size"],
        "mtime_ns": binary_file["mtime_ns"],
        "ctime_ns": binary_file["ctime_ns"],
        "sha256": binary_file["sha256"],
        "payload_hex": cargo["binary_hex"],
    }
    sealed_binary = {
        **sealed_binary_body,
        "manifest_sha256": sha256(_j(sealed_binary_body)).hexdigest(),
    }
    stage_3_evidence = {
        "rust_test": cargo["rust_test"],
        "rust_release_build": cargo["rust_release_build"],
        "binary_detach": cargo["binary_detach_evidence"],
        "binary": sealed_binary,
        "rust_snapshot_post_build": cargo["rust_snapshot_post_build"],
        "cargo_snapshot_post_build": cargo_snapshot,
        "cargo_tree_post_build": cargo_tree,
    }
    stage_3 = _stage_row(3, commit, stage_3_evidence)
    stages = [stage_1, stage_2, stage_3]
    stage_roots = [
        [row["stage_id"], row["stage_evidence_root"]] for row in stages
    ]
    binary = {
        "schema_version": "hegel-phase3a-q05b-fresh-prebuilt-rust-binary-identity/1",
        "binary_path": binary_file["path"],
        "device": binary_file["device"],
        "inode": binary_file["inode"],
        "nlink": binary_file["nlink"],
        "uid": binary_file["uid"],
        "gid": binary_file["gid"],
        "mode": binary_file["mode"],
        "size": binary_file["size"],
        "mtime_ns": binary_file["mtime_ns"],
        "ctime_ns": binary_file["ctime_ns"],
        "sha256": binary_file["sha256"],
        "sealed_binary_manifest_sha256": sealed_binary["manifest_sha256"],
        "stage_3_binary_evidence_sha256": sha256(
            _j(sealed_binary)
        ).hexdigest(),
    }
    fresh = AD.build_fresh_runtime_evidence_set_v1(
        commit, image_rows, actor_rows, cargo_material, cargo_snapshot,
        cargo_tree, seccomp_rows, binary,
    )
    offline = {
        "schema_version": "hegel-phase3a-q05b-fresh-offline-build-identity/1",
        "stage_3_root": stages[2]["stage_evidence_root"],
        "rust_test_transcript_sha256": sha256(_j(cargo["rust_test"])).hexdigest(),
        "rust_release_build_transcript_sha256": sha256(_j(cargo["rust_release_build"])).hexdigest(),
        "rust_snapshot_manifest_sha256": actor_rows[1]["snapshot_evidence"]["manifest_sha256"],
        "cargo_snapshot_manifest_sha256": cargo_snapshot["manifest_sha256"],
        "cargo_tree_manifest_sha256": cargo_tree["manifest_sha256"],
        "binary_manifest_sha256": binary["sealed_binary_manifest_sha256"],
        "stage_3_evidence_sha256": sha256(_j(stage_3_evidence)).hexdigest(),
    }
    git = _git_transcript(commit)
    preimages = [
        {"stage_1_root": stage_roots[0][1], "requested_source_commit": commit, "fresh_head_commit": commit, "clean": True, "porcelain_line_count": 0, "git_source_transcript": git},
        {"stage_1_root": stage_roots[0][1], "config_relative_path": "config/phase3_q05b_dual_isolation_v1.json", "commit_a_config_hex": config_bytes.hex(), "runtime_loaded_config_hex": config_bytes.hex(), "config_length": len(config_bytes), "config_sha256": sha256(config_bytes).hexdigest()},
        {"stage_1_root": stage_roots[0][1], "engineering_status": AD.COMMIT_A_ACTUAL_ENGINEERING_STATUS, "actual_preconditions": deepcopy(AD.COMMIT_A_ACTUAL_PRECONDITIONS_V1), "entrypoint": "run_actual_v1", "entrypoint_implemented": True, "conditional_single_attempt_policy": "CONDITIONAL_SINGLE_ATTEMPT_RUNTIME_ADMISSION"},
        {"stage_1_root": stage_roots[0][1], "stage_3_root": stage_roots[2][1], "artifact_absence_evidence": absence},
        {"stage_1_root": stage_roots[0][1], "image_rows": image_rows, "fresh_runtime_evidence_root": fresh["fresh_runtime_evidence_root"]},
        {"stage_1_root": stage_roots[0][1], "stage_2_root": stage_roots[1][1], "actor_rows": actor_rows, "fresh_runtime_evidence_root": fresh["fresh_runtime_evidence_root"]},
        {"stage_2_root": stage_roots[1][1], "stage_3_root": stage_roots[2][1], "cargo_lock_sha256": sha256(bytes.fromhex(cargo["lock_hex"])).hexdigest(), "cargo_material_identity": cargo_material, "cargo_material_identity_root": fresh["cargo"]["material_identity_root"], "cargo_snapshot_evidence": cargo_snapshot, "cargo_snapshot_evidence_root": fresh["cargo"]["snapshot_evidence_root"], "cargo_tree_evidence": cargo_tree, "cargo_tree_evidence_root": fresh["cargo"]["tree_evidence_root"], "offline_build_identity": offline, "offline_build_identity_root": AD.fresh_runtime_evidence_object_root_v1("OFFLINE_BUILD_TRANSCRIPT", "rust", offline), "fresh_runtime_evidence_root": fresh["fresh_runtime_evidence_root"]},
        {"stage_2_root": stage_roots[1][1], "stage_3_root": stage_roots[2][1], "seccomp_rows": seccomp_rows, "binary_identity": binary, "binary_identity_root": fresh["binary"]["identity_root"], "fresh_runtime_evidence_root": fresh["fresh_runtime_evidence_root"]},
        {"stage_1_root": stage_roots[0][1], "planned_command_registry_sha256": sha256(_j(planned_commands)).hexdigest(), "command_mount_resource_policy_sha256": AD.command_mount_resource_policy_root_v1(config_bytes), "prelaunch_policy_bound": True},
        {"stage_1_root": stage_roots[0][1], "qualification_authority": deepcopy(AD.ACTUAL_ADMISSION_QUALIFICATION_AUTHORITY), "closed_q1_authority": deepcopy(AD.ACTUAL_ADMISSION_CLOSED_Q1_AUTHORITY)},
        {"prior_stage_root_rows": stage_roots, "policy_name": "FRESH_SOURCE_IMAGE_RUNTIME_SNAPSHOT_REPLAY_BEFORE_PREDICATE19", "policy_bound_at_admission": True, "fulfilled_at_admission": False},
        {"stage_1_root": stage_roots[0][1], "artifact_path": artifact_path, "policy_name": "DIRFD_NOFOLLOW_FSYNC_LINK_NOREPLACE_UNLINK_FSYNC", "policy_bound_at_admission": True, "fulfilled_at_admission": False},
    ]
    bundle = AD.build_actual_precondition_bundle_v1(commit, config_bytes, artifact_path, work, stage_roots, preimages)
    decision = AD.build_actual_admission_decision_v1(commit, config_bytes, artifact_path, _ATTEMPT_NONCE, bundle)
    boundary = AD.build_stage3_to4_admission_boundary_v1(commit, config_bytes, artifact_path, bundle, decision)
    boundary_payload = _j(boundary)
    issued = AD.build_actual_admission_issued_marker_evidence_v1(
        decision["attempt_id"], boundary["boundary_root"], boundary_payload,
        file_device=1, file_inode=70, file_nlink=1, file_mode=0o444,
        work_root_device=1, work_root_inode=2, work_root_mode=0o700,
    )
    issue = AD.build_actual_admission_issue_record_v1(boundary, issued)
    spending = AD.build_actual_admission_spending_intent_v1(issue)
    consumed = AD.build_actual_admission_consumed_marker_evidence_v1(
        issue, spending, spending_file_device=1, spending_file_inode=71,
        spending_file_nlink=1, spending_file_mode=0o444, file_device=1,
        file_inode=70, file_nlink=2, file_mode=0o444,
        work_root_device=1, work_root_inode=2, work_root_mode=0o700,
    )
    bindings = {
        role_id: _mount_binding(
            role_id,
            actors[role_id - 1]["command"],
            fresh,
            five_sidecars,
            endpoint,
            config,
        )
        for role_id in (1, 2, 3)
    }
    launches = [_mount_launch_replay(bindings[role_id]) for role_id in (1, 2)]
    checkpoint_1 = AD.build_fresh_runtime_checkpoint_v1(
        commit, artifact_path, 1, decision["attempt_id"],
        boundary["boundary_root"], issue["issue_record_root"],
        consumed["consumed_marker_root"], fresh, fresh, absence,
        [bindings[1], bindings[2]], None, None,
    )
    stage5_live = AD.build_actual_admission_live_marker_replay_v1(
        "STAGE_05_BEFORE_EVIDENCE", issue, consumed, work_root_device=1,
        work_root_inode=2, work_root_nlink=2, work_root_mode=0o700,
        issued_file_device=1, issued_file_inode=70, issued_file_nlink=2,
        consumed_file_device=1, consumed_file_inode=70,
        consumed_file_nlink=2, spending_file_device=1,
        spending_file_inode=71, spending_file_nlink=1,
    )
    injected = {
        "actual_admission_attempt_id": decision["attempt_id"],
        "actual_admission_boundary_root": boundary["boundary_root"],
        "actual_admission_issue_record_root": issue["issue_record_root"],
        "actual_admission_consumed_marker_evidence": consumed,
        "actual_admission_work_root_replay": {
            "schema_version": "hegel-phase3a-q05b-admission-work-root-replay/1",
            "absolute_path": work["absolute_path"], "device": 1, "inode": 2,
            "nlink": 2, "mode": 0o700,
            "path_matches_anchored_descriptor": True,
        },
        "actual_admission_consume_git_source_transcript": git,
        "actual_admission_consume_artifact_absence": absence,
        "actual_admission_fresh_checkpoint_root_rows": [[
            1, AD.ACTUAL_FRESH_RUNTIME_CHECKPOINT_REGISTRY[0][1],
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
    stage5 = AD.build_actual_stage_5_evidence_v1(
        commit,
        [actors[0]["control_evidence"], actors[1]["control_evidence"]],
        five_sidecars,
        endpoint,
        strict_endpoint_replay_roots,
        injected,
    )
    stage5 = AD.validate_actual_stage_5_evidence_v1(
        stage5,
        commit,
        issue_record=issue,
        consumed_marker_evidence=consumed,
        checkpoint_1=checkpoint_1,
        mount_launch_replay_rows=launches,
    )
    dynamic = AD.build_dynamic_mount_authority_set_v1(
        commit, stage5, five_sidecars["python_output_tree"],
        five_sidecars["rust_output_tree"], endpoint["sealed_stdout_tree"],
        issue_record=issue,
        consumed_marker_evidence=consumed,
        checkpoint_1=checkpoint_1,
        mount_launch_replay_rows=launches,
    )
    checkpoints = [checkpoint_1]
    for checkpoint_id in (2, 3):
        checkpoint_bindings = {2: [bindings[3]], 3: [bindings[1], bindings[2], bindings[3]]}[checkpoint_id]
        checkpoints.append(AD.build_fresh_runtime_checkpoint_v1(
            commit, artifact_path, checkpoint_id, decision["attempt_id"],
            boundary["boundary_root"], issue["issue_record_root"],
            consumed["consumed_marker_root"], fresh, fresh, absence,
            checkpoint_bindings, dynamic, stage5,
            stage_5_issue_record=issue,
            stage_5_consumed_marker_evidence=consumed,
            stage_5_checkpoint_1=checkpoint_1,
            stage_5_mount_launch_replay_rows=launches,
        ))
    live = AD.build_actual_admission_live_marker_replay_v1(
        "PRE_ARTIFACT_ASSEMBLY", issue, consumed, work_root_device=1,
        work_root_inode=2, work_root_nlink=2, work_root_mode=0o700,
        issued_file_device=1, issued_file_inode=70, issued_file_nlink=2,
        consumed_file_device=1, consumed_file_inode=70,
        consumed_file_nlink=2, spending_file_device=1,
        spending_file_inode=71, spending_file_nlink=1,
    )
    work_replay = {
        "schema_version": "hegel-phase3a-q05b-admission-work-root-replay/1",
        "absolute_path": work["absolute_path"], "device": 1, "inode": 2,
        "nlink": 2, "mode": 0o700,
        "path_matches_anchored_descriptor": True,
    }
    return A.build_actual_admission_artifact_evidence_v1(
        source_commit=commit, artifact_path=artifact_path,
        commit_a_config_bytes=config_bytes,
        commit_a_config_git_blob_oid=config_oid,
        prior_stage_evidence_rows=stages, issue_record=issue,
        consumed_marker_evidence=consumed,
        consume_work_root_replay=work_replay,
        consume_git_source_transcript=git,
        consume_artifact_absence_evidence=absence,
        fresh_runtime_checkpoint_rows=checkpoints,
        pre_artifact_live_marker_replay=live,
        anti_replay_scope=AD.ACTUAL_ADMISSION_RUN_LOCAL_ANTI_REPLAY_SCOPE,
        stage_5_evidence=stage5,
        stage_5_actor_completion_rows=[
            actors[0]["control_evidence"], actors[1]["control_evidence"]
        ],
        stage_5_strict_endpoint_replay_roots=strict_endpoint_replay_roots,
        stage_5_live_marker_replay=stage5_live,
        stage_5_mount_launch_replay_rows=launches,
        five_sidecars=five_sidecars, endpoint_stdout_set=endpoint,
    )


@pytest.fixture(scope="module")
def actual_artifact(request: pytest.FixtureRequest) -> dict[str, object]:
    limits = C.PreflightLimitsV1(maximum_ast_node_count=3)
    leaf = W.full_v16_leaf_manifest_v1()
    snapshots = tuple(S.build_q1_partition_snapshot_v1(index, limits=limits) for index in (1, 2))
    partitions = tuple(W.node3_partition_evidence_v1(snapshot, P.records_from_partition_snapshot_v1(snapshot), V.build_q1_semantic_coverage_v1(snapshot)) for snapshot in snapshots)
    sidecar = W.sidecar_manifest_v1(leaf, *partitions)
    golden = W.node3_golden_manifest_v1(leaf, snapshots[0], partitions[0], snapshots[1], partitions[1], sidecar)
    payloads = (leaf.canonical_bytes, partitions[0].canonical_bytes, partitions[1].canonical_bytes, sidecar.canonical_bytes, golden.canonical_bytes)
    roots = (leaf.manifest_root, partitions[0].evidence_root, partitions[1].evidence_root, sidecar.manifest_root, golden.manifest_root)
    sidecar_rows = [{"path": path.decode(), "mode": 0o444, "length": len(payload), "raw_sha256": sha256(payload).hexdigest(), "content_root": root.hex(), "cbor_hex": payload.hex()} for path, payload, root in zip(W.ORDERED_OUTPUT_RELATIVE_PATHS, payloads, roots, strict=True)]

    config = json.loads((ROOT / "config/phase3_q05b_dual_isolation_v1.json").read_text())
    config["engineering_status"] = AD.COMMIT_A_ACTUAL_ENGINEERING_STATUS
    config["actual_preconditions"] = deepcopy(AD.COMMIT_A_ACTUAL_PRECONDITIONS_V1)
    config["held_actor_protocol"]["wrapper_script_sha256"] = sha256(b"synthetic-held-wrapper").hexdigest()
    runtime_seccomp = _RUNTIME_SECCOMP_PAYLOAD
    build_seccomp = _BUILD_SECCOMP_PAYLOAD
    config["seccomp"]["runtime_profile_sha256"] = sha256(runtime_seccomp).hexdigest(); config["seccomp"]["build_profile_sha256"] = sha256(build_seccomp).hexdigest()
    actor_paths = {
        "PYTHON_ENDPOINT": ["config/phase3_q05b_dual_isolation_v1.json", "tools/synthetic_python.py"],
        "RUST_ENDPOINT": ["rust/q1_archive_projection_oracle/Cargo.lock", "rust/q1_archive_projection_oracle/src/main.rs"],
        "TRUSTED_HOST_REPLAY": [config["seccomp"]["build_profile"], config["seccomp"]["runtime_profile"], "src/hegel_machine/phase3_q05b_host_replay_v1.py", "src/hegel_machine/phase3_q05b_negative_vectors_v1.py"],
    }
    for paths in actor_paths.values(): paths.sort()
    config["source_allowlist_policy"]["actor_rows"] = [[index, actor, len(actor_paths[actor]), sha256(_j(actor_paths[actor])).hexdigest()] for index, actor in enumerate(("PYTHON_ENDPOINT", "RUST_ENDPOINT", "TRUSTED_HOST_REPLAY"), 1)]
    static_policy = {
        key: value for key, value in config.items()
        if key not in {"engineering_status", "actual_preconditions"}
    }
    synthetic_static_root = sha256(
        AD.ACTUAL_COMMIT_A_STATIC_POLICY_ROOT_DOMAIN + _j(static_policy)
    ).hexdigest()
    synthetic_command_root = AD.command_mount_resource_policy_root_v1(config)
    policy_patch = pytest.MonkeyPatch()
    policy_patch.setattr(AD, "EXPECTED_COMMIT_A_STATIC_POLICY_ROOT", synthetic_static_root)
    policy_patch.setattr(
        AD, "EXPECTED_COMMAND_MOUNT_RESOURCE_POLICY_ROOT", synthetic_command_root
    )
    request.addfinalizer(policy_patch.undo)
    layout = _production_layout()
    cargo_material = _cargo_material(layout["cargo_home"])
    source_payloads = {
        "config/phase3_q05b_dual_isolation_v1.json": (0o100644, _j(config)),
        "tools/synthetic_python.py": (0o100644, b"VALUE = 1\n"),
        "rust/q1_archive_projection_oracle/Cargo.lock": (0o100644, cargo_material["lock"]),
        "rust/q1_archive_projection_oracle/src/main.rs": (0o100644, b"fn main() {}\n"),
        config["seccomp"]["build_profile"]: (0o100644, build_seccomp),
        config["seccomp"]["runtime_profile"]: (0o100644, runtime_seccomp),
        "src/hegel_machine/phase3_q05b_host_replay_v1.py": (0o100644, b"HOST = True\n"),
        "src/hegel_machine/phase3_q05b_negative_vectors_v1.py": (0o100644, b"NEGATIVE = True\n"),
    }
    commit, tree_oid, blob_rows, tree_rows, commit_hex = _git_closure(source_payloads)
    table = {path: (mode, oid, bytes.fromhex(payload_hex)) for path, mode, oid, payload_hex in blob_rows}
    assert A._isolation_config_v1(table) == config
    snapshot_roots = {
        "PYTHON_ENDPOINT": layout["python_snapshot"],
        "RUST_ENDPOINT": layout["rust_snapshot"],
        "TRUSTED_HOST_REPLAY": layout["host_snapshot"],
    }
    actors = [_source(actor, commit, actor_paths[actor], table, snapshot_roots[actor]) for actor in actor_paths]
    actor_map = {actor["actor_id"]: actor for actor in actors}
    authority = _docker_authority(commit)
    slots = {row["slot"]: row for row in authority["ordered_slot_rows"]}

    sidecar_map = dict(sorted(zip((path.decode() for path in W.ORDERED_OUTPUT_RELATIVE_PATHS), payloads, strict=True)))
    five_sidecars = {"canonical_rows": sidecar_rows, "python_output_tree": _tree(sidecar_map, layout["python_output"], ("neutral", "preimages")), "rust_output_tree": _tree(sidecar_map, layout["rust_output"], ("neutral", "preimages"))}
    pinned_images = [["python", _image_evidence(config["images"]["python_endpoint"], config["runtime_command_inspect_policy"]["environment_rows"][0][2])], ["rust", _image_evidence(config["images"]["rust_build"], config["runtime_command_inspect_policy"]["environment_rows"][1][2])]]
    runtime_seccomp_evidence = _seccomp_from_snapshot(
        actor_map["TRUSTED_HOST_REPLAY"]["snapshot_identity"],
        config["seccomp"]["runtime_profile"],
        runtime_seccomp,
    )
    build_seccomp_evidence = _seccomp_from_snapshot(
        actor_map["TRUSTED_HOST_REPLAY"]["snapshot_identity"],
        config["seccomp"]["build_profile"],
        build_seccomp,
    )
    cargo = _cargo(
        config,
        actor_map["RUST_ENDPOINT"],
        build_seccomp,
        source_commit=commit,
        authority=authority,
        material=cargo_material,
        layout=layout,
        build_seccomp_evidence=build_seccomp_evidence,
    )
    cargo["rust_image_inspect_hex"] = pinned_images[1][1]["raw_inspect_hex"]
    cargo["rust_image_inspect_sha256"] = pinned_images[1][1]["raw_inspect_sha256"]
    actor_map["RUST_ENDPOINT"]["runtime_identity_sha256"] = cargo["binary_runtime_identity_sha256"]
    stdout_python = _actor_stdout(actor_map["PYTHON_ENDPOINT"], dict(W.ACTOR_IMPLEMENTATION_ID_REGISTRY)["PYTHON_ENDPOINT"], payloads, sidecar, golden)
    stdout_rust = _actor_stdout(actor_map["RUST_ENDPOINT"], dict(W.ACTOR_IMPLEMENTATION_ID_REGISTRY)["RUST_ENDPOINT"], payloads, sidecar, golden)
    stdout_manifest = H.sealed_actor_stdout_manifest_bytes_v1(stdout_python, stdout_rust)
    endpoint = {"python_stdout_hex": stdout_python.hex(), "rust_stdout_hex": stdout_rust.hex(), "manifest_hex": stdout_manifest.hex(), "sealed_stdout_tree": _tree({"manifest.json": stdout_manifest, "python.stdout": stdout_python, "rust.stdout": stdout_rust}, layout["stdout_root"], ())}

    actor_map["PYTHON_ENDPOINT"]["command"] = _command(config, 1, [(actor_map["PYTHON_ENDPOINT"]["snapshot_identity"]["root_path"], "/snapshot", True), (layout["python_output"], "/output", False), (layout["python_control"], "/control", False)], "", actor_map["PYTHON_ENDPOINT"]["runtime_identity_sha256"], slots["PYTHON_ENDPOINT"], layout)
    actor_map["RUST_ENDPOINT"]["command"] = _command(config, 2, [(cargo["binary_path"], "/runtime/hegel-q1-archive-projection-oracle", True), (layout["rust_output"], "/output", False), (layout["rust_control"], "/control", False)], "", actor_map["RUST_ENDPOINT"]["runtime_identity_sha256"], slots["RUST_ENDPOINT"], layout)
    actor_map["TRUSTED_HOST_REPLAY"]["command"] = _command(config, 3, [(actor_map["TRUSTED_HOST_REPLAY"]["snapshot_identity"]["root_path"], "/snapshot", True), (layout["python_output"], "/inputs/python", True), (layout["rust_output"], "/inputs/rust", True), (layout["stdout_root"] + "/python.stdout", "/inputs/stdout/python.stdout", True), (layout["stdout_root"] + "/rust.stdout", "/inputs/stdout/rust.stdout", True), (layout["stdout_root"] + "/manifest.json", "/inputs/stdout/manifest.json", True), (layout["host_control"], "/control", False), (layout["host_staging"], "/staging", False)], actor_map["TRUSTED_HOST_REPLAY"]["source_evidence"]["source_identity_sha256"], actor_map["TRUSTED_HOST_REPLAY"]["runtime_identity_sha256"], slots["TRUSTED_HOST_REPLAY"], layout)

    replayed = A._sidecars_v1(sidecar_rows)
    dual = A._dual_replay_v1(endpoint, replayed, bytes.fromhex(actor_map["TRUSTED_HOST_REPLAY"]["source_evidence"]["source_identity_sha256"]), bytes.fromhex(actor_map["TRUSTED_HOST_REPLAY"]["runtime_identity_sha256"]))
    negative_object = _negative_object_for_test(); negative_cbor = canonical_cbor_encode(negative_object.canonical_object()); negative_roots = dict(negative_object.category_roots)
    negative = {"canonical_cbor_hex": negative_cbor.hex(), "corpus_root": negative_object.corpus_root.hex(), "category13_root": negative_roots[13].hex(), "category18_root": negative_roots[18].hex()}
    witness = H.host_semantic_witness_bytes_v1(dual, negative_cbor, negative_object.corpus_root, negative_object.category_roots)
    witness_value = H.decode_host_semantic_witness_v1(witness, dual, negative_cbor, negative_object.corpus_root, negative_object.category_roots)
    loaded_rows = [["hegel_machine", None, None], ["hegel_machine.phase3_q05b_host_replay_v1", "src/hegel_machine/phase3_q05b_host_replay_v1.py", sha256(table["src/hegel_machine/phase3_q05b_host_replay_v1.py"][2]).hexdigest()], ["hegel_machine.phase3_q05b_negative_vectors_v1", "src/hegel_machine/phase3_q05b_negative_vectors_v1.py", sha256(table["src/hegel_machine/phase3_q05b_negative_vectors_v1.py"][2]).hexdigest()]]
    loaded_root = sha256(b"HEGEL/Q05B/HOST/LOADED_MODULE_CLOSURE/V1\x00" + _j(loaded_rows)).hexdigest()
    host_control_value = {"action_id": "trusted-host-semantic-replay-v1", "actor_id": "TRUSTED_HOST_REPLAY", "file_count": 6, "final_isolation_root": None, "implementation_id": "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_HOST_REPLAY_V1", "loaded_module_root": loaded_root, "loaded_module_rows": loaded_rows, "q1_formal_roots": None, "q1_gate_count": 0, "q1_gate_mask": 0, "q1_output_slots": [None] * 8, "q1_state": "NOT_RUN", "qualification_receipt": None, "runtime_identity_sha256": actor_map["TRUSTED_HOST_REPLAY"]["runtime_identity_sha256"], "schema_version": "hegel-phase3a-q05b-host-semantic-control-envelope/1", "semantic_replay_root": dual.dual_replay_root.hex(), "source_identity_sha256": actor_map["TRUSTED_HOST_REPLAY"]["source_evidence"]["source_identity_sha256"], "status": "HOST_SEMANTIC_WITNESS_EMITTED_NOT_RECEIPT", "witness_length": len(witness), "witness_relative_path": "host-semantic-witness.json", "witness_root": witness_value["witness_root"], "witness_sha256": sha256(witness).hexdigest()}
    host_control = _j(host_control_value)

    stdouts = (stdout_python, stdout_rust, host_control); resources = []
    for role, (actor, stdout) in enumerate(zip(actors, stdouts, strict=True), 1):
        completion = _control_identity(actor["actor_id"], stdout, False)
        resource = _resource(actor["command"], role, config, chr(96 + role) * 64, completion["manifest_sha256"], actor["actor_id"])
        slot = slots[actor["actor_id"]]
        actor["control_evidence"] = _control(actor, stdout, resource, config, runtime_seccomp, authority, slot, runtime_seccomp_evidence); resources.append(resource)
    staged = [[row["path"], len(payload), sha256(payload).hexdigest(), 0o444] for row, payload in zip(sidecar_rows, payloads, strict=True)]
    host_binding = {"host_actor_row": actor_map["TRUSTED_HOST_REPLAY"], "host_control_sha256": sha256(host_control).hexdigest(), "host_final_resource": resources[2], "loaded_module_root": loaded_root, "semantic_replay_root": dual.dual_replay_root.hex(), "witness_root": witness_value["witness_root"]}
    staging_payloads = dict(sorted({**{f"sidecars/{path}": payload for path, payload in sidecar_map.items()}, "host-semantic-witness.json": witness}.items()))
    host_stage = {"staged_sidecar_rows": staged, "witness_hex": witness.hex(), "witness_root": witness_value["witness_root"], "host_control_stdout_hex": host_control.hex(), "loaded_module_rows": loaded_rows, "loaded_module_root": loaded_root, "staging_tree": _tree(staging_payloads, layout["host_staging"], ("sidecars", "sidecars/neutral", "sidecars/preimages")), "host_execution_binding_preimage": host_binding}

    scratch_registries = ([[root.hex() for root in replay.scratch_ledger_roots] for replay in dual.python.partition_replays], [[root.hex() for root in replay.scratch_ledger_roots] for replay in dual.rust.partition_replays], witness_value["host_scratch_partition_roots"])
    producers = (dual.python.host_replay_root.hex(), dual.rust.host_replay_root.hex(), witness_value["host_scratch_evidence_root"])
    scratch = []
    for actor, roots_value, producer in zip(("PYTHON_ENDPOINT", "RUST_ENDPOINT", "TRUSTED_HOST_REPLAY"), scratch_registries, producers, strict=True):
        preimage = {"actor_id": actor, "partition_scratch_ledger_roots": roots_value, "producer_replay_root": producer}
        scratch.append({**preimage, "scratch_root": A._json_root("HEGEL/Q05B/ACTUAL/SCRATCH_ACTOR/V1", preimage).hex()})
    source = {"source_commit": commit, "source_commit_raw20_hex": commit, "project_tree_prefix": "Hegel Machine", "git_blob_payload_table": blob_rows, "git_commit_object_hex": commit_hex, "git_tree_object_rows": tree_rows, "pinned_image_rows": pinned_images, "external_commit_replay": {"commit": commit, "tree_oid": tree_oid, "head_clean_before": True, "head_clean_after": True}, "actor_source_path_rows": [[actor, actor_paths[actor]] for actor in actor_paths], "full_leaf_manifest_root": leaf.manifest_root.hex(), "q0_receipt_root": W.Q0_SATURATION_RECEIPT_ROOT_FROM_Q1_PREREGISTRATION.hex(), "q1_projection_profile_root": golden.q1_projection_profile_root.hex(), "q1_semantic_binding_root": golden.q1_semantic_binding_root.hex(), "qualification_predicate_registry_root": W.QUALIFICATION_PREDICATE_REGISTRY_ROOT.hex(), "qualification_tag_registry_root": W.QUALIFICATION_TAG_REGISTRY_ROOT.hex(), "qualification_wire_profile_root": W.qualification_wire_profile_root_v1().hex()}
    actual_admission = _actual_admission_section(
        config=config,
        config_oid=table["config/phase3_q05b_dual_isolation_v1.json"][1],
        commit=commit,
        source_wire_profile=source,
        layout=layout,
        actors=actors,
        pinned_images=pinned_images,
        cargo=cargo,
        runtime_seccomp=runtime_seccomp,
        build_seccomp=build_seccomp,
        five_sidecars=five_sidecars,
        endpoint=endpoint,
        strict_endpoint_replay_roots=[
            dual.python.host_replay_root.hex(),
            dual.rust.host_replay_root.hex(),
        ],
    )
    evidence = {"source_wire_profile": source, "five_sidecars": five_sidecars, "endpoint_stdout_set": endpoint, "host_stage": host_stage, "actor_rows": actors, "cargo_build_binary": cargo, "final_resource_rows": resources, "negative_corpus": negative, "scratch_rows": scratch, "actual_admission": actual_admission}
    isolation = {"actual_admission": actual_admission, "actor_rows": actors, "cargo_build_binary": cargo, "endpoint_stdout_set": endpoint, "final_resource_rows": resources, "five_sidecars": five_sidecars, "host_stage": host_stage, "negative_corpus": negative, "scratch_rows": scratch, "source_wire_profile": source}
    evidence["semantic_execution"] = {"semantic_component_root": dual.predicate11_semantic_component_root.hex(), "host_execution_binding_preimage": host_binding, "resource_preimage": {"final_resource_rows": resources}, "isolation_preimage": isolation, "bundle_preimage": {"actual_admission_evidence_root": actual_admission["actual_admission_evidence_root"], "five_sidecars": five_sidecars, "host_witness_root": witness_value["witness_root"], "scratch_rows": scratch, "semantic_replay_root": dual.dual_replay_root.hex()}}
    return A.build_actual_artifact_v1(evidence)


def test_strict_roundtrip(actual_artifact: dict[str, object]) -> None:
    candidate = A.replay_actual_evidence_1_19_v1(actual_artifact["sections"])
    assert candidate["qualification_count"] == 19
    assert candidate["actual_admission_evidence_root"] == actual_artifact[
        "derived"
    ]["actual_admission_evidence_root"]
    payload = A.canonical_actual_artifact_bytes_v1(actual_artifact)
    assert A.decode_and_replay_actual_artifact_v1(payload) == actual_artifact
    assert A.actual_artifact_summary_v1(actual_artifact)["artifact_set_root"] == actual_artifact["derived"]["artifact_set_root"]


def test_proc_nofile_kernel_space_padding_accepts() -> None:
    sample = _direct_proc_nofile_sample(
        b"Max open files            256                  256"
        b"                  files     \n"
    )
    payload = bytes.fromhex(sample["proc_limits_payload_hex"])
    assert payload.endswith(b"files     \n")
    replayed = A._replay_live_sample_v1(
        sample,
        [1, "PYTHON_ENDPOINT", "0-11"],
        1,
        False,
    )
    assert replayed["nofile_soft"] == 256
    assert replayed["nofile_hard"] == 256


@pytest.mark.parametrize(
    "payload",
    (
        b"Max open files\t256 256 files     \n",
        b"Max open files            256                  256                  files extra\n",
        b"Max open files            257                  256                  files     \n",
    ),
)
def test_proc_nofile_rejects_non_kernel_rows(
    payload: bytes,
) -> None:
    sample = _direct_proc_nofile_sample(payload)
    with pytest.raises(A.Q05BActualArtifactError) as rejected:
        A._replay_live_sample_v1(
            sample,
            [1, "PYTHON_ENDPOINT", "0-11"],
            1,
            False,
        )
    assert rejected.value.code == "REJECT_Q05B_ARTIFACT_RESOURCE"


@pytest.mark.parametrize(
    ("relative_path", "absolute_path", "payload"),
    (
        (
            "config/seccomp/runtime.json",
            "/sealed/runtime-seccomp.json",
            _RUNTIME_SECCOMP_PAYLOAD,
        ),
        (
            "config/seccomp/build.json",
            "/sealed/build-seccomp.json",
            _BUILD_SECCOMP_PAYLOAD,
        ),
    ),
)
def test_docker29_inline_seccomp_accepts_exact_sealed_policy(
    relative_path: str,
    absolute_path: str,
    payload: bytes,
) -> None:
    evidence = _seccomp(relative_path, absolute_path, payload)
    A._validate_inspect_security_options_v1(
        _docker29_security_options(payload),
        ["no-new-privileges", f"seccomp={absolute_path}"],
        payload,
        evidence,
        "focused Docker 29 inspect",
    )


def test_docker29_inline_seccomp_accepts_canonical_key_order_normalization(
) -> None:
    payload = b'{\n  "z": [1, true, null],\n  "a": {"b": "c"}\n}\n'
    absolute_path = "/sealed/runtime-seccomp.json"
    evidence = _seccomp("config/seccomp/runtime.json", absolute_path, payload)
    A._validate_inspect_security_options_v1(
        [
            "no-new-privileges",
            'seccomp={"a":{"b":"c"},"z":[1,true,null]}',
        ],
        ["no-new-privileges", f"seccomp={absolute_path}"],
        payload,
        evidence,
        "focused Docker 29 canonical inspect",
    )


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("missing", "REJECT_Q05B_ARTIFACT_ISOLATION"),
        ("reordered", "REJECT_Q05B_ARTIFACT_ISOLATION"),
        ("extra", "REJECT_Q05B_ARTIFACT_ISOLATION"),
        ("nnp_suffix", "REJECT_Q05B_ARTIFACT_ISOLATION"),
        ("command_path", "REJECT_Q05B_ARTIFACT_ISOLATION"),
        ("inline_path", "REJECT_Q05B_ARTIFACT_JSON"),
        ("duplicate", "REJECT_Q05B_ARTIFACT_JSON"),
        ("nonfinite", "REJECT_Q05B_ARTIFACT_JSON"),
        ("float", "REJECT_Q05B_ARTIFACT_JSON"),
        ("inner_type", "REJECT_Q05B_ARTIFACT_ISOLATION"),
        ("top_array", "REJECT_Q05B_ARTIFACT_ISOLATION"),
        ("raw_sha", "REJECT_Q05B_ARTIFACT_ISOLATION"),
        ("wrong_policy", "REJECT_Q05B_ARTIFACT_ISOLATION"),
    ),
)
def test_docker29_inline_seccomp_rejects_registry_and_inner_aliases(
    mutation: str,
    expected_code: str,
) -> None:
    absolute_path = "/sealed/runtime-seccomp.json"
    evidence = _seccomp(
        "config/seccomp/runtime.json",
        absolute_path,
        _RUNTIME_SECCOMP_PAYLOAD,
    )
    observed: object = _docker29_security_options(_RUNTIME_SECCOMP_PAYLOAD)
    command_security = [
        "no-new-privileges",
        f"seccomp={absolute_path}",
    ]
    if mutation == "missing":
        observed = ["no-new-privileges"]
    elif mutation == "reordered":
        observed = list(reversed(observed))
    elif mutation == "extra":
        observed = [*observed, "apparmor=unconfined"]
    elif mutation == "nnp_suffix":
        observed[0] = "no-new-privileges:true"
    elif mutation == "command_path":
        command_security[1] = "seccomp=/sealed/other.json"
    elif mutation == "inline_path":
        observed[1] = f"seccomp={absolute_path}"
    elif mutation == "duplicate":
        observed[1] = (
            'seccomp={"synthetic":"runtime-seccomp",'
            '"synthetic":"runtime-seccomp"}'
        )
    elif mutation == "nonfinite":
        observed[1] = 'seccomp={"synthetic":NaN}'
    elif mutation == "float":
        observed[1] = 'seccomp={"synthetic":1.0}'
    elif mutation == "inner_type":
        observed[1] = 'seccomp={"synthetic":true}'
    elif mutation == "top_array":
        observed[1] = 'seccomp=["runtime-seccomp"]'
    elif mutation == "raw_sha":
        evidence["payload_sha256"] = "0" * 64
    else:
        assert mutation == "wrong_policy"
        observed[1] = "seccomp=" + _BUILD_SECCOMP_PAYLOAD.decode(
            "utf-8", "strict"
        )
    with pytest.raises(A.Q05BActualArtifactError) as rejected:
        A._validate_inspect_security_options_v1(
            observed,
            command_security,
            _RUNTIME_SECCOMP_PAYLOAD,
            evidence,
            f"focused Docker 29 inspect {mutation}",
        )
    assert rejected.value.code == expected_code


def test_docker29_inline_seccomp_rejects_bool_int_alias() -> None:
    payload = b'{"defaultErrnoRet":1}\n'
    absolute_path = "/sealed/runtime-seccomp.json"
    evidence = _seccomp("config/seccomp/runtime.json", absolute_path, payload)
    with pytest.raises(A.Q05BActualArtifactError) as rejected:
        A._validate_inspect_security_options_v1(
            [
                "no-new-privileges",
                'seccomp={"defaultErrnoRet":true}',
            ],
            ["no-new-privileges", f"seccomp={absolute_path}"],
            payload,
            evidence,
            "focused Docker 29 bool/int alias",
        )
    assert rejected.value.code == "REJECT_Q05B_ARTIFACT_ISOLATION"


def test_docker29_inline_seccomp_fixture_covers_every_inspect_surface(
    actual_artifact: dict[str, object],
) -> None:
    sections = actual_artifact["sections"]
    runtime_expected = _docker29_security_options(_RUNTIME_SECCOMP_PAYLOAD)
    for actor in sections["actor_rows"]:
        control = actor["control_evidence"]
        for sample in control["final_resource_transcript"][
            "live_sample_objects"
        ]:
            for field in ("inspect_payload_hex", "inspect_after_payload_hex"):
                document = json.loads(bytes.fromhex(sample[field]))[0]
                assert document["HostConfig"]["SecurityOpt"] == runtime_expected
        post = json.loads(bytes.fromhex(control["post_exit_inspect_hex"]))[0]
        assert post["HostConfig"]["SecurityOpt"] == runtime_expected

    build_expected = _docker29_security_options(_BUILD_SECCOMP_PAYLOAD)
    cargo = sections["cargo_build_binary"]
    for name in ("rust_test", "rust_release_build"):
        for field in ("live_inspect_hex", "post_inspect_hex"):
            document = json.loads(bytes.fromhex(cargo[name][field]))[0]
            assert document["HostConfig"]["SecurityOpt"] == build_expected


def test_layered_fixture_build(actual_artifact: dict[str, object]) -> None:
    assert actual_artifact["derived"]["qualification_count"] == 20
    assert len(A.SECTION_NAMES) == 11
    assert "actual_admission" in actual_artifact["sections"]


def test_actual_admission_section_is_causal_and_noncyclic(
    actual_artifact: dict[str, object],
) -> None:
    section = actual_artifact["sections"]["actual_admission"]
    admission_root = section["actual_admission_evidence_root"]
    body = deepcopy(section)
    body.pop("actual_admission_evidence_root")
    assert admission_root not in json.dumps(body, sort_keys=True)
    assert "actual_admission_evidence_root" not in section["root_registry"]
    assert [
        row["stage_id"] for row in section["prior_stage_evidence_rows"]
    ] == [1, 2, 3]
    serialized = json.dumps(section, sort_keys=True)
    for forbidden in (
        "artifact_set_root",
        "final_delivery",
        "postpublication",
        "prepublication",
        "stage_08",
        "stage_09",
        "stage_10",
    ):
        assert forbidden not in serialized
    semantic = actual_artifact["sections"]["semantic_execution"]
    assert semantic["isolation_preimage"]["actual_admission"] == section
    assert semantic["bundle_preimage"]["actual_admission_evidence_root"] == (
        admission_root
    )
    assert actual_artifact["derived"]["actual_admission_evidence_root"] == (
        admission_root
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("fresh_runtime_checkpoint_rows", 0, "checkpoint_id"), True),
        (("consumed_marker_evidence", "spent_before_preflight"), 1),
        (("root_registry", "boundary_root"), "00" * 32),
        (("stage_5_live_marker_replay", "work_root_nlink"), True),
        ((
            "stage_5_mount_launch_replay_rows", 0,
            "all_paths_match_prelaunch_held_descriptors",
        ), 1),
        (("pre_artifact_live_marker_replay", "work_root_nlink"), True),
    ),
)
def test_actual_admission_tamper_and_type_alias_fail_closed(
    actual_artifact: dict[str, object],
    path: tuple[object, ...],
    replacement: object,
) -> None:
    sections = deepcopy(actual_artifact["sections"])
    cursor: object = sections["actual_admission"]
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(A.Q05BActualArtifactError) as failure:
        A.replay_actual_evidence_1_19_v1(sections)
    assert failure.value.code == "REJECT_Q05B_ARTIFACT_ADMISSION"


def test_commit_a_config_is_single_authority_and_type_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The module-scoped synthetic artifact fixture installs synthetic policy
    # roots until teardown.  Replay the production config under the production
    # roots captured before that fixture is entered, then restore the synthetic
    # roots for any later fixture consumers.
    with monkeypatch.context() as production_policy:
        production_policy.setattr(
            AD,
            "EXPECTED_COMMIT_A_STATIC_POLICY_ROOT",
            _PRODUCTION_COMMIT_A_STATIC_POLICY_ROOT,
        )
        production_policy.setattr(
            AD,
            "EXPECTED_COMMAND_MOUNT_RESOURCE_POLICY_ROOT",
            _PRODUCTION_COMMAND_MOUNT_RESOURCE_POLICY_ROOT,
        )
        current = (
            ROOT / "config/phase3_q05b_dual_isolation_v1.json"
        ).read_bytes()
        table = {
            "config/phase3_q05b_dual_isolation_v1.json": (
                0o100644,
                "00" * 20,
                current,
            )
        }
        decoded = A._isolation_config_v1(table)
        assert (
            decoded["engineering_status"]
            == AD.COMMIT_A_ACTUAL_ENGINEERING_STATUS
        )
        assert (
            decoded["actual_preconditions"]
            == AD.COMMIT_A_ACTUAL_PRECONDITIONS_V1
        )
        assert "current_actual_admitted" not in decoded["actual_preconditions"]

        tampered = deepcopy(decoded)
        tampered["actual_preconditions"]["actual_entrypoint_implemented"] = 1
        with pytest.raises(A.Q05BActualArtifactError) as alias:
            A._isolation_config_v1(
                {
                    "config/phase3_q05b_dual_isolation_v1.json": (
                        0o100644,
                        "00" * 20,
                        _j(tampered),
                    )
                }
            )
        assert alias.value.code == "REJECT_Q05B_ARTIFACT_SOURCE"
    with pytest.raises(A.Q05BActualArtifactError):
        A._require_type_exact_v1({"head_clean_before": 1}, {"head_clean_before": True}, "external")


def test_public_artifact_canonical_byte_cap_is_exact_and_type_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert type(A.ACTUAL_ARTIFACT_MAX_CANONICAL_BYTES) is int
    assert A.ACTUAL_ARTIFACT_MAX_CANONICAL_BYTES == 64 * 1024 * 1024

    class ExactLimitReached(Exception):
        pass

    def stop_after_size_gate(_payload: bytes) -> object:
        raise ExactLimitReached

    monkeypatch.setattr(A, "_strict_json", stop_after_size_gate)
    exact = b" " * A.ACTUAL_ARTIFACT_MAX_CANONICAL_BYTES
    with pytest.raises(ExactLimitReached):
        A.decode_and_replay_actual_artifact_v1(exact)
    del exact
    with pytest.raises(A.Q05BActualArtifactError) as oversized:
        A.decode_and_replay_actual_artifact_v1(
            b" " * (A.ACTUAL_ARTIFACT_MAX_CANONICAL_BYTES + 1)
        )
    assert oversized.value.code == "REJECT_Q05B_ARTIFACT_SIZE"
    for alias in (bytearray(b"{}\n"), memoryview(b"{}\n")):
        with pytest.raises(A.Q05BActualArtifactError) as type_alias:
            A.decode_and_replay_actual_artifact_v1(alias)
        assert type_alias.value.code == "REJECT_Q05B_ARTIFACT_SIZE"

    synthetic = {"synthetic": True}
    synthetic_payload = A._canonical_json(synthetic)
    monkeypatch.setattr(
        A,
        "_replay_actual_evidence_v1",
        lambda _evidence, *, candidate_only: synthetic,
    )
    monkeypatch.setattr(
        A, "ACTUAL_ARTIFACT_MAX_CANONICAL_BYTES", len(synthetic_payload)
    )
    assert A.build_actual_artifact_v1({}) == synthetic
    monkeypatch.setattr(
        A, "ACTUAL_ARTIFACT_MAX_CANONICAL_BYTES", len(synthetic_payload) - 1
    )
    with pytest.raises(A.Q05BActualArtifactError) as build_oversized:
        A.build_actual_artifact_v1({})
    assert build_oversized.value.code == "REJECT_Q05B_ARTIFACT_SIZE"
    with pytest.raises(A.Q05BActualArtifactError) as build_alias:
        A.build_actual_artifact_v1([])
    assert build_alias.value.code == "REJECT_Q05B_ARTIFACT_SCHEMA"


def test_cargo_schema_discriminator_is_exact() -> None:
    config = json.loads((ROOT / "config/phase3_q05b_dual_isolation_v1.json").read_text())
    build_seccomp = _BUILD_SECCOMP_PAYLOAD
    actor = {"source_evidence": {"source_identity_sha256": "11" * 32}, "snapshot_identity": {"root_path": "/sealed/rust-snapshot"}}
    pinned = _image_evidence(config["images"]["rust_build"], config["runtime_command_inspect_policy"]["environment_rows"][1][2])
    cargo = _cargo(config, actor, build_seccomp)
    cargo["rust_image_inspect_hex"] = pinned["raw_inspect_hex"]
    cargo["rust_image_inspect_sha256"] = pinned["raw_inspect_sha256"]
    A._validate_cargo_v1(cargo, config, "11" * 32, "/sealed/rust-snapshot", build_seccomp, pinned["image_id"])
    cargo["schema_version"] = "hegel-phase3a-q05b-cargo-build-binary/1"
    with pytest.raises(A.Q05BActualArtifactError) as error:
        A._validate_cargo_v1(cargo, config, "11" * 32, "/sealed/rust-snapshot", build_seccomp, pinned["image_id"])
    assert error.value.code == "REJECT_Q05B_ARTIFACT_CARGO"


def test_cargo_detach_coordinated_rehash_cannot_escape_sealed_binary_crosscheck() -> None:
    config = json.loads((ROOT / "config/phase3_q05b_dual_isolation_v1.json").read_text())
    build_seccomp = _BUILD_SECCOMP_PAYLOAD
    actor = {"source_evidence": {"source_identity_sha256": "11" * 32}, "snapshot_identity": {"root_path": "/sealed/rust-snapshot"}}
    pinned = _image_evidence(config["images"]["rust_build"], config["runtime_command_inspect_policy"]["environment_rows"][1][2])
    cargo = _cargo(config, actor, build_seccomp)
    cargo["rust_image_inspect_hex"] = pinned["raw_inspect_hex"]
    cargo["rust_image_inspect_sha256"] = pinned["raw_inspect_sha256"]
    detach = cargo["binary_detach_evidence"]
    for name in (
        "detached_parent_before",
        "detached_parent_after",
        "detached_fd",
        "detached_path_identity",
    ):
        detach[name]["uid"] += 1
    body = dict(detach)
    body.pop("manifest_sha256")
    detach["manifest_sha256"] = sha256(_j(body)).hexdigest()
    with pytest.raises(A.Q05BActualArtifactError) as rejected:
        A._validate_cargo_v1(cargo, config, "11" * 32, "/sealed/rust-snapshot", build_seccomp, pinned["image_id"])
    assert rejected.value.code == "REJECT_Q05B_ARTIFACT_CARGO"


def test_cargo_detach_rehashed_source_path_swap_and_uid_type_alias_fail_closed() -> None:
    config = json.loads((ROOT / "config/phase3_q05b_dual_isolation_v1.json").read_text())
    build_seccomp = _BUILD_SECCOMP_PAYLOAD
    actor = {"source_evidence": {"source_identity_sha256": "11" * 32}, "snapshot_identity": {"root_path": "/sealed/rust-snapshot"}}
    pinned = _image_evidence(config["images"]["rust_build"], config["runtime_command_inspect_policy"]["environment_rows"][1][2])
    cargo = _cargo(config, actor, build_seccomp)
    cargo["rust_image_inspect_hex"] = pinned["raw_inspect_hex"]
    cargo["rust_image_inspect_sha256"] = pinned["raw_inspect_sha256"]

    swapped = deepcopy(cargo)
    detach = swapped["binary_detach_evidence"]
    detach["source_path"] = "/sealed/target-output/release/swapped"
    body = dict(detach)
    body.pop("manifest_sha256")
    detach["manifest_sha256"] = sha256(_j(body)).hexdigest()
    with pytest.raises(A.Q05BActualArtifactError) as path_rejected:
        A._validate_cargo_v1(swapped, config, "11" * 32, "/sealed/rust-snapshot", build_seccomp, pinned["image_id"])
    assert path_rejected.value.code == "REJECT_Q05B_ARTIFACT_CARGO"

    type_alias = deepcopy(cargo)
    type_alias["binary_file_identity"]["uid"] = True
    with pytest.raises(A.Q05BActualArtifactError) as uid_rejected:
        A._validate_cargo_v1(type_alias, config, "11" * 32, "/sealed/rust-snapshot", build_seccomp, pinned["image_id"])
    assert uid_rejected.value.code == "REJECT_Q05B_ARTIFACT_CARGO"


def test_cargo_admission_join_rejects_rehashed_detach_source_identity() -> None:
    config, build_seccomp, pinned, cargo = _direct_cargo_join_inputs_v1()
    admission = _minimal_join_admission_v1(cargo)
    A._cross_cargo_actual_admission_v1(cargo, admission)
    tampered = deepcopy(cargo)
    for name in (
        "source_fd_before",
        "source_fd_after",
        "source_path_before",
        "source_path_after",
    ):
        tampered["binary_detach_evidence"][name]["inode"] += 100
    _rehash_detach_v1(tampered)
    _validate_direct_cargo_v1(tampered, config, build_seccomp, pinned)
    with pytest.raises(A.Q05BActualArtifactError) as rejected:
        A._cross_cargo_actual_admission_v1(tampered, admission)
    assert rejected.value.code == "REJECT_Q05B_ARTIFACT_CARGO"


def test_cargo_admission_join_rejects_rehashed_detached_principal() -> None:
    config, build_seccomp, pinned, cargo = _direct_cargo_join_inputs_v1()
    admission = _minimal_join_admission_v1(cargo)
    tampered = deepcopy(cargo)
    detach = tampered["binary_detach_evidence"]
    for name in ("detached_parent_before", "detached_parent_after"):
        detach[name]["uid"] += 1
    for name in ("detached_fd", "detached_path_identity"):
        detach[name]["inode"] += 100
        detach[name]["uid"] += 1
    tampered["binary_file_identity"]["inode"] += 100
    tampered["binary_file_identity"]["uid"] += 1
    _rehash_detach_v1(tampered)
    _validate_direct_cargo_v1(tampered, config, build_seccomp, pinned)
    with pytest.raises(A.Q05BActualArtifactError) as rejected:
        A._cross_cargo_actual_admission_v1(tampered, admission)
    assert rejected.value.code == "REJECT_Q05B_ARTIFACT_CARGO"


def test_cargo_admission_join_rejects_rehashed_rust_transcript() -> None:
    config, build_seccomp, pinned, cargo = _direct_cargo_join_inputs_v1()
    admission = _minimal_join_admission_v1(cargo)
    tampered = deepcopy(cargo)
    transcript = tampered["rust_test"]
    stdout = b"cargo synthetic success with alternate transcript\n"
    transcript["stdout_hex"] = stdout.hex()
    transcript["stdout_sha256"] = sha256(stdout).hexdigest()
    transcript["stdout_length"] = len(stdout)
    body = dict(transcript)
    body.pop("evidence_sha256")
    transcript["evidence_sha256"] = sha256(_j(body)).hexdigest()
    _validate_direct_cargo_v1(tampered, config, build_seccomp, pinned)
    with pytest.raises(A.Q05BActualArtifactError) as rejected:
        A._cross_cargo_actual_admission_v1(tampered, admission)
    assert rejected.value.code == "REJECT_Q05B_ARTIFACT_CARGO"


def test_cargo_admission_join_rejects_fully_coordinated_admission_b() -> None:
    config, build_seccomp, pinned, cargo_a = _direct_cargo_join_inputs_v1()
    cargo_b = deepcopy(cargo_a)
    detach = cargo_b["binary_detach_evidence"]
    for name in ("detached_parent_before", "detached_parent_after"):
        detach[name]["uid"] += 1
    for name in ("detached_fd", "detached_path_identity"):
        detach[name]["inode"] += 200
        detach[name]["uid"] += 1
    cargo_b["binary_file_identity"]["inode"] += 200
    cargo_b["binary_file_identity"]["uid"] += 1
    _rehash_detach_v1(cargo_b)
    _validate_direct_cargo_v1(cargo_b, config, build_seccomp, pinned)
    admission_b = _minimal_join_admission_v1(cargo_b)
    A._cross_cargo_actual_admission_v1(cargo_b, admission_b)
    with pytest.raises(A.Q05BActualArtifactError) as rejected:
        A._cross_cargo_actual_admission_v1(cargo_a, admission_b)
    assert rejected.value.code == "REJECT_Q05B_ARTIFACT_CARGO"


def test_direct_stage12_top_join_accepts_production_shape() -> None:
    inputs = _direct_stage12_join_inputs_v1()
    stage_1 = inputs["admission"]["prior_stage_evidence_rows"][0]["evidence"]
    assert set(stage_1) == A._ACTUAL_STAGE_1_EVIDENCE_KEYS
    assert len(stage_1) == 12
    _call_direct_stage12_join_v1(inputs)


def test_direct_docker_ownership_join_accepts_five_unique_slots() -> None:
    inputs = _direct_docker_ownership_inputs_v1()
    _call_direct_docker_ownership_v1(inputs)
    names = [
        row["container_name"]
        for row in inputs["authority"]["ordered_slot_rows"]
    ]
    assert len(names) == len(set(names)) == 5
    other_names = {
        row["container_name"]
        for row in _docker_authority("ab" * 20, b"B" * 32)[
            "ordered_slot_rows"
        ]
    }
    assert set(names).isdisjoint(other_names)


def test_dynamic_actor_commands_and_inspects_accept_authority_names() -> None:
    inputs = _direct_stage12_join_inputs_v1()
    inputs["config"]["held_actor_protocol"]["wrapper_script_sha256"] = sha256(
        b"synthetic-held-wrapper"
    ).hexdigest()
    for role, actor in enumerate(inputs["actors"], 1):
        command_sha256, mount_sha256 = A._command_mount_registry_v1(
            actor["command"], role, actor["actor_id"], inputs["config"]
        )
        assert command_sha256 == sha256(_j(actor["command"])).hexdigest()
        assert len(mount_sha256) == 64
        cid = chr(96 + role) * 64
        payload = _inspect(actor["command"], role, inputs["config"], cid, True)
        sample = {
            "container_id": cid,
            "cpuset_cpus": {1: "0-11", 2: "12-23", 3: "0-11"}[role],
            "inspect_payload_hex": payload.hex(),
            "inspect_after_payload_hex": payload.hex(),
        }
        pinned = inputs["pinned_images"][
            "rust" if role == 2 else "python"
        ]
        A._validate_live_inspect_policy_v1(
            sample,
            actor["command"],
            role,
            inputs["config"],
            pinned["image_id"],
            _RUNTIME_SECCOMP_PAYLOAD,
            actor["control_evidence"]["seccomp_evidence"],
        )


@pytest.mark.parametrize("mutation", ("nonce", "source_commit"))
def test_docker_ownership_coordinated_authority_rehash_still_joins_decision(
    mutation: str,
) -> None:
    inputs = _direct_docker_ownership_inputs_v1()
    if mutation == "nonce":
        inputs["decision"]["attempt_nonce_hex"] = (b"B" * 32).hex()
        inputs["authority"] = _docker_authority("ab" * 20, b"B" * 32)
    else:
        inputs["decision"]["source_commit"] = "cd" * 20
        inputs["authority"] = _docker_authority("cd" * 20)
    with pytest.raises(A.Q05BActualArtifactError):
        _call_direct_docker_ownership_v1(inputs)


@pytest.mark.parametrize(
    "mutation", ("namespace", "name", "slot", "label")
)
def test_docker_ownership_coordinated_manifest_rehash_rejects_slot_tamper(
    mutation: str,
) -> None:
    inputs = _direct_docker_ownership_inputs_v1()
    authority = inputs["authority"]
    if mutation == "namespace":
        authority["execution_namespace"] = "00" * 32
        authority["ordered_slot_rows"] = AD._docker_slot_rows_from_namespace_v1(
            authority["source_commit"], authority["execution_namespace"]
        )
    else:
        row = authority["ordered_slot_rows"][0]
        if mutation == "name":
            row["container_name"] = row["container_name"][:-1] + "x"
        elif mutation == "slot":
            row["slot"] = "RUST_RELEASE"
            row["labels"][1][1] = "RUST_RELEASE"
            next(
                label
                for label in row["expected_container_labels"]
                if label[0] == AD.DOCKER_RESERVED_LABEL_KEYS[1]
            )[1] = "RUST_RELEASE"
        else:
            row["labels"][1][1] = "RUST_RELEASE"
            next(
                label
                for label in row["expected_container_labels"]
                if label[0] == AD.DOCKER_RESERVED_LABEL_KEYS[1]
            )[1] = "RUST_RELEASE"
    authority["initial_name_absence_rows"] = [
        AD._build_docker_initial_name_absence_row_from_spec_v1(
            spec,
            _absence(spec["container_name"]),
            _absence(spec["container_name"]),
        )
        for spec in authority["ordered_slot_rows"]
    ]
    body = dict(authority)
    body.pop("manifest_sha256")
    authority["manifest_sha256"] = sha256(
        AD.DOCKER_EXECUTION_AUTHORITY_ROOT_DOMAIN + _j(body)
    ).hexdigest()
    with pytest.raises(A.Q05BActualArtifactError):
        _call_direct_docker_ownership_v1(inputs)


def test_docker_ownership_rejects_cargo_test_release_slot_swap() -> None:
    inputs = _direct_docker_ownership_inputs_v1()
    cargo = inputs["cargo"]
    cargo["rust_test"], cargo["rust_release_build"] = (
        cargo["rust_release_build"],
        cargo["rust_test"],
    )
    with pytest.raises(A.Q05BActualArtifactError):
        _call_direct_docker_ownership_v1(inputs)


def test_docker_ownership_rejects_coordinated_foreign_actor_inspect() -> None:
    inputs = _direct_docker_ownership_inputs_v1()
    actor = inputs["actors"][0]
    control = actor["control_evidence"]
    foreign = json.loads(
        bytes.fromhex(control["held_final_resource"]["inspect_payload_hex"])
    )
    foreign[0]["Id"] = "f" * 64
    foreign[0]["Name"] = "/foreign-container"
    payload = _j(foreign)
    sample = control["held_final_resource"]
    sample["inspect_payload_hex"] = payload.hex()
    sample["inspect_after_payload_hex"] = payload.hex()
    control["live_ownership_inspect_evidence"] = _owned_inspect(
        payload,
        actor["command"],
        "f" * 64,
        inputs["authority"],
        inputs["slots"]["PYTHON_ENDPOINT"],
    )
    with pytest.raises(A.Q05BActualArtifactError):
        _call_direct_docker_ownership_v1(inputs)


@pytest.mark.parametrize("target_kind", ("name", "aba_id", "force_flag"))
def test_docker_ownership_rejects_name_aba_or_force_success_cleanup(
    target_kind: str,
) -> None:
    inputs = _direct_docker_ownership_inputs_v1()
    success = inputs["cargo"]["rust_test"]
    if target_kind == "force_flag":
        target = success["cidfile_evidence"]["container_id"]
        success["explicit_remove_command"].insert(3, "-f")
    else:
        target = (
            success["docker_execution_slot_row"]["container_name"]
            if target_kind == "name"
            else "f" * 64
        )
        success["explicit_remove_command"][-1] = target
        success["docker_absence_evidence"] = _absence(target)
    with pytest.raises(A.Q05BActualArtifactError):
        _call_direct_docker_ownership_v1(inputs)


def test_docker_ownership_rejects_extra_pinned_base_label() -> None:
    inputs = _direct_docker_ownership_inputs_v1()
    evidence = inputs["pinned"]["python"]
    document = json.loads(bytes.fromhex(evidence["raw_inspect_hex"]))
    document[0]["Config"]["Labels"] = {"foreign.base": "unexpected"}
    payload = _j(document)
    evidence["raw_inspect_hex"] = payload.hex()
    evidence["raw_inspect_sha256"] = sha256(payload).hexdigest()
    body = dict(evidence)
    body.pop("evidence_sha256")
    evidence["evidence_sha256"] = sha256(_j(body)).hexdigest()
    with pytest.raises(A.Q05BActualArtifactError):
        _call_direct_docker_ownership_v1(inputs)


@pytest.mark.parametrize("mutation", ("drop_rust_base", "extra_label"))
def test_docker_ownership_rejects_nonexact_container_config_labels(
    mutation: str,
) -> None:
    inputs = _direct_docker_ownership_inputs_v1()
    actor = inputs["actors"][1]
    control = actor["control_evidence"]
    document = json.loads(
        bytes.fromhex(control["held_final_resource"]["inspect_payload_hex"])
    )
    labels = document[0]["Config"]["Labels"]
    if mutation == "drop_rust_base":
        labels.pop("org.opencontainers.image.source")
    else:
        labels["foreign.extra"] = "unexpected"
    payload = _j(document)
    sample = control["held_final_resource"]
    sample["inspect_payload_hex"] = payload.hex()
    sample["inspect_after_payload_hex"] = payload.hex()
    control["live_ownership_inspect_evidence"] = _owned_inspect(
        payload,
        actor["command"],
        control["container_id"],
        inputs["authority"],
        inputs["slots"]["RUST_ENDPOINT"],
    )
    with pytest.raises(A.Q05BActualArtifactError):
        _call_direct_docker_ownership_v1(inputs)


@pytest.mark.parametrize("stage_id", (1, 2))
def test_prior_stage12_registry_rejects_truncated_synthetic(stage_id: int) -> None:
    inputs = _direct_stage12_join_inputs_v1()
    stages = deepcopy(inputs["admission"]["prior_stage_evidence_rows"])
    stages[2] = _stage_row(3, inputs["commit"], {})
    stages[stage_id - 1] = _stage_row(
        stage_id,
        inputs["commit"],
        {"synthetic": "opaque"},
    )
    with pytest.raises(A.Q05BActualArtifactError) as rejected:
        A._actual_prior_stage_rows_v1(stages, inputs["commit"])
    assert rejected.value.code == "REJECT_Q05B_ARTIFACT_ADMISSION"


@pytest.mark.parametrize(
    "mismatch",
    (
        "stage1_source",
        "stage1_config",
        "stage1_source_closure",
        "stage1_image",
        "stage1_planned",
        "stage1_host_template_identity",
        "stage1_layout",
        "stage1_cargo_cache_shape",
        "stage1_cargo_cache_no_mount",
        "stage1_q1_authority",
        "stage2_snapshot",
        "stage2_lock",
        "stage2_cargo",
        "stage2_seccomp",
        "stage2_build_seccomp_consumer",
    ),
)
def test_stage12_join_rejects_join_local_admission_top_mismatch(
    mismatch: str,
) -> None:
    inputs = _direct_stage12_join_inputs_v1()
    admission = inputs["admission"]
    bundle = inputs["bundle"]
    stage_1 = deepcopy(admission["prior_stage_evidence_rows"][0]["evidence"])
    stage_2 = deepcopy(admission["prior_stage_evidence_rows"][1]["evidence"])
    ordered = bundle["ordered_precondition_rows"]

    if mismatch == "stage1_source":
        actor_id = "PYTHON_ENDPOINT"
        source_evidence = stage_1["source_evidence"][actor_id]
        source_evidence["source_identity_sha256"] = "a1" * 32
        fresh = ordered[5]["preimage"]["actor_rows"][0]
        fresh["source_identity"]["source_identity_sha256"] = "a1" * 32
        fresh["source_identity"]["stage_1_source_evidence_sha256"] = sha256(
            _j(source_evidence)
        ).hexdigest()
        fresh["source_identity_root"] = AD.fresh_runtime_evidence_object_root_v1(
            "ACTOR_SOURCE", actor_id, fresh["source_identity"]
        )
    elif mismatch == "stage1_config":
        config_payload = bytes.fromhex(stage_1["config_hex"]) + b"\n"
        stage_1["config_hex"] = config_payload.hex()
        stage_1["config_sha256"] = sha256(config_payload).hexdigest()
    elif mismatch == "stage1_source_closure":
        closure = stage_1["source_object_closure"]
        closure["project_tree_object_id"] = "ef" * 20
        closure_body = dict(closure)
        closure_body.pop("closure_sha256")
        closure["closure_sha256"] = sha256(_j(closure_body)).hexdigest()
    elif mismatch == "stage1_image":
        image = stage_1["image_evidence"]["python"]
        raw = json.loads(bytes.fromhex(image["raw_inspect_hex"]).decode("ascii"))
        raw[0]["Architecture"] = "arm64"
        raw_payload = _j(raw)
        image["architecture"] = "arm64"
        image["raw_inspect_hex"] = raw_payload.hex()
        image["raw_inspect_sha256"] = sha256(raw_payload).hexdigest()
        image_body = dict(image)
        image_body.pop("evidence_sha256")
        image["evidence_sha256"] = sha256(_j(image_body)).hexdigest()
        row = ordered[4]["preimage"]["image_rows"][0]
        row["evidence"] = deepcopy(image)
        row["evidence_root"] = AD.fresh_runtime_evidence_object_root_v1(
            "PINNED_IMAGE", "python", row["evidence"]
        )
    elif mismatch == "stage1_planned":
        command = stage_1["planned_commands"]["python"]
        command[command.index("hegel-q05b-held-actor") + 1] = "/bin/false"
        ordered[8]["preimage"]["planned_command_registry_sha256"] = sha256(
            _j(stage_1["planned_commands"])
        ).hexdigest()
    elif mismatch == "stage1_host_template_identity":
        command = stage_1["planned_commands"]["host_template"]
        flag = "--host-source-identity-root-hex"
        command[command.index(flag) + 1] = "a5" * 32
        ordered[8]["preimage"]["planned_command_registry_sha256"] = sha256(
            _j(stage_1["planned_commands"])
        ).hexdigest()
    elif mismatch == "stage1_layout":
        stage_1["layout"]["host_output"] = "/sealed/alternate-host-output-unused"
        bundle["work_root_identity"]["layout_sha256"] = sha256(
            _j(stage_1["layout"])
        ).hexdigest()
    elif mismatch == "stage1_cargo_cache_shape":
        stage_1["cargo_cache_root_identity"][2] = 0
    elif mismatch == "stage1_cargo_cache_no_mount":
        stage_1["cargo_cache_source"] = stage_1["layout"]["python_snapshot"]
    elif mismatch == "stage1_q1_authority":
        stage_1["q1_authority"]["state"] = "M3"
    elif mismatch == "stage2_snapshot":
        snapshot = stage_2["snapshot_evidence"]["PYTHON_ENDPOINT"]
        snapshot["root_inode"] += 100
        snapshot_body = dict(snapshot)
        snapshot_body.pop("manifest_sha256")
        snapshot["manifest_sha256"] = sha256(_j(snapshot_body)).hexdigest()
        fresh = ordered[5]["preimage"]["actor_rows"][0]
        fresh["snapshot_evidence"] = deepcopy(snapshot)
        fresh["snapshot_evidence_root"] = (
            AD.fresh_runtime_evidence_object_root_v1(
                "ACTOR_SNAPSHOT", "PYTHON_ENDPOINT", snapshot
            )
        )
    elif mismatch == "stage2_lock":
        lock = bytes.fromhex(stage_2["cargo_lock_hex"]) + b"\n"
        stage_2["cargo_lock_hex"] = lock.hex()
        stage_2["cargo_lock_sha256"] = sha256(lock).hexdigest()
        ordered[6]["preimage"]["cargo_lock_sha256"] = sha256(lock).hexdigest()
    elif mismatch == "stage2_cargo":
        cargo_evidence = stage_2["cargo_evidence"]
        tree = cargo_evidence["sealed_tree_identity"]
        tree["root_nlink"] += 1
        tree_body = dict(tree)
        tree_body.pop("manifest_sha256")
        tree["manifest_sha256"] = sha256(_j(tree_body)).hexdigest()
        cargo_evidence["root_nlink"] = tree["root_nlink"]
        material = ordered[6]["preimage"]["cargo_material_identity"]
        material["root_nlink"] = tree["root_nlink"]
        material["sealed_tree_manifest_sha256"] = tree["manifest_sha256"]
        material["stage_2_cargo_evidence_sha256"] = sha256(
            _j(cargo_evidence)
        ).hexdigest()
        ordered[6]["preimage"]["cargo_tree_evidence"] = deepcopy(tree)
    elif mismatch == "stage2_seccomp":
        snapshot = stage_2["snapshot_evidence"]["TRUSTED_HOST_REPLAY"]
        relative = inputs["config"]["seccomp"]["runtime_profile"]
        file_row = next(row for row in snapshot["file_rows"] if row[0] == relative)
        file_row[2] += 100
        snapshot_body = dict(snapshot)
        snapshot_body.pop("manifest_sha256")
        snapshot["manifest_sha256"] = sha256(_j(snapshot_body)).hexdigest()
        policy = stage_2["seccomp_evidence"]["runtime"]
        policy["file_inode"] = file_row[2]
        policy_body = dict(policy)
        policy_body.pop("manifest_sha256")
        policy["manifest_sha256"] = sha256(_j(policy_body)).hexdigest()
        fresh_actor = ordered[5]["preimage"]["actor_rows"][2]
        fresh_actor["snapshot_evidence"] = deepcopy(snapshot)
        fresh_actor["snapshot_evidence_root"] = (
            AD.fresh_runtime_evidence_object_root_v1(
                "ACTOR_SNAPSHOT", "TRUSTED_HOST_REPLAY", snapshot
            )
        )
        seccomp_row = ordered[7]["preimage"]["seccomp_rows"][0]
        seccomp_row["evidence"] = deepcopy(policy)
        seccomp_row["evidence_root"] = (
            AD.fresh_runtime_evidence_object_root_v1(
                "SECCOMP_POLICY", "runtime", policy
            )
        )
    else:
        inputs["cargo"]["rust_test"]["seccomp_evidence"] = deepcopy(
            stage_2["seccomp_evidence"]["runtime"]
        )

    admission["prior_stage_evidence_rows"][0] = _stage_row(
        1, inputs["commit"], stage_1
    )
    admission["prior_stage_evidence_rows"][1] = _stage_row(
        2, inputs["commit"], stage_2
    )
    _reencode_direct_stage12_admission_v1(inputs)
    with pytest.raises(A.Q05BActualArtifactError):
        _call_direct_stage12_join_v1(inputs)


def test_docker_cid_payload_is_exact_64_lowerhex_without_lf() -> None:
    container_id = "ab" * 32
    evidence = _cid(container_id, "/sealed/actor.cid")
    assert A._validate_docker_cid_payload_v1(
        evidence, container_id, "test cidfile"
    ) == container_id.encode("ascii")

    newline = deepcopy(evidence)
    payload = (container_id + "\n").encode("ascii")
    newline["payload_hex"] = payload.hex()
    newline["payload_sha256"] = sha256(payload).hexdigest()
    newline["file_size"] = len(payload)
    body = dict(newline)
    body.pop("manifest_sha256")
    newline["manifest_sha256"] = sha256(_j(body)).hexdigest()
    with pytest.raises(A.Q05BActualArtifactError) as rejected:
        A._validate_docker_cid_payload_v1(
            newline, container_id, "test cidfile"
        )
    assert rejected.value.code == "REJECT_Q05B_ARTIFACT_ISOLATION"
