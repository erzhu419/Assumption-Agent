#!/usr/bin/env python3
"""One-shot outer campaign for the target-blind full-node6 diagnostic."""

from __future__ import annotations

import argparse
import fcntl
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, NoReturn, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config/phase3_q1_full_node6_campaign_v1.json"
CORE_PATH = (
    PROJECT_ROOT
    / "src/hegel_machine/phase3_q1_full_node6_campaign_v1.py"
)
TECHNICAL_CONFIG_PATH = (
    PROJECT_ROOT
    / "config/phase3_q1_full_node6_capacity_preflight_v1.json"
)
SUPERVISOR_PATH = (
    PROJECT_ROOT / "tools/phase3_q1_full_node6_dual_supervisor_v1.py"
)
FROZEN_PREFLIGHT_CONFIG_PATH = (
    PROJECT_ROOT / "config/phase3_q1_capacity_preflight_v1.json"
)
PENDING_EXIT_CODE = 75
MAX_SMALL_JSON = 16 * 1024 * 1024
MINIMUM_AVAILABLE_MEMORY_BYTES = 30 * 1024 * 1024 * 1024
MINIMUM_AVAILABLE_STORAGE_BYTES = 164 * 1024 * 1024 * 1024
PYTHON_IMAGE = (
    "python@sha256:"
    "e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
)
RUST_IMAGE = (
    "rust@sha256:"
    "38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)
PINNED_PYTHON_RUNTIME_VERSION = "3.11.15"
PYTHON_SOURCE_RECIPE_MODULE_COUNT = 33


def _load_core() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_hegel_q1_full_node6_campaign_core_v1",
        CORE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("campaign core import specification is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_core = _load_core()


def _execution_source_registry_v1() -> dict[str, tuple[str, ...] | int]:
    name = "_hegel_q1_full_node6_supervisor_registry_v1"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SUPERVISOR_PATH)
    if spec is None or spec.loader is None:
        _terminal(
            "EXECUTION_SOURCE_CLOSURE_DIFFERS",
            "supervisor source registry import specification is unavailable",
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        python_paths = tuple(module.PYTHON_ENDPOINT_SOURCE_PATHS)
        rust_paths = tuple(module.RUST_ENDPOINT_SOURCE_PATHS)
        host_paths = tuple(module.HOST_REPLAY_SOURCE_PATHS)
        union_count = int(module.EXECUTION_CLOSURE_ROW_COUNT)
    except Exception as error:
        _terminal("EXECUTION_SOURCE_CLOSURE_DIFFERS", str(error))
    return {
        "python": python_paths,
        "rust": rust_paths,
        "host": host_paths,
        "union_count": union_count,
    }


class GatePending(RuntimeError):
    def __init__(
        self,
        phase: str,
        condition: str,
        detail: str,
        *,
        capsule: Mapping[str, object] | None = None,
        capsule_digest_hint: str | None = None,
    ) -> None:
        super().__init__(f"{phase}/{condition}: {detail}")
        self.phase = phase
        self.condition = condition
        self.detail = detail
        self.capsule = dict(capsule) if capsule is not None else None
        self.capsule_digest_hint = capsule_digest_hint


class CampaignTerminal(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _terminal(code: str, detail: str) -> NoReturn:
    raise CampaignTerminal(code, detail)


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            _terminal("JSON_DUPLICATE_KEY", f"duplicate key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> NoReturn:
    _terminal("JSON_NONFINITE", f"non-finite token {value!r}")


def _strict_json(raw: bytes, label: str, maximum: int = MAX_SMALL_JSON) -> object:
    if type(raw) is not bytes or len(raw) > maximum:
        _terminal("JSON_SIZE", f"{label} exceeds its exact read bound")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _terminal("JSON_WIRE", f"{label}: {error}")


def _read_regular(path: Path, label: str, maximum: int = MAX_SMALL_JSON) -> bytes:
    try:
        status = path.stat(follow_symlinks=False)
        if not path.is_file() or path.is_symlink() or status.st_size > maximum:
            raise OSError("not a bounded ordinary file")
        return path.read_bytes()
    except OSError as error:
        raise GatePending("PATH_AND_FORMAT", "READONLY_FILE_UNAVAILABLE", f"{label}: {error}")


def load_contract_v1() -> dict[str, Any]:
    raw = _read_regular(CONFIG_PATH, "campaign contract")
    value = _strict_json(raw, "campaign contract")
    try:
        return _core.validate_contract_v1(value)
    except Exception as error:
        _terminal("CONTRACT_REPLAY_FAILED", str(error))


def _git(*arguments: str, timeout: float = 60.0) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise GatePending(
            "SOURCE_LINEAGE_AND_OBJECT_CLOSURE",
            "READONLY_COMMAND_TIMEOUT",
            f"git {' '.join(arguments)} timed out after {error.timeout}s",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise GatePending(
            "SOURCE_LINEAGE_AND_OBJECT_CLOSURE",
            "READONLY_COMMAND_FAILED",
            f"git {' '.join(arguments)}: {error}",
        )
    try:
        return completed.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        _terminal("SOURCE_WIRE", f"git output is not UTF-8: {error}")


def _source_gate(contract: Mapping[str, Any]) -> dict[str, object]:
    lineage = contract["source_lineage"]
    if Path(str(lineage["project_root"])) != PROJECT_ROOT:
        _terminal("SOURCE_ROOT_DIFFERS", "project root differs from contract")
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise GatePending(
            "SOURCE_LINEAGE_AND_OBJECT_CLOSURE",
            "SOURCE_NOT_CLEAN_COMMITTED",
            "commit the exact additive successor before capsule generation",
        )
    head = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    prefix = _git("rev-parse", "--show-prefix")
    if prefix != "Hegel Machine/":
        _terminal("SOURCE_PREFIX_DIFFERS", f"observed Git prefix={prefix!r}")
    ancestor = str(lineage["required_ancestor_commit"])
    parent = _git("rev-parse", "HEAD^")
    if parent != ancestor:
        _terminal(
            "SOURCE_LINEAGE_DIFFERS",
            f"successor HEAD must be the direct child of {ancestor}; observed parent={parent}",
        )
    observed_repo_paths = tuple(
        row
        for row in _git(
            "diff", "--name-only", "--diff-filter=A", ancestor, "HEAD"
        ).splitlines()
        if row
    )
    if any(not row.startswith(prefix) for row in observed_repo_paths):
        _terminal(
            "SOURCE_ADDITIVE_PATHS_DIFFER",
            f"an additive path escaped {prefix!r}: {observed_repo_paths!r}",
        )
    observed_paths = tuple(row.removeprefix(prefix) for row in observed_repo_paths)
    expected_paths = tuple(lineage["additive_paths_in_exact_order"])
    if observed_paths != expected_paths:
        _terminal(
            "SOURCE_ADDITIVE_PATHS_DIFFER",
            f"expected={expected_paths!r}; observed={observed_paths!r}",
        )
    non_additive = _git("diff", "--name-only", "--diff-filter=DMRTUXB", ancestor, "HEAD")
    if non_additive:
        _terminal("SOURCE_NOT_ADDITIVE", f"non-additive paths: {non_additive}")
    try:
        subprocess.run(
            ["git", "fsck", "--connectivity-only", "--no-dangling"],
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise GatePending(
            "SOURCE_LINEAGE_AND_OBJECT_CLOSURE",
            "READONLY_COMMAND_TIMEOUT",
            "git fsck connectivity check timed out",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise GatePending(
            "SOURCE_LINEAGE_AND_OBJECT_CLOSURE",
            "TARGET_LOCAL_OBJECT_CLOSURE_FSCK",
            str(error),
        )
    rows: list[list[object]] = []
    for relative in expected_paths:
        try:
            listing = subprocess.run(
                ["git", "ls-tree", "-z", "-l", "HEAD", "--", relative],
                cwd=PROJECT_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            ).stdout
        except subprocess.TimeoutExpired:
            raise GatePending(
                "SOURCE_LINEAGE_AND_OBJECT_CLOSURE",
                "READONLY_COMMAND_TIMEOUT",
                f"git ls-tree timed out for {relative}",
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise GatePending(
                "SOURCE_LINEAGE_AND_OBJECT_CLOSURE",
                "TARGET_LOCAL_OBJECT_CLOSURE_FSCK",
                f"{relative}: {error}",
            )
        if not listing.endswith(b"\x00") or listing.count(b"\x00") != 1:
            _terminal("SOURCE_TREE_ROW_DIFFERS", relative)
        header, path_raw = listing[:-1].split(b"\t", 1)
        fields = header.decode("ascii").split()
        if len(fields) != 4 or path_raw.decode("utf-8") != relative:
            _terminal("SOURCE_TREE_ROW_DIFFERS", relative)
        mode, kind, object_id, size_text = fields
        if kind != "blob" or mode not in ("100644", "100755"):
            _terminal("SOURCE_TREE_ROW_DIFFERS", relative)
        rows.append([relative, mode, object_id, int(size_text)])
    closure = _core.domain_digest_v1(
        b"HEGEL/Q1/FULL_NODE6/SOURCE_CLOSURE/V1\x00",
        rows,
    )
    return {
        "source_commit": head,
        "source_tree": tree,
        "source_closure_digest": closure,
        "source_rows": rows,
        "git_subdirectory_prefix": prefix,
    }


def _sha256_file(
    path: Path,
    label: str,
    *,
    maximum: int | None = None,
    phase: str = "PREDECESSOR_TERMINAL_AND_ARTIFACT",
) -> tuple[str, int]:
    try:
        status = path.stat(follow_symlinks=False)
        if not path.is_file() or path.is_symlink():
            raise OSError("not an ordinary file")
        if maximum is not None and status.st_size > maximum:
            raise OSError("file exceeds bound")
        digest = sha256()
        with path.open("rb") as stream:
            while True:
                block = stream.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise GatePending(
            phase,
            "READONLY_FILE_UNAVAILABLE",
            f"{label}: {error}",
        )
    return "sha256:" + digest.hexdigest(), status.st_size


def _predecessor_gate(contract: Mapping[str, Any]) -> dict[str, object]:
    expected = contract["predecessor"]
    ledger_path = Path(str(expected["terminal_ledger_path"]))
    raw = _read_regular(ledger_path, "predecessor terminal ledger")
    raw_digest = "sha256:" + sha256(raw).hexdigest()
    if raw_digest != expected["terminal_ledger_file_sha256"]:
        _terminal("PREDECESSOR_LEDGER_BYTES_DIFFER", raw_digest)
    value = _strict_json(raw, "predecessor terminal ledger")
    if type(value) is not dict:
        _terminal("PREDECESSOR_LEDGER_WIRE", "terminal ledger must be one object")
    if (
        value.get("record_digest") != expected["terminal_ledger_record_digest"]
        or value.get("capsule_digest") != expected["capsule_digest"]
        or value.get("status") != expected["terminal_status"]
    ):
        _terminal("PREDECESSOR_TERMINAL_DIFFERS", "terminal ledger binding differs")
    artifact_path = Path(str(expected["artifact_path"]))
    artifact_digest, artifact_size = _sha256_file(
        artifact_path,
        "predecessor artifact",
        maximum=2 * 1024 * 1024 * 1024,
    )
    if (
        artifact_size != expected["artifact_bytes"]
        or artifact_digest != expected["artifact_sha256"]
    ):
        _terminal(
            "PREDECESSOR_ARTIFACT_DIFFERS",
            f"size={artifact_size}; digest={artifact_digest}",
        )
    return {
        "terminal_ledger_file_sha256": raw_digest,
        "terminal_ledger_record_digest": value["record_digest"],
        "artifact_sha256": artifact_digest,
        "artifact_bytes": artifact_size,
        "terminal_status": value["status"],
    }


def _technical_contract_gate(contract: Mapping[str, Any]) -> dict[str, object]:
    raw = _read_regular(TECHNICAL_CONFIG_PATH, "technical capacity contract")
    value = _strict_json(raw, "technical capacity contract")
    frozen_raw = _read_regular(
        FROZEN_PREFLIGHT_CONFIG_PATH,
        "frozen capacity preregistration contract",
    )
    frozen = _strict_json(frozen_raw, "frozen capacity preregistration contract")
    if type(value) is not dict:
        _terminal("TECHNICAL_CONTRACT_WIRE", "technical contract must be one object")
    if type(frozen) is not dict:
        _terminal("TECHNICAL_CONTRACT_WIRE", "frozen preregistration must be one object")
    target = contract["target"]
    resources = value.get("resource_envelope", {})
    measurement = value.get("measurement_policy", {})
    isolation = value.get("runtime_isolation_profile", {})
    frozen_isolation = frozen.get("runtime_isolation_profile")
    source_closure = value.get("execution_source_closure")
    source_registry = _execution_source_registry_v1()
    expected_source_closure = {
        "python_endpoint_relative_paths_in_exact_order": list(
            source_registry["python"]
        ),
        "python_endpoint_row_count": len(source_registry["python"]),
        "rust_release_build_relative_paths_in_exact_order": list(
            source_registry["rust"]
        ),
        "rust_release_build_row_count": len(source_registry["rust"]),
        "endpoint_execution_union_row_count": source_registry["union_count"],
        "host_replay_relative_paths_in_exact_order": list(source_registry["host"]),
        "host_replay_row_count": len(source_registry["host"]),
        "target_split_role_files_mounted": False,
    }
    inherited_isolation = dict(isolation) if type(isolation) is dict else {}
    endpoint_user = inherited_isolation.pop("endpoint_user", None)
    network = inherited_isolation.pop("network", None)
    pull_policy = inherited_isolation.pop("pull_policy", None)
    if (
        value.get("action_id") != target["action_id"]
        or value.get("authority", {}).get("allowed_success_status")
        != target["allowed_success_status"]
        or value.get("authority", {}).get("q1_state") != "NOT_RUN"
        or measurement.get(
            "maximum_quotient_measurements_per_endpoint_and_signature"
        )
        != 1
        or measurement.get("per_signature_wall_time_seconds") != 172800
        or measurement.get("ordered_signature_count") != 2
        or measurement.get("bounded_projection_wall_time_seconds") != 172800
        or measurement.get("endpoint_pair_process_deadline_seconds") != 518400
        or measurement.get("trusted_host_replay_process_deadline_seconds")
        != 172800
        or value.get("execution_policy", {}).get(
            "outer_supervisor_wall_time_seconds"
        )
        != 693600
        or resources.get(
            "outer_preparation_and_final_publication_reserve_bytes"
        )
        != 4 * 1024 * 1024 * 1024
        or resources.get("python_runtime_version") != "3.11.15"
        or resources.get("python_image") != PYTHON_IMAGE
        or resources.get("rust_image") != RUST_IMAGE
        or endpoint_user != "65534:65534"
        or network != "none"
        or pull_policy != "never"
        or inherited_isolation != frozen_isolation
        or value.get("measurement_contract") != frozen.get("measurement_contract")
        or source_closure != expected_source_closure
    ):
        _terminal("TECHNICAL_CONTRACT_DIFFERS", "technical authority differs")
    return {
        "raw_sha256": "sha256:" + sha256(raw).hexdigest(),
        "projection_profile_root": value["semantic_and_projection_binding"][
            "projection_profile_root"
        ],
        "semantic_binding_root": value["semantic_and_projection_binding"][
            "semantic_binding_root"
        ],
        "action_id": value["action_id"],
        "frozen_measurement_contract_replayed": True,
        "frozen_runtime_isolation_profile_replayed": True,
        "execution_source_closure_replayed": True,
        "python_endpoint_source_row_count": len(source_registry["python"]),
        "rust_release_build_source_row_count": len(source_registry["rust"]),
        "endpoint_execution_union_row_count": source_registry["union_count"],
        "host_replay_source_row_count": len(source_registry["host"]),
    }


def _run_readonly_command(
    command: Sequence[str],
    *,
    phase: str,
    condition: str,
    timeout: float,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise GatePending(phase, "READONLY_COMMAND_TIMEOUT", " ".join(command))
    except (OSError, subprocess.CalledProcessError) as error:
        raise GatePending(phase, condition, f"{' '.join(command)}: {error}")


def _materialize_source_snapshot_v1(
    relative_paths: Sequence[str],
    destination: Path,
) -> None:
    try:
        destination.mkdir(mode=0o755)
        for relative in relative_paths:
            if (
                type(relative) is not str
                or not relative
                or relative.startswith("/")
                or ".." in Path(relative).parts
            ):
                raise OSError(f"invalid execution source path {relative!r}")
            source = PROJECT_ROOT / relative
            status = source.stat(follow_symlinks=False)
            if source.is_symlink() or not source.is_file():
                raise OSError(f"execution source is not an ordinary file: {relative}")
            target = destination / relative
            target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            os.chmod(target, status.st_mode & 0o555 or 0o444)
    except OSError as error:
        raise GatePending(
            "PYTHON_RUST_OFFLINE_DEPENDENCIES",
            "EXACT_EXECUTION_SOURCE_CLOSURE_UNAVAILABLE",
            str(error),
        )


def _remove_dependency_probe_v1(probe_root: Path) -> None:
    try:
        shutil.rmtree(probe_root)
        parent = os.open(probe_root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except OSError as error:
        raise GatePending(
            "PYTHON_RUST_OFFLINE_DEPENDENCIES",
            "TRANSIENT_DEPENDENCY_PROBE_CLEANUP_FAILED",
            str(error),
        )


def _dependency_gate(contract: Mapping[str, Any]) -> dict[str, object]:
    python = _run_readonly_command(
        ["/usr/bin/python3.10", "-I", "-S", "-B", "--version"],
        phase="PYTHON_RUST_OFFLINE_DEPENDENCIES",
        condition="PYTHON310_UNAVAILABLE",
        timeout=15,
    )
    python_image = _run_readonly_command(
        ["docker", "image", "inspect", "--format", "{{.Id}}", PYTHON_IMAGE],
        phase="PYTHON_RUST_OFFLINE_DEPENDENCIES",
        condition="PINNED_PYTHON_IMAGE_UNAVAILABLE",
        timeout=30,
    )
    rust_image = _run_readonly_command(
        ["docker", "image", "inspect", "--format", "{{.Id}}", RUST_IMAGE],
        phase="PYTHON_RUST_OFFLINE_DEPENDENCIES",
        condition="PINNED_RUST_BUILD_IMAGE_UNAVAILABLE",
        timeout=30,
    )
    source_registry = _execution_source_registry_v1()
    python_source_paths = tuple(source_registry["python"])
    rust_source_paths = tuple(source_registry["rust"])
    endpoint = str(contract["target"]["python_endpoint_relative_path"])
    if not python_source_paths or python_source_paths[0] != endpoint:
        _terminal(
            "EXECUTION_SOURCE_CLOSURE_DIFFERS",
            "Python endpoint is not the first exact execution source row",
        )
    cargo_cache = Path(str(contract["fixed_paths"]["cargo_cache_source"]))
    probe_root = Path(str(contract["fixed_paths"]["probe_root"]))
    probe_created = False
    python_snapshot = probe_root / "python-source"
    rust_snapshot = probe_root / "rust-release-source"
    try:
        cargo_cache.stat(follow_symlinks=False)
        if cargo_cache.is_symlink() or not cargo_cache.is_dir():
            raise OSError("Cargo cache is not an ordinary directory")
        if os.path.lexists(probe_root):
            raise OSError("dependency probe root already exists")
        probe_root.mkdir(mode=0o700)
        probe_created = True
        cargo_home = probe_root / "cargo-home"
        build_root = probe_root / "build"
        test_root = probe_root / "test"
        _materialize_source_snapshot_v1(python_source_paths, python_snapshot)
        _materialize_source_snapshot_v1(rust_source_paths, rust_snapshot)
        shutil.copytree(cargo_cache, cargo_home)
        build_root.mkdir(mode=0o700)
        test_root.mkdir(mode=0o700)
    except (OSError, GatePending) as error:
        if probe_created:
            try:
                _remove_dependency_probe_v1(probe_root)
            except GatePending as cleanup_error:
                raise cleanup_error from error
        if isinstance(error, GatePending):
            raise
        raise GatePending(
            "PYTHON_RUST_OFFLINE_DEPENDENCIES",
            "CARGO_OFFLINE_CLOSURE_UNAVAILABLE",
            str(error),
        )

    manifest_parent = Path(
        str(contract["target"]["rust_manifest_relative_path"])
    ).parent.as_posix()
    common = [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--cpuset-cpus",
        "0-11",
        "--memory",
        str(8 * 1024 * 1024 * 1024),
        "--memory-swap",
        str(8 * 1024 * 1024 * 1024),
        "--pids-limit",
        "256",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=1073741824",
        "--mount",
        f"type=bind,src={PROJECT_ROOT},dst=/snapshot,readonly",
        "--mount",
        f"type=bind,src={cargo_home},dst=/cargo",
        "--mount",
        f"type=bind,src={build_root},dst=/build",
        "--mount",
        f"type=bind,src={test_root},dst=/test-target",
        "--env",
        "CARGO_HOME=/cargo",
        "--env",
        "CARGO_NET_OFFLINE=true",
        "--workdir",
        f"/snapshot/{manifest_parent}",
        RUST_IMAGE,
        "/usr/local/cargo/bin/cargo",
    ]
    release_common = [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--cpuset-cpus",
        "12-23",
        "--memory",
        str(15_032_385_536),
        "--memory-swap",
        str(15_032_385_536),
        "--pids-limit",
        "128",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=1073741824",
        "--mount",
        f"type=bind,src={rust_snapshot},dst=/snapshot,readonly",
        "--mount",
        f"type=bind,src={build_root},dst=/build",
        "--mount",
        f"type=bind,src={cargo_home},dst=/cargo",
        "--env",
        "CARGO_HOME=/cargo",
        "--env",
        "CARGO_NET_OFFLINE=true",
        "--workdir",
        f"/snapshot/{manifest_parent}",
        RUST_IMAGE,
        "/usr/local/cargo/bin/cargo",
    ]
    try:
        python_recipe_code = (
            "import runpy,sys;"
            f"m=runpy.run_path('/snapshot/{endpoint}');"
            "rows=m['_loaded_project_modules']();"
            "print(sys.version.split()[0],len(rows))"
        )
        python_recipe = _run_readonly_command(
            [
                "docker",
                "run",
                "--rm",
                "--pull",
                "never",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--security-opt",
                f"seccomp={PROJECT_ROOT / 'config/phase3_internal_actor_seccomp_v1.json'}",
                "--pids-limit",
                "128",
                "--ulimit",
                "nofile=256:256",
                "--ipc",
                "private",
                "--user",
                "65534:65534",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=1g,mode=1777",
                "--mount",
                f"type=bind,src={python_snapshot},dst=/snapshot,readonly",
                "--workdir",
                "/snapshot",
                PYTHON_IMAGE,
                "/usr/local/bin/python3",
                "-I",
                "-S",
                "-B",
                "-c",
                python_recipe_code,
            ],
            phase="PYTHON_RUST_OFFLINE_DEPENDENCIES",
            condition="PYTHON_SOURCE_RECIPE_QUALIFICATION_FAILED",
            timeout=300,
        )
        expected_python_recipe = (
            f"{PINNED_PYTHON_RUNTIME_VERSION} "
            f"{PYTHON_SOURCE_RECIPE_MODULE_COUNT}\n"
        ).encode("ascii")
        if python_recipe.stdout != expected_python_recipe or python_recipe.stderr:
            raise GatePending(
                "PYTHON_RUST_OFFLINE_DEPENDENCIES",
                "PYTHON_SOURCE_RECIPE_QUALIFICATION_FAILED",
                "pinned Python runtime or exact project-module closure differs",
            )
        metadata = _run_readonly_command(
            release_common
            + [
                "metadata",
                "--locked",
                "--offline",
                "--no-deps",
                "--format-version=1",
            ],
            phase="PYTHON_RUST_OFFLINE_DEPENDENCIES",
            condition="CARGO_OFFLINE_CLOSURE_UNAVAILABLE",
            timeout=120,
        )
        _run_readonly_command(
            common
            + [
                "test",
                "--locked",
                "--offline",
                "--all-targets",
                "--target-dir",
                "/test-target",
            ],
            phase="PYTHON_RUST_OFFLINE_DEPENDENCIES",
            condition="RUST_SOURCE_RECIPE_QUALIFICATION_FAILED",
            timeout=1800,
        )
        _run_readonly_command(
            release_common
            + [
                "build",
                "--release",
                "--offline",
                "--locked",
                "--target-dir",
                "/build/target",
            ],
            phase="PYTHON_RUST_OFFLINE_DEPENDENCIES",
            condition="RUST_SOURCE_RECIPE_QUALIFICATION_FAILED",
            timeout=1800,
        )
        binary = (
            build_root
            / "target/release"
            / str(contract["target"]["rust_binary_name"])
        )
        binary_digest, binary_bytes = _sha256_file(
            binary,
            "qualified offline Rust release binary",
            maximum=512 * 1024 * 1024,
            phase="PYTHON_RUST_OFFLINE_DEPENDENCIES",
        )
        receipt = {
            "python_version": (python.stdout + python.stderr).decode("ascii").strip(),
            "python_image_id": python_image.stdout.decode("ascii").strip(),
            "python_endpoint_runtime_version": PINNED_PYTHON_RUNTIME_VERSION,
            "python_endpoint_loaded_project_module_count": (
                PYTHON_SOURCE_RECIPE_MODULE_COUNT
            ),
            "python_endpoint_source_recipe": "PASS",
            "python_endpoint_source_row_count": len(python_source_paths),
            "rust_image_id": rust_image.stdout.decode("ascii").strip(),
            "cargo_metadata_sha256": "sha256:" + sha256(metadata.stdout).hexdigest(),
            "cargo_test_locked_offline_all_targets": "PASS",
            "rust_release_build_source_row_count": len(rust_source_paths),
            "endpoint_execution_union_row_count": source_registry["union_count"],
            "exact_execution_source_closure": "PASS",
            "rust_release_binary_sha256": binary_digest,
            "rust_release_binary_bytes": binary_bytes,
        }
    finally:
        _remove_dependency_probe_v1(probe_root)
    return receipt


def _effective_cpu_set() -> set[int]:
    try:
        return set(os.sched_getaffinity(0))
    except (AttributeError, OSError) as error:
        raise GatePending(
            "CPU_TOPOLOGY_AND_EXACT_AFFINITY",
            "SCHED_AFFINITY_API_UNAVAILABLE",
            str(error),
        )


def _cpu_gate() -> dict[str, object]:
    cpus = _effective_cpu_set()
    required = set(range(24))
    if not required.issubset(cpus):
        raise GatePending(
            "CPU_TOPOLOGY_AND_EXACT_AFFINITY",
            "REQUIRED_EFFECTIVE_CPU_SET_UNAVAILABLE",
            f"required=0-23; observed={','.join(str(item) for item in sorted(cpus))}",
        )
    return {"effective_cpu_set": sorted(cpus), "required_endpoint_cpu_set": list(range(24))}


def _docker_gate() -> dict[str, object]:
    server = _run_readonly_command(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        phase="PINNED_LOCAL_IMAGES_AND_DAEMON",
        condition="LOCAL_DOCKER_DAEMON_UNAVAILABLE",
        timeout=20,
    )
    images = (PYTHON_IMAGE, RUST_IMAGE)
    rows: list[list[str]] = []
    for image in images:
        inspected = _run_readonly_command(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            phase="PINNED_LOCAL_IMAGES_AND_DAEMON",
            condition="PINNED_LOCAL_IMAGE_UNAVAILABLE",
            timeout=30,
        )
        rows.append([image, inspected.stdout.decode("ascii").strip()])
    return {"docker_server_version": server.stdout.decode("ascii").strip(), "images": rows}


def _mem_available_bytes() -> int:
    try:
        rows = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise GatePending("MEMORY_AND_STORAGE_RESERVATION", "MEMINFO_UNAVAILABLE", str(error))
    for row in rows:
        if row.startswith("MemAvailable:"):
            fields = row.split()
            if len(fields) == 3 and fields[2] == "kB" and fields[1].isdigit():
                return int(fields[1]) * 1024
    raise GatePending("MEMORY_AND_STORAGE_RESERVATION", "MEMINFO_UNAVAILABLE", "MemAvailable missing")


def _resource_gate(contract: Mapping[str, Any]) -> dict[str, object]:
    available_memory = _mem_available_bytes()
    storage_root = Path(str(contract["fixed_paths"]["run_root"])).parent
    try:
        storage = os.statvfs(storage_root)
    except OSError as error:
        raise GatePending("MEMORY_AND_STORAGE_RESERVATION", "STORAGE_STAT_UNAVAILABLE", str(error))
    available_storage = storage.f_bavail * storage.f_frsize
    if available_memory < MINIMUM_AVAILABLE_MEMORY_BYTES:
        raise GatePending(
            "MEMORY_AND_STORAGE_RESERVATION",
            "REQUIRED_AVAILABLE_MEMORY_UNAVAILABLE",
            f"required={MINIMUM_AVAILABLE_MEMORY_BYTES}; observed={available_memory}",
        )
    if available_storage < MINIMUM_AVAILABLE_STORAGE_BYTES:
        raise GatePending(
            "MEMORY_AND_STORAGE_RESERVATION",
            "REQUIRED_AVAILABLE_STORAGE_UNAVAILABLE",
            f"required={MINIMUM_AVAILABLE_STORAGE_BYTES}; observed={available_storage}",
        )
    return {
        "available_memory_bytes": available_memory,
        "required_available_memory_bytes": MINIMUM_AVAILABLE_MEMORY_BYTES,
        "available_storage_bytes": available_storage,
        "required_available_storage_bytes": MINIMUM_AVAILABLE_STORAGE_BYTES,
    }


def _state_absence_gate(contract: Mapping[str, Any], *, allow_ledger: bool) -> dict[str, object]:
    fixed = contract["fixed_paths"]
    names = (
        "artifact_path",
        "evidence_root",
        "trajectory_root",
        "run_root",
        "probe_root",
    ) + (() if allow_ledger else ("ledger_root",))
    present = [name for name in names if os.path.lexists(Path(str(fixed[name])))]
    if present:
        raise GatePending(
            "CURRENT_FUTURE_FORMAL_AND_PROBE_ROOTS_ABSENT",
            "PRECONSUMPTION_STATE_ROOT_PRESENT",
            ",".join(present),
        )
    return {"proved_absent": list(names)}


def _probe_gate(contract: Mapping[str, Any]) -> dict[str, object]:
    path = Path(str(contract["fixed_paths"]["probe_root"]))
    try:
        path.mkdir(mode=0o700)
        marker = path / "admission.probe"
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, b"FULL_NODE6_PRECONSUMPTION_PROBE\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        marker.unlink()
        path.rmdir()
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except OSError as error:
        try:
            marker = path / "admission.probe"
            if marker.is_file() and not marker.is_symlink():
                marker.unlink()
            if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
                path.rmdir()
        except OSError:
            pass
        raise GatePending(
            "PATH_AND_FORMAT",
            "PRECONSUMPTION_WRITE_PROBE_FAILED",
            str(error),
        )
    return {"probe_path": str(path), "probe_removed": not os.path.lexists(path)}


def derive_preflight_v1(*, include_live_gates: bool = True) -> dict[str, object]:
    contract = load_contract_v1()
    source = _source_gate(contract)
    predecessor = _predecessor_gate(contract)
    technical = _technical_contract_gate(contract)
    dependency = _dependency_gate(contract)
    static = {
        "authorization_digest": _core.OWNER_AUTHORIZATION_DIGEST,
        "source": source,
        "predecessor": predecessor,
        "technical_contract": technical,
        "dependencies": dependency,
    }
    capsule = _core.build_capsule_v1(
        contract,
        source_commit=source["source_commit"],
        source_tree=source["source_tree"],
        source_closure_digest=source["source_closure_digest"],
        static_gate_receipts=static,
    )
    live: dict[str, object] = {}
    if include_live_gates:
        try:
            live["docker"] = _docker_gate()
            live["cpu"] = _cpu_gate()
            live["resources"] = _resource_gate(contract)
            live["absence"] = _state_absence_gate(contract, allow_ledger=False)
            live["write_probe"] = _probe_gate(contract)
        except GatePending as error:
            error.capsule = dict(capsule)
            error.capsule_digest_hint = str(capsule["capsule_digest"])
            raise
    return {
        "schema_version": "hegel-phase3a-q1-full-node6-preflight-result/1",
        "status": _core.PREFLIGHT_ADMITTED,
        "capsule": capsule,
        "capsule_digest": capsule["capsule_digest"],
        "static_gate_receipts": static,
        "live_gate_receipts": live,
        "campaign_execution_count": 0,
        "authorization_consumption_count": 0,
        "supervisor_popen_count": 0,
        "q1_state": "NOT_RUN",
    }


class _CampaignLock:
    def __init__(
        self,
        path: Path,
        *,
        capsule_digest_hint: str | None = None,
    ) -> None:
        self.path = path
        self.descriptor = -1
        self.capsule_digest_hint = capsule_digest_hint

    def __enter__(self) -> "_CampaignLock":
        try:
            self.descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(self.descriptor, fcntl.LOCK_EX)
        except OSError as error:
            raise GatePending(
                "PATH_AND_FORMAT",
                "CAMPAIGN_LOCK_UNAVAILABLE",
                str(error),
                capsule_digest_hint=self.capsule_digest_hint,
            )
        return self

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        if self.descriptor >= 0:
            try:
                fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self.descriptor)
                self.descriptor = -1


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


def _publish_no_replace(root: Path, name: str, value: object) -> bytes:
    payload = _core.canonical_json_bytes_v1(value)
    final_path = root / name
    root_fd = -1
    descriptor = -1
    temporary_path: Path | None = None
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        descriptor, temporary_text = tempfile.mkstemp(
            prefix=f".{name}.tmp-",
            dir=root,
        )
        temporary_path = Path(temporary_text)
        _write_all(descriptor, payload)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary_path, final_path, follow_symlinks=False)
        os.unlink(temporary_path)
        temporary_path = None
        os.fsync(root_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        if root_fd >= 0:
            os.close(root_fd)
    return payload


def _publish_or_replay_exact(root: Path, name: str, value: object) -> bytes:
    expected = _core.canonical_json_bytes_v1(value)
    try:
        return _publish_no_replace(root, name, value)
    except OSError:
        path = root / name
        try:
            status = path.stat(follow_symlinks=False)
            if path.is_symlink() or not path.is_file() or status.st_size != len(expected):
                raise OSError("existing publication is not the exact ordinary file")
            observed = path.read_bytes()
        except OSError:
            raise
        if observed != expected:
            raise
        return observed


_LEDGER_FIELDS = {
    "schema_version",
    "contract_digest",
    "capsule_digest",
    "authorization_digest",
    "sequence",
    "record_type",
    "status",
    "previous_record_digest_or_null",
    "body",
    "record_digest",
}
_CONSUMED_BODY_FIELDS = {
    "capsule",
    "maximum_supervisor_popen_count",
    "maximum_measurements_per_endpoint_signature",
}
_TERMINAL_BODY_FIELDS = {
    "outcome",
    "outcome_digest",
    "success",
    "retry_or_resume_allowed",
    "supervisor_return_code",
    "external_outcome_path",
    "external_outcome_state",
    "external_outcome_error_or_null",
}


def _read_record(path: Path) -> dict[str, object] | None:
    if not os.path.lexists(path):
        return None
    try:
        raw = _read_regular(path, path.name)
    except GatePending as error:
        _terminal("LEDGER_RECORD_UNREADABLE", f"{path.name}: {error.detail}")
    value = _strict_json(raw, path.name)
    if type(value) is not dict:
        _terminal("LEDGER_WIRE", f"{path.name} must be one object")
    return value


def _validate_ledger_digest_v1(value: Mapping[str, object], label: str) -> None:
    if set(value) != _LEDGER_FIELDS:
        _terminal("LEDGER_WIRE", f"{label} field registry differs")
    unsigned = dict(value)
    observed = unsigned.pop("record_digest")
    expected = _core.domain_digest_v1(_core.DOMAINS["ledger"], unsigned)
    if observed != expected:
        _terminal("LEDGER_DIGEST_DIFFERS", f"{label} record digest differs")


def _validate_consumed_record_v1(
    value: object,
    contract: Mapping[str, Any],
    expected_capsule: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if type(value) is not dict:
        _terminal("LEDGER_WIRE", "consumed ledger must be one object")
    _validate_ledger_digest_v1(value, "consumed")
    body = value.get("body")
    if type(body) is not dict or set(body) != _CONSUMED_BODY_FIELDS:
        _terminal("LEDGER_WIRE", "consumed body field registry differs")
    try:
        capsule = _core.validate_capsule_v1(body["capsule"], contract)
    except Exception as error:
        _terminal("LEDGER_CAPSULE_DIFFERS", str(error))
    if (
        value.get("schema_version") != _core.LEDGER_SCHEMA_VERSION
        or value.get("contract_digest") != contract["contract_digest"]
        or value.get("capsule_digest") != expected_capsule
        or value.get("capsule_digest") != capsule["capsule_digest"]
        or value.get("authorization_digest") != _core.OWNER_AUTHORIZATION_DIGEST
        or value.get("sequence") != 1
        or value.get("record_type") != "CONSUMED"
        or value.get("status") != _core.CONSUMED
        or value.get("previous_record_digest_or_null") is not None
        or body.get("maximum_supervisor_popen_count") != 1
        or body.get("maximum_measurements_per_endpoint_signature") != 1
    ):
        _terminal("LEDGER_CONSUMED_DIFFERS", "consumed ledger semantics differ")
    return dict(value), dict(capsule)


def _read_external_outcome_v1(
    path: Path,
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
) -> dict[str, object]:
    try:
        raw = _read_regular(path, "external campaign outcome")
    except GatePending as error:
        _terminal("POSTCONSUMPTION_OUTCOME_UNREADABLE", error.detail)
    value = _strict_json(raw, "external campaign outcome")
    return _validate_outcome_v1(value, contract, capsule)


def _validate_terminal_record_v1(
    value: object,
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    consumed: Mapping[str, Any],
) -> tuple[dict[str, object], dict[str, object]]:
    if type(value) is not dict:
        _terminal("LEDGER_WIRE", "terminal ledger must be one object")
    _validate_ledger_digest_v1(value, "terminal")
    body = value.get("body")
    if type(body) is not dict or set(body) != _TERMINAL_BODY_FIELDS:
        _terminal("LEDGER_WIRE", "terminal body field registry differs")
    outcome = _validate_outcome_v1(body.get("outcome"), contract, capsule)
    expected_outcome_path = Path(str(contract["fixed_paths"]["evidence_root"])) / (
        "campaign.outcome.json"
    )
    external_state = body.get("external_outcome_state")
    external_error = body.get("external_outcome_error_or_null")
    if (
        value.get("schema_version") != _core.LEDGER_SCHEMA_VERSION
        or value.get("contract_digest") != contract["contract_digest"]
        or value.get("capsule_digest") != capsule["capsule_digest"]
        or value.get("authorization_digest") != _core.OWNER_AUTHORIZATION_DIGEST
        or value.get("sequence") != 2
        or value.get("record_type") != "TERMINAL"
        or value.get("status") != outcome["status"]
        or value.get("previous_record_digest_or_null") != consumed["record_digest"]
        or body.get("outcome_digest") != outcome["outcome_digest"]
        or body.get("success") != outcome["success"]
        or body.get("retry_or_resume_allowed") is not False
        or body.get("supervisor_return_code") != outcome["supervisor_return_code"]
        or body.get("external_outcome_path") != os.fspath(expected_outcome_path)
        or external_state not in ("EXACT_PUBLISHED", "UNAVAILABLE_OR_INVALID")
        or (
            external_state == "EXACT_PUBLISHED"
            and external_error is not None
        )
        or (
            external_state == "UNAVAILABLE_OR_INVALID"
            and (type(external_error) is not str or not external_error or len(external_error) > 4096)
        )
    ):
        _terminal("LEDGER_TERMINAL_DIFFERS", "terminal ledger semantics differ")
    if external_state == "EXACT_PUBLISHED":
        external = _read_external_outcome_v1(expected_outcome_path, contract, capsule)
        if external != outcome:
            _terminal("LEDGER_TERMINAL_DIFFERS", "external outcome differs from terminal outcome")
    return dict(value), outcome


def _recover_existing_outcome_v1(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
) -> tuple[dict[str, object] | None, str | None]:
    path = Path(str(contract["fixed_paths"]["evidence_root"])) / "campaign.outcome.json"
    if not os.path.lexists(path):
        return None, "campaign.outcome.json is absent"
    try:
        return _read_external_outcome_v1(path, contract, capsule), None
    except BaseException as error:
        return None, f"{type(error).__name__}:{error}"[:4096]


def _command_result_v1(
    capsule: Mapping[str, Any],
    outcome: Mapping[str, object],
    *,
    replayed_existing_terminal: bool,
) -> dict[str, object]:
    this_entry = 0 if replayed_existing_terminal else 1
    return {
        "schema_version": "hegel-phase3a-q1-full-node6-command-result/1",
        "status": outcome["status"],
        "capsule_digest": capsule["capsule_digest"],
        "campaign_execution_count": outcome["campaign_execution_count"],
        "authorization_consumption_count": outcome["authorization_consumption_count"],
        "supervisor_popen_count": outcome["supervisor_popen_count"],
        "supervisor_return_code": outcome["supervisor_return_code"],
        "artifact_sha256_or_null": outcome["artifact_sha256_or_null"],
        "replayed_existing_terminal": replayed_existing_terminal,
        "campaign_execution_count_this_entry": this_entry,
        "authorization_consumption_count_this_entry": this_entry,
        "supervisor_popen_count_this_entry": 0
        if replayed_existing_terminal
        else outcome["supervisor_popen_count"],
        "retry_or_resume_allowed": False,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
    }


def _replay_existing(contract: Mapping[str, Any], expected_capsule: str) -> dict[str, object] | None:
    ledger = Path(str(contract["fixed_paths"]["ledger_root"]))
    if not os.path.lexists(ledger):
        return None
    if not ledger.is_dir() or ledger.is_symlink():
        _terminal("LEDGER_ROOT_DIFFERS", str(ledger))
    consumed = _read_record(ledger / "campaign.consumed.json")
    terminal = _read_record(ledger / "campaign.terminal.json")
    if consumed is None and terminal is None:
        try:
            ledger.rmdir()
        except OSError as error:
            _terminal(
                "LEDGER_ROOT_UNREADABLE_OR_NOT_EMPTY",
                f"cannot prove and remove an empty ledger root: {error}",
            )
        return None
    if consumed is None:
        _terminal("LEDGER_SEQUENCE_DIFFERS", "terminal exists without consumed")
    checked_consumed, capsule = _validate_consumed_record_v1(
        consumed,
        contract,
        expected_capsule,
    )
    if terminal is not None:
        _checked_terminal, outcome = _validate_terminal_record_v1(
            terminal,
            contract,
            capsule,
            checked_consumed,
        )
        return _command_result_v1(
            capsule,
            outcome,
            replayed_existing_terminal=True,
        )
    outcome, recovery_error = _recover_existing_outcome_v1(contract, capsule)
    if outcome is None:
        stdout_path = Path(str(contract["fixed_paths"]["stdout_path"]))
        stderr_path = Path(str(contract["fixed_paths"]["stderr_path"]))
        outcome = _unknown_outcome_value(
            contract,
            capsule,
            supervisor_popen_count=None,
            return_code=None,
            stdout_sha256=_best_effort_digest(stdout_path),
            stderr_sha256=_best_effort_digest(stderr_path),
            detail=(
                "CONSUMED_WITHOUT_TERMINAL_REENTRY:"
                + (recovery_error or "existing outcome is unavailable")
            ),
        )
    checked_outcome, _terminal_record = _finalize_consumed_v1(
        contract,
        capsule,
        checked_consumed,
        outcome,
    )
    return _command_result_v1(
        capsule,
        checked_outcome,
        replayed_existing_terminal=True,
    )


def _supervisor_command(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
) -> list[str]:
    target = contract["target"]
    fixed = contract["fixed_paths"]
    return [
        str(contract["execution_policy"]["supervisor_interpreter"]),
        "-I",
        "-S",
        "-B",
        os.fspath(PROJECT_ROOT / str(target["supervisor_relative_path"])),
        "--run",
        "--project-root",
        os.fspath(PROJECT_ROOT),
        "--source-commit",
        str(capsule["source_commit"]),
        "--capsule-digest",
        str(capsule["capsule_digest"]),
        "--artifact",
        str(fixed["artifact_path"]),
        "--run-root",
        str(fixed["run_root"]),
        "--trajectory-root",
        str(fixed["trajectory_root"]),
    ]


def _outcome_value(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    *,
    return_code: int | None,
    terminal_status: str | None,
    replay_passed: bool,
    stdout_sha256: str | None,
    stderr_sha256: str | None,
    artifact_sha256: str | None,
    decision: object,
    supervisor_popen_count: int | None = 1,
    postconsumption_error_or_null: str | None = None,
) -> dict[str, object]:
    value = {
        "schema_version": _core.OUTCOME_SCHEMA_VERSION,
        "contract_digest": contract["contract_digest"],
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": _core.OWNER_AUTHORIZATION_DIGEST,
        "campaign_execution_count": 1,
        "authorization_consumption_count": 1,
        "supervisor_popen_count": supervisor_popen_count,
        "supervisor_return_code": return_code,
        "supervisor_terminal_status_or_null": terminal_status,
        "supervisor_replay_passed": replay_passed,
        "supervisor_stdout_sha256": stdout_sha256,
        "supervisor_stderr_sha256": stderr_sha256,
        "artifact_sha256_or_null": artifact_sha256,
        "postconsumption_error_or_null": postconsumption_error_or_null,
        "status": decision.status,
        "success": decision.success,
        "retry_or_resume_allowed": False,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "outcome_digest": None,
    }
    unsigned = dict(value)
    unsigned.pop("outcome_digest")
    value["outcome_digest"] = _core.domain_digest_v1(_core.DOMAINS["outcome"], unsigned)
    return value


_OUTCOME_FIELDS = {
    "schema_version",
    "contract_digest",
    "capsule_digest",
    "authorization_digest",
    "campaign_execution_count",
    "authorization_consumption_count",
    "supervisor_popen_count",
    "supervisor_return_code",
    "supervisor_terminal_status_or_null",
    "supervisor_replay_passed",
    "supervisor_stdout_sha256",
    "supervisor_stderr_sha256",
    "artifact_sha256_or_null",
    "postconsumption_error_or_null",
    "status",
    "success",
    "retry_or_resume_allowed",
    "q1_state",
    "q1_gate_count",
    "q1_gate_mask",
    "q1_output_slots",
    "outcome_digest",
}


def _validate_outcome_v1(
    value: object,
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
) -> dict[str, object]:
    if type(value) is not dict or set(value) != _OUTCOME_FIELDS:
        _terminal("POSTCONSUMPTION_OUTCOME_INVALID", "outcome field registry differs")
    unsigned = dict(value)
    observed_digest = unsigned.pop("outcome_digest")
    digests = (
        value.get("supervisor_stdout_sha256"),
        value.get("supervisor_stderr_sha256"),
        value.get("artifact_sha256_or_null"),
    )
    popen_count = value.get("supervisor_popen_count")
    return_code = value.get("supervisor_return_code")
    error = value.get("postconsumption_error_or_null")
    status = value.get("status")
    terminal_status = value.get("supervisor_terminal_status_or_null")
    stdout_digest, stderr_digest, artifact_digest = digests
    if (
        value.get("schema_version") != _core.OUTCOME_SCHEMA_VERSION
        or value.get("contract_digest") != contract["contract_digest"]
        or value.get("capsule_digest") != capsule["capsule_digest"]
        or value.get("authorization_digest") != _core.OWNER_AUTHORIZATION_DIGEST
        or value.get("campaign_execution_count") != 1
        or value.get("authorization_consumption_count") != 1
        or popen_count not in (None, 0, 1)
        or (return_code is not None and type(return_code) is not int)
        or (
            terminal_status is not None
            and (
                type(terminal_status) is not str
                or not terminal_status
                or len(terminal_status) > 512
            )
        )
        or type(value.get("supervisor_replay_passed")) is not bool
        or any(
            item is not None
            and (type(item) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", item) is None)
            for item in digests
        )
        or (error is not None and (type(error) is not str or not error or len(error) > 4096))
        or status
        not in (
            _core.SATURATED,
            _core.GUARD_HIT,
            _core.RESOURCE_INFEASIBLE,
            _core.FAILED_TERMINAL,
            _core.OUTCOME_UNKNOWN_TERMINAL,
        )
        or type(value.get("success")) is not bool
        or value.get("retry_or_resume_allowed") is not False
        or value.get("q1_state") != "NOT_RUN"
        or value.get("q1_gate_count") != 0
        or value.get("q1_gate_mask") != 0
        or value.get("q1_output_slots") != [None] * 8
        or observed_digest
        != _core.domain_digest_v1(_core.DOMAINS["outcome"], unsigned)
    ):
        _terminal("POSTCONSUMPTION_OUTCOME_INVALID", "outcome semantics differ")
    if status == _core.SATURATED and (
        value.get("success") is not True
        or popen_count != 1
        or return_code != 0
        or terminal_status != _core.SATURATED
        or value.get("supervisor_replay_passed") is not True
        or stdout_digest is None
        or stderr_digest is None
        or artifact_digest is None
        or error is not None
    ):
        _terminal("POSTCONSUMPTION_OUTCOME_INVALID", "saturated outcome differs")
    if status == _core.OUTCOME_UNKNOWN_TERMINAL and (
        value.get("success") is not False
        or type(error) is not str
        or terminal_status is not None
        or value.get("supervisor_replay_passed") is not False
        or artifact_digest is not None
    ):
        _terminal("POSTCONSUMPTION_OUTCOME_INVALID", "unknown outcome differs")
    if status == _core.RESOURCE_INFEASIBLE and (
        value.get("success") is not False
        or popen_count != 1
        or return_code != 3
        or terminal_status != _core.RESOURCE_INFEASIBLE
        or value.get("supervisor_replay_passed") is not True
        or stdout_digest is None
        or stderr_digest is None
        or artifact_digest is None
        or error is not None
    ):
        _terminal("POSTCONSUMPTION_OUTCOME_INVALID", "infeasible outcome differs")
    if status == _core.GUARD_HIT and (
        value.get("success") is not False
        or popen_count != 1
        or return_code != 0
        or terminal_status != _core.GUARD_HIT
        or value.get("supervisor_replay_passed") is not True
        or stdout_digest is None
        or stderr_digest is None
        or artifact_digest is None
        or error is not None
    ):
        _terminal("POSTCONSUMPTION_OUTCOME_INVALID", "guard outcome differs")
    published_artifact_pair = (return_code, terminal_status) in (
        (0, _core.SATURATED),
        (0, _core.GUARD_HIT),
        (3, _core.RESOURCE_INFEASIBLE),
    )
    if status == _core.FAILED_TERMINAL and (
        value.get("success") is not False
        or popen_count != 1
        or type(return_code) is not int
        or stdout_digest is None
        or stderr_digest is None
        or value.get("supervisor_replay_passed") is not False
        or error is not None
        or (artifact_digest is not None) != published_artifact_pair
    ):
        _terminal("POSTCONSUMPTION_OUTCOME_INVALID", "failed outcome differs")
    return dict(value)


def _unknown_outcome_value(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    *,
    supervisor_popen_count: int | None,
    return_code: int | None,
    stdout_sha256: str | None,
    stderr_sha256: str | None,
    detail: str,
) -> dict[str, object]:
    decision = _core.CampaignDecisionV1(
        _core.OUTCOME_UNKNOWN_TERMINAL,
        False,
        False,
    )
    return _outcome_value(
        contract,
        capsule,
        return_code=return_code,
        terminal_status=None,
        replay_passed=False,
        stdout_sha256=stdout_sha256,
        stderr_sha256=stderr_sha256,
        artifact_sha256=None,
        decision=decision,
        supervisor_popen_count=supervisor_popen_count,
        postconsumption_error_or_null=detail[:4096],
    )


def _best_effort_digest(path: Path) -> str | None:
    try:
        status = path.stat(follow_symlinks=False)
        if path.is_symlink() or not path.is_file() or status.st_size > MAX_SMALL_JSON:
            return None
        return "sha256:" + sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _ensure_private_directory(path: Path) -> None:
    if os.path.lexists(path):
        try:
            status = path.stat(follow_symlinks=False)
        except OSError as error:
            raise OSError(f"cannot inspect {path}: {error}") from error
        if path.is_symlink() or not path.is_dir() or (status.st_mode & 0o777) != 0o700:
            raise OSError(f"existing directory identity differs: {path}")
        return
    path.mkdir(mode=0o700)


def _finalize_consumed_v1(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    consumed: Mapping[str, Any],
    outcome: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    checked_consumed, checked_capsule = _validate_consumed_record_v1(
        consumed,
        contract,
        str(capsule["capsule_digest"]),
    )
    if checked_capsule != capsule:
        _terminal("LEDGER_CAPSULE_DIFFERS", "finalizer capsule differs from consumed capsule")
    checked_outcome = _validate_outcome_v1(outcome, contract, capsule)
    evidence = Path(str(contract["fixed_paths"]["evidence_root"]))
    outcome_path = evidence / "campaign.outcome.json"
    external_state = "EXACT_PUBLISHED"
    external_error: str | None = None
    try:
        _ensure_private_directory(evidence)
        _publish_or_replay_exact(evidence, outcome_path.name, checked_outcome)
    except BaseException as error:
        external_state = "UNAVAILABLE_OR_INVALID"
        external_error = f"{type(error).__name__}:{error}"[:4096]
    terminal = _core.build_ledger_record_v1(
        contract,
        capsule,
        sequence=2,
        record_type="TERMINAL",
        status=str(checked_outcome["status"]),
        previous_record_digest_or_null=str(checked_consumed["record_digest"]),
        body={
            "outcome": checked_outcome,
            "outcome_digest": checked_outcome["outcome_digest"],
            "success": checked_outcome["success"],
            "retry_or_resume_allowed": False,
            "supervisor_return_code": checked_outcome["supervisor_return_code"],
            "external_outcome_path": os.fspath(outcome_path),
            "external_outcome_state": external_state,
            "external_outcome_error_or_null": external_error,
        },
    )
    ledger = Path(str(contract["fixed_paths"]["ledger_root"]))
    try:
        _publish_or_replay_exact(ledger, "campaign.terminal.json", terminal)
    except BaseException as error:
        _terminal(
            "POSTCONSUMPTION_TERMINAL_PUBLICATION_FAILED",
            f"{type(error).__name__}:{error}",
        )
    return checked_outcome, terminal


def _close_descriptor(descriptor: int) -> None:
    if descriptor < 0:
        return
    try:
        os.close(descriptor)
    except OSError:
        pass


def _stop_child_best_effort(child: subprocess.Popen[bytes] | None) -> None:
    if child is None:
        return
    try:
        if child.poll() is None:
            child.kill()
        child.wait()
    except BaseException:
        pass


def _execute_postconsumption_v1(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    consumed: Mapping[str, Any],
) -> dict[str, object]:
    evidence = Path(str(contract["fixed_paths"]["evidence_root"]))
    stdout_path = Path(str(contract["fixed_paths"]["stdout_path"]))
    stderr_path = Path(str(contract["fixed_paths"]["stderr_path"]))
    artifact_path = Path(str(contract["fixed_paths"]["artifact_path"]))
    child: subprocess.Popen[bytes] | None = None
    supervisor_popen_count = 0
    return_code: int | None = None
    stdout_digest: str | None = None
    stderr_digest: str | None = None
    outcome: dict[str, object]
    try:
        _ensure_private_directory(evidence)
        stdout_fd = -1
        stderr_fd = -1
        try:
            stdout_fd = os.open(
                stdout_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o400,
            )
            stderr_fd = os.open(
                stderr_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o400,
            )
            child = subprocess.Popen(
                _supervisor_command(contract, capsule),
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout_fd,
                stderr=stderr_fd,
                close_fds=True,
                env={
                    "HOME": "/nonexistent",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PATH": "/usr/bin:/bin",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
            supervisor_popen_count = 1
        finally:
            _close_descriptor(stdout_fd)
            _close_descriptor(stderr_fd)
        try:
            return_code = child.wait(
                timeout=float(contract["execution_policy"]["supervisor_timeout_seconds"])
            )
        except subprocess.TimeoutExpired:
            return_code = 124
            child.kill()
            child.wait()
        stdout_raw = _read_regular(stdout_path, "supervisor stdout")
        stderr_raw = _read_regular(stderr_path, "supervisor stderr")
        stdout_digest = "sha256:" + sha256(stdout_raw).hexdigest()
        stderr_digest = "sha256:" + sha256(stderr_raw).hexdigest()
        terminal_status: str | None = None
        stdout_value: object = None
        if stdout_raw:
            try:
                stdout_value = _strict_json(stdout_raw, "supervisor stdout")
            except CampaignTerminal:
                stdout_value = None
            if type(stdout_value) is dict and type(stdout_value.get("terminal_status")) is str:
                terminal_status = stdout_value["terminal_status"]
            elif type(stdout_value) is dict and type(stdout_value.get("status")) is str:
                terminal_status = stdout_value["status"]
        artifact_digest: str | None = None
        replay_passed = False
        if (return_code, terminal_status) in (
            (0, _core.SATURATED),
            (0, _core.GUARD_HIT),
            (3, _core.RESOURCE_INFEASIBLE),
        ):
            artifact_digest, _artifact_size = _sha256_file(
                artifact_path,
                "full-node6 diagnostic artifact",
                maximum=2 * 1024 * 1024 * 1024,
            )
            replay_passed = (
                type(stdout_value) is dict
                and stdout_value.get("artifact_sha256") == artifact_digest
                and stdout_value.get("host_replay_passed") is True
                and stdout_value.get("q1_state") == "NOT_RUN"
                and stdout_value.get("q1_gate_count") == 0
                and stdout_value.get("q1_gate_mask") == 0
                and stdout_value.get("q1_output_slots") == [None] * 8
            )
        decision = _core.decide_supervisor_return_v1(
            return_code=return_code,
            terminal_status_or_null=terminal_status,
            replay_passed=replay_passed,
        )
        outcome = _outcome_value(
            contract,
            capsule,
            return_code=return_code,
            terminal_status=terminal_status,
            replay_passed=replay_passed,
            stdout_sha256=stdout_digest,
            stderr_sha256=stderr_digest,
            artifact_sha256=artifact_digest,
            decision=decision,
            supervisor_popen_count=supervisor_popen_count,
        )
    except BaseException as error:
        _stop_child_best_effort(child)
        if stdout_digest is None:
            stdout_digest = _best_effort_digest(stdout_path)
        if stderr_digest is None:
            stderr_digest = _best_effort_digest(stderr_path)
        outcome = _unknown_outcome_value(
            contract,
            capsule,
            supervisor_popen_count=supervisor_popen_count,
            return_code=return_code,
            stdout_sha256=stdout_digest,
            stderr_sha256=stderr_digest,
            detail=f"{type(error).__name__}:{error}",
        )
    checked_outcome, _terminal_record = _finalize_consumed_v1(
        contract,
        capsule,
        consumed,
        outcome,
    )
    return _command_result_v1(
        capsule,
        checked_outcome,
        replayed_existing_terminal=False,
    )


def execute_once_v1(expected_capsule_digest: str) -> dict[str, object]:
    if type(expected_capsule_digest) is not str or re.fullmatch(
        r"sha256:[0-9a-f]{64}", expected_capsule_digest
    ) is None:
        raise GatePending("PATH_AND_FORMAT", "CAPSULE_DIGEST_REQUIRED", "use exact preflight digest")
    contract = load_contract_v1()
    lock_path = Path(str(contract["fixed_paths"]["campaign_lock_path"]))
    with _CampaignLock(
        lock_path,
        capsule_digest_hint=expected_capsule_digest,
    ):
        replayed = _replay_existing(contract, expected_capsule_digest)
        if replayed is not None:
            return replayed
        try:
            preflight = derive_preflight_v1(include_live_gates=True)
        except GatePending as error:
            if error.capsule_digest_hint is None:
                error.capsule_digest_hint = expected_capsule_digest
            raise
        capsule = preflight["capsule"]
        if capsule["capsule_digest"] != expected_capsule_digest:
            _terminal(
                "CAPSULE_REPLAY_DIFFERS",
                f"expected={expected_capsule_digest}; replayed={capsule['capsule_digest']}",
            )
        try:
            _state_absence_gate(contract, allow_ledger=False)
        except GatePending as error:
            error.capsule = dict(capsule)
            error.capsule_digest_hint = expected_capsule_digest
            raise
        ledger = Path(str(contract["fixed_paths"]["ledger_root"]))
        try:
            ledger.mkdir(mode=0o700)
        except OSError as error:
            raise GatePending(
                "PATH_AND_FORMAT",
                "LEDGER_ROOT_CREATION_FAILED",
                str(error),
                capsule=capsule,
                capsule_digest_hint=expected_capsule_digest,
            )
        consumed = _core.build_ledger_record_v1(
            contract,
            capsule,
            sequence=1,
            record_type="CONSUMED",
            status=_core.CONSUMED,
            previous_record_digest_or_null=None,
            body={
                "capsule": capsule,
                "maximum_supervisor_popen_count": 1,
                "maximum_measurements_per_endpoint_signature": 1,
            },
        )
        try:
            _publish_or_replay_exact(ledger, "campaign.consumed.json", consumed)
        except BaseException as error:
            _terminal(
                "CONSUMED_LEDGER_PUBLICATION_FAILED",
                f"{type(error).__name__}:{error}",
            )
        return _execute_postconsumption_v1(contract, capsule, consumed)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise GatePending("PATH_AND_FORMAT", "CLI_ARGUMENTS_DIFFER", message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--capsule-digest")
    return parser


def _emit(value: object, *, stream: object = sys.stdout) -> None:
    stream.write(_core.canonical_json_bytes_v1(value).decode("ascii"))
    stream.flush()


def main(arguments: Sequence[str] | None = None) -> int:
    capsule: dict[str, object] | None = None
    contract: dict[str, object] | None = None
    requested_capsule_digest: str | None = None
    try:
        options = _parser().parse_args(arguments)
        if (
            options.run
            and type(options.capsule_digest) is str
            and re.fullmatch(r"sha256:[0-9a-f]{64}", options.capsule_digest)
            is not None
        ):
            requested_capsule_digest = options.capsule_digest
        contract = load_contract_v1()
        if options.preflight:
            if options.capsule_digest is not None:
                raise GatePending("PATH_AND_FORMAT", "PREFLIGHT_CAPSULE_ARGUMENT_FORBIDDEN", "omit digest")
            result = derive_preflight_v1(include_live_gates=True)
            _emit(result)
            return 0
        if options.capsule_digest is None:
            raise GatePending("PATH_AND_FORMAT", "CAPSULE_DIGEST_REQUIRED", "provide exact preflight digest")
        result = execute_once_v1(options.capsule_digest)
        _emit(result)
        return 0 if result["status"] == _core.SATURATED else 1
    except GatePending as error:
        if (
            error.capsule_digest_hint is None
            and requested_capsule_digest is not None
        ):
            error.capsule_digest_hint = requested_capsule_digest
        if capsule is None and error.capsule is not None:
            capsule = error.capsule
        if contract is None:
            try:
                contract = load_contract_v1()
            except Exception:
                contract = None
        if capsule is not None and contract is not None:
            value = _core.build_pending_v1(
                contract,
                capsule,
                phase=error.phase,
                condition=error.condition,
                detail=error.detail,
            )
        else:
            value = {
                "schema_version": _core.PENDING_SCHEMA_VERSION,
                "status": _core.PENDING_NOT_CONSUMED,
                "phase": error.phase,
                "condition": error.condition,
                "detail": error.detail,
                "capsule_digest_or_null": error.capsule_digest_hint,
                "campaign_execution_count": 0,
                "authorization_consumption_count": 0,
                "supervisor_popen_count": 0,
                "q1_state": "NOT_RUN",
                "q1_gate_count": 0,
                "q1_gate_mask": 0,
                "q1_output_slots": [None] * 8,
            }
        _emit(value)
        return PENDING_EXIT_CODE
    except (CampaignTerminal, _core.FullNode6CampaignValidationError) as error:
        code = error.code if isinstance(error, CampaignTerminal) else "CAMPAIGN_WIRE_FAILED"
        detail = error.detail if isinstance(error, CampaignTerminal) else str(error)
        _emit(
            {
                "schema_version": "hegel-phase3a-q1-full-node6-campaign-error/1",
                "status": code,
                "detail": detail,
                "retry_or_resume_allowed": False,
                "q1_state": "NOT_RUN",
                "q1_gate_count": 0,
                "q1_gate_mask": 0,
                "q1_output_slots": [None] * 8,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
