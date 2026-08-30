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
import signal
import shutil
import subprocess
import sys
import tempfile
import time
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
ACTIVE_CONTAINER_MEMORY_BYTES = 14 * 1024 * 1024 * 1024
HOST_CONTROL_PLANE_MEMORY_RESERVE_BYTES = 2 * 1024 * 1024 * 1024
MINIMUM_AVAILABLE_MEMORY_BYTES = (
    ACTIVE_CONTAINER_MEMORY_BYTES + HOST_CONTROL_PLANE_MEMORY_RESERVE_BYTES
)
MINIMUM_AVAILABLE_STORAGE_BYTES = 164 * 1024 * 1024 * 1024
PYTHON_IMAGE = (
    "python@sha256:"
    "e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
)
RUST_IMAGE = (
    "rust@sha256:"
    "38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)
DOCKER = "/usr/bin/docker"
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


def _load_supervisor_module_v1() -> Any:
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
    except (OSError, ImportError) as error:
        raise GatePending(
            "PATH_AND_FORMAT",
            "SUPERVISOR_SOURCE_IMPORT_UNAVAILABLE",
            str(error),
        ) from error
    except Exception as error:
        _terminal("EXECUTION_SOURCE_CLOSURE_DIFFERS", str(error))
    return module


def _execution_source_registry_v1() -> dict[str, tuple[str, ...] | int]:
    module = _load_supervisor_module_v1()
    try:
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
            "commit the exact source-closed successor before capsule generation",
        )
    head = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    prefix = _git("rev-parse", "--show-prefix")
    if prefix != "Hegel Machine/":
        _terminal("SOURCE_PREFIX_DIFFERS", f"observed Git prefix={prefix!r}")
    ancestor = str(lineage["required_ancestor_commit"])
    required_parent = str(lineage["required_parent_commit"])
    parent = _git("rev-parse", "HEAD^")
    if parent != required_parent:
        _terminal(
            "SOURCE_LINEAGE_DIFFERS",
            (
                f"successor HEAD must be the direct child of {required_parent}; "
                f"observed parent={parent}"
            ),
        )
    merge_base = _git("merge-base", ancestor, required_parent)
    if merge_base != ancestor:
        _terminal(
            "SOURCE_LINEAGE_DIFFERS",
            (
                f"required parent must descend from {ancestor}; "
                f"observed merge-base={merge_base}"
            ),
        )
    observed_additive_repo_paths = tuple(
        row
        for row in _git(
            "diff", "--name-only", "--diff-filter=A", ancestor, "HEAD"
        ).splitlines()
        if row
    )
    if any(not row.startswith(prefix) for row in observed_additive_repo_paths):
        _terminal(
            "SOURCE_ADDITIVE_PATHS_DIFFER",
            f"an additive path escaped {prefix!r}: {observed_additive_repo_paths!r}",
        )
    observed_additive_paths = tuple(
        row.removeprefix(prefix) for row in observed_additive_repo_paths
    )
    expected_additive_paths = tuple(lineage["additive_paths_in_exact_order"])
    if observed_additive_paths != expected_additive_paths:
        _terminal(
            "SOURCE_ADDITIVE_PATHS_DIFFER",
            (
                f"expected={expected_additive_paths!r}; "
                f"observed={observed_additive_paths!r}"
            ),
        )
    observed_modified_repo_paths = tuple(
        row
        for row in _git(
            "diff", "--name-only", "--diff-filter=M", ancestor, "HEAD"
        ).splitlines()
        if row
    )
    if any(not row.startswith(prefix) for row in observed_modified_repo_paths):
        _terminal(
            "SOURCE_MODIFIED_PATHS_DIFFER",
            f"a modified path escaped {prefix!r}: {observed_modified_repo_paths!r}",
        )
    observed_modified_paths = tuple(
        row.removeprefix(prefix) for row in observed_modified_repo_paths
    )
    expected_modified_paths = tuple(lineage["modified_paths_in_exact_order"])
    if observed_modified_paths != expected_modified_paths:
        _terminal(
            "SOURCE_MODIFIED_PATHS_DIFFER",
            (
                f"expected={expected_modified_paths!r}; "
                f"observed={observed_modified_paths!r}"
            ),
        )
    forbidden_changes = _git(
        "diff", "--name-only", "--diff-filter=DRTUXB", ancestor, "HEAD"
    )
    if forbidden_changes:
        _terminal(
            "SOURCE_CHANGE_KIND_DIFFERS",
            f"forbidden changed paths: {forbidden_changes}",
        )
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
    source_paths = expected_additive_paths + expected_modified_paths
    for relative in source_paths:
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


def _superseded_campaign_gate(contract: Mapping[str, Any]) -> dict[str, object]:
    expected = contract["superseded_campaign"]
    consumed_path = Path(str(expected["consumed_ledger_path"]))
    consumed_raw = _read_regular(
        consumed_path,
        "superseded campaign consumed ledger",
    )
    consumed_raw_digest = "sha256:" + sha256(consumed_raw).hexdigest()
    consumed = _strict_json(consumed_raw, "superseded campaign consumed ledger")
    if type(consumed) is not dict:
        _terminal(
            "SUPERSEDED_CAMPAIGN_CONSUMED_DIFFERS",
            "consumed ledger must be one object",
        )
    _validate_ledger_digest_v1(consumed, "superseded campaign consumed")
    consumed_body = consumed.get("body")
    consumed_capsule = (
        consumed_body.get("capsule") if type(consumed_body) is dict else None
    )
    consumed_authorization = (
        consumed_capsule.get("authorization")
        if type(consumed_capsule) is dict
        else None
    )
    unsigned_capsule = (
        dict(consumed_capsule) if type(consumed_capsule) is dict else {}
    )
    observed_capsule_digest = unsigned_capsule.pop("capsule_digest", None)
    unsigned_authorization = (
        dict(consumed_authorization)
        if type(consumed_authorization) is dict
        else {}
    )
    observed_authorization_digest = unsigned_authorization.pop(
        "authorization_digest",
        None,
    )
    if (
        consumed_raw_digest != expected["consumed_ledger_file_sha256"]
        or consumed.get("record_digest")
        != expected["consumed_ledger_record_digest"]
        or consumed.get("capsule_digest") != expected["capsule_digest"]
        or consumed.get("contract_digest") != expected["contract_digest"]
        or consumed.get("authorization_digest")
        != expected["authorization_digest"]
        or consumed.get("schema_version")
        != "hegel-phase3a-q1-full-node6-ledger-record/3"
        or consumed.get("sequence") != 1
        or consumed.get("record_type") != "CONSUMED"
        or consumed.get("status") != _core.CONSUMED
        or consumed.get("previous_record_digest_or_null") is not None
        or type(consumed_capsule) is not dict
        or consumed_capsule.get("capsule_digest") != expected["capsule_digest"]
        or observed_capsule_digest != expected["capsule_digest"]
        or _core.domain_digest_v1(_core.DOMAINS["capsule"], unsigned_capsule)
        != observed_capsule_digest
        or consumed_capsule.get("source_commit") != expected["source_commit"]
        or consumed_capsule.get("source_tree") != expected["source_tree"]
        or consumed_capsule.get("source_closure_digest")
        != expected["source_closure_digest"]
        or consumed_capsule.get("contract_digest") != expected["contract_digest"]
        or type(consumed_authorization) is not dict
        or consumed_authorization.get("schema_version")
        != expected["authorization_schema_version"]
        or consumed_authorization.get("authorization_text")
        != expected["authorization_text"]
        or consumed_authorization.get("authorization_digest")
        != expected["authorization_digest"]
        or observed_authorization_digest != expected["authorization_digest"]
        or _core.domain_digest_v1(
            b"HEGEL/Q1/FULL_NODE6/AUTHORIZATION/SUCCESSOR_V5/V1\x00",
            unsigned_authorization,
        )
        != observed_authorization_digest
    ):
        _terminal(
            "SUPERSEDED_CAMPAIGN_CONSUMED_DIFFERS",
            "consumed ledger binding differs",
        )
    ledger_path = Path(str(expected["terminal_ledger_path"]))
    raw = _read_regular(ledger_path, "superseded campaign terminal ledger")
    raw_digest = "sha256:" + sha256(raw).hexdigest()
    if raw_digest != expected["terminal_ledger_file_sha256"]:
        _terminal("SUPERSEDED_CAMPAIGN_LEDGER_BYTES_DIFFER", raw_digest)
    value = _strict_json(raw, "superseded campaign terminal ledger")
    if type(value) is not dict or type(value.get("body")) is not dict:
        _terminal(
            "SUPERSEDED_CAMPAIGN_LEDGER_WIRE",
            "terminal ledger must contain one body object",
        )
    _validate_ledger_digest_v1(value, "superseded campaign terminal")
    body = value["body"]
    outcome = body.get("outcome")
    if type(outcome) is not dict:
        _terminal(
            "SUPERSEDED_CAMPAIGN_LEDGER_WIRE",
            "terminal ledger must contain one outcome object",
        )
    expected_outcome = {
        "artifact_sha256_or_null": expected["artifact_sha256_or_null"],
        "authorization_consumption_count": expected[
            "authorization_consumption_count"
        ],
        "authorization_digest": expected["authorization_digest"],
        "campaign_execution_count": expected["campaign_execution_count"],
        "capsule_digest": expected["capsule_digest"],
        "contract_digest": expected["contract_digest"],
        "outcome_digest": expected["outcome_digest"],
        "postconsumption_cleanup": expected["postconsumption_cleanup"],
        "postconsumption_error_or_null": expected[
            "postconsumption_error_or_null"
        ],
        "q1_gate_count": expected["q1_gate_count"],
        "q1_gate_mask": expected["q1_gate_mask"],
        "q1_output_slots": expected["q1_output_slots"],
        "q1_state": expected["q1_state"],
        "retry_or_resume_allowed": expected["retry_or_resume_allowed"],
        "schema_version": "hegel-phase3a-q1-full-node6-outcome/2",
        "status": expected["terminal_status"],
        "success": False,
        "supervisor_popen_count": expected["supervisor_popen_count"],
        "supervisor_replay_passed": expected["supervisor_replay_passed"],
        "supervisor_return_code": expected["supervisor_return_code"],
        "supervisor_stderr_sha256": expected[
            "supervisor_stderr_file_sha256"
        ],
        "supervisor_stdout_sha256": expected[
            "supervisor_stdout_file_sha256"
        ],
        "supervisor_terminal_status_or_null": expected["terminal_status"],
    }
    unsigned_outcome = dict(outcome)
    observed_outcome_digest = unsigned_outcome.pop("outcome_digest", None)
    if (
        value.get("record_digest") != expected["terminal_ledger_record_digest"]
        or value.get("previous_record_digest_or_null")
        != expected["consumed_ledger_record_digest"]
        or value.get("capsule_digest") != expected["capsule_digest"]
        or value.get("contract_digest") != expected["contract_digest"]
        or value.get("authorization_digest") != expected["authorization_digest"]
        or value.get("schema_version")
        != "hegel-phase3a-q1-full-node6-ledger-record/3"
        or value.get("sequence") != 2
        or value.get("record_type") != "TERMINAL"
        or value.get("status") != expected["terminal_status"]
        or body.get("outcome_digest") != expected["outcome_digest"]
        or body.get("external_outcome_path") != expected["outcome_path"]
        or body.get("external_outcome_state") != "EXACT_PUBLISHED"
        or body.get("external_outcome_error_or_null") is not None
        or body.get("success") is not False
        or body.get("retry_or_resume_allowed") is not False
        or body.get("supervisor_return_code")
        != expected["supervisor_return_code"]
        or outcome != expected_outcome
        or observed_outcome_digest != expected["outcome_digest"]
        or _core.domain_digest_v1(_core.DOMAINS["outcome"], unsigned_outcome)
        != observed_outcome_digest
    ):
        _terminal(
            "SUPERSEDED_CAMPAIGN_TERMINAL_DIFFERS",
            "failed predecessor terminal boundary differs",
        )
    outcome_path = Path(str(expected["outcome_path"]))
    outcome_raw = _read_regular(outcome_path, "superseded campaign outcome")
    outcome_file_digest = "sha256:" + sha256(outcome_raw).hexdigest()
    external_outcome = _strict_json(outcome_raw, "superseded campaign outcome")
    if (
        outcome_file_digest != expected["outcome_file_sha256"]
        or external_outcome != outcome
    ):
        _terminal(
            "SUPERSEDED_CAMPAIGN_OUTCOME_DIFFERS",
            "external outcome binding differs",
        )
    supervisor_stdout_path = Path(str(expected["supervisor_stdout_path"]))
    supervisor_stdout_raw = _read_regular(
        supervisor_stdout_path,
        "superseded campaign supervisor stdout",
    )
    supervisor_stdout_digest = "sha256:" + sha256(
        supervisor_stdout_raw
    ).hexdigest()
    supervisor_stdout = _strict_json(
        supervisor_stdout_raw,
        "superseded campaign supervisor stdout",
    )
    expected_supervisor_stdout = {
        "error_code": expected["error_code"],
        "error_detail": expected["error_detail"],
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "q1_state": "NOT_RUN",
        "schema_version": "hegel-phase3a-q1-full-node6-supervisor-error/1",
        "terminal_status": expected["terminal_status"],
    }
    supervisor_stderr_path = Path(str(expected["supervisor_stderr_path"]))
    supervisor_stderr_raw = _read_regular(
        supervisor_stderr_path,
        "superseded campaign supervisor stderr",
    )
    supervisor_stderr_digest = "sha256:" + sha256(
        supervisor_stderr_raw
    ).hexdigest()
    expected_supervisor_stderr = (
        f'{expected["error_code"]}: {expected["error_detail"]}\n'.encode("ascii")
    )
    if (
        supervisor_stdout_digest
        != expected["supervisor_stdout_file_sha256"]
        or supervisor_stdout != expected_supervisor_stdout
        or supervisor_stderr_digest
        != expected["supervisor_stderr_file_sha256"]
        or supervisor_stderr_raw != expected_supervisor_stderr
    ):
        _terminal(
            "SUPERSEDED_CAMPAIGN_SUPERVISOR_EVIDENCE_DIFFERS",
            "supervisor stdout or stderr binding differs",
        )
    event_log_path = Path(str(expected["trajectory_event_log_path"]))
    event_log_raw = _read_regular(
        event_log_path,
        "superseded campaign trajectory event log",
    )
    event_log_digest = "sha256:" + sha256(event_log_raw).hexdigest()
    try:
        event_lines = event_log_raw.splitlines()
        events = [
            _strict_json(line, f"superseded trajectory event {index}")
            for index, line in enumerate(event_lines, 1)
        ]
    except CampaignTerminal:
        raise
    expected_sequence = (
        ("SUPERVISOR", "STARTED"),
        ("SOURCE_REPLAY", "STARTED"),
        ("SOURCE_REPLAY", "COMPLETED"),
        ("LOCAL_IMAGE_ADMISSION", "STARTED"),
        ("LOCAL_IMAGE_ADMISSION", "COMPLETED"),
        ("SOURCE_SNAPSHOTS", "STARTED"),
        ("SOURCE_SNAPSHOTS", "COMPLETED"),
        ("SECCOMP_PROFILE_PUBLICATION", "STARTED"),
        ("SECCOMP_PROFILE_PUBLICATION", "COMPLETED"),
        ("ADMISSION_PUBLICATION", "STARTED"),
        ("ADMISSION_PUBLICATION", "COMPLETED"),
        ("RUST_OFFLINE_BUILD", "STARTED"),
        ("RUST_OFFLINE_BUILD", "COMPLETED"),
        ("SNAPSHOT_READONLY", "STARTED"),
        ("SNAPSHOT_READONLY", "COMPLETED"),
        ("MEASUREMENT_AND_REPLAY", "STARTED"),
        ("ENDPOINT_COMMAND_PREPARATION", "STARTED"),
        ("ENDPOINT_COMMAND_PREPARATION", "COMPLETED"),
        ("SCRATCH_PREPARATION", "STARTED"),
        ("SCRATCH_PREPARATION", "COMPLETED"),
        ("ENDPOINT_SEQUENCE", "STARTED"),
        ("PYTHON_ENDPOINT", "STARTED"),
        ("PYTHON_ENDPOINT", "COMPLETED"),
        ("RUST_ENDPOINT", "STARTED"),
        ("RUST_ENDPOINT", "COMPLETED"),
        ("SCRATCH_CLEANUP", "STARTED"),
        ("SCRATCH_CLEANUP", "COMPLETED"),
        ("ENDPOINT_SEQUENCE", "COMPLETED"),
        ("ENDPOINT_OUTPUT_SEAL", "STARTED"),
        ("ENDPOINT_OUTPUT_SEAL", "COMPLETED"),
        ("TRUSTED_HOST_REPLAY", "STARTED"),
        ("TRUSTED_HOST_REPLAY", "FAILED"),
        ("MEASUREMENT_AND_REPLAY", "FAILED"),
        ("TERMINAL", "FAILED"),
    )
    observed_sequence: list[tuple[object, object]] = []
    for index, event in enumerate(events, 1):
        if (
            type(event) is not dict
            or event.get("schema_version")
            != "hegel-phase3a-q1-full-node6-supervisor-trajectory-event/1"
            or event.get("event_ordinal") != index
            or event.get("capsule_digest") != expected["capsule_digest"]
            or event.get("source_commit") != expected["source_commit"]
        ):
            _terminal(
                "SUPERSEDED_CAMPAIGN_TRAJECTORY_DIFFERS",
                f"trajectory event {index} differs",
            )
        observed_sequence.append((event.get("phase"), event.get("status")))
    if (
        event_log_digest != expected["trajectory_event_log_file_sha256"]
        or len(events) != expected["trajectory_event_count"]
        or tuple(observed_sequence) != expected_sequence
    ):
        _terminal(
            "SUPERSEDED_CAMPAIGN_TRAJECTORY_DIFFERS",
            f"observed sequence={tuple(observed_sequence)!r}",
        )
    trajectory_terminal_path = Path(str(expected["trajectory_terminal_path"]))
    trajectory_terminal_raw = _read_regular(
        trajectory_terminal_path,
        "superseded campaign terminal trajectory",
    )
    trajectory_terminal_digest = "sha256:" + sha256(
        trajectory_terminal_raw
    ).hexdigest()
    trajectory_terminal = _strict_json(
        trajectory_terminal_raw,
        "superseded campaign terminal trajectory",
    )
    if (
        trajectory_terminal_digest
        != expected["trajectory_terminal_file_sha256"]
        or type(trajectory_terminal) is not dict
        or trajectory_terminal.get("capsule_digest") != expected["capsule_digest"]
        or trajectory_terminal.get("source_commit") != expected["source_commit"]
        or trajectory_terminal.get("event_count")
        != expected["trajectory_event_count"]
        or trajectory_terminal.get("event_log_path") != os.fspath(event_log_path)
        or trajectory_terminal.get("terminal_status")
        != expected["terminal_status"]
        or trajectory_terminal.get("error_code_or_null")
        != expected["error_code"]
        or trajectory_terminal.get("error_detail_or_null")
        != expected["error_detail"]
    ):
        _terminal(
            "SUPERSEDED_CAMPAIGN_TRAJECTORY_DIFFERS",
            "terminal trajectory binding differs",
        )
    python_starts = observed_sequence.count(("PYTHON_ENDPOINT", "STARTED"))
    rust_starts = observed_sequence.count(("RUST_ENDPOINT", "STARTED"))
    host_replay_starts = observed_sequence.count(("TRUSTED_HOST_REPLAY", "STARTED"))
    if (
        python_starts != expected["python_endpoint_start_count"]
        or rust_starts != expected["rust_endpoint_start_count"]
        or host_replay_starts != expected["trusted_host_replay_start_count"]
        or python_starts + rust_starts
        != expected["measurement_endpoint_start_count"]
        or python_starts != expected["python_odd_measurement_count"]
        or python_starts != expected["python_sink_measurement_count"]
        or rust_starts != expected["rust_odd_measurement_count"]
        or rust_starts != expected["rust_sink_measurement_count"]
        or host_replay_starts
        != expected["trusted_host_replay_measurement_count"]
    ):
        _terminal(
            "SUPERSEDED_CAMPAIGN_TRAJECTORY_DIFFERS",
            "measurement actor start counts differ",
        )
    python_completed = events[22]
    rust_completed = events[24]
    scratch_completed = events[26]
    sequence_completed = events[27]
    host_failed = events[31]
    measurement_failed = events[32]
    terminal_failed = events[33]
    failure_detail = f'{expected["error_code"]}: {expected["error_detail"]}'
    if (
        python_completed.get("return_code_or_null")
        != expected["endpoint_results"]["python"]["return_code"]
        or python_completed.get("command_artifact_path_or_null")
        != expected["resource_evidence"]["python"]["path"]
        or rust_completed.get("return_code_or_null")
        != expected["endpoint_results"]["rust"]["return_code"]
        or rust_completed.get("command_artifact_path_or_null")
        != expected["resource_evidence"]["rust"]["path"]
        or scratch_completed.get("command_artifact_path_or_null")
        != expected["scratch_cleanup"]["path"]
        or sequence_completed.get("return_code_or_null")
        != 0
        or sequence_completed.get("command_artifact_path_or_null")
        != expected["resource_evidence"]["aggregate"]["path"]
        or sequence_completed.get("detail_or_null")
        != "python_return_code=0;rust_return_code=0"
        or host_failed.get("return_code_or_null") is not None
        or host_failed.get("detail_or_null") != failure_detail
        or measurement_failed.get("detail_or_null") != failure_detail
        or terminal_failed.get("decision_or_null")
        != expected["terminal_status"]
        or terminal_failed.get("detail_or_null") != expected["error_detail"]
    ):
        _terminal(
            "SUPERSEDED_CAMPAIGN_TRAJECTORY_DIFFERS",
            "endpoint return-code or terminal-stage evidence differs",
        )

    def read_bound_json(
        path_text: object,
        label: str,
        expected_digest: object,
    ) -> tuple[dict[str, object], str, bytes]:
        evidence_raw = _read_regular(Path(str(path_text)), label)
        evidence_digest = "sha256:" + sha256(evidence_raw).hexdigest()
        evidence = _strict_json(evidence_raw, label)
        if evidence_digest != expected_digest or type(evidence) is not dict:
            _terminal(
                "SUPERSEDED_CAMPAIGN_RUN_EVIDENCE_DIFFERS",
                f"{label} bytes or wire differs",
            )
        return evidence, evidence_digest, evidence_raw

    endpoint_receipts: dict[str, dict[str, object]] = {}
    for actor in ("python", "rust"):
        endpoint_expected = expected["endpoint_results"][actor]
        endpoint_stdout, endpoint_stdout_digest, _ = read_bound_json(
            endpoint_expected["stdout_path"],
            f"superseded campaign {actor} endpoint stdout",
            endpoint_expected["stdout_file_sha256"],
        )
        endpoint_stderr_path = Path(str(endpoint_expected["stderr_path"]))
        endpoint_stderr_raw = _read_regular(
            endpoint_stderr_path,
            f"superseded campaign {actor} endpoint stderr",
        )
        endpoint_stderr_digest = "sha256:" + sha256(
            endpoint_stderr_raw
        ).hexdigest()
        stdout_summary = endpoint_expected["stdout_summary"]
        partitions = endpoint_stdout.get("partitions")
        if (
            any(
                endpoint_stdout.get(name) != expected_value
                for name, expected_value in stdout_summary.items()
            )
            or endpoint_stderr_digest
            != endpoint_expected["stderr_file_sha256"]
            or endpoint_stderr_raw != b""
            or type(partitions) is not list
            or len(partitions) != 2
            or [
                row.get("input_signature_id")
                for row in partitions
                if type(row) is dict
            ]
            != [1, 2]
            or [
                row.get("terminal_status")
                for row in partitions
                if type(row) is dict
            ]
            != [_core.SATURATED, _core.SATURATED]
        ):
            _terminal(
                "SUPERSEDED_CAMPAIGN_ENDPOINT_EVIDENCE_DIFFERS",
                f"{actor} endpoint saturated evidence differs",
            )
        endpoint_receipts[actor] = {
            "return_code": endpoint_expected["return_code"],
            "stdout_file_sha256": endpoint_stdout_digest,
            "stderr_file_sha256": endpoint_stderr_digest,
            "terminal_status": endpoint_stdout["terminal_status"],
            "actual_preflight_peak_scratch_bytes": endpoint_stdout[
                "actual_preflight_peak_scratch_bytes"
            ],
            "actual_preflight_terminal_live_scratch_bytes": endpoint_stdout[
                "actual_preflight_terminal_live_scratch_bytes"
            ],
            "q1_state": endpoint_stdout["q1_state"],
            "q1_gate_count": endpoint_stdout["q1_gate_count"],
            "q1_gate_mask": endpoint_stdout["q1_gate_mask"],
            "q1_output_slots": endpoint_stdout["q1_output_slots"],
        }

    host_expected = expected["trusted_host_replay_failure"]
    host_stdout, host_stdout_digest, _ = read_bound_json(
        host_expected["stdout_path"],
        "superseded campaign trusted host replay stdout",
        host_expected["stdout_file_sha256"],
    )
    host_stderr_raw = _read_regular(
        Path(str(host_expected["stderr_path"])),
        "superseded campaign trusted host replay stderr",
    )
    host_stderr_digest = "sha256:" + sha256(host_stderr_raw).hexdigest()
    if (
        host_stdout != host_expected["stdout_object"]
        or host_stderr_digest != host_expected["stderr_file_sha256"]
        or host_stderr_raw.decode("ascii") != host_expected["stderr_text"]
    ):
        _terminal(
            "SUPERSEDED_CAMPAIGN_HOST_REPLAY_EVIDENCE_DIFFERS",
            "trusted host replay stdout or stderr evidence differs",
        )
    host_receipt = {
        "return_code": host_expected["return_code"],
        "stdout_file_sha256": host_stdout_digest,
        "stderr_file_sha256": host_stderr_digest,
        "error_code": host_stdout["error_code"],
        "error_detail": host_stdout["error_detail"],
        "q1_state": host_stdout["q1_state"],
        "q1_gate_count": host_stdout["q1_gate_count"],
        "q1_gate_mask": host_stdout["q1_gate_mask"],
        "q1_output_slots": host_stdout["q1_output_slots"],
    }

    resource_expected = expected["resource_evidence"]
    resource_values: dict[str, dict[str, object]] = {}
    resource_digests: dict[str, str] = {}
    for actor in ("python", "rust", "aggregate", "host"):
        actor_expected = resource_expected[actor]
        resource_value, resource_digest, _ = read_bound_json(
            actor_expected["path"],
            f"superseded campaign {actor} resource evidence",
            actor_expected["file_sha256"],
        )
        resource_values[actor] = resource_value
        resource_digests[actor] = resource_digest
    resource_schema = (
        "hegel-phase3a-q1-full-node6-container-resource-evidence/1"
    )
    python_rows = resource_values["python"].get("rows")
    rust_rows = resource_values["rust"].get("rows")
    aggregate_rows = resource_values["aggregate"].get("rows")
    host_rows = resource_values["host"].get("rows")
    if (
        resource_values["python"].get("schema_version") != resource_schema
        or resource_values["rust"].get("schema_version") != resource_schema
        or resource_values["aggregate"].get("schema_version")
        != resource_schema
        or resource_values["host"].get("schema_version") != resource_schema
        or type(python_rows) is not list
        or len(python_rows) != resource_expected["python"]["row_count"]
        or type(rust_rows) is not list
        or len(rust_rows) != resource_expected["rust"]["row_count"]
        or type(aggregate_rows) is not list
        or len(aggregate_rows) != resource_expected["aggregate"]["row_count"]
        or type(host_rows) is not list
        or len(host_rows) != resource_expected["host"]["row_count"]
        or aggregate_rows != python_rows + rust_rows
        or type(python_rows[0]) is not dict
        or type(rust_rows[0]) is not dict
        or type(host_rows[0]) is not dict
    ):
        _terminal(
            "SUPERSEDED_CAMPAIGN_RESOURCE_EVIDENCE_DIFFERS",
            "per-endpoint and aggregate resource rows differ",
        )
    python_row = python_rows[0]
    rust_row = rust_rows[0]
    host_row = host_rows[0]
    if (
        python_row.get("actor_id")
        != resource_expected["python"]["actor_id"]
        or resource_expected["python"]["actor_command_return_code"]
        != expected["endpoint_results"]["python"]["return_code"]
        or python_row.get("actor_command_return_code")
        != resource_expected["python"]["actor_command_return_code"]
        or python_row.get("endpoint_sequence_ordinal")
        != resource_expected["python"]["endpoint_sequence_ordinal"]
        or rust_row.get("actor_id") != resource_expected["rust"]["actor_id"]
        or resource_expected["rust"]["actor_command_return_code"]
        != expected["endpoint_results"]["rust"]["return_code"]
        or rust_row.get("actor_command_return_code")
        != resource_expected["rust"]["actor_command_return_code"]
        or rust_row.get("endpoint_sequence_ordinal")
        != resource_expected["rust"]["endpoint_sequence_ordinal"]
        or host_row.get("actor_id") != resource_expected["host"]["actor_id"]
        or resource_expected["host"]["actor_command_return_code"]
        != host_expected["return_code"]
        or host_row.get("actor_command_return_code")
        != resource_expected["host"]["actor_command_return_code"]
        or [python_row.get("actor_id"), rust_row.get("actor_id")]
        != resource_expected["aggregate"]["actor_order"]
        or python_row.get("maximum_concurrently_live_endpoint_containers")
        != resource_expected["aggregate"][
            "maximum_concurrently_live_endpoint_containers"
        ]
        or rust_row.get("maximum_concurrently_live_endpoint_containers")
        != resource_expected["aggregate"][
            "maximum_concurrently_live_endpoint_containers"
        ]
    ):
        _terminal(
            "SUPERSEDED_CAMPAIGN_RESOURCE_EVIDENCE_DIFFERS",
            "resource actor, return-code, ordinal or live-count differs",
        )
    python_start = python_row.get("measurement_interval_start_monotonic_ns")
    python_exit = python_row.get("container_exit_monotonic_ns")
    rust_start = rust_row.get("measurement_interval_start_monotonic_ns")
    rust_exit = rust_row.get("container_exit_monotonic_ns")
    host_start = host_row.get("measurement_interval_start_monotonic_ns")
    host_exit = host_row.get("container_exit_monotonic_ns")
    if (
        any(
            type(item) is not int
            for item in (
                python_start,
                python_exit,
                rust_start,
                rust_exit,
                host_start,
                host_exit,
            )
        )
        or not (
            python_start
            < python_exit
            <= rust_start
            < rust_exit
            <= host_start
            < host_exit
        )
        or python_row.get("predecessor_container_exit_monotonic_ns") is not None
        or rust_row.get("predecessor_container_exit_monotonic_ns")
        != python_exit
        or resource_expected["aggregate"]["measurement_intervals_overlap"]
        is not False
    ):
        _terminal(
            "SUPERSEDED_CAMPAIGN_RESOURCE_EVIDENCE_DIFFERS",
            "strict Python-then-Rust non-overlap evidence differs",
        )

    scratch_expected = expected["scratch_cleanup"]
    scratch_value, scratch_digest, _ = read_bound_json(
        scratch_expected["path"],
        "superseded campaign scratch cleanup",
        scratch_expected["file_sha256"],
    )
    expected_scratch_rows = [
        {
            "host_path": os.fspath(
                Path(
                    str(expected["endpoint_results"]["python"]["stdout_path"])
                ).parent
                / "python-scratch"
            ),
            "root_absent_after_cleanup": True,
        },
        {
            "host_path": os.fspath(
                Path(
                    str(expected["endpoint_results"]["rust"]["stdout_path"])
                ).parent
                / "rust-scratch"
            ),
            "root_absent_after_cleanup": True,
        },
    ]
    if scratch_value != {
        "failure_rows": scratch_expected["failure_rows"],
        "rows": expected_scratch_rows,
        "schema_version": "hegel-phase3a-q1-full-node6-scratch-cleanup/1",
        "terminal_absence_proved": scratch_expected[
            "terminal_absence_proved"
        ],
    }:
        _terminal(
            "SUPERSEDED_CAMPAIGN_SCRATCH_CLEANUP_DIFFERS",
            "scratch terminal-absence evidence differs",
        )
    host_scratch_expected = expected["host_replay_scratch_cleanup"]
    host_scratch_value, host_scratch_digest, _ = read_bound_json(
        host_scratch_expected["path"],
        "superseded campaign trusted host replay scratch cleanup",
        host_scratch_expected["file_sha256"],
    )
    expected_host_scratch_rows = [
        {
            "host_path": os.fspath(
                Path(str(host_expected["stdout_path"])).parent
                / "host-replay-scratch"
            ),
            "root_absent_after_cleanup": True,
        }
    ]
    if host_scratch_value != {
        "failure_rows": host_scratch_expected["failure_rows"],
        "rows": expected_host_scratch_rows,
        "schema_version": "hegel-phase3a-q1-full-node6-scratch-cleanup/1",
        "terminal_absence_proved": host_scratch_expected[
            "terminal_absence_proved"
        ],
    }:
        _terminal(
            "SUPERSEDED_CAMPAIGN_HOST_SCRATCH_CLEANUP_DIFFERS",
            "trusted host replay scratch terminal-absence evidence differs",
        )
    return {
        "capsule_digest": value["capsule_digest"],
        "consumed_ledger_file_sha256": consumed_raw_digest,
        "consumed_ledger_record_digest": value["previous_record_digest_or_null"],
        "measurement_outcome": "KNOWN_TERMINAL_FAILURE",
        "measurement_endpoint_start_count": python_starts + rust_starts,
        "python_endpoint_start_count": python_starts,
        "rust_endpoint_start_count": rust_starts,
        "trusted_host_replay_start_count": host_replay_starts,
        "python_odd_measurement_count": python_starts,
        "python_sink_measurement_count": python_starts,
        "rust_odd_measurement_count": rust_starts,
        "rust_sink_measurement_count": rust_starts,
        "trusted_host_replay_measurement_count": host_replay_starts,
        "campaign_execution_count": outcome["campaign_execution_count"],
        "authorization_consumption_count": outcome[
            "authorization_consumption_count"
        ],
        "supervisor_popen_count": outcome["supervisor_popen_count"],
        "supervisor_return_code": outcome["supervisor_return_code"],
        "supervisor_replay_passed": outcome["supervisor_replay_passed"],
        "supervisor_stdout_file_sha256": supervisor_stdout_digest,
        "supervisor_stderr_file_sha256": supervisor_stderr_digest,
        "endpoint_results": endpoint_receipts,
        "trusted_host_replay_failure": host_receipt,
        "resource_evidence_file_sha256": resource_digests,
        "scratch_cleanup_file_sha256": {
            "endpoints": scratch_digest,
            "trusted_host_replay": host_scratch_digest,
        },
        "scratch_terminal_absence_proved": scratch_value[
            "terminal_absence_proved"
        ]
        and host_scratch_value["terminal_absence_proved"],
        "postconsumption_cleanup": outcome["postconsumption_cleanup"],
        "outcome_digest": body["outcome_digest"],
        "outcome_file_sha256": outcome_file_digest,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "q1_state": "NOT_RUN",
        "retry_or_resume_allowed": False,
        "terminal_ledger_file_sha256": raw_digest,
        "terminal_ledger_record_digest": value["record_digest"],
        "terminal_status": value["status"],
        "trajectory_event_count": len(events),
        "trajectory_event_log_file_sha256": event_log_digest,
        "trajectory_terminal_file_sha256": trajectory_terminal_digest,
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
        or measurement.get("python_and_rust_run_concurrently") is not False
        or measurement.get("python_then_rust_run_strictly_sequentially")
        is not True
        or measurement.get("maximum_concurrently_live_measurement_containers")
        != 1
        or measurement.get("endpoint_process_deadline_seconds") != 518400
        or measurement.get("endpoint_sequence_process_budget_seconds")
        != 1036800
        or measurement.get("trusted_host_replay_process_deadline_seconds")
        != 172800
        or value.get("execution_policy", {}).get(
            "outer_supervisor_wall_time_seconds"
        )
        != 1212000
        or value.get("execution_policy", {}).get("endpoint_execution_order")
        != ["PYTHON_ENDPOINT", "RUST_ENDPOINT"]
        or value.get("execution_policy", {}).get(
            "maximum_concurrently_live_endpoint_containers"
        )
        != 1
        or resources.get("maximum_concurrently_live_container_memory_bytes")
        != ACTIVE_CONTAINER_MEMORY_BYTES
        or resources.get("host_control_plane_memory_reserve_bytes")
        != HOST_CONTROL_PLANE_MEMORY_RESERVE_BYTES
        or resources.get("minimum_available_memory_bytes")
        != MINIMUM_AVAILABLE_MEMORY_BYTES
        or any(
            resources.get(actor, {}).get("memory_bytes")
            != ACTIVE_CONTAINER_MEMORY_BYTES
            or resources.get(actor, {}).get("memory_swap_bytes")
            != ACTIVE_CONTAINER_MEMORY_BYTES
            for actor in (
                "python_endpoint",
                "rust_endpoint",
                "trusted_host_replay",
            )
        )
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
        "endpoint_execution_order": ["PYTHON_ENDPOINT", "RUST_ENDPOINT"],
        "maximum_concurrently_live_measurement_containers": 1,
        "minimum_available_memory_bytes": MINIMUM_AVAILABLE_MEMORY_BYTES,
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
    except subprocess.CalledProcessError as error:
        stdout_tail = (error.stdout or b"")[-4096:].decode("utf-8", "replace")
        stderr_tail = (error.stderr or b"")[-4096:].decode("utf-8", "replace")
        raise GatePending(
            phase,
            condition,
            (
                f"{' '.join(command)}: return_code={error.returncode}; "
                f"stdout_tail={stdout_tail!r}; stderr_tail={stderr_tail!r}"
            ),
        )
    except OSError as error:
        raise GatePending(phase, condition, f"{' '.join(command)}: {error}")


def _remove_probe_root_v1(
    probe_root: Path,
    *,
    phase: str,
    condition: str,
) -> None:
    try:
        shutil.rmtree(probe_root)
        parent = os.open(probe_root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except OSError as error:
        raise GatePending(
            phase,
            condition,
            str(error),
        )


def _remove_dependency_probe_v1(probe_root: Path) -> None:
    try:
        for directory, _names, _files in os.walk(probe_root):
            Path(directory).chmod(0o700)
    except OSError as error:
        raise GatePending(
            "PYTHON_RUST_OFFLINE_DEPENDENCIES",
            "TRANSIENT_DEPENDENCY_PROBE_CLEANUP_FAILED",
            str(error),
        )
    _remove_probe_root_v1(
        probe_root,
        phase="PYTHON_RUST_OFFLINE_DEPENDENCIES",
        condition="TRANSIENT_DEPENDENCY_PROBE_CLEANUP_FAILED",
    )


def _dependency_gate(
    contract: Mapping[str, Any],
    source_commit: str,
) -> dict[str, object]:
    supervisor = _load_supervisor_module_v1()
    python_source_paths = tuple(supervisor.PYTHON_ENDPOINT_SOURCE_PATHS)
    rust_source_paths = tuple(supervisor.RUST_ENDPOINT_SOURCE_PATHS)
    host_source_paths = tuple(supervisor.HOST_REPLAY_SOURCE_PATHS)
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
    host_snapshot = probe_root / "host-replay-source"
    python_rows: Sequence[Mapping[str, object]] = ()
    rust_rows: Sequence[Mapping[str, object]] = ()
    host_rows: Sequence[Mapping[str, object]] = ()
    repository_prefix = ""
    try:
        if os.path.lexists(probe_root):
            raise OSError("dependency probe root already exists")
        probe_root.mkdir(mode=0o700)
        probe_created = True
        cargo_home = probe_root / "cargo-home"
        build_root = probe_root / "build"
        test_root = probe_root / "test"
        repository_prefix = supervisor._repository_tree_prefix_v1(PROJECT_ROOT)
        python_rows = supervisor._materialize_snapshot(
            PROJECT_ROOT,
            source_commit,
            python_snapshot,
            python_source_paths,
        )
        rust_rows = supervisor._materialize_snapshot(
            PROJECT_ROOT,
            source_commit,
            rust_snapshot,
            rust_source_paths,
        )
        host_rows = supervisor._materialize_host_replay_snapshot(
            PROJECT_ROOT,
            source_commit,
            host_snapshot,
        )
        supervisor._verify_snapshot_manifest_v1(python_snapshot, python_rows)
        supervisor._verify_snapshot_manifest_v1(rust_snapshot, rust_rows)
        supervisor._verify_snapshot_manifest_v1(host_snapshot, host_rows)
        if (
            len(python_rows) != len(python_source_paths)
            or len(rust_rows) != len(rust_source_paths)
            or len(host_rows) != len(host_source_paths)
            or int(supervisor.EXECUTION_CLOSURE_ROW_COUNT)
            != len(set(python_source_paths).union(rust_source_paths))
        ):
            _terminal(
                "EXACT_COMMIT_SNAPSHOT_CLOSURE_DIFFERS",
                "materialized snapshot row registry differs",
            )
        supervisor._make_tree_readonly(python_snapshot)
        cargo_cache.stat(follow_symlinks=False)
        if cargo_cache.is_symlink() or not cargo_cache.is_dir():
            raise OSError("Cargo cache is not an ordinary directory")
        shutil.copytree(cargo_cache, cargo_home)
        build_root.mkdir(mode=0o700)
        test_root.mkdir(mode=0o700)
    except (
        OSError,
        GatePending,
        CampaignTerminal,
        supervisor.FullNode6SupervisorError,
    ) as error:
        if probe_created:
            try:
                _remove_dependency_probe_v1(probe_root)
            except GatePending as cleanup_error:
                raise cleanup_error from error
        if isinstance(error, GatePending):
            raise
        if isinstance(error, CampaignTerminal):
            raise
        if isinstance(error, supervisor.FullNode6SupervisorError):
            raise GatePending(
                "EXACT_COMMIT_SNAPSHOT_MATERIALIZATION",
                "EXACT_COMMIT_BLOB_MATERIALIZATION_UNAVAILABLE",
                f"{error.code}: {error.detail}",
            ) from error
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
            "exact_commit_snapshot_materialization": "PASS",
            "host_replay_source_row_count": len(host_rows),
            "repository_tree_prefix": repository_prefix,
            "rust_image_id": rust_image.stdout.decode("ascii").strip(),
            "cargo_metadata_sha256": "sha256:" + sha256(metadata.stdout).hexdigest(),
            "cargo_test_locked_offline_all_targets": "PASS",
            "rust_release_build_source_row_count": len(rust_source_paths),
            "endpoint_execution_union_row_count": int(
                supervisor.EXECUTION_CLOSURE_ROW_COUNT
            ),
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


def _expected_runtime_isolation_live_gauntlet_receipt_v1() -> dict[str, object]:
    return {
        "actor_order": [
            "PYTHON_ENDPOINT",
            "RUST_ENDPOINT",
            "TRUSTED_HOST_REPLAY",
        ],
        "actor_results": {
            "PYTHON_ENDPOINT": "PASS",
            "RUST_ENDPOINT": "PASS",
            "TRUSTED_HOST_REPLAY": "PASS",
        },
        "container_workload": "/bin/true",
        "network_isolation": "PASS",
        "pinned_images": {
            "PYTHON_ENDPOINT": PYTHON_IMAGE,
            "RUST_ENDPOINT": RUST_IMAGE,
            "TRUSTED_HOST_REPLAY": PYTHON_IMAGE,
        },
        "production_endpoint_start_count": 0,
        "probe_root_cleanup": "PASS",
        "quotient_engine_invocation_count": 0,
        "schema_version": (
            "hegel-phase3a-q1-full-node6-runtime-isolation-live-gauntlet/1"
        ),
        "signature_selection_count": 0,
        "status": "PASS",
        "supervisor_popen_count": 0,
    }


def _expected_runtime_isolation_live_gauntlet_cleanup_receipt_v1() -> dict[str, object]:
    return {
        "container_absence": {
            "PYTHON_CONTROL": "PASS",
            "RUST_CONTROL": "PASS",
            "TRUSTED_HOST_REPLAY_CONTROL": "PASS",
        },
        "probe_root_absence": "PASS",
        "probe_root_parent_fsync": "PASS",
        "schema_version": (
            "hegel-phase3a-q1-full-node6-runtime-isolation-live-gauntlet-cleanup/1"
        ),
        "status": "PASS",
    }


def _replay_runtime_isolation_live_gauntlet_cleanup_or_terminal_v1(
    supervisor: Any,
    probe_root: Path,
    identity_suffix: str,
    primary_detail: str,
) -> dict[str, object]:
    try:
        cleanup = supervisor.runtime_isolation_live_gauntlet_cleanup_v1(
            probe_root,
            identity_suffix,
        )
    except (supervisor.FullNode6SupervisorError, OSError) as cleanup_error:
        _terminal(
            "RUNTIME_ISOLATION_LIVE_GAUNTLET_CLEANUP_UNCERTAIN",
            f"primary={primary_detail}; "
            f"cleanup={type(cleanup_error).__name__}:{cleanup_error}",
        )
    if (
        cleanup != _expected_runtime_isolation_live_gauntlet_cleanup_receipt_v1()
        or os.path.lexists(probe_root)
    ):
        _terminal(
            "RUNTIME_ISOLATION_LIVE_GAUNTLET_CLEANUP_UNCERTAIN",
            f"primary={primary_detail}; cleanup receipt or probe-root absence differs",
        )
    return cleanup


def _runtime_isolation_live_gauntlet_gate(
    contract: Mapping[str, Any],
    capsule_digest: str,
) -> dict[str, object]:
    probe_root = Path(str(contract["fixed_paths"]["probe_root"]))
    try:
        supervisor = _load_supervisor_module_v1()
    except GatePending as error:
        raise GatePending(
            "RUNTIME_ISOLATION_LIVE_GAUNTLET",
            "RUNTIME_ISOLATION_CONTROL_UNAVAILABLE",
            f"{error.phase}/{error.condition}: {error.detail}",
        ) from error
    identity_suffix = capsule_digest[-12:]
    try:
        receipt = supervisor.runtime_isolation_live_gauntlet_v1(
            PROJECT_ROOT,
            probe_root,
            identity_suffix,
        )
    except (supervisor.FullNode6SupervisorError, OSError) as error:
        _replay_runtime_isolation_live_gauntlet_cleanup_or_terminal_v1(
            supervisor,
            probe_root,
            identity_suffix,
            f"{type(error).__name__}:{error}",
        )
        code = getattr(error, "code", type(error).__name__)
        detail = getattr(error, "detail", str(error))
        raise GatePending(
            "RUNTIME_ISOLATION_LIVE_GAUNTLET",
            "RUNTIME_ISOLATION_CONTROL_UNAVAILABLE",
            f"{code}: {detail}; cleanup_replay=PASS",
        ) from error
    expected = _expected_runtime_isolation_live_gauntlet_receipt_v1()
    if receipt != expected or os.path.lexists(probe_root):
        _replay_runtime_isolation_live_gauntlet_cleanup_or_terminal_v1(
            supervisor,
            probe_root,
            identity_suffix,
            "exact live-gauntlet receipt or probe-root absence differs",
        )
        raise GatePending(
            "RUNTIME_ISOLATION_LIVE_GAUNTLET",
            "RUNTIME_ISOLATION_CONTROL_UNAVAILABLE",
            "exact live-gauntlet receipt or probe-root absence differs; "
            "cleanup_replay=PASS",
        )
    return expected


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
        "active_container_memory_bytes": ACTIVE_CONTAINER_MEMORY_BYTES,
        "available_memory_bytes": available_memory,
        "required_available_memory_bytes": MINIMUM_AVAILABLE_MEMORY_BYTES,
        "host_control_plane_memory_reserve_bytes": (
            HOST_CONTROL_PLANE_MEMORY_RESERVE_BYTES
        ),
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
    superseded_campaign = _superseded_campaign_gate(contract)
    technical = _technical_contract_gate(contract)
    dependency = _dependency_gate(contract, str(source["source_commit"]))
    static = {
        "authorization_digest": _core.OWNER_AUTHORIZATION_DIGEST,
        "source": source,
        "predecessor": predecessor,
        "superseded_campaign": superseded_campaign,
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
            live["runtime_isolation"] = _runtime_isolation_live_gauntlet_gate(
                contract,
                str(capsule["capsule_digest"]),
            )
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


def _publish_consumed_or_pending_v1(
    ledger: Path,
    consumed: Mapping[str, object],
    capsule: Mapping[str, object],
    expected_capsule_digest: str,
) -> None:
    final_path = ledger / "campaign.consumed.json"
    expected = _core.canonical_json_bytes_v1(consumed)
    try:
        _publish_no_replace(ledger, final_path.name, consumed)
        return
    except BaseException as publication_error:
        try:
            if os.path.lexists(final_path):
                status = final_path.stat(follow_symlinks=False)
                if (
                    final_path.is_symlink()
                    or not final_path.is_file()
                    or status.st_size != len(expected)
                    or final_path.read_bytes() != expected
                ):
                    _terminal(
                        "CONSUMED_LEDGER_PUBLICATION_UNCERTAIN",
                        "final consumed path exists but is not exact",
                    )
                remaining = tuple(ledger.iterdir())
                if len(remaining) != 1 or remaining[0] != final_path:
                    _terminal(
                        "CONSUMED_LEDGER_PUBLICATION_UNCERTAIN",
                        (
                            "exact final coexists with unexpected ledger entries: "
                            f"{[path.name for path in remaining]!r}"
                        ),
                    )
                descriptor = os.open(
                    ledger,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
                )
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                return
            if not ledger.is_dir() or ledger.is_symlink():
                _terminal(
                    "CONSUMED_LEDGER_PUBLICATION_UNCERTAIN",
                    "ledger root state is not an exact empty directory",
                )
            remaining = tuple(ledger.iterdir())
            if remaining:
                _terminal(
                    "CONSUMED_LEDGER_PUBLICATION_UNCERTAIN",
                    f"ledger root retains entries: {[path.name for path in remaining]!r}",
                )
            ledger.rmdir()
            parent_descriptor = os.open(
                ledger.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
            )
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
            if os.path.lexists(ledger) or os.path.lexists(final_path):
                _terminal(
                    "CONSUMED_LEDGER_PUBLICATION_UNCERTAIN",
                    "clean pre-consumption rollback could not be proved",
                )
        except CampaignTerminal:
            raise
        except OSError as recovery_error:
            _terminal(
                "CONSUMED_LEDGER_PUBLICATION_UNCERTAIN",
                (
                    f"publication={type(publication_error).__name__}:"
                    f"{publication_error}; recovery={type(recovery_error).__name__}:"
                    f"{recovery_error}"
                ),
            )
        raise GatePending(
            "PATH_AND_FORMAT",
            "CONSUMED_LEDGER_PUBLICATION_NOT_COMMITTED",
            f"{type(publication_error).__name__}:{publication_error}",
            capsule=capsule,
            capsule_digest_hint=expected_capsule_digest,
        ) from publication_error


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
    "runtime_isolation_live_gauntlet_receipt",
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
        or body.get("runtime_isolation_live_gauntlet_receipt")
        != _expected_runtime_isolation_live_gauntlet_receipt_v1()
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
            postconsumption_cleanup=_postconsumption_cleanup_v1(
                None,
                capsule,
                contract,
                supervisor_popen_attempted=True,
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
    postconsumption_cleanup: Mapping[str, object],
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
        "postconsumption_cleanup": dict(postconsumption_cleanup),
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
    "postconsumption_cleanup",
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
    cleanup = value.get("postconsumption_cleanup")
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
        or type(cleanup) is not dict
        or set(cleanup)
        != {
            "schema_version",
            "status",
            "supervisor_process_group",
            "production_containers",
            "control_containers",
            "probe_root",
            "production_scratch",
        }
        or cleanup.get("schema_version")
        != "hegel-phase3a-q1-full-node6-postconsumption-cleanup/1"
        or cleanup.get("status") not in ("PASS", "FAILED")
        or any(
            type(cleanup.get(name)) is not str or not cleanup.get(name)
            for name in (
                "supervisor_process_group",
                "production_containers",
                "control_containers",
                "probe_root",
                "production_scratch",
            )
        )
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
        or cleanup.get("status") != "PASS"
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
        or cleanup.get("status") != "PASS"
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
        or cleanup.get("status") != "PASS"
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
        or cleanup.get("status") != "PASS"
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
    postconsumption_cleanup: Mapping[str, object],
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
        postconsumption_cleanup=postconsumption_cleanup,
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


def _stop_child_best_effort(child: subprocess.Popen[bytes] | None) -> str:
    if child is None:
        return "NOT_STARTED"
    failures: list[str] = []
    group_kill = "ATTEMPTED"
    group_absence = "ATTEMPTED"
    fallback_kill = "NOT_REQUIRED"
    try:
        os.killpg(child.pid, signal.SIGKILL)
        group_kill = "PASS"
    except ProcessLookupError:
        group_kill = "ABSENT"
        group_absence = "PASS"
    except BaseException as error:
        group_kill = "FAILED"
        failures.append(f"killpg:{type(error).__name__}:{error}")
        fallback_kill = "ATTEMPTED"
        try:
            child.kill()
            fallback_kill = "PASS"
        except BaseException as fallback_error:
            fallback_kill = "FAILED"
            failures.append(
                f"kill:{type(fallback_error).__name__}:{fallback_error}"
            )
    wait_state = "ATTEMPTED"
    try:
        child.wait(timeout=30)
        wait_state = "PASS"
    except subprocess.TimeoutExpired as error:
        failures.append(f"wait1:{type(error).__name__}:{error}")
        if fallback_kill == "NOT_REQUIRED":
            fallback_kill = "ATTEMPTED"
            try:
                child.kill()
                fallback_kill = "PASS"
            except BaseException as fallback_error:
                fallback_kill = "FAILED"
                failures.append(
                    f"kill:{type(fallback_error).__name__}:{fallback_error}"
                )
        try:
            child.wait(timeout=30)
            wait_state = "PASS_AFTER_FALLBACK"
        except BaseException as final_wait_error:
            wait_state = "FAILED"
            failures.append(
                f"wait2:{type(final_wait_error).__name__}:{final_wait_error}"
            )
    except BaseException as error:
        wait_state = "FAILED"
        failures.append(f"wait:{type(error).__name__}:{error}")
    if group_absence != "PASS":
        deadline = time.monotonic() + 5.0
        while True:
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                group_absence = "PASS"
                break
            except BaseException as error:
                group_absence = "FAILED"
                failures.append(
                    f"group_absence:{type(error).__name__}:{error}"
                )
                break
            if time.monotonic() >= deadline:
                group_absence = "FAILED"
                failures.append("group_absence:timeout")
                break
            time.sleep(0.05)
    status = (
        "PASS"
        if group_absence == "PASS" and wait_state.startswith("PASS")
        else "FAILED"
    )
    detail = (
        f"{status}:group_kill={group_kill},group_absence={group_absence},"
        f"fallback_kill={fallback_kill},wait={wait_state}"
    )
    if failures:
        detail += "; failures=" + "|".join(failures)[:800]
    return detail


def _cleanup_named_containers_best_effort_v1(names: tuple[str, ...]) -> str:
    failures: list[str] = []
    for name in names:
        removal_failure: str | None = None
        try:
            removed = subprocess.run(
                [DOCKER, "rm", "-f", name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            removal_failure = f"{name}:remove:{type(error).__name__}:{error}"
        else:
            removed_stderr = removed.stderr.decode("utf-8", "replace")
            if (
                removed.returncode != 0
                and "No such container" not in removed_stderr
            ):
                removal_failure = (
                    f"{name}:remove:return_code={removed.returncode}:"
                    f"{removed_stderr[-300:]}"
                )
        try:
            inspected = subprocess.run(
                [DOCKER, "container", "inspect", name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            detail = f"{name}:inspect:{type(error).__name__}:{error}"
            if removal_failure is not None:
                detail += f"; remove={removal_failure}"
            failures.append(detail)
            continue
        inspected_stderr = inspected.stderr.decode("utf-8", "replace")
        if inspected.returncode == 0 or not any(
            marker in inspected_stderr
            for marker in ("No such container", "No such object")
        ):
            detail = (
                f"{name}:inspect:return_code={inspected.returncode}:"
                f"{inspected_stderr[-300:]}"
            )
            if removal_failure is not None:
                detail += f"; remove={removal_failure}"
            failures.append(detail)
    if failures:
        return "FAILED:" + "|".join(failures)[:900]
    return "PASS"


def _cleanup_production_containers_best_effort_v1(
    capsule: Mapping[str, Any],
) -> str:
    suffix = str(capsule["capsule_digest"])[-12:]
    return _cleanup_named_containers_best_effort_v1(
        (
            f"hegel-q1-node6-python-{suffix}",
            f"hegel-q1-node6-rust-{suffix}",
            f"hegel-q1-node6-host-replay-{suffix}",
        )
    )


def _cleanup_control_containers_best_effort_v1(
    capsule: Mapping[str, Any],
) -> str:
    suffix = str(capsule["capsule_digest"])[-12:]
    return _cleanup_named_containers_best_effort_v1(
        (
            f"hegel-q1-node6-isolation-control-python-{suffix}",
            f"hegel-q1-node6-isolation-control-rust-{suffix}",
            f"hegel-q1-node6-isolation-control-host-{suffix}",
        )
    )


def _cleanup_probe_root_best_effort_v1(contract: Mapping[str, Any]) -> str:
    probe_root = Path(str(contract["fixed_paths"]["probe_root"]))
    try:
        if probe_root.is_symlink() or (os.path.lexists(probe_root) and not probe_root.is_dir()):
            probe_root.unlink()
        elif probe_root.exists():
            shutil.rmtree(probe_root)
        if os.path.lexists(probe_root):
            raise OSError("probe-root absence proof failed")
        descriptor = os.open(
            probe_root.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        return f"FAILED:{type(error).__name__}:{error}"[:900]
    return "PASS"


def _cleanup_production_scratch_best_effort_v1(
    contract: Mapping[str, Any],
) -> str:
    run_root = Path(str(contract["fixed_paths"]["run_root"]))
    scratch_roots = tuple(
        run_root / name
        for name in (
            "python-scratch",
            "rust-scratch",
            "host-replay-scratch",
        )
    )
    failures: list[str] = []
    for root in scratch_roots:
        try:
            if root.is_symlink():
                raise OSError("scratch root is a symlink")
            if root.exists():
                if not root.is_dir():
                    raise OSError("scratch root is not a directory")
                shutil.rmtree(root)
            if os.path.lexists(root):
                raise OSError("scratch root absence proof failed")
        except OSError as error:
            failures.append(f"{root.name}:{type(error).__name__}:{error}")
    sync_root = (
        run_root
        if run_root.is_dir() and not run_root.is_symlink()
        else run_root.parent
    )
    try:
        if os.path.lexists(run_root) and sync_root != run_root:
            raise OSError("run root identity differs")
        descriptor = os.open(
            sync_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        failures.append(f"fsync:{type(error).__name__}:{error}")
    if failures:
        return "FAILED:" + "|".join(failures)[:900]
    return "PASS"


def _postconsumption_cleanup_v1(
    child: subprocess.Popen[bytes] | None,
    capsule: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    supervisor_popen_attempted: bool,
) -> dict[str, object]:
    def process_cleanup() -> str:
        if child is None and supervisor_popen_attempted:
            return "FAILED:supervisor Popen attempted but child handle unavailable"
        return _stop_child_best_effort(child)

    operations = (
        ("supervisor_process_group", process_cleanup),
        (
            "production_containers",
            lambda: _cleanup_production_containers_best_effort_v1(capsule),
        ),
        (
            "control_containers",
            lambda: _cleanup_control_containers_best_effort_v1(capsule),
        ),
        ("probe_root", lambda: _cleanup_probe_root_best_effort_v1(contract)),
        (
            "production_scratch",
            lambda: _cleanup_production_scratch_best_effort_v1(contract),
        ),
    )
    receipt: dict[str, object] = {
        "schema_version": (
            "hegel-phase3a-q1-full-node6-postconsumption-cleanup/1"
        )
    }
    for name, operation in operations:
        try:
            receipt[name] = operation()
        except BaseException as error:
            receipt[name] = f"FAILED:{type(error).__name__}:{error}"[:900]
    process = str(receipt["supervisor_process_group"])
    receipt["status"] = (
        "PASS"
        if (process == "NOT_STARTED" or process.startswith("PASS:"))
        and all(
            receipt[name] == "PASS"
            for name in (
                "production_containers",
                "control_containers",
                "probe_root",
                "production_scratch",
            )
        )
        else "FAILED"
    )
    return receipt


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
            supervisor_popen_count = 1
            child = subprocess.Popen(
                _supervisor_command(contract, capsule),
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout_fd,
                stderr=stderr_fd,
                close_fds=True,
                start_new_session=True,
                env={
                    "HOME": "/nonexistent",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PATH": "/usr/bin:/bin",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
        finally:
            _close_descriptor(stdout_fd)
            _close_descriptor(stderr_fd)
        try:
            return_code = child.wait(
                timeout=float(contract["execution_policy"]["supervisor_timeout_seconds"])
            )
        except subprocess.TimeoutExpired:
            return_code = 124
            raise CampaignTerminal(
                "SUPERVISOR_TIMEOUT",
                "return_code=124",
            )
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
        cleanup = _postconsumption_cleanup_v1(
            child,
            capsule,
            contract,
            supervisor_popen_attempted=supervisor_popen_count == 1,
        )
        if cleanup["status"] != "PASS":
            outcome = _unknown_outcome_value(
                contract,
                capsule,
                supervisor_popen_count=supervisor_popen_count,
                return_code=return_code,
                stdout_sha256=stdout_digest,
                stderr_sha256=stderr_digest,
                detail="POSTCONSUMPTION_CLEANUP_UNCERTAIN",
                postconsumption_cleanup=cleanup,
            )
        else:
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
                postconsumption_cleanup=cleanup,
                supervisor_popen_count=supervisor_popen_count,
            )
    except BaseException as error:
        cleanup = _postconsumption_cleanup_v1(
            child,
            capsule,
            contract,
            supervisor_popen_attempted=supervisor_popen_count == 1,
        )
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
            postconsumption_cleanup=cleanup,
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
                "runtime_isolation_live_gauntlet_receipt": preflight[
                    "live_gate_receipts"
                ]["runtime_isolation"],
            },
        )
        _publish_consumed_or_pending_v1(
            ledger,
            consumed,
            capsule,
            expected_capsule_digest,
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
