#!/usr/bin/env python3
"""Outer, fail-closed controller for the one authorized Q1 start campaign.

The controller publishes the capsule, records its deterministic authorization,
and consumes that authorization exactly once before handing control to the
three-actor supervisor.  Every entry replays the immutable predecessor and all
local execution bindings.  It never reads target, split, or role inputs and it
has no network operation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from hashlib import sha256
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import types
from typing import Any, Callable, Final, Iterator, Mapping, NoReturn, Sequence


sys.dont_write_bytecode = True
PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
PACKAGE_ROOT: Final = PROJECT_ROOT / "src" / "hegel_machine"
PACKAGE_NAME: Final = "hegel_machine"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_ROOT)]
    package.__package__ = PACKAGE_NAME
    sys.modules[PACKAGE_NAME] = package

campaign = importlib.import_module("hegel_machine.phase3_q1_start_campaign_v6")
campaign_v2 = importlib.import_module("hegel_machine.phase3_q1_start_campaign_v2")
campaign_v3 = importlib.import_module("hegel_machine.phase3_q1_start_campaign_v3")
campaign_v4 = importlib.import_module("hegel_machine.phase3_q1_start_campaign_v4")
campaign_v5 = importlib.import_module("hegel_machine.phase3_q1_start_campaign_v5")
successor = importlib.import_module(
    "hegel_machine.phase3_q1_prestart_successor_candidate_v6"
)
prestart = importlib.import_module(
    "hegel_machine.phase3_q1_prestart_authoritative_amendment_v1"
)
completion = importlib.import_module(
    "hegel_machine.phase3_q1_start_completion_wire_v1"
)

DEFAULT_CONTRACT: Final = PROJECT_ROOT / "config/phase3_q1_start_v6.json"
PRIOR_CONTRACT_RELATIVE: Final = (
    "config/phase3_q1_prestart_authoritative_amendment_v1.json"
)
FAILED_V2_CONTRACT_RELATIVE: Final = "config/phase3_q1_start_v2.json"
FAILED_V3_CONTRACT_RELATIVE: Final = "config/phase3_q1_start_v3.json"
FAILED_V4_CONTRACT_RELATIVE: Final = "config/phase3_q1_start_v4.json"
FAILED_V5_CONTRACT_RELATIVE: Final = "config/phase3_q1_start_v5.json"
SUPERVISOR_RELATIVE: Final = "tools/phase3_q1_start_supervisor_v6.py"
PENDING_EXIT_CODE: Final = 75
TERMINAL_EXIT_CODE: Final = 2
NOT_AUTHORIZED_EXIT_CODE: Final = 4
NOT_AUTHORIZED: Final = "Q1_START_V6_NOT_AUTHORIZED"
CONTROLLER_PENDING_SCHEMA: Final = "hegel-phase3a-q1-start-controller-pending/6"
CONTROLLER_RESULT_SCHEMA: Final = "hegel-phase3a-q1-start-controller-result/6"
PROJECTION_PROFILE_ROOT: Final = (
    "sha256:aa441cdc49ab60324483b9aa44e9fdfc324a6ad49a6bff50af6daa775209816d"
)
_TAGGED = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ACTOR_ORDER: Final = ("python", "rust", "host")
_PYTHON_FORMAL_COMMON: Final = (
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink2_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink3_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink4_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink5_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink6_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_v1.py",
    "src/hegel_machine/phase3_m3_dsl_core_v1.py",
    "src/hegel_machine/phase3_m3_record_wire_v1.py",
    "src/hegel_machine/phase3_m3_shrink1_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink2_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink3_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink4_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink5_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink6_core_v1.py",
    "src/hegel_machine/phase3_q0_evaluator_v1.py",
    "src/hegel_machine/phase3_q0_input_adapter_v1.py",
    "src/hegel_machine/phase3_q1_archive_projection_v1.py",
    "src/hegel_machine/phase3_q1_capacity_preflight_v1.py",
    "src/hegel_machine/phase3_q1_external_sort_profile_v1.py",
    "src/hegel_machine/phase3_q1_formal_archive_contract_v1.py",
    "src/hegel_machine/phase3_q1_partition_snapshot_v1.py",
    "src/hegel_machine/phase3_q1_quotient_contract_v1.py",
    "src/hegel_machine/phase3_q1_universe_v1.py",
    "src/hegel_machine/strict_ast_shrink1_v1.py",
    "src/hegel_machine/strict_ast_shrink2_v1.py",
    "src/hegel_machine/strict_ast_shrink3_v1.py",
    "src/hegel_machine/strict_ast_shrink4_v1.py",
    "src/hegel_machine/strict_ast_shrink5_v1.py",
    "src/hegel_machine/strict_ast_shrink6_v1.py",
    "src/hegel_machine/strict_ast_v1.py",
    "src/hegel_machine/strict_cbor_v1.py",
)
_ACTOR_SOURCE_RELATIVE_PATHS: Final = {
    "python": tuple(sorted((*_PYTHON_FORMAL_COMMON,
        "src/hegel_machine/phase3_q1_full_node6_capacity_preflight_v1.py",
        "src/hegel_machine/phase3_q1_full_node6_formal_materializer_v1.py",
        "src/hegel_machine/phase3_q1_full_node6_projection_v1.py",
        "src/hegel_machine/phase3_q1_prestart_wire_v1.py",
        "tools/phase3_q1_full_node6_formal_materializer_entrypoint_v1.py",
    ))),
    "host": tuple(sorted((*_PYTHON_FORMAL_COMMON,
        "src/hegel_machine/phase3_q1_formal_host_replay_v1.py",
        "tools/phase3_q1_formal_host_replay_v1.py",
    ))),
    "rust": (
        "rust/q1_archive_projection_oracle/Cargo.toml",
        "rust/q1_archive_projection_oracle/src/lib.rs",
        "rust/q1_full_capacity_preflight_engine/Cargo.lock",
        "rust/q1_full_capacity_preflight_engine/Cargo.toml",
        "rust/q1_full_capacity_preflight_engine/src/formal_materializer_main.rs",
        "rust/q1_full_capacity_preflight_engine/src/frozen_v1.rs",
        "rust/q1_full_capacity_preflight_engine/src/lib.rs",
        "rust/strict_canonicalizer/Cargo.toml",
        "rust/strict_canonicalizer/src/lib.rs",
        "rust/strict_canonicalizer_shrink1/Cargo.toml",
        "rust/strict_canonicalizer_shrink1/src/lib.rs",
        "rust/strict_canonicalizer_shrink2/Cargo.toml",
        "rust/strict_canonicalizer_shrink2/src/lib.rs",
        "rust/strict_canonicalizer_shrink3/Cargo.toml",
        "rust/strict_canonicalizer_shrink3/src/lib.rs",
        "rust/strict_canonicalizer_shrink4/Cargo.toml",
        "rust/strict_canonicalizer_shrink4/src/lib.rs",
        "rust/strict_canonicalizer_shrink5/Cargo.toml",
        "rust/strict_canonicalizer_shrink5/src/lib.rs",
        "rust/strict_canonicalizer_shrink6/Cargo.toml",
        "rust/strict_canonicalizer_shrink6/src/lib.rs",
    ),
}
_SUPERVISOR_AUTHORITY_KEYS: Final = frozenset(
    {
        "start_capsule_path",
        "start_capsule_raw_sha256",
        "start_authorization_path",
        "start_authorization_raw_sha256",
        "start_authorization_digest",
        "successor_prestart_bundle_path",
        "successor_prestart_bundle_raw_sha256",
        "successor_prestart_bundle_root",
        "base_authoritative_artifact_path",
        "base_authoritative_artifact_raw_sha256",
        "failed_start_v2_capsule_path",
        "failed_start_v2_capsule_raw_sha256",
        "failed_start_v2_capsule_digest",
        "failed_start_v2_authorization_path",
        "failed_start_v2_authorization_raw_sha256",
        "failed_start_v2_authorization_digest",
        "failed_start_v2_consumption_path",
        "failed_start_v2_consumption_raw_sha256",
        "failed_start_v2_consumption_record_digest",
        "failed_start_v2_terminal_path",
        "failed_start_v2_terminal_raw_sha256",
        "failed_start_v2_terminal_record_digest",
        "failed_start_v3_capsule_path",
        "failed_start_v3_capsule_raw_sha256",
        "failed_start_v3_capsule_digest",
        "failed_start_v3_authorization_path",
        "failed_start_v3_authorization_raw_sha256",
        "failed_start_v3_authorization_digest",
        "failed_start_v3_consumption_path",
        "failed_start_v3_consumption_raw_sha256",
        "failed_start_v3_consumption_record_digest",
        "failed_start_v3_terminal_path",
        "failed_start_v3_terminal_raw_sha256",
        "failed_start_v3_terminal_record_digest",
        "failed_start_v4_capsule_path",
        "failed_start_v4_capsule_raw_sha256",
        "failed_start_v4_capsule_digest",
        "failed_start_v4_authorization_path",
        "failed_start_v4_authorization_raw_sha256",
        "failed_start_v4_authorization_digest",
        "failed_start_v4_consumption_path",
        "failed_start_v4_consumption_raw_sha256",
        "failed_start_v4_consumption_record_digest",
        "failed_start_v4_terminal_path",
        "failed_start_v4_terminal_raw_sha256",
        "failed_start_v4_terminal_record_digest",
        "failed_start_v5_capsule_path",
        "failed_start_v5_capsule_raw_sha256",
        "failed_start_v5_capsule_digest",
        "failed_start_v5_authorization_path",
        "failed_start_v5_authorization_raw_sha256",
        "failed_start_v5_authorization_digest",
        "failed_start_v5_consumption_path",
        "failed_start_v5_consumption_raw_sha256",
        "failed_start_v5_consumption_record_digest",
        "failed_start_v5_terminal_path",
        "failed_start_v5_terminal_raw_sha256",
        "failed_start_v5_terminal_record_digest",
        "container_user",
        "corrected_resource_envelope_root",
        "actor_source_manifest_roots",
        "actor_source_closure_root",
    }
)
_SUPERVISOR_SUCCESS_KEYS: Final = frozenset(
    {
        "schema_version",
        "status",
        "capsule_digest",
        "source_commit",
        "run_genesis_root",
        "prestart_admission_receipt_root",
        "completed_stages",
        "checkpoint_paths",
        "host_result",
        "gate15_candidate",
        "gate16_candidate",
        "gate17_candidate",
        "gate18_runtime_receipt_inputs",
        "q1_completion_receipt_issued",
        "state_published",
    }
) | _SUPERVISOR_AUTHORITY_KEYS


class Q1StartControllerError(RuntimeError):
    pass


class ControllerPending(Q1StartControllerError):
    def __init__(self, phase: str, condition: str, detail: str) -> None:
        super().__init__(f"{phase}/{condition}: {detail}")
        self.phase = phase
        self.condition = condition
        self.detail = detail


class ControllerTerminal(Q1StartControllerError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class ConsumptionReplayTerminal(ControllerTerminal):
    """A present consumption final path whose exact meaning cannot replay."""


def _terminal(code: str, detail: str) -> NoReturn:
    raise ControllerTerminal(code, detail)


@dataclass(frozen=True, slots=True)
class SourceStateV6:
    identity: Mapping[str, object]
    rows: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class PredecessorReplayV6:
    base_capsule: Mapping[str, object]
    base_authorization: Mapping[str, object]
    base_consumption: Mapping[str, object]
    base_artifact: Mapping[str, object]
    failed_capsule: Mapping[str, object]
    failed_authorization: Mapping[str, object]
    failed_consumption: Mapping[str, object]
    failed_terminal: Mapping[str, object]
    failed_v3_capsule: Mapping[str, object]
    failed_v3_authorization: Mapping[str, object]
    failed_v3_consumption: Mapping[str, object]
    failed_v3_terminal: Mapping[str, object]
    failed_v4_capsule: Mapping[str, object]
    failed_v4_authorization: Mapping[str, object]
    failed_v4_consumption: Mapping[str, object]
    failed_v4_terminal: Mapping[str, object]
    failed_v5_capsule: Mapping[str, object]
    failed_v5_authorization: Mapping[str, object]
    failed_v5_consumption: Mapping[str, object]
    failed_v5_terminal: Mapping[str, object]
    successor_bundle: Mapping[str, object]


SourceStateReader = Callable[[Any], SourceStateV6]
PredecessorReader = Callable[[Any], PredecessorReplayV6]
FormalBlobReader = Callable[[Path, str, str], bytes]
LiveGate = Callable[[Any, Mapping[str, object]], None]
SupervisorInvoker = Callable[[Path, Path, Path], Mapping[str, object]]
StartupSmoke = Callable[[Any, Mapping[str, object]], Mapping[str, object]]
CompletionBuilder = Callable[
    [Mapping[str, object], Any, Mapping[str, object], Mapping[str, object]], bytes
]


@dataclass(frozen=True, slots=True)
class ControllerDependenciesV6:
    source_state: SourceStateReader
    replay_predecessor: PredecessorReader
    formal_blob: FormalBlobReader
    live_gate: LiveGate
    startup_smoke: StartupSmoke
    invoke_supervisor: SupervisorInvoker
    build_completion_artifact: CompletionBuilder


def _run_local(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: int = 60,
) -> bytes:
    try:
        result = subprocess.run(
            list(argv),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ControllerPending("LOCAL_DEPENDENCY", "READONLY_COMMAND_UNAVAILABLE", str(error)) from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace")[:1024]
        raise ControllerPending(
            "LOCAL_DEPENDENCY",
            "READONLY_COMMAND_FAILED",
            f"{argv[0]} exited {result.returncode}: {detail}",
        )
    return result.stdout


def _git_context(project_root: Path) -> tuple[Path, str]:
    root = Path(
        _run_local(("git", "rev-parse", "--show-toplevel"), cwd=project_root)
        .decode("ascii")
        .strip()
    )
    try:
        prefix = project_root.resolve(strict=True).relative_to(
            root.resolve(strict=True)
        ).as_posix()
    except (OSError, ValueError) as error:
        raise ControllerPending(
            "SOURCE_IDENTITY", "PROJECT_ROOT_NOT_IN_GIT_WORKTREE", str(error)
        ) from error
    return root, "" if prefix == "." else prefix + "/"


def _actual_source_state(contract: Any) -> SourceStateV6:
    project_root = Path(contract.value["fixed_paths"]["project_root"])
    repo_root, prefix = _git_context(project_root)

    def git_identity() -> tuple[str, str]:
        dirty = _run_local(
            ("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"),
            cwd=repo_root,
        )
        allowed_dirty = b""
        fixed = contract.value["fixed_paths"]
        artifact = Path(fixed["artifact_path"])
        if (
            Path(fixed["consumption_path"]).exists()
            and not Path(fixed["terminal_path"]).exists()
            and artifact.exists()
        ):
            try:
                artifact_info = artifact.lstat()
                artifact_relative = artifact.resolve(strict=True).relative_to(
                    repo_root.resolve(strict=True)
                )
            except (OSError, ValueError) as error:
                raise ControllerPending(
                    "SOURCE_IDENTITY",
                    "RECOVERY_ARTIFACT_IDENTITY_DIFFERS",
                    str(error),
                ) from error
            if artifact.is_symlink() or not stat.S_ISREG(artifact_info.st_mode):
                raise ControllerPending(
                    "SOURCE_IDENTITY",
                    "RECOVERY_ARTIFACT_IDENTITY_DIFFERS",
                    str(artifact),
                )
            allowed_dirty = b"?? " + os.fsencode(artifact_relative.as_posix()) + b"\x00"
        if dirty != allowed_dirty:
            raise ControllerPending(
                "SOURCE_IDENTITY",
                "WORKTREE_NOT_EXACTLY_CLEAN",
                "tracked or untracked controller worktree state is present",
            )
        commit = _run_local(("git", "rev-parse", "HEAD"), cwd=repo_root).decode("ascii").strip()
        tree = _run_local(("git", "rev-parse", "HEAD^{tree}"), cwd=repo_root).decode("ascii").strip()
        return commit, tree

    commit_before, tree_before = git_identity()
    rows: list[Mapping[str, object]] = []
    for relative in contract.value["controller_source_closure_relative_paths"]:
        path = project_root / relative
        try:
            info = path.lstat()
            resolved = path.resolve(strict=True)
            payload = path.read_bytes()
        except OSError as error:
            raise ControllerPending(
                "SOURCE_IDENTITY", "CONTROLLER_SOURCE_UNAVAILABLE", f"{relative}: {error}"
            ) from error
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) or resolved != path:
            raise ControllerPending(
                "SOURCE_IDENTITY", "CONTROLLER_SOURCE_IDENTITY_DIFFERS", relative
            )
        committed = _run_local(
            ("git", "show", f"{commit_before}:{prefix}{relative}"), cwd=repo_root
        )
        if payload != committed:
            raise ControllerPending(
                "SOURCE_IDENTITY", "CONTROLLER_SOURCE_BYTES_DIFFER", relative
            )
        rows.append(
            {
                "relative_path": relative,
                "byte_count": len(payload),
                "raw_sha256": campaign.raw_sha256_v6(payload),
            }
        )
    commit_after, tree_after = git_identity()
    if (commit_after, tree_after) != (commit_before, tree_before):
        raise ControllerPending(
            "SOURCE_IDENTITY", "CONTROLLER_SOURCE_CHANGED_DURING_REPLAY", str(project_root)
        )
    root = campaign.source_closure_root_v6(rows)
    return SourceStateV6(
        {
            "source_commit": commit_before,
            "source_tree": tree_before,
            "source_closure_root": root,
        },
        tuple(rows),
    )


def _load_prior_object(
    path: Path,
    label: str,
    parser: Callable[[bytes, str], dict[str, Any]],
) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ControllerPending("PREDECESSOR_REPLAY", "PREDECESSOR_FILE_UNAVAILABLE", f"{label}: {error}") from error
    try:
        return raw, parser(raw, label)
    except Exception as error:
        raise ControllerPending("PREDECESSOR_REPLAY", "PREDECESSOR_FILE_INVALID", f"{label}: {error}") from error


def _actual_replay_predecessor(contract: Any) -> PredecessorReplayV6:
    base = contract.value["base_authoritative_predecessor"]
    failed = contract.value["failed_start_v2_predecessor"]
    failed_v3 = contract.value["failed_start_v3_predecessor"]
    failed_v4 = contract.value["failed_start_v4_predecessor"]
    failed_v5 = contract.value["failed_start_v5_predecessor"]
    project_root = Path(contract.value["fixed_paths"]["project_root"])
    prior_contract_path = project_root / PRIOR_CONTRACT_RELATIVE
    try:
        prior_contract = prestart.load_contract_v1(prior_contract_path)
        capsule_raw, capsule_value = _load_prior_object(
            Path(base["prestart_capsule_path"]),
            "prestart capsule",
            prestart.strict_json_bytes_v1,
        )
        prior_capsule = prestart.validate_capsule_v1(capsule_value, prior_contract)
        if (
            prior_capsule["capsule_digest"] != base["prestart_capsule_digest"]
            or capsule_raw != prestart.canonical_json_bytes_v1(prior_capsule)
        ):
            raise ValueError("prestart capsule binding differs")

        authorization_raw, authorization_value = _load_prior_object(
            Path(base["prestart_authorization_path"]),
            "prestart authorization",
            prestart.strict_json_bytes_v1,
        )
        authorization = prestart._validate_authorization_v1(
            authorization_value, prior_contract, prior_capsule
        )
        if (
            authorization["authorization_digest"]
            != base["prestart_authorization_digest"]
            or authorization_raw != prestart.canonical_json_bytes_v1(authorization)
        ):
            raise ValueError("prestart authorization binding differs")

        consumption_raw, consumption_value = _load_prior_object(
            Path(base["prestart_consumption_path"]),
            "prestart consumption",
            prestart.strict_json_bytes_v1,
        )
        consumption = prestart._validate_consumption_v1(
            consumption_value, prior_contract, prior_capsule
        )
        if (
            consumption["record_digest"]
            != base["prestart_consumption_record_digest"]
            or consumption_raw != prestart.canonical_json_bytes_v1(consumption)
        ):
            raise ValueError("prestart consumption binding differs")

        artifact_raw, artifact_value = _load_prior_object(
            Path(base["prestart_artifact_path"]),
            "prestart artifact",
            prestart.strict_json_bytes_v1,
        )
        if campaign.raw_sha256_v6(artifact_raw) != base["prestart_artifact_raw_sha256"]:
            raise ValueError("prestart artifact raw identity differs")
        artifact = prestart._validate_replayed_artifact_v1(
            artifact_raw,
            artifact_value,
            prior_contract,
            prior_capsule,
            consumption,
        )
        if (
            artifact["source_commit"] != base["formal_source_commit"]
            or artifact["source_tree"] != base["formal_source_tree"]
            or artifact["run_genesis_root"] != base["run_genesis_root"]
            or artifact["prestart_admission_receipt_root"]
            != base["prestart_admission_receipt_root"]
            or artifact["actor_implementation_roots"]
            != base["actor_implementation_roots"]
        ):
            raise ValueError("prestart artifact authority differs")

        v2_contract = campaign_v2.load_contract_v2(
            project_root / FAILED_V2_CONTRACT_RELATIVE
        )
        failed_capsule_raw, failed_capsule_value = _load_prior_object(
            Path(failed["capsule_path"]),
            "failed v2 capsule",
            campaign_v2.strict_json_bytes_v2,
        )
        failed_capsule = campaign_v2.validate_capsule_v2(
            failed_capsule_value, v2_contract
        )
        if (
            failed_capsule["capsule_digest"] != failed["capsule_digest"]
            or campaign.raw_sha256_v6(failed_capsule_raw)
            != failed["capsule_raw_sha256"]
            or failed_capsule_raw
            != campaign_v2.canonical_json_bytes_v2(failed_capsule)
        ):
            raise ValueError("failed v2 capsule binding differs")
        failed_authorization_raw, failed_authorization_value = _load_prior_object(
            Path(failed["authorization_path"]),
            "failed v2 authorization",
            campaign_v2.strict_json_bytes_v2,
        )
        failed_authorization = campaign_v2.validate_authorization_v2(
            failed_authorization_value, v2_contract, failed_capsule
        )
        if (
            failed_authorization["authorization_digest"]
            != failed["authorization_digest"]
            or campaign.raw_sha256_v6(failed_authorization_raw)
            != failed["authorization_raw_sha256"]
            or failed_authorization_raw
            != campaign_v2.canonical_json_bytes_v2(failed_authorization)
        ):
            raise ValueError("failed v2 authorization binding differs")
        failed_consumption_raw, failed_consumption_value = _load_prior_object(
            Path(failed["consumption_path"]),
            "failed v2 consumption",
            campaign_v2.strict_json_bytes_v2,
        )
        failed_consumption = campaign_v2.validate_consumption_v2(
            failed_consumption_value,
            v2_contract,
            failed_capsule,
            failed_authorization,
        )
        if (
            failed_consumption["record_digest"]
            != failed["consumption_record_digest"]
            or campaign.raw_sha256_v6(failed_consumption_raw)
            != failed["consumption_raw_sha256"]
            or failed_consumption_raw
            != campaign_v2.canonical_json_bytes_v2(failed_consumption)
        ):
            raise ValueError("failed v2 consumption binding differs")
        failed_terminal_raw, failed_terminal_value = _load_prior_object(
            Path(failed["terminal_path"]),
            "failed v2 terminal",
            campaign_v2.strict_json_bytes_v2,
        )
        failed_terminal = campaign_v2.validate_terminal_record_v2(
            failed_terminal_value,
            v2_contract,
            failed_capsule,
            failed_authorization,
        )
        if (
            campaign.raw_sha256_v6(failed_terminal_raw)
            != failed["terminal_raw_sha256"]
            or failed_terminal["record_digest"] != failed["terminal_record_digest"]
            or failed_terminal["artifact_raw_sha256"] is not None
        ):
            raise ValueError("failed v2 terminal binding differs")

        v3_contract = campaign_v3.load_contract_v3(
            project_root / FAILED_V3_CONTRACT_RELATIVE
        )
        failed_v3_capsule_raw, failed_v3_capsule_value = _load_prior_object(
            Path(failed_v3["capsule_path"]),
            "failed v3 capsule",
            campaign_v3.strict_json_bytes_v3,
        )
        failed_v3_capsule = campaign_v3.validate_capsule_v3(
            failed_v3_capsule_value, v3_contract
        )
        if (
            failed_v3_capsule["capsule_digest"] != failed_v3["capsule_digest"]
            or campaign.raw_sha256_v6(failed_v3_capsule_raw)
            != failed_v3["capsule_raw_sha256"]
            or failed_v3_capsule_raw
            != campaign_v3.canonical_json_bytes_v3(failed_v3_capsule)
        ):
            raise ValueError("failed v3 capsule binding differs")
        failed_v3_authorization_raw, failed_v3_authorization_value = (
            _load_prior_object(
                Path(failed_v3["authorization_path"]),
                "failed v3 authorization",
                campaign_v3.strict_json_bytes_v3,
            )
        )
        failed_v3_authorization = campaign_v3.validate_authorization_v3(
            failed_v3_authorization_value,
            v3_contract,
            failed_v3_capsule,
        )
        if (
            failed_v3_authorization["authorization_digest"]
            != failed_v3["authorization_digest"]
            or campaign.raw_sha256_v6(failed_v3_authorization_raw)
            != failed_v3["authorization_raw_sha256"]
            or failed_v3_authorization_raw
            != campaign_v3.canonical_json_bytes_v3(failed_v3_authorization)
        ):
            raise ValueError("failed v3 authorization binding differs")
        failed_v3_consumption_raw, failed_v3_consumption_value = (
            _load_prior_object(
                Path(failed_v3["consumption_path"]),
                "failed v3 consumption",
                campaign_v3.strict_json_bytes_v3,
            )
        )
        failed_v3_consumption = campaign_v3.validate_consumption_v3(
            failed_v3_consumption_value,
            v3_contract,
            failed_v3_capsule,
            failed_v3_authorization,
        )
        if (
            failed_v3_consumption["record_digest"]
            != failed_v3["consumption_record_digest"]
            or campaign.raw_sha256_v6(failed_v3_consumption_raw)
            != failed_v3["consumption_raw_sha256"]
            or failed_v3_consumption_raw
            != campaign_v3.canonical_json_bytes_v3(failed_v3_consumption)
        ):
            raise ValueError("failed v3 consumption binding differs")
        failed_v3_terminal_raw, failed_v3_terminal_value = _load_prior_object(
            Path(failed_v3["terminal_path"]),
            "failed v3 terminal",
            campaign_v3.strict_json_bytes_v3,
        )
        failed_v3_terminal = campaign_v3.validate_terminal_record_v3(
            failed_v3_terminal_value,
            v3_contract,
            failed_v3_capsule,
            failed_v3_authorization,
        )
        if (
            campaign.raw_sha256_v6(failed_v3_terminal_raw)
            != failed_v3["terminal_raw_sha256"]
            or failed_v3_terminal["record_digest"]
            != failed_v3["terminal_record_digest"]
            or failed_v3_terminal["artifact_raw_sha256"] is not None
            or failed_v3_terminal["status"]
            != "Q1_START_V3_INCONCLUSIVE_TERMINAL_NO_RESUME"
            or failed_v3_terminal["outcome"]["actor_start_count_by_actor"]
            != {"host": 0, "python": 0, "rust": 0}
        ):
            raise ValueError("failed v3 terminal binding differs")

        v4_contract = campaign_v4.load_contract_v4(
            project_root / FAILED_V4_CONTRACT_RELATIVE
        )
        failed_v4_capsule_raw, failed_v4_capsule_value = _load_prior_object(
            Path(failed_v4["capsule_path"]),
            "failed v4 capsule",
            campaign_v4.strict_json_bytes_v4,
        )
        failed_v4_capsule = campaign_v4.validate_capsule_v4(
            failed_v4_capsule_value, v4_contract
        )
        if (
            failed_v4_capsule["capsule_digest"] != failed_v4["capsule_digest"]
            or campaign.raw_sha256_v6(failed_v4_capsule_raw)
            != failed_v4["capsule_raw_sha256"]
            or failed_v4_capsule_raw
            != campaign_v4.canonical_json_bytes_v4(failed_v4_capsule)
        ):
            raise ValueError("failed v4 capsule binding differs")
        failed_v4_authorization_raw, failed_v4_authorization_value = (
            _load_prior_object(
                Path(failed_v4["authorization_path"]),
                "failed v4 authorization",
                campaign_v4.strict_json_bytes_v4,
            )
        )
        failed_v4_authorization = campaign_v4.validate_authorization_v4(
            failed_v4_authorization_value,
            v4_contract,
            failed_v4_capsule,
        )
        if (
            failed_v4_authorization["authorization_digest"]
            != failed_v4["authorization_digest"]
            or campaign.raw_sha256_v6(failed_v4_authorization_raw)
            != failed_v4["authorization_raw_sha256"]
            or failed_v4_authorization_raw
            != campaign_v4.canonical_json_bytes_v4(failed_v4_authorization)
        ):
            raise ValueError("failed v4 authorization binding differs")
        failed_v4_consumption_raw, failed_v4_consumption_value = (
            _load_prior_object(
                Path(failed_v4["consumption_path"]),
                "failed v4 consumption",
                campaign_v4.strict_json_bytes_v4,
            )
        )
        failed_v4_consumption = campaign_v4.validate_consumption_v4(
            failed_v4_consumption_value,
            v4_contract,
            failed_v4_capsule,
            failed_v4_authorization,
        )
        if (
            failed_v4_consumption["record_digest"]
            != failed_v4["consumption_record_digest"]
            or campaign.raw_sha256_v6(failed_v4_consumption_raw)
            != failed_v4["consumption_raw_sha256"]
            or failed_v4_consumption_raw
            != campaign_v4.canonical_json_bytes_v4(failed_v4_consumption)
        ):
            raise ValueError("failed v4 consumption binding differs")
        failed_v4_terminal_raw, failed_v4_terminal_value = _load_prior_object(
            Path(failed_v4["terminal_path"]),
            "failed v4 terminal",
            campaign_v4.strict_json_bytes_v4,
        )
        failed_v4_terminal = campaign_v4.validate_terminal_record_v4(
            failed_v4_terminal_value,
            v4_contract,
            failed_v4_capsule,
            failed_v4_authorization,
        )
        failed_v4_outcome = failed_v4_terminal["outcome"]
        if (
            campaign.raw_sha256_v6(failed_v4_terminal_raw)
            != failed_v4["terminal_raw_sha256"]
            or failed_v4_terminal["record_digest"]
            != failed_v4["terminal_record_digest"]
            or failed_v4_terminal["artifact_raw_sha256"] is not None
            or failed_v4_terminal["status"]
            != "Q1_START_V4_INCONCLUSIVE_TERMINAL_NO_RESUME"
            or failed_v4_outcome["actor_start_count_by_actor"]
            != {"host": 0, "python": 1, "rust": 0}
            or failed_v4_outcome["failure_code"]
            != "Q1_START_ENDPOINT_REPLAY_MISMATCH"
            or failed_v4_outcome["failure_detail"]
            != (
                "python:Q1FormalHostReplayError:"
                "REJECT_Q1_FORMAL_ENDPOINT_RESULT: "
                "same_run_record_set_root differs from replay"
            )
            or failed_v4_outcome["q1_state"] != "NOT_RUN"
            or failed_v4_outcome["q1_gate_count"] != 14
            or failed_v4_outcome["q1_gate_mask"] != 0x003FFF
            or failed_v4_outcome["q1_output_slots"] != [None] * 8
            or failed_v4_outcome["network_accessed"] is not False
            or failed_v4_outcome["target_truth_accessed"] is not False
            or failed_v4_outcome["split_accessed"] is not False
            or failed_v4_outcome["role_evaluation_performed"] is not False
        ):
            raise ValueError("failed v4 terminal binding differs")

        v5_contract = campaign_v5.load_contract_v5(
            project_root / FAILED_V5_CONTRACT_RELATIVE
        )
        failed_v5_capsule_raw, failed_v5_capsule_value = _load_prior_object(
            Path(failed_v5["capsule_path"]),
            "failed v5 capsule",
            campaign_v5.strict_json_bytes_v5,
        )
        failed_v5_capsule = campaign_v5.validate_capsule_v5(
            failed_v5_capsule_value, v5_contract
        )
        if (
            failed_v5_capsule["capsule_digest"] != failed_v5["capsule_digest"]
            or campaign.raw_sha256_v6(failed_v5_capsule_raw)
            != failed_v5["capsule_raw_sha256"]
            or failed_v5_capsule_raw
            != campaign_v5.canonical_json_bytes_v5(failed_v5_capsule)
        ):
            raise ValueError("failed v5 capsule binding differs")
        failed_v5_authorization_raw, failed_v5_authorization_value = (
            _load_prior_object(
                Path(failed_v5["authorization_path"]),
                "failed v5 authorization",
                campaign_v5.strict_json_bytes_v5,
            )
        )
        failed_v5_authorization = campaign_v5.validate_authorization_v5(
            failed_v5_authorization_value,
            v5_contract,
            failed_v5_capsule,
        )
        if (
            failed_v5_authorization["authorization_digest"]
            != failed_v5["authorization_digest"]
            or campaign.raw_sha256_v6(failed_v5_authorization_raw)
            != failed_v5["authorization_raw_sha256"]
            or failed_v5_authorization_raw
            != campaign_v5.canonical_json_bytes_v5(failed_v5_authorization)
        ):
            raise ValueError("failed v5 authorization binding differs")
        failed_v5_consumption_raw, failed_v5_consumption_value = (
            _load_prior_object(
                Path(failed_v5["consumption_path"]),
                "failed v5 consumption",
                campaign_v5.strict_json_bytes_v5,
            )
        )
        failed_v5_consumption = campaign_v5.validate_consumption_v5(
            failed_v5_consumption_value,
            v5_contract,
            failed_v5_capsule,
            failed_v5_authorization,
        )
        if (
            failed_v5_consumption["record_digest"]
            != failed_v5["consumption_record_digest"]
            or campaign.raw_sha256_v6(failed_v5_consumption_raw)
            != failed_v5["consumption_raw_sha256"]
            or failed_v5_consumption_raw
            != campaign_v5.canonical_json_bytes_v5(failed_v5_consumption)
        ):
            raise ValueError("failed v5 consumption binding differs")
        failed_v5_terminal_raw, failed_v5_terminal_value = _load_prior_object(
            Path(failed_v5["terminal_path"]),
            "failed v5 terminal",
            campaign_v5.strict_json_bytes_v5,
        )
        failed_v5_terminal = campaign_v5.validate_terminal_record_v5(
            failed_v5_terminal_value,
            v5_contract,
            failed_v5_capsule,
            failed_v5_authorization,
        )
        failed_v5_outcome = failed_v5_terminal["outcome"]
        if (
            campaign.raw_sha256_v6(failed_v5_terminal_raw)
            != failed_v5["terminal_raw_sha256"]
            or failed_v5_terminal["record_digest"]
            != failed_v5["terminal_record_digest"]
            or failed_v5_terminal["artifact_raw_sha256"] is not None
            or failed_v5_terminal["status"]
            != "Q1_START_V5_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME"
            or failed_v5_outcome["actor_start_count_by_actor"]
            != {"host": 0, "python": 1, "rust": 0}
            or failed_v5_outcome["failure_code"] != failed_v5["failure_code"]
            or failed_v5_outcome["failure_detail"] != failed_v5["failure_detail"]
            or failed_v5_outcome["q1_state"] != "NOT_RUN"
            or failed_v5_outcome["q1_gate_count"] != 14
            or failed_v5_outcome["q1_gate_mask"] != 0x003FFF
            or failed_v5_outcome["q1_output_slots"] != [None] * 8
            or failed_v5_outcome["network_accessed"] is not False
            or failed_v5_outcome["target_truth_accessed"] is not False
            or failed_v5_outcome["split_accessed"] is not False
            or failed_v5_outcome["role_evaluation_performed"] is not False
        ):
            raise ValueError("failed v5 terminal binding differs")

    except ControllerPending:
        raise
    except Exception as error:
        raise ControllerPending(
            "PREDECESSOR_REPLAY", "PREDECESSOR_AUTHORITY_DIFFERS", str(error)
        ) from error
    return PredecessorReplayV6(
        prior_capsule,
        authorization,
        consumption,
        artifact,
        failed_capsule,
        failed_authorization,
        failed_consumption,
        failed_terminal,
        failed_v3_capsule,
        failed_v3_authorization,
        failed_v3_consumption,
        failed_v3_terminal,
        failed_v4_capsule,
        failed_v4_authorization,
        failed_v4_consumption,
        failed_v4_terminal,
        failed_v5_capsule,
        failed_v5_authorization,
        failed_v5_consumption,
        failed_v5_terminal,
        {},
    )


def _actual_formal_blob(project_root: Path, commit: str, relative: str) -> bytes:
    repo_root, prefix = _git_context(project_root)
    return _run_local(("git", "show", f"{commit}:{prefix}{relative}"), cwd=repo_root)


def prepare_capsule_inputs_v6(
    contract: Any,
    run_id: str,
    created_at_unix_seconds: int,
    rust_binary_path: Path,
) -> dict[str, object]:
    """Build the fresh successor bundle and capsule inputs from committed bytes."""

    source_before = _actual_source_state(contract)
    project_root = Path(contract.value["fixed_paths"]["project_root"])
    commit = str(source_before.identity["source_commit"])
    predecessor = _actual_replay_predecessor(contract)
    manifests: dict[str, object] = {}
    descriptors: dict[str, object] = {}
    mount_rows: list[dict[str, object]] = []
    for actor in _ACTOR_ORDER:
        rows: list[dict[str, object]] = []
        for relative in _ACTOR_SOURCE_RELATIVE_PATHS[actor]:
            committed = _actual_formal_blob(project_root, commit, relative)
            current = _regular_file(
                project_root / relative,
                f"current {actor} source {relative}",
            )
            if current != committed:
                raise ControllerPending(
                    "SOURCE_IDENTITY", "ACTOR_SOURCE_BYTES_NOT_COMMITTED", relative
                )
            rows.append(
                {
                    "relative_path": relative,
                    "byte_count": len(current),
                    "raw_sha256": campaign.raw_sha256_v6(current),
                }
            )
        manifest_root = successor.actor_source_manifest_root_v6(actor, rows)
        manifests[actor] = {
            "source_manifest_root": manifest_root,
            "rows": rows,
        }
        entrypoint = successor.ACTOR_ENTRYPOINT_RELATIVE_PATHS[actor]
        entrypoint_rows = [row for row in rows if row["relative_path"] == entrypoint]
        if len(entrypoint_rows) != 1:
            raise ControllerPending(
                "SOURCE_IDENTITY", "ACTOR_ENTRYPOINT_NOT_IN_MANIFEST", actor
            )
        implementation_id, action_id = successor.ACTOR_IMPLEMENTATION_REGISTRY[
            actor
        ]
        descriptors[actor] = {
            "implementation_id": implementation_id,
            "action_id": action_id,
            "entrypoint_relative_path": entrypoint,
            "entrypoint_raw_sha256": entrypoint_rows[0]["raw_sha256"],
            **dict(successor.ACTOR_BOUND_ENVIRONMENT_ROOTS[actor]),
        }
        mount_rows.append(
            {
                "actor_id": actor,
                "read_only_source_relative_paths": [
                    row["relative_path"] for row in rows
                ],
                "target_truth_mounted": False,
                "split_mounted": False,
                "role_mounted": False,
            }
        )
    base_path = Path(
        contract.value["base_authoritative_predecessor"]["prestart_artifact_path"]
    )
    failed_path = Path(contract.value["failed_start_v5_predecessor"]["terminal_path"])
    base_payload = _regular_file(base_path, "base authoritative artifact")
    failed_payload = _regular_file(failed_path, "failed Q1-start-v5 terminal")
    bundle = successor.derive_q1_prestart_successor_candidate_v6(
        base_payload,
        failed_payload,
        v6_scratch_observations=dict(successor.EXPECTED_V6_SCRATCH_OBSERVATIONS),
        corrected_resource_envelope=dict(
            successor.EXPECTED_CORRECTED_RESOURCE_ENVELOPE
        ),
        source_identity=dict(source_before.identity),
        actor_source_manifests=manifests,
        actor_implementation_descriptors=descriptors,
        mount_manifest={
            "schema_version": successor.MOUNT_MANIFEST_SCHEMA_VERSION,
            "rows": mount_rows,
        },
        input_absence_manifest={
            "schema_version": successor.INPUT_ABSENCE_SCHEMA_VERSION,
            "formal_input_keys": [],
            "target_truth_present": False,
            "split_present": False,
            "role_present": False,
        },
        operator_owned_output_freezer_compatibility_descriptor=dict(
            successor.EXPECTED_OUTPUT_FREEZER_COMPATIBILITY
        ),
        formal_output_layout_root=successor.FORMAL_OUTPUT_LAYOUT_ROOT,
        run_id=bytes.fromhex(run_id),
        created_at_unix_seconds=created_at_unix_seconds,
    )
    binary = _regular_file(rust_binary_path, "fresh Rust formal binary", executable=True)
    if rust_binary_path != Path(contract.value["fixed_paths"]["rust_binary_path"]):
        raise ControllerPending(
            "LOCAL_BINDINGS", "RUST_BINARY_PATH_DIFFERS", str(rust_binary_path)
        )
    source_after = _actual_source_state(contract)
    if source_after != source_before:
        raise ControllerPending(
            "SOURCE_IDENTITY", "SOURCE_CHANGED_DURING_CAPSULE_PREPARATION", commit
        )
    inputs = {
        "schema_version": campaign.CAPSULE_INPUTS_SCHEMA_VERSION,
        "controller_source_identity": dict(source_before.identity),
        "controller_source_closure_rows": [
            dict(row) for row in source_before.rows
        ],
        "actor_source_manifests": manifests,
        "rust_binary": {
            "path": str(rust_binary_path),
            "byte_count": len(binary),
            "raw_sha256": campaign.raw_sha256_v6(binary),
        },
        "successor_prestart_bundle": bundle,
        "run_id": run_id,
        "created_at_unix_seconds": created_at_unix_seconds,
    }
    campaign.build_capsule_v6(contract, inputs)
    return inputs


def _parse_cpuset(value: str) -> frozenset[int]:
    result: set[int] = set()
    for item in value.split(","):
        if "-" in item:
            left, right = (int(part) for part in item.split("-", 1))
            result.update(range(left, right + 1))
        else:
            result.add(int(item))
    return frozenset(result)


def _regular_file(path: Path, label: str, *, executable: bool = False) -> bytes:
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
        payload = path.read_bytes()
    except OSError as error:
        raise ControllerPending("LOCAL_BINDINGS", "BOUND_FILE_UNAVAILABLE", f"{label}: {error}") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or resolved != path:
        raise ControllerPending("LOCAL_BINDINGS", "BOUND_FILE_IDENTITY_DIFFERS", label)
    if executable and info.st_mode & 0o111 == 0:
        raise ControllerPending("LOCAL_BINDINGS", "BOUND_FILE_NOT_EXECUTABLE", label)
    return payload


def _actual_live_gate(contract: Any, capsule: Mapping[str, object]) -> None:
    runtime = contract.value["runtime"]
    fixed = contract.value["fixed_paths"]
    actual_uid = os.geteuid()
    actual_gid = os.getegid()
    if (
        actual_uid != runtime["actor_uid"]
        or actual_gid != runtime["actor_gid"]
    ):
        raise ControllerPending(
            "LOCAL_BINDINGS",
            "ACTOR_OUTPUT_OWNER_IDENTITY_DIFFERS",
            (
                f"expected={runtime['actor_uid']}:{runtime['actor_gid']} "
                f"actual={actual_uid}:{actual_gid}"
            ),
        )
    binary_path = Path(fixed["rust_binary_path"])
    binary = _regular_file(binary_path, "Rust formal binary", executable=True)
    try:
        binary_mode = stat.S_IMODE(binary_path.lstat().st_mode)
    except OSError as error:
        raise ControllerPending(
            "LOCAL_BINDINGS", "RUST_BINARY_BINDING_DIFFERS", str(error)
        ) from error
    if binary_mode & 0o005 != 0o005:
        raise ControllerPending(
            "LOCAL_BINDINGS",
            "RUST_BINARY_CONTAINER_MODE_DIFFERS",
            f"{binary_path}: {binary_mode:04o}",
        )
    if (
        len(binary) != capsule["rust_binary"]["byte_count"]
        or campaign.raw_sha256_v6(binary) != capsule["rust_binary"]["raw_sha256"]
    ):
        raise ControllerPending("LOCAL_BINDINGS", "RUST_BINARY_BINDING_DIFFERS", str(fixed["rust_binary_path"]))
    seccomp = _regular_file(Path(runtime["seccomp_profile_path"]), "seccomp profile")
    if campaign.raw_sha256_v6(seccomp) != runtime["seccomp_profile_raw_sha256"]:
        raise ControllerPending("LOCAL_BINDINGS", "SECCOMP_BINDING_DIFFERS", str(runtime["seccomp_profile_path"]))
    try:
        parsed = json.loads(seccomp.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControllerPending("LOCAL_BINDINGS", "SECCOMP_JSON_INVALID", str(error)) from error
    if type(parsed) is not dict:
        raise ControllerPending("LOCAL_BINDINGS", "SECCOMP_JSON_INVALID", "profile is not an object")

    docker = Path(runtime["docker_path"])
    _regular_file(docker, "Docker CLI", executable=True)
    for image in (runtime["python_image"], runtime["rust_image"]):
        try:
            result = subprocess.run(
                [str(docker), "image", "inspect", image],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ControllerPending("LOCAL_BINDINGS", "LOCAL_IMAGE_INSPECT_UNAVAILABLE", str(error)) from error
        if result.returncode != 0:
            raise ControllerPending(
                "LOCAL_BINDINGS",
                "PINNED_LOCAL_IMAGE_UNAVAILABLE",
                f"{image}: {result.stderr.decode('utf-8', 'replace')[:512]}",
            )
    try:
        effective = frozenset(os.sched_getaffinity(0))
    except (AttributeError, OSError) as error:
        raise ControllerPending("LOCAL_CAPACITY", "EFFECTIVE_CPU_SET_UNAVAILABLE", str(error)) from error
    required = frozenset().union(
        *(
            _parse_cpuset(runtime[name])
            for name in ("python_cpuset", "rust_cpuset", "host_cpuset")
        )
    )
    if not required.issubset(effective):
        raise ControllerPending(
            "LOCAL_CAPACITY", "REQUIRED_EFFECTIVE_CPU_SET_UNAVAILABLE", f"required={sorted(required)} effective={sorted(effective)}"
        )
    try:
        available_kib = next(
            int(line.split()[1])
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
            if line.startswith("MemAvailable:")
        )
    except (OSError, StopIteration, ValueError, IndexError) as error:
        raise ControllerPending("LOCAL_CAPACITY", "AVAILABLE_MEMORY_UNAVAILABLE", str(error)) from error
    available = available_kib * 1024
    if available < runtime["minimum_available_memory_bytes"]:
        raise ControllerPending("LOCAL_CAPACITY", "INSUFFICIENT_AVAILABLE_MEMORY", str(available))
    checked_devices: set[int] = set()
    for path_text in (fixed["run_root"], fixed["ledger_root"], str(Path(fixed["artifact_path"]).parent)):
        path = Path(path_text)
        try:
            info = path.stat()
            free = shutil.disk_usage(path).free
        except OSError as error:
            raise ControllerPending("LOCAL_CAPACITY", "AVAILABLE_STORAGE_UNAVAILABLE", f"{path}: {error}") from error
        if info.st_dev in checked_devices:
            continue
        checked_devices.add(info.st_dev)
        if free < runtime["minimum_available_storage_bytes"]:
            raise ControllerPending("LOCAL_CAPACITY", "INSUFFICIENT_AVAILABLE_STORAGE", f"{path}: {free}")


def _load_supervisor_module() -> Any:
    name = "_hegel_phase3_q1_start_supervisor_v6"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = PROJECT_ROOT / SUPERVISOR_RELATIVE
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ControllerPending("LOCAL_DEPENDENCY", "SUPERVISOR_MODULE_UNAVAILABLE", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(name, None)
        raise ControllerPending("LOCAL_DEPENDENCY", "SUPERVISOR_MODULE_UNAVAILABLE", str(error)) from error
    return module


_DIAGNOSTIC_BOUND: Final = 65_536
_STARTUP_SMOKE_FAILURE_STATUS: Final = (
    "Q1_START_V6_AUTHORITY_STARTUP_SMOKE_FAILED"
)
_STARTUP_SMOKE_FAILURE_KEYS: Final = frozenset(
    {
        "schema_version",
        "status",
        "error_code",
        "error_detail",
        "campaign_supervisor_popen_count",
        "actor_popen_count",
        "checkpoint_or_mutable_roots_created",
        "ownership_probe_run_popen_count",
        "ownership_probe_cleanup_popen_count",
        "ownership_probe_absence_check_popen_count",
        "ownership_probe_container_absence_proven",
        "ownership_probe_temp_root_absence_proven",
        "docker_invoked",
    }
)


def _bounded_diagnostic(payload: bytes) -> str:
    """Return the complete diagnostic when bounded, otherwise preserve both ends."""

    if len(payload) <= _DIAGNOSTIC_BOUND:
        return payload.decode("utf-8", "replace")
    half = _DIAGNOSTIC_BOUND // 2
    head = payload[:half].decode("utf-8", "replace")
    tail = payload[-half:].decode("utf-8", "replace")
    return f"{head}\n...<{len(payload) - 2 * half} diagnostic bytes omitted>...\n{tail}"


def _validate_startup_smoke_failure_receipt(
    receipt: Mapping[str, object],
) -> None:
    """Validate the isolated supervisor's exact pre-consumption failure audit."""

    supervisor = _load_supervisor_module()
    run_count = receipt.get("ownership_probe_run_popen_count")
    cleanup_count = receipt.get("ownership_probe_cleanup_popen_count")
    absence_count = receipt.get("ownership_probe_absence_check_popen_count")
    container_absent = receipt.get("ownership_probe_container_absence_proven")
    temp_root_absent = receipt.get("ownership_probe_temp_root_absence_proven")
    docker_invoked = receipt.get("docker_invoked")
    if (
        type(receipt) is not dict
        or set(receipt) != _STARTUP_SMOKE_FAILURE_KEYS
        or receipt.get("schema_version") != supervisor.AUTHORITY_SMOKE_SCHEMA
        or receipt.get("status") != _STARTUP_SMOKE_FAILURE_STATUS
        or type(receipt.get("error_code")) is not str
        or not receipt.get("error_code")
        or type(receipt.get("error_detail")) is not str
        or receipt.get("campaign_supervisor_popen_count") != 0
        or receipt.get("actor_popen_count") != 0
        or receipt.get("checkpoint_or_mutable_roots_created") is not False
        or type(run_count) is not int
        or run_count not in (0, 1)
        or type(cleanup_count) is not int
        or cleanup_count not in (0, 1)
        or type(absence_count) is not int
        or absence_count not in (0, 1)
        or cleanup_count != absence_count
        or cleanup_count > run_count
        or type(container_absent) is not bool
        or type(temp_root_absent) is not bool
        or type(docker_invoked) is not bool
        or docker_invoked != (run_count == 1)
        or (cleanup_count == 0 and container_absent is not True)
    ):
        raise ControllerPending(
            "SUPERVISOR_STARTUP_SMOKE",
            "AUTHORITY_STARTUP_SMOKE_FAILURE_RECEIPT_DIFFERS",
            repr(receipt),
        )


def _actual_startup_smoke(
    contract: Any,
    run_spec: Mapping[str, object],
) -> Mapping[str, object]:
    """Run the exact supervisor entry in a fresh isolated interpreter.

    The v3 failure was hidden because the same-process loader had already
    imported ``importlib.util``.  This gate therefore executes the bound script
    directly with the same interpreter flags as the later campaign supervisor,
    while selecting its validation-only startup-smoke mode.
    """

    python_path = Path(contract.value["runtime"]["supervisor_python_path"])
    supervisor_path = PROJECT_ROOT / SUPERVISOR_RELATIVE
    run_spec_path = Path(contract.value["fixed_paths"]["run_spec_path"])
    expected_payload = campaign.canonical_json_bytes_v6(run_spec)
    observed_payload = _regular_file(run_spec_path, "Q1 start v6 run spec")
    if observed_payload != expected_payload:
        raise ControllerPending(
            "SUPERVISOR_STARTUP_SMOKE",
            "BOUND_RUN_SPEC_DIFFERS",
            str(run_spec_path),
        )
    _regular_file(python_path, "bound supervisor Python", executable=True)
    _regular_file(supervisor_path, "bound v6 supervisor source")
    try:
        process = subprocess.run(
            [
                str(python_path),
                "-I",
                "-S",
                "-B",
                str(supervisor_path),
                "--startup-smoke",
                "--run-spec",
                str(run_spec_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            # The supervisor owns the probe's bounded Docker run, cleanup, and
            # exact absence check.  An outer timeout could kill the supervisor
            # between those steps and leave container absence unproved.
            timeout=None,
        )
    except OSError as error:
        raise ControllerPending(
            "SUPERVISOR_STARTUP_SMOKE",
            "AUTHORITY_STARTUP_SMOKE_PROCESS_UNAVAILABLE",
            str(error),
        ) from error
    if (
        process.returncode not in (0, 1)
        or not process.stdout
        or len(process.stdout) > campaign.MAX_SMALL_JSON
    ):
        raise ControllerPending(
            "SUPERVISOR_STARTUP_SMOKE",
            "AUTHORITY_STARTUP_SMOKE_PROCESS_FAILED",
            (
                f"exit={process.returncode}; "
                f"stdout={_bounded_diagnostic(process.stdout)}; "
                f"stderr={_bounded_diagnostic(process.stderr)}"
            ),
        )
    try:
        result = campaign.strict_json_bytes_v6(
            process.stdout,
            "isolated supervisor startup-smoke receipt",
        )
    except Exception as error:
        raise ControllerPending(
            "SUPERVISOR_STARTUP_SMOKE",
            "AUTHORITY_STARTUP_SMOKE_RESULT_INVALID",
            (
                f"stdout={_bounded_diagnostic(process.stdout)}; "
                f"stderr={_bounded_diagnostic(process.stderr)}; parse={error}"
            ),
        ) from error
    if process.returncode == 1:
        _validate_startup_smoke_failure_receipt(result)
        raise ControllerPending(
            "SUPERVISOR_STARTUP_SMOKE",
            "AUTHORITY_STARTUP_SMOKE_FAILED",
            process.stdout.decode("ascii"),
        )
    return result


def _actual_invoke_supervisor(
    python_path: Path,
    supervisor_path: Path,
    run_spec_path: Path,
) -> Mapping[str, object]:
    try:
        result = subprocess.run(
            [str(python_path), "-I", "-S", "-B", str(supervisor_path), "--run-spec", str(run_spec_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=None,
        )
    except OSError as error:
        raise ControllerPending(
            "SUPERVISOR_START",
            "SUPERVISOR_POPEN_UNAVAILABLE",
            str(error),
        ) from error
    if len(result.stdout) > campaign.MAX_SMALL_JSON:
        _terminal("Q1_START_SUPERVISOR_OUTCOME_UNKNOWN", "supervisor stdout exceeds bound")
    try:
        outcome = campaign.strict_json_bytes_v6(result.stdout, "supervisor outcome")
    except Exception as error:
        _terminal(
            "Q1_START_SUPERVISOR_OUTCOME_UNKNOWN",
            (
                f"exit={result.returncode}; stdout={_bounded_diagnostic(result.stdout)}; "
                f"stderr={_bounded_diagnostic(result.stderr)}; parse={error}"
            ),
        )
    expected_codes = {0, 1, PENDING_EXIT_CODE}
    if result.returncode not in expected_codes:
        _terminal(
            "Q1_START_SUPERVISOR_OUTCOME_UNKNOWN",
            (
                f"unexpected exit {result.returncode}; "
                f"stderr={_bounded_diagnostic(result.stderr)}"
            ),
        )
    return outcome


def _actual_build_completion_artifact(
    supervisor_outcome: Mapping[str, object],
    contract: Any,
    capsule: Mapping[str, object],
    run_spec: Mapping[str, object],
) -> bytes:
    """Adapt only the supervisor's persisted files and frozen receipt inputs."""

    supervisor = _load_supervisor_module()

    def root(value: object, label: str) -> bytes:
        if type(value) is not str or _TAGGED.fullmatch(value) is None:
            _terminal("Q1_START_COMPLETION_ADAPTER_INVALID", f"{label} root differs")
        return bytes.fromhex(value.removeprefix("sha256:"))

    def files(root_path: Path) -> tuple[Any, ...]:
        try:
            rows = []
            for path in root_path.rglob("*"):
                info = path.lstat()
                if path.is_symlink() or not (
                    stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)
                ):
                    raise OSError(f"unsupported output entry {path}")
                if stat.S_ISREG(info.st_mode):
                    rows.append(
                        completion.Q1FileEvidenceV1(
                            path.relative_to(root_path).as_posix(), path.read_bytes()
                        )
                    )
        except OSError as error:
            _terminal("Q1_START_COMPLETION_OUTPUT_UNAVAILABLE", str(error))
        return tuple(sorted(rows, key=lambda row: row.relative_path))

    gate16 = supervisor_outcome.get("gate16_candidate")
    gate18 = supervisor_outcome.get("gate18_runtime_receipt_inputs")
    if type(gate16) is not dict or type(gate18) is not dict:
        _terminal("Q1_START_COMPLETION_ADAPTER_INVALID", "Gate 16/18 inputs differ")
    signature_preimages = gate16.get("signature_state_preimages")
    if type(signature_preimages) is not list or len(signature_preimages) != 2:
        _terminal("Q1_START_COMPLETION_ADAPTER_INVALID", "signature preimages differ")
    signatures = tuple(
        completion.Q1SignatureCompletionRowV1(
            row["input_signature_id"],
            root(row["fixed_point_record_root"], "fixed point"),
            root(row["signature_manifest_root"], "signature manifest"),
            root(row["saturation_state_root"], "saturation state"),
            completion.signature_guard_ceiling_rows_v1(row["input_signature_id"]),
        )
        for row in signature_preimages
        if type(row) is dict
    )
    if len(signatures) != 2:
        _terminal("Q1_START_COMPLETION_ADAPTER_INVALID", "signature rows differ")
    run_genesis = root(run_spec["run_genesis_root"], "run genesis")
    prestart_receipt = root(
        run_spec["prestart_admission_receipt_root"], "prestart receipt"
    )
    host_result = supervisor_outcome.get("host_result")
    if type(host_result) is not dict:
        _terminal("Q1_START_COMPLETION_ADAPTER_INVALID", "host result differs")
    closure_bundle = root(host_result.get("closure_bundle_root"), "closure bundle")
    binding_kwargs = {
        "expected_capsule_digest": run_spec["start_capsule_digest"],
        "expected_source_commit": run_spec["source_commit"],
        "expected_run_genesis_root": run_genesis,
        "expected_prestart_admission_receipt_root": prestart_receipt,
        "expected_closure_bundle_root": closure_bundle,
        "expected_signature_rows": signatures,
    }
    stage_rows_value = gate18.get("stage_rows")
    if type(stage_rows_value) is not list or len(stage_rows_value) != 3:
        _terminal("Q1_START_COMPLETION_ADAPTER_INVALID", "Gate 18 stage rows differ")
    stage_rows = {row.get("stage"): row for row in stage_rows_value if type(row) is dict}
    if tuple(row.get("stage") for row in stage_rows_value if type(row) is dict) != _ACTOR_ORDER:
        _terminal("Q1_START_COMPLETION_ADAPTER_INVALID", "Gate 18 actor order differs")

    output_files: dict[str, tuple[Any, ...]] = {}
    isolation_adapters: dict[str, Any] = {}
    for stage in _ACTOR_ORDER:
        stage_row = stage_rows[stage]
        observed_files = files(Path(run_spec[f"{stage}_output"]))
        output_files[stage] = observed_files
        source_rows = tuple(
            completion.Q1SourceManifestEvidenceRowV1(
                row["relative_path"],
                row["byte_count"],
                root(row["raw_sha256"], f"{stage} source"),
            )
            for row in run_spec["actor_source_manifests"][stage]
        )
        runtime_evidence = stage_row.get("runtime_evidence")
        if type(runtime_evidence) is not dict or type(runtime_evidence.get("mounts")) is not list:
            _terminal("Q1_START_COMPLETION_ADAPTER_INVALID", f"{stage} runtime evidence differs")
        mount_rows = tuple(
            completion.Q1MountManifestEvidenceRowV1(
                row["destination"], row["source"], row["read_write"]
            )
            for row in runtime_evidence["mounts"]
            if type(row) is dict
        )
        adapter = completion.Q1IsolationEvidenceAdapterV1(
            actor_id=stage.encode("ascii"),
            actor_implementation_root=root(
                run_spec["actor_implementation_roots"][stage],
                f"{stage} implementation",
            ),
            source_manifest_rows=source_rows,
            mount_manifest_rows=mount_rows,
            output_files=observed_files,
            scratch_path=run_spec[f"{stage}_scratch"],
        )
        source_replay = stage_row.get("source_manifest_replay")
        v3_source_root = supervisor._object_root(
            supervisor._SOURCE_CLOSURE_DOMAIN,
            run_spec["actor_source_manifests"][stage],
        )
        v3_mount_root = supervisor._object_root(
            supervisor._MOUNT_REGISTRY_DOMAIN,
            runtime_evidence["mounts"],
        )
        v3_output_root = supervisor._output_tree_closure_root(
            Path(run_spec[f"{stage}_output"])
        )
        v3_scratch_root = supervisor._object_root(
            supervisor._SCRATCH_ABSENCE_DOMAIN,
            {"path": run_spec[f"{stage}_scratch"], "absent": True},
        )
        if (
            type(source_replay) is not dict
            or source_replay.get("source_closure_root") != v3_source_root
            or stage_row.get("mount_registry_root") != v3_mount_root
            or stage_row.get("output_tree_closure_root") != v3_output_root
            or stage_row.get("scratch_absence_root") != v3_scratch_root
        ):
            _terminal(
                "Q1_START_COMPLETION_ADAPTER_INVALID",
                f"{stage} persisted closure replay differs",
            )
        isolation_adapters[stage] = adapter

    formal_paths = gate16.get("formal_relative_paths")
    if type(formal_paths) is not list or not formal_paths:
        _terminal("Q1_START_COMPLETION_ADAPTER_INVALID", "formal path registry differs")
    python_by_path = {row.relative_path: row for row in output_files["python"]}
    try:
        formal_files = tuple(python_by_path[path] for path in formal_paths)
    except KeyError as error:
        _terminal("Q1_START_COMPLETION_ADAPTER_INVALID", f"formal file absent: {error}")
    v3_formal_file_root = supervisor._file_closure_root(
        Path(run_spec["python_output"]),
        formal_paths,
        supervisor._FORMAL_FILE_CLOSURE_DOMAIN,
    )
    if gate16.get("formal_file_closure_root") != v3_formal_file_root:
        _terminal("Q1_START_COMPLETION_ADAPTER_INVALID", "formal file closure differs")
    formal_file_root = completion.formal_output_file_closure_root_v1(formal_files)

    def runtime_values(stage: str) -> tuple[int, int]:
        runtime = stage_rows[stage]["runtime_evidence"]
        started = runtime.get("measurement_interval_start_monotonic_ns")
        returned = runtime.get("actor_return_monotonic_ns")
        exited = runtime.get("container_exit_monotonic_ns")
        wall_ns = runtime.get("measurement_wall_time_ns")
        peak = runtime.get("memory_peak_bytes")
        if (
            type(started) is not int
            or type(returned) is not int
            or type(exited) is not int
            or type(wall_ns) is not int
            or not started <= returned <= exited
            or wall_ns != exited - started
            or type(peak) is not int
            or peak < 0
        ):
            _terminal("Q1_START_COMPLETION_ADAPTER_INVALID", f"{stage} runtime counters differ")
        return peak, wall_ns

    endpoints: dict[str, Any] = {}
    for stage in ("python", "rust"):
        by_path = {row.relative_path: row for row in output_files[stage]}
        try:
            result_payload = by_path["endpoint-result.json"].payload
        except KeyError as error:
            _terminal("Q1_START_COMPLETION_OUTPUT_UNAVAILABLE", f"{stage}: {error}")
        try:
            result_value = campaign.strict_json_bytes_v6(
                result_payload, f"{stage} endpoint result"
            )
        except Exception as error:
            _terminal("Q1_START_COMPLETION_ADAPTER_INVALID", str(error))
        memory_peak, wall_ns = runtime_values(stage)
        endpoints[stage] = completion.Q1EndpointCompletionEvidenceV1(
            actor_id=stage.encode("ascii"),
            actor_implementation_root=root(
                run_spec["actor_implementation_roots"][stage],
                f"{stage} implementation",
            ),
            run_genesis_root=run_genesis,
            prestart_admission_receipt_root=prestart_receipt,
            endpoint_result_root=completion.endpoint_result_root_v1(
                stage.encode("ascii"), result_payload, **binding_kwargs
            ),
            closure_bundle_root=closure_bundle,
            signature_rows=signatures,
            actual_output_bytes=sum(len(row.payload) for row in output_files[stage]),
            actual_peak_scratch_bytes=result_value["actual_formal_peak_scratch_bytes"],
            actual_peak_rss_bytes=memory_peak,
            actual_wall_time_ns=wall_ns,
        )

    host_files = {row.relative_path: row for row in output_files["host"]}
    try:
        host_payload = host_files["host-result.json"].payload
    except KeyError as error:
        _terminal("Q1_START_COMPLETION_OUTPUT_UNAVAILABLE", f"host: {error}")
    host_memory_peak, host_wall_ns = runtime_values("host")
    host_runtime = stage_rows["host"]["runtime_evidence"]
    if (
        host_runtime.get("actor_terminal_scratch_payload_bytes") != 0
        or host_runtime.get("supervisor_control_peak_payload_bytes") != 0
        or "--scratch-dir" in host_runtime.get("command", [])
    ):
        _terminal("Q1_START_COMPLETION_ADAPTER_INVALID", "host scratch evidence differs")
    host_peak_scratch = 0
    agreement = completion.Q1DualReplayAgreementV1(
        run_genesis_root=run_genesis,
        prestart_admission_receipt_root=prestart_receipt,
        python_endpoint_evidence_root=endpoints["python"].evidence_root,
        rust_endpoint_evidence_root=endpoints["rust"].evidence_root,
        formal_file_closure_root=formal_file_root,
        signature_rows=signatures,
        closure_bundle_root=closure_bundle,
        host_actor_implementation_root=root(
            run_spec["actor_implementation_roots"]["host"], "host implementation"
        ),
        host_replay_result_root=completion.host_replay_result_root_v1(
            host_payload, **binding_kwargs
        ),
        actual_host_output_bytes=sum(len(row.payload) for row in output_files["host"]),
        actual_host_peak_scratch_bytes=host_peak_scratch,
        actual_host_peak_rss_bytes=host_memory_peak,
        actual_host_wall_time_ns=host_wall_ns,
    )
    actor_rows = tuple(
        isolation_adapters[stage].isolation_row() for stage in _ACTOR_ORDER
    )
    try:
        mount_manifest_root = completion.runtime_mount_registry_root_v1(actor_rows)
        input_absence_root = (
            completion.formal_input_key_registry_absence_root_v1()
        )
    except AttributeError as error:
        raise ControllerPending(
            "LOCAL_DEPENDENCY",
            "COMPLETION_AGGREGATE_ROOT_ADAPTER_UNAVAILABLE",
            str(error),
        ) from error
    # The successor bundle validator proves the v3-domain empty-input
    # manifest, while this completion wire independently proves the v1-domain
    # empty formal-input-key registry.  Their digests intentionally differ by
    # domain and are not compared byte-for-byte.
    successor.validate_q1_prestart_successor_candidate_v6(
        capsule["successor_prestart_bundle"]
    )
    ledger = completion.Q1TargetBlindAccessLedgerV1(
        run_genesis_root=run_genesis,
        prestart_admission_receipt_root=prestart_receipt,
        dual_replay_agreement_root=agreement.agreement_root,
        mount_manifest_root=mount_manifest_root,
        input_key_registry_absence_root=input_absence_root,
        actor_rows=actor_rows,
    )
    subjects = (
        (
            run_genesis,
            prestart_receipt,
            (
                endpoints["python"].canonical_object(),
                endpoints["rust"].canonical_object(),
            ),
        ),
        (
            endpoints["python"].evidence_root,
            endpoints["rust"].evidence_root,
            formal_file_root,
            tuple(row.canonical_object() for row in signatures),
            closure_bundle,
            True,
            True,
        ),
        (agreement.canonical_object(), agreement.agreement_root),
        (ledger.canonical_object(), ledger.ledger_root),
    )
    receipt = completion.Q1CompletionReceiptV1(
        completion.Q1CompletionCoreV1(
            run_genesis_root=run_genesis,
            prestart_admission_receipt_root=prestart_receipt,
            gate_subject_rows_15_through_18=subjects,
            output_slots_1_through_7=completion.output_slots_1_through_7_v1(
                agreement, ledger
            ),
        )
    )
    completion.decode_completion_receipt_v1(receipt.canonical_bytes)
    return receipt.canonical_bytes


def default_dependencies_v6() -> ControllerDependenciesV6:
    return ControllerDependenciesV6(
        _actual_source_state,
        _actual_replay_predecessor,
        _actual_formal_blob,
        _actual_live_gate,
        _actual_startup_smoke,
        _actual_invoke_supervisor,
        _actual_build_completion_artifact,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    try:
        if not path.exists():
            path.mkdir(mode=0o700)
            _fsync_directory(path.parent)
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISDIR(info.st_mode) or path.resolve(strict=True) != path:
            raise OSError("directory identity differs")
    except OSError as error:
        raise ControllerPending("PATH_AND_FORMAT", "PROTOCOL_DIRECTORY_UNAVAILABLE", f"{path}: {error}") from error


def _ensure_roots(contract: Any) -> None:
    fixed = contract.value["fixed_paths"]
    for path in (
        Path(fixed["run_root"]),
        Path(fixed["ledger_root"]),
        Path(fixed["artifact_path"]).parent,
    ):
        _ensure_directory(path)


@contextmanager
def _campaign_lock(contract: Any) -> Iterator[None]:
    path = Path(contract.value["fixed_paths"]["campaign_lock_path"])
    try:
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as error:
        raise ControllerPending("PATH_AND_FORMAT", "CAMPAIGN_LOCK_UNAVAILABLE", str(error)) from error
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ControllerPending("PATH_AND_FORMAT", "CAMPAIGN_LOCK_BUSY", str(error)) from error
        yield
    finally:
        os.close(descriptor)


def _stage_path(path: Path, digest: str) -> Path:
    return path.with_name(path.name + ".stage-" + digest[-16:])


def _publish_bytes(path: Path, payload: bytes, digest: str, *, mode: int = 0o400) -> None:
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise ControllerPending("PATH_AND_FORMAT", "PUBLISHED_FILE_UNAVAILABLE", f"{path}: {error}") from error
        if existing != payload:
            raise ControllerPending("PATH_AND_FORMAT", "NO_REPLACE_PATH_OCCUPIED", str(path))
        return
    stage = _stage_path(path, digest)
    try:
        if stage.exists():
            stage.unlink()
            _fsync_directory(path.parent)
        descriptor = os.open(
            stage,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            mode,
        )
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(stage, path, follow_symlinks=False)
        _fsync_directory(path.parent)
        stage.unlink()
        _fsync_directory(path.parent)
    except OSError as error:
        raise ControllerPending("PATH_AND_FORMAT", "DURABLE_NO_REPLACE_PUBLICATION_FAILED", f"{path}: {error}") from error


def _publish_json(path: Path, value: Mapping[str, object], digest: str) -> None:
    _publish_bytes(path, campaign.canonical_json_bytes_v6(dict(value)), digest)


def _load_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ControllerPending("PATH_AND_FORMAT", "REQUIRED_RECORD_UNAVAILABLE", f"{label}: {error}") from error
    try:
        return raw, campaign.strict_json_bytes_v6(raw, label)
    except Exception as error:
        raise ControllerPending("PATH_AND_FORMAT", "REQUIRED_RECORD_INVALID", f"{label}: {error}") from error


def _load_capsule(contract: Any, capsule_digest: str) -> dict[str, Any]:
    if type(capsule_digest) is not str or _TAGGED.fullmatch(capsule_digest) is None:
        raise ControllerPending("PATH_AND_FORMAT", "CAPSULE_DIGEST_INVALID", str(capsule_digest))
    path = Path(contract.value["fixed_paths"]["capsule_path"])
    raw, value = _load_json(path, "Q1 start capsule")
    try:
        capsule = campaign.validate_capsule_v6(value, contract)
    except Exception as error:
        raise ControllerPending("PATH_AND_FORMAT", "CAPSULE_REPLAY_DIFFERS", str(error)) from error
    if capsule["capsule_digest"] != capsule_digest or raw != campaign.canonical_json_bytes_v6(capsule):
        raise ControllerPending("PATH_AND_FORMAT", "CAPSULE_REPLAY_DIFFERS", capsule_digest)
    return capsule


def _replay_controller_source(contract: Any, capsule: Mapping[str, object], dependencies: ControllerDependenciesV6) -> None:
    observed = dependencies.source_state(contract)
    if (
        dict(observed.identity) != capsule["controller_source_identity"]
        or [dict(row) for row in observed.rows] != capsule["controller_source_closure_rows"]
    ):
        raise ControllerPending("SOURCE_IDENTITY", "CAPSULE_CONTROLLER_SOURCE_DIFFERS", str(contract.path))


def _replay_preconsumption_gates(
    contract: Any,
    capsule: Mapping[str, object],
    dependencies: ControllerDependenciesV6,
) -> PredecessorReplayV6:
    _replay_controller_source(contract, capsule, dependencies)
    predecessor = dependencies.replay_predecessor(contract)
    try:
        bundle = successor.validate_q1_prestart_successor_candidate_v6(
            capsule.get("successor_prestart_bundle")
        )
    except Exception as error:
        raise ControllerPending(
            "PREDECESSOR_REPLAY", "SUCCESSOR_BUNDLE_CAPSULE_BINDING_DIFFERS", str(error)
        ) from error
    bundle_path = Path(contract.value["successor_prestart_bundle_descriptor"]["path"])
    if bundle_path.exists():
        try:
            external = successor.replay_q1_prestart_successor_candidate_v6(
                bundle_path.read_bytes()
            )
        except Exception as error:
            raise ControllerPending(
                "PREDECESSOR_REPLAY", "SUCCESSOR_BUNDLE_EXTERNAL_REPLAY_DIFFERS", str(error)
            ) from error
        if external != bundle:
            raise ControllerPending(
                "PREDECESSOR_REPLAY",
                "SUCCESSOR_BUNDLE_CAPSULE_BINDING_DIFFERS",
                str(bundle_path),
            )
    dependencies.live_gate(contract, capsule)
    return PredecessorReplayV6(
        predecessor.base_capsule,
        predecessor.base_authorization,
        predecessor.base_consumption,
        predecessor.base_artifact,
        predecessor.failed_capsule,
        predecessor.failed_authorization,
        predecessor.failed_consumption,
        predecessor.failed_terminal,
        predecessor.failed_v3_capsule,
        predecessor.failed_v3_authorization,
        predecessor.failed_v3_consumption,
        predecessor.failed_v3_terminal,
        predecessor.failed_v4_capsule,
        predecessor.failed_v4_authorization,
        predecessor.failed_v4_consumption,
        predecessor.failed_v4_terminal,
        predecessor.failed_v5_capsule,
        predecessor.failed_v5_authorization,
        predecessor.failed_v5_consumption,
        predecessor.failed_v5_terminal,
        bundle,
    )


def _snapshot_replay(root: Path, rows: Sequence[Mapping[str, object]]) -> None:
    try:
        info = root.lstat()
        if root.is_symlink() or not stat.S_ISDIR(info.st_mode) or root.resolve(strict=True) != root:
            raise OSError("snapshot root identity differs")
        observed = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )
    except OSError as error:
        raise ControllerPending("SOURCE_SNAPSHOT", "SOURCE_SNAPSHOT_UNAVAILABLE", f"{root}: {error}") from error
    expected = [str(row["relative_path"]) for row in rows]
    if observed != expected:
        raise ControllerPending("SOURCE_SNAPSHOT", "SOURCE_SNAPSHOT_REGISTRY_DIFFERS", str(root))
    for row in rows:
        path = root / str(row["relative_path"])
        payload = _regular_file(path, f"snapshot {row['relative_path']}")
        if len(payload) != row["byte_count"] or campaign.raw_sha256_v6(payload) != row["raw_sha256"]:
            raise ControllerPending("SOURCE_SNAPSHOT", "SOURCE_SNAPSHOT_BYTES_DIFFER", str(path))


def _materialize_snapshot(
    project_root: Path,
    root: Path,
    rows: Sequence[Mapping[str, object]],
    commit: str,
    capsule_digest: str,
    blob_reader: FormalBlobReader,
) -> None:
    if root.exists():
        _snapshot_replay(root, rows)
        return
    stage = _stage_path(root, capsule_digest)
    try:
        if stage.exists():
            shutil.rmtree(stage)
            _fsync_directory(root.parent)
        stage.mkdir(mode=0o700)
        cache: dict[str, bytes] = {}
        for row in rows:
            relative = str(row["relative_path"])
            payload = cache.setdefault(relative, blob_reader(project_root, commit, relative))
            if len(payload) != row["byte_count"] or campaign.raw_sha256_v6(payload) != row["raw_sha256"]:
                raise ControllerPending("SOURCE_SNAPSHOT", "FORMAL_GIT_BLOB_DIFFERS", relative)
            target = stage / relative
            target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o444,
            )
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
        for directory in sorted(
            (item for item in stage.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            os.chmod(directory, 0o555)
            _fsync_directory(directory)
        os.chmod(stage, 0o555)
        _fsync_directory(stage)
        os.rename(stage, root)
        _fsync_directory(root.parent)
    except ControllerPending:
        raise
    except OSError as error:
        raise ControllerPending("SOURCE_SNAPSHOT", "SOURCE_SNAPSHOT_PUBLICATION_FAILED", f"{root}: {error}") from error
    _snapshot_replay(root, rows)


def _materialize_bound_inputs(
    contract: Any,
    capsule: Mapping[str, object],
    predecessor: PredecessorReplayV6,
    dependencies: ControllerDependenciesV6,
) -> dict[str, object]:
    fixed = contract.value["fixed_paths"]
    runtime = contract.value["runtime"]
    project_root = Path(fixed["project_root"])
    bundle = dict(predecessor.successor_bundle)
    source_identity = bundle["source_identity"]
    commit = source_identity["source_commit"]
    manifest_registry = capsule.get("actor_source_manifests")
    if type(manifest_registry) is not dict or set(manifest_registry) != set(
        _ACTOR_ORDER
    ):
        raise ControllerPending(
            "SOURCE_SNAPSHOT", "ACTOR_SOURCE_MANIFEST_REGISTRY_DIFFERS", "capsule"
        )
    for stage in _ACTOR_ORDER:
        manifest = manifest_registry[stage]
        if type(manifest) is not dict or set(manifest) != {
            "source_manifest_root",
            "rows",
        }:
            raise ControllerPending(
                "SOURCE_SNAPSHOT", "ACTOR_SOURCE_MANIFEST_REGISTRY_DIFFERS", stage
            )
        rows = manifest["rows"]
        if (
            type(rows) is not list
            or manifest["source_manifest_root"]
            != bundle["actor_source_manifest_roots"][stage]
            or successor.actor_source_manifest_root_v6(stage, rows)
            != bundle["actor_source_manifest_roots"][stage]
        ):
            raise ControllerPending(
                "SOURCE_SNAPSHOT", "ACTOR_SOURCE_MANIFEST_REPLAY_DIFFERS", stage
            )
        _materialize_snapshot(
            project_root,
            Path(fixed[f"{stage}_source_root"]),
            rows,
            commit,
            str(capsule["capsule_digest"]),
            dependencies.formal_blob,
        )
    admission: dict[str, object] = {
        "schema_version": "hegel-q1-formal-archive-materializer-admission/1",
        "action_id": "materialize-full-node6-formal-archive-v1",
        "capsule_digest": capsule["capsule_digest"],
        "source_commit": commit,
        "run_genesis_root": bundle["run_genesis_root"],
        "prestart_admission_receipt_root": bundle["receipt_root"],
        "projection_profile_root": PROJECTION_PROFILE_ROOT,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_total": 20,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "diagnostic_only": False,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
    }
    admission_path = Path(fixed["admission_path"])
    admission_payload = campaign.canonical_json_bytes_v6(admission)
    _publish_bytes(
        admission_path,
        admission_payload,
        str(capsule["capsule_digest"]),
        mode=0o444,
    )
    try:
        admission_info = admission_path.lstat()
        admission_mode = stat.S_IMODE(admission_info.st_mode)
    except OSError as error:
        raise ControllerPending(
            "LOCAL_BINDINGS",
            "PYTHON_ADMISSION_REPLAY_DIFFERS",
            str(error),
        ) from error
    if (
        admission_path.is_symlink()
        or not stat.S_ISREG(admission_info.st_mode)
        or admission_mode != 0o444
    ):
        raise ControllerPending(
            "LOCAL_BINDINGS",
            "PYTHON_ADMISSION_MODE_DIFFERS",
            f"{admission_path}: {admission_mode:04o}",
        )
    try:
        admission_raw = admission_path.read_bytes()
    except OSError as error:
        raise ControllerPending(
            "LOCAL_BINDINGS",
            "PYTHON_ADMISSION_REPLAY_DIFFERS",
            str(error),
        ) from error
    if admission_raw != admission_payload:
        raise ControllerPending("LOCAL_BINDINGS", "PYTHON_ADMISSION_REPLAY_DIFFERS", str(admission_path))

    bundle_payload = successor.canonical_json_bytes_v6(bundle)
    bundle_path = Path(
        contract.value["successor_prestart_bundle_descriptor"]["path"]
    )
    _ensure_directory(bundle_path.parent)
    _publish_bytes(
        bundle_path,
        bundle_payload,
        str(capsule["capsule_digest"]),
    )
    if (
        bundle_path.read_bytes() != bundle_payload
        or successor.replay_q1_prestart_successor_candidate_v6(bundle_payload)
        != bundle
    ):
        raise ControllerPending(
            "LOCAL_BINDINGS", "SUCCESSOR_BUNDLE_REPLAY_DIFFERS", str(bundle_path)
        )

    capsule_path = Path(fixed["capsule_path"])
    capsule_raw = _regular_file(capsule_path, "Q1 start v6 capsule")
    expected_authorization = campaign.build_authorization_v6(contract, capsule)

    run_spec: dict[str, object] = {
        "schema_version": "hegel-phase3a-q1-start-successor-supervisor-run-spec/6",
        "start_capsule_digest": capsule["capsule_digest"],
        "start_capsule": str(capsule_path),
        "start_capsule_raw_sha256": campaign.raw_sha256_v6(capsule_raw),
        "start_authorization": fixed["authorization_path"],
        "start_authorization_raw_sha256": campaign.raw_sha256_v6(
            campaign.canonical_json_bytes_v6(expected_authorization)
        ),
        "start_authorization_digest": expected_authorization["authorization_digest"],
        "successor_prestart_bundle": str(bundle_path),
        "successor_prestart_bundle_raw_sha256": campaign.raw_sha256_v6(
            bundle_payload
        ),
        "successor_prestart_bundle_root": bundle["bundle_root"],
        "base_authoritative_artifact": contract.value[
            "base_authoritative_predecessor"
        ]["prestart_artifact_path"],
        "base_authoritative_artifact_raw_sha256": contract.value[
            "base_authoritative_predecessor"
        ]["prestart_artifact_raw_sha256"],
        "failed_start_v2_capsule": contract.value[
            "failed_start_v2_predecessor"
        ]["capsule_path"],
        "failed_start_v2_capsule_raw_sha256": contract.value[
            "failed_start_v2_predecessor"
        ]["capsule_raw_sha256"],
        "failed_start_v2_capsule_digest": contract.value[
            "failed_start_v2_predecessor"
        ]["capsule_digest"],
        "failed_start_v2_authorization": contract.value[
            "failed_start_v2_predecessor"
        ]["authorization_path"],
        "failed_start_v2_authorization_raw_sha256": contract.value[
            "failed_start_v2_predecessor"
        ]["authorization_raw_sha256"],
        "failed_start_v2_authorization_digest": contract.value[
            "failed_start_v2_predecessor"
        ]["authorization_digest"],
        "failed_start_v2_consumption": contract.value[
            "failed_start_v2_predecessor"
        ]["consumption_path"],
        "failed_start_v2_consumption_raw_sha256": contract.value[
            "failed_start_v2_predecessor"
        ]["consumption_raw_sha256"],
        "failed_start_v2_consumption_record_digest": contract.value[
            "failed_start_v2_predecessor"
        ]["consumption_record_digest"],
        "failed_start_v2_terminal": contract.value[
            "failed_start_v2_predecessor"
        ]["terminal_path"],
        "failed_start_v2_terminal_raw_sha256": contract.value[
            "failed_start_v2_predecessor"
        ]["terminal_raw_sha256"],
        "failed_start_v2_terminal_record_digest": contract.value[
            "failed_start_v2_predecessor"
        ]["terminal_record_digest"],
        "failed_start_v3_capsule": contract.value[
            "failed_start_v3_predecessor"
        ]["capsule_path"],
        "failed_start_v3_capsule_raw_sha256": contract.value[
            "failed_start_v3_predecessor"
        ]["capsule_raw_sha256"],
        "failed_start_v3_capsule_digest": contract.value[
            "failed_start_v3_predecessor"
        ]["capsule_digest"],
        "failed_start_v3_authorization": contract.value[
            "failed_start_v3_predecessor"
        ]["authorization_path"],
        "failed_start_v3_authorization_raw_sha256": contract.value[
            "failed_start_v3_predecessor"
        ]["authorization_raw_sha256"],
        "failed_start_v3_authorization_digest": contract.value[
            "failed_start_v3_predecessor"
        ]["authorization_digest"],
        "failed_start_v3_consumption": contract.value[
            "failed_start_v3_predecessor"
        ]["consumption_path"],
        "failed_start_v3_consumption_raw_sha256": contract.value[
            "failed_start_v3_predecessor"
        ]["consumption_raw_sha256"],
        "failed_start_v3_consumption_record_digest": contract.value[
            "failed_start_v3_predecessor"
        ]["consumption_record_digest"],
        "failed_start_v3_terminal": contract.value[
            "failed_start_v3_predecessor"
        ]["terminal_path"],
        "failed_start_v3_terminal_raw_sha256": contract.value[
            "failed_start_v3_predecessor"
        ]["terminal_raw_sha256"],
        "failed_start_v3_terminal_record_digest": contract.value[
            "failed_start_v3_predecessor"
        ]["terminal_record_digest"],
        "failed_start_v4_capsule": contract.value[
            "failed_start_v4_predecessor"
        ]["capsule_path"],
        "failed_start_v4_capsule_raw_sha256": contract.value[
            "failed_start_v4_predecessor"
        ]["capsule_raw_sha256"],
        "failed_start_v4_capsule_digest": contract.value[
            "failed_start_v4_predecessor"
        ]["capsule_digest"],
        "failed_start_v4_authorization": contract.value[
            "failed_start_v4_predecessor"
        ]["authorization_path"],
        "failed_start_v4_authorization_raw_sha256": contract.value[
            "failed_start_v4_predecessor"
        ]["authorization_raw_sha256"],
        "failed_start_v4_authorization_digest": contract.value[
            "failed_start_v4_predecessor"
        ]["authorization_digest"],
        "failed_start_v4_consumption": contract.value[
            "failed_start_v4_predecessor"
        ]["consumption_path"],
        "failed_start_v4_consumption_raw_sha256": contract.value[
            "failed_start_v4_predecessor"
        ]["consumption_raw_sha256"],
        "failed_start_v4_consumption_record_digest": contract.value[
            "failed_start_v4_predecessor"
        ]["consumption_record_digest"],
        "failed_start_v4_terminal": contract.value[
            "failed_start_v4_predecessor"
        ]["terminal_path"],
        "failed_start_v4_terminal_raw_sha256": contract.value[
            "failed_start_v4_predecessor"
        ]["terminal_raw_sha256"],
        "failed_start_v4_terminal_record_digest": contract.value[
            "failed_start_v4_predecessor"
        ]["terminal_record_digest"],
        "failed_start_v5_capsule": contract.value[
            "failed_start_v5_predecessor"
        ]["capsule_path"],
        "failed_start_v5_capsule_raw_sha256": contract.value[
            "failed_start_v5_predecessor"
        ]["capsule_raw_sha256"],
        "failed_start_v5_capsule_digest": contract.value[
            "failed_start_v5_predecessor"
        ]["capsule_digest"],
        "failed_start_v5_authorization": contract.value[
            "failed_start_v5_predecessor"
        ]["authorization_path"],
        "failed_start_v5_authorization_raw_sha256": contract.value[
            "failed_start_v5_predecessor"
        ]["authorization_raw_sha256"],
        "failed_start_v5_authorization_digest": contract.value[
            "failed_start_v5_predecessor"
        ]["authorization_digest"],
        "failed_start_v5_consumption": contract.value[
            "failed_start_v5_predecessor"
        ]["consumption_path"],
        "failed_start_v5_consumption_raw_sha256": contract.value[
            "failed_start_v5_predecessor"
        ]["consumption_raw_sha256"],
        "failed_start_v5_consumption_record_digest": contract.value[
            "failed_start_v5_predecessor"
        ]["consumption_record_digest"],
        "failed_start_v5_terminal": contract.value[
            "failed_start_v5_predecessor"
        ]["terminal_path"],
        "failed_start_v5_terminal_raw_sha256": contract.value[
            "failed_start_v5_predecessor"
        ]["terminal_raw_sha256"],
        "failed_start_v5_terminal_record_digest": contract.value[
            "failed_start_v5_predecessor"
        ]["terminal_record_digest"],
        "corrected_resource_envelope_root": bundle[
            "corrected_resource_envelope_root"
        ],
        "actor_source_manifest_roots": dict(
            bundle["actor_source_manifest_roots"]
        ),
        "actor_source_closure_root": bundle["actor_source_closure_root"],
        "source_commit": commit,
        "run_genesis_root": bundle["run_genesis_root"],
        "prestart_admission_receipt_root": bundle["receipt_root"],
        "container_user": f"{runtime['actor_uid']}:{runtime['actor_gid']}",
        "python_image": runtime["python_image"],
        "rust_image": runtime["rust_image"],
        "rust_binary_raw_sha256": capsule["rust_binary"]["raw_sha256"],
        "seccomp_profile_raw_sha256": runtime["seccomp_profile_raw_sha256"],
        "python_admission_raw_sha256": campaign.raw_sha256_v6(admission_raw),
        "actor_implementation_roots": dict(bundle["actor_implementation_roots"]),
        "python_source_root": fixed["python_source_root"],
        "rust_source_root": fixed["rust_source_root"],
        "host_source_root": fixed["host_source_root"],
        "rust_binary": fixed["rust_binary_path"],
        "python_admission_json": fixed["admission_path"],
        "seccomp_profile": runtime["seccomp_profile_path"],
        "run_root": fixed["run_root"],
        "checkpoint_root": fixed["checkpoint_root"],
        "python_output": fixed["python_output_path"],
        "python_scratch": fixed["python_scratch_path"],
        "rust_output": fixed["rust_output_path"],
        "rust_scratch": fixed["rust_scratch_path"],
        "host_output": fixed["host_output_path"],
        "host_scratch": fixed["host_scratch_path"],
        "runtime_limits": {
            "python": {
                "cpuset_cpus": runtime["python_cpuset"],
                "memory_bytes": runtime["endpoint_memory_bytes"],
                "output_bytes": runtime["endpoint_output_bytes"],
                "scratch_bytes": runtime["endpoint_scratch_bytes"],
                "stdout_bytes": runtime["endpoint_output_bytes"],
                "wall_time_seconds": runtime["endpoint_wall_seconds"],
            },
            "rust": {
                "cpuset_cpus": runtime["rust_cpuset"],
                "memory_bytes": runtime["endpoint_memory_bytes"],
                "output_bytes": runtime["endpoint_output_bytes"],
                "scratch_bytes": runtime["endpoint_scratch_bytes"],
                "stdout_bytes": runtime["endpoint_output_bytes"],
                "wall_time_seconds": runtime["endpoint_wall_seconds"],
            },
            "host": {
                "cpuset_cpus": runtime["host_cpuset"],
                "memory_bytes": runtime["host_memory_bytes"],
                "output_bytes": runtime["host_output_bytes"],
                "scratch_bytes": runtime["host_scratch_bytes"],
                "stdout_bytes": runtime["host_output_bytes"],
                "wall_time_seconds": runtime["host_wall_seconds"],
            },
        },
        "actor_source_manifests": {
            stage: [dict(row) for row in manifest_registry[stage]["rows"]]
            for stage in _ACTOR_ORDER
        },
    }
    payload = campaign.canonical_json_bytes_v6(run_spec)
    try:
        _load_supervisor_module().load_run_spec_v6(payload)
    except Exception as error:
        raise ControllerPending("LOCAL_BINDINGS", "RUN_SPEC_VALIDATION_DIFFERS", str(error)) from error
    run_spec_path = Path(fixed["run_spec_path"])
    _publish_bytes(run_spec_path, payload, str(capsule["capsule_digest"]))
    if run_spec_path.read_bytes() != payload:
        raise ControllerPending("LOCAL_BINDINGS", "RUN_SPEC_REPLAY_DIFFERS", str(run_spec_path))
    return run_spec


def _prove_fresh_preconsumption_absence(contract: Any) -> None:
    """Prove the smoke did not create any campaign execution state."""

    fixed = contract.value["fixed_paths"]
    names = (
        "consumption_path",
        "checkpoint_root",
        "python_output_path",
        "python_scratch_path",
        "rust_output_path",
        "rust_scratch_path",
        "host_output_path",
        "host_scratch_path",
        "outcome_path",
        "terminal_path",
        "artifact_path",
    )
    occupied = [str(Path(fixed[name])) for name in names if Path(fixed[name]).exists()]
    if occupied:
        raise ControllerPending(
            "SUPERVISOR_STARTUP_SMOKE",
            "SMOKE_OR_FUTURE_EXECUTION_ROOTS_NOT_ABSENT",
            ",".join(occupied),
        )


def _startup_smoke_receipt(
    contract: Any,
    run_spec: Mapping[str, object],
    dependencies: ControllerDependenciesV6,
) -> Mapping[str, object]:
    try:
        receipt = dependencies.startup_smoke(contract, run_spec)
    except ControllerPending:
        raise
    except Exception as error:
        import traceback

        detail = _bounded_diagnostic(
            "".join(traceback.format_exception(error)).encode("utf-8", "replace")
        )
        raise ControllerPending(
            "SUPERVISOR_STARTUP_SMOKE",
            "AUTHORITY_STARTUP_SMOKE_FAILED",
            detail,
        ) from error
    supervisor = _load_supervisor_module()
    mutable = [
        str(Path(run_spec[name]))
        for name in (
            "checkpoint_root",
            "python_output",
            "python_scratch",
            "rust_output",
            "rust_scratch",
            "host_output",
            "host_scratch",
        )
    ]
    expected_keys = {
        "schema_version", "status", "capsule_digest", "start_authorization_digest",
        "failed_start_v2_terminal_record_digest",
        "failed_start_v3_terminal_record_digest",
        "failed_start_v4_terminal_record_digest",
        "failed_start_v5_terminal_record_digest",
        "container_user", "host_effective_user",
        "container_user_matches_host_effective_user",
        "ownership_freeze_probe", "ownership_probe_popen_count",
        "successor_prestart_bundle_root",
        "authority_replay", "stage_binding_replays", "mutable_roots_proven_absent",
        "campaign_supervisor_popen_count", "actor_popen_count",
        "checkpoint_or_mutable_roots_created", "docker_invoked",
    }
    if (
        type(receipt) is not dict
        or set(receipt) != expected_keys
        or receipt.get("schema_version") != supervisor.AUTHORITY_SMOKE_SCHEMA
        or receipt.get("status") != supervisor.AUTHORITY_SMOKE_STATUS
        or receipt.get("capsule_digest") != run_spec["start_capsule_digest"]
        or receipt.get("start_authorization_digest")
        != run_spec["start_authorization_digest"]
        or receipt.get("failed_start_v2_terminal_record_digest")
        != run_spec["failed_start_v2_terminal_record_digest"]
        or receipt.get("failed_start_v3_terminal_record_digest")
        != run_spec["failed_start_v3_terminal_record_digest"]
        or receipt.get("failed_start_v4_terminal_record_digest")
        != run_spec["failed_start_v4_terminal_record_digest"]
        or receipt.get("failed_start_v5_terminal_record_digest")
        != run_spec["failed_start_v5_terminal_record_digest"]
        or receipt.get("container_user") != run_spec["container_user"]
        or receipt.get("host_effective_user")
        != f"{os.geteuid()}:{os.getegid()}"
        or receipt.get("container_user_matches_host_effective_user") is not True
        or receipt.get("successor_prestart_bundle_root")
        != run_spec["successor_prestart_bundle_root"]
        or type(receipt.get("authority_replay")) is not dict
        or type(receipt.get("stage_binding_replays")) is not dict
        or set(receipt["stage_binding_replays"]) != set(_ACTOR_ORDER)
        or receipt.get("mutable_roots_proven_absent") != mutable
        or receipt.get("campaign_supervisor_popen_count") != 0
        or receipt.get("actor_popen_count") != 0
        or receipt.get("ownership_probe_popen_count") != 1
        or receipt.get("checkpoint_or_mutable_roots_created") is not False
        or receipt.get("docker_invoked") is not True
    ):
        raise ControllerPending(
            "SUPERVISOR_STARTUP_SMOKE",
            "AUTHORITY_STARTUP_SMOKE_RESULT_DIFFERS",
            repr(receipt),
        )
    probe = receipt.get("ownership_freeze_probe")
    expected_probe_keys = {
        "schema_version", "status", "container_user", "python_image",
        "network_mode", "container_exit_code", "observed_uid", "observed_gid",
        "probe_file_relative_path", "probe_file_mode_after_freeze",
        "probe_directory_mode_after_freeze", "host_freeze_succeeded",
        "docker_popen_count", "probe_root_removed",
    }
    if (
        type(probe) is not dict
        or set(probe) != expected_probe_keys
        or probe.get("schema_version") != supervisor.OWNERSHIP_FREEZE_PROBE_SCHEMA
        or probe.get("status") != supervisor.OWNERSHIP_FREEZE_PROBE_STATUS
        or probe.get("container_user") != run_spec["container_user"]
        or probe.get("python_image") != run_spec["python_image"]
        or probe.get("network_mode") != "none"
        or probe.get("container_exit_code") != 0
        or probe.get("observed_uid") != contract.value["runtime"]["actor_uid"]
        or probe.get("observed_gid") != contract.value["runtime"]["actor_gid"]
        or probe.get("probe_file_relative_path")
        != "signature-02/stream-04/chunk-000000.framed"
        or probe.get("probe_file_mode_after_freeze") != "0444"
        or probe.get("probe_directory_mode_after_freeze") != "0555"
        or probe.get("host_freeze_succeeded") is not True
        or probe.get("docker_popen_count") != 1
        or probe.get("probe_root_removed") is not True
    ):
        raise ControllerPending(
            "SUPERVISOR_STARTUP_SMOKE",
            "OUTPUT_OWNERSHIP_FREEZE_PROBE_RESULT_DIFFERS",
            repr(probe),
        )
    return receipt


def _pending_record(
    contract: Any,
    capsule: Mapping[str, object],
    pending: ControllerPending,
    *,
    authorization: Mapping[str, object] | None = None,
    consumption: Mapping[str, object] | None = None,
    completed: list[str] | None = None,
) -> dict[str, Any]:
    status = campaign.PENDING_CLEAN_CHECKPOINT if consumption is not None else campaign.PENDING_NOT_CONSUMED
    return campaign.build_pending_record_v6(
        contract,
        capsule,
        status=status,
        phase=pending.phase,
        condition=pending.condition,
        detail=pending.detail,
        authorization_digest=(authorization or {}).get("authorization_digest"),
        consumption_record_digest=(consumption or {}).get("record_digest"),
        completed_actors=completed or [],
    )


def _unbound_pending(contract_path: Path, capsule_digest: str | None, error: BaseException) -> dict[str, object]:
    return {
        "schema_version": CONTROLLER_PENDING_SCHEMA,
        "status": campaign.PENDING_NOT_CONSUMED,
        "contract_raw_sha256": None,
        "capsule_digest": capsule_digest,
        "phase": "PATH_AND_FORMAT",
        "condition": "CONTRACT_OR_CAPSULE_INPUTS_UNAVAILABLE",
        "detail": f"{contract_path}: {error}",
        "authorization_consumption_count": 0,
        "logical_campaign_count": 0,
        "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
        "network_accessed": False,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
    }


def _read_authorization(contract: Any, capsule: Mapping[str, object]) -> dict[str, Any] | None:
    path = Path(contract.value["fixed_paths"]["authorization_path"])
    if not path.exists():
        return None
    raw, value = _load_json(path, "Q1 start authorization")
    try:
        authorization = campaign.validate_authorization_v6(value, contract, capsule)
    except Exception as error:
        raise ControllerPending("PATH_AND_FORMAT", "AUTHORIZATION_REPLAY_DIFFERS", str(error)) from error
    if raw != campaign.canonical_json_bytes_v6(authorization):
        raise ControllerPending("PATH_AND_FORMAT", "AUTHORIZATION_REPLAY_DIFFERS", str(path))
    return authorization


def _read_consumption(
    contract: Any,
    capsule: Mapping[str, object],
    authorization: Mapping[str, object],
) -> dict[str, Any] | None:
    path = Path(contract.value["fixed_paths"]["consumption_path"])
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ConsumptionReplayTerminal(
            "Q1_START_CONSUMPTION_REPLAY_DIFFERS",
            str(error),
        ) from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ConsumptionReplayTerminal(
            "Q1_START_CONSUMPTION_REPLAY_DIFFERS",
            str(path),
        )
    try:
        raw, value = _load_json(path, "Q1 start consumption")
        consumption = campaign.validate_consumption_v6(value, contract, capsule, authorization)
    except Exception as error:
        raise ConsumptionReplayTerminal(
            "Q1_START_CONSUMPTION_REPLAY_DIFFERS",
            str(error),
        ) from error
    if raw != campaign.canonical_json_bytes_v6(consumption):
        raise ConsumptionReplayTerminal(
            "Q1_START_CONSUMPTION_REPLAY_DIFFERS",
            str(path),
        )
    return consumption


def _validate_completion_projection(
    contract: Any,
    capsule: Mapping[str, object],
    receipt: Any,
    outcome: Mapping[str, object],
) -> None:
    expected_slots = ["sha256:" + row[2].hex() for row in receipt.output_slots]
    if (
        outcome.get("status") != campaign.COMPLETE
        or outcome.get("q1_state") != "COMPLETE"
        or outcome.get("q1_gate_count") != 20
        or outcome.get("q1_gate_mask") != 1_048_575
        or outcome.get("q1_output_slots") != expected_slots
        or outcome.get("completion_receipt_root")
        != "sha256:" + receipt.receipt_root.hex()
    ):
        _terminal(
            "Q1_START_TERMINAL_ARTIFACT_AUTHORITY_DIFFERS",
            "COMPLETE projection differs from receipt",
        )
    try:
        gate15 = receipt.core.gate_subject_rows_15_through_18[0]
        endpoint_objects = gate15[2]
        gate18 = receipt.core.gate_subject_rows_15_through_18[3]
        ledger_object = gate18[0]
        actor_objects = ledger_object[9]
        receipt_roots = {
            endpoint_objects[0][2].decode("ascii"): "sha256:" + endpoint_objects[0][3].hex(),
            endpoint_objects[1][2].decode("ascii"): "sha256:" + endpoint_objects[1][3].hex(),
            actor_objects[2][2].decode("ascii"): "sha256:" + actor_objects[2][3].hex(),
        }
    except (IndexError, AttributeError, UnicodeDecodeError, TypeError) as error:
        _terminal("Q1_START_TERMINAL_ARTIFACT_AUTHORITY_DIFFERS", str(error))
    bundle = capsule.get("successor_prestart_bundle")
    if (
        type(bundle) is not dict
        or receipt_roots != bundle.get("actor_implementation_roots")
    ):
        _terminal(
            "Q1_START_TERMINAL_ARTIFACT_AUTHORITY_DIFFERS",
            "receipt actor implementation registry differs",
        )


def _complete_outcome_v6(
    capsule: Mapping[str, object],
    receipt: Any,
    supervisor_outcome_raw: bytes,
) -> dict[str, Any]:
    return {
        "status": campaign.COMPLETE,
        "capsule_digest": capsule["capsule_digest"],
        "actor_start_count_by_actor": {"host": 1, "python": 1, "rust": 1},
        "network_accessed": False,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "q1_state": "COMPLETE",
        "q1_gate_count": 20,
        "q1_gate_mask": 1_048_575,
        "q1_output_slots": [
            "sha256:" + row[2].hex() for row in receipt.output_slots
        ],
        "completion_receipt_root": "sha256:" + receipt.receipt_root.hex(),
        "supervisor_outcome_raw_sha256": campaign.raw_sha256_v6(
            supervisor_outcome_raw
        ),
    }


def _validate_receipt_authority(
    contract: Any,
    capsule: Mapping[str, object],
    receipt: Any,
    outcome: Mapping[str, object],
) -> None:
    bundle = capsule.get("successor_prestart_bundle")
    if (
        type(bundle) is not dict
        or
        receipt.core.run_genesis_root.hex()
        != str(bundle.get("run_genesis_root")).removeprefix("sha256:")
        or receipt.core.prestart_admission_receipt_root.hex()
        != str(bundle.get("receipt_root")).removeprefix("sha256:")
    ):
        _terminal(
            "Q1_START_COMPLETION_AUTHORITY_DIFFERS",
            "receipt predecessor roots differ",
        )
    _validate_completion_projection(contract, capsule, receipt, outcome)


def _read_terminal(
    contract: Any,
    capsule: Mapping[str, object],
    authorization: Mapping[str, object],
) -> dict[str, Any] | None:
    path = Path(contract.value["fixed_paths"]["terminal_path"])
    if not path.exists():
        return None
    raw, value = _load_json(path, "Q1 start terminal")
    try:
        terminal = campaign.validate_terminal_record_v6(value, contract, capsule, authorization)
    except Exception as error:
        _terminal("Q1_START_TERMINAL_REPLAY_DIFFERS", str(error))
    if raw != campaign.canonical_json_bytes_v6(terminal):
        _terminal("Q1_START_TERMINAL_REPLAY_DIFFERS", str(path))
    artifact_digest = terminal["artifact_raw_sha256"]
    artifact_path = Path(contract.value["fixed_paths"]["artifact_path"])
    if artifact_digest is not None:
        try:
            artifact = artifact_path.read_bytes()
            receipt = completion.decode_completion_receipt_v1(artifact)
        except Exception as error:
            _terminal("Q1_START_TERMINAL_ARTIFACT_REPLAY_DIFFERS", str(error))
        if campaign.raw_sha256_v6(artifact) != artifact_digest:
            _terminal("Q1_START_TERMINAL_ARTIFACT_REPLAY_DIFFERS", str(artifact_path))
        _validate_completion_projection(
            contract, capsule, receipt, terminal["outcome"]
        )
    elif artifact_path.exists():
        if (
            terminal["outcome"].get("failure_code")
            != "Q1_START_UNSEALED_ORPHAN_ARTIFACT"
        ):
            _terminal("Q1_START_NON_SUCCESS_ARTIFACT_PRESENT", str(artifact_path))
    return terminal


def _read_partial_run_spec(
    contract: Any,
    capsule: Mapping[str, object],
) -> dict[str, Any]:
    path = Path(contract.value["fixed_paths"]["run_spec_path"])
    try:
        raw = path.read_bytes()
        value = campaign.strict_json_bytes_v6(raw, "partial-finalization run spec")
        spec = _load_supervisor_module().load_run_spec_v6(raw)
    except Exception as error:
        _terminal("Q1_START_PARTIAL_FINALIZATION_RUN_SPEC_INVALID", str(error))
    expected_binding, expected_authority = _expected_supervisor_bindings(
        contract, capsule
    )
    if (
        raw != campaign.canonical_json_bytes_v6(value)
        or spec.binding() != expected_binding
        or spec.authority_binding() != expected_authority
    ):
        _terminal(
            "Q1_START_PARTIAL_FINALIZATION_RUN_SPEC_INVALID",
            str(path),
        )
    return value


def _read_partial_supervisor_success(
    contract: Any,
    capsule: Mapping[str, object],
) -> tuple[bytes, dict[str, Any]]:
    supervisor = _load_supervisor_module()
    path = (
        Path(contract.value["fixed_paths"]["checkpoint_root"])
        / "supervisor-outcome.json"
    )
    try:
        raw = path.read_bytes()
        value = campaign.strict_json_bytes_v6(
            raw, "partial-finalization supervisor success"
        )
    except Exception as error:
        _terminal("Q1_START_PARTIAL_SUPERVISOR_SUCCESS_INVALID", str(error))
    expected_checkpoints = [
        str(_stage_state_paths(contract, stage)[1]) for stage in _ACTOR_ORDER
    ]
    if (
        raw != campaign.canonical_json_bytes_v6(value)
        or frozenset(value) != _SUPERVISOR_SUCCESS_KEYS
        or value.get("schema_version") != supervisor.OUTCOME_SCHEMA
        or value.get("status") != supervisor.SUCCESS_STATUS
        or value.get("completed_stages") != list(_ACTOR_ORDER)
        or value.get("checkpoint_paths") != expected_checkpoints
        or value.get("q1_completion_receipt_issued") is not False
        or value.get("state_published") is not False
    ):
        _terminal("Q1_START_PARTIAL_SUPERVISOR_SUCCESS_INVALID", str(path))
    _supervisor_binding(contract, capsule, value)
    return raw, value


def _rebuild_partial_complete(
    contract: Any,
    capsule: Mapping[str, object],
    dependencies: ControllerDependenciesV6,
) -> tuple[bytes, Any, dict[str, Any]]:
    run_spec = _read_partial_run_spec(contract, capsule)
    supervisor_raw, supervisor_outcome = _read_partial_supervisor_success(
        contract, capsule
    )
    try:
        expected_artifact = dependencies.build_completion_artifact(
            supervisor_outcome,
            contract,
            capsule,
            run_spec,
        )
        receipt = completion.decode_completion_receipt_v1(expected_artifact)
    except ControllerTerminal:
        raise
    except ControllerPending as error:
        _terminal(
            "Q1_START_PARTIAL_COMPLETION_REPLAY_UNAVAILABLE",
            f"{error.condition}: {error.detail}",
        )
    except Exception as error:
        _terminal("Q1_START_PARTIAL_COMPLETION_REPLAY_INVALID", str(error))
    outcome = _complete_outcome_v6(capsule, receipt, supervisor_raw)
    _validate_receipt_authority(contract, capsule, receipt, outcome)
    return expected_artifact, receipt, outcome


def _seal_terminal_only(
    contract: Any,
    capsule: Mapping[str, object],
    authorization: Mapping[str, object],
    outcome: Mapping[str, object],
    *,
    artifact_raw_sha256: str | None,
) -> dict[str, Any]:
    terminal = campaign.build_terminal_record_v6(
        contract,
        capsule,
        authorization,
        outcome,
        artifact_raw_sha256=artifact_raw_sha256,
    )
    _publish_json(
        Path(contract.value["fixed_paths"]["terminal_path"]),
        terminal,
        str(capsule["capsule_digest"]),
    )
    return terminal


def _partial_failure_starts(
    contract: Any,
    candidate: Mapping[str, object] | None,
    *,
    artifact_present: bool,
) -> dict[str, int]:
    try:
        return _durable_start_counts(contract)
    except ControllerTerminal:
        counts = None if candidate is None else candidate.get(
            "actor_start_count_by_actor"
        )
        allowed = (
            {"host": 0, "python": 0, "rust": 0},
            {"host": 0, "python": 1, "rust": 0},
            {"host": 0, "python": 1, "rust": 1},
            {"host": 1, "python": 1, "rust": 1},
        )
        if counts in allowed:
            return dict(counts)
        if artifact_present:
            return {"host": 1, "python": 1, "rust": 1}
        return {"host": 0, "python": 0, "rust": 0}


def _seal_unrecoverable_partial(
    contract: Any,
    capsule: Mapping[str, object],
    authorization: Mapping[str, object],
    candidate: Mapping[str, object] | None,
    error: ControllerTerminal,
    *,
    artifact_present: bool,
) -> dict[str, Any]:
    code = (
        "Q1_START_UNSEALED_ORPHAN_ARTIFACT"
        if artifact_present
        else "Q1_START_UNSEALED_ORPHAN_OUTCOME"
    )
    outcome = {
        "status": campaign.OUTCOME_UNKNOWN,
        "capsule_digest": capsule["capsule_digest"],
        "actor_start_count_by_actor": _partial_failure_starts(
            contract,
            candidate,
            artifact_present=artifact_present,
        ),
        "network_accessed": False,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "failure_code": code,
        "failure_detail": f"{error.code}: {error.detail}",
    }
    outcome_path = Path(contract.value["fixed_paths"]["outcome_path"])
    if not outcome_path.exists():
        _publish_json(outcome_path, outcome, str(capsule["capsule_digest"]))
    return _seal_terminal_only(
        contract,
        capsule,
        authorization,
        outcome,
        artifact_raw_sha256=None,
    )


def _recover_partial_finalization(
    contract: Any,
    capsule: Mapping[str, object],
    authorization: Mapping[str, object],
    dependencies: ControllerDependenciesV6,
) -> dict[str, Any] | None:
    fixed = contract.value["fixed_paths"]
    outcome_path = Path(fixed["outcome_path"])
    artifact_path = Path(fixed["artifact_path"])
    if not outcome_path.exists() and not artifact_path.exists():
        return None
    candidate: dict[str, Any] | None = None
    try:
        # Recovery may tolerate only the exact fixed artifact as worktree
        # dirtiness; HEAD/tree and every capsule-bound controller source byte
        # still replay before any completion or failure terminal is sealed.
        _replay_controller_source(contract, capsule, dependencies)
        if outcome_path.exists():
            raw = outcome_path.read_bytes()
            candidate = campaign.strict_json_bytes_v6(
                raw, "partial-finalization outer outcome"
            )
            if raw != campaign.canonical_json_bytes_v6(candidate):
                _terminal(
                    "Q1_START_PARTIAL_OUTCOME_INVALID",
                    str(outcome_path),
                )
            if candidate.get("status") == campaign.COMPLETE:
                if not artifact_path.exists():
                    _terminal(
                        "Q1_START_PARTIAL_COMPLETION_ARTIFACT_ABSENT",
                        str(artifact_path),
                    )
                expected_artifact, _receipt, expected_outcome = (
                    _rebuild_partial_complete(contract, capsule, dependencies)
                )
                artifact = artifact_path.read_bytes()
                if artifact != expected_artifact or candidate != expected_outcome:
                    _terminal(
                        "Q1_START_PARTIAL_COMPLETE_REPLAY_DIFFERS",
                        str(outcome_path),
                    )
                return _seal_terminal_only(
                    contract,
                    capsule,
                    authorization,
                    candidate,
                    artifact_raw_sha256=campaign.raw_sha256_v6(artifact),
                )
            if artifact_path.exists():
                _terminal(
                    "Q1_START_PARTIAL_NON_SUCCESS_ARTIFACT_PRESENT",
                    str(artifact_path),
                )
            campaign.build_terminal_record_v6(
                contract,
                capsule,
                authorization,
                candidate,
                artifact_raw_sha256=None,
            )
            return _seal_terminal_only(
                contract,
                capsule,
                authorization,
                candidate,
                artifact_raw_sha256=None,
            )

        expected_artifact, _receipt, expected_outcome = _rebuild_partial_complete(
            contract, capsule, dependencies
        )
        artifact = artifact_path.read_bytes()
        if artifact != expected_artifact:
            _terminal(
                "Q1_START_PARTIAL_COMPLETION_ARTIFACT_DIFFERS",
                str(artifact_path),
            )
        _publish_json(
            outcome_path,
            expected_outcome,
            str(capsule["capsule_digest"]),
        )
        return _seal_terminal_only(
            contract,
            capsule,
            authorization,
            expected_outcome,
            artifact_raw_sha256=campaign.raw_sha256_v6(artifact),
        )
    except ControllerPending as error:
        terminal_error = ControllerTerminal(
            "Q1_START_PARTIAL_FINALIZATION_PENDING_NOT_ALLOWED",
            f"{error.condition}: {error.detail}",
        )
        return _seal_unrecoverable_partial(
            contract,
            capsule,
            authorization,
            candidate,
            terminal_error,
            artifact_present=artifact_path.exists(),
        )
    except ControllerTerminal as error:
        return _seal_unrecoverable_partial(
            contract,
            capsule,
            authorization,
            candidate,
            error,
            artifact_present=artifact_path.exists(),
        )
    except Exception as error:
        return _seal_unrecoverable_partial(
            contract,
            capsule,
            authorization,
            candidate,
            ControllerTerminal(
                "Q1_START_PARTIAL_FINALIZATION_REPLAY_EXCEPTION",
                str(error),
            ),
            artifact_present=artifact_path.exists(),
        )


def _not_authorized(contract: Any, capsule: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": CONTROLLER_RESULT_SCHEMA,
        "status": NOT_AUTHORIZED,
        "contract_raw_sha256": contract.raw_sha256,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_consumption_count": 0,
        "logical_campaign_count": 0,
        "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "network_accessed": False,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
    }


def preflight_v6(
    contract_path: Path,
    capsule_inputs_path: Path,
    *,
    dependencies: ControllerDependenciesV6 | None = None,
) -> dict[str, Any]:
    dependencies = dependencies or default_dependencies_v6()
    try:
        contract = campaign.load_contract_v6(contract_path)
        inputs_raw = capsule_inputs_path.read_bytes()
        inputs = campaign.strict_json_bytes_v6(inputs_raw, "Q1 start capsule inputs")
        capsule = campaign.build_capsule_v6(contract, inputs)
    except (OSError, Exception) as error:
        return _unbound_pending(contract_path, None, error)
    try:
        with _campaign_lock(contract):
            _ensure_roots(contract)
            _replay_preconsumption_gates(contract, capsule, dependencies)
            fixed = contract.value["fixed_paths"]
            capsule_path = Path(fixed["capsule_path"])
            replayed = capsule_path.exists()
            if not capsule_path.exists():
                forbidden = [
                    Path(fixed[name])
                    for name in (
                        "authorization_path", "consumption_path", "terminal_path",
                        "admission_path", "run_spec_path", "outcome_path", "artifact_path",
                        "checkpoint_root",
                        "python_source_root", "rust_source_root", "host_source_root",
                        "python_output_path", "python_scratch_path",
                        "rust_output_path", "rust_scratch_path",
                        "host_output_path", "host_scratch_path",
                    )
                ]
                occupied = [str(path) for path in forbidden if path.exists()]
                if occupied:
                    raise ControllerPending("PATH_AND_FORMAT", "FUTURE_RUN_ROOTS_NOT_ABSENT", ",".join(occupied))
            _publish_json(capsule_path, capsule, str(capsule["capsule_digest"]))
        return {**capsule, "read_only_replay": replayed}
    except ControllerPending as error:
        return _pending_record(contract, capsule, error)


def record_authorization_v6(
    contract_path: Path,
    capsule_digest: str,
    *,
    dependencies: ControllerDependenciesV6 | None = None,
) -> dict[str, Any]:
    dependencies = dependencies or default_dependencies_v6()
    try:
        contract = campaign.load_contract_v6(contract_path)
    except Exception as error:
        return _unbound_pending(contract_path, capsule_digest, error)
    capsule: dict[str, Any] | None = None
    try:
        with _campaign_lock(contract):
            _ensure_roots(contract)
            capsule = _load_capsule(contract, capsule_digest)
            predecessor = _replay_preconsumption_gates(contract, capsule, dependencies)
            _materialize_bound_inputs(contract, capsule, predecessor, dependencies)
            authorization = campaign.build_authorization_v6(contract, capsule)
            path = Path(contract.value["fixed_paths"]["authorization_path"])
            _publish_json(path, authorization, capsule_digest)
            replay = _read_authorization(contract, capsule)
            assert replay is not None
            return replay
    except ControllerPending as error:
        if capsule is None:
            return _unbound_pending(contract_path, capsule_digest, error)
        return _pending_record(contract, capsule, error)


def _prefix_counts(completed: Sequence[str], *, active: str | None = None) -> dict[str, int]:
    started = list(completed)
    if active in _ACTOR_ORDER and active not in started:
        started.append(active)
    if started not in ([], ["python"], ["python", "rust"], ["python", "rust", "host"]):
        _terminal("Q1_START_ACTOR_PREFIX_INVALID", repr(started))
    return {stage: int(stage in started) for stage in ("host", "python", "rust")}


def _stage_state_paths(contract: Any, stage: str) -> tuple[Path, Path]:
    checkpoint = Path(contract.value["fixed_paths"]["checkpoint_root"])
    ordinal = _ACTOR_ORDER.index(stage) + 1
    return (
        checkpoint / f"{ordinal:02d}-{stage}-intent.json",
        checkpoint / f"{ordinal:02d}-{stage}-complete.json",
    )


def _durable_completed_prefix(contract: Any) -> list[str]:
    """Strictly replay a clean checkpoint prefix without invoking an actor."""

    supervisor = _load_supervisor_module()
    run_spec_path = Path(contract.value["fixed_paths"]["run_spec_path"])
    try:
        spec = supervisor.load_run_spec_v6(run_spec_path.read_bytes())
        commands = supervisor.docker_commands_v6(spec)
    except (OSError, Exception) as error:
        _terminal("Q1_START_DURABLE_RUN_SPEC_REPLAY_DIFFERS", str(error))
    completed: list[str] = []
    found_gap = False
    for stage in _ACTOR_ORDER:
        intent, checkpoint = _stage_state_paths(contract, stage)
        if intent.exists() and checkpoint.exists() and not found_gap:
            try:
                supervisor._replay_checkpoint(
                    spec,
                    stage,
                    commands[stage],
                    supervisor.default_endpoint_replayer_v6,
                )
            except (
                supervisor.CompletedCheckpointBindingPending,
                supervisor.PendingCondition,
            ):
                # The checkpoint and output were validated before the mutable
                # current binding became temporarily unavailable.
                pass
            except supervisor.Q1StartSupervisorError as error:
                _terminal("Q1_START_DURABLE_CHECKPOINT_REPLAY_DIFFERS", str(error))
            except Exception as error:
                _terminal("Q1_START_DURABLE_CHECKPOINT_REPLAY_DIFFERS", str(error))
            completed.append(stage)
            continue
        if not intent.exists() and not checkpoint.exists():
            found_gap = True
            continue
        _terminal(
            "Q1_START_DURABLE_STAGE_STATE_NOT_CLEAN",
            f"{stage}: intent={intent.exists()} checkpoint={checkpoint.exists()}",
        )
    if completed == list(_ACTOR_ORDER):
        _terminal(
            "Q1_START_COMPLETE_CHECKPOINT_WITHOUT_TERMINAL",
            str(contract.value["fixed_paths"]["checkpoint_root"]),
        )
    return completed


def _durable_start_counts(contract: Any) -> dict[str, int]:
    started: list[str] = []
    seen_unstarted = False
    seen_partial = False
    for stage in _ACTOR_ORDER:
        intent, checkpoint = _stage_state_paths(contract, stage)
        has_intent, has_checkpoint = intent.exists(), checkpoint.exists()
        if has_checkpoint and not has_intent:
            _terminal("Q1_START_CHECKPOINT_WITHOUT_INTENT", stage)
        if has_intent:
            if seen_unstarted or seen_partial:
                _terminal("Q1_START_DURABLE_STAGE_ORDER_DIFFERS", stage)
            started.append(stage)
            if not has_checkpoint:
                seen_partial = True
        else:
            seen_unstarted = True
    return _prefix_counts(started)


def _starts_from_supervisor_outcome(
    contract: Any,
    outcome: Mapping[str, object] | None,
) -> dict[str, int]:
    if outcome is None:
        return _durable_start_counts(contract)
    completed = outcome.get("completed_stages")
    if type(completed) is not list:
        return _durable_start_counts(contract)
    if outcome.get("status") == getattr(_load_supervisor_module(), "SUCCESS_STATUS"):
        return _prefix_counts(_ACTOR_ORDER)
    if outcome.get("status") == getattr(_load_supervisor_module(), "PENDING_STATUS"):
        return _durable_start_counts(contract)
    active = outcome.get("active_stage")
    active_started = None
    if active in _ACTOR_ORDER and _stage_state_paths(contract, str(active))[0].exists():
        active_started = str(active)
    return _prefix_counts(completed, active=active_started)


def _expected_supervisor_bindings(
    contract: Any,
    capsule: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    bundle = capsule.get("successor_prestart_bundle")
    descriptor = capsule.get("successor_prestart_bundle_descriptor")
    base = capsule.get("base_authoritative_predecessor")
    failed = capsule.get("failed_start_v2_predecessor")
    failed_v3 = capsule.get("failed_start_v3_predecessor")
    failed_v4 = capsule.get("failed_start_v4_predecessor")
    failed_v5 = capsule.get("failed_start_v5_predecessor")
    if not all(
        type(value) is dict
        for value in (
            bundle,
            descriptor,
            base,
            failed,
            failed_v3,
            failed_v4,
            failed_v5,
        )
    ):
        _terminal("Q1_START_SUPERVISOR_BINDING_DIFFERS", "capsule authority shape differs")
    assert isinstance(bundle, dict)
    assert isinstance(descriptor, dict)
    assert isinstance(base, dict)
    assert isinstance(failed, dict)
    assert isinstance(failed_v3, dict)
    assert isinstance(failed_v4, dict)
    assert isinstance(failed_v5, dict)
    source = bundle.get("source_identity")
    if type(source) is not dict:
        _terminal("Q1_START_SUPERVISOR_BINDING_DIFFERS", "bundle source identity differs")
    authorization = campaign.build_authorization_v6(contract, capsule)
    fixed = contract.value["fixed_paths"]
    binding = {
        "capsule_digest": capsule["capsule_digest"],
        "source_commit": source["source_commit"],
        "run_genesis_root": bundle["run_genesis_root"],
        "prestart_admission_receipt_root": bundle["receipt_root"],
    }
    authority = {
        "start_capsule_path": fixed["capsule_path"],
        "start_capsule_raw_sha256": campaign.raw_sha256_v6(
            campaign.canonical_json_bytes_v6(dict(capsule))
        ),
        "start_authorization_path": fixed["authorization_path"],
        "start_authorization_raw_sha256": campaign.raw_sha256_v6(
            campaign.canonical_json_bytes_v6(authorization)
        ),
        "start_authorization_digest": authorization["authorization_digest"],
        "successor_prestart_bundle_path": descriptor["path"],
        "successor_prestart_bundle_raw_sha256": descriptor["raw_sha256"],
        "successor_prestart_bundle_root": descriptor["bundle_root"],
        "base_authoritative_artifact_path": base["prestart_artifact_path"],
        "base_authoritative_artifact_raw_sha256": base[
            "prestart_artifact_raw_sha256"
        ],
        "failed_start_v2_capsule_path": failed["capsule_path"],
        "failed_start_v2_capsule_raw_sha256": failed["capsule_raw_sha256"],
        "failed_start_v2_capsule_digest": failed["capsule_digest"],
        "failed_start_v2_authorization_path": failed["authorization_path"],
        "failed_start_v2_authorization_raw_sha256": failed[
            "authorization_raw_sha256"
        ],
        "failed_start_v2_authorization_digest": failed["authorization_digest"],
        "failed_start_v2_consumption_path": failed["consumption_path"],
        "failed_start_v2_consumption_raw_sha256": failed[
            "consumption_raw_sha256"
        ],
        "failed_start_v2_consumption_record_digest": failed[
            "consumption_record_digest"
        ],
        "failed_start_v2_terminal_path": failed["terminal_path"],
        "failed_start_v2_terminal_raw_sha256": failed["terminal_raw_sha256"],
        "failed_start_v2_terminal_record_digest": failed["terminal_record_digest"],
        "failed_start_v3_capsule_path": failed_v3["capsule_path"],
        "failed_start_v3_capsule_raw_sha256": failed_v3["capsule_raw_sha256"],
        "failed_start_v3_capsule_digest": failed_v3["capsule_digest"],
        "failed_start_v3_authorization_path": failed_v3["authorization_path"],
        "failed_start_v3_authorization_raw_sha256": failed_v3[
            "authorization_raw_sha256"
        ],
        "failed_start_v3_authorization_digest": failed_v3[
            "authorization_digest"
        ],
        "failed_start_v3_consumption_path": failed_v3["consumption_path"],
        "failed_start_v3_consumption_raw_sha256": failed_v3[
            "consumption_raw_sha256"
        ],
        "failed_start_v3_consumption_record_digest": failed_v3[
            "consumption_record_digest"
        ],
        "failed_start_v3_terminal_path": failed_v3["terminal_path"],
        "failed_start_v3_terminal_raw_sha256": failed_v3[
            "terminal_raw_sha256"
        ],
        "failed_start_v3_terminal_record_digest": failed_v3[
            "terminal_record_digest"
        ],
        "failed_start_v4_capsule_path": failed_v4["capsule_path"],
        "failed_start_v4_capsule_raw_sha256": failed_v4["capsule_raw_sha256"],
        "failed_start_v4_capsule_digest": failed_v4["capsule_digest"],
        "failed_start_v4_authorization_path": failed_v4["authorization_path"],
        "failed_start_v4_authorization_raw_sha256": failed_v4[
            "authorization_raw_sha256"
        ],
        "failed_start_v4_authorization_digest": failed_v4[
            "authorization_digest"
        ],
        "failed_start_v4_consumption_path": failed_v4["consumption_path"],
        "failed_start_v4_consumption_raw_sha256": failed_v4[
            "consumption_raw_sha256"
        ],
        "failed_start_v4_consumption_record_digest": failed_v4[
            "consumption_record_digest"
        ],
        "failed_start_v4_terminal_path": failed_v4["terminal_path"],
        "failed_start_v4_terminal_raw_sha256": failed_v4[
            "terminal_raw_sha256"
        ],
        "failed_start_v4_terminal_record_digest": failed_v4[
            "terminal_record_digest"
        ],
        "failed_start_v5_capsule_path": failed_v5["capsule_path"],
        "failed_start_v5_capsule_raw_sha256": failed_v5["capsule_raw_sha256"],
        "failed_start_v5_capsule_digest": failed_v5["capsule_digest"],
        "failed_start_v5_authorization_path": failed_v5["authorization_path"],
        "failed_start_v5_authorization_raw_sha256": failed_v5[
            "authorization_raw_sha256"
        ],
        "failed_start_v5_authorization_digest": failed_v5[
            "authorization_digest"
        ],
        "failed_start_v5_consumption_path": failed_v5["consumption_path"],
        "failed_start_v5_consumption_raw_sha256": failed_v5[
            "consumption_raw_sha256"
        ],
        "failed_start_v5_consumption_record_digest": failed_v5[
            "consumption_record_digest"
        ],
        "failed_start_v5_terminal_path": failed_v5["terminal_path"],
        "failed_start_v5_terminal_raw_sha256": failed_v5[
            "terminal_raw_sha256"
        ],
        "failed_start_v5_terminal_record_digest": failed_v5[
            "terminal_record_digest"
        ],
        "container_user": (
            f"{contract.value['runtime']['actor_uid']}:"
            f"{contract.value['runtime']['actor_gid']}"
        ),
        "corrected_resource_envelope_root": bundle[
            "corrected_resource_envelope_root"
        ],
        "actor_source_manifest_roots": dict(bundle["actor_source_manifest_roots"]),
        "actor_source_closure_root": bundle["actor_source_closure_root"],
    }
    return binding, authority


def _supervisor_binding(contract: Any, capsule: Mapping[str, object], outcome: Mapping[str, object]) -> None:
    binding, authority = _expected_supervisor_bindings(contract, capsule)
    expected = {**binding, **authority}
    if any(outcome.get(key) != value for key, value in expected.items()):
        _terminal("Q1_START_SUPERVISOR_BINDING_DIFFERS", repr(outcome))


def _persist_terminal(
    contract: Any,
    capsule: Mapping[str, object],
    authorization: Mapping[str, object],
    outcome: Mapping[str, object],
    *,
    artifact_raw_sha256: str | None,
) -> dict[str, Any]:
    digest = str(capsule["capsule_digest"])
    fixed = contract.value["fixed_paths"]
    _publish_json(Path(fixed["outcome_path"]), outcome, digest)
    terminal = campaign.build_terminal_record_v6(
        contract,
        capsule,
        authorization,
        outcome,
        artifact_raw_sha256=artifact_raw_sha256,
    )
    _publish_json(Path(fixed["terminal_path"]), terminal, digest)
    return terminal


def _terminal_from_controller_error(
    contract: Any,
    capsule: Mapping[str, object],
    authorization: Mapping[str, object],
    error: ControllerTerminal,
    *,
    starts: Mapping[str, int],
) -> dict[str, Any]:
    outcome = {
        "status": campaign.OUTCOME_UNKNOWN,
        "capsule_digest": capsule["capsule_digest"],
        "actor_start_count_by_actor": dict(starts),
        "network_accessed": False,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "failure_code": error.code,
        "failure_detail": error.detail,
    }
    return _persist_terminal(contract, capsule, authorization, outcome, artifact_raw_sha256=None)


def _handle_supervisor_outcome(
    contract: Any,
    capsule: Mapping[str, object],
    authorization: Mapping[str, object],
    consumption: Mapping[str, object],
    run_spec: Mapping[str, object],
    supervisor_outcome: Mapping[str, object],
    dependencies: ControllerDependenciesV6,
) -> dict[str, Any]:
    supervisor = _load_supervisor_module()
    status = supervisor_outcome.get("status")
    _supervisor_binding(contract, capsule, supervisor_outcome)
    completed = supervisor_outcome.get("completed_stages")
    if type(completed) is not list or completed not in ([], ["python"], ["python", "rust"], ["python", "rust", "host"]):
        _terminal("Q1_START_SUPERVISOR_ACTOR_PREFIX_DIFFERS", repr(completed))
    if status == supervisor.PENDING_STATUS:
        if (
            completed == ["python", "rust", "host"]
            or supervisor_outcome.get("resume_allowed") is not True
            or supervisor_outcome.get("next_actor_intent_absent") is not True
            or type(supervisor_outcome.get("pending_code")) is not str
            or type(supervisor_outcome.get("pending_detail")) is not str
        ):
            _terminal("Q1_START_SUPERVISOR_PENDING_WIRE_DIFFERS", repr(supervisor_outcome))
        fixed = contract.value["fixed_paths"]
        if completed == []:
            if supervisor_outcome.get("actor_popen_count_this_entry") != 0:
                _terminal("Q1_START_ZERO_CHECKPOINT_POPEN_DIFFERS", repr(supervisor_outcome))
            if any(_stage_state_paths(contract, stage)[0].exists() for stage in _ACTOR_ORDER):
                _terminal(
                    "Q1_START_ZERO_CHECKPOINT_INTENT_PRESENT",
                    str(fixed["checkpoint_root"]),
                )
        pending = ControllerPending(
            "SUPERVISOR_CLEAN_CHECKPOINT",
            str(supervisor_outcome["pending_code"]),
            str(supervisor_outcome["pending_detail"]),
        )
        return _pending_record(
            contract,
            capsule,
            pending,
            authorization=authorization,
            consumption=consumption,
            completed=list(completed),
        )
    if status == supervisor.TERMINAL_STATUS:
        if (
            supervisor_outcome.get("resume_allowed") is not False
            or supervisor_outcome.get("q1_completion_receipt_issued") is not False
            or supervisor_outcome.get("state_published") is not False
            or type(supervisor_outcome.get("error_code")) is not str
            or type(supervisor_outcome.get("error_detail")) is not str
        ):
            _terminal("Q1_START_SUPERVISOR_TERMINAL_WIRE_DIFFERS", repr(supervisor_outcome))
        active = supervisor_outcome.get("active_stage")
        active_started: str | None = None
        if active in _ACTOR_ORDER:
            intent = _stage_state_paths(contract, str(active))[0]
            if intent.exists():
                active_started = str(active)
        code = str(supervisor_outcome["error_code"])
        unknown = (
            "UNKNOWN" in code
            or "DURABILITY" in code
            or code == "Q1_START_POST_INTENT_EXCEPTION"
        )
        outcome = {
            "status": campaign.OUTCOME_UNKNOWN if unknown else campaign.INCONCLUSIVE,
            "capsule_digest": capsule["capsule_digest"],
            "actor_start_count_by_actor": _prefix_counts(completed, active=active_started),
            "network_accessed": False,
            "target_truth_accessed": False,
            "split_accessed": False,
            "role_evaluation_performed": False,
            "q1_state": "NOT_RUN",
            "q1_gate_count": 14,
            "q1_gate_mask": 16_383,
            "q1_output_slots": [None] * 8,
            "failure_code": code,
            "failure_detail": supervisor_outcome["error_detail"],
            "supervisor_outcome_raw_sha256": campaign.raw_sha256_v6(
                campaign.canonical_json_bytes_v6(dict(supervisor_outcome))
            ),
        }
        return _persist_terminal(contract, capsule, authorization, outcome, artifact_raw_sha256=None)
    if status != supervisor.SUCCESS_STATUS:
        _terminal("Q1_START_SUPERVISOR_STATUS_UNKNOWN", repr(status))
    if (
        completed != list(_ACTOR_ORDER)
        or supervisor_outcome.get("q1_completion_receipt_issued") is not False
        or supervisor_outcome.get("state_published") is not False
    ):
        _terminal("Q1_START_SUPERVISOR_SUCCESS_WIRE_DIFFERS", repr(supervisor_outcome))
    try:
        payload = dependencies.build_completion_artifact(
            supervisor_outcome, contract, capsule, run_spec
        )
    except (ControllerPending, ControllerTerminal):
        raise
    except Exception as error:
        _terminal("Q1_START_COMPLETION_ADAPTER_FAILED", str(error))
    try:
        receipt = completion.decode_completion_receipt_v1(payload)
    except Exception as error:
        _terminal("Q1_START_COMPLETION_ARTIFACT_INVALID", str(error))
    artifact_path = Path(contract.value["fixed_paths"]["artifact_path"])
    supervisor_raw = campaign.canonical_json_bytes_v6(dict(supervisor_outcome))
    outcome = _complete_outcome_v6(capsule, receipt, supervisor_raw)
    _validate_receipt_authority(contract, capsule, receipt, outcome)
    _publish_bytes(artifact_path, payload, str(capsule["capsule_digest"]))
    return _persist_terminal(
        contract,
        capsule,
        authorization,
        outcome,
        artifact_raw_sha256=campaign.raw_sha256_v6(payload),
    )


def run_v6(
    contract_path: Path,
    capsule_digest: str,
    *,
    dependencies: ControllerDependenciesV6 | None = None,
) -> dict[str, Any]:
    dependencies = dependencies or default_dependencies_v6()
    try:
        contract = campaign.load_contract_v6(contract_path)
    except Exception as error:
        return _unbound_pending(contract_path, capsule_digest, error)
    capsule: dict[str, Any] | None = None
    authorization: dict[str, Any] | None = None
    consumption: dict[str, Any] | None = None
    supervisor_outcome: Mapping[str, object] | None = None
    supervisor_invoked = False
    try:
        with _campaign_lock(contract):
            _ensure_roots(contract)
            capsule = _load_capsule(contract, capsule_digest)
            authorization = _read_authorization(contract, capsule)
            if authorization is None:
                return _not_authorized(contract, capsule)
            try:
                consumption = _read_consumption(contract, capsule, authorization)
            except ConsumptionReplayTerminal as error:
                return _terminal_from_controller_error(
                    contract,
                    capsule,
                    authorization,
                    error,
                    starts=_durable_start_counts(contract),
                )
            existing_terminal = _read_terminal(contract, capsule, authorization)
            if existing_terminal is not None:
                if consumption is None:
                    _terminal(
                        "Q1_START_TERMINAL_WITHOUT_CONSUMPTION",
                        str(contract.value["fixed_paths"]["terminal_path"]),
                    )
                return existing_terminal
            if consumption is not None:
                recovered = _recover_partial_finalization(
                    contract,
                    capsule,
                    authorization,
                    dependencies,
                )
                if recovered is not None:
                    return recovered
            predecessor = _replay_preconsumption_gates(contract, capsule, dependencies)
            run_spec = _materialize_bound_inputs(contract, capsule, predecessor, dependencies)
            if consumption is None:
                fixed = contract.value["fixed_paths"]
                _startup_smoke_receipt(contract, run_spec, dependencies)
                _prove_fresh_preconsumption_absence(contract)
                candidate_consumption = campaign.build_consumption_v6(
                    contract,
                    capsule,
                    authorization,
                )
                _publish_json(
                    Path(fixed["consumption_path"]),
                    candidate_consumption,
                    capsule_digest,
                )
                replayed_consumption = _read_consumption(contract, capsule, authorization)
                assert replayed_consumption is not None
                consumption = replayed_consumption
            supervisor_outcome = dependencies.invoke_supervisor(
                Path(contract.value["runtime"]["supervisor_python_path"]),
                Path(contract.value["fixed_paths"]["project_root"]) / SUPERVISOR_RELATIVE,
                Path(contract.value["fixed_paths"]["run_spec_path"]),
            )
            supervisor_invoked = True
            return _handle_supervisor_outcome(
                contract,
                capsule,
                authorization,
                consumption,
                run_spec,
                supervisor_outcome,
                dependencies,
            )
    except ControllerPending as error:
        if capsule is None:
            return _unbound_pending(contract_path, capsule_digest, error)
        if consumption is None and authorization is not None:
            consumption_path = Path(
                contract.value["fixed_paths"]["consumption_path"]
            )
            if consumption_path.exists():
                try:
                    replayed_consumption = _read_consumption(
                        contract,
                        capsule,
                        authorization,
                    )
                except ConsumptionReplayTerminal as replay_error:
                    return _terminal_from_controller_error(
                        contract,
                        capsule,
                        authorization,
                        replay_error,
                        starts=_durable_start_counts(contract),
                    )
                assert replayed_consumption is not None
                consumption = replayed_consumption
        if consumption is not None and authorization is not None:
            if supervisor_invoked:
                recovered = _recover_partial_finalization(
                    contract,
                    capsule,
                    authorization,
                    dependencies,
                )
                if recovered is not None:
                    return recovered
                terminal_error = ControllerTerminal(
                    "Q1_START_POST_SUPERVISOR_PENDING_NOT_RESUMABLE",
                    f"{error.condition}: {error.detail}",
                )
                return _terminal_from_controller_error(
                    contract,
                    capsule,
                    authorization,
                    terminal_error,
                    starts=_starts_from_supervisor_outcome(contract, supervisor_outcome),
                )
            try:
                completed = _durable_completed_prefix(contract)
            except ControllerTerminal as terminal_error:
                return _terminal_from_controller_error(
                    contract,
                    capsule,
                    authorization,
                    terminal_error,
                    starts=_durable_start_counts(contract),
                )
            return _pending_record(
                contract,
                capsule,
                error,
                authorization=authorization,
                consumption=consumption,
                completed=completed,
            )
        return _pending_record(
            contract,
            capsule,
            error,
            authorization=authorization,
            consumption=consumption,
            completed=[],
        )
    except ControllerTerminal as error:
        if capsule is None or authorization is None or consumption is None:
            if capsule is not None:
                return _pending_record(
                    contract,
                    capsule,
                    ControllerPending("PRECONSUMPTION_VALIDATION", error.code, error.detail),
                    authorization=authorization,
                )
            return _unbound_pending(contract_path, capsule_digest, error)
        if supervisor_invoked:
            recovered = _recover_partial_finalization(
                contract,
                capsule,
                authorization,
                dependencies,
            )
            if recovered is not None:
                return recovered
        return _terminal_from_controller_error(
            contract,
            capsule,
            authorization,
            error,
            starts=_starts_from_supervisor_outcome(contract, supervisor_outcome),
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the exact target-blind Q1 start campaign")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--record-authorization", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--capsule-inputs", type=Path)
    parser.add_argument("--capsule-digest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.preflight:
        if arguments.capsule_inputs is None or arguments.capsule_digest is not None:
            parser.error("--preflight requires --capsule-inputs and forbids --capsule-digest")
        result = preflight_v6(arguments.contract, arguments.capsule_inputs)
    elif arguments.record_authorization:
        if arguments.capsule_digest is None or arguments.capsule_inputs is not None:
            parser.error("--record-authorization requires --capsule-digest and forbids --capsule-inputs")
        result = record_authorization_v6(arguments.contract, arguments.capsule_digest)
    else:
        if arguments.capsule_digest is None or arguments.capsule_inputs is not None:
            parser.error("--run requires --capsule-digest and forbids --capsule-inputs")
        result = run_v6(arguments.contract, arguments.capsule_digest)
    sys.stdout.buffer.write(campaign.canonical_json_bytes_v6(result))
    status = result.get("status")
    if status in (campaign.PENDING_NOT_CONSUMED, campaign.PENDING_CLEAN_CHECKPOINT):
        return PENDING_EXIT_CODE
    if status in (campaign.INCONCLUSIVE, campaign.OUTCOME_UNKNOWN):
        return TERMINAL_EXIT_CODE
    if status == NOT_AUTHORIZED:
        return NOT_AUTHORIZED_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ControllerDependenciesV6",
    "ControllerPending",
    "ControllerTerminal",
    "PredecessorReplayV6",
    "SourceStateV6",
    "default_dependencies_v6",
    "main",
    "preflight_v6",
    "prepare_capsule_inputs_v6",
    "record_authorization_v6",
    "run_v6",
]
