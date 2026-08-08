"""Attempt-4 recovery after the terminal R3.1 audit-installer false negative.

R3.1 atomically published its exact attempt-start record and therefore
consumed ordinal 3.  It then compared the decoded JSON object with the typed
builder object and mistook ``list``/``tuple`` representation differences for
different evidence.  The exact attempt was terminalized before admission or
actor execution.  R4 binds that immutable seven-record chain, uses a fresh
audit namespace and authorization, and admits only recovery attempt ordinal 4.

This module never starts M3, redraws a seed, or opens/hashes the raw seed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Callable, Final, Mapping, NoReturn, Sequence

from . import phase3_m25_a8_recovery_amendment_r3_v1 as _r31
from .phase3_m25_a8_recovery_amendment_v1 import A8R1RecoveryDockerActorsV1
from .phase3_m25_formal_container_executor_v1 import (
    FAIL_RECOVERY_SOURCE_ADMISSION,
    FormalContainerExecutorError,
    PendingCeremonyRecoveryV1,
    _canonical_json as _executor_canonical_json,
    _continue_pre_stage_pending_recovery_core_v1,
    _replay_public_gate_evidence_with_fixed_a8_r3_basis_v1,
    _transport,
    acquire_pending_ceremony_recovery_v1,
)
from .phase3_m25_formal_static_basis_v1 import (
    FORMAL_GIT_ENVIRONMENT_V1,
    FORMAL_GIT_EXECUTABLE,
)


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
DEFAULT_MANIFEST_PATH: Final = (
    PROJECT_ROOT / "config/phase3_m25_a8_recovery_amendment_r4_v1.json"
)
A8_BASIS_COMMIT: Final = _r31.A8_BASIS_COMMIT
R1_AMENDMENT_COMMIT: Final = _r31.R1_AMENDMENT_COMMIT
R2_AMENDMENT_COMMIT: Final = _r31.R2_AMENDMENT_COMMIT
R3_AMENDMENT_COMMIT: Final = _r31.R3_AMENDMENT_COMMIT
R31_AMENDMENT_COMMIT: Final = "6c1b73064d292d57d5a9c35fd83c75caff57c300"
FIXED_RUN_ID_HEX: Final = _r31.FIXED_RUN_ID_HEX
FIXED_LEDGER_ID_HEX: Final = _r31.FIXED_LEDGER_ID_HEX
FIXED_RUN_ID: Final = _r31.FIXED_RUN_ID
FIXED_LEDGER_ID: Final = _r31.FIXED_LEDGER_ID
R31_TERMINAL_AUDIT_DIRECTORY: Final = _r31.FIXED_R3_AUDIT_DIRECTORY
FIXED_R4_AUDIT_DIRECTORY: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/"
    "phase3-m25-0af65964235390ce2bebefea7379eaa9c50eda24/"
    "recovery-audit-r4-e4af9f57c38fb298462ec628c4ed8a03-attempt-4"
)
MANIFEST_SCHEMA: Final = "hegel-phase3-m25-a8-recovery-amendment-r4/1"
AUDIT_SCHEMA_PREFIX: Final = "hegel-phase3-m25-a8-r4-recovery-audit"
FAIL_AMENDMENT: Final = "FAIL_M25_A8_R4_RECOVERY_AMENDMENT"
OWNER_CONFIRMATION: Final = (
    "AUTHORIZE_A8_R4_ATTEMPT_4_CANONICAL_BYTES_"
    "COMPLETE_ONLY_REAL_PENDING_RESUME"
)
CONTINUATION_ACTION: Final = (
    "POST_R31_TERMINAL_CANONICAL_INSTALLER_RECOVERY_CONTINUATION"
)
SOURCE_ADMISSION_CONTINUATION_ACTION: Final = (
    "CODE_AMENDMENT_RECOVERY_CONTINUATION"
)
AUTHORIZATION_REVISION_ID: Final = "R4_CANONICAL_AUDIT_INSTALLER_V1"
R31_DEFECT_CODE: Final = (
    "R31_ATTEMPT_START_RUNTIME_METADATA_LIST_TUPLE_EQUALITY"
)
R31_TERMINAL_AUDIT_RAW_SHA256: Final = {
    "preflight.json": "4a75c7cc1dfd02266e92b67f401d8c5a302f0fcfbc6960b8d803409c7e3f850a",
    "incident-diagnostic.json": "6d632eb02092822aefed5af5653489c12245ce94b28b43b53ae6daa06f9252e7",
    "a8-validation-receipt.json": "ef18694aa41a78389cef2265eb121174f2e68548928f89f7fcad3f55fb261ee4",
    "authorization-request.json": "386c7fa735024896b1fbcc5c6728371b3323370a62efb1367f6a6d109c44221e",
    "authorization.json": "fe5037abaa35aeb2a53b650a7cc5f1330e2198e2d90f065276c1526e4e81dc9b",
    "attempt-start.json": "09bbc99ad2b33930a043b0178bc5c1ebc3f71dfb09b025a412fbb00224493312",
    "failure.json": "90c176985d83780440007d2111577c0dc5ffbae5430eae523919653b7b6b0153",
}
R31_TERMINAL_AUDIT_RECEIPT_SHA256: Final = {
    "preflight.json": "f0d49565f622247bb488e844144b80a7a092ce0e4a123b78b0195690cc0e7258",
    "incident-diagnostic.json": "535bc494835cd5a00a396d92f2883225f22dba89e5e86ca00563709709f3fa67",
    "a8-validation-receipt.json": "83b1ad690914d9dfd5cd402d5c734a1250a3b450c9e0b3ecf4655cfb97c6ba47",
    "authorization-request.json": "69aa057f2326e459ea8def7e0501f586f9b67a4f47bad799509db52e88f2967a",
    "authorization.json": "629f00a809214e7979ba8ee35c37567b22be09878ba8c9d3723167ab8fdffa4c",
    "attempt-start.json": "92c127b3961e277b6f1f0a7ca34eeb04b420f5b801260f76406cb6b2c0aeb50f",
    "failure.json": "0eae7fa631bd7df2d6e446a220d78783387eb92a18b77c0c062e99957a6b883d",
}
R31_TERMINAL_CHAIN_ROOT_SHA256: Final = (
    "d4bb2c5984405d127537bde1e973f175b630a16bcaa8ec4fe15617e665400093"
)
R31_FAILURE_DETAIL_SHA256: Final = (
    "82a96d8a342b0ae22668763f738cea241d7c4ed34a00ed71b6dacd5193f694b2"
)
R4_RUNTIME_EXCEPTION_PATHS: Final = frozenset(
    {
        *_r31.R3_RUNTIME_EXCEPTION_PATHS,
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_amendment_r4_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_cli_r4_v1.py",
    }
)
_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")


class A8R4RecoveryAmendmentError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(detail: str) -> NoReturn:
    raise A8R4RecoveryAmendmentError(FAIL_AMENDMENT, detail)


def _canonical_json(value: object) -> bytes:
    return _r31._canonical_json(value)


def _receipt_record_bytes_v1(fields: Mapping[str, object]) -> bytes:
    return _canonical_json(_r31._r2._with_receipt_sha256(fields))


def _git(repository_root: Path, arguments: Sequence[str]) -> bytes:
    completed = subprocess.run(
        [str(FORMAL_GIT_EXECUTABLE), *arguments],
        cwd=repository_root.resolve(strict=True),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
        env=dict(FORMAL_GIT_ENVIRONMENT_V1),
    )
    if completed.returncode != 0:
        _fail(
            "formal Git check failed: "
            + completed.stderr.decode("utf-8", "replace")[-400:]
        )
    return completed.stdout


def _load_manifest(path: Path) -> tuple[dict[str, object], bytes]:
    if path.is_symlink():
        _fail("R4 amendment manifest may not be a symlink")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"R4 amendment manifest is invalid JSON: {exc}")
    required = {
        "schema", "source_commit_selector", "sole_parent_commit",
        "formal_repository_commit", "fixed_run_id_hex", "fixed_ledger_id_hex",
        "recovery_attempt_ordinal", "exact_changed_paths", "source_bindings",
        "complete_seed_resume_only", "formal_identity_entropy_draw_count",
        "ephemeral_container_nonce_allowed", "ordinary_execute_allowed",
        "ordinary_recovery_cross_basis_allowed", "parent_amendment_commit",
        "fixed_r31_terminal_audit_directory", "fixed_r4_audit_directory",
        "r31_terminal_audit_raw_sha256",
        "r31_terminal_audit_receipt_sha256",
        "r31_terminal_chain_root_sha256", "r31_failure_detail_sha256",
        "expected_live_bundle_sha256",
        "expected_a8_validation_receipt_sha256", "fixed_continuity_sha256",
        "continuation_action", "owner_confirmation", "fixed_runtime_artifacts",
        "a8_validator_execution", "authorization_revision_id",
        "r31_defect_code",
    }
    expected_validator = {
        "python_executable": _r31.FIXED_PYTHON_EXECUTABLE.as_posix(),
        "python_executable_sha256": _r31.FIXED_PYTHON_EXECUTABLE_SHA256,
        "python_executable_mode_octal": "0755",
        "isolated_flags": ["-I", "-S", "-B"],
        "python_pycache_prefix": _r31.FIXED_PYCACHE_PREFIX,
        "a8_import_closure_sha256_root": (
            _r31.EXPECTED_A8_IMPORT_CLOSURE_SHA256_ROOT
        ),
        "validator_dependency_closure_sha256_root": (
            _r31.EXPECTED_A8_VALIDATOR_DEPENDENCY_CLOSURE_SHA256_ROOT
        ),
        "tool_path": _r31.A8_VALIDATOR_TOOL.as_posix(),
        "formal_repository_root": _r31.FIXED_FORMAL_REPOSITORY_ROOT.as_posix(),
        "formal_repository_commit": A8_BASIS_COMMIT,
    }
    if (
        type(value) is not dict
        or _canonical_json(value) != raw
        or set(value) != required
    ):
        _fail("R4 amendment manifest is not canonical exact JSON")
    if (
        value.get("schema") != MANIFEST_SCHEMA
        or value.get("source_commit_selector") != "HEAD"
        or value.get("sole_parent_commit") != R31_AMENDMENT_COMMIT
        or value.get("parent_amendment_commit") != R31_AMENDMENT_COMMIT
        or value.get("formal_repository_commit") != A8_BASIS_COMMIT
        or value.get("fixed_run_id_hex") != FIXED_RUN_ID_HEX
        or value.get("fixed_ledger_id_hex") != FIXED_LEDGER_ID_HEX
        or value.get("recovery_attempt_ordinal") != 4
        or value.get("complete_seed_resume_only") is not True
        or value.get("formal_identity_entropy_draw_count") != 0
        or value.get("ephemeral_container_nonce_allowed") is not True
        or value.get("ordinary_execute_allowed") is not False
        or value.get("ordinary_recovery_cross_basis_allowed") is not False
        or value.get("fixed_r31_terminal_audit_directory")
        != R31_TERMINAL_AUDIT_DIRECTORY.as_posix()
        or value.get("fixed_r4_audit_directory")
        != FIXED_R4_AUDIT_DIRECTORY.as_posix()
        or value.get("r31_terminal_audit_raw_sha256")
        != R31_TERMINAL_AUDIT_RAW_SHA256
        or value.get("r31_terminal_audit_receipt_sha256")
        != R31_TERMINAL_AUDIT_RECEIPT_SHA256
        or value.get("r31_terminal_chain_root_sha256")
        != R31_TERMINAL_CHAIN_ROOT_SHA256
        or value.get("r31_failure_detail_sha256")
        != R31_FAILURE_DETAIL_SHA256
        or value.get("expected_live_bundle_sha256")
        != _r31._r2.EXPECTED_LIVE_BUNDLE_SHA256
        or value.get("expected_a8_validation_receipt_sha256")
        != _r31.EXPECTED_A8_VALIDATION_RECEIPT_RAW_SHA256
        or value.get("fixed_continuity_sha256") != _r31.FIXED_CONTINUITY_SHA256
        or value.get("continuation_action") != CONTINUATION_ACTION
        or value.get("owner_confirmation") != OWNER_CONFIRMATION
        or tuple(value.get("fixed_runtime_artifacts", ()))
        != _r31.FIXED_RUNTIME_ARTIFACTS
        or value.get("a8_validator_execution") != expected_validator
        or value.get("authorization_revision_id")
        != AUTHORIZATION_REVISION_ID
        or value.get("r31_defect_code") != R31_DEFECT_CODE
    ):
        _fail("R4 amendment manifest fixed policy differs")
    return value, raw


def inspect_r4_source_preflight_v1(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, object]:
    manifest, manifest_raw = _load_manifest(manifest_path)
    head = _git(repository_root, ("rev-parse", "--verify", "HEAD^{commit}")).decode(
        "ascii"
    ).strip()
    parents = _git(repository_root, ("show", "-s", "--format=%P", head)).decode(
        "ascii"
    ).strip().split()
    if _HEX_40.fullmatch(head) is None or parents != [R31_AMENDMENT_COMMIT]:
        _fail("R4 must be one committed sole child of the terminal R3.1 amendment")
    if _git(repository_root, ("status", "--porcelain=v1", "--untracked-files=all")):
        _fail("R4 repository tree/index is not clean")
    changed_lines = _git(
        repository_root,
        (
            "diff-tree", "--no-commit-id", "--name-status", "-r",
            "--no-renames", R31_AMENDMENT_COMMIT, head,
        ),
    ).decode("utf-8", "strict").splitlines()
    actual = tuple(
        {"status": line.split("\t", 1)[0], "path": line.split("\t", 1)[1]}
        for line in changed_lines
        if line
    )
    if (
        type(manifest.get("exact_changed_paths")) is not list
        or tuple(manifest["exact_changed_paths"]) != actual
    ):
        _fail("R4 changed-path allowlist differs")
    manifest_relative = manifest_path.resolve(strict=True).relative_to(
        repository_root.resolve(strict=True)
    ).as_posix()
    changed_paths = {str(row["path"]) for row in actual}
    _r31._verify_changed_index_flags_v1(repository_root, changed_paths)
    _r31._verify_changed_worktree_blob_v1(
        repository_root=repository_root,
        head=head,
        relative=manifest_relative,
        expected_sha256=hashlib.sha256(manifest_raw).hexdigest(),
    )
    bindings = manifest.get("source_bindings")
    expected_paths = tuple(
        str(row["path"])
        for row in actual
        if str(row["path"]) != manifest_relative
    )
    if not _r31._source_binding_paths_are_exact_v1(bindings, expected_paths):
        _fail("R4 source bindings do not equal changed paths minus manifest")
    verified: list[dict[str, object]] = []
    assert isinstance(bindings, list)
    for row in bindings:
        if type(row) is not dict or set(row) != {
            "path", "parent_sha256_or_null", "r4_sha256"
        }:
            _fail("R4 source-binding row differs")
        path = row.get("path")
        old_hash = row.get("parent_sha256_or_null")
        new_hash = row.get("r4_sha256")
        if (
            type(path) is not str
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
            or (
                old_hash is not None
                and (type(old_hash) is not str or _HEX_64.fullmatch(old_hash) is None)
            )
            or type(new_hash) is not str
            or _HEX_64.fullmatch(new_hash) is None
        ):
            _fail("R4 source-binding value differs")
        _r31._verify_changed_worktree_blob_v1(
            repository_root=repository_root,
            head=head,
            relative=path,
            expected_sha256=new_hash,
        )
        if old_hash is None:
            probe = subprocess.run(
                [
                    str(FORMAL_GIT_EXECUTABLE), "cat-file", "-e",
                    f"{R31_AMENDMENT_COMMIT}:{path}",
                ],
                cwd=repository_root.resolve(strict=True),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=60,
                env=dict(FORMAL_GIT_ENVIRONMENT_V1),
            )
            if probe.returncode == 0:
                _fail(f"R4 source unexpectedly existed in R3.1: {path}")
        elif hashlib.sha256(
            _git(repository_root, ("show", f"{R31_AMENDMENT_COMMIT}:{path}"))
        ).hexdigest() != old_hash:
            _fail(f"parent R3.1 source blob hash differs: {path}")
        verified.append(dict(row))
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-preflight/1",
        "amendment_commit": head,
        "sole_parent_commit": R31_AMENDMENT_COMMIT,
        "parent_amendment_commit": R31_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 4,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r31_terminal_chain_root_sha256": R31_TERMINAL_CHAIN_ROOT_SHA256,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "source_bindings": verified,
        "repository_clean": True,
        "exact_changed_paths_verified": True,
        "changed_worktree_blobs_equal_head": True,
        "changed_path_index_flags_normal": True,
        "formal_identity_entropy_draw_count": 0,
        "m3_start_allowed": False,
        "fixed_audit_directory": FIXED_R4_AUDIT_DIRECTORY.as_posix(),
    }


def _require_existing_audit_directory(path: Path, repository_root: Path) -> Path:
    if path.is_symlink():
        _fail("R4 audit directory may not be a symlink")
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    repository = repository_root.resolve(strict=True)
    if (
        resolved != FIXED_R4_AUDIT_DIRECTORY
        or resolved == repository
        or repository in resolved.parents
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
    ):
        _fail("R4 audit directory is not the fixed caller-owned mode-0700 path")
    return resolved


def _create_or_resume_prepare_audit_directory(
    path: Path, repository_root: Path
) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute != FIXED_R4_AUDIT_DIRECTORY:
        _fail("R4 audit directory differs from fixed attempt-4 path")
    repository = repository_root.resolve(strict=True)
    parent = absolute.parent.resolve(strict=True)
    if absolute == repository or repository in absolute.parents:
        _fail("R4 audit directory must be repository-external")
    if absolute.is_symlink():
        _fail("R4 attempt-4 audit directory may not be a symlink")
    if not absolute.exists():
        os.mkdir(absolute, 0o700)
        os.chmod(absolute, 0o700)
        _r31._fsync_directory_v1(parent)
    return _require_existing_audit_directory(absolute, repository_root)


def _install_prepare_record_v1(path: Path, raw: bytes) -> None:
    try:
        _r31._install_prepare_record_v1(path, raw)
    except _r31.A8R3RecoveryAmendmentError as exc:
        _fail("R4 preparation install rejected: " + exc.detail)


def _install_exact_audit_record_v1(
    path: Path, expected: Mapping[str, object], raw: bytes
) -> None:
    try:
        _r31._install_exact_audit_record_v1(path, expected, raw)
    except _r31.A8R3RecoveryAmendmentError as exc:
        _fail("R4 exact audit install rejected: " + exc.detail)


def _discard_non_authoritative_next_v1(path: Path) -> None:
    try:
        _r31._discard_non_authoritative_next_v1(path)
    except _r31.A8R3RecoveryAmendmentError as exc:
        _fail("R4 hidden audit cleanup rejected: " + exc.detail)


def _exact_audit_record_is_visible_v1(path: Path, raw: bytes) -> bool:
    try:
        return _r31._exact_audit_record_is_visible_v1(path, raw)
    except _r31.A8R3RecoveryAmendmentError as exc:
        _fail("R4 visible audit check rejected: " + exc.detail)


def _validation_request_from_incident_v1(
    incident: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    try:
        return _r31._validation_request_from_incident_v1(incident)
    except _r31.A8R3RecoveryAmendmentError as exc:
        _fail("R4 validation-request construction rejected: " + exc.detail)


def _run_a8_validator_v1(
    request: Mapping[str, object],
) -> tuple[dict[str, object], bytes]:
    try:
        return _r31._run_a8_validator_v1(request)
    except _r31.A8R3RecoveryAmendmentError as exc:
        _fail("R4 isolated A8 validation rejected: " + exc.detail)


def _validate_runtime_artifacts_before_attempt_v1(
    *, rust_formal_replay_binary: Path, rust_bridge_dag_replay_binary: Path,
    rust_bridge_dag_qualification_report: Path,
    expected_bindings: object,
) -> tuple[dict[str, object], ...]:
    try:
        return _r31._r2._validate_runtime_artifacts_before_attempt_v1(
            rust_formal_replay_binary=rust_formal_replay_binary,
            rust_bridge_dag_replay_binary=rust_bridge_dag_replay_binary,
            rust_bridge_dag_qualification_report=(
                rust_bridge_dag_qualification_report
            ),
            expected_bindings=expected_bindings,
        )
    except _r31._r2.A8R2RecoveryAmendmentError as exc:
        _fail("R4 runtime artifact validation rejected: " + exc.detail)


def _r31_terminal_chain_snapshot_v1() -> tuple[dict[str, object], ...]:
    audit = R31_TERMINAL_AUDIT_DIRECTORY
    if audit.is_symlink():
        _fail("R3.1 terminal audit directory may not be a symlink")
    metadata = audit.stat()
    order = (
        "preflight.json", "incident-diagnostic.json",
        "a8-validation-receipt.json", "authorization-request.json",
        "authorization.json", "attempt-start.json", "failure.json",
    )
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
        or {path.name for path in audit.iterdir()} != set(order)
    ):
        _fail("R3.1 terminal audit is not the exact seven-record chain")
    records: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for name in order:
        value, raw = _r31._r2._read_canonical_audit(audit / name)
        item = (audit / name).stat()
        raw_sha = hashlib.sha256(raw).hexdigest()
        if (
            raw_sha != R31_TERMINAL_AUDIT_RAW_SHA256[name]
            or value.get("receipt_sha256")
            != R31_TERMINAL_AUDIT_RECEIPT_SHA256[name]
            or stat.S_IMODE(item.st_mode) != 0o600
            or item.st_uid != os.getuid()
            or item.st_gid != os.getgid()
            or item.st_nlink != 1
        ):
            _fail(f"R3.1 terminal audit identity differs: {name}")
        records[name] = value
        rows.append(
            {
                "name": name,
                "raw_sha256": raw_sha,
                "receipt_sha256": value["receipt_sha256"],
                "size_bytes": item.st_size,
                "mode_octal": "0600",
            }
        )
    preflight = records["preflight.json"]
    incident = records["incident-diagnostic.json"]
    validation = records["a8-validation-receipt.json"]
    request = records["authorization-request.json"]
    authorization = records["authorization.json"]
    attempt = records["attempt-start.json"]
    failure = records["failure.json"]
    common = {
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
    }
    if (
        any(
            any(record.get(key) != expected for key, expected in common.items())
            for record in records.values()
        )
        or any(
            records[name].get("recovery_attempt_ordinal") != 3
            for name in order
            if name != "a8-validation-receipt.json"
        )
        or preflight.get("schema")
        != f"{_r31.AUDIT_SCHEMA_PREFIX}-preflight/1"
        or preflight.get("amendment_commit") != R31_AMENDMENT_COMMIT
        or preflight.get("sole_parent_commit") != R3_AMENDMENT_COMMIT
        or preflight.get("parent_amendment_commit") != R3_AMENDMENT_COMMIT
        or preflight.get("manifest_sha256")
        != "46eaeaaaf29164066dd94bd75bd042d724778f14cbd1e18c41c4821035fc2cf9"
        or preflight.get("authorization_revision_id")
        != _r31.AUTHORIZATION_REVISION_ID
        or preflight.get("formal_identity_entropy_draw_count") != 0
        or preflight.get("m3_start_allowed") is not False
        or incident.get("schema")
        != f"{_r31.AUDIT_SCHEMA_PREFIX}-incident-diagnostic/1"
        or incident.get("authorization_revision_id")
        != _r31.AUTHORIZATION_REVISION_ID
        or incident.get("continuation_action") != _r31.CONTINUATION_ACTION
        or incident.get("r3_preattempt_prefix_root_sha256")
        != _r31.R3_PREATTEMPT_PREFIX_ROOT_SHA256
        or incident.get("raw_seed_bytes_read_by_r3_orchestrator") is not False
        or incident.get("raw_seed_sha256_computed") is not False
        or incident.get("formal_identity_entropy_draw_count") != 0
        or validation.get("schema")
        != "hegel-phase3-m25-a8-r3-a8-validation-receipt/1"
        or validation.get("formal_identity_entropy_draw_count") != 0
        or validation.get("raw_seed_bytes_read") is not False
        or validation.get("raw_seed_sha256_computed") is not False
        or validation.get("m3_start_invoked") is not False
        or request.get("schema")
        != f"{_r31.AUDIT_SCHEMA_PREFIX}-authorization-request/1"
        or request.get("amendment_commit") != R31_AMENDMENT_COMMIT
        or request.get("preflight_sha256")
        != R31_TERMINAL_AUDIT_RAW_SHA256["preflight.json"]
        or request.get("incident_diagnostic_sha256")
        != R31_TERMINAL_AUDIT_RAW_SHA256["incident-diagnostic.json"]
        or request.get("a8_validation_receipt_sha256")
        != R31_TERMINAL_AUDIT_RAW_SHA256["a8-validation-receipt.json"]
        or request.get("authorization_revision_id")
        != _r31.AUTHORIZATION_REVISION_ID
        or request.get("continuation_action") != _r31.CONTINUATION_ACTION
        or request.get("ordinary_execute_allowed") is not False
        or request.get("redraw_allowed") is not False
        or request.get("m3_start_allowed") is not False
        or request.get("formal_identity_entropy_draw_count") != 0
        or authorization.get("schema")
        != f"{_r31.AUDIT_SCHEMA_PREFIX}-authorization/1"
        or authorization.get("amendment_commit") != R31_AMENDMENT_COMMIT
        or authorization.get("preflight_sha256")
        != R31_TERMINAL_AUDIT_RAW_SHA256["preflight.json"]
        or authorization.get("incident_diagnostic_sha256")
        != R31_TERMINAL_AUDIT_RAW_SHA256["incident-diagnostic.json"]
        or authorization.get("a8_validation_receipt_sha256")
        != R31_TERMINAL_AUDIT_RAW_SHA256["a8-validation-receipt.json"]
        or authorization.get("authorization_request_sha256")
        != R31_TERMINAL_AUDIT_RAW_SHA256["authorization-request.json"]
        or authorization.get("authorization_revision_id")
        != _r31.AUTHORIZATION_REVISION_ID
        or authorization.get("continuation_action")
        != _r31.CONTINUATION_ACTION
        or authorization.get("authorization_actor") != "PROJECT_OWNER"
        or authorization.get("owner_authorized_fixed_transaction_only")
        is not True
        or authorization.get("ordinary_execute_invoked") is not False
        or authorization.get("redraw_allowed") is not False
        or authorization.get("m3_start_allowed") is not False
        or authorization.get("formal_identity_entropy_draw_count") != 0
        or attempt.get("schema")
        != f"{_r31.AUDIT_SCHEMA_PREFIX}-attempt-start/1"
        or attempt.get("amendment_commit") != R31_AMENDMENT_COMMIT
        or attempt.get("authorization_revision_id")
        != _r31.AUTHORIZATION_REVISION_ID
        or attempt.get("authorization_sha256")
        != R31_TERMINAL_AUDIT_RAW_SHA256["authorization.json"]
        or attempt.get("a8_validation_receipt_sha256")
        != R31_TERMINAL_AUDIT_RAW_SHA256["a8-validation-receipt.json"]
        or attempt.get("formal_identity_entropy_draw_count") != 0
        or attempt.get("raw_seed_bytes_read_by_r3_orchestrator") is not False
        or attempt.get("raw_seed_sha256_computed") is not False
        or attempt.get("m3_start_invoked") is not False
        or failure.get("schema")
        != f"{_r31.AUDIT_SCHEMA_PREFIX}-failure/1"
        or failure.get("amendment_commit") != R31_AMENDMENT_COMMIT
        or failure.get("attempt_start_sha256")
        != R31_TERMINAL_AUDIT_RAW_SHA256["attempt-start.json"]
        or failure.get("admission_sha256_or_null") is not None
        or failure.get("failure_phase") != "ATTEMPT_START_DURABILITY"
        or failure.get("failure_code") != _r31.FAIL_AMENDMENT
        or failure.get("failure_detail_sha256")
        != R31_FAILURE_DETAIL_SHA256
        or failure.get("a8_validation_receipt_sha256")
        != R31_TERMINAL_AUDIT_RAW_SHA256["a8-validation-receipt.json"]
        or failure.get("authorization_revision_id")
        != _r31.AUTHORIZATION_REVISION_ID
        or failure.get("formal_identity_entropy_draw_count") != 0
        or any(
            failure.get(name) is not False
            for name in (
                "raw_seed_bytes_read_by_r3_orchestrator",
                "raw_seed_sha256_computed", "m3_start_invoked",
            )
        )
    ):
        _fail("R3.1 terminal audit provenance or failure fields differ")
    if hashlib.sha256(_canonical_json(rows)).hexdigest() != (
        R31_TERMINAL_CHAIN_ROOT_SHA256
    ):
        _fail("R3.1 terminal chain root differs")
    return tuple(rows)


def _build_incident_diagnostic_v1(
    *, custody_directory: Path, public_evidence_path: Path,
    public_promotion_path: Path,
) -> dict[str, object]:
    try:
        base = dict(
            _r31._build_incident_diagnostic_v1(
                custody_directory=custody_directory,
                public_evidence_path=public_evidence_path,
                public_promotion_path=public_promotion_path,
            )
        )
    except _r31.A8R3RecoveryAmendmentError as exc:
        _fail("R3.1 continuity verifier rejected R4 incident: " + exc.detail)
    rows = _r31_terminal_chain_snapshot_v1()
    stored_attempt, stored_attempt_raw = _r31._r2._read_canonical_audit(
        R31_TERMINAL_AUDIT_DIRECTORY / "attempt-start.json"
    )
    runtime_rows = _validate_runtime_artifacts_before_attempt_v1(
        rust_formal_replay_binary=Path(
            str(_r31.FIXED_RUNTIME_ARTIFACTS[0]["path"])
        ),
        rust_bridge_dag_replay_binary=Path(
            str(_r31.FIXED_RUNTIME_ARTIFACTS[1]["path"])
        ),
        rust_bridge_dag_qualification_report=Path(
            str(_r31.FIXED_RUNTIME_ARTIFACTS[2]["path"])
        ),
        expected_bindings=base.get("runtime_artifact_bindings"),
    )
    rebuilt_attempt = dict(stored_attempt)
    rebuilt_attempt["runtime_artifact_metadata"] = runtime_rows
    mismatch_fields = tuple(
        sorted(
            key
            for key in set(stored_attempt) | set(rebuilt_attempt)
            if stored_attempt.get(key) != rebuilt_attempt.get(key)
        )
    )
    if (
        stored_attempt_raw != _canonical_json(rebuilt_attempt)
        or stored_attempt == rebuilt_attempt
        or mismatch_fields != ("runtime_artifact_metadata",)
    ):
        _fail("R3.1 attempt-start representation defect evidence differs")
    base["schema"] = f"{AUDIT_SCHEMA_PREFIX}-incident-diagnostic/1"
    base["continuation_action"] = CONTINUATION_ACTION
    base["authorization_revision_id"] = AUTHORIZATION_REVISION_ID
    base["recovery_attempt_ordinal"] = 4
    base["r31_amendment_commit"] = R31_AMENDMENT_COMMIT
    base["r31_terminal_audit_directory"] = (
        R31_TERMINAL_AUDIT_DIRECTORY.as_posix()
    )
    base["r31_terminal_chain"] = rows
    base["r31_terminal_chain_root_sha256"] = R31_TERMINAL_CHAIN_ROOT_SHA256
    base["r31_attempt_start_raw_sha256"] = (
        R31_TERMINAL_AUDIT_RAW_SHA256["attempt-start.json"]
    )
    base["r31_failure_raw_sha256"] = (
        R31_TERMINAL_AUDIT_RAW_SHA256["failure.json"]
    )
    base["r31_failure_receipt_sha256"] = (
        R31_TERMINAL_AUDIT_RECEIPT_SHA256["failure.json"]
    )
    base["r31_admission_sha256_or_null"] = None
    base["r31_failure_phase"] = "ATTEMPT_START_DURABILITY"
    base["r31_failure_detail_sha256"] = R31_FAILURE_DETAIL_SHA256
    base["r31_defect_code"] = R31_DEFECT_CODE
    base["r31_attempt_start_canonical_bytes_valid"] = True
    base["r31_attempt_start_python_object_equality"] = False
    base["r31_attempt_start_representation_mismatch_fields"] = mismatch_fields
    base.pop("raw_seed_bytes_read_by_r3_orchestrator", None)
    base["raw_seed_bytes_read_by_r4_orchestrator"] = False
    return base


def _build_source_admission_v1(
    *, amendment_commit: object, incident_raw: bytes, validation_raw: bytes,
    validation: Mapping[str, object], unchanged_inputs: Mapping[str, str],
) -> dict[str, object]:
    if (
        type(amendment_commit) is not str
        or _HEX_40.fullmatch(amendment_commit) is None
    ):
        _fail("R4 source admission amendment commit differs")
    if (
        len(unchanged_inputs) != _r31.EXPECTED_UNCHANGED_A8_INPUT_COUNT
        or hashlib.sha256(_executor_canonical_json(unchanged_inputs)).hexdigest()
        != _r31.EXPECTED_UNCHANGED_A8_INPUT_ROOT
    ):
        _fail("R4 source admission unchanged-input root differs")
    return {
        "schema": "hegel-phase3-m25-a8-r4-source-admission/1",
        "basis_commit": A8_BASIS_COMMIT,
        "r1_amendment_commit": R1_AMENDMENT_COMMIT,
        "r2_amendment_commit": R2_AMENDMENT_COMMIT,
        "r3_amendment_commit": R3_AMENDMENT_COMMIT,
        "r31_amendment_commit": R31_AMENDMENT_COMMIT,
        "r4_amendment_commit": amendment_commit,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 4,
        "continuation_action": SOURCE_ADMISSION_CONTINUATION_ACTION,
        "r1_failure_raw_sha256": _r31._r2.R1_AUDIT_RAW_SHA256["failure.json"],
        "r1_failure_receipt_sha256": _r31._r2.R1_FAILURE_RECEIPT_SHA256,
        "r2_terminal_chain_root_sha256": _r31.R2_TERMINAL_CHAIN_ROOT_SHA256,
        "r2_attempt_start_raw_sha256": _r31.R2_AUDIT_RAW_SHA256["attempt-start.json"],
        "r2_failure_raw_sha256": _r31.R2_AUDIT_RAW_SHA256["failure.json"],
        "r2_failure_receipt_sha256": _r31.R2_AUDIT_RECEIPT_SHA256["failure.json"],
        "r2_admission_sha256_or_null": None,
        "r3_preattempt_prefix_root_sha256": (
            _r31.R3_PREATTEMPT_PREFIX_ROOT_SHA256
        ),
        "r31_terminal_chain_root_sha256": R31_TERMINAL_CHAIN_ROOT_SHA256,
        "r31_attempt_start_raw_sha256": R31_TERMINAL_AUDIT_RAW_SHA256["attempt-start.json"],
        "r31_attempt_start_receipt_sha256": (
            R31_TERMINAL_AUDIT_RECEIPT_SHA256["attempt-start.json"]
        ),
        "r31_failure_raw_sha256": R31_TERMINAL_AUDIT_RAW_SHA256["failure.json"],
        "r31_failure_receipt_sha256": R31_TERMINAL_AUDIT_RECEIPT_SHA256["failure.json"],
        "r31_admission_sha256_or_null": None,
        "r31_failure_code": _r31.FAIL_AMENDMENT,
        "r31_failure_phase": "ATTEMPT_START_DURABILITY",
        "r31_failure_detail_sha256": R31_FAILURE_DETAIL_SHA256,
        "incident_diagnostic_sha256": hashlib.sha256(incident_raw).hexdigest(),
        "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
        "cross_basis_recovery_authorized": True,
        "formal_identity_entropy_draw_count": 0,
        "complete_seed_resume_only": True,
        "ordinary_execute_allowed": False,
        "redraw_allowed": False,
        "m3_start_allowed": False,
        "prevalidated_report_basis": True,
        "prevalidated_transaction_bundle": True,
        "unchanged_a8_input_sha256": dict(unchanged_inputs),
        "unchanged_a8_input_sha256_root": _r31.EXPECTED_UNCHANGED_A8_INPUT_ROOT,
        "actor_report_sha256": validation["actor_report_sha256"],
        "errata_report_sha256": validation["errata_report_sha256"],
        "live_bundle_sha256": validation["live_bundle_sha256"],
    }


def _unchanged_a8_input_bindings_v1() -> dict[str, str]:
    bindings: dict[str, str] = {}
    repository = REPOSITORY_ROOT.resolve(strict=True)
    for path in _r31.REQUIRED_COMMIT_A_INPUTS:
        relative = path.resolve(strict=True).relative_to(repository).as_posix()
        if relative in R4_RUNTIME_EXCEPTION_PATHS:
            continue
        if path.is_symlink():
            _fail(f"unchanged A8 runtime input is a symlink: {relative}")
        current = path.read_bytes()
        frozen = _git(repository, ("show", f"{A8_BASIS_COMMIT}:{relative}"))
        if current != frozen:
            _fail(f"unchanged A8 runtime input differs: {relative}")
        bindings[relative] = hashlib.sha256(current).hexdigest()
    root = hashlib.sha256(_executor_canonical_json(bindings)).hexdigest()
    if (
        len(bindings) != _r31.EXPECTED_UNCHANGED_A8_INPUT_COUNT
        or root != _r31.EXPECTED_UNCHANGED_A8_INPUT_ROOT
    ):
        _fail("unchanged A8 runtime closure/root differs")
    return bindings


def _authorization_request_fields(
    *, amendment_commit: object, preflight_raw: bytes, incident_raw: bytes,
    validation_raw: bytes,
) -> dict[str, object]:
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-authorization-request/1",
        "amendment_commit": amendment_commit,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 4,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r31_terminal_chain_root_sha256": R31_TERMINAL_CHAIN_ROOT_SHA256,
        "r31_attempt_start_raw_sha256": (
            R31_TERMINAL_AUDIT_RAW_SHA256["attempt-start.json"]
        ),
        "r31_failure_raw_sha256": R31_TERMINAL_AUDIT_RAW_SHA256["failure.json"],
        "r31_failure_receipt_sha256": (
            R31_TERMINAL_AUDIT_RECEIPT_SHA256["failure.json"]
        ),
        "r31_admission_sha256_or_null": None,
        "r31_defect_code": R31_DEFECT_CODE,
        "continuation_action": CONTINUATION_ACTION,
        "preflight_sha256": hashlib.sha256(preflight_raw).hexdigest(),
        "incident_diagnostic_sha256": hashlib.sha256(incident_raw).hexdigest(),
        "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
        "requested_action": "COMPLETE_ONLY_REAL_PENDING_RESUME",
        "ordinary_execute_allowed": False,
        "redraw_allowed": False,
        "abort_allowed": False,
        "poststage_recovery_allowed": False,
        "formal_identity_entropy_draw_count": 0,
        "m3_start_allowed": False,
    }


def prepare_fixed_a8_r4_authorization_v1(
    *, audit_directory: Path, custody_directory: Path,
    public_evidence_path: Path, public_promotion_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> None:
    preflight = inspect_r4_source_preflight_v1(
        repository_root=repository_root, manifest_path=manifest_path
    )
    incident = _build_incident_diagnostic_v1(
        custody_directory=custody_directory,
        public_evidence_path=public_evidence_path,
        public_promotion_path=public_promotion_path,
    )
    request, _actor, _errata, _bundle = (
        _validation_request_from_incident_v1(incident)
    )
    _validation, validation_raw = _run_a8_validator_v1(request)
    audit = _create_or_resume_prepare_audit_directory(
        audit_directory, repository_root
    )
    order = (
        "preflight.json", "incident-diagnostic.json",
        "a8-validation-receipt.json", "authorization-request.json",
    )
    observed = {path.name for path in audit.iterdir()}
    allowed = set(order) | {"." + name + ".next" for name in order}
    if not observed.issubset(allowed):
        _fail("R4 preparation audit contains a non-prefix record")
    visible = [name in observed for name in order]
    if any(
        visible[index] and not all(visible[:index])
        for index in range(len(order))
    ):
        _fail("R4 preparation visible records are not an exact prefix")
    preflight_raw = _receipt_record_bytes_v1(preflight)
    incident_raw = _receipt_record_bytes_v1(incident)
    request_raw = _receipt_record_bytes_v1(
        _authorization_request_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
        )
    )
    for name, payload in zip(
        order,
        (preflight_raw, incident_raw, validation_raw, request_raw),
        strict=True,
    ):
        _install_prepare_record_v1(audit / name, payload)


def _expected_authorization_fields(
    *, amendment_commit: object, preflight_raw: bytes, incident_raw: bytes,
    validation_raw: bytes, request_raw: bytes,
) -> dict[str, object]:
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-authorization/1",
        "amendment_commit": amendment_commit,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 4,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r31_terminal_chain_root_sha256": R31_TERMINAL_CHAIN_ROOT_SHA256,
        "r31_attempt_start_raw_sha256": (
            R31_TERMINAL_AUDIT_RAW_SHA256["attempt-start.json"]
        ),
        "r31_failure_raw_sha256": R31_TERMINAL_AUDIT_RAW_SHA256["failure.json"],
        "r31_failure_receipt_sha256": (
            R31_TERMINAL_AUDIT_RECEIPT_SHA256["failure.json"]
        ),
        "r31_admission_sha256_or_null": None,
        "r31_defect_code": R31_DEFECT_CODE,
        "continuation_action": CONTINUATION_ACTION,
        "preflight_sha256": hashlib.sha256(preflight_raw).hexdigest(),
        "incident_diagnostic_sha256": hashlib.sha256(incident_raw).hexdigest(),
        "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
        "authorization_request_sha256": hashlib.sha256(request_raw).hexdigest(),
        "authorization_actor": "PROJECT_OWNER",
        "owner_authorized_fixed_transaction_only": True,
        "ordinary_execute_invoked": False,
        "redraw_allowed": False,
        "abort_allowed": False,
        "poststage_recovery_allowed": False,
        "formal_identity_entropy_draw_count": 0,
        "m3_start_allowed": False,
    }


def write_fixed_a8_r4_owner_authorization_v1(
    *, audit_directory: Path, owner_confirmation: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    if owner_confirmation != OWNER_CONFIRMATION:
        _fail("R4 owner confirmation phrase differs")
    audit = _require_existing_audit_directory(audit_directory, repository_root)
    expected = {
        "preflight.json", "incident-diagnostic.json",
        "a8-validation-receipt.json", "authorization-request.json",
    }
    actual = {path.name for path in audit.iterdir()}
    if (
        not expected.issubset(actual)
        or not actual.issubset(
            expected | {"authorization.json", ".authorization.json.next"}
        )
    ):
        _fail("R4 pre-authorization audit path set differs")
    preflight, preflight_raw = _r31._r2._read_canonical_audit(
        audit / "preflight.json"
    )
    _incident, incident_raw = _r31._r2._read_canonical_audit(
        audit / "incident-diagnostic.json"
    )
    _validation, validation_raw, _row = _r31._r2._read_canonical_regular(
        audit / "a8-validation-receipt.json", mode=0o600
    )
    request, request_raw = _r31._r2._read_canonical_audit(
        audit / "authorization-request.json"
    )
    expected_request = _r31._r2._with_receipt_sha256(
        _authorization_request_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
        )
    )
    if request != expected_request:
        _fail("R4 authorization request differs")
    authorization_raw = _receipt_record_bytes_v1(
        _expected_authorization_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
            request_raw=request_raw,
        )
    )
    _install_prepare_record_v1(audit / "authorization.json", authorization_raw)


def _validate_final_publication_v1(
    *, payload: dict[str, object], promotion: dict[str, object],
    custody_directory: Path, stage_directory: Path,
    public_evidence_path: Path, public_promotion_path: Path,
    replay: Callable[[Mapping[str, object]], dict[str, object]],
) -> dict[str, object]:
    try:
        return _r31._validate_final_publication_v1(
            payload=payload,
            promotion=promotion,
            custody_directory=custody_directory,
            stage_directory=stage_directory,
            public_evidence_path=public_evidence_path,
            public_promotion_path=public_promotion_path,
            replay=replay,
        )
    except _r31.A8R3RecoveryAmendmentError as exc:
        _fail("R4 final publication replay rejected: " + exc.detail)


def execute_fixed_a8_r4_recovery_v1(
    *, custody_directory: Path, rust_formal_replay_binary: Path,
    rust_bridge_dag_replay_binary: Path,
    rust_bridge_dag_qualification_report: Path,
    public_evidence_path: Path, public_promotion_path: Path,
    audit_directory: Path,
    repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> tuple[dict[str, object], dict[str, object]]:
    preflight_now = inspect_r4_source_preflight_v1(
        repository_root=repository_root, manifest_path=manifest_path
    )
    audit = _require_existing_audit_directory(audit_directory, repository_root)
    terminal_names = (
        "attempt-start.json", "admission.json", "failure.json", "finalize.json"
    )
    if any(
        (audit / name).exists() or (audit / name).is_symlink()
        for name in terminal_names
    ):
        _fail("R4 attempt-4 was already consumed or has a terminal record")
    expected_before = {
        "preflight.json", "incident-diagnostic.json",
        "a8-validation-receipt.json", "authorization-request.json",
        "authorization.json",
    }
    observed_before = {path.name for path in audit.iterdir()}
    if (
        not expected_before.issubset(observed_before)
        or not observed_before.issubset(
            expected_before | {".attempt-start.json.next"}
        )
    ):
        _fail("R4 pre-attempt audit path set differs")
    preflight, preflight_raw = _r31._r2._read_canonical_audit(
        audit / "preflight.json"
    )
    _incident, incident_raw = _r31._r2._read_canonical_audit(
        audit / "incident-diagnostic.json"
    )
    validation, validation_raw, _validation_row = (
        _r31._r2._read_canonical_regular(
            audit / "a8-validation-receipt.json", mode=0o600
        )
    )
    request, request_raw = _r31._r2._read_canonical_audit(
        audit / "authorization-request.json"
    )
    authorization, authorization_raw = _r31._r2._read_canonical_audit(
        audit / "authorization.json"
    )
    if preflight != _r31._r2._with_receipt_sha256(preflight_now):
        _fail("stored R4 preflight differs from current clean R4 source")
    incident_now = _build_incident_diagnostic_v1(
        custody_directory=custody_directory,
        public_evidence_path=public_evidence_path,
        public_promotion_path=public_promotion_path,
    )
    if incident_raw != _receipt_record_bytes_v1(incident_now):
        _fail("stored R4 incident canonical bytes differ before attempt")
    validation_request, actor_report, errata_report, _expected_bundle = (
        _validation_request_from_incident_v1(incident_now)
    )
    validation_now, validation_now_raw = _run_a8_validator_v1(
        validation_request
    )
    if validation_now_raw != validation_raw or validation_now != validation:
        _fail("prepare/execute isolated A8 validation receipts differ")
    if request != _r31._r2._with_receipt_sha256(
        _authorization_request_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
        )
    ):
        _fail("stored R4 authorization request differs")
    if authorization != _r31._r2._with_receipt_sha256(
        _expected_authorization_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
            request_raw=request_raw,
        )
    ):
        _fail("stored R4 owner authorization differs")
    runtime_rows = _validate_runtime_artifacts_before_attempt_v1(
        rust_formal_replay_binary=rust_formal_replay_binary,
        rust_bridge_dag_replay_binary=rust_bridge_dag_replay_binary,
        rust_bridge_dag_qualification_report=rust_bridge_dag_qualification_report,
        expected_bindings=incident_now.get("runtime_artifact_bindings"),
    )
    unchanged_inputs = _unchanged_a8_input_bindings_v1()

    actors: A8R1RecoveryDockerActorsV1 | None = None
    attempt_start_raw: bytes | None = None
    admission_raw: bytes | None = None
    finalize_raw: bytes | None = None
    failure_phase = "PRE_ATTEMPT_ACQUIRE"
    try:
        actors = A8R1RecoveryDockerActorsV1(
            basis_commit=A8_BASIS_COMMIT,
            custody_directory=custody_directory,
            rust_formal_replay_binary=rust_formal_replay_binary,
            rust_bridge_dag_replay_binary=rust_bridge_dag_replay_binary,
            rust_bridge_dag_qualification_report=(
                rust_bridge_dag_qualification_report
            ),
            timestamp=0,
        )
        with acquire_pending_ceremony_recovery_v1(
            custody_directory=custody_directory,
            public_evidence_path=public_evidence_path,
            public_promotion_path=public_promotion_path,
            actors=actors,
        ) as recovery:
            if (
                recovery.basis_commit != A8_BASIS_COMMIT
                or recovery.run_id != FIXED_RUN_ID
                or recovery.ledger_id != FIXED_LEDGER_ID
                or recovery.marker_snapshot.state != "PENDING"
                or recovery.journal_state != "RESERVED"
            ):
                _fail(
                    "R4 acquired recovery is not the fixed A8 "
                    "PENDING/RESERVED transaction"
                )
            actors.timestamp = recovery.marker_snapshot.created_at_unix_seconds
            if (
                _canonical_json(_transport(recovery.prestage_intent_fields.get(
                    "actor_qualification_report"
                ))) != _canonical_json(actor_report)
                or _canonical_json(_transport(recovery.prestage_intent_fields.get(
                    "errata_qualification_report"
                ))) != _canonical_json(errata_report)
            ):
                _fail("R4 acquired diagnostic reports differ from A8 receipt")
            source_admission = _build_source_admission_v1(
                amendment_commit=preflight["amendment_commit"],
                incident_raw=incident_raw,
                validation_raw=validation_raw,
                validation=validation,
                unchanged_inputs=unchanged_inputs,
            )
            failure_phase = "ATTEMPT_START_DURABILITY"
            attempt, attempt_start_raw = _r31._build_exact_audit_record_v1(
                {
                    "schema": f"{AUDIT_SCHEMA_PREFIX}-attempt-start/1",
                    "amendment_commit": preflight["amendment_commit"],
                    "formal_repository_commit": A8_BASIS_COMMIT,
                    "run_id_hex": FIXED_RUN_ID_HEX,
                    "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                    "recovery_attempt_ordinal": 4,
                    "authorization_revision_id": AUTHORIZATION_REVISION_ID,
                    "r31_terminal_chain_root_sha256": (
                        R31_TERMINAL_CHAIN_ROOT_SHA256
                    ),
                    "r31_attempt_start_raw_sha256": (
                        R31_TERMINAL_AUDIT_RAW_SHA256["attempt-start.json"]
                    ),
                    "r31_failure_raw_sha256": (
                        R31_TERMINAL_AUDIT_RAW_SHA256["failure.json"]
                    ),
                    "r31_failure_receipt_sha256": (
                        R31_TERMINAL_AUDIT_RECEIPT_SHA256["failure.json"]
                    ),
                    "r31_admission_sha256_or_null": None,
                    "r31_defect_code": R31_DEFECT_CODE,
                    "continuation_action": CONTINUATION_ACTION,
                    "authorization_sha256": hashlib.sha256(
                        authorization_raw
                    ).hexdigest(),
                    "a8_validation_receipt_sha256": hashlib.sha256(
                        validation_raw
                    ).hexdigest(),
                    "runtime_artifact_metadata": runtime_rows,
                    "complete_seed_resume_only": True,
                    "accepted_worker_mode": "REAL_PENDING_RESUME",
                    "formal_identity_entropy_draw_count": 0,
                    "ordinary_execute_invoked": False,
                    "m3_start_invoked": False,
                    "raw_seed_bytes_read_by_r4_orchestrator": False,
                    "raw_seed_sha256_computed": False,
                }
            )
            _install_exact_audit_record_v1(
                audit / "attempt-start.json", attempt, attempt_start_raw
            )
            failure_phase = "SOURCE_ADMISSION_DURABILITY"
            admission, admission_raw = _r31._build_exact_audit_record_v1(
                {
                    "schema": f"{AUDIT_SCHEMA_PREFIX}-admission/1",
                    "amendment_commit": preflight["amendment_commit"],
                    "formal_repository_commit": A8_BASIS_COMMIT,
                    "run_id_hex": FIXED_RUN_ID_HEX,
                    "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                    "recovery_attempt_ordinal": 4,
                    "authorization_revision_id": AUTHORIZATION_REVISION_ID,
                    "r31_terminal_chain_root_sha256": (
                        R31_TERMINAL_CHAIN_ROOT_SHA256
                    ),
                    "r31_failure_raw_sha256": (
                        R31_TERMINAL_AUDIT_RAW_SHA256["failure.json"]
                    ),
                    "attempt_start_sha256": hashlib.sha256(
                        attempt_start_raw
                    ).hexdigest(),
                    "authorization_sha256": hashlib.sha256(
                        authorization_raw
                    ).hexdigest(),
                    "a8_validation_receipt_sha256": hashlib.sha256(
                        validation_raw
                    ).hexdigest(),
                    "source_admission": source_admission,
                    "source_admission_sha256": hashlib.sha256(
                        _executor_canonical_json(source_admission)
                    ).hexdigest(),
                    "complete_seed_resume_only": True,
                    "accepted_worker_mode": "REAL_PENDING_RESUME",
                    "formal_identity_entropy_draw_count": 0,
                    "raw_seed_bytes_read_by_r4_orchestrator": False,
                    "raw_seed_sha256_computed": False,
                    "m3_start_invoked": False,
                }
            )
            _install_exact_audit_record_v1(
                audit / "admission.json", admission, admission_raw
            )

            def guard(candidate: PendingCeremonyRecoveryV1) -> Mapping[str, object]:
                if candidate is not recovery:
                    raise FormalContainerExecutorError(
                        FAIL_RECOVERY_SOURCE_ADMISSION,
                        "R4 admission may authorize only the acquired recovery object",
                    )
                return source_admission

            failure_phase = "COMPLETE_ONLY_FORMAL_CORE"
            payload, promotion = _continue_pre_stage_pending_recovery_core_v1(
                recovery=recovery,
                actors=actors,
                source_admission_guard=guard,
                complete_seed_resume_only=True,
                static_rust_binary_path=rust_formal_replay_binary,
            )
            stage_directory = recovery.stage_directory
            frozen_prestage_intent = dict(recovery.prestage_intent_fields)
        failure_phase = "FINAL_PUBLIC_REPLAY"

        def final_replay(candidate: Mapping[str, object]) -> dict[str, object]:
            return _replay_public_gate_evidence_with_fixed_a8_r3_basis_v1(
                candidate,
                source_admission=source_admission,
                prestage_intent_fields=frozen_prestage_intent,
            )

        final = _validate_final_publication_v1(
            payload=payload,
            promotion=promotion,
            custody_directory=custody_directory,
            stage_directory=stage_directory,
            public_evidence_path=public_evidence_path,
            public_promotion_path=public_promotion_path,
            replay=final_replay,
        )
        failure_phase = "FINALIZE_DURABILITY"
        finalize, finalize_raw = _r31._build_exact_audit_record_v1(
            {
                "schema": f"{AUDIT_SCHEMA_PREFIX}-finalize/1",
                "amendment_commit": preflight["amendment_commit"],
                "formal_repository_commit": A8_BASIS_COMMIT,
                "run_id_hex": FIXED_RUN_ID_HEX,
                "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                "recovery_attempt_ordinal": 4,
                "authorization_revision_id": AUTHORIZATION_REVISION_ID,
                "r31_terminal_chain_root_sha256": (
                    R31_TERMINAL_CHAIN_ROOT_SHA256
                ),
                "r31_failure_raw_sha256": (
                    R31_TERMINAL_AUDIT_RAW_SHA256["failure.json"]
                ),
                "attempt_start_sha256": hashlib.sha256(
                    attempt_start_raw
                ).hexdigest(),
                "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
                "a8_validation_receipt_sha256": hashlib.sha256(
                    validation_raw
                ).hexdigest(),
                **final,
                "formal_gates_after": 24,
                "child_state": "NOT_RUN",
                "m3_start_invoked": False,
                "accepted_worker_mode": "REAL_PENDING_RESUME",
                "raw_seed_bytes_read_by_r4_orchestrator": False,
                "raw_seed_sha256_computed": False,
                "formal_identity_entropy_draw_count": 0,
            }
        )
        _install_exact_audit_record_v1(
            audit / "finalize.json", finalize, finalize_raw
        )
        return payload, promotion
    except BaseException as exc:
        finalize_path = audit / "finalize.json"
        finalize_visible = (
            finalize_raw is not None
            and _exact_audit_record_is_visible_v1(finalize_path, finalize_raw)
        )
        if finalize_visible:
            _install_prepare_record_v1(finalize_path, finalize_raw)
            return payload, promotion
        if finalize_raw is not None:
            _discard_non_authoritative_next_v1(finalize_path)
        attempt_path = audit / "attempt-start.json"
        if (
            attempt_start_raw is not None
            and _exact_audit_record_is_visible_v1(
                attempt_path, attempt_start_raw
            )
        ):
            _install_prepare_record_v1(attempt_path, attempt_start_raw)
            failure_path = audit / "failure.json"
            if not failure_path.exists() and not failure_path.is_symlink():
                admission_visible = (
                    admission_raw is not None
                    and _exact_audit_record_is_visible_v1(
                        audit / "admission.json", admission_raw
                    )
                )
                if admission_visible:
                    _install_prepare_record_v1(
                        audit / "admission.json", admission_raw
                    )
                elif admission_raw is not None:
                    _discard_non_authoritative_next_v1(
                        audit / "admission.json"
                    )
                failure, failure_raw = _r31._build_exact_audit_record_v1(
                    {
                        "schema": f"{AUDIT_SCHEMA_PREFIX}-failure/1",
                        "amendment_commit": preflight["amendment_commit"],
                        "formal_repository_commit": A8_BASIS_COMMIT,
                        "run_id_hex": FIXED_RUN_ID_HEX,
                        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                        "recovery_attempt_ordinal": 4,
                        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
                        "r31_terminal_chain_root_sha256": (
                            R31_TERMINAL_CHAIN_ROOT_SHA256
                        ),
                        "r31_failure_raw_sha256": (
                            R31_TERMINAL_AUDIT_RAW_SHA256["failure.json"]
                        ),
                        "attempt_start_sha256": hashlib.sha256(
                            attempt_start_raw
                        ).hexdigest(),
                        "admission_sha256_or_null": (
                            None
                            if not admission_visible
                            else hashlib.sha256(admission_raw).hexdigest()
                        ),
                        "a8_validation_receipt_sha256": hashlib.sha256(
                            validation_raw
                        ).hexdigest(),
                        "failure_code": str(
                            getattr(exc, "code", type(exc).__name__)
                        ),
                        "failure_phase": failure_phase,
                        "failure_detail_sha256": hashlib.sha256(
                            str(exc).encode("utf-8", "replace")
                        ).hexdigest(),
                        "formal_identity_entropy_draw_count": 0,
                        "raw_seed_bytes_read_by_r4_orchestrator": False,
                        "raw_seed_sha256_computed": False,
                        "m3_start_invoked": False,
                    }
                )
                try:
                    _install_exact_audit_record_v1(
                        failure_path, failure, failure_raw
                    )
                except BaseException:
                    if not _exact_audit_record_is_visible_v1(
                        failure_path, failure_raw
                    ):
                        _discard_non_authoritative_next_v1(failure_path)
                    raise
        raise
    finally:
        if actors is not None:
            actors.close()


__all__ = [
    "A8R4RecoveryAmendmentError",
    "DEFAULT_MANIFEST_PATH",
    "execute_fixed_a8_r4_recovery_v1",
    "inspect_r4_source_preflight_v1",
    "prepare_fixed_a8_r4_authorization_v1",
    "write_fixed_a8_r4_owner_authorization_v1",
]
