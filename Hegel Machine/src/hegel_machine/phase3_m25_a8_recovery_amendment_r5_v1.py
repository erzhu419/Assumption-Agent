"""Attempt-5 recovery after the terminal R4r2 start/cleanup failure.

R4 revision 2 consumed attempt ordinal 4.  Its canonical failure record retained
only the aggregate cleanup error, so the original primary exception is not
recoverable from that record.  Non-canonical read-only code/daemon/inode
forensics and a seed-free seven-file synthetic replay indicate that no formal
actor container started and identify the start-order defect; that diagnosis
remains explicitly non-canonical historical evidence.

This module binds the exact eight-record R4r2 terminal chain and creates one
fresh attempt-5 namespace.  It never opens or hashes the retained raw seed,
never redraws formal identity, and never starts M3.
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

from . import phase3_m25_a8_recovery_amendment_r4_v1 as _r4
from .phase3_m25_a8_recovery_amendment_v1 import A8R1RecoveryDockerActorsV1
from . import phase3_m25_formal_container_executor_v1 as _executor
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
    PROJECT_ROOT / "config/phase3_m25_a8_recovery_amendment_r5_v1.json"
)
A8_BASIS_COMMIT: Final = _r4.A8_BASIS_COMMIT
R4_AMENDMENT_COMMIT: Final = "f24bae3c4fd1f4480e0aa9ecba69ac945779828d"
FIXED_RUN_ID_HEX: Final = _r4.FIXED_RUN_ID_HEX
FIXED_LEDGER_ID_HEX: Final = _r4.FIXED_LEDGER_ID_HEX
FIXED_RUN_ID: Final = _r4.FIXED_RUN_ID
FIXED_LEDGER_ID: Final = _r4.FIXED_LEDGER_ID
R4_TERMINAL_AUDIT_DIRECTORY: Final = _r4.FIXED_R4_AUDIT_DIRECTORY
FIXED_R5_AUDIT_DIRECTORY: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/"
    "phase3-m25-0af65964235390ce2bebefea7379eaa9c50eda24/"
    "recovery-audit-r5-e4af9f57c38fb298462ec628c4ed8a03-attempt-5"
)
MANIFEST_SCHEMA: Final = "hegel-phase3-m25-a8-recovery-amendment-r5/1"
AUDIT_SCHEMA_PREFIX: Final = "hegel-phase3-m25-a8-r5-recovery-audit"
SOURCE_ADMISSION_SCHEMA: Final = "hegel-phase3-m25-a8-r5-source-admission/1"
FAIL_AMENDMENT: Final = "FAIL_M25_A8_R5_RECOVERY_AMENDMENT"
OWNER_CONFIRMATION: Final = (
    "AUTHORIZE_A8_R5_ATTEMPT_5_RECOVERY_START_ORDER_AND_COMPOSITE_FAILURE_"
    "COMPLETE_ONLY_REAL_PENDING_RESUME"
)
CONTINUATION_ACTION: Final = (
    "POST_R4R2_TERMINAL_RECOVERY_START_ORDER_AND_COMPOSITE_FAILURE_CONTINUATION"
)
SOURCE_ADMISSION_CONTINUATION_ACTION: Final = "CODE_AMENDMENT_RECOVERY_CONTINUATION"
AUTHORIZATION_REVISION_ID: Final = (
    "R5_RECOVERY_START_ORDER_AND_COMPOSITE_FAILURE_EVIDENCE_V1"
)
R4_TERMINAL_AUDIT_ORDER: Final = (
    "preflight.json",
    "incident-diagnostic.json",
    "a8-validation-receipt.json",
    "authorization-request.json",
    "authorization.json",
    "attempt-start.json",
    "admission.json",
    "failure.json",
)
R4_TERMINAL_AUDIT_RAW_SHA256: Final = {
    "preflight.json": "70c0eb68ea1012abfbebb09976b2f013a973396b50765366851d8259deb39d88",
    "incident-diagnostic.json": "8d7ab9b7f42346b6d4dc74dcf471d424dca897c0cd5e1238ae53f386829b387c",
    "a8-validation-receipt.json": "ef18694aa41a78389cef2265eb121174f2e68548928f89f7fcad3f55fb261ee4",
    "authorization-request.json": "28c4c66a3af948de64a76da69fcd841b6b0b61f5433469b2ed0dcd2126d07552",
    "authorization.json": "8b8045c8dd0825ed72463a11f3991b8cb0867782c9f33accef32523b472bfacb",
    "attempt-start.json": "fbd2b8c3d9b9168fe97a9b08fde1d65467e0b2a07e6c3cf84645c44cb868dc52",
    "admission.json": "c4fe711ad4fce230a3a34fe0eb4511e1560c12b789f69b48d5df1798f907b89f",
    "failure.json": "b02b61161b3a083953ff687b0168aa0d1b545ae495d971100d25f38cbbd550d5",
}
R4_TERMINAL_AUDIT_RECEIPT_SHA256: Final = {
    "preflight.json": "fc9bc903aab8388e1f232c66a565a714227ff07fba28767f8ca4b6f0e773ec5d",
    "incident-diagnostic.json": "825fff41c912db2069aec220f5e82f99c4f83743f8b09be3901cb80246073230",
    "a8-validation-receipt.json": "83b1ad690914d9dfd5cd402d5c734a1250a3b450c9e0b3ecf4655cfb97c6ba47",
    "authorization-request.json": "a8e415fb0f9b58966f90470c334884e47120ebe4b43338a24629a700b4a12332",
    "authorization.json": "b2d455dfdc8c6de2353624edf190b4cdf08f63833ea8e7af88202045ff94f3ad",
    "attempt-start.json": "4644fdfb423a992bcea156141f9f689636b7c223742233bb7779fdb3f127461e",
    "admission.json": "752ef70b901d5238b7c7c23033e42af4047a729356b74f2dae760442fafd54e9",
    "failure.json": "2f3b2cc21e0cd88ead075c840b54864572311cb83d87f4af60b4866ffa3cc22e",
}
R4_TERMINAL_AUDIT_SIZE_BYTES: Final = {
    "preflight.json": 3392,
    "incident-diagnostic.json": 20966,
    "a8-validation-receipt.json": 14588,
    "authorization-request.json": 2131,
    "authorization.json": 2251,
    "attempt-start.json": 4313,
    "admission.json": 16770,
    "failure.json": 1820,
}
R4_TERMINAL_CHAIN_ROOT_SHA256: Final = (
    "9c0c0b8f05e97ec6b87c0ac9b4a36823f5338ce69053f442e9b1cbf1137f00d5"
)
R4_FAILURE_CODE: Final = "FAIL_M25_FORMAL_CONTAINER_RUNTIME"
R4_FAILURE_PHASE: Final = "COMPLETE_ONLY_FORMAL_CORE"
R4_RECORDED_MASKING_CLEANUP_DETAIL_SHA256: Final = (
    "8b0d1b49ee29954a00f06e36aef9b3dc585e2079ccfb16c116b89a7b027c3ac2"
)
R4_LEGACY_PRIMARY_STATUS: Final = "UNRECOVERABLE_FROM_CANONICAL_R4_RECORD"
R4_INFERRED_PRIMARY_EXCEPTION: Final = (
    "FAIL_M25_FORMAL_CONTAINER_RUNTIME: frozen runtime seccomp snapshot is absent"
)
R4_INFERRED_PRIMARY_EXCEPTION_SHA256: Final = (
    "e6fdc153a3354d552e62e2dd6ccc24a14d1df32346e4a54c9c9bab82ebd9e459"
)
R4_INFERRED_PRIMARY_DETAIL_SHA256: Final = (
    "97bfff75a02b68f52d46d404bb1a19471984bd00d52fc57b578fa76a1fcde229"
)
R4_INFERENCE_AUTHORITY: Final = "NON_CANONICAL_READ_ONLY_FORENSIC_INFERENCE"
R4_SYNTHETIC_REPRODUCTION: Final = {
    "schema": "hegel-phase3-m25-r4-failure-synthetic-reproduction/1",
    "attestation_status": "UNATTESTED_NONCANONICAL_DIAGNOSTIC",
    "custody_shape": "EXACT_7_NAMES_AND_MODES_SEED_SIZE_ONLY",
    "directory_mode_octal": "0700",
    "all_file_modes_octal": "0600",
    "flock_held_across_handoff_and_reclaim": True,
    "image_digest": "sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3",
    "pull_policy": "never",
    "network_mode": "none",
    "cap_drop_all_then_add_chown_only": True,
    "handoff_1000_to_65534_passed": True,
    "reclaim_65534_to_1000_passed": True,
    "class_state_reproduction_runtime_seccomp_path_was_null": True,
    "class_state_reproduction_exact_primary_sha256": (
        R4_INFERRED_PRIMARY_EXCEPTION_SHA256
    ),
    "raw_seed_used": False,
    "formal_recovery_invoked": False,
    "m3_start_invoked": False,
}
R5_RUNTIME_EXCEPTION_PATHS: Final = frozenset(
    {
        *_r4.R4_RUNTIME_EXCEPTION_PATHS,
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_amendment_r5_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_cli_r5_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_amendment_r6_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_cli_r6_v1.py",
    }
)
_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")


class A8R5RecoveryAmendmentError(FormalContainerExecutorError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(code, detail)


def _fail(detail: str) -> NoReturn:
    raise A8R5RecoveryAmendmentError(FAIL_AMENDMENT, detail)


def _canonical_json(value: object) -> bytes:
    return _r4._canonical_json(value)


def _receipt_record_bytes_v1(fields: Mapping[str, object]) -> bytes:
    return _canonical_json(_r4._r31._r2._with_receipt_sha256(fields))


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


def _manifest_fixed_policy_v1() -> dict[str, object]:
    return {
        "schema": MANIFEST_SCHEMA,
        "source_commit_selector": "HEAD",
        "sole_parent_commit": R4_AMENDMENT_COMMIT,
        "parent_amendment_commit": R4_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "fixed_run_id_hex": FIXED_RUN_ID_HEX,
        "fixed_ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 5,
        "fixed_r4_terminal_audit_directory": R4_TERMINAL_AUDIT_DIRECTORY.as_posix(),
        "fixed_r5_audit_directory": FIXED_R5_AUDIT_DIRECTORY.as_posix(),
        "r4_terminal_audit_raw_sha256": R4_TERMINAL_AUDIT_RAW_SHA256,
        "r4_terminal_audit_receipt_sha256": R4_TERMINAL_AUDIT_RECEIPT_SHA256,
        "r4_terminal_audit_size_bytes": R4_TERMINAL_AUDIT_SIZE_BYTES,
        "r4_terminal_chain_root_sha256": R4_TERMINAL_CHAIN_ROOT_SHA256,
        "r4_failure_code": R4_FAILURE_CODE,
        "r4_failure_phase": R4_FAILURE_PHASE,
        "r4_failure_detail_sha256": R4_RECORDED_MASKING_CLEANUP_DETAIL_SHA256,
        "r4_legacy_primary_status": R4_LEGACY_PRIMARY_STATUS,
        "r4_inferred_primary_exception_sha256": R4_INFERRED_PRIMARY_EXCEPTION_SHA256,
        "r4_inferred_primary_detail_sha256": R4_INFERRED_PRIMARY_DETAIL_SHA256,
        "r4_inference_authority": R4_INFERENCE_AUTHORITY,
        "r4_synthetic_reproduction_sha256": hashlib.sha256(
            _canonical_json(R4_SYNTHETIC_REPRODUCTION)
        ).hexdigest(),
        "continuation_action": CONTINUATION_ACTION,
        "owner_confirmation": OWNER_CONFIRMATION,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "complete_seed_resume_only": True,
        "formal_identity_entropy_draw_count": 0,
        "ephemeral_container_nonce_allowed": True,
        "ordinary_execute_allowed": False,
        "ordinary_recovery_cross_basis_allowed": False,
        "redraw_allowed": False,
        "m3_start_allowed": False,
    }


def _load_manifest(path: Path) -> tuple[dict[str, object], bytes]:
    if path.is_symlink():
        _fail("R5 amendment manifest may not be a symlink")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"R5 amendment manifest is invalid JSON: {exc}")
    fixed = _manifest_fixed_policy_v1()
    required = {*fixed, "exact_changed_paths", "source_bindings"}
    if type(value) is not dict or _canonical_json(value) != raw or set(value) != required:
        _fail("R5 amendment manifest is not canonical exact JSON")
    if any(value.get(key) != expected for key, expected in fixed.items()):
        _fail("R5 amendment manifest fixed policy differs")
    if type(value.get("exact_changed_paths")) is not list or type(
        value.get("source_bindings")
    ) is not list:
        _fail("R5 amendment manifest source allowlists differ")
    return value, raw


def _runtime_exception_source_bindings_v1(
    *, repository_root: Path, head: str,
    relative_paths: Sequence[str] | None = None,
) -> tuple[dict[str, object], ...]:
    """Bind every source omitted from the frozen 95-input runtime map."""

    paths = tuple(sorted(R5_RUNTIME_EXCEPTION_PATHS if relative_paths is None else relative_paths))
    if len(paths) != len(set(paths)) or not paths:
        _fail("R5 runtime-exception path registry is empty or duplicated")
    try:
        _r4._r31._verify_changed_index_flags_v1(repository_root, set(paths))
    except _r4._r31.A8R3RecoveryAmendmentError as exc:
        _fail("R5 runtime-exception index flags rejected: " + exc.detail)
    rows: list[dict[str, object]] = []
    repository = repository_root.resolve(strict=True)
    for relative in paths:
        if (
            type(relative) is not str
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
        ):
            _fail("R5 runtime-exception source path is malformed")
        head_raw = _git(repository, ("show", f"{head}:{relative}"))
        digest = hashlib.sha256(head_raw).hexdigest()
        try:
            _r4._r31._verify_changed_worktree_blob_v1(
                repository_root=repository,
                head=head,
                relative=relative,
                expected_sha256=digest,
            )
        except _r4._r31.A8R3RecoveryAmendmentError as exc:
            _fail("R5 runtime-exception worktree binding rejected: " + exc.detail)
        tree_line = _git(repository, ("ls-tree", head, "--", relative)).decode(
            "utf-8", "strict"
        ).strip()
        parts = tree_line.split(None, 3)
        if (
            len(parts) != 4
            or parts[1] != "blob"
            or parts[3] != relative
            or parts[0] not in {"100644", "100755"}
        ):
            _fail(f"R5 runtime-exception Git mode/blob differs: {relative}")
        expected_worktree_mode = 0o755 if parts[0] == "100755" else 0o644
        rows.append(
            {
                "path": relative,
                "git_mode": parts[0],
                "head_blob_sha256": digest,
                "worktree_sha256": digest,
                "worktree_mode_octal": f"{expected_worktree_mode:04o}",
            }
        )
    return tuple(rows)


def inspect_r5_source_preflight_v1(
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
    if _HEX_40.fullmatch(head) is None or parents != [R4_AMENDMENT_COMMIT]:
        _fail("R5 must be one committed sole child of terminal R4 revision 2")
    if _git(repository_root, ("status", "--porcelain=v1", "--untracked-files=all")):
        _fail("R5 repository tree/index is not clean")
    changed_lines = _git(
        repository_root,
        (
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            "--no-renames",
            R4_AMENDMENT_COMMIT,
            head,
        ),
    ).decode("utf-8", "strict").splitlines()
    actual = tuple(
        {"status": line.split("\t", 1)[0], "path": line.split("\t", 1)[1]}
        for line in changed_lines
        if line
    )
    if tuple(manifest["exact_changed_paths"]) != actual:
        _fail("R5 changed-path allowlist differs")
    manifest_relative = manifest_path.resolve(strict=True).relative_to(
        repository_root.resolve(strict=True)
    ).as_posix()
    changed_paths = {str(row["path"]) for row in actual}
    try:
        _r4._r31._verify_changed_index_flags_v1(repository_root, changed_paths)
        _r4._r31._verify_changed_worktree_blob_v1(
            repository_root=repository_root,
            head=head,
            relative=manifest_relative,
            expected_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        )
    except _r4._r31.A8R3RecoveryAmendmentError as exc:
        _fail("R5 changed-source preflight rejected: " + exc.detail)
    expected_paths = tuple(
        str(row["path"])
        for row in actual
        if str(row["path"]) != manifest_relative
    )
    bindings = manifest["source_bindings"]
    if not _r4._r31._source_binding_paths_are_exact_v1(bindings, expected_paths):
        _fail("R5 source bindings do not equal changed paths minus manifest")
    verified: list[dict[str, object]] = []
    for row in bindings:
        if type(row) is not dict or set(row) != {
            "path",
            "parent_sha256_or_null",
            "r5_sha256",
        }:
            _fail("R5 source-binding row differs")
        relative = row.get("path")
        parent_digest = row.get("parent_sha256_or_null")
        current_digest = row.get("r5_sha256")
        if (
            type(relative) is not str
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or (
                parent_digest is not None
                and (
                    type(parent_digest) is not str
                    or _HEX_64.fullmatch(parent_digest) is None
                )
            )
            or type(current_digest) is not str
            or _HEX_64.fullmatch(current_digest) is None
        ):
            _fail("R5 source-binding value differs")
        try:
            _r4._r31._verify_changed_worktree_blob_v1(
                repository_root=repository_root,
                head=head,
                relative=relative,
                expected_sha256=current_digest,
            )
        except _r4._r31.A8R3RecoveryAmendmentError as exc:
            _fail("R5 source-binding worktree verification rejected: " + exc.detail)
        parent_probe = subprocess.run(
            [
                str(FORMAL_GIT_EXECUTABLE),
                "cat-file",
                "-e",
                f"{R4_AMENDMENT_COMMIT}:{relative}",
            ],
            cwd=repository_root.resolve(strict=True),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=60,
            env=dict(FORMAL_GIT_ENVIRONMENT_V1),
        )
        if parent_digest is None:
            if parent_probe.returncode == 0:
                _fail(f"R5 source unexpectedly existed in R4r2: {relative}")
        elif (
            parent_probe.returncode != 0
            or hashlib.sha256(
                _git(repository_root, ("show", f"{R4_AMENDMENT_COMMIT}:{relative}"))
            ).hexdigest()
            != parent_digest
        ):
            _fail(f"parent R4r2 source blob hash differs: {relative}")
        verified.append(dict(row))
    runtime_exception_bindings = _runtime_exception_source_bindings_v1(
        repository_root=repository_root, head=head
    )
    if len(runtime_exception_bindings) != 15:
        _fail("R5 runtime-exception source registry is not the current 15 paths")
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-preflight/1",
        "amendment_commit": head,
        "sole_parent_commit": R4_AMENDMENT_COMMIT,
        "parent_amendment_commit": R4_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 5,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r4_terminal_chain_root_sha256": R4_TERMINAL_CHAIN_ROOT_SHA256,
        "r4_attempt4_consumed": True,
        "r4_finalize_sha256_or_null": None,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "source_bindings": verified,
        "runtime_exception_source_bindings": runtime_exception_bindings,
        "runtime_exception_source_bindings_sha256": hashlib.sha256(
            _canonical_json(runtime_exception_bindings)
        ).hexdigest(),
        "repository_clean": True,
        "exact_changed_paths_verified": True,
        "changed_worktree_blobs_equal_head": True,
        "changed_path_index_flags_normal": True,
        "formal_identity_entropy_draw_count": 0,
        "m3_start_allowed": False,
        "fixed_audit_directory": FIXED_R5_AUDIT_DIRECTORY.as_posix(),
    }


def _require_existing_audit_directory(path: Path, repository_root: Path) -> Path:
    if path.is_symlink():
        _fail("R5 audit directory may not be a symlink")
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    repository = repository_root.resolve(strict=True)
    if (
        resolved != FIXED_R5_AUDIT_DIRECTORY
        or resolved == repository
        or repository in resolved.parents
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
    ):
        _fail("R5 audit directory is not the fixed caller-owned mode-0700 path")
    return resolved


def _create_or_resume_prepare_audit_directory(
    path: Path, repository_root: Path
) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute != FIXED_R5_AUDIT_DIRECTORY:
        _fail("R5 audit directory differs from fixed attempt-5 path")
    repository = repository_root.resolve(strict=True)
    parent = absolute.parent.resolve(strict=True)
    if absolute == repository or repository in absolute.parents:
        _fail("R5 audit directory must be repository-external")
    if absolute.is_symlink():
        _fail("R5 attempt-5 audit directory may not be a symlink")
    if not absolute.exists():
        os.mkdir(absolute, 0o700)
        os.chmod(absolute, 0o700)
        _r4._r31._fsync_directory_v1(parent)
    return _require_existing_audit_directory(absolute, repository_root)


def _install_prepare_record_v1(path: Path, raw: bytes) -> None:
    try:
        _r4._r31._install_prepare_record_v1(path, raw)
    except _r4._r31.A8R3RecoveryAmendmentError as exc:
        _fail("R5 preparation install rejected: " + exc.detail)


def _install_exact_audit_record_v1(
    path: Path, expected: Mapping[str, object], raw: bytes
) -> None:
    try:
        _r4._r31._install_exact_audit_record_v1(path, expected, raw)
    except _r4._r31.A8R3RecoveryAmendmentError as exc:
        _fail("R5 exact audit install rejected: " + exc.detail)


def _discard_non_authoritative_next_v1(path: Path) -> None:
    try:
        _r4._r31._discard_non_authoritative_next_v1(path)
    except _r4._r31.A8R3RecoveryAmendmentError as exc:
        _fail("R5 hidden audit cleanup rejected: " + exc.detail)


def _exact_audit_record_is_visible_v1(path: Path, raw: bytes) -> bool:
    try:
        return _r4._r31._exact_audit_record_is_visible_v1(path, raw)
    except _r4._r31.A8R3RecoveryAmendmentError as exc:
        _fail("R5 visible audit check rejected: " + exc.detail)


def _read_canonical_audit_v1(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        return _r4._r31._r2._read_canonical_audit(path)
    except _r4._r31._r2.A8R2RecoveryAmendmentError as exc:
        _fail("R5 canonical audit read rejected: " + exc.detail)


def _read_canonical_regular_v1(
    path: Path, *, mode: int
) -> tuple[dict[str, object], bytes, dict[str, object]]:
    try:
        return _r4._r31._r2._read_canonical_regular(path, mode=mode)
    except _r4._r31._r2.A8R2RecoveryAmendmentError as exc:
        _fail("R5 canonical regular-file read rejected: " + exc.detail)


def _r4_terminal_chain_snapshot_v1() -> tuple[dict[str, object], ...]:
    """Verify the exact consumed R4r2 chain before authorizing ordinal 5."""

    audit = R4_TERMINAL_AUDIT_DIRECTORY
    if audit.is_symlink():
        _fail("R4r2 terminal audit directory may not be a symlink")
    try:
        metadata = audit.stat()
    except OSError as exc:
        _fail(f"R4r2 terminal audit directory is absent: {exc}")
    if (
        audit.resolve(strict=True) != R4_TERMINAL_AUDIT_DIRECTORY
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
        or {path.name for path in audit.iterdir()} != set(R4_TERMINAL_AUDIT_ORDER)
    ):
        _fail("R4r2 terminal audit is not the exact eight-record chain")
    records: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for name in R4_TERMINAL_AUDIT_ORDER:
        value, raw = _read_canonical_audit_v1(audit / name)
        item = (audit / name).stat()
        raw_sha = hashlib.sha256(raw).hexdigest()
        receipt = value.get("receipt_sha256")
        if (
            raw_sha != R4_TERMINAL_AUDIT_RAW_SHA256[name]
            or receipt != R4_TERMINAL_AUDIT_RECEIPT_SHA256[name]
            or type(receipt) is not str
            or _HEX_64.fullmatch(receipt) is None
            or item.st_size != R4_TERMINAL_AUDIT_SIZE_BYTES[name]
            or stat.S_IMODE(item.st_mode) != 0o600
            or item.st_uid != os.getuid()
            or item.st_gid != os.getgid()
            or item.st_nlink != 1
        ):
            _fail(f"R4r2 terminal audit identity differs: {name}")
        records[name] = value
        rows.append(
            {
                "name": name,
                "raw_sha256": raw_sha,
                "receipt_sha256": receipt,
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
    admission = records["admission.json"]
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
            records[name].get("recovery_attempt_ordinal") != 4
            for name in R4_TERMINAL_AUDIT_ORDER
            if name != "a8-validation-receipt.json"
        )
        or preflight.get("schema")
        != f"{_r4.AUDIT_SCHEMA_PREFIX}-preflight/1"
        or preflight.get("amendment_commit") != R4_AMENDMENT_COMMIT
        or preflight.get("sole_parent_commit") != _r4.R31_AMENDMENT_COMMIT
        or preflight.get("ordinal4_consumed") is not False
        or incident.get("schema")
        != f"{_r4.AUDIT_SCHEMA_PREFIX}-incident-diagnostic/1"
        or incident.get("marker_state") != "PENDING"
        or incident.get("journal_state") != "RESERVED"
        or incident.get("raw_seed_bytes_read_by_r4_orchestrator") is not False
        or incident.get("raw_seed_sha256_computed") is not False
        or validation.get("schema")
        != "hegel-phase3-m25-a8-r3-a8-validation-receipt/1"
        or validation.get("raw_seed_bytes_read") is not False
        or validation.get("raw_seed_sha256_computed") is not False
        or validation.get("m3_start_invoked") is not False
        or request.get("preflight_sha256")
        != R4_TERMINAL_AUDIT_RAW_SHA256["preflight.json"]
        or request.get("incident_diagnostic_sha256")
        != R4_TERMINAL_AUDIT_RAW_SHA256["incident-diagnostic.json"]
        or request.get("a8_validation_receipt_sha256")
        != R4_TERMINAL_AUDIT_RAW_SHA256["a8-validation-receipt.json"]
        or authorization.get("authorization_request_sha256")
        != R4_TERMINAL_AUDIT_RAW_SHA256["authorization-request.json"]
        or attempt.get("authorization_sha256")
        != R4_TERMINAL_AUDIT_RAW_SHA256["authorization.json"]
        or attempt.get("a8_validation_receipt_sha256")
        != R4_TERMINAL_AUDIT_RAW_SHA256["a8-validation-receipt.json"]
        or admission.get("attempt_start_sha256")
        != R4_TERMINAL_AUDIT_RAW_SHA256["attempt-start.json"]
        or admission.get("authorization_sha256")
        != R4_TERMINAL_AUDIT_RAW_SHA256["authorization.json"]
        or admission.get("a8_validation_receipt_sha256")
        != R4_TERMINAL_AUDIT_RAW_SHA256["a8-validation-receipt.json"]
        or failure.get("attempt_start_sha256")
        != R4_TERMINAL_AUDIT_RAW_SHA256["attempt-start.json"]
        or failure.get("admission_sha256_or_null")
        != R4_TERMINAL_AUDIT_RAW_SHA256["admission.json"]
        or failure.get("failure_code") != R4_FAILURE_CODE
        or failure.get("failure_phase") != R4_FAILURE_PHASE
        or failure.get("failure_detail_sha256")
        != R4_RECORDED_MASKING_CLEANUP_DETAIL_SHA256
        or any(
            record.get("formal_identity_entropy_draw_count") != 0
            for record in records.values()
        )
        or any(
            failure.get(name) is not False
            for name in (
                "raw_seed_bytes_read_by_r4_orchestrator",
                "raw_seed_sha256_computed",
                "m3_start_invoked",
            )
        )
    ):
        _fail("R4r2 terminal audit provenance or terminal fields differ")
    if hashlib.sha256(_canonical_json(rows)).hexdigest() != (
        R4_TERMINAL_CHAIN_ROOT_SHA256
    ):
        _fail("R4r2 terminal chain root differs")
    return tuple(rows)


def _validation_request_from_incident_v1(
    incident: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    try:
        return _r4._validation_request_from_incident_v1(incident)
    except _r4.A8R4RecoveryAmendmentError as exc:
        _fail("R5 validation-request construction rejected: " + exc.detail)


def _run_a8_validator_v1(
    request: Mapping[str, object],
) -> tuple[dict[str, object], bytes]:
    try:
        return _r4._run_a8_validator_v1(request)
    except _r4.A8R4RecoveryAmendmentError as exc:
        _fail("R5 isolated A8 validation rejected: " + exc.detail)


def _validate_runtime_artifacts_before_attempt_v1(**kwargs: object) -> tuple[dict[str, object], ...]:
    try:
        return _r4._validate_runtime_artifacts_before_attempt_v1(**kwargs)
    except _r4.A8R4RecoveryAmendmentError as exc:
        _fail("R5 runtime artifact qualification rejected: " + exc.detail)


def _stable_runtime_projection_v1(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    return _r4._stable_runtime_projection_v1(rows)


def _build_incident_diagnostic_v1(
    *, custody_directory: Path, public_evidence_path: Path,
    public_promotion_path: Path,
) -> dict[str, object]:
    try:
        base = dict(
            _r4._build_incident_diagnostic_v1(
                custody_directory=custody_directory,
                public_evidence_path=public_evidence_path,
                public_promotion_path=public_promotion_path,
            )
        )
    except _r4.A8R4RecoveryAmendmentError as exc:
        _fail("R4r2 continuity verifier rejected R5 incident: " + exc.detail)
    rows = _r4_terminal_chain_snapshot_v1()
    docker = base.get("docker_state")
    seeds = base.get("seed_prefix_metadata")
    seed = next(
        (
            row
            for row in seeds
            if type(row) is dict and row.get("name") == "split_master_seed.bin"
        ),
        None,
    ) if type(seeds) in {list, tuple} else None
    if (
        base.get("marker_state") != "PENDING"
        or base.get("journal_state") != "RESERVED"
        or type(docker) is not dict
        or docker.get("run_labelled_container_count") != 0
        or docker.get("fixed_key_volume_count") != 4
        or docker.get("network_operation_invoked") is not False
        or type(seed) is not dict
        or seed.get("size_bytes") != 32
        or seed.get("mode_octal") != "0600"
        or seed.get("raw_seed") is not True
        or seed.get("raw_bytes_read") is not False
        or seed.get("sha256_computed") is not False
        or base.get("formal_identity_entropy_draw_count") != 0
        or base.get("m3_start_allowed") is not False
    ):
        _fail("R5 live preflight is not exact PENDING/RESERVED seed-safe state")
    base["schema"] = f"{AUDIT_SCHEMA_PREFIX}-incident-diagnostic/1"
    base["continuation_action"] = CONTINUATION_ACTION
    base["authorization_revision_id"] = AUTHORIZATION_REVISION_ID
    base["recovery_attempt_ordinal"] = 5
    base["r4_amendment_commit"] = R4_AMENDMENT_COMMIT
    base["r4_terminal_audit_directory"] = R4_TERMINAL_AUDIT_DIRECTORY.as_posix()
    base["r4_terminal_chain"] = rows
    base["r4_terminal_chain_root_sha256"] = R4_TERMINAL_CHAIN_ROOT_SHA256
    for key, name in (
        ("r4_preflight", "preflight.json"),
        ("r4_incident_diagnostic", "incident-diagnostic.json"),
        ("r4_a8_validation", "a8-validation-receipt.json"),
        ("r4_authorization_request", "authorization-request.json"),
        ("r4_authorization", "authorization.json"),
        ("r4_attempt_start", "attempt-start.json"),
        ("r4_admission", "admission.json"),
        ("r4_failure", "failure.json"),
    ):
        base[key + "_raw_sha256"] = R4_TERMINAL_AUDIT_RAW_SHA256[name]
        base[key + "_receipt_sha256"] = R4_TERMINAL_AUDIT_RECEIPT_SHA256[name]
    base["r4_attempt4_consumed"] = True
    base["r4_finalize_sha256_or_null"] = None
    base["r4_failure_code"] = R4_FAILURE_CODE
    base["r4_failure_phase"] = R4_FAILURE_PHASE
    base["r4_failure_detail_sha256"] = R4_RECORDED_MASKING_CLEANUP_DETAIL_SHA256
    base["r4_legacy_primary_status"] = R4_LEGACY_PRIMARY_STATUS
    base["r4_inference_authority"] = R4_INFERENCE_AUTHORITY
    base["r4_inferred_primary_exception_sha256"] = R4_INFERRED_PRIMARY_EXCEPTION_SHA256
    base["r4_inferred_primary_detail_sha256"] = R4_INFERRED_PRIMARY_DETAIL_SHA256
    base["r4_synthetic_reproduction"] = R4_SYNTHETIC_REPRODUCTION
    base["r4_synthetic_reproduction_sha256"] = hashlib.sha256(
        _canonical_json(R4_SYNTHETIC_REPRODUCTION)
    ).hexdigest()
    base["r4_inference_is_canonical_failure_evidence"] = False
    base.pop("raw_seed_bytes_read_by_r4_orchestrator", None)
    base["raw_seed_bytes_read_by_r5_orchestrator"] = False
    return base


def inspect_fixed_a8_r5_preflight_v1(
    *, custody_directory: Path, public_evidence_path: Path,
    public_promotion_path: Path, repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, object]:
    source = inspect_r5_source_preflight_v1(
        repository_root=repository_root, manifest_path=manifest_path
    )
    incident = _build_incident_diagnostic_v1(
        custody_directory=custody_directory,
        public_evidence_path=public_evidence_path,
        public_promotion_path=public_promotion_path,
    )
    return {
        **source,
        "marker_state": incident["marker_state"],
        "journal_state": incident["journal_state"],
        "run_labelled_container_count": incident["docker_state"]["run_labelled_container_count"],
        "fixed_key_volume_count": incident["docker_state"]["fixed_key_volume_count"],
        "raw_seed_bytes_read_by_r5_orchestrator": False,
        "raw_seed_sha256_computed": False,
        "r4_failure_primary_canonically_recoverable": False,
        "r4_inferred_root_cause_is_noncanonical": True,
    }


def _unchanged_a8_input_bindings_v1() -> dict[str, str]:
    bindings: dict[str, str] = {}
    repository = REPOSITORY_ROOT.resolve(strict=True)
    for path in _r4._r31.REQUIRED_COMMIT_A_INPUTS:
        absolute = Path(os.path.abspath(os.fspath(path)))
        try:
            relative = absolute.relative_to(repository).as_posix()
        except ValueError:
            _fail("unchanged A8 runtime input escapes repository")
        if relative in R5_RUNTIME_EXCEPTION_PATHS:
            continue
        frozen = _git(repository, ("show", f"{A8_BASIS_COMMIT}:{relative}"))
        digest = hashlib.sha256(frozen).hexdigest()
        try:
            _r4._r31._verify_changed_worktree_blob_v1(
                repository_root=repository,
                head=A8_BASIS_COMMIT,
                relative=relative,
                expected_sha256=digest,
            )
        except _r4._r31.A8R3RecoveryAmendmentError as exc:
            _fail("unchanged A8 descriptor binding rejected: " + exc.detail)
        bindings[relative] = digest
    root = hashlib.sha256(_executor_canonical_json(bindings)).hexdigest()
    if (
        len(bindings) != _r4._r31.EXPECTED_UNCHANGED_A8_INPUT_COUNT
        or root != _r4._r31.EXPECTED_UNCHANGED_A8_INPUT_ROOT
    ):
        _fail("unchanged A8 runtime closure/root differs")
    return bindings


def _build_source_admission_v1(
    *, amendment_commit: object, incident_raw: bytes, validation_raw: bytes,
    validation: Mapping[str, object], unchanged_inputs: Mapping[str, str],
) -> dict[str, object]:
    if type(amendment_commit) is not str or _HEX_40.fullmatch(amendment_commit) is None:
        _fail("R5 source admission amendment commit differs")
    if (
        len(unchanged_inputs) != _r4._r31.EXPECTED_UNCHANGED_A8_INPUT_COUNT
        or hashlib.sha256(_executor_canonical_json(unchanged_inputs)).hexdigest()
        != _r4._r31.EXPECTED_UNCHANGED_A8_INPUT_ROOT
    ):
        _fail("R5 source admission unchanged-input root differs")
    return {
        "schema": SOURCE_ADMISSION_SCHEMA,
        "basis_commit": A8_BASIS_COMMIT,
        "parent_r4_amendment_commit": R4_AMENDMENT_COMMIT,
        "r5_amendment_commit": amendment_commit,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 5,
        "continuation_action": SOURCE_ADMISSION_CONTINUATION_ACTION,
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
        "unchanged_a8_input_sha256_root": _r4._r31.EXPECTED_UNCHANGED_A8_INPUT_ROOT,
        "actor_report_sha256": validation["actor_report_sha256"],
        "errata_report_sha256": validation["errata_report_sha256"],
        "live_bundle_sha256": validation["live_bundle_sha256"],
        "r4_terminal_chain_root_sha256": R4_TERMINAL_CHAIN_ROOT_SHA256,
        "r4_preflight_raw_sha256": R4_TERMINAL_AUDIT_RAW_SHA256["preflight.json"],
        "r4_preflight_receipt_sha256": R4_TERMINAL_AUDIT_RECEIPT_SHA256["preflight.json"],
        "r4_incident_diagnostic_raw_sha256": R4_TERMINAL_AUDIT_RAW_SHA256["incident-diagnostic.json"],
        "r4_incident_diagnostic_receipt_sha256": R4_TERMINAL_AUDIT_RECEIPT_SHA256["incident-diagnostic.json"],
        "r4_a8_validation_raw_sha256": R4_TERMINAL_AUDIT_RAW_SHA256["a8-validation-receipt.json"],
        "r4_a8_validation_receipt_sha256": R4_TERMINAL_AUDIT_RECEIPT_SHA256["a8-validation-receipt.json"],
        "r4_authorization_request_raw_sha256": R4_TERMINAL_AUDIT_RAW_SHA256["authorization-request.json"],
        "r4_authorization_request_receipt_sha256": R4_TERMINAL_AUDIT_RECEIPT_SHA256["authorization-request.json"],
        "r4_authorization_raw_sha256": R4_TERMINAL_AUDIT_RAW_SHA256["authorization.json"],
        "r4_authorization_receipt_sha256": R4_TERMINAL_AUDIT_RECEIPT_SHA256["authorization.json"],
        "r4_attempt_start_raw_sha256": R4_TERMINAL_AUDIT_RAW_SHA256["attempt-start.json"],
        "r4_attempt_start_receipt_sha256": R4_TERMINAL_AUDIT_RECEIPT_SHA256["attempt-start.json"],
        "r4_admission_raw_sha256": R4_TERMINAL_AUDIT_RAW_SHA256["admission.json"],
        "r4_admission_receipt_sha256": R4_TERMINAL_AUDIT_RECEIPT_SHA256["admission.json"],
        "r4_failure_raw_sha256": R4_TERMINAL_AUDIT_RAW_SHA256["failure.json"],
        "r4_failure_receipt_sha256": R4_TERMINAL_AUDIT_RECEIPT_SHA256["failure.json"],
        "r4_failure_code": R4_FAILURE_CODE,
        "r4_failure_phase": R4_FAILURE_PHASE,
        "r4_failure_detail_sha256": R4_RECORDED_MASKING_CLEANUP_DETAIL_SHA256,
    }


def _authorization_request_fields(
    *, amendment_commit: object, preflight_raw: bytes, incident_raw: bytes,
    validation_raw: bytes,
) -> dict[str, object]:
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-authorization-request/1",
        "amendment_commit": amendment_commit,
        "parent_r4_amendment_commit": R4_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 5,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r4_terminal_chain_root_sha256": R4_TERMINAL_CHAIN_ROOT_SHA256,
        "r4_failure_raw_sha256": R4_TERMINAL_AUDIT_RAW_SHA256["failure.json"],
        "r4_failure_receipt_sha256": R4_TERMINAL_AUDIT_RECEIPT_SHA256["failure.json"],
        "r4_failure_code": R4_FAILURE_CODE,
        "r4_failure_phase": R4_FAILURE_PHASE,
        "r4_failure_detail_sha256": R4_RECORDED_MASKING_CLEANUP_DETAIL_SHA256,
        "r4_legacy_primary_status": R4_LEGACY_PRIMARY_STATUS,
        "r4_inference_authority": R4_INFERENCE_AUTHORITY,
        "r4_inferred_primary_exception_sha256": R4_INFERRED_PRIMARY_EXCEPTION_SHA256,
        "r4_inferred_primary_detail_sha256": R4_INFERRED_PRIMARY_DETAIL_SHA256,
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


def _expected_authorization_fields(
    *, amendment_commit: object, preflight_raw: bytes, incident_raw: bytes,
    validation_raw: bytes, request_raw: bytes,
) -> dict[str, object]:
    return {
        **_authorization_request_fields(
            amendment_commit=amendment_commit,
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
        ),
        "schema": f"{AUDIT_SCHEMA_PREFIX}-authorization/1",
        "authorization_request_sha256": hashlib.sha256(request_raw).hexdigest(),
        "authorization_actor": "PROJECT_OWNER",
        "owner_authorized_fixed_transaction_only": True,
        "ordinary_execute_invoked": False,
    }


def prepare_fixed_a8_r5_authorization_v1(
    *, audit_directory: Path, custody_directory: Path,
    public_evidence_path: Path, public_promotion_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> None:
    preflight = inspect_r5_source_preflight_v1(
        repository_root=repository_root, manifest_path=manifest_path
    )
    incident = _build_incident_diagnostic_v1(
        custody_directory=custody_directory,
        public_evidence_path=public_evidence_path,
        public_promotion_path=public_promotion_path,
    )
    request, _actor, _errata, _bundle = _validation_request_from_incident_v1(incident)
    _validation, validation_raw = _run_a8_validator_v1(request)
    _r4_terminal_chain_snapshot_v1()
    audit = _create_or_resume_prepare_audit_directory(audit_directory, repository_root)
    order = (
        "preflight.json", "incident-diagnostic.json",
        "a8-validation-receipt.json", "authorization-request.json",
    )
    observed = {path.name for path in audit.iterdir()}
    allowed = set(order) | {"." + name + ".next" for name in order}
    if not observed.issubset(allowed):
        _fail("R5 preparation audit contains a non-prefix record")
    visible = [name in observed for name in order]
    if any(visible[index] and not all(visible[:index]) for index in range(len(order))):
        _fail("R5 preparation visible records are not an exact prefix")
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
        order, (preflight_raw, incident_raw, validation_raw, request_raw), strict=True
    ):
        _install_prepare_record_v1(audit / name, payload)


def write_fixed_a8_r5_owner_authorization_v1(
    *, audit_directory: Path, owner_confirmation: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    if owner_confirmation != OWNER_CONFIRMATION:
        _fail("R5 owner confirmation phrase differs")
    audit = _require_existing_audit_directory(audit_directory, repository_root)
    expected = {
        "preflight.json", "incident-diagnostic.json",
        "a8-validation-receipt.json", "authorization-request.json",
    }
    actual = {path.name for path in audit.iterdir()}
    if not expected.issubset(actual) or not actual.issubset(
        expected | {"authorization.json", ".authorization.json.next"}
    ):
        _fail("R5 pre-authorization audit path set differs")
    preflight, preflight_raw = _read_canonical_audit_v1(audit / "preflight.json")
    _incident, incident_raw = _read_canonical_audit_v1(audit / "incident-diagnostic.json")
    _validation, validation_raw, _row = _read_canonical_regular_v1(
        audit / "a8-validation-receipt.json", mode=0o600
    )
    request, request_raw = _read_canonical_audit_v1(
        audit / "authorization-request.json"
    )
    if request != _r4._r31._r2._with_receipt_sha256(
        _authorization_request_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
        )
    ):
        _fail("R5 authorization request differs")
    raw = _receipt_record_bytes_v1(
        _expected_authorization_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
            request_raw=request_raw,
        )
    )
    _install_prepare_record_v1(audit / "authorization.json", raw)


def _validate_final_publication_v1(**kwargs: object) -> dict[str, object]:
    try:
        return _r4._validate_final_publication_v1(**kwargs)
    except _r4.A8R4RecoveryAmendmentError as exc:
        _fail("R5 final publication replay rejected: " + exc.detail)


def _combine_failures_v1(
    primary: BaseException, cleanup: BaseException, *, phase: str
) -> BaseException:
    return _executor.combine_formal_failures_v1(primary, cleanup, phase=phase)


def _failure_evidence_v1(exc: BaseException) -> dict[str, object]:
    evidence = _executor.formal_failure_evidence_v1(exc)
    if type(evidence) is not dict:
        _fail("executor failure evidence is not an object")
    return evidence


def _leaf_failure_rows_v1(
    evidence: Mapping[str, object], *, role: str = "PRIMARY"
) -> tuple[dict[str, object], ...]:
    if evidence.get("kind") == "SINGLE":
        return (
            {
                "role": role,
                "exception_type": evidence.get("exception_type"),
                "code": evidence.get("code"),
                "detail_sha256": evidence.get("detail_sha256"),
            },
        )
    if evidence.get("kind") == "ERROR_GRAPH_TERMINAL":
        return (
            {
                "role": role,
                "exception_type": evidence.get("exception_type"),
                "code": "FORMAL_FAILURE_EVIDENCE_" + str(evidence.get("reason")),
                "detail_sha256": hashlib.sha256(
                    _canonical_json(evidence)
                ).hexdigest(),
            },
        )
    if evidence.get("kind") != "PRIMARY_AND_CLEANUP":
        _fail("executor failure evidence kind differs")
    primary = evidence.get("primary")
    cleanup = evidence.get("cleanup")
    if type(primary) is not dict or type(cleanup) is not dict:
        _fail("executor composite failure evidence shape differs")
    return (
        *_leaf_failure_rows_v1(primary, role=role),
        *_leaf_failure_rows_v1(
            cleanup,
            role="FINAL_CLOSE" if evidence.get("combination_phase") == "R5_OUTER_FINAL_CLOSE" else "CLEANUP",
        ),
    )


def _failure_record_fields_v1(
    *, amendment_commit: object, attempt_start_raw: bytes,
    admission_raw: bytes | None, validation_raw: bytes, failure_phase: str,
    exc: BaseException,
) -> dict[str, object]:
    evidence = _failure_evidence_v1(exc)
    leaves = _leaf_failure_rows_v1(evidence)
    primary = leaves[0]
    cleanups = [row for row in leaves[1:] if row["role"] == "CLEANUP"]
    final_close = next((row for row in leaves[1:] if row["role"] == "FINAL_CLOSE"), None)
    if failure_phase == "R5_OUTER_FINAL_CLOSE" and final_close is None:
        final_close = {**primary, "role": "FINAL_CLOSE"}
    evidence_sha = hashlib.sha256(_canonical_json(evidence)).hexdigest()
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-failure/1",
        "amendment_commit": amendment_commit,
        "parent_r4_amendment_commit": R4_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 5,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r4_terminal_chain_root_sha256": R4_TERMINAL_CHAIN_ROOT_SHA256,
        "r4_failure_raw_sha256": R4_TERMINAL_AUDIT_RAW_SHA256["failure.json"],
        "r4_failure_receipt_sha256": R4_TERMINAL_AUDIT_RECEIPT_SHA256["failure.json"],
        "r4_failure_code": R4_FAILURE_CODE,
        "r4_failure_phase": R4_FAILURE_PHASE,
        "r4_failure_detail_sha256": R4_RECORDED_MASKING_CLEANUP_DETAIL_SHA256,
        "r4_legacy_primary_status": R4_LEGACY_PRIMARY_STATUS,
        "r4_inference_authority": R4_INFERENCE_AUTHORITY,
        "r4_inferred_primary_exception_sha256": R4_INFERRED_PRIMARY_EXCEPTION_SHA256,
        "r4_inferred_primary_detail_sha256": R4_INFERRED_PRIMARY_DETAIL_SHA256,
        "attempt_start_sha256": hashlib.sha256(attempt_start_raw).hexdigest(),
        "admission_sha256_or_null": None if admission_raw is None else hashlib.sha256(admission_raw).hexdigest(),
        "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
        "failure_code": primary["code"],
        "failure_phase": failure_phase,
        "failure_detail_sha256": primary["detail_sha256"],
        "formal_failure_evidence": evidence,
        "formal_failure_evidence_sha256": evidence_sha,
        "primary_failure": primary,
        "cleanup_failures": cleanups,
        "final_close_failure_or_null": final_close,
        "formal_identity_entropy_draw_count": 0,
        "raw_seed_bytes_read_by_r5_orchestrator": False,
        "raw_seed_sha256_computed": False,
        "m3_start_invoked": False,
    }


def execute_fixed_a8_r5_recovery_v1(
    *, custody_directory: Path, rust_formal_replay_binary: Path,
    rust_bridge_dag_replay_binary: Path,
    rust_bridge_dag_qualification_report: Path,
    public_evidence_path: Path, public_promotion_path: Path,
    audit_directory: Path, repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> tuple[dict[str, object], dict[str, object]]:
    preflight_now = inspect_r5_source_preflight_v1(
        repository_root=repository_root, manifest_path=manifest_path
    )
    audit = _require_existing_audit_directory(audit_directory, repository_root)
    terminal_names = (
        "attempt-start.json", "admission.json", "failure.json", "finalize.json"
    )
    if any((audit / name).exists() or (audit / name).is_symlink() for name in terminal_names):
        _fail("R5 attempt-5 was already consumed or has a terminal record")
    expected_before = {
        "preflight.json", "incident-diagnostic.json",
        "a8-validation-receipt.json", "authorization-request.json",
        "authorization.json",
    }
    observed_before = {path.name for path in audit.iterdir()}
    if not expected_before.issubset(observed_before) or not observed_before.issubset(
        expected_before | {".attempt-start.json.next"}
    ):
        _fail("R5 pre-attempt audit path set differs")
    preflight, preflight_raw = _read_canonical_audit_v1(audit / "preflight.json")
    _incident, incident_raw = _read_canonical_audit_v1(audit / "incident-diagnostic.json")
    validation, validation_raw, _row = _read_canonical_regular_v1(
        audit / "a8-validation-receipt.json", mode=0o600
    )
    request, request_raw = _read_canonical_audit_v1(
        audit / "authorization-request.json"
    )
    authorization, authorization_raw = _read_canonical_audit_v1(
        audit / "authorization.json"
    )
    if preflight_raw != _receipt_record_bytes_v1(preflight_now):
        _fail("stored R5 preflight differs from current clean R5 source")
    incident_now = _build_incident_diagnostic_v1(
        custody_directory=custody_directory,
        public_evidence_path=public_evidence_path,
        public_promotion_path=public_promotion_path,
    )
    if incident_raw != _receipt_record_bytes_v1(incident_now):
        _fail("stored R5 incident canonical bytes differ before attempt")
    validation_request, actor_report, errata_report, _expected_bundle = (
        _validation_request_from_incident_v1(incident_now)
    )
    validation_now, validation_now_raw = _run_a8_validator_v1(validation_request)
    if validation_now_raw != validation_raw or validation_now != validation:
        _fail("prepare/execute isolated A8 validation receipts differ")
    if request != _r4._r31._r2._with_receipt_sha256(
        _authorization_request_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
        )
    ):
        _fail("stored R5 authorization request differs")
    if authorization != _r4._r31._r2._with_receipt_sha256(
        _expected_authorization_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
            request_raw=request_raw,
        )
    ):
        _fail("stored R5 owner authorization differs")
    runtime_rows = _validate_runtime_artifacts_before_attempt_v1(
        rust_formal_replay_binary=rust_formal_replay_binary,
        rust_bridge_dag_replay_binary=rust_bridge_dag_replay_binary,
        rust_bridge_dag_qualification_report=rust_bridge_dag_qualification_report,
        expected_bindings=incident_now.get("runtime_artifact_bindings"),
    )
    if _stable_runtime_projection_v1(runtime_rows) != tuple(
        incident_now["live_runtime_stable_projection"]
    ):
        _fail("post-validator runtime stable projection differs")
    _r4_terminal_chain_snapshot_v1()
    unchanged_inputs = _unchanged_a8_input_bindings_v1()

    actors: A8R1RecoveryDockerActorsV1 | None = None
    attempt_start_raw: bytes | None = None
    admission_raw: bytes | None = None
    finalize_raw: bytes | None = None
    payload: dict[str, object] | None = None
    promotion: dict[str, object] | None = None
    final: dict[str, object] | None = None
    primary_exc: BaseException | None = None
    failure_phase = "PRE_ATTEMPT_ACQUIRE"
    try:
        actors = A8R1RecoveryDockerActorsV1(
            basis_commit=A8_BASIS_COMMIT,
            custody_directory=custody_directory,
            rust_formal_replay_binary=rust_formal_replay_binary,
            rust_bridge_dag_replay_binary=rust_bridge_dag_replay_binary,
            rust_bridge_dag_qualification_report=rust_bridge_dag_qualification_report,
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
                _fail("R5 acquired recovery is not the fixed A8 PENDING/RESERVED transaction")
            actors.timestamp = recovery.marker_snapshot.created_at_unix_seconds
            if (
                _canonical_json(_transport(recovery.prestage_intent_fields.get("actor_qualification_report")))
                != _canonical_json(actor_report)
                or _canonical_json(_transport(recovery.prestage_intent_fields.get("errata_qualification_report")))
                != _canonical_json(errata_report)
            ):
                _fail("R5 acquired diagnostic reports differ from A8 receipt")
            source_admission = _build_source_admission_v1(
                amendment_commit=preflight["amendment_commit"],
                incident_raw=incident_raw,
                validation_raw=validation_raw,
                validation=validation,
                unchanged_inputs=unchanged_inputs,
            )
            common = {
                "amendment_commit": preflight["amendment_commit"],
                "parent_r4_amendment_commit": R4_AMENDMENT_COMMIT,
                "formal_repository_commit": A8_BASIS_COMMIT,
                "run_id_hex": FIXED_RUN_ID_HEX,
                "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                "recovery_attempt_ordinal": 5,
                "authorization_revision_id": AUTHORIZATION_REVISION_ID,
                "r4_terminal_chain_root_sha256": R4_TERMINAL_CHAIN_ROOT_SHA256,
                "r4_failure_raw_sha256": R4_TERMINAL_AUDIT_RAW_SHA256["failure.json"],
                "r4_failure_receipt_sha256": R4_TERMINAL_AUDIT_RECEIPT_SHA256["failure.json"],
                "r4_failure_code": R4_FAILURE_CODE,
                "r4_failure_phase": R4_FAILURE_PHASE,
                "r4_failure_detail_sha256": R4_RECORDED_MASKING_CLEANUP_DETAIL_SHA256,
                "r4_legacy_primary_status": R4_LEGACY_PRIMARY_STATUS,
                "r4_inference_authority": R4_INFERENCE_AUTHORITY,
                "r4_inferred_primary_exception_sha256": R4_INFERRED_PRIMARY_EXCEPTION_SHA256,
                "r4_inferred_primary_detail_sha256": R4_INFERRED_PRIMARY_DETAIL_SHA256,
            }
            failure_phase = "ATTEMPT_START_DURABILITY"
            attempt, attempt_start_raw = _r4._r31._build_exact_audit_record_v1(
                {
                    "schema": f"{AUDIT_SCHEMA_PREFIX}-attempt-start/1",
                    **common,
                    "continuation_action": CONTINUATION_ACTION,
                    "authorization_sha256": hashlib.sha256(authorization_raw).hexdigest(),
                    "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
                    "runtime_artifact_metadata": runtime_rows,
                    "runtime_artifact_stable_projection": _stable_runtime_projection_v1(runtime_rows),
                    "runtime_long_lived_identity_excludes": ["st_dev", "st_ino"],
                    "runtime_descriptor_bound_toctou_verified": True,
                    "complete_seed_resume_only": True,
                    "accepted_worker_mode": "REAL_PENDING_RESUME",
                    "formal_identity_entropy_draw_count": 0,
                    "ordinary_execute_invoked": False,
                    "m3_start_invoked": False,
                    "raw_seed_bytes_read_by_r5_orchestrator": False,
                    "raw_seed_sha256_computed": False,
                }
            )
            _install_exact_audit_record_v1(audit / "attempt-start.json", attempt, attempt_start_raw)
            failure_phase = "SOURCE_ADMISSION_DURABILITY"
            admission, admission_raw = _r4._r31._build_exact_audit_record_v1(
                {
                    "schema": f"{AUDIT_SCHEMA_PREFIX}-admission/1",
                    **common,
                    "attempt_start_sha256": hashlib.sha256(attempt_start_raw).hexdigest(),
                    "authorization_sha256": hashlib.sha256(authorization_raw).hexdigest(),
                    "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
                    "source_admission": source_admission,
                    "source_admission_sha256": hashlib.sha256(
                        _executor_canonical_json(source_admission)
                    ).hexdigest(),
                    "complete_seed_resume_only": True,
                    "accepted_worker_mode": "REAL_PENDING_RESUME",
                    "formal_identity_entropy_draw_count": 0,
                    "raw_seed_bytes_read_by_r5_orchestrator": False,
                    "raw_seed_sha256_computed": False,
                    "m3_start_invoked": False,
                }
            )
            _install_exact_audit_record_v1(audit / "admission.json", admission, admission_raw)

            def guard(candidate: PendingCeremonyRecoveryV1) -> Mapping[str, object]:
                if candidate is not recovery:
                    raise FormalContainerExecutorError(
                        FAIL_RECOVERY_SOURCE_ADMISSION,
                        "R5 admission may authorize only the acquired recovery object",
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
    except BaseException as exc:
        primary_exc = exc

    if actors is not None:
        try:
            actors.close()
        except BaseException as close_exc:
            if primary_exc is None:
                primary_exc = close_exc
                failure_phase = "R5_OUTER_FINAL_CLOSE"
            else:
                primary_exc = _combine_failures_v1(
                    primary_exc, close_exc, phase="R5_OUTER_FINAL_CLOSE"
                )
        actors = None

    if primary_exc is None:
        try:
            if (
                payload is None or promotion is None or final is None
                or attempt_start_raw is None or admission_raw is None
            ):
                _fail("R5 successful core result is incomplete")
            failure_phase = "FINALIZE_DURABILITY"
            finalize, finalize_raw = _r4._r31._build_exact_audit_record_v1(
                {
                    "schema": f"{AUDIT_SCHEMA_PREFIX}-finalize/1",
                    "amendment_commit": preflight["amendment_commit"],
                    "parent_r4_amendment_commit": R4_AMENDMENT_COMMIT,
                    "formal_repository_commit": A8_BASIS_COMMIT,
                    "run_id_hex": FIXED_RUN_ID_HEX,
                    "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                    "recovery_attempt_ordinal": 5,
                    "authorization_revision_id": AUTHORIZATION_REVISION_ID,
                    "r4_terminal_chain_root_sha256": R4_TERMINAL_CHAIN_ROOT_SHA256,
                    "r4_failure_raw_sha256": R4_TERMINAL_AUDIT_RAW_SHA256["failure.json"],
                    "r4_failure_receipt_sha256": R4_TERMINAL_AUDIT_RECEIPT_SHA256["failure.json"],
                    "attempt_start_sha256": hashlib.sha256(attempt_start_raw).hexdigest(),
                    "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
                    "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
                    **final,
                    "formal_gates_after": 24,
                    "child_state": "NOT_RUN",
                    "m3_start_invoked": False,
                    "accepted_worker_mode": "REAL_PENDING_RESUME",
                    "raw_seed_bytes_read_by_r5_orchestrator": False,
                    "raw_seed_sha256_computed": False,
                    "formal_identity_entropy_draw_count": 0,
                }
            )
            _install_exact_audit_record_v1(audit / "finalize.json", finalize, finalize_raw)
        except BaseException as exc:
            primary_exc = exc

    def post_primary_call(operation: Callable[[], object], *, phase: str) -> object:
        try:
            return operation()
        except BaseException as followup_exc:
            if primary_exc is not None and followup_exc is not primary_exc:
                combined = _combine_failures_v1(
                    primary_exc, followup_exc, phase=phase
                )
                raise combined from primary_exc
            raise

    def lstat_path_exists(path: Path) -> bool:
        try:
            path.lstat()
        except FileNotFoundError:
            return False
        return True

    finalize_visible = False
    if finalize_raw is not None:
        finalize_visible = bool(
            post_primary_call(
                lambda: _exact_audit_record_is_visible_v1(
                    audit / "finalize.json", finalize_raw
                ),
                phase="R5_FINALIZE_VISIBILITY_RESOLUTION",
            )
        )
    if finalize_visible:
        post_primary_call(
            lambda: _install_prepare_record_v1(
                audit / "finalize.json", finalize_raw
            ),
            phase="R5_FINALIZE_VISIBLE_REPAIR",
        )
        if payload is None or promotion is None:
            _fail("visible R5 finalize lacks return payload")
        return payload, promotion
    if finalize_raw is not None:
        post_primary_call(
            lambda: _discard_non_authoritative_next_v1(audit / "finalize.json"),
            phase="R5_FINALIZE_HIDDEN_DISCARD",
        )
    if primary_exc is None:
        _fail("R5 execution ended without result or failure")

    attempt_visible = bool(
        attempt_start_raw is not None
        and post_primary_call(
            lambda: _exact_audit_record_is_visible_v1(
                audit / "attempt-start.json", attempt_start_raw
            ),
            phase="R5_ATTEMPT_VISIBILITY_RESOLUTION",
        )
    )
    if attempt_visible:
        post_primary_call(
            lambda: _install_prepare_record_v1(
                audit / "attempt-start.json", attempt_start_raw
            ),
            phase="R5_ATTEMPT_VISIBLE_REPAIR",
        )
        admission_visible = bool(
            admission_raw is not None
            and post_primary_call(
                lambda: _exact_audit_record_is_visible_v1(
                    audit / "admission.json", admission_raw
                ),
                phase="R5_ADMISSION_VISIBILITY_RESOLUTION",
            )
        )
        if admission_visible:
            post_primary_call(
                lambda: _install_prepare_record_v1(
                    audit / "admission.json", admission_raw
                ),
                phase="R5_ADMISSION_VISIBLE_REPAIR",
            )
        elif admission_raw is not None:
            post_primary_call(
                lambda: _discard_non_authoritative_next_v1(
                    audit / "admission.json"
                ),
                phase="R5_ADMISSION_HIDDEN_DISCARD",
            )
            admission_raw = None
        failure_path = audit / "failure.json"
        built_failure = post_primary_call(
            lambda: _r4._r31._build_exact_audit_record_v1(
                _failure_record_fields_v1(
                    amendment_commit=preflight["amendment_commit"],
                    attempt_start_raw=attempt_start_raw,
                    admission_raw=admission_raw if admission_visible else None,
                    validation_raw=validation_raw,
                    failure_phase=failure_phase,
                    exc=primary_exc,
                )
            ),
            phase="R5_FAILURE_EVIDENCE_BUILD",
        )
        if type(built_failure) is not tuple or len(built_failure) != 2:
            mismatch = A8R5RecoveryAmendmentError(
                FAIL_AMENDMENT, "R5 failure record builder result differs"
            )
            combined = _combine_failures_v1(
                primary_exc, mismatch, phase="R5_FAILURE_EVIDENCE_BUILD"
            )
            raise combined from primary_exc
        failure, failure_raw = built_failure
        failure_exists = bool(
            post_primary_call(
                lambda: lstat_path_exists(failure_path),
                phase="R5_FAILURE_PATH_PROBE",
            )
        )
        if failure_exists:
            existing_failure, existing_raw = post_primary_call(
                lambda: _read_canonical_audit_v1(failure_path),
                phase="R5_EXISTING_FAILURE_VALIDATION",
            )
            if existing_failure != failure or existing_raw != failure_raw:
                mismatch = A8R5RecoveryAmendmentError(
                    FAIL_AMENDMENT,
                    "existing R5 failure record differs from exact primary evidence",
                )
                combined = _combine_failures_v1(
                    primary_exc, mismatch, phase="R5_EXISTING_FAILURE_CONFLICT"
                )
                raise combined from primary_exc
        else:
            try:
                _install_exact_audit_record_v1(failure_path, failure, failure_raw)
            except BaseException as audit_exc:
                accumulated = _combine_failures_v1(
                    primary_exc, audit_exc, phase="R5_FAILURE_AUDIT_DURABILITY"
                )
                try:
                    failure_visible = _exact_audit_record_is_visible_v1(
                        failure_path, failure_raw
                    )
                except BaseException as visibility_exc:
                    combined = _combine_failures_v1(
                        accumulated,
                        visibility_exc,
                        phase="R5_FAILURE_VISIBILITY_RESOLUTION",
                    )
                    raise combined from primary_exc
                if not failure_visible:
                    try:
                        _discard_non_authoritative_next_v1(failure_path)
                    except BaseException as discard_exc:
                        combined = _combine_failures_v1(
                            accumulated,
                            discard_exc,
                            phase="R5_FAILURE_HIDDEN_DISCARD",
                        )
                        raise combined from primary_exc
                    try:
                        _install_prepare_record_v1(failure_path, failure_raw)
                    except BaseException as retry_exc:
                        combined = _combine_failures_v1(
                            accumulated,
                            retry_exc,
                            phase="R5_FAILURE_CANONICAL_RETRY",
                        )
                        raise combined from primary_exc
                raise accumulated from primary_exc
    raise primary_exc


__all__ = [
    "A8R5RecoveryAmendmentError",
    "DEFAULT_MANIFEST_PATH",
    "execute_fixed_a8_r5_recovery_v1",
    "inspect_fixed_a8_r5_preflight_v1",
    "inspect_r5_source_preflight_v1",
    "prepare_fixed_a8_r5_authorization_v1",
    "write_fixed_a8_r5_owner_authorization_v1",
]
