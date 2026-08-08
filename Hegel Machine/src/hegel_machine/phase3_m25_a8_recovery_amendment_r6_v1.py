"""Attempt-6 recovery after the terminal R5 static-replay/outer-close failure.

R5 consumed attempt ordinal 5.  Its canonical failure record retained a
sanitized ``FormalStaticBasisError`` type plus an exact outer-close evidence
hash; the underlying static-basis code and detail are not canonically
recoverable.  Read-only source/path forensics reproduce a missing explicit
Rust-binary argument and a second-close defect.  Those diagnoses remain
non-canonical and are re-qualified seed-free before ordinal 6 can be consumed.

This module binds the exact eight-record R5 terminal chain and creates one
fresh attempt-6 namespace.  It never opens or hashes the retained raw seed,
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
from . import phase3_m25_a8_recovery_amendment_r5_v1 as _r5
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
    FormalStaticBasisError,
    build_python_static_replay_receipt_v1,
    run_rust_static_replay_receipt_v1,
)


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
DEFAULT_MANIFEST_PATH: Final = (
    PROJECT_ROOT / "config/phase3_m25_a8_recovery_amendment_r6_v1.json"
)
A8_BASIS_COMMIT: Final = _r5.A8_BASIS_COMMIT
R5_AMENDMENT_COMMIT: Final = "0024f8117f6ad20bd004f1a6024987d923f2b7ad"
FIXED_RUN_ID_HEX: Final = _r5.FIXED_RUN_ID_HEX
FIXED_LEDGER_ID_HEX: Final = _r5.FIXED_LEDGER_ID_HEX
FIXED_RUN_ID: Final = _r5.FIXED_RUN_ID
FIXED_LEDGER_ID: Final = _r5.FIXED_LEDGER_ID
FIXED_FORMAL_RUST_BINARY: Final = _r4._r31._r2.FIXED_FORMAL_RUST_BINARY
FIXED_FORMAL_RUST_BINARY_SHA256: Final = (
    _r4._r31._r2.FIXED_FORMAL_RUST_BINARY_SHA256
)
R5_TERMINAL_AUDIT_DIRECTORY: Final = _r5.FIXED_R5_AUDIT_DIRECTORY
FIXED_R6_AUDIT_DIRECTORY: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/"
    "phase3-m25-0af65964235390ce2bebefea7379eaa9c50eda24/"
    "recovery-audit-r6-e4af9f57c38fb298462ec628c4ed8a03-attempt-6"
)
MANIFEST_SCHEMA: Final = "hegel-phase3-m25-a8-recovery-amendment-r6/1"
AUDIT_SCHEMA_PREFIX: Final = "hegel-phase3-m25-a8-r6-recovery-audit"
SOURCE_ADMISSION_SCHEMA: Final = "hegel-phase3-m25-a8-r6-source-admission/1"
FAIL_AMENDMENT: Final = "FAIL_M25_A8_R6_RECOVERY_AMENDMENT"
OWNER_CONFIRMATION: Final = (
    "AUTHORIZE_A8_R6_ATTEMPT_6_SEALED_STATIC_DUAL_AND_SOURCE_CAPABILITY_"
    "IDEMPOTENT_CLOSE_COMPLETE_ONLY_REAL_PENDING_RESUME"
)
CONTINUATION_ACTION: Final = (
    "POST_R5_TERMINAL_STATIC_REPLAY_PATH_AND_IDEMPOTENT_CLOSE_CONTINUATION"
)
SOURCE_ADMISSION_CONTINUATION_ACTION: Final = "CODE_AMENDMENT_RECOVERY_CONTINUATION"
AUTHORIZATION_REVISION_ID: Final = (
    "R6_SEALED_STATIC_DUAL_AND_SOURCE_CAPABILITY_PREQUALIFICATION_V1"
)
R5_TERMINAL_AUDIT_ORDER: Final = (
    "preflight.json",
    "incident-diagnostic.json",
    "a8-validation-receipt.json",
    "authorization-request.json",
    "authorization.json",
    "attempt-start.json",
    "admission.json",
    "failure.json",
)
R5_TERMINAL_AUDIT_RAW_SHA256: Final = {
    "preflight.json": "93618b7c41dfc171a2fe7805ec391d70d032b079264bf3afc107da9bd4b9ffa8",
    "incident-diagnostic.json": "176e4b180a2d63aa8277527c5463102bf1fae2bb5ac6da50e0125a0f861727e0",
    "a8-validation-receipt.json": "ef18694aa41a78389cef2265eb121174f2e68548928f89f7fcad3f55fb261ee4",
    "authorization-request.json": "998f82945a2f6b23a929c227bf98715b6f8684d624827a19458b3a916c905139",
    "authorization.json": "bd48b5acbb212963393f040143f1d9679a16245e7bbb6a15abce8cf0633b37c2",
    "attempt-start.json": "eb748aba5c9c6abff9344aa91f4c670b68b2bac0a7675a24a0754181a808829a",
    "admission.json": "aff16f8c4a4ace64d44685530ad10cdee2d2d97104495e8fc05b36e9bff99dd9",
    "failure.json": "3ae5164908a41ebf1b32b255bf3d0c73e821b843c17251f8e79ff7879f49ae4c",
}
R5_TERMINAL_AUDIT_RECEIPT_SHA256: Final = {
    "preflight.json": "42c570bbb27c18ad8a6443ed9e74994c58013c2f57168db03d6b871a9a9727db",
    "incident-diagnostic.json": "5448340b35fb04730de8a2dc3278283ea15f3b24a11bda6028d3c5ffeebd5145",
    "a8-validation-receipt.json": "83b1ad690914d9dfd5cd402d5c734a1250a3b450c9e0b3ecf4655cfb97c6ba47",
    "authorization-request.json": "ac1340b0539e5a70f478d0723b298c2d27c282049f0fb9e66e27805fb0680b97",
    "authorization.json": "19be355f31a2bc42af9c4695e73757c93584d404c28f92700057aa54a09c0c7a",
    "attempt-start.json": "9a1d28fa48f7f1961668d041a9bd322b536d2a288a39ea9d24b95047ed069873",
    "admission.json": "ad77e0e3441367470bb17c5983337eb092b279d06c2765d71c649dee254441ad",
    "failure.json": "ece0e9f5eba90cad6a685849fea03463f932ad997f0e75fadce98b7aceabd3f6",
}
R5_TERMINAL_AUDIT_SIZE_BYTES: Final = {
    "preflight.json": 7608,
    "incident-diagnostic.json": 26367,
    "a8-validation-receipt.json": 14588,
    "authorization-request.json": 2017,
    "authorization.json": 2225,
    "attempt-start.json": 4199,
    "admission.json": 17216,
    "failure.json": 3185,
}
R5_TERMINAL_CHAIN_ROOT_SHA256: Final = (
    "bcbe5e09f843b71e7448159307a02f698ace61fdccdff80767f3c826b6fb245b"
)
R5_FAILURE_CODE: Final = "FormalStaticBasisError"
R5_FAILURE_PHASE: Final = "COMPLETE_ONLY_FORMAL_CORE"
R5_RECORDED_PRIMARY_DETAIL_SHA256: Final = (
    "b93bf9270ba596c2a92134501c790dfea1a8ec535e86863fb18ff3487ca4eece"
)
R5_FORMAL_FAILURE_EVIDENCE_SHA256: Final = (
    "a64aed1283957993fb2fdd8eda72e4beceb29d9f5f90dc9cf5b6c82f4b234c37"
)
R5_FINAL_CLOSE_FAILURE_CODE: Final = "FAIL_M25_FORMAL_CONTAINER_RUNTIME"
R5_FINAL_CLOSE_DETAIL_SHA256: Final = (
    "3faf8cd39fb25c1fe439ecaef0bb15cfb02f535a09b79e00d0c90f8f10abdd8b"
)
R5_PRIMARY_STATUS: Final = "CANONICAL_SANITIZED_TYPE_ONLY"
R5_DIAGNOSED_PRIMARY_EXCEPTION: Final = (
    "FAIL_M25_STATIC_RUST_REPLAY_POLICY: Rust binary path differs from "
    "implementation binding"
)
R5_DIAGNOSED_PRIMARY_EXCEPTION_SHA256: Final = (
    "501211867d8d8d19804517b036229a30b9d99591ba967f6d4588915a63e5e99b"
)
R5_DIAGNOSED_PRIMARY_DETAIL_SHA256: Final = (
    "67fa5110f9126ca28915eee90ab7f67b8ef53e96ac7ce815e06b2a7278aa2634"
)
R5_DIAGNOSIS_AUTHORITY: Final = "NON_CANONICAL_READ_ONLY_FORENSIC_INFERENCE"
R5_DIAGNOSTIC_REPRODUCTION: Final = {
    "schema": "hegel-phase3-m25-r5-failure-read-only-diagnosis/1",
    "attestation_status": "UNATTESTED_NONCANONICAL_DIAGNOSTIC",
    "canonical_primary_status": R5_PRIMARY_STATUS,
    "canonical_primary_detail_sha256": R5_RECORDED_PRIMARY_DETAIL_SHA256,
    "diagnosed_primary_exception_sha256": (
        R5_DIAGNOSED_PRIMARY_EXCEPTION_SHA256
    ),
    "diagnosed_primary_detail_sha256": R5_DIAGNOSED_PRIMARY_DETAIL_SHA256,
    "diagnosed_basis_bound_binary_path": (
        "/home/erzhu419/mine_code/Asumption Agent/Hegel Machine/rust/"
        "formal_bridge_m25/target/debug/hegel-formal-bridge-m25"
    ),
    "diagnosed_default_binary_path": (
        "/home/erzhu419/.local/state/hegel-machine/"
        "a8-recovery-amendment-r5-worktree/Hegel Machine/rust/"
        "formal_bridge_m25/target/debug/hegel-formal-bridge-m25"
    ),
    "diagnosed_paths_equal": False,
    "final_close_detail_sha256": R5_FINAL_CLOSE_DETAIL_SHA256,
    "final_close_exact_preimage_source_reproduced": True,
    "formal_actor_containers_after_failure": 0,
    "retained_role_private_volume_count": 4,
    "raw_seed_used": False,
    "formal_recovery_invoked": False,
    "m3_start_invoked": False,
}
R6_RUNTIME_EXCEPTION_PATHS: Final = frozenset(
    {
        *_r5.R5_RUNTIME_EXCEPTION_PATHS,
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_amendment_r6_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_cli_r6_v1.py",
    }
)
_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")


class A8R6RecoveryAmendmentError(FormalContainerExecutorError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(code, detail)


class A8R6RecoveryDockerActorsV1(A8R1RecoveryDockerActorsV1):
    """R6 actor with bounded same-object close idempotence.

    The latch is set only after the inherited close has completed all
    container, custody, retained-volume and local-runtime checks.  A failed
    first close is therefore never suppressed; only the amendment's immediate
    outer/final second close becomes a no-op.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._r6_close_succeeded = False

    def close(self) -> None:
        if self._r6_close_succeeded:
            return
        super().close()
        self._r6_close_succeeded = True


def _fail(detail: str) -> NoReturn:
    raise A8R6RecoveryAmendmentError(FAIL_AMENDMENT, detail)


def _canonical_json(value: object) -> bytes:
    return _r4._canonical_json(value)


def _receipt_record_bytes_v1(fields: Mapping[str, object]) -> bytes:
    return _canonical_json(_r4._r31._r2._with_receipt_sha256(fields))


def _require_exact_receipt_raw_v1(
    stored_raw: bytes, expected_fields: Mapping[str, object], *, label: str
) -> None:
    """Require exact canonical bytes; Python object equality is not authority."""

    if type(stored_raw) is not bytes or stored_raw != _receipt_record_bytes_v1(
        expected_fields
    ):
        _fail(f"R6 {label} canonical bytes differ")


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
        "sole_parent_commit": R5_AMENDMENT_COMMIT,
        "parent_amendment_commit": R5_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "fixed_run_id_hex": FIXED_RUN_ID_HEX,
        "fixed_ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 6,
        "fixed_r5_terminal_audit_directory": R5_TERMINAL_AUDIT_DIRECTORY.as_posix(),
        "fixed_r6_audit_directory": FIXED_R6_AUDIT_DIRECTORY.as_posix(),
        "r5_terminal_audit_raw_sha256": R5_TERMINAL_AUDIT_RAW_SHA256,
        "r5_terminal_audit_receipt_sha256": R5_TERMINAL_AUDIT_RECEIPT_SHA256,
        "r5_terminal_audit_size_bytes": R5_TERMINAL_AUDIT_SIZE_BYTES,
        "r5_terminal_chain_root_sha256": R5_TERMINAL_CHAIN_ROOT_SHA256,
        "r5_failure_code": R5_FAILURE_CODE,
        "r5_failure_phase": R5_FAILURE_PHASE,
        "r5_failure_detail_sha256": R5_RECORDED_PRIMARY_DETAIL_SHA256,
        "r5_formal_failure_evidence_sha256": (
            R5_FORMAL_FAILURE_EVIDENCE_SHA256
        ),
        "r5_final_close_failure_code": R5_FINAL_CLOSE_FAILURE_CODE,
        "r5_final_close_failure_detail_sha256": (
            R5_FINAL_CLOSE_DETAIL_SHA256
        ),
        "r5_primary_status": R5_PRIMARY_STATUS,
        "r5_diagnosed_primary_exception_sha256": R5_DIAGNOSED_PRIMARY_EXCEPTION_SHA256,
        "r5_diagnosed_primary_detail_sha256": R5_DIAGNOSED_PRIMARY_DETAIL_SHA256,
        "r5_diagnosis_authority": R5_DIAGNOSIS_AUTHORITY,
        "r5_diagnostic_reproduction_sha256": hashlib.sha256(
            _canonical_json(R5_DIAGNOSTIC_REPRODUCTION)
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
        _fail("R6 amendment manifest may not be a symlink")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"R6 amendment manifest is invalid JSON: {exc}")
    fixed = _manifest_fixed_policy_v1()
    required = {*fixed, "exact_changed_paths", "source_bindings"}
    if type(value) is not dict or _canonical_json(value) != raw or set(value) != required:
        _fail("R6 amendment manifest is not canonical exact JSON")
    fixed_projection = {key: value[key] for key in fixed}
    if _canonical_json(fixed_projection) != _canonical_json(fixed):
        _fail("R6 amendment manifest fixed policy differs")
    if type(value.get("exact_changed_paths")) is not list or type(
        value.get("source_bindings")
    ) is not list:
        _fail("R6 amendment manifest source allowlists differ")
    return value, raw


def _runtime_exception_source_bindings_v1(
    *, repository_root: Path, head: str,
    relative_paths: Sequence[str] | None = None,
) -> tuple[dict[str, object], ...]:
    """Bind every source omitted from the frozen 95-input runtime map."""

    paths = tuple(sorted(R6_RUNTIME_EXCEPTION_PATHS if relative_paths is None else relative_paths))
    if len(paths) != len(set(paths)) or not paths:
        _fail("R6 runtime-exception path registry is empty or duplicated")
    try:
        _r4._r31._verify_changed_index_flags_v1(repository_root, set(paths))
    except _r4._r31.A8R3RecoveryAmendmentError as exc:
        _fail("R6 runtime-exception index flags rejected: " + exc.detail)
    rows: list[dict[str, object]] = []
    repository = repository_root.resolve(strict=True)
    for relative in paths:
        if (
            type(relative) is not str
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
        ):
            _fail("R6 runtime-exception source path is malformed")
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
            _fail("R6 runtime-exception worktree binding rejected: " + exc.detail)
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
            _fail(f"R6 runtime-exception Git mode/blob differs: {relative}")
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


def inspect_r6_source_preflight_v1(
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
    if _HEX_40.fullmatch(head) is None or parents != [R5_AMENDMENT_COMMIT]:
        _fail("R6 must be one committed sole child of terminal R5")
    if _git(repository_root, ("status", "--porcelain=v1", "--untracked-files=all")):
        _fail("R6 repository tree/index is not clean")
    changed_lines = _git(
        repository_root,
        (
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            "--no-renames",
            R5_AMENDMENT_COMMIT,
            head,
        ),
    ).decode("utf-8", "strict").splitlines()
    actual = tuple(
        {"status": line.split("\t", 1)[0], "path": line.split("\t", 1)[1]}
        for line in changed_lines
        if line
    )
    if tuple(manifest["exact_changed_paths"]) != actual:
        _fail("R6 changed-path allowlist differs")
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
        _fail("R6 changed-source preflight rejected: " + exc.detail)
    expected_paths = tuple(
        str(row["path"])
        for row in actual
        if str(row["path"]) != manifest_relative
    )
    bindings = manifest["source_bindings"]
    if not _r4._r31._source_binding_paths_are_exact_v1(bindings, expected_paths):
        _fail("R6 source bindings do not equal changed paths minus manifest")
    verified: list[dict[str, object]] = []
    for row in bindings:
        if type(row) is not dict or set(row) != {
            "path",
            "parent_sha256_or_null",
            "r6_sha256",
        }:
            _fail("R6 source-binding row differs")
        relative = row.get("path")
        parent_digest = row.get("parent_sha256_or_null")
        current_digest = row.get("r6_sha256")
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
            _fail("R6 source-binding value differs")
        try:
            _r4._r31._verify_changed_worktree_blob_v1(
                repository_root=repository_root,
                head=head,
                relative=relative,
                expected_sha256=current_digest,
            )
        except _r4._r31.A8R3RecoveryAmendmentError as exc:
            _fail("R6 source-binding worktree verification rejected: " + exc.detail)
        parent_probe = subprocess.run(
            [
                str(FORMAL_GIT_EXECUTABLE),
                "cat-file",
                "-e",
                f"{R5_AMENDMENT_COMMIT}:{relative}",
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
                _fail(f"R6 source unexpectedly existed in R5: {relative}")
        elif (
            parent_probe.returncode != 0
            or hashlib.sha256(
                _git(repository_root, ("show", f"{R5_AMENDMENT_COMMIT}:{relative}"))
            ).hexdigest()
            != parent_digest
        ):
            _fail(f"parent R5 source blob hash differs: {relative}")
        verified.append(dict(row))
    runtime_exception_bindings = _runtime_exception_source_bindings_v1(
        repository_root=repository_root, head=head
    )
    if len(runtime_exception_bindings) != 15:
        _fail("R6 runtime-exception source registry is not the frozen 15 paths")
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-preflight/1",
        "amendment_commit": head,
        "sole_parent_commit": R5_AMENDMENT_COMMIT,
        "parent_amendment_commit": R5_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 6,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r5_terminal_chain_root_sha256": R5_TERMINAL_CHAIN_ROOT_SHA256,
        "r5_attempt5_consumed": True,
        "r5_finalize_sha256_or_null": None,
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
        "fixed_audit_directory": FIXED_R6_AUDIT_DIRECTORY.as_posix(),
    }


def _require_existing_audit_directory(path: Path, repository_root: Path) -> Path:
    if path.is_symlink():
        _fail("R6 audit directory may not be a symlink")
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    repository = repository_root.resolve(strict=True)
    if (
        resolved != FIXED_R6_AUDIT_DIRECTORY
        or resolved == repository
        or repository in resolved.parents
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
    ):
        _fail("R6 audit directory is not the fixed caller-owned mode-0700 path")
    return resolved


def _create_or_resume_prepare_audit_directory(
    path: Path, repository_root: Path
) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute != FIXED_R6_AUDIT_DIRECTORY:
        _fail("R6 audit directory differs from fixed attempt-6 path")
    repository = repository_root.resolve(strict=True)
    parent = absolute.parent.resolve(strict=True)
    if absolute == repository or repository in absolute.parents:
        _fail("R6 audit directory must be repository-external")
    if absolute.is_symlink():
        _fail("R6 attempt-6 audit directory may not be a symlink")
    if not absolute.exists():
        os.mkdir(absolute, 0o700)
        os.chmod(absolute, 0o700)
        _r4._r31._fsync_directory_v1(parent)
    return _require_existing_audit_directory(absolute, repository_root)


def _install_prepare_record_v1(path: Path, raw: bytes) -> None:
    try:
        _r4._r31._install_prepare_record_v1(path, raw)
    except _r4._r31.A8R3RecoveryAmendmentError as exc:
        _fail("R6 preparation install rejected: " + exc.detail)


def _install_exact_audit_record_v1(
    path: Path, expected: Mapping[str, object], raw: bytes
) -> None:
    try:
        _r4._r31._install_exact_audit_record_v1(path, expected, raw)
    except _r4._r31.A8R3RecoveryAmendmentError as exc:
        _fail("R6 exact audit install rejected: " + exc.detail)


def _discard_non_authoritative_next_v1(path: Path) -> None:
    try:
        _r4._r31._discard_non_authoritative_next_v1(path)
    except _r4._r31.A8R3RecoveryAmendmentError as exc:
        _fail("R6 hidden audit cleanup rejected: " + exc.detail)


def _exact_audit_record_is_visible_v1(path: Path, raw: bytes) -> bool:
    try:
        return _r4._r31._exact_audit_record_is_visible_v1(path, raw)
    except _r4._r31.A8R3RecoveryAmendmentError as exc:
        _fail("R6 visible audit check rejected: " + exc.detail)


def _read_canonical_audit_v1(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        return _r4._r31._r2._read_canonical_audit(path)
    except _r4._r31._r2.A8R2RecoveryAmendmentError as exc:
        _fail("R6 canonical audit read rejected: " + exc.detail)


def _read_canonical_regular_v1(
    path: Path, *, mode: int
) -> tuple[dict[str, object], bytes, dict[str, object]]:
    try:
        return _r4._r31._r2._read_canonical_regular(path, mode=mode)
    except _r4._r31._r2.A8R2RecoveryAmendmentError as exc:
        _fail("R6 canonical regular-file read rejected: " + exc.detail)


_R6_PRE_ATTEMPT_AUDIT_NAMES: Final = (
    "preflight.json",
    "incident-diagnostic.json",
    "a8-validation-receipt.json",
    "static-qualification.json",
    "source-capability-qualification.json",
    "authorization-request.json",
    "authorization.json",
)


def _read_pre_attempt_audit_snapshot_v1(
    audit: Path, *, allow_attempt_next: bool
) -> tuple[tuple[int, int, int, int, int], dict[str, bytes], dict[str, tuple[object, ...]]]:
    """Descriptor-read the immutable owner prefix and retain inode identities."""

    metadata = audit.lstat()
    directory_identity = (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_gid,
    )
    observed = {path.name for path in audit.iterdir()}
    allowed = set(_R6_PRE_ATTEMPT_AUDIT_NAMES)
    if allow_attempt_next:
        allowed.add(".attempt-start.json.next")
    if not set(_R6_PRE_ATTEMPT_AUDIT_NAMES).issubset(observed) or not observed.issubset(
        allowed
    ):
        _fail("R6 pre-attempt audit namespace differs")
    raws: dict[str, bytes] = {}
    inodes: dict[str, tuple[object, ...]] = {}
    for name in _R6_PRE_ATTEMPT_AUDIT_NAMES:
        value, raw, row = _read_canonical_regular_v1(audit / name, mode=0o600)
        body = dict(value)
        receipt = body.pop("receipt_sha256", None)
        if (
            type(receipt) is not str
            or receipt != hashlib.sha256(_canonical_json(body)).hexdigest()
        ):
            _fail(f"R6 pre-attempt audit self-hash differs: {name}")
        raws[name] = raw
        inodes[name] = (
            row["st_dev"],
            row["st_ino"],
            row["mode_octal"],
            row["size_bytes"],
            row["uid"],
            row["gid"],
        )
    return directory_identity, raws, inodes


def _recheck_pre_attempt_audit_under_lock_v1(
    *,
    audit: Path,
    expected_directory_identity: tuple[int, int, int, int, int],
    expected_raws: Mapping[str, bytes],
    expected_inodes: Mapping[str, tuple[object, ...]],
) -> None:
    """Linearize attempt consumption against the held formal recovery flock."""

    _discard_non_authoritative_next_v1(audit / "attempt-start.json")
    directory_identity, raws, inodes = _read_pre_attempt_audit_snapshot_v1(
        audit, allow_attempt_next=False
    )
    if (
        directory_identity != expected_directory_identity
        or dict(raws) != dict(expected_raws)
        or dict(inodes) != dict(expected_inodes)
    ):
        _fail("R6 pre-attempt audit prefix changed under recovery lock")


def _r5_terminal_chain_snapshot_v1() -> tuple[dict[str, object], ...]:
    """Verify the exact consumed R5 chain before authorizing ordinal 6."""

    audit = R5_TERMINAL_AUDIT_DIRECTORY
    if audit.is_symlink():
        _fail("R5 terminal audit directory may not be a symlink")
    try:
        metadata = audit.stat()
    except OSError as exc:
        _fail(f"R5 terminal audit directory is absent: {exc}")
    if (
        audit.resolve(strict=True) != R5_TERMINAL_AUDIT_DIRECTORY
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
        or {path.name for path in audit.iterdir()} != set(R5_TERMINAL_AUDIT_ORDER)
    ):
        _fail("R5 terminal audit is not the exact eight-record chain")
    records: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for name in R5_TERMINAL_AUDIT_ORDER:
        value, raw = _read_canonical_audit_v1(audit / name)
        item = (audit / name).stat()
        raw_sha = hashlib.sha256(raw).hexdigest()
        receipt = value.get("receipt_sha256")
        if (
            raw_sha != R5_TERMINAL_AUDIT_RAW_SHA256[name]
            or receipt != R5_TERMINAL_AUDIT_RECEIPT_SHA256[name]
            or type(receipt) is not str
            or _HEX_64.fullmatch(receipt) is None
            or item.st_size != R5_TERMINAL_AUDIT_SIZE_BYTES[name]
            or stat.S_IMODE(item.st_mode) != 0o600
            or item.st_uid != os.getuid()
            or item.st_gid != os.getgid()
            or item.st_nlink != 1
        ):
            _fail(f"R5 terminal audit identity differs: {name}")
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
            records[name].get("recovery_attempt_ordinal") != 5
            for name in R5_TERMINAL_AUDIT_ORDER
            if name != "a8-validation-receipt.json"
        )
        or preflight.get("schema")
        != f"{_r5.AUDIT_SCHEMA_PREFIX}-preflight/1"
        or preflight.get("amendment_commit") != R5_AMENDMENT_COMMIT
        or preflight.get("sole_parent_commit") != _r5.R4_AMENDMENT_COMMIT
        or preflight.get("r4_attempt4_consumed") is not True
        or incident.get("schema")
        != f"{_r5.AUDIT_SCHEMA_PREFIX}-incident-diagnostic/1"
        or incident.get("marker_state") != "PENDING"
        or incident.get("journal_state") != "RESERVED"
        or incident.get("raw_seed_bytes_read_by_r5_orchestrator") is not False
        or incident.get("raw_seed_sha256_computed") is not False
        or validation.get("schema")
        != "hegel-phase3-m25-a8-r3-a8-validation-receipt/1"
        or validation.get("raw_seed_bytes_read") is not False
        or validation.get("raw_seed_sha256_computed") is not False
        or validation.get("m3_start_invoked") is not False
        or request.get("preflight_sha256")
        != R5_TERMINAL_AUDIT_RAW_SHA256["preflight.json"]
        or request.get("incident_diagnostic_sha256")
        != R5_TERMINAL_AUDIT_RAW_SHA256["incident-diagnostic.json"]
        or request.get("a8_validation_receipt_sha256")
        != R5_TERMINAL_AUDIT_RAW_SHA256["a8-validation-receipt.json"]
        or authorization.get("authorization_request_sha256")
        != R5_TERMINAL_AUDIT_RAW_SHA256["authorization-request.json"]
        or attempt.get("schema")
        != f"{_r5.AUDIT_SCHEMA_PREFIX}-attempt-start/1"
        or attempt.get("authorization_sha256")
        != R5_TERMINAL_AUDIT_RAW_SHA256["authorization.json"]
        or attempt.get("a8_validation_receipt_sha256")
        != R5_TERMINAL_AUDIT_RAW_SHA256["a8-validation-receipt.json"]
        or admission.get("attempt_start_sha256")
        != R5_TERMINAL_AUDIT_RAW_SHA256["attempt-start.json"]
        or admission.get("authorization_sha256")
        != R5_TERMINAL_AUDIT_RAW_SHA256["authorization.json"]
        or admission.get("a8_validation_receipt_sha256")
        != R5_TERMINAL_AUDIT_RAW_SHA256["a8-validation-receipt.json"]
        or admission.get("source_admission", {}).get("schema")
        != _r5.SOURCE_ADMISSION_SCHEMA
        or failure.get("schema")
        != f"{_r5.AUDIT_SCHEMA_PREFIX}-failure/1"
        or failure.get("attempt_start_sha256")
        != R5_TERMINAL_AUDIT_RAW_SHA256["attempt-start.json"]
        or failure.get("admission_sha256_or_null")
        != R5_TERMINAL_AUDIT_RAW_SHA256["admission.json"]
        or failure.get("failure_code") != R5_FAILURE_CODE
        or failure.get("failure_phase") != R5_FAILURE_PHASE
        or failure.get("failure_detail_sha256")
        != R5_RECORDED_PRIMARY_DETAIL_SHA256
        or failure.get("formal_failure_evidence_sha256")
        != R5_FORMAL_FAILURE_EVIDENCE_SHA256
        or failure.get("primary_failure")
        != {
            "code": R5_FAILURE_CODE,
            "detail_sha256": R5_RECORDED_PRIMARY_DETAIL_SHA256,
            "exception_type": "FormalStaticBasisError",
            "role": "PRIMARY",
        }
        or failure.get("final_close_failure_or_null")
        != {
            "code": R5_FINAL_CLOSE_FAILURE_CODE,
            "detail_sha256": R5_FINAL_CLOSE_DETAIL_SHA256,
            "exception_type": "FormalContainerExecutorError",
            "role": "FINAL_CLOSE",
        }
        or failure.get("cleanup_failures") != []
        or any(
            record.get("amendment_commit") != R5_AMENDMENT_COMMIT
            for name, record in records.items()
            if name not in {
                "a8-validation-receipt.json",
                "incident-diagnostic.json",
            }
        )
        or any(
            record.get("authorization_revision_id")
            != _r5.AUTHORIZATION_REVISION_ID
            for name, record in records.items()
            if name != "a8-validation-receipt.json"
        )
        or any(
            record.get("formal_identity_entropy_draw_count") != 0
            for record in records.values()
        )
        or any(
            failure.get(name) is not False
            for name in (
                "raw_seed_bytes_read_by_r5_orchestrator",
                "raw_seed_sha256_computed",
                "m3_start_invoked",
            )
        )
    ):
        _fail("R5 terminal audit provenance or terminal fields differ")
    if hashlib.sha256(_canonical_json(rows)).hexdigest() != (
        R5_TERMINAL_CHAIN_ROOT_SHA256
    ):
        _fail("R5 terminal chain root differs")
    return tuple(rows)


def _validation_request_from_incident_v1(
    incident: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    try:
        return _r4._validation_request_from_incident_v1(incident)
    except _r4.A8R4RecoveryAmendmentError as exc:
        _fail("R6 validation-request construction rejected: " + exc.detail)


def _run_a8_validator_v1(
    request: Mapping[str, object],
) -> tuple[dict[str, object], bytes]:
    try:
        return _r4._run_a8_validator_v1(request)
    except _r4.A8R4RecoveryAmendmentError as exc:
        _fail("R6 isolated A8 validation rejected: " + exc.detail)


def _validate_runtime_artifacts_before_attempt_v1(**kwargs: object) -> tuple[dict[str, object], ...]:
    try:
        return _r4._validate_runtime_artifacts_before_attempt_v1(**kwargs)
    except _r4.A8R4RecoveryAmendmentError as exc:
        _fail("R6 runtime artifact qualification rejected: " + exc.detail)


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
        _fail("R4 continuity verifier rejected R6 incident: " + exc.detail)
    rows = _r5_terminal_chain_snapshot_v1()
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
        _fail("R6 live preflight is not exact PENDING/RESERVED seed-safe state")
    base["schema"] = f"{AUDIT_SCHEMA_PREFIX}-incident-diagnostic/1"
    base["continuation_action"] = CONTINUATION_ACTION
    base["authorization_revision_id"] = AUTHORIZATION_REVISION_ID
    base["recovery_attempt_ordinal"] = 6
    base["r5_amendment_commit"] = R5_AMENDMENT_COMMIT
    base["r5_terminal_audit_directory"] = R5_TERMINAL_AUDIT_DIRECTORY.as_posix()
    base["r5_terminal_chain"] = rows
    base["r5_terminal_chain_root_sha256"] = R5_TERMINAL_CHAIN_ROOT_SHA256
    for key, name in (
        ("r5_preflight", "preflight.json"),
        ("r5_incident_diagnostic", "incident-diagnostic.json"),
        ("r5_a8_validation", "a8-validation-receipt.json"),
        ("r5_authorization_request", "authorization-request.json"),
        ("r5_authorization", "authorization.json"),
        ("r5_attempt_start", "attempt-start.json"),
        ("r5_admission", "admission.json"),
        ("r5_failure", "failure.json"),
    ):
        base[key + "_raw_sha256"] = R5_TERMINAL_AUDIT_RAW_SHA256[name]
        base[key + "_receipt_sha256"] = R5_TERMINAL_AUDIT_RECEIPT_SHA256[name]
    base["r5_attempt5_consumed"] = True
    base["r5_finalize_sha256_or_null"] = None
    base["r5_failure_code"] = R5_FAILURE_CODE
    base["r5_failure_phase"] = R5_FAILURE_PHASE
    base["r5_failure_detail_sha256"] = R5_RECORDED_PRIMARY_DETAIL_SHA256
    base["r5_formal_failure_evidence_sha256"] = (
        R5_FORMAL_FAILURE_EVIDENCE_SHA256
    )
    base["r5_final_close_failure_code"] = R5_FINAL_CLOSE_FAILURE_CODE
    base["r5_final_close_failure_detail_sha256"] = (
        R5_FINAL_CLOSE_DETAIL_SHA256
    )
    base["r5_primary_status"] = R5_PRIMARY_STATUS
    base["r5_diagnosis_authority"] = R5_DIAGNOSIS_AUTHORITY
    base["r5_diagnosed_primary_exception_sha256"] = R5_DIAGNOSED_PRIMARY_EXCEPTION_SHA256
    base["r5_diagnosed_primary_detail_sha256"] = R5_DIAGNOSED_PRIMARY_DETAIL_SHA256
    base["r5_diagnostic_reproduction"] = R5_DIAGNOSTIC_REPRODUCTION
    base["r5_diagnostic_reproduction_sha256"] = hashlib.sha256(
        _canonical_json(R5_DIAGNOSTIC_REPRODUCTION)
    ).hexdigest()
    base["r5_diagnosis_is_canonical_failure_evidence"] = False
    base.pop("raw_seed_bytes_read_by_r4_orchestrator", None)
    base["raw_seed_bytes_read_by_r6_orchestrator"] = False
    return base


def inspect_fixed_a8_r6_preflight_v1(
    *, custody_directory: Path, public_evidence_path: Path,
    public_promotion_path: Path, repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, object]:
    source = inspect_r6_source_preflight_v1(
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
        "raw_seed_bytes_read_by_r6_orchestrator": False,
        "raw_seed_sha256_computed": False,
        "r5_failure_underlying_code_detail_canonically_recoverable": False,
        "r5_diagnosed_root_cause_is_noncanonical": True,
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
        if relative in R6_RUNTIME_EXCEPTION_PATHS:
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
    static_qualification_raw: bytes, validation: Mapping[str, object],
    unchanged_inputs: Mapping[str, str],
) -> dict[str, object]:
    if type(amendment_commit) is not str or _HEX_40.fullmatch(amendment_commit) is None:
        _fail("R6 source admission amendment commit differs")
    if (
        len(unchanged_inputs) != _r4._r31.EXPECTED_UNCHANGED_A8_INPUT_COUNT
        or hashlib.sha256(_executor_canonical_json(unchanged_inputs)).hexdigest()
        != _r4._r31.EXPECTED_UNCHANGED_A8_INPUT_ROOT
    ):
        _fail("R6 source admission unchanged-input root differs")
    return {
        "schema": SOURCE_ADMISSION_SCHEMA,
        "basis_commit": A8_BASIS_COMMIT,
        "parent_r5_amendment_commit": R5_AMENDMENT_COMMIT,
        "r6_amendment_commit": amendment_commit,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 6,
        "continuation_action": SOURCE_ADMISSION_CONTINUATION_ACTION,
        "incident_diagnostic_sha256": hashlib.sha256(incident_raw).hexdigest(),
        "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
        "static_preconsumption_qualification_sha256": hashlib.sha256(
            static_qualification_raw
        ).hexdigest(),
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
        "r5_terminal_chain_root_sha256": R5_TERMINAL_CHAIN_ROOT_SHA256,
        "r5_preflight_raw_sha256": R5_TERMINAL_AUDIT_RAW_SHA256["preflight.json"],
        "r5_preflight_receipt_sha256": R5_TERMINAL_AUDIT_RECEIPT_SHA256["preflight.json"],
        "r5_incident_diagnostic_raw_sha256": R5_TERMINAL_AUDIT_RAW_SHA256["incident-diagnostic.json"],
        "r5_incident_diagnostic_receipt_sha256": R5_TERMINAL_AUDIT_RECEIPT_SHA256["incident-diagnostic.json"],
        "r5_a8_validation_raw_sha256": R5_TERMINAL_AUDIT_RAW_SHA256["a8-validation-receipt.json"],
        "r5_a8_validation_receipt_sha256": R5_TERMINAL_AUDIT_RECEIPT_SHA256["a8-validation-receipt.json"],
        "r5_authorization_request_raw_sha256": R5_TERMINAL_AUDIT_RAW_SHA256["authorization-request.json"],
        "r5_authorization_request_receipt_sha256": R5_TERMINAL_AUDIT_RECEIPT_SHA256["authorization-request.json"],
        "r5_authorization_raw_sha256": R5_TERMINAL_AUDIT_RAW_SHA256["authorization.json"],
        "r5_authorization_receipt_sha256": R5_TERMINAL_AUDIT_RECEIPT_SHA256["authorization.json"],
        "r5_attempt_start_raw_sha256": R5_TERMINAL_AUDIT_RAW_SHA256["attempt-start.json"],
        "r5_attempt_start_receipt_sha256": R5_TERMINAL_AUDIT_RECEIPT_SHA256["attempt-start.json"],
        "r5_admission_raw_sha256": R5_TERMINAL_AUDIT_RAW_SHA256["admission.json"],
        "r5_admission_receipt_sha256": R5_TERMINAL_AUDIT_RECEIPT_SHA256["admission.json"],
        "r5_failure_raw_sha256": R5_TERMINAL_AUDIT_RAW_SHA256["failure.json"],
        "r5_failure_receipt_sha256": R5_TERMINAL_AUDIT_RECEIPT_SHA256["failure.json"],
        "r5_failure_code": R5_FAILURE_CODE,
        "r5_failure_phase": R5_FAILURE_PHASE,
        "r5_failure_detail_sha256": R5_RECORDED_PRIMARY_DETAIL_SHA256,
        "r5_formal_failure_evidence_sha256": (
            R5_FORMAL_FAILURE_EVIDENCE_SHA256
        ),
        "r5_final_close_failure_code": R5_FINAL_CLOSE_FAILURE_CODE,
        "r5_final_close_failure_detail_sha256": (
            R5_FINAL_CLOSE_DETAIL_SHA256
        ),
    }


def _static_qualification_fields_v1(
    *,
    amendment_commit: str,
    static_dual: _executor._PrevalidatedPendingRecoveryStaticDualV1,
) -> dict[str, object]:
    """Project one executor-sealed dual qualification into public evidence."""

    if (
        type(static_dual)
        is not _executor._PrevalidatedPendingRecoveryStaticDualV1
        or static_dual._seal
        is not _executor._PREVALIDATED_PENDING_STATIC_DUAL_SEAL
        or static_dual.recovery.basis_commit != A8_BASIS_COMMIT
        or static_dual.recovery.run_id != FIXED_RUN_ID
        or static_dual.recovery.ledger_id != FIXED_LEDGER_ID
    ):
        _fail("R6 static pre-consumption capability differs")
    basis = static_dual.basis
    python_receipt = dict(static_dual.python_receipt)
    rust_receipt = dict(static_dual.rust_receipt)
    entries = rust_receipt.get("entries")
    try:
        bound_path = Path(
            str(basis.implementation_inputs["rust_binary_path"])
        ).resolve(strict=True)
        fixed_path = FIXED_FORMAL_RUST_BINARY.resolve(strict=True)
    except OSError as exc:
        _fail(f"R6 fixed formal Rust path cannot be resolved: {exc}")
    bound_digest = bytes(
        basis.implementation_inputs["rust_binary_sha256"]
    ).hex()
    root_rows = [
        {"root_name": name, "root_hex": digest.hex()}
        for name, digest in static_dual.static_roots.items()
    ]
    implementation_root_rows = [
        {"root_name": name, "root_hex": digest.hex()}
        for name, digest in static_dual.implementation_roots.items()
    ]
    if (
        bound_path != fixed_path
        or bound_digest != FIXED_FORMAL_RUST_BINARY_SHA256
        or type(entries) is not list
        or len(entries) != len(basis.gate19_plan)
        or len(root_rows) != 6
        or any(
            type(row["root_hex"]) is not str
            or _HEX_64.fullmatch(str(row["root_hex"])) is None
            for row in root_rows + implementation_root_rows
        )
        or static_dual.frozen_daemon_binding
        != static_dual.static_daemon_binding
        or rust_receipt.get("network_mode_none") is not True
        or rust_receipt.get("pull_policy_never") is not True
        or rust_receipt.get("seed_key_signature_or_state_created") is not False
        or static_dual.actors._actor_start_attempted
        or static_dual.actors._containers
        or static_dual.actors._state_volumes
    ):
        _fail("R6 seed-free static qualification scope differs")
    fields: dict[str, object] = {
        "schema": "hegel-phase3-m25-a8-r6-static-preconsumption-qualification/2",
        "amendment_commit": amendment_commit,
        "parent_r5_amendment_commit": R5_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 6,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r5_terminal_chain_root_sha256": R5_TERMINAL_CHAIN_ROOT_SHA256,
        "prestage_intent_sha256": static_dual.prestage_intent_sha256,
        "basis_bound_rust_binary_path": bound_path.as_posix(),
        "basis_bound_rust_binary_sha256": bound_digest,
        "explicit_fixed_main_rust_path_passed": True,
        "python_static_receipt_full_sha256": hashlib.sha256(
            _executor_canonical_json(python_receipt)
        ).hexdigest(),
        "python_static_receipt_sha256": python_receipt["receipt_sha256"],
        "rust_static_receipt_full_sha256": hashlib.sha256(
            _executor_canonical_json(rust_receipt)
        ).hexdigest(),
        "rust_static_receipt_sha256": rust_receipt["receipt_sha256"],
        "rust_static_entry_count": len(entries),
        "rust_static_binary_sha256": rust_receipt["binary_sha256"],
        "rust_static_image_ref": rust_receipt["container_image_ref_or_null"],
        "dual_static_root_rows": root_rows,
        "dual_static_root_rows_sha256": hashlib.sha256(
            _executor_canonical_json(root_rows)
        ).hexdigest(),
        "implementation_root_rows": implementation_root_rows,
        "implementation_root_rows_sha256": hashlib.sha256(
            _executor_canonical_json(implementation_root_rows)
        ).hexdigest(),
        "frozen_daemon_receipt_binding_hex": (
            static_dual.frozen_daemon_binding.hex()
        ),
        "static_daemon_receipt_binding_hex": (
            static_dual.static_daemon_binding.hex()
        ),
        "parent_absence_bundle_root_hex": (
            static_dual.parent_absence.audit_bundle_root.hex()
        ),
        "parent_absence_bundle_fields_sha256": hashlib.sha256(
            _executor_canonical_json(
                _transport(static_dual.parent_absence.audit_bundle_fields)
            )
        ).hexdigest(),
        "network_mode_none": True,
        "pull_policy_never": True,
        "dual_static_replay_validated": True,
        "frozen_daemon_binding_validated": True,
        "parent_absence_replay_passed": True,
        "purpose_actor_start_attempted": False,
        "role_private_volume_created": False,
        "formal_identity_entropy_draw_count": 0,
        "raw_seed_bytes_read_by_r6_orchestrator": False,
        "raw_seed_sha256_computed": False,
        "m3_start_invoked": False,
    }
    fields["qualification_fingerprint_sha256"] = hashlib.sha256(
        _canonical_json(fields)
    ).hexdigest()
    return fields


def _source_capability_qualification_fields_v1(
    *,
    amendment_commit: str,
    source_admission: Mapping[str, object],
    static_qualification_raw: bytes,
    formal_prefix: _executor._PrevalidatedPendingRecoveryFormalPrefixV1,
) -> dict[str, object]:
    """Project the executor-created fixed-A8 capability without serializing it."""

    if (
        type(formal_prefix)
        is not _executor._PrevalidatedPendingRecoveryFormalPrefixV1
        or formal_prefix._seal
        is not _executor._PREVALIDATED_PENDING_FORMAL_PREFIX_SEAL
        or formal_prefix.consumed
        or formal_prefix.source_admission_bytes
        != _executor_canonical_json(source_admission)
    ):
        _fail("R6 formal source-capability qualification differs")
    capability = formal_prefix.fixed_capability
    receipt = dict(capability.validation_receipt)
    key_rows = [
        {
            "purpose_id": purpose,
            "qualification_only_key_id_hex": (
                capability.live_protocol.qualification_key_ids[purpose].hex()
            ),
        }
        for purpose in (1, 2, 3, 4)
    ]
    fields: dict[str, object] = {
        "schema": "hegel-phase3-m25-a8-r6-source-capability-qualification/1",
        "amendment_commit": amendment_commit,
        "parent_r5_amendment_commit": R5_AMENDMENT_COMMIT,
        "formal_repository_commit": capability.basis_commit,
        "run_id_hex": capability.run_id.hex(),
        "ledger_id_hex": capability.ledger_id.hex(),
        "recovery_attempt_ordinal": 6,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r5_terminal_chain_root_sha256": R5_TERMINAL_CHAIN_ROOT_SHA256,
        "prestage_intent_sha256": (
            formal_prefix.static_dual.prestage_intent_sha256
        ),
        "static_preconsumption_qualification_sha256": hashlib.sha256(
            static_qualification_raw
        ).hexdigest(),
        "source_admission_sha256": hashlib.sha256(
            formal_prefix.source_admission_bytes
        ).hexdigest(),
        "source_admission_schema": source_admission["schema"],
        "fixed_validation_receipt_full_sha256": hashlib.sha256(
            _executor_canonical_json(receipt)
        ).hexdigest(),
        "fixed_validation_receipt_sha256": receipt["receipt_sha256"],
        "live_protocol_bundle_content_id_hex": (
            capability.live_protocol.bundle_content_id.hex()
        ),
        "qualification_key_id_rows": key_rows,
        "qualification_key_id_rows_sha256": hashlib.sha256(
            _executor_canonical_json(key_rows)
        ).hexdigest(),
        "executor_private_seal_validated": True,
        "same_recovery_object_bound": True,
        "same_actor_object_bound": True,
        "one_shot_capability_unconsumed": True,
        "formal_identity_entropy_draw_count": 0,
        "raw_seed_bytes_read_by_r6_orchestrator": False,
        "raw_seed_sha256_computed": False,
        "purpose_actor_start_attempted": False,
        "m3_start_invoked": False,
    }
    fields["qualification_fingerprint_sha256"] = hashlib.sha256(
        _canonical_json(fields)
    ).hexdigest()
    return fields


def _qualify_acquired_r6_formal_prefix_v1(
    *,
    recovery: PendingCeremonyRecoveryV1,
    actors: A8R6RecoveryDockerActorsV1,
    amendment_commit: str,
    incident_raw: bytes,
    validation_raw: bytes,
    validation: Mapping[str, object],
    unchanged_inputs: Mapping[str, str],
    rust_formal_replay_binary: Path,
) -> tuple[
    dict[str, object],
    bytes,
    dict[str, object],
    bytes,
    dict[str, object],
    _executor._PrevalidatedPendingRecoveryFormalPrefixV1,
]:
    """Run both sealed pre-consumption qualifications under the recovery lock."""

    if (
        recovery.lock_descriptor < 0
        or recovery.basis_commit != A8_BASIS_COMMIT
        or recovery.run_id != FIXED_RUN_ID
        or recovery.ledger_id != FIXED_LEDGER_ID
        or recovery.marker_snapshot.state != "PENDING"
        or recovery.journal_state != "RESERVED"
    ):
        _fail("R6 acquired recovery is not fixed PENDING/RESERVED")
    static_dual = _executor._prevalidate_pending_recovery_static_dual_v1(
        recovery=recovery,
        actors=actors,
        static_rust_binary_path=rust_formal_replay_binary,
    )
    static_fields = _static_qualification_fields_v1(
        amendment_commit=amendment_commit,
        static_dual=static_dual,
    )
    static_raw = _receipt_record_bytes_v1(static_fields)
    source_admission = _build_source_admission_v1(
        amendment_commit=amendment_commit,
        incident_raw=incident_raw,
        validation_raw=validation_raw,
        static_qualification_raw=static_raw,
        validation=validation,
        unchanged_inputs=unchanged_inputs,
    )
    formal_prefix = _executor._prevalidate_pending_recovery_formal_prefix_v1(
        recovery=recovery,
        actors=actors,
        source_admission=source_admission,
        static_dual=static_dual,
    )
    capability_fields = _source_capability_qualification_fields_v1(
        amendment_commit=amendment_commit,
        source_admission=source_admission,
        static_qualification_raw=static_raw,
        formal_prefix=formal_prefix,
    )
    capability_raw = _receipt_record_bytes_v1(capability_fields)
    return (
        static_fields,
        static_raw,
        capability_fields,
        capability_raw,
        source_admission,
        formal_prefix,
    )


def _authorization_request_fields(
    *, amendment_commit: object, preflight_raw: bytes, incident_raw: bytes,
    validation_raw: bytes, static_qualification_raw: bytes,
    source_capability_qualification_raw: bytes,
) -> dict[str, object]:
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-authorization-request/1",
        "amendment_commit": amendment_commit,
        "parent_r5_amendment_commit": R5_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 6,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r5_terminal_chain_root_sha256": R5_TERMINAL_CHAIN_ROOT_SHA256,
        "r5_failure_raw_sha256": R5_TERMINAL_AUDIT_RAW_SHA256["failure.json"],
        "r5_failure_receipt_sha256": R5_TERMINAL_AUDIT_RECEIPT_SHA256["failure.json"],
        "r5_failure_code": R5_FAILURE_CODE,
        "r5_failure_phase": R5_FAILURE_PHASE,
        "r5_failure_detail_sha256": R5_RECORDED_PRIMARY_DETAIL_SHA256,
        "r5_formal_failure_evidence_sha256": R5_FORMAL_FAILURE_EVIDENCE_SHA256,
        "r5_final_close_failure_code": R5_FINAL_CLOSE_FAILURE_CODE,
        "r5_final_close_failure_detail_sha256": R5_FINAL_CLOSE_DETAIL_SHA256,
        "r5_primary_status": R5_PRIMARY_STATUS,
        "r5_diagnosis_authority": R5_DIAGNOSIS_AUTHORITY,
        "r5_diagnosed_primary_exception_sha256": R5_DIAGNOSED_PRIMARY_EXCEPTION_SHA256,
        "r5_diagnosed_primary_detail_sha256": R5_DIAGNOSED_PRIMARY_DETAIL_SHA256,
        "continuation_action": CONTINUATION_ACTION,
        "preflight_sha256": hashlib.sha256(preflight_raw).hexdigest(),
        "incident_diagnostic_sha256": hashlib.sha256(incident_raw).hexdigest(),
        "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
        "static_preconsumption_qualification_sha256": hashlib.sha256(
            static_qualification_raw
        ).hexdigest(),
        "source_capability_qualification_sha256": hashlib.sha256(
            source_capability_qualification_raw
        ).hexdigest(),
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
    validation_raw: bytes, static_qualification_raw: bytes,
    source_capability_qualification_raw: bytes, request_raw: bytes,
) -> dict[str, object]:
    return {
        **_authorization_request_fields(
            amendment_commit=amendment_commit,
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
            static_qualification_raw=static_qualification_raw,
            source_capability_qualification_raw=(
                source_capability_qualification_raw
            ),
        ),
        "schema": f"{AUDIT_SCHEMA_PREFIX}-authorization/1",
        "authorization_request_sha256": hashlib.sha256(request_raw).hexdigest(),
        "authorization_actor": "PROJECT_OWNER",
        "owner_authorized_fixed_transaction_only": True,
        "ordinary_execute_invoked": False,
    }


def prepare_fixed_a8_r6_authorization_v1(
    *, audit_directory: Path, custody_directory: Path,
    public_evidence_path: Path, public_promotion_path: Path,
    rust_formal_replay_binary: Path,
    rust_bridge_dag_replay_binary: Path,
    rust_bridge_dag_qualification_report: Path,
    repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> None:
    preflight = inspect_r6_source_preflight_v1(
        repository_root=repository_root, manifest_path=manifest_path
    )
    incident = _build_incident_diagnostic_v1(
        custody_directory=custody_directory,
        public_evidence_path=public_evidence_path,
        public_promotion_path=public_promotion_path,
    )
    request, _actor, _errata, _bundle = _validation_request_from_incident_v1(incident)
    validation, validation_raw = _run_a8_validator_v1(request)
    _r5_terminal_chain_snapshot_v1()
    runtime_rows = _validate_runtime_artifacts_before_attempt_v1(
        rust_formal_replay_binary=rust_formal_replay_binary,
        rust_bridge_dag_replay_binary=rust_bridge_dag_replay_binary,
        rust_bridge_dag_qualification_report=rust_bridge_dag_qualification_report,
        expected_bindings=incident.get("runtime_artifact_bindings"),
    )
    if _stable_runtime_projection_v1(runtime_rows) != tuple(
        incident["live_runtime_stable_projection"]
    ):
        _fail("R6 prepare runtime stable projection differs")
    unchanged_inputs = _unchanged_a8_input_bindings_v1()
    preflight_raw = _receipt_record_bytes_v1(preflight)
    incident_raw = _receipt_record_bytes_v1(incident)
    actors = A8R6RecoveryDockerActorsV1(
        basis_commit=A8_BASIS_COMMIT,
        custody_directory=custody_directory,
        rust_formal_replay_binary=rust_formal_replay_binary,
        rust_bridge_dag_replay_binary=rust_bridge_dag_replay_binary,
        rust_bridge_dag_qualification_report=rust_bridge_dag_qualification_report,
        timestamp=0,
    )
    primary: BaseException | None = None
    try:
        with acquire_pending_ceremony_recovery_v1(
            custody_directory=custody_directory,
            public_evidence_path=public_evidence_path,
            public_promotion_path=public_promotion_path,
            actors=actors,
        ) as recovery:
            (
                _static_qualification,
                static_qualification_raw,
                _capability_qualification,
                capability_qualification_raw,
                _source_admission,
                _formal_prefix,
            ) = _qualify_acquired_r6_formal_prefix_v1(
                recovery=recovery,
                actors=actors,
                amendment_commit=str(preflight["amendment_commit"]),
                incident_raw=incident_raw,
                validation_raw=validation_raw,
                validation=validation,
                unchanged_inputs=unchanged_inputs,
                rust_formal_replay_binary=rust_formal_replay_binary,
            )
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            actors.close()
        except BaseException as cleanup:
            if primary is not None:
                raise _combine_failures_v1(
                    primary,
                    cleanup,
                    phase="R6_PREPARE_QUALIFICATION_CLEANUP",
                ) from primary
            raise
    audit = _create_or_resume_prepare_audit_directory(audit_directory, repository_root)
    order = (
        "preflight.json", "incident-diagnostic.json",
        "a8-validation-receipt.json", "static-qualification.json",
        "source-capability-qualification.json",
        "authorization-request.json",
    )
    observed = {path.name for path in audit.iterdir()}
    allowed = set(order) | {"." + name + ".next" for name in order}
    if not observed.issubset(allowed):
        _fail("R6 preparation audit contains a non-prefix record")
    visible = [name in observed for name in order]
    if any(visible[index] and not all(visible[:index]) for index in range(len(order))):
        _fail("R6 preparation visible records are not an exact prefix")
    request_raw = _receipt_record_bytes_v1(
        _authorization_request_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
            static_qualification_raw=static_qualification_raw,
            source_capability_qualification_raw=capability_qualification_raw,
        )
    )
    for name, payload in zip(
        order,
        (
            preflight_raw,
            incident_raw,
            validation_raw,
            static_qualification_raw,
            capability_qualification_raw,
            request_raw,
        ),
        strict=True,
    ):
        _install_prepare_record_v1(audit / name, payload)


def write_fixed_a8_r6_owner_authorization_v1(
    *, audit_directory: Path, owner_confirmation: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    if owner_confirmation != OWNER_CONFIRMATION:
        _fail("R6 owner confirmation phrase differs")
    audit = _require_existing_audit_directory(audit_directory, repository_root)
    expected = {
        "preflight.json", "incident-diagnostic.json",
        "a8-validation-receipt.json", "static-qualification.json",
        "source-capability-qualification.json",
        "authorization-request.json",
    }
    actual = {path.name for path in audit.iterdir()}
    if not expected.issubset(actual) or not actual.issubset(
        expected | {"authorization.json", ".authorization.json.next"}
    ):
        _fail("R6 pre-authorization audit path set differs")
    preflight, preflight_raw = _read_canonical_audit_v1(audit / "preflight.json")
    _incident, incident_raw = _read_canonical_audit_v1(audit / "incident-diagnostic.json")
    _validation, validation_raw, _row = _read_canonical_regular_v1(
        audit / "a8-validation-receipt.json", mode=0o600
    )
    _static_qualification, static_qualification_raw = (
        _read_canonical_audit_v1(audit / "static-qualification.json")
    )
    _capability_qualification, capability_qualification_raw = (
        _read_canonical_audit_v1(
            audit / "source-capability-qualification.json"
        )
    )
    _request, request_raw = _read_canonical_audit_v1(
        audit / "authorization-request.json"
    )
    _require_exact_receipt_raw_v1(
        request_raw,
        _authorization_request_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
            static_qualification_raw=static_qualification_raw,
            source_capability_qualification_raw=capability_qualification_raw,
        ),
        label="authorization request",
    )
    raw = _receipt_record_bytes_v1(
        _expected_authorization_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
            static_qualification_raw=static_qualification_raw,
            source_capability_qualification_raw=capability_qualification_raw,
            request_raw=request_raw,
        )
    )
    _install_prepare_record_v1(audit / "authorization.json", raw)


def _validate_final_publication_v1(**kwargs: object) -> dict[str, object]:
    try:
        return _r4._validate_final_publication_v1(**kwargs)
    except _r4.A8R4RecoveryAmendmentError as exc:
        _fail("R6 final publication replay rejected: " + exc.detail)


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
            role="FINAL_CLOSE" if evidence.get("combination_phase") == "R6_OUTER_FINAL_CLOSE" else "CLEANUP",
        ),
    )


def _failure_record_fields_v1(
    *, amendment_commit: object, attempt_start_raw: bytes,
    admission_raw: bytes | None, validation_raw: bytes,
    static_qualification_raw: bytes,
    source_capability_qualification_raw: bytes,
    failure_phase: str,
    exc: BaseException,
) -> dict[str, object]:
    evidence = _failure_evidence_v1(exc)
    leaves = _leaf_failure_rows_v1(evidence)
    primary = leaves[0]
    cleanups = [row for row in leaves[1:] if row["role"] == "CLEANUP"]
    final_close = next((row for row in leaves[1:] if row["role"] == "FINAL_CLOSE"), None)
    if failure_phase == "R6_OUTER_FINAL_CLOSE" and final_close is None:
        final_close = {**primary, "role": "FINAL_CLOSE"}
    evidence_sha = hashlib.sha256(_canonical_json(evidence)).hexdigest()
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-failure/1",
        "amendment_commit": amendment_commit,
        "parent_r5_amendment_commit": R5_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 6,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r5_terminal_chain_root_sha256": R5_TERMINAL_CHAIN_ROOT_SHA256,
        "r5_failure_raw_sha256": R5_TERMINAL_AUDIT_RAW_SHA256["failure.json"],
        "r5_failure_receipt_sha256": R5_TERMINAL_AUDIT_RECEIPT_SHA256["failure.json"],
        "r5_failure_code": R5_FAILURE_CODE,
        "r5_failure_phase": R5_FAILURE_PHASE,
        "r5_failure_detail_sha256": R5_RECORDED_PRIMARY_DETAIL_SHA256,
        "r5_formal_failure_evidence_sha256": R5_FORMAL_FAILURE_EVIDENCE_SHA256,
        "r5_final_close_failure_code": R5_FINAL_CLOSE_FAILURE_CODE,
        "r5_final_close_failure_detail_sha256": R5_FINAL_CLOSE_DETAIL_SHA256,
        "r5_primary_status": R5_PRIMARY_STATUS,
        "r5_diagnosis_authority": R5_DIAGNOSIS_AUTHORITY,
        "r5_diagnosed_primary_exception_sha256": R5_DIAGNOSED_PRIMARY_EXCEPTION_SHA256,
        "r5_diagnosed_primary_detail_sha256": R5_DIAGNOSED_PRIMARY_DETAIL_SHA256,
        "attempt_start_sha256": hashlib.sha256(attempt_start_raw).hexdigest(),
        "admission_sha256_or_null": None if admission_raw is None else hashlib.sha256(admission_raw).hexdigest(),
        "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
        "static_preconsumption_qualification_sha256": hashlib.sha256(
            static_qualification_raw
        ).hexdigest(),
        "source_capability_qualification_sha256": hashlib.sha256(
            source_capability_qualification_raw
        ).hexdigest(),
        "failure_code": primary["code"],
        "failure_phase": failure_phase,
        "failure_detail_sha256": primary["detail_sha256"],
        "formal_failure_evidence": evidence,
        "formal_failure_evidence_sha256": evidence_sha,
        "primary_failure": primary,
        "cleanup_failures": cleanups,
        "final_close_failure_or_null": final_close,
        "formal_identity_entropy_draw_count": 0,
        "raw_seed_bytes_read_by_r6_orchestrator": False,
        "raw_seed_sha256_computed": False,
        "m3_start_invoked": False,
    }


def execute_fixed_a8_r6_recovery_v1(
    *, custody_directory: Path, rust_formal_replay_binary: Path,
    rust_bridge_dag_replay_binary: Path,
    rust_bridge_dag_qualification_report: Path,
    public_evidence_path: Path, public_promotion_path: Path,
    audit_directory: Path, repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> tuple[dict[str, object], dict[str, object]]:
    preflight_now = inspect_r6_source_preflight_v1(
        repository_root=repository_root, manifest_path=manifest_path
    )
    audit = _require_existing_audit_directory(audit_directory, repository_root)
    (
        audit_directory_identity,
        pre_attempt_raws,
        pre_attempt_inodes,
    ) = _read_pre_attempt_audit_snapshot_v1(
        audit, allow_attempt_next=True
    )
    preflight_raw = pre_attempt_raws["preflight.json"]
    incident_raw = pre_attempt_raws["incident-diagnostic.json"]
    validation_raw = pre_attempt_raws["a8-validation-receipt.json"]
    static_qualification_raw = pre_attempt_raws["static-qualification.json"]
    capability_qualification_raw = pre_attempt_raws[
        "source-capability-qualification.json"
    ]
    request_raw = pre_attempt_raws["authorization-request.json"]
    authorization_raw = pre_attempt_raws["authorization.json"]
    preflight = json.loads(preflight_raw)
    validation = json.loads(validation_raw)
    if type(preflight) is not dict or type(validation) is not dict:
        _fail("R6 prepared preflight/validation objects differ")
    if preflight_raw != _receipt_record_bytes_v1(preflight_now):
        _fail("stored R6 preflight differs from current clean R6 source")
    incident_now = _build_incident_diagnostic_v1(
        custody_directory=custody_directory,
        public_evidence_path=public_evidence_path,
        public_promotion_path=public_promotion_path,
    )
    if incident_raw != _receipt_record_bytes_v1(incident_now):
        _fail("stored R6 incident canonical bytes differ before attempt")
    validation_request, actor_report, errata_report, _expected_bundle = (
        _validation_request_from_incident_v1(incident_now)
    )
    _validation_now, validation_now_raw = _run_a8_validator_v1(validation_request)
    if validation_now_raw != validation_raw:
        _fail("prepare/execute isolated A8 validation receipts differ")
    _require_exact_receipt_raw_v1(
        request_raw,
        _authorization_request_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
            static_qualification_raw=static_qualification_raw,
            source_capability_qualification_raw=capability_qualification_raw,
        ),
        label="stored authorization request",
    )
    _require_exact_receipt_raw_v1(
        authorization_raw,
        _expected_authorization_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
            static_qualification_raw=static_qualification_raw,
            source_capability_qualification_raw=capability_qualification_raw,
            request_raw=request_raw,
        ),
        label="stored owner authorization",
    )
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
    _r5_terminal_chain_snapshot_v1()
    unchanged_inputs = _unchanged_a8_input_bindings_v1()

    actors: A8R6RecoveryDockerActorsV1 | None = None
    attempt_start_raw: bytes | None = None
    admission_raw: bytes | None = None
    finalize_raw: bytes | None = None
    payload: dict[str, object] | None = None
    promotion: dict[str, object] | None = None
    final: dict[str, object] | None = None
    primary_exc: BaseException | None = None
    failure_phase = "PRE_ATTEMPT_ACQUIRE"
    try:
        actors = A8R6RecoveryDockerActorsV1(
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
                _fail("R6 acquired recovery is not the fixed A8 PENDING/RESERVED transaction")
            actors.timestamp = recovery.marker_snapshot.created_at_unix_seconds
            if (
                _canonical_json(_transport(recovery.prestage_intent_fields.get("actor_qualification_report")))
                != _canonical_json(actor_report)
                or _canonical_json(_transport(recovery.prestage_intent_fields.get("errata_qualification_report")))
                != _canonical_json(errata_report)
            ):
                _fail("R6 acquired diagnostic reports differ from A8 receipt")
            (
                _static_qualification_now,
                static_qualification_now_raw,
                _capability_qualification_now,
                capability_qualification_now_raw,
                source_admission,
                formal_prefix,
            ) = _qualify_acquired_r6_formal_prefix_v1(
                recovery=recovery,
                actors=actors,
                amendment_commit=str(preflight["amendment_commit"]),
                incident_raw=incident_raw,
                validation_raw=validation_raw,
                validation=validation,
                unchanged_inputs=unchanged_inputs,
                rust_formal_replay_binary=rust_formal_replay_binary,
            )
            if static_qualification_now_raw != static_qualification_raw:
                _fail(
                    "stored R6 static pre-consumption qualification differs"
                )
            if (
                capability_qualification_now_raw
                != capability_qualification_raw
            ):
                _fail("stored R6 source-capability qualification differs")
            _recheck_pre_attempt_audit_under_lock_v1(
                audit=audit,
                expected_directory_identity=audit_directory_identity,
                expected_raws=pre_attempt_raws,
                expected_inodes=pre_attempt_inodes,
            )
            common = {
                "amendment_commit": preflight["amendment_commit"],
                "parent_r5_amendment_commit": R5_AMENDMENT_COMMIT,
                "formal_repository_commit": A8_BASIS_COMMIT,
                "run_id_hex": FIXED_RUN_ID_HEX,
                "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                "recovery_attempt_ordinal": 6,
                "authorization_revision_id": AUTHORIZATION_REVISION_ID,
                "r5_terminal_chain_root_sha256": R5_TERMINAL_CHAIN_ROOT_SHA256,
                "r5_failure_raw_sha256": R5_TERMINAL_AUDIT_RAW_SHA256["failure.json"],
                "r5_failure_receipt_sha256": R5_TERMINAL_AUDIT_RECEIPT_SHA256["failure.json"],
                "r5_failure_code": R5_FAILURE_CODE,
                "r5_failure_phase": R5_FAILURE_PHASE,
                "r5_failure_detail_sha256": R5_RECORDED_PRIMARY_DETAIL_SHA256,
                "r5_formal_failure_evidence_sha256": R5_FORMAL_FAILURE_EVIDENCE_SHA256,
                "r5_final_close_failure_code": R5_FINAL_CLOSE_FAILURE_CODE,
                "r5_final_close_failure_detail_sha256": R5_FINAL_CLOSE_DETAIL_SHA256,
                "r5_primary_status": R5_PRIMARY_STATUS,
                "r5_diagnosis_authority": R5_DIAGNOSIS_AUTHORITY,
                "r5_diagnosed_primary_exception_sha256": R5_DIAGNOSED_PRIMARY_EXCEPTION_SHA256,
                "r5_diagnosed_primary_detail_sha256": R5_DIAGNOSED_PRIMARY_DETAIL_SHA256,
                "static_preconsumption_qualification_sha256": hashlib.sha256(
                    static_qualification_raw
                ).hexdigest(),
                "source_capability_qualification_sha256": hashlib.sha256(
                    capability_qualification_raw
                ).hexdigest(),
            }
            failure_phase = "ATTEMPT_START_DURABILITY"
            attempt, attempt_start_raw = _r4._r31._build_exact_audit_record_v1(
                {
                    "schema": f"{AUDIT_SCHEMA_PREFIX}-attempt-start/1",
                    **common,
                    "continuation_action": CONTINUATION_ACTION,
                    "authorization_sha256": hashlib.sha256(authorization_raw).hexdigest(),
                    "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
                    "static_preconsumption_qualification_sha256": hashlib.sha256(
                        static_qualification_raw
                    ).hexdigest(),
                    "source_capability_qualification_sha256": hashlib.sha256(
                        capability_qualification_raw
                    ).hexdigest(),
                    "runtime_artifact_metadata": runtime_rows,
                    "runtime_artifact_stable_projection": _stable_runtime_projection_v1(runtime_rows),
                    "runtime_long_lived_identity_excludes": ["st_dev", "st_ino"],
                    "runtime_descriptor_bound_toctou_verified": True,
                    "complete_seed_resume_only": True,
                    "accepted_worker_mode": "REAL_PENDING_RESUME",
                    "formal_identity_entropy_draw_count": 0,
                    "ordinary_execute_invoked": False,
                    "m3_start_invoked": False,
                    "raw_seed_bytes_read_by_r6_orchestrator": False,
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
                    "raw_seed_bytes_read_by_r6_orchestrator": False,
                    "raw_seed_sha256_computed": False,
                    "m3_start_invoked": False,
                }
            )
            _install_exact_audit_record_v1(audit / "admission.json", admission, admission_raw)

            failure_phase = "COMPLETE_ONLY_FORMAL_CORE"
            payload, promotion = _continue_pre_stage_pending_recovery_core_v1(
                recovery=recovery,
                actors=actors,
                complete_seed_resume_only=True,
                static_rust_binary_path=rust_formal_replay_binary,
                prevalidated_formal_prefix=formal_prefix,
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
                failure_phase = "R6_OUTER_FINAL_CLOSE"
            else:
                primary_exc = _combine_failures_v1(
                    primary_exc, close_exc, phase="R6_OUTER_FINAL_CLOSE"
                )
        actors = None

    if primary_exc is None:
        try:
            if (
                payload is None or promotion is None or final is None
                or attempt_start_raw is None or admission_raw is None
            ):
                _fail("R6 successful core result is incomplete")
            failure_phase = "FINALIZE_DURABILITY"
            finalize, finalize_raw = _r4._r31._build_exact_audit_record_v1(
                {
                    "schema": f"{AUDIT_SCHEMA_PREFIX}-finalize/1",
                    "amendment_commit": preflight["amendment_commit"],
                    "parent_r5_amendment_commit": R5_AMENDMENT_COMMIT,
                    "formal_repository_commit": A8_BASIS_COMMIT,
                    "run_id_hex": FIXED_RUN_ID_HEX,
                    "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                    "recovery_attempt_ordinal": 6,
                    "authorization_revision_id": AUTHORIZATION_REVISION_ID,
                    "r5_terminal_chain_root_sha256": R5_TERMINAL_CHAIN_ROOT_SHA256,
                    "r5_failure_raw_sha256": R5_TERMINAL_AUDIT_RAW_SHA256["failure.json"],
                    "r5_failure_receipt_sha256": R5_TERMINAL_AUDIT_RECEIPT_SHA256["failure.json"],
                    "r5_formal_failure_evidence_sha256": R5_FORMAL_FAILURE_EVIDENCE_SHA256,
                    "r5_final_close_failure_code": R5_FINAL_CLOSE_FAILURE_CODE,
                    "r5_final_close_failure_detail_sha256": R5_FINAL_CLOSE_DETAIL_SHA256,
                    "attempt_start_sha256": hashlib.sha256(attempt_start_raw).hexdigest(),
                    "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
                    "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
                    "static_preconsumption_qualification_sha256": hashlib.sha256(
                        static_qualification_raw
                    ).hexdigest(),
                    "source_capability_qualification_sha256": hashlib.sha256(
                        capability_qualification_raw
                    ).hexdigest(),
                    **final,
                    "formal_gates_after": 24,
                    "child_state": "NOT_RUN",
                    "m3_start_invoked": False,
                    "accepted_worker_mode": "REAL_PENDING_RESUME",
                    "raw_seed_bytes_read_by_r6_orchestrator": False,
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
                phase="R6_FINALIZE_VISIBILITY_RESOLUTION",
            )
        )
    if finalize_visible:
        post_primary_call(
            lambda: _install_prepare_record_v1(
                audit / "finalize.json", finalize_raw
            ),
            phase="R6_FINALIZE_VISIBLE_REPAIR",
        )
        if payload is None or promotion is None:
            _fail("visible R6 finalize lacks return payload")
        return payload, promotion
    if finalize_raw is not None:
        post_primary_call(
            lambda: _discard_non_authoritative_next_v1(audit / "finalize.json"),
            phase="R6_FINALIZE_HIDDEN_DISCARD",
        )
    if primary_exc is None:
        _fail("R6 execution ended without result or failure")

    attempt_visible = bool(
        attempt_start_raw is not None
        and post_primary_call(
            lambda: _exact_audit_record_is_visible_v1(
                audit / "attempt-start.json", attempt_start_raw
            ),
            phase="R6_ATTEMPT_VISIBILITY_RESOLUTION",
        )
    )
    if attempt_visible:
        post_primary_call(
            lambda: _install_prepare_record_v1(
                audit / "attempt-start.json", attempt_start_raw
            ),
            phase="R6_ATTEMPT_VISIBLE_REPAIR",
        )
        admission_visible = bool(
            admission_raw is not None
            and post_primary_call(
                lambda: _exact_audit_record_is_visible_v1(
                    audit / "admission.json", admission_raw
                ),
                phase="R6_ADMISSION_VISIBILITY_RESOLUTION",
            )
        )
        if admission_visible:
            post_primary_call(
                lambda: _install_prepare_record_v1(
                    audit / "admission.json", admission_raw
                ),
                phase="R6_ADMISSION_VISIBLE_REPAIR",
            )
        elif admission_raw is not None:
            post_primary_call(
                lambda: _discard_non_authoritative_next_v1(
                    audit / "admission.json"
                ),
                phase="R6_ADMISSION_HIDDEN_DISCARD",
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
                    static_qualification_raw=static_qualification_raw,
                    source_capability_qualification_raw=(
                        capability_qualification_raw
                    ),
                    failure_phase=failure_phase,
                    exc=primary_exc,
                )
            ),
            phase="R6_FAILURE_EVIDENCE_BUILD",
        )
        if type(built_failure) is not tuple or len(built_failure) != 2:
            mismatch = A8R6RecoveryAmendmentError(
                FAIL_AMENDMENT, "R6 failure record builder result differs"
            )
            combined = _combine_failures_v1(
                primary_exc, mismatch, phase="R6_FAILURE_EVIDENCE_BUILD"
            )
            raise combined from primary_exc
        failure, failure_raw = built_failure
        failure_exists = bool(
            post_primary_call(
                lambda: lstat_path_exists(failure_path),
                phase="R6_FAILURE_PATH_PROBE",
            )
        )
        if failure_exists:
            existing_failure, existing_raw = post_primary_call(
                lambda: _read_canonical_audit_v1(failure_path),
                phase="R6_EXISTING_FAILURE_VALIDATION",
            )
            if existing_failure != failure or existing_raw != failure_raw:
                mismatch = A8R6RecoveryAmendmentError(
                    FAIL_AMENDMENT,
                    "existing R6 failure record differs from exact primary evidence",
                )
                combined = _combine_failures_v1(
                    primary_exc, mismatch, phase="R6_EXISTING_FAILURE_CONFLICT"
                )
                raise combined from primary_exc
        else:
            try:
                _install_exact_audit_record_v1(failure_path, failure, failure_raw)
            except BaseException as audit_exc:
                accumulated = _combine_failures_v1(
                    primary_exc, audit_exc, phase="R6_FAILURE_AUDIT_DURABILITY"
                )
                try:
                    failure_visible = _exact_audit_record_is_visible_v1(
                        failure_path, failure_raw
                    )
                except BaseException as visibility_exc:
                    combined = _combine_failures_v1(
                        accumulated,
                        visibility_exc,
                        phase="R6_FAILURE_VISIBILITY_RESOLUTION",
                    )
                    raise combined from primary_exc
                if not failure_visible:
                    try:
                        _discard_non_authoritative_next_v1(failure_path)
                    except BaseException as discard_exc:
                        combined = _combine_failures_v1(
                            accumulated,
                            discard_exc,
                            phase="R6_FAILURE_HIDDEN_DISCARD",
                        )
                        raise combined from primary_exc
                    try:
                        _install_prepare_record_v1(failure_path, failure_raw)
                    except BaseException as retry_exc:
                        combined = _combine_failures_v1(
                            accumulated,
                            retry_exc,
                            phase="R6_FAILURE_CANONICAL_RETRY",
                        )
                        raise combined from primary_exc
                raise accumulated from primary_exc
    raise primary_exc


__all__ = [
    "A8R6RecoveryAmendmentError",
    "DEFAULT_MANIFEST_PATH",
    "execute_fixed_a8_r6_recovery_v1",
    "inspect_fixed_a8_r6_preflight_v1",
    "inspect_r6_source_preflight_v1",
    "prepare_fixed_a8_r6_authorization_v1",
    "write_fixed_a8_r6_owner_authorization_v1",
]
