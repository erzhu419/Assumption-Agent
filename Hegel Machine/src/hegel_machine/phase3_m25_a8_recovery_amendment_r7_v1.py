"""Sealed ordinal-7 post-stage recovery after the terminal R6 failure.

R6 consumed ordinal 6 after it had durably staged the complete public payload.
Its formal core then failed while inspecting actor-owned custody.  This
amendment binds the exact ten-record R6 terminal chain and permits only an
idempotent continuation of that already-staged transaction.  It creates no
opaque identity, key, signature, seed, static qualification, source
qualification, or M3 state.
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

from . import phase3_m25_a8_recovery_amendment_r6_v1 as _r6
from . import phase3_m25_formal_container_executor_v1 as _executor
from .phase3_m25_formal_container_executor_v1 import (
    FormalCeremonyTransactionV1,
    FormalContainerExecutorError,
    _continue_post_stage_transaction_recovery_core_v1,
    _replay_public_gate_evidence_with_fixed_a8_r6_direct_child_basis_v1,
)
from .phase3_m25_formal_static_basis_v1 import (
    FORMAL_GIT_ENVIRONMENT_V1,
    FORMAL_GIT_EXECUTABLE,
)


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
DEFAULT_MANIFEST_PATH: Final = (
    PROJECT_ROOT / "config/phase3_m25_a8_recovery_amendment_r7_v1.json"
)
A8_BASIS_COMMIT: Final = _r6.A8_BASIS_COMMIT
R6_AMENDMENT_COMMIT: Final = "e10fa89575af19c85e9744533e16d648463be451"
FIXED_RUN_ID_HEX: Final = _r6.FIXED_RUN_ID_HEX
FIXED_LEDGER_ID_HEX: Final = _r6.FIXED_LEDGER_ID_HEX
FIXED_RUN_ID: Final = _r6.FIXED_RUN_ID
FIXED_LEDGER_ID: Final = _r6.FIXED_LEDGER_ID
FIXED_FORMAL_RUST_BINARY: Final = _r6.FIXED_FORMAL_RUST_BINARY
FIXED_R6_AUDIT_DIRECTORY: Final = _r6.FIXED_R6_AUDIT_DIRECTORY
FIXED_R7_AUDIT_DIRECTORY: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/"
    "phase3-m25-0af65964235390ce2bebefea7379eaa9c50eda24/"
    "recovery-audit-r7-e4af9f57c38fb298462ec628c4ed8a03-attempt-7"
)
FIXED_STAGE_DIRECTORY: Final = Path(
    "/home/erzhu419/mine_code/Asumption Agent/Hegel Machine/"
    "artifacts/phase3_m25_external/formal_genesis_v2/"
    ".hegel-m25-stage-e4af9f57c38fb298462ec628c4ed8a03"
)
MANIFEST_SCHEMA: Final = "hegel-phase3-m25-a8-recovery-amendment-r7/1"
AUDIT_SCHEMA_PREFIX: Final = "hegel-phase3-m25-a8-r7-poststage-recovery-audit"
FAIL_AMENDMENT: Final = "FAIL_M25_A8_R7_POSTSTAGE_RECOVERY_AMENDMENT"
OWNER_CONFIRMATION: Final = (
    "AUTHORIZE_A8_R7_ATTEMPT_7_FIXED_STAGED_POSTSTAGE_PENDING_"
    "IDEMPOTENT_CONTINUATION_ONLY"
)
CONTINUATION_ACTION: Final = "POST_R6_TERMINAL_FIXED_STAGED_POSTSTAGE_CONTINUATION"
AUTHORIZATION_REVISION_ID: Final = "R7_FIXED_STAGED_POSTSTAGE_ONLY_V1"
ACCEPTED_WORKER_MODE: Final = "FIXED_STAGED_POSTSTAGE_PENDING_RESUME"
FIXED_PRESTAGE_INTENT_SHA256: Final = (
    "89d8414cd68adaa084b3dfe865abf5d9245806764d89413dc7d1503e6dffc0ab"
)
FIXED_R6_SOURCE_ADMISSION_SHA256: Final = (
    "c2003bcc77db04c2672d59d1aada6e45baeb4275df6f8a3f8304c68f8ef26828"
)

R6_TERMINAL_AUDIT_ORDER: Final = (
    "preflight.json",
    "incident-diagnostic.json",
    "a8-validation-receipt.json",
    "static-qualification.json",
    "source-capability-qualification.json",
    "authorization-request.json",
    "authorization.json",
    "attempt-start.json",
    "admission.json",
    "failure.json",
)
R6_TERMINAL_AUDIT_RAW_SHA256: Final = {
    "preflight.json": "8801f51922bbff27c850d7120e0767251f5f2fff12c018d743dfec847d2ef3f8",
    "incident-diagnostic.json": "078a9f41a6b1b59d130229023013ec5e8c8b04e7a2d5dfe16742f519e32e4a2e",
    "a8-validation-receipt.json": "ef18694aa41a78389cef2265eb121174f2e68548928f89f7fcad3f55fb261ee4",
    "static-qualification.json": "8e6e147e4e8c6795282becdac6b4ffcfab30ca0ea2fd53c8e467d2500b096195",
    "source-capability-qualification.json": "372196e69cfd6f51b78415385e3626774cde8b4c87559b409a993d6d9bfdd994",
    "authorization-request.json": "b39694f956009d85850f30a8eabbc5aa02946a97be500e286bf0997f42fd3b2d",
    "authorization.json": "371f64c9eb136ec12b9c95834f8882dfdf06f3e82081ec4dc2f1e4cd3563d8a5",
    "attempt-start.json": "515a9db53127680700c9c641dc69e85288e6d33988d8de76a34592b4dbbb0f02",
    "admission.json": "58fb263c85c3daee7be06b5cf7fada311aa5aa2546a2bed82444676156a98d6d",
    "failure.json": "72ff3cd32994ad112c2a2998a8d932262cc69197f6c3a0f9b3785694761b5796",
}
R6_TERMINAL_AUDIT_RECEIPT_SHA256: Final = {
    "preflight.json": "880232875050a0e2e28caa3c4c3f6dc067151ac54c34868a11ed6a2fdf5d39ff",
    "incident-diagnostic.json": "0e6c5cf1350c5ef1e1aca6dfa64a9e178aa0b52d812f1c1c0e3b7817ba1db6a6",
    "a8-validation-receipt.json": "83b1ad690914d9dfd5cd402d5c734a1250a3b450c9e0b3ecf4655cfb97c6ba47",
    "static-qualification.json": "e87c230bf778a2d0918070b5dc791a3e7e8831dcd5bb17303ef808305bd7a3c1",
    "source-capability-qualification.json": "e73a5e6b52160517dd2e2038ce9290d02ae6ebda6d2353e9550431304554c159",
    "authorization-request.json": "e1e71954b450eda55289fa4c4e1914030df78729d8ac0a29c4c774650a7b6a89",
    "authorization.json": "2fec3ab77e8e557bcbb1a99da3d1303ea78f36324ce665fef6e8bef884649b43",
    "attempt-start.json": "dfd8041c8570dc357af37080538546f606749eb7921e79b1a9f2a29d4cc5270a",
    "admission.json": "afde75fa16aadcc41d7ed40ab820bfde16ff07b2e21b2f0cd123a1ed85dbe751",
    "failure.json": "ab404b602683b21d685d2579bfc9fbcad07c2e5d9992a68f05c22a9f20fe413a",
}
R6_TERMINAL_AUDIT_SIZE_BYTES: Final = {
    "preflight.json": 8721,
    "incident-diagnostic.json": 26982,
    "a8-validation-receipt.json": 14588,
    "static-qualification.json": 3798,
    "source-capability-qualification.json": 2252,
    "authorization-request.json": 2488,
    "authorization.json": 2696,
    "attempt-start.json": 4670,
    "admission.json": 18068,
    "failure.json": 3108,
}
R6_TERMINAL_CHAIN_ROOT_SHA256: Final = (
    "d17b2fc442226b1800f7f4900b52dbca824f5391ca0ab0ec1d4f6fc034711de2"
)
R6_FAILURE_CODE: Final = "FAIL_M25_FORMAL_CUSTODY_STATE"
R6_FAILURE_PHASE: Final = "COMPLETE_ONLY_FORMAL_CORE"
R6_FAILURE_DETAIL_SHA256: Final = (
    "8e0d912ce22e0abe4f684bb889920b586baba13a80db6c046d5c9777e3f41415"
)
R6_FORMAL_FAILURE_EVIDENCE_SHA256: Final = (
    "efb520477e05a6d67ccba40eb29157b15b15e105c333a09ad922e7766525bcd9"
)
R7_RUNTIME_EXCEPTION_PATHS: Final = frozenset(
    {
        *_r6.R6_RUNTIME_EXCEPTION_PATHS,
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_amendment_r7_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_cli_r7_v1.py",
    }
)
_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")


class A8R7RecoveryAmendmentError(FormalContainerExecutorError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(code, detail)


class _R7TerminalAuthorityResolutionError(Exception):
    """Propagate an audit error without authorizing a second terminal record."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(type(error).__name__)
        self.error = error


class A8R7RecoveryDockerActorsV1(_r6.A8R6RecoveryDockerActorsV1):
    """R7 actor; R6's successful-close latch prevents an outer double close."""


def _fail(detail: str) -> NoReturn:
    raise A8R7RecoveryAmendmentError(FAIL_AMENDMENT, detail)


def _canonical_json(value: object) -> bytes:
    return _executor._canonical_json(value)


def _receipt_record_bytes_v1(fields: Mapping[str, object]) -> bytes:
    return _canonical_json(_r6._r4._r31._r2._with_receipt_sha256(fields))


def _require_exact_receipt_raw_v1(
    stored_raw: bytes, expected_fields: Mapping[str, object], *, label: str
) -> None:
    if type(stored_raw) is not bytes or stored_raw != _receipt_record_bytes_v1(
        expected_fields
    ):
        _fail(f"R7 {label} canonical bytes differ")


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
        _fail("formal Git check failed: " + completed.stderr.decode("utf-8", "replace")[-400:])
    return completed.stdout


def _manifest_fixed_policy_v1() -> dict[str, object]:
    return {
        "schema": MANIFEST_SCHEMA,
        "source_commit_selector": "HEAD",
        "sole_parent_commit": R6_AMENDMENT_COMMIT,
        "parent_amendment_commit": R6_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "fixed_run_id_hex": FIXED_RUN_ID_HEX,
        "fixed_ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 7,
        "fixed_r6_terminal_audit_directory": FIXED_R6_AUDIT_DIRECTORY.as_posix(),
        "fixed_r7_audit_directory": FIXED_R7_AUDIT_DIRECTORY.as_posix(),
        "r6_terminal_chain_root_sha256": R6_TERMINAL_CHAIN_ROOT_SHA256,
        "r6_failure_code": R6_FAILURE_CODE,
        "r6_failure_phase": R6_FAILURE_PHASE,
        "r6_failure_detail_sha256": R6_FAILURE_DETAIL_SHA256,
        "r6_formal_failure_evidence_sha256": R6_FORMAL_FAILURE_EVIDENCE_SHA256,
        "fixed_prestage_intent_sha256": FIXED_PRESTAGE_INTENT_SHA256,
        "fixed_r6_source_admission_sha256": FIXED_R6_SOURCE_ADMISSION_SHA256,
        "continuation_action": CONTINUATION_ACTION,
        "owner_confirmation": OWNER_CONFIRMATION,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "poststage_only": True,
        "formal_identity_entropy_draw_count": 0,
        "ephemeral_container_nonce_allowed": True,
        "ordinary_execute_allowed": False,
        "prestage_core_allowed": False,
        "signing_allowed": False,
        "static_rebuild_allowed": False,
        "source_rebuild_allowed": False,
        "redraw_allowed": False,
        "m3_start_allowed": False,
    }


def _load_manifest(path: Path) -> tuple[dict[str, object], bytes]:
    if path.is_symlink():
        _fail("R7 amendment manifest may not be a symlink")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"R7 amendment manifest is invalid JSON: {exc}")
    fixed = _manifest_fixed_policy_v1()
    required = {*fixed, "exact_changed_paths", "source_bindings"}
    if type(value) is not dict or _canonical_json(value) != raw or set(value) != required:
        _fail("R7 amendment manifest is not canonical exact JSON")
    if _canonical_json({key: value[key] for key in fixed}) != _canonical_json(fixed):
        _fail("R7 amendment manifest fixed policy differs")
    if type(value.get("exact_changed_paths")) is not list or type(value.get("source_bindings")) is not list:
        _fail("R7 amendment manifest source allowlists differ")
    return value, raw


def _r6_terminal_chain_snapshot_v1() -> tuple[dict[str, object], ...]:
    audit = FIXED_R6_AUDIT_DIRECTORY
    if audit.is_symlink():
        _fail("R6 terminal audit directory may not be a symlink")
    metadata = audit.stat()
    if (
        audit.resolve(strict=True) != FIXED_R6_AUDIT_DIRECTORY
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
        or {path.name for path in audit.iterdir()} != set(R6_TERMINAL_AUDIT_ORDER)
    ):
        _fail("R6 terminal audit is not the exact ten-record chain")
    records: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for name in R6_TERMINAL_AUDIT_ORDER:
        value, raw = _r6._read_canonical_audit_v1(audit / name)
        item = (audit / name).stat()
        raw_sha = hashlib.sha256(raw).hexdigest()
        receipt = value.get("receipt_sha256")
        if (
            raw_sha != R6_TERMINAL_AUDIT_RAW_SHA256[name]
            or receipt != R6_TERMINAL_AUDIT_RECEIPT_SHA256[name]
            or item.st_size != R6_TERMINAL_AUDIT_SIZE_BYTES[name]
            or stat.S_IMODE(item.st_mode) != 0o600
            or item.st_uid != os.getuid()
            or item.st_gid != os.getgid()
            or item.st_nlink != 1
        ):
            _fail(f"R6 terminal audit identity differs: {name}")
        records[name] = value
        rows.append({
            "name": name,
            "raw_sha256": raw_sha,
            "receipt_sha256": receipt,
            "size_bytes": item.st_size,
            "mode_octal": "0600",
        })
    failure = records["failure.json"]
    admission = records["admission.json"]
    source_admission = admission.get("source_admission")
    common = {
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
    }
    if (
        any(any(record.get(key) != expected for key, expected in common.items()) for record in records.values())
        or any(record.get("recovery_attempt_ordinal") != 6 for name, record in records.items() if name != "a8-validation-receipt.json")
        or records["preflight.json"].get("amendment_commit") != R6_AMENDMENT_COMMIT
        or admission.get("attempt_start_sha256") != R6_TERMINAL_AUDIT_RAW_SHA256["attempt-start.json"]
        or admission.get("source_admission_sha256") != FIXED_R6_SOURCE_ADMISSION_SHA256
        or type(source_admission) is not dict
        or hashlib.sha256(_canonical_json(source_admission)).hexdigest() != FIXED_R6_SOURCE_ADMISSION_SHA256
        or failure.get("attempt_start_sha256") != R6_TERMINAL_AUDIT_RAW_SHA256["attempt-start.json"]
        or failure.get("admission_sha256_or_null") != R6_TERMINAL_AUDIT_RAW_SHA256["admission.json"]
        or failure.get("failure_code") != R6_FAILURE_CODE
        or failure.get("failure_phase") != R6_FAILURE_PHASE
        or failure.get("failure_detail_sha256") != R6_FAILURE_DETAIL_SHA256
        or failure.get("formal_failure_evidence_sha256") != R6_FORMAL_FAILURE_EVIDENCE_SHA256
        or failure.get("final_close_failure_or_null") is not None
        or failure.get("cleanup_failures") != []
        or failure.get("raw_seed_bytes_read_by_r6_orchestrator") is not False
        or failure.get("raw_seed_sha256_computed") is not False
        or failure.get("m3_start_invoked") is not False
    ):
        _fail("R6 terminal audit provenance or terminal fields differ")
    if hashlib.sha256(_canonical_json(rows)).hexdigest() != R6_TERMINAL_CHAIN_ROOT_SHA256:
        _fail("R6 terminal chain root differs")
    return tuple(rows)


def _fixed_r6_source_admission_v1() -> dict[str, object]:
    _r6_terminal_chain_snapshot_v1()
    admission, _raw = _r6._read_canonical_audit_v1(
        FIXED_R6_AUDIT_DIRECTORY / "admission.json"
    )
    source = admission.get("source_admission")
    if type(source) is not dict or hashlib.sha256(_canonical_json(source)).hexdigest() != FIXED_R6_SOURCE_ADMISSION_SHA256:
        _fail("fixed R6 source admission differs")
    try:
        _executor._validate_recovery_source_admission_v1(
            source,
            basis_commit=A8_BASIS_COMMIT,
            run_id=FIXED_RUN_ID,
            ledger_id=FIXED_LEDGER_ID,
        )
    except FormalContainerExecutorError as exc:
        _fail("fixed R6 source admission replay rejected: " + exc.detail)
    return dict(source)


def _runtime_exception_source_bindings_v1(
    *, repository_root: Path, head: str,
) -> tuple[dict[str, object], ...]:
    paths = tuple(sorted(R7_RUNTIME_EXCEPTION_PATHS))
    if len(paths) != 17:
        _fail("R7 runtime-exception registry is not the frozen 17 paths")
    try:
        _r6._r4._r31._verify_changed_index_flags_v1(repository_root, set(paths))
    except _r6._r4._r31.A8R3RecoveryAmendmentError as exc:
        _fail("R7 runtime-exception index flags rejected: " + exc.detail)
    rows: list[dict[str, object]] = []
    for relative in paths:
        raw = _git(repository_root, ("show", f"{head}:{relative}"))
        digest = hashlib.sha256(raw).hexdigest()
        try:
            _r6._r4._r31._verify_changed_worktree_blob_v1(
                repository_root=repository_root,
                head=head,
                relative=relative,
                expected_sha256=digest,
            )
        except _r6._r4._r31.A8R3RecoveryAmendmentError as exc:
            _fail("R7 runtime-exception worktree binding rejected: " + exc.detail)
        tree_line = _git(repository_root, ("ls-tree", head, "--", relative)).decode(
            "utf-8", "strict"
        ).strip()
        parts = tree_line.split(None, 3)
        if len(parts) != 4 or parts[1] != "blob" or parts[3] != relative or parts[0] not in {"100644", "100755"}:
            _fail(f"R7 runtime-exception Git mode/blob differs: {relative}")
        rows.append({
            "path": relative,
            "git_mode": parts[0],
            "head_blob_sha256": digest,
            "worktree_sha256": digest,
            "worktree_mode_octal": "0755" if parts[0] == "100755" else "0644",
        })
    return tuple(rows)


def inspect_r7_source_preflight_v1(
    *, repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, object]:
    manifest, manifest_raw = _load_manifest(manifest_path)
    head = _git(repository_root, ("rev-parse", "--verify", "HEAD^{commit}")).decode("ascii").strip()
    parents = _git(repository_root, ("show", "-s", "--format=%P", head)).decode("ascii").strip().split()
    if _HEX_40.fullmatch(head) is None or parents != [R6_AMENDMENT_COMMIT]:
        _fail("R7 must be one committed sole child of terminal R6")
    if _git(repository_root, ("status", "--porcelain=v1", "--untracked-files=all")):
        _fail("R7 repository tree/index is not clean")
    changed_lines = _git(
        repository_root,
        ("diff-tree", "--no-commit-id", "--name-status", "-r", "--no-renames", R6_AMENDMENT_COMMIT, head),
    ).decode("utf-8", "strict").splitlines()
    actual = tuple(
        {"status": line.split("\t", 1)[0], "path": line.split("\t", 1)[1]}
        for line in changed_lines if line
    )
    if tuple(manifest["exact_changed_paths"]) != actual:
        _fail("R7 changed-path allowlist differs")
    manifest_relative = manifest_path.resolve(strict=True).relative_to(repository_root.resolve(strict=True)).as_posix()
    changed_paths = {str(row["path"]) for row in actual}
    try:
        _r6._r4._r31._verify_changed_index_flags_v1(repository_root, changed_paths)
        _r6._r4._r31._verify_changed_worktree_blob_v1(
            repository_root=repository_root,
            head=head,
            relative=manifest_relative,
            expected_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        )
    except _r6._r4._r31.A8R3RecoveryAmendmentError as exc:
        _fail("R7 changed-source preflight rejected: " + exc.detail)
    expected_paths = tuple(str(row["path"]) for row in actual if str(row["path"]) != manifest_relative)
    bindings = manifest["source_bindings"]
    if not _r6._r4._r31._source_binding_paths_are_exact_v1(bindings, expected_paths):
        _fail("R7 source bindings do not equal changed paths minus manifest")
    verified: list[dict[str, object]] = []
    for row in bindings:
        if type(row) is not dict or set(row) != {"path", "parent_sha256_or_null", "r7_sha256"}:
            _fail("R7 source-binding row differs")
        relative = row.get("path")
        parent_digest = row.get("parent_sha256_or_null")
        current_digest = row.get("r7_sha256")
        if (
            type(relative) is not str
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or (parent_digest is not None and (type(parent_digest) is not str or _HEX_64.fullmatch(parent_digest) is None))
            or type(current_digest) is not str
            or _HEX_64.fullmatch(current_digest) is None
        ):
            _fail("R7 source-binding value differs")
        try:
            _r6._r4._r31._verify_changed_worktree_blob_v1(
                repository_root=repository_root,
                head=head,
                relative=relative,
                expected_sha256=current_digest,
            )
        except _r6._r4._r31.A8R3RecoveryAmendmentError as exc:
            _fail("R7 source-binding worktree verification rejected: " + exc.detail)
        probe = subprocess.run(
            [str(FORMAL_GIT_EXECUTABLE), "cat-file", "-e", f"{R6_AMENDMENT_COMMIT}:{relative}"],
            cwd=repository_root.resolve(strict=True),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=60,
            env=dict(FORMAL_GIT_ENVIRONMENT_V1),
        )
        if parent_digest is None:
            if probe.returncode == 0:
                _fail(f"R7 source unexpectedly existed in R6: {relative}")
        elif probe.returncode != 0 or hashlib.sha256(_git(repository_root, ("show", f"{R6_AMENDMENT_COMMIT}:{relative}"))).hexdigest() != parent_digest:
            _fail(f"parent R6 source blob hash differs: {relative}")
        verified.append(dict(row))
    runtime_bindings = _runtime_exception_source_bindings_v1(
        repository_root=repository_root, head=head
    )
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-preflight/1",
        "amendment_commit": head,
        "sole_parent_commit": R6_AMENDMENT_COMMIT,
        "parent_amendment_commit": R6_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 7,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r6_terminal_chain_root_sha256": R6_TERMINAL_CHAIN_ROOT_SHA256,
        "r6_attempt6_consumed": True,
        "r6_finalize_sha256_or_null": None,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "source_bindings": verified,
        "runtime_exception_source_bindings": runtime_bindings,
        "runtime_exception_source_bindings_sha256": hashlib.sha256(_canonical_json(runtime_bindings)).hexdigest(),
        "repository_clean": True,
        "exact_changed_paths_verified": True,
        "changed_worktree_blobs_equal_head": True,
        "changed_path_index_flags_normal": True,
        "formal_identity_entropy_draw_count": 0,
        "m3_start_allowed": False,
        "fixed_audit_directory": FIXED_R7_AUDIT_DIRECTORY.as_posix(),
    }


def _require_existing_audit_directory(path: Path, repository_root: Path) -> Path:
    if path.is_symlink():
        _fail("R7 audit directory may not be a symlink")
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    repository = repository_root.resolve(strict=True)
    if (
        resolved != FIXED_R7_AUDIT_DIRECTORY
        or resolved == repository
        or repository in resolved.parents
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
    ):
        _fail("R7 audit directory is not the fixed caller-owned mode-0700 path")
    return resolved


def _create_or_resume_prepare_audit_directory(path: Path, repository_root: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute != FIXED_R7_AUDIT_DIRECTORY:
        _fail("R7 audit directory differs from fixed attempt-7 path")
    repository = repository_root.resolve(strict=True)
    parent = absolute.parent.resolve(strict=True)
    if absolute == repository or repository in absolute.parents:
        _fail("R7 audit directory must be repository-external")
    if absolute.is_symlink():
        _fail("R7 attempt-7 audit directory may not be a symlink")
    if not absolute.exists():
        os.mkdir(absolute, 0o700)
        os.chmod(absolute, 0o700)
        _r6._r4._r31._fsync_directory_v1(parent)
    return _require_existing_audit_directory(absolute, repository_root)


def _install_prepare_record_v1(path: Path, raw: bytes) -> None:
    try:
        _r6._r4._r31._install_prepare_record_v1(path, raw)
    except _r6._r4._r31.A8R3RecoveryAmendmentError as exc:
        _fail("R7 preparation install rejected: " + exc.detail)


def _install_exact_audit_record_v1(path: Path, expected: Mapping[str, object], raw: bytes) -> None:
    try:
        _r6._r4._r31._install_exact_audit_record_v1(path, expected, raw)
    except _r6._r4._r31.A8R3RecoveryAmendmentError as exc:
        _fail("R7 exact audit install rejected: " + exc.detail)


def _exact_audit_record_is_visible_v1(path: Path, raw: bytes) -> bool:
    try:
        return _r6._exact_audit_record_is_visible_v1(path, raw)
    except _r6.A8R6RecoveryAmendmentError as exc:
        _fail("R7 visible audit check rejected: " + exc.detail)


def _discard_non_authoritative_next_v1(path: Path) -> None:
    try:
        _r6._discard_non_authoritative_next_v1(path)
    except _r6.A8R6RecoveryAmendmentError as exc:
        _fail("R7 hidden audit cleanup rejected: " + exc.detail)


def _read_canonical_regular_v1(path: Path, *, mode: int) -> tuple[dict[str, object], bytes, dict[str, object]]:
    try:
        return _r6._read_canonical_regular_v1(path, mode=mode)
    except _r6.A8R6RecoveryAmendmentError as exc:
        _fail("R7 canonical regular-file read rejected: " + exc.detail)


def _read_canonical_audit_v1(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        return _r6._read_canonical_audit_v1(path)
    except _r6.A8R6RecoveryAmendmentError as exc:
        _fail("R7 canonical audit read rejected: " + exc.detail)


_R7_PRE_ATTEMPT_AUDIT_NAMES: Final = (
    "preflight.json",
    "incident-diagnostic.json",
    "poststage-qualification.json",
    "authorization-request.json",
    "authorization.json",
)


def _read_pre_attempt_audit_snapshot_v1(
    audit: Path, *, allow_attempt_next: bool,
) -> tuple[tuple[int, int, int, int, int], dict[str, bytes], dict[str, tuple[object, ...]]]:
    metadata = audit.lstat()
    directory_identity = (metadata.st_dev, metadata.st_ino, stat.S_IMODE(metadata.st_mode), metadata.st_uid, metadata.st_gid)
    observed = {path.name for path in audit.iterdir()}
    allowed = set(_R7_PRE_ATTEMPT_AUDIT_NAMES)
    if allow_attempt_next:
        allowed.add(".attempt-start.json.next")
    if not set(_R7_PRE_ATTEMPT_AUDIT_NAMES).issubset(observed) or not observed.issubset(allowed):
        _fail("R7 pre-attempt audit namespace differs")
    raws: dict[str, bytes] = {}
    inodes: dict[str, tuple[object, ...]] = {}
    for name in _R7_PRE_ATTEMPT_AUDIT_NAMES:
        value, raw, row = _read_canonical_regular_v1(audit / name, mode=0o600)
        body = dict(value)
        receipt = body.pop("receipt_sha256", None)
        if type(receipt) is not str or receipt != hashlib.sha256(_canonical_json(body)).hexdigest():
            _fail(f"R7 pre-attempt audit self-hash differs: {name}")
        raws[name] = raw
        inodes[name] = (row["st_dev"], row["st_ino"], row["mode_octal"], row["size_bytes"], row["uid"], row["gid"])
    return directory_identity, raws, inodes


def _recheck_pre_attempt_audit_under_lock_v1(
    *, audit: Path,
    expected_directory_identity: tuple[int, int, int, int, int],
    expected_raws: Mapping[str, bytes],
    expected_inodes: Mapping[str, tuple[object, ...]],
) -> None:
    try:
        _r6._discard_non_authoritative_next_v1(audit / "attempt-start.json")
    except _r6.A8R6RecoveryAmendmentError as exc:
        _fail("R7 hidden attempt cleanup rejected: " + exc.detail)
    identity, raws, inodes = _read_pre_attempt_audit_snapshot_v1(audit, allow_attempt_next=False)
    if identity != expected_directory_identity or raws != dict(expected_raws) or inodes != dict(expected_inodes):
        _fail("R7 pre-attempt audit prefix changed under transaction lock")


def _frozen_runtime_bindings_v1() -> dict[str, object]:
    r2 = _r6._r4._r31._r2
    return {
        "formal_rust_replay_binary_path": r2.FIXED_FORMAL_RUST_BINARY.as_posix(),
        "formal_rust_replay_binary_sha256": r2.FIXED_FORMAL_RUST_BINARY_SHA256,
        "rust_bridge_dag_replay_binary_path": r2.FIXED_BRIDGE_RUST_BINARY.as_posix(),
        "rust_bridge_dag_replay_binary_sha256": r2.FIXED_BRIDGE_RUST_BINARY_SHA256,
        "rust_bridge_dag_qualification_report_path": r2.FIXED_BRIDGE_REPORT.as_posix(),
        "rust_bridge_dag_qualification_report_raw_sha256": r2.FIXED_BRIDGE_REPORT_RAW_SHA256,
        "rust_bridge_dag_qualification_report_diagnostic_sha256": r2.FIXED_BRIDGE_REPORT_DIAGNOSTIC_SHA256,
    }


def _validate_runtime_artifacts_before_attempt_v1(**kwargs: object) -> tuple[dict[str, object], ...]:
    try:
        return _r6._validate_runtime_artifacts_before_attempt_v1(
            **kwargs, expected_bindings=_frozen_runtime_bindings_v1()
        )
    except _r6.A8R6RecoveryAmendmentError as exc:
        _fail("R7 runtime artifact qualification rejected: " + exc.detail)


def _stable_runtime_projection_v1(rows: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    return _r6._stable_runtime_projection_v1(rows)


def _load_fixed_prestage_intent_v1() -> dict[str, object]:
    path = FIXED_STAGE_DIRECTORY / "prestage-intent.json"
    transported, raw, _row = _read_canonical_regular_v1(path, mode=0o600)
    if hashlib.sha256(raw).hexdigest() != FIXED_PRESTAGE_INTENT_SHA256:
        _fail("fixed staged prestage-intent hash differs")
    try:
        validated = _executor.validate_prestage_intent_fields_v1(
            transported,
            basis_commit=A8_BASIS_COMMIT,
            run_id=FIXED_RUN_ID,
            ledger_id=FIXED_LEDGER_ID,
        )
    except FormalContainerExecutorError as exc:
        _fail("fixed staged prestage intent rejected: " + exc.detail)
    return dict(validated)


def _fixed_replay_v1(
    *, source_admission: Mapping[str, object],
    prestage_intent: Mapping[str, object],
    amendment_commit: str,
) -> Callable[[Mapping[str, object]], dict[str, object]]:
    source_bytes = _canonical_json(source_admission)
    intent_bytes = _canonical_json(prestage_intent)
    if type(amendment_commit) is not str or _HEX_40.fullmatch(amendment_commit) is None:
        _fail("R7 fixed replay child commit is malformed")

    def replay(candidate: Mapping[str, object]) -> dict[str, object]:
        if _canonical_json(source_admission) != source_bytes or _canonical_json(prestage_intent) != intent_bytes:
            _fail("R7 fixed replay inputs changed in-process")
        result = _replay_public_gate_evidence_with_fixed_a8_r6_direct_child_basis_v1(
            candidate,
            source_admission=source_admission,
            prestage_intent_fields=prestage_intent,
            historical_direct_child_commit=amendment_commit,
        )
        if _canonical_json(source_admission) != source_bytes or _canonical_json(prestage_intent) != intent_bytes:
            _fail("R7 fixed replay inputs changed during delegated replay")
        return result

    return replay


def _docker_snapshot_v1() -> dict[str, object]:
    try:
        return dict(_r6._r4._r31._r2._docker_read_only_state_v1())
    except _r6._r4._r31._r2.A8R2RecoveryAmendmentError as exc:
        _fail("R7 Docker continuity snapshot rejected: " + exc.detail)


def _qualify_poststage_locked_v1(
    *, transaction: FormalCeremonyTransactionV1,
    actors: A8R7RecoveryDockerActorsV1,
    amendment_commit: str,
    source_admission: Mapping[str, object],
    runtime_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    marker = transaction._recovery_marker_snapshot
    prestage = transaction._prestage_intent_fields
    frozen_daemon = None if prestage is None else prestage.get("live_actor_protocol_daemon_receipt_binding")
    if (
        transaction.basis_commit != A8_BASIS_COMMIT
        or transaction.run_id != FIXED_RUN_ID
        or transaction.ledger_id != FIXED_LEDGER_ID
        or transaction.recovery_phase != "STAGED_PENDING"
        or transaction._state != "STAGED_PROSPECTIVE_REPLAY_PASSED"
        or marker is None
        or marker.state != "PENDING"
        or transaction._lock_descriptor is None
        or transaction._stage_directory != FIXED_STAGE_DIRECTORY
        or transaction._prestage_intent_bytes is None
        or hashlib.sha256(transaction._prestage_intent_bytes).hexdigest() != FIXED_PRESTAGE_INTENT_SHA256
        or type(frozen_daemon) is not bytes
        or len(frozen_daemon) != 32
        or actors._actor_start_attempted
        or actors._containers
        or actors._state_volumes
    ):
        _fail("R7 transaction is not the exact locked STAGED_PENDING/PENDING state")
    actors.validate_frozen_daemon_receipt_binding_v1(frozen_daemon)
    docker = _docker_snapshot_v1()
    if docker.get("run_labelled_container_count") != 0 or docker.get("fixed_key_volume_count") != 4 or docker.get("network_operation_invoked") is not False:
        _fail("R7 live Docker state is not zero containers / four retained volumes")
    seed_metadata = (transaction.custody_directory / "split_master_seed.bin").lstat()
    if (
        not stat.S_ISREG(seed_metadata.st_mode)
        or stat.S_IMODE(seed_metadata.st_mode) != 0o600
        or seed_metadata.st_size != 32
    ):
        _fail("R7 raw-seed metadata differs; bytes were not read")
    terminal_rows = _r6_terminal_chain_snapshot_v1()
    incident = {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-incident-diagnostic/1",
        "amendment_commit": amendment_commit,
        "parent_r6_amendment_commit": R6_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 7,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "continuation_action": CONTINUATION_ACTION,
        "r6_terminal_audit_directory": FIXED_R6_AUDIT_DIRECTORY.as_posix(),
        "r6_terminal_chain": terminal_rows,
        "r6_terminal_chain_root_sha256": R6_TERMINAL_CHAIN_ROOT_SHA256,
        "r6_failure_code": R6_FAILURE_CODE,
        "r6_failure_phase": R6_FAILURE_PHASE,
        "r6_failure_detail_sha256": R6_FAILURE_DETAIL_SHA256,
        "r6_formal_failure_evidence_sha256": R6_FORMAL_FAILURE_EVIDENCE_SHA256,
        "r6_attempt6_consumed": True,
        "r6_finalize_sha256_or_null": None,
        "recovery_phase": transaction.recovery_phase,
        "transaction_journal_state": transaction._state,
        "marker_state": marker.state,
        "docker_state": docker,
        "raw_seed_metadata": {
            "kind": "regular",
            "mode_octal": "0600",
            "size_bytes": 32,
            "raw_bytes_read": False,
            "sha256_computed": False,
        },
        "formal_identity_entropy_draw_count": 0,
        "raw_seed_bytes_read_by_r7_orchestrator": False,
        "raw_seed_sha256_computed": False,
        "m3_start_invoked": False,
    }
    staged = transaction._staged_payloads
    if set(staged) != {"evidence", "promotion", "receipt"}:
        _fail("R7 staged public payload set differs")
    staged_rows = tuple(
        {"role": role, "sha256": hashlib.sha256(staged[role]).hexdigest(), "size_bytes": len(staged[role])}
        for role in ("evidence", "promotion", "receipt")
    )
    qualification = {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-poststage-qualification/1",
        "amendment_commit": amendment_commit,
        "parent_r6_amendment_commit": R6_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 7,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r6_terminal_chain_root_sha256": R6_TERMINAL_CHAIN_ROOT_SHA256,
        "fixed_r6_source_admission_sha256": hashlib.sha256(_canonical_json(source_admission)).hexdigest(),
        "fixed_prestage_intent_sha256": FIXED_PRESTAGE_INTENT_SHA256,
        "recovery_phase": "STAGED_PENDING",
        "transaction_journal_state": "STAGED_PROSPECTIVE_REPLAY_PASSED",
        "marker_state": "PENDING",
        "staged_public_payload_rows": staged_rows,
        "staged_public_payload_rows_sha256": hashlib.sha256(_canonical_json(staged_rows)).hexdigest(),
        "runtime_artifact_metadata": tuple(runtime_rows),
        "runtime_artifact_stable_projection": _stable_runtime_projection_v1(runtime_rows),
        "frozen_daemon_receipt_binding_hex": frozen_daemon.hex(),
        "fixed_replay_passed": True,
        "poststage_core_only": True,
        "purpose_actor_start_attempted": False,
        "prestage_core_invoked": False,
        "ordinary_execute_invoked": False,
        "signing_invoked": False,
        "static_rebuild_invoked": False,
        "source_rebuild_invoked": False,
        "formal_identity_entropy_draw_count": 0,
        "raw_seed_bytes_read_by_r7_orchestrator": False,
        "raw_seed_sha256_computed": False,
        "m3_start_invoked": False,
    }
    qualification["qualification_fingerprint_sha256"] = hashlib.sha256(_canonical_json(qualification)).hexdigest()
    return incident, qualification


def _authorization_request_fields(
    *, amendment_commit: object, preflight_raw: bytes,
    incident_raw: bytes, qualification_raw: bytes,
) -> dict[str, object]:
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-authorization-request/1",
        "amendment_commit": amendment_commit,
        "parent_r6_amendment_commit": R6_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 7,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r6_terminal_chain_root_sha256": R6_TERMINAL_CHAIN_ROOT_SHA256,
        "r6_failure_code": R6_FAILURE_CODE,
        "r6_failure_phase": R6_FAILURE_PHASE,
        "r6_failure_detail_sha256": R6_FAILURE_DETAIL_SHA256,
        "r6_formal_failure_evidence_sha256": R6_FORMAL_FAILURE_EVIDENCE_SHA256,
        "preflight_sha256": hashlib.sha256(preflight_raw).hexdigest(),
        "incident_diagnostic_sha256": hashlib.sha256(incident_raw).hexdigest(),
        "poststage_qualification_sha256": hashlib.sha256(qualification_raw).hexdigest(),
        "continuation_action": CONTINUATION_ACTION,
        "requested_action": ACCEPTED_WORKER_MODE,
        "poststage_only": True,
        "ordinary_execute_allowed": False,
        "prestage_core_allowed": False,
        "signing_allowed": False,
        "static_rebuild_allowed": False,
        "source_rebuild_allowed": False,
        "redraw_allowed": False,
        "abort_allowed": False,
        "formal_identity_entropy_draw_count": 0,
        "m3_start_allowed": False,
    }


def _expected_authorization_fields(
    *, amendment_commit: object, preflight_raw: bytes,
    incident_raw: bytes, qualification_raw: bytes, request_raw: bytes,
) -> dict[str, object]:
    return {
        **_authorization_request_fields(
            amendment_commit=amendment_commit,
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            qualification_raw=qualification_raw,
        ),
        "schema": f"{AUDIT_SCHEMA_PREFIX}-authorization/1",
        "authorization_request_sha256": hashlib.sha256(request_raw).hexdigest(),
        "authorization_actor": "PROJECT_OWNER",
        "owner_authorized_fixed_transaction_only": True,
        "ordinary_execute_invoked": False,
    }


def _close_transaction_and_actor_v1(
    transaction: FormalCeremonyTransactionV1 | None,
    actors: A8R7RecoveryDockerActorsV1 | None,
    primary: BaseException | None,
    *, phase: str,
) -> BaseException | None:
    combined = primary
    if actors is not None:
        try:
            actors.close()
        except BaseException as exc:
            combined = exc if combined is None else _executor.combine_formal_failures_v1(combined, exc, phase=phase + "_ACTOR_CLOSE")
    if transaction is not None:
        try:
            transaction.close_lock()
        except BaseException as exc:
            combined = exc if combined is None else _executor.combine_formal_failures_v1(combined, exc, phase=phase + "_TRANSACTION_CLOSE")
    return combined


def prepare_fixed_a8_r7_authorization_v1(
    *, audit_directory: Path, custody_directory: Path,
    public_evidence_path: Path, public_promotion_path: Path,
    rust_formal_replay_binary: Path, rust_bridge_dag_replay_binary: Path,
    rust_bridge_dag_qualification_report: Path,
    repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> None:
    preflight = inspect_r7_source_preflight_v1(repository_root=repository_root, manifest_path=manifest_path)
    source_admission = _fixed_r6_source_admission_v1()
    prestage_intent = _load_fixed_prestage_intent_v1()
    replay = _fixed_replay_v1(
        source_admission=source_admission,
        prestage_intent=prestage_intent,
        amendment_commit=str(preflight["amendment_commit"]),
    )
    runtime_rows = _validate_runtime_artifacts_before_attempt_v1(
        rust_formal_replay_binary=rust_formal_replay_binary,
        rust_bridge_dag_replay_binary=rust_bridge_dag_replay_binary,
        rust_bridge_dag_qualification_report=rust_bridge_dag_qualification_report,
    )
    actors = A8R7RecoveryDockerActorsV1(
        basis_commit=A8_BASIS_COMMIT,
        custody_directory=custody_directory,
        rust_formal_replay_binary=rust_formal_replay_binary,
        rust_bridge_dag_replay_binary=rust_bridge_dag_replay_binary,
        rust_bridge_dag_qualification_report=rust_bridge_dag_qualification_report,
        timestamp=0,
    )
    transaction: FormalCeremonyTransactionV1 | None = None
    primary: BaseException | None = None
    try:
        transaction = FormalCeremonyTransactionV1.rehydrate_post_stage_v1(
            custody_directory=custody_directory,
            public_evidence_path=public_evidence_path,
            public_promotion_path=public_promotion_path,
            replay=replay,
            actors=actors,
        )
        incident, qualification = _qualify_poststage_locked_v1(
            transaction=transaction,
            actors=actors,
            amendment_commit=str(preflight["amendment_commit"]),
            source_admission=source_admission,
            runtime_rows=runtime_rows,
        )
    except BaseException as exc:
        primary = exc
    primary = _close_transaction_and_actor_v1(transaction, actors, primary, phase="R7_PREPARE")
    if primary is not None:
        raise primary
    preflight_raw = _receipt_record_bytes_v1(preflight)
    incident_raw = _receipt_record_bytes_v1(incident)
    qualification_raw = _receipt_record_bytes_v1(qualification)
    request_raw = _receipt_record_bytes_v1(_authorization_request_fields(
        amendment_commit=preflight["amendment_commit"],
        preflight_raw=preflight_raw,
        incident_raw=incident_raw,
        qualification_raw=qualification_raw,
    ))
    audit = _create_or_resume_prepare_audit_directory(audit_directory, repository_root)
    order = ("preflight.json", "incident-diagnostic.json", "poststage-qualification.json", "authorization-request.json")
    observed = {path.name for path in audit.iterdir()}
    allowed = set(order) | {"." + name + ".next" for name in order}
    if not observed.issubset(allowed):
        _fail("R7 preparation audit contains a non-prefix record")
    visible = [name in observed for name in order]
    if any(visible[index] and not all(visible[:index]) for index in range(len(order))):
        _fail("R7 preparation visible records are not an exact prefix")
    for name, raw in zip(order, (preflight_raw, incident_raw, qualification_raw, request_raw), strict=True):
        _install_prepare_record_v1(audit / name, raw)


def write_fixed_a8_r7_owner_authorization_v1(
    *, audit_directory: Path, owner_confirmation: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    if owner_confirmation != OWNER_CONFIRMATION:
        _fail("R7 owner confirmation phrase differs")
    audit = _require_existing_audit_directory(audit_directory, repository_root)
    expected = {"preflight.json", "incident-diagnostic.json", "poststage-qualification.json", "authorization-request.json"}
    actual = {path.name for path in audit.iterdir()}
    if not expected.issubset(actual) or not actual.issubset(expected | {"authorization.json", ".authorization.json.next"}):
        _fail("R7 pre-authorization audit path set differs")
    preflight, preflight_raw = _read_canonical_audit_v1(audit / "preflight.json")
    _incident, incident_raw = _read_canonical_audit_v1(audit / "incident-diagnostic.json")
    _qualification, qualification_raw = _read_canonical_audit_v1(audit / "poststage-qualification.json")
    _request, request_raw = _read_canonical_audit_v1(audit / "authorization-request.json")
    _require_exact_receipt_raw_v1(
        request_raw,
        _authorization_request_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            qualification_raw=qualification_raw,
        ),
        label="authorization request",
    )
    raw = _receipt_record_bytes_v1(_expected_authorization_fields(
        amendment_commit=preflight["amendment_commit"],
        preflight_raw=preflight_raw,
        incident_raw=incident_raw,
        qualification_raw=qualification_raw,
        request_raw=request_raw,
    ))
    _install_prepare_record_v1(audit / "authorization.json", raw)


def inspect_fixed_a8_r7_preflight_v1(
    *, repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, object]:
    source = inspect_r7_source_preflight_v1(repository_root=repository_root, manifest_path=manifest_path)
    rows = _r6_terminal_chain_snapshot_v1()
    journal, journal_raw, _row = _read_canonical_regular_v1(
        FIXED_STAGE_DIRECTORY / "transaction-journal.json", mode=0o600
    )
    docker = _docker_snapshot_v1()
    if journal.get("state") != "STAGED_PROSPECTIVE_REPLAY_PASSED" or docker.get("run_labelled_container_count") != 0 or docker.get("fixed_key_volume_count") != 4:
        _fail("R7 public post-stage preflight state differs")
    return {
        **source,
        "r6_terminal_record_count": len(rows),
        "r6_terminal_chain_root_sha256": R6_TERMINAL_CHAIN_ROOT_SHA256,
        "stage_directory": FIXED_STAGE_DIRECTORY.as_posix(),
        "transaction_journal_state": journal["state"],
        "transaction_journal_sha256": hashlib.sha256(journal_raw).hexdigest(),
        "expected_locked_recovery_phase": "STAGED_PENDING",
        "expected_marker_state": "PENDING",
        "run_labelled_container_count": 0,
        "fixed_key_volume_count": 4,
        "raw_seed_bytes_read_by_r7_orchestrator": False,
        "raw_seed_sha256_computed": False,
        "m3_start_allowed": False,
    }


def _attempt_common_v1(
    *, amendment_commit: object, qualification_raw: bytes,
) -> dict[str, object]:
    return {
        "amendment_commit": amendment_commit,
        "parent_r6_amendment_commit": R6_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 7,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r6_terminal_chain_root_sha256": R6_TERMINAL_CHAIN_ROOT_SHA256,
        "r6_failure_code": R6_FAILURE_CODE,
        "r6_failure_phase": R6_FAILURE_PHASE,
        "r6_failure_detail_sha256": R6_FAILURE_DETAIL_SHA256,
        "r6_formal_failure_evidence_sha256": R6_FORMAL_FAILURE_EVIDENCE_SHA256,
        "poststage_qualification_sha256": hashlib.sha256(qualification_raw).hexdigest(),
    }


def _leaf_failure_rows_v1(
    evidence: Mapping[str, object], *, role: str = "PRIMARY",
) -> tuple[dict[str, object], ...]:
    if evidence.get("kind") == "SINGLE":
        return ({
            "role": role,
            "exception_type": evidence.get("exception_type"),
            "code": evidence.get("code"),
            "detail_sha256": evidence.get("detail_sha256"),
        },)
    if evidence.get("kind") == "ERROR_GRAPH_TERMINAL":
        return ({
            "role": role,
            "exception_type": evidence.get("exception_type"),
            "code": "FORMAL_FAILURE_EVIDENCE_" + str(evidence.get("reason")),
            "detail_sha256": hashlib.sha256(_canonical_json(evidence)).hexdigest(),
        },)
    if evidence.get("kind") != "PRIMARY_AND_CLEANUP":
        _fail("executor failure evidence kind differs")
    primary = evidence.get("primary")
    cleanup = evidence.get("cleanup")
    if type(primary) is not dict or type(cleanup) is not dict:
        _fail("executor composite failure evidence shape differs")
    combination_phase = evidence.get("combination_phase")
    cleanup_role = (
        "FINAL_CLOSE"
        if type(combination_phase) is str
        and combination_phase.startswith("R7_OUTER_FINAL_CLOSE")
        else "CLEANUP"
    )
    return (
        *_leaf_failure_rows_v1(primary, role=role),
        *_leaf_failure_rows_v1(cleanup, role=cleanup_role),
    )


def _failure_record_fields_v1(
    *, amendment_commit: object, qualification_raw: bytes,
    attempt_start_raw: bytes, admission_raw: bytes | None,
    failure_phase: str, exc: BaseException,
) -> dict[str, object]:
    evidence = _executor.formal_failure_evidence_v1(exc)
    leaves = _leaf_failure_rows_v1(evidence)
    primary = leaves[0]
    final_close = next((row for row in leaves[1:] if row["role"] == "FINAL_CLOSE"), None)
    if failure_phase == "R7_OUTER_FINAL_CLOSE" and final_close is None:
        final_close = {**primary, "role": "FINAL_CLOSE"}
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-failure/1",
        **_attempt_common_v1(amendment_commit=amendment_commit, qualification_raw=qualification_raw),
        "attempt_start_sha256": hashlib.sha256(attempt_start_raw).hexdigest(),
        "admission_sha256_or_null": None if admission_raw is None else hashlib.sha256(admission_raw).hexdigest(),
        "failure_code": primary["code"],
        "failure_phase": failure_phase,
        "failure_detail_sha256": primary["detail_sha256"],
        "formal_failure_evidence": evidence,
        "formal_failure_evidence_sha256": hashlib.sha256(_canonical_json(evidence)).hexdigest(),
        "primary_failure": primary,
        "cleanup_failures": [row for row in leaves[1:] if row["role"] == "CLEANUP"],
        "final_close_failure_or_null": final_close,
        "accepted_worker_mode": ACCEPTED_WORKER_MODE,
        "prestage_core_invoked": False,
        "ordinary_execute_invoked": False,
        "signing_invoked": False,
        "static_rebuild_invoked": False,
        "source_rebuild_invoked": False,
        "formal_identity_entropy_draw_count": 0,
        "raw_seed_bytes_read_by_r7_orchestrator": False,
        "raw_seed_sha256_computed": False,
        "m3_start_invoked": False,
    }


def _install_candidate_resolving_visibility_v1(
    *, path: Path, expected: Mapping[str, object], raw: bytes, phase: str,
) -> tuple[str, BaseException | None]:
    """Install one candidate, resolving before-link versus after-link faults."""

    try:
        _install_exact_audit_record_v1(path, expected, raw)
        return "VISIBLE", None
    except BaseException as install_exc:
        try:
            visible = _exact_audit_record_is_visible_v1(path, raw)
        except BaseException as visibility_exc:
            combined = _executor.combine_formal_failures_v1(
                install_exc, visibility_exc, phase=phase + "_VISIBILITY"
            )
            return "UNKNOWN", combined
        if visible:
            try:
                _install_prepare_record_v1(path, raw)
            except BaseException as repair_exc:
                combined = _executor.combine_formal_failures_v1(
                    install_exc, repair_exc, phase=phase + "_VISIBLE_REPAIR"
                )
                return "VISIBLE_REPAIR_FAILED", combined
            return "VISIBLE_REPAIRED", install_exc
        try:
            _discard_non_authoritative_next_v1(path)
        except BaseException as discard_exc:
            combined = _executor.combine_formal_failures_v1(
                install_exc, discard_exc, phase=phase + "_HIDDEN_DISCARD"
            )
            return "UNKNOWN", combined
        return "HIDDEN", install_exc


def _path_exists_via_lstat_v1(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _terminalize_failure_v1(
    *, path: Path, failure: Mapping[str, object], failure_raw: bytes,
    primary: BaseException,
) -> None:
    """Make failure terminal without losing the primary error graph."""

    if _path_exists_via_lstat_v1(path):
        existing, existing_raw = _read_canonical_audit_v1(path)
        if existing != failure or existing_raw != failure_raw:
            mismatch = A8R7RecoveryAmendmentError(
                FAIL_AMENDMENT,
                "existing R7 failure record differs from exact primary evidence",
            )
            combined = _executor.combine_formal_failures_v1(
                primary, mismatch, phase="R7_EXISTING_FAILURE_CONFLICT"
            )
            raise combined from primary
        return
    try:
        _install_exact_audit_record_v1(path, failure, failure_raw)
        return
    except BaseException as audit_exc:
        accumulated = _executor.combine_formal_failures_v1(
            primary, audit_exc, phase="R7_FAILURE_AUDIT_DURABILITY"
        )
        try:
            visible = _exact_audit_record_is_visible_v1(path, failure_raw)
        except BaseException as visibility_exc:
            combined = _executor.combine_formal_failures_v1(
                accumulated,
                visibility_exc,
                phase="R7_FAILURE_VISIBILITY_RESOLUTION",
            )
            raise combined from primary
        if visible:
            try:
                _install_prepare_record_v1(path, failure_raw)
            except BaseException as repair_exc:
                combined = _executor.combine_formal_failures_v1(
                    accumulated,
                    repair_exc,
                    phase="R7_FAILURE_VISIBLE_REPAIR",
                )
                raise combined from primary
            raise accumulated from primary
        try:
            _discard_non_authoritative_next_v1(path)
        except BaseException as discard_exc:
            combined = _executor.combine_formal_failures_v1(
                accumulated,
                discard_exc,
                phase="R7_FAILURE_HIDDEN_DISCARD",
            )
            raise combined from primary
        try:
            _install_prepare_record_v1(path, failure_raw)
        except BaseException as retry_exc:
            combined = _executor.combine_formal_failures_v1(
                accumulated,
                retry_exc,
                phase="R7_FAILURE_CANONICAL_RETRY",
            )
            raise combined from primary
        raise accumulated from primary


def execute_fixed_a8_r7_recovery_v1(
    *, custody_directory: Path, rust_formal_replay_binary: Path,
    rust_bridge_dag_replay_binary: Path,
    rust_bridge_dag_qualification_report: Path,
    public_evidence_path: Path, public_promotion_path: Path,
    audit_directory: Path, repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> tuple[dict[str, object], dict[str, object]]:
    preflight_now = inspect_r7_source_preflight_v1(repository_root=repository_root, manifest_path=manifest_path)
    source_admission = _fixed_r6_source_admission_v1()
    prestage_intent = _load_fixed_prestage_intent_v1()
    replay = _fixed_replay_v1(
        source_admission=source_admission,
        prestage_intent=prestage_intent,
        amendment_commit=str(preflight_now["amendment_commit"]),
    )
    runtime_rows = _validate_runtime_artifacts_before_attempt_v1(
        rust_formal_replay_binary=rust_formal_replay_binary,
        rust_bridge_dag_replay_binary=rust_bridge_dag_replay_binary,
        rust_bridge_dag_qualification_report=rust_bridge_dag_qualification_report,
    )
    audit = _require_existing_audit_directory(audit_directory, repository_root)
    directory_identity, pre_raws, pre_inodes = _read_pre_attempt_audit_snapshot_v1(audit, allow_attempt_next=True)
    preflight_raw = pre_raws["preflight.json"]
    incident_raw = pre_raws["incident-diagnostic.json"]
    qualification_raw = pre_raws["poststage-qualification.json"]
    request_raw = pre_raws["authorization-request.json"]
    authorization_raw = pre_raws["authorization.json"]
    preflight = json.loads(preflight_raw)
    if type(preflight) is not dict or preflight_raw != _receipt_record_bytes_v1(preflight_now):
        _fail("stored R7 preflight differs from current clean R7 source")
    _require_exact_receipt_raw_v1(
        request_raw,
        _authorization_request_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            qualification_raw=qualification_raw,
        ),
        label="stored authorization request",
    )
    _require_exact_receipt_raw_v1(
        authorization_raw,
        _expected_authorization_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            qualification_raw=qualification_raw,
            request_raw=request_raw,
        ),
        label="stored owner authorization",
    )

    actors = A8R7RecoveryDockerActorsV1(
        basis_commit=A8_BASIS_COMMIT,
        custody_directory=custody_directory,
        rust_formal_replay_binary=rust_formal_replay_binary,
        rust_bridge_dag_replay_binary=rust_bridge_dag_replay_binary,
        rust_bridge_dag_qualification_report=rust_bridge_dag_qualification_report,
        timestamp=0,
    )
    transaction: FormalCeremonyTransactionV1 | None = None
    attempt_start_raw: bytes | None = None
    admission_raw: bytes | None = None
    payload: dict[str, object] | None = None
    promotion: dict[str, object] | None = None
    primary: BaseException | None = None
    suppress_failure_terminal = False
    failure_phase = "POSTSTAGE_REHYDRATION"
    try:
        transaction = FormalCeremonyTransactionV1.rehydrate_post_stage_v1(
            custody_directory=custody_directory,
            public_evidence_path=public_evidence_path,
            public_promotion_path=public_promotion_path,
            replay=replay,
            actors=actors,
        )
        incident_now, qualification_now = _qualify_poststage_locked_v1(
            transaction=transaction,
            actors=actors,
            amendment_commit=str(preflight["amendment_commit"]),
            source_admission=source_admission,
            runtime_rows=runtime_rows,
        )
        if incident_raw != _receipt_record_bytes_v1(incident_now) or qualification_raw != _receipt_record_bytes_v1(qualification_now):
            _fail("stored R7 post-stage qualification differs under transaction lock")
        _recheck_pre_attempt_audit_under_lock_v1(
            audit=audit,
            expected_directory_identity=directory_identity,
            expected_raws=pre_raws,
            expected_inodes=pre_inodes,
        )
        actors.timestamp = transaction._recovery_marker_snapshot.created_at_unix_seconds  # type: ignore[union-attr]
        common = _attempt_common_v1(amendment_commit=preflight["amendment_commit"], qualification_raw=qualification_raw)
        failure_phase = "ATTEMPT_START_DURABILITY"
        attempt, attempt_candidate_raw = _r6._r4._r31._build_exact_audit_record_v1({
            "schema": f"{AUDIT_SCHEMA_PREFIX}-attempt-start/1",
            **common,
            "continuation_action": CONTINUATION_ACTION,
            "authorization_sha256": hashlib.sha256(authorization_raw).hexdigest(),
            "fixed_r6_source_admission_sha256": FIXED_R6_SOURCE_ADMISSION_SHA256,
            "fixed_prestage_intent_sha256": FIXED_PRESTAGE_INTENT_SHA256,
            "runtime_artifact_metadata": runtime_rows,
            "runtime_artifact_stable_projection": _stable_runtime_projection_v1(runtime_rows),
            "accepted_worker_mode": ACCEPTED_WORKER_MODE,
            "poststage_core_only": True,
            "formal_identity_entropy_draw_count": 0,
            "raw_seed_bytes_read_by_r7_orchestrator": False,
            "raw_seed_sha256_computed": False,
            "m3_start_invoked": False,
        })
        attempt_status, attempt_install_error = _install_candidate_resolving_visibility_v1(
            path=audit / "attempt-start.json",
            expected=attempt,
            raw=attempt_candidate_raw,
            phase="R7_ATTEMPT_INSTALL",
        )
        if attempt_status.startswith("VISIBLE"):
            attempt_start_raw = attempt_candidate_raw
        if attempt_install_error is not None:
            raise attempt_install_error
        failure_phase = "POSTSTAGE_ADMISSION_DURABILITY"
        admission, admission_candidate_raw = _r6._r4._r31._build_exact_audit_record_v1({
            "schema": f"{AUDIT_SCHEMA_PREFIX}-admission/1",
            **common,
            "attempt_start_sha256": hashlib.sha256(attempt_start_raw).hexdigest(),
            "authorization_sha256": hashlib.sha256(authorization_raw).hexdigest(),
            "fixed_r6_source_admission": source_admission,
            "fixed_r6_source_admission_sha256": FIXED_R6_SOURCE_ADMISSION_SHA256,
            "fixed_prestage_intent_sha256": FIXED_PRESTAGE_INTENT_SHA256,
            "accepted_worker_mode": ACCEPTED_WORKER_MODE,
            "poststage_core_only": True,
            "formal_identity_entropy_draw_count": 0,
            "raw_seed_bytes_read_by_r7_orchestrator": False,
            "raw_seed_sha256_computed": False,
            "m3_start_invoked": False,
        })
        admission_status, admission_install_error = _install_candidate_resolving_visibility_v1(
            path=audit / "admission.json",
            expected=admission,
            raw=admission_candidate_raw,
            phase="R7_ADMISSION_INSTALL",
        )
        if admission_status.startswith("VISIBLE"):
            admission_raw = admission_candidate_raw
        if admission_status == "UNKNOWN" and admission_install_error is not None:
            raise _R7TerminalAuthorityResolutionError(admission_install_error)
        if admission_install_error is not None:
            raise admission_install_error
        failure_phase = "POSTSTAGE_ONLY_FORMAL_CORE"
        payload, promotion = _continue_post_stage_transaction_recovery_core_v1(
            transaction=transaction,
            actors=actors,
            replay=replay,
        )
    except _R7TerminalAuthorityResolutionError as terminal_error:
        primary = terminal_error.error
        suppress_failure_terminal = True
    except BaseException as exc:
        primary = exc
    primary_before_close = primary
    primary = _close_transaction_and_actor_v1(transaction, actors, primary, phase="R7_OUTER_FINAL_CLOSE")
    if primary_before_close is None and primary is not None:
        failure_phase = "R7_OUTER_FINAL_CLOSE"
    if suppress_failure_terminal:
        assert primary is not None
        raise primary

    if primary is None:
        try:
            if payload is None or promotion is None or attempt_start_raw is None or admission_raw is None:
                _fail("R7 successful post-stage core result is incomplete")
            final = _r6._validate_final_publication_v1(
                payload=payload,
                promotion=promotion,
                custody_directory=custody_directory,
                stage_directory=FIXED_STAGE_DIRECTORY,
                public_evidence_path=public_evidence_path,
                public_promotion_path=public_promotion_path,
                replay=replay,
            )
            finalize, finalize_raw = _r6._r4._r31._build_exact_audit_record_v1({
                "schema": f"{AUDIT_SCHEMA_PREFIX}-finalize/1",
                **_attempt_common_v1(amendment_commit=preflight["amendment_commit"], qualification_raw=qualification_raw),
                "attempt_start_sha256": hashlib.sha256(attempt_start_raw).hexdigest(),
                "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
                **final,
                "formal_gates_after": 24,
                "child_state": "NOT_RUN",
                "accepted_worker_mode": ACCEPTED_WORKER_MODE,
                "prestage_core_invoked": False,
                "ordinary_execute_invoked": False,
                "signing_invoked": False,
                "static_rebuild_invoked": False,
                "source_rebuild_invoked": False,
                "formal_identity_entropy_draw_count": 0,
                "raw_seed_bytes_read_by_r7_orchestrator": False,
                "raw_seed_sha256_computed": False,
                "m3_start_invoked": False,
            })
            finalize_status, finalize_install_error = _install_candidate_resolving_visibility_v1(
                path=audit / "finalize.json",
                expected=finalize,
                raw=finalize_raw,
                phase="R7_FINALIZE_INSTALL",
            )
            if finalize_status == "VISIBLE_REPAIRED":
                return payload, promotion
            if finalize_status in {"VISIBLE_REPAIR_FAILED", "UNKNOWN"} and finalize_install_error is not None:
                raise _R7TerminalAuthorityResolutionError(finalize_install_error)
            if finalize_install_error is not None:
                raise finalize_install_error
            return payload, promotion
        except _R7TerminalAuthorityResolutionError as terminal_error:
            raise terminal_error.error
        except BaseException as exc:
            primary = exc
            failure_phase = "FINALIZE_DURABILITY"

    if attempt_start_raw is not None:
        failure, failure_raw = _r6._r4._r31._build_exact_audit_record_v1(
            _failure_record_fields_v1(
                amendment_commit=preflight["amendment_commit"],
                qualification_raw=qualification_raw,
                attempt_start_raw=attempt_start_raw,
                admission_raw=admission_raw,
                failure_phase=failure_phase,
                exc=primary,
            )
        )
        _terminalize_failure_v1(
            path=audit / "failure.json",
            failure=failure,
            failure_raw=failure_raw,
            primary=primary,
        )
    assert primary is not None
    raise primary


__all__ = [
    "A8R7RecoveryAmendmentError",
    "DEFAULT_MANIFEST_PATH",
    "execute_fixed_a8_r7_recovery_v1",
    "inspect_fixed_a8_r7_preflight_v1",
    "inspect_r7_source_preflight_v1",
    "prepare_fixed_a8_r7_authorization_v1",
    "write_fixed_a8_r7_owner_authorization_v1",
]
