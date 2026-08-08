"""Attempt-3 recovery with an isolated frozen-A8 report validator.

R3 is a fresh, one-shot continuation of the exact R2 terminal failure.  The
formal identity remains A8.  Diagnostic reports are validated by a fixed
``python -I -S -B -X pycache_prefix=...`` child rooted at the unchanged main
A8 worktree, while the parent binds that deterministic receipt to the fixed
R1/R2 provenance, runtime artifacts, and the complete-only recovery core.  No
entry point in this module starts M3, redraws a seed, or opens/hashes the raw
seed.

R3.1 is a pre-attempt verifier erratum layered on the pushed R3 amendment.
The original R3 authorization prefix is preserved and bound as an immutable,
unconsumed superseded prefix.  Attempt ordinal 3 remains unconsumed because
the old prefix contains no ``attempt-start.json``.  R3.1 admits only exact
canonical incident bytes; it never coerces diagnostic Python representations.
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

from . import phase3_m25_a8_recovery_amendment_r2_v1 as _r2
from .phase3_m25_a8_recovery_amendment_v1 import A8R1RecoveryDockerActorsV1
from .phase3_m25_container_ceremony_v1 import (
    TECHNICAL_ACTOR_DISCLOSURE_V1,
    read_marker_snapshot_v1,
)
from .phase3_m25_formal_container_executor_v1 import (
    FAIL_RECOVERY_SOURCE_ADMISSION,
    FormalContainerExecutorError,
    PendingCeremonyRecoveryV1,
    REQUIRED_COMMIT_A_INPUTS,
    _canonical_json as _executor_canonical_json,
    _continue_pre_stage_pending_recovery_core_v1,
    _replay_public_gate_evidence_with_fixed_a8_r3_basis_v1,
    _restore,
    _transport,
    acquire_pending_ceremony_recovery_v1,
    load_gate_evidence_inputs_v1,
)
from .phase3_m25_external_v1 import assert_public_payload_contains_no_secret_fields
from .phase3_m25_formal_static_basis_v1 import (
    FORMAL_GIT_ENVIRONMENT_V1,
    FORMAL_GIT_EXECUTABLE,
)
from .phase3_m25_wire_v1 import M3_RUN_OUTPUT_ROOTS


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
DEFAULT_MANIFEST_PATH: Final = (
    PROJECT_ROOT / "config/phase3_m25_a8_recovery_amendment_r31_v1.json"
)
A8_VALIDATOR_TOOL: Final = (
    PROJECT_ROOT / "tools/phase3_m25_a8_recovery_report_validator_r3_v1.py"
)
R31_HISTORICAL_A8_VALIDATOR_TOOL: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/"
    "a8-recovery-amendment-worktree/Hegel Machine/tools/"
    "phase3_m25_a8_recovery_report_validator_r3_v1.py"
)
FIXED_PYTHON_EXECUTABLE: Final = Path("/usr/bin/python3.10")
FIXED_PYTHON_EXECUTABLE_SHA256: Final = (
    "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
)
FIXED_PYCACHE_PREFIX: Final = "/nonexistent/hegel-r3-pycache"
EXPECTED_A8_IMPORT_CLOSURE_SHA256_ROOT: Final = (
    "d071923c4f926104d78ef082f36aec66ff33221a530ad68a9bdf7cfe3f644d77"
)
EXPECTED_A8_VALIDATOR_DEPENDENCY_CLOSURE_SHA256_ROOT: Final = (
    "f39b2f922af5723ee50374b4f04be5c6525a58a87e19de9376d2525a108d1dc7"
)
EXPECTED_A8_VALIDATION_RECEIPT_RAW_SHA256: Final = (
    "ef18694aa41a78389cef2265eb121174f2e68548928f89f7fcad3f55fb261ee4"
)
FIXED_FORMAL_REPOSITORY_ROOT: Final = Path(
    "/home/erzhu419/mine_code/Asumption Agent"
)
A8_BASIS_COMMIT: Final = _r2.A8_BASIS_COMMIT
R1_AMENDMENT_COMMIT: Final = _r2.R1_AMENDMENT_COMMIT
R2_AMENDMENT_COMMIT: Final = "ec7c04cf62190558c72448639d7e3cd13a5b6903"
R3_AMENDMENT_COMMIT: Final = "52a4a61934a73c70dc09b919cae377db166eaedf"
FIXED_RUN_ID_HEX: Final = _r2.FIXED_RUN_ID_HEX
FIXED_LEDGER_ID_HEX: Final = _r2.FIXED_LEDGER_ID_HEX
FIXED_RUN_ID: Final = _r2.FIXED_RUN_ID
FIXED_LEDGER_ID: Final = _r2.FIXED_LEDGER_ID
R1_AUDIT_DIRECTORY: Final = _r2.R1_AUDIT_DIRECTORY
R2_AUDIT_DIRECTORY: Final = _r2.FIXED_R2_AUDIT_DIRECTORY
R3_PREATTEMPT_AUDIT_DIRECTORY: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/"
    "phase3-m25-0af65964235390ce2bebefea7379eaa9c50eda24/"
    "recovery-audit-r3-e4af9f57c38fb298462ec628c4ed8a03-attempt-3"
)
FIXED_R3_AUDIT_DIRECTORY: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/"
    "phase3-m25-0af65964235390ce2bebefea7379eaa9c50eda24/"
    "recovery-audit-r31-e4af9f57c38fb298462ec628c4ed8a03-"
    "attempt-3-revision-1"
)
MANIFEST_SCHEMA: Final = "hegel-phase3-m25-a8-recovery-amendment-r31/1"
AUDIT_SCHEMA_PREFIX: Final = "hegel-phase3-m25-a8-r31-recovery-audit"
R3_PREATTEMPT_AUDIT_SCHEMA_PREFIX: Final = (
    "hegel-phase3-m25-a8-r3-recovery-audit"
)
FAIL_AMENDMENT: Final = "FAIL_M25_A8_R31_RECOVERY_AMENDMENT"
OWNER_CONFIRMATION: Final = (
    "AUTHORIZE_A8_R31_ATTEMPT_3_REVISION_1_CANONICAL_BYTES_"
    "COMPLETE_ONLY_REAL_PENDING_RESUME"
)
CONTINUATION_ACTION: Final = "PRE_ATTEMPT_VERIFIER_ERRATUM_CONTINUATION"
SOURCE_ADMISSION_CONTINUATION_ACTION: Final = (
    "CODE_AMENDMENT_RECOVERY_CONTINUATION"
)
AUTHORIZATION_REVISION_ID: Final = "R31_CANONICAL_INCIDENT_BYTES_V1"
PRE_ATTEMPT_DEFECT_CODE: Final = (
    "PRE_ATTEMPT_SUPERSEDED_IMPLEMENTATION_DEFECT_LIST_TUPLE_EQUALITY"
)
R3_PREATTEMPT_AUDIT_RAW_SHA256: Final = {
    "preflight.json": "3e6820c0f76e8a8b77de3f3888bb5f072e59a2e4fb95b79533ce6bf80f685b5a",
    "incident-diagnostic.json": "d0b27d5c7f1f00a74873bac2394f05fb6666a29e07fdbf9886999f0dddbebc21",
    "a8-validation-receipt.json": "ef18694aa41a78389cef2265eb121174f2e68548928f89f7fcad3f55fb261ee4",
    "authorization-request.json": "28fb786ab5d0017c295b4ac5efee1ff26deabf9dbb024c732ebc565a4048c28d",
    "authorization.json": "c6d5c04ee1cc499b8a697e2a4359144f713a3711cad578fc953e3b203f1b7721",
}
R3_PREATTEMPT_AUDIT_RECEIPT_SHA256: Final = {
    "preflight.json": "a058b0f18eb53cfd58e720041c4d9a7b9985532360c410959be1140855d38f98",
    "incident-diagnostic.json": "73060741d5f78efdc27c8273285a4c967954c72c234c2956a074270981cf5ee9",
    "a8-validation-receipt.json": "83b1ad690914d9dfd5cd402d5c734a1250a3b450c9e0b3ecf4655cfb97c6ba47",
    "authorization-request.json": "f133d0dd69e1a61c9c3ca89b3465433406d367c1a4b16c59e1065f9a104e6209",
    "authorization.json": "1cdfd80310184da56052df4445db8795148f8c9db50eef7595608fe79b9853b0",
}
R3_PREATTEMPT_PREFIX_ROOT_SHA256: Final = (
    "9771b20bf63f1095456618d3ccd4c9db0c54c693307314b8aea72afa18249999"
)
R3_PREATTEMPT_MANIFEST_SHA256: Final = (
    "2f6a9f6b100e6881f24072ad1a07b675d2831c27d756574deea2fc0a6f217178"
)
R3_PREATTEMPT_REPRESENTATION_MISMATCH_FIELDS: Final = (
    "additional_stage_continuity_metadata",
    "docker_state.fixed_key_volume_label_rows",
    "docker_state.fixed_key_volume_names",
    "docker_state.run_labelled_container_names",
    "fixed_stage_inventory",
    "public_reservation_metadata",
    "r1_failure_chain",
    "r2_terminal_chain",
    "seed_prefix_metadata",
)
R2_AUDIT_RAW_SHA256: Final = {
    "preflight.json": "549e2f2654e8d4b334ae63d33314083c2b9a44c2c16914b31a95549af84a6afe",
    "incident-diagnostic.json": "a2eaeb9534c519bc94f2c687d5d6529d86795bc52d4812736040fdbe3ad0a0c0",
    "authorization-request.json": "7f0dd107b03a92082694f8bf6d3da024fff1679699fc394d89a99d5a651fe755",
    "authorization.json": "98d3b26863f04273472c1322f21f529b3da83a5eee898bdf4a6944148f788cdf",
    "attempt-start.json": "b4b817878d84c6506739f30adc4f38689791c37e3ee786e5c855b86df4a4f0e0",
    "failure.json": "bd64cfa99885dd60750615fcb23abd960aed78ef676a0d2d4d8ed942e5395d56",
}
R2_AUDIT_RECEIPT_SHA256: Final = {
    "preflight.json": "fb9aa896527536e1e8af7dafd8925aff7cde5832ec6b61628fe0fc956c25cf86",
    "incident-diagnostic.json": "14b44b91b168fe123136109c258b8eb3eee48f4b9fe2b537e6d254eda163e5a0",
    "authorization-request.json": "7eea1d4b4dd304daad0f1de452fbec6a9f0e39acfe15422dc6982ad8f6f6166c",
    "authorization.json": "af521ae0d5f02bf63d8e07b0f47e8efec7db780b7ea6ec97fb1d69953ed9fd13",
    "attempt-start.json": "37b789682e10f9b31f11c270ad7041e04a541063f960fb7dcbd4c1726eb7a6ba",
    "failure.json": "87b400cf0070efdb3e2f9d7b37dc09675258c5b0341ce629b7c7b6c5431f3f58",
}
R2_TERMINAL_CHAIN_ROOT_SHA256: Final = (
    "76379650dbb142f791d26ca50b24cf308d7deb04bed6eae2e4d84aae4171ac0b"
)
R2_FAILURE_DETAIL_SHA256: Final = (
    "545088b7f95361390cb4af9b2a64886dfb4929ee5f1506c1f6b02ba55569a7ac"
)
EXPECTED_UNCHANGED_A8_INPUT_COUNT: Final = 95
EXPECTED_UNCHANGED_A8_INPUT_ROOT: Final = (
    "51b71d2da5d593d9b208f0119b619761a50b1b8823635df0698965723abf5d40"
)
FIXED_RUNTIME_ARTIFACTS: Final = _r2.FIXED_RUNTIME_ARTIFACTS
FIXED_CONTINUITY_SHA256: Final = _r2.FIXED_CONTINUITY_SHA256
R3_RUNTIME_EXCEPTION_PATHS: Final = frozenset(
    {
        *_r2.R2_RUNTIME_EXCEPTION_PATHS,
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_amendment_r3_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_cli_r3_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_amendment_r4_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_cli_r4_v1.py",
    }
)
_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")
EXPECTED_A8_VALIDATION_RECEIPT_KEYS: Final = frozenset(
    {
        "schema",
        "formal_repository_root",
        "formal_repository_path_sha256",
        "formal_repository_commit",
        "python_executable",
        "python_executable_sha256",
        "python_isolated",
        "python_no_site",
        "python_bytecode_disabled",
        "python_pycache_prefix",
        "a8_import_closure_sha256_root",
        "a8_validator_dependency_closure_sha256_root",
        "run_id_hex",
        "ledger_id_hex",
        "actor_report_sha256",
        "errata_report_sha256",
        "live_bundle_sha256",
        "live_bundle_content_id_hex",
        "qualification_key_id_rows",
        "commit_a_input_sha256",
        "commit_a_input_sha256_root",
        "commit_a_input_count",
        "actor_technical_eligible",
        "errata_status",
        "transaction_bundle_replay_passed",
        "formal_identity_entropy_draw_count",
        "contains_raw_seed",
        "contains_private_key",
        "raw_seed_bytes_read",
        "raw_seed_sha256_computed",
        "m3_start_invoked",
        "receipt_sha256",
    }
)


class A8R3RecoveryAmendmentError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(detail: str) -> NoReturn:
    raise A8R3RecoveryAmendmentError(FAIL_AMENDMENT, detail)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


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


def _load_manifest(path: Path) -> tuple[dict[str, object], bytes]:
    if path.is_symlink():
        _fail("R3 amendment manifest may not be a symlink")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"R3 amendment manifest is invalid JSON: {exc}")
    required = {
        "schema", "source_commit_selector", "sole_parent_commit",
        "formal_repository_commit", "fixed_run_id_hex", "fixed_ledger_id_hex",
        "recovery_attempt_ordinal", "exact_changed_paths", "source_bindings",
        "complete_seed_resume_only", "formal_identity_entropy_draw_count",
        "ephemeral_container_nonce_allowed", "ordinary_execute_allowed",
        "ordinary_recovery_cross_basis_allowed", "fixed_r1_audit_directory",
        "fixed_r2_audit_directory", "fixed_r3_preattempt_audit_directory",
        "fixed_r31_audit_directory", "parent_amendment_commit",
        "r1_audit_raw_sha256", "r1_failure_receipt_sha256",
        "r2_audit_raw_sha256", "r2_audit_receipt_sha256",
        "r2_terminal_chain_root_sha256", "expected_live_bundle_sha256",
        "expected_a8_validation_receipt_sha256",
        "fixed_continuity_sha256", "continuation_action", "owner_confirmation",
        "fixed_runtime_artifacts", "a8_validator_execution",
        "r3_preattempt_audit_raw_sha256",
        "r3_preattempt_audit_receipt_sha256",
        "r3_preattempt_prefix_root_sha256", "authorization_revision_id",
        "pre_attempt_defect_code",
    }
    expected_validator = {
        "python_executable": FIXED_PYTHON_EXECUTABLE.as_posix(),
        "python_executable_sha256": FIXED_PYTHON_EXECUTABLE_SHA256,
        "python_executable_mode_octal": "0755",
        "isolated_flags": ["-I", "-S", "-B"],
        "python_pycache_prefix": FIXED_PYCACHE_PREFIX,
        "a8_import_closure_sha256_root": (
            EXPECTED_A8_IMPORT_CLOSURE_SHA256_ROOT
        ),
        "validator_dependency_closure_sha256_root": (
            EXPECTED_A8_VALIDATOR_DEPENDENCY_CLOSURE_SHA256_ROOT
        ),
        # This manifest is immutable evidence for the already-terminal R3.1
        # commit.  Its historical worktree path must not be rewritten merely
        # because a later sole-child recovery uses a new clean worktree.  R4
        # binds its current validator path in its own manifest.
        "tool_path": R31_HISTORICAL_A8_VALIDATOR_TOOL.as_posix(),
        "formal_repository_root": FIXED_FORMAL_REPOSITORY_ROOT.as_posix(),
        "formal_repository_commit": A8_BASIS_COMMIT,
    }
    if type(value) is not dict or _canonical_json(value) != raw or set(value) != required:
        _fail("R3 amendment manifest is not canonical exact JSON")
    if (
        value.get("schema") != MANIFEST_SCHEMA
        or value.get("source_commit_selector") != "HEAD"
        or value.get("sole_parent_commit") != R3_AMENDMENT_COMMIT
        or value.get("parent_amendment_commit") != R3_AMENDMENT_COMMIT
        or value.get("formal_repository_commit") != A8_BASIS_COMMIT
        or value.get("fixed_run_id_hex") != FIXED_RUN_ID_HEX
        or value.get("fixed_ledger_id_hex") != FIXED_LEDGER_ID_HEX
        or value.get("recovery_attempt_ordinal") != 3
        or value.get("complete_seed_resume_only") is not True
        or value.get("formal_identity_entropy_draw_count") != 0
        or value.get("ephemeral_container_nonce_allowed") is not True
        or value.get("ordinary_execute_allowed") is not False
        or value.get("ordinary_recovery_cross_basis_allowed") is not False
        or value.get("fixed_r1_audit_directory") != R1_AUDIT_DIRECTORY.as_posix()
        or value.get("fixed_r2_audit_directory") != R2_AUDIT_DIRECTORY.as_posix()
        or value.get("fixed_r3_preattempt_audit_directory")
        != R3_PREATTEMPT_AUDIT_DIRECTORY.as_posix()
        or value.get("fixed_r31_audit_directory")
        != FIXED_R3_AUDIT_DIRECTORY.as_posix()
        or value.get("r1_audit_raw_sha256") != _r2.R1_AUDIT_RAW_SHA256
        or value.get("r1_failure_receipt_sha256") != _r2.R1_FAILURE_RECEIPT_SHA256
        or value.get("r2_audit_raw_sha256") != R2_AUDIT_RAW_SHA256
        or value.get("r2_audit_receipt_sha256") != R2_AUDIT_RECEIPT_SHA256
        or value.get("r2_terminal_chain_root_sha256")
        != R2_TERMINAL_CHAIN_ROOT_SHA256
        or value.get("r3_preattempt_audit_raw_sha256")
        != R3_PREATTEMPT_AUDIT_RAW_SHA256
        or value.get("r3_preattempt_audit_receipt_sha256")
        != R3_PREATTEMPT_AUDIT_RECEIPT_SHA256
        or value.get("r3_preattempt_prefix_root_sha256")
        != R3_PREATTEMPT_PREFIX_ROOT_SHA256
        or value.get("expected_live_bundle_sha256")
        != _r2.EXPECTED_LIVE_BUNDLE_SHA256
        or value.get("expected_a8_validation_receipt_sha256")
        != EXPECTED_A8_VALIDATION_RECEIPT_RAW_SHA256
        or value.get("fixed_continuity_sha256") != FIXED_CONTINUITY_SHA256
        or value.get("continuation_action") != CONTINUATION_ACTION
        or value.get("authorization_revision_id") != AUTHORIZATION_REVISION_ID
        or value.get("pre_attempt_defect_code") != PRE_ATTEMPT_DEFECT_CODE
        or value.get("owner_confirmation") != OWNER_CONFIRMATION
        or tuple(value.get("fixed_runtime_artifacts", ())) != FIXED_RUNTIME_ARTIFACTS
        or value.get("a8_validator_execution") != expected_validator
    ):
        _fail("R3.1 amendment manifest fixed policy differs")
    return value, raw


def _read_committed_worktree_blob_v1(
    path: Path, *, expected_mode: int
) -> bytes:
    """Read one changed worktree blob through a bound regular-file inode."""

    if path.is_symlink():
        _fail(f"R3 changed worktree path is a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
        before = path.lstat()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        _fail(f"R3 changed worktree path cannot be opened: {path}: {exc}")
    try:
        metadata = os.fstat(descriptor)
        if (
            resolved != path
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != expected_mode
            or metadata.st_uid != os.getuid()
            or metadata.st_gid != os.getgid()
            or (metadata.st_dev, metadata.st_ino) != (before.st_dev, before.st_ino)
        ):
            _fail(f"R3 changed worktree inode or mode differs: {path}")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _fail(f"R3 changed worktree read was short: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"R3 changed worktree blob grew while read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _verify_changed_worktree_blob_v1(
    *, repository_root: Path, head: str, relative: str, expected_sha256: str
) -> None:
    """Bind a changed path's current bytes and metadata to its HEAD blob."""

    repository = repository_root.resolve(strict=True)
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        _fail(f"R3 changed worktree path is not repository-relative: {relative}")
    tree_raw = _git(repository, ("ls-tree", "-z", head, "--", relative))
    rows = tuple(row for row in tree_raw.split(b"\0") if row)
    if len(rows) != 1:
        _fail(f"R3 changed path is not one committed blob: {relative}")
    try:
        metadata_raw, tree_path_raw = rows[0].split(b"\t", 1)
        mode_raw, kind_raw, _object_id = metadata_raw.decode(
            "ascii", "strict"
        ).split(" ")
        tree_path = tree_path_raw.decode("utf-8", "strict")
    except (UnicodeDecodeError, ValueError):
        _fail(f"R3 changed path tree row is malformed: {relative}")
    mode_by_git = {"100644": 0o644, "100755": 0o755}
    if kind_raw != "blob" or tree_path != relative or mode_raw not in mode_by_git:
        _fail(f"R3 changed path tree identity differs: {relative}")
    committed = _git(repository, ("show", f"{head}:{relative}"))
    current = _read_committed_worktree_blob_v1(
        repository / relative_path, expected_mode=mode_by_git[mode_raw]
    )
    if (
        current != committed
        or hashlib.sha256(committed).hexdigest() != expected_sha256
    ):
        _fail(f"R3 changed worktree blob differs from HEAD: {relative}")


def _verify_changed_index_flags_v1(
    repository_root: Path, changed_paths: set[str]
) -> None:
    """Reject assume-unchanged/skip-worktree or any non-normal index tag."""

    rows = tuple(
        row
        for row in _git(
            repository_root,
            ("ls-files", "-v", "-z", "--", *sorted(changed_paths)),
        ).split(b"\0")
        if row
    )
    observed: set[str] = set()
    for row in rows:
        if len(row) < 3 or row[1:2] != b" ":
            _fail("R3 changed-path index row is malformed")
        try:
            relative = row[2:].decode("utf-8", "strict")
        except UnicodeDecodeError:
            _fail("R3 changed-path index row is not UTF-8")
        if row[0:1] != b"H" or relative in observed:
            _fail(f"R3 changed path has a non-normal index flag: {relative}")
        observed.add(relative)
    if observed != changed_paths:
        _fail("R3 changed-path index set differs")


def _source_binding_paths_are_exact_v1(
    bindings: object, expected_paths: Sequence[str]
) -> bool:
    """Require one ordinary JSON row per changed source, in Git diff order."""

    if type(bindings) is not list or len(bindings) != len(expected_paths):
        return False
    observed: list[str] = []
    for row in bindings:
        if type(row) is not dict or type(row.get("path")) is not str:
            return False
        observed.append(row["path"])
    return tuple(observed) == tuple(expected_paths)


def inspect_r3_source_preflight_v1(
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
    if _HEX_40.fullmatch(head) is None or parents != [R3_AMENDMENT_COMMIT]:
        _fail("R3.1 must be one committed sole child of the frozen R3 amendment")
    if _git(repository_root, ("status", "--porcelain=v1", "--untracked-files=all")):
        _fail("R3 repository tree/index is not clean")
    changed_lines = _git(
        repository_root,
        (
            "diff-tree", "--no-commit-id", "--name-status", "-r",
            "--no-renames", R3_AMENDMENT_COMMIT, head,
        ),
    ).decode("utf-8", "strict").splitlines()
    actual = tuple(
        {"status": line.split("\t", 1)[0], "path": line.split("\t", 1)[1]}
        for line in changed_lines
        if line
    )
    if type(manifest.get("exact_changed_paths")) is not list or tuple(
        manifest["exact_changed_paths"]
    ) != actual:
        _fail("R3 changed-path allowlist differs")
    manifest_relative = manifest_path.resolve(strict=True).relative_to(
        repository_root.resolve(strict=True)
    ).as_posix()
    changed_paths = {str(row["path"]) for row in actual}
    _verify_changed_index_flags_v1(repository_root, changed_paths)
    _verify_changed_worktree_blob_v1(
        repository_root=repository_root,
        head=head,
        relative=manifest_relative,
        expected_sha256=hashlib.sha256(manifest_raw).hexdigest(),
    )
    bindings = manifest.get("source_bindings")
    expected_binding_paths = tuple(
        str(row["path"])
        for row in actual
        if str(row["path"]) != manifest_relative
    )
    if not _source_binding_paths_are_exact_v1(
        bindings, expected_binding_paths
    ):
        _fail("R3 source bindings do not equal changed paths minus manifest")
    verified: list[dict[str, object]] = []
    for row in bindings:
        if type(row) is not dict or set(row) != {
            "path", "parent_sha256_or_null", "r31_sha256"
        }:
            _fail("R3 source-binding row differs")
        path = row.get("path")
        old_hash = row.get("parent_sha256_or_null")
        new_hash = row.get("r31_sha256")
        if (
            type(path) is not str
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
            or (old_hash is not None and (
                type(old_hash) is not str or _HEX_64.fullmatch(old_hash) is None
            ))
            or type(new_hash) is not str
            or _HEX_64.fullmatch(new_hash) is None
        ):
            _fail("R3 source-binding value differs")
        _verify_changed_worktree_blob_v1(
            repository_root=repository_root,
            head=head,
            relative=path,
            expected_sha256=new_hash,
        )
        if old_hash is None:
            probe = subprocess.run(
                [str(FORMAL_GIT_EXECUTABLE), "cat-file", "-e", f"{R3_AMENDMENT_COMMIT}:{path}"],
                cwd=repository_root.resolve(strict=True),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=60,
                env=dict(FORMAL_GIT_ENVIRONMENT_V1),
            )
            if probe.returncode == 0:
                _fail(f"R3.1 source unexpectedly existed in parent R3: {path}")
        elif hashlib.sha256(
            _git(repository_root, ("show", f"{R3_AMENDMENT_COMMIT}:{path}"))
        ).hexdigest() != old_hash:
            _fail(f"parent R3 source blob hash differs: {path}")
        verified.append(dict(row))
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-preflight/1",
        "amendment_commit": head,
        "sole_parent_commit": R3_AMENDMENT_COMMIT,
        "parent_amendment_commit": R3_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 3,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r3_preattempt_prefix_root_sha256": R3_PREATTEMPT_PREFIX_ROOT_SHA256,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "source_bindings": verified,
        "repository_clean": True,
        "exact_changed_paths_verified": True,
        "changed_worktree_blobs_equal_head": True,
        "changed_path_index_flags_normal": True,
        "formal_identity_entropy_draw_count": 0,
        "m3_start_allowed": False,
        "fixed_audit_directory": FIXED_R3_AUDIT_DIRECTORY.as_posix(),
    }


def _require_existing_audit_directory(path: Path, repository_root: Path) -> Path:
    if path.is_symlink():
        _fail("R3 audit directory may not be a symlink")
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    repository = repository_root.resolve(strict=True)
    if (
        resolved != FIXED_R3_AUDIT_DIRECTORY
        or resolved == repository
        or repository in resolved.parents
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
    ):
        _fail("R3 audit directory is not the fixed caller-owned mode-0700 path")
    return resolved


def _create_or_resume_prepare_audit_directory(
    path: Path, repository_root: Path
) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute != FIXED_R3_AUDIT_DIRECTORY:
        _fail("R3 audit directory differs from fixed attempt-3 path")
    repository = repository_root.resolve(strict=True)
    parent = absolute.parent.resolve(strict=True)
    if absolute == repository or repository in absolute.parents:
        _fail("R3 audit directory must be repository-external")
    if absolute.is_symlink():
        _fail("R3 attempt-3 audit directory may not be a symlink")
    if not absolute.exists():
        os.mkdir(absolute, 0o700)
        os.chmod(absolute, 0o700)
        descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return _require_existing_audit_directory(absolute, repository_root)


def _fsync_directory_v1(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all_v1(descriptor: int, payload: bytes) -> None:
    offset = 0
    view = memoryview(payload)
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            _fail("R3 audit write was short")
        offset += written


def _read_exact_regular_bytes_v1(path: Path) -> bytes:
    if path.is_symlink():
        _fail(f"R3 preparation record is a symlink: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        linked = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.getuid()
            or metadata.st_gid != os.getgid()
            or metadata.st_nlink not in {1, 2}
            or (metadata.st_dev, metadata.st_ino) != (linked.st_dev, linked.st_ino)
        ):
            _fail(f"R3 preparation record inode/mode differs: {path.name}")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _fail(f"R3 preparation record read was short: {path.name}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"R3 preparation record grew while read: {path.name}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _discard_non_authoritative_next_v1(path: Path) -> None:
    """Remove and fsync one hidden audit temp that has no visible authority."""

    if path.exists() or path.is_symlink():
        _fail(f"R3 cannot discard .next for a visible record: {path.name}")
    temporary = path.with_name("." + path.name + ".next")
    if not temporary.exists() and not temporary.is_symlink():
        return
    if temporary.is_symlink():
        _fail(f"R3 non-authoritative .next is a symlink: {path.name}")
    metadata = temporary.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
        or metadata.st_nlink != 1
    ):
        _fail(f"R3 non-authoritative .next metadata differs: {path.name}")
    temporary.unlink()
    _fsync_directory_v1(path.parent)


def _install_prepare_record_v1(path: Path, payload: bytes) -> None:
    """Crash-resumably install one exact pre-attempt record.

    The visible name is installed by no-replace hard link only after the
    ``.next`` inode is fully written and fsynced.  A partial ``.next`` has no
    authority and may be replaced; a visible record is immutable and must
    already equal the deterministic expected bytes.
    """

    temporary = path.with_name("." + path.name + ".next")
    if path.exists() or path.is_symlink():
        if _read_exact_regular_bytes_v1(path) != payload:
            _fail(f"R3 existing preparation record differs: {path.name}")
        if temporary.exists() or temporary.is_symlink():
            try:
                same = temporary.samefile(path)
            except OSError:
                same = False
            if not same:
                _fail(f"R3 committed preparation record has a foreign .next: {path.name}")
            temporary.unlink()
        # Also repairs the post-unlink directory-fsync fault window: an exact
        # visible record with no .next is re-fsynced before it is accepted.
        _fsync_directory_v1(path.parent)
        return
    # A hidden inode has no authority, even when its cached bytes happen to be
    # complete. Discard/recreate it so every future link follows a successful
    # file fsync in this invocation; this closes the prior-file-fsync crash gap.
    _discard_non_authoritative_next_v1(path)
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        _write_all_v1(descriptor, payload)
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory_v1(path.parent)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError:
        if _read_exact_regular_bytes_v1(path) != payload:
            _fail(f"R3 preparation record raced with foreign bytes: {path.name}")
    _fsync_directory_v1(path.parent)
    if not temporary.samefile(path):
        _fail(f"R3 preparation hard-link identity differs: {path.name}")
    temporary.unlink()
    _fsync_directory_v1(path.parent)


def _receipt_record_bytes_v1(fields: Mapping[str, object]) -> bytes:
    return _canonical_json(_r2._with_receipt_sha256(fields))


def _incident_receipt_bytes_equal_v1(
    stored_raw: object, rebuilt_fields: Mapping[str, object]
) -> bool:
    """Compare the authoritative canonical incident bytes, never Python shapes."""

    return type(stored_raw) is bytes and stored_raw == _receipt_record_bytes_v1(
        rebuilt_fields
    )


def _build_exact_audit_record_v1(
    fields: Mapping[str, object],
) -> tuple[dict[str, object], bytes]:
    body = _r2._with_receipt_sha256(fields)
    return body, _canonical_json(body)


def _install_exact_audit_record_v1(
    path: Path, expected: Mapping[str, object], raw: bytes
) -> None:
    """Atomically publish one fully written, self-hashed audit record."""

    if raw != _canonical_json(dict(expected)):
        _fail(f"R3 expected audit bytes differ before install: {path.name}")
    try:
        _install_prepare_record_v1(path, raw)
    except Exception:
        if _exact_audit_record_is_visible_v1(path, raw):
            # Publication is all-or-nothing: a post-link fsync fault may leave
            # the exact visible inode plus its non-authoritative .next link.
            # Stabilize/clean that state, but preserve the original exception
            # so the caller terminalizes instead of continuing silently.
            _install_prepare_record_v1(path, raw)
        raise
    # The reader independently enforces canonical JSON and the self-receipt.
    # Compare its authoritative bytes, not Python container shapes: JSON arrays
    # deserialize as lists even when a typed builder supplied tuples.
    _observed, observed_raw = _r2._read_canonical_audit(path)
    if observed_raw != raw:
        _fail(f"R3 installed audit record differs: {path.name}")


def _exact_audit_record_is_visible_v1(path: Path, raw: bytes) -> bool:
    """Return whether an exact atomic consumption record is currently visible."""

    if not path.exists() and not path.is_symlink():
        return False
    return _read_exact_regular_bytes_v1(path) == raw


def _r2_terminal_chain_snapshot_v1() -> tuple[dict[str, object], ...]:
    if R2_AUDIT_DIRECTORY.is_symlink():
        _fail("R2 audit directory may not be a symlink")
    metadata = R2_AUDIT_DIRECTORY.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
        or {path.name for path in R2_AUDIT_DIRECTORY.iterdir()}
        != set(R2_AUDIT_RAW_SHA256)
    ):
        _fail("R2 audit directory is not the exact six-record terminal chain")
    order = (
        "preflight.json", "incident-diagnostic.json",
        "authorization-request.json", "authorization.json",
        "attempt-start.json", "failure.json",
    )
    records: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for name in order:
        path = R2_AUDIT_DIRECTORY / name
        value, raw = _r2._read_canonical_audit(path)
        digest = hashlib.sha256(raw).hexdigest()
        if (
            digest != R2_AUDIT_RAW_SHA256[name]
            or value.get("receipt_sha256") != R2_AUDIT_RECEIPT_SHA256[name]
        ):
            _fail(f"R2 terminal audit identity differs: {name}")
        item = path.stat()
        rows.append(
            {
                "name": name,
                "raw_sha256": digest,
                "receipt_sha256": value["receipt_sha256"],
                "mode_octal": "0600",
                "size_bytes": item.st_size,
                "st_dev": item.st_dev,
                "st_ino": item.st_ino,
                "uid": item.st_uid,
                "gid": item.st_gid,
            }
        )
        records[name] = value
    common = {
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
    }
    if any(any(record.get(k) != v for k, v in common.items()) for record in records.values()):
        _fail("R2 terminal chain transaction identity differs")
    if (
        records["preflight.json"].get("schema")
        != "hegel-phase3-m25-a8-r2-recovery-audit-preflight/1"
        or records["preflight.json"].get("amendment_commit") != R2_AMENDMENT_COMMIT
        or records["preflight.json"].get("sole_parent_commit") != R1_AMENDMENT_COMMIT
        or records["preflight.json"].get("recovery_attempt_ordinal") != 2
        or records["authorization-request.json"].get("preflight_sha256")
        != R2_AUDIT_RAW_SHA256["preflight.json"]
        or records["authorization-request.json"].get("incident_diagnostic_sha256")
        != R2_AUDIT_RAW_SHA256["incident-diagnostic.json"]
        or records["authorization.json"].get("authorization_request_sha256")
        != R2_AUDIT_RAW_SHA256["authorization-request.json"]
        or records["authorization.json"].get("authorization_actor") != "PROJECT_OWNER"
        or records["attempt-start.json"].get("authorization_sha256")
        != R2_AUDIT_RAW_SHA256["authorization.json"]
        or records["attempt-start.json"].get("recovery_attempt_ordinal") != 2
    ):
        _fail("R2 terminal chain provenance links differ")
    failure = records["failure.json"]
    if (
        failure.get("schema")
        != "hegel-phase3-m25-a8-r2-recovery-audit-failure/1"
        or failure.get("amendment_commit") != R2_AMENDMENT_COMMIT
        or failure.get("recovery_attempt_ordinal") != 2
        or failure.get("attempt_start_sha256")
        != R2_AUDIT_RAW_SHA256["attempt-start.json"]
        or failure.get("admission_sha256_or_null") is not None
        or failure.get("failure_code") != "FAIL_CONTAINER_ACTOR_REPORT_INVALID"
        or failure.get("failure_phase") != "SOURCE_ADMISSION"
        or failure.get("failure_detail_sha256") != R2_FAILURE_DETAIL_SHA256
        or failure.get("formal_identity_entropy_draw_count") != 0
        or failure.get("raw_seed_bytes_read_by_r2_orchestrator") is not False
        or failure.get("raw_seed_sha256_computed") is not False
        or failure.get("m3_start_invoked") is not False
    ):
        _fail("R2 terminal failure receipt fields differ")
    if hashlib.sha256(_canonical_json(rows)).hexdigest() != R2_TERMINAL_CHAIN_ROOT_SHA256:
        _fail("R2 terminal chain root differs")
    return tuple(rows)


def _r3_preattempt_prefix_snapshot_v1() -> tuple[dict[str, object], ...]:
    """Bind the immutable R3 preparation prefix that never consumed attempt 3."""

    audit = R3_PREATTEMPT_AUDIT_DIRECTORY
    if audit.is_symlink():
        _fail("R3 pre-attempt audit directory may not be a symlink")
    metadata = audit.stat()
    order = (
        "preflight.json",
        "incident-diagnostic.json",
        "a8-validation-receipt.json",
        "authorization-request.json",
        "authorization.json",
    )
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
        or {path.name for path in audit.iterdir()} != set(order)
    ):
        _fail("R3 pre-attempt audit is not the exact five-record prefix")
    records: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for name in order:
        path = audit / name
        value, raw = _r2._read_canonical_audit(path)
        digest = hashlib.sha256(raw).hexdigest()
        item = path.stat()
        if (
            digest != R3_PREATTEMPT_AUDIT_RAW_SHA256[name]
            or value.get("receipt_sha256")
            != R3_PREATTEMPT_AUDIT_RECEIPT_SHA256[name]
            or stat.S_IMODE(item.st_mode) != 0o600
            or item.st_uid != os.getuid()
            or item.st_gid != os.getgid()
            or item.st_nlink != 1
        ):
            _fail(f"R3 pre-attempt audit identity differs: {name}")
        rows.append(
            {
                "name": name,
                "raw_sha256": digest,
                "receipt_sha256": value["receipt_sha256"],
                "size_bytes": item.st_size,
                "mode_octal": "0600",
            }
        )
        records[name] = value
    preflight = records["preflight.json"]
    incident = records["incident-diagnostic.json"]
    validation = records["a8-validation-receipt.json"]
    request = records["authorization-request.json"]
    authorization = records["authorization.json"]
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
        or
        preflight.get("schema")
        != f"{R3_PREATTEMPT_AUDIT_SCHEMA_PREFIX}-preflight/1"
        or preflight.get("amendment_commit") != R3_AMENDMENT_COMMIT
        or preflight.get("sole_parent_commit") != R2_AMENDMENT_COMMIT
        or preflight.get("manifest_sha256") != R3_PREATTEMPT_MANIFEST_SHA256
        or preflight.get("recovery_attempt_ordinal") != 3
        or incident.get("schema")
        != f"{R3_PREATTEMPT_AUDIT_SCHEMA_PREFIX}-incident-diagnostic/1"
        or incident.get("recovery_attempt_ordinal") != 3
        or validation.get("schema")
        != "hegel-phase3-m25-a8-r3-a8-validation-receipt/1"
        or request.get("schema")
        != f"{R3_PREATTEMPT_AUDIT_SCHEMA_PREFIX}-authorization-request/1"
        or request.get("amendment_commit") != R3_AMENDMENT_COMMIT
        or request.get("recovery_attempt_ordinal") != 3
        or request.get("preflight_sha256")
        != R3_PREATTEMPT_AUDIT_RAW_SHA256["preflight.json"]
        or request.get("incident_diagnostic_sha256")
        != R3_PREATTEMPT_AUDIT_RAW_SHA256["incident-diagnostic.json"]
        or request.get("a8_validation_receipt_sha256")
        != R3_PREATTEMPT_AUDIT_RAW_SHA256["a8-validation-receipt.json"]
        or authorization.get("schema")
        != f"{R3_PREATTEMPT_AUDIT_SCHEMA_PREFIX}-authorization/1"
        or authorization.get("amendment_commit") != R3_AMENDMENT_COMMIT
        or authorization.get("recovery_attempt_ordinal") != 3
        or authorization.get("authorization_request_sha256")
        != R3_PREATTEMPT_AUDIT_RAW_SHA256["authorization-request.json"]
        or authorization.get("preflight_sha256")
        != R3_PREATTEMPT_AUDIT_RAW_SHA256["preflight.json"]
        or authorization.get("incident_diagnostic_sha256")
        != R3_PREATTEMPT_AUDIT_RAW_SHA256["incident-diagnostic.json"]
        or authorization.get("a8_validation_receipt_sha256")
        != R3_PREATTEMPT_AUDIT_RAW_SHA256["a8-validation-receipt.json"]
        or authorization.get("authorization_actor") != "PROJECT_OWNER"
        or authorization.get("owner_authorized_fixed_transaction_only") is not True
    ):
        _fail("R3 pre-attempt audit provenance links differ")
    if hashlib.sha256(_canonical_json(rows)).hexdigest() != (
        R3_PREATTEMPT_PREFIX_ROOT_SHA256
    ):
        _fail("R3 pre-attempt audit prefix root differs")
    return tuple(rows)


def _r3_preattempt_representation_mismatch_fields_v1(
    stored: Mapping[str, object], rebuilt: Mapping[str, object]
) -> tuple[str, ...]:
    """Locate the old list/tuple mismatch without normalizing either record."""

    missing = object()
    observed: list[str] = []
    for key in sorted(set(stored) | set(rebuilt)):
        stored_value = stored.get(key, missing)
        rebuilt_value = rebuilt.get(key, missing)
        if (
            key == "docker_state"
            and isinstance(stored_value, Mapping)
            and isinstance(rebuilt_value, Mapping)
        ):
            for nested in sorted(set(stored_value) | set(rebuilt_value)):
                if stored_value.get(nested, missing) != rebuilt_value.get(
                    nested, missing
                ):
                    observed.append(f"docker_state.{nested}")
        elif stored_value != rebuilt_value:
            observed.append(key)
    return tuple(observed)


def _build_incident_diagnostic_v1(
    *, custody_directory: Path, public_evidence_path: Path, public_promotion_path: Path
) -> dict[str, object]:
    try:
        base = dict(
            _r2._build_incident_diagnostic_v1(
                custody_directory=custody_directory,
                public_evidence_path=public_evidence_path,
                public_promotion_path=public_promotion_path,
            )
        )
    except _r2.A8R2RecoveryAmendmentError as exc:
        _fail("R2 continuity verifier rejected R3 incident: " + exc.detail)
    r2_rows = _r2_terminal_chain_snapshot_v1()
    # First reproduce the superseded R3 incident exactly.  Its bytes were
    # correct; only the old direct Python list/tuple equality was defective.
    base["schema"] = (
        f"{R3_PREATTEMPT_AUDIT_SCHEMA_PREFIX}-incident-diagnostic/1"
    )
    base["r2_amendment_commit"] = R2_AMENDMENT_COMMIT
    base["recovery_attempt_ordinal"] = 3
    base["r2_terminal_chain"] = r2_rows
    base["r2_terminal_chain_root_sha256"] = R2_TERMINAL_CHAIN_ROOT_SHA256
    base["r2_attempt_start_raw_sha256"] = R2_AUDIT_RAW_SHA256["attempt-start.json"]
    base["r2_failure_raw_sha256"] = R2_AUDIT_RAW_SHA256["failure.json"]
    base["r2_failure_receipt_sha256"] = R2_AUDIT_RECEIPT_SHA256["failure.json"]
    base["r2_admission_sha256_or_null"] = None
    base.pop("raw_seed_bytes_read_by_r2_orchestrator", None)
    base["raw_seed_bytes_read_by_r3_orchestrator"] = False
    legacy_stored, legacy_raw = _r2._read_canonical_audit(
        R3_PREATTEMPT_AUDIT_DIRECTORY / "incident-diagnostic.json"
    )
    legacy_expected = _r2._with_receipt_sha256(base)
    legacy_expected_raw = _canonical_json(legacy_expected)
    mismatch_fields = _r3_preattempt_representation_mismatch_fields_v1(
        legacy_stored, legacy_expected
    )
    if (
        legacy_raw != legacy_expected_raw
        or hashlib.sha256(legacy_raw).hexdigest()
        != R3_PREATTEMPT_AUDIT_RAW_SHA256["incident-diagnostic.json"]
        or legacy_stored == legacy_expected
        or mismatch_fields != R3_PREATTEMPT_REPRESENTATION_MISMATCH_FIELDS
    ):
        _fail("R3 pre-attempt incident defect evidence differs")
    preattempt_rows = _r3_preattempt_prefix_snapshot_v1()
    base["schema"] = f"{AUDIT_SCHEMA_PREFIX}-incident-diagnostic/1"
    base["continuation_action"] = CONTINUATION_ACTION
    base["r3_preattempt_audit_directory"] = (
        R3_PREATTEMPT_AUDIT_DIRECTORY.as_posix()
    )
    base["r3_preattempt_prefix"] = preattempt_rows
    base["r3_preattempt_prefix_root_sha256"] = (
        R3_PREATTEMPT_PREFIX_ROOT_SHA256
    )
    base["r3_preattempt_state"] = (
        "PRE_ATTEMPT_SUPERSEDED_IMPLEMENTATION_DEFECT"
    )
    base["r3_preattempt_attempt_start_sha256_or_null"] = None
    base["pre_attempt_defect_code"] = PRE_ATTEMPT_DEFECT_CODE
    base["pre_attempt_stored_incident_raw_sha256"] = hashlib.sha256(
        legacy_raw
    ).hexdigest()
    base["pre_attempt_rebuilt_incident_raw_sha256"] = hashlib.sha256(
        legacy_expected_raw
    ).hexdigest()
    base["pre_attempt_canonical_bytes_equal"] = True
    base["pre_attempt_python_object_equality"] = False
    base["pre_attempt_representation_mismatch_fields"] = mismatch_fields
    base["authorization_revision_id"] = AUTHORIZATION_REVISION_ID
    return base


def _validation_request_from_incident_v1(
    incident: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    stage = Path(str(incident["stage_directory"]))
    intent_transport, _raw, _row = _r2._read_canonical_regular(
        stage / "prestage-intent.json", mode=0o600
    )
    restored = _restore(intent_transport)
    if type(restored) is not dict:
        _fail("R3 prestage intent is not transport-restorable")
    actor = _transport(restored.get("actor_qualification_report"))
    errata = _transport(restored.get("errata_qualification_report"))
    bundle = _transport(restored.get("live_actor_protocol_qualification_bundle"))
    rows = restored.get("qualification_only_key_id_rows")
    content_id = restored.get("live_actor_protocol_qualification_bundle_content_id")
    if (
        type(actor) is not dict
        or type(errata) is not dict
        or type(bundle) is not dict
        or type(rows) is not tuple
        or len(rows) != 4
        or type(content_id) is not bytes
        or len(content_id) != 32
    ):
        _fail("R3 prestage diagnostic JSON identity differs")
    key_rows: list[dict[str, object]] = []
    for expected, row in enumerate(rows, start=1):
        if (
            type(row) is not dict
            or row.get("purpose_id") != expected
            or type(row.get("qualification_only_key_id_16_bytes")) is not bytes
            or len(row["qualification_only_key_id_16_bytes"]) != 16
        ):
            _fail("R3 prestage qualification key row differs")
        key_rows.append(
            {
                "purpose_id": expected,
                "qualification_only_key_id_hex": row[
                    "qualification_only_key_id_16_bytes"
                ].hex(),
            }
        )
    actor_hash = hashlib.sha256(_canonical_json(actor)).hexdigest()
    errata_hash = hashlib.sha256(_canonical_json(errata)).hexdigest()
    bundle_hash = hashlib.sha256(_canonical_json(bundle)).hexdigest()
    if (
        restored.get("actor_qualification_report_sha256") != actor_hash
        or restored.get("errata_qualification_report_sha256") != errata_hash
        or type(restored.get("live_actor_protocol_qualification_bundle_sha256"))
        is not bytes
        or restored["live_actor_protocol_qualification_bundle_sha256"].hex()
        != bundle_hash
        or bundle_hash != _r2.EXPECTED_LIVE_BUNDLE_SHA256
    ):
        _fail("R3 diagnostic JSON canonical identity differs")
    request = {
        "schema": "hegel-phase3-m25-a8-r3-validation-request/1",
        "basis_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "actor_qualification_report": actor,
        "actor_report_sha256": actor_hash,
        "errata_qualification_report": errata,
        "errata_report_sha256": errata_hash,
        "live_actor_protocol_qualification_bundle": bundle,
        "live_bundle_sha256": bundle_hash,
        "expected_live_bundle_content_id_hex": content_id.hex(),
        "expected_qualification_key_id_rows": key_rows,
        "contains_raw_seed": False,
        "contains_private_key": False,
        "m3_start_allowed": False,
    }
    return request, actor, errata, bundle


def _run_a8_validator_v1(request: Mapping[str, object]) -> tuple[dict[str, object], bytes]:
    tool_metadata = A8_VALIDATOR_TOOL.stat()
    if (
        A8_VALIDATOR_TOOL.is_symlink()
        or not A8_VALIDATOR_TOOL.is_file()
        or stat.S_IMODE(tool_metadata.st_mode) != 0o644
        or tool_metadata.st_uid != os.getuid()
        or tool_metadata.st_gid != os.getgid()
        or FIXED_PYTHON_EXECUTABLE.is_symlink()
        or FIXED_PYTHON_EXECUTABLE.resolve(strict=True) != FIXED_PYTHON_EXECUTABLE
        or not stat.S_ISREG(FIXED_PYTHON_EXECUTABLE.stat().st_mode)
        or stat.S_IMODE(FIXED_PYTHON_EXECUTABLE.stat().st_mode) != 0o755
        or FIXED_PYTHON_EXECUTABLE.stat().st_uid != 0
        or FIXED_PYTHON_EXECUTABLE.stat().st_gid != 0
        or hashlib.sha256(FIXED_PYTHON_EXECUTABLE.read_bytes()).hexdigest()
        != FIXED_PYTHON_EXECUTABLE_SHA256
    ):
        _fail("fixed A8 validator executable identity differs")
    completed = subprocess.run(
        [
            FIXED_PYTHON_EXECUTABLE.as_posix(),
            "-I",
            "-S",
            "-B",
            "-X",
            f"pycache_prefix={FIXED_PYCACHE_PREFIX}",
            A8_VALIDATOR_TOOL.as_posix(),
        ],
        input=_canonical_json(dict(request)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=300,
        cwd=PROJECT_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    if (
        completed.returncode != 0
        or completed.stderr
        or not completed.stdout
        or len(completed.stdout) > 4 * 1024 * 1024
    ):
        _fail(
            "isolated A8 validator rejected the fixed reports: "
            + completed.stderr.decode("utf-8", "replace")[-500:]
        )
    try:
        receipt = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"isolated A8 validator output is invalid: {exc}")
    if type(receipt) is not dict or completed.stdout != _canonical_json(receipt):
        _fail("isolated A8 validator output is not canonical JSON")
    body = dict(receipt)
    claimed = body.pop("receipt_sha256", None)
    input_map = receipt.get("commit_a_input_sha256")
    if (
        set(receipt) != EXPECTED_A8_VALIDATION_RECEIPT_KEYS
        or hashlib.sha256(completed.stdout).hexdigest()
        != EXPECTED_A8_VALIDATION_RECEIPT_RAW_SHA256
        or claimed != hashlib.sha256(_canonical_json(body)).hexdigest()
        or receipt.get("schema")
        != "hegel-phase3-m25-a8-r3-a8-validation-receipt/1"
        or receipt.get("formal_repository_root")
        != FIXED_FORMAL_REPOSITORY_ROOT.as_posix()
        or receipt.get("formal_repository_path_sha256")
        != hashlib.sha256(
            FIXED_FORMAL_REPOSITORY_ROOT.as_posix().encode("utf-8")
        ).hexdigest()
        or receipt.get("formal_repository_commit") != A8_BASIS_COMMIT
        or receipt.get("python_executable")
        != FIXED_PYTHON_EXECUTABLE.as_posix()
        or receipt.get("python_executable_sha256")
        != FIXED_PYTHON_EXECUTABLE_SHA256
        or receipt.get("python_isolated") is not True
        or receipt.get("python_no_site") is not True
        or receipt.get("python_bytecode_disabled") is not True
        or receipt.get("python_pycache_prefix") != FIXED_PYCACHE_PREFIX
        or receipt.get("a8_import_closure_sha256_root")
        != EXPECTED_A8_IMPORT_CLOSURE_SHA256_ROOT
        or receipt.get("a8_validator_dependency_closure_sha256_root")
        != EXPECTED_A8_VALIDATOR_DEPENDENCY_CLOSURE_SHA256_ROOT
        or receipt.get("run_id_hex") != FIXED_RUN_ID_HEX
        or receipt.get("ledger_id_hex") != FIXED_LEDGER_ID_HEX
        or receipt.get("actor_report_sha256") != request["actor_report_sha256"]
        or receipt.get("errata_report_sha256") != request["errata_report_sha256"]
        or receipt.get("live_bundle_sha256") != request["live_bundle_sha256"]
        or receipt.get("live_bundle_content_id_hex")
        != request["expected_live_bundle_content_id_hex"]
        or receipt.get("qualification_key_id_rows")
        != request["expected_qualification_key_id_rows"]
        or receipt.get("commit_a_input_count") != 98
        or type(input_map) is not dict
        or len(input_map) != 98
        or any(
            type(path) is not str
            or type(digest) is not str
            or _HEX_64.fullmatch(digest) is None
            for path, digest in (
                input_map.items() if type(input_map) is dict else ()
            )
        )
        or receipt.get("commit_a_input_sha256_root")
        != hashlib.sha256(_canonical_json(input_map)).hexdigest()
        or receipt.get("actor_technical_eligible") is not True
        or receipt.get("errata_status")
        != request["errata_qualification_report"].get("status")
        or receipt.get("transaction_bundle_replay_passed") is not True
        or receipt.get("formal_identity_entropy_draw_count") != 0
        or any(
            receipt.get(key) is not False
            for key in (
                "contains_raw_seed", "contains_private_key", "raw_seed_bytes_read",
                "raw_seed_sha256_computed", "m3_start_invoked",
            )
        )
    ):
        _fail("isolated A8 validation receipt fields differ")
    return receipt, completed.stdout


def _unchanged_a8_input_bindings_v1() -> dict[str, str]:
    bindings: dict[str, str] = {}
    repository = REPOSITORY_ROOT.resolve(strict=True)
    for path in REQUIRED_COMMIT_A_INPUTS:
        relative = path.resolve(strict=True).relative_to(repository).as_posix()
        if relative in R3_RUNTIME_EXCEPTION_PATHS:
            continue
        if path.is_symlink():
            _fail(f"unchanged A8 runtime input is a symlink: {relative}")
        current = path.read_bytes()
        frozen = _git(repository, ("show", f"{A8_BASIS_COMMIT}:{relative}"))
        if current != frozen:
            _fail(f"unchanged A8 runtime input differs: {relative}")
        bindings[relative] = hashlib.sha256(current).hexdigest()
    root = hashlib.sha256(_executor_canonical_json(bindings)).hexdigest()
    if len(bindings) != EXPECTED_UNCHANGED_A8_INPUT_COUNT or root != EXPECTED_UNCHANGED_A8_INPUT_ROOT:
        _fail("unchanged A8 runtime closure/root differs")
    return bindings


def _build_source_admission_v1(
    *, amendment_commit: object, incident_raw: bytes, validation_raw: bytes,
    validation: Mapping[str, object], unchanged_inputs: Mapping[str, str],
) -> dict[str, object]:
    if type(amendment_commit) is not str or _HEX_40.fullmatch(amendment_commit) is None:
        _fail("R3 source admission amendment commit differs")
    if (
        len(unchanged_inputs) != EXPECTED_UNCHANGED_A8_INPUT_COUNT
        or hashlib.sha256(_executor_canonical_json(unchanged_inputs)).hexdigest()
        != EXPECTED_UNCHANGED_A8_INPUT_ROOT
    ):
        _fail("R3 source admission unchanged-input root differs")
    return {
        "schema": "hegel-phase3-m25-a8-r3-source-admission/1",
        "basis_commit": A8_BASIS_COMMIT,
        "r1_amendment_commit": R1_AMENDMENT_COMMIT,
        "r2_amendment_commit": R2_AMENDMENT_COMMIT,
        "r3_amendment_commit": amendment_commit,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 3,
        "continuation_action": SOURCE_ADMISSION_CONTINUATION_ACTION,
        "r1_failure_raw_sha256": _r2.R1_AUDIT_RAW_SHA256["failure.json"],
        "r1_failure_receipt_sha256": _r2.R1_FAILURE_RECEIPT_SHA256,
        "r2_terminal_chain_root_sha256": R2_TERMINAL_CHAIN_ROOT_SHA256,
        "r2_attempt_start_raw_sha256": R2_AUDIT_RAW_SHA256["attempt-start.json"],
        "r2_failure_raw_sha256": R2_AUDIT_RAW_SHA256["failure.json"],
        "r2_failure_receipt_sha256": R2_AUDIT_RECEIPT_SHA256["failure.json"],
        "r2_admission_sha256_or_null": None,
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
        "unchanged_a8_input_sha256_root": EXPECTED_UNCHANGED_A8_INPUT_ROOT,
        "actor_report_sha256": validation["actor_report_sha256"],
        "errata_report_sha256": validation["errata_report_sha256"],
        "live_bundle_sha256": validation["live_bundle_sha256"],
    }


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
        "recovery_attempt_ordinal": 3,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r3_preattempt_prefix_root_sha256": R3_PREATTEMPT_PREFIX_ROOT_SHA256,
        "r3_preattempt_attempt_start_sha256_or_null": None,
        "pre_attempt_defect_code": PRE_ATTEMPT_DEFECT_CODE,
        "continuation_action": CONTINUATION_ACTION,
        "preflight_sha256": hashlib.sha256(preflight_raw).hexdigest(),
        "incident_diagnostic_sha256": hashlib.sha256(incident_raw).hexdigest(),
        "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
        "r1_failure_raw_sha256": _r2.R1_AUDIT_RAW_SHA256["failure.json"],
        "r1_failure_receipt_sha256": _r2.R1_FAILURE_RECEIPT_SHA256,
        "r2_terminal_chain_root_sha256": R2_TERMINAL_CHAIN_ROOT_SHA256,
        "r2_attempt_start_raw_sha256": R2_AUDIT_RAW_SHA256["attempt-start.json"],
        "r2_failure_raw_sha256": R2_AUDIT_RAW_SHA256["failure.json"],
        "r2_failure_receipt_sha256": R2_AUDIT_RECEIPT_SHA256["failure.json"],
        "r2_admission_sha256_or_null": None,
        "requested_action": "COMPLETE_ONLY_REAL_PENDING_RESUME",
        "ordinary_execute_allowed": False,
        "redraw_allowed": False,
        "abort_allowed": False,
        "poststage_recovery_allowed": False,
        "formal_identity_entropy_draw_count": 0,
        "m3_start_allowed": False,
    }


def prepare_fixed_a8_r3_authorization_v1(
    *, audit_directory: Path, custody_directory: Path,
    public_evidence_path: Path, public_promotion_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> None:
    preflight = inspect_r3_source_preflight_v1(
        repository_root=repository_root, manifest_path=manifest_path
    )
    incident = _build_incident_diagnostic_v1(
        custody_directory=custody_directory,
        public_evidence_path=public_evidence_path,
        public_promotion_path=public_promotion_path,
    )
    request, _actor, _errata, _bundle = _validation_request_from_incident_v1(incident)
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
        _fail("R3 preparation audit contains a non-prefix record")
    visible = [name in observed for name in order]
    if any(visible[index] and not all(visible[:index]) for index in range(len(order))):
        _fail("R3 preparation visible records are not an exact prefix")
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
        "recovery_attempt_ordinal": 3,
        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
        "r3_preattempt_prefix_root_sha256": R3_PREATTEMPT_PREFIX_ROOT_SHA256,
        "r3_preattempt_attempt_start_sha256_or_null": None,
        "pre_attempt_defect_code": PRE_ATTEMPT_DEFECT_CODE,
        "continuation_action": CONTINUATION_ACTION,
        "preflight_sha256": hashlib.sha256(preflight_raw).hexdigest(),
        "incident_diagnostic_sha256": hashlib.sha256(incident_raw).hexdigest(),
        "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
        "authorization_request_sha256": hashlib.sha256(request_raw).hexdigest(),
        "r2_terminal_chain_root_sha256": R2_TERMINAL_CHAIN_ROOT_SHA256,
        "r2_failure_raw_sha256": R2_AUDIT_RAW_SHA256["failure.json"],
        "r2_failure_receipt_sha256": R2_AUDIT_RECEIPT_SHA256["failure.json"],
        "authorization_actor": "PROJECT_OWNER",
        "owner_authorized_fixed_transaction_only": True,
        "ordinary_execute_invoked": False,
        "redraw_allowed": False,
        "abort_allowed": False,
        "poststage_recovery_allowed": False,
        "formal_identity_entropy_draw_count": 0,
        "m3_start_allowed": False,
    }


def write_fixed_a8_r3_owner_authorization_v1(
    *, audit_directory: Path, owner_confirmation: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    if owner_confirmation != OWNER_CONFIRMATION:
        _fail("R3 owner confirmation phrase differs")
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
        _fail("R3 pre-authorization audit path set differs")
    preflight, preflight_raw = _r2._read_canonical_audit(audit / "preflight.json")
    _incident, incident_raw = _r2._read_canonical_audit(
        audit / "incident-diagnostic.json"
    )
    _validation, validation_raw, _row = _r2._read_canonical_regular(
        audit / "a8-validation-receipt.json", mode=0o600
    )
    request, request_raw = _r2._read_canonical_audit(
        audit / "authorization-request.json"
    )
    expected_request = _r2._with_receipt_sha256(
        _authorization_request_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
        )
    )
    if request != expected_request:
        _fail("R3 authorization request differs")
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
    evidence_raw = public_evidence_path.read_bytes()
    promotion_raw = public_promotion_path.read_bytes()
    receipt_path = public_promotion_path.with_name(
        public_promotion_path.name + ".publication-receipt.json"
    )
    receipt_raw = receipt_path.read_bytes()
    seed_verification_raw = (
        stage_directory / "seed-custody-verification.json"
    ).read_bytes()
    replayed = replay(payload)
    if (
        evidence_raw != _executor_canonical_json(payload)
        or promotion != replayed
        or promotion_raw != _executor_canonical_json(replayed)
    ):
        _fail("R3 final public evidence/promotion replay differs")
    gate_report = replayed.get("gate_report")
    inputs = load_gate_evidence_inputs_v1(payload)
    if (
        not isinstance(gate_report, Mapping)
        or gate_report.get("gates_after") != 24
        or gate_report.get("all_gates_15_24_passed") is not True
        or gate_report.get("child_state") != "NOT_RUN"
        or gate_report.get("m3_run_started") is not False
        or len(M3_RUN_OUTPUT_ROOTS) != 15
        or any(
            inputs.run_genesis_fields.get(f"{name}_or_null") is not None
            for name in M3_RUN_OUTPUT_ROOTS
        )
    ):
        _fail("R3 final replay is not 24/24 NOT_RUN with 15 null outputs")
    marker = read_marker_snapshot_v1(
        custody_directory / "split_seed_instantiation.marker"
    )
    if (
        marker.state != "COMPLETE"
        or marker != inputs.marker_snapshot
        or marker.seed_commitment_manifest_root is None
    ):
        _fail("R3 final COMPLETE marker differs")
    try:
        publication_receipt = json.loads(receipt_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"R3 publication receipt is invalid: {exc}")
    if (
        type(publication_receipt) is not dict
        or _canonical_json(publication_receipt) != receipt_raw
        or publication_receipt.get("basis_commit") != A8_BASIS_COMMIT
        or publication_receipt.get("run_id_hex") != FIXED_RUN_ID_HEX
        or publication_receipt.get("ledger_id_hex") != FIXED_LEDGER_ID_HEX
        or publication_receipt.get("public_evidence_sha256")
        != hashlib.sha256(evidence_raw).hexdigest()
        or publication_receipt.get("public_promotion_sha256")
        != hashlib.sha256(promotion_raw).hexdigest()
        or publication_receipt.get("prospective_public_replay_passed") is not True
        or publication_receipt.get("marker_was_complete_during_staging") is not False
        or publication_receipt.get("actor_cleanup_required_before_publication") is not True
        or publication_receipt.get("authority_disclosure")
        != dict(TECHNICAL_ACTOR_DISCLOSURE_V1)
        or any(
            publication_receipt.get(name) is not False
            for name in (
                "contains_private_key", "contains_raw_split_seed",
                "contains_split_assignment_rows",
            )
        )
        or publication_receipt.get(
            "seed_custody_verification_receipt_sha256_or_null"
        ) != hashlib.sha256(seed_verification_raw).hexdigest()
    ):
        _fail("R3 official publication receipt differs")
    assert_public_payload_contains_no_secret_fields(publication_receipt)
    return {
        "public_evidence_sha256": hashlib.sha256(evidence_raw).hexdigest(),
        "public_promotion_sha256": hashlib.sha256(promotion_raw).hexdigest(),
        "publication_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "seed_custody_verification_receipt_sha256": hashlib.sha256(
            seed_verification_raw
        ).hexdigest(),
        "complete_marker_seed_commitment_manifest_root_hex": (
            marker.seed_commitment_manifest_root.hex()
        ),
        "complete_marker_custodian_key_id_hex": marker.custodian_key_id.hex(),
    }


def execute_fixed_a8_r3_recovery_v1(
    *, custody_directory: Path, rust_formal_replay_binary: Path,
    rust_bridge_dag_replay_binary: Path,
    rust_bridge_dag_qualification_report: Path,
    public_evidence_path: Path, public_promotion_path: Path,
    audit_directory: Path,
    repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> tuple[dict[str, object], dict[str, object]]:
    preflight_now = inspect_r3_source_preflight_v1(
        repository_root=repository_root, manifest_path=manifest_path
    )
    audit = _require_existing_audit_directory(audit_directory, repository_root)
    terminal_names = ("attempt-start.json", "admission.json", "failure.json", "finalize.json")
    if any((audit / name).exists() or (audit / name).is_symlink() for name in terminal_names):
        _fail("R3 attempt-3 was already consumed or has a terminal record")
    expected_before = {
        "preflight.json", "incident-diagnostic.json", "a8-validation-receipt.json",
        "authorization-request.json", "authorization.json",
    }
    observed_before = {path.name for path in audit.iterdir()}
    if (
        not expected_before.issubset(observed_before)
        or not observed_before.issubset(
            expected_before | {".attempt-start.json.next"}
        )
    ):
        _fail("R3 pre-attempt audit path set differs")
    preflight, preflight_raw = _r2._read_canonical_audit(audit / "preflight.json")
    incident, incident_raw = _r2._read_canonical_audit(
        audit / "incident-diagnostic.json"
    )
    validation, validation_raw, _validation_row = _r2._read_canonical_regular(
        audit / "a8-validation-receipt.json", mode=0o600
    )
    request, request_raw = _r2._read_canonical_audit(
        audit / "authorization-request.json"
    )
    authorization, authorization_raw = _r2._read_canonical_audit(
        audit / "authorization.json"
    )
    if preflight != _r2._with_receipt_sha256(preflight_now):
        _fail("stored R3 preflight differs from current clean R3 source")
    incident_now = _build_incident_diagnostic_v1(
        custody_directory=custody_directory,
        public_evidence_path=public_evidence_path,
        public_promotion_path=public_promotion_path,
    )
    # Canonical bytes are the authority. JSON arrays intentionally deserialize
    # as lists while the deterministic builder uses typed tuples in nine
    # diagnostic fields; direct Python object equality caused the superseded
    # R3 pre-attempt rejection despite byte-identical receipts.
    if not _incident_receipt_bytes_equal_v1(incident_raw, incident_now):
        _fail("stored R3 incident differs before attempt")
    validation_request, actor_report, errata_report, _expected_bundle = (
        _validation_request_from_incident_v1(incident_now)
    )
    validation_now, validation_now_raw = _run_a8_validator_v1(validation_request)
    if validation_now_raw != validation_raw or validation_now != validation:
        _fail("prepare/execute isolated A8 validation receipts differ")
    if request != _r2._with_receipt_sha256(
        _authorization_request_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
        )
    ):
        _fail("stored R3 authorization request differs")
    if authorization != _r2._with_receipt_sha256(
        _expected_authorization_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            validation_raw=validation_raw,
            request_raw=request_raw,
        )
    ):
        _fail("stored R3 owner authorization differs")
    runtime_rows = _r2._validate_runtime_artifacts_before_attempt_v1(
        rust_formal_replay_binary=rust_formal_replay_binary,
        rust_bridge_dag_replay_binary=rust_bridge_dag_replay_binary,
        rust_bridge_dag_qualification_report=rust_bridge_dag_qualification_report,
        expected_bindings=incident.get("runtime_artifact_bindings"),
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
                _fail("R3 acquired recovery is not the fixed A8 PENDING/RESERVED transaction")
            actors.timestamp = recovery.marker_snapshot.created_at_unix_seconds
            if (
                _canonical_json(_transport(recovery.prestage_intent_fields.get(
                    "actor_qualification_report"
                ))) != _canonical_json(actor_report)
                or _canonical_json(_transport(recovery.prestage_intent_fields.get(
                    "errata_qualification_report"
                ))) != _canonical_json(errata_report)
            ):
                _fail("R3 acquired diagnostic reports differ from A8 receipt")
            source_admission = _build_source_admission_v1(
                amendment_commit=preflight["amendment_commit"],
                incident_raw=incident_raw,
                validation_raw=validation_raw,
                validation=validation,
                unchanged_inputs=unchanged_inputs,
            )
            failure_phase = "ATTEMPT_START_DURABILITY"
            _attempt, attempt_start_raw = _build_exact_audit_record_v1(
                {
                    "schema": f"{AUDIT_SCHEMA_PREFIX}-attempt-start/1",
                    "amendment_commit": preflight["amendment_commit"],
                    "formal_repository_commit": A8_BASIS_COMMIT,
                    "run_id_hex": FIXED_RUN_ID_HEX,
                    "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                    "recovery_attempt_ordinal": 3,
                    "authorization_revision_id": AUTHORIZATION_REVISION_ID,
                    "r3_preattempt_prefix_root_sha256": (
                        R3_PREATTEMPT_PREFIX_ROOT_SHA256
                    ),
                    "r3_preattempt_attempt_start_sha256_or_null": None,
                    "pre_attempt_defect_code": PRE_ATTEMPT_DEFECT_CODE,
                    "continuation_action": CONTINUATION_ACTION,
                    "authorization_sha256": hashlib.sha256(authorization_raw).hexdigest(),
                    "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
                    "r2_terminal_chain_root_sha256": R2_TERMINAL_CHAIN_ROOT_SHA256,
                    "r2_failure_raw_sha256": R2_AUDIT_RAW_SHA256["failure.json"],
                    "runtime_artifact_metadata": runtime_rows,
                    "complete_seed_resume_only": True,
                    "accepted_worker_mode": "REAL_PENDING_RESUME",
                    "formal_identity_entropy_draw_count": 0,
                    "ordinary_execute_invoked": False,
                    "m3_start_invoked": False,
                    "raw_seed_bytes_read_by_r3_orchestrator": False,
                    "raw_seed_sha256_computed": False,
                },
            )
            _install_exact_audit_record_v1(
                audit / "attempt-start.json", _attempt, attempt_start_raw
            )
            failure_phase = "SOURCE_ADMISSION_DURABILITY"
            _admission, admission_raw = _build_exact_audit_record_v1(
                {
                    "schema": f"{AUDIT_SCHEMA_PREFIX}-admission/1",
                    "amendment_commit": preflight["amendment_commit"],
                    "formal_repository_commit": A8_BASIS_COMMIT,
                    "run_id_hex": FIXED_RUN_ID_HEX,
                    "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                    "recovery_attempt_ordinal": 3,
                    "authorization_revision_id": AUTHORIZATION_REVISION_ID,
                    "r3_preattempt_prefix_root_sha256": (
                        R3_PREATTEMPT_PREFIX_ROOT_SHA256
                    ),
                    "r3_preattempt_attempt_start_sha256_or_null": None,
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
                    "raw_seed_bytes_read_by_r3_orchestrator": False,
                    "raw_seed_sha256_computed": False,
                    "m3_start_invoked": False,
                },
            )
            _install_exact_audit_record_v1(
                audit / "admission.json", _admission, admission_raw
            )

            def guard(candidate: PendingCeremonyRecoveryV1) -> Mapping[str, object]:
                if candidate is not recovery:
                    raise FormalContainerExecutorError(
                        FAIL_RECOVERY_SOURCE_ADMISSION,
                        "R3 admission may authorize only the acquired recovery object",
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
        finalize, finalize_raw = _build_exact_audit_record_v1(
            {
                "schema": f"{AUDIT_SCHEMA_PREFIX}-finalize/1",
                "amendment_commit": preflight["amendment_commit"],
                "formal_repository_commit": A8_BASIS_COMMIT,
                "run_id_hex": FIXED_RUN_ID_HEX,
                "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                "recovery_attempt_ordinal": 3,
                "authorization_revision_id": AUTHORIZATION_REVISION_ID,
                "r3_preattempt_prefix_root_sha256": (
                    R3_PREATTEMPT_PREFIX_ROOT_SHA256
                ),
                "r3_preattempt_attempt_start_sha256_or_null": None,
                "r2_failure_raw_sha256": R2_AUDIT_RAW_SHA256["failure.json"],
                "r2_failure_receipt_sha256": R2_AUDIT_RECEIPT_SHA256["failure.json"],
                "attempt_start_sha256": hashlib.sha256(attempt_start_raw).hexdigest(),
                "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
                "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
                **final,
                "formal_gates_after": 24,
                "child_state": "NOT_RUN",
                "m3_start_invoked": False,
                "accepted_worker_mode": "REAL_PENDING_RESUME",
                "raw_seed_bytes_read_by_r3_orchestrator": False,
                "raw_seed_sha256_computed": False,
                "formal_identity_entropy_draw_count": 0,
            },
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
            # The formal publication and its exact finalize record both exist;
            # only a post-link durability call raised.  Finish cleaning the
            # atomic installer and report the already-verified success.
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
            # If publication of the exact hard link succeeded but a following
            # directory fsync raised, resume the atomic installer to remove its
            # non-authoritative .next link before terminalizing the attempt.
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
                failure, failure_raw = _build_exact_audit_record_v1(
                    {
                        "schema": f"{AUDIT_SCHEMA_PREFIX}-failure/1",
                        "amendment_commit": preflight["amendment_commit"],
                        "formal_repository_commit": A8_BASIS_COMMIT,
                        "run_id_hex": FIXED_RUN_ID_HEX,
                        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                        "recovery_attempt_ordinal": 3,
                        "authorization_revision_id": AUTHORIZATION_REVISION_ID,
                        "r3_preattempt_prefix_root_sha256": (
                            R3_PREATTEMPT_PREFIX_ROOT_SHA256
                        ),
                        "r3_preattempt_attempt_start_sha256_or_null": None,
                        "r2_failure_raw_sha256": R2_AUDIT_RAW_SHA256["failure.json"],
                        "r2_failure_receipt_sha256": R2_AUDIT_RECEIPT_SHA256["failure.json"],
                        "attempt_start_sha256": hashlib.sha256(attempt_start_raw).hexdigest(),
                        "admission_sha256_or_null": (
                            None if not admission_visible
                            else hashlib.sha256(admission_raw).hexdigest()
                        ),
                        "a8_validation_receipt_sha256": hashlib.sha256(validation_raw).hexdigest(),
                        "failure_code": str(getattr(exc, "code", type(exc).__name__)),
                        "failure_phase": failure_phase,
                        "failure_detail_sha256": hashlib.sha256(
                            str(exc).encode("utf-8", "replace")
                        ).hexdigest(),
                        "formal_identity_entropy_draw_count": 0,
                        "raw_seed_bytes_read_by_r3_orchestrator": False,
                        "raw_seed_sha256_computed": False,
                        "m3_start_invoked": False,
                    },
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
    "A8R3RecoveryAmendmentError",
    "DEFAULT_MANIFEST_PATH",
    "execute_fixed_a8_r3_recovery_v1",
    "inspect_r3_source_preflight_v1",
    "prepare_fixed_a8_r3_authorization_v1",
    "write_fixed_a8_r3_owner_authorization_v1",
]
