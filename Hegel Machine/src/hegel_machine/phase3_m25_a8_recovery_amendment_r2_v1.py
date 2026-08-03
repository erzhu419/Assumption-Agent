"""Attempt-2, complete-seed-only recovery for the fixed failed A8 ceremony.

R2 is diagnostic recovery provenance.  It neither changes the formal A8
repository identity nor provides an ordinary execution/retry path.  The only
state-changing entry point consumes a new attempt-2 authorization exactly once
and resumes the already-complete seed prefix of the fixed PENDING transaction.
"""

from __future__ import annotations

import hashlib
import json
import os
import fcntl
from pathlib import Path
import re
import stat
import subprocess
from typing import Final, Mapping, NoReturn, Sequence

from .phase3_m25_a8_recovery_amendment_v1 import (
    A8R1RecoveryDockerActorsV1,
    A8RecoveryAmendmentError as A8R1RecoveryAmendmentError,
)
from .phase3_m25_container_ceremony_v1 import (
    TECHNICAL_ACTOR_DISCLOSURE_V1,
    read_marker_snapshot_v1,
    validate_ceremony_admission_v1,
)
from .phase3_m25_formal_container_executor_v1 import (
    FAIL_RECOVERY_SOURCE_ADMISSION,
    FormalContainerExecutorError,
    PendingCeremonyRecoveryV1,
    REQUIRED_COMMIT_A_INPUTS,
    FormalCeremonyTransactionV1,
    _canonical_json as _executor_canonical_json,
    _continue_pre_stage_pending_recovery_core_v1,
    _restore,
    _transport,
    acquire_pending_ceremony_recovery_v1,
    load_gate_evidence_inputs_v1,
    replay_public_gate_evidence_v1,
)
from .phase3_m25_formal_static_basis_v1 import (
    FORMAL_GIT_ENVIRONMENT_V1,
    FORMAL_GIT_EXECUTABLE,
)
from .phase3_container_actor_runtime_v1 import validate_qualification_report
from .phase3_m25_errata_qualification_v1 import (
    validate_dual_errata_qualification_report,
)
from .phase3_m25_external_v1 import assert_public_payload_contains_no_secret_fields
from .phase3_m25_wire_v1 import M3_RUN_OUTPUT_ROOTS


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
DEFAULT_MANIFEST_PATH: Final = (
    PROJECT_ROOT / "config/phase3_m25_a8_recovery_amendment_r2_v1.json"
)

A8_BASIS_COMMIT: Final = "0af65964235390ce2bebefea7379eaa9c50eda24"
R1_AMENDMENT_COMMIT: Final = "0349131599a688470c15eded51f942eefeded392"
FIXED_RUN_ID_HEX: Final = "e4af9f57c38fb298462ec628c4ed8a03"
FIXED_LEDGER_ID_HEX: Final = "ec849e2f1e2e1163cfc450370b25b484"
FIXED_RUN_ID: Final = bytes.fromhex(FIXED_RUN_ID_HEX)
FIXED_LEDGER_ID: Final = bytes.fromhex(FIXED_LEDGER_ID_HEX)
R1_AUDIT_DIRECTORY: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/"
    "phase3-m25-0af65964235390ce2bebefea7379eaa9c50eda24/"
    "recovery-audit-e4af9f57c38fb298462ec628c4ed8a03"
)
FIXED_R2_AUDIT_DIRECTORY: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/"
    "phase3-m25-0af65964235390ce2bebefea7379eaa9c50eda24/"
    "recovery-audit-r2-e4af9f57c38fb298462ec628c4ed8a03-attempt-2"
)
MANIFEST_SCHEMA: Final = "hegel-phase3-m25-a8-recovery-amendment-r2/1"
AUDIT_SCHEMA_PREFIX: Final = "hegel-phase3-m25-a8-r2-recovery-audit"
FAIL_AMENDMENT: Final = "FAIL_M25_A8_R2_RECOVERY_AMENDMENT"
OWNER_CONFIRMATION: Final = (
    "AUTHORIZE_A8_R2_ATTEMPT_2_COMPLETE_ONLY_REAL_PENDING_RESUME"
)
EXPECTED_LIVE_BUNDLE_SHA256: Final = (
    "b1866e49a3d7aa3b4a649f94a5595591576a0d72e25bd844f280953ace643404"
)
R1_AUDIT_RAW_SHA256: Final = {
    "preflight.json": "42110a9cfd9a5a5d416bf8fd09cebb5dab7fed38cf2d72a40db291b798856a1e",
    "authorization-request.json": "6ed40f5a116bbf98516d003e2761640b8029141e383ae8ea1291cb0307f7af05",
    "authorization.json": "14a108b28bf7ee4e47c28d292238b62c62a8b302dbabae7f1a57973a63711569",
    "failure.json": "d4b7be4432b4101de5aab1693e37ae5769d1587155d634b4e746fee60109168a",
}
R1_FAILURE_RECEIPT_SHA256: Final = (
    "ce8948da791a1c42d934ec4a3752ba4bbe5484f96add28f9df5e094444ecb658"
)
FIXED_CONTINUITY_SHA256: Final = {
    "phase3_m25_ceremony.lock": "f71d9cb18f0aa74f055afaf496ef6f104f8e88a576dfb284516037d5763badb6",
    "split_seed_generation.intent": "98a8eed68ec8c3b3d180248de1b1cafe6e6553f03c5d60312cf5c8006aea8821",
    "split_seed_generation.complete": "f8b1df9f665199b9c9dd936c3ef0bf118b875593916579b81876865f7b09b75e",
    "split_seed_instantiation.marker": "760b95cdd4c14792c920aa9ca577cb72b40d992f2b498e672eb6c4d4b3073e77",
    "transaction-journal.json": "46815d92b0bc353db06bf120cc7fb5cad754e31bb99e4b682fb97cba7a944646",
    "prestage-intent.json": "89d8414cd68adaa084b3dfe865abf5d9245806764d89413dc7d1503e6dffc0ab",
    "actor-trust-checkpoint.json": "32a4809a91fa1e92f610bcb01a21e3e8ec7d5b4c813e80e1346526720baf7cb5",
    "live-qualification-bundle.json": EXPECTED_LIVE_BUNDLE_SHA256,
    "recovery-anchor.json": "05a634775c17b65e4363fdf4b00c643ec2e73badd95d66d85d245d75766d9c46",
    "recovery-anchor.ready.json": "7efa3b5f414a710939f86be5f06cb04ae1cd841635b80a4c014866da44dfd4c2",
    ".phase3_m25_formal_gate_evidence_v1.json.hegel-reserved": "e9549ba6a368d9933b4ffb3d4cd65c0a970cf19f09fe95065f0c96f463389034",
    ".phase3_m25_gate_promotion_v1.json.hegel-reserved": "8fd07f7157ed66218ea237da4cbc2b4e6d139dca3cd9119bf96b020bfe261396",
    ".phase3_m25_gate_promotion_v1.json.publication-receipt.json.hegel-reserved": "2cbfc45b473df7b9ecde7054bbf970499daa126f844958be318685e7c7b3adfc",
}
FIXED_SPLIT_VERSION_DIGEST_HEX: Final = (
    "903f6535244c54c6c5e35c23dc8c7e5b453732a8fe749161851580aad0e9655e"
)
FIXED_PENDING_CUSTODIAN_KEY_ID_HEX: Final = "61f5caeede2973d1a42ff60b0b9dcbbb"
FIXED_FORMAL_RUST_BINARY: Final = Path(
    "/home/erzhu419/mine_code/Asumption Agent/Hegel Machine/rust/"
    "formal_bridge_m25/target/debug/hegel-formal-bridge-m25"
)
FIXED_FORMAL_RUST_BINARY_SHA256: Final = (
    "d38eabce2be158326fe16a7185ffc2c9be1262ce8d5098afed25eff431465093"
)
FIXED_BRIDGE_RUST_BINARY: Final = Path(
    "/home/erzhu419/mine_code/Asumption Agent/Hegel Machine/rust/"
    "m25_bridge_dag_replay/target/commit_a_qualified/"
    "hegel-m25-bridge-dag-replay"
)
FIXED_BRIDGE_RUST_BINARY_SHA256: Final = (
    "ec5938c598775eee5975028fa31622aeee4328b28abfd14209676c6fb104644a"
)
FIXED_BRIDGE_REPORT: Final = Path(
    "/home/erzhu419/mine_code/Asumption Agent/Hegel Machine/artifacts/"
    "phase3_m25_external/phase3_m25_bridge_dag_rust_binary_qualification_v1.json"
)
FIXED_BRIDGE_REPORT_RAW_SHA256: Final = (
    "5997beef28ac4edfd462361b1552cd7682d1e95f431e4986f845aa680bc213c9"
)
FIXED_BRIDGE_REPORT_DIAGNOSTIC_SHA256: Final = (
    "2341fc8258d399bf4c57c569a2c0d8927a9135cc1bd67ca4efb66a729d402c29"
)
FIXED_RUNTIME_ARTIFACTS: Final = (
    {
        "path": FIXED_FORMAL_RUST_BINARY.as_posix(),
        "sha256": FIXED_FORMAL_RUST_BINARY_SHA256,
        "mode_octal": "0755",
        "diagnostic_sha256_or_null": None,
    },
    {
        "path": FIXED_BRIDGE_RUST_BINARY.as_posix(),
        "sha256": FIXED_BRIDGE_RUST_BINARY_SHA256,
        "mode_octal": "0755",
        "diagnostic_sha256_or_null": None,
    },
    {
        "path": FIXED_BRIDGE_REPORT.as_posix(),
        "sha256": FIXED_BRIDGE_REPORT_RAW_SHA256,
        "mode_octal": "0644",
        "diagnostic_sha256_or_null": FIXED_BRIDGE_REPORT_DIAGNOSTIC_SHA256,
    },
)
FIXED_STAGE_INVENTORY: Final = frozenset(
    {
        "actor-trust-checkpoint.json",
        "live-qualification-bundle.json",
        "prestage-intent.json",
        "recovery-anchor.json",
        "recovery-anchor.ready.json",
        "transaction-journal.json",
    }
)
R2_RUNTIME_EXCEPTION_PATHS: Final = frozenset(
    {
        "Hegel Machine/src/hegel_machine/phase3_m25_formal_container_executor_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_amendment_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_cli_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_amendment_r2_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_cli_r2_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m3_implementation_qualification_v1.py",
        "Hegel Machine/rust/formal_bridge_m25/src/lib.rs",
    }
)


class A8R2RecoveryAmendmentError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(detail: str) -> NoReturn:
    raise A8R2RecoveryAmendmentError(FAIL_AMENDMENT, detail)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _git(repository_root: Path, arguments: Sequence[str]) -> bytes:
    if (
        not FORMAL_GIT_EXECUTABLE.is_file()
        or FORMAL_GIT_EXECUTABLE.resolve(strict=True) != FORMAL_GIT_EXECUTABLE
        or not arguments
        or any(type(value) is not str or not value or "\0" in value for value in arguments)
    ):
        _fail("formal Git executable or arguments differ")
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
        _fail("formal Git check failed: " + completed.stderr.decode("utf-8", "replace")[-500:])
    return completed.stdout


def inspect_r2_source_preflight_v1(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, object]:
    """Verify clean R2 as the sole direct child of the frozen R1 amendment."""

    manifest, manifest_raw = _load_manifest(manifest_path)
    head = _git(repository_root, ["rev-parse", "--verify", "HEAD^{commit}"]).decode("ascii").strip()
    parents = _git(repository_root, ["show", "-s", "--format=%P", head]).decode("ascii").strip().split()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None or parents != [R1_AMENDMENT_COMMIT]:
        _fail("R2 must be one committed sole child of R1")
    if _git(repository_root, ["status", "--porcelain=v1", "--untracked-files=all"]):
        _fail("R2 repository tree/index is not clean")
    changed_lines = _git(
        repository_root,
        ["diff-tree", "--no-commit-id", "--name-status", "-r", "--no-renames", R1_AMENDMENT_COMMIT, head],
    ).decode("utf-8", "strict").splitlines()
    actual_changes = tuple(
        {"status": line.split("\t", 1)[0], "path": line.split("\t", 1)[1]}
        for line in changed_lines if line
    )
    if type(manifest.get("exact_changed_paths")) is not list or tuple(manifest["exact_changed_paths"]) != actual_changes:
        _fail("R2 changed-path allowlist differs")
    manifest_relative = manifest_path.resolve(strict=True).relative_to(repository_root.resolve(strict=True)).as_posix()
    changed_paths = {str(row["path"]) for row in actual_changes}
    bindings = manifest.get("source_bindings")
    if type(bindings) is not list or {
        str(row.get("path")) for row in bindings if isinstance(row, Mapping)
    } != changed_paths - {manifest_relative}:
        _fail("R2 source bindings do not equal changed paths minus manifest")
    verified: list[dict[str, object]] = []
    for row in bindings:
        if type(row) is not dict or set(row) != {"path", "r1_sha256_or_null", "r2_sha256"}:
            _fail("R2 source-binding row differs")
        path = row.get("path")
        old_hash = row.get("r1_sha256_or_null")
        new_hash = row.get("r2_sha256")
        if (
            type(path) is not str or not path or path.startswith("/") or ".." in Path(path).parts
            or (old_hash is not None and (type(old_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", old_hash) is None))
            or type(new_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", new_hash) is None
        ):
            _fail("R2 source-binding value differs")
        if hashlib.sha256(_git(repository_root, ["show", f"{head}:{path}"])).hexdigest() != new_hash:
            _fail(f"R2 source blob hash differs: {path}")
        if old_hash is None:
            probe = subprocess.run(
                [str(FORMAL_GIT_EXECUTABLE), "cat-file", "-e", f"{R1_AMENDMENT_COMMIT}:{path}"],
                cwd=repository_root.resolve(strict=True), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                timeout=60, env=dict(FORMAL_GIT_ENVIRONMENT_V1),
            )
            if probe.returncode == 0:
                _fail(f"R2 source unexpectedly existed in R1: {path}")
        elif hashlib.sha256(_git(repository_root, ["show", f"{R1_AMENDMENT_COMMIT}:{path}"])).hexdigest() != old_hash:
            _fail(f"R1 source blob hash differs: {path}")
        verified.append(dict(row))
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-preflight/1",
        "amendment_commit": head,
        "sole_parent_commit": R1_AMENDMENT_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 2,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "source_bindings": verified,
        "repository_clean": True,
        "exact_changed_paths_verified": True,
        "formal_identity_entropy_draw_count": 0,
        "m3_start_allowed": False,
        "fixed_audit_directory": FIXED_R2_AUDIT_DIRECTORY.as_posix(),
    }


def _load_manifest(path: Path) -> tuple[dict[str, object], bytes]:
    if path.is_symlink():
        _fail("amendment manifest may not be a symlink")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"amendment manifest is invalid JSON: {exc}")
    required = {
        "schema", "source_commit_selector", "sole_parent_commit",
        "formal_repository_commit", "fixed_run_id_hex", "fixed_ledger_id_hex",
        "recovery_attempt_ordinal", "exact_changed_paths", "source_bindings",
        "complete_seed_resume_only", "formal_identity_entropy_draw_count",
        "ephemeral_container_nonce_allowed", "ordinary_execute_allowed",
        "ordinary_recovery_cross_basis_allowed", "fixed_r1_audit_directory",
        "fixed_r2_audit_directory", "r1_audit_raw_sha256",
        "r1_failure_receipt_sha256", "expected_live_bundle_sha256",
        "fixed_continuity_sha256", "continuation_action",
        "owner_confirmation", "fixed_runtime_artifacts",
    }
    if type(value) is not dict or _canonical_json(value) != raw or set(value) != required:
        _fail("amendment manifest is not canonical exact JSON")
    if (
        value.get("schema") != MANIFEST_SCHEMA
        or value.get("source_commit_selector") != "HEAD"
        or value.get("sole_parent_commit") != R1_AMENDMENT_COMMIT
        or value.get("formal_repository_commit") != A8_BASIS_COMMIT
        or value.get("fixed_run_id_hex") != FIXED_RUN_ID_HEX
        or value.get("fixed_ledger_id_hex") != FIXED_LEDGER_ID_HEX
        or value.get("recovery_attempt_ordinal") != 2
        or value.get("complete_seed_resume_only") is not True
        or value.get("formal_identity_entropy_draw_count") != 0
        or value.get("ephemeral_container_nonce_allowed") is not True
        or value.get("ordinary_execute_allowed") is not False
        or value.get("ordinary_recovery_cross_basis_allowed") is not False
        or value.get("fixed_r1_audit_directory") != R1_AUDIT_DIRECTORY.as_posix()
        or value.get("fixed_r2_audit_directory") != FIXED_R2_AUDIT_DIRECTORY.as_posix()
        or value.get("r1_audit_raw_sha256") != R1_AUDIT_RAW_SHA256
        or value.get("r1_failure_receipt_sha256") != R1_FAILURE_RECEIPT_SHA256
        or value.get("expected_live_bundle_sha256") != EXPECTED_LIVE_BUNDLE_SHA256
        or value.get("fixed_continuity_sha256") != FIXED_CONTINUITY_SHA256
        or value.get("continuation_action")
        != "CODE_AMENDMENT_RECOVERY_CONTINUATION"
        or value.get("owner_confirmation") != OWNER_CONFIRMATION
        or type(value.get("fixed_runtime_artifacts")) is not list
        or tuple(value.get("fixed_runtime_artifacts", ())) != FIXED_RUNTIME_ARTIFACTS
    ):
        _fail("amendment manifest fixed policy differs")
    return value, raw


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _stat_row(path: Path, *, raw_seed: bool = False) -> dict[str, object]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"bound artifact is not regular: {path.name}")
    return {
        "name": path.name,
        "mode_octal": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "size_bytes": metadata.st_size,
        "st_dev": metadata.st_dev,
        "st_ino": metadata.st_ino,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "raw_seed": raw_seed,
        "raw_bytes_read": False,
        "sha256_computed": False,
    }


def _read_canonical_regular(
    path: Path, *, mode: int
) -> tuple[dict[str, object], bytes, dict[str, object]]:
    if path.is_symlink():
        _fail(f"bound JSON may not be a symlink: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        path_metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
            or metadata.st_size < 2
            or metadata.st_size > 8 * 1024 * 1024
        ):
            _fail(f"bound JSON inode/size differs: {path.name}")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _fail(f"bound JSON read was short: {path.name}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"bound JSON grew while read: {path.name}")
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    row = {
        "name": path.name,
        "mode_octal": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "size_bytes": metadata.st_size,
        "st_dev": metadata.st_dev,
        "st_ino": metadata.st_ino,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "raw_seed": False,
        "raw_bytes_read": False,
        "sha256_computed": False,
    }
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"bound JSON is invalid: {path.name}: {exc}")
    if (
        row["mode_octal"] != f"{mode:04o}"
        or type(value) is not dict
        or _canonical_json(value) != raw
    ):
        _fail(f"bound JSON bytes/mode differ: {path.name}")
    row["sha256"] = hashlib.sha256(raw).hexdigest()
    row["raw_bytes_read"] = True
    row["sha256_computed"] = True
    return value, raw, row


def _write_audit(
    path: Path, fields: Mapping[str, object]
) -> tuple[dict[str, object], bytes]:
    body = dict(fields)
    body["receipt_sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    payload = _canonical_json(body)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                _fail("audit write was short")
            offset += written
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return body, payload


def _read_canonical_audit(path: Path) -> tuple[dict[str, object], bytes]:
    value, raw, row = _read_canonical_regular(path, mode=0o600)
    del row
    body = dict(value)
    claimed = body.pop("receipt_sha256", None)
    if claimed != hashlib.sha256(_canonical_json(body)).hexdigest():
        _fail(f"audit self-hash differs: {path.name}")
    return value, raw


def _with_receipt_sha256(fields: Mapping[str, object]) -> dict[str, object]:
    result = dict(fields)
    result["receipt_sha256"] = hashlib.sha256(_canonical_json(result)).hexdigest()
    return result


def _require_existing_r2_audit_directory(
    path: Path, repository_root: Path
) -> Path:
    if path.is_symlink():
        _fail("R2 audit directory may not be a symlink")
    resolved = path.resolve(strict=True)
    if resolved != FIXED_R2_AUDIT_DIRECTORY:
        _fail("R2 audit directory differs from fixed attempt-2 path")
    repository = repository_root.resolve(strict=True)
    metadata = resolved.stat()
    if (
        resolved == repository
        or repository in resolved.parents
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
    ):
        _fail("R2 audit directory is not caller-owned repository-external mode 0700")
    return resolved


def _create_r2_audit_directory(path: Path, repository_root: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute != FIXED_R2_AUDIT_DIRECTORY:
        _fail("R2 audit directory differs from fixed attempt-2 path")
    repository = repository_root.resolve(strict=True)
    parent = absolute.parent.resolve(strict=True)
    if absolute == repository or repository in absolute.parents:
        _fail("R2 audit directory must be repository-external")
    if absolute.exists() or absolute.is_symlink():
        _fail("R2 attempt-2 audit directory already exists")
    os.mkdir(absolute, 0o700)
    os.chmod(absolute, 0o700)
    parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return _require_existing_r2_audit_directory(absolute, repository_root)


def _r1_failure_chain_snapshot_v1() -> tuple[dict[str, object], ...]:
    if R1_AUDIT_DIRECTORY.is_symlink():
        _fail("R1 audit directory may not be a symlink")
    directory_metadata = R1_AUDIT_DIRECTORY.stat()
    if (
        not stat.S_ISDIR(directory_metadata.st_mode)
        or stat.S_IMODE(directory_metadata.st_mode) != 0o700
        or directory_metadata.st_uid != os.getuid()
        or {path.name for path in R1_AUDIT_DIRECTORY.iterdir()}
        != set(R1_AUDIT_RAW_SHA256)
    ):
        _fail("R1 audit directory is not the exact terminal four-record chain")
    rows: list[dict[str, object]] = []
    records: dict[str, dict[str, object]] = {}
    for name in (
        "preflight.json",
        "authorization-request.json",
        "authorization.json",
        "failure.json",
    ):
        path = R1_AUDIT_DIRECTORY / name
        record, raw = _read_canonical_audit(path)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != R1_AUDIT_RAW_SHA256[name]:
            _fail(f"R1 audit raw hash differs: {name}")
        metadata = path.stat()
        rows.append(
            {
                "name": name,
                "raw_sha256": digest,
                "receipt_sha256": record["receipt_sha256"],
                "mode_octal": "0600",
                "size_bytes": metadata.st_size,
                "st_dev": metadata.st_dev,
                "st_ino": metadata.st_ino,
                "uid": metadata.st_uid,
                "gid": metadata.st_gid,
            }
        )
        records[name] = record
    failure = records["failure.json"]
    preflight = records["preflight.json"]
    request = records["authorization-request.json"]
    authorization = records["authorization.json"]
    if (
        set(preflight)
        != {
            "schema", "amendment_commit", "sole_parent_commit",
            "formal_repository_commit", "run_id_hex", "ledger_id_hex",
            "manifest_sha256", "source_bindings", "repository_clean",
            "exact_changed_paths_verified", "formal_identity_entropy_draw_count",
            "ephemeral_container_nonce_allowed", "m3_start_allowed",
            "fixed_audit_directory", "receipt_sha256",
        }
        or preflight.get("schema")
        != "hegel-phase3-m25-a8-recovery-audit-preflight/1"
        or preflight.get("amendment_commit") != R1_AMENDMENT_COMMIT
        or preflight.get("sole_parent_commit") != A8_BASIS_COMMIT
        or preflight.get("manifest_sha256")
        != "9ddd56e446e4c219840e4f8ba12f4ddc59ed32032a2c9347824b521ce52bd3df"
        or not isinstance(preflight.get("source_bindings"), list)
        or not preflight.get("source_bindings")
        or preflight.get("repository_clean") is not True
        or preflight.get("exact_changed_paths_verified") is not True
        or preflight.get("formal_identity_entropy_draw_count") != 0
        or preflight.get("ephemeral_container_nonce_allowed") is not True
        or preflight.get("m3_start_allowed") is not False
        or preflight.get("fixed_audit_directory") != R1_AUDIT_DIRECTORY.as_posix()
    ):
        _fail("R1 preflight receipt fields differ")
    if (
        set(request)
        != {
            "schema", "amendment_commit", "formal_repository_commit",
            "run_id_hex", "ledger_id_hex", "preflight_sha256",
            "requested_action", "ordinary_execute_allowed", "redraw_allowed",
            "abort_allowed", "poststage_recovery_allowed",
            "formal_identity_entropy_draw_count", "receipt_sha256",
        }
        or request.get("schema")
        != "hegel-phase3-m25-a8-recovery-audit-authorization-request/1"
        or request.get("amendment_commit") != R1_AMENDMENT_COMMIT
        or request.get("preflight_sha256")
        != R1_AUDIT_RAW_SHA256["preflight.json"]
        or request.get("requested_action") != "COMPLETE_ONLY_REAL_PENDING_RESUME"
        or request.get("ordinary_execute_allowed") is not False
        or request.get("redraw_allowed") is not False
        or request.get("abort_allowed") is not False
        or request.get("poststage_recovery_allowed") is not False
        or request.get("formal_identity_entropy_draw_count") != 0
    ):
        _fail("R1 authorization-request receipt fields differ")
    if (
        set(authorization)
        != {
            "schema", "amendment_commit", "formal_repository_commit",
            "run_id_hex", "ledger_id_hex", "preflight_sha256",
            "authorization_request_sha256", "authorization_actor",
            "owner_authorized_fixed_transaction_only", "ordinary_execute_invoked",
            "redraw_allowed", "abort_allowed", "poststage_recovery_allowed",
            "formal_identity_entropy_draw_count", "receipt_sha256",
        }
        or authorization.get("schema")
        != "hegel-phase3-m25-a8-recovery-audit-authorization/1"
        or authorization.get("amendment_commit") != R1_AMENDMENT_COMMIT
        or authorization.get("preflight_sha256")
        != R1_AUDIT_RAW_SHA256["preflight.json"]
        or authorization.get("authorization_request_sha256")
        != R1_AUDIT_RAW_SHA256["authorization-request.json"]
        or authorization.get("authorization_actor") != "PROJECT_OWNER"
        or authorization.get("owner_authorized_fixed_transaction_only") is not True
        or authorization.get("ordinary_execute_invoked") is not False
        or authorization.get("redraw_allowed") is not False
        or authorization.get("abort_allowed") is not False
        or authorization.get("poststage_recovery_allowed") is not False
        or authorization.get("formal_identity_entropy_draw_count") != 0
    ):
        _fail("R1 authorization receipt fields differ")
    if (
        set(failure)
        != {
            "schema", "formal_repository_commit", "run_id_hex", "ledger_id_hex",
            "failure_code", "formal_identity_entropy_draw_count",
            "raw_seed_bytes_read_by_amendment_orchestrator",
            "raw_seed_sha256_computed", "receipt_sha256",
        }
        or
        failure.get("schema") != "hegel-phase3-m25-a8-recovery-audit-failure/1"
        or failure.get("formal_repository_commit") != A8_BASIS_COMMIT
        or failure.get("run_id_hex") != FIXED_RUN_ID_HEX
        or failure.get("ledger_id_hex") != FIXED_LEDGER_ID_HEX
        or failure.get("failure_code")
        != "FAIL_M25_FORMAL_CEREMONY_LOCKED_OR_RESERVED"
        or failure.get("formal_identity_entropy_draw_count") != 0
        or failure.get("raw_seed_bytes_read_by_amendment_orchestrator") is not False
        or failure.get("raw_seed_sha256_computed") is not False
        or failure.get("receipt_sha256") != R1_FAILURE_RECEIPT_SHA256
    ):
        _fail("R1 failure receipt fields differ")
    for record in records.values():
        if (
            record.get("formal_repository_commit") != A8_BASIS_COMMIT
            or record.get("run_id_hex") != FIXED_RUN_ID_HEX
            or record.get("ledger_id_hex") != FIXED_LEDGER_ID_HEX
        ):
            _fail("R1 audit chain transaction identity differs")
    return tuple(rows)


def _sequence_diagnostics(live: object, restored: object) -> dict[str, int]:
    counts = {
        "sequence_representation_mismatch_count": 0,
        "mapping_key_mismatch_count": 0,
        "sequence_length_mismatch_count": 0,
        "scalar_value_mismatch_count": 0,
        "other_type_mismatch_count": 0,
    }

    def visit(left: object, right: object) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if set(left) != set(right):
                counts["mapping_key_mismatch_count"] += 1
            for key in set(left) & set(right):
                visit(left[key], right[key])
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if type(left) is not type(right):
                counts["sequence_representation_mismatch_count"] += 1
            if len(left) != len(right):
                counts["sequence_length_mismatch_count"] += 1
            for left_item, right_item in zip(left, right):
                visit(left_item, right_item)
            return
        if type(left) is not type(right):
            counts["other_type_mismatch_count"] += 1
        elif left != right:
            counts["scalar_value_mismatch_count"] += 1

    visit(live, restored)
    return counts


def _seed_prefix_stat_only_snapshot(custody: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    intent, intent_raw, intent_row = _read_canonical_regular(
        custody / "split_seed_generation.intent", mode=0o600
    )
    if hashlib.sha256(intent_raw).hexdigest() != FIXED_CONTINUITY_SHA256[
        "split_seed_generation.intent"
    ]:
        _fail("seed-generation intent continuity hash differs")
    if intent != {
        "schema": "hegel-phase3-m25-seed-generation-intent/1",
        "state": "CSPRNG_CALL_COMMITTED_NO_REDRAW",
    }:
        _fail("seed-generation intent differs")
    rows.append(intent_row)

    seed_path = custody / "split_master_seed.bin"
    seed_row = _stat_row(seed_path, raw_seed=True)
    if seed_row["mode_octal"] != "0600" or seed_row["size_bytes"] != 32:
        _fail("raw seed metadata differs; bytes were not read")
    rows.append(seed_row)

    completion, _completion_raw, completion_row = _read_canonical_regular(
        custody / "split_seed_generation.complete", mode=0o600
    )
    if completion_row.get("sha256") != FIXED_CONTINUITY_SHA256[
        "split_seed_generation.complete"
    ]:
        _fail("seed-generation completion continuity hash differs")
    if (
        set(completion)
        != {
            "attempt", "intent_sha256", "schema", "seed_commitment_hex",
            "seed_length_bytes",
        }
        or completion.get("attempt") != 1
        or completion.get("intent_sha256") != hashlib.sha256(intent_raw).hexdigest()
        or completion.get("schema")
        != "hegel-phase3-m25-seed-generation-complete/1"
        or type(completion.get("seed_commitment_hex")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", str(completion["seed_commitment_hex"])) is None
        or completion.get("seed_length_bytes") != 32
    ):
        _fail("seed-generation completion differs")
    rows.append(completion_row)
    return tuple(rows)


def _probe_formal_lock_available_v1(lock_path: Path) -> bool:
    descriptor = os.open(lock_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("fixed formal transaction lock has a live holder")
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(descriptor)
    return True


def _docker_read_only_state_v1() -> dict[str, object]:
    executable = Path("/usr/bin/docker")
    if (
        not executable.is_file()
        or executable.is_symlink()
        or executable.resolve(strict=True) != executable
    ):
        _fail("fixed Docker executable is unavailable")
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
    }
    docker_override_keys = (
        "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"
    )
    if any(key in os.environ for key in docker_override_keys):
        _fail("Docker endpoint override environment is forbidden for R2")

    def run(arguments: Sequence[str]) -> tuple[str, ...]:
        completed = subprocess.run(
            [str(executable), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
            env=environment,
        )
        if completed.returncode != 0:
            _fail(
                "read-only Docker state probe failed: "
                + completed.stderr.decode("utf-8", "replace")[-300:]
            )
        return tuple(
            line for line in completed.stdout.decode("utf-8", "strict").splitlines()
            if line
        )

    if run(("context", "show")) != ("default",):
        _fail("R2 Docker probe requires the local default context")
    endpoint_lines = run((
        "context", "inspect", "default", "--format",
        "{{json .Endpoints.docker.Host}}",
    ))
    if endpoint_lines != ('"unix:///var/run/docker.sock"',):
        _fail("R2 Docker probe requires the fixed local Unix socket")

    volume_prefix = f"hegel-m25-state-{FIXED_RUN_ID_HEX}-p"
    observed_volumes = tuple(
        sorted(name for name in run(("volume", "ls", "--format", "{{.Name}}"))
               if name.startswith(volume_prefix))
    )
    expected_volumes = tuple(f"{volume_prefix}{purpose}" for purpose in (1, 2, 3, 4))
    if observed_volumes != expected_volumes:
        _fail("fixed four Docker key volumes are not exactly present")
    volume_label_rows: list[dict[str, object]] = []
    for purpose, name in zip((1, 2, 3, 4), expected_volumes):
        label_lines = run(("volume", "inspect", name, "--format", "{{json .Labels}}"))
        if len(label_lines) != 1:
            _fail("fixed Docker key volume inspect result differs")
        try:
            labels = json.loads(label_lines[0])
        except json.JSONDecodeError as exc:
            _fail(f"fixed Docker key volume labels are invalid: {exc}")
        if (
            type(labels) is not dict
            or labels.get("hegel.m25.run") != FIXED_RUN_ID_HEX
            or labels.get("hegel.m25.purpose") != str(purpose)
            or labels.get("hegel.m25.basis") != A8_BASIS_COMMIT
            or labels.get("hegel.m25.state") != "true"
        ):
            _fail("fixed Docker key volume labels differ")
        volume_label_rows.append(
            {
                "volume_name": name,
                "purpose_id": purpose,
                "run_id_hex": labels["hegel.m25.run"],
                "basis_commit": labels["hegel.m25.basis"],
                "state_label": labels["hegel.m25.state"],
            }
        )
    run_labelled_containers = tuple(sorted(run((
        "ps", "-a", "--filter", f"label=hegel.m25.run={FIXED_RUN_ID_HEX}",
        "--format", "{{.ID}}",
    ))))
    if run_labelled_containers:
        _fail("fixed run has a surviving labelled Docker container")
    return {
        "fixed_key_volume_names": observed_volumes,
        "fixed_key_volume_label_rows": tuple(volume_label_rows),
        "fixed_key_volume_count": 4,
        "run_labelled_container_names": run_labelled_containers,
        "run_labelled_container_count": 0,
        "docker_context": "default",
        "docker_endpoint": "unix:///var/run/docker.sock",
        "probe_read_only": True,
        "network_operation_invoked": False,
    }


def _build_incident_diagnostic_v1(
    *,
    custody_directory: Path,
    public_evidence_path: Path,
    public_promotion_path: Path,
) -> dict[str, object]:
    custody = custody_directory.resolve(strict=True)
    evidence = Path(os.path.abspath(os.fspath(public_evidence_path)))
    promotion = Path(os.path.abspath(os.fspath(public_promotion_path)))
    if evidence.parent.resolve(strict=True) != evidence.parent or promotion.parent != evidence.parent:
        _fail("fixed public output paths differ or contain a symlinked parent")
    stage = evidence.parent / f".hegel-m25-stage-{FIXED_RUN_ID_HEX}"
    if stage.is_symlink() or not stage.is_dir() or stat.S_IMODE(stage.stat().st_mode) != 0o700:
        _fail("fixed PENDING stage differs")
    if {path.name for path in stage.iterdir()} != FIXED_STAGE_INVENTORY:
        _fail("fixed PENDING stage inventory differs or contains a next/unknown file")
    expected_custody_inventory = {
        "phase3_m25_ceremony.lock",
        f"opaque-run-{FIXED_RUN_ID_HEX}.reserved",
        f"opaque-ledger-{FIXED_LEDGER_ID_HEX}.reserved",
        "split_seed_instantiation.marker",
        "split_seed_generation.intent",
        "split_master_seed.bin",
        "split_seed_generation.complete",
    }
    if {path.name for path in custody.iterdir()} != expected_custody_inventory:
        _fail("fixed custody inventory differs")

    r1_rows = _r1_failure_chain_snapshot_v1()
    lock, lock_raw, lock_row = _read_canonical_regular(
        custody / "phase3_m25_ceremony.lock", mode=0o600
    )
    if (
        hashlib.sha256(lock_raw).hexdigest()
        != FIXED_CONTINUITY_SHA256["phase3_m25_ceremony.lock"]
        or lock.get("schema") != "hegel-phase3-m25-persistent-ceremony-lock/4"
        or lock.get("basis_commit") != A8_BASIS_COMMIT
        or lock.get("run_id_hex") != FIXED_RUN_ID_HEX
        or lock.get("ledger_id_hex") != FIXED_LEDGER_ID_HEX
        or lock.get("custody_directory") != custody.as_posix()
        or lock.get("public_evidence_path") != evidence.as_posix()
        or lock.get("public_promotion_path") != promotion.as_posix()
        or lock.get("stage_directory_name") != stage.name
    ):
        _fail("persistent lock does not bind the fixed A8 transaction paths")
    expected_intent_sha = lock.get("prestage_intent_sha256_or_null")
    if type(expected_intent_sha) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_intent_sha) is None:
        _fail("persistent lock prestage digest differs")
    intent_transport, intent_raw, intent_row = _read_canonical_regular(
        stage / "prestage-intent.json", mode=0o600
    )
    if (
        hashlib.sha256(intent_raw).hexdigest() != expected_intent_sha
        or hashlib.sha256(intent_raw).hexdigest()
        != FIXED_CONTINUITY_SHA256["prestage-intent.json"]
    ):
        _fail("prestage intent differs from persistent lock")
    try:
        restored_intent = _restore(dict(intent_transport))
    except (TypeError, ValueError) as exc:
        _fail(f"prestage intent restore failed: {exc}")
    if type(restored_intent) is not dict:
        _fail("restored prestage intent is not a mapping")
    runtime_bindings = restored_intent.get("runtime_binding_fields")
    if (
        type(runtime_bindings) is not dict
        or runtime_bindings.get("formal_rust_replay_binary_path")
        != FIXED_FORMAL_RUST_BINARY.as_posix()
        or runtime_bindings.get("formal_rust_replay_binary_sha256")
        != bytes.fromhex(FIXED_FORMAL_RUST_BINARY_SHA256)
        or runtime_bindings.get("rust_bridge_dag_replay_binary_path")
        != FIXED_BRIDGE_RUST_BINARY.as_posix()
        or runtime_bindings.get("rust_bridge_dag_replay_binary_sha256")
        != bytes.fromhex(FIXED_BRIDGE_RUST_BINARY_SHA256)
        or runtime_bindings.get("rust_bridge_dag_qualification_report_sha256")
        != bytes.fromhex(FIXED_BRIDGE_REPORT_DIAGNOSTIC_SHA256)
    ):
        _fail("prestage intent runtime bindings differ from fixed A8 artifacts")
    runtime_artifact_bindings = {
        "formal_rust_replay_binary_path": FIXED_FORMAL_RUST_BINARY.as_posix(),
        "formal_rust_replay_binary_sha256": FIXED_FORMAL_RUST_BINARY_SHA256,
        "rust_bridge_dag_replay_binary_path": FIXED_BRIDGE_RUST_BINARY.as_posix(),
        "rust_bridge_dag_replay_binary_sha256": FIXED_BRIDGE_RUST_BINARY_SHA256,
        "rust_bridge_dag_qualification_report_path": FIXED_BRIDGE_REPORT.as_posix(),
        "rust_bridge_dag_qualification_report_raw_sha256": (
            FIXED_BRIDGE_REPORT_RAW_SHA256
        ),
        "rust_bridge_dag_qualification_report_diagnostic_sha256": (
            FIXED_BRIDGE_REPORT_DIAGNOSTIC_SHA256
        ),
    }
    expected_bundle = restored_intent.get("live_actor_protocol_qualification_bundle")
    expected_bundle_sha = restored_intent.get(
        "live_actor_protocol_qualification_bundle_sha256"
    )
    live_bundle, live_raw, live_row = _read_canonical_regular(
        stage / "live-qualification-bundle.json", mode=0o600
    )
    diagnostics = _sequence_diagnostics(live_bundle, expected_bundle)
    if (
        type(expected_bundle) is not dict
        or type(expected_bundle_sha) is not bytes
        or len(expected_bundle_sha) != 32
        or live_bundle == expected_bundle
        or live_bundle != _transport(expected_bundle)
        or live_raw != _executor_canonical_json(expected_bundle)
        or hashlib.sha256(live_raw).hexdigest() != EXPECTED_LIVE_BUNDLE_SHA256
        or expected_bundle_sha.hex() != EXPECTED_LIVE_BUNDLE_SHA256
        or diagnostics
        != {
            "sequence_representation_mismatch_count": 208,
            "mapping_key_mismatch_count": 0,
            "sequence_length_mismatch_count": 0,
            "scalar_value_mismatch_count": 0,
            "other_type_mismatch_count": 0,
        }
    ):
        _fail("incident is not the frozen list/tuple-only transport mismatch")
    marker = read_marker_snapshot_v1(custody / "split_seed_instantiation.marker")
    marker_fields, marker_raw, marker_row = _read_canonical_regular(
        custody / "split_seed_instantiation.marker", mode=0o600
    )
    journal, journal_raw, journal_row = _read_canonical_regular(
        stage / "transaction-journal.json", mode=0o600
    )
    if (
        marker.state != "PENDING"
        or hashlib.sha256(marker_raw).hexdigest()
        != FIXED_CONTINUITY_SHA256["split_seed_instantiation.marker"]
        or marker_fields.get("split_version_digest_hex")
        != FIXED_SPLIT_VERSION_DIGEST_HEX
        or marker_fields.get("custodian_key_id_hex")
        != FIXED_PENDING_CUSTODIAN_KEY_ID_HEX
        or marker_fields.get("seed_commitment_manifest_root_hex_or_null") is not None
        or hashlib.sha256(journal_raw).hexdigest()
        != FIXED_CONTINUITY_SHA256["transaction-journal.json"]
        or journal.get("basis_commit") != A8_BASIS_COMMIT
        or journal.get("run_id_hex") != FIXED_RUN_ID_HEX
        or journal.get("ledger_id_hex") != FIXED_LEDGER_ID_HEX
        or journal.get("state") != "RESERVED"
        or journal.get("marker_complete") is not False
        or journal.get("actors_absent") is not False
        or journal.get("public_outputs_complete") is not False
    ):
        _fail("incident transaction is not PENDING / RESERVED")
    bound_stage_rows: list[dict[str, object]] = []
    for name in (
        "actor-trust-checkpoint.json",
        "recovery-anchor.json",
        "recovery-anchor.ready.json",
    ):
        _value, raw, row = _read_canonical_regular(stage / name, mode=0o600)
        if hashlib.sha256(raw).hexdigest() != FIXED_CONTINUITY_SHA256[name]:
            _fail(f"fixed stage continuity hash differs: {name}")
        bound_stage_rows.append(row)
    reservation_rows: list[dict[str, object]] = []
    for output in (
        evidence,
        promotion,
        promotion.with_name(promotion.name + ".publication-receipt.json"),
    ):
        if output.exists() or output.is_symlink():
            _fail("incident unexpectedly has a final public output")
        reservation = output.with_name(f".{output.name}.hegel-reserved")
        _reservation_fields, reservation_raw, reservation_row = (
            _read_canonical_regular(reservation, mode=0o600)
        )
        if hashlib.sha256(reservation_raw).hexdigest() != FIXED_CONTINUITY_SHA256[
            reservation.name
        ]:
            _fail("incident public reservation continuity hash differs")
        reservation_rows.append(reservation_row)
    lock_available = _probe_formal_lock_available_v1(
        custody / "phase3_m25_ceremony.lock"
    )
    docker_state = _docker_read_only_state_v1()
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-incident-diagnostic/1",
        "formal_repository_commit": A8_BASIS_COMMIT,
        "r1_amendment_commit": R1_AMENDMENT_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 2,
        "continuation_action": "CODE_AMENDMENT_RECOVERY_CONTINUATION",
        "custody_directory": custody.as_posix(),
        "public_evidence_path": evidence.as_posix(),
        "public_promotion_path": promotion.as_posix(),
        "stage_directory": stage.as_posix(),
        "r1_failure_chain": r1_rows,
        "r1_failure_receipt_sha256": R1_FAILURE_RECEIPT_SHA256,
        "persistent_lock_sha256": hashlib.sha256(lock_raw).hexdigest(),
        "persistent_lock_metadata": lock_row,
        "formal_lock_has_no_live_holder": lock_available,
        "prestage_intent_sha256": hashlib.sha256(intent_raw).hexdigest(),
        "prestage_intent_metadata": intent_row,
        "runtime_artifact_bindings": runtime_artifact_bindings,
        "standalone_live_bundle_sha256": hashlib.sha256(live_raw).hexdigest(),
        "embedded_live_bundle_canonical_sha256": hashlib.sha256(
            _executor_canonical_json(expected_bundle)
        ).hexdigest(),
        "live_bundle_metadata": live_row,
        "old_direct_python_equality": False,
        "transport_domain_equality": True,
        "canonical_payload_exact_equality": True,
        "transport_mismatch_diagnostics": diagnostics,
        "marker_state": marker.state,
        "marker_created_at_unix_seconds": marker.created_at_unix_seconds,
        "marker_sha256": hashlib.sha256(marker_raw).hexdigest(),
        "marker_metadata": marker_row,
        "marker_split_version_digest_hex": FIXED_SPLIT_VERSION_DIGEST_HEX,
        "marker_custodian_key_id_hex": FIXED_PENDING_CUSTODIAN_KEY_ID_HEX,
        "marker_seed_commitment_manifest_root_hex_or_null": None,
        "journal_state": journal["state"],
        "journal_sha256": hashlib.sha256(journal_raw).hexdigest(),
        "journal_metadata": journal_row,
        "fixed_stage_inventory": tuple(sorted(FIXED_STAGE_INVENTORY)),
        "additional_stage_continuity_metadata": tuple(bound_stage_rows),
        "public_reservation_metadata": tuple(reservation_rows),
        "docker_state": docker_state,
        "seed_prefix_metadata": _seed_prefix_stat_only_snapshot(custody),
        "raw_seed_bytes_read_by_r2_orchestrator": False,
        "raw_seed_sha256_computed": False,
        "formal_identity_entropy_draw_count": 0,
        "ordinary_execute_allowed": False,
        "redraw_allowed": False,
        "abort_allowed": False,
        "poststage_recovery_allowed": False,
        "m3_start_allowed": False,
    }


def _authorization_request_fields(
    *, amendment_commit: object, preflight_raw: bytes, incident_raw: bytes
) -> dict[str, object]:
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-authorization-request/1",
        "amendment_commit": amendment_commit,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 2,
        "continuation_action": "CODE_AMENDMENT_RECOVERY_CONTINUATION",
        "preflight_sha256": hashlib.sha256(preflight_raw).hexdigest(),
        "incident_diagnostic_sha256": hashlib.sha256(incident_raw).hexdigest(),
        "r1_failure_raw_sha256": R1_AUDIT_RAW_SHA256["failure.json"],
        "r1_failure_receipt_sha256": R1_FAILURE_RECEIPT_SHA256,
        "requested_action": "COMPLETE_ONLY_REAL_PENDING_RESUME",
        "ordinary_execute_allowed": False,
        "redraw_allowed": False,
        "abort_allowed": False,
        "poststage_recovery_allowed": False,
        "formal_identity_entropy_draw_count": 0,
    }


def prepare_fixed_a8_r2_authorization_v1(
    *,
    audit_directory: Path,
    custody_directory: Path,
    public_evidence_path: Path,
    public_promotion_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, object]:
    """Create the fresh attempt-2 audit and write preflight/incident/request."""

    preflight = inspect_r2_source_preflight_v1(
        repository_root=repository_root, manifest_path=manifest_path
    )
    incident = _build_incident_diagnostic_v1(
        custody_directory=custody_directory,
        public_evidence_path=public_evidence_path,
        public_promotion_path=public_promotion_path,
    )
    audit = _create_r2_audit_directory(audit_directory, repository_root)
    preflight_record, preflight_raw = _write_audit(audit / "preflight.json", preflight)
    _incident_record, incident_raw = _write_audit(
        audit / "incident-diagnostic.json", incident
    )
    request, _request_raw = _write_audit(
        audit / "authorization-request.json",
        _authorization_request_fields(
            amendment_commit=preflight_record["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
        ),
    )
    return request


def write_fixed_a8_r2_owner_authorization_v1(
    *,
    audit_directory: Path,
    owner_confirmation: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, object]:
    """Consume a separate explicit owner action for fixed attempt ordinal 2."""

    if owner_confirmation != OWNER_CONFIRMATION:
        _fail("owner confirmation phrase differs")
    audit = _require_existing_r2_audit_directory(audit_directory, repository_root)
    if {path.name for path in audit.iterdir()} != {
        "preflight.json", "incident-diagnostic.json", "authorization-request.json"
    }:
        _fail("R2 pre-authorization audit path set differs")
    preflight, preflight_raw = _read_canonical_audit(audit / "preflight.json")
    _incident, incident_raw = _read_canonical_audit(
        audit / "incident-diagnostic.json"
    )
    request, request_raw = _read_canonical_audit(audit / "authorization-request.json")
    if request != _with_receipt_sha256(
        _authorization_request_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
        )
    ):
        _fail("attempt-2 authorization request differs")
    authorization, _raw = _write_audit(
        audit / "authorization.json",
        {
            "schema": f"{AUDIT_SCHEMA_PREFIX}-authorization/1",
            "amendment_commit": preflight["amendment_commit"],
            "formal_repository_commit": A8_BASIS_COMMIT,
            "run_id_hex": FIXED_RUN_ID_HEX,
            "ledger_id_hex": FIXED_LEDGER_ID_HEX,
            "recovery_attempt_ordinal": 2,
            "continuation_action": "CODE_AMENDMENT_RECOVERY_CONTINUATION",
            "preflight_sha256": hashlib.sha256(preflight_raw).hexdigest(),
            "incident_diagnostic_sha256": hashlib.sha256(incident_raw).hexdigest(),
            "authorization_request_sha256": hashlib.sha256(request_raw).hexdigest(),
            "r1_failure_raw_sha256": R1_AUDIT_RAW_SHA256["failure.json"],
            "authorization_actor": "PROJECT_OWNER",
            "owner_authorized_fixed_transaction_only": True,
            "ordinary_execute_invoked": False,
            "redraw_allowed": False,
            "abort_allowed": False,
            "poststage_recovery_allowed": False,
            "formal_identity_entropy_draw_count": 0,
        },
    )
    return authorization


def _expected_authorization_fields(
    *,
    amendment_commit: object,
    preflight_raw: bytes,
    incident_raw: bytes,
    request_raw: bytes,
) -> dict[str, object]:
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-authorization/1",
        "amendment_commit": amendment_commit,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "recovery_attempt_ordinal": 2,
        "continuation_action": "CODE_AMENDMENT_RECOVERY_CONTINUATION",
        "preflight_sha256": hashlib.sha256(preflight_raw).hexdigest(),
        "incident_diagnostic_sha256": hashlib.sha256(incident_raw).hexdigest(),
        "authorization_request_sha256": hashlib.sha256(request_raw).hexdigest(),
        "r1_failure_raw_sha256": R1_AUDIT_RAW_SHA256["failure.json"],
        "authorization_actor": "PROJECT_OWNER",
        "owner_authorized_fixed_transaction_only": True,
        "ordinary_execute_invoked": False,
        "redraw_allowed": False,
        "abort_allowed": False,
        "poststage_recovery_allowed": False,
        "formal_identity_entropy_draw_count": 0,
    }


def _hash_fixed_runtime_binary_v1(
    path: Path, *, expected_path: Path, expected_sha256: str
) -> dict[str, object]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if (
        absolute != expected_path
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != expected_path
    ):
        _fail("runtime binary path differs from the fixed main-A8 artifact")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    hasher = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        path_metadata = absolute.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o755
            or (metadata.st_dev, metadata.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
            or metadata.st_size <= 0
        ):
            _fail("fixed runtime binary inode/mode differs")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    finally:
        os.close(descriptor)
    if hasher.hexdigest() != expected_sha256:
        _fail("fixed runtime binary SHA-256 differs")
    return {
        "path": absolute.as_posix(),
        "sha256": expected_sha256,
        "mode_octal": "0755",
        "size_bytes": metadata.st_size,
        "st_dev": metadata.st_dev,
        "st_ino": metadata.st_ino,
    }


def _validate_runtime_artifacts_before_attempt_v1(
    *,
    rust_formal_replay_binary: Path,
    rust_bridge_dag_replay_binary: Path,
    rust_bridge_dag_qualification_report: Path,
    expected_bindings: object,
) -> tuple[dict[str, object], ...]:
    frozen_bindings = {
        "formal_rust_replay_binary_path": FIXED_FORMAL_RUST_BINARY.as_posix(),
        "formal_rust_replay_binary_sha256": FIXED_FORMAL_RUST_BINARY_SHA256,
        "rust_bridge_dag_replay_binary_path": FIXED_BRIDGE_RUST_BINARY.as_posix(),
        "rust_bridge_dag_replay_binary_sha256": FIXED_BRIDGE_RUST_BINARY_SHA256,
        "rust_bridge_dag_qualification_report_path": FIXED_BRIDGE_REPORT.as_posix(),
        "rust_bridge_dag_qualification_report_raw_sha256": (
            FIXED_BRIDGE_REPORT_RAW_SHA256
        ),
        "rust_bridge_dag_qualification_report_diagnostic_sha256": (
            FIXED_BRIDGE_REPORT_DIAGNOSTIC_SHA256
        ),
    }
    if expected_bindings != frozen_bindings:
        _fail("stored incident runtime bindings differ")
    rows = [
        _hash_fixed_runtime_binary_v1(
            rust_formal_replay_binary,
            expected_path=FIXED_FORMAL_RUST_BINARY,
            expected_sha256=FIXED_FORMAL_RUST_BINARY_SHA256,
        ),
        _hash_fixed_runtime_binary_v1(
            rust_bridge_dag_replay_binary,
            expected_path=FIXED_BRIDGE_RUST_BINARY,
            expected_sha256=FIXED_BRIDGE_RUST_BINARY_SHA256,
        ),
    ]
    report_absolute = Path(os.path.abspath(os.fspath(rust_bridge_dag_qualification_report)))
    if (
        report_absolute != FIXED_BRIDGE_REPORT
        or report_absolute.is_symlink()
        or report_absolute.resolve(strict=True) != FIXED_BRIDGE_REPORT
    ):
        _fail("bridge qualification report path differs from fixed main-A8 artifact")
    report, report_raw, report_row = _read_canonical_regular(
        report_absolute, mode=0o644
    )
    if (
        hashlib.sha256(report_raw).hexdigest() != FIXED_BRIDGE_REPORT_RAW_SHA256
        or report.get("diagnostic_report_sha256")
        != "sha256:" + FIXED_BRIDGE_REPORT_DIAGNOSTIC_SHA256
    ):
        _fail("bridge qualification report identity differs")
    report_row["path"] = report_absolute.as_posix()
    rows.append(report_row)
    return tuple(rows)


def _validate_final_publication_v1(
    *,
    payload: dict[str, object],
    promotion: dict[str, object],
    custody_directory: Path,
    stage_directory: Path,
    public_evidence_path: Path,
    public_promotion_path: Path,
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
    if evidence_raw != _executor_canonical_json(payload):
        _fail("final public evidence bytes differ from returned payload")
    replayed_promotion = replay_public_gate_evidence_v1(payload)
    if promotion != replayed_promotion or promotion_raw != _executor_canonical_json(
        replayed_promotion
    ):
        _fail("final public replay differs from promotion bytes")
    gate_report = replayed_promotion.get("gate_report")
    if (
        not isinstance(gate_report, Mapping)
        or gate_report.get("gates_after") != 24
        or gate_report.get("all_gates_15_24_passed") is not True
        or gate_report.get("child_state") != "NOT_RUN"
        or gate_report.get("m3_run_started") is not False
    ):
        _fail("final public promotion is not replayed 24/24 NOT_RUN")
    inputs = load_gate_evidence_inputs_v1(payload)
    if (
        len(M3_RUN_OUTPUT_ROOTS) != 15
        or any(
            inputs.run_genesis_fields.get(f"{name}_or_null") is not None
            for name in M3_RUN_OUTPUT_ROOTS
        )
    ):
        _fail("final M3 genesis does not retain all 15 output roots null")
    complete_marker = read_marker_snapshot_v1(
        custody_directory / "split_seed_instantiation.marker"
    )
    if (
        complete_marker.state != "COMPLETE"
        or complete_marker != inputs.marker_snapshot
        or complete_marker.seed_commitment_manifest_root is None
        or complete_marker.custodian_key_id != inputs.marker_snapshot.custodian_key_id
    ):
        _fail("official COMPLETE marker differs from final replay inputs")
    try:
        publication_receipt = json.loads(receipt_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"publication receipt is invalid: {exc}")
    expected_receipt_fields = {
        "schema", "basis_commit", "run_id_hex", "ledger_id_hex",
        "public_evidence_sha256", "public_promotion_sha256",
        "seed_custody_verification_receipt_sha256_or_null",
        "prospective_public_replay_passed", "marker_was_complete_during_staging",
        "actor_cleanup_required_before_publication", "authority_disclosure",
        "contains_private_key", "contains_raw_split_seed",
        "contains_split_assignment_rows",
    }
    if (
        type(publication_receipt) is not dict
        or set(publication_receipt) != expected_receipt_fields
        or _canonical_json(publication_receipt) != receipt_raw
        or publication_receipt.get("schema")
        != "hegel-phase3-m25-publication-receipt/1"
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
        or type(
            publication_receipt.get(
                "seed_custody_verification_receipt_sha256_or_null"
            )
        ) is not str
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(
                publication_receipt.get(
                    "seed_custody_verification_receipt_sha256_or_null"
                )
            ),
        ) is None
        or any(
            publication_receipt.get(name) is not False
            for name in (
                "contains_private_key", "contains_raw_split_seed",
                "contains_split_assignment_rows",
            )
        )
    ):
        _fail("official publication receipt differs from final public bytes")
    assert_public_payload_contains_no_secret_fields(publication_receipt)
    if publication_receipt[
        "seed_custody_verification_receipt_sha256_or_null"
    ] != hashlib.sha256(seed_verification_raw).hexdigest():
        _fail("publication receipt does not bind durable seed-custody verification")
    return {
        "public_evidence_sha256": hashlib.sha256(evidence_raw).hexdigest(),
        "public_promotion_sha256": hashlib.sha256(promotion_raw).hexdigest(),
        "publication_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "seed_custody_verification_receipt_sha256": hashlib.sha256(
            seed_verification_raw
        ).hexdigest(),
        "complete_marker_seed_commitment_manifest_root_hex": (
            complete_marker.seed_commitment_manifest_root.hex()
        ),
        "complete_marker_custodian_key_id_hex": complete_marker.custodian_key_id.hex(),
    }


def execute_fixed_a8_r2_recovery_v1(
    *,
    custody_directory: Path,
    rust_formal_replay_binary: Path,
    rust_bridge_dag_replay_binary: Path,
    rust_bridge_dag_qualification_report: Path,
    public_evidence_path: Path,
    public_promotion_path: Path,
    audit_directory: Path,
    repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> tuple[dict[str, object], dict[str, object]]:
    """Consume attempt-2 once and resume only the fixed complete-seed transaction."""

    audit = _require_existing_r2_audit_directory(audit_directory, repository_root)
    actual_before = {path.name for path in audit.iterdir()}
    if "attempt-start.json" in actual_before:
        _fail("R2 attempt-2 was already consumed and may never be invoked again")
    if actual_before != {
        "preflight.json", "incident-diagnostic.json",
        "authorization-request.json", "authorization.json",
    }:
        _fail("R2 pre-attempt audit path set differs")
    custody = custody_directory.resolve(strict=True)
    evidence_parent = public_evidence_path.parent.resolve(strict=True)
    promotion_parent = public_promotion_path.parent.resolve(strict=True)
    if any(
        _paths_overlap(audit, path)
        for path in (custody, evidence_parent, promotion_parent, R1_AUDIT_DIRECTORY)
    ):
        _fail("R2 audit overlaps custody, public output, or R1 audit storage")

    preflight, preflight_raw = _read_canonical_audit(audit / "preflight.json")
    current_preflight = inspect_r2_source_preflight_v1(
        repository_root=repository_root, manifest_path=manifest_path
    )
    if preflight_raw != _canonical_json(_with_receipt_sha256(current_preflight)):
        _fail("stored preflight differs from current clean R2 source")
    incident, incident_raw = _read_canonical_audit(
        audit / "incident-diagnostic.json"
    )
    current_incident = _build_incident_diagnostic_v1(
        custody_directory=custody_directory,
        public_evidence_path=public_evidence_path,
        public_promotion_path=public_promotion_path,
    )
    if incident_raw != _canonical_json(_with_receipt_sha256(current_incident)):
        _fail("stored incident differs from the current immutable transaction")
    request, request_raw = _read_canonical_audit(
        audit / "authorization-request.json"
    )
    if request != _with_receipt_sha256(
        _authorization_request_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
        )
    ):
        _fail("attempt-2 authorization request differs")
    authorization, authorization_raw = _read_canonical_audit(
        audit / "authorization.json"
    )
    if authorization != _with_receipt_sha256(
        _expected_authorization_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
            incident_raw=incident_raw,
            request_raw=request_raw,
        )
    ):
        _fail("independent attempt-2 authorization differs")

    runtime_artifact_metadata = _validate_runtime_artifacts_before_attempt_v1(
        rust_formal_replay_binary=rust_formal_replay_binary,
        rust_bridge_dag_replay_binary=rust_bridge_dag_replay_binary,
        rust_bridge_dag_qualification_report=rust_bridge_dag_qualification_report,
        expected_bindings=incident.get("runtime_artifact_bindings"),
    )

    attempt_start, attempt_start_raw = _write_audit(
        audit / "attempt-start.json",
        {
            "schema": f"{AUDIT_SCHEMA_PREFIX}-attempt-start/1",
            "amendment_commit": preflight["amendment_commit"],
            "formal_repository_commit": A8_BASIS_COMMIT,
            "run_id_hex": FIXED_RUN_ID_HEX,
            "ledger_id_hex": FIXED_LEDGER_ID_HEX,
            "recovery_attempt_ordinal": 2,
            "continuation_action": "CODE_AMENDMENT_RECOVERY_CONTINUATION",
            "preflight_sha256": hashlib.sha256(preflight_raw).hexdigest(),
            "incident_diagnostic_sha256": hashlib.sha256(incident_raw).hexdigest(),
            "authorization_request_sha256": hashlib.sha256(request_raw).hexdigest(),
            "authorization_sha256": hashlib.sha256(authorization_raw).hexdigest(),
            "r1_failure_raw_sha256": R1_AUDIT_RAW_SHA256["failure.json"],
            "r1_failure_receipt_sha256": R1_FAILURE_RECEIPT_SHA256,
            "runtime_artifact_metadata": runtime_artifact_metadata,
            "accepted_worker_mode": "REAL_PENDING_RESUME",
            "complete_seed_resume_only": True,
            "ordinary_execute_invoked": False,
            "formal_identity_entropy_draw_count": 0,
            "m3_start_invoked": False,
        },
    )
    del attempt_start

    actors: A8R1RecoveryDockerActorsV1 | None = None
    admission_raw: bytes | None = None
    failure_phase = "ACTOR_CONSTRUCTION"
    try:
        actors = A8R1RecoveryDockerActorsV1(
            basis_commit=A8_BASIS_COMMIT,
            custody_directory=custody_directory,
            rust_formal_replay_binary=rust_formal_replay_binary,
            rust_bridge_dag_replay_binary=rust_bridge_dag_replay_binary,
            rust_bridge_dag_qualification_report=rust_bridge_dag_qualification_report,
            timestamp=0,
        )
        failure_phase = "FORMAL_RECOVERY_ACQUIRE"
        with acquire_pending_ceremony_recovery_v1(
            custody_directory=custody_directory,
            public_evidence_path=public_evidence_path,
            public_promotion_path=public_promotion_path,
            actors=actors,
        ) as recovery:
            if _paths_overlap(audit, recovery.stage_directory.resolve(strict=True)):
                _fail("R2 audit overlaps the reserved formal stage")
            if (
                recovery.basis_commit != A8_BASIS_COMMIT
                or recovery.run_id != FIXED_RUN_ID
                or recovery.ledger_id != FIXED_LEDGER_ID
                or recovery.marker_snapshot.state != "PENDING"
                or recovery.journal_state != "RESERVED"
            ):
                _fail("acquired recovery is not the fixed A8 PENDING/RESERVED transaction")
            seed_metadata = _seed_prefix_stat_only_snapshot(
                recovery.custody_directory
            )
            actors.timestamp = recovery.marker_snapshot.created_at_unix_seconds
            failure_phase = "SOURCE_ADMISSION"

            def build_source_admission(
                candidate: PendingCeremonyRecoveryV1,
            ) -> Mapping[str, object]:
                if candidate is not recovery:
                    raise FormalContainerExecutorError(
                        FAIL_RECOVERY_SOURCE_ADMISSION,
                        "R2 admission may authorize only the acquired recovery object",
                    )
                intent = candidate.prestage_intent_fields
                actor_report = intent.get("actor_qualification_report")
                errata_report = intent.get("errata_qualification_report")
                if not isinstance(actor_report, Mapping) or not isinstance(
                    errata_report, Mapping
                ):
                    raise FormalContainerExecutorError(
                        FAIL_RECOVERY_SOURCE_ADMISSION,
                        "R2 recovery lacks frozen A8 qualification reports",
                    )
                unchanged_commit_a_inputs = tuple(
                    path for path in REQUIRED_COMMIT_A_INPUTS
                    if path.resolve().relative_to(repository_root.resolve()).as_posix()
                    not in R2_RUNTIME_EXCEPTION_PATHS
                )
                commit_a_admission = validate_ceremony_admission_v1(
                    actor_qualification_report=actor_report,
                    errata_qualification_report=errata_report,
                    basis_commit=A8_BASIS_COMMIT,
                    committed_input_paths=unchanged_commit_a_inputs,
                )
                actor_binding = validate_qualification_report(actor_report)
                validate_dual_errata_qualification_report(errata_report)
                if (
                    actor_binding.get("basis_commit") != A8_BASIS_COMMIT
                    or actor_binding.get("technical_actor_eligible") is not True
                    or errata_report.get("implementation_basis_commit")
                    != A8_BASIS_COMMIT
                ):
                    raise FormalContainerExecutorError(
                        FAIL_RECOVERY_SOURCE_ADMISSION,
                        "R2 qualification reports do not bind eligible A8 actors",
                    )
                return {
                    "schema": "hegel-phase3-m25-a8-r2-source-admission/1",
                    "basis_commit": A8_BASIS_COMMIT,
                    "r1_amendment_commit": R1_AMENDMENT_COMMIT,
                    "r2_amendment_commit": preflight["amendment_commit"],
                    "run_id_hex": FIXED_RUN_ID_HEX,
                    "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                    "recovery_attempt_ordinal": 2,
                    "continuation_action": "CODE_AMENDMENT_RECOVERY_CONTINUATION",
                    "r1_failure_raw_sha256": R1_AUDIT_RAW_SHA256["failure.json"],
                    "incident_diagnostic_sha256": hashlib.sha256(
                        incident_raw
                    ).hexdigest(),
                    "cross_basis_recovery_authorized": True,
                    "formal_identity_entropy_draw_count": 0,
                    "complete_seed_resume_only": True,
                    "unchanged_a8_input_sha256": commit_a_admission["input_sha256"],
                    "unchanged_a8_input_sha256_root": hashlib.sha256(
                        _canonical_json(commit_a_admission["input_sha256"])
                    ).hexdigest(),
                }

            frozen_source_admission = dict(build_source_admission(recovery))
            _admission, admission_raw = _write_audit(
                audit / "admission.json",
                {
                    "schema": f"{AUDIT_SCHEMA_PREFIX}-admission/1",
                    "amendment_commit": preflight["amendment_commit"],
                    "formal_repository_commit": A8_BASIS_COMMIT,
                    "run_id_hex": FIXED_RUN_ID_HEX,
                    "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                    "recovery_attempt_ordinal": 2,
                    "r1_failure_raw_sha256": R1_AUDIT_RAW_SHA256["failure.json"],
                    "incident_diagnostic_sha256": hashlib.sha256(
                        incident_raw
                    ).hexdigest(),
                    "attempt_start_sha256": hashlib.sha256(
                        attempt_start_raw
                    ).hexdigest(),
                    "authorization_sha256": hashlib.sha256(
                        authorization_raw
                    ).hexdigest(),
                    "marker_state": "PENDING",
                    "journal_state": "RESERVED",
                    "seed_artifact_metadata": seed_metadata,
                    "source_admission": frozen_source_admission,
                    "source_admission_sha256": hashlib.sha256(
                        _canonical_json(frozen_source_admission)
                    ).hexdigest(),
                    "complete_seed_resume_only": True,
                    "accepted_worker_mode": "REAL_PENDING_RESUME",
                    "formal_identity_entropy_draw_count": 0,
                    "ephemeral_container_nonce_allowed": True,
                    "cross_basis_recovery_authorized": True,
                    "raw_seed_bytes_read_by_r2_orchestrator": False,
                    "raw_seed_sha256_computed": False,
                },
            )

            def guard(candidate: PendingCeremonyRecoveryV1) -> Mapping[str, object]:
                if candidate is not recovery:
                    raise FormalContainerExecutorError(
                        FAIL_RECOVERY_SOURCE_ADMISSION,
                        "R2 frozen admission may authorize only the acquired recovery object",
                    )
                replayed = dict(build_source_admission(candidate))
                if replayed != frozen_source_admission:
                    raise FormalContainerExecutorError(
                        FAIL_RECOVERY_SOURCE_ADMISSION,
                        "R2 source admission changed after durable admission record",
                    )
                return frozen_source_admission

            failure_phase = "COMPLETE_ONLY_FORMAL_CORE"
            payload, promotion = _continue_pre_stage_pending_recovery_core_v1(
                recovery=recovery,
                actors=actors,
                source_admission_guard=guard,
                complete_seed_resume_only=True,
                static_rust_binary_path=rust_formal_replay_binary,
            )
            stage_directory = recovery.stage_directory

        failure_phase = "FINAL_PUBLIC_REPLAY"
        if admission_raw is None:
            _fail("R2 admission receipt is absent after core completion")
        final = _validate_final_publication_v1(
            payload=payload,
            promotion=promotion,
            custody_directory=custody_directory,
            stage_directory=stage_directory,
            public_evidence_path=public_evidence_path,
            public_promotion_path=public_promotion_path,
        )
        current_r1_rows = _r1_failure_chain_snapshot_v1()
        if _canonical_json(current_r1_rows) != _canonical_json(
            incident.get("r1_failure_chain")
        ):
            _fail("R1 terminal failure chain changed before R2 finalize")
        failure_phase = "FINALIZE_DURABILITY"
        _write_audit(
            audit / "finalize.json",
            {
                "schema": f"{AUDIT_SCHEMA_PREFIX}-finalize/1",
                "amendment_commit": preflight["amendment_commit"],
                "formal_repository_commit": A8_BASIS_COMMIT,
                "run_id_hex": FIXED_RUN_ID_HEX,
                "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                "recovery_attempt_ordinal": 2,
                "r1_failure_raw_sha256": R1_AUDIT_RAW_SHA256["failure.json"],
                "r1_failure_receipt_sha256": R1_FAILURE_RECEIPT_SHA256,
                "incident_diagnostic_sha256": hashlib.sha256(
                    incident_raw
                ).hexdigest(),
                "attempt_start_sha256": hashlib.sha256(
                    attempt_start_raw
                ).hexdigest(),
                "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
                **final,
                "formal_gates_after": 24,
                "child_state": "NOT_RUN",
                "m3_start_invoked": False,
                "accepted_worker_mode": "REAL_PENDING_RESUME",
                "raw_seed_bytes_read_by_r2_orchestrator": False,
                "raw_seed_sha256_computed": False,
                "formal_identity_entropy_draw_count": 0,
                "ephemeral_container_nonce_allowed": True,
            },
        )
        return payload, promotion
    except BaseException as exc:
        failure_path = audit / "failure.json"
        if not failure_path.exists() and not failure_path.is_symlink():
            fields: dict[str, object] = {
                "schema": f"{AUDIT_SCHEMA_PREFIX}-failure/1",
                "amendment_commit": preflight["amendment_commit"],
                "formal_repository_commit": A8_BASIS_COMMIT,
                "run_id_hex": FIXED_RUN_ID_HEX,
                "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                "recovery_attempt_ordinal": 2,
                "r1_failure_raw_sha256": R1_AUDIT_RAW_SHA256["failure.json"],
                "r1_failure_receipt_sha256": R1_FAILURE_RECEIPT_SHA256,
                "incident_diagnostic_sha256": hashlib.sha256(
                    incident_raw
                ).hexdigest(),
                "attempt_start_sha256": hashlib.sha256(
                    attempt_start_raw
                ).hexdigest(),
                "admission_sha256_or_null": (
                    None if admission_raw is None
                    else hashlib.sha256(admission_raw).hexdigest()
                ),
                "failure_code": (
                    FAIL_AMENDMENT
                    if isinstance(exc, A8R1RecoveryAmendmentError)
                    else str(getattr(exc, "code", type(exc).__name__))
                ),
                "failure_phase": failure_phase,
                "failure_detail_sha256": hashlib.sha256(
                    str(exc).encode("utf-8", "replace")
                ).hexdigest(),
                "formal_identity_entropy_draw_count": 0,
                "raw_seed_bytes_read_by_r2_orchestrator": False,
                "raw_seed_sha256_computed": False,
                "m3_start_invoked": False,
            }
            _write_audit(failure_path, fields)
        if isinstance(exc, A8R1RecoveryAmendmentError):
            raise A8R2RecoveryAmendmentError(
                FAIL_AMENDMENT,
                "R1 actor helper rejected the attempt-2 binding: " + exc.detail,
            ) from exc
        raise
    finally:
        if actors is not None:
            actors.close()


__all__ = [
    "A8R2RecoveryAmendmentError",
    "DEFAULT_MANIFEST_PATH",
    "execute_fixed_a8_r2_recovery_v1",
    "inspect_r2_source_preflight_v1",
    "prepare_fixed_a8_r2_authorization_v1",
    "write_fixed_a8_r2_owner_authorization_v1",
]
