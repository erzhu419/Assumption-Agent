"""Bounded R1 recovery amendment for one failed A8 M2.5 transaction.

This module does not provide a second genesis path.  It can only resume the
already-completed seed state of the exact A8 run/ledger pair frozen below.
The formal object repository commit remains A8; the direct-child R1 commit is
diagnostic recovery provenance only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Final, Mapping, NoReturn, Sequence

from .phase3_m25_formal_container_executor_v1 import (
    DockerCeremonyActorsV1,
    FAIL_RECOVERY_SOURCE_ADMISSION,
    FormalContainerExecutorError,
    PendingCeremonyRecoveryV1,
    REQUIRED_COMMIT_A_INPUTS,
    _continue_pre_stage_pending_recovery_core_v1,
    acquire_pending_ceremony_recovery_v1,
    load_gate_evidence_inputs_v1,
    replay_public_gate_evidence_v1,
)
from .phase3_m25_container_ceremony_v1 import (
    TECHNICAL_ACTOR_DISCLOSURE_V1,
    read_marker_snapshot_v1,
    validate_ceremony_admission_v1,
)
from .phase3_m25_formal_static_basis_v1 import (
    FORMAL_GIT_ENVIRONMENT_V1,
    FORMAL_GIT_EXECUTABLE,
)
from .phase3_container_actor_runtime_v1 import validate_qualification_report
from .phase3_m25_errata_qualification_v1 import (
    validate_dual_errata_qualification_report,
)
from .phase3_m25_bridge_dag_binary_qualification_v1 import (
    BridgeDagBinaryQualificationError,
    validate_rust_bridge_dag_binary_qualification_report_v1,
)
from .phase3_m25_wire_v1 import M3_RUN_OUTPUT_ROOTS
from .phase3_m25_external_v1 import assert_public_payload_contains_no_secret_fields


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
DEFAULT_MANIFEST_PATH: Final = (
    PROJECT_ROOT / "config/phase3_m25_a8_recovery_amendment_r1_v1.json"
)

A8_BASIS_COMMIT: Final = "0af65964235390ce2bebefea7379eaa9c50eda24"
FIXED_RUN_ID_HEX: Final = "e4af9f57c38fb298462ec628c4ed8a03"
FIXED_LEDGER_ID_HEX: Final = "ec849e2f1e2e1163cfc450370b25b484"
FIXED_RUN_ID: Final = bytes.fromhex(FIXED_RUN_ID_HEX)
FIXED_LEDGER_ID: Final = bytes.fromhex(FIXED_LEDGER_ID_HEX)
FIXED_AUDIT_DIRECTORY: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/"
    "phase3-m25-0af65964235390ce2bebefea7379eaa9c50eda24/"
    "recovery-audit-e4af9f57c38fb298462ec628c4ed8a03"
)
MANIFEST_SCHEMA: Final = "hegel-phase3-m25-a8-recovery-amendment-r1/1"
AUDIT_SCHEMA_PREFIX: Final = "hegel-phase3-m25-a8-recovery-audit"
FAIL_AMENDMENT = "FAIL_M25_A8_R1_RECOVERY_AMENDMENT"
R1_RUNTIME_EXCEPTION_PATHS: Final = frozenset({
    "Hegel Machine/src/hegel_machine/phase3_m25_formal_container_executor_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_amendment_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_a8_recovery_cli_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_implementation_qualification_v1.py",
    "Hegel Machine/rust/formal_bridge_m25/src/lib.rs",
})


class A8R1RecoveryDockerActorsV1(DockerCeremonyActorsV1):
    """Recovery-only bridge binding for explicit main-worktree A8 artifacts."""

    def validate_rust_bridge_dag_binding(self) -> bytes:
        try:
            report_path = self.rust_bridge_dag_qualification_report
            binary_path = self.rust_bridge_dag_replay_binary
            if report_path.is_symlink() or binary_path.is_symlink():
                _fail("A8 bridge report/binary may not be a symlink")
            report_metadata = report_path.stat()
            binary_metadata = binary_path.stat()
            report_raw = report_path.read_bytes()
            report = json.loads(report_raw)
            if (
                not stat.S_ISREG(report_metadata.st_mode)
                or stat.S_IMODE(report_metadata.st_mode) != 0o644
                or type(report) is not dict
                or _canonical_json(report) != report_raw
                or not stat.S_ISREG(binary_metadata.st_mode)
                or stat.S_IMODE(binary_metadata.st_mode) != 0o755
            ):
                _fail("A8 bridge report/binary bytes or mode differ")
            digest_text = validate_rust_bridge_dag_binary_qualification_report_v1(
                report,
                expected_basis_commit=A8_BASIS_COMMIT,
                verify_commit_sources=False,
                verify_persisted_binary=False,
            )
            source = report.get("source")
            bindings = source.get("bindings") if isinstance(source, Mapping) else None
            if not isinstance(bindings, Mapping):
                _fail("A8 bridge report source bindings are absent")
            for relative, expected in bindings.items():
                if (
                    type(relative) is not str
                    or type(expected) is not str
                    or hashlib.sha256(
                        _git(REPOSITORY_ROOT, ["show", f"{A8_BASIS_COMMIT}:{relative}"])
                    ).hexdigest()
                    != expected.removeprefix("sha256:")
                ):
                    _fail(f"A8 bridge report source binding differs: {relative}")
            digest = bytes.fromhex(digest_text.removeprefix("sha256:"))
            binary_hasher = hashlib.sha256()
            descriptor = os.open(
                binary_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    binary_hasher.update(chunk)
            finally:
                os.close(descriptor)
            if binary_hasher.digest() != digest:
                _fail("explicit A8 bridge binary digest differs from report")
            report_identifier = report.get("diagnostic_report_sha256")
            if (
                type(report_identifier) is not str
                or re.fullmatch(r"sha256:[0-9a-f]{64}", report_identifier) is None
            ):
                _fail("A8 bridge report diagnostic ID differs")
            self._bound_rust_bridge_dag_digest = digest
            self._bound_rust_bridge_dag_report_sha256 = bytes.fromhex(
                report_identifier.removeprefix("sha256:")
            )
            return digest
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, BridgeDagBinaryQualificationError) as exc:
            raise FormalContainerExecutorError(
                FAIL_RECOVERY_SOURCE_ADMISSION,
                f"explicit A8 bridge binding is invalid: {exc}",
            ) from exc


class A8RecoveryAmendmentError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(detail: str) -> NoReturn:
    raise A8RecoveryAmendmentError(FAIL_AMENDMENT, detail)


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


def _load_manifest(path: Path) -> tuple[dict[str, object], bytes]:
    if path.is_symlink():
        _fail("amendment manifest may not be a symlink")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"amendment manifest is invalid JSON: {exc}")
    if type(value) is not dict or _canonical_json(value) != raw:
        _fail("amendment manifest is not canonical exact JSON")
    required = {
        "schema",
        "source_commit_selector",
        "sole_parent_commit",
        "formal_repository_commit",
        "fixed_run_id_hex",
        "fixed_ledger_id_hex",
        "exact_changed_paths",
        "source_bindings",
        "complete_seed_resume_only",
        "formal_identity_entropy_draw_count",
        "ephemeral_container_nonce_allowed",
        "ordinary_execute_allowed",
        "ordinary_recovery_cross_basis_allowed",
        "fixed_audit_directory",
    }
    if set(value) != required:
        _fail("amendment manifest field set differs")
    if (
        value.get("schema") != MANIFEST_SCHEMA
        or value.get("source_commit_selector") != "HEAD"
        or value.get("sole_parent_commit") != A8_BASIS_COMMIT
        or value.get("formal_repository_commit") != A8_BASIS_COMMIT
        or value.get("fixed_run_id_hex") != FIXED_RUN_ID_HEX
        or value.get("fixed_ledger_id_hex") != FIXED_LEDGER_ID_HEX
        or value.get("complete_seed_resume_only") is not True
        or value.get("formal_identity_entropy_draw_count") != 0
        or value.get("ephemeral_container_nonce_allowed") is not True
        or value.get("ordinary_execute_allowed") is not False
        or value.get("ordinary_recovery_cross_basis_allowed") is not False
        or value.get("fixed_audit_directory")
        != FIXED_AUDIT_DIRECTORY.as_posix()
    ):
        _fail("amendment manifest fixed policy differs")
    return value, raw


def inspect_r1_source_preflight_v1(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, object]:
    """Verify direct-child R1, clean checkout, allowlisted diff and blob hashes."""

    manifest, manifest_raw = _load_manifest(manifest_path)
    head = _git(repository_root, ["rev-parse", "--verify", "HEAD^{commit}"]).decode("ascii").strip()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None or head == A8_BASIS_COMMIT:
        _fail("HEAD selector is not one committed R1 child")
    parents = _git(repository_root, ["show", "-s", "--format=%P", head]).decode("ascii").strip().split()
    if parents != [A8_BASIS_COMMIT]:
        _fail("R1 must have A8 as its sole parent")
    if _git(repository_root, ["status", "--porcelain=v1", "--untracked-files=all"]):
        _fail("R1 repository tree/index is not clean")

    changed_lines = _git(
        repository_root,
        ["diff-tree", "--no-commit-id", "--name-status", "-r", "--no-renames", A8_BASIS_COMMIT, head],
    ).decode("utf-8", "strict").splitlines()
    actual_changes = tuple(
        {"status": line.split("\t", 1)[0], "path": line.split("\t", 1)[1]}
        for line in changed_lines
        if line
    )
    expected_changes = manifest.get("exact_changed_paths")
    if type(expected_changes) is not list or tuple(expected_changes) != actual_changes:
        _fail("R1 changed-path allowlist differs")

    bindings = manifest.get("source_bindings")
    if type(bindings) is not list or not bindings:
        _fail("R1 source bindings are absent")
    manifest_relative = manifest_path.resolve(strict=True).relative_to(
        repository_root.resolve(strict=True)
    ).as_posix()
    changed_paths = {str(row["path"]) for row in actual_changes}
    binding_paths = {
        str(row.get("path")) for row in bindings if isinstance(row, Mapping)
    }
    if binding_paths != changed_paths - {manifest_relative}:
        _fail("R1 source bindings do not equal changed paths minus manifest")
    verified: list[dict[str, object]] = []
    for row in bindings:
        if type(row) is not dict or set(row) != {
            "path", "a8_sha256_or_null", "r1_sha256"
        }:
            _fail("R1 source-binding row differs")
        path = row.get("path")
        a8_hash = row.get("a8_sha256_or_null")
        r1_hash = row.get("r1_sha256")
        if (
            type(path) is not str
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
            or (a8_hash is not None and (type(a8_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", a8_hash) is None))
            or type(r1_hash) is not str
            or re.fullmatch(r"[0-9a-f]{64}", r1_hash) is None
        ):
            _fail("R1 source-binding value differs")
        r1_blob = _git(repository_root, ["show", f"{head}:{path}"])
        if hashlib.sha256(r1_blob).hexdigest() != r1_hash:
            _fail(f"R1 source blob hash differs: {path}")
        if a8_hash is None:
            probe = subprocess.run(
                [str(FORMAL_GIT_EXECUTABLE), "cat-file", "-e", f"{A8_BASIS_COMMIT}:{path}"],
                cwd=repository_root.resolve(strict=True),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=60,
                env=dict(FORMAL_GIT_ENVIRONMENT_V1),
            )
            if probe.returncode == 0:
                _fail(f"R1 source unexpectedly existed in A8: {path}")
        else:
            a8_blob = _git(repository_root, ["show", f"{A8_BASIS_COMMIT}:{path}"])
            if hashlib.sha256(a8_blob).hexdigest() != a8_hash:
                _fail(f"A8 source blob hash differs: {path}")
        verified.append({"path": path, "r1_sha256": r1_hash, "a8_sha256_or_null": a8_hash})

    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-preflight/1",
        "amendment_commit": head,
        "sole_parent_commit": A8_BASIS_COMMIT,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "source_bindings": verified,
        "repository_clean": True,
        "exact_changed_paths_verified": True,
        "formal_identity_entropy_draw_count": 0,
        "ephemeral_container_nonce_allowed": True,
        "m3_start_allowed": False,
        "fixed_audit_directory": FIXED_AUDIT_DIRECTORY.as_posix(),
    }


def _prepare_audit_directory(path: Path, repository_root: Path) -> Path:
    if path.is_symlink():
        _fail("audit directory may not be a symlink")
    resolved = path.resolve(strict=True)
    if resolved != FIXED_AUDIT_DIRECTORY.resolve(strict=False):
        _fail("audit directory differs from the transaction-frozen external path")
    repository = repository_root.resolve(strict=True)
    if resolved == repository or repository in resolved.parents:
        _fail("audit directory must be repository-external")
    metadata = resolved.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
    ):
        _fail("audit directory must be caller-owned mode 0700")
    return resolved


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _write_audit(path: Path, fields: Mapping[str, object]) -> tuple[dict[str, object], bytes]:
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
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return body, payload


def _read_canonical_audit(path: Path) -> tuple[dict[str, object], bytes]:
    if path.is_symlink():
        _fail(f"audit input may not be a symlink: {path.name}")
    metadata = path.stat()
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"audit input is invalid: {path.name}: {exc}")
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or type(value) is not dict
        or _canonical_json(value) != raw
    ):
        _fail(f"audit input bytes/mode differ: {path.name}")
    body = dict(value)
    claimed = body.pop("receipt_sha256", None)
    if claimed != hashlib.sha256(_canonical_json(body)).hexdigest():
        _fail(f"audit input self-hash differs: {path.name}")
    return value, raw


def _with_receipt_sha256(fields: Mapping[str, object]) -> dict[str, object]:
    result = dict(fields)
    result["receipt_sha256"] = hashlib.sha256(_canonical_json(result)).hexdigest()
    return result


def _authorization_request_fields(
    *, amendment_commit: object, preflight_raw: bytes
) -> dict[str, object]:
    return {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-authorization-request/1",
        "amendment_commit": amendment_commit,
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "preflight_sha256": hashlib.sha256(preflight_raw).hexdigest(),
        "requested_action": "COMPLETE_ONLY_REAL_PENDING_RESUME",
        "ordinary_execute_allowed": False,
        "redraw_allowed": False,
        "abort_allowed": False,
        "poststage_recovery_allowed": False,
        "formal_identity_entropy_draw_count": 0,
    }


def prepare_fixed_a8_r1_authorization_v1(
    *,
    audit_directory: Path,
    repository_root: Path = REPOSITORY_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, object]:
    """Write preflight/request only; a separate operator supplies authorization."""

    audit = _prepare_audit_directory(audit_directory, repository_root)
    preflight = inspect_r1_source_preflight_v1(
        repository_root=repository_root, manifest_path=manifest_path
    )
    preflight_record, preflight_raw = _write_audit(audit / "preflight.json", preflight)
    request, _raw = _write_audit(
        audit / "authorization-request.json",
        _authorization_request_fields(
            amendment_commit=preflight_record["amendment_commit"],
            preflight_raw=preflight_raw,
        ),
    )
    return request


def write_fixed_a8_r1_owner_authorization_v1(
    *,
    audit_directory: Path,
    owner_confirmation: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, object]:
    """Separate explicit owner action; writes one O_EXCL canonical authorization."""

    if owner_confirmation != "AUTHORIZE_A8_R1_COMPLETE_ONLY_REAL_PENDING_RESUME":
        _fail("owner confirmation phrase differs")
    audit = _prepare_audit_directory(audit_directory, repository_root)
    preflight, preflight_raw = _read_canonical_audit(audit / "preflight.json")
    request, request_raw = _read_canonical_audit(audit / "authorization-request.json")
    expected_request = _with_receipt_sha256(
        _authorization_request_fields(
            amendment_commit=preflight["amendment_commit"],
            preflight_raw=preflight_raw,
        )
    )
    if (
        preflight.get("formal_repository_commit") != A8_BASIS_COMMIT
        or preflight.get("run_id_hex") != FIXED_RUN_ID_HEX
        or preflight.get("ledger_id_hex") != FIXED_LEDGER_ID_HEX
        or request != expected_request
    ):
        _fail("authorization inputs differ from the fixed transaction")
    authorization, _raw = _write_audit(
        audit / "authorization.json",
        {
            "schema": f"{AUDIT_SCHEMA_PREFIX}-authorization/1",
            "amendment_commit": preflight["amendment_commit"],
            "formal_repository_commit": A8_BASIS_COMMIT,
            "run_id_hex": FIXED_RUN_ID_HEX,
            "ledger_id_hex": FIXED_LEDGER_ID_HEX,
            "preflight_sha256": hashlib.sha256(preflight_raw).hexdigest(),
            "authorization_request_sha256": hashlib.sha256(request_raw).hexdigest(),
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


def _require_complete_seed_metadata(custody: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for name, expected_size in (
        ("split_seed_generation.intent", None),
        ("split_master_seed.bin", 32),
        ("split_seed_generation.complete", None),
    ):
        path = custody / name
        if path.is_symlink():
            _fail(f"complete-only seed artifact is a symlink: {name}")
        metadata = path.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (expected_size is not None and metadata.st_size != expected_size)
        ):
            _fail(f"complete-only seed artifact metadata differs: {name}")
        row: dict[str, object] = {
            "name": name,
            "kind": "regular",
            "mode_octal": "0600",
            "size_bytes": metadata.st_size,
            "st_dev": metadata.st_dev,
            "st_ino": metadata.st_ino,
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
            "raw_bytes_read": False,
            "sha256_computed": False,
        }
        if name != "split_master_seed.bin":
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                opened = os.fstat(descriptor)
                if (
                    (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
                    or stat.S_IMODE(opened.st_mode) != 0o600
                ):
                    _fail(f"seed metadata file raced while opened: {name}")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
            finally:
                os.close(descriptor)
            row["sha256"] = hashlib.sha256(b"".join(chunks)).hexdigest()
            row["raw_bytes_read"] = True
            row["sha256_computed"] = True
        rows.append(row)
    return tuple(rows)


def execute_fixed_a8_r1_recovery_v1(
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
    """Resume the one fixed A8 transaction; never execute, redraw, abort or start M3."""

    audit = _prepare_audit_directory(audit_directory, repository_root)
    terminal_receipts = tuple(
        name
        for name in ("failure.json", "finalize.json")
        if (audit / name).exists() or (audit / name).is_symlink()
    )
    if terminal_receipts:
        _fail(
            "R1 recovery audit is terminal and may not be retried: "
            + ",".join(terminal_receipts)
        )
    protected_paths = (
        custody_directory.resolve(strict=True),
        public_evidence_path.parent.resolve(strict=True),
        public_promotion_path.parent.resolve(strict=True),
    )
    if any(_paths_overlap(audit, path) for path in protected_paths):
        _fail("audit directory overlaps custody or public output storage")
    preflight_record, preflight_raw = _read_canonical_audit(audit / "preflight.json")
    current_preflight = inspect_r1_source_preflight_v1(
        repository_root=repository_root, manifest_path=manifest_path
    )
    current_preflight["receipt_sha256"] = hashlib.sha256(
        _canonical_json(current_preflight)
    ).hexdigest()
    if current_preflight != preflight_record:
        _fail("stored preflight differs from current clean R1 source")
    request_record, request_raw = _read_canonical_audit(
        audit / "authorization-request.json"
    )
    authorization_record, authorization_raw = _read_canonical_audit(
        audit / "authorization.json"
    )
    expected_authorization = {
        "schema": f"{AUDIT_SCHEMA_PREFIX}-authorization/1",
        "amendment_commit": preflight_record["amendment_commit"],
        "formal_repository_commit": A8_BASIS_COMMIT,
        "run_id_hex": FIXED_RUN_ID_HEX,
        "ledger_id_hex": FIXED_LEDGER_ID_HEX,
        "preflight_sha256": hashlib.sha256(preflight_raw).hexdigest(),
        "authorization_request_sha256": hashlib.sha256(request_raw).hexdigest(),
        "authorization_actor": "PROJECT_OWNER",
        "owner_authorized_fixed_transaction_only": True,
        "ordinary_execute_invoked": False,
        "redraw_allowed": False,
        "abort_allowed": False,
        "poststage_recovery_allowed": False,
        "formal_identity_entropy_draw_count": 0,
    }
    expected_authorization["receipt_sha256"] = hashlib.sha256(
        _canonical_json(expected_authorization)
    ).hexdigest()
    if authorization_record != expected_authorization:
        _fail("independent canonical authorization differs")
    expected_request = _with_receipt_sha256(
        _authorization_request_fields(
            amendment_commit=preflight_record["amendment_commit"],
            preflight_raw=preflight_raw,
        )
    )
    if request_record != expected_request:
        _fail("authorization request exact fields differ")

    actors = A8R1RecoveryDockerActorsV1(
        basis_commit=A8_BASIS_COMMIT,
        custody_directory=custody_directory,
        rust_formal_replay_binary=rust_formal_replay_binary,
        rust_bridge_dag_replay_binary=rust_bridge_dag_replay_binary,
        rust_bridge_dag_qualification_report=rust_bridge_dag_qualification_report,
        timestamp=0,
    )
    try:
        with acquire_pending_ceremony_recovery_v1(
            custody_directory=custody_directory,
            public_evidence_path=public_evidence_path,
            public_promotion_path=public_promotion_path,
            actors=actors,
        ) as recovery:
            if _paths_overlap(audit, recovery.stage_directory.resolve(strict=True)):
                _fail("audit directory overlaps the reserved formal stage")
            if (
                recovery.basis_commit != A8_BASIS_COMMIT
                or recovery.run_id != FIXED_RUN_ID
                or recovery.ledger_id != FIXED_LEDGER_ID
                or recovery.marker_snapshot.state != "PENDING"
                or recovery.journal_state != "RESERVED"
            ):
                _fail("acquired recovery identity is not the fixed A8 RESERVED transaction")
            seed_metadata = _require_complete_seed_metadata(recovery.custody_directory)
            actors.timestamp = recovery.marker_snapshot.created_at_unix_seconds

            def build_source_admission(
                candidate: PendingCeremonyRecoveryV1,
            ) -> Mapping[str, object]:
                if candidate is not recovery:
                    raise FormalContainerExecutorError(
                        FAIL_RECOVERY_SOURCE_ADMISSION,
                        "R1 admission may authorize only the acquired recovery object",
                    )
                intent = candidate.prestage_intent_fields
                actor_report = intent.get("actor_qualification_report")
                errata_report = intent.get("errata_qualification_report")
                if not isinstance(actor_report, Mapping) or not isinstance(
                    errata_report, Mapping
                ):
                    raise FormalContainerExecutorError(
                        FAIL_RECOVERY_SOURCE_ADMISSION,
                        "R1 recovery lacks frozen A8 qualification reports",
                    )
                unchanged_commit_a_inputs = tuple(
                    path
                    for path in REQUIRED_COMMIT_A_INPUTS
                    if path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
                    not in R1_RUNTIME_EXCEPTION_PATHS
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
                        "R1 recovery qualification reports do not bind eligible A8 actors",
                    )
                return {
                    "schema": "hegel-phase3-m25-a8-r1-source-admission/1",
                    "basis_commit": A8_BASIS_COMMIT,
                    "run_id_hex": FIXED_RUN_ID_HEX,
                    "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                    "cross_basis_recovery_authorized": True,
                    "formal_identity_entropy_draw_count": 0,
                    "complete_seed_resume_only": True,
                    "unchanged_a8_input_sha256": commit_a_admission["input_sha256"],
                    "unchanged_a8_input_sha256_root": hashlib.sha256(
                        _canonical_json(commit_a_admission["input_sha256"])
                    ).hexdigest(),
                }

            frozen_source_admission = dict(build_source_admission(recovery))
            admission_record, admission_raw = _write_audit(
                audit / "admission.json",
                {
                    "schema": f"{AUDIT_SCHEMA_PREFIX}-admission/1",
                    "amendment_commit": preflight_record["amendment_commit"],
                    "basis_commit": A8_BASIS_COMMIT,
                    "run_id_hex": FIXED_RUN_ID_HEX,
                    "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                    "authorization_sha256": hashlib.sha256(authorization_raw).hexdigest(),
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
                },
            )

            def guard(candidate: PendingCeremonyRecoveryV1) -> Mapping[str, object]:
                if candidate is not recovery:
                    raise FormalContainerExecutorError(
                        FAIL_RECOVERY_SOURCE_ADMISSION,
                        "R1 frozen admission may authorize only the acquired recovery object",
                    )
                replayed_source_admission = dict(build_source_admission(candidate))
                if replayed_source_admission != frozen_source_admission:
                    raise FormalContainerExecutorError(
                        FAIL_RECOVERY_SOURCE_ADMISSION,
                        "R1 source admission changed after durable admission record",
                    )
                return frozen_source_admission

            payload, promotion = _continue_pre_stage_pending_recovery_core_v1(
                recovery=recovery,
                actors=actors,
                source_admission_guard=guard,
                complete_seed_resume_only=True,
                static_rust_binary_path=rust_formal_replay_binary,
            )
        evidence_raw = public_evidence_path.read_bytes()
        promotion_raw = public_promotion_path.read_bytes()
        receipt_path = public_promotion_path.with_name(
            public_promotion_path.name + ".publication-receipt.json"
        )
        receipt_raw = receipt_path.read_bytes()
        seed_verification_raw = (
            recovery.stage_directory / "seed-custody-verification.json"
        ).read_bytes()
        replayed_promotion = replay_public_gate_evidence_v1(payload)
        if _canonical_json(replayed_promotion) != promotion_raw:
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
            or complete_marker.custodian_key_id
            != inputs.marker_snapshot.custodian_key_id
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
            )
            is not str
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(
                    publication_receipt.get(
                        "seed_custody_verification_receipt_sha256_or_null"
                    )
                ),
            )
            is None
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
        _write_audit(
            audit / "finalize.json",
            {
                "schema": f"{AUDIT_SCHEMA_PREFIX}-finalize/1",
                "amendment_commit": preflight_record["amendment_commit"],
                "formal_repository_commit": A8_BASIS_COMMIT,
                "run_id_hex": FIXED_RUN_ID_HEX,
                "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
                "public_evidence_sha256": hashlib.sha256(evidence_raw).hexdigest(),
                "public_promotion_sha256": hashlib.sha256(promotion_raw).hexdigest(),
                "publication_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
                "seed_custody_verification_receipt_sha256": hashlib.sha256(
                    seed_verification_raw
                ).hexdigest(),
                "complete_marker_seed_commitment_manifest_root_hex": (
                    complete_marker.seed_commitment_manifest_root.hex()
                ),
                "complete_marker_custodian_key_id_hex": (
                    complete_marker.custodian_key_id.hex()
                ),
                "formal_gates_after": 24,
                "child_state": "NOT_RUN",
                "m3_start_invoked": False,
                "accepted_worker_mode": "REAL_PENDING_RESUME",
                "raw_seed_bytes_read_by_amendment_orchestrator": False,
                "raw_seed_sha256_computed": False,
                "formal_identity_entropy_draw_count": 0,
                "ephemeral_container_nonce_allowed": True,
            },
        )
        return payload, promotion
    except BaseException as exc:
        if not (audit / "failure.json").exists():
            code = getattr(exc, "code", type(exc).__name__)
            _write_audit(
                audit / "failure.json",
                {
                    "schema": f"{AUDIT_SCHEMA_PREFIX}-failure/1",
                    "formal_repository_commit": A8_BASIS_COMMIT,
                    "run_id_hex": FIXED_RUN_ID_HEX,
                    "ledger_id_hex": FIXED_LEDGER_ID_HEX,
                    "failure_code": str(code),
                    "formal_identity_entropy_draw_count": 0,
                    "raw_seed_bytes_read_by_amendment_orchestrator": False,
                    "raw_seed_sha256_computed": False,
                },
            )
        raise
    finally:
        actors.close()


__all__ = [
    "A8RecoveryAmendmentError",
    "execute_fixed_a8_r1_recovery_v1",
    "inspect_r1_source_preflight_v1",
    "prepare_fixed_a8_r1_authorization_v1",
    "write_fixed_a8_r1_owner_authorization_v1",
]
