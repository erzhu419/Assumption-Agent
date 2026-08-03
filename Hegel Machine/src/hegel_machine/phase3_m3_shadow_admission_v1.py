"""Owner-authorized, non-authoritative Phase-3 M3 shadow admission.

This module deliberately implements a *separate* state machine.  It never
modifies the formal M2.5 gate count, creates a formal root, or claims that the
locally isolated workers are external actors.  Admission is side-effect free
with respect to secret material; the explicit start transaction is the first
operation permitted to invoke the purpose-separated runtime ceremony.

The public JSON is diagnostic.  The identities which matter are strict CBOR
arrays hashed in the document-local ``HEGEL/INTERNAL_SHADOW`` domain.  Shadow
digest fields therefore use ``*_digest`` and never masquerade as formal roots.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Final, Iterator, Mapping, NoReturn, Sequence

from .phase3_m25_errata_qualification_v1 import (
    POST_COMMIT_REPORT_PATH,
    validate_checked_errata_qualification_report,
)
from .phase3_m3_shadow_wire_v1 import (
    FORMAL_CHILD_DSL_ID,
    FORMAL_MACHINE_FREEZE_ID,
    FORMAL_TRACK_SNAPSHOT,
    SHADOW_ALL_GATES_BITSET,
    SHADOW_ARTIFACT_KIND,
    SHADOW_ARTIFACT_KIND_ID,
    SHADOW_GATE_COUNT,
    SHADOW_ISOLATION_INVARIANT_BITSET,
    SHADOW_ISOLATION_PROFILE_ID,
    SHADOW_OBJECT_TAGS,
    SHADOW_PURPOSE_IDS,
    SHADOW_SCHEMA_REGISTRY,
    SHADOW_TRACK_ID,
    ShadowAdmissionGateId,
    ShadowProbePhaseId,
    build_shadow_object,
    decode_shadow_object,
    git_sha1_commit_id,
    require_shadow_admission,
    shadow_digest_v1,
    shadow_object_digest,
    shadow_purpose_worker_digest_set_v1,
    shadow_security_probe_set_digest_v1,
    shadow_tree_digest_v1,
    validate_formal_track_snapshot,
    validate_shadow_artifact_header,
    validate_shadow_state_transition,
)
from .strict_cbor_v1 import canonical_cbor_decode, canonical_cbor_encode


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent

SCHEMA_VERSION: Final = "hegel-phase3-m3-shadow-admission/1"
START_SCHEMA_VERSION: Final = "hegel-phase3-m3-shadow-start/1"
ARTIFACT_KIND_ID: Final = SHADOW_ARTIFACT_KIND_ID
ARTIFACT_KIND: Final = SHADOW_ARTIFACT_KIND
FORMAL_TRACK_STATUS: Final = "FROZEN_PRE_GENESIS_BASELINE_14_OF_24_NOT_RUN"
FORMAL_FOLLOW_ON_RECOMMENDATION: Final = (
    "SEPARATE_OWNER_AMENDED_HARDENED_OFFLINE_CONTAINER_CEREMONY_REQUIRED"
)
CLAIM_BOUNDARY: Final = (
    "A 12/12 result qualifies internal isolation mechanics only. It does not "
    "advance formal Gate 15-24 and may only feed a separate owner-amended "
    "hardened offline-container ceremony. The embedded 14/24 / NOT_RUN value "
    "is a revalidated pre-genesis eligibility baseline, not a historical "
    "snapshot presented as live formal-state authority."
)
MACHINE_FREEZE_ID: Final = FORMAL_MACHINE_FREEZE_ID
CHILD_DSL_ID: Final = FORMAL_CHILD_DSL_ID
ISOLATION_PROFILE_ID: Final = SHADOW_ISOLATION_PROFILE_ID
PURPOSE_IDS: Final = tuple(item.value for item in SHADOW_PURPOSE_IDS)

AMENDMENT_RELATIVE_PATH: Final = (
    "docs/Hegel_Machine_Phase3A_Internal_Shadow_Execution_Amendment_v1.md"
)
CHECKED_REPORT_RELATIVE_PATH: Final = (
    "artifacts/phase3_m25_external/phase3_m25_errata_qualification_v1.json"
)
CHECKED_REPORT_PATH: Final = POST_COMMIT_REPORT_PATH
DEFAULT_ADMISSION_ARTIFACT_PATH: Final = (
    PROJECT_ROOT
    / "artifacts"
    / "phase3_internal_shadow"
    / "phase3_m3_shadow_admission_v1.json"
)
DEFAULT_START_ARTIFACT_PATH: Final = (
    PROJECT_ROOT
    / "artifacts"
    / "phase3_internal_shadow"
    / "phase3_m3_shadow_start_v1.json"
)

SHADOW_GATE_NAMES: Final = tuple(gate.name for gate in ShadowAdmissionGateId)

# These files extend the checked errata source closure for the shadow basis.
# The errata artifact contributes its own complete SOURCE_PATHS dynamically.
SHADOW_BASIS_PATHS: Final = (
    AMENDMENT_RELATIVE_PATH,
    CHECKED_REPORT_RELATIVE_PATH,
    "src/hegel_machine/phase3_m3_shadow_runtime_v1.py",
    "src/hegel_machine/phase3_m3_shadow_wire_v1.py",
    "src/hegel_machine/phase3_m3_shadow_admission_v1.py",
    "src/hegel_machine/phase3_m3_shadow_cli_v1.py",
    "config/phase3_container_actor_profile_v1.json",
    "config/phase3_internal_actor_seccomp_v1.json",
    "docs/Hegel_Machine_Owner_Accepted_Container_Technical_Actor_Eligibility_Amendment_v1.md",
    "docs/phase3_m25_external_genesis_operator_runbook.md",
    "tests/test_phase3_m3_shadow_runtime_v1.py",
    "tests/test_phase3_m3_shadow_wire_v1.py",
    "tests/test_phase3_m3_shadow_admission_v1.py",
)

# The v1 shadow wire encodes the pre-genesis 14/24 / NOT_RUN baseline.  Once a
# formal promotion is committed, v1 must stop rather than mislabel 24/24 (or a
# later M3 state) with those frozen counters.  A future dynamic formal-state
# binding requires a versioned shadow wire.
FORMAL_BASELINE_SUPERSEDING_PATHS: Final = (
    (
        "artifacts/phase3_m25_external/formal_genesis_v2/"
        "phase3_m25_formal_gate_evidence_v1.json"
    ),
    (
        "artifacts/phase3_m25_external/formal_genesis_v2/"
        "phase3_m25_gate_promotion_v1.json"
    ),
    (
        "artifacts/phase3_m25_external/formal_genesis_v2/"
        "phase3_m25_gate_promotion_v1.json.publication-receipt.json"
    ),
)

FORMAL_TRACK: Final = dict(FORMAL_TRACK_SNAPSHOT)

_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_DIGEST_RE: Final = re.compile(r"^[0-9a-f]{64}$")


class ShadowAdmissionError(RuntimeError):
    """Stable fail-closed shadow admission/start error."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise ShadowAdmissionError(code, detail)


def formal_track_snapshot() -> dict[str, object]:
    """Return the frozen pre-genesis eligibility baseline, not a live query."""

    return dict(FORMAL_TRACK)


def _git_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_SSH_COMMAND": "false",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def _git(*arguments: str, input_bytes: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=REPOSITORY_ROOT,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
        env=_git_environment(),
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        _fail("FAIL_SHADOW_BASIS_COMMIT_MISMATCH", detail or "Git command failed")
    return completed.stdout


def _require_commit_id(value: object) -> str:
    if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
        _fail(
            "FAIL_SHADOW_BASIS_COMMIT_MISMATCH",
            "basis commit must be a lowercase 40-hex SHA-1",
        )
    return value


def repository_head_commit() -> str:
    return _require_commit_id(_git("rev-parse", "HEAD").decode("ascii").strip())


def _repo_relative(relative_to_project: str) -> str:
    if not relative_to_project or relative_to_project.startswith(("/", "../")):
        _fail("FAIL_SHADOW_BASIS_COMMIT_MISMATCH", "unsafe project-relative path")
    candidate = Path(relative_to_project)
    if ".." in candidate.parts or candidate.is_absolute():
        _fail("FAIL_SHADOW_BASIS_COMMIT_MISMATCH", "unsafe project-relative path")
    return f"Hegel Machine/{candidate.as_posix()}"


def _git_blob(commit_id: str, project_relative_path: str) -> bytes:
    return _git("show", f"{commit_id}:{_repo_relative(project_relative_path)}")


def _assert_basis_reachable(commit_id: str) -> None:
    """Require the execution basis to be the exact current HEAD.

    The historical implementation accepted any reachable ancestor while
    importing the runtime from the current worktree.  That mixed two source
    identities.  Exact HEAD plus the clean/stability checks below makes the
    runtime and claimed snapshot one basis.
    """

    if repository_head_commit() != commit_id:
        _fail(
            "FAIL_SHADOW_BASIS_COMMIT_MISMATCH",
            "basis commit must equal the current repository HEAD",
        )


def _assert_frozen_formal_baseline_current(commit_id: str) -> None:
    """Fail once committed formal evidence supersedes shadow-wire v1."""

    _assert_basis_reachable(commit_id)
    rows = _git(
        "ls-tree",
        "-z",
        commit_id,
        "--",
        *(_repo_relative(path) for path in FORMAL_BASELINE_SUPERSEDING_PATHS),
    )
    if rows:
        _fail(
            "FAIL_SHADOW_FORMAL_STATE_MUTATION",
            "shadow wire v1 is inapplicable after formal evidence/promotion is committed",
        )


def _assert_shadow_execution_basis_stable(commit_id: str) -> None:
    """Recheck HEAD, clean Hegel bytes, and the v1 formal baseline guard."""

    _assert_frozen_formal_baseline_current(commit_id)
    _assert_hegel_worktree_clean()


def _assert_hegel_worktree_clean() -> None:
    status = _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "Hegel Machine",
    )
    if status:
        _fail(
            "FAIL_SHADOW_BASIS_COMMIT_MISMATCH",
            "Hegel Machine must be clean; admission never binds working-tree bytes",
        )


def _git_tree_manifest(commit_id: str) -> tuple[list[tuple[object, ...]], str]:
    """Return an exact blob manifest and the committed Hegel subtree ID."""

    tree_id = _git(
        "rev-parse", f"{commit_id}:Hegel Machine"
    ).decode("ascii").strip()
    if _COMMIT_RE.fullmatch(tree_id) is None:
        _fail("FAIL_SHADOW_BASIS_COMMIT_MISMATCH", "invalid Hegel subtree ID")
    raw = _git(
        "ls-tree", "-r", "-l", "-z", commit_id, "--", "Hegel Machine"
    )
    rows: list[tuple[object, ...]] = []
    for record in raw.split(b"\x00"):
        if not record:
            continue
        try:
            header, path = record.split(b"\t", 1)
            mode, kind, object_id, size = header.split(None, 3)
        except ValueError:
            _fail("FAIL_SHADOW_BASIS_COMMIT_MISMATCH", "malformed Git tree row")
        if kind != b"blob" or size == b"-":
            _fail(
                "FAIL_SHADOW_BASIS_COMMIT_MISMATCH",
                "Hegel snapshot contains a non-blob entry",
            )
        if not path.startswith(b"Hegel Machine/"):
            _fail("FAIL_SHADOW_BASIS_COMMIT_MISMATCH", "tree escaped Hegel scope")
        try:
            size_value = int(size)
        except ValueError:
            _fail("FAIL_SHADOW_BASIS_COMMIT_MISMATCH", "invalid Git blob size")
        rows.append((path, mode, bytes.fromhex(object_id.decode("ascii")), size_value))
    if not rows or rows != sorted(rows, key=lambda row: row[0]):
        _fail("FAIL_SHADOW_BASIS_COMMIT_MISMATCH", "snapshot manifest is not sorted")
    return rows, tree_id


def shadow_digest(domain: str, value: object) -> bytes:
    schema = SHADOW_SCHEMA_REGISTRY.get(domain)
    if schema is None:
        _fail("FAIL_SHADOW_DOMAIN_COLLISION", f"unregistered shadow domain: {domain}")
    return shadow_digest_v1(schema.digest_domain, value)


def shadow_tree_digest(domain: str, records: Sequence[object]) -> bytes:
    return shadow_tree_digest_v1(domain, records)


def _wire_object(name: str, fields: Mapping[str, object]) -> tuple[object, ...]:
    return build_shadow_object(name, fields)


def _wire_report(name: str, value: tuple[object, ...]) -> dict[str, object]:
    validate_shadow_wire_object(name, value)
    encoded = canonical_cbor_encode(value)
    decoded = decode_shadow_object(encoded, expected_name=name)
    digest = shadow_object_digest(name, decoded.fields)
    return {
        "schema": name,
        "numeric_tag": SHADOW_OBJECT_TAGS[name],
        "cbor_hex": encoded.hex(),
        "digest": digest.hex(),
    }


def validate_shadow_wire_object(name: str, value: object) -> None:
    if name not in SHADOW_SCHEMA_REGISTRY:
        _fail("FAIL_SHADOW_DOMAIN_COLLISION", "unknown shadow object")
    if type(value) is not tuple or len(value) < 3:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", f"{name} is not an array")
    encoded = canonical_cbor_encode(value)
    decoded = decode_shadow_object(encoded, expected_name=name)
    if decoded.value != value:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", f"{name} is not canonical")
    digest = shadow_object_digest(name, decoded.fields)
    if len(digest) != 32:
        _fail("FAIL_SHADOW_DOMAIN_COLLISION", "shadow digest length differs")


def _decode_wire_report(report: object, expected_name: str) -> tuple[object, ...]:
    if type(report) is not dict or set(report) != {
        "schema", "numeric_tag", "cbor_hex", "digest"
    }:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "wire report fields differ")
    if (
        report["schema"] != expected_name
        or report["numeric_tag"] != SHADOW_OBJECT_TAGS[expected_name]
    ):
        _fail("FAIL_SHADOW_DOMAIN_COLLISION", "wire report identity differs")
    if type(report["cbor_hex"]) is not str or type(report["digest"]) is not str:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "wire hex fields must be text")
    try:
        encoded = bytes.fromhex(report["cbor_hex"])
        decoded = decode_shadow_object(encoded, expected_name=expected_name)
        value = decoded.value
    except (ValueError, TypeError):
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "wire CBOR is invalid")
    if type(value) is not tuple:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "wire object is not an array")
    validate_shadow_wire_object(expected_name, value)
    if report["digest"] != shadow_object_digest(expected_name, decoded.fields).hex():
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "wire digest differs")
    return value


def _wire_fields(name: str, value: tuple[object, ...]) -> dict[str, object]:
    schema = SHADOW_SCHEMA_REGISTRY[name]
    return dict(zip(schema.fields, value[3:], strict=True))


def _formal_git_id(commit_id: str) -> tuple[int, bytes]:
    return git_sha1_commit_id(bytes.fromhex(_require_commit_id(commit_id)))


def _artifact_blob_and_bindings(
    basis_commit_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate the checked artifact and every source blob at the basis."""

    checked_blob = _git_blob(basis_commit_id, CHECKED_REPORT_RELATIVE_PATH)
    try:
        live_blob = CHECKED_REPORT_PATH.read_bytes()
        report = json.loads(checked_blob.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        _fail("FAIL_SHADOW_BASELINE_DUAL_MISMATCH", "checked errata artifact is invalid")
    if checked_blob != live_blob:
        _fail(
            "FAIL_SHADOW_BASIS_COMMIT_MISMATCH",
            "checked errata artifact differs from its committed blob",
        )
    if type(report) is not dict:
        _fail("FAIL_SHADOW_BASELINE_DUAL_MISMATCH", "checked report is not an object")
    try:
        validate_checked_errata_qualification_report(report)
    except Exception as exc:
        _fail("FAIL_SHADOW_BASELINE_DUAL_MISMATCH", str(exc))
    source_bindings = report.get("source_bindings")
    if type(source_bindings) is not dict or not source_bindings:
        _fail("FAIL_SHADOW_BASELINE_DUAL_MISMATCH", "source bindings are absent")
    for relative, expected in sorted(source_bindings.items()):
        if (
            type(relative) is not str
            or type(expected) is not str
            or _SHA256_RE.fullmatch(expected) is None
        ):
            _fail("FAIL_SHADOW_BASELINE_DUAL_MISMATCH", "source binding is malformed")
        observed = "sha256:" + sha256(_git_blob(basis_commit_id, relative)).hexdigest()
        if observed != expected:
            _fail(
                "FAIL_SHADOW_BASIS_COMMIT_MISMATCH",
                f"basis source differs from checked binding: {relative}",
            )
    implementation_basis = _require_commit_id(report.get("implementation_basis_commit"))
    completed = subprocess.run(
        [
            "/usr/bin/git",
            "merge-base",
            "--is-ancestor",
            implementation_basis,
            basis_commit_id,
        ],
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
        env=_git_environment(),
    )
    if completed.returncode != 0:
        _fail(
            "FAIL_SHADOW_BASIS_COMMIT_MISMATCH",
            "checked implementation basis is not an ancestor of the shadow basis",
        )
    boundary = report.get("authority_boundary")
    if type(boundary) is not dict or any(
        (
            boundary.get("m3_gates_before") != 14,
            boundary.get("m3_gates_after") != 14,
            boundary.get("child_state") != "NOT_RUN",
            boundary.get("formal_roots_generated") is not False,
            boundary.get("m3_run_started") is not False,
        )
    ):
        _fail("FAIL_SHADOW_FORMAL_STATE_MUTATION", "checked formal boundary differs")
    summary: dict[str, object] = {
        "path": CHECKED_REPORT_RELATIVE_PATH,
        "committed_blob_sha256": sha256(checked_blob).hexdigest(),
        "implementation_basis_commit": implementation_basis,
        "diagnostic_report_id": report.get("diagnostic_report_id"),
        "status": report.get("status"),
        "source_binding_count": len(source_bindings),
        "source_bindings_match_shadow_basis": True,
        "checked_artifact_matches_shadow_basis_blob": True,
    }
    return report, summary


def _bound_project_paths(report: Mapping[str, object]) -> tuple[str, ...]:
    source_bindings = report.get("source_bindings")
    assert isinstance(source_bindings, dict)
    result = set(SHADOW_BASIS_PATHS)
    result.update(str(path) for path in source_bindings)
    # The explicit normative bindings may include paths already in SOURCE_PATHS;
    # SHADOW_BASIS_PATHS adds the new owner amendment and orchestration closure.
    return tuple(sorted(result))


def _snapshot_receipt(
    basis_commit_id: str, report: Mapping[str, object]
) -> tuple[dict[str, object], tuple[str, ...]]:
    rows, tree_id = _git_tree_manifest(basis_commit_id)
    manifest_digest = shadow_tree_digest("COMMITTED_HEGEL_SNAPSHOT", rows)
    bound_paths = _bound_project_paths(report)
    bound_rows: list[tuple[object, ...]] = []
    for path in bound_paths:
        payload = _git_blob(basis_commit_id, path)
        bound_rows.append((path.encode("utf-8"), sha256(payload).digest(), len(payload)))
    bound_digest = shadow_tree_digest("BOUND_RUNTIME_INPUT_SET", bound_rows)
    return (
        {
            "basis_commit_id": basis_commit_id,
            "hegel_subtree_git_id": tree_id,
            "entry_count": len(rows),
            "snapshot_manifest_digest": manifest_digest.hex(),
            "bound_input_count": len(bound_rows),
            "bound_input_manifest_digest": bound_digest.hex(),
            "detached_git_objects_only": True,
            "live_worktree_input_count": 0,
            "read_only_materialization_required": True,
        },
        bound_paths,
    )


@contextmanager
def _detached_readonly_inputs(
    basis_commit_id: str, project_relative_paths: Sequence[str]
) -> Iterator[dict[str, Path]]:
    directory = Path(tempfile.mkdtemp(prefix="hegel-shadow-basis-"))
    os.chmod(directory, 0o700)
    inputs: dict[str, Path] = {}
    try:
        for index, relative in enumerate(project_relative_paths):
            payload = _git_blob(basis_commit_id, relative)
            label = f"input_{index:04d}_{sha256(relative.encode('utf-8')).hexdigest()[:16]}"
            target = directory / label
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        _fail("FAIL_SHADOW_SNAPSHOT_NOT_READ_ONLY", "short snapshot write")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.chmod(target, 0o400)
            inputs[label] = target
        os.chmod(directory, 0o500)
        yield inputs
    finally:
        try:
            os.chmod(directory, 0o700)
            for child in directory.iterdir():
                if child.is_file() and not child.is_symlink():
                    os.chmod(child, 0o600)
            shutil.rmtree(directory)
        except OSError:
            _fail("FAIL_SHADOW_SNAPSHOT_NOT_READ_ONLY", "snapshot cleanup failed")


def _json_digest(value: object) -> bytes:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return sha256(encoded).digest()


def _static_plan_digests(snapshot: Mapping[str, object]) -> dict[str, bytes]:
    return {
        "worker_launch_plan": _json_digest(
            {
                "purpose_ids": list(PURPOSE_IDS),
                "distinct_workers": True,
                "isolation_profile_id": ISOLATION_PROFILE_ID,
                "basis_commit_id": snapshot["basis_commit_id"],
                "bound_input_manifest_digest": snapshot["bound_input_manifest_digest"],
            }
        ),
        "fd_policy": _json_digest(
            {
                "fd3": "SEED_TO_CUSTODIAN_CALCULATORS_ONLY",
                "fd4": "ROLE_PHASE_ONLY",
                "fd5": "PUBLIC_ALLOWLIST_ONLY",
            }
        ),
        "output_allowlist": _json_digest(
            {"artifact_kind": ARTIFACT_KIND, "secret_fields": False, "formal_claims": False}
        ),
        "secret_lint_policy": _json_digest(
            {"raw_seed": "FORBIDDEN", "private_key": "FORBIDDEN", "core_dump": "FORBIDDEN"}
        ),
    }


def _call_runtime_probe(
    *,
    basis_commit_id: str,
    shadow_run_id: bytes,
    input_files: Mapping[str, Path],
    python_calculator_path: Path,
    rust_calculator_path: Path,
) -> dict[str, object]:
    try:
        from . import phase3_m3_shadow_runtime_v1 as runtime
    except Exception as exc:
        _fail("FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE", str(exc))
    probe = getattr(runtime, "probe_shadow_admission", None)
    if not callable(probe):
        _fail(
            "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE",
            "side-effect-free runtime admission probe API is unavailable",
        )
    try:
        result = probe(
            basis_commit_id=basis_commit_id,
            shadow_run_id=shadow_run_id,
            input_files=dict(input_files),
            python_calculator_path=python_calculator_path,
            rust_calculator_path=rust_calculator_path,
        )
        _validate_runtime_probe_report(result)
    except ShadowAdmissionError:
        raise
    except Exception as exc:
        _fail("FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE", str(exc))
    if type(result) is not dict:
        _fail("FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE", "probe report is not an object")
    return result


def _validate_runtime_probe_report(report: object) -> None:
    try:
        from . import phase3_m3_shadow_runtime_v1 as runtime

        runtime.validate_shadow_admission_report(report)
    except Exception as exc:
        _fail("FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE", str(exc))


def _probe_pass_and_digest(
    report: Mapping[str, object], expected_run_id: str, expected_basis: str
) -> bytes:
    run_id = report.get("shadow_run_id_hex", report.get("ceremony_id_hex"))
    basis = report.get("basis_commit_id", report.get("basis_commit_id_or_null"))
    passed = report.get("admission_probe_pass", report.get("pass"))
    if passed is None:
        passed = report.get("admission_status") == "INTERNAL_SHADOW_ADMISSION_PASS"
    if run_id != expected_run_id or basis != expected_basis or passed is not True:
        _fail("FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE", "probe identity or result differs")
    boundary = report.get("formal_track", report.get("authority_boundary"))
    if type(boundary) is not dict:
        _fail("FAIL_SHADOW_FORMAL_STATUS_OMITTED", "probe omits formal boundary")
    if "gates_satisfied" in boundary:
        if not (
            boundary.get("gates_satisfied") == 14
            and boundary.get("gates_total") == 24
            and boundary.get("m3_state_name") == "NOT_RUN"
            and boundary.get("formal_roots") is None
        ):
            _fail("FAIL_SHADOW_FORMAL_STATE_MUTATION", "probe formal status differs")
    else:
        roots_unchanged = boundary.get(
            "formal_roots_issued",
            report.get("side_effects", {}).get("formal_root_issued")
            if isinstance(report.get("side_effects"), dict)
            else None,
        )
        if not (
            boundary.get("formal_gate_delta") == 0
            and boundary.get("formal_state_before") == "NOT_RUN"
            and boundary.get("formal_state_after") == "NOT_RUN"
            and roots_unchanged is False
        ):
            _fail("FAIL_SHADOW_FORMAL_STATE_MUTATION", "probe authority boundary differs")
    for forbidden in (
        "seed_generated",
        "key_generated",
        "marker_generated",
        "ledger_generated",
    ):
        if report.get(forbidden, False) is not False:
            _fail("FAIL_SHADOW_SECRET_PERSISTENCE_POLICY", f"probe set {forbidden}")
    side_effects = report.get("side_effects")
    if type(side_effects) is dict and any(item is not False for item in side_effects.values()):
        _fail("FAIL_SHADOW_SECRET_PERSISTENCE_POLICY", "probe side effect was non-false")
    return _json_digest(report)


def _calculator_endpoint_binding(report: Mapping[str, object]) -> dict[str, str]:
    plan = report.get("isolation_plan_inputs")
    if type(plan) is not dict:
        _fail(
            "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE",
            "calculator endpoint isolation plan is absent",
        )
    raw = plan.get("calculator_endpoint_sha256_hex")
    if type(raw) is not dict or set(raw) != {"python", "rust"}:
        _fail(
            "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE",
            "calculator endpoint binding set differs",
        )
    result: dict[str, str] = {}
    for endpoint_id in ("python", "rust"):
        digest = raw[endpoint_id]
        if type(digest) is not str or _DIGEST_RE.fullmatch(digest) is None:
            _fail(
                "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE",
                "calculator endpoint digest differs",
            )
        result[endpoint_id] = digest
    if result["python"] == result["rust"]:
        _fail(
            "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE",
            "calculator endpoint implementation digests collide",
        )
    return result


def _security_probe_wire_set(
    *,
    probe_report: Mapping[str, object],
    shadow_run_id: bytes,
    basis_commit_id: str,
    phase: ShadowProbePhaseId,
    observed_at_unix_seconds: int,
) -> tuple[bytes, list[dict[str, object]]]:
    receipts = probe_report.get("purpose_probe_receipts")
    if type(receipts) is not list or len(receipts) != 4:
        _fail("FAIL_SHADOW_SECURITY_PROBE_SET_INCOMPLETE", "probe receipt set differs")
    fields_rows: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for purpose_id, raw_receipt in zip(PURPOSE_IDS, receipts, strict=True):
        if type(raw_receipt) is not dict or raw_receipt.get("purpose_id") != purpose_id:
            _fail("FAIL_SHADOW_PURPOSE_SET", "probe purpose order differs")
        process = raw_receipt.get("process")
        if type(process) is not dict:
            _fail("FAIL_SHADOW_SECURITY_PROBE_SET_INCOMPLETE", "probe process is absent")
        security = process.get("security_evidence")
        if type(security) is not dict:
            _fail("FAIL_SHADOW_SECURITY_PROBE_SET_INCOMPLETE", "security evidence is absent")
        attack_rows = security.get("attack_syscall_errno_rows")
        if attack_rows != [
            {"attack_id": attack_id, "errno": "EPERM"}
            for attack_id in range(1, 7)
        ]:
            _fail("FAIL_SHADOW_SECCOMP_ATTACK_SYSCALL_NOT_EPERM", "attack rows differ")
        landlock_status = security.get("landlock_status")
        if landlock_status == "ENFORCED":
            landlock_id = 1
            gap = False
        elif landlock_status == "UNAVAILABLE_NONBLOCKING_GAP_DISCLOSED":
            landlock_id = 2
            gap = True
        else:
            _fail("FAIL_SHADOW_LANDLOCK_GAP_NOT_DISCLOSED", "Landlock status differs")
        incident_count = security.get("transient_capability_probe_incident_count")
        if type(incident_count) is not int or incident_count < 0:
            _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "incident count differs")
        fields: dict[str, object] = {
            "artifact_kind_id": ARTIFACT_KIND_ID,
            "shadow_run_id": shadow_run_id,
            "shadow_purpose_id": purpose_id,
            "probe_phase_id": phase.value,
            "worker_instance_id": _json_digest(process)[:16],
            "basis_commit_id": _formal_git_id(basis_commit_id),
            "proc_status_seccomp_value": security.get("seccomp_mode"),
            "proc_status_no_new_privs_value": security.get("no_new_privs"),
            "attack_syscall_errno_rows": tuple((attack_id, 1) for attack_id in range(1, 7)),
            "landlock_status_id": landlock_id,
            "landlock_nonblocking_gap_disclosed": gap,
            "transient_capability_probe_incident_count": incident_count,
            "transient_capability_probe_incident_digest_or_null": None,
            "observed_at_unix_seconds": observed_at_unix_seconds,
            "external_security_attestation_claim": False,
        }
        value = build_shadow_object("ShadowSecurityProbeReceiptV1", fields)
        fields_rows.append(fields)
        reports.append(_wire_report("ShadowSecurityProbeReceiptV1", value))
    digest = shadow_security_probe_set_digest_v1(fields_rows, expected_phase=phase)
    return digest, reports


def build_policy_binding(
    *, basis_commit_id: str, amendment_blob_sha256: bytes
) -> tuple[object, ...]:
    if type(amendment_blob_sha256) is not bytes or len(amendment_blob_sha256) != 32:
        _fail("FAIL_SHADOW_POLICY_NOT_BOUND", "amendment digest must be 32 bytes")
    return _wire_object(
        "ShadowPolicyBindingV1",
        {
            "artifact_kind_id": ARTIFACT_KIND_ID,
            "shadow_track_id": SHADOW_TRACK_ID.encode("ascii"),
            "formal_machine_freeze_id": MACHINE_FREEZE_ID.encode("ascii"),
            "formal_child_dsl_id": CHILD_DSL_ID.encode("ascii"),
            "amendment_git_blob_sha256": amendment_blob_sha256,
            "basis_commit_id": _formal_git_id(basis_commit_id),
        },
    )


def build_isolation_plan(
    *,
    shadow_run_id: bytes,
    basis_commit_id: str,
    snapshot_manifest_digest: bytes,
    worker_launch_plan_digest: bytes,
    required_security_probe_digest: bytes,
    fd_policy_digest: bytes,
    output_allowlist_digest: bytes,
    secret_lint_policy_digest: bytes,
) -> tuple[object, ...]:
    digests = (
        snapshot_manifest_digest,
        worker_launch_plan_digest,
        required_security_probe_digest,
        fd_policy_digest,
        output_allowlist_digest,
        secret_lint_policy_digest,
    )
    if type(shadow_run_id) is not bytes or len(shadow_run_id) != 16 or not any(shadow_run_id):
        _fail("FAIL_SHADOW_PURPOSE_SET", "shadow run ID must be nonzero 16 bytes")
    if any(type(item) is not bytes or len(item) != 32 for item in digests):
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "plan digests must be 32 bytes")
    return _wire_object(
        "ShadowIsolationPlanV1",
        {
            "artifact_kind_id": ARTIFACT_KIND_ID,
            "shadow_run_id": shadow_run_id,
            "basis_commit_id": _formal_git_id(basis_commit_id),
            "snapshot_manifest_digest": snapshot_manifest_digest,
            "purpose_ids": PURPOSE_IDS,
            "isolation_profile_id": ISOLATION_PROFILE_ID,
            "worker_launch_plan_digest": worker_launch_plan_digest,
            "required_security_probe_digest": required_security_probe_digest,
            "fd_policy_digest": fd_policy_digest,
            "output_allowlist_digest": output_allowlist_digest,
            "secret_lint_policy_digest": secret_lint_policy_digest,
            "external_independence_claim": False,
        },
    )


def build_admission_receipt(
    *,
    shadow_run_id: bytes,
    policy_binding_digest: bytes,
    isolation_plan_digest: bytes,
    basis_commit_id: str,
    admitted_at_unix_seconds: int,
) -> tuple[object, ...]:
    return _wire_object(
        "ShadowAdmissionReceiptV1",
        {
            "artifact_kind_id": ARTIFACT_KIND_ID,
            "shadow_run_id": shadow_run_id,
            "policy_binding_digest": policy_binding_digest,
            "isolation_plan_digest": isolation_plan_digest,
            "basis_commit_id": _formal_git_id(basis_commit_id),
            "shadow_gate_bitset": SHADOW_ALL_GATES_BITSET,
            "shadow_gate_count": SHADOW_GATE_COUNT,
            "formal_gates_satisfied": 14,
            "formal_gates_total": 24,
            "formal_m3_state_id": 0,
            "formal_roots_all_null": True,
            "external_actor_evidence": False,
            "admitted_at_unix_seconds": admitted_at_unix_seconds,
        },
    )


def build_state_record(
    *,
    shadow_run_id: bytes,
    transition_index: int,
    previous_state_record_digest: bytes | None,
    from_state_id: int,
    to_state_id: int,
    transition_reason_id: int,
    triggering_shadow_receipt_digest: bytes | None,
    recorded_at_unix_seconds: int,
) -> tuple[object, ...]:
    if (from_state_id, to_state_id, transition_reason_id) not in {(0, 1, 1), (1, 2, 2)}:
        _fail("FAIL_SHADOW_INVALID_TRANSITION", "admission/start transition differs")
    validate_shadow_state_transition(from_state_id, to_state_id, transition_reason_id)
    if transition_index == 0:
        if previous_state_record_digest is not None or (from_state_id, to_state_id) != (0, 1):
            _fail("FAIL_SHADOW_INVALID_TRANSITION", "initial transition linkage differs")
    elif transition_index == 1:
        if (
            type(previous_state_record_digest) is not bytes
            or len(previous_state_record_digest) != 32
        ):
            _fail("FAIL_SHADOW_INVALID_TRANSITION", "start transition lacks previous digest")
    else:
        _fail("FAIL_SHADOW_INVALID_TRANSITION", "unexpected transition index")
    return _wire_object(
        "ShadowStateRecordV1",
        {
            "artifact_kind_id": ARTIFACT_KIND_ID,
            "shadow_run_id": shadow_run_id,
            "transition_index": transition_index,
            "previous_state_record_digest_or_null": previous_state_record_digest,
            "from_shadow_state_id": from_state_id,
            "to_shadow_state_id": to_state_id,
            "transition_reason_id": transition_reason_id,
            "triggering_shadow_receipt_digest_or_null": triggering_shadow_receipt_digest,
            "recorded_at_unix_seconds": recorded_at_unix_seconds,
            "formal_gates_satisfied": 14,
            "formal_gates_total": 24,
            "formal_m3_state_id": 0,
        },
    )


def _gate_results() -> list[dict[str, object]]:
    require_shadow_admission({gate: True for gate in ShadowAdmissionGateId})
    return [
        {"gate_id": index, "gate_name": name, "pass": True}
        for index, name in enumerate(SHADOW_GATE_NAMES, start=1)
    ]


def _claim_lint(value: object, *, path: str = "$", formal_scope: bool = False) -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", f"non-text key at {path}")
            child_path = f"{path}.{key}"
            child_formal = formal_scope or key == "formal_track"
            if key.endswith("_root"):
                _fail("FAIL_SHADOW_FORBIDDEN_CLAIM", f"root-shaped field at {child_path}")
            normalized = key.lower()
            if any(token in normalized for token in ("raw_seed", "private_key", "master_seed")):
                _fail("FAIL_SHADOW_SECRET_MATERIAL_DETECTED_IN_OUTPUT", child_path)
            if (
                key in {"formal_evidence_claim", "external_independence_claim"}
                and child is not False
            ):
                _fail("FAIL_SHADOW_FORBIDDEN_CLAIM", child_path)
            if (
                key in {"active_promotion_allowed", "m3_run_started"}
                and child_formal
                and child is not False
            ):
                _fail("FAIL_SHADOW_FORMAL_STATE_MUTATION", child_path)
            _claim_lint(child, path=child_path, formal_scope=child_formal)
    elif type(value) is list:
        for index, child in enumerate(value):
            _claim_lint(child, path=f"{path}[{index}]", formal_scope=formal_scope)
    elif type(value) not in {str, int, bool, type(None)}:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", f"non-JSON value at {path}")


def _validate_common_artifact(value: object, schema_version: str) -> dict[str, object]:
    if type(value) is not dict:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "artifact must be an object")
    if value.get("schema_version") != schema_version:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "artifact schema differs")
    try:
        validate_shadow_artifact_header(
            artifact_kind_id=value.get("artifact_kind_id"),
            artifact_kind=value.get("artifact_kind"),
            external_independence_claim=value.get("external_independence_claim"),
            formal_evidence_claim=value.get("formal_evidence_claim"),
        )
        formal_snapshot = value.get("formal_track")
        if not isinstance(formal_snapshot, Mapping):
            _fail("FAIL_SHADOW_FORMAL_STATUS_OMITTED", "formal snapshot is absent")
        validate_formal_track_snapshot(formal_snapshot)
    except ShadowAdmissionError:
        raise
    except Exception as exc:
        code = getattr(exc, "code", "FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED")
        _fail(str(code), str(exc))
    if value.get("formal_track_status") != FORMAL_TRACK_STATUS:
        _fail("FAIL_SHADOW_FORMAL_STATE_MUTATION", "formal snapshot differs")
    _claim_lint(value)
    return value


def validate_admission_artifact(value: object) -> None:
    report = _validate_common_artifact(value, SCHEMA_VERSION)
    if set(report) != {
        "schema_version", "artifact_kind_id", "artifact_kind", "formal_track_status",
        "formal_track", "external_independence_claim", "formal_evidence_claim",
        "claim_boundary", "formal_follow_on_recommendation",
        "basis_commit_id", "shadow_run_id", "checked_errata", "snapshot",
        "probe_report", "security_probe_wire_receipts", "gate_results",
        "shadow_track", "wire_objects",
    }:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "admission fields differ")
    if (
        report["claim_boundary"] != CLAIM_BOUNDARY
        or report["formal_follow_on_recommendation"] != FORMAL_FOLLOW_ON_RECOMMENDATION
    ):
        _fail("FAIL_SHADOW_FORBIDDEN_CLAIM", "shadow follow-on boundary differs")
    basis = _require_commit_id(report["basis_commit_id"])
    run_hex = report["shadow_run_id"]
    if type(run_hex) is not str or len(run_hex) != 32:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "run ID differs")
    try:
        run_id = bytes.fromhex(run_hex)
    except ValueError:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "run ID is not hex")
    _validate_runtime_probe_report(report["probe_report"])
    _probe_pass_and_digest(report["probe_report"], run_hex, basis)
    gates = report["gate_results"]
    if gates != _gate_results():
        _fail("FAIL_SHADOW_ADMISSION_INCOMPLETE", "gate registry is not exact 12/12")
    if report["shadow_track"] != {
        "admission_gates": "12/12",
        "admission_gate_bitset": "0x0fff",
        "state_id": 1,
        "state": "ADMITTED_NOT_STARTED",
        "purpose_ids": list(PURPOSE_IDS),
    }:
        _fail("FAIL_SHADOW_ADMISSION_INCOMPLETE", "shadow admission state differs")
    wires = report["wire_objects"]
    if type(wires) is not dict or set(wires) != {
        "policy_binding", "isolation_plan", "admission_receipt", "state_record"
    }:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "admission wire set differs")
    policy = _decode_wire_report(wires["policy_binding"], "ShadowPolicyBindingV1")
    plan = _decode_wire_report(wires["isolation_plan"], "ShadowIsolationPlanV1")
    receipt = _decode_wire_report(wires["admission_receipt"], "ShadowAdmissionReceiptV1")
    state = _decode_wire_report(wires["state_record"], "ShadowStateRecordV1")
    commit_wire = _formal_git_id(basis)
    if policy[8] != commit_wire or plan[4] != run_id or plan[5] != commit_wire:
        _fail("FAIL_SHADOW_BASIS_COMMIT_MISMATCH", "wire basis/run identity differs")
    policy_digest = bytes.fromhex(wires["policy_binding"]["digest"])
    plan_digest = bytes.fromhex(wires["isolation_plan"]["digest"])
    receipt_digest = bytes.fromhex(wires["admission_receipt"]["digest"])
    if receipt[4:8] != (run_id, policy_digest, plan_digest, commit_wire):
        _fail("FAIL_SHADOW_ADMISSION_INCOMPLETE", "admission wire linkage differs")
    if receipt[8:15] != (0x0FFF, 12, 14, 24, 0, True, False):
        _fail("FAIL_SHADOW_FORMAL_STATE_MUTATION", "admission counters differ")
    probe_wires = report["security_probe_wire_receipts"]
    if type(probe_wires) is not list or len(probe_wires) != 4:
        _fail("FAIL_SHADOW_SECURITY_PROBE_SET_INCOMPLETE", "wire probe set differs")
    probe_fields = [
        _wire_fields(
            "ShadowSecurityProbeReceiptV1",
            _decode_wire_report(item, "ShadowSecurityProbeReceiptV1"),
        )
        for item in probe_wires
    ]
    if shadow_security_probe_set_digest_v1(
        probe_fields, expected_phase=ShadowProbePhaseId.ADMISSION_PROBE
    ) != plan[10]:
        _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", "plan probe digest differs")
    observed_times = {item["observed_at_unix_seconds"] for item in probe_fields}
    if len(observed_times) != 1:
        _fail("FAIL_SHADOW_SECURITY_PROBE_SET_INCOMPLETE", "probe times differ")
    recomputed_probe_digest, _ = _security_probe_wire_set(
        probe_report=report["probe_report"],
        shadow_run_id=run_id,
        basis_commit_id=basis,
        phase=ShadowProbePhaseId.ADMISSION_PROBE,
        observed_at_unix_seconds=int(next(iter(observed_times))),
    )
    if recomputed_probe_digest != plan[10]:
        _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", "probe report/wire binding differs")
    if not (
        state[4] == run_id
        and state[5] == 0
        and state[6] is None
        and state[7:11] == (0, 1, 1, receipt_digest)
    ):
        _fail("FAIL_SHADOW_INVALID_TRANSITION", "admission state wire differs")
    if state[-3:] != (14, 24, 0):
        _fail("FAIL_SHADOW_FORMAL_STATE_MUTATION", "state formal counters differ")


def _new_run_id() -> bytes:
    value = os.urandom(16)
    if len(value) != 16 or not any(value):
        _fail("FAIL_SHADOW_EXECUTION", "OS CSPRNG returned an invalid public run ID")
    return value


def admit_internal_shadow(
    *,
    basis_commit_id: str | None = None,
    shadow_run_id: bytes | None = None,
    admitted_at_unix_seconds: int | None = None,
    python_calculator_path: Path | str,
    rust_calculator_path: Path | str,
) -> dict[str, object]:
    """Evaluate 12/12 gates without generating any key, seed, or marker."""

    basis = (
        repository_head_commit()
        if basis_commit_id is None
        else _require_commit_id(basis_commit_id)
    )
    _assert_shadow_execution_basis_stable(basis)
    checked, checked_summary = _artifact_blob_and_bindings(basis)
    snapshot, bound_paths = _snapshot_receipt(basis, checked)
    amendment_blob = _git_blob(basis, AMENDMENT_RELATIVE_PATH)
    policy = build_policy_binding(
        basis_commit_id=basis,
        amendment_blob_sha256=sha256(amendment_blob).digest(),
    )
    policy_report = _wire_report("ShadowPolicyBindingV1", policy)

    run_id = _new_run_id() if shadow_run_id is None else shadow_run_id
    if type(run_id) is not bytes or len(run_id) != 16 or not any(run_id):
        _fail("FAIL_SHADOW_PURPOSE_SET", "caller run ID must be nonzero 16 bytes")
    with _detached_readonly_inputs(basis, bound_paths) as detached_inputs:
        probe_report = _call_runtime_probe(
            basis_commit_id=basis,
            shadow_run_id=run_id,
            input_files=detached_inputs,
            python_calculator_path=Path(python_calculator_path),
            rust_calculator_path=Path(rust_calculator_path),
        )
    _assert_shadow_execution_basis_stable(basis)
    _probe_pass_and_digest(probe_report, run_id.hex(), basis)
    timestamp = int(time.time()) if admitted_at_unix_seconds is None else admitted_at_unix_seconds
    if type(timestamp) is not int or timestamp < 0:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "admission timestamp differs")
    probe_digest, security_probe_wire_receipts = _security_probe_wire_set(
        probe_report=probe_report,
        shadow_run_id=run_id,
        basis_commit_id=basis,
        phase=ShadowProbePhaseId.ADMISSION_PROBE,
        observed_at_unix_seconds=timestamp,
    )
    plan_digests = _static_plan_digests(snapshot)
    plan = build_isolation_plan(
        shadow_run_id=run_id,
        basis_commit_id=basis,
        snapshot_manifest_digest=bytes.fromhex(str(snapshot["snapshot_manifest_digest"])),
        worker_launch_plan_digest=plan_digests["worker_launch_plan"],
        required_security_probe_digest=probe_digest,
        fd_policy_digest=plan_digests["fd_policy"],
        output_allowlist_digest=plan_digests["output_allowlist"],
        secret_lint_policy_digest=plan_digests["secret_lint_policy"],
    )
    plan_report = _wire_report("ShadowIsolationPlanV1", plan)
    receipt = build_admission_receipt(
        shadow_run_id=run_id,
        policy_binding_digest=bytes.fromhex(policy_report["digest"]),
        isolation_plan_digest=bytes.fromhex(plan_report["digest"]),
        basis_commit_id=basis,
        admitted_at_unix_seconds=timestamp,
    )
    receipt_report = _wire_report("ShadowAdmissionReceiptV1", receipt)
    state = build_state_record(
        shadow_run_id=run_id,
        transition_index=0,
        previous_state_record_digest=None,
        from_state_id=0,
        to_state_id=1,
        transition_reason_id=1,
        triggering_shadow_receipt_digest=bytes.fromhex(receipt_report["digest"]),
        recorded_at_unix_seconds=timestamp,
    )
    artifact: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind_id": ARTIFACT_KIND_ID,
        "artifact_kind": ARTIFACT_KIND,
        "formal_track_status": FORMAL_TRACK_STATUS,
        "formal_track": formal_track_snapshot(),
        "external_independence_claim": False,
        "formal_evidence_claim": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "formal_follow_on_recommendation": FORMAL_FOLLOW_ON_RECOMMENDATION,
        "basis_commit_id": basis,
        "shadow_run_id": run_id.hex(),
        "checked_errata": checked_summary,
        "snapshot": snapshot,
        "probe_report": probe_report,
        "security_probe_wire_receipts": security_probe_wire_receipts,
        "gate_results": _gate_results(),
        "shadow_track": {
            "admission_gates": "12/12",
            "admission_gate_bitset": "0x0fff",
            "state_id": 1,
            "state": "ADMITTED_NOT_STARTED",
            "purpose_ids": list(PURPOSE_IDS),
        },
        "wire_objects": {
            "policy_binding": policy_report,
            "isolation_plan": plan_report,
            "admission_receipt": receipt_report,
            "state_record": _wire_report("ShadowStateRecordV1", state),
        },
    }
    validate_admission_artifact(artifact)
    _assert_shadow_execution_basis_stable(basis)
    return artifact


def _runtime_ceremony(
    *,
    state_directory: Path,
    input_files: Mapping[str, Path],
    run_id: bytes,
    python_calculator_path: Path,
    rust_calculator_path: Path,
) -> dict[str, object]:
    try:
        from . import phase3_m3_shadow_runtime_v1 as runtime

        creator = getattr(runtime, "create_shadow_state_directory", None)
        if not state_directory.exists():
            if not callable(creator):
                _fail("FAIL_SHADOW_EXECUTION", "runtime state creator is unavailable")
            creator(state_directory)
        result = runtime.run_internal_shadow_ceremony(
            state_directory=state_directory,
            input_files=dict(input_files),
            python_calculator_path=python_calculator_path,
            rust_calculator_path=rust_calculator_path,
            ceremony_id=run_id,
        )
        _validate_runtime_ceremony_report(result)
    except ShadowAdmissionError:
        raise
    except Exception as exc:
        _fail("FAIL_SHADOW_EXECUTION", str(exc))
    if type(result) is not dict or result.get("ceremony_id_hex") != run_id.hex():
        _fail("FAIL_SHADOW_EXECUTION", "runtime ceremony identity differs")
    return result


def _validate_runtime_ceremony_report(report: object) -> None:
    try:
        from . import phase3_m3_shadow_runtime_v1 as runtime

        runtime.validate_shadow_runtime_report(report)
    except Exception as exc:
        _fail("FAIL_SHADOW_EXECUTION", str(exc))


def _runtime_worker_wire_set(
    *,
    admission: Mapping[str, object],
    runtime_report: Mapping[str, object],
) -> tuple[tuple[bytes, bytes, bytes, bytes], list[dict[str, object]]]:
    envelopes = runtime_report.get("envelopes")
    isolation = runtime_report.get("process_isolation")
    if type(envelopes) is not list or len(envelopes) != 4 or type(isolation) is not dict:
        _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", "runtime worker evidence is absent")
    processes = isolation.get("role_processes")
    if type(processes) is not list or len(processes) != 4:
        _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", "runtime role process set differs")
    snapshot = admission["snapshot"]
    assert isinstance(snapshot, dict)
    run_id = bytes.fromhex(str(admission["shadow_run_id"]))
    basis_wire = _formal_git_id(str(admission["basis_commit_id"]))
    fields_rows: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for purpose_id, (envelope, process) in enumerate(
        zip(envelopes, processes, strict=True), start=1
    ):
        if type(envelope) is not dict or type(process) is not dict:
            _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", "runtime worker row differs")
        try:
            payload = canonical_cbor_decode(bytes.fromhex(str(envelope["payload_cbor_hex"])))
            assert isinstance(payload, tuple)
            worker_instance_id = payload[11]
            key_id = bytes.fromhex(str(envelope["key_id_hex"]))
            public_key = bytes.fromhex(str(envelope["public_key_hex"]))
        except (ValueError, KeyError, AssertionError, IndexError):
            _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", "runtime envelope identity differs")
        security = process.get("security_evidence")
        if type(security) is not dict:
            _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", "worker security evidence is absent")
        fields: dict[str, object] = {
            "artifact_kind_id": ARTIFACT_KIND_ID,
            "shadow_run_id": run_id,
            "shadow_purpose_id": purpose_id,
            "worker_instance_id": worker_instance_id,
            "isolation_profile_id": ISOLATION_PROFILE_ID,
            "basis_commit_id": basis_wire,
            "snapshot_manifest_digest": bytes.fromhex(str(snapshot["snapshot_manifest_digest"])),
            "executable_manifest_digest": _json_digest(
                {"runtime_schema": runtime_report.get("schema_version"), "purpose_id": purpose_id}
            ),
            "environment_manifest_digest": _json_digest(process.get("environment_keys")),
            "namespace_manifest_digest": _json_digest(
                {
                    "namespace_links": security.get("namespace_links"),
                    "namespace_unshared": security.get("namespace_unshared_from_orchestrator"),
                }
            ),
            "ephemeral_key_id": key_id,
            "ephemeral_public_key": public_key,
            "key_epoch": 0,
            "external_independence_claim": False,
        }
        value = build_shadow_object("ShadowPurposeWorkerManifestV1", fields)
        fields_rows.append(fields)
        reports.append(_wire_report("ShadowPurposeWorkerManifestV1", value))
    return shadow_purpose_worker_digest_set_v1(fields_rows), reports


def _runtime_isolation_manifest(
    *,
    admission: Mapping[str, object],
    purpose_worker_digests: tuple[bytes, bytes, bytes, bytes],
    required_security_probe_digest: bytes,
    created_at: int,
) -> tuple[object, ...]:
    run_id = bytes.fromhex(str(admission["shadow_run_id"]))
    basis = str(admission["basis_commit_id"])
    snapshot = admission["snapshot"]
    assert isinstance(snapshot, dict)
    admission_plan = admission["wire_objects"]
    assert isinstance(admission_plan, dict)
    plan = _decode_wire_report(admission_plan["isolation_plan"], "ShadowIsolationPlanV1")
    return _wire_object(
        "ShadowIsolationManifestV1",
        {
            "artifact_kind_id": ARTIFACT_KIND_ID,
            "shadow_run_id": run_id,
            "basis_commit_id": _formal_git_id(basis),
            "snapshot_manifest_digest": bytes.fromhex(str(snapshot["snapshot_manifest_digest"])),
            "purpose_worker_digests": purpose_worker_digests,
            "isolation_invariant_bitset": SHADOW_ISOLATION_INVARIANT_BITSET,
            "required_security_probe_digest": required_security_probe_digest,
            "fd_policy_digest": plan[11],
            "output_allowlist_digest": plan[12],
            "secret_lint_policy_digest": plan[13],
            "created_at_unix_seconds": created_at,
            "external_independence_claim": False,
        },
    )


def _run_genesis(
    *, admission: Mapping[str, object], isolation_manifest_digest: bytes, created_at: int
) -> tuple[object, ...]:
    wires = admission["wire_objects"]
    assert isinstance(wires, dict)
    return _wire_object(
        "ShadowRunGenesisV1",
        {
            "artifact_kind_id": ARTIFACT_KIND_ID,
            "shadow_run_id": bytes.fromhex(str(admission["shadow_run_id"])),
            "policy_binding_digest": bytes.fromhex(str(wires["policy_binding"]["digest"])),
            "admission_receipt_digest": bytes.fromhex(str(wires["admission_receipt"]["digest"])),
            "isolation_manifest_digest": isolation_manifest_digest,
            "basis_commit_id": _formal_git_id(str(admission["basis_commit_id"])),
            "initial_shadow_state_id": 1,
            "canonical_program_archive_digest_or_null": None,
            "program_chunk_manifest_digest_or_null": None,
            "bucket_accounting_digest_or_null": None,
            "odd_output_archive_digest_or_null": None,
            "odd_match_set_digest_or_null": None,
            "odd_role_receipt_digest_or_null": None,
            "sink_output_archive_digest_or_null": None,
            "sink_match_set_digest_or_null": None,
            "sink_role_receipt_digest_or_null": None,
            "dual_replay_agreement_digest_or_null": None,
            "created_at_unix_seconds": created_at,
            "formal_run_genesis_claim": False,
        },
    )


def validate_start_artifact(value: object) -> None:
    report = _validate_common_artifact(value, START_SCHEMA_VERSION)
    if set(report) != {
        "schema_version", "artifact_kind_id", "artifact_kind", "formal_track_status",
        "formal_track", "external_independence_claim", "formal_evidence_claim",
        "claim_boundary", "formal_follow_on_recommendation",
        "basis_commit_id", "shadow_run_id", "admission_artifact_digest",
        "admission_receipt_digest", "admission_state_record_digest",
        "gate_results", "shadow_track", "runtime_report",
        "purpose_worker_wire_manifests", "runtime_security_probe_wire_receipts",
        "wire_objects",
    }:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "start fields differ")
    if (
        report["claim_boundary"] != CLAIM_BOUNDARY
        or report["formal_follow_on_recommendation"] != FORMAL_FOLLOW_ON_RECOMMENDATION
    ):
        _fail("FAIL_SHADOW_FORBIDDEN_CLAIM", "shadow follow-on boundary differs")
    if report["gate_results"] != _gate_results():
        _fail("FAIL_SHADOW_ADMISSION_INCOMPLETE", "start did not revalidate 12/12")
    _validate_runtime_ceremony_report(report["runtime_report"])
    if report["shadow_track"] != {
        "admission_gates": "12/12",
        "state_id": 2,
        "state": "RUNNING_CANONICAL_ENUMERATION",
        "start_action": "phase3-m3-shadow-start",
        "purpose_ids": list(PURPOSE_IDS),
    }:
        _fail("FAIL_SHADOW_START_NOT_EXPLICIT", "start state differs")
    wires = report["wire_objects"]
    if type(wires) is not dict or set(wires) != {
        "isolation_manifest",
        "run_genesis",
        "state_record",
    }:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "start wire set differs")
    isolation = _decode_wire_report(wires["isolation_manifest"], "ShadowIsolationManifestV1")
    genesis = _decode_wire_report(wires["run_genesis"], "ShadowRunGenesisV1")
    state = _decode_wire_report(wires["state_record"], "ShadowStateRecordV1")
    run_id = bytes.fromhex(str(report["shadow_run_id"]))
    basis_wire = _formal_git_id(str(report["basis_commit_id"]))
    if (
        isolation[4:6] != (run_id, basis_wire)
        or isolation[8] != SHADOW_ISOLATION_INVARIANT_BITSET
    ):
        _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", "runtime isolation wire differs")
    worker_wires = report["purpose_worker_wire_manifests"]
    if type(worker_wires) is not list or len(worker_wires) != 4:
        _fail("FAIL_SHADOW_PURPOSE_SET", "worker wire set differs")
    worker_fields = [
        _wire_fields(
            "ShadowPurposeWorkerManifestV1",
            _decode_wire_report(item, "ShadowPurposeWorkerManifestV1"),
        )
        for item in worker_wires
    ]
    if shadow_purpose_worker_digest_set_v1(worker_fields) != tuple(isolation[7]):
        _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", "worker digest set differs")
    runtime_report = report["runtime_report"]
    assert isinstance(runtime_report, dict)
    runtime_envelopes = runtime_report.get("envelopes")
    if type(runtime_envelopes) is not list or len(runtime_envelopes) != 4:
        _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", "runtime envelope set differs")
    for purpose_id, (fields, envelope) in enumerate(
        zip(worker_fields, runtime_envelopes, strict=True), start=1
    ):
        if type(envelope) is not dict:
            _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", "runtime envelope row differs")
        try:
            payload = canonical_cbor_decode(bytes.fromhex(str(envelope["payload_cbor_hex"])))
            assert isinstance(payload, tuple)
            runtime_identity = (
                payload[5], purpose_id, payload[11],
                bytes.fromhex(str(envelope["key_id_hex"])),
                bytes.fromhex(str(envelope["public_key_hex"])),
            )
        except (ValueError, KeyError, AssertionError, IndexError):
            _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", "runtime envelope decode failed")
        wire_identity = (
            fields["shadow_run_id"], fields["shadow_purpose_id"],
            fields["worker_instance_id"], fields["ephemeral_key_id"],
            fields["ephemeral_public_key"],
        )
        if runtime_identity != wire_identity:
            _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", "worker/runtime identity differs")
    probe_wires = report["runtime_security_probe_wire_receipts"]
    if type(probe_wires) is not list or len(probe_wires) != 4:
        _fail("FAIL_SHADOW_SECURITY_PROBE_SET_INCOMPLETE", "runtime probe wire set differs")
    probe_fields = [
        _wire_fields(
            "ShadowSecurityProbeReceiptV1",
            _decode_wire_report(item, "ShadowSecurityProbeReceiptV1"),
        )
        for item in probe_wires
    ]
    if shadow_security_probe_set_digest_v1(
        probe_fields, expected_phase=ShadowProbePhaseId.START_RUNTIME_PROBE
    ) != isolation[9]:
        _fail("FAIL_SHADOW_RUNTIME_SECURITY_PROBE_MISMATCH", "runtime probe digest differs")
    observed_times = {item["observed_at_unix_seconds"] for item in probe_fields}
    fresh_probe = runtime_report.get("fresh_admission_probes")
    if len(observed_times) != 1 or type(fresh_probe) is not dict:
        _fail("FAIL_SHADOW_RUNTIME_SECURITY_PROBE_MISMATCH", "runtime probe binding differs")
    recomputed_probe_digest, _ = _security_probe_wire_set(
        probe_report=fresh_probe,
        shadow_run_id=run_id,
        basis_commit_id=str(report["basis_commit_id"]),
        phase=ShadowProbePhaseId.START_RUNTIME_PROBE,
        observed_at_unix_seconds=int(next(iter(observed_times))),
    )
    if recomputed_probe_digest != isolation[9]:
        _fail("FAIL_SHADOW_RUNTIME_SECURITY_PROBE_MISMATCH", "runtime report/wire probe differs")
    if genesis[4] != run_id or genesis[9] != 1 or any(item is not None for item in genesis[10:20]):
        _fail("FAIL_SHADOW_START_NOT_EXPLICIT", "run genesis differs")
    if genesis[-1] is not False:
        _fail("FAIL_SHADOW_FORBIDDEN_CLAIM", "genesis claimed formal authority")
    previous = report["admission_artifact_digest"]
    if type(previous) is not str or _DIGEST_RE.fullmatch(previous) is None:
        _fail("FAIL_SHADOW_INVALID_TRANSITION", "admission artifact digest differs")
    admission_receipt_digest = report["admission_receipt_digest"]
    admission_state_digest = report["admission_state_record_digest"]
    if (
        type(admission_receipt_digest) is not str
        or _DIGEST_RE.fullmatch(admission_receipt_digest) is None
        or type(admission_state_digest) is not str
        or _DIGEST_RE.fullmatch(admission_state_digest) is None
        or genesis[6] != bytes.fromhex(admission_receipt_digest)
        or state[6] != bytes.fromhex(admission_state_digest)
    ):
        _fail("FAIL_SHADOW_INVALID_TRANSITION", "admission wire linkage differs")
    if state[4] != run_id or state[5] != 1 or state[7:10] != (1, 2, 2):
        _fail("FAIL_SHADOW_INVALID_TRANSITION", "start transition differs")
    if state[-3:] != (14, 24, 0):
        _fail("FAIL_SHADOW_FORMAL_STATE_MUTATION", "start formal counters differ")


def _admission_artifact_digest(admission: Mapping[str, object]) -> str:
    return _json_digest(admission).hex()


def start_internal_shadow(
    admission_artifact: Mapping[str, object],
    *,
    state_directory: Path | str,
    python_calculator_path: Path | str,
    rust_calculator_path: Path | str,
    started_at_unix_seconds: int | None = None,
) -> dict[str, object]:
    """Enter shadow enumeration only while the v1 formal baseline still applies."""

    validate_admission_artifact(admission_artifact)
    basis = _require_commit_id(admission_artifact["basis_commit_id"])
    _assert_shadow_execution_basis_stable(basis)
    checked, _ = _artifact_blob_and_bindings(basis)
    snapshot, bound_paths = _snapshot_receipt(basis, checked)
    if snapshot != admission_artifact["snapshot"]:
        _fail("FAIL_SHADOW_SNAPSHOT_MUTATED", "basis snapshot differs from admission")
    run_id = bytes.fromhex(str(admission_artifact["shadow_run_id"]))
    # Re-run all non-secret admission probes with the same public run ID.
    with _detached_readonly_inputs(basis, bound_paths) as detached_inputs:
        probe = _call_runtime_probe(
            basis_commit_id=basis,
            shadow_run_id=run_id,
            input_files=detached_inputs,
            python_calculator_path=Path(python_calculator_path),
            rust_calculator_path=Path(rust_calculator_path),
        )
    _assert_shadow_execution_basis_stable(basis)
    _probe_pass_and_digest(probe, run_id.hex(), basis)
    admitted_probe = admission_artifact["probe_report"]
    assert isinstance(admitted_probe, dict)
    if _calculator_endpoint_binding(probe) != _calculator_endpoint_binding(
        admitted_probe
    ):
        _fail(
            "FAIL_SHADOW_RUNTIME_PLAN_MISMATCH",
            "start calculator endpoints differ from admission",
        )
    with _detached_readonly_inputs(basis, bound_paths) as detached_inputs:
        runtime_report = _runtime_ceremony(
            state_directory=Path(state_directory),
            input_files=detached_inputs,
            run_id=run_id,
            python_calculator_path=Path(python_calculator_path),
            rust_calculator_path=Path(rust_calculator_path),
        )
    _assert_shadow_execution_basis_stable(basis)
    timestamp = int(time.time()) if started_at_unix_seconds is None else started_at_unix_seconds
    if type(timestamp) is not int or timestamp < 0:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "start timestamp differs")
    fresh_runtime_probe = runtime_report.get("fresh_admission_probes")
    if type(fresh_runtime_probe) is not dict:
        _fail("FAIL_SHADOW_RUNTIME_SECURITY_PROBE_MISMATCH", "runtime reprobe is absent")
    if _calculator_endpoint_binding(fresh_runtime_probe) != _calculator_endpoint_binding(
        admitted_probe
    ):
        _fail(
            "FAIL_SHADOW_RUNTIME_SECURITY_PROBE_MISMATCH",
            "runtime calculator endpoints differ from admission",
        )
    runtime_probe_digest, runtime_security_probe_wire_receipts = _security_probe_wire_set(
        probe_report=fresh_runtime_probe,
        shadow_run_id=run_id,
        basis_commit_id=basis,
        phase=ShadowProbePhaseId.START_RUNTIME_PROBE,
        observed_at_unix_seconds=timestamp,
    )
    worker_digests, purpose_worker_wire_manifests = _runtime_worker_wire_set(
        admission=admission_artifact,
        runtime_report=runtime_report,
    )
    isolation = _runtime_isolation_manifest(
        admission=admission_artifact,
        purpose_worker_digests=worker_digests,
        required_security_probe_digest=runtime_probe_digest,
        created_at=timestamp,
    )
    isolation_report = _wire_report("ShadowIsolationManifestV1", isolation)
    genesis = _run_genesis(
        admission=admission_artifact,
        isolation_manifest_digest=bytes.fromhex(isolation_report["digest"]),
        created_at=timestamp,
    )
    genesis_report = _wire_report("ShadowRunGenesisV1", genesis)
    admission_wires = admission_artifact["wire_objects"]
    assert isinstance(admission_wires, dict)
    state = build_state_record(
        shadow_run_id=run_id,
        transition_index=1,
        previous_state_record_digest=bytes.fromhex(admission_wires["state_record"]["digest"]),
        from_state_id=1,
        to_state_id=2,
        transition_reason_id=2,
        triggering_shadow_receipt_digest=bytes.fromhex(genesis_report["digest"]),
        recorded_at_unix_seconds=timestamp,
    )
    artifact: dict[str, object] = {
        "schema_version": START_SCHEMA_VERSION,
        "artifact_kind_id": ARTIFACT_KIND_ID,
        "artifact_kind": ARTIFACT_KIND,
        "formal_track_status": FORMAL_TRACK_STATUS,
        "formal_track": formal_track_snapshot(),
        "external_independence_claim": False,
        "formal_evidence_claim": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "formal_follow_on_recommendation": FORMAL_FOLLOW_ON_RECOMMENDATION,
        "basis_commit_id": basis,
        "shadow_run_id": run_id.hex(),
        "admission_artifact_digest": _admission_artifact_digest(admission_artifact),
        "admission_receipt_digest": str(admission_wires["admission_receipt"]["digest"]),
        "admission_state_record_digest": str(admission_wires["state_record"]["digest"]),
        "gate_results": _gate_results(),
        "purpose_worker_wire_manifests": purpose_worker_wire_manifests,
        "runtime_security_probe_wire_receipts": runtime_security_probe_wire_receipts,
        "shadow_track": {
            "admission_gates": "12/12",
            "state_id": 2,
            "state": "RUNNING_CANONICAL_ENUMERATION",
            "start_action": "phase3-m3-shadow-start",
            "purpose_ids": list(PURPOSE_IDS),
        },
        "runtime_report": runtime_report,
        "wire_objects": {
            "isolation_manifest": isolation_report,
            "run_genesis": genesis_report,
            "state_record": _wire_report("ShadowStateRecordV1", state),
        },
    }
    validate_start_artifact(artifact)
    _assert_shadow_execution_basis_stable(basis)
    return artifact


def load_admission_artifact(path: Path | str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail("FAIL_SHADOW_ADMISSION_INCOMPLETE", str(exc))
    validate_admission_artifact(value)
    assert isinstance(value, dict)
    return value


def write_json_exclusive(path: Path | str, value: Mapping[str, object]) -> Path:
    """Publish one JSON artifact once; never overwrite an earlier run."""

    target = Path(path)
    if target.is_symlink():
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "artifact path is a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(target, flags, 0o644)
    except FileExistsError:
        _fail("FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED", "artifact already exists")
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                _fail("FAIL_SHADOW_EXECUTION", "short artifact write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return target


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_KIND_ID",
    "DEFAULT_ADMISSION_ARTIFACT_PATH",
    "DEFAULT_START_ARTIFACT_PATH",
    "FORMAL_TRACK_STATUS",
    "SCHEMA_VERSION",
    "SHADOW_GATE_NAMES",
    "START_SCHEMA_VERSION",
    "ShadowAdmissionError",
    "admit_internal_shadow",
    "build_admission_receipt",
    "build_isolation_plan",
    "build_policy_binding",
    "build_state_record",
    "formal_track_snapshot",
    "load_admission_artifact",
    "repository_head_commit",
    "shadow_digest",
    "shadow_tree_digest",
    "start_internal_shadow",
    "validate_admission_artifact",
    "validate_shadow_wire_object",
    "validate_start_artifact",
    "write_json_exclusive",
]
