"""Formal M3 start-state consumption and dual-enumeration publication.

This module joins the exact Commit-B public replay, the persisted index-zero
state, the Commit-A implementation qualification, and the offline Docker
runner.  A successful run publishes one exact dual agreement and one index-one
``DSL_TOO_LARGE`` terminal state.  The host replays public frozen target
definitions and split commitments while the enumerator containers receive
neither target inputs nor split inputs and role evaluation remains closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from types import MappingProxyType
from typing import Final, Mapping, NoReturn

from . import phase3_m3_implementation_qualification_v1 as _qualification
from .phase3_m25_commit_b_publication_audit_v1 import canonical_json_v1
from .phase3_m25_external_v1 import assert_public_payload_contains_no_secret_fields
from .phase3_m25_formal_static_basis_v1 import REPOSITORY_ROOT, _git_blob
from .phase3_m25_wire_v1 import (
    candidate_content_root,
    decode_formal_object,
    encode_formal_object,
    id_digest_v1,
    validate_timestamp_ordering_v1,
)
from .phase3_m3_dual_enumeration_supervisor_v1 import (
    COMMIT_A,
    CANONICAL_PROGRAM_BUDGET,
    FAIL_DUAL,
    FROZEN_IMPLEMENTATIONS,
    RAW_OPERATOR_APPLICATION_CAP,
    EnumerationInvocationV1,
    M3DualEnumerationOutcomeV1,
    _run_m3_dual_enumeration_core_v1,
)
from .phase3_m3_offline_docker_runner_v1 import (
    ATTEMPT_INTENT_SCHEMA,
    COMPLETION_MARKER_SCHEMA,
    FAILURE_CLEANUP_POLICY,
    FAIL_TERMINALIZE,
    JOURNAL_COMPLETION_SCHEMA,
    MAX_ENUMERATION_SECONDS,
    PROBE_COMPLETION_SCHEMA,
    PROBE_START_SCHEMA,
    PYTHON_PROBE_MAXIMUM_SECONDS,
    RESTART_POLICY,
    START_MARKER_SCHEMA,
    OfflineDockerEnumerationRunnerV1,
    _invocation_digest_v1,
)
from . import phase3_m3_local_admission_v1 as _local_admission
from .phase3_m3_start_v1 import (
    FORMAL_RUN_ID_HEX,
    PUBLICATION_COMMIT_B,
    ReplayedGatePublicationV1,
    canonical_run_root_v1,
    canonical_start_state_path_v1,
    canonical_terminal_outcome_path_v1,
    read_start_publication_receipt_v1,
    read_state_file_v1,
    replay_gate_publication_v1,
    require_canonical_start_state_path_v1,
    strict_json_loads_v1,
    validate_state_document_v1,
    verify_m3_start_v1,
)


SCHEMA: Final = "hegel-phase3-m3-formal-enumeration-outcome/1"
ARTIFACT_KIND: Final = "FORMAL_M3_DUAL_ENUMERATION_TERMINAL"
FAILURE_SCHEMA: Final = "hegel-phase3-m3-formal-failure-outcome/1"
FAILURE_ARTIFACT_KIND: Final = "FORMAL_M3_INCONCLUSIVE_TERMINAL"
QUALIFICATION_RECEIPT_PATH: Final = (
    "Hegel Machine/artifacts/phase3_m25_external/"
    "phase3_m3_implementation_qualification_v1.json"
)
MAX_OUTCOME_BYTES: Final = 16 * 1024 * 1024

FAIL_INPUT = "FAIL_M3_FORMAL_EXECUTION_INPUT"
FAIL_BINDING = "FAIL_M3_FORMAL_EXECUTION_BINDING"
FAIL_OUTPUT = "FAIL_M3_FORMAL_EXECUTION_OUTPUT"
FAIL_OUTCOME = "FAIL_M3_FORMAL_EXECUTION_OUTCOME"
FAIL_ALREADY_TERMINAL = "FAIL_M3_FORMAL_EXECUTION_ALREADY_TERMINAL"

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
CANONICAL_ATTEMPT_ID: Final = "attempt-1"
_EXECUTION_LOCK_NAME: Final = ".phase3-m3-formal-execution.lock"
_PENDING_NAME_ATTEMPTS: Final = 16
_PREPARED_SEAL = object()


class M3FormalExecutionError(RuntimeError):
    """Stable fail-closed error for formal enumeration orchestration."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise M3FormalExecutionError(code, detail)


@dataclass(frozen=True, slots=True)
class PreparedM3FormalExecutionV1:
    repository_root: Path
    persisted_start_path: Path
    replay: ReplayedGatePublicationV1
    start_document: Mapping[str, object]
    start_record_fields: Mapping[str, object]
    start_record_root: bytes
    start_publication_receipt: Mapping[str, object]
    implementation_qualification_receipt: Mapping[str, object]
    committed_golden: Mapping[str, object]
    local_admission_artifact: Mapping[str, object]
    runtime_source_manifest: Mapping[str, object]
    local_admission_receipt: Mapping[str, object]
    _seal: object


@dataclass(frozen=True, slots=True)
class FormalExecutionPublicationV1:
    status: str
    document: Mapping[str, object]
    outcome_path: Path
    attempt_root: Path | None


def _strict_committed_json(payload: bytes, *, label: str) -> Mapping[str, object]:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(FAIL_INPUT, f"{label} repeats a JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", "strict"), object_pairs_hook=no_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(FAIL_INPUT, f"{label} is not strict JSON: {type(exc).__name__}")
    if type(value) is not dict:
        _fail(FAIL_INPUT, f"{label} must be an object")
    return value


def _stable_file_identity(path: Path, *, maximum_bytes: int) -> tuple[int, str]:
    descriptor: int | None = None
    try:
        lexical = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        before = os.fstat(descriptor)
        if (
            stat.S_ISLNK(lexical.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or (lexical.st_dev, lexical.st_ino) != (before.st_dev, before.st_ino)
            or before.st_size > maximum_bytes
        ):
            _fail(FAIL_OUTCOME, f"archive file identity differs: {path.name}")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1_048_576))
            if not chunk:
                _fail(FAIL_OUTCOME, f"archive file ended early: {path.name}")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(FAIL_OUTCOME, f"archive file grew while read: {path.name}")
        after = os.fstat(descriptor)
        namespace = path.lstat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mode,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (namespace.st_dev, namespace.st_ino) != (
            after.st_dev,
            after.st_ino,
        ):
            _fail(FAIL_OUTCOME, f"archive file changed while read: {path.name}")
        return before.st_size, digest.hexdigest()
    except M3FormalExecutionError:
        raise
    except OSError as exc:
        _fail(FAIL_OUTCOME, f"archive file cannot be read: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _stable_file_payload_v1(path: Path, *, maximum_bytes: int) -> bytes:
    descriptor: int | None = None
    try:
        lexical = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        before = os.fstat(descriptor)
        if (
            stat.S_ISLNK(lexical.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or (lexical.st_dev, lexical.st_ino) != (before.st_dev, before.st_ino)
            or before.st_size > maximum_bytes
        ):
            _fail(FAIL_OUTCOME, f"evidence file identity differs: {path.name}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1_048_576))
            if not chunk:
                _fail(FAIL_OUTCOME, f"evidence file ended early: {path.name}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(FAIL_OUTCOME, f"evidence file grew while read: {path.name}")
        after = os.fstat(descriptor)
        namespace = path.lstat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mode,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (namespace.st_dev, namespace.st_ino) != (
            after.st_dev,
            after.st_ino,
        ):
            _fail(FAIL_OUTCOME, f"evidence file changed while read: {path.name}")
        return b"".join(chunks)
    except M3FormalExecutionError:
        raise
    except OSError as exc:
        _fail(FAIL_OUTCOME, f"evidence file cannot be read: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_canonical_runner_json_v1(
    path: Path,
    *,
    expected_fields: set[str],
) -> dict[str, object]:
    payload = _stable_file_payload_v1(path, maximum_bytes=1_048_576)
    value = strict_json_loads_v1(payload, label=f"runner evidence {path.name}")
    if canonical_json_v1(value) != payload or set(value) != expected_fields:
        _fail(FAIL_OUTCOME, f"runner evidence schema differs: {path.name}")
    return value


def prepare_formal_execution_v1(
    state_path: Path,
    evidence_bytes: bytes,
    promotion_bytes: bytes,
    *,
    repository: Path = REPOSITORY_ROOT,
    publication_commit: str = PUBLICATION_COMMIT_B,
    expected_admission_revision: str,
) -> PreparedM3FormalExecutionV1:
    """Verify all immutable inputs without starting an enumerator."""

    repository = repository.resolve(strict=True)
    try:
        require_canonical_start_state_path_v1(state_path, FORMAL_RUN_ID_HEX)
        state_bytes = read_state_file_v1(state_path)
        state = strict_json_loads_v1(state_bytes, label="M3 start state")
        if canonical_json_v1(state) != state_bytes:
            _fail(FAIL_INPUT, "M3 start state is not canonical JSON")
        validate_state_document_v1(state)
        start_publication_receipt = read_start_publication_receipt_v1(
            state_path,
            state,
        )
        verified = verify_m3_start_v1(
            state_bytes,
            evidence_bytes,
            promotion_bytes,
            repository=repository,
            publication_commit=publication_commit,
        )
        replay = replay_gate_publication_v1(
            evidence_bytes,
            promotion_bytes,
            repository=repository,
            publication_commit=publication_commit,
        )
    except M3FormalExecutionError:
        raise
    except Exception as exc:
        _fail(FAIL_INPUT, f"start/publication replay failed: {exc}")
    if verified != state or replay.publication_commit != PUBLICATION_COMMIT_B:
        _fail(FAIL_BINDING, "start state does not bind the authoritative Commit-B")
    require_canonical_start_state_path_v1(state_path, state["run_id_hex"])
    try:
        start_cbor = bytes.fromhex(str(state["state_record_cbor_hex"]))
        decoded_start = decode_formal_object(
            start_cbor, expected_name="M3RunStateRecordV1"
        )
        start_root = candidate_content_root(
            "M3RunStateRecordV1", decoded_start.fields
        )
    except Exception as exc:
        _fail(FAIL_INPUT, f"start state formal object failed replay: {exc}")
    if start_root.hex() != state.get("state_record_root_hex"):
        _fail(FAIL_BINDING, "start state root differs")

    qualification = _strict_committed_json(
        _git_blob(repository, PUBLICATION_COMMIT_B, QUALIFICATION_RECEIPT_PATH),
        label="Commit-B-published Commit-A implementation qualification receipt",
    )
    try:
        golden, _preimage, _root = _qualification.load_committed_dual_golden_v1(
            repository, COMMIT_A
        )
        _qualification.validate_qualification_receipt_v1(
            qualification, golden=golden, basis_commit=COMMIT_A
        )
    except Exception as exc:
        _fail(FAIL_BINDING, f"Commit-A implementation qualification differs: {exc}")
    candidate = replay.gate_inputs.execution_candidate_fields
    for name, field in (
        ("python", "python_implementation_binding_root"),
        ("rust", "rust_implementation_binding_root"),
    ):
        row = qualification.get(name)
        if (
            not isinstance(row, Mapping)
            or row.get("implementation_binding_root")
            != FROZEN_IMPLEMENTATIONS[name].implementation_binding_root.hex()
            or candidate.get(field)
            != FROZEN_IMPLEMENTATIONS[name].implementation_binding_root
        ):
            _fail(FAIL_BINDING, f"{name} candidate/qualification binding differs")
    try:
        local_admission = _local_admission.validate_live_local_admission_v1(
            expected_admission_revision,
            repository_root=repository,
        )

        def plain(value: Mapping[str, object], *, label: str) -> dict[str, object]:
            return dict(
                _strict_committed_json(
                    _local_admission.canonical_json_v1(value),
                    label=label,
                )
            )

        admission_artifact = plain(
            local_admission.artifact_fields,
            label="live local admission artifact",
        )
        runtime_manifest = plain(
            local_admission.manifest_fields,
            label="live local admission runtime manifest",
        )
        admission_receipt = plain(
            local_admission.receipt_fields,
            label="live local admission receipt",
        )
    except Exception as exc:
        _fail(FAIL_BINDING, f"local two-commit admission failed replay: {exc}")
    if (
        start_publication_receipt.get("local_admission_artifact")
        != admission_artifact
        or start_publication_receipt.get("local_admission_runtime_manifest")
        != runtime_manifest
        or start_publication_receipt.get("local_admission_receipt")
        != admission_receipt
    ):
        _fail(FAIL_BINDING, "explicit start does not bind the live local admission")
    return PreparedM3FormalExecutionV1(
        repository_root=repository,
        persisted_start_path=state_path,
        replay=replay,
        start_document=MappingProxyType(dict(state)),
        start_record_fields=MappingProxyType(dict(decoded_start.fields)),
        start_record_root=start_root,
        start_publication_receipt=MappingProxyType(
            dict(start_publication_receipt)
        ),
        implementation_qualification_receipt=MappingProxyType(dict(qualification)),
        committed_golden=golden,
        local_admission_artifact=MappingProxyType(admission_artifact),
        runtime_source_manifest=MappingProxyType(runtime_manifest),
        local_admission_receipt=MappingProxyType(admission_receipt),
        _seal=_PREPARED_SEAL,
    )


def _formal_entry(name: str, fields: Mapping[str, object]) -> dict[str, object]:
    cbor = encode_formal_object(name, fields)
    decoded = decode_formal_object(cbor, expected_name=name)
    if encode_formal_object(name, decoded.fields) != cbor:
        _fail(FAIL_OUTCOME, f"{name} CBOR round trip differs")
    root = candidate_content_root(name, fields)
    return {
        "schema_name": name,
        "cbor_hex": cbor.hex(),
        "content_root_hex": root.hex(),
    }


def _validated_local_admission_identity_v1(
    prepared: PreparedM3FormalExecutionV1,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    artifact = prepared.local_admission_artifact
    manifest = prepared.runtime_source_manifest
    receipt = prepared.local_admission_receipt
    try:
        _local_admission.validate_local_admission_receipt_v1(
            receipt,
            artifact_fields=artifact,
            manifest_fields=manifest,
        )
    except Exception as exc:
        _fail(FAIL_BINDING, f"local admission receipt failed replay: {exc}")
    rows = manifest.get("runtime_source_files")
    observed_paths = {
        row.get("repository_path")
        for row in rows
        if type(row) is dict
    } if type(rows) is list else set()
    if observed_paths != set(_local_admission.M3_RUNTIME_SOURCE_PATHS):
        _fail(FAIL_BINDING, "local admission runtime path closure differs")
    return dict(artifact), dict(manifest), dict(receipt)


def _replay_live_local_admission_identity_v1(
    prepared: PreparedM3FormalExecutionV1,
) -> None:
    expected_artifact, expected_manifest, expected_receipt = (
        _validated_local_admission_identity_v1(
        prepared
        )
    )
    revision = expected_receipt.get("approval_commit_d")
    if type(revision) is not str:
        _fail(FAIL_BINDING, "local admission Commit D identity is absent")
    try:
        observed = _local_admission.validate_live_local_admission_v1(
            revision,
            repository_root=prepared.repository_root,
        )
    except Exception as exc:
        _fail(FAIL_BINDING, f"live local admission replay failed: {exc}")
    if (
        _local_admission.canonical_json_v1(observed.artifact_fields)
        != _local_admission.canonical_json_v1(expected_artifact)
        or _local_admission.canonical_json_v1(observed.manifest_fields)
        != _local_admission.canonical_json_v1(expected_manifest)
        or _local_admission.canonical_json_v1(observed.receipt_fields)
        != _local_admission.canonical_json_v1(expected_receipt)
    ):
        _fail(FAIL_BINDING, "live local admission identity changed")


def _replay_live_start_publication_v1(
    prepared: PreparedM3FormalExecutionV1,
) -> None:
    """Re-read the canonical state/sidecar pair and require exact identity."""

    try:
        state_payload = read_state_file_v1(prepared.persisted_start_path)
        state = strict_json_loads_v1(state_payload, label="live M3 start state")
        if canonical_json_v1(state) != state_payload:
            raise ValueError("start state is not canonical JSON")
        validate_state_document_v1(state)
        receipt = read_start_publication_receipt_v1(
            prepared.persisted_start_path,
            state,
        )
    except Exception as exc:
        _fail(FAIL_BINDING, f"live explicit-start publication failed replay: {exc}")
    if (
        state != dict(prepared.start_document)
        or receipt != dict(prepared.start_publication_receipt)
    ):
        _fail(FAIL_BINDING, "live explicit-start publication changed")


def _archive_manifest(output_root: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    expected = {
        "archive/report.json",
        "archive/canonical_program_records.cborframed",
        "archive/program_chunk_manifests.cborframed",
        "archive/bucket_accounting_records.cborframed",
        "execution-stdout.json",
        "execution-stderr.bin",
        "process-completion.json",
    }
    try:
        output_metadata = output_root.lstat()
        output_entries = tuple(output_root.iterdir())
    except OSError as exc:
        _fail(FAIL_OUTCOME, f"formal enumeration root is unavailable: {exc}")
    if (
        stat.S_ISLNK(output_metadata.st_mode)
        or not stat.S_ISDIR(output_metadata.st_mode)
        or {entry.name for entry in output_entries} != {"python", "rust"}
        or any(entry.is_symlink() or not entry.is_dir() for entry in output_entries)
    ):
        _fail(FAIL_OUTCOME, "formal enumeration implementation set differs")
    for implementation in ("python", "rust"):
        root = output_root / implementation
        root_metadata = root.lstat()
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
            root_metadata.st_mode
        ):
            _fail(FAIL_OUTCOME, f"{implementation} output directory differs")
        expected_nodes = {"archive", *expected}
        observed_nodes = {
            path.relative_to(root).as_posix(): path for path in root.rglob("*")
        }
        if set(observed_nodes) != expected_nodes:
            _fail(FAIL_OUTCOME, f"{implementation} persisted file set differs")
        archive_metadata = observed_nodes["archive"].lstat()
        if (
            stat.S_ISLNK(archive_metadata.st_mode)
            or not stat.S_ISDIR(archive_metadata.st_mode)
        ):
            _fail(FAIL_OUTCOME, f"{implementation} archive directory differs")
        rows: list[dict[str, object]] = []
        for relative in sorted(expected):
            path = observed_nodes[relative]
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                _fail(FAIL_OUTCOME, f"{implementation} archive contains a non-regular file")
            byte_length, digest = _stable_file_identity(
                path, maximum_bytes=128 * 1024 * 1024
            )
            rows.append(
                {
                    "relative_path": relative,
                    "byte_length": byte_length,
                    "sha256": digest,
                }
            )
        result[implementation] = rows
    return result


def _runner_evidence_manifest(attempt_root: Path) -> list[dict[str, object]]:
    expected = {
        "python-runtime-probe-stdout.json",
        "runner-attempt-intent.json",
        "runner-preflight.json",
        "runner-journal/python-probe-started.json",
        "runner-journal/python-probe-completed.json",
        "runner-journal/python-started.json",
        "runner-journal/python-completed.json",
        "runner-journal/rust-started.json",
        "runner-journal/rust-completed.json",
    }
    expected_top_nodes = {
        "immutable-inputs",
        "formal-enumeration",
        "runner-journal",
        "python-runtime-probe-stdout.json",
        "runner-attempt-intent.json",
        "runner-preflight.json",
    }
    top_nodes = {path.name: path for path in attempt_root.iterdir()}
    if set(top_nodes) != expected_top_nodes:
        _fail(FAIL_OUTCOME, "formal attempt top-level node set differs")
    for directory_name in (
        "immutable-inputs",
        "formal-enumeration",
        "runner-journal",
    ):
        metadata = top_nodes[directory_name].lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            _fail(FAIL_OUTCOME, f"formal attempt directory differs: {directory_name}")
    for filename in (
        "python-runtime-probe-stdout.json",
        "runner-attempt-intent.json",
        "runner-preflight.json",
    ):
        metadata = top_nodes[filename].lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            _fail(FAIL_OUTCOME, f"formal attempt file differs: {filename}")
    journal_root = attempt_root / "runner-journal"
    journal_metadata = journal_root.lstat()
    if (
        stat.S_ISLNK(journal_metadata.st_mode)
        or not stat.S_ISDIR(journal_metadata.st_mode)
    ):
        _fail(FAIL_OUTCOME, "formal runner journal directory differs")
    observed = {
        path.relative_to(attempt_root).as_posix()
        for path in journal_root.iterdir()
    }
    observed.update(
        path.name
        for path in attempt_root.iterdir()
        if path.name != "runner-journal"
        and (
            path.name.startswith("runner-")
            or path.name == "python-runtime-probe-stdout.json"
        )
    )
    if observed != expected:
        _fail(FAIL_OUTCOME, "formal runner evidence file set differs")
    rows: list[dict[str, object]] = []
    for relative in sorted(expected):
        metadata = (attempt_root / relative).lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            _fail(FAIL_OUTCOME, "formal runner evidence contains a non-regular node")
        size, digest = _stable_file_identity(
            attempt_root / relative,
            maximum_bytes=1_048_576,
        )
        rows.append(
            {
                "relative_path": relative,
                "byte_length": size,
                "sha256": digest,
            }
        )
    return rows


def _runtime_golden_v1(
    prepared: PreparedM3FormalExecutionV1,
    report: Mapping[str, object],
) -> Mapping[str, object]:
    expected = dict(prepared.committed_golden["expected"])  # type: ignore[arg-type]
    expected["canonical_program_archive_root"] = report.get(
        "canonical_program_archive_root_or_null"
    )
    expected["program_chunk_manifest_root"] = report.get(
        "program_chunk_manifest_root_or_null"
    )
    roots = prepared.replay.qualified_gate_evidence.formal_roots
    value = dict(prepared.committed_golden)
    value["binding_roots"] = {
        name: roots[name].hex()
        for name in (
            "child_dsl_spec_root",
            "operator_semantics_root",
            "identifier_registry_root",
        )
    }
    value["expected"] = expected
    return MappingProxyType(value)


def _replay_persisted_enumerator_archives_v1(
    *,
    prepared: PreparedM3FormalExecutionV1,
    attempt_root: Path,
    receipts: Mapping[str, Mapping[str, object]],
    archive_manifest: Mapping[str, object],
) -> None:
    """Replay both complete archives again from a published diagnostic outcome."""

    roots = prepared.replay.qualified_gate_evidence.formal_roots
    binding_roots = (
        roots["child_dsl_spec_root"],
        roots["operator_semantics_root"],
        roots["identifier_registry_root"],
    )
    archives: dict[str, Mapping[str, object]] = {}
    closure_status_ids: set[int] = set()
    for implementation in ("python", "rust"):
        implementation_root = (
            attempt_root / "formal-enumeration" / implementation
        )
        stdout_path = implementation_root / "execution-stdout.json"
        manifest_rows = archive_manifest.get(implementation)
        if type(manifest_rows) is not list:
            _fail(FAIL_OUTCOME, f"{implementation} archive manifest is absent")
        manifest_by_path = {
            row.get("relative_path"): row
            for row in manifest_rows
            if type(row) is dict
        }

        def bind_consumed_payload(relative_path: str, payload: bytes) -> None:
            row = manifest_by_path.get(relative_path)
            if (
                type(row) is not dict
                or row.get("byte_length") != len(payload)
                or row.get("sha256") != hashlib.sha256(payload).hexdigest()
            ):
                _fail(
                    FAIL_OUTCOME,
                    f"{implementation} consumed bytes differ at {relative_path}",
                )

        try:
            stdout_payload = _qualification._read_regular_file(
                stdout_path,
                maximum_bytes=1_048_576,
                label=f"formal {implementation} persisted stdout",
            )
            bind_consumed_payload("execution-stdout.json", stdout_payload)
            report = _qualification._parse_single_json(
                stdout_payload,
                label=f"formal {implementation} persisted report",
            )
            receipt = receipts[implementation]
            closure_status_id = receipt.get("closure_status_id")
            if closure_status_id != 2:
                raise ValueError("unsupported persisted closure status")
            validated = _qualification.validate_enumerator_report_v1(
                dict(report),
                implementation=implementation,
                golden=_runtime_golden_v1(prepared, report),
            )
        except Exception as exc:
            _fail(
                FAIL_OUTCOME,
                f"{implementation} persisted report failed replay: {exc}",
            )
        closure_status_ids.add(closure_status_id)

        def report_hex_or_null(field: str) -> str | None:
            value = receipt[field]
            if value is None:
                return None
            if type(value) is not bytes or len(value) != 32:
                _fail(
                    FAIL_OUTCOME,
                    f"{implementation} receipt digest differs at {field}",
                )
            return value.hex()

        report_receipt_values = {
            "closure_status_id": closure_status_id,
            "raw_operator_application_count": receipt[
                "raw_operator_application_count"
            ],
            "canonical_program_count": receipt["canonical_program_count"],
            "closure_cardinality_or_null": receipt[
                "closure_cardinality_or_null"
            ],
            "frontier_exhausted": receipt["frontier_exhausted"],
            "all_type_buckets_closed": receipt["all_type_buckets_closed"],
            "raw_expansion_limit_hit": receipt["raw_expansion_limit_hit"],
            "wall_clock_abort_hit": receipt["wall_clock_abort_hit"],
            "canonical_program_archive_root_or_null": report_hex_or_null(
                "canonical_program_archive_root_or_null"
            ),
            "program_chunk_manifest_root_or_null": report_hex_or_null(
                "program_chunk_manifest_root_or_null"
            ),
            "bucket_accounting_root_or_null": report_hex_or_null(
                "bucket_accounting_root_or_null"
            ),
            "first_out_of_budget_program_hash_or_null": report_hex_or_null(
                "first_out_of_budget_program_hash_or_null"
            ),
        }
        if (
            validated.get("closure_status")
            != "DSL_TOO_LARGE"
            or any(
                validated.get(field) != expected
                for field, expected in report_receipt_values.items()
            )
        ):
            _fail(
                FAIL_OUTCOME,
                f"{implementation} persisted report/receipt binding differs",
            )
        try:
            archive = _qualification._host_validate_enumerator_archive_v1(
                implementation_root,
                implementation=implementation,
                stdout_report=validated,
                roots=binding_roots,
            )
            archives[implementation] = archive
            report_payload = archive.get("report_payload")
            streams = archive.get("streams")
            if type(report_payload) is not bytes or not isinstance(streams, Mapping):
                _fail(
                    FAIL_OUTCOME,
                    f"{implementation} replayed archive byte snapshot differs",
                )
            bind_consumed_payload("archive/report.json", report_payload)
            for stream_name, filename in (
                ("canonical_program_records", "canonical_program_records.cborframed"),
                ("program_chunk_manifests", "program_chunk_manifests.cborframed"),
                ("bucket_accounting_records", "bucket_accounting_records.cborframed"),
            ):
                payload = streams.get(stream_name)
                if type(payload) is not bytes:
                    _fail(
                        FAIL_OUTCOME,
                        f"{implementation} replayed {stream_name} bytes differ",
                    )
                bind_consumed_payload(f"archive/{filename}", payload)
        except Exception as exc:
            _fail(
                FAIL_OUTCOME,
                f"{implementation} persisted archive failed replay: {exc}",
            )
    if closure_status_ids == {2}:
        try:
            _qualification._validate_dual_archive_bytes_equal_v1(
                archives["python"], archives["rust"]
            )
        except Exception as exc:
            _fail(FAIL_OUTCOME, f"persisted dual archive equality differs: {exc}")
    else:
        _fail(FAIL_OUTCOME, "persisted dual closure statuses differ")

    # Detect a namespace or byte replacement concurrent with the strict replay.
    for implementation in ("python", "rust"):
        rows = archive_manifest[implementation]
        for row in rows:
            observed_size, observed_sha = _stable_file_identity(
                attempt_root
                / "formal-enumeration"
                / implementation
                / row["relative_path"],
                maximum_bytes=128 * 1024 * 1024,
            )
            if (
                observed_size != row["byte_length"]
                or observed_sha != row["sha256"]
            ):
                _fail(
                    FAIL_OUTCOME,
                    f"{implementation} archive changed during formal replay",
                )


def build_outcome_document_v1(
    prepared: PreparedM3FormalExecutionV1,
    outcome: M3DualEnumerationOutcomeV1,
    *,
    attempt_id: str,
    attempt_root: Path,
    preflight_receipt: Mapping[str, object],
    enumeration_output_root: Path,
) -> dict[str, object]:
    if attempt_id != CANONICAL_ATTEMPT_ID:
        _fail(FAIL_OUTCOME, "attempt ID is not the unique frozen attempt")
    admission_artifact, runtime_manifest, admission_receipt = (
        _validated_local_admission_identity_v1(prepared)
    )
    closure_status_id = outcome.agreement_fields["agreed_closure_status_id"]
    if closure_status_id != 2:
        _fail(FAIL_OUTCOME, "formal outcome closure status is unsupported")
    closure_status = "DSL_TOO_LARGE"
    witness = outcome.agreement_fields[
        "first_out_of_budget_program_hash_or_null"
    ]
    formal_objects = {
        "python_enumeration_receipt": _formal_entry(
            "M3ImplementationEnumerationReceiptV1",
            outcome.python_receipt_fields,
        ),
        "rust_enumeration_receipt": _formal_entry(
            "M3ImplementationEnumerationReceiptV1",
            outcome.rust_receipt_fields,
        ),
        "dual_replay_agreement": _formal_entry(
            "M3DualReplayAgreementV1", outcome.agreement_fields
        ),
        "terminal_state_record": _formal_entry(
            "M3RunStateRecordV1", outcome.terminal_state_fields
        ),
    }
    if (
        formal_objects["python_enumeration_receipt"]["content_root_hex"]
        != outcome.python_receipt_root.hex()
        or formal_objects["rust_enumeration_receipt"]["content_root_hex"]
        != outcome.rust_receipt_root.hex()
        or formal_objects["dual_replay_agreement"]["content_root_hex"]
        != outcome.agreement_root.hex()
        or formal_objects["terminal_state_record"]["content_root_hex"]
        != outcome.terminal_state_root.hex()
    ):
        _fail(FAIL_OUTCOME, "formal outcome roots differ during serialization")
    document: dict[str, object] = {
        "schema": SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "publication_commit_b": PUBLICATION_COMMIT_B,
        "basis_commit_a": COMMIT_A,
        "attempt_id": attempt_id,
        "attempt_root_path_sha256": hashlib.sha256(
            attempt_root.as_posix().encode("utf-8")
        ).hexdigest(),
        "run_id_hex": outcome.terminal_state_fields["run_id"].hex(),
        "execution_manifest_root_hex": outcome.terminal_state_fields[
            "execution_manifest_root"
        ].hex(),
        "start_state_record_root_hex": prepared.start_record_root.hex(),
        "start_state_artifact_sha256": prepared.start_document[
            "state_artifact_sha256"
        ],
        "start_publication_receipt": dict(prepared.start_publication_receipt),
        "closure_status": closure_status,
        "canonical_program_count": outcome.python_receipt_fields[
            "canonical_program_count"
        ],
        "raw_operator_application_count": outcome.python_receipt_fields[
            "raw_operator_application_count"
        ],
        "first_out_of_budget_program_hash_hex_or_null": (
            None if witness is None else witness.hex()
        ),
        "role_evaluation_started": False,
        "enumerator_container_split_inputs_accessed": False,
        "host_public_split_commitments_and_roots_replayed": True,
        "raw_split_seed_accessed": False,
        "split_assignment_rows_accessed": False,
        "enumerator_container_target_inputs_accessed": False,
        "host_frozen_public_target_definitions_loaded": True,
        "contains_private_key": False,
        "contains_raw_split_seed": False,
        "local_admission_artifact": admission_artifact,
        "runtime_source_manifest": runtime_manifest,
        "local_admission_receipt": admission_receipt,
        "offline_runtime_preflight": dict(preflight_receipt),
        "runner_evidence_files": _runner_evidence_manifest(attempt_root),
        "formal_objects": formal_objects,
        "archive_files": _archive_manifest(enumeration_output_root),
    }
    assert_public_payload_contains_no_secret_fields(document)
    document["outcome_artifact_sha256"] = hashlib.sha256(
        canonical_json_v1(document)
    ).hexdigest()
    validate_outcome_document_v1(document, prepared=prepared)
    return document


def build_failure_outcome_document_v1(
    prepared: PreparedM3FormalExecutionV1,
    *,
    attempt_id: str,
    attempt_root: Path,
    error: Exception,
    preflight_receipt_or_null: Mapping[str, object] | None,
) -> dict[str, object]:
    """Terminalize a stable failed attempt without inventing closure evidence."""

    admission_artifact, runtime_manifest, admission_receipt = (
        _validated_local_admission_identity_v1(prepared)
    )
    code = getattr(error, "code", type(error).__name__)
    if type(code) is not str or not code or "\x00" in code:
        code = "UNCLASSIFIED_FORMAL_EXECUTION_FAILURE"
    if code == FAIL_TERMINALIZE:
        _fail(
            FAIL_TERMINALIZE,
            "unsafe container containment cannot become a formal terminal outcome",
        )
    semantic_failure = code == FAIL_DUAL
    detail_digest = hashlib.sha256(
        b"HEGEL/M3/FAILURE_DETAIL/V1\x00"
        + str(error).encode("utf-8", "replace")
    ).digest()
    run_id = prepared.start_record_fields["run_id"]
    manifest_root = prepared.start_record_fields["execution_manifest_root"]
    partial_fields: dict[str, object] = {
        "run_id": run_id,
        # The local supervisor is acting as the diagnostic auditor here; this
        # is not an enumerator success receipt and grants no authority claim.
        "implementation_id": 4,
        "terminal_failure_code_id_digest": id_digest_v1(code),
        "completed_bucket_count": 0,
        "partial_bucket_accounting_root_or_null": None,
        "partial_log_digest": detail_digest,
        "authoritative_claim_allowed": False,
    }
    partial_entry = _formal_entry("PartialDiagnosticBundleV1", partial_fields)
    partial_root = bytes.fromhex(partial_entry["content_root_hex"])
    terminal_fields: dict[str, object] = {
        "run_id": run_id,
        "transition_index": 1,
        "previous_state_record_root_or_null": prepared.start_record_root,
        "from_state_id": 1,
        "from_phase_id": 1,
        "to_state_id": 5 if semantic_failure else 6,
        "to_phase_id": 0,
        "transition_reason_id": 6 if semantic_failure else 7,
        "execution_manifest_root": manifest_root,
        "triggering_receipt_root_or_null": partial_root,
        "recorded_at_unix_seconds": max(
            int(time.time()),
            prepared.start_record_fields["recorded_at_unix_seconds"],
        ),
    }
    terminal_entry = _formal_entry("M3RunStateRecordV1", terminal_fields)
    document: dict[str, object] = {
        "schema": FAILURE_SCHEMA,
        "artifact_kind": FAILURE_ARTIFACT_KIND,
        "publication_commit_b": PUBLICATION_COMMIT_B,
        "basis_commit_a": COMMIT_A,
        "attempt_id": attempt_id,
        "attempt_root_path_sha256": hashlib.sha256(
            attempt_root.as_posix().encode("utf-8")
        ).hexdigest(),
        "run_id_hex": run_id.hex(),
        "execution_manifest_root_hex": manifest_root.hex(),
        "start_state_record_root_hex": prepared.start_record_root.hex(),
        "start_state_artifact_sha256": prepared.start_document[
            "state_artifact_sha256"
        ],
        "start_publication_receipt": dict(prepared.start_publication_receipt),
        "terminal_status": (
            "INCONCLUSIVE_SEMANTICS"
            if semantic_failure
            else "INCONCLUSIVE_EXECUTION"
        ),
        "failure_code": code,
        "failure_code_id_digest_hex": id_digest_v1(code).hex(),
        "failure_detail_digest_hex": detail_digest.hex(),
        "role_evaluation_started": False,
        "enumerator_container_split_inputs_accessed": False,
        "host_public_split_commitments_and_roots_replayed": True,
        "raw_split_seed_accessed": False,
        "split_assignment_rows_accessed": False,
        "enumerator_container_target_inputs_accessed": False,
        "host_frozen_public_target_definitions_loaded": True,
        "contains_private_key": False,
        "contains_raw_split_seed": False,
        "local_admission_artifact": admission_artifact,
        "runtime_source_manifest": runtime_manifest,
        "local_admission_receipt": admission_receipt,
        "offline_runtime_preflight_or_null": (
            None
            if preflight_receipt_or_null is None
            else dict(preflight_receipt_or_null)
        ),
        "formal_objects": {
            "partial_diagnostic_bundle": partial_entry,
            "terminal_state_record": terminal_entry,
        },
    }
    assert_public_payload_contains_no_secret_fields(document)
    document["outcome_artifact_sha256"] = hashlib.sha256(
        canonical_json_v1(document)
    ).hexdigest()
    validate_failure_outcome_document_v1(document, prepared=prepared)
    return document


def _validate_offline_runtime_preflight_v1(
    preflight: object,
    *,
    prepared: PreparedM3FormalExecutionV1,
) -> None:
    expected_fields = {
        "basis_commit",
        "python_source_root",
        "rust_source_root",
        "python_input_tree_sha256",
        "rust_input_tree_sha256",
        "all_immutable_inputs_sha256",
        "rust_binary_sha256",
        "runtime_seccomp_sha256",
        "python_probe_stdout_sha256",
        "python_probe_start_sha256",
        "python_probe_completion_sha256",
        "python_probe_container_name",
        "docker_daemon_receipt_binding",
        "pull_policy",
        "network_mode",
        "maximum_enumeration_seconds",
        "container_names",
        "attempt_intent_sha256",
    }
    qualification_daemon = prepared.implementation_qualification_receipt.get(
        "local_docker_daemon_receipt_binding"
    )
    attempt_root = (
        canonical_run_root_v1(prepared.start_document["run_id_hex"])
        / "attempts"
        / CANONICAL_ATTEMPT_ID
    )
    attempt_token = hashlib.sha256(
        attempt_root.as_posix().encode("utf-8")
    ).hexdigest()[:16]
    expected_container_names = {
        name: f"hegel-m3-{attempt_token}-{name}"
        for name in ("python", "rust")
    }
    expected_probe_container_name = f"hegel-m3-{attempt_token}-python-probe"
    if (
        type(preflight) is not dict
        or set(preflight) != expected_fields
        or preflight.get("basis_commit") != COMMIT_A
        or preflight.get("python_source_root")
        != FROZEN_IMPLEMENTATIONS["python"].source_root.hex()
        or preflight.get("rust_source_root")
        != FROZEN_IMPLEMENTATIONS["rust"].source_root.hex()
        or preflight.get("rust_binary_sha256")
        != FROZEN_IMPLEMENTATIONS["rust"].binary_digest.hex()
        or preflight.get("docker_daemon_receipt_binding") != qualification_daemon
        or preflight.get("pull_policy") != "never"
        or preflight.get("network_mode") != "none"
        or preflight.get("maximum_enumeration_seconds")
        != MAX_ENUMERATION_SECONDS
        or preflight.get("container_names") != expected_container_names
        or preflight.get("python_probe_container_name")
        != expected_probe_container_name
    ):
        _fail(FAIL_OUTCOME, "offline runtime preflight identity differs")
    for digest_field in (
        "python_input_tree_sha256",
        "rust_input_tree_sha256",
        "all_immutable_inputs_sha256",
        "runtime_seccomp_sha256",
        "python_probe_stdout_sha256",
        "python_probe_start_sha256",
        "python_probe_completion_sha256",
        "attempt_intent_sha256",
    ):
        if (
            type(preflight.get(digest_field)) is not str
            or _SHA256_RE.fullmatch(preflight[digest_field]) is None
        ):
            _fail(FAIL_OUTCOME, f"preflight digest differs: {digest_field}")


def _validate_embedded_local_admission_identity_v1(
    document: Mapping[str, object],
    *,
    prepared: PreparedM3FormalExecutionV1,
) -> None:
    expected_artifact, expected_manifest, expected_receipt = (
        _validated_local_admission_identity_v1(prepared)
    )
    artifact = document.get("local_admission_artifact")
    manifest = document.get("runtime_source_manifest")
    receipt = document.get("local_admission_receipt")
    if (
        type(artifact) is not dict
        or type(manifest) is not dict
        or type(receipt) is not dict
    ):
        _fail(FAIL_OUTCOME, "formal outcome local admission identity is absent")
    try:
        _local_admission.validate_local_admission_receipt_v1(
            receipt,
            artifact_fields=artifact,
            manifest_fields=manifest,
        )
    except Exception as exc:
        _fail(FAIL_OUTCOME, f"formal outcome local admission replay failed: {exc}")
    if (
        artifact != expected_artifact
        or manifest != expected_manifest
        or receipt != expected_receipt
    ):
        _fail(FAIL_OUTCOME, "formal outcome local admission binding differs")


def _validate_runner_evidence_semantics_v1(
    *,
    prepared: PreparedM3FormalExecutionV1,
    attempt_root: Path,
    preflight: Mapping[str, object],
    receipts: Mapping[str, Mapping[str, object]],
    archive_manifest: Mapping[str, object],
    runner_rows: Mapping[str, Mapping[str, object]],
) -> None:
    intent = _read_canonical_runner_json_v1(
        attempt_root / "runner-attempt-intent.json",
        expected_fields={
            "schema",
            "basis_commit",
            "attempt_root_path_sha256",
            "implementation_qualification_receipt_root",
            "python_implementation_binding_root",
            "rust_implementation_binding_root",
            "all_immutable_inputs_sha256",
            "enumeration_output_relative_path",
            "pull_policy",
            "network_mode",
            "restart_policy",
            "failure_cleanup_policy",
            "maximum_enumeration_seconds",
            "container_names",
            "python_probe_container_name",
            "python_probe_maximum_seconds",
            "python_probe_auto_remove",
        },
    )
    qualification_root = prepared.implementation_qualification_receipt.get(
        "receipt_root"
    )
    attempt_token = hashlib.sha256(
        attempt_root.as_posix().encode("utf-8")
    ).hexdigest()[:16]
    expected_probe_container_name = f"hegel-m3-{attempt_token}-python-probe"
    if (
        intent.get("schema") != ATTEMPT_INTENT_SCHEMA
        or intent.get("basis_commit") != COMMIT_A
        or intent.get("attempt_root_path_sha256")
        != hashlib.sha256(attempt_root.as_posix().encode("utf-8")).hexdigest()
        or intent.get("implementation_qualification_receipt_root")
        != qualification_root
        or intent.get("python_implementation_binding_root")
        != FROZEN_IMPLEMENTATIONS["python"].implementation_binding_root.hex()
        or intent.get("rust_implementation_binding_root")
        != FROZEN_IMPLEMENTATIONS["rust"].implementation_binding_root.hex()
        or intent.get("all_immutable_inputs_sha256")
        != preflight.get("all_immutable_inputs_sha256")
        or intent.get("enumeration_output_relative_path") != "formal-enumeration"
        or intent.get("pull_policy") != "never"
        or intent.get("network_mode") != "none"
        or intent.get("restart_policy") != RESTART_POLICY
        or intent.get("failure_cleanup_policy") != FAILURE_CLEANUP_POLICY
        or intent.get("maximum_enumeration_seconds")
        != MAX_ENUMERATION_SECONDS
        or intent.get("container_names") != preflight.get("container_names")
        or intent.get("python_probe_container_name")
        != expected_probe_container_name
        or intent.get("python_probe_container_name")
        != preflight.get("python_probe_container_name")
        or intent.get("python_probe_maximum_seconds")
        != PYTHON_PROBE_MAXIMUM_SECONDS
        or intent.get("python_probe_auto_remove") is not False
    ):
        _fail(FAIL_OUTCOME, "formal runner attempt intent differs")

    probe_started = _read_canonical_runner_json_v1(
        attempt_root / "runner-journal/python-probe-started.json",
        expected_fields={
            "schema",
            "container_name",
            "attempt_intent_sha256",
            "image_ref",
            "started_at_unix_seconds",
        },
    )
    probe_completed = _read_canonical_runner_json_v1(
        attempt_root / "runner-journal/python-probe-completed.json",
        expected_fields={
            "schema",
            "container_name",
            "attempt_intent_sha256",
            "image_ref",
            "binary_path",
            "binary_sha256",
            "version_sha256",
            "stdout_sha256",
            "started_at_unix_seconds",
            "finished_at_unix_seconds",
            "docker_started_at",
            "docker_finished_at",
        },
    )
    probe_stdout_path = attempt_root / "python-runtime-probe-stdout.json"
    probe_stdout_payload = _stable_file_payload_v1(
        probe_stdout_path,
        maximum_bytes=1_048_576,
    )
    probe_stdout = _strict_committed_json(
        probe_stdout_payload,
        label="named Python runtime probe stdout",
    )
    if (
        set(probe_stdout) != {"binary_path", "binary_sha256", "version"}
        or canonical_json_v1(probe_stdout) != probe_stdout_payload
    ):
        _fail(FAIL_OUTCOME, "named Python runtime probe stdout schema differs")
    probe_stdout_sha256 = hashlib.sha256(probe_stdout_payload).hexdigest()
    probe_binary_path = probe_stdout.get("binary_path")
    probe_binary_sha256 = probe_stdout.get("binary_sha256")
    probe_version = probe_stdout.get("version")
    probe_started_at = probe_started.get("started_at_unix_seconds")
    probe_finished_at = probe_completed.get("finished_at_unix_seconds")
    expected_intent_sha256 = preflight.get("attempt_intent_sha256")
    python_image_ref = FROZEN_IMPLEMENTATIONS["python"].image_ref
    python_binary_sha256 = FROZEN_IMPLEMENTATIONS["python"].binary_digest.hex()
    if (
        probe_started.get("schema") != PROBE_START_SCHEMA
        or probe_completed.get("schema") != PROBE_COMPLETION_SCHEMA
        or probe_started.get("container_name") != expected_probe_container_name
        or probe_completed.get("container_name") != expected_probe_container_name
        or probe_started.get("attempt_intent_sha256")
        != expected_intent_sha256
        or probe_completed.get("attempt_intent_sha256")
        != expected_intent_sha256
        or probe_started.get("image_ref") != python_image_ref
        or probe_completed.get("image_ref") != python_image_ref
        or type(probe_binary_path) is not str
        or not probe_binary_path.startswith("/usr/local/bin/python")
        or probe_binary_sha256 != python_binary_sha256
        or type(probe_version) is not str
        or probe_completed.get("binary_path") != probe_binary_path
        or probe_completed.get("binary_sha256") != python_binary_sha256
        or probe_completed.get("version_sha256")
        != hashlib.sha256(probe_version.encode("utf-8")).hexdigest()
        or probe_completed.get("stdout_sha256") != probe_stdout_sha256
        or type(probe_started_at) is not int
        or type(probe_finished_at) is not int
        or probe_completed.get("started_at_unix_seconds") != probe_started_at
        or probe_finished_at < probe_started_at
        or type(probe_completed.get("docker_started_at")) is not str
        or not probe_completed["docker_started_at"]
        or type(probe_completed.get("docker_finished_at")) is not str
        or not probe_completed["docker_finished_at"]
    ):
        _fail(FAIL_OUTCOME, "named Python runtime probe identity differs")
    try:
        validate_timestamp_ordering_v1(
            prepared.start_record_fields["recorded_at_unix_seconds"],
            probe_started_at,
        )
        validate_timestamp_ordering_v1(probe_started_at, probe_finished_at)
    except Exception as exc:
        _fail(FAIL_OUTCOME, f"named Python runtime probe timestamp differs: {exc}")
    for relative_path, preflight_field in (
        (
            "python-runtime-probe-stdout.json",
            "python_probe_stdout_sha256",
        ),
        (
            "runner-journal/python-probe-started.json",
            "python_probe_start_sha256",
        ),
        (
            "runner-journal/python-probe-completed.json",
            "python_probe_completion_sha256",
        ),
    ):
        if runner_rows[relative_path].get("sha256") != preflight.get(
            preflight_field
        ):
            _fail(FAIL_OUTCOME, "named Python runtime probe hash binding differs")
    if (
        runner_rows["python-runtime-probe-stdout.json"].get("byte_length")
        != len(probe_stdout_payload)
        or preflight.get("python_probe_stdout_sha256")
        != probe_stdout_sha256
    ):
        _fail(FAIL_OUTCOME, "named Python runtime probe stdout binding differs")

    candidate = prepared.replay.gate_inputs.execution_candidate_fields
    invocation_digests: set[str] = set()
    for implementation in ("python", "rust"):
        frozen = FROZEN_IMPLEMENTATIONS[implementation]
        invocation = EnumerationInvocationV1(
            implementation=implementation,
            implementation_id=frozen.implementation_id,
            basis_commit=COMMIT_A,
            source_root=frozen.source_root,
            binary_digest=frozen.binary_digest,
            image_ref=frozen.image_ref,
            implementation_binding_root=frozen.implementation_binding_root,
            bound_executable_locator=frozen.bound_executable_locator,
            child_dsl_spec_root=candidate["child_dsl_spec_root"],
            operator_semantics_root=candidate["operator_semantics_root"],
            identifier_registry_root=candidate["identifier_registry_root"],
            canonical_program_budget=CANONICAL_PROGRAM_BUDGET,
            raw_operator_application_cap=RAW_OPERATOR_APPLICATION_CAP,
            pull_policy="never",
            network_mode="none",
            output_parent=(
                attempt_root / "formal-enumeration" / implementation
            ),
        )
        invocation_digest = _invocation_digest_v1(
            invocation,
            attempt_root=attempt_root,
        )
        invocation_digests.add(invocation_digest)
        started = _read_canonical_runner_json_v1(
            attempt_root / f"runner-journal/{implementation}-started.json",
            expected_fields={
                "schema",
                "implementation",
                "implementation_id",
                "container_name",
                "invocation_sha256",
                "attempt_intent_sha256",
                "started_at_unix_seconds",
            },
        )
        completed = _read_canonical_runner_json_v1(
            attempt_root / f"runner-journal/{implementation}-completed.json",
            expected_fields={
                "schema",
                "implementation",
                "implementation_id",
                "container_name",
                "invocation_sha256",
                "attempt_intent_sha256",
                "started_at_unix_seconds",
                "finished_at_unix_seconds",
                "process_completion_sha256",
            },
        )
        process = _read_canonical_runner_json_v1(
            attempt_root
            / f"formal-enumeration/{implementation}/process-completion.json",
            expected_fields={
                "schema",
                "implementation",
                "implementation_id",
                "container_name",
                "invocation_sha256",
                "attempt_intent_sha256",
                "started_at_unix_seconds",
                "finished_at_unix_seconds",
                "process_exit_code",
                "stdout_sha256",
                "stderr_sha256",
                "pull_policy",
                "network_mode",
                "docker_started_at",
                "docker_finished_at",
                "docker_oom_killed",
                "docker_error",
            },
        )
        common = {
            "implementation": implementation,
            "implementation_id": frozen.implementation_id,
            "container_name": preflight["container_names"][implementation],
            "invocation_sha256": invocation_digest,
            "attempt_intent_sha256": preflight["attempt_intent_sha256"],
        }
        if any(
            item.get(field) != expected
            for item in (started, completed, process)
            for field, expected in common.items()
        ):
            _fail(FAIL_OUTCOME, f"{implementation} runner evidence identity differs")
        enumerator_started_at = started.get("started_at_unix_seconds")
        if (
            type(enumerator_started_at) is not int
            or enumerator_started_at < probe_finished_at
        ):
            _fail(
                FAIL_OUTCOME,
                f"{implementation} enumeration predates Python runtime probe",
            )
        receipt = receipts[implementation]
        if (
            started.get("schema") != START_MARKER_SCHEMA
            or completed.get("schema") != JOURNAL_COMPLETION_SCHEMA
            or process.get("schema") != COMPLETION_MARKER_SCHEMA
            or started.get("started_at_unix_seconds")
            != receipt.get("started_at_unix_seconds")
            or completed.get("started_at_unix_seconds")
            != receipt.get("started_at_unix_seconds")
            or process.get("started_at_unix_seconds")
            != receipt.get("started_at_unix_seconds")
            or completed.get("finished_at_unix_seconds")
            != receipt.get("finished_at_unix_seconds")
            or process.get("finished_at_unix_seconds")
            != receipt.get("finished_at_unix_seconds")
            or process.get("process_exit_code") != 0
            or process.get("pull_policy") != "never"
            or process.get("network_mode") != "none"
            or process.get("docker_oom_killed") is not False
            or process.get("docker_error") != ""
            or type(process.get("docker_started_at")) is not str
            or not process["docker_started_at"]
            or type(process.get("docker_finished_at")) is not str
            or not process["docker_finished_at"]
        ):
            _fail(FAIL_OUTCOME, f"{implementation} runner completion differs")
        rows = archive_manifest[implementation]
        by_path = {row["relative_path"]: row for row in rows}
        process_row = by_path["process-completion.json"]
        stdout_row = by_path["execution-stdout.json"]
        stderr_row = by_path["execution-stderr.bin"]
        if (
            completed.get("process_completion_sha256")
            != process_row["sha256"]
            or process.get("stdout_sha256") != stdout_row["sha256"]
            or process.get("stderr_sha256") != stderr_row["sha256"]
            or stderr_row["byte_length"] != 0
        ):
            _fail(FAIL_OUTCOME, f"{implementation} runner file cross-link differs")
    if len(invocation_digests) != 2:
        _fail(FAIL_OUTCOME, "formal runner invocation identities are not independent")


def validate_failure_outcome_document_v1(
    document: Mapping[str, object],
    *,
    prepared: PreparedM3FormalExecutionV1,
) -> None:
    required = {
        "schema",
        "artifact_kind",
        "publication_commit_b",
        "basis_commit_a",
        "attempt_id",
        "attempt_root_path_sha256",
        "run_id_hex",
        "execution_manifest_root_hex",
        "start_state_record_root_hex",
        "start_state_artifact_sha256",
        "start_publication_receipt",
        "terminal_status",
        "failure_code",
        "failure_code_id_digest_hex",
        "failure_detail_digest_hex",
        "role_evaluation_started",
        "enumerator_container_split_inputs_accessed",
        "host_public_split_commitments_and_roots_replayed",
        "raw_split_seed_accessed",
        "split_assignment_rows_accessed",
        "enumerator_container_target_inputs_accessed",
        "host_frozen_public_target_definitions_loaded",
        "contains_private_key",
        "contains_raw_split_seed",
        "local_admission_artifact",
        "runtime_source_manifest",
        "local_admission_receipt",
        "offline_runtime_preflight_or_null",
        "formal_objects",
        "outcome_artifact_sha256",
    }
    if type(document) is not dict or set(document) != required:
        _fail(FAIL_OUTCOME, "formal failure outcome field set differs")
    body = dict(document)
    claimed = body.pop("outcome_artifact_sha256")
    code = document.get("failure_code")
    if code == FAIL_TERMINALIZE:
        _fail(
            FAIL_TERMINALIZE,
            "unsafe container containment is not a valid terminal outcome",
        )
    expected_attempt_root = (
        canonical_run_root_v1(prepared.start_document["run_id_hex"])
        / "attempts"
        / CANONICAL_ATTEMPT_ID
    )
    expected_terminal_status = (
        "INCONCLUSIVE_SEMANTICS"
        if code == FAIL_DUAL
        else "INCONCLUSIVE_EXECUTION"
    )
    if (
        document.get("schema") != FAILURE_SCHEMA
        or document.get("artifact_kind") != FAILURE_ARTIFACT_KIND
        or document.get("publication_commit_b") != PUBLICATION_COMMIT_B
        or document.get("basis_commit_a") != COMMIT_A
        or document.get("attempt_id") != CANONICAL_ATTEMPT_ID
        or document.get("attempt_root_path_sha256")
        != hashlib.sha256(
            expected_attempt_root.as_posix().encode("utf-8")
        ).hexdigest()
        or document.get("run_id_hex") != prepared.start_document.get("run_id_hex")
        or document.get("execution_manifest_root_hex")
        != prepared.start_document.get("execution_manifest_root_hex")
        or document.get("start_state_record_root_hex")
        != prepared.start_record_root.hex()
        or document.get("start_state_artifact_sha256")
        != prepared.start_document.get("state_artifact_sha256")
        or document.get("start_publication_receipt")
        != dict(prepared.start_publication_receipt)
        or document.get("terminal_status") != expected_terminal_status
        or type(code) is not str
        or not code
        or document.get("failure_code_id_digest_hex") != id_digest_v1(code).hex()
        or any(
            document.get(field) is not False
            for field in (
                "role_evaluation_started",
                "enumerator_container_split_inputs_accessed",
                "raw_split_seed_accessed",
                "split_assignment_rows_accessed",
                "enumerator_container_target_inputs_accessed",
                "contains_private_key",
                "contains_raw_split_seed",
            )
        )
        or document.get("host_public_split_commitments_and_roots_replayed")
        is not True
        or document.get("host_frozen_public_target_definitions_loaded") is not True
        or type(claimed) is not str
        or claimed != hashlib.sha256(canonical_json_v1(body)).hexdigest()
    ):
        _fail(FAIL_OUTCOME, "formal failure outcome identity differs")
    _validate_embedded_local_admission_identity_v1(document, prepared=prepared)
    preflight_or_null = document.get("offline_runtime_preflight_or_null")
    if preflight_or_null is not None:
        _validate_offline_runtime_preflight_v1(
            preflight_or_null,
            prepared=prepared,
        )
    for digest_field in (
        "attempt_root_path_sha256",
        "start_state_record_root_hex",
        "start_state_artifact_sha256",
        "execution_manifest_root_hex",
        "failure_code_id_digest_hex",
        "failure_detail_digest_hex",
        "outcome_artifact_sha256",
    ):
        if (
            type(document.get(digest_field)) is not str
            or _SHA256_RE.fullmatch(document[digest_field]) is None
        ):
            _fail(FAIL_OUTCOME, f"formal failure digest differs: {digest_field}")
    objects = document.get("formal_objects")
    if type(objects) is not dict or set(objects) != {
        "partial_diagnostic_bundle",
        "terminal_state_record",
    }:
        _fail(FAIL_OUTCOME, "formal failure object set differs")
    decoded: dict[str, Mapping[str, object]] = {}
    roots: dict[str, bytes] = {}
    for key, schema_name in (
        ("partial_diagnostic_bundle", "PartialDiagnosticBundleV1"),
        ("terminal_state_record", "M3RunStateRecordV1"),
    ):
        row = objects[key]
        if type(row) is not dict:
            _fail(FAIL_OUTCOME, "formal failure object entry differs")
        try:
            payload = bytes.fromhex(row["cbor_hex"])
            item = decode_formal_object(payload, expected_name=schema_name)
            root = candidate_content_root(schema_name, item.fields)
        except Exception as exc:
            _fail(FAIL_OUTCOME, f"formal failure object replay failed: {exc}")
        if (
            set(row) != {"schema_name", "cbor_hex", "content_root_hex"}
            or row["schema_name"] != schema_name
            or row["content_root_hex"] != root.hex()
            or encode_formal_object(schema_name, item.fields) != payload
        ):
            _fail(FAIL_OUTCOME, "formal failure CBOR/root differs")
        decoded[key] = item.fields
        roots[key] = root
    partial = decoded["partial_diagnostic_bundle"]
    terminal = decoded["terminal_state_record"]
    semantic = document["terminal_status"] == "INCONCLUSIVE_SEMANTICS"
    if (
        partial.get("run_id") != prepared.start_record_fields["run_id"]
        or partial.get("implementation_id") != 4
        or partial.get("terminal_failure_code_id_digest") != id_digest_v1(code)
        or partial.get("completed_bucket_count") != 0
        or partial.get("partial_bucket_accounting_root_or_null") is not None
        or partial.get("partial_log_digest").hex()
        != document["failure_detail_digest_hex"]
        or partial.get("authoritative_claim_allowed") is not False
        or terminal.get("run_id") != prepared.start_record_fields["run_id"]
        or terminal.get("execution_manifest_root")
        != prepared.start_record_fields["execution_manifest_root"]
        or terminal.get("previous_state_record_root_or_null")
        != prepared.start_record_root
        or terminal.get("triggering_receipt_root_or_null")
        != roots["partial_diagnostic_bundle"]
        or terminal.get("to_state_id") != (5 if semantic else 6)
        or terminal.get("transition_reason_id") != (6 if semantic else 7)
        or (
            terminal.get("transition_index"),
            terminal.get("from_state_id"),
            terminal.get("from_phase_id"),
            terminal.get("to_phase_id"),
        )
        != (1, 1, 1, 0)
    ):
        _fail(FAIL_OUTCOME, "formal failure state chain differs")
    try:
        validate_timestamp_ordering_v1(
            prepared.start_record_fields["recorded_at_unix_seconds"],
            terminal["recorded_at_unix_seconds"],
        )
    except Exception as exc:
        _fail(FAIL_OUTCOME, f"formal failure timestamp differs: {exc}")
    assert_public_payload_contains_no_secret_fields(document)


def validate_outcome_document_v1(
    document: Mapping[str, object],
    *,
    prepared: PreparedM3FormalExecutionV1,
) -> None:
    required = {
        "schema",
        "artifact_kind",
        "publication_commit_b",
        "basis_commit_a",
        "attempt_id",
        "attempt_root_path_sha256",
        "run_id_hex",
        "execution_manifest_root_hex",
        "start_state_record_root_hex",
        "start_state_artifact_sha256",
        "start_publication_receipt",
        "closure_status",
        "canonical_program_count",
        "raw_operator_application_count",
        "first_out_of_budget_program_hash_hex_or_null",
        "role_evaluation_started",
        "enumerator_container_split_inputs_accessed",
        "host_public_split_commitments_and_roots_replayed",
        "raw_split_seed_accessed",
        "split_assignment_rows_accessed",
        "enumerator_container_target_inputs_accessed",
        "host_frozen_public_target_definitions_loaded",
        "contains_private_key",
        "contains_raw_split_seed",
        "local_admission_artifact",
        "runtime_source_manifest",
        "local_admission_receipt",
        "offline_runtime_preflight",
        "runner_evidence_files",
        "formal_objects",
        "archive_files",
        "outcome_artifact_sha256",
    }
    if type(document) is not dict or set(document) != required:
        _fail(FAIL_OUTCOME, "formal outcome field set differs")
    body = dict(document)
    claimed = body.pop("outcome_artifact_sha256")
    if (
        document.get("schema") != SCHEMA
        or document.get("artifact_kind") != ARTIFACT_KIND
        or document.get("publication_commit_b") != PUBLICATION_COMMIT_B
        or document.get("basis_commit_a") != COMMIT_A
        or document.get("run_id_hex")
        != prepared.start_document.get("run_id_hex")
        or document.get("execution_manifest_root_hex")
        != prepared.start_document.get("execution_manifest_root_hex")
        or document.get("attempt_root_path_sha256")
        != hashlib.sha256(
            (
                canonical_run_root_v1(prepared.start_document["run_id_hex"])
                / "attempts"
                / CANONICAL_ATTEMPT_ID
            )
            .as_posix()
            .encode("utf-8")
        ).hexdigest()
        or document.get("start_state_record_root_hex")
        != prepared.start_record_root.hex()
        or document.get("start_state_artifact_sha256")
        != prepared.start_document.get("state_artifact_sha256")
        or document.get("start_publication_receipt")
        != dict(prepared.start_publication_receipt)
        or document.get("closure_status") != "DSL_TOO_LARGE"
        or any(
            document.get(field) is not False
            for field in (
                "role_evaluation_started",
                "enumerator_container_split_inputs_accessed",
                "raw_split_seed_accessed",
                "split_assignment_rows_accessed",
                "enumerator_container_target_inputs_accessed",
                "contains_private_key",
                "contains_raw_split_seed",
            )
        )
        or document.get("host_public_split_commitments_and_roots_replayed")
        is not True
        or document.get("host_frozen_public_target_definitions_loaded") is not True
        or type(claimed) is not str
        or claimed != hashlib.sha256(canonical_json_v1(body)).hexdigest()
    ):
        _fail(FAIL_OUTCOME, "formal outcome identity or self-hash differs")
    _validate_embedded_local_admission_identity_v1(document, prepared=prepared)
    if (
        type(document.get("attempt_id")) is not str
        or document["attempt_id"] != CANONICAL_ATTEMPT_ID
    ):
        _fail(FAIL_OUTCOME, "formal outcome attempt ID differs")
    for field in (
        "attempt_root_path_sha256",
        "execution_manifest_root_hex",
        "start_state_record_root_hex",
        "start_state_artifact_sha256",
        "outcome_artifact_sha256",
    ):
        if type(document.get(field)) is not str or _SHA256_RE.fullmatch(document[field]) is None:
            _fail(FAIL_OUTCOME, f"formal outcome digest is malformed: {field}")
    witness_hex_or_null = document.get(
        "first_out_of_budget_program_hash_hex_or_null"
    )
    if witness_hex_or_null is not None and (
        type(witness_hex_or_null) is not str
        or _SHA256_RE.fullmatch(witness_hex_or_null) is None
    ):
        _fail(FAIL_OUTCOME, "formal outcome witness digest is malformed")
    if type(document.get("run_id_hex")) is not str or re.fullmatch(
        r"[0-9a-f]{32}", document["run_id_hex"]
    ) is None:
        _fail(FAIL_OUTCOME, "formal outcome run ID is malformed")
    objects = document.get("formal_objects")
    expected_objects = {
        "python_enumeration_receipt": "M3ImplementationEnumerationReceiptV1",
        "rust_enumeration_receipt": "M3ImplementationEnumerationReceiptV1",
        "dual_replay_agreement": "M3DualReplayAgreementV1",
        "terminal_state_record": "M3RunStateRecordV1",
    }
    if type(objects) is not dict or set(objects) != set(expected_objects):
        _fail(FAIL_OUTCOME, "formal outcome object set differs")
    decoded: dict[str, Mapping[str, object]] = {}
    roots: dict[str, bytes] = {}
    for key, schema_name in expected_objects.items():
        row = objects[key]
        if type(row) is not dict or set(row) != {
            "schema_name",
            "cbor_hex",
            "content_root_hex",
        }:
            _fail(FAIL_OUTCOME, f"formal outcome entry differs: {key}")
        try:
            cbor = bytes.fromhex(row["cbor_hex"])
            item = decode_formal_object(cbor, expected_name=schema_name)
            if encode_formal_object(schema_name, item.fields) != cbor:
                raise ValueError("noncanonical formal CBOR")
            root = candidate_content_root(schema_name, item.fields)
        except Exception as exc:
            _fail(FAIL_OUTCOME, f"formal outcome object {key} failed replay: {exc}")
        if row["schema_name"] != schema_name or row["content_root_hex"] != root.hex():
            _fail(FAIL_OUTCOME, f"formal outcome object root differs: {key}")
        decoded[key] = item.fields
        roots[key] = root
    python_receipt = decoded["python_enumeration_receipt"]
    rust_receipt = decoded["rust_enumeration_receipt"]
    agreement = decoded["dual_replay_agreement"]
    terminal = decoded["terminal_state_record"]
    run_id = bytes.fromhex(document["run_id_hex"])
    manifest_root = bytes.fromhex(document["execution_manifest_root_hex"])
    candidate = prepared.replay.gate_inputs.execution_candidate_fields
    expected_status_id = 2
    expected_raw_count = 3_292_439
    expected_program_count = CANONICAL_PROGRAM_BUDGET
    receipt_common = {
        "run_id": run_id,
        "execution_manifest_root": manifest_root,
        "child_dsl_spec_root": candidate["child_dsl_spec_root"],
        "operator_semantics_root": candidate["operator_semantics_root"],
        "identifier_registry_root": candidate["identifier_registry_root"],
        "canonical_ast_schema_root": candidate["canonical_ast_schema_root"],
        "canonical_cbor_profile_root": candidate["canonical_cbor_profile_root"],
        "closure_status_id": expected_status_id,
        "raw_operator_application_count": expected_raw_count,
        "canonical_program_count": expected_program_count,
        "closure_cardinality_or_null": None,
        "frontier_exhausted": False,
        "all_type_buckets_closed": False,
        "raw_expansion_limit_hit": False,
        "wall_clock_abort_hit": False,
        "partial_diagnostic_bundle_root_or_null": None,
        "process_exit_code": 0,
    }
    for name, receipt in (
        ("python", python_receipt),
        ("rust", rust_receipt),
    ):
        frozen = FROZEN_IMPLEMENTATIONS[name]
        exact = {
            **receipt_common,
            "implementation_id": frozen.implementation_id,
            "implementation_source_root": frozen.source_root,
            "implementation_binary_digest": frozen.binary_digest,
            "environment_image_digest": frozen.execution_environment_spec_root,
        }
        if any(receipt.get(field) != value for field, value in exact.items()):
            _fail(FAIL_OUTCOME, f"{name} formal enumeration receipt differs")
    agreement_root_fields = {
        "canonical_program_archive_root_or_null",
        "program_chunk_manifest_root_or_null",
        "bucket_accounting_root_or_null",
        "first_out_of_budget_program_hash_or_null",
    }
    if any(
        python_receipt.get(field) != rust_receipt.get(field)
        or python_receipt.get(field) != agreement.get(field)
        for field in agreement_root_fields
    ):
        _fail(FAIL_OUTCOME, "receipt/agreement archive or witness roots differ")
    if (
        python_receipt.get("implementation_id") != 1
        or rust_receipt.get("implementation_id") != 2
        or python_receipt.get("run_id") != run_id
        or rust_receipt.get("run_id") != run_id
        or agreement.get("run_id") != run_id
        or terminal.get("run_id") != run_id
        or any(
            fields.get("execution_manifest_root") != manifest_root
            for fields in (python_receipt, rust_receipt, agreement, terminal)
        )
        or agreement.get("python_enumeration_receipt_root")
        != roots["python_enumeration_receipt"]
        or agreement.get("rust_enumeration_receipt_root")
        != roots["rust_enumeration_receipt"]
        or agreement.get("enumeration_agreement") is not True
        or agreement.get("agreed_closure_status_id") != expected_status_id
        or agreement.get("canonical_program_count_or_null")
        != expected_program_count
        or agreement.get("closure_cardinality_or_null") is not None
        or agreement.get("role_agreement_entries") != ()
        or agreement.get("role_agreement_status_id") != 0
        or agreement.get("mismatch_record_root_or_null") is not None
        or terminal.get("previous_state_record_root_or_null")
        != prepared.start_record_root
        or terminal.get("triggering_receipt_root_or_null")
        != roots["dual_replay_agreement"]
        or type(agreement.get("first_out_of_budget_program_hash_or_null"))
        is not bytes
        or document.get("first_out_of_budget_program_hash_hex_or_null")
        != (
            None
            if agreement.get("first_out_of_budget_program_hash_or_null") is None
            else agreement.get("first_out_of_budget_program_hash_or_null").hex()
        )
        or (
            terminal.get("transition_index"),
            terminal.get("from_state_id"),
            terminal.get("from_phase_id"),
            terminal.get("to_state_id"),
            terminal.get("to_phase_id"),
            terminal.get("transition_reason_id"),
        )
        != (
            1,
            1,
            1,
            3,
            0,
            3,
        )
    ):
        _fail(FAIL_OUTCOME, "formal outcome state/receipt chain differs")
    if (
        document.get("canonical_program_count") != expected_program_count
        or document.get("raw_operator_application_count") != expected_raw_count
    ):
        _fail(FAIL_OUTCOME, "formal outcome count identity differs")
    try:
        start_timestamp = prepared.start_record_fields["recorded_at_unix_seconds"]
        validate_timestamp_ordering_v1(
            start_timestamp, python_receipt["started_at_unix_seconds"]
        )
        validate_timestamp_ordering_v1(
            start_timestamp, rust_receipt["started_at_unix_seconds"]
        )
        for receipt in (python_receipt, rust_receipt):
            validate_timestamp_ordering_v1(
                receipt["started_at_unix_seconds"],
                receipt["finished_at_unix_seconds"],
            )
            validate_timestamp_ordering_v1(
                receipt["finished_at_unix_seconds"],
                agreement["created_at_unix_seconds"],
            )
        validate_timestamp_ordering_v1(
            agreement["created_at_unix_seconds"],
            terminal["recorded_at_unix_seconds"],
        )
    except Exception as exc:
        _fail(FAIL_OUTCOME, f"formal outcome timestamp ordering differs: {exc}")

    attempt_root = (
        canonical_run_root_v1(document["run_id_hex"])
        / "attempts"
        / CANONICAL_ATTEMPT_ID
    )
    _validate_offline_runtime_preflight_v1(
        document.get("offline_runtime_preflight"),
        prepared=prepared,
    )
    runner_rows = document.get("runner_evidence_files")
    expected_runner_paths = {
        "python-runtime-probe-stdout.json",
        "runner-attempt-intent.json",
        "runner-preflight.json",
        "runner-journal/python-probe-started.json",
        "runner-journal/python-probe-completed.json",
        "runner-journal/python-started.json",
        "runner-journal/python-completed.json",
        "runner-journal/rust-started.json",
        "runner-journal/rust-completed.json",
    }
    if (
        type(runner_rows) is not list
        or len(runner_rows) != len(expected_runner_paths)
        or [
            row.get("relative_path") if type(row) is dict else None
            for row in runner_rows
        ]
        != sorted(expected_runner_paths)
    ):
        _fail(FAIL_OUTCOME, "formal runner evidence manifest differs")
    runner_by_path: dict[str, Mapping[str, object]] = {}
    for row in runner_rows:
        if (
            type(row) is not dict
            or set(row) != {"relative_path", "byte_length", "sha256"}
            or type(row.get("relative_path")) is not str
            or type(row.get("byte_length")) is not int
            or row["byte_length"] < 1
            or type(row.get("sha256")) is not str
            or _SHA256_RE.fullmatch(row["sha256"]) is None
        ):
            _fail(FAIL_OUTCOME, "formal runner evidence row differs")
        runner_by_path[row["relative_path"]] = row
        observed_size, observed_sha = _stable_file_identity(
            attempt_root / row["relative_path"],
            maximum_bytes=1_048_576,
        )
        if observed_size != row["byte_length"] or observed_sha != row["sha256"]:
            _fail(FAIL_OUTCOME, "formal runner evidence file hash differs")
    if (
        runner_by_path["runner-attempt-intent.json"]["sha256"]
        != document["offline_runtime_preflight"]["attempt_intent_sha256"]
    ):
        _fail(FAIL_OUTCOME, "runner attempt-intent/preflight binding differs")
    expected_preflight_payload = canonical_json_v1(
        document["offline_runtime_preflight"]
    )
    if (
        runner_by_path["runner-preflight.json"]["byte_length"]
        != len(expected_preflight_payload)
        or runner_by_path["runner-preflight.json"]["sha256"]
        != hashlib.sha256(expected_preflight_payload).hexdigest()
    ):
        _fail(FAIL_OUTCOME, "persisted runner preflight bytes differ")

    archives = document.get("archive_files")
    expected_archive_paths = {
        "archive/report.json",
        "archive/canonical_program_records.cborframed",
        "archive/program_chunk_manifests.cborframed",
        "archive/bucket_accounting_records.cborframed",
        "execution-stdout.json",
        "execution-stderr.bin",
        "process-completion.json",
    }
    if type(archives) is not dict or set(archives) != {"python", "rust"}:
        _fail(FAIL_OUTCOME, "archive manifest implementation set differs")
    if document.get("attempt_root_path_sha256") != hashlib.sha256(
        attempt_root.as_posix().encode("utf-8")
    ).hexdigest():
        _fail(FAIL_OUTCOME, "formal outcome attempt-root binding differs")
    for implementation in ("python", "rust"):
        rows = archives[implementation]
        if (
            type(rows) is not list
            or len(rows) != len(expected_archive_paths)
            or [
                row.get("relative_path") if type(row) is dict else None
                for row in rows
            ]
            != sorted(expected_archive_paths)
        ):
            _fail(FAIL_OUTCOME, f"{implementation} archive manifest differs")
        for row in rows:
            if (
                type(row) is not dict
                or set(row) != {"relative_path", "byte_length", "sha256"}
                or type(row.get("byte_length")) is not int
                or row["byte_length"] < 0
                or type(row.get("sha256")) is not str
                or _SHA256_RE.fullmatch(row["sha256"]) is None
            ):
                _fail(FAIL_OUTCOME, f"{implementation} archive row differs")
            observed_size, observed_sha = _stable_file_identity(
                attempt_root
                / "formal-enumeration"
                / implementation
                / row["relative_path"],
                maximum_bytes=128 * 1024 * 1024,
            )
            if (
                observed_size != row["byte_length"]
                or observed_sha != row["sha256"]
            ):
                _fail(FAIL_OUTCOME, f"{implementation} archive file hash differs")
    _validate_runner_evidence_semantics_v1(
        prepared=prepared,
        attempt_root=attempt_root,
        preflight=document["offline_runtime_preflight"],
        receipts={"python": python_receipt, "rust": rust_receipt},
        archive_manifest=archives,
        runner_rows=runner_by_path,
    )
    _replay_persisted_enumerator_archives_v1(
        prepared=prepared,
        attempt_root=attempt_root,
        receipts={"python": python_receipt, "rust": rust_receipt},
        archive_manifest=archives,
    )
    assert_public_payload_contains_no_secret_fields(document)


def _read_existing_outcome(path: Path) -> bytes:
    if not path.is_absolute():
        _fail(FAIL_OUTPUT, "formal outcome path must be absolute")
    directory_descriptor: int | None = None
    try:
        directory_descriptor = _open_outcome_directory_v1(path.parent)
        payload = _read_outcome_at_v1(directory_descriptor, path.name)
        assert payload is not None
        return payload
    except M3FormalExecutionError:
        raise
    except OSError as exc:
        _fail(FAIL_OUTPUT, f"cannot read existing formal outcome: {type(exc).__name__}")
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def _open_outcome_directory_v1(parent: Path) -> int:
    """Open and pin the canonical caller-owned outcome directory."""

    try:
        lexical = parent.lstat()
        if parent.resolve(strict=True) != parent:
            _fail(FAIL_OUTPUT, "formal outcome parent is not a canonical real path")
        descriptor = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or (lexical.st_dev, lexical.st_ino)
            != (observed.st_dev, observed.st_ino)
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) != 0o700
        ):
            os.close(descriptor)
            _fail(FAIL_OUTPUT, "formal outcome parent identity differs")
        return descriptor
    except M3FormalExecutionError:
        raise
    except OSError as exc:
        _fail(FAIL_OUTPUT, f"formal outcome parent is unavailable: {type(exc).__name__}")


def _assert_pinned_run_directory_v1(parent: Path, descriptor: int) -> None:
    """Require a pinned run-root descriptor to remain at its canonical name."""

    try:
        lexical = parent.lstat()
        observed = os.fstat(descriptor)
        resolved = parent.resolve(strict=True)
    except OSError as exc:
        _fail(FAIL_OUTPUT, f"formal run root identity is unavailable: {type(exc).__name__}")
    if (
        resolved != parent
        or not stat.S_ISDIR(observed.st_mode)
        or (lexical.st_dev, lexical.st_ino)
        != (observed.st_dev, observed.st_ino)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        _fail(FAIL_OUTPUT, "formal run root identity changed while leased")


def _stable_metadata_identity_v1(metadata: os.stat_result) -> tuple[int, ...]:
    """Return every stable, security-relevant field used for inode replay."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_rdev,
        metadata.st_size,
        metadata.st_blksize,
        metadata.st_blocks,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _complete_metadata_identity_v1(metadata: os.stat_result) -> tuple[int, ...]:
    """Add access time when comparing two post-read views of one inode."""

    return _stable_metadata_identity_v1(metadata) + (metadata.st_atime_ns,)


def _assert_owned_directory_link_at_v1(
    parent_descriptor: int,
    name: str,
    directory_descriptor: int,
    *,
    label: str,
) -> None:
    """Bind an opened directory to its no-follow name under a pinned parent."""

    try:
        opened = os.fstat(directory_descriptor)
        namespace = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        _fail(FAIL_OUTPUT, f"{label} identity is unavailable: {type(exc).__name__}")
    if (
        not stat.S_ISDIR(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o700
        or _complete_metadata_identity_v1(namespace)
        != _complete_metadata_identity_v1(opened)
    ):
        _fail(FAIL_OUTPUT, f"{label} owner, mode, or namespace identity differs")


def _open_or_create_owned_directory_at_v1(
    parent_descriptor: int,
    name: str,
    *,
    label: str,
) -> int:
    """Create/open one directory component relative to an already-pinned fd."""

    if (
        type(name) is not str
        or not name
        or name in {".", ".."}
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
        or "\x00" in name
    ):
        _fail(FAIL_OUTPUT, f"{label} is not one lexical directory component")
    descriptor: int | None = None
    created = False
    try:
        parent = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.geteuid()
            or stat.S_IMODE(parent.st_mode) != 0o700
        ):
            _fail(FAIL_OUTPUT, f"{label} parent directory identity differs")
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
            created = True
        except FileExistsError:
            pass
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or opened.st_uid != os.geteuid():
            _fail(FAIL_OUTPUT, f"{label} is not an owned directory")
        if created:
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
            os.fsync(parent_descriptor)
        _assert_owned_directory_link_at_v1(
            parent_descriptor,
            name,
            descriptor,
            label=label,
        )
        result = descriptor
        descriptor = None
        return result
    except M3FormalExecutionError:
        raise
    except OSError as exc:
        _fail(FAIL_OUTPUT, f"cannot create or open {label}: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_or_create_attempt_tree_v1(
    lease_directory: int,
    run_root: Path,
    attempt_id: str,
) -> tuple[Path, int, int]:
    """Create ``attempts/attempt-1`` solely with mkdirat/openat operations."""

    if attempt_id != CANONICAL_ATTEMPT_ID:
        _fail(FAIL_OUTPUT, "formal attempt directory ID differs")
    _assert_pinned_run_directory_v1(run_root, lease_directory)
    attempts_descriptor: int | None = None
    attempt_descriptor: int | None = None
    try:
        attempts_descriptor = _open_or_create_owned_directory_at_v1(
            lease_directory,
            "attempts",
            label="formal attempts directory",
        )
        attempt_descriptor = _open_or_create_owned_directory_at_v1(
            attempts_descriptor,
            attempt_id,
            label="formal attempt directory",
        )
        # Creating the second level changes the parent directory metadata, so
        # replay both links only after the complete tree exists.
        _assert_owned_directory_link_at_v1(
            lease_directory,
            "attempts",
            attempts_descriptor,
            label="formal attempts directory",
        )
        _assert_owned_directory_link_at_v1(
            attempts_descriptor,
            attempt_id,
            attempt_descriptor,
            label="formal attempt directory",
        )
        _assert_pinned_run_directory_v1(run_root, lease_directory)
        result = (
            run_root / "attempts" / attempt_id,
            attempts_descriptor,
            attempt_descriptor,
        )
        attempts_descriptor = None
        attempt_descriptor = None
        return result
    finally:
        if attempt_descriptor is not None:
            os.close(attempt_descriptor)
        if attempts_descriptor is not None:
            os.close(attempts_descriptor)


def _assert_pinned_attempt_tree_v1(
    lease_directory: int,
    attempts_descriptor: int,
    attempt_descriptor: int,
    *,
    run_root: Path,
    attempt_id: str,
) -> None:
    """Replay both retained directory descriptors before formal publication."""

    _assert_pinned_run_directory_v1(run_root, lease_directory)
    _assert_owned_directory_link_at_v1(
        lease_directory,
        "attempts",
        attempts_descriptor,
        label="formal attempts directory",
    )
    _assert_owned_directory_link_at_v1(
        attempts_descriptor,
        attempt_id,
        attempt_descriptor,
        label="formal attempt directory",
    )


def _acquire_execution_lease_v1(run_root: Path) -> tuple[int, int]:
    """Acquire the crash-released single-writer lease for one formal run."""

    directory_descriptor = _open_outcome_directory_v1(run_root)
    lock_descriptor: int | None = None
    complete = False
    try:
        try:
            lock_descriptor = os.open(
                _EXECUTION_LOCK_NAME,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=directory_descriptor,
            )
            os.fchmod(lock_descriptor, 0o600)
            os.fsync(lock_descriptor)
            os.fsync(directory_descriptor)
        except FileExistsError:
            lock_descriptor = os.open(
                _EXECUTION_LOCK_NAME,
                os.O_RDWR
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_descriptor,
            )
        metadata = os.fstat(lock_descriptor)
        named = os.stat(
            _EXECUTION_LOCK_NAME,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != 0
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino) != (named.st_dev, named.st_ino)
        ):
            _fail(FAIL_OUTPUT, "formal execution lockfile identity differs")
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        _assert_pinned_run_directory_v1(run_root, directory_descriptor)
        locked = os.fstat(lock_descriptor)
        locked_named = os.stat(
            _EXECUTION_LOCK_NAME,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            (locked.st_dev, locked.st_ino)
            != (locked_named.st_dev, locked_named.st_ino)
            or not stat.S_ISREG(locked.st_mode)
            or locked.st_uid != os.geteuid()
            or stat.S_IMODE(locked.st_mode) != 0o600
            or locked.st_size != 0
            or locked.st_nlink != 1
        ):
            _fail(FAIL_OUTPUT, "formal execution lockfile changed while waiting")
        complete = True
        return directory_descriptor, lock_descriptor
    except M3FormalExecutionError:
        raise
    except OSError as exc:
        _fail(FAIL_OUTPUT, f"formal execution lease failed: {type(exc).__name__}")
    finally:
        if lock_descriptor is not None and not complete:
            os.close(lock_descriptor)
        if not complete:
            os.close(directory_descriptor)


def _release_execution_lease_v1(
    directory_descriptor: int,
    lock_descriptor: int,
) -> None:
    """Release a formal lease; closing the fd also releases it after crashes."""

    try:
        os.close(lock_descriptor)
    finally:
        os.close(directory_descriptor)


def _read_outcome_at_v1(
    directory_descriptor: int,
    name: str,
    *,
    missing_ok: bool = False,
) -> bytes | None:
    """Read one outcome relative to a pinned directory without following links."""

    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_descriptor,
        )
    except FileNotFoundError:
        if missing_ok:
            return None
        _fail(FAIL_OUTPUT, "existing formal outcome is absent")
    except OSError as exc:
        _fail(FAIL_OUTPUT, f"cannot open existing formal outcome: {type(exc).__name__}")
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size > MAX_OUTCOME_BYTES
        ):
            _fail(FAIL_OUTPUT, "existing formal outcome is not a bounded owned file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1_048_576))
            if not chunk:
                _fail(FAIL_OUTPUT, "existing formal outcome is truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(FAIL_OUTPUT, "existing formal outcome grew while being read")
        after = os.fstat(descriptor)
        namespace = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            _stable_metadata_identity_v1(before)
            != _stable_metadata_identity_v1(after)
            or _complete_metadata_identity_v1(namespace)
            != _complete_metadata_identity_v1(after)
        ):
            _fail(FAIL_OUTPUT, "existing formal outcome changed while being read")
        return b"".join(chunks)
    except M3FormalExecutionError:
        raise
    except OSError as exc:
        _fail(
            FAIL_OUTPUT,
            "cannot verify existing formal outcome namespace: "
            f"{type(exc).__name__}",
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_outcome_exact_once(
    path: Path,
    document: Mapping[str, object],
    *,
    directory_descriptor: int | None = None,
) -> str:
    payload = canonical_json_v1(document)
    if not path.is_absolute():
        _fail(FAIL_OUTPUT, "formal outcome path must be absolute")
    parent = path.parent
    owns_directory_descriptor = directory_descriptor is None
    pending_name: str | None = None
    descriptor: int | None = None
    pending_present = False

    def existing_status() -> str:
        assert directory_descriptor is not None
        raw = _read_outcome_at_v1(directory_descriptor, path.name)
        if raw == payload:
            return "ALREADY_TERMINAL_IDENTICAL"
        _fail(FAIL_ALREADY_TERMINAL, "a different formal outcome already exists")

    try:
        if directory_descriptor is None:
            directory_descriptor = _open_outcome_directory_v1(parent)
        else:
            _assert_pinned_run_directory_v1(parent, directory_descriptor)
        if (
            _read_outcome_at_v1(
                directory_descriptor, path.name, missing_ok=True
            )
            is not None
        ):
            return existing_status()
        for _attempt in range(_PENDING_NAME_ATTEMPTS):
            token = os.urandom(16).hex()
            pending_name = (
                f".{path.name}.{document['outcome_artifact_sha256']}."
                f"{os.getpid()}.{time.time_ns()}.{token}.pending"
            )
            try:
                descriptor = os.open(
                    pending_name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=directory_descriptor,
                )
                pending_present = True
                os.fchmod(descriptor, 0o600)
                break
            except FileExistsError:
                pending_name = None
        if descriptor is None or pending_name is None:
            _fail(FAIL_OUTPUT, "cannot allocate a unique formal outcome pending file")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail(FAIL_OUTPUT, "short formal outcome write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(
                pending_name,
                path.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            status = "TERMINAL_PUBLISHED_NEW"
        except FileExistsError:
            status = existing_status()
            os.unlink(pending_name, dir_fd=directory_descriptor)
            pending_present = False
            os.fsync(directory_descriptor)
            return status
        pending_metadata = os.stat(
            pending_name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        target_metadata = os.stat(
            path.name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(target_metadata.st_mode)
            or stat.S_IMODE(target_metadata.st_mode) != 0o600
            or (pending_metadata.st_dev, pending_metadata.st_ino)
            != (target_metadata.st_dev, target_metadata.st_ino)
            or _read_outcome_at_v1(directory_descriptor, path.name) != payload
        ):
            _fail(FAIL_OUTPUT, "published formal outcome inode or bytes differ")
        os.fsync(directory_descriptor)
        os.unlink(pending_name, dir_fd=directory_descriptor)
        pending_present = False
        os.fsync(directory_descriptor)
        return status
    except M3FormalExecutionError:
        raise
    except OSError as exc:
        _fail(FAIL_OUTPUT, f"formal outcome publication failed: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor is not None:
            if pending_present and pending_name is not None:
                try:
                    os.unlink(pending_name, dir_fd=directory_descriptor)
                    os.fsync(directory_descriptor)
                except FileNotFoundError:
                    pass
            if owns_directory_descriptor:
                os.close(directory_descriptor)


def _verified_existing_publication_v1(
    payload: bytes,
    *,
    prepared: PreparedM3FormalExecutionV1,
    outcome_path: Path,
) -> FormalExecutionPublicationV1:
    existing = strict_json_loads_v1(payload, label="formal M3 outcome")
    if canonical_json_v1(existing) != payload:
        _fail(FAIL_OUTCOME, "existing formal outcome is not canonical JSON")
    if existing.get("schema") == SCHEMA:
        validate_outcome_document_v1(existing, prepared=prepared)
    elif existing.get("schema") == FAILURE_SCHEMA:
        validate_failure_outcome_document_v1(existing, prepared=prepared)
    else:
        _fail(FAIL_OUTCOME, "existing formal outcome schema is unknown")
    return FormalExecutionPublicationV1(
        status="ALREADY_TERMINAL_VERIFIED",
        document=MappingProxyType(dict(existing)),
        outcome_path=outcome_path,
        attempt_root=None,
    )


def execute_formal_m3_v1(
    prepared: PreparedM3FormalExecutionV1,
    *,
    run_root: Path,
    attempt_id: str,
    outcome_path: Path,
) -> FormalExecutionPublicationV1:
    """Run a fresh attempt, or verify and return an existing terminal outcome."""

    if (
        not isinstance(prepared, PreparedM3FormalExecutionV1)
        or prepared._seal is not _PREPARED_SEAL
    ):
        _fail(FAIL_INPUT, "formal execution lacks a verified persisted-start capability")
    _replay_live_start_publication_v1(prepared)
    _replay_live_local_admission_identity_v1(prepared)
    if attempt_id != CANONICAL_ATTEMPT_ID:
        _fail(FAIL_OUTPUT, "formal execution permits only the unique attempt-1")
    if not run_root.is_absolute() or not outcome_path.is_absolute():
        _fail(FAIL_OUTPUT, "run and outcome paths must be absolute")
    run_id_hex = prepared.start_document["run_id_hex"]
    if (
        run_root != canonical_run_root_v1(run_id_hex)
        or outcome_path != canonical_terminal_outcome_path_v1(run_id_hex)
        or canonical_start_state_path_v1(run_id_hex).parent != run_root
    ):
        _fail(FAIL_OUTPUT, "formal execution paths are not canonical for this run")
    if outcome_path.exists() or outcome_path.is_symlink():
        return _verified_existing_publication_v1(
            _read_existing_outcome(outcome_path),
            prepared=prepared,
            outcome_path=outcome_path,
        )
    lease_directory, lease_descriptor = _acquire_execution_lease_v1(run_root)
    attempts_descriptor: int | None = None
    attempt_descriptor: int | None = None
    try:
        # Acquiring the lease can block behind another caller.  Replay both
        # persisted authorization identities after that wait, before either
        # accepting the winner's terminal record or entering Docker.
        _assert_pinned_run_directory_v1(run_root, lease_directory)
        _replay_live_start_publication_v1(prepared)
        _replay_live_local_admission_identity_v1(prepared)
        # This second check is the concurrency boundary.  A caller that waited
        # behind another execution verifies the winner and never enters Docker.
        existing_payload = _read_outcome_at_v1(
            lease_directory,
            outcome_path.name,
            missing_ok=True,
        )
        if existing_payload is not None:
            return _verified_existing_publication_v1(
                existing_payload,
                prepared=prepared,
                outcome_path=outcome_path,
            )
        (
            attempt_root,
            attempts_descriptor,
            attempt_descriptor,
        ) = _open_or_create_attempt_tree_v1(
            lease_directory,
            run_root,
            attempt_id,
        )
        _assert_pinned_attempt_tree_v1(
            lease_directory,
            attempts_descriptor,
            attempt_descriptor,
            run_root=run_root,
            attempt_id=attempt_id,
        )

        runner: OfflineDockerEnumerationRunnerV1 | None = None
        try:
            with OfflineDockerEnumerationRunnerV1(
                repository_root=prepared.repository_root,
                attempt_root=attempt_root,
                implementation_qualification_receipt=(
                    prepared.implementation_qualification_receipt
                ),
            ) as runner:
                enumeration_output = attempt_root / "formal-enumeration"
                outcome = _run_m3_dual_enumeration_core_v1(
                    qualified_gate_evidence=prepared.replay.qualified_gate_evidence,
                    execution_candidate_fields=(
                        prepared.replay.gate_inputs.execution_candidate_fields
                    ),
                    run_genesis_fields=prepared.replay.gate_inputs.run_genesis_fields,
                    start_record_fields=prepared.start_record_fields,
                    start_record_root=prepared.start_record_root,
                    implementation_qualification_receipt=(
                        prepared.implementation_qualification_receipt
                    ),
                    committed_golden=prepared.committed_golden,
                    output_root=enumeration_output,
                    runner=runner,
                    resume_existing_output=True,
                )
                runner.verify_inputs_stable_v1()
                _replay_live_start_publication_v1(prepared)
                _replay_live_local_admission_identity_v1(prepared)
                if runner.preflight_receipt is None:
                    _fail(FAIL_OUTPUT, "formal runner preflight receipt is absent")
                document = build_outcome_document_v1(
                    prepared,
                    outcome,
                    attempt_id=attempt_id,
                    attempt_root=attempt_root,
                    preflight_receipt=runner.preflight_receipt,
                    enumeration_output_root=enumeration_output,
                )
        except Exception as exc:
            if getattr(exc, "code", None) == FAIL_TERMINALIZE:
                # A formal terminal record is forbidden until both named
                # containers are proven absent or non-running.  Preserve RUNNING
                # and require recovery/manual containment instead of lying about
                # a completed execution failure transition.
                raise M3FormalExecutionError(
                    FAIL_TERMINALIZE,
                    "named Docker containers are not proven non-running; "
                    "formal terminalization is forbidden",
                ) from exc
            document = build_failure_outcome_document_v1(
                prepared,
                attempt_id=attempt_id,
                attempt_root=attempt_root,
                error=exc,
                preflight_receipt_or_null=(
                    None if runner is None else runner.preflight_receipt
                ),
            )
        _assert_pinned_attempt_tree_v1(
            lease_directory,
            attempts_descriptor,
            attempt_descriptor,
            run_root=run_root,
            attempt_id=attempt_id,
        )
        _assert_pinned_run_directory_v1(run_root, lease_directory)
        status = _write_outcome_exact_once(
            outcome_path,
            document,
            directory_descriptor=lease_directory,
        )
        return FormalExecutionPublicationV1(
            status=status,
            document=MappingProxyType(document),
            outcome_path=outcome_path,
            attempt_root=attempt_root,
        )
    finally:
        if attempt_descriptor is not None:
            os.close(attempt_descriptor)
        if attempts_descriptor is not None:
            os.close(attempts_descriptor)
        _release_execution_lease_v1(lease_directory, lease_descriptor)


__all__ = [
    "ARTIFACT_KIND",
    "CANONICAL_ATTEMPT_ID",
    "FormalExecutionPublicationV1",
    "M3FormalExecutionError",
    "PreparedM3FormalExecutionV1",
    "QUALIFICATION_RECEIPT_PATH",
    "SCHEMA",
    "FAILURE_ARTIFACT_KIND",
    "FAILURE_SCHEMA",
    "build_failure_outcome_document_v1",
    "build_outcome_document_v1",
    "execute_formal_m3_v1",
    "prepare_formal_execution_v1",
    "validate_outcome_document_v1",
    "validate_failure_outcome_document_v1",
]
