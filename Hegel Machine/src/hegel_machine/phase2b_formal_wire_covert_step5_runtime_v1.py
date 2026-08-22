"""Step-5 producer-side retention mechanics for the frozen formal audit.

This module deliberately has a narrow claim boundary.  It source-pins the
existing auditor and QV2 implementation, interposes only the two frozen
artifact cleanup callables, and requires a caller-supplied durable retention
sink to acknowledge each original worker file before the original cleanup is
allowed to run.  It does not implement a CAS, a signer, an authority ceremony,
runtime-image attestation, or R03/R04/R05 qualification.

The interposition is intentionally prospective.  It cannot bless a prior run
whose original transient worker files were already deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import math
import os
import re
import stat
import threading
from typing import Final, Mapping, Protocol

from . import phase2b_formal_covert_auditor_v1 as auditor
from . import phase2b_formal_wire_covert_qualification_v2 as qv2
from .rfc8785_v1 import (
    canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1,
)


SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-runtime/1"
)
RETENTION_REQUEST_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-scm-request/1"
)
RETENTION_ACK_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-retention-ack/1"
)
STATISTICAL_RESULTS_CHUNK_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-statistical-results-chunk/1"
)
STATISTICAL_RESULTS_MANIFEST_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-statistical-results-manifest/2"
)
FEATURE_MANIFEST_CHUNK_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-feature-manifest-chunk/1"
)

AUDITOR_SOURCE_SHA256: Final = (
    "5d77403dc41c661cb8a8d7deb90c9b2644de7a27fda9369c37f76eb82474a9be"
)
QV2_SOURCE_SHA256: Final = (
    "921bbb771d27c339acaf2c5e1e48f35a899c56552929065c02894e590d32e2ed"
)

MODE_ORDER: Final = ("KNOWN_NULL", "KNOWN_BINDING_LEAK")
INVARIANT_BASENAME: Final = "invariant-result.json"
STATISTICS_BASENAMES: Final = tuple(
    f"target-{target_index}.json"
    for target_index in range(len(auditor.TARGETS))
)
EXPECTED_OBJECT_ORDER: Final = tuple(
    (mode, phase, target_index)
    for mode in MODE_ORDER
    for phase, target_index in (
        (("INVARIANCE", None),)
        + tuple(("STATISTICS", index) for index in range(len(auditor.TARGETS)))
    )
)
EXPECTED_RETAINED_OBJECTS: Final = 12
EXPECTED_CLEANUP_CALLS: Final = 4
MAXIMUM_SOURCE_BYTES: Final = 2 * 1_048_576
MAXIMUM_RESULT_CHUNK_BYTES: Final = 2 * 1_048_576
RESULT_ROWS_PER_NONFINAL_CHUNK: Final = 1_024
FEATURE_MANIFEST_ENTRIES_PER_NONFINAL_CHUNK: Final = 256
MAXIMUM_STATISTICS_ARTIFACT_BYTES: Final = (
    auditor.WORKER_RESULT_ARTIFACT_FIXED_BYTES
    + auditor.WORKER_RESULT_ARTIFACT_BYTES_PER_FEATURE * auditor.MAXIMUM_FEATURES
)

RETENTION_CLAIM_BOUNDARY: Final = (
    "ORIGINAL_TRANSIENT_WORKER_BYTES_RETAINED_BEFORE_FROZEN_CLEANUP_"
    "UNDER_CALLER_SUPPLIED_SINK_ACK_NOT_CAS_IMPLEMENTATION_NOT_SIGNATURE_"
    "AUTHORITY_NOT_RUNTIME_IMAGE_ATTESTATION_NOT_R03_R04_R05"
)
RESULT_PROJECTION_CLAIM_BOUNDARY: Final = (
    "STATISTICAL_RESULTS_TARGET_FAMILY_AND_EXACT_FEATURE_MANIFEST_PREIMAGE_"
    "SUBPROJECTION_ONLY_NOT_WORKER_RETENTION_NOT_RECEIPT_CORE_"
    "NOT_RUNTIME_OR_RESOURCE_OR_QUOTA_AUTHORITY_NOT_R03_R04_R05"
)
TRUSTED_BOOTSTRAP_REQUIREMENT: Final = (
    "IMPORT_THIS_MODULE_ONLY_FROM_A_SOURCE_PINNED_FRESH_PROCESS_BOOTSTRAP_"
    "PREIMPORT_MONKEYPATCHING_IS_OUTSIDE_THIS_IN_PROCESS_CHECK"
)
IN_PROCESS_MUTATION_NONCLAIM: Final = (
    "SAME_PROCESS_CODE_OR_FUNCTION_DEFAULT_MUTATION_IS_OUTSIDE_THE_STEP5_"
    "AUTHORITY_BOUNDARY_AND_REQUIRES_FUTURE_FIXED_RUNTIME_IMAGE_CONTROLS"
)

_IDENTIFIER_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{7,511}")
_SHA256_PATTERN: Final = re.compile(r"sha256:[0-9a-f]{64}")
_AUDITOR_RECEIPT_ID_PATTERN: Final = re.compile(
    r"phase2b_formal_covert_audit_transcript_[0-9a-f]{64}"
)
_CLEANUP_PARAMETER_NAMES: Final = (
    "artifact_directory",
    "artifact_directory_fd",
    "artifact_directory_device",
    "artifact_directory_inode",
)
_CLEANUP_PARAMETER_ANNOTATIONS: Final = ("str", "int", "int", "int")

_ORIGINAL_INVARIANT_CLEANUP: Final = (
    auditor._cleanup_invariant_artifact_directory
)
_ORIGINAL_STATISTICS_CLEANUP: Final = (
    auditor._cleanup_statistics_artifact_directory
)
_ORIGINAL_TARGET_FEATURE_PROJECTION: Final = (
    auditor._target_feature_metric_projection
)
_ORIGINAL_TARGET_FAMILY_PROJECTION: Final = (
    auditor._target_family_metric_projection
)
_ORIGINAL_RECEIPT_VALIDATOR: Final = (
    auditor.validate_formal_covert_audit_receipt_v1
)


def _captured_cleanup_callables_v1(
    invariant_cleanup: object = _ORIGINAL_INVARIANT_CLEANUP,
    statistics_cleanup: object = _ORIGINAL_STATISTICS_CLEANUP,
) -> tuple[object, object]:
    """Return import-time identities, not rebindable ``_ORIGINAL_*`` globals."""

    return invariant_cleanup, statistics_cleanup


def _captured_projection_callables_v1(
    target_feature_projection: object = _ORIGINAL_TARGET_FEATURE_PROJECTION,
    target_family_projection: object = _ORIGINAL_TARGET_FAMILY_PROJECTION,
    receipt_validator: object = _ORIGINAL_RECEIPT_VALIDATOR,
) -> tuple[object, object, object]:
    """Return import-time identities for the source-pinned projection surface."""

    return (
        target_feature_projection,
        target_family_projection,
        receipt_validator,
    )


def _require_identifier(value: object, name: str) -> str:
    if type(value) is not str or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} is not one bounded exact identifier")
    return value


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} is not one canonical SHA-256 content address")
    return value


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_nlink == right.st_nlink
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _hash_open_fd_bounded(fd: int, maximum_bytes: int) -> tuple[str, int]:
    if type(fd) is not int or fd < 0:
        raise TypeError("bounded hash needs one open file descriptor")
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise TypeError("bounded hash needs one positive exact byte bound")
    before = os.fstat(fd)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= maximum_bytes
    ):
        raise ValueError("bounded hash target is not one bounded regular link")
    digest = hashlib.sha256()
    total = 0
    while total < before.st_size:
        length = min(1_048_576, before.st_size - total)
        chunk = os.pread(fd, length, total)
        if not chunk:
            raise ValueError("bounded hash observed a premature end of file")
        digest.update(chunk)
        total += len(chunk)
    if os.pread(fd, 1, total):
        raise ValueError("bounded hash target grew beyond its initial size")
    after = os.fstat(fd)
    if not _same_file_identity(before, after) or total != before.st_size:
        raise ValueError("bounded hash target changed while it was read")
    return "sha256:" + digest.hexdigest(), total


def _hash_regular_path_bounded(path: str, maximum_bytes: int) -> str:
    if type(path) is not str or not os.path.isabs(path):
        raise ValueError("source pin path must be absolute")
    realpath = os.path.realpath(path)
    before = os.lstat(realpath)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1:
        raise ValueError("source pin is not a regular file")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(realpath, flags)
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink < 1
            or not _same_file_identity(before, opened)
        ):
            raise ValueError("source pin changed during no-follow open")
        digest, _ = _hash_open_fd_bounded_allow_multiple_links_v1(
            fd, maximum_bytes
        )
        final = os.fstat(fd)
        if not _same_file_identity(opened, final):
            raise ValueError("source pin changed while it was hashed")
        return digest
    finally:
        os.close(fd)


def _hash_open_fd_bounded_allow_multiple_links_v1(
    fd: int,
    maximum_bytes: int,
) -> tuple[str, int]:
    before = os.fstat(fd)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink < 1
        or not 0 < before.st_size <= maximum_bytes
    ):
        raise ValueError("source hash target is not one bounded regular file")
    digest = hashlib.sha256()
    total = 0
    while total < before.st_size:
        chunk = os.pread(fd, min(1_048_576, before.st_size - total), total)
        if not chunk:
            raise ValueError("source hash observed a premature end of file")
        digest.update(chunk)
        total += len(chunk)
    after = os.fstat(fd)
    if not _same_file_identity(before, after) or total != before.st_size:
        raise ValueError("source hash target changed while it was read")
    return "sha256:" + digest.hexdigest(), total


def _require_exact_cleanup_callable(
    value: object,
    *,
    expected: object,
    expected_name: str,
) -> None:
    if value is not expected or not inspect.isfunction(value):
        raise RuntimeError(f"{expected_name} cleanup callable identity drift")
    if (
        getattr(value, "__module__", None) != auditor.__name__
        or getattr(value, "__name__", None) != expected_name
        or getattr(value, "__qualname__", None) != expected_name
    ):
        raise RuntimeError(f"{expected_name} cleanup callable provenance drift")
    signature = inspect.signature(value)
    parameters = tuple(signature.parameters.values())
    if (
        tuple(item.name for item in parameters) != _CLEANUP_PARAMETER_NAMES
        or any(
            item.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            or item.default is not inspect.Parameter.empty
            for item in parameters
        )
        or tuple(item.annotation for item in parameters)
        != _CLEANUP_PARAMETER_ANNOTATIONS
        or signature.return_annotation != "None"
    ):
        raise RuntimeError(f"{expected_name} cleanup callable signature drift")
    code_path = os.path.realpath(value.__code__.co_filename)
    module_path = os.path.realpath(auditor.__file__)
    if code_path != module_path:
        raise RuntimeError(f"{expected_name} cleanup code/source path drift")


def _require_source_pins_and_metric_projection_identities_v1(
    auditor_source_sha256: str = AUDITOR_SOURCE_SHA256,
    qv2_source_sha256: str = QV2_SOURCE_SHA256,
    target_feature_projection: object = _ORIGINAL_TARGET_FEATURE_PROJECTION,
    target_family_projection: object = _ORIGINAL_TARGET_FAMILY_PROJECTION,
    receipt_validator: object = _ORIGINAL_RECEIPT_VALIDATOR,
) -> None:
    if qv2.auditor is not auditor:
        raise RuntimeError("QV2 and Step-5 do not share the exact auditor module object")
    auditor_path = os.path.realpath(auditor.__file__)
    qv2_path = os.path.realpath(qv2.__file__)
    if os.path.basename(auditor_path) != "phase2b_formal_covert_auditor_v1.py":
        raise RuntimeError("auditor source basename drift")
    if os.path.basename(qv2_path) != "phase2b_formal_wire_covert_qualification_v2.py":
        raise RuntimeError("QV2 source basename drift")
    if _hash_regular_path_bounded(auditor_path, MAXIMUM_SOURCE_BYTES) != (
        "sha256:" + auditor_source_sha256
    ):
        raise RuntimeError("auditor source SHA-256 drift")
    if _hash_regular_path_bounded(qv2_path, MAXIMUM_SOURCE_BYTES) != (
        "sha256:" + qv2_source_sha256
    ):
        raise RuntimeError("QV2 source SHA-256 drift")
    if (
        auditor._target_feature_metric_projection
        is not target_feature_projection
        or auditor._target_family_metric_projection
        is not target_family_projection
        or auditor.validate_formal_covert_audit_receipt_v1
        is not receipt_validator
    ):
        raise RuntimeError(
            "auditor metric projection/receipt validator callable identity drift"
        )


def _require_source_pins_and_callable_identities_v1(
    invariant_cleanup: object = _ORIGINAL_INVARIANT_CLEANUP,
    statistics_cleanup: object = _ORIGINAL_STATISTICS_CLEANUP,
) -> None:
    _require_source_pins_and_metric_projection_identities_v1()
    _require_exact_cleanup_callable(
        auditor._cleanup_invariant_artifact_directory,
        expected=invariant_cleanup,
        expected_name="_cleanup_invariant_artifact_directory",
    )
    _require_exact_cleanup_callable(
        auditor._cleanup_statistics_artifact_directory,
        expected=statistics_cleanup,
        expected_name="_cleanup_statistics_artifact_directory",
    )


def _require_single_native_thread_v1() -> int:
    if not sys_platform_is_linux_v1():
        raise RuntimeError("Step-5 retention is Linux-only")
    task_names = tuple(sorted(os.listdir("/proc/self/task")))
    if any(re.fullmatch(r"[1-9][0-9]*", item) is None for item in task_names):
        raise RuntimeError("native task directory contains a non-canonical task ID")
    native_id = threading.get_native_id()
    if (
        threading.active_count() != 1
        or task_names != (str(native_id),)
        or threading.current_thread() is not threading.main_thread()
    ):
        raise RuntimeError("Step-5 retention requires one exact main/native thread")
    return native_id


def sys_platform_is_linux_v1() -> bool:
    return os.name == "posix" and os.path.isdir("/proc/self/task")


def _require_no_audit_begun_v1() -> None:
    with auditor._IN_PROCESS_RECEIPT_REGISTRY_LOCK:
        if (
            auditor._IN_PROCESS_EXECUTED_RECEIPTS
            or auditor._IN_PROCESS_RECEIPT_RESERVATIONS
        ):
            raise RuntimeError(
                "Step-5 retention must be installed before any audit reservation or receipt"
            )


@dataclass(frozen=True, slots=True)
class RetentionIngestAckV1:
    schema_version: str
    attempt_context_id: str
    retention_session_id: str
    request_id: str
    ordinal: int
    mode: str
    phase: str
    source_basename: str
    content_address: str
    sha256: str
    size_bytes: int
    durable_before_return: bool
    readback_verified: bool
    store_receipt_ref: str

    def __post_init__(self) -> None:
        if type(self) is not RetentionIngestAckV1:
            raise TypeError("retention acknowledgement must use the exact DTO")
        if (
            type(self.schema_version) is not str
            or self.schema_version != RETENTION_ACK_SCHEMA_VERSION
        ):
            raise ValueError("retention acknowledgement schema drift")
        _require_identifier(self.attempt_context_id, "ack attempt context ID")
        _require_identifier(self.retention_session_id, "ack retention session ID")
        _require_sha256(self.request_id, "ack request ID")
        if type(self.ordinal) is not int or not 0 <= self.ordinal < EXPECTED_RETAINED_OBJECTS:
            raise ValueError("retention acknowledgement ordinal drift")
        if (
            type(self.mode) is not str
            or self.mode not in MODE_ORDER
            or type(self.phase) is not str
            or self.phase not in {"INVARIANCE", "STATISTICS"}
        ):
            raise ValueError("retention acknowledgement mode/phase drift")
        if (
            type(self.source_basename) is not str
            or not self.source_basename
            or "/" in self.source_basename
        ):
            raise ValueError("retention acknowledgement basename drift")
        _require_sha256(self.sha256, "ack SHA-256")
        _require_sha256(self.content_address, "ack content address")
        if self.content_address != self.sha256:
            raise ValueError("retention acknowledgement is not content addressed")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("retention acknowledgement byte count drift")
        if self.durable_before_return is not True or self.readback_verified is not True:
            raise ValueError("retention acknowledgement lacks durable readback")
        _require_identifier(self.store_receipt_ref, "store receipt reference")


class RetentionSink(Protocol):
    """Injected transport/CAS boundary; implementations live outside this module."""

    def ingest_open_fd(
        self,
        metadata: Mapping[str, object],
        fd: int,
        sha256: str,
        size_bytes: int,
    ) -> RetentionIngestAckV1:
        """Durably ingest one borrowed read-only descriptor before returning."""


@dataclass(frozen=True, slots=True)
class RetainedWorkerObjectV1:
    attempt_context_id: str
    retention_session_id: str
    mode: str
    phase: str
    target_index: int | None
    source_basename: str
    ordinal: int
    request_id: str
    sha256: str
    size_bytes: int
    acknowledgement: RetentionIngestAckV1

    def __post_init__(self) -> None:
        if type(self) is not RetainedWorkerObjectV1:
            raise TypeError("retained worker object must use the exact DTO")
        _require_identifier(self.attempt_context_id, "retained attempt context ID")
        _require_identifier(
            self.retention_session_id,
            "retained retention session ID",
        )
        if (
            type(self.mode) is not str
            or self.mode not in MODE_ORDER
            or type(self.phase) is not str
            or self.phase not in {"INVARIANCE", "STATISTICS"}
        ):
            raise ValueError("retained worker mode/phase drift")
        if self.phase == "INVARIANCE":
            if self.target_index is not None or self.source_basename != INVARIANT_BASENAME:
                raise ValueError("retained invariant worker identity drift")
        elif (
            type(self.target_index) is not int
            or not 0 <= self.target_index < len(auditor.TARGETS)
            or self.source_basename != STATISTICS_BASENAMES[self.target_index]
        ):
            raise ValueError("retained statistics worker identity drift")
        if type(self.ordinal) is not int or not 0 <= self.ordinal < EXPECTED_RETAINED_OBJECTS:
            raise ValueError("retained worker ordinal drift")
        _require_sha256(self.request_id, "retained worker request ID")
        _require_sha256(self.sha256, "retained worker SHA-256")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("retained worker size drift")
        if type(self.acknowledgement) is not RetentionIngestAckV1:
            raise TypeError("retained worker acknowledgement must use the exact DTO")
        self.acknowledgement.__post_init__()
        if (
            self.acknowledgement.ordinal != self.ordinal
            or self.acknowledgement.attempt_context_id
            != self.attempt_context_id
            or self.acknowledgement.retention_session_id
            != self.retention_session_id
            or self.acknowledgement.request_id != self.request_id
            or self.acknowledgement.mode != self.mode
            or self.acknowledgement.phase != self.phase
            or self.acknowledgement.source_basename != self.source_basename
            or self.acknowledgement.sha256 != self.sha256
            or self.acknowledgement.size_bytes != self.size_bytes
        ):
            raise ValueError("retained worker acknowledgement linkage drift")


_RETENTION_SESSION_GUARD: Final = threading.Lock()
_ACTIVE_RETENTION_SESSION: "RetentionSessionV1 | None" = None


class RetentionSessionV1:
    """Prospective, one-shot interposition over the two frozen cleanup globals."""

    def __init__(
        self,
        *,
        attempt_context_id: str,
        retention_session_id: str,
        sink: RetentionSink,
    ) -> None:
        self.attempt_context_id = _require_identifier(
            attempt_context_id, "attempt context ID"
        )
        self.retention_session_id = _require_identifier(
            retention_session_id, "retention session ID"
        )
        if sink is None or not callable(getattr(sink, "ingest_open_fd", None)):
            raise TypeError("retention sink lacks ingest_open_fd")
        self._sink = sink
        self._used = False
        self._entered = False
        self._owner_native_id: int | None = None
        self._current_mode: str | None = None
        self._next_mode_index = 0
        self._cleanup_index_in_mode = 0
        self._completed_cleanup_calls = 0
        self._sink_callback_active = False
        self._sink_call_count = 0
        self._acknowledged_object_count = 0
        self._failed_closed_reason: str | None = None
        self._records: list[RetainedWorkerObjectV1] = []
        (
            self._original_invariant_cleanup,
            self._original_statistics_cleanup,
        ) = _captured_cleanup_callables_v1()
        self._invariant_wrapper = self._retain_then_cleanup_invariant
        self._statistics_wrapper = self._retain_then_cleanup_statistics

    @property
    def retained_objects(self) -> tuple[RetainedWorkerObjectV1, ...]:
        return tuple(self._records)

    @property
    def completed_cleanup_calls(self) -> int:
        return self._completed_cleanup_calls

    @property
    def sink_call_count(self) -> int:
        return self._sink_call_count

    @property
    def acknowledged_object_count(self) -> int:
        return self._acknowledged_object_count

    @property
    def failed_closed_reason(self) -> str | None:
        return self._failed_closed_reason

    def __enter__(self) -> "RetentionSessionV1":
        global _ACTIVE_RETENTION_SESSION
        if self._used or self._entered:
            raise RuntimeError("retention session is one-shot")
        self._used = True
        with _RETENTION_SESSION_GUARD:
            if _ACTIVE_RETENTION_SESSION is not None:
                raise RuntimeError("another retention session is already active")
            _ACTIVE_RETENTION_SESSION = self
        try:
            _require_source_pins_and_callable_identities_v1()
            captured_cleanup_callables = _captured_cleanup_callables_v1()
            if (
                self._original_invariant_cleanup
                is not captured_cleanup_callables[0]
                or self._original_statistics_cleanup
                is not captured_cleanup_callables[1]
            ):
                raise RuntimeError("session-captured cleanup identity drift")
            self._owner_native_id = _require_single_native_thread_v1()
            _require_no_audit_begun_v1()
            auditor._cleanup_invariant_artifact_directory = self._invariant_wrapper
            auditor._cleanup_statistics_artifact_directory = self._statistics_wrapper
            if (
                auditor._cleanup_invariant_artifact_directory
                is not self._invariant_wrapper
                or auditor._cleanup_statistics_artifact_directory
                is not self._statistics_wrapper
            ):
                raise RuntimeError("cleanup interposition identity drift")
            self._entered = True
            return self
        except BaseException:
            auditor._cleanup_invariant_artifact_directory = (
                self._original_invariant_cleanup
            )
            auditor._cleanup_statistics_artifact_directory = (
                self._original_statistics_cleanup
            )
            with _RETENTION_SESSION_GUARD:
                if _ACTIVE_RETENTION_SESSION is self:
                    _ACTIVE_RETENTION_SESSION = None
            raise

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> bool:
        global _ACTIVE_RETENTION_SESSION
        problems: list[str] = []
        if auditor._cleanup_invariant_artifact_directory is not self._invariant_wrapper:
            problems.append("invariant cleanup global changed while session was active")
        if auditor._cleanup_statistics_artifact_directory is not self._statistics_wrapper:
            problems.append("statistics cleanup global changed while session was active")
        auditor._cleanup_invariant_artifact_directory = (
            self._original_invariant_cleanup
        )
        auditor._cleanup_statistics_artifact_directory = (
            self._original_statistics_cleanup
        )
        if (
            auditor._cleanup_invariant_artifact_directory
            is not self._original_invariant_cleanup
        ):
            problems.append("invariant cleanup global was not restored")
        if (
            auditor._cleanup_statistics_artifact_directory
            is not self._original_statistics_cleanup
        ):
            problems.append("statistics cleanup global was not restored")
        with _RETENTION_SESSION_GUARD:
            if _ACTIVE_RETENTION_SESSION is not self:
                problems.append("active retention session identity changed")
            else:
                _ACTIVE_RETENTION_SESSION = None
        self._entered = False
        if exc is None:
            try:
                self.assert_complete()
            except BaseException as completion_error:
                problems.append(
                    f"{type(completion_error).__name__}: {completion_error}"
                )
        if problems:
            detail = "; ".join(problems)
            if exc is not None:
                try:
                    exc.add_note("Step-5 retention restoration/completion: " + detail)
                except AttributeError:
                    pass
                return False
            raise RuntimeError("Step-5 retention restoration/completion failed: " + detail)
        return False

    def begin_mode(self, mode: str) -> None:
        self._require_active_owner()
        if (
            type(mode) is not str
            or mode not in MODE_ORDER
            or self._current_mode is not None
            or self._next_mode_index >= len(MODE_ORDER)
            or mode != MODE_ORDER[self._next_mode_index]
        ):
            raise RuntimeError("retention mode order drift")
        self._current_mode = mode
        self._cleanup_index_in_mode = 0

    def crosscheck_and_complete_mode(
        self,
        receipt: auditor.FormalCovertAuditReceiptV1,
    ) -> None:
        self._require_active_owner()
        mode = self._current_mode
        if mode is None or self._cleanup_index_in_mode != 2:
            raise RuntimeError("mode cleanup coverage is incomplete")
        if type(receipt) is not auditor.FormalCovertAuditReceiptV1:
            raise TypeError("mode crosscheck requires the exact live audit receipt")
        invariant = receipt.invariant_runtime_evidence
        statistics = receipt.statistics_runtime_evidence
        if (
            type(invariant) is not auditor.InvariantRuntimeEvidenceV1
            or type(statistics) is not auditor.StatisticalRuntimeEvidenceV1
        ):
            raise ValueError("mode receipt lacks executed worker runtime evidence")
        mode_records = tuple(item for item in self._records if item.mode == mode)
        if len(mode_records) != 1 + len(auditor.TARGETS):
            raise ValueError("mode retained-worker cardinality drift")
        invariant_record = mode_records[0]
        if (
            invariant_record.phase != "INVARIANCE"
            or invariant_record.sha256 != invariant.worker_result_artifact_sha256
            or invariant_record.size_bytes != invariant.worker_result_artifact_bytes
        ):
            raise ValueError("retained invariant worker/receipt evidence drift")
        statistic_records = mode_records[1:]
        if tuple(item.sha256 for item in statistic_records) != (
            statistics.worker_result_artifact_sha256
        ) or tuple(item.size_bytes for item in statistic_records) != (
            statistics.worker_result_artifact_bytes
        ):
            raise ValueError("retained statistics workers/receipt evidence drift")
        self._current_mode = None
        self._cleanup_index_in_mode = 0
        self._next_mode_index += 1

    def assert_complete(self) -> tuple[RetainedWorkerObjectV1, ...]:
        _require_single_native_thread_v1()
        if (
            self._failed_closed_reason is not None
            or self._sink_callback_active
            or self._current_mode is not None
            or self._next_mode_index != len(MODE_ORDER)
            or self._completed_cleanup_calls != EXPECTED_CLEANUP_CALLS
            or self._sink_call_count != EXPECTED_RETAINED_OBJECTS
            or self._acknowledged_object_count != EXPECTED_RETAINED_OBJECTS
            or len(self._records) != EXPECTED_RETAINED_OBJECTS
        ):
            raise RuntimeError("Step-5 retention session is not exactly complete")
        observed_order = tuple(
            (item.mode, item.phase, item.target_index) for item in self._records
        )
        if observed_order != EXPECTED_OBJECT_ORDER or tuple(
            item.ordinal for item in self._records
        ) != tuple(range(EXPECTED_RETAINED_OBJECTS)):
            raise RuntimeError("Step-5 retained object order/ordinal drift")
        for item in self._records:
            item.__post_init__()
        return tuple(self._records)

    def _require_active_owner(self) -> None:
        if (
            not self._entered
            or _ACTIVE_RETENTION_SESSION is not self
            or self._owner_native_id != threading.get_native_id()
            or self._failed_closed_reason is not None
            or self._sink_callback_active
            or auditor._cleanup_invariant_artifact_directory
            is not self._invariant_wrapper
            or auditor._cleanup_statistics_artifact_directory
            is not self._statistics_wrapper
        ):
            raise RuntimeError(
                "retention session ownership/global/fail-closed state drift"
            )
        _require_single_native_thread_v1()

    def _mark_failed_closed(self, exc: BaseException, stage: str) -> None:
        if self._failed_closed_reason is None:
            self._failed_closed_reason = (
                f"{stage}:{type(exc).__name__}:{str(exc)[:512]}"
            )

    def _retain_then_cleanup_invariant(
        self,
        artifact_directory: str,
        artifact_directory_fd: int,
        artifact_directory_device: int,
        artifact_directory_inode: int,
    ) -> None:
        try:
            snapshots = self._retain_directory(
                phase="INVARIANCE",
                expected_basenames=(INVARIANT_BASENAME,),
                maximum_bytes=auditor.INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND,
                artifact_directory=artifact_directory,
                artifact_directory_fd=artifact_directory_fd,
                artifact_directory_device=artifact_directory_device,
                artifact_directory_inode=artifact_directory_inode,
            )
            self._require_precleanup_directory_snapshot(
                artifact_directory_fd,
                (INVARIANT_BASENAME,),
                snapshots,
            )
            self._require_active_owner()
            _require_exact_cleanup_callable(
                self._original_invariant_cleanup,
                expected=_captured_cleanup_callables_v1()[0],
                expected_name="_cleanup_invariant_artifact_directory",
            )
            self._original_invariant_cleanup(
                artifact_directory,
                artifact_directory_fd,
                artifact_directory_device,
                artifact_directory_inode,
            )
            self._require_active_owner()
            self._require_postcleanup_absence(
                artifact_directory,
                artifact_directory_fd,
            )
            self._cleanup_index_in_mode += 1
            self._completed_cleanup_calls += 1
        except BaseException as cleanup_error:
            self._mark_failed_closed(cleanup_error, "INVARIANCE_CLEANUP")
            raise

    def _retain_then_cleanup_statistics(
        self,
        artifact_directory: str,
        artifact_directory_fd: int,
        artifact_directory_device: int,
        artifact_directory_inode: int,
    ) -> None:
        try:
            snapshots = self._retain_directory(
                phase="STATISTICS",
                expected_basenames=STATISTICS_BASENAMES,
                maximum_bytes=MAXIMUM_STATISTICS_ARTIFACT_BYTES,
                artifact_directory=artifact_directory,
                artifact_directory_fd=artifact_directory_fd,
                artifact_directory_device=artifact_directory_device,
                artifact_directory_inode=artifact_directory_inode,
            )
            self._require_precleanup_directory_snapshot(
                artifact_directory_fd,
                STATISTICS_BASENAMES,
                snapshots,
            )
            self._require_active_owner()
            _require_exact_cleanup_callable(
                self._original_statistics_cleanup,
                expected=_captured_cleanup_callables_v1()[1],
                expected_name="_cleanup_statistics_artifact_directory",
            )
            self._original_statistics_cleanup(
                artifact_directory,
                artifact_directory_fd,
                artifact_directory_device,
                artifact_directory_inode,
            )
            self._require_active_owner()
            self._require_postcleanup_absence(
                artifact_directory,
                artifact_directory_fd,
            )
            self._cleanup_index_in_mode += 1
            self._completed_cleanup_calls += 1
        except BaseException as cleanup_error:
            self._mark_failed_closed(cleanup_error, "STATISTICS_CLEANUP")
            raise

    def _retain_directory(
        self,
        *,
        phase: str,
        expected_basenames: tuple[str, ...],
        maximum_bytes: int,
        artifact_directory: str,
        artifact_directory_fd: int,
        artifact_directory_device: int,
        artifact_directory_inode: int,
    ) -> dict[str, os.stat_result]:
        self._require_active_owner()
        mode = self._current_mode
        if self._cleanup_index_in_mode not in {0, 1}:
            raise RuntimeError("cleanup hook was invoked more than twice in one mode")
        expected_phase = (
            "INVARIANCE"
            if self._cleanup_index_in_mode == 0
            else "STATISTICS"
        )
        if (
            type(mode) is not str
            or mode not in MODE_ORDER
            or type(phase) is not str
            or phase != expected_phase
        ):
            raise RuntimeError("cleanup hook phase/mode order drift")
        if (
            type(artifact_directory) is not str
            or not os.path.isabs(artifact_directory)
            or type(artifact_directory_fd) is not int
            or type(artifact_directory_device) is not int
            or type(artifact_directory_inode) is not int
        ):
            raise TypeError("cleanup hook directory arguments are not exact")
        expected_prefix = (
            "hegel-formal-covert-invariant-"
            if phase == "INVARIANCE"
            else "hegel-formal-covert-stats-"
        )
        if not os.path.basename(artifact_directory).startswith(expected_prefix):
            raise ValueError("cleanup hook artifact directory prefix drift")
        retained = os.fstat(artifact_directory_fd)
        path_stat = os.lstat(artifact_directory)
        if (
            not stat.S_ISDIR(retained.st_mode)
            or not stat.S_ISDIR(path_stat.st_mode)
            or retained.st_dev != artifact_directory_device
            or retained.st_ino != artifact_directory_inode
            or path_stat.st_dev != retained.st_dev
            or path_stat.st_ino != retained.st_ino
        ):
            raise ValueError("cleanup hook retained directory identity drift")
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        scan_fd = os.open(".", directory_flags, dir_fd=artifact_directory_fd)
        try:
            observed_basenames = tuple(os.listdir(scan_fd))
        finally:
            os.close(scan_fd)
        if (
            len(observed_basenames) != len(expected_basenames)
            or set(observed_basenames) != set(expected_basenames)
            or any(
                type(name) is not str
                or name in {"", ".", ".."}
                or "/" in name
                for name in observed_basenames
            )
        ):
            raise ValueError("cleanup hook artifact basename coverage drift")
        staged: list[RetainedWorkerObjectV1] = []
        snapshots: dict[str, os.stat_result] = {}
        for ordinal_offset, basename in enumerate(expected_basenames):
            target_index = (
                None if phase == "INVARIANCE" else STATISTICS_BASENAMES.index(basename)
            )
            retained_object, retained_snapshot = self._retain_one_file(
                    mode=mode,
                    phase=phase,
                    target_index=target_index,
                    basename=basename,
                    directory_fd=artifact_directory_fd,
                    maximum_bytes=maximum_bytes,
                    ordinal=len(self._records) + ordinal_offset,
            )
            staged.append(retained_object)
            snapshots[basename] = retained_snapshot
        self._records.extend(staged)
        return snapshots

    def _require_precleanup_directory_snapshot(
        self,
        directory_fd: int,
        expected_basenames: tuple[str, ...],
        expected_snapshots: Mapping[str, os.stat_result],
    ) -> None:
        self._require_active_owner()
        if set(expected_snapshots) != set(expected_basenames):
            raise ValueError("pre-cleanup snapshot key coverage drift")
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        scan_fd = os.open(".", directory_flags, dir_fd=directory_fd)
        try:
            observed_basenames = tuple(os.listdir(scan_fd))
        finally:
            os.close(scan_fd)
        if (
            len(observed_basenames) != len(expected_basenames)
            or set(observed_basenames) != set(expected_basenames)
        ):
            raise ValueError("pre-cleanup directory basename coverage drift")
        for basename in expected_basenames:
            observed = os.stat(
                basename,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            expected = expected_snapshots[basename]
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or not _same_file_identity(expected, observed)
            ):
                raise ValueError(
                    "pre-cleanup worker artifact identity/timestamp drift"
                )

    def _require_postcleanup_absence(
        self,
        artifact_directory: str,
        artifact_directory_fd: int,
    ) -> None:
        retained = os.fstat(artifact_directory_fd)
        if not stat.S_ISDIR(retained.st_mode):
            raise ValueError("frozen cleanup changed the retained dirfd type")
        if sys_platform_is_linux_v1() and retained.st_nlink != 0:
            raise ValueError(
                "frozen cleanup returned without unlinking the retained directory inode"
            )
        if tuple(os.listdir(artifact_directory_fd)):
            raise ValueError("frozen cleanup returned with a nonempty retained dirfd")
        try:
            os.lstat(artifact_directory)
        except FileNotFoundError:
            return
        raise ValueError("frozen cleanup returned while its pathname still exists")

    def _retain_one_file(
        self,
        *,
        mode: str,
        phase: str,
        target_index: int | None,
        basename: str,
        directory_fd: int,
        maximum_bytes: int,
        ordinal: int,
    ) -> tuple[RetainedWorkerObjectV1, os.stat_result]:
        self._require_active_owner()
        before = os.stat(basename, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum_bytes
        ):
            raise ValueError("worker artifact is not one bounded regular link")
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(basename, flags, dir_fd=directory_fd)
        try:
            opened = os.fstat(fd)
            if not _same_file_identity(before, opened):
                raise ValueError("worker artifact changed during no-follow open")
            sha256, size_bytes = _hash_open_fd_bounded(fd, maximum_bytes)
            metadata_without_request_id: dict[str, object] = {
                "schema_version": RETENTION_REQUEST_SCHEMA_VERSION,
                "attempt_context_id": self.attempt_context_id,
                "retention_session_id": self.retention_session_id,
                "ordinal": ordinal,
                "mode": mode,
                "phase": phase,
                "target_index_or_null": target_index,
                "source_basename": basename,
                "auditor_reported_sha256": sha256,
                "auditor_reported_size_bytes": size_bytes,
                "fd_count": 1,
            }
            request_id = (
                "sha256:"
                + hashlib.sha256(
                    canonicalize_rfc8785_v1(metadata_without_request_id)
                ).hexdigest()
            )
            metadata: dict[str, object] = {
                **metadata_without_request_id,
                "request_id": request_id,
            }
            if set(metadata) != {
                "schema_version",
                "attempt_context_id",
                "retention_session_id",
                "ordinal",
                "mode",
                "phase",
                "target_index_or_null",
                "source_basename",
                "auditor_reported_sha256",
                "auditor_reported_size_bytes",
                "fd_count",
                "request_id",
            }:
                raise RuntimeError("SCM retention request field set drift")
            frozen_metadata = dict(metadata)
            os.lseek(fd, 0, os.SEEK_SET)
            if self._sink_callback_active:
                raise RuntimeError("retention sink callback reentrancy drift")
            self._sink_call_count += 1
            self._sink_callback_active = True
            try:
                acknowledgement = self._sink.ingest_open_fd(
                    metadata,
                    fd,
                    sha256,
                    size_bytes,
                )
            finally:
                self._sink_callback_active = False
            self._require_active_owner()
            if metadata != frozen_metadata:
                raise ValueError("retention sink mutated the SCM request metadata")
            if type(acknowledgement) is not RetentionIngestAckV1:
                raise TypeError("retention sink returned another acknowledgement type")
            acknowledgement.__post_init__()
            if (
                acknowledgement.attempt_context_id != self.attempt_context_id
                or acknowledgement.retention_session_id
                != self.retention_session_id
                or acknowledgement.request_id != request_id
                or acknowledgement.ordinal != ordinal
                or acknowledgement.mode != mode
                or acknowledgement.phase != phase
                or acknowledgement.source_basename != basename
                or acknowledgement.sha256 != sha256
                or acknowledgement.size_bytes != size_bytes
            ):
                raise ValueError("retention acknowledgement request linkage drift")
            self._acknowledged_object_count += 1
            after_sink = os.fstat(fd)
            name_after_sink = os.stat(
                basename, dir_fd=directory_fd, follow_symlinks=False
            )
            if (
                not _same_file_identity(opened, after_sink)
                or not _same_file_identity(opened, name_after_sink)
            ):
                raise ValueError("worker artifact changed during sink ingestion")
            replay_sha256, replay_size = _hash_open_fd_bounded(fd, maximum_bytes)
            if replay_sha256 != sha256 or replay_size != size_bytes:
                raise ValueError("worker artifact changed after sink acknowledgement")
            retained_object = RetainedWorkerObjectV1(
                attempt_context_id=self.attempt_context_id,
                retention_session_id=self.retention_session_id,
                mode=mode,
                phase=phase,
                target_index=target_index,
                source_basename=basename,
                ordinal=ordinal,
                request_id=request_id,
                sha256=sha256,
                size_bytes=size_bytes,
                acknowledgement=acknowledgement,
            )
            return retained_object, name_after_sink
        finally:
            os.close(fd)


_TARGET_FEATURE_ROW_KEYS: Final = {
    "target",
    "feature_id",
    "raw_nmi_hex",
    "symmetric_permutation_mean_nmi_hex",
    "bias_corrected_nmi_hex",
    "balanced_accuracy_advantage_hex",
    "observed_omnibus_hex",
    "pairwise_exceedance_count",
    "pairwise_raw_p",
    "pairwise_holm_adjusted_p",
    "feature_max_t_exceedance_count",
    "feature_max_t_p",
    "target_holm_critical_value",
    "target_family_holm_rejected",
    "material_effect",
    "feature_localization_rejected",
    "failure_witness",
    "passed",
}
_TARGET_FAMILY_ROW_KEYS: Final = {
    "target",
    "observed_target_max_omnibus_hex",
    "exceedance_count",
    "raw_p",
    "holm_adjusted_p",
    "holm_rank",
    "holm_critical_value",
    "holm_rejected",
}
_FEATURE_MANIFEST_ENTRY_KEYS: Final = {
    "feature_index",
    "concrete_feature_id",
    "semantic_aliases",
}
_FEATURE_MANIFEST_CHUNK_KEYS: Final = {
    "schema_version",
    "auditor_schema_version",
    "auditor_semantics_version",
    "attempt_context_id",
    "retention_session_id",
    "mode",
    "receipt_id",
    "feature_manifest_root",
    "concrete_feature_count",
    "chunk_index",
    "feature_start",
    "feature_end_exclusive",
    "feature_count",
    "channel_groups",
    "unique_id_feature_family",
    "uuid_leaf_count",
    "channel_leaf_count",
    "raw_semantic_feature_count",
    "deduplicated_feature_count",
    "entries",
    "projection_scope",
    "claim_boundary",
    "trusted_bootstrap_requirement",
}
_RESULT_CHUNK_KEYS: Final = {
    "schema_version",
    "auditor_schema_version",
    "auditor_semantics_version",
    "attempt_context_id",
    "retention_session_id",
    "mode",
    "receipt_id",
    "audited_row_count",
    "feature_manifest_root",
    "concrete_feature_count",
    "target_feature_result_count",
    "chunk_kind",
    "chunk_index",
    "row_start",
    "row_end_exclusive",
    "row_count",
    "row_projection_sha256",
    "rows",
    "projection_scope",
    "claim_boundary",
    "trusted_bootstrap_requirement",
}
_RESULT_MANIFEST_KEYS: Final = {
    "schema_version",
    "auditor_schema_version",
    "auditor_semantics_version",
    "attempt_context_id",
    "retention_session_id",
    "mode",
    "receipt_id",
    "audited_row_count",
    "feature_manifest_root",
    "concrete_feature_count",
    "target_order",
    "target_feature_result_count",
    "target_family_result_count",
    "rows_per_nonfinal_target_feature_chunk",
    "maximum_chunk_bytes",
    "feature_manifest_chunk_schema_version",
    "maximum_feature_manifest_entries_per_chunk",
    "maximum_feature_manifest_chunk_bytes",
    "feature_manifest_chunks",
    "target_feature_chunks",
    "target_family_chunk",
    "projection_scope",
    "claim_boundary",
    "trusted_bootstrap_requirement",
}
_FEATURE_MANIFEST_CHUNK_REF_KEYS: Final = {
    "chunk_index",
    "feature_start",
    "feature_end_exclusive",
    "sha256",
    "size_bytes",
}
_RESULT_CHUNK_REF_KEYS: Final = {
    "chunk_index",
    "row_start",
    "row_end_exclusive",
    "row_projection_sha256",
    "sha256",
    "size_bytes",
}
_STATISTICAL_RESULTS_PROJECTION_SCOPE: Final = (
    "STATISTICAL_RESULTS_TARGET_FAMILY_AND_EXACT_FEATURE_MANIFEST_PREIMAGE_"
    "ONLY_NOT_FULL_RECEIPT"
)


def _decoded_safe_integer_v1(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = 2**53,
) -> int:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or not value.is_integer()
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} is not one decoded exact safe integer")
    return int(value)


def _require_receipt_id_v1(value: object, name: str) -> str:
    if type(value) is not str or _AUDITOR_RECEIPT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} is not one exact auditor receipt ID")
    return value


def _require_canonical_object_v1(payload: object, name: str) -> dict[str, object]:
    if type(payload) is not bytes or not 0 < len(payload) <= MAXIMUM_RESULT_CHUNK_BYTES:
        raise ValueError(f"{name} is not bounded exact bytes")
    decoded = decode_and_require_canonical_rfc8785_v1(payload)
    if type(decoded) is not dict:
        raise ValueError(f"{name} is not one canonical object")
    return decoded


def _require_canonical_hex_float_v1(value: object, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} is not a hexadecimal binary64 string")
    try:
        replay = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not a hexadecimal binary64 string") from exc
    if not math.isfinite(replay) or replay.hex() != value:
        raise ValueError(f"{name} is not one canonical finite binary64 string")
    return value


def _require_probability_projection_v1(value: object, name: str) -> None:
    if type(value) is not dict or set(value) != {"numerator", "denominator"}:
        raise ValueError(f"{name} probability field set drift")
    numerator = _decoded_safe_integer_v1(value["numerator"], name + " numerator")
    denominator = _decoded_safe_integer_v1(
        value["denominator"],
        name + " denominator",
        minimum=1,
    )
    if numerator > denominator or math.gcd(numerator, denominator) != 1:
        raise ValueError(f"{name} is not one reduced unit-interval probability")


def _validate_target_feature_projection_row_v1(
    value: object,
    *,
    expected_target: str,
) -> None:
    if type(value) is not dict or set(value) != _TARGET_FEATURE_ROW_KEYS:
        raise ValueError("target-feature result row field set drift")
    if value["target"] != expected_target:
        raise ValueError("target-feature result row target/order drift")
    if type(value["feature_id"]) is not str or not value["feature_id"]:
        raise ValueError("target-feature result row feature ID drift")
    for name in (
        "raw_nmi_hex",
        "symmetric_permutation_mean_nmi_hex",
        "bias_corrected_nmi_hex",
        "balanced_accuracy_advantage_hex",
        "observed_omnibus_hex",
    ):
        _require_canonical_hex_float_v1(value[name], name)
    for name in (
        "pairwise_exceedance_count",
        "feature_max_t_exceedance_count",
    ):
        _decoded_safe_integer_v1(
            value[name],
            name,
            maximum=auditor.LABEL_PERMUTATIONS,
        )
    for name in (
        "pairwise_raw_p",
        "pairwise_holm_adjusted_p",
        "feature_max_t_p",
        "target_holm_critical_value",
    ):
        _require_probability_projection_v1(value[name], name)
    for name in (
        "target_family_holm_rejected",
        "material_effect",
        "feature_localization_rejected",
        "failure_witness",
        "passed",
    ):
        if type(value[name]) is not bool:
            raise ValueError(f"target-feature result {name} type drift")


def _validate_target_family_projection_row_v1(
    value: object,
    *,
    expected_target: str,
) -> None:
    if type(value) is not dict or set(value) != _TARGET_FAMILY_ROW_KEYS:
        raise ValueError("target-family result row field set drift")
    if value["target"] != expected_target:
        raise ValueError("target-family result row target/order drift")
    _require_canonical_hex_float_v1(
        value["observed_target_max_omnibus_hex"],
        "observed target maximum omnibus",
    )
    _decoded_safe_integer_v1(
        value["exceedance_count"],
        "target-family exceedance count",
        maximum=auditor.LABEL_PERMUTATIONS,
    )
    _decoded_safe_integer_v1(
        value["holm_rank"],
        "target-family Holm rank",
        minimum=1,
        maximum=len(auditor.TARGETS),
    )
    for name in ("raw_p", "holm_adjusted_p", "holm_critical_value"):
        _require_probability_projection_v1(value[name], name)
    if type(value["holm_rejected"]) is not bool:
        raise ValueError("target-family Holm decision type drift")


@dataclass(frozen=True, slots=True)
class CanonicalFeatureManifestChunkV1:
    attempt_context_id: str
    retention_session_id: str
    mode: str
    receipt_id: str
    feature_manifest_root: str
    concrete_feature_count: int
    chunk_index: int
    feature_start: int
    feature_end_exclusive: int
    sha256: str
    size_bytes: int
    canonical_bytes: bytes

    def __post_init__(self) -> None:
        if type(self) is not CanonicalFeatureManifestChunkV1:
            raise TypeError("feature-manifest chunk must use the exact DTO")
        _require_identifier(
            self.attempt_context_id,
            "feature-manifest attempt context ID",
        )
        _require_identifier(
            self.retention_session_id,
            "feature-manifest retention session ID",
        )
        if type(self.mode) is not str or self.mode not in MODE_ORDER:
            raise ValueError("feature-manifest chunk mode drift")
        _require_receipt_id_v1(self.receipt_id, "feature-manifest receipt ID")
        if (
            type(self.feature_manifest_root) is not str
            or not self.feature_manifest_root
        ):
            raise ValueError("feature-manifest root drift")
        if (
            type(self.concrete_feature_count) is not int
            or not 1 <= self.concrete_feature_count <= auditor.MAXIMUM_FEATURES
            or type(self.chunk_index) is not int
            or self.chunk_index < 0
            or type(self.feature_start) is not int
            or type(self.feature_end_exclusive) is not int
            or not 0 <= self.feature_start < self.feature_end_exclusive
            <= self.concrete_feature_count
        ):
            raise ValueError("feature-manifest chunk range/count drift")
        _require_sha256(self.sha256, "feature-manifest chunk SHA-256")
        if (
            type(self.canonical_bytes) is not bytes
            or type(self.size_bytes) is not int
            or self.size_bytes != len(self.canonical_bytes)
            or not 0 < self.size_bytes <= MAXIMUM_RESULT_CHUNK_BYTES
            or self.sha256
            != "sha256:" + hashlib.sha256(self.canonical_bytes).hexdigest()
        ):
            raise ValueError("feature-manifest chunk bytes/content-address drift")
        body = _require_canonical_object_v1(
            self.canonical_bytes,
            "feature-manifest chunk",
        )
        if set(body) != _FEATURE_MANIFEST_CHUNK_KEYS:
            raise ValueError("feature-manifest chunk canonical field set drift")
        string_bindings = {
            "schema_version": FEATURE_MANIFEST_CHUNK_SCHEMA_VERSION,
            "auditor_schema_version": auditor.SCHEMA_VERSION,
            "auditor_semantics_version": auditor.SEMANTICS_VERSION,
            "attempt_context_id": self.attempt_context_id,
            "retention_session_id": self.retention_session_id,
            "mode": self.mode,
            "receipt_id": self.receipt_id,
            "feature_manifest_root": self.feature_manifest_root,
            "projection_scope": _STATISTICAL_RESULTS_PROJECTION_SCOPE,
            "claim_boundary": RESULT_PROJECTION_CLAIM_BOUNDARY,
            "trusted_bootstrap_requirement": TRUSTED_BOOTSTRAP_REQUIREMENT,
        }
        if any(body[name] != expected for name, expected in string_bindings.items()):
            raise ValueError("feature-manifest chunk string binding drift")
        feature_count = self.feature_end_exclusive - self.feature_start
        integer_bindings = {
            "concrete_feature_count": self.concrete_feature_count,
            "chunk_index": self.chunk_index,
            "feature_start": self.feature_start,
            "feature_end_exclusive": self.feature_end_exclusive,
            "feature_count": feature_count,
            "deduplicated_feature_count": self.concrete_feature_count,
        }
        for name, expected in integer_bindings.items():
            if _decoded_safe_integer_v1(body[name], name) != expected:
                raise ValueError("feature-manifest chunk integer binding drift")
        if (
            body["channel_groups"] != list(auditor.CHANNEL_GROUPS)
            or body["unique_id_feature_family"]
            != list(auditor.UNIQUE_ID_FEATURE_FAMILY)
        ):
            raise ValueError("feature-manifest family projection drift")
        uuid_leaf_count = _decoded_safe_integer_v1(
            body["uuid_leaf_count"],
            "feature-manifest UUID leaf count",
            minimum=1,
        )
        channel_leaf_count = _decoded_safe_integer_v1(
            body["channel_leaf_count"],
            "feature-manifest channel leaf count",
            minimum=1,
        )
        raw_count = _decoded_safe_integer_v1(
            body["raw_semantic_feature_count"],
            "feature-manifest raw semantic count",
            minimum=self.concrete_feature_count,
        )
        if uuid_leaf_count < 1 or channel_leaf_count < 1 or raw_count < 1:
            raise ValueError("feature-manifest count projection drift")
        entries = body["entries"]
        if type(entries) is not list or len(entries) != feature_count:
            raise ValueError("feature-manifest chunk entry coverage drift")
        for offset, entry in enumerate(entries):
            if type(entry) is not dict or set(entry) != _FEATURE_MANIFEST_ENTRY_KEYS:
                raise ValueError("feature-manifest entry field set drift")
            if _decoded_safe_integer_v1(
                entry["feature_index"],
                "feature-manifest feature index",
            ) != self.feature_start + offset:
                raise ValueError("feature-manifest feature index/order drift")
            concrete_feature_id = entry["concrete_feature_id"]
            semantic_aliases = entry["semantic_aliases"]
            if (
                type(concrete_feature_id) is not str
                or type(semantic_aliases) is not list
                or not semantic_aliases
                or any(type(alias) is not str or not alias for alias in semantic_aliases)
                or semantic_aliases != sorted(set(semantic_aliases))
            ):
                raise ValueError("feature-manifest alias projection drift")
            if concrete_feature_id != auditor.stable_hash(
                {"aliases": tuple(semantic_aliases)},
                prefix="phase2b_formal_covert_feature_",
            ):
                raise ValueError("feature-manifest concrete ID preimage replay drift")


def _feature_manifest_chunk_body_v1(
    *,
    manifest: auditor.FeatureManifestV1,
    attempt_context_id: str,
    retention_session_id: str,
    mode: str,
    receipt_id: str,
    chunk_index: int,
    feature_start: int,
    feature_end_exclusive: int,
) -> dict[str, object]:
    entries = [
        {
            "feature_index": feature_index,
            "concrete_feature_id": item.concrete_feature_id,
            "semantic_aliases": list(item.semantic_aliases),
        }
        for feature_index, item in enumerate(
            manifest.alias_coverage[feature_start:feature_end_exclusive],
            start=feature_start,
        )
    ]
    return {
        "schema_version": FEATURE_MANIFEST_CHUNK_SCHEMA_VERSION,
        "auditor_schema_version": auditor.SCHEMA_VERSION,
        "auditor_semantics_version": auditor.SEMANTICS_VERSION,
        "attempt_context_id": attempt_context_id,
        "retention_session_id": retention_session_id,
        "mode": mode,
        "receipt_id": receipt_id,
        "feature_manifest_root": manifest.manifest_root,
        "concrete_feature_count": manifest.deduplicated_feature_count,
        "chunk_index": chunk_index,
        "feature_start": feature_start,
        "feature_end_exclusive": feature_end_exclusive,
        "feature_count": feature_end_exclusive - feature_start,
        "channel_groups": list(manifest.channel_groups),
        "unique_id_feature_family": list(manifest.unique_id_feature_family),
        "uuid_leaf_count": manifest.uuid_leaf_count,
        "channel_leaf_count": manifest.channel_leaf_count,
        "raw_semantic_feature_count": manifest.raw_semantic_feature_count,
        "deduplicated_feature_count": manifest.deduplicated_feature_count,
        "entries": entries,
        "projection_scope": _STATISTICAL_RESULTS_PROJECTION_SCOPE,
        "claim_boundary": RESULT_PROJECTION_CLAIM_BOUNDARY,
        "trusted_bootstrap_requirement": TRUSTED_BOOTSTRAP_REQUIREMENT,
    }


def _canonical_feature_manifest_chunks_v1(
    *,
    manifest: auditor.FeatureManifestV1,
    attempt_context_id: str,
    retention_session_id: str,
    mode: str,
    receipt_id: str,
) -> tuple[CanonicalFeatureManifestChunkV1, ...]:
    if type(manifest) is not auditor.FeatureManifestV1:
        raise TypeError("feature-manifest projection requires the exact DTO")
    observed_root = manifest.manifest_root
    manifest.__post_init__()
    if manifest.manifest_root != observed_root:
        raise ValueError("feature-manifest live root changed during exact replay")
    total = manifest.deduplicated_feature_count
    if not 1 <= total <= auditor.MAXIMUM_FEATURES:
        raise ValueError("feature-manifest projection count exceeds its bound")
    chunks: list[CanonicalFeatureManifestChunkV1] = []
    feature_start = 0
    while feature_start < total:
        chunk_index = len(chunks)
        maximum_end = min(
            total,
            feature_start + FEATURE_MANIFEST_ENTRIES_PER_NONFINAL_CHUNK,
        )

        def payload_for(feature_end_exclusive: int) -> bytes:
            return canonicalize_rfc8785_v1(
                _feature_manifest_chunk_body_v1(
                    manifest=manifest,
                    attempt_context_id=attempt_context_id,
                    retention_session_id=retention_session_id,
                    mode=mode,
                    receipt_id=receipt_id,
                    chunk_index=chunk_index,
                    feature_start=feature_start,
                    feature_end_exclusive=feature_end_exclusive,
                )
            )

        selected_end = maximum_end
        selected_payload = payload_for(selected_end)
        if len(selected_payload) > MAXIMUM_RESULT_CHUNK_BYTES:
            low = feature_start + 1
            high = maximum_end
            selected_end = feature_start
            selected_payload = b""
            while low <= high:
                candidate_end = (low + high) // 2
                candidate_payload = payload_for(candidate_end)
                if len(candidate_payload) <= MAXIMUM_RESULT_CHUNK_BYTES:
                    selected_end = candidate_end
                    selected_payload = candidate_payload
                    low = candidate_end + 1
                else:
                    high = candidate_end - 1
            if selected_end == feature_start:
                raise ValueError(
                    "one canonical feature-manifest entry exceeds two MiB"
                )
        chunk = CanonicalFeatureManifestChunkV1(
            attempt_context_id=attempt_context_id,
            retention_session_id=retention_session_id,
            mode=mode,
            receipt_id=receipt_id,
            feature_manifest_root=manifest.manifest_root,
            concrete_feature_count=total,
            chunk_index=chunk_index,
            feature_start=feature_start,
            feature_end_exclusive=selected_end,
            sha256="sha256:" + hashlib.sha256(selected_payload).hexdigest(),
            size_bytes=len(selected_payload),
            canonical_bytes=selected_payload,
        )
        chunks.append(chunk)
        feature_start = selected_end
    return tuple(chunks)


def _feature_manifest_chunk_reference_v1(
    value: CanonicalFeatureManifestChunkV1,
) -> dict[str, object]:
    return {
        "chunk_index": value.chunk_index,
        "feature_start": value.feature_start,
        "feature_end_exclusive": value.feature_end_exclusive,
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
    }


def _require_decoded_feature_manifest_chunk_reference_v1(
    value: object,
    expected: CanonicalFeatureManifestChunkV1,
    name: str,
) -> None:
    if type(value) is not dict or set(value) != _FEATURE_MANIFEST_CHUNK_REF_KEYS:
        raise ValueError(f"{name} field set drift")
    for field_name in (
        "chunk_index",
        "feature_start",
        "feature_end_exclusive",
        "size_bytes",
    ):
        if _decoded_safe_integer_v1(
            value[field_name],
            name + " " + field_name,
        ) != getattr(expected, field_name):
            raise ValueError(f"{name} integer binding drift")
    if value["sha256"] != expected.sha256:
        raise ValueError(f"{name} content reference drift")


def _replay_feature_manifest_chunks_v1(
    *,
    chunks: tuple[CanonicalFeatureManifestChunkV1, ...],
    attempt_context_id: str,
    retention_session_id: str,
    mode: str,
    receipt_id: str,
    feature_manifest_root: str,
    concrete_feature_count: int,
) -> auditor.FeatureManifestV1:
    if type(chunks) is not tuple or not chunks:
        raise ValueError("feature-manifest chunk container is empty or non-exact")
    expected_start = 0
    concrete_feature_ids: list[str] = []
    alias_coverage: list[auditor.FeatureAliasCoverageV1] = []
    replay_metadata: tuple[tuple[str, ...], tuple[str, ...], int, int, int, int] | None = None
    for expected_index, chunk in enumerate(chunks):
        if type(chunk) is not CanonicalFeatureManifestChunkV1:
            raise TypeError("feature-manifest chunk uses another DTO")
        chunk.__post_init__()
        if (
            chunk.attempt_context_id != attempt_context_id
            or chunk.retention_session_id != retention_session_id
            or chunk.mode != mode
            or chunk.receipt_id != receipt_id
            or chunk.feature_manifest_root != feature_manifest_root
            or chunk.concrete_feature_count != concrete_feature_count
            or chunk.chunk_index != expected_index
            or chunk.feature_start != expected_start
        ):
            raise ValueError("feature-manifest chunk common binding/order drift")
        body = _require_canonical_object_v1(
            chunk.canonical_bytes,
            "feature-manifest replay chunk",
        )
        channel_groups = tuple(body["channel_groups"])
        unique_id_feature_family = tuple(body["unique_id_feature_family"])
        metadata = (
            channel_groups,
            unique_id_feature_family,
            _decoded_safe_integer_v1(
                body["uuid_leaf_count"],
                "feature-manifest replay UUID leaf count",
                minimum=1,
            ),
            _decoded_safe_integer_v1(
                body["channel_leaf_count"],
                "feature-manifest replay channel leaf count",
                minimum=1,
            ),
            _decoded_safe_integer_v1(
                body["raw_semantic_feature_count"],
                "feature-manifest replay raw count",
                minimum=concrete_feature_count,
            ),
            _decoded_safe_integer_v1(
                body["deduplicated_feature_count"],
                "feature-manifest replay deduplicated count",
                minimum=1,
            ),
        )
        if replay_metadata is None:
            replay_metadata = metadata
        elif metadata != replay_metadata:
            raise ValueError("feature-manifest metadata differs across chunks")
        entries = body["entries"]
        if type(entries) is not list:
            raise ValueError("feature-manifest replay entries are not an exact list")
        for entry in entries:
            concrete_feature_id = entry["concrete_feature_id"]
            semantic_aliases = tuple(entry["semantic_aliases"])
            concrete_feature_ids.append(concrete_feature_id)
            alias_coverage.append(
                auditor.FeatureAliasCoverageV1(
                    concrete_feature_id=concrete_feature_id,
                    semantic_aliases=semantic_aliases,
                )
            )
        expected_start = chunk.feature_end_exclusive
    if (
        expected_start != concrete_feature_count
        or len(concrete_feature_ids) != concrete_feature_count
        or replay_metadata is None
        or replay_metadata[5] != concrete_feature_count
    ):
        raise ValueError("feature-manifest chunk coverage is incomplete")
    replayed = auditor.FeatureManifestV1(
        concrete_feature_ids=tuple(concrete_feature_ids),
        alias_coverage=tuple(alias_coverage),
        channel_groups=replay_metadata[0],
        unique_id_feature_family=replay_metadata[1],
        uuid_leaf_count=replay_metadata[2],
        channel_leaf_count=replay_metadata[3],
        raw_semantic_feature_count=replay_metadata[4],
        deduplicated_feature_count=replay_metadata[5],
    )
    if replayed.manifest_root != feature_manifest_root:
        raise ValueError("feature-manifest preimage does not replay the bound root")
    regenerated = _canonical_feature_manifest_chunks_v1(
        manifest=replayed,
        attempt_context_id=attempt_context_id,
        retention_session_id=retention_session_id,
        mode=mode,
        receipt_id=receipt_id,
    )
    if regenerated != chunks:
        raise ValueError("feature-manifest chunk split/canonical bytes are not exact")
    return replayed


@dataclass(frozen=True, slots=True)
class CanonicalResultChunkV1:
    attempt_context_id: str
    retention_session_id: str
    mode: str
    receipt_id: str
    audited_row_count: int
    feature_manifest_root: str
    concrete_feature_count: int
    target_feature_result_count: int
    chunk_kind: str
    chunk_index: int
    row_start: int
    row_end_exclusive: int
    row_projection_sha256: str
    sha256: str
    size_bytes: int
    canonical_bytes: bytes

    def __post_init__(self) -> None:
        if type(self) is not CanonicalResultChunkV1:
            raise TypeError("result chunk must use the exact DTO")
        _require_identifier(self.attempt_context_id, "result attempt context ID")
        _require_identifier(
            self.retention_session_id,
            "result retention session ID",
        )
        if type(self.mode) is not str or self.mode not in MODE_ORDER:
            raise ValueError("result chunk mode drift")
        _require_receipt_id_v1(self.receipt_id, "result chunk receipt ID")
        if self.audited_row_count != auditor.MAXIMUM_ROWS:
            raise ValueError("result chunk is not bound to the exact 1,024-row audit")
        if type(self.feature_manifest_root) is not str or not self.feature_manifest_root:
            raise ValueError("result chunk feature-manifest root drift")
        if (
            type(self.concrete_feature_count) is not int
            or not 1 <= self.concrete_feature_count <= auditor.MAXIMUM_FEATURES
            or self.target_feature_result_count
            != len(auditor.TARGETS) * self.concrete_feature_count
        ):
            raise ValueError("result chunk feature/result count drift")
        if (
            type(self.chunk_kind) is not str
            or self.chunk_kind
            not in {"TARGET_FEATURE_RESULTS", "TARGET_FAMILY_RESULTS"}
        ):
            raise ValueError("result chunk kind drift")
        if (
            type(self.chunk_index) is not int
            or self.chunk_index < 0
            or type(self.row_start) is not int
            or type(self.row_end_exclusive) is not int
            or not 0 <= self.row_start < self.row_end_exclusive
        ):
            raise ValueError("result chunk range/index drift")
        row_count = self.row_end_exclusive - self.row_start
        if self.chunk_kind == "TARGET_FEATURE_RESULTS":
            if (
                row_count > RESULT_ROWS_PER_NONFINAL_CHUNK
                or self.row_end_exclusive > self.target_feature_result_count
            ):
                raise ValueError("target-feature result chunk range exceeds its bound")
        elif (
            self.chunk_index != 0
            or self.row_start != 0
            or self.row_end_exclusive != len(auditor.TARGETS)
        ):
            raise ValueError("target-family result chunk range drift")
        _require_sha256(self.row_projection_sha256, "result row projection SHA-256")
        _require_sha256(self.sha256, "result chunk SHA-256")
        if (
            type(self.canonical_bytes) is not bytes
            or type(self.size_bytes) is not int
            or self.size_bytes != len(self.canonical_bytes)
            or not 0 < self.size_bytes <= MAXIMUM_RESULT_CHUNK_BYTES
            or self.sha256
            != "sha256:" + hashlib.sha256(self.canonical_bytes).hexdigest()
        ):
            raise ValueError("result chunk bytes/content-address drift")
        body = _require_canonical_object_v1(
            self.canonical_bytes,
            "statistical-results chunk",
        )
        if set(body) != _RESULT_CHUNK_KEYS:
            raise ValueError("statistical-results chunk canonical field set drift")
        string_bindings = {
            "schema_version": STATISTICAL_RESULTS_CHUNK_SCHEMA_VERSION,
            "auditor_schema_version": auditor.SCHEMA_VERSION,
            "auditor_semantics_version": auditor.SEMANTICS_VERSION,
            "attempt_context_id": self.attempt_context_id,
            "retention_session_id": self.retention_session_id,
            "mode": self.mode,
            "receipt_id": self.receipt_id,
            "feature_manifest_root": self.feature_manifest_root,
            "chunk_kind": self.chunk_kind,
            "row_projection_sha256": self.row_projection_sha256,
            "projection_scope": _STATISTICAL_RESULTS_PROJECTION_SCOPE,
            "claim_boundary": RESULT_PROJECTION_CLAIM_BOUNDARY,
            "trusted_bootstrap_requirement": TRUSTED_BOOTSTRAP_REQUIREMENT,
        }
        if any(body[name] != expected for name, expected in string_bindings.items()):
            raise ValueError("statistical-results chunk string binding drift")
        integer_bindings = {
            "audited_row_count": self.audited_row_count,
            "concrete_feature_count": self.concrete_feature_count,
            "target_feature_result_count": self.target_feature_result_count,
            "chunk_index": self.chunk_index,
            "row_start": self.row_start,
            "row_end_exclusive": self.row_end_exclusive,
            "row_count": row_count,
        }
        for name, expected in integer_bindings.items():
            if _decoded_safe_integer_v1(body[name], name) != expected:
                raise ValueError("statistical-results chunk integer binding drift")
        rows = body["rows"]
        if type(rows) is not list or len(rows) != row_count:
            raise ValueError("statistical-results chunk row coverage drift")
        observed_rows_sha256 = (
            "sha256:" + hashlib.sha256(canonicalize_rfc8785_v1(rows)).hexdigest()
        )
        if observed_rows_sha256 != self.row_projection_sha256:
            raise ValueError("statistical-results chunk row projection hash drift")
        if self.chunk_kind == "TARGET_FEATURE_RESULTS":
            for offset, row in enumerate(rows):
                global_index = self.row_start + offset
                expected_target = auditor.TARGETS[
                    global_index // self.concrete_feature_count
                ]
                _validate_target_feature_projection_row_v1(
                    row,
                    expected_target=expected_target,
                )
        else:
            for expected_target, row in zip(auditor.TARGETS, rows, strict=True):
                _validate_target_family_projection_row_v1(
                    row,
                    expected_target=expected_target,
                )


def _chunk_reference_v1(value: CanonicalResultChunkV1) -> dict[str, object]:
    return {
        "chunk_index": value.chunk_index,
        "row_start": value.row_start,
        "row_end_exclusive": value.row_end_exclusive,
        "row_projection_sha256": value.row_projection_sha256,
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
    }


def _require_decoded_chunk_reference_v1(
    value: object,
    expected: CanonicalResultChunkV1,
    name: str,
) -> None:
    if type(value) is not dict or set(value) != _RESULT_CHUNK_REF_KEYS:
        raise ValueError(f"{name} field set drift")
    for field_name in ("chunk_index", "row_start", "row_end_exclusive", "size_bytes"):
        if _decoded_safe_integer_v1(value[field_name], name + " " + field_name) != getattr(
            expected,
            field_name,
        ):
            raise ValueError(f"{name} integer binding drift")
    if (
        value["row_projection_sha256"] != expected.row_projection_sha256
        or value["sha256"] != expected.sha256
    ):
        raise ValueError(f"{name} content reference drift")


@dataclass(frozen=True, slots=True)
class StatisticalResultsProjectionV1:
    attempt_context_id: str
    retention_session_id: str
    mode: str
    receipt_id: str
    audited_row_count: int
    feature_manifest_root: str
    concrete_feature_count: int
    target_feature_result_count: int
    feature_manifest_chunks: tuple[CanonicalFeatureManifestChunkV1, ...]
    target_feature_chunks: tuple[CanonicalResultChunkV1, ...]
    target_family_chunk: CanonicalResultChunkV1
    manifest_sha256: str
    manifest_size_bytes: int
    manifest_canonical_bytes: bytes

    def __post_init__(self) -> None:
        if type(self) is not StatisticalResultsProjectionV1:
            raise TypeError("statistical-results projection must use the exact DTO")
        _require_identifier(self.attempt_context_id, "projection attempt context ID")
        _require_identifier(
            self.retention_session_id,
            "projection retention session ID",
        )
        if type(self.mode) is not str or self.mode not in MODE_ORDER:
            raise ValueError("statistical-results projection mode drift")
        _require_receipt_id_v1(self.receipt_id, "projection receipt ID")
        if self.audited_row_count != auditor.MAXIMUM_ROWS:
            raise ValueError("projection is not bound to the exact 1,024-row audit")
        if type(self.feature_manifest_root) is not str or not self.feature_manifest_root:
            raise ValueError("projection feature-manifest root drift")
        if (
            type(self.concrete_feature_count) is not int
            or not 1 <= self.concrete_feature_count <= auditor.MAXIMUM_FEATURES
            or self.target_feature_result_count
            != len(auditor.TARGETS) * self.concrete_feature_count
        ):
            raise ValueError("projection feature/result count drift")
        if (
            type(self.feature_manifest_chunks) is not tuple
            or not self.feature_manifest_chunks
            or type(self.target_feature_chunks) is not tuple
            or not self.target_feature_chunks
            or type(self.target_family_chunk) is not CanonicalResultChunkV1
        ):
            raise ValueError("statistical-results chunk container drift")
        replayed_feature_manifest = _replay_feature_manifest_chunks_v1(
            chunks=self.feature_manifest_chunks,
            attempt_context_id=self.attempt_context_id,
            retention_session_id=self.retention_session_id,
            mode=self.mode,
            receipt_id=self.receipt_id,
            feature_manifest_root=self.feature_manifest_root,
            concrete_feature_count=self.concrete_feature_count,
        )
        expected_common = (
            self.attempt_context_id,
            self.retention_session_id,
            self.mode,
            self.receipt_id,
            self.audited_row_count,
            self.feature_manifest_root,
            self.concrete_feature_count,
            self.target_feature_result_count,
        )
        expected_start = 0
        for expected_index, item in enumerate(self.target_feature_chunks):
            if type(item) is not CanonicalResultChunkV1:
                raise TypeError("target-feature chunk uses another DTO")
            item.__post_init__()
            observed_common = (
                item.attempt_context_id,
                item.retention_session_id,
                item.mode,
                item.receipt_id,
                item.audited_row_count,
                item.feature_manifest_root,
                item.concrete_feature_count,
                item.target_feature_result_count,
            )
            if (
                observed_common != expected_common
                or item.chunk_kind != "TARGET_FEATURE_RESULTS"
                or item.chunk_index != expected_index
                or item.row_start != expected_start
                or (
                    expected_index < len(self.target_feature_chunks) - 1
                    and item.row_end_exclusive - item.row_start
                    != RESULT_ROWS_PER_NONFINAL_CHUNK
                )
                or item.row_end_exclusive - item.row_start
                > RESULT_ROWS_PER_NONFINAL_CHUNK
            ):
                raise ValueError("statistical target-feature chunk binding/order drift")
            decoded_chunk = _require_canonical_object_v1(
                item.canonical_bytes,
                "statistical target-feature replay chunk",
            )
            for offset, row in enumerate(decoded_chunk["rows"]):
                global_index = item.row_start + offset
                expected_feature_id = replayed_feature_manifest.concrete_feature_ids[
                    global_index % self.concrete_feature_count
                ]
                if row["feature_id"] != expected_feature_id:
                    raise ValueError(
                        "statistical target-feature row/feature-manifest preimage drift"
                    )
            expected_start = item.row_end_exclusive
        if expected_start != self.target_feature_result_count:
            raise ValueError("statistical target-feature chunk coverage is incomplete")
        self.target_family_chunk.__post_init__()
        family_common = (
            self.target_family_chunk.attempt_context_id,
            self.target_family_chunk.retention_session_id,
            self.target_family_chunk.mode,
            self.target_family_chunk.receipt_id,
            self.target_family_chunk.audited_row_count,
            self.target_family_chunk.feature_manifest_root,
            self.target_family_chunk.concrete_feature_count,
            self.target_family_chunk.target_feature_result_count,
        )
        if (
            family_common != expected_common
            or self.target_family_chunk.chunk_kind != "TARGET_FAMILY_RESULTS"
        ):
            raise ValueError("statistical target-family chunk binding drift")
        _require_sha256(self.manifest_sha256, "statistical-results manifest SHA-256")
        if (
            type(self.manifest_canonical_bytes) is not bytes
            or type(self.manifest_size_bytes) is not int
            or self.manifest_size_bytes != len(self.manifest_canonical_bytes)
            or not 0 < self.manifest_size_bytes <= MAXIMUM_RESULT_CHUNK_BYTES
            or self.manifest_sha256
            != "sha256:"
            + hashlib.sha256(self.manifest_canonical_bytes).hexdigest()
        ):
            raise ValueError("statistical-results manifest bytes/content-address drift")
        body = _require_canonical_object_v1(
            self.manifest_canonical_bytes,
            "statistical-results manifest",
        )
        if set(body) != _RESULT_MANIFEST_KEYS:
            raise ValueError("statistical-results manifest canonical field set drift")
        string_bindings = {
            "schema_version": STATISTICAL_RESULTS_MANIFEST_SCHEMA_VERSION,
            "auditor_schema_version": auditor.SCHEMA_VERSION,
            "auditor_semantics_version": auditor.SEMANTICS_VERSION,
            "attempt_context_id": self.attempt_context_id,
            "retention_session_id": self.retention_session_id,
            "mode": self.mode,
            "receipt_id": self.receipt_id,
            "feature_manifest_root": self.feature_manifest_root,
            "feature_manifest_chunk_schema_version": (
                FEATURE_MANIFEST_CHUNK_SCHEMA_VERSION
            ),
            "projection_scope": _STATISTICAL_RESULTS_PROJECTION_SCOPE,
            "claim_boundary": RESULT_PROJECTION_CLAIM_BOUNDARY,
            "trusted_bootstrap_requirement": TRUSTED_BOOTSTRAP_REQUIREMENT,
        }
        if any(body[name] != expected for name, expected in string_bindings.items()):
            raise ValueError("statistical-results manifest string binding drift")
        integer_bindings = {
            "audited_row_count": self.audited_row_count,
            "concrete_feature_count": self.concrete_feature_count,
            "target_feature_result_count": self.target_feature_result_count,
            "target_family_result_count": len(auditor.TARGETS),
            "rows_per_nonfinal_target_feature_chunk": RESULT_ROWS_PER_NONFINAL_CHUNK,
            "maximum_chunk_bytes": MAXIMUM_RESULT_CHUNK_BYTES,
            "maximum_feature_manifest_entries_per_chunk": (
                FEATURE_MANIFEST_ENTRIES_PER_NONFINAL_CHUNK
            ),
            "maximum_feature_manifest_chunk_bytes": MAXIMUM_RESULT_CHUNK_BYTES,
        }
        for name, expected in integer_bindings.items():
            if _decoded_safe_integer_v1(body[name], name) != expected:
                raise ValueError("statistical-results manifest integer binding drift")
        if body["target_order"] != list(auditor.TARGETS):
            raise ValueError("statistical-results manifest target order drift")
        feature_refs = body["feature_manifest_chunks"]
        if (
            type(feature_refs) is not list
            or len(feature_refs) != len(self.feature_manifest_chunks)
        ):
            raise ValueError(
                "statistical-results manifest feature-chunk-ref count drift"
            )
        for index, (observed, expected) in enumerate(
            zip(feature_refs, self.feature_manifest_chunks, strict=True)
        ):
            _require_decoded_feature_manifest_chunk_reference_v1(
                observed,
                expected,
                f"feature-manifest chunk ref {index}",
            )
        refs = body["target_feature_chunks"]
        if type(refs) is not list or len(refs) != len(self.target_feature_chunks):
            raise ValueError("statistical-results manifest chunk-ref count drift")
        for index, (observed, expected) in enumerate(
            zip(refs, self.target_feature_chunks, strict=True)
        ):
            _require_decoded_chunk_reference_v1(
                observed,
                expected,
                f"target-feature chunk ref {index}",
            )
        _require_decoded_chunk_reference_v1(
            body["target_family_chunk"],
            self.target_family_chunk,
            "target-family chunk ref",
        )


def _canonical_result_chunk_v1(
    *,
    attempt_context_id: str,
    retention_session_id: str,
    mode: str,
    receipt_id: str,
    audited_row_count: int,
    feature_manifest_root: str,
    concrete_feature_count: int,
    target_feature_result_count: int,
    chunk_kind: str,
    chunk_index: int,
    row_start: int,
    row_end_exclusive: int,
    rows: list[dict[str, object]],
) -> CanonicalResultChunkV1:
    rows_sha256 = (
        "sha256:" + hashlib.sha256(canonicalize_rfc8785_v1(rows)).hexdigest()
    )
    body: dict[str, object] = {
        "schema_version": STATISTICAL_RESULTS_CHUNK_SCHEMA_VERSION,
        "auditor_schema_version": auditor.SCHEMA_VERSION,
        "auditor_semantics_version": auditor.SEMANTICS_VERSION,
        "attempt_context_id": attempt_context_id,
        "retention_session_id": retention_session_id,
        "mode": mode,
        "receipt_id": receipt_id,
        "audited_row_count": audited_row_count,
        "feature_manifest_root": feature_manifest_root,
        "concrete_feature_count": concrete_feature_count,
        "target_feature_result_count": target_feature_result_count,
        "chunk_kind": chunk_kind,
        "chunk_index": chunk_index,
        "row_start": row_start,
        "row_end_exclusive": row_end_exclusive,
        "row_count": len(rows),
        "row_projection_sha256": rows_sha256,
        "rows": rows,
        "projection_scope": _STATISTICAL_RESULTS_PROJECTION_SCOPE,
        "claim_boundary": RESULT_PROJECTION_CLAIM_BOUNDARY,
        "trusted_bootstrap_requirement": TRUSTED_BOOTSTRAP_REQUIREMENT,
    }
    payload = canonicalize_rfc8785_v1(body)
    if not 0 < len(payload) <= MAXIMUM_RESULT_CHUNK_BYTES:
        raise ValueError("canonical statistical-results chunk exceeds two MiB")
    return CanonicalResultChunkV1(
        attempt_context_id=attempt_context_id,
        retention_session_id=retention_session_id,
        mode=mode,
        receipt_id=receipt_id,
        audited_row_count=audited_row_count,
        feature_manifest_root=feature_manifest_root,
        concrete_feature_count=concrete_feature_count,
        target_feature_result_count=target_feature_result_count,
        chunk_kind=chunk_kind,
        chunk_index=chunk_index,
        row_start=row_start,
        row_end_exclusive=row_end_exclusive,
        row_projection_sha256=rows_sha256,
        sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        canonical_bytes=payload,
    )


def _project_statistical_results_chunks_with_dependencies_v1(
    receipt: auditor.FormalCovertAuditReceiptV1,
    *,
    attempt_context_id: str,
    retention_session_id: str,
    mode: str,
    receipt_validator: object,
    target_feature_projection: object,
    target_family_projection: object,
) -> StatisticalResultsProjectionV1:
    if (
        not callable(receipt_validator)
        or not callable(target_feature_projection)
        or not callable(target_family_projection)
    ):
        raise TypeError("statistical projection dependencies are not callable")
    attempt_context_id = _require_identifier(
        attempt_context_id,
        "statistical projection attempt context ID",
    )
    retention_session_id = _require_identifier(
        retention_session_id,
        "statistical projection retention session ID",
    )
    if type(mode) is not str or mode not in MODE_ORDER:
        raise ValueError("statistical projection mode drift")
    if type(receipt) is not auditor.FormalCovertAuditReceiptV1:
        raise TypeError("statistical projection requires the exact receipt DTO")
    validated_receipt_id = receipt_validator(receipt)
    if validated_receipt_id != receipt.receipt_id:
        raise ValueError("statistical projection live receipt ID drift")
    _require_receipt_id_v1(receipt.receipt_id, "statistical projection receipt ID")
    expected_disposition = (
        auditor.FormalCovertDisposition.PASS
        if mode == "KNOWN_NULL"
        else auditor.FormalCovertDisposition.FAIL
    )
    if receipt.disposition is not expected_disposition:
        raise ValueError("statistical projection mode/disposition direction drift")
    if receipt.audited_row_count != auditor.MAXIMUM_ROWS:
        raise ValueError("statistical projection is not the exact 1,024-row run")
    manifest = receipt.feature_manifest
    if type(manifest) is not auditor.FeatureManifestV1:
        raise TypeError("statistical projection lacks the exact feature manifest")
    observed_manifest_root = manifest.manifest_root
    manifest.__post_init__()
    if manifest.manifest_root != observed_manifest_root:
        raise ValueError("statistical projection feature-manifest live root drift")
    concrete_feature_count = manifest.deduplicated_feature_count
    expected_order = tuple(
        (target, feature_id)
        for target in auditor.TARGETS
        for feature_id in manifest.concrete_feature_ids
    )
    if (
        not receipt.results
        or len(receipt.results) != len(expected_order)
        or tuple((item.target, item.feature_id) for item in receipt.results)
        != expected_order
    ):
        raise ValueError("statistical target-feature coverage/order drift")
    if tuple(item.target for item in receipt.target_family_results) != auditor.TARGETS:
        raise ValueError("statistical target-family coverage/order drift")
    total_results = len(receipt.results)
    feature_manifest_chunks = _canonical_feature_manifest_chunks_v1(
        manifest=manifest,
        attempt_context_id=attempt_context_id,
        retention_session_id=retention_session_id,
        mode=mode,
        receipt_id=receipt.receipt_id,
    )
    chunks: list[CanonicalResultChunkV1] = []
    for chunk_index, row_start in enumerate(
        range(0, total_results, RESULT_ROWS_PER_NONFINAL_CHUNK)
    ):
        row_end = min(
            total_results,
            row_start + RESULT_ROWS_PER_NONFINAL_CHUNK,
        )
        rows = [
            target_feature_projection(item)
            for item in receipt.results[row_start:row_end]
        ]
        if row_end < total_results and len(rows) != RESULT_ROWS_PER_NONFINAL_CHUNK:
            raise RuntimeError(
                "non-final statistical-results chunk is not exactly 1,024 rows"
            )
        chunks.append(
            _canonical_result_chunk_v1(
                attempt_context_id=attempt_context_id,
                retention_session_id=retention_session_id,
                mode=mode,
                receipt_id=receipt.receipt_id,
                audited_row_count=receipt.audited_row_count,
                feature_manifest_root=manifest.manifest_root,
                concrete_feature_count=concrete_feature_count,
                target_feature_result_count=total_results,
                chunk_kind="TARGET_FEATURE_RESULTS",
                chunk_index=chunk_index,
                row_start=row_start,
                row_end_exclusive=row_end,
                rows=rows,
            )
        )
    family_rows = [
        target_family_projection(item)
        for item in receipt.target_family_results
    ]
    family_chunk = _canonical_result_chunk_v1(
        attempt_context_id=attempt_context_id,
        retention_session_id=retention_session_id,
        mode=mode,
        receipt_id=receipt.receipt_id,
        audited_row_count=receipt.audited_row_count,
        feature_manifest_root=manifest.manifest_root,
        concrete_feature_count=concrete_feature_count,
        target_feature_result_count=total_results,
        chunk_kind="TARGET_FAMILY_RESULTS",
        chunk_index=0,
        row_start=0,
        row_end_exclusive=len(family_rows),
        rows=family_rows,
    )
    manifest_body: dict[str, object] = {
        "schema_version": STATISTICAL_RESULTS_MANIFEST_SCHEMA_VERSION,
        "auditor_schema_version": auditor.SCHEMA_VERSION,
        "auditor_semantics_version": auditor.SEMANTICS_VERSION,
        "attempt_context_id": attempt_context_id,
        "retention_session_id": retention_session_id,
        "mode": mode,
        "receipt_id": receipt.receipt_id,
        "audited_row_count": receipt.audited_row_count,
        "feature_manifest_root": manifest.manifest_root,
        "concrete_feature_count": concrete_feature_count,
        "target_order": list(auditor.TARGETS),
        "target_feature_result_count": total_results,
        "target_family_result_count": len(family_rows),
        "rows_per_nonfinal_target_feature_chunk": RESULT_ROWS_PER_NONFINAL_CHUNK,
        "maximum_chunk_bytes": MAXIMUM_RESULT_CHUNK_BYTES,
        "feature_manifest_chunk_schema_version": (
            FEATURE_MANIFEST_CHUNK_SCHEMA_VERSION
        ),
        "maximum_feature_manifest_entries_per_chunk": (
            FEATURE_MANIFEST_ENTRIES_PER_NONFINAL_CHUNK
        ),
        "maximum_feature_manifest_chunk_bytes": MAXIMUM_RESULT_CHUNK_BYTES,
        "feature_manifest_chunks": [
            _feature_manifest_chunk_reference_v1(item)
            for item in feature_manifest_chunks
        ],
        "target_feature_chunks": [_chunk_reference_v1(item) for item in chunks],
        "target_family_chunk": _chunk_reference_v1(family_chunk),
        "projection_scope": _STATISTICAL_RESULTS_PROJECTION_SCOPE,
        "claim_boundary": RESULT_PROJECTION_CLAIM_BOUNDARY,
        "trusted_bootstrap_requirement": TRUSTED_BOOTSTRAP_REQUIREMENT,
    }
    manifest_payload = canonicalize_rfc8785_v1(manifest_body)
    if not 0 < len(manifest_payload) <= MAXIMUM_RESULT_CHUNK_BYTES:
        raise ValueError("canonical statistical-results manifest exceeds two MiB")
    projection = StatisticalResultsProjectionV1(
        attempt_context_id=attempt_context_id,
        retention_session_id=retention_session_id,
        mode=mode,
        receipt_id=receipt.receipt_id,
        audited_row_count=receipt.audited_row_count,
        feature_manifest_root=manifest.manifest_root,
        concrete_feature_count=concrete_feature_count,
        target_feature_result_count=total_results,
        feature_manifest_chunks=feature_manifest_chunks,
        target_feature_chunks=tuple(chunks),
        target_family_chunk=family_chunk,
        manifest_sha256=(
            "sha256:" + hashlib.sha256(manifest_payload).hexdigest()
        ),
        manifest_size_bytes=len(manifest_payload),
        manifest_canonical_bytes=manifest_payload,
    )
    projection.__post_init__()
    return projection


def project_statistical_results_chunks_v1(
    receipt: auditor.FormalCovertAuditReceiptV1,
    *,
    attempt_context_id: str,
    retention_session_id: str,
    mode: str,
) -> StatisticalResultsProjectionV1:
    """Project validated results and the exact replayable feature preimage.

    Import-time defaults capture the source pins, receipt validator, and metric
    projectors so synchronized rebinding of the public ``_ORIGINAL_*`` module
    names cannot silently replace them.  Arbitrary same-process code/function-
    default mutation remains outside this module's authority boundary, as made
    explicit by :data:`IN_PROCESS_MUTATION_NONCLAIM`.

    The live receipt content ID and caller-supplied attempt/session/mode context
    are embedded in every chunk and the exact chunk-reference manifest.  This
    is not evidence that original worker bytes were retained; that independent
    fact can only be joined later with a successful ``RetentionSessionV1``
    evidence manifest.
    """

    _require_source_pins_and_metric_projection_identities_v1()
    (
        target_feature_projection,
        target_family_projection,
        receipt_validator,
    ) = _captured_projection_callables_v1()
    return _project_statistical_results_chunks_with_dependencies_v1(
        receipt,
        attempt_context_id=attempt_context_id,
        retention_session_id=retention_session_id,
        mode=mode,
        receipt_validator=receipt_validator,
        target_feature_projection=target_feature_projection,
        target_family_projection=target_family_projection,
    )


__all__ = [
    "AUDITOR_SOURCE_SHA256",
    "CanonicalFeatureManifestChunkV1",
    "CanonicalResultChunkV1",
    "EXPECTED_CLEANUP_CALLS",
    "EXPECTED_OBJECT_ORDER",
    "EXPECTED_RETAINED_OBJECTS",
    "FEATURE_MANIFEST_CHUNK_SCHEMA_VERSION",
    "FEATURE_MANIFEST_ENTRIES_PER_NONFINAL_CHUNK",
    "IN_PROCESS_MUTATION_NONCLAIM",
    "MODE_ORDER",
    "QV2_SOURCE_SHA256",
    "RETENTION_ACK_SCHEMA_VERSION",
    "RETENTION_CLAIM_BOUNDARY",
    "RETENTION_REQUEST_SCHEMA_VERSION",
    "RESULT_PROJECTION_CLAIM_BOUNDARY",
    "RESULT_ROWS_PER_NONFINAL_CHUNK",
    "RetainedWorkerObjectV1",
    "RetentionIngestAckV1",
    "RetentionSessionV1",
    "RetentionSink",
    "SCHEMA_VERSION",
    "STATISTICAL_RESULTS_CHUNK_SCHEMA_VERSION",
    "STATISTICAL_RESULTS_MANIFEST_SCHEMA_VERSION",
    "StatisticalResultsProjectionV1",
    "TRUSTED_BOOTSTRAP_REQUIREMENT",
    "project_statistical_results_chunks_v1",
]
