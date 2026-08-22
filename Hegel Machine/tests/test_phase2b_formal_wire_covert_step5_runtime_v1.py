from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Callable

import pytest

from hegel_machine import phase2b_formal_covert_auditor_v1 as auditor
from hegel_machine import phase2b_formal_wire_covert_step5_runtime_v1 as runtime


ATTEMPT_ID = "attempt-context-step5-0001"
SESSION_ID = "retention-session-step5-0001"
OTHER_ATTEMPT_ID = "attempt-context-step5-0002"
OTHER_SESSION_ID = "retention-session-step5-0002"
RECEIPT_ID = "phase2b_formal_covert_audit_transcript_" + "1" * 64
OTHER_RECEIPT_ID = "phase2b_formal_covert_audit_transcript_" + "2" * 64


class _StringSubclass(str):
    pass


class _EqualToEverything:
    def __init__(self, hash_as: str) -> None:
        self._hash_as = hash_as

    def __eq__(self, other: object) -> bool:
        del other
        return True

    def __ne__(self, other: object) -> bool:
        del other
        return False

    def __hash__(self) -> int:
        return hash(self._hash_as)


@pytest.fixture(autouse=True)
def _isolate_pytest_plugin_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the KAT focused when installed pytest plugins own helper threads."""

    monkeypatch.setattr(
        runtime,
        "_require_single_native_thread_v1",
        lambda: threading.get_native_id(),
    )


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _unsafe_ack(**values: object) -> runtime.RetentionIngestAckV1:
    ack = object.__new__(runtime.RetentionIngestAckV1)
    defaults: dict[str, object] = {
        "schema_version": runtime.RETENTION_ACK_SCHEMA_VERSION,
        "attempt_context_id": ATTEMPT_ID,
        "retention_session_id": SESSION_ID,
        "request_id": "sha256:" + "0" * 64,
        "ordinal": 0,
        "mode": "KNOWN_NULL",
        "phase": "INVARIANCE",
        "source_basename": runtime.INVARIANT_BASENAME,
        "content_address": "sha256:" + "0" * 64,
        "sha256": "sha256:" + "0" * 64,
        "size_bytes": 1,
        "durable_before_return": True,
        "readback_verified": True,
        "store_receipt_ref": "store-receipt-step5-0001",
    }
    defaults.update(values)
    for name, value in defaults.items():
        object.__setattr__(ack, name, value)
    return ack


class GoodSink:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def _ack(
        self,
        metadata: dict[str, object],
        sha256: str,
        size_bytes: int,
    ) -> runtime.RetentionIngestAckV1:
        return runtime.RetentionIngestAckV1(
            schema_version=runtime.RETENTION_ACK_SCHEMA_VERSION,
            attempt_context_id=str(metadata["attempt_context_id"]),
            retention_session_id=str(metadata["retention_session_id"]),
            request_id=str(metadata["request_id"]),
            ordinal=int(metadata["ordinal"]),
            mode=str(metadata["mode"]),
            phase=str(metadata["phase"]),
            source_basename=str(metadata["source_basename"]),
            content_address=sha256,
            sha256=sha256,
            size_bytes=size_bytes,
            durable_before_return=True,
            readback_verified=True,
            store_receipt_ref=f"store-receipt-step5-{int(metadata['ordinal']):04d}",
        )

    def ingest_open_fd(
        self,
        metadata: dict[str, object],
        fd: int,
        sha256: str,
        size_bytes: int,
    ) -> runtime.RetentionIngestAckV1:
        payload = os.pread(fd, size_bytes, 0)
        assert len(payload) == size_bytes
        assert _sha256(payload) == sha256
        self.requests.append(dict(metadata))
        return self._ack(metadata, sha256, size_bytes)


class NoReadFalseSink(GoodSink):
    def ingest_open_fd(
        self,
        metadata: dict[str, object],
        fd: int,
        sha256: str,
        size_bytes: int,
    ) -> runtime.RetentionIngestAckV1:
        del fd
        return _unsafe_ack(
            attempt_context_id=metadata["attempt_context_id"],
            retention_session_id=metadata["retention_session_id"],
            request_id=metadata["request_id"],
            ordinal=metadata["ordinal"],
            mode=metadata["mode"],
            phase=metadata["phase"],
            source_basename=metadata["source_basename"],
            content_address=sha256,
            sha256=sha256,
            size_bytes=size_bytes,
            durable_before_return=False,
            readback_verified=False,
        )


class MutatingMetadataSink(GoodSink):
    def ingest_open_fd(
        self,
        metadata: dict[str, object],
        fd: int,
        sha256: str,
        size_bytes: int,
    ) -> runtime.RetentionIngestAckV1:
        del fd
        ack = self._ack(metadata, sha256, size_bytes)
        metadata["mode"] = "KNOWN_BINDING_LEAK"
        return ack


class WrongAckSink(GoodSink):
    def __init__(self, field: str, value: object) -> None:
        super().__init__()
        self.field = field
        self.value = value

    def ingest_open_fd(
        self,
        metadata: dict[str, object],
        fd: int,
        sha256: str,
        size_bytes: int,
    ) -> runtime.RetentionIngestAckV1:
        payload = os.pread(fd, size_bytes, 0)
        assert _sha256(payload) == sha256
        ack = self._ack(metadata, sha256, size_bytes)
        return replace(ack, **{self.field: self.value})


class ReentrantSink(GoodSink):
    session: runtime.RetentionSessionV1

    def ingest_open_fd(
        self,
        metadata: dict[str, object],
        fd: int,
        sha256: str,
        size_bytes: int,
    ) -> runtime.RetentionIngestAckV1:
        del metadata, fd, sha256, size_bytes
        self.session.begin_mode("KNOWN_NULL")
        raise AssertionError("reentrant begin_mode unexpectedly returned")


class GlobalCleanupMutationSink(GoodSink):
    def ingest_open_fd(
        self,
        metadata: dict[str, object],
        fd: int,
        sha256: str,
        size_bytes: int,
    ) -> runtime.RetentionIngestAckV1:
        ack = super().ingest_open_fd(metadata, fd, sha256, size_bytes)
        auditor._cleanup_invariant_artifact_directory = lambda *args: None
        return ack


class SameInodeRewriteSink(GoodSink):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def ingest_open_fd(
        self,
        metadata: dict[str, object],
        fd: int,
        sha256: str,
        size_bytes: int,
    ) -> runtime.RetentionIngestAckV1:
        ack = super().ingest_open_fd(metadata, fd, sha256, size_bytes)
        with self.path.open("r+b") as stream:
            stream.write(b"Z" * size_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        return ack


class FailOnCallSink(GoodSink):
    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call
        self.calls = 0

    def ingest_open_fd(
        self,
        metadata: dict[str, object],
        fd: int,
        sha256: str,
        size_bytes: int,
    ) -> runtime.RetentionIngestAckV1:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("injected partial sink failure")
        return super().ingest_open_fd(metadata, fd, sha256, size_bytes)


def _artifact_directory(
    tmp_path: Path,
    phase: str,
    *,
    extra_file: bool = False,
) -> tuple[Path, int, os.stat_result]:
    prefix = (
        "hegel-formal-covert-invariant-"
        if phase == "INVARIANCE"
        else "hegel-formal-covert-stats-"
    )
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=tmp_path))
    names = (
        (runtime.INVARIANT_BASENAME,)
        if phase == "INVARIANCE"
        else runtime.STATISTICS_BASENAMES
    )
    for index, name in enumerate(names):
        (path / name).write_bytes(f"worker:{phase}:{index}".encode())
    if extra_file:
        (path / "unexpected.json").write_bytes(b"unexpected")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    return path, fd, os.fstat(fd)


def _invoke_cleanup(
    phase: str,
    path: Path,
    fd: int,
    identity: os.stat_result,
) -> None:
    function = (
        auditor._cleanup_invariant_artifact_directory
        if phase == "INVARIANCE"
        else auditor._cleanup_statistics_artifact_directory
    )
    function(str(path), fd, identity.st_dev, identity.st_ino)


def _fake_crosscheck_receipt(
    records: tuple[runtime.RetainedWorkerObjectV1, ...],
) -> auditor.FormalCovertAuditReceiptV1:
    invariant_record = records[0]
    statistic_records = records[1:]
    invariant = object.__new__(auditor.InvariantRuntimeEvidenceV1)
    object.__setattr__(
        invariant,
        "worker_result_artifact_sha256",
        invariant_record.sha256,
    )
    object.__setattr__(
        invariant,
        "worker_result_artifact_bytes",
        invariant_record.size_bytes,
    )
    statistics = object.__new__(auditor.StatisticalRuntimeEvidenceV1)
    object.__setattr__(
        statistics,
        "worker_result_artifact_sha256",
        tuple(item.sha256 for item in statistic_records),
    )
    object.__setattr__(
        statistics,
        "worker_result_artifact_bytes",
        tuple(item.size_bytes for item in statistic_records),
    )
    receipt = object.__new__(auditor.FormalCovertAuditReceiptV1)
    object.__setattr__(receipt, "invariant_runtime_evidence", invariant)
    object.__setattr__(receipt, "statistics_runtime_evidence", statistics)
    return receipt


def _run_one_cleanup_failure(
    tmp_path: Path,
    sink: runtime.RetentionSink,
    *,
    phase: str = "INVARIANCE",
) -> tuple[runtime.RetentionSessionV1, Path]:
    session = runtime.RetentionSessionV1(
        attempt_context_id=ATTEMPT_ID,
        retention_session_id=SESSION_ID,
        sink=sink,
    )
    path, fd, identity = _artifact_directory(tmp_path, phase)
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            with session:
                session.begin_mode("KNOWN_NULL")
                _invoke_cleanup(phase, path, fd, identity)
    finally:
        os.close(fd)
    return session, path


def test_source_pins_claim_split_and_request_v1() -> None:
    runtime._require_source_pins_and_callable_identities_v1()
    assert runtime.RETENTION_REQUEST_SCHEMA_VERSION.endswith("/1")
    assert runtime.FEATURE_MANIFEST_CHUNK_SCHEMA_VERSION.endswith("/1")
    assert runtime.STATISTICAL_RESULTS_MANIFEST_SCHEMA_VERSION.endswith("/2")
    assert runtime.RETENTION_CLAIM_BOUNDARY != runtime.RESULT_PROJECTION_CLAIM_BOUNDARY
    assert "ORIGINAL_TRANSIENT_WORKER_BYTES" in runtime.RETENTION_CLAIM_BOUNDARY
    assert "NOT_WORKER_RETENTION" in runtime.RESULT_PROJECTION_CLAIM_BOUNDARY
    assert "NOT_RECEIPT_CORE" in runtime.RESULT_PROJECTION_CLAIM_BOUNDARY
    assert "AUTHORITY" in runtime.RESULT_PROJECTION_CLAIM_BOUNDARY
    assert "PREIMPORT_MONKEYPATCHING" in runtime.TRUSTED_BOOTSTRAP_REQUIREMENT
    assert "SAME_PROCESS_CODE" in runtime.IN_PROCESS_MUTATION_NONCLAIM


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "schema_version",
            _StringSubclass(runtime.RETENTION_ACK_SCHEMA_VERSION),
        ),
        ("mode", _StringSubclass("KNOWN_NULL")),
        ("phase", _StringSubclass("INVARIANCE")),
        ("content_address", _StringSubclass("sha256:" + "0" * 64)),
        (
            "schema_version",
            _EqualToEverything(runtime.RETENTION_ACK_SCHEMA_VERSION),
        ),
        ("mode", _EqualToEverything("KNOWN_NULL")),
        ("phase", _EqualToEverything("INVARIANCE")),
        ("content_address", _EqualToEverything("sha256:" + "0" * 64)),
    ),
)
def test_ack_schema_enum_and_address_require_exact_strings(
    field: str,
    value: object,
) -> None:
    valid = _unsafe_ack()
    valid.__post_init__()
    with pytest.raises((TypeError, ValueError)):
        replace(valid, **{field: value})


def test_retained_worker_rejects_duck_typed_acknowledgement() -> None:
    valid_ack = _unsafe_ack()
    valid_ack.__post_init__()

    class DuckAck:
        def __post_init__(self) -> None:
            raise AssertionError("duck acknowledgement was consumed")

    duck = DuckAck()
    for name in (
        "ordinal",
        "attempt_context_id",
        "retention_session_id",
        "request_id",
        "mode",
        "phase",
        "source_basename",
        "sha256",
        "size_bytes",
    ):
        setattr(duck, name, getattr(valid_ack, name))
    with pytest.raises(TypeError, match="exact DTO"):
        runtime.RetainedWorkerObjectV1(
            attempt_context_id=ATTEMPT_ID,
            retention_session_id=SESSION_ID,
            mode="KNOWN_NULL",
            phase="INVARIANCE",
            target_index=None,
            source_basename=runtime.INVARIANT_BASENAME,
            ordinal=0,
            request_id=valid_ack.request_id,
            sha256=valid_ack.sha256,
            size_bytes=valid_ack.size_bytes,
            acknowledgement=duck,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("mode", _StringSubclass("KNOWN_NULL")),
        ("phase", _StringSubclass("INVARIANCE")),
        ("mode", _EqualToEverything("KNOWN_NULL")),
        ("phase", _EqualToEverything("INVARIANCE")),
    ),
)
def test_retained_worker_mode_and_phase_require_exact_strings(
    field: str,
    value: object,
) -> None:
    valid_ack = _unsafe_ack()
    valid_ack.__post_init__()
    arguments: dict[str, object] = {
        "attempt_context_id": ATTEMPT_ID,
        "retention_session_id": SESSION_ID,
        "mode": "KNOWN_NULL",
        "phase": "INVARIANCE",
        "target_index": None,
        "source_basename": runtime.INVARIANT_BASENAME,
        "ordinal": 0,
        "request_id": valid_ack.request_id,
        "sha256": valid_ack.sha256,
        "size_bytes": valid_ack.size_bytes,
        "acknowledgement": valid_ack,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match="mode/phase drift"):
        runtime.RetainedWorkerObjectV1(**arguments)  # type: ignore[arg-type]


def test_retention_exact_twelve_objects_four_cleanups_and_request_ids(
    tmp_path: Path,
) -> None:
    sink = GoodSink()
    session = runtime.RetentionSessionV1(
        attempt_context_id=ATTEMPT_ID,
        retention_session_id=SESSION_ID,
        sink=sink,
    )
    with session:
        for mode in runtime.MODE_ORDER:
            session.begin_mode(mode)
            start = len(session.retained_objects)
            for phase in ("INVARIANCE", "STATISTICS"):
                path, fd, identity = _artifact_directory(tmp_path, phase)
                try:
                    _invoke_cleanup(phase, path, fd, identity)
                    assert not path.exists()
                    assert tuple(os.listdir(fd)) == ()
                    assert os.fstat(fd).st_nlink == 0
                finally:
                    os.close(fd)
            records = session.retained_objects[start:]
            session.crosscheck_and_complete_mode(_fake_crosscheck_receipt(records))
        retained = session.assert_complete()
    assert len(retained) == runtime.EXPECTED_RETAINED_OBJECTS == 12
    assert session.completed_cleanup_calls == runtime.EXPECTED_CLEANUP_CALLS == 4
    assert session.sink_call_count == session.acknowledged_object_count == 12
    assert session.failed_closed_reason is None
    assert tuple((item.mode, item.phase, item.target_index) for item in retained) == (
        runtime.EXPECTED_OBJECT_ORDER
    )
    assert len(sink.requests) == 12
    expected_keys = {
        "schema_version",
        "attempt_context_id",
        "retention_session_id",
        "request_id",
        "ordinal",
        "mode",
        "phase",
        "target_index_or_null",
        "source_basename",
        "auditor_reported_sha256",
        "auditor_reported_size_bytes",
        "fd_count",
    }
    for request in sink.requests:
        assert set(request) == expected_keys
        request_id = request.pop("request_id")
        assert request_id == _sha256(runtime.canonicalize_rfc8785_v1(request))
        request["request_id"] = request_id


@pytest.mark.parametrize(
    ("sink", "match"),
    (
        (NoReadFalseSink(), "durable readback"),
        (MutatingMetadataSink(), "mutated"),
        (WrongAckSink("attempt_context_id", OTHER_ATTEMPT_ID), "linkage"),
        (WrongAckSink("retention_session_id", OTHER_SESSION_ID), "linkage"),
        (WrongAckSink("mode", "KNOWN_BINDING_LEAK"), "linkage"),
        (WrongAckSink("request_id", "sha256:" + "f" * 64), "linkage"),
        (WrongAckSink("ordinal", 1), "linkage"),
        (WrongAckSink("phase", "STATISTICS"), "linkage"),
        (WrongAckSink("source_basename", "target-0.json"), "linkage"),
        (WrongAckSink("size_bytes", 2), "linkage"),
    ),
)
def test_sink_false_mutation_and_cross_context_ack_fail_closed(
    tmp_path: Path,
    sink: runtime.RetentionSink,
    match: str,
) -> None:
    session = runtime.RetentionSessionV1(
        attempt_context_id=ATTEMPT_ID,
        retention_session_id=SESSION_ID,
        sink=sink,
    )
    path, fd, identity = _artifact_directory(tmp_path, "INVARIANCE")
    try:
        with pytest.raises((TypeError, ValueError, RuntimeError), match=match):
            with session:
                session.begin_mode("KNOWN_NULL")
                _invoke_cleanup("INVARIANCE", path, fd, identity)
    finally:
        os.close(fd)
    assert path.exists()
    assert session.failed_closed_reason is not None
    assert session.completed_cleanup_calls == 0


def test_sink_reentrancy_fails_before_cleanup(tmp_path: Path) -> None:
    sink = ReentrantSink()
    session = runtime.RetentionSessionV1(
        attempt_context_id=ATTEMPT_ID,
        retention_session_id=SESSION_ID,
        sink=sink,
    )
    sink.session = session
    path, fd, identity = _artifact_directory(tmp_path, "INVARIANCE")
    try:
        with pytest.raises(RuntimeError, match="ownership/global/fail-closed"):
            with session:
                session.begin_mode("KNOWN_NULL")
                _invoke_cleanup("INVARIANCE", path, fd, identity)
    finally:
        os.close(fd)
    assert path.exists()
    assert session.failed_closed_reason is not None


def test_begin_mode_rejects_string_subclass_before_audit() -> None:
    session = runtime.RetentionSessionV1(
        attempt_context_id=ATTEMPT_ID,
        retention_session_id=SESSION_ID,
        sink=GoodSink(),
    )
    with pytest.raises(RuntimeError, match="mode order drift"):
        with session:
            session.begin_mode(_StringSubclass("KNOWN_NULL"))


def test_sink_global_cleanup_mutation_is_detected_and_restored(tmp_path: Path) -> None:
    original = runtime._ORIGINAL_INVARIANT_CLEANUP
    sink = GlobalCleanupMutationSink()
    session = runtime.RetentionSessionV1(
        attempt_context_id=ATTEMPT_ID,
        retention_session_id=SESSION_ID,
        sink=sink,
    )
    path, fd, identity = _artifact_directory(tmp_path, "INVARIANCE")
    try:
        with pytest.raises(RuntimeError, match="ownership/global/fail-closed"):
            with session:
                session.begin_mode("KNOWN_NULL")
                _invoke_cleanup("INVARIANCE", path, fd, identity)
    finally:
        os.close(fd)
    assert auditor._cleanup_invariant_artifact_directory is original
    assert path.exists()


def test_symlink_hardlink_and_extra_file_are_rejected(tmp_path: Path) -> None:
    for mutation in ("SYMLINK", "HARDLINK", "EXTRA"):
        sink = GoodSink()
        session = runtime.RetentionSessionV1(
            attempt_context_id=ATTEMPT_ID,
            retention_session_id=SESSION_ID,
            sink=sink,
        )
        path, fd, identity = _artifact_directory(
            tmp_path,
            "INVARIANCE",
            extra_file=mutation == "EXTRA",
        )
        artifact = path / runtime.INVARIANT_BASENAME
        if mutation == "SYMLINK":
            target = tmp_path / f"symlink-target-{path.name}.bin"
            target.write_bytes(b"outside")
            artifact.unlink()
            artifact.symlink_to(target)
        elif mutation == "HARDLINK":
            outside = tmp_path / f"hardlink-{path.name}.bin"
            os.link(artifact, outside)
        try:
            with pytest.raises(ValueError):
                with session:
                    session.begin_mode("KNOWN_NULL")
                    _invoke_cleanup("INVARIANCE", path, fd, identity)
        finally:
            os.close(fd)
        assert session.completed_cleanup_calls == 0
        assert session.failed_closed_reason is not None


def test_same_inode_rewrite_after_ack_is_rejected(tmp_path: Path) -> None:
    path, fd, identity = _artifact_directory(tmp_path, "INVARIANCE")
    sink = SameInodeRewriteSink(path / runtime.INVARIANT_BASENAME)
    session = runtime.RetentionSessionV1(
        attempt_context_id=ATTEMPT_ID,
        retention_session_id=SESSION_ID,
        sink=sink,
    )
    try:
        with pytest.raises(ValueError, match="changed during sink ingestion"):
            with session:
                session.begin_mode("KNOWN_NULL")
                _invoke_cleanup("INVARIANCE", path, fd, identity)
    finally:
        os.close(fd)
    assert path.exists()
    assert session.acknowledged_object_count == 1
    assert session.failed_closed_reason is not None


def _forged_cleanup(
    action: Callable[[str, int, int, int], None],
    expected_name: str,
) -> Callable[[str, int, int, int], None]:
    namespace: dict[str, object] = {
        "__name__": auditor.__name__,
        "_ACTION": action,
    }
    source = (
        "from __future__ import annotations\n"
        f"def {expected_name}(artifact_directory: str, artifact_directory_fd: int, "
        "artifact_directory_device: int, artifact_directory_inode: int) -> None:\n"
        "    _ACTION(artifact_directory, artifact_directory_fd, "
        "artifact_directory_device, artifact_directory_inode)\n"
    )
    exec(compile(source, auditor.__file__, "exec"), namespace)
    value = namespace[expected_name]
    assert callable(value)
    return value  # type: ignore[return-value]


def test_import_time_cleanup_and_source_sha_captures_reject_synchronized_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_invariant, _ = runtime._captured_cleanup_callables_v1()
    fake = _forged_cleanup(
        lambda *args: None,
        "_cleanup_invariant_artifact_directory",
    )
    monkeypatch.setattr(runtime, "AUDITOR_SOURCE_SHA256", "0" * 64)
    monkeypatch.setattr(runtime, "QV2_SOURCE_SHA256", "1" * 64)
    monkeypatch.setattr(runtime, "_ORIGINAL_INVARIANT_CLEANUP", fake)
    monkeypatch.setattr(auditor, "_cleanup_invariant_artifact_directory", fake)
    assert runtime._captured_cleanup_callables_v1()[0] is captured_invariant
    with pytest.raises(RuntimeError, match="cleanup callable identity drift"):
        runtime._require_source_pins_and_callable_identities_v1()


@pytest.mark.parametrize(
    ("source_module", "basename", "live_pin_name", "match"),
    (
        (
            auditor,
            "phase2b_formal_covert_auditor_v1.py",
            "AUDITOR_SOURCE_SHA256",
            "auditor source SHA-256 drift",
        ),
        (
            runtime.qv2,
            "phase2b_formal_wire_covert_qualification_v2.py",
            "QV2_SOURCE_SHA256",
            "QV2 source SHA-256 drift",
        ),
    ),
)
def test_import_time_source_sha_capture_rejects_forged_path_and_live_pin_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_module: object,
    basename: str,
    live_pin_name: str,
    match: str,
) -> None:
    forged_source = tmp_path / basename
    forged_payload = b"# synchronized forged source and live pin\n"
    forged_source.write_bytes(forged_payload)
    monkeypatch.setattr(source_module, "__file__", str(forged_source))
    monkeypatch.setattr(
        runtime,
        live_pin_name,
        hashlib.sha256(forged_payload).hexdigest(),
    )
    with pytest.raises(RuntimeError, match=match):
        runtime._require_source_pins_and_metric_projection_identities_v1()


def test_import_time_projection_and_validator_captures_reject_synchronized_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_feature_projection(value: object) -> dict[str, object]:
        del value
        return {}

    def fake_family_projection(value: object) -> dict[str, object]:
        del value
        return {}

    def fake_validator(value: object) -> str:
        del value
        return RECEIPT_ID

    monkeypatch.setattr(
        runtime,
        "_ORIGINAL_TARGET_FEATURE_PROJECTION",
        fake_feature_projection,
    )
    monkeypatch.setattr(
        runtime,
        "_ORIGINAL_TARGET_FAMILY_PROJECTION",
        fake_family_projection,
    )
    monkeypatch.setattr(runtime, "_ORIGINAL_RECEIPT_VALIDATOR", fake_validator)
    monkeypatch.setattr(
        auditor,
        "_target_feature_metric_projection",
        fake_feature_projection,
    )
    monkeypatch.setattr(
        auditor,
        "_target_family_metric_projection",
        fake_family_projection,
    )
    monkeypatch.setattr(
        auditor,
        "validate_formal_covert_audit_receipt_v1",
        fake_validator,
    )
    captured = runtime._captured_projection_callables_v1()
    assert fake_feature_projection not in captured
    assert fake_family_projection not in captured
    assert fake_validator not in captured
    with pytest.raises(RuntimeError, match="projection/receipt validator"):
        runtime.project_statistical_results_chunks_v1(
            _fake_statistical_receipt(),
            attempt_context_id=ATTEMPT_ID,
            retention_session_id=SESSION_ID,
            mode="KNOWN_NULL",
        )


def test_rename_away_residual_directory_fails_held_dirfd_unlink_proof() -> None:
    with tempfile.TemporaryDirectory(
        prefix="step5-native-dirfd-kat-",
        dir="/tmp",
    ) as native_tmp:
        native_tmp_path = Path(native_tmp)
        path, fd, identity = _artifact_directory(
            native_tmp_path,
            "INVARIANCE",
        )
        residual_path = native_tmp_path / "renamed-away-residual-directory"

        def action(
            artifact_directory: str,
            artifact_directory_fd: int,
            artifact_directory_device: int,
            artifact_directory_inode: int,
        ) -> None:
            del artifact_directory_device, artifact_directory_inode
            os.unlink(runtime.INVARIANT_BASENAME, dir_fd=artifact_directory_fd)
            os.rename(artifact_directory, residual_path)

        mutant = _forged_cleanup(action, "_cleanup_invariant_artifact_directory")
        session = runtime.RetentionSessionV1(
            attempt_context_id=ATTEMPT_ID,
            retention_session_id=SESSION_ID,
            sink=GoodSink(),
        )
        try:
            mutant(str(path), fd, identity.st_dev, identity.st_ino)
            assert not path.exists()
            assert residual_path.is_dir()
            assert tuple(os.listdir(fd)) == ()
            assert os.fstat(fd).st_nlink != 0
            with pytest.raises(
                ValueError,
                match="unlinking the retained directory inode",
            ):
                session._require_postcleanup_absence(str(path), fd)
        finally:
            os.close(fd)
            if residual_path.exists():
                residual_path.rmdir()


@pytest.mark.parametrize("behavior", ("RESIDUAL", "RAISE"))
def test_session_local_cleanup_identity_rebinding_is_nonqualifying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
) -> None:
    real = runtime._ORIGINAL_INVARIANT_CLEANUP

    def action(*args: object) -> None:
        del args
        if behavior == "RAISE":
            raise RuntimeError("injected original cleanup failure")

    fake = _forged_cleanup(action, "_cleanup_invariant_artifact_directory")
    sink = GoodSink()
    session = runtime.RetentionSessionV1(
        attempt_context_id=ATTEMPT_ID,
        retention_session_id=SESSION_ID,
        sink=sink,
    )
    path, fd, identity = _artifact_directory(tmp_path, "INVARIANCE")
    try:
        with pytest.raises((ValueError, RuntimeError)):
            with session:
                session.begin_mode("KNOWN_NULL")
                monkeypatch.setattr(runtime, "_ORIGINAL_INVARIANT_CLEANUP", fake)
                session._original_invariant_cleanup = fake
                _invoke_cleanup("INVARIANCE", path, fd, identity)
    finally:
        os.close(fd)
        auditor._cleanup_invariant_artifact_directory = real
    assert path.exists()
    assert session.acknowledged_object_count == 1
    assert session.completed_cleanup_calls == 0
    assert session.failed_closed_reason is not None


def test_partial_sink_failure_never_runs_statistics_cleanup(tmp_path: Path) -> None:
    sink = FailOnCallSink(fail_on_call=3)
    session = runtime.RetentionSessionV1(
        attempt_context_id=ATTEMPT_ID,
        retention_session_id=SESSION_ID,
        sink=sink,
    )
    invariant_path, invariant_fd, invariant_identity = _artifact_directory(
        tmp_path,
        "INVARIANCE",
    )
    stats_path, stats_fd, stats_identity = _artifact_directory(
        tmp_path,
        "STATISTICS",
    )
    try:
        with pytest.raises(RuntimeError, match="injected partial sink failure"):
            with session:
                session.begin_mode("KNOWN_NULL")
                _invoke_cleanup(
                    "INVARIANCE",
                    invariant_path,
                    invariant_fd,
                    invariant_identity,
                )
                _invoke_cleanup("STATISTICS", stats_path, stats_fd, stats_identity)
    finally:
        os.close(invariant_fd)
        os.close(stats_fd)
    assert not invariant_path.exists()
    assert stats_path.exists()
    assert session.sink_call_count == 3
    assert session.acknowledged_object_count == 2
    assert len(session.retained_objects) == 1
    assert session.completed_cleanup_calls == 1
    assert session.failed_closed_reason is not None


def test_cleanup_global_drift_on_exit_is_reported_and_restored() -> None:
    original = runtime._ORIGINAL_INVARIANT_CLEANUP
    session = runtime.RetentionSessionV1(
        attempt_context_id=ATTEMPT_ID,
        retention_session_id=SESSION_ID,
        sink=GoodSink(),
    )
    with pytest.raises(RuntimeError, match="restoration/completion"):
        with session:
            auditor._cleanup_invariant_artifact_directory = lambda *args: None
    assert auditor._cleanup_invariant_artifact_directory is original


def _probability(numerator: int, denominator: int) -> auditor.ExactProbabilityV1:
    return auditor.ExactProbabilityV1(numerator, denominator)


def _fake_statistical_receipt(
    *,
    receipt_id: str = RECEIPT_ID,
    disposition: auditor.FormalCovertDisposition = auditor.FormalCovertDisposition.PASS,
    feature_variant: str = "base",
) -> auditor.FormalCovertAuditReceiptV1:
    semantic_aliases = tuple(
        sorted(
            set(
                auditor.CHANNEL_GROUPS
                + auditor.UNIQUE_ID_FEATURE_FAMILY
                + (f"uuid_leaf/kat-variant/{feature_variant}",)
            )
        )
    )
    feature_id = auditor.stable_hash(
        {"aliases": semantic_aliases},
        prefix="phase2b_formal_covert_feature_",
    )
    manifest = auditor.FeatureManifestV1(
        concrete_feature_ids=(feature_id,),
        alias_coverage=(
            auditor.FeatureAliasCoverageV1(
                concrete_feature_id=feature_id,
                semantic_aliases=semantic_aliases,
            ),
        ),
        channel_groups=auditor.CHANNEL_GROUPS,
        unique_id_feature_family=auditor.UNIQUE_ID_FEATURE_FAMILY,
        uuid_leaf_count=1,
        channel_leaf_count=1,
        raw_semantic_feature_count=len(semantic_aliases),
        deduplicated_feature_count=1,
    )
    results: list[auditor.TargetFeatureResultV1] = []
    families: list[auditor.TargetFamilyMaxTResultV1] = []
    one = _probability(1, 1)
    for rank, target in enumerate(auditor.TARGETS, start=1):
        critical = _probability(1, 100 * (len(auditor.TARGETS) - rank + 1))
        results.append(
            auditor.TargetFeatureResultV1(
                target=target,
                feature_id=feature_id,
                raw_nmi=0.0,
                permutation_mean_nmi=0.0,
                bias_corrected_nmi=0.0,
                balanced_accuracy_advantage=0.0,
                observed_omnibus=0.0,
                pairwise_exceedance_count=auditor.LABEL_PERMUTATIONS,
                pairwise_raw_p=one,
                pairwise_holm_adjusted_p=one,
                feature_max_t_exceedance_count=auditor.LABEL_PERMUTATIONS,
                feature_max_t_p=one,
                target_holm_critical_value=critical,
                target_family_holm_rejected=False,
                material_effect=False,
                feature_localization_rejected=False,
                failure_witness=False,
                passed=True,
            )
        )
        families.append(
            auditor.TargetFamilyMaxTResultV1(
                target=target,
                observed_target_max_omnibus=0.0,
                exceedance_count=auditor.LABEL_PERMUTATIONS,
                raw_p=one,
                holm_adjusted_p=one,
                holm_rank=rank,
                holm_critical_value=critical,
                holm_rejected=False,
            )
        )
    receipt = object.__new__(auditor.FormalCovertAuditReceiptV1)
    object.__setattr__(receipt, "receipt_id", receipt_id)
    object.__setattr__(receipt, "disposition", disposition)
    object.__setattr__(receipt, "audited_row_count", auditor.MAXIMUM_ROWS)
    object.__setattr__(receipt, "feature_manifest", manifest)
    object.__setattr__(receipt, "results", tuple(results))
    object.__setattr__(receipt, "target_family_results", tuple(families))
    return receipt


@pytest.fixture
def projection_factory(
) -> Callable[..., runtime.StatisticalResultsProjectionV1]:
    def validator(receipt: auditor.FormalCovertAuditReceiptV1) -> str:
        return receipt.receipt_id

    (
        target_feature_projection,
        target_family_projection,
        _,
    ) = runtime._captured_projection_callables_v1()

    def factory(
        *,
        attempt_context_id: str = ATTEMPT_ID,
        retention_session_id: str = SESSION_ID,
        mode: str = "KNOWN_NULL",
        receipt_id: str = RECEIPT_ID,
        feature_variant: str = "base",
        receipt_disposition: auditor.FormalCovertDisposition | None = None,
    ) -> runtime.StatisticalResultsProjectionV1:
        disposition = receipt_disposition or (
            auditor.FormalCovertDisposition.PASS
            if mode == "KNOWN_NULL"
            else auditor.FormalCovertDisposition.FAIL
        )
        return runtime._project_statistical_results_chunks_with_dependencies_v1(
            _fake_statistical_receipt(
                receipt_id=receipt_id,
                disposition=disposition,
                feature_variant=feature_variant,
            ),
            attempt_context_id=attempt_context_id,
            retention_session_id=retention_session_id,
            mode=mode,
            receipt_validator=validator,
            target_feature_projection=target_feature_projection,
            target_family_projection=target_family_projection,
        )

    return factory


def _replace_canonical_payload(
    value: runtime.CanonicalResultChunkV1,
    payload: bytes,
) -> runtime.CanonicalResultChunkV1:
    return replace(
        value,
        canonical_bytes=payload,
        size_bytes=len(payload),
        sha256=_sha256(payload),
    )


def _replace_manifest_payload(
    value: runtime.StatisticalResultsProjectionV1,
    payload: bytes,
) -> runtime.StatisticalResultsProjectionV1:
    return replace(
        value,
        manifest_canonical_bytes=payload,
        manifest_size_bytes=len(payload),
        manifest_sha256=_sha256(payload),
    )


def test_statistical_results_projection_positive_is_canonical_and_narrow(
    projection_factory: Callable[..., runtime.StatisticalResultsProjectionV1],
) -> None:
    projection = projection_factory()
    projection.__post_init__()
    assert projection.attempt_context_id == ATTEMPT_ID
    assert projection.retention_session_id == SESSION_ID
    assert projection.mode == "KNOWN_NULL"
    assert projection.receipt_id == RECEIPT_ID
    assert projection.audited_row_count == 1_024
    assert projection.target_feature_result_count == len(auditor.TARGETS)
    assert len(projection.feature_manifest_chunks) == 1
    assert len(projection.target_feature_chunks) == 1
    feature_chunk = projection.feature_manifest_chunks[0]
    feature_body = runtime.decode_and_require_canonical_rfc8785_v1(
        feature_chunk.canonical_bytes
    )
    assert feature_body["feature_manifest_root"] == projection.feature_manifest_root
    assert feature_body["entries"][0]["concrete_feature_id"] == (
        json.loads(projection.target_feature_chunks[0].canonical_bytes)["rows"][0][
            "feature_id"
        ]
    )
    assert feature_body["channel_groups"] == list(auditor.CHANNEL_GROUPS)
    assert feature_body["unique_id_feature_family"] == list(
        auditor.UNIQUE_ID_FEATURE_FAMILY
    )
    assert feature_chunk.size_bytes <= runtime.MAXIMUM_RESULT_CHUNK_BYTES
    chunk = projection.target_feature_chunks[0]
    assert runtime.decode_and_require_canonical_rfc8785_v1(
        chunk.canonical_bytes
    )
    manifest = runtime.decode_and_require_canonical_rfc8785_v1(
        projection.manifest_canonical_bytes
    )
    assert manifest["receipt_id"] == RECEIPT_ID
    assert manifest["mode"] == "KNOWN_NULL"
    assert manifest["schema_version"].endswith("/2")
    assert manifest["feature_manifest_chunks"][0]["sha256"] == (
        feature_chunk.sha256
    )
    assert manifest["projection_scope"].endswith("NOT_FULL_RECEIPT")
    assert "NOT_WORKER_RETENTION" in manifest["claim_boundary"]
    assert "NOT_RECEIPT_CORE" in manifest["claim_boundary"]
    assert runtime.RETENTION_CLAIM_BOUNDARY not in chunk.canonical_bytes.decode()


def test_projection_dtos_reject_string_subclass_enums(
    projection_factory: Callable[..., runtime.StatisticalResultsProjectionV1],
) -> None:
    projection = projection_factory()
    with pytest.raises(ValueError, match="feature-manifest chunk mode drift"):
        replace(
            projection.feature_manifest_chunks[0],
            mode=_StringSubclass("KNOWN_NULL"),
        )
    with pytest.raises(ValueError, match="result chunk mode drift"):
        replace(
            projection.target_feature_chunks[0],
            mode=_StringSubclass("KNOWN_NULL"),
        )
    with pytest.raises(ValueError, match="result chunk kind drift"):
        replace(
            projection.target_feature_chunks[0],
            chunk_kind=_StringSubclass("TARGET_FEATURE_RESULTS"),
        )
    with pytest.raises(ValueError, match="projection mode drift"):
        replace(projection, mode=_StringSubclass("KNOWN_NULL"))
    with pytest.raises(ValueError, match="statistical projection mode drift"):
        projection_factory(mode=_StringSubclass("KNOWN_NULL"))


def test_fully_rehashed_unknown_canonical_mode_and_chunk_kind_reject(
    projection_factory: Callable[..., runtime.StatisticalResultsProjectionV1],
) -> None:
    projection = projection_factory()

    feature_chunk = projection.feature_manifest_chunks[0]
    feature_body = json.loads(feature_chunk.canonical_bytes)
    feature_body["mode"] = "UNKNOWN_TRANSPORT_MODE"
    feature_payload = runtime.canonicalize_rfc8785_v1(feature_body)
    with pytest.raises(ValueError, match="feature-manifest chunk mode drift"):
        replace(
            feature_chunk,
            mode=_EqualToEverything("KNOWN_NULL"),
            canonical_bytes=feature_payload,
            size_bytes=len(feature_payload),
            sha256=_sha256(feature_payload),
        )

    result_chunk = projection.target_feature_chunks[0]
    mode_body = json.loads(result_chunk.canonical_bytes)
    mode_body["mode"] = "UNKNOWN_TRANSPORT_MODE"
    mode_payload = runtime.canonicalize_rfc8785_v1(mode_body)
    with pytest.raises(ValueError, match="result chunk mode drift"):
        replace(
            result_chunk,
            mode=_EqualToEverything("KNOWN_NULL"),
            canonical_bytes=mode_payload,
            size_bytes=len(mode_payload),
            sha256=_sha256(mode_payload),
        )

    kind_body = json.loads(result_chunk.canonical_bytes)
    kind_body["chunk_kind"] = "UNKNOWN_TRANSPORT_KIND"
    kind_payload = runtime.canonicalize_rfc8785_v1(kind_body)
    with pytest.raises(ValueError, match="result chunk kind drift"):
        replace(
            result_chunk,
            chunk_kind=_EqualToEverything("TARGET_FEATURE_RESULTS"),
            canonical_bytes=kind_payload,
            size_bytes=len(kind_payload),
            sha256=_sha256(kind_payload),
        )


def test_feature_manifest_projection_multichunk_replays_exact_dto_and_stays_bounded() -> None:
    coverage: list[auditor.FeatureAliasCoverageV1] = []
    base_aliases = tuple(
        sorted(set(auditor.CHANNEL_GROUPS + auditor.UNIQUE_ID_FEATURE_FAMILY))
    )
    alias_sets = [base_aliases] + [
        (f"uuid_leaf/feature-manifest-multichunk/{index:04d}",)
        for index in range(1, 257)
    ]
    for semantic_aliases in alias_sets:
        feature_id = auditor.stable_hash(
            {"aliases": semantic_aliases},
            prefix="phase2b_formal_covert_feature_",
        )
        coverage.append(
            auditor.FeatureAliasCoverageV1(
                concrete_feature_id=feature_id,
                semantic_aliases=semantic_aliases,
            )
        )
    coverage.sort(key=lambda item: item.concrete_feature_id)
    manifest = auditor.FeatureManifestV1(
        concrete_feature_ids=tuple(item.concrete_feature_id for item in coverage),
        alias_coverage=tuple(coverage),
        channel_groups=auditor.CHANNEL_GROUPS,
        unique_id_feature_family=auditor.UNIQUE_ID_FEATURE_FAMILY,
        uuid_leaf_count=1,
        channel_leaf_count=1,
        raw_semantic_feature_count=sum(
            len(item.semantic_aliases) for item in coverage
        ),
        deduplicated_feature_count=len(coverage),
    )
    chunks = runtime._canonical_feature_manifest_chunks_v1(
        manifest=manifest,
        attempt_context_id=ATTEMPT_ID,
        retention_session_id=SESSION_ID,
        mode="KNOWN_NULL",
        receipt_id=RECEIPT_ID,
    )
    assert len(chunks) == 2
    assert (chunks[0].feature_start, chunks[0].feature_end_exclusive) == (0, 256)
    assert (chunks[1].feature_start, chunks[1].feature_end_exclusive) == (256, 257)
    assert all(0 < item.size_bytes <= runtime.MAXIMUM_RESULT_CHUNK_BYTES for item in chunks)
    replayed = runtime._replay_feature_manifest_chunks_v1(
        chunks=chunks,
        attempt_context_id=ATTEMPT_ID,
        retention_session_id=SESSION_ID,
        mode="KNOWN_NULL",
        receipt_id=RECEIPT_ID,
        feature_manifest_root=manifest.manifest_root,
        concrete_feature_count=manifest.deduplicated_feature_count,
    )
    assert replayed == manifest


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("attempt_context_id", OTHER_ATTEMPT_ID),
        ("retention_session_id", OTHER_SESSION_ID),
        ("mode", "KNOWN_BINDING_LEAK"),
        ("receipt_id", OTHER_RECEIPT_ID),
        ("feature_manifest_root", "feature-root-other-0001"),
        ("chunk_index", 1),
        ("row_start", 1),
        ("row_end_exclusive", 4),
        ("row_projection_sha256", "sha256:" + "e" * 64),
    ),
)
def test_chunk_dto_fields_are_exactly_bound_to_canonical_payload(
    projection_factory: Callable[..., runtime.StatisticalResultsProjectionV1],
    field: str,
    value: object,
) -> None:
    chunk = projection_factory().target_feature_chunks[0]
    with pytest.raises(ValueError):
        replace(chunk, **{field: value})


@pytest.mark.parametrize("mutation", ("UNKNOWN", "BODY_MODE", "BODY_RECEIPT", "BODY_ROWS"))
def test_chunk_canonical_body_mutations_are_rejected(
    projection_factory: Callable[..., runtime.StatisticalResultsProjectionV1],
    mutation: str,
) -> None:
    chunk = projection_factory().target_feature_chunks[0]
    body = json.loads(chunk.canonical_bytes)
    if mutation == "UNKNOWN":
        body["unknown"] = True
    elif mutation == "BODY_MODE":
        body["mode"] = "KNOWN_BINDING_LEAK"
    elif mutation == "BODY_RECEIPT":
        body["receipt_id"] = OTHER_RECEIPT_ID
    else:
        body["rows"][0]["feature_id"] = "feature-splice"
    payload = runtime.canonicalize_rfc8785_v1(body)
    with pytest.raises(ValueError):
        _replace_canonical_payload(chunk, payload)


def test_full_feature_id_row_chunk_and_manifest_rehash_rejects_against_replayed_manifest(
    projection_factory: Callable[..., runtime.StatisticalResultsProjectionV1],
) -> None:
    projection = projection_factory()
    alternate_receipt = _fake_statistical_receipt(feature_variant="mutant-preimage")
    alternate_feature_id = alternate_receipt.feature_manifest.concrete_feature_ids[0]
    assert alternate_feature_id != (
        json.loads(projection.feature_manifest_chunks[0].canonical_bytes)["entries"][0][
            "concrete_feature_id"
        ]
    )

    chunk = projection.target_feature_chunks[0]
    chunk_body = json.loads(chunk.canonical_bytes)
    chunk_body["rows"][0]["feature_id"] = alternate_feature_id
    row_projection_sha256 = _sha256(
        runtime.canonicalize_rfc8785_v1(chunk_body["rows"])
    )
    chunk_body["row_projection_sha256"] = row_projection_sha256
    chunk_payload = runtime.canonicalize_rfc8785_v1(chunk_body)
    mutated_chunk = replace(
        chunk,
        row_projection_sha256=row_projection_sha256,
        canonical_bytes=chunk_payload,
        size_bytes=len(chunk_payload),
        sha256=_sha256(chunk_payload),
    )

    manifest_body = json.loads(projection.manifest_canonical_bytes)
    manifest_body["target_feature_chunks"][0] = runtime._chunk_reference_v1(
        mutated_chunk
    )
    manifest_payload = runtime.canonicalize_rfc8785_v1(manifest_body)
    with pytest.raises(ValueError, match="row/feature-manifest preimage drift"):
        replace(
            projection,
            target_feature_chunks=(mutated_chunk,),
            manifest_canonical_bytes=manifest_payload,
            manifest_size_bytes=len(manifest_payload),
            manifest_sha256=_sha256(manifest_payload),
        )


@pytest.mark.parametrize(
    "payload",
    (
        b"{}",
        b'{"a":1,"a":1}',
        b'{ "a":1}',
        b"x" * (runtime.MAXIMUM_RESULT_CHUNK_BYTES + 1),
    ),
    ids=("empty-object", "duplicate-key", "noncanonical", "oversize"),
)
def test_self_hashed_empty_duplicate_noncanonical_and_oversize_chunks_reject(
    projection_factory: Callable[..., runtime.StatisticalResultsProjectionV1],
    payload: bytes,
) -> None:
    chunk = projection_factory().target_feature_chunks[0]
    with pytest.raises((ValueError, TypeError)):
        _replace_canonical_payload(chunk, payload)


def test_cross_attempt_mode_receipt_and_manifest_splices_are_rejected(
    projection_factory: Callable[..., runtime.StatisticalResultsProjectionV1],
) -> None:
    base = projection_factory()
    variants = (
        projection_factory(attempt_context_id=OTHER_ATTEMPT_ID),
        projection_factory(retention_session_id=OTHER_SESSION_ID),
        projection_factory(mode="KNOWN_BINDING_LEAK"),
        projection_factory(receipt_id=OTHER_RECEIPT_ID),
        projection_factory(feature_variant="other-feature-manifest"),
    )
    for other in variants:
        with pytest.raises(ValueError):
            replace(
                base,
                target_feature_chunks=(other.target_feature_chunks[0],),
            )
        with pytest.raises(ValueError):
            replace(
                base,
                feature_manifest_chunks=other.feature_manifest_chunks,
            )
        with pytest.raises(ValueError):
            _replace_manifest_payload(base, other.manifest_canonical_bytes)


def test_manifest_exact_refs_unknown_duplicate_and_noncanonical_reject(
    projection_factory: Callable[..., runtime.StatisticalResultsProjectionV1],
) -> None:
    projection = projection_factory()
    body = json.loads(projection.manifest_canonical_bytes)
    body["target_feature_chunks"][0]["sha256"] = "sha256:" + "d" * 64
    with pytest.raises(ValueError):
        _replace_manifest_payload(
            projection,
            runtime.canonicalize_rfc8785_v1(body),
        )
    body = json.loads(projection.manifest_canonical_bytes)
    body["unknown"] = True
    with pytest.raises(ValueError):
        _replace_manifest_payload(
            projection,
            runtime.canonicalize_rfc8785_v1(body),
        )
    body = json.loads(projection.manifest_canonical_bytes)
    body["target_feature_chunks"].append(body["target_feature_chunks"][0])
    with pytest.raises(ValueError):
        _replace_manifest_payload(
            projection,
            runtime.canonicalize_rfc8785_v1(body),
        )
    for payload in (b'{"a":1,"a":1}', b'{ "a":1}', b"{}"):
        with pytest.raises((ValueError, TypeError)):
            _replace_manifest_payload(projection, payload)
    with pytest.raises((ValueError, TypeError)):
        _replace_manifest_payload(
            projection,
            b"x" * (runtime.MAXIMUM_RESULT_CHUNK_BYTES + 1),
        )


def test_projector_rejects_cross_mode_receipt_direction(
    projection_factory: Callable[..., runtime.StatisticalResultsProjectionV1],
) -> None:
    with pytest.raises(ValueError, match="mode/disposition"):
        projection_factory(
            mode="KNOWN_BINDING_LEAK",
            receipt_disposition=auditor.FormalCovertDisposition.PASS,
        )
