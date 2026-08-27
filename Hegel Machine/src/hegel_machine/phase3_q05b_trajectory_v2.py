"""Durable, fail-closed trajectory records for Q0.5b shadow execution.

The writer in this module only records an execution trajectory.  It grants no
execution, admission, production, Meta, adoption, or retry authority.  A
trajectory is an append-only sequence of canonical single-LF JSON event files
followed by one terminal manifest.  Each event is domain-hashed and binds the
previous event digest.  Publication uses an owner-private directory, exclusive
no-follow staging creation, an atomic no-replace hard-link commit, immutable-
by-mode records, and file/directory fsync.  A crash before commit leaves one
strictly recognizable hidden torn tail and never invalidates the durable prefix.

V2 freezes the held-actor sampling policy at a 125 ms target wait and a 250 ms
maximum gap.  Sample ordinals start at one and are contiguous independently
for each role; every later sample binds the preceding finish timestamp.  All
monotonic timestamps and durations are exact JSON integers bounded as uint64,
while sequence/count fields use their own smaller counter bound.

The verifier is read-only.  A valid namespace is a complete terminal trace, an
explicit incomplete prefix, or ``INCOMPLETE_TORN_TAIL`` with the committed
prefix summarized separately.  It never repairs or deletes evidence.
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
import threading
import time
from types import MappingProxyType
from typing import Callable, Final, Mapping, NoReturn


__all__ = (
    "DurableTrajectoryWriterV2",
    "TrajectoryErrorV2",
    "TrajectoryVerificationV2",
    "EVENT_SCHEMA_V2",
    "TERMINAL_MANIFEST_SCHEMA_V2",
    "TERMINAL_MANIFEST_BASENAME_V2",
    "TARGET_WAIT_NS_V2",
    "MAXIMUM_GAP_NS_V2",
    "GAP_EXCEEDED_ERROR_CODE_V2",
    "SAMPLE_ROLE_ACTOR_MAP_V2",
    "VerifiedFailureRowV2",
    "VerifiedOpenSpanContextV2",
    "canonical_trajectory_json_bytes_v2",
    "create_durable_trajectory_writer_v2",
    "verify_durable_trajectory_v2",
)


EVENT_SCHEMA_V2: Final = "hegel-phase3-q05b-durable-trajectory-event/2"
TERMINAL_MANIFEST_SCHEMA_V2: Final = (
    "hegel-phase3-q05b-durable-trajectory-terminal-manifest/2"
)
TERMINAL_MANIFEST_BASENAME_V2: Final = "terminal-manifest-v2.json"
TARGET_WAIT_NS_V2: Final = 125_000_000
MAXIMUM_GAP_NS_V2: Final = 250_000_000
GAP_EXCEEDED_ERROR_CODE_V2: Final = "HELD_ACTOR_GAP_EXCEEDED"
SAMPLE_ROLE_ACTOR_MAP_V2: Final = MappingProxyType(
    {
        "1": "PYTHON_ENDPOINT",
        "2": "RUST_ENDPOINT",
        "3": "TRUSTED_HOST_REPLAY",
    }
)

FAIL_INPUT_V2: Final = "FAIL_Q05B_TRAJECTORY_V2_INPUT"
FAIL_STATE_V2: Final = "FAIL_Q05B_TRAJECTORY_V2_STATE"
FAIL_FILESYSTEM_V2: Final = "FAIL_Q05B_TRAJECTORY_V2_FILESYSTEM"
FAIL_CANONICAL_V2: Final = "FAIL_Q05B_TRAJECTORY_V2_CANONICAL"
FAIL_SCHEMA_V2: Final = "FAIL_Q05B_TRAJECTORY_V2_SCHEMA"
FAIL_CHAIN_V2: Final = "FAIL_Q05B_TRAJECTORY_V2_CHAIN"

_EVENT_DOMAIN: Final = b"HEGEL/PHASE3/Q05B/TRAJECTORY/EVENT/V2\0"
_MANIFEST_DOMAIN: Final = b"HEGEL/PHASE3/Q05B/TRAJECTORY/MANIFEST/V2\0"
_EVENT_BASENAME = re.compile(r"event-([0-9]{20})-v2\.json\Z")
_STAGING_EVENT_BASENAME = re.compile(
    r"\.event-([0-9]{20})-v2\.json\.staging-v2\Z"
)
_STAGING_MANIFEST_BASENAME: Final = (
    f".{TERMINAL_MANIFEST_BASENAME_V2}.staging-v2"
)
_SAFE_ROOT_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+-]{0,255}\Z")
_ERROR_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_TAGGED_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_COUNTER: Final = (1 << 53) - 1
_MAX_UINT64: Final = (1 << 64) - 1
_MAX_RECORD_BYTES: Final = 1024 * 1024
_CLOCK_ID: Final = "CLOCK_MONOTONIC"

_SAMPLE_FIELDS: Final = (
    "role_id_or_null",
    "sample_ordinal_or_null",
    "previous_finish_ns_or_null",
    "scheduled_start_ns_or_null",
    "actual_start_ns_or_null",
    "finish_ns_or_null",
    "collection_duration_ns_or_null",
    "scheduler_lateness_ns_or_null",
    "target_wait_ns_or_null",
    "gap_ns_or_null",
    "maximum_gap_ns_or_null",
    "gap_exceeded_or_null",
)
_SAMPLE_OPERATIONS: Final = frozenset(
    {"LIVE_RESOURCE_SAMPLE", "HELD_FINAL_RESOURCE_SAMPLE"}
)
_EVENT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "trace_id",
        "producer_id",
        "clock_id",
        "sequence",
        "timestamp_ns",
        "event_type",
        "span_id",
        "operation",
        "stage_or_null",
        "substep_or_null",
        "actor_or_null",
        "detail_or_null",
        "error_code_or_null",
        "error_detail_or_null",
        *_SAMPLE_FIELDS,
        "previous_record_digest_or_null",
        "record_digest",
    }
)
_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema_version",
        "trace_id",
        "producer_id",
        "clock_id",
        "outcome",
        "event_count",
        "first_event_record_digest",
        "final_event_record_digest",
        "terminal_event_type",
        "terminal_at_ns",
        "open_span_ids_in_begin_order",
        "closed_span_count",
        "manifest_digest",
    }
)


class TrajectoryErrorV2(ValueError):
    """Stable fail-closed error for trajectory writing and verification."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise TrajectoryErrorV2(code, detail)


def _exact_integer(
    value: object, label: str, *, minimum: int = 0, maximum: int = _MAX_COUNTER
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(FAIL_SCHEMA_V2, f"{label} must be an exact safe integer")
    return value


def _bounded_text(value: object, label: str, *, maximum: int = 4096) -> str:
    if type(value) is not str or not 1 <= len(value.encode("utf-8")) <= maximum:
        _fail(FAIL_SCHEMA_V2, f"{label} must be bounded nonempty text")
    if "\x00" in value:
        _fail(FAIL_SCHEMA_V2, f"{label} contains NUL")
    return value


def _optional_text(value: object, label: str, *, maximum: int = 4096) -> str | None:
    if value is None:
        return None
    return _bounded_text(value, label, maximum=maximum)


def _identifier(value: object, label: str) -> str:
    text = _bounded_text(value, label, maximum=256)
    if _IDENTIFIER.fullmatch(text) is None:
        _fail(FAIL_SCHEMA_V2, f"{label} is not an exact identifier")
    return text


def _tagged_digest(value: object, label: str) -> str:
    if type(value) is not str or _TAGGED_SHA256.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V2, f"{label} is not a tagged SHA-256 digest")
    return value


def _validate_json_tree(value: object, label: str = "value") -> None:
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return
    if value_type is list:
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{label}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                _fail(FAIL_CANONICAL_V2, f"{label} has a non-string key")
            _validate_json_tree(item, f"{label}.{key}")
        return
    _fail(FAIL_CANONICAL_V2, f"{label} contains a non-exact JSON type")


def canonical_trajectory_json_bytes_v2(value: object) -> bytes:
    """Return compact, sorted, ASCII JSON followed by exactly one LF."""

    _validate_json_tree(value)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise TrajectoryErrorV2(
            FAIL_CANONICAL_V2, "canonical JSON encoding failed"
        ) from error
    return (encoded + "\n").encode("ascii")


def _domain_digest(domain: bytes, value: Mapping[str, object], excluded: str) -> str:
    if excluded not in value:
        _fail(FAIL_SCHEMA_V2, f"digest exclusion field absent: {excluded}")
    body = {key: item for key, item in value.items() if key != excluded}
    return "sha256:" + hashlib.sha256(
        domain + canonical_trajectory_json_bytes_v2(body)[:-1]
    ).hexdigest()


def _event_digest(value: Mapping[str, object]) -> str:
    return _domain_digest(_EVENT_DOMAIN, value, "record_digest")


def _manifest_digest(value: Mapping[str, object]) -> str:
    return _domain_digest(_MANIFEST_DOMAIN, value, "manifest_digest")


def _normalized_absolute_path(value: str | os.PathLike[str], label: str) -> str:
    try:
        raw = os.fspath(value)
    except TypeError as error:
        raise TrajectoryErrorV2(FAIL_INPUT_V2, f"{label} must be a path") from error
    if type(raw) is not str or not raw or not os.path.isabs(raw):
        _fail(FAIL_INPUT_V2, f"{label} must be an absolute text path")
    if raw != os.path.normpath(raw):
        _fail(FAIL_INPUT_V2, f"{label} must already be lexically normalized")
    return raw


def _open_parent_and_validate(
    root: str | os.PathLike[str], expected_parent: str | os.PathLike[str]
) -> tuple[str, str, str, int]:
    root_text = _normalized_absolute_path(root, "trajectory root")
    parent_text = _normalized_absolute_path(expected_parent, "expected parent")
    if os.path.dirname(root_text) != parent_text:
        _fail(FAIL_INPUT_V2, "trajectory root parent differs from fixed parent")
    basename = os.path.basename(root_text)
    if _SAFE_ROOT_BASENAME.fullmatch(basename) is None:
        _fail(FAIL_INPUT_V2, "trajectory root basename is not safe")
    try:
        before = os.lstat(parent_text)
    except OSError as error:
        raise TrajectoryErrorV2(
            FAIL_FILESYSTEM_V2, "fixed parent lstat failed"
        ) from error
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) & 0o022
    ):
        _fail(
            FAIL_FILESYSTEM_V2,
            "fixed parent must be an owner-controlled non-writable-by-others directory",
        )
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        _fail(FAIL_FILESYSTEM_V2, "O_NOFOLLOW and O_DIRECTORY are required")
    try:
        parent_fd = os.open(
            parent_text,
            os.O_RDONLY | os.O_CLOEXEC | nofollow | directory,
        )
    except OSError as error:
        raise TrajectoryErrorV2(
            FAIL_FILESYSTEM_V2, "fixed parent open failed"
        ) from error
    after = os.fstat(parent_fd)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        os.close(parent_fd)
        _fail(FAIL_FILESYSTEM_V2, "fixed parent identity changed during open")
    return root_text, parent_text, basename, parent_fd


def _require_root_metadata(root_fd: int) -> os.stat_result:
    metadata = os.fstat(root_fd)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        _fail(FAIL_FILESYSTEM_V2, "trajectory root must be owner 0700 directory")
    return metadata


def _root_generation(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_existing_root(
    root: str | os.PathLike[str], expected_parent: str | os.PathLike[str]
) -> tuple[str, str, int, int]:
    root_text, _parent_text, basename, parent_fd = _open_parent_and_validate(
        root, expected_parent
    )
    root_fd: int | None = None
    try:
        root_fd = os.open(
            basename,
            os.O_RDONLY
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW")
            | getattr(os, "O_DIRECTORY"),
            dir_fd=parent_fd,
        )
        path_metadata = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
        opened_metadata = _require_root_metadata(root_fd)
        if (path_metadata.st_dev, path_metadata.st_ino) != (
            opened_metadata.st_dev,
            opened_metadata.st_ino,
        ):
            _fail(FAIL_FILESYSTEM_V2, "trajectory root identity changed during open")
    except TrajectoryErrorV2:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)
        raise
    except OSError as error:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)
        raise TrajectoryErrorV2(
            FAIL_FILESYSTEM_V2, "trajectory root secure open failed"
        ) from error
    assert root_fd is not None
    return root_text, basename, parent_fd, root_fd


def _read_secure_file(root_fd: int, basename: str) -> bytes:
    try:
        before = os.stat(basename, dir_fd=root_fd, follow_symlinks=False)
    except OSError as error:
        raise TrajectoryErrorV2(
            FAIL_FILESYSTEM_V2, f"secure lstat failed: {basename}"
        ) from error
    if not stat.S_ISREG(before.st_mode):
        _fail(FAIL_FILESYSTEM_V2, f"record is not regular: {basename}")
    try:
        descriptor = os.open(
            basename,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW"),
            dir_fd=root_fd,
        )
    except OSError as error:
        raise TrajectoryErrorV2(
            FAIL_FILESYSTEM_V2, f"secure open failed: {basename}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (metadata.st_dev, metadata.st_ino):
            _fail(FAIL_FILESYSTEM_V2, f"record identity changed: {basename}")
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or not 1 <= metadata.st_size <= _MAX_RECORD_BYTES
        ):
            _fail(FAIL_FILESYSTEM_V2, f"record metadata differs: {basename}")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                _fail(FAIL_FILESYSTEM_V2, f"record read was short: {basename}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(FAIL_FILESYSTEM_V2, f"record grew during read: {basename}")
        after = os.fstat(descriptor)
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mode,
            metadata.st_nlink,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
            after.st_nlink,
        ):
            _fail(FAIL_FILESYSTEM_V2, f"record changed during read: {basename}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _opened_entry_metadata(root_fd: int, basename: str) -> os.stat_result:
    """Open one namespace entry without following it and freeze its identity."""

    before = os.stat(basename, dir_fd=root_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        _fail(FAIL_FILESYSTEM_V2, f"torn-tail entry is not regular: {basename}")
    try:
        descriptor = os.open(
            basename,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW"),
            dir_fd=root_fd,
        )
    except OSError as error:
        raise TrajectoryErrorV2(
            FAIL_FILESYSTEM_V2, f"torn-tail secure open failed: {basename}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            _fail(FAIL_FILESYSTEM_V2, f"torn-tail identity changed: {basename}")
        return opened
    finally:
        os.close(descriptor)


def _validate_torn_staging_pair(
    root_fd: int,
    *,
    staging_basename: str,
    final_basename: str,
    final_present: bool,
) -> None:
    """Require a staging-only inode or the exact two-link pre-commit pair."""

    staged = _opened_entry_metadata(root_fd, staging_basename)
    staged_mode = stat.S_IMODE(staged.st_mode)
    if (
        not stat.S_ISREG(staged.st_mode)
        or staged.st_uid != os.geteuid()
        or staged_mode not in (0o600, 0o444)
        or not 0 <= staged.st_size <= _MAX_RECORD_BYTES
    ):
        _fail(FAIL_FILESYSTEM_V2, "torn-tail staging metadata differs")
    if not final_present:
        if staged.st_nlink != 1:
            _fail(FAIL_FILESYSTEM_V2, "standalone torn-tail staging has aliases")
        return
    final = _opened_entry_metadata(root_fd, final_basename)
    if (
        staged_mode != 0o444
        or stat.S_IMODE(final.st_mode) != 0o444
        or not stat.S_ISREG(final.st_mode)
        or final.st_uid != os.geteuid()
        or staged.st_nlink != 2
        or final.st_nlink != 2
        or (staged.st_dev, staged.st_ino) != (final.st_dev, final.st_ino)
        or not 1 <= final.st_size <= _MAX_RECORD_BYTES
    ):
        _fail(FAIL_FILESYSTEM_V2, "torn-tail final/staging pair differs")


def _decode_canonical(payload: bytes, label: str) -> dict[str, object]:
    if (
        type(payload) is not bytes
        or not 2 <= len(payload) <= _MAX_RECORD_BYTES
        or not payload.endswith(b"\n")
        or payload.endswith(b"\n\n")
    ):
        _fail(FAIL_CANONICAL_V2, f"{label} must be bounded single-LF bytes")
    try:
        text = payload[:-1].decode("ascii")
    except UnicodeDecodeError as error:
        raise TrajectoryErrorV2(
            FAIL_CANONICAL_V2, f"{label} is not canonical ASCII"
        ) from error

    def reject_constant(value: str) -> NoReturn:
        _fail(FAIL_CANONICAL_V2, f"{label} contains non-finite number: {value}")

    def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(FAIL_CANONICAL_V2, f"{label} contains duplicate key")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=exact_object,
            parse_constant=reject_constant,
            parse_float=lambda value: _fail(
                FAIL_CANONICAL_V2, f"{label} contains a float"
            ),
        )
    except TrajectoryErrorV2:
        raise
    except (TypeError, ValueError, RecursionError) as error:
        raise TrajectoryErrorV2(
            FAIL_CANONICAL_V2, f"{label} JSON decode failed"
        ) from error
    if type(decoded) is not dict or canonical_trajectory_json_bytes_v2(decoded) != payload:
        _fail(FAIL_CANONICAL_V2, f"{label} canonical readback differs")
    return decoded


def _validate_sample_fields(record: Mapping[str, object]) -> None:
    values = [record[field] for field in _SAMPLE_FIELDS]
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        _fail(FAIL_SCHEMA_V2, "sample metrics must be all-null or complete")
    _identifier(record["role_id_or_null"], "sample role id")
    _exact_integer(record["sample_ordinal_or_null"], "sample ordinal", minimum=1)
    previous = _exact_integer(
        record["previous_finish_ns_or_null"],
        "previous finish",
        maximum=_MAX_UINT64,
    )
    scheduled = _exact_integer(
        record["scheduled_start_ns_or_null"],
        "scheduled start",
        maximum=_MAX_UINT64,
    )
    actual = _exact_integer(
        record["actual_start_ns_or_null"],
        "actual start",
        maximum=_MAX_UINT64,
    )
    finish = _exact_integer(
        record["finish_ns_or_null"], "finish", maximum=_MAX_UINT64
    )
    duration = _exact_integer(
        record["collection_duration_ns_or_null"],
        "collection duration",
        maximum=_MAX_UINT64,
    )
    lateness = _exact_integer(
        record["scheduler_lateness_ns_or_null"],
        "scheduler lateness",
        maximum=_MAX_UINT64,
    )
    wait = _exact_integer(
        record["target_wait_ns_or_null"],
        "target wait",
        minimum=1,
        maximum=_MAX_UINT64,
    )
    gap = _exact_integer(
        record["gap_ns_or_null"], "sample gap", maximum=_MAX_UINT64
    )
    limit = _exact_integer(
        record["maximum_gap_ns_or_null"],
        "maximum sample gap",
        minimum=1,
        maximum=_MAX_UINT64,
    )
    exceeded = record["gap_exceeded_or_null"]
    if type(exceeded) is not bool:
        _fail(FAIL_SCHEMA_V2, "sample gap-exceeded flag must be exact boolean")
    if wait != TARGET_WAIT_NS_V2 or limit != MAXIMUM_GAP_NS_V2:
        _fail(FAIL_SCHEMA_V2, "sample timing policy differs from frozen v2 policy")
    if scheduled < previous or scheduled != previous + wait:
        _fail(FAIL_SCHEMA_V2, "sample target-wait arithmetic differs")
    if actual < scheduled or lateness != actual - scheduled:
        _fail(FAIL_SCHEMA_V2, "sample scheduler-lateness arithmetic differs")
    if finish < actual or duration != finish - actual:
        _fail(FAIL_SCHEMA_V2, "sample collection-duration arithmetic differs")
    if gap != actual - previous:
        _fail(FAIL_SCHEMA_V2, "sample gap arithmetic differs")
    if exceeded is not (gap > limit):
        _fail(FAIL_SCHEMA_V2, "sample gap boundary decision differs")
def _validate_event_record(
    record: object,
    *,
    expected_sequence: int,
    expected_trace_id: str | None,
    expected_producer_id: str | None,
    expected_previous_digest: str | None,
) -> dict[str, object]:
    if type(record) is not dict or frozenset(record) != _EVENT_FIELDS:
        _fail(FAIL_SCHEMA_V2, "event fields differ")
    if record["schema_version"] != EVENT_SCHEMA_V2:
        _fail(FAIL_SCHEMA_V2, "event schema differs")
    trace_id = _identifier(record["trace_id"], "trace id")
    if expected_trace_id is not None and trace_id != expected_trace_id:
        _fail(FAIL_CHAIN_V2, "trace id changed")
    producer_id = _identifier(record["producer_id"], "producer id")
    if expected_producer_id is not None and producer_id != expected_producer_id:
        _fail(FAIL_CHAIN_V2, "producer id changed")
    if record["clock_id"] != _CLOCK_ID:
        _fail(FAIL_SCHEMA_V2, "event clock id differs")
    sequence = _exact_integer(record["sequence"], "event sequence", minimum=1)
    if sequence != expected_sequence:
        _fail(FAIL_CHAIN_V2, "event sequence differs")
    _exact_integer(
        record["timestamp_ns"], "event timestamp", maximum=_MAX_UINT64
    )
    event_type = record["event_type"]
    if event_type not in ("BEGIN", "OK", "FAIL"):
        _fail(FAIL_SCHEMA_V2, "event type differs")
    _identifier(record["span_id"], "span id")
    operation = _identifier(record["operation"], "operation")
    _optional_text(record["stage_or_null"], "stage", maximum=256)
    _optional_text(record["substep_or_null"], "substep", maximum=256)
    _optional_text(record["actor_or_null"], "actor", maximum=256)
    _optional_text(record["detail_or_null"], "detail")
    if event_type == "FAIL":
        code = record["error_code_or_null"]
        if type(code) is not str or _ERROR_CODE.fullmatch(code) is None:
            _fail(FAIL_SCHEMA_V2, "FAIL event error code differs")
        _bounded_text(record["error_detail_or_null"], "FAIL error detail")
    elif record["error_code_or_null"] is not None or record["error_detail_or_null"] is not None:
        _fail(FAIL_SCHEMA_V2, "non-FAIL event contains error fields")
    _validate_sample_fields(record)
    has_sample_metrics = record["role_id_or_null"] is not None
    if has_sample_metrics and operation not in _SAMPLE_OPERATIONS:
        _fail(FAIL_SCHEMA_V2, "sample metrics appear on a non-sample operation")
    if operation in _SAMPLE_OPERATIONS:
        actor = record["actor_or_null"]
        if type(actor) is not str or actor not in SAMPLE_ROLE_ACTOR_MAP_V2.values():
            _fail(FAIL_SCHEMA_V2, "sample operation actor differs from frozen map")
        if has_sample_metrics:
            expected_actor = SAMPLE_ROLE_ACTOR_MAP_V2.get(
                str(record["role_id_or_null"])
            )
            if expected_actor != actor:
                _fail(FAIL_SCHEMA_V2, "sample role and actor binding differs")
    if event_type == "OK" and operation in _SAMPLE_OPERATIONS and not has_sample_metrics:
        _fail(FAIL_SCHEMA_V2, "successful sample terminal event lacks metrics")
    if (
        event_type != "BEGIN"
        and has_sample_metrics
        and int(record["timestamp_ns"]) < int(record["finish_ns_or_null"])
    ):
        _fail(FAIL_SCHEMA_V2, "sample terminal event precedes sample finish")
    gap_exceeded = record["gap_exceeded_or_null"]
    if gap_exceeded is True and (
        event_type != "FAIL"
        or record["error_code_or_null"] != GAP_EXCEEDED_ERROR_CODE_V2
    ):
        _fail(
            FAIL_SCHEMA_V2,
            "exceeded sample gap is not an exact gap-failure terminal event",
        )
    if (
        operation in _SAMPLE_OPERATIONS
        and record["error_code_or_null"] == GAP_EXCEEDED_ERROR_CODE_V2
        and (not has_sample_metrics or gap_exceeded is not True)
    ):
        _fail(FAIL_SCHEMA_V2, "gap-failure code lacks exceeded sample metrics")
    previous = record["previous_record_digest_or_null"]
    if previous is not None:
        _tagged_digest(previous, "previous event digest")
    if previous != expected_previous_digest:
        _fail(FAIL_CHAIN_V2, "previous event digest differs")
    _tagged_digest(record["record_digest"], "event record digest")
    if record["record_digest"] != _event_digest(record):
        _fail(FAIL_CHAIN_V2, "event record digest differs")
    return record


def _context(record: Mapping[str, object]) -> tuple[object, object, object, object]:
    return (
        record["operation"],
        record["stage_or_null"],
        record["substep_or_null"],
        record["actor_or_null"],
    )


def _require_sample_enclosure(
    record: Mapping[str, object],
    active: Mapping[str, tuple[object, object, object, object]],
) -> None:
    operation = str(record["operation"])
    if operation == "LIVE_RESOURCE_SAMPLE":
        required_outer = "LIVE_RESOURCE_SAMPLER"
    elif operation == "HELD_FINAL_RESOURCE_SAMPLE":
        required_outer = "HELD_ACTOR_CLOSE"
    else:
        return
    actor = record["actor_or_null"]
    if not any(
        context[0] == required_outer and context[3] == actor
        for context in active.values()
    ):
        _fail(
            FAIL_CHAIN_V2,
            f"sample event lacks open {required_outer} span for actor",
        )


def _sample_tuple(record: Mapping[str, object]) -> tuple[object, ...] | None:
    if record["role_id_or_null"] is None:
        return None
    return tuple(record[field] for field in _SAMPLE_FIELDS)


def _register_new_sample(
    sample: tuple[object, ...],
    role_state: dict[str, tuple[int, int]],
) -> None:
    role_id = str(sample[0])
    ordinal = int(sample[1])
    previous_finish = int(sample[2])
    finish = int(sample[5])
    prior = role_state.get(role_id)
    if prior is None:
        if ordinal != 1:
            _fail(FAIL_CHAIN_V2, "first sample ordinal for role is not one")
    else:
        prior_ordinal, prior_finish = prior
        if ordinal != prior_ordinal + 1:
            _fail(FAIL_CHAIN_V2, "sample ordinal for role is not contiguous")
        if previous_finish != prior_finish:
            _fail(FAIL_CHAIN_V2, "sample previous-finish chain differs for role")
    role_state[role_id] = (ordinal, finish)


def _apply_sample_transition(
    record: Mapping[str, object],
    *,
    span_samples: dict[str, tuple[object, ...] | None],
    role_state: dict[str, tuple[int, int]],
) -> None:
    """Replay one sample event against the frozen v2 per-role policy.

    The normal producer writes metrics only on terminal sample events.  For
    crash-analysis compatibility, a BEGIN may also contain the same complete,
    non-exceeded metrics as its terminal event; that exact duplicate is one
    observation, not a second ordinal.  Exceeded gaps are FAIL-terminal-only.
    A different duplicate, a reused ordinal on another span, or a broken
    previous-finish link fails closed.
    """

    span_id = str(record["span_id"])
    event_type = str(record["event_type"])
    sample = _sample_tuple(record)
    if event_type == "BEGIN":
        if span_id in span_samples:
            _fail(FAIL_CHAIN_V2, "sample span registry contains duplicate BEGIN")
        span_samples[span_id] = sample
        if sample is not None:
            _register_new_sample(sample, role_state)
        return
    if span_id not in span_samples:
        _fail(FAIL_CHAIN_V2, "sample terminal event has no BEGIN registry")
    begin_sample = span_samples.pop(span_id)
    if sample is None:
        if begin_sample is not None:
            _fail(FAIL_CHAIN_V2, "BEGIN sample metrics lack terminal duplication")
        return
    if begin_sample is not None:
        if sample != begin_sample:
            _fail(FAIL_CHAIN_V2, "BEGIN and terminal sample metrics differ")
        return
    _register_new_sample(sample, role_state)


@dataclass(frozen=True)
class VerifiedOpenSpanContextV2:
    """Same-pass immutable context for one still-open BEGIN."""

    begin_sequence: int
    span_id: str
    operation: str
    stage_or_null: str | None
    substep_or_null: str | None
    actor_or_null: str | None
    detail_or_null: str | None
    begin_record_digest: str


@dataclass(frozen=True)
class VerifiedFailureRowV2:
    """Same-pass immutable reason and path for one verified FAIL event."""

    sequence: int
    timestamp_ns: int
    span_id: str
    operation: str
    stage_or_null: str | None
    substep_or_null: str | None
    actor_or_null: str | None
    detail_or_null: str | None
    error_code: str
    error_detail: str
    role_id_or_null: str | None
    sample_ordinal_or_null: int | None
    previous_finish_ns_or_null: int | None
    scheduled_start_ns_or_null: int | None
    actual_start_ns_or_null: int | None
    finish_ns_or_null: int | None
    collection_duration_ns_or_null: int | None
    scheduler_lateness_ns_or_null: int | None
    target_wait_ns_or_null: int | None
    gap_ns_or_null: int | None
    maximum_gap_ns_or_null: int | None
    gap_exceeded_or_null: bool | None
    record_digest: str


def _open_span_row(record: Mapping[str, object]) -> VerifiedOpenSpanContextV2:
    return VerifiedOpenSpanContextV2(
        begin_sequence=int(record["sequence"]),
        span_id=str(record["span_id"]),
        operation=str(record["operation"]),
        stage_or_null=record["stage_or_null"],
        substep_or_null=record["substep_or_null"],
        actor_or_null=record["actor_or_null"],
        detail_or_null=record["detail_or_null"],
        begin_record_digest=str(record["record_digest"]),
    )


def _failure_row(record: Mapping[str, object]) -> VerifiedFailureRowV2:
    return VerifiedFailureRowV2(
        sequence=int(record["sequence"]),
        timestamp_ns=int(record["timestamp_ns"]),
        span_id=str(record["span_id"]),
        operation=str(record["operation"]),
        stage_or_null=record["stage_or_null"],
        substep_or_null=record["substep_or_null"],
        actor_or_null=record["actor_or_null"],
        detail_or_null=record["detail_or_null"],
        error_code=str(record["error_code_or_null"]),
        error_detail=str(record["error_detail_or_null"]),
        role_id_or_null=record["role_id_or_null"],
        sample_ordinal_or_null=record["sample_ordinal_or_null"],
        previous_finish_ns_or_null=record["previous_finish_ns_or_null"],
        scheduled_start_ns_or_null=record["scheduled_start_ns_or_null"],
        actual_start_ns_or_null=record["actual_start_ns_or_null"],
        finish_ns_or_null=record["finish_ns_or_null"],
        collection_duration_ns_or_null=record["collection_duration_ns_or_null"],
        scheduler_lateness_ns_or_null=record["scheduler_lateness_ns_or_null"],
        target_wait_ns_or_null=record["target_wait_ns_or_null"],
        gap_ns_or_null=record["gap_ns_or_null"],
        maximum_gap_ns_or_null=record["maximum_gap_ns_or_null"],
        gap_exceeded_or_null=record["gap_exceeded_or_null"],
        record_digest=str(record["record_digest"]),
    )


@dataclass(frozen=True)
class TrajectoryVerificationV2:
    """Immutable summary of a strictly verified trajectory namespace."""

    status: str
    complete: bool
    trace_id_or_null: str | None
    producer_id_or_null: str | None
    event_count: int
    final_sequence: int
    first_event_record_digest_or_null: str | None
    final_event_record_digest_or_null: str | None
    terminal_outcome_or_null: str | None
    terminal_manifest_digest_or_null: str | None
    open_span_ids_in_begin_order: tuple[str, ...]
    torn_tail_staging_basename_or_null: str | None
    canonical_event_payloads_in_order: tuple[bytes, ...]
    open_span_contexts_in_begin_order: tuple[VerifiedOpenSpanContextV2, ...]
    failure_rows_in_sequence_order: tuple[VerifiedFailureRowV2, ...]
    terminal_error_code_or_null: str | None
    terminal_error_detail_or_null: str | None


def _event_basename(sequence: int) -> str:
    return f"event-{sequence:020d}-v2.json"


def _staging_basename(final_basename: str) -> str:
    return f".{final_basename}.staging-v2"


class DurableTrajectoryWriterV2:
    """Single-process, thread-safe durable trajectory writer.

    Construct instances with :func:`create_durable_trajectory_writer_v2`.
    Once any validation or publication operation fails, the instance is
    permanently poisoned and every later operation fails closed.
    """

    __slots__ = (
        "_root_text",
        "_basename",
        "_parent_fd",
        "_root_fd",
        "_root_identity",
        "_directory_generation",
        "_trace_id",
        "_producer_id",
        "_clock_ns",
        "_lock",
        "_poisoned",
        "_closed",
        "_terminal",
        "_events",
        "_active",
        "_seen_spans",
        "_span_samples",
        "_role_sample_state",
        "_closed_count",
        "_failure_count",
        "_last_timestamp",
    )

    def __init__(self) -> None:
        raise TypeError("use create_durable_trajectory_writer_v2")

    @property
    def root(self) -> Path:
        return Path(self._root_text)

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def producer_id(self) -> str:
        return self._producer_id

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def _require_live(self) -> None:
        if self._closed:
            _fail(FAIL_STATE_V2, "trajectory writer is closed")
        if self._poisoned:
            _fail(FAIL_STATE_V2, "trajectory writer is permanently poisoned")
        if self._terminal:
            _fail(FAIL_STATE_V2, "trajectory already has a terminal manifest")

    def _poison(self, error: BaseException) -> NoReturn:
        self._poisoned = True
        if isinstance(error, TrajectoryErrorV2):
            raise error
        raise TrajectoryErrorV2(
            FAIL_FILESYSTEM_V2, "durable trajectory publication failed"
        ) from error

    def _check_namespace(self) -> None:
        root_metadata = _require_root_metadata(self._root_fd)
        if (root_metadata.st_dev, root_metadata.st_ino) != self._root_identity:
            _fail(FAIL_FILESYSTEM_V2, "held trajectory root identity changed")
        if _root_generation(root_metadata) != self._directory_generation:
            _fail(FAIL_FILESYSTEM_V2, "trajectory directory generation changed")
        path_metadata = os.stat(
            self._basename, dir_fd=self._parent_fd, follow_symlinks=False
        )
        if (path_metadata.st_dev, path_metadata.st_ino) != self._root_identity:
            _fail(FAIL_FILESYSTEM_V2, "trajectory root path was replaced")
        if self._events and _read_secure_file(
            self._root_fd, _event_basename(len(self._events))
        ) != self._events[-1]:
            _fail(FAIL_FILESYSTEM_V2, "latest published event readback differs")

    def _publish(self, basename: str, payload: bytes) -> None:
        if type(payload) is not bytes or not 1 <= len(payload) <= _MAX_RECORD_BYTES:
            _fail(FAIL_INPUT_V2, "publication payload must be bounded bytes")
        staging = _staging_basename(basename)
        descriptor = os.open(
            staging,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW"),
            0o600,
            dir_fd=self._root_fd,
        )
        try:
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    _fail(FAIL_FILESYSTEM_V2, "record write made no progress")
                written += count
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        # The staging inode is durable before its final name becomes visible.
        # linkat is atomic and no-replace; retaining staging until the first
        # post-link directory fsync lets a verifier distinguish an uncertain
        # tail from the previously committed prefix.
        os.fsync(self._root_fd)
        os.link(
            staging,
            basename,
            src_dir_fd=self._root_fd,
            dst_dir_fd=self._root_fd,
            follow_symlinks=False,
        )
        os.fsync(self._root_fd)
        os.unlink(staging, dir_fd=self._root_fd)
        os.fsync(self._root_fd)
        if _read_secure_file(self._root_fd, basename) != payload:
            _fail(FAIL_FILESYSTEM_V2, "durable record readback differs")
        self._directory_generation = _root_generation(
            _require_root_metadata(self._root_fd)
        )

    def _timestamp(self) -> int:
        value = self._clock_ns()
        timestamp = _exact_integer(
            value, "clock timestamp", maximum=_MAX_UINT64
        )
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            _fail(FAIL_STATE_V2, "trajectory clock moved backwards")
        return timestamp

    def _append(
        self,
        *,
        event_type: str,
        span_id: str,
        operation: str,
        stage: str | None,
        substep: str | None,
        actor: str | None,
        detail: str | None,
        error_code: str | None,
        error_detail: str | None,
        role_id: str | None,
        sample_ordinal: int | None,
        previous_finish_ns: int | None,
        scheduled_start_ns: int | None,
        actual_start_ns: int | None,
        finish_ns: int | None,
        collection_duration_ns: int | None,
        scheduler_lateness_ns: int | None,
        target_wait_ns: int | None,
        gap_ns: int | None,
        maximum_gap_ns: int | None,
        gap_exceeded: bool | None,
    ) -> dict[str, object]:
        with self._lock:
            self._require_live()
            try:
                self._check_namespace()
                sequence = len(self._events) + 1
                timestamp = self._timestamp()
                record: dict[str, object] = {
                    "schema_version": EVENT_SCHEMA_V2,
                    "trace_id": self._trace_id,
                    "producer_id": self._producer_id,
                    "clock_id": _CLOCK_ID,
                    "sequence": sequence,
                    "timestamp_ns": timestamp,
                    "event_type": event_type,
                    "span_id": span_id,
                    "operation": operation,
                    "stage_or_null": stage,
                    "substep_or_null": substep,
                    "actor_or_null": actor,
                    "detail_or_null": detail,
                    "error_code_or_null": error_code,
                    "error_detail_or_null": error_detail,
                    "role_id_or_null": role_id,
                    "sample_ordinal_or_null": sample_ordinal,
                    "previous_finish_ns_or_null": previous_finish_ns,
                    "scheduled_start_ns_or_null": scheduled_start_ns,
                    "actual_start_ns_or_null": actual_start_ns,
                    "finish_ns_or_null": finish_ns,
                    "collection_duration_ns_or_null": collection_duration_ns,
                    "scheduler_lateness_ns_or_null": scheduler_lateness_ns,
                    "target_wait_ns_or_null": target_wait_ns,
                    "gap_ns_or_null": gap_ns,
                    "maximum_gap_ns_or_null": maximum_gap_ns,
                    "gap_exceeded_or_null": gap_exceeded,
                    "previous_record_digest_or_null": (
                        None
                        if not self._events
                        else _decode_canonical(self._events[-1], "prior event")[
                            "record_digest"
                        ]
                    ),
                    "record_digest": None,
                }
                record["record_digest"] = _event_digest(record)
                _validate_event_record(
                    record,
                    expected_sequence=sequence,
                    expected_trace_id=self._trace_id,
                    expected_producer_id=self._producer_id,
                    expected_previous_digest=record["previous_record_digest_or_null"],
                )
                span = _identifier(span_id, "span id")
                context = _context(record)
                next_active = dict(self._active)
                next_seen = set(self._seen_spans)
                next_span_samples = dict(self._span_samples)
                next_role_sample_state = dict(self._role_sample_state)
                next_closed_count = self._closed_count
                next_failure_count = self._failure_count
                _require_sample_enclosure(record, next_active)
                if event_type == "BEGIN":
                    if span in next_seen:
                        _fail(FAIL_STATE_V2, "span id was already used")
                    next_seen.add(span)
                    next_active[span] = context
                else:
                    if span not in next_active:
                        _fail(FAIL_STATE_V2, "terminal span event has no open BEGIN")
                    if next_active[span] != context:
                        _fail(FAIL_STATE_V2, "terminal span context differs from BEGIN")
                    next_active.pop(span)
                    next_closed_count += 1
                    if event_type == "FAIL":
                        next_failure_count += 1
                _apply_sample_transition(
                    record,
                    span_samples=next_span_samples,
                    role_state=next_role_sample_state,
                )
                payload = canonical_trajectory_json_bytes_v2(record)
                self._publish(_event_basename(sequence), payload)
                self._events.append(payload)
                self._last_timestamp = timestamp
                self._active = next_active
                self._seen_spans = next_seen
                self._span_samples = next_span_samples
                self._role_sample_state = next_role_sample_state
                self._closed_count = next_closed_count
                self._failure_count = next_failure_count
                return dict(record)
            except BaseException as error:
                self._poison(error)

    @staticmethod
    def _sample_arguments(
        *,
        role_id: str | None,
        sample_ordinal: int | None,
        previous_finish_ns: int | None,
        scheduled_start_ns: int | None,
        actual_start_ns: int | None,
        finish_ns: int | None,
        collection_duration_ns: int | None,
        scheduler_lateness_ns: int | None,
        target_wait_ns: int | None,
        gap_ns: int | None,
        maximum_gap_ns: int | None,
        gap_exceeded: bool | None,
    ) -> dict[str, object]:
        return {
            "role_id": role_id,
            "sample_ordinal": sample_ordinal,
            "previous_finish_ns": previous_finish_ns,
            "scheduled_start_ns": scheduled_start_ns,
            "actual_start_ns": actual_start_ns,
            "finish_ns": finish_ns,
            "collection_duration_ns": collection_duration_ns,
            "scheduler_lateness_ns": scheduler_lateness_ns,
            "target_wait_ns": target_wait_ns,
            "gap_ns": gap_ns,
            "maximum_gap_ns": maximum_gap_ns,
            "gap_exceeded": gap_exceeded,
        }

    def begin(
        self,
        span_id: str,
        operation: str,
        *,
        stage: str | None = None,
        substep: str | None = None,
        actor: str | None = None,
        detail: str | None = None,
        role_id: str | None = None,
        sample_ordinal: int | None = None,
        previous_finish_ns: int | None = None,
        scheduled_start_ns: int | None = None,
        actual_start_ns: int | None = None,
        finish_ns: int | None = None,
        collection_duration_ns: int | None = None,
        scheduler_lateness_ns: int | None = None,
        target_wait_ns: int | None = None,
        gap_ns: int | None = None,
        maximum_gap_ns: int | None = None,
        gap_exceeded: bool | None = None,
    ) -> dict[str, object]:
        """Durably append a unique span BEGIN event."""

        return self._append(
            event_type="BEGIN",
            span_id=span_id,
            operation=operation,
            stage=stage,
            substep=substep,
            actor=actor,
            detail=detail,
            error_code=None,
            error_detail=None,
            **self._sample_arguments(
                role_id=role_id,
                sample_ordinal=sample_ordinal,
                previous_finish_ns=previous_finish_ns,
                scheduled_start_ns=scheduled_start_ns,
                actual_start_ns=actual_start_ns,
                finish_ns=finish_ns,
                collection_duration_ns=collection_duration_ns,
                scheduler_lateness_ns=scheduler_lateness_ns,
                target_wait_ns=target_wait_ns,
                gap_ns=gap_ns,
                maximum_gap_ns=maximum_gap_ns,
                gap_exceeded=gap_exceeded,
            ),
        )

    def _end_context(self, span_id: str) -> tuple[str, str | None, str | None, str | None]:
        with self._lock:
            if span_id not in self._active:
                _fail(FAIL_STATE_V2, "terminal span event has no open BEGIN")
            operation, stage, substep, actor = self._active[span_id]
            return str(operation), stage, substep, actor

    def ok(
        self,
        span_id: str,
        *,
        detail: str | None = None,
        role_id: str | None = None,
        sample_ordinal: int | None = None,
        previous_finish_ns: int | None = None,
        scheduled_start_ns: int | None = None,
        actual_start_ns: int | None = None,
        finish_ns: int | None = None,
        collection_duration_ns: int | None = None,
        scheduler_lateness_ns: int | None = None,
        target_wait_ns: int | None = None,
        gap_ns: int | None = None,
        maximum_gap_ns: int | None = None,
        gap_exceeded: bool | None = None,
    ) -> dict[str, object]:
        """Durably close an open span with an OK event."""

        with self._lock:
            self._require_live()
            try:
                operation, stage, substep, actor = self._end_context(span_id)
                return self._append(
                    event_type="OK",
                    span_id=span_id,
                    operation=operation,
                    stage=stage,
                    substep=substep,
                    actor=actor,
                    detail=detail,
                    error_code=None,
                    error_detail=None,
                    **self._sample_arguments(
                        role_id=role_id,
                        sample_ordinal=sample_ordinal,
                        previous_finish_ns=previous_finish_ns,
                        scheduled_start_ns=scheduled_start_ns,
                        actual_start_ns=actual_start_ns,
                        finish_ns=finish_ns,
                        collection_duration_ns=collection_duration_ns,
                        scheduler_lateness_ns=scheduler_lateness_ns,
                        target_wait_ns=target_wait_ns,
                        gap_ns=gap_ns,
                        maximum_gap_ns=maximum_gap_ns,
                        gap_exceeded=gap_exceeded,
                    ),
                )
            except BaseException as error:
                self._poison(error)

    def fail(
        self,
        span_id: str,
        *,
        error_code: str,
        error_detail: str,
        detail: str | None = None,
        role_id: str | None = None,
        sample_ordinal: int | None = None,
        previous_finish_ns: int | None = None,
        scheduled_start_ns: int | None = None,
        actual_start_ns: int | None = None,
        finish_ns: int | None = None,
        collection_duration_ns: int | None = None,
        scheduler_lateness_ns: int | None = None,
        target_wait_ns: int | None = None,
        gap_ns: int | None = None,
        maximum_gap_ns: int | None = None,
        gap_exceeded: bool | None = None,
    ) -> dict[str, object]:
        """Durably close an open span with a reason-bearing FAIL event."""

        with self._lock:
            self._require_live()
            try:
                operation, stage, substep, actor = self._end_context(span_id)
                return self._append(
                    event_type="FAIL",
                    span_id=span_id,
                    operation=operation,
                    stage=stage,
                    substep=substep,
                    actor=actor,
                    detail=detail,
                    error_code=error_code,
                    error_detail=error_detail,
                    **self._sample_arguments(
                        role_id=role_id,
                        sample_ordinal=sample_ordinal,
                        previous_finish_ns=previous_finish_ns,
                        scheduled_start_ns=scheduled_start_ns,
                        actual_start_ns=actual_start_ns,
                        finish_ns=finish_ns,
                        collection_duration_ns=collection_duration_ns,
                        scheduler_lateness_ns=scheduler_lateness_ns,
                        target_wait_ns=target_wait_ns,
                        gap_ns=gap_ns,
                        maximum_gap_ns=maximum_gap_ns,
                        gap_exceeded=gap_exceeded,
                    ),
                )
            except BaseException as error:
                self._poison(error)

    def _seal(self, outcome: str) -> dict[str, object]:
        with self._lock:
            self._require_live()
            try:
                self._check_namespace()
                if not self._events:
                    _fail(FAIL_STATE_V2, "cannot seal an empty trajectory")
                last = _decode_canonical(self._events[-1], "last event")
                expected_type = "OK" if outcome == "OK" else "FAIL"
                if last["event_type"] != expected_type:
                    _fail(FAIL_STATE_V2, "terminal event type differs from outcome")
                if outcome == "OK" and self._active:
                    _fail(FAIL_STATE_V2, "successful terminal trace has open spans")
                if outcome == "OK" and self._failure_count:
                    _fail(FAIL_STATE_V2, "failure-tainted trace cannot seal OK")
                timestamp = self._timestamp()
                first = _decode_canonical(self._events[0], "first event")
                manifest: dict[str, object] = {
                    "schema_version": TERMINAL_MANIFEST_SCHEMA_V2,
                    "trace_id": self._trace_id,
                    "producer_id": self._producer_id,
                    "clock_id": _CLOCK_ID,
                    "outcome": outcome,
                    "event_count": len(self._events),
                    "first_event_record_digest": first["record_digest"],
                    "final_event_record_digest": last["record_digest"],
                    "terminal_event_type": last["event_type"],
                    "terminal_at_ns": timestamp,
                    "open_span_ids_in_begin_order": list(self._active),
                    "closed_span_count": self._closed_count,
                    "manifest_digest": None,
                }
                manifest["manifest_digest"] = _manifest_digest(manifest)
                _validate_manifest(
                    manifest,
                    trace_id=self._trace_id,
                    producer_id=self._producer_id,
                    event_count=len(self._events),
                    first_digest=str(first["record_digest"]),
                    final_digest=str(last["record_digest"]),
                    final_event_type=str(last["event_type"]),
                    last_timestamp=int(last["timestamp_ns"]),
                    open_span_ids=tuple(self._active),
                    closed_count=self._closed_count,
                    failure_count=self._failure_count,
                )
                payload = canonical_trajectory_json_bytes_v2(manifest)
                self._publish(TERMINAL_MANIFEST_BASENAME_V2, payload)
                self._terminal = True
                self._last_timestamp = timestamp
                return dict(manifest)
            except BaseException as error:
                self._poison(error)

    def seal_ok(self) -> dict[str, object]:
        """Publish an OK terminal manifest after a final OK event."""

        return self._seal("OK")

    def seal_fail(self) -> dict[str, object]:
        """Publish a FAIL terminal manifest after a final FAIL event."""

        return self._seal("FAIL")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            os.close(self._root_fd)
            os.close(self._parent_fd)

    def __enter__(self) -> DurableTrajectoryWriterV2:
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()


def create_durable_trajectory_writer_v2(
    root: str | os.PathLike[str],
    *,
    expected_parent: str | os.PathLike[str],
    trace_id: str,
    producer_id: str,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> DurableTrajectoryWriterV2:
    """Create one new exclusive owner-private trajectory root and writer."""

    trace = _identifier(trace_id, "trace id")
    producer = _identifier(producer_id, "producer id")
    if not callable(clock_ns):
        _fail(FAIL_INPUT_V2, "clock_ns must be callable")
    root_text, _parent_text, basename, parent_fd = _open_parent_and_validate(
        root, expected_parent
    )
    root_fd: int | None = None
    try:
        os.mkdir(basename, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        root_fd = os.open(
            basename,
            os.O_RDONLY
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW")
            | getattr(os, "O_DIRECTORY"),
            dir_fd=parent_fd,
        )
        root_metadata = _require_root_metadata(root_fd)
        fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except TrajectoryErrorV2:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)
        raise
    except (OSError, BlockingIOError) as error:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)
        raise TrajectoryErrorV2(
            FAIL_FILESYSTEM_V2, "exclusive trajectory root creation failed"
        ) from error
    assert root_fd is not None
    writer = object.__new__(DurableTrajectoryWriterV2)
    writer._root_text = root_text
    writer._basename = basename
    writer._parent_fd = parent_fd
    writer._root_fd = root_fd
    writer._root_identity = (root_metadata.st_dev, root_metadata.st_ino)
    writer._directory_generation = _root_generation(root_metadata)
    writer._trace_id = trace
    writer._producer_id = producer
    writer._clock_ns = clock_ns
    writer._lock = threading.RLock()
    writer._poisoned = False
    writer._closed = False
    writer._terminal = False
    writer._events = []
    writer._active = {}
    writer._seen_spans = set()
    writer._span_samples = {}
    writer._role_sample_state = {}
    writer._closed_count = 0
    writer._failure_count = 0
    writer._last_timestamp = None
    return writer


def _validate_manifest(
    value: object,
    *,
    trace_id: str,
    producer_id: str,
    event_count: int,
    first_digest: str,
    final_digest: str,
    final_event_type: str,
    last_timestamp: int,
    open_span_ids: tuple[str, ...],
    closed_count: int,
    failure_count: int,
) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != _MANIFEST_FIELDS:
        _fail(FAIL_SCHEMA_V2, "terminal manifest fields differ")
    if value["schema_version"] != TERMINAL_MANIFEST_SCHEMA_V2:
        _fail(FAIL_SCHEMA_V2, "terminal manifest schema differs")
    if value["trace_id"] != trace_id:
        _fail(FAIL_CHAIN_V2, "terminal manifest trace id differs")
    if value["producer_id"] != producer_id:
        _fail(FAIL_CHAIN_V2, "terminal manifest producer id differs")
    if value["clock_id"] != _CLOCK_ID:
        _fail(FAIL_SCHEMA_V2, "terminal manifest clock id differs")
    if value["outcome"] not in ("OK", "FAIL"):
        _fail(FAIL_SCHEMA_V2, "terminal outcome differs")
    expected_outcome = "FAIL" if failure_count else "OK"
    if value["outcome"] != expected_outcome:
        _fail(FAIL_CHAIN_V2, "terminal outcome differs from failure-event history")
    if _exact_integer(value["event_count"], "terminal event count", minimum=1) != event_count:
        _fail(FAIL_CHAIN_V2, "terminal event count differs")
    if value["first_event_record_digest"] != first_digest:
        _fail(FAIL_CHAIN_V2, "terminal first-event digest differs")
    if value["final_event_record_digest"] != final_digest:
        _fail(FAIL_CHAIN_V2, "terminal final-event digest differs")
    _tagged_digest(value["first_event_record_digest"], "manifest first digest")
    _tagged_digest(value["final_event_record_digest"], "manifest final digest")
    if value["terminal_event_type"] != final_event_type:
        _fail(FAIL_CHAIN_V2, "terminal event type differs")
    expected_type = "OK" if value["outcome"] == "OK" else "FAIL"
    if final_event_type != expected_type:
        _fail(FAIL_SCHEMA_V2, "terminal outcome and event type differ")
    terminal_at = _exact_integer(
        value["terminal_at_ns"], "terminal timestamp", maximum=_MAX_UINT64
    )
    if terminal_at < last_timestamp:
        _fail(FAIL_CHAIN_V2, "terminal timestamp precedes final event")
    raw_open = value["open_span_ids_in_begin_order"]
    if type(raw_open) is not list:
        _fail(FAIL_SCHEMA_V2, "terminal open-span registry differs")
    observed_open = tuple(_identifier(item, "manifest open span id") for item in raw_open)
    if len(set(observed_open)) != len(observed_open) or observed_open != open_span_ids:
        _fail(FAIL_CHAIN_V2, "terminal open-span registry differs")
    if value["outcome"] == "OK" and observed_open:
        _fail(FAIL_SCHEMA_V2, "successful terminal manifest has open spans")
    if _exact_integer(value["closed_span_count"], "closed span count") != closed_count:
        _fail(FAIL_CHAIN_V2, "terminal closed-span count differs")
    _tagged_digest(value["manifest_digest"], "terminal manifest digest")
    if value["manifest_digest"] != _manifest_digest(value):
        _fail(FAIL_CHAIN_V2, "terminal manifest digest differs")
    return value


def verify_durable_trajectory_v2(
    root: str | os.PathLike[str],
    *,
    expected_parent: str | os.PathLike[str],
    allow_incomplete: bool = True,
) -> TrajectoryVerificationV2:
    """Read-only strict verification of a complete trace or incomplete prefix."""

    if type(allow_incomplete) is not bool:
        _fail(FAIL_INPUT_V2, "allow_incomplete must be exact boolean")
    _root_text, _basename, parent_fd, root_fd = _open_existing_root(
        root, expected_parent
    )
    try:
        names = os.listdir(root_fd)
        if len(names) != len(set(names)):
            _fail(FAIL_FILESYSTEM_V2, "trajectory namespace contains duplicate names")
        event_rows: list[tuple[int, str]] = []
        manifest_present = False
        staging_rows: list[tuple[str, int | None, str]] = []
        for name in names:
            match = _EVENT_BASENAME.fullmatch(name)
            staging_match = _STAGING_EVENT_BASENAME.fullmatch(name)
            if match is not None:
                sequence = int(match.group(1))
                if sequence < 1:
                    _fail(FAIL_SCHEMA_V2, "event basename sequence differs")
                event_rows.append((sequence, name))
            elif staging_match is not None:
                sequence = int(staging_match.group(1))
                if sequence < 1:
                    _fail(FAIL_SCHEMA_V2, "staging event sequence differs")
                staging_rows.append(("event", sequence, name))
            elif name == TERMINAL_MANIFEST_BASENAME_V2:
                manifest_present = True
            elif name == _STAGING_MANIFEST_BASENAME:
                staging_rows.append(("manifest", None, name))
            else:
                _fail(FAIL_FILESYSTEM_V2, f"unexpected trajectory entry: {name}")
        if len(staging_rows) > 1:
            _fail(FAIL_FILESYSTEM_V2, "multiple torn-tail staging entries exist")
        torn_tail: str | None = None
        if staging_rows:
            kind, staging_sequence, staging_name = staging_rows[0]
            torn_tail = staging_name
            if kind == "event":
                assert staging_sequence is not None
                if manifest_present:
                    _fail(FAIL_CHAIN_V2, "event torn tail follows terminal manifest")
                final_basename = _event_basename(staging_sequence)
                final_present = any(
                    sequence == staging_sequence for sequence, _name in event_rows
                )
                _validate_torn_staging_pair(
                    root_fd,
                    staging_basename=staging_name,
                    final_basename=final_basename,
                    final_present=final_present,
                )
                if final_present:
                    event_rows = [
                        row for row in event_rows if row[0] != staging_sequence
                    ]
                if staging_sequence != len(event_rows) + 1:
                    _fail(FAIL_CHAIN_V2, "event torn tail is not the next sequence")
            else:
                if not event_rows:
                    _fail(FAIL_CHAIN_V2, "manifest torn tail exists without events")
                _validate_torn_staging_pair(
                    root_fd,
                    staging_basename=staging_name,
                    final_basename=TERMINAL_MANIFEST_BASENAME_V2,
                    final_present=manifest_present,
                )
                manifest_present = False
        event_rows.sort()
        if [sequence for sequence, _name in event_rows] != list(
            range(1, len(event_rows) + 1)
        ):
            _fail(FAIL_CHAIN_V2, "trajectory contains a dropped or reordered sequence")
        if not event_rows:
            if manifest_present:
                _fail(FAIL_CHAIN_V2, "terminal manifest exists without events")
            if not allow_incomplete:
                _fail(FAIL_STATE_V2, "trajectory is an incomplete empty prefix")
            return TrajectoryVerificationV2(
                status=(
                    "INCOMPLETE_TORN_TAIL"
                    if torn_tail is not None
                    else "INCOMPLETE_PREFIX"
                ),
                complete=False,
                trace_id_or_null=None,
                producer_id_or_null=None,
                event_count=0,
                final_sequence=0,
                first_event_record_digest_or_null=None,
                final_event_record_digest_or_null=None,
                terminal_outcome_or_null=None,
                terminal_manifest_digest_or_null=None,
                open_span_ids_in_begin_order=(),
                torn_tail_staging_basename_or_null=torn_tail,
                canonical_event_payloads_in_order=(),
                open_span_contexts_in_begin_order=(),
                failure_rows_in_sequence_order=(),
                terminal_error_code_or_null=None,
                terminal_error_detail_or_null=None,
            )

        trace_id: str | None = None
        producer_id: str | None = None
        previous_digest: str | None = None
        previous_timestamp: int | None = None
        first_digest: str | None = None
        final_event: dict[str, object] | None = None
        active: dict[str, tuple[object, object, object, object]] = {}
        open_rows: dict[str, VerifiedOpenSpanContextV2] = {}
        seen: set[str] = set()
        span_samples: dict[str, tuple[object, ...] | None] = {}
        role_sample_state: dict[str, tuple[int, int]] = {}
        closed_count = 0
        failure_count = 0
        failure_rows: list[VerifiedFailureRowV2] = []
        canonical_payloads: list[bytes] = []
        for expected_sequence, name in event_rows:
            payload = _read_secure_file(root_fd, name)
            decoded = _decode_canonical(payload, name)
            record = _validate_event_record(
                decoded,
                expected_sequence=expected_sequence,
                expected_trace_id=trace_id,
                expected_producer_id=producer_id,
                expected_previous_digest=previous_digest,
            )
            if expected_sequence == 1 and record["event_type"] != "BEGIN":
                _fail(FAIL_CHAIN_V2, "first trajectory event is not BEGIN")
            timestamp = int(record["timestamp_ns"])
            if previous_timestamp is not None and timestamp < previous_timestamp:
                _fail(FAIL_CHAIN_V2, "event timestamps moved backwards")
            span_id = str(record["span_id"])
            context = _context(record)
            _require_sample_enclosure(record, active)
            if record["event_type"] == "BEGIN":
                if span_id in seen:
                    _fail(FAIL_CHAIN_V2, "duplicate span BEGIN")
                seen.add(span_id)
                active[span_id] = context
                open_rows[span_id] = _open_span_row(record)
            else:
                if span_id not in active:
                    _fail(FAIL_CHAIN_V2, "terminal span event has no open BEGIN")
                if active[span_id] != context:
                    _fail(FAIL_CHAIN_V2, "terminal span context differs")
                active.pop(span_id)
                open_rows.pop(span_id)
                closed_count += 1
                if record["event_type"] == "FAIL":
                    failure_count += 1
                    failure_rows.append(_failure_row(record))
            _apply_sample_transition(
                record,
                span_samples=span_samples,
                role_state=role_sample_state,
            )
            trace_id = str(record["trace_id"])
            producer_id = str(record["producer_id"])
            previous_digest = str(record["record_digest"])
            previous_timestamp = timestamp
            first_digest = first_digest or previous_digest
            final_event = record
            canonical_payloads.append(payload)

        assert trace_id is not None
        assert producer_id is not None
        assert first_digest is not None
        assert previous_digest is not None
        assert previous_timestamp is not None
        assert final_event is not None
        if not manifest_present:
            if not allow_incomplete:
                _fail(FAIL_STATE_V2, "trajectory is an incomplete event prefix")
            return TrajectoryVerificationV2(
                status=(
                    "INCOMPLETE_TORN_TAIL"
                    if torn_tail is not None
                    else "INCOMPLETE_PREFIX"
                ),
                complete=False,
                trace_id_or_null=trace_id,
                producer_id_or_null=producer_id,
                event_count=len(event_rows),
                final_sequence=len(event_rows),
                first_event_record_digest_or_null=first_digest,
                final_event_record_digest_or_null=previous_digest,
                terminal_outcome_or_null=None,
                terminal_manifest_digest_or_null=None,
                open_span_ids_in_begin_order=tuple(active),
                torn_tail_staging_basename_or_null=torn_tail,
                canonical_event_payloads_in_order=tuple(canonical_payloads),
                open_span_contexts_in_begin_order=tuple(open_rows.values()),
                failure_rows_in_sequence_order=tuple(failure_rows),
                terminal_error_code_or_null=(
                    str(final_event["error_code_or_null"])
                    if final_event["event_type"] == "FAIL"
                    else None
                ),
                terminal_error_detail_or_null=(
                    str(final_event["error_detail_or_null"])
                    if final_event["event_type"] == "FAIL"
                    else None
                ),
            )

        manifest_payload = _read_secure_file(root_fd, TERMINAL_MANIFEST_BASENAME_V2)
        manifest = _decode_canonical(manifest_payload, "terminal manifest")
        _validate_manifest(
            manifest,
            trace_id=trace_id,
            producer_id=producer_id,
            event_count=len(event_rows),
            first_digest=first_digest,
            final_digest=previous_digest,
            final_event_type=str(final_event["event_type"]),
            last_timestamp=previous_timestamp,
            open_span_ids=tuple(active),
            closed_count=closed_count,
            failure_count=failure_count,
        )
        outcome = str(manifest["outcome"])
        return TrajectoryVerificationV2(
            status=f"COMPLETE_{outcome}",
            complete=True,
            trace_id_or_null=trace_id,
            producer_id_or_null=producer_id,
            event_count=len(event_rows),
            final_sequence=len(event_rows),
            first_event_record_digest_or_null=first_digest,
            final_event_record_digest_or_null=previous_digest,
            terminal_outcome_or_null=outcome,
            terminal_manifest_digest_or_null=str(manifest["manifest_digest"]),
            open_span_ids_in_begin_order=tuple(active),
            torn_tail_staging_basename_or_null=None,
            canonical_event_payloads_in_order=tuple(canonical_payloads),
            open_span_contexts_in_begin_order=tuple(open_rows.values()),
            failure_rows_in_sequence_order=tuple(failure_rows),
            terminal_error_code_or_null=(
                str(final_event["error_code_or_null"])
                if final_event["event_type"] == "FAIL"
                else None
            ),
            terminal_error_detail_or_null=(
                str(final_event["error_detail_or_null"])
                if final_event["event_type"] == "FAIL"
                else None
            ),
        )
    except TrajectoryErrorV2:
        raise
    except OSError as error:
        raise TrajectoryErrorV2(
            FAIL_FILESYSTEM_V2, "trajectory verification filesystem read failed"
        ) from error
    finally:
        os.close(root_fd)
        os.close(parent_fd)
