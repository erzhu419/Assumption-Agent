"""Fail-closed file entrypoint for the frozen Phase-2B V2 recognizer.

The container receives one public V2 input archive and one canonical
``ExecutionFreezeManifest`` in the fixed read-only input directory.  It emits
one V2 prediction archive into the fixed output directory.  This module is a
runtime mechanism; its stdout receipt is not, by itself, an external runtime
attestation, a score, or Phase-2B exit evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from typing import Final, Sequence

from .hashing import canonical_json, stable_hash
from .phase2b_protocol import (
    BaselineKind,
    BaselineRegistration,
    ExecutionFreezeManifest,
)
from .phase2b_recognizer_input_archive_v2 import (
    ARCHIVE_HEADER_BYTES_V2,
    MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2,
    DecodedRecognizerInputArchiveV2,
    RecognizerInputArchiveDispositionV2,
    decode_public_recognizer_input_archive_v2,
)
from .phase2b_recognizer_prediction_archive_v2 import (
    MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2,
    PREDICTION_ARCHIVE_HEADER_BYTES_V2,
    DecodedRecognizerPredictionArchiveV2,
    PredictionArchiveDispositionV2,
    build_recognizer_prediction_archive_v2,
    decode_public_recognizer_prediction_archive_v2,
)
from .phase2b_runner import TOTAL_RECOGNIZER_CASE_COUNT


RECOGNIZER_RUNTIME_COMMAND_V2: Final = "phase2b-recognizer-v2"
RECOGNIZER_RUNTIME_RECEIPT_SCHEMA_VERSION_V2: Final = (
    "hegel-machine-phase2b-recognizer-runtime-receipt/2"
)
RECOGNIZER_RUNTIME_GENERIC_FAILURE_REASON_V2: Final = (
    "phase2b_recognizer_runtime_v2_failed"
)

RECOGNIZER_INPUT_DIRECTORY_V2: Final = "/phase2b/input"
RECOGNIZER_OUTPUT_DIRECTORY_V2: Final = "/phase2b/output"
RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2: Final = (
    "recognizer_input_archive_v2.bin"
)
EXECUTION_FREEZE_MANIFEST_FILENAME_V2: Final = (
    "execution_freeze_manifest.json"
)
RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2: Final = (
    "recognizer_prediction_archive_v2.bin"
)

MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2: Final = 65_536
_READ_CHUNK_BYTES_V2: Final = 1 << 20
_MAXIMUM_PATH_BYTES_V2: Final = 4096
_PENDING_OUTPUT_FILENAME_V2: Final = (
    "." + RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2 + ".pending"
)
_MANIFEST_FIELDS_V2: Final = (
    "protocol_id",
    "exact_freeze_id",
    "git_commit",
    "recognizer_image_digest",
    "configuration_sha256",
    "theory_version_id",
    "adapter_implementation_sha256",
    "selector_implementation_sha256",
    "verifier_registry_sha256",
    "baseline_registrations",
    "isolation_profile_id",
)
_BASELINE_FIELDS_V2: Final = (
    "kind",
    "baseline_spec_id",
    "implementation_id",
    "artifact_sha256",
    "frozen_before_holdout_generation",
)


def encode_execution_freeze_manifest_json_v2(
    manifest: ExecutionFreezeManifest,
) -> bytes:
    """Encode the exact closed manifest representation accepted by this CLI."""

    if type(manifest) is not ExecutionFreezeManifest:
        raise TypeError("runtime manifest encoder requires exact execution freeze")
    manifest.__post_init__()
    encoded = canonical_json(manifest).encode("utf-8")
    if not 1 <= len(encoded) <= MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2:
        raise ValueError("runtime execution freeze manifest byte bound drift")
    return encoded


def _closed_object_v2(
    value: object,
    fields: tuple[str, ...],
    name: str,
) -> dict[str, object]:
    if type(value) is not dict or tuple(sorted(value)) != tuple(sorted(fields)):
        raise ValueError(f"{name} closed field manifest drift")
    return value


def _exact_text_v2(value: object, name: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise TypeError(f"{name} must be exact nonempty text")
    return value


def _reject_json_constant_v2(value: str) -> object:
    raise ValueError("runtime manifest JSON constant is not finite JSON")


def _unique_json_object_v2(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if type(key) is not str or key in value:
            raise ValueError("runtime manifest JSON repeats an object key")
        value[key] = item
    return value


def decode_execution_freeze_manifest_json_v2(
    payload: bytes,
) -> ExecutionFreezeManifest:
    """Decode only the exact canonical, closed manifest JSON representation."""

    if type(payload) is not bytes or not (
        1 <= len(payload) <= MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2
    ):
        raise TypeError("runtime execution freeze manifest bytes drift")
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object_v2,
            parse_constant=_reject_json_constant_v2,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("runtime execution freeze manifest JSON rejected") from exc
    mapping = _closed_object_v2(raw, _MANIFEST_FIELDS_V2, "runtime manifest")
    raw_baselines = mapping["baseline_registrations"]
    if type(raw_baselines) is not list or len(raw_baselines) != len(BaselineKind):
        raise TypeError("runtime manifest baseline list drift")
    baselines: list[BaselineRegistration] = []
    for raw_baseline in raw_baselines:
        baseline = _closed_object_v2(
            raw_baseline,
            _BASELINE_FIELDS_V2,
            "runtime baseline registration",
        )
        if type(baseline["frozen_before_holdout_generation"]) is not bool:
            raise TypeError("runtime baseline frozen flag exact bool drift")
        baselines.append(
            BaselineRegistration(
                kind=BaselineKind(_exact_text_v2(baseline["kind"], "baseline kind")),
                baseline_spec_id=_exact_text_v2(
                    baseline["baseline_spec_id"],
                    "baseline specification ID",
                ),
                implementation_id=_exact_text_v2(
                    baseline["implementation_id"],
                    "baseline implementation ID",
                ),
                artifact_sha256=_exact_text_v2(
                    baseline["artifact_sha256"],
                    "baseline artifact SHA-256",
                ),
                frozen_before_holdout_generation=baseline[
                    "frozen_before_holdout_generation"
                ],
            )
        )
    text_fields = tuple(
        name for name in _MANIFEST_FIELDS_V2 if name != "baseline_registrations"
    )
    texts = {
        name: _exact_text_v2(mapping[name], f"execution freeze {name}")
        for name in text_fields
    }
    manifest = ExecutionFreezeManifest(
        protocol_id=texts["protocol_id"],
        exact_freeze_id=texts["exact_freeze_id"],
        git_commit=texts["git_commit"],
        recognizer_image_digest=texts["recognizer_image_digest"],
        configuration_sha256=texts["configuration_sha256"],
        theory_version_id=texts["theory_version_id"],
        adapter_implementation_sha256=texts["adapter_implementation_sha256"],
        selector_implementation_sha256=texts["selector_implementation_sha256"],
        verifier_registry_sha256=texts["verifier_registry_sha256"],
        baseline_registrations=tuple(baselines),
        isolation_profile_id=texts["isolation_profile_id"],
    )
    if encode_execution_freeze_manifest_json_v2(manifest) != payload:
        raise ValueError("runtime execution freeze manifest is not canonical JSON")
    return manifest


def _canonical_absolute_directory_v2(value: object) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError("runtime directory path shape drift")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("runtime directory path encoding drift") from exc
    if (
        len(encoded) > _MAXIMUM_PATH_BYTES_V2
        or not os.path.isabs(value)
        or os.path.normpath(value) != value
        or os.path.abspath(value) != value
        or value == os.sep
    ):
        raise ValueError("runtime directory path canonicality drift")
    return value


def _secure_flags_v2() -> tuple[int, int, int]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    if not nofollow or not directory or not nonblock:
        raise OSError("runtime secure path flags unavailable")
    common = getattr(os, "O_CLOEXEC", 0) | nofollow | nonblock
    return common, directory, nofollow


def _open_directory_nofollow_v2(path: object) -> int:
    canonical = _canonical_absolute_directory_v2(path)
    common, directory, _ = _secure_flags_v2()
    components = canonical.split(os.sep)[1:]
    if not components or any(not item or item in (".", "..") for item in components):
        raise ValueError("runtime directory component drift")
    descriptor = os.open(os.sep, os.O_RDONLY | common | directory)
    try:
        for component in components:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | common | directory,
                dir_fd=descriptor,
            )
            before_close = os.fstat(next_descriptor)
            if not stat.S_ISDIR(before_close.st_mode):
                os.close(next_descriptor)
                raise ValueError("runtime path component is not a directory")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _stat_fingerprint_v2(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_bounded_named_file_v2(
    directory_descriptor: int,
    filename: str,
    *,
    minimum_bytes: int,
    maximum_bytes: int,
) -> tuple[bytes, tuple[int, ...]]:
    if (
        type(directory_descriptor) is not int
        or type(filename) is not str
        or not filename
        or os.sep in filename
        or filename in (".", "..")
        or type(minimum_bytes) is not int
        or type(maximum_bytes) is not int
        or not 1 <= minimum_bytes <= maximum_bytes
    ):
        raise ValueError("runtime named-file contract drift")
    common, _, _ = _secure_flags_v2()
    descriptor = os.open(
        filename,
        os.O_RDONLY | common,
        dir_fd=directory_descriptor,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("runtime input is not a single-link regular file")
        if not minimum_bytes <= before.st_size <= maximum_bytes:
            raise ValueError("runtime input byte bound drift")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(_READ_CHUNK_BYTES_V2, maximum_bytes + 1 - total),
            )
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError("runtime input grew beyond byte cap")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        payload = b"".join(chunks)
        if (
            _stat_fingerprint_v2(before) != _stat_fingerprint_v2(after)
            or len(payload) != before.st_size
            or not minimum_bytes <= len(payload) <= maximum_bytes
        ):
            raise ValueError("runtime input changed during bounded read")
        return payload, _stat_fingerprint_v2(before)
    finally:
        os.close(descriptor)


def _require_exact_directory_names_v2(
    directory_descriptor: int,
    expected_names: frozenset[str],
    name: str,
) -> None:
    if (
        type(directory_descriptor) is not int
        or type(expected_names) is not frozenset
        or any(type(item) is not str or not item for item in expected_names)
    ):
        raise TypeError(f"runtime {name} namespace contract drift")
    observed: list[str] = []
    with os.scandir(directory_descriptor) as entries:
        for entry in entries:
            if len(observed) >= len(expected_names) + 1:
                raise ValueError(f"runtime {name} namespace exceeds its exact cap")
            if type(entry.name) is not str or not entry.name:
                raise ValueError(f"runtime {name} namespace contains an invalid name")
            observed.append(entry.name)
    if len(observed) != len(set(observed)) or frozenset(observed) != expected_names:
        raise ValueError(f"runtime {name} namespace is not exact")


def _same_inode_v2(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_ISREG(left.st_mode)
        and stat.S_ISREG(right.st_mode)
        and left.st_size == right.st_size
    )


def _same_directory_inode_v2(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_ISDIR(left.st_mode)
        and stat.S_ISDIR(right.st_mode)
        and left.st_uid == right.st_uid
        and left.st_gid == right.st_gid
    )


def _require_directory_path_identity_v2(
    path: str,
    held_descriptor: int,
    name: str,
) -> None:
    rebound_descriptor = _open_directory_nofollow_v2(path)
    try:
        if not _same_directory_inode_v2(
            os.fstat(held_descriptor),
            os.fstat(rebound_descriptor),
        ):
            raise ValueError(f"runtime {name} directory path identity drift")
    finally:
        os.close(rebound_descriptor)


def _write_all_v2(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise OSError("runtime output short write")
        offset += written


def _publish_prediction_archive_noreplace_v2(
    payload: bytes,
    directory_descriptor: int,
    initial_directory_fingerprint: tuple[int, ...],
) -> None:
    if type(payload) is not bytes or not (
        PREDICTION_ARCHIVE_HEADER_BYTES_V2
        <= len(payload)
        <= MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2
    ):
        raise ValueError("runtime prediction output byte bound drift")
    if type(directory_descriptor) is not int or type(initial_directory_fingerprint) is not tuple:
        raise TypeError("runtime output directory binding drift")
    output_descriptor: int | None = None
    try:
        if _stat_fingerprint_v2(os.fstat(directory_descriptor)) != initial_directory_fingerprint:
            raise ValueError("runtime output directory changed before publication")
        _require_exact_directory_names_v2(
            directory_descriptor,
            frozenset(),
            "output-before-publication",
        )
        common, _, nofollow = _secure_flags_v2()
        output_descriptor = os.open(
            _PENDING_OUTPUT_FILENAME_V2,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | common | nofollow,
            0o600,
            dir_fd=directory_descriptor,
        )
        _write_all_v2(output_descriptor, payload)
        current = os.fstat(output_descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or current.st_size != len(payload)
        ):
            raise ValueError("runtime pending output inode drift")
        os.fsync(output_descriptor)
        os.fchmod(output_descriptor, 0o444)
        os.fsync(output_descriptor)
        # Persist the pending entry first.  A crash after this point may leave
        # the fixed pending name, the fixed final name, or both hard-link names;
        # none is accepted as success without the process exit and receipt.
        os.fsync(directory_descriptor)
        pending = os.stat(
            _PENDING_OUTPUT_FILENAME_V2,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        held = os.fstat(output_descriptor)
        if not _same_inode_v2(pending, held) or held.st_nlink != 1:
            raise ValueError("runtime pending output path-to-inode drift")
        # link(2) is an atomic no-replace publication operation: an existing
        # final name causes EEXIST and is never overwritten.
        os.link(
            _PENDING_OUTPUT_FILENAME_V2,
            RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        final = os.stat(
            RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        held = os.fstat(output_descriptor)
        if not _same_inode_v2(final, held) or held.st_nlink != 2:
            raise ValueError("runtime final output path-to-inode drift")
        os.fsync(output_descriptor)
        os.fsync(directory_descriptor)
        os.unlink(_PENDING_OUTPUT_FILENAME_V2, dir_fd=directory_descriptor)
        held = os.fstat(output_descriptor)
        final = os.stat(
            RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if not _same_inode_v2(final, held) or held.st_nlink != 1:
            raise ValueError("runtime final output single-link closure drift")
        os.fsync(output_descriptor)
        os.fsync(directory_descriptor)
        _require_exact_directory_names_v2(
            directory_descriptor,
            frozenset({RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2}),
            "output-after-publication",
        )
    finally:
        if output_descriptor is not None:
            os.close(output_descriptor)


def _parse_arguments_v2(argv: Sequence[str] | None) -> None:
    supplied: object = sys.argv[1:] if argv is None else argv
    if type(supplied) not in (list, tuple):
        raise ValueError("runtime CLI argument container drift")
    raw = tuple(supplied)
    expected = (
        "--input",
        RECOGNIZER_INPUT_DIRECTORY_V2,
        "--output",
        RECOGNIZER_OUTPUT_DIRECTORY_V2,
    )
    if raw != expected or any(type(item) is not str for item in raw):
        raise ValueError("runtime CLI requires the exact frozen mount arguments")


def _canonical_json_line_v2(value: dict[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"


def _run_recognizer_v2() -> dict[str, object]:
    input_directory_descriptor = _open_directory_nofollow_v2(
        RECOGNIZER_INPUT_DIRECTORY_V2
    )
    output_directory_descriptor: int | None = None
    try:
        output_directory_descriptor = _open_directory_nofollow_v2(
            RECOGNIZER_OUTPUT_DIRECTORY_V2
        )
        directory_before = os.fstat(input_directory_descriptor)
        output_directory_before = os.fstat(output_directory_descriptor)
        if (
            directory_before.st_dev == output_directory_before.st_dev
            and directory_before.st_ino == output_directory_before.st_ino
        ):
            raise ValueError("runtime input and output directories alias")
        _require_directory_path_identity_v2(
            RECOGNIZER_INPUT_DIRECTORY_V2,
            input_directory_descriptor,
            "input",
        )
        _require_directory_path_identity_v2(
            RECOGNIZER_OUTPUT_DIRECTORY_V2,
            output_directory_descriptor,
            "output",
        )
        _require_exact_directory_names_v2(
            input_directory_descriptor,
            frozenset(
                {
                    RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2,
                    EXECUTION_FREEZE_MANIFEST_FILENAME_V2,
                }
            ),
            "input",
        )
        _require_exact_directory_names_v2(
            output_directory_descriptor,
            frozenset(),
            "output",
        )
        input_bytes, input_file_fingerprint = _read_bounded_named_file_v2(
            input_directory_descriptor,
            RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2,
            minimum_bytes=ARCHIVE_HEADER_BYTES_V2,
            maximum_bytes=MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2,
        )
        manifest_bytes, manifest_file_fingerprint = _read_bounded_named_file_v2(
            input_directory_descriptor,
            EXECUTION_FREEZE_MANIFEST_FILENAME_V2,
            minimum_bytes=2,
            maximum_bytes=MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2,
        )
        directory_after = os.fstat(input_directory_descriptor)
        if _stat_fingerprint_v2(directory_before) != _stat_fingerprint_v2(
            directory_after
        ):
            raise ValueError("runtime input directory changed during read")
        _require_exact_directory_names_v2(
            input_directory_descriptor,
            frozenset(
                {
                    RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2,
                    EXECUTION_FREEZE_MANIFEST_FILENAME_V2,
                }
            ),
            "input-after-read",
        )
        _require_directory_path_identity_v2(
            RECOGNIZER_INPUT_DIRECTORY_V2,
            input_directory_descriptor,
            "input-after-read",
        )
        _require_directory_path_identity_v2(
            RECOGNIZER_OUTPUT_DIRECTORY_V2,
            output_directory_descriptor,
            "output-before-build",
        )

        manifest = decode_execution_freeze_manifest_json_v2(manifest_bytes)
        input_archive = decode_public_recognizer_input_archive_v2(input_bytes)
        if (
            type(input_archive) is not DecodedRecognizerInputArchiveV2
            or input_archive.disposition
            is not RecognizerInputArchiveDispositionV2.COMPLETE
            or input_archive.archive != input_bytes
            or type(input_archive.rows) is not tuple
            or len(input_archive.rows) != TOTAL_RECOGNIZER_CASE_COUNT
        ):
            raise ValueError("runtime decoded V2 input archive closure drift")
        prediction_archive = build_recognizer_prediction_archive_v2(
            input_archive=input_archive,
            execution_freeze_manifest=manifest,
        )
        if (
            type(prediction_archive) is not DecodedRecognizerPredictionArchiveV2
            or prediction_archive.disposition
            is not PredictionArchiveDispositionV2.COMPLETE
            or type(prediction_archive.archive) is not bytes
            or type(prediction_archive.records) is not tuple
            or len(prediction_archive.records) != TOTAL_RECOGNIZER_CASE_COUNT
        ):
            raise ValueError("runtime V2 prediction builder abstained")
        replay = decode_public_recognizer_prediction_archive_v2(
            prediction_archive.archive
        )
        if (
            type(replay) is not DecodedRecognizerPredictionArchiveV2
            or replay.disposition is not PredictionArchiveDispositionV2.COMPLETE
            or replay.archive != prediction_archive.archive
            or replay.archive_id != prediction_archive.archive_id
            or replay.records != prediction_archive.records
        ):
            raise ValueError("runtime V2 prediction output replay drift")

        # The recognizer must remain blind for the whole computation, not just
        # while the two inputs are initially read.  Re-open and replay both
        # fixed inputs after the builder, and reject namespace, inode, metadata,
        # or byte drift before publishing any prediction.
        if _stat_fingerprint_v2(os.fstat(input_directory_descriptor)) != (
            _stat_fingerprint_v2(directory_before)
        ):
            raise ValueError("runtime input directory changed during build")
        input_bytes_after, input_file_fingerprint_after = (
            _read_bounded_named_file_v2(
                input_directory_descriptor,
                RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2,
                minimum_bytes=ARCHIVE_HEADER_BYTES_V2,
                maximum_bytes=MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2,
            )
        )
        manifest_bytes_after, manifest_file_fingerprint_after = (
            _read_bounded_named_file_v2(
                input_directory_descriptor,
                EXECUTION_FREEZE_MANIFEST_FILENAME_V2,
                minimum_bytes=2,
                maximum_bytes=MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2,
            )
        )
        if (
            input_bytes_after != input_bytes
            or input_file_fingerprint_after != input_file_fingerprint
            or manifest_bytes_after != manifest_bytes
            or manifest_file_fingerprint_after != manifest_file_fingerprint
        ):
            raise ValueError("runtime input files changed during build")

        _publish_prediction_archive_noreplace_v2(
            prediction_archive.archive,
            output_directory_descriptor,
            _stat_fingerprint_v2(output_directory_before),
        )
        _require_exact_directory_names_v2(
            input_directory_descriptor,
            frozenset(
                {
                    RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2,
                    EXECUTION_FREEZE_MANIFEST_FILENAME_V2,
                }
            ),
            "input-after-build",
        )
        _require_directory_path_identity_v2(
            RECOGNIZER_INPUT_DIRECTORY_V2,
            input_directory_descriptor,
            "input-after-build",
        )
        _require_directory_path_identity_v2(
            RECOGNIZER_OUTPUT_DIRECTORY_V2,
            output_directory_descriptor,
            "output-after-publication",
        )
        receipt_without_id: dict[str, object] = {
            "schema_version": RECOGNIZER_RUNTIME_RECEIPT_SCHEMA_VERSION_V2,
            "disposition": "COMPLETE",
            "reason": "phase2b_recognizer_runtime_v2_complete",
            "input_archive_id": input_archive.archive_id,
            "input_archive_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "execution_freeze_manifest_id": manifest.manifest_id,
            "execution_freeze_manifest_sha256": hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
            "prediction_archive_id": prediction_archive.archive_id,
            "prediction_archive_sha256": hashlib.sha256(
                prediction_archive.archive
            ).hexdigest(),
            "prediction_case_count": len(prediction_archive.records),
            "prediction_archive_bytes": len(prediction_archive.archive),
            "output_filename": RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2,
        }
        return {
            **receipt_without_id,
            "receipt_id": stable_hash(
                receipt_without_id,
                prefix="phase2b_recognizer_runtime_receipt_v2_",
            ),
        }
    finally:
        if output_directory_descriptor is not None:
            os.close(output_directory_descriptor)
        os.close(input_directory_descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the exact fixed-mount recognizer CLI with an atomic failure surface."""

    try:
        _parse_arguments_v2(argv)
        receipt = _run_recognizer_v2()
        line = _canonical_json_line_v2(receipt)
    except Exception:
        sys.stderr.write(
            _canonical_json_line_v2(
                {
                    "disposition": "ABSTAIN",
                    "reason": RECOGNIZER_RUNTIME_GENERIC_FAILURE_REASON_V2,
                }
            )
        )
        return 2
    sys.stdout.write(line)
    return 0


__all__ = (
    "EXECUTION_FREEZE_MANIFEST_FILENAME_V2",
    "MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2",
    "RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2",
    "RECOGNIZER_INPUT_DIRECTORY_V2",
    "RECOGNIZER_OUTPUT_DIRECTORY_V2",
    "RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2",
    "RECOGNIZER_RUNTIME_COMMAND_V2",
    "RECOGNIZER_RUNTIME_GENERIC_FAILURE_REASON_V2",
    "RECOGNIZER_RUNTIME_RECEIPT_SCHEMA_VERSION_V2",
    "decode_execution_freeze_manifest_json_v2",
    "encode_execution_freeze_manifest_json_v2",
    "main",
)


if __name__ == "__main__":
    raise SystemExit(main())
