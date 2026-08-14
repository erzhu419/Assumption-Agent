"""Fixed container runtime for non-authoritative Phase-2B scorer mechanics.

The command accepts one bounded scorer bundle, independently reconstructs the
strict input/prediction/partition and post-reveal gate graphs through the
bundle codec, invokes the three frozen mechanics exactly once, and publishes
two canonical public mechanics reports.  It does not authenticate a reveal,
perform actual scoring or formal gate evaluation, or establish C1 evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Final, Mapping, NoReturn, Sequence

from .phase2b_c1_container_actor_profile_v2 import (
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256,
)
from .phase2b_scorer_bundle_v1 import (
    MAXIMUM_PHASE2B_SCORER_BUNDLE_BYTES_V1,
    PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME,
    PHASE2B_SCORER_BUNDLE_V1_FILENAME,
    PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME,
    PHASE2B_SCORING_SOURCE_SHA256_V1,
    GATE_REPORT_DOMAIN_V1,
    GATE_REPORT_ID_PREFIX_V1,
    SCORE_REPORT_DOMAIN_V1,
    SCORE_REPORT_ID_PREFIX_V1,
    DecodedPhase2BScorerBundleV1,
    EvaluatedPhase2BScorerBundleV1,
    decode_phase2b_scorer_bundle_v1,
    evaluate_phase2b_scorer_bundle_v1,
)


SCORER_RUNTIME_RECEIPT_SCHEMA_VERSION_V1: Final = (
    "hegel-machine-phase2b-scorer-runtime-receipt/1"
)
SCORER_RUNTIME_CLAIM_LEVEL_V1: Final = (
    "NON_AUTHORITATIVE_LOCAL_CONTAINER_SCORER_MECHANICS_ONLY"
)
SCORER_RUNTIME_COMPLETE_REASON_V1: Final = (
    "phase2b_scorer_runtime_v1_complete_not_actual_scoring"
)
SCORER_RUNTIME_RECEIPT_ID_PREFIX_V1: Final = (
    "phase2b_scorer_runtime_receipt_v1_"
)
MAXIMUM_SCORER_REPORT_BYTES_V1: Final = 256 * 1024 * 1024
MAXIMUM_SCORER_RUNTIME_STDOUT_BYTES_V1: Final = 65_536
MAXIMUM_PATH_BYTES_V1: Final = 4_096
_READ_CHUNK_BYTES_V1: Final = 1 << 20
_PENDING_SUFFIX_V1: Final = ".next"


class Phase2BScorerRuntimeCliErrorV1(RuntimeError):
    """Stable internal rejection; the public CLI emits only a generic line."""


def _fail(detail: str) -> NoReturn:
    raise Phase2BScorerRuntimeCliErrorV1(detail)


def _canonical_json_line_v1(value: Mapping[str, object]) -> bytes:
    try:
        payload = (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        _fail(f"runtime receipt cannot be encoded: {exc}")
    if len(payload) > MAXIMUM_SCORER_RUNTIME_STDOUT_BYTES_V1:
        _fail("runtime receipt exceeds its frozen bound")
    return payload


def _report_ids_v1(score_report: bytes, gate_report: bytes) -> tuple[str, str]:
    if type(score_report) is not bytes or type(gate_report) is not bytes:
        _fail("report IDs require exact bytes")
    return (
        SCORE_REPORT_ID_PREFIX_V1
        + hashlib.sha256(SCORE_REPORT_DOMAIN_V1 + score_report).hexdigest(),
        GATE_REPORT_ID_PREFIX_V1
        + hashlib.sha256(GATE_REPORT_DOMAIN_V1 + gate_report).hexdigest(),
    )


def _parse_arguments_v1(argv: Sequence[str] | None) -> tuple[str, str]:
    supplied: object = sys.argv[1:] if argv is None else argv
    if type(supplied) not in {list, tuple}:
        _fail("argument vector exact type differs")
    raw = tuple(supplied)
    if len(raw) != 4 or any(type(item) is not str for item in raw):
        _fail("argument vector must contain exactly four strings")
    values: dict[str, str] = {}
    for index in (0, 2):
        flag, value = raw[index], raw[index + 1]
        if flag not in {"--input", "--output"} or flag in values:
            _fail("argument names differ from the fixed surface")
        values[flag] = value
    if set(values) != {"--input", "--output"}:
        _fail("required arguments are absent")
    return values["--input"], values["--output"]


def _canonical_absolute_path_v1(value: object, *, label: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        _fail(f"{label} is not exact text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        _fail(f"{label} is not UTF-8: {exc}")
    path = Path(value)
    if (
        len(encoded) > MAXIMUM_PATH_BYTES_V1
        or not path.is_absolute()
        or Path(os.path.abspath(os.fspath(path))) != path
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        _fail(f"{label} is not one normalized absolute path")
    return path


def _required_open_flag_v1(name: str) -> int:
    value = getattr(os, name, None)
    if type(value) is not int or value == 0:
        _fail(f"required secure-open flag {name} is unavailable")
    return value


def _directory_flags_v1() -> int:
    return (
        os.O_RDONLY
        | _required_open_flag_v1("O_DIRECTORY")
        | _required_open_flag_v1("O_NOFOLLOW")
        | _required_open_flag_v1("O_NONBLOCK")
        | _required_open_flag_v1("O_CLOEXEC")
    )


def _regular_flags_v1() -> int:
    return (
        os.O_RDONLY
        | _required_open_flag_v1("O_NOFOLLOW")
        | _required_open_flag_v1("O_NONBLOCK")
        | _required_open_flag_v1("O_CLOEXEC")
        | getattr(os, "O_NOCTTY", 0)
    )


def _open_directory_componentwise_v1(path: Path, *, label: str) -> int:
    descriptor: int | None = None
    try:
        descriptor = os.open("/", _directory_flags_v1())
        for component in path.parts[1:]:
            next_descriptor = os.open(
                component,
                _directory_flags_v1(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            _fail(f"{label} is not a directory")
        return descriptor
    except Phase2BScorerRuntimeCliErrorV1:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        _fail(f"{label} cannot be opened componentwise: {exc}")


def _fingerprint_v1(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_bundle_v1(path: Path) -> bytes:
    parent_descriptor = _open_directory_componentwise_v1(
        path.parent,
        label="scorer bundle parent",
    )
    descriptor: int | None = None
    recheck: int | None = None
    try:
        descriptor = os.open(path.name, _regular_flags_v1(), dir_fd=parent_descriptor)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 1 <= before.st_size <= MAXIMUM_PHASE2B_SCORER_BUNDLE_BYTES_V1
        ):
            _fail("scorer bundle is not one bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while total <= MAXIMUM_PHASE2B_SCORER_BUNDLE_BYTES_V1:
            chunk = os.read(
                descriptor,
                min(
                    _READ_CHUNK_BYTES_V1,
                    MAXIMUM_PHASE2B_SCORER_BUNDLE_BYTES_V1 + 1 - total,
                ),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        payload = b"".join(chunks)
        if (
            _fingerprint_v1(before) != _fingerprint_v1(after)
            or len(payload) != before.st_size
            or len(payload) > MAXIMUM_PHASE2B_SCORER_BUNDLE_BYTES_V1
        ):
            _fail("scorer bundle changed or exceeded its bound during read")
        recheck = os.open(path.name, _regular_flags_v1(), dir_fd=parent_descriptor)
        if _fingerprint_v1(os.fstat(recheck)) != _fingerprint_v1(before):
            _fail("scorer bundle namespace identity changed during read")
        return payload
    except Phase2BScorerRuntimeCliErrorV1:
        raise
    except OSError as exc:
        _fail(f"scorer bundle cannot be read: {exc}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if recheck is not None:
            os.close(recheck)
        os.close(parent_descriptor)


def _directory_names_v1(descriptor: int) -> frozenset[str]:
    try:
        names = os.listdir(descriptor)
    except OSError as exc:
        _fail(f"output directory cannot be enumerated: {exc}")
    if any(type(name) is not str or not name or "/" in name for name in names):
        _fail("output directory contains a malformed name")
    return frozenset(names)


def _write_all_v1(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        try:
            written = os.write(descriptor, view[offset:])
        except OSError as exc:
            _fail(f"report write failed: {exc}")
        if written <= 0:
            _fail("report write was short")
        offset += written


def _publish_two_reports_v1(
    directory_descriptor: int,
    reports: tuple[tuple[str, bytes], tuple[str, bytes]],
) -> None:
    allowed = frozenset(
        name
        for filename, _payload in reports
        for name in (filename, filename + _PENDING_SUFFIX_V1)
    )
    if _directory_names_v1(directory_descriptor):
        _fail("scorer output directory must be initially empty")
    created: list[tuple[str, tuple[int, int]]] = []
    committed: list[tuple[str, tuple[int, int]]] = []
    try:
        for filename, payload in reports:
            if (
                type(payload) is not bytes
                or not 2 <= len(payload) <= MAXIMUM_SCORER_REPORT_BYTES_V1
                or not payload.startswith(b"{")
                or not payload.endswith(b"}\n")
                or payload.count(b"\n") != 1
            ):
                _fail(f"{filename} framing or bound differs")
            pending = filename + _PENDING_SUFFIX_V1
            descriptor = os.open(
                pending,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | _required_open_flag_v1("O_NOFOLLOW")
                | _required_open_flag_v1("O_CLOEXEC"),
                0o600,
                dir_fd=directory_descriptor,
            )
            try:
                opened = os.fstat(descriptor)
                created.append((pending, (opened.st_dev, opened.st_ino)))
                _write_all_v1(descriptor, payload)
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fsync(directory_descriptor)
        for filename, _payload in reports:
            pending = filename + _PENDING_SUFFIX_V1
            os.link(
                pending,
                filename,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            source_metadata = os.stat(
                pending,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            committed.append((filename, (source_metadata.st_dev, source_metadata.st_ino)))
        os.fsync(directory_descriptor)
        for filename, _payload in reports:
            os.unlink(filename + _PENDING_SUFFIX_V1, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
        if _directory_names_v1(directory_descriptor) != frozenset(
            filename for filename, _payload in reports
        ):
            _fail("scorer output namespace differs after publication")
    except BaseException:
        # The directory is private to this invocation.  Still remove only the
        # exact inodes this function created or linked, never a replacement.
        for name, identity in reversed((*committed, *created)):
            try:
                metadata = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    stat.S_ISREG(metadata.st_mode)
                    and (metadata.st_dev, metadata.st_ino) == identity
                ):
                    os.unlink(name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        try:
            os.fsync(directory_descriptor)
        except OSError:
            pass
        raise
    if not _directory_names_v1(directory_descriptor) <= allowed:
        _fail("scorer output directory gained a foreign entry")


def _receipt_v1(
    *,
    bundle_payload: bytes,
    decoded: DecodedPhase2BScorerBundleV1,
    evaluation: EvaluatedPhase2BScorerBundleV1,
    score_report: bytes,
    gate_report: bytes,
) -> bytes:
    required_attributes = (
        "scoring_mechanics",
        "replay_input_contract",
        "available_gate_mechanics",
    )
    try:
        values = {name: getattr(evaluation, name) for name in required_attributes}
        score_id = getattr(values["scoring_mechanics"], "result_id")
        replay_id = getattr(values["replay_input_contract"], "result_id")
        gate_id = getattr(values["available_gate_mechanics"], "result_id")
    except Exception as exc:
        _fail(f"bundle evaluation receipt surface differs: {type(exc).__name__}")
    score_report_id, gate_report_id = _report_ids_v1(score_report, gate_report)
    body: dict[str, object] = {
        "schema_version": SCORER_RUNTIME_RECEIPT_SCHEMA_VERSION_V1,
        "claim_level": SCORER_RUNTIME_CLAIM_LEVEL_V1,
        "reason": SCORER_RUNTIME_COMPLETE_REASON_V1,
        "bundle_filename": PHASE2B_SCORER_BUNDLE_V1_FILENAME,
        "bundle_id": decoded.bundle_id,
        "bundle_sha256": hashlib.sha256(bundle_payload).hexdigest(),
        "authority_requirement_id": decoded.authority_requirement_id,
        "actor_profile_content_id": decoded.actor_profile_content_id,
        "actor_profile_canonical_sha256": (
            PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
        ),
        "campaign_id": decoded.campaign_id,
        "attempt_id": decoded.attempt_id,
        "execution_registry_id": decoded.execution_registry_id,
        "scorer_image_digest": decoded.scorer_image_digest,
        "scorer_source_root_sha256": decoded.scorer_source_root_sha256,
        "protocol_id": decoded.protocol_id,
        "exact_freeze_id": decoded.exact_freeze_id,
        "execution_freeze_manifest_id": decoded.execution_freeze_manifest_id,
        "scorer_run_manifest_id": decoded.run_manifest_id,
        "scoring_source_sha256": PHASE2B_SCORING_SOURCE_SHA256_V1,
        "score_mechanics_result_id": score_id,
        "replay_input_contract_result_id": replay_id,
        "available_gate_mechanics_result_id": gate_id,
        "score_report_filename": PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME,
        "score_report_id": score_report_id,
        "score_report_sha256": hashlib.sha256(score_report).hexdigest(),
        "score_report_bytes": len(score_report),
        "available_gate_report_filename": (
            PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME
        ),
        "available_gate_report_id": gate_report_id,
        "available_gate_report_sha256": hashlib.sha256(gate_report).hexdigest(),
        "available_gate_report_bytes": len(gate_report),
        "local_container_mechanics_complete": True,
        "strict_input_prediction_replayed": True,
        "public_partition_rebuilt": True,
        "nine_metric_mechanics_materialized": True,
        "ten_overall_gate_mechanics_materialized": True,
        "twenty_four_slice_gate_mechanics_materialized": True,
        "two_unavailable_gate_inputs_retained": True,
        "reveal_authority_verified": False,
        "prediction_commit_before_reveal_verified": False,
        "scoring_performed": False,
        "prediction_scored": False,
        "formal_gate_evaluation_performed": False,
        "actual_960_case_run_verified": False,
        "effect_evidence": False,
        "formal_c1_report_verified": False,
        "c1_exit_evidence": False,
    }
    canonical_without_id = json.dumps(
        body,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    body["receipt_id"] = SCORER_RUNTIME_RECEIPT_ID_PREFIX_V1 + hashlib.sha256(
        b"HEGEL/PHASE2B/SCORER_RUNTIME_RECEIPT/V1\x00"
        + canonical_without_id
    ).hexdigest()
    return _canonical_json_line_v1(body)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the exact bundle-to-two-reports scorer mechanics command."""

    output_descriptor: int | None = None
    try:
        input_text, output_text = _parse_arguments_v1(argv)
        input_path = _canonical_absolute_path_v1(input_text, label="scorer bundle")
        output_path = _canonical_absolute_path_v1(output_text, label="scorer output")
        if input_path.name != PHASE2B_SCORER_BUNDLE_V1_FILENAME:
            _fail("scorer bundle basename differs")
        if input_path.parent == output_path or output_path in input_path.parents:
            _fail("scorer input and output paths overlap")
        bundle_payload = _read_bundle_v1(input_path)
        decoded = decode_phase2b_scorer_bundle_v1(bundle_payload)
        evaluation = evaluate_phase2b_scorer_bundle_v1(decoded)
        if type(evaluation) is not EvaluatedPhase2BScorerBundleV1:
            _fail("bundle evaluation did not return exact success")
        score_report = evaluation.score_report_bytes
        gate_report = evaluation.available_gate_report_bytes
        if (
            type(score_report) is not bytes
            or type(gate_report) is not bytes
            or _report_ids_v1(score_report, gate_report)
            != (evaluation.score_report_id, evaluation.available_gate_report_id)
        ):
            _fail("bundle evaluation canonical report bytes differ")
        receipt = _receipt_v1(
            bundle_payload=bundle_payload,
            decoded=decoded,
            evaluation=evaluation,
            score_report=score_report,
            gate_report=gate_report,
        )
        output_descriptor = _open_directory_componentwise_v1(
            output_path,
            label="scorer output",
        )
        _publish_two_reports_v1(
            output_descriptor,
            (
                (PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME, score_report),
                (
                    PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME,
                    gate_report,
                ),
            ),
        )
    except Exception:
        # No caller-controlled exception detail crosses the fixed public CLI.
        return 2
    finally:
        if output_descriptor is not None:
            os.close(output_descriptor)
    sys.stdout.buffer.write(receipt)
    sys.stdout.buffer.flush()
    return 0


__all__ = (
    "PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME",
    "GATE_REPORT_DOMAIN_V1",
    "GATE_REPORT_ID_PREFIX_V1",
    "MAXIMUM_SCORER_REPORT_BYTES_V1",
    "MAXIMUM_SCORER_RUNTIME_STDOUT_BYTES_V1",
    "PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME",
    "SCORE_REPORT_DOMAIN_V1",
    "SCORE_REPORT_ID_PREFIX_V1",
    "PHASE2B_SCORER_BUNDLE_V1_FILENAME",
    "SCORER_RUNTIME_CLAIM_LEVEL_V1",
    "SCORER_RUNTIME_COMPLETE_REASON_V1",
    "SCORER_RUNTIME_RECEIPT_ID_PREFIX_V1",
    "SCORER_RUNTIME_RECEIPT_SCHEMA_VERSION_V1",
    "Phase2BScorerRuntimeCliErrorV1",
    "main",
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
