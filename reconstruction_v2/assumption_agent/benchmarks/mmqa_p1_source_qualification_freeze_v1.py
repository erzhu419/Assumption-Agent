"""Build the MMQA P1 prequalification freeze from opaque source bindings.

This command verifies the committed custody, study design, download
authorization, aggregate acquisition receipt, final qualifier implementation,
final qualifier tests, and all four downloaded source files.  Source files are
opened only as private regular byte streams: exact size, Git-blob SHA-1, and
the acquisition receipt's SHA-256 are checked without decompression, decoding,
JSONL parsing, row access, or content output.

Only after every binding passes is the qualifier-compatible freeze written
once with O_EXCL, mode 0600, fsync, and a parent-directory fsync.  Tests use
small synthetic opaque byte streams and never open a formal MMQA source.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any


VERSION = "mmqa_p1_source_qualification_freeze_v1"
STUDY_ID = "MMQA_P1_LOCAL_PROOF_E5_V1"
EXPECTED_CUSTODY_SELF_SHA256 = (
    "e82cb94e54a3020d1f2e41f47ed4141d19b448db985479551b1d933b43bf15f5"
)
EXPECTED_DESIGN_SELF_SHA256 = (
    "eefa61986bd2f58efa26564dc0709728e0323660f23ae532819f4fa98f0601b3"
)
EXPECTED_AUTHORIZATION_SELF_SHA256 = (
    "08f4bbc25c7d15182b16da909d535a4492e80c302940742e1e92c2828d7360cb"
)
EXPECTED_QUALIFIER_SHA256 = (
    "3a3e2106631488d5dfed2a19542ddf1b7da4497bb7687c4e4db28ee372a972a5"
)
EXPECTED_TEST_SHA256 = (
    "599b213928a4ab4a64c8979325fd1602b04b00a339989b03bda93af19c413064"
)

CUSTODY_RELATIVE = Path("manifests/mmqa_p1_source_custody_v1.json")
DESIGN_RELATIVE = Path("manifests/mmqa_p1_local_proof_e5_study_design_v1.json")
AUTHORIZATION_RELATIVE = (
    Path("manifests/mmqa_p1_source_download_authorization_v1.json")
)
DOWNLOAD_RECEIPT_RELATIVE = (
    Path("manifests/mmqa_p1_source_download_receipt_v1.json")
)
FREEZE_RELATIVE = (
    Path("manifests/mmqa_p1_source_qualification_freeze_v1.json")
)
QUALIFIER_RELATIVE = (
    Path("assumption_agent/benchmarks/mmqa_p1_source_qualification_v1.py")
)
TEST_RELATIVE = Path("tests/test_mmqa_p1_source_qualification_v1.py")
SOURCE_ROOT_RELATIVE = Path("artifacts/mmqa_p1_official_source_v1")
FILE_ORDER = (
    "MMQA_train.jsonl.gz",
    "MMQA_dev.jsonl.gz",
    "MMQA_tables.jsonl.gz",
    "MMQA_texts.jsonl.gz",
)
READ_CHUNK_BYTES = 8 << 20
MAXIMUM_METADATA_BYTES = 4 << 20
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class MMQAP1SourceQualificationFreezeError(RuntimeError):
    """The prequalification freeze failed closed."""


@dataclass(frozen=True)
class SourceFileBinding:
    file_name: str
    expected_size_bytes: int
    expected_git_blob_sha1: str


FORMAL_FILES = {
    "MMQA_train.jsonl.gz": SourceFileBinding(
        "MMQA_train.jsonl.gz",
        11_698_210,
        "a6f55fedf35225a217defa3777338f66716304a2",
    ),
    "MMQA_dev.jsonl.gz": SourceFileBinding(
        "MMQA_dev.jsonl.gz",
        1_310_976,
        "7b268187629fe10e2f7678b039baf49c50b29e80",
    ),
    "MMQA_tables.jsonl.gz": SourceFileBinding(
        "MMQA_tables.jsonl.gz",
        10_344_191,
        "c2a8c4add0f12c60cdedd91ab193483bfe0ffa6f",
    ),
    "MMQA_texts.jsonl.gz": SourceFileBinding(
        "MMQA_texts.jsonl.gz",
        45_851_194,
        "debfcc4389f2ddd84647f8b6a2bde3ef41431343",
    ),
}

_RECEIPT_KEYS = {
    "authorization_self_sha256",
    "completed_file_count",
    "dataset_byte_decode_decompress_JSONL_or_row_parse_count",
    "files",
    "model_action_embedding_reranking_score_or_online_evaluator_count",
    "network_attempt_count",
    "network_attempt_count_per_file_maximum",
    "nonmatching_host_redirect_count",
    "one_shot_attempt_marker_sha256",
    "response_body_or_URL_query_output_count",
    "retry_resume_range_mirror_or_provider_switch_count",
    "schema",
    "self_sha256",
    "source_custody_self_sha256",
    "source_root_relative",
    "status",
    "study_design_self_sha256",
    "study_id",
}
_RECEIPT_FILE_KEYS = {
    "file_attempt_marker_sha256",
    "file_name",
    "git_blob_sha1",
    "sha256",
    "size_bytes",
}


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MMQAP1SourceQualificationFreezeError(
            "qualification freeze metadata is invalid"
        ) from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _self_hashed(body: Mapping[str, Any]) -> dict[str, Any]:
    if "self_sha256" in body:
        raise MMQAP1SourceQualificationFreezeError(
            "qualification freeze body already has a self hash"
        )
    result = dict(body)
    result["self_sha256"] = _semantic_hash(result)
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise MMQAP1SourceQualificationFreezeError(
                "qualification freeze parent is unsafe"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while True:
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            if cursor.parent == cursor:
                raise MMQAP1SourceQualificationFreezeError(
                    "qualification freeze parent is unavailable"
                )
            missing.append(cursor)
            cursor = cursor.parent
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise MMQAP1SourceQualificationFreezeError(
                "qualification freeze parent is unsafe"
            )
        break
    for directory in reversed(missing):
        os.mkdir(directory, 0o700)
        os.chmod(directory, 0o700)
        _fsync_directory(directory)
        _fsync_directory(directory.parent)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(value)
    _ensure_directory(path.parent)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def _read_regular_bytes(
    path: Path,
    *,
    maximum_bytes: int | None = None,
    required_mode: int | None = None,
    error_label: str,
) -> bytes:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as exc:
        raise MMQAP1SourceQualificationFreezeError(
            f"{error_label} is unavailable"
        ) from exc
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise MMQAP1SourceQualificationFreezeError(
                f"{error_label} is not a unique regular file"
            )
        if (
            required_mode is not None
            and stat.S_IMODE(before.st_mode) != required_mode
        ):
            raise MMQAP1SourceQualificationFreezeError(
                f"{error_label} mode drifted"
            )
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise MMQAP1SourceQualificationFreezeError(
                f"{error_label} exceeds its byte bound"
            )
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise MMQAP1SourceQualificationFreezeError(
                    f"{error_label} ended early"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise MMQAP1SourceQualificationFreezeError(
                f"{error_label} grew during read"
            )
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise MMQAP1SourceQualificationFreezeError(
                f"{error_label} changed during read"
            )
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _load_json_file(
    path: Path,
    *,
    required_mode: int | None = None,
    error_label: str,
) -> Mapping[str, Any]:
    raw = _read_regular_bytes(
        path,
        maximum_bytes=MAXIMUM_METADATA_BYTES,
        required_mode=required_mode,
        error_label=error_label,
    )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MMQAP1SourceQualificationFreezeError(
            f"{error_label} is invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise MMQAP1SourceQualificationFreezeError(
            f"{error_label} shape drifted"
        )
    return value


def _load_bound_manifest(
    path: Path, expected_self_sha256: str, *, label: str
) -> Mapping[str, Any]:
    if not _HEX64.fullmatch(expected_self_sha256):
        raise MMQAP1SourceQualificationFreezeError(
            f"{label} binding is invalid"
        )
    value = _load_json_file(path, error_label=label)
    if value.get("self_sha256") != expected_self_sha256:
        raise MMQAP1SourceQualificationFreezeError(
            f"{label} self hash drifted"
        )
    body = dict(value)
    body.pop("self_sha256", None)
    if _semantic_hash(body) != expected_self_sha256:
        raise MMQAP1SourceQualificationFreezeError(
            f"{label} semantic hash drifted"
        )
    if value.get("study_id") != STUDY_ID:
        raise MMQAP1SourceQualificationFreezeError(
            f"{label} study identity drifted"
        )
    return value


def _stream_file_sha256(path: Path, *, label: str) -> str:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as exc:
        raise MMQAP1SourceQualificationFreezeError(
            f"{label} is unavailable"
        ) from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise MMQAP1SourceQualificationFreezeError(
                f"{label} is not a unique regular file"
            )
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise MMQAP1SourceQualificationFreezeError(
                    f"{label} ended early"
                )
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise MMQAP1SourceQualificationFreezeError(
                f"{label} grew during read"
            )
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise MMQAP1SourceQualificationFreezeError(
                f"{label} changed during read"
            )
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _verify_implementation(project: Path) -> None:
    qualifier_sha256 = _stream_file_sha256(
        project / QUALIFIER_RELATIVE, label="final qualifier"
    )
    test_sha256 = _stream_file_sha256(
        project / TEST_RELATIVE, label="final qualifier test"
    )
    if qualifier_sha256 != EXPECTED_QUALIFIER_SHA256:
        raise MMQAP1SourceQualificationFreezeError(
            "final qualifier SHA256 drifted"
        )
    if test_sha256 != EXPECTED_TEST_SHA256:
        raise MMQAP1SourceQualificationFreezeError(
            "final qualifier test SHA256 drifted"
        )


def _validate_receipt(
    value: Mapping[str, Any],
) -> tuple[str, dict[str, str]]:
    if set(value) != _RECEIPT_KEYS:
        raise MMQAP1SourceQualificationFreezeError(
            "aggregate acquisition receipt shape drifted"
        )
    claimed = value.get("self_sha256")
    if not isinstance(claimed, str) or not _HEX64.fullmatch(claimed):
        raise MMQAP1SourceQualificationFreezeError(
            "aggregate acquisition receipt self hash is invalid"
        )
    body = dict(value)
    body.pop("self_sha256", None)
    if _semantic_hash(body) != claimed:
        raise MMQAP1SourceQualificationFreezeError(
            "aggregate acquisition receipt semantic hash drifted"
        )
    required = {
        "authorization_self_sha256": EXPECTED_AUTHORIZATION_SELF_SHA256,
        "completed_file_count": len(FILE_ORDER),
        "dataset_byte_decode_decompress_JSONL_or_row_parse_count": 0,
        "model_action_embedding_reranking_score_or_online_evaluator_count": 0,
        "network_attempt_count": len(FILE_ORDER),
        "network_attempt_count_per_file_maximum": 1,
        "nonmatching_host_redirect_count": 0,
        "response_body_or_URL_query_output_count": 0,
        "retry_resume_range_mirror_or_provider_switch_count": 0,
        "schema": "mmqa_p1_source_download_receipt_v1",
        "source_custody_self_sha256": EXPECTED_CUSTODY_SELF_SHA256,
        "source_root_relative": SOURCE_ROOT_RELATIVE.as_posix(),
        "status": (
            "four_fixed_sources_downloaded_identity_verified_not_parsed"
        ),
        "study_design_self_sha256": EXPECTED_DESIGN_SELF_SHA256,
        "study_id": STUDY_ID,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise MMQAP1SourceQualificationFreezeError(
                "aggregate acquisition receipt binding drifted"
            )
    marker_sha256 = value.get("one_shot_attempt_marker_sha256")
    if not isinstance(marker_sha256, str) or not _HEX64.fullmatch(
        marker_sha256
    ):
        raise MMQAP1SourceQualificationFreezeError(
            "aggregate acquisition attempt marker binding drifted"
        )
    rows = value.get("files")
    if not isinstance(rows, list) or len(rows) != len(FILE_ORDER):
        raise MMQAP1SourceQualificationFreezeError(
            "aggregate acquisition source registry drifted"
        )
    observed: dict[str, str] = {}
    for file_name, row in zip(FILE_ORDER, rows, strict=True):
        contract = FORMAL_FILES[file_name]
        if not isinstance(row, Mapping) or set(row) != _RECEIPT_FILE_KEYS:
            raise MMQAP1SourceQualificationFreezeError(
                "aggregate acquisition source entry drifted"
            )
        file_attempt_sha256 = row.get("file_attempt_marker_sha256")
        source_sha256 = row.get("sha256")
        if (
            row.get("file_name") != file_name
            or row.get("git_blob_sha1")
            != contract.expected_git_blob_sha1
            or row.get("size_bytes") != contract.expected_size_bytes
            or not isinstance(file_attempt_sha256, str)
            or not _HEX64.fullmatch(file_attempt_sha256)
            or not isinstance(source_sha256, str)
            or not _HEX64.fullmatch(source_sha256)
        ):
            raise MMQAP1SourceQualificationFreezeError(
                "aggregate acquisition source binding drifted"
            )
        observed[file_name] = source_sha256
    return claimed, observed


def _load_download_receipt(
    project: Path,
) -> tuple[str, dict[str, str]]:
    value = _load_json_file(
        project / DOWNLOAD_RECEIPT_RELATIVE,
        required_mode=0o600,
        error_label="aggregate acquisition receipt",
    )
    return _validate_receipt(value)


def _verify_source_root(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise MMQAP1SourceQualificationFreezeError(
            "fixed source root is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise MMQAP1SourceQualificationFreezeError(
            "fixed source root is not a private directory"
        )


def _verify_one_source(
    path: Path,
    contract: SourceFileBinding,
    expected_sha256: str,
) -> str:
    if not _HEX40.fullmatch(contract.expected_git_blob_sha1):
        raise MMQAP1SourceQualificationFreezeError(
            "fixed source Git-blob binding is invalid"
        )
    if not _HEX64.fullmatch(expected_sha256):
        raise MMQAP1SourceQualificationFreezeError(
            "fixed source SHA256 binding is invalid"
        )
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as exc:
        raise MMQAP1SourceQualificationFreezeError(
            "fixed source file is unavailable"
        ) from exc
    sha256 = hashlib.sha256()
    git_blob_sha1 = hashlib.sha1()  # nosec B324: immutable Git object identity
    git_blob_sha1.update(
        f"blob {contract.expected_size_bytes}\0".encode("ascii")
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise MMQAP1SourceQualificationFreezeError(
                "fixed source file is not a private unique regular file"
            )
        if before.st_size != contract.expected_size_bytes:
            raise MMQAP1SourceQualificationFreezeError(
                "fixed source file size drifted"
            )
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise MMQAP1SourceQualificationFreezeError(
                    "fixed source file ended early"
                )
            sha256.update(chunk)
            git_blob_sha1.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise MMQAP1SourceQualificationFreezeError(
                "fixed source file grew during read"
            )
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise MMQAP1SourceQualificationFreezeError(
                "fixed source file changed during read"
            )
    finally:
        os.close(descriptor)
    if git_blob_sha1.hexdigest() != contract.expected_git_blob_sha1:
        raise MMQAP1SourceQualificationFreezeError(
            "fixed source Git-blob identity drifted"
        )
    observed_sha256 = sha256.hexdigest()
    if observed_sha256 != expected_sha256:
        raise MMQAP1SourceQualificationFreezeError(
            "fixed source SHA256 identity drifted"
        )
    return observed_sha256


def _verify_sources(
    project: Path, expected_sha256_by_file: Mapping[str, str]
) -> dict[str, str]:
    if set(expected_sha256_by_file) != set(FILE_ORDER):
        raise MMQAP1SourceQualificationFreezeError(
            "fixed source SHA256 registry drifted"
        )
    source_root = project / SOURCE_ROOT_RELATIVE
    _verify_source_root(source_root)
    try:
        present_names = {entry.name for entry in source_root.iterdir()}
    except OSError as exc:
        raise MMQAP1SourceQualificationFreezeError(
            "fixed source registry is unavailable"
        ) from exc
    if present_names != set(FILE_ORDER):
        raise MMQAP1SourceQualificationFreezeError(
            "fixed source file set drifted"
        )
    observed: dict[str, str] = {}
    for file_name in FILE_ORDER:
        observed[file_name] = _verify_one_source(
            source_root / file_name,
            FORMAL_FILES[file_name],
            expected_sha256_by_file[file_name],
        )
    return observed


def build_qualification_freeze(
    project_root: str | Path,
) -> Mapping[str, Any]:
    """Verify opaque bindings and exclusively create the formal freeze."""

    project = Path(project_root).resolve(strict=True)
    if not project.is_dir():
        raise MMQAP1SourceQualificationFreezeError(
            "remote project root is invalid"
        )
    freeze_path = project / FREEZE_RELATIVE
    if freeze_path.exists() or freeze_path.is_symlink():
        raise MMQAP1SourceQualificationFreezeError(
            "qualification freeze is already consumed"
        )

    _load_bound_manifest(
        project / CUSTODY_RELATIVE,
        EXPECTED_CUSTODY_SELF_SHA256,
        label="source custody",
    )
    _load_bound_manifest(
        project / DESIGN_RELATIVE,
        EXPECTED_DESIGN_SELF_SHA256,
        label="study design",
    )
    _load_bound_manifest(
        project / AUTHORIZATION_RELATIVE,
        EXPECTED_AUTHORIZATION_SELF_SHA256,
        label="download authorization",
    )
    _verify_implementation(project)
    _receipt_self_sha256, expected_source_sha256 = _load_download_receipt(
        project
    )
    observed_source_sha256 = _verify_sources(
        project, expected_source_sha256
    )
    if observed_source_sha256 != expected_source_sha256:
        raise MMQAP1SourceQualificationFreezeError(
            "fixed source aggregate SHA256 binding drifted"
        )

    body = {
        "download_authorization_self_sha256": (
            EXPECTED_AUTHORIZATION_SELF_SHA256
        ),
        "qualifier_sha256": EXPECTED_QUALIFIER_SHA256,
        "schema": VERSION,
        "source_custody_self_sha256": EXPECTED_CUSTODY_SELF_SHA256,
        "source_sha256_by_file": {
            file_name: observed_source_sha256[file_name]
            for file_name in sorted(observed_source_sha256)
        },
        "status": "frozen_before_unique_formal_qualification",
        "study_design_self_sha256": EXPECTED_DESIGN_SELF_SHA256,
        "study_id": STUDY_ID,
        "test_sha256": EXPECTED_TEST_SHA256,
    }
    freeze = _self_hashed(body)
    _write_exclusive(freeze_path, freeze)
    return freeze


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="remote reconstruction_v2 project root",
    )
    arguments = parser.parse_args(argv)
    try:
        freeze = build_qualification_freeze(arguments.project)
    except MMQAP1SourceQualificationFreezeError:
        print(
            "MMQA P1 source qualification freeze failed closed.",
            file=sys.stderr,
        )
        return 2
    print(_canonical_bytes(freeze).decode("ascii"), end="")
    return 0


def main() -> int:
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DOWNLOAD_RECEIPT_RELATIVE",
    "FILE_ORDER",
    "FORMAL_FILES",
    "FREEZE_RELATIVE",
    "MMQAP1SourceQualificationFreezeError",
    "SOURCE_ROOT_RELATIVE",
    "SourceFileBinding",
    "VERSION",
    "build_qualification_freeze",
]
