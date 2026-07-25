"""One-shot opaque-byte acquisition for the pinned MMQA P1 source.

The CLI verifies the committed custody, design, and download authorization,
then performs at most one HTTPS GET for each of the four authorized files.  A
durable attempt marker is consumed before the first network call.  Each body is
streamed to a new mode-0600 O_EXCL part file, checked against its exact byte
size and Git-blob SHA-1 while a SHA-256 receipt is computed, fsynced, and
atomically renamed into the exact source directory used by the qualifier.

No response body is decompressed, decoded, parsed, printed, or included in a
receipt.  There is no retry, resume, range request, mirror, or provider switch.
Tests use synthetic byte streams and must never call the formal URLs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, BinaryIO
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)


VERSION = "mmqa_p1_source_acquisition_v1"
STUDY_ID = "MMQA_P1_LOCAL_PROOF_E5_V1"
SOURCE_COMMIT = "4dd14328c6d02a4daa357cc6032915a0b14602e3"
EXPECTED_HOST = "raw.githubusercontent.com"
EXPECTED_CUSTODY_SELF_SHA256 = (
    "e82cb94e54a3020d1f2e41f47ed4141d19b448db985479551b1d933b43bf15f5"
)
EXPECTED_DESIGN_SELF_SHA256 = (
    "eefa61986bd2f58efa26564dc0709728e0323660f23ae532819f4fa98f0601b3"
)
EXPECTED_AUTHORIZATION_SELF_SHA256 = (
    "08f4bbc25c7d15182b16da909d535a4492e80c302940742e1e92c2828d7360cb"
)

# ``--project`` is the reconstruction_v2 project root on the remote host.
# This relative source path is exactly the qualifier's PROJECT_ROOT-relative
# source root.
SOURCE_ROOT_RELATIVE = Path("artifacts/mmqa_p1_official_source_v1")
CONTROL_ROOT_RELATIVE = Path("artifacts/mmqa_p1_source_acquisition_v1")
ATTEMPT_MARKER_RELATIVE = CONTROL_ROOT_RELATIVE / "download.one_shot_attempt.json"
FILE_ATTEMPT_ROOT_RELATIVE = CONTROL_ROOT_RELATIVE / "network_attempts"
RECEIPT_RELATIVE = Path("manifests/mmqa_p1_source_download_receipt_v1.json")
FAILURE_RECEIPT_RELATIVE = (
    Path("manifests/mmqa_p1_source_download_terminal_failure_v1.json")
)
CUSTODY_RELATIVE = Path("manifests/mmqa_p1_source_custody_v1.json")
DESIGN_RELATIVE = Path("manifests/mmqa_p1_local_proof_e5_study_design_v1.json")
AUTHORIZATION_RELATIVE = (
    Path("manifests/mmqa_p1_source_download_authorization_v1.json")
)

READ_CHUNK_BYTES = 1 << 20
HTTP_TIMEOUT_SECONDS = 900
REMOTE_RUNTIME_PREFIX = Path("runtime/reconstruction_v2")
FILE_ORDER = (
    "MMQA_train.jsonl.gz",
    "MMQA_dev.jsonl.gz",
    "MMQA_tables.jsonl.gz",
    "MMQA_texts.jsonl.gz",
)
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class MMQAP1SourceAcquisitionError(RuntimeError):
    """The frozen one-shot acquisition contract failed closed."""


@dataclass(frozen=True)
class DownloadFile:
    file_name: str
    url: str
    expected_size_bytes: int
    expected_git_blob_sha1: str


@dataclass(frozen=True)
class DownloadPlan:
    files: tuple[DownloadFile, ...]
    authorization_self_sha256: str
    source_custody_self_sha256: str
    study_design_self_sha256: str
    source_root_relative: Path = SOURCE_ROOT_RELATIVE


FORMAL_FILES = {
    "MMQA_train.jsonl.gz": DownloadFile(
        "MMQA_train.jsonl.gz",
        "https://raw.githubusercontent.com/allenai/multimodalqa/"
        + SOURCE_COMMIT
        + "/dataset/MMQA_train.jsonl.gz",
        11_698_210,
        "a6f55fedf35225a217defa3777338f66716304a2",
    ),
    "MMQA_dev.jsonl.gz": DownloadFile(
        "MMQA_dev.jsonl.gz",
        "https://raw.githubusercontent.com/allenai/multimodalqa/"
        + SOURCE_COMMIT
        + "/dataset/MMQA_dev.jsonl.gz",
        1_310_976,
        "7b268187629fe10e2f7678b039baf49c50b29e80",
    ),
    "MMQA_tables.jsonl.gz": DownloadFile(
        "MMQA_tables.jsonl.gz",
        "https://raw.githubusercontent.com/allenai/multimodalqa/"
        + SOURCE_COMMIT
        + "/dataset/MMQA_tables.jsonl.gz",
        10_344_191,
        "c2a8c4add0f12c60cdedd91ab193483bfe0ffa6f",
    ),
    "MMQA_texts.jsonl.gz": DownloadFile(
        "MMQA_texts.jsonl.gz",
        "https://raw.githubusercontent.com/allenai/multimodalqa/"
        + SOURCE_COMMIT
        + "/dataset/MMQA_texts.jsonl.gz",
        45_851_194,
        "debfcc4389f2ddd84647f8b6a2bde3ef41431343",
    ),
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
        raise MMQAP1SourceAcquisitionError(
            "aggregate acquisition receipt is invalid"
        ) from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _self_hashed(body: Mapping[str, Any]) -> dict[str, Any]:
    if "self_sha256" in body:
        raise MMQAP1SourceAcquisitionError(
            "aggregate receipt already contains a self hash"
        )
    result = dict(body)
    result["self_sha256"] = _semantic_hash(result)
    return result


def _git_blob_sha1_for_payload(raw: bytes) -> str:
    digest = hashlib.sha1()  # nosec B324: immutable Git object identity
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise MMQAP1SourceAcquisitionError(
                "acquisition directory is unsafe"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path, *, private_new: bool = True) -> None:
    missing: list[Path] = []
    cursor = path
    while True:
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            if cursor.parent == cursor:
                raise MMQAP1SourceAcquisitionError(
                    "acquisition directory parent is unavailable"
                )
            missing.append(cursor)
            cursor = cursor.parent
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise MMQAP1SourceAcquisitionError(
                "acquisition directory path is unsafe"
            )
        break
    for directory in reversed(missing):
        os.mkdir(directory, 0o700 if private_new else 0o755)
        if private_new:
            os.chmod(directory, 0o700)
        _fsync_directory(directory)
        _fsync_directory(directory.parent)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
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


def _load_bound_manifest(
    path: Path, expected_self_sha256: str
) -> Mapping[str, Any]:
    if not _HEX64.fullmatch(expected_self_sha256):
        raise MMQAP1SourceAcquisitionError("manifest binding is not frozen")
    try:
        value = json.loads(path.read_text("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MMQAP1SourceAcquisitionError(
            "bound acquisition manifest is unavailable"
        ) from exc
    if not isinstance(value, Mapping):
        raise MMQAP1SourceAcquisitionError(
            "bound acquisition manifest shape drifted"
        )
    if value.get("self_sha256") != expected_self_sha256:
        raise MMQAP1SourceAcquisitionError(
            "bound acquisition manifest self hash drifted"
        )
    body = dict(value)
    body.pop("self_sha256", None)
    if _semantic_hash(body) != expected_self_sha256:
        raise MMQAP1SourceAcquisitionError(
            "bound acquisition manifest semantic hash drifted"
        )
    if value.get("study_id") != STUDY_ID:
        raise MMQAP1SourceAcquisitionError(
            "bound acquisition study identity drifted"
        )
    return value


def _validate_source_url(url: str, file_name: str) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise MMQAP1SourceAcquisitionError(
            "authorized source URL is invalid"
        ) from exc
    expected_path = (
        f"/allenai/multimodalqa/{SOURCE_COMMIT}/dataset/{file_name}"
    )
    if (
        parsed.scheme != "https"
        or parsed.hostname != EXPECTED_HOST
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != expected_path
        or parsed.query
        or parsed.fragment
    ):
        raise MMQAP1SourceAcquisitionError(
            "authorized source URL contract drifted"
        )


def _validate_plan(plan: DownloadPlan) -> None:
    if not _HEX64.fullmatch(plan.authorization_self_sha256):
        raise MMQAP1SourceAcquisitionError(
            "download authorization binding is invalid"
        )
    if not _HEX64.fullmatch(plan.source_custody_self_sha256):
        raise MMQAP1SourceAcquisitionError(
            "source custody binding is invalid"
        )
    if not _HEX64.fullmatch(plan.study_design_self_sha256):
        raise MMQAP1SourceAcquisitionError("study design binding is invalid")
    if plan.source_root_relative != SOURCE_ROOT_RELATIVE:
        raise MMQAP1SourceAcquisitionError("source root contract drifted")
    if tuple(file.file_name for file in plan.files) != FILE_ORDER:
        raise MMQAP1SourceAcquisitionError("download file order drifted")
    for file in plan.files:
        if (
            Path(file.file_name).name != file.file_name
            or file.file_name in {"", ".", ".."}
            or isinstance(file.expected_size_bytes, bool)
            or not isinstance(file.expected_size_bytes, int)
            or file.expected_size_bytes < 1
            or not _HEX40.fullmatch(file.expected_git_blob_sha1)
        ):
            raise MMQAP1SourceAcquisitionError(
                "download file contract drifted"
            )
        _validate_source_url(file.url, file.file_name)


def _load_authorized_plan(project: Path) -> DownloadPlan:
    _load_bound_manifest(
        project / CUSTODY_RELATIVE, EXPECTED_CUSTODY_SELF_SHA256
    )
    _load_bound_manifest(project / DESIGN_RELATIVE, EXPECTED_DESIGN_SELF_SHA256)
    authorization = _load_bound_manifest(
        project / AUTHORIZATION_RELATIVE,
        EXPECTED_AUTHORIZATION_SELF_SHA256,
    )
    if (
        authorization.get("status")
        != "authorized_once_before_formal_persisted_source_download"
        or authorization.get("source_custody_self_sha256")
        != EXPECTED_CUSTODY_SELF_SHA256
        or authorization.get("study_design_self_sha256")
        != EXPECTED_DESIGN_SELF_SHA256
        or authorization.get("total_authorized_bytes") != 69_204_571
    ):
        raise MMQAP1SourceAcquisitionError(
            "download authorization contract drifted"
        )
    download = authorization.get("one_shot_four_file_download")
    if not isinstance(download, Mapping):
        raise MMQAP1SourceAcquisitionError(
            "download authorization file registry drifted"
        )
    if (
        download.get("network_attempt_count_per_file") != 1
        or download.get("retry_resume_range_request_or_mirror_fallback_allowed")
        is not False
    ):
        raise MMQAP1SourceAcquisitionError(
            "download authorization retry contract drifted"
        )
    registry = download.get("files")
    if not isinstance(registry, Mapping) or set(registry) != set(FORMAL_FILES):
        raise MMQAP1SourceAcquisitionError(
            "download authorization file registry drifted"
        )
    files: list[DownloadFile] = []
    for file_name in FILE_ORDER:
        value = registry.get(file_name)
        expected = FORMAL_FILES[file_name]
        if not isinstance(value, Mapping):
            raise MMQAP1SourceAcquisitionError(
                "download authorization file entry drifted"
            )
        fixed_path = value.get("fixed_remote_relative_path")
        expected_remote = (
            REMOTE_RUNTIME_PREFIX / SOURCE_ROOT_RELATIVE / file_name
        ).as_posix()
        if (
            value.get("url") != expected.url
            or value.get("expected_size_bytes") != expected.expected_size_bytes
            or value.get("expected_git_blob_sha1")
            != expected.expected_git_blob_sha1
            or fixed_path != expected_remote
        ):
            raise MMQAP1SourceAcquisitionError(
                "download authorization file entry drifted"
            )
        files.append(expected)
    plan = DownloadPlan(
        files=tuple(files),
        authorization_self_sha256=EXPECTED_AUTHORIZATION_SELF_SHA256,
        source_custody_self_sha256=EXPECTED_CUSTODY_SELF_SHA256,
        study_design_self_sha256=EXPECTED_DESIGN_SELF_SHA256,
    )
    _validate_plan(plan)
    return plan


class _SameHostHTTPSRedirectHandler(HTTPRedirectHandler):
    def __init__(self, expected_host: str) -> None:
        super().__init__()
        self.expected_host = expected_host

    def redirect_request(
        self,
        req: Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> Request | None:
        try:
            parsed = urlsplit(newurl)
            port = parsed.port
        except ValueError as exc:
            raise HTTPError(req.full_url, code, "redirect rejected", headers, fp) from exc
        if (
            parsed.scheme != "https"
            or parsed.hostname != self.expected_host
            or port not in {None, 443}
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise HTTPError(req.full_url, code, "redirect rejected", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_open(url: str):
    _validate_source_url(url, Path(urlsplit(url).path).name)
    request = Request(
        url,
        method="GET",
        headers={
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
            "Connection": "close",
            "User-Agent": VERSION,
        },
    )
    opener = build_opener(
        ProxyHandler({}), _SameHostHTTPSRedirectHandler(EXPECTED_HOST)
    )
    return opener.open(request, timeout=HTTP_TIMEOUT_SECONDS)


def _response_status(response: BinaryIO) -> int | None:
    status = getattr(response, "status", None)
    if status is None:
        getcode = getattr(response, "getcode", None)
        if callable(getcode):
            status = getcode()
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def _validate_response(response: BinaryIO, file: DownloadFile) -> None:
    if _response_status(response) != 200:
        raise MMQAP1SourceAcquisitionError(
            "authorized source response status drifted"
        )
    geturl = getattr(response, "geturl", None)
    final_url = geturl() if callable(geturl) else None
    if not isinstance(final_url, str):
        raise MMQAP1SourceAcquisitionError(
            "authorized source response URL is unavailable"
        )
    try:
        parsed = urlsplit(final_url)
        port = parsed.port
    except ValueError as exc:
        raise MMQAP1SourceAcquisitionError(
            "authorized source response URL is invalid"
        ) from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != EXPECTED_HOST
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise MMQAP1SourceAcquisitionError(
            "authorized source response host drifted"
        )
    headers = getattr(response, "headers", None)
    if headers is not None and hasattr(headers, "get"):
        content_encoding = headers.get("Content-Encoding")
        if content_encoding not in {None, "", "identity"}:
            raise MMQAP1SourceAcquisitionError(
                "authorized source response encoding drifted"
            )
        content_length = headers.get("Content-Length")
        if content_length not in {None, ""}:
            try:
                parsed_length = int(content_length)
            except (TypeError, ValueError) as exc:
                raise MMQAP1SourceAcquisitionError(
                    "authorized source response length is invalid"
                ) from exc
            if parsed_length != file.expected_size_bytes:
                raise MMQAP1SourceAcquisitionError(
                    "authorized source response length drifted"
                )


def _stream_verified_part(
    response: BinaryIO, part_path: Path, file: DownloadFile
) -> dict[str, Any]:
    descriptor = os.open(
        part_path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    sha256 = hashlib.sha256()
    git_blob_sha1 = hashlib.sha1()  # nosec B324: immutable Git identity
    git_blob_sha1.update(
        f"blob {file.expected_size_bytes}\0".encode("ascii")
    )
    size = 0
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            while True:
                chunk = response.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, bytes):
                    raise MMQAP1SourceAcquisitionError(
                        "authorized source stream emitted non-bytes"
                    )
                size += len(chunk)
                if size > file.expected_size_bytes:
                    raise MMQAP1SourceAcquisitionError(
                        "authorized source stream exceeded fixed size"
                    )
                sha256.update(chunk)
                git_blob_sha1.update(chunk)
                handle.write(chunk)
            if size != file.expected_size_bytes:
                raise MMQAP1SourceAcquisitionError(
                    "authorized source stream size drifted"
                )
            if git_blob_sha1.hexdigest() != file.expected_git_blob_sha1:
                raise MMQAP1SourceAcquisitionError(
                    "authorized source Git-blob identity drifted"
                )
            handle.flush()
            os.fsync(handle.fileno())
            metadata = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size != file.expected_size_bytes
            ):
                raise MMQAP1SourceAcquisitionError(
                    "verified source part metadata drifted"
                )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return {
        "git_blob_sha1": file.expected_git_blob_sha1,
        "sha256": sha256.hexdigest(),
        "size_bytes": size,
    }


def _promote_verified_part(part_path: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise MMQAP1SourceAcquisitionError(
            "fixed source destination is already consumed"
        )
    # The parent is a new mode-0700 directory owned by this attempt, so the
    # absent-target check cannot race with another user.  POSIX rename is
    # atomic and never exposes an unverified destination body.
    os.rename(part_path, destination)
    _fsync_directory(destination.parent)
    metadata = destination.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise MMQAP1SourceAcquisitionError(
            "promoted source file metadata drifted"
        )


def _consume_attempt_marker(project: Path, plan: DownloadPlan) -> str:
    body = {
        "authorization_self_sha256": plan.authorization_self_sha256,
        "file_count": len(plan.files),
        "network_attempt_count_at_marker": 0,
        "retry_resume_mirror_or_provider_switch_count": 0,
        "schema": f"{VERSION}_one_shot_attempt_v1",
        "source_byte_decode_decompress_or_parse_count": 0,
        "status": "consumed_before_source_root_check_or_network",
        "study_id": STUDY_ID,
    }
    return _write_json_exclusive(
        project / ATTEMPT_MARKER_RELATIVE, _self_hashed(body)
    )


def _consume_file_attempt_marker(
    project: Path, ordinal: int, file_name: str
) -> str:
    body = {
        "file_name": file_name,
        "network_attempt_ordinal": ordinal,
        "network_attempts_for_this_file_after_marker_maximum": 1,
        "schema": f"{VERSION}_file_network_attempt_v1",
        "status": "consumed_immediately_before_the_only_network_open",
        "study_id": STUDY_ID,
    }
    path = (
        project
        / FILE_ATTEMPT_ROOT_RELATIVE
        / f"{ordinal:02d}.{file_name}.attempt.json"
    )
    return _write_json_exclusive(path, _self_hashed(body))


def acquire_plan_once(
    project_root: str | Path,
    plan: DownloadPlan,
    *,
    opener: Callable[[str], BinaryIO] = _default_open,
) -> Mapping[str, Any]:
    """Execute one already-validated four-file plan without parsing bytes."""

    project = Path(project_root).resolve(strict=True)
    if not project.is_dir():
        raise MMQAP1SourceAcquisitionError("remote project root is invalid")
    _validate_plan(plan)
    attempt_marker_path = project / ATTEMPT_MARKER_RELATIVE
    if attempt_marker_path.exists() or attempt_marker_path.is_symlink():
        raise MMQAP1SourceAcquisitionError(
            "formal source acquisition attempt is already consumed"
        )
    attempt_marker_sha256 = _consume_attempt_marker(project, plan)
    source_root = project / plan.source_root_relative
    receipt_path = project / RECEIPT_RELATIVE
    failure_path = project / FAILURE_RECEIPT_RELATIVE
    attempted_file_count = 0
    completed: list[dict[str, Any]] = []
    current_part: Path | None = None
    stage = "validate_fresh_destination"
    try:
        if (
            source_root.exists()
            or source_root.is_symlink()
            or receipt_path.exists()
            or receipt_path.is_symlink()
        ):
            raise MMQAP1SourceAcquisitionError(
                "formal source acquisition destination is already consumed"
            )
        _ensure_directory(source_root.parent)
        os.mkdir(source_root, 0o700)
        os.chmod(source_root, 0o700)
        _fsync_directory(source_root)
        _fsync_directory(source_root.parent)
        for ordinal, file in enumerate(plan.files, start=1):
            stage = "consume_file_network_attempt"
            file_marker_sha256 = _consume_file_attempt_marker(
                project, ordinal, file.file_name
            )
            attempted_file_count += 1
            stage = "perform_only_network_open"
            response = opener(file.url)
            try:
                with response:
                    stage = "validate_fixed_https_response"
                    _validate_response(response, file)
                    part_path = source_root / (
                        "." + file.file_name + ".one_shot.part"
                    )
                    current_part = part_path
                    stage = "stream_and_verify_opaque_bytes"
                    binding = _stream_verified_part(response, part_path, file)
            except AttributeError as exc:
                raise MMQAP1SourceAcquisitionError(
                    "authorized source response context drifted"
                ) from exc
            stage = "atomic_promote_verified_bytes"
            destination = source_root / file.file_name
            _promote_verified_part(current_part, destination)
            current_part = None
            completed.append(
                {
                    "file_attempt_marker_sha256": file_marker_sha256,
                    "file_name": file.file_name,
                    "git_blob_sha1": binding["git_blob_sha1"],
                    "sha256": binding["sha256"],
                    "size_bytes": binding["size_bytes"],
                }
            )
        stage = "write_aggregate_download_receipt"
        body = {
            "authorization_self_sha256": plan.authorization_self_sha256,
            "completed_file_count": len(completed),
            "dataset_byte_decode_decompress_JSONL_or_row_parse_count": 0,
            "files": completed,
            "model_action_embedding_reranking_score_or_online_evaluator_count": 0,
            "network_attempt_count": attempted_file_count,
            "network_attempt_count_per_file_maximum": 1,
            "nonmatching_host_redirect_count": 0,
            "response_body_or_URL_query_output_count": 0,
            "retry_resume_range_mirror_or_provider_switch_count": 0,
            "schema": "mmqa_p1_source_download_receipt_v1",
            "source_custody_self_sha256": plan.source_custody_self_sha256,
            "source_root_relative": plan.source_root_relative.as_posix(),
            "status": "four_fixed_sources_downloaded_identity_verified_not_parsed",
            "study_design_self_sha256": plan.study_design_self_sha256,
            "study_id": STUDY_ID,
            "one_shot_attempt_marker_sha256": attempt_marker_sha256,
        }
        receipt = _self_hashed(body)
        _write_json_exclusive(receipt_path, receipt)
        return receipt
    except BaseException:
        if current_part is not None:
            try:
                current_part.unlink()
            except OSError:
                pass
        failure_body = {
            "attempted_file_count": attempted_file_count,
            "completed_file_count": len(completed),
            "dataset_byte_decode_decompress_JSONL_or_row_parse_count": 0,
            "failure_stage": stage,
            "model_action_embedding_reranking_score_or_online_evaluator_count": 0,
            "one_shot_attempt_marker_sha256": attempt_marker_sha256,
            "response_body_URL_or_URL_query_included": False,
            "retry_resume_range_mirror_or_provider_switch_count": 0,
            "schema": "mmqa_p1_source_download_terminal_failure_v1",
            "status": "terminal_failure_attempt_consumed_no_retry",
            "study_id": STUDY_ID,
        }
        try:
            _write_json_exclusive(failure_path, _self_hashed(failure_body))
        except BaseException:
            pass
        raise MMQAP1SourceAcquisitionError(
            "formal source acquisition failed closed"
        ) from None


def run_authorized_acquisition(
    project_root: str | Path,
    *,
    opener: Callable[[str], BinaryIO] = _default_open,
) -> Mapping[str, Any]:
    project = Path(project_root).resolve(strict=True)
    if not project.is_dir():
        raise MMQAP1SourceAcquisitionError("remote project root is invalid")
    # Metadata-only validation happens before the durable formal attempt.
    plan = _load_authorized_plan(project)
    return acquire_plan_once(project, plan, opener=opener)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="remote reconstruction_v2 project root containing manifests/",
    )
    arguments = parser.parse_args(argv)
    try:
        receipt = run_authorized_acquisition(arguments.project)
    except MMQAP1SourceAcquisitionError:
        print(
            "MMQA P1 source acquisition failed closed; inspect the aggregate terminal receipt.",
            file=sys.stderr,
        )
        return 2
    print(_canonical_bytes(receipt).decode("ascii"), end="")
    return 0


def main() -> int:
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_RELATIVE",
    "DownloadFile",
    "DownloadPlan",
    "FAILURE_RECEIPT_RELATIVE",
    "FILE_ORDER",
    "FORMAL_FILES",
    "MMQAP1SourceAcquisitionError",
    "RECEIPT_RELATIVE",
    "SOURCE_ROOT_RELATIVE",
    "VERSION",
    "acquire_plan_once",
    "run_authorized_acquisition",
]
