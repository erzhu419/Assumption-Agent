"""One-shot byte-only downloader for the pinned official TAT-QA P18 source.

This step is intentionally separate from trusted acquisition.  It follows five
commit-addressed official URLs, writes their response bodies without decoding
or parsing a dataset row, and emits only aggregate byte bindings.  The trusted
acquisition later verifies every byte against this receipt before its own
exclusive attempt marker is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from typing import Any, BinaryIO, Callable, Mapping, Sequence
from urllib.request import Request, urlopen


VERSION = "tatqa_p18_source_download_v1"
SOURCE_COMMIT = "870accc41953dcde885aabeb963d94aabdc0fbc3"
CUSTODY_SELF_SHA256 = "0544098eb1bad00bf559f15ab35692ae0fe0382d9c7de9ce4f2221a6d7aed6d8"
SOURCE_ROOT_RELATIVE = Path("artifacts/tatqa_p18_official_source_v1/TAT-QA")
RECEIPT_RELATIVE = Path(
    "artifacts/tatqa_p18_official_source_v1/source.download.receipt.json"
)
SOURCE_FILES = (
    "LICENSE",
    "dataset_raw/tatqa_dataset_dev.json",
    "dataset_raw/tatqa_dataset_train.json",
    "dataset_tagop/tatqa_dataset_dev.json",
    "dataset_tagop/tatqa_dataset_train.json",
)
RAW_BASE = (
    "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/"
    + SOURCE_COMMIT
    + "/"
)
MAXIMUM_FILE_BYTES = 512 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024
HTTP_TIMEOUT_SECONDS = 300


class TatqaP18SourceDownloadError(RuntimeError):
    """Pinned source download or exclusive custody failed closed."""


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
        raise TatqaP18SourceDownloadError("receipt is not canonical JSON") from exc


def _stable_hash(value: object) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _self_hashed(body: Mapping[str, Any]) -> dict[str, Any]:
    if "self_sha256" in body:
        raise TatqaP18SourceDownloadError("receipt body already has a self hash")
    result = dict(body)
    result["self_sha256"] = _stable_hash(result)
    return result


def _safe_destination(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
        raise TatqaP18SourceDownloadError("source relative path is unsafe")
    destination = root.joinpath(*value.parts)
    if root not in destination.parents:
        raise TatqaP18SourceDownloadError("source destination escaped its root")
    return destination


def stream_response_exclusive(
    response: BinaryIO,
    destination: Path,
    *,
    maximum_bytes: int = MAXIMUM_FILE_BYTES,
) -> dict[str, Any]:
    """Copy one opaque response stream to one new regular file and hash it."""

    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes < 1:
        raise TatqaP18SourceDownloadError("download byte bound drifted")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    digest = hashlib.sha256()
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
                    raise TatqaP18SourceDownloadError("download stream emitted non-bytes")
                size += len(chunk)
                if size > maximum_bytes:
                    raise TatqaP18SourceDownloadError("download exceeded frozen byte bound")
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
                raise TatqaP18SourceDownloadError("downloaded file mode drifted")
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    if size < 1:
        raise TatqaP18SourceDownloadError("downloaded file is empty")
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def _default_open(url: str):
    request = Request(
        url,
        method="GET",
        headers={"Accept": "application/octet-stream", "User-Agent": VERSION},
    )
    return urlopen(request, timeout=HTTP_TIMEOUT_SECONDS)


def _write_receipt_exclusive(path: Path, receipt: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(receipt)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _default_verify_freeze(project: Path) -> Mapping[str, Any]:
    # Import lazily so the byte-only downloader remains independently testable.
    # The verifier reads only committed metadata and implementation files; it
    # cannot locate or open a source payload.
    from . import tatqa_p18_acquisition_v1 as acquisition

    try:
        acquisition._verify_contracts(project)
        return acquisition._verify_freeze(project)
    except acquisition.TatqaP18AcquisitionError as exc:
        raise TatqaP18SourceDownloadError(
            "committed P18 implementation freeze is not qualified"
        ) from exc


def download_pinned_source(
    project_root: str | Path,
    *,
    opener: Callable[[str], BinaryIO] = _default_open,
    freeze_verifier: Callable[[Path], Mapping[str, Any]] = _default_verify_freeze,
) -> dict[str, Any]:
    """Download the five frozen files once without decoding any response body."""

    project = Path(project_root).resolve(strict=True)
    if not project.is_dir():
        raise TatqaP18SourceDownloadError("project root is invalid")
    # This metadata-only check is deliberately before even a destination stat
    # or mkdir.  A failed freeze therefore cannot burn a download attempt and
    # cannot expose any formal source byte or row.
    freeze = freeze_verifier(project)
    freeze_self_sha256 = freeze.get("self_sha256")
    if (
        not isinstance(freeze_self_sha256, str)
        or len(freeze_self_sha256) != 64
        or any(character not in "0123456789abcdef" for character in freeze_self_sha256)
    ):
        raise TatqaP18SourceDownloadError("implementation freeze receipt drifted")
    source_root = project / SOURCE_ROOT_RELATIVE
    artifact_root = source_root.parent
    receipt_path = project / RECEIPT_RELATIVE
    if artifact_root.exists() or artifact_root.is_symlink():
        raise TatqaP18SourceDownloadError("source-download root is already consumed")
    artifact_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    artifact_root.mkdir(mode=0o700)
    source_root.mkdir(mode=0o700)
    rows: list[dict[str, Any]] = []
    try:
        for relative in SOURCE_FILES:
            url = RAW_BASE + relative
            destination = _safe_destination(source_root, relative)
            with opener(url) as response:
                binding = stream_response_exclusive(response, destination)
            rows.append(
                {
                    "relative_path": relative,
                    "sha256": binding["sha256"],
                    "size_bytes": binding["size_bytes"],
                }
            )
        body = {
            "schema": "tatqa_p18_source_download_receipt_v1",
            "version": VERSION,
            "status": "source_download_complete_unopened_by_acquisition",
            "source_commit": SOURCE_COMMIT,
            "source_custody_self_sha256": CUSTODY_SELF_SHA256,
            "implementation_freeze_self_sha256": freeze_self_sha256,
            "source_root_relative": SOURCE_ROOT_RELATIVE.as_posix(),
            "files": rows,
            "dataset_payload_decode_or_row_parse_count": 0,
            "test_split_request_count": 0,
            "official_commit_addressed_GET_count": len(SOURCE_FILES),
            "online_evaluator_or_model_call_count": 0,
            "retry_or_mirror_switch_count": 0,
        }
        receipt = _self_hashed(body)
        _write_receipt_exclusive(receipt_path, receipt)
        return receipt
    except BaseException:
        # A partial root is itself a burned terminal attempt.  Preserve it for
        # audit; never delete it and silently retry.
        failure = _self_hashed(
            {
                "schema": "tatqa_p18_source_download_terminal_failure_v1",
                "version": VERSION,
                "status": "terminal_no_retry_or_mirror_switch",
                "completed_file_count": len(rows),
                "source_or_payload_content_included": False,
            }
        )
        try:
            _write_receipt_exclusive(
                artifact_root / "source.download.terminal_failure.json", failure
            )
        except BaseException:
            pass
        raise


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    arguments = parser.parse_args(argv)
    download_pinned_source(arguments.project)
    return 0


def main() -> int:
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CUSTODY_SELF_SHA256",
    "RECEIPT_RELATIVE",
    "SOURCE_COMMIT",
    "SOURCE_FILES",
    "SOURCE_ROOT_RELATIVE",
    "TatqaP18SourceDownloadError",
    "VERSION",
    "download_pinned_source",
    "stream_response_exclusive",
]
