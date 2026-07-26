"""One-shot byte-only downloader for the frozen MAUD extraction P2 source.

The downloader neither decodes JSON nor imports a dataset library.  It performs
exactly three commit-addressed GETs, validates byte size and the Git blob
identity while streaming, and writes an aggregate-only receipt.  A partial
root is deliberately preserved as a consumed terminal attempt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import BinaryIO, Callable, ContextManager, Mapping
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)


VERSION = "maud_extraction_p2_download_v1"
STUDY_ID = "MAUD_EXTRACTION_P2_CGROUP_BOUNDED_EVALUATOR_V1"
SOURCE_COMMIT = "89cc5f6ce210268f170aa019ea204ed4e608c604"
SOURCE_CUSTODY_SELF_SHA256 = (
    "b92d3228feb743bf3695e933217b0592025bbd9428d38fd1189af53832de3882"
)
READ_BYTES = 1024 * 1024
TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class FrozenSource:
    split: str
    relative_path: str
    size_bytes: int
    git_blob_sha1: str

    @property
    def url(self) -> str:
        return (
            "https://raw.githubusercontent.com/"
            "The-Atticus-Project/maud-extraction/"
            f"{SOURCE_COMMIT}/{self.relative_path}"
        )

    @property
    def local_name(self) -> str:
        return f"{self.split}.json"


SOURCES = (
    FrozenSource(
        "train",
        "maud_data/maud_squad_split_answers/maud_squad_train.json",
        49_039_965,
        "0f5d178b1b6e0850d5d1b63bb031006cd546291b",
    ),
    FrozenSource(
        "dev",
        "maud_data/maud_squad_split_answers/maud_squad_dev.json",
        6_133_011,
        "99d974c96dc97f9d3fb3262a120a71f643b91d8d",
    ),
    FrozenSource(
        "test",
        "maud_data/maud_squad_split_answers/maud_squad_test.json",
        6_169_945,
        "54a85610a480f121a9d308b293bd46fcafc9df86",
    ),
)


class MaudDownloadError(RuntimeError):
    """The one-shot byte acquisition contract failed closed."""


class _NoRedirect(HTTPRedirectHandler):
    """Turn every 3xx into a terminal HTTPError without a follow-up GET."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


_NO_REDIRECT_OPENER = build_opener(_NoRedirect())


def canonical_bytes(value: object) -> bytes:
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
        raise MaudDownloadError("receipt is not canonical JSON") from exc


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value).rstrip(b"\n")).hexdigest()


def self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in body:
        raise MaudDownloadError("body already contains self_sha256")
    result = dict(body)
    result["self_sha256"] = semantic_sha256(result)
    return result


def write_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    raw = canonical_bytes(payload)
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(fd, "wb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def git_blob_sha1(size: int, digest: "hashlib._Hash") -> str:  # type: ignore[name-defined]
    """Return a blob hash from a digest that already contains the Git header."""

    if size < 0:
        raise MaudDownloadError("negative source size")
    return digest.hexdigest()


def stream_frozen_source(
    stream: BinaryIO,
    destination: Path,
    frozen: FrozenSource,
) -> dict[str, object]:
    """Write and validate one exact source stream without decoding it."""

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    sha256 = hashlib.sha256()
    blob = hashlib.sha1()
    blob.update(f"blob {frozen.size_bytes}\0".encode("ascii"))
    size = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            while True:
                chunk = stream.read(READ_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, bytes):
                    raise MaudDownloadError("source stream returned non-bytes")
                size += len(chunk)
                if size > frozen.size_bytes:
                    raise MaudDownloadError("source exceeded frozen size")
                sha256.update(chunk)
                blob.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
            metadata = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise MaudDownloadError("source file mode drifted")
    except BaseException:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    if size != frozen.size_bytes:
        raise MaudDownloadError("source size does not match frozen metadata")
    observed_blob = git_blob_sha1(size, blob)
    if observed_blob != frozen.git_blob_sha1:
        raise MaudDownloadError("source Git blob identity drifted")
    return {
        "split": frozen.split,
        "official_relative_path": frozen.relative_path,
        "local_name": frozen.local_name,
        "size_bytes": size,
        "git_blob_sha1": observed_blob,
        "sha256": sha256.hexdigest(),
    }


def default_opener(url: str) -> ContextManager[BinaryIO]:
    request = Request(
        url,
        method="GET",
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": VERSION,
        },
    )
    return _NO_REDIRECT_OPENER.open(request, timeout=TIMEOUT_SECONDS)


def download_pinned_sources(
    output_root: str | Path,
    *,
    opener: Callable[[str], ContextManager[BinaryIO]] = default_opener,
) -> dict[str, object]:
    """Consume the one permitted three-file source acquisition attempt."""

    root = Path(output_root)
    if root.exists() or root.is_symlink():
        raise MaudDownloadError("download root is already consumed")
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.mkdir(mode=0o700)
    write_exclusive(
        root / "download.attempt.json",
        self_hashed(
            {
                "schema": f"{VERSION}_attempt_v1",
                "study_id": STUDY_ID,
                "source_custody_self_sha256": SOURCE_CUSTODY_SELF_SHA256,
                "retry_resume_or_mirror_switch_count": 0,
            }
        ),
    )
    rows: list[dict[str, object]] = []
    try:
        for frozen in SOURCES:
            with opener(frozen.url) as response:
                final_url = getattr(response, "geturl", lambda: frozen.url)()
                if final_url != frozen.url:
                    raise MaudDownloadError("source URL redirected")
                rows.append(
                    stream_frozen_source(
                        response,
                        root / "source_bytes" / frozen.local_name,
                        frozen,
                    )
                )
        receipt = self_hashed(
            {
                "schema": f"{VERSION}_receipt_v1",
                "version": VERSION,
                "study_id": STUDY_ID,
                "status": "three_frozen_byte_streams_downloaded_not_JSON_parsed",
                "source_commit": SOURCE_COMMIT,
                "source_custody_self_sha256": SOURCE_CUSTODY_SELF_SHA256,
                "files": rows,
                "file_count": len(rows),
                "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
                "GET_count": len(rows),
                "JSON_parse_or_row_open_count": 0,
                "retry_resume_or_mirror_switch_count": 0,
                "online_evaluator_or_model_call_count": 0,
            }
        )
        write_exclusive(root / "download.receipt.json", receipt)
        return receipt
    except BaseException as exc:
        failure = self_hashed(
            {
                "schema": f"{VERSION}_terminal_failure_v1",
                "version": VERSION,
                "study_id": STUDY_ID,
                "status": "terminal_no_retry_resume_or_mirror_switch",
                "completed_file_count": len(rows),
                "error_type": type(exc).__name__,
                "source_content_included": False,
            }
        )
        try:
            write_exclusive(root / "download.terminal.json", failure)
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt = download_pinned_sources(args.output_root)
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
