"""Create and verify the local WikiSQL source-custody commitment.

This command is run on the local acquisition machine after the implementation
freeze and before transport to 311linux.  It is the only pre-formal component
that reads the downloaded archive.  The formal deployment builder consumes
only this content-free commitment plus remote file metadata; the remote source
payload remains unread until the durable formal attempt and live receipts
exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Mapping, Sequence

from replication_runtime.wikisql_uao_formal_v5 import runner as formal


EXPECTED_SOURCE_BYTES = 26164664
EXPECTED_SOURCE_GIT_BLOB_SHA1 = (
    "941de4cb2ad5fa7aeb2e37d314468636ce070af7"
)
OFFICIAL_REPOSITORY_COMMIT = (
    "a9e07caff1472ed242bf101c0b6fc6cd5a6fbabf"
)
CUSTODY_SCHEMA = "wikisql_uao_p4_source_custody_v1"
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_CUSTODY_KEYS = frozenset(
    {
        "API_or_online_evaluation_count",
        "archive_git_blob_sha1",
        "archive_sha256",
        "archive_size_bytes",
        "formal_source_access_count",
        "formal_source_member_open_count",
        "local_acquisition_archive_read_count",
        "official_repository_commit",
        "schema",
        "self_sha256",
        "source_payload_read_context",
        "study_id",
    }
)


class WikiSQLUAOSourceCustodyError(RuntimeError):
    """The official archive or its content-free custody receipt drifted."""


def _direct_file(path: Path, field: str) -> os.stat_result:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise WikiSQLUAOSourceCustodyError(
            f"{field} is unavailable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or resolved != path
    ):
        raise WikiSQLUAOSourceCustodyError(
            f"{field} is not a direct canonical file"
        )
    return metadata


def _archive_identity(path: Path) -> tuple[str, int, str]:
    metadata = _direct_file(path, "local WikiSQL acquisition archive")
    digest_sha256 = hashlib.sha256()
    digest_sha1 = hashlib.sha1()
    digest_sha1.update(f"blob {metadata.st_size}\0".encode("ascii"))
    observed = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                observed += len(chunk)
                digest_sha256.update(chunk)
                digest_sha1.update(chunk)
    except OSError as exc:
        raise WikiSQLUAOSourceCustodyError(
            "local WikiSQL acquisition archive cannot be addressed"
        ) from exc
    git_blob_sha1 = digest_sha1.hexdigest()
    if (
        observed != metadata.st_size
        or observed != EXPECTED_SOURCE_BYTES
        or git_blob_sha1 != EXPECTED_SOURCE_GIT_BLOB_SHA1
    ):
        raise WikiSQLUAOSourceCustodyError(
            "official WikiSQL source identity drifted"
        )
    return digest_sha256.hexdigest(), observed, git_blob_sha1


def create_receipt(source_archive: Path) -> Mapping[str, object]:
    archive_sha256, size, git_blob_sha1 = _archive_identity(
        source_archive
    )
    return formal._self_hashed(
        {
            "API_or_online_evaluation_count": 0,
            "archive_git_blob_sha1": git_blob_sha1,
            "archive_sha256": archive_sha256,
            "archive_size_bytes": size,
            "formal_source_access_count": 0,
            "formal_source_member_open_count": 0,
            "local_acquisition_archive_read_count": 1,
            "official_repository_commit": OFFICIAL_REPOSITORY_COMMIT,
            "schema": CUSTODY_SCHEMA,
            "source_payload_read_context": (
                "local_acquisition_only_not_formal_runtime"
            ),
            "study_id": formal.STUDY_ID,
        }
    )


def load_receipt(path: Path) -> Mapping[str, object]:
    metadata = _direct_file(path, "source custody receipt")
    if stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}:
        raise WikiSQLUAOSourceCustodyError(
            "source custody receipt mode drifted"
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WikiSQLUAOSourceCustodyError(
            "source custody receipt is malformed"
        ) from exc
    if (
        type(value) is not dict
        or set(value) != _CUSTODY_KEYS
        or formal.canonical_json_bytes(value) != raw
    ):
        raise WikiSQLUAOSourceCustodyError(
            "source custody receipt shape drifted"
        )
    supplied = value["self_sha256"]
    body = {
        key: child
        for key, child in value.items()
        if key != "self_sha256"
    }
    if (
        not isinstance(supplied, str)
        or _HEX64.fullmatch(supplied) is None
        or formal.semantic_sha256(body) != supplied
        or value["schema"] != CUSTODY_SCHEMA
        or value["study_id"] != formal.STUDY_ID
        or value["official_repository_commit"]
        != OFFICIAL_REPOSITORY_COMMIT
        or value["archive_size_bytes"] != EXPECTED_SOURCE_BYTES
        or value["archive_git_blob_sha1"]
        != EXPECTED_SOURCE_GIT_BLOB_SHA1
        or not isinstance(value["archive_sha256"], str)
        or _HEX64.fullmatch(value["archive_sha256"]) is None
        or not isinstance(value["archive_git_blob_sha1"], str)
        or _HEX40.fullmatch(value["archive_git_blob_sha1"]) is None
        or value["API_or_online_evaluation_count"] != 0
        or value["formal_source_access_count"] != 0
        or value["formal_source_member_open_count"] != 0
        or value["local_acquisition_archive_read_count"] != 1
        or value["source_payload_read_context"]
        != "local_acquisition_only_not_formal_runtime"
    ):
        raise WikiSQLUAOSourceCustodyError(
            "source custody receipt identity drifted"
        )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    receipt = create_receipt(arguments.source_archive)
    formal._write_once(arguments.output, receipt, mode=0o600)
    print(
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
