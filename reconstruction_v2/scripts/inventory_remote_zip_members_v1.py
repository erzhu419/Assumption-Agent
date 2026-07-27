#!/usr/bin/env python3
"""Inventory selected members of a remote ZIP without opening member payloads.

The caller supplies an already frozen byte length and HTTP validators.  A
single suffix Range GET must contain the EOCD, any ZIP64 EOCD, and the complete
central directory.  The script emits aggregate topology plus bindings for only
the explicitly named members.  It never requests a local header or compressed
member bytes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import PurePosixPath
import struct
import sys
from typing import Any
from urllib.request import Request, urlopen


VERSION = "inventory_remote_zip_members_v1"
EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
CENTRAL_SIGNATURE = b"PK\x01\x02"
MAX_TAIL_BYTES = 16 * 1024 * 1024


class InventoryError(RuntimeError):
    """Remote archive topology failed closed."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalize_etag(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().strip('"')


def _range_get(url: str, start: int, end: int) -> tuple[bytes, Any]:
    request = Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{end}",
            "User-Agent": f"{VERSION}/1",
        },
        method="GET",
    )
    with urlopen(request, timeout=120) as response:
        status = getattr(response, "status", None)
        if status != 206:
            raise InventoryError(f"range status is {status}, expected 206")
        content_range = response.headers.get("Content-Range")
        expected_prefix = f"bytes {start}-{end}/"
        if not content_range or not content_range.startswith(expected_prefix):
            raise InventoryError("Content-Range does not match request")
        payload = response.read()
        if len(payload) != end - start + 1:
            raise InventoryError("range response byte count drifted")
        return payload, response.headers


def _zip64_extra_values(
    extra: bytes,
    *,
    uncompressed_size: int,
    compressed_size: int,
    local_header_offset: int,
    disk_start: int,
) -> tuple[int, int, int, int]:
    offset = 0
    while offset + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, offset)
        offset += 4
        data = extra[offset : offset + data_size]
        if len(data) != data_size:
            raise InventoryError("truncated central-directory extra field")
        offset += data_size
        if header_id != 0x0001:
            continue
        cursor = 0

        def take_u64() -> int:
            nonlocal cursor
            if cursor + 8 > len(data):
                raise InventoryError("truncated ZIP64 u64 field")
            value = struct.unpack_from("<Q", data, cursor)[0]
            cursor += 8
            return value

        def take_u32() -> int:
            nonlocal cursor
            if cursor + 4 > len(data):
                raise InventoryError("truncated ZIP64 u32 field")
            value = struct.unpack_from("<L", data, cursor)[0]
            cursor += 4
            return value

        if uncompressed_size == 0xFFFFFFFF:
            uncompressed_size = take_u64()
        if compressed_size == 0xFFFFFFFF:
            compressed_size = take_u64()
        if local_header_offset == 0xFFFFFFFF:
            local_header_offset = take_u64()
        if disk_start == 0xFFFF:
            disk_start = take_u32()
        break
    if (
        uncompressed_size == 0xFFFFFFFF
        or compressed_size == 0xFFFFFFFF
        or local_header_offset == 0xFFFFFFFF
        or disk_start == 0xFFFF
    ):
        raise InventoryError("required ZIP64 extra field is absent")
    return (
        uncompressed_size,
        compressed_size,
        local_header_offset,
        disk_start,
    )


def _parse_central_directory(
    central: bytes,
) -> tuple[list[dict[str, Any]], int]:
    entries: list[dict[str, Any]] = []
    offset = 0
    while offset < len(central):
        if offset + 46 > len(central):
            raise InventoryError("truncated central-directory header")
        fields = struct.unpack_from("<4s6H3L5H2L", central, offset)
        if fields[0] != CENTRAL_SIGNATURE:
            raise InventoryError("central-directory signature drifted")
        (
            _signature,
            version_made,
            version_needed,
            flags,
            method,
            modified_time,
            modified_date,
            crc32,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
            comment_length,
            disk_start,
            internal_attributes,
            external_attributes,
            local_header_offset,
        ) = fields
        variable_start = offset + 46
        variable_end = (
            variable_start + name_length + extra_length + comment_length
        )
        if variable_end > len(central):
            raise InventoryError("central-directory variable fields truncated")
        raw_name = central[variable_start : variable_start + name_length]
        extra_start = variable_start + name_length
        extra = central[extra_start : extra_start + extra_length]
        encoding = "utf-8" if flags & 0x0800 else "cp437"
        try:
            name = raw_name.decode(encoding)
        except UnicodeDecodeError as exc:
            raise InventoryError("member name decode failed") from exc
        (
            uncompressed_size,
            compressed_size,
            local_header_offset,
            disk_start,
        ) = _zip64_extra_values(
            extra,
            uncompressed_size=uncompressed_size,
            compressed_size=compressed_size,
            local_header_offset=local_header_offset,
            disk_start=disk_start,
        )
        if disk_start != 0:
            raise InventoryError("multi-disk ZIP is not authorized")
        entries.append(
            {
                "compressed_bytes": compressed_size,
                "compression_method": method,
                "crc32": f"{crc32:08x}",
                "external_attributes": external_attributes,
                "flags": flags,
                "internal_attributes": internal_attributes,
                "local_header_offset": local_header_offset,
                "modified_date": modified_date,
                "modified_time": modified_time,
                "name": name,
                "uncompressed_bytes": uncompressed_size,
                "version_made": version_made,
                "version_needed": version_needed,
            }
        )
        offset = variable_end
    return entries, offset


def _is_unsafe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        path.is_absolute()
        or "\x00" in name
        or "\\" in name
        or any(part in {"", ".", ".."} for part in path.parts)
    )


def inventory(
    *,
    url: str,
    expected_size: int,
    expected_etag: str,
    expected_last_modified: str,
    selected_members: list[str],
) -> dict[str, Any]:
    if expected_size <= 0:
        raise InventoryError("expected size must be positive")
    if not selected_members or len(set(selected_members)) != len(selected_members):
        raise InventoryError("selected members must be unique and nonempty")
    tail_bytes = min(expected_size, MAX_TAIL_BYTES)
    tail_start = expected_size - tail_bytes
    tail, headers = _range_get(url, tail_start, expected_size - 1)
    if _normalize_etag(headers.get("ETag")) != _normalize_etag(expected_etag):
        raise InventoryError("ETag drifted")
    if headers.get("Last-Modified", "").strip() != expected_last_modified:
        raise InventoryError("Last-Modified drifted")

    eocd_offset_in_tail = tail.rfind(EOCD_SIGNATURE)
    if eocd_offset_in_tail < 0 or eocd_offset_in_tail + 22 > len(tail):
        raise InventoryError("EOCD is absent")
    (
        _signature,
        disk_number,
        central_disk,
        entries_on_disk,
        total_entries,
        central_size,
        central_offset,
        comment_length,
    ) = struct.unpack_from("<4s4H2LH", tail, eocd_offset_in_tail)
    if disk_number != 0 or central_disk != 0:
        raise InventoryError("multi-disk ZIP is not authorized")
    if eocd_offset_in_tail + 22 + comment_length != len(tail):
        raise InventoryError("EOCD comment or suffix length drifted")

    zip64 = (
        entries_on_disk == 0xFFFF
        or total_entries == 0xFFFF
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
    )
    if zip64:
        locator_position = eocd_offset_in_tail - 20
        if locator_position < 0:
            raise InventoryError("ZIP64 locator is absent from suffix")
        locator = struct.unpack_from("<4sLQL", tail, locator_position)
        if locator[0] != ZIP64_LOCATOR_SIGNATURE:
            raise InventoryError("ZIP64 locator signature drifted")
        if locator[1] != 0 or locator[3] != 1:
            raise InventoryError("multi-disk ZIP64 is not authorized")
        zip64_offset = locator[2]
        zip64_position = zip64_offset - tail_start
        if zip64_position < 0 or zip64_position + 56 > len(tail):
            raise InventoryError("ZIP64 EOCD is outside the single suffix GET")
        zip64_fields = struct.unpack_from(
            "<4sQ2H2L4Q", tail, zip64_position
        )
        if zip64_fields[0] != ZIP64_EOCD_SIGNATURE:
            raise InventoryError("ZIP64 EOCD signature drifted")
        if zip64_fields[4] != 0 or zip64_fields[5] != 0:
            raise InventoryError("multi-disk ZIP64 is not authorized")
        entries_on_disk = zip64_fields[6]
        total_entries = zip64_fields[7]
        central_size = zip64_fields[8]
        central_offset = zip64_fields[9]

    if entries_on_disk != total_entries:
        raise InventoryError("central-directory entry count is split")
    central_start_in_tail = central_offset - tail_start
    central_end_in_tail = central_start_in_tail + central_size
    if (
        central_start_in_tail < 0
        or central_end_in_tail > len(tail)
        or central_end_in_tail > eocd_offset_in_tail
    ):
        raise InventoryError(
            "complete central directory is outside the single suffix GET"
        )
    central = tail[central_start_in_tail:central_end_in_tail]
    entries, parsed_bytes = _parse_central_directory(central)
    if parsed_bytes != central_size or len(entries) != total_entries:
        raise InventoryError("central-directory totals drifted")

    names = [entry["name"] for entry in entries]
    counts = Counter(names)
    selected: dict[str, Any] = {}
    by_name = {entry["name"]: entry for entry in entries}
    for name in selected_members:
        if counts[name] != 1:
            raise InventoryError(f"selected member count is not one: {name}")
        selected[name] = by_name[name]
    aggregate = {
        "archive_byte_count": expected_size,
        "archive_url": url,
        "central_directory_byte_count": central_size,
        "central_directory_offset": central_offset,
        "central_directory_sha256": _sha256(central),
        "duplicate_member_name_count": sum(
            count - 1 for count in counts.values() if count > 1
        ),
        "entry_count": len(entries),
        "etag": _normalize_etag(headers.get("ETag")),
        "last_modified": headers.get("Last-Modified", "").strip(),
        "selected_members": selected,
        "suffix_range_GET": {
            "byte_count": len(tail),
            "end": expected_size - 1,
            "start": tail_start,
        },
        "unsafe_member_name_count": sum(
            _is_unsafe_member(name) for name in names
        ),
        "zip64": zip64,
    }
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--expected-size", required=True, type=int)
    parser.add_argument("--expected-etag", required=True)
    parser.add_argument("--expected-last-modified", required=True)
    parser.add_argument("--member", action="append", required=True)
    args = parser.parse_args()
    try:
        result = inventory(
            url=args.url,
            expected_size=args.expected_size,
            expected_etag=args.expected_etag,
            expected_last_modified=args.expected_last_modified,
            selected_members=args.member,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "schema": VERSION,
                    "status": "failed_closed",
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 1
    envelope = {
        "result": result,
        "schema": VERSION,
        "status": "central_directory_inventory_complete",
    }
    envelope["self_sha256"] = _sha256(_canonical_bytes(envelope))
    sys.stdout.buffer.write(_canonical_bytes(envelope) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
