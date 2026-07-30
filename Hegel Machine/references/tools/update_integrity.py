#!/usr/bin/env python3
"""Refresh artifact integrity fields and the flat checksum ledger."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest.json"
CHECKSUMS = ROOT / "checksums.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(relative_path: str) -> dict[str, object]:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(relative_path)
    media_type, encoding = mimetypes.guess_type(path.name)
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "media_type": media_type or "application/octet-stream",
        "content_encoding": encoding,
    }


def main() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for section in ("references", "repositories"):
        for item in data[section]:
            item["artifacts"] = [describe(path) for path in item.get("files", [])]

    reference_statuses = Counter(item["status"] for item in data["references"])
    repository_statuses = Counter(item["status"] for item in data["repositories"])
    data["summary"] = {
        "reference_records": len(data["references"]),
        "reference_statuses": dict(sorted(reference_statuses.items())),
        "repository_records": len(data["repositories"]),
        "repository_statuses": dict(sorted(repository_statuses.items())),
        "repository_source_archives": sum(
            1
            for item in data["repositories"]
            if any(path.endswith(".tar.gz") for path in item.get("files", []))
        ),
        "valid_pdf_artifacts": len(list((ROOT / "papers").glob("*.pdf"))),
    }
    MANIFEST.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    entries: list[tuple[str, str]] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == CHECKSUMS:
            continue
        relative = path.relative_to(ROOT).as_posix()
        entries.append((sha256(path), relative))
    CHECKSUMS.write_text(
        "".join(f"{digest}  {relative}\n" for digest, relative in entries),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
