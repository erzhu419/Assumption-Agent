from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import shutil
import tempfile

import pytest

from assumption_agent.benchmarks import tatqa_p21_source_download_v1 as source


def _qualified_freeze(_project: Path) -> dict[str, str]:
    return {"self_sha256": "f" * 64}


class _Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


@pytest.fixture
def linux_tmp() -> Path:
    root = Path(tempfile.mkdtemp(prefix="tatqa-p18-source-", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root)


def test_stream_response_is_opaque_exclusive_and_bounded(linux_tmp: Path) -> None:
    destination = linux_tmp / "nested" / "payload.json"
    binding = source.stream_response_exclusive(_Response(b"not parsed\xff\x00"), destination)
    assert destination.read_bytes() == b"not parsed\xff\x00"
    assert binding["size_bytes"] == len(destination.read_bytes())
    assert len(binding["sha256"]) == 64
    assert oct(destination.stat().st_mode & 0o777) == "0o600"
    with pytest.raises(FileExistsError):
        source.stream_response_exclusive(_Response(b"second"), destination)
    with pytest.raises(source.TatqaP21SourceDownloadError, match="exceeded"):
        source.stream_response_exclusive(
            _Response(b"oversized"), linux_tmp / "too-large", maximum_bytes=3
        )


def test_download_writes_exact_registry_and_never_parses_payload(linux_tmp: Path) -> None:
    project = linux_tmp / "project"
    project.mkdir()
    calls: list[str] = []

    def opener(url: str):
        calls.append(url)
        relative = url.split(source.SOURCE_COMMIT + "/", 1)[1]
        return _Response(("opaque:" + relative).encode("utf-8") + b"\xff")

    receipt = source.download_pinned_source(
        project, opener=opener, freeze_verifier=_qualified_freeze
    )
    assert tuple(row["relative_path"] for row in receipt["files"]) == source.SOURCE_FILES
    assert receipt["dataset_payload_decode_or_row_parse_count"] == 0
    assert receipt["test_split_request_count"] == 0
    assert len(calls) == len(source.SOURCE_FILES)
    assert all(source.SOURCE_COMMIT in url for url in calls)
    stored = json.loads((project / source.RECEIPT_RELATIVE).read_text("ascii"))
    assert stored == receipt
    assert all(
        (project / source.SOURCE_ROOT_RELATIVE / relative).is_file()
        for relative in source.SOURCE_FILES
    )
    with pytest.raises(source.TatqaP21SourceDownloadError, match="consumed"):
        source.download_pinned_source(
            project, opener=opener, freeze_verifier=_qualified_freeze
        )


def test_partial_failure_burns_root_and_does_not_retry(linux_tmp: Path) -> None:
    project = linux_tmp / "project"
    project.mkdir()
    count = 0

    def opener(_url: str):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("offline")
        return _Response(b"first")

    with pytest.raises(OSError):
        source.download_pinned_source(
            project, opener=opener, freeze_verifier=_qualified_freeze
        )
    assert count == 2
    failure = (
        project
        / source.SOURCE_ROOT_RELATIVE.parent
        / "source.download.terminal_failure.json"
    )
    assert failure.is_file()
    with pytest.raises(source.TatqaP21SourceDownloadError, match="consumed"):
        source.download_pinned_source(
            project, opener=opener, freeze_verifier=_qualified_freeze
        )


def test_unqualified_freeze_fails_before_download_root_or_network(
    linux_tmp: Path,
) -> None:
    project = linux_tmp / "project"
    project.mkdir()
    calls = 0

    def opener(_url: str):
        nonlocal calls
        calls += 1
        return _Response(b"must not be opened")

    def reject(_project: Path):
        raise source.TatqaP21SourceDownloadError("freeze invalid")

    with pytest.raises(source.TatqaP21SourceDownloadError, match="freeze"):
        source.download_pinned_source(
            project, opener=opener, freeze_verifier=reject
        )
    assert calls == 0
    assert not (project / source.SOURCE_ROOT_RELATIVE.parent).exists()
