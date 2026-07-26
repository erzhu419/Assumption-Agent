from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import io
import json
import stat
from urllib.error import HTTPError

import pytest

from assumption_agent.benchmarks import maud_extraction_p2_download_v1 as subject


def _frozen(split: str, payload: bytes) -> subject.FrozenSource:
    blob = hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()
    return subject.FrozenSource(split, f"data/{split}.json", len(payload), blob)


def test_stream_frozen_source_binds_bytes_and_mode(tmp_path):
    payload = b'{"data":[]}\n'
    frozen = _frozen("train", payload)
    destination = tmp_path / "train.json"
    row = subject.stream_frozen_source(io.BytesIO(payload), destination, frozen)
    assert row["git_blob_sha1"] == frozen.git_blob_sha1
    assert row["sha256"] == hashlib.sha256(payload).hexdigest()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_stream_rejects_size_or_blob_drift(tmp_path):
    payload = b"abc"
    with pytest.raises(subject.MaudDownloadError):
        subject.stream_frozen_source(
            io.BytesIO(payload),
            tmp_path / "bad-size",
            replace(_frozen("dev", payload), size_bytes=4),
        )
    with pytest.raises(subject.MaudDownloadError):
        subject.stream_frozen_source(
            io.BytesIO(payload),
            tmp_path / "bad-blob",
            replace(_frozen("dev", payload), git_blob_sha1="0" * 40),
        )


def test_download_is_one_shot_and_never_parses_json(tmp_path, monkeypatch):
    payloads = {
        "train": b"not-json-train",
        "dev": b"not-json-dev",
        "test": b"not-json-test",
    }
    frozen = tuple(_frozen(split, payload) for split, payload in payloads.items())
    monkeypatch.setattr(subject, "SOURCES", frozen)

    @contextmanager
    def opener(url):
        row = next(item for item in frozen if item.url == url)
        yield io.BytesIO(payloads[row.split])

    root = tmp_path / "source"
    receipt = subject.download_pinned_sources(root, opener=opener)
    assert receipt["GET_count"] == 3
    assert receipt["JSON_parse_or_row_open_count"] == 0
    assert receipt["total_size_bytes"] == sum(map(len, payloads.values()))
    persisted = json.loads((root / "download.receipt.json").read_text("ascii"))
    assert persisted == receipt
    with pytest.raises(subject.MaudDownloadError):
        subject.download_pinned_sources(root, opener=opener)


def test_partial_failure_is_terminal_and_preserved(tmp_path, monkeypatch):
    payloads = {"train": b"a", "dev": b"b", "test": b"c"}
    frozen = tuple(_frozen(split, payload) for split, payload in payloads.items())
    monkeypatch.setattr(subject, "SOURCES", frozen)
    count = 0

    @contextmanager
    def opener(url):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("offline")
        row = next(item for item in frozen if item.url == url)
        yield io.BytesIO(payloads[row.split])

    root = tmp_path / "source"
    with pytest.raises(OSError):
        subject.download_pinned_sources(root, opener=opener)
    failure = json.loads((root / "download.terminal.json").read_text("ascii"))
    assert failure["completed_file_count"] == 1
    assert failure["source_content_included"] is False


def test_default_opener_disables_redirect_before_any_followup_get(monkeypatch):
    calls = []

    class FakeNoRedirectOpener:
        def open(self, request, *, timeout):
            calls.append((request.full_url, timeout))
            raise HTTPError(
                request.full_url,
                302,
                "redirect refused",
                {"Location": "https://example.invalid/second"},
                None,
            )

    monkeypatch.setattr(subject, "_NO_REDIRECT_OPENER", FakeNoRedirectOpener())
    with pytest.raises(HTTPError):
        subject.default_opener("https://example.invalid/first")
    assert calls == [
        ("https://example.invalid/first", subject.TIMEOUT_SECONDS)
    ]
