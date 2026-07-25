from __future__ import annotations

from dataclasses import replace
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import stat
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from assumption_agent.benchmarks import mmqa_p1_source_acquisition_v1 as a
from assumption_agent.benchmarks import mmqa_p1_source_qualification_v1 as q


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Response(BytesIO):
    def __init__(
        self,
        body: bytes,
        url: str,
        *,
        content_length: int | None = None,
        content_encoding: str = "identity",
        status: int = 200,
    ) -> None:
        super().__init__(body)
        self.status = status
        self.headers = {
            "Content-Length": str(
                len(body) if content_length is None else content_length
            ),
            "Content-Encoding": content_encoding,
        }
        self._url = url

    def geturl(self) -> str:
        return self._url


def _payloads() -> dict[str, bytes]:
    return {
        file_name: (
            b"SYNTHETIC_OPAQUE_SECRET_BYTES_"
            + file_name.encode("ascii")
            + b"_"
            + bytes([ordinal]) * (ordinal + 3)
        )
        for ordinal, file_name in enumerate(a.FILE_ORDER, start=1)
    }


def _plan(
    payloads: dict[str, bytes] | None = None,
) -> a.DownloadPlan:
    bodies = payloads or _payloads()
    files = []
    for file_name in a.FILE_ORDER:
        body = bodies[file_name]
        files.append(
            a.DownloadFile(
                file_name=file_name,
                url=(
                    "https://raw.githubusercontent.com/allenai/"
                    f"multimodalqa/{a.SOURCE_COMMIT}/dataset/{file_name}"
                ),
                expected_size_bytes=len(body),
                expected_git_blob_sha1=a._git_blob_sha1_for_payload(body),
            )
        )
    return a.DownloadPlan(
        files=tuple(files),
        authorization_self_sha256="1" * 64,
        source_custody_self_sha256="2" * 64,
        study_design_self_sha256="3" * 64,
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("ascii"))
    assert isinstance(value, dict)
    return value


def _assert_valid_self_hash(value: dict[str, object]) -> None:
    body = dict(value)
    self_hash = body.pop("self_sha256")
    assert self_hash == a._semantic_hash(body)


def test_committed_authorization_loads_exact_four_file_plan_read_only() -> None:
    plan = a._load_authorized_plan(PROJECT_ROOT)
    assert tuple(file.file_name for file in plan.files) == a.FILE_ORDER
    assert plan.source_root_relative == a.SOURCE_ROOT_RELATIVE
    assert {
        file.file_name: (
            file.url,
            file.expected_size_bytes,
            file.expected_git_blob_sha1,
        )
        for file in plan.files
    } == {
        file_name: (
            expected.url,
            expected.expected_size_bytes,
            expected.expected_git_blob_sha1,
        )
        for file_name, expected in a.FORMAL_FILES.items()
    }
    assert (
        q.SOURCE_ROOT.relative_to(q.PROJECT_ROOT)
        == a.SOURCE_ROOT_RELATIVE
    )


def test_success_streams_each_file_once_and_emits_aggregate_only_receipt(
    tmp_path: Path,
) -> None:
    payloads = _payloads()
    plan = _plan(payloads)
    calls: list[str] = []

    def opener(url: str) -> _Response:
        calls.append(url)
        file_name = Path(url.split("?", 1)[0]).name
        ordinal = len(calls)
        assert (
            tmp_path / a.ATTEMPT_MARKER_RELATIVE
        ).is_file(), "global marker must precede network"
        assert (
            tmp_path
            / a.FILE_ATTEMPT_ROOT_RELATIVE
            / f"{ordinal:02d}.{file_name}.attempt.json"
        ).is_file(), "per-file marker must immediately precede network"
        return _Response(payloads[file_name], url)

    receipt = dict(a.acquire_plan_once(tmp_path, plan, opener=opener))

    assert calls == [file.url for file in plan.files]
    assert len(set(calls)) == len(plan.files) == 4
    assert receipt["completed_file_count"] == 4
    assert receipt["network_attempt_count"] == 4
    assert receipt["network_attempt_count_per_file_maximum"] == 1
    assert (
        receipt["retry_resume_range_mirror_or_provider_switch_count"] == 0
    )
    assert (
        receipt["dataset_byte_decode_decompress_JSONL_or_row_parse_count"]
        == 0
    )
    assert receipt["nonmatching_host_redirect_count"] == 0
    assert receipt["response_body_or_URL_query_output_count"] == 0
    _assert_valid_self_hash(receipt)

    source_root = tmp_path / a.SOURCE_ROOT_RELATIVE
    for file in plan.files:
        destination = source_root / file.file_name
        assert destination.read_bytes() == payloads[file.file_name]
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600
        assert not (
            source_root / f".{file.file_name}.one_shot.part"
        ).exists()
    assert stat.S_IMODE(source_root.stat().st_mode) == 0o700

    receipt_path = tmp_path / a.RECEIPT_RELATIVE
    persisted = _read_json(receipt_path)
    assert persisted == receipt
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert len(
        list((tmp_path / a.FILE_ATTEMPT_ROOT_RELATIVE).glob("*.json"))
    ) == 4

    all_metadata = b"\n".join(
        path.read_bytes() for path in tmp_path.rglob("*.json")
    )
    for body in payloads.values():
        assert body not in all_metadata
    for file in plan.files:
        assert file.url.encode("ascii") not in all_metadata

    with pytest.raises(
        a.MMQAP1SourceAcquisitionError, match="already consumed"
    ):
        a.acquire_plan_once(tmp_path, plan, opener=opener)
    assert calls == [file.url for file in plan.files]


def test_network_failure_is_terminal_sanitized_and_not_retried(
    tmp_path: Path,
) -> None:
    payloads = _payloads()
    plan = _plan(payloads)
    calls: list[str] = []
    leaked_diagnostic = (
        "SECRET_RESPONSE_BODY https://forbidden.example/private?token=SECRET"
    )

    def opener(url: str) -> _Response:
        calls.append(url)
        if len(calls) == 2:
            raise OSError(leaked_diagnostic)
        file_name = Path(url).name
        return _Response(payloads[file_name], url)

    with pytest.raises(
        a.MMQAP1SourceAcquisitionError,
        match="formal source acquisition failed closed",
    ) as caught:
        a.acquire_plan_once(tmp_path, plan, opener=opener)
    assert leaked_diagnostic not in str(caught.value)
    assert calls == [file.url for file in plan.files[:2]]

    failure_path = tmp_path / a.FAILURE_RECEIPT_RELATIVE
    failure_raw = failure_path.read_bytes()
    failure = _read_json(failure_path)
    _assert_valid_self_hash(failure)
    assert failure["attempted_file_count"] == 2
    assert failure["completed_file_count"] == 1
    assert failure["status"] == "terminal_failure_attempt_consumed_no_retry"
    assert failure["response_body_URL_or_URL_query_included"] is False
    assert leaked_diagnostic.encode("ascii") not in failure_raw
    assert b"forbidden.example" not in failure_raw
    assert b"token=SECRET" not in failure_raw
    assert stat.S_IMODE(failure_path.stat().st_mode) == 0o600
    assert (tmp_path / a.ATTEMPT_MARKER_RELATIVE).is_file()
    assert not (tmp_path / a.RECEIPT_RELATIVE).exists()

    before = list(calls)
    with pytest.raises(
        a.MMQAP1SourceAcquisitionError, match="already consumed"
    ):
        a.acquire_plan_once(tmp_path, plan, opener=opener)
    assert calls == before


def test_short_body_consumes_attempt_and_removes_unverified_part(
    tmp_path: Path,
) -> None:
    payloads = _payloads()
    plan = _plan(payloads)
    first = plan.files[0]
    calls = 0

    def opener(url: str) -> _Response:
        nonlocal calls
        calls += 1
        assert url == first.url
        return _Response(
            payloads[first.file_name][:-1],
            url,
            content_length=first.expected_size_bytes,
        )

    with pytest.raises(a.MMQAP1SourceAcquisitionError):
        a.acquire_plan_once(tmp_path, plan, opener=opener)
    assert calls == 1
    source_root = tmp_path / a.SOURCE_ROOT_RELATIVE
    assert not (source_root / first.file_name).exists()
    assert not (
        source_root / f".{first.file_name}.one_shot.part"
    ).exists()
    failure = _read_json(tmp_path / a.FAILURE_RECEIPT_RELATIVE)
    assert failure["failure_stage"] == "stream_and_verify_opaque_bytes"
    assert failure["attempted_file_count"] == 1


def test_wrong_git_blob_identity_consumes_attempt_without_promotion(
    tmp_path: Path,
) -> None:
    payloads = _payloads()
    original = _plan(payloads)
    first = replace(original.files[0], expected_git_blob_sha1="0" * 40)
    plan = replace(original, files=(first,) + original.files[1:])
    calls: list[str] = []

    def opener(url: str) -> _Response:
        calls.append(url)
        return _Response(payloads[Path(url).name], url)

    with pytest.raises(a.MMQAP1SourceAcquisitionError):
        a.acquire_plan_once(tmp_path, plan, opener=opener)
    assert calls == [first.url]
    source_root = tmp_path / a.SOURCE_ROOT_RELATIVE
    assert not (source_root / first.file_name).exists()
    assert not any(source_root.glob("*.part"))


def test_nonmatching_final_redirect_host_fails_before_body_read(
    tmp_path: Path,
) -> None:
    payloads = _payloads()
    plan = _plan(payloads)
    first = plan.files[0]
    read_count = 0

    class CountingResponse(_Response):
        def read(self, size: int = -1) -> bytes:
            nonlocal read_count
            read_count += 1
            return super().read(size)

    def opener(url: str) -> CountingResponse:
        return CountingResponse(
            payloads[first.file_name],
            "https://forbidden.example/dataset/" + first.file_name,
        )

    with pytest.raises(a.MMQAP1SourceAcquisitionError):
        a.acquire_plan_once(tmp_path, plan, opener=opener)
    assert read_count == 0
    failure = _read_json(tmp_path / a.FAILURE_RECEIPT_RELATIVE)
    assert failure["failure_stage"] == "validate_fixed_https_response"
    assert b"forbidden.example" not in (
        tmp_path / a.FAILURE_RECEIPT_RELATIVE
    ).read_bytes()


def test_redirect_handler_rejects_nonmatching_host_without_network() -> None:
    handler = a._SameHostHTTPSRedirectHandler(a.EXPECTED_HOST)
    request = Request(
        "https://raw.githubusercontent.com/allenai/multimodalqa/"
        f"{a.SOURCE_COMMIT}/dataset/{a.FILE_ORDER[0]}"
    )
    with pytest.raises(HTTPError, match="redirect rejected"):
        handler.redirect_request(
            request,
            BytesIO(),
            302,
            "Found",
            {},
            "https://forbidden.example/private?secret=1",
        )


def test_nonbytes_stream_is_terminal_and_sanitized(tmp_path: Path) -> None:
    payloads = _payloads()
    plan = _plan(payloads)
    first = plan.files[0]

    class NonBytesResponse(_Response):
        def read(self, size: int = -1):  # type: ignore[no-untyped-def]
            return "SECRET_NONBYTE_BODY"

    def opener(url: str) -> NonBytesResponse:
        return NonBytesResponse(
            payloads[first.file_name],
            url,
            content_length=first.expected_size_bytes,
        )

    with pytest.raises(a.MMQAP1SourceAcquisitionError):
        a.acquire_plan_once(tmp_path, plan, opener=opener)
    failure_raw = (tmp_path / a.FAILURE_RECEIPT_RELATIVE).read_bytes()
    assert b"SECRET_NONBYTE_BODY" not in failure_raw
    assert not (
        tmp_path
        / a.SOURCE_ROOT_RELATIVE
        / f".{first.file_name}.one_shot.part"
    ).exists()


def test_stream_part_uses_o_excl_and_never_overwrites_existing_file(
    tmp_path: Path,
) -> None:
    payloads = _payloads()
    file = _plan(payloads).files[0]
    part_path = tmp_path / ".existing.part"
    part_path.write_bytes(b"DO_NOT_OVERWRITE")
    os.chmod(part_path, 0o600)
    response = _Response(payloads[file.file_name], file.url)

    with pytest.raises(FileExistsError):
        a._stream_verified_part(response, part_path, file)
    assert part_path.read_bytes() == b"DO_NOT_OVERWRITE"


def test_invalid_plan_is_rejected_before_attempt_or_network(
    tmp_path: Path,
) -> None:
    plan = replace(_plan(), source_root_relative=Path("other/source"))
    calls = 0

    def opener(url: str) -> _Response:
        nonlocal calls
        calls += 1
        raise AssertionError(url)

    with pytest.raises(
        a.MMQAP1SourceAcquisitionError, match="source root contract drifted"
    ):
        a.acquire_plan_once(tmp_path, plan, opener=opener)
    assert calls == 0
    assert not (tmp_path / a.ATTEMPT_MARKER_RELATIVE).exists()


def test_sha256_receipt_matches_streamed_opaque_bytes(tmp_path: Path) -> None:
    payloads = _payloads()
    plan = _plan(payloads)

    def opener(url: str) -> _Response:
        return _Response(payloads[Path(url).name], url)

    receipt = a.acquire_plan_once(tmp_path, plan, opener=opener)
    files = receipt["files"]
    assert isinstance(files, list)
    assert {
        value["file_name"]: value["sha256"]
        for value in files
        if isinstance(value, dict)
    } == {
        file_name: hashlib.sha256(body).hexdigest()
        for file_name, body in payloads.items()
    }
