from __future__ import annotations

from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

import pytest

from assumption_agent.benchmarks import (
    locomo_p0_public_source_qualification_v1 as source,
)


@pytest.fixture
def linux_tmp() -> Path:
    root = Path(tempfile.mkdtemp(prefix="locomo-p0-", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root)


class _Response(BytesIO):
    def __init__(self, payload: bytes, url: str) -> None:
        super().__init__(payload)
        self._url = url

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _git_blob(raw: bytes) -> str:
    return hashlib.sha1(  # noqa: S324 - Git object identity.
        f"blob {len(raw)}\0".encode("ascii") + raw
    ).hexdigest()


def _conversation(index: int, *, quota: int = 12) -> dict[str, object]:
    speaker_a = f"PRIVATE SPEAKER A {index}"
    speaker_b = f"PRIVATE SPEAKER B {index}"
    turns = [
        {
            "speaker": speaker_a if turn % 2 else speaker_b,
            "dia_id": f"D1:{turn}",
            "text": f"PRIVATE TURN {index} {turn}",
        }
        for turn in range(1, 6)
    ]
    qas: list[dict[str, object]] = []
    for category in source.P1_FAMILY_CATEGORY_IDS:
        for item in range(quota):
            qas.append(
                {
                    "question": f"PRIVATE QUESTION {index} {category} {item}",
                    "answer": f"PRIVATE ANSWER {index} {category} {item}",
                    "category": category,
                    "evidence": ["D1:1", "(D1:2)"],
                }
            )
    qas.append(
        {
            "question": f"PRIVATE OPEN DOMAIN {index}",
            "answer": f"PRIVATE OPEN ANSWER {index}",
            "category": 3,
            "evidence": ["D1:3"],
        }
    )
    qas.append(
        {
            "question": f"PRIVATE ADVERSARIAL {index}",
            "adversarial_answer": "PRIVATE NO INFORMATION",
            "category": 5,
            "evidence": [],
        }
    )
    return {
        "sample_id": f"PRIVATE SAMPLE {index}",
        "conversation": {
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "session_1": turns,
            "session_1_date_time": "PRIVATE DATE",
        },
        "observation": {},
        "session_summary": {},
        "event_summary": {},
        "qa": qas,
    }


def _payloads(*, quota: int = 12) -> dict[str, bytes]:
    return {
        "license": b"PRIVATE SYNTHETIC LICENSE\n",
        "readme": b"PRIVATE SYNTHETIC README\n",
        "data": json.dumps(
            [_conversation(index, quota=quota) for index in range(10)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8"),
    }


def _fixture(
    tmp_path: Path,
    *,
    quota: int = 12,
) -> tuple[
    Path,
    dict[str, source.SourceFileContract],
    dict[str, bytes],
]:
    root = tmp_path / "source"
    root.mkdir(parents=True)
    payloads = _payloads(quota=quota)
    relatives = {
        "license": "LICENSE.txt",
        "readme": "README.MD",
        "data": "data/locomo10.json",
    }
    contracts: dict[str, source.SourceFileContract] = {}
    for key, raw in payloads.items():
        relative = relatives[key]
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        contracts[key] = source.SourceFileContract(
            key=key,
            relative_path=relative,
            size_bytes=len(raw),
            git_blob_sha1=_git_blob(raw),
            raw_url=f"https://raw.githubusercontent.com/example/repo/commit/{relative}",
            file_sha256=hashlib.sha256(raw).hexdigest(),
            semantic_json=key == "data",
        )
    return root, contracts, payloads


def _replace_data(
    root: Path,
    contracts: dict[str, source.SourceFileContract],
    raw: bytes,
) -> None:
    (root / "data/locomo10.json").write_bytes(raw)
    contracts["data"] = source.SourceFileContract(
        key="data",
        relative_path="data/locomo10.json",
        size_bytes=len(raw),
        git_blob_sha1=_git_blob(raw),
        raw_url=contracts["data"].raw_url,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        semantic_json=True,
    )


def test_strict_qualification_emits_only_safe_aggregates_and_fixed_feasibility(
    tmp_path: Path,
) -> None:
    root, contracts, _payload = _fixture(tmp_path)
    receipt = source.qualify_source(
        source_root=root,
        expected_files=contracts,
    )
    rendered = json.dumps(receipt, sort_keys=True)

    assert receipt["status"] == "qualified_public_non_scoring_schema_topology"
    qualification = receipt["qualification"]
    assert qualification["aggregate_counts"]["conversation_count"] == 10
    assert qualification["partition_feasibility"] == {
        "all_conversations_meet_every_family_quota": True,
        "conversation_count_exactly_ten": True,
        "fixed_partition_shape": {
            "A_form_and_label_free_F_search": 2,
            "A_hold": 4,
            "M_search": 4,
        },
        "partition_feasible_without_selecting_conversations": True,
        "selected_conversation_count": 0,
    }
    assert {
        key: value["minimum_per_conversation"]
        for key, value in qualification["family_capacity"].items()
    } == {"MULTI_HOP": 12, "SINGLE_HOP": 12, "TEMPORAL": 12}
    assert qualification["total_schema_anomaly_count"] == 0
    assert receipt["access_boundary"] == {
        "action_evaluator_qrel_or_score_count": 0,
        "conversation_cohort_or_secret_count": 0,
        "individual_source_value_output_count": 0,
        "public_data_JSON_decode_count": 1,
        "source_file_identity_read_count": 3,
    }
    body = dict(receipt)
    declared = body.pop("self_sha256")
    assert declared == source._stable_hash(body)
    for forbidden in (
        "PRIVATE SAMPLE",
        "PRIVATE SPEAKER",
        "PRIVATE DATE",
        "PRIVATE TURN",
        "PRIVATE QUESTION",
        "PRIVATE ANSWER",
        "D1:1",
    ):
        assert forbidden not in rendered


def test_quota_or_evidence_mapping_failure_is_terminal_without_selection(
    tmp_path: Path,
) -> None:
    root, contracts, _payload = _fixture(tmp_path, quota=11)
    rows = json.loads((root / "data/locomo10.json").read_text("utf-8"))
    rows[0]["qa"][0]["evidence"] = ["D9:999"]
    _replace_data(
        root,
        contracts,
        json.dumps(rows, separators=(",", ":")).encode("utf-8"),
    )

    receipt = source.qualify_source(
        source_root=root,
        expected_files=contracts,
    )
    assert receipt["status"] == "terminal_not_qualified_no_same_source_revision"
    qualification = receipt["qualification"]
    assert not qualification["partition_feasibility"][
        "partition_feasible_without_selecting_conversations"
    ]
    assert qualification["partition_feasibility"]["selected_conversation_count"] == 0
    assert qualification["schema_anomaly_count"][
        "evidence_dia_id_not_in_conversation"
    ] == 1


@pytest.mark.parametrize(
    "raw,error",
    [
        (
            b'[{"sample_id":"PRIVATE A","sample_id":"PRIVATE B"}]',
            "duplicate",
        ),
        (b'[{"value":NaN}]', "non-finite"),
    ],
)
def test_duplicate_keys_and_nonfinite_numbers_fail_closed(
    tmp_path: Path,
    raw: bytes,
    error: str,
) -> None:
    root, contracts, _payload = _fixture(tmp_path)
    _replace_data(root, contracts, raw)
    with pytest.raises(source.LocomoP0QualificationError, match=error):
        source.qualify_source(
            source_root=root,
            expected_files=contracts,
        )


def test_byte_identity_is_checked_before_json_decode(tmp_path: Path) -> None:
    root, contracts, _payload = _fixture(tmp_path)
    data = contracts["data"]
    contracts["data"] = source.SourceFileContract(
        key=data.key,
        relative_path=data.relative_path,
        size_bytes=data.size_bytes,
        git_blob_sha1="0" * 40,
        raw_url=data.raw_url,
        file_sha256=data.file_sha256,
        semantic_json=True,
    )
    with pytest.raises(source.LocomoP0QualificationError, match="identity"):
        source.qualify_source(
            source_root=root,
            expected_files=contracts,
        )


def test_one_shot_acquisition_writes_0600_receipt_and_cannot_replay(
    linux_tmp: Path,
) -> None:
    _root, contracts, payloads = _fixture(linux_tmp / "fixture")
    project = linux_tmp / "project"
    project.mkdir()
    work = linux_tmp / "work"
    calls: list[str] = []
    by_url = {
        contracts[key].raw_url: payloads[key]
        for key in contracts
    }

    def opener(url: str) -> _Response:
        calls.append(url)
        return _Response(by_url[url], url)

    receipt = source.acquire_and_qualify(
        project_root=project,
        work_root=work,
        opener=opener,
        expected_files=contracts,
        manifest_verifier=lambda _project: None,
    )
    assert receipt["status"] == "qualified_public_non_scoring_schema_topology"
    assert calls == [
        contracts[key].raw_url for key in ("license", "readme", "data")
    ]
    receipt_path = work / "qualification.receipt.safe.json"
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert json.loads(receipt_path.read_text("ascii")) == receipt
    with pytest.raises(
        source.LocomoP0QualificationError,
        match="already consumed",
    ):
        source.acquire_and_qualify(
            project_root=project,
            work_root=work,
            opener=opener,
            expected_files=contracts,
            manifest_verifier=lambda _project: None,
        )
    assert len(calls) == 3


def test_exclusive_receipt_refuses_overwrite(linux_tmp: Path) -> None:
    destination = linux_tmp / "receipt.json"
    source.write_json_exclusive(destination, {"safe": True})
    assert stat.S_IMODE(os.stat(destination).st_mode) == 0o600
    with pytest.raises(source.LocomoP0QualificationError, match="exclusively"):
        source.write_json_exclusive(destination, {"safe": False})
