from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile
from typing import Any

import pytest

from assumption_agent.benchmarks import (
    scifact_direct_evidence_source_qualification_v1 as qualifier,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _document(doc_id: int, *, sentence_count: int = 5) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "title": f"private title {doc_id}",
        "abstract": [
            f"private sentence {doc_id} {index}" for index in range(sentence_count)
        ],
        "structured": False,
    }


def _claim(
    claim_id: int,
    doc_id: int,
    *,
    label: str,
    sentences: list[int],
    extra_cited: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "id": claim_id,
        "claim": f"private claim {claim_id}",
        "evidence": {
            str(doc_id): [{"label": label, "sentences": sentences}],
        },
        "cited_doc_ids": [doc_id, *(extra_cited or [])],
    }


def _three_families(prefix: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents = [_document(prefix + index) for index in range(3)]
    claims = [
        _claim(prefix + 10, prefix, label="CONTRADICT", sentences=[1]),
        _claim(prefix + 11, prefix + 1, label="SUPPORT", sentences=[0, 2]),
        _claim(prefix + 12, prefix + 2, label="SUPPORT", sentences=[3]),
    ]
    return documents, claims


def _verify_self_hash(receipt: dict[str, Any]) -> None:
    body = copy.deepcopy(receipt)
    declared = body.pop("qualification_sha256")
    assert hashlib.sha256(_canonical(body)).hexdigest() == declared


def test_balanced_component_disjoint_source_passes_without_selection() -> None:
    train_documents, train_claims = _three_families(1000)
    dev_documents, dev_claims = _three_families(2000)
    receipt = qualifier.qualify_decoded_sources(
        _jsonl(train_documents + dev_documents),
        _jsonl(train_claims),
        _jsonl(dev_claims),
        source_binding={"fixture": "synthetic"},
        train_demands={family: 1 for family in qualifier.FAMILY_ORDER},
        dev_demands={family: 1 for family in qualifier.FAMILY_ORDER},
    )
    assert receipt["status"] == "qualified_source_capacity_no_selection"
    assert receipt["terminal_reason_counts"] == {
        "schema_error_count": 0,
        "mapping_error_count": 0,
        "unsatisfied_capacity_count": 0,
    }
    for split in qualifier.SPLIT_ORDER:
        capacity = receipt["simultaneous_component_disjoint_capacity"][split]
        assert capacity["simultaneous_family_capacity_saturated"] is True
        assert capacity["maximum_flow_assigned_total"] == 3
    serialized = _canonical(receipt).decode("ascii")
    assert "private claim" not in serialized
    assert "private title" not in serialized
    assert "private sentence" not in serialized
    assert ":1000:" not in serialized
    assert receipt["claim_boundary"]["test_member_payload_opened"] is False
    _verify_self_hash(receipt)


def test_cross_split_component_and_public_example_component_are_excluded() -> None:
    train_documents, train_claims = _three_families(3000)
    dev_documents, dev_claims = _three_families(4000)
    shared_doc = 9999
    train_claims[0]["cited_doc_ids"].append(shared_doc)
    dev_claims[0]["cited_doc_ids"].append(shared_doc)
    denied_document = _document(min(qualifier.DENY_DOC_IDS))
    denied_doc_id = denied_document["doc_id"]
    train_documents.append(denied_document)
    train_claims.append(
        _claim(8888, denied_doc_id, label="CONTRADICT", sentences=[0])
    )
    receipt = qualifier.qualify_decoded_sources(
        _jsonl(train_documents + dev_documents),
        _jsonl(train_claims),
        _jsonl(dev_claims),
        source_binding={},
        train_demands={family: 1 for family in qualifier.FAMILY_ORDER},
        dev_demands={family: 1 for family in qualifier.FAMILY_ORDER},
    )
    assert receipt["status"] == "terminal_source_infeasible_no_selection"
    graph = receipt["candidate_and_component_aggregates"]["component_graph"]
    assert graph["cross_split_component_count"] == 1
    assert graph["declared_public_example_intersecting_component_count"] == 1
    train = receipt["candidate_and_component_aggregates"]["candidate_splits"][
        "train"
    ]
    assert train["cross_split_component_excluded_candidate_counts"][
        "CONTRADICT_SINGLE"
    ] == 1
    assert train["public_example_component_excluded_candidate_counts"][
        "CONTRADICT_SINGLE"
    ] == 1


def test_large_rationale_is_ineligible_and_bad_position_is_mapping_error() -> None:
    documents, claims = _three_families(5000)
    claims.append(_claim(6000, 5000, label="SUPPORT", sentences=[0, 1, 2, 3]))
    claims.append(_claim(6001, 5001, label="SUPPORT", sentences=[99]))
    receipt = qualifier.qualify_decoded_sources(
        _jsonl(documents),
        _jsonl(claims),
        _jsonl([]),
        source_binding={},
        train_demands={family: 1 for family in qualifier.FAMILY_ORDER},
        dev_demands={family: 0 for family in qualifier.FAMILY_ORDER},
    )
    train = receipt["candidate_and_component_aggregates"]["candidate_splits"][
        "train"
    ]
    assert train["source_ineligible_reason_counts"] == {
        "mapping_error": 1,
        "rationale_size_above_Set3": 1,
    }
    assert train["mapping_error_reason_counts"] == {
        "rationale_position_out_of_bounds": 1
    }
    assert receipt["status"] == "terminal_source_infeasible_no_selection"


def _tar_bytes(members: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, raw in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    return output.getvalue()


def test_archive_extraction_opens_only_allowed_members() -> None:
    members = {
        "data/claims_train.jsonl": b"train\n",
        "data/claims_dev.jsonl": b"dev\n",
        "data/corpus.jsonl": b"corpus\n",
        qualifier.TEST_MEMBER: b"forbidden-private-test-content\n",
        "data/cross_validation/fold_1/claims_train_1.jsonl": b"other\n",
    }
    archive_raw = _tar_bytes(members)
    with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
        root = Path(temporary)
        archive_path = root / "data.tar.gz"
        archive_path.write_bytes(archive_raw)
        archive_path.chmod(0o600)
        specs = {
            "train": (
                "data/claims_train.jsonl",
                len(members["data/claims_train.jsonl"]),
            ),
            "dev": (
                "data/claims_dev.jsonl",
                len(members["data/claims_dev.jsonl"]),
            ),
            "corpus": ("data/corpus.jsonl", len(members["data/corpus.jsonl"])),
        }
        destination = root / "private"
        bindings = qualifier.extract_allowed_members_once(
            archive_path,
            destination,
            expected_archive_size=len(archive_raw),
            expected_archive_sha256=hashlib.sha256(archive_raw).hexdigest(),
            member_specs=specs,
        )
        assert set(path.name for path in destination.iterdir()) == {
            "claims_train.jsonl",
            "claims_dev.jsonl",
            "corpus.jsonl",
        }
        assert all(
            stat.S_IMODE(binding.private_path.stat().st_mode) == 0o600
            for binding in bindings.values()
        )
        assert b"forbidden-private-test-content" not in b"".join(
            path.read_bytes() for path in destination.iterdir()
        )
        with pytest.raises(qualifier.OneShotRefusal):
            qualifier.extract_allowed_members_once(
                archive_path,
                destination,
                expected_archive_size=len(archive_raw),
                expected_archive_sha256=hashlib.sha256(archive_raw).hexdigest(),
                member_specs=specs,
            )


def test_archive_identity_drift_fails_before_private_root(tmp_path: Path) -> None:
    archive_path = tmp_path / "data.tar.gz"
    archive_path.write_bytes(b"wrong")
    destination = tmp_path / "private"
    with pytest.raises(qualifier.SciFactQualificationError, match="size drifted"):
        qualifier.extract_allowed_members_once(
            archive_path,
            destination,
            expected_archive_size=99,
            expected_archive_sha256="0" * 64,
            member_specs={},
        )
    assert not destination.exists()


def test_duplicate_JSON_keys_are_schema_errors_not_item_output() -> None:
    documents, claims = _three_families(7000)
    bad = b'{"id":1,"id":2}\n'
    receipt = qualifier.qualify_decoded_sources(
        _jsonl(documents),
        _jsonl(claims) + bad,
        b"",
        source_binding={},
        train_demands={family: 1 for family in qualifier.FAMILY_ORDER},
        dev_demands={family: 0 for family in qualifier.FAMILY_ORDER},
    )
    assert receipt["terminal_reason_counts"]["schema_error_count"] == 1
    assert receipt["source_aggregates"]["claims"]["train"][
        "invalid_row_reason_counts"
    ] == {"json_line": 1}
    assert receipt["status"] == "terminal_source_infeasible_no_selection"
