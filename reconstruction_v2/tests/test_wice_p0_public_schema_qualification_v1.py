from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from assumption_agent.benchmarks.wice_p0_public_schema_qualification_v1 import (
    WiceP0QualificationError,
    qualify_source,
    write_receipt_exclusive,
)


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(  # noqa: S324 - Git object identity.
        f"blob {len(raw)}\0".encode("ascii") + raw
    ).hexdigest()


def _row(
    *,
    identifier: str,
    claim: str,
    label: str,
    evidence: list[str],
    supporting_sentences: list[list[int]],
) -> dict[str, object]:
    return {
        "label": label,
        "supporting_sentences": supporting_sentences,
        "claim": claim,
        "evidence": evidence,
        "meta": {
            "id": identifier,
            "claim_title": "PRIVATE TITLE",
            "claim_section": "PRIVATE SECTION",
            "claim_context": "PRIVATE CONTEXT",
        },
    }


def _payloads() -> dict[str, list[dict[str, object]]]:
    shared_claim = "PRIVATE SHARED CLAIM"
    shared_evidence = [
        "PRIVATE SHARED EVIDENCE ZERO",
        "PRIVATE SHARED EVIDENCE ONE",
    ]
    return {
        "train": [
            _row(
                identifier="PRIVATE TRAIN ID 0",
                claim=shared_claim,
                label="supported",
                evidence=shared_evidence,
                supporting_sentences=[[0]],
            ),
            _row(
                identifier="PRIVATE TRAIN ID 1",
                claim=shared_claim,
                label="supported",
                evidence=shared_evidence,
                supporting_sentences=[[0]],
            ),
            _row(
                identifier="PRIVATE TRAIN ID 2",
                claim="PRIVATE TWO HOP CLAIM",
                label="partially_supported",
                evidence=[
                    "PRIVATE TWO HOP EVIDENCE ZERO",
                    "PRIVATE TWO HOP EVIDENCE ONE",
                    "PRIVATE TWO HOP EVIDENCE TWO",
                ],
                supporting_sentences=[[0, 1], [1, 2]],
            ),
            _row(
                identifier="PRIVATE TRAIN ID 3",
                claim="PRIVATE THREE HOP CLAIM",
                label="supported",
                evidence=[
                    "PRIVATE THREE HOP EVIDENCE ZERO",
                    "PRIVATE THREE HOP EVIDENCE ONE",
                    "PRIVATE THREE HOP EVIDENCE TWO",
                ],
                supporting_sentences=[[0, 1, 2]],
            ),
            _row(
                identifier="PRIVATE TRAIN ID 4",
                claim="PRIVATE UNSUPPORTED CLAIM",
                label="not_supported",
                evidence=["PRIVATE UNSUPPORTED EVIDENCE"],
                supporting_sentences=[[]],
            ),
        ],
        "dev": [
            _row(
                identifier="PRIVATE DEV ID 0",
                claim=shared_claim,
                label="supported",
                evidence=shared_evidence,
                supporting_sentences=[[1]],
            )
        ],
        "test": [
            _row(
                identifier="PRIVATE TEST ID 0",
                claim="PRIVATE TEST THREE HOP CLAIM",
                label="partially_supported",
                evidence=[
                    "PRIVATE TEST EVIDENCE ZERO",
                    "PRIVATE TEST EVIDENCE ONE",
                    "PRIVATE TEST EVIDENCE TWO",
                ],
                supporting_sentences=[[0, 1, 2]],
            )
        ],
    }


def _fixture(
    tmp_path: Path,
    *,
    embed_sha256: bool = True,
) -> tuple[
    Path,
    dict[str, dict[str, object]],
    dict[str, str],
]:
    root = tmp_path / "source"
    bindings: dict[str, dict[str, object]] = {}
    sha256s: dict[str, str] = {}
    for split, rows in _payloads().items():
        relative = f"data/entailment_retrieval/claim/{split}.jsonl"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (
            "\n".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True)
                for row in rows
            )
            + "\n"
        ).encode("utf-8")
        path.write_bytes(raw)
        sha256s[split] = hashlib.sha256(raw).hexdigest()
        bindings[split] = {
            "git_blob_sha1": _git_blob_sha1(raw),
            "relative_path": relative,
            "size_bytes": len(raw),
        }
        if embed_sha256:
            bindings[split]["file_sha256"] = sha256s[split]
    return root, bindings, sha256s


def test_qualification_emits_only_safe_aggregates_and_family_capacity(
    tmp_path: Path,
) -> None:
    root, bindings, _sha256s = _fixture(tmp_path)
    receipt = qualify_source(source_root=root, expected_files=bindings)
    rendered = json.dumps(receipt, sort_keys=True)

    assert receipt["status"] == (
        "qualified_public_non_scoring_schema_topology"
    )
    assert receipt["total_schema_anomaly_count"] == 0
    assert receipt["split_receipts"]["train"][
        "eligible_family_capacity_count"
    ] == {
        "MIN_SUPPORTING_SET_SIZE_1": 2,
        "MIN_SUPPORTING_SET_SIZE_2": 1,
        "MIN_SUPPORTING_SET_SIZE_GE_3": 1,
    }
    assert receipt["split_receipts"]["dev"][
        "eligible_family_capacity_count"
    ] == {"MIN_SUPPORTING_SET_SIZE_1": 1}
    assert "test" not in receipt["split_receipts"]
    assert receipt["source_files"]["test"]["json_decode_count"] == 0
    assert receipt["source_files"]["test"]["raw_newline_count"] == 1
    assert receipt["split_semantic_access_policy"] == {
        "identity_only_splits": ["test"],
        "semantic_aggregate_splits": ["train", "dev"],
    }
    assert receipt["split_receipts"]["train"][
        "minimum_valid_supporting_set_size_histogram"
    ] == {"1": 2, "2": 1, "3": 1}
    assert receipt["p1_quota_or_cohort_assumption_count"] == 0
    assert receipt["split_receipts"]["train"][
        "eligible_family_unique_evidence_component_count"
    ] == {
        "MIN_SUPPORTING_SET_SIZE_1": 1,
        "MIN_SUPPORTING_SET_SIZE_2": 1,
        "MIN_SUPPORTING_SET_SIZE_GE_3": 1,
    }
    assert receipt["access_boundary"] == {
        "action_model_evaluator_qrel_or_score_count": 0,
        "individual_claim_evidence_meta_identifier_or_support_index_output_count": 0,
        "private_cohort_or_secret_count": 0,
        "public_source_file_identity_read_count": 3,
        "public_source_split_json_decode_count": 2,
        "test_json_decode_count": 0,
    }
    for forbidden in (
        "PRIVATE SHARED CLAIM",
        "PRIVATE SHARED EVIDENCE",
        "PRIVATE TITLE",
        "PRIVATE SECTION",
        "PRIVATE CONTEXT",
        "PRIVATE TRAIN ID",
    ):
        assert forbidden not in rendered


def test_component_collision_counts_cover_within_and_cross_split(
    tmp_path: Path,
) -> None:
    root, bindings, _sha256s = _fixture(tmp_path)
    receipt = qualify_source(source_root=root, expected_files=bindings)
    within = receipt["split_receipts"]["train"][
        "component_collision_count"
    ]
    cross = receipt["cross_split_component_collision_count"]

    assert within["claim"] == {
        "collision_excess_member_count": 1,
        "collision_group_count": 1,
        "collision_member_count": 2,
        "collision_row_count": 2,
        "unique_component_count": 4,
    }
    assert within["evidence_list"] == {
        "collision_excess_member_count": 1,
        "collision_group_count": 1,
        "collision_member_count": 2,
        "collision_row_count": 2,
        "unique_component_count": 4,
    }
    assert within["claim_evidence_pair"] == {
        "collision_excess_member_count": 1,
        "collision_group_count": 1,
        "collision_member_count": 2,
        "collision_row_count": 2,
        "unique_component_count": 4,
    }
    assert cross["claim"] == {
        "cross_split_collision_excess_member_count": 2,
        "cross_split_collision_group_count": 1,
        "cross_split_collision_member_count": 3,
        "cross_split_collision_row_count": 3,
    }
    assert cross["evidence_list"] == cross["claim"]
    assert cross["claim_evidence_pair"] == cross["claim"]


def test_external_sha256_bindings_are_required_and_verified(
    tmp_path: Path,
) -> None:
    root, bindings, sha256s = _fixture(
        tmp_path,
        embed_sha256=False,
    )
    receipt = qualify_source(
        source_root=root,
        expected_files=bindings,
        expected_sha256s=sha256s,
    )
    assert receipt["status"] == (
        "qualified_public_non_scoring_schema_topology"
    )

    missing = dict(sha256s)
    missing.pop("test")
    with pytest.raises(
        WiceP0QualificationError,
        match="required for every split",
    ):
        qualify_source(
            source_root=root,
            expected_files=bindings,
            expected_sha256s=missing,
        )

    wrong = dict(sha256s)
    wrong["dev"] = "0" * 64
    with pytest.raises(
        WiceP0QualificationError,
        match="whole-file SHA-256",
    ):
        qualify_source(
            source_root=root,
            expected_files=bindings,
            expected_sha256s=wrong,
        )


def test_git_blob_identity_is_verified_before_schema_parse(
    tmp_path: Path,
) -> None:
    root, bindings, _sha256s = _fixture(tmp_path)
    bindings["train"]["git_blob_sha1"] = "0" * 40
    with pytest.raises(
        WiceP0QualificationError,
        match="Git blob identity",
    ):
        qualify_source(source_root=root, expected_files=bindings)


def test_supporting_set_anomalies_are_aggregate_and_block_qualification(
    tmp_path: Path,
) -> None:
    root, bindings, _sha256s = _fixture(tmp_path)
    train_path = root / str(bindings["train"]["relative_path"])
    rows = _payloads()["train"]
    rows[0]["supporting_sentences"] = [
        [0],
        [0],
        [1, 1],
        [99],
        [],
    ]
    raw = (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in rows
        )
        + "\n"
    ).encode("utf-8")
    train_path.write_bytes(raw)
    bindings["train"].update(
        {
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_sha1": _git_blob_sha1(raw),
            "size_bytes": len(raw),
        }
    )

    receipt = qualify_source(source_root=root, expected_files=bindings)
    anomalies = receipt["split_receipts"]["train"][
        "schema_anomaly_count"
    ]
    assert receipt["status"] == "not_qualified_public_schema_anomalies"
    assert anomalies["supporting_alternative_duplicate"] == 1
    assert anomalies["supporting_index_duplicate_within_set"] == 1
    assert anomalies["supporting_index_out_of_range"] == 1
    assert anomalies["eligible_label_has_empty_supporting_set"] == 1
    rendered = json.dumps(receipt, sort_keys=True)
    assert "PRIVATE SHARED CLAIM" not in rendered
    assert "PRIVATE SHARED EVIDENCE" not in rendered


def test_test_split_is_identity_only_even_when_body_is_invalid_json(
    tmp_path: Path,
) -> None:
    root, bindings, _sha256s = _fixture(tmp_path)
    test_path = root / str(bindings["test"]["relative_path"])
    raw = (
        b'{"label":"supported","label":"not_supported",'
        b'"supporting_sentences":[[]],"claim":"PRIVATE DUPLICATE",'
        b'"evidence":[],"meta":{}}\n'
    )
    test_path.write_bytes(raw)
    bindings["test"].update(
        {
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_sha1": _git_blob_sha1(raw),
            "size_bytes": len(raw),
        }
    )

    receipt = qualify_source(source_root=root, expected_files=bindings)
    assert "test" not in receipt["split_receipts"]
    assert receipt["source_files"]["test"]["json_decode_count"] == 0
    assert receipt["source_files"]["test"]["raw_newline_count"] == 1
    assert receipt["status"] == (
        "qualified_public_non_scoring_schema_topology"
    )
    assert "PRIVATE DUPLICATE" not in json.dumps(receipt, sort_keys=True)


def test_semantic_split_contract_cannot_open_test(
    tmp_path: Path,
) -> None:
    root, bindings, _sha256s = _fixture(tmp_path)
    with pytest.raises(
        WiceP0QualificationError,
        match="exactly train then dev",
    ):
        qualify_source(
            source_root=root,
            expected_files=bindings,
            semantic_splits=("train", "dev", "test"),
        )


def test_receipt_is_exclusive_mode_0600_and_self_hashed(
    tmp_path: Path,
) -> None:
    mode_probe = tmp_path / "mode_probe"
    probe_fd = os.open(
        mode_probe,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    os.close(probe_fd)
    if stat.S_IMODE(mode_probe.stat().st_mode) != 0o600:
        pytest.skip("temporary filesystem does not preserve POSIX mode bits")

    root, bindings, _sha256s = _fixture(tmp_path)
    receipt = qualify_source(source_root=root, expected_files=bindings)
    body = dict(receipt)
    self_sha256 = body.pop("self_sha256")
    expected_self_sha256 = hashlib.sha256(
        json.dumps(
            body,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    assert self_sha256 == expected_self_sha256

    output = tmp_path / "receipt.json"
    write_receipt_exclusive(output, receipt)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text(encoding="ascii")) == receipt
    with pytest.raises(
        WiceP0QualificationError,
        match="exclusively",
    ):
        write_receipt_exclusive(output, receipt)
