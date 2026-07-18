from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import unicodedata

import pytest

from assumption_agent.benchmarks import hover_source_qualification_v1 as module


def _fixture() -> tuple[list[dict[str, object]], sqlite3.Connection]:
    payload: list[dict[str, object]] = []
    documents: list[tuple[str, str]] = []
    for hop in module.HOP_ORDER:
        for ordinal in range(48):
            titles = [
                f"Caf\u00e9 gold {hop}-{ordinal}-{position}"
                for position in range(hop)
            ]
            payload.append(
                {
                    "uid": f"uid-{hop}-{ordinal}",
                    "hpqa_id": f"seed-{hop}-{ordinal}",
                    "claim": f"Private claim {hop} {ordinal}",
                    "num_hops": hop,
                    "supporting_facts": [
                        [title, position] for position, title in enumerate(titles)
                    ],
                    "label": "SUPPORTED",
                }
            )
            documents.extend(
                (
                    unicodedata.normalize("NFD", title),
                    f"Private body for {title}",
                )
                for title in titles
            )
    for ordinal in range(609 - len(documents)):
        documents.append((f"Filler {ordinal}", f"Filler body {ordinal}"))
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE documents (id TEXT PRIMARY KEY, text TEXT NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO documents (id, text) VALUES (?, ?)", documents
    )
    connection.commit()
    return payload, connection


def _qualify(
    payload: list[dict[str, object]], connection: sqlite3.Connection
) -> dict[str, object]:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return module.qualify_payload(
        payload,
        connection,
        expected_row_count=144,
        expected_hop_counts={2: 48, 3: 48, 4: 48},
        source_size=len(raw),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_git_blob_sha1="0" * 40,
        db_size=1,
        db_sha256="1" * 64,
        formal_identity_enforced=False,
    )


def test_aggregate_qualification_is_capacity_complete_and_content_free() -> None:
    payload, connection = _fixture()
    receipt = _qualify(payload, connection)
    assert receipt["status"] == "passed_source_qualification_no_selection"
    assert receipt["capacity"]["exact_three_hop_b_matching_capacity_met"] is True
    assert receipt["capacity"]["maximum_selected_gold_occurrences"] == 432
    assert receipt["capacity"]["minimum_filler_slots"] == 177
    assert receipt["hop_and_structure"]["hop_counts"] == {
        "2": 48,
        "3": 48,
        "4": 48,
    }
    body = dict(receipt)
    declared = body.pop("qualification_sha256")
    assert declared == module._stable_hash(body)
    serialized = json.dumps(receipt, ensure_ascii=True, sort_keys=True)
    assert "uid-2-0" not in serialized
    assert "seed-2-0" not in serialized
    assert "Private claim" not in serialized
    assert "Caf" not in serialized
    assert "Private body" not in serialized
    assert receipt["claim_boundary"]["selection_secret_or_cohort_created"] is False
    connection.close()


def test_title_resolution_is_exact_nfd_not_fuzzy() -> None:
    payload, connection = _fixture()
    connection.execute(
        "UPDATE documents SET id = 'Cafe wrong' WHERE id = ?",
        (unicodedata.normalize("NFD", "Caf\u00e9 gold 2-0-0"),),
    )
    connection.commit()
    with pytest.raises(
        module.HoVerSourceQualificationError,
        match="b-matching capacity",
    ):
        _qualify(payload, connection)
    connection.close()


def test_hpqa_group_overlap_is_rejected_by_hall_capacity() -> None:
    payload, connection = _fixture()
    for row in payload:
        if row["num_hops"] == 3:
            suffix = str(row["hpqa_id"]).rsplit("-", 1)[-1]
            row["hpqa_id"] = f"seed-2-{suffix}"
    with pytest.raises(
        module.HoVerSourceQualificationError,
        match="b-matching capacity",
    ):
        _qualify(payload, connection)
    connection.close()


def test_normalized_claim_collision_excludes_whole_group() -> None:
    payload, connection = _fixture()
    payload[1]["claim"] = "  PRIVATE   CLAIM 2 0  "
    with pytest.raises(
        module.HoVerSourceQualificationError,
        match="b-matching capacity",
    ):
        _qualify(payload, connection)
    connection.close()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(num_hops=True),
        lambda row: row.update(supporting_facts=[["Title", -1]]),
        lambda row: row.pop("hpqa_id"),
    ],
)
def test_consumed_schema_drift_is_terminal(mutation) -> None:
    payload, connection = _fixture()
    changed = copy.deepcopy(payload)
    mutation(changed[0])
    with pytest.raises(module.HoVerSourceQualificationError):
        _qualify(changed, connection)
    connection.close()


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    with pytest.raises(module.HoVerSourceQualificationError):
        module._decode_strict_json(b'{"uid":"a","uid":"b"}')
    with pytest.raises(module.HoVerSourceQualificationError):
        module._decode_strict_json(b'{"value":NaN}')
