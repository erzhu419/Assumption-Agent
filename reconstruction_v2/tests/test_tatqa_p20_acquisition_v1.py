from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

import pytest

from assumption_agent.benchmarks import tatqa_p20_acquisition_v1 as acquisition


def _raw_question(*, uid: str, order: int, family: str) -> dict[str, Any]:
    return {
        "uid": uid,
        "order": order,
        "question": f"Which synthetic fact belongs to {uid}?",
        "answer": f"answer-{uid}",
        "derivation": "",
        "answer_type": "span",
        "answer_from": family,
        "rel_paragraphs": ["1"],
        "req_comparison": False,
        "scale": "",
    }


def _mapping(family: str) -> dict[str, Any]:
    if family == "table":
        return {"table": [[1, 1]], "paragraph": {}}
    if family == "text":
        return {"table": [], "paragraph": {"1": [[0, 5]]}}
    if family == "table-text":
        return {
            "table": [[1, 1], [1, 0]],
            "paragraph": {"2": [[0, 4]]},
        }
    raise AssertionError(family)


def _tagop_question(raw: Mapping[str, Any], *, mapping: object | None = None) -> dict[str, Any]:
    return {
        "uid": raw["uid"],
        "question": raw["question"],
        "answer": raw["answer"],
        "derivation": raw["derivation"],
        "answer_type": raw["answer_type"],
        "answer_from": raw["answer_from"],
        "scale": raw["scale"],
        "facts": [],
        "mapping": _mapping(str(raw["answer_from"])) if mapping is None else mapping,
    }


def _context(
    *,
    table_uid: str,
    questions: list[tuple[str, str]],
    table_row_count: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = [["Metric", "Value"]]
    matrix.extend(
        [[f"metric-{row}", f"value-{row}"] for row in range(1, table_row_count)]
    )
    paragraphs = [
        {
            "uid": f"{table_uid}-p1",
            "order": 1,
            "text": f"alpha evidence for {table_uid}",
        },
        {
            "uid": f"{table_uid}-p2",
            "order": 2,
            "text": f"beta evidence for {table_uid}",
        },
    ]
    raw_questions = [
        _raw_question(uid=uid, order=index + 1, family=family)
        for index, (uid, family) in enumerate(questions)
    ]
    raw = {
        "table": {"uid": table_uid, "table": matrix},
        "paragraphs": paragraphs,
        "questions": raw_questions,
    }
    tagop = {
        "table": copy.deepcopy(raw["table"]),
        "paragraphs": copy.deepcopy(paragraphs),
        "questions": [_tagop_question(row) for row in raw_questions],
    }
    return raw, tagop


def _population_payloads() -> tuple[dict[str, object], dict[str, object]]:
    raw_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    tagop_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    counts = {"train": 38, "dev": 10}
    source_family = {"TABLE": "table", "TEXT": "text", "TABLE_TEXT": "table-text"}
    for split in acquisition.SPLIT_ORDER:
        for family in acquisition.FAMILY_ORDER:
            for index in range(counts[split]):
                stem = f"{split}-{family.lower()}-{index}"
                raw, tagop = _context(
                    table_uid=f"table-{stem}",
                    questions=[(f"question-{stem}", source_family[family])],
                )
                raw_by_split[split].append(raw)
                tagop_by_split[split].append(tagop)
    return raw_by_split, tagop_by_split


def _source_bytes() -> dict[str, bytes]:
    raw, tagop = _population_payloads()
    encode = lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8")
    return {
        "LICENSE": b"synthetic fixture only\n",
        "dataset_raw/tatqa_dataset_dev.json": encode(raw["dev"]),
        "dataset_raw/tatqa_dataset_train.json": encode(raw["train"]),
        "dataset_tagop/tatqa_dataset_dev.json": encode(tagop["dev"]),
        "dataset_tagop/tatqa_dataset_train.json": encode(tagop["train"]),
    }


def _direct_candidate(
    *, table_uid: str, question_uid: str, family: str, split: str = "train"
) -> acquisition.Candidate:
    units = (
        acquisition.CanonicalUnit("T:0", "TABLE_HEADER|C0=Metric||C1=Value"),
        acquisition.CanonicalUnit("T:1", "TABLE_ROW_1|Metric=a||Value=b"),
        acquisition.CanonicalUnit("T:2", "TABLE_ROW_2|Metric=c||Value=d"),
        acquisition.CanonicalUnit("P:1", "PARAGRAPH_1|alpha evidence"),
        acquisition.CanonicalUnit("P:2", "PARAGRAPH_2|beta evidence"),
    )
    gold = {
        "TABLE": ("T:1",),
        "TEXT": ("P:1",),
        "TABLE_TEXT": ("T:1", "P:1"),
    }[family]
    return acquisition.Candidate(
        split=split,
        source_context_ordinal=0,
        source_question_ordinal=0,
        table_uid=table_uid,
        question_uid=question_uid,
        question=f"Which fact belongs to {question_uid}?",
        units=units,
        family=family,
        gold_unit_ids=gold,
    )


def _walk_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(key)
            keys.update(_walk_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_walk_keys(nested))
    return keys


def test_parser_serializes_units_and_projects_cells_and_spans_to_whole_units() -> None:
    raw, tagop = _context(
        table_uid="context-1", questions=[("question-1", "table-text")]
    )
    raw["table"]["table"][0][1] = "  Ｖａｌｕｅ\tName  "
    tagop["table"] = copy.deepcopy(raw["table"])
    tagop["questions"][0]["mapping"] = {
        "table": [[1, 0], [1, 1], [1, 1]],
        "paragraph": {"1": [[0, 5], [6, 14]]},
    }

    candidates, aggregate = acquisition.parse_source_pair(
        [raw], [tagop], split="train"
    )

    assert aggregate["eligible_question_count_by_family"] == {
        "TABLE": 0,
        "TEXT": 0,
        "TABLE_TEXT": 1,
    }
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.gold_unit_ids == ("T:1", "P:1")
    assert [unit.unit_id for unit in candidate.units] == [
        "T:0",
        "T:1",
        "T:2",
        "P:1",
        "P:2",
    ]
    assert candidate.units[0].text == "TABLE_HEADER|C0=Metric||C1=Value Name"
    assert candidate.units[1].text.startswith("TABLE_ROW_1|Metric=")
    assert candidate.units[3].text.startswith("PARAGRAPH_1|alpha evidence")


def test_parser_rejects_raw_tagop_identity_and_exact_schema_drift() -> None:
    raw, tagop = _context(table_uid="context", questions=[("question", "table")])
    tagop["questions"][0]["question"] += " changed"
    with pytest.raises(acquisition.TatqaP20AcquisitionError, match="identity"):
        acquisition.parse_source_pair([raw], [tagop], split="train")

    raw, tagop = _context(table_uid="context", questions=[("question", "table")])
    raw["questions"][0]["unexpected"] = True
    with pytest.raises(acquisition.TatqaP20AcquisitionError, match="schema"):
        acquisition.parse_source_pair([raw], [tagop], split="train")

    raw, tagop = _context(table_uid="context", questions=[("question", "table")])
    tagop["questions"][0]["unexpected"] = True
    with pytest.raises(acquisition.TatqaP20AcquisitionError, match="schema"):
        acquisition.parse_source_pair([raw], [tagop], split="train")


def test_parser_rejects_ragged_table_bad_paragraph_order_and_four_unit_context() -> None:
    raw, tagop = _context(table_uid="ragged", questions=[("q", "table")])
    raw["table"]["table"][1].pop()
    tagop["table"] = copy.deepcopy(raw["table"])
    with pytest.raises(acquisition.TatqaP20AcquisitionError, match="rectangular"):
        acquisition.parse_source_pair([raw], [tagop], split="train")

    raw, tagop = _context(table_uid="orders", questions=[("q", "table")])
    raw["paragraphs"][1]["order"] = 1
    tagop["paragraphs"] = copy.deepcopy(raw["paragraphs"])
    with pytest.raises(acquisition.TatqaP20AcquisitionError, match="unique and positive"):
        acquisition.parse_source_pair([raw], [tagop], split="train")

    raw, tagop = _context(
        table_uid="four-units", questions=[("q", "table")], table_row_count=2
    )
    with pytest.raises(acquisition.TatqaP20AcquisitionError, match="fewer than five"):
        acquisition.parse_source_pair([raw], [tagop], split="train")


@pytest.mark.parametrize("drifted_order", (True, 1.0))
def test_acquisition_rejects_bool_or_float_numeric_order_tamper(
    drifted_order: object,
) -> None:
    raw, tagop = _context(table_uid="numeric-type", questions=[("q", "table")])
    raw["paragraphs"][0]["order"] = drifted_order
    tagop["paragraphs"] = copy.deepcopy(raw["paragraphs"])
    with pytest.raises(acquisition.TatqaP20AcquisitionError, match="unique and positive"):
        acquisition.parse_source_pair([raw], [tagop], split="train")


def test_mapping_bounds_nonblank_and_family_consistency_are_fail_closed() -> None:
    paragraphs = {1: "alpha evidence", 2: "    "}
    with pytest.raises(acquisition.TatqaP20AcquisitionError, match="out of bounds"):
        acquisition.project_gold_mapping(
            {"table": [[3, 0]], "paragraph": {}},
            table_row_count=3,
            table_column_count=2,
            paragraph_text_by_order=paragraphs,
        )
    with pytest.raises(acquisition.TatqaP20AcquisitionError, match="blank"):
        acquisition.project_gold_mapping(
            {"table": [], "paragraph": {"2": [[0, 4]]}},
            table_row_count=3,
            table_column_count=2,
            paragraph_text_by_order=paragraphs,
        )
    with pytest.raises(acquisition.TatqaP20AcquisitionError, match="canonical positive"):
        acquisition.project_gold_mapping(
            {"table": [], "paragraph": {"01": [[0, 1]]}},
            table_row_count=3,
            table_column_count=2,
            paragraph_text_by_order=paragraphs,
        )

    raw, tagop = _context(table_uid="family", questions=[("q", "table")])
    tagop["questions"][0]["mapping"] = {
        "table": [],
        "paragraph": {"1": [[0, 5]]},
    }
    candidates, aggregate = acquisition.parse_source_pair(
        [raw], [tagop], split="train"
    )
    assert candidates == ()
    assert aggregate["family_mapping_inconsistent_question_count"] == 1


def test_public_example_uid_excludes_the_entire_context() -> None:
    public_uid = next(iter(acquisition.PUBLIC_QUESTION_UIDS))
    raw, tagop = _context(
        table_uid="otherwise-private-context",
        questions=[(public_uid, "table"), ("sibling-question", "text")],
    )
    candidates, aggregate = acquisition.parse_source_pair(
        [raw], [tagop], split="train"
    )
    assert candidates == ()
    assert aggregate["public_example_excluded_context_count"] == 1
    assert aggregate["public_example_excluded_question_count"] == 2
    assert aggregate["eligible_question_count_by_family"] == {
        "TABLE": 0,
        "TEXT": 0,
        "TABLE_TEXT": 0,
    }


def test_hmac_message_has_explicit_framed_bytes_and_shuffle_omits_family() -> None:
    candidate = _direct_candidate(
        table_uid="表-1", question_uid="q-1", family="TABLE_TEXT"
    )
    values = [b"train", b"TABLE_TEXT", "表-1".encode(), b"q-1"]
    expected = b"TATQA_P20_SELECT_V1" + b"".join(
        len(value).to_bytes(4, "big") + value for value in values
    )
    assert acquisition.selection_hmac_message(candidate) == expected
    shuffle = acquisition._shuffle_hmac_message("A_form", candidate)
    assert b"TABLE_TEXT" not in shuffle
    assert b"A_form" in shuffle and "表-1".encode() in shuffle


def test_deterministic_augmenting_match_repairs_a_greedy_context_collision() -> None:
    table_shared = _direct_candidate(
        table_uid="shared", question_uid="table-shared", family="TABLE"
    )
    table_alternative = _direct_candidate(
        table_uid="alternative", question_uid="table-alternative", family="TABLE"
    )
    text_only = _direct_candidate(
        table_uid="shared", question_uid="text-shared", family="TEXT"
    )
    slots = (
        acquisition.Slot(0, "A_form", "train", "TABLE", 0),
        acquisition.Slot(1, "A_form", "train", "TEXT", 0),
    )
    order = {
        "table-shared": 0,
        "table-alternative": 1,
        "text-shared": 0,
    }
    matched = acquisition.deterministic_augmenting_match(
        slots,
        (table_shared, table_alternative, text_only),
        order_key=lambda row: order[row.question_uid],
    )
    assert matched[0].question_uid == "table-alternative"
    assert matched[1].question_uid == "text-shared"
    assert len({row.table_uid for row in matched.values()}) == 2

    priority_slots = (
        acquisition.Slot(0, "A_form", "train", "TABLE", 0),
        acquisition.Slot(1, "A_form", "train", "TABLE", 1),
    )
    priority = acquisition.deterministic_augmenting_match(
        priority_slots,
        (table_shared, table_alternative),
        order_key=lambda row: order[row.question_uid],
    )
    assert priority[0].question_uid == "table-shared"
    assert priority[1].question_uid == "table-alternative"


def test_aggregate_qualification_proves_fixed_capacity_without_any_identifier() -> None:
    raw, tagop = _population_payloads()
    qualified = acquisition.qualify_decoded_sources(
        raw_by_split=raw, tagop_by_split=tagop
    )
    aggregate = qualified.public_aggregate

    assert len(qualified.candidates) == acquisition.TOTAL_SELECTED_ITEMS
    assert aggregate["one_context_one_question_capacity_proven"] is True
    assert aggregate["fixed_block_counts"] == acquisition.BLOCK_COUNTS
    assert aggregate["eligible_question_count_by_split_and_family"] == {
        "train": {family: 38 for family in acquisition.FAMILY_ORDER},
        "dev": {family: 10 for family in acquisition.FAMILY_ORDER},
    }
    assert not {
        "uid",
        "id",
        "table_uid",
        "question_uid",
        "question",
        "answer",
        "mapping",
        "gold_unit_ids",
        "items",
    }.intersection(_walk_keys(aggregate))
    acquisition.verify_self_hash(aggregate, "aggregate_qualification_sha256")


def test_fixed_matching_and_secret_block_shuffle_are_balanced_and_disjoint() -> None:
    raw, tagop = _population_payloads()
    qualified = acquisition.qualify_decoded_sources(
        raw_by_split=raw, tagop_by_split=tagop
    )
    first = acquisition.select_blocks(qualified.candidates, secret=b"A" * 32)
    repeated = acquisition.select_blocks(qualified.candidates, secret=b"A" * 32)
    changed = acquisition.select_blocks(qualified.candidates, secret=b"B" * 32)

    assert first == repeated
    assert first != changed
    all_rows = [row for block in acquisition.BLOCK_ORDER for row in first[block]]
    assert len(all_rows) == len({row.table_uid for row in all_rows}) == 144
    for block in acquisition.BLOCK_ORDER:
        assert len(first[block]) == acquisition.BLOCK_COUNTS[block]
        assert Counter(row.family for row in first[block]) == Counter(
            {
                family: acquisition.PER_FAMILY_QUOTA[block]
                for family in acquisition.FAMILY_ORDER
            }
        )
        grouped = tuple(
            family
            for family in acquisition.FAMILY_ORDER
            for _ in range(acquisition.PER_FAMILY_QUOTA[block])
        )
        assert tuple(row.family for row in first[block]) != grouped


def test_private_views_are_late_field_free_F_has_no_labels_and_M_is_presealed() -> None:
    raw, tagop = _population_payloads()
    qualified = acquisition.qualify_decoded_sources(
        raw_by_split=raw, tagop_by_split=tagop
    )
    selected = acquisition.select_blocks(qualified.candidates, secret=b"A" * 32)
    commitment = acquisition.selection_secret_commitment(b"A" * 32)
    views, labels, ledger = acquisition.materialize_private_payloads(
        selected=selected,
        selection_secret_commitment_sha256=commitment,
    )

    assert set(views) == set(acquisition.BLOCK_ORDER)
    assert set(labels) == {"A_form", "A_hold", "M_search"}
    assert "F_search" not in labels
    assert views["M_search"]["access_state"].startswith("presealed")
    assert labels["M_search"]["access_state"].startswith("presealed")
    for view in views.values():
        acquisition.assert_view_firewall(view)
        assert not acquisition._VIEW_FORBIDDEN_KEYS.intersection(_walk_keys(view))
    for label in labels.values():
        assert set(label["items"][0]) == {
            "ordinal",
            "item_commitment_sha256",
            "family",
            "gold_unit_ids",
        }
    assert not {
        "answer",
        "mapping",
        "answer_sha256",
        "mapping_sha256",
    }.intersection(_walk_keys(ledger))
    drifted = copy.deepcopy(views["A_form"])
    drifted["items"][0]["family"] = "TABLE"
    with pytest.raises(acquisition.TatqaP20AcquisitionError, match="forbidden"):
        acquisition.assert_view_firewall(drifted)


def test_missing_freeze_or_source_receipt_fails_before_source_payload_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acquisition, "_verify_contracts", lambda _root: ({}, {}))
    opened: list[bool] = []

    def forbidden_open(*_args: object, **_kwargs: object) -> dict[str, bytes]:
        opened.append(True)
        raise AssertionError("source payload must remain unopened")

    monkeypatch.setattr(acquisition, "_read_and_verify_source_files", forbidden_open)
    with pytest.raises(acquisition.TatqaP20AcquisitionError, match="freeze"):
        acquisition.run_trusted_acquisition(tmp_path)
    assert opened == []
    assert not (tmp_path / acquisition.ACQUISITION_ROOT_RELATIVE).exists()
    monkeypatch.setattr(
        acquisition, "_verify_freeze", lambda _root: {"self_sha256": "f" * 64}
    )
    with pytest.raises(acquisition.TatqaP20AcquisitionError, match="download receipt"):
        acquisition.run_trusted_acquisition(tmp_path)
    assert opened == []
    assert not (tmp_path / acquisition.ACQUISITION_ROOT_RELATIVE).exists()


def test_public_freeze_wrapper_binds_exact_configured_evidence_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprint = tmp_path / "manifests/fingerprint.json"
    canary = tmp_path / "manifests/canary.json"
    frozen = {
        "runtime_fingerprint_binding": {
            "relative_path": "manifests/fingerprint.json"
        },
        "production_canary_binding": {
            "relative_path": "manifests/canary.json"
        },
    }
    monkeypatch.setattr(acquisition, "_verify_freeze", lambda _root: frozen)

    assert acquisition.verify_implementation_freeze(
        tmp_path,
        runtime_fingerprint_path=fingerprint,
        production_canary_path=canary,
    ) is frozen
    with pytest.raises(acquisition.TatqaP20AcquisitionError, match="evidence path"):
        acquisition.verify_implementation_freeze(
            tmp_path,
            runtime_fingerprint_path=tmp_path / "alternate/fingerprint.json",
            production_canary_path=canary,
        )

def test_formal_wrapper_is_one_shot_and_writes_only_0600_private_packs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(acquisition, "_verify_contracts", lambda _root: ({}, {}))
    monkeypatch.setattr(
        acquisition, "_verify_freeze", lambda _root: {"self_sha256": "1" * 64}
    )
    monkeypatch.setattr(
        acquisition,
        "_verify_source_download_receipt",
        lambda _root: {
            "self_sha256": "2" * 64,
            "implementation_freeze_self_sha256": "1" * 64,
        },
    )
    monkeypatch.setattr(
        acquisition,
        "_read_and_verify_source_files",
        lambda _root, _receipt: _source_bytes(),
    )
    monkeypatch.setattr(acquisition.os, "urandom", lambda size: b"S" * size)

    with tempfile.TemporaryDirectory(prefix="tatqa-p18-acquisition-", dir="/tmp") as value:
        project = Path(value)
        receipt = acquisition.run_trusted_acquisition(project)
        root = project / acquisition.ACQUISITION_ROOT_RELATIVE
        try:
            assert receipt["status"] == "trusted_one_shot_acquisition_complete"
            assert receipt["F_search_label_pack_created"] is False
            assert receipt["M_search_view_and_labels_presealed"] is True
            assert receipt["source_item_or_identifier_persisted_publicly"] is False
            assert not (root / "F_search.labels.sealed.json").exists()
            assert (root / acquisition.LABEL_FILENAMES["M_search"]).is_file()
            expected_private = {
                acquisition.MARKER_FILENAME,
                acquisition.SECRET_FILENAME,
                acquisition.LEDGER_FILENAME,
                acquisition.PUBLIC_RECEIPT_FILENAME,
                *acquisition.VIEW_FILENAMES.values(),
                *acquisition.LABEL_FILENAMES.values(),
            }
            assert {path.name for path in root.iterdir()} == expected_private
            for path in root.iterdir():
                assert stat.S_IMODE(path.stat().st_mode) == 0o600
            public_text = json.dumps(receipt, sort_keys=True)
            assert "question-train" not in public_text
            assert "table-train" not in public_text
            assert "selection_secret" not in _walk_keys(receipt)

            monkeypatch.setattr(
                acquisition,
                "_verify_contracts",
                lambda _root: (_ for _ in ()).throw(
                    AssertionError("one-shot refusal must precede precondition access")
                ),
            )
            with pytest.raises(acquisition.TatqaP20OneShotRefusal):
                acquisition.run_trusted_acquisition(project)
        finally:
            root.chmod(0o700)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    with pytest.raises(acquisition.TatqaP20AcquisitionError, match="duplicate"):
        acquisition.strict_json_loads(b'{"x":1,"x":2}', label="fixture")
    with pytest.raises(acquisition.TatqaP20AcquisitionError, match="NaN"):
        acquisition.strict_json_loads(b'{"x":NaN}', label="fixture")
