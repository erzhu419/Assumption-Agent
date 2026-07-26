from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import stat
import tempfile
from typing import Iterator

import pytest

from assumption_agent.benchmarks import maud_extraction_p1_source_v1 as source


SECRET = b"M" * 32


def _title_for_block(block: str) -> str:
    for index in range(10_000):
        title = f"synthetic contract {block} {index}"
        observed = source.assign_block(
            "TRAIN", source.normalize_contract_title(title), SECRET
        )
        if observed == block:
            return title
    raise AssertionError(f"no synthetic title for {block}")


def _qa(
    deal_point_type: str,
    *,
    ordinal: int,
    answers: list[dict[str, object]] | None = None,
    impossible: object | None = None,
) -> dict[str, object]:
    answer_values = [] if answers is None else answers
    impossible_value = not answer_values if impossible is None else impossible
    return {
        "id": f"synthetic-q-{ordinal:02d}",
        "question": source.question_for_type(deal_point_type),
        "answers": answer_values,
        "is_impossible": impossible_value,
        "upstream_extra": {"ignored": [1, True, None]},
    }


def _contract(
    *,
    title: str,
    context: str = "abcdefghijklmno 法律",
    answers_by_type: dict[str, list[dict[str, object]]] | None = None,
    aliases: dict[str, str] | None = None,
    impossible_override_by_type: dict[str, object] | None = None,
) -> dict[str, object]:
    answer_map = answers_by_type or {}
    alias_map = aliases or {}
    impossible_map = impossible_override_by_type or {}
    qas = [
        _qa(
            alias_map.get(deal_point_type, deal_point_type),
            ordinal=index,
            answers=answer_map.get(deal_point_type),
            impossible=impossible_map.get(deal_point_type),
        )
        for index, deal_point_type in enumerate(source.DEAL_POINT_TYPES)
    ]
    return {
        "title": title,
        "paragraphs": [
            {
                "context": context,
                "qas": qas,
                "paragraph_extra": "allowed",
            }
        ],
        "contract_extra": ["allowed", {"nested": True}],
    }


def _document(*contracts: dict[str, object]) -> bytes:
    return json.dumps(
        {
            "version": "2.0",
            "top_level_extra_before": {"nested": [1, 2, 3]},
            "data": list(contracts),
            "top_level_extra_after": "allowed",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


@pytest.fixture
def linux_private_directory() -> Iterator[Path]:
    parent = "/tmp" if Path("/tmp").is_dir() else None
    with tempfile.TemporaryDirectory(
        prefix="maud-source-v1-", dir=parent
    ) as value:
        yield Path(value)


def test_frozen_public_registry_aliases_and_exposure_denylist() -> None:
    assert source.VERSION == "maud_extraction_p1_source_v1"
    assert len(source.DEAL_POINT_TYPES) == len(set(source.DEAL_POINT_TYPES)) == 22
    assert len(source.TYPE_ALIASES) == 2
    assert len(source.EXCLUDED_NORMALIZED_TITLE_SHA256S) == 16
    assert source.question_for_type("No-Shop") == (
        'Highlight the parts of the text (if any) related to "No-Shop" '
        "that should be reviewed by a lawyer."
    )
    assert source.contract_title_sha256("  CONTRACT_3\n") == (
        "fcf2822d878e9b74a8fba51c92e5326ca152989cad7e2239654a462658be08a1"
    )

    canonical, family = source.deal_point_type_and_family(
        source.question_for_type("Fiduciary exception to COR convent")
    )
    assert canonical == "Fiduciary exception to COR covenant"
    assert family == "protection_exception_remedy"
    canonical, family = source.deal_point_type_and_family(
        source.question_for_type("Negative interim operating convenant")
    )
    assert canonical == "Negative interim operating covenant"
    assert family == "condition_obligation"
    curly = source.question_for_type("No-Shop").replace(
        '"No-Shop"', "“No-Shop”"
    )
    assert source.deal_point_type_and_family(curly) == (
        "No-Shop",
        "protection_exception_remedy",
    )
    assert source.deal_point_type_and_family(curly[:-1]) == (
        "No-Shop",
        "protection_exception_remedy",
    )
    with pytest.raises(source.MaudSourceError, match="frozen template"):
        source.deal_point_type_and_family(
            source.question_for_type("No-Shop").replace(
                "text (if any) related", "text, if any, related"
            )
        )
    with pytest.raises(source.MaudSourceError, match="not quoted"):
        source.deal_point_type_and_family(
            source.question_for_type("No-Shop").replace(
                '"No-Shop"', "'No-Shop'"
            )
        )


def test_source_boundary_imports_only_the_python_standard_library() -> None:
    tree = ast.parse(Path(source.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert imported == {
        "__future__",
        "codecs",
        "contextlib",
        "dataclasses",
        "hashlib",
        "hmac",
        "json",
        "os",
        "pathlib",
        "re",
        "stat",
        "typing",
        "unicodedata",
    }


def test_streams_utf8_extra_fields_and_preserves_all_exact_answers() -> None:
    deal_type = source.DEAL_POINT_TYPES[0]
    contract = _contract(
        title="untouched synthetic dev contract",
        answers_by_type={
            deal_type: [
                {"text": "cdef", "answer_start": 2},
                {"text": "cdef", "answer_start": 2},
                {"text": "fghi", "answer_start": 5},
                {"text": "jkl", "answer_start": 9},
            ]
        },
    )
    prepared = source.parse_split(
        io.BytesIO(_document(contract)),
        split="DEV",
        selection_secret=SECRET,
        stream_chunk_size=3,
    )

    assert prepared.source_contract_count == 1
    assert prepared.excluded_contract_count == 0
    assert len(prepared.contracts) == 1
    parsed_contract = prepared.contracts[0]
    assert parsed_contract.block == "A_hold"
    assert parsed_contract.context == "abcdefghijklmno 法律"
    assert len(parsed_contract.items) == 22
    item = parsed_contract.items[0]
    assert item.deal_point_type == deal_type
    assert [(span.start, span.end, span.text) for span in item.spans or ()] == [
        (2, 6, "cdef"),
        (5, 9, "fghi"),
        (9, 12, "jkl"),
    ]
    assert item.merged_intervals == ((2, 12),)
    assert all(value.gold_semantically_opened for value in parsed_contract.items)


def test_whole_contract_exclusion_precedes_schema_and_partition() -> None:
    excluded = {
        "title": " \nCONTRACT_3\t",
        # Structurally valid but deliberately not a SQuAD paragraph.  It must
        # remain uninterpreted because the whole title lineage is excluded.
        "paragraphs": {
            "private_primary_lineage": "MUST_NOT_BE_PARSED_OR_REPLACED"
        },
    }
    retained = _contract(title="retained dev contract")
    prepared = source.parse_split(
        io.BytesIO(_document(excluded, retained)),
        split="DEV",
        selection_secret=SECRET,
        stream_chunk_size=5,
    )

    assert prepared.source_contract_count == 2
    assert prepared.excluded_contract_count == 1
    assert len(prepared.contracts) == 1
    assert prepared.contracts[0].block == "A_hold"


def test_train_contract_hmac_is_disjoint_and_f_gold_is_never_decoded() -> None:
    a_title = _title_for_block("A_form")
    f_title = _title_for_block("F_search")
    first_type = source.DEAL_POINT_TYPES[0]
    a_contract = _contract(
        title=a_title,
        answers_by_type={
            first_type: [{"text": "abc", "answer_start": 0}]
        },
    )
    f_contract = _contract(
        title=f_title,
        # Deliberately non-SQuAD answer semantics.  Structural JSON traversal
        # succeeds, proving F_search never decodes answer text or answerability.
        answers_by_type={
            deal_point_type: [
                {
                    "text": "F_GOLD_MUST_NEVER_BE_SEMANTICALLY_DECODED",
                    "answer_start": -999,
                }
            ]
            for deal_point_type in source.DEAL_POINT_TYPES
        },
        impossible_override_by_type={
            deal_point_type: "PRIVATE_F_ANSWERABILITY_SENTINEL"
            for deal_point_type in source.DEAL_POINT_TYPES
        },
    )

    prepared = source.parse_split(
        io.BytesIO(_document(a_contract, f_contract)),
        split="TRAIN",
        selection_secret=SECRET,
        stream_chunk_size=7,
    )
    assert [contract.block for contract in prepared.contracts] == [
        "A_form",
        "F_search",
    ]
    assert all(
        item.spans is not None
        for item in prepared.contracts_for("A_form")[0].items
    )
    assert all(
        item.spans is None and item.merged_intervals is None
        for item in prepared.contracts_for("F_search")[0].items
    )
    with pytest.raises(source.MaudSourceError, match="forbidden"):
        prepared.gold_pack("F_search")

    repeated = source.parse_split(
        io.BytesIO(_document(a_contract, f_contract)),
        split="TRAIN",
        selection_secret=SECRET,
        stream_chunk_size=19,
    )
    changed = source.parse_split(
        io.BytesIO(_document(a_contract, _contract(title=f_title))),
        split="TRAIN",
        selection_secret=b"N" * 32,
    )
    assert [
        (value.work_id, value.block) for value in prepared.contracts
    ] == [(value.work_id, value.block) for value in repeated.contracts]
    assert [value.work_id for value in prepared.contracts] != [
        value.work_id for value in changed.contracts
    ]


def test_action_projection_contains_no_gold_or_source_identity() -> None:
    first_type = source.DEAL_POINT_TYPES[0]
    prepared = source.parse_split(
        io.BytesIO(
            _document(
                _contract(
                    title="action projection dev contract",
                    answers_by_type={
                        first_type: [
                            {"text": "abc", "answer_start": 0}
                        ]
                    },
                )
            )
        ),
        split="DEV",
        selection_secret=SECRET,
    )
    action = prepared.action_view("A_hold")
    serialized = json.dumps(action, ensure_ascii=False, sort_keys=True)

    assert action["answerability_gold_text_offset_or_span_included"] is False
    assert "answer_start" not in serialized
    assert '"answers"' not in serialized
    assert '"spans"' not in serialized
    assert '"title"' not in serialized
    assert "synthetic-q-" not in serialized
    assert action["item_count"] == 22


def test_alias_is_canonicalized_but_question_is_preserved() -> None:
    canonical = "Fiduciary exception to COR covenant"
    alias = "Fiduciary exception to COR convent"
    prepared = source.parse_split(
        io.BytesIO(
            _document(
                _contract(
                    title="alias dev contract",
                    aliases={canonical: alias},
                )
            )
        ),
        split="DEV",
        selection_secret=SECRET,
    )
    item = next(
        value
        for value in prepared.contracts[0].items
        if value.deal_point_type == canonical
    )
    assert item.question == source.question_for_type(alias)
    assert item.family == "protection_exception_remedy"


def test_exact_offset_and_registry_drift_are_terminal() -> None:
    first_type = source.DEAL_POINT_TYPES[0]
    bad_offset = _contract(
        title="bad offset dev contract",
        answers_by_type={
            first_type: [{"text": "abc", "answer_start": 1}]
        },
    )
    with pytest.raises(source.MaudSourceError, match="exact context offsets"):
        source.parse_split(
            io.BytesIO(_document(bad_offset)),
            split="DEV",
            selection_secret=SECRET,
        )

    duplicate_type = _contract(title="duplicate type dev contract")
    qas = duplicate_type["paragraphs"][0]["qas"]
    qas[-1]["question"] = qas[0]["question"]
    with pytest.raises(source.MaudSourceError, match="duplicate deal-point"):
        source.parse_split(
            io.BytesIO(_document(duplicate_type)),
            split="DEV",
            selection_secret=SECRET,
        )


class _UnreadableUntilAuthorized:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.read_count = 0
        self.stream = io.BytesIO(raw)

    def read(self, size: int = -1) -> bytes:
        self.read_count += 1
        return self.stream.read(size)


def test_test_parse_capability_is_checked_before_any_source_read() -> None:
    reader = _UnreadableUntilAuthorized(
        _document(_contract(title="untouched test contract"))
    )
    with pytest.raises(source.MaudSourceError, match="explicit"):
        source.parse_split(
            reader,
            split="TEST",
            selection_secret=SECRET,
        )
    assert reader.read_count == 0

    bad = source.TestParseCapability(
        a_hold_promotion_receipt_sha256="not-a-hash"
    )
    with pytest.raises(source.MaudSourceError, match="receipt"):
        source.parse_split(
            reader,
            split="TEST",
            selection_secret=SECRET,
            test_parse_capability=bad,
        )
    assert reader.read_count == 0

    authorized_reader = _UnreadableUntilAuthorized(
        _document(_contract(title="untouched test contract"))
    )
    prepared = source.parse_split(
        authorized_reader,
        split="TEST",
        selection_secret=SECRET,
        test_parse_capability=source.TestParseCapability(
            a_hold_promotion_receipt_sha256="a" * 64
        ),
        stream_chunk_size=11,
    )
    assert authorized_reader.read_count > 0
    assert prepared.contracts[0].block == "M_search"


def test_private_gold_pack_is_self_hashed_exclusive_and_mode_0600(
    linux_private_directory: Path,
) -> None:
    first_type = source.DEAL_POINT_TYPES[0]
    prepared = source.parse_split(
        io.BytesIO(
            _document(
                _contract(
                    title="gold pack dev contract",
                    answers_by_type={
                        first_type: [
                            {"text": "abc", "answer_start": 0}
                        ]
                    },
                )
            )
        ),
        split="DEV",
        selection_secret=SECRET,
    )
    gold = prepared.gold_pack("A_hold")
    destination = linux_private_directory / "A_hold.gold.private.json"
    file_sha256 = source.write_gold_pack_exclusive(destination, gold)

    assert len(file_sha256) == 64
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert json.loads(destination.read_text(encoding="ascii")) == gold
    with pytest.raises(source.MaudSourceError, match="exclusive"):
        source.write_gold_pack_exclusive(destination, gold)

    forged = dict(gold)
    forged["block"] = "F_search"
    forged_body = dict(forged)
    forged_body.pop("gold_pack_sha256")
    forged["gold_pack_sha256"] = source.stable_hash(forged_body)
    with pytest.raises(source.MaudSourceError, match="forbidden"):
        source.write_gold_pack_exclusive(
            linux_private_directory / "F_search.gold.json", forged
        )


def test_duplicate_ids_unknown_types_and_malformed_stream_are_rejected() -> None:
    duplicate_id = _contract(title="duplicate ID dev contract")
    qas = duplicate_id["paragraphs"][0]["qas"]
    qas[-1]["id"] = qas[0]["id"]
    with pytest.raises(source.MaudSourceError, match="duplicate QA IDs"):
        source.parse_split(
            io.BytesIO(_document(duplicate_id)),
            split="DEV",
            selection_secret=SECRET,
        )

    unknown = _contract(title="unknown type dev contract")
    unknown["paragraphs"][0]["qas"][0]["question"] = (
        source.question_for_type(source.DEAL_POINT_TYPES[0]).replace(
            source.DEAL_POINT_TYPES[0], "A type not in the registry"
        )
    )
    with pytest.raises(source.MaudSourceError, match="unknown deal-point"):
        source.parse_split(
            io.BytesIO(_document(unknown)),
            split="DEV",
            selection_secret=SECRET,
        )

    with pytest.raises(source.MaudSourceError, match="ended unexpectedly"):
        source.parse_split(
            io.BytesIO(b'{"data":[{"title":"truncated"'),
            split="DEV",
            selection_secret=SECRET,
            stream_chunk_size=2,
        )
