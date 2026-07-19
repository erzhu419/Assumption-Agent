from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Iterator

import pytest

from assumption_agent.benchmarks import hybridqa_direct_acquisition_v1 as acquisition


@pytest.fixture
def private_project_root() -> Iterator[Path]:
    # Codex App may point pytest's default tmp root at DrvFS, which cannot
    # represent the mandatory 0600 formal artifacts.  The formal workspace is
    # Linux-native; exercise the same permission semantics when available.
    linux_tmp = Path("/tmp")
    parent = str(linux_tmp) if linux_tmp.is_dir() else None
    with tempfile.TemporaryDirectory(prefix="hybridqa-acquisition-", dir=parent) as value:
        yield Path(value)


def _table(*, table_id: str, answer_cell: str = "answer", links: tuple[str, ...] = ("/wiki/p",)) -> dict[str, object]:
    return {
        "header": [["name", []], ["value", []]],
        "data": [
            [["bridge", list(links)], [answer_cell, []]],
            [["other", []], ["value", []]],
        ],
        "title": f"title {table_id}",
    }


def _row(*, qid: str, table_id: str, answer: str) -> dict[str, str]:
    return {
        "answer-text": answer,
        "question": f"Which value belongs to {qid}",
        "question_id": qid,
        "question_postag": "WDT NN VBZ TO NNP",
        "table_id": table_id,
    }


def _candidate(index: int, family: str) -> acquisition.Candidate:
    table = f"{family}-table-{index}"
    if family == "TABLE_ONLY":
        gold = (acquisition.UnitKey("table_row", table, "0"),)
    elif family == "PASSAGE_ONLY":
        gold = (
            acquisition.UnitKey("linked_passage", table, f"/wiki/p-{index}"),
            acquisition.UnitKey("table_row", table, "0"),
        )
    else:
        gold = (
            acquisition.UnitKey("linked_passage", table, f"/wiki/p-{index}"),
            acquisition.UnitKey("table_row", table, "0"),
            acquisition.UnitKey("table_row", table, "1"),
        )
    return acquisition.Candidate(
        source_ordinal=index,
        question_id=f"{family}-q-{index}",
        table_id=table,
        question=f"Which synthetic value {index}",
        question_postag="WDT JJ NN CD",
        family=family,
        gold_unit_keys=tuple(sorted(gold)),
    )


def _pool() -> tuple[acquisition.Candidate, ...]:
    return tuple(
        _candidate(index, family)
        for family in acquisition.FAMILIES
        for index in range(60)
    )


def _units_for_selected(
    selected: dict[str, tuple[acquisition.Candidate, ...]],
) -> tuple[acquisition.CorpusUnit, ...]:
    units: dict[acquisition.UnitKey, acquisition.CorpusUnit] = {}
    for block in acquisition.BLOCK_ORDER:
        for candidate in selected[block]:
            for key in candidate.gold_unit_keys:
                if key.unit_type == "table_row":
                    unit = acquisition.CorpusUnit(
                        key, "table", "header: value", int(key.local_key), ()
                    )
                else:
                    unit = acquisition.CorpusUnit(
                        key, "passage", "passage body", None, (key.local_key,)
                    )
                units[key] = unit
    index = 0
    while len(units) < 900:
        key = acquisition.UnitKey("table_row", f"distractor-{index}", "0")
        units[key] = acquisition.CorpusUnit(
            key, "distractor", f"body {index}", 0, ()
        )
        index += 1
    return tuple(units.values())


def test_clean_classifier_forms_three_exclusive_locus_families() -> None:
    table = _table(table_id="t")
    request = {"/wiki/p": "the passage contains passageanswer exactly"}

    table_only, status = acquisition.classify_candidate(
        source_ordinal=0,
        row=_row(qid="tq", table_id="t", answer="answer"),
        table=table,
        request=request,
    )
    passage_only, _ = acquisition.classify_candidate(
        source_ordinal=1,
        row=_row(qid="pq", table_id="t", answer="passageanswer"),
        table=table,
        request=request,
    )
    dual_table = _table(table_id="t", answer_cell="dual")
    dual, _ = acquisition.classify_candidate(
        source_ordinal=2,
        row=_row(qid="dq", table_id="t", answer="dual"),
        table=dual_table,
        request={"/wiki/p": "dual appears here"},
    )

    assert status == "eligible"
    assert table_only is not None and table_only.family == "TABLE_ONLY"
    assert len(table_only.gold_unit_keys) == 1
    assert passage_only is not None and passage_only.family == "PASSAGE_ONLY"
    assert len(passage_only.gold_unit_keys) == 2
    assert dual is not None and dual.family == "DUAL_TABLE_PASSAGE"
    assert len(dual.gold_unit_keys) == 2  # answer and bridge coincide in row zero


def test_ambiguous_passage_or_table_locus_is_excluded_not_tie_broken() -> None:
    table = _table(table_id="t", links=("/wiki/p", "/wiki/q"))
    candidate, reason = acquisition.classify_candidate(
        source_ordinal=0,
        row=_row(qid="q", table_id="t", answer="same"),
        table=table,
        request={"/wiki/p": "same", "/wiki/q": "same"},
    )
    assert candidate is None
    assert reason == "ambiguous_answer_passage"

    duplicate = _table(table_id="t", answer_cell="same", links=())
    duplicate["data"][1][1][0] = "same"
    candidate, reason = acquisition.classify_candidate(
        source_ordinal=0,
        row=_row(qid="q", table_id="t", answer="same"),
        table=duplicate,
        request={},
    )
    assert candidate is None
    assert reason == "ambiguous_table_answer_cell"


def test_empty_source_title_uses_table_id_only_for_corpus_projection() -> None:
    table = _table(table_id="stable-table-id")
    table["title"] = "  "
    units = acquisition.decoded_corpus_units(
        table_id="stable-table-id",
        table=table,
        request={"/wiki/p": "passage body"},
    )

    assert units
    assert all(
        unit.title == "stable-table-id"
        for unit in units
        if unit.key.unit_type == "table_row"
    )


def test_empty_referenced_passage_is_safe_for_classification_and_nonempty_for_corpus() -> None:
    table = _table(table_id="t")
    candidate, reason = acquisition.classify_candidate(
        source_ordinal=0,
        row=_row(qid="q", table_id="t", answer="answer"),
        table=table,
        request={"/wiki/p": ""},
    )
    units = acquisition.decoded_corpus_units(
        table_id="t",
        table=table,
        request={"/wiki/p": ""},
    )

    assert reason == "eligible"
    assert candidate is not None and candidate.family == "TABLE_ONLY"
    passage = next(unit for unit in units if unit.key.unit_type == "linked_passage")
    assert passage.title == passage.body == "p"


def test_private_hmac_selection_is_reproducible_disjoint_and_secret_sensitive() -> None:
    pool = _pool()
    first = acquisition.select_blocks(pool, secret=b"A" * 32)
    repeated = acquisition.select_blocks(pool, secret=b"A" * 32)
    second = acquisition.select_blocks(pool, secret=b"B" * 32)

    assert first == repeated
    assert first != second
    rows = [candidate for block in acquisition.BLOCK_ORDER for candidate in first[block]]
    assert len(rows) == 144
    assert len({row.question_id for row in rows}) == 144
    assert len({row.table_id for row in rows}) == 144
    for block in acquisition.BLOCK_ORDER:
        assert len(first[block]) == acquisition.BLOCK_COUNTS[block]
        assert {
            family: sum(row.family == family for row in first[block])
            for family in acquisition.FAMILIES
        } == {
            family: acquisition.PER_FAMILY_QUOTA[block]
            for family in acquisition.FAMILIES
        }


def test_exact_609_corpus_and_private_pack_separation() -> None:
    selected = acquisition.select_blocks(_pool(), secret=b"A" * 32)
    stream = _units_for_selected(selected)
    corpus, mapping = acquisition.form_fixed_corpus(
        selected=selected, unit_stream=stream, secret=b"A" * 32
    )
    repeated, repeated_mapping = acquisition.form_fixed_corpus(
        selected=selected, unit_stream=stream, secret=b"A" * 32
    )
    changed, _ = acquisition.form_fixed_corpus(
        selected=selected, unit_stream=stream, secret=b"B" * 32
    )
    packs = acquisition.form_private_packs(
        selected=selected, corpus=corpus, unit_to_index=mapping
    )

    assert len(corpus) == len(mapping) == acquisition.CORPUS_UNIT_COUNT
    assert corpus == repeated and mapping == repeated_mapping
    assert corpus != changed
    assert "F_search.labels.sealed.json" not in packs
    assert "A_form.labels.sealed.json" in packs
    assert packs["F_search.view.private.json"]["labels_family_gold_or_table_included"] is False
    corpus_pack = packs[acquisition.CORPUS_FILENAME]
    acquisition.verify_self_hash(corpus_pack, "corpus_pack_sha256")
    documents = [f'{unit["title"]}\n\n{unit["body"]}' for unit in corpus_pack["units"]]
    assert corpus_pack["duplicate_text_group_count"] == sum(
        documents.count(document) > 1 for document in set(documents)
    )
    assert corpus_pack["duplicate_text_unit_count"] == sum(
        documents.count(document) for document in set(documents) if documents.count(document) > 1
    )
    assert corpus_pack["duplicate_text_group_count"] > 0
    assert corpus_pack["duplicate_expansion_delegated_to_frozen_official_HippoRAG_adapter"] is True
    assert [
        (unit["title"], unit["body"]) for unit in corpus_pack["units"]
    ] == [(unit.title, unit.body) for unit in corpus]


def test_exact_corpus_handles_forced_digest_and_hash_collisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = acquisition.select_blocks(_pool(), secret=b"A" * 32)
    stream = _units_for_selected(selected)
    monkeypatch.setattr(acquisition, "_hmac_digest", lambda *_args, **_kwargs: b"\0" * 32)
    monkeypatch.setattr(acquisition, "stable_hash", lambda _value: "0" * 64)

    corpus, mapping = acquisition.form_fixed_corpus(
        selected=selected,
        unit_stream=stream,
        secret=b"A" * 32,
    )

    assert len(corpus) == len(mapping) == acquisition.CORPUS_UNIT_COUNT
    assert len({unit.key for unit in corpus}) == acquisition.CORPUS_UNIT_COUNT


def test_source_validation_failure_is_terminal_before_secret(
    private_project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        acquisition,
        "_verify_implementation_freeze",
        lambda _project: {"freeze_sha256": "f" * 64},
    )

    def fail(_project: Path):
        raise RuntimeError("synthetic source failure")

    monkeypatch.setattr(acquisition.source_qualification, "qualify_official_source", fail)
    with pytest.raises(acquisition.HybridQaAcquisitionError, match="terminally"):
        acquisition.run_formal_acquisition(private_project_root)
    root = private_project_root / acquisition.ACQUISITION_RELATIVE
    assert (root / acquisition.FAILURE_FILENAME).is_file()
    assert not (root / acquisition.SECRET_FILENAME).exists()
    with pytest.raises(acquisition.HybridQaAcquisitionError, match="nonreusable"):
        acquisition.run_formal_acquisition(private_project_root)


def test_one_shot_wrapper_writes_complete_packs_and_never_F_labels(
    private_project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = acquisition.select_blocks(_pool(), secret=b"A" * 32)
    stream = _units_for_selected(selected)
    monkeypatch.setattr(
        acquisition,
        "_verify_implementation_freeze",
        lambda _project: {"freeze_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        acquisition.source_qualification,
        "qualify_official_source",
        lambda _project: {"schema": "synthetic_aggregate_receipt"},
    )
    monkeypatch.setattr(acquisition.os, "urandom", lambda _count: b"A" * 32)
    monkeypatch.setattr(acquisition, "_candidate_pool", lambda _project: (_pool(), {}))
    monkeypatch.setattr(acquisition, "_official_unit_stream", lambda _project: iter(stream))

    receipt = acquisition.run_formal_acquisition(private_project_root)
    root = private_project_root / acquisition.ACQUISITION_RELATIVE

    assert receipt["status"] == "formal_acquisition_complete"
    assert (root / acquisition.PUBLIC_FILENAME).is_file()
    assert (root / acquisition.SECRET_FILENAME).stat().st_mode & 0o777 == 0o600
    assert not (root / "F_search.labels.sealed.json").exists()
    assert (root / "M_search.labels.sealed.json").is_file()
    public_text = json.dumps(receipt, sort_keys=True)
    assert "selection_secret_sha256" not in receipt
    assert receipt["selection_secret_persisted_publicly"] is False
    assert len(receipt["selection_secret_commitment_sha256"]) == 64
    assert "Which synthetic value" not in public_text
    assert "-table-" not in public_text
