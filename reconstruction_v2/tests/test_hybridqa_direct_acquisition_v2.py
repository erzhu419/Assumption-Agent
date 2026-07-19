from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterator

import pytest

from assumption_agent.benchmarks import hybridqa_direct_acquisition_v2 as acquisition


def test_v2_identity_and_reserved_freeze_registry() -> None:
    assert acquisition.VERSION == "hybridqa_direct_acquisition_v2"
    assert acquisition.DESIGN_SHA256 == (
        "4840a4471b0b51909d7e3568eb10dccc4ebb59e2bfebcad964b14789db1d94ad"
    )
    assert acquisition.DEV_RELATIVE.as_posix() == "released_data/dev.json"
    assert acquisition.FORMAL_ROOT_RELATIVE.as_posix() == (
        "artifacts/hybridqa_p6_e2_formal_v2"
    )
    required = set(acquisition.IMPLEMENTATION_FREEZE_REQUIRED_PATHS)
    assert {
        "assumption_agent/benchmarks/hybridqa_direct_acquisition_v2.py",
        "assumption_agent/benchmarks/hybridqa_isolated_bootstrap_v2.py",
        "assumption_agent/benchmarks/hybridqa_local_runtime_v2.py",
        "assumption_agent/benchmarks/hybridqa_p6_e2_formal_controller_v2.py",
        "tests/test_hybridqa_direct_acquisition_v2.py",
        "tests/test_hybridqa_isolated_bootstrap_v2.py",
        "tests/test_hybridqa_local_runtime_v2.py",
        "tests/test_hybridqa_p6_e2_formal_controller_v2.py",
    }.issubset(required)
    assert tuple(sorted(required)) == acquisition.IMPLEMENTATION_FREEZE_REQUIRED_PATHS


def _write_synthetic_freeze(
    project: Path, *, paths: tuple[str, ...]
) -> dict[str, object]:
    files = [
        {
            "relative_path": relative,
            "sha256": hashlib.sha256((project / relative).read_bytes()).hexdigest(),
        }
        for relative in paths
    ]
    body: dict[str, object] = {
        "schema": "hybridqa_p6_e2_implementation_freeze_v2",
        "version": "v2",
        "status": "implementation_frozen",
        "design_sha256": acquisition.DESIGN_SHA256,
        "required_path_registry_sha256": acquisition.stable_hash(list(paths)),
        "implementation_file_count": len(paths),
        "freeze_semantics": acquisition.IMPLEMENTATION_FREEZE_SEMANTICS,
        "files": files,
    }
    value = acquisition.self_hashed(body, "freeze_sha256")
    destination = project / acquisition.IMPLEMENTATION_FREEZE_RELATIVE
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(acquisition._canonical_bytes(value, newline=True))
    return value


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


def test_implementation_freeze_requires_exact_sorted_registry(
    private_project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    member = private_project_root / "frozen_member.py"
    member.write_text("VALUE = 1\n", encoding="ascii")
    paths = ("frozen_member.py",)
    monkeypatch.setattr(acquisition, "IMPLEMENTATION_FREEZE_REQUIRED_PATHS", paths)
    expected = _write_synthetic_freeze(private_project_root, paths=paths)
    assert acquisition._verify_implementation_freeze(private_project_root) == expected

    duplicated = ("frozen_member.py", "frozen_member.py")
    _write_synthetic_freeze(private_project_root, paths=duplicated)
    monkeypatch.setattr(
        acquisition, "IMPLEMENTATION_FREEZE_REQUIRED_PATHS", paths
    )
    with pytest.raises(acquisition.HybridQaAcquisitionError, match="contract drifted"):
        acquisition._verify_implementation_freeze(private_project_root)


def test_implementation_freeze_rejects_symlink_member(
    private_project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = private_project_root / "target.py"
    target.write_text("VALUE = 1\n", encoding="ascii")
    link = private_project_root / "frozen_member.py"
    link.symlink_to(target)
    paths = ("frozen_member.py",)
    monkeypatch.setattr(acquisition, "IMPLEMENTATION_FREEZE_REQUIRED_PATHS", paths)
    _write_synthetic_freeze(private_project_root, paths=paths)
    with pytest.raises(acquisition.HybridQaAcquisitionError, match="symlink"):
        acquisition._verify_implementation_freeze(private_project_root)


def test_item_commitment_binds_question_postag_and_ordinal() -> None:
    common = {
        "block": "A_form",
        "question": "Which synthetic value",
    }
    first = acquisition.item_commitment(
        **common, ordinal=0, question_postag="WDT JJ NN"
    )
    assert first != acquisition.item_commitment(
        **common, ordinal=0, question_postag="WDT NN VB"
    )
    assert first != acquisition.item_commitment(
        **common, ordinal=1, question_postag="WDT JJ NN"
    )


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


def test_v1_public_boundary_is_strict_and_never_reads_private_packs(
    private_project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedded = {"schema": "synthetic_qualification"}
    public_body: dict[str, object] = {
        "schema": "hybridqa_direct_acquisition_v1_public_receipt",
        "version": "hybridqa_direct_acquisition_v1",
        "status": "formal_acquisition_complete",
        "design_sha256": (
            "028f6a58b4e7809e6165cc04e1356aa1b7904dfbe3a8bee18e92ecf00360de34"
        ),
        "implementation_freeze_sha256": "1" * 64,
        "source_qualification_receipt": embedded,
        "selection_secret_commitment_sha256": "2" * 64,
        "selection_secret_persisted_publicly": False,
        "candidate_counts_by_family": {},
        "typed_exclusion_counts": {},
        "block_counts": dict(acquisition.BLOCK_COUNTS),
        "per_family_quota": dict(acquisition.PER_FAMILY_QUOTA),
        "selected_question_count": 144,
        "selected_table_count": 144,
        "question_and_table_disjoint": True,
        "corpus_unit_count": acquisition.CORPUS_UNIT_COUNT,
        "corpus_unit_type_counts": {},
        "private_pack_file_sha256s": {},
        "F_search_label_pack_created": False,
        "raw_question_answer_table_or_unit_identity_persisted_publicly": False,
        "online_evaluator_calls": 0,
        "retry_replay_or_resample": 0,
    }
    public = acquisition.self_hashed(
        public_body, "acquisition_receipt_sha256"
    )
    public_raw = acquisition._canonical_bytes(public, newline=True)

    failure_body: dict[str, object] = {
        "schema": "hybridqa_p6_e2_formal_controller_v1_terminal_failure",
        "version": "hybridqa_p6_e2_formal_controller_v1",
        "status": "terminal_no_retry_replay_resample_or_threshold_change",
        "failure_stage": "durable_initial_label_free_archives",
        "exception_class": "HybridQaFormalControllerError",
        "exception_message_sha256": "3" * 64,
        "item_level_result_or_raw_content_persisted_publicly": False,
        "online_evaluator_calls": 0,
    }
    failure = acquisition.self_hashed(failure_body, "failure_sha256")
    failure_raw = acquisition._canonical_bytes(failure, newline=True)

    disposition_body: dict[str, object] = {
        "schema": (
            "hybridqa_p6_e2_formal_v1_implementation_failure_disposition_v1"
        ),
        "version": "v1",
        "status": "implementation_invalid_efficacy_unknown_terminal",
        "recorded_date": "2026-07-19",
        "formal_identity": {
            "acquisition_receipt_sha256": public[
                "acquisition_receipt_sha256"
            ],
            "terminal_failure_sha256": failure["failure_sha256"],
            "acquisition_public_file_sha256": hashlib.sha256(
                public_raw
            ).hexdigest(),
            "terminal_failure_file_sha256": hashlib.sha256(
                failure_raw
            ).hexdigest(),
        },
        "failure_boundary": {
            "late_label_pack_opens": 0,
            "performance_or_efficacy_claim_authorized": False,
        },
        "root_cause": {
            "class": "deterministic_archive_verifier_implementation_bug"
        },
        "terminal_disposition": {
            "current_root_and_144_item_TRAIN_cohort_reusable": False,
            "post_hoc_scoring_salvage_or_label_open_authorized": False,
            "retry_replay_resample_or_secret_rotation_authorized": False,
        },
        "next_authorized_study": {
            "source_epoch": (
                "official_DEV_only_independent_from_the_failed_TRAIN_cohort"
            ),
            "official_TEST_opened_or_used": False,
            "v1_TRAIN_rows_or_private_item_contents_reopened": False,
        },
    }
    disposition = acquisition.self_hashed(
        disposition_body, "disposition_sha256"
    )

    for relative, raw in (
        (acquisition.V1_PUBLIC_ACQUISITION_RELATIVE, public_raw),
        (acquisition.V1_TERMINAL_FAILURE_RELATIVE, failure_raw),
        (
            acquisition.V1_FAILURE_DISPOSITION_RELATIVE,
            acquisition._canonical_bytes(disposition, newline=True),
        ),
    ):
        path = private_project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    monkeypatch.setattr(
        acquisition,
        "V1_ACQUISITION_RECEIPT_SHA256",
        public["acquisition_receipt_sha256"],
    )
    monkeypatch.setattr(
        acquisition,
        "V1_TERMINAL_FAILURE_SHA256",
        failure["failure_sha256"],
    )
    monkeypatch.setattr(
        acquisition,
        "V1_FAILURE_DISPOSITION_SHA256",
        disposition["disposition_sha256"],
    )
    monkeypatch.setattr(acquisition, "_qualification_code_sha256", lambda _p: "4" * 64)
    monkeypatch.setattr(
        acquisition.source_qualification,
        "verify_qualification_receipt",
        lambda receipt, *, expected_qualification_code_sha256: {
            **receipt,
            "source_custody": {"synthetic": True},
        },
    )
    monkeypatch.setattr(
        acquisition, "_verify_current_source_custody", lambda *_args: None
    )
    observed_paths: list[str] = []
    original_reader = acquisition._read_frozen_regular

    def recording_reader(project: Path, relative: str, *, label: str) -> bytes:
        observed_paths.append(relative)
        return original_reader(project, relative, label=label)

    monkeypatch.setattr(acquisition, "_read_frozen_regular", recording_reader)
    receipt = acquisition._verify_v1_public_source_boundary(private_project_root)

    assert receipt["v1_acquisition_receipt_sha256"] == public[
        "acquisition_receipt_sha256"
    ]
    assert set(observed_paths) == {
        acquisition.V1_PUBLIC_ACQUISITION_RELATIVE.as_posix(),
        acquisition.V1_TERMINAL_FAILURE_RELATIVE.as_posix(),
        acquisition.V1_FAILURE_DISPOSITION_RELATIVE.as_posix(),
    }
    assert not any("private" in relative for relative in observed_paths)


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

    monkeypatch.setattr(acquisition, "_verify_v1_public_source_boundary", fail)
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
        acquisition,
        "_verify_v1_public_source_boundary",
        lambda _project: {
            "source_qualification_receipt": {
                "schema": "synthetic_aggregate_receipt"
            },
            "v1_acquisition_receipt_sha256": (
                acquisition.V1_ACQUISITION_RECEIPT_SHA256
            ),
            "v1_terminal_failure_sha256": (
                acquisition.V1_TERMINAL_FAILURE_SHA256
            ),
            "v1_failure_disposition_sha256": (
                acquisition.V1_FAILURE_DISPOSITION_SHA256
            ),
        },
    )
    monkeypatch.setattr(
        acquisition, "_verify_current_source_custody", lambda *_args: None
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
    assert receipt["selection_split"] == "official_DEV_only"
    assert receipt["v1_TRAIN_rows_or_private_packs_opened"] is False
    assert receipt["predecessor_cohort_replay_or_salvage"] is False
    assert len(receipt["selection_secret_commitment_sha256"]) == 64
    assert "Which synthetic value" not in public_text
    assert "-table-" not in public_text
