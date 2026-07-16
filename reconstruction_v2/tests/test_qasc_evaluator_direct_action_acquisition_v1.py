from __future__ import annotations

from collections import Counter
import hashlib
import hmac
import json
import math
from pathlib import Path
import tarfile
from typing import Any

import pytest

from assumption_agent.benchmarks import (
    qasc_evaluator_direct_action_acquisition_v1 as study,
)
from assumption_agent.models import stable_hash


def _row(
    index: int,
    *,
    member: str = "TRAIN",
    formatted_question: str | None = None,
    stem: str | None = None,
    fact1: str | None = None,
    fact2: str | None = None,
    answer_key: str = "A",
) -> dict[str, Any]:
    choices = [
        {"label": chr(ord("A") + choice), "text": f"choice {index} {choice}"}
        for choice in range(8)
    ]
    return {
        "id": f"{member.lower()}-{index:06d}",
        "question": {
            "stem": stem or f"Which synthetic result applies to item {index}?",
            "choices": choices,
        },
        "answerKey": answer_key,
        "fact1": fact1 or f"Synthetic fact one for item {index}.",
        "fact2": fact2 or f"Synthetic fact two for item {index}.",
        "combinedfact": f"Synthetic combined fact for item {index}.",
        "formatted_question": formatted_question
        or f"Which synthetic result applies to item {index}?",
    }


def _candidate(index: int, *, member: str = "TRAIN", **kwargs: Any) -> study.Candidate:
    parsed = study._parse_candidate(
        _row(index, member=member, **kwargs),
        source_member=member,
        source_row_ordinal=index,
    )
    assert parsed is not None
    return parsed


def _statistics(texts: list[str]) -> study.CorpusStatistics:
    token_rows = [study.tokenize(text) for text in texts]
    df: Counter[str] = Counter()
    for tokens in token_rows:
        df.update(set(tokens))
    return study.CorpusStatistics(
        raw_line_count=len(texts),
        eligible_document_count=len(texts),
        total_token_count=sum(len(tokens) for tokens in token_rows),
        average_document_length=(
            sum(len(tokens) for tokens in token_rows) / len(texts)
        ),
        document_frequency=dict(df),
        chunk_count=1,
    )


def _distractors(prefix: str = "d") -> tuple[study.BM25Candidate, ...]:
    return tuple(
        study.BM25Candidate(
            score_int=1000 - index,
            normalized_fact=study.normalize_text(
                f"{prefix} distractor fact {index} with common signal."
            ),
            exact_fact=f"{prefix} distractor fact {index} with common signal.",
            source_ordinal=index,
        )
        for index in range(study.HARD_DISTRACTOR_COUNT)
    )


def test_normalization_tokenizer_and_query_are_exact() -> None:
    assert study.normalize_text("  Ｃafé\tStraße １２ ") == "café strasse 12"
    assert study.tokenize("A_b—Ｃafé,１２!") == ("a", "b", "café", "12")
    choices = (("A", "alpha"), ("B", "beta"))
    assert study.canonical_query("Question?", choices) == (
        "Question? [CHOICES] A: alpha B: beta"
    )


def test_collision_uses_stem_while_identity_uses_formatted_question_and_never_answer_key() -> None:
    first = _candidate(1, answer_key="A")
    second_raw = _row(1, answer_key="B")
    second = study._parse_candidate(
        second_raw, source_member="TRAIN", source_row_ordinal=1
    )
    assert second is not None
    assert first.identity_commitment_sha256 == second.identity_commitment_sha256
    assert first.label_free_row_sha256 == second.label_free_row_sha256
    changed = _candidate(1, formatted_question="A changed formatted question.")
    assert changed.normalized_question_sha256 == first.normalized_question_sha256
    assert changed.identity_commitment_sha256 != first.identity_commitment_sha256
    changed_stem = _candidate(1, stem="A changed collision stem.")
    assert changed_stem.normalized_question_sha256 != first.normalized_question_sha256
    assert changed_stem.identity_commitment_sha256 == first.identity_commitment_sha256


@pytest.mark.parametrize(
    "mutation",
    (
        lambda row: row["question"]["choices"].__setitem__(
            1, {"label": "B", "text": row["question"]["choices"][0]["text"]}
        ),
        lambda row: row.update(fact2=row["fact1"].upper()),
        lambda row: row.update(formatted_question=""),
        lambda row: row["question"].update(choices=row["question"]["choices"][:7]),
    ),
)
def test_candidate_parser_fails_closed_on_schema_drift(mutation: Any) -> None:
    row = _row(2)
    mutation(row)
    assert (
        study._parse_candidate(
            row, source_member="TRAIN", source_row_ordinal=2
        )
        is None
    )


@pytest.mark.parametrize("answer_key", ("Z", None, {"opaque": "value"}))
def test_answer_key_never_changes_candidate_eligibility_identity_or_hmac(
    answer_key: object,
) -> None:
    valid = _candidate(7, answer_key="A")
    raw = _row(7)
    raw["answerKey"] = answer_key
    opaque = study._parse_candidate(
        raw, source_member="TRAIN", source_row_ordinal=7
    )
    assert opaque is not None
    assert opaque.identity_commitment_sha256 == valid.identity_commitment_sha256
    assert opaque.label_free_row_sha256 == valid.label_free_row_sha256
    secret = bytes(range(32))
    assert study._selection_digest(opaque, block="A_form", secret=secret) == (
        study._selection_digest(valid, block="A_form", secret=secret)
    )


def test_selection_hmac_matches_frozen_message() -> None:
    row = _candidate(3)
    secret = bytes(range(32))
    expected = hmac.new(
        secret,
        (
            f"{study.SELECTION_DOMAIN_SEPARATOR}\0select\0A_form\0TRAIN\0"
            f"{row.identity_commitment_sha256}"
        ).encode(),
        hashlib.sha256,
    ).digest()
    assert study._selection_digest(row, block="A_form", secret=secret) == expected


def test_selection_enforces_collisions_groups_and_fact2_boundaries() -> None:
    candidates: list[study.Candidate] = []
    for index in range(260):
        candidates.append(
            _candidate(
                1000 + index,
                member="TRAIN",
                fact1=f"train fact1 group {index}",
                fact2=f"train fact2 group {index % 190}",
            )
        )
    for index in range(110):
        fact2 = (
            f"train fact2 group {index}"
            if index < 20
            else f"dev fact2 group {index}"
        )
        candidates.append(
            _candidate(
                5000 + index,
                member="DEV",
                fact1=f"dev fact1 group {index}",
                fact2=fact2,
            )
        )
    collision_question = "An exact stem collision."
    collision_a = _candidate(9001, stem=collision_question)
    collision_b = _candidate(
        9002, member="DEV", stem=collision_question.upper()
    )
    candidates.extend((collision_a, collision_b))
    selected, stats = study._select_candidates(
        candidates, secret=b"s" * 32, enforce_formal_counts=False
    )
    assert {block: len(rows) for block, rows in selected.items()} == study.BLOCK_COUNTS
    selected_ids = {
        row.item_id for block in study.BLOCK_ORDER for row in selected[block]
    }
    assert collision_a.item_id not in selected_ids
    assert collision_b.item_id not in selected_ids
    train = [
        row
        for block in ("A_form", "F_search", "A_hold")
        for row in selected[block]
    ]
    assert len({row.normalized_fact1 for row in train}) == 192
    formation_fact2 = {
        row.normalized_fact2
        for block in ("A_form", "F_search")
        for row in selected[block]
    }
    assert not formation_fact2 & {
        row.normalized_fact2 for row in selected["A_hold"]
    }
    assert not {row.normalized_fact2 for row in train} & {
        row.normalized_fact2 for row in selected["M_search"]
    }
    assert len({row.normalized_fact1 for row in selected["M_search"]}) == 64
    assert stats["preselection_exclusions"]["TRAIN"][
        "normalized_question_collision"
    ] == 1
    assert stats["preselection_exclusions"]["DEV"][
        "normalized_question_collision"
    ] == 1


def test_bm25_scalar_matches_frozen_formula_and_ties() -> None:
    query = ("alpha", "common")
    document = ("common", "common", "beta")
    df = {"alpha": 2, "common": 7}
    observed = study.bm25_score_int(
        query_tokens=query,
        document_tokens=document,
        document_count=10,
        total_token_count=40,
        document_frequency=df,
    )
    avgdl = 4.0
    expected_float = math.log1p((10 - 7 + 0.5) / (7 + 0.5)) * 2 * 2.2 / (
        2 + 1.2 * (0.25 + 0.75 * 3 / avgdl)
    )
    assert observed == round(expected_float * 1_000_000_000_000)
    rows: dict[str, study.BM25Candidate] = {}
    later = study.BM25Candidate(10, "zeta", "zeta", 2)
    earlier = study.BM25Candidate(10, "alpha", "alpha", 9)
    study._update_top_unique(rows, later, limit=1)
    study._update_top_unique(rows, earlier, limit=1)
    assert tuple(rows) == ("alpha",)


def test_batched_common_token_scores_are_exact_and_repeatable() -> None:
    receipt = study.run_synthetic_bm25_batch_diagnostic()
    assert receipt["repeat_exact"] is True
    assert receipt["scalar_equality_checked_pair_count"] == 32768
    assert receipt["common_token_present_in_every_document_and_query"] is True


def test_byte_chunks_preserve_exact_global_line_ordinals(tmp_path: Path) -> None:
    lines = [f"line {index} common signal\n" for index in range(113)]
    path = tmp_path / "corpus.txt"
    path.write_text("".join(lines), encoding="utf-8")
    chunks = study._byte_chunks(path.stat().st_size, 7)
    pass1 = [
        study._pass1_worker((str(path), start, end, ("common",)))
        for start, end in chunks
    ]
    offsets: list[int] = []
    running = 0
    for row in pass1:
        offsets.append(running)
        running += row["raw_line_count"]
    observed: list[tuple[int, str]] = []
    for chunk_index, (start, end) in enumerate(chunks):
        observed.extend(
            (offsets[chunk_index] + local, text)
            for local, text, _tokens in study._iter_chunk_lines(
                str(path), start, end
            )
        )
    assert [ordinal for ordinal, _text in observed] == list(range(113))
    assert [text for _ordinal, text in observed] == [line.rstrip("\n") for line in lines]


def test_distributed_topk_retains_frontier_for_exact_global_certificate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.txt"
    lines = [f"common signal unique{index}\n" for index in range(90)]
    path.write_text("".join(lines), encoding="utf-8")
    chunks = study._byte_chunks(path.stat().st_size, 3)
    union = ("common", "signal")
    pass1 = [
        study._pass1_worker((str(path), start, end, union))
        for start, end in chunks
    ]
    raw_counts = [row["raw_line_count"] for row in pass1]
    offsets: list[int] = []
    running = 0
    for count in raw_counts:
        offsets.append(running)
        running += count
    df: Counter[str] = Counter()
    for row in pass1:
        df.update(row["document_frequency"])
    arguments = [
        (
            str(path),
            start,
            end,
            offsets[index],
            ((0, union, ("gold one", "gold two")),),
            90,
            sum(row["total_token_count"] for row in pass1),
            dict(df),
        )
        for index, (start, end) in enumerate(chunks)
    ]
    worker_rows = [study._pass2_worker(row) for row in arguments]
    for result, count in zip(worker_rows, raw_counts, strict=True):
        assert len(result["retained"][0]) == min(
            study.LOCAL_DISTRIBUTED_RETAIN_COUNT, count
        )
        assert (result["frontier"][0] is not None) == (
            count > study.LOCAL_DISTRIBUTED_RETAIN_COUNT
        )
    merged: dict[str, study.BM25Candidate] = {}
    for result in worker_rows:
        for candidate in result["retained"][0]:
            study._update_top_unique(
                merged, candidate, limit=study.HARD_DISTRACTOR_COUNT
            )
    assert len(merged) == 30
    expected = sorted(
        (
            study.BM25Candidate(
                score_int=study.bm25_score_int(
                    query_tokens=union,
                    document_tokens=study.tokenize(line),
                    document_count=90,
                    total_token_count=sum(
                        row["total_token_count"] for row in pass1
                    ),
                    document_frequency=df,
                ),
                normalized_fact=study.normalize_text(line),
                exact_fact=line,
                source_ordinal=index,
            )
            for index, line in enumerate(line.rstrip("\n") for line in lines)
        ),
        key=study._bm25_candidate_key,
    )[:30]
    assert sorted(merged.values(), key=study._bm25_candidate_key) == expected


def test_private_view_is_gold_free_and_label_joins_only_by_view_hash() -> None:
    candidate = _candidate(10)
    distractors = _distractors()
    texts = [candidate.fact1, candidate.fact2, *[row.exact_fact for row in distractors]]
    view, label = study.build_private_pair(
        candidate=candidate,
        block="A_form",
        distractors=distractors,
        statistics=_statistics(texts),
        secret=b"d" * 32,
    )
    assert set(view) == study.VIEW_KEYS
    assert set(label) == study.LABEL_KEYS
    assert "answerKey" not in view
    assert "identity_commitment_sha256" not in view
    assert "label_envelope_sha256" not in view
    assert label["view_sha256"] == stable_hash(view)
    assert sorted(label["gold_document_ids"]) == sorted(
        [label["fact1_document_id"], label["fact2_document_id"]]
    )
    assert view["raw_ranking"] == [
        row["doc_id"]
        for row in sorted(
            view["documents"],
            key=lambda row: (-row["bm25_score_int"], row["doc_id"]),
        )[:5]
    ]
    bad_raw = json.loads(json.dumps(view))
    bad_raw["raw_ranking"] = list(reversed(bad_raw["raw_ranking"]))
    with pytest.raises(study.QASCAcquisitionError, match="gold-free view"):
        study._validate_view_only(bad_raw, expected_block="A_form")
    bad_choices = json.loads(json.dumps(view))
    bad_choices["choices"][1]["text"] = bad_choices["choices"][0]["text"]
    with pytest.raises(study.QASCAcquisitionError, match="gold-free view"):
        study._validate_view_only(bad_choices, expected_block="A_form")


@pytest.mark.parametrize("answer_key", ("Z", None, {"opaque": "value"}))
def test_malformed_answer_key_fails_only_after_candidate_has_been_selected(
    answer_key: object,
) -> None:
    raw = _row(12)
    raw["answerKey"] = answer_key
    candidate = study._parse_candidate(
        raw, source_member="TRAIN", source_row_ordinal=12
    )
    assert candidate is not None
    distractors = _distractors("late-label")
    texts = [candidate.fact1, candidate.fact2, *[row.exact_fact for row in distractors]]
    with pytest.raises(study.QASCAcquisitionError, match="answerKey|label envelope"):
        study.build_private_pair(
            candidate=candidate,
            block="A_form",
            distractors=distractors,
            statistics=_statistics(texts),
            secret=b"l" * 32,
        )


def test_view_and_label_loaders_are_physically_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(study.BLOCK_COUNTS, "A_form", 1)
    candidate = _candidate(11)
    distractors = _distractors("x")
    texts = [candidate.fact1, candidate.fact2, *[row.exact_fact for row in distractors]]
    view, label = study.build_private_pair(
        candidate=candidate,
        block="A_form",
        distractors=distractors,
        statistics=_statistics(texts),
        secret=b"e" * 32,
    )
    view_path = tmp_path / "A_form.views.jsonl"
    label_path = tmp_path / "A_form.labels.jsonl"
    view_file, view_set = study._write_jsonl_exclusive(view_path, (view,))
    label_raw = study._canonical_bytes(label) + b"\n"
    commitment = study.BlockCommitment(
        block="A_form",
        source_member="TRAIN",
        count=1,
        view_file_sha256=view_file,
        label_file_sha256=hashlib.sha256(label_raw).hexdigest(),
        view_commitment_set_sha256=view_set,
        label_commitment_set_sha256=stable_hash([stable_hash(label)]),
        joined_commitment_set_sha256=study._joined_commitment((view,), (label,)),
    )
    # The label file does not exist: view loading must still succeed.
    views = study.load_private_views(
        view_path=view_path, commitment=commitment, expected_block="A_form"
    )
    assert len(views) == 1
    study._atomic_write_exclusive(label_path, label_raw, mode=0o600)
    labels = study.load_private_labels(
        label_path=label_path, commitment=commitment, expected_block="A_form"
    )
    joined = study.join_private_block(
        views=views,
        labels=labels,
        commitment=commitment,
        expected_block="A_form",
    )
    assert joined == ((view, label),)


def test_atomic_writer_never_replaces_existing_path(tmp_path: Path) -> None:
    target = tmp_path / "exclusive.json"
    study._atomic_write_exclusive(target, b"first\n", mode=0o600)
    with pytest.raises(FileExistsError):
        study._atomic_write_exclusive(target, b"second\n", mode=0o600)
    assert target.read_bytes() == b"first\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_safe_corpus_extraction_verifies_archive_member_and_reuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    member_raw = b"alpha fact\nbeta fact\n"
    archive_path = tmp_path / "corpus.tar.gz"
    source = tmp_path / "source.txt"
    source.write_bytes(member_raw)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source, arcname=study.CORPUS_MEMBER_PATH)
    destination = tmp_path / "extracted.txt"
    monkeypatch.setattr(study, "CORPUS_ARCHIVE_SHA256", study._sha256_file(archive_path))
    monkeypatch.setattr(study, "CORPUS_MEMBER_SHA256", hashlib.sha256(member_raw).hexdigest())
    monkeypatch.setattr(study, "CORPUS_MEMBER_SIZE", len(member_raw))
    monkeypatch.setattr(study, "_CORPUS_EXTRACTION_SAFETY_MARGIN_BYTES", 0)
    first = study.prepare_unlabeled_corpus(
        corpus_archive=archive_path, extracted_corpus=destination
    )
    second = study.prepare_unlabeled_corpus(
        corpus_archive=archive_path, extracted_corpus=destination
    )
    assert destination.read_bytes() == member_raw
    assert first["reused_verified_extraction"] is False
    assert second["reused_verified_extraction"] is True


def test_synthetic_dataset_loader_opens_train_and_dev_but_not_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoded = {
        "TRAIN": study._canonical_bytes(_row(1, member="TRAIN")) + b"\n",
        "DEV": study._canonical_bytes(_row(2, member="DEV")) + b"\n",
    }
    archive_path = tmp_path / "dataset.tar.gz"
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_bytes(encoded["TRAIN"])
    dev.write_bytes(encoded["DEV"])
    test.write_bytes(b"this is deliberately invalid and must never be parsed\n")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(train, arcname=study.SOURCE_MEMBER_PATHS["TRAIN"])
        archive.add(dev, arcname=study.SOURCE_MEMBER_PATHS["DEV"])
        archive.add(test, arcname="QASC_Dataset/test.jsonl")
    monkeypatch.setattr(study, "DATASET_ARCHIVE_SHA256", study._sha256_file(archive_path))
    monkeypatch.setattr(
        study,
        "SOURCE_MEMBER_SHA256S",
        {member: hashlib.sha256(raw).hexdigest() for member, raw in encoded.items()},
    )
    monkeypatch.setattr(study, "SOURCE_MEMBER_ROW_COUNTS", {"TRAIN": 1, "DEV": 1})
    rows = study.load_formal_candidates(archive_path)
    assert [row.source_member for row in rows] == ["TRAIN", "DEV"]


def test_public_protocol_binding_matches_all_committed_dependencies() -> None:
    project = Path(__file__).resolve().parents[1]
    if not (project / study.INFRASTRUCTURE_DIAGNOSTIC_RELATIVE).is_file():
        pytest.skip("row-free infrastructure diagnostic is formed after implementation commit")
    bindings = study.public_protocol_bindings(project)
    assert bindings["design"]["design_sha256"] == study.DESIGN_SHA256
    assert bindings["source_qualification"]["qualification_sha256"] == (
        study.SOURCE_QUALIFICATION_SHA256
    )
    assert bindings["source_custody"]["custody_sha256"] == study.SOURCE_CUSTODY_SHA256
    assert bindings["source_access_addendum"]["addendum_sha256"] == (
        study.SOURCE_ADDENDUM_SHA256
    )
    assert bindings["nli_asset"]["asset_sha256"] == study.NLI_ASSET_SHA256


def test_preregistration_opens_no_qasc_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = b"p" * 32
    monkeypatch.setattr(
        study, "_canonical_selection_secret", lambda _project, path: (path, secret)
    )
    monkeypatch.setattr(study, "public_protocol_bindings", lambda _project: {})
    monkeypatch.setattr(
        study,
        "implementation_binding",
        lambda _project: {"files": [], "set_sha256": stable_hash([])},
    )
    monkeypatch.setattr(
        study.tarfile,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("QASC source opened during preregistration")
        ),
    )
    payload = study.build_preregistration(
        project=Path(__file__).resolve().parents[1],
        selection_secret_path=Path("unused"),
    )
    assert payload["source"]["formal_QA_rows_opened"] == 0
    assert payload["source"]["TEST_reopened_by_formal_acquisition"] is False
    assert payload["safety"]["performance_scores_computed"] == 0


def test_public_safety_rejects_any_private_view_or_label_field() -> None:
    study._assert_public_safe({"counts": {"selected": 256}})
    with pytest.raises(study.QASCAcquisitionError, match="private fields"):
        study._assert_public_safe({"nested": {"answerKey": "A"}})


def test_formal_acquisition_rejects_imported_direct_call() -> None:
    assert study._CLEAN_MODULE_CLI_ACTIVE is False
    with pytest.raises(study.QASCAcquisitionError, match="only through clean CLI"):
        study.acquire_private_blocks(
            project=Path("unused"),
            preregistration_path=Path("unused"),
            selection_secret_path=Path("unused"),
            dataset_archive_path=Path("unused"),
            corpus_archive_path=Path("unused"),
            extracted_corpus_path=Path("unused"),
            private_root=Path("unused"),
            private_locator_path=Path("unused"),
            public_receipt_path=Path("unused"),
        )


def test_selection_secret_loader_decodes_hex_lf_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = bytes(range(32))
    path = tmp_path / "selection_v2.key"
    path.write_bytes(secret.hex().encode("ascii") + b"\n")
    path.chmod(0o600)
    monkeypatch.setattr(
        study,
        "_canonical_private_path",
        lambda **_kwargs: path,
    )
    monkeypatch.setattr(
        study, "SELECTION_SECRET_COMMITMENT_SHA256", hashlib.sha256(secret).hexdigest()
    )
    assert study.load_selection_secret(
        project=tmp_path, selection_secret_path=path
    ) == secret
