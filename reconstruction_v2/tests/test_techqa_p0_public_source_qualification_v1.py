from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from unittest import mock

import pytest

from assumption_agent.benchmarks import techqa_p0_public_source_qualification_v1 as p0
from assumption_agent.benchmarks import techqa_p1_formal_v1 as formal


@pytest.fixture
def posix_tmp() -> Path:
    root = Path(tempfile.mkdtemp(prefix="techqa-p0-test-", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _documents() -> dict[str, dict[str, str]]:
    return {
        f"d{ordinal:02d}": {
            "_id": f"d{ordinal:02d}",
            "title": f"Document {ordinal}",
            "text": (
                "valid answer and technical reference text"
                if ordinal == 0
                else f"technical material number {ordinal}"
            ),
            "harmless_document_extra": "aggregate-only",
        }
        for ordinal in range(50)
    }


def _question(
    *,
    split: str,
    family: str,
    ordinal: int,
) -> dict[str, object]:
    title = {
        p0.INFORMATION: "Database configuration reference",
        p0.PROCEDURE: "How to configure the database",
        p0.TROUBLESHOOT: "Why does the database report an error",
    }[family]
    return {
        "ANSWERABLE": "Y",
        "DOC_IDS": [f"d{index:02d}" for index in reversed(range(50))],
        "DOCUMENT": "d00",
        "END_OFFSET": "5",
        "QUESTION_ID": f"{split}-{family}-{ordinal:03d}",
        "QUESTION_TEXT": (
            f"Public technical support question body {split} {ordinal}."
        ),
        "QUESTION_TITLE": title,
        "START_OFFSET": "0",
        "harmless_query_extra": {"kind": "public"},
    }


def _query_rows(
    split: str,
    minimum: int,
) -> list[dict[str, object]]:
    return [
        _question(split=split, family=family, ordinal=ordinal)
        for family in p0.FAMILIES
        for ordinal in range(minimum)
    ]


def _members(
    *,
    train: list[dict[str, object]] | None = None,
    dev: list[dict[str, object]] | None = None,
    corpus: dict[str, dict[str, str]] | None = None,
) -> list[tuple[str, bytes, str]]:
    return [
        ("TechQA/data/training_Q_A.json", json.dumps(
            train
            if train is not None
            else _query_rows("TRAIN", formal.SOURCE_MINIMUM_FAMILY_COUNTS["TRAIN"]),
            ensure_ascii=False,
        ).encode("utf-8"), "file"),
        ("TechQA/data/dev_Q_A.json", json.dumps(
            dev
            if dev is not None
            else _query_rows("DEV", formal.SOURCE_MINIMUM_FAMILY_COUNTS["DEV"]),
            ensure_ascii=False,
        ).encode("utf-8"), "file"),
        ("TechQA/data/training_dev_technotes.json", json.dumps(
            corpus if corpus is not None else _documents(),
            ensure_ascii=False,
        ).encode("utf-8"), "file"),
        ("TechQA/README.txt", b"synthetic public fixture", "file"),
    ]


def _archive(
    root: Path,
    members: list[tuple[str, bytes, str]],
) -> tuple[Path, p0.ArchiveContract]:
    path = root / "synthetic-TechQA.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, raw, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "elsewhere"
                archive.addfile(info)
            else:
                raise AssertionError(kind)
    return path, p0.ArchiveContract(
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _tiny_kvitems(source):
    value = json.load(source)
    assert isinstance(value, dict)
    yield from value.items()


def _qualify(
    path: Path,
    contract: p0.ArchiveContract,
) -> dict[str, object]:
    with mock.patch.object(p0, "_ijson_kvitems", _tiny_kvitems):
        return p0.qualify_archive(
            archive_path=path,
            qualified_source_root=path.parent / "qualified_source",
            eligibility_manifest_path=(
                path.parent / "eligibility.private.json"
            ),
            archive_contract=contract,
        )


def test_classifier_is_the_single_frozen_formal_classifier() -> None:
    fixtures = (
        ("How to fix the issue", "instructions", p0.TROUBLESHOOT),
        ("How can I configure this", "", p0.PROCEDURE),
        ("Product compatibility matrix", "", p0.INFORMATION),
        ("A change request", "", p0.INFORMATION),
    )
    for title, text, expected in fixtures:
        assert p0.operational_family(title, text) == expected
        assert formal.operational_family(title, text) == expected
    assert p0.FAMILIES == formal.FAMILY_IDS
    assert p0.SOURCE_MINIMUM_FAMILY_COUNTS is formal.SOURCE_MINIMUM_FAMILY_COUNTS


def test_streaming_qualification_emits_only_safe_aggregates(
    posix_tmp: Path,
) -> None:
    members = _members()
    path, contract = _archive(posix_tmp, members)
    receipt = _qualify(path, contract)
    assert receipt["status"] == (
        "qualified_public_non_scoring_schema_and_family_capacity"
    )
    assert receipt["archive"]["streaming_semantic_pass_count"] == 2
    assert receipt["corpus_aggregate"]["document_count"] == 50
    assert (
        receipt["corpus_aggregate"]["answerable_gold_span_validation_count"]
        == sum(
            formal.SOURCE_MINIMUM_FAMILY_COUNTS[split] * len(p0.FAMILIES)
            for split in ("TRAIN", "DEV")
        )
    )
    for split in ("TRAIN", "DEV"):
        counts = receipt["corpus_aggregate"][
            "split_post_corpus_eligibility"
        ][split][
            "eligible_unique_normalized_query_family_count"
        ]
        assert counts == {
            family: formal.SOURCE_MINIMUM_FAMILY_COUNTS[split]
            for family in p0.FAMILIES
        }
    rendered = p0.canonical_bytes(receipt).decode("ascii")
    for forbidden in (
        "TRAIN-INFORMATION-000",
        '"d00"',
        "valid answer",
        "Public technical support question body",
        "aggregate-only",
    ):
        assert forbidden not in rendered
    assert receipt["access_boundary"] == {
        "action_model_qrel_evaluator_or_score_count": 0,
        "cohort_assignment_or_secret_count": 0,
        "individual_query_document_or_span_value_output_count": 0,
        "online_or_API_evaluation_count": 0,
        "source_archive_full_extraction_count": 0,
        "source_archive_whitelisted_member_extraction_count": 3,
    }
    qualified = posix_tmp / "qualified_source"
    assert {path.name for path in qualified.iterdir()} == {
        p0.TRAIN_QA_BASENAME,
        p0.DEV_QA_BASENAME,
        p0.CORPUS_BASENAME,
    }
    expected = {
        Path(name).name: raw
        for name, raw, kind in members
        if kind == "file" and Path(name).name in p0.TARGET_BASENAMES
    }
    for basename, raw in expected.items():
        persisted = qualified / basename
        assert persisted.read_bytes() == raw
        assert persisted.stat().st_mode & 0o777 == 0o600
    private_manifest_path = posix_tmp / "eligibility.private.json"
    private_manifest = json.loads(
        private_manifest_path.read_text("ascii")
    )
    assert private_manifest_path.stat().st_mode & 0o777 == 0o600
    assert private_manifest["eligibility_rule_version"] == (
        p0.ELIGIBILITY_RULE_VERSION
    )
    assert private_manifest["self_sha256"] == p0.stable_hash(
        {
            key: value
            for key, value in private_manifest.items()
            if key != "self_sha256"
        }
    )
    assert receipt["private_eligibility_manifest_binding"][
        "self_sha256"
    ] == private_manifest["self_sha256"]
    assert receipt["private_eligibility_manifest_binding"][
        "file_sha256"
    ] == hashlib.sha256(private_manifest_path.read_bytes()).hexdigest()
    assert private_manifest["source_member_content_sha256"] == {
        basename: receipt["archive"]["target_members"][basename][
            "content_sha256"
        ]
        for basename in sorted(p0.TARGET_BASENAMES)
    }


@pytest.mark.parametrize(
    "failure",
    (
        "oversized_query",
        "duplicate_normalized_query",
        "cross_split_normalized_query",
    ),
)
def test_unique_action_compatible_query_capacity_is_post_corpus_and_disjoint(
    posix_tmp: Path,
    failure: str,
) -> None:
    train = _query_rows(
        "TRAIN",
        formal.SOURCE_MINIMUM_FAMILY_COUNTS["TRAIN"],
    )
    dev = _query_rows(
        "DEV",
        formal.SOURCE_MINIMUM_FAMILY_COUNTS["DEV"],
    )
    if failure == "oversized_query":
        train[0]["QUESTION_TEXT"] = "x" * formal.core.MAX_QUERY_CHARACTERS
    elif failure == "duplicate_normalized_query":
        train[0]["QUESTION_TITLE"] = train[1]["QUESTION_TITLE"]
        train[0]["QUESTION_TEXT"] = train[1]["QUESTION_TEXT"]
    else:
        dev[0]["QUESTION_TITLE"] = train[0]["QUESTION_TITLE"]
        dev[0]["QUESTION_TEXT"] = train[0]["QUESTION_TEXT"]
    path, contract = _archive(
        posix_tmp,
        _members(train=train, dev=dev),
    )
    with pytest.raises(p0.TechqaP0QualificationError):
        _qualify(path, contract)


@pytest.mark.parametrize("candidate_drift", ("missing", "oversized"))
def test_ineligible_extra_candidate_row_does_not_reduce_valid_capacity(
    posix_tmp: Path,
    candidate_drift: str,
) -> None:
    train = _query_rows(
        "TRAIN",
        formal.SOURCE_MINIMUM_FAMILY_COUNTS["TRAIN"],
    )
    extra = _question(
        split="TRAIN",
        family=p0.INFORMATION,
        ordinal=999,
    )
    extra["DOC_IDS"] = [
        "d50" if value == "d49" else value
        for value in extra["DOC_IDS"]
    ]
    train.append(extra)
    corpus = _documents()
    if candidate_drift == "oversized":
        corpus["d50"] = {
            "_id": "d50",
            "title": "Oversized candidate",
            "text": "x" * (
                formal.core.MAX_DOCUMENT_FIELD_CHARACTERS + 1
            ),
        }
    path, contract = _archive(
        posix_tmp,
        _members(train=train, corpus=corpus),
    )
    receipt = _qualify(path, contract)
    eligibility = receipt["corpus_aggregate"][
        "split_post_corpus_eligibility"
    ]["TRAIN"]
    assert eligibility[
        "eligible_unique_normalized_query_family_count"
    ][p0.INFORMATION] == formal.SOURCE_MINIMUM_FAMILY_COUNTS["TRAIN"]
    expected_reason = {
        "missing": "candidate_document_missing_from_corpus",
        "oversized": (
            "candidate_document_shared_character_or_lexical_bound"
        ),
    }[candidate_drift]
    assert eligibility["ineligible_answerable_row_reason_count"] == {
        expected_reason: 1
    }
    private_manifest = json.loads(
        (posix_tmp / "eligibility.private.json").read_text("ascii")
    )
    private_ids = {
        row["question_id"]
        for row in private_manifest[
            "eligible_answerable_rows_by_split"
        ]["TRAIN"]
    }
    assert extra["QUESTION_ID"] not in private_ids


@pytest.mark.parametrize("failure", ("duplicate_pool", "gold_missing", "bad_span"))
def test_query_and_gold_contracts_fail_closed(
    posix_tmp: Path,
    failure: str,
) -> None:
    train = _query_rows(
        "TRAIN",
        formal.SOURCE_MINIMUM_FAMILY_COUNTS["TRAIN"],
    )
    if failure == "duplicate_pool":
        train[0]["DOC_IDS"] = ["d00"] * 50
    elif failure == "gold_missing":
        train[0]["DOCUMENT"] = "outside"
    else:
        train[0]["START_OFFSET"] = "9999"
        train[0]["END_OFFSET"] = "10000"
    path, contract = _archive(posix_tmp, _members(train=train))
    with pytest.raises(p0.TechqaP0QualificationError):
        _qualify(path, contract)


@pytest.mark.parametrize("failure", ("traversal", "link", "duplicate_target"))
def test_tar_whitelist_rejects_unsafe_or_duplicate_members(
    posix_tmp: Path,
    failure: str,
) -> None:
    members = _members()
    if failure == "traversal":
        members.insert(0, ("../escape.txt", b"x", "file"))
    elif failure == "link":
        members.insert(0, ("TechQA/link", b"", "symlink"))
    else:
        members.insert(
            1,
            (
                "other/training_Q_A.json",
                members[0][1],
                "file",
            ),
        )
    path, contract = _archive(posix_tmp, members)
    with pytest.raises(p0.TechqaP0QualificationError):
        _qualify(path, contract)


def test_archive_identity_is_checked_before_tar_semantics(
    posix_tmp: Path,
) -> None:
    path, contract = _archive(posix_tmp, _members())
    bad = p0.ArchiveContract(
        filename=path.name,
        size_bytes=contract.size_bytes,
        sha256="0" * 64,
    )
    with mock.patch.object(
        p0,
        "_collect_query_members",
        side_effect=AssertionError("semantic access happened"),
    ):
        with pytest.raises(p0.TechqaP0QualificationError):
            p0.qualify_archive(
                archive_path=path,
                qualified_source_root=posix_tmp / "qualified_source",
                eligibility_manifest_path=(
                    posix_tmp / "eligibility.private.json"
                ),
                archive_contract=bad,
            )


def test_qualification_failure_preserves_private_partial_source_evidence(
    posix_tmp: Path,
) -> None:
    train = _query_rows(
        "TRAIN",
        formal.SOURCE_MINIMUM_FAMILY_COUNTS["TRAIN"],
    )
    train[0]["START_OFFSET"] = "9999"
    train[0]["END_OFFSET"] = "10000"
    path, contract = _archive(posix_tmp, _members(train=train))
    with pytest.raises(p0.TechqaP0QualificationError):
        _qualify(path, contract)
    qualified = posix_tmp / "qualified_source"
    assert (qualified / p0.TRAIN_QA_BASENAME).is_file()
    assert (qualified / p0.DEV_QA_BASENAME).is_file()
    # The same corpus bytes consumed before span failure are retained; the
    # terminal, rather than the directory name, determines qualification.
    assert (qualified / p0.CORPUS_BASENAME).is_file()
    assert not (posix_tmp / "eligibility.private.json").exists()


def test_referenced_ids_must_map_to_unique_official_document_bytes(
    posix_tmp: Path,
) -> None:
    corpus = _documents()
    corpus["d01"]["title"] = corpus["d00"]["title"]
    corpus["d01"]["text"] = corpus["d00"]["text"]
    path, contract = _archive(posix_tmp, _members(corpus=corpus))
    with pytest.raises(p0.TechqaP0QualificationError):
        _qualify(path, contract)


def test_acquisition_marker_precedes_the_only_download_invocation(
    posix_tmp: Path,
) -> None:
    work = posix_tmp / "work"
    runtime = posix_tmp / "runtime"
    runtime.mkdir()
    manifest = posix_tmp / "venv_files.sha256"
    manifest.write_text("synthetic", encoding="ascii")
    calls: list[tuple[str, ...]] = []

    def runner(command, environment, cwd):
        assert (work / "attempt.marker.json").is_file()
        assert not calls
        assert environment["HF_ENDPOINT"] == "https://huggingface.co"
        assert environment["HF_HUB_DISABLE_XET"] == "0"
        assert environment["HF_XET_HIGH_PERFORMANCE"] == "1"
        calls.append(tuple(command))
        (work / "source" / p0.ARCHIVE_FILENAME).write_bytes(b"fixture")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b"ok",
            stderr=b"",
        )

    safe_receipt = p0.self_hashed(
        {
            "schema": f"{p0.VERSION}_safe_aggregate_receipt",
            "status": "qualified_public_non_scoring_schema_and_family_capacity",
            "study_id": p0.STUDY_ID,
        }
    )
    with (
        mock.patch.object(
            p0,
            "_runtime_versions",
            return_value=p0.PINNED_HF_RUNTIME,
        ),
        mock.patch.object(
            p0,
            "qualify_archive",
            return_value=safe_receipt,
        ),
    ):
        terminal = p0.acquire_and_qualify(
            work_root=work,
            hf_runtime_root=runtime,
            hf_runtime_manifest=manifest,
            hf_runtime_manifest_sha256="1" * 64,
            download_runner=runner,
        )
    assert len(calls) == 1
    assert calls[0][1:4] == (
        "download",
        p0.HF_REPOSITORY_ID,
        p0.ARCHIVE_FILENAME,
    )
    assert terminal["status"] == "qualified_public_non_scoring_source"
    assert (work / "p0_terminal.json").stat().st_mode & 0o777 == 0o600


def test_download_failure_is_terminal_and_never_retried(
    posix_tmp: Path,
) -> None:
    work = posix_tmp / "failed"
    runtime = posix_tmp / "runtime"
    runtime.mkdir()
    manifest = posix_tmp / "venv_files.sha256"
    manifest.write_text("synthetic", encoding="ascii")
    count = 0

    def runner(command, environment, cwd):
        nonlocal count
        count += 1
        return subprocess.CompletedProcess(
            command,
            9,
            stdout=b"",
            stderr=b"failure",
        )

    with mock.patch.object(
        p0,
        "_runtime_versions",
        return_value=p0.PINNED_HF_RUNTIME,
    ):
        with pytest.raises(p0.TechqaP0QualificationError):
            p0.acquire_and_qualify(
                work_root=work,
                hf_runtime_root=runtime,
                hf_runtime_manifest=manifest,
                hf_runtime_manifest_sha256="2" * 64,
                download_runner=runner,
            )
    assert count == 1
    terminal = json.loads((work / "p0_terminal.json").read_text("ascii"))
    assert terminal["status"] == "implementation_source_or_infrastructure_invalid"
    assert terminal["retry_resume_or_second_invocation_authorized"] is False
