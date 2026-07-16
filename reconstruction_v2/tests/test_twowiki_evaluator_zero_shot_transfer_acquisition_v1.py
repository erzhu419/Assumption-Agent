from __future__ import annotations

from collections import Counter
from io import BytesIO
import hashlib
import hmac
import json
from pathlib import Path
import tempfile
from typing import Any, Iterator
import zipfile

import pytest

from assumption_agent.benchmarks import (
    twowiki_evaluator_zero_shot_transfer_acquisition_v1 as study,
)
from assumption_agent.models import stable_hash


@pytest.fixture
def private_tmp_path() -> Iterator[Path]:
    parent = Path(__file__).resolve().parents[1] / "artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="twowiki-acquisition-test-", dir=parent) as root:
        yield Path(root)


def _row(
    index: int,
    *,
    question_type: str = "compositional",
    question: str | None = None,
    support_titles: tuple[int, ...] = (0, 2),
) -> dict[str, Any]:
    context = [
        [
            f"Title {index}-{document}",
            [
                f"Sentence zero for item {index} document {document}.",
                f"Sentence one contains signal {index}-{document}.",
            ],
        ]
        for document in range(10)
    ]
    return {
        "_id": f"id-{index:06d}",
        "type": question_type,
        "question": question or f"Which signal belongs to unique item {index}?",
        "context": context,
        "supporting_facts": [[context[position][0], 1] for position in support_titles],
        "evidences": [["entity", "relation", f"value-{index}"]],
        "answer": f"Answer {index}",
    }


def _candidate(raw: dict[str, Any], member: str) -> study._CandidateIdentity:
    normalized = study._normalize_source_row(raw, source_member=member)
    assert normalized is not None
    identity = study._identity_hashes(raw, source_member=member)
    return study._CandidateIdentity(
        source_member=member,
        question_type=normalized["question_type"],
        item_id=normalized["item_id"],
        normalized_question_sha256=normalized["normalized_question_sha256"],
        canonical_question_plus_ordered_context_sha256=normalized[
            "canonical_question_plus_ordered_context_sha256"
        ],
        canonical_row_sha256=normalized["canonical_row_sha256"],
        identity_commitment_sha256=identity[3],
    )


def _collision(raw: dict[str, Any], member: str) -> study._CollisionIdentity:
    identity = study._collision_identity(raw, source_member=member)
    assert identity is not None
    return identity


def _collision_from_candidate(
    row: study._CandidateIdentity,
) -> study._CollisionIdentity:
    return study._CollisionIdentity(
        source_member=row.source_member,
        item_id=row.item_id,
        normalized_question_sha256=row.normalized_question_sha256,
    )


def _empty_historical() -> study._HistoricalDenylist:
    return study._HistoricalDenylist(
        item_ids=frozenset(),
        normalized_question_sha256s=frozenset(),
        canonical_question_context_sha256s=frozenset(),
        canonical_row_sha256s=frozenset(),
        binding={"set_commitments": {}},
    )


def test_question_normalization_is_exact_nfkc_casefold_word_token_contract() -> None:
    assert study.normalize_question("  Ｃafé—ALPHA_2?!  ") == "café alpha_2"
    assert study.normalize_question("Straße  １２") == "strasse 12"


def test_source_normalizer_maps_exact_order_and_variable_support_documents() -> None:
    raw = _row(1, question_type="inference", support_titles=(1, 4, 7))
    normalized = study._normalize_source_row(raw, source_member="train.json")
    assert normalized is not None
    assert set(normalized) == study.PRIVATE_BLOCK_ROW_KEYS
    assert normalized["source_member"] == "train.json"
    assert normalized["question_type"] == "inference"
    assert normalized["support_indices"] == [1, 4, 7]
    assert [row["paragraph_idx"] for row in normalized["corpus"]] == list(range(10))
    assert normalized["corpus"][4]["paragraph_title"] == raw["context"][4][0]
    assert normalized["corpus"][4]["paragraph_text"] == " ".join(
        raw["context"][4][1]
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda row: row.update(context=row["context"][:9]),
        lambda row: row["context"].__setitem__(1, row["context"][0]),
        lambda row: row["context"][0][1].__setitem__(0, ""),
        lambda row: row.update(supporting_facts=[[row["context"][0][0], 99]]),
        lambda row: row.update(type="unknown"),
        lambda row: row.update(answer=""),
    ),
)
def test_source_normalizer_rejects_data_integrity_failures(mutation: Any) -> None:
    row = _row(2)
    mutation(row)
    assert study._normalize_source_row(row, source_member="train.json") is None


def test_identity_commitment_matches_frozen_canonical_json_definition() -> None:
    raw = _row(3, question_type="comparison")
    normalized_question = study.normalize_question(raw["question"])
    normalized_sha = hashlib.sha256(normalized_question.encode("utf-8")).hexdigest()
    context_sha = hashlib.sha256(
        json.dumps(
            {"question": raw["question"], "context": raw["context"]},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    row_sha = hashlib.sha256(
        json.dumps(
            raw,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    body = {
        "member": "dev.json",
        "item_id_sha256": hashlib.sha256(raw["_id"].encode("utf-8")).hexdigest(),
        "normalized_question_sha256": normalized_sha,
        "canonical_question_plus_ordered_context_sha256": context_sha,
        "canonical_row_sha256": row_sha,
    }
    expected_commitment = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert study._identity_hashes(raw, source_member="dev.json") == (
        normalized_sha,
        context_sha,
        row_sha,
        expected_commitment,
    )


def test_canonical_json_rejects_nonfinite_values() -> None:
    with pytest.raises(study.TwoWikiAcquisitionError, match="canonical-JSON"):
        study._canonical_json_bytes({"bad": float("nan")})


def test_selection_hmac_uses_exact_identity_commitment_and_tiebreak_contract() -> None:
    secret = bytes(range(32))
    identity = "a" * 64
    expected = hmac.new(
        secret,
        (
            f"{study.VERSION}\0train.json\0comparison\0{identity}"
        ).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert study._selection_key(
        source_member="train.json",
        question_type="comparison",
        identity_commitment_sha256=identity,
        secret=secret,
    ) == expected


def test_streaming_json_array_decoder_handles_unicode_and_small_chunks() -> None:
    raw = json.dumps(
        [{"x": "中文"}, 2, ["three"]], ensure_ascii=False
    ).encode("utf-8")
    assert list(study._iter_json_array_stream(BytesIO(raw), chunk_size=1)) == [
        {"x": "中文"},
        2,
        ["three"],
    ]


@pytest.mark.parametrize("raw", (b"{}", b"[1,]", b"[1 2]", b"[1] trailing"))
def test_streaming_json_array_decoder_rejects_noncanonical_container(raw: bytes) -> None:
    with pytest.raises(study.TwoWikiAcquisitionError):
        list(study._iter_json_array_stream(BytesIO(raw), chunk_size=1))


def test_synthetic_zip_scan_hashes_streams_and_counts_without_formal_source(
    private_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_rows = [_row(7001, question_type="comparison"), _row(7002)]
    dev_rows = [_row(8001, question_type="inference")]
    test_row = _row(9001, question_type="bridge_comparison")
    test_row["answer"] = ""
    test_row["supporting_facts"] = []
    test_rows = [test_row]
    encoded = {
        "train.json": json.dumps(train_rows, ensure_ascii=False).encode("utf-8"),
        "dev.json": json.dumps(dev_rows, ensure_ascii=False).encode("utf-8"),
        "test.json": json.dumps(test_rows, ensure_ascii=False).encode("utf-8"),
    }
    monkeypatch.setattr(
        study,
        "ARCHIVE_MEMBER_SHA256S",
        {member: hashlib.sha256(raw).hexdigest() for member, raw in encoded.items()},
    )
    monkeypatch.setattr(
        study,
        "ARCHIVE_MEMBER_ROW_COUNTS",
        {"train.json": 2, "dev.json": 1, "test.json": 1},
    )
    source = private_tmp_path / "synthetic.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("release/train.json", encoded["train.json"])
        archive.writestr("release/dev.json", encoded["dev.json"])
        archive.writestr("release/test.json", encoded["test.json"])
    with zipfile.ZipFile(source) as archive:
        members = study._exact_zip_members(archive)
        identities, collision, counts, rejected, collision_rejected = (
            study._scan_source_metadata(archive, members)
        )
    assert counts == {"train.json": 2, "dev.json": 1, "test.json": 1}
    assert rejected == {"train.json": 0, "dev.json": 0}
    assert len(identities) == 3
    assert len(collision) == 4
    assert collision_rejected == {"train.json": 0, "dev.json": 0, "test.json": 0}
    assert {row.source_member for row in identities} == {"train.json", "dev.json"}


def test_exact_zip_member_resolution_rejects_duplicate_basenames(
    private_tmp_path: Path,
) -> None:
    source = private_tmp_path / "ambiguous.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("one/train.json", "[]")
        archive.writestr("two/train.json", "[]")
        archive.writestr("dev.json", "[]")
        archive.writestr("test.json", "[]")
    with zipfile.ZipFile(source) as archive:
        with pytest.raises(study.TwoWikiAcquisitionError, match="ambiguous"):
            study._exact_zip_members(archive)


def test_selection_excludes_all_cross_split_question_collisions_and_historical_rows() -> None:
    identities: list[study._CandidateIdentity] = []
    collision_identities: list[study._CollisionIdentity] = []
    index = 1000
    for member, minimum in (("train.json", 14), ("dev.json", 8)):
        for question_type in study.QUESTION_TYPES:
            for _ in range(minimum):
                raw = _row(index, question_type=question_type)
                identities.append(_candidate(raw, member))
                collision_identities.append(_collision(raw, member))
                index += 1
    collision_question = "This exact normalized question collides across splits."
    train_collision = _candidate(
        _row(index, question_type="comparison", question=collision_question),
        "train.json",
    )
    index += 1
    dev_collision = _candidate(
        _row(index, question_type="comparison", question=collision_question.upper()),
        "dev.json",
    )
    index += 1
    historical_row = _candidate(_row(index, question_type="inference"), "dev.json")
    identities.extend((train_collision, dev_collision, historical_row))
    collision_identities.extend(
        (
            _collision(_row(index - 2, question_type="comparison", question=collision_question), "train.json"),
            _collision(_row(index - 1, question_type="comparison", question=collision_question.upper()), "dev.json"),
            _collision_from_candidate(historical_row),
        )
    )
    test_collision_raw = _row(
        index + 1,
        question_type="comparison",
        question=identities[0].normalized_question_sha256,
    )
    test_collision_raw["question"] = _row(1000)["question"].upper()
    test_collision_raw["answer"] = ""
    test_collision_raw["supporting_facts"] = []
    collision_identities.append(_collision(test_collision_raw, "test.json"))
    historical = study._HistoricalDenylist(
        item_ids=frozenset({historical_row.item_id}),
        normalized_question_sha256s=frozenset(),
        canonical_question_context_sha256s=frozenset(),
        canonical_row_sha256s=frozenset(),
        binding={"set_commitments": {}},
    )
    selected, stats = study._select_identities(
        identities,
        collision_identities=collision_identities,
        historical=historical,
        secret=bytes(range(32)),
    )
    selected_ids = {
        row.item_id for block in study.BLOCK_ORDER for row in selected[block]
    }
    assert train_collision.item_id not in selected_ids
    assert dev_collision.item_id not in selected_ids
    assert historical_row.item_id not in selected_ids
    assert identities[0].item_id not in selected_ids
    assert stats["normalized_question_collision_class_count"] == 2
    assert stats["exclusion_counts_by_member"]["train.json"][
        "question_collision"
    ] == 2
    assert stats["exclusion_counts_by_member"]["dev.json"][
        "question_collision"
    ] == 1
    assert stats["exclusion_counts_by_member"]["dev.json"]["historical"] == 1
    assert {block: len(rows) for block, rows in selected.items()} == study.BLOCK_COUNTS


def test_selection_is_stratified_by_member_and_all_four_exact_types() -> None:
    identities: list[study._CandidateIdentity] = []
    index = 3000
    for member, count in (("train.json", 13), ("dev.json", 7)):
        for question_type in study.QUESTION_TYPES:
            for _ in range(count):
                identities.append(_candidate(_row(index, question_type=question_type), member))
                index += 1
    selected, _stats = study._select_identities(
        identities,
        collision_identities=[
            _collision_from_candidate(row) for row in identities
        ],
        historical=_empty_historical(),
        secret=b"z" * 32,
    )
    for block in study.BLOCK_ORDER:
        assert Counter(row.question_type for row in selected[block]) == Counter(
            {
                question_type: study.BLOCK_PER_TYPE_COUNTS[block]
                for question_type in study.QUESTION_TYPES
            }
        )
        assert {row.source_member for row in selected[block]} == {
            study.BLOCK_SOURCE_MEMBERS[block]
        }


def test_collision_scan_uses_each_available_identity_field_fail_closed() -> None:
    identities: list[study._CandidateIdentity] = []
    raw_rows: list[dict[str, Any]] = []
    index = 20000
    for member, count in (("train.json", 14), ("dev.json", 8)):
        for question_type in study.QUESTION_TYPES:
            for _ in range(count):
                raw = _row(index, question_type=question_type)
                raw_rows.append(raw)
                identities.append(_candidate(raw, member))
                index += 1
    collision_identities = [
        _collision_from_candidate(row) for row in identities
    ]
    missing_id = _row(
        index,
        question=raw_rows[0]["question"].upper(),
    )
    missing_id["_id"] = ""
    missing_question = _row(index + 1)
    missing_question["_id"] = identities[1].item_id
    missing_question["question"] = ""
    collision_identities.extend(
        (
            _collision(missing_id, "test.json"),
            _collision(missing_question, "test.json"),
        )
    )
    selected, stats = study._select_identities(
        identities,
        collision_identities=collision_identities,
        historical=_empty_historical(),
        secret=b"q" * 32,
    )
    selected_ids = {
        row.item_id for block in study.BLOCK_ORDER for row in selected[block]
    }
    assert identities[0].item_id not in selected_ids
    assert identities[1].item_id not in selected_ids
    assert stats["normalized_question_collision_class_count"] == 1
    assert stats["item_id_collision_class_count"] == 1


def test_private_block_roundtrip_and_tamper_rejection(private_tmp_path: Path) -> None:
    output = private_tmp_path / "A_hold.jsonl"
    rows: list[dict[str, Any]] = []
    index = 5000
    for question_type in study.QUESTION_TYPES:
        for _ in range(study.BLOCK_PER_TYPE_COUNTS["A_hold"]):
            normalized = study._normalize_source_row(
                _row(index, question_type=question_type), source_member="train.json"
            )
            assert normalized is not None
            normalized["block"] = "A_hold"
            rows.append(normalized)
            index += 1
    file_sha, set_sha = study._write_jsonl_exclusive(output, rows)
    commitment = study.BlockCommitment(
        block="A_hold",
        source_member="train.json",
        question_type_counts={
            question_type: 12 for question_type in study.QUESTION_TYPES
        },
        count=48,
        file_sha256=file_sha,
        item_commitment_set_sha256=set_sha,
    )
    loaded = study.load_private_block(
        output, commitment=commitment, expected_block="A_hold"
    )
    assert len(loaded) == 48
    output.write_bytes(output.read_bytes().replace(b"Answer 5000", b"Answer X000", 1))
    with pytest.raises(study.TwoWikiAcquisitionError, match="hash drifted"):
        study.load_private_block(output, commitment=commitment)


def test_atomic_writer_never_replaces_existing_path(private_tmp_path: Path) -> None:
    target = private_tmp_path / "artifact.json"
    study._atomic_write_exclusive(target, b"first\n", mode=0o600)
    with pytest.raises(FileExistsError):
        study._atomic_write_exclusive(target, b"second\n", mode=0o600)
    assert target.read_bytes() == b"first\n"
    assert not list(private_tmp_path.glob(".*.tmp"))


def test_public_protocol_binding_matches_committed_design_and_source_receipts() -> None:
    project = Path(__file__).resolve().parents[1]
    bindings = study.public_protocol_bindings(project)
    assert bindings["design"]["design_sha256"] == study.DESIGN_SHA256
    assert bindings["source_qualification"]["qualification_sha256"] == (
        study.SOURCE_QUALIFICATION_SHA256
    )
    assert bindings["source_custody"]["receipt_sha256"] == study.SOURCE_CUSTODY_SHA256
    assert bindings["source_access_addendum"]["addendum_sha256"] == (
        study.SOURCE_ACCESS_ADDENDUM_SHA256
    )


def test_historical_denylist_uses_exact_clean_hipporag_artifact() -> None:
    project = Path(__file__).resolve().parents[1]
    path = project.parent / study.HISTORICAL_QUERY_WORKSPACE_RELATIVE
    denylist = study.load_historical_denylist(project=project, path=path)
    assert len(denylist.item_ids) == 1000
    assert denylist.binding["file_sha256"] == study.HISTORICAL_QUERY_SHA256
    assert denylist.binding["set_commitments"]["item_id_set_sha256"] == (
        study.HISTORICAL_ID_SET_SHA256
    )
    assert "item_ids" not in denylist.binding


def test_preregistration_opens_no_official_archive_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = bytes(range(32))
    monkeypatch.setattr(
        study,
        "_canonical_selection_secret",
        lambda _project, path: (path, secret),
    )
    monkeypatch.setattr(
        study,
        "load_historical_denylist",
        lambda **_kwargs: study._HistoricalDenylist(
            item_ids=frozenset(),
            normalized_question_sha256s=frozenset(),
            canonical_question_context_sha256s=frozenset(),
            canonical_row_sha256s=frozenset(),
            binding={
                "row_count": 1000,
                "set_commitments": {},
                "item_level_content_persisted_publicly": False,
            },
        ),
    )
    monkeypatch.setattr(study, "public_protocol_bindings", lambda _project: {})
    monkeypatch.setattr(
        study, "implementation_binding", lambda _project: {"files": [], "set_sha256": stable_hash([])}
    )
    monkeypatch.setattr(
        study.zipfile,
        "ZipFile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("official source archive opened")
        ),
    )
    project = Path(__file__).resolve().parents[1]
    payload = study.build_preregistration(
        project=project,
        selection_secret_path=Path("unused"),
        historical_queries_path=Path("unused"),
    )
    assert payload["source"]["official_archive_rows_opened"] == 0
    assert payload["safety"]["official_archive_rows_read"] == 0
    assert payload["selection"]["selected_count"] == 72


def test_formal_source_open_occurs_only_after_durable_marker(
    private_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = private_tmp_path / "project"
    project.mkdir()
    (project / "manifests").mkdir()
    (project / "artifacts").mkdir()
    source = project / study.SOURCE_ARCHIVE_RELATIVE
    source.parent.mkdir(parents=True)
    source.write_bytes(b"not-opened-as-a-real-archive")
    prereg_path = project / study.PREREGISTRATION_RELATIVE
    prereg_path.write_text("{}\n", encoding="utf-8")
    secret = bytes(range(32))
    prereg = {
        "preregistration_sha256": "a" * 64,
        "public_protocol_bindings": {
            "design": {"file_sha256": "b" * 64},
            "source_qualification": {"file_sha256": "c" * 64},
            "source_custody": {"file_sha256": "d" * 64},
            "source_access_addendum": {"file_sha256": "f" * 64},
        },
        "implementation": {"files": [], "set_sha256": stable_hash([])},
        "historical_denylist": {"set_commitments": {}},
    }
    historical = study._HistoricalDenylist(
        item_ids=frozenset(),
        normalized_question_sha256s=frozenset(),
        canonical_question_context_sha256s=frozenset(),
        canonical_row_sha256s=frozenset(),
        binding={"set_commitments": {}},
    )
    monkeypatch.setattr(study, "verify_preregistration", lambda **_kwargs: prereg)
    monkeypatch.setattr(
        study,
        "_committed_binding",
        lambda **_kwargs: {
            "file_sha256": "e" * 64,
            "head_blob_sha256": "e" * 64,
            "clean_tracked_HEAD_blob": True,
        },
    )
    monkeypatch.setattr(
        study,
        "_canonical_selection_secret",
        lambda _project, path: (path, secret),
    )
    monkeypatch.setattr(study, "load_historical_denylist", lambda **_kwargs: historical)
    monkeypatch.setattr(
        study,
        "_canonical_private_path",
        lambda *, project, supplied, relative, require_file, field: project / relative,
    )
    original_sha = study._sha256_file
    monkeypatch.setattr(
        study,
        "_sha256_file",
        lambda path: study.OFFICIAL_ARCHIVE_SHA256 if path == source else original_sha(path),
    )

    class SourceOpened(RuntimeError):
        pass

    def open_after_marker(*_args: Any, **_kwargs: Any) -> Any:
        marker = project / study.CONSUMPTION_RELATIVE
        assert marker.is_file()
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        assert marker_payload["source_archive_opened_before_consumption"] is False
        raise SourceOpened

    monkeypatch.setattr(study.zipfile, "ZipFile", open_after_marker)
    arguments = {
        "project": project,
        "preregistration_path": prereg_path,
        "selection_secret_path": Path("unused"),
        "historical_queries_path": Path("unused"),
        "source_archive_path": source,
        "private_root": project / study.PRIVATE_PACK_ROOT_RELATIVE,
        "private_locator_path": project / study.PRIVATE_LOCATOR_RELATIVE,
        "public_receipt_path": project / study.ACQUISITION_RELATIVE,
    }
    with pytest.raises(SourceOpened):
        study.acquire_private_blocks(**arguments)
    marker = project / study.CONSUMPTION_RELATIVE
    assert marker.is_file()
    with pytest.raises(FileExistsError, match="already consumed"):
        study.acquire_private_blocks(**arguments)


def _minimal_public_receipt(private_tmp_path: Path) -> Path:
    def custody(digest: str) -> dict[str, Any]:
        return {
            "file_sha256": digest,
            "head_blob_sha256": digest,
            "clean_tracked_HEAD_blob": True,
        }

    protocol: dict[str, Any] = {}
    for role, relative, schema, file_hash, semantic_field, semantic_hash in (
        (
            "design",
            study.DESIGN_RELATIVE,
            study.DESIGN_SCHEMA,
            study.DESIGN_FILE_SHA256,
            "design_sha256",
            study.DESIGN_SHA256,
        ),
        (
            "source_qualification",
            study.SOURCE_QUALIFICATION_RELATIVE,
            study.SOURCE_QUALIFICATION_SCHEMA,
            study.SOURCE_QUALIFICATION_FILE_SHA256,
            "qualification_sha256",
            study.SOURCE_QUALIFICATION_SHA256,
        ),
        (
            "source_custody",
            study.SOURCE_CUSTODY_RELATIVE,
            study.SOURCE_CUSTODY_SCHEMA,
            study.SOURCE_CUSTODY_FILE_SHA256,
            "receipt_sha256",
            study.SOURCE_CUSTODY_SHA256,
        ),
        (
            "source_access_addendum",
            study.SOURCE_ACCESS_ADDENDUM_RELATIVE,
            study.SOURCE_ACCESS_ADDENDUM_SCHEMA,
            study.SOURCE_ACCESS_ADDENDUM_FILE_SHA256,
            "addendum_sha256",
            study.SOURCE_ACCESS_ADDENDUM_SHA256,
        ),
    ):
        protocol[role] = {
            "relative_path": relative,
            "schema": schema,
            "file_sha256": file_hash,
            semantic_field: semantic_hash,
            "committed_custody": custody(file_hash),
        }
    implementation_rows = [
        {
            "path": relative,
            "sha256": hashlib.sha256(relative.encode()).hexdigest(),
            "head_blob_sha256": hashlib.sha256(relative.encode()).hexdigest(),
            "clean_tracked_HEAD_blob": True,
        }
        for relative in study.IMPLEMENTATION_RELATIVE_FILES
    ]
    implementation = {
        "files": implementation_rows,
        "set_sha256": stable_hash(implementation_rows),
    }
    historical_commitments = {
        "item_id_set_sha256": study.HISTORICAL_ID_SET_SHA256,
        "normalized_question_sha256_set_sha256": "4" * 64,
        "canonical_question_plus_ordered_context_sha256_set_sha256": "5" * 64,
        "canonical_row_sha256_set_sha256": "6" * 64,
    }
    historical = {
        "workspace_relative_path": study.HISTORICAL_QUERY_WORKSPACE_RELATIVE.as_posix(),
        "file_sha256": study.HISTORICAL_QUERY_SHA256,
        "hipporag_commit": study.HIPPORAG_COMMIT,
        "git_blob_sha1": study.HIPPORAG_QUERY_GIT_BLOB_SHA1,
        "clean_tracked_HEAD_blob": True,
        "row_count": 1000,
        "set_counts": {
            "item_ids": 1000,
            "normalized_questions": 1000,
            "canonical_question_plus_ordered_contexts": 1000,
            "canonical_rows": 1000,
        },
        "set_commitments": historical_commitments,
        "item_level_content_persisted_publicly": False,
    }
    commitments = [
        study.BlockCommitment(
            block=block,
            source_member=study.BLOCK_SOURCE_MEMBERS[block],
            question_type_counts={
                question_type: study.BLOCK_PER_TYPE_COUNTS[block]
                for question_type in study.QUESTION_TYPES
            },
            count=study.BLOCK_COUNTS[block],
            file_sha256=hashlib.sha256(block.encode()).hexdigest(),
            item_commitment_set_sha256=hashlib.sha256(
                f"items:{block}".encode()
            ).hexdigest(),
        )
        for block in study.BLOCK_ORDER
    ]
    receipt: dict[str, Any] = {
        "schema": study.ACQUISITION_SCHEMA,
        "decision": "fresh_two_block_private_pack_formed_no_measurement_authority",
        "preregistration_sha256": "1" * 64,
        "preregistration_custody": custody("7" * 64),
        "public_protocol_bindings": protocol,
        "implementation": implementation,
        "selection_runtime": study.selection_runtime_binding(),
        "historical_denylist": historical,
        "source": {
            "archive_sha256": study.OFFICIAL_ARCHIVE_SHA256,
            "member_sha256s": dict(study.ARCHIVE_MEMBER_SHA256S),
            "source_row_counts": dict(study.ARCHIVE_MEMBER_ROW_COUNTS),
            "data_integrity_rejected_counts": {"train.json": 0, "dev.json": 0},
            "collision_metadata_rejected_counts": {
                "train.json": 0,
                "dev.json": 0,
                "test.json": 0,
            },
            "collision_only_members_never_selected": ["test.json"],
        },
        "selection": {
            "method": "private_HMAC_rank_within_exact_source_member_and_question_type",
            "domain_separator": study.SELECTION_DOMAIN_SEPARATOR,
            "selection_secret_commitment_sha256": (
                "fc1589f1c5453a2c115f89b315e11e0c9182e65e741afc53fc552ca4d5733d26"
            ),
            "question_type_order": list(study.QUESTION_TYPES),
            "block_source_members": dict(study.BLOCK_SOURCE_MEMBERS),
            "block_counts": dict(study.BLOCK_COUNTS),
            "selected_count": study.SELECTED_COUNT,
            "eligible_counts_by_member_and_type": {
                member: {
                    question_type: (
                        study.SOURCE_MEMBER_ROW_COUNTS[member]
                        - minimum * (len(study.QUESTION_TYPES) - 1)
                        if question_type == study.QUESTION_TYPES[0]
                        else minimum
                    )
                    for question_type in study.QUESTION_TYPES
                }
                for member, minimum in (("train.json", 12), ("dev.json", 6))
            },
            "exclusion_counts_by_member": {
                member: {
                    "historical": 0,
                    "question_collision": 0,
                    "duplicate_item_id": 0,
                }
                for member in study.SOURCE_MEMBER_SHA256S
            },
            "normalized_question_collision_class_count": 0,
            "item_id_collision_class_count": 0,
            "collision_scan_member_counts": dict(study.ARCHIVE_MEMBER_ROW_COUNTS),
        },
        "commitments": {
            "block_files": [row.to_dict() for row in commitments],
            "private_pack_sha256": stable_hash([row.to_dict() for row in commitments]),
            "private_locator_file_sha256": "3" * 64,
            "private_row_key_set_sha256": stable_hash(
                sorted(study.PRIVATE_BLOCK_ROW_KEYS)
            ),
            "item_ids_persisted_publicly": False,
            "private_paths_persisted_publicly": False,
        },
        "prospective_ordering": {
            "preregistration_committed_before_consumption": True,
            "persistence_preflight_complete_before_consumption": True,
            "pack_root_created_before_consumption": True,
            "consumption_persisted_before_source_archive_open": True,
            "source_rows_opened_before_consumption": 0,
            "acquisition_consumption_file_sha256": "8" * 64,
            "acquisition_consumption_sha256": "9" * 64,
            "retry_replay_resample_authorized": False,
        },
        "safety": {
            "formation_executed": False,
            "measurement_executed": False,
            "model_calls": 0,
            "network_calls": 0,
            "online_evaluator_calls": 0,
            "scores_computed": 0,
        },
    }
    receipt["acquisition_sha256"] = stable_hash(receipt)
    path = private_tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_public_receipt_loader_is_exact_and_rejects_tampering(private_tmp_path: Path) -> None:
    path = _minimal_public_receipt(private_tmp_path)
    receipt, blocks = study.load_acquisition_binding(path)
    assert receipt["selection"]["selected_count"] == 72
    assert [row.block for row in blocks] == ["A_hold", "M_search"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selection"]["selected_count"] = 73
    payload["acquisition_sha256"] = stable_hash(
        {key: value for key, value in payload.items() if key != "acquisition_sha256"}
    )
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(study.TwoWikiAcquisitionError, match="contract drifted"):
        study.load_acquisition_binding(path)
    path = _minimal_public_receipt(private_tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selection"]["eligible_counts_by_member_and_type"]["train.json"][
        study.QUESTION_TYPES[0]
    ] -= 1
    payload["acquisition_sha256"] = stable_hash(
        {key: value for key, value in payload.items() if key != "acquisition_sha256"}
    )
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(study.TwoWikiAcquisitionError, match="contract drifted"):
        study.load_acquisition_binding(path)
