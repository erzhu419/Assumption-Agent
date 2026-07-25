from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest

from assumption_agent.benchmarks import mmqa_p1_source_qualification_v1 as q


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION_PATH = (
    PROJECT_ROOT / "manifests/mmqa_p1_source_download_authorization_v1.json"
)
PLACEHOLDER = "TO_BE_PATCHED_DOWNLOAD_AUTHORIZATION_SELF_SHA256"
TYPE_BY_FAMILY = {family: exact for exact, family in q.FAMILY_BY_EXACT_TYPE.items()}


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).rstrip(b"\n")).hexdigest()


def _gzip_jsonl(rows: list[dict[str, object]]) -> bytes:
    return gzip.compress(b"".join(_canonical(row) for row in rows), mtime=0)


def _cell(gold_name: str, *, canonical_variant: bool = False) -> dict[str, object]:
    title = gold_name.replace(" ", "_") if canonical_variant else gold_name
    scheme = "http" if canonical_variant else "https"
    return {
        "text": "SECRET_TABLE_CELL_" + gold_name,
        "links": [
            {
                "text": gold_name,
                "wiki_title": title,
                "url": f"{scheme}://en.wikipedia.org/wiki/{title}",
            }
        ],
    }


def _records(
    *, shared_dev_table: bool = False, canonical_variant: bool = False
) -> dict[str, list[dict[str, object]]]:
    texts: list[dict[str, object]] = []
    tables_by_id: dict[str, dict[str, object]] = {}
    train: list[dict[str, object]] = []
    dev: list[dict[str, object]] = []

    for split, per_family, target in (("TRAIN", 1, train), ("DEV", 3, dev)):
        for family_index, family in enumerate(q.FAMILIES):
            for item_index in range(per_family):
                suffix = f"{split}_{family_index}_{item_index}"
                gold_id = "SECRET_GOLD_TEXT_ID_" + suffix
                distractor_id = "SECRET_DISTRACTOR_TEXT_ID_" + suffix
                gold_name = "Secret Gold Article " + suffix
                distractor_name = "Secret Distractor Article " + suffix
                texts.extend(
                    [
                        {
                            "id": gold_id,
                            "title": gold_name,
                            "url": "https://en.wikipedia.org/wiki/"
                            + gold_name.replace(" ", "_"),
                            "text": "SECRET_GOLD_TEXT_CONTENT_" + suffix,
                        },
                        {
                            "id": distractor_id,
                            "title": distractor_name,
                            "url": "https://en.wikipedia.org/wiki/"
                            + distractor_name.replace(" ", "_"),
                            "text": "SECRET_DISTRACTOR_TEXT_CONTENT_" + suffix,
                        },
                    ]
                )
                if split == "DEV" and shared_dev_table:
                    table_id = "SECRET_SHARED_DEV_TABLE_ID"
                else:
                    table_id = "SECRET_TABLE_ID_" + suffix
                if table_id not in tables_by_id:
                    tables_by_id[table_id] = {
                        "id": table_id,
                        "title": "SECRET_TABLE_TITLE_" + table_id,
                        "url": "https://en.wikipedia.org/wiki/Secret_Table",
                        "table": {
                            "table_name": "SECRET_TABLE_NAME",
                            "header": [{"column_name": "SECRET_COLUMN"}],
                            "table_rows": [],
                        },
                    }
                table_rows = tables_by_id[table_id]["table"]["table_rows"]
                assert isinstance(table_rows, list)
                row_index = len(table_rows)
                table_rows.append(
                    [_cell(gold_name, canonical_variant=canonical_variant)]
                )
                target.append(
                    {
                        "qid": "SECRET_QID_" + suffix,
                        "question": "SECRET_QUESTION_CONTENT_" + suffix,
                        "answers": [
                            {
                                "answer": "SECRET_ANSWER_" + suffix,
                                "type": "string",
                                "modality": "table",
                                "text_instances": [],
                                "table_indices": [[row_index, 0]],
                                "image_instances": [],
                            }
                        ],
                        "metadata": {
                            "type": TYPE_BY_FAMILY[family],
                            "modalities": ["table", "text"],
                            "text_doc_ids": [gold_id, distractor_id],
                            "image_doc_ids": [],
                            "table_id": table_id,
                            "intermediate_answers": [
                                "SECRET_INTERMEDIATE_ANSWER_" + suffix
                            ],
                        },
                        "supporting_context": [
                            {"doc_id": table_id, "doc_part": "table"},
                            {"doc_id": gold_id, "doc_part": "text"},
                        ],
                    }
                )
    return {
        "MMQA_train.jsonl.gz": train,
        "MMQA_dev.jsonl.gz": dev,
        "MMQA_tables.jsonl.gz": list(tables_by_id.values()),
        "MMQA_texts.jsonl.gz": texts,
    }


def _write_fixture(
    tmp_path: Path,
    records: dict[str, list[dict[str, object]]],
    *,
    train_quota: int = 1,
    dev_quotas: dict[str, int] | None = None,
) -> tuple[dict[str, Path], q.QualificationContract]:
    paths: dict[str, Path] = {}
    contracts: dict[str, q.SourceFileContract] = {}
    for file_name, rows in records.items():
        raw = _gzip_jsonl(rows)
        path = tmp_path / file_name
        path.write_bytes(raw)
        os.chmod(path, 0o600)
        paths[file_name] = path
        contracts[file_name] = q.SourceFileContract(
            file_name=file_name,
            size_bytes=len(raw),
            git_blob_sha1=q._git_blob_sha1(raw),
            maximum_uncompressed_bytes=20_000_000,
            maximum_records=10_000,
            maximum_line_bytes=2_000_000,
        )
    contract = q.QualificationContract(
        files=contracts,
        expected_train_rows=len(records["MMQA_train.jsonl.gz"]),
        expected_dev_rows=len(records["MMQA_dev.jsonl.gz"]),
        train_quota_per_family=train_quota,
        dev_block_quotas=dev_quotas
        or {"F_search": 1, "A_hold": 1, "M_search": 1},
    )
    return paths, contract


def test_valid_fixture_emits_only_aggregate_capacity(tmp_path: Path) -> None:
    paths, contract = _write_fixture(tmp_path, _records())
    result = q._qualify_sources(paths, contract)
    assert result["qualified"] is True
    assert result["TRAIN"]["eligible_count_by_family"] == {
        family: 1 for family in q.FAMILIES
    }
    assert result["DEV"]["eligible_count_by_family"] == {
        family: 3 for family in q.FAMILIES
    }
    assert result["DEV"]["component_disjoint_capacity"]["qualified"] is True
    assert result["DEV"]["component_disjoint_capacity"]["block_capacity"] == {
        block: {family: 1 for family in q.FAMILIES}
        for block in ("F_search", "A_hold", "M_search")
    }
    serialized = json.dumps(result, sort_keys=True)
    for forbidden in (
        "SECRET_QID",
        "SECRET_QUESTION",
        "SECRET_ANSWER",
        "SECRET_GOLD_TEXT_ID",
        "SECRET_TABLE_ID",
        "SECRET_TABLE_CELL",
        "SECRET_INTERMEDIATE",
    ):
        assert forbidden not in serialized


def test_exact_canonical_title_and_url_link_is_eligible(tmp_path: Path) -> None:
    paths, contract = _write_fixture(
        tmp_path, _records(canonical_variant=True)
    )
    assert q._qualify_sources(paths, contract)["qualified"] is True


def test_all_four_blob_identities_pass_before_any_gzip_parse(
    tmp_path: Path,
) -> None:
    paths, contract = _write_fixture(tmp_path, _records())
    bad_path = paths["MMQA_dev.jsonl.gz"]
    raw = bytearray(bad_path.read_bytes())
    raw[-1] ^= 1
    bad_path.write_bytes(raw)
    os.chmod(bad_path, 0o600)
    with pytest.raises(q.MMQAP1SourceQualificationError, match="Git-blob"):
        q._qualify_sources(paths, contract)


def test_private_regular_file_is_required(tmp_path: Path) -> None:
    paths, contract = _write_fixture(tmp_path, _records())
    os.chmod(paths["MMQA_train.jsonl.gz"], 0o644)
    with pytest.raises(q.MMQAP1SourceQualificationError, match="private"):
        q._qualify_sources(paths, contract)


def test_bounded_gzip_jsonl_rejects_long_line() -> None:
    raw = gzip.compress(b'{"x":"' + b"a" * 500 + b'"}\n', mtime=0)
    contract = q.SourceFileContract(
        file_name="synthetic.jsonl.gz",
        size_bytes=len(raw),
        git_blob_sha1=q._git_blob_sha1(raw),
        maximum_uncompressed_bytes=10_000,
        maximum_records=10,
        maximum_line_bytes=64,
    )
    with pytest.raises(q.MMQAP1SourceQualificationError, match="bounded"):
        list(q._iter_gzip_jsonl(raw, contract))


def test_duplicate_json_key_is_rejected() -> None:
    with pytest.raises(q.MMQAP1SourceQualificationError, match="duplicate"):
        q._parse_json_line(b'{"id":"a","id":"b"}')


def test_nonfinite_json_value_is_rejected() -> None:
    with pytest.raises(q.MMQAP1SourceQualificationError, match="non-finite"):
        q._parse_json_line(b'{"value":NaN}')


def test_gold_support_text_must_resolve_inside_candidate_pool(
    tmp_path: Path,
) -> None:
    records = _records()
    row = records["MMQA_dev.jsonl.gz"][0]
    row["supporting_context"][1]["doc_id"] = "SECRET_NOT_A_CANDIDATE"
    paths, contract = _write_fixture(tmp_path, records)
    with pytest.raises(q.MMQAP1SourceQualificationError, match="capacity"):
        q._qualify_sources(paths, contract)


def test_table_support_must_resolve_to_exact_anchor(tmp_path: Path) -> None:
    records = _records()
    row = records["MMQA_dev.jsonl.gz"][0]
    row["supporting_context"][0]["doc_id"] = "SECRET_OTHER_TABLE"
    paths, contract = _write_fixture(tmp_path, records)
    with pytest.raises(q.MMQAP1SourceQualificationError, match="capacity"):
        q._qualify_sources(paths, contract)


def test_exact_gold_row_text_pair_is_required(tmp_path: Path) -> None:
    records = _records()
    table_id = records["MMQA_dev.jsonl.gz"][0]["metadata"]["table_id"]
    table = next(
        row
        for row in records["MMQA_tables.jsonl.gz"]
        if row["id"] == table_id
    )
    table["table"]["table_rows"][0][0]["links"][0]["wiki_title"] = (
        "SECRET_UNRELATED_TITLE"
    )
    table["table"]["table_rows"][0][0]["links"][0]["url"] = (
        "https://en.wikipedia.org/wiki/SECRET_UNRELATED_TITLE"
    )
    paths, contract = _write_fixture(tmp_path, records)
    with pytest.raises(q.MMQAP1SourceQualificationError, match="capacity"):
        q._qualify_sources(paths, contract)


def test_gold_row_union_must_have_at_most_four_rows(tmp_path: Path) -> None:
    records = _records()
    item = records["MMQA_dev.jsonl.gz"][0]
    table_id = item["metadata"]["table_id"]
    table = next(
        row
        for row in records["MMQA_tables.jsonl.gz"]
        if row["id"] == table_id
    )
    gold_id = item["supporting_context"][1]["doc_id"]
    text = next(
        row for row in records["MMQA_texts.jsonl.gz"] if row["id"] == gold_id
    )
    for _ in range(4):
        table["table"]["table_rows"].append([_cell(text["title"])])
    item["answers"][0]["table_indices"] = [[index, 0] for index in range(5)]
    paths, contract = _write_fixture(tmp_path, records)
    with pytest.raises(q.MMQAP1SourceQualificationError, match="capacity"):
        q._qualify_sources(paths, contract)


def test_answer_table_index_column_must_resolve(tmp_path: Path) -> None:
    records = _records()
    records["MMQA_dev.jsonl.gz"][0]["answers"][0]["table_indices"] = [[0, 9]]
    paths, contract = _write_fixture(tmp_path, records)
    with pytest.raises(q.MMQAP1SourceQualificationError, match="capacity"):
        q._qualify_sources(paths, contract)


def test_text_pool_requires_a_resolved_nongold_distractor(
    tmp_path: Path,
) -> None:
    records = _records()
    item = records["MMQA_dev.jsonl.gz"][0]
    item["metadata"]["text_doc_ids"] = [
        item["supporting_context"][1]["doc_id"]
    ]
    paths, contract = _write_fixture(tmp_path, records)
    with pytest.raises(q.MMQAP1SourceQualificationError, match="capacity"):
        q._qualify_sources(paths, contract)


def test_component_overlap_cannot_be_split_across_dev_blocks(
    tmp_path: Path,
) -> None:
    paths, contract = _write_fixture(
        tmp_path, _records(shared_dev_table=True)
    )
    with pytest.raises(q.MMQAP1SourceQualificationError, match="component"):
        q._qualify_sources(paths, contract)


def test_unselected_question_type_does_not_substitute_a_family(
    tmp_path: Path,
) -> None:
    records = _records()
    records["MMQA_dev.jsonl.gz"][0]["metadata"]["type"] = "TextQ"
    paths, contract = _write_fixture(tmp_path, records)
    with pytest.raises(q.MMQAP1SourceQualificationError, match="capacity"):
        q._qualify_sources(paths, contract)


def test_expected_sha256_freeze_binding_is_enforced(tmp_path: Path) -> None:
    paths, contract = _write_fixture(tmp_path, _records())
    expected = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    expected["MMQA_dev.jsonl.gz"] = "0" * 64
    with pytest.raises(q.MMQAP1SourceQualificationError, match="SHA256"):
        q._qualify_sources(
            paths, contract, expected_sha256_by_file=expected
        )


def test_one_shot_markers_cannot_be_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "attempt/marker.json"
    source_marker = tmp_path / "attempt/source.json"
    monkeypatch.setattr(q, "MARKER_PATH", marker)
    monkeypatch.setattr(q, "SOURCE_OPEN_MARKER_PATH", source_marker)
    assert len(q._consume_marker()) == 64
    assert len(q._consume_source_open_marker()) == 64
    with pytest.raises(FileExistsError):
        q._consume_marker()
    with pytest.raises(FileExistsError):
        q._consume_source_open_marker()


def test_exclusive_result_cannot_be_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    q._write_exclusive(path, {"schema": "first"})
    with pytest.raises(FileExistsError):
        q._write_exclusive(path, {"schema": "replacement"})
    assert json.loads(path.read_text("ascii")) == {"schema": "first"}


def test_authorization_and_frozen_source_bindings_are_exact() -> None:
    q._load_verified_manifest(q.CUSTODY_PATH, q.EXPECTED_CUSTODY_SELF_SHA256)
    q._load_verified_manifest(q.DESIGN_PATH, q.EXPECTED_DESIGN_SELF_SHA256)
    authorization = json.loads(AUTHORIZATION_PATH.read_text("ascii"))
    body = dict(authorization)
    claimed = body.pop("self_sha256")
    assert _semantic_hash(body) == claimed
    assert q.EXPECTED_DOWNLOAD_AUTHORIZATION_SELF_SHA256 == claimed
    assert authorization["source_custody_self_sha256"] == (
        q.EXPECTED_CUSTODY_SELF_SHA256
    )
    assert authorization["study_design_self_sha256"] == q.EXPECTED_DESIGN_SELF_SHA256
    assert authorization["total_authorized_bytes"] == 69_204_571
    files = authorization["one_shot_four_file_download"]["files"]
    assert set(files) == set(q.FORMAL_CONTRACT.files)
    for file_name, contract in q.FORMAL_CONTRACT.files.items():
        assert files[file_name]["expected_size_bytes"] == contract.size_bytes
        assert files[file_name]["expected_git_blob_sha1"] == contract.git_blob_sha1
    current = Path(q.__file__).read_bytes()
    normalized = current.replace(claimed.encode("ascii"), PLACEHOLDER.encode("ascii"))
    assert hashlib.sha256(normalized).hexdigest() == authorization[
        "implementation_binding"
    ]["qualifier_sha256_before_authorized_constant_patch"]
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == authorization[
        "implementation_binding"
    ]["test_sha256"]


def test_fixture_mutation_does_not_modify_original_builder() -> None:
    original = _records()
    mutated = copy.deepcopy(original)
    mutated["MMQA_dev.jsonl.gz"][0]["metadata"]["type"] = "TextQ"
    assert original["MMQA_dev.jsonl.gz"][0]["metadata"]["type"] != "TextQ"
