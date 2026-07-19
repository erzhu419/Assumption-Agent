from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any

import pytest

from assumption_agent.benchmarks import (
    eraser_evidence_inference_source_qualification_v1 as audit,
)


PRIVATE_MARKERS = (
    "PRIVATE_ARCHIVE_ROOT_DO_NOT_LEAK",
    "PRIVATE_PROMPT_DO_NOT_LEAK",
    "PRIVATE_QUERY_DO_NOT_LEAK",
    "PRIVATE_TOKEN_DO_NOT_LEAK",
    "PRIVATE_INTERVENTION_DO_NOT_LEAK",
    "PRIVATE_COMPARATOR_DO_NOT_LEAK",
    "PRIVATE_OUTCOME_DO_NOT_LEAK",
    "PRIVATE_TEST_ROW_DO_NOT_OPEN",
    "PRIVATE_TEST_DOCUMENT_DO_NOT_OPEN",
    "PRIVATE_UNREFERENCED_DOCUMENT_DO_NOT_OPEN",
)

SYNTHETIC_GIT_BLOB = "a" * 40
SYNTHETIC_GIT_COMMIT = "b" * 40


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _with_self_hash(body: dict[str, Any], field: str) -> bytes:
    payload = copy.deepcopy(body)
    payload[field] = hashlib.sha256(_json_bytes(payload)).hexdigest()
    return _json_bytes(payload)


def _write_tar_member(bundle: tarfile.TarFile, name: str, raw: bytes) -> None:
    member = tarfile.TarInfo(name=name)
    member.size = len(raw)
    member.mtime = 0
    bundle.addfile(member, io.BytesIO(raw))


def _document_tokens(ordinal: int) -> tuple[str, ...]:
    return tuple(
        f"PRIVATE_TOKEN_DO_NOT_LEAK_{ordinal}_{position}" for position in range(10)
    )


def _annotation(
    *,
    split: str,
    family: str,
    ordinal: int,
    docid: str,
) -> dict[str, Any]:
    tokens = _document_tokens(ordinal)
    return {
        "annotation_id": f"PRIVATE_PROMPT_DO_NOT_LEAK_{split}_{ordinal}",
        "query": f"PRIVATE_QUERY_DO_NOT_LEAK_{split}_{ordinal}",
        "evidences": [
            [
                {
                    "text": " ".join(tokens[:2]),
                    "docid": docid,
                    "start_token": 0,
                    "end_token": 2,
                    "start_sentence": 0,
                    "end_sentence": 1,
                }
            ],
            [
                {
                    "text": list(tokens[2:]),
                    "docid": docid,
                    "start_token": 2,
                    "end_token": 4,
                    "start_sentence": 1,
                    "end_sentence": 2,
                }
            ],
        ],
        "classification": family,
        "query_type": "PRIVATE_QUERY_TYPE_DO_NOT_LEAK",
        "docids": [docid],
    }


def _datasets() -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, bytes],
    list[dict[str, str]],
]:
    datasets = {"train": [], "val": []}
    documents: dict[str, bytes] = {}
    sidecar_rows: list[dict[str, str]] = []
    global_ordinal = 0
    per_family = {"train": 28, "val": 20}
    pmcid = 1_000_000
    for split in ("train", "val"):
        for family in audit.RELATION_FAMILIES:
            for _ in range(per_family[split]):
                docid = f"PMC{pmcid}"
                annotation = _annotation(
                    split=split,
                    family=family,
                    ordinal=global_ordinal,
                    docid=docid,
                )
                datasets[split].append(annotation)
                tokens = _document_tokens(global_ordinal)
                documents[docid] = "".join(
                    f"{tokens[position]} {tokens[position + 1]}\n"
                    for position in range(0, 10, 2)
                ).encode("utf-8")
                sidecar_rows.append(
                    {
                        "PromptID": annotation["annotation_id"],
                        "PMCID": str(pmcid),
                        "Outcome": (
                            f"PRIVATE_OUTCOME_DO_NOT_LEAK_{global_ordinal}"
                        ),
                        "Intervention": (
                            f"PRIVATE_INTERVENTION_DO_NOT_LEAK_{global_ordinal}"
                        ),
                        "Comparator": (
                            f"PRIVATE_COMPARATOR_DO_NOT_LEAK_{global_ordinal}"
                        ),
                    }
                )
                global_ordinal += 1
                pmcid += 1
    return datasets, documents, sidecar_rows


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "PromptID",
            "PMCID",
            "Outcome",
            "Intervention",
            "Comparator",
            "UnusedOfficialColumn",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "UnusedOfficialColumn": "PRIVATE_UNUSED_DO_NOT_LEAK"})
    return output.getvalue().encode("utf-8")


def _source_manifests(
    tmp_path: Path,
    *,
    archive_sha256: str,
    archive_size: int,
    sidecar_sha256: str,
    sidecar_size: int,
) -> tuple[Path, Path, Path, Path, Path]:
    custody_body = {
        "schema": "eraser_evidence_inference_source_custody_v1",
        "archive_metadata": {
            "content_length": archive_size,
            "local_ignored_relative_path": "synthetic/evidence_inference.tar.gz",
        },
        "claim_boundary": {
            "archive_body_downloaded_or_opened": False,
            "dataset_member_or_row_listed_parsed_or_hashed": False,
            "retrieval_action_evaluator_or_score_run": False,
            "selection_secret_or_cohort_created": False,
            "test_query_document_or_label_opened": False,
        },
        "prospective_split_policy": {
            "test": "never open synthetic TEST",
        },
        "terminal_policy": {
            "test_use_authorized": False,
            "online_evaluation_fallback": False,
        },
    }
    custody_raw = _with_self_hash(custody_body, "source_custody_sha256")
    custody = tmp_path / "custody.json"
    custody.write_bytes(custody_raw)
    custody_payload = json.loads(custody_raw)

    access_body = {
        "schema": "eraser_evidence_inference_source_access_v1",
        "archive_binding": {
            "sha256": archive_sha256,
            "byte_size": archive_size,
            "local_relative_path": "synthetic/evidence_inference.tar.gz",
        },
        "custody_binding": {
            "custody_file_sha256": hashlib.sha256(custody_raw).hexdigest(),
            "custody_self_sha256": custody_payload["source_custody_sha256"],
            "custody_commit": "synthetic",
        },
        "pre_member_access_state": {
            "archive_member_content_opened_or_extracted": False,
            "archive_member_list_created": False,
            "dataset_member_individually_hashed": False,
            "source_schema_or_row_parsed": False,
            "test_member_name_query_document_label_or_content_opened": False,
        },
    }
    access = tmp_path / "access.json"
    access_raw = _with_self_hash(access_body, "source_access_sha256")
    access.write_bytes(access_raw)
    access_payload = json.loads(access_raw)

    prompt_access_body = {
        "schema": "eraser_evidence_inference_prompt_sidecar_access_v1",
        "access_boundary": {
            "content_rows_listed_parsed_or_printed": False,
            "exact_file_stat_git_blob_and_whole_file_sha256_only": True,
            "prompt_or_article_values_opened": False,
            "test_prompt_values_opened_or_used": False,
        },
        "binding": {
            "byte_size": sidecar_size,
            "git_blob_sha1": SYNTHETIC_GIT_BLOB,
            "git_commit": SYNTHETIC_GIT_COMMIT,
            "repository_path": "annotations/prompts_merged.csv",
            "sha256": sidecar_sha256,
        },
        "sidecar_contract": {
            "binding_key": "exact synthetic PromptID",
            "label_free_fields": ["Intervention", "Comparator", "Outcome"],
        },
        "source_custody_binding": {
            "custody_self_sha256": custody_payload["source_custody_sha256"],
        },
    }
    prompt_access = tmp_path / "prompt_access.json"
    prompt_access.write_bytes(
        _with_self_hash(
            prompt_access_body,
            "prompt_sidecar_access_sha256",
        )
    )

    tar_amendment_body = {
        "schema": "eraser_evidence_inference_tar_header_access_amendment_v1",
        "authorized_header_boundary": {
            "aggregate_counts": "synthetic aggregate counts only",
            "in_memory_routing": "synthetic nonpersistent header routing",
            "member_name_persistence_output_or_hash": False,
            "test_member_content_extract_open_read_hash_or_parse": False,
            "test_only_document_content_extract_open_read_hash_or_parse": False,
        },
        "base_bindings": {
            "archive_sha256": archive_sha256,
            "custody_self_sha256": custody_payload["source_custody_sha256"],
            "source_access_self_sha256": access_payload["source_access_sha256"],
        },
        "claim_boundary": {
            "action_evaluator_retrieval_or_score_changed": False,
            "archive_member_header_or_content_access_before_this_amendment": False,
            "cohort_family_quota_or_selection_changed": False,
            "online_evaluation_authorized": False,
            "test_query_document_label_or_content_authorized": False,
        },
        "narrow_supersession": {
            "unchanged_clause": "synthetic TEST content remains sealed",
        },
        "status": (
            "prospective_container_routing_correction_before_any_"
            "archive_member_header_access"
        ),
    }
    tar_amendment_raw = _with_self_hash(
        tar_amendment_body,
        "tar_header_access_amendment_sha256",
    )
    tar_amendment = tmp_path / "tar_header_amendment.json"
    tar_amendment.write_bytes(tar_amendment_raw)
    tar_amendment_payload = json.loads(tar_amendment_raw)

    design_amendment_body = {
        "schema": "eraser_evidence_inference_r7_e3_design_amendment_v1",
        "base_design_binding": {
            "design_self_sha256": "c" * 64,
        },
        "change_scope": {
            "action_operator_feature_evaluator_or_score_change": False,
            "cohort_family_quota_split_or_selection_change": False,
            "new_gate_threshold_retry_or_online_fallback": False,
            "source_qualification_test_policy": (
                "synthetic nonpersistent header routing only"
            ),
        },
        "prospective_state": {
            "archive_member_header_or_content_access_before_this_amendment": False,
            "private_assignment_or_secret_created": False,
            "retrieval_action_evaluator_or_score_run": False,
        },
        "tar_header_access_binding": {
            "amendment_file_sha256": hashlib.sha256(
                tar_amendment_raw
            ).hexdigest(),
            "amendment_self_sha256": tar_amendment_payload[
                "tar_header_access_amendment_sha256"
            ],
        },
    }
    design_amendment = tmp_path / "design_amendment.json"
    design_amendment.write_bytes(
        _with_self_hash(design_amendment_body, "design_amendment_sha256")
    )
    return custody, access, prompt_access, tar_amendment, design_amendment


def _fixture(
    tmp_path: Path,
    *,
    invalid_span: bool = False,
    empty_group: bool = False,
    cross_split_article_overlap: bool = False,
    sidecar_missing: bool = False,
    sidecar_duplicate: bool = False,
    sidecar_incomplete: bool = False,
    duplicate_normalized_query: bool = False,
    short_document: bool = False,
) -> dict[str, Any]:
    datasets, documents, sidecar_rows = _datasets()
    if invalid_span:
        evidence = datasets["train"][0]["evidences"][0][0]
        evidence["start_token"] = 99
        evidence["end_token"] = 100
    if empty_group:
        datasets["train"][0]["evidences"][0] = []
    if cross_split_article_overlap:
        old_docid = datasets["val"][0]["docids"][0]
        new_docid = datasets["train"][0]["docids"][0]
        datasets["val"][0]["docids"] = [new_docid]
        for group in datasets["val"][0]["evidences"]:
            for evidence in group:
                evidence["docid"] = new_docid
        del documents[old_docid]
        target_prompt = datasets["val"][0]["annotation_id"]
        for row in sidecar_rows:
            if row["PromptID"] == target_prompt:
                row["PMCID"] = new_docid.removeprefix("PMC")
                break
    if sidecar_missing:
        sidecar_rows.pop(0)
    if sidecar_duplicate:
        sidecar_rows.append(dict(sidecar_rows[0]))
    if sidecar_incomplete:
        sidecar_rows[0]["Outcome"] = ""
    if duplicate_normalized_query:
        datasets["train"][1]["query"] = (
            "  " + datasets["train"][0]["query"].upper() + "  "
        )
    if short_document:
        first_docid = datasets["train"][0]["docids"][0]
        documents[first_docid] = b"\n".join(
            documents[first_docid].splitlines()[:4]
        ) + b"\n"
    sidecar_rows.append(
        {
            "PromptID": "PRIVATE_TEST_ROW_DO_NOT_OPEN",
            "PMCID": "9999999",
            "Outcome": "PRIVATE_TEST_ROW_DO_NOT_OPEN",
            "Intervention": "PRIVATE_TEST_ROW_DO_NOT_OPEN",
            "Comparator": "PRIVATE_TEST_ROW_DO_NOT_OPEN",
        }
    )
    sidecar = tmp_path / "prompts_merged.csv"
    sidecar.write_bytes(_csv_bytes(sidecar_rows))

    root = "PRIVATE_ARCHIVE_ROOT_DO_NOT_LEAK/evidence_inference"
    archive = tmp_path / "evidence_inference.tar.gz"
    with tarfile.open(archive, mode="w:gz") as bundle:
        for split in ("train", "val"):
            raw = b"\n".join(
                json.dumps(row, ensure_ascii=False).encode("utf-8")
                for row in datasets[split]
            )
            _write_tar_member(bundle, f"{root}/{split}.jsonl", raw)
        _write_tar_member(
            bundle,
            f"{root}/test.jsonl",
            b'{"secret":"PRIVATE_TEST_ROW_DO_NOT_OPEN"}',
        )
        for docid, raw in documents.items():
            _write_tar_member(bundle, f"{root}/docs/{docid}", raw)
        _write_tar_member(
            bundle,
            f"{root}/docs/PRIVATE_UNREFERENCED_DOCUMENT_DO_NOT_OPEN",
            b"PRIVATE_UNREFERENCED_DOCUMENT_DO_NOT_OPEN",
        )
        link = tarfile.TarInfo(f"{root}/PRIVATE_TEST_DOCUMENT_DO_NOT_OPEN")
        link.type = tarfile.SYMTYPE
        link.linkname = "test.jsonl"
        bundle.addfile(link)

    archive_raw = archive.read_bytes()
    sidecar_raw = sidecar.read_bytes()
    custody, access, prompt_access, tar_amendment, design_amendment = (
        _source_manifests(
        tmp_path,
        archive_sha256=hashlib.sha256(archive_raw).hexdigest(),
        archive_size=len(archive_raw),
        sidecar_sha256=hashlib.sha256(sidecar_raw).hexdigest(),
        sidecar_size=len(sidecar_raw),
        )
    )
    return {
        "archive": archive,
        "custody": custody,
        "access": access,
        "sidecar": sidecar,
        "prompt_access": prompt_access,
        "tar_amendment": tar_amendment,
        "design_amendment": design_amendment,
        "archive_sha256": hashlib.sha256(archive_raw).hexdigest(),
        "archive_size": len(archive_raw),
        "sidecar_sha256": hashlib.sha256(sidecar_raw).hexdigest(),
        "sidecar_size": len(sidecar_raw),
    }


def _qualify(fixture: dict[str, Any]) -> dict[str, Any]:
    return audit.qualify_archive(
        fixture["archive"],
        fixture["custody"],
        fixture["access"],
        fixture["sidecar"],
        fixture["prompt_access"],
        fixture["tar_amendment"],
        fixture["design_amendment"],
        expected_archive_sha256=fixture["archive_sha256"],
        expected_archive_size=fixture["archive_size"],
        expected_prompt_sidecar_sha256=fixture["sidecar_sha256"],
        expected_prompt_sidecar_size=fixture["sidecar_size"],
        expected_prompt_sidecar_git_blob_sha1=SYNTHETIC_GIT_BLOB,
        expected_prompt_sidecar_git_commit=SYNTHETIC_GIT_COMMIT,
        expected_annotation_counts={"train": 84, "val": 60},
        expected_article_counts={"train": 84, "val": 60},
        cohort_demands=audit.FORMAL_COHORT_DEMANDS,
        enforce_formal_manifest_identity=False,
    )


def _contains_list(value: Any) -> bool:
    if isinstance(value, list):
        return True
    if isinstance(value, dict):
        return any(_contains_list(child) for child in value.values())
    return False


def test_aggregate_only_capacity_and_content_open_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    opened: list[str] = []
    original_extractfile = tarfile.TarFile.extractfile

    def spy_extractfile(
        bundle: tarfile.TarFile, member: tarfile.TarInfo | str
    ):
        opened.append(member.name if isinstance(member, tarfile.TarInfo) else member)
        return original_extractfile(bundle, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", spy_extractfile)
    receipt = _qualify(fixture)

    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    for marker in PRIVATE_MARKERS:
        assert marker not in serialized
    assert not _contains_list(receipt)
    assert all(not name.endswith("test.jsonl") for name in opened)
    assert all("PRIVATE_UNREFERENCED" not in name for name in opened)
    assert all("PRIVATE_TEST_DOCUMENT" not in name for name in opened)
    assert len(opened) == 2 + 84 + 60

    assert receipt["status"] == "passed_source_qualification_no_selection"
    assert receipt["opened_content_boundary"] == {
        "authorized_split_member_count": 2,
        "nonpersistent_in_memory_tar_header_routing_used": True,
        "referenced_document_member_count": 144,
        "test_member_content_open_count": 0,
        "unreferenced_document_content_open_count": 0,
        "member_name_or_path_emitted_count": 0,
    }
    assert receipt["independent_structured_prompt_binding"] == {
        "authorized_prompt_id_count": 144,
        "exact_one_to_one_match_count": 144,
        "missing_match_count": 0,
        "duplicate_or_ambiguous_match_count": 0,
        "matched_prompt_counts_by_split": {"train": 84, "val": 60},
        "independent_ico_field_count_per_prompt": 3,
        "unique_structured_ico_hash_count": 144,
        "duplicate_structured_ico_hash_occurrence_count": 0,
        "query_string_reverse_parsing_used": False,
        "unreferenced_or_test_row_persisted_or_emitted_count": 0,
    }
    assert receipt["split_aggregates"]["train"][
        "annotation_and_class_counts"
    ]["relation_family_counts"] == {
        "NO_SIGNIFICANT_DIFFERENCE": 28,
        "SIGNIFICANTLY_DECREASED": 28,
        "SIGNIFICANTLY_INCREASED": 28,
    }
    assert receipt["split_aggregates"]["val"][
        "annotation_and_class_counts"
    ]["relation_family_counts"] == {
        "NO_SIGNIFICANT_DIFFERENCE": 20,
        "SIGNIFICANTLY_DECREASED": 20,
        "SIGNIFICANTLY_INCREASED": 20,
    }
    train_gold = receipt["split_aggregates"]["train"][
        "gold_flattened_rationale_semantics"
    ]
    assert train_gold == {
        "all_alternative_evidence_groups_sentence_span_union_used": True,
        "best_group_or_single_group_selection_used": False,
        "complete_annotation_union_sentence_occurrence_count": 168,
        "complete_annotation_union_sentence_cardinality_counts": {"2": 84},
    }
    assert receipt["article_disjoint_capacity"]["train"][
        "maximum_article_disjoint_assignment_count"
    ] == 84
    assert receipt["article_disjoint_capacity"]["train"][
        "exact_article_disjoint_capacity_met"
    ] is True
    assert receipt["article_disjoint_capacity"]["val"][
        "maximum_article_disjoint_assignment_count"
    ] == 60
    assert receipt["article_disjoint_capacity"]["val"][
        "exact_article_disjoint_capacity_met"
    ] is True
    body = dict(receipt)
    declared = body.pop("qualification_sha256")
    assert declared == audit._stable_hash(body)


@pytest.mark.parametrize(
    "mutation", ["invalid_span", "empty_group", "short_document"]
)
def test_incomplete_evidence_or_alternative_group_removes_capacity(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(
        tmp_path,
        invalid_span=mutation == "invalid_span",
        empty_group=mutation == "empty_group",
        short_document=mutation == "short_document",
    )
    receipt = _qualify(fixture)
    assert receipt["status"] == "terminal_source_infeasible_no_selection"
    capacity = receipt["article_disjoint_capacity"]["train"]
    assert capacity["maximum_article_disjoint_assignment_count"] == 83
    assert capacity["article_disjoint_assignment_shortfall_count"] == 1
    assert capacity["exact_article_disjoint_capacity_met"] is False


def test_train_validation_article_overlap_is_terminal_without_resampling(
    tmp_path: Path,
) -> None:
    receipt = _qualify(_fixture(tmp_path, cross_split_article_overlap=True))
    assert receipt["status"] == "terminal_source_infeasible_no_selection"
    assert receipt["cross_split_article_disjointness"] == {
        "train_referenced_article_count": 84,
        "validation_referenced_article_count": 60,
        "train_validation_article_overlap_count": 1,
        "article_disjoint": False,
    }


def test_whole_normalized_duplicate_query_group_is_excluded_before_capacity(
    tmp_path: Path,
) -> None:
    receipt = _qualify(_fixture(tmp_path, duplicate_normalized_query=True))
    assert receipt["status"] == "terminal_source_infeasible_no_selection"
    assert receipt["duplicate_normalized_query_group_exclusion"] == {
        "normalization": "Unicode_NFKC_then_whitespace_collapse_then_casefold",
        "duplicate_group_count": 1,
        "excluded_annotation_count": 2,
        "excluded_group_or_query_value_emitted": False,
    }
    assert receipt["split_aggregates"]["train"][
        "annotation_completeness"
    ]["whole_duplicate_query_group_excluded_annotation_count"] == 2
    assert receipt["article_disjoint_capacity"]["train"][
        "maximum_article_disjoint_assignment_count"
    ] == 82


@pytest.mark.parametrize(
    ("keyword", "message"),
    [
        ("sidecar_missing", "absent from the prompt sidecar"),
        ("sidecar_duplicate", "duplicated in the prompt sidecar"),
        ("sidecar_incomplete", "ICO field is incomplete"),
    ],
)
def test_prompt_sidecar_missing_duplicate_or_incomplete_is_terminal(
    tmp_path: Path, keyword: str, message: str
) -> None:
    with pytest.raises(
        audit.EraserEvidenceInferenceQualificationError,
        match=message,
    ):
        _qualify(_fixture(tmp_path, **{keyword: True}))


def test_prompt_sidecar_manifest_tamper_fails_before_row_stream(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    payload = json.loads(fixture["prompt_access"].read_text(encoding="utf-8"))
    payload["binding"]["byte_size"] += 1
    fixture["prompt_access"].write_bytes(_json_bytes(payload))
    with pytest.raises(
        audit.EraserEvidenceInferenceQualificationError,
        match="self hash drifted",
    ):
        _qualify(fixture)


def test_container_amendment_scope_tamper_fails_before_archive_header_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    payload = json.loads(
        fixture["design_amendment"].read_text(encoding="utf-8")
    )
    payload.pop("design_amendment_sha256")
    payload["change_scope"][
        "action_operator_feature_evaluator_or_score_change"
    ] = True
    fixture["design_amendment"].write_bytes(
        _with_self_hash(payload, "design_amendment_sha256")
    )
    opened = False

    def forbidden_tar_open(*_args: Any, **_kwargs: Any):
        nonlocal opened
        opened = True
        raise AssertionError("archive header scan occurred before amendment check")

    monkeypatch.setattr(tarfile, "open", forbidden_tar_open)
    with pytest.raises(
        audit.EraserEvidenceInferenceQualificationError,
        match="design amendment scope",
    ):
        _qualify(fixture)
    assert opened is False


def test_duplicate_json_object_key_is_rejected_without_opening_test(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    # Rebuild only the authorized TRAIN member with a nested duplicate key.
    datasets, documents, _rows = _datasets()
    train_lines = [
        json.dumps(row, ensure_ascii=False).encode("utf-8")
        for row in datasets["train"]
    ]
    train_lines[0] = train_lines[0].replace(
        b'"classification":',
        b'"classification":"SIGNIFICANTLY_DECREASED","classification":',
        1,
    )
    val_raw = b"\n".join(
        json.dumps(row, ensure_ascii=False).encode("utf-8")
        for row in datasets["val"]
    )
    root = "PRIVATE_ARCHIVE_ROOT_DO_NOT_LEAK/evidence_inference"
    with tarfile.open(fixture["archive"], mode="w:gz") as bundle:
        _write_tar_member(bundle, f"{root}/train.jsonl", b"\n".join(train_lines))
        _write_tar_member(bundle, f"{root}/val.jsonl", val_raw)
        _write_tar_member(
            bundle,
            f"{root}/test.jsonl",
            b'{"secret":"PRIVATE_TEST_ROW_DO_NOT_OPEN"}',
        )
        for docid, raw in documents.items():
            _write_tar_member(bundle, f"{root}/docs/{docid}", raw)
    raw = fixture["archive"].read_bytes()
    fixture["archive_sha256"] = hashlib.sha256(raw).hexdigest()
    fixture["archive_size"] = len(raw)
    custody, access, prompt_access, tar_amendment, design_amendment = (
        _source_manifests(
        tmp_path,
        archive_sha256=fixture["archive_sha256"],
        archive_size=fixture["archive_size"],
        sidecar_sha256=fixture["sidecar_sha256"],
        sidecar_size=fixture["sidecar_size"],
        )
    )
    fixture["custody"] = custody
    fixture["access"] = access
    fixture["prompt_access"] = prompt_access
    fixture["tar_amendment"] = tar_amendment
    fixture["design_amendment"] = design_amendment
    with pytest.raises(
        audit.EraserEvidenceInferenceQualificationError,
        match="duplicate JSON object key",
    ):
        _qualify(fixture)
