from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import warnings
import zipfile

import pytest

from assumption_agent.benchmarks import contractnli_fresh_source_qualification_v1 as q


def _canonical_hash(payload: dict[str, object]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_manifest(path: Path, schema: str, hash_field: str) -> Path:
    payload: dict[str, object] = {"schema": schema, "synthetic_fixture": True}
    payload[hash_field] = _canonical_hash(payload)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def synthetic_manifests(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        _write_manifest(
            tmp_path / "custody.json",
            q.FORMAL_CUSTODY_SCHEMA,
            "custody_sha256",
        ),
        _write_manifest(
            tmp_path / "addendum.json",
            q.FORMAL_ADDENDUM_SCHEMA,
            "addendum_sha256",
        ),
        _write_manifest(
            tmp_path / "member.json",
            q.FORMAL_MEMBER_SCHEMA,
            "source_member_binding_sha256",
        ),
    )


def _labels() -> dict[str, dict[str, str]]:
    return {
        f"label_{index:02d}": {"hypothesis": f"Synthetic hypothesis {index}"}
        for index in range(q.LABEL_COUNT)
    }


def _word_spans(text: str) -> list[list[int]]:
    return [[match.start(), match.end()] for match in re.finditer(r"\S+", text)]


def _document(
    document_id: int,
    *,
    text: str | None = None,
    file_name: str = "private.pdf",
    url: str = "https://private.invalid/contract",
    eligible_labels: int = 1,
    duplicate_gold: bool = False,
) -> dict[str, object]:
    if text is None:
        text = " ".join([f"word{index}" for index in range(18)] + [f"unique{document_id}"])
    spans = _word_spans(text)[:18]
    assert len(spans) == 18
    annotations: dict[str, dict[str, object]] = {}
    for index, label_id in enumerate(_labels()):
        if index < eligible_labels:
            annotations[label_id] = {
                "choice": "Entailment" if index % 2 == 0 else "Contradiction",
                "spans": [0, 0] if duplicate_gold else [0],
            }
        else:
            annotations[label_id] = {"choice": "NotMentioned", "spans": []}
    return {
        "id": document_id,
        "text": text,
        "file_name": file_name,
        "url": url,
        "document_type": "search-pdf",
        "spans": spans,
        "annotation_sets": [{"annotations": annotations}],
    }


def _dataset(documents: list[object]) -> dict[str, object]:
    return {"labels": _labels(), "documents": documents}


def _write_zip(
    path: Path,
    train: dict[str, object] | bytes,
    *,
    decoys: bool = True,
    extra_infos: list[tuple[zipfile.ZipInfo | str, bytes]] | None = None,
) -> Path:
    if isinstance(train, dict):
        train_raw = json.dumps(train, ensure_ascii=False).encode("utf-8")
    else:
        train_raw = train
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(q.TRAIN_MEMBER, train_raw)
        if decoys:
            archive.writestr("contract-nli/dev.json", b"not JSON and must stay unopened")
            archive.writestr("contract-nli/test.json", b"not JSON and must stay unopened")
            archive.writestr("contract-nli/raw/private.txt", b"private decoy")
        for info, value in extra_infos or []:
            archive.writestr(info, value)
    return path


def _build_synthetic(
    tmp_path: Path,
    manifests: tuple[Path, Path, Path],
    documents: list[object],
) -> dict[str, object]:
    archive = _write_zip(tmp_path / "synthetic.zip", _dataset(documents))
    return q.build_qualification(archive, *manifests)


def test_clean_worker_opens_exact_train_and_ignores_decoys(
    tmp_path: Path,
    synthetic_manifests: tuple[Path, Path, Path],
) -> None:
    train_payload = _dataset([_document(1)])
    train_raw = json.dumps(train_payload, ensure_ascii=False).encode("utf-8")
    archive = _write_zip(tmp_path / "synthetic.zip", train_raw)
    receipt = q.run_clean_qualification(archive, *synthetic_manifests)

    assert receipt["status"] == q.STATUS_DIAGNOSTIC
    assert receipt["archive"]["train_member"] == q.TRAIN_MEMBER
    assert receipt["archive"]["train_member_sha256"] == hashlib.sha256(
        train_raw
    ).hexdigest()
    assert receipt["qualification_operations"] == {
        "train_members_opened": 1,
        "dev_members_opened": 0,
        "test_members_opened": 0,
        "raw_contract_members_opened": 0,
        "selection_or_sampling_operations": 0,
        "selection_secret_files_opened": 0,
        "concrete_document_or_label_identifiers_emitted": 0,
        "source_text_span_gold_or_annotation_rows_emitted": 0,
        "source_member_provenance_fingerprints_emitted": 1,
        "private_row_content_fingerprints_emitted": 0,
    }
    serialized = json.dumps(receipt, sort_keys=True)
    assert "private decoy" not in serialized
    assert "Synthetic hypothesis" not in serialized
    assert "word0" not in serialized


def test_official_loader_span_branches_preserve_identity_and_duplicates(
    tmp_path: Path,
    synthetic_manifests: tuple[Path, Path, Path],
) -> None:
    text = "A A BC D E F G H I J K L M N O P Q R tail"
    document = _document(7, text=text, duplicate_gold=True)
    spans = document["spans"]
    assert isinstance(spans, list)
    spans[2] = [0, 3]  # shared start and overlap; do not coalesce it
    spans[4] = list(spans[3])  # repeated boundary remains a distinct node
    receipt = _build_synthetic(tmp_path, synthetic_manifests, [document])
    graph = receipt["aggregate"]["addressable_graph"]

    assert graph["valid_document_node_count_total"] == 18
    assert graph["shared_start_count"] == 2
    assert graph["repeated_boundary_count"] == 1
    assert graph["duplicate_node_text_count"] >= 2
    assert graph["overlapping_span_pair_count"] >= 2
    assert receipt["aggregate"]["eligibility"]["duplicate_gold_index_count"] == 1
    assert receipt["aggregate"]["eligibility"]["document_with_eligible_item_count"] == 1


def test_document_anomalies_are_counted_and_do_not_abort(
    tmp_path: Path,
    synthetic_manifests: tuple[Path, Path, Path],
) -> None:
    documents: list[object] = [_document(1)]

    documents.append("not-an-object")
    invalid_type = _document(2)
    invalid_type["text"] = 3
    documents.append(invalid_type)
    invalid_doc_type = _document(3)
    invalid_doc_type["document_type"] = "other"
    documents.append(invalid_doc_type)
    invalid_span = _document(4)
    invalid_span["spans"] = [[0, 999_999]]
    documents.append(invalid_span)
    invalid_sets = _document(5)
    invalid_sets["annotation_sets"] = []
    documents.append(invalid_sets)
    invalid_keys = _document(6)
    annotations = invalid_keys["annotation_sets"][0]["annotations"]
    del annotations[next(iter(annotations))]
    documents.append(invalid_keys)
    invalid_annotation = _document(7)
    annotations = invalid_annotation["annotation_sets"][0]["annotations"]
    annotations[next(iter(annotations))] = []
    documents.append(invalid_annotation)
    invalid_choice = _document(8)
    annotations = invalid_choice["annotation_sets"][0]["annotations"]
    annotations[next(iter(annotations))]["choice"] = "Unknown"
    documents.append(invalid_choice)
    invalid_consistency = _document(9)
    annotations = invalid_consistency["annotation_sets"][0]["annotations"]
    label_id = list(annotations)[1]
    annotations[label_id] = {"choice": "NotMentioned", "spans": [0]}
    documents.append(invalid_consistency)
    invalid_gold = _document(10)
    annotations = invalid_gold["annotation_sets"][0]["annotations"]
    annotations[next(iter(annotations))]["spans"] = [999]
    documents.append(invalid_gold)
    documents.extend([_document(11), _document(11)])

    receipt = _build_synthetic(tmp_path, synthetic_manifests, documents)
    root = receipt["aggregate"]["root"]
    anomalies = receipt["aggregate"]["document_anomalies"]

    assert root == {
        "label_count": 17,
        "document_count": 13,
        "valid_document_count": 1,
        "invalid_document_count": 12,
    }
    assert anomalies == {
        "document_not_object": 1,
        "required_field_type": 1,
        "duplicate_document_id": 2,
        "invalid_document_type": 1,
        "invalid_span": 1,
        "invalid_annotation_set": 1,
        "invalid_annotation_keys": 1,
        "invalid_annotation": 1,
        "invalid_choice": 1,
        "invalid_choice_span_consistency": 1,
        "invalid_gold_index": 1,
    }


def test_normalized_content_groups_one_document_cap_and_exposure(
    tmp_path: Path,
    synthetic_manifests: tuple[Path, Path, Path],
) -> None:
    base = " ".join(f"term{index}" for index in range(18))
    normalized_duplicate = "  ".join(f"TERM{index}" for index in range(18))
    exposed_text = q.FULL_TEXT_SIGNATURES[0] + " " + base
    documents = [
        _document(1, text=base, eligible_labels=2),
        _document(2, text=normalized_duplicate),
        _document(3, text=exposed_text),
        _document(4, file_name="EXAMPLE.PDF"),
        _document(5, url="HTTPS://EXAMPLECONTRACT.COM/private"),
    ]
    receipt = _build_synthetic(tmp_path, synthetic_manifests, documents)
    eligibility = receipt["aggregate"]["eligibility"]

    assert eligibility["eligible_item_count_before_one_per_document_cap"] == 6
    assert eligibility["document_with_eligible_item_count"] == 5
    assert eligibility["exposure_excluded_document_count"] == 3
    assert eligibility["full_text_signature_excluded_document_count"] == 1
    assert eligibility["metadata_file_excluded_document_count"] == 1
    assert eligibility["metadata_url_excluded_document_count"] == 1
    assert eligibility["eligible_document_count_after_exposure"] == 2
    assert eligibility["eligible_normalized_content_group_count"] == 1
    assert eligibility["duplicate_content_group_count"] == 1
    assert eligibility["documents_in_duplicate_content_groups"] == 2


def test_capacity_uses_256_distinct_content_groups() -> None:
    aggregate = q._aggregate_train(
        _dataset([_document(index) for index in range(q.MIN_CONTENT_GROUPS)])
    )
    eligibility = aggregate["eligibility"]
    assert eligibility["eligible_document_count_after_exposure"] == 256
    assert eligibility["eligible_normalized_content_group_count"] == 256
    assert eligibility["capacity_satisfied"] is True


def test_gold_zero_and_more_than_five_are_valid_but_ineligible() -> None:
    eligible = _document(1)
    zero_gold = _document(2)
    zero_annotations = zero_gold["annotation_sets"][0]["annotations"]
    zero_annotations[next(iter(zero_annotations))]["spans"] = []
    six_gold = _document(3)
    six_annotations = six_gold["annotation_sets"][0]["annotations"]
    six_annotations[next(iter(six_annotations))]["spans"] = list(range(6))

    aggregate = q._aggregate_train(_dataset([eligible, zero_gold, six_gold]))
    assert aggregate["root"]["valid_document_count"] == 3
    assert aggregate["root"]["invalid_document_count"] == 0
    assert aggregate["eligibility"]["eligible_item_count_before_one_per_document_cap"] == 1
    assert aggregate["eligibility"]["document_with_eligible_item_count"] == 1


def test_node_eligibility_is_inclusive_18_through_128() -> None:
    documents: list[dict[str, object]] = []
    for document_id, node_count in enumerate((17, 18, 128, 129), start=1):
        text = " ".join(f"d{document_id}node{index}" for index in range(node_count))
        document = _document(document_id, text=text if node_count >= 18 else None)
        if node_count == 17:
            document["spans"] = document["spans"][:17]
        else:
            document["spans"] = _word_spans(text)
        documents.append(document)

    aggregate = q._aggregate_train(_dataset(documents))
    graph = aggregate["addressable_graph"]
    eligibility = aggregate["eligibility"]
    assert aggregate["root"]["valid_document_count"] == 4
    assert graph["valid_document_node_count_min"] == 17
    assert graph["valid_document_node_count_max"] == 129
    assert graph["node_eligible_document_count"] == 2
    assert eligibility["document_with_eligible_item_count"] == 2
    assert eligibility["eligible_normalized_content_group_count"] == 2


def test_unknown_fields_nfkc_groups_and_literal_filename_casefold() -> None:
    base = " ".join(f"term{index}" for index in range(18))
    fullwidth = "".join(
        chr(ord(character) + 0xFEE0) if "!" <= character <= "~" else character
        for character in base.upper()
    )
    first = _document(1, text=base)
    first["supplemental_document_field"] = "ignored private value"
    first_annotations = first["annotation_sets"][0]["annotations"]
    first_annotations[next(iter(first_annotations))]["supplemental"] = "ignored"
    second = _document(2, text=fullwidth, file_name="ｅｘａｍｐｌｅ．ｐｄｆ")
    payload = _dataset([first, second])
    payload["supplemental_root_field"] = "ignored"
    for label in payload["labels"].values():
        label["short_description"] = "ignored"

    aggregate = q._aggregate_train(payload)
    eligibility = aggregate["eligibility"]
    assert aggregate["root"]["valid_document_count"] == 2
    assert eligibility["metadata_file_excluded_document_count"] == 0
    assert eligibility["eligible_normalized_content_group_count"] == 1
    assert eligibility["documents_in_duplicate_content_groups"] == 2


def test_lone_surrogate_document_is_counted_but_label_hypothesis_is_root_fatal(
    tmp_path: Path,
    synthetic_manifests: tuple[Path, Path, Path],
) -> None:
    surrogate_text = "\ud800 " + " ".join(f"word{index}" for index in range(17))
    invalid_document = _document(1, text=surrogate_text)
    raw = json.dumps(
        _dataset([invalid_document, _document(2)]), ensure_ascii=True
    ).encode("utf-8")
    archive = _write_zip(tmp_path / "surrogate-document.zip", raw)
    receipt = q.build_qualification(archive, *synthetic_manifests)
    assert receipt["aggregate"]["root"]["valid_document_count"] == 1
    assert receipt["aggregate"]["document_anomalies"]["required_field_type"] == 1

    payload = _dataset([_document(3)])
    payload["labels"][next(iter(payload["labels"]))]["hypothesis"] = "\ud800"
    raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    archive = _write_zip(tmp_path / "surrogate-label.zip", raw)
    with pytest.raises(q.ContractNliQualificationError, match="hypothesis"):
        q.build_qualification(archive, *synthetic_manifests)


def test_formal_train_size_and_crc_binding_on_synthetic_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_zip(tmp_path / "synthetic.zip", _dataset([_document(1)]))
    binding = q._hash_archive(archive)
    with zipfile.ZipFile(archive) as reader:
        info = reader.getinfo(q.TRAIN_MEMBER)
        expected_size = info.file_size
        expected_crc = f"{info.CRC & 0xFFFFFFFF:08x}"

    monkeypatch.setattr(q, "FORMAL_TRAIN_SIZE", expected_size + 1)
    monkeypatch.setattr(q, "FORMAL_TRAIN_CRC32", expected_crc)
    with pytest.raises(q.ContractNliQualificationError, match="TRAIN member binding"):
        q._read_train_member(archive, formal=True, initial_archive_binding=binding)

    monkeypatch.setattr(q, "FORMAL_TRAIN_SIZE", expected_size)
    monkeypatch.setattr(q, "FORMAL_TRAIN_CRC32", "00000000")
    with pytest.raises(q.ContractNliQualificationError, match="TRAIN member binding"):
        q._read_train_member(archive, formal=True, initial_archive_binding=binding)

    monkeypatch.setattr(q, "FORMAL_TRAIN_CRC32", expected_crc)
    raw, receipt = q._read_train_member(
        archive, formal=True, initial_archive_binding=binding
    )
    assert len(raw) == expected_size
    assert receipt["train_crc32"] == expected_crc
    assert receipt["train_member_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    "unsafe_name",
    ["../escape", "/absolute", "safe\\windows", "safe/../escape"],
)
def test_zip_unsafe_paths_are_fatal(
    tmp_path: Path,
    synthetic_manifests: tuple[Path, Path, Path],
    unsafe_name: str,
) -> None:
    archive = _write_zip(
        tmp_path / "unsafe.zip",
        _dataset([_document(1)]),
        extra_infos=[(unsafe_name, b"decoy")],
    )
    with pytest.raises(q.ContractNliQualificationError, match="unsafe"):
        q.build_qualification(archive, *synthetic_manifests)


def test_zip_symlink_encryption_and_duplicate_train_are_fatal(
    tmp_path: Path,
    synthetic_manifests: tuple[Path, Path, Path],
) -> None:
    symlink = zipfile.ZipInfo("contract-nli/link")
    symlink.create_system = 3
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = _write_zip(
        tmp_path / "symlink.zip",
        _dataset([_document(1)]),
        extra_infos=[(symlink, b"target")],
    )
    with pytest.raises(q.ContractNliQualificationError, match="nonregular"):
        q.build_qualification(archive, *synthetic_manifests)

    encrypted = zipfile.ZipInfo("contract-nli/encrypted")
    encrypted.flag_bits = 1
    with pytest.raises(q.ContractNliQualificationError, match="encrypted"):
        q._validate_zip_info(encrypted)

    duplicate = tmp_path / "duplicate.zip"
    train_raw = json.dumps(_dataset([_document(1)])).encode()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(duplicate, "w") as writer:
            writer.writestr(q.TRAIN_MEMBER, train_raw)
            writer.writestr(q.TRAIN_MEMBER, train_raw)
    with pytest.raises(q.ContractNliQualificationError, match="duplicate"):
        q.build_qualification(duplicate, *synthetic_manifests)


def test_other_train_basename_decoy_is_rejected_by_custody_contract(
    tmp_path: Path,
    synthetic_manifests: tuple[Path, Path, Path],
) -> None:
    archive = _write_zip(
        tmp_path / "train-decoy.zip",
        _dataset([_document(1)]),
        extra_infos=[("decoy/train.json", b"must not open")],
    )
    with pytest.raises(q.ContractNliQualificationError, match="basename"):
        q.build_qualification(archive, *synthetic_manifests)


def test_safe_unopened_decoy_names_do_not_add_extra_gates(
    tmp_path: Path,
    synthetic_manifests: tuple[Path, Path, Path],
) -> None:
    archive = _write_zip(
        tmp_path / "safe-decoys.zip",
        _dataset([_document(1)]),
        extra_infos=[
            ("safe//empty-component", b"unopened"),
            ("safe/./dot-component", b"unopened"),
            ("C:/drive-looking-decoy", b"unopened"),
        ],
    )
    receipt = q.build_qualification(archive, *synthetic_manifests)
    assert receipt["qualification_operations"]["train_members_opened"] == 1


def test_duplicate_json_key_and_root_schema_are_fatal(
    tmp_path: Path,
    synthetic_manifests: tuple[Path, Path, Path],
) -> None:
    duplicate_key = b'{"labels":{},"labels":{},"documents":[]}'
    archive = _write_zip(tmp_path / "duplicate-key.zip", duplicate_key)
    with pytest.raises(q.ContractNliQualificationError, match="duplicate JSON"):
        q.build_qualification(archive, *synthetic_manifests)

    bad_root = _write_zip(
        tmp_path / "bad-root.zip",
        {"labels": {}, "documents": []},
    )
    with pytest.raises(q.ContractNliQualificationError, match="17"):
        q.build_qualification(bad_root, *synthetic_manifests)


def test_formal_outer_mismatch_stops_before_zip_open(
    tmp_path: Path,
    synthetic_manifests: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_zip(tmp_path / "synthetic.zip", _dataset([_document(1)]))

    def forbidden_zip_open(*args: object, **kwargs: object) -> None:
        raise AssertionError("formal mismatch must stop before central-directory access")

    monkeypatch.setattr(q.zipfile, "ZipFile", forbidden_zip_open)
    with pytest.raises(q.ContractNliQualificationError, match="outer archive"):
        q.build_qualification(
            archive,
            *synthetic_manifests,
            enforce_formal_bindings=True,
        )


def test_parent_uses_strict_duplicate_key_parser_for_worker_stdout(
    tmp_path: Path,
    synthetic_manifests: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _write_zip(tmp_path / "synthetic.zip", _dataset([_document(1)]))
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=b'{"schema":"first","schema":"second"}',
        stderr=b"",
    )
    monkeypatch.setattr(q.subprocess, "run", lambda *args, **kwargs: completed)
    with pytest.raises(q.ContractNliQualificationError, match="duplicate JSON"):
        q.run_clean_qualification(archive, *synthetic_manifests)


def test_public_formal_manifests_bind_official_reader_branch() -> None:
    repository = Path(__file__).resolve().parents[1]
    custody, custody_receipt = q._read_manifest(
        repository / "manifests/contractnli_graph_evaluator_source_custody_v1.json",
        schema=q.FORMAL_CUSTODY_SCHEMA,
        hash_field="custody_sha256",
    )
    addendum, addendum_receipt = q._read_manifest(
        repository / "manifests/contractnli_source_access_addendum_v1.json",
        schema=q.FORMAL_ADDENDUM_SCHEMA,
        hash_field="addendum_sha256",
    )
    member, member_receipt = q._read_manifest(
        repository / "manifests/contractnli_source_member_binding_v1.json",
        schema=q.FORMAL_MEMBER_SCHEMA,
        hash_field="source_member_binding_sha256",
    )
    q._validate_formal_manifests(
        custody,
        custody_receipt,
        addendum,
        addendum_receipt,
        member,
        member_receipt,
    )


def test_receipt_shape_rejects_added_private_fields(
    tmp_path: Path,
    synthetic_manifests: tuple[Path, Path, Path],
) -> None:
    receipt = _build_synthetic(tmp_path, synthetic_manifests, [_document(123456)])
    tampered = copy.deepcopy(receipt)
    tampered["private_document_id"] = 123456
    body = dict(tampered)
    body.pop("qualification_sha256")
    tampered["qualification_sha256"] = q._semantic_hash(body)
    with pytest.raises(q.ContractNliQualificationError, match="shape"):
        q._validate_child_receipt(tampered)

    tampered = copy.deepcopy(receipt)
    tampered["archive"]["private_rows"] = ["secret"]
    body = dict(tampered)
    body.pop("qualification_sha256")
    tampered["qualification_sha256"] = q._semantic_hash(body)
    with pytest.raises(q.ContractNliQualificationError):
        q._validate_child_receipt(tampered)

    tampered = copy.deepcopy(receipt)
    tampered["archive"]["train_member_sha256"] = "not-a-sha"
    body = dict(tampered)
    body.pop("qualification_sha256")
    tampered["qualification_sha256"] = q._semantic_hash(body)
    with pytest.raises(q.ContractNliQualificationError, match="archive operation"):
        q._validate_child_receipt(tampered)


def test_formal_attempt_is_consumed_before_any_input_read(
    tmp_path: Path,
    synthetic_manifests: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "custody" / "attempt.marker"
    marker.parent.mkdir(mode=0o700)
    monkeypatch.setattr(q, "_formal_attempt_marker_path", lambda: marker)
    calls = 0

    def forbidden_input_read(path: Path, *, label: str) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("probe: source validation happens after marker")

    monkeypatch.setattr(q, "_require_regular_file", forbidden_input_read)
    with pytest.raises(AssertionError, match="probe"):
        q.run_clean_qualification(
            tmp_path / "never-read.zip",
            *synthetic_manifests,
            enforce_formal_bindings=True,
        )
    assert marker.exists()
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    assert calls == 1

    with pytest.raises(q.ContractNliQualificationError, match="already consumed"):
        q.run_clean_qualification(
            tmp_path / "never-read.zip",
            *synthetic_manifests,
            enforce_formal_bindings=True,
        )
    assert calls == 1


def test_atomic_output_is_0644_and_never_overwrites(tmp_path: Path) -> None:
    destination = tmp_path / "receipt.json"
    q._write_json_exclusive(destination, {"public": 1})
    assert stat.S_IMODE(destination.stat().st_mode) == 0o644
    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        q._write_json_exclusive(destination, {"public": 2})
    assert destination.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_formal_parent_requires_output_before_reading_any_input() -> None:
    with pytest.raises(SystemExit):
        q.main(
            [
                "--archive",
                "does-not-exist.zip",
                "--custody-manifest",
                "does-not-exist-custody.json",
                "--source-access-addendum",
                "does-not-exist-addendum.json",
                "--source-member-binding",
                "does-not-exist-member.json",
                "--formal",
            ]
        )


def test_source_has_no_archive_extraction_or_selection_secret_access() -> None:
    source = Path(q.__file__).read_text(encoding="utf-8")
    assert ".extract(" not in source
    assert ".extractall(" not in source
    assert "selection.key" not in source
    assert "secret_relative_path" not in source
