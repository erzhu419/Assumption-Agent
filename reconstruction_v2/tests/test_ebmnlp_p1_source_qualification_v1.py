from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
from typing import Iterable

import pytest

from assumption_agent.benchmarks import ebmnlp_p1_source_qualification_v1 as q


TRAIN_PMIDS = ("101", "102", "103", "104", "105", "106")
TEST_PMIDS = ("201", "202", "203", "204")


def _tokens(pmid: str) -> bytes:
    return f"trial participant treatment outcome {pmid}\n".encode()


def _text(pmid: str) -> bytes:
    return f"Synthetic private abstract {pmid}.\n".encode()


def _labels(pmid: str, role: str, variant: int = 0) -> bytes:
    role_index = q.ROLE_ORDER.index(role)
    values = [
        "1" if (position + role_index + variant) % 3 == 0 else "0"
        for position in range(5)
    ]
    return (" ".join(values) + "\n").encode()


def _member_rows(
    *,
    label_variant: int = 0,
) -> list[tuple[tarfile.TarInfo, bytes | None]]:
    rows: list[tuple[tarfile.TarInfo, bytes | None]] = []

    def regular(name: str, raw: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(raw)
        info.mode = 0o600
        rows.append((info, raw))

    root = q.ARCHIVE_ROOT
    regular(f"{root}/README.md", b"synthetic ignored public metadata\n")
    for split, pmids in (("train", TRAIN_PMIDS), ("test/gold", TEST_PMIDS)):
        for pmid in pmids:
            regular(f"{root}/documents/{pmid}.tokens", _tokens(pmid))
            regular(f"{root}/documents/{pmid}.text", _text(pmid))
            regular(
                f"{root}/documents/{pmid}.pos",
                b"NN NN NN NN CD\n",
            )
            for role in q.ROLE_ORDER:
                regular(
                    f"{root}/annotations/aggregated/starting_spans/"
                    f"{role}/{split}/{pmid}.ann",
                    _labels(pmid, role, label_variant),
                )
    return rows


def _tar_bytes(rows: Iterable[tuple[tarfile.TarInfo, bytes | None]]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        for info, raw in rows:
            bundle.addfile(info, None if raw is None else io.BytesIO(raw))
    return output.getvalue()


def _write_source(
    tmp_path: Path,
    *,
    rows: list[tuple[tarfile.TarInfo, bytes | None]] | None = None,
    label_variant: int = 0,
) -> tuple[Path, q.QualificationContract]:
    raw = _tar_bytes(rows if rows is not None else _member_rows(label_variant=label_variant))
    path = tmp_path / "ebm_nlp_2_00.tar.gz"
    path.write_bytes(raw)
    path.chmod(0o600)
    contract = q.QualificationContract(
        archive_sha256=hashlib.sha256(raw).hexdigest(),
        archive_size_bytes=len(raw),
        total_public_abstract_count=10,
        train_abstract_count=6,
        test_abstract_count=4,
        blocks=q.BlockCounts(
            G_form=2,
            A_form=1,
            F_search=1,
            A_hold=1,
            M_search=1,
        ),
        maximum_archive_member_count=1_000,
        maximum_total_declared_member_bytes=10_000_000,
        maximum_document_member_bytes=1_000_000,
        maximum_label_member_bytes=1_000_000,
        maximum_ignored_regular_member_bytes=1_000_000,
        maximum_tokens_per_document=1_000,
    )
    return path, contract


def _acquire(
    tmp_path: Path,
    *,
    rows: list[tuple[tarfile.TarInfo, bytes | None]] | None = None,
    label_variant: int = 0,
    root_name: str = "private",
) -> q.AcquisitionResult:
    source, contract = _write_source(
        tmp_path, rows=rows, label_variant=label_variant
    )
    return q.acquire_once(
        archive_path=source,
        private_root=tmp_path / root_name,
        contract=contract,
        secret_factory=lambda length: b"s" * length,
    )


def _authorization(
    result: q.AcquisitionResult,
    stage: str,
    *,
    prerequisites_sealed: bool = True,
    promotion_authorized: bool = False,
) -> q.LabelOpenAuthorization:
    return q.LabelOpenAuthorization(
        stage=stage,
        source_sha256=result.inventory.archive_sha256,
        assignment_sha256=result.assignment.assignment_sha256,
        prerequisites_sealed=prerequisites_sealed,
        promotion_authorized=promotion_authorized,
    )


def _replace_member(
    rows: list[tuple[tarfile.TarInfo, bytes | None]],
    name: str,
    raw: bytes,
) -> list[tuple[tarfile.TarInfo, bytes | None]]:
    output: list[tuple[tarfile.TarInfo, bytes | None]] = []
    found = False
    for info, payload in rows:
        if info.name == name:
            replacement = tarfile.TarInfo(name)
            replacement.size = len(raw)
            replacement.mode = 0o600
            output.append((replacement, raw))
            found = True
        else:
            output.append((info, payload))
    assert found
    return output


def test_header_qualification_exact_topology_without_payload_values(
    tmp_path: Path,
) -> None:
    source, contract = _write_source(tmp_path)
    inventory = q.qualify_archive_headers(source, contract)
    assert len(inventory.documents) == 10
    assert len(inventory.labels["TRAIN"]["participants"]) == 6
    assert len(inventory.labels["TEST"]["outcomes"]) == 4
    assert inventory.ignored_regular_member_count == 11
    serialized = json.dumps(
        {
            "regular": inventory.regular_member_count,
            "directories": inventory.directory_member_count,
            "ignored": inventory.ignored_regular_member_count,
        }
    )
    assert "Synthetic private" not in serialized


@pytest.mark.parametrize(
    ("name", "member_type", "linkname"),
    [
        ("../escape", tarfile.REGTYPE, ""),
        ("/absolute", tarfile.REGTYPE, ""),
        (f"{q.ARCHIVE_ROOT}/documents/../escape", tarfile.REGTYPE, ""),
        (f"{q.ARCHIVE_ROOT}\\documents\\101.tokens", tarfile.REGTYPE, ""),
        (f"{q.ARCHIVE_ROOT}/link", tarfile.SYMTYPE, "../../escape"),
        (f"{q.ARCHIVE_ROOT}/hard", tarfile.LNKTYPE, "target"),
        (f"{q.ARCHIVE_ROOT}/fifo", tarfile.FIFOTYPE, ""),
    ],
)
def test_unsafe_link_traversal_and_nonfile_members_are_rejected(
    tmp_path: Path, name: str, member_type: bytes, linkname: str
) -> None:
    rows = _member_rows()
    info = tarfile.TarInfo(name)
    info.type = member_type
    info.linkname = linkname
    raw = b"x" if member_type == tarfile.REGTYPE else None
    info.size = 1 if raw is not None else 0
    rows.append((info, raw))
    source, contract = _write_source(tmp_path, rows=rows)
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="unsafe|root|link|non-file"):
        q.qualify_archive_headers(source, contract)
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    "name",
    [
        f"{q.ARCHIVE_ROOT}/documents/not-a-pmid.tokens",
        f"{q.ARCHIVE_ROOT}/documents/101.json",
        (
            f"{q.ARCHIVE_ROOT}/annotations/aggregated/starting_spans/"
            "unknown/train/101.ann"
        ),
        (
            f"{q.ARCHIVE_ROOT}/annotations/aggregated/starting_spans/"
            "participants/test/101.ann"
        ),
    ],
)
def test_near_miss_controlled_namespace_paths_are_rejected(
    tmp_path: Path, name: str
) -> None:
    rows = _member_rows()
    info = tarfile.TarInfo(name)
    info.size = 2
    rows.append((info, b"x\n"))
    source, contract = _write_source(tmp_path, rows=rows)
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="exact frozen path"):
        q.qualify_archive_headers(source, contract)


def test_duplicate_member_path_is_rejected(tmp_path: Path) -> None:
    rows = _member_rows()
    duplicate_name = f"{q.ARCHIVE_ROOT}/documents/101.tokens"
    info = tarfile.TarInfo(duplicate_name)
    info.size = len(_tokens("101"))
    rows.append((info, _tokens("101")))
    source, contract = _write_source(tmp_path, rows=rows)
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="duplicate"):
        q.qualify_archive_headers(source, contract)


@pytest.mark.parametrize(
    "missing_name",
    [
        f"{q.ARCHIVE_ROOT}/documents/101.text",
        (
            f"{q.ARCHIVE_ROOT}/annotations/aggregated/starting_spans/"
            "outcomes/train/101.ann"
        ),
    ],
)
def test_document_pair_and_three_role_completeness_are_required(
    tmp_path: Path, missing_name: str
) -> None:
    rows = [
        (info, raw) for info, raw in _member_rows() if info.name != missing_name
    ]
    source, contract = _write_source(tmp_path, rows=rows)
    with pytest.raises(
        q.EbmNlpP1SourceQualificationError,
        match="pair|role annotation|identity sets",
    ):
        q.qualify_archive_headers(source, contract)


def test_exact_public_count_contract_is_required(tmp_path: Path) -> None:
    source, contract = _write_source(tmp_path)
    wrong = replace(contract, total_public_abstract_count=11)
    with pytest.raises(
        q.EbmNlpP1SourceQualificationError, match="internally inconsistent|count"
    ):
        q.qualify_archive_headers(source, wrong)


def test_cross_split_pmid_is_rejected_before_selection(tmp_path: Path) -> None:
    rows = _member_rows()
    for role in q.ROLE_ORDER:
        name = (
            f"{q.ARCHIVE_ROOT}/annotations/aggregated/starting_spans/"
            f"{role}/test/gold/101.ann"
        )
        info = tarfile.TarInfo(name)
        raw = _labels("101", role)
        info.size = len(raw)
        rows.append((info, raw))
    source, contract = _write_source(tmp_path, rows=rows)
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="both official splits"):
        q.qualify_archive_headers(source, contract)


def test_distinct_pmids_may_not_share_tokens_text_digest_pair(
    tmp_path: Path,
) -> None:
    rows = _member_rows()
    rows = _replace_member(
        rows, f"{q.ARCHIVE_ROOT}/documents/102.tokens", _tokens("101")
    )
    rows = _replace_member(
        rows, f"{q.ARCHIVE_ROOT}/documents/102.text", _text("101")
    )
    source, contract = _write_source(tmp_path, rows=rows)
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="digest pair"):
        q.acquire_once(
            archive_path=source,
            private_root=tmp_path / "burned",
            contract=contract,
            secret_factory=lambda length: b"s" * length,
        )
    assert (tmp_path / "burned" / "acquisition.attempt_consumed.json").exists()


def test_acquisition_is_one_shot_private_and_opens_no_labels(
    tmp_path: Path,
) -> None:
    result = _acquire(tmp_path)
    assert set(result.assignment.blocks) == set(q.BLOCK_ORDER)
    selected_count = sum(len(value) for value in result.assignment.blocks.values())
    assert selected_count == 6
    receipt = json.loads(result.receipt_path.read_text("ascii"))
    assert receipt["annotation_payload_open_count"] == 0
    assert receipt["selection_inputs"] == "study_id_official_split_PMID_only"
    assert stat.S_IMODE(result.private_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(result.receipt_path.stat().st_mode) == 0o600
    for path in (result.private_root / "documents").iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="exactly once"):
        q.acquire_once(
            archive_path=result.archive_path,
            private_root=result.private_root,
            contract=result.contract,
            secret_factory=lambda length: b"s" * length,
        )


def test_hmac_assignment_is_independent_of_label_payloads(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _acquire(first_dir, label_variant=0)
    second = _acquire(second_dir, label_variant=1)
    assert dict(first.assignment.blocks) == dict(second.assignment.blocks)
    assert first.assignment.assignment_sha256 != second.assignment.assignment_sha256
    # The assignment hash binds the source hash, while the selected identities
    # themselves are deliberately unaffected by annotation payload values.


@pytest.mark.parametrize(
    "raw",
    [
        b"0 1 -1 0 1\n",
        b"0 1 1.0 0 1\n",
        "0 1 ² 0 1\n".encode(),
        b"0 1 01 0 1\n",
        b"0 1 x 0 1\n",
    ],
)
def test_label_parser_rejects_non_documented_integer_tokens(raw: bytes) -> None:
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="integer"):
        q.parse_label_payload(raw, expected_token_count=5)


def test_label_parser_requires_exact_document_token_length() -> None:
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="length"):
        q.parse_label_payload(b"0 1 0 1\n", expected_token_count=5)
    assert q.parse_label_payload(b"0 1 0 2 0\n", expected_token_count=5) == (
        0,
        1,
        0,
        2,
        0,
    )


def test_f_search_labels_are_unconditionally_inaccessible(
    tmp_path: Path,
) -> None:
    result = _acquire(tmp_path)
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="permanently"):
        q.open_labels_for_stage(
            result,
            stage="F_search",
            authorization=_authorization(result, "F_search"),
        )
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="permanently"):
        q.open_f_search_labels(result)
    assert not (
        result.private_root
        / "label_open_markers/F_search.attempt_consumed.json"
    ).exists()


def test_a_form_requires_sealed_prerequisite_and_opens_once(
    tmp_path: Path,
) -> None:
    result = _acquire(tmp_path)
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="capability"):
        q.open_labels_for_stage(
            result,
            stage="A_form",
            authorization=_authorization(
                result, "A_form", prerequisites_sealed=False
            ),
        )
    assert not (
        result.private_root / "label_open_markers/A_form.attempt_consumed.json"
    ).exists()
    labels = q.open_labels_for_stage(
        result,
        stage="A_form",
        authorization=_authorization(result, "A_form"),
    )
    assert len(labels) == 1
    assert set(next(iter(labels.values()))) == set(q.ROLE_ORDER)
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="consumed"):
        q.open_labels_for_stage(
            result,
            stage="A_form",
            authorization=_authorization(result, "A_form"),
        )


def test_m_search_requires_promotion_before_any_attempt_is_consumed(
    tmp_path: Path,
) -> None:
    result = _acquire(tmp_path)
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="promotion"):
        q.open_labels_for_stage(
            result,
            stage="M_search",
            authorization=_authorization(
                result, "M_search", promotion_authorized=False
            ),
        )
    marker = (
        result.private_root
        / "label_open_markers/M_search.attempt_consumed.json"
    )
    assert not marker.exists()
    labels = q.open_labels_for_stage(
        result,
        stage="M_search",
        authorization=_authorization(
            result, "M_search", promotion_authorized=True
        ),
    )
    assert len(labels) == 1
    assert marker.exists()


def test_label_payload_is_not_parsed_until_its_stage_is_authorized(
    tmp_path: Path,
) -> None:
    rows = _member_rows()
    # Determine F from the fixed test secret without touching annotation data.
    source, contract = _write_source(tmp_path, rows=rows)
    assignment = q.assign_blocks(
        {"TRAIN": TRAIN_PMIDS, "TEST": TEST_PMIDS},
        secret=b"s" * 32,
        contract=contract,
    )
    f_pmid = assignment.pmids("F_search")[0]
    invalid_name = (
        f"{q.ARCHIVE_ROOT}/annotations/aggregated/starting_spans/"
        f"participants/train/{f_pmid}.ann"
    )
    invalid_rows = _replace_member(rows, invalid_name, b"not labels\n")
    source, contract = _write_source(tmp_path, rows=invalid_rows)
    result = q.acquire_once(
        archive_path=source,
        private_root=tmp_path / "private",
        contract=contract,
        secret_factory=lambda length: b"s" * length,
    )
    assert result.assignment.pmids("F_search") == (f_pmid,)
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="permanently"):
        q.open_labels_for_stage(
            result,
            stage="F_search",
            authorization=_authorization(result, "F_search"),
        )


def test_authorization_is_bound_to_source_assignment_and_stage(
    tmp_path: Path,
) -> None:
    result = _acquire(tmp_path)
    wrong = q.LabelOpenAuthorization(
        stage="A_hold",
        source_sha256="0" * 64,
        assignment_sha256=result.assignment.assignment_sha256,
        prerequisites_sealed=True,
    )
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="capability"):
        q.open_labels_for_stage(result, stage="A_hold", authorization=wrong)
    wrong_stage = _authorization(result, "A_form")
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="capability"):
        q.open_labels_for_stage(
            result, stage="A_hold", authorization=wrong_stage
        )


def test_archive_identity_and_private_mode_are_enforced(tmp_path: Path) -> None:
    source, contract = _write_source(tmp_path)
    source.chmod(0o644)
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="private mode"):
        q.qualify_archive_headers(source, contract)
    source.chmod(0o600)
    with pytest.raises(q.EbmNlpP1SourceQualificationError, match="SHA256"):
        q.qualify_archive_headers(
            source, replace(contract, archive_sha256="0" * 64)
        )


def test_no_retry_rescue_or_tar_extract_surface_is_exported() -> None:
    exported = set(q.__all__)
    assert all(
        forbidden not in name.lower()
        for name in exported
        for forbidden in ("retry", "rescue", "replacement", "resample")
    )
    assert not hasattr(q, "extractall")
