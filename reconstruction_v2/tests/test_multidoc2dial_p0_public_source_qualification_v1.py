from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator
from unittest import mock
import zipfile

import pytest

from assumption_agent.benchmarks import (
    multidoc2dial_p0_public_source_qualification_v1 as p0,
)


@pytest.fixture
def posix_tmp() -> Iterator[Path]:
    root = Path(tempfile.mkdtemp(prefix="mdd-p0-test-", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _events(value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        yield "start_map", None
        for key, member in value.items():
            yield "map_key", key
            yield from _events(member)
        yield "end_map", None
    elif isinstance(value, list):
        yield "start_array", None
        for member in value:
            yield from _events(member)
        yield "end_array", None
    elif isinstance(value, str):
        yield "string", value
    elif type(value) is bool:
        yield "boolean", value
    elif type(value) is int:
        yield "number", value
    elif value is None:
        yield "null", None
    else:
        raise AssertionError(type(value))


def _synthetic_basic_parse(source) -> Iterator[tuple[str, Any]]:
    yield from _events(json.load(source))


def _documents() -> dict[str, Any]:
    domains: dict[str, Any] = {}
    for domain in p0.DOMAINS:
        text = f"Exact public passage for {domain}."
        domains[domain] = {
            f"{domain}-doc": {
                "doc_id": f"{domain}-doc",
                "title": f"{domain} public title",
                "doc_text": text,
                "spans": {
                    "0": {
                        "id_sp": "0",
                        "start_sp": 0,
                        "end_sp": len(text),
                        "text_sp": text,
                        "id_sec": f"{domain}-section-0",
                        "start_sec": 0,
                        "end_sec": len(text),
                        "text_sec": text,
                        "title": "",
                        "parent_titles": [
                            {
                                "id_sp": f"{domain}-parent-0",
                                "text": f"{domain} parent",
                                "level": "1",
                            }
                        ],
                        "tag": "p",
                    }
                },
                "public_extra": {"kind": "document"},
            }
        }
    return {"doc_data": domains}


def _family_da(family: str, ordinal: int) -> str:
    if family == p0.CONDITION_QUERY:
        return "query_condition"
    if family == p0.SOLUTION_QUERY:
        return "query_solution"
    if family == p0.POLAR_CLARIFICATION:
        return (
            "response_positive"
            if ordinal % 2 == 0
            else "response_negative"
        )
    raise AssertionError(family)


def _dialogues(
    split: str,
    *,
    counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    minimums = p0.MINIMUM_ELIGIBLE_FAMILY_COUNTS[split]
    requested = (
        counts
        if counts is not None
        else {
            family: max(
                minimums[family],
                p0.MINIMUM_ELIGIBLE_DOMAIN_FAMILY_COUNTS[split][
                    family
                ]
                * len(p0.DOMAINS),
            )
            for family in p0.FAMILIES
        }
    )
    by_domain: dict[str, list[dict[str, Any]]] = {
        domain: [] for domain in p0.DOMAINS
    }
    response_acts = (
        "respond_solution",
        "respond_solution_positive",
        "respond_solution_negative",
    )
    for family in p0.FAMILIES:
        for ordinal in range(requested[family]):
            domain = p0.DOMAINS[ordinal % len(p0.DOMAINS)]
            dialogue_id = f"{split}-{family}-{ordinal:04d}"
            by_domain[domain].append(
                {
                    "dial_id": dialogue_id,
                    "turns": [
                        {
                            "turn_id": 1,
                            "role": "user",
                            "da": _family_da(family, ordinal),
                            "utterance": (
                                f"Unique {split} {family} history "
                                f"{ordinal} in {domain}?"
                            ),
                            "references": [],
                        },
                        {
                            "turn_id": 2,
                            "role": "agent",
                            "da": response_acts[
                                ordinal % len(response_acts)
                            ],
                            "utterance": (
                                f"Grounded synthetic response {ordinal}."
                            ),
                            "references": [
                                {
                                    "doc_id": f"{domain}-doc",
                                    "id_sp": "0",
                                    "label": "solution",
                                }
                            ],
                        },
                    ],
                    "public_extra": "dialogue",
                }
            )
    if split == "TRAIN":
        domain = p0.DOMAINS[0]
        by_domain[domain].append(
            {
                "dial_id": "TRAIN-public-no-solution",
                "turns": [
                    {
                        "turn_id": 1,
                        "role": "user",
                        "da": "query_condition",
                        "utterance": "Unique ineligible no solution input.",
                        "references": [],
                    },
                    {
                        "turn_id": 2,
                        "role": "agent",
                        "da": "respond_no_solution",
                        "utterance": "No grounded solution is available.",
                        "references": [],
                    },
                ],
            }
        )
    return {"dial_data": by_domain}


def _member_values() -> dict[str, Any]:
    return {
        p0.DOCUMENT_MEMBER: _documents(),
        p0.TRAIN_MEMBER: _dialogues("TRAIN"),
        p0.VALIDATION_MEMBER: _dialogues("VALIDATION"),
        # Deliberately not valid JSON.  Qualification must never open it.
        p0.TEST_MEMBER: b"TEST-PAYLOAD-MUST-NOT-BE-OPENED",
    }


def _archive(
    root: Path,
    *,
    values: dict[str, Any] | None = None,
    extra_member: tuple[str, bytes] | None = None,
) -> tuple[Path, p0.ArchiveContract]:
    path = root / "synthetic-multidoc2dial.zip"
    selected = values if values is not None else _member_values()
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(f"{p0.ARCHIVE_ROOT}/", b"")
        for name in sorted(selected):
            value = selected[name]
            raw = (
                value
                if isinstance(value, bytes)
                else json.dumps(value, ensure_ascii=False).encode("utf-8")
            )
            archive.writestr(name, raw)
        if extra_member is not None:
            archive.writestr(*extra_member)
    return path, p0.ArchiveContract(
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _qualify(
    root: Path,
    path: Path,
    contract: p0.ArchiveContract,
) -> dict[str, Any]:
    with mock.patch.object(
        p0,
        "_ijson_basic_parse",
        _synthetic_basic_parse,
    ):
        return p0.qualify_archive(
            archive_path=path,
            eligibility_manifest_path=(
                root / "eligibility.private.json"
            ),
            archive_contract=contract,
        )


def test_official_archive_identity_and_public_registries_are_frozen() -> None:
    assert p0.ARCHIVE_SIZE_BYTES == 6_868_509
    assert p0.ARCHIVE_GIT_BLOB_SHA1 == (
        "9d8dd4a24cb60ce90bb5f14730fdd1d3ca191672"
    )
    assert p0.ARCHIVE_SHA256 == (
        "f0c034c249663d7b3cb08b19cf2cc2c3d101372485be982621d4711931a1ce00"
    )
    assert p0.REGULAR_MEMBER_WHITELIST == {
        "multidoc2dial/multidoc2dial_doc.json",
        "multidoc2dial/multidoc2dial_dial_train.json",
        "multidoc2dial/multidoc2dial_dial_validation.json",
        "multidoc2dial/multidoc2dial_dial_test.json",
    }
    assert p0.DOMAIN_SET == {"dmv", "ssa", "studentaid", "va"}
    assert p0.INPUT_DA_TO_FAMILY == {
        "query_condition": p0.CONDITION_QUERY,
        "query_solution": p0.SOLUTION_QUERY,
        "response_positive": p0.POLAR_CLARIFICATION,
        "response_negative": p0.POLAR_CLARIFICATION,
    }


def test_qualification_is_safe_aggregate_only_and_never_opens_test(
    posix_tmp: Path,
) -> None:
    path, contract = _archive(posix_tmp)
    opened: list[str] = []
    original_open = zipfile.ZipFile.open

    def guarded_open(self, name, *args, **kwargs):
        resolved = name.filename if isinstance(name, zipfile.ZipInfo) else name
        if resolved == p0.TEST_MEMBER:
            raise AssertionError("TEST payload was opened")
        opened.append(resolved)
        return original_open(self, name, *args, **kwargs)

    with mock.patch.object(zipfile.ZipFile, "open", guarded_open):
        receipt = _qualify(posix_tmp, path, contract)
    assert opened == [
        p0.DOCUMENT_MEMBER,
        p0.TRAIN_MEMBER,
        p0.VALIDATION_MEMBER,
    ]
    assert receipt["status"] == (
        "qualified_public_non_scoring_schema_grounding_and_"
        "source_native_family_capacity"
    )
    assert receipt["archive_topology"]["test_payload_open_count"] == 0
    assert receipt["access_boundary"] == {
        "source_payload_member_open_count": 3,
        "document_payload_member_open_count": 1,
        "train_payload_member_open_count": 1,
        "validation_payload_member_open_count": 1,
        "test_payload_member_open_count": 0,
        "source_full_extraction_count": 0,
        "secret_or_cohort_assignment_count": 0,
        "action_model_evaluator_or_score_count": 0,
        "individual_identifier_text_or_qrel_value_output_count": 0,
        "online_or_API_evaluation_count": 0,
    }
    assert receipt["source_native_family_registry"][
        "observed_equals_frozen_registry"
    ]
    for split in ("TRAIN", "VALIDATION"):
        assert receipt["dialogue_aggregate"][split][
            "eligible_family_count"
        ] == {
            family: max(
                p0.MINIMUM_ELIGIBLE_FAMILY_COUNTS[split][family],
                p0.MINIMUM_ELIGIBLE_DOMAIN_FAMILY_COUNTS[split][
                    family
                ]
                * len(p0.DOMAINS),
            )
            for family in p0.FAMILIES
        }
    rendered = p0.canonical_bytes(receipt).decode("ascii")
    for forbidden in (
        "TRAIN-CONDITION_QUERY-0000",
        "Unique TRAIN",
        "dmv-doc",
        "Exact public passage",
        '"id_sp"',
        '"doc_id"',
    ):
        assert forbidden not in rendered

    private_path = posix_tmp / "eligibility.private.json"
    manifest = json.loads(private_path.read_text("ascii"))
    assert private_path.stat().st_mode & 0o777 == 0o600
    assert manifest["self_sha256"] == p0.stable_hash(
        {
            key: value
            for key, value in manifest.items()
            if key != "self_sha256"
        }
    )
    assert set(manifest["eligible_rows_by_split"]) == {
        "TRAIN",
        "VALIDATION",
    }
    for rows in manifest["eligible_rows_by_split"].values():
        for row in rows:
            assert set(row) == {
                "opaque_item_id",
                "domain",
                "family",
                "normalized_query_sha256",
                "dialogue_sha256",
            }
            assert len(row["opaque_item_id"]) == 64
            assert len(row["normalized_query_sha256"]) == 64
            assert len(row["dialogue_sha256"]) == 64
    assert receipt["private_eligibility_manifest_binding"][
        "self_sha256"
    ] == manifest["self_sha256"]


def test_wrong_archive_identity_fails_before_zip_open(
    posix_tmp: Path,
) -> None:
    path, contract = _archive(posix_tmp)
    wrong = p0.ArchiveContract(
        filename=contract.filename,
        size_bytes=contract.size_bytes,
        sha256="f" * 64,
    )
    with mock.patch.object(
        p0.zipfile,
        "ZipFile",
        side_effect=AssertionError("central directory opened"),
    ):
        with pytest.raises(
            p0.MultiDoc2DialP0QualificationError,
            match="byte identity",
        ):
            p0.qualify_archive(
                archive_path=path,
                eligibility_manifest_path=(
                    posix_tmp / "eligibility.private.json"
                ),
                archive_contract=wrong,
            )


@pytest.mark.parametrize(
    "extra_name",
    (
        "multidoc2dial/unexpected.json",
        "../escape.json",
    ),
)
def test_zip_topology_rejects_extra_or_traversing_regular_member(
    posix_tmp: Path,
    extra_name: str,
) -> None:
    path, contract = _archive(
        posix_tmp,
        extra_member=(extra_name, b"{}"),
    )
    with mock.patch.object(
        p0,
        "_ijson_basic_parse",
        _synthetic_basic_parse,
    ):
        with pytest.raises(p0.MultiDoc2DialP0QualificationError):
            p0.qualify_archive(
                archive_path=path,
                eligibility_manifest_path=(
                    posix_tmp / "eligibility.private.json"
                ),
                archive_contract=contract,
            )


def test_exact_grounding_and_passage_bounds_fail_closed(
    posix_tmp: Path,
) -> None:
    values = _member_values()
    documents = values[p0.DOCUMENT_MEMBER]
    assert isinstance(documents, dict)
    documents["doc_data"]["dmv"]["dmv-doc"]["spans"]["0"][
        "end_sp"
    ] -= 1
    path, contract = _archive(posix_tmp, values=values)
    with pytest.raises(
        p0.MultiDoc2DialP0QualificationError,
        match="offsets",
    ):
        _qualify(posix_tmp, path, contract)


def test_malformed_parent_titles_fail_closed(
    posix_tmp: Path,
) -> None:
    values = _member_values()
    documents = values[p0.DOCUMENT_MEMBER]
    assert isinstance(documents, dict)
    parent = documents["doc_data"]["dmv"]["dmv-doc"]["spans"]["0"][
        "parent_titles"
    ][0]
    parent["level"] = 1
    path, contract = _archive(posix_tmp, values=values)
    with pytest.raises(
        p0.MultiDoc2DialP0QualificationError,
        match="parent title level",
    ):
        _qualify(posix_tmp, path, contract)


def test_section_offsets_and_exact_text_fail_closed(
    posix_tmp: Path,
) -> None:
    values = _member_values()
    documents = values[p0.DOCUMENT_MEMBER]
    assert isinstance(documents, dict)
    span = documents["doc_data"]["dmv"]["dmv-doc"]["spans"]["0"]
    span["end_sec"] -= 1
    path, contract = _archive(posix_tmp, values=values)
    with pytest.raises(
        p0.MultiDoc2DialP0QualificationError,
        match="section offsets",
    ):
        _qualify(posix_tmp, path, contract)


def test_reference_to_unknown_passage_fails_closed(
    posix_tmp: Path,
) -> None:
    values = _member_values()
    train = values[p0.TRAIN_MEMBER]
    assert isinstance(train, dict)
    first = train["dial_data"]["dmv"][0]
    first["turns"][1]["references"][0]["id_sp"] = "unknown"
    path, contract = _archive(posix_tmp, values=values)
    with pytest.raises(
        p0.MultiDoc2DialP0QualificationError,
        match="does not exactly map",
    ):
        _qualify(posix_tmp, path, contract)


def test_unregistered_user_act_cannot_enter_eligibility(
    posix_tmp: Path,
) -> None:
    values = _member_values()
    train = values[p0.TRAIN_MEMBER]
    assert isinstance(train, dict)
    first = train["dial_data"]["dmv"][0]
    first["turns"][0]["da"] = "respond_solution"
    path, contract = _archive(posix_tmp, values=values)
    with pytest.raises(
        p0.MultiDoc2DialP0QualificationError,
        match="entered eligibility",
    ):
        _qualify(posix_tmp, path, contract)


def test_cross_split_dialogue_and_group_overlap_fails_closed(
    posix_tmp: Path,
) -> None:
    values = _member_values()
    train = values[p0.TRAIN_MEMBER]
    validation = values[p0.VALIDATION_MEMBER]
    assert isinstance(train, dict)
    assert isinstance(validation, dict)
    validation["dial_data"]["dmv"][0]["dial_id"] = (
        train["dial_data"]["dmv"][0]["dial_id"]
    )
    path, contract = _archive(posix_tmp, values=values)
    with pytest.raises(
        p0.MultiDoc2DialP0QualificationError,
        match="dialogue IDs overlap",
    ):
        _qualify(posix_tmp, path, contract)


def test_within_split_normalized_history_duplicates_are_grouped_and_kept(
    posix_tmp: Path,
) -> None:
    values = _member_values()
    train = values[p0.TRAIN_MEMBER]
    assert isinstance(train, dict)
    first, second = train["dial_data"]["dmv"][:2]
    second["turns"][0]["utterance"] = first["turns"][0]["utterance"].upper()
    path, contract = _archive(posix_tmp, values=values)
    receipt = _qualify(posix_tmp, path, contract)
    grouping = receipt["dialogue_aggregate"]["TRAIN"][
        "normalized_query_grouping"
    ]
    assert grouping["duplicate_group_count"] == 1
    assert grouping["duplicate_row_count"] == 2
    assert grouping["excess_duplicate_row_count"] == 1
    assert grouping["maximum_selected_items_per_group"] == 1
    manifest = json.loads(
        (posix_tmp / "eligibility.private.json").read_text("ascii")
    )
    assert manifest["query_group_contract"] == p0.QUERY_GROUP_CONTRACT
    hashes = [
        row["normalized_query_sha256"]
        for row in manifest["eligible_rows_by_split"]["TRAIN"]
    ]
    counts = {value: hashes.count(value) for value in set(hashes)}
    assert sorted(counts.values())[-1] == 2


def test_cross_split_query_overlap_excludes_every_row_in_group(
    posix_tmp: Path,
) -> None:
    values = _member_values()
    values[p0.TRAIN_MEMBER] = _dialogues(
        "TRAIN",
        counts={family: 56 for family in p0.FAMILIES},
    )
    values[p0.VALIDATION_MEMBER] = _dialogues(
        "VALIDATION",
        counts={family: 36 for family in p0.FAMILIES},
    )
    train = values[p0.TRAIN_MEMBER]
    validation = values[p0.VALIDATION_MEMBER]
    assert isinstance(train, dict)
    assert isinstance(validation, dict)
    first, second = train["dial_data"]["dmv"][:2]
    shared_text = first["turns"][0]["utterance"]
    second["turns"][0]["utterance"] = shared_text.upper()
    validation["dial_data"]["dmv"][0]["turns"][0][
        "utterance"
    ] = shared_text
    path, contract = _archive(posix_tmp, values=values)
    receipt = _qualify(posix_tmp, path, contract)
    exclusion = receipt["cross_split_normalized_query_exclusion"]
    assert exclusion["overlap_group_count"] == 1
    assert exclusion["TRAIN"]["excluded_row_count"] == 2
    assert exclusion["VALIDATION"]["excluded_row_count"] == 1
    assert exclusion["post_exclusion_overlap_group_count"] == 0
    assert receipt["dialogue_aggregate"]["TRAIN"]["domain"]["dmv"][
        "eligible_family_count"
    ][p0.CONDITION_QUERY] == 12
    assert receipt["dialogue_aggregate"]["VALIDATION"]["domain"]["dmv"][
        "eligible_family_count"
    ][p0.CONDITION_QUERY] == 8
    manifest = json.loads(
        (posix_tmp / "eligibility.private.json").read_text("ascii")
    )
    train_hashes = {
        row["normalized_query_sha256"]
        for row in manifest["eligible_rows_by_split"]["TRAIN"]
    }
    validation_hashes = {
        row["normalized_query_sha256"]
        for row in manifest["eligible_rows_by_split"]["VALIDATION"]
    }
    assert not train_hashes & validation_hashes
    assert len(manifest["eligible_rows_by_split"]["TRAIN"]) == (
        56 * len(p0.FAMILIES) - 2
    )
    assert len(manifest["eligible_rows_by_split"]["VALIDATION"]) == (
        36 * len(p0.FAMILIES) - 1
    )
    overlap_hash = p0.core.normalized_query_sha256(
        (p0.core.DialogueTurn(role="user", text=shared_text),)
    )
    assert overlap_hash not in p0.canonical_bytes(receipt).decode("ascii")


def test_cross_split_query_exclusion_rechecks_domain_capacity(
    posix_tmp: Path,
) -> None:
    values = _member_values()
    train = values[p0.TRAIN_MEMBER]
    validation = values[p0.VALIDATION_MEMBER]
    assert isinstance(train, dict)
    assert isinstance(validation, dict)
    for split_name, source in (
        ("TRAIN", train),
        ("VALIDATION", validation),
    ):
        extra = json.loads(
            json.dumps(source["dial_data"]["ssa"][0])
        )
        extra["dial_id"] = f"{split_name}-extra-ssa-condition"
        extra["turns"][0]["utterance"] = (
            f"Unique {split_name} surplus outside dmv."
        )
        source["dial_data"]["ssa"].append(extra)
    validation["dial_data"]["dmv"][0]["turns"][0]["utterance"] = (
        train["dial_data"]["dmv"][0]["turns"][0]["utterance"]
    )
    path, contract = _archive(posix_tmp, values=values)
    with pytest.raises(
        p0.MultiDoc2DialP0QualificationError,
        match="domain-by-family capacity",
    ):
        _qualify(posix_tmp, path, contract)
    assert not (posix_tmp / "eligibility.private.json").exists()


def test_family_capacity_is_checked_before_manifest_write(
    posix_tmp: Path,
) -> None:
    values = _member_values()
    values[p0.VALIDATION_MEMBER] = _dialogues(
        "VALIDATION",
        counts={
            p0.CONDITION_QUERY: 23,
            p0.SOLUTION_QUERY: 24,
            p0.POLAR_CLARIFICATION: 24,
        },
    )
    path, contract = _archive(posix_tmp, values=values)
    with pytest.raises(
        p0.MultiDoc2DialP0QualificationError,
        match="capacity is insufficient",
    ):
        _qualify(posix_tmp, path, contract)
    assert not (posix_tmp / "eligibility.private.json").exists()


def test_domain_by_family_capacity_is_checked(
    posix_tmp: Path,
) -> None:
    values = _member_values()
    validation = values[p0.VALIDATION_MEMBER]
    assert isinstance(validation, dict)
    validation["dial_data"]["dmv"].pop()
    path, contract = _archive(posix_tmp, values=values)
    with pytest.raises(
        p0.MultiDoc2DialP0QualificationError,
        match="domain-by-family capacity",
    ):
        _qualify(posix_tmp, path, contract)


def test_consecutive_same_role_turns_are_valid_history(
    posix_tmp: Path,
) -> None:
    values = _member_values()
    train = values[p0.TRAIN_MEMBER]
    assert isinstance(train, dict)
    dialogue = train["dial_data"]["dmv"][0]
    dialogue["turns"].insert(
        0,
        {
            "turn_id": 0,
            "role": "user",
            "da": "query_condition",
            "utterance": "Earlier consecutive user clarification.",
            "references": [],
        },
    )
    path, contract = _archive(posix_tmp, values=values)
    receipt = _qualify(posix_tmp, path, contract)
    assert receipt["dialogue_aggregate"]["TRAIN"]["turn_count"] > 0


def test_custody_dialogue_hash_excludes_rows_without_leaking_id(
    posix_tmp: Path,
) -> None:
    values = _member_values()
    values[p0.TRAIN_MEMBER] = _dialogues(
        "TRAIN",
        counts={family: 52 for family in p0.FAMILIES},
    )
    excluded_id = f"TRAIN-{p0.CONDITION_QUERY}-0000"
    excluded_hash = hashlib.sha256(excluded_id.encode("utf-8")).hexdigest()
    path, contract = _archive(posix_tmp, values=values)
    with mock.patch.object(
        p0,
        "EXCLUDED_DIALOGUE_ID_SHA256",
        frozenset({excluded_hash}),
    ):
        receipt = _qualify(posix_tmp, path, contract)
    exclusion = receipt["dialogue_aggregate"]["TRAIN"][
        "custody_exclusion"
    ]
    assert exclusion["excluded_dialogue_count"] == 1
    assert exclusion["excluded_eligible_row_count"] == 1
    assert excluded_id not in p0.canonical_bytes(receipt).decode("ascii")
    manifest = json.loads(
        (posix_tmp / "eligibility.private.json").read_text("ascii")
    )
    assert len(manifest["eligible_rows_by_split"]["TRAIN"]) == (
        52 * len(p0.FAMILIES) - 1
    )


def test_private_manifest_is_exclusive(
    posix_tmp: Path,
) -> None:
    path, contract = _archive(posix_tmp)
    private = posix_tmp / "eligibility.private.json"
    private.write_text("already here", encoding="ascii")
    with pytest.raises(
        p0.MultiDoc2DialP0QualificationError,
        match="not fresh",
    ):
        with mock.patch.object(
            p0.zipfile,
            "ZipFile",
            side_effect=AssertionError("source must not open"),
        ):
            p0.qualify_archive(
                archive_path=path,
                eligibility_manifest_path=private,
                archive_contract=contract,
            )
