from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest import mock

import pytest

from assumption_agent.benchmarks import (
    bioasq_p0_public_source_qualification_v1 as subject,
)


def _question(
    family: str,
    index: int,
    *,
    prefix: str = "PRIVATE",
    document: str | None = None,
    query: str | None = None,
    snippet: str | None = None,
) -> dict[str, object]:
    token = f"{family}-{index:03d}"
    document_value = (
        document
        if document is not None
        else f"https://private.invalid/document/{token}"
    )
    return {
        "body": query if query is not None else f"{prefix} QUERY {token}",
        "documents": [document_value],
        "exact_answer": f"{prefix} ANSWER {token}",
        "id": f"{prefix.lower()}-{token}",
        "ideal_answer": [f"{prefix} IDEAL {token}"],
        "snippets": [
            {
                "beginSection": "abstract",
                "document": document_value,
                "endSection": "abstract",
                "text": (
                    snippet
                    if snippet is not None
                    else f"{prefix} SNIPPET {token}"
                ),
            }
        ],
        "type": family,
    }


def _payload(per_family: int = 56) -> dict[str, object]:
    return {
        "questions": [
            _question(family, index)
            for family in subject.FAMILIES
            for index in range(per_family)
        ]
    }


def _raw(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _contract(
    value: int,
    *,
    expected_question_count: int | None = None,
) -> subject.QualificationContract:
    return subject.QualificationContract(
        {family: value for family in subject.FAMILIES},
        expected_question_count=(
            4 * value
            if expected_question_count is None
            else expected_question_count
        ),
    )


def _qualify_raw(
    raw: bytes,
    *,
    demand: int,
    expected_question_count: int | None = None,
) -> subject.QualificationResult:
    return subject._qualify_decoded_source(
        raw,
        source_binding={
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "synthetic_source_free_canary_input": True,
        },
        contract=_contract(
            demand,
            expected_question_count=expected_question_count,
        ),
        source_open_count=0,
        real_source_access_count=0,
    )


def _verify_self_hash(value: dict[str, object]) -> None:
    body = copy.deepcopy(value)
    declared = body.pop("self_sha256")
    assert declared == subject.stable_hash(body)


def test_production_path_opens_hashes_and_decodes_source_exactly_once(
    tmp_path: Path,
) -> None:
    raw = _raw(_payload())
    source = tmp_path / "training11b.json"
    source.write_bytes(raw)
    source_contract = subject.SourceFileContract(
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )

    with (
        mock.patch.object(
            subject,
            "_open_binary",
            wraps=subject._open_binary,
        ) as open_spy,
        mock.patch.object(
            subject,
            "_decode_strict_json",
            wraps=subject._decode_strict_json,
        ) as decode_spy,
    ):
        result = subject.qualify_source_path(
            source_path=source,
            source_contract=source_contract,
            contract=_contract(56),
        )

    assert open_spy.call_count == 1
    assert decode_spy.call_count == 1
    assert result.safe_receipt["status"] == (
        "qualified_public_non_scoring_schema_component_capacity"
    )
    assert result.safe_receipt["access_boundary"] == {
        "action_model_retrieval_evaluator_or_score_count": 0,
        "cohort_assignment_or_selection_secret_count": 0,
        "individual_item_query_document_snippet_or_commitment_published": (
            False
        ),
        "online_or_API_evaluation_count": 0,
        "real_source_access_count": 1,
        "source_hash_count": 1,
        "source_json_decode_count": 1,
        "source_open_count": 1,
    }


def test_identity_failure_precedes_json_decode(tmp_path: Path) -> None:
    raw = _raw(_payload())
    source = tmp_path / "training11b.json"
    source.write_bytes(raw)
    with mock.patch.object(
        subject,
        "_decode_strict_json",
        wraps=subject._decode_strict_json,
    ) as decode_spy:
        with pytest.raises(
            subject.BioasqP0QualificationError,
            match="identity drifted",
        ):
            subject.qualify_source_path(
                source_path=source,
                source_contract=subject.SourceFileContract(
                    size_bytes=len(raw),
                    sha256="0" * 64,
                ),
                contract=_contract(56),
            )
    assert decode_spy.call_count == 0


@pytest.mark.parametrize(
    "raw",
    [
        b'{"questions":[],"questions":[]}',
        (
            b'{"questions":[{"body":"x","documents":[],"id":"x",'
            b'"id":"y","snippets":[],"type":"yesno"}]}'
        ),
    ],
)
def test_duplicate_json_keys_fail_closed(raw: bytes) -> None:
    with pytest.raises(
        subject.BioasqP0QualificationError,
        match="duplicate object key",
    ):
        _qualify_raw(raw, demand=1)


def test_four_family_schema_and_snippet_document_resolution_fail_closed() -> None:
    unknown_family = _payload(per_family=1)
    unknown_family["questions"][0]["type"] = "other"  # type: ignore[index]
    with pytest.raises(
        subject.BioasqP0QualificationError,
        match="four-family registry",
    ):
        _qualify_raw(_raw(unknown_family), demand=1)

    unresolved = _payload(per_family=1)
    unresolved["questions"][0]["snippets"][0]["document"] = (  # type: ignore[index]
        "https://private.invalid/not-listed"
    )
    with pytest.raises(
        subject.BioasqP0QualificationError,
        match="absent from question documents",
    ):
        _qualify_raw(_raw(unresolved), demand=1)


def test_exact_top_level_question_count_is_required() -> None:
    with pytest.raises(
        subject.BioasqP0QualificationError,
        match="question count drifted",
    ):
        _qualify_raw(
            _raw(_payload(per_family=1)),
            demand=1,
            expected_question_count=5,
        )


def test_query_document_and_snippet_each_join_components() -> None:
    rows = [
        _question("yesno", 0, query="SHARED QUERY"),
        _question("factoid", 0, query="  shared   query "),
        _question(
            "list",
            0,
            document="https://private.invalid/shared-document",
        ),
        _question(
            "summary",
            0,
            document="https://private.invalid/shared-document",
        ),
        _question("yesno", 1, snippet="SHARED SNIPPET"),
        _question("factoid", 1, snippet=" shared snippet "),
        _question("list", 1),
        _question("summary", 1),
    ]
    result = _qualify_raw(
        _raw({"questions": rows}),
        demand=1,
        expected_question_count=8,
    )
    aggregate = result.safe_receipt["component_aggregate"]
    assert aggregate["component_count"] == 6
    assert aggregate["multi_question_component_count"] == 2
    assert aggregate["row_count_in_multi_question_components"] == 4
    assert result.safe_receipt["capacity"][
        "simultaneous_component_capacity_saturated"
    ] is True


def test_capacity_is_simultaneous_not_four_independent_marginals() -> None:
    questions: list[dict[str, object]] = []
    for component in range(2):
        shared_document = (
            f"https://private.invalid/shared-component/{component}"
        )
        for family in subject.FAMILIES:
            questions.append(
                _question(
                    family,
                    component,
                    prefix=f"PRIVATE-{component}",
                    document=shared_document,
                )
            )
    result = _qualify_raw(
        _raw({"questions": questions}),
        demand=2,
        expected_question_count=8,
    )
    capacity = result.safe_receipt["capacity"]
    assert capacity["assignable_component_count_by_family"] == {
        family: 2 for family in subject.FAMILIES
    }
    assert capacity["required_total"] == 8
    assert capacity["maximum_flow_assigned_total"] == 2
    assert capacity["simultaneous_component_capacity_saturated"] is False
    assert result.safe_receipt["status"] == (
        "terminal_public_source_component_capacity_insufficient"
    )


def test_safe_receipt_is_aggregate_only_and_private_rows_are_commitments() -> None:
    value = _payload()
    raw = _raw(value)
    result = _qualify_raw(raw, demand=56)
    safe = dict(result.safe_receipt)
    private = dict(result.private_manifest)
    _verify_self_hash(safe)
    _verify_self_hash(private)

    safe_rendered = subject.canonical_bytes(safe).decode("ascii")
    private_rendered = subject.canonical_bytes(private).decode("ascii")
    for forbidden in (
        "PRIVATE QUERY",
        "PRIVATE SNIPPET",
        "PRIVATE ANSWER",
        "PRIVATE IDEAL",
        "https://private.invalid",
    ):
        assert forbidden not in safe_rendered
        assert forbidden not in private_rendered
    assert "rows" not in safe
    assert safe["access_boundary"][
        "individual_item_query_document_snippet_or_commitment_published"
    ] is False

    rows = private["rows"]
    assert len(rows) == 4 * 56
    allowed = {
        "component_commitment",
        "family",
        "opaque_item_commitment",
        "query_commitment",
        "snippet_commitments",
    }
    for row in rows:
        assert set(row) == allowed
        assert row["family"] in subject.FAMILIES
        for key in (
            "component_commitment",
            "opaque_item_commitment",
            "query_commitment",
        ):
            assert len(row[key]) == 64
        assert row["snippet_commitments"]
        assert all(len(value) == 64 for value in row["snippet_commitments"])


def test_snippet_commitment_binds_case_preserved_document_nul_text() -> None:
    value = _payload(per_family=1)
    yesno = value["questions"][0]  # type: ignore[index]
    factoid = value["questions"][1]  # type: ignore[index]
    yesno["snippets"][0]["text"] = "Case  Sensitive Evidence"  # type: ignore[index]
    factoid["snippets"][0]["text"] = "Case Sensitive Evidence"  # type: ignore[index]
    result = _qualify_raw(_raw(value), demand=1)
    rows = {
        row["family"]: row
        for row in result.private_manifest["rows"]
    }
    yesno_document = yesno["documents"][0]  # type: ignore[index]
    expected = subject._commit(
        "normalized_gold_snippet",
        yesno_document + "\0" + "Case Sensitive Evidence",
    )
    assert rows["yesno"]["snippet_commitments"] == [expected]
    assert (
        rows["yesno"]["snippet_commitments"]
        != rows["factoid"]["snippet_commitments"]
    )


def test_outputs_are_exclusive_canonical_mode_0600_and_bound(
    tmp_path: Path,
) -> None:
    raw = _raw(_payload())
    source = tmp_path / "training11b.json"
    source.write_bytes(raw)
    private_path = tmp_path / "work" / "eligibility.private.json"
    safe_path = tmp_path / "work" / "qualification.safe.json"
    result = subject.qualify_source_path(
        source_path=source,
        source_contract=subject.SourceFileContract(
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        ),
        private_manifest_path=private_path,
        safe_receipt_path=safe_path,
        contract=_contract(56),
    )
    assert private_path.read_bytes() == subject.canonical_bytes(
        result.private_manifest
    )
    assert safe_path.read_bytes() == subject.canonical_bytes(
        result.safe_receipt
    )
    assert stat_mode(private_path) == 0o600
    assert stat_mode(safe_path) == 0o600
    assert hashlib.sha256(private_path.read_bytes()).hexdigest() == (
        result.safe_receipt["private_manifest_binding"]["file_sha256"]
    )
    with pytest.raises(
        subject.BioasqP0QualificationError,
        match="written exclusively",
    ):
        subject.qualify_source_path(
            source_path=source,
            source_contract=subject.SourceFileContract(
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            ),
            private_manifest_path=private_path,
            safe_receipt_path=safe_path,
            contract=_contract(56),
        )


def stat_mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777


def test_cli_formal_failure_writes_aggregate_only_no_retry_receipt(
    tmp_path: Path,
) -> None:
    private_value = "PRIVATE FAILURE VALUE"
    raw = (
        b'{"questions":[{"body":"'
        + private_value.encode("ascii")
        + b'","body":"duplicate"}]}'
    )
    source = tmp_path / "training11b.json"
    source.write_bytes(raw)
    private_path = tmp_path / "work" / "eligibility.private.json"
    safe_path = tmp_path / "work" / "qualification.safe.json"
    result = subject.main(
        [
            "--source",
            str(source),
            "--expected-size-bytes",
            str(len(raw)),
            "--expected-sha256",
            hashlib.sha256(raw).hexdigest(),
            "--private-manifest",
            str(private_path),
            "--safe-receipt",
            str(safe_path),
        ]
    )
    assert result == 1
    assert not private_path.exists()
    receipt = json.loads(safe_path.read_text(encoding="ascii"))
    assert receipt["status"] == (
        "terminal_public_source_qualification_failure_no_retry"
    )
    assert receipt["failure_stage"] == "decode_strict_json"
    assert receipt["access_boundary"] == {
        "action_model_retrieval_evaluator_or_score_count": 0,
        "cohort_assignment_or_selection_secret_count": 0,
        "individual_item_query_document_snippet_or_commitment_published": (
            False
        ),
        "online_or_API_evaluation_count": 0,
        "real_source_access_count": 1,
        "source_hash_count": 1,
        "source_json_decode_count": 1,
        "source_open_count": 1,
    }
    assert private_value not in safe_path.read_text(encoding="ascii")
    _verify_self_hash(receipt)


def test_source_free_canary_uses_same_production_entrypoint_and_no_source() -> None:
    with (
        mock.patch.object(
            subject,
            "_qualify_decoded_source",
            wraps=subject._qualify_decoded_source,
        ) as qualify_spy,
        mock.patch.object(
            subject,
            "_open_binary",
            side_effect=AssertionError("canary opened a source path"),
        ) as open_spy,
    ):
        receipt = subject.run_source_free_canary()
    assert qualify_spy.call_count == 1
    assert open_spy.call_count == 0
    assert receipt["status"] == (
        "passed_source_free_production_parser_component_canary"
    )
    assert receipt["parser_component_entrypoint"] == (
        "_qualify_decoded_source"
    )
    assert receipt["formal_source_access_count"] == 0
    assert receipt["source_open_count"] == 0
    assert receipt["source_json_decode_count"] == 1
    assert receipt["external_distribution_import_count"] == 0
    assert receipt["synthetic_component_count"] == 4 * 56
    assert receipt["synthetic_component_capacity_saturated"] is True
    assert "Synthetic source-free" not in (
        subject.canonical_bytes(receipt).decode("ascii")
    )
    _verify_self_hash(dict(receipt))


def test_module_imports_are_stdlib_only() -> None:
    path = Path(subject.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots - set(sys.stdlib_module_names) - {
        "__future__"
    } == set()
