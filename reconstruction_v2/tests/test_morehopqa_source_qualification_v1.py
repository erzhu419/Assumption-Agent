from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from assumption_agent.benchmarks import morehopqa_source_qualification_v1 as audit


PRIVATE_MARKERS = (
    "PRIVATE_ID",
    "PRIVATE_QUESTION",
    "PRIVATE_ANSWER",
    "PRIVATE_TITLE",
    "PRIVATE_BODY",
    "PRIVATE_SUPPORT",
    "PRIVATE_PATTERN",
    "PRIVATE_EXTRA_KEY_DO_NOT_LEAK",
    "PRIVATE_EXTRA_VALUE_DO_NOT_LEAK",
    "PRIVATE_DETAILS_KEY_DO_NOT_LEAK",
    "PRIVATE_DETAILS_VALUE_DO_NOT_LEAK",
)


def _reasoning(index: int) -> str:
    if 150 <= index < 182:
        return "Symbolic"
    if 182 <= index < 214:
        return "Arithmetic"
    if 214 <= index < 246:
        return "Commonsense"
    return "[[Symbolic / Arithmetic、Commonsense]]"


def _row(index: int) -> dict[str, Any]:
    title_a = f"PRIVATE_TITLE_A_{index}"
    title_b = f"PRIVATE_TITLE_B_{index}"
    return {
        "_id": f"PRIVATE_ID_{index}",
        "question": f"PRIVATE_QUESTION_{index}?",
        "answer": f"PRIVATE_ANSWER_{index}",
        "previous_question": f"PRIVATE_PREVIOUS_QUESTION_{index}?",
        "previous_answer": f"PRIVATE_PREVIOUS_ANSWER_{index}",
        "question_decomposition": [
            {
                "sub_id": "1",
                "question": f"PRIVATE_SUBQUESTION_A_{index}?",
                "answer": f"PRIVATE_SUBANSWER_A_{index}",
                "paragraph_support_title": f"  private_title_a_{index}  ",
                "PRIVATE_DETAILS_KEY_DO_NOT_LEAK": {
                    "value": "PRIVATE_DETAILS_VALUE_DO_NOT_LEAK"
                },
            },
            {
                "sub_id": "2",
                "question": f"PRIVATE_SUBQUESTION_B_{index}?",
                "answer": f"PRIVATE_SUBANSWER_B_{index}",
                "paragraph_support_title": title_b,
            },
            {
                "sub_id": "3",
                "question": f"PRIVATE_REASONING_QUESTION_{index}?",
                "answer": f"PRIVATE_REASONING_ANSWER_{index}",
                "paragraph_support_title": "",
            },
        ],
        "context": [
            [title_a, [f"PRIVATE_BODY_A_{index}", "PRIVATE_BODY_SHARED"]],
            [title_b, [f"PRIVATE_BODY_B_{index}"]],
        ],
        "answer_type": audit.ANSWER_TYPES[index % len(audit.ANSWER_TYPES)],
        "previous_answer_type": "person",
        "no_of_hops": 3,
        "reasoning_type": _reasoning(index),
        "pattern": f"PRIVATE_PATTERN_{index}",
        "subquestion_patterns": [f"PRIVATE_SUBPATTERN_{index}"],
        "cutted_question": f"PRIVATE_CUTTED_QUESTION_{index}",
        "ques_on_last_hop": f"PRIVATE_LAST_HOP_{index}?",
        "PRIVATE_EXTRA_KEY_DO_NOT_LEAK": "PRIVATE_EXTRA_VALUE_DO_NOT_LEAK",
    }


def _payload() -> list[dict[str, Any]]:
    return [_row(index) for index in range(audit.FORMAL_ROOT_COUNT)]


def _raw(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _exclusion_raw(payload: list[dict[str, Any]] | None = None) -> bytes:
    source = _payload() if payload is None else payload
    return _raw(
        [
            {
                "_id": row["_id"],
                "PRIVATE_EXTRA_KEY_DO_NOT_LEAK": (
                    "PRIVATE_EXTRA_VALUE_DO_NOT_LEAK"
                ),
            }
            for row in source[: audit.FORMAL_EXCLUSION_ROOT_COUNT]
        ]
    )


def _build(payload: list[dict[str, Any]]) -> dict[str, Any]:
    return audit.build_synthetic_aggregate(_raw(payload), _exclusion_raw())


def _without_hash(receipt: dict[str, Any]) -> dict[str, Any]:
    body = dict(receipt)
    body.pop("qualification_sha256")
    return body


def _contains_list(value: Any) -> bool:
    if isinstance(value, list):
        return True
    if isinstance(value, dict):
        return any(_contains_list(child) for child in value.values())
    return False


def _git(project: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(project), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _git_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts" / "morehopqa_official_source_v1").mkdir(
        parents=True
    )
    (project / "manifests").mkdir()
    (project / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    _git(project, "init", "-q")
    _git(project, "config", "user.email", "synthetic@example.invalid")
    _git(project, "config", "user.name", "Synthetic Test")
    _git(project, "add", ".gitignore")
    _git(project, "commit", "-q", "-m", "synthetic root")
    return project


def test_valid_aggregate_is_exact_and_contains_no_item_values() -> None:
    receipt = _build(_payload())
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    for marker in PRIVATE_MARKERS:
        assert marker.casefold() not in serialized.casefold()
    assert not _contains_list(receipt)
    assert receipt["schema"] == audit.SCHEMA
    assert receipt["source_binding"]["formal_source_identity_enforced"] is False
    exclusion = receipt["official_public_example_exclusion"]
    assert exclusion["formal_exclusion_identity_enforced"] is False
    assert exclusion["deny_id_count"] == 150
    assert exclusion["matched_source_item_count"] == 150
    assert exclusion["excluded_structurally_eligible_item_count"] == 150
    assert len(exclusion["deny_id_set_sha256"]) == 64
    assert receipt["identity_uniqueness"] == {
        "unique_id_count": 1118,
        "duplicate_id_count": 0,
        "unique_normalized_question_count": 1118,
        "duplicate_normalized_question_count": 0,
    }
    assert receipt["reasoning"]["token_set_counts"] == {
        "Symbolic": 32,
        "Arithmetic": 32,
        "Commonsense": 32,
        "Symbolic+Arithmetic": 0,
        "Symbolic+Commonsense": 0,
        "Arithmetic+Commonsense": 0,
        "Symbolic+Arithmetic+Commonsense": 1022,
    }
    assert receipt["contexts"]["item_context_cardinality_counts"] == {"2": 1118}
    assert receipt["contexts"][
        "unique_normalized_title_and_body_document_count"
    ] == 2236
    assert receipt["support_title_resolution"][
        "exactly_resolved_support_reference_count"
    ] == 2236
    assert receipt["support_title_resolution"]["empty_support_reference_count"] == 1118
    assert receipt["gold_and_capacity"][
        "fully_resolved_distinct_gold_document_cardinality_counts"
    ] == {"2": 1118}
    assert receipt["gold_and_capacity"][
        "exact_three_family_b_matching_capacity_met"
    ] is True
    assert receipt["gold_and_capacity"][
        "eligible_normalized_question_count_after_public_exclusion"
    ] == 968
    assert receipt["gold_and_capacity"][
        "conservative_eligible_token_set_counts"
    ] == {
        "Symbolic": 32,
        "Arithmetic": 32,
        "Commonsense": 32,
        "Symbolic+Arithmetic": 0,
        "Symbolic+Commonsense": 0,
        "Arithmetic+Commonsense": 0,
        "Symbolic+Arithmetic+Commonsense": 872,
    }
    assert receipt["qualification_sha256"] == hashlib.sha256(
        json.dumps(
            _without_hash(receipt),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants() -> None:
    raw = _raw(_payload())
    duplicate = raw.replace(
        b'"_id":"PRIVATE_ID_0"',
        b'"_id":"PRIVATE_ID_0","_id":"PRIVATE_ID_DUPLICATE"',
        1,
    )
    with pytest.raises(
        audit.MoreHopQASourceQualificationError, match="duplicate JSON object key"
    ):
        audit.build_synthetic_aggregate(duplicate, _exclusion_raw())
    nonfinite = raw.replace(b'"no_of_hops":3', b'"no_of_hops":NaN', 1)
    with pytest.raises(
        audit.MoreHopQASourceQualificationError, match="non-finite"
    ):
        audit.build_synthetic_aggregate(nonfinite, _exclusion_raw())
    exclusion_duplicate = _exclusion_raw().replace(
        b'"_id":"PRIVATE_ID_0"',
        b'"_id":"PRIVATE_ID_0","_id":"PRIVATE_ID_DUPLICATE"',
        1,
    )
    with pytest.raises(
        audit.MoreHopQASourceQualificationError, match="duplicate JSON object key"
    ):
        audit.build_synthetic_aggregate(raw, exclusion_duplicate)


def test_root_count_and_public_schema_types_are_strict() -> None:
    short = _payload()[:-1]
    with pytest.raises(
        audit.MoreHopQASourceQualificationError, match="frozen 1118"
    ):
        audit.build_synthetic_aggregate(_raw(short), _exclusion_raw())

    malformed = _payload()
    malformed[0]["context"][0][1] = "PRIVATE_BODY_NOT_A_LIST"
    with pytest.raises(
        audit.MoreHopQASourceQualificationError, match="wrong type"
    ):
        audit.build_synthetic_aggregate(_raw(malformed), _exclusion_raw())

    malformed = _payload()
    del malformed[0]["question_decomposition"][0]["sub_id"]
    with pytest.raises(
        audit.MoreHopQASourceQualificationError, match="missing a public field"
    ):
        audit.build_synthetic_aggregate(_raw(malformed), _exclusion_raw())


def test_duplicate_id_is_terminal() -> None:
    payload = _payload()
    payload[1]["_id"] = payload[0]["_id"]
    with pytest.raises(
        audit.MoreHopQASourceQualificationError, match="duplicate_id_count=1"
    ):
        _build(payload)


def test_normalized_question_collision_is_counted_and_conservatively_deduplicated() -> None:
    payload = _payload()
    payload[151]["question"] = "  PRIVATE_QUESTION_150?  "
    receipt = _build(payload)
    assert receipt["identity_uniqueness"] == {
        "unique_id_count": 1118,
        "duplicate_id_count": 0,
        "unique_normalized_question_count": 1117,
        "duplicate_normalized_question_count": 1,
    }
    assert receipt["gold_and_capacity"][
        "eligible_normalized_question_count_after_public_exclusion"
    ] == 967


def test_reasoning_parser_accepts_unfrozen_delimiters_but_rejects_residue() -> None:
    payload = _payload()
    payload[96]["reasoning_type"] = "{Commonsense & Symbolic | Arithmetic}"
    receipt = _build(payload)
    assert receipt["reasoning"]["token_set_counts"][
        "Symbolic+Arithmetic+Commonsense"
    ] == 1022

    payload = _payload()
    payload[0]["reasoning_type"] = "Symbolic + Logical"
    with pytest.raises(
        audit.MoreHopQASourceQualificationError,
        match=r"unknown_residue_count=7",
    ):
        _build(payload)


def test_reasoning_duplicate_or_missing_registry_token_is_terminal() -> None:
    payload = _payload()
    payload[0]["reasoning_type"] = "Symbolic / Symbolic"
    payload[1]["reasoning_type"] = "[ / ]"
    with pytest.raises(
        audit.MoreHopQASourceQualificationError,
        match=r"duplicate_token_count=1; missing_token_count=1",
    ):
        _build(payload)


def test_variable_context_and_gold_cardinality_are_aggregated_without_two_or_609() -> None:
    payload = _payload()
    payload[0]["context"].append(
        ["PRIVATE_TITLE_C_0", ["PRIVATE_BODY_C_0"]]
    )
    payload[0]["question_decomposition"].append(
        {
            "sub_id": "4",
            "question": "PRIVATE_SUBQUESTION_C_0?",
            "answer": "PRIVATE_SUBANSWER_C_0",
            "paragraph_support_title": "private_title_c_0",
        }
    )
    receipt = _build(payload)
    assert receipt["contexts"]["item_context_cardinality_counts"] == {
        "2": 1117,
        "3": 1,
    }
    assert receipt["contexts"]["total_context_document_occurrences"] == 2237
    assert receipt["gold_and_capacity"][
        "fully_resolved_distinct_gold_document_cardinality_counts"
    ] == {"2": 1117, "3": 1}
    assert receipt["gold_and_capacity"]["fixed_gold_document_cardinality_assumed"] is False
    assert receipt["gold_and_capacity"]["fixed_corpus_size_assumed"] is False


def test_missing_and_ambiguous_supports_reduce_exact_family_capacity() -> None:
    payload = _payload()
    payload[150]["question_decomposition"][0][
        "paragraph_support_title"
    ] = "PRIVATE_SUPPORT_MISSING"
    payload[182]["context"].append(
        [" private_title_a_182 ", ["PRIVATE_BODY_AMBIGUOUS_182"]]
    )
    receipt = _build(payload)
    resolution = receipt["support_title_resolution"]
    assert resolution["missing_support_reference_count"] == 1
    assert resolution["ambiguous_support_reference_count"] == 1
    assert resolution["fully_exactly_resolved_item_count"] == 1116
    capacity = receipt["gold_and_capacity"]
    assert capacity[
        "eligible_normalized_question_count_after_public_exclusion"
    ] == 966
    assert capacity["exact_three_family_b_matching_capacity_met"] is True


def test_exact_family_b_matching_capacity_failure_is_terminal() -> None:
    payload = _payload()
    for row in payload[audit.FORMAL_EXCLUSION_ROOT_COUNT :]:
        row["reasoning_type"] = "Symbolic"
    receipt = _build(payload)
    capacity = receipt["gold_and_capacity"]
    assert capacity["hall_shortfall_counts"]["Arithmetic"] == 72
    assert capacity["hall_shortfall_counts"]["Commonsense"] == 72
    assert capacity["exact_three_family_b_matching_capacity_met"] is False
    assert receipt["qualification_status"].startswith("terminal_source_infeasible")


def test_full_source_document_identity_deduplicates_exact_normalized_content() -> None:
    payload = _payload()
    payload[1]["context"] = deepcopy(payload[0]["context"])
    payload[1]["question_decomposition"][0][
        "paragraph_support_title"
    ] = payload[0]["question_decomposition"][0]["paragraph_support_title"]
    payload[1]["question_decomposition"][1][
        "paragraph_support_title"
    ] = payload[0]["question_decomposition"][1]["paragraph_support_title"]
    receipt = _build(payload)
    contexts = receipt["contexts"]
    assert contexts["total_context_document_occurrences"] == 2236
    assert contexts["unique_normalized_title_and_body_document_count"] == 2234
    assert contexts[
        "duplicate_normalized_title_and_body_document_occurrences"
    ] == 2


def test_fixed_ignored_source_binding_uses_only_synthetic_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _git_project(tmp_path)
    source = project / audit.FORMAL_SOURCE_RELATIVE_PATH
    exclusion = project / audit.FORMAL_EXCLUSION_RELATIVE_PATH
    exclusion.parent.mkdir(parents=True)
    raw = _raw(_payload())
    exclusion_raw = _exclusion_raw()
    source.write_bytes(raw)
    exclusion.write_bytes(exclusion_raw)
    source.chmod(0o600)
    exclusion.chmod(0o600)
    monkeypatch.setattr(audit, "FORMAL_SOURCE_SIZE", len(raw))
    monkeypatch.setattr(audit, "FORMAL_SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(
        audit, "FORMAL_SOURCE_GIT_BLOB_SHA1", audit._git_blob_sha1(raw)
    )
    monkeypatch.setattr(audit, "FORMAL_EXCLUSION_SIZE", len(exclusion_raw))
    monkeypatch.setattr(
        audit,
        "FORMAL_EXCLUSION_SHA256",
        hashlib.sha256(exclusion_raw).hexdigest(),
    )
    monkeypatch.setattr(
        audit,
        "FORMAL_EXCLUSION_GIT_BLOB_SHA1",
        audit._git_blob_sha1(exclusion_raw),
    )
    # Pytest's configured Windows-mounted temp root reports 0777 after chmod.
    monkeypatch.setattr(audit.stat, "S_IMODE", lambda _mode: 0o600)
    receipt = audit.build_aggregate(project)
    assert receipt["source_binding"]["formal_source_identity_enforced"] is True
    assert receipt["source_binding"]["size"] == len(raw)

    _git(project, "add", "-f", audit.FORMAL_SOURCE_RELATIVE_PATH.as_posix())
    _git(project, "commit", "-q", "-m", "synthetic tracked source")
    with pytest.raises(
        audit.MoreHopQASourceQualificationError, match="tracked"
    ):
        audit.build_aggregate(project)


def test_fixed_cli_consumes_marker_and_exclusively_publishes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _git_project(tmp_path)
    calls = 0
    synthetic_receipt = _build(_payload())

    def fake_build(candidate: str | Path) -> dict[str, Any]:
        nonlocal calls
        assert Path(candidate) == project
        assert (project / audit.FORMAL_ATTEMPT_MARKER_RELATIVE_PATH).is_file()
        calls += 1
        return synthetic_receipt

    monkeypatch.setattr(audit, "build_aggregate", fake_build)
    assert audit.main(["--project", str(project)]) == 0
    marker = project / audit.FORMAL_ATTEMPT_MARKER_RELATIVE_PATH
    output = project / audit.FORMAL_OUTPUT_RELATIVE_PATH
    assert marker.is_file()
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == synthetic_receipt
    assert calls == 1

    with pytest.raises(FileExistsError, match="already exists"):
        audit.main(["--project", str(project)])
    assert calls == 1
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_exclusive_writer_never_overwrites_existing_manifest(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    output.write_text("sentinel", encoding="utf-8")
    with pytest.raises(FileExistsError):
        audit._write_json_exclusive(output, {"schema": audit.SCHEMA})
    assert output.read_text(encoding="utf-8") == "sentinel"
    assert not list(tmp_path.glob(".receipt.json.*.tmp"))
