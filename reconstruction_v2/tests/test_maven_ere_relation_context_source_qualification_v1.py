from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import stat
import tempfile
from typing import Any

import pytest

from assumption_agent.benchmarks import (
    maven_ere_relation_context_source_qualification_v1 as qualifier,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _mention(identifier: str, trigger: str, sentence: int) -> dict[str, Any]:
    return {
        "id": identifier,
        "trigger_word": trigger,
        "sent_id": sentence,
        "offset": [0, 1],
    }


def _document(
    tag: str,
    family: str,
    *,
    extra_family: str | None = None,
    duplicate_fine_label: bool = False,
) -> dict[str, Any]:
    temporal = {label: [] for label in qualifier.TEMPORAL_LABELS}
    causal = {label: [] for label in qualifier.CAUSAL_LABELS}
    subevent: list[list[str]] = []

    def add(target_family: str) -> None:
        if target_family == "TEMPORAL":
            temporal["BEFORE"].append(["E1", "E2"])
            if duplicate_fine_label:
                temporal["OVERLAP"].append(["E1", "E2"])
        elif target_family == "CAUSAL":
            causal["CAUSE"].append(["E1", "E2"])
        elif target_family == "SUBEVENT":
            subevent.append(["E1", "E2"])
        else:
            raise AssertionError(target_family)

    add(family)
    if extra_family is not None:
        add(extra_family)
    return {
        "id": f"doc-{tag}",
        "title": f"Title {tag}",
        "tokens": [
            [f"trigger-{tag}-{index}", "context"] for index in range(6)
        ],
        "sentences": [f"ignored sentence {index}" for index in range(6)],
        "events": [
            {
                "id": "E1",
                "type": "Cause_expansion",
                "type_id": 1,
                "mention": [_mention(f"M1-{tag}", f"head-{tag}", 0)],
            },
            {
                "id": "E2",
                "type": "Effect",
                "type_id": 2,
                "mention": [_mention(f"M2-{tag}", f"tail-{tag}", 1)],
            },
        ],
        "TIMEX": [],
        "temporal_relations": temporal,
        "causal_relations": causal,
        "subevent_relations": subevent,
    }


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _patch_small_contract(monkeypatch: pytest.MonkeyPatch, count: int = 3) -> None:
    monkeypatch.setattr(
        qualifier,
        "EXPECTED_LINE_COUNTS",
        {"train": count, "valid": count},
    )
    monkeypatch.setattr(
        qualifier,
        "REQUIRED_PER_SPLIT_FAMILY",
        {
            (split, family): 1
            for split in ("train", "valid")
            for family in qualifier.FAMILY_ORDER
        },
    )


def _three_family_rows(prefix: str) -> list[dict[str, Any]]:
    return [
        _document(f"{prefix}-{family}", family)
        for family in qualifier.FAMILY_ORDER
    ]


def _verify_self_hash(receipt: dict[str, Any]) -> None:
    body = copy.deepcopy(receipt)
    declared = body.pop("qualification_sha256")
    assert hashlib.sha256(_canonical(body)).hexdigest() == declared


def test_small_reader_equivalent_source_passes_without_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_contract(monkeypatch)
    receipt = qualifier.qualify_decoded_jsonl(
        _jsonl(_three_family_rows("train")),
        _jsonl(_three_family_rows("valid")),
        formal_identity_enforced=False,
        source_binding={"fixture": "synthetic"},
    )
    assert receipt["status"] == "passed_source_qualification_no_selection"
    assert receipt["simultaneous_document_assignment_capacity"][
        "deterministic_max_flow_assigned_document_count"
    ] == 6
    assert receipt["claim_boundary"] == {
        "qualification_only_no_efficacy_claim": True,
        "selection_secret_generated_or_opened": False,
        "cohort_selected_or_materialized": False,
        "retrieval_action_evaluator_classifier_or_score_run": False,
        "online_or_external_evaluation_used": False,
        "hidden_TEST_member_opened": False,
        "document_item_title_alias_trigger_relation_pair_ordinal_or_per_document_hash_emitted": False,
    }
    serialized = _canonical(receipt).decode("ascii")
    assert "trigger-train" not in serialized
    assert "Title train" not in serialized
    _verify_self_hash(receipt)


def test_empty_tokens_are_allowed_but_empty_rendered_sentence_is_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_contract(monkeypatch, count=4)
    train = _three_family_rows("train")
    allowed = _document("allowed-empty-token", "TEMPORAL")
    allowed["tokens"][2] = ["", "still-content"]
    train.append(allowed)
    valid = _three_family_rows("valid")
    invalid = _document("invalid-empty-sentence", "TEMPORAL")
    invalid["tokens"][2] = [""]
    valid.append(invalid)

    receipt = qualifier.qualify_decoded_jsonl(
        _jsonl(train),
        _jsonl(valid),
        formal_identity_enforced=False,
        source_binding={},
    )
    assert receipt["status"] == "passed_source_qualification_no_selection"
    assert receipt["split_aggregates"]["train"][
        "reader_invalid_document_count"
    ] == 0
    assert receipt["split_aggregates"]["valid"][
        "invalid_document_reason_counts"
    ] == {"empty_rendered_sentence": 1}


def test_cross_family_pairs_use_pre_row_priority_and_fine_ambiguity_is_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_contract(monkeypatch, count=5)
    train = _three_family_rows("train") + [
        _document("ambiguous", "TEMPORAL", extra_family="CAUSAL"),
        _document("multi-fine", "TEMPORAL", duplicate_fine_label=True),
    ]
    valid = _three_family_rows("valid") + [
        _document("ambiguous-v", "TEMPORAL", extra_family="SUBEVENT"),
        _document("multi-fine-v", "TEMPORAL", duplicate_fine_label=True),
    ]
    receipt = qualifier.qualify_decoded_jsonl(
        _jsonl(train),
        _jsonl(valid),
        formal_identity_enforced=False,
        source_binding={},
    )
    assert receipt["status"] == "passed_source_qualification_no_selection"
    assert receipt["split_aggregates"]["train"][
        "eligible_unique_family_pair_candidate_counts"
    ] == {"TEMPORAL": 1, "CAUSAL": 2, "SUBEVENT": 1}
    assert receipt["split_aggregates"]["valid"][
        "eligible_unique_family_pair_candidate_counts"
    ] == {"TEMPORAL": 1, "CAUSAL": 1, "SUBEVENT": 2}


def test_invalid_JSON_and_schema_documents_are_aggregated_and_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_contract(monkeypatch, count=5)
    valid_rows = _three_family_rows("valid")
    valid_rows.extend(
        (
            _document("extra-valid-a", "TEMPORAL"),
            _document("extra-valid-b", "CAUSAL"),
        )
    )
    train_raw = _jsonl(_three_family_rows("train"))
    train_raw += b'{"id":"duplicate","id":"again"}\n'
    train_raw += b"not-json\n"
    receipt = qualifier.qualify_decoded_jsonl(
        train_raw,
        _jsonl(valid_rows),
        formal_identity_enforced=False,
        source_binding={},
    )
    assert receipt["status"] == "passed_source_qualification_no_selection"
    assert receipt["split_aggregates"]["train"][
        "invalid_document_reason_counts"
    ] == {"json_line": 2}
    assert receipt["split_aggregates"]["train"][
        "reader_valid_document_count"
    ] == 3


def test_line_count_drift_and_capacity_shortfall_are_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_contract(monkeypatch, count=4)
    receipt = qualifier.qualify_decoded_jsonl(
        _jsonl(_three_family_rows("train")),
        _jsonl(_three_family_rows("valid")),
        formal_identity_enforced=False,
        source_binding={},
    )
    assert receipt["status"] == "terminal_source_incompatible_no_selection"
    assert receipt["terminal_reason_counts"]["split_line_count_drift_count"] == 2

    _patch_small_contract(monkeypatch, count=3)
    train = [_document(f"only-temporal-{index}", "TEMPORAL") for index in range(3)]
    valid = [_document(f"only-temporal-v-{index}", "TEMPORAL") for index in range(3)]
    receipt = qualifier.qualify_decoded_jsonl(
        _jsonl(train),
        _jsonl(valid),
        formal_identity_enforced=False,
        source_binding={},
    )
    assert receipt["terminal_reason_counts"][
        "simultaneous_assignment_shortfall_count"
    ] == 1


def test_collision_component_cap_is_global_across_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_contract(monkeypatch)
    train = _three_family_rows("same")
    valid = copy.deepcopy(train)
    receipt = qualifier.qualify_decoded_jsonl(
        _jsonl(train),
        _jsonl(valid),
        formal_identity_enforced=False,
        source_binding={},
    )
    capacity = receipt["simultaneous_document_assignment_capacity"]
    assert capacity["collision_component_count"] == 3
    assert capacity["multi_document_collision_component_count"] == 3
    assert capacity["deterministic_max_flow_assigned_document_count"] == 3
    assert receipt["status"] == "terminal_source_incompatible_no_selection"


def test_bound_source_requires_regular_mode0600_and_exact_hash() -> None:
    with tempfile.TemporaryDirectory(prefix="maven-qualifier-", dir="/tmp") as root:
        path = Path(root) / "source.jsonl"
        raw = b"{}\n"
        path.write_bytes(raw)
        path.chmod(0o600)
        assert qualifier._read_bound_source(
            path,
            size=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        ) == raw
        path.chmod(0o644)
        with pytest.raises(
            qualifier.MavenEreSourceQualificationError,
            match="identity drifted",
        ):
            qualifier._read_bound_source(
                path,
                size=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            )


def test_formal_output_requires_exact_empty_mode0700() -> None:
    with tempfile.TemporaryDirectory(prefix="maven-output-", dir="/tmp") as root:
        project = Path(root) / "project"
        expected = project / qualifier.FORMAL_OUTPUT_RELATIVE_PATH
        expected.mkdir(parents=True, mode=0o700)
        expected.chmod(0o700)
        assert qualifier._require_formal_output(project, expected) == expected.resolve()
        assert stat.S_IMODE(expected.stat().st_mode) == 0o700
        (expected / "existing").write_text("x", "ascii")
        with pytest.raises(qualifier.OneShotRefusal):
            qualifier._require_formal_output(project, expected)
