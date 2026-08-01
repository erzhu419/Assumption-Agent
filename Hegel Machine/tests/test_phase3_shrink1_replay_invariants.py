from __future__ import annotations

import json
from pathlib import Path

import pytest

from hegel_machine import phase3_shrink1_replay_v1 as replay


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "parent_dsl_version",
        "dsl_version",
        "freeze_version",
        "ast_schema_id",
        "cbor_profile_id",
        "ast_hash_domain",
    ],
)
def test_golden_vector_metadata_drift_fails_closed(
    tmp_path: Path, field: str
) -> None:
    payload = json.loads(replay.GOLDEN_VECTOR_PATH.read_text(encoding="utf-8"))
    payload[field] = "DRIFT"
    path = tmp_path / "vectors.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        replay.python_shrink1_vector_report(path)


def test_golden_vector_groups_must_be_lists(tmp_path: Path) -> None:
    payload = json.loads(replay.GOLDEN_VECTOR_PATH.read_text(encoding="utf-8"))
    payload["formal_reject_vectors"] = {"not": "a list"}
    path = tmp_path / "vectors.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="formal_reject_vectors must be a list"):
        replay.python_shrink1_vector_report(path)


def test_python_source_commitment_includes_the_capacity_generator() -> None:
    capacity_source = (
        replay.PROJECT_ROOT
        / "src"
        / "hegel_machine"
        / "phase3_shrink1_capacity_v1.py"
    )
    assert capacity_source in replay.PYTHON_STRICT_SOURCES
    assert replay.PYTHON_STRICT_SOURCES.count(capacity_source) == 1
    assert replay.PYTHON_CAPACITY_SOURCES == replay.PYTHON_STRICT_SOURCES


def test_python_subset_status_fails_closed_on_generator_count_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        replay,
        "iter_shrink1_capacity_candidate_asts",
        lambda: iter([("scalar_const", 0, 1)]),
    )

    report = replay.python_shrink1_capacity_replay()

    assert report["source_candidate_count"] == 1
    assert report["subset_status"] == "INCONCLUSIVE_EXECUTION"
    assert report["executed_closure_status"] == "NOT_RUN"
    assert report["complete_closure_enumerated"] is False


def _frozen_shape_report(*, implementation: str) -> dict[str, object]:
    return {
        "implementation": implementation,
        "dsl_version": replay.DSL_VERSION,
        "freeze_version": replay.FREEZE_VERSION,
        "source_candidate_count": replay.EXPECTED_SHRINK1_SOURCE_COUNT,
        "accepted_source_count": replay.EXPECTED_SHRINK1_SOURCE_COUNT,
        "accepted_unique_count": replay.EXPECTED_SHRINK1_SOURCE_COUNT,
        "rejected_count": 0,
        "rejection_counts": {},
        "rewrite_collapsed_count": 0,
        "accepted_set_commitment": (
            replay.EXPECTED_SHRINK1_CAPACITY_SET_COMMITMENT
        ),
        "canonical_program_budget": replay.CANONICAL_PROGRAM_BUDGET,
        "first_out_of_budget_ordinal": None,
        "first_out_of_budget_cbor_hex": None,
        "first_out_of_budget_ast_hash": None,
        "subset_status": "VERIFIED_WITHIN_BUDGET",
        "executed_closure_status": "NOT_RUN",
        "complete_closure_enumerated": False,
        "interpreted_as_complete_closure": False,
    }


def _patch_dual_inputs(
    monkeypatch: pytest.MonkeyPatch,
    python: dict[str, object],
    rust: dict[str, object],
) -> None:
    gate = {
        "status": "VERIFIED",
        "gate_report_id": "phase3_shrink1_dual_strict_gate_test",
        "golden_vector_sha256": "sha256:" + "22" * 32,
        "rust": {
            "source_root": "sha256:" + "33" * 32,
            "binary_sha256": "sha256:" + "44" * 32,
        },
    }
    monkeypatch.setattr(
        replay,
        "dual_shrink1_strict_gate_report",
        lambda rust_binary=replay.DEFAULT_RUST_BINARY: gate,
    )
    monkeypatch.setattr(replay, "python_shrink1_capacity_replay", lambda: python)
    monkeypatch.setattr(
        replay,
        "rust_shrink1_capacity_replay",
        lambda rust_binary=replay.DEFAULT_RUST_BINARY: rust,
    )


def test_dual_verdict_rejects_equal_but_incomplete_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = _frozen_shape_report(implementation="python")
    rust = _frozen_shape_report(implementation="rust")
    for report in (python, rust):
        report["accepted_source_count"] = replay.EXPECTED_SHRINK1_SOURCE_COUNT - 1
        report["accepted_unique_count"] = replay.EXPECTED_SHRINK1_SOURCE_COUNT - 1
        report["rejected_count"] = 1
        report["rejection_counts"] = {"REJECT_TEST": 1}
    _patch_dual_inputs(monkeypatch, python, rust)

    report = replay.dual_shrink1_capacity_replay_report()

    assert report["dual_replay_equal"] is True
    assert report["status"] == "INCONCLUSIVE_EXECUTION"
    assert report["accepted_unique_count_le_50000"] is False
    assert report["required_next_action"] == (
        "INVESTIGATE_SHRINK1_DUAL_REPLAY_INVARIANT_FAILURE"
    )
    assert report["complete_claim_allowed"] is False


def test_dual_verdict_exposes_any_out_of_budget_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = _frozen_shape_report(implementation="python")
    rust = _frozen_shape_report(implementation="rust")
    for report in (python, rust):
        report["first_out_of_budget_ordinal"] = 50_001
        report["first_out_of_budget_cbor_hex"] = "00"
        report["first_out_of_budget_ast_hash"] = "sha256:" + "55" * 32
    _patch_dual_inputs(monkeypatch, python, rust)

    report = replay.dual_shrink1_capacity_replay_report()

    assert report["status"] == "INCONCLUSIVE_EXECUTION"
    assert report["first_out_of_budget_witness"] == {
        "python": {
            "first_out_of_budget_ordinal": 50_001,
            "first_out_of_budget_cbor_hex": "00",
            "first_out_of_budget_ast_hash": "sha256:" + "55" * 32,
        },
        "rust": {
            "first_out_of_budget_ordinal": 50_001,
            "first_out_of_budget_cbor_hex": "00",
            "first_out_of_budget_ast_hash": "sha256:" + "55" * 32,
        },
    }


def test_dual_verdict_rejects_equal_tampered_commitments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = _frozen_shape_report(implementation="python")
    rust = _frozen_shape_report(implementation="rust")
    tampered_commitment = "sha256:" + "aa" * 32
    for replay_report in (python, rust):
        replay_report["accepted_set_commitment"] = tampered_commitment
    _patch_dual_inputs(monkeypatch, python, rust)

    report = replay.dual_shrink1_capacity_replay_report()

    assert report["dual_replay_equal"] is True
    assert report["status"] == "INCONCLUSIVE_EXECUTION"
    assert report["accepted_unique_count_le_50000"] is False
    assert report["required_next_action"] == (
        "INVESTIGATE_SHRINK1_DUAL_REPLAY_INVARIANT_FAILURE"
    )
    assert report["complete_claim_allowed"] is False
