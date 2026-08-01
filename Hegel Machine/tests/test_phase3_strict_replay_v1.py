from __future__ import annotations

import json
from pathlib import Path

import pytest

from hegel_machine.phase3_strict_replay_v1 import (
    DEFAULT_RUST_BINARY,
    PYTHON_CAPACITY_SOURCES,
    dual_capacity_replay_report,
    dual_strict_gate_report,
    python_capacity_replay,
    python_golden_vector_report,
)


ROOT = Path(__file__).resolve().parents[1]


def test_capacity_source_root_binds_pure_generator_without_report_id_cycle() -> None:
    relative = {path.relative_to(ROOT).as_posix() for path in PYTHON_CAPACITY_SOURCES}
    assert "src/hegel_machine/phase3_capacity_witness_v1.py" in relative
    assert "src/hegel_machine/phase3_closure_preflight.py" not in relative
    generator = (
        ROOT / "src" / "hegel_machine" / "phase3_capacity_witness_v1.py"
    ).read_text(encoding="utf-8")
    assert "phase3_dual_strict_gate_" not in generator
    assert "phase3_dual_strict_capacity_replay_" not in generator


def test_python_strict_golden_replay_passes_all_shared_vectors() -> None:
    report = python_golden_vector_report()
    assert report["vector_count"] == 48
    assert report["passed_count"] == 48
    assert report["failed_count"] == 0
    assert report["accepted_result_count"] == 36
    assert report["rejected_result_count"] == 12
    assert report["all_expectations_match"] is True


def test_python_strict_capacity_replay_crosses_budget_with_frozen_commitment() -> None:
    report = python_capacity_replay()
    assert report["source_candidate_count"] == 64_680
    assert report["accepted_source_count"] == 64_680
    assert report["accepted_unique_count"] == 64_680
    assert report["rejected_count"] == 0
    assert report["rewrite_collapsed_count"] == 0
    assert report["accepted_set_commitment"] == (
        "sha256:c1a02a66a8d6d8f75204cb3daf03ab0b01c2b3b8e486d0ab3d481ee3be43c930"
    )
    assert report["first_out_of_budget_ordinal"] == 50_001
    assert report["first_out_of_budget_ast_hash"] == (
        "sha256:7c7f786c2cc57d31506b3c61d162d175c7f69a2878a089c72c9d053694cba948"
    )


def test_checked_in_gate_artifacts_keep_certificate_boundaries_closed() -> None:
    gate = json.loads(
        (ROOT / "artifacts" / "phase3_dual_strict_gate_v1.json").read_text(
            encoding="utf-8"
        )
    )
    capacity = json.loads(
        (
            ROOT
            / "artifacts"
            / "phase3_dual_strict_capacity_replay_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert gate["status"] == "VERIFIED"
    assert gate["strict_acceptance_implementation_verified"] is True
    assert gate["cross_language_vector_identity_equal"] is True
    assert gate["formal_root_generation_allowed"] is False
    assert gate["outside_certificate_issued"] is False
    assert capacity["dual_replay_equal"] is True
    assert (
        "src/hegel_machine/phase3_capacity_witness_v1.py"
        in capacity["execution_bindings"]["python_capacity_source_paths"]
    )
    assert (
        "src/hegel_machine/phase3_closure_preflight.py"
        not in capacity["execution_bindings"]["python_capacity_source_paths"]
    )
    assert capacity["executed_closure_status"] == "DSL_TOO_LARGE"
    assert capacity["dsl_too_large_claim_allowed"] is True
    assert capacity["complete_closure_enumerated"] is False
    assert capacity["formal_archive_roots_generated"] is False
    assert capacity["outside_certificate_issued"] is False
    assert capacity["active_promotion_allowed"] is False


@pytest.mark.skipif(
    not DEFAULT_RUST_BINARY.is_file(),
    reason="compiled Rust release replay is an execution artifact, not checked in",
)
def test_compiled_rust_release_replays_checked_in_gate_artifacts() -> None:
    expected_gate = json.loads(
        (ROOT / "artifacts" / "phase3_dual_strict_gate_v1.json").read_text(
            encoding="utf-8"
        )
    )
    expected_capacity = json.loads(
        (
            ROOT
            / "artifacts"
            / "phase3_dual_strict_capacity_replay_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert dual_strict_gate_report() == expected_gate
    assert dual_capacity_replay_report() == expected_capacity
