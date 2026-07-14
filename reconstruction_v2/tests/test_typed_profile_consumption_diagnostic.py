from __future__ import annotations

from pathlib import Path

from assumption_agent.benchmarks.runtime_profile_injection import (
    RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION,
)
from assumption_agent.benchmarks.paper_protocol import PaperProtocol
from assumption_agent.benchmarks.typed_profile_consumption_diagnostic import (
    PROFILE_CONSUMPTION_DIAGNOSTIC_VERSION,
    _read_preregistration,
    _request_from_row,
)
from assumption_agent.splits import SplitManifest


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "manifests"
    / "skilllearn_typed_profile_consumption_diagnostic_v1.json"
)


def test_preregistration_binds_exact_six_consumed_validation_trials() -> None:
    project_root, preregistration = _read_preregistration(PREREGISTRATION)
    assert project_root == ROOT
    assert preregistration["diagnostic_policy"] == (
        PROFILE_CONSUMPTION_DIAGNOSTIC_VERSION
    )
    assert preregistration["execution_contract"] == {
        "claim_eligible": False,
        "development_validation_consumed_before_preregistration": True,
        "evaluation_mode": "offline_post_agent_verifier",
        "fresh_validation": False,
        "new_policy_off_model_calls": 0,
        "new_policy_on_trial_budget": 6,
        "parallel_workers": 6,
        "promotion_evaluated": False,
        "proposal_or_training_model_calls": 0,
        "retry_allowed": False,
        "sealed_test_bytes_exposed_to_model": False,
        "sealed_test_scoring_performed": False,
        "stored_raw_baseline_reused": True,
        "test_infrastructure_metadata_inspected": True,
        "test_task_input_bytes_inspected": False,
        "test_trial_executed": False,
    }
    rows = preregistration["trial_requests"]
    assert len(rows) == 6
    assert {(row["generation"], row["item_id"]) for row in rows} == {
        (generation, item_id)
        for generation in (1, 2)
        for item_id in {
            "organize-messy-files-3",
            "stock-data-visualization-3",
            "temperature-simulation-1",
        }
    }


def test_frozen_requests_differ_only_by_runtime_profile_delivery() -> None:
    project_root, preregistration = _read_preregistration(PREREGISTRATION)
    protocol = PaperProtocol.read(
        project_root / preregistration["sources"]["paper_protocol"]["path"]
    )
    manifest = SplitManifest.read(
        project_root / preregistration["sources"]["manifest"]["path"]
    )
    for row in preregistration["trial_requests"]:
        old_request, old_source = _request_from_row(
            project_root=project_root,
            preregistration=preregistration,
            protocol=protocol,
            manifest=manifest,
            row=row,
            delivery_enabled=False,
        )
        new_request, new_source = _request_from_row(
            project_root=project_root,
            preregistration=preregistration,
            protocol=protocol,
            manifest=manifest,
            row=row,
            delivery_enabled=True,
        )
        old_payload = old_request.to_dict()
        new_payload = new_request.to_dict()
        assert old_source == new_source
        assert old_request.request_hash == row["old_request_hash"]
        assert new_request.request_hash == row["new_request_hash"]
        assert "portable_capability_delivery_mode" not in old_payload
        assert new_payload.pop("portable_capability_delivery_mode") == (
            RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION
        )
        assert new_payload == old_payload
