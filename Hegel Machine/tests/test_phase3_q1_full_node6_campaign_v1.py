from __future__ import annotations

import json
from pathlib import Path

from hegel_machine import phase3_q1_full_node6_campaign_v1 as campaign


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "config/phase3_q1_full_node6_campaign_v1.json"


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _capsule() -> dict[str, object]:
    return campaign.build_capsule_v1(
        _contract(),
        source_commit="1" * 40,
        source_tree="2" * 40,
        source_closure_digest="sha256:" + "3" * 64,
        static_gate_receipts={
            "PREDECESSOR": {"digest": "sha256:" + "4" * 64},
            "SOURCE": {"digest": "sha256:" + "5" * 64},
        },
    )


def test_contract_and_authorization_replay_exactly() -> None:
    value = campaign.validate_contract_v1(_contract())
    assert value["schema_version"] == (
        "hegel-phase3a-q1-full-node6-campaign-contract/3"
    )
    assert value["contract_id"] == (
        "hegel-phase3a-q1-full-node6-campaign-successor-v3"
    )
    assert campaign.DOMAINS["authorization"] == (
        b"HEGEL/Q1/FULL_NODE6/AUTHORIZATION/SUCCESSOR_V3/V1\x00"
    )
    assert campaign.DOMAINS["contract"] == (
        b"HEGEL/Q1/FULL_NODE6/CONTRACT/SUCCESSOR_V3/V1\x00"
    )
    assert value["contract_digest"] == (
        "sha256:526150772bf047971c2cd22c8fb08b15a5f0f479f92d399d89777af6c9f82392"
    )
    assert value["authorization"] == campaign.OWNER_AUTHORIZATION
    assert value["authorization"]["authorization_digest"] == (
        "sha256:5d1973633d122d03a8d7877208f19e4baf60a458933d765cd7c104f646d45644"
    )
    assert "authorization_state" not in value["authorization"]
    assert campaign.OWNER_AUTHORIZATION_TEXT == (
        "授权 Q1 full-node6 capacity-preflight diagnostic "
        "successor-v3 distinct-outer actual；确认 successor-v2 measurement outcome "
        "UNKNOWN、terminal no-resume，且 prior unknown conservatively counts as "
        "measurement；另行授权 exact v3 capsule-only additional fresh "
        "remeasurement；禁止复用 successor-v2 work product"
    )
    assert campaign.LEDGER_SCHEMA_VERSION == (
        "hegel-phase3a-q1-full-node6-ledger-record/2"
    )
    assert value["authorization"]["superseded_v2_resume_allowed"] is False
    assert value["authorization"]["future_exact_user_authorization_required"] is True
    assert value["authorization"]["prior_unknown_conservatively_counts_as_measurement"] is True
    assert value["authorization"]["new_measurement_budget_scope"] == "V3_CAPSULE_ONLY"
    assert value["authorization"]["new_measurement_is_explicitly_additional"] is True
    assert value["authorization"]["v2_work_product_reuse_allowed"] is False
    assert value["source_lineage"]["required_ancestor_commit"] == (
        "33dc9cbb7d753f83650169d6d6b9caa508927764"
    )
    assert value["source_lineage"]["required_parent_commit"] == (
        "1321eb3be3c28f823fba04b947bb96ee16d35f6c"
    )
    assert value["superseded_campaign"]["capsule_digest"] == (
        "sha256:8eec9c3afaaea443553cd886c84879f465e0b15f406c7f88a6bfef512867cdde"
    )
    assert value["superseded_campaign"]["measurement_outcome"] == "UNKNOWN"
    assert value["superseded_campaign"]["python_endpoint_start_count"] == 1
    assert value["superseded_campaign"]["rust_endpoint_start_count"] == 0
    assert value["superseded_campaign"]["trusted_host_replay_start_count"] == 0
    assert value["preconsumption_gate_order"][3:8] == [
        "PREDECESSOR_TERMINAL_AND_ARTIFACT",
        "SUPERSEDED_CAMPAIGN_TERMINAL",
        "TECHNICAL_CONTRACT_AND_FROZEN_SEMANTICS",
        "EXACT_COMMIT_SNAPSHOT_MATERIALIZATION",
        "PYTHON_RUST_OFFLINE_DEPENDENCIES",
    ]
    assert value["preconsumption_gate_order"][9:12] == [
        "CPU_TOPOLOGY_AND_EXACT_AFFINITY",
        "RUNTIME_ISOLATION_LIVE_GAUNTLET",
        "MEMORY_AND_STORAGE_RESERVATION",
    ]
    assert value["execution_policy"]["runtime_isolation_live_gauntlet_required_receipt"] == {
        "probe_command": "/bin/true",
        "quotient_engine_invocation_count": 0,
        "signature_selection_count": 0,
        "production_endpoint_start_count": 0,
        "supervisor_popen_count": 0,
    }
    fixed = value["fixed_paths"]
    for name in (
        "campaign_lock_path",
        "ledger_root",
        "evidence_root",
        "trajectory_root",
        "run_root",
        "probe_root",
        "artifact_path",
        "stdout_path",
        "stderr_path",
    ):
        assert "q1-full-node6-capacity-v3" in fixed[name] or (
            name == "artifact_path"
            and fixed[name].endswith("phase3_q1_full_node6_capacity_preflight_v3.json")
        )


def test_capsule_is_deterministic_and_keeps_q1_closed() -> None:
    first = _capsule()
    second = _capsule()
    assert first == second
    assert campaign.validate_capsule_v1(first, _contract()) == first
    assert first["maximum_supervisor_popen_count"] == 1
    assert first["maximum_measurements_per_endpoint_signature"] == 1
    assert first["q1_state"] == "NOT_RUN"
    assert first["q1_gate_count"] == 0
    assert first["q1_gate_mask"] == 0
    assert first["q1_output_slots"] == [None] * 8


def test_pending_is_preconsumption_and_stateless() -> None:
    value = campaign.build_pending_v1(
        _contract(),
        _capsule(),
        phase="MEMORY_AND_STORAGE_RESERVATION",
        condition="INSUFFICIENT_FREE_BYTES",
        detail="live reservation gate is not yet admitted",
    )
    assert value["status"] == campaign.PENDING_NOT_CONSUMED
    assert value["campaign_execution_count"] == 0
    assert value["authorization_consumption_count"] == 0
    assert value["supervisor_popen_count"] == 0
    assert value["retry_transition_count"] == 0
    assert value["q1_state"] == "NOT_RUN"
    assert value["q1_gate_count"] == 0
    assert value["q1_gate_mask"] == 0
    assert value["q1_output_slots"] == [None] * 8


def test_ledger_sequence_and_terminal_decision_are_closed() -> None:
    consumed = campaign.build_ledger_record_v1(
        _contract(),
        _capsule(),
        sequence=1,
        record_type="CONSUMED",
        status=campaign.CONSUMED,
        previous_record_digest_or_null=None,
        body={"supervisor_popen_limit": 1},
    )
    terminal = campaign.build_ledger_record_v1(
        _contract(),
        _capsule(),
        sequence=2,
        record_type="TERMINAL",
        status=campaign.SATURATED,
        previous_record_digest_or_null=consumed["record_digest"],
        body={"supervisor_popen_count": 1},
    )
    assert consumed["schema_version"] == (
        "hegel-phase3a-q1-full-node6-ledger-record/2"
    )
    assert terminal["schema_version"] == campaign.LEDGER_SCHEMA_VERSION
    assert terminal["previous_record_digest_or_null"] == consumed["record_digest"]
    success = campaign.decide_supervisor_return_v1(
        return_code=0,
        terminal_status_or_null=campaign.SATURATED,
        replay_passed=True,
    )
    assert success == campaign.CampaignDecisionV1(
        campaign.SATURATED,
        True,
        False,
    )
    guard = campaign.decide_supervisor_return_v1(
        return_code=0,
        terminal_status_or_null=campaign.GUARD_HIT,
        replay_passed=True,
    )
    assert guard == campaign.CampaignDecisionV1(
        campaign.GUARD_HIT,
        False,
        False,
    )
    malformed_guard = campaign.decide_supervisor_return_v1(
        return_code=1,
        terminal_status_or_null=campaign.GUARD_HIT,
        replay_passed=False,
    )
    assert malformed_guard.status == campaign.FAILED_TERMINAL
    failure = campaign.decide_supervisor_return_v1(
        return_code=1,
        terminal_status_or_null=None,
        replay_passed=False,
    )
    assert failure.status == campaign.FAILED_TERMINAL
    assert failure.retry_or_resume_allowed is False
    infeasible = campaign.decide_supervisor_return_v1(
        return_code=3,
        terminal_status_or_null=campaign.RESOURCE_INFEASIBLE,
        replay_passed=True,
    )
    assert infeasible == campaign.CampaignDecisionV1(
        campaign.RESOURCE_INFEASIBLE,
        False,
        False,
    )
