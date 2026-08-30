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
    assert value["contract_digest"] == (
        "sha256:d4de15401d4b5c71b245d2754f11c4d02b845782d71d1526869b2305c778a070"
    )
    assert value["authorization"] == campaign.OWNER_AUTHORIZATION
    assert campaign.OWNER_AUTHORIZATION_TEXT == (
        "授权 Q1 full-node6 capacity-preflight diagnostic successor-v2 actual"
    )
    assert value["source_lineage"]["required_ancestor_commit"] == (
        "33dc9cbb7d753f83650169d6d6b9caa508927764"
    )
    assert value["source_lineage"]["required_parent_commit"] == (
        "473b39da5b95670a74f347b4e6933e507c4e7c57"
    )
    assert value["superseded_campaign"]["capsule_digest"] == (
        "sha256:85fc53352b957995162254d45bf32c75cf1f1444a01f2676cce1e773ea357c90"
    )
    assert value["superseded_campaign"]["measurement_endpoint_start_count"] == 0
    assert value["preconsumption_gate_order"][3:8] == [
        "PREDECESSOR_TERMINAL_AND_ARTIFACT",
        "SUPERSEDED_CAMPAIGN_TERMINAL",
        "TECHNICAL_CONTRACT_AND_FROZEN_SEMANTICS",
        "EXACT_COMMIT_SNAPSHOT_MATERIALIZATION",
        "PYTHON_RUST_OFFLINE_DEPENDENCIES",
    ]


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
