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
        "hegel-phase3a-q1-full-node6-campaign-contract/6"
    )
    assert value["contract_id"] == (
        "hegel-phase3a-q1-full-node6-campaign-successor-v6"
    )
    assert campaign.DOMAINS["authorization"] == (
        b"HEGEL/Q1/FULL_NODE6/AUTHORIZATION/SUCCESSOR_V6/V1\x00"
    )
    assert campaign.DOMAINS["contract"] == (
        b"HEGEL/Q1/FULL_NODE6/CONTRACT/SUCCESSOR_V6/V1\x00"
    )
    assert value["contract_digest"] == (
        "sha256:1addc9ac8fe99ef4bed62146d41668911ff04a002b5a421389e1f4eb041c67ac"
    )
    assert value["authorization"] == campaign.OWNER_AUTHORIZATION
    assert value["authorization"]["authorization_digest"] == (
        "sha256:6da8ac001e4213960daf5733b64e081834589063a715f7d63cef5d33a7a1ced1"
    )
    assert "authorization_state" not in value["authorization"]
    assert campaign.OWNER_AUTHORIZATION_TEXT == (
        "授权 Q1 full-node6 capacity-preflight diagnostic "
        "successor-v6 distinct-outer actual；确认 successor-v5 known terminal "
        "failure、terminal no-resume，且 Python Odd/Sink 与 Rust Odd/Sink "
        "measurement 各一次 conservatively count，trusted host replay "
        "一次 conservatively count；另行授权 exact v6 capsule-only additional "
        "fresh remeasurement；禁止恢复 successor-v5 或复用 successor-v5 work product"
    )
    assert campaign.LEDGER_SCHEMA_VERSION == (
        "hegel-phase3a-q1-full-node6-ledger-record/3"
    )
    assert value["authorization"]["superseded_v5_resume_allowed"] is False
    assert value["authorization"]["future_exact_user_authorization_required"] is True
    assert value["authorization"]["prior_python_odd_measurement_count"] == 1
    assert value["authorization"]["prior_python_sink_measurement_count"] == 1
    assert value["authorization"]["prior_rust_odd_measurement_count"] == 1
    assert value["authorization"]["prior_rust_sink_measurement_count"] == 1
    assert value["authorization"]["prior_trusted_host_replay_measurement_count"] == 1
    assert value["authorization"]["new_measurement_budget_scope"] == "V6_CAPSULE_ONLY"
    assert value["authorization"]["new_measurement_is_explicitly_additional"] is True
    assert value["authorization"]["v5_work_product_reuse_allowed"] is False
    assert value["source_lineage"]["required_ancestor_commit"] == (
        "33dc9cbb7d753f83650169d6d6b9caa508927764"
    )
    assert value["source_lineage"]["required_parent_commit"] == (
        "3086adf389b8d9fa1a93c0c36fd300ed8bca19db"
    )
    assert value["source_lineage"]["exact_additions_and_modifications_only"] is True
    assert "docs/Hegel_Machine_Current_State_Reconciliation_v6.md" in value[
        "source_lineage"
    ]["additive_paths_in_exact_order"]
    assert "docs/Hegel_Machine_Current_State_Reconciliation_v7.md" in value[
        "source_lineage"
    ]["additive_paths_in_exact_order"]
    assert value["source_lineage"]["modified_paths_in_exact_order"] == [
        "src/hegel_machine/phase3_q1_capacity_preflight_v1.py",
        "src/hegel_machine/phase3_q1_partition_snapshot_v1.py",
        "tests/test_phase3_q1_capacity_preflight_v1.py",
        "tests/test_phase3_q1_partition_snapshot_v1.py",
    ]
    assert value["superseded_campaign"]["capsule_digest"] == (
        "sha256:6819416eed0a41573f21b4280a6a39e97481b87f8dd1c9e0054529e478bc7d9d"
    )
    assert value["superseded_campaign"]["measurement_outcome"] == "KNOWN_TERMINAL_FAILURE"
    assert value["superseded_campaign"]["python_endpoint_start_count"] == 1
    assert value["superseded_campaign"]["rust_endpoint_start_count"] == 1
    assert value["superseded_campaign"]["trusted_host_replay_start_count"] == 1
    assert value["superseded_campaign"]["campaign_execution_count"] == 1
    assert value["superseded_campaign"]["authorization_consumption_count"] == 1
    assert value["superseded_campaign"]["supervisor_popen_count"] == 1
    assert value["superseded_campaign"]["supervisor_return_code"] == 1
    assert value["superseded_campaign"]["endpoint_results"]["python"][
        "stdout_summary"
    ]["terminal_status"] == campaign.SATURATED
    assert value["superseded_campaign"]["endpoint_results"]["rust"][
        "stdout_summary"
    ]["terminal_status"] == campaign.SATURATED
    assert value["superseded_campaign"]["trusted_host_replay_failure"][
        "stdout_object"
    ]["error_detail"] == "RUST_ENDPOINT actual scratch accounting differs"
    assert value["superseded_campaign"]["resource_evidence"]["aggregate"][
        "measurement_intervals_overlap"
    ] is False
    assert value["superseded_campaign"]["scratch_cleanup"][
        "terminal_absence_proved"
    ] is True
    assert value["superseded_campaign"]["postconsumption_cleanup"]["status"] == "PASS"
    assert value["superseded_campaign"]["q1_output_slots"] == [None] * 8
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
    assert value["execution_policy"][
        "postconsumption_cleanup_receipt_required_on_all_child_return_paths"
    ] is True
    assert value["execution_policy"]["postconsumption_cleanup_required_components"] == [
        "SUPERVISOR_PROCESS_GROUP_ABSENCE",
        "EXACT_PRODUCTION_CONTAINER_ABSENCE",
        "EXACT_CONTROL_CONTAINER_ABSENCE",
        "PROBE_ROOT_ABSENCE",
        "EXACT_PRODUCTION_SCRATCH_ABSENCE",
    ]
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
        assert "q1-full-node6-capacity-v6" in fixed[name] or (
            name == "artifact_path"
            and fixed[name].endswith("phase3_q1_full_node6_capacity_preflight_v6.json")
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
        "hegel-phase3a-q1-full-node6-ledger-record/3"
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
