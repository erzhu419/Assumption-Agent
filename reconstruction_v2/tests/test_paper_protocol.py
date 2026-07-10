from __future__ import annotations

import copy
from pathlib import Path

from assumption_agent.benchmarks.paper_protocol import PaperProtocol
from assumption_agent.benchmarks.paper_report import (
    PaperTrialRecord,
    build_paper_report,
    render_markdown,
)
from assumption_agent.models import stable_hash


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests" / "skilllearn_paper_protocol_v2.json"
RUOLI_PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_ruoli_gpt54mini.json"
)
MANIFEST_HASH = stable_hash({"manifest": "paper-test"})


def test_paper_protocol_freezes_primary_design() -> None:
    protocol = PaperProtocol.read(PROTOCOL)

    assert protocol.validate_structure() == []
    assert protocol.payload["expected_primary_counts"] == {
        "train": 42,
        "validation": 18,
        "test": 35,
    }
    assert protocol.payload["benchmark_subset"] == {
        "policy": "exclude_external_credentials_by_family_v1",
        "eligible_instance_count": 95,
        "excluded_instance_count": 5,
        "excluded_families": ["github-repo-analytics"],
        "excluded_required_env_names": ["GH_TOKEN"],
    }
    assert protocol.payload["phases"]["sealed_test"]["repeats"] == 3
    assert protocol.payload["statistics"]["analysis_unit"] == "benchmark_item"


def test_v3_protocol_freezes_ruoli_for_every_arm() -> None:
    protocol = PaperProtocol.read(RUOLI_PROTOCOL)

    assert protocol.validate_structure() == []
    assert protocol.payload["model"] == "gpt-5.4-mini"
    assert protocol.payload["proposal_provider_chain"] == ["openai_compatible"]
    assert protocol.payload["trial_provider_mode"] == "openai_compatible"
    assert protocol.payload["provider_endpoint_origin"] == "https://ruoli.dev"
    assert (
        protocol.payload["execution"]["provider_route_policy"]
        == "single_model_single_provider_all_arms_v1"
    )
    assert protocol.payload["execution"]["openai_compatible_codex_config"] == (
        "codex_custom_responses_provider_v1"
    )
    assert protocol.payload["execution"]["counterfactual_replay_policy"] == (
        "behavior_identical_validation_replay_v1"
    )


def test_v3_protocol_rejects_route_drift() -> None:
    protocol = PaperProtocol.read(RUOLI_PROTOCOL)
    mutations = (
        ("model", "gpt-5.3-codex-spark", "paper_model_route_mismatch"),
        ("proposal_provider_chain", ["codex_app_server"], "proposal_provider_route_mismatch"),
        ("trial_provider_mode", "codex_subscription", "trial_provider_route_mismatch"),
        (
            "provider_endpoint_origin",
            "https://other.example",
            "provider_endpoint_route_mismatch",
        ),
    )
    for key, value, expected_issue in mutations:
        payload = copy.deepcopy(protocol.payload)
        payload[key] = value
        assert expected_issue in PaperProtocol(protocol.path, payload).validate_structure()

    payload = copy.deepcopy(protocol.payload)
    del payload["execution"]["provider_route_policy"]
    assert "provider_route_policy_mismatch" in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()

    payload = copy.deepcopy(protocol.payload)
    del payload["execution"]["openai_compatible_codex_config"]
    assert "openai_compatible_codex_config_mismatch" in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()

    payload = copy.deepcopy(protocol.payload)
    del payload["execution"]["counterfactual_replay_policy"]
    assert "counterfactual_replay_policy_mismatch" in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()


def test_paper_report_uses_item_clustered_pairs_and_exact_mcnemar() -> None:
    protocol = PaperProtocol.read(PROTOCOL)
    report = build_paper_report(
        _records(protocol),
        protocol=protocol,
        protocol_lock={
            "claim_eligible": True,
            "lock_hash": "lock-1",
            "primary_manifest_hash": MANIFEST_HASH,
        },
    )

    primary = report["comparisons_vs_raw"]["promoted_v2"]
    assert report["primary_claim_eligible"] is True
    assert primary["complete_paired_item_count"] == 35
    assert primary["majority_gain_count"] == 10
    assert primary["majority_harm_count"] == 0
    assert primary["mean_paired_success_delta"] > 0
    assert primary["mcnemar_p_value"] < 0.01
    assert primary["holm_adjusted_p_value"] < 0.05
    markdown = render_markdown(report)
    assert "promoted_v2" in markdown
    assert "95% clustered CI" in markdown


def test_paper_report_blocks_provider_mismatch() -> None:
    protocol = PaperProtocol.read(PROTOCOL)
    records = list(_records(protocol))
    target = next(
        index
        for index, row in enumerate(records)
        if row.control_id == "promoted_v2" and row.repeat == 1
    )
    row = records[target]
    records[target] = PaperTrialRecord(
        **{
            **row.__dict__,
            "provider_fingerprint": "different-provider",
        }
    )

    report = build_paper_report(
        records,
        protocol=protocol,
        protocol_lock={
            "claim_eligible": True,
            "lock_hash": "lock-1",
            "primary_manifest_hash": MANIFEST_HASH,
        },
    )

    primary = report["comparisons_vs_raw"]["promoted_v2"]
    assert report["primary_claim_eligible"] is False
    assert primary["claim_valid"] is False
    assert "provider_mismatch" in primary["invalid_reasons"]


def test_paper_report_blocks_agent_runtime_mismatch() -> None:
    protocol = PaperProtocol.read(PROTOCOL)
    records = list(_records(protocol))
    target = next(
        index
        for index, row in enumerate(records)
        if row.control_id == "promoted_v2" and row.repeat == 1
    )
    row = records[target]
    records[target] = PaperTrialRecord(
        **{
            **row.__dict__,
            "agent_runtime_key": "b" * 64,
        }
    )

    report = build_paper_report(
        records,
        protocol=protocol,
        protocol_lock={
            "claim_eligible": True,
            "lock_hash": "lock-1",
            "primary_manifest_hash": MANIFEST_HASH,
        },
    )

    primary = report["comparisons_vs_raw"]["promoted_v2"]
    assert report["primary_claim_eligible"] is False
    assert "agent_runtime_mismatch" in primary["invalid_reasons"]


def _records(protocol: PaperProtocol) -> tuple[PaperTrialRecord, ...]:
    controls = [str(row["id"]) for row in protocol.payload["controls"]]
    success_limits = {
        "raw_no_skill": 10,
        "static_generic_v2": 12,
        "v2_no_recursive_repair": 16,
        "promoted_v2": 20,
        "skilllearn_b1_sonnet": 18,
        "human_authored": 25,
    }
    rows: list[PaperTrialRecord] = []
    for item_index in range(35):
        item_hash = stable_hash({"item": item_index})
        for control in controls:
            for repeat in range(1, 4):
                success = item_index < success_limits[control]
                rows.append(
                    PaperTrialRecord(
                        item_id_hash=item_hash,
                        family_hash=stable_hash({"family": item_index % 19}),
                        split="test",
                        control_id=control,
                        protocol_hash=protocol.protocol_hash,
                        manifest_hash=MANIFEST_HASH,
                        evaluator_epoch="skilllearn-eval-paper-test",
                        pair_id=stable_hash(
                            {"item": item_index, "repeat": repeat}
                        )[:20],
                        repeat=repeat,
                        success=success,
                        score=float(success),
                        valid=True,
                        provider_fingerprint="provider-fixed",
                        fairness_fingerprint="budget-fixed",
                        total_tokens=100,
                        steps=10,
                        duration_seconds=1.0,
                        prebuilt_image_key=stable_hash({"image": item_index}),
                        prebuilt_image_id=(
                            "sha256:" + stable_hash({"image_id": item_index})
                        ),
                        agent_runtime_key="a" * 64,
                        agent_runtime_version="codex-cli 0.144.1",
                        observation_hash=stable_hash(
                            {"item": item_index, "control": control, "repeat": repeat}
                        ),
                    )
                )
    return tuple(rows)
