from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from assumption_agent.benchmarks import paper_protocol as paper_protocol_module
from assumption_agent.benchmarks.docker_egress import (
    DockerEgressPolicy,
    TRIAL_NETWORK_BUDGET_POLICY_VERSION,
)
from assumption_agent.benchmarks.paper_protocol import (
    PaperProtocol,
    validate_protocol_lock_for_execution,
)
from assumption_agent.benchmarks.skilllearn_experiment import _experiment_phase_name
from assumption_agent.benchmarks.paper_report import (
    PaperTrialRecord,
    build_paper_report,
    render_markdown,
)
from assumption_agent.models import stable_hash
from assumption_agent.splits import SplitManifest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_2_ruoli_gpt54mini.json"
)
RUOLI_PROTOCOL = PROTOCOL
V31_PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_ruoli_gpt54mini.json"
)
MANIFEST_HASH = stable_hash({"manifest": "paper-test"})


def test_paper_protocol_freezes_primary_design() -> None:
    protocol = PaperProtocol.read(PROTOCOL)

    assert protocol.validate_structure() == []
    assert protocol.payload["expected_primary_counts"] == {
        "train": 38,
        "validation": 16,
        "test": 32,
    }
    assert protocol.payload["benchmark_subset"] == {
        "policy": "exclude_external_credentials_and_offline_blockers_v1",
        "eligible_instance_count": 86,
        "eligible_family_count": 16,
        "excluded_instance_count": 14,
        "excluded_families": [
            "fix-security-bug",
            "github-repo-analytics",
            "nlp-paper-reproduction",
            "python-scala-translation",
        ],
        "excluded_item_ids": ["weighted-gdp-calculation-2"],
        "excluded_required_env_names": ["GH_TOKEN"],
        "offline_blocked_families": [
            "fix-security-bug",
            "nlp-paper-reproduction",
            "python-scala-translation",
        ],
        "offline_blocked_item_ids": ["weighted-gdp-calculation-2"],
    }
    assert protocol.payload["phases"]["sealed_test"]["repeats"] == 3
    assert protocol.payload["statistics"]["analysis_unit"] == "benchmark_item"
    assert protocol.payload["offline_readiness_receipt"] == (
        "manifests/skilllearn_offline_readiness_receipt_v1.json"
    )


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
    assert protocol.payload["execution"]["trial_network_byte_limit"] == 64 * 1024 * 1024
    assert protocol.payload["protocol_version"] == "3.2.0"
    assert protocol.promotion_gate_spec.to_dict() == protocol.payload["promotion"]
    assert protocol.payload["evolution"]["minimum_trigger_support"] == 2
    assert protocol.payload["phases"]["smoke"]["parallel_workers"] == 4
    assert protocol.payload["phases"]["smoke"]["max_generations"] == 1
    assert (
        protocol.payload["phases"]["smoke"]["max_consecutive_non_promotions"]
        == 1
    )


def test_v31_protocol_remains_valid_as_historical_evidence() -> None:
    protocol = PaperProtocol.read(V31_PROTOCOL)

    assert protocol.payload["protocol_version"] == "3.1.0"
    assert protocol.payload["execution"]["trial_network_byte_limit"] == 32 * 1024 * 1024


@pytest.mark.parametrize(
    "field_name",
    (
        "metric",
        "minimum_pairs",
        "confidence",
        "minimum_net_gain_count",
        "minimum_activation_rate",
        "minimum_effect_lower_bound",
        "maximum_harm_rate",
        "maximum_cost_ratio",
        "baseline_safety_policy",
        "candidate_threshold_policy",
    ),
)
def test_v3_protocol_requires_complete_promotion_contract(field_name: str) -> None:
    protocol = PaperProtocol.read(RUOLI_PROTOCOL)
    payload = copy.deepcopy(protocol.payload)
    del payload["promotion"][field_name]

    assert f"promotion_field_missing:{field_name}" in PaperProtocol(
        protocol.path, payload
    ).validate_structure()


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("metric", ""),
        ("minimum_pairs", 0),
        ("confidence", 1.0),
        ("minimum_net_gain_count", -1),
        ("minimum_activation_rate", 1.1),
        ("minimum_effect_lower_bound", -1.1),
        ("maximum_harm_rate", 1.1),
        ("maximum_cost_ratio", 0.9),
        ("baseline_safety_policy", "declared_string_is_observed_fallback"),
        ("candidate_threshold_policy", "candidate_controls_thresholds"),
    ),
)
def test_v3_protocol_rejects_invalid_promotion_contract(
    field_name: str,
    invalid_value: object,
) -> None:
    protocol = PaperProtocol.read(RUOLI_PROTOCOL)
    payload = copy.deepcopy(protocol.payload)
    payload["promotion"][field_name] = invalid_value

    issues = PaperProtocol(protocol.path, payload).validate_structure()
    assert any(
        issue in {
            "promotion_policy_invalid",
            "promotion_baseline_safety_policy_mismatch",
            "promotion_metric_primary_metric_mismatch",
        }
        for issue in issues
    )


def test_v3_protocol_rejects_unknown_promotion_field() -> None:
    protocol = PaperProtocol.read(RUOLI_PROTOCOL)
    payload = copy.deepcopy(protocol.payload)
    payload["promotion"]["minimum_lcb_alias"] = -1.0

    assert "promotion_field_unknown:minimum_lcb_alias" in PaperProtocol(
        protocol.path, payload
    ).validate_structure()


def test_execution_lock_revalidates_frozen_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = PaperProtocol.read(RUOLI_PROTOCOL)
    manifest = SplitManifest.read(ROOT / str(protocol.payload["primary_manifest"]))
    lock = _execution_lock(protocol)
    _patch_execution_environment(monkeypatch, protocol, lock)

    assert validate_protocol_lock_for_execution(
        protocol,
        lock,
        manifest,
        ROOT,
    ) == lock["lock_hash"]

    monkeypatch.setattr(
        paper_protocol_module,
        "configured_trial_network_byte_limit",
        lambda: int(protocol.payload["execution"]["trial_network_byte_limit"]) + 1,
    )
    with pytest.raises(PermissionError, match="trial network budget"):
        validate_protocol_lock_for_execution(protocol, lock, manifest, ROOT)


def test_execution_lock_binds_tracked_offline_readiness_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = PaperProtocol.read(RUOLI_PROTOCOL)
    manifest = SplitManifest.read(ROOT / str(protocol.payload["primary_manifest"]))
    lock = _execution_lock(protocol)
    lock["offline_readiness_receipt_hash"] = "0" * 64
    lock["lock_hash"] = stable_hash(
        {key: value for key, value in lock.items() if key != "lock_hash"}
    )
    _patch_execution_environment(monkeypatch, protocol, lock)

    with pytest.raises(PermissionError, match="offline readiness receipt lock mismatch"):
        validate_protocol_lock_for_execution(protocol, lock, manifest, ROOT)


def test_experiment_phase_owns_parallel_selection() -> None:
    protocol = PaperProtocol.read(RUOLI_PROTOCOL)
    manifest = SplitManifest.read(ROOT / str(protocol.payload["primary_manifest"]))
    smoke = protocol.payload["phases"]["smoke"]

    assert _experiment_phase_name(
        protocol,
        manifest=manifest,
        train_ids=manifest.train_ids,
        validation_ids=manifest.validation_ids,
    ) == "development"
    assert _experiment_phase_name(
        protocol,
        manifest=manifest,
        train_ids=manifest.train_ids[: int(smoke["train_count"])],
        validation_ids=manifest.validation_ids[: int(smoke["validation_count"])],
    ) == "smoke"
    assert _experiment_phase_name(
        protocol,
        manifest=manifest,
        train_ids=manifest.train_ids[:3],
        validation_ids=manifest.validation_ids[:2],
    ) is None


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

    payload = copy.deepcopy(protocol.payload)
    payload["execution"]["trial_network_byte_limit"] = 32 * 1024 * 1024
    assert "trial_network_byte_limit_mismatch" in PaperProtocol(
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
    assert primary["complete_paired_item_count"] == 32
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


def _execution_lock(protocol: PaperProtocol) -> dict[str, object]:
    primary = SplitManifest.read(ROOT / str(protocol.payload["primary_manifest"]))
    secondary = SplitManifest.read(ROOT / str(protocol.payload["secondary_manifest"]))
    readiness = json.loads(
        (ROOT / str(protocol.payload["offline_readiness_receipt"])).read_text(
            encoding="utf-8"
        )
    )
    egress = DockerEgressPolicy.from_values(
        base_url=str(protocol.payload["provider_endpoint_origin"]),
        allowed_ipv4s=tuple(protocol.payload["provider_endpoint_ipv4s"]),
    )
    provider_status = {
        "passed": True,
        "provider_chain_valid": True,
        "requested_providers": list(protocol.payload["proposal_provider_chain"]),
        "ready_providers": list(protocol.payload["proposal_provider_chain"]),
        "openai_compatible_config_present": True,
        "model": protocol.payload["model"],
        "secret_value_persisted": False,
    }
    lock: dict[str, object] = {
        "claim_eligible": True,
        "protocol_id": protocol.id,
        "protocol_hash": protocol.protocol_hash,
        "primary_manifest_hash": primary.manifest_hash,
        "secondary_manifest_hash": secondary.manifest_hash,
        "offline_readiness_receipt_path": protocol.payload[
            "offline_readiness_receipt"
        ],
        "offline_readiness_receipt_hash": stable_hash(readiness),
        "model": protocol.payload["model"],
        "proposal_provider_chain": list(protocol.payload["proposal_provider_chain"]),
        "trial_provider_mode": protocol.payload["trial_provider_mode"],
        "provider_endpoint_origin": protocol.payload["provider_endpoint_origin"],
        "container_egress": egress.provenance(),
        "trial_network_budget": {
            "policy": TRIAL_NETWORK_BUDGET_POLICY_VERSION,
            "byte_limit": protocol.payload["execution"]["trial_network_byte_limit"],
        },
        "provider_status": provider_status,
        "max_steps": protocol.payload["max_steps"],
        "execution": dict(protocol.payload["execution"]),
        "evolution": dict(protocol.payload["evolution"]),
        "promotion": protocol.promotion_gate_spec.to_dict(),
        "code_fingerprint": {"file_count": 1, "tree_hash": "locked"},
        "git": {"commit": "locked-commit", "scoped_dirty": False},
    }
    lock["lock_hash"] = stable_hash(lock)
    return lock


def _patch_execution_environment(
    monkeypatch: pytest.MonkeyPatch,
    protocol: PaperProtocol,
    lock: dict[str, object],
) -> None:
    egress = DockerEgressPolicy.from_values(
        base_url=str(protocol.payload["provider_endpoint_origin"]),
        allowed_ipv4s=tuple(protocol.payload["provider_endpoint_ipv4s"]),
    )
    monkeypatch.setattr(
        paper_protocol_module,
        "configured_model",
        lambda **_: str(protocol.payload["model"]),
    )
    monkeypatch.setattr(
        paper_protocol_module,
        "configured_provider_chain",
        lambda: tuple(protocol.payload["proposal_provider_chain"]),
    )
    monkeypatch.setattr(
        paper_protocol_module,
        "configured_skilllearn_provider_mode",
        lambda: str(protocol.payload["trial_provider_mode"]),
    )
    monkeypatch.setattr(
        paper_protocol_module,
        "configured_api_origin",
        lambda: str(protocol.payload["provider_endpoint_origin"]),
    )
    monkeypatch.setattr(
        paper_protocol_module.DockerEgressPolicy,
        "from_env",
        classmethod(lambda cls: egress),
    )
    monkeypatch.setattr(
        paper_protocol_module,
        "configured_trial_network_byte_limit",
        lambda: int(protocol.payload["execution"]["trial_network_byte_limit"]),
    )
    monkeypatch.setattr(
        paper_protocol_module,
        "proposal_provider_status",
        lambda: dict(lock["provider_status"]),
    )
    monkeypatch.setattr(
        paper_protocol_module,
        "_code_fingerprint",
        lambda _: dict(lock["code_fingerprint"]),
    )
    monkeypatch.setattr(
        paper_protocol_module,
        "_git_state",
        lambda _: dict(lock["git"]),
    )


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
    for item_index in range(32):
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
