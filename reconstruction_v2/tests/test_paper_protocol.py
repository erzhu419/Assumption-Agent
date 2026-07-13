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
    ACTIONABLE_CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION,
    CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION,
    CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION,
    COUNTERFACTUAL_INVALID_EVIDENCE_POLICY_VERSION,
    PaperProtocol,
    validate_protocol_lock_for_execution,
)
from assumption_agent.benchmarks.codex_execution_policy import (
    LEGACY_CODEX_AGENT_EXECUTION_POLICY,
    LOW_REASONING_LOCAL_COMPACTION_POLICY,
    MODEL_ONLY_ACTION_BUDGET_POLICY,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    MODEL_INFERENCE_CONCURRENCY_POLICY_VERSION,
)
from assumption_agent.benchmarks.skilllearn_experiment import _experiment_phase_name
from assumption_agent.benchmarks.paper_report import (
    PaperTrialRecord,
    build_paper_report,
    render_markdown,
)
from assumption_agent.models import stable_hash
from assumption_agent.evolution import (
    PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
)
from assumption_agent.proposer import (
    LEGACY_PROPOSAL_DIVERSITY_POLICY_VERSION,
    PROPOSAL_DIVERSITY_POLICY_VERSION,
    REPAIR_REQUEST_SCOPE_POLICY_VERSION,
)
from assumption_agent.benchmarks.skilllearn_compiler import (
    LEGACY_SKILL_ACTION_LOWERING_VERSION,
    SKILL_ACTION_LOWERING_VERSION,
)
from assumption_agent.splits import SplitManifest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_2_ruoli_gpt54mini.json"
)
RUOLI_PROTOCOL = PROTOCOL
V31_PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_ruoli_gpt54mini.json"
)
V33_PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_3_ruoli_gpt54mini.json"
)
V34_PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_4_ruoli_gpt54mini.json"
)
V35_PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_5_ruoli_gpt54mini.json"
)
V36_PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_6_ruoli_gpt54mini.json"
)
V37_PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_7_ruoli_gpt54mini.json"
)
V38_PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_8_ruoli_gpt54mini.json"
)
V39_PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_9_ruoli_gpt54mini.json"
)
V310_PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_10_ruoli_gpt54mini.json"
)
V311_PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_11_ruoli_gpt54mini.json"
)
V312_PROTOCOL = (
    ROOT / "manifests" / "skilllearn_paper_protocol_v3_12_ruoli_gpt54mini.json"
)
MANIFEST_HASH = stable_hash({"manifest": "paper-test"})


def test_historical_codex_execution_policy_hashes_remain_immutable() -> None:
    assert LEGACY_CODEX_AGENT_EXECUTION_POLICY.policy_hash == (
        "11a53dab8f63a0dec666996eb4b5dafed351c6cc278b5a93717b7649fee0e54c"
    )
    assert LOW_REASONING_LOCAL_COMPACTION_POLICY.policy_hash == (
        "44b1744deaa2604df54d4d66cc4ad0cfaccdb99f20adfb1343e24088b73bab9f"
    )


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
    assert protocol.codex_agent_execution_policy == LEGACY_CODEX_AGENT_EXECUTION_POLICY
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
    assert protocol.codex_agent_execution_policy == LEGACY_CODEX_AGENT_EXECUTION_POLICY


def test_v33_protocol_freezes_low_reasoning_early_local_compaction() -> None:
    protocol = PaperProtocol.read(V33_PROTOCOL)

    assert protocol.payload["protocol_version"] == "3.3.0"
    assert protocol.payload["execution"]["trial_network_byte_limit"] == 64 * 1024 * 1024
    assert protocol.codex_agent_execution_policy == LOW_REASONING_LOCAL_COMPACTION_POLICY
    assert protocol.payload["execution"]["codex_agent_execution_policy"] == {
        "version": "codex_low_reasoning_early_local_compaction_v1",
        "model_reasoning_effort": "low",
        "model_verbosity": "low",
        "model_auto_compact_token_limit": 32768,
        "model_auto_compact_token_limit_scope": "body_after_prefix",
        "tool_output_token_limit": 10000,
        "enable_request_compression": True,
        "remote_compaction_v2": False,
    }
    assert protocol.payload["phases"] == PaperProtocol.read(PROTOCOL).payload["phases"]
    assert protocol.payload["promotion"] == PaperProtocol.read(PROTOCOL).payload["promotion"]

    v32 = copy.deepcopy(PaperProtocol.read(PROTOCOL).payload)
    v33 = copy.deepcopy(protocol.payload)
    for payload in (v32, v33):
        payload.pop("protocol_id")
        payload.pop("protocol_version")
        payload["execution"].pop("codex_agent_execution_policy", None)
    assert v33 == v32


def test_v34_protocol_freezes_model_only_action_budget() -> None:
    protocol = PaperProtocol.read(V34_PROTOCOL)

    assert protocol.validate_structure() == []
    assert protocol.payload["protocol_version"] == "3.4.0"
    assert protocol.payload["max_steps"] == 100
    assert protocol.payload["execution"]["trial_network_byte_limit"] == 64 * 1024 * 1024
    assert protocol.codex_agent_execution_policy == MODEL_ONLY_ACTION_BUDGET_POLICY
    assert protocol.payload["execution"]["codex_network_minimization"] == (
        "model_only_no_remote_tools_v3"
    )
    assert protocol.payload["execution"]["codex_agent_execution_policy"] == (
        MODEL_ONLY_ACTION_BUDGET_POLICY.to_dict()
    )

    v33 = copy.deepcopy(PaperProtocol.read(V33_PROTOCOL).payload)
    v34 = copy.deepcopy(protocol.payload)
    for payload in (v33, v34):
        payload.pop("protocol_id")
        payload.pop("protocol_version")
        payload["execution"].pop("codex_agent_execution_policy")
        payload["execution"].pop("codex_network_minimization")
        payload["execution"].pop("development_prewarm")
    assert v34 == v33


def test_v35_protocol_changes_only_online_parallelism() -> None:
    protocol = PaperProtocol.read(V35_PROTOCOL)

    assert protocol.validate_structure() == []
    assert protocol.payload["protocol_version"] == "3.5.0"
    assert protocol.codex_agent_execution_policy == MODEL_ONLY_ACTION_BUDGET_POLICY
    assert {
        name: phase["parallel_workers"]
        for name, phase in protocol.payload["phases"].items()
    } == {
        "smoke": 1,
        "development": 1,
        "family_out_development": 1,
        "sealed_test": 1,
        "family_out_transfer": 1,
    }

    v34 = copy.deepcopy(PaperProtocol.read(V34_PROTOCOL).payload)
    v35 = copy.deepcopy(protocol.payload)
    for payload in (v34, v35):
        payload.pop("protocol_id")
        payload.pop("protocol_version")
    for phase in v35["phases"].values():
        phase["parallel_workers"] = 4
    assert v35 == v34


def test_v36_protocol_changes_only_contrastive_evidence_contract() -> None:
    protocol = PaperProtocol.read(V36_PROTOCOL)

    assert protocol.validate_structure() == []
    assert protocol.payload["protocol_version"] == "3.6.0"
    assert protocol.codex_agent_execution_policy == MODEL_ONLY_ACTION_BUDGET_POLICY
    assert protocol.payload["execution"]["trial_network_byte_limit"] == (
        64 * 1024 * 1024
    )
    assert protocol.payload["execution"]["development_prewarm"] == (
        "all_manifest_images_and_offline_verifiers_v4"
    )
    assert protocol.payload["execution"]["proposal_candidate_selection"] == (
        CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION
    )
    assert protocol.payload["execution"][
        "contrastive_training_evidence_policy"
    ] == CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
    assert protocol.payload["execution"][
        "counterfactual_invalid_evidence_policy"
    ] == COUNTERFACTUAL_INVALID_EVIDENCE_POLICY_VERSION

    v35 = copy.deepcopy(PaperProtocol.read(V35_PROTOCOL).payload)
    v36 = copy.deepcopy(protocol.payload)
    for payload in (v35, v36):
        payload.pop("protocol_id")
        payload.pop("protocol_version")
    v36["execution"]["proposal_candidate_selection"] = v35["execution"][
        "proposal_candidate_selection"
    ]
    v36["execution"].pop("contrastive_training_evidence_policy")
    v36["execution"].pop("counterfactual_invalid_evidence_policy")
    assert v36 == v35


def test_v37_protocol_changes_only_online_parallelism() -> None:
    protocol = PaperProtocol.read(V37_PROTOCOL)

    assert protocol.validate_structure() == []
    assert protocol.payload["protocol_version"] == "3.7.0"
    assert protocol.codex_agent_execution_policy == MODEL_ONLY_ACTION_BUDGET_POLICY
    assert {
        name: phase["parallel_workers"]
        for name, phase in protocol.payload["phases"].items()
    } == {
        "smoke": 6,
        "development": 6,
        "family_out_development": 6,
        "sealed_test": 6,
        "family_out_transfer": 6,
    }

    v36 = copy.deepcopy(PaperProtocol.read(V36_PROTOCOL).payload)
    v37 = copy.deepcopy(protocol.payload)
    for payload in (v36, v37):
        payload.pop("protocol_id")
        payload.pop("protocol_version")
    for phase in v37["phases"].values():
        phase["parallel_workers"] = 1
    assert v37 == v36


def test_v38_protocol_changes_only_supported_online_parallelism() -> None:
    protocol = PaperProtocol.read(V38_PROTOCOL)

    assert protocol.validate_structure() == []
    assert protocol.payload["protocol_version"] == "3.8.0"
    assert protocol.codex_agent_execution_policy == MODEL_ONLY_ACTION_BUDGET_POLICY
    assert {
        name: phase["parallel_workers"]
        for name, phase in protocol.payload["phases"].items()
    } == {
        "smoke": 2,
        "development": 2,
        "family_out_development": 2,
        "sealed_test": 2,
        "family_out_transfer": 2,
    }

    v37 = copy.deepcopy(PaperProtocol.read(V37_PROTOCOL).payload)
    v38 = copy.deepcopy(protocol.payload)
    for payload in (v37, v38):
        payload.pop("protocol_id")
        payload.pop("protocol_version")
    for phase in v38["phases"].values():
        phase["parallel_workers"] = 6
    assert v38 == v37


def test_v39_protocol_adds_outer_parallelism_with_one_shared_model_slot() -> None:
    protocol = PaperProtocol.read(V39_PROTOCOL)

    assert protocol.validate_structure() == []
    assert protocol.payload["protocol_version"] == "3.9.0"
    assert {
        name: phase["parallel_workers"]
        for name, phase in protocol.payload["phases"].items()
    } == {
        "smoke": 6,
        "development": 6,
        "family_out_development": 6,
        "sealed_test": 6,
        "family_out_transfer": 6,
    }
    assert protocol.payload["execution"][
        "model_inference_concurrency_policy"
    ] == MODEL_INFERENCE_CONCURRENCY_POLICY_VERSION
    assert protocol.payload["execution"]["model_inference_slots"] == 1

    v38 = copy.deepcopy(PaperProtocol.read(V38_PROTOCOL).payload)
    v39 = copy.deepcopy(protocol.payload)
    for payload in (v38, v39):
        payload.pop("protocol_id")
        payload.pop("protocol_version")
    for phase in v39["phases"].values():
        phase["parallel_workers"] = 2
    v39["execution"].pop("model_inference_concurrency_policy")
    v39["execution"].pop("model_inference_slots")
    assert v39 == v38


def test_v310_protocol_changes_only_prospective_diverse_candidate_search() -> None:
    protocol = PaperProtocol.read(V310_PROTOCOL)

    assert protocol.validate_structure() == []
    assert protocol.payload["protocol_version"] == "3.10.0"
    assert protocol.payload["execution"]["proposal_candidate_selection"] == (
        PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION
    )
    assert protocol.payload["execution"]["proposal_diversity_policy"] == (
        LEGACY_PROPOSAL_DIVERSITY_POLICY_VERSION
    )
    assert protocol.payload["execution"]["proposal_response_max_tokens"] == 8000

    v39 = copy.deepcopy(PaperProtocol.read(V39_PROTOCOL).payload)
    v310 = copy.deepcopy(protocol.payload)
    for payload in (v39, v310):
        payload.pop("protocol_id")
        payload.pop("protocol_version")
    v310["execution"]["proposal_candidate_selection"] = v39["execution"][
        "proposal_candidate_selection"
    ]
    v310["execution"].pop("proposal_diversity_policy")
    v310["execution"].pop("proposal_response_max_tokens")
    assert v310 == v39


def test_v311_changes_only_actionable_pre_gate_search_and_lowering() -> None:
    protocol = PaperProtocol.read(V311_PROTOCOL)

    assert protocol.validate_structure() == []
    assert protocol.payload["protocol_version"] == "3.11.0"
    execution = protocol.payload["execution"]
    assert execution["proposal_diversity_policy"] == (
        PROPOSAL_DIVERSITY_POLICY_VERSION
    )
    assert execution["contrastive_training_evidence_policy"] == (
        ACTIONABLE_CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
    )
    assert execution["skill_action_lowering"] == SKILL_ACTION_LOWERING_VERSION

    v310 = copy.deepcopy(PaperProtocol.read(V310_PROTOCOL).payload)
    v311 = copy.deepcopy(protocol.payload)
    for payload in (v310, v311):
        payload.pop("protocol_id")
        payload.pop("protocol_version")
    v311["execution"]["proposal_diversity_policy"] = (
        LEGACY_PROPOSAL_DIVERSITY_POLICY_VERSION
    )
    v311["execution"]["contrastive_training_evidence_policy"] = (
        CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
    )
    v311["execution"]["skill_action_lowering"] = (
        LEGACY_SKILL_ACTION_LOWERING_VERSION
    )
    assert v311 == v310


def test_v312_changes_only_repair_request_scope() -> None:
    protocol = PaperProtocol.read(V312_PROTOCOL)

    assert protocol.validate_structure() == []
    assert protocol.payload["protocol_version"] == "3.12.0"
    assert protocol.payload["execution"]["repair_request_scope_policy"] == (
        REPAIR_REQUEST_SCOPE_POLICY_VERSION
    )

    v311 = copy.deepcopy(PaperProtocol.read(V311_PROTOCOL).payload)
    v312 = copy.deepcopy(protocol.payload)
    for payload in (v311, v312):
        payload.pop("protocol_id")
        payload.pop("protocol_version")
    v312["execution"].pop("repair_request_scope_policy")
    assert v312 == v311


@pytest.mark.parametrize("mutation", ("missing", "drifted"))
def test_v312_rejects_repair_request_scope_drift(mutation: str) -> None:
    protocol = PaperProtocol.read(V312_PROTOCOL)
    payload = copy.deepcopy(protocol.payload)
    if mutation == "missing":
        payload["execution"].pop("repair_request_scope_policy")
    else:
        payload["execution"]["repair_request_scope_policy"] = "drifted"

    assert "repair_request_scope_policy_mismatch" in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()


def test_v311_rejects_v312_only_repair_request_scope() -> None:
    protocol = PaperProtocol.read(V311_PROTOCOL)
    payload = copy.deepcopy(protocol.payload)
    payload["execution"]["repair_request_scope_policy"] = (
        REPAIR_REQUEST_SCOPE_POLICY_VERSION
    )

    assert "repair_request_scope_policy_unexpected" in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()


@pytest.mark.parametrize(
    ("field_name", "drifted_value", "expected_issue"),
    (
        (
            "proposal_diversity_policy",
            LEGACY_PROPOSAL_DIVERSITY_POLICY_VERSION,
            "proposal_diversity_policy_mismatch",
        ),
        (
            "contrastive_training_evidence_policy",
            CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION,
            "contrastive_training_evidence_policy_mismatch",
        ),
        (
            "skill_action_lowering",
            LEGACY_SKILL_ACTION_LOWERING_VERSION,
            "skill_action_lowering_mismatch",
        ),
    ),
)
def test_v311_rejects_actionable_contract_drift(
    field_name: str,
    drifted_value: str,
    expected_issue: str,
) -> None:
    protocol = PaperProtocol.read(V311_PROTOCOL)
    payload = copy.deepcopy(protocol.payload)
    payload["execution"][field_name] = drifted_value

    assert expected_issue in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()


@pytest.mark.parametrize(
    ("field_name", "drifted_value", "expected_issue"),
    (
        (
            "proposal_candidate_selection",
            "drifted",
            "proposal_candidate_selection_mismatch",
        ),
        (
            "proposal_diversity_policy",
            "drifted",
            "proposal_diversity_policy_mismatch",
        ),
        (
            "proposal_response_max_tokens",
            7999,
            "proposal_response_max_tokens_mismatch",
        ),
    ),
)
def test_v310_protocol_rejects_candidate_search_contract_drift(
    field_name: str,
    drifted_value: object,
    expected_issue: str,
) -> None:
    protocol = PaperProtocol.read(V310_PROTOCOL)
    payload = copy.deepcopy(protocol.payload)
    payload["execution"][field_name] = drifted_value

    assert expected_issue in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()


@pytest.mark.parametrize(
    "field_name",
    ("proposal_diversity_policy", "proposal_response_max_tokens"),
)
def test_v39_rejects_v310_only_candidate_search_fields(field_name: str) -> None:
    protocol = PaperProtocol.read(V39_PROTOCOL)
    payload = copy.deepcopy(protocol.payload)
    payload["execution"][field_name] = (
        PROPOSAL_DIVERSITY_POLICY_VERSION
        if field_name.endswith("policy")
        else 8000
    )

    assert f"{field_name}_unexpected" in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()


def test_v310_requires_exact_three_candidate_budget() -> None:
    protocol = PaperProtocol.read(V310_PROTOCOL)
    payload = copy.deepcopy(protocol.payload)
    payload["evolution"]["proposal_candidates_per_generation"] = 2

    assert "proposal_diversity_candidate_count_mismatch" in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()


@pytest.mark.parametrize(
    "protocol_path",
    (V31_PROTOCOL, PROTOCOL, V33_PROTOCOL, V34_PROTOCOL, V35_PROTOCOL),
)
def test_v31_through_v35_keep_historical_candidate_selection(
    protocol_path: Path,
) -> None:
    protocol = PaperProtocol.read(protocol_path)

    assert protocol.payload["execution"]["proposal_candidate_selection"] == (
        "train_static_support_then_complexity_v1"
    )


@pytest.mark.parametrize(
    ("field_name", "expected_issue"),
    (
        (
            "proposal_candidate_selection",
            "proposal_candidate_selection_mismatch",
        ),
        (
            "contrastive_training_evidence_policy",
            "contrastive_training_evidence_policy_mismatch",
        ),
        (
            "counterfactual_invalid_evidence_policy",
            "counterfactual_invalid_evidence_policy_mismatch",
        ),
    ),
)
@pytest.mark.parametrize(
    "protocol_path",
    (
        V36_PROTOCOL,
        V37_PROTOCOL,
        V38_PROTOCOL,
        V39_PROTOCOL,
        V310_PROTOCOL,
        V311_PROTOCOL,
        V312_PROTOCOL,
    ),
)
def test_contrastive_protocol_rejects_contract_drift(
    protocol_path: Path,
    field_name: str,
    expected_issue: str,
) -> None:
    protocol = PaperProtocol.read(protocol_path)
    payload = copy.deepcopy(protocol.payload)
    payload["execution"][field_name] = "drifted"

    assert expected_issue in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()


@pytest.mark.parametrize(
    "field_name",
    (
        "contrastive_training_evidence_policy",
        "counterfactual_invalid_evidence_policy",
    ),
)
def test_v31_through_v35_reject_v36_only_contract_fields(
    field_name: str,
) -> None:
    protocol = PaperProtocol.read(V35_PROTOCOL)
    payload = copy.deepcopy(protocol.payload)
    payload["execution"][field_name] = "hybrid-policy"

    assert f"{field_name}_unexpected" in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()


@pytest.mark.parametrize(
    "field_name",
    ("model_inference_concurrency_policy", "model_inference_slots"),
)
def test_v38_rejects_v39_only_model_slot_fields(field_name: str) -> None:
    protocol = PaperProtocol.read(V38_PROTOCOL)
    payload = copy.deepcopy(protocol.payload)
    payload["execution"][field_name] = (
        MODEL_INFERENCE_CONCURRENCY_POLICY_VERSION
        if field_name.endswith("policy")
        else 1
    )

    assert f"{field_name}_unexpected" in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()


@pytest.mark.parametrize(
    "field_name",
    tuple(MODEL_ONLY_ACTION_BUDGET_POLICY.to_dict()),
)
def test_v34_protocol_rejects_agent_execution_policy_drift(field_name: str) -> None:
    protocol = PaperProtocol.read(V34_PROTOCOL)
    payload = copy.deepcopy(protocol.payload)
    del payload["execution"]["codex_agent_execution_policy"][field_name]

    assert "codex_agent_execution_policy_mismatch" in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()


@pytest.mark.parametrize(
    "field_name",
    tuple(LOW_REASONING_LOCAL_COMPACTION_POLICY.to_dict()),
)
def test_v33_protocol_rejects_agent_execution_policy_drift(field_name: str) -> None:
    protocol = PaperProtocol.read(V33_PROTOCOL)
    payload = copy.deepcopy(protocol.payload)
    del payload["execution"]["codex_agent_execution_policy"][field_name]

    assert "codex_agent_execution_policy_mismatch" in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()


def test_v32_rejects_silent_v33_agent_execution_policy() -> None:
    protocol = PaperProtocol.read(PROTOCOL)
    payload = copy.deepcopy(protocol.payload)
    payload["execution"]["codex_agent_execution_policy"] = (
        LOW_REASONING_LOCAL_COMPACTION_POLICY.to_dict()
    )

    assert "legacy_codex_agent_execution_policy_must_be_implicit" in PaperProtocol(
        protocol.path,
        payload,
    ).validate_structure()


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


@pytest.mark.parametrize(
    "protocol_path",
    (
        V33_PROTOCOL,
        V34_PROTOCOL,
        V35_PROTOCOL,
        V36_PROTOCOL,
        V37_PROTOCOL,
        V38_PROTOCOL,
        V39_PROTOCOL,
        V310_PROTOCOL,
        V311_PROTOCOL,
        V312_PROTOCOL,
    ),
)
def test_versioned_execution_lock_binds_resolved_agent_policy(
    monkeypatch: pytest.MonkeyPatch,
    protocol_path: Path,
) -> None:
    protocol = PaperProtocol.read(protocol_path)
    manifest = SplitManifest.read(ROOT / str(protocol.payload["primary_manifest"]))
    lock = _execution_lock(protocol)
    _patch_execution_environment(monkeypatch, protocol, lock)

    lock["resolved_codex_agent_execution_policy_hash"] = "0" * 64
    lock["lock_hash"] = stable_hash(
        {key: value for key, value in lock.items() if key != "lock_hash"}
    )
    with pytest.raises(PermissionError, match="resolved Codex agent execution policy"):
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


def test_v34_paper_report_binds_action_budget_and_token_completeness() -> None:
    protocol = PaperProtocol.read(V34_PROTOCOL)
    records = list(_records(protocol))
    lock = {
        "claim_eligible": True,
        "lock_hash": "lock-v34",
        "primary_manifest_hash": MANIFEST_HASH,
    }

    report = build_paper_report(records, protocol=protocol, protocol_lock=lock)

    assert report["primary_claim_eligible"] is True
    assert report["step_budget_cost_accounting_policy"] == (
        "uniform_codex_action_start_cost_v1"
    )
    assert report["control_summaries"]["raw_no_skill"][
        "step_budget_token_usage_complete_count"
    ] == 96

    target = records[0]
    records[0] = PaperTrialRecord(
        **{
            **target.__dict__,
            "step_budget_token_usage_complete": False,
        }
    )
    blocked = build_paper_report(records, protocol=protocol, protocol_lock=lock)
    assert "record_step_budget_receipt_mismatch" in blocked["claim_blockers"]


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
    if protocol.codex_agent_execution_policy != LEGACY_CODEX_AGENT_EXECUTION_POLICY:
        lock["resolved_codex_agent_execution_policy"] = (
            protocol.codex_agent_execution_policy.to_dict()
        )
        lock["resolved_codex_agent_execution_policy_hash"] = (
            protocol.codex_agent_execution_policy.policy_hash
        )
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
                        codex_agent_execution_policy_hash=(
                            protocol.codex_agent_execution_policy.policy_hash
                            if protocol.codex_agent_execution_policy
                            != LEGACY_CODEX_AGENT_EXECUTION_POLICY
                            else ""
                        ),
                        step_budget_policy=(
                            str(
                                protocol.codex_agent_execution_policy
                                .action_budget_policy
                            )
                            if protocol.codex_agent_execution_policy
                            .action_budget_enforced
                            else ""
                        ),
                        step_budget_unit=(
                            str(
                                protocol.codex_agent_execution_policy
                                .action_budget_unit
                            )
                            if protocol.codex_agent_execution_policy
                            .action_budget_enforced
                            else ""
                        ),
                        step_budget_limit=(
                            int(protocol.payload["max_steps"])
                            if protocol.codex_agent_execution_policy
                            .action_budget_enforced
                            else 0
                        ),
                        step_budget_token_usage_complete=(
                            protocol.codex_agent_execution_policy
                            .action_budget_enforced
                        ),
                        step_budget_receipt_hash=(
                            "c" * 64
                            if protocol.codex_agent_execution_policy
                            .action_budget_enforced
                            else ""
                        ),
                        observation_hash=stable_hash(
                            {"item": item_index, "control": control, "repeat": repeat}
                        ),
                    )
                )
    return tuple(rows)
