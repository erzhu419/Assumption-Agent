from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..evaluation import (
    PROSPECTIVE_ABSTENTION_PAIRED_GUARD,
    PromotionGateSpec,
)
from ..models import HypothesisProgram, stable_hash
from ..evolution import (
    CANDIDATE_BUNDLE_POLICY_VERSION,
    COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSION,
    COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION,
    COUNTERFACTUAL_REPLAY_POLICY_VERSION,
    PROGRAM_SET_COUNTERFACTUAL_REPLAY_POLICY_VERSION,
    PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
    TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
)
from ..provider_chain import configured_provider_chain, proposal_provider_status
from ..proposer import (
    LEGACY_PROPOSAL_DIVERSITY_POLICY_VERSION,
    PROPOSAL_DIVERSITY_POLICY_VERSION,
    REPAIR_REQUEST_SCOPE_POLICY_VERSION,
    ROOT_PROPOSAL_REPLAY_POLICY_VERSION,
)
from ..secure_env import (
    configured_api_origin,
    configured_model,
    configured_skilllearn_provider_mode,
    load_dotenv,
    map_legacy_model_env,
)
from ..splits import SplitManifest
from .preflight import build_preflight
from .codex_execution_policy import (
    CodexAgentExecutionPolicy,
    LEGACY_CODEX_AGENT_EXECUTION_POLICY,
    codex_agent_execution_policy_for_protocol_version,
    declared_policy_matches,
)
from .docker_egress import (
    DEFAULT_TRIAL_NETWORK_BYTE_LIMIT,
    DEPENDENCY_CACHE_POLICY_VERSION,
    DOCKER_EGRESS_POLICY_VERSION,
    PROVIDER_DNS_POLICY_VERSION,
    TRIAL_NETWORK_BUDGET_POLICY_VERSION,
    DockerEgressPolicy,
    configured_trial_network_byte_limit,
)
from .skilllearn_compiler import (
    LEGACY_SKILL_ACTION_LOWERING_VERSION,
    SKILL_ACTION_LOWERING_VERSION,
    SKILL_FALLBACK_SEMANTICS_VERSION,
    SKILL_ROUTING_VERSION,
)
from .prewarm import development_prewarm_version_for_protocol
from .skilllearn_lifecycle import (
    BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION,
    CODEX_NETWORK_MINIMIZATION_VERSION,
    MODEL_ONLY_TOOL_POLICY_VERSION,
    MODEL_INFERENCE_CONCURRENCY_POLICY_VERSION,
    INVALID_TRIAL_RETRY_POLICY_VERSION,
    LOCAL_EVIDENCE_TRANSPORT_VERSION,
    SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION,
    NETWORK_SCOPE_AUDIT_VERSION,
    OPENAI_COMPATIBLE_CODEX_CONFIG_VERSION,
    PREBUILT_IMAGE_POLICY_VERSION,
    PROPOSAL_FAILURE_ISOLATION_POLICY_VERSION,
    PROVIDER_ROUTE_POLICY_VERSION,
    PROVIDER_FAILURE_POLICY_VERSION,
    RUNNER_AGENT_REGISTRY_ISOLATION_VERSION,
    SHARED_AGENT_RUNTIME_BUILDER_IMAGE,
    SHARED_CODEX_CLI_PACKAGE,
    SHARED_CODEX_CLI_VERSION,
    TRAINING_EVIDENCE_POLICY_VERSION,
    TRAINING_EVIDENCE_REPLAY_POLICY_VERSION,
    TRIAL_TIMEOUT_POLICY_VERSION,
    VERIFIER_EXECUTION_RECEIPT_POLICY_VERSION,
    VERIFIER_ISOLATION_VERSION,
    codex_network_minimization_for_policy,
)
from .offline_verifier import OFFLINE_VERIFIER_POLICY_VERSION
from .skilllearnbench import (
    OFFLINE_READY_SUBSET_POLICY,
    SkillLearnBenchAdapter,
)


PAPER_ROUTES_BY_MAJOR: dict[int, dict[str, Any]] = {
    3: {
        "model": "gpt-5.4-mini",
        "proposal_provider_chain": ["openai_compatible"],
        "trial_provider_mode": "openai_compatible",
        "provider_endpoint_origin": "https://ruoli.dev",
        "provider_endpoint_ipv4s": ["45.78.76.197"],
    },
}

OFFLINE_READINESS_RECEIPT_VERSION = "skilllearn_offline_readiness_receipt_v1"
TRIAL_NETWORK_BYTE_LIMIT_BY_PROTOCOL_VERSION = {
    "3.1.0": DEFAULT_TRIAL_NETWORK_BYTE_LIMIT,
    "3.2.0": 64 * 1024 * 1024,
    "3.3.0": 64 * 1024 * 1024,
    "3.4.0": 64 * 1024 * 1024,
    "3.5.0": 64 * 1024 * 1024,
    "3.6.0": 64 * 1024 * 1024,
    "3.7.0": 64 * 1024 * 1024,
    "3.8.0": 64 * 1024 * 1024,
    "3.9.0": 64 * 1024 * 1024,
    "3.10.0": 64 * 1024 * 1024,
    "3.11.0": 64 * 1024 * 1024,
    "3.12.0": 64 * 1024 * 1024,
    "3.13.0": 64 * 1024 * 1024,
    "3.14.0": 64 * 1024 * 1024,
}

CONTRASTIVE_PROTOCOL_VERSIONS = frozenset(
    {
        "3.6.0",
        "3.7.0",
        "3.8.0",
        "3.9.0",
        "3.10.0",
        "3.11.0",
        "3.12.0",
        "3.13.0",
        "3.14.0",
    }
)
MODEL_SLOT_PROTOCOL_VERSIONS = frozenset(
    {"3.9.0", "3.10.0", "3.11.0", "3.12.0", "3.13.0", "3.14.0"}
)
PROPOSAL_DIVERSITY_PROTOCOL_VERSIONS = frozenset(
    {"3.10.0", "3.11.0", "3.12.0", "3.13.0", "3.14.0"}
)
ACTIONABLE_DIRECTIVE_PROTOCOL_VERSIONS = frozenset(
    {"3.11.0", "3.12.0", "3.13.0", "3.14.0"}
)
REPAIR_REQUEST_SCOPE_PROTOCOL_VERSIONS = frozenset(
    {"3.12.0", "3.13.0", "3.14.0"}
)
CANDIDATE_BUNDLE_PROTOCOL_VERSIONS = frozenset({"3.13.0", "3.14.0"})
FAMILY_SUPPORT_BUNDLE_PROTOCOL_VERSIONS = frozenset({"3.14.0"})
SHARED_BASELINE_ARM_REPLAY_PROTOCOL_VERSIONS = frozenset({"3.14.0"})

CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION = (
    "train_contrastive_precision_then_support_v1"
)
CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION = (
    "valid_train_failures_and_success_controls_v1"
)
ACTIONABLE_CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION = (
    "valid_train_failures_actionable_feedback_and_success_controls_v2"
)
COUNTERFACTUAL_INVALID_EVIDENCE_POLICY_VERSION = (
    "generation_terminal_non_claim_v1"
)


@dataclass(frozen=True)
class PaperProtocol:
    path: Path
    payload: Mapping[str, Any]

    @property
    def id(self) -> str:
        return str(self.payload.get("protocol_id") or "")

    @property
    def protocol_hash(self) -> str:
        return stable_hash(self.payload)

    @property
    def promotion_gate_spec(self) -> PromotionGateSpec:
        promotion = self.payload.get("promotion")
        if not isinstance(promotion, Mapping):
            raise ValueError("paper protocol promotion policy is missing")
        return PromotionGateSpec.from_mapping(promotion)

    @property
    def codex_agent_execution_policy(self) -> CodexAgentExecutionPolicy:
        policy = codex_agent_execution_policy_for_protocol_version(
            self.payload.get("protocol_version")
        )
        if policy is None:
            raise ValueError("paper protocol has no supported Codex execution policy")
        return policy

    @classmethod
    def read(cls, path: str | Path) -> "PaperProtocol":
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("paper protocol must contain one JSON object")
        protocol = cls(path=source, payload=payload)
        issues = protocol.validate_structure()
        if issues:
            raise ValueError(f"invalid paper protocol: {issues}")
        return protocol

    def validate_structure(self) -> list[str]:
        issues: list[str] = []
        protocol_version = str(self.payload.get("protocol_version") or "")
        major = _protocol_major(self.payload.get("protocol_version"))
        if not self.id:
            issues.append("protocol_id_missing")
        if not isinstance(self.payload.get("agent_id"), str) or not str(
            self.payload.get("agent_id") or ""
        ).strip():
            issues.append("agent_id_missing")
        max_steps = self.payload.get("max_steps")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            issues.append("max_steps_invalid")
        readiness_path = self.payload.get("offline_readiness_receipt")
        if not isinstance(readiness_path, str) or not readiness_path.strip():
            issues.append("offline_readiness_receipt_missing")
        if self.payload.get("benchmark") != "skilllearnbench":
            issues.append("benchmark_mismatch")
        route = PAPER_ROUTES_BY_MAJOR.get(major)
        if route is None:
            issues.append("unsupported_protocol_version")
        else:
            if self.payload.get("model") != route["model"]:
                issues.append("paper_model_route_mismatch")
            if list(self.payload.get("proposal_provider_chain") or []) != route[
                "proposal_provider_chain"
            ]:
                issues.append("proposal_provider_route_mismatch")
            if self.payload.get("trial_provider_mode") != route["trial_provider_mode"]:
                issues.append("trial_provider_route_mismatch")
            if route.get("provider_endpoint_origin") and self.payload.get(
                "provider_endpoint_origin"
            ) != route["provider_endpoint_origin"]:
                issues.append("provider_endpoint_route_mismatch")
            if list(self.payload.get("provider_endpoint_ipv4s") or []) != list(
                route.get("provider_endpoint_ipv4s") or []
            ):
                issues.append("provider_endpoint_ipv4_route_mismatch")
        execution = self.payload.get("execution")
        if not isinstance(execution, Mapping):
            issues.append("execution_policy_missing")
        else:
            if execution.get("prebuilt_image_policy") != PREBUILT_IMAGE_POLICY_VERSION:
                issues.append("prebuilt_image_policy_mismatch")
            if (
                major is not None and major >= 2
                and execution.get("runner_agent_registry_isolation")
                != RUNNER_AGENT_REGISTRY_ISOLATION_VERSION
            ):
                issues.append("runner_agent_registry_isolation_mismatch")
            if (
                major is not None
                and major >= 2
                and execution.get("development_prewarm")
                != development_prewarm_version_for_protocol(
                    self.payload.get("protocol_version")
                )
            ):
                issues.append("development_prewarm_mismatch")
            if (
                major is not None and major >= 2
                and execution.get("trial_timeout_policy")
                != TRIAL_TIMEOUT_POLICY_VERSION
            ):
                issues.append("trial_timeout_policy_mismatch")
            if (
                major is not None and major >= 2
                and execution.get("provider_failure_policy")
                != PROVIDER_FAILURE_POLICY_VERSION
            ):
                issues.append("provider_failure_policy_mismatch")
            if (
                major is not None and major >= 2
                and execution.get("training_evidence_policy")
                != TRAINING_EVIDENCE_POLICY_VERSION
            ):
                issues.append("training_evidence_policy_mismatch")
            if execution.get("agent_runtime_builder") != SHARED_AGENT_RUNTIME_BUILDER_IMAGE:
                issues.append("agent_runtime_builder_mismatch")
            if execution.get("agent_runtime_package") != SHARED_CODEX_CLI_PACKAGE:
                issues.append("agent_runtime_package_mismatch")
            if execution.get("agent_runtime_version") != SHARED_CODEX_CLI_VERSION:
                issues.append("agent_runtime_version_mismatch")
            expected_candidate_selection = (
                COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION
                if protocol_version in FAMILY_SUPPORT_BUNDLE_PROTOCOL_VERSIONS
                else (
                    COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSION
                    if protocol_version in CANDIDATE_BUNDLE_PROTOCOL_VERSIONS
                    else (
                        PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION
                        if protocol_version in PROPOSAL_DIVERSITY_PROTOCOL_VERSIONS
                        else (
                            CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION
                            if protocol_version in CONTRASTIVE_PROTOCOL_VERSIONS
                            else TRAIN_ONLY_CANDIDATE_SELECTION_VERSION
                        )
                    )
                )
            )
            if execution.get("proposal_candidate_selection") != (
                expected_candidate_selection
            ):
                issues.append("proposal_candidate_selection_mismatch")
            if protocol_version in CONTRASTIVE_PROTOCOL_VERSIONS:
                expected_contrastive_evidence_policy = (
                    ACTIONABLE_CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
                    if protocol_version in ACTIONABLE_DIRECTIVE_PROTOCOL_VERSIONS
                    else CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION
                )
                if execution.get("contrastive_training_evidence_policy") != (
                    expected_contrastive_evidence_policy
                ):
                    issues.append("contrastive_training_evidence_policy_mismatch")
                if execution.get("counterfactual_invalid_evidence_policy") != (
                    COUNTERFACTUAL_INVALID_EVIDENCE_POLICY_VERSION
                ):
                    issues.append("counterfactual_invalid_evidence_policy_mismatch")
            else:
                for field in (
                    "contrastive_training_evidence_policy",
                    "counterfactual_invalid_evidence_policy",
                ):
                    if field in execution:
                        issues.append(f"{field}_unexpected")
            if protocol_version in PROPOSAL_DIVERSITY_PROTOCOL_VERSIONS:
                expected_proposal_diversity_policy = (
                    PROPOSAL_DIVERSITY_POLICY_VERSION
                    if protocol_version in ACTIONABLE_DIRECTIVE_PROTOCOL_VERSIONS
                    else LEGACY_PROPOSAL_DIVERSITY_POLICY_VERSION
                )
                if execution.get("proposal_diversity_policy") != (
                    expected_proposal_diversity_policy
                ):
                    issues.append("proposal_diversity_policy_mismatch")
                if execution.get("proposal_response_max_tokens") != 8000:
                    issues.append("proposal_response_max_tokens_mismatch")
            else:
                for field in (
                    "proposal_diversity_policy",
                    "proposal_response_max_tokens",
                ):
                    if field in execution:
                        issues.append(f"{field}_unexpected")
            if protocol_version in REPAIR_REQUEST_SCOPE_PROTOCOL_VERSIONS:
                if execution.get("repair_request_scope_policy") != (
                    REPAIR_REQUEST_SCOPE_POLICY_VERSION
                ):
                    issues.append("repair_request_scope_policy_mismatch")
            elif "repair_request_scope_policy" in execution:
                issues.append("repair_request_scope_policy_unexpected")
            if protocol_version in CANDIDATE_BUNDLE_PROTOCOL_VERSIONS:
                if execution.get("candidate_bundle_policy") != (
                    CANDIDATE_BUNDLE_POLICY_VERSION
                ):
                    issues.append("candidate_bundle_policy_mismatch")
            elif "candidate_bundle_policy" in execution:
                issues.append("candidate_bundle_policy_unexpected")
            if execution.get("runtime_candidate_kinds") != ["task", "policy"]:
                issues.append("runtime_candidate_kinds_mismatch")
            if (
                execution.get("evaluator_hypothesis_mode")
                != "separate_epoch_challenger_not_in_primary_runtime"
            ):
                issues.append("evaluator_hypothesis_mode_mismatch")
            if execution.get("skill_routing") != SKILL_ROUTING_VERSION:
                issues.append("skill_routing_mismatch")
            expected_skill_action_lowering = (
                SKILL_ACTION_LOWERING_VERSION
                if protocol_version in ACTIONABLE_DIRECTIVE_PROTOCOL_VERSIONS
                else LEGACY_SKILL_ACTION_LOWERING_VERSION
            )
            if execution.get("skill_action_lowering") != (
                expected_skill_action_lowering
            ):
                issues.append("skill_action_lowering_mismatch")
            if (
                execution.get("skill_fallback_semantics")
                != SKILL_FALLBACK_SEMANTICS_VERSION
            ):
                issues.append("skill_fallback_semantics_mismatch")
            if execution.get("verifier_isolation") != VERIFIER_ISOLATION_VERSION:
                issues.append("verifier_isolation_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("verifier_execution_receipt_policy")
                != VERIFIER_EXECUTION_RECEIPT_POLICY_VERSION
            ):
                issues.append("verifier_execution_receipt_policy_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("offline_verifier_policy")
                != OFFLINE_VERIFIER_POLICY_VERSION
            ):
                issues.append("offline_verifier_policy_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("model_only_tool_policy")
                != MODEL_ONLY_TOOL_POLICY_VERSION
            ):
                issues.append("model_only_tool_policy_mismatch")
            if execution.get("parallel_unit") != "benchmark_item":
                issues.append("parallel_unit_invalid")
            if execution.get("within_pair_execution") != "sequential_balanced_order":
                issues.append("within_pair_execution_invalid")
            if protocol_version in MODEL_SLOT_PROTOCOL_VERSIONS:
                if execution.get("model_inference_concurrency_policy") != (
                    MODEL_INFERENCE_CONCURRENCY_POLICY_VERSION
                ):
                    issues.append("model_inference_concurrency_policy_mismatch")
                if execution.get("model_inference_slots") != 1:
                    issues.append("model_inference_slots_mismatch")
            else:
                for field in (
                    "model_inference_concurrency_policy",
                    "model_inference_slots",
                ):
                    if field in execution:
                        issues.append(f"{field}_unexpected")
            if (
                major is not None
                and major >= 3
                and execution.get("provider_route_policy")
                != PROVIDER_ROUTE_POLICY_VERSION
            ):
                issues.append("provider_route_policy_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("openai_compatible_codex_config")
                != OPENAI_COMPATIBLE_CODEX_CONFIG_VERSION
            ):
                issues.append("openai_compatible_codex_config_mismatch")
            expected_counterfactual_replay_policy = (
                PROGRAM_SET_COUNTERFACTUAL_REPLAY_POLICY_VERSION
                if protocol_version in CANDIDATE_BUNDLE_PROTOCOL_VERSIONS
                else COUNTERFACTUAL_REPLAY_POLICY_VERSION
            )
            if (
                major is not None
                and major >= 3
                and execution.get("counterfactual_replay_policy")
                != expected_counterfactual_replay_policy
            ):
                issues.append("counterfactual_replay_policy_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("root_proposal_replay_policy")
                != ROOT_PROPOSAL_REPLAY_POLICY_VERSION
            ):
                issues.append("root_proposal_replay_policy_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("training_evidence_replay_policy")
                != TRAINING_EVIDENCE_REPLAY_POLICY_VERSION
            ):
                issues.append("training_evidence_replay_policy_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("invalid_trial_retry_policy")
                != INVALID_TRIAL_RETRY_POLICY_VERSION
            ):
                issues.append("invalid_trial_retry_policy_mismatch")
            if (
                major is not None
                and major >= 3
                and not 1 <= int(execution.get("invalid_trial_max_attempts") or 0) <= 5
            ):
                issues.append("invalid_trial_max_attempts_invalid")
            if (
                major is not None
                and major >= 3
                and not _is_nonnegative_number(
                    execution.get("invalid_trial_retry_backoff_seconds")
                )
            ):
                issues.append("invalid_trial_retry_backoff_invalid")
            if (
                major is not None
                and major >= 3
                and not 1 <= int(execution.get("invalid_trial_retry_workers") or 0) <= 4
            ):
                issues.append("invalid_trial_retry_workers_invalid")
            if (
                major is not None
                and major >= 3
                and execution.get("local_evidence_transport")
                != LOCAL_EVIDENCE_TRANSPORT_VERSION
            ):
                issues.append("local_evidence_transport_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("network_scope_audit")
                != NETWORK_SCOPE_AUDIT_VERSION
            ):
                issues.append("network_scope_audit_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("proposal_failure_isolation_policy")
                != PROPOSAL_FAILURE_ISOLATION_POLICY_VERSION
            ):
                issues.append("proposal_failure_isolation_policy_mismatch")
            resolved_codex_policy = codex_agent_execution_policy_for_protocol_version(
                self.payload.get("protocol_version")
            )
            if (
                major is not None
                and major >= 3
                and execution.get("codex_network_minimization")
                != (
                    codex_network_minimization_for_policy(
                        resolved_codex_policy
                    )
                    if resolved_codex_policy is not None
                    else CODEX_NETWORK_MINIMIZATION_VERSION
                )
            ):
                issues.append("codex_network_minimization_mismatch")
            declared_codex_policy = execution.get("codex_agent_execution_policy")
            if resolved_codex_policy is None:
                issues.append("codex_agent_execution_policy_protocol_version_unsupported")
            elif str(self.payload.get("protocol_version") or "") in {"3.1.0", "3.2.0"}:
                if declared_codex_policy is not None:
                    issues.append("legacy_codex_agent_execution_policy_must_be_implicit")
            elif not declared_policy_matches(
                resolved_codex_policy,
                declared_codex_policy,
            ):
                issues.append("codex_agent_execution_policy_mismatch")
            expected_baseline_replay_policy = (
                SHARED_BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
                if protocol_version
                in SHARED_BASELINE_ARM_REPLAY_PROTOCOL_VERSIONS
                else BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
            )
            if (
                major is not None
                and major >= 3
                and execution.get("baseline_arm_evidence_replay_policy")
                != expected_baseline_replay_policy
            ):
                issues.append("baseline_arm_evidence_replay_policy_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("container_egress_policy")
                != DOCKER_EGRESS_POLICY_VERSION
            ):
                issues.append("container_egress_policy_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("dependency_cache_policy")
                != DEPENDENCY_CACHE_POLICY_VERSION
            ):
                issues.append("dependency_cache_policy_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("provider_dns_policy")
                != PROVIDER_DNS_POLICY_VERSION
            ):
                issues.append("provider_dns_policy_mismatch")
            if (
                major is not None
                and major >= 3
                and execution.get("trial_network_budget_policy")
                != TRIAL_NETWORK_BUDGET_POLICY_VERSION
            ):
                issues.append("trial_network_budget_policy_mismatch")
            if major is not None and major >= 3:
                expected_network_byte_limit = (
                    TRIAL_NETWORK_BYTE_LIMIT_BY_PROTOCOL_VERSION.get(
                        str(self.payload.get("protocol_version") or "")
                    )
                )
                if expected_network_byte_limit is None:
                    issues.append("trial_network_byte_limit_protocol_version_unsupported")
                elif (
                    execution.get("trial_network_byte_limit")
                    != expected_network_byte_limit
                ):
                    issues.append("trial_network_byte_limit_mismatch")
        evolution = self.payload.get("evolution")
        if not isinstance(evolution, Mapping):
            issues.append("evolution_budget_missing")
        else:
            minimum_trigger_support = evolution.get("minimum_trigger_support")
            if (
                isinstance(minimum_trigger_support, bool)
                or not isinstance(minimum_trigger_support, int)
                or minimum_trigger_support <= 0
            ):
                issues.append("evolution_minimum_trigger_support_invalid")
            if not 1 <= int(evolution.get("max_generations") or 0) <= 10:
                issues.append("evolution_generation_budget_invalid")
            if not 1 <= int(evolution.get("max_consecutive_non_promotions") or 0) <= int(
                evolution.get("max_generations") or 0
            ):
                issues.append("evolution_early_stop_invalid")
            if not 1 <= int(evolution.get("proposal_candidates_per_generation") or 0) <= 10:
                issues.append("evolution_candidate_budget_invalid")
            if (
                protocol_version in PROPOSAL_DIVERSITY_PROTOCOL_VERSIONS
                and evolution.get("proposal_candidates_per_generation") != 3
            ):
                issues.append("proposal_diversity_candidate_count_mismatch")
        phases = self.payload.get("phases")
        if not isinstance(phases, Mapping):
            issues.append("phases_missing")
        else:
            sealed = phases.get("sealed_test")
            if not isinstance(sealed, Mapping) or sealed.get("single_access") is not True:
                issues.append("sealed_test_not_single_access")
            if not isinstance(sealed, Mapping) or int(sealed.get("repeats") or 0) < 1:
                issues.append("sealed_test_repeats_missing")
            for phase_name in (
                "smoke",
                "development",
                "family_out_development",
                "sealed_test",
                "family_out_transfer",
            ):
                phase = phases.get(phase_name)
                workers = phase.get("parallel_workers") if isinstance(phase, Mapping) else None
                if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
                    issues.append(f"phase_parallel_workers_invalid:{phase_name}")
            smoke = phases.get("smoke")
            if isinstance(smoke, Mapping):
                smoke_generations = smoke.get("max_generations")
                smoke_non_promotions = smoke.get(
                    "max_consecutive_non_promotions"
                )
                if (
                    isinstance(smoke_generations, bool)
                    or not isinstance(smoke_generations, int)
                    or smoke_generations <= 0
                ):
                    issues.append("smoke_generation_budget_invalid")
                if (
                    isinstance(smoke_non_promotions, bool)
                    or not isinstance(smoke_non_promotions, int)
                    or not isinstance(smoke_generations, int)
                    or not 1 <= smoke_non_promotions <= smoke_generations
                ):
                    issues.append("smoke_early_stop_invalid")
        controls = self.payload.get("controls")
        if not isinstance(controls, list) or not controls:
            issues.append("controls_missing")
        else:
            control_ids = [str(row.get("id") or "") for row in controls if isinstance(row, Mapping)]
            if len(control_ids) != len(set(control_ids)):
                issues.append("duplicate_control_id")
            for required in ("raw_no_skill", "promoted_v2"):
                if required not in control_ids:
                    issues.append(f"required_control_missing:{required}")
        statistics = self.payload.get("statistics")
        if not isinstance(statistics, Mapping):
            issues.append("statistics_missing")
        elif statistics.get("analysis_unit") != "benchmark_item":
            issues.append("analysis_unit_not_item")
        subset = self.payload.get("benchmark_subset")
        if subset is not None:
            if not isinstance(subset, Mapping):
                issues.append("benchmark_subset_invalid")
            elif subset.get("policy") not in {
                "full_inventory_v1",
                "exclude_external_credentials_by_family_v1",
                OFFLINE_READY_SUBSET_POLICY,
            }:
                issues.append("benchmark_subset_policy_invalid")
        promotion = self.payload.get("promotion")
        if not isinstance(promotion, Mapping):
            issues.append("promotion_policy_missing")
        else:
            required_promotion_fields = {
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
            }
            for field_name in sorted(required_promotion_fields - set(promotion)):
                issues.append(f"promotion_field_missing:{field_name}")
            for field_name in sorted(set(promotion) - required_promotion_fields):
                issues.append(f"promotion_field_unknown:{field_name}")
            if "metric" in promotion and not isinstance(promotion["metric"], str):
                issues.append("promotion_field_type_invalid:metric")
            integer_fields = {"minimum_pairs", "minimum_net_gain_count"}
            numeric_fields = {
                "confidence",
                "minimum_activation_rate",
                "minimum_effect_lower_bound",
                "maximum_harm_rate",
                "maximum_cost_ratio",
            }
            for field_name in sorted(integer_fields & set(promotion)):
                value = promotion[field_name]
                if isinstance(value, bool) or not isinstance(value, int):
                    issues.append(f"promotion_field_type_invalid:{field_name}")
            for field_name in sorted(numeric_fields & set(promotion)):
                value = promotion[field_name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    issues.append(f"promotion_field_type_invalid:{field_name}")
            if not any(issue.startswith("promotion_field_") for issue in issues):
                try:
                    PromotionGateSpec.from_mapping(promotion)
                except (KeyError, TypeError, ValueError, OverflowError):
                    issues.append("promotion_policy_invalid")
            if promotion.get("baseline_safety_policy") != (
                PROSPECTIVE_ABSTENTION_PAIRED_GUARD
            ):
                issues.append("promotion_baseline_safety_policy_mismatch")
            if (
                isinstance(statistics, Mapping)
                and promotion.get("metric") != statistics.get("primary_metric")
            ):
                issues.append("promotion_metric_primary_metric_mismatch")
        return sorted(set(issues))


def _protocol_major(value: Any) -> int | None:
    try:
        return int(str(value).split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def _is_nonnegative_number(value: Any) -> bool:
    try:
        return float(value) >= 0
    except (TypeError, ValueError):
        return False


def _read_offline_readiness_receipt(
    protocol: PaperProtocol,
    *,
    project_root: Path,
    primary: SplitManifest,
    secondary: SplitManifest,
) -> tuple[Mapping[str, Any], str]:
    raw_path = str(protocol.payload.get("offline_readiness_receipt") or "")
    if not raw_path:
        raise ValueError("offline readiness receipt path is missing")
    source = (project_root / raw_path).resolve()
    if source != project_root and project_root not in source.parents:
        raise PermissionError("offline readiness receipt escaped the project root")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("offline readiness receipt must contain one JSON object")

    subset = protocol.payload.get("benchmark_subset")
    if not isinstance(subset, Mapping):
        raise ValueError("offline readiness receipt requires a benchmark subset")
    expected = {
        "receipt_version": OFFLINE_READINESS_RECEIPT_VERSION,
        "primary_manifest": str(protocol.payload["primary_manifest"]),
        "primary_manifest_hash": primary.manifest_hash,
        "primary_counts": _counts(primary),
        "secondary_manifest": str(protocol.payload["secondary_manifest"]),
        "secondary_manifest_hash": secondary.manifest_hash,
        "secondary_counts": _counts(secondary),
        "eligible_instance_count": int(subset["eligible_instance_count"]),
        "eligible_family_count": int(subset["eligible_family_count"]),
        "excluded_instance_count": int(subset["excluded_instance_count"]),
        "probe_split": "train",
        "matrix_blockers": [],
        "manifest_execution_ready": True,
        "matrix_passed": True,
        "all_selected_item_static_preflight_passed": True,
        "model_executed": False,
        "online_evaluator_used": False,
        "sealed_test_semantics_accessed": False,
        "sealed_test_bytes_exposed_to_model": False,
        "raw_content_persisted": False,
        "claim_scope": "offline_runtime_readiness_only_not_task_accuracy",
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise ValueError(
            "offline readiness receipt mismatch: " + ",".join(sorted(mismatches))
        )
    source_matrix_hash = str(payload.get("source_matrix_receipt_hash") or "")
    if len(source_matrix_hash) != 64:
        raise ValueError("offline readiness source matrix receipt hash is invalid")
    profile_count = int(payload.get("profile_count") or 0)
    passed_profile_count = int(payload.get("passed_profile_count") or 0)
    family_count = int(payload.get("train_family_probe_count") or 0)
    passed_family_count = int(payload.get("passed_train_family_probe_count") or 0)
    if profile_count <= 0 or passed_profile_count != profile_count:
        raise ValueError("offline readiness profile matrix is incomplete")
    if family_count <= 0 or passed_family_count != family_count:
        raise ValueError("offline readiness family matrix is incomplete")
    return payload, stable_hash(payload)


def build_protocol_lock(
    protocol: PaperProtocol,
    *,
    project_root: str | Path,
    benchmark_root: str | Path,
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve()
    benchmark = Path(benchmark_root).expanduser().resolve()
    primary_path = project / str(protocol.payload["primary_manifest"])
    secondary_path = project / str(protocol.payload["secondary_manifest"])
    primary = SplitManifest.read(primary_path)
    secondary = SplitManifest.read(secondary_path)
    adapter = SkillLearnBenchAdapter(benchmark)
    inventory = adapter.discover()
    subset_policy = dict(protocol.payload.get("benchmark_subset") or {})
    if subset_policy.get("policy") == OFFLINE_READY_SUBSET_POLICY:
        eligible_inventory = adapter.offline_ready_items()
        subset_summary = adapter.offline_ready_summary()
    elif subset_policy.get("policy") == "exclude_external_credentials_by_family_v1":
        eligible_inventory = adapter.credential_independent_items()
        subset_summary = adapter.credential_independent_summary()
    else:
        eligible_inventory = inventory
        subset_summary = {
            "policy": "full_inventory_v1",
            "eligible_instance_count": len(inventory),
            "excluded_instance_count": 0,
            "excluded_families": [],
            "excluded_required_env_names": [],
            "secret_value_persisted": False,
        }
    eligible_ids = {row.id for row in eligible_inventory}
    primary_ids = {*primary.train_ids, *primary.validation_ids, *primary.test_ids}
    secondary_ids = {*secondary.train_ids, *secondary.validation_ids, *secondary.test_ids}
    issues: list[str] = []
    offline_readiness_receipt_hash: str | None = None
    try:
        _, offline_readiness_receipt_hash = _read_offline_readiness_receipt(
            protocol,
            project_root=project,
            primary=primary,
            secondary=secondary,
        )
    except (KeyError, OSError, TypeError, ValueError, PermissionError) as exc:
        issues.append(f"offline_readiness_receipt_invalid:{type(exc).__name__}")
    if primary_ids != eligible_ids:
        issues.append("primary_manifest_inventory_mismatch")
    if secondary_ids != eligible_ids:
        issues.append("secondary_manifest_inventory_mismatch")
    for key in (
        "policy",
        "eligible_instance_count",
        "eligible_family_count",
        "excluded_instance_count",
        "excluded_families",
        "excluded_item_ids",
        "excluded_required_env_names",
        "offline_blocked_families",
        "offline_blocked_item_ids",
    ):
        if key in subset_policy and subset_policy.get(key) != subset_summary.get(key):
            issues.append(f"benchmark_subset_mismatch:{key}")
    if _counts(primary) != dict(protocol.payload["expected_primary_counts"]):
        issues.append("primary_count_mismatch")
    if _counts(secondary) != dict(protocol.payload["expected_secondary_counts"]):
        issues.append("secondary_count_mismatch")
    if configured_model() != protocol.payload["model"]:
        issues.append("configured_model_mismatch")
    if list(configured_provider_chain()) != list(protocol.payload["proposal_provider_chain"]):
        issues.append("proposal_provider_chain_mismatch")
    trial_provider_mode = configured_skilllearn_provider_mode()
    if trial_provider_mode != protocol.payload["trial_provider_mode"]:
        issues.append("trial_provider_mode_mismatch")
    api_origin = configured_api_origin()
    expected_api_origin = str(protocol.payload.get("provider_endpoint_origin") or "")
    if expected_api_origin and api_origin != expected_api_origin:
        issues.append("configured_provider_endpoint_origin_mismatch")
    expected_api_ipv4s = tuple(
        sorted(str(value) for value in protocol.payload.get("provider_endpoint_ipv4s") or [])
    )
    try:
        egress_policy = DockerEgressPolicy.from_env()
    except (TypeError, ValueError):
        egress_policy = None
        issues.append("configured_container_egress_policy_invalid")
    if egress_policy is not None:
        if egress_policy.endpoint_origin != expected_api_origin:
            issues.append("configured_egress_endpoint_origin_mismatch")
        if egress_policy.allowed_ipv4s != expected_api_ipv4s:
            issues.append("configured_provider_endpoint_ipv4_mismatch")
    try:
        trial_network_byte_limit = configured_trial_network_byte_limit()
    except ValueError:
        trial_network_byte_limit = None
        issues.append("configured_trial_network_byte_limit_invalid")
    if trial_network_byte_limit != protocol.payload["execution"].get(
        "trial_network_byte_limit"
    ):
        issues.append("configured_trial_network_byte_limit_mismatch")
    static_program_path = project / "baselines" / "static_generic_program.json"
    static_program = HypothesisProgram.from_dict(
        json.loads(static_program_path.read_text(encoding="utf-8"))
    )
    static_issues = static_program.validate()
    if static_issues:
        issues.extend(f"static_program:{issue}" for issue in static_issues)
    source_issues = _validate_control_sources(protocol, project)
    issues.extend(source_issues)
    code_fingerprint = _code_fingerprint(project)
    git_state = _git_state(project)
    preflight = build_preflight(
        benchmark,
        trial_provider_mode=trial_provider_mode,
        item_ids=eligible_ids,
    )
    provider_status = proposal_provider_status()
    claim_eligible = not issues and not git_state["scoped_dirty"] and not preflight["blockers"]
    lock = {
        "lock_version": (
            "paper_protocol_lock_v2" if expected_api_origin else "paper_protocol_lock_v1"
        ),
        "protocol_id": protocol.id,
        "protocol_hash": protocol.protocol_hash,
        "primary_manifest_hash": primary.manifest_hash,
        "secondary_manifest_hash": secondary.manifest_hash,
        "primary_counts": _counts(primary),
        "secondary_counts": _counts(secondary),
        "inventory_count": len(inventory),
        "inventory_hash": stable_hash(
            {"item_hashes": sorted(row.id_hash for row in inventory)}
        ),
        "eligible_inventory_count": len(eligible_inventory),
        "eligible_inventory_hash": stable_hash(
            {"item_hashes": sorted(row.id_hash for row in eligible_inventory)}
        ),
        "selected_benchmark_fingerprint": adapter.selected_payload_fingerprint(
            eligible_ids
        ),
        "offline_readiness_receipt_path": str(
            protocol.payload["offline_readiness_receipt"]
        ),
        "offline_readiness_receipt_hash": offline_readiness_receipt_hash,
        "benchmark_subset": subset_summary,
        "model": configured_model(),
        "proposal_provider_chain": list(configured_provider_chain()),
        "trial_provider_mode": trial_provider_mode,
        "provider_endpoint_origin": api_origin or None,
        "container_egress": (
            egress_policy.provenance() if egress_policy is not None else None
        ),
        "trial_network_budget": {
            "policy": TRIAL_NETWORK_BUDGET_POLICY_VERSION,
            "byte_limit": trial_network_byte_limit,
        },
        "provider_status": provider_status,
        "max_steps": int(protocol.payload["max_steps"]),
        "execution": dict(protocol.payload["execution"]),
        "resolved_codex_agent_execution_policy": (
            protocol.codex_agent_execution_policy.to_dict()
        ),
        "resolved_codex_agent_execution_policy_hash": (
            protocol.codex_agent_execution_policy.policy_hash
        ),
        "evolution": dict(protocol.payload["evolution"]),
        "promotion": protocol.promotion_gate_spec.to_dict(),
        "static_program_hash": static_program.payload_hash,
        "code_fingerprint": code_fingerprint,
        "git": git_state,
        "preflight": preflight,
        "validation_issues": sorted(set(issues)),
        "claim_eligible": claim_eligible,
        "test_infrastructure_inspected": bool(
            primary.test_ids or secondary.test_ids
        ),
        "sealed_test_scoring_performed": False,
        "sealed_test_bytes_exposed_to_model": False,
        "sealed_test_content_accessed": False,
        "secret_value_persisted": False,
        "raw_content_persisted": False,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
    }
    lock["lock_hash"] = stable_hash(lock)
    return lock


def validate_protocol_lock_for_execution(
    protocol: PaperProtocol,
    lock: Mapping[str, Any],
    manifest: SplitManifest,
    project_root: str | Path,
    benchmark_root: str | Path | None = None,
) -> str:
    """Validate the one frozen run contract before any model execution."""

    project = Path(project_root).expanduser().resolve()
    if lock.get("claim_eligible") is not True:
        raise PermissionError("execution requires a claim-eligible protocol lock")
    if lock.get("protocol_id") != protocol.id or lock.get(
        "protocol_hash"
    ) != protocol.protocol_hash:
        raise PermissionError("execution protocol lock mismatch")
    declared_hash = str(lock.get("lock_hash") or "")
    calculated_hash = stable_hash(
        {key: value for key, value in lock.items() if key != "lock_hash"}
    )
    if not declared_hash or declared_hash != calculated_hash:
        raise PermissionError("execution protocol lock content hash mismatch")
    if lock.get("promotion") != protocol.promotion_gate_spec.to_dict():
        raise PermissionError("execution promotion contract lock mismatch")
    if lock.get("execution") != dict(protocol.payload["execution"]):
        raise PermissionError("execution policy lock mismatch")
    resolved_policy_fields_present = any(
        key in lock
        for key in (
            "resolved_codex_agent_execution_policy",
            "resolved_codex_agent_execution_policy_hash",
        )
    )
    resolved_policy_fields_required = (
        protocol.codex_agent_execution_policy
        != LEGACY_CODEX_AGENT_EXECUTION_POLICY
    )
    if (resolved_policy_fields_required or resolved_policy_fields_present) and (
        lock.get("resolved_codex_agent_execution_policy")
        != protocol.codex_agent_execution_policy.to_dict()
        or lock.get("resolved_codex_agent_execution_policy_hash")
        != protocol.codex_agent_execution_policy.policy_hash
    ):
        raise PermissionError("resolved Codex agent execution policy lock mismatch")
    if lock.get("evolution") != dict(protocol.payload["evolution"]):
        raise PermissionError("execution evolution budget lock mismatch")
    if lock.get("max_steps") != int(protocol.payload["max_steps"]):
        raise PermissionError("execution step budget lock mismatch")
    if manifest.manifest_hash not in {
        lock.get("primary_manifest_hash"),
        lock.get("secondary_manifest_hash"),
    }:
        raise PermissionError("execution manifest lock mismatch")

    primary = SplitManifest.read(project / str(protocol.payload["primary_manifest"]))
    secondary = SplitManifest.read(project / str(protocol.payload["secondary_manifest"]))
    try:
        _, readiness_hash = _read_offline_readiness_receipt(
            protocol,
            project_root=project,
            primary=primary,
            secondary=secondary,
        )
    except (KeyError, OSError, TypeError, ValueError, PermissionError) as exc:
        raise PermissionError("execution offline readiness receipt is invalid") from exc
    if lock.get("offline_readiness_receipt_path") != str(
        protocol.payload["offline_readiness_receipt"]
    ) or lock.get("offline_readiness_receipt_hash") != readiness_hash:
        raise PermissionError("execution offline readiness receipt lock mismatch")

    try:
        current_model = configured_model(enforce_policy=False)
        current_provider_chain = list(configured_provider_chain())
        current_trial_provider_mode = configured_skilllearn_provider_mode()
        current_api_origin = configured_api_origin()
        current_egress = DockerEgressPolicy.from_env()
        current_trial_network_byte_limit = configured_trial_network_byte_limit()
        current_provider_status = proposal_provider_status()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise PermissionError("execution protocol environment is invalid") from exc
    if current_model != protocol.payload["model"] or lock.get("model") != current_model:
        raise PermissionError("execution model environment mismatch")
    if current_provider_chain != list(protocol.payload["proposal_provider_chain"]) or lock.get(
        "proposal_provider_chain"
    ) != current_provider_chain:
        raise PermissionError("execution proposal provider environment mismatch")
    if (
        current_trial_provider_mode != protocol.payload["trial_provider_mode"]
        or lock.get("trial_provider_mode") != current_trial_provider_mode
    ):
        raise PermissionError("execution trial provider environment mismatch")
    if (
        current_api_origin != protocol.payload.get("provider_endpoint_origin")
        or lock.get("provider_endpoint_origin") != current_api_origin
    ):
        raise PermissionError("execution provider endpoint environment mismatch")
    expected_egress = DockerEgressPolicy.from_values(
        base_url=str(protocol.payload.get("provider_endpoint_origin") or ""),
        allowed_ipv4s=tuple(protocol.payload.get("provider_endpoint_ipv4s") or ()),
    )
    if current_egress != expected_egress or lock.get(
        "container_egress"
    ) != current_egress.provenance():
        raise PermissionError("execution container egress environment mismatch")
    expected_network_budget = {
        "policy": TRIAL_NETWORK_BUDGET_POLICY_VERSION,
        "byte_limit": int(protocol.payload["execution"]["trial_network_byte_limit"]),
    }
    if (
        current_trial_network_byte_limit != expected_network_budget["byte_limit"]
        or lock.get("trial_network_budget") != expected_network_budget
    ):
        raise PermissionError("execution trial network budget environment mismatch")
    if current_provider_status != lock.get("provider_status") or not current_provider_status.get(
        "passed"
    ):
        raise PermissionError("execution proposal provider readiness changed after lock")

    if benchmark_root is not None:
        selected_ids = {
            *manifest.train_ids,
            *manifest.validation_ids,
            *manifest.test_ids,
        }
        current_benchmark = SkillLearnBenchAdapter(
            benchmark_root
        ).selected_payload_fingerprint(selected_ids)
        if lock.get("selected_benchmark_fingerprint") != current_benchmark:
            raise PermissionError("execution benchmark payload changed after protocol lock")
    if lock.get("code_fingerprint") != _code_fingerprint(project):
        raise PermissionError("execution code changed after protocol lock")
    git_state = _git_state(project)
    locked_git = dict(lock.get("git") or {})
    if git_state.get("scoped_dirty") or git_state.get("commit") != locked_git.get(
        "commit"
    ):
        raise PermissionError("execution source tree changed after protocol lock")
    return declared_hash


def _counts(manifest: SplitManifest) -> dict[str, int]:
    return {
        "train": len(manifest.train_ids),
        "validation": len(manifest.validation_ids),
        "test": len(manifest.test_ids),
    }


def _validate_control_sources(
    protocol: PaperProtocol,
    project_root: Path,
) -> list[str]:
    issues: list[str] = []
    dynamic = {"none", "no_recursive_archive_incumbent", "frozen_archive_incumbent"}
    for row in protocol.payload["controls"]:
        if not isinstance(row, Mapping):
            issues.append("malformed_control")
            continue
        source = str(row.get("source") or "")
        if source in dynamic:
            continue
        if not (project_root / source).exists():
            issues.append(f"control_source_missing:{row.get('id')}")
    return issues


def _code_fingerprint(project_root: Path) -> dict[str, Any]:
    roots = (
        project_root / "assumption_agent",
        project_root / "tests",
        project_root / "scripts",
        project_root / "baselines",
    )
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    files.extend(
        path
        for path in (
            project_root / "pyproject.toml",
            project_root / "ARCHITECTURE.md",
            project_root / "BENCHMARK_PROTOCOL.md",
        )
        if path.is_file()
    )
    rows = [
        {
            "path": str(path.relative_to(project_root)),
            "content_hash": stable_hash({"bytes": path.read_bytes().hex()}),
        }
        for path in sorted(set(files))
    ]
    return {
        "file_count": len(rows),
        "tree_hash": stable_hash(rows),
    }


def _git_state(project_root: Path) -> dict[str, Any]:
    repository_root = project_root.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        relative = str(project_root.relative_to(repository_root))
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", relative],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"commit": "", "scoped_dirty": True, "scoped_change_count": -1}
    return {
        "commit": commit,
        "scoped_dirty": bool(status),
        "scoped_change_count": len(status),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and audit the paper experiment protocol.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-claim-eligible", action="store_true")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    map_legacy_model_env()
    protocol = PaperProtocol.read(args.protocol)
    lock = build_protocol_lock(
        protocol,
        project_root=args.project_root,
        benchmark_root=args.benchmark_root,
    )
    _write_json(args.out, lock)
    print(json.dumps(lock, indent=2, sort_keys=True))
    if args.require_claim_eligible and not lock["claim_eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
