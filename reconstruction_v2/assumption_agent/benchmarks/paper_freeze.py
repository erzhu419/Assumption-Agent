from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..evaluation import PairSummary, PromotionGateSpec, promotion_summary_blockers
from ..models import HypothesisProgram, HypothesisStatus, SplitName, stable_hash
from ..splits import SplitManifest
from .paper_controls import ControlSource, control_config_hash, source_tree_hash
from .codex_execution_policy import LEGACY_CODEX_AGENT_EXECUTION_POLICY
from .paper_protocol import (
    COUNTERFACTUAL_INVALID_EVIDENCE_POLICY_VERSION,
    CONTRASTIVE_PROTOCOL_VERSIONS,
    PaperProtocol,
    _code_fingerprint,
    _git_state,
    validate_protocol_lock_for_execution,
)
from .skilllearn_compiler import (
    SkillLearnProgramCompiler,
    skilllearn_program_treatment_hash,
)
from .skilllearnbench import SkillLearnBenchAdapter


_PROMOTION_DECISION_POLICY = "evaluator_owned_paired_validation_v2"
_PROPOSAL_MODEL_FAILURE_CLAIM_BLOCKER = (
    "proposal_model_failure_evidence_present"
)
_INVALID_COUNTERFACTUAL_EVIDENCE_CLAIM_BLOCKER = (
    "invalid_counterfactual_evidence_present"
)
_PROMOTION_THRESHOLD_KEYS = frozenset(
    {
        "minimum_effect_lower_bound",
        "maximum_harm_rate",
        "maximum_cost_ratio",
    }
)
_PROMOTION_DECISION_KEYS = frozenset(
    {
        "allowed",
        "blockers",
        "summary",
        "effect_lower_bound",
        "evaluator_epoch",
        "promotion_contract",
        "candidate_metric",
        "candidate_thresholds",
        "effective_thresholds",
        "policy",
    }
)
_PROMOTION_SUMMARY_LEGACY_KEYS = frozenset(
    {
        "pair_count",
        "baseline_success_count",
        "candidate_success_count",
        "gain_count",
        "harm_count",
        "tie_count",
        "activation_count",
        "selection_change_count",
        "baseline_preserved_count",
        "invalid_pair_count",
        "provider_mismatch_count",
        "budget_mismatch_count",
        "baseline_mean_cost",
        "candidate_mean_cost",
        "cost_ratio",
        "mean_effect",
        "effect_standard_error",
        "effect_lower_bound",
        "harm_rate",
        "activation_rate",
    }
)
_PROMOTION_SUMMARY_LEGACY_COUNT_KEYS = frozenset(
    {
        "pair_count",
        "baseline_success_count",
        "candidate_success_count",
        "gain_count",
        "harm_count",
        "tie_count",
        "activation_count",
        "selection_change_count",
        "baseline_preserved_count",
        "invalid_pair_count",
        "provider_mismatch_count",
        "budget_mismatch_count",
    }
)
_PROMOTION_SUMMARY_DIAGNOSTIC_COUNT_KEYS = frozenset(
    {
        "valid_activation_count",
        "activated_gain_count",
        "activated_harm_count",
        "abstention_count",
    }
)
_PROMOTION_SUMMARY_DIAGNOSTIC_DERIVED_KEYS = frozenset(
    {
        "activation_precision",
        "activation_precision_defined",
        "activated_harm_rate",
        "activated_harm_rate_defined",
        "abstention_rate",
    }
)
_PROMOTION_SUMMARY_V3_6_KEYS = (
    _PROMOTION_SUMMARY_LEGACY_KEYS
    | _PROMOTION_SUMMARY_DIAGNOSTIC_COUNT_KEYS
    | _PROMOTION_SUMMARY_DIAGNOSTIC_DERIVED_KEYS
)
_LEGACY_PROMOTION_SUMMARY_PROTOCOL_VERSIONS = frozenset(
    {"3.1.0", "3.2.0", "3.3.0", "3.4.0", "3.5.0"}
)
_PROMOTION_SUMMARY_BASE_FLOAT_KEYS = frozenset(
    {
        "baseline_mean_cost",
        "candidate_mean_cost",
        "cost_ratio",
        "mean_effect",
        "effect_standard_error",
    }
)


@dataclass(frozen=True)
class FrozenArchive:
    archive_hash: str
    incumbent_id: str | None
    evaluator_epoch: str
    active_programs: tuple[HypothesisProgram, ...]
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_hash": self.archive_hash,
            "incumbent_id": self.incumbent_id,
            "evaluator_epoch": self.evaluator_epoch,
            "active_hypothesis_ids": [row.id for row in self.active_programs],
            "active_program_hashes": [row.payload_hash for row in self.active_programs],
            "content_hash": self.content_hash,
        }


def freeze_paper_workspace(
    *,
    protocol: PaperProtocol,
    protocol_lock: Mapping[str, Any],
    manifest: SplitManifest,
    benchmark_root: str | Path,
    project_root: str | Path,
    recursive_report_path: str | Path,
    recursive_archive_path: str | Path,
    no_recursive_report_path: str | Path,
    no_recursive_archive_path: str | Path,
    controls_output_root: str | Path,
) -> dict[str, Any]:
    project = Path(project_root).resolve()
    benchmark = Path(benchmark_root).resolve()
    _validate_protocol_lock(
        protocol,
        protocol_lock,
        manifest,
        project,
        benchmark,
    )
    recursive_report = _read_mapping(recursive_report_path, "recursive development report")
    no_recursive_report = _read_mapping(
        no_recursive_report_path,
        "no-recursive development report",
    )
    evaluator_epoch = f"skilllearn-eval-{manifest.manifest_hash[:12]}"
    _validate_development_report(
        recursive_report,
        protocol=protocol,
        manifest=manifest,
        recursive_validation_enabled=True,
        protocol_lock_hash=str(protocol_lock.get("lock_hash") or ""),
    )
    _validate_development_report(
        no_recursive_report,
        protocol=protocol,
        manifest=manifest,
        recursive_validation_enabled=False,
        protocol_lock_hash=str(protocol_lock.get("lock_hash") or ""),
    )
    recursive_archive = read_frozen_archive(
        recursive_archive_path,
        expected_evaluator_epoch=evaluator_epoch,
        expected_report=recursive_report,
        promotion_spec=protocol.promotion_gate_spec,
    )
    no_recursive_archive = read_frozen_archive(
        no_recursive_archive_path,
        expected_evaluator_epoch=evaluator_epoch,
        expected_report=no_recursive_report,
        promotion_spec=protocol.promotion_gate_spec,
    )
    require_promoted_recursive_candidate(recursive_archive)
    controls_root = Path(controls_output_root).resolve()
    if controls_root.exists():
        raise FileExistsError("paper control output must not already exist")
    controls_root.mkdir(parents=True)
    adapter = SkillLearnBenchAdapter(benchmark)
    items = adapter.discover()
    static_program = HypothesisProgram.from_dict(
        _read_mapping(project / "baselines" / "static_generic_program.json", "static program")
    )
    if static_program.validate() or static_program.status is not HypothesisStatus.PROMOTED:
        raise ValueError("static paper control is not a valid promoted program")
    archives = {
        "promoted_v2": recursive_archive,
        "v2_no_recursive_repair": no_recursive_archive,
    }
    control_sets: dict[str, Any] = {}
    for split in (SplitName.VALIDATION, SplitName.TEST):
        controls = _compile_control_set(
            protocol=protocol,
            manifest=manifest,
            items=items,
            project_root=project,
            output_root=controls_root / split.value,
            split=split,
            static_program=static_program,
            archives=archives,
        )
        control_sets[split.value] = {
            "controls": [
                {
                    "id": row.id,
                    "root": str(row.root) if row.root else None,
                    "source_hash": source_tree_hash(row.root) if row.root else None,
                }
                for row in sorted(controls, key=lambda value: value.id)
            ],
            "config_hash": control_config_hash(controls),
            "target_item_set_hash": stable_hash(
                {"item_ids": sorted(manifest.ids_for(split))}
            ),
        }
    receipt = {
        "receipt_version": "paper_freeze_receipt_v1",
        "frozen": True,
        "protocol_id": protocol.id,
        "protocol_hash": protocol.protocol_hash,
        "protocol_lock_hash": protocol_lock.get("lock_hash"),
        "codex_agent_execution_policy_hash": (
            protocol.codex_agent_execution_policy.policy_hash
        ),
        "manifest_hash": manifest.manifest_hash,
        "manifest_role": _manifest_role(protocol_lock, manifest),
        "evaluator_epoch": evaluator_epoch,
        "recursive_archive": recursive_archive.to_dict(),
        "no_recursive_archive": no_recursive_archive.to_dict(),
        "recursive_report_hash": _file_content_hash(recursive_report_path),
        "no_recursive_report_hash": _file_content_hash(no_recursive_report_path),
        "control_sets": control_sets,
        "code_fingerprint": protocol_lock.get("code_fingerprint"),
        "git_commit": dict(protocol_lock.get("git") or {}).get("commit"),
        "selected_candidate_available": bool(recursive_archive.active_programs),
        "test_content_accessed": False,
        "secret_value_persisted": False,
        "raw_content_persisted": False,
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    return receipt


def require_promoted_recursive_candidate(archive: FrozenArchive) -> None:
    """Refuse to freeze controls when development produced no incumbent."""
    if archive.incumbent_id is None or not archive.active_programs:
        raise PermissionError(
            "paper freeze requires a promoted recursive candidate"
        )


def read_frozen_archive(
    path: str | Path,
    *,
    expected_evaluator_epoch: str,
    expected_report: Mapping[str, Any],
    promotion_spec: PromotionGateSpec,
) -> FrozenArchive:
    source = Path(path)
    payload = _read_mapping(source, "policy archive")
    hypotheses_payload = payload.get("hypotheses")
    nodes_payload = payload.get("nodes")
    scores_payload = payload.get("score_records")
    if not isinstance(hypotheses_payload, Mapping):
        raise ValueError("archive hypotheses are malformed")
    if not isinstance(nodes_payload, Mapping):
        raise ValueError("archive nodes are malformed")
    if not isinstance(scores_payload, Mapping):
        raise ValueError("archive score records are malformed")
    hypotheses: dict[str, HypothesisProgram] = {}
    for key, row in hypotheses_payload.items():
        if not isinstance(row, Mapping):
            raise ValueError("archive hypothesis row is malformed")
        try:
            program = HypothesisProgram.from_dict(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("archive hypothesis payload is malformed") from exc
        if program.id != key or program.validate():
            raise ValueError("archive hypothesis identity or contract is invalid")
        if program.evaluator_epoch != expected_evaluator_epoch:
            raise ValueError("archive hypothesis evaluator epoch mismatch")
        hypotheses[str(key)] = program
    for key, row in nodes_payload.items():
        if not isinstance(row, Mapping) or row.get("id") != key:
            raise ValueError("archive node identity is invalid")
        active = {str(value) for value in row.get("active_hypothesis_ids", [])}
        if not active <= set(hypotheses):
            raise ValueError("archive node references an unknown hypothesis")
        if row.get("evaluator_epoch_id") != expected_evaluator_epoch:
            raise ValueError("archive node evaluator epoch mismatch")
    incumbent_id = str(payload["incumbent_id"]) if payload.get("incumbent_id") else None
    active_programs: tuple[HypothesisProgram, ...] = ()
    incumbent_rows = [
        str(key) for key, row in nodes_payload.items()
        if isinstance(row, Mapping) and row.get("status") == "incumbent"
    ]
    if incumbent_id is None:
        if incumbent_rows:
            raise ValueError("archive has an incumbent node but no incumbent ID")
    else:
        incumbent = nodes_payload.get(incumbent_id)
        if not isinstance(incumbent, Mapping) or incumbent.get("status") != "incumbent":
            raise ValueError("archive incumbent node is missing or not incumbent")
        if incumbent_rows != [incumbent_id]:
            raise ValueError("archive must have exactly one declared incumbent")
        active_programs = tuple(
            hypotheses[str(hypothesis_id)]
            for hypothesis_id in incumbent.get("active_hypothesis_ids", [])
        )
        if any(row.status is not HypothesisStatus.PROMOTED for row in active_programs):
            raise ValueError("archive incumbent contains a non-promoted hypothesis")
    calculated_hash = stable_hash(
        {
            "hypotheses": {
                key: value.payload_hash for key, value in sorted(hypotheses.items())
            },
            "nodes": {
                str(key): stable_hash(dict(value))
                for key, value in sorted(nodes_payload.items())
                if isinstance(value, Mapping)
            },
            "scores": {
                str(key): dict(value)
                for key, value in sorted(scores_payload.items())
                if isinstance(value, Mapping)
            },
            "incumbent_id": incumbent_id,
        }
    )
    if calculated_hash != payload.get("archive_hash"):
        raise ValueError("archive content hash mismatch")
    if expected_report.get("archive_hash") != calculated_hash:
        raise ValueError("development report and archive hash differ")
    generations = expected_report.get("generations")
    if not isinstance(generations, list) or not generations or any(
        not isinstance(row, Mapping) for row in generations
    ):
        raise ValueError("development report generation history is malformed")
    generation_rows = tuple(generations)
    for row in generation_rows:
        decision = row.get("promotion_decision")
        accepted_id = str(row.get("accepted_hypothesis_id") or "")
        if decision is None:
            if accepted_id:
                raise ValueError(
                    "development accepted hypothesis has no promotion decision"
                )
            continue
        if not isinstance(decision, Mapping) or not accepted_id:
            raise ValueError("development promotion decision identity is malformed")
        program = hypotheses.get(accepted_id)
        if program is None:
            raise ValueError("development decision references an unknown hypothesis")
        if program.expected_effect.metric != promotion_spec.metric:
            raise ValueError("archive candidate metric is not protocol-owned")
        expected_candidate = {
            "minimum_effect_lower_bound": program.expected_effect.minimum_delta,
            "maximum_harm_rate": program.expected_effect.maximum_harm_rate,
            "maximum_cost_ratio": program.expected_effect.maximum_cost_ratio,
        }
        if decision.get("candidate_metric") != program.expected_effect.metric:
            raise ValueError("archive candidate metric and decision differ")
        if decision.get("candidate_thresholds") != expected_candidate:
            raise ValueError("archive candidate thresholds and decision differ")
        if decision.get("effective_thresholds") != promotion_spec.effective_thresholds(
            program
        ):
            raise ValueError("archive effective thresholds and decision differ")
        try:
            archive_treatment_hash = skilllearn_program_treatment_hash(program)
        except ValueError as exc:
            raise ValueError("archive candidate treatment cannot be lowered") from exc
        if row.get("evaluated_candidate_treatment_hash") != archive_treatment_hash:
            raise ValueError(
                "archive candidate treatment and development evidence differ"
            )
        expected_status = (
            HypothesisStatus.PROMOTED
            if decision.get("allowed") is True
            else HypothesisStatus.REJECTED
        )
        if program.status is not expected_status:
            raise ValueError("archive candidate status and promotion decision differ")
    any_promoted = any(bool(row.get("promoted")) for row in generation_rows)
    if any_promoted != bool(active_programs):
        raise ValueError("development promotion history and archive incumbent differ")
    return FrozenArchive(
        archive_hash=calculated_hash,
        incumbent_id=incumbent_id,
        evaluator_epoch=expected_evaluator_epoch,
        active_programs=active_programs,
        content_hash=_file_content_hash(source),
    )


def _compile_control_set(
    *,
    protocol: PaperProtocol,
    manifest: SplitManifest,
    items: Sequence[Any],
    project_root: Path,
    output_root: Path,
    split: SplitName,
    static_program: HypothesisProgram,
    archives: Mapping[str, FrozenArchive],
) -> tuple[ControlSource, ...]:
    compiler = SkillLearnProgramCompiler()
    controls: list[ControlSource] = []
    for control in protocol.payload["controls"]:
        control_id = str(control["id"])
        source = str(control["source"])
        if source == "none":
            controls.append(ControlSource(control_id, None))
            continue
        if source == "baselines/static_generic_program.json":
            programs = (static_program,)
        elif source == "frozen_archive_incumbent":
            programs = archives["promoted_v2"].active_programs
        elif source == "no_recursive_archive_incumbent":
            programs = archives["v2_no_recursive_repair"].active_programs
        else:
            root = (project_root / source).resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"external control source is missing: {control_id}")
            controls.append(ControlSource(control_id, root))
            continue
        result = compiler.compile(
            programs=programs,
            items=items,
            split_manifest=manifest,
            output_root=output_root,
            method_name=control_id,
            allowed_statuses={HypothesisStatus.PROMOTED},
            target_item_ids=manifest.ids_for(split),
            target_split=split.value,
            trace_id=f"paper-freeze:{split.value}:{control_id}",
        )
        controls.append(ControlSource(control_id, result.output_root.resolve()))
    return tuple(controls)


def _validate_protocol_lock(
    protocol: PaperProtocol,
    lock: Mapping[str, Any],
    manifest: SplitManifest,
    project_root: Path,
    benchmark_root: Path | None = None,
) -> None:
    validate_protocol_lock_for_execution(
        protocol,
        lock,
        manifest,
        project_root,
        benchmark_root,
    )


def _validate_development_report(
    report: Mapping[str, Any],
    *,
    protocol: PaperProtocol,
    manifest: SplitManifest,
    recursive_validation_enabled: bool,
    protocol_lock_hash: str | None = None,
) -> None:
    if report.get("mode") != "execute" or report.get("executed") is not True:
        raise ValueError("paper freeze requires an executed development report")
    if report.get("test_content_accessed") is not False:
        raise PermissionError("development report accessed sealed test content")
    preflight = report.get("preflight")
    if not isinstance(preflight, Mapping) or preflight.get("blockers"):
        raise ValueError("development report has preflight blockers")
    plan = report.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError("development report plan is missing")
    secondary = manifest.protocol == "family_out"
    phase_name = "family_out_development" if secondary else "development"
    development = protocol.payload["phases"][phase_name]
    expected = {
        "paper_protocol_id": protocol.id,
        "paper_protocol_hash": protocol.protocol_hash,
        "promotion_contract": protocol.promotion_gate_spec.to_dict(),
        "protocol_lock_hash": protocol_lock_hash,
        "manifest_hash": manifest.manifest_hash,
        "experiment_phase": phase_name,
        "train_count": int(development["train_count"]),
        "validation_count": int(development["validation_count"]),
        "agent_id": protocol.payload["agent_id"],
        "model": protocol.payload["model"],
        "trial_provider_mode": protocol.payload["trial_provider_mode"],
        "max_steps": int(protocol.payload["max_steps"]),
        "parallel_workers": int(development["parallel_workers"]),
        "minimum_trigger_support": int(
            protocol.payload["evolution"]["minimum_trigger_support"]
        ),
        "recursive_validation_enabled": recursive_validation_enabled,
        "max_generations": int(protocol.payload["evolution"]["max_generations"]),
        "max_consecutive_non_promotions": int(
            protocol.payload["evolution"]["max_consecutive_non_promotions"]
        ),
        "proposal_candidates_per_generation": int(
            protocol.payload["evolution"]["proposal_candidates_per_generation"]
        ),
        "test_content_accessed": False,
    }
    if protocol.codex_agent_execution_policy != LEGACY_CODEX_AGENT_EXECUTION_POLICY:
        expected["codex_agent_execution_policy"] = (
            protocol.codex_agent_execution_policy.to_dict()
        )
        expected["codex_agent_execution_policy_hash"] = (
            protocol.codex_agent_execution_policy.policy_hash
        )
    registry_isolation = protocol.payload["execution"].get(
        "runner_agent_registry_isolation"
    )
    if registry_isolation:
        expected["runner_agent_registry_isolation"] = registry_isolation
    prewarm_version = protocol.payload["execution"].get("development_prewarm")
    if prewarm_version:
        expected["development_prewarm_version"] = prewarm_version
        expected["prewarm_passed"] = True
    timeout_policy = protocol.payload["execution"].get("trial_timeout_policy")
    if timeout_policy:
        expected["trial_timeout_policy"] = timeout_policy
    for field in (
        "proposal_candidate_selection",
        "proposal_diversity_policy",
        "proposal_response_max_tokens",
        "repair_request_scope_policy",
        "contrastive_training_evidence_policy",
        "counterfactual_invalid_evidence_policy",
        "provider_failure_policy",
        "provider_route_policy",
        "counterfactual_replay_policy",
        "root_proposal_replay_policy",
        "training_evidence_replay_policy",
        "invalid_trial_retry_policy",
        "invalid_trial_max_attempts",
        "invalid_trial_retry_backoff_seconds",
        "invalid_trial_retry_workers",
        "local_evidence_transport",
        "network_scope_audit",
        "proposal_failure_isolation_policy",
        "openai_compatible_codex_config",
        "codex_network_minimization",
        "model_only_tool_policy",
        "model_inference_concurrency_policy",
        "model_inference_slots",
        "verifier_execution_receipt_policy",
        "offline_verifier_policy",
        "container_egress_policy",
        "dependency_cache_policy",
        "provider_dns_policy",
        "trial_network_budget_policy",
        "trial_network_byte_limit",
        "baseline_arm_evidence_replay_policy",
        "training_evidence_policy",
        "skill_routing",
        "skill_action_lowering",
        "skill_fallback_semantics",
    ):
        value = protocol.payload["execution"].get(field)
        if value:
            expected[field] = value
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"development report plan mismatch: {key}")
    if prewarm_version and not str(plan.get("prewarm_receipt_hash") or ""):
        raise ValueError("development report has no prewarm receipt provenance")
    generation = report.get("generation")
    generations = report.get("generations")
    if not isinstance(generation, Mapping) or not isinstance(generations, list) or not generations:
        raise ValueError("development generation summary is missing")
    if int(report.get("generation_count") or 0) != len(generations):
        raise ValueError("development generation count mismatch")
    if len(generations) > int(protocol.payload["evolution"]["max_generations"]):
        raise ValueError("development exceeded the frozen generation budget")
    _validate_performance_claim_binding(
        report,
        generations,
        strict_counterfactual_binding=(
            protocol.payload["execution"].get(
                "counterfactual_invalid_evidence_policy"
            )
            == COUNTERFACTUAL_INVALID_EVIDENCE_POLICY_VERSION
        ),
    )
    promotion_spec = protocol.promotion_gate_spec
    for row in generations:
        if not isinstance(row, Mapping):
            raise ValueError("development generation row is malformed")
        _validate_generation_training_evidence(
            row,
            protocol_version=str(protocol.payload.get("protocol_version") or ""),
            expected_policy=str(
                protocol.payload["execution"].get(
                    "contrastive_training_evidence_policy"
                )
                or ""
            ),
        )
        _validate_generation_promotion_decision(
            row,
            promotion_spec=promotion_spec,
            expected_evaluator_epoch=f"skilllearn-eval-{manifest.manifest_hash[:12]}",
            protocol_version=str(protocol.payload.get("protocol_version") or ""),
        )
    if not recursive_validation_enabled and any(
        int(row.get("recursive_depth") or 0) != 0
        for row in generations
        if isinstance(row, Mapping)
    ):
        raise ValueError("no-recursive control unexpectedly used recursive repair")


def _validate_performance_claim_binding(
    report: Mapping[str, Any],
    generations: Sequence[Any],
    *,
    strict_counterfactual_binding: bool = False,
) -> None:
    proposal_model_failure_count = 0
    invalid_counterfactual_pair_count = 0
    counterfactual_provider_mismatch_count = 0
    counterfactual_budget_mismatch_count = 0
    for row in generations:
        if not isinstance(row, Mapping):
            raise ValueError("development generation row is malformed")
        value = row.get("proposal_model_failure_count")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "development generation proposal model failure count is malformed"
            )
        proposal_model_failure_count += value
        invalid_count, provider_count, budget_count = (
            _generation_counterfactual_evidence_counts(row)
        )
        invalid_counterfactual_pair_count += invalid_count
        counterfactual_provider_mismatch_count += provider_count
        counterfactual_budget_mismatch_count += budget_count

    reported_count = report.get("proposal_model_failure_count")
    if (
        isinstance(reported_count, bool)
        or not isinstance(reported_count, int)
        or reported_count < 0
    ):
        raise ValueError(
            "development proposal model failure count is malformed"
        )
    if reported_count != proposal_model_failure_count:
        raise ValueError("development proposal model failure count mismatch")

    expected_present = proposal_model_failure_count > 0
    reported_present = report.get("proposal_model_failures_present")
    if not isinstance(reported_present, bool):
        raise ValueError(
            "development proposal model failure presence is malformed"
        )
    if reported_present is not expected_present:
        raise ValueError("development proposal model failure presence mismatch")

    if strict_counterfactual_binding:
        _validate_reported_nonnegative_count(
            report,
            field="invalid_counterfactual_pair_count",
            expected=invalid_counterfactual_pair_count,
            label="invalid counterfactual pair",
        )
        reported_invalid_present = report.get(
            "invalid_counterfactual_pairs_present"
        )
        if not isinstance(reported_invalid_present, bool):
            raise ValueError(
                "development invalid counterfactual pair presence is malformed"
            )
        if reported_invalid_present is not bool(invalid_counterfactual_pair_count):
            raise ValueError(
                "development invalid counterfactual pair presence mismatch"
            )
        _validate_reported_nonnegative_count(
            report,
            field="counterfactual_provider_mismatch_count",
            expected=counterfactual_provider_mismatch_count,
            label="counterfactual provider mismatch",
        )
        _validate_reported_nonnegative_count(
            report,
            field="counterfactual_budget_mismatch_count",
            expected=counterfactual_budget_mismatch_count,
            label="counterfactual budget mismatch",
        )

    invalid_counterfactual_evidence_present = any(
        (
            invalid_counterfactual_pair_count,
            counterfactual_provider_mismatch_count,
            counterfactual_budget_mismatch_count,
        )
    )
    expected_claim_eligible = not expected_present and (
        not invalid_counterfactual_evidence_present
        or not strict_counterfactual_binding
    )
    reported_claim_eligible = report.get("performance_claim_eligible")
    if not isinstance(reported_claim_eligible, bool):
        raise ValueError("development performance claim eligibility is malformed")
    if reported_claim_eligible is not expected_claim_eligible:
        raise ValueError("development performance claim eligibility mismatch")

    expected_blockers: list[str] = []
    if expected_present:
        expected_blockers.append(_PROPOSAL_MODEL_FAILURE_CLAIM_BLOCKER)
    if (
        strict_counterfactual_binding
        and invalid_counterfactual_evidence_present
    ):
        expected_blockers.append(
            _INVALID_COUNTERFACTUAL_EVIDENCE_CLAIM_BLOCKER
        )
    reported_blockers = report.get("performance_claim_blockers")
    if not isinstance(reported_blockers, list) or any(
        not isinstance(blocker, str) for blocker in reported_blockers
    ):
        raise ValueError("development performance claim blockers are malformed")
    if reported_blockers != expected_blockers:
        raise ValueError("development performance claim blockers mismatch")

    if proposal_model_failure_count > 0:
        raise ValueError("development report contains proposal model failures")
    if invalid_counterfactual_evidence_present:
        raise ValueError(
            "development report contains invalid counterfactual evidence"
        )
    if reported_claim_eligible is not True:
        raise ValueError("development report is not eligible for performance claims")


def _generation_counterfactual_evidence_counts(
    generation: Mapping[str, Any],
) -> tuple[int, int, int]:
    decision = generation.get("promotion_decision")
    reported_summary = generation.get("promotion_summary")
    if decision is None:
        if reported_summary is not None:
            raise ValueError(
                "development generation promotion summary has no decision"
            )
        return 0, 0, 0
    if not isinstance(decision, Mapping):
        raise ValueError("development promotion decision is malformed")
    summary = decision.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("development promotion summary is malformed")
    if reported_summary is not None and reported_summary != summary:
        raise ValueError("development generation promotion summary mismatch")
    counts: list[int] = []
    for field in (
        "invalid_pair_count",
        "provider_mismatch_count",
        "budget_mismatch_count",
    ):
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"development promotion summary {field} is malformed"
            )
        counts.append(value)
    return counts[0], counts[1], counts[2]


def _validate_generation_training_evidence(
    generation: Mapping[str, Any],
    *,
    protocol_version: str,
    expected_policy: str,
) -> None:
    if protocol_version not in CONTRASTIVE_PROTOCOL_VERSIONS:
        return
    if generation.get("contrastive_training_evidence_policy") != expected_policy:
        raise ValueError(
            "development generation contrastive evidence policy mismatch"
        )
    counts: dict[str, int] = {}
    for field in (
        "train_observation_count",
        "valid_train_observation_count",
        "training_residual_count",
        "success_control_count",
        "example_count",
    ):
        value = generation.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "development generation contrastive evidence count is malformed"
            )
        counts[field] = value
    if counts["valid_train_observation_count"] != counts["train_observation_count"]:
        raise ValueError(
            "development generation contains invalid training evidence"
        )
    if (
        counts["training_residual_count"] + counts["success_control_count"]
        != counts["example_count"]
        or counts["example_count"] != counts["valid_train_observation_count"]
    ):
        raise ValueError(
            "development generation contrastive evidence counts are inconsistent"
        )


def _validate_reported_nonnegative_count(
    report: Mapping[str, Any],
    *,
    field: str,
    expected: int,
    label: str,
) -> None:
    reported = report.get(field)
    if isinstance(reported, bool) or not isinstance(reported, int) or reported < 0:
        raise ValueError(f"development {label} count is malformed")
    if reported != expected:
        raise ValueError(f"development {label} count mismatch")


def _validate_generation_promotion_decision(
    row: Mapping[str, Any],
    *,
    promotion_spec: PromotionGateSpec,
    expected_evaluator_epoch: str,
    protocol_version: str,
) -> None:
    promoted = row.get("promoted")
    if not isinstance(promoted, bool):
        raise ValueError("development generation promotion status is malformed")
    accepted_id = str(row.get("accepted_hypothesis_id") or "")
    decision = row.get("promotion_decision")
    treatment_hash = row.get("evaluated_candidate_treatment_hash")
    if decision is None:
        if promoted or accepted_id or treatment_hash is not None:
            raise ValueError("accepted or promoted generation has no promotion decision")
        return
    if not isinstance(decision, Mapping):
        raise ValueError("development promotion decision is malformed")
    if set(decision) != _PROMOTION_DECISION_KEYS:
        raise ValueError("development promotion decision schema mismatch")
    if not accepted_id:
        raise ValueError("development promotion decision has no accepted hypothesis")
    if (
        not isinstance(treatment_hash, str)
        or len(treatment_hash) != 64
        or treatment_hash != treatment_hash.lower()
        or any(character not in "0123456789abcdef" for character in treatment_hash)
    ):
        raise ValueError("development evaluated candidate treatment hash is malformed")
    allowed = decision.get("allowed")
    if not isinstance(allowed, bool) or allowed != promoted:
        raise ValueError("development promotion decision status mismatch")
    blockers = decision.get("blockers")
    if not isinstance(blockers, list) or any(
        not isinstance(blocker, str) for blocker in blockers
    ):
        raise ValueError("development promotion blockers are malformed")
    if allowed == bool(blockers):
        raise ValueError("development promotion blockers contradict decision status")
    if decision.get("promotion_contract") != promotion_spec.to_dict():
        raise ValueError("development promotion decision contract mismatch")
    if decision.get("candidate_metric") != promotion_spec.metric:
        raise ValueError("development candidate metric mismatch")
    if decision.get("evaluator_epoch") != expected_evaluator_epoch:
        raise ValueError("development promotion evaluator epoch mismatch")
    if decision.get("policy") != _PROMOTION_DECISION_POLICY:
        raise ValueError("development promotion decision policy mismatch")

    summary = _pair_summary_from_mapping(
        decision.get("summary"),
        confidence=promotion_spec.confidence,
        protocol_version=protocol_version,
    )
    effect_lower_bound = decision.get("effect_lower_bound")
    expected_lower_bound = summary.effect_lower_bound(promotion_spec.confidence)
    if (
        isinstance(effect_lower_bound, bool)
        or not isinstance(effect_lower_bound, (int, float))
        or not math.isfinite(effect_lower_bound)
        or not math.isclose(
            float(effect_lower_bound),
            expected_lower_bound,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("development promotion effect lower bound is malformed")

    candidate = _promotion_threshold_mapping(
        decision.get("candidate_thresholds"),
        label="candidate",
    )
    effective = _promotion_threshold_mapping(
        decision.get("effective_thresholds"),
        label="effective",
    )
    expected_effective = promotion_spec.effective_thresholds_from_candidate(candidate)
    if effective != expected_effective:
        raise ValueError("development effective promotion thresholds mismatch")
    expected_blockers = promotion_summary_blockers(
        promotion_spec,
        summary,
        effective_thresholds=effective,
    )
    if tuple(blockers) != expected_blockers:
        raise ValueError("development promotion summary blockers mismatch")
    if allowed is not (not expected_blockers):
        raise ValueError("development promotion summary decision mismatch")


def _pair_summary_from_mapping(
    payload: Any,
    *,
    confidence: float,
    protocol_version: str,
) -> PairSummary:
    if protocol_version in CONTRASTIVE_PROTOCOL_VERSIONS:
        expected_keys = _PROMOTION_SUMMARY_V3_6_KEYS
        count_keys = (
            _PROMOTION_SUMMARY_LEGACY_COUNT_KEYS
            | _PROMOTION_SUMMARY_DIAGNOSTIC_COUNT_KEYS
        )
    elif protocol_version in _LEGACY_PROMOTION_SUMMARY_PROTOCOL_VERSIONS:
        expected_keys = _PROMOTION_SUMMARY_LEGACY_KEYS
        count_keys = _PROMOTION_SUMMARY_LEGACY_COUNT_KEYS
    else:
        raise ValueError("development promotion summary protocol version unsupported")
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError("development promotion summary schema mismatch")
    counts: dict[str, int] = {}
    for key in count_keys:
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("development promotion summary count is malformed")
        counts[key] = value
    pair_count = counts["pair_count"]
    if any(value > pair_count for value in counts.values()):
        raise ValueError("development promotion summary count exceeds pair count")
    if counts["gain_count"] + counts["harm_count"] + counts["tie_count"] != pair_count:
        raise ValueError("development promotion summary outcomes are inconsistent")
    if (
        counts["candidate_success_count"] - counts["baseline_success_count"]
        != counts["gain_count"] - counts["harm_count"]
    ):
        raise ValueError("development promotion summary successes are inconsistent")

    numeric: dict[str, float] = {}
    for key in _PROMOTION_SUMMARY_BASE_FLOAT_KEYS:
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("development promotion summary value is malformed")
        normalized = float(value)
        if math.isnan(normalized) or (
            key != "cost_ratio" and not math.isfinite(normalized)
        ):
            raise ValueError("development promotion summary value is malformed")
        numeric[key] = normalized
    if (
        numeric["baseline_mean_cost"] < 0.0
        or numeric["candidate_mean_cost"] < 0.0
        or numeric["cost_ratio"] < 0.0
        or numeric["effect_standard_error"] < 0.0
        or not -1.0 <= numeric["mean_effect"] <= 1.0
    ):
        raise ValueError("development promotion summary value is out of range")
    expected_mean_effect = (
        (counts["gain_count"] - counts["harm_count"]) / pair_count
        if pair_count
        else 0.0
    )
    if not math.isclose(
        numeric["mean_effect"],
        expected_mean_effect,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("development promotion summary mean effect is inconsistent")

    summary_counts = dict(counts)
    if protocol_version not in CONTRASTIVE_PROTOCOL_VERSIONS:
        summary_counts.update(
            {
                "valid_activation_count": 0,
                "activated_gain_count": 0,
                "activated_harm_count": 0,
                "abstention_count": pair_count - counts["activation_count"],
            }
        )
    summary = PairSummary(
        **summary_counts,
        **numeric,
    )
    expected_derived = summary.to_dict(confidence=confidence)
    for key in ("effect_lower_bound", "harm_rate", "activation_rate"):
        value = payload[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not math.isclose(
                float(value),
                float(expected_derived[key]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("development promotion summary derived value is inconsistent")
    if protocol_version in CONTRASTIVE_PROTOCOL_VERSIONS:
        _validate_pair_diagnostics(
            payload,
            counts=counts,
            expected_derived=expected_derived,
        )
    return summary


def _validate_pair_diagnostics(
    payload: Mapping[str, Any],
    *,
    counts: Mapping[str, int],
    expected_derived: Mapping[str, float | int | bool | None],
) -> None:
    pair_count = counts["pair_count"]
    activation_count = counts["activation_count"]
    invalid_pair_count = counts["invalid_pair_count"]
    provider_mismatch_count = counts["provider_mismatch_count"]
    budget_mismatch_count = counts["budget_mismatch_count"]
    valid_activation_count = counts["valid_activation_count"]
    activated_gain_count = counts["activated_gain_count"]
    activated_harm_count = counts["activated_harm_count"]
    invalid_union_max = min(
        pair_count,
        invalid_pair_count + provider_mismatch_count + budget_mismatch_count,
    )
    invalid_union_min = max(
        invalid_pair_count,
        provider_mismatch_count,
        budget_mismatch_count,
    )
    if not (
        max(0, activation_count - invalid_union_max)
        <= valid_activation_count
        <= min(activation_count, pair_count - invalid_union_min)
    ):
        raise ValueError(
            "development promotion summary valid activations are inconsistent"
        )
    if not (
        max(0, counts["gain_count"] - invalid_union_max)
        <= activated_gain_count
        <= min(counts["gain_count"], valid_activation_count)
    ) or not (
        max(0, counts["harm_count"] - invalid_union_max)
        <= activated_harm_count
        <= min(counts["harm_count"], valid_activation_count)
    ) or (
        counts["gain_count"]
        - activated_gain_count
        + counts["harm_count"]
        - activated_harm_count
        > invalid_union_max
    ) or (
        activated_gain_count + activated_harm_count > valid_activation_count
    ):
        raise ValueError(
            "development promotion summary activated outcomes are inconsistent"
        )
    if counts["abstention_count"] != pair_count - activation_count:
        raise ValueError(
            "development promotion summary abstentions are inconsistent"
        )

    expected_defined = valid_activation_count > 0
    for key in (
        "activation_precision_defined",
        "activated_harm_rate_defined",
    ):
        if payload[key] is not expected_defined:
            raise ValueError(
                "development promotion summary diagnostic definition is inconsistent"
            )
    for key in (
        "activation_precision",
        "activated_harm_rate",
        "abstention_rate",
    ):
        value = payload[key]
        expected = expected_derived[key]
        if expected is None:
            if value is not None:
                raise ValueError(
                    "development promotion summary diagnostic value is inconsistent"
                )
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not math.isclose(
                float(value),
                float(expected),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "development promotion summary diagnostic value is inconsistent"
            )


def _promotion_threshold_mapping(
    payload: Any,
    *,
    label: str,
) -> dict[str, float]:
    if not isinstance(payload, Mapping) or set(payload) != _PROMOTION_THRESHOLD_KEYS:
        raise ValueError(f"development {label} promotion threshold schema mismatch")
    normalized: dict[str, float] = {}
    for key in _PROMOTION_THRESHOLD_KEYS:
        value = payload[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"development {label} promotion threshold is malformed")
        normalized[key] = float(value)
    return normalized


def _manifest_role(lock: Mapping[str, Any], manifest: SplitManifest) -> str:
    if manifest.manifest_hash == lock.get("primary_manifest_hash"):
        return "primary_instance_holdout"
    if manifest.manifest_hash == lock.get("secondary_manifest_hash"):
        return "secondary_family_out"
    raise PermissionError("manifest is not part of the frozen paper protocol")


def _read_mapping(path: str | Path, label: str) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _file_content_hash(path: str | Path) -> str:
    return stable_hash({"bytes": Path(path).read_bytes().hex()})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze two development archives and compile immutable paper controls."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--recursive-report", type=Path, required=True)
    parser.add_argument("--recursive-archive", type=Path, required=True)
    parser.add_argument("--no-recursive-report", type=Path, required=True)
    parser.add_argument("--no-recursive-archive", type=Path, required=True)
    parser.add_argument("--controls-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = PaperProtocol.read(args.protocol)
    lock = _read_mapping(args.protocol_lock, "protocol lock")
    manifest = SplitManifest.read(args.manifest)
    receipt = freeze_paper_workspace(
        protocol=protocol,
        protocol_lock=lock,
        manifest=manifest,
        benchmark_root=args.benchmark_root,
        project_root=args.project_root,
        recursive_report_path=args.recursive_report,
        recursive_archive_path=args.recursive_archive,
        no_recursive_report_path=args.no_recursive_report,
        no_recursive_archive_path=args.no_recursive_archive,
        controls_output_root=args.controls_out,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "receipt_hash": receipt["receipt_hash"],
                "selected_candidate_available": receipt["selected_candidate_available"],
                "test_content_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
