from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..archive import PolicyArchive
from ..evaluation import PromotionGate
from ..events import Event, JsonlEventSink
from ..evolution import (
    COUNTERFACTUAL_REPLAY_POLICY_VERSION,
    CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION,
    PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
    TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
    CounterfactualEvidenceReplayCache,
)
from ..models import stable_hash
from ..provider_chain import build_proposal_model, proposal_provider_status
from ..proposer import (
    LEGACY_PROPOSAL_DIVERSITY_POLICY_VERSION,
    PROPOSAL_DIVERSITY_POLICY_VERSION,
    ROOT_PROPOSAL_REPLAY_POLICY_VERSION,
    HypothesisProposalCallError,
    StructuredHypothesisProposer,
)
from ..secure_env import (
    configured_model,
    load_dotenv,
    map_legacy_model_env,
)
from ..splits import SplitAccessGuard, SplitManifest
from ..validation import (
    EvaluatorEpochCheck,
    RecursiveValidationEngine,
    RuntimeCandidateKindCheck,
    RuntimeActionCheck,
    SchemaCheck,
    TrainingSupportCheck,
    TriggerVocabularyCheck,
)
from .preflight import build_preflight
from .prewarm import validate_development_prewarm_receipt
from .paper_protocol import (
    COUNTERFACTUAL_INVALID_EVIDENCE_POLICY_VERSION,
    PaperProtocol,
    validate_protocol_lock_for_execution,
)
from .skilllearn_compiler import (
    SKILL_ACTION_LOWERING_VERSION,
    SKILL_FALLBACK_SEMANTICS_VERSION,
    SKILL_ROUTING_VERSION,
)
from .docker_egress import (
    DEPENDENCY_CACHE_POLICY_VERSION,
    DOCKER_EGRESS_POLICY_VERSION,
    PROVIDER_DNS_POLICY_VERSION,
    TRIAL_NETWORK_BUDGET_POLICY_VERSION,
)
from .skilllearn_lifecycle import (
    BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION,
    CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSIONS,
    MODEL_INFERENCE_CONCURRENCY_POLICY_VERSION,
    MODEL_ONLY_TOOL_POLICY_VERSION,
    OPENAI_COMPATIBLE_CODEX_CONFIG_VERSION,
    PREBUILT_IMAGE_POLICY_VERSION,
    PROVIDER_FAILURE_POLICY_VERSION,
    PROVIDER_ROUTE_POLICY_VERSION,
    RUNNER_AGENT_REGISTRY_ISOLATION_VERSION,
    INVALID_TRIAL_RETRY_POLICY_VERSION,
    LOCAL_EVIDENCE_TRANSPORT_VERSION,
    NETWORK_SCOPE_AUDIT_VERSION,
    PROPOSAL_FAILURE_ISOLATION_POLICY_VERSION,
    TRAINING_EVIDENCE_POLICY_VERSION,
    TRAINING_EVIDENCE_REPLAY_POLICY_VERSION,
    TRIAL_TIMEOUT_POLICY_VERSION,
    VERIFIER_EXECUTION_RECEIPT_POLICY_VERSION,
    SkillLearnBackendPool,
    SkillLearnEvolutionHarness,
    SkillLearnGenerationResult,
    SkillLearnPrebuiltImageCache,
    SkillLearnProviderCircuit,
    SkillLearnModelInferenceLimiter,
    SkillLearnSubprocessBackend,
    TrainingEvidenceReplayCache,
    codex_network_minimization_for_policy,
)
from .offline_verifier import OFFLINE_VERIFIER_POLICY_VERSION
from .skilllearnbench import SkillLearnBenchAdapter


_PROPOSAL_MODEL_FAILURE_CLAIM_BLOCKER = (
    "proposal_model_failure_evidence_present"
)
_INVALID_COUNTERFACTUAL_EVIDENCE_CLAIM_BLOCKER = (
    "invalid_counterfactual_evidence_present"
)
_PROMOTION_SUMMARY_DIAGNOSTIC_KEYS = frozenset(
    {
        "valid_activation_count",
        "activated_gain_count",
        "activated_harm_count",
        "abstention_count",
        "activation_precision",
        "activation_precision_defined",
        "activated_harm_rate",
        "activated_harm_rate_defined",
        "abstention_rate",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or execute one guarded SkillLearnBench self-evolution generation."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--archive-out", type=Path)
    parser.add_argument("--paired-no-recursive-out", type=Path)
    parser.add_argument("--paired-no-recursive-archive-out", type=Path)
    parser.add_argument("--prewarm-receipt", type=Path)
    parser.add_argument("--train-id", action="append", default=[])
    parser.add_argument("--validation-id", action="append", default=[])
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--disable-recursive-repair", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    paired_ablation = bool(
        args.paired_no_recursive_out and args.paired_no_recursive_archive_out
    )

    load_dotenv(args.env_file)
    model_presence = map_legacy_model_env()
    provider_status = proposal_provider_status()
    paper_protocol = PaperProtocol.read(args.protocol)
    codex_agent_execution_policy = paper_protocol.codex_agent_execution_policy
    promotion_spec = paper_protocol.promotion_gate_spec
    execution_contract = paper_protocol.payload["execution"]
    evolution_contract = paper_protocol.payload["evolution"]
    agent_id = str(paper_protocol.payload["agent_id"])
    model = str(paper_protocol.payload["model"])
    trial_provider_mode = str(paper_protocol.payload["trial_provider_mode"])
    max_steps = int(paper_protocol.payload["max_steps"])
    minimum_trigger_support = int(evolution_contract["minimum_trigger_support"])
    candidate_selection_policy = str(
        execution_contract["proposal_candidate_selection"]
    )
    contrastive_training_evidence_policy = execution_contract.get(
        "contrastive_training_evidence_policy"
    )
    proposal_diversity_policy = execution_contract.get(
        "proposal_diversity_policy"
    )
    if proposal_diversity_policy is not None:
        proposal_diversity_policy = str(proposal_diversity_policy)
    proposal_response_max_tokens = execution_contract.get(
        "proposal_response_max_tokens"
    )
    if proposal_response_max_tokens is not None:
        if isinstance(proposal_response_max_tokens, bool):
            raise ValueError("proposal response token budget must be an integer")
        proposal_response_max_tokens = int(proposal_response_max_tokens)
        if proposal_response_max_tokens <= 0:
            raise ValueError("proposal response token budget must be positive")
    if contrastive_training_evidence_policy is not None:
        contrastive_training_evidence_policy = str(
            contrastive_training_evidence_policy
        )
    if candidate_selection_policy not in {
        TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
        CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION,
        PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
    }:
        raise ValueError("unsupported protocol candidate selection policy")
    if contrastive_training_evidence_policy not in {
        None,
        *CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSIONS,
    }:
        raise ValueError("unsupported protocol contrastive training evidence policy")
    if (
        contrastive_training_evidence_policy
        in CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSIONS
    ) != (
        candidate_selection_policy in {
            CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION,
            PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
        }
    ):
        raise ValueError(
            "protocol contrastive evidence and candidate selection policies must be paired"
        )
    diversity_enabled = (
        candidate_selection_policy
        == PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION
    )
    if diversity_enabled != (
        proposal_diversity_policy
        in {
            LEGACY_PROPOSAL_DIVERSITY_POLICY_VERSION,
            PROPOSAL_DIVERSITY_POLICY_VERSION,
        }
        and proposal_response_max_tokens is not None
    ):
        raise ValueError(
            "coverage-aware selection requires the proposal diversity and response-budget contracts"
        )
    counterfactual_invalid_evidence_policy = execution_contract.get(
        "counterfactual_invalid_evidence_policy"
    )
    if counterfactual_invalid_evidence_policy is not None:
        counterfactual_invalid_evidence_policy = str(
            counterfactual_invalid_evidence_policy
        )
    if counterfactual_invalid_evidence_policy not in {
        None,
        COUNTERFACTUAL_INVALID_EVIDENCE_POLICY_VERSION,
    }:
        raise ValueError("unsupported counterfactual invalid evidence policy")
    invalid_trial_max_attempts = int(
        execution_contract["invalid_trial_max_attempts"]
    )
    invalid_trial_retry_backoff_seconds = float(
        execution_contract["invalid_trial_retry_backoff_seconds"]
    )
    invalid_trial_retry_workers = int(
        execution_contract["invalid_trial_retry_workers"]
    )
    model_inference_concurrency_policy = execution_contract.get(
        "model_inference_concurrency_policy"
    )
    model_inference_slots = int(execution_contract.get("model_inference_slots") or 0)
    if model_inference_concurrency_policy not in {
        None,
        MODEL_INFERENCE_CONCURRENCY_POLICY_VERSION,
    }:
        raise ValueError("unsupported model inference concurrency policy")
    if (model_inference_concurrency_policy is None) != (model_inference_slots == 0):
        raise ValueError("model inference concurrency policy and slots must be paired")
    manifest = SplitManifest.read(args.manifest)
    protocol_root = paper_protocol.path.parent.parent
    allowed_manifest_paths = {
        (protocol_root / str(paper_protocol.payload[key])).resolve()
        for key in ("primary_manifest", "secondary_manifest")
    }
    if args.manifest.expanduser().resolve() not in allowed_manifest_paths:
        raise ValueError("experiment manifest is not owned by the frozen paper protocol")
    protocol_lock_hash: str | None = None
    protocol_lock_error: str | None = None
    if args.protocol_lock:
        protocol_lock_payload = json.loads(
            args.protocol_lock.read_text(encoding="utf-8")
        )
        if not isinstance(protocol_lock_payload, Mapping):
            raise ValueError("protocol lock must contain one JSON object")
        try:
            protocol_lock_hash = validate_protocol_lock_for_execution(
                paper_protocol,
                protocol_lock_payload,
                manifest,
                protocol_root,
                args.root,
            )
        except PermissionError as exc:
            protocol_lock_error = str(exc)
    prewarm_receipt_hash: str | None = None
    if args.prewarm_receipt:
        prewarm_payload = json.loads(args.prewarm_receipt.read_text(encoding="utf-8"))
        if not isinstance(prewarm_payload, Mapping):
            raise ValueError("development prewarm receipt must contain one JSON object")
        prewarm_receipt_hash = validate_development_prewarm_receipt(
            prewarm_payload,
            manifest=manifest,
            expected_version=str(execution_contract["development_prewarm"]),
        )
    adapter = SkillLearnBenchAdapter(args.root)
    items = adapter.discover()
    inventory_ids = {item.id for item in items}
    manifest_ids = {*manifest.train_ids, *manifest.validation_ids, *manifest.test_ids}
    if not manifest_ids <= inventory_ids:
        raise ValueError("manifest contains IDs absent from the local SkillLearnBench inventory")

    train_ids = _select_ids(args.train_id, manifest.train_ids, args.train_limit)
    validation_ids = _select_ids(
        args.validation_id,
        manifest.validation_ids,
        args.validation_limit,
    )
    experiment_phase = _experiment_phase_name(
        paper_protocol,
        manifest=manifest,
        train_ids=train_ids,
        validation_ids=validation_ids,
    )
    parallel_workers = (
        int(paper_protocol.payload["phases"][experiment_phase]["parallel_workers"])
        if experiment_phase is not None
        else 0
    )
    if experiment_phase == "smoke":
        phase_contract = paper_protocol.payload["phases"]["smoke"]
        max_generations = int(phase_contract["max_generations"])
        max_consecutive_non_promotions = int(
            phase_contract["max_consecutive_non_promotions"]
        )
    else:
        max_generations = int(evolution_contract["max_generations"])
        max_consecutive_non_promotions = int(
            evolution_contract["max_consecutive_non_promotions"]
        )
    proposal_candidates_per_generation = int(
        evolution_contract["proposal_candidates_per_generation"]
    )
    plan = {
        "protocol": manifest.protocol,
        "manifest_hash": manifest.manifest_hash,
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "sealed_test_count": len(manifest.test_ids),
        "train_item_set_hash": _item_set_hash(train_ids),
        "validation_item_set_hash": _item_set_hash(validation_ids),
        "train_family_count": len({manifest.family_by_id[item_id] for item_id in train_ids}),
        "validation_family_count": len(
            {manifest.family_by_id[item_id] for item_id in validation_ids}
        ),
        "paper_protocol_id": paper_protocol.id,
        "paper_protocol_hash": paper_protocol.protocol_hash,
        "promotion_contract": promotion_spec.to_dict(),
        "protocol_lock_hash": protocol_lock_hash,
        "experiment_phase": experiment_phase,
        "minimum_pairs": promotion_spec.minimum_pairs,
        "minimum_trigger_support": minimum_trigger_support,
        "recursive_validation_enabled": not args.disable_recursive_repair,
        "max_generations": max_generations,
        "max_consecutive_non_promotions": max_consecutive_non_promotions,
        "proposal_candidates_per_generation": proposal_candidates_per_generation,
        **(
            {
                "proposal_diversity_policy": proposal_diversity_policy,
                "proposal_response_max_tokens": proposal_response_max_tokens,
            }
            if diversity_enabled
            else {}
        ),
        "agent_id": agent_id,
        "model": model,
        "trial_provider_mode": trial_provider_mode,
        "max_steps": max_steps,
        "parallel_workers": parallel_workers,
        **(
            {
                "model_inference_concurrency_policy": (
                    model_inference_concurrency_policy
                ),
                "model_inference_slots": model_inference_slots,
            }
            if model_inference_concurrency_policy is not None
            else {}
        ),
        "prebuilt_image_policy": PREBUILT_IMAGE_POLICY_VERSION,
        "runner_agent_registry_isolation": RUNNER_AGENT_REGISTRY_ISOLATION_VERSION,
        "trial_timeout_policy": TRIAL_TIMEOUT_POLICY_VERSION,
        "provider_failure_policy": PROVIDER_FAILURE_POLICY_VERSION,
        "provider_route_policy": PROVIDER_ROUTE_POLICY_VERSION,
        "counterfactual_replay_policy": COUNTERFACTUAL_REPLAY_POLICY_VERSION,
        "baseline_arm_evidence_replay_policy": (
            BASELINE_ARM_EVIDENCE_REPLAY_POLICY_VERSION
        ),
        "root_proposal_replay_policy": ROOT_PROPOSAL_REPLAY_POLICY_VERSION,
        "training_evidence_replay_policy": (
            TRAINING_EVIDENCE_REPLAY_POLICY_VERSION
        ),
        "invalid_trial_retry_policy": INVALID_TRIAL_RETRY_POLICY_VERSION,
        "invalid_trial_max_attempts": invalid_trial_max_attempts,
        "invalid_trial_retry_backoff_seconds": invalid_trial_retry_backoff_seconds,
        "invalid_trial_retry_workers": invalid_trial_retry_workers,
        "local_evidence_transport": LOCAL_EVIDENCE_TRANSPORT_VERSION,
        "network_scope_audit": NETWORK_SCOPE_AUDIT_VERSION,
        "proposal_failure_isolation_policy": (
            PROPOSAL_FAILURE_ISOLATION_POLICY_VERSION
        ),
        "openai_compatible_codex_config": (
            OPENAI_COMPATIBLE_CODEX_CONFIG_VERSION
            if trial_provider_mode == "openai_compatible"
            else None
        ),
        "codex_network_minimization": codex_network_minimization_for_policy(
            codex_agent_execution_policy
        ),
        "codex_agent_execution_policy": codex_agent_execution_policy.to_dict(),
        "codex_agent_execution_policy_hash": (
            codex_agent_execution_policy.policy_hash
        ),
        "model_only_tool_policy": MODEL_ONLY_TOOL_POLICY_VERSION,
        "verifier_execution_receipt_policy": (
            VERIFIER_EXECUTION_RECEIPT_POLICY_VERSION
        ),
        "offline_verifier_policy": OFFLINE_VERIFIER_POLICY_VERSION,
        "container_egress_policy": DOCKER_EGRESS_POLICY_VERSION,
        "dependency_cache_policy": DEPENDENCY_CACHE_POLICY_VERSION,
        "provider_dns_policy": PROVIDER_DNS_POLICY_VERSION,
        "trial_network_budget_policy": TRIAL_NETWORK_BUDGET_POLICY_VERSION,
        "trial_network_byte_limit": int(
            execution_contract["trial_network_byte_limit"]
        ),
        "training_evidence_policy": TRAINING_EVIDENCE_POLICY_VERSION,
        "development_prewarm_version": execution_contract["development_prewarm"],
        "prewarm_passed": prewarm_receipt_hash is not None,
        "prewarm_receipt_hash": prewarm_receipt_hash,
        "proposal_candidate_selection": candidate_selection_policy,
        "contrastive_training_evidence_policy": (
            contrastive_training_evidence_policy
        ),
        "counterfactual_invalid_evidence_policy": (
            counterfactual_invalid_evidence_policy
        ),
        "runtime_candidate_kinds": ["task", "policy"],
        "evaluator_hypothesis_mode": (
            "separate_epoch_challenger_not_in_primary_runtime"
        ),
        "skill_routing": SKILL_ROUTING_VERSION,
        "skill_action_lowering": SKILL_ACTION_LOWERING_VERSION,
        "skill_fallback_semantics": SKILL_FALLBACK_SEMANTICS_VERSION,
        "proposal_api_present": bool(
            model_presence["base_url_present"] and model_presence["api_key_present"]
        ),
        "proposal_provider_ready": provider_status["passed"],
        "proposal_provider_chain": provider_status["requested_providers"],
        "proposal_ready_providers": provider_status["ready_providers"],
        "test_content_accessed": False,
        "raw_content_persisted": False,
        "paired_first_generation_ablation": paired_ablation,
    }
    blockers: list[str] = []
    if args.execute and args.protocol_lock is None:
        blockers.append("execute_requires_protocol_lock")
    if args.protocol_lock is not None and protocol_lock_error is not None:
        blockers.append("protocol_lock_validation_failed")
    plan["protocol_lock_validation_error"] = protocol_lock_error
    if experiment_phase is None:
        blockers.append("selection_does_not_match_frozen_experiment_phase")
    plan["promotion_evidence_underpowered"] = (
        len(validation_ids) < promotion_spec.minimum_pairs
    )
    if len(train_ids) < minimum_trigger_support:
        blockers.append("minimum_trigger_support_exceeds_training_selection")
    if args.execute and prewarm_receipt_hash is None:
        blockers.append("execute_requires_passed_development_prewarm")
    if bool(args.paired_no_recursive_out) != bool(args.paired_no_recursive_archive_out):
        blockers.append("paired_ablation_requires_both_output_paths")
    if paired_ablation and args.disable_recursive_repair:
        blockers.append("paired_ablation_primary_arm_must_enable_recursive_repair")
    plan["plan_blockers"] = blockers
    sink = JsonlEventSink(args.events)
    sink.emit(
        Event(
            event="skilllearn_experiment_planned",
            stage="benchmark.skilllearn.plan",
            trace_id=manifest.manifest_hash[:20],
            payload=plan,
        )
    )

    if not args.execute:
        report = {
            "mode": "dry_run",
            "plan": plan,
            "preflight": build_preflight(
                args.root,
                trial_provider_mode=trial_provider_mode,
                item_ids=(*manifest.train_ids, *manifest.validation_ids, *manifest.test_ids),
            ),
            "executed": False,
            "secret_value_persisted": False,
        }
        _write_report(args.out, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if blockers:
        raise RuntimeError(f"experiment plan is blocked: {blockers}")
    preflight = build_preflight(
        args.root,
        trial_provider_mode=trial_provider_mode,
        item_ids=(*manifest.train_ids, *manifest.validation_ids, *manifest.test_ids),
    )
    if preflight["blockers"]:
        raise RuntimeError(f"SkillLearnBench execution preflight failed: {preflight['blockers']}")

    if configured_model() != model:
        raise RuntimeError("proposal model and benchmark model must match")
    proposal_model = build_proposal_model(
        event_sink=sink,
        max_tokens=proposal_response_max_tokens,
    )
    proposal_model.complete_with_trace(
        {
            "request_kind": "health_probe",
            "contract_version": "v1",
            "instruction": "Return a JSON object with ok=true.",
            "output_schema": {"ok": True},
        },
        trace_id="skilllearn-model-health-preflight",
    )
    proposer = StructuredHypothesisProposer(proposal_model, event_sink=sink)
    validator = RecursiveValidationEngine(
        [
            SchemaCheck(),
            RuntimeCandidateKindCheck(),
            TriggerVocabularyCheck(),
            TrainingSupportCheck(min_support=minimum_trigger_support),
            RuntimeActionCheck(),
            EvaluatorEpochCheck(),
        ],
        proposer=None if args.disable_recursive_repair else proposer,
        event_sink=sink,
    )
    archive = PolicyArchive(event_sink=sink)
    guard = SplitAccessGuard(manifest, event_sink=sink)
    prebuilt_cache = SkillLearnPrebuiltImageCache(args.root, event_sink=sink)
    provider_circuit = SkillLearnProviderCircuit()
    model_inference_limiter = (
        SkillLearnModelInferenceLimiter(model_inference_slots)
        if model_inference_concurrency_policy is not None
        else None
    )
    backends = tuple(
        SkillLearnSubprocessBackend(
            args.root,
            agent_id=agent_id,
            model=model,
            max_steps=max_steps,
            provider_mode=trial_provider_mode,
            trials_dir=args.work_dir / "upstream_trials",
            prebuilt_cache=prebuilt_cache,
            provider_circuit=provider_circuit,
            model_inference_limiter=model_inference_limiter,
            codex_agent_execution_policy=codex_agent_execution_policy,
            event_sink=sink,
        )
        for _ in range(parallel_workers)
    )
    backend = backends[0] if len(backends) == 1 else SkillLearnBackendPool(backends)
    harness = SkillLearnEvolutionHarness(
        adapter=adapter,
        manifest=manifest,
        guard=guard,
        backend=backend,
        proposer=proposer,
        validator=validator,
        promotion_gate=PromotionGate(
            promotion_spec,
            event_sink=sink,
        ),
        archive=archive,
        evaluator_epoch=f"skilllearn-eval-{manifest.manifest_hash[:12]}",
        output_root=args.work_dir / "compiled_skills",
        proposal_candidates_per_generation=proposal_candidates_per_generation,
        candidate_selection_policy=candidate_selection_policy,
        contrastive_training_evidence_policy=(
            contrastive_training_evidence_policy
        ),
        parallel_workers=parallel_workers,
        invalid_trial_max_attempts=invalid_trial_max_attempts,
        invalid_trial_retry_backoff_seconds=invalid_trial_retry_backoff_seconds,
        invalid_trial_retry_workers=invalid_trial_retry_workers,
        event_sink=sink,
    )
    archive_path = args.archive_out or args.work_dir / "archive.json"
    if paired_ablation:
        no_recursive_archive = PolicyArchive(event_sink=sink)
        no_recursive_guard = SplitAccessGuard(manifest, event_sink=sink)
        no_recursive_harness = SkillLearnEvolutionHarness(
            adapter=adapter,
            manifest=manifest,
            guard=no_recursive_guard,
            backend=backend,
            proposer=proposer,
            validator=RecursiveValidationEngine(
                [
                    SchemaCheck(),
                    RuntimeCandidateKindCheck(),
                    TriggerVocabularyCheck(),
                    TrainingSupportCheck(min_support=minimum_trigger_support),
                    RuntimeActionCheck(),
                    EvaluatorEpochCheck(),
                ],
                proposer=None,
                event_sink=sink,
            ),
            promotion_gate=PromotionGate(
                promotion_spec,
                event_sink=sink,
            ),
            archive=no_recursive_archive,
            evaluator_epoch=f"skilllearn-eval-{manifest.manifest_hash[:12]}",
            output_root=args.work_dir / "compiled_skills_no_recursive",
            proposal_candidates_per_generation=proposal_candidates_per_generation,
            candidate_selection_policy=candidate_selection_policy,
            contrastive_training_evidence_policy=(
                contrastive_training_evidence_policy
            ),
            parallel_workers=parallel_workers,
            invalid_trial_max_attempts=invalid_trial_max_attempts,
            invalid_trial_retry_backoff_seconds=invalid_trial_retry_backoff_seconds,
            invalid_trial_retry_workers=invalid_trial_retry_workers,
            event_sink=sink,
        )
        paired = _run_paired_arms(
            recursive_harness=harness,
            no_recursive_harness=no_recursive_harness,
            train_ids=train_ids,
            validation_ids=validation_ids,
            manifest_hash=manifest.manifest_hash,
            max_generations=max_generations,
            max_consecutive_non_promotions=max_consecutive_non_promotions,
            counterfactual_invalid_evidence_policy=(
                counterfactual_invalid_evidence_policy
            ),
        )
        plan["shared_first_generation_checkpoint_hash"] = paired["checkpoint_hash"]
        no_recursive_plan = {
            **plan,
            "recursive_validation_enabled": False,
            "paired_arm": "no_recursive_repair",
        }
        plan["paired_arm"] = "recursive_repair"
        archive.write(archive_path)
        no_recursive_archive.write(args.paired_no_recursive_archive_out)
        report = _execution_report(
            plan=plan,
            preflight=preflight,
            generations=paired["recursive_generations"],
            stop_reason=str(paired["recursive_stop_reason"]),
            archive=archive,
            archive_path=archive_path,
            guard=guard,
        )
        no_recursive_report = _execution_report(
            plan=no_recursive_plan,
            preflight=preflight,
            generations=paired["no_recursive_generations"],
            stop_reason=str(paired["no_recursive_stop_reason"]),
            archive=no_recursive_archive,
            archive_path=args.paired_no_recursive_archive_out,
            guard=no_recursive_guard,
        )
        _write_report(args.out, report)
        _write_report(args.paired_no_recursive_out, no_recursive_report)
        print(
            json.dumps(
                {
                    "recursive_report": report,
                    "no_recursive_report": no_recursive_report,
                    "shared_first_generation_checkpoint_hash": paired["checkpoint_hash"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    generations, stop_reason = _run_single_arm(
        harness,
        train_ids=train_ids,
        validation_ids=validation_ids,
        manifest_hash=manifest.manifest_hash,
        max_generations=max_generations,
        max_consecutive_non_promotions=max_consecutive_non_promotions,
        counterfactual_invalid_evidence_policy=(
            counterfactual_invalid_evidence_policy
        ),
    )
    archive.write(archive_path)
    report = _execution_report(
        plan=plan,
        preflight=preflight,
        generations=generations,
        stop_reason=stop_reason,
        archive=archive,
        archive_path=archive_path,
        guard=guard,
    )
    _write_report(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _run_single_arm(
    harness: SkillLearnEvolutionHarness,
    *,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    manifest_hash: str,
    max_generations: int,
    max_consecutive_non_promotions: int,
    counterfactual_invalid_evidence_policy: str | None = None,
) -> tuple[list[SkillLearnGenerationResult], str]:
    generations: list[SkillLearnGenerationResult] = []
    training_replay_cache = TrainingEvidenceReplayCache(
        event_sink=harness.event_sink
    )
    counterfactual_replay_cache = CounterfactualEvidenceReplayCache(
        event_sink=harness.event_sink
    )
    consecutive = 0
    stop_reason = "max_generations_reached"
    for generation_index in range(1, max_generations + 1):
        generation = harness.run_generation(
            train_item_ids=train_ids,
            validation_item_ids=validation_ids,
            training_replay_cache=training_replay_cache,
            counterfactual_replay_cache=counterfactual_replay_cache,
            trace_id=f"skilllearn-generation-{manifest_hash[:12]}-g{generation_index}",
        )
        generations.append(generation)
        active, consecutive, reason = _advance_arm(
            generation,
            consecutive_non_promotions=consecutive,
            maximum=max_consecutive_non_promotions,
            counterfactual_invalid_evidence_policy=(
                counterfactual_invalid_evidence_policy
            ),
        )
        if not active:
            stop_reason = reason
            break
    return generations, stop_reason


def _run_paired_arms(
    *,
    recursive_harness: SkillLearnEvolutionHarness,
    no_recursive_harness: SkillLearnEvolutionHarness,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    manifest_hash: str,
    max_generations: int,
    max_consecutive_non_promotions: int,
    counterfactual_invalid_evidence_policy: str | None = None,
) -> dict[str, Any]:
    if (
        recursive_harness.candidate_selection_policy
        != no_recursive_harness.candidate_selection_policy
        or recursive_harness.contrastive_training_evidence_policy
        != no_recursive_harness.contrastive_training_evidence_policy
    ):
        raise ValueError("paired arms must share training evidence and selection policies")
    shared_trace = f"skilllearn-paired-{manifest_hash[:12]}-g1"
    replay_cache = CounterfactualEvidenceReplayCache(
        event_sink=recursive_harness.event_sink
    )
    training_replay_cache = TrainingEvidenceReplayCache(
        event_sink=recursive_harness.event_sink
    )
    observations = recursive_harness.collect_training_observations(
        train_item_ids=train_ids,
        training_replay_cache=training_replay_cache,
        trace_id=f"{shared_trace}:shared-train",
    )
    residuals = recursive_harness.residual_miner.mine(
        observations,
        trace_id=f"{shared_trace}:shared-residuals",
    )
    proposal_error: HypothesisProposalCallError | None = None
    try:
        proposals = (
            recursive_harness.propose_candidates(
                residuals,
                trace_id=f"{shared_trace}:shared-root",
            )
            if any(not row.baseline_success for row in residuals)
            else ()
        )
    except HypothesisProposalCallError as exc:
        proposal_error = exc
        proposals = ()
    labeled_transition_ids = sorted(row.transition_id for row in residuals)
    checkpoint_descriptor: dict[str, Any] = {
        "observation_hashes": [row.observation_hash for row in observations],
        "transition_ids": labeled_transition_ids,
        "proposal_hashes": [row.payload_hash for row in proposals],
        "manifest_hash": manifest_hash,
    }
    if recursive_harness.contrastive_training_evidence_policy:
        checkpoint_descriptor.update(
            {
                "labeled_transition_ids": labeled_transition_ids,
                "contrastive_training_evidence_policy": (
                    recursive_harness.contrastive_training_evidence_policy
                ),
                "candidate_selection_policy": (
                    recursive_harness.candidate_selection_policy
                ),
            }
        )
    checkpoint_hash = stable_hash(checkpoint_descriptor)
    failure_count = sum(not row.baseline_success for row in residuals)
    success_control_count = sum(row.baseline_success for row in residuals)
    checkpoint_payload: dict[str, Any] = {
        "checkpoint_hash": checkpoint_hash,
        "observation_count": len(observations),
        "residual_count": failure_count,
        "success_control_count": success_control_count,
        "example_count": len(residuals),
        "proposal_count": len(proposals),
        "observation_set_hash": stable_hash(
            {"hashes": sorted(row.observation_hash for row in observations)}
        ),
        "transition_set_hash": stable_hash({"ids": labeled_transition_ids}),
        "proposal_set_hash": stable_hash(
            {"hashes": sorted(row.payload_hash for row in proposals)}
        ),
        "test_content_accessed": False,
        "raw_content_persisted": False,
    }
    if recursive_harness.contrastive_training_evidence_policy:
        checkpoint_payload.update(
            {
                "labeled_transition_ids": labeled_transition_ids,
                "contrastive_training_evidence_policy": (
                    recursive_harness.contrastive_training_evidence_policy
                ),
                "candidate_selection_policy": (
                    recursive_harness.candidate_selection_policy
                ),
            }
        )
    recursive_harness.event_sink.emit(
        Event(
            event="skilllearn_paired_ablation_checkpoint_frozen",
            stage="benchmark.skilllearn.paired_ablation",
            trace_id=shared_trace,
            payload=checkpoint_payload,
        )
    )
    if proposal_error is not None:
        recursive_generation = recursive_harness.record_proposal_failure(
            observations=observations,
            residuals=residuals,
            error=proposal_error,
            trace_id=f"{shared_trace}:recursive",
        )
        no_recursive_generation = no_recursive_harness.record_proposal_failure(
            observations=observations,
            residuals=residuals,
            error=proposal_error,
            trace_id=f"{shared_trace}:no-recursive",
        )
        return {
            "checkpoint_hash": checkpoint_hash,
            "recursive_generations": [recursive_generation],
            "recursive_stop_reason": "proposal_model_failure",
            "no_recursive_generations": [no_recursive_generation],
            "no_recursive_stop_reason": "proposal_model_failure",
        }
    recursive_generations = [
        recursive_harness.run_generation_from_evidence(
            observations=observations,
            residuals=residuals,
            validation_item_ids=validation_ids,
            proposal_candidates=proposals,
            counterfactual_replay_cache=replay_cache,
            trace_id=f"{shared_trace}:recursive",
        )
    ]
    no_recursive_generations = [
        no_recursive_harness.run_generation_from_evidence(
            observations=observations,
            residuals=residuals,
            validation_item_ids=validation_ids,
            proposal_candidates=proposals,
            counterfactual_replay_cache=replay_cache,
            trace_id=f"{shared_trace}:no-recursive",
        )
    ]
    recursive_active, recursive_consecutive, recursive_stop = _advance_arm(
        recursive_generations[0],
        consecutive_non_promotions=0,
        maximum=max_consecutive_non_promotions,
        counterfactual_invalid_evidence_policy=(
            counterfactual_invalid_evidence_policy
        ),
    )
    no_recursive_active, no_recursive_consecutive, no_recursive_stop = _advance_arm(
        no_recursive_generations[0],
        consecutive_non_promotions=0,
        maximum=max_consecutive_non_promotions,
        counterfactual_invalid_evidence_policy=(
            counterfactual_invalid_evidence_policy
        ),
    )
    for generation_index in range(2, max_generations + 1):
        if recursive_active:
            generation = recursive_harness.run_generation(
                train_item_ids=train_ids,
                validation_item_ids=validation_ids,
                training_replay_cache=training_replay_cache,
                counterfactual_replay_cache=replay_cache,
                trace_id=(
                    f"skilllearn-paired-{manifest_hash[:12]}-recursive-g{generation_index}"
                ),
            )
            recursive_generations.append(generation)
            recursive_active, recursive_consecutive, recursive_stop = _advance_arm(
                generation,
                consecutive_non_promotions=recursive_consecutive,
                maximum=max_consecutive_non_promotions,
                counterfactual_invalid_evidence_policy=(
                    counterfactual_invalid_evidence_policy
                ),
            )
        if no_recursive_active:
            generation = no_recursive_harness.run_generation(
                train_item_ids=train_ids,
                validation_item_ids=validation_ids,
                training_replay_cache=training_replay_cache,
                counterfactual_replay_cache=replay_cache,
                trace_id=(
                    f"skilllearn-paired-{manifest_hash[:12]}-no-recursive-g{generation_index}"
                ),
            )
            no_recursive_generations.append(generation)
            no_recursive_active, no_recursive_consecutive, no_recursive_stop = _advance_arm(
                generation,
                consecutive_non_promotions=no_recursive_consecutive,
                maximum=max_consecutive_non_promotions,
                counterfactual_invalid_evidence_policy=(
                    counterfactual_invalid_evidence_policy
                ),
            )
        if not recursive_active and not no_recursive_active:
            break
    if recursive_active:
        recursive_stop = "max_generations_reached"
    if no_recursive_active:
        no_recursive_stop = "max_generations_reached"
    return {
        "checkpoint_hash": checkpoint_hash,
        "recursive_generations": recursive_generations,
        "recursive_stop_reason": recursive_stop,
        "no_recursive_generations": no_recursive_generations,
        "no_recursive_stop_reason": no_recursive_stop,
    }


def _advance_arm(
    generation: SkillLearnGenerationResult,
    *,
    consecutive_non_promotions: int,
    maximum: int,
    counterfactual_invalid_evidence_policy: str | None = None,
) -> tuple[bool, int, str]:
    generation_row = generation.to_dict()
    if generation_row["proposal_model_failure_count"]:
        return False, consecutive_non_promotions, "proposal_model_failure"
    if (
        counterfactual_invalid_evidence_policy
        == COUNTERFACTUAL_INVALID_EVIDENCE_POLICY_VERSION
        and any(_generation_counterfactual_evidence_counts(generation_row))
    ):
        return (
            False,
            consecutive_non_promotions,
            "invalid_counterfactual_evidence",
        )
    if generation.reason in {
        "no_valid_failed_training_rows",
        "duplicate_hypothesis_behavior",
    }:
        return False, consecutive_non_promotions, generation.reason
    consecutive = (
        0
        if generation.evolution and generation.evolution.promoted
        else consecutive_non_promotions + 1
    )
    if consecutive >= maximum:
        return False, consecutive, "consecutive_non_promotion_limit"
    return True, consecutive, "max_generations_reached"


def _execution_report(
    *,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    generations: Sequence[SkillLearnGenerationResult],
    stop_reason: str,
    archive: PolicyArchive,
    archive_path: Path,
    guard: SplitAccessGuard,
) -> dict[str, Any]:
    if not generations:
        raise ValueError("execution report requires at least one generation")
    strict_counterfactual_claim_binding = (
        plan.get("counterfactual_invalid_evidence_policy")
        == COUNTERFACTUAL_INVALID_EVIDENCE_POLICY_VERSION
    )
    generation_rows = [row.to_dict() for row in generations]
    if not strict_counterfactual_claim_binding:
        for row in generation_rows:
            _strip_promotion_summary_diagnostics(row)
    proposal_model_failure_count = 0
    invalid_counterfactual_pair_count = 0
    counterfactual_provider_mismatch_count = 0
    counterfactual_budget_mismatch_count = 0
    for row in generation_rows:
        value = row.get("proposal_model_failure_count")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "generation proposal model failure count is malformed"
            )
        proposal_model_failure_count += value
        invalid_count, provider_count, budget_count = (
            _generation_counterfactual_evidence_counts(row)
        )
        invalid_counterfactual_pair_count += invalid_count
        counterfactual_provider_mismatch_count += provider_count
        counterfactual_budget_mismatch_count += budget_count
    invalid_counterfactual_evidence_present = any(
        (
            invalid_counterfactual_pair_count,
            counterfactual_provider_mismatch_count,
            counterfactual_budget_mismatch_count,
        )
    )
    performance_claim_eligible = proposal_model_failure_count == 0 and (
        not invalid_counterfactual_evidence_present
        or not strict_counterfactual_claim_binding
    )
    performance_claim_blockers: list[str] = []
    if proposal_model_failure_count:
        performance_claim_blockers.append(
            _PROPOSAL_MODEL_FAILURE_CLAIM_BLOCKER
        )
    if (
        strict_counterfactual_claim_binding
        and invalid_counterfactual_evidence_present
    ):
        performance_claim_blockers.append(
            _INVALID_COUNTERFACTUAL_EVIDENCE_CLAIM_BLOCKER
        )
    report = {
        "mode": "execute",
        "plan": dict(plan),
        "preflight": dict(preflight),
        "generation": dict(generation_rows[-1]),
        "generations": generation_rows,
        "generation_count": len(generation_rows),
        "evolution_stop_reason": stop_reason,
        "proposal_model_failure_count": proposal_model_failure_count,
        "proposal_model_failures_present": bool(proposal_model_failure_count),
        "performance_claim_eligible": performance_claim_eligible,
        "performance_claim_blockers": performance_claim_blockers,
        "archive_hash": archive.to_dict()["archive_hash"],
        "archive_path_hash": _path_hash(archive_path),
        "executed": True,
        "test_content_accessed": guard.test_accessed,
        "secret_value_persisted": False,
    }
    if strict_counterfactual_claim_binding:
        report.update(
            {
                "invalid_counterfactual_pair_count": (
                    invalid_counterfactual_pair_count
                ),
                "invalid_counterfactual_pairs_present": bool(
                    invalid_counterfactual_pair_count
                ),
                "counterfactual_provider_mismatch_count": (
                    counterfactual_provider_mismatch_count
                ),
                "counterfactual_budget_mismatch_count": (
                    counterfactual_budget_mismatch_count
                ),
            }
        )
    return report


def _strip_promotion_summary_diagnostics(generation: Mapping[str, Any]) -> None:
    summaries: list[Mapping[str, Any]] = []
    reported_summary = generation.get("promotion_summary")
    if isinstance(reported_summary, Mapping):
        summaries.append(reported_summary)
    decision = generation.get("promotion_decision")
    if isinstance(decision, Mapping):
        decision_summary = decision.get("summary")
        if isinstance(decision_summary, Mapping):
            summaries.append(decision_summary)
    for summary in summaries:
        if not isinstance(summary, dict):
            raise ValueError("generation promotion summary must be mutable")
        for key in _PROMOTION_SUMMARY_DIAGNOSTIC_KEYS:
            summary.pop(key, None)


def _generation_counterfactual_evidence_counts(
    generation: Mapping[str, Any],
) -> tuple[int, int, int]:
    reported_summary = generation.get("promotion_summary")
    decision = generation.get("promotion_decision")
    if decision is None:
        if reported_summary is not None:
            raise ValueError("generation promotion summary has no decision")
        return 0, 0, 0
    if not isinstance(decision, Mapping):
        raise ValueError("generation promotion decision is malformed")
    summary = decision.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("generation promotion summary is malformed")
    if reported_summary is not None and reported_summary != summary:
        raise ValueError("generation promotion summary mismatch")
    counts: list[int] = []
    for field in (
        "invalid_pair_count",
        "provider_mismatch_count",
        "budget_mismatch_count",
    ):
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"generation promotion summary {field} is malformed"
            )
        counts.append(value)
    return counts[0], counts[1], counts[2]


def _experiment_phase_name(
    protocol: PaperProtocol,
    *,
    manifest: SplitManifest,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
) -> str | None:
    full_phase = (
        "family_out_development"
        if manifest.protocol == "family_out"
        else "development"
    )
    full = protocol.payload["phases"][full_phase]
    if (
        tuple(train_ids) == manifest.train_ids
        and tuple(validation_ids) == manifest.validation_ids
        and len(train_ids) == int(full["train_count"])
        and len(validation_ids) == int(full["validation_count"])
    ):
        return full_phase
    smoke = protocol.payload["phases"]["smoke"]
    smoke_train_count = int(smoke["train_count"])
    smoke_validation_count = int(smoke["validation_count"])
    if (
        tuple(train_ids) == manifest.train_ids[:smoke_train_count]
        and tuple(validation_ids)
        == manifest.validation_ids[:smoke_validation_count]
    ):
        return "smoke"
    return None


def _select_ids(selected: Sequence[str], allowed: Sequence[str], limit: int | None) -> tuple[str, ...]:
    values = tuple(selected) if selected else tuple(allowed)
    unexpected = sorted(set(values) - set(allowed))
    if unexpected:
        raise PermissionError("selected item IDs are outside the frozen split")
    if limit is not None:
        if limit <= 0:
            raise ValueError("split limits must be positive")
        values = values[:limit]
    if not values:
        raise ValueError("experiment split selection cannot be empty")
    return values


def _item_set_hash(item_ids: Sequence[str]) -> str:
    return stable_hash({"item_ids": sorted(item_ids)})


def _path_hash(path: Path) -> str:
    return stable_hash({"path": str(path)})


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
