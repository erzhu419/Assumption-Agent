from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..archive import PolicyArchive
from ..evaluation import PromotionGate, PromotionGateSpec
from ..events import Event, JsonlEventSink
from ..evolution import (
    COUNTERFACTUAL_REPLAY_POLICY_VERSION,
    TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
    CounterfactualEvidenceReplayCache,
)
from ..models import stable_hash
from ..provider_chain import build_proposal_model, proposal_provider_status
from ..proposer import StructuredHypothesisProposer
from ..secure_env import (
    alternate_model_allowed,
    configured_model,
    configured_skilllearn_provider_mode,
    load_dotenv,
    map_legacy_model_env,
    paper_model_allowed,
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
from .prewarm import (
    DEVELOPMENT_PREWARM_VERSION,
    validate_development_prewarm_receipt,
)
from .skilllearn_compiler import SKILL_ROUTING_VERSION
from .skilllearn_lifecycle import (
    EPHEMERAL_AUTH_CLEANUP_VERSION,
    OPENAI_COMPATIBLE_CODEX_CONFIG_VERSION,
    PREBUILT_IMAGE_POLICY_VERSION,
    PROVIDER_FAILURE_POLICY_VERSION,
    PROVIDER_ROUTE_POLICY_VERSION,
    RUNNER_AGENT_REGISTRY_ISOLATION_VERSION,
    TRAINING_EVIDENCE_POLICY_VERSION,
    TRIAL_TIMEOUT_POLICY_VERSION,
    SkillLearnBackendPool,
    SkillLearnEvolutionHarness,
    SkillLearnGenerationResult,
    SkillLearnPrebuiltImageCache,
    SkillLearnProviderCircuit,
    SkillLearnSubprocessBackend,
)
from .skilllearnbench import SkillLearnBenchAdapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or execute one guarded SkillLearnBench self-evolution generation."
    )
    parser.add_argument("--root", type=Path, required=True)
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
    parser.add_argument("--agent-id", default="codex")
    parser.add_argument("--model", default="gpt-5.3-codex-spark")
    parser.add_argument(
        "--trial-provider-mode",
        choices=("codex_subscription", "openai_compatible"),
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--minimum-pairs", type=int, default=10)
    parser.add_argument("--minimum-trigger-support", type=int, default=2)
    parser.add_argument("--max-generations", type=int, default=1)
    parser.add_argument("--max-consecutive-non-promotions", type=int, default=1)
    parser.add_argument("--proposal-candidates-per-generation", type=int, default=3)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--disable-recursive-repair", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    paired_ablation = bool(
        args.paired_no_recursive_out and args.paired_no_recursive_archive_out
    )

    load_dotenv(args.env_file)
    model_presence = map_legacy_model_env()
    provider_status = proposal_provider_status()
    trial_provider_mode = args.trial_provider_mode or configured_skilllearn_provider_mode()
    manifest = SplitManifest.read(args.manifest)
    prewarm_receipt_hash: str | None = None
    if args.prewarm_receipt:
        prewarm_payload = json.loads(args.prewarm_receipt.read_text(encoding="utf-8"))
        if not isinstance(prewarm_payload, Mapping):
            raise ValueError("development prewarm receipt must contain one JSON object")
        prewarm_receipt_hash = validate_development_prewarm_receipt(
            prewarm_payload,
            manifest=manifest,
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
        "minimum_pairs": args.minimum_pairs,
        "minimum_trigger_support": args.minimum_trigger_support,
        "recursive_validation_enabled": not args.disable_recursive_repair,
        "max_generations": args.max_generations,
        "max_consecutive_non_promotions": args.max_consecutive_non_promotions,
        "proposal_candidates_per_generation": args.proposal_candidates_per_generation,
        "agent_id": args.agent_id,
        "model": args.model,
        "trial_provider_mode": trial_provider_mode,
        "max_steps": args.max_steps,
        "parallel_workers": args.parallel_workers,
        "prebuilt_image_policy": PREBUILT_IMAGE_POLICY_VERSION,
        "runner_agent_registry_isolation": RUNNER_AGENT_REGISTRY_ISOLATION_VERSION,
        "trial_timeout_policy": TRIAL_TIMEOUT_POLICY_VERSION,
        "provider_failure_policy": PROVIDER_FAILURE_POLICY_VERSION,
        "provider_route_policy": PROVIDER_ROUTE_POLICY_VERSION,
        "counterfactual_replay_policy": COUNTERFACTUAL_REPLAY_POLICY_VERSION,
        "openai_compatible_codex_config": (
            OPENAI_COMPATIBLE_CODEX_CONFIG_VERSION
            if trial_provider_mode == "openai_compatible"
            else None
        ),
        "ephemeral_auth_cleanup": EPHEMERAL_AUTH_CLEANUP_VERSION,
        "training_evidence_policy": TRAINING_EVIDENCE_POLICY_VERSION,
        "development_prewarm_version": DEVELOPMENT_PREWARM_VERSION,
        "prewarm_passed": prewarm_receipt_hash is not None,
        "prewarm_receipt_hash": prewarm_receipt_hash,
        "proposal_candidate_selection": TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
        "runtime_candidate_kinds": ["task", "policy"],
        "evaluator_hypothesis_mode": (
            "separate_epoch_challenger_not_in_primary_runtime"
        ),
        "skill_routing": SKILL_ROUTING_VERSION,
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
    if not paper_model_allowed(args.model) and not alternate_model_allowed():
        blockers.append("execution_model_not_protocol_approved")
    if len(validation_ids) < args.minimum_pairs:
        blockers.append("minimum_pairs_exceeds_validation_selection")
    if len(train_ids) < args.minimum_trigger_support:
        blockers.append("minimum_trigger_support_exceeds_training_selection")
    if args.max_generations <= 0:
        blockers.append("max_generations_must_be_positive")
    if args.max_consecutive_non_promotions <= 0:
        blockers.append("max_consecutive_non_promotions_must_be_positive")
    if args.proposal_candidates_per_generation <= 0:
        blockers.append("proposal_candidates_per_generation_must_be_positive")
    if args.parallel_workers <= 0:
        blockers.append("parallel_workers_must_be_positive")
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

    if configured_model() != args.model:
        raise RuntimeError("proposal model and benchmark model must match")
    proposal_model = build_proposal_model(event_sink=sink)
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
            TrainingSupportCheck(min_support=args.minimum_trigger_support),
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
    backends = tuple(
        SkillLearnSubprocessBackend(
            args.root,
            agent_id=args.agent_id,
            model=args.model,
            max_steps=args.max_steps,
            provider_mode=trial_provider_mode,
            trials_dir=args.work_dir / "upstream_trials",
            prebuilt_cache=prebuilt_cache,
            provider_circuit=provider_circuit,
            event_sink=sink,
        )
        for _ in range(args.parallel_workers)
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
            PromotionGateSpec(
                minimum_pairs=args.minimum_pairs,
                confidence=0.9,
                minimum_net_gain_count=1,
                minimum_activation_rate=0.1,
            ),
            event_sink=sink,
        ),
        archive=archive,
        evaluator_epoch=f"skilllearn-eval-{manifest.manifest_hash[:12]}",
        output_root=args.work_dir / "compiled_skills",
        proposal_candidates_per_generation=args.proposal_candidates_per_generation,
        parallel_workers=args.parallel_workers,
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
                    TrainingSupportCheck(min_support=args.minimum_trigger_support),
                    RuntimeActionCheck(),
                    EvaluatorEpochCheck(),
                ],
                proposer=None,
                event_sink=sink,
            ),
            promotion_gate=PromotionGate(
                PromotionGateSpec(
                    minimum_pairs=args.minimum_pairs,
                    confidence=0.9,
                    minimum_net_gain_count=1,
                    minimum_activation_rate=0.1,
                ),
                event_sink=sink,
            ),
            archive=no_recursive_archive,
            evaluator_epoch=f"skilllearn-eval-{manifest.manifest_hash[:12]}",
            output_root=args.work_dir / "compiled_skills_no_recursive",
            proposal_candidates_per_generation=args.proposal_candidates_per_generation,
            parallel_workers=args.parallel_workers,
            event_sink=sink,
        )
        paired = _run_paired_arms(
            recursive_harness=harness,
            no_recursive_harness=no_recursive_harness,
            train_ids=train_ids,
            validation_ids=validation_ids,
            manifest_hash=manifest.manifest_hash,
            max_generations=args.max_generations,
            max_consecutive_non_promotions=args.max_consecutive_non_promotions,
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
        max_generations=args.max_generations,
        max_consecutive_non_promotions=args.max_consecutive_non_promotions,
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
) -> tuple[list[SkillLearnGenerationResult], str]:
    generations: list[SkillLearnGenerationResult] = []
    consecutive = 0
    stop_reason = "max_generations_reached"
    for generation_index in range(1, max_generations + 1):
        generation = harness.run_generation(
            train_item_ids=train_ids,
            validation_item_ids=validation_ids,
            trace_id=f"skilllearn-generation-{manifest_hash[:12]}-g{generation_index}",
        )
        generations.append(generation)
        active, consecutive, reason = _advance_arm(
            generation,
            consecutive_non_promotions=consecutive,
            maximum=max_consecutive_non_promotions,
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
) -> dict[str, Any]:
    shared_trace = f"skilllearn-paired-{manifest_hash[:12]}-g1"
    replay_cache = CounterfactualEvidenceReplayCache(
        event_sink=recursive_harness.event_sink
    )
    observations = recursive_harness.collect_training_observations(
        train_item_ids=train_ids,
        trace_id=f"{shared_trace}:shared-train",
    )
    residuals = recursive_harness.residual_miner.mine(
        observations,
        trace_id=f"{shared_trace}:shared-residuals",
    )
    proposals = (
        recursive_harness.propose_candidates(
            residuals,
            trace_id=f"{shared_trace}:shared-root",
        )
        if residuals
        else ()
    )
    checkpoint_hash = stable_hash(
        {
            "observation_hashes": [row.observation_hash for row in observations],
            "transition_ids": sorted(row.transition_id for row in residuals),
            "proposal_hashes": [row.payload_hash for row in proposals],
            "manifest_hash": manifest_hash,
        }
    )
    recursive_harness.event_sink.emit(
        Event(
            event="skilllearn_paired_ablation_checkpoint_frozen",
            stage="benchmark.skilllearn.paired_ablation",
            trace_id=shared_trace,
            payload={
                "checkpoint_hash": checkpoint_hash,
                "observation_count": len(observations),
                "residual_count": len(residuals),
                "proposal_count": len(proposals),
                "observation_set_hash": stable_hash(
                    {"hashes": sorted(row.observation_hash for row in observations)}
                ),
                "transition_set_hash": stable_hash(
                    {"ids": sorted(row.transition_id for row in residuals)}
                ),
                "proposal_set_hash": stable_hash(
                    {"hashes": sorted(row.payload_hash for row in proposals)}
                ),
                "test_content_accessed": False,
                "raw_content_persisted": False,
            },
        )
    )
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
    )
    no_recursive_active, no_recursive_consecutive, no_recursive_stop = _advance_arm(
        no_recursive_generations[0],
        consecutive_non_promotions=0,
        maximum=max_consecutive_non_promotions,
    )
    for generation_index in range(2, max_generations + 1):
        if recursive_active:
            generation = recursive_harness.run_generation(
                train_item_ids=train_ids,
                validation_item_ids=validation_ids,
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
            )
        if no_recursive_active:
            generation = no_recursive_harness.run_generation(
                train_item_ids=train_ids,
                validation_item_ids=validation_ids,
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
) -> tuple[bool, int, str]:
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
    return {
        "mode": "execute",
        "plan": dict(plan),
        "preflight": dict(preflight),
        "generation": generations[-1].to_dict(),
        "generations": [row.to_dict() for row in generations],
        "generation_count": len(generations),
        "evolution_stop_reason": stop_reason,
        "archive_hash": archive.to_dict()["archive_hash"],
        "archive_path_hash": _path_hash(archive_path),
        "executed": True,
        "test_content_accessed": guard.test_accessed,
        "secret_value_persisted": False,
    }


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
