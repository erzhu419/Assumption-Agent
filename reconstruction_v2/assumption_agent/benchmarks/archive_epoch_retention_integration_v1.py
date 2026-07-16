"""Zero-model integration for archive retention and evaluator epochs.

This is a synthetic mechanism diagnostic, not a performance benchmark.  It
isolates archive retention from recursive repair and exercises the existing
promotion, runtime, evaluator-transition, selective-invalidation, and archive
persistence paths without touching benchmark data or a model provider.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..archive import (
    AnchorScore,
    ArchiveNode,
    ArchiveNodeStatus,
    EvaluatorEpoch,
    EvaluatorEpochController,
    EvaluatorSpec,
    PolicyArchive,
    ScoreRecord,
)
from ..evaluation import (
    CANDIDATE_MAY_ONLY_TIGHTEN,
    RUNTIME_BASELINE_PRESERVATION,
    CounterfactualRunner,
    PromotionGate,
    PromotionGateSpec,
)
from ..events import MemoryEventSink
from ..models import (
    ActionNode,
    ExpectedEffect,
    ExternalOutcome,
    FeaturePredicate,
    HypothesisKind,
    HypothesisProgram,
    HypothesisStatus,
    LaneResult,
    SplitName,
    TaskInput,
    TriggerSpec,
    VerifierContract,
    stable_hash,
)
from ..runtime import LaneRegistry, PolicyRuntime


INTEGRATION_VERSION = "archive_epoch_retention_integration_v1"
REEVALUATION_POLICY = "same_behavior_rebased_new_epoch_full_fixed_item_set_v2"
RETENTION_ESTIMAND = "Y(P_plus_Q)-Y(Q)"
ANCHOR_POLICY = "deterministic_matched_synthetic_anchor_execution_v1"


@dataclass(frozen=True)
class _StaticLane:
    name: str
    answer: str
    confidence: float
    cost: float = 1.0

    def run(
        self,
        task: TaskInput,
        parameters: Mapping[str, Any],
    ) -> LaneResult:
        return LaneResult(
            lane=self.name,
            answer=self.answer,
            confidence=self.confidence,
            cost=self.cost,
            metadata={
                "provider_fingerprint": "synthetic-offline-v1",
                "fairness_fingerprint": "synthetic-budget-v1",
            },
        )


@dataclass(frozen=True)
class _ExactEvaluator:
    id: str
    epoch: str
    match_policy: str = "exact_v1"

    def matches(self, predicted: object, expected: object) -> bool:
        left = str(predicted)
        right = str(expected)
        if self.match_policy == "exact_v1":
            return left == right
        if self.match_policy == "trim_casefold_v1":
            return left.strip().casefold() == right.strip().casefold()
        raise ValueError("unknown synthetic evaluator match policy")

    def evaluate(
        self,
        task: TaskInput,
        execution: Any,
    ) -> ExternalOutcome:
        success = self.matches(
            execution.selected_result.answer,
            task.payload["expected"],
        )
        return ExternalOutcome(
            task_id=task.id,
            success=success,
            score=float(success),
            evaluator_id=self.id,
            evaluator_epoch=self.epoch,
            metrics={
                "task_success": float(success),
                "evaluation_valid": 1.0,
            },
        )


def _program(
    program_id: str,
    lane: str,
    *,
    priority: int,
    epoch: str = "epoch-0",
    status: HypothesisStatus = HypothesisStatus.CANDIDATE,
) -> HypothesisProgram:
    return HypothesisProgram(
        id=program_id,
        kind=HypothesisKind.POLICY,
        statement="Use one frozen synthetic lane for the declared relation family.",
        trigger=TriggerSpec(
            all_of=(FeaturePredicate("family", "eq", "retention_relation"),)
        ),
        anti_trigger=TriggerSpec(),
        action_graph=(
            ActionNode("enable", "enable_lane", lane),
            ActionNode(
                "prioritize",
                "prioritize_lane",
                lane,
                priority,
                ("enable",),
            ),
        ),
        expected_effect=ExpectedEffect(
            metric="task_success",
            minimum_delta=0.0,
            maximum_harm_rate=0.0,
            maximum_cost_ratio=3.0,
        ),
        verifier=VerifierContract(
            checks=("synthetic_exact_match",),
            anchor_id="synthetic-retention-anchor-v1",
            repair_on_failure=False,
            max_repair_depth=0,
        ),
        evaluator_epoch=epoch,
        fallback="preserve_baseline",
        status=status,
    )


def _program_behavior_hash(program: HypothesisProgram) -> str:
    payload = program.to_dict()
    for field in ("id", "evaluator_epoch", "status"):
        payload.pop(field, None)
    return stable_hash(payload)


def _tasks(prefix: str, count: int = 4) -> tuple[TaskInput, ...]:
    return tuple(
        TaskInput(
            id=f"{prefix}-{index}",
            family="retention_relation",
            features={"family": "retention_relation"},
            payload={"expected": "retained-answer"},
        )
        for index in range(count)
    )


def _fixed_evaluator_anchor() -> tuple[dict[str, str], ...]:
    """Return matched synthetic rows that distinguish the two evaluators."""

    return (
        {"row_id": "a0", "prediction": "Alpha", "expected": "Alpha"},
        {"row_id": "a1", "prediction": " beta ", "expected": "beta"},
        {"row_id": "a2", "prediction": "GAMMA", "expected": "gamma"},
        {"row_id": "a3", "prediction": "Delta", "expected": "Delta"},
        {"row_id": "a4", "prediction": "EPSILON ", "expected": "epsilon"},
        {"row_id": "a5", "prediction": "zeta", "expected": "zeta"},
        {"row_id": "a6", "prediction": "Eta", "expected": "ETA"},
        {"row_id": "a7", "prediction": "theta", "expected": "theta"},
    )


def _score_fixed_anchor(
    evaluator: _ExactEvaluator,
    spec: EvaluatorSpec,
    rows: Sequence[Mapping[str, str]],
) -> AnchorScore:
    return AnchorScore(
        evaluator_id=spec.id,
        anchor_manifest_hash=spec.anchor_manifest_hash,
        successes=sum(
            evaluator.matches(row["prediction"], row["expected"])
            for row in rows
        ),
        total=len(rows),
    )


def _promotion_spec() -> PromotionGateSpec:
    return PromotionGateSpec(
        metric="task_success",
        minimum_pairs=4,
        confidence=0.9,
        minimum_net_gain_count=4,
        minimum_activation_rate=1.0,
        minimum_effect_lower_bound=0.0,
        maximum_harm_rate=0.0,
        maximum_cost_ratio=3.0,
        baseline_safety_policy=RUNTIME_BASELINE_PRESERVATION,
        candidate_threshold_policy=CANDIDATE_MAY_ONLY_TIGHTEN,
    )


def _runtime(event_sink: MemoryEventSink) -> PolicyRuntime:
    return PolicyRuntime(
        registry=LaneRegistry(
            (
                _StaticLane("raw", "raw-answer", 0.90),
                _StaticLane("retained", "retained-answer", 0.95),
                _StaticLane("neutral", "neutral-answer", 0.92),
            )
        ),
        baseline_lane="raw",
        event_sink=event_sink,
        runtime_version="synthetic-retention-runtime-v1",
    )


def _score_executions(
    *,
    runtime: PolicyRuntime,
    evaluator: _ExactEvaluator,
    tasks: Sequence[TaskInput],
    programs: Sequence[HypothesisProgram],
    trace_prefix: str,
) -> dict[str, Any]:
    executions = tuple(
        runtime.execute(
            task,
            programs,
            allowed_statuses={
                HypothesisStatus.CANDIDATE,
                HypothesisStatus.SHADOW,
                HypothesisStatus.PROMOTED,
            },
            trace_id=f"{trace_prefix}:{index}",
        )
        for index, task in enumerate(tasks)
    )
    outcomes = tuple(
        evaluator.evaluate(task, execution)
        for task, execution in zip(tasks, executions)
    )
    return {
        "success_count": sum(row.success for row in outcomes),
        "total": len(outcomes),
        "activation_count": sum(row.action_activated for row in executions),
        "selected_lane_set_hash": stable_hash(
            sorted(row.selected_result.lane for row in executions)
        ),
        "execution_set_hash": stable_hash(
            [
                {
                    "task_id_hash": stable_hash({"task_id": task.id}),
                    "plan_hash": execution.plan_hash,
                    "activated_hypothesis_ids": list(
                        execution.activated_hypothesis_ids
                    ),
                    "success": outcome.success,
                }
                for task, execution, outcome in zip(tasks, executions, outcomes)
            ]
        ),
    }


def _restore_archive(
    payload: Mapping[str, Any],
    *,
    event_sink: MemoryEventSink | None = None,
) -> PolicyArchive:
    """Restore the public archive schema for an exact checkpoint fork.

    The function is intentionally local to this versioned diagnostic.  It does
    not add a second production archive loader or a promotion surface.
    """

    if payload.get("raw_content_persisted") is not False:
        raise ValueError("archive checkpoint safety marker is missing")
    archive = PolicyArchive(event_sink=event_sink)
    hypotheses = payload.get("hypotheses")
    nodes = payload.get("nodes")
    scores = payload.get("score_records")
    if not all(isinstance(value, Mapping) for value in (hypotheses, nodes, scores)):
        raise ValueError("archive checkpoint mappings are malformed")
    archive.hypotheses = {
        str(key): HypothesisProgram.from_dict(value)
        for key, value in hypotheses.items()
        if isinstance(value, Mapping)
    }
    if set(archive.hypotheses) != set(hypotheses):
        raise ValueError("archive checkpoint hypothesis rows are malformed")
    archive.nodes = {
        str(key): ArchiveNode(
            id=str(value["id"]),
            parent_id=(
                str(value["parent_id"]) if value.get("parent_id") else None
            ),
            active_hypothesis_ids=tuple(
                str(item) for item in value["active_hypothesis_ids"]
            ),
            evaluator_epoch_id=str(value["evaluator_epoch_id"]),
            runtime_version=str(value["runtime_version"]),
            generation=int(value["generation"]),
            status=ArchiveNodeStatus(str(value["status"])),
        )
        for key, value in nodes.items()
        if isinstance(value, Mapping)
    }
    if set(archive.nodes) != set(nodes):
        raise ValueError("archive checkpoint node rows are malformed")
    archive.score_records = {
        str(key): ScoreRecord(
            id=str(value["id"]),
            archive_node_id=str(value["archive_node_id"]),
            split=str(value["split"]),
            evaluator_epoch_id=str(value["evaluator_epoch_id"]),
            metric=str(value["metric"]),
            successes=int(value["successes"]),
            total=int(value["total"]),
            item_set_hash=str(value["item_set_hash"]),
            valid=bool(value["valid"]),
            invalidation_reason=str(value["invalidation_reason"]),
        )
        for key, value in scores.items()
        if isinstance(value, Mapping)
    }
    if set(archive.score_records) != set(scores):
        raise ValueError("archive checkpoint score rows are malformed")
    incumbent_id = payload.get("incumbent_id")
    archive.incumbent_id = str(incumbent_id) if incumbent_id else None
    archive.typed_bindings = {
        str(key): dict(value)
        for key, value in dict(payload.get("typed_bindings") or {}).items()
    }
    archive.typed_selection_history = {
        str(key): dict(value)
        for key, value in dict(
            payload.get("typed_selection_history") or {}
        ).items()
    }
    # JSON persistence changes tuples to lists but not the canonical byte/hash
    # identity used by the archive contract.
    if stable_hash(archive.to_dict()) != stable_hash(dict(payload)):
        raise ValueError("archive checkpoint round trip drifted")
    return archive


def _current_epoch_incumbent_ready(
    archive: PolicyArchive,
    *,
    epoch_id: str,
    metric: str,
) -> bool:
    if archive.incumbent_id is None:
        return False
    incumbent = archive.nodes.get(archive.incumbent_id)
    if incumbent is None or incumbent.evaluator_epoch_id != epoch_id:
        return False
    return any(
        record.archive_node_id == archive.incumbent_id
        and record.evaluator_epoch_id == epoch_id
        and record.metric == metric
        and record.valid
        and record.total > 0
        for record in archive.score_records.values()
    )


def run_integration(output_dir: str | Path | None = None) -> dict[str, Any]:
    sink = MemoryEventSink()
    runtime = _runtime(sink)
    old_evaluator = _ExactEvaluator("synthetic-evaluator-old", "epoch-0")
    runner = CounterfactualRunner(
        runtime=runtime,
        evaluator=old_evaluator,
        event_sink=sink,
    )
    archive = PolicyArchive(event_sink=sink)
    retained = _program("program-P", "retained", priority=10)
    generation_one_tasks = _tasks("generation-one")

    archive.register_hypothesis(retained, trace_id="g1")
    generation_one_node = archive.create_node(
        active_hypothesis_ids=(retained.id,),
        evaluator_epoch_id=old_evaluator.epoch,
        runtime_version=runtime.runtime_version,
        trace_id="g1",
    )
    generation_one_pairs = runner.run(
        generation_one_tasks,
        program=retained,
        split=SplitName.VALIDATION,
        trace_id="g1",
    )
    generation_one_decision = PromotionGate(
        _promotion_spec(), event_sink=sink
    ).evaluate(
        retained,
        generation_one_pairs,
        sealed_test_accessed=False,
        trace_id="g1",
    )
    generation_one_score = archive.record_score(
        archive_node_id=generation_one_node.id,
        split=SplitName.VALIDATION.value,
        evaluator_epoch_id=old_evaluator.epoch,
        metric="task_success",
        successes=generation_one_decision.summary.candidate_success_count,
        total=generation_one_decision.summary.pair_count,
        item_ids=tuple(task.id for task in generation_one_tasks),
    )
    archive.apply_promotion(
        candidate_node_id=generation_one_node.id,
        decision=generation_one_decision,
        trace_id="g1",
    )
    if not generation_one_decision.allowed or archive.incumbent_id is None:
        raise RuntimeError("synthetic generation one did not promote")

    # This objective does not depend on the replaceable evaluator epoch and is
    # the negative control for selective invalidation.
    independent_score = archive.record_score(
        archive_node_id=generation_one_node.id,
        split=SplitName.VALIDATION.value,
        evaluator_epoch_id="fixed-objective-v1",
        metric="runtime_integrity",
        successes=4,
        total=4,
        item_ids=tuple(task.id for task in generation_one_tasks),
    )

    checkpoint = archive.to_dict()
    retention_arm_archive = _restore_archive(checkpoint, event_sink=sink)
    no_retention_arm_archive = _restore_archive(checkpoint, event_sink=sink)
    if retention_arm_archive.to_dict() != no_retention_arm_archive.to_dict():
        raise RuntimeError("checkpoint forks differ before treatment")
    retained_from_checkpoint = retention_arm_archive.hypotheses[retained.id]
    challenger = _program("program-Q", "neutral", priority=5)
    generation_two_tasks = _tasks("generation-two")
    arms = {
        "empty": _score_executions(
            runtime=runtime,
            evaluator=old_evaluator,
            tasks=generation_two_tasks,
            programs=(),
            trace_prefix="g2-empty",
        ),
        "P": _score_executions(
            runtime=runtime,
            evaluator=old_evaluator,
            tasks=generation_two_tasks,
            programs=(retained_from_checkpoint,),
            trace_prefix="g2-P",
        ),
        "Q": _score_executions(
            runtime=runtime,
            evaluator=old_evaluator,
            tasks=generation_two_tasks,
            programs=(challenger,),
            trace_prefix="g2-Q",
        ),
        "P_plus_Q": _score_executions(
            runtime=runtime,
            evaluator=old_evaluator,
            tasks=generation_two_tasks,
            programs=(retained_from_checkpoint, challenger),
            trace_prefix="g2-PQ",
        ),
    }
    retention_effect_count = (
        arms["P_plus_Q"]["success_count"] - arms["Q"]["success_count"]
    )

    anchor_rows = _fixed_evaluator_anchor()
    anchor_manifest_hash = stable_hash(
        {
            "policy": ANCHOR_POLICY,
            "rows": list(anchor_rows),
        }
    )
    incumbent_spec = EvaluatorSpec(
        id=old_evaluator.id,
        version="1",
        implementation_hash=stable_hash(
            {"match_policy": old_evaluator.match_policy}
        ),
        criteria_hash=stable_hash({"criterion": "synthetic_string_match"}),
        anchor_manifest_hash=anchor_manifest_hash,
    )
    challenger_evaluator_for_anchor = _ExactEvaluator(
        "synthetic-evaluator-new",
        "anchor-probe",
        "trim_casefold_v1",
    )
    challenger_spec = EvaluatorSpec(
        id=challenger_evaluator_for_anchor.id,
        version="1",
        implementation_hash=stable_hash(
            {"match_policy": challenger_evaluator_for_anchor.match_policy}
        ),
        criteria_hash=incumbent_spec.criteria_hash,
        anchor_manifest_hash=anchor_manifest_hash,
    )
    incumbent_anchor_score = _score_fixed_anchor(
        old_evaluator,
        incumbent_spec,
        anchor_rows,
    )
    challenger_anchor_score = _score_fixed_anchor(
        challenger_evaluator_for_anchor,
        challenger_spec,
        anchor_rows,
    )
    epoch_controller = EvaluatorEpochController(
        EvaluatorEpoch("epoch-0", 0, incumbent_spec),
        confidence=0.9,
        event_sink=sink,
    )
    transition = epoch_controller.consider_challenger(
        challenger_spec,
        incumbent_score=incumbent_anchor_score,
        challenger_score=challenger_anchor_score,
        archive=retention_arm_archive,
        trace_id="evaluator-transition",
    )
    ready_before_reevaluation = _current_epoch_incumbent_ready(
        retention_arm_archive,
        epoch_id=transition.next_epoch.id,
        metric="task_success",
    )
    new_evaluator = _ExactEvaluator(
        challenger_spec.id,
        transition.next_epoch.id,
        challenger_evaluator_for_anchor.match_policy,
    )
    rebased_retained = replace(
        retained_from_checkpoint,
        id="program-P-epoch-1",
        evaluator_epoch=transition.next_epoch.id,
        status=HypothesisStatus.CANDIDATE,
    )
    retention_arm_archive.register_hypothesis(
        rebased_retained,
        trace_id="incumbent-reevaluation",
    )
    reevaluation_node = retention_arm_archive.create_node(
        active_hypothesis_ids=(rebased_retained.id,),
        evaluator_epoch_id=transition.next_epoch.id,
        runtime_version=runtime.runtime_version,
        parent_id=str(retention_arm_archive.incumbent_id),
        trace_id="incumbent-reevaluation",
    )
    reevaluation_runner = CounterfactualRunner(
        runtime=runtime,
        evaluator=new_evaluator,
        event_sink=sink,
    )
    reevaluation_pairs = reevaluation_runner.run(
        generation_two_tasks,
        program=rebased_retained,
        split=SplitName.VALIDATION,
        trace_id="incumbent-reevaluation",
    )
    reevaluation_decision = PromotionGate(
        _promotion_spec(), event_sink=sink
    ).evaluate(
        rebased_retained,
        reevaluation_pairs,
        sealed_test_accessed=False,
        trace_id="incumbent-reevaluation",
    )
    reevaluation_score = retention_arm_archive.record_score(
        archive_node_id=reevaluation_node.id,
        split=SplitName.VALIDATION.value,
        evaluator_epoch_id=transition.next_epoch.id,
        metric="task_success",
        successes=reevaluation_decision.summary.candidate_success_count,
        total=reevaluation_decision.summary.pair_count,
        item_ids=tuple(task.id for task in generation_two_tasks),
    )
    retention_arm_archive.apply_promotion(
        candidate_node_id=reevaluation_node.id,
        decision=reevaluation_decision,
        trace_id="incumbent-reevaluation",
    )
    ready_after_reevaluation = _current_epoch_incumbent_ready(
        retention_arm_archive,
        epoch_id=transition.next_epoch.id,
        metric="task_success",
    )

    final_archive = retention_arm_archive.to_dict()
    reloaded = _restore_archive(final_archive)
    reload_exact = reloaded.to_dict() == final_archive
    invariants = {
        "generation_one_promoted": generation_one_decision.allowed,
        "checkpoint_forks_exact": False,
        "retention_estimand_positive": retention_effect_count == 4,
        "retention_is_only_g2_treatment": (
            arms["empty"]["success_count"] == arms["Q"]["success_count"]
            and arms["P"]["success_count"]
            == arms["P_plus_Q"]["success_count"]
        ),
        "evaluator_challenger_promoted": transition.promoted,
        "evaluator_anchor_scores_executed": (
            incumbent_anchor_score.successes == 4
            and challenger_anchor_score.successes == 8
            and incumbent_anchor_score.total
            == challenger_anchor_score.total
            == len(anchor_rows)
        ),
        "old_epoch_dependent_score_invalidated": (
            not retention_arm_archive.score_records[generation_one_score.id].valid
            and generation_one_score.id
            in transition.invalidated_score_record_ids
        ),
        "independent_objective_preserved": retention_arm_archive.score_records[
            independent_score.id
        ].valid,
        "old_only_incumbent_not_epoch_ready": not ready_before_reevaluation,
        "incumbent_reevaluated_in_new_epoch": (
            ready_after_reevaluation
            and reevaluation_score.valid
            and reevaluation_score.evaluator_epoch_id
            == transition.next_epoch.id
            and reevaluation_decision.allowed
        ),
        "incumbent_rebased_to_new_epoch_node": (
            retention_arm_archive.incumbent_id == reevaluation_node.id
            and retention_arm_archive.nodes[
                reevaluation_node.id
            ].evaluator_epoch_id
            == transition.next_epoch.id
            and retention_arm_archive.nodes[
                generation_one_node.id
            ].status
            == ArchiveNodeStatus.SUPERSEDED
        ),
        "incumbent_behavior_retained_after_revalidation": (
            _program_behavior_hash(retained_from_checkpoint)
            == _program_behavior_hash(rebased_retained)
        ),
        "archive_reload_exact": reload_exact,
    }
    # The archive hash necessarily changes after invalidation and reevaluation;
    # compare the arms at the fork, not after the epoch intervention.
    invariants["checkpoint_forks_exact"] = (
        no_retention_arm_archive.to_dict() == checkpoint
        and _restore_archive(checkpoint).to_dict() == checkpoint
    )
    integration_passed = all(invariants.values())

    transition_payload = asdict(transition)
    transition_payload["invalidated_score_record_ids"] = list(
        transition.invalidated_score_record_ids
    )
    transition_payload["next_epoch"] = asdict(transition.next_epoch)
    report: dict[str, Any] = {
        "version": INTEGRATION_VERSION,
        "integration_passed": integration_passed,
        "diagnostic_only": True,
        "performance_gate": False,
        "claim_scope": (
            "synthetic_L0_integration_only_not_L4_or_L5_performance"
        ),
        "generation_one": {
            "candidate_program_hash": retained.payload_hash,
            "promotion_allowed": generation_one_decision.allowed,
            "promotion_decision_hash": stable_hash(
                generation_one_decision.to_dict()
            ),
            "pair_count": generation_one_decision.summary.pair_count,
            "gain_count": generation_one_decision.summary.gain_count,
            "harm_count": generation_one_decision.summary.harm_count,
            "incumbent_node_id": generation_one_node.id,
            "checkpoint_hash": checkpoint["archive_hash"],
        },
        "generation_two_retention": {
            "checkpoint_fork_policy": "same_archive_bytes_two_arms_v1",
            "repair_participated": False,
            "challenger_program_hash": challenger.payload_hash,
            "arm_counts": {
                key: {
                    "success_count": value["success_count"],
                    "total": value["total"],
                    "activation_count": value["activation_count"],
                    "execution_set_hash": value["execution_set_hash"],
                }
                for key, value in arms.items()
            },
            "estimand": RETENTION_ESTIMAND,
            "retention_effect_count": retention_effect_count,
        },
        "evaluator_epoch": {
            "transition": transition_payload,
            "anchor": {
                "policy": ANCHOR_POLICY,
                "manifest_hash": anchor_manifest_hash,
                "row_count": len(anchor_rows),
                "incumbent_successes": incumbent_anchor_score.successes,
                "challenger_successes": challenger_anchor_score.successes,
                "scores_computed_by_evaluator_implementations": True,
                "raw_rows_persisted": False,
            },
            "old_epoch_score_record_id": generation_one_score.id,
            "independent_score_record_id": independent_score.id,
            "reevaluation_score_record_id": reevaluation_score.id,
            "old_incumbent_node_id": generation_one_node.id,
            "rebased_incumbent_node_id": reevaluation_node.id,
            "reevaluation_promotion_decision_hash": stable_hash(
                reevaluation_decision.to_dict()
            ),
            "retained_behavior_hash": _program_behavior_hash(
                rebased_retained
            ),
            "reevaluation_policy": REEVALUATION_POLICY,
            "ready_before_reevaluation": ready_before_reevaluation,
            "ready_after_reevaluation": ready_after_reevaluation,
        },
        "persistence": {
            "archive_hash": final_archive["archive_hash"],
            "archive_reload_exact": reload_exact,
            "event_count": len(sink.events),
            "event_set_hash": stable_hash(sink.events),
        },
        "invariants": invariants,
        "model_calls": 0,
        "task_backend_calls": 0,
        "online_evaluator_calls": 0,
        "sealed_or_test_content_accessed": False,
        "raw_content_persisted": False,
    }
    report["report_hash"] = stable_hash(report)
    if not integration_passed:
        raise RuntimeError("archive/epoch integration invariants failed")

    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "archive.json").write_text(
            json.dumps(final_archive, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (root / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        loaded_report = json.loads(
            (root / "report.json").read_text(encoding="utf-8")
        )
        declared_hash = loaded_report.pop("report_hash")
        if stable_hash(loaded_report) != declared_hash:
            raise RuntimeError("persisted integration report hash mismatch")
        loaded_archive = json.loads(
            (root / "archive.json").read_text(encoding="utf-8")
        )
        _restore_archive(loaded_archive)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_integration(args.output_dir)
    print(
        json.dumps(
            {
                "integration_passed": report["integration_passed"],
                "report_hash": report["report_hash"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
