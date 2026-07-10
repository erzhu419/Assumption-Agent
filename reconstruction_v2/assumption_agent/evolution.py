from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .archive import ArchiveNode, PolicyArchive
from .evaluation import CounterfactualRunner, PairSummary, PromotionDecision, PromotionGate
from .events import Event, EventSink, NullEventSink
from .models import (
    HypothesisProgram,
    HypothesisStatus,
    ResidualExample,
    SplitName,
    TaskInput,
    stable_hash,
)
from .proposer import StructuredHypothesisProposer
from .splits import AccessPhase, SplitAccessGuard
from .validation import RecursiveValidationEngine, ValidationContext, ValidationTree


@dataclass(frozen=True)
class EvolutionRunResult:
    trace_id: str
    root_hypothesis_id: str
    accepted_hypothesis_id: str | None
    validation_tree: ValidationTree
    promotion_decision: PromotionDecision | None
    archive_node: ArchiveNode | None
    promoted: bool
    reason: str


class EvolutionKernel:
    """One controlled train-propose-repair-validate-promote generation."""

    def __init__(
        self,
        *,
        proposer: StructuredHypothesisProposer,
        validator: RecursiveValidationEngine,
        counterfactual_runner: CounterfactualRunner,
        promotion_gate: PromotionGate,
        archive: PolicyArchive,
        split_guard: SplitAccessGuard,
        proposal_candidates_per_generation: int = 3,
        event_sink: EventSink | None = None,
    ) -> None:
        self.proposer = proposer
        self.validator = validator
        self.counterfactual_runner = counterfactual_runner
        self.promotion_gate = promotion_gate
        self.archive = archive
        self.split_guard = split_guard
        if proposal_candidates_per_generation <= 0:
            raise ValueError("proposal candidate count must be positive")
        self.proposal_candidates_per_generation = proposal_candidates_per_generation
        self.event_sink = event_sink or NullEventSink()
        self._promotion_feedback: list[dict[str, object]] = []

    def evolve_once(
        self,
        *,
        residuals: Sequence[ResidualExample],
        validation_tasks: Sequence[TaskInput],
        validation_context: ValidationContext,
        proposal_candidates: Sequence[HypothesisProgram] | None = None,
        trace_id: str = "evolution_generation",
    ) -> EvolutionRunResult:
        if validation_context.evaluator_epoch != self.counterfactual_runner.evaluator.epoch:
            raise ValueError("validation context and counterfactual evaluator epoch differ")
        for task in validation_tasks:
            self.split_guard.authorize(task.id, AccessPhase.PROMOTION)
        proposals = tuple(proposal_candidates or self.propose_candidates(
            residuals,
            validation_context=validation_context,
            trace_id=trace_id,
        ))
        if not proposals:
            raise ValueError("evolution requires at least one proposal candidate")
        if any(row.evaluator_epoch != validation_context.evaluator_epoch for row in proposals):
            raise ValueError("shared proposal candidate crossed evaluator epochs")
        known_behaviors = {
            _behavior_hash(program) for program in self.archive.hypotheses.values()
        }
        root = next(
            (program for program in proposals if _behavior_hash(program) not in known_behaviors),
            None,
        )
        if root is None:
            duplicate = proposals[0]
            tree = ValidationTree(
                root_id=duplicate.id,
                nodes=(),
                accepted_program=None,
            )
            return self._result(
                trace_id=trace_id,
                root=duplicate,
                accepted=None,
                tree=tree,
                decision=None,
                archive_node=None,
                reason="duplicate_hypothesis_behavior",
            )
        if root.id in self.archive.hypotheses:
            root = replace(root, id=f"{root.id}-{root.payload_hash[:10]}")
        tree = self.validator.validate(root, validation_context, trace_id=trace_id)
        for node in tree.nodes:
            self.archive.register_hypothesis(node.program, trace_id=trace_id)
        accepted = tree.accepted_program
        if accepted is None:
            for node in tree.nodes:
                self.archive.set_hypothesis_status(
                    node.program.id,
                    HypothesisStatus.REJECTED,
                    trace_id=trace_id,
                )
            return self._result(
                trace_id=trace_id,
                root=root,
                accepted=None,
                tree=tree,
                decision=None,
                archive_node=None,
                reason="recursive_validation_rejected",
            )
        for node in tree.nodes:
            if node.program.id != accepted.id:
                self.archive.set_hypothesis_status(
                    node.program.id,
                    HypothesisStatus.REJECTED,
                    trace_id=trace_id,
                )

        parent = self.archive.nodes.get(self.archive.incumbent_id) if self.archive.incumbent_id else None
        baseline_programs = tuple(
            self.archive.hypotheses[hypothesis_id]
            for hypothesis_id in (parent.active_hypothesis_ids if parent else ())
        )
        active_ids = tuple(sorted({*(parent.active_hypothesis_ids if parent else ()), accepted.id}))
        candidate_node = self.archive.create_node(
            active_hypothesis_ids=active_ids,
            evaluator_epoch_id=accepted.evaluator_epoch,
            runtime_version=self.counterfactual_runner.runtime.runtime_version,
            parent_id=parent.id if parent else None,
            trace_id=trace_id,
        )
        pairs = self.counterfactual_runner.run(
            validation_tasks,
            program=accepted,
            baseline_programs=baseline_programs,
            split=SplitName.VALIDATION,
            trace_id=trace_id,
        )
        decision = self.promotion_gate.evaluate(
            accepted,
            pairs,
            sealed_test_accessed=self.split_guard.test_accessed,
            trace_id=trace_id,
        )
        self._promotion_feedback.append(
            {
                "hypothesis_id": accepted.id,
                "hypothesis_hash": accepted.payload_hash,
                "allowed": decision.allowed,
                "blockers": list(decision.blockers),
                "pair_summary": decision.summary.to_dict(confidence=decision.confidence),
            }
        )
        summary = decision.summary
        self.archive.record_score(
            archive_node_id=candidate_node.id,
            split=SplitName.VALIDATION.value,
            evaluator_epoch_id=accepted.evaluator_epoch,
            metric=accepted.expected_effect.metric,
            successes=summary.candidate_success_count,
            total=summary.pair_count,
            item_ids=tuple(pair.task_id for pair in pairs),
        )
        candidate_node = self.archive.apply_promotion(
            candidate_node_id=candidate_node.id,
            decision=decision,
            trace_id=trace_id,
        )
        return self._result(
            trace_id=trace_id,
            root=root,
            accepted=accepted,
            tree=tree,
            decision=decision,
            archive_node=candidate_node,
            reason="promoted" if decision.allowed else "promotion_gate_rejected",
        )

    def propose_candidates(
        self,
        residuals: Sequence[ResidualExample],
        *,
        validation_context: ValidationContext,
        trace_id: str,
    ) -> tuple[HypothesisProgram, ...]:
        return self.proposer.propose(
            residuals,
            evaluator_epoch=validation_context.evaluator_epoch,
            max_hypotheses=self.proposal_candidates_per_generation,
            capabilities={
                "available_lanes": sorted(validation_context.available_lanes),
                "baseline_lane": validation_context.baseline_lane,
                "prior_hypotheses": self._prior_hypothesis_context(),
                "prior_promotion_feedback": list(self._promotion_feedback),
                "novel_hypothesis_required": True,
            },
            trace_id=trace_id,
        )

    def _prior_hypothesis_context(self) -> list[dict[str, object]]:
        return [
            {
                "hypothesis_id": program.id,
                "behavior_hash": _behavior_hash(program),
                "status": program.status.value,
                "statement": program.statement,
                "trigger": program.to_dict()["trigger"],
                "anti_trigger": program.to_dict()["anti_trigger"],
                "action_graph": program.to_dict()["action_graph"],
            }
            for program in sorted(self.archive.hypotheses.values(), key=lambda row: row.id)
        ]

    def _result(
        self,
        *,
        trace_id: str,
        root: HypothesisProgram,
        accepted: HypothesisProgram | None,
        tree: ValidationTree,
        decision: PromotionDecision | None,
        archive_node: ArchiveNode | None,
        reason: str,
    ) -> EvolutionRunResult:
        result = EvolutionRunResult(
            trace_id=trace_id,
            root_hypothesis_id=root.id,
            accepted_hypothesis_id=accepted.id if accepted else None,
            validation_tree=tree,
            promotion_decision=decision,
            archive_node=archive_node,
            promoted=bool(decision and decision.allowed),
            reason=reason,
        )
        self.event_sink.emit(
            Event(
                event="evolution_generation_completed",
                stage="evolution",
                trace_id=trace_id,
                payload={
                    "root_hypothesis_id": root.id,
                    "accepted_hypothesis_id": result.accepted_hypothesis_id,
                    "recursive_node_count": len(tree.nodes),
                    "recursive_depth": tree.recursion_depth,
                    "promotion_allowed": result.promoted,
                    "promotion_blockers": list(decision.blockers) if decision else [],
                    "archive_node_id": archive_node.id if archive_node else None,
                    "reason": reason,
                    "generation_hash": stable_hash(
                        {
                            "root": root.payload_hash,
                            "accepted": accepted.payload_hash if accepted else None,
                            "decision": decision.to_dict() if decision else None,
                            "archive_node": archive_node.payload_hash if archive_node else None,
                        }
                    ),
                },
            )
        )
        return result


def _behavior_hash(program: HypothesisProgram) -> str:
    payload = program.to_dict()
    for key in (
        "id",
        "status",
        "parent_id",
        "lineage",
        "created_from_transition_ids",
    ):
        payload.pop(key, None)
    return stable_hash(payload)
