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


TRAIN_ONLY_CANDIDATE_SELECTION_VERSION = "train_static_support_then_complexity_v1"


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
    proposal_candidate_count: int = 1
    static_accepted_candidate_count: int = 0
    static_validation_node_count: int = 0
    static_validation_max_recursion_depth: int = 0
    repaired_candidate_count: int = 0


@dataclass(frozen=True)
class _StaticCandidateAudit:
    root: HypothesisProgram
    tree: ValidationTree
    accepted: HypothesisProgram | None
    training_score: tuple[int, int, int, int, str]


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
        novel_roots: list[HypothesisProgram] = []
        generation_behaviors: set[str] = set()
        known_ids = set(self.archive.hypotheses)
        for proposal in proposals:
            behavior = _behavior_hash(proposal)
            if behavior in known_behaviors or behavior in generation_behaviors:
                continue
            root = proposal
            if root.id in known_ids:
                root = replace(root, id=f"{root.id}-{root.payload_hash[:10]}")
            known_ids.add(root.id)
            generation_behaviors.add(behavior)
            novel_roots.append(root)
        if not novel_roots:
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
                proposal_candidate_count=len(proposals),
                static_accepted_candidate_count=0,
                static_validation_node_count=0,
                static_validation_max_recursion_depth=0,
                repaired_candidate_count=0,
            )
        audits: list[_StaticCandidateAudit] = []
        for index, root in enumerate(novel_roots):
            candidate_trace = f"{trace_id}:static-{index + 1}"
            tree = self.validator.validate(
                root,
                validation_context,
                trace_id=candidate_trace,
            )
            for node in tree.nodes:
                self.archive.register_hypothesis(node.program, trace_id=candidate_trace)
            accepted = tree.accepted_program
            if accepted is None:
                for node in tree.nodes:
                    self.archive.set_hypothesis_status(
                        node.program.id,
                        HypothesisStatus.REJECTED,
                        trace_id=candidate_trace,
                    )
            else:
                for node in tree.nodes:
                    if node.program.id != accepted.id:
                        self.archive.set_hypothesis_status(
                            node.program.id,
                            HypothesisStatus.REJECTED,
                            trace_id=candidate_trace,
                        )
            audits.append(
                _StaticCandidateAudit(
                    root=root,
                    tree=tree,
                    accepted=accepted,
                    training_score=_training_candidate_score(
                        accepted,
                        validation_context.residuals,
                    ),
                )
            )
        eligible = sorted(
            (audit for audit in audits if audit.accepted is not None),
            key=lambda audit: audit.training_score,
        )
        static_node_count = sum(len(audit.tree.nodes) for audit in audits)
        static_max_depth = max(
            (audit.tree.recursion_depth for audit in audits),
            default=0,
        )
        repaired_candidate_count = sum(
            audit.tree.recursion_depth > 0 for audit in audits
        )
        self.event_sink.emit(
            Event(
                event="hypothesis_training_candidate_selection_completed",
                stage="evolution.train_selection",
                trace_id=trace_id,
                payload={
                    "proposal_candidate_count": len(proposals),
                    "novel_candidate_count": len(audits),
                    "static_accepted_candidate_count": len(eligible),
                    "static_validation_node_count": static_node_count,
                    "static_validation_max_recursion_depth": static_max_depth,
                    "repaired_candidate_count": repaired_candidate_count,
                    "candidates": [
                        {
                            "root_id": audit.root.id,
                            "root_hash": audit.root.payload_hash,
                            "accepted_id": audit.accepted.id if audit.accepted else None,
                            "accepted_hash": (
                                audit.accepted.payload_hash if audit.accepted else None
                            ),
                            "training_score": list(audit.training_score[:-1]),
                            "selected": bool(eligible and audit is eligible[0]),
                        }
                        for audit in audits
                    ],
                    "selection_uses_validation_outcomes": False,
                    "selection_policy": TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
                },
            )
        )
        if not eligible:
            rejected = audits[0]
            return self._result(
                trace_id=trace_id,
                root=rejected.root,
                accepted=None,
                tree=rejected.tree,
                decision=None,
                archive_node=None,
                reason="recursive_validation_rejected",
                proposal_candidate_count=len(proposals),
                static_accepted_candidate_count=0,
                static_validation_node_count=static_node_count,
                static_validation_max_recursion_depth=static_max_depth,
                repaired_candidate_count=repaired_candidate_count,
            )
        selected = eligible[0]
        root = selected.root
        tree = selected.tree
        accepted = selected.accepted
        assert accepted is not None
        for audit in eligible[1:]:
            assert audit.accepted is not None
            if audit.accepted.id != accepted.id:
                self.archive.set_hypothesis_status(
                    audit.accepted.id,
                    HypothesisStatus.SHADOW,
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
            proposal_candidate_count=len(proposals),
            static_accepted_candidate_count=len(eligible),
            static_validation_node_count=static_node_count,
            static_validation_max_recursion_depth=static_max_depth,
            repaired_candidate_count=repaired_candidate_count,
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
                "runtime_trigger_contract": {
                    "allowed_feature_catalog": dict(
                        validation_context.trigger_feature_catalog
                    ),
                    "forbidden_context_only_keys": [
                        "task_instruction",
                        "observed_metrics",
                        "execution_signals",
                    ],
                    "context_is_for_action_design_only": True,
                },
                "runtime_candidate_kinds": sorted(
                    kind.value for kind in validation_context.allowed_runtime_kinds
                ),
                "evaluator_hypotheses_require_separate_epoch_challenger": True,
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
        proposal_candidate_count: int = 1,
        static_accepted_candidate_count: int = 0,
        static_validation_node_count: int = 0,
        static_validation_max_recursion_depth: int = 0,
        repaired_candidate_count: int = 0,
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
            proposal_candidate_count=proposal_candidate_count,
            static_accepted_candidate_count=static_accepted_candidate_count,
            static_validation_node_count=static_validation_node_count,
            static_validation_max_recursion_depth=static_validation_max_recursion_depth,
            repaired_candidate_count=repaired_candidate_count,
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
                    "proposal_candidate_count": proposal_candidate_count,
                    "static_accepted_candidate_count": static_accepted_candidate_count,
                    "static_validation_node_count": static_validation_node_count,
                    "static_validation_max_recursion_depth": (
                        static_validation_max_recursion_depth
                    ),
                    "repaired_candidate_count": repaired_candidate_count,
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


def _training_candidate_score(
    program: HypothesisProgram | None,
    residuals: Sequence[ResidualExample],
) -> tuple[int, int, int, int, str]:
    if program is None:
        return (0, 10**9, 10**9, 10**9, "f" * 64)
    train_rows = [row for row in residuals if row.split is SplitName.TRAIN]
    support = sum(program.matches(row.features) for row in train_rows)
    anti_support = sum(
        not program.anti_trigger.is_empty
        and program.anti_trigger.matches(row.features)
        for row in train_rows
    )
    predicate_count = sum(
        len(group)
        for group in (
            program.trigger.all_of,
            program.trigger.any_of,
            program.trigger.none_of,
            program.anti_trigger.all_of,
            program.anti_trigger.any_of,
            program.anti_trigger.none_of,
        )
    )
    return (
        -support,
        anti_support,
        predicate_count,
        len(program.action_graph),
        program.payload_hash,
    )
