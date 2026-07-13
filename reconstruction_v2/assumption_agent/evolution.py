from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from math import ceil
from typing import Any, Sequence

from .archive import ArchiveNode, PolicyArchive
from .evaluation import (
    CounterfactualRunner,
    PairSummary,
    PromotionDecision,
    PromotionGate,
    counterfactual_pair_evidence_valid,
)
from .events import Event, EventSink, NullEventSink
from .models import (
    CounterfactualPair,
    HypothesisProgram,
    HypothesisStatus,
    ResidualExample,
    SplitName,
    TaskInput,
    stable_hash,
)
from .proposer import (
    PROPOSAL_DIVERSITY_POLICY_VERSION,
    StructuredHypothesisProposer,
)
from .splits import AccessPhase, SplitAccessGuard
from .validation import RecursiveValidationEngine, ValidationContext, ValidationTree


TRAIN_ONLY_CANDIDATE_SELECTION_VERSION = "train_static_support_then_complexity_v1"
CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION = (
    "train_contrastive_precision_then_support_v1"
)
PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION = (
    "train_contrastive_instance_family_coverage_then_precision_v1"
)
CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION = (
    "valid_train_failures_and_success_controls_v1"
)
ACTIONABLE_CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION = (
    "valid_train_failures_actionable_feedback_and_success_controls_v2"
)
CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSIONS = frozenset(
    {
        CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION,
        ACTIONABLE_CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSION,
    }
)
COUNTERFACTUAL_REPLAY_POLICY_VERSION = (
    "behavior_identical_validation_replay_v1"
)


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
    repair_model_failure_count: int = 0
    evaluated_candidate_behavior_hash: str | None = None


@dataclass(frozen=True)
class _StaticCandidateAudit:
    root: HypothesisProgram
    tree: ValidationTree
    accepted: HypothesisProgram | None
    training_score: tuple[Any, ...]
    training_metrics: "_TrainingCandidateMetrics"


@dataclass(frozen=True)
class _TrainingCandidateMetrics:
    failure_count: int
    success_control_count: int
    failure_activation_count: int
    failure_activation_family_count: int
    train_family_count: int
    success_false_positive_activation_count: int
    success_anti_trigger_protection_count: int
    failure_anti_trigger_block_count: int
    predicate_count: int
    action_count: int

    @property
    def activation_count(self) -> int:
        return (
            self.failure_activation_count
            + self.success_false_positive_activation_count
        )

    @property
    def precision(self) -> Fraction:
        if not self.activation_count:
            return Fraction(0, 1)
        return Fraction(self.failure_activation_count, self.activation_count)

    def to_dict(
        self,
        *,
        family_coverage_target: int | None = None,
    ) -> dict[str, Any]:
        example_count = self.failure_count + self.success_control_count
        abstention_count = example_count - self.activation_count
        payload = {
            "failure_count": self.failure_count,
            "success_control_count": self.success_control_count,
            "example_count": example_count,
            "failure_activation_count": self.failure_activation_count,
            "success_false_positive_activation_count": (
                self.success_false_positive_activation_count
            ),
            "success_anti_trigger_protection_count": (
                self.success_anti_trigger_protection_count
            ),
            "failure_anti_trigger_block_count": (
                self.failure_anti_trigger_block_count
            ),
            "activation_precision_numerator": self.failure_activation_count,
            "activation_precision_denominator": self.activation_count,
            "train_abstention_proxy_numerator": abstention_count,
            "train_abstention_proxy_denominator": example_count,
            "predicate_count": self.predicate_count,
            "action_count": self.action_count,
        }
        if family_coverage_target is not None:
            family_deficit = max(
                family_coverage_target - self.failure_activation_family_count,
                0,
            )
            payload.update(
                {
                    "failure_activation_family_count": (
                        self.failure_activation_family_count
                    ),
                    "train_family_count": self.train_family_count,
                    "failure_activation_family_target": family_coverage_target,
                    "failure_activation_family_deficit": family_deficit,
                    "failure_activation_family_target_met": family_deficit == 0,
                }
            )
        return payload


@dataclass(frozen=True)
class _CounterfactualEvidenceRecord:
    pairs: tuple[CounterfactualPair, ...]
    source_trace_id: str
    pair_set_hash: str


class CounterfactualEvidenceReplayCache:
    """Reuse validation evidence only when both executable policies are identical."""

    def __init__(self, *, event_sink: EventSink | None = None) -> None:
        self.event_sink = event_sink or NullEventSink()
        self._records: dict[str, _CounterfactualEvidenceRecord] = {}

    def run_or_replay(
        self,
        *,
        runner: CounterfactualRunner,
        tasks: Sequence[TaskInput],
        program: HypothesisProgram,
        baseline_programs: Sequence[HypothesisProgram],
        split: SplitName,
        trace_id: str,
    ) -> tuple[CounterfactualPair, ...]:
        if split is not SplitName.VALIDATION:
            raise PermissionError(
                "counterfactual replay is restricted to unsealed validation evidence"
            )
        descriptor = _counterfactual_replay_descriptor(
            runner=runner,
            tasks=tasks,
            program=program,
            baseline_programs=baseline_programs,
            split=split,
        )
        replay_key = stable_hash(descriptor)
        record = self._records.get(replay_key)
        if record is not None:
            _validate_replayed_pairs(
                record.pairs,
                tasks=tasks,
                split=split,
                evaluator_epoch=str(descriptor["evaluator_epoch"]),
            )
            self.event_sink.emit(
                Event(
                    event="counterfactual_evidence_replayed",
                    stage="evolution.counterfactual_replay",
                    trace_id=trace_id,
                    payload={
                        "policy": COUNTERFACTUAL_REPLAY_POLICY_VERSION,
                        "replay_key": replay_key,
                        "source_trace_id": record.source_trace_id,
                        "target_trace_id": trace_id,
                        "pair_set_hash": record.pair_set_hash,
                        "pair_count": len(record.pairs),
                        "candidate_behavior_hash": descriptor[
                            "candidate_behavior_hash"
                        ],
                        "baseline_behavior_set_hash": descriptor[
                            "baseline_behavior_set_hash"
                        ],
                        "task_set_hash": descriptor["task_set_hash"],
                        "behavior_identical": True,
                        "new_counterfactual_executions": 0,
                        "sealed_test_accessed": False,
                        "raw_content_persisted": False,
                    },
                )
            )
            return record.pairs

        pairs = tuple(
            runner.run(
                tasks,
                program=program,
                baseline_programs=baseline_programs,
                split=split,
                trace_id=trace_id,
            )
        )
        _validate_replayed_pairs(
            pairs,
            tasks=tasks,
            split=split,
            evaluator_epoch=str(descriptor["evaluator_epoch"]),
        )
        pair_set_hash = _counterfactual_pair_set_hash(pairs)
        invalid_pair_count = sum(not _counterfactual_pair_valid(row) for row in pairs)
        if invalid_pair_count:
            self.event_sink.emit(
                Event(
                    event="counterfactual_evidence_not_recorded_invalid",
                    stage="evolution.counterfactual_replay",
                    trace_id=trace_id,
                    payload={
                        "policy": COUNTERFACTUAL_REPLAY_POLICY_VERSION,
                        "replay_key": replay_key,
                        "source_trace_id": trace_id,
                        "pair_set_hash": pair_set_hash,
                        "pair_count": len(pairs),
                        "invalid_pair_count": invalid_pair_count,
                        "candidate_behavior_hash": descriptor[
                            "candidate_behavior_hash"
                        ],
                        "baseline_behavior_set_hash": descriptor[
                            "baseline_behavior_set_hash"
                        ],
                        "task_set_hash": descriptor["task_set_hash"],
                        "sealed_test_accessed": False,
                        "raw_content_persisted": False,
                    },
                )
            )
            return pairs
        self._records[replay_key] = _CounterfactualEvidenceRecord(
            pairs=pairs,
            source_trace_id=trace_id,
            pair_set_hash=pair_set_hash,
        )
        self.event_sink.emit(
            Event(
                event="counterfactual_evidence_recorded",
                stage="evolution.counterfactual_replay",
                trace_id=trace_id,
                payload={
                    "policy": COUNTERFACTUAL_REPLAY_POLICY_VERSION,
                    "replay_key": replay_key,
                    "source_trace_id": trace_id,
                    "pair_set_hash": pair_set_hash,
                    "pair_count": len(pairs),
                    "candidate_behavior_hash": descriptor[
                        "candidate_behavior_hash"
                    ],
                    "baseline_behavior_set_hash": descriptor[
                        "baseline_behavior_set_hash"
                    ],
                    "task_set_hash": descriptor["task_set_hash"],
                    "sealed_test_accessed": False,
                    "raw_content_persisted": False,
                },
            )
        )
        return pairs


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
        candidate_selection_policy: str = TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
        contrastive_training_evidence_policy: str | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        if candidate_selection_policy not in {
            TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
            CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION,
            PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
        }:
            raise ValueError(
                f"unsupported candidate selection policy: {candidate_selection_policy}"
            )
        if contrastive_training_evidence_policy not in {
            None,
            *CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSIONS,
        }:
            raise ValueError(
                "unsupported contrastive training evidence policy: "
                f"{contrastive_training_evidence_policy}"
            )
        contrastive_enabled = (
            contrastive_training_evidence_policy
            in CONTRASTIVE_TRAINING_EVIDENCE_POLICY_VERSIONS
        )
        if contrastive_enabled != (
            candidate_selection_policy
            in {
                CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION,
                PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
            }
        ):
            raise ValueError(
                "contrastive evidence and candidate selection policies must be paired"
            )
        self.proposer = proposer
        self.validator = validator
        self.counterfactual_runner = counterfactual_runner
        self.promotion_gate = promotion_gate
        self.archive = archive
        self.split_guard = split_guard
        if proposal_candidates_per_generation <= 0:
            raise ValueError("proposal candidate count must be positive")
        if (
            candidate_selection_policy
            == PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION
            and proposal_candidates_per_generation != 3
        ):
            raise ValueError(
                "coverage-aware proposal diversity requires exactly three candidates"
            )
        self.proposal_candidates_per_generation = proposal_candidates_per_generation
        self.candidate_selection_policy = candidate_selection_policy
        self.contrastive_training_evidence_policy = (
            contrastive_training_evidence_policy
        )
        self.event_sink = event_sink or NullEventSink()
        self._promotion_feedback: list[dict[str, object]] = []

    def evolve_once(
        self,
        *,
        residuals: Sequence[ResidualExample],
        validation_tasks: Sequence[TaskInput],
        validation_context: ValidationContext,
        proposal_candidates: Sequence[HypothesisProgram] | None = None,
        counterfactual_replay_cache: CounterfactualEvidenceReplayCache | None = None,
        trace_id: str = "evolution_generation",
    ) -> EvolutionRunResult:
        if validation_context.evaluator_epoch != self.counterfactual_runner.evaluator.epoch:
            raise ValueError("validation context and counterfactual evaluator epoch differ")
        if (
            validation_context.contrastive_training_evidence_policy
            != self.contrastive_training_evidence_policy
        ):
            raise ValueError(
                "validation context contrastive training evidence policy mismatch"
            )
        for task in validation_tasks:
            self.split_guard.authorize(task.id, AccessPhase.PROMOTION)
        proposals = tuple(
            _with_primary_metric(program, self.promotion_gate.spec.metric)
            for program in (
                proposal_candidates
                or self.propose_candidates(
                    residuals,
                    validation_context=validation_context,
                    trace_id=trace_id,
                )
            )
        )
        if not proposals:
            raise ValueError("evolution requires at least one proposal candidate")
        if any(row.evaluator_epoch != validation_context.evaluator_epoch for row in proposals):
            raise ValueError("shared proposal candidate crossed evaluator epochs")
        known_behaviors = {
            _runner_behavior_hash(self.counterfactual_runner, program)
            for program in self.archive.hypotheses.values()
        }
        novel_roots: list[HypothesisProgram] = []
        generation_behaviors: set[str] = set()
        known_ids = set(self.archive.hypotheses)
        for proposal in proposals:
            behavior = _runner_behavior_hash(self.counterfactual_runner, proposal)
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
        family_coverage_target = _training_family_coverage_target(
            validation_context.residuals,
            minimum_activation_rate=(
                self.promotion_gate.spec.minimum_activation_rate
            ),
        )
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
                        selection_policy=self.candidate_selection_policy,
                        family_coverage_target=family_coverage_target,
                    ),
                    training_metrics=_training_candidate_metrics(
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
        repair_model_failure_count = sum(
            audit.tree.repair_model_failure_count for audit in audits
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
                    "repair_model_failure_count": repair_model_failure_count,
                    "candidates": [
                        {
                            "root_id": audit.root.id,
                            "root_hash": audit.root.payload_hash,
                            "accepted_id": audit.accepted.id if audit.accepted else None,
                            "accepted_hash": (
                                audit.accepted.payload_hash if audit.accepted else None
                            ),
                            "training_score": (
                                list(audit.training_score[:-1])
                                if self.candidate_selection_policy
                                == TRAIN_ONLY_CANDIDATE_SELECTION_VERSION
                                else audit.training_metrics.to_dict()
                            ),
                            "contrastive_training_metrics": (
                                audit.training_metrics.to_dict(
                                    family_coverage_target=(
                                        family_coverage_target
                                        if self.candidate_selection_policy
                                        == PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION
                                        else None
                                    )
                                )
                                if self.candidate_selection_policy
                                in {
                                    CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION,
                                    PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
                                }
                                else None
                            ),
                            "selected": bool(eligible and audit is eligible[0]),
                        }
                        for audit in audits
                    ],
                    "selection_uses_validation_outcomes": False,
                    "selection_uses_validation": False,
                    "selection_policy": self.candidate_selection_policy,
                    "contrastive_training_evidence_policy": (
                        self.contrastive_training_evidence_policy
                    ),
                },
            )
        )
        if repair_model_failure_count:
            selected_audit = eligible[0] if eligible else audits[0]
            for audit in eligible:
                assert audit.accepted is not None
                self.archive.set_hypothesis_status(
                    audit.accepted.id,
                    HypothesisStatus.SHADOW,
                    trace_id=trace_id,
                )
            self.event_sink.emit(
                Event(
                    event="evolution_generation_blocked_by_repair_model_failure",
                    stage="evolution.train_selection",
                    trace_id=trace_id,
                    payload={
                        "repair_model_failure_count": repair_model_failure_count,
                        "proposal_candidate_count": len(proposals),
                        "static_accepted_candidate_count": len(eligible),
                        "counterfactual_validation_executed": False,
                        "archive_promotion_allowed": False,
                        "raw_content_persisted": False,
                    },
                )
            )
            return self._result(
                trace_id=trace_id,
                root=selected_audit.root,
                accepted=None,
                tree=selected_audit.tree,
                decision=None,
                archive_node=None,
                reason="proposal_model_failed",
                proposal_candidate_count=len(proposals),
                static_accepted_candidate_count=len(eligible),
                static_validation_node_count=static_node_count,
                static_validation_max_recursion_depth=static_max_depth,
                repaired_candidate_count=repaired_candidate_count,
                repair_model_failure_count=repair_model_failure_count,
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
                repair_model_failure_count=repair_model_failure_count,
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
        if counterfactual_replay_cache is None:
            pairs = self.counterfactual_runner.run(
                validation_tasks,
                program=accepted,
                baseline_programs=baseline_programs,
                split=SplitName.VALIDATION,
                trace_id=trace_id,
            )
        else:
            pairs = counterfactual_replay_cache.run_or_replay(
                runner=self.counterfactual_runner,
                tasks=validation_tasks,
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
                "metric": self.promotion_gate.spec.metric,
            }
        )
        summary = decision.summary
        invalid_counterfactual_evidence = any(
            (
                summary.invalid_pair_count,
                summary.provider_mismatch_count,
                summary.budget_mismatch_count,
            )
        )
        self.archive.record_score(
            archive_node_id=candidate_node.id,
            split=SplitName.VALIDATION.value,
            evaluator_epoch_id=accepted.evaluator_epoch,
            metric=self.promotion_gate.spec.metric,
            successes=summary.candidate_success_count,
            total=summary.pair_count,
            item_ids=tuple(pair.task_id for pair in pairs),
            valid=not invalid_counterfactual_evidence,
            invalidation_reason=(
                ""
                if not invalid_counterfactual_evidence
                else "invalid_counterfactual_evidence"
            ),
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
            repair_model_failure_count=repair_model_failure_count,
            evaluated_candidate_behavior_hash=_runner_behavior_hash(
                self.counterfactual_runner,
                accepted,
            ),
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
                "action_contract": {
                    "allowed_action_operations": sorted(
                        validation_context.allowed_action_operations
                    ),
                    "semantics": validation_context.action_semantics,
                    "external_evidence_is_hidden": (
                        validation_context.external_evidence_is_hidden
                    ),
                },
                "primary_metric": self.promotion_gate.spec.metric,
                "evaluator_hypotheses_require_separate_epoch_challenger": True,
                "prior_hypotheses": self._prior_hypothesis_context(),
                "prior_promotion_feedback": list(self._promotion_feedback),
                "novel_hypothesis_required": True,
                **(
                    {
                        "proposal_batch_contract": {
                            "policy": PROPOSAL_DIVERSITY_POLICY_VERSION,
                            "required_count": (
                                self.proposal_candidates_per_generation
                            ),
                            "diversity_unit": (
                                "train_failure_activation_or_action_treatment"
                            ),
                            "max_action_nodes_per_hypothesis": 4,
                            "profile_roles": [
                                "train_only_precision_anchor",
                                "train_only_cross_family_coverage",
                                "train_only_action_treatment_diversity",
                            ],
                            "compact_output": True,
                        },
                        "train_coverage_objective": {
                            "policy": self.candidate_selection_policy,
                            "evidence_scope": "train_only",
                            "coverage_unit": "distinct_failure_family",
                            "minimum_activation_rate": (
                                self.promotion_gate.spec.minimum_activation_rate
                            ),
                            "train_family_count": len(
                                {
                                    row.family
                                    for row in residuals
                                    if row.split is SplitName.TRAIN
                                }
                            ),
                            "failure_activation_family_target": (
                                _training_family_coverage_target(
                                    residuals,
                                    minimum_activation_rate=(
                                        self.promotion_gate.spec.minimum_activation_rate
                                    ),
                                )
                            ),
                            "coverage_reward_capped_at_target": True,
                            "validation_features_used": False,
                            "validation_outcomes_used": False,
                        },
                    }
                    if self.candidate_selection_policy
                    == PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION
                    else {}
                ),
                **(
                    {
                        "training_evidence_contract": {
                            "policy": self.contrastive_training_evidence_policy,
                            "label_field": "baseline_success",
                            "failure_label": False,
                            "success_control_label": True,
                            "success_control_role": (
                                "anti_trigger_negative_control"
                            ),
                            "context_may_be_used_for_trigger": False,
                            "context_may_shape_actions": True,
                        }
                    }
                    if self.contrastive_training_evidence_policy
                    else {}
                ),
            },
            trace_id=trace_id,
        )

    def _prior_hypothesis_context(self) -> list[dict[str, object]]:
        return [
            {
                "hypothesis_id": program.id,
                "behavior_hash": _runner_behavior_hash(
                    self.counterfactual_runner,
                    program,
                ),
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
        repair_model_failure_count: int = 0,
        evaluated_candidate_behavior_hash: str | None = None,
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
            repair_model_failure_count=repair_model_failure_count,
            evaluated_candidate_behavior_hash=evaluated_candidate_behavior_hash,
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
                    "repair_model_failure_count": repair_model_failure_count,
                    "evaluated_candidate_behavior_hash": (
                        evaluated_candidate_behavior_hash
                    ),
                    "generation_hash": stable_hash(
                        {
                            "root": root.payload_hash,
                            "accepted": accepted.payload_hash if accepted else None,
                            "decision": decision.to_dict() if decision else None,
                            "archive_node": archive_node.payload_hash if archive_node else None,
                            "evaluated_candidate_behavior_hash": (
                                evaluated_candidate_behavior_hash
                            ),
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


def _runner_behavior_hash(
    runner: CounterfactualRunner,
    program: HypothesisProgram,
) -> str:
    backend_hash = getattr(runner, "behavior_hash", None)
    if callable(backend_hash):
        try:
            value = str(backend_hash(program)).strip()
        except ValueError:
            value = ""
        if value:
            return value
    return _behavior_hash(program)


def _counterfactual_replay_descriptor(
    *,
    runner: CounterfactualRunner,
    tasks: Sequence[TaskInput],
    program: HypothesisProgram,
    baseline_programs: Sequence[HypothesisProgram],
    split: SplitName,
) -> dict[str, object]:
    evaluator_epoch = str(getattr(runner.evaluator, "epoch", ""))
    runtime_version = str(getattr(runner.runtime, "runtime_version", ""))
    evidence_execution_policy_hash = str(
        getattr(runner, "evidence_execution_policy_hash", "")
    )
    if not evaluator_epoch or not runtime_version:
        raise ValueError("counterfactual replay requires frozen evaluator and runtime versions")
    task_rows = [
        {
            "task_id": task.id,
            "family": task.family,
            "feature_hash": stable_hash(dict(task.features)),
        }
        for task in tasks
    ]
    baseline_behavior_hashes = sorted(
        _runner_behavior_hash(runner, row) for row in baseline_programs
    )
    descriptor: dict[str, object] = {
        "policy": COUNTERFACTUAL_REPLAY_POLICY_VERSION,
        "split": split.value,
        "evaluator_epoch": evaluator_epoch,
        "runtime_version": runtime_version,
        "candidate_behavior_hash": _runner_behavior_hash(runner, program),
        "baseline_behavior_set_hash": stable_hash(baseline_behavior_hashes),
        "task_set_hash": stable_hash(task_rows),
    }
    if evidence_execution_policy_hash:
        descriptor["evidence_execution_policy_hash"] = (
            evidence_execution_policy_hash
        )
    return descriptor


def _validate_replayed_pairs(
    pairs: Sequence[CounterfactualPair],
    *,
    tasks: Sequence[TaskInput],
    split: SplitName,
    evaluator_epoch: str,
) -> None:
    if tuple(row.task_id for row in pairs) != tuple(row.id for row in tasks):
        raise PermissionError("counterfactual replay task identity mismatch")
    if any(
        row.split is not split or row.evaluator_epoch != evaluator_epoch
        for row in pairs
    ):
        raise PermissionError("counterfactual replay crossed split or evaluator epoch")


def _with_primary_metric(
    program: HypothesisProgram,
    primary_metric: str,
) -> HypothesisProgram:
    if program.expected_effect.metric == primary_metric:
        return program
    return replace(
        program,
        expected_effect=replace(program.expected_effect, metric=primary_metric),
    )


def _counterfactual_pair_set_hash(
    pairs: Sequence[CounterfactualPair],
) -> str:
    return stable_hash(
        [
            {
                "task_id": row.task_id,
                "split": row.split.value,
                "evaluator_epoch": row.evaluator_epoch,
                "baseline_plan_hash": row.baseline.plan_hash,
                "candidate_plan_hash": row.candidate.plan_hash,
                "baseline_success": row.baseline_outcome.success,
                "candidate_success": row.candidate_outcome.success,
                "baseline_score": row.baseline_outcome.score,
                "candidate_score": row.candidate_outcome.score,
                "baseline_cost": row.baseline.total_cost,
                "candidate_cost": row.candidate.total_cost,
                "baseline_observation_hash": row.baseline.selected_result.metadata.get(
                    "observation_hash"
                ),
                "candidate_observation_hash": row.candidate.selected_result.metadata.get(
                    "observation_hash"
                ),
                "baseline_evaluation_valid": row.baseline_outcome.metrics.get(
                    "evaluation_valid", 1.0
                ),
                "candidate_evaluation_valid": row.candidate_outcome.metrics.get(
                    "evaluation_valid", 1.0
                ),
                "baseline_provider_fingerprint": row.baseline.selected_result.metadata.get(
                    "provider_fingerprint"
                ),
                "candidate_provider_fingerprint": row.candidate.selected_result.metadata.get(
                    "provider_fingerprint"
                ),
                "baseline_fairness_fingerprint": row.baseline.selected_result.metadata.get(
                    "fairness_fingerprint"
                ),
                "candidate_fairness_fingerprint": row.candidate.selected_result.metadata.get(
                    "fairness_fingerprint"
                ),
            }
            for row in pairs
        ]
    )


def _counterfactual_pair_valid(pair: CounterfactualPair) -> bool:
    return counterfactual_pair_evidence_valid(pair)


def _training_candidate_score(
    program: HypothesisProgram | None,
    residuals: Sequence[ResidualExample],
    *,
    selection_policy: str = TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
    family_coverage_target: int = 0,
) -> tuple[Any, ...]:
    if selection_policy not in {
        TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
        CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION,
        PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
    }:
        raise ValueError(f"unsupported candidate selection policy: {selection_policy}")
    if family_coverage_target < 0:
        raise ValueError("training family coverage target cannot be negative")
    if program is None:
        if (
            selection_policy
            == PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION
        ):
            return (
                10**9,
                10**9,
                10**9,
                10**9,
                10**9,
                10**9,
                "f" * 64,
            )
        return (0, 10**9, 10**9, 10**9, "f" * 64)
    metrics = _training_candidate_metrics(program, residuals)
    legacy_score = (
        -metrics.failure_activation_count,
        metrics.failure_anti_trigger_block_count,
        metrics.predicate_count,
        metrics.action_count,
        program.payload_hash,
    )
    if (
        selection_policy == TRAIN_ONLY_CANDIDATE_SELECTION_VERSION
        or (
            selection_policy == CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION
            and metrics.success_control_count == 0
        )
    ):
        return legacy_score
    if (
        selection_policy
        == PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION
    ):
        return (
            max(
                family_coverage_target
                - metrics.failure_activation_family_count,
                0,
            ),
            -metrics.precision,
            metrics.success_false_positive_activation_count,
            -metrics.failure_activation_count,
            metrics.predicate_count,
            metrics.action_count,
            program.payload_hash,
        )
    return (
        -metrics.precision,
        metrics.success_false_positive_activation_count,
        -metrics.failure_activation_count,
        metrics.predicate_count,
        metrics.action_count,
        program.payload_hash,
    )


def _training_candidate_metrics(
    program: HypothesisProgram | None,
    residuals: Sequence[ResidualExample],
) -> _TrainingCandidateMetrics:
    train_rows = [row for row in residuals if row.split is SplitName.TRAIN]
    failure_rows = [row for row in train_rows if not row.baseline_success]
    success_controls = [row for row in train_rows if row.baseline_success]
    train_family_count = len({row.family for row in train_rows})
    if program is None:
        return _TrainingCandidateMetrics(
            failure_count=len(failure_rows),
            success_control_count=len(success_controls),
            failure_activation_count=0,
            failure_activation_family_count=0,
            train_family_count=train_family_count,
            success_false_positive_activation_count=0,
            success_anti_trigger_protection_count=0,
            failure_anti_trigger_block_count=0,
            predicate_count=0,
            action_count=0,
        )
    failure_activation_count = sum(
        program.matches(row.features) for row in failure_rows
    )
    failure_activation_family_count = len(
        {
            row.family
            for row in failure_rows
            if program.matches(row.features)
        }
    )
    success_false_positive_activation_count = sum(
        program.matches(row.features) for row in success_controls
    )
    success_anti_trigger_protection_count = sum(
        program.trigger.matches(row.features)
        and not program.anti_trigger.is_empty
        and program.anti_trigger.matches(row.features)
        for row in success_controls
    )
    failure_anti_trigger_block_count = sum(
        not program.anti_trigger.is_empty
        and program.anti_trigger.matches(row.features)
        for row in failure_rows
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
    return _TrainingCandidateMetrics(
        failure_count=len(failure_rows),
        success_control_count=len(success_controls),
        failure_activation_count=failure_activation_count,
        failure_activation_family_count=failure_activation_family_count,
        train_family_count=train_family_count,
        success_false_positive_activation_count=(
            success_false_positive_activation_count
        ),
        success_anti_trigger_protection_count=(
            success_anti_trigger_protection_count
        ),
        failure_anti_trigger_block_count=failure_anti_trigger_block_count,
        predicate_count=predicate_count,
        action_count=len(program.action_graph),
    )


def _training_family_coverage_target(
    residuals: Sequence[ResidualExample],
    *,
    minimum_activation_rate: float,
) -> int:
    train_family_count = len(
        {
            row.family
            for row in residuals
            if row.split is SplitName.TRAIN
        }
    )
    if minimum_activation_rate <= 0.0:
        return 0
    return max(1, ceil(minimum_activation_rate * train_family_count))
