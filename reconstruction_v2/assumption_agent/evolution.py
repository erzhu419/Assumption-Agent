from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations
from math import ceil
from typing import Any, Mapping, Sequence

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
    FAMILY_SLOT_PROPOSAL_FORMATION_POLICY_VERSION,
    PROPOSAL_DIVERSITY_POLICY_VERSION,
    REPAIR_REQUEST_SCOPE_POLICY_VERSION,
    StructuredHypothesisProposer,
    TRAIN_ACTION_DESIGN_POLICY_VERSION,
    TRAIN_ACTION_DESIGN_POLICY_VERSIONS,
    train_action_quality_contract,
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
COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSION = (
    "train_contrastive_complementary_family_bundle_precision_first_v1"
)
COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION = (
    "train_contrastive_complementary_family_support_bundle_precision_first_v2"
)
COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS = frozenset(
    {
        COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSION,
        COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION,
    }
)
CANDIDATE_BUNDLE_POLICY_VERSION = (
    "train_only_union_program_set_single_paired_validation_conservative_thresholds_v1"
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
PROGRAM_SET_COUNTERFACTUAL_REPLAY_POLICY_VERSION = (
    "behavior_identical_validation_program_set_replay_v2"
)
PROPOSAL_FORMATION_POLICY_VERSION = (
    FAMILY_SLOT_PROPOSAL_FORMATION_POLICY_VERSION
)
PROPOSAL_FORMATION_POLICY_VERSIONS = frozenset(
    {PROPOSAL_FORMATION_POLICY_VERSION}
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
    selected_candidate_hypothesis_ids: tuple[str, ...] = ()


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
class _FamilyProfilePrimitive:
    kind: str
    value: str
    train_failure_evidence_count: int

    @property
    def reusable(self) -> bool:
        return self.train_failure_evidence_count >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "train_failure_evidence_count": self.train_failure_evidence_count,
            "reusable_across_same_family_failures": self.reusable,
        }


@dataclass(frozen=True)
class _FamilyProposalSlot:
    target_family: str
    failures: tuple[ResidualExample, ...]
    profile_items: tuple[tuple[str, Mapping[str, Any]], ...]
    profile_evidence_hash: str
    preferred_primitives: tuple[_FamilyProfilePrimitive, ...]
    failed_primitives: tuple[_FamilyProfilePrimitive, ...]
    prior_use_count: int

    @property
    def reusable_preferred_primitive_count(self) -> int:
        return sum(row.reusable for row in self.preferred_primitives)


@dataclass(frozen=True)
class _TrainingCandidateBundleMetrics:
    failure_count: int
    success_control_count: int
    failure_activation_count: int
    failure_activation_family_count: int
    train_family_count: int
    success_false_positive_activation_count: int
    overlap_count: int
    bundle_size: int
    complexity: int

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

    def to_dict(self, *, family_coverage_target: int) -> dict[str, Any]:
        family_deficit = max(
            family_coverage_target - self.failure_activation_family_count,
            0,
        )
        return {
            "failure_count": self.failure_count,
            "success_control_count": self.success_control_count,
            "failure_activation_count": self.failure_activation_count,
            "failure_activation_family_count": (
                self.failure_activation_family_count
            ),
            "train_family_count": self.train_family_count,
            "success_false_positive_activation_count": (
                self.success_false_positive_activation_count
            ),
            "activation_precision_numerator": self.failure_activation_count,
            "activation_precision_denominator": self.activation_count,
            "failure_activation_family_target": family_coverage_target,
            "failure_activation_family_deficit": family_deficit,
            "failure_activation_family_target_met": family_deficit == 0,
            "overlap_count": self.overlap_count,
            "bundle_size": self.bundle_size,
            "failure_support": self.failure_activation_count,
            "complexity": self.complexity,
        }


@dataclass(frozen=True)
class _TrainingCandidateBundleAudit:
    members: tuple[_StaticCandidateAudit, ...]
    metrics: _TrainingCandidateBundleMetrics
    canonical_set_hash: str
    ranking_score: tuple[Any, ...]

    @property
    def accepted_hypothesis_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                audit.accepted.id
                for audit in self.members
                if audit.accepted is not None
            )
        )


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
        return self._run_or_replay_programs(
            runner=runner,
            tasks=tasks,
            programs=(program,),
            baseline_programs=baseline_programs,
            split=split,
            trace_id=trace_id,
            program_set=False,
        )

    def run_or_replay_bundle(
        self,
        *,
        runner: CounterfactualRunner,
        tasks: Sequence[TaskInput],
        programs: Sequence[HypothesisProgram],
        baseline_programs: Sequence[HypothesisProgram],
        split: SplitName,
        trace_id: str,
    ) -> tuple[CounterfactualPair, ...]:
        return self._run_or_replay_programs(
            runner=runner,
            tasks=tasks,
            programs=programs,
            baseline_programs=baseline_programs,
            split=split,
            trace_id=trace_id,
            program_set=True,
        )

    def _run_or_replay_programs(
        self,
        *,
        runner: CounterfactualRunner,
        tasks: Sequence[TaskInput],
        programs: Sequence[HypothesisProgram],
        baseline_programs: Sequence[HypothesisProgram],
        split: SplitName,
        trace_id: str,
        program_set: bool,
    ) -> tuple[CounterfactualPair, ...]:
        if split is not SplitName.VALIDATION:
            raise PermissionError(
                "counterfactual replay is restricted to unsealed validation evidence"
            )
        canonical_programs = tuple(sorted(programs, key=lambda row: row.id))
        if not canonical_programs:
            raise ValueError("counterfactual replay bundle cannot be empty")
        if len({row.id for row in canonical_programs}) != len(canonical_programs):
            raise ValueError("counterfactual replay bundle contains duplicate IDs")
        descriptor = _counterfactual_replay_descriptor(
            runner=runner,
            tasks=tasks,
            programs=canonical_programs,
            baseline_programs=baseline_programs,
            split=split,
            program_set=program_set,
        )
        replay_key = stable_hash(descriptor)
        evidence_identity: dict[str, object] = {
            "candidate_behavior_hash": descriptor["candidate_behavior_hash"],
            "baseline_behavior_set_hash": descriptor[
                "baseline_behavior_set_hash"
            ],
            "task_set_hash": descriptor["task_set_hash"],
        }
        if program_set:
            evidence_identity.update(
                {
                    "candidate_hypothesis_ids": [
                        row.id for row in canonical_programs
                    ],
                    "candidate_behavior_set_hash": descriptor[
                        "candidate_behavior_set_hash"
                    ],
                }
            )
        replay_policy = (
            PROGRAM_SET_COUNTERFACTUAL_REPLAY_POLICY_VERSION
            if program_set
            else COUNTERFACTUAL_REPLAY_POLICY_VERSION
        )
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
                        "policy": replay_policy,
                        "replay_key": replay_key,
                        "source_trace_id": record.source_trace_id,
                        "target_trace_id": trace_id,
                        "pair_set_hash": record.pair_set_hash,
                        "pair_count": len(record.pairs),
                        **evidence_identity,
                        "behavior_identical": True,
                        "new_counterfactual_executions": 0,
                        "sealed_test_accessed": False,
                        "raw_content_persisted": False,
                    },
                )
            )
            return record.pairs

        if program_set:
            pairs = tuple(
                runner.run_bundle(
                    tasks,
                    programs=canonical_programs,
                    baseline_programs=baseline_programs,
                    split=split,
                    trace_id=trace_id,
                )
            )
        else:
            pairs = tuple(
                runner.run(
                    tasks,
                    program=canonical_programs[0],
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
                        "policy": replay_policy,
                        "replay_key": replay_key,
                        "source_trace_id": trace_id,
                        "pair_set_hash": pair_set_hash,
                        "pair_count": len(pairs),
                        "invalid_pair_count": invalid_pair_count,
                        **evidence_identity,
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
                    "policy": replay_policy,
                    "replay_key": replay_key,
                    "source_trace_id": trace_id,
                    "pair_set_hash": pair_set_hash,
                    "pair_count": len(pairs),
                    **evidence_identity,
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
        candidate_bundle_policy: str | None = None,
        contrastive_training_evidence_policy: str | None = None,
        train_action_design_policy: str | None = None,
        proposal_formation_policy: str | None = None,
        repair_request_scope_policy: str | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        if candidate_selection_policy not in {
            TRAIN_ONLY_CANDIDATE_SELECTION_VERSION,
            CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION,
            PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
            *COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS,
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
        if repair_request_scope_policy not in {
            None,
            REPAIR_REQUEST_SCOPE_POLICY_VERSION,
        }:
            raise ValueError(
                f"unsupported repair request scope policy: {repair_request_scope_policy}"
            )
        if train_action_design_policy not in {
            None,
            *TRAIN_ACTION_DESIGN_POLICY_VERSIONS,
        }:
            raise ValueError(
                "unsupported TRAIN action design policy: "
                f"{train_action_design_policy}"
            )
        if proposal_formation_policy not in {
            None,
            *PROPOSAL_FORMATION_POLICY_VERSIONS,
        }:
            raise ValueError(
                "unsupported proposal formation policy: "
                f"{proposal_formation_policy}"
            )
        if (
            proposal_formation_policy == PROPOSAL_FORMATION_POLICY_VERSION
            and train_action_design_policy
            != TRAIN_ACTION_DESIGN_POLICY_VERSION
        ):
            raise ValueError(
                "profile-grounded family-slot proposal formation requires "
                f"TRAIN action design policy {TRAIN_ACTION_DESIGN_POLICY_VERSION}"
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
                *COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS,
            }
        ):
            raise ValueError(
                "contrastive evidence and candidate selection policies must be paired"
            )
        bundle_selection_enabled = (
            candidate_selection_policy
            in COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS
        )
        if bundle_selection_enabled != (
            candidate_bundle_policy == CANDIDATE_BUNDLE_POLICY_VERSION
        ):
            raise ValueError(
                "complementary bundle selection and candidate bundle policies "
                "must be paired"
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
            in {
                PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
                *COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS,
            }
            and proposal_candidates_per_generation != 3
        ):
            raise ValueError(
                "coverage-aware proposal diversity requires exactly three candidates"
            )
        if (
            proposal_formation_policy == PROPOSAL_FORMATION_POLICY_VERSION
            and proposal_candidates_per_generation != 3
        ):
            raise ValueError(
                "profile-grounded family-slot proposal formation requires "
                "exactly three candidates"
            )
        self.proposal_candidates_per_generation = proposal_candidates_per_generation
        self.candidate_selection_policy = candidate_selection_policy
        self.candidate_bundle_policy = candidate_bundle_policy
        self.contrastive_training_evidence_policy = (
            contrastive_training_evidence_policy
        )
        self.train_action_design_policy = train_action_design_policy
        self.proposal_formation_policy = proposal_formation_policy
        self.repair_request_scope_policy = repair_request_scope_policy
        self.event_sink = event_sink or NullEventSink()
        self._promotion_feedback: list[dict[str, object]] = []
        self._proposal_family_use_counts: dict[str, int] = {}
        self._recorded_proposal_family_usage: dict[
            str,
            tuple[tuple[str, ...], tuple[str | None, ...]],
        ] = {}

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
        if (
            validation_context.repair_request_scope_policy
            != self.repair_request_scope_policy
        ):
            raise ValueError(
                "validation context repair request scope policy mismatch"
            )
        if (
            validation_context.train_action_design_policy
            != self.train_action_design_policy
        ):
            raise ValueError(
                "validation context TRAIN action design policy mismatch"
            )
        for task in validation_tasks:
            self.split_guard.authorize(task.id, AccessPhase.PROMOTION)
        shared_proposals_supplied = bool(proposal_candidates)
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
        if (
            shared_proposals_supplied
            and self.proposal_formation_policy
            == PROPOSAL_FORMATION_POLICY_VERSION
        ):
            self._record_matched_proposal_families(
                proposals,
                residuals=residuals,
                validation_context=validation_context,
                trace_id=trace_id,
                source="shared_proposal_candidates",
                requested_targets=self.proposer.family_slot_targets_for(
                    proposals
                ),
            )
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
        candidate_bundle_audits: tuple[_TrainingCandidateBundleAudit, ...] = ()
        if (
            self.candidate_selection_policy
            in COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS
            and eligible
        ):
            candidate_bundle_audits = _training_candidate_bundle_audits(
                eligible,
                validation_context.residuals,
                family_coverage_target=family_coverage_target,
                selection_policy=self.candidate_selection_policy,
            )
            selected_audits = candidate_bundle_audits[0].members
            selected_candidate_set_hash = candidate_bundle_audits[
                0
            ].canonical_set_hash
        else:
            selected_audits = tuple(eligible[:1])
            selected_candidate_set_hash = (
                _candidate_audit_set_hash(selected_audits)
                if selected_audits
                else None
            )
        selected_candidate_hypothesis_ids = tuple(
            sorted(
                audit.accepted.id
                for audit in selected_audits
                if audit.accepted is not None
            )
        )
        selected_candidate_id_set = set(selected_candidate_hypothesis_ids)
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
                                        in {
                                            PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
                                            *COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS,
                                        }
                                        else None
                                    )
                                )
                                if self.candidate_selection_policy
                                in {
                                    CONTRASTIVE_TRAIN_CANDIDATE_SELECTION_VERSION,
                                    PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
                                    *COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS,
                                }
                                else None
                            ),
                            "selected": bool(
                                audit.accepted
                                and audit.accepted.id
                                in selected_candidate_id_set
                            ),
                        }
                        for audit in audits
                    ],
                    "selection_uses_validation_outcomes": False,
                    "selection_uses_validation": False,
                    "selection_policy": self.candidate_selection_policy,
                    **(
                        {
                            "candidate_bundle_policy": (
                                self.candidate_bundle_policy
                            ),
                            "selected_candidate_hypothesis_ids": list(
                                selected_candidate_hypothesis_ids
                            ),
                            "selected_candidate_set_hash": (
                                selected_candidate_set_hash
                            ),
                            "candidate_subsets": [
                                _training_candidate_bundle_event_row(
                                    audit,
                                    family_coverage_target=(
                                        family_coverage_target
                                    ),
                                    selection_policy=(
                                        self.candidate_selection_policy
                                    ),
                                    selected=(index == 0),
                                )
                                for index, audit in enumerate(
                                    candidate_bundle_audits
                                )
                            ],
                        }
                        if self.candidate_bundle_policy
                        else {}
                    ),
                    "contrastive_training_evidence_policy": (
                        self.contrastive_training_evidence_policy
                    ),
                },
            )
        )
        if repair_model_failure_count:
            selected_audit = (
                min(
                    selected_audits,
                    key=lambda audit: (
                        audit.accepted.id if audit.accepted else audit.root.id
                    ),
                )
                if selected_audits
                else audits[0]
            )
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
                selected_candidate_hypothesis_ids=(),
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
                selected_candidate_hypothesis_ids=(),
            )
        selected = min(
            selected_audits,
            key=lambda audit: (
                audit.accepted.id if audit.accepted else audit.root.id
            ),
        )
        root = selected.root
        tree = selected.tree
        accepted = selected.accepted
        assert accepted is not None
        selected_programs = tuple(
            sorted(
                (
                    audit.accepted
                    for audit in selected_audits
                    if audit.accepted is not None
                ),
                key=lambda program: program.id,
            )
        )
        for audit in eligible:
            assert audit.accepted is not None
            if audit.accepted.id not in selected_candidate_id_set:
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
        active_ids = tuple(
            sorted(
                {
                    *(parent.active_hypothesis_ids if parent else ()),
                    *(program.id for program in selected_programs),
                }
            )
        )
        candidate_node = self.archive.create_node(
            active_hypothesis_ids=active_ids,
            evaluator_epoch_id=accepted.evaluator_epoch,
            runtime_version=self.counterfactual_runner.runtime.runtime_version,
            parent_id=parent.id if parent else None,
            trace_id=trace_id,
        )
        bundle_selection = (
            self.candidate_selection_policy
            in COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS
        )
        if counterfactual_replay_cache is None:
            if bundle_selection:
                pairs = self.counterfactual_runner.run_bundle(
                    validation_tasks,
                    programs=selected_programs,
                    baseline_programs=baseline_programs,
                    split=SplitName.VALIDATION,
                    trace_id=trace_id,
                )
            else:
                pairs = self.counterfactual_runner.run(
                    validation_tasks,
                    program=accepted,
                    baseline_programs=baseline_programs,
                    split=SplitName.VALIDATION,
                    trace_id=trace_id,
                )
        else:
            if bundle_selection:
                pairs = counterfactual_replay_cache.run_or_replay_bundle(
                    runner=self.counterfactual_runner,
                    tasks=validation_tasks,
                    programs=selected_programs,
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
        if bundle_selection:
            decision = self.promotion_gate.evaluate_bundle(
                selected_programs,
                pairs,
                sealed_test_accessed=self.split_guard.test_accessed,
                trace_id=trace_id,
            )
        else:
            decision = self.promotion_gate.evaluate(
                accepted,
                pairs,
                sealed_test_accessed=self.split_guard.test_accessed,
                trace_id=trace_id,
            )
        self._promotion_feedback.append(
            {
                "hypothesis_id": (
                    f"program_set_{selected_candidate_set_hash[:16]}"
                    if bundle_selection and selected_candidate_set_hash
                    else accepted.id
                ),
                "hypothesis_hash": (
                    selected_candidate_set_hash
                    if bundle_selection and selected_candidate_set_hash
                    else accepted.payload_hash
                ),
                **(
                    {
                        "candidate_unit": "program_set",
                        "selected_candidate_hypothesis_ids": list(
                            selected_candidate_hypothesis_ids
                        ),
                        "selected_candidate_set_hash": (
                            selected_candidate_set_hash
                        ),
                    }
                    if bundle_selection
                    else {}
                ),
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
            retain_rejected_hypotheses_as_shadow=bundle_selection,
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
            evaluated_candidate_behavior_hash=(
                _runner_behavior_set_hash(
                    self.counterfactual_runner,
                    selected_programs,
                )
                if bundle_selection
                else _runner_behavior_hash(
                    self.counterfactual_runner,
                    accepted,
                )
            ),
            selected_candidate_hypothesis_ids=(
                selected_candidate_hypothesis_ids
            ),
        )

    def propose_candidates(
        self,
        residuals: Sequence[ResidualExample],
        *,
        validation_context: ValidationContext,
        trace_id: str,
    ) -> tuple[HypothesisProgram, ...]:
        if (
            self.proposal_formation_policy
            == PROPOSAL_FORMATION_POLICY_VERSION
        ):
            return self._propose_family_slot_candidates(
                residuals,
                validation_context=validation_context,
                trace_id=trace_id,
            )
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
                        "action_context_profile_hash",
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
                        "action_quality_contract": train_action_quality_contract(
                            validation_context.train_action_design_policy
                        ),
                        "train_action_design_profiles": {
                            str(key): dict(value)
                            for key, value in sorted(
                                validation_context.action_design_profiles.items()
                            )
                        },
                    }
                    if validation_context.train_action_design_policy
                    else {}
                ),
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
                            **(
                                {
                                    "candidate_bundle_policy": (
                                        self.candidate_bundle_policy
                                    ),
                                    "candidate_unit": (
                                        "complementary_program_set"
                                    ),
                                    "component_precision_precedes_bundle_coverage": True,
                                    "bundle_selected_before_validation": True,
                                    **(
                                        {
                                            "actual_family_count_precedes_failure_support": True,
                                            "failure_support_precedes_bundle_size": True,
                                        }
                                        if self.candidate_selection_policy
                                        == COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION
                                        else {}
                                    ),
                                }
                                if self.candidate_bundle_policy
                                else {}
                            ),
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
                            **(
                                {
                                    "family_target_deficit_capped_at_target": True,
                                    "post_target_actual_family_count_tiebreak": True,
                                }
                                if self.candidate_selection_policy
                                == COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION
                                else {"coverage_reward_capped_at_target": True}
                            ),
                            "validation_features_used": False,
                            "validation_outcomes_used": False,
                        },
                    }
                    if self.candidate_selection_policy
                    in {
                        PROSPECTIVE_FAMILY_COVERAGE_CANDIDATE_SELECTION_VERSION,
                        *COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS,
                    }
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

    def _propose_family_slot_candidates(
        self,
        residuals: Sequence[ResidualExample],
        *,
        validation_context: ValidationContext,
        trace_id: str,
    ) -> tuple[HypothesisProgram, ...]:
        issues = [
            issue
            for residual in residuals
            for issue in residual.validate()
        ]
        if issues:
            raise PermissionError(
                f"proposal data isolation failed: {sorted(set(issues))}"
            )
        success_controls = tuple(
            sorted(
                (
                    row
                    for row in residuals
                    if row.split is SplitName.TRAIN and row.baseline_success
                ),
                key=_residual_stable_order,
            )
        )
        ranked_slots = _rank_family_proposal_slots(
            residuals,
            profiles=validation_context.action_design_profiles,
            family_use_counts=self._proposal_family_use_counts,
        )
        if len(ranked_slots) < self.proposal_candidates_per_generation:
            raise ValueError(
                "profile-grounded family-slot proposal formation requires "
                "three distinct TRAIN failure families"
            )
        selected_slots = ranked_slots[: self.proposal_candidates_per_generation]
        slot_plan_rows = [
            _family_slot_event_row(slot, slot_index=index)
            for index, slot in enumerate(selected_slots, start=1)
        ]
        slot_plan_hash = stable_hash(
            {
                "policy": PROPOSAL_FORMATION_POLICY_VERSION,
                "slots": slot_plan_rows,
            }
        )
        self.event_sink.emit(
            Event(
                event="proposal_family_slot_plan_created",
                stage="proposal.family_slots",
                trace_id=trace_id,
                payload={
                    "policy": PROPOSAL_FORMATION_POLICY_VERSION,
                    "slot_count": len(selected_slots),
                    "distinct_target_family_count": len(
                        {slot.target_family for slot in selected_slots}
                    ),
                    "available_train_failure_family_count": len(ranked_slots),
                    "train_success_control_count": len(success_controls),
                    "slot_plan_hash": slot_plan_hash,
                    "slots": slot_plan_rows,
                    "validation_features_used": False,
                    "validation_outcomes_used": False,
                    "verifier_content_used": False,
                    "test_content_used": False,
                    "raw_content_persisted": False,
                },
            )
        )
        programs: list[HypothesisProgram] = []
        for index, slot in enumerate(selected_slots, start=1):
            slot_id = f"train-family-slot-{index}"
            scoped_residuals = (*slot.failures, *success_controls)
            capability_payload = self._family_slot_capabilities(
                slot,
                slot_id=slot_id,
                success_control_count=len(success_controls),
                validation_context=validation_context,
            )
            proposed = self.proposer.propose(
                scoped_residuals,
                evaluator_epoch=validation_context.evaluator_epoch,
                max_hypotheses=1,
                capabilities=capability_payload,
                trace_id=f"{trace_id}:family-slot-{index}",
            )
            programs.append(proposed[0])

        requested_targets = tuple(
            slot.target_family for slot in selected_slots
        )
        self.proposer.record_family_slot_batch(
            programs,
            requested_targets,
        )
        matched_families = self._record_matched_proposal_families(
            programs,
            residuals=residuals,
            validation_context=validation_context,
            trace_id=trace_id,
            source="generated_family_slots",
            requested_targets=requested_targets,
        )
        for index, (slot, program, matched_family) in enumerate(
            zip(selected_slots, programs, matched_families),
            start=1,
        ):
            candidate_audit = _family_slot_candidate_audit(
                program,
                target_slot=slot,
                all_slots=ranked_slots,
            )
            preferred_rows = [
                row.to_dict() for row in slot.preferred_primitives
            ]
            failed_rows = [row.to_dict() for row in slot.failed_primitives]
            self.event_sink.emit(
                Event(
                    event="proposal_family_slot_completed",
                    stage="proposal.family_slots",
                    trace_id=f"{trace_id}:family-slot-{index}",
                    payload={
                        "policy": PROPOSAL_FORMATION_POLICY_VERSION,
                        "slot_id": f"train-family-slot-{index}",
                        "slot_plan_hash": slot_plan_hash,
                        "target_family": slot.target_family,
                        "target_family_hash": stable_hash(
                            {"family": slot.target_family}
                        ),
                        "profile_evidence_hash": slot.profile_evidence_hash,
                        "preferred_primitive_count": len(preferred_rows),
                        "preferred_primitive_set_hash": stable_hash(
                            {"primitives": preferred_rows}
                        ),
                        "failed_primitive_count": len(failed_rows),
                        "failed_primitive_set_hash": stable_hash(
                            {"primitives": failed_rows}
                        ),
                        "candidate_hash": program.payload_hash,
                        "matched_family_hash": (
                            stable_hash({"family": matched_family})
                            if matched_family is not None
                            else None
                        ),
                        **candidate_audit,
                        "response_rejected_by_diversity": False,
                        "proposal_retry_by_diversity": False,
                        "validation_features_used": False,
                        "validation_outcomes_used": False,
                        "verifier_content_used": False,
                        "test_content_used": False,
                        "raw_content_persisted": False,
                    },
                )
            )
        result = tuple(programs)
        self.event_sink.emit(
            Event(
                event="proposal_family_slots_completed",
                stage="proposal.family_slots",
                trace_id=trace_id,
                payload={
                    "policy": PROPOSAL_FORMATION_POLICY_VERSION,
                    "slot_plan_hash": slot_plan_hash,
                    "candidate_count": len(result),
                    "candidate_set_hash": stable_hash(
                        {
                            "candidate_hashes": [
                                row.payload_hash for row in result
                            ]
                        }
                    ),
                    "matched_family_count": sum(
                        family is not None for family in matched_families
                    ),
                    "distinct_matched_family_count": len(
                        {family for family in matched_families if family}
                    ),
                    "validation_features_used": False,
                    "validation_outcomes_used": False,
                    "verifier_content_used": False,
                    "test_content_used": False,
                    "raw_content_persisted": False,
                },
            )
        )
        return result

    def _family_slot_capabilities(
        self,
        slot: _FamilyProposalSlot,
        *,
        slot_id: str,
        success_control_count: int,
        validation_context: ValidationContext,
    ) -> dict[str, Any]:
        preferred_primitives = [
            row.to_dict() for row in slot.preferred_primitives
        ]
        failed_primitives = [row.to_dict() for row in slot.failed_primitives]
        return {
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
                    "action_context_profile_hash",
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
            "prior_hypotheses": [],
            "prior_promotion_feedback": [],
            "prior_history_excluded_from_family_slot_proposal": True,
            "novel_hypothesis_required": True,
            **(
                {
                    "action_quality_contract": train_action_quality_contract(
                        validation_context.train_action_design_policy
                    ),
                    "train_action_design_profiles": {
                        profile_hash: dict(profile)
                        for profile_hash, profile in slot.profile_items
                    },
                }
                if validation_context.train_action_design_policy
                else {}
            ),
            "family_slot_contract": {
                "policy": PROPOSAL_FORMATION_POLICY_VERSION,
                "slot_id": slot_id,
                "target_failure_family": slot.target_family,
                "target_failure_family_hash": stable_hash(
                    {"family": slot.target_family}
                ),
                "target_failure_support_count": len(slot.failures),
                "success_control_count": success_control_count,
                "profile_evidence_hash": slot.profile_evidence_hash,
                "profile_reference_count": len(slot.profile_items),
                "portable_recipe_policy": {
                    "literal_hardcoding_allowed_only_when": (
                        "the_identical_literal_is_observed_in_at_least_two_"
                        "target_family_train_failures"
                    ),
                    "minimum_same_family_train_evidence_for_literal": 2,
                    "otherwise_extract_from": "current_task_or_artifact",
                    "preferred_allowlisted_profile_primitives": (
                        preferred_primitives
                    ),
                    "failed_profile_primitives_to_avoid": failed_primitives,
                    "reusable_preferred_primitive_count": (
                        slot.reusable_preferred_primitive_count
                    ),
                    "profile_primitive_allowlist": [
                        "executable",
                        "environment_os_package",
                        "environment_python_package",
                        "artifact_task_local_path",
                        "artifact_copied_file",
                        "artifact_environment_source_file",
                        "artifact_command_path",
                    ],
                    "validation_features_used": False,
                    "validation_outcomes_used": False,
                    "verifier_content_used": False,
                    "test_content_used": False,
                },
                "response_field": "hypothesis",
                "response_type": "object",
                "required_count": 1,
                "response_rejection_by_diversity_allowed": False,
                "proposal_retry_by_diversity_allowed": False,
                "validation_features_used": False,
                "validation_outcomes_used": False,
                "verifier_content_used": False,
                "test_content_used": False,
            },
            **(
                {
                    "training_evidence_contract": {
                        "policy": self.contrastive_training_evidence_policy,
                        "label_field": "baseline_success",
                        "failure_label": False,
                        "success_control_label": True,
                        "success_control_role": "anti_trigger_negative_control",
                        "context_may_be_used_for_trigger": False,
                        "context_may_shape_actions": True,
                    }
                }
                if self.contrastive_training_evidence_policy
                else {}
            ),
        }

    def _record_matched_proposal_families(
        self,
        programs: Sequence[HypothesisProgram],
        *,
        residuals: Sequence[ResidualExample],
        validation_context: ValidationContext,
        trace_id: str,
        source: str,
        requested_targets: Sequence[str | None],
    ) -> tuple[str | None, ...]:
        if len(programs) != len(requested_targets):
            raise ValueError(
                "family-slot requested targets must align with proposals"
            )
        ranked_slots = _rank_family_proposal_slots(
            residuals,
            profiles=validation_context.action_design_profiles,
            family_use_counts=self._proposal_family_use_counts,
        )
        available_families = {
            slot.target_family for slot in ranked_slots
        }
        seen_requested_targets: set[str] = set()
        canonical_requested_targets: list[str | None] = []
        for target in requested_targets:
            canonical = str(target or "").strip()
            if (
                not canonical
                or canonical not in available_families
                or canonical in seen_requested_targets
            ):
                canonical_requested_targets.append(None)
                continue
            canonical_requested_targets.append(canonical)
            seen_requested_targets.add(canonical)
        requested_target_tuple = tuple(canonical_requested_targets)
        program_hashes = tuple(program.payload_hash for program in programs)
        proposal_set_hash = stable_hash(
            {"candidate_hashes": sorted(program_hashes)}
        )
        requested_target_hashes = [
            stable_hash({"family": target}) if target is not None else None
            for target in requested_target_tuple
        ]
        usage_identity_hash = stable_hash(
            {
                "proposal_set_hash": proposal_set_hash,
                "requested_target_hashes": requested_target_hashes,
            }
        )
        recorded = self._recorded_proposal_family_usage.get(
            usage_identity_hash
        )
        if recorded is not None:
            recorded_program_hashes, recorded_matches = recorded
            actual_match_by_program_hash = dict(
                zip(recorded_program_hashes, recorded_matches)
            )
            replayed_actual_matches = tuple(
                actual_match_by_program_hash.get(program_hash)
                for program_hash in program_hashes
            )
            replayed_actual_hashes = [
                stable_hash({"family": family})
                for family in replayed_actual_matches
                if family is not None
            ]
            self.event_sink.emit(
                Event(
                    event="proposal_family_slot_usage_replayed",
                    stage="proposal.family_slots",
                    trace_id=trace_id,
                    payload={
                        "policy": PROPOSAL_FORMATION_POLICY_VERSION,
                        "source": source,
                        "proposal_set_hash": proposal_set_hash,
                        "usage_identity_hash": usage_identity_hash,
                        "candidate_count": len(programs),
                        "requested_target_count": sum(
                            target is not None
                            for target in requested_target_tuple
                        ),
                        "distinct_requested_target_count": len(
                            {
                                target
                                for target in requested_target_tuple
                                if target is not None
                            }
                        ),
                        "requested_target_set_hash": stable_hash(
                            {
                                "family_hashes": sorted(
                                    target_hash
                                    for target_hash in requested_target_hashes
                                    if target_hash is not None
                                )
                            }
                        ),
                        "actual_matched_count": len(
                            replayed_actual_hashes
                        ),
                        "distinct_actual_matched_family_count": len(
                            set(replayed_actual_hashes)
                        ),
                        "actual_matched_family_set_hash": stable_hash(
                            {
                                "family_hashes": sorted(
                                    replayed_actual_hashes
                                )
                            }
                        ),
                        "proposal_set_replayed": True,
                        "family_use_updated": False,
                        "new_family_use_count": 0,
                        "family_use_count_state_hash": (
                            _family_use_count_state_hash(
                                self._proposal_family_use_counts
                            )
                        ),
                        "validation_features_used": False,
                        "validation_outcomes_used": False,
                        "verifier_content_used": False,
                        "test_content_used": False,
                        "raw_content_persisted": False,
                    },
                )
            )
            return replayed_actual_matches
        actual_matches: list[str | None] = []
        for program in programs:
            actual_match = next(
                (
                    slot.target_family
                    for slot in ranked_slots
                    if any(
                        _program_matches_residual(program, failure)
                        for failure in slot.failures
                    )
                ),
                None,
            )
            actual_matches.append(actual_match)
        for requested_target in requested_target_tuple:
            if requested_target is None:
                continue
            self._proposal_family_use_counts[requested_target] = (
                self._proposal_family_use_counts.get(requested_target, 0) + 1
            )
        result = tuple(actual_matches)
        self._recorded_proposal_family_usage[usage_identity_hash] = (
            program_hashes,
            result,
        )
        actual_match_hashes = [
            stable_hash({"family": family})
            for family in actual_matches
            if family is not None
        ]
        requested_target_count = sum(
            target is not None for target in requested_target_tuple
        )
        self.event_sink.emit(
            Event(
                event="proposal_family_slot_usage_recorded",
                stage="proposal.family_slots",
                trace_id=trace_id,
                payload={
                    "policy": PROPOSAL_FORMATION_POLICY_VERSION,
                    "source": source,
                    "proposal_set_hash": proposal_set_hash,
                    "usage_identity_hash": usage_identity_hash,
                    "candidate_count": len(programs),
                    "requested_target_count": requested_target_count,
                    "distinct_requested_target_count": len(
                        {
                            target
                            for target in requested_target_tuple
                            if target is not None
                        }
                    ),
                    "requested_target_set_hash": stable_hash(
                        {
                            "family_hashes": sorted(
                                target_hash
                                for target_hash in requested_target_hashes
                                if target_hash is not None
                            )
                        }
                    ),
                    "actual_matched_count": len(actual_match_hashes),
                    "distinct_actual_matched_family_count": len(
                        set(actual_match_hashes)
                    ),
                    "actual_matched_family_set_hash": stable_hash(
                        {"family_hashes": sorted(actual_match_hashes)}
                    ),
                    "proposal_set_replayed": False,
                    "family_use_updated": requested_target_count > 0,
                    "new_family_use_count": requested_target_count,
                    "family_use_count_state_hash": (
                        _family_use_count_state_hash(
                            self._proposal_family_use_counts
                        )
                    ),
                    "validation_features_used": False,
                    "validation_outcomes_used": False,
                    "verifier_content_used": False,
                    "test_content_used": False,
                    "raw_content_persisted": False,
                },
            )
        )
        return result

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
        selected_candidate_hypothesis_ids: tuple[str, ...] = (),
    ) -> EvolutionRunResult:
        canonical_selected_ids = tuple(
            sorted(
                set(
                    selected_candidate_hypothesis_ids
                    or ((accepted.id,) if accepted else ())
                )
            )
        )
        result = EvolutionRunResult(
            trace_id=trace_id,
            root_hypothesis_id=root.id,
            accepted_hypothesis_id=(
                canonical_selected_ids[0]
                if canonical_selected_ids
                else (accepted.id if accepted else None)
            ),
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
            selected_candidate_hypothesis_ids=canonical_selected_ids,
        )
        self.event_sink.emit(
            Event(
                event="evolution_generation_completed",
                stage="evolution",
                trace_id=trace_id,
                payload={
                    "root_hypothesis_id": root.id,
                    "accepted_hypothesis_id": result.accepted_hypothesis_id,
                    **(
                        {
                            "selected_candidate_hypothesis_ids": list(
                                canonical_selected_ids
                            ),
                            "candidate_bundle_policy": (
                                self.candidate_bundle_policy
                            ),
                        }
                        if self.candidate_bundle_policy
                        else {}
                    ),
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
                            **(
                                {
                                    "selected_candidate_hypothesis_hashes": [
                                        self.archive.hypotheses[
                                            hypothesis_id
                                        ].payload_hash
                                        for hypothesis_id in canonical_selected_ids
                                        if hypothesis_id
                                        in self.archive.hypotheses
                                    ]
                                }
                                if self.candidate_bundle_policy
                                else {}
                            ),
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


def _residual_stable_order(residual: ResidualExample) -> tuple[str, str, str]:
    return (
        residual.family,
        residual.transition_id,
        stable_hash(
            {
                "features": dict(residual.features),
                "failure_type": residual.failure_type,
                "baseline_success": residual.baseline_success,
            }
        ),
    )


def _family_use_count_state_hash(
    family_use_counts: Mapping[str, int],
) -> str:
    return stable_hash(
        {
            "family_use_counts": sorted(
                (
                    stable_hash({"family": family}),
                    count,
                )
                for family, count in family_use_counts.items()
            )
        }
    )


def _rank_family_proposal_slots(
    residuals: Sequence[ResidualExample],
    *,
    profiles: Mapping[str, Mapping[str, Any]],
    family_use_counts: Mapping[str, int],
) -> tuple[_FamilyProposalSlot, ...]:
    failures_by_family: dict[str, list[ResidualExample]] = {}
    for residual in residuals:
        if residual.split is not SplitName.TRAIN or residual.baseline_success:
            continue
        failures_by_family.setdefault(residual.family, []).append(residual)
    slots: list[_FamilyProposalSlot] = []
    for family, unsorted_failures in failures_by_family.items():
        failures = tuple(sorted(unsorted_failures, key=_residual_stable_order))
        referenced_profile_hashes = tuple(
            sorted(
                {
                    str(profile_hash)
                    for residual in failures
                    if (
                        profile_hash := residual.context.get(
                            "action_context_profile_hash"
                        )
                    )
                }
            )
        )
        profile_items = tuple(
            (profile_hash, profiles[profile_hash])
            for profile_hash in referenced_profile_hashes
            if isinstance(profiles.get(profile_hash), Mapping)
        )
        positive_counts: dict[tuple[str, str], int] = {}
        failed_counts: dict[tuple[str, str], int] = {}
        for residual in failures:
            profile_hash = str(
                residual.context.get("action_context_profile_hash") or ""
            )
            profile = profiles.get(profile_hash)
            if not isinstance(profile, Mapping):
                continue
            positive, failed = _allowlisted_profile_primitives(profile)
            for primitive in positive:
                positive_counts[primitive] = positive_counts.get(primitive, 0) + 1
            for primitive in failed:
                failed_counts[primitive] = failed_counts.get(primitive, 0) + 1
        failed_executables = {
            value.lower()
            for (kind, value), count in failed_counts.items()
            if kind == "executable" and count > 0
        }
        preferred_primitives = tuple(
            sorted(
                (
                    _FamilyProfilePrimitive(
                        kind=kind,
                        value=value,
                        train_failure_evidence_count=count,
                    )
                    for (kind, value), count in positive_counts.items()
                    if (kind, value) not in failed_counts
                    and not (
                        kind
                        in {
                            "executable",
                            "environment_os_package",
                            "environment_python_package",
                        }
                        and value.lower() in failed_executables
                    )
                ),
                key=lambda row: (
                    -row.train_failure_evidence_count,
                    row.kind,
                    row.value,
                ),
            )
        )
        failed_primitives = tuple(
            sorted(
                (
                    _FamilyProfilePrimitive(
                        kind=kind,
                        value=value,
                        train_failure_evidence_count=count,
                    )
                    for (kind, value), count in failed_counts.items()
                ),
                key=lambda row: (
                    -row.train_failure_evidence_count,
                    row.kind,
                    row.value,
                ),
            )
        )
        profile_evidence_hash = stable_hash(
            {
                "profile_references": [
                    {
                        "profile_hash": profile_hash,
                        "profile_payload_hash": stable_hash(dict(profile)),
                    }
                    for profile_hash, profile in profile_items
                ]
            }
        )
        slots.append(
            _FamilyProposalSlot(
                target_family=family,
                failures=failures,
                profile_items=profile_items,
                profile_evidence_hash=profile_evidence_hash,
                preferred_primitives=preferred_primitives,
                failed_primitives=failed_primitives,
                prior_use_count=int(family_use_counts.get(family, 0)),
            )
        )
    return tuple(
        sorted(
            slots,
            key=lambda slot: (
                slot.prior_use_count,
                -int(slot.reusable_preferred_primitive_count > 0),
                -slot.reusable_preferred_primitive_count,
                -len(slot.failures),
                slot.target_family,
                slot.profile_evidence_hash,
            ),
        )
    )


def _allowlisted_profile_primitives(
    profile: Mapping[str, Any],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    preferred: set[tuple[str, str]] = set()
    failed: set[tuple[str, str]] = set()
    environment = profile.get("runtime_environment")
    if isinstance(environment, Mapping):
        environment_fields = {
            "declared_os_packages": "environment_os_package",
            "declared_python_packages": "environment_python_package",
            "declared_task_local_paths": "artifact_task_local_path",
            "copied_task_files": "artifact_copied_file",
            "environment_source_files": "artifact_environment_source_file",
        }
        for field, kind in environment_fields.items():
            values = environment.get(field)
            if not isinstance(values, (list, tuple)):
                continue
            for value in values:
                canonical = _canonical_profile_primitive_value(kind, value)
                if canonical:
                    preferred.add((kind, canonical))
    trace = profile.get("baseline_action_trace")
    command_rows = (
        trace.get("command_signatures") if isinstance(trace, Mapping) else None
    )
    if isinstance(command_rows, (list, tuple)):
        for command in command_rows:
            if not isinstance(command, Mapping):
                continue
            executable = _canonical_profile_primitive_value(
                "executable",
                command.get("executable_basename"),
            )
            status = str(command.get("status") or "").strip().lower()
            exit_code = command.get("exit_code")
            succeeded = status in {
                "success",
                "succeeded",
                "completed",
                "passed",
                "ok",
            } or (
                isinstance(exit_code, int)
                and not isinstance(exit_code, bool)
                and exit_code == 0
            )
            did_fail = status in {
                "failed",
                "failure",
                "error",
                "timed_out",
                "timeout",
            } or (
                isinstance(exit_code, int)
                and not isinstance(exit_code, bool)
                and exit_code != 0
            )
            if did_fail:
                if executable:
                    failed.add(("executable", executable))
                continue
            if not succeeded:
                continue
            if executable:
                preferred.add(("executable", executable))
            task_paths = command.get("task_local_paths")
            if isinstance(task_paths, (list, tuple)):
                for path in task_paths:
                    canonical_path = _canonical_profile_primitive_value(
                        "artifact_command_path",
                        path,
                    )
                    if canonical_path:
                        preferred.add(
                            ("artifact_command_path", canonical_path)
                        )
    failed_executables = {
        value
        for kind, value in failed
        if kind == "executable"
    }
    preferred = {
        (kind, value)
        for kind, value in preferred
        if not (
            kind
            in {
                "executable",
                "environment_os_package",
                "environment_python_package",
            }
            and value in failed_executables
        )
    }
    return preferred, failed


def _canonical_profile_primitive_value(kind: str, value: Any) -> str:
    canonical = str(value or "").strip()
    if not canonical:
        return ""
    if kind in {
        "executable",
        "environment_os_package",
        "environment_python_package",
    }:
        return canonical.lower()
    return canonical


def _family_slot_event_row(
    slot: _FamilyProposalSlot,
    *,
    slot_index: int,
) -> dict[str, Any]:
    preferred_rows = [row.to_dict() for row in slot.preferred_primitives]
    failed_rows = [row.to_dict() for row in slot.failed_primitives]
    return {
        "slot_id": f"train-family-slot-{slot_index}",
        "target_family": slot.target_family,
        "target_family_hash": stable_hash({"family": slot.target_family}),
        "target_failure_support_count": len(slot.failures),
        "prior_family_use_count": slot.prior_use_count,
        "profile_reference_count": len(slot.profile_items),
        "profile_evidence_hash": slot.profile_evidence_hash,
        "preferred_primitive_count": len(preferred_rows),
        "preferred_primitive_set_hash": stable_hash(
            {"primitives": preferred_rows}
        ),
        "reusable_preferred_primitive_count": (
            slot.reusable_preferred_primitive_count
        ),
        "failed_primitive_count": len(failed_rows),
        "failed_primitive_set_hash": stable_hash(
            {"primitives": failed_rows}
        ),
        "raw_content_persisted": False,
    }


def _program_matches_residual(
    program: HypothesisProgram,
    residual: ResidualExample,
) -> bool:
    try:
        return bool(program.matches(residual.features))
    except (TypeError, ValueError, OverflowError):
        return False


def _family_slot_candidate_audit(
    program: HypothesisProgram,
    *,
    target_slot: _FamilyProposalSlot,
    all_slots: Sequence[_FamilyProposalSlot],
) -> dict[str, Any]:
    action_text = " ".join(
        text
        for action in program.action_graph
        for value in (action.target, action.value)
        for text in _nested_action_strings(value)
    ).lower()
    bound_preferred = tuple(
        row
        for row in target_slot.preferred_primitives
        if row.value.lower() in action_text
    )
    bound_failed = tuple(
        row
        for row in target_slot.failed_primitives
        if row.value.lower() in action_text
    )
    portable_delta_kinds: set[str] = set()
    if any(
        row.kind.startswith("executable")
        or row.kind.startswith("environment")
        for row in bound_preferred
    ):
        portable_delta_kinds.add(
            "profile_grounded_executable_or_environment"
        )
    if any(row.kind.startswith("artifact") for row in bound_preferred):
        portable_delta_kinds.add("profile_grounded_artifact")
    if (
        any(
            token in action_text
            for token in ("extract", "read", "parse", "inspect")
        )
        and any(
            token in action_text
            for token in ("current task", "artifact", "file", "path")
        )
    ):
        portable_delta_kinds.add("current_task_or_artifact_extraction")
    target_support = sum(
        _program_matches_residual(program, failure)
        for failure in target_slot.failures
    )
    matched_family_count = sum(
        any(
            _program_matches_residual(program, failure)
            for failure in slot.failures
        )
        for slot in all_slots
    )
    return {
        "candidate_matched_target_support": target_support,
        "candidate_matched_target": target_support > 0,
        "matched_family_count": matched_family_count,
        "profile_binding_count": len(bound_preferred),
        "failed_profile_binding_count": len(bound_failed),
        "portable_delta_kinds": sorted(portable_delta_kinds),
    }


def _nested_action_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            child
            for key, item in value.items()
            for child in (
                *_nested_action_strings(key),
                *_nested_action_strings(item),
            )
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(
            child
            for item in value
            for child in _nested_action_strings(item)
        )
    if value is None:
        return ()
    return (str(value),)


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


def _runner_behavior_set_hash(
    runner: CounterfactualRunner,
    programs: Sequence[HypothesisProgram],
) -> str:
    if not programs:
        raise ValueError("candidate behavior set cannot be empty")
    canonical_programs = tuple(sorted(programs, key=lambda row: row.id))
    backend_hash = getattr(runner, "behavior_set_hash", None)
    if callable(backend_hash):
        try:
            value = str(backend_hash(canonical_programs)).strip()
        except (TypeError, ValueError):
            value = ""
        if value:
            return value
    return stable_hash(
        sorted(
            _runner_behavior_hash(runner, program)
            for program in canonical_programs
        )
    )


def _counterfactual_replay_descriptor(
    *,
    runner: CounterfactualRunner,
    tasks: Sequence[TaskInput],
    programs: Sequence[HypothesisProgram],
    baseline_programs: Sequence[HypothesisProgram],
    split: SplitName,
    program_set: bool,
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
    if not programs:
        raise ValueError("counterfactual replay candidate set cannot be empty")
    candidate_behavior_set_hash = _runner_behavior_set_hash(runner, programs)
    candidate_behavior_hash = _runner_behavior_hash(runner, programs[0])
    if program_set and len(programs) > 1:
        candidate_behavior_hash = candidate_behavior_set_hash
    baseline_behavior_hashes = sorted(
        _runner_behavior_hash(runner, row) for row in baseline_programs
    )
    descriptor: dict[str, object] = {
        "policy": (
            PROGRAM_SET_COUNTERFACTUAL_REPLAY_POLICY_VERSION
            if program_set
            else COUNTERFACTUAL_REPLAY_POLICY_VERSION
        ),
        "split": split.value,
        "evaluator_epoch": evaluator_epoch,
        "runtime_version": runtime_version,
        "candidate_behavior_hash": candidate_behavior_hash,
        "baseline_behavior_set_hash": stable_hash(baseline_behavior_hashes),
        "task_set_hash": stable_hash(task_rows),
    }
    if program_set:
        descriptor["candidate_behavior_set_hash"] = (
            candidate_behavior_set_hash
        )
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
        *COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS,
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


def _training_candidate_bundle_metrics(
    programs: Sequence[HypothesisProgram],
    residuals: Sequence[ResidualExample],
) -> _TrainingCandidateBundleMetrics:
    if not programs:
        raise ValueError("training candidate bundle cannot be empty")
    train_rows = [row for row in residuals if row.split is SplitName.TRAIN]
    failure_rows = [row for row in train_rows if not row.baseline_success]
    success_controls = [row for row in train_rows if row.baseline_success]

    def activation_multiplicity(row: ResidualExample) -> int:
        return sum(program.matches(row.features) for program in programs)

    failure_multiplicities = [
        activation_multiplicity(row) for row in failure_rows
    ]
    success_multiplicities = [
        activation_multiplicity(row) for row in success_controls
    ]
    failure_activation_count = sum(count > 0 for count in failure_multiplicities)
    success_false_positive_activation_count = sum(
        count > 0 for count in success_multiplicities
    )
    complexity = sum(
        metrics.predicate_count + metrics.action_count
        for metrics in (
            _training_candidate_metrics(program, residuals)
            for program in programs
        )
    )
    return _TrainingCandidateBundleMetrics(
        failure_count=len(failure_rows),
        success_control_count=len(success_controls),
        failure_activation_count=failure_activation_count,
        failure_activation_family_count=len(
            {
                row.family
                for row, count in zip(
                    failure_rows,
                    failure_multiplicities,
                    strict=True,
                )
                if count > 0
            }
        ),
        train_family_count=len({row.family for row in train_rows}),
        success_false_positive_activation_count=(
            success_false_positive_activation_count
        ),
        overlap_count=sum(
            max(count - 1, 0)
            for count in (*failure_multiplicities, *success_multiplicities)
        ),
        bundle_size=len(programs),
        complexity=complexity,
    )


def _candidate_audit_set_hash(
    audits: Sequence[_StaticCandidateAudit],
) -> str:
    accepted_programs = tuple(
        audit.accepted for audit in audits if audit.accepted is not None
    )
    if not accepted_programs:
        raise ValueError("candidate audit set cannot be empty")
    return stable_hash(
        {
            "accepted_hypothesis_ids": sorted(
                row.id for row in accepted_programs
            ),
            "accepted_behavior_hashes": sorted(
                _behavior_hash(row) for row in accepted_programs
            ),
        }
    )


def _training_candidate_bundle_audits(
    eligible: Sequence[_StaticCandidateAudit],
    residuals: Sequence[ResidualExample],
    *,
    family_coverage_target: int,
    selection_policy: str = (
        COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSION
    ),
) -> tuple[_TrainingCandidateBundleAudit, ...]:
    if family_coverage_target < 0:
        raise ValueError("training family coverage target cannot be negative")
    if (
        selection_policy
        not in COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSIONS
    ):
        raise ValueError(
            f"unsupported candidate bundle selection policy: {selection_policy}"
        )
    canonical_eligible = tuple(
        sorted(
            eligible,
            key=lambda audit: (
                audit.accepted.id if audit.accepted else audit.root.id
            ),
        )
    )
    bundle_audits: list[_TrainingCandidateBundleAudit] = []
    for bundle_size in range(1, len(canonical_eligible) + 1):
        for members in combinations(canonical_eligible, bundle_size):
            programs = tuple(
                audit.accepted
                for audit in members
                if audit.accepted is not None
            )
            if len(programs) != len(members):
                raise ValueError("bundle selection received an ineligible candidate")
            metrics = _training_candidate_bundle_metrics(programs, residuals)
            canonical_set_hash = _candidate_audit_set_hash(members)
            family_deficit = max(
                family_coverage_target
                - metrics.failure_activation_family_count,
                0,
            )
            ranking_prefix = (
                -metrics.precision,
                family_deficit,
                metrics.success_false_positive_activation_count,
                metrics.overlap_count,
            )
            if (
                selection_policy
                == COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION
            ):
                ranking_score = (
                    *ranking_prefix,
                    -metrics.failure_activation_family_count,
                    -metrics.failure_activation_count,
                    metrics.bundle_size,
                    metrics.complexity,
                    canonical_set_hash,
                )
            else:
                ranking_score = (
                    *ranking_prefix,
                    metrics.bundle_size,
                    -metrics.failure_activation_count,
                    metrics.complexity,
                    canonical_set_hash,
                )
            bundle_audits.append(
                _TrainingCandidateBundleAudit(
                    members=members,
                    metrics=metrics,
                    canonical_set_hash=canonical_set_hash,
                    ranking_score=ranking_score,
                )
            )
    return tuple(sorted(bundle_audits, key=lambda audit: audit.ranking_score))


def _training_candidate_bundle_event_row(
    audit: _TrainingCandidateBundleAudit,
    *,
    family_coverage_target: int,
    selection_policy: str = (
        COMPLEMENTARY_FAMILY_BUNDLE_CANDIDATE_SELECTION_VERSION
    ),
    selected: bool,
) -> dict[str, Any]:
    accepted_programs = tuple(
        row.accepted for row in audit.members if row.accepted is not None
    )
    return {
        "accepted_hypothesis_ids": list(audit.accepted_hypothesis_ids),
        "accepted_hypothesis_hashes": sorted(
            row.payload_hash for row in accepted_programs
        ),
        "accepted_behavior_hashes": sorted(
            _behavior_hash(row) for row in accepted_programs
        ),
        "root_hypothesis_ids": sorted(row.root.id for row in audit.members),
        "root_hypothesis_hashes": sorted(
            row.root.payload_hash for row in audit.members
        ),
        "canonical_set_hash": audit.canonical_set_hash,
        "union_training_metrics": audit.metrics.to_dict(
            family_coverage_target=family_coverage_target
        ),
        "ranking_priority": (
            [
                "precision_desc",
                "family_target_deficit_asc",
                "success_false_positives_asc",
                "overlap_asc",
                "actual_family_count_desc",
                "failure_support_desc",
                "bundle_size_asc",
                "complexity_asc",
                "canonical_set_hash_asc",
            ]
            if selection_policy
            == COMPLEMENTARY_FAMILY_SUPPORT_BUNDLE_CANDIDATE_SELECTION_VERSION
            else [
                "precision_desc",
                "family_target_deficit_asc",
                "success_false_positives_asc",
                "overlap_asc",
                "bundle_size_asc",
                "failure_support_desc",
                "complexity_asc",
                "canonical_set_hash_asc",
            ]
        ),
        "selected": selected,
        "selection_uses_validation": False,
    }


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
