"""API-blinded Phase-2 recognition over frozen structural projections.

This module deliberately starts *after* raw extraction.  Every episode carries
the same outer schema and at least one candidate projection for each of the six
frozen law families.  A projection is a hypothesis supplied by an upstream,
frozen adapter; it is not a gold label and it is not evidence that extraction
or open-world discovery has been solved.

Semantic/name metadata is stored separately for provenance and is never read by
the recognizer.  Acceptance requires one and only one passing law/role/scale
proposal, completed evidence for every frozen family, and a preregistered
tolerance-normalized margin.  Otherwise the recognizer abstains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from .hashing import canonical_json, stable_hash
from .laws import LawEvaluation, evaluate_law
from .schema import (
    EvidenceSplit,
    FrozenPairs,
    LawKind,
    Scalar,
    TheoryState,
    freeze_pairs,
    require_tuple,
)


RECOGNITION_IMPLEMENTATION_ID = (
    "recognition_source_sha256_" + sha256(Path(__file__).read_bytes()).hexdigest()
)


@dataclass(frozen=True, slots=True)
class StructuralProjection:
    """A frozen law/role/scale proposal expressed in verifier observables.

    ``law_kind`` is the family being tested, not a hidden answer.  An
    :class:`UnboundStructuralEpisode` must contain every frozen family, so the
    recognizer cannot infer a family merely from a missing proposal.
    """

    projection_id: str
    law_id: str
    law_kind: LawKind
    role_assignments: tuple[tuple[str, str], ...]
    scale_id: str
    evaluator_epoch: str
    source_observation_ids: tuple[str, ...]
    observables_json: str

    def __post_init__(self) -> None:
        require_tuple(self.role_assignments, "projection role assignments")
        require_tuple(
            self.source_observation_ids,
            "projection source observation ids",
        )
        if not all(
            (
                self.projection_id,
                self.law_id,
                self.scale_id,
                self.evaluator_epoch,
                self.source_observation_ids,
            )
        ):
            raise ValueError("projection identity and provenance are required")
        roles = [role for role, _ in self.role_assignments]
        entities = [entity_id for _, entity_id in self.role_assignments]
        if not roles or len(set(roles)) != len(roles):
            raise ValueError("projection roles must be present and unique")
        if len(set(entities)) != len(entities):
            raise ValueError("projection roles require distinct entities")
        if len(set(self.source_observation_ids)) != len(
            self.source_observation_ids
        ):
            raise ValueError("projection repeats a source observation")
        try:
            observables = json.loads(self.observables_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("projection observables must be valid JSON") from exc
        if not isinstance(observables, dict) or not observables:
            raise ValueError("projection observables must be a nonempty object")
        if canonical_json(observables) != self.observables_json:
            raise ValueError("projection observables must use canonical JSON")
        object.__setattr__(
            self,
            "role_assignments",
            tuple(sorted(self.role_assignments)),
        )
        object.__setattr__(
            self,
            "source_observation_ids",
            tuple(sorted(self.source_observation_ids)),
        )

    @classmethod
    def from_mapping(
        cls,
        *,
        projection_id: str,
        law_id: str,
        law_kind: LawKind,
        role_assignments: Mapping[str, str],
        scale_id: str,
        evaluator_epoch: str,
        source_observation_ids: tuple[str, ...],
        observables: Mapping[str, Any],
    ) -> "StructuralProjection":
        return cls(
            projection_id=projection_id,
            law_id=law_id,
            law_kind=law_kind,
            role_assignments=tuple(sorted(role_assignments.items())),
            scale_id=scale_id,
            evaluator_epoch=evaluator_epoch,
            source_observation_ids=source_observation_ids,
            observables_json=canonical_json(observables),
        )

    @property
    def observables(self) -> dict[str, Any]:
        return json.loads(self.observables_json)

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="projection_")


@dataclass(frozen=True, slots=True)
class UnboundStructuralEpisode:
    """Uniform episode envelope without a gold law, role map, or scale."""

    episode_id: str
    observation_ids: tuple[str, ...]
    typed_entities: tuple[tuple[str, str], ...]
    candidate_projections: tuple[StructuralProjection, ...]
    available_scale_ids: tuple[str, ...]
    evaluator_epoch: str
    scope: tuple[str, ...]
    split: EvidenceSplit
    data_cutoff: str
    semantic_metadata: FrozenPairs = ()

    def __post_init__(self) -> None:
        for name in (
            "observation_ids",
            "typed_entities",
            "candidate_projections",
            "available_scale_ids",
            "scope",
            "semantic_metadata",
        ):
            require_tuple(getattr(self, name), f"unbound episode {name}")
        if not all(
            (
                self.episode_id,
                self.observation_ids,
                self.typed_entities,
                self.candidate_projections,
                self.available_scale_ids,
                self.evaluator_epoch,
                self.scope,
                self.data_cutoff,
            )
        ):
            raise ValueError("unbound episode identity and structural envelope are required")

        observation_ids = tuple(sorted(self.observation_ids))
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("unbound episode repeats an observation")
        typed_entities = tuple(sorted(self.typed_entities))
        entity_ids = [entity_id for entity_id, entity_type in typed_entities]
        if any(not entity_id or not entity_type for entity_id, entity_type in typed_entities):
            raise ValueError("typed entities need nonempty ids and types")
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError("unbound episode repeats a typed entity")
        available_scales = tuple(sorted(self.available_scale_ids))
        if len(set(available_scales)) != len(available_scales):
            raise ValueError("unbound episode repeats an available scale")

        projections = tuple(
            sorted(
                self.candidate_projections,
                key=lambda item: (
                    item.law_kind.value,
                    item.law_id,
                    item.scale_id,
                    item.role_assignments,
                    item.projection_id,
                    item.content_id,
                ),
            )
        )
        projection_ids = [item.projection_id for item in projections]
        if len(set(projection_ids)) != len(projection_ids):
            raise ValueError("unbound episode repeats a projection id")
        if {item.law_kind for item in projections} != set(LawKind):
            raise ValueError(
                "unbound episode needs candidate projections for all six law families"
            )
        known_entities = set(entity_ids)
        known_observations = set(observation_ids)
        for projection in projections:
            if projection.evaluator_epoch != self.evaluator_epoch:
                raise ValueError("projection and episode evaluator epochs disagree")
            if projection.scale_id not in available_scales:
                raise ValueError("projection uses an unavailable episode scale")
            if not set(projection.source_observation_ids).issubset(
                known_observations
            ):
                raise ValueError("projection cites evidence outside the episode")
            if any(
                entity_id not in known_entities
                for _, entity_id in projection.role_assignments
            ):
                raise ValueError("projection binds an untyped episode entity")

        object.__setattr__(self, "observation_ids", observation_ids)
        object.__setattr__(self, "typed_entities", typed_entities)
        object.__setattr__(self, "candidate_projections", projections)
        object.__setattr__(self, "available_scale_ids", available_scales)
        object.__setattr__(self, "scope", tuple(sorted(self.scope)))
        object.__setattr__(
            self,
            "semantic_metadata",
            freeze_pairs(self.semantic_metadata),
        )

    @classmethod
    def from_projections(
        cls,
        *,
        episode_id: str,
        observation_ids: tuple[str, ...],
        typed_entities: Mapping[str, str],
        candidate_projections: tuple[StructuralProjection, ...],
        available_scale_ids: tuple[str, ...],
        evaluator_epoch: str,
        scope: tuple[str, ...],
        split: EvidenceSplit,
        data_cutoff: str,
        semantic_metadata: Mapping[str, Scalar] | FrozenPairs = (),
    ) -> "UnboundStructuralEpisode":
        frozen_metadata = (
            freeze_pairs(semantic_metadata)
            if isinstance(semantic_metadata, dict)
            else semantic_metadata
        )
        return cls(
            episode_id=episode_id,
            observation_ids=observation_ids,
            typed_entities=tuple(sorted(typed_entities.items())),
            candidate_projections=candidate_projections,
            available_scale_ids=available_scale_ids,
            evaluator_epoch=evaluator_epoch,
            scope=scope,
            split=split,
            data_cutoff=data_cutoff,
            semantic_metadata=frozen_metadata,
        )

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="unbound_episode_")


@dataclass(frozen=True, slots=True)
class RecognitionPolicy:
    """Frozen separation and evidence-completeness policy."""

    minimum_normalized_margin: float = 1.0
    require_complete_family_coverage: bool = True
    require_completed_binding_competitor: bool = False
    require_completed_scale_competitor: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_normalized_margin, bool)
            or not isinstance(self.minimum_normalized_margin, (int, float))
            or not isfinite(self.minimum_normalized_margin)
            or self.minimum_normalized_margin < 0
        ):
            raise ValueError("recognition margin must be finite and nonnegative")
        flags = (
            self.require_complete_family_coverage,
            self.require_completed_binding_competitor,
            self.require_completed_scale_competitor,
        )
        if any(type(flag) is not bool for flag in flags):
            raise TypeError("recognition policy flags must be booleans")

    @property
    def policy_id(self) -> str:
        return stable_hash(self, prefix="recognition_policy_")


DEFAULT_RECOGNITION_POLICY = RecognitionPolicy()


@dataclass(frozen=True, slots=True)
class EvaluatedProposal:
    projection_id: str
    projection_content_id: str
    law_id: str
    law_kind: LawKind
    role_assignments: tuple[tuple[str, str], ...]
    scale_id: str
    evaluator_epoch: str
    evaluation: LawEvaluation

    def __post_init__(self) -> None:
        require_tuple(self.role_assignments, "evaluated proposal role assignments")
        if not all(
            (
                self.projection_id,
                self.projection_content_id,
                self.law_id,
                self.scale_id,
                self.evaluator_epoch,
            )
        ):
            raise ValueError("evaluated proposal lacks a content binding")
        if self.evaluation.kind is not self.law_kind:
            raise ValueError("proposal and evaluation law kinds disagree")

    @property
    def proposal_id(self) -> str:
        return stable_hash(self, prefix="evaluated_proposal_")

    @property
    def normalized_score(self) -> float | None:
        """Return distance on the proposal's own acceptance-boundary scale.

        A score at or below one is accepted and a score above one is rejected.
        Exact-zero tolerances use a finite fallback: an exact fit scores zero;
        any positive residual scores strictly above the boundary.
        """

        if self.evaluation.abstained:
            return None
        residual = self.evaluation.residual
        if residual is None:
            raise AssertionError("completed evaluation lost its residual")
        if self.evaluation.tolerance > 0:
            score = residual / self.evaluation.tolerance
        else:
            score = 0.0 if residual == 0 else 1.0 + residual
        if not isfinite(score):
            raise ValueError("proposal normalized score is not finite")
        return score


class RecognitionDisposition(str, Enum):
    UNIQUE_MATCH = "unique_verified_match"
    ABSTAIN = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class _DecisionDerivation:
    disposition: RecognitionDisposition
    reason: str
    selected_proposal_id: str | None
    best_normalized_score: float | None
    runner_up_normalized_score: float | None
    normalized_margin: float | None


def _rank_completed(
    proposals: tuple[EvaluatedProposal, ...],
) -> tuple[EvaluatedProposal, ...]:
    completed = tuple(
        proposal for proposal in proposals if not proposal.evaluation.abstained
    )
    return tuple(
        sorted(
            completed,
            key=lambda proposal: (
                proposal.normalized_score,
                proposal.law_kind.value,
                proposal.proposal_id,
            ),
        )
    )


def _derive_decision(
    proposals: tuple[EvaluatedProposal, ...],
    policy: RecognitionPolicy,
) -> _DecisionDerivation:
    completed = tuple(
        proposal for proposal in proposals if not proposal.evaluation.abstained
    )
    ranked = _rank_completed(proposals)
    passing = tuple(proposal for proposal in completed if proposal.evaluation.passed)
    completed_families = {proposal.law_kind for proposal in completed}
    best_score = ranked[0].normalized_score if ranked else None
    runner_up_score = ranked[1].normalized_score if len(ranked) > 1 else None
    normalized_margin = (
        runner_up_score - best_score
        if best_score is not None and runner_up_score is not None
        else None
    )

    selected: EvaluatedProposal | None = None
    if (
        policy.require_complete_family_coverage
        and completed_families != set(LawKind)
    ):
        reason = "incomplete_family_coverage"
    elif not passing:
        reason = "no_passing_proposal"
    elif len(passing) != 1:
        reason = "ambiguous_multiple_passing_proposals"
    else:
        candidate = passing[0]
        binding_competitors = tuple(
            proposal
            for proposal in completed
            if proposal.law_id == candidate.law_id
            and proposal.scale_id == candidate.scale_id
            and proposal.role_assignments != candidate.role_assignments
        )
        scale_competitors = tuple(
            proposal
            for proposal in completed
            if proposal.law_id == candidate.law_id
            and proposal.role_assignments == candidate.role_assignments
            and proposal.scale_id != candidate.scale_id
        )
        competitors = tuple(proposal for proposal in ranked if proposal is not candidate)
        if (
            policy.require_completed_binding_competitor
            and not binding_competitors
        ):
            reason = "missing_completed_binding_competitor"
        elif policy.require_completed_scale_competitor and not scale_competitors:
            reason = "missing_completed_scale_competitor"
        elif not competitors:
            reason = "missing_normalized_competitor"
        else:
            candidate_score = candidate.normalized_score
            competitor_score = competitors[0].normalized_score
            if candidate_score is None or competitor_score is None:
                raise AssertionError("ranked proposals must have normalized scores")
            candidate_margin = competitor_score - candidate_score
            normalized_margin = candidate_margin
            if candidate_margin < policy.minimum_normalized_margin:
                reason = "insufficient_normalized_margin"
            else:
                selected = candidate
                reason = "unique_pass_with_normalized_margin"

    return _DecisionDerivation(
        disposition=(
            RecognitionDisposition.UNIQUE_MATCH
            if selected is not None
            else RecognitionDisposition.ABSTAIN
        ),
        reason=reason,
        selected_proposal_id=(
            selected.proposal_id if selected is not None else None
        ),
        best_normalized_score=best_score,
        runner_up_normalized_score=runner_up_score,
        normalized_margin=normalized_margin,
    )


@dataclass(frozen=True, slots=True)
class RecognitionDecision:
    recognition_implementation_id: str
    theory_version_id: str
    episode_content_id: str
    evaluator_epoch: str
    policy_id: str
    policy_minimum_normalized_margin: float
    policy_require_complete_family_coverage: bool
    policy_require_completed_binding_competitor: bool
    policy_require_completed_scale_competitor: bool
    disposition: RecognitionDisposition
    reason: str
    evaluated_proposals: tuple[EvaluatedProposal, ...]
    selected_proposal_id: str | None
    best_normalized_score: float | None
    runner_up_normalized_score: float | None
    normalized_margin: float | None

    def __post_init__(self) -> None:
        require_tuple(self.evaluated_proposals, "decision evaluated proposals")
        if self.recognition_implementation_id != RECOGNITION_IMPLEMENTATION_ID:
            raise ValueError("decision uses a different recognition implementation")
        if not all(
            (
                self.theory_version_id,
                self.recognition_implementation_id,
                self.episode_content_id,
                self.evaluator_epoch,
                self.policy_id,
                self.reason,
                self.evaluated_proposals,
            )
        ):
            raise ValueError("recognition decision lacks replay bindings")
        proposal_ids = [item.proposal_id for item in self.evaluated_proposals]
        if len(set(proposal_ids)) != len(proposal_ids):
            raise ValueError("recognition decision repeats an evaluated proposal")
        if any(
            item.evaluator_epoch != self.evaluator_epoch
            for item in self.evaluated_proposals
        ):
            raise ValueError("decision combines evaluator epochs")
        policy = RecognitionPolicy(
            minimum_normalized_margin=self.policy_minimum_normalized_margin,
            require_complete_family_coverage=(
                self.policy_require_complete_family_coverage
            ),
            require_completed_binding_competitor=(
                self.policy_require_completed_binding_competitor
            ),
            require_completed_scale_competitor=(
                self.policy_require_completed_scale_competitor
            ),
        )
        if self.policy_id != policy.policy_id:
            raise ValueError("decision policy id disagrees with its frozen values")
        numeric = tuple(
            value
            for value in (
                self.best_normalized_score,
                self.runner_up_normalized_score,
                self.normalized_margin,
            )
            if value is not None
        )
        if any(not isfinite(value) for value in numeric):
            raise ValueError("decision normalized summary must be finite")
        derived = _derive_decision(self.evaluated_proposals, policy)
        actual = (
            self.disposition,
            self.reason,
            self.selected_proposal_id,
            self.best_normalized_score,
            self.runner_up_normalized_score,
            self.normalized_margin,
        )
        expected = (
            derived.disposition,
            derived.reason,
            derived.selected_proposal_id,
            derived.best_normalized_score,
            derived.runner_up_normalized_score,
            derived.normalized_margin,
        )
        if actual != expected:
            raise ValueError("decision disagrees with deterministic derivation")

    @property
    def selected_proposal(self) -> EvaluatedProposal | None:
        return next(
            (
                proposal
                for proposal in self.evaluated_proposals
                if proposal.proposal_id == self.selected_proposal_id
            ),
            None,
        )

    @property
    def abstained(self) -> bool:
        return self.disposition is RecognitionDisposition.ABSTAIN

    @property
    def decision_id(self) -> str:
        return stable_hash(self, prefix="recognition_decision_")


def _proposal_sort_key(proposal: EvaluatedProposal) -> tuple[object, ...]:
    normalized_score = proposal.normalized_score
    return (
        proposal.law_kind.value,
        proposal.law_id,
        proposal.scale_id,
        proposal.role_assignments,
        proposal.projection_id,
        normalized_score is None,
        normalized_score if normalized_score is not None else 0.0,
        proposal.proposal_id,
    )


def recognize_structural_law(
    *,
    theory: TheoryState,
    episode: UnboundStructuralEpisode,
    policy: RecognitionPolicy = DEFAULT_RECOGNITION_POLICY,
) -> RecognitionDecision:
    """Recognize one frozen law/role/scale proposal or abstain.

    There is intentionally no gold family, gold role assignment, gold scale, or
    semantic score parameter.  Invalid registry bindings are contract errors;
    missing measurements and empirical ambiguity produce policy-bound abstention.
    """

    if episode.evaluator_epoch != theory.evaluator.epoch:
        raise ValueError("episode and theory evaluator epochs disagree")
    if episode.data_cutoff != theory.data_cutoff:
        raise ValueError("episode and theory data cutoffs disagree")
    if not set(episode.scope).intersection(theory.scope):
        raise ValueError("episode is outside the frozen theory scope")

    laws = {law.law_id: law for law in theory.relation_laws}
    registered_scales = {scale.scale_id for scale in theory.scales}
    tolerances = {
        functional.functional_id: functional.tolerance
        for functional in theory.violation_functionals
    }
    evaluated: list[EvaluatedProposal] = []
    for projection in episode.candidate_projections:
        try:
            law = laws[projection.law_id]
        except KeyError as exc:
            raise ValueError(
                f"projection references an unknown law: {projection.law_id}"
            ) from exc
        if projection.law_kind is not law.kind:
            raise ValueError("projection law id and family disagree")
        if projection.evaluator_epoch != theory.evaluator.epoch:
            raise ValueError("projection uses a different evaluator epoch")
        if (
            projection.scale_id not in registered_scales
            or projection.scale_id not in law.scale_ids
        ):
            raise ValueError("projection uses an unregistered law scale")
        if set(role for role, _ in projection.role_assignments) != set(law.roles):
            raise ValueError("projection does not cover the law role schema")
        try:
            tolerance = tolerances[law.violation_functional_id]
        except KeyError as exc:
            raise ValueError("law lacks a registered violation tolerance") from exc
        evaluation = evaluate_law(
            law.kind,
            projection.observables,
            tolerance=tolerance,
        )
        evaluated.append(
            EvaluatedProposal(
                projection_id=projection.projection_id,
                projection_content_id=projection.content_id,
                law_id=law.law_id,
                law_kind=law.kind,
                role_assignments=projection.role_assignments,
                scale_id=projection.scale_id,
                evaluator_epoch=projection.evaluator_epoch,
                evaluation=evaluation,
            )
        )

    evaluated_proposals = tuple(sorted(evaluated, key=_proposal_sort_key))
    derived = _derive_decision(evaluated_proposals, policy)
    return RecognitionDecision(
        recognition_implementation_id=RECOGNITION_IMPLEMENTATION_ID,
        theory_version_id=theory.version_id,
        episode_content_id=episode.content_id,
        evaluator_epoch=episode.evaluator_epoch,
        policy_id=policy.policy_id,
        policy_minimum_normalized_margin=policy.minimum_normalized_margin,
        policy_require_complete_family_coverage=(
            policy.require_complete_family_coverage
        ),
        policy_require_completed_binding_competitor=(
            policy.require_completed_binding_competitor
        ),
        policy_require_completed_scale_competitor=(
            policy.require_completed_scale_competitor
        ),
        disposition=derived.disposition,
        reason=derived.reason,
        evaluated_proposals=evaluated_proposals,
        selected_proposal_id=derived.selected_proposal_id,
        best_normalized_score=derived.best_normalized_score,
        runner_up_normalized_score=derived.runner_up_normalized_score,
        normalized_margin=derived.normalized_margin,
    )


def replay_recognition_decision(
    *,
    theory: TheoryState,
    episode: UnboundStructuralEpisode,
    policy: RecognitionPolicy,
    decision: RecognitionDecision,
) -> RecognitionDecision:
    """Recompute a decision from its authoritative inputs or fail closed."""

    recomputed = recognize_structural_law(
        theory=theory,
        episode=episode,
        policy=policy,
    )
    if recomputed != decision:
        raise ValueError("recognition decision fails deterministic replay")
    return recomputed


PRESERVATION_CHECK_SCHEMA: tuple[str, ...] = (
    "both_unique_matches",
    "theory_version_preserved",
    "evaluator_epoch_preserved",
    "law_preserved",
    "scale_preserved",
    "role_map_preserved",
    "residual_preserved",
)


@dataclass(frozen=True, slots=True)
class PreservationWitness:
    source_decision_id: str
    target_decision_id: str
    source_theory_version_id: str
    target_theory_version_id: str
    evaluator_epoch: str
    law_id: str | None
    entity_map: tuple[tuple[str, str], ...]
    scale_map: tuple[tuple[str, str], ...]
    maximum_residual_drift: float
    observed_residual_drift: float | None
    checks: tuple[tuple[str, bool], ...]
    passed: bool

    def __post_init__(self) -> None:
        for name in ("entity_map", "scale_map", "checks"):
            require_tuple(getattr(self, name), f"preservation {name}")
        if not all(
            (
                self.source_decision_id,
                self.target_decision_id,
                self.source_theory_version_id,
                self.target_theory_version_id,
                self.evaluator_epoch,
            )
        ):
            raise ValueError("preservation witness lacks decision and epoch bindings")
        if (
            not isfinite(self.maximum_residual_drift)
            or self.maximum_residual_drift < 0
        ):
            raise ValueError("preservation drift budget must be finite and nonnegative")
        if self.observed_residual_drift is not None and (
            not isfinite(self.observed_residual_drift)
            or self.observed_residual_drift < 0
        ):
            raise ValueError("observed preservation drift must be finite and nonnegative")
        check_names = [name for name, _ in self.checks]
        if tuple(check_names) != PRESERVATION_CHECK_SCHEMA:
            raise ValueError("preservation checks do not match the fixed schema")
        if self.passed is not all(passed for _, passed in self.checks):
            raise ValueError("preservation pass flag disagrees with its checks")

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.checks if not passed)

    @property
    def witness_id(self) -> str:
        return stable_hash(self, prefix="preservation_witness_")


def verify_preservation(
    *,
    source: RecognitionDecision,
    target: RecognitionDecision,
    entity_map: tuple[tuple[str, str], ...],
    scale_map: tuple[tuple[str, str], ...],
    evaluator_epoch: str,
    maximum_residual_drift: float = 0.01,
) -> PreservationWitness:
    """Verify a bounded cross-episode preservation claim, fail closed.

    The witness checks only a concrete law/role/scale correspondence.  It does
    not claim functoriality, cross-domain ontology identity, or raw extraction.
    """

    require_tuple(entity_map, "preservation entity map")
    require_tuple(scale_map, "preservation scale map")
    if not isfinite(maximum_residual_drift) or maximum_residual_drift < 0:
        raise ValueError("preservation drift budget must be finite and nonnegative")

    canonical_entity_map = tuple(sorted(entity_map))
    canonical_scale_map = tuple(sorted(scale_map))
    source_proposal = source.selected_proposal
    target_proposal = target.selected_proposal
    both_selected = (
        source.disposition is RecognitionDisposition.UNIQUE_MATCH
        and target.disposition is RecognitionDisposition.UNIQUE_MATCH
        and source_proposal is not None
        and target_proposal is not None
    )

    epoch_ok = (
        both_selected
        and source.evaluator_epoch == evaluator_epoch
        and target.evaluator_epoch == evaluator_epoch
        and source_proposal.evaluator_epoch == evaluator_epoch
        and target_proposal.evaluator_epoch == evaluator_epoch
    )
    theory_ok = bool(
        both_selected
        and source.theory_version_id == target.theory_version_id
    )
    law_ok = bool(
        both_selected
        and source_proposal.law_id == target_proposal.law_id
        and source_proposal.law_kind is target_proposal.law_kind
    )

    scale_sources = [left for left, _ in canonical_scale_map]
    scale_targets = [right for _, right in canonical_scale_map]
    scale_mapping_valid = (
        len(set(scale_sources)) == len(scale_sources)
        and len(set(scale_targets)) == len(scale_targets)
    )
    scale_lookup = dict(canonical_scale_map) if scale_mapping_valid else {}
    scale_ok = bool(
        both_selected
        and scale_mapping_valid
        and scale_lookup.get(source_proposal.scale_id) == target_proposal.scale_id
    )

    entity_sources = [left for left, _ in canonical_entity_map]
    entity_targets = [right for _, right in canonical_entity_map]
    entity_mapping_valid = (
        len(set(entity_sources)) == len(entity_sources)
        and len(set(entity_targets)) == len(entity_targets)
    )
    entity_lookup = dict(canonical_entity_map) if entity_mapping_valid else {}
    role_ok = False
    if both_selected and entity_mapping_valid:
        source_roles = dict(source_proposal.role_assignments)
        target_roles = dict(target_proposal.role_assignments)
        role_ok = (
            set(source_roles) == set(target_roles)
            and set(entity_lookup) == set(source_roles.values())
            and set(entity_lookup.values()) == set(target_roles.values())
            and all(
                entity_lookup[source_entity] == target_roles[role]
                for role, source_entity in source_roles.items()
            )
        )

    source_residual = (
        source_proposal.evaluation.residual if source_proposal is not None else None
    )
    target_residual = (
        target_proposal.evaluation.residual if target_proposal is not None else None
    )
    observed_drift = (
        abs(source_residual - target_residual)
        if source_residual is not None and target_residual is not None
        else None
    )
    residual_ok = bool(
        both_selected
        and observed_drift is not None
        and observed_drift <= maximum_residual_drift
    )
    checks = (
        ("both_unique_matches", bool(both_selected)),
        ("theory_version_preserved", theory_ok),
        ("evaluator_epoch_preserved", bool(epoch_ok)),
        ("law_preserved", law_ok),
        ("scale_preserved", scale_ok),
        ("role_map_preserved", role_ok),
        ("residual_preserved", residual_ok),
    )
    return PreservationWitness(
        source_decision_id=source.decision_id,
        target_decision_id=target.decision_id,
        source_theory_version_id=source.theory_version_id,
        target_theory_version_id=target.theory_version_id,
        evaluator_epoch=evaluator_epoch,
        law_id=(source_proposal.law_id if law_ok and source_proposal else None),
        entity_map=canonical_entity_map,
        scale_map=canonical_scale_map,
        maximum_residual_drift=maximum_residual_drift,
        observed_residual_drift=observed_drift,
        checks=checks,
        passed=all(passed for _, passed in checks),
    )
