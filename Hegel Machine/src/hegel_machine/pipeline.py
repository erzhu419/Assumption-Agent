"""Typed Phase-2 binding and verification pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from .domain import LawBinding, StructuralEpisode, VerifiedLawMatch
from .laws import LawEvaluation, evaluate_law
from .schema import RelationLaw, TheoryState


@dataclass(frozen=True, slots=True)
class VerificationOutcome:
    episode_id: str
    law_id: str
    binding_id: str
    evaluation: LawEvaluation
    match: VerifiedLawMatch | None
    audit_events: tuple[str, ...]


def verify_binding(
    *,
    episode: StructuralEpisode,
    law: RelationLaw,
    binding: LawBinding,
    tolerance: float,
) -> VerificationOutcome:
    events: list[str] = []
    if binding.law_id != law.law_id or binding.law_kind is not law.kind:
        raise ValueError("binding targets a different law")
    bound_roles = tuple(role for role, _ in binding.role_assignments)
    if set(bound_roles) != set(law.roles):
        raise ValueError("binding does not cover the law role schema")
    if binding.scale_id != episode.scale_id or binding.scale_id not in law.scale_ids:
        raise ValueError("binding, episode, and law scales are incompatible")
    if not set(binding.source_span_ids).issubset(episode.observation_ids):
        raise ValueError("binding cites evidence outside the episode")
    object_ids = {object_id for object_id, _ in episode.object_types}
    assigned_entities = [entity_id for _, entity_id in binding.role_assignments]
    if any(entity_id not in object_ids for entity_id in assigned_entities):
        raise ValueError("binding references an untyped or nonexistent entity")
    if len(set(assigned_entities)) != len(assigned_entities):
        raise ValueError("distinct law roles require distinct bound entities")
    expected_assignments = dict(episode.role_candidates)
    if dict(binding.role_assignments) != expected_assignments:
        raise ValueError("binding conflicts with the episode role candidates")
    episode_witnesses = dict(episode.role_observable_witnesses)
    law_witnesses = dict(law.role_observable_requirements)
    for role in law.roles:
        if not set(law_witnesses[role]).issubset(episode_witnesses[role]):
            raise ValueError(f"role {role} lacks its observable witness contract")

    observables = episode.observables
    missing = tuple(
        name for name in law.required_observables if name not in observables
    )
    if missing:
        events.append("abstain:missing_observables:" + ",".join(missing))
    evaluation = evaluate_law(law.kind, observables, tolerance=tolerance)
    if evaluation.abstained:
        events.append("abstain:" + evaluation.reason)
        return VerificationOutcome(
            episode.episode_id,
            law.law_id,
            binding.binding_id,
            evaluation,
            None,
            tuple(events),
        )
    events.append(f"residual:{evaluation.residual:.12g}")
    if not evaluation.passed:
        events.append("rejected_by_executable_violation")
        return VerificationOutcome(
            episode.episode_id,
            law.law_id,
            binding.binding_id,
            evaluation,
            None,
            tuple(events),
        )

    match = VerifiedLawMatch(
        source_episode_id=episode.episode_id,
        law_id=law.law_id,
        binding_id=binding.binding_id,
        bound_entities=binding.role_assignments,
        witnessed_observables=episode.role_observable_witnesses,
        verified_constraints=(law.violation_functional_id,),
        evaluation=evaluation,
    )
    events.append("accepted_verified_law_match")
    return VerificationOutcome(
        episode.episode_id,
        law.law_id,
        binding.binding_id,
        evaluation,
        match,
        tuple(events),
    )


def verify_against_frozen_library(
    *,
    theory: TheoryState,
    episode: StructuralEpisode,
    bindings: tuple[LawBinding, ...],
) -> tuple[VerificationOutcome, ...]:
    """Run only explicitly supplied typed bindings against frozen law code."""

    if not set(episode.scope).intersection(theory.scope):
        raise ValueError("episode is outside the frozen theory scope")
    if episode.data_cutoff != theory.data_cutoff:
        raise ValueError("episode cutoff does not match the frozen theory cutoff")
    registered_scales = {scale.scale_id for scale in theory.scales}
    if episode.scale_id not in registered_scales:
        raise ValueError("episode uses a scale absent from the frozen theory")
    laws = {law.law_id: law for law in theory.relation_laws}
    tolerances = {
        functional.functional_id: functional.tolerance
        for functional in theory.violation_functionals
    }
    outcomes: list[VerificationOutcome] = []
    for binding in bindings:
        if binding.law_id not in laws:
            raise ValueError(f"binding references unknown law: {binding.law_id}")
        law = laws[binding.law_id]
        outcomes.append(
            verify_binding(
                episode=episode,
                law=law,
                binding=binding,
                tolerance=tolerances[law.violation_functional_id],
            )
        )
    return tuple(outcomes)
