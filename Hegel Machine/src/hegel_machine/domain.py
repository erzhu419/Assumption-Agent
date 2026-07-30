"""Objects separating observations, claims, bindings, and treatments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from .hashing import canonical_json, stable_hash
from .laws import LawEvaluation
from .schema import EvidenceSplit, LawKind, PreregisteredPrediction


def _require_tuple(value: object, name: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be an immutable tuple")


@dataclass(frozen=True, slots=True)
class StructuralEpisode:
    episode_id: str
    observation_ids: tuple[str, ...]
    object_types: tuple[tuple[str, str], ...]
    role_candidates: tuple[tuple[str, str], ...]
    role_observable_witnesses: tuple[tuple[str, tuple[str, ...]], ...]
    observables_json: str
    scale_id: str
    scope: tuple[str, ...]
    split: EvidenceSplit
    data_cutoff: str

    def __post_init__(self) -> None:
        for name in (
            "observation_ids",
            "object_types",
            "role_candidates",
            "role_observable_witnesses",
            "scope",
        ):
            _require_tuple(getattr(self, name), name)
        if not self.observation_ids or not self.object_types:
            raise ValueError("episode needs observations and typed objects")
        if not self.episode_id or not self.scale_id or not self.scope or not self.data_cutoff:
            raise ValueError("episode needs identity, scale, scope, and cutoff")
        payload = json.loads(self.observables_json)
        if not isinstance(payload, dict) or not payload:
            raise ValueError("episode observables must be a nonempty JSON object")
        if canonical_json(payload) != self.observables_json:
            raise ValueError("episode observables must use canonical JSON")
        object_ids = {object_id for object_id, _ in self.object_types}
        if len(object_ids) != len(self.object_types):
            raise ValueError("episode repeats a typed object")
        if len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError("episode repeats an observation")
        roles = [role for role, _ in self.role_candidates]
        witness_role_list = [
            role for role, _ in self.role_observable_witnesses
        ]
        if len(set(roles)) != len(roles) or len(set(witness_role_list)) != len(
            witness_role_list
        ):
            raise ValueError("episode repeats a role or witness contract")
        role_entities = [entity_id for _, entity_id in self.role_candidates]
        if not role_entities or any(entity_id not in object_ids for entity_id in role_entities):
            raise ValueError("role candidates must reference typed episode objects")
        witness_roles = {role for role, _ in self.role_observable_witnesses}
        if witness_roles != {role for role, _ in self.role_candidates}:
            raise ValueError("each role candidate needs observable witnesses")
        observable_names = set(payload)
        if any(
            not witnesses or not set(witnesses).issubset(observable_names)
            for _, witnesses in self.role_observable_witnesses
        ):
            raise ValueError("role witness cites a missing episode observable")

    @classmethod
    def from_mapping(
        cls,
        *,
        episode_id: str,
        observation_ids: tuple[str, ...],
        object_types: Mapping[str, str],
        role_candidates: Mapping[str, str],
        role_observable_witnesses: Mapping[str, tuple[str, ...]],
        observables: Mapping[str, Any],
        scale_id: str,
        scope: tuple[str, ...],
        split: EvidenceSplit,
        data_cutoff: str,
    ) -> "StructuralEpisode":
        return cls(
            episode_id,
            observation_ids,
            tuple(sorted(object_types.items())),
            tuple(sorted(role_candidates.items())),
            tuple(sorted(role_observable_witnesses.items())),
            canonical_json(observables),
            scale_id,
            scope,
            split,
            data_cutoff,
        )

    @property
    def observables(self) -> dict[str, Any]:
        return json.loads(self.observables_json)

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="episode_")


@dataclass(frozen=True, slots=True)
class LawBinding:
    binding_id: str
    law_id: str
    law_kind: LawKind
    role_assignments: tuple[tuple[str, str], ...]
    source_span_ids: tuple[str, ...]
    scale_id: str

    def __post_init__(self) -> None:
        _require_tuple(self.role_assignments, "role assignments")
        _require_tuple(self.source_span_ids, "source span ids")
        if not self.role_assignments or not self.source_span_ids:
            raise ValueError("binding needs typed roles and evidence spans")
        if not self.binding_id or not self.law_id or not self.scale_id:
            raise ValueError("binding identity and scale are required")
        roles = [role for role, _ in self.role_assignments]
        if len(set(roles)) != len(roles):
            raise ValueError("binding assigns a role more than once")
        if len(set(self.source_span_ids)) != len(self.source_span_ids):
            raise ValueError("binding repeats an evidence span")


@dataclass(frozen=True, slots=True)
class VerifiedLawMatch:
    source_episode_id: str
    law_id: str
    binding_id: str
    bound_entities: tuple[tuple[str, str], ...]
    witnessed_observables: tuple[tuple[str, tuple[str, ...]], ...]
    verified_constraints: tuple[str, ...]
    evaluation: LawEvaluation

    def __post_init__(self) -> None:
        for name in (
            "bound_entities",
            "witnessed_observables",
            "verified_constraints",
        ):
            _require_tuple(getattr(self, name), name)
        if not self.evaluation.passed or self.evaluation.abstained:
            raise ValueError("law match requires a passed executable evaluation")
        if not all(
            (
                self.source_episode_id,
                self.law_id,
                self.binding_id,
                self.verified_constraints,
            )
        ):
            raise ValueError("law match is missing a content binding")

    @property
    def match_id(self) -> str:
        return stable_hash(self, prefix="law_match_")


@dataclass(frozen=True, slots=True)
class ObjectHypothesis:
    hypothesis_id: str
    statement: str
    scope: tuple[str, ...]
    bound_entities: tuple[str, ...]
    observable_predictions: tuple[PreregisteredPrediction, ...]
    counter_predictions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HypothesisClaim:
    claim_id: str
    family_id: str
    statement: str
    scope: tuple[str, ...]
    mechanism_edges: tuple[tuple[str, str, str], ...]
    preconditions: tuple[str, ...]
    predictions: tuple[PreregisteredPrediction, ...]
    counter_predictions: tuple[str, ...]
    discriminating_test_ids: tuple[str, ...]
    description_length: float
    evidence_receipt_ids: tuple[str, ...]
    lineage: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.predictions or not self.counter_predictions:
            raise ValueError("a claim needs positive and counter predictions")
        if not self.discriminating_test_ids:
            raise ValueError("a claim needs a discriminating test")
        if self.description_length < 0 or not isfinite(self.description_length):
            raise ValueError("claim description length must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class TreatmentProgram:
    program_id: str
    source_claim_id: str
    trigger: str
    anti_trigger: str
    action_dag: tuple[tuple[str, str], ...]
    expected_effect: str
    verifier_id: str
    fallback_id: str
    evaluator_epoch: str
    parent_program_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_claim_id:
            raise ValueError("treatment cannot exist without a source claim")
        if not self.verifier_id or not self.fallback_id:
            raise ValueError("treatment needs verifier and fallback")
