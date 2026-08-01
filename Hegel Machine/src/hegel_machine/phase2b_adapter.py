"""Family-neutral-shaped evidence to internal candidate enumeration.

The public caller cannot provide a projection grid.  This adapter derives every
role binding and declared scale hypothesis from the public evidence bundle and
a frozen internal registry.  It stops without returning a partial grid when a
budget, role-registry, or transform-path invariant is violated.

This module deliberately does not contain a generator, answer key, evaluator,
or Phase-2A fixture adapter.  Verifier-specific projection compilation is a
subsequent internal step and is not claimed complete by this enumeration layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
import re
from typing import Mapping
from uuid import UUID

from .hashing import stable_hash
from .phase2b_wire import PublicEvidenceBundle, RoleBinding
from .schema import LawKind, RelationLaw, TheoryState, require_tuple


def _uuid4(value: str, name: str) -> str:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{name} must be an opaque UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{name} must be a canonical lowercase UUIDv4")
    return value


@dataclass(frozen=True, slots=True)
class ObservableChannelBinding:
    quantity_id: str
    observable_id: str

    def __post_init__(self) -> None:
        _uuid4(self.quantity_id, "quantity ID")
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.observable_id) is None:
            raise ValueError("internal observable ID is malformed")


@dataclass(frozen=True, slots=True)
class LawWireBinding:
    law_id: str
    law_kind: LawKind
    family_id: str
    role_ids: tuple[tuple[str, str], ...]
    required_observable_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_tuple(self.role_ids, "law wire role IDs")
        require_tuple(self.required_observable_ids, "required observable IDs")
        if not self.law_id or not self.role_ids or not self.required_observable_ids:
            raise ValueError("law wire binding is incomplete")
        _uuid4(self.family_id, "family ID")
        if self.role_ids != tuple(sorted(self.role_ids)):
            raise ValueError("law wire roles must use canonical order")
        roles = [role for role, _ in self.role_ids]
        wire_ids = [wire_id for _, wire_id in self.role_ids]
        if len(roles) != len(set(roles)) or len(wire_ids) != len(set(wire_ids)):
            raise ValueError("law wire binding repeats a role")
        for _, wire_id in self.role_ids:
            _uuid4(wire_id, "wire role ID")
        if self.required_observable_ids != tuple(
            sorted(self.required_observable_ids)
        ):
            raise ValueError("required observable IDs must use canonical order")


@dataclass(frozen=True, slots=True)
class Phase2BAdapterRegistry:
    theory_version_id: str
    law_bindings: tuple[LawWireBinding, ...]
    observable_channels: tuple[ObservableChannelBinding, ...]
    maximum_candidate_count: int = 50_000

    def __post_init__(self) -> None:
        require_tuple(self.law_bindings, "adapter law bindings")
        require_tuple(self.observable_channels, "adapter observable channels")
        if not self.theory_version_id:
            raise ValueError("adapter registry needs a theory version")
        if {item.law_kind for item in self.law_bindings} != set(LawKind):
            raise ValueError("adapter registry must cover all law families")
        if len({item.law_id for item in self.law_bindings}) != len(
            self.law_bindings
        ):
            raise ValueError("adapter registry repeats a law")
        if len({item.family_id for item in self.law_bindings}) != len(
            self.law_bindings
        ):
            raise ValueError("adapter registry family IDs must be unique")
        quantity_ids = [item.quantity_id for item in self.observable_channels]
        observable_ids = [item.observable_id for item in self.observable_channels]
        if len(quantity_ids) != len(set(quantity_ids)):
            raise ValueError("adapter registry repeats a quantity channel")
        if len(observable_ids) != len(set(observable_ids)):
            raise ValueError("adapter registry repeats an observable channel")
        registered = set(observable_ids)
        if any(
            not set(law.required_observable_ids).issubset(registered)
            for law in self.law_bindings
        ):
            raise ValueError("law binding references an unknown observable channel")
        if (
            isinstance(self.maximum_candidate_count, bool)
            or not isinstance(self.maximum_candidate_count, int)
            or self.maximum_candidate_count <= 0
        ):
            raise ValueError("adapter candidate budget must be positive")

    @property
    def registry_id(self) -> str:
        return stable_hash(self, prefix="phase2b_adapter_registry_")

    @classmethod
    def from_theory(
        cls,
        theory: TheoryState,
        *,
        family_ids: Mapping[LawKind, str],
        role_ids: Mapping[tuple[str, str], str],
        quantity_ids: Mapping[str, str],
        maximum_candidate_count: int = 50_000,
    ) -> "Phase2BAdapterRegistry":
        if set(family_ids) != set(LawKind):
            raise ValueError("family ID registry must cover all law kinds")
        law_bindings = []
        required_observables: set[str] = set()
        for law in theory.relation_laws:
            law_roles = []
            for role in law.roles:
                key = (law.law_id, role)
                if key not in role_ids:
                    raise ValueError(f"missing opaque role ID for {key}")
                law_roles.append((role, role_ids[key]))
            required_observables.update(law.required_observables)
            law_bindings.append(
                LawWireBinding(
                    law_id=law.law_id,
                    law_kind=law.kind,
                    family_id=family_ids[law.kind],
                    role_ids=tuple(sorted(law_roles)),
                    required_observable_ids=tuple(sorted(law.required_observables)),
                )
            )
        if set(quantity_ids) != required_observables:
            raise ValueError("quantity registry must exactly cover theory observables")
        channels = tuple(
            ObservableChannelBinding(quantity_id, observable_id)
            for observable_id, quantity_id in sorted(quantity_ids.items())
        )
        return cls(
            theory_version_id=theory.version_id,
            law_bindings=tuple(sorted(law_bindings, key=lambda item: item.law_id)),
            observable_channels=channels,
            maximum_candidate_count=maximum_candidate_count,
        )


class AdapterDisposition(str, Enum):
    COMPLETE = "complete"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class CandidateHypothesis:
    law_id: str
    law_kind: LawKind
    family_id: str
    role_binding: tuple[tuple[str, str], ...]
    public_binding: tuple[RoleBinding, ...]
    scale_hypothesis_id: str
    transform_path_ids: tuple[str, ...]
    source_observation_ids: tuple[str, ...]
    required_observable_ids: tuple[str, ...]
    registry_id: str

    def __post_init__(self) -> None:
        for name in (
            "role_binding",
            "public_binding",
            "transform_path_ids",
            "source_observation_ids",
            "required_observable_ids",
        ):
            require_tuple(getattr(self, name), f"candidate hypothesis {name}")
        if not all(
            (
                self.law_id,
                self.family_id,
                self.scale_hypothesis_id,
                self.registry_id,
                self.role_binding,
                self.public_binding,
            )
        ):
            raise ValueError("candidate hypothesis identity is incomplete")
        _uuid4(self.family_id, "candidate family ID")
        _uuid4(self.scale_hypothesis_id, "candidate scale hypothesis ID")
        if self.role_binding != tuple(sorted(self.role_binding)):
            raise ValueError("candidate role binding must be canonical")
        if self.public_binding != tuple(
            sorted(self.public_binding, key=lambda item: item.role_id)
        ):
            raise ValueError("candidate public binding must be canonical")

    @property
    def candidate_id(self) -> str:
        return stable_hash(self, prefix="phase2b_candidate_hypothesis_")

    @property
    def footprint_id(self) -> str:
        return stable_hash(
            {
                "candidate_id": self.candidate_id,
                "source_observation_ids": self.source_observation_ids,
            },
            prefix="phase2b_candidate_footprint_",
        )


@dataclass(frozen=True, slots=True)
class AdapterEnumerationResult:
    disposition: AdapterDisposition
    reason: str
    bundle_content_id: str
    registry_id: str
    candidate_budget: int
    hypotheses: tuple[CandidateHypothesis, ...]

    def __post_init__(self) -> None:
        require_tuple(self.hypotheses, "adapter hypotheses")
        if not self.reason or not self.bundle_content_id or not self.registry_id:
            raise ValueError("adapter result identity is incomplete")
        if self.disposition is AdapterDisposition.COMPLETE and not self.hypotheses:
            raise ValueError("complete adapter result cannot be empty")
        if self.disposition is AdapterDisposition.ABSTAIN and self.hypotheses:
            raise ValueError("abstaining adapter cannot return a partial candidate grid")

    @property
    def result_id(self) -> str:
        return stable_hash(self, prefix="phase2b_adapter_result_")

    @property
    def candidate_grid_commitment(self):
        """Bind the selector to every hypothesis in this complete enumeration."""

        if self.disposition is not AdapterDisposition.COMPLETE:
            raise ValueError("an abstaining adapter has no candidate-grid commitment")
        from .phase2b_selector import CandidateGridCell, CandidateGridCommitment

        return CandidateGridCommitment(
            adapter_result_id=self.result_id,
            bundle_content_id=self.bundle_content_id,
            registry_id=self.registry_id,
            expected_cells=tuple(
                CandidateGridCell(
                    candidate_id=item.candidate_id,
                    law_kind=item.law_kind,
                    role_binding=item.role_binding,
                    scale_hypothesis_id=item.scale_hypothesis_id,
                    footprint_id=item.footprint_id,
                )
                for item in self.hypotheses
            ),
        )


def _abstain(
    reason: str,
    bundle: PublicEvidenceBundle,
    registry: Phase2BAdapterRegistry,
) -> AdapterEnumerationResult:
    return AdapterEnumerationResult(
        disposition=AdapterDisposition.ABSTAIN,
        reason=reason,
        bundle_content_id=bundle.content_id,
        registry_id=registry.registry_id,
        candidate_budget=registry.maximum_candidate_count,
        hypotheses=(),
    )


def _unique_transform_paths(
    bundle: PublicEvidenceBundle,
) -> dict[str, tuple[str, ...]] | None:
    graph = bundle.aggregation_graph
    outgoing: dict[str, list[tuple[str, str]]] = {
        scale_id: [] for scale_id in graph.scale_ids
    }
    for edge in graph.edges:
        outgoing[edge.source_scale_id].append(
            (edge.target_scale_id, edge.transform_id)
        )
    paths: dict[str, set[tuple[str, ...]]] = {
        scale_id: set() for scale_id in graph.scale_ids
    }
    frontier = [(root, ()) for root in graph.root_scale_ids]
    while frontier:
        scale_id, path = frontier.pop()
        if path in paths[scale_id]:
            continue
        paths[scale_id].add(path)
        for target, transform_id in outgoing[scale_id]:
            frontier.append((target, path + (transform_id,)))
    if any(len(scale_paths) != 1 for scale_paths in paths.values()):
        return None
    return {scale_id: next(iter(scale_paths)) for scale_id, scale_paths in paths.items()}


def enumerate_candidate_hypotheses(
    bundle: PublicEvidenceBundle,
    registry: Phase2BAdapterRegistry,
) -> AdapterEnumerationResult:
    """Enumerate a complete internal family × binding × scale grid."""

    bundle_role_ids = set(bundle.role_ids)
    if any(
        not {wire_id for _, wire_id in law.role_ids}.issubset(bundle_role_ids)
        for law in registry.law_bindings
    ):
        return _abstain("registry_role_absent_from_bundle", bundle, registry)
    channel_by_observable = {
        item.observable_id: item.quantity_id for item in registry.observable_channels
    }
    if any(
        channel not in set(bundle.quantity_ids)
        for channel in channel_by_observable.values()
    ):
        return _abstain("registry_quantity_absent_from_bundle", bundle, registry)
    scale_paths = _unique_transform_paths(bundle)
    if scale_paths is None:
        return _abstain("nonunique_transform_path", bundle, registry)

    entities_by_role: dict[str, tuple[str, ...]] = {
        role_id: tuple(
            entity.entity_id
            for entity in bundle.entity_candidates
            if role_id in entity.role_candidate_ids
        )
        for role_id in bundle.role_ids
    }
    projected_count = 0
    binding_registry: list[
        tuple[LawWireBinding, tuple[tuple[str, str, str], ...]]
    ] = []
    for law in registry.law_bindings:
        role_options = []
        for semantic_role, wire_role in law.role_ids:
            options = entities_by_role.get(wire_role, ())
            if not options:
                return _abstain("incomplete_role_candidate_coverage", bundle, registry)
            role_options.append((semantic_role, wire_role, options))
        bindings = []
        for chosen in product(*(item[2] for item in role_options)):
            if len(chosen) != len(set(chosen)):
                continue
            bindings.append(
                tuple(
                    (semantic_role, wire_role, entity_id)
                    for (semantic_role, wire_role, _), entity_id in zip(
                        role_options,
                        chosen,
                        strict=True,
                    )
                )
            )
        if not bindings:
            return _abstain("no_injective_role_binding", bundle, registry)
        projected_count += len(bindings) * len(scale_paths)
        if projected_count > registry.maximum_candidate_count:
            return _abstain("candidate_budget_exceeded", bundle, registry)
        binding_registry.extend((law, binding) for binding in bindings)

    hypotheses = []
    for law, binding in binding_registry:
        bound_entities = {entity_id for _, _, entity_id in binding}
        required_quantities = {
            channel_by_observable[item] for item in law.required_observable_ids
        }
        footprint = tuple(
            sorted(
                observation.observation_id
                for observation in bundle.observations
                if observation.quantity_id in required_quantities
                and bool(set(observation.entity_ids).intersection(bound_entities))
            )
        )
        for scale_id, transform_path in sorted(scale_paths.items()):
            hypotheses.append(
                CandidateHypothesis(
                    law_id=law.law_id,
                    law_kind=law.law_kind,
                    family_id=law.family_id,
                    role_binding=tuple(
                        sorted(
                            (semantic_role, entity_id)
                            for semantic_role, _, entity_id in binding
                        )
                    ),
                    public_binding=tuple(
                        sorted(
                            (
                                RoleBinding(wire_role, entity_id)
                                for _, wire_role, entity_id in binding
                            ),
                            key=lambda item: item.role_id,
                        )
                    ),
                    scale_hypothesis_id=scale_id,
                    transform_path_ids=transform_path,
                    source_observation_ids=footprint,
                    required_observable_ids=law.required_observable_ids,
                    registry_id=registry.registry_id,
                )
            )
    canonical = tuple(sorted(hypotheses, key=lambda item: item.candidate_id))
    return AdapterEnumerationResult(
        disposition=AdapterDisposition.COMPLETE,
        reason="complete_internal_candidate_grid",
        bundle_content_id=bundle.content_id,
        registry_id=registry.registry_id,
        candidate_budget=registry.maximum_candidate_count,
        hypotheses=canonical,
    )


__all__ = (
    "AdapterDisposition",
    "AdapterEnumerationResult",
    "CandidateHypothesis",
    "LawWireBinding",
    "ObservableChannelBinding",
    "Phase2BAdapterRegistry",
    "enumerate_candidate_hypotheses",
)
