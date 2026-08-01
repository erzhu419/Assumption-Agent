"""Phase-2A controlled typed-selector mechanics qualification.

The recognizer receives a uniform envelope of frozen family/role/scale
projections.  The answer key lives only in this benchmark evaluator.  Raw
evidence extraction, external sealing, and open-world law discovery remain out
of scope and are reported as such in the generated artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from .bootstrap import initial_theory
from .hashing import canonical_json, stable_hash
from .milestones import (
    CURRENT_SCALE_CAPABILITY_NAME,
    CURRENT_TYPED_SELECTION_CAPABILITY_NAME,
    PHASE2A,
    PHASE2A_LEGACY_REPORT_STATUS,
)
from .recognition import (
    RecognitionDecision,
    RecognitionDisposition,
    RecognitionPolicy,
    RECOGNITION_IMPLEMENTATION_ID,
    StructuralProjection,
    UnboundStructuralEpisode,
    recognize_structural_law,
    replay_recognition_decision,
    verify_preservation,
)
from .schema import (
    EvidenceSplit,
    LawKind,
    RelationLaw,
    ScaleContext,
    TheoryState,
    freeze_pairs,
    require_tuple,
)


DYNAMIC_SCALE = "phase2_dynamic_law"
CAUSAL_SCALE = "phase2_causal_mechanism"
EXIT_SCALES = (DYNAMIC_SCALE, CAUSAL_SCALE)
PROJECTION_ADAPTER_IMPLEMENTATION_ID = (
    "projection_adapter_source_sha256_"
    + sha256(Path(__file__).read_bytes()).hexdigest()
)


PASS_OBSERVABLES: Mapping[LawKind, Mapping[str, Any]] = {
    LawKind.SYMMETRY: {
        "forward": (2.0, 4.0),
        "transformed": (2.0, 4.0),
        "common_codomains": True,
    },
    LawKind.MONOTONICITY: {
        "x_low": 1.0,
        "x_high": 2.0,
        "y_low": 3.0,
        "y_high": 5.0,
        "direction": 1.0,
    },
    LawKind.CONSERVATION: {
        "storage_delta": 1.0,
        "inflows": (10.0,),
        "outflows": (9.0,),
        "sources": (),
        "sinks": (),
        "boundary_observed": True,
    },
    LawKind.COMPLEMENTARITY: {
        "u_empty": 0.0,
        "u_a": 1.0,
        "u_b": 1.0,
        "u_ab": 3.0,
        "expected_interaction": 1.0,
        "interaction_margin": 0.5,
    },
    LawKind.NEGATIVE_FEEDBACK: {
        "disturbance_delta": 2.0,
        "response_delta": -1.0,
        "deviation_before_response": 2.0,
        "deviation_after_response": 1.0,
        "controlled_quantity_observed": True,
        "disturbance_precedes_response": True,
        "system_induced_response": True,
        "same_controlled_quantity": True,
        "local_stability_window_observed": True,
        "response_margin": 0.5,
        "mitigation_margin": 0.5,
    },
    LawKind.LOCALITY: {
        "conditional_a": (0.7, 0.3),
        "conditional_b": (0.7, 0.3),
        "blanket_observed": True,
        "same_blanket_state": True,
    },
}


FAIL_OBSERVABLES: Mapping[LawKind, Mapping[str, Any]] = {
    LawKind.SYMMETRY: {
        "forward": (2.0, 4.0),
        "transformed": (2.0, 5.0),
        "common_codomains": True,
    },
    LawKind.MONOTONICITY: {
        "x_low": 1.0,
        "x_high": 2.0,
        "y_low": 5.0,
        "y_high": 3.0,
        "direction": 1.0,
    },
    LawKind.CONSERVATION: {
        "storage_delta": 1.0,
        "inflows": (10.0,),
        "outflows": (7.0,),
        "sources": (),
        "sinks": (),
        "boundary_observed": True,
    },
    LawKind.COMPLEMENTARITY: {
        "u_empty": 0.0,
        "u_a": 1.0,
        "u_b": 1.0,
        "u_ab": 1.0,
        "expected_interaction": 1.0,
        "interaction_margin": 0.5,
    },
    LawKind.NEGATIVE_FEEDBACK: {
        "disturbance_delta": 2.0,
        "response_delta": 1.0,
        "deviation_before_response": 2.0,
        "deviation_after_response": 3.0,
        "controlled_quantity_observed": True,
        "disturbance_precedes_response": True,
        "system_induced_response": True,
        "same_controlled_quantity": True,
        "local_stability_window_observed": True,
        "response_margin": 0.5,
        "mitigation_margin": 0.5,
    },
    LawKind.LOCALITY: {
        "conditional_a": (0.7, 0.3),
        "conditional_b": (0.2, 0.8),
        "blanket_observed": True,
        "same_blanket_state": True,
    },
}


HARD_NEGATIVE_OBSERVABLES: Mapping[LawKind, Mapping[str, Any]] = {
    LawKind.SYMMETRY: {
        "forward": (3.0, 1.0),
        "transformed": (-3.0, 1.0),
        "common_codomains": True,
    },
    LawKind.MONOTONICITY: {
        "x_low": -2.0,
        "x_high": 4.0,
        "y_low": 8.0,
        "y_high": -1.0,
        "direction": 1.0,
    },
    LawKind.CONSERVATION: {
        "storage_delta": -3.0,
        "inflows": (4.0,),
        "outflows": (11.0,),
        "sources": (),
        "sinks": (),
        "boundary_observed": True,
    },
    LawKind.COMPLEMENTARITY: {
        "u_empty": 1.0,
        "u_a": 4.0,
        "u_b": 3.0,
        "u_ab": 5.0,
        "expected_interaction": 1.0,
        "interaction_margin": 1.0,
    },
    LawKind.NEGATIVE_FEEDBACK: {
        "disturbance_delta": 2.0,
        "response_delta": -1.0,
        "deviation_before_response": 2.0,
        "deviation_after_response": 1.0,
        "controlled_quantity_observed": True,
        "disturbance_precedes_response": True,
        "system_induced_response": False,
        "same_controlled_quantity": True,
        "local_stability_window_observed": True,
        "response_margin": 0.5,
        "mitigation_margin": 0.5,
    },
    LawKind.LOCALITY: {
        "conditional_a": (0.9, 0.1),
        "conditional_b": (0.4, 0.6),
        "blanket_observed": True,
        "same_blanket_state": True,
    },
}


@dataclass(frozen=True, slots=True)
class Phase2ExitAnswer:
    """Private evaluator key; never accepted by the recognition API."""

    episode_id: str
    control: str
    expected_disposition: RecognitionDisposition
    expected_kind: LawKind | None
    expected_roles: tuple[tuple[str, str], ...]
    expected_scale_id: str | None
    pair_id: str | None = None


@dataclass(frozen=True, slots=True)
class Phase2ExitThresholds:
    family_classification: float = 0.95
    binding_accuracy: float = 0.95
    scale_selection: float = 0.95
    hard_negative_rejection: float = 0.95
    role_binding_counterfactual_rejection: float = 0.95
    scale_counterfactual_rejection: float = 0.95
    sign_flip_sensitivity: float = 0.95
    deterministic_abstention_accuracy: float = 0.95
    shared_measurement_reuse: float = 1.0
    adapter_replay: float = 1.0
    identifier_value_invariance: float = 1.0
    preservation: float = 1.0

    def __post_init__(self) -> None:
        if any(
            isinstance(getattr(self, item.name), bool)
            or not isinstance(getattr(self, item.name), (int, float))
            or not isfinite(getattr(self, item.name))
            or not 0 <= getattr(self, item.name) <= 1
            for item in fields(self)
        ):
            raise ValueError("Phase-2 exit thresholds must be finite values in [0, 1]")

    @property
    def policy_id(self) -> str:
        return stable_hash(self, prefix="phase2_exit_policy_")


DEFAULT_EXIT_THRESHOLDS = Phase2ExitThresholds()


PHASE2_EXIT_RECOGNITION_POLICY = RecognitionPolicy(
    minimum_normalized_margin=1.0,
    require_complete_family_coverage=True,
    require_completed_binding_competitor=True,
    require_completed_scale_competitor=True,
)


@dataclass(frozen=True, slots=True)
class SharedMeasurement:
    """One anonymous value in the common typed measurement bundle."""

    measurement_id: str
    channel_id: str
    scale_id: str
    bound_entity_ids: tuple[str, ...]
    value_json: str

    def __post_init__(self) -> None:
        require_tuple(self.bound_entity_ids, "measurement bound entities")
        if not all(
            (
                self.measurement_id,
                self.channel_id,
                self.scale_id,
                self.value_json,
            )
        ):
            raise ValueError("shared measurement is missing a content binding")
        value = json.loads(self.value_json)
        if canonical_json(value) != self.value_json:
            raise ValueError("shared measurement value must use canonical JSON")

    @property
    def value(self) -> Any:
        return json.loads(self.value_json)

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="shared_measurement_")


@dataclass(frozen=True, slots=True)
class SharedEvidenceBundle:
    """Uniform verifier-ready witness table used by every candidate adapter."""

    observation_id: str
    typed_entities: tuple[tuple[str, str], ...]
    measurements: tuple[SharedMeasurement, ...]

    def __post_init__(self) -> None:
        require_tuple(self.typed_entities, "shared evidence typed entities")
        require_tuple(self.measurements, "shared evidence measurements")
        if not self.observation_id or not self.typed_entities or not self.measurements:
            raise ValueError("shared evidence bundle must be nonempty")
        entity_ids = tuple(entity_id for entity_id, _ in self.typed_entities)
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("shared evidence repeats an entity")
        measurement_ids = tuple(item.measurement_id for item in self.measurements)
        if len(measurement_ids) != len(set(measurement_ids)):
            raise ValueError("shared evidence repeats a measurement id")
        witness_keys = tuple(
            (item.channel_id, item.scale_id, item.bound_entity_ids)
            for item in self.measurements
        )
        if len(witness_keys) != len(set(witness_keys)):
            raise ValueError("shared evidence repeats a measurement witness key")
        if any(
            not set(item.bound_entity_ids).issubset(entity_ids)
            for item in self.measurements
        ):
            raise ValueError("shared measurement references an unknown entity")
        object.__setattr__(self, "typed_entities", tuple(sorted(self.typed_entities)))
        object.__setattr__(
            self,
            "measurements",
            tuple(sorted(self.measurements, key=lambda item: item.measurement_id)),
        )

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="shared_evidence_")


@dataclass(frozen=True, slots=True)
class FrozenProjectionAdapter:
    """Frozen mapping from anonymous channels to one executable verifier schema."""

    adapter_id: str
    implementation_registry_id: str
    theory_version_id: str
    verifier_registry_id: str
    evaluator_epoch: str
    law_id: str
    law_kind: LawKind
    roles: tuple[str, ...]
    observable_channels: tuple[tuple[str, str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        require_tuple(self.roles, "adapter roles")
        require_tuple(self.observable_channels, "adapter observable channels")
        if self.implementation_registry_id != PROJECTION_ADAPTER_IMPLEMENTATION_ID:
            raise ValueError("projection adapter implementation registry drift")
        if not all(
            (
                self.adapter_id,
                self.implementation_registry_id,
                self.theory_version_id,
                self.verifier_registry_id,
                self.evaluator_epoch,
                self.law_id,
                self.roles,
            )
        ):
            raise ValueError("projection adapter identity is incomplete")
        for binding in self.observable_channels:
            require_tuple(binding, "adapter observable binding")
            if len(binding) != 3:
                raise ValueError("adapter observable binding must have three fields")
            require_tuple(binding[2], "adapter observable witness roles")
            if not set(binding[2]).issubset(self.roles):
                raise ValueError("adapter observable cites an unknown witness role")
        observable_names = tuple(
            name for name, _, _ in self.observable_channels
        )
        channel_ids = tuple(
            channel for _, channel, _ in self.observable_channels
        )
        if (
            not observable_names
            or len(observable_names) != len(set(observable_names))
            or len(channel_ids) != len(set(channel_ids))
        ):
            raise ValueError("adapter observables and channels must be unique")

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="projection_adapter_")

    def resolve_measurements(
        self,
        *,
        bundle: SharedEvidenceBundle,
        role_assignments: Mapping[str, str],
        scale_id: str,
    ) -> tuple[tuple[str, SharedMeasurement], ...]:
        """Resolve observable witnesses without fabricating missing values."""

        if set(role_assignments) != set(self.roles):
            raise ValueError("adapter projection does not cover its role schema")
        lookup = {
            (item.channel_id, item.scale_id, item.bound_entity_ids): item
            for item in bundle.measurements
        }
        resolved = []
        for observable_name, channel_id, witness_roles in self.observable_channels:
            bound_entities = tuple(
                role_assignments[role] for role in witness_roles
            )
            measurement = lookup.get((channel_id, scale_id, bound_entities))
            if measurement is not None:
                resolved.append((observable_name, measurement))
        return tuple(resolved)

    def project(
        self,
        *,
        bundle: SharedEvidenceBundle,
        role_assignments: Mapping[str, str],
        scale_id: str,
        evaluator_epoch: str,
    ) -> StructuralProjection:
        if evaluator_epoch != self.evaluator_epoch:
            raise ValueError("adapter and projection evaluator epochs disagree")
        resolved = self.resolve_measurements(
            bundle=bundle,
            role_assignments=role_assignments,
            scale_id=scale_id,
        )
        observables = {
            observable_name: measurement.value
            for observable_name, measurement in resolved
        }
        projection_nonce = stable_hash(
            (
                bundle.content_id,
                self.content_id,
                tuple(sorted(role_assignments.items())),
                scale_id,
            )
        )[:20]
        return StructuralProjection.from_mapping(
            projection_id=f"projection_{projection_nonce}",
            law_id=self.law_id,
            law_kind=self.law_kind,
            role_assignments=role_assignments,
            scale_id=scale_id,
            evaluator_epoch=evaluator_epoch,
            source_observation_ids=(bundle.observation_id,),
            observables=observables,
        )


@dataclass(frozen=True, slots=True)
class Phase2ExitCase:
    episode: UnboundStructuralEpisode
    evidence: SharedEvidenceBundle
    answer: Phase2ExitAnswer
    projection_replay_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_tuple(self.projection_replay_ids, "projection replay ids")
        if self.episode.episode_id != self.answer.episode_id:
            raise ValueError("benchmark episode and private answer disagree")
        if self.episode.observation_ids != (self.evidence.observation_id,):
            raise ValueError("benchmark episode and evidence provenance disagree")
        if len(self.projection_replay_ids) != len(
            self.episode.candidate_projections
        ):
            raise ValueError("benchmark projection replay coverage is incomplete")


def phase2_exit_theory() -> TheoryState:
    """Freeze a two-scale evaluator derived from, but separate from, v0.1."""

    parent = initial_theory()
    evaluator = replace(
        parent.evaluator,
        evaluator_id="phase2_blinded_recognition_evaluator",
        epoch="phase2_exit_epoch_0002",
        version="0.2.0",
        failure_modes=parent.evaluator.failure_modes
        + ("family_ambiguity", "binding_ambiguity", "scale_trap"),
        adversarial_case_ids=parent.evaluator.adversarial_case_ids
        + ("adv_blinded_family", "adv_scale_projection"),
    )
    scales = (
        ScaleContext(
            scale_id=DYNAMIC_SCALE,
            task_id="blinded_known_law_recognition",
            axes=("typed_episode", "dynamic_response", "declared_time_window"),
            aggregation="maximum_normalized_violation",
            validity_scope=("controlled_uniform_candidate_projection",),
        ),
        ScaleContext(
            scale_id=CAUSAL_SCALE,
            task_id="blinded_known_law_recognition",
            axes=("typed_episode", "causal_roles", "declared_boundary"),
            aggregation="maximum_normalized_violation",
            validity_scope=("controlled_uniform_candidate_projection",),
        ),
    )
    laws = tuple(replace(law, scale_ids=EXIT_SCALES) for law in parent.relation_laws)
    probes = tuple(
        replace(
            probe,
            evaluator_epoch=evaluator.epoch,
            task_ids=("blinded_known_law_recognition",),
        )
        for probe in parent.probes
    )
    return replace(
        parent,
        schema_version="hegel-machine-theory/0.2",
        parent_version_id=parent.version_id,
        relation_laws=laws,
        probes=probes,
        scales=scales,
        evaluator=evaluator,
    )


def frozen_projection_adapters(
    theory: TheoryState,
) -> tuple[FrozenProjectionAdapter, ...]:
    adapters = []
    for law in theory.relation_laws:
        channels = []
        for observable_name in law.required_observables:
            witness_roles = tuple(
                role
                for role, witness_names in law.role_observable_requirements
                if observable_name in witness_names
            )
            channels.append(
                (
                    observable_name,
                    "channel_"
                    + stable_hash(
                        (
                            "phase2_exit_channel_v2",
                            law.violation_functional_id,
                            observable_name,
                            witness_roles,
                        )
                    )[:20],
                    witness_roles,
                )
            )
        channels_tuple = tuple(channels)
        adapters.append(
            FrozenProjectionAdapter(
                adapter_id=(
                    "adapter_"
                    + stable_hash(
                        (
                            "phase2_exit_adapter_v2",
                            law.law_id,
                            law.roles,
                            channels_tuple,
                        )
                    )[:20]
                ),
                implementation_registry_id=PROJECTION_ADAPTER_IMPLEMENTATION_ID,
                theory_version_id=theory.version_id,
                verifier_registry_id=theory.verifier_registry_id,
                evaluator_epoch=theory.evaluator.epoch,
                law_id=law.law_id,
                law_kind=law.kind,
                roles=law.roles,
                observable_channels=channels_tuple,
            )
        )
    return tuple(adapters)


def _opaque_identifier(namespace: str, *parts: object) -> str:
    return f"{namespace}_{stable_hash((namespace, *parts))[:20]}"


def _role_maps(
    law: RelationLaw, *, case_token: str
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    entity_ids = tuple(
        _opaque_identifier("object", case_token, law.kind.value, slot)
        for slot in range(len(law.roles) + 1)
    )
    correct = dict(zip(law.roles, entity_ids[:-1], strict=True))
    wrong = dict(correct)
    wrong[law.roles[-1]] = entity_ids[-1]
    typed = {entity_id: "anonymous_state" for entity_id in entity_ids}
    return correct, wrong, typed


def _semantic_metadata(
    *, target_kind: LawKind, case_token: str
) -> dict[str, float | str | bool]:
    kinds = tuple(LawKind)
    decoy_index = int(stable_hash(("semantic_decoy", case_token))[-4:], 16) % len(
        kinds
    )
    decoy_kind = kinds[decoy_index]
    if decoy_kind is target_kind:
        decoy_kind = kinds[(decoy_index + 1) % len(kinds)]
    scores = {
        f"semantic_score::{kind.value}": 0.10 + index * 0.01
        for index, kind in enumerate(kinds)
    }
    scores[f"semantic_score::{decoy_kind.value}"] = 0.98
    return {
        **scores,
        "semantic_name_hint": _opaque_identifier("lexical", case_token),
        "semantic_control_only": True,
    }


def _payload_for_candidate(
    *,
    law_kind: LawKind,
    target_kind: LawKind,
    control: str,
    is_correct_binding: bool,
    scale_id: str,
    gold_scale_id: str,
    ambiguous_with: LawKind | None,
) -> dict[str, Any]:
    if _candidate_should_pass(
        law_kind=law_kind,
        target_kind=target_kind,
        control=control,
        is_correct_binding=is_correct_binding,
        scale_id=scale_id,
        gold_scale_id=gold_scale_id,
        ambiguous_with=ambiguous_with,
    ):
        return dict(PASS_OBSERVABLES[law_kind])
    if control == "sign_or_constraint_flip":
        return dict(FAIL_OBSERVABLES[law_kind])
    return dict(HARD_NEGATIVE_OBSERVABLES[law_kind])


def _candidate_should_pass(
    *,
    law_kind: LawKind,
    target_kind: LawKind,
    control: str,
    is_correct_binding: bool,
    scale_id: str,
    gold_scale_id: str,
    ambiguous_with: LawKind | None,
) -> bool:
    if not is_correct_binding or scale_id != gold_scale_id:
        return False
    if control in {"low_semantic_positive", "entity_rename"}:
        return law_kind is target_kind
    return control == "ambiguous" and law_kind in {
        target_kind,
        ambiguous_with,
    }


def _binding_counterfactual_payload(law_kind: LawKind) -> dict[str, Any]:
    """Return a violation triggered only by the changed-role footprint."""

    payload = dict(FAIL_OBSERVABLES[law_kind])
    if law_kind is LawKind.MONOTONICITY:
        # The lower-role PASS value y_low=3 is intentionally shared.  A strict
        # y_high drop is therefore needed; the generic FAIL fixture's equality
        # would remain nondecreasing after footprint composition.
        payload["y_high"] = 1.0
    return payload


def _build_case(
    theory: TheoryState,
    *,
    case_token: str,
    target_kind: LawKind,
    control: str,
    gold_scale_id: str,
    ambiguous_with: LawKind | None = None,
) -> Phase2ExitCase:
    episode_id = _opaque_identifier("blind_case", case_token)
    observation_id = _opaque_identifier("observation", case_token)
    expect_match = control in {"low_semantic_positive", "entity_rename"}
    typed_entities: dict[str, str] = {}
    correct_maps: dict[LawKind, dict[str, str]] = {}
    wrong_maps: dict[LawKind, dict[str, str]] = {}
    measurements: dict[
        tuple[str, str, tuple[str, ...]], SharedMeasurement
    ] = {}
    adapters = frozen_projection_adapters(theory)
    adapters_by_kind = {adapter.law_kind: adapter for adapter in adapters}

    for law in theory.relation_laws:
        adapter = adapters_by_kind[law.kind]
        correct_roles, wrong_roles, law_entities = _role_maps(
            law, case_token=case_token
        )
        correct_maps[law.kind] = correct_roles
        wrong_maps[law.kind] = wrong_roles
        typed_entities.update(law_entities)
        for scale_id in EXIT_SCALES:
            correct_payload = _payload_for_candidate(
                law_kind=law.kind,
                target_kind=target_kind,
                control=control,
                is_correct_binding=True,
                scale_id=scale_id,
                gold_scale_id=gold_scale_id,
                ambiguous_with=ambiguous_with,
            )
            correct_passes = _candidate_should_pass(
                law_kind=law.kind,
                target_kind=target_kind,
                control=control,
                is_correct_binding=True,
                scale_id=scale_id,
                gold_scale_id=gold_scale_id,
                ambiguous_with=ambiguous_with,
            )
            wrong_unique_payload = (
                _binding_counterfactual_payload(law.kind)
                if correct_passes
                else _payload_for_candidate(
                    law_kind=law.kind,
                    target_kind=target_kind,
                    control=control,
                    is_correct_binding=False,
                    scale_id=scale_id,
                    gold_scale_id=gold_scale_id,
                    ambiguous_with=ambiguous_with,
                )
            )
            missing_name = (
                law.required_observables[0]
                if control == "missing_evidence" and law.kind is target_kind
                else None
            )
            for (
                observable_name,
                channel_id,
                witness_roles,
            ) in adapter.observable_channels:
                if observable_name == missing_name:
                    continue
                correct_entities = tuple(
                    correct_roles[role] for role in witness_roles
                )
                wrong_entities = tuple(
                    wrong_roles[role] for role in witness_roles
                )
                values_by_entities = {
                    correct_entities: correct_payload[observable_name],
                }
                if wrong_entities != correct_entities:
                    values_by_entities[wrong_entities] = wrong_unique_payload[
                        observable_name
                    ]
                for bound_entities, value in values_by_entities.items():
                    key = (channel_id, scale_id, bound_entities)
                    measurement_nonce = stable_hash(
                        (case_token, *key)
                    )[:20]
                    measurement = SharedMeasurement(
                        measurement_id=f"measurement_{measurement_nonce}",
                        channel_id=channel_id,
                        scale_id=scale_id,
                        bound_entity_ids=bound_entities,
                        value_json=canonical_json(value),
                    )
                    existing = measurements.get(key)
                    if existing is not None and existing != measurement:
                        raise ValueError(
                            "shared evidence assigns two values to one witness key"
                        )
                    measurements[key] = measurement

    evidence = SharedEvidenceBundle(
        observation_id=observation_id,
        typed_entities=tuple(sorted(typed_entities.items())),
        measurements=tuple(measurements.values()),
    )
    projections = []
    for law in theory.relation_laws:
        adapter = adapters_by_kind[law.kind]
        for role_map in (correct_maps[law.kind], wrong_maps[law.kind]):
            for scale_id in EXIT_SCALES:
                projections.append(
                    adapter.project(
                        bundle=evidence,
                        role_assignments=role_map,
                        scale_id=scale_id,
                        evaluator_epoch=theory.evaluator.epoch,
                    )
                )
    if stable_hash(case_token)[-1] in "13579bdf":
        projections.reverse()
    episode = UnboundStructuralEpisode.from_projections(
        episode_id=episode_id,
        observation_ids=(observation_id,),
        typed_entities=dict(evidence.typed_entities),
        candidate_projections=tuple(projections),
        available_scale_ids=EXIT_SCALES,
        evaluator_epoch=theory.evaluator.epoch,
        scope=theory.scope,
        split=EvidenceSplit.VALIDATION,
        data_cutoff=theory.data_cutoff,
        semantic_metadata=_semantic_metadata(
            target_kind=target_kind,
            case_token=case_token,
        ),
    )
    answer = Phase2ExitAnswer(
        episode_id=episode_id,
        control=control,
        expected_disposition=(
            RecognitionDisposition.UNIQUE_MATCH
            if expect_match
            else RecognitionDisposition.ABSTAIN
        ),
        expected_kind=target_kind if expect_match else None,
        expected_roles=(
            tuple(sorted(correct_maps[target_kind].items())) if expect_match else ()
        ),
        expected_scale_id=gold_scale_id if expect_match else None,
        pair_id=(
            stable_hash((target_kind.value, gold_scale_id), prefix="pair_")
            if expect_match
            else None
        ),
    )
    replayed = []
    for law in theory.relation_laws:
        adapter = adapters_by_kind[law.kind]
        for role_map in (correct_maps[law.kind], wrong_maps[law.kind]):
            for scale_id in EXIT_SCALES:
                replayed.append(
                    adapter.project(
                        bundle=evidence,
                        role_assignments=role_map,
                        scale_id=scale_id,
                        evaluator_epoch=theory.evaluator.epoch,
                    ).content_id
                )
    return Phase2ExitCase(
        episode=episode,
        evidence=evidence,
        answer=answer,
        projection_replay_ids=tuple(sorted(replayed)),
    )


def controlled_blinded_cases(
    theory: TheoryState | None = None,
) -> tuple[Phase2ExitCase, ...]:
    theory = theory or phase2_exit_theory()
    cases: list[Phase2ExitCase] = []
    case_index = 0

    def append_case(
        kind: LawKind,
        control: str,
        scale_id: str,
        ambiguous_with: LawKind | None = None,
    ) -> None:
        nonlocal case_index
        case_token = stable_hash(
            ("phase2_exit_case_token_v2", case_index),
            prefix="case_token_",
        )
        cases.append(
            _build_case(
                theory,
                case_token=case_token,
                target_kind=kind,
                control=control,
                gold_scale_id=scale_id,
                ambiguous_with=ambiguous_with,
            )
        )
        case_index += 1

    for kind in LawKind:
        for scale_id in EXIT_SCALES:
            append_case(kind, "low_semantic_positive", scale_id)
            append_case(kind, "entity_rename", scale_id)
        nuisance_scale = EXIT_SCALES[
            int(stable_hash(("nuisance_scale", kind.value))[-1], 16)
            % len(EXIT_SCALES)
        ]
        append_case(kind, "high_semantic_hard_negative", nuisance_scale)
        append_case(kind, "sign_or_constraint_flip", nuisance_scale)
        append_case(kind, "missing_evidence", nuisance_scale)
    append_case(
        LawKind.SYMMETRY,
        "ambiguous",
        DYNAMIC_SCALE,
        ambiguous_with=LawKind.MONOTONICITY,
    )
    return tuple(cases)


def controlled_blinded_corpus(
    theory: TheoryState | None = None,
) -> tuple[tuple[UnboundStructuralEpisode, ...], tuple[Phase2ExitAnswer, ...]]:
    """Return public episodes and a separately held evaluator answer table."""

    cases = controlled_blinded_cases(theory)
    return (
        tuple(case.episode for case in cases),
        tuple(case.answer for case in cases),
    )


def _selected_signature(
    decision: RecognitionDecision,
) -> tuple[LawKind, tuple[tuple[str, str], ...], str] | None:
    proposal = decision.selected_proposal
    if proposal is None:
        return None
    return proposal.law_kind, proposal.role_assignments, proposal.scale_id


def _semantic_control_family(
    episode: UnboundStructuralEpisode,
) -> LawKind | None:
    scores = {
        kind: float(dict(episode.semantic_metadata)[f"semantic_score::{kind.value}"])
        for kind in LawKind
    }
    kind, score = max(scores.items(), key=lambda item: (item[1], item[0].value))
    return kind if score >= 0.50 else None


def _outer_schema_fingerprint(episode: UnboundStructuralEpisode) -> str:
    family_counts = tuple(
        sorted(
            (
                kind.value,
                sum(
                    projection.law_kind is kind
                    for projection in episode.candidate_projections
                ),
            )
            for kind in LawKind
        )
    )
    return stable_hash(
        {
            "episode_fields": tuple(item.name for item in fields(episode)),
            "projection_fields": tuple(
                item.name for item in fields(StructuralProjection)
            ),
            "projection_count": len(episode.candidate_projections),
            "family_counts": family_counts,
            "available_scale_count": len(episode.available_scale_ids),
        },
        prefix="blind_schema_",
    )


def _ratio(passed: int, total: int) -> float:
    if total < 1:
        raise ValueError("a benchmark metric needs at least one case")
    return passed / total


def _candidate_grid_complete(episode: UnboundStructuralEpisode) -> bool:
    """Require the same family x binding x scale candidate grid in every case."""

    if set(episode.available_scale_ids) != set(EXIT_SCALES):
        return False
    for kind in LawKind:
        family = tuple(
            projection
            for projection in episode.candidate_projections
            if projection.law_kind is kind
        )
        role_maps = {projection.role_assignments for projection in family}
        scales = {projection.scale_id for projection in family}
        observed = {
            (projection.role_assignments, projection.scale_id)
            for projection in family
        }
        expected = {
            (role_map, scale_id)
            for role_map in role_maps
            for scale_id in scales
        }
        if (
            len(family) != 4
            or len(role_maps) != 2
            or scales != set(EXIT_SCALES)
            or observed != expected
        ):
            return False
    return True


def _replay_projection_ids(
    *, theory: TheoryState, case: Phase2ExitCase
) -> tuple[str, ...]:
    adapters = {
        adapter.law_id: adapter for adapter in frozen_projection_adapters(theory)
    }
    replayed = []
    for projection in case.episode.candidate_projections:
        adapter = adapters[projection.law_id]
        replayed.append(
            adapter.project(
                bundle=case.evidence,
                role_assignments=dict(projection.role_assignments),
                scale_id=projection.scale_id,
                evaluator_epoch=theory.evaluator.epoch,
            ).content_id
        )
    return tuple(sorted(replayed))


def _measurement_consumption_counts(
    *, theory: TheoryState, case: Phase2ExitCase
) -> dict[str, int]:
    adapters = {
        adapter.law_id: adapter for adapter in frozen_projection_adapters(theory)
    }
    counts: dict[str, int] = {}
    for projection in case.episode.candidate_projections:
        resolved = adapters[projection.law_id].resolve_measurements(
            bundle=case.evidence,
            role_assignments=dict(projection.role_assignments),
            scale_id=projection.scale_id,
        )
        for _, measurement in resolved:
            counts[measurement.content_id] = (
                counts.get(measurement.content_id, 0) + 1
            )
    return counts


def _abstention_reason(control: str) -> str:
    return {
        "high_semantic_hard_negative": "no_passing_proposal",
        "sign_or_constraint_flip": "no_passing_proposal",
        "missing_evidence": "incomplete_family_coverage",
        "ambiguous": "ambiguous_multiple_passing_proposals",
    }[control]


def _rename_episode_identifiers(
    episode: UnboundStructuralEpisode,
) -> tuple[UnboundStructuralEpisode, dict[str, str]]:
    observation_map = {
        observation_id: _opaque_identifier(
            "renamed_observation", episode.content_id, index
        )
        for index, observation_id in enumerate(episode.observation_ids)
    }
    entity_map = {
        entity_id: _opaque_identifier("renamed_object", episode.content_id, index)
        for index, (entity_id, _) in enumerate(episode.typed_entities)
    }
    projections = tuple(
        replace(
            projection,
            projection_id=_opaque_identifier(
                "renamed_projection", episode.content_id, index
            ),
            role_assignments=tuple(
                (role, entity_map[entity_id])
                for role, entity_id in projection.role_assignments
            ),
            source_observation_ids=tuple(
                observation_map[observation_id]
                for observation_id in projection.source_observation_ids
            ),
        )
        for index, projection in enumerate(episode.candidate_projections)
    )
    renamed = replace(
        episode,
        episode_id=_opaque_identifier("renamed_case", episode.content_id),
        observation_ids=tuple(observation_map.values()),
        typed_entities=tuple(
            (entity_map[entity_id], entity_type)
            for entity_id, entity_type in episode.typed_entities
        ),
        candidate_projections=projections,
    )
    return renamed, entity_map


def run_phase2_exit_benchmark(
    thresholds: Phase2ExitThresholds = DEFAULT_EXIT_THRESHOLDS,
) -> dict[str, Any]:
    theory = phase2_exit_theory()
    cases = controlled_blinded_cases(theory)
    episodes = tuple(case.episode for case in cases)
    answers = tuple(case.answer for case in cases)
    answer_by_id = {answer.episode_id: answer for answer in answers}
    decisions = tuple(
        recognize_structural_law(
            theory=theory,
            episode=episode,
            policy=PHASE2_EXIT_RECOGNITION_POLICY,
        )
        for episode in episodes
    )

    records: list[dict[str, Any]] = []
    family_pass = binding_pass = scale_pass = answerable_count = 0
    exact_pass = semantic_exact_pass = 0
    hard_negative_pass = hard_negative_count = 0
    sign_flip_pass = sign_flip_count = 0
    abstain_pass = abstain_count = 0
    role_counter_pass = scale_counter_pass = 0
    adapter_replay_pass = decision_replay_pass = 0
    measurement_reuse_pass = 0
    identifier_invariance_pass = 0
    preservation_pairs: dict[
        str, list[tuple[Phase2ExitCase, RecognitionDecision]]
    ] = {}

    for case, decision in zip(cases, decisions, strict=True):
        episode = case.episode
        answer = answer_by_id[episode.episode_id]
        signature = _selected_signature(decision)
        semantic_kind = _semantic_control_family(episode)
        replay_ids = _replay_projection_ids(theory=theory, case=case)
        adapter_replay_ok = (
            replay_ids == case.projection_replay_ids
            and replay_ids
            == tuple(
                sorted(
                    projection.content_id
                    for projection in episode.candidate_projections
                )
            )
        )
        adapter_replay_pass += int(adapter_replay_ok)
        consumption_counts = _measurement_consumption_counts(
            theory=theory,
            case=case,
        )
        evidence_measurement_ids = {
            measurement.content_id for measurement in case.evidence.measurements
        }
        reused_measurement_count = sum(
            count > 1 for count in consumption_counts.values()
        )
        measurement_reuse_ok = (
            set(consumption_counts) == evidence_measurement_ids
            and reused_measurement_count > 0
        )
        measurement_reuse_pass += int(measurement_reuse_ok)
        try:
            replay_recognition_decision(
                theory=theory,
                episode=episode,
                policy=PHASE2_EXIT_RECOGNITION_POLICY,
                decision=decision,
            )
        except ValueError:
            decision_replay_ok = False
        else:
            decision_replay_ok = True
        decision_replay_pass += int(decision_replay_ok)
        renamed_episode, entity_rename_map = _rename_episode_identifiers(episode)
        renamed_decision = recognize_structural_law(
            theory=theory,
            episode=renamed_episode,
            policy=PHASE2_EXIT_RECOGNITION_POLICY,
        )
        original_selected = decision.selected_proposal
        renamed_selected = renamed_decision.selected_proposal
        if original_selected is None:
            binding_renamed = renamed_selected is None
        else:
            expected_renamed_roles = tuple(
                sorted(
                    (role, entity_rename_map[entity_id])
                    for role, entity_id in original_selected.role_assignments
                )
            )
            binding_renamed = (
                renamed_selected is not None
                and renamed_selected.role_assignments == expected_renamed_roles
            )
        identifier_invariance_ok = (
            renamed_decision.disposition is decision.disposition
            and renamed_decision.reason == decision.reason
            and (
                renamed_selected.law_kind if renamed_selected is not None else None
            )
            is (
                original_selected.law_kind if original_selected is not None else None
            )
            and (
                renamed_selected.scale_id if renamed_selected is not None else None
            )
            == (
                original_selected.scale_id if original_selected is not None else None
            )
            and binding_renamed
            and renamed_decision.normalized_margin == decision.normalized_margin
        )
        identifier_invariance_pass += int(identifier_invariance_ok)

        if answer.expected_disposition is RecognitionDisposition.UNIQUE_MATCH:
            answerable_count += 1
            family_ok = signature is not None and signature[0] is answer.expected_kind
            binding_ok = signature is not None and signature[1] == answer.expected_roles
            scale_ok = signature is not None and signature[2] == answer.expected_scale_id
            family_pass += int(family_ok)
            binding_pass += int(binding_ok)
            scale_pass += int(scale_ok)
            exact_ok = bool(family_ok and binding_ok and scale_ok)
            semantic_ok = semantic_kind is answer.expected_kind
            wrong_binding = tuple(
                item
                for item in decision.evaluated_proposals
                if item.law_kind is answer.expected_kind
                and item.scale_id == answer.expected_scale_id
                and item.role_assignments != answer.expected_roles
            )
            wrong_scale = tuple(
                item
                for item in decision.evaluated_proposals
                if item.law_kind is answer.expected_kind
                and item.role_assignments == answer.expected_roles
                and item.scale_id != answer.expected_scale_id
            )
            role_counter_ok = bool(wrong_binding) and all(
                not item.evaluation.abstained and not item.evaluation.passed
                for item in wrong_binding
            )
            scale_counter_ok = bool(wrong_scale) and all(
                not item.evaluation.abstained and not item.evaluation.passed
                for item in wrong_scale
            )
            role_counter_pass += int(role_counter_ok)
            scale_counter_pass += int(scale_counter_ok)
            if answer.pair_id is not None:
                preservation_pairs.setdefault(answer.pair_id, []).append(
                    (case, decision)
                )
        else:
            family_ok = binding_ok = scale_ok = None
            exact_ok = (
                decision.abstained
                and decision.reason == _abstention_reason(answer.control)
            )
            semantic_ok = semantic_kind is None
            role_counter_ok = scale_counter_ok = None
            abstain_count += 1
            abstain_pass += int(exact_ok)
        exact_pass += int(exact_ok)
        semantic_exact_pass += int(semantic_ok)
        if answer.control == "high_semantic_hard_negative":
            hard_negative_count += 1
            hard_negative_pass += int(
                decision.abstained
                and decision.reason == "no_passing_proposal"
                and all(
                    not proposal.evaluation.abstained
                    and not proposal.evaluation.passed
                    for proposal in decision.evaluated_proposals
                )
            )
        if answer.control == "sign_or_constraint_flip":
            sign_flip_count += 1
            sign_flip_pass += int(
                decision.abstained
                and decision.reason == "no_passing_proposal"
                and all(
                    not proposal.evaluation.abstained
                    and not proposal.evaluation.passed
                    for proposal in decision.evaluated_proposals
                )
            )
        records.append(
            {
                "episode_id": episode.episode_id,
                "episode_content_id": episode.content_id,
                "shared_evidence_id": case.evidence.content_id,
                "projection_replay_set_id": stable_hash(
                    case.projection_replay_ids,
                    prefix="projection_replay_set_",
                ),
                "control": answer.control,
                "preservation_pair_id": answer.pair_id,
                "expected_disposition": answer.expected_disposition.value,
                "expected_kind": (
                    answer.expected_kind.value if answer.expected_kind else None
                ),
                "expected_scale_id": answer.expected_scale_id,
                "decision_id": decision.decision_id,
                "disposition": decision.disposition.value,
                "reason": decision.reason,
                "selected_kind": (
                    signature[0].value if signature is not None else None
                ),
                "selected_scale_id": signature[2] if signature is not None else None,
                "family_correct": family_ok,
                "binding_correct": binding_ok,
                "scale_correct": scale_ok,
                "role_counterfactual_rejected": role_counter_ok,
                "scale_counterfactual_rejected": scale_counter_ok,
                "semantic_control_kind": (
                    semantic_kind.value if semantic_kind is not None else None
                ),
                "semantic_control_exact": semantic_ok,
                "structural_exact": exact_ok,
                "adapter_projection_replay": adapter_replay_ok,
                "shared_measurement_reuse": measurement_reuse_ok,
                "shared_measurement_count": len(case.evidence.measurements),
                "reused_measurement_count": reused_measurement_count,
                "decision_replay": decision_replay_ok,
                "identifier_value_invariance": identifier_invariance_ok,
                "candidate_grid_complete": _candidate_grid_complete(episode),
                "evaluated_projection_count": len(decision.evaluated_proposals),
            }
        )

    witnesses = []
    expected_pair_ids = {
        answer.pair_id for answer in answers if answer.pair_id is not None
    }
    preservation_pair_coverage = bool(expected_pair_ids) and all(
        len(preservation_pairs.get(pair_id, ())) == 2
        for pair_id in expected_pair_ids
    )
    for pair_id in sorted(expected_pair_ids):
        paired = preservation_pairs.get(pair_id, ())
        if len(paired) != 2:
            continue
        source_case, source = paired[0]
        target_case, target = paired[1]
        source_roles = dict(source_case.answer.expected_roles)
        target_roles = dict(target_case.answer.expected_roles)
        common_roles = sorted(set(source_roles).intersection(target_roles))
        entity_map = tuple(
            (source_roles[role], target_roles[role]) for role in common_roles
        )
        scale_map = tuple((scale_id, scale_id) for scale_id in EXIT_SCALES)
        witness = verify_preservation(
            source=source,
            target=target,
            entity_map=entity_map,
            scale_map=scale_map,
            evaluator_epoch=theory.evaluator.epoch,
        )
        witnesses.append(witness)

    # Replace every semantic hint while keeping all structural projections fixed.
    metadata_invariant = True
    for episode, original in zip(episodes, decisions, strict=True):
        perturbed = replace(
            episode,
            semantic_metadata=freeze_pairs(
                {
                    **{
                        f"semantic_score::{kind.value}": 1.0 - index * 0.01
                        for index, kind in enumerate(LawKind)
                    },
                    "semantic_name_hint": "renamed-unrelated-domain",
                    "semantic_control_only": True,
                }
            ),
        )
        repeated = recognize_structural_law(
            theory=theory,
            episode=perturbed,
            policy=PHASE2_EXIT_RECOGNITION_POLICY,
        )
        metadata_invariant = metadata_invariant and (
            repeated.disposition is original.disposition
            and _selected_signature(repeated) == _selected_signature(original)
            and repeated.reason == original.reason
            and repeated.evaluated_proposals == original.evaluated_proposals
            and repeated.normalized_margin == original.normalized_margin
        )

    total_count = len(episodes)
    structural_accuracy = _ratio(exact_pass, total_count)
    semantic_accuracy = _ratio(semantic_exact_pass, total_count)
    metrics = {
        "family_classification_accuracy": _ratio(family_pass, answerable_count),
        "binding_accuracy": _ratio(binding_pass, answerable_count),
        "scale_selection_accuracy": _ratio(scale_pass, answerable_count),
        "hard_negative_rejection": _ratio(hard_negative_pass, hard_negative_count),
        "role_binding_counterfactual_rejection": _ratio(
            role_counter_pass, answerable_count
        ),
        "scale_counterfactual_rejection": _ratio(
            scale_counter_pass, answerable_count
        ),
        "sign_flip_sensitivity": _ratio(sign_flip_pass, sign_flip_count),
        "deterministic_abstention_accuracy": _ratio(abstain_pass, abstain_count),
        "shared_measurement_reuse_accuracy": _ratio(
            measurement_reuse_pass, total_count
        ),
        "adapter_projection_replay_accuracy": _ratio(
            adapter_replay_pass, total_count
        ),
        "decision_replay_accuracy": _ratio(decision_replay_pass, total_count),
        "identifier_value_invariance_accuracy": _ratio(
            identifier_invariance_pass, total_count
        ),
        "cross_episode_preservation": _ratio(
            sum(witness.passed for witness in witnesses), len(expected_pair_ids)
        ),
        "structural_exact_decision_accuracy": structural_accuracy,
        "synthetic_semantic_decoy_accuracy": semantic_accuracy,
        "structural_gain_over_synthetic_decoy": structural_accuracy
        - semantic_accuracy,
    }
    exit_checks = {
        "family_classification": metrics["family_classification_accuracy"]
        >= thresholds.family_classification,
        "binding_accuracy": metrics["binding_accuracy"]
        >= thresholds.binding_accuracy,
        "scale_selection": metrics["scale_selection_accuracy"]
        >= thresholds.scale_selection,
        "hard_negative_rejection": metrics["hard_negative_rejection"]
        >= thresholds.hard_negative_rejection,
        "role_binding_counterfactual_rejection": metrics[
            "role_binding_counterfactual_rejection"
        ]
        >= thresholds.role_binding_counterfactual_rejection,
        "scale_counterfactual_rejection": metrics[
            "scale_counterfactual_rejection"
        ]
        >= thresholds.scale_counterfactual_rejection,
        "sign_flip_sensitivity": metrics["sign_flip_sensitivity"]
        >= thresholds.sign_flip_sensitivity,
        "deterministic_abstention": metrics[
            "deterministic_abstention_accuracy"
        ]
        >= thresholds.deterministic_abstention_accuracy,
        "shared_measurement_reuse": metrics[
            "shared_measurement_reuse_accuracy"
        ]
        >= thresholds.shared_measurement_reuse,
        "adapter_projection_replay": metrics[
            "adapter_projection_replay_accuracy"
        ]
        >= thresholds.adapter_replay,
        "decision_replay": metrics["decision_replay_accuracy"] == 1.0,
        "identifier_value_invariance": metrics[
            "identifier_value_invariance_accuracy"
        ]
        >= thresholds.identifier_value_invariance,
        "preservation": metrics["cross_episode_preservation"]
        >= thresholds.preservation,
        "preservation_pair_coverage": preservation_pair_coverage,
        "uniform_outer_schema": len(
            {_outer_schema_fingerprint(episode) for episode in episodes}
        )
        == 1,
        "complete_candidate_grid": all(
            _candidate_grid_complete(episode) for episode in episodes
        ),
        "semantic_metadata_invariance": metadata_invariant,
    }
    adapters = frozen_projection_adapters(theory)
    report: dict[str, Any] = {
        "benchmark": "phase2_api_blinded_selector_mechanics_v2",
        "schema_version": "hegel-machine-phase2-exit/0.2",
        "milestone_id": PHASE2A.machine_id,
        "milestone_name": PHASE2A.name,
        "capability_name": CURRENT_TYPED_SELECTION_CAPABILITY_NAME,
        "scale_capability_name": CURRENT_SCALE_CAPABILITY_NAME,
        "synthetic": True,
        "development_fixture_only": True,
        "source_visible_generator": True,
        "sealed_holdout": False,
        "formal_phase2_exit_claim": False,
        "context_conditioned_scale_inference_qualified": False,
        "status": (
            PHASE2A_LEGACY_REPORT_STATUS
            if all(exit_checks.values())
            else "engineering_qualification_failed"
        ),
        "claim_scope": (
            "controlled source-visible role-scoped synthetic witness replay and "
            "frozen candidate selection qualification; not family-neutral raw "
            "evidence, raw extraction, sealed holdout, external efficacy, open-world "
            "discovery, or a formal Phase-2 exit"
        ),
        "theory_version_id": theory.version_id,
        "ontology_registry_id": theory.ontology_registry_id,
        "verifier_registry_id": theory.verifier_registry_id,
        "evaluator_epoch": theory.evaluator.epoch,
        "data_cutoff": theory.data_cutoff,
        "evidence_split": EvidenceSplit.VALIDATION.value,
        "threshold_policy_id": thresholds.policy_id,
        "threshold_values": {
            item.name: getattr(thresholds, item.name) for item in fields(thresholds)
        },
        "recognition_policy_id": PHASE2_EXIT_RECOGNITION_POLICY.policy_id,
        "recognition_implementation_id": RECOGNITION_IMPLEMENTATION_ID,
        "recognition_policy": {
            item.name: getattr(PHASE2_EXIT_RECOGNITION_POLICY, item.name)
            for item in fields(PHASE2_EXIT_RECOGNITION_POLICY)
        },
        "adapter_registry_id": stable_hash(
            adapters, prefix="projection_adapter_registry_"
        ),
        "projection_adapter_implementation_id": (
            PROJECTION_ADAPTER_IMPLEMENTATION_ID
        ),
        "case_count": total_count,
        "answerable_case_count": answerable_count,
        "abstention_case_count": abstain_count,
        "projection_count_per_case": len(episodes[0].candidate_projections),
        "law_family_count": len(LawKind),
        "scale_count": len(EXIT_SCALES),
        "expected_preservation_pair_count": len(expected_pair_ids),
        "preservation_mapping_source": (
            "frozen evaluator answer table established before recognition; "
            "scale correspondence is the preregistered identity map"
        ),
        "common_outer_schema": exit_checks["uniform_outer_schema"],
        "outer_schema_fingerprint": _outer_schema_fingerprint(episodes[0]),
        "recognizer_receives_answer_key": False,
        "fixture_values_conditioned_on_evaluator_case_spec": True,
        "independent_raw_evidence_projection_qualified": False,
        "untrusted_recognizer_isolation_qualified": False,
        "candidate_labels_are_hypotheses_not_answers": True,
        "cross_candidate_measurement_reuse_required": True,
        "raw_extractor_qualified": False,
        "semantic_metadata_used_for_acceptance": False,
        "semantic_control_is_real_embedding_baseline": False,
        "abstention_is_statistically_calibrated": False,
        "identifier_blinding_scope": (
            "the recognizer is tested under a consistent renaming of every episode, "
            "observation, entity, and projection id; the generator and evaluator "
            "table remain source-visible, so identifiers are not a secrecy boundary"
        ),
        "active_graph_mutated": False,
        "metrics": metrics,
        "exit_checks": exit_checks,
        "preservation_witness_ids": [witness.witness_id for witness in witnesses],
        "preservation_witnesses": [
            {
                "witness_id": witness.witness_id,
                "source_decision_id": witness.source_decision_id,
                "target_decision_id": witness.target_decision_id,
                "law_id": witness.law_id,
                "entity_map": [list(pair) for pair in witness.entity_map],
                "scale_map": [list(pair) for pair in witness.scale_map],
                "observed_residual_drift": witness.observed_residual_drift,
                "checks": [
                    {"name": name, "passed": passed}
                    for name, passed in witness.checks
                ],
                "passed": witness.passed,
            }
            for witness in witnesses
        ],
        "records": records,
    }
    report["report_id"] = stable_hash(report, prefix="phase2_exit_")
    return report


__all__ = [
    "DEFAULT_EXIT_THRESHOLDS",
    "PHASE2_EXIT_RECOGNITION_POLICY",
    "FrozenProjectionAdapter",
    "Phase2ExitAnswer",
    "Phase2ExitCase",
    "Phase2ExitThresholds",
    "SharedEvidenceBundle",
    "SharedMeasurement",
    "controlled_blinded_cases",
    "controlled_blinded_corpus",
    "frozen_projection_adapters",
    "phase2_exit_theory",
    "run_phase2_exit_benchmark",
]
