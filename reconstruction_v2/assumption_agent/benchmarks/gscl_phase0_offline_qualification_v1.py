"""Single-lineage, non-scoring offline qualification for the GSCL kernel.

This harness is intentionally iterable.  It is not a formal study, does not
open labels, and does not measure downstream efficacy.  Failures are collected
across all five frozen law families instead of failing fast.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
from itertools import combinations
from typing import Any, Mapping

from assumption_agent.generalized_structural_correspondence_v1 import (
    ConstraintParticipant,
    CorrespondenceDisposition,
    EvidenceSpanRef,
    ExactRational,
    ExecutableLawSchema,
    GSCLSchemaRegistry,
    HARD_NEGATIVE_OPERATOR_CONTRACT_HASHES,
    HyperRoleEndpoint,
    LawBinding,
    ObservableBinding,
    ObservationStatus,
    RESIDUAL_KERNEL_CONTRACT_HASH,
    ResidualComponent,
    ResidualDisposition,
    RoleBinding,
    RoleTargetKind,
    StructuralConstraint,
    StructuralEpisode,
    StructuralHyperrelation,
    StructuralObject,
    StructuralQuantity,
    StructuralRelation,
    TypedObservable,
    build_gscl_schema_registry_v1,
    canonical_structural_signature,
    compare_structural_bindings,
    strict_canonical_bytes,
    strict_content_hash,
    validate_law_binding,
)
from assumption_agent.structural_law_residuals_v1 import (
    ResidualPolicy,
    build_law_residual_receipt,
    evaluate_bound_law,
    evaluate_transformed_law,
    verify_law_residual_receipt_trusted,
)
from assumption_agent.universal_assumption_ontology_v1 import (
    build_universal_assumption_ontology_v1,
)


QUALIFICATION_VERSION = "gscl_phase0_offline_qualification_v1"
QUALIFICATION_CONTRACT = {
    "version": QUALIFICATION_VERSION,
    "scope": "phase0_kernel_only",
    "formal_result": False,
    "efficacy_evidence": False,
    "full_qualification_ready": False,
    "law_count": 5,
    "per_law_cases": [
        "satisfied_primary",
        "entity_renamed_isomorphic_primary",
        "all_frozen_hard_negatives",
        "missing_or_not_applicable",
    ],
    "global_cases": [
        "strict_json_rejection",
        "frozen_registry_remapping_rejection",
        "preregistered_role_observable_semantic_attack_rejection",
        "receipt_builder_internal_recomputation",
        "frozen_hard_negative_operator_derivation",
        "forged_verifier_rejection",
        "false_satisfied_disposition_rejection",
        "safe_private_separation",
        "same_process_byte_exact_replay",
    ],
    "declared_capability_surface": {
        "external_benchmark_source_access": False,
        "model_access": False,
        "network_access": False,
        "api_access": False,
        "online_evaluator_access": False,
        "validation_access": False,
        "test_access": False,
    },
    "runtime_access_audited": False,
}
QUALIFICATION_CONTRACT_HASH = strict_content_hash(
    QUALIFICATION_CONTRACT
)
EXTENDED_QUALIFICATION_VERSION = (
    "gscl_unified_offline_qualification_v1"
)
EXTENDED_QUALIFICATION_CONTRACT = {
    "version": EXTENDED_QUALIFICATION_VERSION,
    "same_iterative_harness_lineage": QUALIFICATION_VERSION,
    "formal_result": False,
    "efficacy_evidence": False,
    "kernel_scope": "phase0_kernel_only",
    "evidence_scope": "controlled_raw_evidence_path",
    "narrative_scope": (
        "source_free_raw_story_to_four_arm_internal_factory_path"
    ),
    "new_formal_study": False,
    "effect_gate_added": False,
    "public_intrinsic_measurement": False,
    "public_intrinsic_freeze_ready": False,
}
EXTENDED_QUALIFICATION_CONTRACT_HASH = strict_content_hash(
    EXTENDED_QUALIFICATION_CONTRACT
)
NARRATIVE_QUALIFICATION_VERSION = (
    "gscl_unified_offline_qualification_v1.narrative_source_free_v1"
)
NARRATIVE_QUALIFICATION_CONTRACT = {
    "version": NARRATIVE_QUALIFICATION_VERSION,
    "same_iterative_harness_lineage": (
        EXTENDED_QUALIFICATION_VERSION
    ),
    "scope": (
        "source_free_raw_story_to_four_arm_internal_factory_path"
    ),
    "formal_result": False,
    "efficacy_evidence": False,
    "new_formal_study": False,
    "effect_gate_added": False,
    "public_intrinsic_measurement": False,
    "public_intrinsic_freeze_ready": False,
    "collect_all": True,
    "same_process_replay_count": 2,
    "checks": [
        "raw_story_to_qwen_contract_test_stub",
        "concrete_frozen_scorer_interface_and_closure",
        "semantic_only_control",
        "legacy_keyword_control",
        "flat_label_no_verifier_control",
        "full_structural_verifier_arm",
        "authoritative_item_cross_binding",
        "supervisor_internal_factory_source_free_adapter",
        "frozen_scorer_subclass_rejection",
        "frozen_scorer_receipt_forgery_rejection",
        "qualification_pack_formal_runtime_rejection",
        "uninitialized_local_qwen_runtime_rejection",
    ],
    "declared_capability_surface": {
        "official_arn_source_access": False,
        "benchmark_label_access": False,
        "frozen_model_execution": False,
        "model_download": False,
        "network_access": False,
        "api_access": False,
        "online_evaluator_access": False,
    },
}
NARRATIVE_QUALIFICATION_CONTRACT_HASH = strict_content_hash(
    NARRATIVE_QUALIFICATION_CONTRACT
)


def _rational(numerator: int, denominator: int = 1) -> dict[str, int]:
    return ExactRational(numerator, denominator).safe_payload()


def _map_payload(rows: Mapping[str, str]) -> dict[str, Any]:
    return {
        "rows": [
            {"source": source, "target": target}
            for source, target in sorted(rows.items())
        ]
    }


def _vector_payload(*values: int) -> dict[str, Any]:
    return {"values": [_rational(value) for value in values]}


def _utility_fold(
    *,
    pair_coefficient: int,
    third_order_coefficient: int,
) -> dict[str, Any]:
    components = (
        "role:component_a",
        "role:component_b",
        "role:component_c",
    )
    rows = []
    for size in range(4):
        for subset_tuple in combinations(components, size):
            subset = frozenset(subset_tuple)
            utility = len(subset)
            if {"role:component_a", "role:component_b"} <= subset:
                utility += pair_coefficient
            if set(components) <= subset:
                utility += third_order_coefficient
            rows.append(
                {
                    "subset": list(sorted(subset)),
                    "utility": _rational(utility),
                }
            )
    return {"rows": rows}


@dataclass(frozen=True)
class FixedLawCase:
    schema: ExecutableLawSchema
    positive_payloads: Mapping[str, Any]
    policy: ResidualPolicy
    abstention_overrides: Mapping[str, Any]
    missing_observable_id: str | None
    expected_abstention: ResidualDisposition


def _case_for_schema(schema: ExecutableLawSchema) -> FixedLawCase:
    if schema.law_kind.value == "equivariance":
        payloads = {
            "input_action": _map_payload(
                {
                    "role:input_before": "role:input_after",
                    "role:input_after": "role:input_before",
                }
            ),
            "output_action": {
                "permutation": [0],
                "signs": [-1],
            },
            "outputs_before": _vector_payload(2),
            "outputs_after": _vector_payload(-2),
        }
        return FixedLawCase(
            schema=schema,
            positive_payloads=payloads,
            policy=ResidualPolicy(law_id=schema.law_id),
            abstention_overrides={},
            missing_observable_id="outputs_after",
            expected_abstention=ResidualDisposition.INCONCLUSIVE,
        )
    if schema.law_kind.value == "monotone_order":
        payloads = {
            "comparable_output_pairs": {
                "pairs": [
                    {"lower": _rational(1), "upper": _rational(2)},
                ]
            },
            "declared_direction": {"direction": 1},
        }
        return FixedLawCase(
            schema=schema,
            positive_payloads=payloads,
            policy=ResidualPolicy(law_id=schema.law_id),
            abstention_overrides={
                "declared_direction": {"direction": 0}
            },
            missing_observable_id=None,
            expected_abstention=ResidualDisposition.NOT_APPLICABLE,
        )
    if schema.law_kind.value == "closed_balance":
        payloads = {
            "boundary_declaration": {
                "boundary_id": "role:system_boundary",
                "complete": True,
            },
            "quantity_ledger": {
                "storage_before": _rational(10),
                "storage_after": _rational(13),
                "inflows": [_rational(5)],
                "outflows": [_rational(2)],
                "sources": [],
                "sinks": [],
            },
        }
        return FixedLawCase(
            schema=schema,
            positive_payloads=payloads,
            policy=ResidualPolicy(law_id=schema.law_id),
            abstention_overrides={
                "boundary_declaration": {
                    "boundary_id": "role:system_boundary",
                    "complete": False,
                }
            },
            missing_observable_id=None,
            expected_abstention=ResidualDisposition.NOT_APPLICABLE,
        )
    if schema.law_kind.value == "path_composition":
        payloads = {
            "finite_domain": {
                "values": [
                    "role:source_state",
                    "local:aux_source",
                ]
            },
            "first_map": _map_payload(
                {
                    "role:source_state": "local:anchor_middle",
                    "local:aux_source": "local:aux_middle",
                }
            ),
            "second_map": _map_payload(
                {
                    "local:anchor_middle": "role:target_state",
                    "local:aux_middle": "local:aux_target",
                }
            ),
            "direct_map": _map_payload(
                {
                    "role:source_state": "role:target_state",
                    "local:aux_source": "local:aux_target",
                }
            ),
        }
        return FixedLawCase(
            schema=schema,
            positive_payloads=payloads,
            policy=ResidualPolicy(law_id=schema.law_id),
            abstention_overrides={},
            missing_observable_id="direct_map",
            expected_abstention=ResidualDisposition.INCONCLUSIVE,
        )
    if schema.law_kind.value == "low_order_interaction":
        positive_fold = _utility_fold(
            pair_coefficient=1,
            third_order_coefficient=0,
        )
        payloads = {
            "components": {
                "values": [
                    "role:component_a",
                    "role:component_b",
                    "role:component_c",
                ]
            },
            "designated_pair": {
                "values": [
                    "role:component_a",
                    "role:component_b",
                ]
            },
            "held_fold_utilities": {
                "folds": [positive_fold, positive_fold]
            },
            "interaction_expectation": {"value": "complementary"},
        }
        return FixedLawCase(
            schema=schema,
            positive_payloads=payloads,
            policy=ResidualPolicy(
                law_id=schema.law_id,
                relation_threshold=ExactRational(1),
            ),
            abstention_overrides={},
            missing_observable_id="held_fold_utilities",
            expected_abstention=ResidualDisposition.INCONCLUSIVE,
        )
    raise KeyError(f"unsupported law kind: {schema.law_kind.value}")


def build_fixed_cases(
    registry: GSCLSchemaRegistry | None = None,
) -> tuple[FixedLawCase, ...]:
    if registry is None:
        ontology = build_universal_assumption_ontology_v1()
        registry = build_gscl_schema_registry_v1(ontology)
    return tuple(
        _case_for_schema(schema)
        for schema in sorted(
            registry.schemas, key=lambda row: row.law_id
        )
    )


def _observable_dimension_unit(
    observable_id: str,
) -> tuple[str, str]:
    if observable_id == "quantity_ledger":
        return ("Mass", "kg")
    if observable_id == "held_fold_utilities":
        return ("Utility", "unitless")
    return ("Scalar", "unitless")


def _semantic_attack_override(
    schema: ExecutableLawSchema,
) -> Mapping[str, Any]:
    if schema.law_kind.value == "equivariance":
        return {"outputs_after": _vector_payload(-2, 999)}
    if schema.law_kind.value == "monotone_order":
        return {
            "comparable_output_pairs": {
                "pairs": [
                    {"lower": _rational(1), "upper": _rational(2)},
                    {"lower": _rational(2), "upper": _rational(3)},
                ]
            }
        }
    if schema.law_kind.value == "closed_balance":
        return {
            "boundary_declaration": {
                "boundary_id": "raw.unbound.boundary",
                "complete": True,
            }
        }
    if schema.law_kind.value == "path_composition":
        return {
            "finite_domain": {
                "values": ["raw.source.id", "local:aux_source"]
            }
        }
    if schema.law_kind.value == "low_order_interaction":
        return {
            "components": {
                "values": [
                    "role:component_a",
                    "role:component_b",
                    "role:unbound_component",
                ]
            }
        }
    raise KeyError(f"unsupported law kind: {schema.law_kind.value}")


def _build_episode_and_binding(
    registry: GSCLSchemaRegistry,
    case: FixedLawCase,
    *,
    case_key: str,
    payload_overrides: Mapping[str, Any] | None = None,
    missing_observable_id: str | None = None,
) -> tuple[StructuralEpisode, LawBinding]:
    schema = case.schema
    source_bytes = (
        f"synthetic offline GSCL evidence for {case_key}".encode("utf-8")
    )
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    span_id = f"{case_key}.span"
    span = EvidenceSpanRef(
        span_id=span_id,
        source_sha256=source_hash,
        start_byte=0,
        end_byte=len(source_bytes),
        span_sha256=source_hash,
    )
    payloads = dict(case.positive_payloads)
    payloads.update(payload_overrides or {})
    observables = []
    for spec in schema.required_observables:
        missing = spec.observable_id == missing_observable_id
        dimension, unit = (
            _observable_dimension_unit(spec.observable_id)
            if spec.unit_required
            else (None, None)
        )
        observables.append(
            TypedObservable(
                observable_id=spec.observable_id,
                value_type=spec.value_type,
                value_payload=(
                    None if missing else payloads[spec.observable_id]
                ),
                evidence_span_ids=() if missing else (span_id,),
                observation_status=(
                    ObservationStatus.UNKNOWN
                    if missing
                    else ObservationStatus.OBSERVED
                ),
                dimension=dimension,
                unit=unit,
            )
        )

    objects = []
    relations = []
    quantities = []
    constraints = []
    role_target_ids: dict[str, str] = {}
    object_role_ids: list[str] = []
    for role in schema.roles:
        if role.target_kind is RoleTargetKind.OBJECT:
            target_id = f"{case_key}.object.{role.role_id}"
            role_target_ids[role.role_id] = target_id
            object_role_ids.append(role.role_id)
            objects.append(
                StructuralObject(
                    object_id=target_id,
                    object_type=role.allowed_target_types[0],
                    evidence_span_ids=(span_id,),
                )
            )
    if len(object_role_ids) < 2:
        raise AssertionError("fixed schemas require at least two object roles")

    object_target_ids = [
        role_target_ids[role_id] for role_id in object_role_ids
    ]
    for role in schema.roles:
        if role.target_kind is RoleTargetKind.RELATION:
            target_id = f"{case_key}.relation.{role.role_id}"
            role_target_ids[role.role_id] = target_id
            relations.append(
                StructuralRelation(
                    relation_id=target_id,
                    relation_type=role.allowed_target_types[0],
                    source_object_id=object_target_ids[0],
                    target_object_id=object_target_ids[1],
                    evidence_span_ids=(span_id,),
                    order_index=0,
                )
            )

    owner_role_by_quantity_role = {
        "output_before": "input_before",
        "output_after": "input_after",
        "lower_value": "lower_state",
        "upper_value": "upper_state",
        "storage_before": "system_boundary",
        "storage_after": "system_boundary",
    }
    quantity_values = {
        "output_before": ExactRational(2),
        "output_after": ExactRational(-2),
        "lower_value": ExactRational(1),
        "upper_value": ExactRational(2),
        "storage_before": ExactRational(10),
        "storage_after": ExactRational(13),
    }
    for role in schema.roles:
        if role.target_kind is RoleTargetKind.QUANTITY:
            target_id = f"{case_key}.quantity.{role.role_id}"
            role_target_ids[role.role_id] = target_id
            owner_role = owner_role_by_quantity_role.get(
                role.role_id, object_role_ids[0]
            )
            dimension, unit = (
                ("Mass", "kg")
                if role.role_id.startswith("storage_")
                else ("Scalar", "unitless")
            )
            quantities.append(
                StructuralQuantity(
                    quantity_id=target_id,
                    owner_object_id=role_target_ids[owner_role],
                    dimension=dimension,
                    unit=unit,
                    value=quantity_values[role.role_id],
                    evidence_span_ids=(span_id,),
                )
            )

    constraint_roles = [
        role
        for role in schema.roles
        if role.target_kind is RoleTargetKind.CONSTRAINT
    ]
    for role in constraint_roles:
        target_id = f"{case_key}.constraint.{role.role_id}"
        role_target_ids[role.role_id] = target_id
        participants = tuple(
            ConstraintParticipant(
                participant_role=other.role_id,
                target_kind=other.target_kind,
                target_id=role_target_ids[other.role_id],
            )
            for other in schema.roles
            if other.role_id != role.role_id
        )
        constraints.append(
            StructuralConstraint(
                constraint_id=target_id,
                constraint_type=role.allowed_target_types[0],
                participants=participants,
                observable_ids=tuple(
                    spec.observable_id
                    for spec in schema.required_observables
                ),
                evidence_span_ids=(span_id,),
            )
        )

    hyperrelation = StructuralHyperrelation(
        hyperrelation_id=f"{case_key}.hyper.joint",
        hyperrelation_type="JointFactor",
        endpoints=tuple(
            HyperRoleEndpoint(
                endpoint_role=role_id,
                object_id=role_target_ids[role_id],
            )
            for role_id in object_role_ids
        ),
        evidence_span_ids=(span_id,),
    )
    boundary_id = role_target_ids.get("system_boundary")
    episode = StructuralEpisode(
        episode_id=f"{case_key}.episode",
        source_sha256=source_hash,
        evidence_spans=(span,),
        objects=tuple(objects),
        relations=tuple(relations),
        quantities=tuple(quantities),
        hyperrelations=(hyperrelation,),
        constraints=tuple(constraints),
        observables=tuple(observables),
        declared_boundary_object_id=boundary_id,
        missing_observables=(
            ()
            if missing_observable_id is None
            else (missing_observable_id,)
        ),
    )
    episode_issues = episode.validate()
    if episode_issues:
        raise AssertionError(
            "fixed episode invalid: " + ",".join(episode_issues)
        )
    binding = LawBinding(
        binding_id=f"{case_key}.binding",
        law_id=schema.law_id,
        registry_hash=registry.registry_hash,
        schema_hash=schema.schema_hash,
        episode_hash=episode.episode_hash,
        role_bindings=tuple(
            RoleBinding(
                role_id=role.role_id,
                target_id=role_target_ids[role.role_id],
                evidence_span_ids=(span_id,),
            )
            for role in schema.roles
        ),
        observable_bindings=tuple(
            ObservableBinding(
                observable_id=observable.observable_id,
                observable_hash=observable.observable_hash,
            )
            for observable in episode.observables
        ),
    )
    binding_issues = validate_law_binding(
        registry, schema, episode, binding
    )
    if binding_issues:
        raise AssertionError(
            "fixed binding invalid: " + ",".join(binding_issues)
        )
    return episode, binding


def _evidence_ids(
    episode: StructuralEpisode,
    binding: LawBinding,
) -> tuple[str, ...]:
    role_evidence = {
        span_id
        for role in binding.role_bindings
        for span_id in role.evidence_span_ids
    }
    observable_evidence = {
        span_id
        for observable_binding in binding.observable_bindings
        for span_id in episode.require_observable(
            observable_binding.observable_id
        ).evidence_span_ids
    }
    return tuple(sorted(role_evidence | observable_evidence))


def _run_law_case(
    registry: GSCLSchemaRegistry,
    case: FixedLawCase,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    schema = case.schema
    suffix = schema.law_id.rsplit(".", 1)[-1]
    issues: list[str] = []
    source_episode, source_binding = _build_episode_and_binding(
        registry, case, case_key=f"source.{suffix}"
    )
    target_episode, target_binding = _build_episode_and_binding(
        registry, case, case_key=f"target.{suffix}"
    )
    source_evaluation = evaluate_bound_law(
        registry,
        schema,
        source_episode,
        source_binding,
        case.policy,
    )
    target_evaluation = evaluate_bound_law(
        registry,
        schema,
        target_episode,
        target_binding,
        case.policy,
    )
    if source_evaluation.disposition is not ResidualDisposition.SATISFIED:
        issues.append(f"primary_not_satisfied.{schema.law_id}")
    if target_evaluation.disposition is not ResidualDisposition.SATISFIED:
        issues.append(f"renamed_primary_not_satisfied.{schema.law_id}")

    source_contrastives = {
        transformation_id: evaluate_transformed_law(
            registry,
            schema,
            source_episode,
            source_binding,
            case.policy,
            transformation_id=transformation_id,
        )
        for transformation_id in sorted(
            schema.hard_negative_transformations
        )
    }
    target_contrastives = {
        transformation_id: evaluate_transformed_law(
            registry,
            schema,
            target_episode,
            target_binding,
            case.policy,
            transformation_id=transformation_id,
        )
        for transformation_id in sorted(
            schema.hard_negative_transformations
        )
    }
    for transformation_id, evaluation in source_contrastives.items():
        if evaluation.disposition is not ResidualDisposition.VIOLATED:
            issues.append(
                f"hard_negative_not_violated.{schema.law_id}.{transformation_id}"
            )
    if set(source_contrastives) != set(
        schema.hard_negative_transformations
    ):
        issues.append(f"hard_negative_coverage_mismatch.{schema.law_id}")

    source_receipt = build_law_residual_receipt(
        registry,
        schema,
        source_episode,
        source_binding,
        case.policy,
        receipt_id=f"source.{suffix}.receipt",
        evidence_span_ids=_evidence_ids(
            source_episode, source_binding
        ),
    )
    target_receipt = build_law_residual_receipt(
        registry,
        schema,
        target_episode,
        target_binding,
        case.policy,
        receipt_id=f"target.{suffix}.receipt",
        evidence_span_ids=_evidence_ids(
            target_episode, target_binding
        ),
    )
    source_trusted_issues = verify_law_residual_receipt_trusted(
        source_receipt,
        registry,
        schema,
        source_episode,
        source_binding,
        case.policy,
    )
    target_trusted_issues = verify_law_residual_receipt_trusted(
        target_receipt,
        registry,
        schema,
        target_episode,
        target_binding,
        case.policy,
    )
    if source_trusted_issues:
        issues.append(f"source_trusted_recomputation_failed.{schema.law_id}")
    if target_trusted_issues:
        issues.append(f"target_trusted_recomputation_failed.{schema.law_id}")

    correspondence = compare_structural_bindings(
        registry,
        schema,
        source_episode,
        source_binding,
        source_receipt,
        target_episode,
        target_binding,
        target_receipt,
        correspondence_id=f"pair.{suffix}.correspondence",
        source_policy=case.policy,
        target_policy=case.policy,
    )
    if correspondence.disposition is not CorrespondenceDisposition.ACCEPTED:
        issues.append(f"renamed_correspondence_not_accepted.{schema.law_id}")
    source_signature = canonical_structural_signature(
        registry, schema, source_episode, source_binding
    )
    target_signature = canonical_structural_signature(
        registry, schema, target_episode, target_binding
    )
    if source_signature != target_signature:
        issues.append(f"entity_renaming_not_invariant.{schema.law_id}")

    abstention_episode, abstention_binding = _build_episode_and_binding(
        registry,
        case,
        case_key=f"abstain.{suffix}",
        payload_overrides=case.abstention_overrides,
        missing_observable_id=case.missing_observable_id,
    )
    abstention_evaluation = evaluate_bound_law(
        registry,
        schema,
        abstention_episode,
        abstention_binding,
        case.policy,
    )
    if (
        abstention_evaluation.disposition
        is not case.expected_abstention
    ):
        issues.append(f"abstention_disposition_mismatch.{schema.law_id}")

    attacked_episode, attacked_binding = _build_episode_and_binding(
        registry,
        case,
        case_key=f"semantic.attack.{suffix}",
        payload_overrides=_semantic_attack_override(schema),
    )
    semantic_attack_rejected = False
    try:
        evaluate_bound_law(
            registry,
            schema,
            attacked_episode,
            attacked_binding,
            case.policy,
        )
    except PermissionError as exc:
        semantic_attack_rejected = "semantic_" in str(exc)
    if not semantic_attack_rejected:
        issues.append(
            f"role_observable_semantic_attack_not_rejected.{schema.law_id}"
        )

    forged_verifier = replace(
        source_receipt, verifier_id="evil.verifier.v1"
    )
    forged_issues = forged_verifier.validate(
        registry,
        schema,
        source_episode,
        source_binding,
        expected_policy_hash=case.policy.policy_hash,
    )
    if "law_residual_verifier_id_mismatch" not in forged_issues:
        issues.append(f"forged_verifier_not_rejected.{schema.law_id}")
    first_component = source_receipt.components[0]
    forged_components = (
        replace(first_component, value=ExactRational(999)),
        *source_receipt.components[1:],
    )
    false_satisfied = replace(
        source_receipt,
        disposition=ResidualDisposition.SATISFIED,
        components=forged_components,
    )
    false_satisfied_issues = false_satisfied.validate(
        registry,
        schema,
        source_episode,
        source_binding,
        expected_policy_hash=case.policy.policy_hash,
    )
    if (
        "law_residual_disposition_component_mismatch"
        not in false_satisfied_issues
    ):
        issues.append(
            f"false_satisfied_disposition_not_rejected.{schema.law_id}"
        )

    safe_payloads = (
        source_episode.safe_payload(),
        source_binding.safe_payload(),
        source_receipt.safe_payload(),
        correspondence.safe_payload(),
    )
    safe_bytes = strict_canonical_bytes(list(safe_payloads))
    raw_identifiers = (
        source_episode.episode_id,
        source_binding.binding_id,
        source_receipt.receipt_id,
        correspondence.correspondence_id,
        source_episode.evidence_spans[0].span_id,
        source_binding.role_bindings[0].target_id,
    )
    if any(
        raw_identifier.encode("utf-8") in safe_bytes
        for raw_identifier in raw_identifiers
    ):
        issues.append(f"safe_payload_identifier_leak.{schema.law_id}")

    row = {
        "law_id": schema.law_id,
        "schema_hash": schema.schema_hash,
        "policy_hash": case.policy.policy_hash,
        "source_receipt_hash": source_receipt.receipt_hash,
        "target_receipt_hash": target_receipt.receipt_hash,
        "correspondence_hash": correspondence.correspondence_hash,
        "primary_disposition": source_evaluation.disposition.value,
        "renamed_primary_disposition": (
            target_evaluation.disposition.value
        ),
        "correspondence_disposition": (
            correspondence.disposition.value
        ),
        "hard_negatives": [
            {
                "transformation_id": transformation_id,
                "operator_contract_hash": (
                    HARD_NEGATIVE_OPERATOR_CONTRACT_HASHES[
                        transformation_id
                    ]
                ),
                "disposition": evaluation.disposition.value,
                "transformed_input_hash": (
                    evaluation.evaluation_input_hash
                ),
            }
            for transformation_id, evaluation in sorted(
                source_contrastives.items()
            )
        ],
        "abstention_disposition": (
            abstention_evaluation.disposition.value
        ),
        "trusted_primary_recomputation": (
            not source_trusted_issues and not target_trusted_issues
        ),
        "receipt_builder_internal_recomputation": True,
        "preregistered_semantic_attack_rejected": (
            semantic_attack_rejected
        ),
        "entity_renaming_signature_equal": (
            source_signature == target_signature
        ),
        "safe_private_separation": not any(
            raw_identifier.encode("utf-8") in safe_bytes
            for raw_identifier in raw_identifiers
        ),
    }
    row["case_commitment"] = strict_content_hash(row)
    return row, tuple(sorted(set(issues)))


def _strict_json_issues() -> tuple[str, ...]:
    issues: list[str] = []

    class CustomValue:
        pass

    invalid_values = (
        0.5,
        float("nan"),
        {"set_value": {1}},
        {"fraction": Fraction(1, 2)},
        {"custom": CustomValue()},
    )
    for index, value in enumerate(invalid_values):
        try:
            strict_canonical_bytes(value)
        except TypeError:
            continue
        issues.append(f"strict_json_accepted_invalid_value.{index}")
    return tuple(issues)


def _run_matrix(
    registry: GSCLSchemaRegistry,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    rows: list[dict[str, Any]] = []
    issues: list[str] = list(_strict_json_issues())
    for case in build_fixed_cases(registry):
        try:
            row, case_issues = _run_law_case(registry, case)
            rows.append(row)
            issues.extend(case_issues)
        except Exception as exc:  # collect-all qualification boundary
            issues.append(
                "law_case_exception."
                f"{case.schema.law_id}.{type(exc).__name__}"
            )
    first_schema = sorted(
        registry.schemas, key=lambda row: row.law_id
    )[0]
    remapped = replace(
        first_schema, residual_function_id="evil.residual.fn"
    )
    remapped_registry = GSCLSchemaRegistry(
        ontology_hash=registry.ontology_hash,
        schemas=tuple(
            remapped if schema.law_id == first_schema.law_id else schema
            for schema in registry.schemas
        ),
    )
    ontology = build_universal_assumption_ontology_v1()
    if (
        "gscl_registry_frozen_contract_mismatch"
        not in remapped_registry.validate(ontology)
    ):
        issues.append("registry_remapping_not_rejected")
    return (
        sorted(rows, key=lambda row: row["law_id"]),
        tuple(sorted(set(issues))),
    )


def run_qualification() -> dict[str, Any]:
    """Run all checks twice and return one safe, non-formal receipt."""

    ontology = build_universal_assumption_ontology_v1()
    ontology_hash_before = ontology.ontology_hash
    registry = build_gscl_schema_registry_v1(ontology)
    first_rows, first_issues = _run_matrix(registry)
    second_rows, second_issues = _run_matrix(registry)
    issues = list(first_issues)
    if strict_canonical_bytes(
        {"rows": first_rows, "issues": list(first_issues)}
    ) != strict_canonical_bytes(
        {"rows": second_rows, "issues": list(second_issues)}
    ):
        issues.append("same_process_byte_exact_replay_failed")
    if ontology.ontology_hash != ontology_hash_before:
        issues.append("uao_ontology_mutated")
    if len(first_rows) != 5:
        issues.append("five_law_case_coverage_failed")
    all_primary_satisfied = len(first_rows) == 5 and all(
        row["primary_disposition"]
        == ResidualDisposition.SATISFIED.value
        for row in first_rows
    )
    all_hard_negatives_rejected = len(first_rows) == 5 and all(
        hard_negative["disposition"]
        == ResidualDisposition.VIOLATED.value
        for row in first_rows
        for hard_negative in row["hard_negatives"]
    )
    all_correspondences_accepted = len(first_rows) == 5 and all(
        row["correspondence_disposition"]
        == CorrespondenceDisposition.ACCEPTED.value
        for row in first_rows
    )
    all_semantic_attacks_rejected = len(first_rows) == 5 and all(
        row["preregistered_semantic_attack_rejected"]
        for row in first_rows
    )
    if not all_primary_satisfied:
        issues.append("all_primary_satisfied_failed")
    if not all_hard_negatives_rejected:
        issues.append("all_hard_negatives_rejected_failed")
    if not all_correspondences_accepted:
        issues.append("all_correspondences_accepted_failed")
    if not all_semantic_attacks_rejected:
        issues.append("all_semantic_attacks_rejected_failed")
    issue_ids = tuple(sorted(set(issues)))
    declared_capability_surface = dict(
        QUALIFICATION_CONTRACT["declared_capability_surface"]
    )
    receipt: dict[str, Any] = {
        "qualification_version": QUALIFICATION_VERSION,
        "qualification_contract_hash": QUALIFICATION_CONTRACT_HASH,
        "qualification_scope": "phase0_kernel_only",
        "status": (
            "PASS_PHASE0_KERNEL_ONLY"
            if not issue_ids
            else "FAIL_PHASE0_KERNEL_QUALIFICATION"
        ),
        "formal_result": False,
        "efficacy_evidence": False,
        "full_qualification_ready": False,
        "ontology_hash": ontology.ontology_hash,
        "registry_hash": registry.registry_hash,
        "residual_kernel_contract_hash": (
            RESIDUAL_KERNEL_CONTRACT_HASH
        ),
        "law_case_count": len(first_rows),
        "law_cases": first_rows,
        "all_primary_satisfied": all_primary_satisfied,
        "all_hard_negatives_rejected": (
            all_hard_negatives_rejected
        ),
        "all_entity_renamed_correspondences_accepted": (
            all_correspondences_accepted
        ),
        "all_preregistered_semantic_attacks_rejected": (
            all_semantic_attacks_rejected
        ),
        "same_process_byte_exact_replay": (
            "same_process_byte_exact_replay_failed" not in issue_ids
        ),
        "declared_capability_surface": declared_capability_surface,
        "runtime_access_audited": False,
        "issue_ids": list(issue_ids),
        "issue_commitment": strict_content_hash(list(issue_ids)),
    }
    receipt["self_hash"] = strict_content_hash(receipt)
    return receipt


def _synthetic_narrative_completion(
    *,
    left: str,
    verb: str,
    right: str,
    polarity: str,
) -> str:
    """Return one canonical grounded proposal for source-free qualification."""

    payload = {
        "generators": [
            {
                "anchor_mention_id": "a0",
                "causal_orientation": "none",
                "generator_id": "g0",
                "generator_kind": "relation",
                "polarity": polarity,
                "slot_mention_ids": ["m0", "m1"],
                "temporal_orientation": "forward",
            }
        ],
        "mentions": [
            {
                "kind": "object",
                "mention_id": "m0",
                "occurrence": 0,
                "quote": left,
            },
            {
                "kind": "object",
                "mention_id": "m1",
                "occurrence": 0,
                "quote": right,
            },
            {
                "kind": "generator",
                "mention_id": "a0",
                "occurrence": 0,
                "quote": verb,
            },
        ],
        "schema_version": "gscl.narrative.extraction.v1",
    }
    return strict_canonical_bytes(payload).decode("ascii")


def _narrative_synthetic_fixture() -> dict[str, Any]:
    """Exercise the story-only extractor contract with a test runtime.

    No benchmark source, model asset, model runtime, network endpoint, or
    label is available to this helper.
    """

    from replication_runtime.gscl_narrative_extractor_v1 import (  # noqa: PLC0415
        contract as extractor_contract,
    )
    from replication_runtime.gscl_narrative_extractor_v1 import (  # noqa: PLC0415
        worker as extractor_worker,
    )
    from assumption_agent.gscl_narrative_correspondence_v1 import (  # noqa: PLC0415
        NarrativeSource,
        parse_untrusted_generator_completion,
    )

    story_rows = (
        ("Aster", "guides", "Birch", "positive"),
        ("Cedar", "guides", "Dune", "positive"),
        ("Ember", "opposes", "Fjord", "negative"),
    )
    stories = tuple(
        f"{left} {verb} {right}."
        for left, verb, right, _ in story_rows
    )
    completions = tuple(
        _synthetic_narrative_completion(
            left=left,
            verb=verb,
            right=right,
            polarity=polarity,
        )
        for left, verb, right, polarity in story_rows
    )
    input_raw = extractor_contract.encode_input(
        batch_id="synthetic-narrative-qualification",
        sequence=0,
        requests=tuple(
            extractor_contract.StoryRequest(
                ordinal=index, story_text=story
            )
            for index, story in enumerate(stories)
        ),
    )
    pack = (
        extractor_contract.admit_story_only_pack_qualification_only(
            input_raw
        )
    )
    closure = extractor_contract.ExecutionClosure(
        prompt_sha256=extractor_worker.PROMPT_SHA256,
        parser_closure_sha256=hashlib.sha256(
            b"source-free-independent-parser"
        ).hexdigest(),
        model_asset_manifest_sha256=hashlib.sha256(
            b"no-model-asset-test-stub"
        ).hexdigest(),
        model_runtime_closure_sha256=hashlib.sha256(
            b"no-model-runtime-test-stub"
        ).hexdigest(),
        target_double_run_receipt_sha256=hashlib.sha256(
            b"test-stub-double-run"
        ).hexdigest(),
    )

    class StubStoryRuntime:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(
            self, story_text: str
        ) -> extractor_worker.GeneratedCompletion:
            self.calls.append(story_text)
            index = stories.index(story_text)
            return extractor_worker.GeneratedCompletion(
                completion=completions[index],
                token_count=32,
                terminated_by_eos=True,
            )

    def independent_parser(
        story_text: str, completion: str
    ) -> object:
        source_id = (
            "synthetic."
            + hashlib.sha256(story_text.encode("utf-8")).hexdigest()[:24]
        )
        return parse_untrusted_generator_completion(
            NarrativeSource(source_id, story_text),
            completion,
        )

    runtime = StubStoryRuntime()
    results, observed_closure = (
        extractor_worker.process_trusted_pack_test_only(
            pack,
            runtime=runtime,
            narrative_parser=independent_parser,
            execution_closure=closure,
        )
    )
    output_raw = extractor_contract.encode_private_output(
        pack=pack,
        execution_closure=observed_closure,
        results=results,
    )
    decoded = extractor_contract.decode_private_output(
        output_raw,
        expected_pack=pack,
        expected_execution_closure=closure,
    )
    if (
        tuple(runtime.calls) != stories
        or any(
            row["generation_valid"] is not True
            for row in decoded["results"]
        )
    ):
        raise RuntimeError("test_stub_story_contract_incomplete")
    extractions = tuple(
        independent_parser(story, completion)
        for story, completion in zip(stories, completions)
    )
    opaque_item_id = hashlib.sha256(
        b"synthetic narrative qualification item"
    ).hexdigest()
    predictor_raw = (
        strict_canonical_bytes(
            {
                "rows": [
                    {
                        "opaque_item_id": opaque_item_id,
                        "query_narrative": stories[0],
                        "first_choice": stories[1],
                        "second_choice": stories[2],
                    }
                ]
            }
        )
        + b"\n"
    )
    return {
        "closure": closure,
        "completions": completions,
        "extractions": extractions,
        "input_raw": input_raw,
        "opaque_item_id": opaque_item_id,
        "output_raw": output_raw,
        "pack": pack,
        "predictor_raw": predictor_raw,
        "stories": stories,
    }


def _deterministic_narrative_test_encoder() -> object:
    """Construct a local deterministic encoder without model assets."""

    import numpy as np  # noqa: PLC0415

    from replication_runtime.qasper_minilm_v1.binding import (  # noqa: PLC0415
        EMBEDDING_DIMENSION,
    )

    class Tokenizer:
        def __call__(
            self, text: str, **_: object
        ) -> dict[str, list[int]]:
            return {"input_ids": list(range(len(text) + 2))}

    class Model:
        tokenizer = Tokenizer()

    class Encoder:
        _model = Model()
        runtime_receipt = {
            "runtime": "deterministic_source_free_test_encoder"
        }
        canary_receipt = {
            "canary": "deterministic_source_free_test_encoder"
        }

        def encode(self, texts: Any) -> Any:
            rows = []
            for text in texts:
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                vector = np.asarray(
                    [
                        (
                            (
                                digest[index % len(digest)]
                                + index
                            )
                            % 251
                        )
                        + 1
                        for index in range(EMBEDDING_DIMENSION)
                    ],
                    dtype=np.float32,
                )
                rows.append(vector / np.linalg.norm(vector))
            return np.vstack(rows).astype(np.float32)

    return Encoder()


def _qualify_narrative_extractor_contract() -> dict[str, Any]:
    fixture = _narrative_synthetic_fixture()
    from replication_runtime.gscl_narrative_extractor_v1 import (  # noqa: PLC0415
        contract as extractor_contract,
    )

    decoded = extractor_contract.decode_private_output(
        fixture["output_raw"],
        expected_pack=fixture["pack"],
        expected_execution_closure=fixture["closure"],
    )
    valid_count = sum(
        row["generation_valid"] is True
        for row in decoded["results"]
    )
    if valid_count != 3:
        raise RuntimeError("extractor_valid_story_count_mismatch")
    return {
        "status": "PASS",
        "story_count": valid_count,
        "input_commitment": fixture["pack"].input_pack_commitment,
        "output_sha256": hashlib.sha256(
            fixture["output_raw"]
        ).hexdigest(),
        "test_stub_used": True,
        "local_qwen_model_executed": False,
        "model_asset_accessed": False,
        "formal_source_accessed": False,
    }


def _qualify_narrative_scorers_and_arms() -> dict[str, Any]:
    from dataclasses import replace as dataclass_replace  # noqa: PLC0415

    from assumption_agent.gscl_arn_intrinsic_arms_v1 import (  # noqa: PLC0415
        IntrinsicArm,
        IntrinsicContractError,
        evaluate_intrinsic_item,
        evaluate_frozen_intrinsic_item,
    )
    from assumption_agent.gscl_arn_intrinsic_scorers_v1 import (  # noqa: PLC0415
        FrozenNarrativeScorers,
        LEGACY_FEATURE_IDS,
        SCORER_CONTRACT_HASH,
    )
    from assumption_agent.gscl_narrative_correspondence_v1 import (  # noqa: PLC0415
        MappingSearchConfig,
    )

    fixture = _narrative_synthetic_fixture()
    query, first, second = fixture["extractions"]
    scorers = FrozenNarrativeScorers.build(
        (query, first, second),
        encoder=_deterministic_narrative_test_encoder(),
    )
    semantic_first = scorers.raw_text_scorer(
        query.source.utf8_bytes, first.source.utf8_bytes
    )
    semantic_second = scorers.raw_text_scorer(
        query.source.utf8_bytes, first.source.utf8_bytes
    )
    legacy_first = scorers.legacy_vectorizer(
        query, LEGACY_FEATURE_IDS
    )
    legacy_second = scorers.legacy_vectorizer(
        query, LEGACY_FEATURE_IDS
    )
    structural_first = scorers.structural_scorer(query, first)
    structural_second = scorers.structural_scorer(query, first)
    if (
        semantic_first != semantic_second
        or legacy_first != legacy_second
        or structural_first.safe_payload()
        != structural_second.safe_payload()
        or scorers.receipt["actual_batch_replay_exact"] is not True
        or scorers.receipt["benchmark_source_accessed"] is not False
        or scorers.receipt["labels_accessed"] is not False
    ):
        raise RuntimeError("concrete_scorer_replay_or_closure_failed")
    authoritative_nonformal_rejected = False
    try:
        evaluate_frozen_intrinsic_item(
            opaque_item_id=fixture["opaque_item_id"],
            query=query,
            candidates=(first, second),
            scorers=scorers,
        )
    except IntrinsicContractError as exc:
        authoritative_nonformal_rejected = (
            exc.issue_id == "frozen_scorer_not_formal"
        )
    if not authoritative_nonformal_rejected:
        raise RuntimeError(
            "authoritative_factory_accepted_qualification_scorer"
        )

    result = evaluate_intrinsic_item(
        opaque_item_id=fixture["opaque_item_id"],
        query=query,
        candidates=(first, second),
        raw_text_scorer=scorers.raw_text_scorer,
        legacy_vectorizer=scorers.legacy_vectorizer,
        legacy_feature_ids=LEGACY_FEATURE_IDS,
        structural_scorer=scorers.structural_scorer,
        mapping_config=MappingSearchConfig(),
        raw_text_scorer_commitment=hashlib.sha256(
            b"qualification.semantic-only"
        ).hexdigest(),
        legacy_vectorizer_commitment=hashlib.sha256(
            b"qualification.legacy-keyword"
        ).hexdigest(),
        structural_scorer_commitment=hashlib.sha256(
            b"qualification.structural-proposal"
        ).hexdigest(),
    )
    result.__post_init__()
    if tuple(prediction.arm for prediction in result.predictions) != tuple(
        IntrinsicArm
    ):
        raise RuntimeError("four_arm_set_incomplete")
    if any(
        receipt.flat_proposal_set_hash
        != receipt.full_proposal_set_hash
        for receipt in result.candidate_receipts
    ):
        raise RuntimeError("flat_full_proposal_set_not_shared")

    cross_binding_rejected = False
    try:
        dataclass_replace(
            result,
            candidate_extraction_hashes=tuple(
                reversed(result.candidate_extraction_hashes)
            ),
        )
    except IntrinsicContractError:
        cross_binding_rejected = True
    if not cross_binding_rejected:
        raise RuntimeError("authoritative_cross_binding_forgery_accepted")

    return {
        "status": "PASS",
        "scorer_contract_hash": SCORER_CONTRACT_HASH,
        "scorer_receipt_self_hash": scorers.receipt["self_hash"],
        "actual_encoder_batch_replay_exact": True,
        "controls_present": [
            "semantic_only",
            "legacy_keyword",
            "flat_label_no_verifier",
            "full_structural_verifier",
        ],
        "arm_count": len(result.predictions),
        "flat_full_proposal_set_shared": True,
        "qualification_cross_binding_forgery_rejected": True,
        "authoritative_factory_rejected_nonformal_scorer": True,
        "formal_source_accessed": False,
        "labels_accessed": False,
    }


def _qualify_narrative_internal_factory_adapter() -> dict[str, Any]:
    from assumption_agent.benchmarks import (  # noqa: PLC0415
        gscl_arn_formal_item_factory_v1 as formal_item_factory,
    )

    fixture = _narrative_synthetic_fixture()
    output = (
        formal_item_factory.build_private_four_arm_output_qualification_only(
            predictor_raw=fixture["predictor_raw"],
            input_batch_raws=(fixture["input_raw"],),
            output_batch_raws=(fixture["output_raw"],),
            encoder=_deterministic_narrative_test_encoder(),
        )
    )
    expected_arms = {
        "semantic_only",
        "legacy_keyword",
        "flat_label_no_verifier",
        "full_gscl",
    }
    if (
        output["lineage"] != "synthetic_source_free_qualification"
        or set(output["by_arm"]) != expected_arms
        or any(len(rows) != 1 for rows in output["by_arm"].values())
        or output["caller_predictions_accepted"] is not False
        or output["caller_commitments_accepted"] is not False
        or output["item_content_emitted"] is not False
    ):
        raise RuntimeError("internal_factory_adapter_contract_failed")
    return {
        "status": "PASS",
        "adapter": (
            "build_private_four_arm_output_qualification_only"
        ),
        "adapter_output_self_hash": output["self_hash"],
        "lineage": output["lineage"],
        "arm_count": len(output["by_arm"]),
        "item_count": output["item_count"],
        "caller_predictions_accepted": False,
        "caller_commitments_accepted": False,
        "formal_output_authority": False,
        "formal_supervisor_mutated": False,
    }


def _qualify_frozen_scorer_authority_negatives() -> dict[str, Any]:
    """Reject subclass and mutable-receipt scorer authority forgeries."""

    from dataclasses import replace as dataclass_replace  # noqa: PLC0415

    from assumption_agent.gscl_arn_intrinsic_arms_v1 import (  # noqa: PLC0415
        evaluate_frozen_intrinsic_item,
    )
    from assumption_agent.gscl_arn_intrinsic_scorers_v1 import (  # noqa: PLC0415
        FrozenNarrativeScorers,
    )
    fixture = _narrative_synthetic_fixture()
    query, first, second = fixture["extractions"]
    scorers = FrozenNarrativeScorers.build(
        (query, first, second),
        encoder=_deterministic_narrative_test_encoder(),
    )

    subclass_rejected = False
    try:
        class ForgedScorers(FrozenNarrativeScorers):
            def raw_text_scorer(
                self, _left: bytes, _right: bytes
            ) -> int:
                return 7

        forged_subclass = ForgedScorers(
            source_vectors=scorers.source_vectors,
            mention_vectors=scorers.mention_vectors,
            primed_extraction_hashes=scorers.primed_extraction_hashes,
            receipt=scorers.receipt,
        )
        evaluate_frozen_intrinsic_item(
            opaque_item_id=fixture["opaque_item_id"],
            query=query,
            candidates=(first, second),
            scorers=forged_subclass,
        )
    except Exception:
        subclass_rejected = True

    forged_receipt_rejected = False
    forged_receipt = dict(scorers.receipt)
    forged_receipt["self_hash"] = "0" * 64
    try:
        forged_scorers = dataclass_replace(
            scorers, receipt=forged_receipt
        )
        evaluate_frozen_intrinsic_item(
            opaque_item_id=fixture["opaque_item_id"],
            query=query,
            candidates=(first, second),
            scorers=forged_scorers,
        )
    except Exception:
        forged_receipt_rejected = True

    if not subclass_rejected:
        raise RuntimeError("frozen_scorer_subclass_forgery_accepted")
    if not forged_receipt_rejected:
        raise RuntimeError("frozen_scorer_receipt_forgery_accepted")
    return {
        "status": "PASS",
        "frozen_scorer_subclass_rejected": True,
        "frozen_scorer_receipt_forgery_rejected": True,
        "formal_result": False,
    }


def _qualify_qwen_formal_authority_negatives() -> dict[str, Any]:
    """Reject qualification custody and an uninitialized exact-type runtime."""

    from replication_runtime.gscl_narrative_extractor_v1 import (  # noqa: PLC0415
        contract as extractor_contract,
    )
    from replication_runtime.gscl_narrative_extractor_v1 import (  # noqa: PLC0415
        worker as extractor_worker,
    )

    fixture = _narrative_synthetic_fixture()
    qualification_pack_rejected = False
    try:
        extractor_contract.require_formal_story_only_pack(
            fixture["pack"]
        )
    except Exception:
        qualification_pack_rejected = True

    uninitialized_runtime_rejected = False
    try:
        fake_runtime = object.__new__(
            extractor_worker.LocalQwenRuntime
        )
        fake_runtime._validate_formal_binding()  # noqa: SLF001
    except Exception:
        uninitialized_runtime_rejected = True
    if not qualification_pack_rejected:
        raise RuntimeError(
            "qualification_pack_accepted_as_formal_custody"
        )
    if not uninitialized_runtime_rejected:
        raise RuntimeError(
            "uninitialized_exact_type_qwen_runtime_accepted_formally"
        )
    return {
        "status": "PASS",
        "qualification_pack_formal_custody_rejected": True,
        "uninitialized_exact_type_qwen_runtime_rejected": True,
        "formal_result": False,
    }


def _run_narrative_source_free_matrix() -> tuple[
    dict[str, dict[str, Any]], tuple[str, ...]
]:
    """Run every source-free narrative check even after an earlier failure."""

    checks = (
        (
            "raw_story_to_qwen_contract_test_stub",
            _qualify_narrative_extractor_contract,
        ),
        (
            "concrete_frozen_scorers_and_four_arms",
            _qualify_narrative_scorers_and_arms,
        ),
        (
            "supervisor_internal_factory_source_free_adapter",
            _qualify_narrative_internal_factory_adapter,
        ),
        (
            "frozen_scorer_authority_negative_checks",
            _qualify_frozen_scorer_authority_negatives,
        ),
        (
            "qwen_formal_authority_negative_checks",
            _qualify_qwen_formal_authority_negatives,
        ),
    )
    rows: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for check_id, check in checks:
        try:
            rows[check_id] = check()
        except Exception as exc:  # collect-all qualification boundary
            issue = f"{check_id}.{type(exc).__name__}.{exc}"
            issues.append(issue)
            rows[check_id] = {
                "status": "FAIL",
                "issue_id": issue,
            }
    return dict(sorted(rows.items())), tuple(sorted(set(issues)))


def run_narrative_source_free_qualification() -> dict[str, Any]:
    """Run the narrative extension twice without source/model/network access."""

    first_checks, first_issues = _run_narrative_source_free_matrix()
    second_checks, second_issues = _run_narrative_source_free_matrix()
    issues = list(first_issues)
    same_process_byte_exact_replay = (
        strict_canonical_bytes(
            {
                "checks": first_checks,
                "issues": list(first_issues),
            }
        )
        == strict_canonical_bytes(
            {
                "checks": second_checks,
                "issues": list(second_issues),
            }
        )
    )
    if not same_process_byte_exact_replay:
        issues.append("same_process_byte_exact_replay_failed")
    issue_ids = tuple(sorted(set(issues)))
    passed = (
        not issue_ids
        and len(first_checks) == 5
        and all(
            row["status"] == "PASS"
            for row in first_checks.values()
        )
    )
    receipt: dict[str, Any] = {
        "qualification_version": (
            NARRATIVE_QUALIFICATION_VERSION
        ),
        "qualification_contract_hash": (
            NARRATIVE_QUALIFICATION_CONTRACT_HASH
        ),
        "same_iterative_harness_lineage": (
            EXTENDED_QUALIFICATION_VERSION
        ),
        "qualification_scope": (
            "source_free_raw_story_to_four_arm_internal_factory_path"
        ),
        "status": (
            "PASS_GSCL_NARRATIVE_SOURCE_FREE_QUALIFICATION"
            if passed
            else "FAIL_GSCL_NARRATIVE_SOURCE_FREE_QUALIFICATION"
        ),
        "formal_result": False,
        "efficacy_evidence": False,
        "new_formal_study": False,
        "effect_gate_added": False,
        "public_intrinsic_measurement": False,
        "public_intrinsic_freeze_ready": False,
        "collect_all": True,
        "same_process_byte_exact_replay": (
            same_process_byte_exact_replay
        ),
        "declared_capability_surface": dict(
            NARRATIVE_QUALIFICATION_CONTRACT[
                "declared_capability_surface"
            ]
        ),
        "checks": first_checks,
        "issue_ids": list(issue_ids),
        "issue_commitment": strict_content_hash(list(issue_ids)),
    }
    receipt["self_hash"] = strict_content_hash(receipt)
    return receipt


def run_extended_qualification() -> dict[str, Any]:
    """Combine the frozen kernel checks and controlled evidence-path checks.

    This is an extension of the same non-scoring harness, not a formal study
    and not a downstream effect measurement.
    """

    from assumption_agent.benchmarks.gscl_controlled_evidence_qualification_v1 import (  # noqa: PLC0415
        run_controlled_evidence_qualification,
    )

    kernel = run_qualification()
    evidence_path = run_controlled_evidence_qualification()
    narrative_path = run_narrative_source_free_qualification()
    issue_ids = tuple(
        sorted(
            {
                *(
                    f"kernel.{issue}"
                    for issue in kernel["issue_ids"]
                ),
                *(
                    f"evidence_path.{issue}"
                    for issue in evidence_path["issue_ids"]
                ),
                *(
                    f"narrative_path.{issue}"
                    for issue in narrative_path["issue_ids"]
                ),
            }
        )
    )
    passed = (
        kernel["status"] == "PASS_PHASE0_KERNEL_ONLY"
        and evidence_path["status"]
        == "PASS_CONTROLLED_EVIDENCE_PATH"
        and narrative_path["status"]
        == "PASS_GSCL_NARRATIVE_SOURCE_FREE_QUALIFICATION"
        and not issue_ids
    )
    receipt: dict[str, Any] = {
        "qualification_version": EXTENDED_QUALIFICATION_VERSION,
        "qualification_contract_hash": (
            EXTENDED_QUALIFICATION_CONTRACT_HASH
        ),
        "status": (
            "PASS_GSCL_UNIFIED_NONSCORING_HARNESS"
            if passed
            else "FAIL_GSCL_UNIFIED_NONSCORING_HARNESS"
        ),
        "formal_result": False,
        "efficacy_evidence": False,
        "new_formal_study": False,
        "effect_gate_added": False,
        "public_intrinsic_measurement": False,
        "public_intrinsic_freeze_ready": False,
        "kernel_receipt": kernel,
        "controlled_evidence_receipt": evidence_path,
        "narrative_source_free_receipt": narrative_path,
        "issue_ids": list(issue_ids),
        "issue_commitment": strict_content_hash(list(issue_ids)),
    }
    receipt["self_hash"] = strict_content_hash(receipt)
    return receipt


__all__ = [
    "FixedLawCase",
    "EXTENDED_QUALIFICATION_CONTRACT",
    "EXTENDED_QUALIFICATION_CONTRACT_HASH",
    "EXTENDED_QUALIFICATION_VERSION",
    "NARRATIVE_QUALIFICATION_CONTRACT",
    "NARRATIVE_QUALIFICATION_CONTRACT_HASH",
    "NARRATIVE_QUALIFICATION_VERSION",
    "QUALIFICATION_CONTRACT",
    "QUALIFICATION_CONTRACT_HASH",
    "QUALIFICATION_VERSION",
    "build_fixed_cases",
    "run_extended_qualification",
    "run_narrative_source_free_qualification",
    "run_qualification",
]
