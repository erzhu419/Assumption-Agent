from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json

import pytest

from assumption_agent.generalized_structural_correspondence_v1 import (
    ConstraintParticipant,
    EvidenceSpanRef,
    ExactRational,
    InferenceProvenance,
    LawBinding,
    LawKind,
    ObservableBinding,
    ObservableValueType,
    ObservationStatus,
    RoleBinding,
    RoleTargetKind,
    StructuralConstraint,
    StructuralEpisode,
    StructuralObject,
    StructuralQuantity,
    StructuralRelation,
    TypedObservable,
    build_gscl_schema_registry_v1,
    canonical_structural_signature,
    strict_canonical_bytes,
    strict_content_hash,
    validate_law_binding,
)
from assumption_agent.universal_assumption_ontology_v1 import (
    T05,
    T09,
    T14,
    T15,
    T17,
    build_universal_assumption_ontology_v1,
)


SOURCE_BYTES = b"alpha beta gamma delta"
SOURCE_HASH = hashlib.sha256(SOURCE_BYTES).hexdigest()


def _span(
    span_id: str,
    start_byte: int,
    end_byte: int,
) -> EvidenceSpanRef:
    return EvidenceSpanRef(
        span_id=span_id,
        source_sha256=SOURCE_HASH,
        start_byte=start_byte,
        end_byte=end_byte,
        span_sha256=hashlib.sha256(
            SOURCE_BYTES[start_byte:end_byte]
        ).hexdigest(),
    )


def _rational_payload(value: int) -> dict[str, int]:
    return {"numerator": value, "denominator": 1}


def _monotone_fixture(
    *,
    reverse_evidence_order: bool = False,
    swap_quantity_owners: bool = False,
) -> tuple[
    object,
    object,
    StructuralEpisode,
    LawBinding,
]:
    ontology = build_universal_assumption_ontology_v1()
    registry = build_gscl_schema_registry_v1(ontology)
    schema = registry.require_law("gscl.v1.t17_monotone_order")
    spans = (
        _span("span.alpha", 0, 5),
        _span("span.beta", 6, 10),
    )
    evidence_ids = tuple(span.span_id for span in spans)
    if reverse_evidence_order:
        spans = tuple(reversed(spans))
        evidence_ids = tuple(reversed(evidence_ids))

    lower_object_id = "sample.lower.state"
    upper_object_id = "sample.upper.state"
    lower_quantity_id = "sample.lower.quantity"
    upper_quantity_id = "sample.upper.quantity"
    objects = (
        StructuralObject(
            object_id=lower_object_id,
            object_type="State",
            evidence_span_ids=evidence_ids,
        ),
        StructuralObject(
            object_id=upper_object_id,
            object_type="State",
            evidence_span_ids=evidence_ids,
        ),
    )
    relation = StructuralRelation(
        relation_id="sample.partial.order",
        relation_type="PartialOrder",
        source_object_id=lower_object_id,
        target_object_id=upper_object_id,
        evidence_span_ids=evidence_ids,
        order_index=0,
    )
    quantities = (
        StructuralQuantity(
            quantity_id=lower_quantity_id,
            owner_object_id=(
                upper_object_id
                if swap_quantity_owners
                else lower_object_id
            ),
            dimension="Score",
            unit="point",
            value=ExactRational(1),
            evidence_span_ids=evidence_ids,
        ),
        StructuralQuantity(
            quantity_id=upper_quantity_id,
            owner_object_id=(
                lower_object_id
                if swap_quantity_owners
                else upper_object_id
            ),
            dimension="Score",
            unit="point",
            value=ExactRational(2),
            evidence_span_ids=evidence_ids,
        ),
    )
    observable_pairs = TypedObservable(
        observable_id="comparable_output_pairs",
        value_type=ObservableValueType.COMPARABLE_PAIRS,
        value_payload={
            "pairs": [
                {
                    "lower": _rational_payload(1),
                    "upper": _rational_payload(2),
                }
            ]
        },
        evidence_span_ids=evidence_ids,
        dimension="Score",
        unit="point",
    )
    observable_direction = TypedObservable(
        observable_id="declared_direction",
        value_type=ObservableValueType.DIRECTION,
        value_payload={"direction": 1},
        evidence_span_ids=evidence_ids,
    )
    constraint = StructuralConstraint(
        constraint_id="sample.monotone.constraint",
        constraint_type="Monotonicity",
        participants=(
            ConstraintParticipant(
                participant_role="lower_state",
                target_kind=RoleTargetKind.OBJECT,
                target_id=lower_object_id,
            ),
            ConstraintParticipant(
                participant_role="upper_state",
                target_kind=RoleTargetKind.OBJECT,
                target_id=upper_object_id,
            ),
            ConstraintParticipant(
                participant_role="order_relation",
                target_kind=RoleTargetKind.RELATION,
                target_id=relation.relation_id,
            ),
            ConstraintParticipant(
                participant_role="lower_value",
                target_kind=RoleTargetKind.QUANTITY,
                target_id=lower_quantity_id,
            ),
            ConstraintParticipant(
                participant_role="upper_value",
                target_kind=RoleTargetKind.QUANTITY,
                target_id=upper_quantity_id,
            ),
        ),
        observable_ids=(
            "comparable_output_pairs",
            "declared_direction",
        ),
        evidence_span_ids=evidence_ids,
    )
    episode = StructuralEpisode(
        episode_id="sample.monotone.episode",
        source_sha256=SOURCE_HASH,
        evidence_spans=spans,
        objects=(
            tuple(reversed(objects))
            if reverse_evidence_order
            else objects
        ),
        relations=(relation,),
        quantities=(
            tuple(reversed(quantities))
            if reverse_evidence_order
            else quantities
        ),
        hyperrelations=(),
        constraints=(constraint,),
        observables=(
            (observable_direction, observable_pairs)
            if reverse_evidence_order
            else (observable_pairs, observable_direction)
        ),
    )
    role_targets = {
        "lower_state": lower_object_id,
        "upper_state": upper_object_id,
        "order_relation": relation.relation_id,
        "lower_value": lower_quantity_id,
        "upper_value": upper_quantity_id,
        "monotone_constraint": constraint.constraint_id,
    }
    role_bindings = tuple(
        RoleBinding(
            role_id=role_id,
            target_id=target_id,
            evidence_span_ids=evidence_ids,
        )
        for role_id, target_id in role_targets.items()
    )
    observable_bindings = tuple(
        ObservableBinding(
            observable_id=observable.observable_id,
            observable_hash=observable.observable_hash,
        )
        for observable in episode.observables
    )
    binding = LawBinding(
        binding_id="sample.monotone.binding",
        law_id=schema.law_id,
        registry_hash=registry.registry_hash,
        schema_hash=schema.schema_hash,
        episode_hash=episode.episode_hash,
        role_bindings=(
            tuple(reversed(role_bindings))
            if reverse_evidence_order
            else role_bindings
        ),
        observable_bindings=observable_bindings,
    )
    assert episode.validate() == ()
    assert validate_law_binding(
        registry, schema, episode, binding
    ) == ()
    return registry, schema, episode, binding


@dataclass
class _CustomValue:
    value: int


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        float("nan"),
        {"nested": {1, 2}},
        Fraction(1, 3),
        _CustomValue(1),
        ("tuple",),
    ],
)
def test_strict_json_rejects_implicit_or_non_json_values(
    value: object,
) -> None:
    with pytest.raises(TypeError):
        strict_canonical_bytes(value)

    assert strict_canonical_bytes(
        {"exact": {"numerator": 1, "denominator": 3}}
    ) == b'{"exact":{"denominator":3,"numerator":1}}'


def test_registry_has_exact_frozen_five_law_mapping() -> None:
    ontology = build_universal_assumption_ontology_v1()
    first = build_gscl_schema_registry_v1(ontology)
    second = build_gscl_schema_registry_v1(ontology)

    assert first.validate(ontology) == ()
    assert first.registry_hash == second.registry_hash
    assert {
        schema.law_id: (schema.ontology_template_id, schema.law_kind)
        for schema in first.schemas
    } == {
        "gscl.v1.t05_pair_interaction": (
            T05,
            LawKind.LOW_ORDER_INTERACTION,
        ),
        "gscl.v1.t09_path_composition": (
            T09,
            LawKind.PATH_COMPOSITION,
        ),
        "gscl.v1.t14_finite_equivariance": (
            T14,
            LawKind.EQUIVARIANCE,
        ),
        "gscl.v1.t15_closed_balance": (
            T15,
            LawKind.CLOSED_BALANCE,
        ),
        "gscl.v1.t17_monotone_order": (
            T17,
            LawKind.MONOTONE_ORDER,
        ),
    }
    assert {
        schema.verifier_contract_hash for schema in first.schemas
    } == {
        first.schemas[0].verifier_contract_hash,
    }


def test_evidence_span_validation_and_byte_verification_are_exact() -> None:
    valid = _span("span.alpha", 0, 5)
    assert valid.validate() == ()
    assert valid.verify_against(SOURCE_BYTES) == ()

    empty = EvidenceSpanRef(
        span_id="span.empty",
        source_sha256=SOURCE_HASH,
        start_byte=5,
        end_byte=5,
        span_sha256=hashlib.sha256(b"").hexdigest(),
    )
    assert "evidence_span_end_invalid" in empty.validate()

    out_of_bounds = EvidenceSpanRef(
        span_id="span.outside",
        source_sha256=SOURCE_HASH,
        start_byte=0,
        end_byte=len(SOURCE_BYTES) + 1,
        span_sha256="0" * 64,
    )
    assert (
        "evidence_span_out_of_bounds"
        in out_of_bounds.verify_against(SOURCE_BYTES)
    )

    wrong_digest = EvidenceSpanRef(
        span_id="span.wrong",
        source_sha256=SOURCE_HASH,
        start_byte=0,
        end_byte=5,
        span_sha256="0" * 64,
    )
    assert (
        "evidence_span_digest_mismatch"
        in wrong_digest.verify_against(SOURCE_BYTES)
    )


def test_evidence_and_collection_order_do_not_change_hashes() -> None:
    _, _, episode, binding = _monotone_fixture()
    _, _, reordered_episode, reordered_binding = _monotone_fixture(
        reverse_evidence_order=True
    )

    assert episode.episode_hash == reordered_episode.episode_hash
    assert binding.binding_hash == reordered_binding.binding_hash
    assert (
        binding.evaluation_input_hash
        == reordered_binding.evaluation_input_hash
    )


def test_inferred_fields_require_bound_provenance() -> None:
    spans = (
        _span("span.alpha", 0, 5),
        _span("span.beta", 6, 10),
    )
    span_ids = tuple(span.span_id for span in spans)
    expected_input_hash = strict_content_hash(
        [
            span.private_payload()
            for span in sorted(spans, key=lambda row: row.span_id)
        ]
    )
    provenance = InferenceProvenance(
        extractor_id="extractor.structural",
        extractor_version="version.one",
        extractor_implementation_hash="a" * 64,
        input_evidence_hash=expected_input_hash,
        calibration_bucket="bucket.high",
        alternative_binding_hashes=("b" * 64,),
    )
    inferred = StructuralObject(
        object_id="sample.inferred.object",
        object_type="State",
        evidence_span_ids=tuple(reversed(span_ids)),
        observation_status=ObservationStatus.INFERRED,
        inference_provenance=provenance,
    )
    spans_by_id = {span.span_id: span for span in spans}

    assert inferred.validate(spans_by_id) == ()
    missing = StructuralObject(
        object_id="sample.missing.provenance",
        object_type="State",
        evidence_span_ids=span_ids,
        observation_status=ObservationStatus.INFERRED,
    )
    assert (
        "inferred_field_provenance_missing"
        in missing.validate(spans_by_id)
    )
    mismatched = StructuralObject(
        object_id="sample.mismatched.provenance",
        object_type="State",
        evidence_span_ids=span_ids,
        observation_status=ObservationStatus.INFERRED,
        inference_provenance=InferenceProvenance(
            extractor_id="extractor.structural",
            extractor_version="version.one",
            extractor_implementation_hash="a" * 64,
            input_evidence_hash="c" * 64,
            calibration_bucket="bucket.high",
        ),
    )
    assert (
        "inferred_field_input_hash_mismatch"
        in mismatched.validate(spans_by_id)
    )


def test_safe_payloads_do_not_expose_raw_ids_or_offsets() -> None:
    _, _, episode, binding = _monotone_fixture()
    span = episode.evidence_spans[0]
    safe_blob = json.dumps(
        {
            "span": span.safe_payload(),
            "episode": episode.safe_payload(),
            "binding": binding.safe_payload(),
        },
        sort_keys=True,
    )

    for raw_value in (
        span.span_id,
        episode.episode_id,
        binding.binding_id,
        episode.objects[0].object_id,
        episode.quantities[0].quantity_id,
    ):
        assert raw_value not in safe_blob
    assert "start_byte" not in safe_blob
    assert "end_byte" not in safe_blob


def test_quantity_owner_swap_changes_role_normalized_signature() -> None:
    registry, schema, episode, binding = _monotone_fixture()
    (
        swapped_registry,
        swapped_schema,
        swapped_episode,
        swapped_binding,
    ) = _monotone_fixture(swap_quantity_owners=True)

    assert registry.registry_hash == swapped_registry.registry_hash
    assert schema.schema_hash == swapped_schema.schema_hash
    assert canonical_structural_signature(
        registry, schema, episode, binding
    ) != canonical_structural_signature(
        swapped_registry,
        swapped_schema,
        swapped_episode,
        swapped_binding,
    )
