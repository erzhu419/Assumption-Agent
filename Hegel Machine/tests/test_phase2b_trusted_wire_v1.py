"""Adversarial public-API tests for the Phase-2B trusted wire authority."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import runpy
import struct
from dataclasses import replace
from pathlib import Path

import pytest

import hegel_machine.phase2b_exact_transform_semantics_v1 as tx
import hegel_machine.phase2b_trusted_wire_v1 as trusted
from hegel_machine.phase2b_wire import (
    PUBLIC_EVIDENCE_SCHEMA_VERSION,
    AggregationEdge,
    AggregationGraph,
    EntityCandidate,
    MeasurementUncertainty,
    Missingness,
    NumericValue,
    PublicEvidenceBundle,
    TaskTarget,
    TemporalSupport,
    TransformOperation,
    TransformSpec,
    TypedObservation,
    UncertaintyModel,
    UnitDimension,
)


def uid(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012x}"


BUNDLE = uid(1)
TASK = uid(2)
ROOT_SCALE = uid(100)
TARGET_SCALE = uid(101)
TRANSFORM = uid(200)
ENTITY = uid(300)
ROLE = uid(301)
QUANTITY = uid(302)
CHANNEL = uid(303)
UNIT = uid(304)
CLOCK = uid(306)
HEADER = struct.Struct(">8sHHI32s32s")


def _payload_and_padding(envelope: bytes) -> tuple[bytes, bytes]:
    magic, version, header_length, payload_length, payload_hash, padding_hash = (
        HEADER.unpack(envelope[: HEADER.size])
    )
    assert magic == b"HGP2BW1\x00"
    assert version == 1
    assert header_length == HEADER.size == 80
    payload = envelope[header_length : header_length + payload_length]
    padding = envelope[header_length + payload_length :]
    assert hashlib.sha256(payload).digest() == payload_hash
    assert hashlib.sha256(padding).digest() == padding_hash
    return payload, padding


def _reframe(payload: bytes, padding: bytes) -> bytes:
    return HEADER.pack(
        b"HGP2BW1\x00",
        1,
        HEADER.size,
        len(payload),
        hashlib.sha256(payload).digest(),
        hashlib.sha256(padding).digest(),
    ) + payload + padding


def _public_padding(payload: bytes, length: int) -> bytes:
    """Independent replay of the explicitly non-secret stage-A padding."""

    domain = b"HEGEL/PHASE2B/PUBLIC_TEST_PADDING/V1\x00"
    digest = hashlib.sha256(payload).digest()
    chunks: list[bytes] = []
    for counter in range(math.ceil(length / 32)):
        chunks.append(
            hashlib.sha256(domain + digest + counter.to_bytes(4, "big")).digest()
        )
    return b"".join(chunks)[:length]


def _stage_a_envelope(payload: bytes) -> bytes:
    padding_length = trusted.ENVELOPE_BYTES - HEADER.size - len(payload)
    return _reframe(payload, _public_padding(payload, padding_length))


def _profile_payload_id(payload: bytes) -> str:
    domain = b"HEGEL/PHASE2B/TRUSTED_WIRE_PAYLOAD_ID/V1\x00"
    return "phase2b_wire_payload_" + hashlib.sha256(domain + payload).hexdigest()


def _base_bundle(
    operation: TransformOperation,
    values: tuple[float, ...],
    *,
    temporal: bool,
) -> PublicEvidenceBundle:
    observations = tuple(
        TypedObservation(
            observation_id=uid(1_000 + index),
            source_channel_id=CHANNEL,
            entity_ids=(ENTITY,),
            role_candidate_ids=(ROLE,),
            quantity_id=QUANTITY,
            value=NumericValue((value,)),
            unit_dimension=UnitDimension((0,) * 7),
            temporal_support=(
                TemporalSupport(CLOCK, float(index), float(index))
                if temporal
                else None
            ),
            spatial_support=None,
            uncertainty=MeasurementUncertainty(
                UncertaintyModel.ABSOLUTE_BOUND, (0.0,)
            ),
            provenance_sha256=str(index + 1) * 64,
            missingness=Missingness.OBSERVED,
        )
        for index, value in enumerate(values)
    )
    return PublicEvidenceBundle(
        schema_version=PUBLIC_EVIDENCE_SCHEMA_VERSION,
        bundle_id=BUNDLE,
        entity_candidates=(EntityCandidate(ENTITY, (ROLE,)),),
        role_ids=(ROLE,),
        quantity_ids=(QUANTITY,),
        observations=observations,
        task_target=TaskTarget(TASK, (ENTITY,), (QUANTITY,)),
        aggregation_graph=AggregationGraph(
            (ROOT_SCALE, TARGET_SCALE),
            (ROOT_SCALE,),
            (AggregationEdge(ROOT_SCALE, TARGET_SCALE, TRANSFORM),),
        ),
        transform_catalog=(TransformSpec(TRANSFORM, operation, ()),),
        missingness_mask=(),
    )


def identity_authority() -> tx.PublicTransformEvidenceBundleV2:
    base = _base_bundle(TransformOperation.IDENTITY, (1.0,), temporal=False)
    source_ref = tx.ComponentRef(ROOT_SCALE, uid(1_000), 0, uid(2_000))
    output_ref = tx.ComponentRef(TARGET_SCALE, uid(4_000), 0, uid(4_100))
    output = tx.ComponentDescriptor(
        output_ref,
        tx.ComponentAxis.SCALAR,
        tx.ComponentValueRole.INTENSIVE,
        UNIT,
        (0,) * 7,
        None,
        None,
        None,
    )
    contract = tx.ExactTransformContract(
        TRANSFORM,
        TransformOperation.IDENTITY,
        ROOT_SCALE,
        TARGET_SCALE,
        (source_ref,),
        (output,),
        (
            tx.DerivedObservationDescriptor(
                TARGET_SCALE,
                uid(4_000),
                CHANNEL,
                (ENTITY,),
                (ROLE,),
                QUANTITY,
                UNIT,
                (0,) * 7,
                None,
                None,
                "0" * 64,
                (uid(1_000),),
                tx.ComponentValueKind.NUMERIC_INTERVAL,
                (output_ref,),
            ),
        ),
        (
            tx.ExactSparseAffineRow(
                output_ref,
                (tx.ExactSparseTerm(source_ref, tx.ONE),),
            ),
        ),
        (),
        tx.IdentityTransformCertificate(),
    )
    # This is a fixed golden issued by the executable transform authority for
    # this entirely public fixture.  The trusted-wire implementation must
    # independently replay it; this test never calls a private provenance signer.
    signed = replace(
        contract,
        output_observations=(
            replace(
                contract.output_observations[0],
                provenance_sha256=(
                    "acac5dc4622c480d4bea6c5d053f9fd3571f34b119ea607bcfba92988e399345"
                ),
            ),
        ),
    )
    return tx.PublicTransformEvidenceBundleV2(
        tx.PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION,
        base,
        (
            tx.ObservationComponentMetadata(
                uid(1_000),
                ROOT_SCALE,
                (uid(2_000),),
                tx.ComponentAxis.SCALAR,
                tx.ComponentValueRole.INTENSIVE,
                UNIT,
            ),
        ),
        (signed,),
    )


def sampling_authority() -> tx.PublicTransformEvidenceBundleV2:
    base = _base_bundle(
        TransformOperation.SAMPLING_RESOLUTION,
        (1.0, 2.0, 3.0),
        temporal=True,
    )
    source_refs = tuple(
        tx.ComponentRef(ROOT_SCALE, uid(1_000 + i), 0, uid(2_000 + 10 * i))
        for i in range(3)
    )
    output_refs = tuple(
        tx.ComponentRef(TARGET_SCALE, uid(4_000 + i), 0, uid(4_100 + i))
        for i in range(2)
    )
    supports = tuple(
        tx.ExactTemporalSupport(CLOCK, tx.ExactTransformAtom(i), tx.ExactTransformAtom(i))
        for i in range(2)
    )
    outputs = tuple(
        tx.ComponentDescriptor(
            ref,
            tx.ComponentAxis.TEMPORAL,
            tx.ComponentValueRole.INTENSIVE,
            UNIT,
            (0,) * 7,
            None,
            support,
            None,
        )
        for ref, support in zip(output_refs, supports, strict=True)
    )
    provenances = (
        "0498867a873b7f2c0f0eeee1bf6317c6dcc14e44d404b0fd2e8a85fc66b463a6",
        "2c5572678ef771d6176832b12436dc35aa69cc289695be5a95a3ad05d42ce4d2",
    )
    observations = tuple(
        tx.DerivedObservationDescriptor(
            TARGET_SCALE,
            ref.observation_id,
            CHANNEL,
            (ENTITY,),
            (ROLE,),
            QUANTITY,
            UNIT,
            (0,) * 7,
            support,
            None,
            provenance,
            (source_refs[i].observation_id,),
            tx.ComponentValueKind.NUMERIC_INTERVAL,
            (ref,),
        )
        for i, (ref, support, provenance) in enumerate(
            zip(output_refs, supports, provenances, strict=True)
        )
    )
    contract = tx.ExactTransformContract(
        TRANSFORM,
        TransformOperation.SAMPLING_RESOLUTION,
        ROOT_SCALE,
        TARGET_SCALE,
        source_refs,
        outputs,
        observations,
        tuple(
            tx.ExactSparseAffineRow(
                output,
                (tx.ExactSparseTerm(source, tx.ONE),),
            )
            for source, output in zip(source_refs[:2], output_refs, strict=True)
        ),
        (),
        tx.SamplingResolutionCertificate(
            tx.ComponentAxis.TEMPORAL,
            source_refs[:2],
            source_refs[2:],
            ((tx.ZERO,), (tx.ONE,)),
            1,
            None,
        ),
    )
    return tx.PublicTransformEvidenceBundleV2(
        tx.PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION,
        base,
        tuple(
            tx.ObservationComponentMetadata(
                uid(1_000 + i),
                ROOT_SCALE,
                (uid(2_000 + 10 * i),),
                tx.ComponentAxis.TEMPORAL,
                tx.ComponentValueRole.INTENSIVE,
                UNIT,
            )
            for i in range(3)
        ),
        (contract,),
    )


def test_fixture_construction_uses_only_public_transform_api() -> None:
    assert identity_authority().transform_contracts[0].operation is TransformOperation.IDENTITY
    assert sampling_authority().transform_contracts[0].operation is TransformOperation.SAMPLING_RESOLUTION


def _compilation(
    authority: tx.PublicTransformEvidenceBundleV2 | None = None,
) -> trusted.TrustedWireProfileCompilationV1:
    result = trusted.compile_transform_authority_profile_mechanics_v1(
        identity_authority() if authority is None else authority
    )
    assert type(result) is trusted.TrustedWireProfileCompilationV1
    assert result.disposition is trusted.ProfileDisposition.COMPLETE
    return result


def test_accepted_jcs_golden_sorts_keys_and_uses_short_escapes() -> None:
    value = {"z": None, "a": [True, False, 0, "\b\t\n\f\r\"\\"]}
    expected = b'{"a":[true,false,0,"\\b\\t\\n\\f\\r\\\"\\\\"],"z":null}'
    assert trusted.encode_phase2b_jcs_profile_v1(value) == expected
    assert trusted.decode_phase2b_jcs_profile_v1(expected) == value


@pytest.mark.parametrize(
    "payload",
    (
        b'{"a":1, "b":2}',
        b'{"b":2,"a":1}',
        b'{"a":1,"a":1}',
        b'{"a":01}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":-0}',
        b'{"a":1.0}',
        b'{"a":5e-324}',
        b'{"a":1e-6}',
        b'{"a":1e-7}',
        b'{"a":1e20}',
        b'{"a":1e21}',
    ),
)
def test_accepted_jcs_decoder_rejects_noncanonical_or_raw_binary64(payload: bytes) -> None:
    with pytest.raises(ValueError):
        trusted.decode_phase2b_jcs_profile_v1(payload)


@pytest.mark.parametrize("value", (-0.0, 5e-324, 1e-6, 1e-7, 1e20, 1e21))
def test_raw_binary64_is_never_accepted_as_a_json_number(value: float) -> None:
    with pytest.raises(TypeError, match="floats"):
        trusted.encode_phase2b_jcs_profile_v1({"value": value})


def test_binary64_authority_values_round_trip_by_exact_bits() -> None:
    compilation = _compilation()
    decoded = trusted.decode_phase2b_jcs_profile_v1(compilation.payload)
    assert type(decoded) is dict
    encoded = decoded["authority"]["base_bundle"]["observations"][0]["value"][  # type: ignore[index]
        "values"
    ][0]
    assert encoded == "f64be:" + struct.pack(">d", 1.0).hex()
    assert struct.unpack(">d", bytes.fromhex(encoded.removeprefix("f64be:")))[0] == 1.0


def test_profile_uses_decimal_string_pairs_for_exact_rationals() -> None:
    decoded = trusted.decode_phase2b_jcs_profile_v1(_compilation().payload)
    coefficient = decoded["authority"]["transform_contracts"][0]["kernel_rows"][0][  # type: ignore[index]
        "terms"
    ][0]["coefficient"]
    assert coefficient == {"denominator_decimal": "1", "numerator_decimal": "1"}


@pytest.mark.parametrize("value", ((1 << 53), -(1 << 53), True))
def test_safe_integer_and_exact_bool_boundaries_do_not_alias(value: object) -> None:
    if type(value) is bool:
        assert trusted.encode_phase2b_jcs_profile_v1(value) == b"true"
        assert trusted.encode_phase2b_jcs_profile_v1(1) == b"1"
    else:
        with pytest.raises(ValueError, match="safe range"):
            trusted.encode_phase2b_jcs_profile_v1(value)


def test_non_ascii_and_lone_surrogate_strings_fail_closed() -> None:
    for value in ("\N{GREEK SMALL LETTER ALPHA}", "\ud800"):
        with pytest.raises(ValueError, match="ASCII"):
            trusted.encode_phase2b_jcs_profile_v1({"value": value})
    with pytest.raises(ValueError, match="ASCII"):
        trusted.decode_phase2b_jcs_profile_v1(b'"\\ud800"')


def test_profile_resource_caps_apply_before_encoding_large_arrays_and_depth() -> None:
    with pytest.raises(ValueError, match="array|entry"):
        trusted.encode_phase2b_jcs_profile_v1([None] * 4_097)
    nested: object = None
    for _ in range(66):
        nested = [nested]
    with pytest.raises(ValueError, match="depth"):
        trusted.encode_phase2b_jcs_profile_v1(nested)


def test_identity_and_sampling_authorities_compile_to_closed_stage_a_payloads() -> None:
    for authority in (identity_authority(), sampling_authority()):
        compilation = _compilation(authority)
        decoded = trusted.decode_phase2b_jcs_profile_v1(compilation.payload)
        assert set(decoded) == {
            "authority",
            "field_manifest_id",
            "jcs_profile_id",
            "schema_version",
        }
        assert decoded["authority"]["schema_version"] == tx.PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION
        assert not compilation.public_id_renaming_applied
        assert not compilation.trusted_wire_builder_implemented
        assert not compilation.secret_padding_replay_implemented


def test_sampling_certificate_keeps_an_explicit_operation_specific_shape() -> None:
    decoded = trusted.decode_phase2b_jcs_profile_v1(
        _compilation(sampling_authority()).payload
    )
    contract = decoded["authority"]["transform_contracts"][0]
    assert contract["operation"] == "sampling_resolution"
    assert set(contract["certificate"]) == {
        "axis",
        "boundary_policy",
        "discarded_inputs",
        "grid_dimension",
        "grid_frame_id",
        "grid_points",
        "kernel_contract",
        "missing_policy",
        "selected_inputs",
    }


def test_forged_transform_provenance_abstains_atomically_without_payload() -> None:
    authority = identity_authority()
    contract = authority.transform_contracts[0]
    forged = replace(
        authority,
        transform_contracts=(
            replace(
                contract,
                output_observations=(
                    replace(contract.output_observations[0], provenance_sha256="f" * 64),
                ),
            ),
        ),
    )
    result = trusted.compile_transform_authority_profile_mechanics_v1(forged)
    assert type(result) is trusted.TrustedWireProfilePreflightRejectionV1
    assert result.disposition is trusted.ProfileDisposition.ABSTAIN
    assert not hasattr(result, "payload")


def test_oversize_exact_rational_is_rejected_before_any_profile_receipt() -> None:
    authority = identity_authority()
    contract = authority.transform_contracts[0]
    row = contract.kernel_rows[0]
    forged_row = replace(
        row,
        terms=(
            replace(row.terms[0], coefficient=tx.ExactTransformAtom(1 << 4_096)),
        ),
    )
    result = trusted.compile_transform_authority_profile_mechanics_v1(
        replace(authority, transform_contracts=(replace(contract, kernel_rows=(forged_row,)),))
    )
    assert type(result) is trusted.TrustedWireProfilePreflightRejectionV1
    assert result.disposition is trusted.ProfileDisposition.ABSTAIN
    assert not hasattr(result, "payload_id")


@pytest.mark.parametrize("forged_version", (object(), "schema-非ascii"))
def test_preflight_rejection_requires_exact_ascii_authority_schema_version(
    forged_version: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        trusted.TrustedWireProfilePreflightRejectionV1(
            trusted.ProfileDisposition.ABSTAIN,
            "test_rejection",
            uid(0),
            forged_version,  # type: ignore[arg-type]
        )


def test_namespace_registry_is_explicitly_10_frozen_plus_6_extensions() -> None:
    audit = _compilation().namespace_audit
    assert audit.frozen_minimum_namespaces == tuple(
        sorted(
            {
                "bundle",
                "observation",
                "entity",
                "role_candidate",
                "quantity",
                "context",
                "task",
                "scale",
                "aggregate_map",
                "transform",
            }
        )
    )
    assert set(audit.schema_registry_namespaces) == set(audit.frozen_minimum_namespaces) | {
        "source_channel",
        "clock",
        "frame",
        "unit",
        "component",
        "quotient_class",
    }
    assert len(audit.schema_registry_namespaces) == 16
    assert audit.zero_occurrence_namespaces == ("aggregate_map", "context")
    assert not audit.formal_uuid_namespace_field_audit


def test_all_uuid_occurrences_have_one_manifest_rule_and_canonical_path() -> None:
    audit = _compilation(sampling_authority()).namespace_audit
    assert audit.occurrences
    assert len({row.rule_id for row in audit.occurrences}) > 8
    assert all(row.json_pointer.startswith("/authority/") for row in audit.occurrences)
    assert audit.occurrences == tuple(
        sorted(audit.occurrences, key=lambda row: (row.namespace, row.json_pointer, row.public_uuid))
    )


def test_uuid_at_an_unregistered_path_and_cross_namespace_alias_fail() -> None:
    with pytest.raises(ValueError, match="manifest"):
        trusted.audit_namespace_paths_v1({"answer": uid(999)})
    authority = trusted.decode_phase2b_jcs_profile_v1(_compilation().payload)["authority"]
    forged = json.loads(json.dumps(authority))
    forged["base_bundle"]["task_target"]["task_id"] = ENTITY
    with pytest.raises(ValueError, match="alias"):
        trusted.audit_namespace_paths_v1(forged)


@pytest.mark.parametrize(
    "uuid_representation",
    (
        "00000000-0000-1000-8000-000000000999",
        uid(999).upper(),
        "{" + uid(999) + "}",
        "urn:uuid:" + uid(999),
    ),
)
def test_unknown_path_rejects_every_uuid_representation_not_only_lowercase_v4(
    uuid_representation: str,
) -> None:
    with pytest.raises(ValueError, match="UUID path"):
        trusted.audit_namespace_paths_v1({"unknown_identifier": uuid_representation})


@pytest.mark.parametrize(
    "invalid_public_uuid",
    (
        "00000000-0000-1000-8000-000000000999",
        uid(999).upper(),
        "{" + uid(999) + "}",
        "urn:uuid:" + uid(999),
    ),
)
def test_known_manifest_path_accepts_only_lowercase_canonical_uuidv4(
    invalid_public_uuid: str,
) -> None:
    authority = trusted.decode_phase2b_jcs_profile_v1(_compilation().payload)[
        "authority"
    ]
    forged = json.loads(json.dumps(authority))
    forged["base_bundle"]["bundle_id"] = invalid_public_uuid
    with pytest.raises(ValueError, match="manifested UUID path"):
        trusted.audit_namespace_paths_v1(forged)


def test_fixed_envelope_golden_header_lengths_hashes_and_stage_a_flags() -> None:
    compilation = _compilation()
    framed = trusted.frame_fixed_envelope_mechanics_v1(compilation)
    assert len(framed.envelope) == trusted.ENVELOPE_BYTES == 65_536
    assert trusted.ENVELOPE_HEADER_BYTES == HEADER.size == 80
    assert framed.padding_bytes >= 32
    payload, padding = _payload_and_padding(framed.envelope)
    assert payload == compilation.payload
    assert len(payload) == framed.payload_bytes
    assert len(padding) == framed.padding_bytes
    assert framed.public_test_padding_used
    assert not framed.secret_padding_replay_verified
    assert not framed.trusted_wire_builder_implemented


def test_structural_decoder_replays_payload_audit_but_not_secret_authority() -> None:
    framed = trusted.frame_fixed_envelope_mechanics_v1(_compilation())
    decoded = trusted.decode_and_audit_fixed_envelope_mechanics_v1(framed.envelope)
    assert decoded.envelope_id == framed.envelope_id
    assert decoded.payload_id == framed.payload_id
    assert decoded.namespace_audit.audit_id == framed.namespace_audit_id
    assert decoded.structural_hashes_verified
    assert decoded.canonical_profile_verified
    assert decoded.public_test_padding_verified
    assert not decoded.secret_padding_replay_verified
    assert not decoded.trusted_wire_builder_implemented


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value[:-1],
        lambda value: b"X" + value[1:],
        lambda value: value[:8] + b"\x00\x02" + value[10:],
        lambda value: value[:10] + b"\x00\x51" + value[12:],
        lambda value: value[:12] + b"\xff\xff\xff\xff" + value[16:],
        lambda value: value[:16] + bytes([value[16] ^ 1]) + value[17:],
        lambda value: value[:-1] + bytes([value[-1] ^ 1]),
    ),
)
def test_fixed_envelope_structural_and_hash_tampering_fails_closed(mutation) -> None:
    envelope = trusted.frame_fixed_envelope_mechanics_v1(_compilation()).envelope
    with pytest.raises(ValueError):
        trusted.decode_and_audit_fixed_envelope_mechanics_v1(mutation(envelope))


def test_outer_unknown_and_answer_fields_are_rejected_even_with_valid_hashes() -> None:
    compilation = _compilation()
    decoded = trusted.decode_phase2b_jcs_profile_v1(compilation.payload)
    for field in ("answer", "unknown"):
        forged = dict(decoded)
        forged[field] = "forbidden"
        payload = trusted.encode_phase2b_jcs_profile_v1(forged)
        with pytest.raises(ValueError, match="schema"):
            trusted.decode_and_audit_fixed_envelope_mechanics_v1(
                _stage_a_envelope(payload)
            )


def test_nested_answer_scale_uuid_cannot_hide_inside_a_transform_contract() -> None:
    compilation = _compilation()
    decoded = trusted.decode_phase2b_jcs_profile_v1(compilation.payload)
    decoded["authority"]["transform_contracts"][0]["answer"] = {  # type: ignore[index]
        "scale_id": uid(999)
    }
    payload = trusted.encode_phase2b_jcs_profile_v1(decoded)
    with pytest.raises(ValueError, match="UUID path"):
        trusted.decode_and_audit_fixed_envelope_mechanics_v1(
            _stage_a_envelope(payload)
        )


def test_manifest_rules_never_use_recursive_double_star_patterns() -> None:
    assert trusted.FIELD_NAMESPACE_RULES
    assert all(
        "**" not in rule.json_pointer_pattern.split("/")
        for rule in trusted.FIELD_NAMESPACE_RULES
    )


@pytest.fixture(scope="module")
def eight_certificate_authorities() -> tuple[
    tuple[str, type[object], tx.PublicTransformEvidenceBundleV2], ...
]:
    fixture_source = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_exact_transform_semantics_v1.py"))
    )
    rows = (
        ("identity", tx.IdentityTransformCertificate, identity_authority()),
        ("unit_conversion", tx.UnitConversionCertificate, fixture_source["unit_authority"]()),
        (
            "coordinate_affine",
            tx.CoordinateAffineCertificate,
            fixture_source["coordinate_authority"](),
        ),
        (
            "temporal_aggregation",
            tx.TemporalAggregationCertificate,
            fixture_source["temporal_authority"](),
        ),
        (
            "spatial_aggregation",
            tx.SpatialAggregationCertificate,
            fixture_source["spatial_authority"](),
        ),
        (
            "sampling_resolution",
            tx.SamplingResolutionCertificate,
            sampling_authority(),
        ),
        (
            "equivalent_split_merge",
            tx.EquivalentSplitMergeCertificate,
            fixture_source["split_authority"](),
        ),
        (
            "coarse_graining",
            tx.CoarseGrainingCertificate,
            fixture_source["coarse_authority"](),
        ),
    )
    return rows


def test_all_eight_real_certificate_authorities_compile_decode_and_audit(
    eight_certificate_authorities: tuple[
        tuple[str, type[object], tx.PublicTransformEvidenceBundleV2], ...
    ],
) -> None:
    assert len(eight_certificate_authorities) == 8
    for operation, certificate_type, authority in eight_certificate_authorities:
        contract = authority.transform_contracts[0]
        assert contract.operation.value == operation
        assert type(contract.certificate) is certificate_type
        compilation = _compilation(authority)
        framed = trusted.frame_fixed_envelope_mechanics_v1(compilation)
        decoded = trusted.decode_and_audit_fixed_envelope_mechanics_v1(
            framed.envelope
        )
        assert decoded.namespace_audit == compilation.namespace_audit
        assert decoded.payload_mapping["authority"]["transform_contracts"][0][  # type: ignore[index]
            "operation"
        ] == operation


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("field_manifest_id", "phase2b_field_manifest_" + "0" * 64),
        ("jcs_profile_id", "phase2b_jcs_profile_" + "0" * 64),
        ("schema_version", "hegel-machine-phase2b-trusted-wire-payload/999"),
    ),
)
def test_compilation_rejects_inner_identity_drift_with_coordinated_outer_roots(
    field: str,
    forged_value: str,
) -> None:
    original = _compilation()
    decoded = trusted.decode_phase2b_jcs_profile_v1(original.payload)
    assert type(decoded) is dict
    decoded[field] = forged_value
    payload = trusted.encode_phase2b_jcs_profile_v1(decoded)
    with pytest.raises(ValueError, match="identity drift"):
        replace(
            original,
            payload=payload,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            payload_id=_profile_payload_id(payload),
        )


@pytest.mark.parametrize("forgery", ("rule", "path", "namespace"))
def test_namespace_occurrence_rejects_wrong_rule_path_or_namespace(
    forgery: str,
) -> None:
    row = _compilation().namespace_audit.occurrences[0]
    changes = {
        "rule": {"rule_id": "phase2b_namespace_rule_" + "0" * 64},
        "path": {"json_pointer": "/authority/unregistered/path"},
        "namespace": {"namespace": "quantity" if row.namespace != "quantity" else "entity"},
    }
    with pytest.raises(ValueError, match="rule|match"):
        replace(row, **changes[forgery])


def test_manifested_uuid_path_changed_to_non_uuid_fails_after_full_reframe() -> None:
    compilation = _compilation()
    decoded = trusted.decode_phase2b_jcs_profile_v1(compilation.payload)
    decoded["authority"]["base_bundle"]["bundle_id"] = "not-a-uuid"  # type: ignore[index]
    payload = trusted.encode_phase2b_jcs_profile_v1(decoded)
    with pytest.raises(ValueError, match="manifested UUID path"):
        trusted.decode_and_audit_fixed_envelope_mechanics_v1(
            _stage_a_envelope(payload)
        )


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("envelope_id", "phase2b_fixed_envelope_" + "0" * 64),
        ("payload_id", "phase2b_wire_payload_" + "0" * 64),
        ("payload_sha256", "0" * 64),
        ("padding_sha256", "0" * 64),
        ("payload_bytes", 1),
        ("padding_bytes", 1),
        ("namespace_audit_id", "phase2b_namespace_audit_" + "0" * 64),
    ),
)
def test_fixed_envelope_receipt_rejects_every_single_field_forgery(
    field: str,
    forged_value: object,
) -> None:
    receipt = trusted.frame_fixed_envelope_mechanics_v1(_compilation())
    with pytest.raises(ValueError, match="lengths drift|does not replay"):
        replace(receipt, **{field: forged_value})


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("envelope_id", "phase2b_fixed_envelope_" + "0" * 64),
        ("payload_id", "phase2b_wire_payload_" + "0" * 64),
        ("payload_sha256", "0" * 64),
        ("padding_sha256", "0" * 64),
        ("payload_bytes", 1),
        ("padding_bytes", 1),
    ),
)
def test_decoded_receipt_rejects_every_single_field_forgery(
    field: str,
    forged_value: object,
) -> None:
    envelope = trusted.frame_fixed_envelope_mechanics_v1(_compilation()).envelope
    receipt = trusted.decode_and_audit_fixed_envelope_mechanics_v1(envelope)
    with pytest.raises(ValueError, match="does not replay"):
        replace(receipt, **{field: forged_value})


def test_decoded_receipt_rejects_a_forged_namespace_audit() -> None:
    envelope = trusted.frame_fixed_envelope_mechanics_v1(_compilation()).envelope
    receipt = trusted.decode_and_audit_fixed_envelope_mechanics_v1(envelope)
    forged_audit = replace(receipt.namespace_audit, occurrences=())
    with pytest.raises(ValueError, match="does not replay"):
        replace(receipt, namespace_audit=forged_audit)


def test_framer_revalidates_object_setattr_polluted_compilation() -> None:
    compilation = _compilation()
    object.__setattr__(compilation, "payload", b"{}")
    with pytest.raises(ValueError, match="schema|identity|payload"):
        trusted.frame_fixed_envelope_mechanics_v1(compilation)


def test_payload_mapping_mutation_cannot_change_or_poison_decoded_receipt() -> None:
    envelope = trusted.frame_fixed_envelope_mechanics_v1(_compilation()).envelope
    receipt = trusted.decode_and_audit_fixed_envelope_mechanics_v1(envelope)
    first = receipt.payload_mapping
    first["answer"] = "forged"
    first["authority"]["base_bundle"]["bundle_id"] = "mutated"  # type: ignore[index]
    fresh = receipt.payload_mapping
    assert "answer" not in fresh
    assert fresh["authority"]["base_bundle"]["bundle_id"] == BUNDLE  # type: ignore[index]
    receipt.__post_init__()


def test_stage_a_compilation_keeps_all_seven_new_claims_false() -> None:
    compilation = _compilation()
    assert not compilation.global_shuffle_applied
    assert not compilation.hmac_uuid_assignment_applied
    assert not compilation.provenance_rebound_to_public_payload
    assert not compilation.typed_authority_decode_replay_implemented
    assert not compilation.batch_atomic_builder_implemented
    assert not compilation.origin_authenticated
    assert not compilation.formal_covert_audit


def test_envelope_receipts_keep_new_authentication_and_audit_claims_false() -> None:
    framed = trusted.frame_fixed_envelope_mechanics_v1(_compilation())
    decoded = trusted.decode_and_audit_fixed_envelope_mechanics_v1(framed.envelope)
    assert not framed.origin_authenticated
    assert not framed.formal_covert_audit
    assert not decoded.typed_authority_decode_replay_implemented
    assert not decoded.origin_authenticated
    assert not decoded.formal_covert_audit


def test_trusted_wire_module_has_no_legacy_normative_hash_or_json_import() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "hegel_machine"
        / "phase2b_trusted_wire_v1.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "canonical_json" not in imported_names
    assert "stable_hash" not in imported_names
