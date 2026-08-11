"""Adversarial public-API tests for strict typed trusted-wire replay."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import hashlib
import inspect
import runpy
import struct
from pathlib import Path

import pytest

import hegel_machine.phase2b_exact_transform_semantics_v1 as transform
import hegel_machine.phase2b_trusted_wire_batch_v1 as batch_wire
import hegel_machine.phase2b_trusted_wire_typed_authority_v1 as typed_codec
import hegel_machine.phase2b_trusted_wire_typed_replay_v1 as typed_replay
from hegel_machine.phase2b_trusted_wire_v1 import (
    ENVELOPE_BYTES,
    ENVELOPE_HEADER_BYTES,
    ENVELOPE_MAGIC,
    FIELD_MANIFEST_ID,
    JCS_PROFILE_ID,
    TRUSTED_WIRE_ENVELOPE_VERSION,
    TRUSTED_WIRE_PAYLOAD_SCHEMA_VERSION,
    decode_phase2b_jcs_profile_v1,
    encode_phase2b_jcs_profile_v1,
)
from hegel_machine.phase2b_wire import NumericValue


HEADER = struct.Struct(">8sHHI32s32s")
RUN_ID = b"R" * 32
SECRET_REPLAY_RECEIPT_DOMAIN = (
    b"HEGEL/PHASE2B/TRUSTED_WIRE/SECRET_REPLAY_RECEIPT/V1\x00"
)
FACTORY_NAMES = (
    "identity_authority",
    "unit_authority",
    "coordinate_authority",
    "temporal_authority",
    "spatial_authority",
    "sampling_authority",
    "split_authority",
    "coarse_authority",
    "two_step_authority",
)
CERTIFICATE_FACTORIES = FACTORY_NAMES[:-1]


def _keys(
    *,
    identifiers: bytes = b"I" * 32,
) -> batch_wire.TrustedWireKeySourcesV1:
    return batch_wire.TrustedWireKeySourcesV1(
        b"S" * 32,
        identifiers,
        b"P" * 32,
    )


@pytest.fixture(scope="module")
def authorities() -> dict[str, transform.PublicTransformEvidenceBundleV2]:
    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_exact_transform_semantics_v1.py"))
    )
    return {name: namespace[name]() for name in FACTORY_NAMES}


@pytest.fixture(scope="module")
def stage_a_sampling_authority() -> transform.PublicTransformEvidenceBundleV2:
    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_trusted_wire_v1.py"))
    )
    return namespace["sampling_authority"]()


@pytest.fixture(scope="module")
def complete_batch(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> tuple[
    tuple[transform.PublicTransformEvidenceBundleV2, ...],
    batch_wire.TrustedWireBatchV1,
]:
    values = tuple(authorities[name] for name in FACTORY_NAMES)
    result = batch_wire.build_trusted_wire_batch_v1(
        authorities=values,
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(result) is batch_wire.TrustedWireBatchV1
    return values, result


@pytest.fixture(scope="module")
def complete_typed_replay(
    complete_batch: tuple[
        tuple[transform.PublicTransformEvidenceBundleV2, ...],
        batch_wire.TrustedWireBatchV1,
    ],
) -> typed_replay.TypedTrustedWireBatchReplayV1:
    authorities, batch = complete_batch
    result = typed_replay.replay_typed_trusted_wire_batch_v1(
        batch=batch,
        run_id=RUN_ID,
        key_sources=_keys(),
        authorities=authorities,
    )
    assert type(result) is typed_replay.TypedTrustedWireBatchReplayV1
    return result


@pytest.fixture(scope="module")
def identity_batch(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> tuple[
    tuple[transform.PublicTransformEvidenceBundleV2, ...],
    batch_wire.TrustedWireBatchV1,
]:
    values = (authorities["identity_authority"],)
    result = batch_wire.build_trusted_wire_batch_v1(
        authorities=values,
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(result) is batch_wire.TrustedWireBatchV1
    return values, result


def _payload_parts(envelope: bytes) -> tuple[bytes, bytes, dict[str, object]]:
    magic, version, header_bytes, payload_bytes, payload_sha, padding_sha = (
        HEADER.unpack(envelope[: HEADER.size])
    )
    assert magic == ENVELOPE_MAGIC
    assert version == TRUSTED_WIRE_ENVELOPE_VERSION
    assert header_bytes == ENVELOPE_HEADER_BYTES == HEADER.size
    payload = envelope[header_bytes : header_bytes + payload_bytes]
    padding = envelope[header_bytes + payload_bytes :]
    assert len(envelope) == ENVELOPE_BYTES
    assert hashlib.sha256(payload).digest() == payload_sha
    assert hashlib.sha256(padding).digest() == padding_sha
    decoded = decode_phase2b_jcs_profile_v1(payload)
    assert type(decoded) is dict
    return payload, padding, decoded


def _reframe(payload: bytes, padding: bytes) -> bytes:
    target_padding_bytes = ENVELOPE_BYTES - ENVELOPE_HEADER_BYTES - len(payload)
    if target_padding_bytes < 0:
        raise ValueError("test payload exceeds the fixed envelope")
    padding = (
        padding[:target_padding_bytes]
        if len(padding) >= target_padding_bytes
        else padding + b"\x00" * (target_padding_bytes - len(padding))
    )
    return HEADER.pack(
        ENVELOPE_MAGIC,
        TRUSTED_WIRE_ENVELOPE_VERSION,
        ENVELOPE_HEADER_BYTES,
        len(payload),
        hashlib.sha256(payload).digest(),
        hashlib.sha256(padding).digest(),
    ) + payload + padding


def _profile(
    authority: transform.PublicTransformEvidenceBundleV2,
) -> dict[str, object]:
    return typed_codec.encode_typed_transform_authority_profile_v1(authority)


def _walk(value: object, path: tuple[object, ...]) -> object:
    current = value
    for part in path:
        current = current[part]  # type: ignore[index]
    return current


def _ref_key(value: dict[str, object]) -> tuple[object, ...]:
    return (
        value["scale_id"],
        value["observation_id"],
        value["ordinal"],
        value["component_id"],
    )


def _grid_key(point: list[dict[str, str]]) -> tuple[Fraction, ...]:
    return tuple(
        Fraction(
            int(atom["numerator_decimal"]),
            int(atom["denominator_decimal"]),
        )
        for atom in point
    )


def _coherent_secret_replay_id(
    receipt: batch_wire.TrustedWireReplayReceiptV1,
    source_authority_content_ids: tuple[str, ...],
) -> str:
    payload = encode_phase2b_jcs_profile_v1(
        {
            "authority_count": receipt.authority_count,
            "batch_id": receipt.batch_id,
            "policy_id": batch_wire.TRUSTED_WIRE_BATCH_POLICY_ID,
            "run_id_commitment": receipt.run_id_commitment,
            "source_authority_content_ids": list(source_authority_content_ids),
        }
    )
    return "phase2b_trusted_wire_secret_replay_" + hashlib.sha256(
        SECRET_REPLAY_RECEIPT_DOMAIN + payload
    ).hexdigest()


def test_public_api_is_closed_and_policy_roots_are_frozen() -> None:
    assert tuple(
        inspect.signature(
            typed_codec.decode_typed_transform_authority_profile_v1
        ).parameters
    ) == ("authority_profile",)
    assert tuple(
        inspect.signature(
            typed_codec.encode_typed_transform_authority_profile_v1
        ).parameters
    ) == ("authority",)
    assert tuple(
        inspect.signature(
            typed_replay.decode_and_replay_typed_trusted_envelope_v1
        ).parameters
    ) == ("envelope",)
    batch_signature = inspect.signature(
        typed_replay.replay_typed_trusted_wire_batch_v1
    )
    assert tuple(batch_signature.parameters) == (
        "batch",
        "run_id",
        "key_sources",
        "authorities",
    )
    assert all(
        value.kind is inspect.Parameter.KEYWORD_ONLY
        for value in batch_signature.parameters.values()
    )
    assert typed_codec.TYPED_AUTHORITY_SCHEMA_ID == (
        "phase2b_trusted_wire_typed_authority_schema_"
        "9429e96b9192db4546b92b011779e99352decb051a023d8883682539c804b730"
    )
    assert typed_codec.TYPED_AUTHORITY_CODEC_POLICY_ID == (
        "phase2b_trusted_wire_typed_authority_codec_policy_"
        "8fc5714399f0e31c43bf8fa3c818c4cc8afb8b4c49b57ae8af309821abfcc4b3"
    )
    assert typed_replay.TYPED_TRUSTED_WIRE_REPLAY_POLICY_ID == (
        "phase2b_typed_trusted_wire_replay_policy_"
        "4fc248af3d0b86a4a7a5f5735c9ffc4f253da8eeddf0cfbec8d5281ec0c5f7d2"
    )
    assert transform.EXACT_TRANSFORM_POLICY_ID == (
        "phase2b_exact_transform_policy_"
        "c49a74a45af3e272d800fece85f6862ae557f0c3b071dded04f6b6c2b8a7862e"
    )
    assert batch_wire.TRUSTED_WIRE_BATCH_SCHEMA_VERSION.endswith("/2")
    assert batch_wire.TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION.endswith("/2")
    assert batch_wire.TRUSTED_WIRE_BATCH_POLICY_ID != (
        "phase2b_trusted_wire_batch_policy_"
        "b50927c5ec9a39af98d1e4674e9d8d365560f2f8c6c9ca59c90c40c65c45d290"
    )


def test_all_eight_certificates_and_two_step_codec_round_trip_exactly(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    expected_types = {
        transform.TransformOperation.IDENTITY: transform.IdentityTransformCertificate,
        transform.TransformOperation.UNIT_CONVERSION: transform.UnitConversionCertificate,
        transform.TransformOperation.COORDINATE_AFFINE: transform.CoordinateAffineCertificate,
        transform.TransformOperation.TEMPORAL_AGGREGATION: transform.TemporalAggregationCertificate,
        transform.TransformOperation.SPATIAL_AGGREGATION: transform.SpatialAggregationCertificate,
        transform.TransformOperation.SAMPLING_RESOLUTION: transform.SamplingResolutionCertificate,
        transform.TransformOperation.EQUIVALENT_SPLIT_MERGE: transform.EquivalentSplitMergeCertificate,
        transform.TransformOperation.COARSE_GRAINING: transform.CoarseGrainingCertificate,
    }
    seen: set[transform.TransformOperation] = set()
    for name in FACTORY_NAMES:
        authority = authorities[name]
        profile = _profile(authority)
        decoded = typed_codec.decode_typed_transform_authority_profile_v1(profile)
        assert decoded == authority
        assert _profile(decoded) == profile
        result = transform.run_exact_transform_semantics(decoded)
        assert type(result) is transform.ExactTransformCompilation
        assert result.disposition is transform.TransformCompilationDisposition.COMPLETE
        assert result.wrapper_content_id == decoded.content_id
        for contract in decoded.transform_contracts:
            assert type(contract.certificate) is expected_types[contract.operation]
            seen.add(contract.operation)
    assert seen == set(expected_types)
    assert len(authorities["two_step_authority"].transform_contracts) == 2


def test_real_batch_payloads_round_trip_byte_exact_and_direct_complete(
    complete_batch: tuple[
        tuple[transform.PublicTransformEvidenceBundleV2, ...],
        batch_wire.TrustedWireBatchV1,
    ],
) -> None:
    _, value = complete_batch
    operations: set[transform.TransformOperation] = set()
    saw_two_step = False
    for row in value.envelopes:
        payload, _, decoded_payload = _payload_parts(row.envelope)
        assert tuple(sorted(decoded_payload)) == tuple(
            sorted(
                (
                    "authority",
                    "field_manifest_id",
                    "jcs_profile_id",
                    "public_provenance_version",
                    "schema_version",
                    "typed_authority_schema_id",
                )
            )
        )
        authority = typed_codec.decode_typed_transform_authority_profile_v1(
            decoded_payload["authority"]
        )
        rebuilt = dict(decoded_payload)
        rebuilt["authority"] = _profile(authority)
        assert encode_phase2b_jcs_profile_v1(rebuilt) == payload
        result = transform.run_exact_transform_semantics(authority)
        assert type(result) is transform.ExactTransformCompilation
        assert result.disposition is transform.TransformCompilationDisposition.COMPLETE
        assert result.wrapper_content_id == authority.content_id
        operations.update(contract.operation for contract in authority.transform_contracts)
        saw_two_step |= len(authority.transform_contracts) == 2
    assert operations == set(transform.TransformOperation)
    assert saw_two_step


def test_old_four_field_v1_payload_and_extra_v2_field_fail_closed(
    complete_batch: tuple[
        tuple[transform.PublicTransformEvidenceBundleV2, ...],
        batch_wire.TrustedWireBatchV1,
    ],
) -> None:
    _, value = complete_batch
    payload, padding, decoded = _payload_parts(value.envelopes[0].envelope)
    old_payload = encode_phase2b_jcs_profile_v1(
        {
            "authority": decoded["authority"],
            "field_manifest_id": FIELD_MANIFEST_ID,
            "jcs_profile_id": JCS_PROFILE_ID,
            "schema_version": TRUSTED_WIRE_PAYLOAD_SCHEMA_VERSION,
        }
    )
    with pytest.raises(ValueError, match="payload schema drift"):
        batch_wire.decode_and_audit_trusted_envelope_v1(
            _reframe(old_payload, padding)
        )
    extra = dict(decoded)
    extra["forbidden"] = None
    with pytest.raises(ValueError, match="payload schema drift"):
        batch_wire.decode_and_audit_trusted_envelope_v1(
            _reframe(encode_phase2b_jcs_profile_v1(extra), padding)
        )
    assert payload != old_payload


@pytest.mark.parametrize(
    "path",
    (
        (),
        ("base_bundle",),
        ("base_bundle", "observations", 0),
        ("observation_metadata", 0),
        ("transform_contracts", 0),
        ("transform_contracts", 0, "input_components", 0),
        ("transform_contracts", 0, "kernel_rows", 0),
        ("transform_contracts", 0, "kernel_rows", 0, "terms", 0),
        ("transform_contracts", 0, "output_observations", 0),
    ),
)
def test_unknown_fields_are_rejected_at_every_schema_layer(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    path: tuple[object, ...],
) -> None:
    profile = _profile(authorities["identity_authority"])
    target = _walk(profile, path)
    assert type(target) is dict
    target["forbidden"] = None
    with pytest.raises((TypeError, ValueError), match="schema|profile"):
        typed_codec.decode_typed_transform_authority_profile_v1(profile)


@pytest.mark.parametrize("factory_name", CERTIFICATE_FACTORIES)
def test_all_certificate_shapes_reject_unknown_and_missing_fields(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    factory_name: str,
) -> None:
    original = _profile(authorities[factory_name])
    certificate = original["transform_contracts"][0]["certificate"]  # type: ignore[index]
    assert type(certificate) is dict and certificate
    with_extra = deepcopy(original)
    with_extra["transform_contracts"][0]["certificate"]["forbidden"] = None  # type: ignore[index]
    with pytest.raises((TypeError, ValueError), match="schema"):
        typed_codec.decode_typed_transform_authority_profile_v1(with_extra)
    with_missing = deepcopy(original)
    missing_certificate = with_missing["transform_contracts"][0]["certificate"]  # type: ignore[index]
    missing_certificate.pop(sorted(missing_certificate)[0])
    with pytest.raises((TypeError, ValueError), match="schema"):
        typed_codec.decode_typed_transform_authority_profile_v1(with_missing)


@pytest.mark.parametrize(
    "path,value",
    (
        (("observation_metadata", 0, "axis"), "unknown-axis"),
        (("transform_contracts", 0, "operation"), "unknown-operation"),
        (("transform_contracts", 0, "certificate", "missing_policy"), "unknown-policy"),
        (("transform_contracts", 0, "output_observations", 0, "value_kind"), "unknown-kind"),
        (("base_bundle", "observations", 0, "uncertainty", "model"), "unknown-model"),
        (("base_bundle", "observations", 0, "missingness"), "unknown-missingness"),
    ),
)
def test_enum_discriminators_fail_closed(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    path: tuple[object, ...],
    value: object,
) -> None:
    profile = _profile(authorities["identity_authority"])
    parent = _walk(profile, path[:-1])
    parent[path[-1]] = value  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown discriminator"):
        typed_codec.decode_typed_transform_authority_profile_v1(profile)


def test_operation_certificate_discriminator_cannot_be_spoofed(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    identity = _profile(authorities["identity_authority"])
    unit = _profile(authorities["unit_authority"])
    identity["transform_contracts"][0]["certificate"] = unit["transform_contracts"][0]["certificate"]  # type: ignore[index]
    with pytest.raises((TypeError, ValueError), match="schema"):
        typed_codec.decode_typed_transform_authority_profile_v1(identity)


@pytest.mark.parametrize(
    "invalid",
    (
        0,
        True,
        "0",
        "f64be:000000000000000",
        "f64be:000000000000000A",
        "f64be:7ff0000000000000",
        "f64be:fff0000000000000",
        "f64be:7ff8000000000000",
    ),
)
def test_float_fields_require_exact_finite_f64be_strings(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    invalid: object,
) -> None:
    profile = _profile(authorities["identity_authority"])
    profile["base_bundle"]["observations"][0]["value"]["values"][0] = invalid  # type: ignore[index]
    with pytest.raises((TypeError, ValueError), match="string|f64be|finite"):
        typed_codec.decode_typed_transform_authority_profile_v1(profile)


@pytest.mark.parametrize(
    "factory_name,path,replacement",
    (
        (
            "identity_authority",
            ("base_bundle", "observations", 0, "uncertainty", "radius", 0),
            0,
        ),
        (
            "temporal_authority",
            ("base_bundle", "observations", 0, "temporal_support", "start"),
            0,
        ),
        (
            "spatial_authority",
            ("base_bundle", "observations", 0, "spatial_support", "lower", 0),
            0,
        ),
        (
            "unit_authority",
            ("base_bundle", "transform_catalog", 0, "parameters"),
            [0],
        ),
    ),
)
def test_every_binary64_schema_family_rejects_raw_jcs_integers(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    factory_name: str,
    path: tuple[object, ...],
    replacement: object,
) -> None:
    profile = _profile(authorities[factory_name])
    parent = _walk(profile, path[:-1])
    parent[path[-1]] = replacement  # type: ignore[index]
    with pytest.raises((TypeError, ValueError), match="string|f64be"):
        typed_codec.decode_typed_transform_authority_profile_v1(profile)


@pytest.mark.parametrize(
    "wire_value,bits",
    (
        ("f64be:8000000000000000", bytes.fromhex("8000000000000000")),
        ("f64be:0000000000000001", bytes.fromhex("0000000000000001")),
    ),
)
def test_negative_zero_and_subnormal_round_trip_by_exact_bits(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    wire_value: str,
    bits: bytes,
) -> None:
    profile = _profile(authorities["identity_authority"])
    profile["base_bundle"]["observations"][0]["value"]["values"][0] = wire_value  # type: ignore[index]
    decoded = typed_codec.decode_typed_transform_authority_profile_v1(profile)
    value = decoded.base_bundle.observations[0].value
    assert type(value) is NumericValue
    assert struct.pack(">d", value.values[0]) == bits
    assert _profile(decoded) == profile


@pytest.mark.parametrize(
    "numerator,denominator",
    (
        ("01", "1"),
        ("-0", "1"),
        ("+1", "1"),
        ("1", "0"),
        ("1", "-1"),
        ("2", "2"),
        ("1" + "0" * 1_300, "1"),
    ),
)
def test_rationals_require_reduced_canonical_decimal_pairs_with_bit_caps(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    numerator: str,
    denominator: str,
) -> None:
    profile = _profile(authorities["identity_authority"])
    atom = profile["transform_contracts"][0]["kernel_rows"][0]["terms"][0]["coefficient"]  # type: ignore[index]
    atom["numerator_decimal"] = numerator
    atom["denominator_decimal"] = denominator
    with pytest.raises((TypeError, ValueError), match="canonical|positive|reduced|bit"):
        typed_codec.decode_typed_transform_authority_profile_v1(profile)


def test_valid_negative_rational_round_trips_exactly(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    profile = _profile(authorities["identity_authority"])
    atom = profile["transform_contracts"][0]["kernel_rows"][0]["terms"][0]["coefficient"]  # type: ignore[index]
    atom["numerator_decimal"] = "-1"
    atom["denominator_decimal"] = "2"
    decoded = typed_codec.decode_typed_transform_authority_profile_v1(profile)
    coefficient = decoded.transform_contracts[0].kernel_rows[0].terms[0].coefficient
    assert coefficient.as_fraction() == Fraction(-1, 2)
    assert _profile(decoded) == profile


@pytest.mark.parametrize(
    "mutation",
    ("kind", "hybrid", "missing", "integer_as_string", "scalar_as_list"),
)
def test_wire_grammar_and_typed_value_union_are_exact(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    mutation: str,
) -> None:
    profile = _profile(authorities["identity_authority"])
    observation = profile["base_bundle"]["observations"][0]  # type: ignore[index]
    if mutation == "kind":
        observation["value"]["kind"] = "numeric"  # type: ignore[index]
    elif mutation == "hybrid":
        observation["value"]["value"] = True  # type: ignore[index]
    elif mutation == "missing":
        observation["value"] = {}
    elif mutation == "integer_as_string":
        profile["transform_contracts"][0]["input_components"][0]["ordinal"] = "0"  # type: ignore[index]
    else:
        profile["transform_contracts"][0]["input_components"][0]["ordinal"] = [0]  # type: ignore[index]
    with pytest.raises((TypeError, ValueError), match="typed-value|integer|schema"):
        typed_codec.decode_typed_transform_authority_profile_v1(profile)


def test_noncanonical_source_order_is_rejected_not_silently_rewritten(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    profile = _profile(authorities["two_step_authority"])
    profile["transform_contracts"].reverse()  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="canonical"):
        typed_codec.decode_typed_transform_authority_profile_v1(profile)


def test_large_untrusted_trees_are_capped_before_schema_set_operations() -> None:
    oversized_root = {f"k{index}": None for index in range(4_097)}
    with pytest.raises(ValueError, match="object.*cap"):
        typed_codec.decode_typed_transform_authority_profile_v1(oversized_root)
    nested = {
        "schema_version": [None] * 4_096,
        "base_bundle": [None] * 4_096,
        "observation_metadata": [None] * 4_096,
        "transform_contracts": [None] * 4_096,
    }
    with pytest.raises(ValueError, match="node cap|total entry cap"):
        typed_codec.decode_typed_transform_authority_profile_v1(nested)


def test_nested_builtin_and_dataclass_subclasses_cannot_spoof_exact_schema(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    class DictSpoof(dict):
        pass

    class ListSpoof(list):
        pass

    for path, wrapper in (
        (("base_bundle",), DictSpoof),
        (("observation_metadata",), ListSpoof),
    ):
        profile = _profile(authorities["identity_authority"])
        parent = _walk(profile, path[:-1])
        original = parent[path[-1]]  # type: ignore[index]
        parent[path[-1]] = wrapper(original)  # type: ignore[index]
        with pytest.raises((TypeError, ValueError), match="non-JCS|exact"):
            typed_codec.decode_typed_transform_authority_profile_v1(profile)

    class RefSpoof(transform.ComponentRef):
        pass

    authority = deepcopy(authorities["identity_authority"])
    contract = authority.transform_contracts[0]
    original_ref = contract.input_components[0]
    spoof = RefSpoof(
        original_ref.scale_id,
        original_ref.observation_id,
        original_ref.ordinal,
        original_ref.component_id,
    )
    object.__setattr__(contract, "input_components", (spoof, *contract.input_components[1:]))
    with pytest.raises(TypeError, match="non-schema"):
        typed_codec.encode_typed_transform_authority_profile_v1(authority)


def test_polluted_typed_authorities_hit_caps_before_profile_emission(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    oversized_tuple = deepcopy(authorities["identity_authority"])
    object.__setattr__(
        oversized_tuple,
        "observation_metadata",
        (oversized_tuple.observation_metadata[0],) * 4_097,
    )
    with pytest.raises(ValueError, match="tuple.*cap"):
        typed_codec.encode_typed_transform_authority_profile_v1(oversized_tuple)

    oversized_string = deepcopy(authorities["identity_authority"])
    object.__setattr__(
        oversized_string.base_bundle.observations[0],
        "provenance_sha256",
        "a" * 2_049,
    )
    with pytest.raises(ValueError, match="string cap"):
        typed_codec.encode_typed_transform_authority_profile_v1(oversized_string)

    oversized_integer = deepcopy(authorities["identity_authority"])
    object.__setattr__(
        oversized_integer.transform_contracts[0].input_components[0],
        "ordinal",
        1 << 53,
    )
    with pytest.raises(ValueError, match="safe-integer cap"):
        typed_codec.encode_typed_transform_authority_profile_v1(oversized_integer)

    oversized_atom = deepcopy(authorities["identity_authority"])
    coefficient = (
        oversized_atom.transform_contracts[0]
        .kernel_rows[0]
        .terms[0]
        .coefficient
    )
    object.__setattr__(coefficient, "numerator", 1 << 4_096)
    with pytest.raises(ValueError, match="bit cap"):
        typed_codec.encode_typed_transform_authority_profile_v1(oversized_atom)


def test_public_provenance_formula_binds_every_semantic_input(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    authority = authorities["two_step_authority"]
    first, second = authority.transform_contracts
    input_observation = first.output_observations[0]
    output_observation = second.output_observations[0]

    def root(
        *,
        source: transform.DerivedObservationDescriptor = input_observation,
        output: transform.DerivedObservationDescriptor = output_observation,
        uncertainty: tuple[str, ...] = ("uncertainty-a",),
        contract: str = "contract-a",
        paths: tuple[str, ...] = ("transform-a", "transform-b"),
        semantics: tuple[str, ...] = ("semantics-a", "semantics-b"),
    ) -> str:
        return transform.expected_derived_observation_provenance_v1(
            descriptor=output,
            input_observations=(source,),
            input_uncertainty_compilation_ids=uncertainty,
            contract_semantics_id=contract,
            ordered_transform_path_ids=paths,
            ordered_contract_semantics_ids=semantics,
        )

    roots = {
        root(),
        root(source=replace(input_observation, provenance_sha256="0" * 64)),
        root(
            source=replace(
                input_observation,
                source_channel_id=input_observation.observation_id,
            )
        ),
        root(uncertainty=("uncertainty-b",)),
        root(contract="contract-b"),
        root(paths=("transform-a", "transform-c")),
        root(semantics=("semantics-a", "semantics-c")),
        root(
            output=replace(
                output_observation,
                quantity_id=output_observation.observation_id,
            )
        ),
    }
    assert len(roots) == 8


def test_stale_derived_provenance_fails_until_native_preframe_compilation(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    authority = authorities["identity_authority"]
    contract = authority.transform_contracts[0]
    stale_contract = replace(
        contract,
        output_observations=(
            replace(contract.output_observations[0], provenance_sha256="0" * 64),
        ),
    )
    stale = replace(authority, transform_contracts=(stale_contract,))
    failed = transform.run_exact_transform_semantics(stale)
    assert type(failed) is transform.ExactTransformCompilation
    assert failed.disposition is transform.TransformCompilationDisposition.ABSTAIN
    assert failed.reason == "transform_bundle_atomic_rejection"
    assert tuple(item.error_code for item in failed.failures) == (
        "output_observation_provenance_mismatch",
    )
    compiled = transform.compile_exact_transform_provenance_v1(stale)
    assert stale.transform_contracts[0].output_observations[0].provenance_sha256 == "0" * 64
    assert compiled != stale
    complete = transform.run_exact_transform_semantics(compiled)
    assert type(complete) is transform.ExactTransformCompilation
    assert complete.disposition is transform.TransformCompilationDisposition.COMPLETE


def test_sampling_pairing_is_grid_exact_and_independent_of_output_uuid_rank(
    stage_a_sampling_authority: transform.PublicTransformEvidenceBundleV2,
) -> None:
    rows_by_key: dict[bytes, tuple[list[tuple[object, ...]], list[tuple[object, ...]]]] = {}
    for identifier in (b"I" * 32, b"J" * 32):
        built = batch_wire.build_trusted_wire_batch_v1(
            authorities=(stage_a_sampling_authority,),
            run_id=RUN_ID,
            key_sources=_keys(identifiers=identifier),
        )
        assert type(built) is batch_wire.TrustedWireBatchV1
        payload, _, decoded_payload = _payload_parts(built.envelopes[0].envelope)
        authority = typed_codec.decode_typed_transform_authority_profile_v1(
            decoded_payload["authority"]
        )
        rebuilt = dict(decoded_payload)
        rebuilt["authority"] = _profile(authority)
        assert encode_phase2b_jcs_profile_v1(rebuilt) == payload
        result = transform.run_exact_transform_semantics(authority)
        assert type(result) is transform.ExactTransformCompilation
        assert result.disposition is transform.TransformCompilationDisposition.COMPLETE
        assert result.transform_policy_id == transform.EXACT_TRANSFORM_POLICY_ID
        profile = decoded_payload["authority"]
        contract = profile["transform_contracts"][0]  # type: ignore[index]
        certificate = contract["certificate"]
        selected = [_ref_key(item) for item in certificate["selected_inputs"]]
        grid = [_grid_key(item) for item in certificate["grid_points"]]
        row_inputs = [
            _ref_key(row["terms"][0]["input_ref"])
            for row in contract["kernel_rows"]
        ]
        assert grid == sorted(grid)
        assert len(grid) == len(set(grid)) == len(selected) == len(set(selected))
        assert set(row_inputs) == set(selected)
        rows_by_key[identifier] = (selected, row_inputs)
    assert rows_by_key[b"I" * 32][0] == rows_by_key[b"I" * 32][1]
    assert rows_by_key[b"J" * 32][0] != rows_by_key[b"J" * 32][1]


def test_sampling_selected_grid_and_kernel_tampering_fail_closed(
    stage_a_sampling_authority: transform.PublicTransformEvidenceBundleV2,
) -> None:
    built = batch_wire.build_trusted_wire_batch_v1(
        authorities=(stage_a_sampling_authority,),
        run_id=RUN_ID,
        key_sources=_keys(identifiers=b"J" * 32),
    )
    assert type(built) is batch_wire.TrustedWireBatchV1
    _, _, decoded_payload = _payload_parts(built.envelopes[0].envelope)
    original = decoded_payload["authority"]
    mutations: list[dict[str, object]] = []

    swapped = deepcopy(original)
    selected = swapped["transform_contracts"][0]["certificate"]["selected_inputs"]  # type: ignore[index]
    selected[0], selected[1] = selected[1], selected[0]
    mutations.append(swapped)

    duplicate_selected = deepcopy(original)
    selected = duplicate_selected["transform_contracts"][0]["certificate"]["selected_inputs"]  # type: ignore[index]
    selected[1] = deepcopy(selected[0])
    mutations.append(duplicate_selected)

    duplicate_grid = deepcopy(original)
    grid = duplicate_grid["transform_contracts"][0]["certificate"]["grid_points"]  # type: ignore[index]
    grid[1] = deepcopy(grid[0])
    mutations.append(duplicate_grid)

    duplicate_kernel_input = deepcopy(original)
    kernel = duplicate_kernel_input["transform_contracts"][0]["kernel_rows"]  # type: ignore[index]
    kernel[1]["terms"][0]["input_ref"] = deepcopy(kernel[0]["terms"][0]["input_ref"])
    mutations.append(duplicate_kernel_input)

    for profile in mutations:
        authority = typed_codec.decode_typed_transform_authority_profile_v1(profile)
        with pytest.raises(ValueError, match="provenance|rejected|sampling|contract"):
            transform.compile_exact_transform_provenance_v1(authority)


def test_public_envelope_and_whole_batch_typed_replay_bind_exact_roots(
    complete_batch: tuple[
        tuple[transform.PublicTransformEvidenceBundleV2, ...],
        batch_wire.TrustedWireBatchV1,
    ],
    complete_typed_replay: typed_replay.TypedTrustedWireBatchReplayV1,
) -> None:
    authorities, batch = complete_batch
    public_rows = tuple(
        typed_replay.decode_and_replay_typed_trusted_envelope_v1(item.envelope)
        for item in batch.envelopes
    )
    for source, row in zip(batch.envelopes, public_rows, strict=True):
        assert row.envelope_id == source.envelope_id
        assert row.payload_sha256 == source.payload_sha256
        assert row.authority_content_id == row.authority.content_id
        assert row.transform_result_id == row.transform_result.result_id
        assert row.transform_result.wrapper_content_id == row.authority_content_id
        assert row.direct_payload_authority_transform_replay_verified
        assert row.lossless_typed_profile_roundtrip_verified
        assert not row.batch_policy_membership_verified
        assert not row.secret_padding_replay_verified
        assert not row.origin_authenticated
        assert not row.formal_covert_audit
        assert not row.sealed_holdout_eligible
        assert not row.c1_exit_evidence

    receipt = complete_typed_replay
    assert type(receipt) is typed_replay.TypedTrustedWireBatchReplayV1
    assert receipt.disposition is typed_replay.TypedTrustedWireReplayDisposition.COMPLETE
    assert receipt.batch == batch
    assert receipt.batch_id == batch.batch_id
    assert receipt.batch_policy_id == batch_wire.TRUSTED_WIRE_BATCH_POLICY_ID
    assert receipt.rows == public_rows
    assert receipt.source_authorities == authorities
    assert receipt.source_authority_content_ids == tuple(
        authority.content_id for authority in authorities
    )
    assert type(receipt.secret_replay_receipt) is batch_wire.TrustedWireReplayReceiptV1
    assert (
        receipt.secret_replay_receipt_id
        == receipt.secret_replay_receipt.replay_receipt_id
    )
    assert receipt.secret_replay_receipt.batch_id == batch.batch_id
    assert (
        receipt.secret_replay_receipt.source_authority_content_ids
        == receipt.source_authority_content_ids
    )
    assert receipt.authority_content_ids == tuple(
        row.authority_content_id for row in public_rows
    )
    assert receipt.transform_result_ids == tuple(
        row.transform_result_id for row in public_rows
    )
    assert receipt.batch_policy_membership_verified
    assert receipt.whole_batch_atomic_typed_replay_verified
    assert receipt.secret_custodian_replay_verified
    assert receipt.direct_payload_authority_transform_replay_verified
    assert not receipt.origin_authenticated
    assert not receipt.formal_covert_audit
    assert not receipt.sealed_holdout_eligible
    assert not receipt.c1_exit_evidence
    assert not batch.typed_authority_decode_replay_implemented


@pytest.mark.parametrize("failure", ("run", "key", "authority"))
def test_custodian_drift_abstains_atomically_without_partial_roots(
    identity_batch: tuple[
        tuple[transform.PublicTransformEvidenceBundleV2, ...],
        batch_wire.TrustedWireBatchV1,
    ],
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    failure: str,
) -> None:
    original_authorities, batch = identity_batch
    run_id = b"W" * 32 if failure == "run" else RUN_ID
    key_sources = _keys(identifiers=b"J" * 32) if failure == "key" else _keys()
    supplied = original_authorities
    if failure == "authority":
        supplied = (authorities["unit_authority"],)
    result = typed_replay.replay_typed_trusted_wire_batch_v1(
        batch=batch,
        run_id=run_id,
        key_sources=key_sources,
        authorities=supplied,
    )
    assert type(result) is typed_replay.TypedTrustedWireBatchReplayRejectionV1
    assert result.disposition is typed_replay.TypedTrustedWireReplayDisposition.ABSTAIN
    assert result.reason == "custodian_or_typed_replay_failed"
    assert result.batch_id == batch.batch_id
    assert result.rows == ()
    assert result.source_authority_content_ids == ()
    assert result.authority_content_ids == ()
    assert result.transform_result_ids == ()


@pytest.mark.parametrize(
    "field_name",
    (
        "rows",
        "source_authority_content_ids",
        "authority_content_ids",
        "transform_result_ids",
    ),
)
@pytest.mark.parametrize("falsy_spoof", ([], None, 0, ""))
def test_rejection_roots_require_exact_empty_tuples(
    field_name: str,
    falsy_spoof: object,
) -> None:
    values = {
        "disposition": typed_replay.TypedTrustedWireReplayDisposition.ABSTAIN,
        "reason": "test_rejection",
        "authority_count": 1,
        "batch_id": None,
        field_name: falsy_spoof,
    }
    with pytest.raises(ValueError, match="cannot expose partial roots"):
        typed_replay.TypedTrustedWireBatchReplayRejectionV1(**values)


def test_polluted_second_row_and_old_policy_abstain_without_any_partial_root(
    complete_batch: tuple[
        tuple[transform.PublicTransformEvidenceBundleV2, ...],
        batch_wire.TrustedWireBatchV1,
    ],
) -> None:
    authorities, original = complete_batch
    for field_name, field_value in (
        ("envelopes", None),
        (
            "policy_id",
            "phase2b_trusted_wire_batch_policy_"
            "b50927c5ec9a39af98d1e4674e9d8d365560f2f8c6c9ca59c90c40c65c45d290",
        ),
    ):
        polluted = deepcopy(original)
        if field_name == "envelopes":
            object.__setattr__(
                polluted.envelopes[1],
                "payload_sha256",
                "0" * 64,
            )
        else:
            object.__setattr__(polluted, field_name, field_value)
        result = typed_replay.replay_typed_trusted_wire_batch_v1(
            batch=polluted,
            run_id=RUN_ID,
            key_sources=_keys(),
            authorities=authorities,
        )
        assert type(result) is typed_replay.TypedTrustedWireBatchReplayRejectionV1
        assert result.reason == "batch_validation_failed"
        assert result.batch_id is None
        assert result.rows == ()
        assert result.source_authority_content_ids == ()
        assert result.authority_content_ids == ()
        assert result.transform_result_ids == ()


def test_legally_reframed_nested_missing_key_returns_batch_abstention(
    identity_batch: tuple[
        tuple[transform.PublicTransformEvidenceBundleV2, ...],
        batch_wire.TrustedWireBatchV1,
    ],
) -> None:
    authorities, original = identity_batch
    polluted = deepcopy(original)
    _, padding, decoded = _payload_parts(polluted.envelopes[0].envelope)
    authority = decoded["authority"]
    authority["base_bundle"]["observations"][0].pop("quantity_id")  # type: ignore[index]
    forged_envelope = _reframe(
        encode_phase2b_jcs_profile_v1(decoded),
        padding,
    )
    object.__setattr__(polluted.envelopes[0], "envelope", forged_envelope)
    result = typed_replay.replay_typed_trusted_wire_batch_v1(
        batch=polluted,
        run_id=RUN_ID,
        key_sources=_keys(),
        authorities=authorities,
    )
    assert type(result) is typed_replay.TypedTrustedWireBatchReplayRejectionV1
    assert result.reason == "batch_validation_failed"
    assert result.batch_id is None
    assert result.rows == ()
    assert result.source_authority_content_ids == ()
    assert result.authority_content_ids == ()
    assert result.transform_result_ids == ()


def test_receipts_reject_forgery_and_remain_frozen(
    complete_batch: tuple[
        tuple[transform.PublicTransformEvidenceBundleV2, ...],
        batch_wire.TrustedWireBatchV1,
    ],
    complete_typed_replay: typed_replay.TypedTrustedWireBatchReplayV1,
) -> None:
    authorities, batch = complete_batch
    row = typed_replay.decode_and_replay_typed_trusted_envelope_v1(
        batch.envelopes[0].envelope
    )

    class StrSpoof(str):
        pass

    with pytest.raises((TypeError, ValueError), match="SHA-256|receipt drift"):
        replace(row, payload_sha256=StrSpoof(row.payload_sha256))
    for mutation in (
        {"payload_sha256": "0" * 64},
        {
            "authority_content_id": (
                "phase2b_public_transform_evidence_" + "0" * 64
            )
        },
        {"transform_result_id": "phase2b_exact_transform_result_" + "0" * 64},
        {
            "replay_policy_id": (
                "phase2b_typed_trusted_wire_replay_policy_" + "0" * 64
            )
        },
        {
            "typed_authority_schema_id": (
                "phase2b_trusted_wire_typed_authority_schema_" + "0" * 64
            )
        },
    ):
        with pytest.raises(ValueError, match="receipt drift"):
            replace(row, **mutation)
    with pytest.raises(ValueError, match="claim boundary"):
        replace(row, origin_authenticated=True)
    with pytest.raises(TypeError, match="issued only"):
        typed_replay.TypedTrustedWireBatchReplayV1()
    receipt = complete_typed_replay
    with pytest.raises(FrozenInstanceError):
        receipt.batch_id = "forged"  # type: ignore[misc]

    reordered = deepcopy(receipt)
    object.__setattr__(
        reordered,
        "source_authorities",
        tuple(reversed(reordered.source_authorities)),
    )
    with pytest.raises(ValueError, match="source authority roots drift"):
        _ = reordered.receipt_id

    polluted_root = deepcopy(receipt)
    object.__setattr__(
        polluted_root,
        "source_authority_content_ids",
        ("phase2b_public_transform_evidence_" + "0" * 64,) * len(
            polluted_root.source_authorities
        ),
    )
    with pytest.raises(ValueError, match="source authority roots drift"):
        _ = polluted_root.receipt_id

    coherent_source_swap = deepcopy(receipt)
    object.__setattr__(
        coherent_source_swap,
        "source_authorities",
        tuple(reversed(coherent_source_swap.source_authorities)),
    )
    object.__setattr__(
        coherent_source_swap,
        "source_authority_content_ids",
        tuple(reversed(coherent_source_swap.source_authority_content_ids)),
    )
    with pytest.raises(ValueError, match="secret receipt binding drift"):
        _ = coherent_source_swap.receipt_id

    nested_source_pollution = deepcopy(receipt)
    object.__setattr__(
        nested_source_pollution.secret_replay_receipt,
        "source_authority_content_ids",
        tuple(
            reversed(
                nested_source_pollution.secret_replay_receipt.source_authority_content_ids
            )
        ),
    )
    with pytest.raises((TypeError, ValueError), match="replay|source|receipt"):
        _ = nested_source_pollution.receipt_id

    nested_id_pollution = deepcopy(receipt)
    object.__setattr__(
        nested_id_pollution,
        "secret_replay_receipt_id",
        "phase2b_trusted_wire_secret_replay_" + "0" * 64,
    )
    with pytest.raises(ValueError, match="secret receipt binding drift"):
        _ = nested_id_pollution.receipt_id

    wrong_root_container = deepcopy(receipt)
    object.__setattr__(
        wrong_root_container,
        "authority_content_ids",
        list(wrong_root_container.authority_content_ids),
    )
    with pytest.raises(TypeError, match="full tuples"):
        _ = wrong_root_container.receipt_id


def test_every_row_and_batch_claim_rejects_integer_bool_aliases(
    complete_typed_replay: typed_replay.TypedTrustedWireBatchReplayV1,
) -> None:
    row_claims = (
        "batch_policy_membership_verified",
        "secret_padding_replay_verified",
        "direct_payload_authority_transform_replay_verified",
        "lossless_typed_profile_roundtrip_verified",
        "origin_authenticated",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "c1_exit_evidence",
    )
    for field_name in row_claims:
        polluted = deepcopy(complete_typed_replay.rows[0])
        original = getattr(polluted, field_name)
        assert type(original) is bool
        object.__setattr__(polluted, field_name, int(original))
        with pytest.raises(TypeError, match="exact bool"):
            _ = polluted.row_id

    batch_claims = (
        "batch_policy_membership_verified",
        "whole_batch_atomic_typed_replay_verified",
        "secret_custodian_replay_verified",
        "direct_payload_authority_transform_replay_verified",
        "origin_authenticated",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "c1_exit_evidence",
    )
    for field_name in batch_claims:
        polluted = deepcopy(complete_typed_replay)
        original = getattr(polluted, field_name)
        assert type(original) is bool
        object.__setattr__(polluted, field_name, int(original))
        with pytest.raises(TypeError, match="exact bool"):
            _ = polluted.receipt_id


def test_polluted_secret_replay_receipt_abstains_without_partial_roots(
    identity_batch: tuple[
        tuple[transform.PublicTransformEvidenceBundleV2, ...],
        batch_wire.TrustedWireBatchV1,
    ],
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_authorities, batch = identity_batch
    secret = batch_wire.verify_trusted_wire_batch_replay_v1(
        batch=batch,
        run_id=RUN_ID,
        key_sources=_keys(),
        authorities=source_authorities,
    )
    mutations = (
        (
            "batch_id",
            "phase2b_trusted_wire_batch_" + "0" * 64,
        ),
        ("replay_verified", 1),
        ("origin_authenticated", 0),
    )
    for field_name, field_value in mutations:
        polluted = deepcopy(secret)
        object.__setattr__(polluted, field_name, field_value)

        def forged_secret_replay(**_: object) -> object:
            return polluted

        monkeypatch.setattr(
            typed_replay,
            "verify_trusted_wire_batch_replay_v1",
            forged_secret_replay,
        )
        result = typed_replay.replay_typed_trusted_wire_batch_v1(
            batch=batch,
            run_id=RUN_ID,
            key_sources=_keys(),
            authorities=source_authorities,
        )
        assert type(result) is typed_replay.TypedTrustedWireBatchReplayRejectionV1
        assert result.reason == "custodian_or_typed_replay_failed"
        assert result.batch_id == batch.batch_id
        assert result.rows == ()
        assert result.source_authority_content_ids == ()
        assert result.authority_content_ids == ()
        assert result.transform_result_ids == ()

    coherent = deepcopy(secret)
    alternate_roots = (authorities["unit_authority"].content_id,)
    object.__setattr__(
        coherent,
        "source_authority_content_ids",
        alternate_roots,
    )
    object.__setattr__(
        coherent,
        "replay_receipt_id",
        _coherent_secret_replay_id(coherent, alternate_roots),
    )
    assert coherent.replay_receipt_id == _coherent_secret_replay_id(
        coherent,
        alternate_roots,
    )

    def forged_coherent_secret_replay(**_: object) -> object:
        return coherent

    monkeypatch.setattr(
        typed_replay,
        "verify_trusted_wire_batch_replay_v1",
        forged_coherent_secret_replay,
    )
    result = typed_replay.replay_typed_trusted_wire_batch_v1(
        batch=batch,
        run_id=RUN_ID,
        key_sources=_keys(),
        authorities=source_authorities,
    )
    assert type(result) is typed_replay.TypedTrustedWireBatchReplayRejectionV1
    assert result.reason == "custodian_or_typed_replay_failed"
    assert result.batch_id == batch.batch_id
    assert result.rows == ()
    assert result.source_authority_content_ids == ()
    assert result.authority_content_ids == ()
    assert result.transform_result_ids == ()
