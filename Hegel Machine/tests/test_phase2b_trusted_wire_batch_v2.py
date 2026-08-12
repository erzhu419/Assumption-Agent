"""Adversarial tests for the compact-authority V2 trusted-wire batch.

The V2 batch is non-authoritative representation and replay mechanics.  These
tests keep its narrow structural/typed/direct-transform checks separate from
origin, formal/covert audit, sealed-holdout, recognizer-effect, and capacity
evidence, none of which this slice establishes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from enum import Enum
from fractions import Fraction
import hashlib
import inspect
import json
from pathlib import Path
import re
import runpy
import struct
from typing import Callable

import pytest

import hegel_machine.phase2b_exact_derived_witness_bridge_v1 as derived_bridge
import hegel_machine.phase2b_exact_transform_semantics_v1 as transform
import hegel_machine.phase2b_trusted_wire_batch_v1 as batch_v1
import hegel_machine.phase2b_trusted_wire_batch_v2 as batch_v2
import hegel_machine.phase2b_trusted_wire_typed_authority_v1 as typed_v1
import hegel_machine.phase2b_trusted_wire_typed_authority_v2 as typed_v2
import hegel_machine.phase2b_trusted_wire_v1 as wire


RUN_ID = b"R" * 32
HEADER = struct.Struct(">8sHHI32s32s")
UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
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
ROOT = Path(__file__).resolve().parents[1]


def _keys(
    *,
    shuffle: bytes = b"S" * 32,
    identifiers: bytes = b"I" * 32,
    padding: bytes = b"P" * 32,
) -> batch_v2.TrustedWireKeySourcesV2:
    return batch_v2.TrustedWireKeySourcesV2(shuffle, identifiers, padding)


def _v1_keys() -> batch_v1.TrustedWireKeySourcesV1:
    return batch_v1.TrustedWireKeySourcesV1(b"S" * 32, b"I" * 32, b"P" * 32)


@pytest.fixture(scope="module")
def authorities() -> dict[str, transform.PublicTransformEvidenceBundleV2]:
    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_exact_transform_semantics_v1.py"))
    )
    return {name: namespace[name]() for name in FACTORY_NAMES}


@pytest.fixture(scope="module")
def v1_identity_authority() -> transform.PublicTransformEvidenceBundleV2:
    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_trusted_wire_batch_v1.py"))
    )
    fixture = namespace["authorities"].__wrapped__()
    return fixture["identity"]


@pytest.fixture(scope="module")
def positive_fixture() -> object:
    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_recognizer_prediction_archive_v1.py"))
    )
    return namespace["public_positive_mechanics_fixture"].__wrapped__()


@pytest.fixture(scope="module")
def stage_a_sampling_authority() -> transform.PublicTransformEvidenceBundleV2:
    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_trusted_wire_typed_replay_v1.py"))
    )
    return namespace["stage_a_sampling_authority"].__wrapped__()


@pytest.fixture(scope="module")
def all_transforms_batch(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> batch_v2.TrustedWireBatchV2:
    result = batch_v2.build_trusted_wire_batch_v2(
        authorities=tuple(authorities[name] for name in FACTORY_NAMES),
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(result) is batch_v2.TrustedWireBatchV2
    assert result.disposition is batch_v2.TrustedWireBatchDispositionV2.COMPLETE
    return result


@pytest.fixture(scope="module")
def v1_all_transforms_batch(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> batch_v1.TrustedWireBatchV1:
    result = batch_v1.build_trusted_wire_batch_v1(
        authorities=tuple(authorities[name] for name in FACTORY_NAMES),
        run_id=RUN_ID,
        key_sources=_v1_keys(),
    )
    assert type(result) is batch_v1.TrustedWireBatchV1
    return result


@pytest.fixture(scope="module")
def positive_batch(
    positive_fixture: object,
) -> batch_v2.TrustedWireBatchV2:
    result = batch_v2.build_trusted_wire_batch_v2(
        authorities=(positive_fixture.source_authority,),
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(result) is batch_v2.TrustedWireBatchV2
    return result


@pytest.fixture(scope="module")
def one_row_batch(
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
) -> batch_v2.TrustedWireBatchV2:
    return batch_v2.TrustedWireBatchV2._issue(
        batch_v2._BATCH_ISSUE_TOKEN_V2,
        run_id_commitment=all_transforms_batch.run_id_commitment,
        envelopes=(all_transforms_batch.envelopes[0],),
        uuid_collision_retry_count=all_transforms_batch.uuid_collision_retry_count,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _unchecked_copy(value: object, **changes: object) -> object:
    copied = object.__new__(type(value))
    for item in fields(value):
        object.__setattr__(
            copied,
            item.name,
            changes.get(item.name, getattr(value, item.name)),
        )
    return copied


def _parts(
    envelope: bytes,
    *,
    magic: bytes,
    version: int,
) -> tuple[bytes, bytes]:
    assert type(envelope) is bytes and len(envelope) == wire.ENVELOPE_BYTES
    observed_magic, observed_version, header_bytes, payload_bytes, ph, dh = (
        HEADER.unpack(envelope[: HEADER.size])
    )
    assert observed_magic == magic
    assert observed_version == version
    assert header_bytes == HEADER.size == wire.ENVELOPE_HEADER_BYTES
    payload = envelope[header_bytes : header_bytes + payload_bytes]
    padding = envelope[header_bytes + payload_bytes :]
    assert hashlib.sha256(payload).digest() == ph
    assert hashlib.sha256(padding).digest() == dh
    return payload, padding


def _v2_payload(envelope: bytes) -> tuple[bytes, bytes, dict[str, object]]:
    payload, padding = _parts(
        envelope,
        magic=batch_v2.TRUSTED_WIRE_ENVELOPE_V2_MAGIC,
        version=batch_v2.TRUSTED_WIRE_ENVELOPE_V2_VERSION,
    )
    mapping = batch_v2._decode_payload_jcs_v2(payload)
    assert type(mapping) is dict
    return payload, padding, mapping


def _reframe_v2(payload: bytes, seed_padding: bytes) -> bytes:
    padding_bytes = wire.ENVELOPE_BYTES - HEADER.size - len(payload)
    if padding_bytes < 0:
        padding = b""
    else:
        padding = (seed_padding + b"\x00" * padding_bytes)[:padding_bytes]
    return HEADER.pack(
        batch_v2.TRUSTED_WIRE_ENVELOPE_V2_MAGIC,
        batch_v2.TRUSTED_WIRE_ENVELOPE_V2_VERSION,
        HEADER.size,
        len(payload),
        hashlib.sha256(payload).digest(),
        hashlib.sha256(padding).digest(),
    ) + payload + padding


def _uuid_values(value: object) -> set[str]:
    result: set[str] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if type(current) is dict:
            stack.extend(current.keys())
            stack.extend(current.values())
        elif type(current) in (list, tuple):
            stack.extend(current)
        elif type(current) is str and UUID4.fullmatch(current):
            result.add(current)
    return result


def _payload_mapping_for_authority(
    authority: transform.PublicTransformEvidenceBundleV2,
) -> dict[str, object]:
    return {
        "authority": typed_v2.encode_typed_transform_authority_profile_v2(
            authority
        ),
        "field_manifest_id": wire.FIELD_MANIFEST_ID,
        "jcs_profile_id": wire.JCS_PROFILE_ID,
        "public_provenance_version": batch_v1.TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION,
        "schema_version": batch_v2.TRUSTED_WIRE_BATCH_V2_PAYLOAD_SCHEMA_VERSION,
        "typed_authority_codec_policy_id": (
            typed_v2.COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID
        ),
        "typed_authority_codec_version": (
            typed_v2.COMPACT_TYPED_AUTHORITY_CODEC_VERSION
        ),
        "typed_authority_schema_id": typed_v2.COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
    }


def _authority_with_walker_cap_drift(
    source: transform.PublicTransformEvidenceBundleV2,
    mutation: str,
) -> transform.PublicTransformEvidenceBundleV2:
    base = source.base_bundle
    if mutation == "entries":
        base = _unchecked_copy(
            base,
            role_ids=("role",) * wire.MAXIMUM_ARRAY_ENTRIES,
            quantity_ids=("quantity",) * wire.MAXIMUM_ARRAY_ENTRIES,
            transform_catalog=("catalog",) * wire.MAXIMUM_ARRAY_ENTRIES,
            missingness_mask=("mask",) * wire.MAXIMUM_ARRAY_ENTRIES,
        )
        return _unchecked_copy(source, base_bundle=base)  # type: ignore[return-value]
    if mutation == "total_string_bytes":
        text = "x" * wire.MAXIMUM_ASCII_STRING_BYTES
        base = _unchecked_copy(
            base,
            role_ids=(text,) * (wire.MAXIMUM_PROFILE_NODES // 8 + 1),
        )
        return _unchecked_copy(source, base_bundle=base)  # type: ignore[return-value]
    if mutation == "uuid_occurrences":
        value = "10000000-0000-4000-8000-000000000001"
        base = _unchecked_copy(
            base,
            role_ids=(value,) * (wire.MAXIMUM_UUID_OCCURRENCES + 1),
        )
        return _unchecked_copy(source, base_bundle=base)  # type: ignore[return-value]
    if mutation == "unique_uuids":
        values = tuple(
            f"10000000-0000-4000-8000-{index:012x}"
            for index in range(wire.MAXIMUM_UNIQUE_UUIDS + 1)
        )
        base = _unchecked_copy(base, role_ids=values)
        return _unchecked_copy(source, base_bundle=base)  # type: ignore[return-value]
    if mutation == "safe_integer":
        contract = source.transform_contracts[0]
        ref = _unchecked_copy(
            contract.input_components[0],
            ordinal=wire.MAXIMUM_SAFE_INTEGER + 1,
        )
        contract = _unchecked_copy(
            contract,
            input_components=(ref, *contract.input_components[1:]),
        )
        return _unchecked_copy(  # type: ignore[return-value]
            source,
            transform_contracts=(contract, *source.transform_contracts[1:]),
        )
    if mutation == "finite_float":
        observation = base.observations[0]
        uncertainty = _unchecked_copy(
            observation.uncertainty,
            radius=(float("inf"), observation.uncertainty.radius[1]),
        )
        observation = _unchecked_copy(observation, uncertainty=uncertainty)
        base = _unchecked_copy(
            base,
            observations=(observation, *base.observations[1:]),
        )
        return _unchecked_copy(source, base_bundle=base)  # type: ignore[return-value]
    if mutation == "rational_bits":
        contract = source.transform_contracts[0]
        row = contract.kernel_rows[0]
        term = row.terms[0]
        atom = _unchecked_copy(
            term.coefficient,
            numerator=1 << wire.MAXIMUM_RATIONAL_BIT_LENGTH,
        )
        term = _unchecked_copy(term, coefficient=atom)
        row = _unchecked_copy(row, terms=(term, *row.terms[1:]))
        contract = _unchecked_copy(
            contract,
            kernel_rows=(row, *contract.kernel_rows[1:]),
        )
        return _unchecked_copy(  # type: ignore[return-value]
            source,
            transform_contracts=(contract, *source.transform_contracts[1:]),
        )
    raise AssertionError(f"unknown walker mutation {mutation}")


def _positive_payload_mapping(positive_fixture: object) -> dict[str, object]:
    return _payload_mapping_for_authority(positive_fixture.public_authority)


def _v1_payload(envelope: bytes) -> dict[str, object]:
    payload, _ = _parts(envelope, magic=wire.ENVELOPE_MAGIC, version=1)
    mapping = wire.decode_phase2b_jcs_profile_v1(payload)
    assert type(mapping) is dict
    return mapping


def test_v1_identity_batch_and_envelope_cryptographic_vector_is_byte_exact(
    v1_identity_authority: transform.PublicTransformEvidenceBundleV2,
) -> None:
    result = batch_v1.build_trusted_wire_batch_v1(
        authorities=(v1_identity_authority,),
        run_id=RUN_ID,
        key_sources=_v1_keys(),
    )
    assert type(result) is batch_v1.TrustedWireBatchV1
    row = result.envelopes[0]
    assert result.batch_id == (
        "phase2b_trusted_wire_batch_"
        "0b558bbe6484e75635909e1a4bbd914db5fa58e5d77c9a7b62ec4a883f1e1d2b"
    )
    assert row.envelope_id == (
        "phase2b_trusted_envelope_"
        "9008e222d08bcf21aa70942db5b06c40dbed36994da9324f88ea094df9d56e4c"
    )
    assert row.payload_sha256 == (
        "01a1c2bd289abcfd263c3f6dacf38d47d8e049d79efd897ee8be1fea1e36a61f"
    )
    assert row.padding_sha256 == (
        "5f37627a6f9d416c95e03f4264a8354535ed58474d8dbb0b5f2f88aafe792214"
    )
    assert hashlib.sha256(row.envelope).hexdigest() == (
        "8c8166d7d0aaebdfb2ad1b50e63f410de75dca1c1e7721178224cf9d209b8b52"
    )
    assert len(row.envelope) == 65_536


def test_v2_full_wrapper_fields_and_positive_mechanics_fit_the_envelope(
    positive_fixture: object,
) -> None:
    mapping = _positive_payload_mapping(positive_fixture)
    assert tuple(sorted(mapping)) == batch_v2.TRUSTED_WIRE_BATCH_V2_PAYLOAD_FIELDS
    payload = batch_v2._encode_payload_jcs_v2(mapping)
    assert payload == _canonical(mapping)
    assert len(payload) == 50_255
    assert len(payload) <= wire.MAXIMUM_PAYLOAD_BYTES == 65_424
    assert wire.MAXIMUM_PAYLOAD_BYTES - len(payload) == 15_169
    assert len(mapping["authority"]["strings"]) == 397  # type: ignore[index]


def test_v2_framing_accepts_65424_and_rejects_65425_only_at_full_wrapper() -> None:
    exact = b"x" * wire.MAXIMUM_PAYLOAD_BYTES
    envelope = batch_v2._frame_secret_payload_v2(exact, b"K" * 32, RUN_ID)
    payload, padding = _parts(
        envelope,
        magic=batch_v2.TRUSTED_WIRE_ENVELOPE_V2_MAGIC,
        version=batch_v2.TRUSTED_WIRE_ENVELOPE_V2_VERSION,
    )
    assert payload == exact
    assert len(padding) == wire.MINIMUM_PADDING_BYTES == 32
    with pytest.raises(ValueError, match="fixed-envelope capacity"):
        batch_v2._frame_secret_payload_v2(exact + b"x", b"K" * 32, RUN_ID)


def test_public_api_is_closed_and_uses_exact_v2_key_custody() -> None:
    build = inspect.signature(batch_v2.build_trusted_wire_batch_v2)
    assert tuple(build.parameters) == ("authorities", "run_id", "key_sources")
    assert all(
        item.kind is inspect.Parameter.KEYWORD_ONLY
        for item in build.parameters.values()
    )
    assert tuple(
        inspect.signature(
            batch_v2.decode_and_audit_trusted_envelope_v2
        ).parameters
    ) == ("envelope",)
    assert set(batch_v2.__all__) == {
        "DEFAULT_TRUSTED_WIRE_BATCH_V2_POLICY",
        "DecodedTrustedEnvelopeV2",
        "IKM_BYTES",
        "MAXIMUM_BATCH_V2_AUTHORITIES",
        "MAXIMUM_BATCH_V2_COLLISION_RETRY_COUNT",
        "MAXIMUM_SHUFFLE_REJECTION_DRAWS",
        "MAXIMUM_UUID_COLLISION_RETRIES",
        "RUN_ID_BYTES",
        "TRUSTED_WIRE_BATCH_V2_PAYLOAD_FIELDS",
        "TRUSTED_WIRE_BATCH_V2_PAYLOAD_SCHEMA_VERSION",
        "TRUSTED_WIRE_BATCH_V2_POLICY_ID",
        "TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION",
        "TRUSTED_WIRE_ENVELOPE_V2_MAGIC",
        "TRUSTED_WIRE_ENVELOPE_V2_VERSION",
        "TrustedWireBatchDispositionV2",
        "TrustedWireBatchPolicyV2",
        "TrustedWireBatchRejectionV2",
        "TrustedWireBatchV2",
        "TrustedWireEnvelopeV2",
        "TrustedWireKeySourcesV2",
        "build_trusted_wire_batch_v2",
        "decode_and_audit_trusted_envelope_v2",
    }
    for forbidden in ("receipt", "policy", "permutation", "root"):
        assert forbidden not in build.parameters
    assert not any("replay_receipt" in name.casefold() for name in batch_v2.__all__)


def test_public_dataclass_field_and_claim_manifests_are_exact() -> None:
    assert tuple(item.name for item in fields(batch_v2.TrustedWireBatchRejectionV2)) == (
        "disposition",
        "reason",
        "authority_count",
        "schema_version",
        "policy_id",
        "envelopes",
        "envelope_ids",
        "authority_content_ids",
        "transform_result_ids",
        "claim_level",
        "recognizer_capacity_evidence",
        "origin_authenticated",
        "formal_uuid_audit",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "c1_exit_evidence",
    )
    assert tuple(item.name for item in fields(batch_v2.DecodedTrustedEnvelopeV2)) == (
        "envelope",
        "envelope_id",
        "payload_sha256",
        "padding_sha256",
        "payload_bytes",
        "padding_bytes",
        "namespace_audit",
        "namespace_audit_id",
        "authority",
        "authority_content_id",
        "transform_result_id",
        "payload_schema_version",
        "public_provenance_version",
        "typed_authority_schema_id",
        "typed_authority_codec_version",
        "typed_authority_codec_policy_id",
        "structural_hashes_verified",
        "public_provenance_verified",
        "typed_authority_decode_replay_verified",
        "direct_exact_transform_replay_verified",
        "secret_padding_replay_verified",
        "origin_authenticated",
        "formal_uuid_audit",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "c1_exit_evidence",
    )
    assert tuple(item.name for item in fields(batch_v2.TrustedWireEnvelopeV2)) == (
        "envelope",
        "envelope_id",
        "payload_sha256",
        "padding_sha256",
        "payload_bytes",
        "padding_bytes",
        "namespace_audit_id",
        "authority_content_id",
        "transform_result_id",
        "typed_authority_schema_id",
        "typed_authority_codec_version",
        "typed_authority_codec_policy_id",
        "structural_hashes_verified",
        "public_provenance_verified",
        "typed_authority_decode_replay_verified",
        "direct_exact_transform_replay_verified",
        "secret_padding_replay_verified",
        "origin_authenticated",
        "formal_uuid_audit",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "c1_exit_evidence",
    )
    assert tuple(item.name for item in fields(batch_v2.TrustedWireBatchV2)) == (
        "disposition",
        "schema_version",
        "payload_schema_version",
        "key_schedule_version",
        "public_provenance_version",
        "typed_authority_schema_id",
        "typed_authority_codec_version",
        "typed_authority_codec_policy_id",
        "exact_transform_provenance_compiler_policy_id",
        "jcs_profile_id",
        "field_manifest_id",
        "policy_id",
        "run_id_commitment",
        "envelopes",
        "envelope_ids",
        "authority_content_ids",
        "transform_result_ids",
        "batch_id",
        "uuid_collision_retry_count",
        "uuid_collision_warning",
        "claim_level",
        "whole_batch_shuffle_publicly_verified",
        "purpose_separated_keys_publicly_verified",
        "post_shuffle_hmac_uuidv4_publicly_verified",
        "secret_hmac_padding_publicly_verified",
        "atomic_complete_batch_structure_verified",
        "typed_authority_decode_replay_verified",
        "direct_exact_transform_replay_verified",
        "recognizer_capacity_evidence",
        "origin_authenticated",
        "formal_uuid_audit",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "c1_exit_evidence",
    )


def test_policy_exact_scalar_contract_and_claim_manifest_precede_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = batch_v2.DEFAULT_TRUSTED_WIRE_BATCH_V2_POLICY
    assert batch_v2.TRUSTED_WIRE_BATCH_V2_POLICY_ID == (
        "phase2b_trusted_wire_batch_v2_policy_"
        "be9672f8efb5867075b27b0342818c9caa97fe434f3bb76f84c612194da5b0e8"
    )

    class StringSubclass(str):
        pass

    class IntegerSubclass(int):
        pass

    for item in fields(policy):
        value = getattr(policy, item.name)
        polluted = (
            StringSubclass(value)
            if type(value) is str
            else IntegerSubclass(value)
        )
        with pytest.raises((TypeError, ValueError)):
            replace(policy, **{item.name: polluted})

    copied = object.__new__(batch_v2.TrustedWireBatchPolicyV2)
    for item in fields(policy):
        object.__setattr__(copied, item.name, getattr(policy, item.name))
    object.__setattr__(
        copied,
        "schema_version",
        StringSubclass(policy.schema_version),
    )

    def forbidden_hash(*args: object, **kwargs: object) -> object:
        raise AssertionError("polluted policy reached hashing")

    monkeypatch.setattr(batch_v2.hashlib, "sha256", forbidden_hash)
    with pytest.raises((TypeError, ValueError)):
        copied.__post_init__()


def test_policy_claim_preimage_lists_every_narrow_and_broad_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []
    original = batch_v2.encode_phase2b_jcs_profile_v1

    def capture(value: object) -> bytes:
        captured.append(value)
        return original(value)

    monkeypatch.setattr(batch_v2, "encode_phase2b_jcs_profile_v1", capture)
    assert batch_v2.DEFAULT_TRUSTED_WIRE_BATCH_V2_POLICY.policy_id.startswith(
        "phase2b_trusted_wire_batch_v2_policy_"
    )
    assert len(captured) == 1 and type(captured[0]) is dict
    claims = captured[0]["claim_contract"]  # type: ignore[index]
    assert claims["envelope_true"] == [  # type: ignore[index]
        "structural_hashes_verified",
        "public_provenance_verified",
        "typed_authority_decode_replay_verified",
        "direct_exact_transform_replay_verified",
    ]
    assert claims["envelope_false"] == [  # type: ignore[index]
        "secret_padding_replay_verified",
        "origin_authenticated",
        "formal_uuid_audit",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "c1_exit_evidence",
    ]
    assert claims["envelope_claims_apply_to"] == [  # type: ignore[index]
        "DecodedTrustedEnvelopeV2",
        "TrustedWireEnvelopeV2",
    ]
    assert claims["rejection_false"] == [  # type: ignore[index]
        "recognizer_capacity_evidence",
        "origin_authenticated",
        "formal_uuid_audit",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "c1_exit_evidence",
    ]
    assert claims["batch_true"] == [  # type: ignore[index]
        "atomic_complete_batch_structure_verified",
        "typed_authority_decode_replay_verified",
        "direct_exact_transform_replay_verified",
    ]
    assert claims["batch_false"] == [  # type: ignore[index]
        "whole_batch_shuffle_publicly_verified",
        "purpose_separated_keys_publicly_verified",
        "post_shuffle_hmac_uuidv4_publicly_verified",
        "secret_hmac_padding_publicly_verified",
        "recognizer_capacity_evidence",
        "origin_authenticated",
        "formal_uuid_audit",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "c1_exit_evidence",
    ]
    assert claims["claim_level_literal"] == wire.NON_AUTHORITATIVE_CLAIM_LEVEL  # type: ignore[index]
    assert claims["exact_bool"] is True  # type: ignore[index]
    manifests = captured[0]["public_dataclass_field_manifests"]  # type: ignore[index]
    for key, cls in (
        ("key_sources", batch_v2.TrustedWireKeySourcesV2),
        ("policy", batch_v2.TrustedWireBatchPolicyV2),
        ("rejection", batch_v2.TrustedWireBatchRejectionV2),
        ("decoded_envelope", batch_v2.DecodedTrustedEnvelopeV2),
        ("issued_envelope", batch_v2.TrustedWireEnvelopeV2),
        ("batch", batch_v2.TrustedWireBatchV2),
    ):
        assert manifests[key] == [item.name for item in fields(cls)]  # type: ignore[index]


@pytest.mark.parametrize(
    "mutation",
    (
        "nested_string_subclass",
        "hostile_tuple_length",
        "nested_dataclass_subclass",
        "wrong_enum_class",
    ),
)
def test_source_nested_exact_type_closure_precedes_deep_validation_and_hashing(
    mutation: str,
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StringSubclass(str):
        pass

    class HostileTuple(tuple):
        def __len__(self) -> int:
            raise AssertionError("non-exact tuple reached length inspection")

    class OperationSpoof(str, Enum):
        IDENTITY = "identity"

    source = authorities["identity_authority"]
    if mutation == "nested_string_subclass":
        base = _unchecked_copy(
            source.base_bundle,
            schema_version=StringSubclass(source.base_bundle.schema_version),
        )
        polluted = _unchecked_copy(source, base_bundle=base)
    elif mutation == "hostile_tuple_length":
        base = _unchecked_copy(
            source.base_bundle,
            observations=HostileTuple(source.base_bundle.observations),
        )
        polluted = _unchecked_copy(source, base_bundle=base)
    elif mutation == "nested_dataclass_subclass":
        class BaseBundleSubclass(type(source.base_bundle)):
            pass

        base = object.__new__(BaseBundleSubclass)
        for item in fields(source.base_bundle):
            object.__setattr__(
                base,
                item.name,
                getattr(source.base_bundle, item.name),
            )
        polluted = _unchecked_copy(source, base_bundle=base)
    else:
        contract = _unchecked_copy(
            source.transform_contracts[0],
            operation=OperationSpoof.IDENTITY,
        )
        polluted = _unchecked_copy(source, transform_contracts=(contract,))

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("polluted source reached deep validation or hashing")

    monkeypatch.setattr(
        batch_v1,
        "_hash_free_authority_preflight",
        forbidden,
    )
    monkeypatch.setattr(
        batch_v2,
        "encode_typed_transform_authority_profile_v1",
        forbidden,
    )
    monkeypatch.setattr(batch_v2, "run_exact_transform_semantics", forbidden)
    monkeypatch.setattr(batch_v2.hashlib, "sha256", forbidden)
    result = batch_v2.build_trusted_wire_batch_v2(
        authorities=(polluted,),  # type: ignore[arg-type]
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(result) is batch_v2.TrustedWireBatchRejectionV2
    assert result.envelopes == result.envelope_ids == ()
    assert result.authority_content_ids == result.transform_result_ids == ()


@pytest.mark.parametrize(
    ("mutation", "reason_fragment"),
    (
        ("entries", "entry"),
        ("total_string_bytes", "string"),
        ("uuid_occurrences", "uuid_occurrence"),
        ("unique_uuids", "unique_uuid"),
        ("safe_integer", "safe_integer"),
        ("finite_float", "nonfinite"),
        ("rational_bits", "rational_bit"),
    ),
)
def test_source_walker_all_resource_caps_reject_before_v1_helpers_and_hashing(
    mutation: str,
    reason_fragment: str,
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polluted = _authority_with_walker_cap_drift(
        authorities["identity_authority"],
        mutation,
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("source cap drift reached V1 helper or hashing")

    monkeypatch.setattr(
        batch_v1,
        "_hash_free_authority_preflight",
        forbidden,
    )
    monkeypatch.setattr(
        batch_v2,
        "encode_typed_transform_authority_profile_v1",
        forbidden,
    )
    monkeypatch.setattr(batch_v2, "run_exact_transform_semantics", forbidden)
    monkeypatch.setattr(batch_v2.hashlib, "sha256", forbidden)
    result = batch_v2.build_trusted_wire_batch_v2(
        authorities=(polluted,),
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(result) is batch_v2.TrustedWireBatchRejectionV2
    assert reason_fragment in result.reason
    assert result.envelopes == result.envelope_ids == ()
    assert result.authority_content_ids == result.transform_result_ids == ()


def test_namespace_audit_occurrence_cap_precedes_nested_validation_sort_and_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = object.__new__(wire.NamespaceFieldAuditV1)
    object.__setattr__(audit, "manifest_id", wire.FIELD_MANIFEST_ID)
    object.__setattr__(audit, "frozen_minimum_namespaces", tuple(range(10)))
    object.__setattr__(audit, "schema_registry_namespaces", tuple(range(16)))
    object.__setattr__(audit, "zero_occurrence_namespaces", tuple(range(2)))
    object.__setattr__(
        audit,
        "occurrences",
        (object(),) * (wire.MAXIMUM_UUID_OCCURRENCES + 1),
    )
    object.__setattr__(audit, "claim_level", wire.NON_AUTHORITATIVE_CLAIM_LEVEL)
    object.__setattr__(audit, "formal_uuid_namespace_field_audit", False)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("oversized audit reached nested validation/sort/hash")

    monkeypatch.setattr(wire.NamespaceFieldAuditV1, "__post_init__", forbidden)
    monkeypatch.setattr(batch_v2.hashlib, "sha256", forbidden)
    with pytest.raises(ValueError, match="occurrence cap"):
        batch_v2._namespace_audit_id_v2(audit)


def test_namespace_occurrence_deep_validation_precedes_audit_sort_and_hash(
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded = batch_v2.decode_and_audit_trusted_envelope_v2(
        all_transforms_batch.envelopes[0].envelope
    )
    valid = decoded.namespace_audit.occurrences[0]
    polluted = object.__new__(wire.NamespaceOccurrenceV1)
    for item in fields(valid):
        object.__setattr__(polluted, item.name, getattr(valid, item.name))
    object.__setattr__(polluted, "json_pointer", "\ud800" * 2_049)

    audit = object.__new__(wire.NamespaceFieldAuditV1)
    for item in fields(decoded.namespace_audit):
        object.__setattr__(
            audit,
            item.name,
            (polluted,) * wire.MAXIMUM_UUID_OCCURRENCES
            if item.name == "occurrences"
            else getattr(decoded.namespace_audit, item.name),
        )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid occurrence reached audit sort or hash")

    monkeypatch.setattr(
        wire.NamespaceOccurrenceV1,
        "__post_init__",
        forbidden,
    )
    monkeypatch.setattr(wire.NamespaceFieldAuditV1, "__post_init__", forbidden)
    monkeypatch.setattr(batch_v2.hashlib, "sha256", forbidden)
    with pytest.raises(ValueError, match="string cap"):
        batch_v2._namespace_audit_id_v2(audit)


@pytest.mark.parametrize(
    "mutation",
    (
        "manifest_subclass",
        "claim_subclass",
        "formal_integer",
        "frozen_item_subclass",
        "registry_item_subclass",
        "zero_item_subclass",
        "occurrence_string_subclass",
    ),
)
def test_namespace_audit_exact_scalars_and_nested_strings_precede_v1_validation(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StringSubclass(str):
        pass

    audit = object.__new__(wire.NamespaceFieldAuditV1)
    object.__setattr__(audit, "manifest_id", wire.FIELD_MANIFEST_ID)
    object.__setattr__(audit, "frozen_minimum_namespaces", ("f",) * 10)
    object.__setattr__(audit, "schema_registry_namespaces", ("r",) * 16)
    object.__setattr__(audit, "zero_occurrence_namespaces", ("z",) * 2)
    object.__setattr__(audit, "occurrences", ())
    object.__setattr__(audit, "claim_level", wire.NON_AUTHORITATIVE_CLAIM_LEVEL)
    object.__setattr__(audit, "formal_uuid_namespace_field_audit", False)
    if mutation == "manifest_subclass":
        object.__setattr__(audit, "manifest_id", StringSubclass(wire.FIELD_MANIFEST_ID))
    elif mutation == "claim_subclass":
        object.__setattr__(
            audit,
            "claim_level",
            StringSubclass(wire.NON_AUTHORITATIVE_CLAIM_LEVEL),
        )
    elif mutation == "formal_integer":
        object.__setattr__(audit, "formal_uuid_namespace_field_audit", 0)
    elif mutation.endswith("item_subclass"):
        name = {
            "frozen_item_subclass": "frozen_minimum_namespaces",
            "registry_item_subclass": "schema_registry_namespaces",
            "zero_item_subclass": "zero_occurrence_namespaces",
        }[mutation]
        values = list(getattr(audit, name))
        values[0] = StringSubclass(values[0])
        object.__setattr__(audit, name, tuple(values))
    else:
        occurrence = object.__new__(wire.NamespaceOccurrenceV1)
        object.__setattr__(occurrence, "namespace", StringSubclass("bundle"))
        object.__setattr__(occurrence, "json_pointer", "/base_bundle/bundle_id")
        object.__setattr__(
            occurrence,
            "public_uuid",
            "00000000-0000-4000-8000-000000000001",
        )
        object.__setattr__(occurrence, "rule_id", "rule")
        object.__setattr__(audit, "occurrences", (occurrence,))

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("non-exact audit reached V1 validation or hash")

    monkeypatch.setattr(wire.NamespaceFieldAuditV1, "__post_init__", forbidden)
    monkeypatch.setattr(wire.NamespaceOccurrenceV1, "__post_init__", forbidden)
    monkeypatch.setattr(batch_v2.hashlib, "sha256", forbidden)
    with pytest.raises((TypeError, ValueError)):
        batch_v2._namespace_audit_id_v2(audit)


def test_payload_root_key_exact_type_precedes_sort_and_authority_decode(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _payload_mapping_for_authority(authorities["identity_authority"])

    class EvilString(str):
        def __lt__(self, other: object) -> bool:
            raise AssertionError("non-exact payload key reached sorting")

    polluted = {
        (EvilString(key) if key == "authority" else key): value
        for key, value in mapping.items()
    }

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("non-exact payload key reached authority decode")

    monkeypatch.setattr(
        batch_v2._typed_v2,
        "_validate_compact_profile",
        forbidden,
    )
    with pytest.raises(TypeError, match="key|exact"):
        batch_v2._encode_payload_jcs_v2(polluted)


@pytest.mark.parametrize(
    ("padding_key", "run_id"),
    (
        (bytearray(b"K" * 32), RUN_ID),
        (b"K" * 32, bytearray(RUN_ID)),
        (b"K" * 31, RUN_ID),
        (b"K" * 32, RUN_ID[:-1]),
    ),
)
def test_frame_validates_exact_key_and_run_before_payload_hash(
    padding_key: object,
    run_id: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid frame custody reached payload hashing")

    monkeypatch.setattr(batch_v2.hashlib, "sha256", forbidden)
    with pytest.raises((TypeError, ValueError), match="padding|run|32"):
        batch_v2._frame_secret_payload_v2(  # type: ignore[arg-type]
            b"valid payload bytes",
            padding_key,
            run_id,
        )


def test_all_eight_certificates_and_two_step_build_decode_and_direct_complete(
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
) -> None:
    assert len(all_transforms_batch.envelopes) == len(FACTORY_NAMES) == 9
    operations: list[transform.TransformOperation] = []
    contract_counts: list[int] = []
    for row in all_transforms_batch.envelopes:
        assert len(row.envelope) == wire.ENVELOPE_BYTES == 65_536
        decoded = batch_v2.decode_and_audit_trusted_envelope_v2(row.envelope)
        assert decoded.authority_content_id == row.authority_content_id
        assert decoded.transform_result_id == row.transform_result_id
        assert decoded.typed_authority_decode_replay_verified
        assert decoded.direct_exact_transform_replay_verified
        compact = _v2_payload(row.envelope)[2]["authority"]
        assert type(compact) is dict
        assert typed_v2.encode_typed_transform_authority_profile_v2(
            decoded.authority
        ) == compact
        replay = transform.run_exact_transform_semantics(decoded.authority)
        assert type(replay) is transform.ExactTransformCompilation
        assert replay.disposition is transform.TransformCompilationDisposition.COMPLETE
        assert replay.result_id == row.transform_result_id
        operations.extend(
            contract.operation for contract in decoded.authority.transform_contracts
        )
        contract_counts.append(len(decoded.authority.transform_contracts))
    assert set(operations) == set(transform.TransformOperation)
    assert contract_counts.count(2) == 1
    assert contract_counts.count(1) == 8


def test_v1_and_v2_share_shuffle_and_live_uuid_allocation_but_not_frames(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
    v1_all_transforms_batch: batch_v1.TrustedWireBatchV1,
) -> None:
    assert len(all_transforms_batch.envelopes) == len(
        v1_all_transforms_batch.envelopes
    )
    all_public_ids: set[str] = set()
    all_source_ids = set().union(
        *(
            _uuid_values(typed_v1.encode_typed_transform_authority_profile_v1(
                authorities[name]
            ))
            for name in FACTORY_NAMES
        )
    )
    assert all_source_ids
    for v1_row, v2_row in zip(
        v1_all_transforms_batch.envelopes,
        all_transforms_batch.envelopes,
        strict=True,
    ):
        v1_mapping = _v1_payload(v1_row.envelope)["authority"]
        assert type(v1_mapping) is dict
        v2_decoded = batch_v2.decode_and_audit_trusted_envelope_v2(v2_row.envelope)
        v2_logical = typed_v1.encode_typed_transform_authority_profile_v1(
            v2_decoded.authority
        )
        assert v2_logical == v1_mapping
        assert v1_row.envelope != v2_row.envelope
        assert v1_row.envelope[:8] == wire.ENVELOPE_MAGIC
        assert v2_row.envelope[:8] == batch_v2.TRUSTED_WIRE_ENVELOPE_V2_MAGIC
        public_ids = _uuid_values(v2_logical)
        assert public_ids
        assert all(UUID4.fullmatch(value) for value in public_ids)
        assert all_public_ids.isdisjoint(public_ids)
        all_public_ids.update(public_ids)
    assert all_source_ids.isdisjoint(all_public_ids)
    for row in all_transforms_batch.envelopes:
        payload, _, _ = _v2_payload(row.envelope)
        assert all(value.encode("ascii") not in payload for value in all_source_ids)


def test_positive_source_expanded_125582_builds_one_full_v2_envelope_and_keeps_decision(
    positive_fixture: object,
    positive_batch: batch_v2.TrustedWireBatchV2,
) -> None:
    expanded = typed_v1.encode_typed_transform_authority_profile_v1(
        positive_fixture.source_authority
    )
    assert len(_canonical(expanded)) == 125_582
    assert len(_canonical(expanded)) > wire.MAXIMUM_PAYLOAD_BYTES
    assert len(positive_batch.envelopes) == 1
    assert positive_batch.batch_id == (
        "phase2b_trusted_wire_batch_v2_"
        "d439bcc69ee644c5b3328d508b1aa493c92c8aa9a7dab2d057284456660ee155"
    )
    row = positive_batch.envelopes[0]
    payload, padding, mapping = _v2_payload(row.envelope)
    assert len(row.envelope) == 65_536
    assert len(payload) == 50_255 < wire.MAXIMUM_PAYLOAD_BYTES
    assert len(padding) == 15_201 >= wire.MINIMUM_PADDING_BYTES
    assert row.payload_bytes == len(payload)
    assert row.padding_bytes == len(padding)
    assert row.payload_sha256 == hashlib.sha256(payload).hexdigest()
    assert row.padding_sha256 == hashlib.sha256(padding).hexdigest()
    assert row.envelope_id == (
        "phase2b_trusted_envelope_v2_"
        "371c69a9dbdfcd993f65a3eac4a6f8a62c40991c6df8405ace0436efe8221f09"
    )
    assert row.payload_sha256 == (
        "d8c5e91b62cea16e29b4e4cb7d58e3550ed985490ba7c6d43936e7ad28b7e1e0"
    )
    assert row.padding_sha256 == (
        "841f2e7f7f1f37b4ad1d16d01d889f57ead04ce67ccc649ffd48e527f683d174"
    )
    assert hashlib.sha256(row.envelope).hexdigest() == (
        "7bed8a1dbd22cdd1e2bc910ec2e9f7dad5ec7c6993e9eac8586255ff5bc94059"
    )
    assert row.namespace_audit_id == (
        "phase2b_namespace_audit_v2_"
        "d230d44adda6f6ab98dc93a5bb31444bbd08cf4e7a75a15d0b5e5f9f871542d5"
    )
    assert row.authority_content_id == (
        "phase2b_public_transform_evidence_"
        "a4d4573ffe5523ff6acea239741031df6b178943aaf7832bb7ab9346ffa32fae"
    )
    assert row.transform_result_id == (
        "phase2b_exact_transform_result_"
        "eaa1a463d1c7b7e7c1b6c1b3f99d1ed2947e28adc96e9d52db6272248fe54cc4"
    )
    assert tuple(sorted(mapping)) == batch_v2.TRUSTED_WIRE_BATCH_V2_PAYLOAD_FIELDS
    decoded = batch_v2.decode_and_audit_trusted_envelope_v2(row.envelope)
    assert decoded.authority == positive_fixture.public_authority
    assert decoded.authority_content_id == positive_fixture.public_authority.content_id
    transform_run = transform.run_exact_transform_semantics(decoded.authority)
    assert transform_run.result_id == row.transform_result_id
    bridge_after = derived_bridge.run_exact_derived_witness_bridge(
        authority=decoded.authority,
        theory=positive_fixture.theory,
        registry=positive_fixture.public_registry.to_adapter_registry(),
    )
    assert type(bridge_after) is derived_bridge.ExactDerivedBridgeRun
    assert bridge_after.compilation == positive_fixture.bridge_run.compilation
    assert bridge_after.decision == positive_fixture.bridge_run.decision


def test_public_batch_claims_do_not_upgrade_secret_mechanics(
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
) -> None:
    value = all_transforms_batch
    assert value.claim_level == wire.NON_AUTHORITATIVE_CLAIM_LEVEL
    assert value.atomic_complete_batch_structure_verified
    assert value.typed_authority_decode_replay_verified
    assert value.direct_exact_transform_replay_verified
    for name in (
        "whole_batch_shuffle_publicly_verified",
        "purpose_separated_keys_publicly_verified",
        "post_shuffle_hmac_uuidv4_publicly_verified",
        "secret_hmac_padding_publicly_verified",
        "recognizer_capacity_evidence",
        "origin_authenticated",
        "formal_uuid_audit",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "c1_exit_evidence",
    ):
        assert getattr(value, name) is False
    for row in value.envelopes:
        decoded = batch_v2.decode_and_audit_trusted_envelope_v2(row.envelope)
        for name in (
            "secret_padding_replay_verified",
            "origin_authenticated",
            "formal_uuid_audit",
            "formal_covert_audit",
            "sealed_holdout_eligible",
            "c1_exit_evidence",
        ):
            assert getattr(row, name) is False
            assert getattr(decoded, name) is False


@pytest.mark.parametrize(
    "name",
    (
        "whole_batch_shuffle_publicly_verified",
        "purpose_separated_keys_publicly_verified",
        "post_shuffle_hmac_uuidv4_publicly_verified",
        "secret_hmac_padding_publicly_verified",
    ),
)
def test_coherent_public_batch_forgery_cannot_turn_secret_claims_true(
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
    name: str,
) -> None:
    source = all_transforms_batch
    forged = object.__new__(batch_v2.TrustedWireBatchV2)
    for item in fields(source):
        object.__setattr__(forged, item.name, getattr(source, item.name))
    object.__setattr__(forged, name, True)
    with pytest.raises(ValueError, match="claim boundary"):
        forged._validate()


def test_private_validated_contexts_reject_missing_or_external_tokens(
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
) -> None:
    batch = all_transforms_batch
    row = batch.envelopes[0]
    decoded = batch_v2.decode_and_audit_trusted_envelope_v2(row.envelope)
    parts = batch_v2._decode_structural_envelope_v2(row.envelope)
    for token in (None, object()):
        with pytest.raises(TypeError, match="context"):
            decoded._validate(parts=parts, context_token=token)
        with pytest.raises(TypeError, match="context"):
            row._validate(decoded=decoded, context_token=token)
        with pytest.raises(TypeError, match="context"):
            batch._validate(validated_rows=batch.envelopes, context_token=token)


def test_private_decoded_parts_and_envelope_contexts_reject_wrong_exact_pairings_cheaply(
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_row, second_row = all_transforms_batch.envelopes[:2]
    first_decoded = batch_v2.decode_and_audit_trusted_envelope_v2(
        first_row.envelope
    )
    second_decoded = batch_v2.decode_and_audit_trusted_envelope_v2(
        second_row.envelope
    )
    second_parts = batch_v2._decode_structural_envelope_v2(second_row.envelope)

    with pytest.raises((TypeError, ValueError), match="context|pair|drift"):
        first_decoded._validate(
            parts=second_parts,
            context_token=batch_v2._DECODED_CONTEXT_TOKEN_V2,
        )
    with pytest.raises((TypeError, ValueError), match="context|pair|drift"):
        first_row._validate(
            decoded=second_decoded,
            context_token=batch_v2._ENVELOPE_CONTEXT_TOKEN_V2,
        )


@pytest.mark.parametrize(
    "kind",
    ("parts_object", "parts_subclass", "decoded_object"),
)
def test_private_context_exact_types_precede_all_deep_validation(
    kind: str,
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = all_transforms_batch.envelopes[0]
    decoded = batch_v2.decode_and_audit_trusted_envelope_v2(row.envelope)
    parts = batch_v2._decode_structural_envelope_v2(row.envelope)

    class PartsSubclass(batch_v2._DecodedEnvelopePartsV2):
        pass

    parts_subclass = object.__new__(PartsSubclass)
    for item in fields(parts):
        object.__setattr__(parts_subclass, item.name, getattr(parts, item.name))

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("non-exact private context reached deep validation")

    monkeypatch.setattr(batch_v2, "_digest", forbidden)
    monkeypatch.setattr(batch_v2, "_source_string_cap_preflight_v2", forbidden)
    if kind == "decoded_object":
        with pytest.raises(TypeError, match="decoded|context|exact"):
            row._validate(
                decoded=object(),  # type: ignore[arg-type]
                context_token=batch_v2._ENVELOPE_CONTEXT_TOKEN_V2,
            )
    else:
        supplied = object() if kind == "parts_object" else parts_subclass
        with pytest.raises(TypeError, match="parts|context|exact"):
            decoded._validate(
                parts=supplied,  # type: ignore[arg-type]
                context_token=batch_v2._DECODED_CONTEXT_TOKEN_V2,
            )


@pytest.mark.parametrize(
    "field_name",
    ("namespace_audit_id", "authority_content_id", "transform_result_id"),
)
def test_issued_row_root_items_digest_reject_hostile_equality_before_projection(
    field_name: str,
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HostileEqualString(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError("non-exact row root reached context equality")

        __hash__ = str.__hash__

    row = all_transforms_batch.envelopes[0]
    decoded = batch_v2.decode_and_audit_trusted_envelope_v2(row.envelope)
    forged = _unchecked_copy(
        row,
        **{field_name: HostileEqualString(getattr(row, field_name))},
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid row root reached envelope replay")

    monkeypatch.setattr(
        batch_v2,
        "decode_and_audit_trusted_envelope_v2",
        forbidden,
    )
    with pytest.raises((TypeError, ValueError), match="ID|exact|string|digest"):
        forged._validate(  # type: ignore[attr-defined]
            decoded=decoded,
            context_token=batch_v2._ENVELOPE_CONTEXT_TOKEN_V2,
        )


def test_public_zero_argument_validation_replays_rows_and_detects_pollution(
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = all_transforms_batch
    row = batch.envelopes[0]
    decoded = batch_v2.decode_and_audit_trusted_envelope_v2(row.envelope)
    raw_calls = 0
    original_raw = batch_v2._decode_structural_envelope_v2

    def count_raw(*args: object, **kwargs: object) -> object:
        nonlocal raw_calls
        raw_calls += 1
        return original_raw(*args, **kwargs)

    monkeypatch.setattr(batch_v2, "_decode_structural_envelope_v2", count_raw)
    decoded._validate()
    assert raw_calls == 1

    row_calls = 0
    original_row = batch_v2.TrustedWireEnvelopeV2._validate

    def count_row(self: object, *args: object, **kwargs: object) -> None:
        nonlocal row_calls
        row_calls += 1
        original_row(self, *args, **kwargs)

    monkeypatch.setattr(batch_v2.TrustedWireEnvelopeV2, "_validate", count_row)
    batch._validate()
    assert row_calls == len(batch.envelopes)
    monkeypatch.setattr(batch_v2, "_decode_structural_envelope_v2", original_raw)

    polluted = object.__new__(batch_v2.TrustedWireEnvelopeV2)
    for item in fields(row):
        object.__setattr__(polluted, item.name, getattr(row, item.name))
    object.__setattr__(polluted, "payload_sha256", "0" * 64)
    with pytest.raises(ValueError, match="drift"):
        original_row(polluted)


def test_claim_and_scalar_pollution_rejects_before_deep_replay(
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = all_transforms_batch
    row = batch.envelopes[0]
    decoded = batch_v2.decode_and_audit_trusted_envelope_v2(row.envelope)

    class StringSubclass(str):
        pass

    class IntegerSubclass(int):
        pass

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("cheap exact-field rejection reached deep replay")

    decoded_claim = object.__new__(batch_v2.DecodedTrustedEnvelopeV2)
    for item in fields(decoded):
        object.__setattr__(decoded_claim, item.name, getattr(decoded, item.name))
    object.__setattr__(decoded_claim, "structural_hashes_verified", 1)
    monkeypatch.setattr(batch_v2, "_decode_structural_envelope_v2", forbidden)
    with pytest.raises(TypeError, match="exact booleans"):
        decoded_claim._validate()

    decoded_identity = object.__new__(batch_v2.DecodedTrustedEnvelopeV2)
    for item in fields(decoded):
        object.__setattr__(decoded_identity, item.name, getattr(decoded, item.name))
    object.__setattr__(
        decoded_identity,
        "payload_schema_version",
        StringSubclass(decoded.payload_schema_version),
    )
    with pytest.raises((TypeError, ValueError), match="schema"):
        decoded_identity._validate()

    issued_length = object.__new__(batch_v2.TrustedWireEnvelopeV2)
    for item in fields(row):
        object.__setattr__(issued_length, item.name, getattr(row, item.name))
    object.__setattr__(issued_length, "payload_bytes", IntegerSubclass(row.payload_bytes))
    monkeypatch.setattr(batch_v2, "decode_and_audit_trusted_envelope_v2", forbidden)
    with pytest.raises(TypeError, match="lengths"):
        issued_length._validate()

    polluted_batch = object.__new__(batch_v2.TrustedWireBatchV2)
    for item in fields(batch):
        object.__setattr__(polluted_batch, item.name, getattr(batch, item.name))
    object.__setattr__(
        polluted_batch,
        "envelope_ids",
        (StringSubclass(batch.envelope_ids[0]), *batch.envelope_ids[1:]),
    )
    monkeypatch.setattr(batch_v2.TrustedWireEnvelopeV2, "_validate", forbidden)
    with pytest.raises((TypeError, ValueError), match="envelope ID"):
        polluted_batch._validate()

    batch_claim = object.__new__(batch_v2.TrustedWireBatchV2)
    for item in fields(batch):
        object.__setattr__(batch_claim, item.name, getattr(batch, item.name))
    object.__setattr__(batch_claim, "atomic_complete_batch_structure_verified", 1)
    with pytest.raises(TypeError, match="exact booleans"):
        batch_claim._validate()


@pytest.mark.parametrize(
    ("field_name", "polluted_value"),
    (
        ("envelope_id", "bad-envelope-id"),
        ("payload_bytes", -1),
        ("payload_schema_version", "wrong-payload-schema"),
    ),
)
def test_decoded_cheap_ids_lengths_and_identity_precede_authority_walker(
    field_name: str,
    polluted_value: object,
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded = batch_v2.decode_and_audit_trusted_envelope_v2(
        all_transforms_batch.envelopes[0].envelope
    )
    forged = _unchecked_copy(decoded, **{field_name: polluted_value})

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("cheap decoded drift reached authority walk or codec")

    monkeypatch.setattr(batch_v2, "_source_string_cap_preflight_v2", forbidden)
    monkeypatch.setattr(
        batch_v2,
        "encode_typed_transform_authority_profile_v2",
        forbidden,
    )
    with pytest.raises((TypeError, ValueError)):
        forged._validate()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "mutation",
    (
        "entries",
        "total_string_bytes",
        "uuid_occurrences",
        "unique_uuids",
        "safe_integer",
        "finite_float",
        "rational_bits",
    ),
)
def test_decoded_forged_authority_uses_same_walker_before_codec_and_replay(
    mutation: str,
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded = batch_v2.decode_and_audit_trusted_envelope_v2(
        all_transforms_batch.envelopes[0].envelope
    )
    polluted = _authority_with_walker_cap_drift(
        authorities["identity_authority"],
        mutation,
    )
    forged = _unchecked_copy(decoded, authority=polluted)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid decoded authority reached codec or replay")

    monkeypatch.setattr(
        batch_v2,
        "encode_typed_transform_authority_profile_v2",
        forbidden,
    )
    monkeypatch.setattr(batch_v2, "_decode_structural_envelope_v2", forbidden)
    with pytest.raises((TypeError, ValueError), match="authority|resource|cap|budget"):
        forged._validate()  # type: ignore[attr-defined]


def test_v2_payload_codec_is_private_exact_and_independent_of_v1_physical_caps(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _payload_mapping_for_authority(authorities["identity_authority"])

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("V2 payload codec delegated to V1 physical encoder")

    monkeypatch.setattr(wire, "encode_phase2b_jcs_profile_v1", forbidden)
    payload = batch_v2._encode_payload_jcs_v2(mapping)
    assert payload == _canonical(mapping)
    assert batch_v2._decode_payload_jcs_v2(payload) == mapping


def test_v2_payload_integer_symmetry_empty_table_and_deep_json_fail_closed(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    maximum = (1 << 53) - 1
    assert batch_v2._parse_json_integer(str(maximum)) == maximum
    assert batch_v2._parse_json_integer(str(-maximum)) == -maximum
    for value in (1 << 53, -(1 << 53)):
        with pytest.raises(ValueError, match="safe range"):
            batch_v2._parse_json_integer(str(value))

    mapping = _payload_mapping_for_authority(authorities["identity_authority"])
    compact = mapping["authority"]
    assert type(compact) is dict and type(compact["strings"]) is list
    compact["strings"][0] = ""  # type: ignore[index]
    with pytest.raises((TypeError, ValueError)):
        batch_v2._encode_payload_jcs_v2(mapping)

    deep = b"[" * 2_000 + b"null" + b"]" * 2_000
    with pytest.raises(ValueError, match="(?:depth|JSON|resource|nested)"):
        batch_v2._decode_payload_jcs_v2(deep)


def test_public_decoder_uses_raw_parts_without_recursive_public_entry(
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = all_transforms_batch.envelopes[0]
    original = batch_v2.decode_and_audit_trusted_envelope_v2

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("public V2 decoder recursively called itself")

    monkeypatch.setattr(batch_v2, "decode_and_audit_trusted_envelope_v2", forbidden)
    decoded = original(row.envelope)
    assert type(decoded) is batch_v2.DecodedTrustedEnvelopeV2
    parts = batch_v2._decode_structural_envelope_v2(row.envelope)
    assert type(parts).__name__ == "_DecodedEnvelopePartsV2"


def test_codec_payload_magic_batch_and_key_types_cross_reject(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
    v1_all_transforms_batch: batch_v1.TrustedWireBatchV1,
) -> None:
    v1_row = v1_all_transforms_batch.envelopes[0]
    v2_row = all_transforms_batch.envelopes[0]
    with pytest.raises(ValueError, match="magic|version"):
        batch_v2.decode_and_audit_trusted_envelope_v2(v1_row.envelope)
    with pytest.raises(ValueError, match="magic|version"):
        batch_v1.decode_and_audit_trusted_envelope_v1(v2_row.envelope)
    with pytest.raises(TypeError, match="exact V2 type"):
        batch_v2.build_trusted_wire_batch_v2(
            authorities=(authorities["identity_authority"],),
            run_id=RUN_ID,
            key_sources=_v1_keys(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="exact frozen type"):
        batch_v1.build_trusted_wire_batch_v1(
            authorities=(authorities["identity_authority"],),
            run_id=RUN_ID,
            key_sources=_keys(),  # type: ignore[arg-type]
        )

    v2_payload, v2_padding, v2_mapping = _v2_payload(v2_row.envelope)
    v1_payload, v1_padding = _parts(
        v1_row.envelope,
        magic=wire.ENVELOPE_MAGIC,
        version=1,
    )
    with pytest.raises(ValueError, match="payload schema|identity"):
        batch_v2.decode_and_audit_trusted_envelope_v2(
            _reframe_v2(v1_payload, v2_padding)
        )
    v1_as_v2 = deepcopy(v2_mapping)
    v1_as_v2["authority"] = typed_v1.encode_typed_transform_authority_profile_v1(
        authorities["identity_authority"]
    )
    with pytest.raises((TypeError, ValueError)):
        batch_v2.decode_and_audit_trusted_envelope_v2(
            _reframe_v2(_canonical(v1_as_v2), v2_padding)
        )
    v2_as_v1 = wire.decode_phase2b_jcs_profile_v1(v1_payload)
    assert type(v2_as_v1) is dict
    v2_as_v1["authority"] = v2_mapping["authority"]
    v2_as_v1["typed_authority_schema_id"] = (
        typed_v2.COMPACT_TYPED_AUTHORITY_SCHEMA_ID
    )
    forged_v1_payload = wire.encode_phase2b_jcs_profile_v1(v2_as_v1)
    forged_v1 = batch_v1._frame_secret_payload(
        forged_v1_payload,
        b"K" * 32,
        RUN_ID,
    )
    with pytest.raises(ValueError, match="identity|schema"):
        batch_v1.decode_and_audit_trusted_envelope_v1(forged_v1)
    with pytest.raises(TypeError, match="exact batch type"):
        batch_v1.verify_trusted_wire_batch_replay_v1(
            batch=all_transforms_batch,  # type: ignore[arg-type]
            run_id=RUN_ID,
            key_sources=_v1_keys(),
            authorities=tuple(authorities[name] for name in FACTORY_NAMES),
        )


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "field_manifest_id",
        "jcs_profile_id",
        "public_provenance_version",
        "typed_authority_schema_id",
        "typed_authority_codec_version",
        "typed_authority_codec_policy_id",
    ),
)
def test_wrong_payload_identity_fields_reject_after_coherent_reframing(
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
    field: str,
) -> None:
    _, padding, mapping = _v2_payload(all_transforms_batch.envelopes[0].envelope)
    mapping[field] = "drift"
    with pytest.raises(ValueError, match="identity"):
        batch_v2.decode_and_audit_trusted_envelope_v2(
            _reframe_v2(_canonical(mapping), padding)
        )


def test_missing_extra_noncanonical_duplicate_hash_and_padding_tamper_reject(
    all_transforms_batch: batch_v2.TrustedWireBatchV2,
) -> None:
    row = all_transforms_batch.envelopes[0]
    payload, padding, mapping = _v2_payload(row.envelope)
    missing = deepcopy(mapping)
    missing.pop("field_manifest_id")
    extra = deepcopy(mapping)
    extra["extra"] = None
    noncanonical = payload.replace(b",", b", ", 1)
    schema_literal = json.dumps(mapping["schema_version"], separators=(",", ":"))
    duplicate = (
        b'{"schema_version":'
        + schema_literal.encode("ascii")
        + b","
        + payload[1:]
    )
    for forged_payload in (
        _canonical(missing),
        _canonical(extra),
        noncanonical,
        duplicate,
    ):
        with pytest.raises((TypeError, ValueError)):
            batch_v2.decode_and_audit_trusted_envelope_v2(
                _reframe_v2(forged_payload, padding)
            )
    tampered_payload = (
        row.envelope[: HEADER.size]
        + bytes([row.envelope[HEADER.size] ^ 1])
        + row.envelope[HEADER.size + 1 :]
    )
    tampered_padding = row.envelope[:-1] + bytes([row.envelope[-1] ^ 1])
    for forged in (row.envelope[:-1], tampered_payload, tampered_padding):
        with pytest.raises(ValueError):
            batch_v2.decode_and_audit_trusted_envelope_v2(forged)


def test_second_case_failure_is_atomic_with_exact_empty_rejection_roots(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = batch_v2._frame_secret_payload_v2
    calls = 0

    def fail_second(*args: object, **kwargs: object) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("injected second-case failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(batch_v2, "_frame_secret_payload_v2", fail_second)
    result = batch_v2.build_trusted_wire_batch_v2(
        authorities=(
            authorities["identity_authority"],
            authorities["unit_authority"],
        ),
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert calls == 2
    assert type(result) is batch_v2.TrustedWireBatchRejectionV2
    assert result.disposition is batch_v2.TrustedWireBatchDispositionV2.ABSTAIN
    assert (
        result.envelopes,
        result.envelope_ids,
        result.authority_content_ids,
        result.transform_result_ids,
    ) == ((), (), (), ())
    assert result.recognizer_capacity_evidence is False
    assert result.origin_authenticated is False
    assert result.formal_covert_audit is False
    assert result.sealed_holdout_eligible is False
    assert result.c1_exit_evidence is False


def test_batch_policy_run_ordered_roots_and_exact_tuple_items_reject_pollution(
    one_row_batch: batch_v2.TrustedWireBatchV2,
) -> None:
    source = one_row_batch

    class StringSubclass(str):
        pass

    mutations = (
        ("policy_id", "phase2b_trusted_wire_batch_v2_policy_" + "0" * 64),
        ("run_id_commitment", "phase2b_trusted_wire_run_v2_" + "0" * 64),
        ("batch_id", "phase2b_trusted_wire_batch_v2_" + "0" * 64),
        (
            "envelope_ids",
            (StringSubclass(source.envelope_ids[0]),),
        ),
        (
            "authority_content_ids",
            (StringSubclass(source.authority_content_ids[0]),),
        ),
        (
            "transform_result_ids",
            (StringSubclass(source.transform_result_ids[0]),),
        ),
    )
    for name, polluted in mutations:
        forged = object.__new__(batch_v2.TrustedWireBatchV2)
        for item in fields(source):
            object.__setattr__(forged, item.name, getattr(source, item.name))
        object.__setattr__(forged, name, polluted)
        with pytest.raises((TypeError, ValueError)):
            forged._validate()


@pytest.mark.parametrize(
    "stored_name",
    ("envelope_ids", "authority_content_ids", "transform_result_ids"),
)
def test_batch_stored_root_items_validate_exact_digest_before_tuple_equality(
    stored_name: str,
    one_row_batch: batch_v2.TrustedWireBatchV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HostileEqualString(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError("non-exact stored root reached tuple equality")

        __hash__ = str.__hash__

    forged = _unchecked_copy(
        one_row_batch,
        **{
            stored_name: (
                HostileEqualString(getattr(one_row_batch, stored_name)[0]),
            )
        },
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid stored root reached deep row validation")

    monkeypatch.setattr(batch_v2.TrustedWireEnvelopeV2, "_validate", forbidden)
    with pytest.raises((TypeError, ValueError), match="ID|root|exact|digest"):
        forged._validate()  # type: ignore[attr-defined]


@pytest.mark.parametrize("kind", ("tuple_subclass", "equal_row_copy"))
def test_batch_private_validated_rows_require_exact_tuple_and_identity_pairing_before_deep(
    kind: str,
    one_row_batch: batch_v2.TrustedWireBatchV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TupleSubclass(tuple):
        pass

    if kind == "tuple_subclass":
        supplied = TupleSubclass(one_row_batch.envelopes)
    else:
        row = one_row_batch.envelopes[0]
        equal_copy = _unchecked_copy(row)
        assert type(equal_copy) is batch_v2.TrustedWireEnvelopeV2
        assert equal_copy == row and equal_copy is not row
        supplied = (equal_copy,)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("wrong validated rows reached IDs, rows, or hashing")

    monkeypatch.setattr(batch_v2, "_batch_id_v2", forbidden)
    monkeypatch.setattr(batch_v2.TrustedWireEnvelopeV2, "_validate", forbidden)
    with pytest.raises((TypeError, ValueError), match="validated|context|identity|exact"):
        one_row_batch._validate(
            validated_rows=supplied,  # type: ignore[arg-type]
            context_token=batch_v2._BATCH_CONTEXT_TOKEN_V2,
        )


def test_run_key_and_source_order_drift_change_the_complete_batch_roots(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    source = (
        authorities["identity_authority"],
        authorities["unit_authority"],
    )

    def build(
        values: tuple[transform.PublicTransformEvidenceBundleV2, ...],
        *,
        run_id: bytes = RUN_ID,
        identifiers: bytes = b"I" * 32,
    ) -> batch_v2.TrustedWireBatchV2:
        result = batch_v2.build_trusted_wire_batch_v2(
            authorities=values,
            run_id=run_id,
            key_sources=_keys(identifiers=identifiers),
        )
        assert type(result) is batch_v2.TrustedWireBatchV2
        return result

    baseline = build(source)
    same = build(source)
    changed_run = build(source, run_id=b"T" * 32)
    changed_key = build(source, identifiers=b"J" * 32)
    changed_order = build(tuple(reversed(source)))
    assert same == baseline
    for changed in (changed_run, changed_key, changed_order):
        assert changed.batch_id != baseline.batch_id
        assert changed.envelope_ids != baseline.envelope_ids


def test_sampling_grid_pairing_is_exact_under_i_and_j_uuid_schedules(
    stage_a_sampling_authority: transform.PublicTransformEvidenceBundleV2,
) -> None:
    rows_by_key: dict[
        bytes,
        tuple[list[tuple[object, ...]], list[tuple[object, ...]]],
    ] = {}

    def ref_key(value: transform.ComponentRef) -> tuple[object, ...]:
        return (
            value.scale_id,
            value.observation_id,
            value.ordinal,
            value.component_id,
        )

    for identifier in (b"I" * 32, b"J" * 32):
        built = batch_v2.build_trusted_wire_batch_v2(
            authorities=(stage_a_sampling_authority,),
            run_id=RUN_ID,
            key_sources=_keys(identifiers=identifier),
        )
        assert type(built) is batch_v2.TrustedWireBatchV2
        decoded = batch_v2.decode_and_audit_trusted_envelope_v2(
            built.envelopes[0].envelope
        )
        result = transform.run_exact_transform_semantics(decoded.authority)
        assert type(result) is transform.ExactTransformCompilation
        assert result.disposition is transform.TransformCompilationDisposition.COMPLETE
        contract = decoded.authority.transform_contracts[0]
        certificate = contract.certificate
        assert type(certificate) is transform.SamplingResolutionCertificate
        selected = [ref_key(item) for item in certificate.selected_inputs]
        grid = [
            tuple(Fraction(item.numerator, item.denominator) for item in point)
            for point in certificate.grid_points
        ]
        row_inputs = [ref_key(row.terms[0].input_ref) for row in contract.kernel_rows]
        assert grid == sorted(grid)
        assert len(grid) == len(set(grid)) == len(selected) == len(set(selected))
        assert set(row_inputs) == set(selected)
        rows_by_key[identifier] = (selected, row_inputs)
    assert rows_by_key[b"I" * 32][0] == rows_by_key[b"I" * 32][1]
    assert rows_by_key[b"J" * 32][0] != rows_by_key[b"J" * 32][1]


def test_single_live_allocation_and_shared_helper_schedule_is_exact(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = (
        "_hash_free_authority_preflight",
        "_derive_keys",
        "_shuffle_indices",
        "_rename_authority_ids",
        "_canonicalize_public_authority",
        "_compile_native_public_provenance",
        "_audit_public_provenance",
    )
    counts = {name: 0 for name in names}
    state_ids: list[tuple[int, int]] = []
    for name in names:
        original = getattr(batch_v1, name)

        def wrapped(
            *args: object,
            __name: str = name,
            __original: Callable[..., object] = original,
            **kwargs: object,
        ) -> object:
            counts[__name] += 1
            if __name == "_rename_authority_ids":
                state_ids.append((id(args[4]), id(args[5])))
            return __original(*args, **kwargs)

        monkeypatch.setattr(batch_v1, name, wrapped)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("V2 builder called a forbidden V1 compiler/core/frame")

    for name in (
        "compile_transform_authority_profile_mechanics_v1",
        "_build_trusted_wire_batch_core_v1",
        "build_trusted_wire_batch_v1",
        "_frame_secret_payload",
    ):
        monkeypatch.setattr(batch_v1, name, forbidden)
    result = batch_v2.build_trusted_wire_batch_v2(
        authorities=(
            authorities["identity_authority"],
            authorities["unit_authority"],
        ),
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(result) is batch_v2.TrustedWireBatchV2
    assert counts == {
        "_hash_free_authority_preflight": 2,
        "_derive_keys": 1,
        "_shuffle_indices": 1,
        "_rename_authority_ids": 2,
        "_canonicalize_public_authority": 4,
        "_compile_native_public_provenance": 4,
        "_audit_public_provenance": 2,
    }
    assert len(state_ids) == 2
    assert len(set(state_ids)) == 1


def test_global_source_and_fresh_decoded_public_uuid_intersection_is_atomic(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shuffle_key, _, _ = batch_v1._derive_keys(RUN_ID, _keys())
    order = batch_v1._shuffle_indices(2, shuffle_key, RUN_ID)
    assert len(order) == 2
    ordered_sources: list[transform.PublicTransformEvidenceBundleV2 | None] = [
        None,
        None,
    ]
    ordered_sources[order[0]] = authorities["identity_authority"]
    ordered_sources[order[1]] = authorities["unit_authority"]
    assert all(item is not None for item in ordered_sources)
    source = tuple(ordered_sources)
    first_ids = _uuid_values(
        typed_v1.encode_typed_transform_authority_profile_v1(source[order[0]])  # type: ignore[arg-type]
    )
    second_ids = _uuid_values(
        typed_v1.encode_typed_transform_authority_profile_v1(source[order[1]])  # type: ignore[arg-type]
    )
    second_only = second_ids - first_ids
    assert second_only
    collision = min(second_only)
    original_candidate = batch_v1._uuid4_candidate

    def collide_with_source(
        key: bytes,
        run_id: bytes,
        namespace: str,
        counter: int,
        retry: int,
    ) -> str:
        if namespace == "bundle" and counter == 0 and retry == 0:
            return collision
        return original_candidate(key, run_id, namespace, counter, retry)

    monkeypatch.setattr(batch_v1, "_uuid4_candidate", collide_with_source)
    result = batch_v2.build_trusted_wire_batch_v2(
        authorities=source,  # type: ignore[arg-type]
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(result) is batch_v2.TrustedWireBatchRejectionV2
    assert result.reason == "source_public_uuid_collision"
    assert result.envelopes == result.envelope_ids == ()
    assert result.authority_content_ids == result.transform_result_ids == ()
