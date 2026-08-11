"""Adversarial public-API tests for keyed Phase-2B trusted-wire batches."""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import re
import runpy
import struct
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path

import pytest

import hegel_machine.phase2b_exact_transform_semantics_v1 as transform_semantics
import hegel_machine.phase2b_trusted_wire_batch_v1 as batch_wire
from hegel_machine.phase2b_exact_transform_semantics_v1 import (
    PublicTransformEvidenceBundleV2,
)
from hegel_machine.phase2b_trusted_wire_v1 import (
    ENVELOPE_BYTES,
    ENVELOPE_HEADER_BYTES,
    ENVELOPE_MAGIC,
    NON_AUTHORITATIVE_CLAIM_LEVEL,
    decode_phase2b_jcs_profile_v1,
    encode_phase2b_jcs_profile_v1,
)


HEADER = struct.Struct(">8sHHI32s32s")
UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
RUN_ID = b"R" * 32


def _keys(
    *,
    shuffle: bytes = b"S" * 32,
    identifiers: bytes = b"I" * 32,
    padding: bytes = b"P" * 32,
) -> batch_wire.TrustedWireKeySourcesV1:
    return batch_wire.TrustedWireKeySourcesV1(shuffle, identifiers, padding)


@pytest.fixture(scope="module")
def authorities() -> dict[str, PublicTransformEvidenceBundleV2]:
    stage_a = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_trusted_wire_v1.py"))
    )
    transforms = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_exact_transform_semantics_v1.py"))
    )
    return {
        "identity": stage_a["identity_authority"](),
        "sampling_resolution": stage_a["sampling_authority"](),
        "unit_conversion": transforms["unit_authority"](),
        "coordinate_affine": transforms["coordinate_authority"](),
        "temporal_aggregation": transforms["temporal_authority"](),
        "spatial_aggregation": transforms["spatial_authority"](),
        "equivalent_split_merge": transforms["split_authority"](),
        "coarse_graining": transforms["coarse_authority"](),
    }


def _build(
    values: tuple[PublicTransformEvidenceBundleV2, ...],
    *,
    run_id: bytes = RUN_ID,
    key_sources: batch_wire.TrustedWireKeySourcesV1 | None = None,
) -> batch_wire.TrustedWireBatchV1:
    result = batch_wire.build_trusted_wire_batch_v1(
        authorities=values,
        run_id=run_id,
        key_sources=_keys() if key_sources is None else key_sources,
    )
    assert type(result) is batch_wire.TrustedWireBatchV1
    assert result.disposition is batch_wire.TrustedWireBatchDisposition.COMPLETE
    return result


def _payload(envelope: bytes) -> tuple[bytes, bytes]:
    magic, version, header_bytes, payload_bytes, payload_sha, padding_sha = (
        HEADER.unpack(envelope[: HEADER.size])
    )
    assert magic == ENVELOPE_MAGIC
    assert version == 1
    assert header_bytes == ENVELOPE_HEADER_BYTES == HEADER.size
    payload = envelope[header_bytes : header_bytes + payload_bytes]
    padding = envelope[header_bytes + payload_bytes :]
    assert hashlib.sha256(payload).digest() == payload_sha
    assert hashlib.sha256(padding).digest() == padding_sha
    return payload, padding


def _reframe(payload: bytes, padding: bytes) -> bytes:
    return HEADER.pack(
        ENVELOPE_MAGIC,
        1,
        ENVELOPE_HEADER_BYTES,
        len(payload),
        hashlib.sha256(payload).digest(),
        hashlib.sha256(padding).digest(),
    ) + payload + padding


def _mapping(row: batch_wire.TrustedWireEnvelopeV1) -> dict[str, object]:
    payload, _ = _payload(row.envelope)
    value = decode_phase2b_jcs_profile_v1(payload)
    assert type(value) is dict
    return value


def _authority(row: batch_wire.TrustedWireEnvelopeV1) -> dict[str, object]:
    value = _mapping(row)["authority"]
    assert type(value) is dict
    return value


def _operation(row: batch_wire.TrustedWireEnvelopeV1) -> str:
    return _authority(row)["transform_contracts"][0]["operation"]  # type: ignore[index,return-value]


def _by_operation(value: batch_wire.TrustedWireBatchV1) -> dict[str, batch_wire.TrustedWireEnvelopeV1]:
    return {_operation(row): row for row in value.envelopes}


def _ref_key(value: dict[str, object]) -> tuple[object, ...]:
    return (
        value["scale_id"],
        value["observation_id"],
        value["ordinal"],
        value["component_id"],
    )


def _assert_ref_array_sorted(values: list[dict[str, object]]) -> None:
    assert values == sorted(values, key=_ref_key)


def test_public_api_has_no_policy_permutation_root_or_receipt_override() -> None:
    build = inspect.signature(batch_wire.build_trusted_wire_batch_v1)
    assert tuple(build.parameters) == ("authorities", "run_id", "key_sources")
    assert all(
        item.kind is inspect.Parameter.KEYWORD_ONLY
        for item in build.parameters.values()
    )
    replay = inspect.signature(batch_wire.verify_trusted_wire_batch_replay_v1)
    assert tuple(replay.parameters) == (
        "batch",
        "run_id",
        "key_sources",
        "authorities",
    )
    assert all(
        item.kind is inspect.Parameter.KEYWORD_ONLY
        for item in replay.parameters.values()
    )
    assert tuple(inspect.signature(batch_wire.decode_and_audit_trusted_envelope_v1).parameters) == (
        "envelope",
    )
    for forbidden in ("policy", "permutation", "root", "receipt"):
        assert forbidden not in build.parameters
        assert forbidden not in replay.parameters


@pytest.mark.parametrize(
    "values",
    (
        (bytearray(32), b"I" * 32, b"P" * 32),
        (b"S" * 31, b"I" * 32, b"P" * 32),
        (b"S" * 32, b"I" * 33, b"P" * 32),
        (b"S" * 32, b"I" * 32, b"P" * 31),
        (b"S" * 32, b"S" * 32, b"P" * 32),
        (b"S" * 32, b"I" * 32, b"I" * 32),
    ),
)
def test_key_sources_require_exact_distinct_32_byte_values(values: tuple[object, object, object]) -> None:
    with pytest.raises(
        (TypeError, ValueError),
        match="exact bytes|32 bytes|pairwise distinct",
    ):
        batch_wire.TrustedWireKeySourcesV1(*values)  # type: ignore[arg-type]


def test_key_source_repr_does_not_disclose_ikm() -> None:
    text = repr(_keys())
    assert "shuffle_ikm" not in text
    assert "id_ikm" not in text
    assert "padding_ikm" not in text
    assert "SSSS" not in text and "IIII" not in text and "PPPP" not in text


def test_zero_and_1025_authority_batches_abstain_without_partial_receipts(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = batch_wire.build_trusted_wire_batch_v1(
        authorities=(), run_id=RUN_ID, key_sources=_keys()
    )
    assert type(empty) is batch_wire.TrustedWireBatchPreflightV1
    assert empty.disposition is batch_wire.TrustedWireBatchDisposition.ABSTAIN
    assert empty.authority_count == 0
    assert empty.policy_id == batch_wire.TRUSTED_WIRE_BATCH_POLICY_ID
    assert not hasattr(empty, "envelopes")

    def forbidden_compile(_: object) -> object:
        raise AssertionError("authority hashing/compilation ran after batch cap")

    monkeypatch.setattr(
        batch_wire,
        "compile_transform_authority_profile_mechanics_v1",
        forbidden_compile,
    )
    oversized = batch_wire.build_trusted_wire_batch_v1(
        authorities=(authorities["identity"],) * 1_025,
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(oversized) is batch_wire.TrustedWireBatchPreflightV1
    assert oversized.authority_count == 1_025
    assert not hasattr(oversized, "envelopes")


def test_exact_input_shapes_and_run_id_are_preflighted(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    with pytest.raises(TypeError, match="tuple"):
        batch_wire.build_trusted_wire_batch_v1(
            authorities=[authorities["identity"]],  # type: ignore[arg-type]
            run_id=RUN_ID,
            key_sources=_keys(),
        )
    with pytest.raises(TypeError, match="exact bytes"):
        batch_wire.build_trusted_wire_batch_v1(
            authorities=(authorities["identity"],),
            run_id=bytearray(RUN_ID),  # type: ignore[arg-type]
            key_sources=_keys(),
        )
    for invalid in (b"R" * 31, b"R" * 33):
        result = batch_wire.build_trusted_wire_batch_v1(
            authorities=(authorities["identity"],),
            run_id=invalid,
            key_sources=_keys(),
        )
        assert type(result) is batch_wire.TrustedWireBatchPreflightV1
        assert "32_bytes" in result.reason
        assert not hasattr(result, "envelopes")


def test_hash_free_authority_shape_preflight_runs_before_compiler(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = deepcopy(authorities["identity"])
    observation = authority.base_bundle.observations[0]
    object.__setattr__(authority.base_bundle, "observations", (observation,) * 2_049)

    def forbidden_compile(_: object) -> object:
        raise AssertionError("compiler ran after hash-free shape rejection")

    monkeypatch.setattr(
        batch_wire,
        "compile_transform_authority_profile_mechanics_v1",
        forbidden_compile,
    )
    result = batch_wire.build_trusted_wire_batch_v1(
        authorities=(authority,), run_id=RUN_ID, key_sources=_keys()
    )
    assert type(result) is batch_wire.TrustedWireBatchPreflightV1
    assert "hash_free_authority_shape" in result.reason
    assert not hasattr(result, "envelopes")


def test_three_case_whole_batch_shuffle_is_deterministic_and_not_input_order(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    inputs = (
        authorities["identity"],
        authorities["sampling_resolution"],
        authorities["unit_conversion"],
    )
    first = _build(inputs)
    second = _build(inputs)
    assert first == second
    assert first.batch_id == second.batch_id
    observed = tuple(_operation(row) for row in first.envelopes)
    assert observed == ("identity", "unit_conversion", "sampling_resolution")
    assert observed != ("identity", "sampling_resolution", "unit_conversion")
    assert first.whole_batch_shuffle_applied
    assert first.purpose_separated_keys_applied


def test_key_changes_respect_causal_layers_not_false_independence(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    inputs = (
        authorities["identity"],
        authorities["sampling_resolution"],
        authorities["unit_conversion"],
    )
    baseline = _build(inputs)
    changed_shuffle = _build(inputs, key_sources=_keys(shuffle=b"B" * 32))
    changed_ids = _build(inputs, key_sources=_keys(identifiers=b"J" * 32))
    changed_padding = _build(inputs, key_sources=_keys(padding=b"Q" * 32))

    baseline_order = tuple(_operation(row) for row in baseline.envelopes)
    assert tuple(_operation(row) for row in changed_shuffle.envelopes) != baseline_order
    assert changed_shuffle.batch_id != baseline.batch_id
    assert {
        row.payload_sha256 for row in changed_shuffle.envelopes
    } != {row.payload_sha256 for row in baseline.envelopes}

    assert tuple(_operation(row) for row in changed_ids.envelopes) == baseline_order
    base_by_operation = _by_operation(baseline)
    ids_by_operation = _by_operation(changed_ids)
    assert all(
        ids_by_operation[name].payload_sha256 != row.payload_sha256
        and ids_by_operation[name].padding_sha256 != row.padding_sha256
        and ids_by_operation[name].envelope != row.envelope
        for name, row in base_by_operation.items()
    )

    assert tuple(_operation(row) for row in changed_padding.envelopes) == baseline_order
    padding_by_operation = _by_operation(changed_padding)
    assert all(
        padding_by_operation[name].payload_sha256 == row.payload_sha256
        and padding_by_operation[name].padding_sha256 != row.padding_sha256
        and padding_by_operation[name].envelope != row.envelope
        for name, row in base_by_operation.items()
    )


def test_duplicate_latent_cases_receive_distinct_case_local_public_ids_and_bytes(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    authority = authorities["identity"]
    result = _build((authority, authority))
    assert len(result.envelopes) == 2
    first, second = result.envelopes
    assert first.payload_sha256 != second.payload_sha256
    assert first.padding_sha256 != second.padding_sha256
    assert first.envelope != second.envelope
    first_authority = _authority(first)
    second_authority = _authority(second)
    assert first_authority["base_bundle"]["bundle_id"] != second_authority[  # type: ignore[index]
        "base_bundle"
    ]["bundle_id"]

    public_sets: list[set[str]] = []
    for row in result.envelopes:
        decoded = batch_wire.decode_and_audit_trusted_envelope_v1(row.envelope)
        public_ids = {item.public_uuid for item in decoded.namespace_audit.occurrences}
        assert public_ids
        assert all(UUID4.fullmatch(value) for value in public_ids)
        # Repeated references inside one case must resolve to the same public ID.
        counts: dict[tuple[str, str], int] = {}
        for item in decoded.namespace_audit.occurrences:
            key = (item.namespace, item.public_uuid)
            counts[key] = counts.get(key, 0) + 1
        assert any(count > 1 for count in counts.values())
        public_sets.append(public_ids)
    # Allocation is batch-global even though the latent->public maps are case-local.
    assert public_sets[0].isdisjoint(public_sets[1])


def test_public_ids_are_lowercase_uuidv4_with_all_variants_and_global_uniqueness(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    result = _build(
        (
            authorities["identity"],
            authorities["sampling_resolution"],
            authorities["unit_conversion"],
        )
    )
    all_ids: list[str] = []
    variants: set[str] = set()
    for row in result.envelopes:
        decoded = batch_wire.decode_and_audit_trusted_envelope_v1(row.envelope)
        for item in decoded.namespace_audit.occurrences:
            assert UUID4.fullmatch(item.public_uuid)
            variants.add(item.public_uuid.split("-")[3][0])
            all_ids.append(item.public_uuid)
    assert variants.issubset(set("89ab")) and variants
    # Repeated references are intentional; distinct IDs never cross namespaces/cases.
    namespace_by_id: dict[str, str] = {}
    for row in result.envelopes:
        decoded = batch_wire.decode_and_audit_trusted_envelope_v1(row.envelope)
        for item in decoded.namespace_audit.occurrences:
            previous = namespace_by_id.setdefault(item.public_uuid, item.namespace)
            assert previous == item.namespace


def test_uuid_collision_retry_sets_warning_and_replays(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_digest = hmac.digest
    uuid_domain = b"HEGEL/PHASE2B/TRUSTED_WIRE/UUID/V1"

    def collide_on_first_try(key: bytes, message: bytes, digest: str) -> bytes:
        if uuid_domain in message and message[-1] == 0:
            return b"\x00" * 32
        return original_digest(key, message, digest)

    monkeypatch.setattr(batch_wire.hmac, "digest", collide_on_first_try)
    result = _build((authorities["identity"],))
    assert result.uuid_collision_retry_count > 0
    assert result.uuid_collision_warning
    replay = batch_wire.verify_trusted_wire_batch_replay_v1(
        batch=result,
        run_id=RUN_ID,
        key_sources=_keys(),
        authorities=(authorities["identity"],),
    )
    assert replay.replay_verified


def test_uuid_collision_retry_exhaustion_abstains_without_partial_batch(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_digest = hmac.digest
    uuid_domain = b"HEGEL/PHASE2B/TRUSTED_WIRE/UUID/V1"

    def exhaust_uuid_candidates(key: bytes, message: bytes, digest: str) -> bytes:
        if uuid_domain in message:
            return b"\x00" * 32
        return original_digest(key, message, digest)

    monkeypatch.setattr(batch_wire.hmac, "digest", exhaust_uuid_candidates)
    result = batch_wire.build_trusted_wire_batch_v1(
        authorities=(authorities["identity"],),
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(result) is batch_wire.TrustedWireBatchPreflightV1
    assert result.disposition is batch_wire.TrustedWireBatchDisposition.ABSTAIN
    assert "renamed_authority" in result.reason
    assert not hasattr(result, "envelopes")


def test_public_decoder_verifies_structure_and_public_provenance_but_not_secret(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    row = _build((authorities["identity"],)).envelopes[0]
    decoded = batch_wire.decode_and_audit_trusted_envelope_v1(row.envelope)
    assert decoded.public_provenance_verified
    assert decoded.structural_hashes_verified
    assert not decoded.secret_padding_replay_verified
    assert not decoded.typed_authority_decode_replay_implemented
    assert not decoded.origin_authenticated
    assert not decoded.formal_covert_audit

    payload, padding = _payload(row.envelope)
    mapping = decode_phase2b_jcs_profile_v1(payload)
    mapping["authority"]["base_bundle"]["observations"][0][  # type: ignore[index]
        "provenance_sha256"
    ] = "0" * 64
    forged_payload = encode_phase2b_jcs_profile_v1(mapping)
    with pytest.raises(ValueError, match="provenance"):
        batch_wire.decode_and_audit_trusted_envelope_v1(
            _reframe(forged_payload, padding)
        )


def test_secret_custodian_replay_and_wrong_run_key_or_authority_rejection(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    inputs = (authorities["identity"], authorities["sampling_resolution"])
    result = _build(inputs)
    receipt = batch_wire.verify_trusted_wire_batch_replay_v1(
        batch=result,
        run_id=RUN_ID,
        key_sources=_keys(),
        authorities=inputs,
    )
    assert receipt.batch_id == result.batch_id
    assert receipt.authority_count == 2
    assert receipt.replay_verified
    assert receipt.secret_key_schedule_replayed
    assert receipt.secret_padding_replayed
    assert receipt.shuffle_and_uuid_assignment_replayed

    attempts = (
        {"run_id": b"X" * 32, "key_sources": _keys(), "authorities": inputs},
        {
            "run_id": RUN_ID,
            "key_sources": _keys(shuffle=b"A" * 32),
            "authorities": inputs,
        },
        {
            "run_id": RUN_ID,
            "key_sources": _keys(identifiers=b"J" * 32),
            "authorities": inputs,
        },
        {
            "run_id": RUN_ID,
            "key_sources": _keys(padding=b"Q" * 32),
            "authorities": inputs,
        },
        {
            "run_id": RUN_ID,
            "key_sources": _keys(),
            "authorities": tuple(reversed(inputs)),
        },
    )
    for attempt in attempts:
        with pytest.raises(ValueError, match="secret replay mismatch"):
            batch_wire.verify_trusted_wire_batch_replay_v1(
                batch=result,
                **attempt,  # type: ignore[arg-type]
            )


def test_batch_envelope_decoded_and_replay_receipts_reject_spoof_or_pollution(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    inputs = (authorities["identity"],)
    result = _build(inputs)
    with pytest.raises(TypeError, match="issued only"):
        batch_wire.TrustedWireBatchV1()
    polluted_batch = deepcopy(result)
    object.__setattr__(polluted_batch, "batch_id", "phase2b_trusted_wire_batch_" + "0" * 64)
    with pytest.raises(ValueError, match="root drift"):
        batch_wire.verify_trusted_wire_batch_replay_v1(
            batch=polluted_batch,
            run_id=RUN_ID,
            key_sources=_keys(),
            authorities=inputs,
        )

    envelope = result.envelopes[0]
    with pytest.raises(ValueError, match="drifts"):
        replace(envelope, payload_sha256="0" * 64)
    decoded = batch_wire.decode_and_audit_trusted_envelope_v1(envelope.envelope)
    with pytest.raises(ValueError, match="drifts"):
        replace(decoded, padding_sha256="0" * 64)

    replay = batch_wire.verify_trusted_wire_batch_replay_v1(
        batch=result,
        run_id=RUN_ID,
        key_sources=_keys(),
        authorities=inputs,
    )
    assert replay.source_authority_content_ids == tuple(
        authority.content_id for authority in inputs
    )
    assert replay.replay_receipt_id.startswith(
        "phase2b_trusted_wire_secret_replay_"
    )
    with pytest.raises(TypeError, match="issued only"):
        batch_wire.TrustedWireReplayReceiptV1()
    object.__setattr__(replay, "replay_verified", False)
    with pytest.raises(ValueError, match="claim boundary"):
        replay._validate()  # noqa: SLF001 - deliberate post-issue pollution audit


def test_secret_replay_receipt_binds_source_order_and_exact_boolean_claims(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    inputs = (authorities["identity"], authorities["unit_conversion"])
    result = _build(inputs)
    replay = batch_wire.verify_trusted_wire_batch_replay_v1(
        batch=result,
        run_id=RUN_ID,
        key_sources=_keys(),
        authorities=inputs,
    )
    wrong_order = deepcopy(replay)
    object.__setattr__(
        wrong_order,
        "source_authority_content_ids",
        tuple(reversed(replay.source_authority_content_ids)),
    )
    with pytest.raises(ValueError, match="receipt root drift"):
        wrong_order._validate()  # noqa: SLF001 - deliberate pollution audit

    bool_pollution = deepcopy(replay)
    object.__setattr__(bool_pollution, "replay_verified", 1)
    with pytest.raises(TypeError, match="exact bool"):
        bool_pollution._validate()  # noqa: SLF001 - deliberate pollution audit


def test_second_authority_failure_abstains_atomically_without_first_envelope(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    valid = authorities["identity"]
    contract = valid.transform_contracts[0]
    forged = replace(
        valid,
        transform_contracts=(
            replace(
                contract,
                output_observations=(
                    replace(contract.output_observations[0], provenance_sha256="0" * 64),
                ),
            ),
        ),
    )
    result = batch_wire.build_trusted_wire_batch_v1(
        authorities=(valid, forged), run_id=RUN_ID, key_sources=_keys()
    )
    assert type(result) is batch_wire.TrustedWireBatchPreflightV1
    assert result.authority_count == 2
    assert not hasattr(result, "envelopes")


def test_fixed_65536_header_hashes_and_secret_padding_tamper_boundaries(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    row = _build((authorities["identity"],)).envelopes[0]
    assert len(row.envelope) == ENVELOPE_BYTES == 65_536
    payload, padding = _payload(row.envelope)
    assert row.payload_bytes == len(payload)
    assert row.padding_bytes == len(padding)
    assert row.padding_bytes >= 32
    assert row.payload_sha256 == hashlib.sha256(payload).hexdigest()
    assert row.padding_sha256 == hashlib.sha256(padding).hexdigest()
    for forged in (
        row.envelope[:-1],
        row.envelope[: HEADER.size] + bytes([row.envelope[HEADER.size] ^ 1]) + row.envelope[HEADER.size + 1 :],
        row.envelope[:-1] + bytes([row.envelope[-1] ^ 1]),
    ):
        with pytest.raises(ValueError):
            batch_wire.decode_and_audit_trusted_envelope_v1(forged)


def test_all_formal_origin_typed_holdout_and_c1_flags_remain_false(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    inputs = (authorities["identity"],)
    result = _build(inputs)
    row = result.envelopes[0]
    decoded = batch_wire.decode_and_audit_trusted_envelope_v1(row.envelope)
    replay = batch_wire.verify_trusted_wire_batch_replay_v1(
        batch=result,
        run_id=RUN_ID,
        key_sources=_keys(),
        authorities=inputs,
    )
    assert result.claim_level == NON_AUTHORITATIVE_CLAIM_LEVEL
    assert not result.typed_authority_decode_replay_implemented
    assert not result.formal_uuid_audit
    assert not result.formal_covert_audit
    assert not result.sealed_holdout_eligible
    assert not result.c1_exit_evidence
    for receipt in (row, decoded):
        assert not receipt.secret_padding_replay_verified
        assert not receipt.typed_authority_decode_replay_implemented
        assert not receipt.origin_authenticated
        assert not receipt.formal_covert_audit
    assert not replay.typed_authority_decode_replay_implemented
    assert not replay.origin_authenticated
    assert not replay.formal_covert_audit
    assert not replay.sealed_holdout_eligible


def test_frozen_policy_id_is_public_content_bound_and_carried_end_to_end(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    expected = (
        "phase2b_trusted_wire_batch_policy_"
        "26aa3ae0ec8c3f646ef4cebd9cb3334f9bd6a5fbc49697cd999a17b21bd56b86"
    )
    assert batch_wire.TRUSTED_WIRE_BATCH_POLICY_ID == expected
    assert batch_wire.DEFAULT_TRUSTED_WIRE_BATCH_POLICY.policy_id == expected
    inputs = (authorities["identity"],)
    result = _build(inputs)
    replay = batch_wire.verify_trusted_wire_batch_replay_v1(
        batch=result,
        run_id=RUN_ID,
        key_sources=_keys(),
        authorities=inputs,
    )
    assert result.policy_id == expected
    assert replay.policy_id == expected


def test_frozen_policy_exposes_every_resource_cap_at_exact_defaults() -> None:
    policy = batch_wire.DEFAULT_TRUSTED_WIRE_BATCH_POLICY
    expected = {
        "schema_version": "hegel-machine-phase2b-trusted-wire-batch-mechanics/2",
        "payload_schema_version": (
            "hegel-machine-phase2b-trusted-wire-batch-payload/2"
        ),
        "key_schedule_version": "hegel-machine-phase2b-trusted-wire-hkdf-sha256/1",
        "public_provenance_version": (
            "hegel-machine-phase2b-trusted-wire-public-provenance/2"
        ),
        "typed_authority_schema_id": (
            "phase2b_trusted_wire_typed_authority_schema_"
            "9429e96b9192db4546b92b011779e99352decb051a023d8883682539c804b730"
        ),
        "typed_authority_codec_version": (
            "hegel-machine-phase2b-trusted-wire-typed-authority-codec/1"
        ),
        "typed_authority_codec_policy_id": (
            "phase2b_trusted_wire_typed_authority_codec_policy_"
            "8fc5714399f0e31c43bf8fa3c818c4cc8afb8b4c49b57ae8af309821abfcc4b3"
        ),
        "exact_transform_validator_policy_id": (
            "phase2b_exact_transform_policy_"
            "c49a74a45af3e272d800fece85f6862ae557f0c3b071dded04f6b6c2b8a7862e"
        ),
        "exact_transform_provenance_compiler_version": (
            "hegel-machine-phase2b-exact-transform-provenance-compiler/1"
        ),
        "exact_transform_provenance_compiler_policy_id": (
            "phase2b_exact_transform_provenance_compiler_policy_"
            "36fee587bcb02e1f447c29d7b68f525f449088c43aab3b76623295a041146087"
        ),
        "maximum_authorities": 1_024,
        "run_id_bytes": 32,
        "ikm_bytes": 32,
        "maximum_uuid_collision_retries": 10,
        "maximum_shuffle_rejection_draws": 32,
        "envelope_bytes": 65_536,
        "header_bytes": 80,
        "minimum_padding_bytes": 32,
        "maximum_payload_bytes": 65_424,
        "maximum_profile_depth": 64,
        "maximum_profile_nodes": 16_384,
        "maximum_array_entries": 4_096,
        "maximum_ascii_string_bytes": 2_048,
        "maximum_uuid_occurrences": 2_048,
        "maximum_unique_uuids": 1_024,
        "maximum_safe_integer": 9_007_199_254_740_991,
        "maximum_rational_bit_length": 4_096,
        "maximum_observations_per_authority": 2_048,
        "maximum_entities_per_authority": 1_024,
        "maximum_contracts_per_authority": 256,
        "maximum_metadata_rows_per_authority": 2_048,
        "total_string_budget_multiplier": 256,
        "namespaces": (
            "aggregate_map",
            "bundle",
            "clock",
            "component",
            "context",
            "entity",
            "frame",
            "observation",
            "quantity",
            "quotient_class",
            "role_candidate",
            "scale",
            "source_channel",
            "task",
            "transform",
            "unit",
        ),
    }
    assert batch_wire.EXACT_TRANSFORM_VALIDATOR_POLICY_ID == expected[
        "exact_transform_validator_policy_id"
    ]
    assert (
        batch_wire.EXACT_TRANSFORM_VALIDATOR_POLICY_ID
        == transform_semantics.EXACT_TRANSFORM_POLICY_ID
    )
    assert {item.name: getattr(policy, item.name) for item in fields(policy)} == expected


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", "drift"),
        ("payload_schema_version", "drift"),
        ("key_schedule_version", "drift"),
        ("public_provenance_version", "drift"),
        ("typed_authority_schema_id", "drift"),
        ("typed_authority_codec_version", "drift"),
        ("typed_authority_codec_policy_id", "drift"),
        ("exact_transform_validator_policy_id", "drift"),
        ("exact_transform_provenance_compiler_version", "drift"),
        ("exact_transform_provenance_compiler_policy_id", "drift"),
        ("maximum_authorities", 1_023),
        ("run_id_bytes", 31),
        ("ikm_bytes", 31),
        ("maximum_uuid_collision_retries", 9),
        ("maximum_shuffle_rejection_draws", 31),
        ("envelope_bytes", 65_535),
        ("header_bytes", 79),
        ("minimum_padding_bytes", 31),
        ("maximum_payload_bytes", 65_423),
        ("maximum_profile_depth", 63),
        ("maximum_profile_nodes", 16_383),
        ("maximum_array_entries", 4_095),
        ("maximum_ascii_string_bytes", 2_047),
        ("maximum_uuid_occurrences", 2_047),
        ("maximum_unique_uuids", 1_023),
        ("maximum_safe_integer", 9_007_199_254_740_990),
        ("maximum_rational_bit_length", 4_095),
        ("maximum_observations_per_authority", 2_047),
        ("maximum_entities_per_authority", 1_023),
        ("maximum_contracts_per_authority", 255),
        ("maximum_metadata_rows_per_authority", 2_047),
        ("total_string_budget_multiplier", 255),
        ("namespaces", tuple(reversed(batch_wire.DEFAULT_TRUSTED_WIRE_BATCH_POLICY.namespaces))),
    ),
)
def test_every_frozen_policy_field_rejects_constructor_drift(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="policy drift"):
        replace(batch_wire.DEFAULT_TRUSTED_WIRE_BATCH_POLICY, **{field: value})


def test_polluted_batch_and_replay_policy_ids_fail_validation(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    inputs = (authorities["identity"],)
    result = _build(inputs)
    polluted_batch = deepcopy(result)
    object.__setattr__(
        polluted_batch,
        "policy_id",
        "phase2b_trusted_wire_batch_policy_" + "0" * 64,
    )
    with pytest.raises(ValueError, match="identity drift"):
        batch_wire.verify_trusted_wire_batch_replay_v1(
            batch=polluted_batch,
            run_id=RUN_ID,
            key_sources=_keys(),
            authorities=inputs,
        )
    replay = batch_wire.verify_trusted_wire_batch_replay_v1(
        batch=result,
        run_id=RUN_ID,
        key_sources=_keys(),
        authorities=inputs,
    )
    object.__setattr__(
        replay,
        "policy_id",
        "phase2b_trusted_wire_batch_policy_" + "0" * 64,
    )
    with pytest.raises(ValueError, match="policy drift"):
        replay._validate()  # noqa: SLF001 - deliberate post-issue pollution audit


def test_frozen_identity_batch_cryptographic_vector(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    result = _build((authorities["identity"],))
    row = result.envelopes[0]
    assert result.run_id_commitment == (
        "phase2b_trusted_wire_run_"
        "20403cd8ab4de999be1b6391561405de317ac008c38dfe71bda1e5cb604c9240"
    )
    assert result.batch_id == (
        "phase2b_trusted_wire_batch_"
        "0b558bbe6484e75635909e1a4bbd914db5fa58e5d77c9a7b62ec4a883f1e1d2b"
    )
    assert row.payload_sha256 == (
        "01a1c2bd289abcfd263c3f6dacf38d47d8e049d79efd897ee8be1fea1e36a61f"
    )
    assert row.padding_sha256 == (
        "5f37627a6f9d416c95e03f4264a8354535ed58474d8dbb0b5f2f88aafe792214"
    )
    assert row.envelope_id == (
        "phase2b_trusted_envelope_"
        "9008e222d08bcf21aa70942db5b06c40dbed36994da9324f88ea094df9d56e4c"
    )
    decoded = batch_wire.decode_and_audit_trusted_envelope_v1(row.envelope)
    first_by_namespace: dict[str, str] = {}
    for occurrence in decoded.namespace_audit.occurrences:
        first_by_namespace.setdefault(occurrence.namespace, occurrence.public_uuid)
    expected_public_ids = {
        "bundle": "08ef7b13-4712-4897-93e8-1dd26ba835b8",
        "component": "563c206d-b38b-414c-a9c3-3841c7be2fea",
        "entity": "2690aedc-dd7d-4bf0-80ca-a92a91c430dc",
        "observation": "739807e5-83ae-4473-b60d-3d0fa0375788",
        "quantity": "1506e7d4-00cf-4c06-93b0-a222ef1e169a",
        "role_candidate": "edbe650c-ec16-4729-8485-83c08d8d799b",
    }
    assert expected_public_ids.items() <= first_by_namespace.items()


@pytest.mark.parametrize(
    "name",
    (
        "identity",
        "unit_conversion",
        "coordinate_affine",
        "temporal_aggregation",
        "spatial_aggregation",
        "sampling_resolution",
        "equivalent_split_merge",
        "coarse_graining",
    ),
)
def test_renamed_authority_set_arrays_refs_rows_and_groups_are_recanonicalized(
    name: str,
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    authority = _authority(_build((authorities[name],)).envelopes[0])
    base = authority["base_bundle"]
    assert base["entity_candidates"] == sorted(
        base["entity_candidates"], key=lambda item: item["entity_id"]
    )
    assert base["role_ids"] == sorted(base["role_ids"])
    assert base["quantity_ids"] == sorted(base["quantity_ids"])
    assert base["observations"] == sorted(
        base["observations"], key=lambda item: item["observation_id"]
    )
    assert base["task_target"]["entity_ids"] == sorted(
        base["task_target"]["entity_ids"]
    )
    assert base["task_target"]["quantity_ids"] == sorted(
        base["task_target"]["quantity_ids"]
    )
    graph = base["aggregation_graph"]
    assert graph["scale_ids"] == sorted(graph["scale_ids"])
    assert graph["root_scale_ids"] == sorted(graph["root_scale_ids"])
    assert graph["edges"] == sorted(
        graph["edges"],
        key=lambda item: (
            item["source_scale_id"],
            item["target_scale_id"],
            item["transform_id"],
        ),
    )
    assert base["transform_catalog"] == sorted(
        base["transform_catalog"], key=lambda item: item["transform_id"]
    )
    assert base["missingness_mask"] == sorted(base["missingness_mask"])
    assert authority["observation_metadata"] == sorted(
        authority["observation_metadata"], key=lambda item: item["observation_id"]
    )
    assert authority["transform_contracts"] == sorted(
        authority["transform_contracts"], key=lambda item: item["transform_id"]
    )
    for contract in authority["transform_contracts"]:
        _assert_ref_array_sorted(contract["input_components"])
        assert contract["output_components"] == sorted(
            contract["output_components"], key=lambda item: _ref_key(item["ref"])
        )
        assert contract["output_observations"] == sorted(
            contract["output_observations"],
            key=lambda item: (item["scale_id"], item["observation_id"]),
        )
        for observation in contract["output_observations"]:
            assert observation["entity_ids"] == sorted(observation["entity_ids"])
            assert observation["role_candidate_ids"] == sorted(
                observation["role_candidate_ids"]
            )
            assert observation["source_observation_ids"] == sorted(
                observation["source_observation_ids"]
            )
            assert observation["component_refs"] == sorted(
                observation["component_refs"], key=lambda item: item["ordinal"]
            )
        assert contract["kernel_rows"] == sorted(
            contract["kernel_rows"], key=lambda item: _ref_key(item["output_ref"])
        )
        for row in contract["kernel_rows"]:
            assert row["terms"] == sorted(
                row["terms"], key=lambda item: _ref_key(item["input_ref"])
            )
        certificate = contract["certificate"]
        for row_name in (
            "inverse_rows",
            "source_commutation_rows",
            "target_commutation_rows",
        ):
            rows = certificate.get(row_name, [])
            assert rows == sorted(rows, key=lambda item: _ref_key(item["output_ref"]))
            for row in rows:
                assert row["terms"] == sorted(
                    row["terms"], key=lambda item: _ref_key(item["input_ref"])
                )
        for group in certificate.get("groups", []):
            _assert_ref_array_sorted(group["input_refs"])
            _assert_ref_array_sorted(group["output_refs"])


def test_all_eight_certificate_authorities_build_public_decode_and_secret_replay(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    expected = (
        "identity",
        "unit_conversion",
        "coordinate_affine",
        "temporal_aggregation",
        "spatial_aggregation",
        "sampling_resolution",
        "equivalent_split_merge",
        "coarse_graining",
    )
    for name in expected:
        authority = authorities[name]
        result = _build((authority,))
        assert len(result.envelopes) == 1
        envelope = result.envelopes[0]
        assert _operation(envelope) == name
        decoded = batch_wire.decode_and_audit_trusted_envelope_v1(envelope.envelope)
        assert decoded.public_provenance_verified
        assert decoded.structural_hashes_verified
        replay = batch_wire.verify_trusted_wire_batch_replay_v1(
            batch=result,
            run_id=RUN_ID,
            key_sources=_keys(),
            authorities=(authority,),
        )
        assert replay.replay_verified
        assert replay.authority_count == 1


def test_sampling_selected_input_and_exact_grid_pairing_survives_uuid_rename(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    authority = _authority(
        _build((authorities["sampling_resolution"],)).envelopes[0]
    )
    observations = {
        item["observation_id"]: item
        for item in authority["base_bundle"]["observations"]
    }
    certificate = authority["transform_contracts"][0]["certificate"]
    selected = certificate["selected_inputs"]
    points = certificate["grid_points"]
    assert len(selected) == len(points) == 2
    exact_points = [
        int(point[0]["numerator_decimal"]) / int(point[0]["denominator_decimal"])
        for point in points
    ]
    assert exact_points == sorted(exact_points) == [0.0, 1.0]
    for ref, exact_point in zip(selected, exact_points, strict=True):
        encoded_start = observations[ref["observation_id"]]["temporal_support"]["start"]
        observed_start = struct.unpack(">d", bytes.fromhex(encoded_start[6:]))[0]
        assert observed_start == exact_point


def test_coarse_group_quotient_pairing_and_all_nested_refs_survive_rename(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    authority = _authority(_build((authorities["coarse_graining"],)).envelopes[0])
    certificate = authority["transform_contracts"][0]["certificate"]
    groups = certificate["groups"]
    quotient_ids = certificate["quotient_class_ids"]
    assert len(groups) == len(quotient_ids) > 0
    assert all(UUID4.fullmatch(value) for value in quotient_ids)
    paired = list(zip(groups, quotient_ids, strict=True))
    assert len(paired) == len(groups)
    for group, quotient_id in paired:
        assert quotient_id
        _assert_ref_array_sorted(group["input_refs"])
        _assert_ref_array_sorted(group["output_refs"])


def test_latent_ids_and_old_array_rank_do_not_survive_public_rename(
    authorities: dict[str, PublicTransformEvidenceBundleV2],
) -> None:
    latent = authorities["sampling_resolution"]
    latent_text = json.dumps(latent.to_mapping(), sort_keys=True)
    latent_ids = set(re.findall(UUID4.pattern[1:-1], latent_text))
    assert latent_ids
    first = _build((latent,), key_sources=_keys(identifiers=b"I" * 32)).envelopes[0]
    second = _build((latent,), key_sources=_keys(identifiers=b"J" * 32)).envelopes[0]
    first_payload, _ = _payload(first.envelope)
    second_payload, _ = _payload(second.envelope)
    assert first_payload != second_payload
    assert all(value.encode("ascii") not in first_payload for value in latent_ids)
    assert all(value.encode("ascii") not in second_payload for value in latent_ids)
    for row in (first, second):
        public = _authority(row)
        public_observation_ids = [
            item["observation_id"]
            for item in public["base_bundle"]["observations"]
        ]
        assert public_observation_ids == sorted(public_observation_ids)
