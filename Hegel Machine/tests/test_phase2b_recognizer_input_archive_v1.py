"""Adversarial public-API tests for the frozen recognizer input archive."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import inspect
import runpy
from pathlib import Path
import struct
from typing import Iterator

import pytest

from hegel_machine.bootstrap import initial_theory
from hegel_machine.phase2b_adapter import (
    LawWireBinding,
    ObservableChannelBinding,
    Phase2BAdapterRegistry,
)
import hegel_machine.phase2b_exact_transform_semantics_v1 as transform
import hegel_machine.phase2b_recognizer_input_archive_v1 as recognizer_archive
import hegel_machine.phase2b_trusted_wire_batch_v1 as batch_wire
import hegel_machine.phase2b_trusted_wire_typed_authority_v1 as typed_codec
import hegel_machine.phase2b_trusted_wire_typed_replay_v1 as typed_replay
from hegel_machine.phase2b_trusted_wire_v1 import (
    decode_phase2b_jcs_profile_v1,
    encode_phase2b_jcs_profile_v1,
)
from hegel_machine.schema import LawKind


RUN_ID = b"R" * 32
LEGACY_IDENTITY_BATCH_ID = (
    "phase2b_trusted_wire_batch_"
    "0b558bbe6484e75635909e1a4bbd914db5fa58e5d77c9a7b62ec4a883f1e1d2b"
)
LEGACY_IDENTITY_ENVELOPE_SHA256 = (
    "8c8166d7d0aaebdfb2ad1b50e63f410de75dca1c1e7721178224cf9d209b8b52"
)
EXPECTED_FROZEN_VOCABULARY = (
    (
        "law_symmetry_v1",
        LawKind.SYMMETRY,
        "F01_symmetry_equivariance",
        ("source", "transformed_source"),
        ("common_codomains", "forward", "transformed"),
    ),
    (
        "law_monotonicity_v1",
        LawKind.MONOTONICITY,
        "F02_monotonicity_order",
        ("lower", "upper"),
        ("direction", "x_high", "x_low", "y_high", "y_low"),
    ),
    (
        "law_conservation_v1",
        LawKind.CONSERVATION,
        "F03_conservation_balance",
        ("sink", "source", "system"),
        (
            "boundary_observed",
            "inflows",
            "outflows",
            "sinks",
            "sources",
            "storage_delta",
        ),
    ),
    (
        "law_complementarity_v1",
        LawKind.COMPLEMENTARITY,
        "F04_additivity_complementarity",
        ("intervention_a", "intervention_b"),
        (
            "expected_interaction",
            "interaction_margin",
            "u_a",
            "u_ab",
            "u_b",
            "u_empty",
        ),
    ),
    (
        "law_negative_feedback_v1",
        LawKind.NEGATIVE_FEEDBACK,
        "F06_negative_feedback_stability",
        ("controlled_quantity", "disturbance", "response"),
        (
            "controlled_quantity_observed",
            "deviation_after_response",
            "deviation_before_response",
            "disturbance_delta",
            "disturbance_precedes_response",
            "local_stability_window_observed",
            "mitigation_margin",
            "response_delta",
            "response_margin",
            "same_controlled_quantity",
            "system_induced_response",
        ),
    ),
    (
        "law_locality_v1",
        LawKind.LOCALITY,
        "F05_locality_composition",
        ("markov_blanket", "outside_context", "target"),
        (
            "blanket_observed",
            "conditional_a",
            "conditional_b",
            "same_blanket_state",
        ),
    ),
)
EXPECTED_BRIDGE_FAMILY_IDS = {
    LawKind.SYMMETRY: "58351910-f1ea-4613-b5b2-47d9cc2f1652",
    LawKind.MONOTONICITY: "16ba12ce-f178-4226-ac97-2120adb62073",
    LawKind.CONSERVATION: "773faef6-c762-4ca6-b389-f2a593cb1f99",
    LawKind.COMPLEMENTARITY: "431cb872-0237-4751-a3f8-e5fc2a2a3b38",
    LawKind.NEGATIVE_FEEDBACK: "1d9fd5a5-ac24-4dd0-9b70-e257391585e5",
    LawKind.LOCALITY: "c4a5cad4-444f-4e54-a341-c21ffe29d2c5",
}
ARCHIVE_HEADER = struct.Struct(">8sHHII32s")


def _uid(index: int) -> str:
    return f"10000000-0000-4000-8000-{index:012x}"


def _keys(
    *,
    shuffle: bytes = b"S" * 32,
    identifiers: bytes = b"I" * 32,
    padding: bytes = b"P" * 32,
) -> batch_wire.TrustedWireKeySourcesV1:
    return batch_wire.TrustedWireKeySourcesV1(
        shuffle,
        identifiers,
        padding,
    )


@dataclass(frozen=True, slots=True)
class _RecognizerFixture:
    source_registry: Phase2BAdapterRegistry
    source_authorities: tuple[transform.PublicTransformEvidenceBundleV2, ...]
    trusted_batch: batch_wire.TrustedWireBatchV1
    typed_receipt: typed_replay.TypedTrustedWireBatchReplayV1


@dataclass(frozen=True, slots=True)
class _IssuedArchiveFixture:
    source: _RecognizerFixture
    source_cases: tuple[recognizer_archive.TrustedRecognizerSourceCaseV1, ...]
    decoded: recognizer_archive.DecodedRecognizerInputArchiveV1
    core_calls: int
    projection_calls: int


def _identity_authority() -> transform.PublicTransformEvidenceBundleV2:
    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_exact_transform_semantics_v1.py"))
    )
    authority = namespace["identity_authority"]()
    assert type(authority) is transform.PublicTransformEvidenceBundleV2
    return authority


def _legacy_stage_b_identity_authority(
) -> transform.PublicTransformEvidenceBundleV2:
    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_trusted_wire_v1.py"))
    )
    authority = namespace["identity_authority"]()
    assert type(authority) is transform.PublicTransformEvidenceBundleV2
    return authority


def _six_family_source_fixture() -> tuple[
    Phase2BAdapterRegistry,
    transform.PublicTransformEvidenceBundleV2,
]:
    """Make one small exact authority carrying the complete frozen vocabulary."""

    theory = initial_theory()
    authority = _identity_authority()

    semantic_roles = tuple(
        (law.law_id, role)
        for law in theory.relation_laws
        for role in law.roles
    )
    semantic_observables = tuple(
        sorted(
            {
                observable
                for law in theory.relation_laws
                for observable in law.required_observables
            }
        )
    )
    assert len(theory.relation_laws) == 6
    assert len(semantic_roles) == 15
    assert len(semantic_observables) == 35
    assert tuple(
        (
            law.law_id,
            law.kind,
            next(
                canonical
                for _, kind, canonical, _, _ in EXPECTED_FROZEN_VOCABULARY
                if kind is law.kind
            ),
            tuple(sorted(law.roles)),
            tuple(sorted(law.required_observables)),
        )
        for law in theory.relation_laws
    ) == EXPECTED_FROZEN_VOCABULARY

    family_ids = {
        kind: _uid(10_000 + ordinal)
        for ordinal, kind in enumerate(LawKind)
    }
    role_ids = {
        key: _uid(20_000 + ordinal)
        for ordinal, key in enumerate(semantic_roles)
    }
    quantity_ids = {
        observable: _uid(30_000 + ordinal)
        for ordinal, observable in enumerate(semantic_observables)
    }

    # The single real observation must be represented in the source registry.
    role_ids[semantic_roles[0]] = authority.base_bundle.role_ids[0]
    quantity_ids[semantic_observables[0]] = authority.base_bundle.quantity_ids[0]
    registry = Phase2BAdapterRegistry.from_theory(
        theory,
        family_ids=family_ids,
        role_ids=role_ids,
        quantity_ids=quantity_ids,
    )

    all_roles = tuple(sorted(role_ids.values()))
    all_quantities = tuple(sorted(quantity_ids.values()))
    base = authority.base_bundle
    expanded_base = replace(
        base,
        entity_candidates=tuple(
            replace(candidate, role_candidate_ids=all_roles)
            for candidate in base.entity_candidates
        ),
        role_ids=all_roles,
        quantity_ids=all_quantities,
        task_target=replace(base.task_target, quantity_ids=all_quantities),
    )
    expanded = transform.compile_exact_transform_provenance_v1(
        replace(authority, base_bundle=expanded_base)
    )
    compilation = transform.run_exact_transform_semantics(expanded)
    assert type(compilation) is transform.ExactTransformCompilation
    assert (
        compilation.disposition
        is transform.TransformCompilationDisposition.COMPLETE
    )
    return registry, expanded


@pytest.fixture(scope="module")
def six_family_source() -> tuple[
    Phase2BAdapterRegistry,
    transform.PublicTransformEvidenceBundleV2,
]:
    return _six_family_source_fixture()


@pytest.fixture(scope="module")
def one_case_fixture(
    six_family_source: tuple[
        Phase2BAdapterRegistry,
        transform.PublicTransformEvidenceBundleV2,
    ],
) -> _RecognizerFixture:
    registry, authority = six_family_source
    authorities = (authority,)
    trusted_batch = batch_wire.build_trusted_wire_batch_v1(
        authorities=authorities,
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(trusted_batch) is batch_wire.TrustedWireBatchV1
    receipt = typed_replay.replay_typed_trusted_wire_batch_v1(
        batch=trusted_batch,
        run_id=RUN_ID,
        key_sources=_keys(),
        authorities=authorities,
    )
    assert type(receipt) is typed_replay.TypedTrustedWireBatchReplayV1
    return _RecognizerFixture(registry, authorities, trusted_batch, receipt)


@pytest.fixture(scope="module")
def two_case_fixture(
    six_family_source: tuple[
        Phase2BAdapterRegistry,
        transform.PublicTransformEvidenceBundleV2,
    ],
) -> _RecognizerFixture:
    registry, first = six_family_source
    second = transform.compile_exact_transform_provenance_v1(
        replace(
            first,
            base_bundle=replace(first.base_bundle, bundle_id=_uid(90_000)),
        )
    )
    authorities = (first, second)
    trusted_batch = batch_wire.build_trusted_wire_batch_v1(
        authorities=authorities,
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(trusted_batch) is batch_wire.TrustedWireBatchV1
    receipt = typed_replay.replay_typed_trusted_wire_batch_v1(
        batch=trusted_batch,
        run_id=RUN_ID,
        key_sources=_keys(),
        authorities=authorities,
    )
    assert type(receipt) is typed_replay.TypedTrustedWireBatchReplayV1
    return _RecognizerFixture(registry, authorities, trusted_batch, receipt)


def _source_cases(
    source: _RecognizerFixture,
) -> tuple[recognizer_archive.TrustedRecognizerSourceCaseV1, ...]:
    return tuple(
        recognizer_archive.TrustedRecognizerSourceCaseV1(
            authority=authority,
            adapter_registry=source.source_registry,
        )
        for authority in source.source_authorities
    )


@pytest.fixture(scope="module")
def one_case_archive(
    one_case_fixture: _RecognizerFixture,
) -> _IssuedArchiveFixture:
    cases = _source_cases(one_case_fixture)
    original_core = batch_wire._build_trusted_wire_batch_core_v1
    original_public_builder = batch_wire.build_trusted_wire_batch_v1
    counters = {"core": 0, "projection": 0}

    def monitored_core(**kwargs: object) -> object:
        counters["core"] += 1
        compiler = kwargs["per_case_projection_compiler"]
        assert callable(compiler)

        def monitored_projection(*args: object, **inner_kwargs: object) -> object:
            counters["projection"] += 1
            return compiler(*args, **inner_kwargs)

        forwarded = dict(kwargs)
        forwarded["per_case_projection_compiler"] = monitored_projection
        return original_core(**forwarded)  # type: ignore[arg-type]

    def forbidden_public_builder(**_: object) -> object:
        raise AssertionError("archive issuer called the legacy public builder")

    batch_wire._build_trusted_wire_batch_core_v1 = monitored_core  # type: ignore[assignment]
    batch_wire.build_trusted_wire_batch_v1 = forbidden_public_builder  # type: ignore[assignment]
    try:
        result = recognizer_archive.issue_trusted_recognizer_input_archive_v1(
            typed_replay=one_case_fixture.typed_receipt,
            run_id=RUN_ID,
            key_sources=_keys(),
            source_cases=cases,
        )
    finally:
        batch_wire._build_trusted_wire_batch_core_v1 = original_core  # type: ignore[assignment]
        batch_wire.build_trusted_wire_batch_v1 = original_public_builder  # type: ignore[assignment]
    assert type(result) is recognizer_archive.DecodedRecognizerInputArchiveV1
    assert recognizer_archive.decode_public_recognizer_input_archive_v1(
        result.archive
    ) == result
    return _IssuedArchiveFixture(
        one_case_fixture,
        cases,
        result,
        counters["core"],
        counters["projection"],
    )


@pytest.fixture(scope="module")
def two_case_archive(
    two_case_fixture: _RecognizerFixture,
) -> _IssuedArchiveFixture:
    cases = _source_cases(two_case_fixture)
    result = recognizer_archive.issue_trusted_recognizer_input_archive_v1(
        typed_replay=two_case_fixture.typed_receipt,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=cases,
    )
    assert type(result) is recognizer_archive.DecodedRecognizerInputArchiveV1
    assert recognizer_archive.decode_public_recognizer_input_archive_v1(
        result.archive
    ) == result
    return _IssuedArchiveFixture(two_case_fixture, cases, result, 1, 2)


def _walk_public_values(value: object) -> Iterator[object]:
    """Walk dataclasses/containers without consulting private implementation state."""

    yield value
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _walk_public_values(getattr(value, field.name))
    elif type(value) is dict:
        for key, item in value.items():
            yield from _walk_public_values(key)
            yield from _walk_public_values(item)
    elif type(value) in (tuple, list):
        for item in value:
            yield from _walk_public_values(item)


def _walk_public_field_names(value: object) -> Iterator[str]:
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield field.name
            yield from _walk_public_field_names(getattr(value, field.name))
    elif type(value) is dict:
        for key, item in value.items():
            if type(key) is str:
                yield key
            yield from _walk_public_field_names(item)
    elif type(value) in (tuple, list):
        for item in value:
            yield from _walk_public_field_names(item)


def _source_wire_ids(fixture: _RecognizerFixture) -> frozenset[str]:
    return frozenset(
        value
        for source in (*fixture.source_authorities, fixture.source_registry)
        for value in _walk_public_values(source)
        if type(value) is str and len(value) == 36
    )


def _public_role_and_quantity_ids(
    authority: transform.PublicTransformEvidenceBundleV2,
) -> frozenset[str]:
    return frozenset(
        authority.base_bundle.role_ids + authority.base_bundle.quantity_ids
    )


def _assert_atomic_rejection(value: object, *, case_count: int) -> None:
    assert type(value) is recognizer_archive.TrustedRecognizerInputArchiveRejectionV1
    assert value.disposition is recognizer_archive.RecognizerInputArchiveDisposition.ABSTAIN
    assert value.case_count == case_count
    assert value.archive is None
    assert value.rows == ()
    assert value.public_registry_ids == ()


def _archive_metadata_and_body(
    archive: bytes,
) -> tuple[dict[str, object], bytes, tuple[object, ...]]:
    header = ARCHIVE_HEADER.unpack_from(archive, 0)
    magic, wire_version, flags, row_count, metadata_bytes, metadata_sha = header
    assert magic == recognizer_archive.ARCHIVE_MAGIC
    assert wire_version == recognizer_archive.ARCHIVE_WIRE_VERSION
    assert flags == 0
    offset = recognizer_archive.ARCHIVE_HEADER_BYTES
    metadata_payload = archive[offset : offset + metadata_bytes]
    assert hashlib.sha256(metadata_payload).digest() == metadata_sha
    metadata = decode_phase2b_jcs_profile_v1(metadata_payload)
    assert type(metadata) is dict
    return metadata, archive[offset + metadata_bytes :], header


def _reframe_archive_metadata(
    archive: bytes,
    metadata: dict[str, object],
) -> bytes:
    _, body, header = _archive_metadata_and_body(archive)
    magic, wire_version, flags, row_count, _, _ = header
    payload = encode_phase2b_jcs_profile_v1(metadata)
    return ARCHIVE_HEADER.pack(
        magic,
        wire_version,
        flags,
        row_count,
        len(payload),
        hashlib.sha256(payload).digest(),
    ) + payload + body


def _replace_first_registry_mapping(
    archive: bytes,
    mapping: dict[str, object],
) -> bytes:
    _, body, _ = _archive_metadata_and_body(archive)
    (registry_bytes,) = struct.unpack_from(">I", body, 0)
    suffix = body[4 + registry_bytes :]
    payload = encode_phase2b_jcs_profile_v1(mapping)
    metadata, _, _ = _archive_metadata_and_body(archive)
    prefix = _reframe_archive_metadata(archive, metadata)
    _, _, reframed_header = _archive_metadata_and_body(prefix)
    metadata_bytes = reframed_header[4]
    body_offset = recognizer_archive.ARCHIVE_HEADER_BYTES + metadata_bytes
    return (
        prefix[:body_offset]
        + struct.pack(">I", len(payload))
        + payload
        + suffix
    )


def _replace_exact_text(value: object, old: str, new: str) -> object:
    if type(value) is str:
        return new if value == old else value
    if type(value) is dict:
        return {
            _replace_exact_text(key, old, new): _replace_exact_text(item, old, new)
            for key, item in value.items()
        }
    if type(value) is list:
        return [_replace_exact_text(item, old, new) for item in value]
    return value


def _replace_authority_uuid(
    authority: transform.PublicTransformEvidenceBundleV2,
    old: str,
    new: str,
) -> transform.PublicTransformEvidenceBundleV2:
    profile = typed_codec.encode_typed_transform_authority_profile_v1(authority)
    replaced = _replace_exact_text(profile, old, new)
    assert type(replaced) is dict
    decoded = typed_codec.decode_typed_transform_authority_profile_v1(replaced)
    return transform.compile_exact_transform_provenance_v1(decoded)


def _replace_registry_uuid(
    registry: Phase2BAdapterRegistry,
    old: str,
    new: str,
) -> Phase2BAdapterRegistry:
    laws = tuple(
        replace(
            law,
            family_id=new if law.family_id == old else law.family_id,
            role_ids=tuple(
                (role, new if wire_id == old else wire_id)
                for role, wire_id in law.role_ids
            ),
        )
        for law in registry.law_bindings
    )
    channels = tuple(
        replace(
            channel,
            quantity_id=new if channel.quantity_id == old else channel.quantity_id,
        )
        for channel in registry.observable_channels
    )
    return replace(registry, law_bindings=laws, observable_channels=channels)


def test_legacy_public_stage_b_identity_is_byte_exact_without_projection_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_core = batch_wire._build_trusted_wire_batch_core_v1
    projection_arguments: list[object] = []

    def monitored_core(**kwargs: object) -> object:
        projection_arguments.append(kwargs["per_case_projection_compiler"])
        return original_core(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        batch_wire,
        "_build_trusted_wire_batch_core_v1",
        monitored_core,
    )
    result = batch_wire.build_trusted_wire_batch_v1(
        authorities=(_legacy_stage_b_identity_authority(),),
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(result) is batch_wire.TrustedWireBatchV1
    assert result.batch_id == LEGACY_IDENTITY_BATCH_ID
    assert len(result.envelopes) == 1
    assert hashlib.sha256(result.envelopes[0].envelope).hexdigest() == (
        LEGACY_IDENTITY_ENVELOPE_SHA256
    )
    assert projection_arguments == [None]


def test_one_case_fixture_is_direct_complete_and_registry_total(
    one_case_fixture: _RecognizerFixture,
) -> None:
    registry = one_case_fixture.source_registry
    authority = one_case_fixture.source_authorities[0]
    assert len(one_case_fixture.trusted_batch.envelopes) == 1
    assert len(one_case_fixture.typed_receipt.rows) == 1
    assert len(registry.law_bindings) == 6
    assert sum(len(binding.role_ids) for binding in registry.law_bindings) == 15
    assert len(registry.observable_channels) == 35
    assert {
        wire_id
        for binding in registry.law_bindings
        for _, wire_id in binding.role_ids
    } == set(authority.base_bundle.role_ids)
    assert {item.quantity_id for item in registry.observable_channels} == set(
        authority.base_bundle.quantity_ids
    )
    result = transform.run_exact_transform_semantics(
        one_case_fixture.typed_receipt.rows[0].authority
    )
    assert type(result) is transform.ExactTransformCompilation
    assert result.disposition is transform.TransformCompilationDisposition.COMPLETE


def test_fixture_covers_exact_six_family_registry_and_cross_case_unlinkability(
    two_case_fixture: _RecognizerFixture,
) -> None:
    registry = two_case_fixture.source_registry
    assert {binding.law_kind for binding in registry.law_bindings} == set(LawKind)
    assert len(registry.law_bindings) == 6
    assert sum(len(binding.role_ids) for binding in registry.law_bindings) == 15
    assert len(registry.observable_channels) == 35

    rows = two_case_fixture.typed_receipt.rows
    assert len(rows) == 2
    public_ids = tuple(_public_role_and_quantity_ids(row.authority) for row in rows)
    assert public_ids[0]
    assert public_ids[1]
    assert public_ids[0].isdisjoint(public_ids[1])

    # Both source cases intentionally reuse exactly the same registry IDs.  The
    # disjoint public sets above therefore exercise cross-case anti-linkability.
    first_source = _public_role_and_quantity_ids(
        two_case_fixture.source_authorities[0]
    )
    second_source = _public_role_and_quantity_ids(
        two_case_fixture.source_authorities[1]
    )
    assert first_source == second_source
    assert first_source.issubset(_source_wire_ids(two_case_fixture))

    source_wire_ids = _source_wire_ids(two_case_fixture)
    for row in rows:
        renamed_wire_ids = {
            value
            for value in _walk_public_values(row.authority)
            if type(value) is str and len(value) == 36
        }
        assert renamed_wire_ids
        assert source_wire_ids.isdisjoint(renamed_wire_ids)


def test_public_archive_api_and_field_manifests_are_closed() -> None:
    assert recognizer_archive.RECOGNIZER_INPUT_ARCHIVE_POLICY_ID == (
        "phase2b_recognizer_input_archive_policy_"
        "3587a8a8576aa217b292174931c47af10e63b7f0251fe8e34970a165a9e01ec2"
    )
    assert recognizer_archive.PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID == (
        "phase2b_public_recognizer_registry_schema_"
        "566c4dfba5190970677df83b4e7fb91fa58b4021df5373e1e659e221da065377"
    )
    assert recognizer_archive.PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID == (
        "phase2b_public_recognizer_family_alias_policy_"
        "73d40ef79d60716f62efb8d779d88d266c98983045074552509781c8bac3fc22"
    )
    issue = inspect.signature(
        recognizer_archive.issue_trusted_recognizer_input_archive_v1
    )
    assert tuple(issue.parameters) == (
        "typed_replay",
        "run_id",
        "key_sources",
        "source_cases",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in issue.parameters.values()
    )
    assert tuple(
        inspect.signature(
            recognizer_archive.decode_public_recognizer_input_archive_v1
        ).parameters
    ) == ("archive",)
    assert not hasattr(recognizer_archive, "TrustedRecognizerInputArchiveV1")
    assert "TrustedRecognizerInputArchiveV1" not in recognizer_archive.__all__

    assert tuple(
        item.name
        for item in fields(recognizer_archive.TrustedRecognizerSourceCaseV1)
    ) == ("authority", "adapter_registry")
    assert tuple(
        item.name
        for item in fields(recognizer_archive.TrustedRecognizerInputRowV1)
    ) == (
        "envelope",
        "envelope_id",
        "payload_sha256",
        "authority_content_id",
        "transform_result_id",
        "public_registry",
        "public_registry_id",
        "row_id",
    )
    assert tuple(
        item.name
        for item in fields(recognizer_archive.DecodedRecognizerInputArchiveV1)
    ) == (
        "disposition",
        "archive",
        "archive_id",
        "schema_version",
        "policy_id",
        "batch_id",
        "batch_policy_id",
        "run_id_commitment",
        "typed_replay_receipt_id",
        "secret_replay_receipt_id",
        "source_registry_id",
        "rows",
        "row_ids",
        "envelope_ids",
        "public_registry_ids",
        "authority_content_ids",
        "transform_result_ids",
        "claim_level",
        "structural_archive_verified",
        "row_bijection_verified",
        "registry_schema_verified",
        "direct_payload_transform_replay_verified",
        "batch_policy_membership_verified",
        "source_registry_projection_verified",
        "secret_custodian_replay_verified",
        "origin_authenticated",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "recognizer_executed",
        "prediction_archive_evaluated",
        "c1_exit_evidence",
    )


def test_one_case_issuer_is_single_live_projection_but_returns_only_safe_decode(
    one_case_archive: _IssuedArchiveFixture,
) -> None:
    source = one_case_archive.source
    result = one_case_archive.decoded
    assert one_case_archive.core_calls == 1
    assert one_case_archive.projection_calls == 1
    assert result.disposition is recognizer_archive.RecognizerInputArchiveDisposition.COMPLETE
    assert result.batch_id == source.trusted_batch.batch_id
    assert result.run_id_commitment == source.trusted_batch.run_id_commitment
    assert result.typed_replay_receipt_id == source.typed_receipt.receipt_id
    assert (
        result.secret_replay_receipt_id
        == source.typed_receipt.secret_replay_receipt_id
    )
    assert result.source_registry_id == source.source_registry.registry_id
    assert result.envelope_ids == tuple(
        row.envelope_id for row in source.typed_receipt.rows
    )
    assert result.authority_content_ids == source.typed_receipt.authority_content_ids
    assert result.transform_result_ids == source.typed_receipt.transform_result_ids
    assert len(result.rows) == len(result.row_ids) == len(result.public_registry_ids) == 1
    assert not hasattr(result, "receipt_id")

    assert all(
        (
            result.structural_archive_verified,
            result.row_bijection_verified,
            result.registry_schema_verified,
            result.direct_payload_transform_replay_verified,
        )
    )
    assert not any(
        (
            result.batch_policy_membership_verified,
            result.source_registry_projection_verified,
            result.secret_custodian_replay_verified,
            result.origin_authenticated,
            result.formal_covert_audit,
            result.sealed_holdout_eligible,
            result.recognizer_executed,
            result.prediction_archive_evaluated,
            result.c1_exit_evidence,
        )
    )


def test_public_registry_exactly_translates_six_frozen_families_and_scope(
    one_case_archive: _IssuedArchiveFixture,
) -> None:
    result = one_case_archive.decoded
    registry = result.rows[0].public_registry
    typed_authority = one_case_archive.source.typed_receipt.rows[0].authority
    by_kind = {item.law_kind: item for item in registry.law_bindings}
    assert tuple(item.law_kind for item in registry.law_bindings) == tuple(LawKind)
    assert len(by_kind) == 6

    semantic_role_keys: set[tuple[str, str]] = set()
    public_role_ids: set[str] = set()
    for law_id, law_kind, canonical_id, roles, observables in (
        EXPECTED_FROZEN_VOCABULARY
    ):
        row = by_kind[law_kind]
        assert row.law_id == law_id
        assert row.canonical_family_id.value == canonical_id
        assert row.bridge_family_id == EXPECTED_BRIDGE_FAMILY_IDS[law_kind]
        assert tuple(role for role, _ in row.role_ids) == roles
        assert row.required_observable_ids == observables
        for semantic_role, public_id in row.role_ids:
            assert (law_id, semantic_role) not in semantic_role_keys
            assert public_id not in public_role_ids
            semantic_role_keys.add((law_id, semantic_role))
            public_role_ids.add(public_id)
    assert len(semantic_role_keys) == 15
    assert public_role_ids == set(typed_authority.base_bundle.role_ids)

    channel_mapping = {
        item.observable_id: item.quantity_id
        for item in registry.observable_channels
    }
    assert len(channel_mapping) == 35
    assert set(channel_mapping) == {
        observable
        for *_, observables in EXPECTED_FROZEN_VOCABULARY
        for observable in observables
    }
    assert len(set(channel_mapping.values())) == 35
    assert set(channel_mapping.values()) == set(
        typed_authority.base_bundle.quantity_ids
    )

    adapter = registry.to_adapter_registry()
    assert adapter.theory_version_id == initial_theory().version_id
    assert adapter.maximum_candidate_count == 50_000
    assert {
        (law.law_id, law.law_kind, law.family_id)
        for law in adapter.law_bindings
    } == {
        (law_id, law_kind, EXPECTED_BRIDGE_FAMILY_IDS[law_kind])
        for law_id, law_kind, *_ in EXPECTED_FROZEN_VOCABULARY
    }


def test_two_case_public_ids_are_globally_unlinked_from_sources_and_each_other(
    two_case_archive: _IssuedArchiveFixture,
) -> None:
    result = two_case_archive.decoded
    assert len(result.rows) == 2
    source_ids = _source_wire_ids(two_case_archive.source)
    public_case_ids: list[set[str]] = []
    registry_scope_ids: list[set[str]] = []
    for row in result.rows:
        typed = typed_replay.decode_and_replay_typed_trusted_envelope_v1(
            row.envelope
        )
        public_authority_ids = {
            value
            for value in _walk_public_values(typed.authority)
            if type(value) is str and len(value) == 36
        }
        public_registry_ids = {
            value
            for value in _walk_public_values(row.public_registry)
            if type(value) is str and len(value) == 36
        }
        all_case_ids = public_authority_ids | public_registry_ids
        assert all_case_ids
        assert source_ids.isdisjoint(all_case_ids)
        public_case_ids.append(
            all_case_ids - set(EXPECTED_BRIDGE_FAMILY_IDS.values())
        )
        registry_scope_ids.append(
            {
                wire_id
                for law in row.public_registry.law_bindings
                for _, wire_id in law.role_ids
            }
            | {
                channel.quantity_id
                for channel in row.public_registry.observable_channels
            }
        )
    assert public_case_ids[0].isdisjoint(public_case_ids[1])
    assert registry_scope_ids[0].isdisjoint(registry_scope_ids[1])


def test_public_archive_recursively_excludes_private_inputs_and_source_roots(
    two_case_archive: _IssuedArchiveFixture,
) -> None:
    result = two_case_archive.decoded
    source = two_case_archive.source
    source_ids = _source_wire_ids(source)
    source_roots = {
        authority.content_id for authority in source.source_authorities
    }
    values = tuple(_walk_public_values(result))
    strings = {item for item in values if type(item) is str}
    assert source_ids.isdisjoint(strings)
    assert source_roots.isdisjoint(strings)
    assert all(type(item) is not typed_replay.TypedTrustedWireBatchReplayV1 for item in values)
    assert all(type(item) is not transform.PublicTransformEvidenceBundleV2 for item in values)
    assert all(type(item) is not Phase2BAdapterRegistry for item in values)

    for forbidden in source_ids | source_roots:
        assert forbidden.encode("ascii") not in result.archive
    for secret in (RUN_ID, b"S" * 32, b"I" * 32, b"P" * 32):
        assert secret not in result.archive

    field_names = set(_walk_public_field_names(result))
    for forbidden_name in (
        "source_authorities",
        "source_authority_content_ids",
        "source_index",
        "output_source_indices",
        "renamings",
        "allocation_state",
        "key_sources",
        "run_id",
        "shuffle_ikm",
        "id_ikm",
        "padding_ikm",
    ):
        assert forbidden_name not in field_names
        assert all(forbidden_name not in item for item in strings)
        assert (
            b'"' + forbidden_name.encode("ascii") + b'":'
            not in result.archive
        )
    assert "source_registry_id" in field_names
    assert result.source_registry_id == source.source_registry.registry_id
    assert b'"execution_commitment":' in result.archive
    assert b'"run_id_commitment":' not in result.archive


def test_exact_top_level_misuse_raises_before_any_archive_root(
    one_case_fixture: _RecognizerFixture,
) -> None:
    cases = _source_cases(one_case_fixture)
    valid = {
        "typed_replay": one_case_fixture.typed_receipt,
        "run_id": RUN_ID,
        "key_sources": _keys(),
        "source_cases": cases,
    }
    mutations = (
        {"typed_replay": object()},
        {"run_id": bytearray(RUN_ID)},
        {"key_sources": object()},
        {"source_cases": list(cases)},
        {"source_cases": (object(),)},
    )
    for mutation in mutations:
        arguments = dict(valid)
        arguments.update(mutation)
        with pytest.raises(TypeError, match="exact|exact Stage-C|source case"):
            recognizer_archive.issue_trusted_recognizer_input_archive_v1(
                **arguments  # type: ignore[arg-type]
            )
    with pytest.raises(TypeError, match="exact bytes"):
        recognizer_archive.decode_public_recognizer_input_archive_v1(
            bytearray(b"archive")  # type: ignore[arg-type]
        )


def test_coherent_public_metadata_forgery_remains_false_claim_raw_decode(
    one_case_archive: _IssuedArchiveFixture,
) -> None:
    original = one_case_archive.decoded
    metadata, _, _ = _archive_metadata_and_body(original.archive)
    metadata["typed_replay_receipt_id"] = (
        "phase2b_typed_trusted_wire_batch_replay_" + "0" * 64
    )
    metadata["secret_replay_receipt_id"] = (
        "phase2b_trusted_wire_secret_replay_" + "0" * 64
    )
    metadata["source_registry_id"] = "phase2b_adapter_registry_" + "0" * 64
    forged_archive = _reframe_archive_metadata(original.archive, metadata)
    forged = recognizer_archive.decode_public_recognizer_input_archive_v1(
        forged_archive
    )
    assert type(forged) is recognizer_archive.DecodedRecognizerInputArchiveV1
    assert forged.archive_id != original.archive_id
    assert forged.row_ids == original.row_ids
    assert forged.typed_replay_receipt_id == metadata["typed_replay_receipt_id"]
    assert not any(
        (
            forged.batch_policy_membership_verified,
            forged.source_registry_projection_verified,
            forged.secret_custodian_replay_verified,
            forged.origin_authenticated,
            forged.formal_covert_audit,
            forged.sealed_holdout_eligible,
            forged.recognizer_executed,
            forged.prediction_archive_evaluated,
            forged.c1_exit_evidence,
        )
    )


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_metadata_and_registry_wire_mappings_are_exact_closed_schemas(
    one_case_archive: _IssuedArchiveFixture,
    mutation: str,
) -> None:
    archive = one_case_archive.decoded.archive
    metadata, body, _ = _archive_metadata_and_body(archive)
    if mutation == "missing":
        metadata.pop("source_registry_id")
    else:
        metadata["source_index"] = 0
    with pytest.raises((TypeError, ValueError), match="field|schema"):
        recognizer_archive.decode_public_recognizer_input_archive_v1(
            _reframe_archive_metadata(archive, metadata)
        )

    (registry_bytes,) = struct.unpack_from(">I", body, 0)
    registry = decode_phase2b_jcs_profile_v1(body[4 : 4 + registry_bytes])
    assert type(registry) is dict
    if mutation == "missing":
        registry.pop("family_alias_policy_id")
    else:
        registry["old_to_new_map"] = []
    with pytest.raises((TypeError, ValueError), match="field|schema"):
        recognizer_archive.decode_public_recognizer_input_archive_v1(
            _replace_first_registry_mapping(archive, registry)
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "theory",
        "missing_law",
        "extra_law",
        "semantic_law",
        "semantic_role",
        "semantic_observable",
        "extra_channel",
        "candidate_cap",
    ),
)
def test_source_registry_drift_abstains_before_live_allocation(
    one_case_fixture: _RecognizerFixture,
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = deepcopy(_source_cases(one_case_fixture)[0])
    registry = case.adapter_registry
    if mutation == "theory":
        object.__setattr__(registry, "theory_version_id", "theory_" + "0" * 64)
    elif mutation == "missing_law":
        object.__setattr__(registry, "law_bindings", registry.law_bindings[:-1])
    elif mutation == "extra_law":
        extra = replace(
            registry.law_bindings[0],
            law_id="law_extra_v1",
            family_id=_uid(99_000),
        )
        object.__setattr__(registry, "law_bindings", (*registry.law_bindings, extra))
    elif mutation == "semantic_law":
        first = replace(registry.law_bindings[0], law_id="law_semantic_drift_v1")
        object.__setattr__(
            registry,
            "law_bindings",
            (first, *registry.law_bindings[1:]),
        )
    elif mutation == "semantic_role":
        first_law = registry.law_bindings[0]
        roles = list(first_law.role_ids)
        roles[0] = ("source_drift", roles[0][1])
        first = replace(first_law, role_ids=tuple(sorted(roles)))
        object.__setattr__(
            registry,
            "law_bindings",
            (first, *registry.law_bindings[1:]),
        )
    elif mutation == "semantic_observable":
        channels = list(registry.observable_channels)
        channels[0] = replace(channels[0], observable_id="drift_observable")
        object.__setattr__(
            registry,
            "observable_channels",
            tuple(sorted(channels, key=lambda item: item.observable_id)),
        )
    elif mutation == "extra_channel":
        extra = ObservableChannelBinding(_uid(99_001), "extra_observable")
        object.__setattr__(
            registry,
            "observable_channels",
            tuple(
                sorted(
                    (*registry.observable_channels, extra),
                    key=lambda item: item.observable_id,
                )
            ),
        )
    else:
        object.__setattr__(registry, "maximum_candidate_count", 49_999)

    def forbidden_core(**_: object) -> object:
        raise AssertionError("live allocation ran after source-registry rejection")

    monkeypatch.setattr(
        batch_wire,
        "_build_trusted_wire_batch_core_v1",
        forbidden_core,
    )
    result = recognizer_archive.issue_trusted_recognizer_input_archive_v1(
        typed_replay=one_case_fixture.typed_receipt,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=(case,),
    )
    _assert_atomic_rejection(result, case_count=1)


def test_distinct_per_case_source_registries_and_source_order_are_not_normalized(
    two_case_fixture: _RecognizerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = list(_source_cases(two_case_fixture))
    changed_laws = list(cases[1].adapter_registry.law_bindings)
    changed_laws[0] = replace(changed_laws[0], family_id=_uid(99_010))
    changed_registry = replace(
        cases[1].adapter_registry,
        law_bindings=tuple(changed_laws),
    )
    cases[1] = recognizer_archive.TrustedRecognizerSourceCaseV1(
        cases[1].authority,
        changed_registry,
    )

    def forbidden_core(**_: object) -> object:
        raise AssertionError("allocation ran after source order/registry drift")

    monkeypatch.setattr(
        batch_wire,
        "_build_trusted_wire_batch_core_v1",
        forbidden_core,
    )
    mismatch = recognizer_archive.issue_trusted_recognizer_input_archive_v1(
        typed_replay=two_case_fixture.typed_receipt,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=tuple(cases),
    )
    _assert_atomic_rejection(mismatch, case_count=2)

    reversed_order = recognizer_archive.issue_trusted_recognizer_input_archive_v1(
        typed_replay=two_case_fixture.typed_receipt,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=tuple(reversed(_source_cases(two_case_fixture))),
    )
    _assert_atomic_rejection(reversed_order, case_count=2)


@pytest.mark.parametrize("drift", ("run", "key"))
def test_run_and_key_drift_abstain_without_partial_archive(
    one_case_fixture: _RecognizerFixture,
    drift: str,
) -> None:
    result = recognizer_archive.issue_trusted_recognizer_input_archive_v1(
        typed_replay=one_case_fixture.typed_receipt,
        run_id=b"X" * 32 if drift == "run" else RUN_ID,
        key_sources=(
            _keys(identifiers=b"J" * 32) if drift == "key" else _keys()
        ),
        source_cases=_source_cases(one_case_fixture),
    )
    _assert_atomic_rejection(result, case_count=1)


def test_stage_c_receipt_or_batch_root_pollution_abstains_atomically(
    one_case_fixture: _RecognizerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polluted = deepcopy(one_case_fixture.typed_receipt)
    object.__setattr__(
        polluted.batch,
        "batch_id",
        "phase2b_trusted_wire_batch_" + "0" * 64,
    )

    def forbidden_core(**_: object) -> object:
        raise AssertionError("allocation ran after Stage-C receipt rejection")

    monkeypatch.setattr(
        batch_wire,
        "_build_trusted_wire_batch_core_v1",
        forbidden_core,
    )
    result = recognizer_archive.issue_trusted_recognizer_input_archive_v1(
        typed_replay=polluted,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=_source_cases(one_case_fixture),
    )
    _assert_atomic_rejection(result, case_count=1)


@pytest.mark.parametrize("location", ("family", "role", "quantity"))
def test_fixed_public_family_alias_collision_with_any_source_namespace_abstains(
    six_family_source: tuple[
        Phase2BAdapterRegistry,
        transform.PublicTransformEvidenceBundleV2,
    ],
    location: str,
) -> None:
    registry, authority = six_family_source
    alias = EXPECTED_BRIDGE_FAMILY_IDS[LawKind.LOCALITY]
    if location == "family":
        old = registry.law_bindings[0].family_id
    elif location == "role":
        old = max(
            wire_id
            for law in registry.law_bindings
            for _, wire_id in law.role_ids
        )
        authority = _replace_authority_uuid(authority, old, alias)
    else:
        old = max(item.quantity_id for item in registry.observable_channels)
        authority = _replace_authority_uuid(authority, old, alias)
    registry = _replace_registry_uuid(registry, old, alias)
    source_authorities = (authority,)
    batch = batch_wire.build_trusted_wire_batch_v1(
        authorities=source_authorities,
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(batch) is batch_wire.TrustedWireBatchV1
    replay = typed_replay.replay_typed_trusted_wire_batch_v1(
        batch=batch,
        run_id=RUN_ID,
        key_sources=_keys(),
        authorities=source_authorities,
    )
    assert type(replay) is typed_replay.TypedTrustedWireBatchReplayV1
    source_case = recognizer_archive.TrustedRecognizerSourceCaseV1(
        authority,
        registry,
    )
    result = recognizer_archive.issue_trusted_recognizer_input_archive_v1(
        typed_replay=replay,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=(source_case,),
    )
    _assert_atomic_rejection(result, case_count=1)


def test_hmac_public_uuid_collision_with_global_source_set_abstains(
    two_case_fixture: _RecognizerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = two_case_fixture.source_registry.law_bindings[0].role_ids[0][1]
    original_candidate = batch_wire._uuid4_candidate

    def colliding_candidate(
        key: bytes,
        run_id: bytes,
        namespace: str,
        counter: int,
        retry: int,
    ) -> str:
        if namespace == "role_candidate" and counter == 0 and retry == 0:
            return target
        return original_candidate(key, run_id, namespace, counter, retry)

    monkeypatch.setattr(batch_wire, "_uuid4_candidate", colliding_candidate)
    authorities = two_case_fixture.source_authorities
    batch = batch_wire.build_trusted_wire_batch_v1(
        authorities=authorities,
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(batch) is batch_wire.TrustedWireBatchV1
    replay = typed_replay.replay_typed_trusted_wire_batch_v1(
        batch=batch,
        run_id=RUN_ID,
        key_sources=_keys(),
        authorities=authorities,
    )
    assert type(replay) is typed_replay.TypedTrustedWireBatchReplayV1
    assert any(
        target
        in {
            value
            for value in _walk_public_values(row.authority)
            if type(value) is str
        }
        for row in replay.rows
    )
    result = recognizer_archive.issue_trusted_recognizer_input_archive_v1(
        typed_replay=replay,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=_source_cases(two_case_fixture),
    )
    _assert_atomic_rejection(result, case_count=2)


def test_projection_callback_failure_never_returns_a_partial_archive(
    one_case_fixture: _RecognizerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rejected_projection(**_: object) -> object:
        raise ValueError("test projection rejection")

    monkeypatch.setattr(
        recognizer_archive,
        "_compile_public_registry_from_live_renamings",
        rejected_projection,
    )
    result = recognizer_archive.issue_trusted_recognizer_input_archive_v1(
        typed_replay=one_case_fixture.typed_receipt,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=_source_cases(one_case_fixture),
    )
    _assert_atomic_rejection(result, case_count=1)


def test_source_count_and_global_uuid_caps_precede_live_allocation(
    one_case_fixture: _RecognizerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _source_cases(one_case_fixture)[0]

    def forbidden_core(**_: object) -> object:
        raise AssertionError("live allocation ran after a public resource cap")

    monkeypatch.setattr(
        batch_wire,
        "_build_trusted_wire_batch_core_v1",
        forbidden_core,
    )
    oversized = recognizer_archive.issue_trusted_recognizer_input_archive_v1(
        typed_replay=one_case_fixture.typed_receipt,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=(case,) * 1_025,
    )
    _assert_atomic_rejection(oversized, case_count=1_025)

    monkeypatch.setattr(recognizer_archive, "MAXIMUM_GLOBAL_SOURCE_UUIDS", 1)
    capped = recognizer_archive.issue_trusted_recognizer_input_archive_v1(
        typed_replay=one_case_fixture.typed_receipt,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=(case,),
    )
    _assert_atomic_rejection(capped, case_count=1)


def test_raw_archive_byte_cap_precedes_hashing(
    one_case_archive: _IssuedArchiveFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recognizer_archive, "MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES", 1)

    def forbidden_hash(*_: object, **__: object) -> object:
        raise AssertionError("archive hashing ran before the byte cap")

    monkeypatch.setattr(recognizer_archive.hashlib, "sha256", forbidden_hash)
    with pytest.raises(ValueError, match="byte budget"):
        recognizer_archive.decode_public_recognizer_input_archive_v1(
            one_case_archive.decoded.archive
        )


def test_public_rows_registries_and_decoded_receipts_cannot_be_forged() -> None:
    for receipt_type in (
        recognizer_archive.PublicRecognizerLawBindingV1,
        recognizer_archive.PublicRecognizerObservableChannelV1,
        recognizer_archive.PublicRecognizerRegistryV1,
        recognizer_archive.TrustedRecognizerInputRowV1,
        recognizer_archive.DecodedRecognizerInputArchiveV1,
    ):
        with pytest.raises(TypeError, match="privately issued"):
            receipt_type()


def test_public_registry_nested_subclasses_cannot_reuse_valid_fields(
    one_case_archive: _IssuedArchiveFixture,
) -> None:
    source_registry = one_case_archive.decoded.rows[0].public_registry
    source_law = source_registry.law_bindings[0]
    source_channel = source_registry.observable_channels[0]

    class LawSpoof(recognizer_archive.PublicRecognizerLawBindingV1):
        pass

    class ChannelSpoof(recognizer_archive.PublicRecognizerObservableChannelV1):
        pass

    class RegistrySpoof(recognizer_archive.PublicRecognizerRegistryV1):
        pass

    def copy_as_subclass(source: object, subclass: type[object]) -> object:
        polluted = object.__new__(subclass)
        for item in fields(source):
            object.__setattr__(polluted, item.name, getattr(source, item.name))
        return polluted

    polluted_law = copy_as_subclass(source_law, LawSpoof)
    with pytest.raises(TypeError, match="exact type"):
        polluted_law._validate()

    polluted_channel = copy_as_subclass(source_channel, ChannelSpoof)
    with pytest.raises(TypeError, match="exact type"):
        polluted_channel._validate()

    polluted_registry = copy_as_subclass(source_registry, RegistrySpoof)
    with pytest.raises(TypeError, match="exact type"):
        polluted_registry._validate()
    with pytest.raises(TypeError, match="exact type"):
        _ = polluted_registry.registry_id


def test_private_validation_contexts_require_internal_tokens(
    one_case_archive: _IssuedArchiveFixture,
) -> None:
    row = one_case_archive.decoded.rows[0]
    typed = typed_replay.decode_and_replay_typed_trusted_envelope_v1(row.envelope)
    with pytest.raises(TypeError, match="token"):
        row._validate(typed_replay=typed)

    parsed = recognizer_archive._parse_public_archive(
        one_case_archive.decoded.archive
    )
    with pytest.raises(TypeError, match="token"):
        one_case_archive.decoded._validate(parsed=parsed)


@pytest.mark.parametrize("field_name", ("rows", "public_registry_ids"))
@pytest.mark.parametrize("falsy_spoof", ([], None, 0, ""))
def test_rejection_partial_roots_require_exact_empty_tuples(
    field_name: str,
    falsy_spoof: object,
) -> None:
    values = {
        "disposition": recognizer_archive.RecognizerInputArchiveDisposition.ABSTAIN,
        "reason": "test_rejection",
        "case_count": 1,
        "batch_id": None,
        field_name: falsy_spoof,
    }
    with pytest.raises(ValueError, match="partial output"):
        recognizer_archive.TrustedRecognizerInputArchiveRejectionV1(**values)


@pytest.mark.parametrize("field_name", ("rows", "public_registry_ids"))
def test_rejection_empty_roots_reject_tuple_subclasses(field_name: str) -> None:
    class TupleSpoof(tuple):
        pass

    values = {
        "disposition": recognizer_archive.RecognizerInputArchiveDisposition.ABSTAIN,
        "reason": "test_rejection",
        "case_count": 1,
        "batch_id": None,
        field_name: TupleSpoof(),
    }
    with pytest.raises((TypeError, ValueError), match="exact empty tuple|partial output"):
        recognizer_archive.TrustedRecognizerInputArchiveRejectionV1(**values)


def test_every_decoded_claim_requires_an_exact_bool(
    one_case_archive: _IssuedArchiveFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_parse(_: object) -> object:
        raise AssertionError("archive parse ran before exact claim-type closure")

    monkeypatch.setattr(
        recognizer_archive,
        "_parse_public_archive",
        forbidden_parse,
    )
    claims = (
        "structural_archive_verified",
        "row_bijection_verified",
        "registry_schema_verified",
        "direct_payload_transform_replay_verified",
        "batch_policy_membership_verified",
        "source_registry_projection_verified",
        "secret_custodian_replay_verified",
        "origin_authenticated",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "recognizer_executed",
        "prediction_archive_evaluated",
        "c1_exit_evidence",
    )
    for field_name in claims:
        polluted = deepcopy(one_case_archive.decoded)
        original = getattr(polluted, field_name)
        assert type(original) is bool
        object.__setattr__(polluted, field_name, int(original))
        with pytest.raises(TypeError, match="exact bool"):
            polluted._validate()

    polluted = deepcopy(one_case_archive.decoded)
    object.__setattr__(polluted, "disposition", "COMPLETE")
    with pytest.raises(ValueError, match="disposition drift"):
        polluted._validate()


def test_all_public_scalar_and_tuple_roots_require_exact_builtin_types(
    one_case_archive: _IssuedArchiveFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StrSpoof(str):
        pass

    class TupleSpoof(tuple):
        pass

    def forbidden_parse(_: object) -> object:
        raise AssertionError("archive parse ran before exact receipt-type closure")

    monkeypatch.setattr(
        recognizer_archive,
        "_parse_public_archive",
        forbidden_parse,
    )
    decoded_text_fields = (
        "archive_id",
        "schema_version",
        "policy_id",
        "batch_id",
        "batch_policy_id",
        "run_id_commitment",
        "typed_replay_receipt_id",
        "secret_replay_receipt_id",
        "source_registry_id",
        "claim_level",
    )
    for field_name in decoded_text_fields:
        polluted = deepcopy(one_case_archive.decoded)
        original = getattr(polluted, field_name)
        object.__setattr__(polluted, field_name, StrSpoof(original))
        with pytest.raises((TypeError, ValueError), match="exact|type|root|receipt"):
            polluted._validate()

    decoded_tuple_fields = (
        "rows",
        "row_ids",
        "envelope_ids",
        "public_registry_ids",
        "authority_content_ids",
        "transform_result_ids",
    )
    for field_name in decoded_tuple_fields:
        polluted = deepcopy(one_case_archive.decoded)
        original = getattr(polluted, field_name)
        object.__setattr__(polluted, field_name, TupleSpoof(original))
        with pytest.raises((TypeError, ValueError), match="exact|tuple|root|receipt"):
            polluted._validate()
    for field_name in decoded_tuple_fields[1:]:
        polluted = deepcopy(one_case_archive.decoded)
        original = getattr(polluted, field_name)
        object.__setattr__(
            polluted,
            field_name,
            (StrSpoof(original[0]), *original[1:]),
        )
        with pytest.raises((TypeError, ValueError), match="exact|tuple|root|receipt"):
            polluted._validate()

    def forbidden_envelope_decode(_: object) -> object:
        raise AssertionError("envelope replay ran before exact row-type closure")

    monkeypatch.setattr(
        recognizer_archive,
        "decode_and_replay_typed_trusted_envelope_v1",
        forbidden_envelope_decode,
    )
    source_row = one_case_archive.decoded.rows[0]
    for field_name in (
        "envelope_id",
        "payload_sha256",
        "authority_content_id",
        "transform_result_id",
        "public_registry_id",
        "row_id",
    ):
        row = deepcopy(source_row)
        original = getattr(row, field_name)
        object.__setattr__(row, field_name, StrSpoof(original))
        with pytest.raises((TypeError, ValueError), match="exact|type|root|row"):
            row._validate()

    def forbidden_registry_encode(_: object) -> object:
        raise AssertionError("registry encoding ran before exact type closure")

    monkeypatch.setattr(
        recognizer_archive,
        "_encode_public_registry",
        forbidden_registry_encode,
    )
    source_registry = source_row.public_registry
    for field_name in (
        "schema_version",
        "theory_version_id",
        "family_alias_policy_id",
    ):
        registry = deepcopy(source_registry)
        original = getattr(registry, field_name)
        object.__setattr__(registry, field_name, StrSpoof(original))
        with pytest.raises((TypeError, ValueError), match="exact|type|drift|prefix"):
            registry._validate()
    for field_name in ("law_bindings", "observable_channels"):
        registry = deepcopy(source_registry)
        original = getattr(registry, field_name)
        object.__setattr__(registry, field_name, TupleSpoof(original))
        with pytest.raises((TypeError, ValueError), match="exact|type|tuple|invalid"):
            registry._validate()


@pytest.mark.parametrize(
    "field_name,replacement",
    (
        ("batch_id", "phase2b_trusted_wire_batch_" + "0" * 64),
        ("run_id_commitment", "phase2b_trusted_wire_run_" + "0" * 64),
        (
            "typed_replay_receipt_id",
            "phase2b_typed_trusted_wire_batch_replay_" + "0" * 64,
        ),
        (
            "secret_replay_receipt_id",
            "phase2b_trusted_wire_secret_replay_" + "0" * 64,
        ),
        ("source_registry_id", "phase2b_adapter_registry_" + "0" * 64),
        ("row_ids", ("phase2b_recognizer_input_row_" + "0" * 64,)),
        (
            "public_registry_ids",
            ("phase2b_public_recognizer_registry_" + "0" * 64,),
        ),
        (
            "authority_content_ids",
            ("phase2b_public_transform_evidence_" + "0" * 64,),
        ),
        (
            "transform_result_ids",
            ("phase2b_exact_transform_result_" + "0" * 64,),
        ),
    ),
)
def test_decoded_receipt_binds_every_public_archive_root(
    one_case_archive: _IssuedArchiveFixture,
    field_name: str,
    replacement: object,
) -> None:
    polluted = deepcopy(one_case_archive.decoded)
    object.__setattr__(polluted, field_name, replacement)
    with pytest.raises(ValueError, match="receipt drift"):
        polluted._validate()
