"""Adversarial tests for the non-authoritative recognizer prediction archive."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from copy import deepcopy
from fractions import Fraction
import hashlib
import inspect
from pathlib import Path
import re
import runpy
from typing import Callable

import pytest

from hegel_machine.bootstrap import initial_theory
import hegel_machine.phase2b_exact_derived_witness_bridge_v1 as derived_bridge
import hegel_machine.phase2b_recognizer_input_archive_v1 as recognizer_input
import hegel_machine.phase2b_recognizer_prediction_archive_v1 as prediction_archive
import hegel_machine.phase2b_runner as runner
import hegel_machine.phase2b_trusted_wire_batch_v1 as trusted_wire
import hegel_machine.phase2b_trusted_wire_typed_authority_v1 as typed_authority
import hegel_machine.phase2b_trusted_wire_typed_replay_v1 as typed_replay
import hegel_machine.phase2b_trusted_wire_v1 as trusted_wire_codec
import hegel_machine.phase2b_unsealed_prediction_evaluator_v1 as unsealed_evaluator
from hegel_machine.phase2b_protocol import ExecutionFreezeManifest
from hegel_machine.phase2b_wire import (
    PREDICTION_SCHEMA_VERSION,
    PredictionBundle,
    PredictionDisposition,
    PredictionReason,
    RoleBinding,
)
from hegel_machine.schema import LawKind


RUN_ID = b"R" * 32


@dataclass(frozen=True, slots=True)
class _OneRowPredictionFixture:
    input_archive: recognizer_input.DecodedRecognizerInputArchiveV1
    execution_freeze_manifest: ExecutionFreezeManifest


@dataclass(frozen=True, slots=True)
class _SyntheticPredictionArchiveFixture:
    """Unbacked structural fixture; it is never capacity or effect evidence."""

    archive: bytes
    decoded: prediction_archive.DecodedRecognizerPredictionArchiveV1


@dataclass(frozen=True, slots=True)
class _PublicPositiveMechanicsFixture:
    """In-memory positive mapping fixture that is known not to fit the wire."""

    theory: object
    source_authority: object
    public_authority: object
    public_registry: recognizer_input.PublicRecognizerRegistryV1
    bridge_run: derived_bridge.ExactDerivedBridgeRun
    source_uuids: frozenset[str]


def _key_sources() -> trusted_wire.TrustedWireKeySourcesV1:
    return trusted_wire.TrustedWireKeySourcesV1(
        b"S" * 32,
        b"I" * 32,
        b"P" * 32,
    )


def _test_namespace(filename: str) -> dict[str, object]:
    return runpy.run_path(str(Path(__file__).with_name(filename)))


def _hex_id(prefix: str, index: int) -> str:
    return prefix + f"{index:064x}"


def _uuid4(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012x}"


def _copy_with_pollution(value: object, **changes: object) -> object:
    polluted = object.__new__(type(value))
    for field in fields(value):
        object.__setattr__(
            polluted,
            field.name,
            changes.get(field.name, getattr(value, field.name)),
        )
    return polluted


def _copy_without_field(value: object, missing_field: str) -> object:
    polluted = object.__new__(type(value))
    for field in fields(value):
        if field.name != missing_field:
            object.__setattr__(polluted, field.name, getattr(value, field.name))
    return polluted


def _shallow_unique_960_input_archive(
    value: recognizer_input.DecodedRecognizerInputArchiveV1,
) -> recognizer_input.DecodedRecognizerInputArchiveV1:
    source = value.rows[0]
    row_ids = tuple(
        _hex_id("phase2b_recognizer_input_row_", index + 10_000)
        for index in range(runner.TOTAL_RECOGNIZER_CASE_COUNT)
    )
    rows = tuple(
        _copy_with_pollution(source, row_id=row_id)
        for row_id in row_ids
    )
    return _copy_with_pollution(
        value,
        rows=rows,
        row_ids=row_ids,
        envelope_ids=tuple(item.envelope_id for item in rows),
        public_registry_ids=tuple(item.public_registry_id for item in rows),
        authority_content_ids=tuple(item.authority_content_id for item in rows),
        transform_result_ids=tuple(item.transform_result_id for item in rows),
    )


def _split_prediction_archive_frames(archive: bytes) -> list[bytes]:
    offset = prediction_archive.PREDICTION_ARCHIVE_HEADER_BYTES
    frames: list[bytes] = []
    while offset < len(archive):
        assert offset + 4 <= len(archive)
        length = int.from_bytes(archive[offset : offset + 4], "big")
        offset += 4
        assert offset + length <= len(archive)
        frames.append(archive[offset : offset + length])
        offset += length
    assert offset == len(archive)
    return frames


def _reframe_prediction_archive(
    frames: list[bytes],
    *,
    record_count: int = runner.TOTAL_RECOGNIZER_CASE_COUNT,
) -> bytes:
    body = b"".join(len(frame).to_bytes(4, "big") + frame for frame in frames)
    header = prediction_archive._ARCHIVE_HEADER.pack(
        prediction_archive.PREDICTION_ARCHIVE_MAGIC,
        prediction_archive.PREDICTION_ARCHIVE_WIRE_VERSION,
        0,
        record_count,
        hashlib.sha256(body).digest(),
    )
    return header + body


def _tamper_prediction_frame_mapping(
    archive: bytes,
    frame_index: int,
    mutate: Callable[[dict[str, object]], None],
) -> bytes:
    frames = _split_prediction_archive_frames(archive)
    mapping = trusted_wire_codec.decode_phase2b_jcs_profile_v1(
        frames[frame_index]
    )
    assert type(mapping) is dict
    mutate(mapping)
    frames[frame_index] = trusted_wire_codec.encode_phase2b_jcs_profile_v1(
        mapping
    )
    return _reframe_prediction_archive(frames)


def _semantic_texts(value: object) -> tuple[str, ...]:
    """Return only decoded semantic field names and text values.

    Fixed headers, body digests, padding, and other opaque binary framing are
    deliberately excluded: an arbitrary digest may contain a forbidden ASCII
    substring without exposing that concept in the public schema.
    """

    texts: list[str] = []
    stack = [value]
    while stack:
        current = stack.pop()
        if type(current) is dict:
            for key, item in current.items():
                assert type(key) is str
                texts.append(key)
                stack.append(item)
        elif type(current) in (list, tuple):
            stack.extend(current)
        elif type(current) is str:
            texts.append(current)
    return tuple(texts)


def _assert_semantic_tokens_absent(
    values: tuple[object, ...],
    forbidden_tokens: tuple[str, ...],
) -> None:
    semantic_token_rows = tuple(
        tuple(re.findall(r"[a-z0-9]+", text.casefold()))
        for value in values
        for text in _semantic_texts(value)
    )
    for forbidden in forbidden_tokens:
        forbidden_parts = tuple(re.findall(r"[a-z0-9]+", forbidden.casefold()))
        assert forbidden_parts
        for semantic_tokens in semantic_token_rows:
            width = len(forbidden_parts)
            assert all(
                semantic_tokens[index : index + width] != forbidden_parts
                for index in range(len(semantic_tokens) - width + 1)
            )


def _prediction_archive_semantic_mappings(archive: bytes) -> tuple[object, ...]:
    return tuple(
        trusted_wire_codec.decode_phase2b_jcs_profile_v1(frame)
        for frame in _split_prediction_archive_frames(archive)
    )


def _recognizer_input_semantic_mappings(
    archive: bytes,
) -> tuple[object, ...]:
    parsed = recognizer_input._parse_public_archive(archive)
    values: list[object] = [parsed.metadata]
    for row in parsed.rows:
        values.append(recognizer_input._registry_mapping(row.public_registry))
        structural = trusted_wire.decode_and_audit_trusted_envelope_v1(
            row.envelope
        )
        start = trusted_wire.ENVELOPE_HEADER_BYTES
        stop = start + structural.payload_bytes
        values.append(
            trusted_wire_codec.decode_phase2b_jcs_profile_v1(
                row.envelope[start:stop]
            )
        )
    return tuple(values)


def _coherent_forbidden_prediction_reason_archive(
    archive: bytes,
    reason: PredictionReason | str,
) -> bytes:
    """Recompute all public roots around one semantically forbidden reason."""

    frames = _split_prediction_archive_frames(archive)
    record_index = 3
    mapping = trusted_wire_codec.decode_phase2b_jcs_profile_v1(
        frames[record_index]
    )
    assert type(mapping) is dict
    prediction_mapping = mapping["prediction"]
    assert type(prediction_mapping) is dict
    prediction_mapping["reason"] = (
        reason.value if type(reason) is PredictionReason else reason
    )
    if type(reason) is PredictionReason:
        prediction_content_id = PredictionBundle.from_mapping(
            prediction_mapping
        ).content_id
    else:
        prediction_content_id = prediction_archive.stable_hash(
            prediction_mapping,
            prefix="phase2b_prediction_",
        )
    mapping["prediction_content_id"] = prediction_content_id
    record_body = dict(mapping)
    record_body.pop("record_id")
    record_payload = trusted_wire_codec.encode_phase2b_jcs_profile_v1(record_body)
    mapping["record_id"] = (
        "phase2b_recognizer_prediction_record_"
        + hashlib.sha256(
            prediction_archive._RECORD_DOMAIN + record_payload
        ).hexdigest()
    )
    frames[record_index] = trusted_wire_codec.encode_phase2b_jcs_profile_v1(
        mapping
    )

    record_ids = []
    for payload in frames[1:]:
        record_mapping = trusted_wire_codec.decode_phase2b_jcs_profile_v1(payload)
        assert type(record_mapping) is dict
        record_id = record_mapping["record_id"]
        assert type(record_id) is str
        record_ids.append(record_id)
    manifest = trusted_wire_codec.decode_phase2b_jcs_profile_v1(frames[0])
    assert type(manifest) is dict
    manifest["prediction_record_ids_root"] = (
        prediction_archive._prediction_record_ids_root(tuple(record_ids))
    )
    frames[0] = trusted_wire_codec.encode_phase2b_jcs_profile_v1(manifest)
    return _reframe_prediction_archive(frames)


def _coherent_duplicate_prediction_content_archive(archive: bytes) -> bytes:
    frames = _split_prediction_archive_frames(archive)
    source = trusted_wire_codec.decode_phase2b_jcs_profile_v1(frames[3])
    target = trusted_wire_codec.decode_phase2b_jcs_profile_v1(frames[4])
    assert type(source) is dict and type(target) is dict
    target["prediction"] = deepcopy(source["prediction"])
    target["prediction_content_id"] = source["prediction_content_id"]
    target["input_payload_sha256"] = source["input_payload_sha256"]
    record_body = dict(target)
    record_body.pop("record_id")
    payload = trusted_wire_codec.encode_phase2b_jcs_profile_v1(record_body)
    target["record_id"] = (
        "phase2b_recognizer_prediction_record_"
        + hashlib.sha256(prediction_archive._RECORD_DOMAIN + payload).hexdigest()
    )
    frames[4] = trusted_wire_codec.encode_phase2b_jcs_profile_v1(target)

    record_ids = []
    for payload in frames[1:]:
        mapping = trusted_wire_codec.decode_phase2b_jcs_profile_v1(payload)
        assert type(mapping) is dict and type(mapping["record_id"]) is str
        record_ids.append(mapping["record_id"])
    manifest = trusted_wire_codec.decode_phase2b_jcs_profile_v1(frames[0])
    assert type(manifest) is dict
    manifest["prediction_record_ids_root"] = (
        prediction_archive._prediction_record_ids_root(tuple(record_ids))
    )
    frames[0] = trusted_wire_codec.encode_phase2b_jcs_profile_v1(manifest)
    return _reframe_prediction_archive(frames)


def _minimum_positive_derived_authority() -> tuple[object, object, object]:
    """Construct the smallest current two-scale, six-family positive witness.

    Every support slice needs all 35 frozen observable channels because the
    candidate grid crosses every frozen law with every support slice.  The
    selector also requires two scales and a failed binding competitor.  The
    cheapest competitor changes only symmetry's ``source`` role, adding one
    scalar-width ``forward`` observation per scale.
    """

    namespace = _test_namespace(
        "test_phase2b_exact_derived_witness_bridge_v1.py"
    )
    original_observations = namespace["_full_observation_mappings"]
    original_base = namespace["_base_mapping"]
    uid = namespace["uid"]

    def minimum_observations(theory: object, ids: dict[str, object], **kwargs: object):
        base = original_observations(theory, ids, **kwargs)
        variant_kwargs = dict(kwargs)
        variant_kwargs["all_entity_variants"] = True
        variants = original_observations(theory, ids, **variant_kwargs)
        symmetry = next(
            law
            for law in theory.relation_laws
            if law.kind is LawKind.SYMMETRY
        )
        source_key = (symmetry.law_id, "source")
        alternate_entity_id = ids["entity_ids"][(source_key, 1)]
        forward_quantity_id = ids["quantity_ids"]["forward"]
        transformed_quantity_id = ids["quantity_ids"]["transformed"]
        for observation in base:
            if observation["quantity_id"] in (
                forward_quantity_id,
                transformed_quantity_id,
            ):
                observation["value"]["values"] = [1.0]
                observation["uncertainty"]["radius"] = [0.0]
        alternate = next(
            observation
            for observation in variants
            if observation["quantity_id"] == forward_quantity_id
            and observation["entity_ids"] == [alternate_entity_id]
        )
        alternate = dict(alternate)
        alternate["value"] = {"kind": "numeric", "values": [2.0]}
        alternate["uncertainty"] = {
            "model": "absolute_bound",
            "radius": [0.0],
        }
        alternate["observation_id"] = uid(99_000)
        alternate["provenance_sha256"] = "e" * 64
        return [*base, alternate]

    def minimum_base(
        theory: object,
        ids: dict[str, object],
        observations: list[dict[str, object]],
        **kwargs: object,
    ) -> dict[str, object]:
        result = original_base(theory, ids, observations, **kwargs)
        symmetry = next(
            law
            for law in theory.relation_laws
            if law.kind is LawKind.SYMMETRY
        )
        source_key = (symmetry.law_id, "source")
        alternate_entity_id = ids["entity_ids"][(source_key, 1)]
        result["entity_candidates"].append(
            {
                "entity_id": alternate_entity_id,
                "role_candidate_ids": [ids["role_ids"][source_key]],
            }
        )
        result["task_target"]["entity_ids"].append(alternate_entity_id)
        return result

    identity_authority = namespace["_identity_authority"]
    identity_authority.__globals__["_full_observation_mappings"] = (
        minimum_observations
    )
    identity_authority.__globals__["_base_mapping"] = minimum_base
    theory, registry, _, authority = identity_authority(
        pass_kinds_by_slice=(frozenset({LawKind.SYMMETRY}),),
    )
    return theory, registry, authority


def _direct_resource_preflight_authority(
    *,
    new_entity_count: int,
    include_complementarity_roles: bool,
) -> tuple[object, object, int]:
    namespace = _test_namespace(
        "test_phase2b_recognizer_input_archive_v1.py"
    )
    registry, authority = namespace["_six_family_source_fixture"]()
    profile = deepcopy(
        typed_authority.encode_typed_transform_authority_profile_v1(authority)
    )
    conservation = next(
        item
        for item in registry.law_bindings
        if item.law_kind is LawKind.CONSERVATION
    )
    role_ids = [wire_id for _, wire_id in conservation.role_ids]
    if include_complementarity_roles:
        complementarity = next(
            item
            for item in registry.law_bindings
            if item.law_kind is LawKind.COMPLEMENTARITY
        )
        role_ids.extend(wire_id for _, wire_id in complementarity.role_ids)
    role_ids = sorted(role_ids)
    base = profile["base_bundle"]
    assert type(base) is dict
    entity_candidates = base["entity_candidates"]
    task_target = base["task_target"]
    assert type(entity_candidates) is list and type(task_target) is dict
    target_entity_ids = task_target["entity_ids"]
    assert type(target_entity_ids) is list
    for index in range(new_entity_count):
        entity_id = f"20000000-0000-4000-8000-{index + 1:012x}"
        entity_candidates.append(
            {
                "entity_id": entity_id,
                "role_candidate_ids": role_ids,
            }
        )
        target_entity_ids.append(entity_id)
    entity_candidates.sort(key=lambda item: item["entity_id"])
    target_entity_ids.sort()
    payload_bytes = trusted_wire_codec._encode_profile_value(profile).encode(
        "ascii"
    )
    modified = typed_authority.decode_typed_transform_authority_profile_v1(
        profile
    )
    return registry, modified, len(payload_bytes)


@pytest.fixture(scope="module")
def one_row_prediction_fixture() -> _OneRowPredictionFixture:
    input_namespace = _test_namespace(
        "test_phase2b_recognizer_input_archive_v1.py"
    )
    registry, authority = input_namespace["_six_family_source_fixture"]()
    authorities = (authority,)
    keys = _key_sources()
    batch = trusted_wire.build_trusted_wire_batch_v1(
        authorities=authorities,
        run_id=RUN_ID,
        key_sources=keys,
    )
    assert type(batch) is trusted_wire.TrustedWireBatchV1
    replay = typed_replay.replay_typed_trusted_wire_batch_v1(
        batch=batch,
        run_id=RUN_ID,
        key_sources=keys,
        authorities=authorities,
    )
    assert type(replay) is typed_replay.TypedTrustedWireBatchReplayV1
    decoded_input = recognizer_input.issue_trusted_recognizer_input_archive_v1(
        typed_replay=replay,
        run_id=RUN_ID,
        key_sources=keys,
        source_cases=(
            recognizer_input.TrustedRecognizerSourceCaseV1(
                authority=authority,
                adapter_registry=registry,
            ),
        ),
    )
    assert type(decoded_input) is recognizer_input.DecodedRecognizerInputArchiveV1

    runner_namespace = _test_namespace("test_phase2b_runner.py")
    execution_freeze_manifest = replace(
        runner_namespace["_freeze_manifest"](),
        theory_version_id=initial_theory().version_id,
    )
    assert type(execution_freeze_manifest) is ExecutionFreezeManifest
    return _OneRowPredictionFixture(decoded_input, execution_freeze_manifest)


@pytest.fixture(scope="module")
def synthetic_prediction_archive_fixture(
    one_row_prediction_fixture: _OneRowPredictionFixture,
) -> _SyntheticPredictionArchiveFixture:
    """Issue 960 private structural records, without inventing trusted inputs.

    This deliberately bypasses the public archive builder because no real 960
    case public input archive exists in the repository.  The public decoder's
    false trust/execution/effect claims are the contract under test.
    """

    count = runner.TOTAL_RECOGNIZER_CASE_COUNT
    row_ids = tuple(
        _hex_id("phase2b_recognizer_input_row_", index + 1)
        for index in range(count)
    )
    manifest = one_row_prediction_fixture.execution_freeze_manifest
    context = prediction_archive.PublicRunContextV1._issue(
        prediction_archive._CONTEXT_ISSUE_TOKEN,
        input_archive_id=_hex_id(
            "phase2b_recognizer_input_archive_",
            1,
        ),
        input_archive_sha256=f"{2:064x}",
        input_row_ids_root=prediction_archive._input_row_ids_root(row_ids),
        protocol_id=prediction_archive._FROZEN_PROTOCOL.protocol_id,
        execution_freeze_manifest_id=manifest.manifest_id,
    )
    records: list[prediction_archive.PublicRecognizerPredictionRecordV1] = []
    for index, row_id in enumerate(row_ids, start=1):
        payload_sha256 = hashlib.sha256(
            f"synthetic-unbacked-payload-{index}".encode("ascii")
        ).hexdigest()
        is_positive = index in (1, 2)
        scales = (
            (_uuid4(10_000 + index),)
            if index == 1
            else (
                (_uuid4(10_000 + index), _uuid4(20_000 + index))
                if index == 2
                else ()
            )
        )
        law_kind = LawKind.SYMMETRY
        decision = (
            prediction_archive.PredictionDecisionV1.ANSWER
            if index == 1
            else (
                prediction_archive.PredictionDecisionV1.ANSWER_SET
                if index == 2
                else prediction_archive.PredictionDecisionV1.ABSTAIN
            )
        )
        prediction = PredictionBundle(
            schema_version=PREDICTION_SCHEMA_VERSION,
            bundle_id=_uuid4(index),
            input_root_sha256=payload_sha256,
            protocol_sha256=context.protocol_id.rsplit("_", 1)[1],
            freeze_manifest_sha256=(
                context.execution_freeze_manifest_id.rsplit("_", 1)[1]
            ),
            disposition=(
                PredictionDisposition.UNIQUE_MATCH
                if is_positive
                else PredictionDisposition.ABSTAIN
            ),
            reason=(
                PredictionReason.UNIQUE_STRUCTURAL_MATCH
                if is_positive
                else PredictionReason.INSUFFICIENT_EVIDENCE
            ),
            family_id=(
                prediction_archive._BRIDGE_FAMILY_BY_KIND[law_kind]
                if is_positive
                else None
            ),
            binding=(
                (
                    RoleBinding(
                        role_id=_uuid4(30_000 + index),
                        entity_id=_uuid4(40_000 + index),
                    ),
                )
                if is_positive
                else ()
            ),
            admissible_scale_ids=scales,
        )
        outcome = prediction_archive.PublicRecognizerPredictionOutcomeV1._issue(
            prediction_archive._RECORD_ISSUE_TOKEN,
            input_row_id=row_id,
            input_payload_sha256=payload_sha256,
            decision=decision,
            canonical_family_id=(
                prediction_archive._CANONICAL_FAMILY_BY_KIND[law_kind]
                if is_positive
                else None
            ),
            prediction=prediction,
            bridge_outcome_id=_hex_id(
                (
                    "phase2b_exact_derived_run_"
                    if is_positive
                    else "phase2b_exact_derived_preflight_"
                ),
                index,
            ),
            bridge_compilation_id=(
                _hex_id("phase2b_exact_derived_bridge_result_", index)
                if is_positive
                else None
            ),
            bridge_decision_id=(
                _hex_id("phase2b_exact_derived_decision_", index)
                if is_positive
                else None
            ),
        )
        records.append(
            prediction_archive.PublicRecognizerPredictionRecordV1._issue(
                prediction_archive._RECORD_ISSUE_TOKEN,
                context=context,
                input_row_id=row_id,
                input_payload_sha256=payload_sha256,
                input_authority_content_id=_hex_id(
                    "phase2b_public_transform_evidence_",
                    index,
                ),
                input_transform_result_id=_hex_id(
                    "phase2b_exact_transform_result_",
                    index,
                ),
                public_registry_id=_hex_id(
                    "phase2b_public_recognizer_registry_",
                    index,
                ),
                outcome=outcome,
            )
        )
    archive = prediction_archive._encode_prediction_archive(
        context=context,
        records=tuple(records),
    )
    decoded = prediction_archive.decode_public_recognizer_prediction_archive_v1(
        archive
    )
    return _SyntheticPredictionArchiveFixture(archive=archive, decoded=decoded)


@pytest.fixture(scope="module")
def public_positive_mechanics_fixture() -> _PublicPositiveMechanicsFixture:
    """Rename the oversized positive witness in memory for mapping tests only."""

    theory, source_registry, source_authority = (
        _minimum_positive_derived_authority()
    )
    profile = typed_authority.encode_typed_transform_authority_profile_v1(
        source_authority
    )
    audit = trusted_wire_codec.audit_namespace_paths_v1(profile)
    _, id_key, _ = trusted_wire._derive_keys(RUN_ID, _key_sources())
    counters = {namespace: 0 for namespace in trusted_wire._NAMESPACES}
    renamings: dict[tuple[str, str], str] = {}
    renamed = trusted_wire._rename_authority_ids(
        profile,
        audit,
        id_key,
        RUN_ID,
        counters,
        set(),
        renamings,
        [0],
    )
    trusted_wire._canonicalize_public_authority(renamed)
    public_authority = trusted_wire._compile_native_public_provenance(renamed)
    public_registry = (
        recognizer_input._compile_public_registry_from_live_renamings(
            source_registry=source_registry,
            renamings=renamings,
            typed_authority=public_authority,
        )
    )
    bridge_run = derived_bridge.run_exact_derived_witness_bridge(
        authority=public_authority,
        theory=theory,
        registry=public_registry.to_adapter_registry(),
    )
    assert type(bridge_run) is derived_bridge.ExactDerivedBridgeRun
    return _PublicPositiveMechanicsFixture(
        theory=theory,
        source_authority=source_authority,
        public_authority=public_authority,
        public_registry=public_registry,
        bridge_run=bridge_run,
        source_uuids=frozenset(
            {
                item.public_uuid for item in audit.occurrences
            }
            | {
                item.family_id for item in source_registry.law_bindings
            }
            | {
                wire_id
                for item in source_registry.law_bindings
                for _, wire_id in item.role_ids
            }
            | {
                item.quantity_id
                for item in source_registry.observable_channels
            }
        ),
    )


def test_prediction_builder_signature_has_no_caller_semantic_or_receipt_inputs() -> None:
    forbidden = {
        "theory",
        "registry",
        "policy",
        "receipt",
        "bridge_run",
        "prediction",
        "attestation",
        "protocol_sha256",
        "freeze_manifest_sha256",
        "input_root_sha256",
    }
    for callable_, expected_parameters in (
        (
            prediction_archive.build_recognizer_prediction_archive_v1,
            ("input_archive", "execution_freeze_manifest"),
        ),
        (
            prediction_archive.recognize_public_input_row_v1,
            ("input_row", "execution_freeze_manifest"),
        ),
    ):
        signature = inspect.signature(callable_)
        assert tuple(signature.parameters) == expected_parameters
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in signature.parameters.values()
        )
        assert forbidden.isdisjoint(signature.parameters)


def test_public_exports_field_manifests_and_policy_vectors_are_frozen() -> None:
    assert prediction_archive.__all__ == (
        "MAXIMUM_PREDICTION_ARCHIVE_BYTES",
        "MAXIMUM_PREDICTION_MANIFEST_BYTES",
        "MAXIMUM_PREDICTION_RECORD_BYTES",
        "PREDICTION_ARCHIVE_HEADER_BYTES",
        "PREDICTION_ARCHIVE_MAGIC",
        "PREDICTION_ARCHIVE_WIRE_VERSION",
        "PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_ID",
        "PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_VERSION",
        "PUBLIC_RUN_CONTEXT_SCHEMA_ID",
        "PUBLIC_RUN_CONTEXT_SCHEMA_VERSION",
        "RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID",
        "RECOGNIZER_PREDICTION_ARCHIVE_SCHEMA_VERSION",
        "DecodedRecognizerPredictionArchiveV1",
        "PredictionDecisionV1",
        "PublicRecognizerPredictionOutcomeV1",
        "PublicRecognizerPredictionRecordV1",
        "PublicRunContextV1",
        "RecognizerPredictionArchiveDisposition",
        "RecognizerPredictionArchiveRejectionV1",
        "build_public_run_context_v1",
        "build_recognizer_prediction_archive_v1",
        "decode_public_recognizer_prediction_archive_v1",
        "recognize_public_input_row_v1",
    )
    assert unsealed_evaluator.__all__ == (
        "UNSEALED_PREDICTION_EVALUATOR_POLICY_ID",
        "UNSEALED_PREDICTION_EVALUATOR_VERSION",
        "UnsealedPredictionEvaluationDisposition",
        "UnsealedPredictionEvaluationRejectionV1",
        "UnsealedPredictionPartitionManifestV1",
        "UnsealedPredictionStructuralEvaluationV1",
        "build_unsealed_prediction_partition_manifest_v1",
        "evaluate_unsealed_prediction_archive_structure_v1",
    )
    assert tuple(
        item.name for item in fields(prediction_archive.PublicRunContextV1)
    ) == (
        "schema_version",
        "input_archive_id",
        "input_archive_sha256",
        "input_row_ids_root",
        "protocol_id",
        "execution_freeze_manifest_id",
        "expected_prediction_count",
        "claim_level",
        "context_id",
    )
    assert tuple(
        item.name
        for item in fields(
            prediction_archive.DecodedRecognizerPredictionArchiveV1
        )
    ) == prediction_archive._DECODED_FIELD_MANIFEST
    assert tuple(
        item.name
        for item in fields(
            prediction_archive.RecognizerPredictionArchiveRejectionV1
        )
    ) == prediction_archive._REJECTION_FIELD_MANIFEST
    assert tuple(
        item.name
        for item in fields(
            unsealed_evaluator.UnsealedPredictionPartitionManifestV1
        )
    ) == unsealed_evaluator._MANIFEST_FIELDS
    assert tuple(
        item.name
        for item in fields(
            unsealed_evaluator.UnsealedPredictionStructuralEvaluationV1
        )
    ) == unsealed_evaluator._EVALUATION_FIELDS
    assert tuple(
        item.name
        for item in fields(
            unsealed_evaluator.UnsealedPredictionEvaluationRejectionV1
        )
    ) == unsealed_evaluator._REJECTION_FIELDS
    assert prediction_archive.PUBLIC_RUN_CONTEXT_SCHEMA_ID == (
        "phase2b_public_run_context_schema_"
        "905e7dbb074744120aa6236b6badd386c84c592b6ee4c30d547eeae835483311"
    )
    assert prediction_archive.PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_ID == (
        "phase2b_public_recognizer_prediction_record_schema_"
        "b3b216471a51d7c9ea90082b119405d81180bc0fa5efca8569483a8c8338955e"
    )
    assert prediction_archive.RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID == (
        "phase2b_recognizer_prediction_archive_policy_"
        "d90a920e516306934c9b12b03a4f317bbe5b8c2202358ed65b091945afdcbeb2"
    )
    assert unsealed_evaluator.UNSEALED_PREDICTION_EVALUATOR_POLICY_ID == (
        "phase2b_unsealed_prediction_evaluator_policy_"
        "194b0f338f17c482c3c9d5cbaf1c438508c906b05da9e2154b06f18e54236e65"
    )


def test_public_decision_enum_uses_split_blind_wire_values() -> None:
    assert tuple(
        (item.name, item.value) for item in prediction_archive.PredictionDecisionV1
    ) == (
        ("ANSWER", "unique_identification"),
        ("ANSWER_SET", "admissible_scale_set"),
        ("ABSTAIN", "abstain"),
    )
    assert all(
        b"answer" not in item.value.encode("ascii").lower()
        for item in prediction_archive.PredictionDecisionV1
    )


def test_abstention_reason_mapping_is_closed_and_unknown_values_reject() -> None:
    class StringSubclass(str):
        pass

    assert len(prediction_archive._ABSTAIN_REASON_PAIRS) == len(
        prediction_archive._ABSTAIN_REASON_MAP
    )
    assert tuple(
        sorted(
            source
            for source, target in prediction_archive._ABSTAIN_REASON_PAIRS
            if target is PredictionReason.RESOURCE_LIMIT
        )
    ) == prediction_archive._EXPECTED_RESOURCE_REASON_SOURCES
    assert tuple(
        sorted(
            source
            for source, target in prediction_archive._ABSTAIN_REASON_PAIRS
            if target is not PredictionReason.RESOURCE_LIMIT
        )
    ) == prediction_archive._ROW_SEMANTIC_REASON_SOURCES
    for source, target in prediction_archive._ABSTAIN_REASON_PAIRS:
        assert prediction_archive._mapped_abstention_reason(source) is target
    for source in prediction_archive._INTEGRITY_OR_DEAD_REASON_SOURCES_REJECTED:
        with pytest.raises(
            prediction_archive._PredictionGateRejected,
            match="unknown_bridge_abstention_reason",
        ):
            prediction_archive._mapped_abstention_reason(source)
    with pytest.raises(
        prediction_archive._PredictionGateRejected,
        match="unknown_bridge_abstention_reason",
    ):
        prediction_archive._mapped_abstention_reason("future_reason")
    with pytest.raises(
        prediction_archive._PredictionGateRejected,
        match="bridge_reason_type_invalid",
    ):
        prediction_archive._mapped_abstention_reason(StringSubclass("bad"))


def test_every_allowed_source_abstention_reason_builds_an_exact_valid_row(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
) -> None:
    decoded = synthetic_prediction_archive_fixture.decoded
    context = decoded.context
    base = decoded.records[2]
    assert base.decision is prediction_archive.PredictionDecisionV1.ABSTAIN
    for index, (source, expected_reason) in enumerate(
        prediction_archive._ABSTAIN_REASON_PAIRS,
        start=1,
    ):
        mapped_reason = prediction_archive._mapped_abstention_reason(source)
        assert mapped_reason is expected_reason
        prediction = replace(base.prediction, reason=mapped_reason)
        outcome = prediction_archive.PublicRecognizerPredictionOutcomeV1._issue(
            prediction_archive._RECORD_ISSUE_TOKEN,
            input_row_id=base.input_row_id,
            input_payload_sha256=base.input_payload_sha256,
            decision=prediction_archive.PredictionDecisionV1.ABSTAIN,
            canonical_family_id=None,
            prediction=prediction,
            bridge_outcome_id=_hex_id(
                "phase2b_exact_derived_preflight_",
                500_000 + index,
            ),
            bridge_compilation_id=None,
            bridge_decision_id=None,
        )
        record = prediction_archive.PublicRecognizerPredictionRecordV1._issue(
            prediction_archive._RECORD_ISSUE_TOKEN,
            context=context,
            input_row_id=base.input_row_id,
            input_payload_sha256=base.input_payload_sha256,
            input_authority_content_id=base.input_authority_content_id,
            input_transform_result_id=base.input_transform_result_id,
            public_registry_id=base.public_registry_id,
            outcome=outcome,
        )
        mapping = prediction_archive._record_mapping(record)
        assert prediction_archive._decode_record(
            mapping,
            context=context,
        ) == record


def test_coherent_invalid_input_reason_archive_is_rejected_despite_recomputed_roots(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
) -> None:
    malformed = _coherent_forbidden_prediction_reason_archive(
        synthetic_prediction_archive_fixture.archive,
        PredictionReason.INVALID_INPUT,
    )
    with pytest.raises(ValueError, match="unissued reason"):
        prediction_archive.decode_public_recognizer_prediction_archive_v1(
            malformed
        )


@pytest.mark.parametrize(
    "reason",
    (
        *prediction_archive._INTEGRITY_OR_DEAD_REASON_SOURCES_REJECTED,
        "future_unknown_reason",
    ),
)
def test_coherent_integrity_or_unknown_wire_reason_is_atomically_rejected(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
    reason: str,
) -> None:
    malformed = _coherent_forbidden_prediction_reason_archive(
        synthetic_prediction_archive_fixture.archive,
        reason,
    )
    with pytest.raises(ValueError, match="prediction reason is unknown"):
        prediction_archive.decode_public_recognizer_prediction_archive_v1(
            malformed
        )


def test_runner_freezes_exactly_960_unlabeled_recognizer_cases() -> None:
    class IntSubclass(int):
        pass

    assert runner.TOTAL_RECOGNIZER_CASE_COUNT == 960
    runner_namespace = _test_namespace("test_phase2b_runner.py")
    spec = runner_namespace["_run_spec"]()
    assert spec.expected_case_count == 960
    with pytest.raises(ValueError, match="exactly 960"):
        replace(spec, expected_case_count=720)
    for invalid in (960.0, IntSubclass(960)):
        with pytest.raises((TypeError, ValueError), match="exactly 960"):
            replace(spec, expected_case_count=invalid)


def test_minimum_current_positive_authority_cannot_cross_one_fixed_envelope() -> None:
    theory, registry, authority = _minimum_positive_derived_authority()
    run = derived_bridge.run_exact_derived_witness_bridge(
        authority=authority,
        theory=theory,
        registry=registry,
    )
    assert run.decision.disposition is (
        derived_bridge.ExactSelectionDisposition.ADMISSIBLE_SCALE_SET
    )
    assert len(theory.relation_laws) == 6
    assert sum(len(law.roles) for law in theory.relation_laws) == 15
    assert len(
        {
            observable
            for law in theory.relation_laws
            for observable in law.required_observables
        }
    ) == 35
    assert len(authority.base_bundle.aggregation_graph.scale_ids) == 2
    assert len(authority.base_bundle.entity_candidates) == 16
    assert len(authority.base_bundle.observations) == 36
    assert len(authority.transform_contracts) == 1
    assert len(authority.transform_contracts[0].output_observations) == 36
    assert run.inventory is not None
    assert len(run.inventory.observations) == 72
    assert sum(
        len(item.component_ids) for item in authority.observation_metadata
    ) == 38
    assert len(authority.transform_contracts[0].output_components) == 38

    profile = typed_authority.encode_typed_transform_authority_profile_v1(
        authority
    )
    canonical_payload = trusted_wire_codec._encode_profile_value(profile).encode(
        "ascii"
    )
    assert len(canonical_payload) == 125_582
    assert (
        len(canonical_payload)
        > trusted_wire_codec.MAXIMUM_PAYLOAD_BYTES
        == 65_424
    )
    with pytest.raises(ValueError, match="fixed-envelope budget"):
        trusted_wire_codec.encode_phase2b_jcs_profile_v1(profile)

    rejected = trusted_wire.build_trusted_wire_batch_v1(
        authorities=(authority,),
        run_id=RUN_ID,
        key_sources=_key_sources(),
    )
    assert type(rejected) is trusted_wire.TrustedWireBatchPreflightV1
    assert rejected.reason == (
        "authority_profile_rejected:profile_compilation_failed:"
        "accepted-JCS payload exceeds the fixed-envelope budget"
    )


@pytest.mark.parametrize(
    (
        "new_entity_count",
        "include_complementarity_roles",
        "expected_bytes",
        "expected_reason",
    ),
    (
        (
            36,
            False,
            17_881,
            "RESOURCE_LIMIT:raw_role_binding_product",
        ),
        (
            29,
            False,
            16_257,
            "RESOURCE_LIMIT:raw_role_binding_scale_product",
        ),
        (
            28,
            True,
            18_209,
            "RESOURCE_LIMIT:projected_candidate_count",
        ),
        (
            21,
            False,
            14_401,
            "RESOURCE_LIMIT:adapter_scan_work",
        ),
    ),
)
def test_wire_fit_authorities_reach_four_direct_bounded_preflight_reasons(
    new_entity_count: int,
    include_complementarity_roles: bool,
    expected_bytes: int,
    expected_reason: str,
) -> None:
    registry, authority, payload_bytes = _direct_resource_preflight_authority(
        new_entity_count=new_entity_count,
        include_complementarity_roles=include_complementarity_roles,
    )
    assert payload_bytes == expected_bytes
    assert payload_bytes < trusted_wire_codec.MAXIMUM_PAYLOAD_BYTES
    transform = derived_bridge.run_exact_transform_semantics(authority)
    assert type(transform) is derived_bridge.ExactTransformCompilation
    assert transform.disposition is derived_bridge.TransformCompilationDisposition.COMPLETE
    result = derived_bridge.run_exact_derived_witness_bridge(
        authority=authority,
        theory=initial_theory(),
        registry=registry,
    )
    assert type(result) is derived_bridge.ExactDerivedBridgePreflightRejection
    assert result.reason == expected_reason
    assert prediction_archive._mapped_abstention_reason(
        result.reason
    ) is PredictionReason.RESOURCE_LIMIT


def test_in_memory_positive_mechanics_recovers_six_dual_family_bindings(
    public_positive_mechanics_fixture: _PublicPositiveMechanicsFixture,
) -> None:
    fixture = public_positive_mechanics_fixture
    registry = fixture.public_registry
    assert len(registry.law_bindings) == len(LawKind) == 6
    assert {item.law_kind for item in registry.law_bindings} == set(LawKind)
    for law in registry.law_bindings:
        assert law.bridge_family_id == (
            prediction_archive._BRIDGE_FAMILY_BY_KIND[law.law_kind]
        )
        assert law.canonical_family_id is (
            prediction_archive._CANONICAL_FAMILY_BY_KIND[law.law_kind]
        )
        assert law.bridge_family_id != law.canonical_family_id.value

    (
        law_kind,
        canonical_family_id,
        bridge_family_id,
        public_binding,
        scales,
    ) = prediction_archive._selected_public_binding(
        bridge_run=fixture.bridge_run,
        public_registry=registry,
    )
    assert law_kind is LawKind.SYMMETRY
    assert canonical_family_id is (
        prediction_archive._CANONICAL_FAMILY_BY_KIND[law_kind]
    )
    assert bridge_family_id == prediction_archive._BRIDGE_FAMILY_BY_KIND[law_kind]
    assert len(scales) == 2
    selected_law = next(
        item for item in registry.law_bindings if item.law_kind is law_kind
    )
    assert {item.role_id for item in public_binding} == {
        wire_id for _, wire_id in selected_law.role_ids
    }
    assert fixture.source_uuids.isdisjoint(
        {
            bridge_family_id,
            *scales,
            *(item.role_id for item in public_binding),
            *(item.entity_id for item in public_binding),
            *(
                wire_id
                for law in registry.law_bindings
                for _, wire_id in law.role_ids
            ),
            *(item.quantity_id for item in registry.observable_channels),
        }
    )
    registry_bytes = recognizer_input._encode_public_registry(registry)
    for source_uuid in fixture.source_uuids:
        assert source_uuid.encode("ascii") not in registry_bytes


def test_same_structure_across_scales_requires_one_distinct_public_binding(
    public_positive_mechanics_fixture: _PublicPositiveMechanicsFixture,
) -> None:
    fixture = public_positive_mechanics_fixture
    decision = fixture.bridge_run.decision
    grid = fixture.bridge_run.compilation.candidate_grid
    assert grid is not None
    matching = tuple(
        item
        for item in grid.candidates
        if item.law_kind is decision.selected_law_kind
        and item.role_binding == decision.selected_role_binding
    )
    assert len(matching) >= 2
    assert len({item.support_slice.scale_id for item in matching}) >= 2
    assert len({item.public_binding for item in matching}) == 1
    selected = prediction_archive._selected_public_binding(
        bridge_run=fixture.bridge_run,
        public_registry=fixture.public_registry,
    )
    assert selected[-1] == decision.admissible_scale_ids


def test_one_matching_candidate_public_binding_drift_rejects_the_mapping(
    public_positive_mechanics_fixture: _PublicPositiveMechanicsFixture,
) -> None:
    fixture = public_positive_mechanics_fixture
    run = fixture.bridge_run
    grid = run.compilation.candidate_grid
    assert grid is not None
    target_index = next(
        index
        for index, item in enumerate(grid.candidates)
        if item.law_kind is run.decision.selected_law_kind
        and item.role_binding == run.decision.selected_role_binding
    )
    target = grid.candidates[target_index]
    first_binding = target.public_binding[0]
    polluted_binding = (
        RoleBinding(first_binding.role_id, _uuid4(900_001)),
        *target.public_binding[1:],
    )
    polluted_candidate = _copy_with_pollution(
        target,
        public_binding=polluted_binding,
    )
    polluted_candidates = list(grid.candidates)
    polluted_candidates[target_index] = polluted_candidate
    polluted_grid = _copy_with_pollution(
        grid,
        candidates=tuple(polluted_candidates),
    )
    polluted_compilation = _copy_with_pollution(
        run.compilation,
        candidate_grid=polluted_grid,
    )
    polluted_run = _copy_with_pollution(
        run,
        compilation=polluted_compilation,
    )
    with pytest.raises(
        prediction_archive._PredictionGateRejected,
        match="selected_public_binding_not_unique",
    ):
        prediction_archive._selected_public_binding(
            bridge_run=polluted_run,
            public_registry=fixture.public_registry,
        )


def test_fault_injected_in_memory_mechanics_exercises_unique_scale_mapping(
    public_positive_mechanics_fixture: _PublicPositiveMechanicsFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise UNIQUE mapping only; this is not a wire-runnable result."""

    fixture = public_positive_mechanics_fixture
    original_compile = derived_bridge._compile_grid

    def injected_compile(
        grid: object,
        *,
        theory: object,
        inventory: object,
        transform_result: object,
    ) -> tuple[object, object, object]:
        evaluations, aggregates, error = original_compile(
            grid,
            theory=theory,
            inventory=inventory,
            transform_result=transform_result,
        )
        assert evaluations is not None and aggregates is not None and error is None
        passing = tuple(
            item
            for item in evaluations
            if item.status is derived_bridge.ExactCandidateStatus.PASS
        )
        assert len(passing) == 2
        target = passing[0]
        assert target.tolerance is not None and target.tolerance.is_point
        tolerance = target.tolerance.lower_fraction
        changed = tuple(
            replace(
                item,
                residual=derived_bridge.ExactInterval.from_fractions(
                    tolerance * 2,
                    tolerance * 2,
                ),
                normalized=derived_bridge.ExactInterval.from_fractions(
                    Fraction(2),
                    Fraction(2),
                ),
            )
            if item is target
            else item
            for item in evaluations
        )
        changed = tuple(sorted(changed, key=lambda item: item.candidate_id))
        return (
            changed,
            derived_bridge._scale_aggregates(
                changed,
                grid,
                grid.candidate_grid_commitment_id,
            ),
            None,
        )

    monkeypatch.setattr(derived_bridge, "_compile_grid", injected_compile)
    run = derived_bridge.run_exact_derived_witness_bridge(
        authority=fixture.public_authority,
        theory=fixture.theory,
        registry=fixture.public_registry.to_adapter_registry(),
    )
    assert type(run) is derived_bridge.ExactDerivedBridgeRun
    assert run.decision.disposition is (
        derived_bridge.ExactSelectionDisposition.UNIQUE_IDENTIFICATION
    )
    selected = prediction_archive._selected_public_binding(
        bridge_run=run,
        public_registry=fixture.public_registry,
    )
    assert len(selected[-1]) == 1


@pytest.mark.parametrize(
    ("source_reason", "stage"),
    (
        ("RESOURCE_LIMIT:candidate_count", "candidate_grid"),
        ("RESOURCE_LIMIT:match_scan_work", "candidate_grid"),
        ("RESOURCE_LIMIT:role_binding_slice_product", "candidate_grid"),
        ("RESOURCE_LIMIT:exact_operation_budget", "compile_grid"),
        ("RESOURCE_LIMIT:exact_fraction_bit_length", "compile_grid"),
    ),
)
def test_actual_committed_bridge_resource_paths_are_namespaced_for_prediction_rows(
    public_positive_mechanics_fixture: _PublicPositiveMechanicsFixture,
    monkeypatch: pytest.MonkeyPatch,
    source_reason: str,
    stage: str,
) -> None:
    fixture = public_positive_mechanics_fixture
    if stage == "candidate_grid":
        def injected_candidate_grid(
            *args: object,
            **kwargs: object,
        ) -> tuple[None, str]:
            return None, source_reason

        monkeypatch.setattr(
            derived_bridge,
            "_candidate_grid",
            injected_candidate_grid,
        )
    else:
        def injected_compile_grid(
            *args: object,
            **kwargs: object,
        ) -> tuple[None, None, str]:
            return None, None, source_reason

        monkeypatch.setattr(
            derived_bridge,
            "_compile_grid",
            injected_compile_grid,
        )
    run = derived_bridge.run_exact_derived_witness_bridge(
        authority=fixture.public_authority,
        theory=fixture.theory,
        registry=fixture.public_registry.to_adapter_registry(),
    )
    assert type(run) is derived_bridge.ExactDerivedBridgeRun
    assert run.compilation.reason == source_reason
    expected_public_source = "bridge_" + source_reason
    assert run.decision.reason == expected_public_source
    assert prediction_archive._mapped_abstention_reason(
        run.decision.reason
    ) is PredictionReason.RESOURCE_LIMIT


@pytest.mark.parametrize(
    ("policy_changes", "expected_reason"),
    (
        (
            {
                "maximum_total_operations": 0,
                "maximum_operations_per_candidate": 0,
            },
            "RESOURCE_LIMIT:exact_operation_budget",
        ),
        (
            {"maximum_fraction_bit_length": 0},
            "RESOURCE_LIMIT:exact_fraction_bit_length",
        ),
    ),
)
def test_actual_arithmetic_budget_emits_the_two_closed_resource_reasons(
    public_positive_mechanics_fixture: _PublicPositiveMechanicsFixture,
    monkeypatch: pytest.MonkeyPatch,
    policy_changes: dict[str, int],
    expected_reason: str,
) -> None:
    run = public_positive_mechanics_fixture.bridge_run
    assert run.inventory is not None
    grid = run.compilation.candidate_grid
    assert grid is not None
    policy = _copy_with_pollution(
        derived_bridge.DEFAULT_EXACT_BRIDGE_POLICY,
        **policy_changes,
    )
    monkeypatch.setattr(
        derived_bridge,
        "DEFAULT_EXACT_BRIDGE_POLICY",
        policy,
    )
    evaluations, aggregates, reason = derived_bridge._compile_grid(
        grid,
        theory=public_positive_mechanics_fixture.theory,
        inventory=run.inventory,
        transform_result=run.transform_result,
    )
    assert evaluations is None
    assert aggregates is None
    assert reason == expected_reason
    assert prediction_archive._mapped_abstention_reason(
        "bridge_" + reason
    ) is PredictionReason.RESOURCE_LIMIT


def test_actual_selection_margin_bit_limit_reason_remains_direct(
    public_positive_mechanics_fixture: _PublicPositiveMechanicsFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compilation = public_positive_mechanics_fixture.bridge_run.compilation
    policy = _copy_with_pollution(
        derived_bridge.DEFAULT_EXACT_BRIDGE_POLICY,
        maximum_fraction_bit_length=0,
    )
    monkeypatch.setattr(
        derived_bridge,
        "DEFAULT_EXACT_BRIDGE_POLICY",
        policy,
    )
    decision = derived_bridge._select_scale_aggregates(
        compilation,
        compilation_id=compilation.result_id,
    )
    assert decision.disposition is derived_bridge.ExactSelectionDisposition.ABSTAIN
    assert decision.reason == "RESOURCE_LIMIT:selection_margin_bit_length"
    assert prediction_archive._mapped_abstention_reason(
        decision.reason
    ) is PredictionReason.RESOURCE_LIMIT


def test_real_one_row_input_closes_as_a_non_authoritative_abstention(
    one_row_prediction_fixture: _OneRowPredictionFixture,
) -> None:
    input_row = one_row_prediction_fixture.input_archive.rows[0]
    result = prediction_archive.recognize_public_input_row_v1(
        input_row=input_row,
        execution_freeze_manifest=(
            one_row_prediction_fixture.execution_freeze_manifest
        ),
    )
    assert type(result) is prediction_archive.PublicRecognizerPredictionOutcomeV1
    assert result.input_row_id == input_row.row_id
    assert result.input_payload_sha256 == input_row.payload_sha256
    assert result.decision is prediction_archive.PredictionDecisionV1.ABSTAIN
    assert result.canonical_family_id is None
    assert result.prediction.disposition.value == "abstain"
    assert result.prediction.family_id is None
    assert result.prediction.binding == ()
    assert result.prediction.admissible_scale_ids == ()
    typed = typed_replay.decode_and_replay_typed_trusted_envelope_v1(
        input_row.envelope
    )
    assert result.prediction.bundle_id == typed.authority.base_bundle.bundle_id
    assert result.claim_level == "NON_AUTHORITATIVE_MECHANICS_ONLY"


def test_public_run_context_root_has_the_exact_eight_field_preimage(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
) -> None:
    context = synthetic_prediction_archive_fixture.decoded.context
    mapping = prediction_archive._context_mapping(context)
    body = prediction_archive._context_mapping_without_id(context)
    assert tuple(sorted(mapping)) == prediction_archive._CONTEXT_FIELDS
    assert tuple(sorted(body)) == (
        "claim_level",
        "execution_freeze_manifest_id",
        "expected_prediction_count",
        "input_archive_id",
        "input_archive_sha256",
        "input_row_ids_root",
        "protocol_id",
        "schema_version",
    )
    assert len(body) == 8
    assert "context_id" not in body
    assert context.context_id == prediction_archive._context_id(body)
    assert prediction_archive._decode_context(mapping) == context
    with pytest.raises(TypeError, match="privately issued"):
        prediction_archive.PublicRunContextV1()


def test_derived_preflight_root_binds_exact_eight_fields_and_domain(
) -> None:
    bridge_result = derived_bridge.ExactDerivedBridgePreflightRejection(
        disposition=derived_bridge.ExactBridgeDisposition.ABSTAIN,
        reason="explicit_support_required",
        bundle_id="00000000-0000-4000-8000-000000000777",
        wrapper_schema_version=(
            "hegel-machine-phase2b-public-transform-evidence/2"
        ),
        theory_schema_version="hegel-machine-theory/0.2",
        registry_theory_version_id=(
            "theory_5f927174709a055a540092049a0af855f"
            "a5f6614133987c7a4d10d7e61733438"
        ),
        bridge_policy_id=(
            "phase2b_exact_derived_bridge_policy_"
            "089f4aa63018ef97cc445154954cf013d9dabd2f3c94285c506a4fbca621b2c8"
        ),
        matcher_semantics_id=(
            "phase2b_exact_derived_matcher_"
            "9c63b7bce0c0010b969679c197676af8e41bfa5a56203bda500b2eedc3068873"
        ),
    )
    assert prediction_archive._PREFLIGHT_OUTCOME_FIELDS == (
        "disposition",
        "reason",
        "bundle_id",
        "wrapper_schema_version",
        "theory_schema_version",
        "registry_theory_version_id",
        "bridge_policy_id",
        "matcher_semantics_id",
    )
    assert prediction_archive._PREFLIGHT_DOMAIN == (
        b"HEGEL/PHASE2B/DERIVED_PREFLIGHT_OUTCOME/V1\x00"
    )
    raw_items = tuple(
        getattr(bridge_result, name)
        for name in prediction_archive._PREFLIGHT_OUTCOME_FIELDS
    )
    preimage = (
        bridge_result.disposition.value,
        *raw_items[1:],
    )
    assert len(preimage) == 8
    payload = trusted_wire_codec.encode_phase2b_jcs_profile_v1(preimage)
    expected = "phase2b_exact_derived_preflight_" + hashlib.sha256(
        prediction_archive._PREFLIGHT_DOMAIN + payload
    ).hexdigest()
    assert prediction_archive._preflight_outcome_id(bridge_result) == expected
    assert expected == (
        "phase2b_exact_derived_preflight_"
        "7527832875e1dbb44cf41b56358909e94861555dcefeb7284befa165d2a8775d"
    )


def test_synthetic_960_archive_is_only_canonical_public_mechanics(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
) -> None:
    archive = synthetic_prediction_archive_fixture.archive
    decoded = synthetic_prediction_archive_fixture.decoded
    replayed = prediction_archive.decode_public_recognizer_prediction_archive_v1(
        archive
    )
    assert replayed == decoded
    assert replayed.archive == archive
    assert replayed.disposition is (
        prediction_archive.RecognizerPredictionArchiveDisposition.COMPLETE
    )
    assert len(replayed.records) == runner.TOTAL_RECOGNIZER_CASE_COUNT == 960
    assert replayed.records == tuple(
        sorted(replayed.records, key=lambda item: item.input_row_id)
    )
    assert len(set(replayed.input_row_ids)) == 960
    assert len(set(replayed.prediction_record_ids)) == 960
    assert len(set(replayed.prediction_content_ids)) == 960
    by_decision = {
        item.decision: item for item in replayed.records[:2]
    }
    assert set(by_decision) == {
        prediction_archive.PredictionDecisionV1.ANSWER,
        prediction_archive.PredictionDecisionV1.ANSWER_SET,
    }
    assert len(
        by_decision[
            prediction_archive.PredictionDecisionV1.ANSWER
        ].prediction.admissible_scale_ids
    ) == 1
    assert len(
        by_decision[
            prediction_archive.PredictionDecisionV1.ANSWER_SET
        ].prediction.admissible_scale_ids
    ) == 2
    for positive in by_decision.values():
        assert positive.prediction.disposition is PredictionDisposition.UNIQUE_MATCH
        assert positive.prediction.reason is (
            PredictionReason.UNIQUE_STRUCTURAL_MATCH
        )
        assert positive.prediction.family_id == (
            prediction_archive._BRIDGE_FAMILY_BY_KIND[LawKind.SYMMETRY]
        )
        assert positive.canonical_family_id is (
            prediction_archive._CANONICAL_FAMILY_BY_KIND[LawKind.SYMMETRY]
        )
        assert positive.prediction.binding
    assert all(
        record.decision is prediction_archive.PredictionDecisionV1.ABSTAIN
        and record.prediction.disposition is PredictionDisposition.ABSTAIN
        for record in replayed.records[2:]
    )

    assert replayed.structural_archive_verified is True
    assert replayed.canonical_record_framing_verified is True
    assert replayed.record_schema_verified is True
    assert replayed.row_root_coverage_verified is True
    for name in (
        "input_archive_membership_verified",
        "execution_manifest_authority_verified",
        "derived_bridge_mapping_verified",
        "runtime_executed",
        "recognizer_capacity_evidence",
        "origin_authenticated",
        "sealed_holdout_eligible",
        "formal_covert_audit",
        "prediction_scored",
        "effect_evidence",
        "c1_exit_evidence",
    ):
        assert getattr(replayed, name) is False


def test_public_archive_wire_is_split_blind_and_contains_no_answer_token(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
) -> None:
    mappings = _prediction_archive_semantic_mappings(
        synthetic_prediction_archive_fixture.archive
    )
    _assert_semantic_tokens_absent(
        mappings,
        (
            "answer",
            "challenge",
            "gold",
            "main",
            "partition",
            "case_type",
            "case_index",
            "case_position",
            "index",
            "ordinal",
            "metric",
            "score",
        ),
    )
    semantic_text = _semantic_texts(mappings)
    assert "abstain" in semantic_text
    assert "unique_identification" in semantic_text
    assert "admissible_scale_set" in semantic_text


def test_recognizer_input_bytes_are_also_free_of_split_and_gold_labels(
    one_row_prediction_fixture: _OneRowPredictionFixture,
) -> None:
    _assert_semantic_tokens_absent(
        _recognizer_input_semantic_mappings(
            one_row_prediction_fixture.input_archive.archive
        ),
        (
            "answer",
            "challenge",
            "gold",
            "main",
            "partition",
            "case_type",
            "case_index",
        ),
    )


@pytest.mark.parametrize("count", (959, 961))
def test_public_builder_rejects_non_960_inputs_without_partial_output(
    one_row_prediction_fixture: _OneRowPredictionFixture,
    count: int,
) -> None:
    input_archive = one_row_prediction_fixture.input_archive
    polluted = _copy_with_pollution(
        input_archive,
        rows=(input_archive.rows[0],) * count,
    )
    result = prediction_archive.build_recognizer_prediction_archive_v1(
        input_archive=polluted,
        execution_freeze_manifest=(
            one_row_prediction_fixture.execution_freeze_manifest
        ),
    )
    assert type(result) is prediction_archive.RecognizerPredictionArchiveRejectionV1
    assert result.disposition is (
        prediction_archive.RecognizerPredictionArchiveDisposition.ABSTAIN
    )
    assert result.reason == "prediction_count_drift"
    assert result.input_count == count
    assert result.archive is None
    assert type(result.records) is tuple and result.records == ()
    assert (
        type(result.prediction_record_ids) is tuple
        and result.prediction_record_ids == ()
    )
    assert result.recognizer_capacity_evidence is False


def test_public_builder_rejects_duplicate_960_rows_before_any_output(
    one_row_prediction_fixture: _OneRowPredictionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_archive = one_row_prediction_fixture.input_archive
    row = input_archive.rows[0]
    repeated = (row,) * runner.TOTAL_RECOGNIZER_CASE_COUNT
    polluted = _copy_with_pollution(
        input_archive,
        rows=repeated,
        row_ids=(row.row_id,) * runner.TOTAL_RECOGNIZER_CASE_COUNT,
        envelope_ids=(row.envelope_id,) * runner.TOTAL_RECOGNIZER_CASE_COUNT,
        public_registry_ids=(
            (row.public_registry_id,) * runner.TOTAL_RECOGNIZER_CASE_COUNT
        ),
        authority_content_ids=(
            (row.authority_content_id,) * runner.TOTAL_RECOGNIZER_CASE_COUNT
        ),
        transform_result_ids=(
            (row.transform_result_id,) * runner.TOTAL_RECOGNIZER_CASE_COUNT
        ),
    )

    def forbidden_deep_replay(*args: object, **kwargs: object) -> None:
        raise AssertionError("duplicate roots must fail before deep archive replay")

    monkeypatch.setattr(
        recognizer_input.DecodedRecognizerInputArchiveV1,
        "_validate",
        forbidden_deep_replay,
    )
    with pytest.raises(ValueError, match="unique|duplicate|repeat"):
        prediction_archive.build_public_run_context_v1(
            input_archive=polluted,
            execution_freeze_manifest=(
                one_row_prediction_fixture.execution_freeze_manifest
            ),
        )
    result = prediction_archive.build_recognizer_prediction_archive_v1(
        input_archive=polluted,
        execution_freeze_manifest=(
            one_row_prediction_fixture.execution_freeze_manifest
        ),
    )
    assert type(result) is prediction_archive.RecognizerPredictionArchiveRejectionV1
    assert result.reason == "input_or_execution_context_invalid"
    assert result.input_count == 960
    assert result.archive is None
    assert result.records == ()
    assert result.prediction_record_ids == ()
    assert result.recognizer_capacity_evidence is False


@pytest.mark.parametrize(
    ("pollution", "expected_reason"),
    (
        ("actual_exact_str", "input_row_id_shallow_invalid"),
        ("stored_str_subclass", "stored_row_id_shallow_invalid"),
    ),
)
def test_shallow_row_id_pollution_rejects_before_deep_archive_replay(
    one_row_prediction_fixture: _OneRowPredictionFixture,
    monkeypatch: pytest.MonkeyPatch,
    pollution: str,
    expected_reason: str,
) -> None:
    class StringSubclass(str):
        pass

    input_archive = _shallow_unique_960_input_archive(
        one_row_prediction_fixture.input_archive
    )
    if pollution == "actual_exact_str":
        rows = list(input_archive.rows)
        rows[17] = _copy_with_pollution(rows[17], row_id="invalid")
        row_ids = list(input_archive.row_ids)
        row_ids[17] = "invalid"
        polluted = _copy_with_pollution(
            input_archive,
            rows=tuple(rows),
            row_ids=tuple(row_ids),
        )
    else:
        row_ids = list(input_archive.row_ids)
        row_ids[17] = StringSubclass(row_ids[17])
        polluted = _copy_with_pollution(
            input_archive,
            row_ids=tuple(row_ids),
        )

    def forbidden_deep_replay(*args: object, **kwargs: object) -> None:
        raise AssertionError("bad shallow row IDs must fail before deep replay")

    monkeypatch.setattr(
        recognizer_input.DecodedRecognizerInputArchiveV1,
        "_validate",
        forbidden_deep_replay,
    )
    with pytest.raises(ValueError, match=expected_reason):
        prediction_archive.build_public_run_context_v1(
            input_archive=polluted,
            execution_freeze_manifest=(
                one_row_prediction_fixture.execution_freeze_manifest
            ),
        )
    result = prediction_archive.build_recognizer_prediction_archive_v1(
        input_archive=polluted,
        execution_freeze_manifest=(
            one_row_prediction_fixture.execution_freeze_manifest
        ),
    )
    assert type(result) is prediction_archive.RecognizerPredictionArchiveRejectionV1
    assert result.reason == "input_or_execution_context_invalid"
    assert result.input_count == runner.TOTAL_RECOGNIZER_CASE_COUNT
    assert result.archive is None
    assert type(result.records) is tuple and result.records == ()
    assert type(result.prediction_record_ids) is tuple
    assert result.prediction_record_ids == ()


def test_bad_execution_manifest_rejects_before_deep_input_replay(
    one_row_prediction_fixture: _OneRowPredictionFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_archive = _shallow_unique_960_input_archive(
        one_row_prediction_fixture.input_archive
    )
    bad_manifest = _copy_with_pollution(
        one_row_prediction_fixture.execution_freeze_manifest,
        protocol_id="invalid",
    )

    def forbidden_deep_replay(*args: object, **kwargs: object) -> None:
        raise AssertionError("bad execution manifest must fail before deep replay")

    monkeypatch.setattr(
        recognizer_input.DecodedRecognizerInputArchiveV1,
        "_validate",
        forbidden_deep_replay,
    )
    with pytest.raises((TypeError, ValueError), match="protocol|manifest"):
        prediction_archive.build_public_run_context_v1(
            input_archive=input_archive,
            execution_freeze_manifest=bad_manifest,
        )


@pytest.mark.parametrize("bad_archive_id", ("bad", None, 0))
@pytest.mark.parametrize("count", (959, 960))
def test_public_builder_sanitizes_invalid_optional_archive_id_on_rejection(
    one_row_prediction_fixture: _OneRowPredictionFixture,
    bad_archive_id: object,
    count: int,
) -> None:
    input_archive = one_row_prediction_fixture.input_archive
    polluted = _copy_with_pollution(
        input_archive,
        archive_id=bad_archive_id,
        rows=(input_archive.rows[0],) * count,
    )
    result = prediction_archive.build_recognizer_prediction_archive_v1(
        input_archive=polluted,
        execution_freeze_manifest=(
            one_row_prediction_fixture.execution_freeze_manifest
        ),
    )
    assert type(result) is prediction_archive.RecognizerPredictionArchiveRejectionV1
    assert result.reason == (
        "prediction_count_drift"
        if count == 959
        else "input_or_execution_context_invalid"
    )
    assert result.input_archive_id is None
    assert result.archive is None
    assert result.records == ()
    assert result.prediction_record_ids == ()
    assert result.recognizer_capacity_evidence is False


def test_public_builder_sanitizes_archive_id_string_subclass_on_rejection(
    one_row_prediction_fixture: _OneRowPredictionFixture,
) -> None:
    class StringSubclass(str):
        pass

    input_archive = one_row_prediction_fixture.input_archive
    polluted = _copy_with_pollution(
        input_archive,
        archive_id=StringSubclass(input_archive.archive_id),
        rows=(input_archive.rows[0],) * 959,
    )
    result = prediction_archive.build_recognizer_prediction_archive_v1(
        input_archive=polluted,
        execution_freeze_manifest=(
            one_row_prediction_fixture.execution_freeze_manifest
        ),
    )
    assert type(result) is prediction_archive.RecognizerPredictionArchiveRejectionV1
    assert result.input_archive_id is None
    assert result.archive is None
    assert result.records == ()
    assert result.prediction_record_ids == ()
    assert result.recognizer_capacity_evidence is False


@pytest.mark.parametrize("missing_field", ("rows", "archive_id"))
def test_public_builder_rejects_exact_input_object_with_missing_slot_atomically(
    one_row_prediction_fixture: _OneRowPredictionFixture,
    missing_field: str,
) -> None:
    polluted = _copy_without_field(
        one_row_prediction_fixture.input_archive,
        missing_field,
    )
    result = prediction_archive.build_recognizer_prediction_archive_v1(
        input_archive=polluted,
        execution_freeze_manifest=(
            one_row_prediction_fixture.execution_freeze_manifest
        ),
    )
    assert type(result) is prediction_archive.RecognizerPredictionArchiveRejectionV1
    assert result.disposition is (
        prediction_archive.RecognizerPredictionArchiveDisposition.ABSTAIN
    )
    assert result.archive is None
    assert result.records == ()
    assert result.prediction_record_ids == ()
    assert result.recognizer_capacity_evidence is False


def test_public_apis_require_exact_top_level_types(
    one_row_prediction_fixture: _OneRowPredictionFixture,
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
) -> None:
    class BytesSubclass(bytes):
        pass

    class InputArchiveSubclass(recognizer_input.DecodedRecognizerInputArchiveV1):
        pass

    with pytest.raises(TypeError, match="exact bytes"):
        prediction_archive.decode_public_recognizer_prediction_archive_v1(
            BytesSubclass(synthetic_prediction_archive_fixture.archive)
        )
    with pytest.raises(TypeError, match="exact decoded input archive"):
        prediction_archive.build_recognizer_prediction_archive_v1(
            input_archive=object.__new__(InputArchiveSubclass),
            execution_freeze_manifest=(
                one_row_prediction_fixture.execution_freeze_manifest
            ),
        )


def test_archive_size_cap_is_checked_before_any_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_hash(*args: object, **kwargs: object) -> object:
        raise AssertionError("hash must not run before the archive byte cap")

    monkeypatch.setattr(prediction_archive.hashlib, "sha256", forbidden_hash)
    with pytest.raises(ValueError, match="byte budget"):
        prediction_archive.decode_public_recognizer_prediction_archive_v1(
            b"x" * (prediction_archive.MAXIMUM_PREDICTION_ARCHIVE_BYTES + 1)
        )


def test_bad_context_identity_is_rejected_before_stable_hash(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = prediction_archive._context_mapping_without_id(
        synthetic_prediction_archive_fixture.decoded.context
    )
    body["input_archive_id"] = "bad"

    def forbidden_hash(*args: object, **kwargs: object) -> object:
        raise AssertionError("context identity must validate before stable_hash")

    monkeypatch.setattr(prediction_archive, "stable_hash", forbidden_hash)
    with pytest.raises(ValueError, match="input archive ID"):
        prediction_archive._context_id(body)


@pytest.mark.parametrize(
    "field_name",
    (
        "input_authority_content_id",
        "input_transform_result_id",
        "public_registry_id",
    ),
)
def test_bad_record_identity_is_rejected_before_sha256(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
) -> None:
    mapping = prediction_archive._record_mapping_without_id(
        synthetic_prediction_archive_fixture.decoded.records[0]
    )
    mapping[field_name] = "bad"

    def forbidden_hash(*args: object, **kwargs: object) -> object:
        raise AssertionError("record identity must validate before sha256")

    monkeypatch.setattr(prediction_archive.hashlib, "sha256", forbidden_hash)
    with pytest.raises(ValueError):
        prediction_archive._record_id(mapping)


@pytest.mark.parametrize(
    ("root_function", "valid_prefix"),
    (
        (
            "_input_row_ids_root",
            "phase2b_recognizer_input_row_",
        ),
        (
            "_prediction_record_ids_root",
            "phase2b_recognizer_prediction_record_",
        ),
    ),
)
def test_sequence_root_validates_all_items_before_sha256(
    monkeypatch: pytest.MonkeyPatch,
    root_function: str,
    valid_prefix: str,
) -> None:
    values = tuple(
        _hex_id(valid_prefix, index + 1)
        for index in range(runner.TOTAL_RECOGNIZER_CASE_COUNT)
    )
    values = (*values[:479], "bad", *values[480:])

    def forbidden_hash(*args: object, **kwargs: object) -> object:
        raise AssertionError("sequence items must validate before sha256")

    monkeypatch.setattr(prediction_archive.hashlib, "sha256", forbidden_hash)
    with pytest.raises(ValueError):
        getattr(prediction_archive, root_function)(values)


def test_decoder_rejects_959_and_961_framed_archives_atomically(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
) -> None:
    frames = _split_prediction_archive_frames(
        synthetic_prediction_archive_fixture.archive
    )
    malformed = (
        _reframe_prediction_archive(frames[:-1], record_count=959),
        _reframe_prediction_archive([*frames, frames[-1]], record_count=961),
        _reframe_prediction_archive(frames[:-1]),
        _reframe_prediction_archive([*frames, frames[-1]]),
    )
    for archive in malformed:
        with pytest.raises(ValueError):
            prediction_archive.decode_public_recognizer_prediction_archive_v1(
                archive
            )


def test_decoder_rejects_duplicate_reordered_and_root_drift_atomically(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
) -> None:
    frames = _split_prediction_archive_frames(
        synthetic_prediction_archive_fixture.archive
    )
    duplicate = list(frames)
    duplicate[2] = duplicate[1]
    reordered = list(frames)
    reordered[1], reordered[2] = reordered[2], reordered[1]
    root_drift = _tamper_prediction_frame_mapping(
        synthetic_prediction_archive_fixture.archive,
        0,
        lambda mapping: mapping.__setitem__(
            "input_row_ids_root",
            _hex_id("phase2b_prediction_input_rows_", 999_999),
        ),
    )
    for archive in (
        _reframe_prediction_archive(duplicate),
        _reframe_prediction_archive(reordered),
        root_drift,
    ):
        with pytest.raises(ValueError):
            prediction_archive.decode_public_recognizer_prediction_archive_v1(
                archive
            )


def test_decoder_rejects_self_consistent_duplicate_prediction_content_id(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
) -> None:
    malformed = _coherent_duplicate_prediction_content_archive(
        synthetic_prediction_archive_fixture.archive
    )
    with pytest.raises(ValueError, match="repeats a committed root"):
        prediction_archive.decode_public_recognizer_prediction_archive_v1(
            malformed
        )


def test_encoder_rejects_valid_records_with_duplicate_prediction_content_id(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
) -> None:
    decoded = synthetic_prediction_archive_fixture.decoded
    source = decoded.records[2]
    target = decoded.records[3]
    outcome = prediction_archive.PublicRecognizerPredictionOutcomeV1._issue(
        prediction_archive._RECORD_ISSUE_TOKEN,
        input_row_id=target.input_row_id,
        input_payload_sha256=source.input_payload_sha256,
        decision=source.decision,
        canonical_family_id=source.canonical_family_id,
        prediction=source.prediction,
        bridge_outcome_id=target.bridge_outcome_id,
        bridge_compilation_id=target.bridge_compilation_id,
        bridge_decision_id=target.bridge_decision_id,
    )
    duplicate = prediction_archive.PublicRecognizerPredictionRecordV1._issue(
        prediction_archive._RECORD_ISSUE_TOKEN,
        context=decoded.context,
        input_row_id=target.input_row_id,
        input_payload_sha256=source.input_payload_sha256,
        input_authority_content_id=target.input_authority_content_id,
        input_transform_result_id=target.input_transform_result_id,
        public_registry_id=target.public_registry_id,
        outcome=outcome,
    )
    assert duplicate.record_id != source.record_id
    assert duplicate.prediction_content_id == source.prediction_content_id
    records = list(decoded.records)
    records[3] = duplicate
    with pytest.raises(ValueError, match="prediction"):
        prediction_archive._encode_prediction_archive(
            context=decoded.context,
            records=tuple(records),
        )


def test_decoder_rejects_unknown_reason_extra_field_and_bool_int_count(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
) -> None:
    def unknown_reason(mapping: dict[str, object]) -> None:
        prediction = mapping["prediction"]
        assert type(prediction) is dict
        prediction["reason"] = "future_unknown_reason"

    def family_drift(mapping: dict[str, object]) -> None:
        mapping["canonical_family_id"] = (
            prediction_archive._CANONICAL_FAMILY_BY_KIND[
                LawKind.MONOTONICITY
            ].value
        )

    malformed = (
        _tamper_prediction_frame_mapping(
            synthetic_prediction_archive_fixture.archive,
            1,
            unknown_reason,
        ),
        _tamper_prediction_frame_mapping(
            synthetic_prediction_archive_fixture.archive,
            1,
            family_drift,
        ),
        _tamper_prediction_frame_mapping(
            synthetic_prediction_archive_fixture.archive,
            1,
            lambda mapping: mapping.__setitem__("unexpected", "field"),
        ),
        _tamper_prediction_frame_mapping(
            synthetic_prediction_archive_fixture.archive,
            1,
            lambda mapping: mapping.pop("bridge_outcome_id"),
        ),
        _tamper_prediction_frame_mapping(
            synthetic_prediction_archive_fixture.archive,
            0,
            lambda mapping: mapping.__setitem__(
                "expected_prediction_count",
                True,
            ),
        ),
    )
    for archive in malformed:
        with pytest.raises((TypeError, ValueError)):
            prediction_archive.decode_public_recognizer_prediction_archive_v1(
                archive
            )


def test_decoder_rejects_noncanonical_record_and_body_digest_drift(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
) -> None:
    frames = _split_prediction_archive_frames(
        synthetic_prediction_archive_fixture.archive
    )
    noncanonical = list(frames)
    noncanonical[1] = b" " + noncanonical[1]
    bad_digest = bytearray(synthetic_prediction_archive_fixture.archive)
    bad_digest[-1] ^= 1
    for archive in (
        _reframe_prediction_archive(noncanonical),
        bytes(bad_digest),
    ):
        with pytest.raises(ValueError):
            prediction_archive.decode_public_recognizer_prediction_archive_v1(
                archive
            )


def test_prediction_bundle_and_nested_bindings_require_exact_builtin_types(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
) -> None:
    class StringSubclass(str):
        pass

    class TupleSubclass(tuple):
        pass

    class RoleBindingSubclass(RoleBinding):
        pass

    prediction = synthetic_prediction_archive_fixture.decoded.records[0].prediction
    nested_subclass = RoleBindingSubclass(
        prediction.binding[0].role_id,
        prediction.binding[0].entity_id,
    )
    polluted_values = (
        _copy_with_pollution(
            prediction,
            schema_version=StringSubclass(prediction.schema_version),
        ),
        _copy_with_pollution(
            prediction,
            binding=TupleSubclass(prediction.binding),
        ),
        _copy_with_pollution(
            prediction,
            binding=(nested_subclass,),
        ),
        _copy_with_pollution(
            prediction,
            admissible_scale_ids=TupleSubclass(
                prediction.admissible_scale_ids
            ),
        ),
        _copy_with_pollution(
            prediction,
            disposition=prediction.disposition.value,
        ),
    )
    for polluted in polluted_values:
        with pytest.raises(TypeError):
            prediction_archive._prediction_mapping(polluted)


def test_context_record_and_decoded_archive_reject_subclass_or_bool_pollution(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StringSubclass(str):
        pass

    class RecordSubclass(prediction_archive.PublicRecognizerPredictionRecordV1):
        pass

    decoded = synthetic_prediction_archive_fixture.decoded
    context = decoded.context
    for polluted_context in (
        _copy_with_pollution(
            context,
            schema_version=StringSubclass(context.schema_version),
        ),
        _copy_with_pollution(context, expected_prediction_count=True),
    ):
        with pytest.raises((TypeError, ValueError)):
            polluted_context._validate()

    record = decoded.records[0]
    subclass_record = object.__new__(RecordSubclass)
    for field in fields(record):
        object.__setattr__(subclass_record, field.name, getattr(record, field.name))
    with pytest.raises(TypeError, match="exact type"):
        subclass_record._validate()
    with pytest.raises(TypeError, match="exact enum"):
        _copy_with_pollution(
            record,
            decision=record.decision.value,
        )._validate()

    def forbidden_parse(*args: object, **kwargs: object) -> object:
        raise AssertionError("claim type gate must precede archive parsing")

    monkeypatch.setattr(
        prediction_archive,
        "_parse_prediction_archive",
        forbidden_parse,
    )
    for changes in (
        {"structural_archive_verified": 1},
        {"recognizer_capacity_evidence": 0},
    ):
        with pytest.raises(TypeError, match="exact bool"):
            _copy_with_pollution(decoded, **changes)._validate()


def test_prediction_public_objects_cannot_be_forged_with_public_constructors() -> None:
    for type_ in (
        prediction_archive.PublicRunContextV1,
        prediction_archive.PublicRecognizerPredictionOutcomeV1,
        prediction_archive.PublicRecognizerPredictionRecordV1,
        prediction_archive.DecodedRecognizerPredictionArchiveV1,
    ):
        with pytest.raises(TypeError, match="privately issued"):
            type_()
    with pytest.raises(TypeError, match="token mismatch"):
        prediction_archive.PublicRunContextV1._issue(
            object(),
            input_archive_id=_hex_id(
                "phase2b_recognizer_input_archive_",
                1,
            ),
            input_archive_sha256=f"{2:064x}",
            input_row_ids_root=_hex_id("phase2b_prediction_input_rows_", 3),
            protocol_id=prediction_archive._FROZEN_PROTOCOL.protocol_id,
            execution_freeze_manifest_id=_hex_id(
                "phase2b_execution_freeze_",
                4,
            ),
        )


def test_prediction_rejection_requires_exact_empty_outputs_and_scalars() -> None:
    class StringSubclass(str):
        pass

    class TupleSubclass(tuple):
        pass

    base = {
        "disposition": (
            prediction_archive.RecognizerPredictionArchiveDisposition.ABSTAIN
        ),
        "reason": "closed_rejection",
        "input_count": 960,
        "input_archive_id": _hex_id(
            "phase2b_recognizer_input_archive_",
            1,
        ),
    }
    for changes in (
        {"reason": StringSubclass("closed_rejection")},
        {"input_count": True},
        {
            "input_archive_id": StringSubclass(
                base["input_archive_id"]
            )
        },
        {"records": TupleSubclass()},
        {"prediction_record_ids": TupleSubclass()},
        {"records": []},
        {"prediction_record_ids": None},
        {"recognizer_capacity_evidence": 0},
        {"recognizer_capacity_evidence": True},
    ):
        with pytest.raises((TypeError, ValueError)):
            prediction_archive.RecognizerPredictionArchiveRejectionV1(
                **{**base, **changes},
            )


def test_unsealed_evaluator_only_confirms_720_240_structure_without_scoring(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
) -> None:
    decoded = synthetic_prediction_archive_fixture.decoded
    main_rows = tuple(sorted(decoded.input_row_ids[:720]))
    conflict_rows = tuple(sorted(decoded.input_row_ids[720:]))
    manifest = (
        unsealed_evaluator.build_unsealed_prediction_partition_manifest_v1(
            prediction_archive=decoded,
            main_row_ids=main_rows,
            semantic_conflict_row_ids=conflict_rows,
        )
    )
    result = (
        unsealed_evaluator.evaluate_unsealed_prediction_archive_structure_v1(
            prediction_archive=decoded,
            partition_manifest=manifest,
        )
    )
    assert type(result) is (
        unsealed_evaluator.UnsealedPredictionStructuralEvaluationV1
    )
    assert result.disposition is (
        unsealed_evaluator.UnsealedPredictionEvaluationDisposition
        .STRUCTURALLY_COMPLETE_NOT_SCORED
    )
    assert (result.main_count, result.semantic_conflict_count, result.total_count) == (
        720,
        240,
        960,
    )
    assert result.structural_completeness_verified is True
    assert result.exact_freeze_id == unsealed_evaluator._EXACT_FREEZE_ID
    assert result.evaluator_policy_id == (
        unsealed_evaluator.UNSEALED_PREDICTION_EVALUATOR_POLICY_ID
    )
    assert manifest.exact_freeze_id == result.exact_freeze_id
    assert manifest.evaluator_policy_id == result.evaluator_policy_id
    assert result.challenge_in_main_denominator is False
    assert result.scoring_performed is False
    assert result.runtime_executed is False
    assert result.recognizer_capacity_evidence is False
    assert type(result.metric_results) is tuple and result.metric_results == ()
    assert type(result.scored_rows) is tuple and result.scored_rows == ()
    for name in (
        "origin_authenticated",
        "sealed_holdout_eligible",
        "formal_covert_audit",
        "effect_evidence",
        "c1_exit_evidence",
    ):
        assert getattr(result, name) is False
    assert result.claim_level == "NON_AUTHORITATIVE_MECHANICS_ONLY"
    _assert_semantic_tokens_absent(
        _prediction_archive_semantic_mappings(
            synthetic_prediction_archive_fixture.archive
        ),
        ("partition",),
    )
    with pytest.raises(TypeError, match="privately issued"):
        unsealed_evaluator.UnsealedPredictionPartitionManifestV1()


def test_unsealed_evaluator_has_no_scorer_or_answer_inputs() -> None:
    signature = inspect.signature(
        unsealed_evaluator.evaluate_unsealed_prediction_archive_structure_v1
    )
    assert tuple(signature.parameters) == (
        "prediction_archive",
        "partition_manifest",
    )
    assert all(
        item.kind is inspect.Parameter.KEYWORD_ONLY
        for item in signature.parameters.values()
    )
    folded = " ".join(signature.parameters).casefold()
    for forbidden in ("score", "metric", "answer", "gold"):
        assert forbidden not in folded


@pytest.mark.parametrize(
    "pollution",
    ("count", "reorder", "overlap", "missing", "archive"),
)
def test_unsealed_evaluator_rejects_partition_drift_without_partial_metrics(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
    pollution: str,
) -> None:
    decoded = synthetic_prediction_archive_fixture.decoded
    main_rows = tuple(sorted(decoded.input_row_ids[:720]))
    conflict_rows = tuple(sorted(decoded.input_row_ids[720:]))
    manifest = (
        unsealed_evaluator.build_unsealed_prediction_partition_manifest_v1(
            prediction_archive=decoded,
            main_row_ids=main_rows,
            semantic_conflict_row_ids=conflict_rows,
        )
    )
    if pollution == "count":
        manifest = _copy_with_pollution(
            manifest,
            main_row_ids=manifest.main_row_ids[:-1],
        )
    elif pollution == "reorder":
        manifest = _copy_with_pollution(
            manifest,
            main_row_ids=(
                manifest.main_row_ids[1],
                manifest.main_row_ids[0],
                *manifest.main_row_ids[2:],
            ),
        )
    elif pollution == "overlap":
        manifest = _copy_with_pollution(
            manifest,
            semantic_conflict_row_ids=(
                manifest.main_row_ids[0],
                *manifest.semantic_conflict_row_ids[1:],
            ),
        )
    elif pollution == "missing":
        manifest = _copy_with_pollution(
            manifest,
            semantic_conflict_row_ids=(
                _hex_id("phase2b_recognizer_input_row_", 999_999),
                *manifest.semantic_conflict_row_ids[1:],
            ),
        )
    else:
        manifest = _copy_with_pollution(
            manifest,
            prediction_archive_id=_hex_id(
                "phase2b_recognizer_prediction_archive_",
                999_999,
            ),
        )
    result = (
        unsealed_evaluator.evaluate_unsealed_prediction_archive_structure_v1(
            prediction_archive=decoded,
            partition_manifest=manifest,
        )
    )
    assert type(result) is (
        unsealed_evaluator.UnsealedPredictionEvaluationRejectionV1
    )
    assert result.disposition is (
        unsealed_evaluator.UnsealedPredictionEvaluationDisposition.ABSTAIN
    )
    assert type(result.metric_results) is tuple and result.metric_results == ()
    assert type(result.scored_rows) is tuple and result.scored_rows == ()
    assert result.structural_completeness_verified is False
    assert result.scoring_performed is False
    assert result.runtime_executed is False
    assert result.recognizer_capacity_evidence is False
    assert result.effect_evidence is False


def test_unsealed_evaluator_rejects_tuple_and_item_subclasses_atomically(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
) -> None:
    class StringSubclass(str):
        pass

    class TupleSubclass(tuple):
        pass

    decoded = synthetic_prediction_archive_fixture.decoded
    manifest = (
        unsealed_evaluator.build_unsealed_prediction_partition_manifest_v1(
            prediction_archive=decoded,
            main_row_ids=tuple(sorted(decoded.input_row_ids[:720])),
            semantic_conflict_row_ids=tuple(sorted(decoded.input_row_ids[720:])),
        )
    )
    polluted_manifests = (
        _copy_with_pollution(
            manifest,
            main_row_ids=TupleSubclass(manifest.main_row_ids),
        ),
        _copy_with_pollution(
            manifest,
            main_row_ids=(
                StringSubclass(manifest.main_row_ids[0]),
                *manifest.main_row_ids[1:],
            ),
        ),
    )
    for polluted in polluted_manifests:
        result = (
            unsealed_evaluator.evaluate_unsealed_prediction_archive_structure_v1(
                prediction_archive=decoded,
                partition_manifest=polluted,
            )
        )
        assert type(result) is (
            unsealed_evaluator.UnsealedPredictionEvaluationRejectionV1
        )
        assert result.metric_results == ()
        assert result.scored_rows == ()
        assert result.scoring_performed is False
        assert result.runtime_executed is False
        assert result.recognizer_capacity_evidence is False
        assert result.effect_evidence is False


@pytest.mark.parametrize("target", ("archive", "manifest"))
def test_malformed_optional_rejection_ids_abstain_without_exception(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
    target: str,
) -> None:
    class StringSubclass(str):
        pass

    decoded = synthetic_prediction_archive_fixture.decoded
    manifest = (
        unsealed_evaluator.build_unsealed_prediction_partition_manifest_v1(
            prediction_archive=decoded,
            main_row_ids=tuple(sorted(decoded.input_row_ids[:720])),
            semantic_conflict_row_ids=tuple(sorted(decoded.input_row_ids[720:])),
        )
    )
    if target == "archive":
        decoded = _copy_with_pollution(
            decoded,
            archive_id=StringSubclass(decoded.archive_id),
        )
    else:
        manifest = _copy_with_pollution(
            manifest,
            manifest_id=StringSubclass(manifest.manifest_id),
        )
    result = unsealed_evaluator.evaluate_unsealed_prediction_archive_structure_v1(
        prediction_archive=decoded,
        partition_manifest=manifest,
    )
    assert type(result) is (
        unsealed_evaluator.UnsealedPredictionEvaluationRejectionV1
    )
    if target == "archive":
        assert result.prediction_archive_id is None
        assert result.partition_manifest_id == manifest.manifest_id
    else:
        assert result.prediction_archive_id == decoded.archive_id
        assert result.partition_manifest_id is None
    assert result.metric_results == ()
    assert result.scored_rows == ()
    assert result.structural_completeness_verified is False
    assert result.scoring_performed is False
    assert result.runtime_executed is False
    assert result.recognizer_capacity_evidence is False
    assert result.effect_evidence is False


def test_unsealed_success_cannot_be_reused_for_a_different_archive_root(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
) -> None:
    decoded = synthetic_prediction_archive_fixture.decoded
    manifest = (
        unsealed_evaluator.build_unsealed_prediction_partition_manifest_v1(
            prediction_archive=decoded,
            main_row_ids=tuple(sorted(decoded.input_row_ids[:720])),
            semantic_conflict_row_ids=tuple(sorted(decoded.input_row_ids[720:])),
        )
    )
    wrong_archive = _copy_with_pollution(
        decoded,
        archive_id=_hex_id(
            "phase2b_recognizer_prediction_archive_",
            987_654,
        ),
    )
    result = unsealed_evaluator.evaluate_unsealed_prediction_archive_structure_v1(
        prediction_archive=wrong_archive,
        partition_manifest=manifest,
    )
    assert type(result) is (
        unsealed_evaluator.UnsealedPredictionEvaluationRejectionV1
    )
    assert result.metric_results == ()
    assert result.scored_rows == ()
    assert result.structural_completeness_verified is False
    assert result.scoring_performed is False
    assert result.runtime_executed is False
    assert result.recognizer_capacity_evidence is False
    assert result.effect_evidence is False


@pytest.mark.parametrize(
    ("target", "missing_field"),
    (
        ("archive", "archive_id"),
        ("archive", "input_row_ids"),
        ("manifest", "manifest_id"),
        ("manifest", "main_row_ids"),
    ),
)
def test_unsealed_evaluator_rejects_exact_missing_slot_atomically(
    synthetic_prediction_archive_fixture: _SyntheticPredictionArchiveFixture,
    target: str,
    missing_field: str,
) -> None:
    decoded = synthetic_prediction_archive_fixture.decoded
    manifest = (
        unsealed_evaluator.build_unsealed_prediction_partition_manifest_v1(
            prediction_archive=decoded,
            main_row_ids=tuple(sorted(decoded.input_row_ids[:720])),
            semantic_conflict_row_ids=tuple(sorted(decoded.input_row_ids[720:])),
        )
    )
    if target == "archive":
        decoded = _copy_without_field(decoded, missing_field)
    else:
        manifest = _copy_without_field(manifest, missing_field)
    result = unsealed_evaluator.evaluate_unsealed_prediction_archive_structure_v1(
        prediction_archive=decoded,
        partition_manifest=manifest,
    )
    assert type(result) is (
        unsealed_evaluator.UnsealedPredictionEvaluationRejectionV1
    )
    assert result.disposition is (
        unsealed_evaluator.UnsealedPredictionEvaluationDisposition.ABSTAIN
    )
    assert result.metric_results == ()
    assert result.scored_rows == ()
    assert result.structural_completeness_verified is False
    assert result.scoring_performed is False
    assert result.runtime_executed is False
    assert result.recognizer_capacity_evidence is False
    assert result.effect_evidence is False
