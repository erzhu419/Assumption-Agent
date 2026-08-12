"""Adversarial contracts for the compact-V2 prediction archive.

Every 960-record fixture in this file is synthetic and unbacked.  It proves
only bounded canonical archive mechanics.  It is not evidence of a trusted
960-row input archive, an actual recognizer run, capacity, scoring, effect, or
C1 exit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import inspect
from pathlib import Path
import runpy
from typing import Callable, Iterator

import pytest

from hegel_machine.bootstrap import initial_theory
import hegel_machine.phase2b_recognizer_input_archive_v1 as input_v1
import hegel_machine.phase2b_recognizer_input_archive_v2 as input_v2
import hegel_machine.phase2b_recognizer_prediction_archive_v1 as archive_v1
import hegel_machine.phase2b_recognizer_prediction_archive_v2 as archive_v2
import hegel_machine.phase2b_recognizer_prediction_v2 as prediction_v2
import hegel_machine.phase2b_protocol as phase2b_protocol
import hegel_machine.phase2b_runner as runner
from hegel_machine.phase2b_protocol import ExecutionFreezeManifest
from hegel_machine.phase2b_trusted_wire_v1 import (
    NON_AUTHORITATIVE_CLAIM_LEVEL,
    decode_phase2b_jcs_profile_v1,
    encode_phase2b_jcs_profile_v1,
)
from hegel_machine.phase2b_wire import (
    PREDICTION_SCHEMA_VERSION,
    PredictionBundle,
    PredictionDisposition,
    PredictionReason,
    RoleBinding,
)


RUN_ID = b"R" * 32
COUNT = runner.TOTAL_RECOGNIZER_CASE_COUNT

CONTEXT_FIELDS = (
    "batch_id",
    "batch_policy_id",
    "claim_level",
    "context_id",
    "execution_freeze_manifest_id",
    "expected_prediction_count",
    "input_archive_id",
    "input_archive_policy_id",
    "input_archive_sha256",
    "input_archive_version",
    "input_row_ids_root",
    "protocol_id",
    "schema_version",
)
RECORD_FIELDS = (
    "bridge_compilation_id",
    "bridge_decision_id",
    "bridge_outcome_id",
    "canonical_family_id",
    "decision",
    "input_authority_content_id",
    "input_envelope_id",
    "input_namespace_audit_id",
    "input_padding_sha256",
    "input_payload_sha256",
    "input_public_registry_id",
    "input_row_id",
    "input_transform_result_id",
    "prediction",
    "prediction_content_id",
    "record_id",
    "run_context_id",
    "schema_version",
)
MANIFEST_FIELDS = (
    "archive_policy_id",
    "archive_schema_version",
    "batch_id",
    "batch_policy_id",
    "claim_level",
    "execution_freeze_manifest_id",
    "expected_prediction_count",
    "input_archive_id",
    "input_archive_policy_id",
    "input_archive_sha256",
    "input_archive_version",
    "input_row_ids_root",
    "prediction_record_ids_root",
    "protocol_id",
    "record_count",
    "run_context_id",
)
TRUE_CLAIMS = (
    "structural_archive_verified",
    "canonical_record_framing_verified",
    "record_schema_verified",
    "row_root_coverage_verified",
)
FALSE_CLAIMS = (
    "input_archive_membership_verified",
    "batch_policy_membership_verified",
    "source_registry_projection_verified",
    "source_public_disjoint_verified",
    "single_live_allocation_verified",
    "secret_custodian_replay_verified",
    "execution_manifest_authority_verified",
    "derived_mapping_verified",
    "recognizer_executed",
    "runtime_executed",
    "actual_960_case_run_verified",
    "recognizer_capacity_evidence",
    "origin_authenticated",
    "formal_uuid_audit",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "prediction_scored",
    "effect_evidence",
    "c1_exit_evidence",
)
DECODED_FIELDS = (
    "disposition",
    "archive",
    "archive_id",
    "schema_version",
    "policy_id",
    "context",
    "records",
    "input_row_ids",
    "prediction_record_ids",
    "prediction_content_ids",
    "claim_level",
    *TRUE_CLAIMS,
    *FALSE_CLAIMS,
)
REJECTION_FIELDS = (
    "disposition",
    "reason",
    "input_count",
    "input_archive_id",
    "archive",
    "records",
    "prediction_record_ids",
    "recognizer_capacity_evidence",
)
ROW_ROOT_FIELDS = (
    "authority_content_id",
    "envelope_id",
    "namespace_audit_id",
    "padding_sha256",
    "payload_sha256",
    "public_registry_id",
    "transform_result_id",
)


@dataclass(frozen=True, slots=True)
class _SyntheticArchiveFixtureV2:
    """Unbacked structural 960-record fixture; never scientific evidence."""

    archive: bytes
    decoded: archive_v2.DecodedRecognizerPredictionArchiveV2


def _namespace(filename: str) -> dict[str, object]:
    return runpy.run_path(str(Path(__file__).with_name(filename)))


def _hex_id(prefix: str, index: int) -> str:
    return prefix + f"{index:064x}"


def _uuid4(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012x}"


def _copy_with_pollution(value: object, **changes: object) -> object:
    copied = object.__new__(type(value))
    for item in fields(value):
        object.__setattr__(
            copied,
            item.name,
            changes.get(item.name, getattr(value, item.name)),
        )
    return copied


def _walk_semantic_strings(value: object) -> Iterator[str]:
    stack = [value]
    while stack:
        current = stack.pop()
        if type(current) is dict:
            for key, item in current.items():
                assert type(key) is str
                yield key
                stack.append(item)
        elif type(current) in (list, tuple):
            stack.extend(current)
        elif type(current) is str:
            yield current


def _prediction(
    *,
    payload_sha256: str,
    freeze_manifest_id: str,
    bundle_id: str,
) -> PredictionBundle:
    return PredictionBundle(
        schema_version=PREDICTION_SCHEMA_VERSION,
        bundle_id=bundle_id,
        input_root_sha256=payload_sha256,
        protocol_sha256=archive_v2._FROZEN_PROTOCOL.protocol_id.rsplit("_", 1)[1],
        freeze_manifest_sha256=freeze_manifest_id.rsplit("_", 1)[1],
        disposition=PredictionDisposition.ABSTAIN,
        reason=PredictionReason.RESOURCE_LIMIT,
        family_id=None,
        binding=(),
        admissible_scale_ids=(),
    )


def _outcome(
    *,
    row_id: str,
    payload_sha256: str,
    freeze_manifest_id: str,
    index: int,
) -> prediction_v2.PublicRecognizerPredictionOutcomeV2:
    return prediction_v2.PublicRecognizerPredictionOutcomeV2._issue(
        prediction_v2._OUTCOME_ISSUE_TOKEN,
        input_row_id=row_id,
        input_payload_sha256=payload_sha256,
        decision=prediction_v2.PredictionDecisionV2.ABSTAIN,
        canonical_family_id=None,
        prediction=_prediction(
            payload_sha256=payload_sha256,
            freeze_manifest_id=freeze_manifest_id,
            bundle_id=_uuid4(index + 1),
        ),
        bridge_outcome_id=_hex_id(
            "phase2b_exact_derived_preflight_v2_",
            index + 1,
        ),
        bridge_compilation_id=None,
        bridge_decision_id=None,
    )


def _positive_outcome(
    *,
    row_id: str,
    payload_sha256: str,
    freeze_manifest_id: str,
    index: int,
    decision: prediction_v2.PredictionDecisionV2,
) -> prediction_v2.PublicRecognizerPredictionOutcomeV2:
    if decision is prediction_v2.PredictionDecisionV2.ANSWER:
        scale_count = 1
    elif decision is prediction_v2.PredictionDecisionV2.ANSWER_SET:
        scale_count = 2
    else:
        raise ValueError("synthetic positive outcome needs an answering decision")
    law_kind, canonical_family_id = (
        archive_v2._FROZEN_EXACT_FREEZE.family_mapping[0]
    )
    prediction = PredictionBundle(
        schema_version=PREDICTION_SCHEMA_VERSION,
        bundle_id=_uuid4(600_000 + index),
        input_root_sha256=payload_sha256,
        protocol_sha256=archive_v2._FROZEN_PROTOCOL.protocol_id.rsplit("_", 1)[1],
        freeze_manifest_sha256=freeze_manifest_id.rsplit("_", 1)[1],
        disposition=PredictionDisposition.UNIQUE_MATCH,
        reason=PredictionReason.UNIQUE_STRUCTURAL_MATCH,
        family_id=archive_v2._BRIDGE_FAMILY_BY_KIND[law_kind],
        binding=(
            RoleBinding(
                role_id=_uuid4(610_000 + index),
                entity_id=_uuid4(620_000 + index),
            ),
        ),
        admissible_scale_ids=tuple(
            _uuid4(630_000 + 10 * index + offset)
            for offset in range(scale_count)
        ),
    )
    return prediction_v2.PublicRecognizerPredictionOutcomeV2._issue(
        prediction_v2._OUTCOME_ISSUE_TOKEN,
        input_row_id=row_id,
        input_payload_sha256=payload_sha256,
        decision=decision,
        canonical_family_id=canonical_family_id,
        prediction=prediction,
        bridge_outcome_id=_hex_id("phase2b_exact_derived_run_", 640_000 + index),
        bridge_compilation_id=_hex_id(
            "phase2b_exact_derived_bridge_result_", 650_000 + index
        ),
        bridge_decision_id=_hex_id(
            "phase2b_exact_derived_decision_", 660_000 + index
        ),
    )


def _row_id_from_roots(row: object) -> str:
    mapping = {
        name: getattr(row, name)
        for name in ROW_ROOT_FIELDS
    }
    return "phase2b_recognizer_input_row_v2_" + hashlib.sha256(
        input_v2._ROW_DOMAIN_V2 + encode_phase2b_jcs_profile_v1(mapping)
    ).hexdigest()


def _copy_root_row(value: object, **changes: object) -> object:
    copied = _copy_with_pollution(value, **changes)
    return _copy_with_pollution(copied, row_id=_row_id_from_roots(copied))


def _synthetic_root_row(index: int) -> object:
    """Return a shallow exact V2 row with synthetic roots and no trust claim."""

    row = object.__new__(input_v2.TrustedRecognizerInputRowV2)
    roots = {
        "authority_content_id": _hex_id(
            "phase2b_public_transform_evidence_", 10_000 + index
        ),
        "envelope_id": _hex_id("phase2b_trusted_envelope_v2_", 20_000 + index),
        "namespace_audit_id": _hex_id(
            "phase2b_namespace_audit_v2_", 30_000 + index
        ),
        "padding_sha256": f"{40_000 + index:064x}",
        "payload_sha256": f"{50_000 + index:064x}",
        "public_registry_id": _hex_id(
            "phase2b_public_recognizer_registry_v2_", 60_000 + index
        ),
        "transform_result_id": _hex_id(
            "phase2b_exact_transform_result_", 70_000 + index
        ),
    }
    for name, value in (
        ("envelope", b"\x00" * input_v2.ENVELOPE_BYTES),
        *roots.items(),
        ("public_registry", object.__new__(input_v2.PublicRecognizerRegistryV2)),
        (
            "row_id",
            "phase2b_recognizer_input_row_v2_" + "0" * 64,
        ),
    ):
        object.__setattr__(row, name, value)
    object.__setattr__(row, "row_id", _row_id_from_roots(row))
    return row


def _shallow_unbacked_input_archive(
    *,
    count: int = COUNT,
    nonlexicographic: bool = True,
) -> input_v2.DecodedRecognizerInputArchiveV2:
    """Forge only the public shallow orchestration surface for builder tests.

    Its bytes do not encode these rows.  Tests using it must replace the public
    input decoder with an explicit parity stub; the fixture is never presented
    as a successfully decoded or trusted input archive.
    """

    indices = list(range(count))
    if nonlexicographic and count >= 2:
        indices[0], indices[1] = indices[1], indices[0]
    rows = tuple(_synthetic_root_row(index) for index in indices)
    value = object.__new__(input_v2.DecodedRecognizerInputArchiveV2)
    fixed: dict[str, object] = {
        "disposition": input_v2.RecognizerInputArchiveDispositionV2.COMPLETE,
        "archive": b"\x00" * input_v2.ARCHIVE_HEADER_BYTES_V2,
        "archive_id": _hex_id("phase2b_recognizer_input_archive_v2_", 90_001),
        "archive_version": input_v2.TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
        "policy_id": input_v2.RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
        "batch_id": _hex_id("phase2b_trusted_wire_batch_v2_", 90_002),
        "batch_policy_id": archive_v2.TRUSTED_WIRE_BATCH_V2_POLICY_ID,
        "typed_replay_policy_id": input_v2.TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID,
        "typed_authority_schema_id": input_v2.COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
        "typed_authority_codec_version": input_v2.COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
        "typed_authority_codec_policy_id": input_v2.COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
        "public_registry_schema_id": input_v2.PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2,
        "rows": rows,
        "row_ids": tuple(row.row_id for row in rows),
        "envelope_ids": tuple(row.envelope_id for row in rows),
        "public_registry_ids": tuple(row.public_registry_id for row in rows),
        "authority_content_ids": tuple(row.authority_content_id for row in rows),
        "transform_result_ids": tuple(row.transform_result_id for row in rows),
        "claim_level": NON_AUTHORITATIVE_CLAIM_LEVEL,
    }
    for name in input_v2._TRUE_CLAIMS_V2:
        fixed[name] = True
    for name in input_v2._FALSE_CLAIMS_V2:
        fixed[name] = False
    assert set(fixed) == {item.name for item in fields(type(value))}
    for name, item in fixed.items():
        object.__setattr__(value, name, item)
    return value


def _assert_atomic_rejection(
    value: archive_v2.RecognizerPredictionArchiveRejectionV2,
    *,
    count: int,
) -> None:
    assert type(value) is archive_v2.RecognizerPredictionArchiveRejectionV2
    assert value.disposition is archive_v2.PredictionArchiveDispositionV2.ABSTAIN
    assert value.input_count == count
    assert value.archive is None
    assert value.records == ()
    assert value.prediction_record_ids == ()
    assert value.recognizer_capacity_evidence is False


def _split_frames(archive: bytes) -> tuple[bytes, list[bytes]]:
    (
        magic,
        version,
        header_bytes,
        manifest_bytes,
        record_count,
        body_sha256,
    ) = archive_v2._PREDICTION_ARCHIVE_HEADER_V2.unpack_from(archive, 0)
    assert magic == archive_v2.PREDICTION_ARCHIVE_MAGIC_V2
    assert version == archive_v2.PREDICTION_ARCHIVE_WIRE_VERSION_V2
    assert header_bytes == archive_v2.PREDICTION_ARCHIVE_HEADER_BYTES_V2
    body = archive[header_bytes:]
    assert hashlib.sha256(body).digest() == body_sha256
    manifest = body[:manifest_bytes]
    offset = manifest_bytes
    frames: list[bytes] = []
    for _ in range(record_count):
        assert offset + 4 <= len(body)
        length = int.from_bytes(body[offset : offset + 4], "big")
        offset += 4
        assert offset + length <= len(body)
        frames.append(body[offset : offset + length])
        offset += length
    assert offset == len(body)
    return manifest, frames


def _reframe(
    manifest: bytes,
    frames: list[bytes],
    *,
    record_count: int = COUNT,
) -> bytes:
    body = manifest + b"".join(
        len(frame).to_bytes(4, "big") + frame for frame in frames
    )
    header = archive_v2._PREDICTION_ARCHIVE_HEADER_V2.pack(
        archive_v2.PREDICTION_ARCHIVE_MAGIC_V2,
        archive_v2.PREDICTION_ARCHIVE_WIRE_VERSION_V2,
        archive_v2.PREDICTION_ARCHIVE_HEADER_BYTES_V2,
        len(manifest),
        record_count,
        hashlib.sha256(body).digest(),
    )
    return header + body


def _tamper_mapping(
    archive: bytes,
    *,
    frame_index: int | None,
    mutate: Callable[[dict[str, object]], None],
) -> bytes:
    manifest, frames = _split_frames(archive)
    if frame_index is None:
        mapping = decode_phase2b_jcs_profile_v1(manifest)
        assert type(mapping) is dict
        mutate(mapping)
        manifest = encode_phase2b_jcs_profile_v1(mapping)
    else:
        mapping = decode_phase2b_jcs_profile_v1(frames[frame_index])
        assert type(mapping) is dict
        mutate(mapping)
        frames[frame_index] = encode_phase2b_jcs_profile_v1(mapping)
    return _reframe(manifest, frames)


def _replace_header(archive: bytes, **changes: object) -> bytes:
    names = (
        "magic",
        "wire_version",
        "header_bytes",
        "manifest_bytes",
        "record_count",
        "body_sha256",
    )
    values = dict(
        zip(
            names,
            archive_v2._PREDICTION_ARCHIVE_HEADER_V2.unpack_from(archive, 0),
            strict=True,
        )
    )
    values.update(changes)
    return archive_v2._PREDICTION_ARCHIVE_HEADER_V2.pack(
        *(values[name] for name in names)
    ) + archive[archive_v2.PREDICTION_ARCHIVE_HEADER_BYTES_V2 :]


@pytest.fixture(scope="module")
def execution_freeze_manifest() -> ExecutionFreezeManifest:
    raw = _namespace("test_phase2b_runner.py")["_freeze_manifest"]()
    current = replace(raw, theory_version_id=initial_theory().version_id)
    assert type(current) is ExecutionFreezeManifest
    return current


@pytest.fixture(scope="module")
def synthetic_archive(
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> _SyntheticArchiveFixtureV2:
    """Build 960 unbacked private records solely to test the public codec."""

    indices = [1, 0, *range(2, COUNT)]
    rows = tuple(_synthetic_root_row(index) for index in indices)
    row_ids = tuple(row.row_id for row in rows)
    assert row_ids != tuple(sorted(row_ids))
    context = archive_v2.PublicPredictionRunContextV2._issue(
        archive_v2._CONTEXT_ISSUE_TOKEN_V2,
        batch_id=_hex_id("phase2b_trusted_wire_batch_v2_", 100_001),
        input_archive_id=_hex_id(
            "phase2b_recognizer_input_archive_v2_", 100_002
        ),
        input_archive_sha256=hashlib.sha256(
            b"synthetic-unbacked-v2-input-archive"
        ).hexdigest(),
        input_row_ids_root=archive_v2._input_row_ids_root_v2(row_ids),
        execution_freeze_manifest_id=execution_freeze_manifest.manifest_id,
    )
    records = tuple(
        archive_v2.PublicRecognizerPredictionRecordV2._issue(
            archive_v2._RECORD_ISSUE_TOKEN_V2,
            context=context,
            input_row=row,
            outcome=_outcome(
                row_id=row.row_id,
                payload_sha256=row.payload_sha256,
                freeze_manifest_id=execution_freeze_manifest.manifest_id,
                index=position,
            ),
        )
        for position, row in enumerate(rows)
    )
    archive = archive_v2._encode_prediction_archive_v2(
        context=context,
        records=records,
    )
    decoded = archive_v2.decode_public_recognizer_prediction_archive_v2(archive)
    assert type(decoded) is archive_v2.DecodedRecognizerPredictionArchiveV2
    return _SyntheticArchiveFixtureV2(archive=archive, decoded=decoded)


def test_public_surface_signatures_and_exact_field_manifests() -> None:
    assert archive_v2.__all__ == (
        "MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2",
        "MAXIMUM_PREDICTION_MANIFEST_BYTES_V2",
        "MAXIMUM_PREDICTION_RECORD_BYTES_V2",
        "PREDICTION_ARCHIVE_HEADER_BYTES_V2",
        "PREDICTION_ARCHIVE_MAGIC_V2",
        "PREDICTION_ARCHIVE_WIRE_VERSION_V2",
        "PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_ID",
        "PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION",
        "PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_ID",
        "PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_VERSION",
        "RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2",
        "RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION",
        "DecodedRecognizerPredictionArchiveV2",
        "PredictionArchiveDispositionV2",
        "PublicPredictionRunContextV2",
        "PublicRecognizerPredictionRecordV2",
        "RecognizerPredictionArchiveRejectionV2",
        "build_recognizer_prediction_archive_v2",
        "decode_public_recognizer_prediction_archive_v2",
    )
    build = inspect.signature(archive_v2.build_recognizer_prediction_archive_v2)
    decode = inspect.signature(archive_v2.decode_public_recognizer_prediction_archive_v2)
    assert tuple(build.parameters) == ("input_archive", "execution_freeze_manifest")
    assert all(
        item.kind is inspect.Parameter.KEYWORD_ONLY
        for item in build.parameters.values()
    )
    assert tuple(decode.parameters) == ("archive",)
    assert tuple(item.name for item in fields(archive_v2.PublicPredictionRunContextV2)) == CONTEXT_FIELDS
    assert tuple(item.name for item in fields(archive_v2.PublicRecognizerPredictionRecordV2)) == RECORD_FIELDS
    assert tuple(item.name for item in fields(archive_v2.DecodedRecognizerPredictionArchiveV2)) == DECODED_FIELDS
    assert tuple(item.name for item in fields(archive_v2.RecognizerPredictionArchiveRejectionV2)) == REJECTION_FIELDS
    assert archive_v2._CONTEXT_FIELDS_V2 == CONTEXT_FIELDS
    assert archive_v2._RECORD_FIELDS_V2 == RECORD_FIELDS
    assert archive_v2._MANIFEST_FIELDS_V2 == MANIFEST_FIELDS
    assert archive_v2._TRUE_DECODED_CLAIMS_V2 == TRUE_CLAIMS
    assert archive_v2._FALSE_DECODED_CLAIMS_V2 == FALSE_CLAIMS
    assert archive_v2.PREDICTION_ARCHIVE_MAGIC_V2 == b"HGP2PA2\x00"
    assert archive_v2.PREDICTION_ARCHIVE_WIRE_VERSION_V2 == 2
    assert archive_v2.PREDICTION_ARCHIVE_HEADER_BYTES_V2 == 52
    assert archive_v2.PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION == (
        "hegel-machine-phase2b-public-prediction-run-context/2"
    )
    assert archive_v2.PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_VERSION == (
        "hegel-machine-phase2b-public-recognizer-prediction-record/2"
    )
    assert archive_v2.RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION == (
        "hegel-machine-phase2b-recognizer-prediction-archive/2"
    )
    assert archive_v2.PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_ID == (
        "phase2b_public_prediction_run_context_schema_v2_"
        "63c6cc528c47e293a270953d52fff19f4b892e6ba6bb69535c96ed089ad513b4"
    )
    assert archive_v2.PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_ID == (
        "phase2b_public_recognizer_prediction_record_schema_v2_"
        "118efe94e5f454ce1a7fa2af7a4ea623773c4fe1d155bfcae602301e588583b8"
    )
    assert archive_v2.RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2 == (
        "phase2b_recognizer_prediction_archive_policy_v2_"
        "925a7e62d285ae8ea58b6c2f4ddea5111fa7482ec7957b82476b1341b41b905b"
    )


def test_policy_binds_v2_dependencies_order_and_non_evidence() -> None:
    policy = archive_v2._ARCHIVE_POLICY_VALUE_V2
    assert archive_v2._FORBIDDEN_RECOGNIZER_FIELD_TOKENS == (
        "answer",
        "case_index",
        "case_position",
        "challenge",
        "gold",
        "index",
        "main",
        "ordinal",
        "partition",
        "score",
        "metric",
    )
    assert policy["forbidden_public_semantic_tokens"] == (
        archive_v2._FORBIDDEN_RECOGNIZER_FIELD_TOKENS
    )
    assert policy["field_manifests"]["manifest"] == MANIFEST_FIELDS
    assert policy["field_manifests"]["decoded"] == DECODED_FIELDS
    assert policy["private_builder"] == {
        "exact_input_archive_type": "DecodedRecognizerInputArchiveV2",
        "exact_execution_freeze_type": "ExecutionFreezeManifest",
        "execution_freeze_validation_order": (
            "exact_manifest_type",
            "local_protocol_exact_freeze_theory_prefixed_SHA256_closure",
            "local_git_image_four_implementation_SHA_and_isolation_closure",
            "local_exact_baseline_tuple_count_items_enums_strings_SHA_and_bool_closure",
            "baseline_registration_post_init",
            "execution_manifest_post_init",
            "current_frozen_identity_comparisons",
            "manifest_content_root",
        ),
        "shallow_exact_960_before_public_input_archive_decode_or_sha256": True,
        "public_input_archive_decode_API_operation_count": 1,
        "upstream_internal_structural_parse_count": (
            "bounded_performance_property_not_a_public_claim"
        ),
        "preserve_input_wire_order": True,
        "single_row_mapper_call_count_per_row": 1,
        "atomic_rejection_without_partial_records": True,
    }
    assert policy["public_decoder"][
        "decoded_issue_reparses_archive_and_requires_supplied_parsed_exact_parity"
    ] is True
    assert policy["public_decoder"][
        "cheap_claim_and_stored_column_closure_before_deep_record_validation"
    ] is True
    assert policy["public_decoder"]["structural_true"] == TRUE_CLAIMS
    assert policy["public_decoder"]["non_evidence_false"] == FALSE_CLAIMS
    assert policy["public_decoder"]["prediction_content_uniqueness_required"] is False
    assert policy["roots"]["input_and_record_sequence_order"] == "preserved_input_wire_order"
    assert policy["scope"] == (
        "public_structural_codec_mechanics_only",
        "synthetic_fixture_not_actual_run",
        "no_runtime_capacity_scoring_effect_or_c1_claim",
    )
    assert policy["dependency_bindings"] == {
        "prediction_row_policy_transitively_binds": (
            "registry_alias_typed_replay_compact_codec_batch_prediction_bundle_"
            "freeze_theory_and_exact_derived_bridge"
        ),
        "protocol_id": archive_v2._FROZEN_PROTOCOL.protocol_id,
        "frozen_theory_version_id": archive_v2._FROZEN_THEORY.version_id,
        "exact_freeze_id": archive_v2._FROZEN_EXACT_FREEZE.freeze_id,
        "exact_freeze_family_mapping": tuple(
            (kind.value, family.value)
            for kind, family in archive_v2._FROZEN_EXACT_FREEZE.family_mapping
        ),
        "input_archive_version": input_v2.TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
        "input_archive_policy_id": input_v2.RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
        "public_registry_version": input_v2.PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION,
        "public_registry_schema_id": input_v2.PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2,
        "public_family_alias_policy_id": input_v2.PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2,
        "typed_replay_version": archive_v2.TYPED_TRUSTED_WIRE_REPLAY_V2_VERSION,
        "typed_replay_policy_id": input_v2.TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID,
        "compact_typed_authority_schema_id": input_v2.COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
        "compact_typed_authority_codec_version": input_v2.COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
        "compact_typed_authority_codec_policy_id": input_v2.COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
        "batch_schema_version": archive_v2.TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION,
        "batch_policy_id": archive_v2.TRUSTED_WIRE_BATCH_V2_POLICY_ID,
        "prediction_bundle_schema_version": PREDICTION_SCHEMA_VERSION,
        "prediction_outcome_schema_id": prediction_v2.PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_ID,
        "prediction_row_policy_id": prediction_v2.RECOGNIZER_PREDICTION_ROW_POLICY_ID_V2,
        "derived_bridge_policy_id": archive_v2.EXACT_DERIVED_BRIDGE_POLICY_ID,
        "derived_matcher_semantics_id": archive_v2.EXACT_DERIVED_MATCHER_SEMANTICS_ID,
        "derived_selection_policy_id": archive_v2.EXACT_DERIVED_SELECTION_POLICY_ID,
    }
    assert policy["canonical_json"] == {
        "accepted_profile_id": archive_v2.JCS_PROFILE_ID,
        "field_manifest_id": archive_v2.FIELD_MANIFEST_ID,
        "exact_decode_reencode": True,
    }


def test_manifest_is_exact_closed_16_fields_with_batch_context() -> None:
    assert len(MANIFEST_FIELDS) == 16
    assert MANIFEST_FIELDS[2:4] == ("batch_id", "batch_policy_id")
    assert set(MANIFEST_FIELDS) - set(CONTEXT_FIELDS) == {
        "archive_policy_id",
        "archive_schema_version",
        "prediction_record_ids_root",
        "record_count",
        "run_context_id",
    }
    assert set(CONTEXT_FIELDS) - set(MANIFEST_FIELDS) == {
        "context_id",
        "schema_version",
    }


def test_v2_preflight_and_shared_run_roots_have_distinct_prefixes(
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> None:
    outcome = _outcome(
        row_id=_hex_id("phase2b_recognizer_input_row_v2_", 1),
        payload_sha256=f"{2:064x}",
        freeze_manifest_id=execution_freeze_manifest.manifest_id,
        index=1,
    )
    assert outcome.bridge_outcome_id.startswith(
        "phase2b_exact_derived_preflight_v2_"
    )
    assert not outcome.bridge_outcome_id.startswith("phase2b_exact_derived_run_")
    with pytest.raises(ValueError, match="prefix"):
        prediction_v2.PublicRecognizerPredictionOutcomeV2._issue(
            prediction_v2._OUTCOME_ISSUE_TOKEN,
            input_row_id=outcome.input_row_id,
            input_payload_sha256=outcome.input_payload_sha256,
            decision=outcome.decision,
            canonical_family_id=None,
            prediction=outcome.prediction,
            bridge_outcome_id=_hex_id("phase2b_exact_derived_preflight_", 1),
            bridge_compilation_id=None,
            bridge_decision_id=None,
        )


def test_public_types_are_private_and_v1_types_cross_reject() -> None:
    for type_ in (
        archive_v2.PublicPredictionRunContextV2,
        archive_v2.PublicRecognizerPredictionRecordV2,
        archive_v2.DecodedRecognizerPredictionArchiveV2,
    ):
        with pytest.raises(TypeError, match="privately issued"):
            type_()
    assert archive_v2.PREDICTION_ARCHIVE_MAGIC_V2 != archive_v1.PREDICTION_ARCHIVE_MAGIC
    assert archive_v2.PREDICTION_ARCHIVE_WIRE_VERSION_V2 != archive_v1.PREDICTION_ARCHIVE_WIRE_VERSION
    with pytest.raises(TypeError):
        archive_v2.build_recognizer_prediction_archive_v2(
            input_archive=object(),
            execution_freeze_manifest=object(),
        )


def test_synthetic_unbacked_960_codec_preserves_wire_order_and_all_row_roots(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    decoded = synthetic_archive.decoded
    assert decoded.disposition is archive_v2.PredictionArchiveDispositionV2.COMPLETE
    assert decoded.archive == synthetic_archive.archive
    assert len(decoded.records) == COUNT == 960
    assert decoded.input_row_ids != tuple(sorted(decoded.input_row_ids))
    assert decoded.input_row_ids == tuple(item.input_row_id for item in decoded.records)
    assert decoded.prediction_record_ids == tuple(item.record_id for item in decoded.records)
    assert decoded.prediction_content_ids == tuple(
        item.prediction_content_id for item in decoded.records
    )
    for record in decoded.records:
        assert record.bridge_outcome_id.startswith(
            "phase2b_exact_derived_preflight_v2_"
        )
        assert record.prediction.reason is PredictionReason.RESOURCE_LIMIT
        assert (
            record.input_authority_content_id,
            record.input_envelope_id,
            record.input_namespace_audit_id,
            record.input_padding_sha256,
            record.input_payload_sha256,
            record.input_public_registry_id,
            record.input_transform_result_id,
        ) == tuple(
            getattr(record, f"input_{name}") for name in ROW_ROOT_FIELDS
        )
        record_root_mapping = {
            name: getattr(record, f"input_{name}") for name in ROW_ROOT_FIELDS
        }
        expected_row_id = "phase2b_recognizer_input_row_v2_" + hashlib.sha256(
            input_v2._ROW_DOMAIN_V2
            + encode_phase2b_jcs_profile_v1(record_root_mapping)
        ).hexdigest()
        assert record.input_row_id == expected_row_id
        assert record.prediction.input_root_sha256 == record.input_payload_sha256
        assert record.run_context_id == decoded.context.context_id
    replay = archive_v2.decode_public_recognizer_prediction_archive_v2(
        synthetic_archive.archive
    )
    assert replay == decoded
    assert archive_v2._encode_prediction_archive_v2(
        context=decoded.context,
        records=decoded.records,
    ) == synthetic_archive.archive


def test_synthetic_public_decode_claim_boundary_is_exact(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    decoded = synthetic_archive.decoded
    assert decoded.claim_level == NON_AUTHORITATIVE_CLAIM_LEVEL
    assert all(getattr(decoded, name) is True for name in TRUE_CLAIMS)
    assert all(getattr(decoded, name) is False for name in FALSE_CLAIMS)
    parsed = archive_v2._parse_prediction_archive_v2(decoded.archive)
    for name in TRUE_CLAIMS:
        polluted = _copy_with_pollution(decoded, **{name: False})
        with pytest.raises(ValueError, match="claim boundary"):
            polluted._validate(
                parsed=parsed,
                token=archive_v2._PARSED_CONTEXT_TOKEN_V2,
            )
    for name in FALSE_CLAIMS:
        polluted = _copy_with_pollution(decoded, **{name: True})
        with pytest.raises(ValueError, match="claim boundary"):
            polluted._validate(
                parsed=parsed,
                token=archive_v2._PARSED_CONTEXT_TOKEN_V2,
            )
    for name in (TRUE_CLAIMS[0], FALSE_CLAIMS[0]):
        polluted = _copy_with_pollution(decoded, **{name: 1})
        with pytest.raises(ValueError, match="claim boundary"):
            polluted._validate(
                parsed=parsed,
                token=archive_v2._PARSED_CONTEXT_TOKEN_V2,
            )


def test_synthetic_one_answer_one_answer_set_and_958_abstentions_roundtrip(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    """Exercise both positive record branches without claiming an actual run."""

    decoded = synthetic_archive.decoded
    rows = (_synthetic_root_row(1), _synthetic_root_row(0))
    decisions = (
        prediction_v2.PredictionDecisionV2.ANSWER,
        prediction_v2.PredictionDecisionV2.ANSWER_SET,
    )
    positive_records = tuple(
        archive_v2.PublicRecognizerPredictionRecordV2._issue(
            archive_v2._RECORD_ISSUE_TOKEN_V2,
            context=decoded.context,
            input_row=row,
            outcome=_positive_outcome(
                row_id=row.row_id,
                payload_sha256=row.payload_sha256,
                freeze_manifest_id=decoded.context.execution_freeze_manifest_id,
                index=index,
                decision=decision,
            ),
        )
        for index, (row, decision) in enumerate(zip(rows, decisions, strict=True))
    )
    records = (*positive_records, *decoded.records[2:])
    archive = archive_v2._encode_prediction_archive_v2(
        context=decoded.context,
        records=records,
    )
    replay = archive_v2.decode_public_recognizer_prediction_archive_v2(archive)

    assert tuple(item.decision for item in replay.records).count(
        prediction_v2.PredictionDecisionV2.ANSWER
    ) == 1
    assert tuple(item.decision for item in replay.records).count(
        prediction_v2.PredictionDecisionV2.ANSWER_SET
    ) == 1
    assert tuple(item.decision for item in replay.records).count(
        prediction_v2.PredictionDecisionV2.ABSTAIN
    ) == 958
    first, second = replay.records[:2]
    assert first.decision.value == "unique_identification"
    assert second.decision.value == "admissible_scale_set"
    for record, expected_scale_count in ((first, 1), (second, 2)):
        law_kind, expected_canonical = (
            archive_v2._FROZEN_EXACT_FREEZE.family_mapping[0]
        )
        assert record.canonical_family_id is expected_canonical
        assert record.prediction.disposition is PredictionDisposition.UNIQUE_MATCH
        assert record.prediction.reason is PredictionReason.UNIQUE_STRUCTURAL_MATCH
        assert record.prediction.family_id == archive_v2._BRIDGE_FAMILY_BY_KIND[law_kind]
        assert len(record.prediction.binding) == 1
        assert len(record.prediction.admissible_scale_ids) == expected_scale_count
        assert record.bridge_outcome_id.startswith("phase2b_exact_derived_run_")
        assert record.bridge_compilation_id is not None
        assert record.bridge_decision_id is not None
    assert all(
        item.prediction.reason is PredictionReason.RESOURCE_LIMIT
        and item.bridge_outcome_id.startswith(
            "phase2b_exact_derived_preflight_v2_"
        )
        and item.bridge_compilation_id is None
        and item.bridge_decision_id is None
        for item in replay.records[2:]
    )
    assert replay.input_row_ids == decoded.input_row_ids
    assert replay.structural_archive_verified is True
    assert replay.derived_mapping_verified is False
    assert replay.actual_960_case_run_verified is False
    assert replay.recognizer_capacity_evidence is False
    assert replay.prediction_scored is False
    assert replay.effect_evidence is False
    assert replay.c1_exit_evidence is False
    assert archive_v2._encode_prediction_archive_v2(
        context=replay.context,
        records=replay.records,
    ) == archive


def test_manifest_and_records_are_closed_and_semantically_split_blind(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    manifest_payload, record_payloads = _split_frames(synthetic_archive.archive)
    manifest = decode_phase2b_jcs_profile_v1(manifest_payload)
    records = tuple(decode_phase2b_jcs_profile_v1(item) for item in record_payloads)
    assert type(manifest) is dict
    assert tuple(sorted(manifest)) == MANIFEST_FIELDS
    assert all(type(item) is dict for item in records)
    assert all(tuple(sorted(item)) == RECORD_FIELDS for item in records)
    forbidden = archive_v2._FORBIDDEN_RECOGNIZER_FIELD_TOKENS
    for value in (manifest, *records):
        for text in _walk_semantic_strings(value):
            folded = text.casefold()
            assert all(token not in folded for token in forbidden)


def test_public_decoder_can_repeat_prediction_content_but_not_row_or_record_roots(
    synthetic_archive: _SyntheticArchiveFixtureV2,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> None:
    decoded = synthetic_archive.decoded
    rows = [_synthetic_root_row(index) for index in [1, 0, *range(2, COUNT)]]
    rows[1] = _copy_root_row(
        rows[1], payload_sha256=rows[0].payload_sha256
    )
    context = archive_v2.PublicPredictionRunContextV2._issue(
        archive_v2._CONTEXT_ISSUE_TOKEN_V2,
        batch_id=decoded.context.batch_id,
        input_archive_id=decoded.context.input_archive_id,
        input_archive_sha256=decoded.context.input_archive_sha256,
        input_row_ids_root=archive_v2._input_row_ids_root_v2(
            tuple(row.row_id for row in rows)
        ),
        execution_freeze_manifest_id=execution_freeze_manifest.manifest_id,
    )
    shared_prediction = _prediction(
        payload_sha256=rows[0].payload_sha256,
        freeze_manifest_id=execution_freeze_manifest.manifest_id,
        bundle_id=_uuid4(900_001),
    )
    records_list: list[archive_v2.PublicRecognizerPredictionRecordV2] = []
    for index, row in enumerate(rows):
        if index < 2:
            outcome = prediction_v2.PublicRecognizerPredictionOutcomeV2._issue(
                prediction_v2._OUTCOME_ISSUE_TOKEN,
                input_row_id=row.row_id,
                input_payload_sha256=row.payload_sha256,
                decision=prediction_v2.PredictionDecisionV2.ABSTAIN,
                canonical_family_id=None,
                prediction=shared_prediction,
                bridge_outcome_id=_hex_id(
                    "phase2b_exact_derived_preflight_v2_", 999_001 + index
                ),
                bridge_compilation_id=None,
                bridge_decision_id=None,
            )
        else:
            outcome = _outcome(
                row_id=row.row_id,
                payload_sha256=row.payload_sha256,
                freeze_manifest_id=execution_freeze_manifest.manifest_id,
                index=index + 1_000,
            )
        records_list.append(
            archive_v2.PublicRecognizerPredictionRecordV2._issue(
                archive_v2._RECORD_ISSUE_TOKEN_V2,
                context=context,
                input_row=row,
                outcome=outcome,
            )
        )
    records = tuple(records_list)
    archive = archive_v2._encode_prediction_archive_v2(
        context=context,
        records=records,
    )
    replay = archive_v2.decode_public_recognizer_prediction_archive_v2(archive)
    assert replay.prediction_content_ids[0] == replay.prediction_content_ids[1]
    assert replay.derived_mapping_verified is False

    with pytest.raises(ValueError, match="repeats a row or record root"):
        archive_v2._encode_prediction_archive_v2(
            context=context,
            records=(records[0], records[0], *records[2:]),
        )


def test_record_stage_shape_rejects_coherent_cross_combinations(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    base = synthetic_archive.decoded.records[0]
    mapping = archive_v2._record_mapping_without_id_v2(base)
    assert str(mapping["bridge_outcome_id"]).startswith(
        "phase2b_exact_derived_preflight_v2_"
    )
    assert mapping["bridge_compilation_id"] is None
    assert mapping["bridge_decision_id"] is None

    preflight_with_run_roots = dict(mapping)
    preflight_with_run_roots.update(
        bridge_compilation_id=_hex_id(
            "phase2b_exact_derived_bridge_result_", 1
        ),
        bridge_decision_id=_hex_id("phase2b_exact_derived_decision_", 1),
    )
    with pytest.raises(ValueError, match="preflight outcome stage shape"):
        archive_v2._record_id_v2(preflight_with_run_roots)

    run_without_roots = dict(mapping)
    run_without_roots["bridge_outcome_id"] = _hex_id(
        "phase2b_exact_derived_run_", 2
    )
    with pytest.raises(ValueError, match="derived-run outcome stage shape"):
        archive_v2._record_id_v2(run_without_roots)

    assert synthetic_archive.decoded.derived_mapping_verified is False


def test_derived_run_with_both_stage_roots_is_structurally_valid_non_evidence(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    decoded = synthetic_archive.decoded
    source = decoded.records[0]
    row = _synthetic_root_row(1)
    outcome = prediction_v2.PublicRecognizerPredictionOutcomeV2._issue(
        prediction_v2._OUTCOME_ISSUE_TOKEN,
        input_row_id=row.row_id,
        input_payload_sha256=row.payload_sha256,
        decision=prediction_v2.PredictionDecisionV2.ABSTAIN,
        canonical_family_id=None,
        prediction=source.prediction,
        bridge_outcome_id=_hex_id("phase2b_exact_derived_run_", 701_001),
        bridge_compilation_id=_hex_id(
            "phase2b_exact_derived_bridge_result_", 701_002
        ),
        bridge_decision_id=_hex_id(
            "phase2b_exact_derived_decision_", 701_003
        ),
    )
    replacement = archive_v2.PublicRecognizerPredictionRecordV2._issue(
        archive_v2._RECORD_ISSUE_TOKEN_V2,
        context=decoded.context,
        input_row=row,
        outcome=outcome,
    )
    archive = archive_v2._encode_prediction_archive_v2(
        context=decoded.context,
        records=(replacement, *decoded.records[1:]),
    )
    replay = archive_v2.decode_public_recognizer_prediction_archive_v2(archive)
    assert replay.records[0].bridge_compilation_id is not None
    assert replay.records[0].bridge_decision_id is not None
    assert replay.derived_mapping_verified is False


def test_coherent_public_prediction_forge_remains_structural_non_evidence(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    decoded = synthetic_archive.decoded
    source = decoded.records[0]
    row = _synthetic_root_row(1)
    assert row.payload_sha256 == source.input_payload_sha256
    forged_prediction = replace(
        source.prediction,
        reason=PredictionReason.INSUFFICIENT_EVIDENCE,
    )
    outcome = prediction_v2.PublicRecognizerPredictionOutcomeV2._issue(
        prediction_v2._OUTCOME_ISSUE_TOKEN,
        input_row_id=row.row_id,
        input_payload_sha256=row.payload_sha256,
        decision=prediction_v2.PredictionDecisionV2.ABSTAIN,
        canonical_family_id=None,
        prediction=forged_prediction,
        bridge_outcome_id=_hex_id("phase2b_exact_derived_run_", 991_001),
        bridge_compilation_id=_hex_id(
            "phase2b_exact_derived_bridge_result_", 991_002
        ),
        bridge_decision_id=_hex_id(
            "phase2b_exact_derived_decision_", 991_003
        ),
    )
    forged_record = archive_v2.PublicRecognizerPredictionRecordV2._issue(
        archive_v2._RECORD_ISSUE_TOKEN_V2,
        context=decoded.context,
        input_row=row,
        outcome=outcome,
    )
    records = (forged_record, *decoded.records[1:])
    archive = archive_v2._encode_prediction_archive_v2(
        context=decoded.context,
        records=records,
    )
    replay = archive_v2.decode_public_recognizer_prediction_archive_v2(archive)
    assert replay.records[0].prediction.reason is PredictionReason.INSUFFICIENT_EVIDENCE
    assert replay.structural_archive_verified is True
    assert replay.derived_mapping_verified is False
    assert replay.actual_960_case_run_verified is False
    assert replay.recognizer_capacity_evidence is False


def test_builder_stages_shallow_then_freeze_decode_and_mapper_in_wire_order(
    monkeypatch: pytest.MonkeyPatch,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> None:
    shallow = _shallow_unbacked_input_archive()
    trace: list[object] = []
    original_shallow = archive_v2._shallow_prediction_input_v2
    original_freeze = archive_v2._validate_execution_freeze_manifest_v2
    original_encode = archive_v2._encode_prediction_archive_v2
    original_output_decode = archive_v2.decode_public_recognizer_prediction_archive_v2

    def traced_shallow(value: object) -> object:
        trace.append("shallow")
        return original_shallow(value)  # type: ignore[arg-type]

    def traced_freeze(value: object) -> None:
        trace.append("freeze")
        original_freeze(value)  # type: ignore[arg-type]

    def input_decode(archive: bytes) -> input_v2.DecodedRecognizerInputArchiveV2:
        trace.append("input_decode")
        assert archive is shallow.archive
        return shallow

    mapper_calls = 0

    def mapper(
        *,
        input_row: input_v2.TrustedRecognizerInputRowV2,
        execution_freeze_manifest: ExecutionFreezeManifest,
    ) -> prediction_v2.PublicRecognizerPredictionOutcomeV2:
        nonlocal mapper_calls
        trace.append(("mapper", mapper_calls, input_row.row_id))
        assert input_row is shallow.rows[mapper_calls]
        mapper_calls += 1
        return _outcome(
            row_id=input_row.row_id,
            payload_sha256=input_row.payload_sha256,
            freeze_manifest_id=execution_freeze_manifest.manifest_id,
            index=200_000 + mapper_calls,
        )

    def output_encode(
        *,
        context: archive_v2.PublicPredictionRunContextV2,
        records: tuple[archive_v2.PublicRecognizerPredictionRecordV2, ...],
    ) -> bytes:
        trace.append("output_encode")
        assert tuple(item.input_row_id for item in records) == shallow.row_ids
        return original_encode(context=context, records=records)

    def output_decode(archive: bytes) -> archive_v2.DecodedRecognizerPredictionArchiveV2:
        trace.append("output_decode")
        return original_output_decode(archive)

    monkeypatch.setattr(archive_v2, "_shallow_prediction_input_v2", traced_shallow)
    monkeypatch.setattr(
        archive_v2, "_validate_execution_freeze_manifest_v2", traced_freeze
    )
    monkeypatch.setattr(
        archive_v2._input_v2, "decode_public_recognizer_input_archive_v2", input_decode
    )
    monkeypatch.setattr(
        archive_v2._prediction_v2, "recognize_public_input_row_v2", mapper
    )
    monkeypatch.setattr(archive_v2, "_encode_prediction_archive_v2", output_encode)
    monkeypatch.setattr(
        archive_v2, "decode_public_recognizer_prediction_archive_v2", output_decode
    )
    result = archive_v2.build_recognizer_prediction_archive_v2(
        input_archive=shallow,
        execution_freeze_manifest=execution_freeze_manifest,
    )
    assert type(result) is archive_v2.DecodedRecognizerPredictionArchiveV2
    assert trace[:3] == ["shallow", "freeze", "input_decode"]
    mapper_trace = [item for item in trace if type(item) is tuple]
    assert len(mapper_trace) == mapper_calls == COUNT
    assert tuple(item[2] for item in mapper_trace) == shallow.row_ids
    assert trace[-2:] == ["output_encode", "output_decode"]


def test_builder_never_reaches_decoder_hash_freeze_or_mapper_on_shallow_drift(
    monkeypatch: pytest.MonkeyPatch,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> None:
    malformed_inputs = tuple(
        _shallow_unbacked_input_archive(count=count)
        for count in (COUNT - 1, COUNT + 1)
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("shallow preflight must reject before deep work")

    for target, name in (
        (archive_v2, "_validate_execution_freeze_manifest_v2"),
        (archive_v2._input_v2, "decode_public_recognizer_input_archive_v2"),
        (archive_v2._prediction_v2, "recognize_public_input_row_v2"),
        (archive_v2.hashlib, "sha256"),
    ):
        monkeypatch.setattr(target, name, forbidden)

    for malformed in malformed_inputs:
        result = archive_v2.build_recognizer_prediction_archive_v2(
            input_archive=malformed,
            execution_freeze_manifest=execution_freeze_manifest,
        )
        _assert_atomic_rejection(result, count=len(malformed.rows))
        assert result.reason == "input_row_count_not_exact_960"


def test_builder_closes_row_types_ids_and_stored_columns_before_deep_work(
    monkeypatch: pytest.MonkeyPatch,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> None:
    base = _shallow_unbacked_input_archive()

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("shallow row drift reached deep work")

    for target, name in (
        (archive_v2, "_validate_execution_freeze_manifest_v2"),
        (archive_v2._input_v2, "decode_public_recognizer_input_archive_v2"),
        (archive_v2._prediction_v2, "recognize_public_input_row_v2"),
        (archive_v2.hashlib, "sha256"),
    ):
        monkeypatch.setattr(target, name, forbidden)

    wrong_type_rows = (object(), *base.rows[1:])
    wrong_type = _copy_with_pollution(base, rows=wrong_type_rows)
    invalid_id_row = _copy_with_pollution(base.rows[0], row_id="bad")
    invalid_id = _copy_with_pollution(
        base,
        rows=(invalid_id_row, *base.rows[1:]),
        row_ids=("bad", *base.row_ids[1:]),
    )
    stored_drift = _copy_with_pollution(
        base,
        row_ids=(base.row_ids[1], *base.row_ids[1:]),
    )
    for malformed in (wrong_type, invalid_id, stored_drift):
        result = archive_v2.build_recognizer_prediction_archive_v2(
            input_archive=malformed,
            execution_freeze_manifest=execution_freeze_manifest,
        )
        _assert_atomic_rejection(result, count=COUNT)


def test_bad_freeze_rejects_before_input_decode_or_mapper(
    monkeypatch: pytest.MonkeyPatch,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> None:
    shallow = _shallow_unbacked_input_archive()

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("bad freeze reached decoder or mapper")

    monkeypatch.setattr(
        archive_v2._input_v2,
        "decode_public_recognizer_input_archive_v2",
        forbidden,
    )
    monkeypatch.setattr(
        archive_v2._prediction_v2,
        "recognize_public_input_row_v2",
        forbidden,
    )
    bad = replace(execution_freeze_manifest, theory_version_id="future_theory")
    result = archive_v2.build_recognizer_prediction_archive_v2(
        input_archive=shallow,
        execution_freeze_manifest=bad,
    )
    _assert_atomic_rejection(result, count=COUNT)


def test_freeze_scalar_formats_reject_before_dynamic_authority_work(
    monkeypatch: pytest.MonkeyPatch,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> None:
    malformed = (
        _copy_with_pollution(
            execution_freeze_manifest,
            git_commit="not-a-full-commit",
        ),
        _copy_with_pollution(
            execution_freeze_manifest,
            recognizer_image_digest="sha256:bad",
        ),
        _copy_with_pollution(
            execution_freeze_manifest,
            configuration_sha256="bad",
        ),
    )
    assert all(type(item) is ExecutionFreezeManifest for item in malformed)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("malformed freeze scalar reached dynamic authority work")

    monkeypatch.setattr(
        phase2b_protocol.BaselineRegistration,
        "__post_init__",
        forbidden,
    )
    monkeypatch.setattr(
        phase2b_protocol.ExecutionFreezeManifest,
        "__post_init__",
        forbidden,
    )
    monkeypatch.setattr(phase2b_protocol, "frozen_phase2b_protocol", forbidden)
    monkeypatch.setattr(phase2b_protocol, "frozen_phase2b_exact_freeze", forbidden)
    monkeypatch.setattr(phase2b_protocol, "stable_hash", forbidden)
    monkeypatch.setattr(archive_v2, "stable_hash", forbidden)
    monkeypatch.setattr(archive_v2.hashlib, "sha256", forbidden)
    for manifest in malformed:
        with pytest.raises(ValueError):
            archive_v2._validate_execution_freeze_manifest_v2(manifest)


def test_public_input_self_decode_parity_rejects_before_mapper(
    monkeypatch: pytest.MonkeyPatch,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> None:
    shallow = _shallow_unbacked_input_archive()
    canonical_drift = _copy_with_pollution(
        shallow,
        row_ids=(shallow.row_ids[1], shallow.row_ids[0], *shallow.row_ids[2:]),
    )
    decode_calls = 0

    def decoder(archive: bytes) -> input_v2.DecodedRecognizerInputArchiveV2:
        nonlocal decode_calls
        decode_calls += 1
        return canonical_drift

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("self-decode parity drift reached mapper")

    monkeypatch.setattr(
        archive_v2._input_v2, "decode_public_recognizer_input_archive_v2", decoder
    )
    monkeypatch.setattr(
        archive_v2._prediction_v2,
        "recognize_public_input_row_v2",
        forbidden,
    )
    result = archive_v2.build_recognizer_prediction_archive_v2(
        input_archive=shallow,
        execution_freeze_manifest=execution_freeze_manifest,
    )
    _assert_atomic_rejection(result, count=COUNT)
    assert decode_calls == 1


@pytest.mark.parametrize("failure_index", (0, COUNT // 2, COUNT - 1))
def test_mapper_failure_first_middle_or_last_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
    execution_freeze_manifest: ExecutionFreezeManifest,
    failure_index: int,
) -> None:
    shallow = _shallow_unbacked_input_archive()
    monkeypatch.setattr(
        archive_v2._input_v2,
        "decode_public_recognizer_input_archive_v2",
        lambda archive: shallow,
    )
    calls = 0

    def mapper(
        *,
        input_row: input_v2.TrustedRecognizerInputRowV2,
        execution_freeze_manifest: ExecutionFreezeManifest,
    ) -> prediction_v2.PublicRecognizerPredictionOutcomeV2:
        nonlocal calls
        index = calls
        calls += 1
        if index == failure_index:
            raise ValueError("synthetic mapper failure")
        return _outcome(
            row_id=input_row.row_id,
            payload_sha256=input_row.payload_sha256,
            freeze_manifest_id=execution_freeze_manifest.manifest_id,
            index=400_000 + index,
        )

    monkeypatch.setattr(
        archive_v2._prediction_v2, "recognize_public_input_row_v2", mapper
    )
    result = archive_v2.build_recognizer_prediction_archive_v2(
        input_archive=shallow,
        execution_freeze_manifest=execution_freeze_manifest,
    )
    _assert_atomic_rejection(result, count=COUNT)
    assert calls == failure_index + 1


def test_post_mapping_encode_and_public_decode_failures_are_atomic(
    monkeypatch: pytest.MonkeyPatch,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> None:
    shallow = _shallow_unbacked_input_archive()
    monkeypatch.setattr(
        archive_v2._input_v2,
        "decode_public_recognizer_input_archive_v2",
        lambda archive: shallow,
    )
    monkeypatch.setattr(
        archive_v2._prediction_v2,
        "recognize_public_input_row_v2",
        lambda *, input_row, execution_freeze_manifest: _outcome(
            row_id=input_row.row_id,
            payload_sha256=input_row.payload_sha256,
            freeze_manifest_id=execution_freeze_manifest.manifest_id,
            index=500_000,
        ),
    )
    original_encode = archive_v2._encode_prediction_archive_v2

    def failed_encode(**kwargs: object) -> bytes:
        raise ValueError("synthetic post-map encoding failure")

    monkeypatch.setattr(archive_v2, "_encode_prediction_archive_v2", failed_encode)
    result = archive_v2.build_recognizer_prediction_archive_v2(
        input_archive=shallow,
        execution_freeze_manifest=execution_freeze_manifest,
    )
    _assert_atomic_rejection(result, count=COUNT)

    monkeypatch.setattr(archive_v2, "_encode_prediction_archive_v2", original_encode)
    original_decode = archive_v2.decode_public_recognizer_prediction_archive_v2

    def failed_decode(archive: bytes) -> object:
        original_decode(archive)
        raise ValueError("synthetic post-encode decode failure")

    monkeypatch.setattr(
        archive_v2, "decode_public_recognizer_prediction_archive_v2", failed_decode
    )
    result = archive_v2.build_recognizer_prediction_archive_v2(
        input_archive=shallow,
        execution_freeze_manifest=execution_freeze_manifest,
    )
    _assert_atomic_rejection(result, count=COUNT)


def test_builder_uses_no_v1_or_private_input_archive_helpers(
    monkeypatch: pytest.MonkeyPatch,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> None:
    shallow = _shallow_unbacked_input_archive()

    forbidden_apis = (
        (archive_v1, "build_recognizer_prediction_archive_v1"),
        (archive_v1, "decode_public_recognizer_prediction_archive_v1"),
        (archive_v1, "recognize_public_input_row_v1"),
        (input_v1, "decode_public_recognizer_input_archive_v1"),
        (input_v2, "_parse_archive_v2"),
        (input_v2, "_DECODE_TOKEN_V2"),
    )
    forbidden_identities = tuple(
        getattr(target, name) for target, name in forbidden_apis
    ) + (input_v2._ROW_TOKEN_V2,)
    module_values = tuple(vars(archive_v2).values())
    builder_values = tuple(
        archive_v2.build_recognizer_prediction_archive_v2.__globals__.values()
    )
    for original in forbidden_identities:
        assert all(original is not value for value in module_values)
        assert all(original is not value for value in builder_values)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("V2 archive builder called a forbidden helper")

    for target, name in forbidden_apis:
        monkeypatch.setattr(target, name, forbidden)

    monkeypatch.setattr(
        archive_v2._input_v2,
        "decode_public_recognizer_input_archive_v2",
        lambda archive: shallow,
    )
    monkeypatch.setattr(
        archive_v2._prediction_v2,
        "recognize_public_input_row_v2",
        lambda *, input_row, execution_freeze_manifest: _outcome(
            row_id=input_row.row_id,
            payload_sha256=input_row.payload_sha256,
            freeze_manifest_id=execution_freeze_manifest.manifest_id,
            index=300_000,
        ),
    )
    result = archive_v2.build_recognizer_prediction_archive_v2(
        input_archive=shallow,
        execution_freeze_manifest=execution_freeze_manifest,
    )
    assert type(result) is archive_v2.DecodedRecognizerPredictionArchiveV2


def test_public_byte_decoder_never_calls_mapper_or_input_archive_apis(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("public byte decoder called nonstructural API")

    for target, name in (
        (archive_v2._prediction_v2, "recognize_public_input_row_v2"),
        (archive_v2._input_v2, "decode_public_recognizer_input_archive_v2"),
        (archive_v2._input_v2, "_parse_archive_v2"),
        (archive_v1, "decode_public_recognizer_prediction_archive_v1"),
    ):
        monkeypatch.setattr(target, name, forbidden)
    decoded = archive_v2.decode_public_recognizer_prediction_archive_v2(
        synthetic_archive.archive
    )
    assert decoded.structural_archive_verified is True
    assert decoded.derived_mapping_verified is False


def test_prediction_bundle_object_pollution_rejects_before_mapping_or_hash(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    class StringSubclass(str):
        pass

    base = synthetic_archive.decoded.records[0]
    polluted = replace(
        base.prediction,
        schema_version=StringSubclass(base.prediction.schema_version),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("polluted PredictionBundle reached mapping or hash")

    monkeypatch.setattr(PredictionBundle, "to_mapping", forbidden)
    monkeypatch.setattr(archive_v2.hashlib, "sha256", forbidden)
    monkeypatch.setattr(archive_v2, "stable_hash", forbidden)
    with pytest.raises((TypeError, ValueError), match="prediction"):
        archive_v2._prediction_mapping_v2(polluted)


def test_record_nested_wire_types_reject_closed_schemas(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    class StringSubclass(str):
        pass

    base = archive_v2._record_mapping_without_id_v2(
        synthetic_archive.decoded.records[0]
    )
    variants: list[dict[str, object]] = []
    for name, value in (
        ("decision", StringSubclass(base["decision"])),
        ("input_padding_sha256", StringSubclass(base["input_padding_sha256"])),
    ):
        mapping = dict(base)
        mapping[name] = value
        variants.append(mapping)
    for nested_name, value in (
        ("schema_version", StringSubclass(PREDICTION_SCHEMA_VERSION)),
        ("binding", ()),
        ("admissible_scale_ids", ()),
        ("reason", True),
    ):
        mapping = dict(base)
        prediction = dict(mapping["prediction"])  # type: ignore[arg-type]
        prediction[nested_name] = value
        mapping["prediction"] = prediction
        variants.append(mapping)

    for mapping in variants:
        with pytest.raises((TypeError, ValueError)):
            archive_v2._record_id_v2(mapping)


def test_decoded_private_issue_reparses_and_binds_exact_archive_bytes(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    parsed = archive_v2._parse_prediction_archive_v2(synthetic_archive.archive)
    tampered = synthetic_archive.archive[:-1] + bytes(
        [synthetic_archive.archive[-1] ^ 1]
    )
    calls = 0
    original_parse = archive_v2._parse_prediction_archive_v2

    def traced_parse(archive: bytes) -> object:
        nonlocal calls
        calls += 1
        return original_parse(archive)

    monkeypatch.setattr(archive_v2, "_parse_prediction_archive_v2", traced_parse)
    with pytest.raises(ValueError):
        archive_v2.DecodedRecognizerPredictionArchiveV2._issue(
            archive_v2._DECODE_ISSUE_TOKEN_V2,
            archive=tampered,
            parsed=parsed,
        )
    assert calls == 1
    with pytest.raises(TypeError, match="token"):
        archive_v2.DecodedRecognizerPredictionArchiveV2._issue(
            object(),
            archive=synthetic_archive.archive,
            parsed=parsed,
        )


def test_decoded_private_issue_rejects_exact_fake_parsed_for_other_valid_bytes(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    old = synthetic_archive.decoded
    source = old.records[0]
    row = _synthetic_root_row(1)
    forged_prediction = replace(
        source.prediction,
        reason=PredictionReason.INSUFFICIENT_EVIDENCE,
    )
    outcome = prediction_v2.PublicRecognizerPredictionOutcomeV2._issue(
        prediction_v2._OUTCOME_ISSUE_TOKEN,
        input_row_id=row.row_id,
        input_payload_sha256=row.payload_sha256,
        decision=prediction_v2.PredictionDecisionV2.ABSTAIN,
        canonical_family_id=None,
        prediction=forged_prediction,
        bridge_outcome_id=_hex_id("phase2b_exact_derived_run_", 992_001),
        bridge_compilation_id=_hex_id(
            "phase2b_exact_derived_bridge_result_", 992_002
        ),
        bridge_decision_id=_hex_id(
            "phase2b_exact_derived_decision_", 992_003
        ),
    )
    replacement = archive_v2.PublicRecognizerPredictionRecordV2._issue(
        archive_v2._RECORD_ISSUE_TOKEN_V2,
        context=old.context,
        input_row=row,
        outcome=outcome,
    )
    other_archive = archive_v2._encode_prediction_archive_v2(
        context=old.context,
        records=(replacement, *old.records[1:]),
    )
    current = archive_v2._parse_prediction_archive_v2(other_archive)
    assert current.records != old.records
    fake = archive_v2._ParsedPredictionArchiveV2(
        archive_id=current.archive_id,
        context=old.context,
        records=old.records,
    )
    with pytest.raises(ValueError, match="supplied parsed context drift"):
        archive_v2.DecodedRecognizerPredictionArchiveV2._issue(
            archive_v2._DECODE_ISSUE_TOKEN_V2,
            archive=other_archive,
            parsed=fake,
        )


def test_decoded_claim_and_stored_columns_close_before_record_deep_replay(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    decoded = synthetic_archive.decoded

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("cheap decoded closure must precede record replay")

    monkeypatch.setattr(
        archive_v2.PublicRecognizerPredictionRecordV2,
        "_validate",
        forbidden,
    )
    for polluted in (
        _copy_with_pollution(decoded, structural_archive_verified=False),
        _copy_with_pollution(decoded, runtime_executed=True),
        _copy_with_pollution(decoded, input_row_ids=decoded.input_row_ids[:-1]),
        _copy_with_pollution(
            decoded,
            prediction_record_ids=decoded.prediction_record_ids[:-1],
        ),
        _copy_with_pollution(
            decoded,
            prediction_content_ids=decoded.prediction_content_ids[:-1],
        ),
    ):
        with pytest.raises((TypeError, ValueError)):
            polluted._validate()


def test_v1_v2_prediction_decoders_cross_reject(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        archive_v1.decode_public_recognizer_prediction_archive_v1(
            synthetic_archive.archive
        )
    wrong_magic = _replace_header(
        synthetic_archive.archive,
        magic=archive_v1.PREDICTION_ARCHIVE_MAGIC,
    )
    with pytest.raises(ValueError, match="header discriminator"):
        archive_v2.decode_public_recognizer_prediction_archive_v2(wrong_magic)


def test_decoder_exact_bytes_and_size_gate_precede_hash(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    class BytesSubclass(bytes):
        pass

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid archive reached SHA-256")

    monkeypatch.setattr(archive_v2.hashlib, "sha256", forbidden)
    for value in (b"", BytesSubclass(synthetic_archive.archive)):
        with pytest.raises((TypeError, ValueError)):
            archive_v2.decode_public_recognizer_prediction_archive_v2(value)


def test_header_discriminators_counts_lengths_and_body_digest_reject(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    archive = synthetic_archive.archive
    variants = (
        _replace_header(archive, magic=b"HGP2BAD\x00"),
        _replace_header(archive, wire_version=1),
        _replace_header(archive, header_bytes=0),
        _replace_header(archive, manifest_bytes=0),
        _replace_header(
            archive,
            manifest_bytes=archive_v2.MAXIMUM_PREDICTION_MANIFEST_BYTES_V2 + 1,
        ),
        _replace_header(archive, record_count=COUNT - 1),
        _replace_header(archive, record_count=COUNT + 1),
        _replace_header(archive, body_sha256=b"\x00" * 32),
    )
    for malformed in variants:
        with pytest.raises(ValueError):
            archive_v2.decode_public_recognizer_prediction_archive_v2(malformed)


def test_manifest_closed_schema_exact_int_and_context_roots_reject(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    variants = (
        _tamper_mapping(
            synthetic_archive.archive,
            frame_index=None,
            mutate=lambda value: value.update(future_field="x"),
        ),
        _tamper_mapping(
            synthetic_archive.archive,
            frame_index=None,
            mutate=lambda value: value.__setitem__("record_count", True),
        ),
        _tamper_mapping(
            synthetic_archive.archive,
            frame_index=None,
            mutate=lambda value: value.__setitem__(
                "batch_id", _hex_id("phase2b_trusted_wire_batch_v2_", 888_001)
            ),
        ),
        _tamper_mapping(
            synthetic_archive.archive,
            frame_index=None,
            mutate=lambda value: value.__setitem__(
                "prediction_record_ids_root",
                _hex_id("phase2b_prediction_records_v2_", 888_002),
            ),
        ),
    )
    for malformed in variants:
        with pytest.raises((TypeError, ValueError)):
            archive_v2.decode_public_recognizer_prediction_archive_v2(malformed)


@pytest.mark.parametrize(
    "count_field",
    ("expected_prediction_count", "record_count"),
)
def test_manifest_counts_reject_before_frozen_protocol_or_hash(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_archive: _SyntheticArchiveFixtureV2,
    count_field: str,
) -> None:
    manifest_payload, _ = _split_frames(synthetic_archive.archive)
    manifest = decode_phase2b_jcs_profile_v1(manifest_payload)
    assert type(manifest) is dict
    manifest[count_field] = True

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid manifest count reached protocol/hash work")

    # The protocol ID is a dynamic property backed by the protocol module's
    # stable hash.  Both that hash and this module's context-root hash remain
    # unreachable until exact builtin count validation has passed.
    monkeypatch.setattr(phase2b_protocol, "stable_hash", forbidden)
    monkeypatch.setattr(archive_v2, "stable_hash", forbidden)
    monkeypatch.setattr(archive_v2.hashlib, "sha256", forbidden)
    with pytest.raises(ValueError, match="manifest count drift"):
        archive_v2._validate_manifest_v2(manifest)


def test_record_extra_unknown_reason_root_and_noncanonical_payload_reject(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    variants = (
        _tamper_mapping(
            synthetic_archive.archive,
            frame_index=0,
            mutate=lambda value: value.update(future_field="x"),
        ),
        _tamper_mapping(
            synthetic_archive.archive,
            frame_index=0,
            mutate=lambda value: value["prediction"].__setitem__(  # type: ignore[union-attr]
                "reason", "future_reason"
            ),
        ),
        _tamper_mapping(
            synthetic_archive.archive,
            frame_index=0,
            mutate=lambda value: value.__setitem__(
                "input_authority_content_id",
                _hex_id("phase2b_public_transform_evidence_", 777_001),
            ),
        ),
    )
    manifest, frames = _split_frames(synthetic_archive.archive)
    frames[0] = b" " + frames[0]
    variants = (*variants, _reframe(manifest, frames))
    for malformed in variants:
        with pytest.raises((TypeError, ValueError)):
            archive_v2.decode_public_recognizer_prediction_archive_v2(malformed)


def test_reordered_duplicate_and_trailing_record_frames_reject(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    manifest, original = _split_frames(synthetic_archive.archive)
    reordered = list(original)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    duplicated = list(original)
    duplicated[1] = duplicated[0]
    trailing = _reframe(manifest, list(original)) + b"x"
    for malformed in (
        _reframe(manifest, reordered),
        _reframe(manifest, duplicated),
        trailing,
    ):
        with pytest.raises(ValueError):
            archive_v2.decode_public_recognizer_prediction_archive_v2(malformed)


def test_forbidden_semantic_keys_and_values_reject_directly() -> None:
    for value in (
        {"gold_label": "safe"},
        {"safe": "future_partition"},
        {"safe": ["case_position_1"]},
        {"score": "safe"},
        {"safe": "score"},
        {"metric": "safe"},
        {"safe": "future_metric"},
    ):
        with pytest.raises(ValueError, match="forbidden"):
            archive_v2._reject_forbidden_public_semantics_v2(value)


def test_encoder_requires_exact_960_tuple_and_exact_records(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    decoded = synthetic_archive.decoded
    for records in (
        list(decoded.records),
        decoded.records[:-1],
        (*decoded.records, decoded.records[-1]),
        (object(), *decoded.records[1:]),
    ):
        with pytest.raises(TypeError, match="exactly 960 exact records"):
            archive_v2._encode_prediction_archive_v2(
                context=decoded.context,
                records=records,  # type: ignore[arg-type]
            )


def test_context_and_sequence_roots_validate_exact_scalars_before_hash(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    context = synthetic_archive.decoded.context
    preimage = archive_v2._context_mapping_without_id_v2(context)
    bad_context = dict(preimage)
    bad_context["expected_prediction_count"] = True
    bad_rows = (*synthetic_archive.decoded.input_row_ids[:-1], "bad")

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid root input reached hash")

    monkeypatch.setattr(archive_v2, "stable_hash", forbidden)
    monkeypatch.setattr(archive_v2.hashlib, "sha256", forbidden)
    with pytest.raises(ValueError, match="prediction count"):
        archive_v2._context_id_v2(bad_context)
    with pytest.raises(ValueError, match="prefixed SHA-256"):
        archive_v2._input_row_ids_root_v2(bad_rows)


def test_record_frame_length_cap_rejects_before_payload_decode(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    archive = synthetic_archive.archive
    header_values = archive_v2._PREDICTION_ARCHIVE_HEADER_V2.unpack_from(archive, 0)
    manifest_length = header_values[3]
    offset = archive_v2.PREDICTION_ARCHIVE_HEADER_BYTES_V2 + manifest_length
    malformed = bytearray(archive)
    malformed[offset : offset + 4] = (
        archive_v2.MAXIMUM_PREDICTION_RECORD_BYTES_V2 + 1
    ).to_bytes(4, "big")
    body = bytes(malformed[archive_v2.PREDICTION_ARCHIVE_HEADER_BYTES_V2 :])
    malformed[: archive_v2.PREDICTION_ARCHIVE_HEADER_BYTES_V2] = (
        archive_v2._PREDICTION_ARCHIVE_HEADER_V2.pack(
            *header_values[:-1],
            hashlib.sha256(body).digest(),
        )
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("oversized frame reached JCS decoder")

    monkeypatch.setattr(archive_v2, "decode_phase2b_jcs_profile_v1", forbidden)
    # The manifest must still decode before the first record length is reached.
    # Count calls by allowing exactly that one canonical manifest operation.
    original_decode = decode_phase2b_jcs_profile_v1
    calls = 0

    def manifest_only(payload: bytes) -> object:
        nonlocal calls
        calls += 1
        if calls > 1:
            return forbidden(payload)
        return original_decode(payload)

    monkeypatch.setattr(
        archive_v2, "decode_phase2b_jcs_profile_v1", manifest_only
    )
    with pytest.raises(ValueError, match="length cap drift"):
        archive_v2.decode_public_recognizer_prediction_archive_v2(bytes(malformed))
    assert calls == 1


def test_rejection_object_requires_exact_empty_non_evidence_shape() -> None:
    base = archive_v2.RecognizerPredictionArchiveRejectionV2(
        disposition=archive_v2.PredictionArchiveDispositionV2.ABSTAIN,
        reason="synthetic_failure",
        input_count=COUNT,
        input_archive_id=None,
    )
    assert tuple(item.name for item in fields(base)) == REJECTION_FIELDS
    for polluted in (
        _copy_with_pollution(base, input_count=True),
        _copy_with_pollution(base, archive=b"x"),
        _copy_with_pollution(base, records=(object(),)),
        _copy_with_pollution(base, prediction_record_ids=("x",)),
        _copy_with_pollution(base, recognizer_capacity_evidence=True),
    ):
        with pytest.raises((TypeError, ValueError)):
            polluted.__post_init__()


def test_record_issue_validates_exact_outcome_before_row_or_context_comparison(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    base_record = synthetic_archive.decoded.records[0]
    row = _synthetic_root_row(1)
    outcome = _outcome(
        row_id=row.row_id,
        payload_sha256=row.payload_sha256,
        freeze_manifest_id=synthetic_archive.decoded.context.execution_freeze_manifest_id,
        index=880_001,
    )
    polluted = _copy_with_pollution(outcome, decision="abstain")

    class HostileString(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError("polluted outcome reached row/context comparison")

    hostile_row = _copy_with_pollution(
        row,
        row_id=HostileString(base_record.input_row_id),
    )
    with pytest.raises(TypeError, match="decision exact enum"):
        archive_v2.PublicRecognizerPredictionRecordV2._issue(
            archive_v2._RECORD_ISSUE_TOKEN_V2,
            context=synthetic_archive.decoded.context,
            input_row=hostile_row,
            outcome=polluted,
        )


def test_record_issue_closes_exact_input_row_roots_before_outcome_comparison(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    row = _synthetic_root_row(1)
    outcome = _outcome(
        row_id=row.row_id,
        payload_sha256=row.payload_sha256,
        freeze_manifest_id=synthetic_archive.decoded.context.execution_freeze_manifest_id,
        index=880_002,
    )

    class HostileString(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError("hostile exact input row reached outcome comparison")

    # The containing row remains the exact private row class; only a stored
    # root scalar is polluted.  Local exact-root closure must reject it before
    # the legal outcome can be compared with that hostile scalar.
    hostile_row = _copy_with_pollution(
        row,
        row_id=HostileString(row.row_id),
    )
    assert type(hostile_row) is input_v2.TrustedRecognizerInputRowV2
    with pytest.raises(ValueError, match="prefixed SHA-256"):
        archive_v2.PublicRecognizerPredictionRecordV2._issue(
            archive_v2._RECORD_ISSUE_TOKEN_V2,
            context=synthetic_archive.decoded.context,
            input_row=hostile_row,
            outcome=outcome,
        )


def test_record_mapping_rejects_hostile_decision_and_family_before_value_access(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    class HostileValue:
        @property
        def value(self) -> object:
            raise AssertionError("hostile record discriminator property was accessed")

    base = synthetic_archive.decoded.records[0]
    for changes in (
        {"decision": HostileValue()},
        {"canonical_family_id": HostileValue()},
    ):
        polluted = _copy_with_pollution(base, **changes)
        with pytest.raises(
            TypeError,
            match="mapping (decision exact enum|canonical family type) drift",
        ):
            archive_v2._record_mapping_without_id_v2(polluted)


def test_shallow_stored_column_subclass_rejects_before_deep_builder_work(
    monkeypatch: pytest.MonkeyPatch,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> None:
    class HostileString(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError("stored root subclass reached parity comparison")

    base = _shallow_unbacked_input_archive()
    polluted = _copy_with_pollution(
        base,
        row_ids=(HostileString(base.row_ids[0]), *base.row_ids[1:]),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("stored-column subclass reached deep builder work")

    for target, name in (
        (archive_v2, "_validate_execution_freeze_manifest_v2"),
        (archive_v2._input_v2, "decode_public_recognizer_input_archive_v2"),
        (archive_v2._prediction_v2, "recognize_public_input_row_v2"),
        (archive_v2.hashlib, "sha256"),
    ):
        monkeypatch.setattr(target, name, forbidden)
    result = archive_v2.build_recognizer_prediction_archive_v2(
        input_archive=polluted,
        execution_freeze_manifest=execution_freeze_manifest,
    )
    _assert_atomic_rejection(result, count=COUNT)


@pytest.mark.parametrize(
    "field_name",
    ("input_row_id", "record_id", "prediction_content_id"),
)
def test_decoded_record_id_subclass_rejects_before_column_parity_or_deep_replay(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_archive: _SyntheticArchiveFixtureV2,
    field_name: str,
) -> None:
    class HostileString(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError("record ID subclass reached column parity")

    decoded = synthetic_archive.decoded
    first = decoded.records[0]
    polluted_first = _copy_with_pollution(
        first,
        **{field_name: HostileString(getattr(first, field_name))},
    )
    polluted = _copy_with_pollution(
        decoded,
        records=(polluted_first, *decoded.records[1:]),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("record ID subclass reached deep replay")

    monkeypatch.setattr(
        archive_v2.PublicRecognizerPredictionRecordV2,
        "_validate",
        forbidden,
    )
    with pytest.raises(ValueError, match="prefixed SHA-256"):
        polluted._validate()


def test_preflight_stage_rejects_non_resource_prediction_reason_explicitly(
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    base = archive_v2._record_mapping_without_id_v2(
        synthetic_archive.decoded.records[0]
    )
    prediction = dict(base["prediction"])  # type: ignore[arg-type]
    prediction["reason"] = PredictionReason.INSUFFICIENT_EVIDENCE.value
    base["prediction"] = prediction
    # This coherent content root ensures rejection is specifically the public
    # preflight-reason gate, not stale prediction-content identity.
    base["prediction_content_id"] = PredictionBundle.from_mapping(
        prediction
    ).content_id
    with pytest.raises(ValueError, match="preflight outcome public reason drift"):
        archive_v2._record_id_v2(base)


def test_record_and_manifest_shape_caps_reject_before_semantic_scanner(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_archive: _SyntheticArchiveFixtureV2,
) -> None:
    record = archive_v2._record_mapping_without_id_v2(
        synthetic_archive.decoded.records[0]
    )
    bad_record = dict(record)
    bad_prediction = dict(bad_record["prediction"])  # type: ignore[arg-type]
    bad_prediction["binding"] = [object()] * 65
    bad_record["prediction"] = bad_prediction
    manifest_payload, _ = _split_frames(synthetic_archive.archive)
    manifest = decode_phase2b_jcs_profile_v1(manifest_payload)
    assert type(manifest) is dict
    bad_manifest_count = dict(manifest)
    bad_manifest_count["record_count"] = True
    bad_manifest_oversized = dict(manifest)
    bad_manifest_oversized["batch_id"] = "x" * (
        archive_v2.MAXIMUM_ASCII_STRING_BYTES + 1
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid shape reached semantic scanner")

    monkeypatch.setattr(
        archive_v2,
        "_reject_forbidden_public_semantics_v2",
        forbidden,
    )
    with pytest.raises(TypeError, match="binding wire drift"):
        archive_v2._validate_record_preimage_v2(bad_record)
    with pytest.raises(ValueError, match="manifest count drift"):
        archive_v2._validate_manifest_v2(bad_manifest_count)
    with pytest.raises(ValueError, match="prefixed SHA-256"):
        archive_v2._validate_manifest_v2(bad_manifest_oversized)
