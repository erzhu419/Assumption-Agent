"""Adversarial contracts for one ephemeral compact-V2 prediction mapping.

The public result is mechanics-only.  These tests deliberately do not treat a
successful row mapping as an input-archive receipt, execution evidence,
capacity evidence, a score, an effect estimate, or C1 exit evidence.
"""

from __future__ import annotations

from dataclasses import fields, replace
import inspect
import json
from pathlib import Path
import runpy

import pytest

from hegel_machine.bootstrap import initial_theory
import hegel_machine.phase2b_exact_derived_witness_bridge_v1 as derived_bridge
import hegel_machine.phase2b_exact_transform_semantics_v1 as exact_transform_semantics
import hegel_machine.phase2b_recognizer_input_archive_v1 as archive_v1
import hegel_machine.phase2b_recognizer_input_archive_v2 as archive_v2
import hegel_machine.phase2b_recognizer_prediction_archive_v1 as prediction_v1
import hegel_machine.phase2b_recognizer_prediction_v2 as prediction_v2
import hegel_machine.phase2b_trusted_wire_batch_v2 as batch_v2
import hegel_machine.phase2b_trusted_wire_typed_authority_v1 as typed_authority_v1
import hegel_machine.phase2b_trusted_wire_typed_replay_v1 as typed_replay_v1
import hegel_machine.phase2b_trusted_wire_typed_replay_v2 as typed_replay_v2
from hegel_machine.phase2b_freeze_v1 import CanonicalFamilyId
from hegel_machine.phase2b_protocol import ExecutionFreezeManifest
from hegel_machine.phase2b_trusted_wire_v1 import NON_AUTHORITATIVE_CLAIM_LEVEL
from hegel_machine.phase2b_wire import (
    PREDICTION_SCHEMA_VERSION,
    PredictionBundle,
    PredictionDisposition,
    PredictionReason,
    RoleBinding,
)


RUN_ID = b"R" * 32
OUTCOME_FIELDS = (
    "input_row_id",
    "input_payload_sha256",
    "decision",
    "canonical_family_id",
    "prediction",
    "bridge_outcome_id",
    "bridge_compilation_id",
    "bridge_decision_id",
    "claim_level",
)


def _namespace(filename: str) -> dict[str, object]:
    return runpy.run_path(str(Path(__file__).with_name(filename)))


def _copy_with_pollution(value: object, **changes: object) -> object:
    copied = object.__new__(type(value))
    for item in fields(value):
        object.__setattr__(
            copied,
            item.name,
            changes.get(item.name, getattr(value, item.name)),
        )
    return copied


def _typed_for_row(
    row: archive_v2.TrustedRecognizerInputRowV2,
) -> batch_v2.DecodedTrustedEnvelopeV2:
    typed = typed_replay_v2.decode_and_replay_typed_trusted_envelope_v2(row.envelope)
    assert type(typed) is batch_v2.DecodedTrustedEnvelopeV2
    return typed


def _keys() -> batch_v2.TrustedWireKeySourcesV2:
    return batch_v2.TrustedWireKeySourcesV2(b"S" * 32, b"I" * 32, b"P" * 32)


def _forbid_v1_public_runtime_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_v1(*args: object, **kwargs: object) -> object:
        raise AssertionError("compact V2 prediction must not call a V1 public API")

    api_names = (
        (typed_replay_v1, "decode_and_replay_typed_trusted_envelope_v1"),
        (typed_replay_v1, "replay_typed_trusted_wire_batch_v1"),
        (prediction_v1, "build_public_run_context_v1"),
        (prediction_v1, "recognize_public_input_row_v1"),
        (prediction_v1, "decode_public_recognizer_prediction_archive_v1"),
        (prediction_v1, "build_recognizer_prediction_archive_v1"),
    )
    original_apis = tuple(getattr(module, name) for module, name in api_names)
    assert len(original_apis) == len({id(item) for item in original_apis}) == 6
    prediction_module_values = tuple(vars(prediction_v2).values())
    recognize_global_values = tuple(
        prediction_v2.recognize_public_input_row_v2.__globals__.values()
    )
    for original in original_apis:
        assert all(original is not value for value in prediction_module_values)
        assert all(original is not value for value in recognize_global_values)
    for module, name in api_names:
        monkeypatch.setattr(module, name, forbidden_v1)


@pytest.fixture(scope="module")
def execution_freeze_manifest() -> ExecutionFreezeManifest:
    manifest = _namespace("test_phase2b_runner.py")["_freeze_manifest"]()
    current = replace(manifest, theory_version_id=initial_theory().version_id)
    assert type(current) is ExecutionFreezeManifest
    return current


@pytest.fixture(scope="module")
def small_v2_row() -> archive_v2.TrustedRecognizerInputRowV2:
    registry, authority = _namespace(
        "test_phase2b_recognizer_input_archive_v1.py"
    )["_six_family_source_fixture"]()
    source = archive_v2.TrustedRecognizerSourceCaseV2(
        authority=authority,
        adapter_registry=registry,
    )
    batch = batch_v2.build_trusted_wire_batch_v2(
        authorities=(authority,),
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(batch) is batch_v2.TrustedWireBatchV2
    decoded = archive_v2.issue_trusted_recognizer_input_archive_v2(
        batch=batch,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=(source,),
    )
    assert type(decoded) is archive_v2.DecodedRecognizerInputArchiveV2
    assert len(decoded.rows) == 1
    return decoded.rows[0]


@pytest.fixture(scope="module")
def small_v2_typed(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
) -> batch_v2.DecodedTrustedEnvelopeV2:
    return _typed_for_row(small_v2_row)


@pytest.fixture(scope="module")
def small_preflight(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    small_v2_typed: batch_v2.DecodedTrustedEnvelopeV2,
) -> derived_bridge.ExactDerivedBridgePreflightRejection:
    return derived_bridge.ExactDerivedBridgePreflightRejection(
        disposition=derived_bridge.ExactBridgeDisposition.ABSTAIN,
        reason=prediction_v2._ACCEPTED_PREFLIGHT_ABSTAIN_SOURCES[0],
        bundle_id=small_v2_typed.authority.base_bundle.bundle_id,
        wrapper_schema_version=small_v2_typed.authority.schema_version,
        theory_schema_version=prediction_v2._FROZEN_THEORY.schema_version,
        registry_theory_version_id=small_v2_row.public_registry.theory_version_id,
        bridge_policy_id=prediction_v2.EXACT_DERIVED_BRIDGE_POLICY_ID,
        matcher_semantics_id=prediction_v2.EXACT_DERIVED_MATCHER_SEMANTICS_ID,
    )


@pytest.fixture(scope="module")
def small_bridge_run(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    small_v2_typed: batch_v2.DecodedTrustedEnvelopeV2,
) -> derived_bridge.ExactDerivedBridgeRun:
    result = derived_bridge.run_exact_derived_witness_bridge(
        authority=small_v2_typed.authority,
        theory=prediction_v2._FROZEN_THEORY,
        registry=small_v2_row.public_registry.to_adapter_registry(),
    )
    assert type(result) is derived_bridge.ExactDerivedBridgeRun
    assert result.disposition is derived_bridge.ExactBridgeDisposition.ABSTAIN
    assert result.decision.disposition is derived_bridge.ExactSelectionDisposition.ABSTAIN
    return result


def test_public_exports_signature_fields_and_frozen_identity() -> None:
    assert prediction_v2.__all__ == (
        "PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_ID",
        "PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_VERSION",
        "RECOGNIZER_PREDICTION_ROW_POLICY_ID_V2",
        "PredictionDecisionV2",
        "PublicRecognizerPredictionOutcomeV2",
        "recognize_public_input_row_v2",
    )
    signature = inspect.signature(prediction_v2.recognize_public_input_row_v2)
    assert tuple(signature.parameters) == (
        "input_row",
        "execution_freeze_manifest",
    )
    assert all(
        item.kind is inspect.Parameter.KEYWORD_ONLY
        for item in signature.parameters.values()
    )
    assert tuple(
        item.name for item in fields(prediction_v2.PublicRecognizerPredictionOutcomeV2)
    ) == OUTCOME_FIELDS == prediction_v2._OUTCOME_FIELDS
    assert prediction_v2.PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_VERSION == (
        "hegel-machine-phase2b-public-recognizer-prediction-outcome/2"
    )
    assert prediction_v2.PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_ID == (
        "phase2b_public_recognizer_prediction_outcome_schema_v2_"
        "5e753645657066fc2dfe3e1f3577a30b47bf9ccc4f429828523de13364e8488c"
    )
    assert prediction_v2.RECOGNIZER_PREDICTION_ROW_POLICY_ID_V2 == (
        "phase2b_recognizer_prediction_row_policy_v2_"
        "11cc0cfa63c308efef20f31844b1354deb0d6de84102c315b5f61a089b2712d9"
    )


def test_policy_closes_dependencies_replay_reuse_and_non_evidence() -> None:
    policy = prediction_v2._PREDICTION_ROW_POLICY_VALUE_V2
    assert tuple(sorted(policy)) == (
        "abstention_reason_mapping",
        "abstention_reason_source_classification",
        "cross_version_rejection",
        "decision_values",
        "dependency_bindings",
        "ephemeral_call_mechanics",
        "non_evidence_claims",
        "outcome_contract",
        "outcome_fields",
        "positive_source_reasons",
        "preflight_outcome",
        "public_api",
        "schema_id",
        "schema_version",
        "validated_replay_reuse",
        "validation_order",
    )
    assert policy["schema_id"] == (
        prediction_v2.PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_ID
    )
    assert policy["outcome_fields"] == OUTCOME_FIELDS
    assert policy["outcome_contract"] == (
        "ephemeral_in_process",
        "privately_issued",
        "no_durable_receipt_id",
        "non_authoritative_mechanics_only",
    )
    assert policy["ephemeral_call_mechanics"] == (
        prediction_v2._EPHEMERAL_MECHANICS_TRUE
    ) == (
        "compact_typed_replay",
        "public_registry_adapter",
        "exact_derived_bridge",
        "closed_prediction_mapping",
    )
    reuse = policy["validated_replay_reuse"]
    assert reuse["public_typed_replay_api_call_count"] == 1
    assert reuse["typed_zero_argument_validation_after_decode"] is False
    assert reuse["input_row_zero_argument_validation_after_decode"] is False
    assert "seven_stored_public_roots" in reuse["local_parity_checks"]
    dependencies = policy["dependency_bindings"]
    assert dependencies == {
        "recognizer_input_archive_version": archive_v2.TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
        "recognizer_input_archive_policy_id": archive_v2.RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
        "public_registry_version": archive_v2.PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION,
        "public_registry_schema_id": archive_v2.PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2,
        "public_family_alias_policy_id": archive_v2.PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2,
        "typed_replay_version": typed_replay_v2.TYPED_TRUSTED_WIRE_REPLAY_V2_VERSION,
        "typed_replay_policy_id": typed_replay_v2.TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID,
        "compact_typed_authority_schema_id": batch_v2.COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
        "compact_typed_authority_codec_version": batch_v2.COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
        "compact_typed_authority_codec_policy_id": batch_v2.COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
        "trusted_wire_batch_schema_version": batch_v2.TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION,
        "trusted_wire_batch_policy_id": batch_v2.TRUSTED_WIRE_BATCH_V2_POLICY_ID,
        "prediction_bundle_schema_version": PREDICTION_SCHEMA_VERSION,
        "frozen_protocol_id": prediction_v2._FROZEN_PROTOCOL.protocol_id,
        "frozen_theory_version_id": prediction_v2._FROZEN_THEORY.version_id,
        "exact_freeze_id": prediction_v2._FROZEN_EXACT_FREEZE.freeze_id,
        "exact_freeze_family_mapping": prediction_v2._FROZEN_FAMILY_MAPPING_VALUES,
        "derived_bridge_policy_id": prediction_v2.EXACT_DERIVED_BRIDGE_POLICY_ID,
        "derived_matcher_semantics_id": prediction_v2.EXACT_DERIVED_MATCHER_SEMANTICS_ID,
        "derived_selection_policy_id": prediction_v2.EXACT_DERIVED_SELECTION_POLICY_ID,
    }
    assert policy["non_evidence_claims"] == prediction_v2._NON_EVIDENCE_CLAIMS == (
        "input_archive_membership",
        "batch_policy_membership",
        "execution_manifest_authority",
        "recognizer_executed",
        "runtime_executed",
        "recognizer_capacity",
        "prediction_scored",
        "effect",
        "origin_authenticated",
        "formal_uuid_audit",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "c1_exit",
    )
    assert policy["positive_source_reasons"] == {
        "complete_bridge_compilation": "complete_exact_derived_witness_candidate_grid",
        "selected_decision": "unique_structure_with_exact_derived_admissible_scales",
    }
    assert {
        "recognizer_capacity",
        "prediction_scored",
        "effect",
        "sealed_holdout_eligible",
        "c1_exit",
    }.issubset(policy["non_evidence_claims"])


def test_reason_conversion_is_exact_closed_and_fail_closed() -> None:
    accepted = set(prediction_v2._ABSTAIN_REASON_MAP)
    expected = set(prediction_v2._EXPECTED_RESOURCE_REASON_SOURCES) | set(
        prediction_v2._ROW_SEMANTIC_REASON_SOURCES
    )
    rejected = set(prediction_v2._INTEGRITY_OR_DEAD_REASON_SOURCES_REJECTED)
    assert accepted == expected
    assert accepted.isdisjoint(rejected)
    compilation_keys = set(
        prediction_v2._ACCEPTED_COMPILATION_ABSTAIN_REASON_KEYS
    )
    compilation_sources = set(
        prediction_v2._ACCEPTED_COMPILATION_ABSTAIN_SOURCES
    )
    complete_selection = set(
        prediction_v2._ACCEPTED_COMPLETE_SELECTION_ABSTAIN_SOURCES
    )
    preflight = set(prediction_v2._ACCEPTED_PREFLIGHT_ABSTAIN_SOURCES)
    assert (len(compilation_keys), len(compilation_sources), len(complete_selection), len(preflight)) == (
        11,
        11,
        11,
        4,
    )
    assert compilation_keys == {"bridge_" + item for item in compilation_sources}
    assert compilation_keys | complete_selection | preflight == accepted
    assert compilation_keys.isdisjoint(complete_selection)
    assert compilation_keys.isdisjoint(preflight)
    assert complete_selection.isdisjoint(preflight)
    assert set(prediction_v2._ABSTAIN_REASON_MAP.values()) == (
        prediction_v2._ALLOWED_ABSTAIN_REASONS
    )
    for source, target in prediction_v2._ABSTAIN_REASON_PAIRS:
        assert prediction_v2._mapped_abstention_reason(source) is target
    for source in (*sorted(rejected), "unknown_reason", None, 0):
        with pytest.raises(prediction_v2._PredictionGateRejected):
            prediction_v2._mapped_abstention_reason(source)


def test_outcome_is_private_ephemeral_and_has_no_receipt_field() -> None:
    with pytest.raises(TypeError, match="privately issued"):
        prediction_v2.PublicRecognizerPredictionOutcomeV2()
    with pytest.raises(TypeError, match="issuer token"):
        prediction_v2.PublicRecognizerPredictionOutcomeV2._issue(
            object(),
            input_row_id="phase2b_recognizer_input_row_v2_" + "0" * 64,
            input_payload_sha256="0" * 64,
            decision=prediction_v2.PredictionDecisionV2.ABSTAIN,
            canonical_family_id=None,
            prediction=object(),
            bridge_outcome_id="phase2b_exact_derived_preflight_v2_" + "0" * 64,
            bridge_compilation_id=None,
            bridge_decision_id=None,
        )
    assert all(
        "receipt" not in name and "archive" not in name and "capacity" not in name
        for name in OUTCOME_FIELDS
    )


def test_exact_top_level_types_and_v1_v2_rows_cross_reject(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> None:
    class RowSubclass(archive_v2.TrustedRecognizerInputRowV2):
        pass

    class ManifestSubclass(ExecutionFreezeManifest):
        pass

    with pytest.raises(TypeError, match="exact input row"):
        prediction_v2.recognize_public_input_row_v2(
            input_row=object.__new__(RowSubclass),
            execution_freeze_manifest=execution_freeze_manifest,
        )
    with pytest.raises(TypeError, match="exact freeze manifest"):
        prediction_v2.recognize_public_input_row_v2(
            input_row=small_v2_row,
            execution_freeze_manifest=object.__new__(ManifestSubclass),
        )
    v1_row_type = archive_v1.TrustedRecognizerInputRowV1
    with pytest.raises(TypeError, match="exact input row"):
        prediction_v2.recognize_public_input_row_v2(
            input_row=object.__new__(v1_row_type),
            execution_freeze_manifest=execution_freeze_manifest,
        )
    with pytest.raises(TypeError):
        prediction_v1.recognize_public_input_row_v1(
            input_row=small_v2_row,
            execution_freeze_manifest=execution_freeze_manifest,
        )
    assert type(prediction_v2.PredictionDecisionV2.ABSTAIN) is not type(
        prediction_v1.PredictionDecisionV1.ABSTAIN
    )


def test_bad_freeze_and_shallow_row_identity_fail_before_typed_decoder(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    execution_freeze_manifest: ExecutionFreezeManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_decoder(*args: object, **kwargs: object) -> object:
        raise AssertionError("bad shallow authority must fail before typed decoder")

    monkeypatch.setattr(
        prediction_v2,
        "decode_and_replay_typed_trusted_envelope_v2",
        forbidden_decoder,
    )
    bad_manifest = _copy_with_pollution(
        execution_freeze_manifest,
        theory_version_id="stale_theory",
    )
    with pytest.raises((TypeError, ValueError), match="manifest|authorit"):
        prediction_v2.recognize_public_input_row_v2(
            input_row=small_v2_row,
            execution_freeze_manifest=bad_manifest,
        )
    for polluted in (
        _copy_with_pollution(small_v2_row, row_id="bad"),
        _copy_with_pollution(small_v2_row, public_registry=object()),
        _copy_with_pollution(small_v2_row, payload_sha256="f" * 64),
    ):
        with pytest.raises((TypeError, ValueError)):
            prediction_v2.recognize_public_input_row_v2(
                input_row=polluted,
                execution_freeze_manifest=execution_freeze_manifest,
            )


@pytest.mark.parametrize(
    ("header_field", "bad_value"),
    (
        ("magic", b"BAD2BW2\x00"),
        ("version", batch_v2.TRUSTED_WIRE_ENVELOPE_V2_VERSION + 1),
        ("header_bytes", batch_v2.ENVELOPE_HEADER_BYTES + 1),
        ("payload_bytes", batch_v2.MAXIMUM_PAYLOAD_BYTES + 1),
    ),
)
def test_all_four_shallow_header_fields_fail_before_typed_decoder(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    execution_freeze_manifest: ExecutionFreezeManifest,
    monkeypatch: pytest.MonkeyPatch,
    header_field: str,
    bad_value: object,
) -> None:
    def forbidden_decoder(*args: object, **kwargs: object) -> object:
        raise AssertionError("bad shallow header must fail before typed decoder")

    monkeypatch.setattr(
        prediction_v2,
        "decode_and_replay_typed_trusted_envelope_v2",
        forbidden_decoder,
    )
    header_names = (
        "magic",
        "version",
        "header_bytes",
        "payload_bytes",
        "payload_hash",
        "padding_hash",
    )
    header = dict(
        zip(
            header_names,
            batch_v2._HEADER_V2.unpack(
                small_v2_row.envelope[: batch_v2.ENVELOPE_HEADER_BYTES]
            ),
            strict=True,
        )
    )
    header[header_field] = bad_value
    polluted_envelope = batch_v2._HEADER_V2.pack(
        *(header[name] for name in header_names)
    ) + small_v2_row.envelope[batch_v2.ENVELOPE_HEADER_BYTES :]
    polluted = _copy_with_pollution(small_v2_row, envelope=polluted_envelope)
    with pytest.raises(ValueError, match="shallow header"):
        prediction_v2.recognize_public_input_row_v2(
            input_row=polluted,
            execution_freeze_manifest=execution_freeze_manifest,
        )


def test_small_v2_mapping_never_calls_v1_typed_or_prediction_apis(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    execution_freeze_manifest: ExecutionFreezeManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_v1_public_runtime_apis(monkeypatch)
    result = prediction_v2.recognize_public_input_row_v2(
        input_row=small_v2_row,
        execution_freeze_manifest=execution_freeze_manifest,
    )
    assert type(result) is prediction_v2.PublicRecognizerPredictionOutcomeV2


def test_actual_small_row_is_abstention_with_one_public_typed_replay_call(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    execution_freeze_manifest: ExecutionFreezeManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    public_replay = prediction_v2.decode_and_replay_typed_trusted_envelope_v2

    def counted_public_replay(envelope: bytes) -> object:
        nonlocal calls
        calls += 1
        return public_replay(envelope)

    def forbidden_row_validate(*args: object, **kwargs: object) -> None:
        raise AssertionError("V2 prediction must reuse its one public typed replay")

    monkeypatch.setattr(
        prediction_v2,
        "decode_and_replay_typed_trusted_envelope_v2",
        counted_public_replay,
    )
    monkeypatch.setattr(
        archive_v2.TrustedRecognizerInputRowV2,
        "_validate",
        forbidden_row_validate,
    )
    result = prediction_v2.recognize_public_input_row_v2(
        input_row=small_v2_row,
        execution_freeze_manifest=execution_freeze_manifest,
    )
    assert calls == 1
    assert type(result) is prediction_v2.PublicRecognizerPredictionOutcomeV2
    assert result.input_row_id == small_v2_row.row_id
    assert result.input_payload_sha256 == small_v2_row.payload_sha256
    assert result.decision is prediction_v2.PredictionDecisionV2.ABSTAIN
    assert result.canonical_family_id is None
    assert type(result.prediction) is PredictionBundle
    assert result.prediction.disposition is PredictionDisposition.ABSTAIN
    assert type(result.prediction.reason) is PredictionReason
    assert result.prediction.family_id is None
    assert result.prediction.binding == ()
    assert result.prediction.admissible_scale_ids == ()
    assert result.prediction.input_root_sha256 == small_v2_row.payload_sha256
    assert result.prediction.protocol_sha256 == (
        prediction_v2._FROZEN_PROTOCOL.protocol_id.rsplit("_", 1)[1]
    )
    assert result.prediction.freeze_manifest_sha256 == (
        execution_freeze_manifest.manifest_id.rsplit("_", 1)[1]
    )
    assert result.claim_level == NON_AUTHORITATIVE_CLAIM_LEVEL


def test_outcome_and_prediction_objects_reject_exact_type_pollution(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> None:
    result = prediction_v2.recognize_public_input_row_v2(
        input_row=small_v2_row,
        execution_freeze_manifest=execution_freeze_manifest,
    )

    class StringSubclass(str):
        pass

    class DecisionSubclass(str):
        pass

    class BundleSubclass(PredictionBundle):
        pass

    class OutcomeSubclass(prediction_v2.PublicRecognizerPredictionOutcomeV2):
        pass

    subclass_outcome = object.__new__(OutcomeSubclass)
    for item in fields(result):
        object.__setattr__(subclass_outcome, item.name, getattr(result, item.name))
    with pytest.raises(TypeError, match="exact type"):
        subclass_outcome._validate()

    polluted_bundle = object.__new__(BundleSubclass)
    for item in fields(result.prediction):
        object.__setattr__(
            polluted_bundle,
            item.name,
            getattr(result.prediction, item.name),
        )
    for polluted in (
        _copy_with_pollution(result, claim_level=StringSubclass(result.claim_level)),
        _copy_with_pollution(result, decision=DecisionSubclass(result.decision.value)),
        _copy_with_pollution(result, prediction=polluted_bundle),
        _copy_with_pollution(result, input_payload_sha256=StringSubclass("0" * 64)),
    ):
        with pytest.raises((TypeError, ValueError)):
            polluted._validate()


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("payload_schema_version", "wrong_payload_schema"),
        ("public_provenance_version", "wrong_provenance"),
        ("typed_authority_schema_id", "wrong_typed_schema"),
        ("typed_authority_codec_version", "wrong_codec_version"),
        ("typed_authority_codec_policy_id", "wrong_codec_policy"),
    ),
)
def test_local_replay_parity_closes_five_wire_discriminators(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    small_v2_typed: batch_v2.DecodedTrustedEnvelopeV2,
    field_name: str,
    bad_value: str,
) -> None:
    polluted = _copy_with_pollution(
        small_v2_typed,
        **{field_name: bad_value},
    )
    with pytest.raises(ValueError, match="discriminator"):
        prediction_v2._validate_input_row_against_typed_v2(small_v2_row, polluted)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("payload_bytes", 0),
        ("padding_bytes", 0),
        ("structural_hashes_verified", False),
        ("secret_padding_replay_verified", True),
    ),
)
def test_local_replay_parity_closes_header_partition_and_claim_manifest(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    small_v2_typed: batch_v2.DecodedTrustedEnvelopeV2,
    field_name: str,
    bad_value: object,
) -> None:
    polluted = _copy_with_pollution(
        small_v2_typed,
        **{field_name: bad_value},
    )
    with pytest.raises((TypeError, ValueError), match="length|claim"):
        prediction_v2._validate_input_row_against_typed_v2(small_v2_row, polluted)


@pytest.mark.parametrize(
    "field_name",
    (
        "envelope_id",
        "payload_sha256",
        "padding_sha256",
        "namespace_audit_id",
        "authority_content_id",
        "transform_result_id",
    ),
)
def test_local_replay_parity_closes_typed_roots(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    small_v2_typed: batch_v2.DecodedTrustedEnvelopeV2,
    field_name: str,
) -> None:
    original = getattr(small_v2_typed, field_name)
    if field_name in {"payload_sha256", "padding_sha256"}:
        bad = ("0" if original[0] != "0" else "1") + original[1:]
    else:
        prefix, digest = original.rsplit("_", 1)
        bad = prefix + "_" + ("0" if digest[0] != "0" else "1") + digest[1:]
    polluted = _copy_with_pollution(small_v2_typed, **{field_name: bad})
    with pytest.raises(ValueError, match="root|parity"):
        prediction_v2._validate_input_row_against_typed_v2(small_v2_row, polluted)


def test_local_replay_parity_closes_registry_as_the_seventh_root(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    small_v2_typed: batch_v2.DecodedTrustedEnvelopeV2,
) -> None:
    polluted = _copy_with_pollution(
        small_v2_row,
        public_registry_id="phase2b_public_recognizer_registry_v2_" + "0" * 64,
    )
    with pytest.raises(ValueError, match="root"):
        prediction_v2._validate_input_row_against_typed_v2(polluted, small_v2_typed)


@pytest.mark.parametrize(
    ("claim_name", "bad_value"),
    tuple((name, False) for name in batch_v2._ENVELOPE_TRUE_CLAIMS_V2)
    + tuple((name, True) for name in batch_v2._ENVELOPE_FALSE_CLAIMS_V2)
    + (
        (batch_v2._ENVELOPE_TRUE_CLAIMS_V2[0], 1),
        (batch_v2._ENVELOPE_FALSE_CLAIMS_V2[0], 0),
    ),
)
def test_local_replay_parity_closes_each_four_true_six_false_claim(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    small_v2_typed: batch_v2.DecodedTrustedEnvelopeV2,
    claim_name: str,
    bad_value: object,
) -> None:
    polluted = _copy_with_pollution(
        small_v2_typed,
        **{claim_name: bad_value},
    )
    with pytest.raises((TypeError, ValueError), match="claim"):
        prediction_v2._validate_input_row_against_typed_v2(small_v2_row, polluted)


def test_local_replay_parity_rejects_bool_subclass_claims(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    small_v2_typed: batch_v2.DecodedTrustedEnvelopeV2,
) -> None:
    # bool is deliberately non-subclassable in CPython.  An int subclass is
    # the closest truthy scalar pollution and must still fail exact-bool gates.
    class BoolLike(int):
        pass

    polluted = _copy_with_pollution(
        small_v2_typed,
        structural_hashes_verified=BoolLike(1),
    )
    with pytest.raises(TypeError, match="exact bool"):
        prediction_v2._validate_input_row_against_typed_v2(small_v2_row, polluted)


@pytest.mark.parametrize(
    "source_reason",
    prediction_v2._ACCEPTED_PREFLIGHT_ABSTAIN_SOURCES,
)
def test_all_four_preflight_stage_reasons_are_accepted(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    small_v2_typed: batch_v2.DecodedTrustedEnvelopeV2,
    small_preflight: derived_bridge.ExactDerivedBridgePreflightRejection,
    execution_freeze_manifest: ExecutionFreezeManifest,
    source_reason: str,
) -> None:
    result = prediction_v2._compile_prediction_outcome_from_bridge(
        input_row=small_v2_row,
        typed=small_v2_typed,
        execution_freeze_manifest_id=execution_freeze_manifest.manifest_id,
        bridge_result=replace(small_preflight, reason=source_reason),
    )
    assert result.decision is prediction_v2.PredictionDecisionV2.ABSTAIN
    assert result.prediction.reason is prediction_v2._ABSTAIN_REASON_MAP[source_reason]


def test_committed_abstention_requires_bridge_prefixed_compilation_reason(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    small_v2_typed: batch_v2.DecodedTrustedEnvelopeV2,
    small_bridge_run: derived_bridge.ExactDerivedBridgeRun,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> None:
    assert small_bridge_run.decision.reason == "bridge_" + small_bridge_run.compilation.reason
    assert small_bridge_run.compilation.reason in (
        prediction_v2._ACCEPTED_COMPILATION_ABSTAIN_SOURCES
    )
    forged_decision = replace(
        small_bridge_run.decision,
        reason=prediction_v2._ACCEPTED_COMPLETE_SELECTION_ABSTAIN_SOURCES[0],
    )
    forged_run = replace(small_bridge_run, decision=forged_decision)
    with pytest.raises(
        prediction_v2._PredictionGateRejected,
        match="^derived_compilation_abstention_reason_stage_drift$",
    ):
        prediction_v2._compile_prediction_outcome_from_bridge(
            input_row=small_v2_row,
            typed=small_v2_typed,
            execution_freeze_manifest_id=execution_freeze_manifest.manifest_id,
            bridge_result=forged_run,
        )

    wrong_stage_reason = (
        prediction_v2._ACCEPTED_COMPLETE_SELECTION_ABSTAIN_SOURCES[0]
    )
    forged_compilation = replace(
        small_bridge_run.compilation,
        reason=wrong_stage_reason,
    )
    forged_decision = replace(
        small_bridge_run.decision,
        reason=wrong_stage_reason,
        bridge_result_id=forged_compilation.result_id,
    )
    coherent_wrong_stage_run = replace(
        small_bridge_run,
        reason=wrong_stage_reason,
        compilation=forged_compilation,
        decision=forged_decision,
    )
    coherent_wrong_stage_run.__post_init__()
    with pytest.raises(
        prediction_v2._PredictionGateRejected,
        match="^derived_compilation_abstention_reason_stage_drift$",
    ):
        prediction_v2._compile_prediction_outcome_from_bridge(
            input_row=small_v2_row,
            typed=small_v2_typed,
            execution_freeze_manifest_id=execution_freeze_manifest.manifest_id,
            bridge_result=coherent_wrong_stage_run,
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "bundle_id",
        "wrapper_schema_version",
        "theory_schema_version",
        "registry_theory_version_id",
        "bridge_policy_id",
        "matcher_semantics_id",
    ),
)
def test_preflight_mapping_binds_all_input_and_policy_authorities(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    small_v2_typed: batch_v2.DecodedTrustedEnvelopeV2,
    small_preflight: derived_bridge.ExactDerivedBridgePreflightRejection,
    execution_freeze_manifest: ExecutionFreezeManifest,
    field_name: str,
) -> None:
    polluted = replace(small_preflight, **{field_name: "wrong_authority"})
    with pytest.raises(
        prediction_v2._PredictionGateRejected,
        match="input_or_policy_binding",
    ):
        prediction_v2._compile_prediction_outcome_from_bridge(
            input_row=small_v2_row,
            typed=small_v2_typed,
            execution_freeze_manifest_id=execution_freeze_manifest.manifest_id,
            bridge_result=polluted,
        )


def test_local_replay_validates_exact_logical_authority_before_content_hash(
    small_v2_row: archive_v2.TrustedRecognizerInputRowV2,
    small_v2_typed: batch_v2.DecodedTrustedEnvelopeV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polluted_authority = _copy_with_pollution(
        small_v2_typed.authority,
        schema_version="bad_schema_version",
    )
    assert type(polluted_authority) is exact_transform_semantics.PublicTransformEvidenceBundleV2
    polluted = _copy_with_pollution(
        small_v2_typed,
        authority=polluted_authority,
    )

    def forbidden_hash(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid logical authority must reject before hashing")

    monkeypatch.setattr(exact_transform_semantics, "stable_hash", forbidden_hash)
    with pytest.raises(ValueError, match="schema version"):
        prediction_v2._validate_input_row_against_typed_v2(small_v2_row, polluted)


def test_real_positive_125582_to_50255_maps_to_exact_public_prediction(
    execution_freeze_manifest: ExecutionFreezeManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _namespace("test_phase2b_recognizer_prediction_archive_v1.py")
    source_fixture = namespace["public_positive_mechanics_fixture"].__wrapped__()
    theory, source_registry, source_authority = namespace[
        "_minimum_positive_derived_authority"
    ]()
    assert source_authority == source_fixture.source_authority
    expanded_profile = typed_authority_v1.encode_typed_transform_authority_profile_v1(
        source_authority
    )
    expanded_bytes = json.dumps(
        expanded_profile,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    assert len(expanded_bytes) == 125_582

    batch = batch_v2.build_trusted_wire_batch_v2(
        authorities=(source_authority,),
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(batch) is batch_v2.TrustedWireBatchV2
    assert len(batch.envelopes) == 1
    assert batch.envelopes[0].payload_bytes == 50_255
    assert batch.envelopes[0].padding_bytes == 15_201
    source_case = archive_v2.TrustedRecognizerSourceCaseV2(
        authority=source_authority,
        adapter_registry=source_registry,
    )
    input_archive = archive_v2.issue_trusted_recognizer_input_archive_v2(
        batch=batch,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=(source_case,),
    )
    assert type(input_archive) is archive_v2.DecodedRecognizerInputArchiveV2
    assert len(input_archive.rows) == 1
    row = input_archive.rows[0]
    assert row.envelope == batch.envelopes[0].envelope
    assert (
        row.envelope_id,
        row.payload_sha256,
        row.padding_sha256,
        row.namespace_audit_id,
        row.authority_content_id,
        row.transform_result_id,
        row.public_registry_id,
    ) == (
        batch.envelopes[0].envelope_id,
        batch.envelopes[0].payload_sha256,
        batch.envelopes[0].padding_sha256,
        batch.envelopes[0].namespace_audit_id,
        batch.envelopes[0].authority_content_id,
        batch.envelopes[0].transform_result_id,
        row.public_registry.registry_id,
    )

    typed = typed_replay_v2.decode_and_replay_typed_trusted_envelope_v2(row.envelope)
    assert type(typed) is batch_v2.DecodedTrustedEnvelopeV2
    assert (
        typed.envelope_id,
        typed.payload_sha256,
        typed.padding_sha256,
        typed.namespace_audit_id,
        typed.authority_content_id,
        typed.transform_result_id,
    ) == (
        row.envelope_id,
        row.payload_sha256,
        row.padding_sha256,
        row.namespace_audit_id,
        row.authority_content_id,
        row.transform_result_id,
    )
    assert typed.authority.content_id == typed.authority_content_id
    bridge = derived_bridge.run_exact_derived_witness_bridge(
        authority=typed.authority,
        theory=theory,
        registry=row.public_registry.to_adapter_registry(),
    )
    assert type(bridge) is derived_bridge.ExactDerivedBridgeRun
    assert bridge.compilation == source_fixture.bridge_run.compilation
    assert bridge.decision == source_fixture.bridge_run.decision
    assert bridge.disposition is derived_bridge.ExactBridgeDisposition.COMPLETE
    assert bridge.reason == "complete_exact_derived_witness_candidate_grid"
    assert bridge.compilation.reason == "complete_exact_derived_witness_candidate_grid"
    assert (
        bridge.decision.reason
        == "unique_structure_with_exact_derived_admissible_scales"
    )
    assert bridge.wrapper_content_id == typed.authority_content_id
    assert bridge.transform_result_id == typed.transform_result_id

    selected_laws = tuple(
        item
        for item in row.public_registry.law_bindings
        if item.law_kind is bridge.decision.selected_law_kind
    )
    assert len(selected_laws) == 1
    selected_law = selected_laws[0]
    semantic_to_wire = dict(selected_law.role_ids)
    semantic_to_entity = dict(bridge.decision.selected_role_binding)
    assert set(semantic_to_wire) == set(semantic_to_entity)
    expected_binding = tuple(
        sorted(
            (
                RoleBinding(semantic_to_wire[semantic_role], entity_id)
                for semantic_role, entity_id in semantic_to_entity.items()
            ),
            key=lambda item: item.role_id,
        )
    )

    _forbid_v1_public_runtime_apis(monkeypatch)
    outcome = prediction_v2.recognize_public_input_row_v2(
        input_row=row,
        execution_freeze_manifest=execution_freeze_manifest,
    )
    assert type(outcome) is prediction_v2.PublicRecognizerPredictionOutcomeV2
    expected_decision = (
        prediction_v2.PredictionDecisionV2.ANSWER
        if len(bridge.decision.admissible_scale_ids) == 1
        else prediction_v2.PredictionDecisionV2.ANSWER_SET
    )
    assert outcome.decision is expected_decision
    assert type(outcome.canonical_family_id) is CanonicalFamilyId
    assert outcome.canonical_family_id is selected_law.canonical_family_id
    assert outcome.input_row_id == row.row_id
    assert outcome.input_payload_sha256 == row.payload_sha256
    assert type(outcome.prediction) is PredictionBundle
    assert outcome.prediction.bundle_id == typed.authority.base_bundle.bundle_id
    assert outcome.prediction.input_root_sha256 == row.payload_sha256
    assert outcome.prediction.protocol_sha256 == (
        prediction_v2._FROZEN_PROTOCOL.protocol_id.rsplit("_", 1)[1]
    )
    assert outcome.prediction.freeze_manifest_sha256 == (
        execution_freeze_manifest.manifest_id.rsplit("_", 1)[1]
    )
    assert outcome.prediction.disposition is PredictionDisposition.UNIQUE_MATCH
    assert outcome.prediction.reason is PredictionReason.UNIQUE_STRUCTURAL_MATCH
    assert outcome.prediction.family_id == selected_law.bridge_family_id
    assert outcome.prediction.binding == expected_binding
    assert all(type(item) is RoleBinding for item in outcome.prediction.binding)
    assert outcome.prediction.admissible_scale_ids == (
        bridge.decision.admissible_scale_ids
    )
    assert outcome.bridge_outcome_id == bridge.run_id
    assert outcome.bridge_compilation_id == bridge.compilation.result_id
    assert outcome.bridge_decision_id == bridge.decision.decision_id
    assert outcome.claim_level == NON_AUTHORITATIVE_CLAIM_LEVEL

    forged_decision = derived_bridge._abstaining_decision(
        prediction_v2._ACCEPTED_COMPILATION_ABSTAIN_REASON_KEYS[0],
        bridge.compilation,
        compilation_id=bridge.compilation.result_id,
        candidate_ids=tuple(
            sorted(item.candidate_id for item in bridge.compilation.evaluations)
        ),
        aggregate_ids=tuple(
            sorted(
                item.scale_aggregate_id
                for item in bridge.compilation.scale_aggregates
            )
        ),
    )
    forged_complete_run = replace(bridge, decision=forged_decision)
    with pytest.raises(
        prediction_v2._PredictionGateRejected,
        match="selection_abstention_reason_stage",
    ):
        prediction_v2._compile_prediction_outcome_from_bridge(
            input_row=row,
            typed=typed,
            execution_freeze_manifest_id=execution_freeze_manifest.manifest_id,
            bridge_result=forged_complete_run,
        )
