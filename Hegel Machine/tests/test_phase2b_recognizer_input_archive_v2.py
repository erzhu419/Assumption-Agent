"""Adversarial contracts for the V2 public recognizer-input archive.

The archive is a non-authoritative public mechanics artifact.  It must not
persist or authenticate source authorities, source roots, run/key material,
secret receipts, allocation state, origin, formal/covert audit, sealed
holdout eligibility, recognizer capacity/effect, or C1 exit evidence.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
import inspect
import json
from pathlib import Path
import runpy
from types import MappingProxyType
from typing import Iterator

import pytest

from hegel_machine.phase2b_adapter import Phase2BAdapterRegistry
import hegel_machine.phase2b_exact_derived_witness_bridge_v1 as derived_bridge
import hegel_machine.phase2b_exact_transform_semantics_v1 as transform
import hegel_machine.phase2b_recognizer_input_archive_v1 as archive_v1
import hegel_machine.phase2b_trusted_wire_batch_v1 as batch_v1
import hegel_machine.phase2b_trusted_wire_batch_v2 as batch_v2
import hegel_machine.phase2b_trusted_wire_typed_authority_v1 as typed_authority_v1
import hegel_machine.phase2b_trusted_wire_typed_replay_v2 as typed_replay_v2
from hegel_machine.phase2b_trusted_wire_v1 import MAXIMUM_UUID_OCCURRENCES

try:
    import hegel_machine.phase2b_recognizer_input_archive_v2 as archive_v2
except ImportError:  # Production lands after this public-contract scaffold.
    archive_v2 = None  # type: ignore[assignment]


RUN_ID = b"R" * 32
PUBLIC_TYPE_NAMES = (
    "TrustedRecognizerSourceCaseV2",
    "PublicRecognizerLawBindingV2",
    "PublicRecognizerObservableChannelV2",
    "PublicRecognizerRegistryV2",
    "TrustedRecognizerInputRowV2",
    "TrustedRecognizerInputArchiveRejectionV2",
    "DecodedRecognizerInputArchiveV2",
    "RecognizerInputArchiveDispositionV2",
)
METADATA_FIELDS = (
    "archive_policy_id",
    "archive_version",
    "batch_id",
    "batch_policy_id",
    "row_count",
    "typed_replay_policy_id",
    "typed_authority_schema_id",
    "typed_authority_codec_version",
    "typed_authority_codec_policy_id",
    "public_registry_schema_id",
)
FORBIDDEN_DURABLE_NAMES = (
    "run_id",
    "run_id_commitment",
    "execution_commitment",
    "collision_retry",
    "collision_retry_count",
    "typed_replay",
    "typed_replay_receipt",
    "typed_replay_receipt_id",
    "secret_replay_receipt",
    "secret_replay_receipt_id",
    "source_authorities",
    "source_authority_content_ids",
    "source_registry",
    "source_registry_id",
    "source_roots",
    "key_sources",
    "shuffle_ikm",
    "id_ikm",
    "padding_ikm",
    "renamings",
    "allocation_state",
    "source_index",
    "output_source_indices",
)


def _keys() -> batch_v2.TrustedWireKeySourcesV2:
    return batch_v2.TrustedWireKeySourcesV2(b"S" * 32, b"I" * 32, b"P" * 32)


def _uid(index: int) -> str:
    return f"10000000-0000-4000-8000-{index:012x}"


def _walk_public(value: object) -> Iterator[object]:
    yield value
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            yield from _walk_public(getattr(value, item.name))
    elif type(value) is dict:
        for key, item in value.items():
            yield from _walk_public(key)
            yield from _walk_public(item)
    elif type(value) in (tuple, list):
        for item in value:
            yield from _walk_public(item)


def _walk_public_field_names(value: object) -> Iterator[str]:
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            yield item.name
            yield from _walk_public_field_names(getattr(value, item.name))
    elif type(value) is dict:
        for key, item in value.items():
            if type(key) is str:
                yield key
            yield from _walk_public_field_names(item)
    elif type(value) in (tuple, list):
        for item in value:
            yield from _walk_public_field_names(item)


def _unchecked_copy(value: object, **changes: object) -> object:
    copied = object.__new__(type(value))
    for item in fields(value):
        object.__setattr__(
            copied,
            item.name,
            changes.get(item.name, getattr(value, item.name)),
        )
    return copied


@pytest.fixture(scope="module")
def six_family_source() -> tuple[
    Phase2BAdapterRegistry,
    transform.PublicTransformEvidenceBundleV2,
]:
    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_recognizer_input_archive_v1.py"))
    )
    registry, authority = namespace["_six_family_source_fixture"]()
    assert type(registry) is Phase2BAdapterRegistry
    assert type(authority) is transform.PublicTransformEvidenceBundleV2
    return registry, authority


@pytest.fixture(scope="module")
def positive_fixture() -> object:
    namespace = runpy.run_path(
        str(
            Path(__file__).with_name(
                "test_phase2b_recognizer_prediction_archive_v1.py"
            )
        )
    )
    return namespace["public_positive_mechanics_fixture"].__wrapped__()


@pytest.fixture(scope="module")
def positive_source() -> tuple[object, Phase2BAdapterRegistry, object]:
    namespace = runpy.run_path(
        str(
            Path(__file__).with_name(
                "test_phase2b_recognizer_prediction_archive_v1.py"
            )
        )
    )
    theory, registry, authority = namespace["_minimum_positive_derived_authority"]()
    assert type(registry) is Phase2BAdapterRegistry
    assert type(authority) is transform.PublicTransformEvidenceBundleV2
    return theory, registry, authority


@pytest.fixture(scope="module")
def positive_batch(positive_fixture: object) -> batch_v2.TrustedWireBatchV2:
    result = batch_v2.build_trusted_wire_batch_v2(
        authorities=(positive_fixture.source_authority,),
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(result) is batch_v2.TrustedWireBatchV2
    return result


@pytest.fixture(scope="module")
def one_case_source_case(
    six_family_source: tuple[
        Phase2BAdapterRegistry,
        transform.PublicTransformEvidenceBundleV2,
    ],
) -> object:
    assert archive_v2 is not None
    registry, authority = six_family_source
    return archive_v2.TrustedRecognizerSourceCaseV2(
        authority=authority,
        adapter_registry=registry,
    )


@pytest.fixture(scope="module")
def one_case_batch(one_case_source_case: object) -> batch_v2.TrustedWireBatchV2:
    result = batch_v2.build_trusted_wire_batch_v2(
        authorities=(one_case_source_case.authority,),
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(result) is batch_v2.TrustedWireBatchV2
    return result


@pytest.fixture(scope="module")
def one_case_archive(one_case_batch: batch_v2.TrustedWireBatchV2, one_case_source_case: object) -> object:
    assert archive_v2 is not None
    result = archive_v2.issue_trusted_recognizer_input_archive_v2(
        batch=one_case_batch,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=(one_case_source_case,),
    )
    assert type(result) is archive_v2.DecodedRecognizerInputArchiveV2
    return result


@pytest.fixture(scope="module")
def two_case_material(
    six_family_source: tuple[
        Phase2BAdapterRegistry,
        transform.PublicTransformEvidenceBundleV2,
    ],
) -> tuple[batch_v2.TrustedWireBatchV2, tuple[object, object]]:
    assert archive_v2 is not None
    registry, first = six_family_source
    first_observation = first.base_bundle.observations[0]
    second = transform.compile_exact_transform_provenance_v1(
        replace(
            first,
            base_bundle=replace(
                first.base_bundle,
                bundle_id=_uid(90_000),
                observations=(
                    replace(
                        first_observation,
                        value=replace(
                            first_observation.value,
                            values=(3.0, -4.0),
                        ),
                    ),
                ),
            ),
        )
    )
    second_compilation = transform.run_exact_transform_semantics(second)
    assert type(second_compilation) is transform.ExactTransformCompilation
    assert (
        second_compilation.disposition
        is transform.TransformCompilationDisposition.COMPLETE
    )
    cases = (
        archive_v2.TrustedRecognizerSourceCaseV2(first, registry),
        archive_v2.TrustedRecognizerSourceCaseV2(second, registry),
    )
    batch = batch_v2.build_trusted_wire_batch_v2(
        authorities=(first, second),
        run_id=RUN_ID,
        key_sources=_keys(),
    )
    assert type(batch) is batch_v2.TrustedWireBatchV2
    return batch, cases


def test_v2_archive_public_api_has_no_receipt_or_source_root_inputs() -> None:
    assert archive_v2 is not None
    issue = inspect.signature(archive_v2.issue_trusted_recognizer_input_archive_v2)
    assert tuple(issue.parameters) == (
        "batch",
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
            archive_v2.decode_public_recognizer_input_archive_v2
        ).parameters
    ) == ("archive",)
    for forbidden in (
        "typed_replay",
        "typed_replay_receipt",
        "typed_replay_receipt_id",
        "secret_replay_receipt",
        "secret_replay_receipt_id",
        "source_authorities",
        "source_roots",
        "source_registry_id",
        "policy",
    ):
        assert forbidden not in issue.parameters

    assert archive_v2.__all__ == (
        "ARCHIVE_HEADER_BYTES_V2",
        "ARCHIVE_MAGIC_V2",
        "ARCHIVE_WIRE_VERSION_V2",
        "FROZEN_BRIDGE_FAMILY_UUID_ALIASES_V2",
        "MAXIMUM_ARCHIVE_METADATA_BYTES_V2",
        "MAXIMUM_GLOBAL_SOURCE_UUIDS_V2",
        "MAXIMUM_PUBLIC_REGISTRY_BYTES_V2",
        "MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2",
        "PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2",
        "PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2",
        "PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION",
        "RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2",
        "TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION",
        "DecodedRecognizerInputArchiveV2",
        "PublicRecognizerLawBindingV2",
        "PublicRecognizerObservableChannelV2",
        "PublicRecognizerRegistryV2",
        "RecognizerInputArchiveDispositionV2",
        "TrustedRecognizerInputArchiveRejectionV2",
        "TrustedRecognizerInputRowV2",
        "TrustedRecognizerSourceCaseV2",
        "decode_public_recognizer_input_archive_v2",
        "issue_trusted_recognizer_input_archive_v2",
    )
    assert "TrustedRecognizerInputArchiveV2" not in archive_v2.__all__


def test_v2_archive_metadata_schema_is_exact_and_secret_free() -> None:
    assert archive_v2 is not None
    assert tuple(archive_v2._METADATA_FIELDS_V2) == METADATA_FIELDS
    policy = archive_v2._ARCHIVE_POLICY_VALUE_V2
    assert tuple(policy["field_manifests"]["metadata"]) == METADATA_FIELDS
    assert policy["public_exclusions"] == (
        "run_commitment",
        "collision_count",
        "typed_receipt",
        "secret_receipt",
        "source_registry",
        "source_roots",
        "source_or_secret_commitments",
    )


def test_v2_archive_wire_public_fields_and_claim_manifests_are_exact() -> None:
    assert archive_v2 is not None
    assert archive_v2.ARCHIVE_MAGIC_V2 == b"HGRIAV2\x00"
    assert archive_v2.ARCHIVE_WIRE_VERSION_V2 == 2
    assert archive_v2.ARCHIVE_HEADER_BYTES_V2 == 52
    assert archive_v2.MAXIMUM_GLOBAL_SOURCE_UUIDS_V2 == 2_097_152
    assert archive_v2.MAXIMUM_GLOBAL_SOURCE_UUIDS_V2 == (
        batch_v2.MAXIMUM_BATCH_V2_AUTHORITIES * MAXIMUM_UUID_OCCURRENCES
    )
    assert (
        archive_v2._ARCHIVE_POLICY_VALUE_V2["caps"]["global_uuid_sidecar_formula"]
        == "MAXIMUM_BATCH_V2_AUTHORITIES*MAXIMUM_UUID_OCCURRENCES_per_source_or_public_sidecar"
    )
    assert tuple(item.name for item in fields(archive_v2.TrustedRecognizerSourceCaseV2)) == (
        "authority",
        "adapter_registry",
    )
    assert tuple(item.name for item in fields(archive_v2.PublicRecognizerLawBindingV2)) == (
        "law_id",
        "law_kind",
        "canonical_family_id",
        "bridge_family_id",
        "role_ids",
        "required_observable_ids",
    )
    assert tuple(
        item.name for item in fields(archive_v2.PublicRecognizerObservableChannelV2)
    ) == ("quantity_id", "observable_id")
    assert tuple(item.name for item in fields(archive_v2.PublicRecognizerRegistryV2)) == (
        "schema_version",
        "theory_version_id",
        "law_bindings",
        "observable_channels",
        "maximum_candidate_count",
        "family_alias_policy_id",
    )
    assert tuple(item.name for item in fields(archive_v2.TrustedRecognizerInputRowV2)) == (
        "envelope",
        "envelope_id",
        "payload_sha256",
        "padding_sha256",
        "namespace_audit_id",
        "authority_content_id",
        "transform_result_id",
        "public_registry",
        "public_registry_id",
        "row_id",
    )
    assert tuple(
        item.name for item in fields(archive_v2.TrustedRecognizerInputArchiveRejectionV2)
    ) == (
        "disposition",
        "reason",
        "case_count",
        "batch_id",
        "archive",
        "rows",
        "row_ids",
        "envelope_ids",
        "public_registry_ids",
        "authority_content_ids",
        "transform_result_ids",
    )
    assert tuple(item.name for item in fields(archive_v2.DecodedRecognizerInputArchiveV2)) == (
        "disposition",
        "archive",
        "archive_id",
        "archive_version",
        "policy_id",
        "batch_id",
        "batch_policy_id",
        "typed_replay_policy_id",
        "typed_authority_schema_id",
        "typed_authority_codec_version",
        "typed_authority_codec_policy_id",
        "public_registry_schema_id",
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
        "registry_authority_exact_scope_verified",
        "compact_typed_replay_verified",
        "direct_payload_transform_replay_verified",
        "cross_row_public_uuid_disjoint_verified",
        "batch_policy_membership_verified",
        "source_registry_projection_verified",
        "source_public_disjoint_verified",
        "single_live_allocation_verified",
        "secret_custodian_replay_verified",
        "origin_authenticated",
        "formal_uuid_audit",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "recognizer_executed",
        "prediction_archive_evaluated",
        "recognizer_capacity_evidence",
        "c1_exit_evidence",
    )
    assert archive_v2._TRUE_CLAIMS_V2 == (
        "structural_archive_verified",
        "row_bijection_verified",
        "registry_schema_verified",
        "registry_authority_exact_scope_verified",
        "compact_typed_replay_verified",
        "direct_payload_transform_replay_verified",
        "cross_row_public_uuid_disjoint_verified",
    )
    assert archive_v2._FALSE_CLAIMS_V2 == (
        "batch_policy_membership_verified",
        "source_registry_projection_verified",
        "source_public_disjoint_verified",
        "single_live_allocation_verified",
        "secret_custodian_replay_verified",
        "origin_authenticated",
        "formal_uuid_audit",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "recognizer_executed",
        "prediction_archive_evaluated",
        "recognizer_capacity_evidence",
        "c1_exit_evidence",
    )


def test_v1_six_family_fixture_remains_a_complete_v2_source_fixture(
    six_family_source: tuple[
        Phase2BAdapterRegistry,
        transform.PublicTransformEvidenceBundleV2,
    ],
) -> None:
    registry, authority = six_family_source
    assert len(registry.law_bindings) == 6
    assert sum(len(item.role_ids) for item in registry.law_bindings) == 15
    assert len(registry.observable_channels) == 35
    compilation = transform.run_exact_transform_semantics(authority)
    assert type(compilation) is transform.ExactTransformCompilation
    assert compilation.disposition is transform.TransformCompilationDisposition.COMPLETE


def test_archive_metadata_constants_match_the_v2_batch_and_typed_replay() -> None:
    assert archive_v2 is not None
    policy = archive_v2._ARCHIVE_POLICY_VALUE_V2
    assert policy["batch_policy_id"] == batch_v2.TRUSTED_WIRE_BATCH_V2_POLICY_ID
    assert (
        policy["typed_replay_policy_id"]
        == typed_replay_v2.TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID
    )
    assert (
        policy["typed_authority"]["schema_id"]
        == batch_v2.COMPACT_TYPED_AUTHORITY_SCHEMA_ID
    )
    assert (
        policy["typed_authority"]["codec_version"]
        == batch_v2.COMPACT_TYPED_AUTHORITY_CODEC_VERSION
    )
    assert (
        policy["typed_authority"]["codec_policy_id"]
        == batch_v2.COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID
    )
    assert archive_v2.RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2 == (
        "phase2b_recognizer_input_archive_policy_v2_"
        "529a91fdf2e8b5d545dd94002eabb4199685ead0577e3d9f803d24963324fc12"
    )
    assert archive_v2.PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2 == (
        "phase2b_public_recognizer_registry_schema_v2_"
        "249ffa6544a3af90a689a29fb40f333365f95fada38581478a0673aa077f72b1"
    )
    assert archive_v2._ARCHIVE_POLICY_VALUE_V2[
        "registry_authority_exact_scope"
    ] == (
        "registry_roles_equals_authority_roles",
        "registry_quantities_equals_authority_quantities_equals_task_target_quantities",
        "task_target_entities_equals_entity_candidate_entities",
        "every_entity_has_nonempty_role_candidates_subset_of_registry_roles",
    )


def test_one_case_success_is_exactly_the_same_safe_public_decode(
    one_case_archive: object,
) -> None:
    assert archive_v2 is not None
    decoded = archive_v2.decode_public_recognizer_input_archive_v2(
        one_case_archive.archive
    )
    assert decoded == one_case_archive
    assert decoded.disposition is archive_v2.RecognizerInputArchiveDispositionV2.COMPLETE
    assert len(decoded.rows) == len(decoded.row_ids) == 1
    assert decoded.batch_policy_id == batch_v2.TRUSTED_WIRE_BATCH_V2_POLICY_ID
    assert (
        decoded.typed_replay_policy_id
        == typed_replay_v2.TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID
    )
    assert all(getattr(decoded, name) is True for name in archive_v2._TRUE_CLAIMS_V2)
    assert all(
        getattr(decoded, name) is False for name in archive_v2._FALSE_CLAIMS_V2
    )


def test_public_success_has_no_durable_secret_source_run_or_receipt_material(
    one_case_archive: object,
    six_family_source: tuple[
        Phase2BAdapterRegistry,
        transform.PublicTransformEvidenceBundleV2,
    ],
) -> None:
    assert archive_v2 is not None
    allowed_false_claim_names = set(archive_v2._FALSE_CLAIMS_V2)
    for item in fields(one_case_archive):
        if item.name in allowed_false_claim_names:
            assert getattr(one_case_archive, item.name) is False
            continue
        assert item.name not in FORBIDDEN_DURABLE_NAMES
    for value in _walk_public(one_case_archive):
        assert type(value) is not batch_v2.TrustedWireBatchV2
        assert type(value) is not batch_v2.TrustedWireKeySourcesV2
        assert type(value) is not typed_replay_v2.TypedTrustedWireBatchReplayV2
        assert type(value) is not Phase2BAdapterRegistry
        assert type(value) is not transform.PublicTransformEvidenceBundleV2
    for forbidden in FORBIDDEN_DURABLE_NAMES:
        assert not hasattr(one_case_archive, forbidden)
    public_names = set(_walk_public_field_names(one_case_archive))
    assert not (public_names & set(FORBIDDEN_DURABLE_NAMES))

    source_registry, source_authority = six_family_source
    source_uuids = archive_v2._profile_uuid4_values_v2(
        typed_authority_v1.encode_typed_transform_authority_profile_v1(
            source_authority
        )
    )
    source_uuids.update(item.family_id for item in source_registry.law_bindings)
    source_uuids.update(
        wire for item in source_registry.law_bindings for _, wire in item.role_ids
    )
    source_uuids.update(
        item.quantity_id for item in source_registry.observable_channels
    )
    public_uuid_strings = {
        item
        for item in _walk_public(one_case_archive)
        if type(item) is str
        and len(item) == 36
        and archive_v2._UUID4_V2.fullmatch(item) is not None
    }
    assert source_uuids.isdisjoint(public_uuid_strings)


def test_one_case_public_registry_carries_exact_six_family_vocabulary(
    one_case_archive: object,
) -> None:
    registry = one_case_archive.rows[0].public_registry
    assert len(registry.law_bindings) == 6
    assert sum(len(item.role_ids) for item in registry.law_bindings) == 15
    assert len(registry.observable_channels) == 35
    assert {
        item.law_kind: item.bridge_family_id for item in registry.law_bindings
    } == dict(archive_v2.FROZEN_BRIDGE_FAMILY_UUID_ALIASES_V2)
    assert len({item.public_registry_id for item in one_case_archive.rows}) == 1


def test_issuer_uses_one_private_v2_core_and_one_live_projection_per_case(
    one_case_batch: batch_v2.TrustedWireBatchV2,
    one_case_source_case: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert archive_v2 is not None
    calls = {
        "typed_core": 0,
        "batch_core": 0,
        "projection": 0,
        "derive": 0,
        "shuffle": 0,
        "rename": 0,
    }
    original_typed_core = archive_v2._replay_typed_trusted_wire_batch_core_v2
    original_batch_core = batch_v2._build_trusted_wire_batch_core_v2
    original_derive = batch_v1._derive_keys
    original_shuffle = batch_v1._shuffle_indices
    original_rename = batch_v1._rename_authority_ids

    def monitored_typed_core(**kwargs: object) -> object:
        calls["typed_core"] += 1
        compiler = kwargs["per_case_projection_compiler"]
        assert callable(compiler)

        def monitored_projection(*args: object, **inner_kwargs: object) -> object:
            calls["projection"] += 1
            return compiler(*args, **inner_kwargs)

        forwarded = dict(kwargs)
        forwarded["per_case_projection_compiler"] = monitored_projection
        return original_typed_core(**forwarded)  # type: ignore[arg-type]

    def monitored_batch_core(**kwargs: object) -> object:
        calls["batch_core"] += 1
        return original_batch_core(**kwargs)  # type: ignore[arg-type]

    def monitored_derive(*args: object, **kwargs: object) -> object:
        calls["derive"] += 1
        return original_derive(*args, **kwargs)

    def monitored_shuffle(*args: object, **kwargs: object) -> object:
        calls["shuffle"] += 1
        return original_shuffle(*args, **kwargs)

    def monitored_rename(*args: object, **kwargs: object) -> object:
        calls["rename"] += 1
        return original_rename(*args, **kwargs)

    monkeypatch.setattr(
        archive_v2,
        "_replay_typed_trusted_wire_batch_core_v2",
        monitored_typed_core,
    )
    monkeypatch.setattr(
        batch_v2,
        "_build_trusted_wire_batch_core_v2",
        monitored_batch_core,
    )
    monkeypatch.setattr(batch_v1, "_derive_keys", monitored_derive)
    monkeypatch.setattr(batch_v1, "_shuffle_indices", monitored_shuffle)
    monkeypatch.setattr(batch_v1, "_rename_authority_ids", monitored_rename)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("V2 archive issuer called a public/V1 builder")

    monkeypatch.setattr(batch_v2, "build_trusted_wire_batch_v2", forbidden)
    monkeypatch.setattr(batch_v1, "build_trusted_wire_batch_v1", forbidden)
    monkeypatch.setattr(batch_v1, "_build_trusted_wire_batch_core_v1", forbidden)
    assert not hasattr(archive_v2, "replay_typed_trusted_wire_batch_v2")

    result = archive_v2.issue_trusted_recognizer_input_archive_v2(
        batch=one_case_batch,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=(one_case_source_case,),
    )
    assert type(result) is archive_v2.DecodedRecognizerInputArchiveV2
    assert calls == {
        "typed_core": 1,
        "batch_core": 1,
        "projection": 1,
        "derive": 1,
        "shuffle": 1,
        "rename": 1,
    }


def test_projection_failure_is_atomic_and_never_encodes_partial_archive(
    one_case_batch: batch_v2.TrustedWireBatchV2,
    one_case_source_case: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert archive_v2 is not None

    def fail_projection(**_: object) -> object:
        raise ValueError("projection failed")

    def forbidden_encode(**_: object) -> bytes:
        raise AssertionError("failed projection reached archive encoding")

    monkeypatch.setattr(archive_v2, "_compile_registry_v2", fail_projection)
    monkeypatch.setattr(archive_v2, "_encode_archive_v2", forbidden_encode)
    result = archive_v2.issue_trusted_recognizer_input_archive_v2(
        batch=one_case_batch,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=(one_case_source_case,),
    )
    assert type(result) is archive_v2.TrustedRecognizerInputArchiveRejectionV2
    assert result.disposition is archive_v2.RecognizerInputArchiveDispositionV2.ABSTAIN
    assert result.batch_id is None and result.archive is None
    assert result.rows == result.row_ids == result.envelope_ids == ()
    assert result.public_registry_ids == result.authority_content_ids == ()
    assert result.transform_result_ids == ()


@pytest.mark.parametrize("drift", ("batch", "run", "key", "source_order"))
def test_batch_run_key_and_source_order_drift_are_atomic(
    drift: str,
    two_case_material: tuple[batch_v2.TrustedWireBatchV2, tuple[object, object]],
) -> None:
    assert archive_v2 is not None
    batch, source_cases = two_case_material
    supplied_batch = batch
    run_id = RUN_ID
    key_sources = _keys()
    supplied_cases = source_cases
    if drift == "batch":
        supplied_batch = _unchecked_copy(
            batch,
            batch_id="phase2b_trusted_wire_batch_v2_" + "0" * 64,
        )  # type: ignore[assignment]
    elif drift == "run":
        run_id = b"Q" * 32
    elif drift == "key":
        key_sources = batch_v2.TrustedWireKeySourcesV2(
            b"S" * 32,
            b"J" * 32,
            b"P" * 32,
        )
    else:
        supplied_cases = tuple(reversed(source_cases))  # type: ignore[assignment]

    result = archive_v2.issue_trusted_recognizer_input_archive_v2(
        batch=supplied_batch,
        run_id=run_id,
        key_sources=key_sources,
        source_cases=supplied_cases,
    )
    assert type(result) is archive_v2.TrustedRecognizerInputArchiveRejectionV2
    assert result.batch_id is None and result.archive is None
    assert result.rows == result.row_ids == result.envelope_ids == ()
    assert result.public_registry_ids == result.authority_content_ids == ()
    assert result.transform_result_ids == ()


def test_fixed_public_alias_collision_with_source_registry_is_atomic(
    one_case_batch: batch_v2.TrustedWireBatchV2,
    six_family_source: tuple[
        Phase2BAdapterRegistry,
        transform.PublicTransformEvidenceBundleV2,
    ],
) -> None:
    assert archive_v2 is not None
    registry, authority = six_family_source
    first = registry.law_bindings[0]
    colliding = replace(
        registry,
        law_bindings=(
            replace(
                first,
                family_id=dict(
                    archive_v2.FROZEN_BRIDGE_FAMILY_UUID_ALIASES_V2
                )[first.law_kind],
            ),
            *registry.law_bindings[1:],
        ),
    )
    source_case = archive_v2.TrustedRecognizerSourceCaseV2(
        authority=authority,
        adapter_registry=colliding,
    )
    result = archive_v2.issue_trusted_recognizer_input_archive_v2(
        batch=one_case_batch,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=(source_case,),
    )
    assert type(result) is archive_v2.TrustedRecognizerInputArchiveRejectionV2
    assert result.batch_id is None and result.archive is None
    assert result.rows == result.row_ids == result.envelope_ids == ()
    assert result.public_registry_ids == result.authority_content_ids == ()
    assert result.transform_result_ids == ()


def test_global_source_public_uuid_collision_is_atomic(
    one_case_batch: batch_v2.TrustedWireBatchV2,
    one_case_source_case: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert archive_v2 is not None
    source_wire_id = one_case_source_case.adapter_registry.law_bindings[0].role_ids[0][1]
    original = archive_v2._profile_uuid4_values_v2
    calls = 0

    def inject_collision(root: object, *, values: set[str] | None = None) -> set[str]:
        nonlocal calls
        calls += 1
        result = original(root, values=values)
        if calls >= 2:
            result.add(source_wire_id)
        return result

    monkeypatch.setattr(
        archive_v2,
        "_profile_uuid4_values_v2",
        inject_collision,
    )
    result = archive_v2.issue_trusted_recognizer_input_archive_v2(
        batch=one_case_batch,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=(one_case_source_case,),
    )
    assert calls >= 2
    assert type(result) is archive_v2.TrustedRecognizerInputArchiveRejectionV2
    assert result.batch_id is None and result.archive is None
    assert result.rows == result.row_ids == result.envelope_ids == ()
    assert result.public_registry_ids == result.authority_content_ids == ()
    assert result.transform_result_ids == ()


def test_v1_v2_archive_magic_and_public_decoders_cross_reject(
    one_case_archive: object,
) -> None:
    assert archive_v2 is not None
    v1_magic_spoof = archive_v1.ARCHIVE_MAGIC + one_case_archive.archive[8:]
    with pytest.raises((TypeError, ValueError)):
        archive_v2.decode_public_recognizer_input_archive_v2(v1_magic_spoof)
    with pytest.raises((TypeError, ValueError)):
        archive_v1.decode_public_recognizer_input_archive_v1(one_case_archive.archive)


def test_archive_exact_type_and_byte_cap_reject_before_any_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert archive_v2 is not None

    def forbidden_hash(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid archive reached hashing")

    monkeypatch.setattr(archive_v2.hashlib, "sha256", forbidden_hash)
    with pytest.raises(TypeError):
        archive_v2.decode_public_recognizer_input_archive_v2(bytearray(b"x" * 64))
    monkeypatch.setattr(
        archive_v2,
        "MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2",
        archive_v2.ARCHIVE_HEADER_BYTES_V2,
    )
    with pytest.raises(ValueError, match="byte cap"):
        archive_v2.decode_public_recognizer_input_archive_v2(
            b"x" * (archive_v2.ARCHIVE_HEADER_BYTES_V2 + 1)
        )


def test_decoded_identity_and_root_columns_reject_exact_type_drift_before_parse(
    one_case_archive: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert archive_v2 is not None

    class StringSubclass(str):
        pass

    class TupleSubclass(tuple):
        pass

    def forbidden_parse(_: object) -> object:
        raise AssertionError("cheap decoded closure reached archive parsing")

    monkeypatch.setattr(archive_v2, "_parse_archive_v2", forbidden_parse)
    identity_fields = (
        "archive_id",
        "archive_version",
        "policy_id",
        "batch_id",
        "batch_policy_id",
        "typed_replay_policy_id",
        "typed_authority_schema_id",
        "typed_authority_codec_version",
        "typed_authority_codec_policy_id",
        "public_registry_schema_id",
        "claim_level",
    )
    for field_name in identity_fields:
        original = getattr(one_case_archive, field_name)
        polluted = _unchecked_copy(
            one_case_archive,
            **{field_name: StringSubclass(original)},
        )
        with pytest.raises((TypeError, ValueError)):
            polluted._validate()

    root_fields = (
        "row_ids",
        "envelope_ids",
        "public_registry_ids",
        "authority_content_ids",
        "transform_result_ids",
    )
    for field_name in root_fields:
        original = getattr(one_case_archive, field_name)
        for value in (
            TupleSubclass(original),
            (StringSubclass(original[0]), *original[1:]),
        ):
            polluted = _unchecked_copy(one_case_archive, **{field_name: value})
            with pytest.raises((TypeError, ValueError)):
                polluted._validate()


def test_nested_public_registry_subclasses_reject_before_encoding_or_sets(
    one_case_archive: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert archive_v2 is not None
    registry = one_case_archive.rows[0].public_registry

    class RegistrySubclass(archive_v2.PublicRecognizerRegistryV2):
        pass

    polluted = object.__new__(RegistrySubclass)
    for item in fields(registry):
        object.__setattr__(polluted, item.name, getattr(registry, item.name))

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("nested exact-type drift reached encode/set work")

    monkeypatch.setattr(archive_v2, "_encode_registry_v2", forbidden)
    with pytest.raises(TypeError, match="exact type"):
        polluted._validate()

    law = registry.law_bindings[0]

    class LawSubclass(type(law)):
        pass

    law_polluted = object.__new__(LawSubclass)
    for item in fields(law):
        object.__setattr__(law_polluted, item.name, getattr(law, item.name))
    polluted_registry = _unchecked_copy(
        registry,
        law_bindings=(law_polluted, *registry.law_bindings[1:]),
    )
    with pytest.raises((TypeError, ValueError)):
        polluted_registry._validate()


@pytest.mark.parametrize(
    "mapping",
    (
        {("role_candidate",): _uid(700_001)},
        {("role_candidate", "not-a-uuid"): _uid(700_002)},
        {("unknown_namespace", _uid(700_003)): _uid(700_004)},
    ),
)
def test_private_live_mapping_entries_close_before_registry_or_set_work(
    mapping: dict[object, object],
    six_family_source: tuple[
        Phase2BAdapterRegistry,
        transform.PublicTransformEvidenceBundleV2,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert archive_v2 is not None
    registry, authority = six_family_source

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("malformed live mapping reached registry/deep set work")

    monkeypatch.setattr(archive_v2, "_validate_source_registry_v2", forbidden)
    with pytest.raises((TypeError, ValueError)):
        archive_v2._compile_registry_v2(
            source_registry=registry,
            renamings=MappingProxyType(mapping),
            typed_authority=authority,
        )


def test_private_parsed_context_cannot_validate_different_archive_bytes(
    one_case_archive: object,
) -> None:
    assert archive_v2 is not None
    parsed = archive_v2._parse_archive_v2(one_case_archive.archive)
    altered = bytearray(one_case_archive.archive)
    altered[-1] ^= 1
    with pytest.raises((TypeError, ValueError)):
        archive_v2.DecodedRecognizerInputArchiveV2._issue(
            archive_v2._DECODE_TOKEN_V2,
            archive=bytes(altered),
            parsed=parsed,
        )


def test_v1_batch_is_not_accepted_as_a_v2_archive_batch(
    six_family_source: tuple[
        Phase2BAdapterRegistry,
        transform.PublicTransformEvidenceBundleV2,
    ],
) -> None:
    assert archive_v2 is not None
    registry, authority = six_family_source
    v1_batch = batch_v1.build_trusted_wire_batch_v1(
        authorities=(authority,),
        run_id=RUN_ID,
        key_sources=batch_v1.TrustedWireKeySourcesV1(
            b"S" * 32,
            b"I" * 32,
            b"P" * 32,
        ),
    )
    assert type(v1_batch) is batch_v1.TrustedWireBatchV1
    source_case = archive_v2.TrustedRecognizerSourceCaseV2(
        authority=authority,
        adapter_registry=registry,
    )
    with pytest.raises(TypeError):
        archive_v2.issue_trusted_recognizer_input_archive_v2(
            batch=v1_batch,
            run_id=RUN_ID,
            key_sources=_keys(),
            source_cases=(source_case,),
        )


def test_real_positive_125582_source_crosses_v2_batch_archive_and_derived_bridge(
    positive_fixture: object,
    positive_source: tuple[object, Phase2BAdapterRegistry, object],
    positive_batch: batch_v2.TrustedWireBatchV2,
) -> None:
    assert archive_v2 is not None
    theory, source_registry, source_authority = positive_source
    assert source_authority == positive_fixture.source_authority
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
    assert len(positive_batch.envelopes) == 1
    assert positive_batch.envelopes[0].payload_bytes == 50_255

    source_case = archive_v2.TrustedRecognizerSourceCaseV2(
        authority=source_authority,
        adapter_registry=source_registry,
    )
    result = archive_v2.issue_trusted_recognizer_input_archive_v2(
        batch=positive_batch,
        run_id=RUN_ID,
        key_sources=_keys(),
        source_cases=(source_case,),
    )
    assert type(result) is archive_v2.DecodedRecognizerInputArchiveV2
    assert archive_v2.decode_public_recognizer_input_archive_v2(result.archive) == result
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.envelope == positive_batch.envelopes[0].envelope
    assert row.envelope_id == positive_batch.envelope_ids[0]
    assert row.authority_content_id == positive_batch.authority_content_ids[0]
    assert row.transform_result_id == positive_batch.transform_result_ids[0]

    typed = typed_replay_v2.decode_and_replay_typed_trusted_envelope_v2(
        row.envelope
    )
    assert typed.authority == positive_fixture.public_authority
    assert typed.transform_result_id == row.transform_result_id
    bridge_after = derived_bridge.run_exact_derived_witness_bridge(
        authority=typed.authority,
        theory=theory,
        registry=row.public_registry.to_adapter_registry(),
    )
    assert type(bridge_after) is derived_bridge.ExactDerivedBridgeRun
    assert bridge_after.compilation == positive_fixture.bridge_run.compilation
    assert bridge_after.decision == positive_fixture.bridge_run.decision
    assert all(getattr(result, name) is False for name in archive_v2._FALSE_CLAIMS_V2)
