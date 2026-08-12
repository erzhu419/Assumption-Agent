"""V2 trusted-wire typed-replay contract tests.

This slice proves only compact trusted-wire representation, direct public
typed replay, and deterministic private custodian rebuild mechanics.  It does
not establish source or secret custody, origin, formal/covert audit,
sealed-holdout, recognizer capacity/effect, or C1 exit evidence.
"""

from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path
import runpy
from typing import Iterator

import pytest

import hegel_machine.phase2b_exact_transform_semantics_v1 as transform
import hegel_machine.phase2b_trusted_wire_batch_v1 as batch_v1
import hegel_machine.phase2b_trusted_wire_batch_v2 as batch_v2
import hegel_machine.phase2b_trusted_wire_typed_replay_v1 as typed_replay_v1
import hegel_machine.phase2b_trusted_wire_typed_replay_v2 as typed_replay_v2


RUN_ID = b"R" * 32
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


def _v1_keys() -> batch_v1.TrustedWireKeySourcesV1:
    return batch_v1.TrustedWireKeySourcesV1(b"S" * 32, b"I" * 32, b"P" * 32)


def _v2_keys() -> batch_v2.TrustedWireKeySourcesV2:
    return batch_v2.TrustedWireKeySourcesV2(b"S" * 32, b"I" * 32, b"P" * 32)


def _unchecked_copy(value: object, **changes: object) -> object:
    copied = object.__new__(type(value))
    for item in fields(value):
        object.__setattr__(
            copied,
            item.name,
            changes.get(item.name, getattr(value, item.name)),
        )
    return copied


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


@pytest.fixture(scope="module")
def authorities() -> dict[str, transform.PublicTransformEvidenceBundleV2]:
    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_exact_transform_semantics_v1.py"))
    )
    return {name: namespace[name]() for name in FACTORY_NAMES}


@pytest.fixture(scope="module")
def v2_batch(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> batch_v2.TrustedWireBatchV2:
    result = batch_v2.build_trusted_wire_batch_v2(
        authorities=(
            authorities["identity_authority"],
            authorities["unit_authority"],
        ),
        run_id=RUN_ID,
        key_sources=_v2_keys(),
    )
    assert type(result) is batch_v2.TrustedWireBatchV2
    return result


@pytest.fixture(scope="module")
def v1_batch(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> batch_v1.TrustedWireBatchV1:
    result = batch_v1.build_trusted_wire_batch_v1(
        authorities=(
            authorities["identity_authority"],
            authorities["unit_authority"],
        ),
        run_id=RUN_ID,
        key_sources=_v1_keys(),
    )
    assert type(result) is batch_v1.TrustedWireBatchV1
    return result


@pytest.fixture(scope="module")
def v2_public_replay(
    v2_batch: batch_v2.TrustedWireBatchV2,
) -> typed_replay_v2.TypedTrustedWireBatchReplayV2:
    result = typed_replay_v2.replay_typed_trusted_wire_batch_v2(batch=v2_batch)
    assert type(result) is typed_replay_v2.TypedTrustedWireBatchReplayV2
    return result


def test_v2_public_typed_replay_api_has_no_custodian_rebuild_inputs() -> None:
    assert tuple(
        inspect.signature(
            typed_replay_v2.decode_and_replay_typed_trusted_envelope_v2
        ).parameters
    ) == ("envelope",)
    replay = inspect.signature(
        typed_replay_v2.replay_typed_trusted_wire_batch_v2
    )
    assert tuple(replay.parameters) == ("batch",)
    assert replay.parameters["batch"].kind is inspect.Parameter.KEYWORD_ONLY
    for forbidden in (
        "run_id",
        "key_sources",
        "authorities",
        "source_authorities",
        "receipt",
        "policy",
    ):
        assert forbidden not in replay.parameters
    assert typed_replay_v2.__all__ == (
        "TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID",
        "TYPED_TRUSTED_WIRE_REPLAY_V2_VERSION",
        "TypedTrustedWireBatchReplayRejectionV2",
        "TypedTrustedWireBatchReplayV2",
        "TypedTrustedWireReplayDispositionV2",
        "decode_and_replay_typed_trusted_envelope_v2",
        "replay_typed_trusted_wire_batch_v2",
    )


def test_v2_typed_replay_field_claim_and_policy_manifests_are_exact() -> None:
    assert tuple(
        item.name
        for item in fields(
            typed_replay_v2.TypedTrustedWireBatchReplayRejectionV2
        )
    ) == (
        "disposition",
        "reason",
        "authority_count",
        "batch_id",
        "rows",
        "row_ids",
        "authority_content_ids",
        "transform_result_ids",
        "replay_policy_id",
        "claim_level",
    )
    assert tuple(
        item.name for item in fields(typed_replay_v2.TypedTrustedWireBatchReplayV2)
    ) == (
        "disposition",
        "batch",
        "batch_id",
        "batch_policy_id",
        "rows",
        "row_ids",
        "authority_content_ids",
        "transform_result_ids",
        "replay_policy_id",
        "claim_level",
        "batch_policy_membership_verified",
        "whole_batch_atomic_typed_replay_verified",
        "compact_authority_canonical_replay_verified",
        "public_provenance_verified",
        "direct_exact_transform_replay_verified",
        "secret_custodian_replay_verified",
        "whole_batch_shuffle_publicly_verified",
        "purpose_separated_keys_publicly_verified",
        "post_shuffle_hmac_uuidv4_publicly_verified",
        "secret_hmac_padding_publicly_verified",
        "source_authority_binding_verified",
        "live_allocation_schedule_verified",
        "recognizer_capacity_evidence",
        "origin_authenticated",
        "formal_uuid_audit",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "c1_exit_evidence",
    )
    policy = typed_replay_v2._REPLAY_POLICY_VALUE_V2
    assert policy["receipt_field_manifest"] == list(
        typed_replay_v2._RECEIPT_FIELD_MANIFEST_V2
    )
    assert policy["rejection_field_manifest"] == list(
        typed_replay_v2._REJECTION_FIELD_MANIFEST_V2
    )
    assert policy["claim_manifest"] == {
        "always_false": list(typed_replay_v2._FALSE_CLAIMS_V2),
        "public_complete_true": list(typed_replay_v2._TRUE_CLAIMS_V2),
    }
    assert policy["custodian_material_in_public_api"] is False
    assert policy["claim_value_type"] == "exact_builtin_bool"
    assert policy["receipt_id_formula"] == {
        "fields": [
            "authority_content_ids",
            "batch_id",
            "batch_policy_id",
            "replay_policy_id",
            "row_ids",
            "transform_result_ids",
        ],
        "prefix": "phase2b_typed_trusted_wire_batch_replay_v2_",
        "validate_before_hash": True,
    }
    assert policy["row_id_formula"] == {
        "fields": [
            "authority_content_id",
            "batch_id",
            "envelope_id",
            "namespace_audit_id",
            "padding_sha256",
            "payload_sha256",
            "replay_policy_id",
            "transform_result_id",
            "typed_authority_codec_policy_id",
            "typed_authority_codec_version",
            "typed_authority_schema_id",
        ],
        "prefix": "phase2b_typed_trusted_wire_row_v2_",
        "validate_before_hash": True,
    }


def test_public_v2_typed_replay_never_rebuilds_or_touches_secret_mechanics(
    v2_batch: batch_v2.TrustedWireBatchV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("public V2 typed replay attempted custodian rebuild")

    monkeypatch.setattr(batch_v2, "build_trusted_wire_batch_v2", forbidden)
    monkeypatch.setattr(batch_v2, "_build_trusted_wire_batch_core_v2", forbidden)
    for name in ("_derive_keys", "_shuffle_indices", "_rename_authority_ids"):
        monkeypatch.setattr(batch_v1, name, forbidden)

    row = typed_replay_v2.decode_and_replay_typed_trusted_envelope_v2(
        v2_batch.envelopes[0].envelope
    )
    result = typed_replay_v2.replay_typed_trusted_wire_batch_v2(batch=v2_batch)
    assert row.authority_content_id == v2_batch.authority_content_ids[0]
    assert row.transform_result_id == v2_batch.transform_result_ids[0]
    assert type(result) is typed_replay_v2.TypedTrustedWireBatchReplayV2
    assert (
        result.disposition
        is typed_replay_v2.TypedTrustedWireReplayDispositionV2.COMPLETE
    )


def test_public_complete_binds_exact_ordered_batch_and_row_roots_without_custodian_data(
    v2_batch: batch_v2.TrustedWireBatchV2,
    v2_public_replay: typed_replay_v2.TypedTrustedWireBatchReplayV2,
) -> None:
    result = v2_public_replay
    assert result.batch is v2_batch
    assert result.batch_id == v2_batch.batch_id
    assert result.batch_policy_id == batch_v2.TRUSTED_WIRE_BATCH_V2_POLICY_ID
    assert result.authority_content_ids == v2_batch.authority_content_ids
    assert result.transform_result_ids == v2_batch.transform_result_ids
    assert tuple(item.envelope_id for item in result.rows) == v2_batch.envelope_ids
    assert tuple(item.payload_sha256 for item in result.rows) == tuple(
        item.payload_sha256 for item in v2_batch.envelopes
    )
    assert tuple(item.padding_sha256 for item in result.rows) == tuple(
        item.padding_sha256 for item in v2_batch.envelopes
    )
    assert tuple(item.namespace_audit_id for item in result.rows) == tuple(
        item.namespace_audit_id for item in v2_batch.envelopes
    )
    assert len(result.rows) == len(result.row_ids) == 2
    assert len(set(result.row_ids)) == 2
    assert result.receipt_id.startswith(
        "phase2b_typed_trusted_wire_batch_replay_v2_"
    )
    for forbidden_name in (
        "source_authorities",
        "source_authority_content_ids",
        "key_sources",
        "shuffle_ikm",
        "id_ikm",
        "padding_ikm",
        "secret_replay_receipt",
        "secret_replay_receipt_id",
    ):
        assert not hasattr(result, forbidden_name)
    assert all(
        type(value) is not batch_v2.TrustedWireKeySourcesV2
        for value in _walk_public(result)
    )
    for name in (
        "whole_batch_shuffle_publicly_verified",
        "purpose_separated_keys_publicly_verified",
        "post_shuffle_hmac_uuidv4_publicly_verified",
        "secret_hmac_padding_publicly_verified",
    ):
        assert getattr(result.batch, name) is False


def test_public_complete_has_only_narrow_public_true_claims(
    v2_public_replay: typed_replay_v2.TypedTrustedWireBatchReplayV2,
) -> None:
    result = v2_public_replay
    assert result.claim_level == "NON_AUTHORITATIVE_MECHANICS_ONLY"
    assert all(getattr(result, name) is True for name in typed_replay_v2._TRUE_CLAIMS_V2)
    assert all(
        getattr(result, name) is False for name in typed_replay_v2._FALSE_CLAIMS_V2
    )


@pytest.mark.parametrize("claim_name", typed_replay_v2._FALSE_CLAIMS_V2)
def test_coherent_public_receipt_cannot_upgrade_false_claims(
    claim_name: str,
    v2_public_replay: typed_replay_v2.TypedTrustedWireBatchReplayV2,
) -> None:
    forged = _unchecked_copy(v2_public_replay, **{claim_name: True})
    with pytest.raises((TypeError, ValueError), match="claim"):
        forged._validate()  # type: ignore[attr-defined]


def test_zero_argument_receipt_validation_replays_every_exact_batch_row(
    v2_public_replay: typed_replay_v2.TypedTrustedWireBatchReplayV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bytes] = []
    original = typed_replay_v2.decode_and_replay_typed_trusted_envelope_v2

    def monitored(envelope: bytes) -> object:
        calls.append(envelope)
        return original(envelope)

    monkeypatch.setattr(
        typed_replay_v2,
        "decode_and_replay_typed_trusted_envelope_v2",
        monitored,
    )
    v2_public_replay._validate()
    assert calls == [item.envelope for item in v2_public_replay.batch.envelopes]


def test_public_second_row_failure_is_atomic_with_exact_empty_roots(
    v2_batch: batch_v2.TrustedWireBatchV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = typed_replay_v2.decode_and_replay_typed_trusted_envelope_v2
    calls = 0

    def fail_second(envelope: bytes) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("second row failed")
        return original(envelope)

    monkeypatch.setattr(
        typed_replay_v2,
        "decode_and_replay_typed_trusted_envelope_v2",
        fail_second,
    )
    result = typed_replay_v2.replay_typed_trusted_wire_batch_v2(batch=v2_batch)
    assert type(result) is typed_replay_v2.TypedTrustedWireBatchReplayRejectionV2
    assert result.disposition is typed_replay_v2.TypedTrustedWireReplayDispositionV2.ABSTAIN
    assert result.rows == result.row_ids == ()
    assert result.authority_content_ids == result.transform_result_ids == ()


@pytest.mark.parametrize(
    ("field_name", "mutation"),
    (
        ("batch_id", "string_subclass"),
        ("row_ids", "tuple_subclass"),
        ("row_ids", "item_subclass"),
        ("authority_content_ids", "item_subclass"),
        ("transform_result_ids", "item_subclass"),
        ("batch_policy_membership_verified", "integer_bool"),
    ),
)
def test_receipt_pollution_rejects_exact_types_before_row_replay_or_hash(
    field_name: str,
    mutation: str,
    v2_public_replay: typed_replay_v2.TypedTrustedWireBatchReplayV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StringSubclass(str):
        pass

    class TupleSubclass(tuple):
        pass

    original = getattr(v2_public_replay, field_name)
    if mutation == "string_subclass":
        polluted = StringSubclass(original)
    elif mutation == "tuple_subclass":
        polluted = TupleSubclass(original)
    elif mutation == "item_subclass":
        polluted = (StringSubclass(original[0]), *original[1:])
    else:
        polluted = 1
    forged = _unchecked_copy(v2_public_replay, **{field_name: polluted})

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("cheap receipt drift reached row replay or hashing")

    monkeypatch.setattr(
        typed_replay_v2,
        "decode_and_replay_typed_trusted_envelope_v2",
        forbidden,
    )
    monkeypatch.setattr(typed_replay_v2, "stable_hash", forbidden)
    with pytest.raises((TypeError, ValueError)):
        forged._validate()  # type: ignore[attr-defined]


@pytest.mark.parametrize("field_name", ("rows", "row_ids", "authority_content_ids", "transform_result_ids"))
def test_rejection_partial_roots_require_exact_empty_tuples(field_name: str) -> None:
    values = {
        "disposition": typed_replay_v2.TypedTrustedWireReplayDispositionV2.ABSTAIN,
        "reason": "test_rejection",
        "authority_count": 1,
        "batch_id": None,
        field_name: [],
    }
    with pytest.raises((TypeError, ValueError), match="partial"):
        typed_replay_v2.TypedTrustedWireBatchReplayRejectionV2(**values)


def test_rejection_scalars_are_exact_and_bounded_before_any_root_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StringSubclass(str):
        pass

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid rejection reached hashing")

    monkeypatch.setattr(typed_replay_v2, "stable_hash", forbidden)
    with pytest.raises((TypeError, ValueError)):
        typed_replay_v2.TypedTrustedWireBatchReplayRejectionV2(
            typed_replay_v2.TypedTrustedWireReplayDispositionV2.ABSTAIN,
            "x" * 2_049,
            1,
            None,
        )
    with pytest.raises((TypeError, ValueError)):
        typed_replay_v2.TypedTrustedWireBatchReplayRejectionV2(
            typed_replay_v2.TypedTrustedWireReplayDispositionV2.ABSTAIN,
            "valid",
            True,
            None,
        )
    with pytest.raises((TypeError, ValueError)):
        typed_replay_v2.TypedTrustedWireBatchReplayRejectionV2(
            typed_replay_v2.TypedTrustedWireReplayDispositionV2.ABSTAIN,
            "valid",
            1,
            None,
            replay_policy_id=StringSubclass(
                typed_replay_v2.TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID
            ),
        )


def test_private_core_builds_once_with_one_live_projection_per_case(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    v2_batch: batch_v2.TrustedWireBatchV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (
        authorities["identity_authority"],
        authorities["unit_authority"],
    )
    calls = {"core": 0, "projection": 0, "derive": 0, "shuffle": 0, "rename": 0}
    original_core = batch_v2._build_trusted_wire_batch_core_v2
    original_derive = batch_v1._derive_keys
    original_shuffle = batch_v1._shuffle_indices
    original_rename = batch_v1._rename_authority_ids

    def monitored_core(**kwargs: object) -> object:
        calls["core"] += 1
        compiler = kwargs["per_case_projection_compiler"]
        assert callable(compiler)

        def monitored_projection(*args: object, **inner_kwargs: object) -> object:
            calls["projection"] += 1
            return compiler(*args, **inner_kwargs)

        forwarded = dict(kwargs)
        forwarded["per_case_projection_compiler"] = monitored_projection
        return original_core(**forwarded)  # type: ignore[arg-type]

    def derive(*args: object, **kwargs: object) -> object:
        calls["derive"] += 1
        return original_derive(*args, **kwargs)

    def shuffle(*args: object, **kwargs: object) -> object:
        calls["shuffle"] += 1
        return original_shuffle(*args, **kwargs)

    def rename(*args: object, **kwargs: object) -> object:
        calls["rename"] += 1
        return original_rename(*args, **kwargs)

    monkeypatch.setattr(batch_v2, "_build_trusted_wire_batch_core_v2", monitored_core)
    monkeypatch.setattr(batch_v1, "_derive_keys", derive)
    monkeypatch.setattr(batch_v1, "_shuffle_indices", shuffle)
    monkeypatch.setattr(batch_v1, "_rename_authority_ids", rename)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("private core called a public or V1 builder")

    monkeypatch.setattr(batch_v2, "build_trusted_wire_batch_v2", forbidden)
    monkeypatch.setattr(batch_v1, "build_trusted_wire_batch_v1", forbidden)
    monkeypatch.setattr(batch_v1, "_build_trusted_wire_batch_core_v1", forbidden)

    def projection(
        source_index: int,
        renamings: object,
        authority: transform.PublicTransformEvidenceBundleV2,
    ) -> object:
        assert type(source_index) is int
        assert renamings
        assert type(authority) is transform.PublicTransformEvidenceBundleV2
        return ("projection", source_index, authority.content_id)

    replay, projections = typed_replay_v2._replay_typed_trusted_wire_batch_core_v2(
        authorities=source,
        run_id=RUN_ID,
        key_sources=_v2_keys(),
        expected_batch=v2_batch,
        per_case_projection_compiler=projection,
    )
    assert type(replay) is typed_replay_v2.TypedTrustedWireBatchReplayV2
    assert replay.batch == v2_batch
    assert len(projections) == len(source)
    assert calls == {
        "core": 1,
        "projection": len(source),
        "derive": 1,
        "shuffle": 1,
        "rename": len(source),
    }


@pytest.mark.parametrize("failure", ("projection", "expected_batch", "public_replay"))
def test_private_core_failure_returns_atomic_empty_projection_and_roots(
    failure: str,
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    v2_batch: batch_v2.TrustedWireBatchV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (
        authorities["identity_authority"],
        authorities["unit_authority"],
    )

    def projection(*args: object, **kwargs: object) -> object:
        if failure == "projection":
            raise ValueError("projection failed")
        return ("projection", args[0])

    expected = v2_batch
    if failure == "expected_batch":
        expected = _unchecked_copy(
            v2_batch,
            batch_id="phase2b_trusted_wire_batch_v2_" + "0" * 64,
        )  # type: ignore[assignment]
    if failure == "public_replay":
        monkeypatch.setattr(
            typed_replay_v2,
            "replay_typed_trusted_wire_batch_v2",
            lambda **_: typed_replay_v2.TypedTrustedWireBatchReplayRejectionV2(
                typed_replay_v2.TypedTrustedWireReplayDispositionV2.ABSTAIN,
                "forced_public_replay_failure",
                len(source),
                v2_batch.batch_id,
            ),
        )

    replay, projections = typed_replay_v2._replay_typed_trusted_wire_batch_core_v2(
        authorities=source,
        run_id=RUN_ID,
        key_sources=_v2_keys(),
        expected_batch=expected,
        per_case_projection_compiler=projection,
    )
    assert type(replay) is typed_replay_v2.TypedTrustedWireBatchReplayRejectionV2
    assert replay.rows == replay.row_ids == ()
    assert replay.authority_content_ids == replay.transform_result_ids == ()
    assert projections == ()


def test_private_core_catches_polluted_batch_rejection_validation(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polluted = object.__new__(batch_v2.TrustedWireBatchRejectionV2)
    valid = batch_v2.TrustedWireBatchRejectionV2(
        batch_v2.TrustedWireBatchDispositionV2.ABSTAIN,
        "controlled_rejection",
        2,
    )
    for item in fields(valid):
        object.__setattr__(polluted, item.name, getattr(valid, item.name))
    object.__setattr__(polluted, "reason", "x" * 2_049)

    monkeypatch.setattr(
        batch_v2,
        "_build_trusted_wire_batch_core_v2",
        lambda **_: (polluted, ()),
    )
    replay, projections = typed_replay_v2._replay_typed_trusted_wire_batch_core_v2(
        authorities=(
            authorities["identity_authority"],
            authorities["unit_authority"],
        ),
        run_id=RUN_ID,
        key_sources=_v2_keys(),
        per_case_projection_compiler=lambda *_: object(),
    )
    assert type(replay) is typed_replay_v2.TypedTrustedWireBatchReplayRejectionV2
    assert replay.rows == replay.row_ids == ()
    assert replay.authority_content_ids == replay.transform_result_ids == ()
    assert projections == ()


def test_v1_and_v2_envelope_batch_and_replay_types_cross_reject(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    v1_batch: batch_v1.TrustedWireBatchV1,
    v2_batch: batch_v2.TrustedWireBatchV2,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        typed_replay_v2.decode_and_replay_typed_trusted_envelope_v2(
            v1_batch.envelopes[0].envelope
        )
    with pytest.raises((TypeError, ValueError)):
        typed_replay_v1.decode_and_replay_typed_trusted_envelope_v1(
            v2_batch.envelopes[0].envelope
        )
    with pytest.raises(TypeError):
        typed_replay_v2.replay_typed_trusted_wire_batch_v2(  # type: ignore[arg-type]
            batch=v1_batch
        )
    with pytest.raises(TypeError):
        typed_replay_v1.replay_typed_trusted_wire_batch_v1(  # type: ignore[arg-type]
            batch=v2_batch,
            run_id=RUN_ID,
            key_sources=_v1_keys(),
            authorities=(
                authorities["identity_authority"],
                authorities["unit_authority"],
            ),
        )
