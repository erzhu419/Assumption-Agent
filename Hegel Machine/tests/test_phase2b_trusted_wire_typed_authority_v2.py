"""Adversarial tests for the lossless compact typed-authority V2 codec.

This codec is representation mechanics only.  The tests deliberately separate
logical authority parity and exact-transform replay from origin, formal audit,
sealed-run, recognizer-effect, and C1 claims, none of which the codec issues.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from typing import Callable

import pytest

import hegel_machine.phase2b_exact_derived_witness_bridge_v1 as derived_bridge
import hegel_machine.phase2b_exact_transform_semantics_v1 as transform
import hegel_machine.phase2b_trusted_wire_typed_authority_v1 as typed_v1
import hegel_machine.phase2b_trusted_wire_typed_authority_v2 as typed_v2
import hegel_machine.phase2b_trusted_wire_v1 as accepted_wire


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

EXPECTED_CERTIFICATE_TYPES = {
    transform.TransformOperation.IDENTITY: transform.IdentityTransformCertificate,
    transform.TransformOperation.UNIT_CONVERSION: transform.UnitConversionCertificate,
    transform.TransformOperation.COORDINATE_AFFINE: transform.CoordinateAffineCertificate,
    transform.TransformOperation.TEMPORAL_AGGREGATION: transform.TemporalAggregationCertificate,
    transform.TransformOperation.SPATIAL_AGGREGATION: transform.SpatialAggregationCertificate,
    transform.TransformOperation.SAMPLING_RESOLUTION: transform.SamplingResolutionCertificate,
    transform.TransformOperation.EQUIVALENT_SPLIT_MERGE: transform.EquivalentSplitMergeCertificate,
    transform.TransformOperation.COARSE_GRAINING: transform.CoarseGrainingCertificate,
}

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def authorities() -> dict[str, transform.PublicTransformEvidenceBundleV2]:
    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_exact_transform_semantics_v1.py"))
    )
    return {name: namespace[name]() for name in FACTORY_NAMES}


@pytest.fixture(scope="module")
def positive_post_rename_fixture() -> object:
    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_recognizer_prediction_archive_v1.py"))
    )
    fixture_function = namespace["public_positive_mechanics_fixture"]
    return fixture_function.__wrapped__()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _profile(
    authority: transform.PublicTransformEvidenceBundleV2,
) -> dict[str, object]:
    return typed_v2.encode_typed_transform_authority_profile_v2(authority)


def _assert_atomic_rejection(
    profile: object,
    *,
    match: str | None = None,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        typed_v2.decode_typed_transform_authority_profile_v2(profile)


def _body_references(value: object) -> set[int]:
    """Validate the public compact grammar and collect string-table indexes."""

    references: set[int] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        exact_type = type(current)
        if exact_type in (type(None), bool):
            continue
        if exact_type is int:
            assert current >= 0
            references.add(current)
            continue
        assert exact_type is list and current
        tag = current[0]
        assert type(tag) is int
        if tag == 0:
            assert len(current) == 2
            assert type(current[1]) is int
            assert abs(current[1]) <= typed_v2.MAXIMUM_SAFE_INTEGER
        elif tag == 1:
            stack.extend(current[1:])
        else:
            assert tag == 2
            assert len(current) % 2 == 1
            keys = current[1::2]
            assert all(type(item) is int and item >= 0 for item in keys)
            assert keys == sorted(set(keys))
            references.update(keys)
            stack.extend(current[2::2])
    return references


def _first_tagged_node(value: object, tag: int) -> list[object]:
    stack = [value]
    while stack:
        current = stack.pop()
        if type(current) is not list or not current:
            continue
        if type(current[0]) is int and current[0] == tag:
            return current
        if current[0] == 0:
            continue
        if current[0] == 1:
            stack.extend(reversed(current[1:]))
        elif current[0] == 2:
            stack.extend(reversed(current[2::2]))
    raise AssertionError(f"compact body has no tag {tag}")


def _object_value(
    object_node: list[object],
    strings: list[str],
    key: str,
) -> object:
    assert object_node[0] == 2
    for index in range(1, len(object_node), 2):
        key_index = object_node[index]
        assert type(key_index) is int
        if strings[key_index] == key:
            return object_node[index + 1]
    raise AssertionError(f"compact object has no key {key!r}")


def _replace_object_value(
    object_node: list[object],
    strings: list[str],
    key: str,
    value: object,
) -> None:
    assert object_node[0] == 2
    for index in range(1, len(object_node), 2):
        key_index = object_node[index]
        assert type(key_index) is int
        if strings[key_index] == key:
            object_node[index + 1] = value
            return
    raise AssertionError(f"compact object has no key {key!r}")


def _table_index(strings: list[str], value: str) -> int:
    return strings.index(value)


def _compact_object(pairs: list[tuple[int, object]]) -> list[object]:
    result: list[object] = [2]
    for key_index, value in sorted(pairs):
        result.extend((key_index, value))
    return result


def _compact_root(body: object, strings: list[str]) -> dict[str, object]:
    return {
        "body": body,
        "codec_version": typed_v2.COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
        "schema_id": typed_v2.COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
        "strings": strings,
    }


def _remap_string_table(
    profile: dict[str, object],
    replacement: Callable[[tuple[str, ...]], tuple[str, ...]],
) -> tuple[list[str], list[object]]:
    """Canonicalize a changed table and remap every existing body reference."""

    old_strings = profile["strings"]
    body = profile["body"]
    assert type(old_strings) is list and type(body) is list
    requested = replacement(tuple(old_strings))
    assert len(requested) == len(set(requested))
    new_strings = sorted(requested)
    new_index = {value: index for index, value in enumerate(new_strings)}
    old_to_new = {index: new_index[value] for index, value in enumerate(old_strings)}

    stack = [body]
    while stack:
        current = stack.pop()
        if type(current) is int:
            raise AssertionError("bare reference cannot be replaced without its parent")
        if type(current) is not list or not current:
            continue
        tag = current[0]
        if tag == 0:
            continue
        if tag == 1:
            for index in range(1, len(current)):
                child = current[index]
                if type(child) is int:
                    current[index] = old_to_new[child]
                else:
                    stack.append(child)
        elif tag == 2:
            for index in range(1, len(current), 2):
                current[index] = old_to_new[current[index]]
                child = current[index + 1]
                if type(child) is int:
                    current[index + 1] = old_to_new[child]
                else:
                    stack.append(child)
    profile["strings"] = new_strings
    return new_strings, body


def test_public_api_and_claim_boundary_are_closed() -> None:
    assert tuple(
        inspect.signature(
            typed_v2.encode_typed_transform_authority_profile_v2
        ).parameters
    ) == ("authority",)
    assert tuple(
        inspect.signature(
            typed_v2.decode_typed_transform_authority_profile_v2
        ).parameters
    ) == ("compact_profile",)
    assert set(typed_v2.__all__) == {
        "COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID",
        "COMPACT_TYPED_AUTHORITY_CODEC_VERSION",
        "COMPACT_TYPED_AUTHORITY_SCHEMA_ID",
        "MAXIMUM_COMPACT_CONTAINER_ENTRIES",
        "MAXIMUM_COMPACT_PROFILE_DEPTH",
        "MAXIMUM_COMPACT_PROFILE_ENTRIES",
        "MAXIMUM_COMPACT_PROFILE_NODES",
        "MAXIMUM_COMPACT_STRING_TABLE_BYTES",
        "MAXIMUM_COMPACT_STRING_TABLE_ENTRIES",
        "MAXIMUM_EXPANDED_ASCII_STRING_BYTES",
        "MAXIMUM_EXPANDED_CONTAINER_ENTRIES",
        "MAXIMUM_EXPANDED_PROFILE_DEPTH",
        "MAXIMUM_EXPANDED_PROFILE_ENTRIES",
        "MAXIMUM_EXPANDED_PROFILE_NODES",
        "MAXIMUM_EXPANDED_TOTAL_STRING_BYTES",
        "MAXIMUM_EXPANDED_UNIQUE_UUIDS",
        "MAXIMUM_EXPANDED_UUID_OCCURRENCES",
        "MAXIMUM_SAFE_INTEGER",
        "decode_typed_transform_authority_profile_v2",
        "encode_typed_transform_authority_profile_v2",
    }
    public_names = " ".join(typed_v2.__all__).casefold()
    for forbidden in (
        "receipt",
        "origin",
        "formal",
        "covert",
        "sealed",
        "c1",
        "effect",
    ):
        assert forbidden not in public_names


def test_resource_caps_bind_the_v1_oracle_and_frozen_raw_formula() -> None:
    assert typed_v2.COMPACT_TYPED_AUTHORITY_SCHEMA_ID == (
        "phase2b_trusted_wire_compact_typed_authority_schema_"
        "9c1a9e7db95ecda7a3568a3974e0d1cf1d4cdc75ebcce6f579dbdf7057bce4f2"
    )
    assert typed_v2.COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID == (
        "phase2b_trusted_wire_compact_typed_authority_codec_policy_"
        "ab01fb9a6f937b3311ec2e622810dbf4acd6023aec4ad765912d169b4dd201e6"
    )
    assert (
        typed_v2.MAXIMUM_EXPANDED_PROFILE_NODES,
        typed_v2.MAXIMUM_EXPANDED_PROFILE_ENTRIES,
        typed_v2.MAXIMUM_EXPANDED_PROFILE_DEPTH,
        typed_v2.MAXIMUM_EXPANDED_CONTAINER_ENTRIES,
        typed_v2.MAXIMUM_EXPANDED_ASCII_STRING_BYTES,
        typed_v2.MAXIMUM_EXPANDED_UUID_OCCURRENCES,
        typed_v2.MAXIMUM_EXPANDED_UNIQUE_UUIDS,
        typed_v2.MAXIMUM_SAFE_INTEGER,
    ) == (
        accepted_wire.MAXIMUM_PROFILE_NODES,
        accepted_wire.MAXIMUM_PROFILE_NODES,
        accepted_wire.MAXIMUM_PROFILE_DEPTH,
        accepted_wire.MAXIMUM_ARRAY_ENTRIES,
        accepted_wire.MAXIMUM_ASCII_STRING_BYTES,
        accepted_wire.MAXIMUM_UUID_OCCURRENCES,
        accepted_wire.MAXIMUM_UNIQUE_UUIDS,
        accepted_wire.MAXIMUM_SAFE_INTEGER,
    ) == (16_384, 16_384, 64, 4_096, 2_048, 2_048, 1_024, (1 << 53) - 1)
    assert typed_v2.MAXIMUM_EXPANDED_TOTAL_STRING_BYTES == 4_194_304
    assert typed_v2.MAXIMUM_COMPACT_PROFILE_NODES == 98_308
    assert typed_v2.MAXIMUM_COMPACT_PROFILE_ENTRIES == 98_308
    assert typed_v2.MAXIMUM_COMPACT_PROFILE_DEPTH == 66
    assert typed_v2.MAXIMUM_COMPACT_CONTAINER_ENTRIES == 32_768
    assert typed_v2.MAXIMUM_COMPACT_STRING_TABLE_ENTRIES == 32_768
    assert typed_v2.MAXIMUM_COMPACT_STRING_TABLE_BYTES == 4_194_304
    assert typed_v2.MAXIMUM_COMPACT_TOTAL_STRING_BYTES == 4_206_592
    assert typed_v2.MAXIMUM_COMPACT_PROFILE_NODES == (
        3
        * (
            typed_v2.MAXIMUM_EXPANDED_PROFILE_NODES
            + typed_v2.MAXIMUM_EXPANDED_PROFILE_ENTRIES
        )
        + 4
    )
    assert typed_v2._COMPACT_CAP_MANIFEST == (
        ("container_entries", 32_768),
        ("depth", 66),
        ("nodes", 98_308),
        (
            "resource_formula",
            "N=expanded_nodes<=16384;E=expanded_entries<=16384;"
            "C=logical_container_nodes;I=logical_integer_leaves;"
            "K=logical_object_entries;S=distinct_string_values_and_keys;"
            "C+I<=N;K<=E;S<=N+E;"
            "body_physical_nodes=N+C+2I+K<=3N+E;"
            "body_physical_entries=E+C+2I+K<=2N+2E;"
            "full_root_adds_S+4_nodes_and_entries;"
            "raw_nodes<=4N+2E+4;raw_entries<=3N+3E+4;"
            "unified_raw_cap=3*(Nmax+Emax)+4=98308;"
            "string_table_cap=Nmax+Emax=32768;"
            "container_cap=max(1+2*expanded_container,string_table_cap)"
            "=32768;depth=expanded_depth+2=66;"
            "raw_string_bytes=table_bytes+6*expanded_ascii_string_bytes",
        ),
        ("scope", "in_memory_codec_preflight_not_envelope_capacity"),
        ("string_table_bytes", 4_194_304),
        ("string_table_entries", 32_768),
        ("total_entries", 98_308),
        ("total_string_bytes", 4_206_592),
    )


def test_all_eight_certificates_and_two_step_round_trip_with_exact_parity(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    operations: set[transform.TransformOperation] = set()
    for name in FACTORY_NAMES:
        authority = authorities[name]
        before = transform.run_exact_transform_semantics(authority)
        assert type(before) is transform.ExactTransformCompilation
        assert before.disposition is transform.TransformCompilationDisposition.COMPLETE

        compact = _profile(authority)
        decoded = typed_v2.decode_typed_transform_authority_profile_v2(compact)
        assert type(decoded) is transform.PublicTransformEvidenceBundleV2
        assert decoded == authority
        assert decoded.content_id == authority.content_id
        assert typed_v1.encode_typed_transform_authority_profile_v1(decoded) == (
            typed_v1.encode_typed_transform_authority_profile_v1(authority)
        )
        assert _profile(decoded) == compact
        assert _canonical_bytes(_profile(decoded)) == _canonical_bytes(compact)
        assert accepted_wire.encode_phase2b_jcs_profile_v1(compact) == (
            _canonical_bytes(compact)
        )

        after = transform.run_exact_transform_semantics(decoded)
        assert type(after) is transform.ExactTransformCompilation
        assert after.disposition is transform.TransformCompilationDisposition.COMPLETE
        assert after.result_id == before.result_id
        assert after.wrapper_content_id == decoded.content_id
        for contract in decoded.transform_contracts:
            assert type(contract.certificate) is EXPECTED_CERTIFICATE_TYPES[
                contract.operation
            ]
            operations.add(contract.operation)

    assert operations == set(transform.TransformOperation)
    two_step = authorities["two_step_authority"]
    assert len(two_step.transform_contracts) == 2
    assert tuple(
        item.transform_id for item in two_step.transform_contracts
    ) == tuple(
        item.transform_id
        for item in typed_v2.decode_typed_transform_authority_profile_v2(
            _profile(two_step)
        ).transform_contracts
    )


def test_compact_root_table_and_body_are_exact_canonical_mechanics(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    compact = _profile(authorities["two_step_authority"])
    assert type(compact) is dict
    assert tuple(sorted(compact)) == (
        "body",
        "codec_version",
        "schema_id",
        "strings",
    )
    assert compact["codec_version"] == typed_v2.COMPACT_TYPED_AUTHORITY_CODEC_VERSION
    assert compact["schema_id"] == typed_v2.COMPACT_TYPED_AUTHORITY_SCHEMA_ID
    strings = compact["strings"]
    body = compact["body"]
    assert type(strings) is list and type(body) is list
    assert body and body[0] == 2
    assert strings == sorted(set(strings))
    assert strings and all(type(item) is str and item.isascii() for item in strings)
    assert len(strings) <= typed_v2.MAXIMUM_COMPACT_STRING_TABLE_ENTRIES
    assert sum(len(item.encode("ascii")) for item in strings) <= (
        typed_v2.MAXIMUM_COMPACT_STRING_TABLE_BYTES
    )
    references = _body_references(body)
    assert references == set(range(len(strings)))

    stack = [body]
    while stack:
        current = stack.pop()
        assert type(current) is not str
        if type(current) is list:
            stack.extend(current)


def test_real_positive_post_rename_profile_is_compact_and_replays_exactly(
    positive_post_rename_fixture: object,
) -> None:
    authority = positive_post_rename_fixture.public_authority
    bridge_before = positive_post_rename_fixture.bridge_run
    assert type(authority) is transform.PublicTransformEvidenceBundleV2
    assert bridge_before.decision.disposition.name == "ADMISSIBLE_SCALE_SET"

    expanded = typed_v1.encode_typed_transform_authority_profile_v1(authority)
    expanded_bytes = accepted_wire._encode_profile_value(expanded).encode("ascii")
    assert len(expanded_bytes) == 125_582
    assert len(expanded_bytes) > accepted_wire.MAXIMUM_PAYLOAD_BYTES == 65_424

    compact = _profile(authority)
    compact_bytes = _canonical_bytes(compact)
    assert accepted_wire.encode_phase2b_jcs_profile_v1(compact) == compact_bytes
    assert len(compact_bytes) == 49_473
    assert len(compact["strings"]) == 397  # type: ignore[arg-type]
    assert len(hashlib.sha256(compact_bytes).hexdigest()) == 64
    assert len(compact_bytes) < accepted_wire.MAXIMUM_PAYLOAD_BYTES
    assert accepted_wire.MAXIMUM_PAYLOAD_BYTES - len(compact_bytes) == 15_951
    decoded = typed_v2.decode_typed_transform_authority_profile_v2(compact)
    assert decoded == authority
    assert decoded.content_id == authority.content_id
    before = transform.run_exact_transform_semantics(authority)
    after = transform.run_exact_transform_semantics(decoded)
    assert type(after) is transform.ExactTransformCompilation
    assert after.disposition is transform.TransformCompilationDisposition.COMPLETE
    assert after.result_id == before.result_id
    bridge_after = derived_bridge.run_exact_derived_witness_bridge(
        authority=decoded,
        theory=positive_post_rename_fixture.theory,
        registry=positive_post_rename_fixture.public_registry.to_adapter_registry(),
    )
    assert type(bridge_after) is derived_bridge.ExactDerivedBridgeRun
    assert bridge_after.compilation == bridge_before.compilation
    assert bridge_after.compilation.result_id == bridge_before.compilation.result_id
    assert bridge_after.decision == bridge_before.decision
    assert bridge_after.decision.decision_id == bridge_before.decision.decision_id
    assert bridge_after.decision.disposition.name == "ADMISSIBLE_SCALE_SET"


@pytest.mark.parametrize("mutation", ("reverse", "duplicate", "unused", "nonascii"))
def test_string_table_requires_sorted_unique_used_ascii_entries(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    mutation: str,
) -> None:
    compact = _profile(authorities["identity_authority"])
    strings = compact["strings"]
    assert type(strings) is list and len(strings) >= 2
    if mutation == "reverse":
        strings[0], strings[1] = strings[1], strings[0]
    elif mutation == "duplicate":
        strings[1] = strings[0]
    elif mutation == "unused":
        strings.append("~unused-table-entry")
    else:
        strings[0] = "nonascii-\u00e9"
    _assert_atomic_rejection(compact)


@pytest.mark.parametrize("invalid", (-1, True, 1 << 53, "0", None))
def test_string_indexes_reject_negative_bool_out_of_range_and_wrong_types(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    invalid: object,
) -> None:
    compact = _profile(authorities["identity_authority"])
    body = compact["body"]
    strings = compact["strings"]
    assert type(body) is list and type(strings) is list
    body[1] = len(strings) if invalid == 1 << 53 else invalid
    _assert_atomic_rejection(compact)


def test_unknown_tags_wrong_arity_and_noncompact_objects_reject_atomically(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    original = _profile(authorities["identity_authority"])
    mutations: tuple[object, ...] = (
        [9],
        [0],
        [0, 1, 2],
        [2, original["body"][1]],  # type: ignore[index]
        {"not": "compact"},
    )
    for body in mutations:
        compact = deepcopy(original)
        compact["body"] = body
        _assert_atomic_rejection(compact)


@pytest.mark.parametrize("inline", ("inline", {"odd": "object"}, 1.5))
def test_inline_string_object_and_float_nodes_are_never_compact_values(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    inline: object,
) -> None:
    compact = _profile(authorities["identity_authority"])
    body = compact["body"]
    assert type(body) is list
    body[2] = inline
    _assert_atomic_rejection(compact, match="inline|string|object|float|non-JCS")


@pytest.mark.parametrize("mutation", ("duplicate", "reverse"))
def test_compact_object_keys_are_unique_and_strictly_increasing(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    mutation: str,
) -> None:
    compact = _profile(authorities["identity_authority"])
    body = compact["body"]
    assert type(body) is list and body[0] == 2 and len(body) >= 5
    if mutation == "duplicate":
        body[3] = body[1]
    else:
        first_pair = body[1:3]
        second_pair = body[3:5]
        body[1:5] = [*second_pair, *first_pair]
    _assert_atomic_rejection(compact)


def test_v1_and_v2_roots_cross_reject_without_normalization(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    authority = authorities["identity_authority"]
    logical_v1 = typed_v1.encode_typed_transform_authority_profile_v1(authority)
    compact_v2 = _profile(authority)
    with pytest.raises((TypeError, ValueError)):
        typed_v2.decode_typed_transform_authority_profile_v2(logical_v1)
    with pytest.raises((TypeError, ValueError)):
        typed_v1.decode_typed_transform_authority_profile_v1(compact_v2)


def test_v1_logical_codec_ids_and_identity_bytes_are_unchanged(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    assert typed_v1.TYPED_AUTHORITY_SCHEMA_ID == (
        "phase2b_trusted_wire_typed_authority_schema_"
        "9429e96b9192db4546b92b011779e99352decb051a023d8883682539c804b730"
    )
    assert typed_v1.TYPED_AUTHORITY_CODEC_POLICY_ID == (
        "phase2b_trusted_wire_typed_authority_codec_policy_"
        "8fc5714399f0e31c43bf8fa3c818c4cc8afb8b4c49b57ae8af309821abfcc4b3"
    )
    logical = typed_v1.encode_typed_transform_authority_profile_v1(
        authorities["identity_authority"]
    )
    payload = _canonical_bytes(logical)
    assert len(payload) == 5_785
    assert hashlib.sha256(payload).hexdigest() == (
        "219d7c3c9d9519f359791f4df8678f1a36749c2ea34450667b1276360db82c9a"
    )


def test_codec_does_not_mutate_input_authority_or_compact_profile(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    authority = authorities["two_step_authority"]
    logical_before = typed_v1.encode_typed_transform_authority_profile_v1(authority)
    compact = _profile(authority)
    compact_before = deepcopy(compact)
    decoded = typed_v2.decode_typed_transform_authority_profile_v2(compact)
    assert compact == compact_before
    assert typed_v1.encode_typed_transform_authority_profile_v1(authority) == (
        logical_before
    )
    assert decoded == authority


def test_decode_and_encode_reject_builtin_and_dataclass_subclasses(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    class DictSubclass(dict):
        pass

    class ListSubclass(list):
        pass

    class IntSubclass(int):
        pass

    class StringSubclass(str):
        pass

    compact = _profile(authorities["identity_authority"])
    _assert_atomic_rejection(DictSubclass(compact))

    for field_name, wrapper in (("strings", ListSubclass), ("body", ListSubclass)):
        polluted = deepcopy(compact)
        polluted[field_name] = wrapper(polluted[field_name])
        _assert_atomic_rejection(polluted)

    polluted_index = deepcopy(compact)
    polluted_index["body"][1] = IntSubclass(polluted_index["body"][1])  # type: ignore[index]
    _assert_atomic_rejection(polluted_index)

    polluted_string = deepcopy(compact)
    polluted_string["strings"][0] = StringSubclass(  # type: ignore[index]
        polluted_string["strings"][0]  # type: ignore[index]
    )
    _assert_atomic_rejection(polluted_string)

    class AuthoritySubclass(transform.PublicTransformEvidenceBundleV2):
        pass

    source = authorities["identity_authority"]
    spoof = object.__new__(AuthoritySubclass)
    for field in fields(source):
        object.__setattr__(spoof, field.name, getattr(source, field.name))
    with pytest.raises(TypeError, match="exact"):
        typed_v2.encode_typed_transform_authority_profile_v2(spoof)


def test_compact_raw_container_node_entry_depth_and_table_caps_fail_closed(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _profile(authorities["identity_authority"])

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("raw resource bomb reached expansion, V1 decode, or hash")

    monkeypatch.setattr(typed_v2, "_expand_compact_value", forbidden)
    monkeypatch.setattr(typed_v2, "stable_hash", forbidden)
    monkeypatch.setattr(
        typed_v2._typed_v1,
        "decode_typed_transform_authority_profile_v1",
        forbidden,
    )

    oversized_container = deepcopy(original)
    oversized_container["body"] = [
        1,
        *([None] * typed_v2.MAXIMUM_COMPACT_CONTAINER_ENTRIES),
    ]
    _assert_atomic_rejection(oversized_container, match="container.*cap")

    oversized_table = deepcopy(original)
    oversized_table["strings"] = [
        f"s{index:04d}"
        for index in range(typed_v2.MAXIMUM_COMPACT_STRING_TABLE_ENTRIES + 1)
    ]
    _assert_atomic_rejection(
        oversized_table,
        match="(?:container|string table count).*(?:cap|bounds)?",
    )

    long_rows = [
        f"{index:04d}-" + "x" * (typed_v2.MAXIMUM_EXPANDED_ASCII_STRING_BYTES - 5)
        for index in range(2_049)
    ]
    oversized_table_bytes = _compact_root([2, 0, None], long_rows)
    _assert_atomic_rejection(oversized_table_bytes, match="string table.*cap")

    nested: object = None
    for _ in range(typed_v2.MAXIMUM_COMPACT_PROFILE_DEPTH + 1):
        nested = [1, nested]
    oversized_depth = deepcopy(original)
    oversized_depth["body"] = nested
    _assert_atomic_rejection(oversized_depth, match="raw depth cap")

    # One outer array holding 12,000 seven-leaf arrays stays under the frozen
    # local-list cap but exceeds the unified aggregate raw node/entry budget.
    aggregate_bomb = deepcopy(original)
    aggregate_bomb["body"] = [
        1,
        *([[1, None, None, None, None, None, None, None]] * 12_000),
    ]
    _assert_atomic_rejection(
        aggregate_bomb,
        match="(?:container|raw (?:node|entry)).*cap",
    )


def test_expanded_node_entry_depth_and_container_caps_precede_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("resource bomb reached expansion, V1 decode, or hash")

    monkeypatch.setattr(typed_v2, "_expand_compact_value", forbidden)
    monkeypatch.setattr(typed_v2, "stable_hash", forbidden)
    monkeypatch.setattr(
        typed_v2._typed_v1,
        "decode_typed_transform_authority_profile_v1",
        forbidden,
    )

    strings = ["root", "value"]
    # Exactly 16,384 logical edges and therefore 16,385 logical nodes:
    # root object edge + four outer-array edges + inner string-reference edges.
    inner_counts = (4_096, 4_096, 4_096, 4_091)
    node_body = [
        2,
        0,
        [1, *([1, *([1] * inner_counts[0])],
                [1, *([1] * inner_counts[1])],
                [1, *([1] * inner_counts[2])],
                [1, *([1] * inner_counts[3])])],
    ]
    _assert_atomic_rejection(
        _compact_root(node_body, strings),
        match="(?:raw|expanded) (?:node|entry) cap",
    )

    # Make the independently bound entry branch observable.
    monkeypatch.setattr(
        typed_v2,
        "MAXIMUM_EXPANDED_PROFILE_NODES",
        typed_v2.MAXIMUM_EXPANDED_PROFILE_NODES * 2,
    )
    entry_body = deepcopy(node_body)
    entry_body[2][4].append(1)  # type: ignore[index,union-attr]
    _assert_atomic_rejection(
        _compact_root(entry_body, strings),
        match="(?:raw|expanded) (?:node|entry) cap",
    )

    oversized_array = _compact_root(
        [2, 0, [1, *([1] * (typed_v2.MAXIMUM_EXPANDED_CONTAINER_ENTRIES + 1))]],
        strings,
    )
    _assert_atomic_rejection(
        oversized_array,
        match="(?:container|expanded array).*cap",
    )

    nested: object = 1
    for _ in range(typed_v2.MAXIMUM_EXPANDED_PROFILE_DEPTH + 1):
        nested = [1, nested]
    expanded_depth = _compact_root([2, 0, nested], strings)
    _assert_atomic_rejection(
        expanded_depth,
        match="(?:raw|expanded) depth cap",
    )


def test_reference_expansion_string_and_uuid_bombs_reject_before_expand_or_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("resource bomb reached expansion, V1 decode, or hashing")

    monkeypatch.setattr(typed_v2, "_expand_compact_value", forbidden)
    monkeypatch.setattr(typed_v2, "stable_hash", forbidden)
    monkeypatch.setattr(
        typed_v2._typed_v1,
        "decode_typed_transform_authority_profile_v1",
        forbidden,
    )

    repeated = "a" * typed_v2.MAXIMUM_EXPANDED_ASCII_STRING_BYTES
    strings = [repeated, "root"]
    repeated_string_body = [2, 1, [1, *([0] * 2_049)]]
    _assert_atomic_rejection(
        _compact_root(repeated_string_body, strings),
        match="expanded string cap",
    )

    repeated_uuid = "00000000-0000-4000-8000-000000000001"
    strings = [repeated_uuid, "root"]
    repeated_uuid_body = [
        2,
        1,
        [1, *([0] * (typed_v2.MAXIMUM_EXPANDED_UUID_OCCURRENCES + 1))],
    ]
    _assert_atomic_rejection(
        _compact_root(repeated_uuid_body, strings),
        match="UUID occurrence cap",
    )

    uuid_values = [
        f"10000000-0000-4000-8000-{index:012x}"
        for index in range(typed_v2.MAXIMUM_EXPANDED_UNIQUE_UUIDS + 1)
    ]
    strings = sorted(["root", *uuid_values])
    root_index = strings.index("root")
    uuid_indices = [strings.index(value) for value in uuid_values]
    unique_uuid_body = [2, root_index, [1, *uuid_indices]]
    _assert_atomic_rejection(
        _compact_root(unique_uuid_body, strings),
        match="unique UUID cap",
    )


def test_uuid_object_keys_charge_repeated_bytes_but_not_uuid_value_occurrences() -> None:
    uuid_key = "00000000-0000-4000-8000-000000000001"
    strings = sorted(["root", uuid_key])
    root_index = strings.index("root")
    uuid_index = strings.index(uuid_key)
    repeated_objects = [
        [2, uuid_index, None]
        for _ in range(2_049)
    ]
    body = [2, root_index, [1, *repeated_objects]]
    plan = typed_v2._plan_expanded_profile(body, tuple(strings))
    assert plan.expanded_uuid_occurrences == 0
    assert plan.expanded_unique_uuids == 0
    assert plan.expanded_string_bytes == (
        len("root") + 2_049 * len(uuid_key)
    )


def test_logical_safe_integer_and_rational_bit_caps_survive_compaction(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    compact = _profile(authorities["identity_authority"])
    integer_node = _first_tagged_node(compact["body"], 0)
    integer_node[1] = typed_v2.MAXIMUM_SAFE_INTEGER + 1
    _assert_atomic_rejection(compact, match="safe-integer cap")

    logical = typed_v1.encode_typed_transform_authority_profile_v1(
        authorities["identity_authority"]
    )
    coefficient = (
        logical["transform_contracts"][0]["kernel_rows"][0]["terms"][0][
            "coefficient"
        ]
    )
    coefficient["numerator_decimal"] = "1" + "0" * 1_300
    rational_bomb = typed_v2._compact_logical_profile(logical)
    _assert_atomic_rejection(rational_bomb, match="bit(?: cap|-length budget)")


def test_encoder_pollution_hits_resource_and_exact_type_gates_before_emission(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = authorities["identity_authority"]

    def forbidden_emit(*args: object, **kwargs: object) -> object:
        raise AssertionError("polluted authority reached logical profile emission")

    monkeypatch.setattr(typed_v1, "_profile_value_unchecked", forbidden_emit)

    oversized_tuple = deepcopy(source)
    object.__setattr__(
        oversized_tuple,
        "observation_metadata",
        (oversized_tuple.observation_metadata[0],) * 4_097,
    )
    with pytest.raises(ValueError, match="tuple.*cap"):
        typed_v2.encode_typed_transform_authority_profile_v2(oversized_tuple)

    oversized_string = deepcopy(source)
    object.__setattr__(
        oversized_string.base_bundle.observations[0],
        "provenance_sha256",
        "a" * 2_049,
    )
    with pytest.raises(ValueError, match="string cap"):
        typed_v2.encode_typed_transform_authority_profile_v2(oversized_string)

    oversized_integer = deepcopy(source)
    object.__setattr__(
        oversized_integer.transform_contracts[0].input_components[0],
        "ordinal",
        1 << 53,
    )
    with pytest.raises(ValueError, match="safe-integer cap"):
        typed_v2.encode_typed_transform_authority_profile_v2(oversized_integer)

    oversized_atom = deepcopy(source)
    atom = oversized_atom.transform_contracts[0].kernel_rows[0].terms[0].coefficient
    object.__setattr__(atom, "numerator", 1 << 4_096)
    with pytest.raises(ValueError, match="bit cap"):
        typed_v2.encode_typed_transform_authority_profile_v2(oversized_atom)

    repeated_uuid = deepcopy(source)
    existing = repeated_uuid.base_bundle.role_ids[0]
    object.__setattr__(
        repeated_uuid.base_bundle,
        "role_ids",
        (existing,) * (typed_v2.MAXIMUM_EXPANDED_UUID_OCCURRENCES + 1),
    )
    with pytest.raises(ValueError, match="UUID occurrence cap"):
        typed_v2.encode_typed_transform_authority_profile_v2(repeated_uuid)

    unique_uuid = deepcopy(source)
    object.__setattr__(
        unique_uuid.base_bundle,
        "role_ids",
        tuple(
            f"10000000-0000-4000-8000-{index:012x}"
            for index in range(typed_v2.MAXIMUM_EXPANDED_UNIQUE_UUIDS + 1)
        ),
    )
    with pytest.raises(ValueError, match="unique UUID cap"):
        typed_v2.encode_typed_transform_authority_profile_v2(unique_uuid)


def test_pythonhashseed_does_not_change_compact_bytes(
    authorities: dict[str, transform.PublicTransformEvidenceBundleV2],
) -> None:
    script = """
import hashlib, json, runpy
from hegel_machine.phase2b_trusted_wire_typed_authority_v2 import encode_typed_transform_authority_profile_v2
namespace = runpy.run_path('tests/test_phase2b_exact_transform_semantics_v1.py')
authority = namespace['two_step_authority']()
profile = encode_typed_transform_authority_profile_v2(authority)
payload = json.dumps(profile, ensure_ascii=False, allow_nan=False, separators=(',', ':'), sort_keys=True).encode('ascii')
print(hashlib.sha256(payload).hexdigest())
"""
    expected = hashlib.sha256(
        _canonical_bytes(_profile(authorities["two_step_authority"]))
    ).hexdigest()
    digests = []
    for seed in ("0", "1", "271828"):
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": seed,
                "PYTHONPATH": str(ROOT / "src"),
            }
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        digests.append(completed.stdout.strip())
    assert digests == [expected, expected, expected]
