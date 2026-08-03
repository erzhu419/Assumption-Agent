from __future__ import annotations

import pytest

from hegel_machine.phase3_m25_wire_v1 import decode_formal_object
from hegel_machine.phase3_m3_bounded_enumerator_v1 import (
    BoundedEnumerationError,
    EnumerationBindingsV1,
    SCOPE_EXTENSIONS,
    enumerate_bounded_closure_v1,
    program_mdl_length_q32,
)
from hegel_machine.strict_ast_shrink1_v1 import (
    canonicalize_shrink1_source_ast,
    decode_shrink1_canonical_ast,
)
from hegel_machine.strict_cbor_v1 import canonical_cbor_encode


def _bindings() -> EnumerationBindingsV1:
    return EnumerationBindingsV1(b"\x11" * 32, b"\x22" * 32, b"\x33" * 32)


def test_scope_extensions_are_the_exact_33_canonical_values() -> None:
    assert len(SCOPE_EXTENSIONS) == 33
    assert len(set(SCOPE_EXTENSIONS)) == 33
    assert sum(len(item) == 0 for item in SCOPE_EXTENSIONS) == 1
    assert sum(len(item) == 1 for item in SCOPE_EXTENSIONS) == 8
    assert sum(len(item) == 2 for item in SCOPE_EXTENSIONS) == 24
    assert all(tuple(sorted(item)) == item for item in SCOPE_EXTENSIONS)


def test_old_program_mdl_q32_uses_exact_frozen_prefix_lengths() -> None:
    scalar = canonicalize_shrink1_source_ast(("scalar_const", 0, 1))
    aggregate = canonicalize_shrink1_source_ast(
        ("aggregate", "sum_v1", "scope_all_observed_v1", "q0", ())
    )
    aggregate_two = canonicalize_shrink1_source_ast(
        (
            "aggregate",
            "sum_v1",
            "scope_all_observed_v1",
            "q0",
            (("c0", False), ("c3", True)),
        )
    )
    assert program_mdl_length_q32(scalar) == 8 << 32
    assert program_mdl_length_q32(aggregate) == 12 << 32
    assert program_mdl_length_q32(aggregate_two) == 19 << 32


def test_small_prefix_is_ordered_and_formal_wire_replays() -> None:
    result = enumerate_bounded_closure_v1(
        _bindings(), canonical_budget=5, raw_application_cap=5_000_000
    )
    assert result.closure_status == "DSL_TOO_LARGE"
    assert result.raw_operator_application_count == 6
    assert result.canonical_program_count == 5
    assert result.traversal_prefix_complete
    assert not result.authoritative_claim_allowed
    assert result.first_out_of_budget_program_hash is not None
    assert len(result.bucket_accounting_records) == 175
    keys = []
    for index, record in enumerate(result.canonical_program_records):
        decoded = decode_formal_object(
            canonical_cbor_encode(record), expected_name="CanonicalProgramRecordV2"
        )
        assert decoded.fields["program_index"] == index
        ast = decode_shrink1_canonical_ast(decoded.fields["canonical_ast_cbor_bytes"])
        keys.append(
            (
                decoded.fields["ast_depth"],
                decoded.fields["ast_node_count"],
                decoded.fields["output_sort_id"],
                ast.root_operator_id,
                decoded.fields["canonical_ast_cbor_bytes"],
            )
        )
    assert keys == sorted(keys)
    assert len(result.program_chunk_manifests) == 1


def test_small_prefix_closes_buckets_in_global_traversal_order() -> None:
    result = enumerate_bounded_closure_v1(
        _bindings(), canonical_budget=10, raw_application_cap=20_000
    )
    # Six Bool leaves close first; the eight Bit leaves are then fully generated
    # before indices 6..9 and witness index 10 are selected from that bucket.
    assert result.raw_operator_application_count == 14
    assert result.closure_status == "DSL_TOO_LARGE"
    assert result.canonical_program_count == 10
    assert not result.authoritative_claim_allowed


def test_raw_cap_fails_closed_before_any_formal_prefix_claim() -> None:
    with pytest.raises(BoundedEnumerationError, match="INCONCLUSIVE_BUDGET") as caught:
        enumerate_bounded_closure_v1(
            _bindings(), canonical_budget=50_000, raw_application_cap=1
        )
    assert caught.value.code == "INCONCLUSIVE_BUDGET"


def test_exact_50001_prefix_matches_independent_replay_golden() -> None:
    result = enumerate_bounded_closure_v1(_bindings())
    assert result.closure_status == "DSL_TOO_LARGE"
    assert result.raw_operator_application_count == 3_292_439
    assert result.canonical_program_count == 50_000
    assert result.canonical_program_archive_root.hex() == (
        "a23151e07f77edcbebe5b7e2e382e1a81b36c6b15c8997899f7f43dcbda874d1"
    )
    assert result.program_chunk_manifest_root.hex() == (
        "98c8deb02a62630f5813717a28c3b9deb5a3845e663b9af5a78fe7f9427f453d"
    )
    assert result.bucket_accounting_root.hex() == (
        "5dd13e5d284785dab7fbe3c16fbb1f1bcba3a44466ab0fb258f75f36ee9661ec"
    )
    assert result.first_out_of_budget_program_hash.hex() == (
        "96200a6a131204315ffcd1efd0aa2dcfe2ce665a2c06516461772c9812f0ec71"
    )
    assert result.first_out_of_budget_cbor.hex() == (
        "820184020383010083000103860003050300828201f58203f5"
    )
    assert result.authoritative_claim_allowed is False
