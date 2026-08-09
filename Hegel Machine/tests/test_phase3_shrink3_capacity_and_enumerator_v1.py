from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hegel_machine import phase3_m3_bounded_enumerator_shrink3_v1 as child
from hegel_machine import phase3_m3_shrink3_diagnostic_profile_v1 as profile
from hegel_machine.phase3_m3_bounded_enumerator_shrink3_v1 import (
    BoundedEnumerationError,
    DUAL_ENUMERATION_QUALIFIED,
    EnumerationBindingsV1,
    _Shrink3Enumerator,
    _witness_status,
    enumerate_bounded_closure_shrink3_v1,
)
from hegel_machine.phase3_shrink3_capacity_v1 import (
    EXPECTED_SHRINK3_SOURCE_COUNT,
    iter_shrink3_capacity_candidate_asts,
    shrink3_constant_atoms_v1,
    shrink3_mixed_atoms_v1,
    shrink3_rational_aggregate_leaves_v1,
)
from hegel_machine.strict_ast_shrink2_v1 import canonicalize_shrink2_source_ast
from hegel_machine.strict_ast_shrink3_v1 import (
    canonicalize_shrink3_source_ast,
    decode_shrink3_canonical_ast,
)


def _bindings() -> EnumerationBindingsV1:
    return EnumerationBindingsV1(bytes(32), bytes((1,)) * 32, bytes((2,)) * 32)


def _contains_binary_operator(value: object, operator_id: int) -> bool:
    if not isinstance(value, tuple):
        return False
    if (
        len(value) == 4
        and value[0] == 2
        and value[1] == operator_id
    ):
        return True
    return any(
        _contains_binary_operator(item, operator_id)
        for item in value
        if isinstance(item, tuple)
    )


def _direct_replay(mode: str) -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[1]
    entrypoint = (
        project_root
        / "src/hegel_machine/phase3_shrink3_capacity_entrypoint_v1.py"
    )
    probe = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(entrypoint), mode],
        cwd=project_root,
        env={"PATH": os.environ.get("PATH", "")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    return json.loads(probe.stdout)


def test_inherited_survivor_subset_is_exact_and_byte_stable() -> None:
    assert len(shrink3_constant_atoms_v1()) == 15
    assert len(shrink3_rational_aggregate_leaves_v1()) == 16
    assert len(shrink3_mixed_atoms_v1()) == 144
    sources = tuple(iter_shrink3_capacity_candidate_asts())
    assert len(sources) == EXPECTED_SHRINK3_SOURCE_COUNT == 2_160

    parents = tuple(canonicalize_shrink2_source_ast(source) for source in sources)
    children = tuple(canonicalize_shrink3_source_ast(source) for source in sources)
    assert len({program.cbor_bytes for program in children}) == 2_160
    assert all(
        parent.cbor_bytes == child.cbor_bytes and parent.hash_id == child.hash_id
        for parent, child in zip(parents, children)
    )
    assert all(
        decode_shrink3_canonical_ast(child.cbor_bytes) == child
        for child in children
    )


def test_direct_python_capacity_replay_is_target_free_and_nonterminal() -> None:
    report = _direct_replay("--capacity-replay")
    assert report["source_candidate_count"] == 2_160
    assert report["accepted_unique_count"] == 2_160
    assert report["parent_identity_match_count"] == 2_160
    assert report["accepted_set_commitment"] == (
        "sha256:9045e4ebe6416dcbf699e7972f25468aef45c0f0aec0e58806061b7ce64d790e"
    )
    assert report["subset_status"] == "SURVIVOR_SUBSET_ONLY_NOT_COMPLETE"
    assert report["executed_closure_status"] == "NOT_RUN"
    assert report["complete_closure_enumerated"] is False
    assert report["formal_roots"] is None
    assert report["target_or_split_modules_loaded"] is False
    assert all(
        "target" not in name and "split" not in name
        for name in report["loaded_hegel_modules"]
    )


def test_direct_python_golden_replay_freezes_36_vectors() -> None:
    report = _direct_replay("--golden-replay")
    assert report["vector_count"] == report["passed_count"] == 36
    assert report["surviving_identity_checks"] == 8
    assert report["source_add_rejection_checks"] == 4
    assert report["source_priority_checks"] == 6
    assert report["formal_add_rejection_checks"] == 3
    assert report["formal_priority_checks"] == 6
    assert report["formal_shape_priority_checks"] == 6
    assert report["formal_alias_or_reserved_checks"] == 3
    assert report["execution_state"] == "NOT_RUN"
    assert report["formal_roots"] is None
    assert report["target_or_split_modules_loaded"] is False


def test_bounded_enumerator_never_archives_add_but_retains_difference() -> None:
    state = _Shrink3Enumerator(raw_cap=500_000)
    state.leaves(5)
    state.binary_and_ternary(depth=1, nodes=3, output_sort_id=5)
    root_operator_ids = {
        program.ast.root_operator_id
        for program in state.groups[("RationalValue", 1, 3)]
    }
    assert 0x0200 not in root_operator_ids
    assert 0x0201 in root_operator_ids

    result = enumerate_bounded_closure_shrink3_v1(
        _bindings(), canonical_budget=2_000, raw_application_cap=500_000
    )
    assert result.dsl_version == "hegel-old-dsl-v1.3.0"
    assert result.closure_status == "DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED"
    assert result.canonical_program_count == 2_000
    assert result.first_out_of_budget_program_hash is not None
    assert result.authoritative_claim_allowed is False
    assert DUAL_ENUMERATION_QUALIFIED is False


def test_prefix_witness_records_and_accounting_are_deterministic() -> None:
    bindings = _bindings()
    first = enumerate_bounded_closure_shrink3_v1(
        bindings, canonical_budget=100, raw_application_cap=100_000
    )
    second = enumerate_bounded_closure_shrink3_v1(
        bindings, canonical_budget=100, raw_application_cap=100_000
    )

    assert first == second
    assert first.closure_status == "DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED"
    assert first.raw_operator_application_count == 279
    assert first.first_out_of_budget_program_hash is not None
    assert first.first_out_of_budget_program_hash.hex() == (
        "e81319a9e65502b66a1b7cac9ba18daaa50411e361e798a7e90c3e4098013e24"
    )
    assert first.first_out_of_budget_cbor is not None
    assert first.first_out_of_budget_cbor.hex() == (
        "8201860003010100828200f58203f4"
    )
    witness = decode_shrink3_canonical_ast(first.first_out_of_budget_cbor)
    assert witness.digest == first.first_out_of_budget_program_hash
    assert not _contains_binary_operator(witness.value, 0)

    records = first.canonical_program_records
    assert len(records) == first.canonical_program_count == 100
    assert tuple(record[3] for record in records) == tuple(range(100))
    for record in records:
        ast = decode_shrink3_canonical_ast(record[4])
        assert not _contains_binary_operator(ast.value, 0)
        assert record[5] == ast.digest
        assert record[6] == child._base.OUTPUT_SORT_IDS[ast.metrics.output_sort]
        assert record[7] == ast.metrics.depth
        assert record[8] == ast.metrics.node_count
        assert record[9] == len(ast.metrics.distinct_bit_slots)
        assert record[10] == child._base.program_mdl_length_q32(ast)
        assert record[11:] == (
            bindings.child_dsl_spec_root,
            bindings.operator_semantics_root,
            bindings.identifier_registry_root,
        )

    manifests = first.program_chunk_manifests
    assert manifests == child._base._chunk_manifests(records)
    assert len(manifests) == 1
    assert manifests[0][3:7] == (0, 0, 99, 100)
    assert manifests[0][7] == child._base.rfc6962_root(list(records))
    assert first.canonical_program_archive_root == child._base.rfc6962_root(
        list(records)
    )
    assert first.program_chunk_manifest_root == child._base.rfc6962_root(
        list(manifests)
    )

    buckets = first.bucket_accounting_records
    assert len(buckets) == 5 * 5 * 7 == 175
    assert tuple(bucket[3] for bucket in buckets) == tuple(range(175))
    assert sum(bucket[7] for bucket in buckets) == (
        first.raw_operator_application_count
    )
    assert sum(bucket[8] for bucket in buckets) == first.canonical_program_count
    assert first.bucket_accounting_root == child._base.rfc6962_root(
        list(buckets)
    )


def test_python_rust_100_program_production_binding_golden_is_exact() -> None:
    bindings = EnumerationBindingsV1(
        *profile.NON_FORMAL_SYNTHETIC_CHILD_BINDINGS
    )
    result = enumerate_bounded_closure_shrink3_v1(
        bindings, canonical_budget=100, raw_application_cap=100_000
    )
    assert result.raw_operator_application_count == 279
    assert result.first_out_of_budget_program_hash is not None
    assert result.first_out_of_budget_program_hash.hex() == (
        "e81319a9e65502b66a1b7cac9ba18daaa50411e361e798a7e90c3e4098013e24"
    )
    assert result.first_out_of_budget_cbor is not None
    assert result.first_out_of_budget_cbor.hex() == (
        "8201860003010100828200f58203f4"
    )
    assert result.canonical_program_archive_root.hex() == (
        "6240741c26ee13b39d4e0c1496c07422029fc97f4670fe99ee1a87ca7b34f838"
    )
    assert result.program_chunk_manifest_root.hex() == (
        "7e5ec72460295d5a07ade499a1f17e445750b60d1dbccb4aa92d64bc26fad992"
    )
    assert result.bucket_accounting_root.hex() == (
        "7aa4b4146bfca7218dd348fe45f6e4cadacc1bb2a46df5941fce8b262d265791"
    )


def test_bounded_enumerator_raw_cap_fails_closed() -> None:
    with pytest.raises(BoundedEnumerationError) as raised:
        enumerate_bounded_closure_shrink3_v1(
            _bindings(), canonical_budget=2_000, raw_application_cap=1
        )
    assert raised.value.code == "INCONCLUSIVE_BUDGET"


def test_frozen_budget_upper_bounds_are_not_expandable() -> None:
    with pytest.raises(ValueError, match="canonical_budget"):
        enumerate_bounded_closure_shrink3_v1(
            _bindings(), canonical_budget=50_001
        )
    with pytest.raises(ValueError, match="raw_application_cap"):
        enumerate_bounded_closure_shrink3_v1(
            _bindings(), raw_application_cap=5_000_001
        )


def test_finite_frontier_complete_and_raw_cap_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OneProgramEnumerator(_Shrink3Enumerator):
        def leaves(self, output_sort_id: int) -> None:
            if output_sort_id == 1:
                self._admit(
                    child._base._strict._Expr(
                        0, 4, "Bool", parameters=(0,)
                    ),
                    (1, 0, 1),
                    known_normal_form=True,
                )

        def unary(self, depth: int, nodes: int, output_sort_id: int) -> None:
            return None

        def binary_and_ternary(
            self, depth: int, nodes: int, output_sort_id: int
        ) -> None:
            return None

        def conjunctions(self, depth: int, nodes: int) -> None:
            return None

    monkeypatch.setattr(child, "_Shrink3Enumerator", _OneProgramEnumerator)
    complete = enumerate_bounded_closure_shrink3_v1(
        _bindings(), canonical_budget=10, raw_application_cap=1
    )
    assert complete.closure_status == "COMPLETE"
    assert complete.raw_operator_application_count == 1
    assert complete.canonical_program_count == 1
    assert complete.first_out_of_budget_program_hash is None
    assert complete.first_out_of_budget_cbor is None
    assert complete.traversal_prefix_complete is True
    assert complete.authoritative_claim_allowed is False

    class _TwoProgramEnumerator(_OneProgramEnumerator):
        def leaves(self, output_sort_id: int) -> None:
            if output_sort_id == 1:
                for context_id in (0, 1):
                    self._admit(
                        child._base._strict._Expr(
                            0, 4, "Bool", parameters=(context_id,)
                        ),
                        (1, 0, 1),
                        known_normal_form=True,
                    )

    monkeypatch.setattr(child, "_Shrink3Enumerator", _TwoProgramEnumerator)
    with pytest.raises(BoundedEnumerationError) as raised:
        enumerate_bounded_closure_shrink3_v1(
            _bindings(), canonical_budget=10, raw_application_cap=1
        )
    assert raised.value.code == "INCONCLUSIVE_BUDGET"


def test_reduced_raw_cap_cannot_claim_frozen_budget_overflow() -> None:
    assert _witness_status(50_000, 5_000_000) == "DSL_TOO_LARGE"
    assert _witness_status(50_000, 4_999_999) == (
        "DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED"
    )
    assert _witness_status(49_999, 5_000_000) == (
        "DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED"
    )


def test_non_formal_enumeration_binding_roots_are_exact() -> None:
    assert profile.diagnostic_root_hex_v1() == {
        "child_dsl_spec_root": (
            "64aaf01392ca89a1ade3a3766d756b53e9b0e7ec6ab4ca2b4fb74ec658490677"
        ),
        "operator_semantics_root": (
            "e3337cc67974c8fbbfa6d8f89301184c1658a98b80c0a1fac11251ede9aa15f1"
        ),
        "identifier_registry_root": (
            "9dd80c452334db8afd9fbb56f1c74f365f63db61ec4c5667bddbb88e57ec05c8"
        ),
        "canonical_ast_schema_root": (
            "bc426bb2dbb519533cfe79a33e5d677d23a56b6f73a4e781f327fc303946f92c"
        ),
        "canonical_cbor_profile_root": (
            "072d40b7db3b91c283ff401b77766149f9b3ae7283cf8fc865090052100ccc96"
        ),
    }


def test_reduced_budget_report_cannot_masquerade_as_terminal() -> None:
    result = enumerate_bounded_closure_shrink3_v1(
        _bindings(), canonical_budget=100, raw_application_cap=100_000
    )
    report = child.diagnostic_report_shrink3_v1(
        result,
        _bindings(),
        canonical_budget=100,
        raw_application_cap=100_000,
    )
    assert report["closure_status"] == "DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED"
    assert report["canonical_program_budget"] == 100
    assert report["raw_operator_application_cap"] == 100_000
    assert report["first_out_of_budget_ordinal_or_null"] == 101
    assert report["authoritative_claim_allowed"] is False
    assert report["execution_state"] == "NOT_RUN"
    assert report["formal_roots"] is None


@pytest.mark.parametrize(
    ("entrypoint_name", "mode", "expected_module_count"),
    [
        (
            "phase3_m3_shrink3_isolated_entrypoint_v1.py",
            "--target-free-self-check",
            14,
        ),
        (
            "phase3_m3_shrink3_dual_diagnostic_entrypoint_v1.py",
            "--host-self-check",
            15,
        ),
    ],
)
def test_new_direct_entrypoints_have_exact_target_free_module_closures(
    entrypoint_name: str,
    mode: str,
    expected_module_count: int,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    entrypoint = project_root / "src/hegel_machine" / entrypoint_name
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(entrypoint), mode],
        cwd=project_root,
        env={"PATH": os.environ.get("PATH", "")},
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["execution_state"] == "NOT_RUN"
    assert report["formal_roots_generated"] is False
    assert report["formal_roots"] is None
    assert report["target_free_isolation_verified"] is True
    assert report["target_or_split_modules_loaded"] is False
    assert len(report["loaded_hegel_modules"]) == expected_module_count
    assert all(
        fragment not in module
        for module in report["loaded_hegel_modules"]
        for fragment in (
            "_target",
            "_split_",
            "_seed",
            "_role",
            "_evaluator",
            "_odd",
            "_sink",
        )
    )
