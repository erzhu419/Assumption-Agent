from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hegel_machine import phase3_m3_bounded_enumerator_shrink5_v1 as child
from hegel_machine import phase3_m3_shrink5_diagnostic_profile_v1 as profile
from hegel_machine.phase3_m3_bounded_enumerator_shrink5_v1 import (
    BoundedEnumerationError,
    DUAL_ENUMERATION_QUALIFIED,
    EnumerationBindingsV1,
    _Shrink5Enumerator,
    _witness_status,
    enumerate_bounded_closure_shrink5_v1,
)
from hegel_machine.phase3_shrink5_capacity_v1 import (
    EXPECTED_SHRINK5_BOUNDARY_SOURCE_COUNT,
    EXPECTED_SHRINK5_SOURCE_COUNT,
    EXPECTED_SHRINK5_SURVIVOR_SOURCE_COUNT,
    iter_shrink5_boundary_candidate_asts,
    iter_shrink5_capacity_candidate_asts,
    shrink5_constant_atoms_v1,
    shrink5_mixed_atoms_v1,
    shrink5_rational_aggregate_leaves_v1,
)
from hegel_machine.strict_ast_shrink4_v1 import canonicalize_shrink4_source_ast
from hegel_machine.strict_ast_shrink5_v1 import (
    canonicalize_shrink5_source_ast,
    decode_shrink5_canonical_ast,
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
        / "src/hegel_machine/phase3_shrink5_capacity_entrypoint_v1.py"
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
    assert len(shrink5_constant_atoms_v1()) == 15
    assert len(shrink5_rational_aggregate_leaves_v1()) == 16
    assert len(shrink5_mixed_atoms_v1()) == 144
    sources = tuple(iter_shrink5_capacity_candidate_asts())
    assert len(sources) == EXPECTED_SHRINK5_SURVIVOR_SOURCE_COUNT == 175
    assert EXPECTED_SHRINK5_SOURCE_COUNT == 2_335

    parents = tuple(canonicalize_shrink4_source_ast(source) for source in sources)
    children = tuple(canonicalize_shrink5_source_ast(source) for source in sources)
    assert len({program.cbor_bytes for program in children}) == 175
    assert all(
        parent.cbor_bytes == child.cbor_bytes and parent.hash_id == child.hash_id
        for parent, child in zip(parents, children)
    )
    assert all(
        decode_shrink5_canonical_ast(child.cbor_bytes) == child
        for child in children
    )

    boundary = tuple(iter_shrink5_boundary_candidate_asts())
    assert len(boundary) == EXPECTED_SHRINK5_BOUNDARY_SOURCE_COUNT == 2_160
    parent_only = tuple(
        canonicalize_shrink4_source_ast(source) for source in boundary
    )
    assert len({program.cbor_bytes for program in parent_only}) == 2_160
    assert all(program.metrics.node_count == 7 for program in parent_only)


def test_direct_python_capacity_replay_is_target_free_and_nonterminal() -> None:
    report = _direct_replay("--capacity-replay")
    assert report["survivor_source_candidate_count"] == 175
    assert report["survivor_accepted_count"] == 175
    assert report["survivor_unique_count"] == 175
    assert report["survivor_parent_identity_match_count"] == 175
    assert report["survivor_accepted_set_commitment"] == (
        "sha256:f5ab7f079ad943d65a74881eb59c7bb46385e1c437ca8ab036bb071dfa3874ac"
    )
    assert report["parent_only_source_candidate_count"] == 2_160
    assert report["parent_only_parent_accepted_count"] == 2_160
    assert report["parent_only_source_child_rejected_count"] == 2_160
    assert report["parent_only_formal_child_rejected_count"] == 2_160
    assert report["parent_only_node_count"] == 7
    assert report["parent_only_set_commitment"] == (
        "sha256:7e0e8780149f03ce85723408f7e3eff2cd684e8938896125cf8e34be9ac70b5e"
    )
    assert report["subset_status"] == (
        "FULL_175_SURVIVOR_AND_2160_PARENT_NODE7_BOUNDARY_SETS_ONLY_NOT_COMPLETE"
    )
    assert report["executed_closure_status"] == "NOT_RUN"
    assert report["complete_closure_enumerated"] is False
    assert report["formal_roots"] is None
    assert report["target_or_split_modules_loaded"] is False
    assert report["maximum_ast_node_count"] == 6
    assert report["maximum_top_level_clauses"] == 2
    assert all(
        "target" not in name and "split" not in name
        for name in report["loaded_hegel_modules"]
    )


def test_direct_python_golden_replay_freezes_22_vectors() -> None:
    report = _direct_replay("--golden-replay")
    assert report["vector_count"] == report["passed_count"] == 22
    assert report["surviving_identity_checks"] == 3
    assert report["source_normalization_before_limit_checks"] == 2
    assert report["source_structural_limit_checks"] == 2
    assert report["source_priority_checks"] == 5
    assert report["formal_surviving_identity_checks"] == 1
    assert report["formal_structural_limit_checks"] == 1
    assert report["formal_priority_checks"] == 8
    assert report["execution_state"] == "NOT_RUN"
    assert report["formal_roots"] is None
    assert report["target_or_split_modules_loaded"] is False
    assert report["maximum_ast_node_count"] == 6


def test_bounded_enumerator_retains_difference_and_generates_only_and2() -> None:
    state = _Shrink5Enumerator(raw_cap=500_000)
    state.leaves(5)
    state.binary_and_ternary(depth=1, nodes=3, output_sort_id=5)
    root_operator_ids = {
        program.ast.root_operator_id
        for program in state.groups[("RationalValue", 1, 3)]
    }
    assert 0x0200 not in root_operator_ids
    assert 0x0201 in root_operator_ids

    conjunction_state = _Shrink5Enumerator(raw_cap=100_000)
    conjunction_state.leaves(1)
    raw_before = conjunction_state.raw_count
    conjunction_state.conjunctions(depth=1, nodes=3)
    generated = conjunction_state.groups.get(("Bool", 1, 3), ())
    assert generated
    assert conjunction_state.raw_count > raw_before
    assert all(
        program.ast.root_operator_id == 0x0400
        and len(program.expr.children) == 2
        for program in generated
    )

    result = enumerate_bounded_closure_shrink5_v1(
        _bindings(), canonical_budget=2_000, raw_application_cap=500_000
    )
    assert result.dsl_version == "hegel-old-dsl-v1.5.0"
    assert result.closure_status == "DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED"
    assert result.canonical_program_count == 2_000
    assert result.first_out_of_budget_program_hash is not None
    assert result.authoritative_claim_allowed is False
    assert DUAL_ENUMERATION_QUALIFIED is False
    report = child.diagnostic_report_shrink5_v1(
        result,
        _bindings(),
        canonical_budget=2_000,
        raw_application_cap=500_000,
    )
    assert report["maximum_top_level_clauses"] == 2
    assert report["maximum_ast_node_count"] == 6
    assert report["and3_generator_attempts_allowed"] is False
    assert report["and3_raw_operator_application_count"] == 0


def test_prefix_witness_records_and_accounting_are_deterministic() -> None:
    bindings = _bindings()
    first = enumerate_bounded_closure_shrink5_v1(
        bindings, canonical_budget=100, raw_application_cap=100_000
    )
    second = enumerate_bounded_closure_shrink5_v1(
        bindings, canonical_budget=100, raw_application_cap=100_000
    )

    assert first == second
    assert first.closure_status == "DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED"
    assert first.raw_operator_application_count > first.canonical_program_count
    assert first.first_out_of_budget_program_hash is not None
    assert first.first_out_of_budget_cbor is not None
    witness = decode_shrink5_canonical_ast(first.first_out_of_budget_cbor)
    assert witness.digest == first.first_out_of_budget_program_hash
    assert not _contains_binary_operator(witness.value, 0)

    records = first.canonical_program_records
    assert len(records) == first.canonical_program_count == 100
    assert tuple(record[3] for record in records) == tuple(range(100))
    for record in records:
        ast = decode_shrink5_canonical_ast(record[4])
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
    assert len(buckets) == 5 * 5 * 6 == 150
    assert tuple(bucket[3] for bucket in buckets) == tuple(range(150))
    assert sum(bucket[7] for bucket in buckets) == (
        first.raw_operator_application_count
    )
    assert sum(bucket[8] for bucket in buckets) == first.canonical_program_count
    assert first.bucket_accounting_root == child._base.rfc6962_root(
        list(buckets)
    )


def test_bounded_enumerator_raw_cap_fails_closed() -> None:
    with pytest.raises(BoundedEnumerationError) as raised:
        enumerate_bounded_closure_shrink5_v1(
            _bindings(), canonical_budget=2_000, raw_application_cap=1
        )
    assert raised.value.code == "INCONCLUSIVE_BUDGET"


def test_frozen_budget_upper_bounds_are_not_expandable() -> None:
    with pytest.raises(ValueError, match="canonical_budget"):
        enumerate_bounded_closure_shrink5_v1(
            _bindings(), canonical_budget=50_001
        )
    with pytest.raises(ValueError, match="raw_application_cap"):
        enumerate_bounded_closure_shrink5_v1(
            _bindings(), raw_application_cap=5_000_001
        )


def test_finite_frontier_complete_and_raw_cap_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OneProgramEnumerator(_Shrink5Enumerator):
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

    monkeypatch.setattr(child, "_Shrink5Enumerator", _OneProgramEnumerator)
    complete = enumerate_bounded_closure_shrink5_v1(
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

    monkeypatch.setattr(child, "_Shrink5Enumerator", _TwoProgramEnumerator)
    with pytest.raises(BoundedEnumerationError) as raised:
        enumerate_bounded_closure_shrink5_v1(
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
            "3340b3278caf562b560cc30cd14d3cd5f1d628e222b43d29d9d1e41b379f5675"
        ),
        "operator_semantics_root": (
            "5d2700884ae7125b9712a2bd06aa929feaf2fad1d4bfcd4fa5953c157a720ee1"
        ),
        "identifier_registry_root": (
            "1b0c141126b278778009d3ebbbf49f5de231ad0166a88a8a9caf367b35bff8ef"
        ),
        "canonical_ast_schema_root": (
            "828fdcc9f16ebd590702ff4297cac6f6ffa19b01299ea7a93753a4fced0961c5"
        ),
        "canonical_cbor_profile_root": (
            "0ccbd740c0b1f6a39fb8151ea56e114561093ee4fccb228bf83a9294e0bae783"
        ),
    }
    assert profile.STRICT_QUALIFICATION_SOURCE_COMMIT == (
        "320b0a3458901090cb738023a4398220fb1d9277"
    )
    assert profile.STRICT_QUALIFICATION_EVIDENCE_COMMIT == (
        "01b66cd8effeab258797998f594b250188d823da"
    )
    assert profile.STRICT_QUALIFICATION_STATUS == (
        "SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS"
    )
    assert profile.DUAL_COMPLETE_ENUMERATION_STATUS == "NOT_RUN"


def test_reduced_budget_report_cannot_masquerade_as_terminal() -> None:
    result = enumerate_bounded_closure_shrink5_v1(
        _bindings(), canonical_budget=100, raw_application_cap=100_000
    )
    report = child.diagnostic_report_shrink5_v1(
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
            "phase3_m3_shrink5_isolated_entrypoint_v1.py",
            "--target-free-self-check",
            20,
        ),
        (
            "phase3_m3_shrink5_dual_diagnostic_entrypoint_v1.py",
            "--host-self-check",
            21,
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
    assert report["strict_qualification_source_commit"] == (
        profile.STRICT_QUALIFICATION_SOURCE_COMMIT
    )
    assert report["strict_qualification_evidence_commit"] == (
        profile.STRICT_QUALIFICATION_EVIDENCE_COMMIT
    )
    assert report["strict_qualification_status"] == (
        "SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS"
    )
    assert report["maximum_top_level_clauses"] == 2
    assert report["maximum_ast_node_count"] == 6
    assert report["and3_generator_attempts_allowed"] is False
    assert report["and3_raw_operator_application_count"] == 0
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
