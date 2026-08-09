from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from hegel_machine import phase3_m3_bounded_enumerator_shrink2_v1 as child
from hegel_machine import phase3_m3_shrink2_diagnostic_profile_v1 as profile
from hegel_machine.phase3_m3_shrink2_core_v1 import (
    ACTIVE_RATIONAL_PARAMETER_IDS,
    TOMBSTONED_RATIONAL_PARAMETER_IDS,
)
from hegel_machine.strict_ast_shrink2_v1 import decode_shrink2_canonical_ast


def _bindings() -> child.EnumerationBindingsV1:
    return child.EnumerationBindingsV1(
        *profile.NON_FORMAL_SYNTHETIC_CHILD_BINDINGS
    )


def test_shrink2_leaf_generator_preserves_sparse_parameter_ids() -> None:
    state = child._Shrink2Enumerator(raw_cap=10_000)
    state.leaves(5)
    rational = state.groups[("RationalValue", 0, 1)]
    constant_ids = tuple(
        program.expr.parameters[0]
        for program in rational
        if program.expr.operator_id == 0
    )
    assert constant_ids == ACTIVE_RATIONAL_PARAMETER_IDS == (1, 3, 5)
    assert not set(constant_ids).intersection(TOMBSTONED_RATIONAL_PARAMETER_IDS)
    assert len(rational) == 3 + 2 * 4 * 2 * 33
    assert all(
        decode_shrink2_canonical_ast(program.ast.cbor_bytes) == program.ast
        for program in rational
    )


def test_diagnostic_prefix_is_bound_but_never_authoritative() -> None:
    first = child.enumerate_bounded_closure_shrink2_v1(
        _bindings(), canonical_budget=100, raw_application_cap=100_000
    )
    second = child.enumerate_bounded_closure_shrink2_v1(
        _bindings(), canonical_budget=100, raw_application_cap=100_000
    )

    assert first == second
    assert first.dsl_version == "hegel-old-dsl-v1.2.0"
    assert first.closure_status == "DSL_TOO_LARGE"
    assert first.canonical_program_count == 100
    assert first.first_out_of_budget_program_hash is not None
    assert first.first_out_of_budget_cbor is not None
    assert first.traversal_prefix_complete is True
    assert first.authoritative_claim_allowed is False
    assert len(first.canonical_program_records) == 100
    assert len(first.program_chunk_manifests) == 1
    assert len(first.bucket_accounting_records) == 175


def test_inactive_fold_results_remain_operator_programs_in_enumeration() -> None:
    state = child._Shrink2Enumerator(raw_cap=100)
    expression = child._parent._strict._Expr(
        2,
        0,
        "RationalValue",
        (
            child._parent._strict._Expr(
                0, 0, "RationalValue", parameters=(5,)
            ),
            child._parent._strict._Expr(
                0, 0, "RationalValue", parameters=(5,)
            ),
        ),
    )
    state._admit(expression, (5, 1, 3))
    programs = state.groups[("RationalValue", 1, 3)]
    assert len(programs) == 1
    assert programs[0].expr.tag == 2
    assert programs[0].expr.operator_id == 0
    assert decode_shrink2_canonical_ast(programs[0].ast.cbor_bytes) == programs[0].ast


def test_frozen_budgets_cannot_be_expanded() -> None:
    with pytest.raises(ValueError, match="canonical_budget"):
        child.enumerate_bounded_closure_shrink2_v1(
            _bindings(), canonical_budget=50_001
        )
    with pytest.raises(ValueError, match="raw_application_cap"):
        child.enumerate_bounded_closure_shrink2_v1(
            _bindings(), raw_application_cap=5_000_001
        )


def test_closed_frontier_emits_complete_without_a_witness(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FiniteEnumerator(child._Shrink2Enumerator):
        def leaves(self, output_sort_id: int) -> None:
            if output_sort_id == 1:
                self._admit(
                    child._parent._strict._Expr(
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

    monkeypatch.setattr(child, "_Shrink2Enumerator", _FiniteEnumerator)
    result = child.enumerate_bounded_closure_shrink2_v1(
        _bindings(), canonical_budget=10, raw_application_cap=10
    )

    assert result.closure_status == "COMPLETE"
    assert result.canonical_program_count == 1
    assert result.raw_operator_application_count == 1
    assert result.first_out_of_budget_program_hash is None
    assert result.first_out_of_budget_cbor is None
    assert result.traversal_prefix_complete is True
    assert result.authoritative_claim_allowed is False


def test_direct_self_check_has_an_exact_target_free_dependency_closure() -> None:
    entrypoint = (
        Path(child.__file__).resolve().parent
        / "phase3_m3_shrink2_isolated_entrypoint_v1.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(entrypoint),
            "--target-free-self-check",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["claim_level"] == "NON_FORMAL_DUAL_CHILD_DIAGNOSTIC"
    assert report["diagnostic_only"] is True
    assert report["authoritative_claim_allowed"] is False
    assert report["execution_state"] == "NOT_RUN"
    assert report["formal_roots_generated"] is False
    assert report["formal_roots"] is None
    assert report["complete_closure_enumerated"] is False
    assert report["target_or_split_modules_loaded"] is False
    assert len(report["loaded_hegel_modules"]) == 11
    assert all(
        fragment not in module
        for module in report["loaded_hegel_modules"]
        for fragment in ("_target", "_split_", "_seed", "_role", "_evaluator")
    )
    assert report["child_dsl_spec_root"] == (
        profile.NON_FORMAL_SYNTHETIC_CHILD_BINDINGS[0].hex()
    )


def test_direct_entrypoint_removes_all_legacy_report_aliases() -> None:
    entrypoint = (
        Path(child.__file__).resolve().parent
        / "phase3_m3_shrink2_isolated_entrypoint_v1.py"
    )
    script = f"""
import runpy
namespace = runpy.run_path({str(entrypoint)!r}, run_name='shrink2_schema_probe')
root = '00' * 32
report = {{
    'closure_status': 'DSL_TOO_LARGE',
    'canonical_program_archive_root': root,
    'program_chunk_manifest_root': root,
    'bucket_accounting_root': root,
    'first_out_of_budget_program_hash_or_null': root,
    'first_out_of_budget_program_cbor_hex_or_null': '00',
    'first_out_of_budget_ordinal_or_null': 50001,
    'canonical_program_budget': 50000,
    'raw_operator_application_cap': 5000000,
    'diagnostic_child_dsl_spec_root': root,
    'diagnostic_operator_semantics_root': root,
    'diagnostic_identifier_registry_root': root,
}}
result = namespace['_augment_report'](
    report, tuple(namespace['_EXPECTED_PROJECT_MODULES'])
)
legacy = {{
    'bucket_accounting_root', 'canonical_program_archive_root',
    'canonical_program_budget', 'diagnostic_child_dsl_spec_root',
    'diagnostic_identifier_registry_root',
    'diagnostic_operator_semantics_root',
    'first_out_of_budget_ordinal_or_null', 'program_chunk_manifest_root',
    'raw_operator_application_cap',
}}
assert not (legacy & set(result))
assert result['target_free_isolation_verified'] is True
assert result['target_or_split_modules_loaded'] is False
assert len(result['loaded_hegel_modules']) == 11
print('PASS')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "PASS"


def test_direct_host_self_check_has_no_package_initializer_contamination() -> None:
    entrypoint = (
        Path(child.__file__).resolve().parent
        / "phase3_m3_shrink2_dual_diagnostic_entrypoint_v1.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(entrypoint),
            "--host-self-check",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["diagnostic_only"] is True
    assert report["execution_state"] == "NOT_RUN"
    assert report["formal_roots"] is None
    assert report["dual_replay_executed"] is False
    assert report["target_free_isolation_verified"] is True
    assert report["target_or_split_modules_loaded"] is False
    assert len(report["loaded_hegel_modules"]) == 12
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


def test_non_formal_binding_roots_are_exact_domain_hashes() -> None:
    assert profile.diagnostic_root_hex_v1() == {
        "child_dsl_spec_root": (
            "281c2f8adc41fdc467613b88c3c0caf3648efa883186bac61aabc2f8b575b3be"
        ),
        "operator_semantics_root": (
            "7e09babc9dd91cfb8cab305f623957dab3f181cc70dacac9035cacf6d019d4bd"
        ),
        "identifier_registry_root": (
            "e6620b5f29151dda2a552425d19f53d5378d91641320d280701871eb5639e699"
        ),
        "canonical_ast_schema_root": (
            "892bd6c958dd0dc300d30c13b5c7a2eaeedd36a0853a775097cc99aa3c2b544e"
        ),
        "canonical_cbor_profile_root": (
            "ab03d374143db9e64520a6f055439108e59ad13912544026789bb0d9558ebd32"
        ),
    }
