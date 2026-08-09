from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hegel_machine import phase3_m3_bounded_enumerator_shrink6_v1 as child
from hegel_machine import phase3_m3_bounded_enumerator_shrink5_v1 as parent_enumerator
from hegel_machine import phase3_m3_shrink5_diagnostic_profile_v1 as parent_profile
from hegel_machine import phase3_m3_shrink6_diagnostic_profile_v1 as profile
from hegel_machine.phase3_m3_bounded_enumerator_shrink6_v1 import (
    BoundedEnumerationError,
    EnumerationBindingsV1,
    _Shrink6Enumerator,
    diagnostic_report_shrink6_v1,
    enumerate_bounded_closure_shrink6_v1,
)
from hegel_machine.phase3_shrink6_capacity_v1 import (
    EXPECTED_SHRINK6_CHALLENGE_SOURCE_COUNT,
    EXPECTED_SHRINK6_FULL_SURVIVOR_SOURCE_COUNT,
    EXPECTED_SHRINK6_FULL_SURVIVOR_UNIQUE_COUNT,
    EXPECTED_SHRINK6_NORMALIZED_SURVIVOR_SOURCE_COUNT,
    EXPECTED_SHRINK6_NORMALIZED_SURVIVOR_UNIQUE_COUNT,
    EXPECTED_SHRINK6_PARENT_CANONICAL_UNIQUE_COUNT,
    EXPECTED_SHRINK6_PARENT_ONLY_FAMILY_COUNTS,
    EXPECTED_SHRINK6_PARENT_ONLY_SOURCE_COUNT,
    SUBSET_STATUS,
    iter_shrink6_depth4_challenge_sources_v1,
)
from hegel_machine.strict_ast_shrink5_v1 import canonicalize_shrink5_source_ast
from hegel_machine.strict_ast_shrink6_v1 import decode_shrink6_canonical_ast


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bindings() -> EnumerationBindingsV1:
    return EnumerationBindingsV1(*profile.NON_FORMAL_SYNTHETIC_CHILD_BINDINGS)


def _capacity_replay() -> dict[str, object]:
    entrypoint = _PROJECT_ROOT / "src/hegel_machine/phase3_shrink6_capacity_entrypoint_v1.py"
    probe = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(entrypoint), "--capacity-replay"],
        cwd=_PROJECT_ROOT,
        env={"PATH": os.environ.get("PATH", "")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    return json.loads(probe.stdout)


def test_depth4_challenge_lattice_has_exact_partition() -> None:
    rows = tuple(iter_shrink6_depth4_challenge_sources_v1())
    assert len(rows) == EXPECTED_SHRINK6_CHALLENGE_SOURCE_COUNT == 1_266
    parents = tuple(canonicalize_shrink5_source_ast(row.source_ast) for row in rows)
    assert EXPECTED_SHRINK6_PARENT_CANONICAL_UNIQUE_COUNT == 1_249
    assert len({program.cbor_bytes for program in parents}) == 1_249
    normalized = [program for program in parents if program.metrics.depth <= 3]
    parent_only = [program for program in parents if program.metrics.depth == 4]
    assert EXPECTED_SHRINK6_NORMALIZED_SURVIVOR_SOURCE_COUNT == 67
    assert len(normalized) == 67
    assert EXPECTED_SHRINK6_NORMALIZED_SURVIVOR_UNIQUE_COUNT == 50
    assert len({program.cbor_bytes for program in normalized}) == 50
    assert EXPECTED_SHRINK6_PARENT_ONLY_SOURCE_COUNT == 1_199
    assert len(parent_only) == 1_199
    assert len({program.cbor_bytes for program in parent_only}) == 1_199
    assert all(program.metrics.node_count == 6 for program in parent_only)
    family_counts = {
        family: sum(
            row.family == family and parent.metrics.depth == 4
            for row, parent in zip(rows, parents)
        )
        for family in ("A", "B_abs", "B_sign")
    }
    assert family_counts == EXPECTED_SHRINK6_PARENT_ONLY_FAMILY_COUNTS


def test_direct_capacity_replay_freezes_all_counts_and_commitments() -> None:
    report = _capacity_replay()
    assert report["challenge_source_candidate_count"] == 1_266
    assert report["challenge_parent_accepted_count"] == 1_266
    assert report["challenge_parent_canonical_unique_count"] == 1_249
    assert report["challenge_source_family_counts"] == {
        "A": 486, "B_abs": 390, "B_sign": 390
    }
    assert report["normalized_survivor_source_count"] == 67
    assert report["normalized_survivor_unique_count"] == 50
    assert report["inherited_survivor_source_count"] == 175
    assert report["inherited_survivor_unique_count"] == 175
    assert EXPECTED_SHRINK6_FULL_SURVIVOR_SOURCE_COUNT == 242
    assert report["survivor_source_candidate_count"] == 242
    assert EXPECTED_SHRINK6_FULL_SURVIVOR_UNIQUE_COUNT == 225
    assert report["survivor_unique_count"] == 225
    assert report["parent_only_source_family_counts"] == {
        "A": 453, "B_abs": 373, "B_sign": 373
    }
    assert report["parent_only_depth"] == 4
    assert report["parent_only_node_count"] == 6
    assert report["challenge_source_lattice_commitment"] == (
        "sha256:a8cfb37278000933c2c51a2797e5bc0f4e7aad6970b37e178fc681f9358574d0"
    )
    assert report["challenge_parent_canonical_set_commitment"] == (
        "sha256:8f125763d3098d087dd7e9eb484b93097295ebd765b6f079795e8009623fb13e"
    )
    assert report["normalized_survivor_set_commitment"] == (
        "sha256:dcbb5562fc754fdef932188b189dbcdc0f7c500d3fc49651ee4dbb0f271afd29"
    )
    assert report["inherited_survivor_set_commitment"] == (
        "sha256:477a5abe659a7a7e7d2d50b2a5bda61b0dae1019c44fe84950c4a05036258619"
    )
    assert report["survivor_accepted_set_commitment"] == (
        "sha256:6787cd6c0782fda149e1ee93b37ca8d425f5ac78850c610e21cebf9da13a16d1"
    )
    assert report["parent_only_set_commitment"] == (
        "sha256:d3eb2b2d9caf1eece5a709d8113540e4709d579cdfbe3194f1cf176c9100b20d"
    )
    assert report["parent_only_source_rejection_outcome_commitment"] == (
        "sha256:9b0b766a4139db6297aea8b6032ad49147c1a26bf9b56291444a83681428cb0e"
    )
    assert report["parent_only_formal_rejection_outcome_commitment"] == (
        "sha256:97d50c34f51683a2502157961acc79d3b4e108b28bdaa266cf3721ffda8b3a96"
    )
    assert report["subset_status"] == SUBSET_STATUS
    assert report["executed_closure_status"] == "NOT_RUN"
    assert report["complete_closure_enumerated"] is False
    assert report["formal_roots"] is None
    assert report["target_or_split_modules_loaded"] is False


def test_depth_three_enumerator_lattice_is_120_buckets_and_and2_only() -> None:
    state = _Shrink6Enumerator(raw_cap=100_000)
    assert len(state.buckets) == 5 * 4 * 6 == 120
    assert max(key[1] for key in state.buckets) == 3
    assert max(key[2] for key in state.buckets) == 6
    state.leaves(1)
    before = state.raw_count
    state.conjunctions(depth=1, nodes=3)
    generated = state.groups[("Bool", 1, 3)]
    assert state.raw_count - before == len(generated) == 15
    assert all(
        program.expr.tag == 4
        and len(program.expr.children) == 2
        and program.ast.metrics.top_level_clause_count == 2
        for program in generated
    )


def test_bounded_prefix_is_deterministic_nonformal_and_depth_limited() -> None:
    bindings = _bindings()
    first = enumerate_bounded_closure_shrink6_v1(
        bindings, canonical_budget=100, raw_application_cap=100_000
    )
    second = enumerate_bounded_closure_shrink6_v1(
        bindings, canonical_budget=100, raw_application_cap=100_000
    )
    assert first == second
    assert first.closure_status == "DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED"
    assert first.authoritative_claim_allowed is False
    assert len(first.bucket_accounting_records) == 120
    assert first.first_out_of_budget_cbor is not None
    witness = decode_shrink6_canonical_ast(first.first_out_of_budget_cbor)
    assert witness.metrics.depth <= 3
    assert all(
        decode_shrink6_canonical_ast(record[4]).metrics.depth <= 3
        for record in first.canonical_program_records
    )
    report = diagnostic_report_shrink6_v1(
        first,
        bindings,
        canonical_budget=100,
        raw_application_cap=100_000,
    )
    assert report["maximum_ast_depth"] == 3
    assert report["formal_bucket_count"] == 120
    assert report["formal_roots"] is None
    assert report["execution_state"] == "NOT_RUN"
    assert {
        "prefix_preservation_expectation_id",
        "prefix_preservation_verified",
    }.isdisjoint(report)

    current_roots = profile.diagnostic_root_hex_v1()
    previous_roots = parent_profile.diagnostic_root_hex_v1()
    assert set(current_roots) == set(previous_roots)
    assert all(
        current_roots[name] != previous_roots[name] for name in current_roots
    )
    for record in first.canonical_program_records:
        assert tuple(root.hex() for root in record[11:14]) == (
            current_roots["child_dsl_spec_root"],
            current_roots["operator_semantics_root"],
            current_roots["identifier_registry_root"],
        )

    parent_bindings = parent_enumerator.EnumerationBindingsV1(
        *parent_profile.NON_FORMAL_SYNTHETIC_CHILD_BINDINGS
    )
    parent = parent_enumerator.enumerate_bounded_closure_shrink5_v1(
        parent_bindings,
        canonical_budget=100,
        raw_application_cap=100_000,
    )
    assert tuple(record[4] for record in first.canonical_program_records) == tuple(
        record[4] for record in parent.canonical_program_records
    )
    assert (
        first.canonical_program_archive_root
        != parent.canonical_program_archive_root
    )
    assert first.program_chunk_manifest_root != parent.program_chunk_manifest_root
    assert first.bucket_accounting_root != parent.bucket_accounting_root


def test_preservation_values_are_preregistered_expectations_not_observations() -> None:
    assert profile.PREFIX_PRESERVATION_EXPECTATION_ID == (
        "SHRINK6_PRESERVE_SHRINK5_PREFIX_THROUGH_CLOSED_BOUNDARY_BUCKET_V1"
    )
    assert profile.PREFIX_PRESERVATION_EXPECTATION_STATUS == (
        "PREREGISTERED_NOT_OBSERVED"
    )
    assert profile.EXPECTED_CANONICAL_PROGRAM_COUNT == 50_000
    assert profile.EXPECTED_FIRST_OUT_OF_BUDGET_ORDINAL == 50_001
    assert profile.EXPECTED_FIRST_OUT_OF_BUDGET_PROGRAM_CBOR_HEX == (
        "820183010384020183000001860003050200818203f5"
    )
    assert profile.EXPECTED_FIRST_OUT_OF_BUDGET_PROGRAM_HASH == (
        "sha256:31320fc9f8926792aaf1416a4963df46a2300d87db8096f42e574a62272a68ee"
    )
    assert profile.EXPECTED_RAW_OPERATOR_APPLICATION_COUNT == 3_120_719
    assert profile.EXPECTED_RESIDUAL_OUT_OF_BUDGET_CANONICAL_PROGRAMS == 2_237
    assert (
        profile.EXPECTED_WITNESS_BUCKET_INDEX,
        profile.EXPECTED_WITNESS_OUTPUT_SORT_ID,
        profile.EXPECTED_WITNESS_AST_DEPTH,
        profile.EXPECTED_WITNESS_AST_NODE_COUNT,
    ) == (63, 3, 2, 4)
    assert child.DIAGNOSTIC_EXECUTION_STATE == "NOT_RUN"
    assert child.DUAL_ENUMERATION_QUALIFIED is False


def test_finite_frontier_complete_and_raw_cap_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OneProgramEnumerator(_Shrink6Enumerator):
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

    monkeypatch.setattr(child, "_Shrink6Enumerator", _OneProgramEnumerator)
    complete = enumerate_bounded_closure_shrink6_v1(
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

    monkeypatch.setattr(child, "_Shrink6Enumerator", _TwoProgramEnumerator)
    with pytest.raises(BoundedEnumerationError) as raised:
        enumerate_bounded_closure_shrink6_v1(
            _bindings(), canonical_budget=10, raw_application_cap=1
        )
    assert raised.value.code == "INCONCLUSIVE_BUDGET"


def test_reduced_budget_report_cannot_masquerade_as_terminal() -> None:
    result = enumerate_bounded_closure_shrink6_v1(
        _bindings(), canonical_budget=100, raw_application_cap=100_000
    )
    report = diagnostic_report_shrink6_v1(
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
            "phase3_m3_shrink6_isolated_entrypoint_v1.py",
            "--target-free-self-check",
            23,
        ),
        (
            "phase3_m3_shrink6_dual_diagnostic_entrypoint_v1.py",
            "--host-self-check",
            24,
        ),
    ],
)
def test_direct_entrypoints_have_exact_target_free_module_closures(
    entrypoint_name: str,
    mode: str,
    expected_module_count: int,
) -> None:
    entrypoint = _PROJECT_ROOT / "src/hegel_machine" / entrypoint_name
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(entrypoint), mode],
        cwd=_PROJECT_ROOT,
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
    assert report["maximum_ast_depth"] == 3
    assert report["maximum_ast_node_count"] == 6
    assert report["maximum_top_level_clauses"] == 2
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


def test_raw_cap_and_budget_expansion_fail_closed() -> None:
    with pytest.raises(BoundedEnumerationError) as raised:
        enumerate_bounded_closure_shrink6_v1(
            _bindings(), canonical_budget=2_000, raw_application_cap=1
        )
    assert raised.value.code == "INCONCLUSIVE_BUDGET"
    with pytest.raises(ValueError, match="canonical_budget"):
        enumerate_bounded_closure_shrink6_v1(_bindings(), canonical_budget=50_001)
    with pytest.raises(ValueError, match="raw_application_cap"):
        enumerate_bounded_closure_shrink6_v1(
            _bindings(), raw_application_cap=5_000_001
        )
