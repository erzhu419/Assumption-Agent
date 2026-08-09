from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hegel_machine import phase3_m3_bounded_enumerator_shrink6_v1 as child
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
    return EnumerationBindingsV1(bytes(32), bytes((1,)) * 32, bytes((2,)) * 32)


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
