"""Target-free 2,160-program AND2 survivor set for shrink step 4.

Every inherited shrink-3 capacity source is already a normalized two-clause
top-level conjunction.  Replaying the complete set therefore qualifies the
sole shrink-4 survivor obligation without sampling: all 2,160 parent programs
must retain identical canonical AST CBOR bytes and hashes in the child.
The set is still a constructive subset, not a complete closure execution.
"""

from __future__ import annotations

from typing import Final, Iterator

from .phase3_shrink3_capacity_v1 import (
    DiagnosticCandidateAst,
    iter_shrink3_capacity_candidate_asts,
    shrink3_constant_atoms_v1,
    shrink3_mixed_atoms_v1,
    shrink3_rational_aggregate_leaves_v1,
)


SHRINK4_CAPACITY_GENERATOR_SCHEMA: Final = (
    "hegel-phase3-shrink4-and2-survivor-capacity-generator/1"
)
EXPECTED_SHRINK4_SOURCE_COUNT: Final = 2_160
SHRINK4_CAPACITY_GENERATOR_RULE: Final = (
    "inherit the exact 2160-source shrink-3 target-free constructive subset; "
    "every source is a normalized top_level_AND with exactly two distinct "
    "clauses; require identical canonical AST CBOR bytes, hashes, and MDL "
    "lengths across shrink step 4"
)


def shrink4_constant_atoms_v1() -> tuple[DiagnosticCandidateAst, ...]:
    return shrink3_constant_atoms_v1()


def shrink4_rational_aggregate_leaves_v1() -> tuple[DiagnosticCandidateAst, ...]:
    return shrink3_rational_aggregate_leaves_v1()


def shrink4_mixed_atoms_v1() -> tuple[DiagnosticCandidateAst, ...]:
    return shrink3_mixed_atoms_v1()


def iter_shrink4_capacity_candidate_asts() -> Iterator[DiagnosticCandidateAst]:
    yield from iter_shrink3_capacity_candidate_asts()


__all__ = [
    "DiagnosticCandidateAst",
    "EXPECTED_SHRINK4_SOURCE_COUNT",
    "SHRINK4_CAPACITY_GENERATOR_RULE",
    "SHRINK4_CAPACITY_GENERATOR_SCHEMA",
    "iter_shrink4_capacity_candidate_asts",
    "shrink4_constant_atoms_v1",
    "shrink4_mixed_atoms_v1",
    "shrink4_rational_aggregate_leaves_v1",
]
