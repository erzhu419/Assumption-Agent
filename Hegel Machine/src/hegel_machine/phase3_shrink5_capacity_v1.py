"""Target-free survivor and seven-node boundary sets for shrink step 5.

The 175 inherited atoms exhaust the parent capacity generator's leaf pools and
must retain exact identity.  The 2,160 inherited AND2 programs are all
seven-node parent programs and must be rejected as ``REJECT_STRUCTURAL_LIMIT``
at both child boundaries.  Neither set is a complete child closure execution.
"""

from __future__ import annotations

from typing import Final, Iterator

from .phase3_shrink4_capacity_v1 import (
    DiagnosticCandidateAst,
    iter_shrink4_capacity_candidate_asts,
    shrink4_constant_atoms_v1,
    shrink4_mixed_atoms_v1,
    shrink4_rational_aggregate_leaves_v1,
)


SHRINK5_CAPACITY_GENERATOR_SCHEMA: Final = (
    "hegel-phase3-shrink5-seven-node-boundary-capacity-generator/1"
)
EXPECTED_SHRINK5_SURVIVOR_SOURCE_COUNT: Final = 175
EXPECTED_SHRINK5_BOUNDARY_SOURCE_COUNT: Final = 2_160
EXPECTED_SHRINK5_SOURCE_COUNT: Final = (
    EXPECTED_SHRINK5_SURVIVOR_SOURCE_COUNT + EXPECTED_SHRINK5_BOUNDARY_SOURCE_COUNT
)
SHRINK5_CAPACITY_GENERATOR_RULE: Final = (
    "replay the exact 175 inherited target-free atom survivors and require "
    "identical parent/child canonical bytes, hashes, and MDL lengths; also "
    "replay the exact 2160-source shrink-4 AND2 capacity set, require every "
    "parent canonical AST to contain exactly seven nodes, and require "
    "REJECT_STRUCTURAL_LIMIT at both shrink-5 source and formal boundaries"
)


def shrink5_constant_atoms_v1() -> tuple[DiagnosticCandidateAst, ...]:
    return shrink4_constant_atoms_v1()


def shrink5_rational_aggregate_leaves_v1() -> tuple[DiagnosticCandidateAst, ...]:
    return shrink4_rational_aggregate_leaves_v1()


def shrink5_mixed_atoms_v1() -> tuple[DiagnosticCandidateAst, ...]:
    return shrink4_mixed_atoms_v1()


def iter_shrink5_capacity_candidate_asts() -> Iterator[DiagnosticCandidateAst]:
    yield from shrink5_constant_atoms_v1()
    yield from shrink5_rational_aggregate_leaves_v1()
    yield from shrink5_mixed_atoms_v1()


def iter_shrink5_boundary_candidate_asts() -> Iterator[DiagnosticCandidateAst]:
    yield from iter_shrink4_capacity_candidate_asts()


__all__ = [
    "DiagnosticCandidateAst",
    "EXPECTED_SHRINK5_BOUNDARY_SOURCE_COUNT",
    "EXPECTED_SHRINK5_SOURCE_COUNT",
    "EXPECTED_SHRINK5_SURVIVOR_SOURCE_COUNT",
    "SHRINK5_CAPACITY_GENERATOR_RULE",
    "SHRINK5_CAPACITY_GENERATOR_SCHEMA",
    "iter_shrink5_boundary_candidate_asts",
    "iter_shrink5_capacity_candidate_asts",
    "shrink5_constant_atoms_v1",
    "shrink5_mixed_atoms_v1",
    "shrink5_rational_aggregate_leaves_v1",
]
