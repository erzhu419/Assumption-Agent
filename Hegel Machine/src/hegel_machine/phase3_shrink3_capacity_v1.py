"""Target-free survivor subset for Phase-3 shrink step 3.

Shrink step 3 removes only ``add``.  The preregistered shrink-2 constructive
subset contains no ``add`` node, so it is inherited byte-for-byte as a
survivor-identity qualification set.  Reusing these sources is intentional:
it qualifies that removing BinaryOperatorId 0 does not perturb any program
identity in this sampled survivor subset.  It is not a new closure sample and
has no cardinality or terminal-state authority.
"""

from __future__ import annotations

from typing import Final, Iterator

from .phase3_shrink2_capacity_v1 import (
    DiagnosticCandidateAst,
    iter_shrink2_capacity_candidate_asts,
    shrink2_constant_atoms_v1,
    shrink2_mixed_atoms_v1,
    shrink2_rational_aggregate_leaves_v1,
)


SHRINK3_CAPACITY_GENERATOR_SCHEMA: Final = (
    "hegel-phase3-shrink3-survivor-capacity-generator/1"
)
EXPECTED_SHRINK3_SOURCE_COUNT: Final = 2_160
SHRINK3_CAPACITY_GENERATOR_RULE: Final = (
    "inherit the exact 2160-source shrink-2 target-free constructive subset; "
    "the subset contains no BinaryOperatorId 0/add node; require every source "
    "to retain identical canonical AST CBOR bytes and hash under shrink step 3"
)


def shrink3_constant_atoms_v1() -> tuple[DiagnosticCandidateAst, ...]:
    """Return the 15 inherited constant comparison atoms."""

    return shrink2_constant_atoms_v1()


def shrink3_rational_aggregate_leaves_v1() -> tuple[DiagnosticCandidateAst, ...]:
    """Return the 16 inherited rational aggregate leaves."""

    return shrink2_rational_aggregate_leaves_v1()


def shrink3_mixed_atoms_v1() -> tuple[DiagnosticCandidateAst, ...]:
    """Return the 144 inherited mixed comparison atoms."""

    return shrink2_mixed_atoms_v1()


def iter_shrink3_capacity_candidate_asts() -> Iterator[DiagnosticCandidateAst]:
    """Yield the inherited survivor subset in its frozen source order."""

    yield from iter_shrink2_capacity_candidate_asts()


__all__ = [
    "DiagnosticCandidateAst",
    "EXPECTED_SHRINK3_SOURCE_COUNT",
    "SHRINK3_CAPACITY_GENERATOR_RULE",
    "SHRINK3_CAPACITY_GENERATOR_SCHEMA",
    "iter_shrink3_capacity_candidate_asts",
    "shrink3_constant_atoms_v1",
    "shrink3_mixed_atoms_v1",
    "shrink3_rational_aggregate_leaves_v1",
]
