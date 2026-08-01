"""Pure generator for the frozen 64,680-program capacity witness.

This module deliberately contains no evidence-report IDs.  Both the legacy
constructive preflight and the strict Python/Rust comparison import this same
generator, so the executed source can be included in a capacity-report source
root without creating a report-ID self-reference.
"""

from __future__ import annotations

from itertools import combinations_with_replacement, product
from typing import Final, Iterator

from .hashing import canonical_json, stable_hash
from .phase3_dsl_v1 import (
    AGGREGATE_CATALOG,
    QUANTITY_IDS,
    RATIONAL_PARAMETER_GRID,
    SCOPE_IDS,
    RationalAtom,
)


CAPACITY_GENERATOR_SCHEMA: Final = "hegel-phase3-capacity-witness-generator/1"
EXPECTED_CAPACITY_SOURCE_COUNT: Final = 64_680
CAPACITY_GENERATOR_RULE: Final = (
    "cartesian_product("
    "constant_atoms(equal_exact combinations_with_replacement of 7 rational "
    "constants + less_equal ordered product of those constants),"
    "one_aggregate_atoms(equal_exact constant x 40 RationalValue aggregate "
    "leaves + both less_equal directions),"
    "wrapped by canonical top_level_AND); expected_count=64680"
)

DiagnosticCandidateAst = tuple[object, ...]


def _diagnostic_hash(ast: DiagnosticCandidateAst) -> str:
    return stable_hash(ast)


def canonical_commutative_children(
    left: DiagnosticCandidateAst,
    right: DiagnosticCandidateAst,
) -> tuple[DiagnosticCandidateAst, DiagnosticCandidateAst]:
    """Apply the frozen diagnostic child order with a deterministic tie break."""

    return tuple(  # type: ignore[return-value]
        sorted(
            (left, right),
            key=lambda child: (_diagnostic_hash(child), canonical_json(child)),
        )
    )


def _scalar_const(parameter: RationalAtom) -> DiagnosticCandidateAst:
    return ("scalar_const", parameter.numerator, parameter.denominator)


def _aggregate(
    aggregate_map_id: str,
    scope_id: str,
    quantity_id: str,
) -> DiagnosticCandidateAst:
    # The proof uses the zero-extension scope form, so it consumes no context
    # clauses while still binding every frozen aggregate identifier.
    return (
        "aggregate",
        aggregate_map_id,
        scope_id,
        quantity_id,
        (),
    )


def _equal_exact(
    left: DiagnosticCandidateAst,
    right: DiagnosticCandidateAst,
) -> DiagnosticCandidateAst:
    first, second = canonical_commutative_children(left, right)
    return ("equal_exact", first, second)


def _less_equal(
    left: DiagnosticCandidateAst,
    right: DiagnosticCandidateAst,
) -> DiagnosticCandidateAst:
    return ("less_equal", left, right)


def _top_level_and(
    left: DiagnosticCandidateAst,
    right: DiagnosticCandidateAst,
) -> DiagnosticCandidateAst:
    first, second = canonical_commutative_children(left, right)
    return ("top_level_AND", first, second)


def _constant_leaves() -> tuple[DiagnosticCandidateAst, ...]:
    return tuple(_scalar_const(parameter) for parameter in RATIONAL_PARAMETER_GRID)


def _rational_aggregate_leaves() -> tuple[DiagnosticCandidateAst, ...]:
    rational_maps = tuple(
        spec.map_id
        for spec in AGGREGATE_CATALOG
        if spec.output_sort == "RationalValue"
    )
    return tuple(
        _aggregate(map_id, scope_id, quantity_id)
        for map_id, scope_id, quantity_id in product(
            rational_maps,
            SCOPE_IDS,
            QUANTITY_IDS,
        )
    )


def _constant_only_atoms() -> tuple[DiagnosticCandidateAst, ...]:
    constants = _constant_leaves()
    equal_atoms = tuple(
        _equal_exact(left, right)
        for left, right in combinations_with_replacement(constants, 2)
    )
    ordered_atoms = tuple(
        _less_equal(left, right) for left, right in product(constants, repeat=2)
    )
    return equal_atoms + ordered_atoms


def _one_aggregate_atoms() -> tuple[DiagnosticCandidateAst, ...]:
    constants = _constant_leaves()
    aggregates = _rational_aggregate_leaves()
    equal_atoms = tuple(
        _equal_exact(constant, aggregate)
        for constant, aggregate in product(constants, aggregates)
    )
    ordered_atoms = tuple(
        atom
        for constant, aggregate in product(constants, aggregates)
        for atom in (
            _less_equal(constant, aggregate),
            _less_equal(aggregate, constant),
        )
    )
    return equal_atoms + ordered_atoms


def iter_capacity_witness_candidate_asts() -> Iterator[DiagnosticCandidateAst]:
    """Yield the exact conservative subset used by the M2 capacity replay."""

    constant_atoms = _constant_only_atoms()
    aggregate_atoms = _one_aggregate_atoms()
    for constant_atom, aggregate_atom in product(
        constant_atoms,
        aggregate_atoms,
    ):
        yield _top_level_and(constant_atom, aggregate_atom)


__all__ = [
    "CAPACITY_GENERATOR_RULE",
    "CAPACITY_GENERATOR_SCHEMA",
    "DiagnosticCandidateAst",
    "EXPECTED_CAPACITY_SOURCE_COUNT",
    "canonical_commutative_children",
    "iter_capacity_witness_candidate_asts",
]
