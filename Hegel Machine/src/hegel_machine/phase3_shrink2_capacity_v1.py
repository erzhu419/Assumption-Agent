"""Pure 2,160-source constructive subset for shrink step 2.

The generator contains only target-independent source ASTs.  It preserves the
parent numeric RationalParameter IDs 1, 3, and 5 and the shrink-1 rational
aggregate IDs 0 and 5.  It carries no target, split, seed, key, evaluator, or
publication identity.
"""

from __future__ import annotations

from itertools import combinations_with_replacement, product
from typing import Final, Iterator

from .hashing import canonical_json, stable_hash
from .phase3_m3_dsl_core_v1 import (
    QUANTITY_IDS,
    RATIONAL_PARAMETER_GRID,
    SCOPE_IDS,
    RationalAtom,
)
from .phase3_m3_shrink2_core_v1 import (
    ACTIVE_RATIONAL_PARAMETER_IDS,
    RATIONAL_ACTIVE_AGGREGATE_NAMES,
)


SHRINK2_CAPACITY_GENERATOR_SCHEMA: Final = (
    "hegel-phase3-shrink2-capacity-generator/1"
)
EXPECTED_SHRINK2_SOURCE_COUNT: Final = 2_160
SHRINK2_CAPACITY_GENERATOR_RULE: Final = (
    "15 active-constant comparison atoms x 144 active-constant/aggregate "
    "comparison atoms -> canonical top_level_AND Cartesian product; "
    "RationalParameterId/v1 active IDs are 1,3,5; rational AggregateMapId/v1 "
    "active IDs are 0,5; expected source count=2160"
)

DiagnosticCandidateAst = tuple[object, ...]


def _diagnostic_hash(ast: DiagnosticCandidateAst) -> str:
    return stable_hash(ast)


def _commutative_children(
    left: DiagnosticCandidateAst,
    right: DiagnosticCandidateAst,
) -> tuple[DiagnosticCandidateAst, DiagnosticCandidateAst]:
    return tuple(  # type: ignore[return-value]
        sorted(
            (left, right),
            key=lambda child: (_diagnostic_hash(child), canonical_json(child)),
        )
    )


def _active_parameters() -> tuple[RationalAtom, ...]:
    result = tuple(RATIONAL_PARAMETER_GRID[index] for index in ACTIVE_RATIONAL_PARAMETER_IDS)
    if tuple((item.numerator, item.denominator) for item in result) != (
        (-1, 1),
        (0, 1),
        (1, 1),
    ):
        raise AssertionError("shrink-2 active RationalParameter identity drift")
    return result


def _constant(parameter: RationalAtom) -> DiagnosticCandidateAst:
    return ("scalar_const", parameter.numerator, parameter.denominator)


def _aggregate(map_name: str, scope: str, quantity: str) -> DiagnosticCandidateAst:
    return ("aggregate", map_name, scope, quantity, ())


def _equal(
    left: DiagnosticCandidateAst, right: DiagnosticCandidateAst
) -> DiagnosticCandidateAst:
    first, second = _commutative_children(left, right)
    return ("equal_exact", first, second)


def _less_equal(
    left: DiagnosticCandidateAst, right: DiagnosticCandidateAst
) -> DiagnosticCandidateAst:
    return ("less_equal", left, right)


def _and(
    left: DiagnosticCandidateAst, right: DiagnosticCandidateAst
) -> DiagnosticCandidateAst:
    first, second = _commutative_children(left, right)
    return ("top_level_AND", first, second)


def shrink2_constant_atoms_v1() -> tuple[DiagnosticCandidateAst, ...]:
    constants = tuple(_constant(item) for item in _active_parameters())
    equal = tuple(
        _equal(left, right)
        for left, right in combinations_with_replacement(constants, 2)
    )
    ordered = tuple(
        _less_equal(left, right) for left, right in product(constants, repeat=2)
    )
    result = equal + ordered
    if len(result) != 15:
        raise AssertionError("shrink-2 constant atom count drift")
    return result


def shrink2_rational_aggregate_leaves_v1() -> tuple[DiagnosticCandidateAst, ...]:
    result = tuple(
        _aggregate(map_name, scope, quantity)
        for map_name, scope, quantity in product(
            RATIONAL_ACTIVE_AGGREGATE_NAMES,
            SCOPE_IDS,
            QUANTITY_IDS,
        )
    )
    if len(result) != 16:
        raise AssertionError("shrink-2 rational aggregate leaf count drift")
    return result


def shrink2_mixed_atoms_v1() -> tuple[DiagnosticCandidateAst, ...]:
    constants = tuple(_constant(item) for item in _active_parameters())
    aggregates = shrink2_rational_aggregate_leaves_v1()
    equal = tuple(
        _equal(constant, aggregate)
        for constant, aggregate in product(constants, aggregates)
    )
    ordered = tuple(
        atom
        for constant, aggregate in product(constants, aggregates)
        for atom in (
            _less_equal(constant, aggregate),
            _less_equal(aggregate, constant),
        )
    )
    result = equal + ordered
    if len(result) != 144:
        raise AssertionError("shrink-2 mixed atom count drift")
    return result


def iter_shrink2_capacity_candidate_asts() -> Iterator[DiagnosticCandidateAst]:
    """Yield the preregistered child subset in deterministic source order."""

    for constant_atom, mixed_atom in product(
        shrink2_constant_atoms_v1(), shrink2_mixed_atoms_v1()
    ):
        yield _and(constant_atom, mixed_atom)


__all__ = [
    "DiagnosticCandidateAst",
    "EXPECTED_SHRINK2_SOURCE_COUNT",
    "SHRINK2_CAPACITY_GENERATOR_RULE",
    "SHRINK2_CAPACITY_GENERATOR_SCHEMA",
    "iter_shrink2_capacity_candidate_asts",
    "shrink2_constant_atoms_v1",
    "shrink2_mixed_atoms_v1",
    "shrink2_rational_aggregate_leaves_v1",
]
