"""Pure 25,872-source constructive subset for shrink step 1.

This source module contains no replay artifact IDs.  It defines only the
preregistered source construction, allowing its exact bytes to participate in
the execution source commitment without a report self-reference.
"""

from __future__ import annotations

from itertools import combinations_with_replacement, product
from typing import Final, Iterator

from .hashing import canonical_json, stable_hash
from .phase3_dsl_v1 import QUANTITY_IDS, RATIONAL_PARAMETER_GRID, SCOPE_IDS, RationalAtom
from .phase3_shrink1_registry_v1 import RATIONAL_ACTIVE_AGGREGATE_NAMES


SHRINK1_CAPACITY_GENERATOR_SCHEMA: Final = "hegel-phase3-shrink1-capacity-generator/1"
EXPECTED_SHRINK1_SOURCE_COUNT: Final = 25_872
SHRINK1_CAPACITY_GENERATOR_RULE: Final = (
    "77 constant comparison atoms x 336 one-aggregate comparison atoms -> "
    "canonical top_level_AND Cartesian product; rational active aggregate "
    "maps are sparse IDs 0 and 5 only; expected source count=25872"
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


def _constant(parameter: RationalAtom) -> DiagnosticCandidateAst:
    return ("scalar_const", parameter.numerator, parameter.denominator)


def _aggregate(map_name: str, scope: str, quantity: str) -> DiagnosticCandidateAst:
    return ("aggregate", map_name, scope, quantity, ())


def _equal(left: DiagnosticCandidateAst, right: DiagnosticCandidateAst) -> DiagnosticCandidateAst:
    first, second = _commutative_children(left, right)
    return ("equal_exact", first, second)


def _less_equal(
    left: DiagnosticCandidateAst, right: DiagnosticCandidateAst
) -> DiagnosticCandidateAst:
    return ("less_equal", left, right)


def _and(left: DiagnosticCandidateAst, right: DiagnosticCandidateAst) -> DiagnosticCandidateAst:
    first, second = _commutative_children(left, right)
    return ("top_level_AND", first, second)


def _constant_atoms() -> tuple[DiagnosticCandidateAst, ...]:
    constants = tuple(_constant(item) for item in RATIONAL_PARAMETER_GRID)
    equal = tuple(
        _equal(left, right)
        for left, right in combinations_with_replacement(constants, 2)
    )
    ordered = tuple(
        _less_equal(left, right) for left, right in product(constants, repeat=2)
    )
    atoms = equal + ordered
    if len(atoms) != 77:
        raise AssertionError("shrink-1 constant atom count drift")
    return atoms


def _rational_aggregate_leaves() -> tuple[DiagnosticCandidateAst, ...]:
    leaves = tuple(
        _aggregate(map_name, scope, quantity)
        for map_name, scope, quantity in product(
            RATIONAL_ACTIVE_AGGREGATE_NAMES,
            SCOPE_IDS,
            QUANTITY_IDS,
        )
    )
    if len(leaves) != 16:
        raise AssertionError("shrink-1 rational aggregate leaf count drift")
    return leaves


def _mixed_atoms() -> tuple[DiagnosticCandidateAst, ...]:
    constants = tuple(_constant(item) for item in RATIONAL_PARAMETER_GRID)
    aggregates = _rational_aggregate_leaves()
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
    atoms = equal + ordered
    if len(atoms) != 336:
        raise AssertionError("shrink-1 mixed atom count drift")
    return atoms


def iter_shrink1_capacity_candidate_asts() -> Iterator[DiagnosticCandidateAst]:
    """Yield the preregistered child-DSL subset in deterministic source order."""

    for constant_atom, mixed_atom in product(_constant_atoms(), _mixed_atoms()):
        yield _and(constant_atom, mixed_atom)


__all__ = [
    "DiagnosticCandidateAst",
    "EXPECTED_SHRINK1_SOURCE_COUNT",
    "SHRINK1_CAPACITY_GENERATOR_RULE",
    "SHRINK1_CAPACITY_GENERATOR_SCHEMA",
    "iter_shrink1_capacity_candidate_asts",
]
