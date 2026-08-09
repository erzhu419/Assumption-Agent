"""Frozen target-free depth-four challenge lattice for shrink step 6.

The lattice is a bounded qualification set, not the full depth-four grammar and
not a closure execution.  It combines 175 inherited atom controls with 1,266
parent-legal nested unary/binary sources.  Parent canonicalization determines
which challenge rows normalize to depth at most three and which remain exact
depth-four, six-node child rejections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterator

from .phase3_m3_dsl_core_v1 import (
    QUANTITY_IDS,
    RATIONAL_PARAMETER_GRID,
    SCOPE_IDS,
)
from .phase3_m3_shrink6_core_v1 import ACTIVE_RATIONAL_PARAMETER_IDS
from .phase3_shrink5_capacity_v1 import (
    DiagnosticCandidateAst,
    iter_shrink5_capacity_candidate_asts,
    shrink5_rational_aggregate_leaves_v1,
)


SHRINK6_CAPACITY_GENERATOR_SCHEMA: Final = (
    "hegel-phase3-shrink6-depth4-challenge-capacity-generator/1"
)
SHRINK6_CAPACITY_GENERATOR_RULE: Final = (
    "family order A,B_abs,B_sign; operand outer, R inner, direction 0 then 1; "
    "R is active constants -1,0,1 followed by the exact inherited 16 rational "
    "aggregate leaves in map/scope/quantity order; A-U1 is bit_to_scalar(bit_at "
    "0..7), int_to_scalar(set_size), int_to_scalar(count_nonzero) in "
    "scope/quantity order, then absolute of the inherited rational aggregates; "
    "B-Q is absolute of the first 17 A-U1 non-rational-aggregate forms; rows "
    "with aggregate-bearing operand and aggregate R are excluded; no source "
    "deduplication; this is FROZEN_DEPTH4_CHALLENGE_LATTICE_ONLY_NOT_COMPLETE"
)
SUBSET_STATUS: Final = "FROZEN_DEPTH4_CHALLENGE_LATTICE_ONLY_NOT_COMPLETE"

EXPECTED_SHRINK6_INHERITED_SURVIVOR_SOURCE_COUNT: Final = 175
EXPECTED_SHRINK6_CHALLENGE_SOURCE_COUNT: Final = 1_266
EXPECTED_SHRINK6_CHALLENGE_PARENT_ACCEPTED_SOURCE_COUNT: Final = 1_266
EXPECTED_SHRINK6_PARENT_ONLY_SOURCE_COUNT: Final = 1_199
EXPECTED_SHRINK6_PARENT_ONLY_UNIQUE_COUNT: Final = 1_199
EXPECTED_SHRINK6_NORMALIZED_SURVIVOR_SOURCE_COUNT: Final = 67
EXPECTED_SHRINK6_NORMALIZED_SURVIVOR_UNIQUE_COUNT: Final = 50
EXPECTED_SHRINK6_FULL_SURVIVOR_SOURCE_COUNT: Final = 242
EXPECTED_SHRINK6_FULL_SURVIVOR_UNIQUE_COUNT: Final = 225
EXPECTED_SHRINK6_PARENT_CANONICAL_UNIQUE_COUNT: Final = 1_249
EXPECTED_SHRINK6_PARENT_ONLY_FAMILY_COUNTS: Final = {
    "A": 453,
    "B_abs": 373,
    "B_sign": 373,
}


@dataclass(frozen=True, slots=True)
class Depth4ChallengeSourceV1:
    family: str
    source_ast: DiagnosticCandidateAst


@dataclass(frozen=True, slots=True)
class _OperandV1:
    source_ast: DiagnosticCandidateAst
    aggregate_bearing: bool


def _constant(parameter_id: int) -> DiagnosticCandidateAst:
    value = RATIONAL_PARAMETER_GRID[parameter_id]
    return ("scalar_const", value.numerator, value.denominator)


def _aggregate(
    map_name: str, scope: str, quantity: str
) -> DiagnosticCandidateAst:
    return ("aggregate", map_name, scope, quantity, ())


def shrink6_r_pool_v1() -> tuple[_OperandV1, ...]:
    constants = tuple(
        _OperandV1(_constant(parameter_id), False)
        for parameter_id in ACTIVE_RATIONAL_PARAMETER_IDS
    )
    aggregates = tuple(
        _OperandV1(source, True)
        for source in shrink5_rational_aggregate_leaves_v1()
    )
    result = constants + aggregates
    if len(result) != 19:
        raise AssertionError("shrink-6 R pool count drift")
    return result


def _count_nonzero_operands() -> tuple[_OperandV1, ...]:
    return tuple(
        _OperandV1(
            ("int_to_scalar", _aggregate("count_nonzero_v1", scope, quantity)),
            True,
        )
        for scope in SCOPE_IDS
        for quantity in QUANTITY_IDS
    )


def shrink6_a_u1_pool_v1() -> tuple[_OperandV1, ...]:
    bit_scalars = tuple(
        _OperandV1(("bit_to_scalar", ("bit_at", slot)), False)
        for slot in range(8)
    )
    set_size = (_OperandV1(("int_to_scalar", ("set_size",)), False),)
    rational_absolute = tuple(
        _OperandV1(("absolute", aggregate.source_ast), True)
        for aggregate in shrink6_r_pool_v1()[3:]
    )
    result = bit_scalars + set_size + _count_nonzero_operands() + rational_absolute
    if len(result) != 33:
        raise AssertionError("shrink-6 A-U1 pool count drift")
    return result


def shrink6_b_q_pool_v1() -> tuple[_OperandV1, ...]:
    base = shrink6_a_u1_pool_v1()[:17]
    result = tuple(
        _OperandV1(("absolute", operand.source_ast), operand.aggregate_bearing)
        for operand in base
    )
    if len(result) != 17:
        raise AssertionError("shrink-6 B-Q pool count drift")
    return result


def _challenge_source(
    family: str,
    left: DiagnosticCandidateAst,
    right: DiagnosticCandidateAst,
) -> DiagnosticCandidateAst:
    difference = ("difference", left, right)
    if family == "A":
        return ("sign", ("absolute", difference))
    if family == "B_abs":
        return ("absolute", difference)
    if family == "B_sign":
        return ("sign", difference)
    raise AssertionError("unknown shrink-6 challenge family")


def iter_shrink6_depth4_challenge_sources_v1(
) -> Iterator[Depth4ChallengeSourceV1]:
    families = (
        ("A", shrink6_a_u1_pool_v1()),
        ("B_abs", shrink6_b_q_pool_v1()),
        ("B_sign", shrink6_b_q_pool_v1()),
    )
    r_pool = shrink6_r_pool_v1()
    count = 0
    for family, operands in families:
        for operand in operands:
            for rational in r_pool:
                if operand.aggregate_bearing and rational.aggregate_bearing:
                    continue
                for direction in (0, 1):
                    left, right = (
                        (operand.source_ast, rational.source_ast)
                        if direction == 0
                        else (rational.source_ast, operand.source_ast)
                    )
                    count += 1
                    yield Depth4ChallengeSourceV1(
                        family,
                        _challenge_source(family, left, right),
                    )
    if count != EXPECTED_SHRINK6_CHALLENGE_SOURCE_COUNT:
        raise AssertionError("shrink-6 challenge lattice count drift")


def iter_shrink6_inherited_survivor_candidate_asts(
) -> Iterator[DiagnosticCandidateAst]:
    yield from iter_shrink5_capacity_candidate_asts()


__all__ = [
    "Depth4ChallengeSourceV1",
    "DiagnosticCandidateAst",
    "EXPECTED_SHRINK6_CHALLENGE_PARENT_ACCEPTED_SOURCE_COUNT",
    "EXPECTED_SHRINK6_CHALLENGE_SOURCE_COUNT",
    "EXPECTED_SHRINK6_FULL_SURVIVOR_SOURCE_COUNT",
    "EXPECTED_SHRINK6_FULL_SURVIVOR_UNIQUE_COUNT",
    "EXPECTED_SHRINK6_INHERITED_SURVIVOR_SOURCE_COUNT",
    "EXPECTED_SHRINK6_NORMALIZED_SURVIVOR_SOURCE_COUNT",
    "EXPECTED_SHRINK6_NORMALIZED_SURVIVOR_UNIQUE_COUNT",
    "EXPECTED_SHRINK6_PARENT_CANONICAL_UNIQUE_COUNT",
    "EXPECTED_SHRINK6_PARENT_ONLY_FAMILY_COUNTS",
    "EXPECTED_SHRINK6_PARENT_ONLY_SOURCE_COUNT",
    "EXPECTED_SHRINK6_PARENT_ONLY_UNIQUE_COUNT",
    "SHRINK6_CAPACITY_GENERATOR_RULE",
    "SHRINK6_CAPACITY_GENERATOR_SCHEMA",
    "SUBSET_STATUS",
    "iter_shrink6_depth4_challenge_sources_v1",
    "iter_shrink6_inherited_survivor_candidate_asts",
    "shrink6_a_u1_pool_v1",
    "shrink6_b_q_pool_v1",
    "shrink6_r_pool_v1",
]
