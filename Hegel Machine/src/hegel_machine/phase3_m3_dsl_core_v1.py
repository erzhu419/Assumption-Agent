"""Target-free old-DSL constants required by the M3 canonical enumerator.

This module is intentionally a closed projection of the frozen v1.0.0 DSL
surface.  It contains no benchmark universe, truth function, split contract,
target role, seed, or evaluator.  The isolated enumerator snapshot mounts this
file instead of :mod:`phase3_dsl_v1`, whose lower half contains the odd and
hidden-sink benchmark definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Final


@dataclass(frozen=True, order=True, slots=True)
class RationalAtom:
    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise TypeError("rational numerator and denominator must be integers")
        if self.denominator <= 0:
            raise ValueError("rational denominator must be positive")
        reduced = Fraction(self.numerator, self.denominator)
        if (reduced.numerator, reduced.denominator) != (
            self.numerator,
            self.denominator,
        ):
            raise ValueError("RationalAtom must already be reduced")


RATIONAL_PARAMETER_GRID: Final = tuple(
    RationalAtom(value.numerator, value.denominator)
    for value in (
        Fraction(-2),
        Fraction(-1),
        Fraction(-1, 2),
        Fraction(0),
        Fraction(1, 2),
        Fraction(1),
        Fraction(2),
    )
)

AGGREGATE_MAP_IDS: Final = (
    "sum_v1",
    "count_nonzero_v1",
    "mean_v1",
    "min_v1",
    "max_v1",
    "signed_balance_v1",
)
SCOPE_IDS: Final = (
    "scope_all_observed_v1",
    "scope_primary_only_v1",
    "scope_boundary_only_v1",
    "control_volume_all_observed_v1",
)
QUANTITY_IDS: Final = ("q0", "q1")
CONTEXT_IDS: Final = ("c0", "c1", "c2", "c3")
TASK_IDS: Final = ("t0", "t1")


@dataclass(frozen=True, slots=True)
class StructuralLimits:
    max_total_ast_depth: int = 4
    max_total_node_count: int = 7
    max_top_level_clauses: int = 3
    max_distinct_bit_slots: int = 4
    max_aggregate_leaves: int = 1
    max_scope_clauses: int = 2
    max_old_law_composition_depth: int = 2
    max_fitted_scalar_parameters: int = 3
    leaf_depth: int = 0
    operator_depth_rule: str = "1 + max(child_depth)"


STRUCTURAL_LIMITS: Final = StructuralLimits()


__all__ = [
    "AGGREGATE_MAP_IDS",
    "CONTEXT_IDS",
    "QUANTITY_IDS",
    "RATIONAL_PARAMETER_GRID",
    "SCOPE_IDS",
    "STRUCTURAL_LIMITS",
    "TASK_IDS",
]
