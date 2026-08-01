"""Constructive capacity preflight for the frozen Phase-3 old DSL.

This module does not enumerate the complete extensional closure.  Instead it
constructs a deliberately small, type-correct subset of diagnostic candidate
ASTs.  The subset exceeds the frozen 50,000-program limit, but the result is
conditional until the strict canonical-AST schema and canonical-CBOR
canonicalizer accept the same structures as canonical programs.  It cannot set the
executed closure status to ``DSL_TOO_LARGE`` by itself.

The proof excludes ``greater_equal`` and every nested arithmetic operator, so
it is a conservative lower bound rather than an estimate of the full grammar.
It also keeps the formal trust boundary explicit: there is no Rust replay,
sealed archive, or bounded frozen-closure certificate here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations_with_replacement, product
from pathlib import Path
import hashlib
from typing import Final, Iterator

from .hashing import canonical_json, stable_hash
from .phase3_dsl_v1 import (
    AGGREGATE_CATALOG,
    BINARY_OPERATORS,
    BOOLEAN_COMPOSITION,
    CLOSURE_BUDGET,
    OLD_DSL_V1,
    QUANTITY_IDS,
    RATIONAL_PARAMETER_GRID,
    SCOPE_IDS,
    STRUCTURAL_LIMITS,
    RationalAtom,
)


PREFLIGHT_SCHEMA_VERSION: Final = "hegel-phase3-closure-capacity-preflight/1.0.0"
CONDITIONAL_CAPACITY_STATUS: Final = (
    "CONDITIONAL_CAPACITY_LOWER_BOUND_EXCEEDS_BUDGET"
)

DiagnosticCandidateAst = tuple[object, ...]


def _diagnostic_hash(ast: DiagnosticCandidateAst) -> str:
    return stable_hash(ast)


def _commutative_children(
    left: DiagnosticCandidateAst,
    right: DiagnosticCandidateAst,
) -> tuple[DiagnosticCandidateAst, DiagnosticCandidateAst]:
    """Apply the frozen child-hash order with a deterministic tie break."""

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
        "scope_extensions",
        (),
    )


def _equal_exact(
    left: DiagnosticCandidateAst,
    right: DiagnosticCandidateAst,
) -> DiagnosticCandidateAst:
    first, second = _commutative_children(left, right)
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
    first, second = _commutative_children(left, right)
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
    """Yield the conservative diagnostic AST subset used by the preflight."""

    constant_atoms = _constant_only_atoms()
    aggregate_atoms = _one_aggregate_atoms()
    for constant_atom, aggregate_atom in product(
        constant_atoms,
        aggregate_atoms,
    ):
        yield _top_level_and(constant_atom, aggregate_atom)


@dataclass(frozen=True, slots=True)
class WitnessAstStats:
    output_sort: str
    depth: int
    node_count: int
    scalar_parameter_occurrences: int
    aggregate_leaf_count: int
    distinct_bit_slots: frozenset[int]
    scope_clause_count: int
    top_level_clause_count: int
    old_law_composition_depth: int


def _expression_spec(expression_id: str):
    specs = BINARY_OPERATORS + BOOLEAN_COMPOSITION
    matches = tuple(spec for spec in specs if spec.expression_id == expression_id)
    if len(matches) != 1:
        raise AssertionError(f"missing or duplicated expression spec: {expression_id}")
    return matches[0]


def _diagnostic_ast_stats(ast: DiagnosticCandidateAst) -> WitnessAstStats:
    """Recompute type and structural limits from one emitted diagnostic AST."""

    if not isinstance(ast, tuple) or not ast or not isinstance(ast[0], str):
        raise AssertionError("diagnostic AST nodes must be tagged tuples")
    tag = ast[0]
    if tag == "scalar_const":
        if len(ast) != 3:
            raise AssertionError("scalar_const diagnostic arity drift")
        parameter = RationalAtom(ast[1], ast[2])  # type: ignore[arg-type]
        if parameter not in RATIONAL_PARAMETER_GRID:
            raise AssertionError("scalar_const parameter left the frozen grid")
        return WitnessAstStats(
            "RationalValue", 0, 1, 1, 0, frozenset(), 0, 0, 0
        )
    if tag == "aggregate":
        if len(ast) != 6 or ast[4] != "scope_extensions":
            raise AssertionError("aggregate diagnostic arity drift")
        map_id, scope_id, quantity_id, extensions = ast[1], ast[2], ast[3], ast[5]
        rational_maps = {
            spec.map_id
            for spec in AGGREGATE_CATALOG
            if spec.output_sort == "RationalValue"
        }
        if map_id not in rational_maps:
            raise AssertionError("capacity witness aggregate is not RationalValue")
        if scope_id not in SCOPE_IDS or quantity_id not in QUANTITY_IDS:
            raise AssertionError("capacity witness aggregate registry drift")
        if not isinstance(extensions, tuple):
            raise AssertionError("scope extensions must be an immutable tuple")
        if len(extensions) > STRUCTURAL_LIMITS.max_scope_clauses:
            raise AssertionError("capacity witness exceeds scope-clause limit")
        return WitnessAstStats(
            "RationalValue", 0, 1, 0, 1, frozenset(), len(extensions), 0, 0
        )
    if tag in {"equal_exact", "less_equal"}:
        if len(ast) != 3:
            raise AssertionError(f"{tag} diagnostic arity drift")
        spec = _expression_spec(tag)
        if spec.input_sorts != ("RationalValue", "RationalValue"):
            raise AssertionError(f"{tag} frozen input typing drift")
        if spec.output_sorts != ("Bool",) or spec.accepted_arities != (2,):
            raise AssertionError(f"{tag} frozen output/arity drift")
        children = tuple(_diagnostic_ast_stats(child) for child in ast[1:])
        if any(child.output_sort != "RationalValue" for child in children):
            raise AssertionError(f"{tag} received a non-rational child")
        if tag == "equal_exact":
            ordered = _commutative_children(ast[1], ast[2])  # type: ignore[arg-type]
            if ast[1:] != ordered:
                raise AssertionError("equal_exact diagnostic child order drift")
        return WitnessAstStats(
            output_sort="Bool",
            depth=1 + max(child.depth for child in children),
            node_count=1 + sum(child.node_count for child in children),
            scalar_parameter_occurrences=sum(
                child.scalar_parameter_occurrences for child in children
            ),
            aggregate_leaf_count=sum(
                child.aggregate_leaf_count for child in children
            ),
            distinct_bit_slots=frozenset().union(
                *(child.distinct_bit_slots for child in children)
            ),
            scope_clause_count=sum(child.scope_clause_count for child in children),
            top_level_clause_count=0,
            old_law_composition_depth=max(
                child.old_law_composition_depth for child in children
            ),
        )
    if tag == "top_level_AND":
        spec = _expression_spec(tag)
        clause_count = len(ast) - 1
        if clause_count not in spec.accepted_arities:
            raise AssertionError("top-level AND diagnostic arity drift")
        if spec.output_sorts != ("Bool",):
            raise AssertionError("top-level AND output typing drift")
        children = tuple(_diagnostic_ast_stats(child) for child in ast[1:])
        if any(child.output_sort != "Bool" for child in children):
            raise AssertionError("top-level AND received a non-boolean child")
        ordered = _commutative_children(ast[1], ast[2])  # type: ignore[arg-type]
        if ast[1:] != ordered:
            raise AssertionError("top-level AND diagnostic child order drift")
        return WitnessAstStats(
            output_sort="Bool",
            depth=1 + max(child.depth for child in children),
            node_count=1 + sum(child.node_count for child in children),
            scalar_parameter_occurrences=sum(
                child.scalar_parameter_occurrences for child in children
            ),
            aggregate_leaf_count=sum(
                child.aggregate_leaf_count for child in children
            ),
            distinct_bit_slots=frozenset().union(
                *(child.distinct_bit_slots for child in children)
            ),
            scope_clause_count=sum(child.scope_clause_count for child in children),
            top_level_clause_count=clause_count,
            old_law_composition_depth=max(
                child.old_law_composition_depth for child in children
            ),
        )
    raise AssertionError(f"unsupported capacity-witness AST tag: {tag}")


@dataclass(frozen=True, slots=True)
class ConstructiveCandidateAstCapacityProof:
    """Exact combinatorial lower bound for typed, limit-conforming candidate ASTs."""

    schema_version: str = PREFLIGHT_SCHEMA_VERSION
    dsl_version: str = OLD_DSL_V1.dsl_version
    dsl_spec_id: str = OLD_DSL_V1.content_id
    scalar_constant_leaf_count: int = 7
    rational_aggregate_leaf_count: int = 40
    constant_equal_atom_count: int = 28
    constant_less_equal_atom_count: int = 49
    mixed_equal_atom_count: int = 280
    mixed_less_equal_atom_count: int = 560
    witness_candidate_ast_count: int = 64_680
    canonical_program_budget: int = 50_000
    witness_ast_depth: int = 2
    witness_node_count: int = 7
    witness_top_level_clause_count: int = 2
    witness_max_scalar_parameter_occurrences: int = 3
    witness_aggregate_leaf_count: int = 1
    witness_distinct_bit_slot_count: int = 0
    proof_operators: tuple[str, ...] = (
        "equal_exact",
        "less_equal",
        "top_level_AND",
    )
    deliberately_excluded_surface: tuple[str, ...] = (
        "greater_equal",
        "all unary operators",
        "add and difference",
        "approx_equal",
        "same_sign and opposite_sign",
        "context_flag and task_flag",
        "bit_at and set_size",
        "top-level AND arities 1 and 3",
        "nonzero scope extensions",
    )

    def __post_init__(self) -> None:
        if self.schema_version != PREFLIGHT_SCHEMA_VERSION:
            raise ValueError("unknown closure-capacity preflight schema")
        if self.dsl_version != OLD_DSL_V1.dsl_version:
            raise ValueError("capacity proof is bound to the frozen DSL version")
        if self.dsl_spec_id != OLD_DSL_V1.content_id:
            raise ValueError("capacity proof is bound to the frozen DSL content id")

        rational_map_count = sum(
            spec.output_sort == "RationalValue" for spec in AGGREGATE_CATALOG
        )
        expected_aggregate_leaves = (
            rational_map_count * len(SCOPE_IDS) * len(QUANTITY_IDS)
        )
        if self.scalar_constant_leaf_count != len(RATIONAL_PARAMETER_GRID):
            raise ValueError("scalar-constant population does not match the DSL")
        if self.rational_aggregate_leaf_count != expected_aggregate_leaves:
            raise ValueError("aggregate-leaf population does not match the DSL")

        constant_count = self.scalar_constant_leaf_count
        aggregate_count = self.rational_aggregate_leaf_count
        expected_constant_equal = constant_count * (constant_count + 1) // 2
        expected_constant_ordered = constant_count**2
        expected_mixed_equal = constant_count * aggregate_count
        expected_mixed_ordered = 2 * constant_count * aggregate_count
        expected_candidate_ast_count = (
            expected_constant_equal + expected_constant_ordered
        ) * (expected_mixed_equal + expected_mixed_ordered)
        expected = (
            expected_constant_equal,
            expected_constant_ordered,
            expected_mixed_equal,
            expected_mixed_ordered,
            expected_candidate_ast_count,
        )
        actual = (
            self.constant_equal_atom_count,
            self.constant_less_equal_atom_count,
            self.mixed_equal_atom_count,
            self.mixed_less_equal_atom_count,
            self.witness_candidate_ast_count,
        )
        if actual != expected:
            raise ValueError("constructive capacity counts are inconsistent")
        if self.canonical_program_budget != CLOSURE_BUDGET.max_canonical_program_count:
            raise ValueError("capacity proof changed the frozen program budget")
        if self.witness_candidate_ast_count <= self.canonical_program_budget:
            raise ValueError("capacity witness must exceed the frozen budget")

        limits = STRUCTURAL_LIMITS
        if self.witness_ast_depth > limits.max_total_ast_depth:
            raise ValueError("capacity witness exceeds the depth limit")
        if self.witness_node_count > limits.max_total_node_count:
            raise ValueError("capacity witness exceeds the node limit")
        if self.witness_top_level_clause_count > limits.max_top_level_clauses:
            raise ValueError("capacity witness exceeds the clause limit")
        if (
            self.witness_max_scalar_parameter_occurrences
            > limits.max_fitted_scalar_parameters
        ):
            raise ValueError("capacity witness exceeds the scalar-parameter limit")
        if self.witness_aggregate_leaf_count > limits.max_aggregate_leaves:
            raise ValueError("capacity witness exceeds the aggregate limit")
        if self.witness_distinct_bit_slot_count > limits.max_distinct_bit_slots:
            raise ValueError("capacity witness exceeds the bit-slot limit")

    @property
    def constant_only_atom_count(self) -> int:
        return self.constant_equal_atom_count + self.constant_less_equal_atom_count

    @property
    def one_aggregate_atom_count(self) -> int:
        return self.mixed_equal_atom_count + self.mixed_less_equal_atom_count

    @property
    def first_out_of_budget_ordinal(self) -> int:
        return self.canonical_program_budget + 1

    @property
    def capacity_status(self) -> str:
        return CONDITIONAL_CAPACITY_STATUS

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="closure_capacity_proof_")


CAPACITY_PROOF: Final = ConstructiveCandidateAstCapacityProof()


@lru_cache(maxsize=1)
def _replay_constructive_subset_frozen() -> tuple[tuple[str, object], ...]:
    """Materialize and independently deduplicate the Python witness subset.

    This is a local deterministic replay, not the required independent Rust
    implementation.  It provides executable evidence that the combinatorial
    formula did not accidentally count duplicate diagnostic AST encodings.
    """

    seen: set[str] = set()
    expected_stats = WitnessAstStats(
        output_sort="Bool",
        depth=CAPACITY_PROOF.witness_ast_depth,
        node_count=CAPACITY_PROOF.witness_node_count,
        scalar_parameter_occurrences=(
            CAPACITY_PROOF.witness_max_scalar_parameter_occurrences
        ),
        aggregate_leaf_count=CAPACITY_PROOF.witness_aggregate_leaf_count,
        distinct_bit_slots=frozenset(),
        scope_clause_count=0,
        top_level_clause_count=CAPACITY_PROOF.witness_top_level_clause_count,
        old_law_composition_depth=0,
    )
    for candidate_ast in iter_capacity_witness_candidate_asts():
        if _diagnostic_ast_stats(candidate_ast) != expected_stats:
            raise AssertionError("capacity witness AST constraints drifted")
        encoded = canonical_json(candidate_ast)
        if encoded in seen:
            raise AssertionError("constructive subset emitted a duplicate AST")
        seen.add(encoded)
    observed_count = len(seen)
    if observed_count != CAPACITY_PROOF.witness_candidate_ast_count:
        raise AssertionError("constructive subset count differs from frozen proof")
    return tuple(
        {
            "diagnostic_subset_python_materialization_complete": True,
            "observed_unique_candidate_ast_count": observed_count,
            "combinatorial_candidate_ast_count": (
                CAPACITY_PROOF.witness_candidate_ast_count
            ),
            "candidate_ast_count_agreement": True,
            "typing_and_ast_limits_recomputed_for_every_candidate_ast": True,
            "diagnostic_commutative_order_recomputed_for_every_candidate_ast": True,
            "encoding": "hegel_machine.hashing.canonical_json (diagnostic only)",
            "formal_canonical_cbor_archive": False,
            "strict_canonicalizer_acceptance_verified": False,
            "rust_replay_complete": False,
        }.items()
    )


def replay_constructive_subset() -> dict[str, object]:
    """Return a fresh replay payload so callers cannot poison cached state."""

    return dict(_replay_constructive_subset_frozen())


def phase3_closure_capacity_preflight_report(
    *, replay_subset: bool = False,
) -> dict[str, object]:
    """Return the fail-closed old-DSL capacity verdict and trust boundary."""

    proof = CAPACITY_PROOF
    replay = replay_constructive_subset() if replay_subset else {
        "diagnostic_subset_python_materialization_complete": False,
        "observed_unique_candidate_ast_count": None,
        "combinatorial_candidate_ast_count": proof.witness_candidate_ast_count,
        "candidate_ast_count_agreement": None,
        "typing_and_ast_limits_recomputed_for_every_candidate_ast": False,
        "diagnostic_commutative_order_recomputed_for_every_candidate_ast": False,
        "encoding": "hegel_machine.hashing.canonical_json (diagnostic only)",
        "formal_canonical_cbor_archive": False,
        "strict_canonicalizer_acceptance_verified": False,
        "rust_replay_complete": False,
    }
    payload: dict[str, object] = {
        "artifact": "phase3_closure_capacity_preflight_v1",
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "implementation_id": (
            "phase3_closure_preflight_source_sha256_"
            + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ),
        "dsl_version": proof.dsl_version,
        "dsl_spec_id": proof.dsl_spec_id,
        "proof_id": proof.content_id,
        "status": proof.capacity_status,
        "executed_closure_status": "NOT_RUN",
        "status_basis": "conditional_candidate_ast_subset_lower_bound",
        "canonical_program_budget": proof.canonical_program_budget,
        "constructive_candidate_ast_count": proof.witness_candidate_ast_count,
        "first_out_of_budget_ordinal": proof.first_out_of_budget_ordinal,
        "proof_counts": {
            "scalar_constant_leaves": proof.scalar_constant_leaf_count,
            "rational_aggregate_leaves": proof.rational_aggregate_leaf_count,
            "constant_equal_atoms": proof.constant_equal_atom_count,
            "constant_less_equal_atoms": proof.constant_less_equal_atom_count,
            "constant_only_atoms": proof.constant_only_atom_count,
            "mixed_equal_atoms": proof.mixed_equal_atom_count,
            "mixed_less_equal_atoms": proof.mixed_less_equal_atom_count,
            "one_aggregate_atoms": proof.one_aggregate_atom_count,
            "candidate_and2_ast_pairs": proof.witness_candidate_ast_count,
        },
        "witness_constraints": {
            "ast_depth": proof.witness_ast_depth,
            "node_count": proof.witness_node_count,
            "top_level_clauses": proof.witness_top_level_clause_count,
            "maximum_scalar_parameter_occurrences": (
                proof.witness_max_scalar_parameter_occurrences
            ),
            "aggregate_leaves": proof.witness_aggregate_leaf_count,
            "distinct_bit_slots": proof.witness_distinct_bit_slot_count,
            "operators": list(proof.proof_operators),
            "deliberately_excluded_surface": list(
                proof.deliberately_excluded_surface
            ),
        },
        "diagnostic_python_subset_replay": replay,
        "complete_closure_enumerated": False,
        "extensional_quotient_computed": False,
        "formal_canonicalizer_implemented": False,
        "python_complete_enumerator_implemented": False,
        "rust_complete_enumerator_implemented": False,
        "outside_frozen_closure_certificate_issued": False,
        "unbounded_outside_language_claim_issued": False,
        "required_next_action": {
            "freeze_strict_canonical_ast_schema_and_acceptance_rules": True,
            "replay_with_formal_canonical_cbor": True,
            "publish_new_dsl_version_if_witnesses_are_accepted": True,
            "conditional_first_frozen_shrink_step": (
                OLD_DSL_V1.shrink_order[0].operation
            ),
            "regenerate_target_commitments_after_version_change": True,
        },
        "claim_boundary": (
            "The diagnostic representation contains 64,680 distinct, typed, "
            "limit-conforming candidate ASTs. If the still-unimplemented strict "
            "canonicalizer accepts them without additional algebraic reduction, "
            "the frozen 50,000 syntactic-program limit is exceeded. Executed "
            "closure status remains NOT_RUN; this is neither DSL_TOO_LARGE yet "
            "nor an extensional target comparison or outside certificate."
        ),
    }
    payload["report_id"] = stable_hash(
        payload,
        prefix="phase3_closure_capacity_preflight_report_",
    )
    return payload
