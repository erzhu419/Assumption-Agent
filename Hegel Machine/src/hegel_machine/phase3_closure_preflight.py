"""Constructive capacity preflight for the frozen Phase-3 old DSL.

This module does not enumerate the complete extensional closure.  Instead it
constructs a deliberately small, type-correct subset of diagnostic candidate
ASTs.  The subset exceeds the frozen 50,000-program limit.  The v1.0.2 strict
canonical-AST/CBOR acceptance rules have now passed the independent
Python/Rust golden-vector gate, and both implementations accepted the same
64,680 unique canonical ASTs.  The verified 50,001st witness discharges the
former conditional capacity result and sets the bounded old-DSL execution
status to ``DSL_TOO_LARGE``.

The proof excludes ``greater_equal`` and every nested arithmetic operator, so
it remains a conservative subset rather than an estimate of the full grammar.
The verified Rust replay is bound below, but there is no sealed full-closure
archive or bounded frozen-closure certificate here.  The cross-language set
commitment is diagnostic and is not a formal RFC6962 root.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
from typing import Final

from .hashing import canonical_json, stable_hash
from .phase3_capacity_witness_v1 import (
    DiagnosticCandidateAst,
    canonical_commutative_children,
    iter_capacity_witness_candidate_asts,
)
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
FREEZE_VERSION: Final = "hegel-freeze-p2b-p3-v1.0.2"
CANONICAL_CBOR_PROFILE_ID: Final = "hegel-cbor-det-v1"
CANONICAL_AST_SCHEMA_ID: Final = "hegel-canonical-ast-v1"
CONDITIONAL_CAPACITY_STATUS: Final = (
    "CONDITIONAL_CAPACITY_LOWER_BOUND_EXCEEDS_BUDGET"
)
DSL_TOO_LARGE_STATUS: Final = "DSL_TOO_LARGE"
DUAL_STRICT_GATE_STATUS: Final = "VERIFIED"
# Frozen evidence bindings copied from the separately generated, checked-in
# dual strict gate and capacity replay artifacts.  This module does not rerun
# Rust or reinterpret their diagnostic commitments as formal archive roots.
DUAL_STRICT_GATE_REPORT_ID: Final = (
    "phase3_dual_strict_gate_"
    "06eae23f68536e3f7e80badb46a5b15e0665072f65477608a3f688e54adefad6"
)
DUAL_STRICT_CAPACITY_REPLAY_REPORT_ID: Final = (
    "phase3_dual_strict_capacity_replay_"
    "f75214e75f5fc3812d7375463ba72c347c9c08bc7bae3b68c87a63b484c4e414"
)
STRICT_CAPACITY_SET_COMMITMENT: Final = (
    "sha256:c1a02a66a8d6d8f75204cb3daf03ab0b01c2b3b8e486d0ab3d481ee3be43c930"
)
FIRST_OUT_OF_BUDGET_AST_HASH: Final = (
    "sha256:7c7f786c2cc57d31506b3c61d162d175c7f69a2878a089c72c9d053694cba948"
)
FIRST_OUT_OF_BUDGET_CBOR_HEX: Final = (
    "820182048284020383000002830000048402038600030000008083000000"
)
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
PREFLIGHT_SOURCE_PATHS: Final = (
    Path(__file__),
    PROJECT_ROOT / "src" / "hegel_machine" / "phase3_capacity_witness_v1.py",
    PROJECT_ROOT / "src" / "hegel_machine" / "phase3_dsl_v1.py",
    PROJECT_ROOT / "src" / "hegel_machine" / "hashing.py",
)


def _preflight_source_root() -> str:
    digest = hashlib.sha256()
    digest.update(b"HEGEL/PREFLIGHT_SOURCE_SET/V1\x00")
    for path in sorted(
        PREFLIGHT_SOURCE_PATHS,
        key=lambda item: item.relative_to(PROJECT_ROOT).as_posix(),
    ):
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()

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
        if len(ast) != 5:
            raise AssertionError("aggregate diagnostic arity drift")
        map_id, scope_id, quantity_id, extensions = ast[1], ast[2], ast[3], ast[4]
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
            ordered = canonical_commutative_children(  # type: ignore[arg-type]
                ast[1], ast[2]
            )
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
        ordered = canonical_commutative_children(  # type: ignore[arg-type]
            ast[1], ast[2]
        )
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
    freeze_version: str = FREEZE_VERSION
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
        if self.freeze_version != FREEZE_VERSION:
            raise ValueError("capacity proof freeze version drift")
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
        return DSL_TOO_LARGE_STATUS

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
        "freeze_version": proof.freeze_version,
        "implementation_id": (
            "phase3_closure_preflight_source_root_sha256_"
            + _preflight_source_root()
        ),
        "dsl_version": proof.dsl_version,
        "dsl_spec_id": proof.dsl_spec_id,
        "proof_id": proof.content_id,
        "status": proof.capacity_status,
        "executed_closure_status": DSL_TOO_LARGE_STATUS,
        "status_basis": "dual_verified_strict_capacity_replay",
        "capacity_condition_discharged": True,
        "canonical_program_budget": proof.canonical_program_budget,
        "constructive_candidate_ast_count": proof.witness_candidate_ast_count,
        "first_out_of_budget_ordinal": proof.first_out_of_budget_ordinal,
        "strict_acceptance_specification_complete": True,
        "strict_acceptance_implementation_verified": True,
        "strict_rewrite_application_pending": False,
        "dual_strict_gate_status": DUAL_STRICT_GATE_STATUS,
        "dual_strict_gate_report_id": DUAL_STRICT_GATE_REPORT_ID,
        "dual_strict_capacity_replay_report_id": (
            DUAL_STRICT_CAPACITY_REPLAY_REPORT_ID
        ),
        "canonical_cbor_profile_id": CANONICAL_CBOR_PROFILE_ID,
        "canonical_ast_schema_id": CANONICAL_AST_SCHEMA_ID,
        "formal_root_generation_allowed": False,
        "formal_roots": None,
        "dsl_too_large_claim_allowed": True,
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
        "strict_capacity_replay": {
            "source_candidate_count": proof.witness_candidate_ast_count,
            "type_rejected_count": 0,
            "limit_rejected_count": 0,
            "other_rejected_count": 0,
            "rewrite_collapsed_count": 0,
            "accepted_strict_canonical_count": proof.witness_candidate_ast_count,
            "first_accepted_out_of_budget_ordinal": (
                proof.first_out_of_budget_ordinal
            ),
            "first_accepted_out_of_budget_ast_hash": (
                FIRST_OUT_OF_BUDGET_AST_HASH
            ),
            "first_accepted_out_of_budget_cbor_hex": (
                FIRST_OUT_OF_BUDGET_CBOR_HEX
            ),
            "python_accepted_set_commitment": STRICT_CAPACITY_SET_COMMITMENT,
            "rust_accepted_set_commitment": STRICT_CAPACITY_SET_COMMITMENT,
            "accepted_set_commitment_is_formal_root": False,
            "dual_replay_equal": True,
        },
        "complete_closure_enumerated": False,
        "extensional_quotient_computed": False,
        "formal_canonicalizer_implemented": True,
        "python_complete_enumerator_implemented": False,
        "rust_complete_enumerator_implemented": False,
        "outside_certificate_allowed": False,
        "outside_frozen_closure_certificate_issued": False,
        "unbounded_outside_language_claim_issued": False,
        "target_synthesis_allowed": False,
        "hidden_sink_formal_verdict_allowed": False,
        "mdl_certificate_allowed": False,
        "active_promotion_allowed": False,
        "phase2b_formal_exit": False,
        "required_next_action": {
            "freeze_strict_canonical_ast_schema_and_acceptance_rules": False,
            "implement_python_strict_acceptance": False,
            "implement_rust_strict_acceptance": False,
            "verify_cross_language_golden_vectors": False,
            "replay_64680_with_strict_canonical_cbor": False,
            "action": "PUBLISH_SHRUNK_OLD_DSL_VERSION_USING_FROZEN_STEP_1",
            "frozen_shrink_step": (
                OLD_DSL_V1.shrink_order[0].operation
            ),
            "regenerate_target_commitments_after_version_change": True,
        },
        "claim_boundary": (
            "Independent Python and Rust strict replay accepted the same 64,680 "
            "unique canonical ASTs and the same 50,001st witness, so the frozen "
            "50,000 syntactic-program budget is exceeded and bounded old-DSL "
            "status is DSL_TOO_LARGE. This is not COMPLETE, a full closure "
            "cardinality, an extensional target verdict, a formal RFC6962 root, "
            "or an outside certificate."
        ),
    }
    payload["report_id"] = stable_hash(
        payload,
        prefix="phase3_closure_capacity_preflight_report_",
    )
    return payload
